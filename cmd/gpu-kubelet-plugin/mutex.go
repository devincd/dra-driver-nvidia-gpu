/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// mutex 包提供了 GPU DRA 驱动中按 GPU PCIe Bus ID 粒度的细粒度互斥锁机制。
//
// 在 VFIO 直通场景下，对 GPU 执行 Configure（绑定 vfio-pci 驱动）或 Unconfigure（切回 nvidia 驱动）
// 操作时，需要确保同一物理 GPU 不会被并发的操作同时修改。如果缺少这种按 GPU 粒度的锁保护，
// 可能出现如下竞态条件：
//   - 线程 A 正在对 GPU-0 执行 vfio-pci 绑定，线程 B 同时对 GPU-0 执行 nvidia 驱动解绑
//   - 两个线程并发修改同一设备节点，导致驱动状态不一致甚至设备不可用
//
// 本文件定义的 PerGPUMutex 通过二级锁结构解决这个问题：
//   - 外层 sync.Mutex：保护 submutex map 本身的并发读写（防止 map 并发写入 panic）
//   - 内层 map[string]*sync.Mutex：每个 PCIe Bus ID 对应一个独立的互斥锁
//
// 这样，不同 GPU 的操作可以完全并行，只有操作同一 GPU 的请求才会互斥等待。
package main

import (
	"sync"
)

// PerGPUMutex 提供按 GPU PCIe Bus ID 粒度的细粒度锁。
//
// 设计思路：
//   - VFIO 直通场景下，Configure/Unconfigure 操作需要按 GPU 加锁，避免同一 GPU 被并发操作
//   - submutex 中的每个条目对应一个物理 GPU（key 为 PCIe Bus ID 字符串，如 "0000:3b:00.0"）
//   - submutex 按需懒初始化：首次请求某个 GPU 的锁时才创建，避免启动时为所有 GPU 预分配锁
//   - 外层嵌入的 sync.Mutex 专门保护 submutex map 的并发访问，不用于保护业务逻辑
//
// 使用方式：
//
//	lock := perGpuLock.Get("0000:3b:00.0")  // 获取该 GPU 的专属锁
//	lock.Lock()                              // 加锁，同一 GPU 的其他操作会阻塞
//	defer lock.Unlock()                      // 解锁
//	// ... 执行 GPU Configure/Unconfigure 操作 ...
type PerGPUMutex struct {
	sync.Mutex                         // 外层锁：仅保护 submutex map 的并发读写，不保护 GPU 操作本身
	submutex     map[string]*sync.Mutex // 内层锁映射：每个 PCIe Bus ID 对应一个独立的业务锁
}

// perGpuLock 是全局的 per-GPU 锁实例，在 init() 中初始化。
//
// 之所以使用全局变量而非在 driver 结构体中持有，是因为该锁需要跨多个组件使用：
//   - VFIO 设备管理器（VfioPciManager）在 Configure/Unconfigure 时使用
//   - 未来可能的其他需要按 GPU 粒度加锁的场景
//
// 全局单例模式也简化了传参，避免将锁引用在多层调用栈中传递。
var perGpuLock *PerGPUMutex

// init 在包加载时初始化全局 per-GPU 锁实例。
//
// 初始化时创建空的 submutex map（不预分配任何 GPU 的锁），后续通过 Get() 按需创建。
// 使用 init() 而非显式初始化函数确保了锁在任何业务代码执行前就已就绪。
func init() {
	perGpuLock = &PerGPUMutex{
		submutex: make(map[string]*sync.Mutex),
	}
}

// Get 返回指定 gpu（通常为 PCIe Bus ID，如 "0000:3b:00.0"）对应的独占锁。
//
// 工作流程：
//  1. 先加外层锁（保护 submutex map 的并发读写）
//  2. 检查该 GPU 的锁是否已存在于 map 中
//  3. 若不存在（首次访问），创建新的 sync.Mutex 并存入 map（懒初始化）
//  4. 返回该 GPU 对应的锁指针
//  5. defer 自动释放外层锁
//
// 返回的 *sync.Mutex 由调用方自行 Lock/Unlock，同一 GPU 的并发操作将在此锁上排队等待。
// 不同 GPU 的操作使用不同的锁，因此可以完全并行执行，不会互相阻塞。
//
// 参数：
//   - gpu: GPU 的 PCIe Bus ID 字符串，作为 map 的 key 和锁的标识符
//
// 返回值：
//   - *sync.Mutex: 该 GPU 对应的独占互斥锁，调用方需自行管理 Lock/Unlock
func (pgm *PerGPUMutex) Get(gpu string) *sync.Mutex {
	pgm.Lock()         // 加外层锁，保护 submutex map 的并发访问
	defer pgm.Unlock() // 确保方法返回时释放外层锁

	// 懒初始化：若该 GPU 的锁尚未创建，则新建并存入 map
	if pgm.submutex[gpu] == nil {
		pgm.submutex[gpu] = &sync.Mutex{}
	}

	// 返回该 GPU 对应的锁指针，调用方将在此锁上进行 Lock/Unlock 操作
	return pgm.submutex[gpu]
}
