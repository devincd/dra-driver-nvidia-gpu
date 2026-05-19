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

// device_health.go 实现了基于 NVML 事件机制的 GPU 设备健康监控功能。
//
// 本文件是 DRA 驱动中设备故障检测的核心模块，通过 NVML 事件订阅机制
// 实时监控 GPU 硬件健康状态，并将故障设备从可分配列表中移除。
//
// 架构设计：
//   - nvmlDeviceHealthMonitor：核心监控结构体，封装 NVML 事件循环和设备查找
//   - devicePlacementMap：三层嵌套 map，实现事件到设备的 O(1) 查找
//   - unhealthy channel：与 driver 通信的异步通知管道
//
// 数据流：
//   NVML 硬件事件 → eventSet.Wait() → run() goroutine 处理
//     → deviceByPlacement.get() O(1) 查找受影响设备
//     → unhealthy channel → driver.deviceHealthEvents() 消费
//     → 更新 ResourceSlice 移除不健康设备
//
// 已知局限：
//   - devicePlacementMap 在启动时构建一次，不跟踪 DynamicMIG 模式下的设备变化
//   - MIG CI ID 当前错误地使用了 GI ID（应使用 ciInfo.Id）
//   - 丢弃 channel 通知时，调度器可能继续分配已不健康的设备
package main

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"k8s.io/klog/v2"
)

// FullGPUInstanceID 是完整物理 GPU（非 MIG 模式）在 devicePlacementMap 中使用的特殊标识符。
// 值 0xFFFFFFFF 远超实际 MIG 实例 ID 的范围（通常从 0 开始到几十），可安全用于区分"完整 GPU"和"MIG 实例"。
// 设计原因：devicePlacementMap 需要统一存储完整 GPU 和 MIG 实例，
// 而 MIG 的 GI/CI ID 从 0 开始编号，需要一个特殊值代表"非 MIG 配置"。
const (
	FullGPUInstanceID uint32 = 0xFFFFFFFF
)

// devicePlacementMap 是三层嵌套 map：parentUUID → GI ID → CI ID → AllocatableDevice。
//
// 存在目的：在 NVML XID 健康事件到达时，O(1) 定位受影响的具体设备，
// 而无需线性扫描 allocatable 列表（在有大量 MIG 设备的场景中性能差异显著）。
// 例如，一个 A100 最多可创建 7 个 MIG 实例，一个节点上可能有多个 A100，
// 线性扫描的开销会随设备数线性增长，而 map 查找始终是 O(1)。
//
// 完整 GPU 使用 FullGPUInstanceID（0xFFFFFFFF）作为 GI 和 CI 的 ID：
//   - placementMap[gpuUUID][FullGPUInstanceID][FullGPUInstanceID] = fullGpuDevice
//
// MIG 实例使用实际的 GI ID 和 CI ID：
//   - placementMap[parentUUID][giID][ciID] = migDevice
//
// 注意：当前实现假设设备集合在健康监控器的生命周期内不变，
// 动态 MIG（DynamicMIG）模式下此假设不成立，需后续修复。
// 在 DynamicMIG 模式下，新创建的 MIG 设备不会出现在查找表中，
// 触发事件时会被忽略并打印调试日志。
type devicePlacementMap map[string]map[uint32]map[uint32]*AllocatableDevice

// nvmlDeviceHealthMonitor 通过 NVML 事件机制持续监控 GPU 硬件健康状态。
//
// 工作原理：
//  1. 启动时订阅 NVML XID 关键错误事件（EventTypeXidCriticalError）
//  2. run() goroutine 在事件循环中等待 NVML 事件（5秒超时轮询）
//  3. 收到关键错误后，通过 deviceByPlacement 查找受影响设备
//  4. 将受影响的设备指针发送到 unhealthy channel
//  5. driver.deviceHealthEvents goroutine 消费 unhealthy channel，
//     将设备标记为不健康并重新发布 ResourceSlice
//
// 设计权衡：
//   - unhealthy channel 带缓冲（大小=设备总数），避免事件处理阻塞 run 循环。
//     若 channel 无缓冲，run() 在发送通知时会被阻塞直到消费者处理完毕，
//     这可能导致 NVML 事件队列溢出，丢失后续硬件事件。
//   - skippedXids 过滤掉应用层错误（如内存访问越界），这类 XID 不代表 GPU 硬件故障。
//     例如 XID 31（内存页面错误）通常是应用程序 bug，而非 GPU 故障。
type nvmlDeviceHealthMonitor struct {
	// nvmllib 是 NVML 库接口，用于初始化 NVML 会话、创建事件集、查询设备句柄等。
	nvmllib nvml.Interface

	// eventSet 是 NVML 事件集，所有 GPU 注册的健康事件通过此集合统一接收。
	// 调用 eventSet.Wait() 可阻塞等待下一个事件。
	eventSet nvml.EventSet

	// unhealthy 是用于通知 driver 有设备变为不健康的 channel（带缓冲）。
	// 缓冲大小等于可分配设备总数，保证短期突发事件不丢失通知。
	unhealthy chan *AllocatableDevice

	// deviceByPlacement 是用于事件到设备的 O(1) 查找表。
	// 当 NVML 事件到达时，事件中包含 (UUID, GI ID, CI ID) 三元组，
	// 通过此 map 可在 O(1) 时间内定位到对应的 AllocatableDevice。
	deviceByPlacement devicePlacementMap

	// skippedXids 是应忽略的 XID 集合（应用级错误，不代表 GPU 故障）。
	// 硬编码忽略常见的应用层 XID，同时支持通过命令行标志添加额外的忽略项。
	skippedXids map[uint64]bool

	// wg 用于等待 run() goroutine 退出，确保 Stop() 时所有资源已释放。
	wg sync.WaitGroup
}

// newNvmlDeviceHealthMonitor 创建 NVML 设备健康监控器实例（不启动监控，仅初始化）。
//
// 初始化时临时调用 NVML Init 以验证 NVML 可用性，随即 Shutdown（为 Start 留出完整初始化机会）。
// 这种"试探性初始化"模式的原因：Start() 中需要长生命周期的 NVML 会话，
// 而 newNvmlDeviceHealthMonitor 可能在 NVML 尚未完全就绪时被调用（如在容器启动早期）。
//
// unhealthy channel 大小设为 allocatable 设备总数，保证短期突发事件不丢失。
// 选择此大小的原因：理论上所有设备可能同时变为不健康（如掉电场景），
// 缓冲大小等于设备总数确保不会因 channel 满而丢弃通知。
//
// 参数：
//   - config：插件配置，包含命令行标志（如 additionalXidsToIgnore）
//   - allocatable：当前节点上所有可分配设备的列表
//   - nvdevlib：设备库实例，提供 NVML 接口
//
// 返回值：
//   - *nvmlDeviceHealthMonitor：初始化完成的监控器实例
//   - error：NVML 库不可用或初始化失败时返回错误
func newNvmlDeviceHealthMonitor(config *Config, allocatable AllocatableDevices, nvdevlib *deviceLib) (*nvmlDeviceHealthMonitor, error) {
	// 检查 NVML 库是否可用
	if nvdevlib.nvmllib == nil {
		return nil, fmt.Errorf("nvml library is nil")
	}

	// 临时初始化 NVML 以验证可用性
	if ret := nvdevlib.nvmllib.Init(); ret != nvml.SUCCESS {
		return nil, fmt.Errorf("failed to initialize NVML: %v", ret)
	}
	// 验证完毕后立即关闭，真正的长生命周期会话在 Start() 中创建
	defer func() {
		_ = nvdevlib.nvmllib.Shutdown()
	}()

	// 构建监控器实例
	m := &nvmlDeviceHealthMonitor{
		nvmllib:           nvdevlib.nvmllib,
		unhealthy:         make(chan *AllocatableDevice, len(allocatable)),
		deviceByPlacement: getDevicePlacementMap(allocatable),
		skippedXids:       xidsToSkip(config.flags.additionalXidsToIgnore),
	}
	return m, nil
}

// Start 启动设备健康监控：
//  1. 初始化 NVML 会话（长生命周期，贯穿监控器整个运行期间）
//  2. 创建 NVML 事件集（EventSet），用于统一接收所有 GPU 的事件
//  3. 为每个 GPU 注册 XID 关键错误事件订阅
//  4. 启动 run() goroutine 开始事件轮询循环
//
// 若初始化失败（NVML Init 或 EventSetCreate），在返回错误前自动调用 NVML Shutdown（通过 defer）。
// 仅在成功时保持 NVML 会话存活（由 Stop() 负责最终关闭）。
// 这个设计确保了资源不会泄漏：无论 Start 成功还是失败，NVML 会话要么被 Start 的 defer 关闭，
// 要么在 Stop 中被正确关闭。
//
// 参数：
//   - ctx：上下文，用于控制 goroutine 的生命周期（ctx.Done() 触发退出）
//
// 返回值：
//   - error：NVML 初始化或事件集创建失败时返回错误
func (m *nvmlDeviceHealthMonitor) Start(ctx context.Context) (rerr error) {
	// 初始化长生命周期的 NVML 会话
	if ret := m.nvmllib.Init(); ret != nvml.SUCCESS {
		return fmt.Errorf("failed to initialize NVML: %v", ret)
	}

	// 若后续步骤失败，确保 NVML 会话被关闭（仅在返回错误时执行）
	defer func() {
		if rerr != nil {
			_ = m.nvmllib.Shutdown()
		}
	}()

	// 创建 NVML 事件集
	klog.V(4).Info("creating NVML events for device health monitor")
	eventSet, ret := m.nvmllib.EventSetCreate()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("failed to create event set: %w", ret)
	}

	// 保存事件集引用，后续 run() 和 Stop() 使用
	m.eventSet = eventSet

	// 为每个 GPU 注册事件订阅
	klog.V(4).Info("registering NVML events for device health monitor")
	m.registerEventsForDevices()

	// 启动后台事件处理 goroutine
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		m.run(ctx)
	}()

	klog.V(4).Info("started device health monitoring")
	return nil
}

// registerEventsForDevices 为 deviceByPlacement 中每个父 GPU 注册 XID 事件订阅。
//
// 订阅的事件类型（三类 OR 组合）：
//   - EventTypeXidCriticalError：XID 关键错误，最重要的故障指标，表示 GPU 遇到了无法恢复的硬件错误
//   - EventTypeDoubleBitEccError：ECC 双比特错误，不可纠正，需标记为不健康
//   - EventTypeSingleBitEccError：ECC 单比特错误，可纠正，但可能预示硬件老化
//
// 若 GPU 不支持事件订阅（老旧 GPU，如 Kepler 架构及更早），只打印 Warning 而非失败：
// 不支持健康检查的设备默认为"健康"，不会被标记为不健康。
// 这是为了兼容性考虑：老旧 GPU 仍然可以使用，只是无法被监控健康状态。
//
// 若无法获取设备句柄或查询支持事件类型，则将该 GPU 下的所有设备标记为不健康（保守策略）：
// 宁可误报不健康（导致设备不可用），也不漏报故障（可能导致任务在故障 GPU 上运行）。
func (m *nvmlDeviceHealthMonitor) registerEventsForDevices() {
	// 构造事件掩码：三类事件按位 OR 组合
	eventMask := uint64(nvml.EventTypeXidCriticalError | nvml.EventTypeDoubleBitEccError | nvml.EventTypeSingleBitEccError)

	// 遍历每个父 GPU
	for parentUUID, giMap := range m.deviceByPlacement {
		// 通过 UUID 获取 NVML 设备句柄
		gpu, ret := m.nvmllib.DeviceGetHandleByUUID(parentUUID)
		if ret != nvml.SUCCESS {
			// 无法获取设备句柄，保守地将该 GPU 下所有设备标记为不健康
			klog.Warningf("Unable to get device handle from UUID[%s]: %v; marking it as unhealthy", parentUUID, ret)
			m.markAllMigDevicesUnhealthy(giMap)
			continue
		}

		// 查询该 GPU 支持的事件类型
		supportedEvents, ret := gpu.GetSupportedEventTypes()
		if ret != nvml.SUCCESS {
			// 无法查询支持的事件类型，保守地标记为不健康
			klog.Warningf("unable to determine the supported events for %s: %v; marking it as unhealthy", parentUUID, ret)
			m.markAllMigDevicesUnhealthy(giMap)
			continue
		}

		// 仅注册该 GPU 实际支持的事件（eventMask 与 supportedEvents 取交集）
		ret = gpu.RegisterEvents(eventMask&supportedEvents, m.eventSet)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			// 老旧 GPU 不支持事件订阅，仅打印警告，不标记为不健康
			klog.Warningf("Device %v is too old to support healthchecking.", parentUUID)
		}
		if ret != nvml.SUCCESS {
			// 注册失败（非 NOT_SUPPORTED），保守地标记为不健康
			klog.Warningf("unable to register events for %s: %v; marking it as unhealthy", parentUUID, ret)
			m.markAllMigDevicesUnhealthy(giMap)
		}
	}
}

// Stop 停止健康监控器：等待 run goroutine 退出，释放事件集，关闭 NVML 会话，关闭 unhealthy channel。
//
// 在 driver.Shutdown() 中调用，确保资源按正确顺序释放：
//  1. m.wg.Wait() - 等待 run goroutine 退出（goroutine 通过 ctx.Done() 感知停止信号）
//  2. m.eventSet.Free() - 释放 NVML 事件集资源
//  3. m.nvmllib.Shutdown() - 关闭 NVML 会话
//  4. close(m.unhealthy) - 关闭 channel，通知消费者（driver）不再有新事件
//
// 这个顺序确保了：goroutine 不再访问 eventSet 后才释放它，
// NVML 会话在所有 NVML 赃源释放后才关闭，
// channel 在 NVML 关闭后才关闭（避免 goroutine 尝试发送到已关闭 channel）。
func (m *nvmlDeviceHealthMonitor) Stop() {
	// nil 检查：允许安全地调用 Stop() 即使监控器未初始化
	if m == nil {
		return
	}
	klog.V(6).Info("stopping health monitor")

	// 等待 run goroutine 完全退出
	m.wg.Wait()

	// 释放 NVML 事件集
	if ret := m.eventSet.Free(); ret != nvml.SUCCESS {
		klog.Warningf("failed to unset events: %v", ret)
	}

	// 关闭 NVML 会话
	if ret := m.nvmllib.Shutdown(); ret != nvml.SUCCESS {
		klog.Warningf("failed to shutdown NVML: %v", ret)
	}

	// 关闭 unhealthy channel，消费者将收到零值并退出循环
	close(m.unhealthy)
}

// run 是事件处理主循环，每次调用 EventSet.Wait(5000) 最多阻塞 5 秒等待 NVML 事件。
//
// 事件处理流程：
//  1. 忽略非 XID 关键错误事件（如 ECC 单/双比特错误，单独由订阅逻辑处理）
//  2. 忽略在 skippedXids 列表中的 XID（应用层错误，GPU 仍然健康）
//  3. 通过 deviceByPlacement.get(UUID, GI, CI) O(1) 定位受影响设备
//  4. 将受影响设备发送到 unhealthy channel，由消费者更新 ResourceSlice
//
// 错误处理：NVML EventSet.Wait 可能返回多种错误，目前对大多数错误只打印日志并继续，
// 仅对 ERROR_GPU_IS_LOST 做特殊处理（标记所有设备为不健康）。
// NVML 文档中对错误码的描述有限，无法针对每种错误做精确处理。
// 选择"继续轮询"而非"退出"的原因：短暂的 NVML 错误可能是暂时的（如驱动重置），
// 退出整个监控循环会丢失后续的恢复事件。
func (m *nvmlDeviceHealthMonitor) run(ctx context.Context) {
	for {
		select {
		// 收到上下文取消信号，退出事件循环
		case <-ctx.Done():
			klog.V(6).Info("Stopping event-driven GPU health monitor...")
			return
		default:
			// 等待 NVML 事件，超时 5000 毫秒
			// 超时设计：5 秒是 NVML 推荐的轮询间隔，在及时性和 CPU 开销之间取得平衡
			event, ret := m.eventSet.Wait(5000)
			if ret == nvml.ERROR_TIMEOUT {
				// 超时是正常行为，继续下一轮轮询
				continue
			}
			if ret != nvml.SUCCESS {
				if ret == nvml.ERROR_GPU_IS_LOST {
					// GPU 完全丢失（如 PCIe 设备被移除），标记所有设备为不健康
					klog.Warningf("GPU is lost error: %v; Marking all devices as unhealthy", ret)
					m.markAllDevicesUnhealthy()
					continue
				}
				// 其他 NVML 错误，打印日志并继续轮询
				klog.V(6).Infof("Error waiting for NVML event: %v. Retrying...", ret)
				continue
			}

			// 解析事件字段
			eType := event.EventType     // 事件类型（如 EventTypeXidCriticalError）
			xid := event.EventData       // XID 错误码（仅对 XID 类型事件有意义）
			gi := event.GpuInstanceId    // 触发事件的 GPU Instance ID
			ci := event.ComputeInstanceId // 触发事件的 Compute Instance ID

			// 仅处理 XID 关键错误事件，跳过其他类型（ECC 等）。
			// ECC 错误虽然也订阅了，但其处理逻辑不同（通常只是记录，不立即标记不健康）。
			if eType != nvml.EventTypeXidCriticalError {
				klog.V(6).Infof("Skipping non-nvmlEventTypeXidCriticalError event: Data=%d, Type=%d, GI=%d, CI=%d", xid, eType, gi, ci)
				continue
			}

			// 跳过在忽略列表中的 XID（应用层错误，如内存越界、图形引擎异常）。
			// 这些 XID 表示应用程序触发了 GPU 错误，但 GPU 硬件本身仍然正常，
			// 不应影响后续工作负载的调度。
			if m.skippedXids[xid] {
				klog.V(6).Infof("Skipping XID event: Data=%d, Type=%d, GI=%d, CI=%d", xid, eType, gi, ci)
				continue
			}

			klog.V(4).Infof("Processing event XID=%d event", xid)

			// 获取触发事件的设备 UUID。失败时保守地标记所有设备为不健康。
			// 这种保守策略避免了在无法确定故障设备时，错误地让工作负载继续在故障 GPU 上运行。
			eventUUID, ret := event.Device.GetUUID()
			if ret != nvml.SUCCESS {
				klog.Warningf("Failed to determine uuid for event %v: %v; Marking all devices as unhealthy.", event, ret)
				m.markAllDevicesUnhealthy()
				continue
			}

			// O(1) 查找受影响设备。若不在 deviceByPlacement 中（如动态 MIG 设备），忽略该事件。
			// 动态 MIG 设备在创建时未注册到查找表，因此其事件无法被处理。
			// 这是一个已知的局限，将在后续版本中修复。
			affectedDevice := m.deviceByPlacement.get(eventUUID, gi, ci)
			if affectedDevice == nil {
				klog.V(6).Infof("Ignoring event for unexpected device (UUID:%s, GI:%d, CI:%d)", eventUUID, gi, ci)
				continue
			}

			// 将受影响设备发送到 unhealthy channel
			klog.V(4).Infof("Sending unhealthy notification for device %s due to event type:%v and event data:%d", affectedDevice.UUID(), eType, xid)
			m.unhealthy <- affectedDevice
		}
	}
}

// Unhealthy 返回不健康设备通知 channel 的只读视图，供 driver 消费。
// 返回只读 channel（<-chan）防止消费者向 channel 发送数据，确保数据流方向正确：
// 只有监控器能发送，driver 只能接收。
//
// 返回值：
//   - <-chan *AllocatableDevice：只读的设备不健康通知 channel
func (m *nvmlDeviceHealthMonitor) Unhealthy() <-chan *AllocatableDevice {
	return m.unhealthy
}

// markAllDevicesUnhealthy 将 deviceByPlacement 中所有已知设备标记为不健康。
// 在 GPU 完全丢失（ERROR_GPU_IS_LOST）或无法解析事件 UUID 时调用（保守策略）。
// 这种保守策略的权衡：可能导致误报（所有设备不可用），但避免了漏报
// （工作负载在故障 GPU 上运行导致数据损坏）。
func (m *nvmlDeviceHealthMonitor) markAllDevicesUnhealthy() {
	for _, giMap := range m.deviceByPlacement {
		m.markAllMigDevicesUnhealthy(giMap)
	}
}

// markAllMigDevicesUnhealthy 将指定父 GPU 下的所有设备（完整 GPU 或 MIG 实例）标记为不健康。
//
// 使用非阻塞发送（select + default）避免因 unhealthy channel 已满而导致 goroutine 死锁。
// 若 channel 满，丢弃通知并打印 Error 日志。
// 丢弃通知的风险：内存中健康状态已更新，但 ResourceSlice 未及时同步，
// 调度器可能继续将任务调度到已不健康的设备。这是当前设计的已知局限。
//
// 可能出现 channel 满的场景：
//   - 多个 GPU 同时故障，产生了大量不健康通知
//   - driver 的消费者 goroutine 处理速度慢于事件产生速度
//
// 未来改进方向：增加 channel 大小或实现背压机制确保不丢失通知。
//
// 参数：
//   - giMap：指定父 GPU 下所有 GI → CI → 设备的映射
func (m *nvmlDeviceHealthMonitor) markAllMigDevicesUnhealthy(giMap map[uint32]map[uint32]*AllocatableDevice) {
	// 遍历该父 GPU 下的所有 GI
	for _, ciMap := range giMap {
		// 遍历该 GI 下的所有 CI 对应的设备
		for _, dev := range ciMap {
			select {
			// 尝试将设备发送到 unhealthy channel
			case m.unhealthy <- dev:
				klog.V(6).Infof("Marked device %s as unhealthy", dev.UUID())
			// channel 已满，丢弃通知
			default:
				klog.Errorf("Unhealthy channel full. Dropping unhealthy notification for device %s", dev.UUID())
			}
		}
	}
}

// getDevicePlacementMap 构造 devicePlacementMap，
// 为健康事件处理器提供 O(1) 的设备查找能力。
//
// 遍历所有可分配设备，按以下规则建立索引：
//   - 完整 GPU：parentUUID = 设备自身 UUID，GI/CI ID = FullGPUInstanceID（0xFFFFFFFF）
//     原因：完整 GPU 没有 MIG 分区，使用特殊 ID 与 MIG 实例区分
//   - 静态 MIG：parentUUID = 父 GPU 的 UUID，GI ID = gIInfo.Id，CI ID = gIInfo.Id（目前使用 GI ID）
//     注意：CI ID 当前错误地使用了 GI ID，应使用 ciInfo.Id，这是一个已知 bug
//   - 其他类型（动态 MIG、VFIO）：跳过，打印调试日志
//     原因：动态 MIG 设备在运行时创建，无法在启动时预知；VFIO 设备不通过 NVML 事件监控
//
// 已知问题：MIG 的 CI ID 当前错误地使用了 GI ID（应使用 ciInfo.Id）。
// 已知局限：DynamicMIG 模式下设备集合可能动态变化，但此 map 只在启动时构建一次，
// 后续新建的 MIG 设备不会自动加入查找表（需后续修复）。
//
// 参数：
//   - allocatable：当前节点上所有可分配设备的列表
//
// 返回值：
//   - devicePlacementMap：构建完成的三层嵌套查找表
func getDevicePlacementMap(allocatable AllocatableDevices) devicePlacementMap {
	placementMap := make(devicePlacementMap)

	for _, d := range allocatable {
		var parentUUID string
		var giID, ciID uint32

		switch d.Type() {
		case GpuDeviceType:
			// 完整 GPU：使用自身 UUID 作为 parentUUID
			parentUUID = d.UUID()
			if parentUUID == "" {
				continue
			}
			// 完整 GPU 没有 MIG 分区，使用特殊 ID
			giID = FullGPUInstanceID
			ciID = FullGPUInstanceID

		case MigStaticDeviceType:
			// 静态 MIG 设备：使用父 GPU 的 UUID
			parentUUID = d.MigStatic.parent.UUID
			if parentUUID == "" {
				continue
			}
			// GI ID 来自 NVML GpuInstanceInfo
			giID = d.MigStatic.gIInfo.Id
			// 注意：这里 CI ID 错误地使用了 GI ID，应使用 ciInfo.Id
			// 这是一个已知 bug，目前未修复
			ciID = d.MigStatic.gIInfo.Id

		default:
			// 其他设备类型（动态 MIG、VFIO）不纳入健康监控查找表
			klog.V(4).Infof("getDevicePlacementMap: skipping device with type: %s", d.Type())
			continue
		}

		// 将设备添加到查找表
		placementMap.addDevice(parentUUID, giID, ciID, d)
	}
	return placementMap
}

// addDevice 向 devicePlacementMap 插入一个设备条目，按需初始化中间层 map。
//
// 三层 map 的惰性初始化：只有当插入新 key 时才创建对应的子 map，
// 避免为不存在的 parentUUID 或 giID 预分配内存。
//
// 参数：
//   - parentUUID：父 GPU 的 UUID
//   - giID：GPU Instance ID（完整 GPU 使用 FullGPUInstanceID）
//   - ciID：Compute Instance ID（完整 GPU 使用 FullGPUInstanceID）
//   - d：待插入的可分配设备指针
func (p devicePlacementMap) addDevice(parentUUID string, giID uint32, ciID uint32, d *AllocatableDevice) {
	// 第一层：parentUUID → GI map，若不存在则初始化
	if _, ok := p[parentUUID]; !ok {
		p[parentUUID] = make(map[uint32]map[uint32]*AllocatableDevice)
	}
	// 第二层：giID → CI map，若不存在则初始化
	if _, ok := p[parentUUID][giID]; !ok {
		p[parentUUID][giID] = make(map[uint32]*AllocatableDevice)
	}
	// 第三层：ciID → 设备指针，直接赋值
	p[parentUUID][giID][ciID] = d
}

// get 通过 (uuid, GI ID, CI ID) 三元组在 devicePlacementMap 中 O(1) 查找设备。
// 若任意层不存在，返回 nil。
//
// 这个方法在事件处理循环中被频繁调用，O(1) 的查找性能至关重要。
// 替代方案（线性扫描所有设备）在大量 MIG 设备的场景下性能不可接受。
//
// 参数：
//   - uuid：设备的父 GPU UUID
//   - gi：GPU Instance ID
//   - ci：Compute Instance ID
//
// 返回值：
//   - *AllocatableDevice：找到的设备指针，未找到返回 nil
func (p devicePlacementMap) get(uuid string, gi, ci uint32) *AllocatableDevice {
	// 第一层查找：parentUUID
	giMap, ok := p[uuid]
	if !ok {
		return nil
	}

	// 第二层查找：GI ID
	ciMap, ok := giMap[gi]
	if !ok {
		return nil
	}

	// 第三层查找：CI ID
	return ciMap[ci]
}

// getAdditionalXids 解析逗号分隔的 XID 字符串，返回有效的 XID 列表。
// 无效值（非数字、空白）被静默忽略并打印调试日志。
//
// 支持的输入格式示例：
//   - "31,43,68" → 解析为 [31, 43, 68]
//   - "31, , abc, 43" → 解析为 [31, 43]（空白和 "abc" 被忽略）
//   - "" → 返回 nil
//
// 用于支持用户通过 --additional-xids-to-ignore 标志自定义忽略的 XID 列表。
// 这在特定工作负载持续触发已知应用层 XID 错误（如 XID 31），
// 但用户确定 GPU 硬件正常时非常有用。
//
// 参数：
//   - input：逗号分隔的 XID 字符串
//
// 返回值：
//   - []uint64：解析成功的 XID 列表
func getAdditionalXids(input string) []uint64 {
	if input == "" {
		return nil
	}

	var additionalXids []uint64
	klog.V(6).Infof("Creating a list of additional xids to ignore: [%s]", input)
	for _, additionalXid := range strings.Split(input, ",") {
		// 去除每项的前后空白
		trimmed := strings.TrimSpace(additionalXid)
		if trimmed == "" {
			// 跳过空白项
			continue
		}
		// 尝试将字符串解析为无符号 64 位整数
		xid, err := strconv.ParseUint(trimmed, 10, 64)
		if err != nil {
			// 非数字值静默忽略，仅打印调试日志
			klog.V(6).Infof("Ignoring malformed Xid value %v: %v", trimmed, err)
			continue
		}
		additionalXids = append(additionalXids, xid)
	}

	return additionalXids
}

// xidsToSkip 构造需要忽略的 XID 集合（硬编码 + 用户自定义）。
//
// 硬编码忽略的 XID（应用层错误，GPU 本身仍然健康）：
//   - 13：图形引擎异常（Graphics Engine Exception）—— 通常由图形应用程序的着色器错误触发
//   - 31：GPU 内存页面错误（GPU memory page fault）—— 应用程序访问了无效的 GPU 内存地址
//   - 43：GPU 停止处理（GPU stopped processing）—— 应用程序导致的 GPU 停止
//   - 45：因前序错误触发的抢占式清理（Preemptive cleanup due to previous errors）—— 上游错误的连锁反应
//   - 68：视频处理器异常（Video processor exception）—— 视频编解码应用程序触发的异常
//   - 109：上下文切换超时（Context Switch Timeout Error）—— 长时间运行的内核导致的超时
//
// 这些 XID 被忽略的共同特征：它们是由应用程序的行为触发的，
// 而非 GPU 硬件本身的故障。重启应用程序通常可以恢复正常运行，
// 不需要将 GPU 标记为不健康（否则会影响该 GPU 上的所有其他工作负载）。
//
// 参考：http://docs.nvidia.com/deploy/xid-errors/index.html#topic_4
//
// 参数：
//   - additionalXids：用户通过 --additional-xids-to-ignore 标志提供的额外 XID 列表（逗号分隔字符串）
//
// 返回值：
//   - map[uint64]bool：以 XID 编号为 key、true 为值的集合，用于 O(1) 查询
func xidsToSkip(additionalXids string) map[uint64]bool {
	// 硬编码的忽略 XID 列表
	ignoredXids := []uint64{
		13,  // 图形引擎异常（应用层错误）
		31,  // GPU 内存页面错误（应用层错误）
		43,  // GPU 停止处理（应用层错误）
		45,  // 前序错误引发的抢占式清理（应用层错误）
		68,  // 视频处理器异常（应用层错误）
		109, // 上下文切换超时（应用层错误）
	}

	// 构建以 XID 编号为 key 的 map，支持 O(1) 查询
	skippedXids := make(map[uint64]bool)
	for _, id := range ignoredXids {
		skippedXids[id] = true
	}

	// 合并用户自定义的额外忽略 XID
	for _, additionalXid := range getAdditionalXids(additionalXids) {
		skippedXids[additionalXid] = true
	}
	return skippedXids
}
