/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// cleanup 包实现了 Checkpoint 中"僵尸 ResourceClaim"的周期性清理机制。
//
// 所谓"僵尸 Claim"是指：Checkpoint 中记录为 PrepareStarted 状态（即 Prepare 事务已开始但未完成），
// 但对应的 ResourceClaim 对象已从 Kubernetes API Server 中删除的条目。
// 典型场景是：kubelet 插件在执行 NodeUnprepareResources 期间崩溃或被强制终止，
// 导致 Unprepare 未完成、Checkpoint 未能更新，残留的 PrepareStarted 记录永远不会被自动清理。
//
// 本包采用生产者-消费者架构：
//   - 生产者 goroutine（triggerPeriodically）：启动时立即执行一次清理，之后每隔 10 分钟向队列提交清理任务。
//   - 消费者 goroutine（worker）：从队列中取出任务信号，执行完整的清理流程。
//
// 队列容量为 1，保证同一时间至多一个清理任务在等待执行，避免任务堆积。
package main

import (
	"context"
	"fmt"
	"sync"
	"time"

	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	draclient "k8s.io/dynamic-resource-allocation/client"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
)

// ResourceClaimCleanupInterval 是 CheckpointCleanupManager 的周期性清理间隔。
// 每隔此时间触发一次对"僵尸 Claim"（PrepareStarted 但已从 API Server 删除）的清理。
const (
	ResourceClaimCleanupInterval = 10 * time.Minute
)

// TypeUnprepCallable 是 Unprepare 操作的函数签名，由 driver.nodeUnprepareResource 满足。
// 定义为类型别名便于在 CheckpointCleanupManager 中注入具体的 Unprepare 实现，
// 同时方便单元测试时替换为 mock 函数。
type TypeUnprepCallable = func(ctx context.Context, claimRef kubeletplugin.NamespacedObject) error

// CheckpointCleanupManager 负责清理"僵尸 Claim"：
// 即 Checkpoint 中处于 PrepareStarted 状态，但对应 ResourceClaim 已从 API Server 删除的条目。
// 典型场景：插件在完成 NodeUnprepareResources 之前崩溃，导致 Checkpoint 未更新。
// 采用 producer-consumer 架构：定时器触发清理任务入队，worker goroutine 消费并执行清理。
type CheckpointCleanupManager struct {
	waitGroup     sync.WaitGroup   // 用于等待生产者和消费者 goroutine 优雅退出
	cancelContext context.CancelFunc // 用于取消内部 context，通知所有 goroutine 停止
	queue         chan struct{}    // 容量为 1 的缓冲 channel，保证同一时间最多一个清理任务排队
	devicestate   *DeviceState     // 设备状态管理器，提供 Checkpoint 读取能力
	draclient     *draclient.Client // DRA 客户端，用于向 API Server 查询 ResourceClaim

	unprepfunc TypeUnprepCallable // 实际执行设备释放的回调函数（即 driver.nodeUnprepareResource）
}

// NewCheckpointCleanupManager 创建 CheckpointCleanupManager 实例。
// 参数：
//   - s: DeviceState 实例，用于读取 Checkpoint 中记录的 Claim 信息
//   - client: DRA 客户端，用于向 API Server 查询 ResourceClaim 是否存在
// queue 使用容量为 1 的缓冲 channel，确保至多有一个清理任务在队列中等待。
// TODO: 评估是否应使用条件变量（condition variable）替代当前 channel 实现。
func NewCheckpointCleanupManager(s *DeviceState, client *draclient.Client) *CheckpointCleanupManager {
	return &CheckpointCleanupManager{
		devicestate: s,
		draclient:   client,
		queue:       make(chan struct{}, 1),
	}
}

// Start 启动 CheckpointCleanupManager 的两个 goroutine：
//   - triggerPeriodically（生产者）：定时器生产者，每隔 ResourceClaimCleanupInterval 提交清理任务
//   - worker（消费者）：从 queue 接收任务信号并执行 cleanup()
//
// 参数：
//   - ctx: 父级 context，用于控制生命周期
//   - unprepfunc: 实际执行 Unprepare 操作的回调函数
//
// 返回值：始终返回 nil（当前无启动失败的场景）
func (m *CheckpointCleanupManager) Start(ctx context.Context, unprepfunc TypeUnprepCallable) error {
	// 基于父 context 创建可取消的子 context，用于在 Stop() 时通知所有 goroutine 退出
	ctx, cancel := context.WithCancel(ctx)
	m.cancelContext = cancel
	m.unprepfunc = unprepfunc

	// 启动生产者 goroutine
	m.waitGroup.Add(1)
	go func() {
		defer m.waitGroup.Done()
		// 启动生产者：周期性地提交清理任务
		m.triggerPeriodically(ctx)
	}()

	// 启动消费者 goroutine
	m.waitGroup.Add(1)
	go func() {
		defer m.waitGroup.Done()
		// 启动消费者：从队列中接收任务并执行清理
		m.worker(ctx)
	}()

	klog.V(6).Infof("CheckpointCleanupManager started")
	return nil
}

// Stop 停止 CheckpointCleanupManager：取消内部 context，等待两个 goroutine 退出后返回。
// 调用 cancelContext() 会使生产者的 ticker 循环和消费者的 queue 循环都收到 ctx.Done() 信号，
// 从而安全退出各自的 goroutine。waitGroup.Wait() 确保两个 goroutine 都完成清理后再返回。
func (m *CheckpointCleanupManager) Stop() error {
	if m.cancelContext != nil {
		m.cancelContext()
	}
	m.waitGroup.Wait()
	return nil
}

// cleanup 是周期性清理的主入口，在插件启动时立即执行一次，之后每隔 10 分钟触发一次。
// 它从 Checkpoint 中找出所有处于 PrepareStarted 状态的 Claim（即"事务中间态"），
// 逐一调用 unprepareIfStale() 判断是否为僵尸 Claim 并按需清理。
// 整个函数是尽力而为（best-effort）的：单个 Claim 的清理失败不影响其他 Claim 的处理。
//
// 为什么不加 DeviceState 锁？
//  1. 此处读 Checkpoint 只是发现"候选 Claim"，不需要强一致性——
//     并发写导致漏读某个 Claim，下一轮清理会补上。
//  2. 真正的"这个 Claim 是否应该清理"判断依赖 API Server（权威来源），
//     而非本地 Checkpoint 状态。
//  3. 实际的 Checkpoint 写操作在 nodeUnprepareResource() 中，
//     该函数已通过 pulock 文件锁保证原子性读写。
//  4. 在整个清理过程（可能涉及多次 API Server 调用，耗时数秒）中持有 DeviceState 锁，
//     会不必要地阻塞正常的 Prepare/Unprepare 请求。
func (m *CheckpointCleanupManager) cleanup(ctx context.Context) {
	// 从 DeviceState 获取当前 Checkpoint 快照
	cp, err := m.devicestate.getCheckpoint(ctx)
	if err != nil {
		klog.Errorf("Checkpointed RC cleanup: unable to get checkpoint: %s", err)
		return
	}

	// 筛选出处于 PrepareStarted 状态的 Claim（事务中间态，可能是僵尸）
	filtered := make(PreparedClaimsByUIDV2)
	for uid, claim := range cp.V2.PreparedClaims {
		if claim.CheckpointState == ClaimCheckpointStatePrepareStarted {
			filtered[uid] = claim
		}
	}

	klog.V(4).Infof("Checkpointed RC cleanup: claims in PrepareStarted state: %d (of %d)", len(filtered), len(cp.V2.PreparedClaims))

	// 对每个候选 Claim 判断是否已从 API Server 删除（即是否为僵尸）
	for cpuid, cpclaim := range filtered {
		m.unprepareIfStale(ctx, cpuid, cpclaim)
	}
}

// unprepareIfStale 判断一个处于 PrepareStarted 状态的 Claim 是否已从 API Server 删除（僵尸），
// 若是则调用 unprepare() 执行清理。
//
// 如何通过 API Server 查找指定 UID 的 Claim？有两种方案：
//
//  方案一：List() + FieldSelector "metadata.uid=<uid>"
//    跨命名空间（甚至单命名空间）执行此查询代价极高，不适合周期性调用。
//
//  方案二：Get() 通过 name/namespace 直接查询，再比对返回对象的 UID
//    对 API Server 友好，是本函数采用的方式。
//    前提：Checkpoint 中必须存有 Name 和 Namespace 字段。
//    兼容性注意：v25.3.x 版本的驱动创建的 Checkpoint 不含 Name 字段，
//    通过检测 Name 为空来识别这种旧格式，跳过处理（待后续补充跨命名空间兜底方案）。
func (m *CheckpointCleanupManager) unprepareIfStale(ctx context.Context, cpuid string, cpclaim PreparedClaim) {
	// 旧版 Checkpoint（v25.3.x）不含 Name 字段，无法使用 Get() 查询，暂时跳过
	if cpclaim.Name == "" {
		klog.V(4).Infof("Checkpointed RC cleanup: skip checkpointed claim '%s': RC name not in checkpoint", cpuid)
		return
	}

	// 通过 name/namespace 直接查询 API Server
	claim, err := m.getClaimByName(ctx, cpclaim.Name, cpclaim.Namespace)
	if err != nil && errors.IsNotFound(err) {
		// Claim 已从 API Server 删除，是僵尸 Claim，执行清理
		klog.V(4).Infof(
			"Checkpointed RC cleanup: partially prepared claim '%s/%s:%s' is stale: not found in API server",
			cpclaim.Namespace,
			cpclaim.Name,
			cpuid)
		m.unprepare(ctx, cpuid, cpclaim)
		return
	}

	// API Server 查询出现临时错误，不需要显式重试——下一轮周期性清理会自动重试
	if err != nil {
		klog.Infof("Checkpointed RC cleanup: skip for checkpointed claim %s: getClaimByName failed (retry later): %s", cpuid, err)
		return
	}

	// 同一命名空间内不可能同时存在两个同名 ResourceClaim。
	// 若查到的 Claim UID 与 Checkpoint 中不同，说明原 Claim 已被删除并重建了同名对象，
	// Checkpoint 中记录的旧 Claim 即为僵尸，需要清理。
	if string(claim.UID) != cpuid {
		klog.V(4).Infof("Checkpointed RC cleanup: partially prepared claim '%s/%s' is stale: UID changed (checkpoint: %s, API server: %s)", cpclaim.Namespace, cpclaim.Name, cpuid, claim.UID)
		m.unprepare(ctx, cpuid, cpclaim)
		return
	}

	// Claim 仍存在于 API Server 且 UID 匹配，说明它是合法的活跃 Claim，无需清理
	klog.V(4).Infof("Checkpointed RC cleanup: partially prepared claim not stale: %s", ResourceClaimToString(claim))
}

// unprepare 对僵尸 Claim 主动发起一次 Unprepare（"自发式 Unprepare"）。
// 预期副作用：Unprepare() 成功后会将该 Claim 从 Checkpoint 中删除，
// 从而终止后续的周期性清理尝试。
//
// 这里只执行一次 Unprepare，依靠后续周期性触发来隐式重试失败的情况，
// 无需在此引入显式重试循环。
// TODO：审查 Unprepare() 中是否存在"成功但不删除 Checkpoint 条目"的代码路径，
//
//	否则该 Claim 会被无限循环清理。
func (m *CheckpointCleanupManager) unprepare(ctx context.Context, uid string, claim PreparedClaim) {
	// 构造 kubeletplugin 需要的 NamespacedObject 引用，包含 UID、Name、Namespace
	claimRef := kubeletplugin.NamespacedObject{
		UID: types.UID(uid),
		NamespacedName: types.NamespacedName{
			Name:      claim.Name,
			Namespace: claim.Namespace,
		},
	}

	// 调用注入的 Unprepare 回调函数执行实际清理
	err := m.unprepfunc(ctx, claimRef)
	if err != nil {
		// 本次清理失败，记录警告日志，等待下一轮周期性触发时重试
		klog.Warningf("Checkpointed RC cleanup: error during unprepare for %s (retried later): %s", claimRef.String(), err)
		return
	}

	// 清理成功，记录信息日志
	klog.Infof("Checkpointed RC cleanup: unprepared stale claim: %s", claimRef.String())
}

// getClaimByName 通过 name/namespace 直接向 API Server 查询 ResourceClaim 对象。
// 设置 20 秒超时：正常情况下 API Server 应低延迟响应；
// 若超时触发，说明集群处于异常状态，此次清理直接放弃（下一轮重试）。
//
// 参数：
//   - ctx: 父级 context，用于传播取消信号
//   - name: ResourceClaim 的名称
//   - ns: ResourceClaim 所在的命名空间
//
// 返回值：
//   - *resourcev1.ResourceClaim: 查询到的 ResourceClaim 对象
//   - error: 查询错误（包括超时和 Not Found）
func (m *CheckpointCleanupManager) getClaimByName(ctx context.Context, name string, ns string) (*resourcev1.ResourceClaim, error) {
	// 创建 20 秒超时的子 context，防止 API Server 无响应时永久阻塞
	childctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()

	// 通过 DRA 客户端向 API Server 发起 Get 请求
	claim, err := m.draclient.ResourceClaims(ns).Get(childctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("error getting resource claim %s/%s: %w", name, ns, err)
	}

	return claim, nil
}

// enqueueCleanup 尝试向队列提交一个清理任务，返回是否提交成功。
// queue 容量为 1，若已有任务在排队则直接丢弃本次提交（非阻塞 select default 分支），
// 避免任务堆积——上一次清理若还未执行完，说明它仍在运行中，多余的入队没有意义。
//
// 返回值：
//   - true: 成功将清理任务入队
//   - false: 队列已满，上一个清理任务仍在等待，本次入队被跳过
func (m *CheckpointCleanupManager) enqueueCleanup() bool {
	select {
	case m.queue <- struct{}{}:
		// 成功写入队列
		return true
	default:
		// channel 已满，说明上一个清理任务仍在队列中等待，不重复入队
		return false
	}
}

// worker 是清理任务的消费者 goroutine，持续运行直到 context 被取消。
// 从 queue 中读取任务信号后执行 cleanup()。
// 潜在耗时点：cplock 文件锁竞争（若有并发写 Checkpoint）、API Server 网络调用。
// 目前没有对单次 cleanup() 执行设置超时，依赖 ctx 取消来中断。
func (m *CheckpointCleanupManager) worker(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			// 收到取消信号，退出 worker goroutine
			return
		case <-m.queue:
			// 从队列中取出清理任务信号，执行清理
			m.cleanup(ctx)
		}
	}
}

// triggerPeriodically 是清理任务的生产者 goroutine：
//  1. 启动后立即同步执行一次 cleanup()（处理插件重启后遗留的僵尸 Claim）
//  2. 之后每隔 ResourceClaimCleanupInterval（10 分钟）向 queue 提交一个清理信号
//
// 注意：多个节点上的插件在驱动升级时可能同时重启，瞬间同时向 API Server 发起查询。
// TODO：引入随机抖动（jitter）以分散 API Server 负载。
func (m *CheckpointCleanupManager) triggerPeriodically(ctx context.Context) {
	// 创建每 10 分钟触发一次的定时器
	ticker := time.NewTicker(ResourceClaimCleanupInterval)
	defer ticker.Stop()

	// 启动时立刻执行一次清理，处理上次运行遗留的僵尸 Claim
	m.cleanup(ctx)
	for {
		select {
		case <-ctx.Done():
			// 收到取消信号，退出生产者 goroutine
			return
		case <-ticker.C:
			// 定时器触发，尝试提交清理任务
			if m.enqueueCleanup() {
				klog.V(6).Infof("Checkpointed RC cleanup: task submitted")
			} else {
				// 上一次清理耗时超过 10 分钟仍未完成，属于异常情况，记录警告
				klog.Warningf("Checkpointed RC cleanup: ongoing, skipped")
			}
		}
	}
}
