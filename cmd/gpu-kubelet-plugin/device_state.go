/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

// device_state.go — 设备状态管理核心实现
//
// 本文件实现了 GPU DRA 驱动的核心状态管理逻辑，负责：
//   - 设备发现与初始化（NewDeviceState）
//   - ResourceClaim 的 Prepare / Unprepare 完整流程
//   - Checkpoint 的原子性读-改-写操作（create/get/update/delete）
//   - OpaqueDeviceConfig 的解析与优先级排序
//   - 设备配置的应用（TimeSlicing、MPS、VFIO）
//   - DynamicMIG 孤儿设备的清理（DestroyUnknownMIGDevices、unpreparePartiallyPrepairedClaim）
//   - 设备健康状态更新与重叠分配检测
//   - 兄弟设备的发现与移除（VFIO 直通模式下保证独占性）
package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"slices"
	"sync"
	"time"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager"
	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"

	"github.com/sirupsen/logrus"

	configapi "github.com/NVIDIA/k8s-dra-driver-gpu/api/nvidia.com/resource/v1beta1"
	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/flock"
)

// OpaqueDeviceConfig 是一个已解码的设备配置项，关联到特定的请求名称列表。
// 来源于 ResourceClaim 或 DeviceClass 中的 opaque parameters，经 runtime.Decoder 解码后得到。
//
// 字段说明：
//   - Requests: 该配置适用的请求名称列表。空列表（len==0）表示为默认配置，适用于所有未匹配到更高优先级配置的请求。
//   - Config:   实际配置对象，类型为 runtime.Object，具体可以是 GpuConfig、MigDeviceConfig 或 VfioDeviceConfig。
//              调用方需要通过类型断言来确定具体的配置类型。
type OpaqueDeviceConfig struct {
	Requests []string       // 该配置适用的请求名称列表（空列表表示适用于所有请求）
	Config   runtime.Object // 实际配置对象（GpuConfig / MigDeviceConfig / VfioDeviceConfig）
}

// DeviceConfigState 记录一组设备在 Prepare 后的配置状态，持久化到 Checkpoint。
// 用于 Unprepare 时正确清理关联资源（如停止 MPS 守护进程、移除 CDI 挂载）。
//
// 字段说明：
//   - MpsControlDaemonID: MPS 守护进程 ID。非空字符串表示已为该设备组启动了 MPS 控制守护进程，
//                         Unprepare 时需要据此找到并停止对应守护进程。
//   - containerEdits:     CDI 容器编辑项。包含 MPS 相关的环境变量设置和文件挂载信息，
//                         用于生成 CDI Claim spec 中的设备条目。
type DeviceConfigState struct {
	MpsControlDaemonID string `json:"mpsControlDaemonID"` // MPS 守护进程 ID（非空表示 MPS 已启动）
	containerEdits     *cdiapi.ContainerEdits              // CDI 容器编辑项（如 MPS 环境变量和挂载）
}

// DeviceState 是整个插件的核心状态容器，封装了所有设备管理操作。
// 通过 sync.Mutex 保护内存状态，通过 cplock 文件锁保护 Checkpoint 文件的跨进程一致性。
//
// 主要职责：
//   - Prepare()：为 ResourceClaim 分配并初始化 GPU/MIG/VFIO 设备，写入 CDI spec 和 Checkpoint
//   - Unprepare()：释放设备、清理 CDI spec、更新 Checkpoint
//   - DestroyUnknownMIGDevices()：启动时清理孤儿 MIG 设备
//
// 字段说明：
//   - sync.Mutex:             互斥锁，保护 allocatable 等内存状态的并发访问安全
//   - cdi:                    CDI Handler，负责生成和管理 CDI spec 文件
//   - tsManager:              TimeSlicing 管理器，控制 GPU 时间切片配置
//   - mpsManager:             MPS 管理器，控制多进程服务守护进程的生命周期
//   - vfioPciManager:         VFIO PCI 管理器，控制 GPU 设备在 nvidia 和 vfio-pci 驱动间切换
//   - checkpointCleanupManager: Checkpoint 清理管理器，处理残留的 Checkpoint 条目
//   - allocatable:            扁平设备映射（设备名 → AllocatableDevice），用于按名称快速查找
//   - config:                 插件配置对象
//   - perGPUAllocatable:      按 GPU minor 号分组的设备映射，用于生成 per-GPU ResourceSlice
//   - nvdevlib:               NVML 设备库封装
//   - checkpointManager:      Kubernetes Checkpoint Manager，负责 Checkpoint 文件的序列化与持久化
//   - cplock:                 Checkpoint 文件读写锁（基于文件系统，支持跨进程同步）
type DeviceState struct {
	sync.Mutex
	cdi                      *CDIHandler
	tsManager                *TimeSlicingManager
	mpsManager               *MpsManager
	vfioPciManager           *VfioPciManager
	checkpointCleanupManager *CheckpointCleanupManager
	allocatable              AllocatableDevices // 扁平设备映射，用于按名称快速查找
	config                   *Config

	// perGPUAllocatable 与 allocatable 包含相同的设备集合，但按物理 GPU minor 号分组。
	// 用于 DynamicMIG 模式下生成 per-GPU ResourceSlice（每个物理 GPU 一个 Slice）。
	perGPUAllocatable PerGPUMinorAllocatableDevices

	nvdevlib          *deviceLib
	checkpointManager checkpointmanager.CheckpointManager

	// cplock 是 Checkpoint 文件的读写锁（基于文件系统，支持跨进程同步）。
	// 确保在插件升级（新旧进程同时运行）期间 Checkpoint 写操作的原子性。
	cplock *flock.Flock
}

// NewDeviceState 初始化 DeviceState，是设备发现和状态恢复的入口。
// 执行顺序：
//
//  1. 确定容器内驱动根路径（containerDriverRoot）和设备节点根路径（devRoot）
//  2. 初始化 NVML 设备库（newDeviceLib）
//  3. 枚举所有可分配设备（enumerateAllPossibleDevices）
//  4. 创建 CDI Handler 并预热设备 spec 缓存
//  5. 按特性门控初始化 TimeSlicing/MPS/VFIO 管理器
//  6. 创建 Checkpoint Manager 并加载已有 Checkpoint（或初始化空 Checkpoint）
//
// 参数：
//   - ctx:    上下文，用于控制生命周期
//   - config: 插件配置对象
//
// 返回值：
//   - *DeviceState: 初始化完成的设备状态实例
//   - error:        任一步骤失败则返回错误
func NewDeviceState(ctx context.Context, config *Config) (*DeviceState, error) {
	// 确定容器内 NVIDIA 驱动根路径
	containerDriverRoot := root(config.flags.containerDriverRoot)
	// 从驱动根路径中提取设备节点根路径（通常为 /dev）
	devRoot := containerDriverRoot.getDevRoot()
	klog.Infof("Using devRoot=%v", devRoot)

	// 初始化 NVML 设备库：加载 NVML 共享库、创建 NVML 会话
	nvdevlib, err := newDeviceLib(containerDriverRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to create device library: %w", err)
	}

	// 枚举节点上所有可分配设备（完整 GPU、MIG 设备、VFIO 设备等）
	allocatable, perGPUAllocatable, err := nvdevlib.enumerateAllPossibleDevices()
	if err != nil {
		return nil, fmt.Errorf("error enumerating all possible devices: %w", err)
	}

	// 主机上 NVIDIA 驱动根路径（用于生成 CDI spec 中的挂载源路径）
	hostDriverRoot := config.flags.hostDriverRoot

	// 配置 CDI 日志：仅当 klog 详细级别 >= 7 时才输出 CDI 库日志，
	// 否则静默丢弃以避免正常级别下的冗余输出
	cdilogger := logrus.New()
	if config.flags.klogVerbosity < 7 {
		klog.Infof("Muting CDI logger (verbosity is smaller 7: %d)", config.flags.klogVerbosity)
		cdilogger.SetOutput(io.Discard)
	}

	// 创建 CDI Handler：负责生成 CDI spec 文件和提供设备名称查询
	cdi, err := NewCDIHandler(
		WithNvml(nvdevlib.nvmllib),
		WithDeviceLib(nvdevlib),
		WithDriverRoot(string(containerDriverRoot)),
		WithDevRoot(devRoot),
		WithTargetDriverRoot(hostDriverRoot),
		WithNVIDIACDIHookPath(config.flags.nvidiaCDIHookPath),
		WithCDIRoot(config.flags.cdiRoot),
		WithLogger(cdilogger),
	)
	if err != nil {
		return nil, fmt.Errorf("unable to create CDI handler: %w", err)
	}

	// 收集所有完整 GPU 的 UUID，用于预热 CDI spec 缓存
	var fullGPUuuids []string
	for _, dev := range allocatable {
		if dev.Gpu != nil {
			fullGPUuuids = append(fullGPUuuids, dev.Gpu.UUID)
		}
	}
	// 预热 CDI 设备 spec 缓存：提前为所有完整 GPU 生成 CDI 设备条目，
	// 避免首次 NodePrepareResources 调用时因缓存未命中导致延迟飙升
	klog.V(2).Infof("Warming up CDI device spec cache for GPUs %v", fullGPUuuids)
	cdi.WarmupDevSpecCache(fullGPUuuids)

	// 按特性门控初始化 TimeSlicing 管理器
	var tsManager *TimeSlicingManager
	if featuregates.Enabled(featuregates.TimeSlicingSettings) {
		tsManager = NewTimeSlicingManager(nvdevlib)
	}

	// 按特性门控初始化 MPS 管理器
	var mpsManager *MpsManager
	if featuregates.Enabled(featuregates.MPSSupport) {
		mpsManager = NewMpsManager(config, nvdevlib, hostDriverRoot, MpsControlDaemonTemplatePath)
	}

	// 若 PassthroughSupport 特性门控启用，则验证直通支持的先决条件
	// （如 IOMMU、VFIO 驱动等）。验证失败将导致插件致命退出，
	// 因为后续 VFIO 设备配置依赖这些条件。
	var vfioPciManager *VfioPciManager
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		vfioPciManager = NewVfioPciManager(string(containerDriverRoot), string(hostDriverRoot), nvdevlib, true /* nvidiaEnabled */)
		if err := vfioPciManager.ValidatePassthroughSupport(); err != nil {
			klog.Fatalf("Failed to validate passthrough support: %v", err)
		}
	}

	// 创建 Checkpoint Manager，负责 Checkpoint 文件的持久化存储和读取
	checkpointManager, err := checkpointmanager.NewCheckpointManager(config.DriverPluginPath())
	if err != nil {
		return nil, fmt.Errorf("unable to create checkpoint manager: %v", err)
	}

	// Checkpoint 文件锁路径，与 Checkpoint 文件位于同一目录
	cpLockPath := filepath.Join(config.DriverPluginPath(), "cp.lock")

	// 构建 DeviceState 实例
	state := &DeviceState{
		cdi:               cdi,
		tsManager:         tsManager,
		mpsManager:        mpsManager,
		vfioPciManager:    vfioPciManager,
		allocatable:       allocatable,
		perGPUAllocatable: perGPUAllocatable,
		config:            config,
		nvdevlib:          nvdevlib,
		checkpointManager: checkpointManager,
		cplock:            flock.NewFlock(cpLockPath),
	}
	// 初始化 Checkpoint 清理管理器，传入自身和 ResourceClaim 客户端
	state.checkpointCleanupManager = NewCheckpointCleanupManager(state, config.clientsets.Resource)

	// 检查是否已有 Checkpoint 文件存在
	checkpoints, err := state.checkpointManager.ListCheckpoints()
	if err != nil {
		return nil, fmt.Errorf("unable to list checkpoints: %v", err)
	}

	// 若已有 Checkpoint 文件则直接加载，否则创建空 Checkpoint
	for _, c := range checkpoints {
		if c == DriverPluginCheckpointFileBasename {
			return state, nil
		}
	}

	// 首次启动：创建空的 Checkpoint 文件
	if err := state.createCheckpoint(ctx, &Checkpoint{}); err != nil {
		return nil, fmt.Errorf("unable to create fresh checkpoint: %v", err)
	}

	return state, nil
}

// Prepare 是 NodePrepareResources 的核心实现，负责为一个 ResourceClaim 准备 GPU 设备。
//
// 关键设计：
//   - 加锁（DeviceState.Mutex）保证单个 Claim 的准备过程不被并发修改状态
//   - 幂等：若 Claim 已处于 PrepareCompleted 状态，直接返回已准备的设备列表
//   - 两阶段 Checkpoint：先写 PrepareStarted（标记事务开始），成功后写 PrepareCompleted
//   - 重叠检测：拒绝将同一设备分配给多个非 admin Claim
//
// 执行顺序：
//  1. 读取 Checkpoint，检查 Claim 状态（幂等/重叠/残留回滚）
//  2. 写入 PrepareStarted Checkpoint（事务开始标记）
//  3. 调用 prepareDevices() 执行实际设备准备（MIG 创建、MPS/TimeSlicing 配置）
//  4. 写入 CDI Claim spec 文件
//  5. 写入 PrepareCompleted Checkpoint（事务提交）
//
// 参数：
//   - ctx:   上下文，用于传递取消信号
//   - claim: 待准备的 ResourceClaim 对象
//
// 返回值：
//   - []kubeletplugin.Device: 准备好的设备列表
//   - error:                 准备过程中的错误
func (s *DeviceState) Prepare(ctx context.Context, claim *resourceapi.ResourceClaim) ([]kubeletplugin.Device, error) {
	tplock0 := time.Now()
	// 获取内存互斥锁，防止并发修改 allocatable 等共享状态
	s.Lock()
	defer s.Unlock()
	klog.V(6).Infof("t_prep_state_lock_acq %.3f s", time.Since(tplock0).Seconds())

	// 获取 Claim UID 作为唯一标识
	claimUID := string(claim.UID)

	// 读取当前 Checkpoint 状态
	tgcp0 := time.Now()
	cp, err := s.getCheckpoint(ctx)
	if err != nil {
		return nil, fmt.Errorf("unable to get checkpoint: %v", err)
	}
	klog.V(7).Infof("t_prep_get_checkpoint %.3f s", time.Since(tgcp0).Seconds())

	// 在将 Checkpoint 状态更新为 PrepareStarted 之前，必须先检查该 Claim 是否已完成准备。
	// 若不先检查就写入 PrepareStarted，会把一个已经完整准备的 Claim 标记为"仅部分准备"，
	// 这可能在后续的 Unprepare() 操作中产生负面副作用（目前 Unprepare 对此情况是空操作：
	// "unprepare noop: claim preparation started but not completed"）。
	// Prepare() 必须是幂等的，因为 kubelet 可能对同一个 Claim 多次调用，
	// 但实际设备准备操作只应执行一次。
	preparedClaim, exists := cp.V2.PreparedClaims[claimUID]
	if exists && preparedClaim.CheckpointState == ClaimCheckpointStatePrepareCompleted {
		// 幂等返回：该 Claim 关联的设备已由本插件准备完毕，无需重复操作。
		klog.V(4).Infof("Skip prepare: claim already in PrepareCompleted state: %s", ResourceClaimToString(claim))
		return preparedClaim.PreparedDevices.GetDevices(), nil
	}

	// 在某些场景下，同一设备可能因调度器中不同 goroutine 之间的竞争条件、
	// 或者 Pod 被强制删除时 kubelet 仍认为设备已分配等原因，被为不同的 Claim 重复准备/分配。
	// 为了防止这种重叠分配，此处检查入站 Claim 请求的每个设备是否已被其他 Claim 准备过，
	// 若已被准备则拒绝该请求（除非先前的准备是以 admin 访问权限执行的）。
	// 更多细节参见：https://github.com/kubernetes/kubernetes/pull/136269
	if err := s.validateNoOverlappingPreparedDevices(cp, claim); err != nil {
		return nil, fmt.Errorf("unable to prepare claim %v: %w", claimUID, err)
	}

	// 与 DynamicMIG 相关：同一个 Claim 的前一次准备尝试可能导致了完整或部分的 GI/CI 创建。
	// 此处回滚那些部分创建的 MIG 设备，然后从零开始重试创建
	// （将来可以优化为"填补空缺"而非全部重建，但当前采用简单策略）。
	if exists && preparedClaim.CheckpointState == ClaimCheckpointStatePrepareStarted {
		klog.V(4).Infof("Claim %s already in PrepareStarted state: attempt rollback before new prepare", ResourceClaimToString(claim))
		if err := s.unpreparePartiallyPrepairedClaim(claimUID, preparedClaim, cp); err != nil {
			return nil, fmt.Errorf("unprepare failed for partially prepared claim %s failed: %w", PreparedClaimToString(&preparedClaim, claimUID), err)
		}
	}

	// 写入 PrepareStarted 状态到 Checkpoint，标记事务开始。
	// 即使后续步骤失败，此状态也会保留，确保重启后能识别并清理部分准备的状态。
	tucp0 := time.Now()
	err = s.updateCheckpoint(ctx, func(cp *Checkpoint) {
		cp.V2.PreparedClaims[claimUID] = PreparedClaim{
			CheckpointState: ClaimCheckpointStatePrepareStarted,
			Status:          claim.Status,
			Name:            claim.Name,
			Namespace:       claim.Namespace,
		}
	})
	if err != nil {
		return nil, fmt.Errorf("unable to update checkpoint: %w", err)
	}
	klog.V(6).Infof("t_prep_update_checkpoint %.3f s", time.Since(tucp0).Seconds())
	klog.V(6).Infof("checkpoint updated for claim %v", claimUID)

	// 执行核心设备准备逻辑
	tprep0 := time.Now()
	preparedDevices, err := s.prepareDevices(ctx, claim)
	klog.V(6).Infof("t_prep_core %.3f s (claim %s)", time.Since(tprep0).Seconds(), ResourceClaimToString(claim))
	if err != nil {
		return nil, fmt.Errorf("prepare devices failed: %w", err)
	}

	// 当 PassthroughSupport 特性启用时，准备完成后需要移除兄弟设备：
	// 如果一个 GPU 以 VFIO 方式分配给某个 Claim，则同一物理 GPU 的普通计算设备
	// 和其上的静态 MIG 设备都不应再被其他 Claim 使用，保证 GPU 的独占性。
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		for _, device := range preparedDevices.GetDevices() {
			allocatableDevice, ok := s.allocatable[device.DeviceName]
			if !ok {
				klog.Warningf("allocatable not found for device: %v", device.DeviceName)
				continue
			}
			// 此处移除兄弟设备是为了保证 VFIO 直通模式下物理 GPU 的独占性：
			// 当 GPU 以 VFIO 方式分配给一个 Claim 时，同一物理 GPU 的普通计算设备
			// 和其上的静态 MIG 设备都不应再被其他 Claim 使用。
			s.allocatable.RemoveSiblingDevices(allocatableDevice)
		}
	}

	// 为该 Claim 创建 CDI spec 文件，其中定义了容器运行时需要的设备注入规则
	tccsf0 := time.Now()
	if err := s.cdi.CreateClaimSpecFile(claimUID, preparedDevices); err != nil {
		return nil, fmt.Errorf("unable to create CDI spec file for claim: %w", err)
	}
	klog.V(7).Infof("t_prep_ccsf %.3f s", time.Since(tccsf0).Seconds())

	// 写入 PrepareCompleted 状态到 Checkpoint，标志事务成功提交。
	// 从此刻起，该 Claim 关联的设备被视为已完全准备好。
	tucp20 := time.Now()
	err = s.updateCheckpoint(ctx, func(cp *Checkpoint) {
		cp.V2.PreparedClaims[claimUID] = PreparedClaim{
			CheckpointState: ClaimCheckpointStatePrepareCompleted,
			Status:          claim.Status,
			PreparedDevices: preparedDevices,
		}
	})
	if err != nil {
		return nil, fmt.Errorf("unable to update checkpoint: %w", err)
	}
	klog.V(6).Infof("checkpoint updated for claim %v", claimUID)
	klog.V(7).Infof("t_prep_ucp2 %.3f s", time.Since(tucp20).Seconds())

	return preparedDevices.GetDevices(), nil
}

// DestroyUnknownMIGDevices 以节点本地 Checkpoint 作为唯一可信状态来源，拆除所有
// 不在已 PrepareCompleted 的 Claim 中引用的 MIG 设备。
//
// 该函数在插件启动时、接受 kubelet 请求之前调用一次。
// TODO：更好的机制是持有 Checkpoint 文件锁（在整个操作期间），以避免与并行的 Prepare/Unprepare
// 操作冲突（例如来自升级期间的独立插件进程）。注意该操作可能耗时较长。
//
// 需要观察在运行工作负载的 MIG 设备上执行拆除时的影响。
// 未来可能需要定期执行此类清理，而非仅在启动时运行一次。
//
// 运行时还有其他清理策略：(1) 定期清理部分准备但已过期的 Claim；(2) 在 Prepare() 路径中
// 对特定非过期 Claim 尝试回滚。尤其是当此方法也定期执行时，与策略 (1) 存在重叠——
// 但有决定性差异：
//
//   - 本方法 (0) 不修改 Checkpoint，仅拆除设备。策略 (1) 确认清理后从 Checkpoint 中移除条目。
//     (1) 也可能拆除本方法 (0) 会拆除的 MIG 设备：哪个先执行都可以。
//
//   - 策略 (2) 是 (1) 无法实现的，因为该 Claim 并非过期状态。前一次 Prepare() 尝试仅部分执行，
//     产生的状态变更可能导致立即重试的 Prepare() 失败（如"资源不足"）。此时 (2) 在重试的
//     Prepare() 调用的业务逻辑中执行主动回滚，帮助用户快速解决冲突。(0) 也可能有帮助但不及时。
//
// 所有清理策略都至关重要但也充满风险——设计或实现不当可能影响工作负载。
//
// 清理意义：
//
//  1. 修复越权创建的 MIG 设备（设计上禁止，但偶尔仍有人手动创建），系统自动恢复。
//  2. 从多步事务中的中断恢复（插件 bug、NVML 管理层问题、侵入性操作等）。
//     随时间推移，状态漂移的空间很大，尤其是在部分执行的事务后。
//
// 参数：
//   - ctx: 上下文
func (s *DeviceState) DestroyUnknownMIGDevices(ctx context.Context) {
	logpfx := "Destroy unknown MIG devices"

	// 读取当前 Checkpoint 状态
	cp, err := s.getCheckpoint(ctx)
	if err != nil {
		klog.Errorf("%s: unable to get checkpoint: %s", logpfx, err)
		return
	}

	// 从 Checkpoint 中筛选出处于 PrepareCompleted 状态的 Claim。
	// 只有这些 Claim 引用的 MIG 设备会被保留。
	// 此逻辑会拆除对应于 PrepareStarted 悬空状态的 MIG 设备——
	// 这仅在不存在重叠的 NodePrepareResources() 执行时才是正确的。
	filtered := make(PreparedClaimsByUIDV2)
	for uid, claim := range cp.V2.PreparedClaims {
		if claim.CheckpointState == ClaimCheckpointStatePrepareCompleted {
			filtered[uid] = claim
		}
	}

	// 收集所有预期存在的设备名称列表
	var expectedDeviceNames []DeviceName
	for _, cpclaim := range filtered {
		for _, res := range cpclaim.Status.Allocation.Devices.Results {
			expectedDeviceNames = append(expectedDeviceNames, res.Device)
		}
	}

	klog.Infof("%s: enter teardown routine (%d expect devices: %s)", logpfx, len(expectedDeviceNames), expectedDeviceNames)

	// 目前采用尽力而为策略：即使出错也不崩溃，继续执行。
	// TODO：可能需要增加超时控制。
	if err := s.nvdevlib.obliterateStaleMIGDevices(expectedDeviceNames); err != nil {
		klog.Errorf("%s: obliterateStaleMIGDevices failed: %s", logpfx, err)
	}

	klog.Infof("%s: done", logpfx)
}

// Unprepare 是 NodeUnprepareResources 的核心实现，负责释放 ResourceClaim 占用的设备。
//
// 幂等：若 Claim 不在 Checkpoint 中，则视为已 Unprepare，直接返回。
// 按 Claim 的 CheckpointState 分支处理：
//   - PrepareStarted：调用 unpreparePartiallyPrepairedClaim 回滚部分准备的 MIG 设备
//   - PrepareCompleted：调用 unprepareDevices 正常拆卸所有设备资源
//
// 成功后：删除 CDI spec 文件、从 Checkpoint 移除该 Claim 条目。
//
// 参数：
//   - ctx:      上下文
//   - claimRef: 待释放的 ResourceClaim 引用（包含命名空间、名称、UID）
//
// 返回值：
//   - error: 释放过程中的错误，nil 表示成功
func (s *DeviceState) Unprepare(ctx context.Context, claimRef kubeletplugin.NamespacedObject) error {
	s.Lock()
	defer s.Unlock()
	klog.V(6).Infof("Unprepare() for claim '%s'", claimRef.String())

	// 读取当前 Checkpoint 状态
	checkpoint, err := s.getCheckpoint(ctx)
	if err != nil {
		return fmt.Errorf("unable to get checkpoint: %v", err)
	}

	// 获取 Claim UID
	claimUID := string(claimRef.UID)

	// 查找该 Claim 在 Checkpoint 中的记录
	pc, exists := checkpoint.V2.PreparedClaims[claimUID]
	if !exists {
		// 不是错误：如果该 Claim UID 不在 Checkpoint 中，说明该设备从未被准备
		// 或已经被 Unprepare（假设 Prepare+Checkpoint 是事务性执行的）。
		// 注意 claimRef.String() 包含命名空间、名称和 UID。
		klog.V(2).Infof("Unprepare noop: claim not found in checkpoint data: %v", claimRef.String())
		return nil
	}

	// 根据 Checkpoint 状态分支处理
	switch pc.CheckpointState {
	case ClaimCheckpointStatePrepareStarted:
		// 仅部分准备：需要回滚已创建的 MIG 设备
		if err := s.unpreparePartiallyPrepairedClaim(claimUID, pc, checkpoint); err != nil {
			return fmt.Errorf("unprepare failed for partially prepared claim %s failed: %w", claimRef.String(), err)
		}
	case ClaimCheckpointStatePrepareCompleted:
		// 完全准备：执行正常的设备拆卸流程
		if err := s.unprepareDevices(ctx, claimUID, pc.PreparedDevices); err != nil {
			return fmt.Errorf("unprepare devices failed for claim %s: %w", claimRef.String(), err)
		}
	default:
		return fmt.Errorf("unsupported ClaimCheckpointState: %v", pc.CheckpointState)
	}

	// 当 PassthroughSupport 特性启用时，Unprepare 后需要重新发现兄弟设备：
	// 例如 GPU 从 VFIO 直通模式恢复为普通计算模式时，其对应的普通计算设备
	// 和静态 MIG 设备需要重新添加到可分配列表中
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		for _, device := range pc.PreparedDevices.GetDevices() {
			allocatableDevice, ok := s.allocatable[device.DeviceName]
			if !ok {
				klog.Warningf("allocatable not found for device: %v", device.DeviceName)
				continue
			}
			// 重新发现该设备的兄弟可分配设备（如 VFIO 对应的 GPU，或 GPU 对应的 VFIO）
			err := s.discoverSiblingAllocatables(allocatableDevice)
			if err != nil {
				return fmt.Errorf("error discovering sibling allocatables: %w", err)
			}
		}
	}

	// 删除该 Claim 的 CDI spec 文件。
	// TODO: 此处删除的是每个 Claim 的 CDI spec 文件，仅在正常路径中执行。
	// 常规运行中不会泄漏文件。但程序启动时或定期地，应清理 CDI spec 目录
	// 以防某些文件因异常未被删除而累积。
	if err := s.cdi.DeleteClaimSpecFile(claimUID); err != nil {
		// 仅记录错误——即使 CDI spec 文件删除失败，也要继续尝试
		// 从 Checkpoint 中移除该 Claim，确保设备释放流程不因非关键失败而中断。
		klog.Errorf("unable to delete CDI spec file for claim %s: %s", claimRef.String(), err)
	}

	// 从 Checkpoint 的 PreparedClaims 映射中移除该 Claim 条目（基于 Claim UID），
	// 标志着该 Claim 关联的所有设备已被完全 Unprepare。
	err = s.deleteClaimFromCheckpoint(ctx, claimRef)
	if err != nil {
		return fmt.Errorf("error deleting claim from checkpoint: %w", err)
	}
	return nil
}

// unpreparePartiallyPrepairedClaim 回滚之前部分执行的 MIG 设备创建操作
//（未通过将 Claim 状态转换为 PrepareCompleted 确认的操作）。
//
// 能否基于 Checkpoint 中的信息安全地回滚？由于未走完正常的创建/准备流程，
// 概念上无法使用特定的 MIG 设备 UUID 作为清理操作的输入。但精确的物理 MIG
// 设备配置是已知的——它编码在规范设备名中。因此我们有机会可靠地识别并拆除
// 孤立的 MIG 设备（由我们创建但从未分配工作负载的设备）。
//
// 关键：此清理方法绝不能误删其他设备。当前的正确做法是验证目前没有其他
// 处于 PrepareCompleted 状态的 Claim 引用了同一设备。
//
// 此处可用信息稀疏——例如 Checkpoint 中只有：
//
//	"checkpointState": "PrepareStarted", "status": {
//	  "allocation": { "devices": { "results": [
//	    { "request": "mig-1g", "driver": "gpu.nvidia.com",
//	      "pool": "gb-nvl-027-compute06", "device": "gpu-1-mig-2g47gb-14-0" }
//	  ]}}}
//
// 另外注意：插件重启时会在 DestroyUnknownMIGDevices()（在驱动接受请求前执行）
// 中拆除此类设备。
//
// 此方法在两种场景下被调用：(1) 已知过期的 Claim（不在 API Server 中）；
// (2) 非过期但当前正在准备的 Claim。两种情况下 `checkpoint` 数据足够新鲜，
// 不会有其他实体合法拥有 `pc` 中表示的设备。
//
// 参数：
//   - cuid:      Claim UID
//   - pc:        已准备 Claim 的 Checkpoint 记录
//   - checkpoint: 完整的 Checkpoint 对象，用于检查其他已准备 Claim 的设备引用
//
// 返回值：
//   - error: 回滚失败时返回错误
func (s *DeviceState) unpreparePartiallyPrepairedClaim(cuid string, pc PreparedClaim, checkpoint *Checkpoint) error {
	// 当 DynamicMIG 特性门控未启用时，部分准备的 Claim 不需要执行任何清理操作，
	// 因为没有 MIG 设备被动态创建。仅记录信息日志。
	if !featuregates.Enabled(featuregates.DynamicMIG) {
		klog.Infof("unprepare noop: preparation started but not completed for claim %s (devices: %v)", PreparedClaimToString(&pc, cuid), pc.Status.Allocation.Devices.Results)
	}

	// 当 DynamicMIG 启用时，尝试识别与 `pc` 对应的孤立 MIG 设备。
	// 为此，检查当前已完全准备的 Claim 使用了哪些设备，
	// 以避免误删仍在被有效使用的 MIG 设备。
	completedClaims := make(PreparedClaimsByUIDV2)
	for cuid, c := range checkpoint.V2.PreparedClaims {
		if c.CheckpointState == ClaimCheckpointStatePrepareCompleted {
			completedClaims[cuid] = c
		}
	}

	// 遍历该部分准备 Claim 中分配的每个设备，尝试拆除孤立的 MIG 设备
	for _, r := range pc.Status.Allocation.Devices.Results {
		devname := r.Device
		// 尝试将设备名解析为 MigSpecTuple（MIG 规范元组）
		ms, err := NewMigSpecTupleFromCanonicalName(devname)
		if err != nil {
			// 设备名称解析失败可能意味着这是一个普通的完整 GPU——
			// 在此情况下无需处理。为检测解析器错误（而非合法的非 MIG 设备名），
			// 仍然记录该错误。
			klog.V(6).Infof("Device name %s failed NewMigSpecTupleFromCanonicalName() parsing (assume this is not a MIG device): %s", devname, err)
			continue
		}

		klog.V(1).Infof("Device %s is a MIG device, DynamicMIG mode: deleteMigDevIfExistsAndNotUsedByCompletedClaim()", devname)
		// 尝试删除 MIG 设备：仅当该设备确实存在且未被已完成的 Claim 引用时才执行删除
		if err := s.deleteMigDevIfExistsAndNotUsedByCompletedClaim(ms, devname, completedClaims); err != nil {
			return fmt.Errorf("deleteMigDevIfExistsAndNotUsedByCompletedClaim failed: %w", err)
		}
	}

	return nil
}

// createCheckpoint 在 cplock 文件锁保护下创建一个新的 Checkpoint 文件。
// 此函数用于首次启动时创建空 Checkpoint。
//
// 参数：
//   - ctx: 上下文
//   - cp:  待持久化的 Checkpoint 对象
//
// 返回值：
//   - error: 获取锁失败或写入文件失败时返回错误
func (s *DeviceState) createCheckpoint(ctx context.Context, cp *Checkpoint) error {
	klog.V(6).Info("acquire cplock (create cp)")
	// 获取 Checkpoint 文件锁，10 秒超时
	release, err := s.cplock.Acquire(ctx, flock.WithTimeout(10*time.Second))
	if err != nil {
		return fmt.Errorf("error acquiring cplock: %w", err)
	}
	defer release()
	klog.V(7).Info("acquired cplock (createCheckpoint)")
	// 通过 Kubernetes CheckpointManager 将 Checkpoint 写入磁盘
	err = s.checkpointManager.CreateCheckpoint(DriverPluginCheckpointFileBasename, cp)
	klog.V(7).Info("create cp: done")
	return err
}

// getCheckpoint 在 cplock 文件锁保护下读取并返回当前 Checkpoint。
// 读取后会自动迁移到最新版本格式。
//
// 参数：
//   - ctx: 上下文
//
// 返回值：
//   - *Checkpoint: 最新版本的 Checkpoint 对象
//   - error:      获取锁失败或读取失败时返回错误
func (s *DeviceState) getCheckpoint(ctx context.Context) (*Checkpoint, error) {
	klog.V(7).Info("acquire cplock (getCheckpoint)")
	// 获取 Checkpoint 文件读锁
	release, err := s.cplock.Acquire(ctx, flock.WithTimeout(10*time.Second))
	if err != nil {
		return nil, fmt.Errorf("error acquiring cplock: %w", err)
	}
	defer release()
	klog.V(7).Info("acquired cplock (getCheckpoint)")

	// 从磁盘读取 Checkpoint 文件
	checkpoint := &Checkpoint{}
	if err := s.checkpointManager.GetCheckpoint(DriverPluginCheckpointFileBasename, checkpoint); err != nil {
		return nil, err
	}

	klog.V(7).Info("checkpoint read")
	// 将旧版本 Checkpoint 迁移到最新版本，并确保 PreparedClaims map 不为 nil
	return checkpoint.ToLatestVersion(), nil
}

// updateCheckpoint 执行 Checkpoint 的原子性"读-改-写"操作。
// 所有 Checkpoint 变更必须通过此函数，在 cplock 文件锁保护下进行，
// 防止多进程（插件升级场景）并发修改同一 Checkpoint 文件。
//
// 执行流程：
//  1. 获取 cplock 文件锁（10 秒超时）
//  2. 从磁盘读取最新 Checkpoint
//  3. 迁移到最新版本格式
//  4. 调用 mutate 回调函数执行状态变更
//  5. 将变更后的 Checkpoint 写回磁盘
//
// 参数：
//   - ctx:    上下文
//   - mutate: 状态变更回调函数，接收最新版 Checkpoint 作为参数
//
// 返回值：
//   - error: 获取锁失败、读取失败或写入失败时返回错误
func (s *DeviceState) updateCheckpoint(ctx context.Context, mutate func(*Checkpoint)) error {
	tucp0 := time.Now()
	klog.V(7).Info("acquire cplock (updateCheckpoint)")
	// 获取 Checkpoint 文件写锁
	release, err := s.cplock.Acquire(ctx, flock.WithTimeout(10*time.Second))
	if err != nil {
		return fmt.Errorf("error acquiring cplock: %w", err)
	}
	defer release()
	klog.V(7).Info("acquired cplock (updateCheckpoint)")

	// 从磁盘读取当前 Checkpoint
	checkpoint := &Checkpoint{}
	if err := s.checkpointManager.GetCheckpoint(DriverPluginCheckpointFileBasename, checkpoint); err != nil {
		return fmt.Errorf("updateCheckpoint: unable to get checkpoint: %w", err)
	}

	// 可能需要迁移到最新版本。此调用还会在 `PreparedClaims` 字段为 nil 时
	// 创建空的 map（确保后续插入操作安全），getCheckpoint() 辅助函数中也调用了此方法。
	cp := checkpoint.ToLatestVersion()
	// 执行调用方提供的状态变更操作
	mutate(cp)

	// 将变更后的 Checkpoint 写回磁盘
	err = s.checkpointManager.CreateCheckpoint(DriverPluginCheckpointFileBasename, cp)
	if err != nil {
		return fmt.Errorf("unable to create checkpoint: %w", err)
	}
	klog.V(6).Infof("t_checkpoint_update_total %.3f s", time.Since(tucp0).Seconds())
	return nil
}

// deleteClaimFromCheckpoint 从 Checkpoint 的 PreparedClaims 映射中删除指定 Claim。
// 此操作封装了 updateCheckpoint，在回调函数中执行 delete 操作。
//
// 参数：
//   - ctx:      上下文
//   - claimRef: 待删除的 ResourceClaim 引用
//
// 返回值：
//   - error: 更新 Checkpoint 失败时返回错误
func (s *DeviceState) deleteClaimFromCheckpoint(ctx context.Context, claimRef kubeletplugin.NamespacedObject) error {
	err := s.updateCheckpoint(ctx, func(cp *Checkpoint) {
		// 从 PreparedClaims 映射中删除该 Claim UID 对应的条目
		delete(cp.V2.PreparedClaims, string(claimRef.UID))
	})
	if err != nil {
		return fmt.Errorf("unable to update checkpoint: %w", err)
	}
	klog.V(6).Infof("Deleted claim from checkpoint: %s", claimRef.String())
	return nil
}

// prepareDevices 是设备准备的核心逻辑，将 ResourceClaim 中的分配结果映射为实际设备操作。
//
// 执行流程：
//  1. 解码 OpaqueDeviceConfig（从 DeviceClass 和 Claim 中按优先级顺序合并）
//  2. 在配置列表最前面插入默认的 GPU、MIG 和 VFIO 设备配置（优先级最低）
//  3. 将每个分配结果（result）匹配到最高优先级的 Config
//  4. 对每个 Config 及其关联设备组执行：Normalize → Validate → applyConfig
//     - applyConfig 会按 Config 类型处理 TimeSlicing、MPS、VFIO 等配置
//  5. 对 DynamicMIG 设备：此时调用 createMigDevice() 创建实际 MIG 实例
//  6. 构建 PreparedDevices 结构体（含 CDI 设备名称、设备类型信息等）
//
// 参数：
//   - ctx:   上下文
//   - claim: 待准备的 ResourceClaim 对象
//
// 返回值：
//   - PreparedDevices: 准备好的设备组列表
//   - error:           任一步骤失败时返回错误
func (s *DeviceState) prepareDevices(ctx context.Context, claim *resourceapi.ResourceClaim) (PreparedDevices, error) {
	// 校验 Claim 是否已被调度器分配
	if claim.Status.Allocation == nil {
		return nil, fmt.Errorf("claim not yet allocated")
	}

	klog.V(6).Infof("Preparing devices for claim %s", ResourceClaimToString(claim))

	// 从 Claim 的分配状态中提取当前驱动的所有 OpaqueDeviceConfig。
	// 这些配置来源于 DeviceClass 和 ResourceClaim，经调度器合并后按优先级排列。
	configs, err := GetOpaqueDeviceConfigs(
		configapi.StrictDecoder,
		DriverName,
		claim.Status.Allocation.Devices.Config,
	)
	if err != nil {
		return nil, fmt.Errorf("error getting opaque device configs: %v", err)
	}

	// 在配置列表最前面插入默认的 GPU、MIG 和 VFIO 设备配置，优先级最低。
	// 保证列表中至少各有一个 len(Requests)==0 的默认配置，在 prepareDevices() 的
	// 后续查找中，若无更高优先级的配置匹配某个请求，将使用这些默认配置作为兜底。
	configs = slices.Insert(configs, 0, &OpaqueDeviceConfig{
		Requests: []string{},
		Config:   configapi.DefaultGpuConfig(),
	})
	configs = slices.Insert(configs, 0, &OpaqueDeviceConfig{
		Requests: []string{},
		Config:   configapi.DefaultMigDeviceConfig(),
	})
	// 当 PassthroughSupport 特性启用时，也插入默认的 VFIO 设备配置
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		configs = slices.Insert(configs, 0, &OpaqueDeviceConfig{
			Requests: []string{},
			Config:   configapi.DefaultVfioDeviceConfig(),
		})
	}

	// 遍历配置列表，为每个分配结果确定其适用的最高优先级配置。
	// 使用 slices.Backward 倒序遍历，因为列表靠后的配置优先级更高，
	// 找到第一个匹配的配置即停止搜索。
	configResultsMap := make(map[runtime.Object][]*resourceapi.DeviceRequestAllocationResult)
	for _, result := range claim.Status.Allocation.Devices.Results {
		// 跳过非本驱动的分配结果
		if result.Driver != DriverName {
			continue
		}
		// 查找设备是否存在于可分配列表中
		device, exists := s.allocatable[result.Device]
		if !exists {
			return nil, fmt.Errorf("requested device is not allocatable: %v", result.Device)
		}
		// 仅当设备健康时才继续配置映射。不健康的设备不应被分配给工作负载。
		if featuregates.Enabled(featuregates.NVMLDeviceHealthCheck) {
			if !device.IsHealthy() {
				return nil, fmt.Errorf("requested device is not healthy: %v", result.Device)
			}
		}
		// 倒序遍历配置列表，找到第一个匹配请求名称的配置
		for _, c := range slices.Backward(configs) {
			if slices.Contains(c.Requests, result.Request) {
				// 类型校验：确保配置类型与设备类型兼容
				if _, ok := c.Config.(*configapi.GpuConfig); ok && device.Type() != GpuDeviceType {
					return nil, fmt.Errorf("cannot apply GpuConfig to device type %s (request: %v)", device.Type(), result.Request)
				}

				if _, ok := c.Config.(*configapi.MigDeviceConfig); ok && !device.IsStaticOrDynMigDevice() {
					return nil, fmt.Errorf("cannot apply MigDeviceConfig to device type %s (request: %v)", device.Type(), result.Request)
				}

				if _, ok := c.Config.(*configapi.VfioDeviceConfig); ok && device.Type() != VfioDeviceType {
					return nil, fmt.Errorf("cannot apply VfioDeviceConfig to device type %s (request: %v)", device.Type(), result.Request)
				}
				// 将该分配结果归入对应配置的列表
				configResultsMap[c.Config] = append(configResultsMap[c.Config], &result)
				break
			}
			// 默认配置（空 Requests 列表）的处理
			if len(c.Requests) == 0 {
				// 类型不匹配时跳过，继续搜索下一个配置
				if _, ok := c.Config.(*configapi.GpuConfig); ok && device.Type() != GpuDeviceType {
					continue
				}
				if _, ok := c.Config.(*configapi.MigDeviceConfig); ok && !device.IsStaticOrDynMigDevice() {
					continue
				}
				if _, ok := c.Config.(*configapi.VfioDeviceConfig); ok && device.Type() != VfioDeviceType {
					continue
				}
				// 类型匹配的默认配置：将分配结果归入，并停止搜索
				configResultsMap[c.Config] = append(configResultsMap[c.Config], &result)
				break
			}
		}
	}

	// 对每个配置及其关联的分配结果，依次执行规范化、验证和应用操作。
	// 跟踪每个设备组的配置状态（如 MPS 守护进程 ID、CDI 编辑项等）。
	preparedDeviceGroupConfigState := make(map[runtime.Object]*DeviceConfigState)
	for c, results := range configResultsMap {
		// 将运行时对象转换为 configapi.Interface 接口类型，以便调用规范化、验证和应用方法
		var config configapi.Interface
		switch castConfig := c.(type) {
		case *configapi.GpuConfig:
			config = castConfig
		case *configapi.MigDeviceConfig:
			config = castConfig
		case *configapi.VfioDeviceConfig:
			config = castConfig
		default:
			return nil, fmt.Errorf("runtime object is not a recognized configuration")
		}

		// 规范化配置：设置隐含的默认值，确保配置完整。
		if err := config.Normalize(); err != nil {
			return nil, fmt.Errorf("error normalizing GPU config: %w", err)
		}

		// 验证配置：检查配置的完整性和一致性，防止无效配置导致运行时错误。
		if err := config.Validate(); err != nil {
			return nil, fmt.Errorf("error validating GPU config: %w", err)
		}

		// 应用配置：根据配置类型执行实际的设备准备操作（时间切片、MPS、VFIO 等）。
		// 注意：如果配置应用于 DynamicMIG 设备，此时该设备尚未创建（MIG UUID 未知）。
		configState, err := s.applyConfig(ctx, config, claim, results)
		if err != nil {
			return nil, fmt.Errorf("error applying config: %w", err)
		}

		// 将设备组的配置状态保存到映射中，后续构建 PreparedDeviceGroup 时使用。
		preparedDeviceGroupConfigState[c] = configState
	}

	// 遍历每个配置及其关联的分配结果，构建要返回的已准备设备列表
	var preparedDevices PreparedDevices
	for c, results := range configResultsMap {
		// 构建设备组，包含该组共享的配置状态
		preparedDeviceGroup := PreparedDeviceGroup{
			ConfigState: *preparedDeviceGroupConfigState[c],
		}

		for _, result := range results {
			cdiDevices := []string{}
			// 此时 Claim 专属的 CDI spec（kind 为 `k8s.gpu.nvidia.com/claim`）尚未生成，
			// 但我们已经知道它将枚举的 ClaimDevice 条目名称（按命名约定）。
			if d := s.cdi.GetClaimDeviceName(string(claim.UID), s.allocatable[result.Device], preparedDeviceGroupConfigState[c].containerEdits); d != "" {
				cdiDevices = append(cdiDevices, d)
			}

			// 构建基础设备对象
			device := &kubeletplugin.Device{
				Requests:     []string{result.Request},
				PoolName:     result.Pool,
				DeviceName:   result.Device,
				CDIDeviceIDs: cdiDevices,
			}

			// 根据设备类型构建对应的已准备设备对象
			adev := s.allocatable[result.Device]
			var preparedDevice PreparedDevice

			switch adev.Type() {
			case GpuDeviceType:
				// 完整物理 GPU：无需额外创建操作
				preparedDevice.Gpu = &PreparedGpu{
					Info:   adev.Gpu,
					Device: device,
				}
			case MigStaticDeviceType:
				// 静态 MIG 设备：已由管理员预先创建
				preparedDevice.Mig = &PreparedMigDevice{
					Concrete: adev.MigStatic.LiveTuple(),
					Device:   device,
				}
			case MigDynamicDeviceType:
				// 动态 MIG 设备：需要通过 NVML API 动态创建 MIG 实例
				migspec := adev.MigDynamic
				// 注意：createMigDevice() 返回后，我们可以立即将有用数据持久化到磁盘
				// （如 MIG 设备 UUID），这些数据有助于更可靠地清理部分准备的状态
				tcmig0 := time.Now()
				migdev, err := s.nvdevlib.createMigDevice(migspec)
				klog.V(6).Infof("t_prep_create_mig_dev %.3f s (claim %s)", time.Since(tcmig0).Seconds(), ResourceClaimToString(claim))
				if err != nil {
					return nil, fmt.Errorf("error creating MIG device: %w", err)
				}
				preparedDevice.Mig = &PreparedMigDevice{
					Concrete: migdev.LiveTuple(),
					Device:   device,
				}
			case VfioDeviceType:
				// VFIO 直通设备：已由 VfioPciManager 配置
				preparedDevice.Vfio = &PreparedVfioDevice{
					Info:   s.allocatable[result.Device].Vfio,
					Device: device,
				}
			}

			klog.V(6).Infof("Prepared device for claim '%s': %s", ResourceClaimToString(claim), device.DeviceName)
			// 此处是更新 Checkpoint 的独特时机，可以反映设备准备的当前状态
			// （仍处于 PrepareStarted 状态，但比之前有更多细节）。在此处与最终
			// Claim 准备完成之间，存在崩溃和提前终止的可能性。
			preparedDeviceGroup.Devices = append(preparedDeviceGroup.Devices, preparedDevice)
		}

		preparedDevices = append(preparedDevices, &preparedDeviceGroup)
	}

	return preparedDevices, nil
}

// unprepareDevices 拆卸一个 Claim 的所有已准备设备，是 Unprepare 正常路径的实现。
// 按设备组逐步清理：
//
//  1. VFIO 设备：调用 VfioPciManager.Unconfigure 将 GPU 切回 nvidia 驱动
//  2. MIG 设备（DynamicMIG）：通过 NVML 销毁 MIG GI/CI 实例
//  3. MPS：停止并删除 MPS 控制守护进程及其工作目录
//  4. TimeSlicing：将 GPU 时间切片恢复为默认配置
//
// 参数：
//   - ctx:      上下文
//   - claimUID: Claim 的唯一标识符
//   - devices:  之前已准备的设备组列表
//
// 返回值：
//   - error: 任一步骤失败时返回错误
func (s *DeviceState) unprepareDevices(ctx context.Context, claimUID string, devices PreparedDevices) error {
	klog.V(6).Infof("Unpreparing claim '%s', previously prepared devices from checkpoint: %v", claimUID, devices.GetDeviceNames())
	for _, group := range devices {
		// 若 PassthroughSupport 特性门控启用，则先 Unconfigure 所有 VFIO 设备，
		// 将 GPU 从 vfio-pci 驱动切换回 nvidia 驱动
		if featuregates.Enabled(featuregates.PassthroughSupport) {
			err := s.unprepareVfioDevices(ctx, group.Devices.VfioDevices())
			if err != nil {
				return err
			}
		}

		// 遍历设备组中的每个设备，根据类型执行对应的拆卸操作
		// TODO: 是否应在 MPS/TimeSlicing 原始拆卸之后执行此操作？
		for _, device := range group.Devices {
			switch device.Type() {
			case GpuDeviceType:
				// 完整物理 GPU：无需额外拆卸操作
				klog.V(4).Infof("Unprepare: regular GPU: noop (GPU %s)", device.Gpu.Info.String())
			case PreparedMigDeviceType:
				if featuregates.Enabled(featuregates.DynamicMIG) {
					// 动态 MIG 设备：通过 NVML API 销毁 GI 和 CI
					mig := device.Mig.Concrete
					klog.V(4).Infof("Unprepare: tear down MIG device '%s' for claim '%s'", mig.MigUUID, claimUID)
					// MIG 设备删除过程中的错误通常很少见但必须被预期到，
					// 并且应该导致 NodeUnprepareResources() 操作失败。
					// 这可能是 'error destroying GPU Instance: In use by another client'
					// 并会在冲突方退出后自行解决。记录显式警告，除了返回错误外。
					err := s.nvdevlib.deleteMigDevice(mig)
					if err != nil {
						klog.Warningf("Error deleting MIG device %s: %s", device.Mig.Device.DeviceName, err)
						return fmt.Errorf("error deleting MIG device %s: %w", device.Mig.Device.DeviceName, err)
					}
				} else {
					// 静态 MIG 设备：由管理员管理，无需插件操作
					klog.V(4).Infof("Unprepare: static MIG: noop (MIG %s)", device.Mig.Concrete.MigUUID)
				}
			}
		}

		// 若 MPSSupport 特性门控启用，停止为每组已准备设备启动的 MPS 控制守护进程，
		// 释放关联的共享内存和管道资源
		if featuregates.Enabled(featuregates.MPSSupport) {
			mpsControlDaemon := s.mpsManager.NewMpsControlDaemon(claimUID, group)
			if err := mpsControlDaemon.Stop(ctx); err != nil {
				return fmt.Errorf("error stopping MPS control daemon: %w", err)
			}
		}

		// 若 TimeSlicingSettings 特性门控启用，将所有完整 GPU 的时间切片恢复为默认配置，
		// 避免上一个 Claim 的自定义时间切片设置影响后续使用同一 GPU 的工作负载
		if featuregates.Enabled(featuregates.TimeSlicingSettings) {
			tsc := configapi.DefaultGpuConfig().Sharing.TimeSlicingConfig
			if err := s.tsManager.SetTimeSlice(group.Devices.GpuUUIDs(), tsc); err != nil {
				return fmt.Errorf("error setting timeslice for devices: %w", err)
			}
		}

	}
	return nil
}

// getAllocatableVfioDevice 根据 UUID 在可分配设备列表中查找对应的 VFIO 设备。
//
// 参数：
//   - uuid: VFIO 设备的 UUID
//
// 返回值：
//   - *AllocatableDevice: 找到的可分配设备对象
//   - error:              未找到时返回错误
func (s *DeviceState) getAllocatableVfioDevice(uuid string) (*AllocatableDevice, error) {
	for _, allocatable := range s.allocatable {
		// 跳过非 VFIO 类型的设备
		if allocatable.Type() != VfioDeviceType {
			continue
		}
		// 匹配 UUID
		if allocatable.Vfio.UUID == uuid {
			return allocatable, nil
		}
	}
	return nil, fmt.Errorf("allocatable device not found for vfio device: %v", uuid)
}

// unprepareVfioDevices 拆卸一组 VFIO 设备，将每个设备从 vfio-pci 驱动切回 nvidia 驱动。
//
// 参数：
//   - ctx:     上下文
//   - devices: 待拆卸的 VFIO 设备列表
//
// 返回值：
//   - error: 查找设备或 Unconfigure 失败时返回错误
func (s *DeviceState) unprepareVfioDevices(ctx context.Context, devices PreparedDeviceList) error {
	for _, device := range devices {
		// 查找 VFIO 设备对应的可分配设备
		vfioAllocatable, err := s.getAllocatableVfioDevice(device.Vfio.Info.UUID)
		if err != nil {
			return fmt.Errorf("error getting allocatable device for vfio device: %w", err)
		}
		// 调用 VfioPciManager 将 GPU 从 vfio-pci 驱动切回 nvidia 驱动
		if err := s.vfioPciManager.Unconfigure(ctx, vfioAllocatable.Vfio); err != nil {
			return fmt.Errorf("error unconfiguring vfio device: %w", err)
		}
	}
	return nil
}

// discoverSiblingAllocatables 发现指定设备的兄弟可分配设备。
// 在 VFIO 直通模式下，GPU 有两种表示形式：普通计算设备（GpuDeviceType）和 VFIO 直通设备（VfioDeviceType）。
// 当一个 Claim 准备了某种类型的设备后，需要移除另一种类型的兄弟设备以保证独占性；
// 而当 Claim 释放后，需要重新发现被移除的兄弟设备以恢复可分配性。
//
// 参数：
//   - device: 需要发现兄弟设备的可分配设备对象
//
// 返回值：
//   - error: 发现过程中出现的错误
//
// 处理逻辑按设备类型分支：
//   - GpuDeviceType: 若启用了 vfioEnabled，则发现对应的 VFIO 设备
//   - VfioDeviceType: 发现对应的完整 GPU 及其上的静态 MIG 设备
//   - MigStaticDeviceType / MigDynamicDeviceType: 待后续实现
func (s *DeviceState) discoverSiblingAllocatables(device *AllocatableDevice) error {
	switch device.Type() {
	case GpuDeviceType:
		// 若该 GPU 未启用 VFIO 直通，则无需发现兄弟设备
		if !device.Gpu.vfioEnabled {
			return nil
		}
		// 发现对应的 VFIO 直通设备并添加到可分配列表
		vfio, err := s.nvdevlib.discoverVfioDevice(device.Gpu)
		if err != nil {
			return fmt.Errorf("error discovering vfio device: %w", err)
		}
		s.allocatable[vfio.CanonicalName()] = vfio
	case VfioDeviceType:
		// 发现对应的完整 GPU 设备
		gpu, migs, err := s.nvdevlib.discoverGPUByPCIBusID(device.Vfio.pcieBusID)
		if err != nil {
			return fmt.Errorf("error discovering gpu by pci bus id: %w", err)
		}
		// 将完整 GPU 添加到可分配列表
		s.allocatable[gpu.CanonicalName()] = gpu
		// 更新 VFIO 设备的父 GPU 引用
		device.Vfio.parent = gpu.Gpu
		// 将该 GPU 上的所有静态 MIG 设备也添加到可分配列表
		for _, mig := range migs {
			s.allocatable[mig.CanonicalName()] = mig
		}
	case MigStaticDeviceType:
		// TODO: 待 DynamicMIG 功能完善后实现静态 MIG 设备的兄弟设备发现逻辑。
		return nil
	case MigDynamicDeviceType:
		// TODO: 待 DynamicMIG 功能完善后实现动态 MIG 设备的兄弟设备发现逻辑。
		return nil
	}
	return nil
}

// applyConfig 根据配置类型将设备配置应用到给定的 Claim 请求上。
// 配置类型分为三种：
//   - GpuConfig / MigDeviceConfig：调用 applySharingConfig 处理共享配置（TimeSlicing、MPS）
//   - VfioDeviceConfig：调用 applyVfioDeviceConfig 处理 VFIO 直通配置
//
// 参数：
//   - ctx:    上下文
//   - config: 已解码的配置接口对象
//   - claim:  当前 ResourceClaim
//   - results: 该配置适用的分配结果列表
//
// 返回值：
//   - *DeviceConfigState: 配置应用后的状态（如 MPS 守护进程 ID、CDI 编辑项等）
//   - error:              配置应用失败时返回错误
func (s *DeviceState) applyConfig(ctx context.Context, config configapi.Interface, claim *resourceapi.ResourceClaim, results []*resourceapi.DeviceRequestAllocationResult) (*DeviceConfigState, error) {
	switch castConfig := config.(type) {
	case *configapi.GpuConfig:
		klog.V(7).Infof("applySharingConfig() for GpuConfig")
		return s.applySharingConfig(ctx, castConfig.Sharing, claim, results)
	case *configapi.MigDeviceConfig:
		klog.V(7).Infof("applySharingConfig() for MigDeviceConfig")
		return s.applySharingConfig(ctx, castConfig.Sharing, claim, results)
	case *configapi.VfioDeviceConfig:
		klog.V(7).Infof("applySharingConfig() for VfioDeviceConfig")
		return s.applyVfioDeviceConfig(ctx, castConfig, claim, results)
	default:
		return nil, fmt.Errorf("unknown config type: %T", castConfig)
	}
}

// applySharingConfig 应用共享配置（TimeSlicing 和 MPS），是 GpuConfig 和 MigDeviceConfig 的核心处理逻辑。
//
// 执行步骤：
//  1. 收集该配置正在应用的请求名称列表和对应的可分配设备映射
//  2. 若配置包含 TimeSlicing 设置且特性门控已启用：
//     - 获取需要修改时间切片设置的物理 GPU UUID 列表（仅完整 GPU，MIG 设备不支持时间切片）
//     - 调用 TimeSlicingManager 设置对应 GPU 的时间切片值
//  3. 若配置包含 MPS 设置且特性门控已启用：
//     - 检查与 DynamicMIG 的兼容性（当前不兼容）
//     - 创建并启动 MPS 控制守护进程
//     - 等待守护进程就绪后获取其 ID 和 CDI 编辑项
//
// 参数：
//   - ctx:    上下文
//   - config: 共享配置对象
//   - claim:  当前 ResourceClaim
//   - results: 该配置适用的分配结果列表
//
// 返回值：
//   - *DeviceConfigState: 配置应用后的状态对象
//   - error:              配置应用失败时返回错误
func (s *DeviceState) applySharingConfig(ctx context.Context, config configapi.Sharing, claim *resourceapi.ResourceClaim, results []*resourceapi.DeviceRequestAllocationResult) (*DeviceConfigState, error) {
	// 收集该配置正在应用的 Claim 请求名称列表
	var requests []string
	for _, r := range results {
		requests = append(requests, r.Request)
	}

	// 收集该配置正在应用的可分配设备映射，便于后续按设备类型筛选
	requestedDevices := make(AllocatableDevices)
	for _, r := range results {
		requestedDevices[r.Device] = s.allocatable[r.Device]
	}

	// 声明一个设备组状态对象，用于在后续步骤中填充
	var configState DeviceConfigState

	// 应用时间切片设置（如果配置中包含且特性门控已启用）
	if featuregates.Enabled(featuregates.TimeSlicingSettings) && config.IsTimeSlicing() {
		tsc, err := config.GetTimeSlicingConfig()
		if err != nil {
			return nil, fmt.Errorf("error getting timeslice config for requests '%v' in claim '%v': %w", requests, claim.UID, err)
		}
		if tsc != nil {
			// 获取需要修改时间切片设置的物理 GPU UUID 列表。
			// 仅对完整 GPU 设置时间切片，MIG 设备不支持直接设置时间切片。
			// 因此 API 不允许在 MigDeviceConfig 中设置 timeSlicingConfig。
			// TODO: 是否应为直通设备也设置时间切片？
			uuids := requestedDevices.GpuUUIDs()
			klog.V(6).Infof("SetTimeSlice() for full GPUs with UUIDs: %s", uuids)
			err = s.tsManager.SetTimeSlice(uuids, tsc)
			if err != nil {
				return nil, fmt.Errorf("error setting timeslice config for requests '%v' in claim '%v': %w", requests, claim.UID, err)
			}
		}
	}

	// 应用 MPS 设置（如果配置中包含且特性门控已启用）。
	// MPS 共享需要启动控制守护进程，并等待其就绪后方可使用。
	if featuregates.Enabled(featuregates.MPSSupport) && config.IsMps() {
		if featuregates.Enabled(featuregates.DynamicMIG) {
			// TODO: 当前 MPS 与 DynamicMIG 不兼容，需要先创建 MIG 设备获取其 UUID，
			// 然后再为该设备启用 MPS——可能需要基于 PreparedDevicesList 而非 AllocatableDevices 来实现。
			return nil, fmt.Errorf("MPS is not yet supported when using featureGates.DynamicMIG=true")
		}
		mpsc, err := config.GetMpsConfig()
		if err != nil {
			return nil, fmt.Errorf("error getting MPS configuration: %w", err)
		}
		// 为该 Claim 创建新的 MPS 控制守护进程实例
		mpsControlDaemon := s.mpsManager.NewMpsControlDaemon(string(claim.UID), requestedDevices)
		// 启动守护进程，包括创建共享内存和命名管道等
		if err := mpsControlDaemon.Start(ctx, mpsc); err != nil {
			return nil, fmt.Errorf("error starting MPS control daemon: %w", err)
		}
		// 等待守护进程就绪（确保共享内存和管道已准备好接受客户端连接）
		if err := mpsControlDaemon.AssertReady(ctx); err != nil {
			return nil, fmt.Errorf("MPS control daemon is not yet ready: %w", err)
		}
		// 保存守护进程 ID 和 CDI 编辑项，后续构建 PreparedDeviceGroup 时使用
		configState.MpsControlDaemonID = mpsControlDaemon.GetID()
		configState.containerEdits = mpsControlDaemon.GetCDIContainerEdits()
	}

	return &configState, nil
}

// applyVfioDeviceConfig 应用 VFIO 设备配置，为每个分配结果中的 VFIO 设备执行 PCI 设备解绑、
// VFIO 驱动绑定和电源管理等操作。
//
// 参数：
//   - ctx:    上下文
//   - config: VFIO 设备配置对象
//   - claim:  当前 ResourceClaim
//   - results: 该配置适用的分配结果列表
//
// 返回值：
//   - *DeviceConfigState: 配置应用后的状态对象（当前为空对象）
//   - error:              配置应用失败时返回错误
func (s *DeviceState) applyVfioDeviceConfig(ctx context.Context, config *configapi.VfioDeviceConfig, claim *resourceapi.ResourceClaim, results []*resourceapi.DeviceRequestAllocationResult) (*DeviceConfigState, error) {
	// 若 PassthroughSupport 特性门控未启用，直接返回空结果
	if !featuregates.Enabled(featuregates.PassthroughSupport) {
		return nil, nil
	}
	var configState DeviceConfigState

	// 为每个分配结果中的 VFIO 设备调用 Configure，
	// 执行 PCI 设备解绑、VFIO 驱动绑定、电源管理等操作。
	for _, r := range results {
		info := s.allocatable[r.Device]
		err := s.vfioPciManager.Configure(ctx, info.Vfio)
		if err != nil {
			return nil, err
		}
	}

	return &configState, nil
}

// GetOpaqueDeviceConfigs 从 possibleConfigs 中提取并解码当前驱动（DriverName）的配置列表。
//
// 优先级规则（从低到高）：
//   1. DeviceClass 中的配置（classConfigs）
//   2. ResourceClaim 中的配置（claimConfigs）
//   3. 同一来源中，列表靠后的配置优先级更高
//
// 返回的列表按优先级从低到高排列，prepareDevices() 中通过 slices.Backward 倒序遍历以取最高优先级。
// 若无任何匹配配置，返回 nil。
//
// 参数：
//   - decoder:         运行时解码器，用于将 Raw 字节反序列化为具体的配置对象
//   - driverName:      当前驱动名称，用于过滤仅属于本驱动的配置
//   - possibleConfigs: 可能的设备分配配置列表（来自 DeviceClass 和 ResourceClaim）
//
// 返回值：
//   - []*OpaqueDeviceConfig: 按优先级排序的已解码配置列表
//   - error:                  解码失败或遇到无效配置来源时返回错误
func GetOpaqueDeviceConfigs(
	decoder runtime.Decoder,
	driverName string,
	possibleConfigs []resourceapi.DeviceAllocationConfiguration,
) ([]*OpaqueDeviceConfig, error) {
	// 按配置来源收集配置，并按反向优先级排列：classConfigs 优先级最低，claimConfigs 优先级最高。
	var classConfigs []resourceapi.DeviceAllocationConfiguration
	var claimConfigs []resourceapi.DeviceAllocationConfiguration
	var candidateConfigs []resourceapi.DeviceAllocationConfiguration
	for _, config := range possibleConfigs {
		switch config.Source {
		case resourceapi.AllocationConfigSourceClass:
			// 来自 DeviceClass 的配置，优先级最低
			classConfigs = append(classConfigs, config)
		case resourceapi.AllocationConfigSourceClaim:
			// 来自 ResourceClaim 的配置，优先级最高
			claimConfigs = append(claimConfigs, config)
		default:
			return nil, fmt.Errorf("invalid config source: %v", config.Source)
		}
	}
	// 合并两个来源的配置：先添加低优先级的 classConfigs，再添加高优先级的 claimConfigs
	candidateConfigs = append(candidateConfigs, classConfigs...)
	candidateConfigs = append(candidateConfigs, claimConfigs...)

	// 解码所有与当前驱动相关的配置，跳过其他驱动的配置
	var resultConfigs []*OpaqueDeviceConfig
	for _, config := range candidateConfigs {
		// 如果 Opaque 字段为 nil，说明驱动不支持某个未来的 API 扩展，需要更新驱动代码以适配
		if config.Opaque == nil {
			return nil, fmt.Errorf("only opaque parameters are supported by this driver")
		}

		// 单个请求可能由不同驱动满足，因此配置中可能包含其他驱动的配置。
		// 这不是错误——驱动必须跳过其他驱动的配置以支持此场景。
		if config.Opaque.Driver != driverName {
			continue
		}

		// 使用运行时解码器将 Raw 字节反序列化为具体的配置对象
		decodedConfig, err := runtime.Decode(decoder, config.Opaque.Parameters.Raw)
		if err != nil {
			return nil, fmt.Errorf("error decoding config parameters: %w", err)
		}

		resultConfig := &OpaqueDeviceConfig{
			Requests: config.Requests,
			Config:   decodedConfig,
		}

		resultConfigs = append(resultConfigs, resultConfig)
	}

	return resultConfigs, nil
}

// UpdateDeviceHealthStatus 更新指定设备的健康状态。
// 调用方需确保在持有 DeviceState.Mutex 锁的情况下调用此方法，
// 或在已获取锁的上下文中调用（如 deviceHealthEvents goroutine 中）。
//
// 参数：
//   - d:  待更新的可分配设备对象
//   - hs: 新的健康状态（Healthy 或 Unhealthy）
func (s *DeviceState) UpdateDeviceHealthStatus(d *AllocatableDevice, hs HealthStatus) {
	s.Lock()
	defer s.Unlock()

	switch d.Type() {
	case GpuDeviceType:
		// 完整物理 GPU：直接更新其 health 字段
		d.Gpu.health = hs
	case MigDynamicDeviceType:
		// 抽象动态 MIG 设备：此处的 AllocatableDevice 是一个抽象表示，
		// 不是具体的 MIG 设备实例，因此无法直接更新其健康状态。
		// 当 GPU 被标记为不健康时，所有基于该 GPU 的动态 MIG 配置也会隐式变为不可用。
		klog.Warningf("UpdateDeviceHealthStatus() called for abstract, dynamic MIG device %s: %s", d.MigDynamic.CanonicalName(), hs)
	case MigStaticDeviceType:
		// 静态 MIG 设备：更新其 health 字段。
		// 注意：目前不确定是否能接收到针对单个 MIG 设备的健康事件，
		// 如果能的话，是否可以据此推断父 GPU 的健康状态？
		d.MigStatic.health = hs
	default:
		// 未知设备类型：无法更新健康状态
		klog.V(6).Infof("Cannot update health status for unknown device type: %s", d.Type())
		return
	}
	klog.V(4).Infof("Updated device: %s health status to %s", d.UUID(), hs)
}

// requestedNonAdminDevices 返回当前 Claim 请求的设备名称集合，排除 admin 访问权限的分配。
// admin 访问允许重叠分配，因此不参与重叠检测。
//
// 参数：
//   - claim: 当前 ResourceClaim 对象
//
// 返回值：
//   - map[string]struct{}: 非 admin 设备名称集合
func (s *DeviceState) requestedNonAdminDevices(claim *resourceapi.ResourceClaim) map[string]struct{} {
	requested := make(map[string]struct{}, len(claim.Status.Allocation.Devices.Results))

	for _, r := range claim.Status.Allocation.Devices.Results {
		// 跳过非本驱动的分配结果
		if r.Driver != DriverName {
			continue
		}
		// 跳过 admin 访问的分配（允许重叠）
		if r.AdminAccess != nil && *r.AdminAccess {
			continue
		}
		requested[r.Device] = struct{}{}
	}
	return requested
}

// validateNoOverlappingPreparedDevices 检查当前 Claim 请求的设备是否已被其他已完全准备的 Claim 占用。
// 仅当存在非 admin 设备重叠时才返回错误，admin 设备允许重叠分配。
//
// 参数：
//   - checkpoint: 当前 Checkpoint 对象
//   - claim:      当前待准备的 ResourceClaim
//
// 返回值：
//   - error: 存在设备重叠时返回错误，否则返回 nil
func (s *DeviceState) validateNoOverlappingPreparedDevices(checkpoint *Checkpoint, claim *resourceapi.ResourceClaim) error {
	claimUID := string(claim.UID)

	// 获取当前 Claim 请求的非 admin 设备集合
	requestedDevices := s.requestedNonAdminDevices(claim)
	if len(requestedDevices) == 0 {
		// 当前 Claim 没有非 admin 设备请求，无需检查
		return nil
	}

	// 遍历 Checkpoint 中所有已准备的 Claim
	for existingClaimUID, pc := range checkpoint.V2.PreparedClaims {
		// 跳过当前 Claim 自身
		if existingClaimUID == claimUID {
			continue
		}
		// 仅检查已完全准备的 Claim
		if pc.CheckpointState != ClaimCheckpointStatePrepareCompleted {
			continue
		}

		// 获取已准备 Claim 中的非 admin 设备集合
		// 仅当设备以 admin 访问权限分配时才允许重叠
		preparedDevices := pc.GetNonAdminDevices()
		if len(preparedDevices) == 0 {
			continue
		}

		// 检查当前请求的设备是否与已准备的设备重叠
		for device := range requestedDevices {
			if _, found := preparedDevices[device]; found {
				return fmt.Errorf(
					"requested device %s is already allocated to different claim %s",
					device, existingClaimUID,
				)
			}
		}
	}
	return nil
}

// deleteMigDevIfExistsAndNotUsedByCompletedClaim 尝试删除指定的 MIG 设备，但仅在以下条件满足时执行：
//  1. 该 MIG 设备在物理 GPU 上确实存在（通过 FindMigDevBySpec 查找）
//  2. 该 MIG 设备未被任何已完全准备（PrepareCompleted）的 Claim 引用
//
// 若设备不存在或已被其他 Claim 引用，则跳过删除。当前采用尽力而为策略，不返回错误
// （即使 NVML 调用失败也仅记录日志），因为在清理路径上不应因单个设备清理失败而中断整个流程。
//
// 参数：
//   - ms:    MIG 规范元组（标识设备配置和放置位置）
//   - dname: 设备规范名称（用于日志和查重）
//   - completelyPreparedClaims: 所有已完全准备的 Claim 映射（用于检查设备引用）
//
// 返回值：
//   - error: 删除失败时返回错误，跳过删除时返回 nil
func (s *DeviceState) deleteMigDevIfExistsAndNotUsedByCompletedClaim(ms *MigSpecTuple, dname DeviceName, completelyPreparedClaims PreparedClaimsByUID) error {
	// 首先检查该设备是否被任何已完全准备的 Claim 引用
	for uid, claim := range completelyPreparedClaims {
		for _, res := range claim.Status.Allocation.Devices.Results {
			if res.Device == dname {
				// 设备正被已完成的 Claim 使用，不能删除
				klog.V(1).Infof("Device %s is in use by completely prepared claim %s", dname, PreparedClaimToString(&claim, uid))
				return nil
			}
		}
	}

	// 设备未被任何已完成的 Claim 使用，尝试在物理 GPU 上查找对应的实际 MIG 设备
	klog.V(1).Infof("Device '%s' is not in use by any completely prepared claim, find corresponding actual MIG device", dname)
	mlt, err := s.nvdevlib.FindMigDevBySpec(ms)
	if err != nil {
		return fmt.Errorf("FindMigDevBySpecTuple() failed for %s: %s", dname, err)
	}

	if mlt == nil {
		// 对应的物理 MIG 设备不存在，无需清理
		klog.V(1).Infof("No live MIG device corresponding to name %s currently exists (nothing to clean up)", dname)
		return nil
	}

	// 找到对应的物理 MIG 设备，尝试拆除
	klog.V(1).Infof("MIG device corresponding to name %s found with UUID %s -- attempt to tear down", dname, mlt.MigUUID)
	if err := s.nvdevlib.deleteMigDevice(mlt); err != nil {
		return fmt.Errorf("MIG device deletion failed: %w", err)
	}

	return nil
}
