/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

// driver.go — GPU DRA 驱动核心实现
//
// 本文件实现了 Kubernetes 动态资源分配（DRA）驱动的核心逻辑，负责：
//   - 驱动实例的创建与生命周期管理（NewDriver / Shutdown）
//   - 向 kubelet 注册 DRA 插件 gRPC 服务
//   - NodePrepareResources / NodeUnprepareResources 的入口及全局互斥
//   - 将节点上的 GPU 设备信息以 ResourceSlice 形式发布到 API Server
//   - NVML 设备健康事件的监听与 ResourceSlice 重发布
//   - 根据 Kubernetes API Server 版本决定 ResourceSlice 发布模式
//     （Split Slices 模式 vs Combined Slices 模式）
package main

import (
	"context"
	"fmt"
	"maps"
	"path/filepath"
	"slices"
	"sync"
	"time"

	"github.com/Masterminds/semver"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/runtime"
	coreclientset "k8s.io/client-go/kubernetes"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/klog/v2"

	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/flock"
)

// DriverPrepUprepFlockFileName 是 Prepare/Unprepare 操作的文件锁名称。
// 用于保证 nodePrepareResource() / nodeUnprepareResource() 调用在节点全局范围内不会交替执行。
// 这是一个基于文件系统的互斥锁，确保同一时刻只有一个 Prepare 或 Unprepare 操作在执行，
// 防止并发操作导致设备状态不一致（例如同时创建和销毁 MIG 设备）。
const DriverPrepUprepFlockFileName = "pu.lock"

// deviceHealthMonitor 是设备健康监控器的接口定义。
// 该接口抽象了 GPU 设备健康状态的监控能力，具体实现由 nvmlDeviceHealthMonitor 提供。
// 设计为接口而非具体类型，便于测试时注入 mock 实现。
type deviceHealthMonitor interface {
	// Start 启动健康监控器，开始监听 NVML 硬件事件。
	// 返回 error 表示启动失败（如 NVML 初始化失败、事件订阅失败等）。
	Start(context.Context) error

	// Stop 停止健康监控器，释放相关资源。
	Stop()

	// Unhealthy 返回一个只读通道，当检测到设备不健康时，会向该通道发送对应的 AllocatableDevice。
	// 消费者应持续从此通道读取，直到通道关闭（发生在 Stop 调用后）。
	Unhealthy() <-chan *AllocatableDevice
}

// driver 是整个 GPU DRA 驱动插件的运行时实例，封装了驱动运行所需的全部状态和依赖。
//
// 字段说明：
//   - client:            Kubernetes core API 客户端，用于访问 API Server（版本探测等）
//   - pluginhelper:      kubelet DRA 插件辅助器，处理 gRPC 服务注册和 ResourceSlice 发布
//   - state:             设备状态管理器，包含可分配设备列表、Checkpoint 等
//   - pulock:            Prepare/Unprepare 文件互斥锁，保证操作的全局互斥性
//   - healthcheck:       HTTP 健康检查服务，提供 /healthz 端点
//   - deviceHealthMonitor: NVML 设备健康监控器（仅在 NVMLDeviceHealthCheck 特性启用时非 nil）
//   - wg:                等待组，用于优雅关闭时等待后台 goroutine 退出
//   - useSplitResourceSlices: 是否使用独立的 ResourceSlice 分别发布 SharedCounters 和 Devices
type driver struct {
	client              coreclientset.Interface
	pluginhelper        *kubeletplugin.Helper
	state               *DeviceState
	pulock              *flock.Flock
	healthcheck         *healthcheck
	deviceHealthMonitor deviceHealthMonitor
	wg                  sync.WaitGroup
	// 标志是否为 SharedCounters 和 Devices 使用独立的 ResourceSlice（k8s 1.35+ 要求），
	// 还是合并到同一个 Slice 中（k8s 1.34 要求）。
	useSplitResourceSlices bool
}

// NewDriver 创建并启动 GPU DRA 驱动的完整运行时实例。
// 它是整个插件的核心初始化入口，按顺序完成以下工作：
//
//  1. 初始化设备状态（发现 GPU/MIG 设备）
//  2. 处理 DynamicMIG 遗留设备（按需）
//  3. 注册 kubelet DRA 插件（建立 gRPC 服务）
//  4. 启动健康检查
//  5. 启动 NVML 设备健康监控（按需）
//  6. 启动 Checkpoint 清理管理器
//  7. 向 Kubernetes API Server 发布 ResourceSlice
//
// 参数：
//   - ctx:    上下文，用于控制生命周期和传递取消信号
//   - config: 插件配置对象，包含命令行参数、API 客户端等
//
// 返回值：
//   - *driver: 初始化完成的驱动实例
//   - error:   任一步骤失败则返回错误
func NewDriver(ctx context.Context, config *Config) (*driver, error) {
	// ──────────────────────────────────────────────
	// 第一步：初始化设备状态
	// ──────────────────────────────────────────────

	// 扫描节点上的 GPU 设备（通过 NVML），构建可分配设备列表，
	// 并加载已有的 Checkpoint 状态（记录已 Prepare 的 Claim 信息）。
	state, err := NewDeviceState(ctx, config)
	if err != nil {
		return nil, err
	}

	// ResourceSlice 发布模式标志：
	//   false（默认）：所有设备发布在单个 ResourceSlice 中（k8s 1.34 兼容模式）。
	//   true          ：按设备类型拆分为多个 ResourceSlice（Split Slices 模式），
	//                   用于 k8s 1.35+，以支持更细粒度的调度决策。
	useSplitSlices := false

	// ──────────────────────────────────────────────
	// 第二步：DynamicMIG 特性门控处理
	// ──────────────────────────────────────────────

	// 仅在 DynamicMIG 特性开启时执行。
	// DynamicMIG 允许插件在运行时动态创建/销毁 MIG（Multi-Instance GPU）分区，
	// 而非依赖静态预分配。
	if featuregates.Enabled(featuregates.DynamicMIG) {
		// 处理插件重启后节点上遗留的"孤儿" MIG 设备。
		//
		// 背景：插件崩溃、重启或管理员干预可能导致节点上存在
		// 不在 Checkpoint 记录中的 MIG 设备实例。对此有三种处理策略：
		//
		// 方案 1（放弃）：假设这些设备由外部管理，不上报。
		//   问题：事实上几乎不可能，且遗留设备会占用 GPU 资源。
		//
		// 方案 2（忽略）：照常上报这些 MIG 设备。
		//   问题：当调度器分配任务后，NodePrepareResources() 会尝试
		//   再次创建同名 MIG 实例，因已存在而报错：
		//   "prepare devices failed: error creating MIG device:
		//    error creating GPU instance for 'gpu-0-mig-1g24gb-0':
		//    Insufficient Resources"
		//
		// 方案 3（采用）：以节点本地 Checkpoint 作为唯一可信状态。
		//   - "部分 Prepared" 状态的 Claim 对应的 MIG 设备 → 销毁（准备未完成，不可信）。
		//   - 完全不在 Checkpoint 中的 MIG 设备 → 销毁（来源不明，不可信）。
		//   - 完全 Prepared 状态的 Claim 对应的 MIG 设备 → 保留。
		//
		// TODO: 评估此逻辑在边缘场景（如调度器状态与本地状态不一致时）
		//       是否过于激进，需进一步审查。
		state.DestroyUnknownMIGDevices(ctx)

		// 通过查询 Kubernetes API Server 版本来决定使用哪种 ResourceSlice 模型。
		// 较新的 API Server 版本支持 Split Slices 模式，以提升大规模集群下的调度性能。
		var err error
		useSplitSlices, err = shouldUseSplitResourceSlices(config.clientsets.Core)
		if err != nil {
			return nil, fmt.Errorf("failed to determine ResourceSlice model: %w", err)
		}
	}

	// ──────────────────────────────────────────────
	// 第三步：构建 driver 实例
	// ──────────────────────────────────────────────

	// Prepare/Unprepare 操作的文件锁路径。
	// 用于防止并发的 NodePrepareResources / NodeUnprepareResources 调用
	// 同时操作同一设备，保证操作的原子性。
	puLockPath := filepath.Join(config.DriverPluginPath(), DriverPrepUprepFlockFileName)

	driver := &driver{
		client:                 config.clientsets.Core,     // Kubernetes core API 客户端
		state:                  state,                      // 设备状态（可分配列表、Checkpoint）
		pulock:                 flock.NewFlock(puLockPath), // Prepare/Unprepare 文件互斥锁
		useSplitResourceSlices: useSplitSlices,             // ResourceSlice 发布模式
	}

	// ──────────────────────────────────────────────
	// 第四步：注册 kubelet DRA 插件
	// ──────────────────────────────────────────────

	// 启动 DRA kubelet 插件 gRPC 服务并完成向 kubelet 的注册。
	// kubeletplugin.Start 内部会：
	//   1. 在 PluginDataDirectoryPath 下创建 Unix domain socket
	//   2. 在 RegistrarDirectoryPath 下创建注册 socket
	//   3. 等待 kubelet pluginwatcher 完成注册握手（GetInfo / NotifyRegistrationStatus）
	//   4. 开始接收 kubelet 的 NodePrepareResources / NodeUnprepareResources 调用
	//
	// Serialize(false)：允许并发处理多个 RPC 请求，
	//   由上方的 pulock 文件锁在业务层保证操作互斥，而非序列化整个 gRPC 服务。
	helper, err := kubeletplugin.Start(
		ctx,
		driver,                                  // DRAPlugin 接口实现
		kubeletplugin.KubeClient(driver.client), // 用于操作 ResourceSlice 等资源
		kubeletplugin.NodeName(config.flags.nodeName),                                    // 当前节点名
		kubeletplugin.DriverName(DriverName),                                             // 驱动名称，如 "gpu.nvidia.com"
		kubeletplugin.Serialize(false),                                                   // 不串行化 RPC（允许并发）
		kubeletplugin.RegistrarDirectoryPath(config.flags.kubeletRegistrarDirectoryPath), // 注册 socket 目录
		kubeletplugin.PluginDataDirectoryPath(config.DriverPluginPath()),                 // 插件数据 socket 目录
	)
	if err != nil {
		return nil, err
	}
	driver.pluginhelper = helper

	// ──────────────────────────────────────────────
	// 第五步：启动健康检查服务
	// ──────────────────────────────────────────────

	// 启动 HTTP 健康检查端点（/healthz），供 kubelet liveness/readiness probe 使用。
	// helper 被传入以便健康检查可以反映 kubelet 插件注册状态。
	healthcheck, err := startHealthcheck(ctx, config, helper)
	if err != nil {
		return nil, fmt.Errorf("start healthcheck: %w", err)
	}
	driver.healthcheck = healthcheck

	// ──────────────────────────────────────────────
	// 第六步：NVML 设备健康监控（特性门控）
	// ──────────────────────────────────────────────

	// 仅在 NVMLDeviceHealthCheck 特性开启时启动 GPU 硬件健康监控。
	// 该监控通过 NVML 持续观察 GPU 的 XID 错误事件，
	// 一旦检测到致命错误，将对应设备标记为不健康并更新 ResourceSlice。
	if featuregates.Enabled(featuregates.NVMLDeviceHealthCheck) {
		// 创建 NVML 设备健康监控器。
		// allocatable：当前可分配的设备列表（仅监控已知设备）。
		// nvdevlib   ：NVML 库的封装，提供事件订阅能力。
		deviceHealthMonitor, err := newNvmlDeviceHealthMonitor(config, state.allocatable, state.nvdevlib)
		if err != nil {
			return nil, fmt.Errorf("failed to create NVML device health monitor: %w", err)
		}
		// 启动监控器，开始订阅 NVML 硬件事件（非阻塞，内部启动独立 goroutine）。
		if err := deviceHealthMonitor.Start(ctx); err != nil {
			return nil, fmt.Errorf("failed to start device health monitor: %w", err)
		}
		driver.deviceHealthMonitor = deviceHealthMonitor
		// 启动设备健康事件处理 goroutine。
		// 负责消费 deviceHealthMonitor 产生的健康状态变更事件，
		// 并将结果同步到 ResourceSlice（标记设备健康/不健康），
		// 供 Kubernetes 调度器感知并避免将新任务调度到故障设备。
		driver.wg.Add(1)
		go func() {
			defer driver.wg.Done()
			driver.deviceHealthEvents(ctx, config.flags.nodeName)
		}()
	}

	// ──────────────────────────────────────────────
	// 第七步：启动 Checkpoint 清理管理器
	// ──────────────────────────────────────────────

	// CheckpointCleanupManager 负责处理已完成但 Checkpoint 尚未清理的 Claim。
	// 典型场景：Pod 删除后，NodeUnprepareResources 调用成功但进程随即崩溃，
	// 导致 Checkpoint 条目残留。重启后该管理器会重新执行清理。
	//
	// nodeUnprepareResource 作为回调函数传入，在清理时被调用以释放对应设备资源。
	if err := state.checkpointCleanupManager.Start(ctx, driver.nodeUnprepareResource); err != nil {
		return nil, fmt.Errorf("error starting CheckpointCleanupManager: %w", err)
	}

	// ──────────────────────────────────────────────
	// 第八步：向 API Server 发布 ResourceSlice
	// ──────────────────────────────────────────────

	// 将节点上当前可用的 GPU 设备信息（类型、数量、属性、健康状态等）
	// 以 ResourceSlice 对象的形式写入 Kubernetes API Server，
	// 使调度器能够感知本节点的 GPU 资源并进行分配决策。
	if err := driver.publishResources(ctx, config); err != nil {
		return nil, err
	}

	// 打印当前 kubelet 插件注册状态（仅在 -v=4 及以上级别输出），用于调试确认。
	klog.V(4).Infof("Current kubelet plugin registration status: %s", helper.RegistrationStatus())

	return driver, nil
}

// GenerateDriverResources 返回此 DRA 驱动向系统公告的 ResourceSlice 集合，
// 使用 Partitionable Devices 范式（KEP 4815）。
//
// 根据 useSplitResourceSlices 标志，选择两种生成策略之一：
//   - Split Slices 模式（k8s 1.35+）：SharedCounters 和 Devices 分布在不同的 Slice 中
//   - Combined Slices 模式（k8s 1.34）：SharedCounters 和 Devices 在同一 Slice 中
//
// 参数：
//   - nodeName: 节点名称，用于标识 ResourceSlice 所属的节点
//
// 返回值：
//   - resourceslice.DriverResources: 包含所有 ResourceSlice 的驱动资源集合
func (d *driver) GenerateDriverResources(nodeName string) resourceslice.DriverResources {
	if d.useSplitResourceSlices {
		// 使用 Split Slices 模式
		return d.generateSplitResourceSlices(nodeName)
	}
	// 使用 Combined Slices 模式
	return d.generateCombinedResourceSlices(nodeName)
}

// generateSplitResourceSlices 为 DynamicMIG 生成 k8s 1.35+ 的 Split ResourceSlices。
// 为 G 块物理 GPU 创建 G+1 个 ResourceSlice：
//   - 一个包含所有 SharedCounterSet 的 Slice（每块 GPU 一个计数器集）。
//   - 每块 GPU 对应一个仅包含设备的 Slice（完整 GPU + MIG 分区）。
//
// 这种拆分方式使得调度器可以独立地处理计数器集和设备，
// 在大规模集群中减少 ResourceSlice 的更新冲突。
//
// 参数：
//   - nodeName: 节点名称
//
// 返回值：
//   - resourceslice.DriverResources: 按节点组织的 ResourceSlice 集合
func (d *driver) generateSplitResourceSlices(nodeName string) resourceslice.DriverResources {
	var gpuslices []resourceslice.Slice
	var allCounterSets []resourceapi.CounterSet

	// 按可预测的顺序（minor 号升序）遍历 `perGPUAllocatable` 映射
	// 确保每次启动时生成的 ResourceSlice 顺序一致，减少不必要的更新操作
	for _, minor := range slices.Sorted(maps.Keys(d.state.perGPUAllocatable)) {
		allocatable := d.state.perGPUAllocatable[minor]
		var deviceSlice resourceslice.Slice

		// 按设备名稳定排序，确保 ResourceSlice 中设备顺序可重现
		for _, devname := range slices.Sorted(maps.Keys(allocatable)) {
			device := allocatable[devname]
			klog.V(4).Infof("About to announce device %s", devname)

			// 完整 GPU：收集其计数器集到 `sharedCountersSlice`。
			// 计数器集表示该物理 GPU 的绝对容量（如显存切片数、计算切片数）。
			if device.Gpu != nil {
				allCounterSets = append(allCounterSets, device.Gpu.PartSharedCounterSets()...)
			}

			// 将设备/分区添加到该 GPU 对应的仅设备 Slice 中。
			// 这包括完整 GPU 本身和所有可能的 MIG 分区配置。
			deviceSlice.Devices = append(deviceSlice.Devices, device.PartGetDevice())
		}
		gpuslices = append(gpuslices, deviceSlice)
	}

	// 构建 SharedCounters 专属的 Slice，包含所有物理 GPU 的计数器集
	sharedCountersSlice := resourceslice.Slice{
		SharedCounters: allCounterSets,
	}

	// 将 `sharedCountersSlice` 排在最前面，确保优先发布计数器信息。
	// 调度器需要先获取计数器集信息才能正确计算 MIG 分区的资源消耗。
	gpuslices = append([]resourceslice.Slice{sharedCountersSlice}, gpuslices...)

	return resourceslice.DriverResources{
		Pools: map[string]resourceslice.Pool{
			nodeName: {Slices: gpuslices},
		},
	}
}

// generateCombinedResourceSlices 为 DynamicMIG 生成 k8s 1.34 的 Combined ResourceSlices。
// 为 G 块物理 GPU 创建 G 个 ResourceSlice，每个 Slice 同时包含
// SharedCounters 和 Devices（k8s 1.34 兼容模式）。
//
// 与 Split 模式不同，此模式下每个 Slice 同时包含计数器集和设备列表，
// 因为 k8s 1.34 不支持独立计数器集的 Slice。
//
// 参数：
//   - nodeName: 节点名称
//
// 返回值：
//   - resourceslice.DriverResources: 按节点组织的 ResourceSlice 集合
func (d *driver) generateCombinedResourceSlices(nodeName string) resourceslice.DriverResources {
	var gpuslices []resourceslice.Slice

	// 按可预测的顺序（minor 号升序）遍历 `perGPUAllocatable` 映射
	for _, minor := range slices.Sorted(maps.Keys(d.state.perGPUAllocatable)) {
		allocatable := d.state.perGPUAllocatable[minor]
		var slice resourceslice.Slice
		countersets := []resourceapi.CounterSet{}

		// 按设备名稳定排序，确保 ResourceSlice 中设备顺序可重现。
		// 这有利于调试和可读性，并减少插件重启时的 Slice diff（diff 会被记录日志）。
		for _, devname := range slices.Sorted(maps.Keys(allocatable)) {
			device := allocatable[devname]
			klog.V(4).Infof("About to announce device %s", devname)

			// 完整 GPU：记录其计数器集，表示绝对容量。
			// 当前预期恰好一个计数器集对应一个物理 GPU。
			if device.Gpu != nil {
				countersets = append(countersets, device.Gpu.PartSharedCounterSets()...)
			}

			// 将该物理 GPU 的所有可分配设备（含未物化的 MIG 设备和物理 GPU 本身）
			// 添加到此 Slice 中。
			slice.Devices = append(slice.Devices, device.PartGetDevice())
		}
		// 将计数器集附加到该 Slice（Combined 模式下与设备共存）
		slice.SharedCounters = countersets
		gpuslices = append(gpuslices, slice)
	}

	return resourceslice.DriverResources{
		Pools: map[string]resourceslice.Pool{
			nodeName: {Slices: gpuslices},
		},
	}
}

// Shutdown 执行驱动的优雅关闭，按逆序释放各子系统资源。
//
// 关闭顺序：
//  1. 停止 HTTP 健康检查服务
//  2. 关闭长生命周期 NVML 会话（DynamicMIG 模式）
//  3. 停止 NVML 设备健康监控器
//  4. 等待后台健康事件处理 goroutine 退出
//  5. 停止 Checkpoint 清理管理器
//  6. 停止 kubelet DRA 插件 gRPC 服务
//
// 返回值：
//   - error: 任一步骤失败则返回错误；driver 为 nil 时安全返回 nil
func (d *driver) Shutdown() error {
	if d == nil {
		// 安全处理 nil 驱动实例
		return nil
	}

	// 第一步：停止健康检查 HTTP 服务
	if d.healthcheck != nil {
		d.healthcheck.Stop()
	}

	// 第二步：关闭长生命周期的 NVML 会话（仅 DynamicMIG 模式需要）
	// DynamicMIG 模式下 NVML 会话在整个驱动生命周期内保持打开，
	// 必须在退出前显式关闭以释放 NVML 资源。
	if featuregates.Enabled(featuregates.DynamicMIG) {
		d.state.nvdevlib.alwaysShutdown()
	}

	// 第三步：停止 NVML 设备健康监控器，使其不再产生新的事件
	if d.deviceHealthMonitor != nil {
		d.deviceHealthMonitor.Stop()
	}

	// 第四步：等待健康事件处理 goroutine 退出
	// 确保所有正在处理的事件完成后再继续关闭
	d.wg.Wait()

	// 第五步：停止 Checkpoint 清理管理器
	if err := d.state.checkpointCleanupManager.Stop(); err != nil {
		return fmt.Errorf("error stopping CheckpointCleanupManager: %w", err)
	}

	// 第六步：停止 kubelet DRA 插件 gRPC 服务
	// 这会关闭与 kubelet 的连接并清理 socket 文件
	d.pluginhelper.Stop()
	return nil
}

// PrepareResourceClaims 实现 DRAPlugin 接口，是 kubelet 调用 NodePrepareResources 的入口。
// 对每个 Claim 并行调用 nodePrepareResource 进行设备准备。
//
// 参数：
//   - ctx:    上下文，用于传递取消信号
//   - claims: 待准备的 ResourceClaim 列表
//
// 返回值：
//   - map[types.UID]kubeletplugin.PrepareResult: 以 Claim UID 为键的准备结果映射
//   - error: 全局错误（当前实现始终返回 nil，单个 Claim 的错误封装在 PrepareResult.Err 中）
func (d *driver) PrepareResourceClaims(ctx context.Context, claims []*resourceapi.ResourceClaim) (map[types.UID]kubeletplugin.PrepareResult, error) {

	if len(claims) == 0 {
		// 空调用可能是健康检查，在高详细级别下记录
		klog.V(7).Infof("PrepareResourceClaims called with %d claim(s)", len(claims))
	} else {
		// 为每个注入的 Claim 记录规范字符串表示——
		// 这对调试非常有帮助。
		klog.V(6).Infof("Prepare called for: %v", ClaimsToStrings(claims))
	}

	// 为每个 Claim 构建准备结果
	results := make(map[types.UID]kubeletplugin.PrepareResult)
	for _, claim := range claims {
		results[claim.UID] = d.nodePrepareResource(ctx, claim)
	}

	return results, nil
}

// UnprepareResourceClaims 实现 DRAPlugin 接口，是 kubelet 调用 NodeUnprepareResources 的入口。
// 对每个 ClaimRef 并行调用 nodeUnprepareResource 进行设备释放。
//
// 参数：
//   - ctx:       上下文，用于传递取消信号
//   - claimRefs: 待释放的 ResourceClaim 引用列表（包含命名空间、名称、UID）
//
// 返回值：
//   - map[types.UID]error: 以 Claim UID 为键的错误映射，单个 Claim 的释放错误体现在对应的值中
//   - error: 全局错误（当前实现始终返回 nil）
func (d *driver) UnprepareResourceClaims(ctx context.Context, claimRefs []kubeletplugin.NamespacedObject) (map[types.UID]error, error) {
	klog.V(6).Infof("Unprepare called for: %v", ClaimRefsToStrings(claimRefs))
	results := make(map[types.UID]error)
	for _, claimRef := range claimRefs {
		results[claimRef.UID] = d.nodeUnprepareResource(ctx, claimRef)
	}

	return results, nil
}

// HandleError 实现 DRAPlugin 接口，处理驱动运行时产生的非致命错误。
// 目前遵循 DRAPlugin API 文档中的建议，使用 Kubernetes 的标准错误处理机制。
//
// 参见：https://pkg.go.dev/k8s.io/apimachinery/pkg/util/runtime#HandleErrorWithContext
//
// 参数：
//   - ctx: 上下文
//   - err:  需要处理的错误
//   - msg:  错误描述消息
func (d *driver) HandleError(ctx context.Context, err error, msg string) {
	// 目前遵循 DRAPlugin API 文档中的建议。
	runtime.HandleErrorWithContext(ctx, err, msg)
}

// nodePrepareResource 是单个 ResourceClaim 设备准备的核心实现。
// 它在 Prepare/Unprepare 全局文件锁的保护下执行，保证同一时刻
// 节点上只有一个 Prepare 或 Unprepare 操作在执行。
//
// 执行流程：
//  1. 获取 pu.lock 文件互斥锁（10 秒超时）
//  2. 调用 DeviceState.Prepare() 执行实际的设备准备逻辑
//  3. 若 PassthroughSupport 特性启用，准备完成后重新发布 ResourceSlice
//     以反映设备状态变更
//
// 参数：
//   - ctx:   上下文
//   - claim: 待准备的 ResourceClaim 对象
//
// 返回值：
//   - kubeletplugin.PrepareResult: 包含准备好的设备列表或错误信息
func (d *driver) nodePrepareResource(ctx context.Context, claim *resourceapi.ResourceClaim) kubeletplugin.PrepareResult {
	// 虽然可以使用更细粒度的 Checkpoint 锁代替全局 PU 锁
	// （已证明在 DynamicMIG 模式下能正确工作），
	// 但出于谨慎考虑，目前保留全局 PU 锁（稍后重新评估性能影响）。
	t0 := time.Now()
	release, err := d.pulock.Acquire(ctx, flock.WithTimeout(10*time.Second))
	if err != nil {
		// 获取锁失败（超时或上下文取消），直接返回错误
		return kubeletplugin.PrepareResult{
			Err: fmt.Errorf("error acquiring prep/unprep lock: %w", err),
		}
	}
	defer release()
	klog.V(6).Infof("t_prep_lock_acq %.3f s", time.Since(t0).Seconds())

	// 生成 Claim 的可读标识字符串
	cs := ResourceClaimToString(claim)

	// 调用 DeviceState.Prepare() 执行核心设备准备逻辑
	tprep0 := time.Now()
	devs, err := d.state.Prepare(ctx, claim)
	klog.V(6).Infof("t_prep %.3f s (claim %s)", time.Since(tprep0).Seconds(), cs)

	if err != nil {
		return kubeletplugin.PrepareResult{
			Err: fmt.Errorf("error preparing devices for claim %s: %w", cs, err),
		}
	}

	// 当 PassthroughSupport 特性门控启用时，设备准备可能导致可分配设备列表变化
	// （例如移除了兄弟设备），需要重新发布 ResourceSlice 以反映最新状态
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		if err = d.publishResources(ctx, d.state.config); err != nil {
			return kubeletplugin.PrepareResult{
				Err: fmt.Errorf("error preparing devices for claim %v: %w", claim.UID, err),
			}
		}
	}

	klog.Infof("Returning newly prepared devices for claim '%s': %v", cs, devs)
	return kubeletplugin.PrepareResult{Devices: devs}
}

// nodeUnprepareResource 是单个 ResourceClaim 设备释放的核心实现。
// 同样在 Prepare/Unprepare 全局文件锁的保护下执行。
//
// 执行流程：
//  1. 获取 pu.lock 文件互斥锁（10 秒超时）
//  2. 调用 DeviceState.Unprepare() 执行实际的设备释放逻辑
//  3. 若 PassthroughSupport 特性启用，释放完成后重新发布 ResourceSlice
//     以反映设备状态变更（如兄弟设备重新变为可分配）
//
// 参数：
//   - ctx:      上下文
//   - claimRef: 待释放的 ResourceClaim 引用（包含命名空间、名称、UID）
//
// 返回值：
//   - error: 释放过程中的错误，nil 表示成功
func (d *driver) nodeUnprepareResource(ctx context.Context, claimRef kubeletplugin.NamespacedObject) error {
	t0 := time.Now()
	release, err := d.pulock.Acquire(ctx, flock.WithTimeout(10*time.Second))
	if err != nil {
		return fmt.Errorf("error acquiring prep/unprep lock: %w", err)
	}
	defer release()
	klog.V(6).Infof("t_unprep_lock_acq %.3f s", time.Since(t0).Seconds())

	// 生成 ClaimRef 的可读标识字符串
	cs := claimRef.String()

	// 调用 DeviceState.Unprepare() 执行核心设备释放逻辑
	tunprep0 := time.Now()
	err = d.state.Unprepare(ctx, claimRef)
	klog.V(6).Infof("t_unprep %.3f s (claim %s)", time.Since(tunprep0).Seconds(), cs)

	if err != nil {
		return fmt.Errorf("error unpreparing devices for claim %v: %w", claimRef.String(), err)
	}

	// 当 PassthroughSupport 特性门控启用时，设备释放可能导致可分配设备列表变化
	// （如兄弟设备重新变为可分配），需要重新发布 ResourceSlice 以反映最新状态
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		if err = d.publishResources(ctx, d.state.config); err != nil {
			return fmt.Errorf("error publishing resources: %w", err)
		}
	}

	return nil
}

// publishResources 将当前节点上的 GPU 设备信息以 ResourceSlice 的形式发布到 Kubernetes API Server。
//
// 工作模式分为两种：
//   - DynamicMIG 启用：使用 KEP 4815（Partitionable Devices）范式构建 ResourceSlice。
//     通过 GenerateDriverResources 生成驱动资源，再由 PublishResources 写入 API Server。
//     控制器辅助函数会在应用前验证 Slice 的正确性。
//   - DynamicMIG 未启用（传统模式）：将所有设备（GPU、MIG、VFIO）平铺到单个 ResourceSlice 中。
//
// 参数：
//   - ctx:   上下文
//   - config: 插件配置对象，用于获取节点名称等信息
//
// 返回值：
//   - error: 发布失败时返回错误
func (d *driver) publishResources(ctx context.Context, config *Config) error {

	if featuregates.Enabled(featuregates.DynamicMIG) {
		// 来自 KEP 4815："我们将在 ResourceSlice 控制器辅助中添加客户端验证，
		// 以便 ResourceSlice 中的任何错误在应用到 API Server 之前就会被捕获"
		// ——以下辅助函数即为此处所指的控制器辅助。
		//
		// TODO: 为错误的 Slice 实现错误处理器：
		// https://github.com/kubernetes/kubernetes/commit/a171795e313ee9f407fef4897c1a1e2052120991
		klog.V(1).Infof("featuregates.DynamicMIG enabled: construct ResourceSlice objects according to KEP 4815 (partitionable devices)")
		resources := d.GenerateDriverResources(config.flags.nodeName)
		if err := d.pluginhelper.PublishResources(ctx, resources); err != nil {
			return err
		}
		return nil
	}

	// 传统模式：枚举所有 GPU、MIG 和 VFIO 设备并发布到单个 ResourceSlice 中
	var resourceSlice resourceslice.Slice
	for _, device := range d.state.allocatable {
		klog.V(4).Infof("About to announce device %s", device.GetDevice().Name)
		resourceSlice.Devices = append(resourceSlice.Devices, device.GetDevice())
	}

	resources := resourceslice.DriverResources{
		Pools: map[string]resourceslice.Pool{
			config.flags.nodeName: {Slices: []resourceslice.Slice{resourceSlice}},
		},
	}

	if err := d.pluginhelper.PublishResources(ctx, resources); err != nil {
		return err
	}

	return nil

}

// deviceHealthEvents 是一个长时间运行的 goroutine，负责持续消费设备健康监控器
// 产生的不健康事件，并据此更新 ResourceSlice。
//
// 当检测到设备不健康时：
//  1. 将内存中的设备健康状态标记为 Unhealthy
//  2. 重新构建仅包含健康设备的 ResourceSlice
//  3. 将新的 ResourceSlice 发布到 API Server
//
// 注意：当前实现没有自动恢复机制。一旦设备被标记为不健康，
// 需要重启驱动才能将设备重新标记为健康。这是设备污点/容忍（KEP-5055）
// 实现前的临时妥协。
//
// 参数：
//   - ctx:      上下文，当其 Done 通道关闭时函数退出
//   - nodeName: 节点名称，用于标识 ResourceSlice 所属的节点
func (d *driver) deviceHealthEvents(ctx context.Context, nodeName string) {
	klog.V(4).Info("Starting to watch for device health notifications")
	for {
		select {
		case <-ctx.Done():
			// 上下文被取消，退出事件处理循环
			klog.V(6).Info("Stop processing device health notifications")
			return
		case device, ok := <-d.deviceHealthMonitor.Unhealthy():
			if !ok {
				// 基于 NVML 的 deviceHealthMonitor 仅在驱动 Shutdown 期间关闭通道。
				klog.V(6).Info("Health monitor channel closed")
				return
			}
			// 获取不健康设备的 UUID
			uuid := device.UUID()

			klog.Warningf("Received unhealthy notification for device: %s", uuid)

			if !device.IsHealthy() {
				// 设备已在内存中被标记为不健康，跳过重复的 ResourceSlice 重发布
				klog.V(6).Infof("Device: %s is already marked unhealthy. Skip republishing ResourceSlice", uuid)
				continue
			}

			// 将设备标记为不健康
			d.state.UpdateDeviceHealthStatus(device, Unhealthy)

			// 构建仅包含健康设备的 ResourceSlice，将不健康设备从公告中移除
			// 目前没有修复循环——如果不健康设备恢复正常，
			// 驱动需要重启才能将设备重新公告到 ResourceSlice 中
			var resourceSlice resourceslice.Slice
			for _, dev := range d.state.allocatable {
				uuid := dev.UUID()
				if dev.IsHealthy() {
					// 健康设备：添加到新的 ResourceSlice 中
					klog.V(6).Infof("Device: %s is healthy, added to ResourceSlice", uuid)
					resourceSlice.Devices = append(resourceSlice.Devices, dev.GetDevice())
				} else {
					// 不健康设备：从 ResourceSlice 中移除，调度器将不再分配
					klog.Warningf("Device: %s is unhealthy, will be removed from ResourceSlice", uuid)
				}
			}

			// 重新发布 ResourceSlice
			klog.V(4).Info("Republishing resourceslice with healthy devices")
			resources := resourceslice.DriverResources{
				Pools: map[string]resourceslice.Pool{
					nodeName: {Slices: []resourceslice.Slice{resourceSlice}},
				},
			}

			// 注意：发布失败时仅记录错误，不进行重试。
			// 若发布失败，内存中的健康状态更新成功，但 API Server 中的
			// ResourceSlice 保持过期状态，仍公告已标记为不健康的设备为可用。
			// 直到后续发布成功前，调度器和其他消费者将继续看到不健康设备
			// 为可用状态，新 Pod 可能被调度到已知不可用的硬件上。
			// 如果发布持续失败（如 API Server 问题），集群可能无限期处于
			// 这种不一致状态。
			// 这是设备污点/容忍（KEP-5055）实现前的临时妥协。
			// 一个临时改进可以是添加重试/退避机制，或切换为增量更新而非全量重发布。
			if err := d.pluginhelper.PublishResources(ctx, resources); err != nil {
				klog.Errorf("Failed to publish resources after device health status update: %v", err)
			} else {
				klog.V(4).Info("Successfully republished resources without unhealthy device")
			}
		}
	}
}

// shouldUseSplitResourceSlices 检测 Kubernetes API Server 版本，
// 决定是否使用独立的 ResourceSlice 分别存储 SharedCounters 和 Devices。
//
// 版本规则：
//   - k8s < 1.35：返回 false，使用 Combined Slices（SharedCounters 和 Devices 在同一 Slice）
//   - k8s >= 1.35：返回 true，使用 Split Slices（SharedCounters 和 Devices 在不同 Slice）
//
// 参数：
//   - client: Kubernetes core API 客户端，用于探测服务器版本
//
// 返回值：
//   - bool:  是否使用 Split ResourceSlices
//   - error: 版本检测失败时返回错误
func shouldUseSplitResourceSlices(client coreclientset.Interface) (bool, error) {
	v, err := getAPIServerVersion(client)
	if err != nil {
		return false, fmt.Errorf("API server version detection failed: %w", err)
	}

	if v.LessThan(semver.MustParse("1.35.0")) {
		// k8s 1.35 以下版本不支持独立计数器集 Slice，使用合并模式
		klog.V(2).Infof("Detected Kubernetes version %s (< 1.35), plan to use combined ResourceSlices with SharedCounters and Devices", v)
		return false, nil
	}

	// k8s 1.35+ 支持独立计数器集 Slice，使用拆分模式
	klog.V(2).Infof("Detected Kubernetes version %s (>= 1.35), plan to use separate ResourceSlices for SharedCounters and Devices", v)
	return true, nil
}

// getAPIServerVersion 通过 Kubernetes Discovery API 获取 API Server 的语义化版本号。
//
// 工作原理：
//  1. 使用 client.Discovery().ServerVersion() 获取服务器版本信息
//  2. 从 v.GitVersion 字段解析语义化版本（如 "v1.35.2"）
//  3. semver.NewVersion 能正确处理 "v" 前缀
//
// 参数：
//   - client: Kubernetes core API 客户端
//
// 返回值：
//   - *semver.Version: 解析后的语义化版本号
//   - error:          获取或解析版本号失败时返回错误
func getAPIServerVersion(client coreclientset.Interface) (*semver.Version, error) {
	discoveryClient := client.Discovery()
	v, err := discoveryClient.ServerVersion()
	if err != nil {
		return nil, fmt.Errorf("failed to get server version: %w", err)
	}

	// `v.GitVersion` 例如 "v1.35.2"；semver.NewVersion 可以处理 "v" 前缀
	semver, err := semver.NewVersion(v.GitVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to parse version '%s': %w", v.GitVersion, err)
	}

	return semver, nil
}

// TODO: 实现 CDI 文件清理循环，移除已删除 ClaimUID 的 CDI spec 文件。
// func (d *driver) cleanupCDIFiles(wg *sync.WaitGroup) chan error {
// 	errors := make(chan error)
// 	return errors
// }
//
// TODO: 实现 MPS 控制文件清理循环，移除已删除 ClaimUID 的 MPS 控制目录。
// func (d *driver) cleanupMpsControlDaemonArtifacts(wg *sync.WaitGroup) chan error {
// 	errors := make(chan error)
// 	return errors
// }
