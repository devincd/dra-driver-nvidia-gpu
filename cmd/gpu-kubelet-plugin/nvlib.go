/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
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

// nvlib.go — NVML 设备库封装与 GPU 设备发现
//
// 本文件封装了 NVML（NVIDIA Management Library）和 nvpci 的能力，提供：
//   - NVML 会话管理（初始化/关闭/长生命周期会话）
//   - GPU 设备枚举与信息采集（UUID、minor 号、内存、型号、MIG 状态等）
//   - MIG 设备的发现与生命周期管理（创建/销毁/查找）
//   - VFIO PCI 设备的枚举与信息采集
//   - GPU 时间切片和计算模式的运行时配置
//   - NVML 设备句柄的缓存与快速查询
//   - 可分配设备集合的构建（完整 GPU / 静态 MIG / 动态 MIG / VFIO）
package main

import (
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/google/uuid"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/klog/v2"

	nvdev "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvlib/pkg/nvpci"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"k8s.io/dynamic-resource-allocation/deviceattribute"

	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
)

// deviceLib 是对 NVML/nvpci 的封装，提供 GPU 设备发现、MIG 管理、VFIO 查询等核心能力。
// 通过内嵌 nvdev.Interface 继承 go-nvlib 的高层设备抽象方法。
//
// 关键设计：DynamicMIG 模式下维护长生命周期的 NVML 会话，并缓存设备句柄以加速操作。
//
// 字段说明：
//   - Interface:            内嵌的 go-nvlib 设备接口，提供 VisitDevices、NewMigProfile 等方法
//   - nvmllib:              NVML 底层接口，提供 Init/Shutdown/DeviceGetHandleByUUID 等原始 NVML API
//   - nvpci:                nvpci 接口，用于发现 PCI 总线上的 NVIDIA GPU 设备（无需 NVML）
//   - driverLibraryPath:    libnvidia-ml.so.1 的绝对路径，用于 LD_PRELOAD（nvidia-smi 等工具依赖）
//   - devRoot:              设备节点根路径（如 /dev）
//   - nvidiaSMIPath:        nvidia-smi 可执行文件路径，用于设置时间切片和计算模式
//   - gpuInfosByUUID:       UUID → GpuInfo 缓存，启动时填充，用于设备健康监控等快速查找
//   - gpuUUIDbyMinor:       minor 号 → UUID 映射，用于通过规范名反查 UUID
//   - devhandleByUUID:      UUID → NVML 设备句柄缓存（DynamicMIG 专用，加速句柄获取）
type deviceLib struct {
	nvdev.Interface
	nvmllib           nvml.Interface
	nvpci             nvpci.Interface
	driverLibraryPath string          // libnvidia-ml.so.1 的绝对路径，用于 LD_PRELOAD
	devRoot           string          // 设备节点根路径
	nvidiaSMIPath     string          // nvidia-smi 可执行文件路径
	gpuInfosByUUID    map[string]*GpuInfo  // UUID → GpuInfo 缓存，启动时填充
	gpuUUIDbyMinor    map[GPUMinor]string  // minor 号 → UUID 映射，用于通过规范名反查 UUID
	devhandleByUUID   map[string]nvml.Device // UUID → NVML 设备句柄缓存（DynamicMIG 专用）
}

// GPUMinor 是 GPU minor 号的类型别名，表示 /dev/nvidia${minor} 中的设备编号
type GPUMinor = int

// PerGPUMinorAllocatableDevices 按 GPU minor 号分组的可分配设备映射
type PerGPUMinorAllocatableDevices map[GPUMinor]AllocatableDevices

// newDeviceLib 创建并初始化 deviceLib 实例。
//
// 执行步骤：
//  1. 从驱动根路径中定位 libnvidia-ml.so.1 和 nvidia-smi
//  2. 使用显式路径创建 NVML 接口（避免依赖 LD_LIBRARY_PATH）
//  3. 创建 nvpci 接口（用于 PCI 设备发现）
//  4. 初始化内部缓存映射
//  5. 若 DynamicMIG 特性启用，初始化长生命周期 NVML 会话
//
// 参数：
//   - driverRoot: 容器内 NVIDIA 驱动根路径
//
// 返回值：
//   - *deviceLib: 初始化完成的设备库实例
//   - error:      任一步骤失败时返回错误
func newDeviceLib(driverRoot root) (*deviceLib, error) {
	// 定位 libnvidia-ml.so.1 的绝对路径
	driverLibraryPath, err := driverRoot.getDriverLibraryPath()
	if err != nil {
		return nil, fmt.Errorf("failed to locate driver libraries: %w", err)
	}

	// 定位 nvidia-smi 可执行文件路径
	nvidiaSMIPath, err := driverRoot.getNvidiaSMIPath()
	if err != nil {
		return nil, fmt.Errorf("failed to locate nvidia-smi: %w", err)
	}

	// 构造 NVML 库时显式指定 libnvidia-ml.so.1 的路径，
	// 避免依赖动态库搜索路径（LD_LIBRARY_PATH），确保在容器化环境中也能正确定位。
	nvmllib := nvml.New(
		nvml.WithLibraryPath(driverLibraryPath),
	)
	// 创建 nvpci 实例，用于发现 PCI 总线上的 GPU 设备
	nvpci := nvpci.New()

	d := deviceLib{
		Interface:         nvdev.New(nvmllib),
		nvmllib:           nvmllib,
		driverLibraryPath: driverLibraryPath,
		devRoot:           driverRoot.getDevRoot(),
		nvidiaSMIPath:     nvidiaSMIPath,
		nvpci:             nvpci,
		gpuInfosByUUID:    make(map[string]*GpuInfo),
		gpuUUIDbyMinor:    make(map[GPUMinor]string),
		devhandleByUUID:   make(map[string]nvml.Device),
	}

	// 当前设计：DynamicMIG 启用时使用长生命周期 NVML 会话。
	// 长生命周期会话意味着 NVML Init 仅调用一次，Shutdown 仅在驱动退出时调用。
	// 这是因为 DynamicMIG 需要跨多次 Prepare/Unprepare 调用保持 NVML 状态，
	// 而反复 Init/Shutdown 会导致性能问题和并发冲突。
	if featuregates.Enabled(featuregates.DynamicMIG) {
		klog.V(1).Infof("DynamicMIG enabled: initialize long-lived NVML session")
		if err := d.Init(); err != nil {
			return nil, fmt.Errorf("failed to initialize NVML: %w", err)
		}
	}

	return &d, nil
}

// prependPathListEnvvar 将指定的路径列表添加到环境变量的前面。
// 这通常用于设置 LD_PRELOAD 和 LD_LIBRARY_PATH 等路径类环境变量，
// 确保新增路径在已有路径之前被搜索。
//
// 参数：
//   - envvar:  环境变量名（如 "LD_PRELOAD" 或 "LD_LIBRARY_PATH"）
//   - prepend: 要添加到前面的路径列表
//
// 返回值：
//   - string: 修改后的环境变量值
func prependPathListEnvvar(envvar string, prepend ...string) string {
	if len(prepend) == 0 {
		// 无需添加，直接返回当前值
		return os.Getenv(envvar)
	}
	// 将当前环境变量值按路径分隔符拆分为列表
	current := filepath.SplitList(os.Getenv(envvar))
	// 将新路径添加到列表前面，然后重新拼接
	return strings.Join(append(prepend, current...), string(filepath.ListSeparator))
}

// setOrOverrideEnvvar 在环境变量列表中添加或覆盖指定的键值对。
// 若键已存在，则先移除旧值再追加新值，确保同一个键只出现一次。
//
// 参数：
//   - envvars: 当前环境变量列表（格式为 "KEY=VALUE"）
//   - key:    要设置的环境变量键
//   - value:  要设置的环境变量值
//
// 返回值：
//   - []string: 修改后的环境变量列表
func setOrOverrideEnvvar(envvars []string, key, value string) []string {
	var updated []string
	// 先过滤掉已存在的同名键
	for _, envvar := range envvars {
		pair := strings.SplitN(envvar, "=", 2)
		if pair[0] == key {
			continue // 跳过已有的同名键
		}
		updated = append(updated, envvar)
	}
	// 追加新的键值对
	return append(updated, fmt.Sprintf("%s=%s", key, value))
}

// Init 初始化 NVML 库。此方法直接调用底层 NVML 的 Init 函数。
// 在 DynamicMIG 模式下，此方法仅在启动时调用一次（长生命周期会话）。
//
// 注意：并发通过 UUID 获取 NVML 设备句柄可能需要 O(10s) 的时间。
// 为了更快的设备管理，维护长期状态：启动时初始化一次，缓存句柄，序列化 NVML 调用。
// TODO: 在 NVML 错误时实现重新初始化路径；希望在性能和健壮性之间取得平衡。
// 缺点：带外 NVML 客户端（如 mig-parted）可能看到"被其他客户端使用"的错误
// （在 DynamicMIG 模式下这是设计所禁止的）。
//
// 返回值：
//   - error: NVML 初始化失败时返回错误
func (l deviceLib) Init() error {
	klog.V(6).Infof("Call NVML Init")
	ret := l.nvmllib.Init()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error initializing NVML: %v", ret)
	}
	return nil
}

// alwaysShutdown 关闭 NVML 库。无论何时调用，都会执行 NVML Shutdown。
// 仅在 DynamicMIG 模式下有意义（关闭长生命周期会话）。
// 错误仅记录警告，不返回——因为 Shutdown 失败通常不影响驱动正常退出。
func (l deviceLib) alwaysShutdown() {
	klog.V(6).Infof("Call NVML shutdown")
	ret := l.nvmllib.Shutdown()
	if ret != nvml.SUCCESS {
		klog.Warningf("error shutting down NVML: %v", ret)
	}
}

// ensureNVML 确保 NVML 已初始化，并返回一个关闭函数和可能的错误。
// 调用方有责任在用完后调用返回的关闭函数（通常通过 defer）。
//
// 行为：
//   - DynamicMIG 已启用：NVML 已在启动时初始化，无需再次初始化，返回空操作关闭函数。
//   - DynamicMIG 未启用：每次调用时临时初始化 NVML，返回的关闭函数会执行 NVML Shutdown。
//
// 返回值：
//   - func():       关闭函数，调用后清理 NVML 会话
//   - nvml.Return:  NVML 返回码，SUCCESS 表示成功
func (l deviceLib) ensureNVML() (func(), nvml.Return) {
	// 长生命周期 NVML：无需初始化，返回空操作关闭函数。
	// 在 DynamicMIG 启用时，我们选择仅在启动时显式初始化 NVML 一次，
	// 而非依赖额外的 Init/Shutdown 配对来管理引用计数。
	if featuregates.Enabled(featuregates.DynamicMIG) {
		return func() {}, nvml.SUCCESS
	}

	// 非 DynamicMIG 模式：临时初始化 NVML
	klog.V(6).Infof("Initializing NVML")
	t0 := time.Now()
	ret := l.nvmllib.Init()
	if ret != nvml.SUCCESS {
		klog.Warningf("Failed to initialize NVML: %s", ret)
		// 初始化失败，无需清理：返回空操作。
		return func() {}, ret
	}
	klog.V(6).Infof("t_nvml_init %.3f s", time.Since(t0).Seconds())

	// 返回关闭函数，调用方通过 defer 确保清理
	return func() { l.alwaysShutdown() }, nvml.SUCCESS
}

// enumerateAllPossibleDevices 枚举节点上所有可分配的设备，包括：
//   - 完整物理 GPU（非 MIG 模式）
//   - 静态 MIG 设备（MIG 模式已启用且预创建）
//   - 动态 MIG 配置（DynamicMIG 特性门控开启时，列出所有可能的 MIG profile+placement 组合）
//   - VFIO 直通设备（PassthroughSupport 特性门控开启时）
//
// 返回两种视图：
//   - AllocatableDevices：扁平 map（设备名 → 设备），用于快速名称查找
//   - PerGPUMinorAllocatableDevices：按物理 GPU minor 号分组，用于生成 per-GPU ResourceSlice
//
// 参数：无（内部使用 deviceLib 已缓存的设备信息）
//
// 返回值：
//   - AllocatableDevices:                   扁平设备映射
//   - PerGPUMinorAllocatableDevices:         按 GPU minor 号分组的设备映射
//   - error:                                枚举过程中出现的错误
func (l deviceLib) enumerateAllPossibleDevices() (AllocatableDevices, PerGPUMinorAllocatableDevices, error) {

	// 首先枚举 NVML 可见的设备（GPU 和 MIG）
	perGPUAllocatable, err := l.GetPerGpuAllocatableDevices()
	if err != nil {
		return nil, nil, fmt.Errorf("error enumerating allocatable devices: %w", err)
	}

	// 若 PassthroughSupport 特性启用，还需枚举 VFIO PCI 直通设备
	if featuregates.Enabled(featuregates.PassthroughSupport) {
		// 发现直通设备并将其插入到返回的可分配设备列表中。
		// 这些设备会合并到之前已创建的 `perGPUAllocatable` 映射中
		passthroughDevices, err := l.enumerateGpuPciDevices(perGPUAllocatable)
		if err != nil {
			return nil, nil, fmt.Errorf("error enumerating GPU PCI devices: %w", err)
		}
		// 合并直通设备到 perGPUAllocatable
		for minor, ptdevs := range passthroughDevices {
			for name, adev := range ptdevs {
				perGPUAllocatable[minor][name] = adev
			}
		}
	}

	// 将 `perGPUAllocatable` 展平为单一映射，方便按设备名快速查找
	all := make(AllocatableDevices)
	for _, devices := range perGPUAllocatable {
		maps.Copy(all, devices)
	}

	return all, perGPUAllocatable, nil
}

// GetPerGpuAllocatableDevices 在启动时调用一次，完成物理 GPU 和 MIG 设备的完整发现。
//
// 发现流程（按 GPU 逐个处理）：
//
//  1. 查询 GPU 基本信息（UUID、minor、内存、型号、MIG 状态等）
//  2. DynamicMIG 分支：列出该 GPU 的所有 MIG Profile 可能配置（抽象 MIG）
//  3. 静态 MIG 分支：查询已创建的 MIG 设备实例
//  4. VFIO 分支：允许在无 MIG 设备时以 VFIO 模式公告 GPU
//
// 结果按 GPU minor 号分组存储于 PerGPUMinorAllocatableDevices。
//
// 参数：
//   - indices: 可选的 GPU 索引列表，用于限制枚举范围（nil 表示枚举所有 GPU）
//
// 返回值：
//   - PerGPUMinorAllocatableDevices: 按 GPU minor 号分组的可分配设备映射
//   - error:                       发现过程中出现的错误
func (l deviceLib) GetPerGpuAllocatableDevices(indices ...int) (PerGPUMinorAllocatableDevices, error) {
	klog.Infof("Traverse GPU devices")

	// 确保 NVML 已初始化
	shutdown, ret := l.ensureNVML()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("ensureNVML failed: %w", ret)
	}
	defer shutdown()

	perGPUAllocatable := make(PerGPUMinorAllocatableDevices)

	// 遍历所有 NVML 可见的 GPU 设备
	err := l.VisitDevices(func(i int, d nvdev.Device) error {
		// 若指定了索引列表，跳过不在列表中的 GPU
		if indices != nil && !slices.Contains(indices, i) {
			return nil
		}

		// 为此物理 GPU 的概念性可分配设备准备数据结构。
		thisGPUAllocatable := make(AllocatableDevices)

		// 获取 GPU 的详细信息
		gpuInfo, err := l.getGpuInfo(i, d)
		if err != nil {
			return fmt.Errorf("error getting info for GPU %v: %w", i, err)
		}

		// 构建完整 GPU 对应的可分配设备对象
		parentdev := &AllocatableDevice{
			Gpu: gpuInfo,
		}

		// 存储 gpuInfo 对象以便后续按 UUID 复用（设备健康监控等场景）
		l.gpuInfosByUUID[gpuInfo.UUID] = gpuInfo
		l.gpuUUIDbyMinor[gpuInfo.minor] = gpuInfo.UUID

		if featuregates.Enabled(featuregates.DynamicMIG) {
			// 尽力填充句柄缓存：存储完整 GPU UUID 到 NVML 设备句柄的映射。
			// 忽略失败（仅影响性能，不影响正确性）。
			if _, ret := l.DeviceGetHandleByUUID(gpuInfo.UUID); ret != nvml.SUCCESS {
				klog.Warningf("DeviceGetHandleByUUIDCached failed: %s", ret)
			}

			// 对此完整 GPU，检查所有 MIG 配置及其可能的放置位置。
			// 副作用：这会丰富 `gpuInfo` 的额外属性（如内存切片数、各 MIG 配置报告的最大容量等）。
			migspecs, err := l.inspectMigProfilesAndPlacements(gpuInfo, d)
			if err != nil {
				return fmt.Errorf("error getting MIG info for GPU %v: %w", i, err)
			}

			// 公告完整物理 GPU
			thisGPUAllocatable[gpuInfo.CanonicalName()] = parentdev

			// 将所有可能的动态 MIG 配置作为抽象可分配设备公告
			for _, migspec := range migspecs {
				dev := &AllocatableDevice{
					MigDynamic: migspec,
				}
				thisGPUAllocatable[migspec.CanonicalName()] = dev
			}

			perGPUAllocatable[gpuInfo.minor] = thisGPUAllocatable

			// 终止此函数——DynamicMIG 与静态 MIG 和 VFIO 直通互斥。
			return nil
		}

		// 静态 MIG 模式：发现该 GPU 上已存在的 MIG 设备
		migdevs, err := l.discoverMigDevicesByGPU(gpuInfo)
		if err != nil {
			return fmt.Errorf("error discovering MIG devices for GPU %q: %w", gpuInfo.CanonicalName(), err)
		}

		if featuregates.Enabled(featuregates.PassthroughSupport) {
			// 仅在未发现 MIG 设备时才允许 VFIO 设备
			klog.Infof("PassthroughSupport enabled, and %d MIG devices found", len(migdevs))
			gpuInfo.vfioEnabled = len(migdevs) == 0
		}

		if !gpuInfo.migEnabled {
			klog.Infof("Adding device %s to allocatable devices", gpuInfo.CanonicalName())
			// 此物理 GPU 未预配置静态 MIG 设备，公告完整 GPU 作为可分配设备，
			// 并终止此物理 GPU 的发现流程。
			thisGPUAllocatable[gpuInfo.CanonicalName()] = parentdev
			perGPUAllocatable[gpuInfo.minor] = thisGPUAllocatable
			return nil
		}

		// 处理预配置的静态 MIG 设备
		for _, mdev := range migdevs {
			klog.Infof("Adding MIG device %s to allocatable devices (parent: %s)", mdev.CanonicalName(), gpuInfo.CanonicalName())
			thisGPUAllocatable[mdev.CanonicalName()] = mdev
		}

		// MIG 模式已启用但未发现任何 MIG 设备——可能是配置错误
		if len(migdevs) == 0 {
			klog.Warningf("Physical GPU %s has MIG mode enabled but no configured MIG devices", gpuInfo.CanonicalName())
		}

		perGPUAllocatable[gpuInfo.minor] = thisGPUAllocatable
		return nil
	})

	if err != nil {
		return nil, fmt.Errorf("error visiting devices: %w", err)
	}

	return perGPUAllocatable, nil
}

// discoverMigDevicesByGPU 发现指定物理 GPU 上已存在的静态 MIG 设备。
//
// 参数：
//   - gpuInfo: 物理 GPU 的信息对象
//
// 返回值：
//   - AllocatableDeviceList: 发现到的 MIG 设备列表
//   - error:               发现过程中出现的错误
func (l deviceLib) discoverMigDevicesByGPU(gpuInfo *GpuInfo) (AllocatableDeviceList, error) {
	var devices AllocatableDeviceList
	// 获取该 GPU 上的所有 MIG 设备
	migs, err := l.getMigDevices(gpuInfo)
	if err != nil {
		return nil, fmt.Errorf("error getting MIG devices for GPU %q: %w", gpuInfo.CanonicalName(), err)
	}

	// 将每个 MIG 设备包装为 AllocatableDevice
	for _, migDeviceInfo := range migs {
		mig := &AllocatableDevice{
			MigStatic: migDeviceInfo,
		}
		devices = append(devices, mig)
	}
	return devices, nil
}

// discoverGPUByPCIBusID 根据 PCI 总线地址查找对应的完整 GPU 设备及其上的静态 MIG 设备。
// 用于 VFIO 直通模式下，当 GPU 从 vfio-pci 驱动切回 nvidia 驱动后重新发现设备。
//
// TODO: 需要 go-nvlib 提供工具函数来实现此功能。
//
// 参数：
//   - pcieBusID: PCI 总线地址（如 "0000:3b:00.0"）
//
// 返回值：
//   - *AllocatableDevice:    找到的完整 GPU 设备
//   - AllocatableDeviceList: 该 GPU 上的静态 MIG 设备列表
//   - error:                 发现过程中出现的错误
func (l deviceLib) discoverGPUByPCIBusID(pcieBusID string) (*AllocatableDevice, AllocatableDeviceList, error) {
	if err := l.Init(); err != nil {
		return nil, nil, err
	}
	defer l.alwaysShutdown()

	var gpu *AllocatableDevice
	var migs AllocatableDeviceList
	err := l.VisitDevices(func(i int, d nvdev.Device) error {
		gpuPCIBusID, err := d.GetPCIBusID()
		if err != nil {
			return fmt.Errorf("error getting PCIe bus ID for device %d: %w", i, err)
		}
		if gpuPCIBusID != pcieBusID {
			return nil // 不是目标设备，继续遍历
		}
		gpuInfo, err := l.getGpuInfo(i, d)
		if err != nil {
			return fmt.Errorf("error getting info for GPU %d: %w", i, err)
		}
		// 发现该 GPU 上的静态 MIG 设备
		migs, err = l.discoverMigDevicesByGPU(gpuInfo)
		if err != nil {
			return fmt.Errorf("error discovering MIG devices for GPU %q: %w", gpuInfo.CanonicalName(), err)
		}
		// 若未发现 MIG 设备，则允许 VFIO 设备
		gpuInfo.vfioEnabled = len(migs) == 0
		gpu = &AllocatableDevice{
			Gpu: gpuInfo,
		}
		return nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("error visiting devices: %w", err)
	}
	return gpu, migs, nil
}

// discoverVfioDevice 发现指定物理 GPU 对应的 VFIO 直通设备。
// 通过 PCI 总线地址在 nvpci 设备列表中匹配，构建 VfioDeviceInfo 对象。
//
// TODO: 需要 go-nvlib 提供工具函数来实现此功能。
//
// 参数：
//   - gpuInfo: 物理 GPU 的信息对象（包含 PCIe 总线地址等）
//
// 返回值：
//   - *AllocatableDevice: 找到的 VFIO 设备
//   - error:             发现过程中出现的错误
func (l deviceLib) discoverVfioDevice(gpuInfo *GpuInfo) (*AllocatableDevice, error) {
	gpus, err := l.nvpci.GetGPUs()
	if err != nil {
		return nil, fmt.Errorf("error getting GPU PCI devices: %w", err)
	}
	for idx, gpu := range gpus {
		if gpu.Address != gpuInfo.pcieBusID {
			continue // 不是目标设备
		}
		vfioDeviceInfo, err := l.getVfioDeviceInfo(idx, gpu)
		if err != nil {
			return nil, fmt.Errorf("error getting VFIO device info: %w", err)
		}
		vfioDeviceInfo.parent = gpuInfo
		return &AllocatableDevice{
			Vfio: vfioDeviceInfo,
		}, nil
	}
	return nil, fmt.Errorf("error discovering VFIO device by PCIe bus ID: %s", gpuInfo.pcieBusID)
}

// obliterateStaleMIGDevices 拆除所有不属于已完成的 Claim 的 MIG 设备。
// 遍历所有物理 GPU，对每个 GPU 上的 MIG 设备进行逐一检查：若设备名不在预期列表中，
// 则调用 deleteMigDevice 尝试销毁该设备。最后若该 GPU 上无剩余 MIG 设备，尝试禁用 MIG 模式。
//
// 此函数用于插件启动时的孤儿 MIG 设备清理。
//
// 参数：
//   - expectedDeviceNames: 预期存在的设备名称列表（来自已 PrepareCompleted 的 Claim）
//
// 返回值：
//   - error: 访问设备或销毁设备失败时返回错误
func (l deviceLib) obliterateStaleMIGDevices(expectedDeviceNames []DeviceName) error {
	err := l.VisitDevices(func(i int, d nvdev.Device) error {
		ginfo, err := l.getGpuInfo(i, d)
		if err != nil {
			return fmt.Errorf("error getting info for GPU %d: %w", i, err)
		}

		// 获取该 GPU 上的所有 MIG 设备
		migs, err := l.getMigDevices(ginfo)
		if err != nil {
			return fmt.Errorf("error getting MIG devices for GPU %d: %w", i, err)
		}

		// 检查每个 MIG 设备是否在预期列表中
		for _, mdi := range migs {
			name := mdi.CanonicalName()
			expected := slices.Contains(expectedDeviceNames, name)
			if !expected {
				// 设备不在预期列表中——尝试拆除
				klog.Warningf("Found unexpected MIG device (%s), attempt to tear down", name)
				if err := l.deleteMigDevice(mdi.LiveTuple()); err != nil {
					return fmt.Errorf("could not delete unexpected MIG device (%s): %w", name, err)
				}
			}
		}

		// 若在此 GPU 上未发现 MIG 设备，MIG 模式可能仍然启用（仅为空配置）。
		// 在这种情况下尝试禁用 MIG 模式。
		if err := l.maybeDisableMigMode(ginfo.UUID, d); err != nil {
			return fmt.Errorf("maybeDisableMigMode failed for GPU %s: %w", ginfo.UUID, err)
		}
		return nil
	})

	if err != nil {
		return fmt.Errorf("error visiting devices: %w", err)
	}
	return nil
}

// getGpuInfo 采集指定 GPU 的完整信息，包括硬件属性和 MIG 配置。
// 这是设备发现过程中最核心的数据采集函数。
//
// 采集的信息包括：
//   - 基础信息：UUID、minor 号、MIG 启用状态
//   - 内存信息：总显存大小
//   - 产品信息：型号名称、品牌、架构
//   - 驱动信息：驱动版本、CUDA 驱动版本
//   - PCI 信息：PCIe 总线地址、PCIe Root 属性
//   - 寻址模式：HMM/ATS/None 等
//   - MIG Profile 信息：所有可能的 GPU Instance Profile、对应的 Compute Instance profile 和可能的放置位置
//   - 健康状态：初始为 Healthy
//
// 参数：
//   - index:  GPU 在 NVML 设备列表中的索引
//   - device: go-nvlib 的 Device 对象
//
// 返回值：
//   - *GpuInfo: 填充完整的 GPU 信息对象
//   - error:   任一信息采集步骤失败时返回错误
func (l deviceLib) getGpuInfo(index int, device nvdev.Device) (*GpuInfo, error) {
	// 获取 GPU 的 minor 号（对应 /dev/nvidia${minor}）
	minor, ret := device.GetMinorNumber()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting minor number for device %d: %v", index, ret)
	}

	// 获取 GPU 的 UUID（全局唯一标识符）
	uuid, ret := device.GetUUID()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting UUID for device %d: %v", index, ret)
	}

	// 检查 MIG 模式是否已启用
	migEnabled, err := device.IsMigEnabled()
	if err != nil {
		return nil, fmt.Errorf("error checking if MIG mode enabled for device %d: %w", index, err)
	}

	// 获取显存信息（总量，单位字节）
	memory, ret := device.GetMemoryInfo()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting memory info for device %d: %v", index, ret)
	}

	// 获取产品名称（如 "NVIDIA A100-SXM4-80GB"）
	productName, ret := device.GetName()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting product name for device %d: %v", index, ret)
	}

	// 获取架构名称（如 "Ampere"）
	architecture, err := device.GetArchitectureAsString()
	if err != nil {
		return nil, fmt.Errorf("error getting architecture for device %d: %w", index, err)
	}

	// 获取品牌名称（如 "Tesla"、"GeForce"）
	brand, err := device.GetBrandAsString()
	if err != nil {
		return nil, fmt.Errorf("error getting brand for device %d: %w", index, err)
	}

	// 获取 CUDA 计算能力版本（如 "8.0"）
	cudaComputeCapability, err := device.GetCudaComputeCapabilityAsString()
	if err != nil {
		return nil, fmt.Errorf("error getting CUDA compute capability for device %d: %w", index, err)
	}

	// 获取 NVIDIA 驱动版本号
	driverVersion, ret := l.nvmllib.SystemGetDriverVersion()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting driver version: %w", err)
	}

	// 获取 CUDA 驱动版本号（原始整数值）
	cudaDriverVersion, ret := l.nvmllib.SystemGetCudaDriverVersion()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting CUDA driver version: %w", err)
	}

	// 获取 PCIe 总线地址（如 "0000:3b:00.0"）
	pcieBusID, err := device.GetPCIBusID()
	if err != nil {
		return nil, fmt.Errorf("error getting PCIe bus ID for device %d: %w", index, err)
	}

	// 获取设备支持的内存寻址模式。
	// 在一致性内存系统中，可能的模式包括：
	//   - HMM  (Hardware Memory Management，硬件内存管理)
	//   - ATS  (Address Translation Service，地址转换服务)
	//   - None (平台支持但当前未激活)
	//   - ""   (平台不支持)
	var addressingMode *string
	if mode, err := device.GetAddressingModeAsString(); err != nil {
		return nil, fmt.Errorf("error getting addressing mode for device %d: %w", index, err)
	} else if mode != "" {
		// 仅在模式非空时设置指针，空字符串表示不支持
		addressingMode = &mode
	}

	// 获取 PCIe Root 属性，用于 NUMA 感知调度
	var pcieRootAttr *deviceattribute.DeviceAttribute
	if attr, err := deviceattribute.GetPCIeRootAttributeByPCIBusID(pcieBusID); err == nil {
		pcieRootAttr = &attr
	} else {
		// PCIe Root 属性获取失败不影响正常运行，仅记录警告
		klog.Warningf("error getting PCIe root for device %d, continuing without attribute: %v", index, err)
	}

	// 枚举该 GPU 支持的所有 MIG Profile 及其可能的放置位置
	var migProfiles []*MigProfileInfo
	for i := 0; i < nvml.GPU_INSTANCE_PROFILE_COUNT; i++ {
		// 获取第 i 个 GPU Instance Profile 的信息
		giProfileInfo, ret := device.GetGpuInstanceProfileInfo(i)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			continue // 设备不支持此 Profile
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			continue // 无效参数（Profile 索引超出范围）
		}
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("error retrieving GpuInstanceProfileInfo for profile %d on GPU %v", i, uuid)
		}

		// 获取此 GPU Instance Profile 的所有可能放置位置
		giPossiblePlacements, ret := device.GetGpuInstancePossiblePlacements(&giProfileInfo)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			continue
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			continue
		}
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("error retrieving GpuInstancePossiblePlacements for profile %d on GPU %v", i, uuid)
		}

		// 将 NVML 放置位置转换为内部 MigDevicePlacement 对象
		var migDevicePlacements []*MigDevicePlacement
		for _, p := range giPossiblePlacements {
			mdp := &MigDevicePlacement{
				GpuInstancePlacement: p,
			}
			migDevicePlacements = append(migDevicePlacements, mdp)
		}

		// 对每个 GPU Instance Profile，遍历所有 Compute Instance Profile 和 Engine Profile 的组合
		for j := 0; j < nvml.COMPUTE_INSTANCE_PROFILE_COUNT; j++ {
			for k := 0; k < nvml.COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT; k++ {
				// 使用 go-nvlib 构建完整的 MIG Profile 对象
				migProfile, err := l.NewMigProfile(i, j, k, giProfileInfo.MemorySizeMB, memory.Total)
				if err != nil {
					return nil, fmt.Errorf("error building MIG profile from GpuInstanceProfileInfo for profile %d on GPU %v", i, uuid)
				}

				// 仅保留 G（GPU 切片数）与 C（计算切片数）相等的 Profile
				// 这是因为在当前 DRA 模型中，每个 MIG 设备必须是完整的 GPU 实例，
				// 不支持仅分配部分计算资源的"部分 CI"配置
				if migProfile.GetInfo().G != migProfile.GetInfo().C {
					continue
				}

				profileInfo := &MigProfileInfo{
					profile:    migProfile,
					placements: migDevicePlacements,
				}

				migProfiles = append(migProfiles, profileInfo)
			}
		}
	}

	// 构建完整的 GpuInfo 对象
	gpuInfo := &GpuInfo{
		UUID:                  uuid,
		minor:                 minor,
		migEnabled:            migEnabled,
		memoryBytes:           memory.Total,
		productName:           productName,
		brand:                 brand,
		architecture:          architecture,
		cudaComputeCapability: cudaComputeCapability,
		driverVersion:         driverVersion,
		cudaDriverVersion:     fmt.Sprintf("%v.%v", cudaDriverVersion/1000, (cudaDriverVersion%1000)/10),
		pcieBusID:             pcieBusID,
		pcieRootAttr:          pcieRootAttr,
		migProfiles:           migProfiles,
		health:                Healthy,
		addressingMode:        addressingMode,
	}

	return gpuInfo, nil
}

// enumerateGpuPciDevices 枚举 PCI 总线上的 GPU 设备，发现 VFIO 直通设备并构建对应的可分配设备对象。
//
// 参数：
//   - devs: 之前已发现的按 GPU minor 号分组的可分配设备映射
//
// 返回值：
//   - PerGPUMinorAllocatableDevices: 更新后的按 GPU minor 号分组的可分配设备映射（包含 VFIO 设备）
//   - error:                                                    枚举过程中出现的错误
func (l deviceLib) enumerateGpuPciDevices(devs PerGPUMinorAllocatableDevices) (PerGPUMinorAllocatableDevices, error) {
	perGPUAllocatable := make(PerGPUMinorAllocatableDevices)

	// 将输入的 `devs` 展平为单一映射
	all := make(AllocatableDevices)
	for _, devices := range perGPUAllocatable {
		maps.Copy(all, devices)
	}

	// 发现 PCI 设备（通过 nvpci 接口，不依赖 NVML）
	gpuPciDevices, err := l.nvpci.GetGPUs()
	if err != nil {
		return nil, fmt.Errorf("error getting GPU PCI devices: %w", err)
	}

	// 对每个发现的 PCI 设备，查找对应的完整 GPU 设备（以获取 minor 号、UUID 等信息）。
	for idx, pci := range gpuPciDevices {
		thisGPUAllocatable := make(AllocatableDevices)
		// 在已知设备列表中查找同 PCI 地址的 GPU
		parent := all.GetGPUByPCIeBusID(pci.Address)

		if parent == nil || !parent.Gpu.vfioEnabled {
			// 跳过未找到的 GPU 或未启用 VFIO 的 GPU
			continue
		}

		// 获取 VFIO 设备的详细信息
		vfioDeviceInfo, err := l.getVfioDeviceInfo(idx, pci)
		if err != nil {
			return nil, fmt.Errorf("error getting GPU info from PCI device: %w", err)
		}
		// 设置 VFIO 设备的父 GPU 引用
		vfioDeviceInfo.parent = parent.Gpu

		thisGPUAllocatable[vfioDeviceInfo.CanonicalName()] = &AllocatableDevice{
			Vfio: vfioDeviceInfo,
		}

		// 将 VFIO 设备归入父 GPU 的 minor 号分组
		perGPUAllocatable[parent.Gpu.minor] = thisGPUAllocatable
	}

	return perGPUAllocatable, nil
}

// getVfioDeviceInfo 从 PCI 设备信息中构建 VfioDeviceInfo 对象。
// VFIO 设备的 UUID 通过 PCI 地址基于 SHA1 命名空间生成（确定性）。
//
// 参数：
//   - idx:    PCI 设备在列表中的索引
//   - device: nvpci 的 NvidiaPCIDevice 对象
//
// 返回值：
//   - *VfioDeviceInfo: VFIO 设备信息对象
//   - error:          信息采集失败时返回错误
func (l deviceLib) getVfioDeviceInfo(idx int, device *nvpci.NvidiaPCIDevice) (*VfioDeviceInfo, error) {
	// 获取 PCIe Root 属性（用于 NUMA 感知调度）
	var pcieRootAttr *deviceattribute.DeviceAttribute
	attr, err := deviceattribute.GetPCIeRootAttributeByPCIBusID(device.Address)
	if err == nil {
		pcieRootAttr = &attr
	} else {
		klog.Warningf("error getting PCIe root for device %s, continuing without attribute: %v", device.Address, err)
	}

	// 获取可寻址内存大小（考虑 BAR 空间）
	_, memoryBytes := device.Resources.GetTotalAddressableMemory(true)

	// 构建 VfioDeviceInfo 对象
	vfioDeviceInfo := &VfioDeviceInfo{
		UUID:                   uuid.NewSHA1(uuid.NameSpaceDNS, []byte(device.Address)).String(),
		index:                  idx,
		productName:            device.DeviceName,
		pcieBusID:              device.Address,
		pcieRootAttr:           pcieRootAttr,
		deviceID:               fmt.Sprintf("0x%04x", device.Device),
		vendorID:               fmt.Sprintf("0x%04x", device.Vendor),
		numaNode:               device.NumaNode,
		iommuGroup:             device.IommuGroup,
		addressableMemoryBytes: memoryBytes,
	}

	return vfioDeviceInfo, nil
}

// getMigDevices 发现指定 GPU 上所有已存在的静态 MIG 设备。
// 通过遍历 NVML 的 MIG 设备索引，逐一获取 GPU Instance 和 Compute Instance 的信息，
// 并与之前采集的 MIG Profile 列表进行匹配，构建完整的 MigDeviceInfo 对象。
//
// 参数：
//   - gpuInfo: 物理 GPU 的信息对象
//
// 返回值：
//   - map[string]*MigDeviceInfo: 以 MIG 设备 UUID 为键的映射
//   - error:                   发现过程中出现的错误
func (l deviceLib) getMigDevices(gpuInfo *GpuInfo) (map[string]*MigDeviceInfo, error) {
	if !gpuInfo.migEnabled {
		// MIG 模式未启用，无 MIG 设备可发现
		return nil, nil
	}

	// 确保 NVML 已初始化
	shutdown, ret := l.ensureNVML()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("ensureNVML failed: %w", ret)
	}
	defer shutdown()

	// 获取父 GPU 的 NVML 设备句柄
	device, ret := l.DeviceGetHandleByUUID(gpuInfo.UUID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU device handle: %v", ret)
	}

	infos := make(map[string]*MigDeviceInfo)
	// 遍历该 GPU 上所有可能的 MIG 设备索引
	err := walkMigDevices(device, func(i int, migDevice nvml.Device) error {
		// 获取 MIG 设备所属的 GPU Instance ID
		giID, ret := migDevice.GetGpuInstanceId()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GPU instance ID for MIG device: %v", ret)
		}
		// 通过 GI ID 获取 GPU Instance 句柄
		gi, ret := device.GetGpuInstanceById(giID)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GPU instance for '%v': %v", giID, ret)
		}
		// 获取 GPU Instance 的详细信息
		giInfo, ret := gi.GetInfo()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GPU instance info for '%v': %v", giID, ret)
		}
		// 获取 MIG 设备所属的 Compute Instance ID
		ciID, ret := migDevice.GetComputeInstanceId()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting Compute instance ID for MIG device: %v", ret)
		}
		// 通过 CI ID 获取 Compute Instance 句柄
		ci, ret := gi.GetComputeInstanceById(ciID)
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting Compute instance for '%v': %v", ciID, ret)
		}
		// 获取 Compute Instance 的详细信息
		ciInfo, ret := ci.GetInfo()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting Compute instance info for '%v': %v", ciID, ret)
		}
		// 获取 MIG 设备的 UUID
		uuid, ret := migDevice.GetUUID()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting UUID for MIG device: %v", ret)
		}

		// 将 MIG 设备的 GI/CI 配置与之前采集的 MIG Profile 列表匹配
		var migProfile *MigProfileInfo
		var giProfileInfo *nvml.GpuInstanceProfileInfo
		var ciProfileInfo *nvml.ComputeInstanceProfileInfo
		for _, profile := range gpuInfo.migProfiles {
			profileInfo := profile.profile.GetInfo()
			// 通过 Profile ID 匹配 GPU Instance
			gipInfo, ret := device.GetGpuInstanceProfileInfo(profileInfo.GIProfileID)
			if ret != nvml.SUCCESS {
				continue
			}
			if giInfo.ProfileId != gipInfo.Id {
				continue
			}
			// 通过 Profile ID 匹配 Compute Instance
			cipInfo, ret := gi.GetComputeInstanceProfileInfo(profileInfo.CIProfileID, profileInfo.CIEngProfileID)
			if ret != nvml.SUCCESS {
				continue
			}
			if ciInfo.ProfileId != cipInfo.Id {
				continue
			}
			// 完全匹配
			migProfile = profile
			giProfileInfo = &gipInfo
			ciProfileInfo = &cipInfo
		}
		if migProfile == nil {
			return fmt.Errorf("error getting profile info for MIG device: %v", uuid)
		}

		// 构建放置位置信息
		placement := MigDevicePlacement{
			GpuInstancePlacement: giInfo.Placement,
		}

		// 构建 MigDeviceInfo 对象
		infos[uuid] = &MigDeviceInfo{
			UUID:           uuid,
			Profile:        migProfile.String(),
			ParentMinor:    gpuInfo.minor,
			ParentUUID:     gpuInfo.UUID,
			CIID:           int(ciInfo.Id),
			GIID:           int(giInfo.Id),
			PlacementStart: int(placement.Start),
			PlacementSize:  int(placement.Size),
			GiProfileID:    int(giProfileInfo.Id),
			parent:         gpuInfo,
			giProfileInfo:  giProfileInfo,
			gIInfo:         &giInfo,
			ciProfileInfo:  ciProfileInfo,
			cIInfo:         &ciInfo,
			pcieBusID:      gpuInfo.pcieBusID,
			pcieRootAttr:   gpuInfo.pcieRootAttr,
			health:         Healthy,
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("error enumerating MIG devices: %w", err)
	}

	if len(infos) == 0 {
		return nil, nil
	}

	return infos, nil
}

// walkMigDevices 遍历指定 NVML 设备上的所有 MIG 设备。
// 使用最大 MIG 设备计数作为遍历范围，跳过不存在的索引（ERROR_NOT_FOUND / ERROR_INVALID_ARGUMENT）。
//
// 参数：
//   - d: 父 GPU 的 NVML Device 句柄
//   - f: 对每个找到的 MIG 设备调用的回调函数
//
// 返回值：
//   - error: 遍历过程中 NVML 调用失败时返回错误
func walkMigDevices(d nvml.Device, f func(i int, d nvml.Device) error) error {
	// 获取该 GPU 支持的最大 MIG 设备数量
	count, ret := nvml.Device(d).GetMaxMigDeviceCount()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error getting max MIG device count: %v", ret)
	}

	for i := 0; i < count; i++ {
		// 尝试获取第 i 个 MIG 设备的句柄
		device, ret := d.GetMigDeviceHandleByIndex(i)
		if ret == nvml.ERROR_NOT_FOUND {
			continue // 该索引无 MIG 设备
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			continue // 无效索引
		}
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting MIG device handle at index '%v': %v", i, ret)
		}
		// 调用回调函数处理该 MIG 设备
		err := f(i, device)
		if err != nil {
			return err
		}
	}
	return nil
}

// setTimeSlice 通过 nvidia-smi 设置指定 GPU 的时间切片值。
// 时间切片控制 GPU 在多个上下文之间的调度间隔（以微秒为单位）。
//
// 参数：
//   - uuids:     目标 GPU 的 UUID 列表
//   - timeSlice: 时间切片值（微秒）
//
// 返回值：
//   - error: nvidia-smi 执行失败时返回错误
func (l deviceLib) setTimeSlice(uuids []string, timeSlice int) error {
	for _, uuid := range uuids {
		// 构造 nvidia-smi compute-policy 命令
		cmd := exec.Command(
			l.nvidiaSMIPath,
			"compute-policy",
			"-i", uuid,
			"--set-timeslice", fmt.Sprintf("%d", timeSlice))

		// 为使 nvidia-smi 正常运行，需要更新 LD_PRELOAD 以包含 libnvidia-ml.so.1 的路径。
		// 在容器环境中，nvidia-smi 可能无法通过默认路径找到 NVML 库。
		cmd.Env = setOrOverrideEnvvar(os.Environ(), "LD_PRELOAD", prependPathListEnvvar("LD_PRELOAD", l.driverLibraryPath))

		output, err := cmd.CombinedOutput()
		if err != nil {
			klog.Errorf("\n%v", string(output))
			return fmt.Errorf("error running nvidia-smi: %w", err)
		}
	}
	return nil
}

// setComputeMode 通过 nvidia-smi 设置指定 GPU 的计算模式。
// 计算模式控制 GPU 是否允许多个进程同时使用（Default / ExclusiveProcess / ExclusiveThread / Prohibited）。
//
// 参数：
//   - uuids: 目标 GPU 的 UUID 列表
//   - mode:  计算模式字符串（如 "Default"、"ExclusiveProcess"）
//
// 返回值：
//   - error: nvidia-smi 执行失败时返回错误
func (l deviceLib) setComputeMode(uuids []string, mode string) error {
	for _, uuid := range uuids {
		// 构造 nvidia-smi -c 命令
		cmd := exec.Command(
			l.nvidiaSMIPath,
			"-i", uuid,
			"-c", mode)

		// 为使 nvidia-smi 正常运行，需要更新 LD_PRELOAD 以包含 libnvidia-ml.so.1 的路径。
		cmd.Env = setOrOverrideEnvvar(os.Environ(), "LD_PRELOAD", prependPathListEnvvar("LD_PRELOAD", l.driverLibraryPath))

		output, err := cmd.CombinedOutput()
		if err != nil {
			klog.Errorf("\n%v", string(output))
			return fmt.Errorf("error running nvidia-smi: %w", err)
		}
	}
	return nil
}

// DeviceGetHandleByUUID 获取物理 GPU 的 NVML 设备句柄。
//
// 行为因 DynamicMIG 模式而异：
//   - DynamicMIG 未启用：直接调用 NVML 的 DeviceGetHandleByUUID()，无缓存。
//   - DynamicMIG 已启用：使用 devhandleByUUID 缓存，实现快速查找。
//     首次查询时调用 NVML 的 DeviceGetHandleByUUID() 并缓存结果，
//     后续相同 UUID 的查询直接返回缓存值。
//
// 注意：此函数仅用于物理完整 GPU，不用于 MIG 设备。
//
// 参数：
//   - uuid: GPU 的 UUID
//
// 返回值：
//   - nvml.Device: NVML 设备句柄
//   - nvml.Return:  NVML 返回码
func (l deviceLib) DeviceGetHandleByUUID(uuid string) (nvml.Device, nvml.Return) {
	// 确保 NVML 已初始化
	shutdown, ret := l.ensureNVML()
	if ret != nvml.SUCCESS {
		return nil, ret
	}
	defer shutdown()

	// 目前仅在 DynamicMIG 启用时使用带缓存句柄的长生命周期 NVML 会话。
	// 其他情况下不做缓存（将来可能需要）。
	if !featuregates.Enabled(featuregates.DynamicMIG) {
		return l.nvmllib.DeviceGetHandleByUUID(uuid)
	}

	// 检查缓存
	dev, exists := l.devhandleByUUID[uuid]
	if exists {
		// 缓存命中：直接返回
		return dev, nvml.SUCCESS
	}

	klog.V(6).Infof("DeviceGetHandleByUUID called for %s, cache miss", uuid)
	// 注意：此调用可能很慢（无句柄缓存时可达 O(10s)）。
	// 理论上此处需要请求合并策略（否则可能出现缓存击穿：
	// 一连串相同 UUID 的请求在远短于调用完成时间的时间窗口内到达，
	// 所有缓存未命中都执行此昂贵查询，尽管只需执行一次）。
	// 目前选择在程序启动期间预热缓存来解决此问题。
	// 假设完整 GPU 的集合是静态的且我们目前没有过期机制（使用长生命周期 map），
	// 该策略可以正常工作。
	t0 := time.Now()
	dev, ret = l.nvmllib.DeviceGetHandleByUUID(uuid)
	klog.V(7).Infof("t_device_get_handle_by_uuid %.3f s", time.Since(t0).Seconds())

	if ret != nvml.SUCCESS {
		return nil, ret
	}

	// 填充 `devhandleByUUID` 映射以实现快速 UUID 查找
	l.devhandleByUUID[uuid] = dev
	return dev, ret
}

// createMigDevice 通过 NVML 动态创建一个 MIG 设备实例（DynamicMIG 专用）。
//
// 创建步骤：
//  1. 获取父 GPU 的 NVML 设备句柄（通过缓存加速）
//  2. 若 MIG 模式未启用，则先启用 MIG 模式（SetMigMode）
//  3. 创建 GPU Instance（GI），指定 Profile 和 Placement
//  4. 在 GI 内创建 Compute Instance（CI）
//  5. 从 CI 获取 MIG 设备 UUID（直接法，避免 O(10s) 的遍历扫描）
//
// 调用方须持有长生命周期 NVML 会话（DynamicMIG 模式保证）。
//
// 参数：
//   - migspec: MIG 规范对象，包含父 GPU、Profile 和 Placement 信息
//
// 返回值：
//   - *MigDeviceInfo: 创建成功后的 MIG 设备信息对象
//   - error:       任一步骤失败时返回错误
func (l deviceLib) createMigDevice(migspec *MigSpec) (*MigDeviceInfo, error) {
	gpu := migspec.Parent
	profile := migspec.Profile
	placement := &migspec.Placement

	// 获取父 GPU 的 NVML 设备句柄（可能从缓存获取，也可能首次查询）
	tdhbu0 := time.Now()
	device, ret := l.DeviceGetHandleByUUID(gpu.UUID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU device handle: %v", ret)
	}
	klog.V(7).Infof("t_prep_create_mig_dev_get_dev_handle %.3f s", time.Since(tdhbu0).Seconds())

	// 用 NVML 句柄创建 go-nvlib 的 Device 对象（提供高层方法）
	tnd0 := time.Now()
	ndev, err := l.NewDevice(device)
	if err != nil {
		return nil, fmt.Errorf("error instantiating nvml dev: %w", err)
	}
	klog.V(7).Infof("t_prep_create_mig_dev_new_dev %.3f s", time.Since(tnd0).Seconds())

	// 检查 MIG 模式是否已启用
	// 注意：NVML 的 GetMigMode 区分当前模式和待生效模式，go-nvlib 尚未暴露此区分
	tcme0 := time.Now()
	migEnabled, err := ndev.IsMigEnabled()
	if err != nil {
		return nil, fmt.Errorf("error checking if MIG mode enabled for device %s: %w", ndev, err)
	}
	klog.V(7).Infof("t_prep_create_mig_dev_check_mig_enabled %.3f s", time.Since(tcme0).Seconds())

	logpfx := fmt.Sprintf("Create %s", migspec.CanonicalName())

	if !migEnabled {
		klog.V(6).Infof("%s: Attempting to enable MIG mode for to-be parent %s", logpfx, gpu.String())
		// 若设备较 A100 更新且当前未使用，则启用 MIG 模式。
		tem0 := time.Now()
		ret, activationStatus := device.SetMigMode(nvml.DEVICE_MIG_ENABLE)
		if ret != nvml.SUCCESS {
			// activationStatus 包含激活失败时的错误码
			klog.Warningf("%s: SetMigMode activationStatus (device %s): %s", logpfx, gpu.String(), activationStatus)
			return nil, fmt.Errorf("error enabling MIG mode for device %s: %v", gpu.String(), ret)
		}
		klog.V(1).Infof("%s: MIG mode now enabled for device %s, t_enable_mig %.3f s", logpfx, gpu.String(), time.Since(tem0).Seconds())
	} else {
		klog.V(6).Infof("%s: MIG mode already enabled for device %s", logpfx, gpu.String())
	}

	profileInfo := profile.GetInfo()

	// 获取 GPU Instance Profile 的 NVML 信息
	tcgigi0 := time.Now()
	giProfileInfo, ret := device.GetGpuInstanceProfileInfo(profileInfo.GIProfileID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU instance profile info for '%v': %v", profile, ret)
	}

	// 使用指定的 Placement 创建 GPU Instance
	gi, ret := device.CreateGpuInstanceWithPlacement(&giProfileInfo, placement)

	// NVML API 歧义：当请求一个特定放置位置且该位置已有 GPU 实例时，
	// NVML 返回 NVML_ERROR_INSUFFICIENT_RESOURCES 而非 NVML_ERROR_ALREADY_EXISTS。
	// 似乎无法仅通过错误码可靠地区分"已存在"和"被其他资源阻挡"，
	// 必须检查设备状态。此外，不确定是否可以安全地复用已存在的实例。
	// 当到达此处时，我们不应期望该实例已存在——
	// 重试创建前应先执行销毁。因此，当前直接返回错误，不区分"已存在"和其他故障。
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error creating GPU instance for '%s': %v", migspec.CanonicalName(), ret)
	}

	// 获取刚创建的 GPU Instance 的信息
	giInfo, ret := gi.GetInfo()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU instance info for '%s': %v", migspec.CanonicalName(), ret)
	}

	// 获取 Compute Instance Profile 的 NVML 信息
	ciProfileInfo, ret := gi.GetComputeInstanceProfileInfo(profileInfo.CIProfileID, profileInfo.CIEngProfileID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting Compute instance profile info for '%v': %v", profile, ret)
	}

	// 在 GPU Instance 内创建 Compute Instance
	ci, ret := gi.CreateComputeInstance(&ciProfileInfo)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error creating Compute instance for '%v': %v", profile, ret)
	}

	// 获取刚创建的 Compute Instance 的信息
	ciInfo, ret := ci.GetInfo()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting GPU instance info for '%v': %v", profile, ret)
	}
	klog.V(6).Infof("t_prep_create_mig_dev_cigi %.3f s", time.Since(tcgigi0).Seconds())

	// 注意：获取刚创建的 MIG 设备 UUID 时，
	// 常见算法需要遍历父 GPU 上的所有 MIG 设备来匹配刚创建的 GIID 和 CIID。
	// 虽然这能正确工作，但实测发现在负载下 NVML 的"遍历所有 MIG 设备"API 调用
	// 可能需要 O(10s) 时间。
	// UUID 也可通过先从 CI 信息获取 MIG 设备句柄再查询 UUID 来获得。
	// MIG 设备句柄与 CI 在 NVML 中是 1:1 对应的，
	// 一旦 CI 已知，即可直接获取 MIG 设备句柄和 UUID，无需遍历索引。
	uuid, ret := ciInfo.Device.GetUUID()
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("error getting UUID from CI info/device for CI %d: %v", ciInfo.Id, ret)
	}

	// 构建 MigDeviceInfo 对象
	// TODO: 此处可能需要与新的 MigLiveTuple 和 MigSpecTuple 类型进行统一，目前有些混乱。
	migDevInfo := &MigDeviceInfo{
		UUID:           uuid,
		CIID:           int(ciInfo.Id),
		GIID:           int(giInfo.Id),
		ParentMinor:    gpu.minor,
		ParentUUID:     gpu.UUID,
		Profile:        profile.String(),
		PlacementStart: int(placement.Start),
		PlacementSize:  int(placement.Size),
		GiProfileID:    int(giProfileInfo.Id),
		gIInfo:         &giInfo,
		cIInfo:         &ciInfo,
		parent:         gpu,
	}

	klog.V(6).Infof("%s: MIG device created on %s: %+v", logpfx, gpu.String(), migDevInfo.LiveTuple())
	return migDevInfo, nil
}

// deleteMigDevice 通过 NVML 销毁一个已存在的 MIG 设备实例（DynamicMIG 专用）。
//
// 销毁顺序严格遵循 NVML 层次：先销毁 CI，再销毁 GI（父不能在子存在时被销毁）。
// 所有 MIG 设备销毁后，若父 GPU 无剩余 MIG 设备，则尝试禁用 MIG 模式（maybeDisableMigMode）。
// 调用方须持有长生命周期 NVML 会话（DynamicMIG 模式保证）。
//
// 参数：
//   - miglt: MIG 设备的实时元组（包含父 GPU UUID、GI ID、CI ID 和 MIG UUID）
//
// 返回值：
//   - error: 任一步骤失败时返回错误
func (l deviceLib) deleteMigDevice(miglt *MigLiveTuple) error {
	parentUUID := miglt.ParentUUID
	giId := miglt.GIID
	ciId := miglt.CIID

	t0 := time.Now()
	migStr := fmt.Sprintf("MIG(parent: %s, %+v)", parentUUID, miglt)
	klog.V(6).Infof("Delete %s", migStr)

	// 获取父 GPU 的 NVML 设备句柄
	parentNvmlDev, ret := l.DeviceGetHandleByUUID(parentUUID)
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error getting device from UUID '%v': %v", parentUUID, ret)
	}

	// 销毁顺序至关重要：必须先销毁计算实例（CI），再销毁 GPU 实例（GI），
	// 因为计算实例创建在 GPU 实例内部，父 GI 无法在子 CI 存在时销毁。
	gi, gires := parentNvmlDev.GetGpuInstanceById(giId)

	// NVML 文档将此错误记录为"若设备未启用 MIG 模式"——对于不太可能出现的
	// 情况（MIG 模式被带外禁用？），应视为删除成功。
	if gires == nvml.ERROR_NOT_SUPPORTED {
		klog.Infof("Delete %s: GetGpuInstanceById yielded ERROR_NOT_SUPPORTED: MIG disabled, treat as success", migStr)
		return nil
	}

	// 其他 NVML 错误（UNINITIALIZED、INVALID_ARGUMENT、NO_PERMISSION）
	if gires != nvml.SUCCESS && gires != nvml.ERROR_NOT_FOUND {
		return fmt.Errorf("error getting GPU instance handle for MIG device: %v", ret)
	}

	if gires == nvml.ERROR_NOT_FOUND {
		// GI 不存在：假设也不存在 CI（因为存在 GI>CI 层级关系），
		// 继续尝试禁用 MIG 模式
		klog.Infof("Delete %s: GI was not found skip CI cleanup", migStr)
		if err := l.maybeDisableMigMode(parentUUID, parentNvmlDev); err != nil {
			return fmt.Errorf("failed maybeDisableMigMode: %w", err)
		}
		return nil
	}

	// GI 有效：尝试获取 CI 句柄
	ci, cires := gi.GetComputeInstanceById(ciId)

	// 此处不可能是 ERROR_NOT_SUPPORTED，但可能是 UNINITIALIZED、
	// INVALID_ARGUMENT 或 NO_PERMISSION——这三种情况值得报错退出。
	if cires != nvml.SUCCESS && cires != nvml.ERROR_NOT_FOUND {
		return fmt.Errorf("error getting Compute instance handle for MIG device %s: %v", migStr, ret)
	}

	// 之前的部分清理可能已删除该 GPU 实例中的 CI。实践中已观察到此情况。
	// 忽略，继续销毁 GPU Instance。
	if cires == nvml.ERROR_NOT_FOUND {
		klog.Infof("Delete %s: CI not found, ignore", migStr)
	} else {
		// 销毁 Compute Instance
		ret := ci.Destroy()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error destroying Compute instance: %v", ret)
		}
	}

	// 销毁 GPU Instance
	// 例如可能因"In use by another client"失败，此时返回错误。
	// 我们可能只完成了部分清理（CI 已销毁；实践中已观察到）。

	// 注意：此操作可能耗时 O(1s)。在支持大量 MIG 设备和高作业吞吐量的机器上，
	// 这可能变得明显。
	// 在压力测试中观察到 prep/unprep 锁获取时间超过 60 秒，
	// 请求堆积时在 10 秒后超时。
	ret = gi.Destroy()
	if ret != nvml.SUCCESS {
		return fmt.Errorf("error destroying GPU Instance: %v", ret)
	}
	klog.V(6).Infof("t_delete_mig_device %.3f s", time.Since(t0).Seconds())

	// 尝试禁用 MIG 模式（仅在该 GPU 无剩余 MIG 设备时生效）
	if err := l.maybeDisableMigMode(parentUUID, parentNvmlDev); err != nil {
		return fmt.Errorf("failed maybeDisableMigMode: %w", err)
	}

	return nil
}

// maybeDisableMigMode 在 GPU 上无剩余 MIG 设备时尝试禁用 MIG 模式。
// 这是一种"尽力而为"的操作——若禁用失败（如设备正被其他客户端使用），
// 仅返回错误，不会影响数据一致性。
//
// 参数：
//   - uuid:    父 GPU 的 UUID
//   - nvmldev: 父 GPU 的 NVML 设备句柄
//
// 返回值：
//   - error: MIG 模式禁用失败时返回错误
func (l deviceLib) maybeDisableMigMode(uuid string, nvmldev nvml.Device) error {
	// 期望父 GPU 已存在于 `l.gpuInfosByUUID` 中
	gpu, ok := l.gpuInfosByUUID[uuid]
	if !ok {
		// TODO: 这是编程错误——应该用 panic 代替返回 error
		return fmt.Errorf("uuid not in gpuInfosByUUID: %s", uuid)
	}

	// 检查该 GPU 上是否还有 MIG 设备
	migs, err := l.getMigDevices(gpu)
	if err != nil {
		return fmt.Errorf("error getting MIG devices for %s: %w", gpu.String(), err)
	}

	if len(migs) > 0 {
		// 仍有 MIG 设备存在，保持 MIG 模式启用
		klog.V(6).Infof("Leaving MIG mode enabled for device %s (currently present MIG devices: %d)", gpu.String(), len(migs))
		return nil
	}

	// 无剩余 MIG 设备，尝试禁用 MIG 模式
	klog.V(6).Infof("Attempting to disable MIG mode for device %s", gpu.String())
	t0 := time.Now()
	ret, activationStatus := nvmldev.SetMigMode(nvml.DEVICE_MIG_DISABLE)
	klog.V(6).Infof("t_disable_mig %.3f s", time.Since(t0).Seconds())
	if ret != nvml.SUCCESS {
		// activationStatus 包含激活失败时的错误码
		klog.Warningf("SetMigMode activationStatus (device %s): %s", gpu.String(), activationStatus)
		// 也可以将其记录为错误并继续，期望在下次 Checkpoint 写入时设备能正确关联。
		// 这可能不是好主意。
		return fmt.Errorf("error disabling MIG mode for device %s: %v", gpu.String(), ret)
	}
	// 注意：到达此处时，禁用 MIG 模式可能仍然失败了。
	// `activationStatus` 可能反映"被另一个客户端使用"。
	klog.V(1).Infof("Called nvml.SetMigMode(nvml.DEVICE_MIG_DISABLE) for device %s, got activationStatus: %s", gpu.String(), activationStatus)
	return nil
}

// inspectMigProfilesAndPlacements 返回某物理 GPU 上所有可能的 MIG 配置的扁平列表。
// 具体而言，此函数发现所有可能的 Profile，然后确定每个 Profile 的可能放置位置。
// 同时，此函数还会丰富 gpuInfo 对象的 DynamicMIG 容量信息。
//
// 参数：
//   - gpuInfo: 物理 GPU 的信息对象
//   - device:  go-nvlib 的 Device 对象
//
// 返回值：
//   - []*MigSpec: 所有 MIG 规范对象（Profile+Placement 组合）列表
//   - error:      遍历过程中出现的错误
func (l deviceLib) inspectMigProfilesAndPlacements(gpuInfo *GpuInfo, device nvdev.Device) ([]*MigSpec, error) {
	var infos []*MigSpec

	// 用于跟踪各容量维度的最大值
	maxCapacities := make(PartCapacityMap)
	maxMemSlicesConsumed := 0

	err := device.VisitMigProfiles(func(migProfile nvdev.MigProfile) error {
		// 跳过 G != C 的 Profile（仅支持完整的 GPU 实例）
		if migProfile.GetInfo().C != migProfile.GetInfo().G {
			return nil
		}

		// 跳过 rev1 的 Compute Instance Profile（与基础 Profile 互斥）
		if migProfile.GetInfo().CIProfileID == nvml.COMPUTE_INSTANCE_PROFILE_1_SLICE_REV1 {
			return nil
		}

		// 获取 GPU Instance Profile 信息
		giProfileInfo, ret := device.GetGpuInstanceProfileInfo(migProfile.GetInfo().GIProfileID)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			return nil
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			return nil
		}
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GI Profile info for MIG profile %v: %w", migProfile, ret)
		}

		// 获取该 Profile 的所有可能放置位置
		giPlacements, ret := device.GetGpuInstancePossiblePlacements(&giProfileInfo)
		if ret == nvml.ERROR_NOT_SUPPORTED {
			return nil
		}
		if ret == nvml.ERROR_INVALID_ARGUMENT {
			return nil
		}
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting GI possible placements for MIG profile %v: %w", migProfile, ret)
		}

		// 为每个放置位置创建 MigSpec 对象
		for _, giPlacement := range giPlacements {
			mi := &MigSpec{
				Parent:        gpuInfo,
				Profile:       migProfile,
				GIProfileInfo: giProfileInfo,
				Placement:     giPlacement,
			}
			infos = append(infos, mi)

			// 假设最大的 MIG 配置消耗所有显存切片，
			// 因此可以通过比较所有 MigPP 对象的 Size 属性并取最大值来确定内存切片数。
			maxMemSlicesConsumed = max(maxMemSlicesConsumed, int(giPlacement.Size))

			// 对所有 MIG 配置，识别每个容量维度的最大值。
			// 它们可能对应同一个 Profile。
			caps := mi.PartCapacities()
			for name, cap := range caps {
				setMax(maxCapacities, name, cap)
			}
		}
		return nil
	})

	klog.V(1).Infof("%s: Per-capacity maximum across all MIG profiles+placements: %v", gpuInfo.String(), maxCapacities)
	klog.V(1).Infof("%s: Largest MIG placement size seen (maxMemSlicesConsumed): %d", gpuInfo.String(), maxMemSlicesConsumed)

	if err != nil {
		return nil, fmt.Errorf("error visiting MIG profiles: %w", err)
	}

	// 就地修改 `gpuInfo` 对象，用 DynamicMIG 的容量信息进行丰富。
	// 假设最大的 MIG Profile 消耗所有内存切片，
	// 因此 maxMemSlicesConsumed 即为 memSliceCount。
	gpuInfo.AddDetailAfterWalkingMigProfiles(maxCapacities, maxMemSlicesConsumed)
	return infos, nil
}

// FindMigDevBySpec 测试当前物理 GPU 上是否存在由指定 MigSpecTuple（配置+放置）
// 描述的 MIG 设备。
//
// 遍历父 GPU 上所有可能的 MIG 设备索引，对每个有效设备获取其 GI 信息，
// 比较 Profile ID 和 Placement Start 是否匹配。
//
// 参数：
//   - ms: MIG 规范元组，包含父 GPU minor 号、Profile ID 和 Placement Start
//
// 返回值：
//   - *MigLiveTuple: 若找到匹配设备，返回其实时元组（包含 GIID、CIID、MigUUID 等）；若未找到返回 nil
//   - error:        NVML API 调用失败时返回非 nil 错误
func (l deviceLib) FindMigDevBySpec(ms *MigSpecTuple) (*MigLiveTuple, error) {
	// 通过 minor 号反查父 GPU 的 UUID
	parentUUID := l.gpuUUIDbyMinor[ms.ParentMinor]
	// 获取父 GPU 的 NVML 设备句柄
	parent, ret := l.DeviceGetHandleByUUID(parentUUID)
	if ret != nvml.SUCCESS {
		return nil, fmt.Errorf("could not get device handle by UUID for %s", parentUUID)
	}

	// 获取该 GPU 支持的最大 MIG 设备数量
	count, _ := parent.GetMaxMigDeviceCount()

	for i := range count {
		// 尝试获取第 i 个 MIG 设备的句柄
		migHandle, ret := parent.GetMigDeviceHandleByIndex(i)
		if ret != nvml.SUCCESS {
			klog.Infof("GetMigDeviceHandleByIndex ret not success")
			// 槽位为空或无效：视为该设备当前不存在
			continue
		}

		// 获取该 MIG 设备所属的 GPU Instance ID
		giId, ret := migHandle.GetGpuInstanceId()
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("failed to get GI ID: %v", ret)
		}

		// 通过 GI ID 获取 GPU Instance 句柄
		giHandle, ret := parent.GetGpuInstanceById(giId)
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("failed to get GI handle for ID %d: %v", giId, ret)
		}

		// 获取 GPU Instance 的详细信息
		giInfo, ret := giHandle.GetInfo()
		if ret != nvml.SUCCESS {
			return nil, fmt.Errorf("failed to get GI info: %v", ret)
		}

		klog.V(7).Infof("FindMigDevBySpec: saw MIG dev with profile id %d and placement start %d", giInfo.ProfileId, giInfo.Placement.Start)

		// 比较 Profile ID
		if int(giInfo.ProfileId) != ms.ProfileID {
			klog.V(7).Infof("profile ID mismatch: looking for %d", ms.ProfileID)
			continue
		}

		// 比较 Placement Start
		if int(giInfo.Placement.Start) != ms.PlacementStart {
			klog.V(7).Infof("placement start mismatch: looking for %d", ms.PlacementStart)
			continue
		}

		klog.V(4).Infof("FindMigDevBySpec: match found for profile ID %d and placement start %d", giInfo.ProfileId, giInfo.Placement.Start)

		// 获取 Compute Instance ID（在本插件管理方式中通常为 0）
		// 若获取失败，说明 MIG 设备可能仅部分创建（仅有 GI 无 CI），此时继续处理
		ciId := 0
		ciId, ret = migHandle.GetComputeInstanceId()
		if ret != nvml.SUCCESS {
			klog.V(4).Infof("FindMigDevBySpec(): failed to get CI ID: %v", ret)
		}

		// 获取 MIG 设备的 UUID
		uuid := ""
		uuid, ret = migHandle.GetUUID()
		if ret != nvml.SUCCESS {
			klog.V(4).Infof("FindMigDevBySpec(): failed to get MIG UUID: %v", ret)
		}

		// 找到匹配配置的设备，返回其句柄。
		// 后续删除操作中 CIID 和 UUID 为零值也是可接受的，
		// 因为对于部分准备的 MIG 设备，CI 可能尚未创建。
		mlt := MigLiveTuple{
			ParentMinor: ms.ParentMinor,
			ParentUUID:  parentUUID,
			GIID:        giId,
			CIID:        ciId,
			MigUUID:     uuid,
		}

		klog.Infof("FindMigDevBySpec result: %+v", mlt)
		return &mlt, nil
	}

	klog.Infof("Iterated through all potential MIG devs -- no candidate found")
	return nil, nil
}

// setMax 就地修改映射 `m`：若当前 QualifiedName 尚不存在，或新值大于已有值则插入/更新。
// 这用于在遍历所有 MIG 配置时追踪各容量维度的最大值。
//
// 参数：
//   - m: 容量映射
//   - k: 容量维度名称
//   - v: 新的容量值
func setMax(m map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, k resourceapi.QualifiedName, v resourceapi.DeviceCapacity) {
	if cur, ok := m[k]; !ok || v.Value.Value() > cur.Value.Value() {
		// 键不存在或新值更大：更新映射
		m[k] = v
	}
}
