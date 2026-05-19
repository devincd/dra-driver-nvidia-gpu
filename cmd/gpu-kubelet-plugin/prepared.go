/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

// prepared 包定义了已 Prepare（准备完毕）的 GPU 设备的数据结构和方法。
//
// 核心概念区分：
//   - AllocatableDevice：可分配的设备（插件启动时发现并公告到 ResourceSlice 中），包含 GpuInfo/MigSpec/MigDeviceInfo/VfioDeviceInfo
//   - PreparedDevice：已 Prepare 的设备实例（NodePrepareResources 成功后持久化到 Checkpoint 中），包含 PreparedGpu/PreparedMigDevice/PreparedVfioDevice
//
// 两者的关键区别在于 MIG 设备的表示方式：
//   - AllocatableDevice 中的 MigDynamic 是抽象描述（无 UUID，无 GI/CI ID，只是"可以创建这种配置"的描述）
//   - PreparedDevice 中的 Mig 是已物化的具体实例（有 UUID、有 GI/CI ID，可用于 Unprepare 时销毁）
//
// 本文件还定义了 PreparedDeviceGroup（设备组）和多层级的 UUID 提取方法，
// 用于 MPS、TimeSlicing 等组件对一组设备进行统一操作。
package main

import (
	"slices"

	"k8s.io/dynamic-resource-allocation/kubeletplugin"
)

// PreparedMigDeviceType 是已 Prepare 的 MIG 设备（无论静态创建还是动态创建）在 PreparedDevice 中的类型标识。
// 对外统一使用 "mig"，与 AllocatableDevice 内部区分 "mig"/"migdyn" 不同。
// 原因：Prepare 完成后，静态和动态 MIG 设备都已物化（有 UUID、GI/CI ID），
// 不再需要区分创建方式，统一用 "mig" 表达"已存在的 MIG 实例"。
const PreparedMigDeviceType = "mig"

// PreparedDeviceList 是同一个 PreparedDeviceGroup 中已 Prepare 设备的列表。
// 例如，一个 Claim 请求了 2 个 GPU，则 PreparedDeviceList 中有 2 个 PreparedDevice 元素。
type PreparedDeviceList []PreparedDevice

// PreparedDevices 是一个 ResourceClaim 中所有已 Prepare 设备组的列表。
// 每个组（PreparedDeviceGroup）对应一套 OpaqueDeviceConfig（GpuConfig/MigDeviceConfig/VfioDeviceConfig）。
// 同一组内的设备共享配置状态（如 MPS 守护进程 ID、CDI 容器编辑项）。
// 一个 Claim 通常只有一个组，但若 Claim 中包含多个 request 且每个 request 匹配不同 Config，
// 则会有多个组。
type PreparedDevices []*PreparedDeviceGroup

// PreparedDevice 表示一个已完成 Prepare 的具体设备实例，序列化后持久化到 Checkpoint。
//
// 与 AllocatableDevice 的区别：
//   - AllocatableDevice 中的 MigDynamic 是抽象描述（无 UUID，无 GI/CI ID）
//   - PreparedDevice 中的 Mig 是已物化的具体实例（有 UUID、有 GI/CI ID，可用于 Unprepare 时销毁）
//
// 三个字段同一时刻只有一个非 nil，通过 Type() 方法判断具体类型。
type PreparedDevice struct {
	// Gpu 表示已 Prepare 的完整物理 GPU
	Gpu *PreparedGpu `json:"gpu"`
	// Mig 表示已 Prepare 的具体 MIG 设备实例（含物理创建信息，用于 Unprepare 时销毁）
	Mig  *PreparedMigDevice  `json:"mig"`
	// Vfio 表示已切换到 vfio-pci 驱动的 GPU 直通设备
	Vfio *PreparedVfioDevice `json:"vfio,omitempty"`
}

// PreparedGpu 记录已 Prepare 的完整物理 GPU 信息。
// Info 用于日志和清理（包含 GPU 的 UUID、型号等属性）。
// Device 用于向 kubelet 返回 CDI 设备标识符（如 "nvidia.com/gpu=GPU-xxx"），
// kubelet 将此标识符写入容器的 CDI 注解中，容器运行时据此注入 GPU 设备。
type PreparedGpu struct {
	Info   *GpuInfo              `json:"info"`   // GPU 属性信息，用于日志和 UUID 查询
	Device *kubeletplugin.Device `json:"device"` // CDI 设备标识符，返回给 kubelet
}

// PreparedMigDevice 记录已 Prepare 的具体 MIG 设备实例。
// Concrete 包含物理创建信息（GI ID、CI ID、UUID），是 Unprepare 时通过 NVML 销毁该设备的依据。
// 使用 (GI ID, CI ID) 二元组定位 MIG 设备，而非仅依赖 UUID，
// 因为 NVML 的 deleteMigDevice() API 通过 GI/CI ID 操作，UUID 仅作为辅助校验。
type PreparedMigDevice struct {
	// Concrete 是该 MIG 设备的物理创建信息（静态/动态均填充此字段）。
	// 包含 ParentMinor（父 GPU minor 号）、ParentUUID（父 GPU UUID）、
	// GIID（GPU Instance ID）、CIID（Compute Instance ID）、MigUUID（MIG 设备 UUID）。
	// 这些信息在 Unprepare 时用于通过 NVML API 精确定位并销毁该 MIG 设备。
	Concrete *MigLiveTuple         `json:"concrete"`
	// Device 是 CDI 设备标识符，返回给 kubelet，格式如 "nvidia.com/mig=MIG-xxx"
	Device   *kubeletplugin.Device `json:"device"`
}

// PreparedVfioDevice 记录已切换到 vfio-pci 驱动的 GPU 信息。
// Unprepare 时通过 Info.pcieBusID 定位设备并调用 VfioPciManager.Unconfigure 切回 nvidia 驱动。
type PreparedVfioDevice struct {
	Info   *VfioDeviceInfo       `json:"info"`   // VFIO 设备属性信息，包含 PCIe Bus ID、IOMMU 组等
	Device *kubeletplugin.Device `json:"device"` // CDI 设备标识符，返回给 kubelet
}

// PreparedDeviceGroup 将一组共享同一 OpaqueDeviceConfig 的已 Prepare 设备聚合在一起。
// ConfigState 保存该组的运行时配置状态（MPS 守护进程 ID、CDI 挂载编辑项），
// 供 Unprepare 时正确清理对应资源：
//   - MpsControlDaemonID 非空 → 需要调用 MpsManager 停止对应的 MPS 守护进程
//   - containerEdits 非空 → CDI spec 中包含了额外的容器编辑（如 MPS 的 shm 挂载和环境变量）
type PreparedDeviceGroup struct {
	// Devices 是该组内所有已 Prepare 的设备列表
	Devices     PreparedDeviceList `json:"devices"`
	// ConfigState 保存该设备组的运行时配置状态，用于 Unprepare 时清理资源
	ConfigState DeviceConfigState  `json:"configState"`
}

// Type 返回该设备的类型字符串。
// 通过检查哪个字段非 nil 来确定设备类型。
// 返回值：
//   - GpuDeviceType ("gpu"): 完整物理 GPU
//   - PreparedMigDeviceType ("mig"): MIG 设备（静态或动态，统一为 "mig"）
//   - VfioDeviceType ("vfio"): VFIO 直通设备
//   - UnknownDeviceType ("unknown"): 未知类型（不应出现）
func (d PreparedDevice) Type() string {
	if d.Gpu != nil {
		return GpuDeviceType
	}
	if d.Mig != nil {
		return PreparedMigDeviceType
	}
	if d.Vfio != nil {
		return VfioDeviceType
	}
	return UnknownDeviceType
}

// CanonicalName 返回该设备的规范名称，用于日志和 Checkpoint 查询。
// 根据设备类型委托给对应子结构的 CanonicalName 方法。
// 对于未知类型会 panic（属于编程错误，不应在运行时出现）。
func (d *PreparedDevice) CanonicalName() string {
	switch d.Type() {
	case GpuDeviceType:
		// GPU 设备使用 GpuInfo.CanonicalName()，格式：gpu-<minor>
		return d.Gpu.Info.CanonicalName()
	case PreparedMigDeviceType:
		// MIG 设备使用 DeviceName（已在 Prepare 时赋值的规范名称）
		return d.Mig.Device.DeviceName
	case VfioDeviceType:
		// VFIO 设备使用 VfioDeviceInfo.CanonicalName()，格式：gpu-vfio-<index>
		return d.Vfio.Info.CanonicalName()
	}
	panic("unexpected type for AllocatableDevice")
}

// Gpus 过滤返回列表中所有完整物理 GPU 设备。
// 用于从混合设备列表中提取 GPU 子集，供 MPS/TimeSlicing 等组件使用。
func (l PreparedDeviceList) Gpus() PreparedDeviceList {
	var devices PreparedDeviceList
	for _, device := range l {
		if device.Type() == GpuDeviceType {
			devices = append(devices, device)
		}
	}
	return devices
}

// MigDevices 过滤返回列表中所有 MIG 设备。
// 静态和动态 MIG 设备统一类型为 PreparedMigDeviceType ("mig")。
func (l PreparedDeviceList) MigDevices() PreparedDeviceList {
	var devices PreparedDeviceList
	for _, device := range l {
		if device.Type() == PreparedMigDeviceType {
			devices = append(devices, device)
		}
	}
	return devices
}

// VfioDevices 过滤返回列表中所有 VFIO 直通设备。
func (l PreparedDeviceList) VfioDevices() PreparedDeviceList {
	var devices PreparedDeviceList
	for _, device := range l {
		if device.Type() == VfioDeviceType {
			devices = append(devices, device)
		}
	}
	return devices
}

// GetDevices 返回所有设备组中的 kubeletplugin.Device 列表，用于向 kubelet 返回 Prepare 结果。
// 遍历所有设备组，收集每个组中的 Device 对象，组成扁平列表。
func (d PreparedDevices) GetDevices() []kubeletplugin.Device {
	var devices []kubeletplugin.Device
	for _, group := range d {
		// 将每个设备组中的设备列表展开
		devices = append(devices, group.GetDevices()...)
	}
	return devices
}

// GetDeviceNames 返回所有设备组中的设备规范名称列表，用于日志记录和 Unprepare 流程。
// 只返回 GPU 和 MIG 设备名称，不包含 VFIO 设备（VFIO 设备的 Unprepare 走不同的代码路径）。
func (d PreparedDevices) GetDeviceNames() []DeviceName {
	var names []DeviceName
	for _, group := range d {
		names = append(names, group.GetDeviceNames()...)
	}
	return names
}

// GetDevices 返回该设备组中的 kubeletplugin.Device 列表。
// 根据设备类型提取对应的 Device 对象：
//   - GPU: 从 PreparedGpu.Device 提取
//   - MIG: 从 PreparedMigDevice.Device 提取
//   - VFIO: 从 PreparedVfioDevice.Device 提取
func (g *PreparedDeviceGroup) GetDevices() []kubeletplugin.Device {
	var devices []kubeletplugin.Device
	for _, device := range g.Devices {
		switch device.Type() {
		case GpuDeviceType:
			devices = append(devices, *device.Gpu.Device)
		case PreparedMigDeviceType:
			devices = append(devices, *device.Mig.Device)
		case VfioDeviceType:
			devices = append(devices, *device.Vfio.Device)
		}
	}
	return devices
}

// GetDeviceNames 返回该设备组中的设备规范名称列表（GPU 和 MIG，不含 VFIO）。
// VFIO 设备的 Unprepare 通过 VfioPciManager.Unconfigure 处理，不走通用的设备名称查找路径。
func (g *PreparedDeviceGroup) GetDeviceNames() []DeviceName {
	var names []DeviceName
	for _, device := range g.Devices {
		switch device.Type() {
		case GpuDeviceType:
			// GPU 使用 GpuInfo.CanonicalName()，如 "gpu-0"
			names = append(names, device.Gpu.Info.CanonicalName())
		case PreparedMigDeviceType:
			// MIG 使用 DeviceName，如 "gpu-0-mig-1g10gb-19-0"
			names = append(names, device.Mig.Device.DeviceName)
		}
	}
	return names
}

// UUIDs 返回该列表中所有设备（完整 GPU + MIG + VFIO）的 UUID 列表（排序后）。
// 排序保证结果稳定，便于日志对比和测试。
// 合并三种设备类型的 UUID 后统一排序。
func (l PreparedDeviceList) UUIDs() []string {
	uuids := append(l.GpuUUIDs(), l.MigDeviceUUIDs()...)
	uuids = append(uuids, l.VfioDeviceUUIDs()...)
	slices.Sort(uuids)
	return uuids
}

// UUIDs 返回该设备组中所有设备的 UUID 列表（排序后）。
// 委托给内部 PreparedDeviceList 的 UUIDs 方法。
func (g *PreparedDeviceGroup) UUIDs() []string {
	uuids := append(g.GpuUUIDs(), g.MigDeviceUUIDs()...)
	uuids = append(uuids, g.VfioDeviceUUIDs()...)
	slices.Sort(uuids)
	return uuids
}

// UUIDs 返回所有设备组中所有设备的 UUID 列表（排序后）。
// 合并所有组的 UUID 后统一排序，确保跨组的 UUID 顺序一致。
func (d PreparedDevices) UUIDs() []string {
	uuids := append(d.GpuUUIDs(), d.MigDeviceUUIDs()...)
	uuids = append(uuids, d.VfioDeviceUUIDs()...)
	slices.Sort(uuids)
	return uuids
}

// GpuUUIDs 返回该列表中所有完整物理 GPU 的 UUID（排序后）。
// 从每个 PreparedGpu.Info.UUID 提取。
func (l PreparedDeviceList) GpuUUIDs() []string {
	var uuids []string
	for _, device := range l.Gpus() {
		uuids = append(uuids, device.Gpu.Info.UUID)
	}
	slices.Sort(uuids)
	return uuids
}

// GpuUUIDs 返回该设备组中所有完整物理 GPU 的 UUID（排序后）。
// 委托给内部 PreparedDeviceList.Gpus().UUIDs()。
func (g *PreparedDeviceGroup) GpuUUIDs() []string {
	return g.Devices.Gpus().UUIDs()
}

// GpuUUIDs 返回所有设备组中完整物理 GPU 的 UUID（排序后）。
// 合并所有组的 GPU UUID 后统一排序。
func (d PreparedDevices) GpuUUIDs() []string {
	var uuids []string
	for _, group := range d {
		uuids = append(uuids, group.GpuUUIDs()...)
	}
	slices.Sort(uuids)
	return uuids
}

// MigDeviceUUIDs 返回该列表中所有 MIG 设备的 UUID（排序后）。
// MIG UUID 从 MigLiveTuple.MigUUID 获取，是物化后分配的唯一标识符。
func (l PreparedDeviceList) MigDeviceUUIDs() []string {
	var uuids []string
	for _, device := range l.MigDevices() {
		uuids = append(uuids, device.Mig.Concrete.MigUUID)
	}
	slices.Sort(uuids)
	return uuids
}

// MigDeviceUUIDs 返回该设备组中所有 MIG 设备的 UUID（排序后）。
// 委托给内部 PreparedDeviceList.MigDevices().UUIDs()。
func (g *PreparedDeviceGroup) MigDeviceUUIDs() []string {
	return g.Devices.MigDevices().UUIDs()
}

// MigDeviceUUIDs 返回所有设备组中 MIG 设备的 UUID（排序后）。
// 合并所有组的 MIG UUID 后统一排序。
func (d PreparedDevices) MigDeviceUUIDs() []string {
	var uuids []string
	for _, group := range d {
		uuids = append(uuids, group.MigDeviceUUIDs()...)
	}
	slices.Sort(uuids)
	return uuids
}

// VfioDeviceUUIDs 返回该设备组中所有 VFIO 设备的 UUID（排序后）。
// 委托给内部 PreparedDeviceList.VfioDevices().UUIDs()。
func (g *PreparedDeviceGroup) VfioDeviceUUIDs() []string {
	return g.Devices.VfioDevices().UUIDs()
}

// VfioDeviceUUIDs 返回该列表中所有 VFIO 设备的 UUID（排序后）。
// VFIO 设备的 UUID 是根据 PCIe Bus ID 确定性生成的 SHA1 UUID，跨重启稳定。
func (l PreparedDeviceList) VfioDeviceUUIDs() []string {
	var uuids []string
	for _, device := range l.VfioDevices() {
		uuids = append(uuids, device.Vfio.Info.UUID)
	}
	slices.Sort(uuids)
	return uuids
}

// VfioDeviceUUIDs 返回所有设备组中 VFIO 设备的 UUID（排序后）。
// 合并所有组的 VFIO UUID 后统一排序。
func (d PreparedDevices) VfioDeviceUUIDs() []string {
	var uuids []string
	for _, group := range d {
		uuids = append(uuids, group.VfioDeviceUUIDs()...)
	}
	slices.Sort(uuids)
	return uuids
}

// GetNonAdminDevices 返回该 PreparedClaim 中所有非管理员访问的设备名称集合。
// 管理员访问（AdminAccess=true）的设备允许被多个 Claim 同时持有，
// 非管理员设备则是互斥的（validateNoOverlappingPreparedDevices 中使用此函数检查冲突）。
// 若 Allocation 为 nil（Claim 尚未被调度），返回空集合。
//
// 返回值：设备规范名称到空结构体的映射（用作 set），仅包含非管理员设备
func (c *PreparedClaim) GetNonAdminDevices() map[string]struct{} {
	// 预分配容量，减少 map 扩容
	requested := make(map[string]struct{}, len(c.Status.Allocation.Devices.Results))

	// 若 Claim 尚未被调度（无 Allocation），返回空集合
	if c.Status.Allocation == nil {
		return requested
	}
	// 遍历所有分配结果
	for _, r := range c.Status.Allocation.Devices.Results {
		// 跳过非本驱动的设备
		if r.Driver != DriverName {
			continue
		}
		// 跳过管理员访问设备（AdminAccess=true）
		if r.AdminAccess != nil && *r.AdminAccess {
			continue
		}
		// 将非管理员设备名加入集合
		requested[r.Device] = struct{}{}
	}
	return requested
}
