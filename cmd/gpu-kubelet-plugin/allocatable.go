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

// allocatable 包定义了节点上所有可分配 GPU 设备的核心数据结构和查询方法。
//
// 核心类型：
//   - DeviceName: 设备在 DRA ResourceSlice 中公告的规范名称类型别名
//   - AllocatableDevices: 设备规范名到设备对象的映射（全局设备注册表）
//   - AllocatableDevice: 单个可分配设备的联合类型（GPU/MIG静态/MIG动态/VFIO）
//   - AllocatableDeviceList: 设备列表，支持过滤和遍历
//
// 设计决策：
//   - AllocatableDevice 使用 union 结构而非 interface，使字段可直接访问，避免类型断言的开销
//   - 代价是调用方需要先通过 Type() 方法判断类型，再访问对应字段
//   - DeviceName 使用 minor 号而非 UUID，因为 minor 号简洁可读，且在节点生命周期内通常稳定
package main

import (
	"slices"

	resourceapi "k8s.io/api/resource/v1"

	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
)

// DeviceName 是设备在 DRA ResourceSlice 中公告的规范名称（如 "gpu-0"、"gpu-0-mig-1g24gb-19-0"）。
//
// 命名规范：
//   - 完整 GPU：gpu-<minor>，如 "gpu-0"、"gpu-1"
//   - 静态/动态 MIG：gpu-<parentMinor>-mig-<profile>-<profileID>-<placementStart>
//   - VFIO 直通：gpu-vfio-<index>
//
// 该名称在节点本地范围内必须唯一（同一节点上不重名），会在错误消息中暴露给用户，
// 并在 NodePrepareResources 请求中被 kubelet 回传，用于在 AllocatableDevices map 中定位设备。
// 为什么用 minor 号而不是 UUID？minor 号简洁可读，且在节点生命周期内足够稳定。
type DeviceName = string

// AllocatableDevices 是节点上所有可分配设备的映射（设备规范名 → 设备对象）。
// 插件启动时由 enumerateAllPossibleDevices() 填充，之后在 Prepare/Unprepare 期间保持不变
// （VFIO 场景下 Prepare/Unprepare 会动态增删兄弟设备）。
type AllocatableDevices map[DeviceName]*AllocatableDevice

// AllocatableDevice 表示节点上一个可被调度器分配的设备。
//
// 四种设备类型通过 union 方式表达，同一时刻只有一个字段非 nil：
//   - Gpu：完整物理 GPU（非 MIG 模式，或 VFIO 模式前的基础信息）
//   - MigDynamic：DynamicMIG 管理的抽象 MIG 配置（未物化实例，无 UUID）
//     代表"可以创建这种配置的 MIG 设备"，在 Prepare 时才真正创建
//   - MigStatic：预创建的静态 MIG 设备实例（已物化，有 UUID）
//   - Vfio：通过 vfio-pci 驱动直通的 GPU（由 PassthroughSupport 特性门控控制）
//
// 为何不用接口（interface）？union 结构使字段直接可访问，避免大量类型断言，
// 代价是调用方需要先检查 Type()。
type AllocatableDevice struct {
	// Gpu 指向完整物理 GPU 的信息，当设备类型为 GpuDeviceType 时非 nil
	Gpu *GpuInfo
	// MigDynamic 指向动态 MIG 设备的抽象描述，当设备类型为 MigDynamicDeviceType 时非 nil
	MigDynamic *MigSpec
	// MigStatic 指向静态 MIG 设备的具体实例信息，当设备类型为 MigStaticDeviceType 时非 nil
	MigStatic *MigDeviceInfo
	// Vfio 指向 VFIO 直通设备的信息，当设备类型为 VfioDeviceType 时非 nil
	Vfio *VfioDeviceInfo
}

// AllocatableDeviceList 是 AllocatableDevice 的切片，方便对设备列表进行过滤和遍历。
// 提供了按类型过滤的便捷方法（GetGPUs、GetVfioDevices 等）。
type AllocatableDeviceList []*AllocatableDevice

// Type 返回该设备的类型字符串（"gpu"/"migdyn"/"mig"/"vfio"/"unknown"）。
// 用于在业务逻辑中根据设备类型选择不同的处理路径。
// 通过检查哪个联合字段非 nil 来判断类型。
func (d AllocatableDevice) Type() string {
	if d.Gpu != nil {
		return GpuDeviceType
	}
	if d.MigDynamic != nil {
		return MigDynamicDeviceType
	}
	if d.MigStatic != nil {
		return MigStaticDeviceType
	}
	if d.Vfio != nil {
		return VfioDeviceType
	}
	return UnknownDeviceType
}

// IsStaticOrDynMigDevice 判断该设备是否为 MIG 设备（无论静态还是动态）。
// 用于 prepareDevices() 中检查 MigDeviceConfig 是否可以应用到该设备。
// 静态 MIG 和动态 MIG 在 Prepare 阶段共享大部分配置逻辑，
// 只是动态 MIG 需要额外执行 MIG 实例的创建/销毁操作。
func (d AllocatableDevice) IsStaticOrDynMigDevice() bool {
	switch d.Type() {
	case MigStaticDeviceType, MigDynamicDeviceType:
		return true
	default:
		return false
	}
}

// CanonicalName 返回该设备的规范名称（与 DeviceName/ResourceSlice 中一致）。
// 根据设备类型委托给对应子结构的 CanonicalName 方法。
// 对于不支持的设备类型会 panic（属于编程错误，不应出现在运行时）。
func (d *AllocatableDevice) CanonicalName() string {
	switch d.Type() {
	case GpuDeviceType:
		return d.Gpu.CanonicalName()
	case MigStaticDeviceType:
		return d.MigStatic.CanonicalName()
	case MigDynamicDeviceType:
		return d.MigDynamic.CanonicalName()
	case VfioDeviceType:
		return d.Vfio.CanonicalName()
	}
	panic("unexpected type for AllocatableDevice")
}

// GetDevice 将 AllocatableDevice 转换为 Kubernetes DRA API 的 resourceapi.Device 对象，
// 用于非 DynamicMIG 模式下的 ResourceSlice 发布。
// DynamicMIG 模式下应使用 PartGetDevice()（KEP 4815 Partitionable Devices 格式）。
//
// 注意：MigDynamicDeviceType 禁止调用此方法（DynamicMIG 设备无物化实例，不能用旧接口），
// 因为动态 MIG 设备没有 UUID 和具体的容量信息，无法映射到传统的 Device 格式。
func (d *AllocatableDevice) GetDevice() resourceapi.Device {
	switch d.Type() {
	case GpuDeviceType:
		return d.Gpu.GetDevice()
	case MigStaticDeviceType:
		return d.MigStatic.GetDevice()
	case MigDynamicDeviceType:
		// 动态 MIG 设备不支持传统 GetDevice() 接口，
		// 因为它没有物化实例（无 UUID、无具体容量）。
		// 调用方应使用 PartGetDevice()（KEP 4815 格式）。
		panic("GetDevice() must currently not be called for MigDynamicDeviceType")
	case VfioDeviceType:
		return d.Vfio.GetDevice()
	}
	panic("unexpected type for AllocatableDevice")
}

// UUID 返回该设备的唯一标识符。
// 对于 GPU、静态 MIG 和 VFIO 设备，返回各自的 UUID。
//
// 注意：DynamicMIG 模式下的抽象 MIG 设备没有 UUID（在 Prepare 时才创建实例并获取 UUID），
// 调用此方法会 panic。该方法存在是为了让 DynamicMIG 禁用时 AllocatableDevices 能实现
// UUIDProvider 接口，供 MPS/TimeSlicing 等组件使用。
//
// 为什么不使用 interface？UUID() 方法是 UUIDProvider 接口的一部分，
// DynamicMIG 开启时不应该调用此方法（由 MigDeviceUUIDs() 中的 panic 守护），
// 但接口实现需要该方法存在。
func (d AllocatableDevice) UUID() string {
	if d.Gpu != nil {
		return d.Gpu.UUID
	}
	if d.MigStatic != nil {
		return d.MigStatic.UUID
	}
	if d.MigDynamic != nil {
		// DynamicMIG 模式下，抽象 MIG 设备没有 UUID（物化前不存在）。
		// 调用方必须保证在 DynamicMIG 开启时不调用此方法。
		panic("unexpected UUID() call for AllocatableDevice of type MigDynamic")
	}
	if d.Vfio != nil {
		return d.Vfio.UUID
	}
	panic("unexpected type for AllocatableDevice")
}

// getDevicesByGPUPCIBusID 返回与指定 PCIe Bus ID 关联的所有设备（包括 GPU、MIG、VFIO）。
// 用于 VFIO 场景中查找同一物理 GPU 的"兄弟设备"（如 GPU 模式和 VFIO 模式之间的切换）。
// 遍历整个设备注册表，匹配每种类型设备中的 PCIe Bus ID 字段。
//
// 参数：
//   - pcieBusID: 目标 GPU 的 PCIe 总线地址
//
// 返回值：与该 PCIe 地址关联的所有 AllocatableDevice 列表
func (d AllocatableDevices) getDevicesByGPUPCIBusID(pcieBusID string) AllocatableDeviceList {
	var devices AllocatableDeviceList
	for _, device := range d {
		switch device.Type() {
		case GpuDeviceType:
			// GPU 设备直接比较其自身的 PCIe 地址
			if device.Gpu.pcieBusID == pcieBusID {
				devices = append(devices, device)
			}
		case MigStaticDeviceType:
			// 静态 MIG 设备比较其父 GPU 的 PCIe 地址
			if device.MigStatic.parent.pcieBusID == pcieBusID {
				devices = append(devices, device)
			}
		case MigDynamicDeviceType:
			// 动态 MIG 设备比较其父 GPU 的 PCIe 地址
			if device.MigDynamic.Parent.pcieBusID == pcieBusID {
				devices = append(devices, device)
			}
		case VfioDeviceType:
			// VFIO 设备比较其自身的 PCIe 地址
			if device.Vfio.pcieBusID == pcieBusID {
				devices = append(devices, device)
			}
		}
	}
	return devices
}

// GetGPUByPCIeBusID 通过 PCIe Bus ID 查找完整物理 GPU（仅返回 GpuDeviceType）。
// 主要用于 VFIO 设备构建时，将 VFIO 设备与其父 GPU 关联，
// 以获取父 GPU 的 minor 号、UUID 等基础信息。
//
// 参数：
//   - pcieBusID: 目标 GPU 的 PCIe 总线地址
//
// 返回值：
//   - *AllocatableDevice: 找到的 GPU 设备（类型为 GpuDeviceType）
//   - nil: 未找到匹配的 GPU
func (d AllocatableDevices) GetGPUByPCIeBusID(pcieBusID string) *AllocatableDevice {
	for _, device := range d {
		// 跳过非 GPU 类型的设备
		if device.Type() != GpuDeviceType {
			continue
		}
		if device.Gpu.pcieBusID == pcieBusID {
			return device
		}
	}
	return nil
}

// GetGPUs 返回所有完整物理 GPU 设备列表（过滤 MIG 和 VFIO）。
// 用于遍历节点上所有可用的完整 GPU，例如在 MPS 守护进程启动时需要知道所有 GPU。
func (d AllocatableDevices) GetGPUs() AllocatableDeviceList {
	var devices AllocatableDeviceList
	for _, device := range d {
		if device.Type() == GpuDeviceType {
			devices = append(devices, device)
		}
	}
	return devices
}

// GetVfioDevices 返回所有 VFIO 直通设备列表。
// 用于遍历节点上所有 VFIO 设备，例如在 ResourceSlice 发布时。
func (d AllocatableDevices) GetVfioDevices() AllocatableDeviceList {
	var devices AllocatableDeviceList
	for _, device := range d {
		if device.Type() == VfioDeviceType {
			devices = append(devices, device)
		}
	}
	return devices
}

// GpuUUIDs 返回所有完整物理 GPU 的 UUID 列表（排序后），实现 UUIDProvider 接口。
// 排序保证结果稳定，便于日志对比和测试断言。
// 该方法供 MPS/TimeSlicing 组件获取需要管理的 GPU 范围。
func (d AllocatableDevices) GpuUUIDs() []string {
	var uuids []string
	for _, dev := range d {
		if dev.Type() == GpuDeviceType {
			uuids = append(uuids, dev.UUID())
		}
	}
	slices.Sort(uuids)
	return uuids
}

// MigDeviceUUIDs 返回所有静态 MIG 设备的 UUID 列表（排序后），实现 UUIDProvider 接口。
// 重要限制：DynamicMIG 特性门控开启时禁止调用此方法，因为动态 MIG 设备在 Prepare 前无 UUID，
// 此方法仅适用于静态 MIG 模式下获取已物化 MIG 设备的 UUID。
//
// 返回值：排序后的 MIG 设备 UUID 列表
//
// 注意：调用此方法前应确保 DynamicMIG 特性门控未开启，否则会触发 panic
func (d AllocatableDevices) MigDeviceUUIDs() []string {
	// DynamicMIG 模式下不存在静态 MIG 设备，不应调用此方法
	if featuregates.Enabled(featuregates.DynamicMIG) {
		panic("MigDeviceUUIDs() unexpectedly called (DynamicMIG is enabled)")
	}
	var uuids []string
	for _, dev := range d {
		if dev.Type() == MigStaticDeviceType {
			uuids = append(uuids, dev.UUID())
		}
	}
	slices.Sort(uuids)
	return uuids
}

// VfioDeviceUUIDs 返回所有 VFIO 直通设备的 UUID 列表（排序后）。
// VFIO 设备的 UUID 是根据 PCIe Bus ID 确定性生成的（SHA1 UUID），因此跨重启稳定。
// 该方法供 UUIDProvider 接口实现使用，允许上层组件统一查询所有设备 UUID。
func (d AllocatableDevices) VfioDeviceUUIDs() []string {
	var uuids []string
	for _, device := range d {
		if device.Type() == VfioDeviceType {
			uuids = append(uuids, device.Vfio.UUID)
		}
	}
	slices.Sort(uuids)
	return uuids
}

// UUIDs 返回所有设备（完整 GPU + MIG）的 UUID 列表，实现 UUIDProvider 接口。
// 注意：DynamicMIG 开启时不应调用此方法（MigDynamic 设备无 UUID，会触发 panic）。
// 该方法合并 GPU 和 MIG 的 UUID 后统一排序返回。
//
// 返回值：排序后的所有设备 UUID 列表
func (d AllocatableDevices) UUIDs() []string {
	var uuids []string
	for _, dev := range d {
		uuids = append(uuids, dev.UUID())
	}
	slices.Sort(uuids)
	return uuids
}

// RemoveSiblingDevices 从 AllocatableDevices map 中删除与给定设备共享同一物理 GPU（PCIe Bus ID）
// 但类型不同的"兄弟设备"。
//
// 使用场景：VFIO Prepare 流程中，当某个 GPU 的 VFIO 设备被准备（切换到 vfio-pci 驱动）后，
// 原先公告的完整 GPU 设备（nvidia 驱动模式）不再可用，需要从 allocatable map 中移除，
// 防止调度器继续将新任务分配到该 GPU 的 nvidia 驱动模式。
// 反之，当 VFIO Unprepare 后，需要重新添加完整 GPU 设备（通过 discoverSiblingAllocatables）。
//
// 当前此函数仅在 PassthroughSupport 特性门控开启时调用，与 DynamicMIG 不兼容。
// 对于 MIG 类型的兄弟设备（MigStatic/MigDynamic），当前直接返回不做处理（TODO）。
//
// 参数：
//   - device: 触发兄弟设备删除的源设备（如刚被 Prepare 的 VFIO 设备）
func (d AllocatableDevices) RemoveSiblingDevices(device *AllocatableDevice) {
	var pciBusID string
	// 提取源设备的 PCIe Bus ID
	switch device.Type() {
	case GpuDeviceType:
		pciBusID = device.Gpu.pcieBusID
	case VfioDeviceType:
		pciBusID = device.Vfio.pcieBusID
	case MigStaticDeviceType:
		// 静态 MIG 设备的兄弟删除逻辑尚未实现
		return
	case MigDynamicDeviceType:
		// 动态 MIG 设备的兄弟删除逻辑尚未实现
		return
	}

	// 查找同一 PCIe 地址下的所有设备
	siblings := d.getDevicesByGPUPCIBusID(pciBusID)
	for _, sibling := range siblings {
		// 跳过与源设备同类型的设备（保留同类型）
		if sibling.Type() == device.Type() {
			continue
		}
		// 删除不同类型的兄弟设备
		switch sibling.Type() {
		case GpuDeviceType:
			// 删除完整 GPU 设备（VFIO Prepare 时需要移除对应的 nvidia 模式 GPU）
			delete(d, sibling.Gpu.CanonicalName())
		case VfioDeviceType:
			// 删除 VFIO 设备（GPU Prepare 时需要移除对应的 VFIO 设备）
			delete(d, sibling.Vfio.CanonicalName())
		case MigStaticDeviceType:
			// 静态 MIG 兄弟设备暂不处理
			continue
		case MigDynamicDeviceType:
			// 动态 MIG 兄弟设备暂不处理
			continue
		}
	}
}

// IsHealthy 返回该设备当前是否健康（由 NVML 健康监控器更新）。
//
// 健康状态规则：
//   - GPU：由 NVML XID 关键错误事件驱动，发生致命错误时标记为 Unhealthy。
//     调度器看到 Unhealthy 的设备会停止向其调度新任务。
//   - 静态 MIG：独立跟踪健康状态（但父设备的健康是否传递尚未确定，TODO）。
//   - 动态 MIG：始终返回 true（抽象设备，物化前无法判断健康状态，保守地视为健康）。
//   - VFIO：未实现健康检查（panic 兜底，因为 VFIO 设备不应出现在健康检查的代码路径中）。
//
// 返回值：
//   - true: 设备健康，可被调度
//   - false: 设备不健康，不应被调度
func (d *AllocatableDevice) IsHealthy() bool {
	switch d.Type() {
	case GpuDeviceType:
		return d.Gpu.health == Healthy
	case MigStaticDeviceType:
		return d.MigStatic.health == Healthy
	case MigDynamicDeviceType:
		// 动态 MIG 设备是抽象表示，在 Prepare 时才物化。
		// 物化前无法判断健康状态，保守地返回 healthy。
		return true
	}
	panic("unexpected type for AllocatableDevice")
}
