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

// deviceinfo.go 定义了 DRA 驱动中所有设备类型的核心数据结构及其与 Kubernetes DRA API 的转换方法。
//
// 本文件是 kubelet 插件的数据模型核心，包含三种设备类型：
//   - GpuInfo：完整物理 GPU，代表节点上的一整块显卡
//   - MigDeviceInfo：MIG（Multi-Instance GPU）设备实例，代表物理 GPU 的一个切片分区
//   - VfioDeviceInfo：VFIO 直通设备，代表通过 vfio-pci 驱动直接分配给容器的 GPU
//
// 每种设备类型均实现了 GetDevice() 方法，将内部数据转换为 Kubernetes DRA API 的
// resourceapi.Device 对象，用于发布到 ResourceSlice 中供调度器分配。
//
// 关键设计决策：
//   - 带 json tag 的字段会被序列化到 Checkpoint，供重启后恢复状态
//   - 无 json tag 的字段是运行时状态，重启后从 NVML 重新获取
//   - MIG 设备的 CanonicalName 编码了物理配置信息（父 GPU + profile + 放置位置），
//     使得名称本身携带了足够的信息进行反解析
package main

import (
	"fmt"

	"github.com/Masterminds/semver"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/dynamic-resource-allocation/deviceattribute"
	"k8s.io/utils/ptr"
)

// HealthStatus 表示设备的健康状态，语义与 Device Plugin v1beta1 API 保持一致。
type HealthStatus string

const (
	// Healthy 表示设备正常运行，可被调度器分配和使用。
	Healthy HealthStatus = "Healthy"

	// Unhealthy 表示设备发生了致命错误（通过 NVMLDeviceHealthCheck 特性门控的 XID 事件上报）。
	// 不健康的设备会从 ResourceSlice 中移除，调度器不会再向其调度新任务。
	// 当前没有自动恢复机制：设备变为不健康后，插件需要重启才能重新公告该设备。
	Unhealthy HealthStatus = "Unhealthy"
)

// GpuInfo 表示一块具体的物理 GPU 及其所有属性，是插件中最核心的数据结构之一。
//
// 生命周期：
//   - 插件启动时由 getGpuInfo() 通过 NVML 查询填充（所有带 json tag 的字段）
//   - 扫描完所有 MIG Profile 后由 AddDetailAfterWalkingMigProfiles() 补充 maxCapacities/memSliceCount
//   - 发布到 Kubernetes API Server 的 ResourceSlice 中作为设备属性（由 PartDevAttributes()/GetDevice() 使用）
//
// 字段说明：
//   - 无 json tag 的字段（minor、migEnabled 等）是运行时状态，不序列化到 Checkpoint
//   - pcieRootAttr 是可选字段，仅在支持 PCIe 根桥属性的环境中填充
//   - maxCapacities/memSliceCount 依赖 MIG Profile 扫描结果，仅在 DynamicMIG 模式下有意义
type GpuInfo struct {
	// UUID 是 GPU 的全局唯一标识符，由 NVML 分配，格式如 "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"。
	// 此字段会序列化到 Checkpoint，用于跨重启稳定标识同一块 GPU。
	UUID string `json:"uuid"`

	// minor 是 GPU 在系统中的次设备号，用于构建 /dev/nvidia<minor> 设备节点路径。
	// 此字段是运行时状态，不序列化到 Checkpoint。
	minor int

	// migEnabled 标记此 GPU 是否已启用 MIG 模式。
	// true 表示 GPU 处于 MIG 模式，可以创建 MIG 设备实例；false 表示处于传统模式。
	// 此字段是运行时状态，不序列化到 Checkpoint。
	migEnabled bool

	// vfioEnabled 标记此 GPU 是否以 VFIO 直通模式运行。
	// true 表示 GPU 通过 vfio-pci 驱动绑定，可直接分配给虚拟机或容器。
	// 此字段是运行时状态，不序列化到 Checkpoint。
	vfioEnabled bool

	// memoryBytes 是 GPU 的总显存大小（字节）。
	// 用于 ResourceSlice 中设备的 "memory" 容量字段。
	memoryBytes uint64

	// productName 是 GPU 的产品名称（如 "NVIDIA A100-SXM4-80GB"），
	// 作为设备属性发布到 ResourceSlice 中，供调度器做产品级匹配。
	productName string

	// brand 是 GPU 的品牌名称（如 "Tesla"、"GeForce"），
	// 作为设备属性发布到 ResourceSlice 中。
	brand string

	// architecture 是 GPU 的微架构代号（如 "Ampere"、"Hopper"），
	// 作为设备属性发布到 ResourceSlice 中。
	architecture string

	// cudaComputeCapability 是 CUDA 计算能力版本号（如 "8.0"、"9.0"），
	// 作为版本类型属性发布到 ResourceSlice 中，用于调度器按计算能力筛选设备。
	cudaComputeCapability string

	// driverVersion 是当前安装的 NVIDIA 驱动版本号（如 "535.129.03"），
	// 作为版本类型属性发布到 ResourceSlice 中。
	driverVersion string

	// cudaDriverVersion 是 CUDA 驱动版本号（如 "12.2"），
	// 作为版本类型属性发布到 ResourceSlice 中。
	cudaDriverVersion string

	// pcieBusID 是 GPU 的 PCIe 总线地址（如 "0000:3b:00.0"），
	// 作为字符串属性发布到 ResourceSlice 中。
	pcieBusID string

	// pcieRootAttr 是可选的 PCIe 根桥属性，仅在支持设备拓扑属性的环境中填充。
	// 当非 nil 时，会追加到设备的 Attributes 映射中。
	pcieRootAttr *deviceattribute.DeviceAttribute

	// migProfiles 是此 GPU 支持的所有 MIG Profile 列表，
	// 每个 MigProfileInfo 描述一种可创建的 MIG 配置（如 "1g.10gb"、"2g.20gb"）。
	// 仅在 MIG 模式启用时有意义。
	migProfiles []*MigProfileInfo

	// addressingMode 是 GPU 的寻址模式属性（如 "HMM"），可选字段。
	// 当非 nil 时，会追加到设备的 Attributes 映射中。
	addressingMode *string

	// health 表示当前 GPU 的健康状态（Healthy 或 Unhealthy）。
	// 由设备健康监控器更新，影响设备是否在 ResourceSlice 中公告。
	health HealthStatus

	// maxCapacities 和 memSliceCount 在扫描完所有 MIG Profile 后由
	// AddDetailAfterWalkingMigProfiles() 填充。
	// maxCapacities 是各容量维度（multiprocessors/memory 等）的最大值，
	// 代表完整 GPU 的全量容量，作为 SharedCounterSet 的初始计数器值。
	// memSliceCount 是该 GPU 支持的显存切片数量，用于 KEP 4815 的内存切片计数器。
	maxCapacities PartCapacityMap
	memSliceCount int
}

// MigDeviceInfo 表示一个具体的、已物化的 MIG 设备实例（无论静态预创建还是动态创建）。
//
// 带 json tag 的字段会序列化到 Checkpoint，供 Unprepare 时精确定位并销毁该 MIG 设备：
//   - UUID：MIG 设备的唯一标识符（物化后分配，跨销毁/重建会变化）
//   - Profile：MIG Profile 名称字符串（如 "1g.10gb"）
//   - ParentUUID/ParentMinor：父 GPU 的 UUID 和 minor 号
//   - GIID/CIID：GPU Instance ID 和 Compute Instance ID（用于 NVML API 操作）
//   - PlacementStart/PlacementSize：内存切片的起始位置和大小（CanonicalName 依赖此值）
//
// 无 json tag 的字段（gIInfo、cIInfo、parent 等）是运行时对象，重启后从 NVML 重新获取。
type MigDeviceInfo struct {
	// UUID 是 MIG 设备的全局唯一标识符，格式如 "MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"。
	// 物化后由 NVML 分配，销毁并重新创建后可能变化。
	// 此字段会序列化到 Checkpoint，用于 Unprepare 时识别要销毁的 MIG 实例。
	UUID string `json:"uuid"`

	// Profile 是 MIG Profile 的名称（如 "1g.10gb"、"3g.40gb"），
	// 描述了该 MIG 实例的资源配额（GPU 计算单元数、显存大小等）。
	// 此字段会序列化到 Checkpoint。
	Profile string `json:"profile"`

	// ParentUUID 是父物理 GPU 的 UUID，标识此 MIG 实例所属的物理 GPU。
	// 此字段会序列化到 Checkpoint。
	ParentUUID string `json:"parentUUID"`

	// GiProfileID 是 GPU Instance 的 Profile ID，用于标识 GI 的类型和资源配额。
	// 与 Profile 字段不同，这是 NVML 内部使用的数字 ID。
	// 此字段会序列化到 Checkpoint。
	GiProfileID int `json:"profileId"`

	// ParentMinor 是父物理 GPU 的 minor 号，用于构建 CanonicalName 和定位设备节点。
	// 此字段会序列化到 Checkpoint。
	ParentMinor int `json:"parentMinor"`

	// CIID 是 Compute Instance 的 ID，用于 NVML API 中定位 CI 进行操作（如销毁）。
	// 此字段会序列化到 Checkpoint。
	CIID int `json:"ciId"`

	// GIID 是 GPU Instance 的 ID，用于 NVML API 中定位 GI 进行操作（如销毁）。
	// 此字段会序列化到 Checkpoint。
	GIID int `json:"giId"`

	// PlacementStart 是 MIG 设备在 GPU 显存中的起始切片位置（从 0 开始编号）。
	// 此字段必须序列化到 Checkpoint，因为 CanonicalName() 依赖 PlacementStart，
	// 而 CanonicalName 用于 ResourceSlice 中的设备查找。
	// 若仅在运行时持有这些值，重启后反序列化的 MigDeviceInfo 将无法生成正确的规范名称。
	PlacementStart int `json:"placementStart"`

	// PlacementSize 是 MIG 设备占用的显存切片数量。
	// 例如 "1g.10gb" Profile 的 PlacementSize 为 1，"2g.20gb" 为 2。
	// 此字段会序列化到 Checkpoint，与 PlacementStart 配合确定 MIG 设备的显存占位范围。
	PlacementSize int `json:"placementSize"`

	// gIInfo 是 GPU Instance 的 NVML 信息对象，包含 GI 的运行时属性。
	// 重启后通过 NVML API 重新获取，不序列化到 Checkpoint。
	gIInfo *nvml.GpuInstanceInfo

	// cIInfo 是 Compute Instance 的 NVML 信息对象，包含 CI 的运行时属性。
	// 重启后通过 NVML API 重新获取，不序列化到 Checkpoint。
	cIInfo *nvml.ComputeInstanceInfo

	// parent 是指向父 GpuInfo 的指针，用于访问父 GPU 的属性（产品名、架构等）。
	// 重启后重新关联，不序列化到 Checkpoint。
	parent *GpuInfo

	// giProfileInfo 是 GPU Instance Profile 的 NVML 信息对象，
	// 包含该 Profile 的资源配额详情（多处理器数、编码器数、显存大小等）。
	// 用于填充 MigDeviceInfo.GetDevice() 的容量字段。
	// 重启后通过 NVML API 重新获取，不序列化到 Checkpoint。
	giProfileInfo *nvml.GpuInstanceProfileInfo

	// ciProfileInfo 是 Compute Instance Profile 的 NVML 信息对象。
	// 重启后通过 NVML API 重新获取，不序列化到 Checkpoint。
	ciProfileInfo *nvml.ComputeInstanceProfileInfo

	// pcieBusID 是 MIG 设备的 PCIe 总线地址，继承自父 GPU。
	// 用于 ResourceSlice 中设备的 pciBusID 属性。
	pcieBusID string

	// pcieRootAttr 是可选的 PCIe 根桥属性，继承自父 GPU。
	// 当非 nil 时，会追加到 MIG 设备的 Attributes 映射中。
	pcieRootAttr *deviceattribute.DeviceAttribute

	// health 表示当前 MIG 设备的健康状态（Healthy 或 Unhealthy）。
	// 继承自父 GPU 的健康状态，由设备健康监控器更新。
	health HealthStatus
}

// VfioDeviceInfo 表示一个通过 vfio-pci 驱动直通的 GPU 设备。
// UUID 是根据 PCIe Bus ID 确定性生成的 SHA1 UUID，跨重启保持稳定。
// iommuGroup 用于确定 /dev/vfio/<iommuGroup> 设备节点路径，注入到容器 CDI spec 中。
type VfioDeviceInfo struct {
	// UUID 是 VFIO 设备的唯一标识符，基于 PCIe Bus ID 通过 SHA1 算法确定性生成。
	// 这确保了同一设备跨重启保持相同的 UUID，便于 Checkpoint 中稳定引用。
	UUID string `json:"uuid"`

	// deviceID 是 GPU 的 PCI 设备 ID（如 "0x20b0"），标识具体的 GPU 型号。
	deviceID string

	// vendorID 是 GPU 的 PCI 厂商 ID（NVIDIA 固定为 "0x10de"）。
	vendorID string

	// index 是 nvpci 扫描时分配的顺序号，用于构建 CanonicalName（格式：gpu-vfio-<index>）。
	index int

	// parent 是指向父 GpuInfo 的指针（若该 VFIO 设备对应一块完整物理 GPU）。
	// 用于访问 GPU 的公共属性（产品名等）。某些场景下可能为 nil。
	parent *GpuInfo

	// productName 是 VFIO 设备的产品名称，用于 ResourceSlice 的设备属性。
	productName string

	// pcieBusID 是 VFIO 设备的 PCIe 总线地址。
	pcieBusID string

	// pcieRootAttr 是可选的 PCIe 根桥属性。
	pcieRootAttr *deviceattribute.DeviceAttribute

	// numaNode 是该设备所在的 NUMA 节点编号（-1 表示无 NUMA 亲和性）。
	// 作为整数属性发布到 ResourceSlice 中，供调度器做 NUMA 感知调度。
	numaNode int

	// iommuGroup 是该设备所属的 IOMMU 组编号，
	// 用于确定 /dev/vfio/<iommuGroup> 设备节点路径，在 CDI spec 中注入到容器。
	iommuGroup int

	// addressableMemoryBytes 是该 VFIO 设备的可寻址显存大小（字节），
	// 作为 "addressableMemory" 容量字段发布到 ResourceSlice 中。
	addressableMemoryBytes uint64
}

// CanonicalName 返回完整物理 GPU 的规范设备名称（格式：gpu-<minor>，如 "gpu-0"）。
//
// 历史背景：早期版本使用 UUID 作为名称，后改为 minor 号。
// 使用 minor 号的原因：简短可读，用户在日志中更容易识别。
// minor 号的稳定性：通常在 GPU 设备不变的情况下保持稳定，
// 但 NVML 文档指出 SetMigMode() 可能导致 minor 号变化（极少数情况）。
//
// 返回值：
//   - DeviceName：规范名称，格式为 "gpu-<minor>"
func (d *GpuInfo) CanonicalName() DeviceName {
	return fmt.Sprintf("gpu-%d", d.minor)
}

// String 返回便于日志阅读的 GPU 描述（格式：gpu-<minor>-<UUID>）。
// 同时包含 minor 号（易于识别）和 UUID（精确唯一），方便调试时快速定位。
//
// 返回值：
//   - string：格式为 "gpu-<minor>-<UUID>" 的字符串
func (d *GpuInfo) String() string {
	return fmt.Sprintf("%s-%s", d.CanonicalName(), d.UUID)
}

// SpecTuple 将 MigDeviceInfo 转换为 MigSpecTuple 三元组描述（parentMinor + profileID + placementStart）。
// MigSpecTuple 是查找和操作 MIG 设备的轻量级标识，不含 UUID（允许在设备不存在时使用）。
//
// 返回值：
//   - *MigSpecTuple：包含父 GPU minor 号、Profile ID 和放置起始位置的三元组
func (m *MigDeviceInfo) SpecTuple() *MigSpecTuple {
	return &MigSpecTuple{
		ParentMinor:    m.ParentMinor,
		ProfileID:      m.GiProfileID,
		PlacementStart: m.PlacementStart,
	}
}

// LiveTuple 将 MigDeviceInfo 转换为 MigLiveTuple，包含物化 MIG 设备的完整标识信息。
// MigLiveTuple 用于 Unprepare 时通过 NVML 定位并销毁 MIG 设备（含 UUID 校验）。
//
// 返回值：
//   - *MigLiveTuple：包含父 GPU minor/UUID、GI/CI ID 和 MIG UUID 的完整标识
func (m *MigDeviceInfo) LiveTuple() *MigLiveTuple {
	return &MigLiveTuple{
		ParentMinor: m.ParentMinor,
		ParentUUID:  m.ParentUUID,
		GIID:        m.GIID,
		CIID:        m.CIID,
		MigUUID:     m.UUID,
	}
}

// CanonicalName 返回 MIG 设备的规范名称，格式：gpu-<parentMinor>-mig-<profile>-<profileID>-<placementStart>
// 该名称在 ResourceSlice 中公告，调度器使用此名称标识 MIG 设备。
// 名称编码了物理配置（parent + profile + placement），可用于反解析确定具体位置。
//
// 返回值：
//   - string：MIG 设备的规范名称
func (d *MigDeviceInfo) CanonicalName() string {
	return d.SpecTuple().ToCanonicalName(d.Profile)
}

// CanonicalName 返回 VFIO 设备的规范名称（格式：gpu-vfio-<index>，如 "gpu-vfio-0"）。
// index 是 nvpci 扫描时分配的顺序号，在节点上按 PCI 枚举顺序固定。
//
// 返回值：
//   - string：VFIO 设备的规范名称
func (d *VfioDeviceInfo) CanonicalName() string {
	return fmt.Sprintf("gpu-vfio-%d", d.index)
}

// AddDetailAfterWalkingMigProfiles 在扫描完所有 MIG Profile 后补充 GPU 的额外属性。
// 必须在 GetPerGpuAllocatableDevices() 中扫描完 MIG Profile 后调用，
// 使 GetDevice() 和 PartGetDevice() 能够包含正确的容量信息。
//
// 参数：
//   - maxcap：各容量维度的最大值映射，代表完整 GPU 的全量容量，
//     在 DynamicMIG 模式下作为 SharedCounterSet 的初始计数器值
//   - memSliceCount：该 GPU 支持的显存切片数量，用于 KEP 4815 的内存切片计数器
func (d *GpuInfo) AddDetailAfterWalkingMigProfiles(maxcap PartCapacityMap, memSliceCount int) {
	d.maxCapacities = maxcap
	d.memSliceCount = memSliceCount
}

// GetDevice 将 GpuInfo 转换为 Kubernetes DRA API 的 resourceapi.Device 对象。
// 该方法用于非 DynamicMIG 模式下的 ResourceSlice 发布（传统单 Slice 格式，无 SharedCounterSet）。
// DynamicMIG 模式下应使用 PartGetDevice()（含 ConsumesCounters 字段，符合 KEP 4815）。
//
// 包含的设备属性：
//   - type：设备类型（"gpu"）
//   - uuid：GPU 的 UUID
//   - productName：产品名称
//   - brand：品牌
//   - architecture：架构
//   - cudaComputeCapability：CUDA 计算能力（版本类型）
//   - driverVersion：驱动版本（版本类型）
//   - cudaDriverVersion：CUDA 驱动版本（版本类型）
//   - pciBusID：PCIe 总线地址
//   - 可选：pcieRootAttr（PCIe 根桥属性）、addressingMode（寻址模式）
//
// 包含的设备容量：
//   - memory：GPU 显存大小（字节，二进制单位）
//
// 返回值：
//   - resourceapi.Device：可发布到 ResourceSlice 中的设备对象
func (d *GpuInfo) GetDevice() resourceapi.Device {
	// 构造基础设备对象，包含名称、属性和容量
	device := resourceapi.Device{
		Name:       d.CanonicalName(),
		Attributes: d.PartDevAttributes(),
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"memory": {
				// 使用 BinarySI 格式（Ki/Mi/Gi）表示显存容量
				Value: *resource.NewQuantity(int64(d.memoryBytes), resource.BinarySI),
			},
		},
	}

	// 若 PCIe 根桥属性存在，追加到设备属性中
	if d.pcieRootAttr != nil {
		device.Attributes[d.pcieRootAttr.Name] = d.pcieRootAttr.Value
	}

	// 若寻址模式属性存在，追加到设备属性中（如 "HMM" 表示统一内存模式）
	if d.addressingMode != nil {
		device.Attributes["addressingMode"] = resourceapi.DeviceAttribute{
			StringValue: d.addressingMode,
		}
	}

	return device
}

// GetDevice 将 MigDeviceInfo 转换为 Kubernetes DRA API 的 resourceapi.Device 对象。
// 用于静态 MIG 模式下的 ResourceSlice 发布（MIG 设备已物化，有 UUID 和资源容量）。
//
// 包含的设备属性：
//   - type：设备类型（"mig"）
//   - uuid：MIG 设备的 UUID
//   - parentUUID：父 GPU 的 UUID
//   - profile：MIG Profile 名称（如 "1g.10gb"）
//   - productName/brand/architecture/cudaComputeCapability/driverVersion/cudaDriverVersion：继承自父 GPU
//   - pciBusID：继承自父 GPU 的 PCIe 总线地址
//   - 可选：pcieRootAttr、addressingMode
//
// 包含的设备容量：
//   - multiprocessors：多处理器数量
//   - copyEngines：拷贝引擎数量
//   - decoders/encoders/jpegEngines/ofaEngines：各类专用引擎数量
//   - memory：显存大小（字节）
//   - memorySliceN：按放置位置列出的内存切片容量（每个值=1，表示被该设备完全占用）
//
// 返回值：
//   - resourceapi.Device：可发布到 ResourceSlice 中的 MIG 设备对象
func (d *MigDeviceInfo) GetDevice() resourceapi.Device {
	// 构造 PCIe 总线地址的限定名称（带标准前缀）
	pciBusIDAttrName := resourceapi.QualifiedName(deviceattribute.StandardDeviceAttributePrefix + "pciBusID")

	// 构造设备对象，包含所有静态属性和容量
	device := resourceapi.Device{
		Name: d.CanonicalName(),
		Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			// API 稳定性要求：对外始终使用 "mig" 字符串，不使用内部常量，
			// 避免内部常量重命名时意外破坏 API 兼容性。
			"type": {
				StringValue: ptr.To("mig"),
			},
			"uuid": {
				StringValue: &d.UUID,
			},
			"parentUUID": {
				StringValue: &d.parent.UUID,
			},
			"profile": {
				StringValue: &d.Profile,
			},
			// 以下属性均继承自父 GPU
			"productName": {
				StringValue: &d.parent.productName,
			},
			"brand": {
				StringValue: &d.parent.brand,
			},
			"architecture": {
				StringValue: &d.parent.architecture,
			},
			// CUDA 计算能力使用版本类型，需通过 semver 解析
			"cudaComputeCapability": {
				VersionValue: ptr.To(semver.MustParse(d.parent.cudaComputeCapability).String()),
			},
			"driverVersion": {
				VersionValue: ptr.To(semver.MustParse(d.parent.driverVersion).String()),
			},
			"cudaDriverVersion": {
				VersionValue: ptr.To(semver.MustParse(d.parent.cudaDriverVersion).String()),
			},
			pciBusIDAttrName: {
				StringValue: &d.pcieBusID,
			},
		},
		// 容量字段来自 GI Profile 信息，描述 MIG 实例的资源配额
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"multiprocessors": {
				Value: *resource.NewQuantity(int64(d.giProfileInfo.MultiprocessorCount), resource.BinarySI),
			},
			"copyEngines": {Value: *resource.NewQuantity(int64(d.giProfileInfo.CopyEngineCount), resource.BinarySI)},
			"decoders":    {Value: *resource.NewQuantity(int64(d.giProfileInfo.DecoderCount), resource.BinarySI)},
			"encoders":    {Value: *resource.NewQuantity(int64(d.giProfileInfo.EncoderCount), resource.BinarySI)},
			"jpegEngines": {Value: *resource.NewQuantity(int64(d.giProfileInfo.JpegCount), resource.BinarySI)},
			"ofaEngines":  {Value: *resource.NewQuantity(int64(d.giProfileInfo.OfaCount), resource.BinarySI)},
			// MemorySizeMB 字段名含 "MB" 但实际单位是 MiB（2^20 字节），
			// 这是 NVML 源码中的文档错误，此处按 MiB 处理。
			// 将 MiB 转换为字节：MiB 数 * 1024 * 1024
			"memory": {Value: *resource.NewQuantity(int64(d.giProfileInfo.MemorySizeMB*1024*1024), resource.BinarySI)},
		},
	}

	// 为该 MIG 设备占用的每个内存切片（按放置位置）添加容量条目。
	// memorySliceN 容量值为 1，代表该切片被该设备完全占用。
	// 调度器可通过这些容量实现基于内存切片位置的精确调度。
	// 例如 PlacementStart=1, PlacementSize=2 时，会添加 memorySlice1 和 memorySlice2 两个容量条目。
	for i := d.PlacementStart; i < d.PlacementStart+d.PlacementSize; i++ {
		capacity := resourceapi.QualifiedName(fmt.Sprintf("memorySlice%d", i))
		device.Capacity[capacity] = resourceapi.DeviceCapacity{
			Value: *resource.NewQuantity(1, resource.BinarySI),
		}
	}

	// 若 PCIe 根桥属性存在，追加到设备属性中
	if d.pcieRootAttr != nil {
		device.Attributes[d.pcieRootAttr.Name] = d.pcieRootAttr.Value
	}

	// 若父 GPU 有寻址模式属性，MIG 设备继承该属性
	if d.parent.addressingMode != nil {
		device.Attributes["addressingMode"] = resourceapi.DeviceAttribute{
			StringValue: d.parent.addressingMode,
		}
	}

	return device
}

// GetDevice 将 VfioDeviceInfo 转换为 Kubernetes DRA API 的 resourceapi.Device 对象。
//
// 包含的设备属性：
//   - type：设备类型（VfioDeviceType，如 "vfio"）
//   - uuid：VFIO 设备的确定性 UUID
//   - deviceID：PCI 设备 ID
//   - vendorID：PCI 厂商 ID
//   - numa：NUMA 节点编号（整数类型）
//   - pciBusID：PCIe 总线地址
//   - productName：产品名称
//   - 可选：pcieRootAttr
//
// 包含的设备容量：
//   - addressableMemory：可寻址显存大小（字节）
//
// 返回值：
//   - resourceapi.Device：可发布到 ResourceSlice 中的 VFIO 设备对象
func (d *VfioDeviceInfo) GetDevice() resourceapi.Device {
	// 构造 PCIe 总线地址的限定名称（带标准前缀）
	pciBusIDAttrName := resourceapi.QualifiedName(deviceattribute.StandardDeviceAttributePrefix + "pciBusID")

	// 构造设备对象，包含所有 VFIO 设备属性
	device := resourceapi.Device{
		Name: d.CanonicalName(),
		Attributes: map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"type": {
				// 使用 VfioDeviceType 常量标识设备类型
				StringValue: ptr.To(VfioDeviceType),
			},
			"uuid": {
				StringValue: &d.UUID,
			},
			"deviceID": {
				StringValue: &d.deviceID,
			},
			"vendorID": {
				StringValue: &d.vendorID,
			},
			"numa": {
				// NUMA 节点使用整数类型属性
				IntValue: ptr.To(int64(d.numaNode)),
			},
			pciBusIDAttrName: {
				StringValue: &d.pcieBusID,
			},
			"productName": {
				StringValue: &d.productName,
			},
		},
		// VFIO 设备只有一个容量维度：可寻址显存
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"addressableMemory": {
				Value: *resource.NewQuantity(int64(d.addressableMemoryBytes), resource.BinarySI),
			},
		},
	}

	// 若 PCIe 根桥属性存在，追加到设备属性中
	if d.pcieRootAttr != nil {
		device.Attributes[d.pcieRootAttr.Name] = d.pcieRootAttr.Value
	}

	return device
}
