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

// partitions 包实现了 KEP 4815 (Partitionable Devices) 中 GPU 和 MIG 设备的分区能力公告。
//
// KEP 4815 核心概念：
//   - Partitionable Devices 允许将一个物理设备分割为多个逻辑分区（如 MIG），
//     每个分区消耗父设备的部分资源，通过 CounterSet 机制跟踪资源占用。
//   - SharedCounterSet 定义了父设备的总资源容量，子分区通过 ConsumesCounters 声明
//     从父设备的 CounterSet 中消耗的资源量。
//   - 调度器根据 CounterSet 的剩余容量判断是否还能分配新的分区，防止资源超卖。
//
// 本文件包含：
//   - PartCapacityMap: 设备容量维度到 DeviceCapacity 的映射类型定义
//   - GpuInfo 的 KEP 4815 方法：PartDevAttributes/PartCapacities/PartSharedCounterSets/PartConsumesCounters/PartGetDevice
//   - MigSpec 的 KEP 4815 方法：PartGetDevice/PartCapacities/PartAttributes/PartConsumesCounters
//   - 辅助函数：capacitiesToCounters/addCountersForMemSlices/memsliceCounterName/intcap
package main

import (
	"fmt"

	"github.com/Masterminds/semver"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/api/validate/constraints"
	"k8s.io/dynamic-resource-allocation/deviceattribute"
	"k8s.io/utils/ptr"
)

// PartCapacityMap 是设备容量维度到 DeviceCapacity 的映射，用于 KEP 4815 Partitionable Devices 公告。
//
// KEP 4815 背景：Partitionable Devices 允许将一个物理设备分割为多个逻辑分区（如 MIG），
// 每个分区消耗父设备的部分资源（通过 CounterSet 机制跟踪资源占用）。
// PartCapacityMap 表示一个设备（全量 GPU 或 MIG 分区）拥有或消耗的各维度容量。
//
// 容量维度示例：multiprocessors/copyEngines/decoders/encoders/jpegEngines/ofaEngines/memory。
type PartCapacityMap map[resourceapi.QualifiedName]resourceapi.DeviceCapacity

// PartDevAttributes 返回完整物理 GPU 的设备属性集合（KEP 4815 格式）。
//
// 这些属性会写入 ResourceSlice，调度器在 DeviceClass 的 selector 中可按这些属性筛选设备。
// 包含的属性（用于调度器过滤）：
//   - type: "gpu"（设备类型标识符）
//   - uuid: GPU 的全局唯一标识符
//   - productName: GPU 型号（如 "NVIDIA H100 80GB HBM3"）
//   - brand: GPU 品牌（如 "NVIDIA"）
//   - architecture: GPU 架构（如 "Hopper"）
//   - cudaComputeCapability: CUDA 计算能力版本（如 "9.0.0"），以 semver 格式存储
//   - driverVersion: NVIDIA 驱动版本（如 "555.42.06"），以 semver 格式存储
//   - cudaDriverVersion: CUDA 驱动版本（如 "12.5.0"），以 semver 格式存储
//   - x.dra.k8s.io/pciBusID: PCIe 总线 ID（如 "0000:00:03.0"），使用标准 DRA 前缀
func (d *GpuInfo) PartDevAttributes() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	// 构建 PCIe 总线 ID 属性名，使用 DRA 标准设备属性前缀
	pciBusIDAttrName := resourceapi.QualifiedName(deviceattribute.StandardDeviceAttributePrefix + "pciBusID")
	return map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		// 设备类型标识符，用于调度器区分 GPU 和其他设备类型
		"type": {
			StringValue: ptr.To(GpuDeviceType),
		},
		// GPU 的全局唯一标识符，NVML 分配，跨重启稳定
		"uuid": {
			StringValue: &d.UUID,
		},
		// GPU 产品名称，如 "NVIDIA H100 80GB HBM3"
		"productName": {
			StringValue: &d.productName,
		},
		// GPU 品牌，通常为 "NVIDIA"
		"brand": {
			StringValue: &d.brand,
		},
		// GPU 微架构名称，如 "Hopper"、"Ampere"
		"architecture": {
			StringValue: &d.architecture,
		},
		// CUDA 计算能力版本，必须为合法 semver 格式（如 "9.0.0"）
		"cudaComputeCapability": {
			VersionValue: ptr.To(semver.MustParse(d.cudaComputeCapability).String()),
		},
		// NVIDIA 驱动版本号，必须为合法 semver 格式
		"driverVersion": {
			VersionValue: ptr.To(semver.MustParse(d.driverVersion).String()),
		},
		// CUDA 驱动版本号，必须为合法 semver 格式
		"cudaDriverVersion": {
			VersionValue: ptr.To(semver.MustParse(d.cudaDriverVersion).String()),
		},
		// PCIe 总线地址，使用 DRA 标准前缀命名
		pciBusIDAttrName: {
			StringValue: &d.pcieBusID,
		},
	}
}

// PartCapacities 返回完整物理 GPU 的资源容量映射（KEP 4815 格式）。
//
// 容量值来自扫描所有 MIG Profile 后取各维度最大值（即完整 GPU 的全量容量）。
// 这些容量值会作为 SharedCounterSet 的计数器总量，供 MIG 分区按需消耗。
//
// 注意：此方法返回的是通过 inspectMigProfilesAndPlacements() 计算出的 maxCapacities，
// 而不是直接查询 NVML 的 GPU 资源总量。两者在正常情况下应该一致。
func (d *GpuInfo) PartCapacities() PartCapacityMap {
	return d.maxCapacities
}

// GetSharedCounterSetName 返回该完整 GPU 对应的 SharedCounterSet 名称。
// 格式：gpu-<minor>-counter-set（如 "gpu-0-counter-set"）
// MIG 分区在 PartConsumesCounters() 中通过此名称引用父 GPU 的计数器集。
func (d *GpuInfo) GetSharedCounterSetName() string {
	return toRFC1123Compliant(fmt.Sprintf("%s-counter-set", d.CanonicalName()))
}

// PartSharedCounterSets 返回该完整 GPU 的 SharedCounterSet 列表（KEP 4815 格式）。
//
// 设计原则：当前每个物理 GPU 对应恰好一个 CounterSet。
// CounterSet 包含：
//   1. 各容量维度的计数器（来自 maxCapacities）：multiprocessors/copyEngines/decoders/encoders/jpegEngines/ofaEngines/memory
//   2. 内存切片计数器（每个切片一个计数器，初始值为 1）：memory-slice-0, memory-slice-1, ...
//
// 内存切片计数器允许调度器追踪哪些物理内存切片已被 MIG 分区占用，
// 防止两个 MIG 设备使用重叠的内存切片（物理上不可能同时使用同一块显存）。
func (d *GpuInfo) PartSharedCounterSets() []resourceapi.CounterSet {
	return []resourceapi.CounterSet{{
		// CounterSet 名称，MIG 分区通过此名称引用
		Name: d.GetSharedCounterSetName(),
		// 计数器映射：包含容量计数器 + 内存切片计数器
		Counters: addCountersForMemSlices(capacitiesToCounters(d.maxCapacities), 0, d.memSliceCount),
	}}
}

// PartConsumesCounters 返回完整物理 GPU 被分配时消耗的计数器列表（KEP 4815 格式）。
//
// 设计目标：
//   1. 当完整 GPU 被分配时，所有可用计数器降为零（不能再分配任何 MIG 设备）
//   2. 当任意 MIG 分区被分配时，完整 GPU 不能再被分配（反向约束）
//
// 实现方式：让完整 GPU 消耗与其 SharedCounterSet 相同的全部计数器，
// 这样任何 MIG 分区占用计数器后，完整 GPU 所需的计数器就不足了（无法满足全部消耗需求）。
//
// 返回值：恰好一个 DeviceCounterConsumption，引用自身 CounterSet 的全部计数器
func (d *GpuInfo) PartConsumesCounters() []resourceapi.DeviceCounterConsumption {
	return []resourceapi.DeviceCounterConsumption{{
		// 引用自身的 CounterSet
		CounterSet: d.GetSharedCounterSetName(),
		// 消耗全部计数器（容量 + 内存切片），与 SharedCounterSet 中的定义完全一致
		Counters:   addCountersForMemSlices(capacitiesToCounters(d.maxCapacities), 0, d.memSliceCount),
	}}
}

// PartGetDevice 返回完整物理 GPU 的 KEP 4815 格式设备描述。
// 包含 Attributes（属性）、Capacity（容量）和 ConsumesCounters（消耗计数器），
// 允许调度器在分配 GPU 时正确扣减 SharedCounterSet 中的计数器。
//
// 与 GetDevice() 的区别：
//   - GetDevice() 返回传统 DRA 格式（无 ConsumesCounters），用于非 DynamicMIG 模式
//   - PartGetDevice() 返回 KEP 4815 格式（含 ConsumesCounters），用于 DynamicMIG 模式
func (d *GpuInfo) PartGetDevice() resourceapi.Device {
	dev := resourceapi.Device{
		// 设备规范名称，与 ResourceSlice 中的设备名一致
		Name:             d.CanonicalName(),
		// 设备属性（类型、UUID、型号等）
		Attributes:       d.PartDevAttributes(),
		// 设备容量（各维度资源量）
		Capacity:         d.PartCapacities(),
		// 设备消耗的计数器列表（完整 GPU 消耗全部计数器）
		ConsumesCounters: d.PartConsumesCounters(),
	}

	// PCIe 根桥属性（pcieRootBusID）并非所有环境都支持，仅在可用时添加
	if d.pcieRootAttr != nil {
		dev.Attributes[d.pcieRootAttr.Name] = d.pcieRootAttr.Value
	}

	return dev
}

// PartGetDevice 返回抽象 MIG 设备的 KEP 4815 格式设备描述。
// 与完整 GPU 不同，MIG 设备没有独立的 SharedCounterSet，
// 而是通过 ConsumesCounters 从父 GPU 的 CounterSet 消耗资源。
//
// MIG 设备的 Capacity 表示自身的资源容量（用于调度器展示），
// ConsumesCounters 表示从父 GPU CounterSet 中消耗的资源量（用于互斥约束）。
func (i *MigSpec) PartGetDevice() resourceapi.Device {
	d := resourceapi.Device{
		// MIG 设备的规范名称，编码了父 GPU、Profile 和放置信息
		Name:             i.CanonicalName(),
		// MIG 设备属性（类型、父 UUID、Profile 等）
		Attributes:       i.PartAttributes(),
		// MIG 设备容量（各维度资源量，来自 NVML Profile 信息）
		Capacity:         i.PartCapacities(),
		// MIG 设备消耗的计数器（从父 GPU CounterSet 中扣减）
		ConsumesCounters: i.PartConsumesCounters(),
	}
	return d
}

// PartCapacities 返回抽象 MIG 设备的资源容量映射（KEP 4815 格式）。
//
// 容量值直接来自 NVML GpuInstanceProfileInfo，代表该 MIG Profile 的各维度资源量。
//
// 内存单位说明：NVML 中 MemorySizeMB 的文档写的是 "MB"，但实际上是 MiB（2^20 字节）。
// 这是 NVML 文档中的错误（已在 nvml.h 中修正），此处按 MiB 处理：×1024×1024。
//
// 关于是否在 capacity 中列出内存切片：
// 内存切片与 ConsumesCounters 中的 memory-slice-N 计数器语义重叠，存在信息冗余。
// 目前的实现选择不在 PartCapacities 中列出内存切片，仅在 ConsumesCounters 中表达。
func (i MigSpec) PartCapacities() PartCapacityMap {
	// GIProfileInfo 是 NVML 返回的 GPU Instance Profile 信息，
	// 包含该 Profile 对应的各维度硬件资源数量
	p := i.GIProfileInfo
	return PartCapacityMap{
		// 多处理器（SM）数量
		"multiprocessors": intcap(p.MultiprocessorCount),
		// 拷贝引擎数量
		"copyEngines":     intcap(p.CopyEngineCount),
		// 视频解码器数量
		"decoders":        intcap(p.DecoderCount),
		// 视频编码器数量
		"encoders":        intcap(p.EncoderCount),
		// JPEG 解码引擎数量
		"jpegEngines":     intcap(p.JpegCount),
		// 光流加速器数量
		"ofaEngines":      intcap(p.OfaCount),
		// 显存大小（MiB → 字节），注意 NVML 文档错误：MB 实为 MiB
		"memory":          intcap(int64(p.MemorySizeMB * 1024 * 1024)),
	}
}

// PartAttributes 返回抽象 MIG 设备的属性集合（KEP 4815 格式）。
//
// 包含的属性：
//   - type: "mig"（设备类型标识符）
//   - parentUUID: 父 GPU 的 UUID（用于关联父子设备）
//   - profile: MIG Profile 字符串（如 "1g.10gb"，用于调度器按 Profile 过滤）
//   - productName/brand/architecture: 继承自父 GPU（调度器可能需要按 GPU 型号筛选 MIG）
//   - cudaComputeCapability/driverVersion/cudaDriverVersion: 继承自父 GPU
//   - x.dra.k8s.io/pciBusID: 继承父 GPU 的 PCIe 总线 ID（MIG 无独立 PCIe 地址）
//
// 注意：MIG 设备没有独立的 UUID（物化前），因此不包含 uuid 属性。
// 物化后的 MIG 实例（PreparedMigDevice）有 UUID，但那是 PreparedDevice 层面的概念，
// 在 AllocatableDevice 阶段 MIG 设备仍然是抽象描述。
func (i MigSpec) PartAttributes() map[resourceapi.QualifiedName]resourceapi.DeviceAttribute {
	// 构建 PCIe 总线 ID 属性名，使用 DRA 标准设备属性前缀
	pciBusIDAttrName := resourceapi.QualifiedName(deviceattribute.StandardDeviceAttributePrefix + "pciBusID")
	return map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		// 设备类型标识符，MIG 设备统一使用 "mig"
		"type": {
			StringValue: ptr.To("mig"),
		},
		// 父 GPU 的 UUID，用于建立 MIG 设备与父 GPU 的关联关系
		"parentUUID": {
			StringValue: &i.Parent.UUID,
		},
		// MIG Profile 字符串，如 "1g.10gb"、"3g.40gb" 等
		"profile": {
			StringValue: ptr.To(i.Profile.String()),
		},
		// 以下属性均继承自父 GPU，调度器可能需要根据 GPU 型号筛选 MIG 设备
		"productName": {
			StringValue: &i.Parent.productName,
		},
		"brand": {
			StringValue: &i.Parent.brand,
		},
		"architecture": {
			StringValue: &i.Parent.architecture,
		},
		"cudaComputeCapability": {
			VersionValue: ptr.To(semver.MustParse(i.Parent.cudaComputeCapability).String()),
		},
		"driverVersion": {
			VersionValue: ptr.To(semver.MustParse(i.Parent.driverVersion).String()),
		},
		"cudaDriverVersion": {
			VersionValue: ptr.To(semver.MustParse(i.Parent.cudaDriverVersion).String()),
		},
		// MIG 设备共享父 GPU 的 PCIe 地址
		pciBusIDAttrName: {
			StringValue: &i.Parent.pcieBusID,
		},
	}
}

// capacitiesToCounters 将 PartCapacityMap 转换为计数器映射，用于 SharedCounterSet/ConsumesCounters。
//
// 转换规则：
//   - 容量键名通过 camelToDNSName 转换为 DNS 命名格式
//     （如 "multiprocessors" → "multiprocessors"，"copyEngines" → "copy-engines"）
//   - 自动从容量名派生计数器名，确保 SharedCounterSet 和 ConsumesCounters 中的计数器名一致
//
// 参数：
//   - m: 源容量映射
//
// 返回值：计数器名称到 Counter 对象的映射
func capacitiesToCounters(m PartCapacityMap) map[string]resourceapi.Counter {
	counters := make(map[string]resourceapi.Counter)
	for name, cap := range m {
		// 将 camelCase 的容量名转换为 DNS 命名格式（小写+连字符），
		// 确保与 DRA API 的命名规范一致
		counters[camelToDNSName(string(name))] = resourceapi.Counter{Value: cap.Value}
	}
	return counters
}

// PartConsumesCounters 返回抽象 MIG 设备被分配时从父 GPU CounterSet 消耗的计数器列表。
//
// 消耗内容：
//   1. 各容量维度的计数器（multiprocessors/copy-engines/decoders/encoders 等）
//   2. 该 MIG Profile 放置区间内的内存切片计数器（memory-slice-<N>）
//
// 示例：B200 上的 "3g.90gb" MIG Profile 消耗 4 个连续内存切片和 3 个 SM 切片。
// 当两个 "3g.90gb" 设备都被分配后，所有 8 个内存切片计数器归零，
// 但还有 1 个 SM 切片剩余（无法再被任何 MIG Profile 使用，成为"未使用容量"）。
//
// CounterSet 引用约定：当前每个完整 GPU 恰好有一个 CounterSet，名称为 'gpu-<minor>-counter-set'。
func (i MigSpec) PartConsumesCounters() []resourceapi.DeviceCounterConsumption {
	return []resourceapi.DeviceCounterConsumption{{
		// 引用父 GPU 的 CounterSet 名称
		CounterSet: i.Parent.GetSharedCounterSetName(),
		// 将 MIG 的容量维度转为计数器 + 添加放置区间内的内存切片计数器
		Counters:   addCountersForMemSlices(capacitiesToCounters(i.PartCapacities()), int(i.Placement.Start), int(i.Placement.Size)),
	}}
}

// PartGetDevice 是 AllocatableDevice 的 KEP 4815 格式版本入口，根据设备类型调用对应实现。
//
// 当前支持情况：
//   - GpuDeviceType: 委托给 GpuInfo.PartGetDevice()
//   - MigDynamicDeviceType: 委托给 MigSpec.PartGetDevice()
//   - MigStaticDeviceType: 不支持（KEP 4815 仅用于 DynamicMIG），调用会 panic
//   - VfioDeviceType: 尚未实现，调用会 panic
//
// 注意：MigStatic 和 VfioDeviceType 的 panic 属于编程错误，
// 调用方必须保证在正确的场景下调用此方法。
func (d *AllocatableDevice) PartGetDevice() resourceapi.Device {
	switch d.Type() {
	case GpuDeviceType:
		return d.Gpu.PartGetDevice()
	case MigStaticDeviceType:
		// 静态 MIG 不使用 KEP 4815 格式，因为静态 MIG 不需要 CounterSet 机制
		panic("PartGetDevice() called for MigStaticDeviceType")
	case MigDynamicDeviceType:
		return d.MigDynamic.PartGetDevice()
	case VfioDeviceType:
		// VFIO 设备的 KEP 4815 支持尚未实现
		panic("not yet implemented")
	}
	panic("unexpected type for AllocatableDevice")
}

// addCountersForMemSlices 为内存切片放置区间（[start, start+size)）添加计数器。
//
// 每个内存切片生成一个计数器，计数器名格式为 "memory-slice-<N>"，初始值为 1。
// 直接修改传入的 counters map 并返回，方便链式调用。
//
// 参数：
//   - counters: 已有的计数器映射（会被就地修改）
//   - start: 内存切片放置的起始索引（包含）
//   - size: 内存切片放置的数量（区间长度）
//
// 返回值：修改后的 counters 映射（与输入引用相同）
func addCountersForMemSlices(counters map[string]resourceapi.Counter, start int, size int) map[string]resourceapi.Counter {
	for i := start; i < start+size; i++ {
		// 每个内存切片计数器值为 1，表示该切片可被一个 MIG 设备占用
		counters[memsliceCounterName(i)] = resourceapi.Counter{Value: *resource.NewQuantity(1, resource.BinarySI)}
	}
	return counters
}

// memsliceCounterName 返回第 i 个内存切片（从 0 开始）的计数器规范名称。
//
// 格式："memory-slice-<i>"（含连字符，符合 DRA 计数器命名规范）。
// SharedCounterSet 和 ConsumesCounters 中的内存切片计数器必须使用此函数生成的相同名称，
// 否则调度器无法正确匹配和扣减计数器。
//
// 参数：
//   - i: 内存切片的索引号（0 起始）
//
// 返回值：规范化的计数器名称字符串
func memsliceCounterName(i int) string {
	return fmt.Sprintf("memory-slice-%d", i)
}

// intcap 将任意整数类型转换为 resourceapi.DeviceCapacity，封装 resource.Quantity 创建。
//
// 泛型参数 T 接受任何整数类型（int/int32/int64/uint32 等），
// 避免为每种整数类型编写重复的类型转换代码。
// 内部统一转为 int64 后创建 BinarySI 格式的 Quantity。
//
// 参数：
//   - i: 任意整数类型的值
//
// 返回值：包含对应 Quantity 的 DeviceCapacity 对象
func intcap[T constraints.Integer](i T) resourceapi.DeviceCapacity {
	return resourceapi.DeviceCapacity{Value: *resource.NewQuantity(int64(i), resource.BinarySI)}
}
