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

// types 包定义了 GPU DRA 驱动中的核心类型、常量和辅助函数。
//
// 本文件包含以下内容：
//   - 设备类型常量：区分 GPU、静态 MIG、动态 MIG、VFIO 和未知设备
//   - UUIDProvider 接口：提供统一的 UUID 查询抽象，供 MPS、TimeSlicing 等组件使用
//   - 字符串格式化工具函数：将 ResourceClaim 和 ClaimRef 转为可读字符串，用于日志输出
//
// 设备类型体系说明：
//   - GpuDeviceType ("gpu")：完整物理 GPU，是最基础的设备类型
//   - MigStaticDeviceType ("mig")：静态预创建的 MIG 设备，在插件启动前已由管理员手动创建
//   - MigDynamicDeviceType ("migdyn")：由 DynamicMIG 特性门控管理的抽象 MIG 设备，在 Prepare 时才真正创建
//   - VfioDeviceType ("vfio")：通过 vfio-pci 驱动直通的 GPU 设备
//
// 注意：MigStaticDeviceType 和 MigDynamicDeviceType 在 API 层面均呈现为 "mig"，
// 这是为了保持 Kubernetes API（ResourceSlice）的稳定性，避免引入新的设备类型导致旧版调度器不兼容。
// 内部逻辑需要区分两者时，使用 MigDynamicDeviceType（"migdyn"）。
package main

import (
	"fmt"

	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
)

// 设备类型常量，用于在内部逻辑中区分可分配设备的种类。
//
// 这些常量在以下场景中使用：
//   - AllocatableDevice.Type() 方法返回设备类型标识
//   - 业务逻辑中根据类型选择不同的处理路径（如 Prepare/Unprepare 流程）
//   - 日志输出中标注设备类型
const (
	// GpuDeviceType 表示完整物理 GPU 设备。
	// 每个 GpuDeviceType 对应节点上一个通过 NVML 枚举到的物理 GPU。
	// 在 ResourceSlice 中设备名为 "gpu-<minor>"，如 "gpu-0"。
	GpuDeviceType = "gpu"

	// MigStaticDeviceType 表示静态预创建的 MIG 设备。
	// 对外 API（ResourceSlice）中呈现为 "mig"，与 MigDynamicDeviceType 共享同一 API 标识。
	// 静态 MIG 设备在插件启动前已由管理员通过 nvidia-smi 创建，插件只负责发现和分配。
	MigStaticDeviceType = "mig"

	// MigDynamicDeviceType 表示由 DynamicMIG 特性门控管理的抽象 MIG 设备。
	// 仅在 DynamicMIG 特性开启时存在，代表"可以创建这种配置的 MIG 设备"的抽象描述。
	// 与 MigStaticDeviceType 的区别：
	//   - MigStatic：已物化的具体实例（有 UUID、GI ID、CI ID）
	//   - MigDynamic：未物化的抽象配置（无 UUID，在 Prepare 时才创建实例）
	// 对外 API 中两者均呈现为 "mig"，内部用 "migdyn" 区分。
	MigDynamicDeviceType = "migdyn"

	// VfioDeviceType 表示通过 vfio-pci 驱动直通的 GPU 设备。
	// 当 PassthroughSupport 特性门控开启时，某些 GPU 可从 nvidia 驱动模式
	// 切换到 vfio-pci 驱动模式，供虚拟机直接访问 GPU 硬件。
	VfioDeviceType = "vfio"

	// UnknownDeviceType 表示未知设备类型，用作兜底标识。
	// 正常运行时不应出现此值，出现则表明存在编程错误或数据损坏。
	UnknownDeviceType = "unknown"
)

// UUIDProvider 是一个接口，允许调用者在不关心设备具体类型（GPU/MIG）的情况下获取 UUID 列表。
//
// 设计动机：
//   - MPS（Multi-Process Service）管理器需要获取设备 UUID 来管理共享 GPU 访问
//   - TimeSlicing（时间切片）管理器需要获取物理 GPU UUID 来配置时间切片策略
//   - 这些组件操作的是"一组设备"而非"某种特定类型的设备"
//
// 通过该接口，调用方无需编写类型判断逻辑（如 if dev.Type() == GpuDeviceType），
// 直接调用 UUIDs()/GpuUUIDs()/MigDeviceUUIDs() 即可获取所需 UUID 列表。
//
// 当前实现者：
//   - AllocatableDevices（map[DeviceName]*AllocatableDevice）
type UUIDProvider interface {
	// UUIDs 返回全部设备（完整物理 GPU + MIG 设备）的 UUID 列表。
	// 返回值已排序以保证稳定性（便于日志对比和测试断言）。
	//
	// 注意：DynamicMIG 特性开启时，MigDynamic 设备无 UUID，调用此方法会 panic。
	// 此场景下调用方应确保只使用 GpuUUIDs()。
	UUIDs() []string

	// GpuUUIDs 仅返回完整物理 GPU 的 UUID 列表（不包含 MIG 设备）。
	// 返回值已排序。
	//
	// 适用场景：
	//   - MPS 管理器仅需管理物理 GPU 级别的共享
	//   - TimeSlicing 配置仅针对物理 GPU
	//   - 健康监控仅需监控物理 GPU 的 XID 错误
	GpuUUIDs() []string

	// MigDeviceUUIDs 仅返回 MIG 设备的 UUID 列表（不包含完整物理 GPU）。
	// 返回值已排序。
	//
	// 重要限制：DynamicMIG 特性门控开启时禁止调用此方法，会触发 panic。
	// 原因：MigDynamic 设备在 Prepare 前不存在 UUID，无法提供此信息。
	// DynamicMIG 开启时，MIG 设备的生命周期完全由 Prepare/Unprepare 驱动，
	// 无法提供稳定的 MIG UUID 列表。
	MigDeviceUUIDs() []string
}

// ResourceClaimToString 将 ResourceClaim 格式化为 "namespace/name:uid" 形式的字符串，用于日志输出。
//
// 格式示例："default/gpu-claim:a1b2c3d4-e5f6-7890-abcd-ef1234567890"
//
// 参数：
//   - rc: 待格式化的 ResourceClaim 指针，不可为 nil
//
// 返回值：
//   - string: 格式化后的字符串，可直接用于 klog.Infof 等日志调用
func ResourceClaimToString(rc *resourcev1.ResourceClaim) string {
	return fmt.Sprintf("%s/%s:%s", rc.Namespace, rc.Name, rc.UID)
}

// PreparedClaimToString 将已 Prepare 的 Claim 格式化为 "namespace/name:uid" 形式的字符串，用于日志输出。
//
// 与 ResourceClaimToString 功能类似，但输入参数不同：
//   - ResourceClaimToString 接收 *ResourceClaim（API 对象）
//   - PreparedClaimToString 接收 *PreparedClaim（内部状态对象）+ uid 字符串
//
// 这种差异存在是因为 PreparedClaim 是 Checkpoint 反序列化后的内部表示，
// 其 UID 字段以独立参数传入而非嵌入结构体中。
//
// 参数：
//   - pc: 已 Prepare 的 Claim 内部状态指针，包含 Namespace 和 Name 字段
//   - uid: Claim 的唯一标识符（由 kubelet 分配），如 "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
//
// 返回值：
//   - string: 格式化后的字符串
func PreparedClaimToString(pc *PreparedClaim, uid string) string {
	return fmt.Sprintf("%s/%s:%s", pc.Namespace, pc.Name, uid)
}

// ClaimsToStrings 将一组 ResourceClaim 批量转为可读字符串列表，用于日志记录。
//
// 当需要在单条日志中输出多个 Claim 时使用，避免为每个 Claim 单独输出一条日志。
// 例如：klog.Infof("Prepare called for: %v", ClaimsToStrings(claims))
//
// 参数：
//   - claims: ResourceClaim 指针切片，每个元素不可为 nil
//
// 返回值：
//   - []string: 每个元素格式为 "namespace/name:uid"，顺序与输入一致
func ClaimsToStrings(claims []*resourcev1.ResourceClaim) []string {
	var results []string
	// 遍历所有 Claim，逐一调用 ResourceClaimToString 格式化
	for _, c := range claims {
		results = append(results, ResourceClaimToString(c))
	}
	return results
}

// ClaimRefsToStrings 将一组 Claim 引用批量转为可读字符串列表，用于日志记录。
//
// 与 ClaimsToStrings 的区别：
//   - ClaimsToStrings 接收完整的 ResourceClaim 对象（来自 kubelet 的 NodePrepareResources RPC）
//   - ClaimRefsToStrings 接收 NamespacedObject 引用（来自 kubelet 的 NodeUnprepareResources RPC）
//
// Unprepare 流程中 kubelet 只传递 Claim 的引用（namespace/name/uid），
// 不传递完整的 ResourceClaim 对象，因此需要单独的转换函数。
//
// 参数：
//   - claimRefs: NamespacedObject 切片，每个元素包含 Namespace、Name、UID 字段
//
// 返回值：
//   - []string: 每个元素由 NamespacedObject.String() 生成
func ClaimRefsToStrings(claimRefs []kubeletplugin.NamespacedObject) []string {
	var results []string
	// 遍历所有 Claim 引用，调用其 String() 方法格式化
	for _, r := range claimRefs {
		results = append(results, r.String())
	}
	return results
}
