// checkpoint.go 实现了 DRA（Dynamic Resource Allocation）驱动插件的检查点（Checkpoint）机制。
//
// 本文件定义了 Checkpoint 结构体及其方法，负责将 ResourceClaim 的准备状态持久化到磁盘，
// 以支持插件崩溃重启后的状态恢复和滚动升级期间的新旧版本兼容。
//
// 核心设计原则——双版本并存策略：
//   - 每次序列化时同时写入 V1（旧版，供旧版插件读取）和 V2（新版，含完整状态信息）两种格式
//   - 新版插件优先读 V2，旧版插件只读 V1
//   - V1 和 V2 各自拥有独立的校验和（Checksum），互不干扰
//   - 当 V1 格式完全退役后，可安全移除相关代码
//
// 实现的接口：kubelet checkpointmanager.Checkpoint
//   - MarshalCheckpoint()：序列化到 JSON
//   - UnmarshalCheckpoint()：从 JSON 反序列化
//   - VerifyChecksum()：验证数据完整性
package main

import (
	"encoding/json"
	"fmt"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

// Checkpoint 是持久化到磁盘的检查点顶层结构，实现了 kubelet checkpointmanager.Checkpoint 接口。
//
// 设计目标：支持插件在新旧版本之间平滑升级（滚动更新）期间的 Checkpoint 兼容性。
//
// 核心策略：每次序列化时同时写入 V1 和 V2 两种格式：
//   - 新版插件：优先读 V2（含完整状态信息）
//   - 旧版插件：只读 V1（仅含完成的 Claim，格式简单）
//
// Checksum 字段属于 V1 层面的校验和（覆盖 V1 字段 + 顶层 Checksum 字段本身以外的所有字段）。
// V2 有自己的内嵌 Checksum（CheckpointV2.Checksum），两者相互独立，
// 使得旧版插件降级读 V1 时也能通过自己的校验。
type Checkpoint struct {
	// Checksum 是针对 V1 格式的校验和（不含 V2 字段）。
	// 计算方式：将 V2 字段临时置 nil，序列化结构，计算 CRC。
	// V1 格式完全退役后此字段将被移除。
	Checksum checksum.Checksum `json:"checksum"`

	// V1 是旧版 Checkpoint 数据，供旧版插件读取（滚动升级兼容）。
	// 序列化时由 MarshalCheckpoint 自动从 V2 降级生成，无需手动维护。
	V1 *CheckpointV1 `json:"v1,omitempty"`

	// V2 是当前主力 Checkpoint 数据，含状态机字段和 Name/Namespace 信息。
	// 读取时优先使用 V2；若 V2 为 nil 则从 V1 升级。
	V2 *CheckpointV2 `json:"v2,omitempty"`
}

// ToLatestVersion 将任意版本的 Checkpoint 迁移到最新版（V2），用于读取后的规范化处理。
// 处理三种情况：
//   - V2 已存在：直接使用（最常见情况）
//   - 仅 V1 存在：执行 V1→V2 升级转换（旧版本数据兼容）
//   - 两者均为 nil：创建空的 V2（首次启动场景）
//
// 副作用：确保 PreparedClaims map 已初始化（避免后续插入操作的 nil map panic）。
func (cp *Checkpoint) ToLatestVersion() *Checkpoint {
	// 创建新的 Checkpoint 实例，避免修改原始数据
	latest := &Checkpoint{}
	switch {
	case cp.V2 != nil:
		// V2 已存在，直接引用（最常见路径：新版插件读取自己写入的 Checkpoint）
		latest.V2 = cp.V2
	case cp.V1 != nil:
		// 仅 V1 存在，执行 V1→V2 升级转换
		// 典型场景：从旧版插件遗留的 Checkpoint 文件中恢复状态
		latest.V2 = cp.V1.ToV2()
	default:
		// 两者均为 nil，创建空 V2
		// 典型场景：插件首次启动，磁盘上无 Checkpoint 文件
		latest.V2 = &CheckpointV2{}
	}
	// 确保 PreparedClaims map 已初始化
	// 如果反序列化的 JSON 中 "preparedClaims" 为空或缺失，map 会是 nil
	// 后续代码若直接对 nil map 执行插入操作会导致 panic
	if latest.V2.PreparedClaims == nil {
		latest.V2.PreparedClaims = make(PreparedClaimsByUID)
	}
	return latest
}

// MarshalCheckpoint 将 Checkpoint 序列化为 JSON 字节切片，实现 checkpointmanager.Checkpoint 接口。
// 序列化流程：
//  1. 升级到最新版本（确保 V2 数据完整）
//  2. 从 V2 降级生成 V1 副本（供旧版插件读取）
//  3. 分别计算 V1 和 V2 的校验和（各自独立，互不干扰）
//  4. JSON 序列化整个 Checkpoint 结构（含 V1、V2、顶层 Checksum）
//
// 这种"同时写双版本"的策略是滚动升级期间兼容性的关键：
// 新版插件写的文件旧版插件能读，确保升级期间不丢失设备状态。
func (cp *Checkpoint) MarshalCheckpoint() ([]byte, error) {
	// 步骤 1：升级到最新版本，确保 V2 数据完整
	cp = cp.ToLatestVersion()

	// 步骤 2：从 V2 降级生成 V1 副本
	// 只保留 PrepareCompleted 状态的 Claim，丢弃 PrepareStarted 条目
	// 因为旧版插件不理解 PrepareStarted 状态
	cp.V1 = cp.V2.ToV1()

	// 步骤 3a：计算 V1 校验和
	if err := cp.SetChecksumV1(); err != nil {
		return nil, fmt.Errorf("error setting v1 checksum: %v", err)
	}

	// 步骤 3b：计算 V2 校验和
	if err := cp.SetChecksumV2(); err != nil {
		return nil, fmt.Errorf("error setting v2 checksum: %v", err)
	}

	// 步骤 4：JSON 序列化
	return json.Marshal(*cp)
}

// SetChecksumV1 计算并设置 V1 格式的校验和（覆盖顶层 Checksum 以外的所有字段）。
// 计算时临时移除 V2 字段（避免其内容干扰 V1 校验和），计算完成后恢复。
// 使用 defer 确保即使 json.Marshal 失败也能正确恢复 V2 字段。
func (cp *Checkpoint) SetChecksumV1() error {
	// 保存 V2 指针，后续恢复用
	v2 := cp.V2
	// 临时将 V2 置 nil，使其不参与 V1 校验和的计算
	cp.V2 = nil
	defer func() {
		// 无论后续操作成功还是失败，都恢复 V2 字段
		cp.V2 = v2
	}()

	// 将顶层 Checksum 置零，确保其不参与自身的计算
	cp.Checksum = 0
	// 序列化仅含 V1 字段的 Checkpoint 结构
	out, err := json.Marshal(*cp)
	if err != nil {
		return err
	}
	// 基于序列化结果计算 CRC 校验和
	cp.Checksum = checksum.New(out)
	return nil
}

// SetChecksumV2 计算并设置 V2 内嵌 Checksum。
// 计算时临时将 V2.Checksum 置零（避免"自引用"循环），计算完成后存入 V2.Checksum。
func (cp *Checkpoint) SetChecksumV2() error {
	// 将 V2 内嵌的 Checksum 置零，避免其参与自身的校验和计算
	cp.V2.Checksum = 0
	// 序列化 V2 结构
	out, err := json.Marshal(*cp.V2)
	if err != nil {
		return err
	}
	// 基于序列化结果计算 CRC 校验和，并存入 V2.Checksum
	cp.V2.Checksum = checksum.New(out)
	return nil
}

// UnmarshalCheckpoint 将 JSON 字节切片反序列化为 Checkpoint 结构，实现 checkpointmanager.Checkpoint 接口。
// 反序列化后通常需要调用 ToLatestVersion() 和 VerifyChecksum() 进行版本规范化和完整性验证。
func (cp *Checkpoint) UnmarshalCheckpoint(data []byte) error {
	return json.Unmarshal(data, cp)
}

// VerifyChecksum 依次验证 V1 和 V2 校验和，保护 Checkpoint 文件的完整性。
// 任意一个校验和验证失败则返回错误，拒绝加载损坏的 Checkpoint。
// 这可以防止磁盘静默损坏（如存储设备写错误、文件系统元数据损坏）导致的数据污染。
func (cp *Checkpoint) VerifyChecksum() error {
	// 先验证 V1 格式的校验和
	if err := cp.VerifyChecksumV1(); err != nil {
		return err
	}
	// 再验证 V2 格式的校验和
	if err := cp.VerifyChecksumV2(); err != nil {
		return err
	}
	return nil
}

// VerifyChecksumV1 验证 V1 格式的校验和。
// 验证方式：临时将 Checksum 和 V2 字段置零/nil，重新序列化，与保存的 Checksum 对比。
// 使用 defer 确保验证完成后恢复这两个字段（无论验证成功还是失败）。
func (cp *Checkpoint) VerifyChecksumV1() error {
	// 保存原始值，验证完成后恢复
	ck := cp.Checksum
	v2 := cp.V2
	// 临时移除 V2 字段，使其不参与 V1 校验和的重新计算
	cp.V2 = nil
	defer func() {
		// 恢复原始值
		cp.Checksum = ck
		cp.V2 = v2
	}()

	// 将 Checksum 置零后重新序列化，计算期望的校验和
	cp.Checksum = 0
	out, err := json.Marshal(*cp)
	if err != nil {
		return err
	}

	// 将重新计算的校验和与保存的值进行对比
	// ck.Verify 内部会比较 CRC 值，不匹配时返回错误
	return ck.Verify(out)
}

// VerifyChecksumV2 验证 V2 内嵌 Checksum。
// 若 V2 为 nil（纯 V1 格式的旧文件），跳过验证直接返回 nil。
// 验证方式：临时将 V2.Checksum 置零，重新序列化 V2，与保存的 Checksum 对比。
func (cp *Checkpoint) VerifyChecksumV2() error {
	// 如果 V2 为 nil，说明这是一个纯 V1 格式的旧 Checkpoint 文件
	// 此时没有 V2 数据需要验证，直接跳过
	if cp.V2 == nil {
		return nil
	}

	// 保存原始 V2.Checksum 值
	ck := cp.V2.Checksum
	defer func() {
		// 验证完成后恢复原始 Checksum 值
		cp.V2.Checksum = ck
	}()
	// 将 V2.Checksum 置零后重新序列化，计算期望的校验和
	cp.V2.Checksum = 0
	out, err := json.Marshal(*cp.V2)
	if err != nil {
		return err
	}
	// 将重新计算的校验和与保存的值进行对比
	return ck.Verify(out)
}
