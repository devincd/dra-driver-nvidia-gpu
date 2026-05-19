// checkpointv.go 定义了 Checkpoint 的 V1 和 V2 两种数据格式，以及版本间的相互转换逻辑。
//
// V1（已废弃）是旧版格式，不含 CheckpointState 字段，无法区分"准备中"和"准备完成"状态，
// 也不含 Name/Namespace 字段，无法按名称查询 Claim。保留仅为向后兼容。
//
// V2（当前主力）新增了：
//   - CheckpointState 字段：支持两阶段提交（PrepareStarted / PrepareCompleted）
//   - Name/Namespace 字段：支持通过 name+namespace 向 API Server 查询 Claim（用于清理）
//   - 内嵌 Checksum：V2 自带独立校验和，即使 V1 字段损坏也能验证 V2 数据完整性
//
// 版本转换规则：
//   - V1→V2：所有条目设为 PrepareCompleted（V1 写入时只记录完成的 Claim）
//   - V2→V1：只保留 PrepareCompleted 状态的 Claim，丢弃 PrepareStarted 条目
package main

import (
	resourceapi "k8s.io/api/resource/v1"

	"k8s.io/kubernetes/pkg/kubelet/checkpointmanager/checksum"
)

// ClaimCheckpointState 描述一个 ResourceClaim 在 Checkpoint 中的准备状态。
// 采用两阶段提交设计，状态流转：Unset → PrepareStarted → PrepareCompleted。
//
// 状态说明：
//   - ClaimCheckpointStateUnset：未设置（旧版 V1 Checkpoint 的兼容值，视为 PrepareCompleted）
//   - ClaimCheckpointStatePrepareStarted：设备准备已开始但尚未完成（事务中间态）。
//     此状态下插件崩溃，重启后需执行回滚或恢复。
//   - ClaimCheckpointStatePrepareCompleted：设备已完全准备好（CDI spec 写入、Checkpoint 更新完成）。
//     只有处于此状态的 Claim，其设备才被认为是可用的。
//
// 这种两阶段 Checkpoint 机制保证了崩溃一致性：
//   - 若插件在 PrepareStarted 后崩溃，CleanupManager 会检测并清理这些"悬挂"Claim。
//   - 若插件在 PrepareCompleted 后崩溃，重启时直接从 Checkpoint 恢复，避免重复准备。
type ClaimCheckpointState string

const (
	// ClaimCheckpointStateUnset 是旧版（V1）Checkpoint 条目的兼容值。
	// V1 格式不含 CheckpointState 字段，反序列化后该字段为零值（空字符串）。
	// ToV2() 转换函数将其视为 PrepareCompleted 处理。
	ClaimCheckpointStateUnset ClaimCheckpointState = ""

	// ClaimCheckpointStatePrepareStarted 表示设备准备已启动但尚未完成（事务中间态）。
	// 在此状态下记录了 Claim 的 Name/Namespace/Status，但 PreparedDevices 为空
	// （MIG 设备可能已部分创建，CDI spec 尚未写入）。
	ClaimCheckpointStatePrepareStarted ClaimCheckpointState = "PrepareStarted"

	// ClaimCheckpointStatePrepareCompleted 表示设备已完全准备好（稳定态）。
	// 此状态下 PreparedDevices 完整且 CDI spec 文件已写入磁盘。
	// Unprepare 只处理处于此状态的 Claim（通过正常流程），
	// 处于 PrepareStarted 状态的 Claim 则走回滚路径。
	ClaimCheckpointStatePrepareCompleted ClaimCheckpointState = "PrepareCompleted"
)

// 最新版 Checkpoint 类型别名，使业务代码无需关心版本号。
// 当 Checkpoint 格式升级时（如从 V2 升级到 V3），只需更新这两个别名，
// 所有使用 PreparedClaimsByUID / PreparedClaim 的代码自动获得新版类型。
type PreparedClaimsByUID = PreparedClaimsByUIDV2
type PreparedClaim = PreparedClaimV2

// CheckpointV2 是当前使用的主力 Checkpoint 格式（相对于已废弃的 V1）。
// 与 V1 相比，新增了以下能力：
//   - CheckpointState 字段：支持两阶段提交（PrepareStarted / PrepareCompleted）
//   - Name/Namespace 字段：支持通过 name+namespace 向 API Server 查询 Claim（用于清理）
//   - 内嵌 Checksum：V2 自带独立校验和，即使 V1 字段损坏也能验证 V2 数据完整性
type CheckpointV2 struct {
	// Checksum 是 V2 数据的 CRC 校验和，序列化时自动计算，反序列化后自动验证。
	// 用于检测磁盘静默损坏或文件截断。
	Checksum       checksum.Checksum     `json:"checksum"`
	PreparedClaims PreparedClaimsByUIDV2 `json:"preparedClaims,omitempty"`
}

// PreparedClaimsByUIDV2 是 Claim UID（字符串）到 PreparedClaimV2 的映射。
// 使用 UID 而非 Name 作为 key，是因为同名 Claim 可能先后被删除重建（UID 不同），
// 如果用 Name 作 key，新建的 Claim 会覆盖旧 Claim 的准备状态，导致旧设备无法清理。
type PreparedClaimsByUIDV2 map[string]PreparedClaimV2

// PreparedClaimV2 记录单个 ResourceClaim 的完整准备状态，持久化到 Checkpoint 文件。
// 序列化后存储在 checkpoint.json 中，供插件崩溃重启后恢复状态。
type PreparedClaimV2 struct {
	// CheckpointState 标记该 Claim 处于哪个准备阶段（事务状态机）。
	// 取值为 ClaimCheckpointStateUnset / PrepareStarted / PrepareCompleted。
	CheckpointState ClaimCheckpointState `json:"checkpointState"`

	// Status 保存 ResourceClaim 的 .Status 快照，包含调度器的分配结果。
	// 分配结果中包含驱动分配的设备名称列表，Unprepare 时（尤其是清理孤儿 Claim 时）
	// 通过此字段找到设备名称进行清理，无需再次向 API Server 查询。
	Status resourceapi.ResourceClaimStatus `json:"status,omitempty"`

	// PreparedDevices 记录已完成准备的设备列表（含 CDI 设备名、MIG 实例信息等）。
	// 仅在 PrepareCompleted 状态下有效；PrepareStarted 状态下此字段为空。
	PreparedDevices PreparedDevices `json:"preparedDevices,omitempty"`

	// Name 和 Namespace 记录 ResourceClaim 的 Kubernetes 名称和命名空间。
	// CheckpointCleanupManager 使用这两个字段通过 Get() API 查询 Claim 是否仍存在，
	// 以判断该 Claim 是否为"僵尸"（已从 API Server 删除但 Checkpoint 未清理）。
	// 注意：v25.3.x 版本的 Checkpoint 不含此字段（空字符串），清理时需跳过。
	Name      string `json:"name,omitempty"`
	Namespace string `json:"namespace,omitempty"`
}

// CheckpointV1 是旧版（已废弃）的 Checkpoint 格式，仅为向后兼容而保留。
// V1 格式的局限：
//   - 不含 CheckpointState 字段，无法区分"准备中"和"准备完成"
//   - 不含 Name/Namespace 字段，无法按名称查询 Claim
//   - Checksum 在顶层 Checkpoint 结构中，而非 V1 内部
// 读取时优先使用 V2，V1 仅作为降级读写支持（确保旧版插件可读）。
type CheckpointV1 struct {
	PreparedClaims PreparedClaimsByUIDV1 `json:"preparedClaims,omitempty"`
}

// PreparedClaimsByUIDV1 是 V1 格式的 Claim UID → PreparedClaimV1 映射。
// 与 V2 的 PreparedClaimsByUIDV2 语义相同，但值类型为 PreparedClaimV1。
type PreparedClaimsByUIDV1 map[string]PreparedClaimV1

// PreparedClaimV1 是旧版 Checkpoint 条目，不含状态字段。
// V1 中的所有条目被视为 PrepareCompleted（因为 V1 写入时只记录完成的 Claim）。
type PreparedClaimV1 struct {
	Status          resourceapi.ResourceClaimStatus `json:"status,omitempty"`
	PreparedDevices PreparedDevices                 `json:"preparedDevices,omitempty"`
}

// ToV2 将 V1 格式的 Checkpoint 升级转换为 V2 格式。
// 转换规则：V1 中没有 CheckpointState 字段，所有条目均被设置为 PrepareCompleted，
// 因为 V1 格式在写入时只包含已完成准备的 Claim。
// 注意：V1 条目不含 Name/Namespace，转换后这两个字段为空（清理时会跳过）。
func (v1 *CheckpointV1) ToV2() *CheckpointV2 {
	// 创建 V2 结构，初始化 PreparedClaims map
	v2 := &CheckpointV2{
		PreparedClaims: make(PreparedClaimsByUIDV2),
	}
	// 遍历 V1 中所有 Claim，逐个转换为 V2 格式
	for claimUID, v1Claim := range v1.PreparedClaims {
		v2.PreparedClaims[claimUID] = PreparedClaimV2{
			// V1 不含状态字段，默认所有条目都是 PrepareCompleted
			CheckpointState: ClaimCheckpointStatePrepareCompleted,
			// 直接复制 Status 和 PreparedDevices
			Status:          v1Claim.Status,
			PreparedDevices: v1Claim.PreparedDevices,
			// Name 和 Namespace 在 V1 中不存在，保持空字符串
		}
	}
	return v2
}

// ToV1 将 V2 格式降级为 V1 格式，供旧版插件读取（滚动升级期间的兼容性保障）。
// 降级规则：只保留处于 PrepareCompleted 状态的 Claim，丢弃 PrepareStarted 条目。
// 理由：旧版插件不理解 PrepareStarted 状态，若读到该状态的条目可能产生错误行为；
// 而 PrepareStarted 条目对应的设备准备尚未完成，旧版插件也无法正确使用这些设备。
func (v2 *CheckpointV2) ToV1() *CheckpointV1 {
	// 创建 V1 结构，初始化 PreparedClaims map
	v1 := &CheckpointV1{
		PreparedClaims: make(PreparedClaimsByUIDV1),
	}
	// 遍历 V2 中所有 Claim，只保留 PrepareCompleted 状态的条目
	for claimUID, v1Claim := range v2.PreparedClaims {
		// 跳过 PrepareStarted 状态的条目
		// 旧版插件无法处理 PrepareStarted 状态，若包含会导致不可预测的行为
		if v1Claim.CheckpointState != ClaimCheckpointStatePrepareCompleted {
			continue
		}
		v1.PreparedClaims[claimUID] = PreparedClaimV1{
			// V1 不含 CheckpointState、Name、Namespace 字段，只复制 Status 和 PreparedDevices
			Status:          v1Claim.Status,
			PreparedDevices: v1Claim.PreparedDevices,
		}
	}
	return v1
}
