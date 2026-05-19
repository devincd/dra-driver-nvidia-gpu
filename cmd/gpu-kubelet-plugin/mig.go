// mig.go 定义了 NVIDIA MIG（Multi-Instance GPU）设备的核心数据结构和命名转换工具。
//
// MIG 技术允许将一块物理 GPU 切分为多个隔离的 GPU 实例，每个实例拥有独立的显存和计算资源。
// 本文件定义了三种描述 MIG 配置的数据结构，分别用于不同场景：
//   - MigSpecTuple：轻量级三元组，用于规范名解析和孤儿设备检测
//   - MigLiveTuple：运行时标识，用于跟踪已存在的 MIG 实例
//   - MigSpec：完整配置描述，用于 ResourceSlice 生成和 MIG 设备创建
//
// 此外还提供了规范设备名称（CanonicalName）的生成和解析工具，
// 以及 RFC 1123 DNS 名称格式转换和驼峰转连字符命名等辅助函数。
package main

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	nvdev "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

// MigSpecTuple 是 MIG 设备配置的三元组抽象描述：父 GPU minor 号 + GPU Instance Profile ID + 放置起始位置。
//
// 与 MigDeviceInfo/MigLiveTuple 的区别：
//   - MigSpecTuple 不含 UUID，不表达"设备是否存在"——仅描述"期望的物理配置"
//   - MigLiveTuple 含 UUID，表达"当前存在的具体 MIG 实例"
//   - MigDeviceInfo 含完整运行时信息，用于 Checkpoint 序列化
//
// 典型用途：
//   1. unpreparePartiallyPreparedClaim() 中根据规范名反解析配置，再通过 FindMigDevBySpec() 查找实例
//   2. DestroyUnknownMIGDevices() 中识别孤儿 MIG 设备
//   3. ToCanonicalName() 生成 ResourceSlice 中的设备名称
//
// Profile ID 说明：这里的 Profile ID 指 GPU Instance Profile ID（GI Profile ID），
// 与 `nvidia-smi mig -lgip` 输出中的 Profile ID 对应，直接传给 nvmlDeviceCreateGpuInstance()。
// Profile ID 定义了切片数量和显存大小，是比字符串（"1g.5gb"）更稳定的程序化标识。
type MigSpecTuple struct {
	// ParentMinor 是父 GPU 的 minor 号（如 nvidia-smi 中的 GPU 0、GPU 1），
	// 用于标识 MIG 设备所属的物理 GPU。
	ParentMinor GPUMinor
	// ProfileID 是 GPU Instance Profile ID，定义了 MIG 实例的计算和显存配额。
	// 例如 A100 上 Profile ID 0 对应 1g.5gb，Profile ID 1 对应 1g.10gb 等。
	ProfileID int
	// PlacementStart 是 MIG 设备在 GPU 内存切片上的放置起始位置索引。
	// GPU 的显存被划分为多个等大小的切片，PlacementStart 表示此 MIG 实例从哪个切片开始占用。
	PlacementStart int
}

// MigLiveTuple 是当前存在的具体 MIG 设备的最小精确标识（三元组：父 GPU + GI + CI）。
//
// 设计原则：在 MIG 设备的生命周期内，可通过 (parentUUID, GIID, CIID) 三元组唯一标识它。
// 但 GIID/CIID 可能在设备销毁重建后被复用（指向不同的物理配置），因此额外跟踪 MigUUID。
// UUID 在每次销毁/重建后变化，可用于区分"旧设备已删除并重建了同名配置"的场景。
//
// 注意：GIID/CIID 与 ProfileID/PlacementStart 之间没有保证的对应关系，
// 不能从 GIID 反推 ProfileID，必须通过 NVML API 查询 GI Info 来获取。
type MigLiveTuple struct {
	// ParentMinor 是父 GPU 的 minor 号，用于日志输出和规范名生成。
	ParentMinor GPUMinor `json:"parentMinor"`
	// GIID 是 GPU Instance 的 NVML 内部标识符，由 NVML 在创建时分配。
	// 同一个 GIID 在 GI 被销毁后可能被新创建的 GI 复用。
	GIID int `json:"giId"`
	// CIID 是 Compute Instance 的 NVML 内部标识符，由 NVML 在创建时分配。
	// 每个 CI 属于一个 GI（GPU Instance），GI 下可以有多个 CI。
	CIID int `json:"ciId"`
	// MigUUID 是 MIG 设备的全局唯一标识符，由 NVML 在设备创建时生成。
	// 每次 GI/CI 销毁重建后 UUID 都会变化，可用于检测设备是否被替换。
	MigUUID string `json:"migUUID"`
	// ParentUUID 与 ParentMinor 提供相同信息（均标识父 GPU），但 UUID 更精确（跨节点唯一）。
	// 在 deleteMigDevice() 中用 UUID 获取 NVML 设备句柄，minor 号用于日志和规范名。
	ParentUUID string `json:"parentUUID"`
}

// MigSpec 是与 MigSpecTuple 语义相同但字段类型更丰富的 MIG 配置描述对象。
//
// 主要用途：
//   1. DynamicMIG 模式中作为 AllocatableDevice.MigDynamic 字段，表示"可创建此配置的抽象设备"
//   2. partitions.go 中用于生成 KEP 4815 格式的 ResourceSlice（含 ConsumesCounters）
//   3. createMigDevice() 中作为输入参数，指导 MIG 实例的创建
//
// 与 MigSpecTuple 的关系：MigSpec 包含 MigSpecTuple 的所有语义信息，
// 但额外保存了 Profile 对象和 GIProfileInfo，避免每次操作时重新查询 NVML。
type MigSpec struct {
	// Parent 是父 GPU 的信息对象，包含 minor 号、UUID、可用 Profile 列表等。
	Parent *GpuInfo
	// Profile 是 MIG Profile 的抽象表示（如 "1g.5gb"），
	// 包含 GI Profile 和 CI Profile 的字符串标识。
	Profile nvdev.MigProfile
	// GIProfileInfo 是 GPU Instance Profile 的 NVML 详细信息，
	// 包含 Profile ID、内存大小、实例数量限制等，用于创建 GI 实例。
	GIProfileInfo nvml.GpuInstanceProfileInfo
	// Placement 是 MIG 设备在 GPU 内存切片上的放置位置信息，
	// 包含起始位置（Start）和大小（Size），定义了该 MIG 实例占用的内存切片范围。
	Placement nvml.GpuInstancePlacement
}

// Tuple 将 MigSpec 转换为轻量级的 MigSpecTuple 三元组。
// 转换时只提取核心配置维度：ParentMinor、ProfileID、PlacementStart，
// 丢弃 Profile 对象和 GIProfileInfo，用于生成规范名和配置比较。
func (m *MigSpec) Tuple() *MigSpecTuple {
	return &MigSpecTuple{
		// 从父 GPU 信息中提取 minor 号
		ParentMinor: m.Parent.minor,
		// 从 GIProfileInfo 中提取 Profile ID（NVML 分配的整数标识符）
		ProfileID: int(m.GIProfileInfo.Id),
		// 从 Placement 中提取放置起始位置索引
		PlacementStart: int(m.Placement.Start),
	}
}

// ToCanonicalName 将 MigSpecTuple 转换为 DRA ResourceSlice 中使用的规范设备名称。
// 格式：gpu-<parentMinor>-mig-<profile>-<profileID>-<placementStart>
// 例如："gpu-0-mig-1g10gb-0-0"
//
// 为什么需要额外传入 profileName 参数？
// MigSpecTuple 故意不存储 Profile 名称（它不是核心配置维度，仅由 Profile ID 隐含推导）。
// 但 CanonicalName 中需要包含可读的 Profile 字符串，因此需要调用方提供。
//
// Profile 名称在名称排序中有实际意义：
// k8s 调度器从 ResourceSlice 中从上到下遍历设备，优先匹配靠前的设备。
// 排序顺序影响调度决策，因此 Profile 名称需要保持稳定一致。
func (m *MigSpecTuple) ToCanonicalName(profileName string) DeviceName {
	// 去除 Profile 名称中的点号（如 "1g.10gb" → "1g10gb"），
	// 因为 DRA 设备名不允许点号
	pname := toRFC1123Compliant(strings.ReplaceAll(profileName, ".", ""))
	// 按规范格式拼接设备名：gpu-<minor>-mig-<profile>-<profileID>-<placementStart>
	return fmt.Sprintf("gpu-%d-mig-%s-%d-%d", m.ParentMinor, pname, m.ProfileID, m.PlacementStart)
}

// NewMigSpecTupleFromCanonicalName 从规范设备名解析出 MigSpecTuple 三元组。
// 用于 unpreparePartiallyPreparedClaim() 中从 Checkpoint 的设备名反解析配置，
// 再通过 FindMigDevBySpec() 在 NVML 中查找对应的物化 MIG 设备。
//
// 正则表达式匹配格式：^gpu-(\d+)-mig-(.+)-(\d+)-(\d+)$
// 四个捕获组分别对应：ParentMinor / Profile名称（被忽略）/ ProfileID / PlacementStart
// Profile 名称被忽略是因为它可以从 Profile ID 通过 NVML API 反查得到
func NewMigSpecTupleFromCanonicalName(n DeviceName) (*MigSpecTuple, error) {
	// 使用预编译的正则表达式匹配规范 MIG 设备名
	matches := canonicalMigNameRegex.FindStringSubmatch(string(n))
	if matches == nil {
		// 设备名格式不匹配，无法解析
		return nil, fmt.Errorf("failed to match MIG device name regex: '%s'", n)
	}

	// 解析三个数字字段：
	// matches[1] = ParentMinor, matches[3] = ProfileID, matches[4] = PlacementStart
	// matches[2] = Profile 名称（忽略，不需要从字符串解析）
	parentMinor, err1 := strconv.Atoi(matches[1])
	profileID, err2 := strconv.Atoi(matches[3])
	placementStart, err3 := strconv.Atoi(matches[4])

	// 检查三个数字字段的解析是否全部成功
	if err1 != nil || err2 != nil || err3 != nil {
		return nil, fmt.Errorf("integer parsing failed (dev name: %s)", n)
	}

	return &MigSpecTuple{
		ParentMinor:    GPUMinor(parentMinor),
		ProfileID:      profileID,
		PlacementStart: placementStart,
	}, nil
}

// CanonicalName 返回 MigSpec 的规范设备名称，内部调用 Tuple().ToCanonicalName()。
// 这是 MigSpec 生成规范名的便捷方法，自动从 Profile 字段提取名称字符串。
func (m *MigSpec) CanonicalName() DeviceName {
	// 先转换为 MigSpecTuple，再用 Profile 的字符串表示作为 profileName
	return m.Tuple().ToCanonicalName(m.Profile.String())
}

// MigProfileInfo 存储单个 MIG Profile 的信息及其在 GPU 上的所有可能放置位置。
// 在 getGpuInfo() 中扫描 GPU 的所有 Profile 时填充，存储在 GpuInfo.migProfiles 中。
type MigProfileInfo struct {
	// profile 是 MIG Profile 的抽象表示，包含 GI Profile 和 CI Profile 的信息
	profile nvdev.MigProfile
	// placements 是该 Profile 在 GPU 上的所有合法放置位置列表。
	// 同一 Profile 可以放置在 GPU 内存切片的不同起始位置，
	// 例如 1g.5gb 在 10gb GPU 上可能有多个合法的放置位置。
	placements []*MigDevicePlacement
}

// String 返回 MIG Profile 的字符串表示（如 "1g.10gb"），用于日志和规范名生成。
// 实现了 fmt.Stringer 接口。
func (p MigProfileInfo) String() string {
	return p.profile.String()
}

// MigDevicePlacement 包装 NVML 的 GpuInstancePlacement，表示 MIG 设备在 GPU 内存切片上的放置位置。
// 放置位置定义了该 MIG 设备占用哪些内存切片（Start 到 Start+Size-1）。
type MigDevicePlacement struct {
	// 内嵌 NVML 的 GpuInstancePlacement 结构体，
	// 包含 Start（起始切片索引）和 Size（占用的切片数量）两个字段
	nvml.GpuInstancePlacement
}

// canonicalMigNameRegex 是解析规范 MIG 设备名称的正则表达式。
// 格式：^gpu-(\d+)-mig-(.+)-(\d+)-(\d+)$
// 捕获组：
//  1. ParentMinor（数字）—— 父 GPU 的 minor 号
//  2. Profile 名称（贪婪匹配中间所有字符）—— 如 "1g10gb"
//  3. ProfileID（数字）—— GPU Instance Profile ID
//  4. PlacementStart（数字）—— 放置起始位置索引
//
// 使用 ^ 和 $ 锚点确保全串匹配（不允许部分匹配）。
// Profile 名称使用贪婪匹配 (.+) 是因为名称中可能包含连字符，
// 而最后的两个数字字段由 -(\d+)-(\d+)$ 明确锚定，不会产生歧义。
var canonicalMigNameRegex = regexp.MustCompile(`^gpu-(\d+)-mig-(.+)-(\d+)-(\d+)$`)

// toRFC1123Compliant 将字符串转换为符合 RFC 1123 的 DNS 名称格式。
//
// 转换规则：
//  1. 转换为小写（DNS 名称不区分大小写，统一小写避免歧义）
//  2. 将非字母数字连字符点号的字符替换为连字符
//  3. 去除首尾连字符（DNS 标签不能以连字符开头或结尾）
//  4. 去除末尾点号（避免 DNS 解析歧义）
//  5. 截断超过 253 字符的字符串（DNS 名称最大长度限制）
//
// 注意：返回值不一定直接可用作 DRA 设备名（DRA 不允许点号），
// 调用方需额外处理点号（如在 MIG 名称中先用 ReplaceAll 去除点号，再调用此函数）。
func toRFC1123Compliant(name string) string {
	// 步骤 1：转换为小写
	name = strings.ToLower(name)
	// 步骤 2：将非 [a-z0-9-.] 的字符替换为连字符
	re := regexp.MustCompile(`[^a-z0-9-.]`)
	name = re.ReplaceAllString(name, "-")
	// 步骤 3：去除首尾连字符
	name = strings.Trim(name, "-")
	// 步骤 4：去除末尾点号
	name = strings.TrimSuffix(name, ".")

	// 步骤 5：截断超过 253 字符的字符串
	if len(name) > 253 {
		name = name[:253]
	}

	return name
}

// camelToDNSName 将驼峰命名字符串转换为 DNS 友好的连字符分隔格式。
// 例如：
//   - "multiProcessors" → "multi-processors"
//   - "HTTPServer" → "http-server"
//   - "copyEngines" → "copy-engines"
//
// 用于将 PartCapacityMap 中的容量键名（如 "multiprocessors"）转换为计数器名称，
// 确保 SharedCounterSet 和 ConsumesCounters 中的名称格式一致。
//
// 转换流程：先插入连字符分隔驼峰单词，再调用 toRFC1123Compliant 做最终规范化。
func camelToDNSName(s string) string {
	// 第一步：在小写/数字后跟大写字母处插入连字符
	// 正则 ([a-z0-9])([A-Z]) 匹配如 "aB" → "a-B"、"1C" → "1-C"
	re1 := regexp.MustCompile("([a-z0-9])([A-Z])")
	result := re1.ReplaceAllString(s, "$1-$2")

	// 第二步：在连续大写字母序列中的最后一个大写字母前插入连字符
	// 正则 ([A-Z]+)([A-Z][a-z]) 匹配如 "HTTPServer" → "HTTP-Server"
	// 这里 [A-Z]+ 匹配连续大写字母（"HTTP"），[A-Z][a-z] 匹配最后一个大写后跟小写（"Server"）
	re2 := regexp.MustCompile("([A-Z]+)([A-Z][a-z])")
	result = re2.ReplaceAllString(result, "$1-$2")

	// 第三步：调用 toRFC1123Compliant 做最终规范化
	// 包括转小写、替换非法字符、去除首尾连字符等
	return toRFC1123Compliant(result)
}
