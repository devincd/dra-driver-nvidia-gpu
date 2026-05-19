/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

// cdi.go 封装了 Container Device Interface (CDI) 规范的生成与管理逻辑。
//
// CDI 是容器运行时（如 containerd、CRI-O）用于声明式描述设备注入的标准接口。
// 本文件的核心职责是为每个 ResourceClaim 生成一份"瞬态 CDI spec 文件"（transient spec），
// 该文件包含了容器启动时需要注入的 GPU 设备节点、驱动库挂载、环境变量等信息。
//
// 核心流程：
//  1. Prepare 阶段：调用 CreateClaimSpecFile() 为 Claim 生成 CDI spec 文件
//     - 完整 GPU：使用缓存的父 GPU CDI 规范
//     - VFIO 设备：使用 VFIO 专属 CDI 编辑项
//     - MIG 设备：使用父 GPU 规范 + 追加 MIG caps 设备节点
//  2. Unprepare 阶段：调用 DeleteClaimSpecFile() 删除对应的 CDI spec 文件
//
// 性能优化：
//   - specCache：以 UUID 为 key 的带 TTL 缓存（5分钟），避免每次 Prepare 都重新扫描驱动文件系统
//   - WarmupDevSpecCache()：插件启动时预热缓存，将首次耗时的 nvcdi 调用提前到启动阶段
//
// CDI spec 文件的路径命名规则：
//   /var/run/cdi/k8s.gpu.nvidia.com-claim-<claimUID>.yaml
// 容器运行时监视该目录，自动发现并解析新增的 CDI spec 文件。
package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/sirupsen/logrus"

	nvdevice "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi"
	"github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi/spec"
	transformroot "github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi/transform/root"
	"k8s.io/klog/v2"

	utilcache "k8s.io/apimachinery/pkg/util/cache"
	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"
	cdiparser "tags.cncf.io/container-device-interface/pkg/parser"
	cdispec "tags.cncf.io/container-device-interface/specs-go"

	"github.com/NVIDIA/k8s-dra-driver-gpu/internal/common"
)

const (
	// cdiVendor 是 CDI spec 文件中的 vendor 字段，格式为 "k8s.gpu.nvidia.com"。
	// vendor 是 CDI 规范中设备命名空间的顶层标识符，用于区分不同驱动提供的设备。
	// 格式遵循 CDI 规范的反域名约定。
	cdiVendor = "k8s." + DriverName

	// cdiClaimClass 是 CDI spec 的 class 字段，标识这是每个 ResourceClaim 独有的临时 spec。
	// CDI 规范中 vendor+class 共同构成设备命名空间，如 "k8s.gpu.nvidia.com/claim"。
	// "claim" 这个 class 名称表明该 spec 是 Claim 级别的，与持久化的 CDI spec（如 "gpu" class）区分。
	cdiClaimClass = "claim"

	// defaultCDIRoot 是 CDI spec 文件的默认输出目录，未通过 --cdi-root 指定时使用。
	// 该目录由容器运行时（如 containerd/CRI-O）监视，用于发现可用的 CDI 规范。
	// 容器运行时通过 inotify 或定期轮询检测该目录下新增的 YAML 文件。
	defaultCDIRoot = "/var/run/cdi"

	// procNvCapsPath 是 NVIDIA capabilities 伪文件系统的路径，
	// 用于获取 MIG 设备（GI/CI）的 minor 号，进而构建 CDI 设备节点规范。
	// 该路径下的文件结构为：/proc/driver/nvidia/capabilities/gpu<N>/mig/gi<M>/ci<K>/access
	// 其中 N=父 GPU minor 号, M=GI ID, K=CI ID
	// 读取 access 文件可获取该 capability 对应的字符设备的 major/minor 号。
	procNvCapsPath = "/proc/driver/nvidia/capabilities"
)

// CDIHandler 封装了所有与 CDI（Container Device Interface）spec 相关的操作。
//
// 核心职责：
//  1. 为每个 ResourceClaim 生成一份临时 CDI spec 文件（CreateClaimSpecFile）
//     该文件包含 GPU/MIG 设备节点、驱动库挂载、环境变量等容器注入信息
//  2. Unprepare 时删除对应 CDI spec 文件（DeleteClaimSpecFile）
//  3. 利用 specCache 缓存昂贵的 nvcdi 调用结果（TTL=5分钟），
//     避免每次 Prepare 时都重新扫描驱动文件系统
//
// CDI spec 文件由容器运行时（如 containerd）读取，
// 用于在容器启动时自动将所需的 GPU 设备和驱动库注入到容器内。
// 每个 Claim 对应一个"瞬态 CDI spec"（transient spec），Claim 释放后删除。
type CDIHandler struct {
	// logger 是 logrus 日志实例，传给 nvcdi 库用于内部日志输出。
	// 若未设置，默认创建一个输出到 io.Discard 的静默 logger。
	logger *logrus.Logger

	// nvml 是 NVML 库的接口实例，用于与 NVIDIA 驱动交互查询 GPU 状态。
	nvml nvml.Interface

	// nvdevice 是 go-nvlib 的设备抽象层接口，提供更高层的 GPU 设备操作。
	nvdevice nvdevice.Interface

	// nvcdiClaim 是专门用于生成 Claim 级别临时 CDI spec 的 nvcdi 实例。
	// 它使用 "nvml" 模式直接通过 NVML 查询设备信息，
	// 并禁用了 nvsandboxutils 以避免不必要的依赖初始化。
	nvcdiClaim nvcdi.Interface

	// driverRoot 是容器内驱动文件系统的根路径（如 "/driver-root"）。
	// 在特权容器部署中，宿主机的驱动目录被挂载到此路径下。
	driverRoot string

	// devRoot 是容器内设备节点文件系统的根路径（如 "/dev"）。
	// 用于定位 /dev/nvidia* 等字符设备节点。
	devRoot string

	// targetDriverRoot 是宿主机上驱动文件系统的根路径（如 "/"）。
	// 在 CDI spec 中需要将容器内路径变换为宿主机路径，此字段用于路径变换的目标前缀。
	targetDriverRoot string

	// nvidiaCDIHookPath 是 nvidia-cdi-hook 工具在宿主机上的绝对路径。
	// nvidia-cdi-hook 用于在容器启动时执行一些设备注入相关的钩子操作（如创建设备节点）。
	nvidiaCDIHookPath string

	// specCache 是以 UUID 为 key 的设备 spec 缓存，减少昂贵的 nvcdi 调用次数。
	// nvcdi.GetDeviceSpecsByID() 首次调用耗时可达数秒（需扫描驱动文件系统），
	// 后续通过缓存可在毫秒级返回。TTL=5分钟：长到避免频繁失效，短到不会永久缓存过期数据。
	// 缓存的过期时间在每次 Set 时单独设置（非全局配置），允许不同 key 使用不同 TTL。
	specCache *utilcache.Expiring

	// cdiRoot 是 CDI spec 文件的输出目录，运行时可通过 --cdi-root 标志覆盖 defaultCDIRoot。
	cdiRoot string
}

// NewCDIHandler 使用 functional options 模式构造 CDIHandler 实例。
// 若关键字段未通过 option 设置，则填入安全默认值：
//   - logger：输出到 io.Discard 的静默 logger
//   - nvml：nvml.New() 默认实例
//   - cdiRoot："/var/run/cdi" 默认目录
//   - nvdevice：基于 nvml 实例创建
//   - nvcdiClaim：使用 "nvml" 模式创建，禁用 nvsandboxutils
//
// 参数：
//   - opts：一系列 cdiOption 函数，用于自定义 CDIHandler 的字段
//
// 返回值：
//   - *CDIHandler：构造完成的 CDI 处理器实例
//   - error：若 nvcdi 库创建失败则返回错误
func NewCDIHandler(opts ...cdiOption) (*CDIHandler, error) {
	// 创建空的 CDIHandler，通过 option 函数逐步填充字段
	h := &CDIHandler{}
	for _, opt := range opts {
		opt(h)
	}

	// 以下为各字段的默认值填充逻辑
	if h.logger == nil {
		// 默认静默 logger，避免 nvcdi 库内部日志干扰插件日志
		h.logger = logrus.New()
		h.logger.SetOutput(io.Discard)
	}
	if h.nvml == nil {
		// 使用默认的 NVML 库实例
		h.nvml = nvml.New()
	}
	if h.cdiRoot == "" {
		// 使用默认的 CDI 规范文件输出目录
		h.cdiRoot = defaultCDIRoot
	}
	if h.nvdevice == nil {
		// 基于 NVML 实例创建 go-nvlib 设备抽象层
		h.nvdevice = nvdevice.New(h.nvml)
	}

	// 若未通过 option 提供 nvcdiClaim，则创建默认实例
	if h.nvcdiClaim == nil {
		nvcdilib, err := nvcdi.New(
			nvcdi.WithDeviceLib(h.nvdevice),           // 使用 go-nvlib 设备库
			nvcdi.WithDriverRoot(h.driverRoot),        // 容器内驱动根路径
			nvcdi.WithDevRoot(h.devRoot),              // 容器内设备根路径
			nvcdi.WithLogger(h.logger),                // 日志实例
			nvcdi.WithNvmlLib(h.nvml),                 // NVML 库实例
			nvcdi.WithMode("nvml"),              // 使用 NVML 模式直接查询 GPU
			nvcdi.WithVendor(cdiVendor),               // CDI vendor 标识
			nvcdi.WithClass(cdiClaimClass),            // CDI class 标识
			nvcdi.WithNVIDIACDIHookPath(h.nvidiaCDIHookPath), // nvidia-cdi-hook 路径
			nvcdi.WithFeatureFlags(nvcdi.FeatureDisableNvsandboxUtils), // 禁用 nvsandboxutils
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create CDI library for claims: %w", err)
		}
		h.nvcdiClaim = nvcdilib
	}

	// 创建带 TTL 的缓存实例，过期时间在每次 Set 时单独设置
	h.specCache = utilcache.NewExpiring()

	return h, nil
}

// writeSpec 将 CDI spec 写入文件系统，并在写入前应用 Root 路径变换。
//
// 路径变换背景：CDI spec 中包含设备节点路径和驱动库路径，
// 这些路径在容器内（driverRoot）和宿主机（targetDriverRoot）可能不同。
// 例如，容器内看到驱动库在 /driver-root/usr/lib/x86_64-linux-gnu/ 下，
// 但宿主机上实际路径是 /usr/lib/x86_64-linux-gnu/。
// transformroot 将 spec 中的容器内路径变换为宿主机路径，
// 确保容器运行时挂载的是宿主机上实际存在的路径，而非容器内的挂载点路径。
//
// 参数：
//   - spec：已构造好的 CDI spec 接口对象
//   - specName：spec 文件名（不含扩展名），如 "k8s.gpu.nvidia.com-claim-<claimUID>"
//
// 返回值：
//   - error：路径变换失败或文件写入失败时返回错误
func (cdi *CDIHandler) writeSpec(spec spec.Interface, specName string) error {
	// 应用 Root 路径变换：将 spec 中的容器内路径替换为宿主机路径
	err := transformroot.New(
		transformroot.WithRoot(cdi.driverRoot),         // 容器内驱动根路径（被替换的源前缀）
		transformroot.WithTargetRoot(cdi.targetDriverRoot), // 宿主机驱动根路径（替换的目标前缀）
		transformroot.WithRelativeTo("host"),            // 相对于宿主机的路径前缀
	).Transform(spec.Raw())
	if err != nil {
		return fmt.Errorf("failed to transform driver root in CDI spec: %w", err)
	}

	klog.V(7).Infof("Write CDI spec: %s", specName)
	// 将变换后的 spec 写入 YAML 文件，路径格式：<cdiRoot>/<specName>.yaml
	return spec.Save(filepath.Join(cdi.cdiRoot, specName+".yaml"))
}

// GetCommonEditsCached 返回所有设备共用的 CDI 容器编辑项（如驱动库挂载、公共设备节点）。
//
// "common edits" 是不特定于某个具体设备的编辑项，通常包括：
//   - NVIDIA 驱动共享库的绑定挂载（/usr/lib/x86_64-linux-gnu/libnvidia*.so 等）
//   - 伪设备节点（如 /dev/nvidiactl、/dev/nvidia-uvm）
//
// 使用缓存的原因：nvcdi.GetCommonEdits() 会扫描驱动文件系统，耗时较长（首次可达数秒）。
// 缓存 TTL 为 5 分钟，在驱动文件不变化的正常运行期间，5 分钟内缓存数据是有效的。
//
// 返回副本（浅拷贝）以防止调用方意外修改缓存条目。
// 浅拷贝足够安全：调用方通常只向列表追加，不修改内部结构。
//
// 返回值：
//   - *cdiapi.ContainerEdits：公共容器编辑项的副本
//   - error：缓存类型不匹配或 nvcdi 调用失败时返回错误
func (cdi *CDIHandler) GetCommonEditsCached() (*cdiapi.ContainerEdits, error) {
	// 使用固定 key "commonEdits" 查询缓存
	key := "commonEdits"
	if v, ok := cdi.specCache.Get(key); ok {
		// 缓存命中，进行类型断言和浅拷贝后返回
		edits, ok := v.(*cdiapi.ContainerEdits)
		if !ok {
			return nil, fmt.Errorf("expected *cdiapi.ContainerEdits, got %T", v)
		}
		clone := *edits
		return &clone, nil
	}

	// 缓存未命中，调用 nvcdi 获取公共编辑项并记录耗时
	t0 := time.Now()
	v, err := cdi.nvcdiClaim.GetCommonEdits()
	klog.V(7).Infof("t_cdi_get_common_edits %.3f s", time.Since(t0).Seconds())

	if err != nil {
		return nil, err
	}

	// 写入缓存，TTL 为 5 分钟
	cdi.specCache.Set(key, v, time.Duration(5*time.Minute))

	// 返回浅拷贝，保护缓存条目不被外部修改
	clone := *v
	return &clone, nil
}

// WarmupDevSpecCache 在插件启动时预热设备 spec 缓存。
//
// nvcdi.GetDeviceSpecsByID() 首次调用耗时可达数秒（需要扫描驱动文件系统、
// 查询 NVML 获取设备信息、构建 CDI 规范等），
// 预热后 Prepare() 期间的 CDI spec 生成将大幅提速（从秒级降至毫秒级）。
//
// 预热只针对完整物理 GPU（传入 fullGPUUUIDs），
// 因为 MIG 设备的 CDI spec 基于父 GPU 的 spec 构建（见 CreateClaimSpecFile），
// 缓存父 GPU 的 spec 即可覆盖 MIG 场景。
//
// 预热失败只打 Warning 日志，不影响启动流程——
// 缓存未命中时会在 Prepare 时按需填充，仅增加首次延迟。
//
// 参数：
//   - uuids：需要预热缓存的完整物理 GPU UUID 列表
func (cdi *CDIHandler) WarmupDevSpecCache(uuids []string) {
	for _, uuid := range uuids {
		_, err := cdi.GetDeviceSpecsByUUIDCached(uuid)
		if err != nil {
			// 预热失败不阻塞启动，运行时会按需填充缓存
			klog.Warningf("Ignore error during cache warmup: GetDeviceSpecsByUUIDCached() failed: %s", err)
		}
	}
}

// GetDeviceSpecsByUUIDCached 返回指定 UUID 设备的 CDI 规范列表（优先从缓存读取）。
//
// 设备规范包含该设备需要在容器内呈现的：
//   - 设备节点（如 /dev/nvidia0）：字符设备的挂载和 cgroup 权限
//   - 驱动库挂载（如 libnvidia-ml.so）：GPU 运行时所需的共享库
//   - 环境变量和钩子（如 nvidia-cdi-hook）：容器启动时执行的额外配置
//
// 返回副本（深拷贝）防止调用方修改缓存条目。
// 与 GetCommonEditsCached 不同，这里使用 copy() 进行深拷贝，
// 因为 []cdispec.Device 是切片，浅拷贝后修改元素仍会影响缓存。
//
// 参数：
//   - uuid：设备的 UUID 字符串
//
// 返回值：
//   - []cdispec.Device：设备 CDI 规范列表的副本
//   - error：缓存类型不匹配或 nvcdi 调用失败时返回错误
func (cdi *CDIHandler) GetDeviceSpecsByUUIDCached(uuid string) ([]cdispec.Device, error) {
	// 使用设备 UUID 作为缓存 key
	key := uuid
	if v, ok := cdi.specCache.Get(key); ok {
		// 缓存命中，进行类型断言和深拷贝后返回
		devs, ok := v.([]cdispec.Device)
		if !ok {
			return nil, fmt.Errorf("expected []cdispec.Device, got %T", v)
		}
		clone := make([]cdispec.Device, len(devs))
		copy(clone, devs)
		return clone, nil
	}

	// 缓存未命中，调用 nvcdi 获取设备规范并记录耗时
	t0 := time.Now()
	devs, err := cdi.nvcdiClaim.GetDeviceSpecsByID(uuid)
	klog.V(1).Infof("GetDeviceSpecsByID() called for %s, t_cdi_get_device_specs_by_id %.3f s", uuid, time.Since(t0).Seconds())
	if err != nil {
		return nil, err
	}

	// 写入缓存，TTL 为 5 分钟
	cdi.specCache.Set(key, devs, time.Duration(5*time.Minute))

	// 返回深拷贝，保护缓存条目不被外部修改
	clone := make([]cdispec.Device, len(devs))
	copy(clone, devs)
	return clone, nil
}

// CreateClaimSpecFile 为一个已完成 Prepare 的 ResourceClaim 生成 CDI spec 文件。
//
// CDI spec 文件是 Claim 级别的"瞬态规范"（transient spec），
// 其生命周期与 Claim 的 Prepare/Unprepare 绑定：Prepare 时创建，Unprepare 时删除。
//
// 规范生成流程：
//  1. 获取公共编辑项（驱动库挂载、元设备等）——所有设备共享
//  2. 对每个已准备设备分类处理：
//     - 完整 GPU：直接使用父 GPU UUID 的缓存 spec
//     - VFIO 直通：使用 VFIO 专属 CDI 编辑项（覆盖 commonEdits）
//     - MIG 设备：使用父 GPU 的 spec（包含 /dev/nvidia<N> 设备节点），
//       额外追加 MIG 专属的 /dev/nvidia-caps/nvidia-cap<GIm> 和 nvidia-cap<CIm> 节点
//  3. 每个设备的 CDI 名称格式为 "<claimUID>-<deviceCanonicalName>"（含 Claim 范围）
//  4. 若该设备组有 ConfigState 容器编辑项（如 MPS shm 挂载），追加到每个设备的 spec 中
//  5. 将最终的 spec 写入文件系统（YAML 格式）
//
// 关于 MIG 设备规范的特殊处理：
//   nvcdi.GetDeviceSpecsByID(MIG_UUID) 可能生成不完整的 spec（已知 issue #787）。
//   因此改用父 GPU 的 spec + 手动追加 MIG caps 设备节点的方式来生成完整的 MIG 规范。
//
// 关于 GPU minor 号与规范名稳定性：
//   NVML 文档指出，SetMigMode() 可能导致设备的 minor 号变化。
//   若 minor 号不稳定，长期关联 CDI spec 与规范名可能引发问题。
//   目前的缓解措施是在每次 Prepare 时动态生成规范（而非依赖长期缓存）。
//
// 参数：
//   - claimUID：ResourceClaim 的唯一标识符，用于生成 CDI spec 文件名和设备名
//   - preparedDevices：已完成 Prepare 的设备列表，包含一个或多个设备组
//
// 返回值：
//   - error：任何步骤失败时返回错误
func (cdi *CDIHandler) CreateClaimSpecFile(claimUID string, preparedDevices PreparedDevices) error {
	// 第 1 步：获取所有设备共享的公共编辑项（驱动库、元设备等）
	commonEdits, err := cdi.GetCommonEditsCached()
	if err != nil {
		return fmt.Errorf("failed to get common CDI spec edits: %w", err)
	}

	// 收集所有设备的 CDI 规范
	var deviceSpecs []cdispec.Device

	// 遍历每个设备组和组内每个设备
	for _, group := range preparedDevices {
		for _, dev := range group.Devices {
			uuid := ""

			// 构造 Claim 级别的 CDI 设备名称，格式：<claimUID>-<deviceCanonicalName>
			// 与 GetClaimDeviceName() 的命名约定保持一致，
			// 确保 kubelet 获取的 CDIDeviceIDs 与 CDI spec 中的设备名匹配
			dname := fmt.Sprintf("%s-%s", claimUID, dev.CanonicalName())

			var dspec cdispec.Device

			// 根据设备类型分别处理
			if dev.Type() == GpuDeviceType {
				// 完整 GPU 设备：从缓存获取该 GPU 的 CDI 规范
				uuid = dev.Gpu.Info.UUID
				// 获取副本（可安全修改，不影响缓存）
				dspecsgpu, err := cdi.GetDeviceSpecsByUUIDCached(uuid)
				if err != nil {
					return fmt.Errorf("unable to get device spec for %s: %w", dname, err)
				}
				// 取第一个 spec（一个 UUID 通常只对应一个 CDI 设备）
				dspec = dspecsgpu[0]
			}

			if dev.Type() == VfioDeviceType {
				// VFIO 设备：使用完全不同的 commonEdits（仅包含 VFIO 相关内容）。
				// 若 preparedDevices 包含 GPU 和 VFIO 混合设备，
				// 当前逻辑会多次覆盖 commonEdits（最后一次生效），属于已知局限。
				commonEdits = GetVfioCommonCDIContainerEdits()
				// VFIO 设备的 CDI 规范直接使用 VFIO 专属编辑项
				dspec = cdispec.Device{
					ContainerEdits: *GetVfioCDIContainerEdits(dev.Vfio.Info).ContainerEdits,
				}
			}

			if dev.Type() == PreparedMigDeviceType {
				// MIG 设备：先获取父 GPU 的规范（含 /dev/nvidia<N> 节点），
				// 再追加 MIG 专属的 caps 字符设备节点（GI 和 CI）。
				// 原因：nvcdi.GetDeviceSpecsByID(MIG_UUID) 生成的 spec 不完整（issue #787）。
				uuid = dev.Mig.Concrete.ParentUUID
				dspecsmig, err := cdi.GetDeviceSpecsByUUIDCached(uuid)
				if err != nil {
					return fmt.Errorf("unable to get device spec for %s: %w", dname, err)
				}
				// 使用父 GPU 的 spec 作为基础
				dspec = dspecsmig[0]

				// 构造 MIG 专属的 caps 设备节点规范
				devnodesForMig, err := cdi.GetDevNodesForMigDevice(dev.Mig.Concrete)
				if err != nil {
					return fmt.Errorf("failed to construct MIG device DeviceNode edits: %w", err)
				}
				klog.V(7).Infof("CDI spec: appending MIG device nodes")
				// 将 MIG caps 设备节点追加到父 GPU 的 spec 中
				dspec.ContainerEdits.DeviceNodes = append(dspec.ContainerEdits.DeviceNodes, devnodesForMig...)
			}

			// 将新生成的规范与 Claim 级别的设备名关联
			dspec.Name = dname

			// 若设备组有额外的容器编辑项（如 MPS shm 挂载和环境变量），追加到此设备的 spec 中。
			// ConfigState.containerEdites 非空表示该设备组有共享配置（如 MPS 或 TimeSlicing）。
			if group.ConfigState.containerEdits != nil {
				deviceEdits := &cdiapi.ContainerEdits{
					ContainerEdits: &dspec.ContainerEdits,
				}
				// Append 合并两个 ContainerEdits（列表追加，不覆盖已有项）
				deviceEdits = deviceEdits.Append(group.ConfigState.containerEdits)
				dspec.ContainerEdits = *deviceEdits.ContainerEdits
			}
			klog.V(7).Infof("Number of device nodes about to inject for device %s: %d", dname, len(dspec.ContainerEdits.DeviceNodes))
			deviceSpecs = append(deviceSpecs, dspec)
		}
	}

	// 使用收集到的所有设备规范和公共编辑项构造完整的 CDI spec 对象
	tws0 := time.Now()
	spec, err := spec.New(
		spec.WithVendor(cdiVendor),         // vendor: "k8s.gpu.nvidia.com"
		spec.WithClass(cdiClaimClass),       // class: "claim"
		spec.WithDeviceSpecs(deviceSpecs),   // 所有设备的 CDI 规范列表
		spec.WithEdits(*commonEdits.ContainerEdits), // 公共编辑项（驱动库挂载等）
	)
	if err != nil {
		return fmt.Errorf("failed to create CDI spec: %w", err)
	}

	// 生成"瞬态规范"文件名（如 "k8s.gpu.nvidia.com-claim-<claimUID>.yaml"）。
	// 瞬态规范（transient spec）是 CDI 规范中专为容器生命周期设计的概念：
	// 与容器生命周期绑定，容器销毁后应删除。
	// GenerateTransientSpecName 生成格式为 "<vendor>-<class>-<suffix>" 的名称。
	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiClaimClass, claimUID)
	klog.V(6).Infof("Writing CDI spec '%s' for claim '%s'", specName, claimUID)

	// 将 spec 写入文件系统
	result := cdi.writeSpec(spec, specName)

	klog.V(7).Infof("t_gen_write_cdi_spec %.3f s", time.Since(tws0).Seconds())
	return result
}

// DeleteClaimSpecFile 删除指定 Claim 对应的 CDI spec 文件。
// 在 Unprepare 流程中调用，防止 CDI spec 文件在 Claim 释放后继续留存。
// 若文件已不存在（如程序崩溃后重启的幂等调用），忽略 "not exist" 错误正常返回。
// 这是幂等性设计的一部分：多次调用 Unprepare 不应报错。
//
// 参数：
//   - claimUID：ResourceClaim 的唯一标识符
//
// 返回值：
//   - error：删除失败（非 "not exist" 错误）时返回错误
func (cdi *CDIHandler) DeleteClaimSpecFile(claimUID string) error {
	// 构造与 CreateClaimSpecFile 一致的瞬态规范文件名
	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiClaimClass, claimUID)
	klog.V(6).Infof("Delete CDI spec file: '%s', claim '%s'", specName, claimUID)
	// 删除 YAML 文件，路径格式：<cdiRoot>/<specName>.yaml
	err := os.Remove(filepath.Join(cdi.cdiRoot, specName+".yaml"))
	if err != nil && !os.IsNotExist(err) {
		// 文件不存在视为正常（幂等性），其他错误向上传递
		return err
	}
	return nil
}

// GetClaimDeviceName 返回设备在 Claim 级别 CDI spec 中的全限定名称（Qualified Name）。
//
// CDI 的设计哲学：所有待注入容器的设备都定义在同一个瞬态 CDI spec 文件中，
// 通过全限定名（vendor/class=name 格式）唯一标识，供容器运行时解析。
//
// 示例返回值：
//
//	"k8s.gpu.nvidia.com/claim=dab5ab50-d59a-42a6-af16-cfd4628c0f7a-gpu-0"
//	                      ↑class   ↑claimUID                     ↑deviceCanonicalName
//
// 该全限定名在 PrepareDevices 响应中作为 CDIDeviceIDs 字段返回给 kubelet，
// kubelet 再将其传递给容器运行时（如 containerd），触发 CDI 设备注入流程。
// 容器运行时解析该名称，在对应的瞬态 CDI spec 文件中查找设备定义，
// 然后执行 spec 中描述的设备节点注入、驱动库挂载、环境变量设置等操作。
//
// 参数：
//   - claimUID：ResourceClaim 的唯一标识符
//   - device：可分配设备的指针
//   - containerEdits：容器的 CDI 编辑项（当前未使用，保留用于未来扩展）
//
// 返回值：
//   - string：CDI 全限定名称，格式为 "k8s.gpu.nvidia.com/claim=<claimUID>-<deviceCanonicalName>"
func (cdi *CDIHandler) GetClaimDeviceName(claimUID string, device *AllocatableDevice, containerEdits *cdiapi.ContainerEdits) string {
	return cdiparser.QualifiedName(cdiVendor, cdiClaimClass, fmt.Sprintf("%s-%s", claimUID, device.CanonicalName()))
}

// GetDevNodesForMigDevice 构造并返回 MIG 设备所需的两个 CDI 字符设备节点规范：
//   - /dev/nvidia-caps/nvidia-cap<GIm>：GPU Instance（GI）的 minor 号对应的设备节点
//   - /dev/nvidia-caps/nvidia-cap<CIm>：Compute Instance（CI）的 minor 号对应的设备节点
//
// 背景：容器化工作负载访问 MIG 设备需要三类字符设备节点：
//  1. /dev/nvidia<Pm>：父 GPU 的设备节点（Pm = parent minor），宿主机上已存在，由父 GPU CDI spec 提供
//  2. /dev/nvidia-caps/nvidia-cap<GIm>：GI 实例的 capabilities 设备节点
//  3. /dev/nvidia-caps/nvidia-cap<CIm>：CI 实例的 capabilities 设备节点
//
// 对于（2）和（3），只需在容器内正确设置 cgroup 权限并"创建"设备节点即可，
// 不需要宿主机上预先存在同名设备文件。
// cdiDevNodeFromNVCapDevInfo() 利用 /proc/driver/nvidia/capabilities/<path>/access 文件
// 解析出对应的设备 major/minor 号，再构造 CDI DeviceNode 规范。
// 容器运行时根据此规范在容器启动时自动创建设备节点并设置 cgroup 权限。
//
// 参数：
//   - mlt：MIG 设备的活跃元组（包含父 GPU minor、GI ID、CI ID）
//
// 返回值：
//   - []*cdispec.DeviceNode：包含 GI 和 CI 两个设备节点规范的切片
//   - error：解析 capabilities 文件失败时返回错误
func (cdi *CDIHandler) GetDevNodesForMigDevice(mlt *MigLiveTuple) ([]*cdispec.DeviceNode, error) {
	// 构造 GI capabilities 文件路径
	// 格式：/proc/driver/nvidia/capabilities/gpu<parentMinor>/mig/gi<giID>/access
	gipath := fmt.Sprintf("%s/gpu%d/mig/gi%d/access", procNvCapsPath, mlt.ParentMinor, mlt.GIID)

	// 构造 CI capabilities 文件路径
	// 格式：/proc/driver/nvidia/capabilities/gpu<parentMinor>/mig/gi<giID>/ci<ciID>/access
	cipath := fmt.Sprintf("%s/gpu%d/mig/gi%d/ci%d/access", procNvCapsPath, mlt.ParentMinor, mlt.GIID, mlt.CIID)

	// 解析 GI capabilities 文件，获取设备 major/minor 号并构造 CDI 设备节点
	giCapsInfo, err := common.ParseNVCapDeviceInfo(gipath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse GI capabilities file %s: %w", gipath, err)
	}

	// 解析 CI capabilities 文件，获取设备 major/minor 号并构造 CDI 设备节点
	ciCapsInfo, err := common.ParseNVCapDeviceInfo(cipath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CI capabilities file %s: %w", cipath, err)
	}

	// 返回 GI 和 CI 两个 CDI 字符设备节点规范
	// CDICharDevNode() 将解析出的 major/minor 号转换为 cdispec.DeviceNode 格式
	devnodes := []*cdispec.DeviceNode{giCapsInfo.CDICharDevNode(), ciCapsInfo.CDICharDevNode()}
	return devnodes, nil
}
