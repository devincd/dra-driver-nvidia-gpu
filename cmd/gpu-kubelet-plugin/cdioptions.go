/**
# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

// cdioptions 包使用函数式选项（functional options）模式为 CDIHandler 提供灵活的构造配置。
//
// CDI（Container Device Interface）是容器设备接口规范，定义了容器运行时如何发现和注入设备。
// 本驱动中的 CDIHandler 负责为每个 ResourceClaim 生成瞬态 CDI spec 文件，
// 容器运行时（containerd/CRI-O）根据这些 spec 将 GPU 设备和驱动库注入到容器内。
//
// 函数式选项模式的优势：
//   - 避免构造函数参数过多：CDIHandler 有 8+ 个可选配置项，使用传统的位置参数会导致难以维护
//   - 向后兼容：新增配置项只需添加 With*() 函数，不影响已有调用代码
//   - 调用方可读性强：WithDriverRoot("/run/nvidia/driver") 比位置参数更直观
//   - 未设置的选项保持零值默认值，由 NewCDIHandler() 填充安全默认值
//
// 使用示例：
//
//	handler, err := NewCDIHandler(
//	    WithDriverRoot("/run/nvidia/driver"),
//	    WithCDIRoot("/var/run/cdi"),
//	    WithLogger(myLogger),
//	    WithNvml(nvmlInstance),
//	)
//
// 本文件定义的选项函数对应 CDIHandler 的各字段，每个 With*() 函数返回一个闭包，
// 闭包在 NewCDIHandler() 中被依次调用以修改 CDIHandler 实例的字段值。
package main

import (
	nvdevice "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/sirupsen/logrus"
)

// cdiOption 是用于构造 CDIHandler 的函数式选项类型。
//
// 该类型定义了一个函数签名，接受 *CDIHandler 作为参数，在函数体内修改其字段值。
// 这种设计模式的核心思想：
//   - 将"如何配置"与"何时配置"解耦
//   - With*() 系列函数负责生成配置逻辑（返回 cdiOption 闭包）
//   - NewCDIHandler() 负责在合适的时机执行所有配置逻辑
//
// 工作流程：
//  1. 调用方通过 With*() 函数创建 cdiOption 闭包列表
//  2. NewCDIHandler() 创建零值 CDIHandler 实例
//  3. 依次调用每个 cdiOption 闭包，将配置值注入到实例中
//  4. 对未设置的字段填充安全默认值
//
// 未通过 With*() 设置的字段将保持零值（空字符串、nil 等），
// NewCDIHandler() 在执行完所有 option 后会检测并填充默认值。
type cdiOption func(*CDIHandler)

// WithDriverRoot 设置 CDIHandler 的容器内驱动根路径。
//
// driverRoot 是 NVIDIA 驱动在容器文件系统中的挂载点路径。
// 在容器化部署场景中（如 NVIDIA GPU Operator 的驱动容器模式），
// 宿主机的驱动文件系统被挂载到容器的此路径下，例如 "/run/nvidia/driver"。
//
// nvcdi 库使用此路径来查找驱动库文件（如 libcuda.so、libnvidia-ml.so），
// 并基于这些文件生成 CDI spec 中的绑定挂载条目。
// 这些挂载条目告诉容器运行时如何将驱动文件注入到工作负载容器中。
//
// 参数：
//   - root: 驱动根路径字符串，如 "/run/nvidia/driver" 或 "/"
//
// 返回值：
//   - cdiOption: 一个闭包，将 CDIHandler.driverRoot 设置为指定值
func WithDriverRoot(root string) cdiOption {
	return func(c *CDIHandler) {
		c.driverRoot = root
	}
}

// WithDevRoot 设置 CDIHandler 的容器内设备节点根路径。
//
// devRoot 是 /dev/nvidia* 等设备节点在容器文件系统中的挂载点路径。
// 在大多数场景中，devRoot 与 driverRoot 相同，因为设备节点通常位于
// 驱动挂载根路径下的 /dev 目录中。
//
// 但在某些特殊部署中，设备节点可能不在驱动根路径下：
//   - 驱动库通过独立卷挂载，设备节点通过 Device Plugin 单独挂载
//   - 使用 CDI 设备注入而非驱动容器模式时
//
// nvcdi 使用 devRoot 来确定设备节点文件（如 /dev/nvidia0、/dev/nvidiactl）
// 的实际路径，从而在 CDI spec 中生成正确的设备节点条目。
//
// 参数：
//   - root: 设备节点根路径字符串，如 "/run/nvidia/driver" 或 "/"
//
// 返回值：
//   - cdiOption: 一个闭包，将 CDIHandler.devRoot 设置为指定值
func WithDevRoot(root string) cdiOption {
	return func(c *CDIHandler) {
		c.devRoot = root
	}
}

// WithTargetDriverRoot 设置 CDIHandler 的宿主机驱动根路径（路径变换的目标路径）。
//
// targetDriverRoot 用于 CDI spec 中的路径变换（root transformation）。
// 在生成 CDI spec 时，nvcdi 库内部发现的路径是基于容器内 driverRoot 的，
// 但容器运行时需要知道这些文件在宿主机上的实际位置才能正确挂载。
//
// 变换逻辑：
//   - 容器内路径: driverRoot + "/usr/lib64/libcuda.so.535.104.05"
//   - 宿主机路径: targetDriverRoot + "/usr/lib64/libcuda.so.535.104.05"
//
// 例如，当 driverRoot="/run/nvidia/driver"、targetDriverRoot="/" 时，
// CDI spec 中的挂载条目会指示容器运行时将宿主机 "/usr/lib64/libcuda.so.535.104.05"
// 挂载到容器内的 "/run/nvidia/driver/usr/lib64/libcuda.so.535.104.05"。
//
// 典型配置：
//   - 宿主机原生驱动: driverRoot="/", targetDriverRoot="/"
//   - 容器化驱动: driverRoot="/run/nvidia/driver", targetDriverRoot="/"
//
// 参数：
//   - root: 宿主机驱动根路径字符串
//
// 返回值：
//   - cdiOption: 一个闭包，将 CDIHandler.targetDriverRoot 设置为指定值
func WithTargetDriverRoot(root string) cdiOption {
	return func(c *CDIHandler) {
		c.targetDriverRoot = root
	}
}

// WithCDIRoot 设置 CDI spec 文件的输出目录。
//
// CDI spec 文件以 YAML 格式存储在此目录中，文件名遵循 CDI 规范的命名约定：
//   - 供应商级 spec: "<vendor>.yaml"（如 "k8s.gpu.nvidia.com.yaml"）
//   - 瞬态 spec: "<vendor>-<class>-<claimUID>.yaml"
//     （如 "k8s.gpu.nvidia.com-claim-dab5ab50-d59a-42a6-af16-cfd4628c0f7a.yaml"）
//
// 容器运行时（containerd/CRI-O）在启动容器前会扫描此目录以发现可用的 CDI spec，
// 然后根据 Pod 中声明的 CDI 设备名（如 "k8s.gpu.nvidia.com/claim=gpu-0"）
// 从 spec 中查找对应的设备注入规则并执行。
//
// 若未通过此选项设置，NewCDIHandler() 会使用默认值 defaultCDIRoot（"/var/run/cdi"）。
// 该目录必须存在且插件进程对其有写权限，否则 spec 文件写入将失败。
//
// 参数：
//   - cdiRoot: CDI spec 文件输出目录路径，如 "/var/run/cdi"
//
// 返回值：
//   - cdiOption: 一个闭包，将 CDIHandler.cdiRoot 设置为指定值
func WithCDIRoot(cdiRoot string) cdiOption {
	return func(c *CDIHandler) {
		c.cdiRoot = cdiRoot
	}
}

// WithNVIDIACDIHookPath 设置 nvidia-cdi-hook 可执行文件在宿主机上的绝对路径。
//
// nvidia-cdi-hook 是 NVIDIA Container Toolkit 提供的钩子程序，
// 在容器创建阶段由容器运行时通过 prestart hook 机制调用。
// 其主要职责包括：
//   - 更新容器内的 ldcache（动态链接器缓存），确保容器能找到 NVIDIA 驱动库
//   - 创建必要的设备节点（如 MIG 设备的 nvidia-caps 节点）
//   - 执行其他容器启动前的 NVIDIA 特定初始化工作
//
// 该路径会被嵌入到 CDI spec 文件的 hook 条目中，格式如：
//
//	hooks:
//	- hookName: prestart
//	  path: /usr/bin/nvidia-cdi-hook
//	  args: ["nvidia-cdi-hook", "update-ldcache"]
//
// 容器运行时在启动容器时读取 spec 中的 hook 定义，按指定路径调用钩子程序。
//
// 若路径不正确，容器启动时钩子执行将失败，导致容器无法正常访问 GPU 设备。
//
// 参数：
//   - path: nvidia-cdi-hook 的绝对路径，如 "/usr/bin/nvidia-cdi-hook"
//
// 返回值：
//   - cdiOption: 一个闭包，将 CDIHandler.nvidiaCDIHookPath 设置为指定值
func WithNVIDIACDIHookPath(path string) cdiOption {
	return func(c *CDIHandler) {
		c.nvidiaCDIHookPath = path
	}
}

// WithNvml 设置 CDIHandler 使用的 NVML 库接口。
//
// NVML（NVIDIA Management Library）是 NVIDIA 提供的 GPU 管理库，
// 提供以下核心功能：
//   - 设备枚举：遍历节点上所有 GPU 设备，获取 UUID、minor 号、PCIe 信息等
//   - 设备状态查询：查询 GPU 利用率、显存使用、温度、功耗等运行时指标
//   - MIG 管理：创建/销毁 GPU Instance（GI）和 Compute Instance（CI）
//   - 事件订阅：监听 GPU XID 错误事件（用于健康监控）
//
// 在 CDIHandler 中，NVML 主要用于：
//   - 通过 nvcdi.GetDeviceSpecsByID(uuid) 查询设备的 CDI spec（设备节点、驱动库挂载等）
//   - 通过 nvcdi.GetCommonEdits() 获取所有设备共享的容器编辑项
//
// 若未通过此选项设置，NewCDIHandler() 会使用 nvml.New() 创建默认实例。
//
// 参数：
//   - n: nvml.Interface 实例，封装了 NVML 库的 Go 语言绑定
//
// 返回值：
//   - cdiOption: 一个闭包，将 CDIHandler.nvml 设置为指定值
func WithNvml(nvml nvml.Interface) cdiOption {
	return func(c *CDIHandler) {
		c.nvml = nvml
	}
}

// WithDeviceLib 设置 CDIHandler 使用的设备枚举和查询库（go-nvlib 的高层封装）。
//
// nvdevice.Interface（go-nvlib/pkg/nvlib/device）是对 NVML 的更高层次抽象，
// 提供以下增强功能：
//   - 结构化的 GPU 设备模型：Gpu、MigDevice 等类型封装了 NVML 返回的原始数据
//   - MIG 设备发现：自动遍历 GPU 下的所有 MIG 实例（GI/CI），简化 MIG 编程
//   - 设备能力查询：便捷地获取 MIG profile 支持情况、可用分区等
//
// 在 CDIHandler 中，nvdevice 主要被 nvcdi 库内部使用：
//   - 用于枚举设备并生成对应的 CDI spec 条目
//   - 用于发现 MIG 设备层级关系（父 GPU → GI → CI）
//
// 若未通过此选项设置，NewCDIHandler() 会使用 nvdevice.New(nvml) 创建默认实例。
//
// 参数：
//   - nvdevice: nvdevice.Interface 实例
//
// 返回值：
//   - cdiOption: 一个闭包，将 CDIHandler.nvdevice 设置为指定值
func WithDeviceLib(nvdevices nvdevice.Interface) cdiOption {
	return func(c *CDIHandler) {
		c.nvdevice = nvdevices
	}
}

// WithLogger 设置 CDIHandler 内部使用的 logrus 日志记录器。
//
// nvcdi 库内部使用 logrus.Logger 记录调试信息（如设备发现过程、spec 生成细节等）。
// 通过注入自定义 logger，可以精确控制 nvcdi 的日志输出行为。
//
// 典型配置策略：
//   - 生产环境：设置输出到 io.Discard，丢弃 nvcdi 的调试日志，减少日志噪音。
//     因为 nvcdi 的日志量较大且对生产运维价值有限。
//   - 调试环境：设置输出到 os.Stdout 并调整级别为 logrus.DebugLevel，
//     以便排查 CDI spec 生成问题（如设备节点缺失、挂载路径错误等）。
//
// 若未通过此选项设置，NewCDIHandler() 会创建一个默认的 logrus.Logger，
// 并将其输出设置为 io.Discard（静默模式），这是生产环境推荐的行为。
//
// 参数：
//   - logger: *logrus.Logger 实例，已配置好输出目标、格式和级别
//
// 返回值：
//   - cdiOption: 一个闭包，将 CDIHandler.logger 设置为指定值
func WithLogger(logger *logrus.Logger) cdiOption {
	return func(c *CDIHandler) {
		c.logger = logger
	}
}
