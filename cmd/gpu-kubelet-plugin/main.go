/*
 * Copyright (c) 2022-2023 NVIDIA CORPORATION.  All rights reserved.
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

// main 包是 GPU DRA (Dynamic Resource Allocation) kubelet 插件的入口点。
//
// 本插件实现了 Kubernetes DRA 驱动接口，负责：
//   - 发现节点上的 NVIDIA GPU 设备并发布 ResourceSlice 到 API Server
//   - 响应 kubelet 的 NodePrepareResources / NodeUnprepareResources gRPC 调用
//   - 生成 CDI (Container Device Interface) spec，供容器运行时注入 GPU 设备
//   - 监控 GPU 健康状态，不健康的设备从 ResourceSlice 中移除
//   - 管理 MIG (Multi-Instance GPU) 设备的创建和销毁
//   - 支持 VFIO 直通模式（PassthroughSupport 特性门控控制）
//   - 支持 MPS (Multi-Process Service) 和 Time-Slicing 共享模式
//   - 周期性清理 Checkpoint 中的"僵尸 ResourceClaim"
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/urfave/cli/v2"

	"k8s.io/component-base/logs"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"

	"github.com/NVIDIA/k8s-dra-driver-gpu/internal/common"
	"github.com/NVIDIA/k8s-dra-driver-gpu/internal/info"
	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
	pkgflags "github.com/NVIDIA/k8s-dra-driver-gpu/pkg/flags"
)

// 驱动相关常量
const (
	// DriverName 是 DRA 驱动的名称，用于在 DeviceClass、ResourceClaim 等资源中标识本驱动。
	// 格式遵循 Kubernetes 扩展资源的域名规范：<domain>/<resource>。
	// 此名称必须与 Helm chart 中的 DriverName 完全一致。
	DriverName = "gpu.nvidia.com"

	// DriverPluginCheckpointFileBasename 是 Checkpoint 文件的基本名称，
	// 存储在 kubelet 插件目录下（如 /var/lib/kubelet/plugins/gpu.nvidia.com/checkpoint.json）。
	// Checkpoint 用于持久化已 Prepare 的 Claim 信息，插件重启后可恢复状态。
	DriverPluginCheckpointFileBasename = "checkpoint.json"
)

// Flags 包含插件的所有命令行参数。
// 这些参数通过环境变量或命令行标志传入，控制插件的运行行为。
type Flags struct {
	// kubeClientConfig 包含 Kubernetes 客户端的配置（API Server 地址、kubeconfig 路径等）
	kubeClientConfig pkgflags.KubeClientConfig

	// nodeName 是当前节点名称，由 DownwardAPI 注入，用于 ResourceSlice 的 nodeName 字段
	nodeName string
	// namespace 是插件运行的命名空间，用于创建/管理 Kubernetes 资源
	namespace string
	// cdiRoot 是 CDI spec 文件的输出目录，容器运行时从此目录读取 CDI spec
	cdiRoot string
	// containerDriverRoot 是容器内 NVIDIA 驱动的挂载路径，用于 CDI spec 生成
	// 当驱动以容器方式部署时，容器内的驱动路径可能与宿主机不同
	containerDriverRoot string
	// hostDriverRoot 是宿主机上 NVIDIA 驱动的安装路径
	// 典型值："/"（驱动直接安装在宿主机）或 "/run/nvidia/driver"（驱动容器部署）
	hostDriverRoot string
	// nvidiaCDIHookPath 是 nvidia-cdi-hook 二进制文件在宿主机文件系统中的绝对路径
	// CDI spec 中引用此路径，容器运行时在注入设备时调用此 hook
	nvidiaCDIHookPath string
	// imageName 是容器镜像的完整名称（含仓库和标签），用于渲染 Helm 模板
	imageName string
	// kubeletRegistrarDirectoryPath 是 kubelet 插件注册目录的绝对路径
	// 插件在此目录下创建注册文件，kubelet 监视此目录以发现新注册的 DRA 插件
	kubeletRegistrarDirectoryPath string
	// kubeletPluginsDirectoryPath 是 kubelet 插件数据目录的绝对路径
	// 插件在此目录下存储 Checkpoint 文件、CDI spec 等持久化数据
	kubeletPluginsDirectoryPath string
	// healthcheckPort 是 gRPC 健康检查服务的端口号
	// 正数：使用指定的端口号；0：随机分配端口；负数：禁用健康检查
	healthcheckPort int
	// klogVerbosity 是 klog 的日志详细程度（0-9），用于运行时动态控制日志级别
	klogVerbosity int
	// additionalXidsToIgnore 是逗号分隔的额外 XID 错误码列表
	// 列在此列表中的 XID 错误不会触发设备标记为 Unhealthy
	additionalXidsToIgnore string
}

// Config 包含插件的完整运行时配置，由命令行参数和 Kubernetes 客户端组合而成。
type Config struct {
	// flags 包含所有命令行参数
	flags *Flags
	// clientsets 包含各类 Kubernetes API 客户端（CoreV1、ResourceV1 等）
	clientsets pkgflags.ClientSets
}

// DriverPluginPath 返回 kubelet 插件目录下本驱动的子目录路径。
// 格式：<kubeletPluginsDirectoryPath>/<DriverName>
// 例如：/var/lib/kubelet/plugins/gpu.nvidia.com/
// 此目录用于存储 Checkpoint 文件、CDI spec 和其他插件持久化数据。
func (c Config) DriverPluginPath() string {
	return filepath.Join(c.flags.kubeletPluginsDirectoryPath, DriverName)
}

// main 是程序入口，创建 CLI 应用并运行。
// 若运行出错（如参数校验失败、驱动启动失败），输出错误信息并以非零退出码退出。
func main() {
	if err := newApp().Run(os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// newApp 创建 urfave/cli 应用实例，定义所有命令行参数和启动逻辑。
// 返回值：配置完整的 *cli.App 实例
func newApp() *cli.App {
	// 初始化日志配置（控制 klog 输出格式、目标等）
	loggingConfig := pkgflags.NewLoggingConfig()
	// 初始化特性门控配置（控制 DynamicMIG、PassthroughSupport 等特性的开关）
	featureGateConfig := pkgflags.NewFeatureGateConfig()
	// 初始化命令行参数结构体
	flags := &Flags{}

	// 定义命令行参数列表
	cliFlags := []cli.Flag{
		// 节点名称，必填项，通常通过 DownwardAPI 从 Node 对象注入
		&cli.StringFlag{
			Name:        "node-name",
			Usage:       "The name of the node to be worked on.",
			Required:    true,
			Destination: &flags.nodeName,
			EnvVars:     []string{"NODE_NAME"},
		},
		// 插件运行的命名空间，默认为 "default"
		&cli.StringFlag{
			Name:        "namespace",
			Usage:       "The namespace used for the custom resources.",
			Value:       "default",
			Destination: &flags.namespace,
			EnvVars:     []string{"NAMESPACE"},
		},
		// CDI spec 文件的输出目录，默认为 /etc/cdi
		&cli.StringFlag{
			Name:        "cdi-root",
			Usage:       "Absolute path to the directory where CDI files will be generated.",
			Value:       "/etc/cdi",
			Destination: &flags.cdiRoot,
			EnvVars:     []string{"CDI_ROOT"},
		},
		// 宿主机上 NVIDIA 驱动的安装根路径，默认为 "/"（驱动直接安装在宿主机）
		&cli.StringFlag{
			Name:        "nvidia-driver-root",
			Aliases:     []string{"host_driver-root"},
			Value:       "/",
			Usage:       "the root path for the NVIDIA driver installation on the host (typical values are '/' or '/run/nvidia/driver')",
			Destination: &flags.hostDriverRoot,
			EnvVars:     []string{"NVIDIA_DRIVER_ROOT", "HOST_DRIVER_ROOT"},
		},
		// 容器内 NVIDIA 驱动的挂载路径，用于 CDI spec 生成
		&cli.StringFlag{
			Name:        "container-driver-root",
			Value:       "/driver-root",
			Usage:       "the path where the NVIDIA driver root is mounted in the container; used for generating CDI specifications",
			Destination: &flags.containerDriverRoot,
			EnvVars:     []string{"DRIVER_ROOT_CTR_PATH"},
		},
		// nvidia-cdi-hook 二进制文件在宿主机上的绝对路径
		&cli.StringFlag{
			Name:        "nvidia-cdi-hook-path",
			Usage:       "Absolute path to the nvidia-cdi-hook executable in the host file system. Used in the generated CDI specification.",
			Destination: &flags.nvidiaCDIHookPath,
			EnvVars:     []string{"NVIDIA_CDI_HOOK_PATH"},
		},
		// 容器镜像名称，必填项，用于 Helm 模板渲染
		&cli.StringFlag{
			Name:        "image-name",
			Usage:       "The full image name to use for rendering templates.",
			Required:    true,
			Destination: &flags.imageName,
			EnvVars:     []string{"IMAGE_NAME"},
		},
		// kubelet 插件注册目录路径
		&cli.StringFlag{
			Name:        "kubelet-registrar-directory-path",
			Usage:       "Absolute path to the directory where kubelet stores plugin registrations.",
			Value:       kubeletplugin.KubeletRegistryDir,
			Destination: &flags.kubeletRegistrarDirectoryPath,
			EnvVars:     []string{"KUBELET_REGISTRAR_DIRECTORY_PATH"},
		},
		// kubelet 插件数据目录路径
		&cli.StringFlag{
			Name:        "kubelet-plugins-directory-path",
			Usage:       "Absolute path to the directory where kubelet stores plugin data.",
			Value:       kubeletplugin.KubeletPluginsDir,
			Destination: &flags.kubeletPluginsDirectoryPath,
			EnvVars:     []string{"KUBELET_PLUGINS_DIRECTORY_PATH"},
		},
		// gRPC 健康检查服务端口
		&cli.IntFlag{
			Name:        "healthcheck-port",
			Usage:       "Port to start a gRPC healthcheck service. When positive, a literal port number. When zero, a random port is allocated. When negative, the healthcheck service is disabled.",
			Value:       -1,
			Destination: &flags.healthcheckPort,
			EnvVars:     []string{"HEALTHCHECK_PORT"},
		},
		// 额外忽略的 XID 错误码（逗号分隔）
		// TODO: 改为 StringSliceFlag 以支持更灵活的配置
		&cli.StringFlag{
			Name:        "additional-xids-to-ignore",
			Usage:       "A comma-separated list of additional XIDs to ignore.",
			Value:       "",
			Destination: &flags.additionalXidsToIgnore,
			EnvVars:     []string{"ADDITIONAL_XIDS_TO_IGNORE"},
		},
	}

	// 追加 Kubernetes 客户端配置相关的参数（kubeconfig、API Server 地址等）
	cliFlags = append(cliFlags, flags.kubeClientConfig.Flags()...)
	// 追加特性门控配置相关的参数
	cliFlags = append(cliFlags, featureGateConfig.Flags()...)
	// 追加日志配置相关的参数（日志级别、格式等）
	cliFlags = append(cliFlags, loggingConfig.Flags()...)

	app := &cli.App{
		Name:            "gpu-kubelet-plugin",
		Usage:           "gpu-kubelet-plugin implements a DRA driver plugin for NVIDIA GPUs.",
		ArgsUsage:       " ",
		HideHelpCommand: true,
		Flags:           cliFlags,
		// Before 在 Action 之前执行，用于参数校验和初始化
		Before: func(c *cli.Context) error {
			// 不接受位置参数
			if c.Args().Len() > 0 {
				return fmt.Errorf("arguments not supported: %v", c.Args().Slice())
			}

			// 日志配置必须在任何日志输出之前应用，
			// 否则 klog 可能使用默认配置输出，导致格式不一致
			err := loggingConfig.Apply()

			// 将 klog 的日志详细程度保存到 config 中，供运行时检查。
			// 这是因为原始的 cliFlags 在 Apply 后不再可直接访问，
			// 但某些组件（如 CDI 生成器）需要根据日志级别决定是否输出详细日志。
			flags.klogVerbosity = int(loggingConfig.Config.Verbosity)
			// 记录启动配置信息，便于排查问题
			pkgflags.LogStartupConfig(flags, loggingConfig)
			return err
		},
		// Action 是应用的主逻辑，在 Before 成功后执行
		Action: func(c *cli.Context) error {
			// 校验特性门控配置是否合法
			if err := featuregates.ValidateFeatureGates(); err != nil {
				return fmt.Errorf("feature gate validation failed: %w", err)
			}

			// 创建 Kubernetes API 客户端集合
			clientSets, err := flags.kubeClientConfig.NewClientSets()
			if err != nil {
				return fmt.Errorf("create client: %w", err)
			}

			// 组装完整的运行时配置
			config := &Config{
				flags:      flags,
				clientsets: clientSets,
			}

			// 启动插件主循环
			return RunPlugin(c.Context, config)
		},
		// After 在 Action 之后执行（无论成功或失败），
		// 用于执行清理工作，如刷新日志缓冲区。
		// 在 urfave cli v2 中，最终的错误来自 Action、Before 或 After
		// 中最后一个非 nil 的返回值。
		After: func(c *cli.Context) error {
			klog.Infof("shutdown")
			// 确保所有日志缓冲区的内容都被写出，防止日志丢失
			logs.FlushLogs()
			return nil
		},
		// 版本号，来自编译时注入的版本信息
		Version: info.GetVersionString(),
	}

	// 移除 -v 作为 --version 的别名，避免与 klog 的 -v（日志详细程度）标志冲突。
	// klog 使用 -v=N 控制日志级别，而 urfave/cli 默认将 -v 映射到 --version。
	f, ok := cli.VersionFlag.(*cli.BoolFlag)
	if ok {
		f.Aliases = nil
	}

	return app
}

// RunPlugin 初始化并运行 GPU kubelet 插件。
// 这是插件的主入口函数，负责：
//  1. 启动调试信号处理器
//  2. 创建插件工作目录
//  3. 配置 nvidia-cdi-hook 二进制路径
//  4. 初始化 CDI spec 输出目录
//  5. 注册系统信号处理器，实现优雅退出
//  6. 创建并启动 DRA Driver 实例
//  7. 等待终止信号并执行优雅关闭
//
// 参数：
//   - ctx: 父级 context，用于控制插件生命周期
//   - config: 插件运行所需的全部配置信息
//
// 返回值：
//   - nil: 正常退出
//   - error: 启动或运行过程中的错误
func RunPlugin(ctx context.Context, config *Config) error {
	// 启动调试信号处理器（通常监听 SIGUSR1 等信号，用于触发 goroutine dump 等调试操作）
	common.StartDebugSignalHandlers()

	// 创建 DRA kubelet 插件的 socket 目录（如 /var/lib/kubelet/plugins/gpu.nvidia.com/）
	// 0750 权限：owner 可读写执行，group 可读执行，other 无权限
	err := os.MkdirAll(config.DriverPluginPath(), 0750)
	if err != nil {
		return err
	}

	// 定位并设置 nvidia-cdi-hook 二进制路径。
	// CDI hook 在容器启动时由容器运行时调用，负责完成 NVIDIA 设备的注入初始化。
	if err := config.setNvidiaCDIHookPath(); err != nil {
		return fmt.Errorf("error setting up nvidia-cdi-hook: %w", err)
	}

	// 检查并初始化 CDI spec 文件的输出目录（即 cdiRoot）。
	// CDI spec 文件由插件写入此目录，容器运行时从这里读取设备规范。
	info, err := os.Stat(config.flags.cdiRoot)
	switch {
	case err != nil && os.IsNotExist(err):
		// 目录不存在：自动创建，权限同上
		err := os.MkdirAll(config.flags.cdiRoot, 0750)
		if err != nil {
			return err
		}
	case err != nil:
		// 其他 stat 错误（如权限不足）：直接返回
		return err
	case !info.IsDir():
		// 路径已存在但不是目录（如被文件占用）：无法使用，返回明确错误
		return fmt.Errorf("path for cdi file generation is not a directory: '%v'", config.flags.cdiRoot)
	}

	// 将系统终止信号（SIGHUP/SIGINT/SIGTERM/SIGQUIT）绑定到 context。
	// 任意信号到达时，ctx 将被取消，触发后续的优雅退出流程。
	// defer cancel() 确保资源在函数退出时被释放。
	ctx, cancel := signal.NotifyContext(ctx, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)
	defer cancel()

	// 创建并启动 GPU DRA Driver 实例。
	// Driver 内部负责：ResourceSlice 发布、CDI spec 生成、kubelet gRPC 服务注册等。
	driver, err := NewDriver(ctx, config)
	if err != nil {
		return fmt.Errorf("error creating driver: %w", err)
	}

	// 阻塞等待，直到 ctx 被取消（即收到终止信号）。
	// 这是插件的主循环挂起点，所有实际工作在 driver 内部的 goroutine 中并发执行。
	<-ctx.Done()

	// ctx.Err() 在正常信号触发时返回 context.Canceled，属于预期情况，无需记录。
	// 若返回其他错误（如 context.DeadlineExceeded），则属于异常情况，记录日志。
	if err := ctx.Err(); err != nil && !errors.Is(err, context.Canceled) {
		// context.Canceled 是正常情况（进程收到信号后取消 context），
		// 只有其他类型的错误才需要记录
		klog.Errorf("error from context: %v", err)
	}

	// 执行 Driver 的优雅关闭：
	// 注销 kubelet plugin socket、清理 CDI spec 文件、释放 GPU 资源等。
	// 关闭失败仅记录日志，不阻断 RunPlugin 返回（避免卡死进程）。
	err = driver.Shutdown()
	if err != nil {
		klog.Errorf("unable to cleanly shutdown driver: %v", err)
	}

	return nil
}

// setNvidiaCDIHookPath 设置 nvidia-cdi-hook 二进制文件的路径。
// 如果通过命令行参数（--nvidia-cdi-hook-path）已设置了路径，则直接使用该路径，不做任何修改。
// 如果路径为空（未设置），则将容器内预置的 /usr/bin/nvidia-cdi-hook 复制到
// DriverPluginPath 目录下（如 /var/lib/kubelet/plugins/gpu.nvidia.com/nvidia-cdi-hook），
// 并将路径设置为目标位置。
//
// 这样做的原因是：容器运行时在宿主机文件系统上查找 CDI hook，
// 而容器内的 /usr/bin/nvidia-cdi-hook 在容器外不可见，
// 因此需要将 hook 二进制复制到 kubelet 插件目录（该目录会被挂载到宿主机）。
//
// 容器镜像中的 /usr/bin/nvidia-cdi-hook 是在构建阶段
// 从 NVIDIA Container Toolkit 镜像复制过来的。
//
// 返回值：
//   - nil: 路径设置成功或路径已存在
//   - error: 读取或复制文件失败
func (c Config) setNvidiaCDIHookPath() error {
	// 若已通过命令行参数设置路径，直接使用
	if c.flags.nvidiaCDIHookPath != "" {
		return nil
	}

	// 容器内预置的 CDI hook 源路径
	sourcePath := "/usr/bin/nvidia-cdi-hook"
	// 目标路径：kubelet 插件目录下的 nvidia-cdi-hook
	targetPath := filepath.Join(c.DriverPluginPath(), "nvidia-cdi-hook")

	// 读取源二进制文件内容
	input, err := os.ReadFile(sourcePath)
	if err != nil {
		return fmt.Errorf("error reading nvidia-cdi-hook: %w", err)
	}

	// 将二进制文件写入目标路径，权限 0755（可执行）
	if err := os.WriteFile(targetPath, input, 0755); err != nil {
		return fmt.Errorf("error copying nvidia-cdi-hook: %w", err)
	}

	// 更新配置中的 CDI hook 路径为目标路径
	c.flags.nvidiaCDIHookPath = targetPath

	return nil
}
