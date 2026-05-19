/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

// sharing.go 实现了 GPU 共享（Time-Slicing 和 MPS）的核心逻辑和管理功能。
//
// 本文件是 kubelet 插件中 GPU 多租户共享的关键模块，负责：
//   - TimeSlicingManager：通过 nvidia-smi 设置 GPU 计算模式和时间切片间隔
//   - MpsManager：管理 MPS 控制守护进程的完整生命周期（创建、启动、停止、清理）
//   - MpsControlDaemon：单个 ResourceClaim 对应的 MPS 守护进程实例
//   - getDefaultShmSize：计算 MPS 共享内存 tmpfs 的默认大小
//
// GPU 共享策略对比：
//   - Time-Slicing：抢占式，不同进程轮流使用 GPU，延迟较高但隔离性好
//   - MPS：协作式，多进程共享 CUDA 上下文并发执行，吞吐量高但隔离性弱
//
// MPS 守护进程部署架构：
//   每个 ResourceClaim 的 MPS 会话对应一个独立的 Kubernetes Deployment，
//   该 Deployment 运行在与工作负载相同的节点上（通过 NodeSelector 绑定）。
//   工作负载容器通过 bind mount 共享 MPS 的 pipe/shm 目录来与 MPS 服务器通信。
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"text/template"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	"k8s.io/mount-utils"

	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"
	cdispec "tags.cncf.io/container-device-interface/specs-go"

	configapi "github.com/NVIDIA/k8s-dra-driver-gpu/api/nvidia.com/resource/v1beta1"
	"github.com/NVIDIA/k8s-dra-driver-gpu/pkg/featuregates"
)

const (
	// MpsControlFilesDirName 是 MPS 控制文件（pipe/shm/log）的子目录名，
	// 拼接到 DriverPluginPath 下构成完整根目录路径。
	// 例如：DriverPluginPath = /var/run/nvidia/dra，则 MPS 根目录为 /var/run/nvidia/dra/mps
	MpsControlFilesDirName = "mps"

	// MpsControlDaemonTemplatePath 是 MPS 控制守护进程 Deployment YAML 模板的容器内路径。
	// 该模板在构建镜像时嵌入，运行时通过 Go text/template 渲染后提交到 Kubernetes API。
	// 模板中包含 Deployment 的完整定义，包括容器镜像、资源限制、环境变量等。
	MpsControlDaemonTemplatePath = "/templates/mps-control-daemon.tmpl.yaml"

	// MpsControlDaemonNameFmt 是 MPS 控制守护进程 Kubernetes Deployment 名称的格式串。
	// %v 占位符填入 ClaimUID（或其派生的哈希 ID），确保每个 Claim 的守护进程名唯一。
	// 例如：mps-control-daemon-dab5ab50-d59a-42a6-af16-cfd4628c0f7a-a1b2c
	MpsControlDaemonNameFmt = "mps-control-daemon-%v"
)

// TimeSlicingManager 管理 GPU 时间切片（Time-Slicing）配置。
//
// GPU 时间切片允许多个 CUDA 工作负载共享同一 GPU，
// 通过 nvidia-smi 命令设置 GPU 计算模式和时间切片间隔（微秒级）。
// 工作原理：当多个 CUDA 进程请求 GPU 时，驱动按时间片轮转分配 GPU 使用权，
// 每个进程在分配的时间片内独占 GPU，时间片结束后被抢占，切换到下一个进程。
//
// 与 MPS 的区别：
//   - 时间切片是抢占式的，不同进程轮流使用 GPU，存在上下文切换开销
//   - MPS 通过共享 CUDA 上下文降低上下文切换开销，允许真正并发执行
//   - 时间切片隔离性更好（每个进程有独立的 CUDA 上下文）
//   - MPS 吞吐量更高（多进程可同时使用 SM），但一个进程崩溃可能影响其他进程
//
// 仅在 TimeSlicingSettings 特性门控开启时创建和使用。
// 调用 SetTimeSlice() 时传入的 UUID 必须是完整物理 GPU（非 MIG 设备）的 UUID，
// 因为 nvidia-smi 不支持直接对 MIG 实例设置时间切片。
type TimeSlicingManager struct {
	// nvdevlib 是设备库实例，提供 NVML 和 nvidia-smi 命令行操作能力。
	nvdevlib *deviceLib
}

// MpsManager 管理 NVIDIA MPS（Multi-Process Service）控制守护进程的完整生命周期。
//
// MPS 允许多个 CUDA 进程共享同一 GPU 的 CUDA 上下文，减少上下文切换开销，
// 提升多进程共享 GPU 场景下的整体吞吐量（典型场景：Kubernetes 多 Pod 共享 GPU）。
//
// 工作原理：
//   1. MPS 服务器进程持有 GPU 的 CUDA 上下文（需 GPU 处于 EXCLUSIVE_PROCESS 模式）
//   2. MPS 客户端进程通过共享内存与 MPS 服务器通信，提交 CUDA 请求
//   3. MPS 服务器协调多个客户端的 CUDA 调用，实现并发执行
//   4. 客户端间共享 CUDA 上下文，避免了传统时间切片的上下文切换开销
//
// 每个 ResourceClaim 的 MPS 会话对应一个独立的 MPS 控制守护进程 Deployment，
// 该 Deployment 运行在与工作负载相同的节点上（通过 NodeSelector 绑定）。
//
// 仅在 MPSSupport 特性门控开启时创建和使用。
// 与 DynamicMIG 不兼容（TODO：DynamicMIG 下需先创建 MIG 设备再启动 MPS），
// 原因是 DynamicMIG 模式下 GPU 需要处于 MIG 模式，而 MPS 要求 EXCLUSIVE_PROCESS 模式，
// 两者在计算模式要求上冲突。
type MpsManager struct {
	// config 是插件全局配置，包含 Kubernetes 客户端、命令行标志等。
	config *Config

	// controlFilesRoot 是 MPS pipe/shm/log 文件的宿主机根目录（DriverPluginPath/mps）。
	// 每个 MPS 守护进程在此目录下创建以自身 ID 命名的子目录。
	controlFilesRoot string

	// hostDriverRoot 是宿主机 NVIDIA 驱动根路径，传入 MPS 守护进程用于查找驱动库。
	// 在特权容器部署中，宿主机驱动被挂载到容器内，MPS 守护进程需要知道宿主机路径
	// 才能正确加载驱动库。
	hostDriverRoot string

	// templatePath 是 MPS 守护进程 Deployment YAML 模板路径（容器内）。
	// 模板使用 Go text/template 语法，运行时填入节点名、设备 UUID 等参数。
	templatePath string

	// nvdevlib 是设备库实例，提供设置 GPU 计算模式等 NVML 操作能力。
	nvdevlib *deviceLib
}

// MpsControlDaemon 表示单个 ResourceClaim 对应的 MPS 控制守护进程实例。
// id 是守护进程的唯一标识符（由 claimUID + 设备 UUID 哈希生成，长度可控）。
// 守护进程运行期间，其 pipe/shm/log 目录由此结构体管理（创建、挂载、清理）。
//
// 生命周期：
//   - 创建：Prepare 阶段，调用 Start() 启动守护进程
//   - 运行：守护进程 Pod 在节点上运行，工作负载容器通过 bind mount 共享 pipe/shm
//   - 停止：Unprepare 阶段，调用 Stop() 停止守护进程并清理本地资源
type MpsControlDaemon struct {
	// id 是守护进程的唯一标识符，格式为 <claimUID>-<hashPrefix5>。
	// 通过 GetMpsControlDaemonID 生成，确保相同输入始终产生相同 ID。
	id string

	// nodeName 是目标节点名，用于 Deployment 的 NodeSelector，
	// 确保守护进程 Pod 调度到与工作负载相同的节点。
	nodeName string

	// namespace 是 Kubernetes 命名空间，MPS 守护进程 Deployment 创建在此命名空间中。
	namespace string

	// name 是 Kubernetes Deployment 名称，格式为 mps-control-daemon-<id>。
	name string

	// rootDir 是工作目录根路径，格式为 controlFilesRoot/<id>。
	// 包含 pipe、shm、log 三个子目录。
	rootDir string

	// pipeDir 是 MPS pipe 目录，路径为 rootDir/pipe。
	// MPS 服务器在此目录创建命名管道文件，客户端通过管道发送 CUDA 请求。
	pipeDir string

	// shmDir 是 MPS shm 目录，路径为 rootDir/shm。
	// 挂载为 tmpfs，CUDA IPC 通过此目录传递共享内存句柄。
	shmDir string

	// logDir 是 MPS 日志目录，路径为 rootDir/log。
	// MPS 服务器和客户端的日志文件存储在此目录中。
	logDir string

	// devices 是该守护进程服务的设备集合，提供 UUID 列表。
	// 接口类型 UUIDProvider 允许适配不同的设备类型（完整 GPU、MIG 设备等）。
	devices UUIDProvider

	// manager 是指向父 MpsManager 的指针，用于访问配置和 K8s 客户端。
	// 守护进程通过 manager 提交 Deployment 创建/删除请求到 Kubernetes API。
	manager *MpsManager
}

// MpsControlDaemonTemplateData 是渲染 MPS Deployment YAML 模板时使用的数据结构体。
// 字段名与模板中的 {{.FieldName}} 占位符对应。
// DefaultPinnedDeviceMemoryLimits 按 UUID 指定每个设备的显存固定限额（可选），
// 用于限制每个 MPS 客户端可固定（pin）的显存量，防止单个客户端独占全部显存。
type MpsControlDaemonTemplateData struct {
	// NodeName 是目标节点名，用于 Deployment 的 NodeSelector。
	NodeName string

	// MpsControlDaemonNamespace 是 MPS 守护进程的 Kubernetes 命名空间。
	MpsControlDaemonNamespace string

	// MpsControlDaemonName 是 MPS 守护进程的 Kubernetes Deployment 名称。
	MpsControlDaemonName string

	// CUDA_VISIBLE_DEVICES 是传递给 MPS 守护进程容器的 CUDA 可见设备环境变量，
	// 值为逗号分隔的 GPU UUID 列表，告诉 MPS 服务器应管理哪些 GPU。
	// nolint:stylecheck 注释抑制 golint 对非 Go 命名约定的警告（保持与 CUDA 环境变量名一致）。
	CUDA_VISIBLE_DEVICES string //nolint:stylecheck

	// DefaultActiveThreadPercentage 是 MPS 服务器的默认活跃线程百分比（0-100），
	// 控制所有 MPS 客户端可并发使用的 SM 线程数占 GPU 总线程数的比例。
	// 空字符串表示不设置（使用 MPS 默认值 100）。
	DefaultActiveThreadPercentage string

	// DefaultPinnedDeviceMemoryLimits 是按 GPU UUID 指定的显存固定限额映射，
	// 格式为 map[uuid]"limit"（如 "80%"、"10GiB"）。
	// 每个设备可单独配置，未配置的设备使用 MPS 默认值（全部可用显存）。
	DefaultPinnedDeviceMemoryLimits map[string]string

	// NvidiaDriverRoot 是宿主机 NVIDIA 驱动根路径，传入 MPS 守护进程容器。
	// MPS 服务器需要加载 NVIDIA 驱动库，此路径告诉容器在何处查找驱动。
	NvidiaDriverRoot string

	// MpsShmDirectory 是 MPS 共享内存目录的宿主机路径，
	// 将被绑定挂载到容器的 /dev/shm，用于 CUDA IPC 通信。
	MpsShmDirectory string

	// MpsPipeDirectory 是 MPS 管道目录的宿主机路径，
	// 将被绑定挂载到容器的 /tmp/nvidia-mps，用于 MPS 客户端/服务器通信。
	MpsPipeDirectory string

	// MpsLogDirectory 是 MPS 日志目录的宿主机路径，
	// 将被绑定挂载到容器的日志目录，用于存储 MPS 服务器日志。
	MpsLogDirectory string

	// MpsImageName 是 MPS 守护进程容器镜像名称，
	// 通常与 DRA 驱动插件使用相同的镜像。
	MpsImageName string

	// FeatureGates 是当前启用的特性门控映射，
	// 用于模板中条件渲染特定的配置项（如资源限制、环境变量等）。
	FeatureGates map[string]bool
}

// NewTimeSlicingManager 构造 TimeSlicingManager 实例。
//
// 参数：
//   - deviceLib：设备库实例，提供 nvidia-smi 命令行操作能力
//
// 返回值：
//   - *TimeSlicingManager：构造完成的时间切片管理器实例
func NewTimeSlicingManager(deviceLib *deviceLib) *TimeSlicingManager {
	return &TimeSlicingManager{
		nvdevlib: deviceLib,
	}
}

// SetTimeSlice 设置指定完整 GPU 的计算模式和时间切片间隔。
//
// 操作顺序（必须严格按此顺序执行）：
//  1. 将 GPU 计算模式设置为 DEFAULT（允许多进程共享，时间切片生效的前提）
//     计算模式说明：
//     - DEFAULT：多个进程可同时使用 GPU（时间切片模式的前提）
//     - EXCLUSIVE_PROCESS：只有一个进程可创建 CUDA 上下文（MPS 模式的前提）
//     - EXCLUSIVE_THREAD：只有一个线程可使用 GPU（已废弃）
//     - PROHIBITED：GPU 不允许任何 CUDA 计算
//  2. 设置时间切片间隔（微秒级，通过 nvidia-smi compute-policy --set-timeslice 实现）
//     时间切片间隔越长，每个进程单次使用 GPU 的时间越长，上下文切换频率越低，
//     但进程间延迟也越大。默认值因驱动版本而异。
//
// 参数：
//   - uuids：必须是完整物理 GPU（非 MIG）的 UUID 列表，调用方须确保这一点
//     原因：nvidia-smi 不支持直接对 MIG 实例设置时间切片
//   - config：时间切片配置，包含 Interval 字段（以微秒为单位的时间切片间隔）
//
// 返回值：
//   - error：设置计算模式或时间切片失败时返回错误
func (t *TimeSlicingManager) SetTimeSlice(uuids []string, config *configapi.TimeSlicingConfig) error {
	// 第 1 步：将计算模式设为 DEFAULT，允许多进程共享 GPU
	err := t.nvdevlib.setComputeMode(uuids, "DEFAULT")
	if err != nil {
		return fmt.Errorf("error setting compute mode: %w", err)
	}

	// 第 2 步：设置时间切片间隔
	err = t.nvdevlib.setTimeSlice(uuids, config.Interval.Int())
	if err != nil {
		return fmt.Errorf("error setting time slice: %w", err)
	}

	return nil
}

// NewMpsManager 构造 MpsManager 实例，计算 MPS 控制文件的根目录路径。
//
// 参数：
//   - config：插件全局配置
//   - deviceLib：设备库实例
//   - hostDriverRoot：宿主机 NVIDIA 驱动根路径
//   - templatePath：MPS 守护进程 Deployment YAML 模板路径
//
// 返回值：
//   - *MpsManager：构造完成的 MPS 管理器实例
func NewMpsManager(config *Config, deviceLib *deviceLib, hostDriverRoot, templatePath string) *MpsManager {
	// 拼接 MPS 控制文件根目录：DriverPluginPath/mps
	controlFilesRoot := filepath.Join(config.DriverPluginPath(), MpsControlFilesDirName)

	return &MpsManager{
		controlFilesRoot: controlFilesRoot,
		hostDriverRoot:   hostDriverRoot,
		templatePath:     templatePath,
		config:           config,
		nvdevlib:         deviceLib,
	}
}

// NewMpsControlDaemon 为指定 Claim 和设备集合创建 MpsControlDaemon 实例。
// 使用 GetMpsControlDaemonID 生成稳定的 ID（相同输入始终产生相同 ID），
// 确保同一 Claim 的多次 Prepare 调用不会创建多个守护进程。
//
// 参数：
//   - claimUID：ResourceClaim 的唯一标识符
//   - devices：该守护进程服务的设备集合，提供 UUID 列表
//
// 返回值：
//   - *MpsControlDaemon：创建完成的 MPS 控制守护进程实例
func (m *MpsManager) NewMpsControlDaemon(claimUID string, devices UUIDProvider) *MpsControlDaemon {
	// 生成稳定的守护进程 ID
	id := m.GetMpsControlDaemonID(claimUID, devices)

	return &MpsControlDaemon{
		id:        id,
		nodeName:  m.config.flags.nodeName,
		namespace: m.config.flags.namespace,
		// Deployment 名称格式：mps-control-daemon-<id>
		name:    fmt.Sprintf(MpsControlDaemonNameFmt, id),
		// 工作目录层级：controlFilesRoot/<id>/pipe、shm、log
		rootDir: fmt.Sprintf("%s/%s", m.controlFilesRoot, id),
		pipeDir: fmt.Sprintf("%s/%s/%s", m.controlFilesRoot, id, "pipe"),
		shmDir:  fmt.Sprintf("%s/%s/%s", m.controlFilesRoot, id, "shm"),
		logDir:  fmt.Sprintf("%s/%s/%s", m.controlFilesRoot, id, "log"),
		devices: devices,
		manager: m,
	}
}

// GetMpsControlDaemonID 基于 claimUID 和设备 UUID 列表的哈希生成唯一守护进程 ID。
//
// 生成策略：将所有设备 UUID 排序后连接为字符串（由 UUIDs() 保证排序稳定），
// 计算 SHA-256 哈希，取前 5 字节的十六进制字符串作为后缀。
// 格式：<claimUID>-<hashPrefix5>
//
// 使用哈希而非直接使用 UUID 原因：
//   - 多设备场景下 UUID 列表可能很长（每个 UUID 约 40 字符）
//   - 哈希后缀使名称长度可控，满足 Kubernetes 资源名称长度限制（253 字符）
//   - 前 5 字节（20 位十六进制字符）提供 2^40 ≈ 1 万亿种可能，碰撞概率极低
//
// 参数：
//   - claimUID：ResourceClaim 的唯一标识符
//   - devices：提供 UUID 列表的设备集合
//
// 返回值：
//   - string：稳定的守护进程 ID，格式为 "<claimUID>-<hashPrefix5>"
func (m *MpsManager) GetMpsControlDaemonID(claimUID string, devices UUIDProvider) string {
	// 将排序后的 UUID 列表用逗号连接为单一字符串
	combined := strings.Join(devices.UUIDs(), ",")
	// 计算 SHA-256 哈希
	hash := sha256.Sum256([]byte(combined))
	// 取前 5 字节（10 个十六进制字符）作为短哈希后缀
	return fmt.Sprintf("%s-%s", claimUID, hex.EncodeToString(hash[:])[:5])
}

// IsControlDaemonStarted 检查指定 ID 的 MPS 控制守护进程 Deployment 是否存在（已启动）。
// 不关注 Deployment 是否 Ready，仅检查对象是否存在。
// 这是因为 Start() 只需确保 Deployment 已创建，就绪状态由 AssertReady() 单独检查。
//
// 参数：
//   - ctx：上下文，用于控制请求超时和取消
//   - id：守护进程的唯一标识符
//
// 返回值：
//   - bool：true 表示 Deployment 已存在
//   - error：查询 Kubernetes API 失败时返回错误
func (m *MpsManager) IsControlDaemonStarted(ctx context.Context, id string) (bool, error) {
	// 构造 Deployment 名称
	name := fmt.Sprintf(MpsControlDaemonNameFmt, id)
	// 查询 Kubernetes API
	_, err := m.config.clientsets.Core.AppsV1().Deployments(m.config.flags.namespace).Get(ctx, name, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		// Deployment 不存在，表示尚未启动
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("failed to get deployment: %w", err)
	}
	// Deployment 存在，表示已启动
	return true, nil
}

// IsControlDaemonStopped 检查指定 ID 的 MPS 控制守护进程 Deployment 是否已停止（不存在）。
// 与 IsControlDaemonStarted 逻辑互补，不存在返回 true。
// 用于 Stop() 方法中判断是否需要执行清理操作。
//
// 参数：
//   - ctx：上下文，用于控制请求超时和取消
//   - id：守护进程的唯一标识符
//
// 返回值：
//   - bool：true 表示 Deployment 不存在（已停止）
//   - error：查询 Kubernetes API 失败时返回错误
func (m *MpsManager) IsControlDaemonStopped(ctx context.Context, id string) (bool, error) {
	// 构造 Deployment 名称
	name := fmt.Sprintf(MpsControlDaemonNameFmt, id)
	// 查询 Kubernetes API
	_, err := m.config.clientsets.Core.AppsV1().Deployments(m.config.flags.namespace).Get(ctx, name, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		// Deployment 不存在，表示已停止
		return true, nil
	}
	if err != nil {
		return false, fmt.Errorf("failed to get deployment: %w", err)
	}
	// Deployment 仍然存在，表示未停止
	return false, nil
}

// GetID 返回守护进程的唯一标识符。
// 用于日志输出和查找对应的 Kubernetes Deployment。
//
// 返回值：
//   - string：守护进程 ID，格式为 "<claimUID>-<hashPrefix5>"
func (m *MpsControlDaemon) GetID() string {
	return m.id
}

// Start 启动 MPS 控制守护进程。
//
// 操作流程（幂等：若 Deployment 已存在则直接返回）：
//  1. 检查 Deployment 是否已存在，已存在则幂等返回
//  2. 渲染 MPS 守护进程 Deployment YAML 模板（填入节点名、命名空间、设备 UUID、显存限制等）
//  3. 创建 pipe/shm/log 目录（以 0755 权限创建，确保守护进程和工作负载均可访问）
//  4. 将 shm 目录挂载为 tmpfs（CUDA IPC 共享内存要求），大小为系统总内存的一半
//  5. 将目标 GPU 的计算模式设置为 EXCLUSIVE_PROCESS（MPS 运行的前提条件）
//  6. 向 Kubernetes 创建 Deployment 对象（若已存在则幂等返回）
//
// 注意：EXCLUSIVE_PROCESS 模式下，只有 MPS 服务器进程可以直接使用 GPU，
// 工作负载进程通过 MPS 客户端接口访问 GPU，这是 MPS 架构的基础。
// MPS 服务器持有唯一的 CUDA 上下文，客户端通过共享内存提交 CUDA 请求，
// 服务器协调多个客户端的请求实现并发执行。
//
// 参数：
//   - ctx：上下文，用于控制请求超时和取消
//   - config：MPS 配置，包含活跃线程百分比和显存固定限额等参数
//
// 返回值：
//   - error：任何步骤失败时返回错误
func (m *MpsControlDaemon) Start(ctx context.Context, config *configapi.MpsConfig) error {
	// 第 1 步：检查守护进程是否已启动（幂等性检查）
	isStarted, err := m.manager.IsControlDaemonStarted(ctx, m.id)
	if err != nil {
		return fmt.Errorf("error checking if control daemon already started: %w", err)
	}

	// 已启动则直接返回，避免重复创建
	if isStarted {
		return nil
	}

	klog.Infof("Starting MPS control daemon for '%v', with settings: %+v", m.id, config)

	// 获取设备 UUID 列表（已排序）
	deviceUUIDs := m.devices.UUIDs()

	// 第 2 步：构造模板渲染数据
	templateData := MpsControlDaemonTemplateData{
		NodeName:                        m.nodeName,
		MpsControlDaemonNamespace:       m.namespace,
		MpsControlDaemonName:            m.name,
		CUDA_VISIBLE_DEVICES:            strings.Join(deviceUUIDs, ","), // 逗号分隔的 UUID 列表
		DefaultActiveThreadPercentage:   "",
		DefaultPinnedDeviceMemoryLimits: nil,
		NvidiaDriverRoot:                m.manager.hostDriverRoot,
		MpsShmDirectory:                 m.shmDir,
		MpsPipeDirectory:                m.pipeDir,
		MpsLogDirectory:                 m.logDir,
		MpsImageName:                    m.manager.config.flags.imageName,
		FeatureGates:                    featuregates.ToMap(),
	}

	// 填充可选配置：默认活跃线程百分比
	if config != nil && config.DefaultActiveThreadPercentage != nil {
		templateData.DefaultActiveThreadPercentage = fmt.Sprintf("%d", *config.DefaultActiveThreadPercentage)
	}

	// 填充可选配置：按设备 UUID 的显存固定限额
	if config != nil {
		// Normalize() 将 PerDevicePinnedMemoryLimit 和 DefaultPinnedDeviceMemoryLimit
		// 合并为按 UUID 的映射，统一处理默认值和设备级覆盖
		limits, err := config.DefaultPerDevicePinnedMemoryLimit.Normalize(deviceUUIDs, config.DefaultPinnedDeviceMemoryLimit)
		if err != nil {
			return fmt.Errorf("error transforming DefaultPerDevicePinnedMemoryLimit into string: %w", err)
		}
		templateData.DefaultPinnedDeviceMemoryLimits = limits
	}

	// 渲染 YAML 模板
	tmpl, err := template.ParseFiles(m.manager.templatePath)
	if err != nil {
		return fmt.Errorf("failed to parse template file: %w", err)
	}

	var deploymentYaml bytes.Buffer
	if err := tmpl.Execute(&deploymentYaml, templateData); err != nil {
		return fmt.Errorf("failed to execute template: %w", err)
	}

	// 将 YAML 反序列化为 Unstructured 对象（通用 Kubernetes 对象）
	var unstructuredObj unstructured.Unstructured
	err = yaml.Unmarshal(deploymentYaml.Bytes(), &unstructuredObj)
	if err != nil {
		return fmt.Errorf("failed to unmarshal yaml: %w", err)
	}

	// 将 Unstructured 对象转换为类型化的 Deployment 对象
	var deployment appsv1.Deployment
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredObj.UnstructuredContent(), &deployment)
	if err != nil {
		return fmt.Errorf("failed to convert unstructured data to typed object: %w", err)
	}

	// 第 3 步：创建本地工作目录（pipe/shm/log）
	// 0755 权限：所有者可读写执行，组和其他用户可读执行
	err = os.MkdirAll(m.shmDir, 0755)
	if err != nil {
		return fmt.Errorf("error creating directory %v: %w", m.shmDir, err)
	}

	err = os.MkdirAll(m.pipeDir, 0755)
	if err != nil {
		return fmt.Errorf("error creating directory %v: %w", m.pipeDir, err)
	}

	err = os.MkdirAll(m.logDir, 0755)
	if err != nil {
		return fmt.Errorf("error creating directory %v: %w", m.logDir, err)
	}

	// 第 4 步：查找 mount 可执行文件
	mountExecutable, err := exec.LookPath("mount")
	if err != nil {
		return fmt.Errorf("error finding 'mount' executable: %w", err)
	}

	// 将 shm 目录挂载为 tmpfs，大小为系统总内存的一半。
	// CUDA IPC 通过 /dev/shm（或绑定挂载的等效路径）传递共享内存句柄，
	// MPS 的 shm 目录绑定挂载到容器内的 /dev/shm，使容器可通过标准路径访问共享内存。
	mounter := mount.New(mountExecutable)
	// 计算默认 shm 大小（系统内存的一半）
	sizeArg := fmt.Sprintf("size=%v", getDefaultShmSize())
	// tmpfs 挂载选项：读写、无 suid、无 device 文件、不可执行、相对时间、大小限制
	mountOptions := []string{"rw", "nosuid", "nodev", "noexec", "relatime", sizeArg}
	err = mounter.Mount("shm", m.shmDir, "tmpfs", mountOptions)
	if err != nil {
		return fmt.Errorf("error mounting %v as tmpfs: %w", m.shmDir, err)
	}

	// 第 5 步：将 GPU 计算模式设置为 EXCLUSIVE_PROCESS
	// MPS 要求 GPU 处于 EXCLUSIVE_PROCESS 计算模式，
	// 此模式下只有一个 CUDA 上下文（MPS 服务器持有），其他进程通过 MPS 客户端接入。
	// 注意：这里使用 GpuUUIDs() 而非 UUIDs()，因为 MIG 设备的 UUID 不支持设置计算模式
	err = m.manager.nvdevlib.setComputeMode(m.devices.GpuUUIDs(), "EXCLUSIVE_PROCESS")
	if err != nil {
		return fmt.Errorf("error setting compute mode: %w", err)
	}

	// 第 6 步：向 Kubernetes 创建 Deployment
	_, err = m.manager.config.clientsets.Core.AppsV1().Deployments(m.namespace).Create(ctx, &deployment, metav1.CreateOptions{})
	if errors.IsAlreadyExists(err) {
		// Deployment 已存在（可能是并发创建），视为幂等成功
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to create deployment: %w", err)
	}

	return nil
}

// AssertReady 等待 MPS 控制守护进程 Pod 进入 Ready 状态。
//
// 使用指数退避重试策略（初始 1s，步进 2，抖动 1s，最大 10s，重试 4 次），
// 检查 Deployment 的 ReadyReplicas 数量，并验证对应 Pod 的容器状态。
// 用于确保 MPS 控制守护进程在工作负载 Pod 启动之前已完全就绪。
//
// 退避参数设计思路：
//   - 初始 1s：给 Pod 足够的启动时间
//   - 步进 2：每次等待时间翻倍，快速增加等待间隔
//   - 抖动 1s：避免多个守护进程同时重试造成 API 服务器压力
//   - 最大 10s：单次等待不超过 10 秒
//   - 4 步：总共约 1+2+4+8=15 秒的等待窗口
//
// 若 4 次重试后守护进程仍未就绪，返回错误，调用方可选择忽略或重试。
//
// 参数：
//   - ctx：上下文，用于控制请求超时和取消
//
// 返回值：
//   - error：守护进程未就绪或查询 Kubernetes API 失败时返回错误
func (m *MpsControlDaemon) AssertReady(ctx context.Context) error {
	// 配置指数退避参数
	backoff := wait.Backoff{
		Duration: time.Second, // 初始等待时间
		Factor:   2,          // 每次等待时间乘以此因子
		Jitter:   1,          // 添加最多 1 秒的随机抖动
		Steps:    4,          // 最多重试 4 次
		Cap:      10 * time.Second, // 单次最大等待时间
	}

	return retry.OnError(
		backoff,
		// 始终重试（所有错误都可重试）
		func(error) bool {
			return true
		},
		func() error {
			// 查询 Deployment 状态
			deployment, err := m.manager.config.clientsets.Core.AppsV1().Deployments(m.namespace).Get(
				ctx,
				m.name,
				metav1.GetOptions{},
			)
			if err != nil {
				return fmt.Errorf("failed to get deployment: %w", err)
			}

			// 检查 Ready 副本数是否为 1
			if deployment.Status.ReadyReplicas != 1 {
				return fmt.Errorf("waiting for MPS control daemon to come online")
			}

			// 获取 Deployment 的标签选择器
			selector := deployment.Spec.Selector.MatchLabels

			// 列出匹配的 Pod
			pods, err := m.manager.config.clientsets.Core.CoreV1().Pods(m.namespace).List(
				ctx,
				metav1.ListOptions{
					LabelSelector: labels.Set(selector).AsSelector().String(),
				},
			)
			if err != nil {
				return fmt.Errorf("error listing pods from deployment")
			}

			// 确认只有一个 Pod
			if len(pods.Items) != 1 {
				return fmt.Errorf("unexpected number of pods in deployment: %v", len(pods.Items))
			}

			// 确认只有一个容器状态
			if len(pods.Items[0].Status.ContainerStatuses) != 1 {
				return fmt.Errorf("unexpected number of container statuses in pod")
			}

			// 确认容器已就绪
			if !pods.Items[0].Status.ContainerStatuses[0].Ready {
				return fmt.Errorf("control daemon not yet ready")
			}

			return nil
		},
	)
}

// GetCDIContainerEdits 返回工作负载容器接入 MPS 所需的 CDI 编辑项。
//
// MPS 接入需要两项容器配置：
//  1. 环境变量 CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps：
//     告知 CUDA MPS 客户端通过哪个目录与 MPS 服务器通信（pipe 方式）
//     MPS 服务器在此目录创建命名管道，客户端连接管道提交 CUDA 请求
//  2. 绑定挂载：
//     - /dev/shm → shmDir：MPS 共享内存（CUDA IPC 用，替换容器默认的 /dev/shm）
//       替换而非追加的原因：CUDA IPC 依赖 /dev/shm 路径，必须确保该路径指向 MPS 的共享内存
//     - /tmp/nvidia-mps → pipeDir：MPS pipe 目录（工作负载通过 pipe 发送 CUDA 请求）
//       工作负载容器内的 CUDA MPS 客户端通过此目录的管道与 MPS 服务器通信
//
// 这些编辑项附加到设备组的 ConfigState，最终合并到 Claim 级别的 CDI spec 文件中。
//
// 返回值：
//   - *cdiapi.ContainerEdits：包含环境变量和挂载点的容器编辑项
func (m *MpsControlDaemon) GetCDIContainerEdits() *cdiapi.ContainerEdits {
	return &cdiapi.ContainerEdits{
		ContainerEdits: &cdispec.ContainerEdits{
			Env: []string{
				// 设置 MPS 管道目录环境变量
				fmt.Sprintf("CUDA_MPS_PIPE_DIRECTORY=%s", "/tmp/nvidia-mps"),
			},
			Mounts: []*cdispec.Mount{
				{
					// 替换容器默认 /dev/shm 为 MPS shm 目录
					ContainerPath: "/dev/shm",
					HostPath:      m.shmDir,
					// rw: 读写权限；nosuid: 忽略 suid 位；nodev: 不解释设备文件；bind: 绑定挂载
					Options:       []string{"rw", "nosuid", "nodev", "bind"},
				},
				{
					// 将 MPS pipe 目录挂载到容器内标准路径
					ContainerPath: "/tmp/nvidia-mps",
					HostPath:      m.pipeDir,
					Options:       []string{"rw", "nosuid", "nodev", "bind"},
				},
			},
		},
	}
}

// Stop 停止并清理 MPS 控制守护进程。
//
// 操作流程：
//  1. 若工作目录不存在，视为已停止（幂等返回）
//  2. 以 Foreground 传播策略删除 Kubernetes Deployment（等待 Pod 真正终止后返回）
//  3. 卸载 shm tmpfs 挂载点（CleanupMountPoint 自动判断是否已挂载）
//  4. 递归删除守护进程工作目录（pipe/shm/log）
//
// 注意：若 Deployment 不存在（IsNotFound），跳过删除步骤（幂等）。
// 这处理了以下场景：守护进程从未启动、守护进程已被外部删除、
// 或在程序崩溃重启后的清理。
//
// 参数：
//   - ctx：上下文，用于控制请求超时和取消
//
// 返回值：
//   - error：任何步骤失败时返回错误
func (m *MpsControlDaemon) Stop(ctx context.Context) error {
	// 第 1 步：检查工作目录是否存在，不存在则视为已停止
	_, err := os.Stat(m.rootDir)
	if os.IsNotExist(err) {
		return nil
	}

	klog.Infof("Stopping MPS control daemon for '%v'", m.id)

	// 第 2 步：删除 Kubernetes Deployment
	// Foreground 传播策略：Kubernetes 等待所有 Pod 删除后才将 Deployment 标记为已删除。
	// 使用 Foreground 而非 Background（默认）的原因：
	// 我们需要确认工作负载 Pod 已真正停止使用 GPU 后，才能安全地清理本地文件系统资源。
	// 如果使用 Background 传播，Deployment 对象可能已删除，但 Pod 仍在运行并使用 GPU，
	// 此时清理 shm/pipe 会导致 Pod 中的 CUDA 操作失败。
	deletePolicy := metav1.DeletePropagationForeground
	deleteOptions := metav1.DeleteOptions{
		PropagationPolicy: &deletePolicy,
	}

	err = m.manager.config.clientsets.Core.AppsV1().Deployments(m.namespace).Delete(ctx, m.name, deleteOptions)
	if err != nil && !errors.IsNotFound(err) {
		// Deployment 不存在视为正常（幂等性），其他错误向上传递
		return fmt.Errorf("failed to delete deployment: %w", err)
	}

	// 第 3 步：查找 mount 可执行文件
	mountExecutable, err := exec.LookPath("mount")
	if err != nil {
		return fmt.Errorf("error finding 'mount' executable: %w", err)
	}

	// 卸载 shm tmpfs 挂载点
	// CleanupMountPoint 会：1) 检查是否是挂载点；2) 执行 umount；3) 删除目录
	// 第三个参数 true 表示如果卸载成功则删除目录
	mounter := mount.New(mountExecutable)
	err = mount.CleanupMountPoint(m.shmDir, mounter, true)
	if err != nil {
		return fmt.Errorf("error unmounting %v: %w", m.shmDir, err)
	}

	// 第 4 步：递归删除守护进程工作目录（包含 pipe、log 和已清空的 shm 子目录）
	err = os.RemoveAll(m.rootDir)
	if err != nil {
		return fmt.Errorf("error removing directory %v: %w", m.rootDir, err)
	}

	return nil
}

// getDefaultShmSize 计算 MPS shm tmpfs 的默认大小，取系统总内存的一半。
//
// 实现方式：读取 /proc/meminfo 中的 MemTotal 字段，除以 2 得到大小。
// 若读取失败（文件不存在、解析错误），回退到 65536k（64 MiB）。
// 单位跟随 /proc/meminfo 报告的单位（通常为 kB），避免单位转换错误。
// 注意：Linux 内核中 kB 实际表示 1024 字节（KiB），而非 1000 字节。
//
// 选择系统内存一半的原因：
//   - CUDA IPC 共享内存通常与 GPU 显存大小相关
//   - 一半系统内存是经验值，确保大多数工作负载的共享内存分配不会失败
//   - 不使用全部内存是为了给系统和其他进程留出足够空间
//
// 返回值：
//   - string：格式为 "<数量><单位>" 的 shm 大小字符串（如 "16384000kB"），或回退值 "65536k"
func getDefaultShmSize() string {
	// 回退值：64 MiB（65536 * 1024 字节）
	const fallbackSize = "65536k"

	// 打开 /proc/meminfo 文件
	meminfo, err := os.Open("/proc/meminfo")
	if err != nil {
		klog.ErrorS(err, "failed to open /proc/meminfo")
		return fallbackSize
	}
	defer func() {
		_ = meminfo.Close()
	}()

	// 逐行扫描查找 MemTotal 行
	scanner := bufio.NewScanner(meminfo)
	for scanner.Scan() {
		line := scanner.Text()
		// 跳过非 MemTotal 行
		if !strings.HasPrefix(line, "MemTotal:") {
			continue
		}

		// 解析 MemTotal 值和单位
		// 格式示例："MemTotal:       32768000 kB"
		parts := strings.SplitN(strings.TrimSpace(strings.TrimPrefix(line, "MemTotal:")), " ", 2)
		// 解析数值部分
		memTotal, err := strconv.Atoi(parts[0])
		if err != nil {
			klog.ErrorS(err, "could not convert MemTotal to an integer")
			return fallbackSize
		}

		// 提取单位（取单位的第一个字符，如 "kB" → "k"）
		var unit string
		if len(parts) == 2 {
			unit = string(parts[1][0])
		}

		// 返回系统内存的一半，保持原始单位
		return fmt.Sprintf("%d%s", memTotal/2, unit)
	}
	// 未找到 MemTotal 行，使用回退值
	return fallbackSize
}
