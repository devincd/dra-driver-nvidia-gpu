// health.go 实现了 DRA 驱动插件的 gRPC 健康检查服务。
//
// 本文件实现了 gRPC Health Checking Protocol（grpc_health_v1.HealthServer），
// 为 Kubernetes 提供对 DRA 驱动插件的健康探测能力。
//
// 工作原理：
//  1. 监听 TCP 端口（由 --healthcheck-port 配置），对外暴露 gRPC 健康检查服务。
//  2. 收到健康检查请求时，向两个 Unix domain socket 发送探测请求：
//     a. 注册 socket（<driver>-reg.sock）：发送 GetInfo 请求，验证插件已向 kubelet 注册
//     b. DRA socket（dra.sock）：发送空的 NodePrepareResources 请求，验证 DRA gRPC 服务可响应
//  3. 两步探测均成功则返回 SERVING（健康），任意失败则返回 NOT_SERVING（不健康）。
//
// 使用场景：作为 Kubernetes Pod 的 liveness/readiness probe，
// 使 kubelet 能够感知 DRA 驱动插件的健康状态并在必要时重启 Pod。
package main

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"path"
	"strconv"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"k8s.io/dynamic-resource-allocation/kubeletplugin"
	"k8s.io/klog/v2"
	drapb "k8s.io/kubelet/pkg/apis/dra/v1beta1"
	registerapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
)

// healthcheck 实现了 gRPC Health Checking Protocol（grpc_health_v1.HealthServer）。
//
// 工作原理：
//  1. 监听 TCP 端口（由 --healthcheck-port 配置），对外暴露 gRPC 健康检查服务。
//  2. 收到健康检查请求时，向两个 Unix domain socket 发送探测请求：
//     a. 注册 socket（<driver>-reg.sock）：发送 GetInfo 请求，验证插件已向 kubelet 注册
//     b. DRA socket（dra.sock）：发送空的 NodePrepareResources 请求，验证 DRA gRPC 服务可响应
//  3. 两步探测均成功则返回 SERVING（健康），任意失败则返回 NOT_SERVING（不健康）。
//
// 使用场景：作为 Kubernetes Pod 的 liveness/readiness probe，
// 使 kubelet 能够感知 DRA 驱动插件的健康状态并在必要时重启 Pod。
type healthcheck struct {
	// UnimplementedHealthServer 内嵌实现了 HealthServer 接口中除 Check 外的所有方法（返回 Unimplemented），
	// 避免实现完整接口而不得不写大量空方法。
	grpc_health_v1.UnimplementedHealthServer

	// server 是健康检查的 gRPC 服务实例，监听 TCP 端口。
	server *grpc.Server

	// wg 用于优雅关闭：等待 Serve goroutine 退出后 Stop() 才返回。
	wg sync.WaitGroup

	// kphelper 用于在 Check() 日志中打印当前的 kubelet 插件注册状态，辅助调试。
	kphelper *kubeletplugin.Helper

	// regClient 向注册 socket（<driver>-reg.sock）发送 GetInfo 探测请求。
	// GetInfo 是 kubelet 在注册握手期间向插件发送的第一个 RPC，
	// 用于验证插件类型和版本信息，成功响应表明注册 socket 可用。
	regClient registerapi.RegistrationClient

	// draClient 向 DRA socket（dra.sock）发送空的 NodePrepareResources 探测请求。
	// 发送空的 Claim 列表，不触发实际设备操作，仅验证 DRA gRPC 服务可正常响应。
	// 空请求在 PrepareResourceClaims() 中会被记录为 V(7) 级别日志，不产生业务副作用。
	draClient drapb.DRAPluginClient
}

// startHealthcheck 启动 gRPC 健康检查服务。
//
// 若端口配置为负数（默认值 -1），则禁用健康检查服务，返回 nil, nil。
// 这允许在不需要健康检查的环境（如测试、单机部署）中免去额外的 TCP 监听。
//
// 初始化流程：
//  1. 在指定 TCP 端口监听
//  2. 建立到注册 socket 的 gRPC 连接（用于 GetInfo 探测）
//  3. 建立到 DRA socket 的 gRPC 连接（用于 NodePrepareResources 探测）
//  4. 创建 gRPC 服务并注册 HealthServer 实现
//  5. 启动 goroutine 开始服务
func startHealthcheck(ctx context.Context, config *Config, helper *kubeletplugin.Helper) (*healthcheck, error) {
	// 读取健康检查端口配置
	port := config.flags.healthcheckPort
	// 端口为负数表示禁用健康检查功能
	if port < 0 {
		return nil, nil
	}

	// 构建监听地址（绑定所有网络接口的指定端口）
	addr := net.JoinHostPort("", strconv.Itoa(port))
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("failed to listen for healthcheck service at %s: %w", addr, err)
	}

	// 构建注册 socket 的 Unix domain socket URL。
	// 格式：unix:///var/lib/kubelet/plugins_registry/gpu.nvidia.com-reg.sock
	// 注意：seamless upgrades 启用时，文件名中可能包含 UID 后缀，此处暂未适配。
	regSockPath := (&url.URL{
		Scheme: "unix",
		Path:   path.Join(config.flags.kubeletRegistrarDirectoryPath, DriverName+"-reg.sock"),
	}).String()
	klog.V(6).Infof("connecting to registration socket path=%s", regSockPath)
	// 建立到注册 socket 的 gRPC 连接（不加密，因为是本地 Unix socket）
	regConn, err := grpc.NewClient(
		regSockPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("connect to registration socket: %w", err)
	}

	// 构建 DRA socket 的 Unix domain socket URL。
	// 格式：unix:///var/lib/kubelet/plugins/gpu.nvidia.com/dra.sock
	draSockPath := (&url.URL{
		Scheme: "unix",
		Path:   path.Join(config.DriverPluginPath(), "dra.sock"),
	}).String()
	klog.V(6).Infof("connecting to DRA socket path=%s", draSockPath)
	// 建立到 DRA socket 的 gRPC 连接（不加密，因为是本地 Unix socket）
	draConn, err := grpc.NewClient(
		draSockPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return nil, fmt.Errorf("connect to DRA socket: %w", err)
	}

	// 创建 gRPC 服务实例
	server := grpc.NewServer()
	// 构建健康检查结构体，初始化两个探测客户端
	healthcheck := &healthcheck{
		server:    server,
		regClient: registerapi.NewRegistrationClient(regConn),
		draClient: drapb.NewDRAPluginClient(draConn),
		kphelper:  helper,
	}
	// 将健康检查服务注册到 gRPC 服务
	grpc_health_v1.RegisterHealthServer(server, healthcheck)

	// 启动 goroutine 在监听器上提供 gRPC 服务
	healthcheck.wg.Add(1)
	go func() {
		defer healthcheck.wg.Done()
		klog.Infof("starting healthcheck service at %s", lis.Addr().String())
		// Serve 会阻塞直到服务停止或监听器关闭
		if err := server.Serve(lis); err != nil {
			klog.Errorf("failed to serve healthcheck service on %s: %v", addr, err)
		}
	}()

	return healthcheck, nil
}

// Stop 优雅停止健康检查服务。
// 调用 GracefulStop 等待正在处理的请求完成后再关闭服务，而非强制中断。
// 通过 wg.Wait 确保 Serve goroutine 完全退出后 Stop 才返回，避免资源泄漏。
func (h *healthcheck) Stop() {
	// 仅在 server 已初始化时执行停止操作
	if h.server != nil {
		klog.Info("Stopping healthcheck service")
		// 优雅停止：等待进行中的 RPC 请求完成后关闭
		// 与 Stop() 不同，GracefulStop 不会中断正在处理的请求
		h.server.GracefulStop()
	}
	// 等待 Serve goroutine 完全退出
	h.wg.Wait()
}

// Check 实现 grpc_health_v1.HealthServer 接口，是健康检查的核心逻辑。
//
// 检查流程（两步探测）：
//  1. 校验 service 名称：只接受 "" 和 "liveness"，其他返回 NotFound 错误。
//  2. 向注册 socket 发送 GetInfo 请求：
//     - 成功：注册 socket 存在且插件已完成注册握手
//     - 失败：返回 NOT_SERVING（插件可能还未注册或注册 socket 已关闭）
//  3. 向 DRA socket 发送空的 NodePrepareResources 请求：
//     - 成功：DRA gRPC 服务可正常响应
//     - 失败：返回 NOT_SERVING（DRA 服务可能尚未启动）
//  4. 两步均成功：返回 SERVING
//
// 为何返回 NOT_SERVING 而不是 error？
// Kubernetes probe 语义要求：unhealthy 不是"错误"，而是一种合法的健康状态。
// 返回 error 会使 probe 判定为探测失败（与不健康不同），触发不同的处理逻辑。
func (h *healthcheck) Check(ctx context.Context, req *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	// 定义允许的 service 名称集合
	// 空字符串 "" 表示整体健康状态，"liveness" 专门用于存活探测
	knownServices := map[string]struct{}{"": {}, "liveness": {}}
	// 如果请求的 service 名称不在允许集合中，返回 NotFound 错误
	if _, known := knownServices[req.GetService()]; !known {
		return nil, status.Error(codes.NotFound, "unknown service")
	}

	// 默认响应为 NOT_SERVING（不健康），只有在两步探测都成功后才改为 SERVING
	status := &grpc_health_v1.HealthCheckResponse{
		Status: grpc_health_v1.HealthCheckResponse_NOT_SERVING,
	}

	// 第一步：探测注册 socket，验证 kubelet 插件注册握手已完成。
	// GetInfo 请求会返回插件的类型和版本信息，
	// 成功响应表明插件已正确注册到 kubelet 的插件注册表。
	info, err := h.regClient.GetInfo(ctx, &registerapi.InfoRequest{})
	if err != nil {
		// 注册 socket 不可达，说明插件注册失败或 socket 已关闭
		klog.ErrorS(err, "failed to call GetInfo")
		return status, nil
	}
	klog.V(7).Infof("Health check: successfully invoked GetInfo: %v", info)

	// 第二步：探测 DRA socket，发送空请求验证 DRA gRPC 服务可响应。
	// 空的 NodePrepareResourcesRequest（Claim 列表为空）不会触发实际设备操作，
	// 在 PrepareResourceClaims() 实现中会被记录为 V(7) 调试日志并立即返回空结果。
	_, err = h.draClient.NodePrepareResources(ctx, &drapb.NodePrepareResourcesRequest{})
	if err != nil {
		// DRA socket 不可达，说明 DRA gRPC 服务可能尚未启动或已崩溃
		klog.ErrorS(err, "failed to call NodePrepareResources")
		return status, nil
	}

	// 两步探测均成功，记录成功日志
	klog.V(7).Info("Health check: success: got NodePrepareResourcesResponse for noop request")
	// 记录当前 kubelet 插件注册状态，辅助调试
	klog.V(6).Infof("Current kubelet plugin registration status: %s", h.kphelper.RegistrationStatus())

	// 两步均成功，将状态更新为 SERVING（健康）
	status.Status = grpc_health_v1.HealthCheckResponse_SERVING
	return status, nil
}
