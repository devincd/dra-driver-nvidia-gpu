/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

// vfio-device 包实现了 GPU 通过 vfio-pci 驱动直通（Passthrough）给虚拟机或容器的完整管理逻辑。
//
// 核心组件 VfioPciManager 负责：
//   - 在插件启动时加载 vfio_pci 内核模块并验证 IOMMU 支持
//   - 在 Prepare 阶段将 GPU 从 nvidia 驱动切换到 vfio-pci 驱动（Configure）
//   - 在 Unprepare 阶段将 GPU 从 vfio-pci 驱动切换回 nvidia 驱动（Unconfigure）
//   - 等待 GPU 设备节点空闲（无进程持有文件描述符）后再执行驱动切换
//   - 生成 CDI 容器编辑项，使容器能够访问 /dev/vfio/<iommuGroup> 设备节点
//
// 驱动切换通过在宿主机 chroot 环境中执行 unbind_from_driver.sh / bind_to_driver.sh 脚本完成。
// 使用 perGpuLock 按 PCIe Bus ID 加锁，防止同一 GPU 并发切换驱动引发竞争条件。
//
// 仅在 PassthroughSupport 特性门控开启时创建和使用 VfioPciManager。
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"k8s.io/klog/v2"
	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"
	cdispec "tags.cncf.io/container-device-interface/specs-go"
)

// 以下常量定义了 vfio-pci 驱动管理所需的系统路径和配置参数
const (
	// kernelIommuGroupPath 是内核 IOMMU 组的 sysfs 路径，用于检查 IOMMU 是否启用
	kernelIommuGroupPath = "/sys/kernel/iommu_groups"
	// vfioPciModule 是 vfio_pci 内核模块的名称，用于 modprobe 加载
	vfioPciModule = "vfio_pci"
	// vfioPciDriver 是 vfio-pci 驱动的名称，用于设备绑定
	vfioPciDriver = "vfio-pci"
	// nvidiaDriver 是 NVIDIA 驱动的名称，用于驱动切换的判断
	nvidiaDriver = "nvidia"
	// hostRoot 是容器内挂载的宿主机根文件系统路径，用于 chroot 执行命令
	hostRoot = "/host-root"
	// sysModulesRoot 是内核模块的 sysfs 路径，用于检查模块是否已加载
	sysModulesRoot = "/sys/module"
	// pciDevicesRoot 是 PCI 设备的 sysfs 路径，用于读取设备驱动绑定信息
	pciDevicesRoot = "/sys/bus/pci/devices"
	// vfioDevicesRoot 是 vfio 设备节点的路径，用于 CDI 注入容器
	vfioDevicesRoot = "/dev/vfio"
	// unbindFromDriverScript 是解绑 GPU 驱动的脚本路径
	unbindFromDriverScript = "/usr/bin/unbind_from_driver.sh"
	// bindToDriverScript 是绑定 GPU 驱动的脚本路径
	bindToDriverScript = "/usr/bin/bind_to_driver.sh"
	// driverResetRetries 是驱动重置重试次数（传递给脚本参数）
	driverResetRetries = "5"
	// gpuFreeCheckInterval 是检查 GPU 是否空闲的轮询间隔
	gpuFreeCheckInterval = 1 * time.Second
	// gpuFreeCheckTimeout 是等待 GPU 空闲的超时时间，超时后返回错误
	gpuFreeCheckTimeout = 60 * time.Second
)

// VfioPciManager 管理 GPU 在 nvidia 驱动和 vfio-pci 驱动之间的切换（直通模式）。
// 工作原理：通过 unbind_from_driver.sh / bind_to_driver.sh 脚本（在宿主机 chroot 环境执行）
// 将 GPU 的 PCI 设备重新绑定到 vfio-pci 驱动，使 VM 或容器可以直通访问硬件。
// 仅在 PassthroughSupport 特性门控开启时创建和使用。
type VfioPciManager struct {
	// containerDriverRoot 是容器内 NVIDIA 驱动根目录路径（用于 CDI spec 生成）
	containerDriverRoot string
	// hostDriverRoot 是宿主机上 NVIDIA 驱动根目录路径（用于设备节点访问）
	hostDriverRoot string
	// driver 是目标驱动名，通常为 "vfio-pci"
	driver string
	// nvlib 是 NVML 设备库封装，提供 GPU PCI 信息查询能力
	nvlib *deviceLib
	// nvidiaEnabled 标识是否允许 GPU 当前绑定在 nvidia 驱动（是则可以切换到 vfio-pci）
	// 若为 false，表示 GPU 只能从 vfio-pci 切回 nvidia，不支持反向操作
	nvidiaEnabled bool
}

// NewVfioPciManager 创建 VfioPciManager 实例。
// 在创建时自动检查 vfio_pci 模块是否已加载，若未加载则尝试 modprobe 加载。
// 若加载失败则直接 fatal 退出（因为没有 vfio_pci 模块，直通功能完全不可用）。
//
// 参数：
//   - containerDriverRoot: 容器内 NVIDIA 驱动根路径
//   - hostDriverRoot: 宿主机 NVIDIA 驱动根路径
//   - nvlib: NVML 设备库封装实例
//   - nvidiaEnabled: 是否允许 GPU 从 nvidia 驱动切换到 vfio-pci
func NewVfioPciManager(containerDriverRoot string, hostDriverRoot string, nvlib *deviceLib, nvidiaEnabled bool) *VfioPciManager {
	vm := &VfioPciManager{
		containerDriverRoot: containerDriverRoot,
		hostDriverRoot:      hostDriverRoot,
		driver:              vfioPciDriver,
		nvlib:               nvlib,
		nvidiaEnabled:       nvidiaEnabled,
	}
	// 检查 vfio_pci 内核模块是否已加载
	if !vm.isVfioPCIModuleLoaded() {
		// 未加载则尝试通过 modprobe 加载
		err := vm.loadVfioPciModule()
		if err != nil {
			// 加载失败是致命错误：没有 vfio_pci 模块，GPU 直通功能完全不可用
			klog.Fatalf("failed to load vfio_pci module: %v", err)
		}
	}

	return vm
}

// ValidatePassthroughSupport 验证 vfio-pci 设备分配是否可用。
// 检查两个前提条件：
//  1. vfio_pci 内核模块已加载
//  2. IOMMU 在内核中已启用（IOMMU 是设备直通的安全基础，防止 DMA 攻击）
//
// 返回值：
//   - nil: 直通功能可用
//   - error: 任一前提条件不满足时的具体错误
func (vm *VfioPciManager) ValidatePassthroughSupport() error {
	// 检查 vfio_pci 模块是否加载
	if !vm.isVfioPCIModuleLoaded() {
		return fmt.Errorf("vfio_pci module is not loaded")
	}
	// 检查 IOMMU 是否启用
	iommuEnabled, err := vm.isIommuEnabled()
	if err != nil {
		return err
	}
	if !iommuEnabled {
		return fmt.Errorf("IOMMU is not enabled in the kernel")
	}
	return nil
}

// isIommuEnabled 检查内核 IOMMU 是否已启用。
// 通过读取 /sys/kernel/iommu_groups 目录来判断：
//   - 目录不存在 → IOMMU 未启用
//   - 目录存在但为空 → IOMMU 未启用
//   - 目录存在且包含条目 → IOMMU 已启用
//
// 返回值：
//   - bool: IOMMU 是否启用
//   - error: 文件系统访问错误（IsNotExist 除外，该情况返回 false, nil）
func (vm *VfioPciManager) isIommuEnabled() (bool, error) {
	// 尝试打开 IOMMU 组目录
	f, err := os.Open(kernelIommuGroupPath)
	if os.IsNotExist(err) {
		// 目录不存在说明内核未启用 IOMMU
		return false, nil
	}
	if err != nil {
		// 其他错误（如权限不足）向上传播
		return false, err
	}
	defer f.Close()

	// 尝试读取目录中的第一个条目名
	_, err = f.Readdirnames(1)
	if err == io.EOF {
		// 目录为空说明内核启用了 IOMMU 支持但无 IOMMU 组（通常也是未启用）
		return false, nil
	}
	if err != nil {
		return false, err
	}

	// 目录中存在条目，IOMMU 已启用
	return true, nil
}

// isVfioPCIModuleLoaded 检查 vfio_pci 内核模块是否已加载。
// 通过检查 /sys/module/vfio_pci 目录是否存在且为目录来判断。
//
// 返回值：
//   - true: 模块已加载
//   - false: 模块未加载或路径异常
func (vm *VfioPciManager) isVfioPCIModuleLoaded() bool {
	// 检查 /sys/module/vfio_pci 路径
	f, err := os.Stat(filepath.Join(sysModulesRoot, vfioPciModule))
	if err != nil {
		if os.IsNotExist(err) {
			// 路径不存在 → 模块未加载
			return false
		}
		// 其他错误（如权限问题）视为致命错误，因为无法确定模块状态
		klog.Fatalf("Failed to check if vfio_pci module is loaded: %v", err)
	}

	if !f.IsDir() {
		// 路径存在但不是目录（理论上不应发生），视为未加载
		return false
	}

	return true
}

// loadVfioPciModule 通过 modprobe 命令加载 vfio_pci 内核模块。
// 在宿主机 chroot 环境中执行 modprobe，因为模块需要加载到宿主机内核中。
//
// 返回值：
//   - nil: 加载成功
//   - error: modprobe 命令执行失败
func (vm *VfioPciManager) loadVfioPciModule() error {
	// 在宿主机 chroot 环境中执行 modprobe vfio_pci
	_, err := execCommandWithChroot(hostRoot, "modprobe", []string{vfioPciModule}) //nolint:gosec
	if err != nil {
		return err
	}

	return nil
}

// WaitForGPUFree 等待 GPU 设备节点空闲（无进程持有文件描述符）。
// 在执行驱动切换前调用，确保没有进程正在使用该 GPU，否则切换会导致进程崩溃。
//
// 实现方式：周期性执行 `fuser <device_node>` 命令检测是否有进程持有设备节点。
//   - fuser 返回 exit code 1 → 无进程持有，GPU 空闲，返回 nil
//   - fuser 返回 exit code 0 → 有进程持有，继续等待
//   - fuser 返回其他错误 → 记录警告日志，继续等待（不立即失败）
//
// 参数：
//   - ctx: context，用于传播取消信号
//   - info: VFIO 设备信息，包含父 GPU 的 minor 号和 PCIe Bus ID
//
// 返回值：
//   - nil: GPU 已空闲
//   - error: 超时等待或 context 被取消
func (vm *VfioPciManager) WaitForGPUFree(ctx context.Context, info *VfioDeviceInfo) error {
	// 若该 VFIO 设备无父 GPU（理论上不应发生），直接返回
	if info.parent == nil {
		return nil
	}

	// 设置超时定时器和轮询定时器
	timeout := time.After(gpuFreeCheckTimeout)
	ticker := time.NewTicker(gpuFreeCheckInterval)
	defer ticker.Stop()

	// 构造 GPU 设备节点路径，如 /host-root/dev/nvidia0
	gpuDeviceNode := filepath.Join(vm.hostDriverRoot, "dev", fmt.Sprintf("nvidia%d", info.parent.minor))
	for {
		select {
		case <-timeout:
			// 超时，GPU 仍被占用，返回错误
			return fmt.Errorf("timed out waiting for gpu to be free")
		case <-ticker.C:
			// 在宿主机 chroot 环境中执行 fuser 命令检测设备节点占用情况
			out, err := execCommandWithChroot(hostRoot, "fuser", []string{gpuDeviceNode}) //nolint:gosec
			if err != nil {
				// fuser 返回非零 exit code
				if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
					// exit code 1 表示无进程持有该设备节点，GPU 已空闲
					return nil
				}
				// 其他错误（如命令不存在）记录警告但不终止，继续轮询
				klog.Errorf("Unexpected error checking if gpu device %q is free: %v", info.pcieBusID, err)
				continue
			}
			// fuser 返回 exit code 0，有进程持有设备节点，记录持有进程信息
			klog.Infof("gpu device %q has open fds by process(es): %s", info.pcieBusID, string(out))
		}
	}
}

// verifyDisabledVFs 验证 GPU 上不存在 SR-IOV 虚拟功能（VF）。
// 当 GPU 启用了 SR-IOV VF 时，不允许解绑驱动，因为 VF 仍然依赖物理功能（PF）的驱动。
// 必须先禁用所有 VF 后才能执行驱动切换。
//
// 参数：
//   - pcieBusID: GPU 的 PCIe 总线地址
//
// 返回值：
//   - nil: 无 VF 或 VF 已禁用
//   - error: 存在活跃的 VF，不允许解绑
func (vm *VfioPciManager) verifyDisabledVFs(pcieBusID string) error {
	// 通过 nvpci 库获取 GPU 的 PCI 信息
	gpu, err := vm.nvlib.nvpci.GetGPUByPciBusID(pcieBusID)
	if err != nil {
		return err
	}
	// 检查物理功能上的 VF 数量
	numVFs := gpu.SriovInfo.PhysicalFunction.NumVFs
	if numVFs > 0 {
		// 有活跃 VF，不允许解绑驱动
		return fmt.Errorf("gpu has %d VFs, cannot unbind", numVFs)
	}
	return nil
}

// Configure 将 GPU 从 nvidia 驱动切换到 vfio-pci 驱动（Prepare 时调用）。
// 完整流程：
//  1. 获取 perGpuLock 按 PCIe Bus ID 加锁，防止同一 GPU 并发切换
//  2. 检查当前驱动，若已是 vfio-pci 则直接返回（幂等）
//  3. 验证当前驱动是否允许切换（nvidiaEnabled 且当前为 nvidia 驱动）
//  4. 等待 GPU 空闲（无进程持有设备节点）
//  5. 验证 GPU 上不存在 SR-IOV VF
//  6. 执行驱动切换（unbind + bind）
//
// 参数：
//   - ctx: context，用于传播取消信号
//   - info: VFIO 设备信息，包含 PCIe Bus ID 和父 GPU 信息
//
// 返回值：
//   - nil: 切换成功或已是目标驱动
//   - error: 任一检查失败或切换出错
func (vm *VfioPciManager) Configure(ctx context.Context, info *VfioDeviceInfo) error {
	// 获取该 GPU 专用的互斥锁，防止并发操作同一 GPU
	perGpuLock.Get(info.pcieBusID).Lock()
	defer perGpuLock.Get(info.pcieBusID).Unlock()

	// 读取当前绑定的驱动
	driver, err := getDriver(pciDevicesRoot, info.pcieBusID)
	if err != nil {
		return err
	}
	// 若已是目标驱动，幂等返回成功
	if driver == vm.driver {
		return nil
	}
	// 仅支持当前绑定在 vfio-pci 或 nvidia 驱动（nvidiaEnabled 为 true 时）上的 GPU
	if !vm.nvidiaEnabled || driver != nvidiaDriver {
		return fmt.Errorf("gpu is bound to %q driver, expected %q or %q", driver, vm.driver, nvidiaDriver)
	}

	// 等待 GPU 空闲，确保没有进程正在使用该设备
	err = vm.WaitForGPUFree(ctx, info)
	if err != nil {
		return err
	}

	// 验证 GPU 上不存在 SR-IOV VF（VF 存在时不允许解绑）
	err = vm.verifyDisabledVFs(info.pcieBusID)
	if err != nil {
		return err
	}

	// 执行驱动切换：先解绑当前驱动，再绑定到目标驱动
	err = vm.changeDriver(info.pcieBusID, vm.driver)
	if err != nil {
		return err
	}
	return nil
}

// Unconfigure 将 GPU 从 vfio-pci 驱动切换回 nvidia 驱动（Unprepare 时调用）。
// 若 nvidiaEnabled 为 false（不支持切回 nvidia 驱动），则直接跳过，不做任何操作。
// 这适用于某些部署场景，GPU 切换到 vfio-pci 后不再需要切回。
//
// 参数：
//   - ctx: context，用于传播取消信号（当前未使用）
//   - info: VFIO 设备信息，包含 PCIe Bus ID
//
// 返回值：
//   - nil: 切换成功、已是目标驱动或不支持切回
//   - error: 驱动切换失败
func (vm *VfioPciManager) Unconfigure(ctx context.Context, info *VfioDeviceInfo) error {
	// 获取该 GPU 专用的互斥锁，防止并发操作同一 GPU
	perGpuLock.Get(info.pcieBusID).Lock()
	defer perGpuLock.Get(info.pcieBusID).Unlock()

	// 若 nvidiaEnabled 为 false，表示部署不支持将 GPU 切回 nvidia 驱动，直接返回
	if !vm.nvidiaEnabled {
		return nil
	}

	// 读取当前绑定的驱动
	driver, err := getDriver(pciDevicesRoot, info.pcieBusID)
	if err != nil {
		return err
	}
	// 若已是 nvidia 驱动，幂等返回成功
	if driver == nvidiaDriver {
		return nil
	}

	// 执行驱动切换：先解绑 vfio-pci，再绑定回 nvidia 驱动
	err = vm.changeDriver(info.pcieBusID, nvidiaDriver)
	if err != nil {
		return err
	}
	return nil
}

// getDriver 读取 PCI 设备当前绑定的驱动名称。
// 通过读取 /sys/bus/pci/devices/<pciAddress>/driver 符号链接的目标路径来获取。
//
// 参数：
//   - pciDevicesRoot: PCI 设备的 sysfs 根路径（如 /sys/bus/pci/devices）
//   - pciAddress: GPU 的 PCIe 总线地址（如 "0000:3b:00.0"）
//
// 返回值：
//   - string: 驱动名称（如 "nvidia"、"vfio-pci"）
//   - error: 读取符号链接失败
func getDriver(pciDevicesRoot, pciAddress string) (string, error) {
	// 读取 driver 符号链接的目标路径
	driverPath, err := os.Readlink(filepath.Join(pciDevicesRoot, pciAddress, "driver"))
	if err != nil {
		return "", err
	}
	// 提取路径最后一段作为驱动名称
	_, driver := filepath.Split(driverPath)
	return driver, nil
}

// changeDriver 执行驱动的完整切换流程：先解绑当前驱动，再绑定到新驱动。
// 这是一个两步操作，如果 bind 失败，GPU 可能处于无驱动绑定的中间状态，
// 但这是可接受的——Unconfigure 时会将它绑回 nvidia 驱动。
//
// 参数：
//   - pciAddress: GPU 的 PCIe 总线地址
//   - driver: 目标驱动名称（如 "vfio-pci" 或 "nvidia"）
//
// 返回值：
//   - nil: 切换成功
//   - error: 解绑或绑定失败
func (vm *VfioPciManager) changeDriver(pciAddress, driver string) error {
	// 第一步：从当前驱动解绑
	err := vm.unbindFromDriver(pciAddress)
	if err != nil {
		return err
	}
	// 第二步：绑定到目标驱动
	err = vm.bindToDriver(pciAddress, driver)
	if err != nil {
		return err
	}
	return nil
}

// unbindFromDriver 将 GPU 从当前驱动解绑。
// 调用 /usr/bin/unbind_from_driver.sh 脚本执行解绑操作。
// 该脚本会处理驱动解绑、设备重置等底层操作。
//
// 参数：
//   - pciAddress: GPU 的 PCIe 总线地址
//
// 返回值：
//   - nil: 解绑成功
//   - error: 脚本执行失败
func (vm *VfioPciManager) unbindFromDriver(pciAddress string) error {
	// 执行解绑脚本，传入 PCI 地址作为参数
	out, err := execCommand(unbindFromDriverScript, []string{pciAddress}) //nolint:gosec
	if err != nil {
		klog.Errorf("Attempting to unbind %s from its driver failed; stdout: %s, err: %v", pciAddress, string(out), err)
		return err
	}
	return nil
}

// bindToDriver 将 GPU 绑定到指定驱动。
// 调用 /usr/bin/bind_to_driver.sh 脚本执行绑定操作。
// 该脚本会处理驱动绑定、设备重置等底层操作。
//
// 参数：
//   - pciAddress: GPU 的 PCIe 总线地址
//   - driver: 目标驱动名称
//
// 返回值：
//   - nil: 绑定成功
//   - error: 脚本执行失败
func (vm *VfioPciManager) bindToDriver(pciAddress, driver string) error {
	// 执行绑定脚本，传入 PCI 地址和驱动名称作为参数
	out, err := execCommand(bindToDriverScript, []string{pciAddress, driver}) //nolint:gosec
	if err != nil {
		klog.Errorf("Attempting to bind %s to %s driver failed; stdout: %s, err: %v", pciAddress, driver, string(out), err)
		return err
	}
	return nil
}

// GetVfioCommonCDIContainerEdits 返回所有 VFIO 直通设备共用的 CDI 容器编辑项。
// 包含：
//   - /dev/vfio/vfio 设备节点：VFIO 容器接口设备，允许用户态程序管理 IOMMU 组
//   - NVIDIA_VISIBLE_DEVICES=void 环境变量：屏蔽 NVIDIA GPU Operator 的自动设备注入，
//     确保只有 CDI 注入的设备对容器可见，防止重复注入
func GetVfioCommonCDIContainerEdits() *cdiapi.ContainerEdits {
	return &cdiapi.ContainerEdits{
		ContainerEdits: &cdispec.ContainerEdits{
			// /dev/vfio/vfio 是 VFIO 的容器级接口设备，所有 VFIO 直通设备共享此设备
			DeviceNodes: []*cdispec.DeviceNode{
				{
					Path: filepath.Join(vfioDevicesRoot, "vfio"),
				},
			},
			// 设置此环境变量防止 NVIDIA GPU Operator 的 pre-start hook 注入 GPU 设备
			// 仅让 CDI 设备注入机制生效
			Env: []string{"NVIDIA_VISIBLE_DEVICES=void"},
		},
	}
}

// GetVfioCDIContainerEdits 返回容器在 GPU 绑定 vfio-pci 驱动时访问该设备所需的 CDI 编辑项。
// 编辑项包含该 GPU 对应的 IOMMU 组设备节点 /dev/vfio/<iommuGroup>，
// 容器运行时通过此设备节点与 VFIO 驱动交互，实现对 GPU 的直通访问。
//
// 参数：
//   - info: VFIO 设备信息，包含 iommuGroup 编号
//
// 返回值：包含 IOMMU 组设备节点的 CDI 容器编辑项
func GetVfioCDIContainerEdits(info *VfioDeviceInfo) *cdiapi.ContainerEdits {
	// 构造 IOMMU 组对应的设备节点路径，如 /dev/vfio/32
	vfioDevicePath := filepath.Join(vfioDevicesRoot, fmt.Sprintf("%d", info.iommuGroup))
	return &cdiapi.ContainerEdits{
		ContainerEdits: &cdispec.ContainerEdits{
			DeviceNodes: []*cdispec.DeviceNode{
				{
					// 该设备节点允许容器通过 VFIO API 直接操作 GPU 硬件
					Path: vfioDevicePath,
				},
			},
		},
	}
}

// execCommandWithChroot 在宿主机 chroot 环境中执行命令。
// 用于在容器化部署中操作宿主机的文件系统和内核模块。
// 实际执行的命令为：chroot <fsRoot> <cmd> <args...>
//
// 参数：
//   - fsRoot: 宿主机根文件系统在容器内的挂载路径（通常为 /host-root）
//   - cmd: 要执行的命令名称
//   - args: 命令参数
//
// 返回值：
//   - []byte: 命令的标准输出和标准错误的合并输出
//   - error: 命令执行失败
func execCommandWithChroot(fsRoot, cmd string, args []string) ([]byte, error) {
	// 构造 chroot 命令参数：chroot <fsRoot> <cmd> <args...>
	chrootArgs := []string{fsRoot, cmd}
	chrootArgs = append(chrootArgs, args...)
	// CombinedOutput 同时捕获 stdout 和 stderr
	return exec.Command("chroot", chrootArgs...).CombinedOutput()
}

// execCommand 直接执行命令（不使用 chroot）。
// 用于执行容器内的脚本，如 unbind_from_driver.sh / bind_to_driver.sh。
//
// 参数：
//   - cmd: 命令路径或名称
//   - args: 命令参数
//
// 返回值：
//   - []byte: 命令的标准输出和标准错误的合并输出
//   - error: 命令执行失败
func execCommand(cmd string, args []string) ([]byte, error) {
	return exec.Command(cmd, args...).CombinedOutput()
}
