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

// root 包定义了 NVIDIA 驱动的文件系统根路径类型及其路径查找方法。
//
// 在容器化部署场景中，NVIDIA 驱动可能安装在容器的非标准路径下（如 "/run/nvidia/driver"），
// 而非传统的 "/" 根路径。root 类型封装了这种路径差异，使驱动文件查找逻辑透明化：
//   - 所有文件查找操作都相对于 root 路径进行
//   - root 路径可以是 "/" （宿主机原生驱动）或 "/run/nvidia/driver"（容器化驱动）
//
// 本文件提供的方法用于：
//   - 查找 libnvidia-ml.so.1 驱动库路径（NVML 初始化的必要依赖）
//   - 查找 nvidia-smi 可执行文件路径（时间切片和计算模式管理的必要依赖）
//   - 判断 root 路径是否为设备节点根路径（包含 /dev 目录）
//   - 获取设备节点的实际根路径（影响 CDI spec 中设备节点路径的生成）
//
// 关键设计决策：
//   - 使用类型别名（type root string）而非结构体，因为 root 本质就是一个路径字符串
//   - 搜索路径列表覆盖主流 Linux 发行版的标准安装路径（RHEL、Ubuntu、Debian 等）
//   - 符号链接递归解析确保返回最终目标路径，避免挂载到容器的链接失效
package main

import (
	"fmt"
	"os"
	"path/filepath"
)

// root 是 NVIDIA 驱动在文件系统中的安装根路径。
//
// 所有文件查找操作（如查找驱动库、设备节点、可执行文件）都相对于这个路径进行。
//
// 典型值：
//   - "/" ：宿主机原生驱动安装（最常见场景）
//   - "/run/nvidia/driver" ：容器化驱动安装（NVIDIA GPU Operator 的驱动容器模式）
//
// 在容器化场景中，宿主机的驱动文件系统被挂载到容器的此路径下，
// 因此需要在所有路径查找操作前添加此前缀，才能正确找到驱动文件。
type root string

// getDriverLibraryPath 在驱动根目录下的标准路径中查找 libnvidia-ml.so.1。
//
// libnvidia-ml.so.1 是 NVIDIA Management Library (NVML) 的核心共享库，
// 所有 GPU 管理操作（设备枚举、状态查询、MIG 配置等）都依赖此库。
// 该库必须在插件启动时成功加载，否则 NVML 初始化将失败。
//
// 搜索路径覆盖了主流 Linux 发行版的标准库安装位置：
//   - /usr/lib64       ：RHEL/CentOS/Fedora 的 64 位库路径
//   - /usr/lib/x86_64-linux-gnu   ：Debian/Ubuntu 的 x86_64 库路径
//   - /usr/lib/aarch64-linux-gnu  ：Debian/Ubuntu 的 ARM64 库路径
//   - /lib64           ：传统 64 位库路径（某些旧版发行版）
//   - /lib/x86_64-linux-gnu       ：Debian/Ubuntu 的系统级 x86_64 库路径
//   - /lib/aarch64-linux-gnu      ：Debian/Ubuntu 的系统级 ARM64 库路径
//
// 搜索策略：
//   1. 先在 root 根路径（即 root + "/"）下查找
//   2. 再依次在各子目录中查找
//   3. 找到后解析符号链接返回最终真实路径
//
// 返回值说明：
//   - 成功时返回库文件的绝对路径（符号链接已解析）
//   - 失败时返回空字符串和 error，表明驱动未正确安装或路径配置有误
//
// 注意：返回的目录也被认为是其他驱动文件（如 nvidia-smi）所在的目录。
func (r root) getDriverLibraryPath() (string, error) {
	// 定义标准库搜索路径，覆盖主流 Linux 发行版
	librarySearchPaths := []string{
		"/usr/lib64",
		"/usr/lib/x86_64-linux-gnu",
		"/usr/lib/aarch64-linux-gnu",
		"/lib64",
		"/lib/x86_64-linux-gnu",
		"/lib/aarch64-linux-gnu",
	}

	// 在 root 路径下的各标准目录中查找 libnvidia-ml.so.1
	libraryPath, err := r.findFile("libnvidia-ml.so.1", librarySearchPaths...)
	if err != nil {
		// 驱动库未找到，NVML 将无法初始化，插件无法正常工作
		return "", err
	}

	return libraryPath, nil
}

// getNvidiaSMIPath 在驱动根目录下查找 nvidia-smi 可执行文件路径。
//
// nvidia-smi 是 NVIDIA System Management Interface 的命令行工具，
// 本插件在以下场景中使用它：
//   - 配置时间切片（Time-Slicing）：通过 nvidia-smi -q -d COMPUTE 查询当前配置，
//     并通过 nvidia-smi compute-policy 设置计算模式
//   - 设置计算模式：通过 nvidia-smi -c EXCLUSIVE_PROCESS 等命令，
//     控制 GPU 的进程访问策略（对 MPS 功能至关重要）
//
// 搜索路径覆盖了常见的二进制文件安装位置：
//   - /opt/bin    ：NVIDIA GPU Operator 自定义安装路径
//   - /usr/bin    ：标准用户二进制路径（大多数发行版）
//   - /usr/sbin   ：系统管理二进制路径
//   - /bin        ：基础二进制路径
//   - /sbin       ：系统管理二进制路径（某些旧版发行版）
//
// 返回值说明：
//   - 成功时返回 nvidia-smi 的绝对路径（符号链接已解析）
//   - 失败时返回空字符串和 error，表明 nvidia-smi 未安装
func (r root) getNvidiaSMIPath() (string, error) {
	// 定义可执行文件搜索路径，覆盖主流安装位置
	binarySearchPaths := []string{
		"/opt/bin",
		"/usr/bin",
		"/usr/sbin",
		"/bin",
		"/sbin",
	}

	// 在 root 路径下的各标准目录中查找 nvidia-smi
	binaryPath, err := r.findFile("nvidia-smi", binarySearchPaths...)
	if err != nil {
		// nvidia-smi 未找到，时间切片和计算模式管理功能将不可用
		return "", err
	}

	return binaryPath, nil
}

// isDevRoot 判断该 root 是否为"设备根"（Device Root）。
//
// 设备根的定义：root 路径下存在 /dev 目录（即 root + "/dev" 是一个有效目录）。
//
// 判断设备根的意义：
//   - CDI spec 中的设备节点路径需要相对于正确的根路径生成
//   - 若 root 是设备根（容器化驱动场景），则 /dev/nvidia0 等设备节点的
//     实际路径为 root + "/dev/nvidia0"（如 "/run/nvidia/driver/dev/nvidia0"）
//   - 若 root 不是设备根（宿主机根路径 "/"），则设备节点直接在 "/dev/nvidia0"
//
// 判断逻辑：
//   1. 拼接 root + "/dev" 路径
//   2. 使用 os.Stat 检查该路径是否存在
//   3. 若存在且为目录，返回 true；否则返回 false
//
// 参数：无（方法接收者为 root 本身）
//
// 返回值：
//   - true:  root 路径下包含 /dev 目录，是设备根
//   - false: root 路径下不包含 /dev 目录，不是设备根
func (r root) isDevRoot() bool {
	// 检查 root + "/dev" 路径是否存在
	stat, err := os.Stat(filepath.Join(string(r), "dev"))
	if err != nil {
		// 路径不存在或无法访问，不是设备根
		return false
	}
	// 确认该路径是目录（而非同名文件）
	return stat.IsDir()
}

// getDevRoot 返回设备节点根路径。
//
// 该方法用于确定 CDI spec 中设备节点路径的前缀。
// 设备根的选择逻辑：
//   - 如果 root 本身包含 /dev 目录（容器化驱动场景），则返回 root 本身
//     例如 root="/run/nvidia/driver" 且该路径下有 /dev 目录 → 返回 "/run/nvidia/driver"
//   - 否则返回 "/"（宿主机根路径），表示设备节点在标准 /dev 路径下
//     例如 root="/run/nvidia/driver" 但该路径下没有 /dev 目录 → 返回 "/"
//
// 这种逻辑的合理性：
//   - 在 NVIDIA GPU Operator 的驱动容器部署中，宿主机的 /dev 目录被挂载到
//     容器的 /run/nvidia/driver/dev 路径下，因此 root 包含 /dev
//   - 在宿主机原生驱动安装中，root="/"，/dev 目录自然在根路径下
//   - 某些特殊部署中，驱动库挂载到容器但设备节点不跟随，
//     此时 root 不包含 /dev，需要回退到宿主机根路径
//
// 返回值：
//   - 设备节点根路径字符串，格式如 "/run/nvidia/driver" 或 "/"
func (r root) getDevRoot() string {
	// 若 root 路径下包含 /dev 目录，则 root 本身就是设备根
	if r.isDevRoot() {
		return string(r)
	}
	// 否则回退到宿主机根路径，设备节点在标准的 /dev 路径下
	return "/"
}

// findFile 在 root 路径及指定的子目录列表中搜索文件 name。
//
// 搜索顺序：
//   1. 先尝试 root 根路径（即 root + "/" + name），覆盖文件直接安装在 root 目录下的情况
//   2. 再依次尝试各子目录（即 root + searchIn[i] + "/" + name）
//
// 例如 root="/run/nvidia/driver"，name="libnvidia-ml.so.1"：
//   - 第一次尝试：/run/nvidia/driver/libnvidia-ml.so.1
//   - 第二次尝试：/run/nvidia/driver/usr/lib64/libnvidia-ml.so.1
//   - 第三次尝试：/run/nvidia/driver/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
//   - ...以此类推
//
// 若文件是符号链接，则递归解析到最终目标路径后返回。
// 这确保了返回的路径始终是实际文件的位置，避免容器挂载后符号链接失效的问题。
//
// 参数：
//   - name: 要查找的文件名（如 "libnvidia-ml.so.1"、"nvidia-smi"）
//   - searchIn: 要搜索的子目录列表（如 "/usr/lib64"、"/usr/bin"）
//
// 返回值：
//   - string: 找到的文件的绝对路径（符号链接已解析）
//   - error: 所有路径均未找到时返回错误
func (r root) findFile(name string, searchIn ...string) (string, error) {
	// 先搜索 root 根路径，再搜索各子目录
	// append([]string{"/"}, searchIn...) 确保根路径排在最前面
	for _, d := range append([]string{"/"}, searchIn...) {
		// 拼接完整候选路径：root + 目录 + 文件名
		l := filepath.Join(string(r), d, name)

		// 尝试解析符号链接，若文件不存在则跳过此路径继续搜索
		candidate, err := resolveLink(l)
		if err != nil {
			// 文件不存在或无法访问，继续尝试下一个路径
			continue
		}

		// 找到文件，返回解析后的真实路径
		return candidate, nil
	}

	// 所有候选路径均未找到目标文件
	return "", fmt.Errorf("error locating %q", name)
}

// resolveLink 解析路径 l：若为符号链接则递归跟随链接；若为普通文件则直接返回。
//
// 此函数等价于 shell 命令 `readlink -f ${l}`，将路径中所有层级的符号链接
// 递归解析为最终的目标路径。
//
// 为什么需要解析符号链接：
//   - NVIDIA 驱动安装通常使用符号链接管理库版本（如 libnvidia-ml.so → libnvidia-ml.so.1 → libnvidia-ml.so.535.104.05）
//   - 在容器挂载场景中，符号链接的原始目标路径可能在容器内不存在
//   - 解析到最终真实路径后，CDI spec 可以正确地挂载实际文件而非符号链接
//
// 参数：
//   - l: 待解析的文件路径（可以是符号链接或普通文件路径）
//
// 返回值：
//   - string: 解析后的真实绝对路径（所有符号链接已跟随）
//   - error: 路径不存在或解析失败时返回错误
func resolveLink(l string) (string, error) {
	// filepath.EvalSymlinks 递归解析路径中所有符号链接
	// 若路径不存在、无法访问或链接循环，则返回错误
	resolved, err := filepath.EvalSymlinks(l)
	if err != nil {
		return "", fmt.Errorf("error resolving link '%v': %v", l, err)
	}
	return resolved, nil
}
