# M0 · 环境闸门与双机工具链

目的：把"环境能不能跑"从后面的里程碑里剥出来，第一周就把答案钉死。产出四样：**RoCE/GPUDirect 实测报告、双机同步与启动基建、CUDA 12.8 + NVSHMEM 工具链、IBGDA 前置条件的结论（能开/不能开）**。

## 现状（已核实，见总览事实清单）

- 两机 RoCE 直连 100GbE（192.168.100.1 ↔ .2），RoCE v2 GID index=3；verbs 设备名 gpu-02=`mlx5_0`、gpu-01=`rocep66s0f0`（**不同名**）。
- 无 nvidia_peermem、无 gdrdrv、PeerMappingOverride 未设、无 sudo。GDR 唯一现实路线 = dmabuf（内核 6.18 支持）。
- `/usr/bin/ib_write_bw` 两机都有（发行版 perftest，**是否编了 CUDA 支持待查**：`ib_write_bw --help | grep -i cuda`，没有就从源码 `./configure --enable-cuda --with-cuda=$HOME/cuda-12.8` 编到 ~/bin）。
- nvcc 只有 13.3，torch 是 cu12.8 → `torch.utils.cpp_extension` 会因 CUDA major 不匹配拒绝编译。

## 任务 1 · 双机基建脚本

新增 `scripts/`（这些脚本后面所有里程碑复用）：

```bash
# scripts/hosts.sh —— 唯一的主机事实源
GPU01=inet-p4lab-gpu-01.mpi-inf.mpg.de
GPU01_IP=192.168.100.1
GPU02_IP=192.168.100.2          # 本机，rank 0 / MASTER
IFNAME=ens5f0np0                # 两机同名
HCA_GPU02=mlx5_0                # 注意两机 verbs 名不同
HCA_GPU01=rocep66s0f0
GID_INDEX=3                     # RoCE v2, IPv4-mapped

# scripts/sync.sh —— gpu-02 → gpu-01 单向同步（代码 + 权重）
rsync -a --delete --exclude .git --exclude tests/out --exclude '__pycache__' \
      ~/CodeRead/vllm/nano-vllm/ $GPU01:~/CodeRead/vllm/nano-vllm/
rsync -a ~/huggingface/Qwen3-0.6B/       $GPU01:~/huggingface/Qwen3-0.6B/
rsync -a ~/huggingface/tiny-qwen3-moe/   $GPU01:~/huggingface/tiny-qwen3-moe/   # M3 产出后
# .venv 单独同步（同绝对路径、同 OS/glibc，可直接搬；--delete 保证两边 site-packages 位级一致）
rsync -a --delete ~/CodeRead/vllm/nano-vllm/.venv/ $GPU01:~/CodeRead/vllm/nano-vllm/.venv/
```

约定：**gpu-01 上不手工改任何代码**，一切以 gpu-02 为准单向覆盖，杜绝双向漂移。

## 任务 2 · 网络与 GPUDirect 分层实测（验收的主体）

按四层递进，每层有明确通过判据，全部命令写进 `scripts/m0_nettest.sh`：

**L1 纯 RDMA（host 内存）**——验证 RoCE 链路与 GID 配置：
```bash
# gpu-01:  ib_write_bw -d rocep66s0f0 -x 3 --report_gbits
# gpu-02:  ib_write_bw -d mlx5_0      -x 3 --report_gbits 192.168.100.1
```
通过判据：BW ≥ 90 Gb/s（100GbE 线速率的 90%）。顺带跑 `ib_send_bw`、**`ib_atomic_bw`（RDMA fetch-add）**——后者是 M5 的 ibgda 计数通知（`MLX5_OPCODE_ATOMIC_MASKED_FA`）能否工作的前置信号；同时 `lspci | grep -i mell` 记录 NIC 具体型号（CX-6/6Dx/7 对 RoCE atomic 支持不同）。

**L2 GPU 内存 GDR**——验证 dmabuf 路线：
```bash
# perftest 若支持: 两端加 --use_cuda=0 --use_cuda_dmabuf 重跑 L1
```
通过判据：GPU 内存版 BW 与 host 版同量级（≥80 Gb/s）。若发行版 perftest 无 CUDA 支持且不想编，可跳过，由 L3 的 NCCL 日志代替证明。

**L3 NCCL 双机（torch 栈，最贴近实际用法）**：
```python
# scripts/m0_nccl_test.py —— 两机各起一个进程
# gpu-02: MASTER_ADDR=192.168.100.2 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 python scripts/m0_nccl_test.py
# gpu-01: 同左 RANK=1
# 内容: init_process_group("nccl") → all_reduce 校验和 → all_to_all_single 1MB~256MB 扫带宽 → barrier
```
环境变量（写进脚本，两机差异用 hostname 分支）：
```bash
export NCCL_SOCKET_IFNAME=ens5f0np0        # bootstrap 走直连口，别绕 bond0
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
# NCCL_IB_HCA 可不设（每机只有一个 ACTIVE 口，自动选中）；若选错则按 hosts.sh 显式指定
```
通过判据：① all_reduce 数值正确；② 大消息 all_to_all busbw ≥ 10 GB/s；③ **INFO 日志出现 `via NET/IB/... /GDRDMA`**（GDR 生效的直接证据，截图存档 `Plan-4/artifacts/`）。若显示 GDR Disabled，排查 dmabuf：`python -c "import torch; print(torch.cuda.is_available())"` + `NCCL_DMABUF_ENABLE=1` 强制，仍不行则记录为"GDR 不可用"并升级为阻塞项（这与用户已验证"gpu12 可用 GPU direct"矛盾，需当面核对他们当时的验证方式）。

**L4 IBGDA 前置**（只做检查与请求，不阻塞其他线）：
- 检查表（两机跑 `scripts/m0_ibgda_check.sh`）：`grep RegistryDwords /proc/driver/nvidia/params`（需含 PeerMappingOverride=1）、`lsmod | grep gdrdrv`、`ls /dev/gdrdrv`。当前已知全部为否。
- 给管理员的请求（二选一，写邮件）：
  - 方案 A（性能最好）：两机 `/etc/modprobe.d/nvidia.conf` 加 `options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"`，update-initramfs + 重启。
  - 方案 B（免改驱动）：两机安装 gdrcopy（deb 包，加载 gdrdrv 模块）→ NVSHMEM 走 CPU 辅助 IBGDA（`NVSHMEM_IBGDA_NIC_HANDLER=cpu`，DeepEP 内核已支持该模式：`ibgda_device.cuh` 的 `use_async_postsend` 分支）。
- 结论记录进 `Plan-4/artifacts/m0-report.md`：A/B/都不行。都不行 → M5 冻结、M6 按 NCCL 后端交付（总览风险表）。

## 任务 3 · CUDA 12.8 用户态工具链 + NVSHMEM（为 M5 铺路，一并做了）

```bash
# 1) CUDA 12.8 toolkit 免 root 安装（两机同路径）
wget .../cuda_12.8.*_linux.run
sh cuda_12.8.*.run --silent --toolkit --toolkitpath=$HOME/cuda-12.8 --no-drm --no-man-page
export CUDA_HOME=$HOME/cuda-12.8   # 写进 scripts/env.sh

# 2) NVSHMEM（pip wheel，装进 nano-vllm/.venv，随任务 1 的 rsync 到 gpu-01）
.venv/bin/pip install "nvidia-nvshmem-cu12>=3.3.9"
# 验证: .venv 下找到 lib/libnvshmem_host.so.3 + libnvshmem_device.a + include/
#（DeepEP setup.py:26-47 专门处理了 pip wheel 只有带版本号 so 名的情况，nano 的 setup 照抄）

# 3) 编译冒烟: 用 CUDA_HOME=$HOME/cuda-12.8 编一个 hello CUDAExtension（sm_89），import 成功
```

nvshmem host 侧自检（不依赖 IBGDA，验证安装本身）：`nvshmem-info -a` 能打印版本与 transport 列表。

## 任务 4 · 双机 Python 环境对齐

- 按任务 1 rsync `.venv`（gpu-01 的 ~/ds-venv 只作参考不使用，避免两套环境）。
- gpu-01 冒烟：`ssh gpu-01 '~/CodeRead/vllm/nano-vllm/.venv/bin/python -c "import torch, flash_attn, transformers; print(torch.__version__, torch.cuda.get_device_name())"'`。
- 产出 `requirements-lock.txt`（pip freeze）入库，之后任何一侧改包必须重新 rsync。

## 验收清单（全过才关闸门）

| # | 项 | 判据 |
|---|---|---|
| 1 | L1 RDMA host BW | ≥ 90 Gb/s，双向 |
| 2 | L1' RDMA atomic | ib_atomic_bw 正常完成（记录 NIC 型号） |
| 3 | L2/L3 GDR 证据 | perftest cuda 版达标 或 NCCL 日志 GDRDMA 字样 |
| 4 | L3 torch 双机 | all_reduce 正确 + all_to_all ≥ 10 GB/s |
| 5 | 工具链 | CUDA_HOME=~/cuda-12.8 能编 sm_89 扩展并 import；nvshmem wheel 就位 |
| 6 | 双机对齐 | gpu-01 冒烟 import 全过；sync.sh 幂等（连跑两遍第二遍无传输） |
| 7 | IBGDA 结论 | m0-report.md 写明 A/B/都不行 + 管理员请求已发出 |

## 边界与坑

- **master 地址必须用 192.168.100.2**：hostname 解析会落到 bond0 的 139.19.59.x（走机房公网交换机），gloo/nccl bootstrap 和小消息都会绕远、还可能被防火墙拦。
- 两机 verbs 设备名不同（udev 策略差异）：任何按名字选卡的 env（NCCL_IB_HCA / NVSHMEM_HCA_LIST）都要按 hostname 分支，统一收口在 `scripts/env.sh`。
- perftest 双机时钟不同步会让延迟数难看，但带宽数不受影响；本计划只看带宽。
- 驱动 595 vs 610：CUDA 13.2/13.3 runtime 均兼容 cu12.8 的 torch（向后兼容），不动它；若管理员顺手升级驱动，重跑本闸门。
