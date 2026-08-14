# M0 · 环境闸门实测报告

实测日期 2026-08-13，gpu-02(rank0) ↔ gpu-01(rank1)，脚本 `scripts/m0_nettest.sh`
与 `scripts/m0_nccl_test.py`。

## 结论速览

| # | 项 | 判据 | 实测 | 结果 |
|---|---|---|---|---|
| 1 | L1 RDMA host BW | ≥ 90 Gb/s | write **92.56**、send **92.58** Gb/s | ✅ |
| 2 | L1' RDMA atomic | 能正常完成 | FETCH_AND_ADD **15.60 MiB/s、2.045 Mpps** | ✅ |
| 3 | L2/L3 GDR 证据 | perftest CUDA 达标 或 NCCL 日志 GDRDMA | GPU 显存 **91.60 Gb/s**（dmabuf）+ NCCL 日志 `via NET/IB/0/GDRDMA` | ✅ |
| 4 | L3 torch 双机 | all_reduce 正确 + a2a ≥ 10 GB/s | 正确；256MB busbw **11.41 GB/s** | ✅ |
| 5 | 工具链 | 能编 sm_89 扩展并 import；nvshmem 就位 | `~/cuda-12.8` + ninja，扩展编译 import 通过；nvshmem wheel 3.7.2 | ✅ |
| 6 | 双机对齐 | gpu-01 冒烟全过；sync 幂等 | torch 2.8.0+cu128 / L40S / NCCL 2.27.3 / transformers 5.14.1 | ✅ |
| 7 | **IBGDA 结论** | A / B / 都不行 | 初判**都不行**；管理员随后开启方案 A → **通过** | ✅（见下方补记） |

**首次判定：M5 冻结，M6 按 NCCL 后端交付**——这正是 Plan-4 总览风险表里预设的分支，
NCCL 后端从一开始就是"保底交付"。

**补记（同日晚，闸门解除）**：用户即本机管理员，用 `scripts/enable_ibgda.sh --apply`
在两机开启了方案 A，**没有重启**。关键观察是：`PeerMappingOverride` 虽然是加载期参数、
运行时改不了，但可以卸载 nvidia 模块再带参数装回去，而本环境恰好满足条件
（无显示服务、无 CUDA 进程、`nvidia` 的 holders 只有 `modeset`/`uvm`，唯一占用
`nvidia-persistenced` 可停）。开启后：

```
$ ./scripts/m0_ibgda_check.sh
gpu-02: RegistryDwords : "PeerMappingOverride=1;"   方案A ✅ 已开   ==> 闸门: 通过
gpu-01: RegistryDwords : "PeerMappingOverride=1;"   方案A ✅ 已开   ==> 闸门: 通过
```

后续的 NVSHMEM/IBGDA 环境验证结果与本轮踩的坑，见
[08-implementation-report.md](../08-implementation-report.md) 的 §4.1 与坑 16-23。

## 1. 链路与 GPUDirect

NIC：活跃的 100GbE 口是 **ConnectX-6 Dx（MT2892，PCI 42:00.0）**；另有一对 ConnectX-6 Lx
在 c5:00.x，链路 DOWN。两机 verbs 名不同（gpu-02 `mlx5_0` / gpu-01 `rocep66s0f0`），
RoCE v2 的 GID index 两机都是 **3**（`0000:...:ffff:c0a8:640{1,2}`）。

```
L1 RDMA write (host mem)                 92.56 Gb/s
L1 RDMA send  (host mem)                 92.58 Gb/s
L1' RDMA atomic fetch-add    15.60 MiB/s / 2.045 Mpps
L2 RDMA write (GPU mem, dmabuf)          91.60 Gb/s     ← GPUDirect 实锤
```

**L2 的关键细节**：不加 `--use_cuda_dmabuf` 时 perftest 走老的 `ibv_reg_mr(CUDA VA)` 路线，
那条路需要 `nvidia_peermem` 内核模块，本机没有 → 服务端直接 `failed to create mr`。
换 dmabuf（内核 6.18 + 驱动 595/610 原生支持，不需要任何额外模块）之后一次通过，
带宽与 host 内存版本只差 1%，说明数据确实是 NIC ↔ GPU 显存直传、没有 host 中转。

**L1' 的意义**：RDMA atomic fetch-add 是 M5 里 ibgda dispatch 的计数通知
（`MLX5_OPCODE_ATOMIC_MASKED_FA`）能否工作的前置信号。它可用，意味着即便将来闸门放开，
也不必启用 Plan-4/06 里准备的"把 amo_nonfetch_add 换成第二笔 rma_p 单写"兜底方案。

## 2. NCCL 双机（torch 栈）

```
[L3-1] all_reduce  OK  (got 3.0, want 3.0)
       bytes        ms   algbw GB/s   busbw GB/s
     1048576     0.082        12.72         6.36
     4194304     0.232        18.10         9.05
    16777216     0.788        21.30        10.65
    67108864     2.987        22.47        11.23
   268435456    11.763        22.82        11.41
[L3-3] gloo broadcast_object_list(~0.5KB): 0.26~0.28 ms/次
```

**踩到的坑：GDR 默认被 NCCL 自己关掉了。** 首次跑日志里是

```
GPU Direct RDMA Disabled for GPU 0 / HCA 0 (distance 8 > 5)
Connected all rings, use ring PXN 0 GDR 0
```

这不是能力问题而是策略问题：GPU 与活跃网卡跨了 host bridge，NCCL 算出的拓扑距离是 8，
超过 `NCCL_NET_GDR_LEVEL` 的默认阈值 5(PXN) 就自动禁用。设
`NCCL_NET_GDR_LEVEL=SYS`（已写进 `scripts/env.sh`）后：

```
NET/IB : GPU Direct RDMA Enabled for HCA 0 'mlx5_0'
GPU Direct RDMA Enabled for GPU 0 / HCA 0 (distance 8 <= 9), read 1 mode Default
Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA
Connected all rings, use ring PXN 0 GDR 1
```

值得记一笔：**开不开 GDR，带宽一模一样（都是 11.4 GB/s）**。100GbE 下瓶颈在网卡，
不在 PCIe 跨桥的那次拷贝。GDR 省的是延迟和 host CPU/内存带宽，不是峰值吞吐——
这条对 M6 解读"EP=2 为什么没变快"很重要。

gloo 控制面 0.26 ms/次是 M1 每步广播的固定开销，符合计划里 <1ms 的预期。

## 3. IBGDA 前置条件（闸门项）

两机检查结果完全一致：

```
/proc/driver/nvidia/params:  RegistryDwords: ""      ← PeerMappingOverride 未设
lsmod | grep -E peermem|gdrdrv:  (空)                ← 两个模块都没有
ls /dev/gdrdrv:  No such file or directory
sudo -n true:  sudo: a password is required          ← 无 root
```

两条开启路径都要管理员：

- **方案 A**（性能最好）：两机 `/etc/modprobe.d/nvidia.conf` 加
  `options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"`，
  `update-initramfs` + 重启。
- **方案 B**（免改驱动）：两机装 gdrcopy（加载 gdrdrv 模块），NVSHMEM 走 CPU 辅助 IBGDA
  （`NVSHMEM_IBGDA_NIC_HANDLER=cpu`，DeepEP 的 `ibgda_device.cuh` 里 `use_async_postsend`
  分支就是这个模式）。

**（当时的）结论：A/B 都不行。** M5 冻结。

**→ 后来解除了**，见本文开头的补记：方案 A 可以靠**热重载 nvidia 模块**生效，不必重启，
本环境恰好满足热重载条件。工具见 `scripts/enable_ibgda.sh`（前置检查 / `--apply` /
`--revert`）与 `scripts/m0_ibgda_check.sh`（两机只读复核）。

## 4. 工具链（为 M5 铺路，已就绪）

机器上 **原本连 nvcc 都没有**——`/usr/local/cuda-13.3` 只有 `lib64/` 和 `targets/`
两个目录（纯 runtime 库），`/usr/local/cuda/bin` 不存在。而且 13.x 与 torch 的 cu12.8
major 不匹配，本来也编不了扩展。

免 root 装用户态 CUDA 12.8：

```
sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit \
   --toolkitpath=$HOME/cuda-12.8 --no-drm --no-man-page --override
→ /home/weihaoni/cuda-12.8/bin/nvcc  release 12.8, V12.8.61
```

冒烟：`TORCH_CUDA_ARCH_LIST=8.9` 下 JIT 编一个 `torch::Tensor` + 自写 kernel 的扩展，
编译并 import 通过（另外补装了 `ninja`，torch 的 cpp_extension 硬依赖它）。

NVSHMEM：`uv pip install nvidia-nvshmem-cu12` → 3.7.2，装在
`.venv/lib/python3.12/site-packages/nvidia/nvshmem/`，含

```
lib/libnvshmem_host.so.3          ← 只有带版本号的 so 名，正是 DeepEP setup.py:26-47
lib/libnvshmem_device.a              专门处理的那种情况，nano 的 setup 照抄即可
lib/libnvshmem_device_sm_89.bc    ← SM89 device bitcode 现成，对 M5 是好消息
include/{nvshmem.h,nvshmemx.h,device/,host/,...}
```

wheel 里**没有** `nvshmem-info` 可执行文件，所以计划里"`nvshmem-info -a` 自检"这一步做不了。
闸门放开后改用自己写的最小 device-initiated put 测试代替
（`nanodeepep/tests/test_ibgda_smoke.py`），结论见 08 报告 §4.1：
NVSHMEM 3.7.2 的 IBGDA transport 在本环境可用，跨机 4B/4KB/1MB 的 put 全对。
另外 wheel 里确实带了 `nvshmem_transport_ibgda.so.6`（当时没注意看，只列了几个 bootstrap so）。

## 5. 双机同步

`scripts/sync.sh`：代码 + 权重（秒级）；`--venv` 追加 7.5G 的 `.venv` 和它依赖的 uv 解释器
（`~/.local/share/uv/python/`，venv 的 `pyvenv.cfg` 里 `home=` 指向它，不搬过去 gpu-01
的 python 起不来）。仓库里的 `.venv` 是符号链接，rsync 时排除、在远端重建。

同步走 **192.168.100.1 直连口**而不是管理网主机名：bond0 只有 2 GbE，直连口 100 GbE，
7.5G 的差别是分钟级 vs 十几分钟。

gpu-01 冒烟：`torch 2.8.0+cu128 / NVIDIA L40S / nccl (2,27,3) / transformers 5.14.1`，
与 gpu-02 完全一致。

踩到的两个脚本坑（都改进了脚本）：

1. `R="rsync -a -e $SSH"` 这种把 `-e "ssh -o BatchMode=yes"` 塞进变量的写法会被词法拆开，
   rsync 只吃到 `-e ssh`，剩下的 `-o BatchMode=yes` 被当成**源路径** →
   `link_stat ".../BatchMode=yes" failed`。改成 shell 函数。
2. 远端起 perftest 服务端时 `pkill -f 'ib_write_bw'` 会匹配到 **ssh 自己那条 bash -c 命令行**，
   服务端还没起来就先把自己的父 shell 杀了，客户端只看到
   `Couldn't connect to 192.168.100.1:18515`。改用 `pkill -x`（按进程名精确匹配）
   + `setsid nohup ... < /dev/null` 让服务端脱离 ssh 会话。
