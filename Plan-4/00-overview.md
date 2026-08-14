# Plan-4 总览 · nano-deepEP + MoE + 跨机 EP（独立并行线）

目标：在 nano-vllm 上跑通 **4 专家的 Qwen3-MoE，专家切分到两台机器**（gpu-02 本机 rank 0 持有 expert 0/1，gpu-01 远端 rank 1 持有 expert 2/3），token 经 **GPUDirect RDMA** 跨机 dispatch/combine。通信库 **nano-deepEP** 基于 `/home/weihaoni/CodeRead/vllm/DeepEP` 的 legacy（V1）low-latency 路径忠实移植，关键代码直接复制。

这条线与 Plan-1-2-3 **并行独立**：M0/M2/M3 不碰 nanovllm 引擎代码，可立即开工；最终在 M4 与 1-2-3 产出的 nano-vllm（统一调度 + 投机解码已完成的现状）汇合。

## 三层工程与里程碑

| 里程碑 | 内容 | 对应用户三层 | 依赖 | 风险 |
|---|---|---|---|---|
| M0 | 环境闸门：RoCE GPUDirect 实测、双机工具链、IBGDA 前置条件 | （地基） | 无 | 中（IBGDA 需管理员） |
| M1 | 多机通信层：打破 `model_runner.py:26` 的 localhost 硬编码，parallel_state，gloo 控制面，双机启动脚本 | ① 多机通信 | M0 | 中 |
| M2 | nano-deepEP 包：API（= deep_ep.Buffer 子集）+ **NCCL 参考后端**（立即可跑，兼作正确性 oracle） | ③ EP dispatch/combine | M0 | 低 |
| M3 | MoE 模型层：tiny Qwen3-MoE 权重生成 + `qwen3_moe.py`（EP=1 本地版），vs HF logits 对拍 | ② MoE 模型层 | 无 | 低 |
| M4 | EP 集成：M1+M2+M3 汇合，双机 4 专家端到端（先用 NCCL 后端） | ①+②+③ | M1,M2,M3 | 中 |
| M5 | IBGDA 后端：复制 DeepEP legacy internode_ll 内核，SM89（L40S）手术移植，standalone 双机验收 | ③（忠实 DeepEP） | M0 闸门 | **高** |
| M6 | 切换 nvshmem 后端端到端 + 三配置基准（EP=1 / EP=2+nccl / EP=2+ibgda）+ 实现报告 | 全部 | M4,M5 | 中 |
| M7 | 延伸（不承诺）：DP attention、recv-hook 重叠、CUDA graph、FP8 dispatch | — | M6 | — |

并行泳道：`M0 → M1`、`M2`、`M3` 三条支线互不依赖可同时推进；M2 的双机测试用裸 `torchrun` 起进程，不等 M1。

## 环境事实清单（2026-08-13 实测核实，计划的一切决策基于此）

| 项 | gpu-02（本机，rank 0/driver） | gpu-01（远端，rank 1） |
|---|---|---|
| 主机名 | inet-p4lab-gpu-02 | inet-p4lab-gpu-01.mpi-inf.mpg.de |
| GPU | 1× L40S 46GB，**SM 8.9（Ada）** | 同 |
| 驱动 / 内核 | 595.71.05 / 6.18.38 | **610.57.04**（版本不一致，NCCL/NVSHMEM 不敏感，记录在案）/ 6.18.38 |
| CUDA toolkit | 仅 /usr/local/cuda-13.3（**与 torch cu12.8 major 不匹配，不能直接编扩展**） | 同 |
| Python 栈 | nano-vllm/.venv：torch 2.8.0+cu128，NCCL 2.27.3，transformers 5.14.1（含 Qwen3Moe），flash-attn 2.8.3 | ~/ds-venv：torch 2.8.0+cu128；**无 nano-vllm，无 CodeRead/vllm 目录** |
| RDMA 网卡 | 4× mlx5，仅 `mlx5_0`(netdev `ens5f0np0`) ACTIVE，**RoCE（Ethernet 链路层）100Gb/s** | 同硬件；verbs 设备名被 udev 改名为 **`rocep66s0f0`**（两机名字不同！） |
| 直连私网 | 192.168.100.2/24 | 192.168.100.1/24（两机网卡直连，无交换机路径干扰） |
| RoCE v2 GID | **index 3**（0000:…:ffff:c0a8:6402） | 同结构（M0 复核） |
| GPU↔活跃网卡拓扑 | GPU—mlx5_0 为 **NODE**（跨 host bridge；离 GPU 最近的 NIC2/3 是 DOWN 的，带宽上限受此影响） | GPU—NIC0 为 PHB（较优） |
| GPUDirect 现状 | `nvidia_peermem` 模块**不存在**（modinfo 为空）、gdrdrv 未装 → GDR 只能走 **dmabuf** 路线（内核 6.18 + 驱动 595/610 支持，NCCL 自动用） | 同 |
| IBGDA 前置 | `PeerMappingOverride` **未设**（/proc/driver/nvidia/params RegistryDwords 为空）、无 gdrcopy、无 sudo → **IBGDA 目前不可用，需管理员**（见 M0） | 同 |
| 文件系统 | home 为本机盘（/dev/md0 ext4），**非 NFS** → 代码/权重/venv 需 rsync 同步 | 同 |
| SSH | gpu-02 → gpu-01 免密直连**已验证可用**（同一 ed25519 key） | — |

## 技术路线决策（为什么是 legacy low-latency）

本仓库 DeepEP 是 **V2 主干**（ElasticBuffer + NCCL Gin 后端，README 要求 SM90、torch≥2.10、NCCL≥2.30.4）——三项全不满足，**V2 不可行**。但仓库完整保留了 legacy V1（`csrc/kernels/legacy/`、`deep_ep/buffers/legacy.py`、`tests/legacy/`），V1 有三条路径：

1. **intranode**（NVLink）：单机多卡用，与我们"两机各一卡"无关。
2. **normal internode**（NVLink+RDMA 混合）：按 `rdma_rank = rank / 8` 分组（`csrc/legacy/buffer.hpp:120`），设计前提是每节点 8 卡 NVLink 整组，2 ranks 时 `num_rdma_ranks = max(1, 2/8) = 1` 直接退化成"同节点"，**不适用**。
3. **low-latency（internode_ll，纯 RDMA + IBGDA）**：`low_latency_mode` 下 `nvshmem_rank = rank, num_nvshmem_ranks = num_ranks`（buffer.hpp:262-263），每个 rank 就是一个 RDMA PE，**任意 rank 数、任意每机卡数都支持**（构造断言 buffer.hpp:113/116 在 LL 模式全部放行，2 ranks 实读代码确认可过）。→ **唯一契合，选它**。

LL 路径名义上是给 decode 用的（`num_max_dispatch_tokens_per_rank` 静态上限），但 nano-vllm 的统一调度（Plan-2）保证每步 token 数 ≤ `max_num_batched_tokens`（测试配置 512），把上限设成它即可**用一条 LL 路径同时覆盖 prefill 和 decode**。M=512、hidden=2048、E=4、R=2 时 RDMA buffer 总量 ≈ 34MB（`LowLatencyLayout` 公式，config.hpp:131-183），可忽略。

## SM89（L40S）可行性盘点（逐文件读过源码的结论）

上游 `setup.py:130-139` 的 `DISABLE_SM90_FEATURES=1` 路径直接 `assert False, 'Not implemented'`（会禁掉 internode/LL）——**上游不支持非 Hopper 编 LL**，所以 nano-deepEP 必须自己做 sm_89 移植。手术量实测盘点：

| 文件 | SM90 依赖 | SM89 处置 |
|---|---|---|
| `internode_ll.cu` dispatch 内核（:128-462） | **无**（FP8 cvt `__nv_cvt_float2_to_fp8x2` SM89 原生支持；其余是 `__ldg`/warp copy/named barrier/协作组） | 原样复制 |
| `internode_ll.cu` combine 内核（:714-1138） | **重度 TMA**：`tma_load_1d/tma_store_1d`（cp.async.bulk）、mbarrier、`elect_one_sync`，发送侧三段流水 + 接收侧 producer/consumer | **手术**：退回 DeepEP 引入 TMA 前的 warp-copy 结构（M5 给逐段方案） |
| `ibgda_device.cuh`（496 行，RDMA 数据面：GPU 直写 mlx5 WQE/门铃） | **无**（纯字节序/原子/内存序 PTX，全 SM70+） | 原样复制；自带 `use_async_postsend` 分支 = CPU 辅助 IBGDA 也已支持 |
| `utils.cuh` | `elect_one_sync`（elect.sync 为 SM90 PTX，:318 有 lane0 fallback）；TMA 段 :336-427 | fallback 生效 + TMA 段删除 |
| `launch.cuh` | `cudaLaunchAttributeClusterDimension`（cluster 为 SM90 专有，:12-15） | 去 cluster attr，**保留 cooperative**（`cg::this_grid().sync()` 需要，dispatch:359 / combine:976） |
| 激进 PTX（`ld.global.nc.L1::no_allocate`） | 仅在 Hopper 验证过 | `DISABLE_AGGRESSIVE_PTX_INSTRS=1`（setup.py:147-150 对非 9.0 架构本来就强制） |

## 双后端策略（对冲 IBGDA 的管理员依赖）

IBGDA 两种开启方式（`docs/nvshmem.md:36-61`）都需要 root：改驱动 regkey `PeerMappingOverride=1`，或装 gdrcopy 内核模块（CPU 辅助模式）。当前两台机器都没配置。因此 nano-deepEP 定义统一 API、两个可切换 transport：

- **`transport="nccl"`（M2）**：`torch.distributed.all_to_all_single` 实现，数据面走 NCCL over RoCE 的 **GPUDirect RDMA**（dmabuf，M0 实测确认）——满足"必须基于 GPUDirect"；立即可跑，且作为 nvshmem 后端的**逐元素对拍 oracle**。
- **`transport="nvshmem"`（M5）**：忠实 DeepEP 的 IBGDA 内核，GPU 发起 RDMA（控制面也在 GPU 上）。这是本计划的最终形态，依赖 M0 闸门（管理员操作）。

两者输出布局/语义完全一致（`packed_recv_x [num_local_experts, num_ranks*M, hidden]`、`recv_count`、`layout_range` 的 pack2(count,begin) 编码），上层 MoE 无感切换。

## 全局验收策略

1. **通信层恒等式**（M2/M5，改编自 `DeepEP/tests/legacy/test_low_latency.py:60-181`）：dispatch → 恒等"假 GEMM" → combine 后，`combined_x == x * Σ(有效 topk_weights)`，bf16 `calc_diff < 1e-5`、fp8 `< 9e-4`；`recv_count[e] == Σ_ranks (all_topk_idx == e)`。
2. **模型层对拍**（M3/M4/M6）：同一份 tiny 随机权重，nano-vllm vs HF `Qwen3MoeForCausalLM`，单步 logits **argmax 全一致 + top-10 logprob ≤ 4 ulp**——直接复用 Plan-1-2-3 的 `tests/harness.py::compare_logprobs` 判据与噪声分析流程（bf16 ulp=0.0625 那套结论继续有效）。
3. **EP 等价性**：EP=2 双机 vs EP=1 单机（同权重同 prompt）greedy 128 token 逐 token 比对；分歧按"分歧点 gap 分位数"流程裁决。
4. **回归**：dense 单机模式下 Plan-1-2-3 的 42 项测试全部不回退（M1/M4 的引擎改动都要跑）。
5. **GPUDirect 证明**（M4/M6）：跑压测时在两机抓 `ethtool -S ens5f0np0` / `/sys/class/infiniband/*/ports/1/counters` 的 RDMA 计数增长，确认数据走 RoCE 直连口而非 bond0；NCCL 侧用 `NCCL_DEBUG=INFO` 日志中的 `[GDRDMA]` 字样存档。

## 关键代码位置速查（现状）

| 文件 | 关键点 |
|---|---|
| `nanovllm/engine/model_runner.py:26` | `dist.init_process_group("nccl", "tcp://localhost:2333", ...)` —— 要打破的硬编码 |
| `nanovllm/engine/model_runner.py:27` | `torch.cuda.set_device(rank)` —— 多机后每机只有 device 0 |
| `nanovllm/engine/model_runner.py:41-89` | TP 控制面：SharedMemory + Event（单机专用，跨机要换 gloo broadcast） |
| `nanovllm/engine/model_runner.py:110` | `num_kv_heads = hf_config.num_key_value_heads // self.world_size` —— EP 模式下按 world 除是**错的**，必须按 tp_size |
| `nanovllm/models/qwen3.py:91` | `Qwen3MLP`（dense）—— MoE 版的替换对象 |
| `nanovllm/models/qwen3.py:29` `layers/linear.py:23-24` `layers/embed_head.py:17-18` | 全部用 `dist.get_world_size()` 当 TP size —— M1 引入 parallel_state 后改为 tp 组 |
| `nanovllm/engine/model_runner.py:9` | 硬编码 `Qwen3ForCausalLM` —— M3 按 `hf_config.architectures` 分发 |
| `DeepEP/deep_ep/buffers/legacy.py:33-136` | Buffer 初始化协议（unique_id all_gather + runtime.sync），nvshmem 后端 Python 侧照抄 |
| `DeepEP/deep_ep/buffers/legacy.py:553-621/624-670` | `low_latency_dispatch/combine` 的完整签名与返回值 —— nano API 的蓝本 |
| `DeepEP/csrc/legacy/buffer.hpp:1456-1596/1598-1715` | C++ 侧 LL dispatch/combine（张量检查、双缓冲、launcher/hook 结构） |
| `DeepEP/csrc/legacy/config.hpp:102-188` | `LowLatencyLayout` 双缓冲布局与 size hint |
| `DeepEP/csrc/kernels/legacy/internode_ll.cu` | 内核本体（dispatch :128 / combine :714） |
| `DeepEP/deep_ep/buffers/legacy.py:609` | `assert nvshmem_qp_depth >= (M+1)*2` —— M=512 时默认 QP_DEPTH=1024 **不够**，必须设 2048（M5 的坑） |

## 实施结果（2026-08-13 完成）

- **[08-implementation-report.md](08-implementation-report.md)** —— 实现总结：改了哪些部分 /
  15 个坑（含 debug 过程）/ 每一步的实测效果 / 遗留与注意事项 / 测试套件与复现命令
- [artifacts/m0-report.md](artifacts/m0-report.md) —— M0 环境闸门实测报告

一句话结论：**M0-M6 全部完成。** tiny-Qwen3-MoE 在两台机器上端到端跑通，NCCL 与
IBGDA 两个后端都通过全部判据、且 combined_x 位级一致。

核心成果：**decode 形态（T=8）下每层 dispatch 从 NCCL 的 563.8 µs 降到 IBGDA 的
17.3 µs（33×）**；端到端 EP=2+IBGDA 反超单机 EP=1 **44%**（1485 vs 1034 tok/s）——
这推翻了计划原本的预期叙事，原因见报告 §3.5。

M0 闸门最初判定 IBGDA「都不行」，后经管理员开启 PeerMappingOverride 解除
（**热重载驱动、未重启**，见 `scripts/enable_ibgda.sh`）。

## 分阶段文档

- [01-m0-environment.md](01-m0-environment.md)
- [02-m1-multinode-runtime.md](02-m1-multinode-runtime.md)
- [03-m2-nano-deepep-nccl.md](03-m2-nano-deepep-nccl.md)
- [04-m3-moe-model.md](04-m3-moe-model.md)
- [05-m4-ep-integration.md](05-m4-ep-integration.md)
- [06-m5-ibgda-port.md](06-m5-ibgda-port.md)
- [07-m6-e2e-bench-and-beyond.md](07-m6-e2e-bench-and-beyond.md)

## 风险总表

| 风险 | 概率 | 后果 | 缓解 |
|---|---|---|---|
| 管理员不给开 PeerMappingOverride / 不装 gdrcopy | 中 | M5/M6 的 ibgda 后端无法运行 | M0 第一周就发请求；NCCL 后端保证 M4/M6 可交付；nano-deepEP 布局与 DeepEP 逐字节一致，环境就绪后仅换 so |
| RoCE 上 RDMA masked-atomic（dispatch 计数通知用）NIC 不支持 | 低-中 | ibgda dispatch 挂死在 recv_count 等待 | M0 用 `ib_atomic_bw` 实测 atomic；不行则 M5 把 `nvshmemi_ibgda_amo_nonfetch_add` 换成第二笔 `rma_p` 写计数（协议等价，DeepEP 的 count 本来就是 `-cnt-1` 单写语义） |
| combine 去 TMA 重写引入数值/性能问题 | 中 | 数值错或慢 | 接收侧 fp32 累加顺序固定按 topk 升序 → 与 NCCL 后端可位级对拍；性能以"打通"为先，报告里记录差距 |
| CUDA 12.8 用户态工具链装不顺（无 root） | 低 | 编不了扩展 | runfile `--toolkitpath=$HOME/cuda-12.8` 免 root；备选 pip 的 nvidia-cuda-nvcc-cu12 系列 |
| 双机库版本漂移导致数值不一致 | 中 | 对拍噪声无法归因 | M0 用 rsync 同步整个 .venv（同绝对路径、同 OS），版本锁死；漂移判据沿用 4-ulp 流程 |
| 两机活跃网卡离 GPU 拓扑不对称（gpu-02 是 NODE） | 确定 | GDR 带宽打折 | 记录为已知事实；100GbE 下瓶颈在网卡不在 PCIe 跨桥，对 tiny 模型无影响；报告里给实测数 |
