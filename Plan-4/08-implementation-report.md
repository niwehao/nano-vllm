# Plan-4 实现总结报告 · nano-deepEP + MoE + 跨机 EP

实施日期 2026-08-13。对应计划：[00-overview.md](00-overview.md) 及 M0–M6 分阶段文档。
结构对齐 [Plan-1-2-3/05-implementation-report.md](../Plan-1-2-3/05-implementation-report.md)。

## 0. 一句话结论

tiny-Qwen3-MoE（4 专家 top-2）已经在 **gpu-02(expert 0/1) + gpu-01(expert 2/3)** 两台机器上
端到端跑通，token 经 **RoCE 上的 GPUDirect RDMA** 逐层 dispatch/combine，输出与单机 EP=1
在 4-ulp 判据下等价；chunked prefill、抢占、投机解码三条既有路径在 EP 下全部复验通过。
**M5（IBGDA 忠实内核）已完成**：M0 闸门最初判定「都不行」，M6 因此先按保底方案用 NCCL
后端交付；随后管理员在两机开启 `PeerMappingOverride=1`（**热重载驱动，未重启**），闸门解除。
DeepEP legacy 的 `internode_ll` 内核已移植到 SM89（dispatch 零手术、combine 去 TMA），
双机验收全过，**三配置基准表填满**。

**最重要的一个数字**：decode 形态（T=8）下每层 dispatch 从 NCCL 的 **563.8 µs 降到
17.3 µs（33×）**；端到端 **EP=2+IBGDA 反超单机 EP=1 44%**（1485 vs 1034 tok/s）——
这**推翻了计划预期的叙事**（原以为 tiny 模型上 EP=2 必然慢于 EP=1），原因见 §3.5。

| 里程碑 | 状态 | 验收 |
|---|---|---|
| M0 环境闸门 | ✅ 7/7（第 7 项 IBGDA 初判「都不行」，管理员开启 PeerMappingOverride 后转为通过） | [artifacts/m0-report.md](artifacts/m0-report.md) |
| M1 多机运行时 | ✅ 4/4 | `tests/test_m1_multinode.py` |
| M2 nano-deepEP + NCCL 后端 | ✅ 6/6（单机退化版 + 双机） | `nanodeepep/tests/test_2rank.py` |
| M3 MoE 模型层（EP=1） | ✅ 5/5 | `tests/test_m3_moe_local.py` |
| M4 EP 集成（双机 4 专家） | ✅ 6/6 | `tests/test_m4_ep2.py` |
| M5 IBGDA 后端 | ✅ 双机验收全过（恒等式 / 确定性 / 与 NCCL 后端 combined_x **位级一致**） | `nanodeepep/tests/{test_ibgda_smoke,test_2rank}.py` |
| M6 基准 + 报告 | ✅ 三配置表填满；两后端端到端对拍 5/5 | `tests/bench_ep.py`、`tests/test_m6_backends.py` + 本文 |
| M4 复跑（IBGDA 后端） | ✅ 6/6 | `EP_TRANSPORT=nvshmem tests/test_m4_ep2.py` |
| 回归：dense 单机 42 项 | ✅ 4/4 套件全绿 | `tests/run_all.py` |

---

## 1. 改了哪些部分

### 1.1 新增的包：`nanodeepep/`（M2）

DeepEP legacy(V1) low-latency 路径的极简移植，定死 API、留两个可切换 transport。

| 文件 | 内容 |
|---|---|
| `__init__.py` | 与 `deep_ep.Buffer` 的取舍对照表；进程级单例 `init_ep_buffer/get_ep_buffer` |
| `buffer.py` | `NanoEPBuffer` 门面 + `EPHandle`（前 5 项与 deep_ep 的 handle 五元组同序同义）+ `pack2/unpack2`（照抄 `internode_ll.cu:411` 的高 32 位 begin / 低 32 位 count 编码） |
| `nccl_backend.py` | ~180 行纯 `torch.distributed` 实现的 dispatch/combine |
| `nvshmem_backend.py` | 占位，构造即抛错并指回 M0 报告与 M5 移植方案 |
| `tests/{common,test_2rank}.py` | 改编自 `DeepEP/tests/legacy/test_low_latency.py` 的恒等式验收 |

为什么只取 low-latency：本仓库 DeepEP 是 V2 主干（要求 SM90 / torch≥2.10 / NCCL≥2.30.4，
三项全不满足）。V1 三条路径里 intranode 是 NVLink 单机用的；normal internode 按
`rdma_rank = rank/8` 分组，2 ranks 会退化成"同节点"；只有 LL 模式下
`nvshmem_rank = rank, num_nvshmem_ranks = num_ranks`，每个 rank 就是一个独立 RDMA PE，
任意 rank 数都支持。

LL 名义上是 decode 专用（`num_max_dispatch_tokens_per_rank` 是静态上限），但
Plan-2 的统一调度保证每步 token 数 ≤ `max_num_batched_tokens`，把 M 设成它，
**一条 LL 路径同时覆盖 prefill 和 decode**。

NCCL 后端与 LL 内核的两处有意差异（都写在代码注释里）：

* LL 内核用静态 M 上限规避掉所有 CPU 同步；本后端要给 `all_to_all_single` 传 splits，
  而 splits 必须是 python list，所以**每次 dispatch 有且仅有一次 D2H 同步**（把 send/recv
  计数一起搬下来那一下）。这是 NCCL 后端的固有代价，也是 IBGDA 的意义所在。
* 加权归约写成 scatter 到 `[T,K,H]` 再沿 k 求和，而不是 `index_add_`：后者在 CUDA 上走
  原子加，同一行的多个副本累加顺序不定、fp32 舍入会飘；scatter+sum 每个 (token,k) 槽
  只写一次、求和顺序固定 → **同输入必位级同输出**（验收 4 依赖这一点）。

### 1.2 多机运行时（M1）

| 文件 | 改动 |
|---|---|
| `nanovllm/utils/parallel.py` | **新增**。parallel_state：`init_distributed` 建 nccl 主组 + gloo 控制面组 + TP/EP 子组；`get_tp_*/get_ep_*/get_cpu_group` |
| `nanovllm/config.py` | 新增 `ep_size / node_rank / master_addr / master_port / ep_transport`；断言 TP 与 EP 不同时 >1、EP 模式必须 `enforce_eager` |
| `nanovllm/engine/model_runner.py` | `init_process_group("nccl","tcp://localhost:2333")` → `parallel.init_distributed(config, rank)`；`set_device(rank)` → EP 模式恒 0；SharedMemory+Event 控制面 → gloo `broadcast_object_list`；KV 头数按 `get_tp_size()` 除；块数由 rank0 广播统一；`_ep_check` 探针 |
| `nanovllm/layers/{linear,embed_head}.py`、`models/qwen3.py` | `dist.get_world_size()/get_rank()` → `parallel.get_tp_size()/get_tp_rank()`；`all_reduce`/`gather` 加 `group=get_tp_group()`，gather 的 dst 用组内 rank0 的**全局** rank |
| `nanovllm/entry_worker.py` | **新增**。非 0 rank 的进程入口 |
| `scripts/` | **新增** `hosts.sh`（拓扑唯一事实源）、`env.sh`、`sync.sh`、`run2.sh`、`launch_both.sh`、`m0_nettest.sh`、`m0_nccl_test.py` |

并行语义定死：EP 模式下 `world = ep_size = 节点数`，**TP=1**，不做 TP×EP 交叉。
单机路径一个字节都没变（`master_addr` 默认仍是 `localhost:2333`）。

### 1.3 MoE 模型层（M3）

| 文件 | 改动 |
|---|---|
| `tools/make_tiny_qwen3_moe.py` | **新增**。seed=42 固定，生成 0.723B 的 tiny-qwen3-moe（hidden 2048、4 层、4 专家 top-2、moe_intermediate 512、vocab 151936 与 Qwen3-0.6B 同分词器） |
| `nanovllm/models/qwen3_moe.py` | **新增**。`Qwen3MoeSparseMoeBlock`（router）+ `FusedExpertsEP`（EP 切分的专家组，`forward_local`/`forward_ep` 两条路）+ DecoderLayer/Model/ForCausalLM 骨架 |
| `nanovllm/utils/loader.py` | 加 experts 分支，排在 packed mapping **之前**；EP 过滤在 `load_expert_weight` 内部 |
| `nanovllm/engine/model_runner.py` | `_MODEL_REGISTRY` 按 `hf_config.architectures[0]` 分发 |

attention / RMSNorm / rope / embed / lm_head 全部原样复用 `qwen3.py`；
`allocate_kv_cache` 与 attention 一行没改（MoE 只换了 MLP）。

三个 dtype 决策逐点抄自本机 transformers 5.14.1 的 `Qwen3MoeTopKRouter`：
① softmax 在 fp32；② `norm_topk_prob` 是 topk **之后**除以 topk 和，也在 fp32；
③ HF 随后把权重 cast 回 bf16 再乘、累加张量也是 bf16 —— **这一点 nano 有意不跟**，
保持 fp32，理由是要与 EP 路径 combine 的 fp32 归约对齐，这样 EP=1 与 EP=2 的差别只剩
"归约顺序/搬运"，不掺 dtype 策略差异。代价见 §3 判据 1：与 HF 差 2 ulp，在判据内。

### 1.4 EP 集成（M4）

* `FusedExpertsEP.forward_ep`：dispatch → **全量 bmm** → combine。
  对 `[L, R*M, H]` 全量算而不读 `recv_count`，因为读它要 D2H 同步，而整条 EP 路径的
  价值就在零 CPU 同步。垃圾行（本步没被写过的 slot）的结果不会进 combine——combine 只按
  handle 标注的有效区间取数，行与行独立。实测全量 bmm 恒 **150 µs/层**，与 T 无关。
* `model_runner.__init__` 里建 buffer 单例，`M = max_num_batched_tokens`，
  放在建模型**之前**（它占的显存要计进 `allocate_kv_cache` 的 used）。
* 形状护栏：`assert x.size(0) <= buf.M`。
* `examples/ep_generate.py` + `scripts/launch_both.sh` 一键双机。

### 1.5 基准与探针（M6）

* `tests/bench_ep.py`：场景 A（8 请求稳态 decode 448 token）/ 场景 B（第 10 步注入
  3×4000-token prefill），记 decode·prefill 吞吐、TBT p50/p99、RoCE 口流量。
* `nanovllm/models/qwen3_moe.py` 的 `NANOVLLM_EP_TIMING=1`：cuda event 打点，
  拆出 dispatch / MoE-GEMM / combine 三段，按本步 T 分组；`reset_ep_timing()` 丢弃预热。
* `NANOVLLM_EP_CHECK=1`：每步跨 rank 比对 input_ids/positions/slot_mapping/context_lens/
  logits_indices/logits 的校验和 —— 把"输入就不同"（搬运 bug）与"输入相同但算出来不同"
  （数值问题）分开。

### 1.5.1 移植时的裁剪原则与"小巧思"

**裁剪**：`nanodeepep/csrc/legacy/` 是从 DeepEP 复制的，但复制不等于全搬。删掉的：

| 删掉的东西 | 为什么与本项目无关 |
|---|---|
| `logfmt_encode` / `logfmt_check_amaxmin`（~115 行） | LogFMT 是 10 bit 动态对数量化，本项目恒 BF16。**必须删而不是留着不用**：它调 `tma_store_fence()`，函数模板里的非依赖名在定义处就要查找，留着编译不过 |
| combine 的 TMA 三段流水（发送侧 ~60 行、接收侧 ~160 行） | SM90 专有，见 §1.5 的手术 |
| `SWITCH_HIDDEN` 的 2560/3072/4096/5120/6144/7168/8192 分支 | 每多一个 hidden 就多实例化一整套内核模板，编译时间与 .so 体积翻倍；本项目只有 2048。原表以注释保留，要支持别的模型照着加回来 |
| mask / shrink 容错（`mask_buffer_ptr` 恒 nullptr）、各类 `*_stats` | 单机双卡的教学项目不需要 rank 容错与在线监控 |
| `zero_copy`（`get_next_low_latency_combine_buffer`） | 入口处 assert 挡掉，省一整条分支 |
| buffer.hpp 的 intranode / normal internode / layout / fabric / IPC 与 NVL barrier（1794 → ~230 行） | `num_nvl_bytes` 恒 0，那些分支根本进不去 |
| async / recv_hook / EventOverlap | nano 无通信重叠需求（M7 再说） |

**原则**：能不动的一个字节都不动（diff 最小 = 忠实性可审计），所有手术处加
`// [nano-deepEP] 修改原因：...`。删除也算手术，同样留注释说明删了什么、为什么。

**几个小巧思**（都是"少写代码"换来的）：

1. **靠 `-DDISABLE_SM90_FEATURES` 白拿一半手术**。上游为 Ampere 准备的这个宏，正好帮我们
   把 `utils.cuh` 里 `elect_one_sync` 切到 lane0 回退、整段 TMA 定义条件编译掉 —— 这两处
   本来要手工删。只需要额外覆盖两点：`launch.cuh` 的启动宏（要 cooperative、不要 cluster）
   和 `compiled.cuh` 的 FP8 分支（那条是给 SM80 的，SM89 原生支持 FP8）。
2. **launch.cuh 的"第三种组合"**。上游只有两条路：cooperative+cluster（SM90）或者退化成
   普通 `<<<>>>`。两条都不能用 —— cluster 是 SM90 专有，但 cooperative 必须留（两个内核都有
   `cg::this_grid().sync()`）。所以给出第三种：cooperative 但不带 cluster，7 行宏搞定。
3. **NCCL 后端的加权归约用 scatter+sum 而不是 `index_add_`**。后者走原子加，同一行多个副本的
   累加顺序不定；scatter 到 `[T,K,H]` 每槽只写一次、再沿 k 求和，顺序固定 →
   同输入必位级同输出。这让"确定性"从一条需要小心维护的性质变成了结构上的必然。
4. **单机 world=1 当快速迭代环境**。nanodeepep 的测试同一份代码在 world=1 下退化成
   自发自收，跑一轮几十秒、还能挂 compute-sanitizer；双机一轮要几分钟且要两端同步。
   M5 调试时先在 world=1 把 dispatch/combine 的数值正确性验完，再上双机 ——
   直接把"内核算错"和"跨机搬运错"分成了两个独立问题。
5. **`scripts/run2.sh` 的 `RUN2_LAUNCHER` 钩子**。想在两端都套上 compute-sanitizer 时，
   不用改测试脚本，`RUN2_LAUNCHER="compute-sanitizer --tool memcheck" ./scripts/run2.sh ...`
   即可。
6. **`sync.sh` 比对两边 `site-packages` 的清单哈希**。装了新包忘了 `--venv` 时，远端会以
   "缺某个 .so" 这种离题形式炸出来；提前提示一句省很多时间。
7. **`tools/check_comments.py`**：token 级比对基准版本的每条注释是否还在，找不到就退回原文
   子串比对（注释被整段"注释掉"保留时算通过）。改代码时最容易犯的错不是主动改注释，
   而是重写一段代码时把嵌在里面的注释连同代码一起换掉、自己毫无察觉。

### 1.6 技术路线：为什么不用 DeepEP V2 / NCCL Gin（实测调研）

计划总览里写了「V2 不可行」，但只给了版本号对照。这里补上**实测的**依据，
因为「既然 NCCL 已经有 GPU-Initiated Networking，为什么还要手工移植 IBGDA 内核」
是个合理的质疑。

**先分清两件事**：①直接用 DeepEP V2（它自带 Gin 后端）；②只借 NCCL 的 GIN 设备 API、
自己写内核。结论不一样。

#### ① 用 DeepEP V2 —— 硬阻塞在 SM90

README 的 Requirements：`Hopper (SM90) GPUs, or other architectures with SM90 PTX ISA
support` / `PyTorch 2.10 and above` / `NCCL 2.30.4 and above`。

| 要求 | 本机 | |
|---|---|---|
| SM90 PTX ISA | L40S = **SM 8.9 (Ada)** | ❌ 无法满足 |
| PyTorch ≥ 2.10 | 2.8.0 | 🟡 能升，但 flash-attn 2.8.3 按它编的，整栈连同 42 项回归基线要重来 |
| NCCL ≥ 2.30.4 | 2.27.3 | ✅ pip 上有 2.31.2 |

SM90 不是形式要求。统计 V2 内核（`deep_ep/include/deep_ep/impls/`）里的架构专有指令：

```
111  mbarrier
  4  cp.async.bulk
  2  setmaxnreg
  2  cluster
  1  elect.sync
```

**111 处 mbarrier** 说明整个 V2 是围绕 TMA + 异步屏障流水线设计的，不是零星使用。
Ada 一条都没有。「用 V2」实际等于把那 11 个 impls 头文件全部重写 —— 比移植 legacy
大一个量级。

#### ② 只借 NCCL 的 GIN 设备 API —— 理论上可行，但不是本项目

把 NCCL 2.31.2 的 wheel 下下来翻头文件，两个发现：

* `nccl_device/core.h:86-90` 的 GIN 后端枚举里有
  **`NCCL_GIN_TYPE_GDAKI = 3`（GPU Direct Async Kernel Initiated）—— 与 IBGDA 是同一套
  技术**（GPU 自己发起 RDMA），只是一个由 NCCL 提供、一个由 NVSHMEM 提供。
* `nccl_device/gin.h` 里 `multimem` / `cp.async.bulk` / `mbarrier` 出现 **0 次**。
  那 148 处 multimem 全在 `multimem__funcs.h`、`lsa_barrier__funcs.h`、
  `reduce_copy__funcs.h` —— 那是 **NVLink 多播**路径，不是网络路径。
  所以 GIN 的网络侧**没有静态架构门**，能不能用是运行时查的
  （DeepEP `csrc/kernels/backend/nccl.cu:88` 判的是 `props.ginType != NCCL_GIN_TYPE_NONE`）。

也就是说：**GIN 这条路不能断言不通**。我们这套 CX-6 Dx + RoCE + Ada 会不会报出可用的
GDAKI，没有实测过。

不走它的三条理由，按分量排：

1. **计划的目标就是忠实移植 DeepEP legacy**（总览原话：「关键代码直接复制」）。走 GIN
   的话 DeepEP 的内核代码一行都用不上，要照着一套全新 API 从零写 dispatch/combine——
   那是另一个项目。
2. **IBGDA 已在本机实测跑通**（§4.1），拿「已验证可用」换「可能可用」不划算。
3. **换 NCCL 有连带风险**：torch 2.8.0 按 NCCL 2.27 编译，顶成 2.31.2 会影响整个 torch 栈，
   而 42 项 dense 回归与 M1-M4 的全部验收都建立在当前这套上。

**留给 M7 的验证入口**（20 分钟可做完，不碰主环境）：在隔离 venv 里装 NCCL 2.31.2、
起双机 comm、查 `ncclCommGetDeviceProperties` 报的 `ginType`。是 `GDAKI` 就说明这条路
开着，是 `NONE`/`PROXY` 就彻底关掉这个疑问。

---

## 2. 踩过的坑（含 debug 过程）

### 坑 1 · 计划里 `layout_range` 的 begin 语义写错了

计划 03-m2 第 54 行写「NCCL 后端令 `begin = 0`（每个 (l,r) 段从段首连续放）」，
并说这与 ibgda 后端一致。**实读内核发现不是这样**：

`internode_ll.cu:260-262` 发送侧确实写到 `rdma_recv_x + local_expert*R*M + rank*M + slot`
（这是 RDMA **中转缓冲**的布局，按源 rank 分段）；但 `:408-411` 接收侧又做了一次压实：

```cuda
recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
```

所以 `packed_recv_x[l]` 的有效行是**前 `recv_count[l]` 行**，`begin` 是 `R*M` 维上的
**绝对下标**。DeepEP 自己的测试也是这么用的（`test_low_latency.py:141` 直接
`recv_x[begin_idx : begin_idx+count]`）。

按计划写会让 combine 从错误的位置取数。改法：nano 也压实在段首，`begin` 按 **rank 升序**
做前缀和。这是 DeepEP 语义的一个合法实例（DeepEP 的段序由 atomicAdd 竞争决定、是到达序，
不确定；nano 恒按 rank 升序），换来"同输入必位级同输出"。

### 坑 2 · `rsync -e "$SSH"` 被词法拆开，把选项当成了源路径

```bash
SSH="ssh -o BatchMode=yes"
R="rsync -a --info=stats1 -e $SSH"
$R "$REPO/" "$GPU01:$REPO/"
```

`$R` 展开后是 `rsync -a -e ssh -o BatchMode=yes ...`，rsync 只吃到 `-e ssh`，
剩下的 `-o BatchMode=yes` 成了源路径：

```
rsync: [sender] link_stat "/home/weihaoni/CodeRead/vllm/nano-vllm/BatchMode=yes"
       failed: No such file or directory (2)
```

改成 shell 函数 `R() { rsync -a -e "ssh -o BatchMode=yes" "$@"; }`。

### 坑 3 · `pkill -f` 自匹配，把自己的父 shell 杀了

`m0_nettest.sh` 起 perftest 服务端：

```bash
$SSH "pkill -f 'ib_write_bw'; sleep 0.3; nohup ib_write_bw ... &"
```

ssh 在远端跑的是 `bash -c "pkill -f 'ib_write_bw'; ... ib_write_bw ..."`，
这条**命令行本身**含 `ib_write_bw` → pkill 把自己的父 shell 干掉了。
现象是客户端只看到 `Couldn't connect to 192.168.100.1:18515`，
远端 `/tmp/m0_srv.log` 根本不存在（服务端从没起来）。

改法：`pkill -x`（按进程名精确匹配，不看命令行）+ `setsid nohup ... < /dev/null`
让服务端脱离 ssh 会话。

### 坑 4 · NCCL 按拓扑距离**默认关掉了** GDR

L3 首跑日志：

```
GPU Direct RDMA Disabled for GPU 0 / HCA 0 (distance 8 > 5)
Connected all rings, use ring PXN 0 GDR 0
```

这不是能力问题而是策略问题：GPU 与活跃网卡跨了 host bridge，NCCL 算出的距离是 8，
超过 `NCCL_NET_GDR_LEVEL` 的默认阈值 5(PXN) 就自动禁用。设 `NCCL_NET_GDR_LEVEL=SYS` 后：

```
NET/IB : GPU Direct RDMA Enabled for HCA 0 'mlx5_0'
Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA
Connected all rings, use ring PXN 0 GDR 1
```

**值得记一笔：开不开 GDR，带宽一模一样（都是 11.4 GB/s）。** 100GbE 下瓶颈在网卡，
不在 PCIe 跨桥那次拷贝。GDR 省的是延迟和 host CPU/内存带宽，不是峰值吞吐。

### 坑 5 · perftest 的 GPU 测试要 `--use_cuda_dmabuf`，否则走 peermem 路线必挂

`ib_write_bw --use_cuda=0` 服务端直接 `failed to create mr / Couldn't create IB resources`。
原因：不加 dmabuf 时 perftest 走老的 `ibv_reg_mr(CUDA VA)` 路线，那条路需要
`nvidia_peermem` 内核模块，本机没有。加 `--use_cuda_dmabuf` 后一次通过：

```
Calling ibv_reg_dmabuf_mr(offset=0, size=131072, addr=0x7cf93ce00000, fd=40) for QP #0
 65536   524176   0.00   91.60 Gb/s
```

91.60 Gb/s vs host 内存 92.56 Gb/s，差 1% —— GPUDirect 实锤。

### 坑 6 · 两台机器**不是位级一致**的，根因是 torch.compile/Triton

M1 的 worker 探针一开就报警：rank1 与 rank0 的 logits 校验和不同，但输入
（input_ids / positions / slot_mapping / context_lens / logits_indices）逐项相同、
输出 shape 相同、`absmax` 也相同 —— 只有求和不同。

**排查过程：**

1. 先怀疑是 EP 的搬运 bug。但探针已经证明输入完全一致，而且判据 2（dense EP=2 rank0 vs
   单机逐 token 全等，6/6）是过的 —— 说明 rank0 那条路没问题。
2. 做**对照实验**：把同一步 forward 分别在两台机器上**单机**跑（`ep_size=1`，
   完全不走任何 EP 代码），比 logprob：

   ```
   gpu-02 argmax 576, gpu-01 argmax 576, 共同候选 9/10, 最大偏差 0.176494
      token 576  : gpu-02 -1.256844  gpu-01 -1.245838  Δ=-0.011006
      token 18374: gpu-02 -3.319344  gpu-01 -3.495838  Δ=+0.176494
   ```

   两机独立跑同一件事就已经不一致了 → **与 EP 代码无关**。
3. 两机的差异清单：GPU 同型号（L40S）、venv 是 rsync 过去的（位级相同）、
   CUDA runtime 随 torch 打包 —— 只剩**驱动版本不同**（595.71.05 vs 610.57.04）。
   驱动会影响什么？`torch.compile` 在运行时用 Triton **生成并自动调优** kernel。
4. 关掉验证：

   | 配置 | argmax | 共同候选 | 最大偏差 | 完全相同 |
   |---|---|---|---|---|
   | torch.compile 开（默认） | 576 vs 576 | 9/10 | 0.176494 | ❌ |
   | `TORCHDYNAMO_DISABLE=1` | 576 vs 576 | 10/10 | **0.000000** | ✅ |

   nano-vllm 里有 4 处 `@torch.compile`（`layernorm.py:16,28`、`rotary_embedding.py:37`、
   `activation.py:8`），逐层累积就成了 ulp 级漂移。

**结论与处置**：判据 3 拆成两遍——`TORCHDYNAMO_DISABLE=1` 那遍是**硬判据**（必须 0 步不一致，
这一遍才真正在测搬运），默认那遍只做记录。M4 的判据本来就是 ulp 级的，不受影响。

顺带一个推论：EP=2 时 rank0 的输出**确实依赖 rank1 的算力**（rank0 的 token 被 dispatch 到
rank1 的专家上算完再 combine 回来），所以这条机器间漂移会进到最终结果里 —— 这正是 M4
用 ulp 判据而不是位级判据的原因。

### 坑 7 · `env MARK=1 python ...` 的标记不在命令行里，`pkill -f` 静默失效

`launch_both.sh` 原本用 `env nano_ep_worker=1 python -m nanovllm.entry_worker` 打标记，
清理时 `pkill -f nano_ep_worker`。但 `env VAR=1 cmd` 之后进程的 argv 是
`[python, -m, nanovllm.entry_worker, ...]`，**环境变量不在 `/proc/pid/cmdline` 里**，
而 `pkill -f` 匹配的正是 cmdline → 一个都杀不到。

这个坑让两件事静默失效：cleanup 从没生效过（只是因为 worker 总是正常收到 exit 才没暴露），
以及 M1 判据 5「kill worker 看 driver 反应」—— 测试显示 `rank0 在 68s 后以 rc=0 退出`，
其实 worker 压根没被杀、任务正常跑完了，什么都没测到。

改法：用命令行里真实出现的模块名，并用中括号防自匹配：

```bash
PAT='nanovllm[.]entry_worker'      # 匹配 "nanovllm.entry_worker"，但 pkill 自己那条
                                   # bash -c 命令行里写的是带中括号的版本，匹配不上
```

### 坑 8 · 调试开关只在 driver 侧生效 → 集合调用错配 → 双双卡死

修好坑 7 之后，判据 3 报出 **65 步不一致，比默认的 64 步还多，连 absmax 都不同了**。

原因：`launch_both.sh` 只透传了 `NANOVLLM_*`，没透传 `TORCHDYNAMO_DISABLE`，
于是变成"rank0 关 dynamo、rank1 开 dynamo"，两边算的根本不是一回事。

更早还撞过它的严重版本：`NANOVLLM_EP_CHECK` 一开始也没透传，只有 rank0 会去调那次
`all_gather_object`，rank1 直接走到下一个集合调用 —— 两边的集合调用序列错开一位：

```
[rank0] all_gather (来自 check_logits_across_ranks) ← 等不到对端
[rank1] broadcast  (来自 allocate_kv_cache 的块数广播) ← 等不到对端
双双 RuntimeError: Timed out waiting 180000ms
```

**教训：凡是会改变集合调用次数的 env，必须原样透传给所有 rank。** 现在
`launch_both.sh` 有一个显式白名单（`NANOVLLM_EP_CHECK / NANOVLLM_EP_TIMING /
NANOVLLM_GLOO_TIMEOUT / TORCHDYNAMO_DISABLE / TORCH_COMPILE_DISABLE`）。

### 坑 9 · 远端命令带 `&` 还不够，**本地的 ssh 也要放后台**

重写 `run2.sh` 时把本地那个 `&` 弄丢了：

```bash
$SSH "... setsid nohup python worker ... &"      # 远端的 & 只让远端 bash 立刻返回
sleep 3                                          # ← 永远走不到这里
```

远端 worker 确实起来了，但 ssh 自己还在等远端会话的 fd 全部关闭，driver 卡在这一行。
现象很迷惑：`pgrep` 看远端进程活得好好的，本机却毫无动静。加回 `... &" >/dev/null 2>&1 &`。

### 坑 10 · 这台机器上**根本没有 nvcc**

计划假设「nvcc 只有 13.3，与 torch 的 cu12.8 major 不匹配」。实际更彻底：
`/usr/local/cuda-13.3` 只有 `lib64/` 和 `targets/`（纯 runtime 库），
`/usr/local/cuda/bin` 这个目录都不存在。torch 的 cpp_extension 直接报
`/usr/local/cuda/bin/nvcc: not found`（另外还缺 `ninja`）。

处置：免 root 装用户态 CUDA 12.8 到 `~/cuda-12.8`（`--toolkitpath` + `--no-drm`），
补装 ninja，冒烟编一个 sm_89 扩展并 import 成功。M5 的工具链因此是**就绪**状态，
只差管理员开 IBGDA。

### 坑 11 · 测试的固定端口被上一次跑挂的孤儿进程占住

`run_all.py` 中途报
`EADDRINUSE ... port: 2333`。原因是我上一轮用 `timeout` 掐断了 `run_all.py`，
父进程死了但子 `gen.py` 还活着占着 TCPStore 端口。

单机测试没必要固定端口，`gen.py` 的 `--master-port` 默认改成 0 = 自动挑一个空闲端口。

### 坑 12 · 两个测试并发跑，撞显存

`assert max_blocks > 0` 挂在 `allocate_kv_cache`。不是代码问题——我把回归放后台又在前台
跑 M3，两个进程各按 `gpu_memory_utilization` 算余量，第二个自然算不出块。
GPU 测试串行跑。

### 坑 13 · 每层延迟剖面被冷启动污染

第一版剖面里 `T=512` 的 GEMM 显示 **2187 µs**，`T=256` 显示 5553 µs —— 但那个 bmm 是对
`[L, R*M, H]` 全量算的，**与 T 无关**，稳态只有 150 µs。原因是首次调用某形状时混着
NCCL 建链、cuBLAS 选算法、显存首触，一次上百毫秒，平均进 8 次调用就把整栏带偏。
加 `reset_ep_timing()`，预热之后清零。（Plan-1-2-3 的坑 9 是同一个教训的另一次发作。）

### 坑 14 · kill 测试的时机：负载太短，刀落下去时任务已经跑完

判据 5 第一版用 6 条 prompt × 512 token，`sleep 45` 后 kill —— 而那个负载正好跑 45s。
测试报 `rank0 在 45s 后以 rc=0 退出`，看起来像"没检测到故障"，其实是根本没制造出故障。
改成 16 条短 prompt × 3072 token（≈70s），30s 时下刀。修好后：
`rank0 在 33s 后以 rc=1 退出` —— 刀落下 3s 内报错，符合预期。

### 坑 15 · dense 模型没有 `num_experts`，EP buffer 不能无条件建

M1 要用 **dense** Qwen3-0.6B 跑 `ep_size=2`（这是等价性验收的关键：TP=1 时前向里没有任何
集合通信，rank0 必须与单机位级一致）。但 `init_ep_buffer` 要读 `hf_config.num_experts`，
dense config 没有这个字段。加 `getattr(hf_config, "num_experts", 0) > 0` 的条件。

### M5 阶段的坑（IBGDA 解禁 → NVSHMEM 环境验证）

这一段的坑有个共同特征：**全都是"看起来成功了、其实什么都没做"**。没有异常、没有报错，
只能靠"结果不对 → 反推有没有真的执行"这种方式抓。跟前面的坑 3、坑 7 是同一类。

#### 坑 16 · `pipefail` + `grep -q` 让 `rmmod` 被静默跳过（我写的脚本的 bug）

`enable_ibgda.sh --apply` 报告一切正常、`nvidia-smi` 也正常，但 `RegistryDwords` 还是空。
查时间戳才发现驱动**根本没重载过**：

```
系统启动               19:40:45
/sys/module/nvidia     19:40:58   ← 开机时加载的，一直没变
/dev/nvidia0           19:40:59
配置文件写入           23:41:16   ← 配置写了，驱动没动
```

根因：

```bash
set -uo pipefail
lsmod | grep -q "^$m " && { rmmod $m || { echo "失败"; return 1; }; }
```

`grep -q` 一匹配就退出并关掉管道 → `lsmod` 还在写，收到 SIGPIPE 退出 **141** →
`pipefail` 把整条管道判为失败 → `&&` 短路 → `rmmod` 一次都没跑。而且因为是短路不是报错，
**一句提示都没有**。当场复现验证过：

```
$ bash -c 'set -uo pipefail; lsmod | grep -q "^nvidia_uvm " && echo 跑 || echo "短路了 ${PIPESTATUS[0]}"'
短路了 141
```

修法：判断模块在不在改用 `[ -d /sys/module/$m ]`（不经管道）；卸完加硬断言
`[ -d /sys/module/nvidia ] && return 1`，不让后面的 `modprobe` 空转成"成功"。

#### 坑 17 · `ssh` 没有 `-t`，远端那台的 sudo 静默没执行

```
ssh 192.168.100.1 'sudo ./scripts/enable_ibgda.sh --apply'
→ sudo: a terminal is required to read the password
```

本机成功、远端失败，而检查脚本要两台都通过才算数。加 `-t` 分配伪终端即可。

#### 坑 18 · glibc 2.41 的 C23 数学函数与 CUDA 12.8 头文件打架

编任何 CUDA 扩展都会撞：

```
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(79): error:
  exception specification is incompatible with that of previous function "cospi"
  (declared at line 2601 of ~/cuda-12.8/include/crt/math_functions.h)
```

glibc 2.41 起把 C23 的 `sinpi/cospi/tanpi` 加进了 `<math.h>`，声明带 `__THROW`
（C++ 里是 `noexcept(true)`）；CUDA 12.8 写在那之前，这几个声明**没带**。而 C++ 里
异常规格属于声明的一部分，两次声明不一致就是硬错误。同一个 CUDA 头文件里
`fabs/fmin/fmax` 全都带 `__THROW`，唯独 pi 系列漏了。

修法：`scripts/patch_cuda_glibc241.sh`（幂等、自带备份与验证），给那 4 个声明补上
`__THROW` —— 与上游 CUDA 12.9 的修法一致。也可以直接换 CUDA ≥ 12.9。
写成脚本而不是手改一遍，是因为重装 toolkit 后会复发，手改的话下次就成了无头案。

#### 坑 19 · torch 的 `-D__CUDA_NO_HALF_OPERATORS__` 把 NVSHMEM 的归约模板打挂

```
nvshmem/include/non_abi/device/coll/reduce.cuh(95): error: no operator "+" matches these operands
```

`torch.utils.cpp_extension` 默认加 `-D__CUDA_NO_HALF_OPERATORS__` 等一串，把
`__half/__nv_bfloat16` 的算术运算符全禁掉；NVSHMEM 的设备端归约模板
（`perform_gpu_rdxn`，用 `+ * < >`）正好要用。修法是在 nvcc 参数末尾加对应的 `-U`
（排在 torch 的 `-D` 之后就能撤销）—— flash-attn 等项目同款处理。

#### 坑 20 · `build_ext --inplace` 的落点按 **cwd** 算

在 `nanodeepep/` 目录里跑 `python setup.py build_ext --inplace`，扩展名叫
`nanodeepep._C`，于是它去找 `nanodeepep/nanodeepep/` 而失败 —— 编译和链接其实全过了，
只死在最后一步拷贝。让 setup.py 自己 `os.chdir(HERE.parent)`，源文件用绝对路径。

#### 坑 21 · `sync.sh` 默认不同步 `.venv`，远端报了个离题的错

```
[rank1] ImportError: libnvshmem_host.so.3: cannot open shared object file
```

nvshmem 是在最后一次 `sync.sh --venv` **之后**才装的，后来的普通 sync 不碰环境。
报错形式（缺 so）离真正的原因（忘了同步环境）很远。修法：`sync.sh` 现在会比对两边
`site-packages` 的清单哈希，不一致就提示跑 `--venv`。

#### 坑 22 · IBGDA 下 put 的**源**缓冲必须在对称堆里，否则静默挂死

第一版冒烟测试里源缓冲用了普通 `cudaMalloc`：init 过了、barrier 过了，
`put_kernel` 一进去就再没出来，超时才被 kill。NVSHMEM 的 API 语义上允许源是本地
非对称内存，但 IBGDA 下是 GPU 直接发 RDMA，**源地址必须是注册过的 RDMA 内存**。
改成 `nvshmem_align` 分配即可。

#### 坑 23 · NVSHMEM 3.x 的 IBGDA 开关变了，DeepEP 用的那个会**静默退回 CPU 代理**

这个最险：冒烟测试**通过了**，三个尺寸的 put 全对，看起来大功告成。
但打开 `NVSHMEM_DEBUG=INFO` 一看：

```
NVSHMEM INFO Selected remote transport: ibrc      ← CPU 代理通道，不是 IBGDA！
```

DeepEP 的 `legacy.py:109` 设的是 `NVSHMEM_IB_ENABLE_IBGDA=1`，那是 NVSHMEM **2.x**
的开关；本机装的 3.7.2 要用 `NVSHMEM_REMOTE_TRANSPORT=ibgda`。只设老开关的话
NVSHMEM 默默选 ibrc，功能完全正常、性能却是另一回事 —— 要是没查这行日志就往下做，
后面整个基准的结论都是错的。

改对之后：

```
NVSHMEM INFO Successfully initialized the transport: IBGDA.
             It will be used for device-side APIs over IB.
```

**教训**：凡是"开启某个加速路径"的配置，都要找到运行时的**正面证据**（日志里那句
"Successfully initialized"），不能因为功能跑通就认为路径生效了。同一个教训在坑 4
（NCCL 按拓扑距离默认关掉 GDR）已经出现过一次。

### M5 内核移植阶段的坑

#### 坑 24 · CUDA 扩展编译的四连击

移植内核前先要能编出来。四个错误接连撞上，每个都不在计划的预案里：

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| a | `mathcalls.h(79): exception specification is incompatible with "cospi"` | glibc 2.41 加了 C23 的 `sinpi/cospi`（带 `__THROW`），CUDA 12.8 的同名声明没带 —— C++ 里异常规格属于声明的一部分 | `scripts/patch_cuda_glibc241.sh`（幂等、备份、自验证），照上游 CUDA 12.9 的做法补 `__THROW` |
| b | `reduce.cuh(95): no operator "+" matches these operands` | torch 的 `-D__CUDA_NO_HALF_OPERATORS__` 把 half/bf16 算术禁掉了，NVSHMEM 的设备端归约模板要用 | nvcc 参数末尾加对应的 `-U`（排在 torch 的 `-D` 之后） |
| c | `could not create 'nanodeepep/_C...so': No such file or directory` | `build_ext --inplace` 的落点按 **cwd** 算；在 `nanodeepep/` 里跑就去找 `nanodeepep/nanodeepep/` | setup.py 自己 `os.chdir(HERE.parent)`，源文件用绝对路径 |
| d | `namespace "at::cuda" has no member "getCurrentCUDAStream"` | 少 include | `#include <c10/cuda/CUDAStream.h>` |

值得单独说 a：这类"工具链版本组合"的坑最容易变成无头案 —— 重装一次 toolkit 就复发，
而当初怎么修的没人记得。所以写成**幂等脚本入库**，而不是手改一遍。

#### 坑 25 · `cudaErrorNoKernelImageForDevice` 是个误导性红鲱鱼

第一次跑 2-rank 报 `illegal memory access`，挂 compute-sanitizer 后满屏
`Program hit cudaErrorNoKernelImageForDevice (error 209)`，看起来像"内核没编成本机架构"。
查下来是**假警报**：`cuobjdump` 明确显示 `arch = sm_89`、dlink 命令里也有
`-gencode=arch=compute_89,code=sm_89`，而 memcheck 本身的 `ERROR SUMMARY: 0 errors`。
那些 209 是 sanitizer 探测 CUDA API 时的噪声，不是我们的错。

**教训**：sanitizer 的 `Program hit` 段是 API 错误回放，与 `ERROR SUMMARY` 的内存错误
是两回事，别混着看。

#### 坑 26 · DeepEP 与 NVSHMEM 3.7.2 的 **RC 队列数组排布不兼容**（M5 的主 bug）

**症状**：R=1（单机自发自收）全部验收通过 —— 恒等式 diff 1.14e-6、memcheck 0 错误；
R=2 一跑就 `illegal memory access`。

**定位过程**（这是整条线里最难的一个，六步收敛）：

1. `CUDA_LAUNCH_BLOCKING=1` → 错误落在 `internode_ll.cu:552`，即 **dispatch 的 launch 点**。
   dispatch 是**零手术**的 → 立刻排除掉 combine（我唯一动过刀的地方）。
2. `cuobjdump` → `arch = sm_89`、SM90 专有 SASS 助记符 0 处 → 排除"编错架构"。
   （途中被 sanitizer 的 `cudaErrorNoKernelImageForDevice` 误导过，那是 API 探测噪声，
   见坑 25。）
3. **R=1 vs R=2 的对照** → 内核数值逻辑没问题，问题一定在跨 rank 路径。
4. 写一个只有 32 线程的最小探针，**直接调 DeepEP 手写 WQE 的
   `nvshmemi_ibgda_put_nbi_warp`**，并把设备侧看到的 IBGDA 状态打出来。
   这一步很关键：之前的冒烟测试用的是 NVSHMEM **官方 API**
   （`nvshmemx_putmem_block`），它验证的是传输层；官方 API 全过**不代表**手写路径能过。
   探针结果：官方 API 三个尺寸全 OK，手写 WQE 崩 —— 问题精确隔离。
5. 状态本身是好的：`num_rc_per_pe=2`、`ndev=1`、`rcs=0x...1200`（有效指针）、
   `log2_cumem_granularity=29`、`peer_heap_base_remote` 两边互相匹配、
   lkey/rkey 索引都在范围内。所以不是"链接进来了但没初始化"。
6. 加 **`-lineinfo`** 重编，sanitizer 直接报出源码行：`ibgda_device.cuh:225`，
   `Invalid __global__ read of size 8 bytes, Address 0xffffffc28`。
   `0xffffffc28 ≈ 2^36` 反推出索引 ≈ 2^32 —— 典型的"空指针 + 大偏移"。
   再看崩溃的规律：**pe=1（目标 pe=0，索引 0）正常，pe=0（目标 pe=1，索引 2）崩**。

**根因**：RC 队列数组的排布在 NVSHMEM 版本间变了。

```cuda
// 本机 NVSHMEM 3.7.2 官方（include/non_abi/device/pt-to-pt/ibgda_device.cuh:1813）
idx = id * npes + pe;                                   // QP-major

// DeepEP legacy 的拷贝（ibgda_device.cuh:81-86）
idx = pe * num_rc_per_pe * ndev + id % (num_rc_per_pe * ndev);   // PE-major
```

后果**不是越界**（npes=2、num_rc_per_pe=2 时两种算法都落在 0..3），而是**拿到连向错误
对端的 QP**。更要命的是 NVSHMEM 只初始化 `pe != mype` 的槽位（官方实现里有
`assert(pe != nvshmemi_device_state_d.mype)`），**self 槽全是 0** —— 于是
`tx_wq.wqe` 是空指针，`ibgda_get_wqe_ptr` 算出来的地址就成了 `0xffffffc28`。

**修法**：把 `ibgda_get_rc` 改成与本机 NVSHMEM 一致的 QP-major 索引（`nanodeepep` 侧
带 `[nano-deepEP]` 注释说明为什么偏离上游）。改完之后 illegal memory access 消失。

**这条坑的普遍教训**：`ibgda_device.cuh` 是 DeepEP 从 NVSHMEM **手抄并改过**的内部头文件，
它对 NVSHMEM 运行时数据结构的**布局**有隐式依赖。struct 定义来自 NVSHMEM 自己的头
（所以字段偏移是对的、编译也过），但**索引约定**这种不写在类型里的东西不会被编译器检查。
DeepEP README 说"兼容 NVSHMEM 3.3.9 及以上"，实测在 3.7.2 上这一处已经不成立。
移植这类"手抄内部头"的代码时，**要逐个函数与目标版本的官方实现对照**，
不能只看编译通过。

#### 坑 27 · 门铃是**批量**按的，单发消息永远发不出去

修掉坑 26 之后，崩溃变成了**挂起**。查 `ibgda_submit_requests`：

```cuda
constexpr int kNumRequestInBatch = 4;
if (kAlwaysDoPostSend or (message_idx + 1) % kNumRequestInBatch == 0)
    ibgda_post_send(qp, new_wqe_idx);      // ← 按门铃
```

**每 4 条消息才按一次门铃**。我的探针只发一条、`message_idx=0` →
`(0+1)%4 = 1 ≠ 0` → 门铃从没按过 → 网卡不发 → 对端等不到 → 挂死。

**这不是移植 bug，是我探针写错了**：dispatch 内核传的 `message_idx` 是 `slot_idx`，
逐 token 递增自然凑够 4 条；最后由 `amo_nonfetch_add` 兜底
（它用的是 `ibgda_submit_requests<true>`，`kAlwaysDoPostSend=true` 强制按门铃）。
单发一条消息的独立测试必须显式写 `nvshmemi_ibgda_put_nbi_warp<true>(...)`。

**教训**：给"批量提交"的接口写单元测试时，要先确认它的 flush 语义，
否则测出来的"挂死"是自己造的。

#### 坑 28 · 两个后端的**确定性语义不同**，判据要分开写

修好门铃后，nanodeepep 双机验收里判据 1/2/3/5 全过，只有判据 4（确定性）失败：

```
第 1 轮哈希漂移: (1664565760, 12632836224) != (1651917312, 12632836224)
                  ^^^^^^^^^^ packed_hash 变了      ^^^^^^^^^^^ combined_hash 一模一样
```

**这不是 bug，是 DeepEP 的固有语义**：内核接收侧用
`atomicAdd(packed_recv_count + l, n)` 取 `begin`，各 rank 段的先后是**到达序**，
不确定；但 `layout_range` 会如实记录，所以 **combined_x 仍然位级稳定**。
而 nccl 后端是我们自己写的，段按 **rank 升序**首尾相接，连 packed 都稳定。

我在 `buffer.py` 的 docstring 里早就写明了这个差异，但测试判据当时没跟上 ——
判据 4 现在按 transport 分开：combined_hash 是两边共同的硬判据，
packed_hash 只在 nccl 后端下检查。

#### 坑 29 · 「两后端必须逐 token 全等」这个前提**过强**

07-m6 任务 1 写的是：「因 M5 验收 3 已证两后端 combine 位级一致，这里**必须逐 token
全等**；任何分歧 = 集成层 bug」。实测：greedy 128 只有 1/6 逐 token 一致，
分歧点在第 33 / 80 / 91 个 token。

**但那不是 bug。** 前提漏了一环：两后端的差别不只是"搬运方式"，还包括
**`packed_recv_x` 的行序**（坑 28）。同一个 token 在两个后端落在**不同的行下标**上，
而 MoE 的专家 GEMM 是 `torch.bmm(recv_x[L, R*M, H], w13.T)` —— cuBLAS 按 M 维分块，
行下标不同就落进不同 tile，split-k 的 fp32 归约顺序随之不同 → **1 ulp**。
combine 本身是位级一致的，它只是忠实地把这 1 ulp 传下去。

**用证据链代替放宽阈值**（`tests/test_m6_backends.py` 的 5 条判据）：

| 判据 | 结果 |
|---|---|
| 单步 logits（无自回归放大） | 候选集合 **10/10 全重合**，最大偏差 **1.00 ulp**，argmax 5/6（另 1 条是 gap=0 的精确并列） |
| **IBGDA 后端自身跑两遍** | **6/6 逐 token 全等** ← 排除"通信不稳定/buffer 复用错乱" |
| greedy 128 的分歧点 | 5 条全部落在最并列的 5.7~20.8%，0 条真分歧 |
| 短负载（64 token，放大不足） | **2/2 逐 token 全等** |
| IBGDA + 投机自身跑两遍 | **2/2 逐 token 全等** |

第 2 条最关键：如果差异来自搬运不可靠，IBGDA 自身重跑不可能稳定。它稳定 →
差异是**确定性的数值差异**。

**投机路径要单独说**：跨后端分歧出现在第 **4** 个 token，太早，不像放大。原因不同、
也更直接 —— 投机的接受判据是"草稿 token == argmax"，1 ulp 一旦让 argmax 翻面，
**当场**就从"接受"变成"拒绝"，不需要任何放大。所以投机路径改验
"IBGDA + 投机自身可复现"（通过）与"IBGDA 下开关投机等价"（M4 判据 5，通过）。

---

## 3. 每一步的效果（全部实测）

### 3.1 M0 · 网络与 GPUDirect

| 层 | 判据 | 实测 |
|---|---|---|
| L1 RDMA write（host） | ≥ 90 Gb/s | **92.56** Gb/s |
| L1 RDMA send（host） | ≥ 90 Gb/s | **92.58** Gb/s |
| L1' RDMA atomic fetch-add | 能完成 | **15.60 MiB/s、2.045 Mpps**（NIC = ConnectX-6 Dx） |
| L2 RDMA write（GPU 显存，dmabuf） | ≥ 80 Gb/s | **91.60** Gb/s |
| L3 NCCL all_to_all 256MB | busbw ≥ 10 GB/s | **11.41** GB/s（线速的 91%） |
| L3 GDR 证据 | 日志出现 GDRDMA | ✅（需 `NCCL_NET_GDR_LEVEL=SYS`，见坑 4） |
| L3 gloo 控制面 | < 1 ms | **0.26–0.29** ms/次 |

### 3.2 M2 · nano-deepEP NCCL 后端（双机，E=4 R=2 K=2 H=2048 M=512）

加权恒等式 `combine(dispatch(x)) == x · Σ_k w_k`，判据 `calc_diff < 1e-5`：

| T | 1 | 7 | 128 | 512 |
|---|---|---|---|---|
| diff | 0.000e+00 | 1.011e-06 | 1.164e-06 | 1.392e-06 |

确定性：同 seed 连跑 20 轮，`packed_recv_x` 与 `combined_x` 的哈希一字不变。
不等长 T（rank0=0、rank1=64）不挂、全 -1 的 token 输出恒为 0。

微基准（双机，cuda event，20 次平均）：

| T | 1 | 8 | 64 | 128 | 512 |
|---|---|---|---|---|---|
| dispatch µs | 400.8 | 563.8 | 563.1 | 571.7 | 731.8 |
| combine µs | 77.2 | 174.3 | 170.5 | 176.2 | 291.8 |
| 发送 MB | 0.01 | 0.07 | 0.52 | 1.05 | 4.19 |

**这张表是整个计划的核心论据**：T=8 时只搬 0.07 MB，100 Gb/s 下线上时间约 5 µs，
而 dispatch 花了 564 µs —— **99% 的开销与数据量无关**，是那一次 D2H 同步 + 3 次 NCCL
集合调用（计数 / 数据 / 元信息）的固定延迟 + python 开销。T 从 8 涨到 512（数据量 ×60），
dispatch 只从 564 涨到 732 µs。IBGDA 要消掉的正是这块固定延迟。

### 3.3 M3 · MoE 模型层 vs HF（tiny-qwen3-moe，6 条 prompt）

| 判据 | 结果 |
|---|---|
| 单步 logits argmax | **6/6 与 HF 完全一致** |
| top-10 logprob 最大偏差 | 0.03129 = **2.0 ulp**（本模型 \|logit\|max≈4.5 → 1 ulp = 0.0156），上限 4 |
| greedy 128 token vs HF | 1/6 逐 token 全等，5 条分歧全部落在近似并列位置（其中 3 条 gap 恰好 = 0.0000），0 条真分歧 |
| 逐层路由一致性 | 2400 个 (token, 层) 位置：14 个只是**顺序**不同（对输出零影响，加法可交换），12 个选中的专家不同 |
| 路由分歧的位置 | 第 k 名与第 k+1 名概率差中位数 0.0841；12 个分歧点全部 ≤ 0.0045，**最高分位数 2.58%** |
| 专家利用率 | layer0 `{3.1%, 32.4%, 27.6%, 36.9%}` / layer3 `{1.3%, 41.2%, 7.5%, 50.0%}`（随机权重，不设阈值） |
| EP 过滤单测 | 9 项全对（w13/w2 落位正确、expert 0/1 一个字节都没进本 rank、`mlp.gate.weight` 不被 experts 正则误捕获） |

**注意 ulp 的口径**：`harness.BF16_ULP = 0.0625` 是按 dense Qwen3-0.6B 的 logits 量级
（8~32）定的常数。tiny-MoE 是随机权重，\|logit\| 只有 ~4，ulp = 2⁻⁸×4 = 0.0156 —— 差 4 倍。
继续套 0.0625 等于把判据放松 4 倍、约等于没判。所以新增了 `harness.ulp_for(absmax)` 现算。

### 3.4 M4 · EP=2 双机 vs EP=1 单机

| # | 判据 | 结果 |
|---|---|---|
| 1 | 单步 logits | argmax 5/6 一致 + 1 条并列翻面（参照分布里 gap = 0.0000）；logprob 最大偏差 0.03129 = **1.0 ulp**（上限 4） |
| 2 | greedy 128 vs EP=1 基线 | 1/6 逐 token 全等，5 条噪声（gap 0~0.2 ulp，全在最并列的 5.5~11.1%），**0 条真分歧** |
| 3 | chunked prefill + EP | 1/2 全等，1 条噪声（0.2 ulp，最并列的 4.7%） |
| 4 | 抢占 + EP | EP=1 抢占 2 次 / EP=2 抢占 2 次；2/6 全等，4 条噪声，0 真分歧 |
| 5 | 投机 + EP | ngram k=2：提出 72 接受 57（**接受率 79%**），开关投机 **2/2 逐 token 完全一致** |
| 6 | GPUDirect 证据 | gpu-02 RoCE **+47.69 MB** / bond0 +0.12 MB（**405×**）；gpu-01 RoCE +47.68 MB / bond0 +0.03 MB（**1767×**）。理论估算 43.5 MB，实测 / 估算 = **1.10** |
| 7 | 回归 | dense 单机 42 项 4/4 套件全绿 |

判据 6 的估算式（第一性原理，计划原话是"与计数器对得上数量级即过"）：
`tokens × K × 0.5(半数目的专家在对端) × H×2B × 2(收发) × L层`
= `1328 × 2 × 0.5 × 4096 × 2 × 4` = 43.5 MB。

### 3.5 M6 · 三配置基准

**注意：M5 冻结，所以 ibgda 一列为空。** 负载 tiny-qwen3-moe，`max_num_batched_tokens=512`，
两轮丢弃预热后测。

| 指标 | EP=1 单机 | EP=2 + nccl | **EP=2 + ibgda** | ibgda/nccl | ibgda/EP=1 |
|---|---|---|---|---|---|
| 场景A decode tok/s（8 请求稳态 448 token） | 1034.5 | 888.4 | **1485.0** | 1.67 | **1.44** |
| 场景A TBT p50 ms | 7.60 | 8.98 | **5.37** | 0.60 | 0.71 |
| 场景A TBT p99 ms | 7.73 | 9.26 | **5.51** | 0.60 | 0.71 |
| 场景B decode tok/s（第 10 步注入 3×4000 prefill） | 1050.7 | 892.1 | **1381.6** | 1.55 | **1.31** |
| 场景B prefill tok/s | 1381.0 | 1172.5 | **1815.9** | 1.55 | **1.31** |
| 场景B TBT p50 ms | 7.58 | 8.92 | **5.35** | 0.60 | 0.71 |
| 场景B TBT p99 ms | 8.43 | 10.27 | **6.42** | 0.63 | 0.76 |
| 场景A RoCE 口流量 | 0 | 146.4 MB | 145.9 MB | — | — |

**计划的预期叙事被推翻了。** 07-m6 里写的是「tiny 模型上 EP=2 总吞吐**低于** EP=1
（复制 attention + 每层 2 次跨机通信 vs 零通信），本计划的价值在机制与延迟剖面」。
NCCL 后端确实如此（0.86×）；但 **IBGDA 后端比单机快 44%**。

原因在每层剖面里看得很清楚 —— 关键是 **EP=1 的本地路径并不便宜**：

| T | EP=1 本地全算 | nccl 合计 | **ibgda 合计** | ibgda dispatch/GEMM/combine |
|---|---|---|---|---|
| 8（decode） | 968.2 | 1194.3 | **348.8** | 144.7 / 84.3 / 119.8 |
| 329 | 1039.1 | 1292.3 | — | 412.0 / 86.2 / 149.5 |
| 512（prefill 块） | 1045.2 | 1475.9 | **730.6** | 426.6 / 86.1 / 217.9 |

`forward_local` 每层要对每个本地专家做 `torch.where` 找命中行，而 `torch.where` 的输出
尺寸是数据依赖的 → **每次都要 D2H 同步**，2 个本地专家 × 4 层 = 每步 8 次同步。
IBGDA 路径**一次 CPU 同步都没有**，且把两个专家的 GEMM 合成一次全量 bmm。
所以在这个 tiny 模型上，"跨机但零同步" 竟然打得过 "本地但频繁同步"。

（注意这不是说 EP 一定更快：换成专家多、本地路径能批量化的真模型，账要重算。
这里如实记录本配置的实测。）

### 3.5.1 nanodeepep 层微基准：IBGDA vs NCCL（双机，cuda event 20 次平均）

| T | nccl dispatch | **ibgda dispatch** | 加速 | nccl combine | **ibgda combine** | 加速 |
|---|---|---|---|---|---|---|
| 1 | 400.8 µs | **11.8** | **34.0×** | 77.2 µs | **10.9** | 7.1× |
| 8（decode 形态） | 563.8 | **17.3** | **32.6×** | 174.3 | **17.6** | 9.9× |
| 64 | 563.1 | — | — | 170.5 | — | — |
| 128 | 571.7 | **83.8** | 6.8× | 176.2 | **64.5** | 2.7× |
| 512（prefill 块） | 731.8 | **308.5** | 2.4× | 291.8 | **216.3** | 1.3× |

**这张表是整条线的核心成果。** 小 T 时 33 倍的差距，正是计划里预测的那条曲线：
NCCL 后端的开销与数据量**无关**（T 从 8 涨到 512、数据量 ×60，dispatch 只从 564 涨到
732 µs），因为它是一次 D2H 同步 + 3 次 NCCL 集合调用的固定延迟；
IBGDA 把这块固定开销整个消掉，耗时终于开始**随数据量走**。

每层 MoE 耗时剖面（µs/层/步，`NANOVLLM_EP_TIMING=1`，已剔除预热）：

| T | EP=1 本地全算 | EP=2 dispatch | EP=2 GEMM | EP=2 combine | EP=2 合计 | 增量 |
|---|---|---|---|---|---|---|
| 8（decode） | 968.2 | 743.9 | 149.3 | 301.0 | 1194.3 | **+226.1** |
| 329 | 1039.1 | 792.7 | 146.6 | 353.0 | 1292.3 | +253.1 |
| 384 | 1019.4 | 858.0 | 148.0 | 356.1 | 1362.2 | +342.8 |
| 480 | 1017.6 | 867.3 | 149.2 | 418.9 | 1435.5 | +417.8 |
| 512（prefill 块） | 1045.2 | 903.0 | 150.0 | 422.9 | 1475.9 | **+430.7** |

**怎么读这张表：**

* **EP=2 比 EP=1 慢 15%**，这与计划的预期叙事一致（"tiny 模型上 EP=2 总吞吐低于 EP=1"）。
  原因很直白：attention 被两机重复计算，MoE 层每步多 2 次跨机集合通信 ×4 层。
* 但慢得比想象少。EP=1 的 `forward_local` 并不便宜（968 µs/层）：它对每个本地专家做
  `torch.where` 找命中行，而 `torch.where` 的输出尺寸是数据依赖的 → **每次都要 D2H 同步**，
  2 个本地专家 ×4 层 = 每步 8 次同步。EP=2 的 dispatch 也有 1 次同步，但把两个专家的
  GEMM 合成了一次全量 bmm（恒 150 µs）。两边的固定开销互相抵掉了大半。
* **EP=2 的 GEMM 恒 150 µs、与 T 无关**，因为是对 `[L, R*M, H] = [2, 1024, 2048]` 全量算的。
  T=8 时算了 1024 行只有 16 行有用 —— 但换来整条路径少一次 D2H 同步，这笔账在 tiny 模型上划算。
* **dispatch 从 T=8 到 T=512 只涨 21%（744→903 µs）**，再次说明成本在固定延迟不在带宽。
  这是 IBGDA 的价值所在：DeepEP 在 H800+CX7 上 LL dispatch 的量级是几十 µs
  （`docs/legacy.md:34-39`），比这里低一个数量级。**把这条曲线压下来就是 M5/M6 的核心成果，
  也正是本环境目前拿不到的那一块。**

---

## 4. 遗留与注意事项

### 4.1 M5（IBGDA）：已完成

**闸门历史**：最初两机都是 `RegistryDwords: ""`、无 gdrdrv、无 sudo → 判定「都不行」，
M5 冻结、M6 按 NCCL 后端交付。随后管理员用 `scripts/enable_ibgda.sh --apply` 在两机开启了
`PeerMappingOverride=1`。**这一步不需要重启** —— 该参数是 nvidia 内核模块的加载期参数，
运行时改不了，但可以卸载模块再带参数装回去，而本环境恰好满足条件（无显示服务、
无 CUDA 进程、`nvidia` 的 holders 只有 `modeset`/`uvm`，唯一占用 `nvidia-persistenced` 可停）。
脚本做了前置检查、硬断言、可逆（`--revert`），只在 `nvidia.ko` 确实在 initramfs 里时才重建
引导镜像（那一步失败会影响开机，所以不吞错误）。

**移植清单**（`nanodeepep/csrc/legacy/`，手术处均带 `[nano-deepEP]` 注释）：

| 文件 | 处置 |
|---|---|
| `ibgda_device.cuh` | 原样复制，**只改 `ibgda_get_rc` 的索引**（坑 26） |
| `utils.cuh` | 原样复制；靠 `-DDISABLE_SM90_FEATURES` 让 `elect_one_sync` 走 lane0 回退、TMA 段条件编译掉 |
| `launch.cuh` | 给出"第三种组合"：cooperative 保留、cluster attr 去掉；`SWITCH_HIDDEN` 精简为 2048 |
| `compiled.cuh` | 恢复 FP8 分支（上游那条 stub 是给 SM80 的，SM89 原生支持） |
| `internode_ll.cu` | **dispatch 零手术**；combine 发送侧 TMA 流水 → `UNROLLED_WARP_COPY`，接收侧 producer/consumer → 每 warp 一个 token 的 fp32 归约；LogFMT 整体删除 |
| `nano_buffer.cu` | `LowLatencyLayout` 双缓冲 + Buffer 宿主侧（buffer.hpp 1794 → 230 行） |

**验收结果**：

```
nanodeepep 双机（transport=nvshmem, E=4 R=2 K=2 H=2048 M=512）
  恒等式 diff:  T=1 0.000e+00 | T=8 1.141e-06 | T=128 1.164e-06 | T=512 1.392e-06
  确定性:      combined_hash=2f0f9c880，20 轮一致
               —— 与 NCCL 后端的 combined_hash **完全相同**（跨后端位级一致）
  不等长 T / 全 -1 行 / recv_count / src_info：全过

M4 七项（EP_TRANSPORT=nvshmem）：6/6
  单步 logits EP=2 vs EP=1：argmax **6/6 一致**（NCCL 后端时是 5/6），偏差 1.0 ulp
  chunked prefill / 抢占 / 投机：全过
  RoCE 流量 48.55 / 48.46 MB（估算 43.5），RoCE:bond0 = 417× / 2095×

M6 两后端对拍：5/5（见坑 29）
dense 单机 42 项回归：4/4 套件全绿
```

**跨后端 `combined_hash` 完全相同**是这条线最强的正确性证据 —— 它不是巧合，是刻意设计
的结果：两个后端都按 **k 升序**做 fp32 归约（nccl 侧用 scatter+sum 而非 `index_add_`，
ibgda 侧是内核里的 `for i in 0..num_topk` 顺序）。

**没做的**：FP8 dispatch（`use_fp8` 通路在内核里还在，host 侧恒传 false）、
LogFMT（已删）、zero_copy（入口 assert 挡掉）、async/recv_hook、mask/shrink 容错。

### 4.2 已知的数值事实

* **两台机器在 torch.compile 开启时不是位级一致的**（坑 6）。这不是 EP 的问题，
  单机独立跑同一步就已经不一致。需要位级复现时加 `TORCHDYNAMO_DISABLE=1`。
* EP=2 时 rank0 的输出**依赖 rank1 的算力**（rank0 的 token 会被 dispatch 到 rank1 的专家上），
  所以上面那条漂移会进入最终结果。M4 的判据因此是 ulp 级的。
* tiny-qwen3-moe 是随机权重，router 的 top-k 边界并列极其常见（2400 个位置里
  第 k / 第 k+1 名概率差的中位数只有 0.084，最小为 0）。所有"argmax 翻面"的判定都必须
  配合并列度分位数看，不能只看是否相等。

### 4.3 没做的事（计划内、明确不承诺）

* **M7 全部**：DP attention（消除两机重复算 attention）、`return_recv_hook` 通信重叠、
  EP 下的 CUDA graph、FP8 dispatch、cached handle。入口点见
  [07-m6-e2e-bench-and-beyond.md](07-m6-e2e-bench-and-beyond.md)。
* **EP + CUDA graph**：`config.__post_init__` 强制 `ep_size > 1 → enforce_eager`。
* **EP 下的 `NANOVLLM_EP_CHECK` 跨 rank topk_idx 校验**：计划 05-m4 的"边界与坑"里提到
  两 rank 独立算 router 可能因 bf16 并列选出不同专家，建议加 checksum 比对或
  "rank0 广播 topk_idx"的一致化兜底。**这一项没实现**。理由：复制计算下每个 rank
  只对自己那份 token 负责（自己 dispatch、自己 combine 回来），rank1 选了不同专家
  只影响 rank1 自己那份被丢弃的结果，不污染 rank0 的输出。若将来做 DP attention
  （两 rank 处理不同的 token），这条就必须补上。
* **TP × EP 交叉**：`config` 里直接断言禁掉。
* **真模型（Qwen3-30B-A3B）**：bf16 ≈ 60GB > 2×46GB，且 128 个专家，需要量化 + 重算 M 上限。
  教学目标已由 tiny 模型达成。

### 4.4 过时的用户注释

按项目约定，**只标注、不修改**：原注释一字未动，在其相邻处另起新行加
`[OLD ↓]/[NEW]` 说明。本轮改动后 `model_runner.py` 里有 5 处与现状不符：

| 原注释 | 为什么过时 |
|---|---|
| `#给 TP 建立控制通道,并让 worker 进入待命状态。` | 现在同时服务单机 TP 与跨机 EP；通道从 SharedMemory 换成 gloo broadcast |
| `#效果是 8 张卡同时进入 run,各算各的那份权重分片,靠 NCCL all-reduce 汇总` | 那是单机 TP 的图景。EP 模式是 2 台机器各跑**同一批**（TP=1，权重不切，前向里没有 all-reduce），只在 MoE 层内做 EP |
| `#共享内存`（`call` 行尾） | 指令通道已改为 gloo `broadcast_object_list` |
| `#创建一块 1MB 的共享内存,取名 "nanovllm"。然后 barrier 等所有 rank 到齐。` | 所属代码已整体退役 |
| `#按名字 attach 到同一块内存` | 同上 |

后两条所属的代码行已删除，注释**原文连同它描述的那行代码**以 `[OLD ↓]` 注释块的形式
保留在原位置备查。

保全验证用 `tools/check_comments.py`（本轮新增，Plan-1-2-3 那轮靠同类脚本抓出过 3 行被
静默删掉的注释）：它把基准版本里的每条注释逐条在工作区里找，token 级找不到就退回原文
子串比对。当前结果：

```
$ .venv/bin/python tools/check_comments.py
  [嵌套保留] nanovllm/engine/model_runner.py: #创建一块 1MB 的共享内存,取名 "nanovllm"。…
  [嵌套保留] nanovllm/engine/model_runner.py: #按名字 attach 到同一块内存
=== 基准 HEAD：原始注释 379 行，缺失 0 行，嵌套保留 2 行 ===
```

---

## 4.5 调试方法论：这一线是怎么定位问题的

整条线踩的 25 个坑里，只有 3 个是"报错信息直接指向原因"的。其余全靠下面这几招。
按**先用哪个**排序：

### ① 时间戳 / 计数器 —— 判断"到底执行了没有"

最常用，也最便宜。适用于"报告成功但结果不对"这一大类（坑 3、7、16 全是）：

```bash
# enable_ibgda.sh 说驱动重载了，真的重载了吗？
系统启动               19:40:45
/sys/module/nvidia     19:40:58   ← 开机时加载的，没变
/dev/nvidia0           19:40:59
配置文件写入           23:41:16   ← 配置写了，驱动没动 → rmmod 根本没跑
```

同类的还有 M4 判据 6 用 `/sys/class/infiniband/*/ports/1/counters/port_xmit_data`
的增量证明流量确实走了 RoCE 口（RoCE:bond0 = 405:1）。

### ② 对照实验 —— 把"是谁的问题"变成二选一

坑 6 的例子：worker 的 logits 与 rank0 不一致。**先不猜**，把同一步 forward 分别在两台
机器上**单机**跑（`ep_size=1`，完全不走 EP 代码）：结果就已经差 0.176 → 与 EP 代码无关。
再关掉 `torch.compile` 重跑 → 位级全等 → 根因锁定在 Triton 的运行时自动调优。
两次对照实验，把一个"哪里都可能"的问题压成了一句话结论。

M5 的 world=1 vs world=2 也是同一招：world=1 全过 → 移植的内核数值正确 →
问题一定在跨 rank 路径。

### ③ 运行时的**正面证据**，而不是"功能跑通了"

坑 4（NCCL 按拓扑距离默认关 GDR）和坑 23（NVSHMEM 3.x 静默退回 ibrc CPU 代理）是同一个
教训的两次发作：**功能全对，但走的不是你以为的那条路**。这两次都只能靠打开
`NCCL_DEBUG=INFO` / `NVSHMEM_DEBUG=INFO` 去找那句
"GPU Direct RDMA Enabled" / "Successfully initialized the transport: IBGDA"。

规则：凡是"开启某个加速路径"的配置，必须找到运行时日志里的正面确认，
不能因为结果正确就认为路径生效了。

### ④ 分层验证 —— 先验环境，再验自己的代码

M5 刻意把"NVSHMEM/IBGDA 在这套硬件上能不能用"和"我们移植 internode_ll 有没有错"分成两步，
中间用一个 50 行的 device-initiated put 测试隔开。这一步过了之后，后面所有故障都能直接
归因到移植上，不用再怀疑环境。M0 的 L1/L2/L3 四层网络测试是同样的思路。

### ⑤ CUDA 内存错误的定位手段（按信息量递增）

`illegal memory access` 是最难查的一类，因为报错点与出错点通常不在一起。手段：

| 手段 | 给出什么 | 代价 |
|---|---|---|
| `CUDA_LAUNCH_BLOCKING=1` | 把异步报错**收敛到出错的那次 launch**，报错信息里直接带源文件行号 | 慢一些，几乎无成本 |
| `compute-sanitizer --tool memcheck` | 越界读写的**精确地址 + 线程 + 调用栈** | 内核慢 10-100 倍 |
| `cuobjdump --dump-sass / -elf-symbols` | 确认 `.so` 里到底编进了哪些架构、有没有 SM90 残留指令 | 秒级 |
| 设备侧 `printf` + `EP_DEVICE_ASSERT` | DeepEP 自带的断言会打印 rank/expert/src_rank，能定位到具体的 (expert, rank) 组合 | 需要重编 |

M5 的实际过程：先 `CUDA_LAUNCH_BLOCKING=1` 把错误收敛到
`internode_ll.cu:552`（dispatch 的 launch 点）→ 排除掉 combine（我唯一动过刀的地方），
把嫌疑范围缩到"零手术的 dispatch 在 R=2 时的环境配置"。再用
`cuobjdump` 确认 `arch = sm_89`、SM90 专有 SASS 助记符 0 处，排除"编错架构"。
memcheck 的 `ERROR SUMMARY: 0 errors` 又排除了普通越界 —— 三步下来，
剩下的解释只能是 IBGDA 设备状态本身。

**一个反面教材**：挂上 sanitizer 后满屏 `Program hit cudaErrorNoKernelImageForDevice`，
差点被带偏去查架构。那是 sanitizer 探测 CUDA API 的噪声（`Program hit` 段 = API 错误回放），
与 `ERROR SUMMARY` 的内存错误是两回事。

### ⑥ 二分：把可变量一个个钉死

M5 定位时依次钉死了：架构（cuobjdump）→ 内核数值正确性（world=1 全过）→
IBGDA 是否真的启用（NVSHMEM_DEBUG 日志）→ QP 数量（`num_rc_handles: 4`）→
环境变量清单（把 DeepEP 里出现的 `NVSHMEM_*` 全 grep 出来逐个对照，
补上漏掉的 `NVSHMEM_MAX_TEAMS` / `NVSHMEM_DISABLE_NVLS`）。每钉死一个，
剩下的可能性就少一半。

## 5. 测试套件说明与复现命令

### 5.1 目录

```
nano-vllm/
├── scripts/
│   ├── hosts.sh            双机拓扑唯一事实源（IP / 网卡名 / GID / 路径）
│   ├── env.sh              运行时 env（NCCL/gloo 锁直连口、GDR_LEVEL、CUDA_HOME…）
│   ├── sync.sh             gpu-02 → gpu-01 单向同步（--venv 连 7.5G 环境一起）
│   ├── run2.sh             双机跑同一条 python 命令（裸 env:// 两 rank）
│   ├── launch_both.sh      EP 双机一键：sync → 起 worker → 跑负载 → 清理
│   ├── m0_nettest.sh       L1/L1'/L2 perftest 分层实测
│   └── m0_nccl_test.py     L3 torch 栈双机 NCCL + gloo 延迟
├── nanodeepep/             nano-deepEP 包（见 §1.1）
│   ├── csrc/legacy/        DeepEP internode_ll 的 SM89 移植（M5）
│   ├── csrc/nano_buffer.cu Buffer 宿主侧 + LowLatencyLayout
│   ├── setup.py            sm_89 + -rdc=true + nvshmem device link + -lineinfo
│   └── tests/test_ibgda_smoke.py   NVSHMEM/IBGDA 环境冒烟 + 手写 WQE 探针
├── tools/make_tiny_qwen3_moe.py
├── examples/ep_generate.py
└── tests/
    ├── test_m1_multinode.py   M1 验收（4 项）
    ├── test_m3_moe_local.py   M3 验收（5 项）
    ├── test_m4_ep2.py         M4 验收（6 项）
    ├── test_m6_backends.py    M6 两后端端到端对拍（5 条证据链判据）
    ├── bench_ep.py            M6 基准
    ├── hf_ref.py              HF 参考实现（单步 logits / greedy / 路由探针）
    └── baselines/greedy_moe_ep1.json   M4/M6 的 EP 对拍基线
```

### 5.2 复现

```bash
cd ~/CodeRead/vllm/nano-vllm

# 0) 一次性：生成 tiny 权重 + 同步到 gpu-01（含 7.5G venv）
.venv/bin/python tools/make_tiny_qwen3_moe.py
./scripts/sync.sh --venv

# M0 环境闸门
./scripts/m0_nettest.sh                       # L1/L1'/L2 perftest
./scripts/run2.sh scripts/m0_nccl_test.py     # L3 双机 NCCL + gloo

# M2 nano-deepEP
.venv/bin/python nanodeepep/tests/test_2rank.py                    # 单机退化版(world=1)
./scripts/run2.sh nanodeepep/tests/test_2rank.py --transport nccl  # 双机

# M1 / M3 / M4
.venv/bin/python tests/test_m1_multinode.py
.venv/bin/python tests/test_m3_moe_local.py
.venv/bin/python tests/test_m4_ep2.py

# M5 IBGDA（先确认闸门，再跑）
./scripts/m0_ibgda_check.sh                         # 两机应显示"闸门: 通过"
sudo ./scripts/enable_ibgda.sh --apply              # 没通过的话，两机各跑一次（不需重启）
CUDA_HOME=$HOME/cuda-12.8 .venv/bin/python nanodeepep/setup.py build_ext --inplace
./scripts/run2.sh nanodeepep/tests/test_ibgda_smoke.py                    # 环境冒烟
./scripts/run2.sh nanodeepep/tests/test_2rank.py --transport nvshmem      # 内核验收
EP_TRANSPORT=nvshmem .venv/bin/python tests/test_m4_ep2.py                # M4 七项复跑
.venv/bin/python tests/test_m6_backends.py                                # 两后端对拍

# M6 基准（三配置）
.venv/bin/python tests/bench_ep.py --out tests/out/bench_ep1_clean.json --ep-size 1
MODEL=~/huggingface/tiny-qwen3-moe MNBT=512 MASTER_PORT=29500 TRANSPORT=nccl \
  ./scripts/launch_both.sh tests/bench_ep.py --out tests/out/bench_ep2_clean.json --ep-size 2
NANOVLLM_EP_TIMING=1 MODEL=~/huggingface/tiny-qwen3-moe MNBT=512 MASTER_PORT=29500 TRANSPORT=nvshmem \
  ./scripts/launch_both.sh tests/bench_ep.py --out tests/out/bench_ep2_ibgda.json \
  --ep-size 2 --ep-transport nvshmem

# 回归（dense 单机 42 项）
.venv/bin/python tests/run_all.py

# EP 端到端 demo
./scripts/launch_both.sh examples/ep_generate.py --max-tokens 32
```

**GPU 测试必须串行跑**（坑 12）。双机测试只能在 gpu-02（rank0）上发起。

### 5.3 调试开关

| env | 作用 |
|---|---|
| `NANOVLLM_EP_CHECK=1` | 每步跨 rank 比对输入/输出校验和，不一致打 `[EP_CHECK]` |
| `NANOVLLM_EP_TIMING=1` | MoE 层 cuda event 打点，拆 dispatch/GEMM/combine 三段 |
| `NANOVLLM_GLOO_TIMEOUT=<秒>` | gloo 控制面超时（默认 180；调试跨机挂死时调小） |
| `TORCHDYNAMO_DISABLE=1` | 关 torch.compile，两机才位级一致（坑 6） |
| `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` | 看 GDRDMA 是否生效 |
| `NVSHMEM_DEBUG=INFO` | 看 `Successfully initialized the transport: IBGDA`（**必查**，否则可能静默走 ibrc） |
| `NANOEP_IBGDA_DEBUG`（编译期，经 `NANOEP_EXTRA`） | 打印 lkey/rkey 查找的中间量 |
| `RUN2_LAUNCHER="compute-sanitizer --tool memcheck"` | 两端都套上 sanitizer |

以上开关**都会被 `launch_both.sh` 透传给 worker**（坑 8 的教训：只在 driver 侧开会造成
集合调用错配死锁）。
