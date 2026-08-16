# Phase 2.5 Step B · varlen CUDA graph 与投机解码共存 —— 方案

写于第一阶段验证通过之后、动手改代码之前。第一阶段的实测数据见
`07-phase2.5-stepb-report.md` 的"第一阶段"一节,本文只引结论。

> **⚠ 本文是动手前的方案,有一处已被实现推翻,原文保留不改以便对照。**
> 下面第五节"padding 行的 k 长度设 0"是**错的**:k 长度 0 会让 flash-attn 走
> early-exit 分支,那条分支把 `softmax_lse` 写到缓冲区外面(输出仍然逐位正确,
> 所以数值测试看不出来)。独立 review 指出、`compute-sanitizer` 复现。
> 最终实现改成"剩余行拆给多个 padding 槽,每槽 q 长度 ≤ 1+k 且 **k 长度 = q 长度**"。
> 全部经过见 `07-phase2.5-stepb-report.md` 的坑 10。

---

## 一、第一阶段结论(决定方案的三个事实)

| 问题 | 结论 |
|---|---|
| Q1 `flash_attn_varlen_func` 能否进 CUDA graph | **能**。捕获成功,replay 与 eager **逐位相同**,20 次 replay 无任何动态显存分配 |
| Q2 `max_seqlen_k` 传 `max_model_len` 是否仍正确 | **正确**。与传真实值的差异 ≤ 0.5 ulp(按张量整体尺度),且两者**与 fp32 参照等距** —— 不是"变差了",是 split-K 归约顺序不同。性能:短上下文最多 1.45× 变慢 |
| Q3 padding 行怎么填 | `cu_seqlens_k` padding 段与前一项相等(k 长度 0)**安全**:不 trap、不 NaN,输出**恰为全 0**;`block_tables` padding 值 -1 也**从不被读**(实测 -1 与 0 逐位相同) |

**Q4(任务书三问之外,自己加的,但它决定了整个方案)**:

> 一张 `max_seqlen_q=K` 的图,能不能跑"批内 q 长度参差不齐"的批?

**能,而且逐位相同。** 关键在于 `cu_seqlens_q` 是**设备张量**,和 `cu_seqlens_k`
一样可以 replay 前刷新;真正被烧进图的 host 标量只有 `max_seqlen_q` / `max_seqlen_k`。
FA 的 grid 只由 `max_seqlen_q` 定上界,每个 (m_block, seq) 自己按 `cu_seqlens_q` 早退。

实测 5 种组合(全命中 q=3×8、全未命中 q=1×8、一半命中、参差 1/2/3、短上下文参差),
**全部 bitwise 相同**;q_len=0 的空 seq 槽安全;padding sink 的 q_len > k_len 也安全。

这一条把"必须强制统一草稿长度"这个前提**直接推翻了**。

---

## 二、vLLM 是怎么做的(来自源码,给行号)

clone 于 `vllm-project/vllm@fe1c3171`(2026-08-16)。

| 任务书的问题 | vLLM 的答案 | 出处 |
|---|---|---|
| graph 分桶维度是什么 | **token 数**(`num_tokens`),外加 `num_reqs` 和一个 `uniform` 标志一起做 key | `vllm/forward_context.py:29-52` (`BatchDescriptor`) |
| draft 长度不固定怎么处理 | **padding 到 `1 + num_spec_tokens`**,但只对**新进入 decode 的请求**做。注释原文:"Pad new decode requests to uniform spec decoding size to preserve full cudagraph for this step" | `vllm/v1/core/sched/scheduler.py:933-947` |
| `max_seqlen_q` 被烧死怎么绕 | **不绕,是故意特化**。`max_query_len = uniform_decode_query_len if uniform_decode else num_tokens`,注释明说 `max_query_len=1` 会切到 "the optimized routine of FA2 for pure decode, i.e., Flashdecode + an optimization for GQA/MQA" | `vllm/v1/worker/gpu_model_runner.py:6000-6007` |
| ngram 返回空提议时那条 seq 走什么路径 | **不 padding**。running 请求的 `num_new_tokens` 退回 1(`num_tokens_with_spec == num_tokens`),整批不再 uniform → 从 FULL cudagraph 掉到 **PIECEWISE** cudagraph(attention 走 eager,其余仍在图里) | `vllm/v1/core/sched/scheduler.py:558-561`;判定 `gpu_model_runner.py:3990-4008`;分派 `vllm/v1/cudagraph_dispatcher.py:143-148` |

**vLLM 之所以能容忍"掉出 uniform"**,是因为它有 PIECEWISE 图这条中间路径。
nano-vllm 没有 torch.compile 切图,只有"整图 or 全 eager"两档 —— 所以掉出去的代价大得多,
这正是本任务要解决的问题。

**vLLM 选了方案 A(强制统一)。本方案不跟它走**,理由见下。这是我的设计选择,不是从 vLLM 抄的。

---

## 三、方案选择

任务书给了 A / B / C 三条路。

- **A 强制统一 draft 长度**(vLLM 的做法):没命中就填 padding token,q 恒为 k+1。
  对 nano-vllm 有一个 vLLM 没有的风险:**padding 草稿会被送进 `sample_speculative` 参与接受判定**。
  若 padding token 恰好等于模型 argmax,它会被"接受"并吐出去,直接污染输出。
  要防住就得在接受规则里加特判 —— 而接受规则是本项目唯一"数学上不允许任何误差"的部分
  (`05-implementation-report.md` 的 Phase 3 正确性表)。为了图化去动它,风险收益比不对。
- **B 按 (batch, draft_len) 二维分桶**:图数量翻倍,而且**根本不成立** ——
  批内每条 seq 的 draft 长度各不相同,`draft_len` 不是批级属性,没法当桶的维度。
- **C(采用)**:利用 Q4 的结论,**一张图直接容纳参差 q**。
  草稿长度保持 ngram 给出的原样,不伪造任何 draft token;
  padding 只发生在**张量缓冲区层面**(多出来的 q 行挂给一条 sink seq,`slot_mapping=-1`,输出丢弃)。
  接受规则、`postprocess`、回滚簿记**一行不改**。

方案 C 相对 A 的额外收益:没命中的 seq 不白算 k 个位置。
方案 C 相对 A 的额外代价:缓冲区仍按最坏情况 `B*(k+1)` 开,但 `cu_seqlens_q` 决定实际算多少行 ——
实测 T=8 与 T=24 的 attention kernel 时间比为 0.996×~1.005×(**padding 行在 attention 里基本免费**),
真实代价在 attention 之外的逐行 GEMM 上,要实测。

---

## 四、要改的文件与函数

### 4.1 `nanovllm/utils/context.py`

新增两个字段(**不动任何已有字段和注释**):

| 字段 | 用途 |
|---|---|
| `max_seqlen_q` / `max_seqlen_k` | 已存在,复用 |
| 无需新增 | —— |

实际上**不需要改 context**:varlen 图路径用的正是现有的 `is_prefill=True` 变长语义
(`cu_seqlens_q` / `cu_seqlens_k` / `slot_mapping` / `block_tables` / `logits_indices`)。
这是方案 C 的一个额外好处 —— 它复用现有 context 形态,不引入第四种语义。

> 若实现中发现确需新字段,在报告里记录。

### 4.2 `nanovllm/engine/model_runner.py`(主要改动)

| 函数 | 改动 | 理由 |
|---|---|---|
| `is_pure_decode` | **不动** | 坑 5:"能用快 kernel"与"能用图"正交。纯 decode 仍必须走 `flash_attn_with_kvcache` |
| `use_cudagraph` | **不动** | 仍只管老的 decode 图 |
| 新增 `is_spec_decode(seqs)` | 无 prefill、且至少一条 `num_scheduled_tokens > 1`、且全部 ≤ `1+k` | 选出"投机批"这一类 |
| 新增 `use_varlen_cudagraph(seqs)` | `not enforce_eager and is_spec_decode(seqs) and len(seqs) <= graph_bs[-1]` | 与上面正交,单独判断 |
| 新增 `capture_varlen_cudagraph()` | 按 `graph_bs` 再录一族图,每张 T = bs*(k+1) 行、bs+1 个 seq 槽(最后一个是 sink);烧 `max_seqlen_q=k+1`、`max_seqlen_k=max_model_len` | 方案 C 的核心 |
| `capture_cudagraph` | **原样保留**,末尾追加"若 k>0 再调 `capture_varlen_cudagraph`" | 不破坏已有三条路径 |
| `run_model` | 增加第三个分支 `use_varlen_graph`:把 `prepare_batch` 产出的 context 张量拷进 varlen 静态缓冲、补 sink、replay、`compute_logits(outputs[:T_bucket])` | —— |
| `run` | 选路增加 varlen 图分支;所有分支都记一次计数 | 硬性要求 1 |
| 新增 `self.exec_stats` | `{"graph_decode", "graph_varlen", "eager"}` 三个计数器 | 硬性要求 1:必须能打印出走图占比 |

选路最终形态(三者互斥,优先级从上到下):

```
pure_decode  -> prepare_decode + (老 decode 图 or eager with_kvcache)      # 原样
spec_decode  -> prepare_batch  + (新 varlen 图 or eager varlen)            # 新增
其它(含 prefill 混批) -> prepare_batch + eager varlen                      # 原样
```

### 4.3 `nanovllm/config.py`

新增 `varlen_cudagraph: bool = True` —— 一个能把新路径整体关掉的开关,
用于 A/B 对照和出问题时回退。登记进 `var.md`。

### 4.4 不改的文件

`scheduler.py`、`block_manager.py`、`sequence.py`、`attention.py`、`sampler.py`、`embed_head.py`
**一行不改**。方案 C 的全部改动都在 `model_runner.py` 的图捕获与选路里。

> 特别是 `attention.py`:varlen 图走的就是现有 `context.is_prefill` 为真的那条分支,
> 不需要新分支,也就不会碰到那条被保留的 `# prefix cache` 注释。

---

## 五、静态缓冲区设计

设 `k = num_speculative_tokens`,`Q = k + 1`,桶 `bs`,`T = bs * Q`。

| 缓冲 | 形状 | replay 前刷新 | padding 填法 |
|---|---|---|---|
| `input_ids` | `[max_T]` | 是 | 0 |
| `positions` | `[max_T]` | 是 | 0 |
| `cu_seqlens_q` | `[max_bs+2]` | 是 | 前 B+1 项为真实前缀和;第 B+2 项 = T(sink 吃掉剩余行);其后全部 = T(q_len 0) |
| `cu_seqlens_k` | `[max_bs+2]` | 是 | sink 及其后 k 长度全为 0(Q3 实测:输出恰为全 0,安全) |
| `slot_mapping` | `[max_T]` | 是 | **-1**(`store_kvcache_kernel` 已跳过,`attention.py:23`) |
| `block_tables` | `[max_bs+1, max_num_blocks]` | 是 | -1(实测从不被读) |
| `outputs` | `[max_T, hidden]` | 图写出 | —— |

host 标量:`max_seqlen_q = Q`(烧死)、`max_seqlen_k = max_model_len`(烧死,Q2 已验证)。

`logits_indices` **不进图**:`compute_logits` 在图外调用,用的是 `prepare_batch` 产出的真实
`logits_indices`,天然只取真实行。

---

## 六、可预见的副作用与受影响的调用方

1. **显存**:多一族图(最多 36 张)+ 一套静态缓冲。缓冲主项是 `outputs [max_T, hidden]`,
   `max_T = 512*3 = 1536`,bf16 hidden=1024 → 3 MB。图本身只存 kernel 调用序列。
   与两族图共用 `graph_pool`。风险低,但要在报告里给实测显存数。
2. **捕获耗时**:启动时多录 36 张图,冷启动变慢。要测。
3. **`enforce_eager=True`**:两族图都不录,行为完全不变。
4. **EP 模式**:`config.py:46-48` 已断言 EP 必须 `enforce_eager`,不受影响。
5. **TP>1**:worker 侧同样会走新分支。`run` 里 rank!=0 的早返回逻辑不变。本机单卡,无法实测(沿用 05 报告的遗留项)。
6. **`num_speculative_tokens=0`**:`Q=1`,varlen 图族**不录**(`is_spec_decode` 恒为假)。行为与今天完全一致。
7. **超桶**(`len(seqs) > graph_bs[-1]`):落回 eager varlen,与今天一致。
8. **`warmup_model`**:走 prefill 形态,不受影响(在 `capture_cudagraph` 之前调用)。

---

## 七、验收与正确性判据(沿用 `05-implementation-report.md:118-162`)

1. **单步 logprob 对拍**:varlen 图 vs eager,首选 token 必须完全一致,logprob 偏差 ≤ 4 ulp。
   阈值不因这次改动放宽;噪声量级仍按坑 4 的方法用数据定。
2. **接受数分布一致**:同一批输入下,图路径与 eager 路径的 `accepted` 必须逐条相同。
3. **三条老路径不回退**:纯 decode+图、纯 decode+eager、混批+eager 各自单独测。
   特别复核坑 5 —— `is_pure_decode` 与 `use_cudagraph` 仍是两个独立判断。
4. **走图占比 ≥ 90%**:`num_speculative_tokens=2`、8 条并发、稳态 decode,
   `exec_stats` 打印 `(graph_decode + graph_varlen) / total`。
5. **净收益**:端到端 tok/s、TBT p50/p99,与 Step A(关掉 varlen 图)对照。
   为负就如实写,不改判据。

---

## 八、风险与退路

| 风险 | 退路 |
|---|---|
| 烧死 `max_seqlen_q=k+1` 让投机批的 attention 变慢 | 已知 FA 对 `max_seqlen_q==1` 有 GQA 特化路径(vLLM 注释与本项目 Q4d 实测一致)。投机批 q 本来就 >1,用不上那条路径,不算损失 |
| 烧死 `max_seqlen_k=max_model_len` 在短上下文上慢 1.45× | 若实测端到端为负收益,退路是按 `max_seqlen_k` 再分几个桶(图数 ×桶数)。先不做 |
| 新路径出正确性问题 | `varlen_cudagraph=False` 一键关掉,退回 Step A |
