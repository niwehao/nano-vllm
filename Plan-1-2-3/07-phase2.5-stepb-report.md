# Phase 2.5 Step B 实现报告 · varlen CUDA graph 与投机解码共存

**环境**:NVIDIA L40S 46GB / torch 2.8.0+cu128 / flash-attn 2.8.3.post1 / driver 595.71.05
**模型**:Qwen3-0.6B(`~/huggingface/Qwen3-0.6B`,bf16),全部结果为真实模型跑出
**代码**:`nanovllm/` 2 个文件 **+187 −1** 行(`config.py` +4,`model_runner.py` +183 −1);
`tests/gen.py` +6;新增 `tests/test_phase25_varlen.py`、`tests/bench_varlen_graph.py`、
`tests/phase25_probes/`(第一阶段的 4 个探针脚本)、`var.md`

**一句话结论**:开着 CUDA graph 时,投机解码原本是**负收益**(2263 → 710 tok/s,
慢 3.2 倍),因为每个投机 step 都掉出图路径。Step B 之后投机批也能 replay 图,
变成 **3883 tok/s**,相对"投机关闭 + decode 图"这个此前最快配置净赚 **1.72×**。
走图 step 占比 **99.6%**(要求 ≥90%);正确性上图路径与 eager **逐位相同**(0.00 ulp)。

> **独立 review 抓到一个真问题**:第一版的 padding sink(q 长度 >0、k 长度 =0)会踩到
> flash-attn 的一个 early-exit 分支,那条分支把 `softmax_lse` 写到缓冲区外面。
> 我用 `compute-sanitizer` 复现并修掉了(坑 10)。**我自己的 25 项测试全绿也没发现它** ——
> 因为输出逐位正确,越界的是一个项目从不读的中间量。

> **环境说明**:任务书写的项目路径(`/Users/apple/...`)和"conda 的 myenv"在本机不存在。
> 实际用的是本机的 `/home/weihaoni/CodeRead/vllm/nano-vllm` 和项目自带的 `.venv`
> (`05-implementation-report.md` 的复现命令用的也是 `.venv/bin/python`)。

---

## 第一阶段:可行性验证

**不修改项目任何文件**,4 个独立探针脚本(现已归档到 `tests/phase25_probes/`)。

### Q1 · `flash_attn_varlen_func` 能被 `torch.cuda.graph` 捕获吗

**能。** batch=8,每条 q_len=1,`cu_seqlens_q = arange(0,9)`,k/v 从假 paged cache 读 + `block_table`:

| 检查项 | 结果 |
|---|---|
| 捕获 | 成功,无报错 |
| replay vs eager(同一批输入) | **逐位完全相同**(0.00 ulp) |
| 换一批数据再 replay | 输出跟着变,且与 eager **逐位相同** |
| 20 次 replay 前后 `memory_allocated` 变化 | **0 B**(无隐性动态分配) |

顺带验了地基:varlen(q_len=1)与现有 decode 路径 `flash_attn_with_kvcache` 的输出
**逐位相同**(0.00 ulp)—— 说明两条路数学上确实等价。

### Q2 · `max_seqlen_k` 传 `max_model_len` 是否仍正确、性能如何

**正确。** 6 个档位(真实 `max_seqlen_k` = 32/128/512/1024/2048/4096)分别传真实值与 4096:

| real_max_k | 32 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| 与传 4096 的偏差(按张量整体尺度算的 ulp) | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 |

0.5 ulp 是"不到一个 bf16 舍入步"。更强的证据是拿 fp32 手算 attention 当参照:

```
real=512 : vs fp32 参照 max|Δ| = 1.069e-03  相对 Frobenius = 2.114e-03
big =4096: vs fp32 参照 max|Δ| = 1.069e-03  相对 Frobenius = 2.176e-03
```

**两者与真值等距** —— 传偏大的 `max_seqlen_k` 不是"算得更差了",只是 split-K 归约顺序不同。

性能(batch=8,图内连录 100 次摊掉 launch 开销,A/B 交替 12 轮取最小值):

| real_max_k | 32 | 128 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| 传真实值 (us) | 9.11 | 9.56 | 18.50 | 23.73 | 34.07 | 135.19 |
| 传 4096 (us) | 13.26 | 13.72 | 20.78 | 30.08 | 32.41 | 134.72 |
| 比值 | 1.455× | 1.435× | 1.123× | 1.268× | 0.951× | 0.997× |

短上下文最多慢 **1.45×**(attention kernel 单次)。这是真实代价,但 attention 只占
decode step 的一小部分,端到端影响见第三节 —— 实测净收益远大于它,所以**没有**去做
"按 `max_seqlen_k` 也分桶"的退路。

### Q3 · padding 行怎么填才安全

`cu_seqlens_k` 的 padding 段与前一项相等(k 长度 0)、`block_tables` padding 值 -1:

| 检查项 | 结果 |
|---|---|
| 是否 trap | **否** |
| padding 行是否 NaN / Inf | **否**,输出**恰为全 0**(FA 主动写了 0,不是未初始化内存) |
| 真实行是否受影响 | 0.0625 ulp,argmax 全一致 |
| 图里 replay 带 padding 的批 | 同上,未 trap |
| 退路方案(k 长度 1、指向 block 0) | 也安全,输出有限非零 |

额外验证:`block_tables` 的 padding 填 -1 与填 0,输出**逐位完全相同** ——
说明这些位置**根本没被读**。

**当时的结论是"走退路没必要,直接用 k 长度 0"。这个结论后来被推翻了 —— 见坑 10。**

> Q3 只问了"输出会不会 NaN / 会不会 trap",这两问的答案确实都是"不会"。
> 但它**没有问内存安全**:k 长度 0 会让 FA 走 early-exit 分支,那条分支把 LSE 写到
> 缓冲区外面(`out` 的偏移是对的,所以输出仍然逐位正确,什么都看不出来)。
> 这条是独立 review 抓出来的,我用 `compute-sanitizer` 复现并修掉了。
> **"不 NaN、不 trap、输出逐位相同"三条全中,依然可以是越界的。**

### Q4 · (任务书三问之外,但它决定了整个方案)

> 一张 `max_seqlen_q=K` 的图,能不能跑"批内 q 长度参差不齐"的批?

任务书假定必须"强制统一 draft 长度"或"二维分桶"。但**这个前提是错的**:
`cu_seqlens_q` 和 `cu_seqlens_k` 一样是**设备张量**,replay 前可以刷新;
真正被烧进图的 host 标量只有 `max_seqlen_q` / `max_seqlen_k`。
FA 的 grid 只由 `max_seqlen_q` 定上界,每个 (m_block, seq) 自己按 `cu_seqlens_q` 早退。

实测:一张 `max_seqlen_q=3, T=24, B=9` 的图,跑 5 种 q 长度组合:

| 组合 | padding 行数 | vs eager |
|---|---:|---|
| 全部命中 q=3×8 | 0 | **逐位相同** |
| 全部未命中 q=1×8 | 16 | **逐位相同** |
| 一半命中 | 8 | **逐位相同** |
| 参差 1/2/3 | 9 | **逐位相同** |
| 短上下文参差 | 10 | **逐位相同** |

另外:`q_len=0` 的空 seq 槽安全;padding sink 的 `q_len > k_len` 也安全。

还有一个对方案很关键的性能事实:**T=8 与 T=24 的 attention kernel 时间比是
0.996×~1.005×** —— 在 attention 里,多算 16 行 padding 基本免费(decode 是访存瓶颈)。

### 决策点结论

Q1 过、Q2 过、Q3 过(且不用退路)、Q4 推翻了原有前提 → **继续第二阶段**。

---

## vLLM 是怎么做的(来自源码,给行号)

clone 于 `vllm-project/vllm@fe1c3171`(2026-08-16)。**以下四条全部来自源码,不是印象。**

| 任务书的问题 | vLLM 的答案 | 出处 |
|---|---|---|
| graph 分桶维度 | **token 数** `num_tokens`,与 `num_reqs`、`uniform` 标志一起做 key | `vllm/forward_context.py:29-52` |
| draft 长度不固定怎么处理 | **padding 到 `1 + num_spec_tokens`**,但只对**新进入 decode 的请求**。注释原文:"Pad new decode requests to uniform spec decoding size to preserve full cudagraph for this step" | `vllm/v1/core/sched/scheduler.py:933-947` |
| `max_seqlen_q` 被烧死怎么绕 | **不绕,故意特化**:`max_query_len = uniform_decode_query_len if uniform_decode else num_tokens`。注释明说 `max_query_len=1` 会切到 "the optimized routine of FA2 for pure decode, i.e., Flashdecode + an optimization for GQA/MQA" | `vllm/v1/worker/gpu_model_runner.py:6000-6007` |
| ngram 返回空提议时那条 seq 走什么 | **不 padding**。running 请求 `num_new_tokens` 退回 1,整批不再 uniform(`_is_uniform_decode` 要求所有请求 token 数都等于 `1+k`)→ 从 FULL cudagraph 掉到 **PIECEWISE** | 判定 `gpu_model_runner.py:3990-4008`;分派 `vllm/v1/cudagraph_dispatcher.py:143-148`;running 路径 `scheduler.py:558-561` |

vLLM 之所以能容忍"掉出 uniform",是因为它有 **PIECEWISE 图**这条中间路径(attention 切出去走 eager,
其余仍在图里)。nano-vllm 没有 torch.compile 切图,只有"整图 or 全 eager"两档 ——
掉出去的代价大得多,这正是本任务要解决的问题。

**vLLM 选的是方案 A。本实现选的是方案 C,这是我的设计选择,不是从 vLLM 抄的。**

**为什么不跟 vLLM 走 A**:nano-vllm 里 padding 出来的草稿 token 会被送进
`sample_speculative` 参与接受判定,一旦某个 padding token 恰好等于模型 argmax,
它会被"接受"并吐出去。要防住就得改接受规则 —— 而接受规则是本项目唯一
"数学上不允许任何误差"的部分(05 报告 Phase 3 正确性表)。为了图化去动它,风险收益比不对。
方案 C 的 padding 只发生在**张量缓冲区层面**,接受规则、`postprocess`、回滚簿记**一行不改**。

---

## 一、改了哪些部分

`nanovllm/` 下只动了 2 个文件,外加测试。

| 文件 | 改动 |
|---|---|
| `config.py` | 新增 `varlen_cudagraph: bool = True`(一键回退到 Step A) |
| `engine/model_runner.py` | `__init__` 加 `exec_stats` 计数与 `varlen_graph_ready` 标志,并在末尾把计数清零(排除 warmup 那次前向);新增 `is_spec_decode` / `use_varlen_cudagraph` 两个判断;新增 `run_varlen_graph`(含 `max_seqlen_k` 的 assert)/ `capture_varlen_cudagraph`;`run` 增加第三条选路分支;`exit` 释放新图 |
| `tests/gen.py` | 加 `--no-varlen-cudagraph`;把 `exec_stats` 并进输出 JSON(前缀 `exec_`) |

**没改**:`scheduler.py`、`block_manager.py`、`sequence.py`、`attention.py`、
`sampler.py`、`embed_head.py`、`context.py`。

`attention.py` 一行没改是方案 C 的直接结果:varlen 图走的就是现有
`context.is_prefill` 为真的那条分支,不需要第四种 context 语义。

### 选路(三者互斥)

```
pure_decode  -> prepare_decode + (老 decode 图 or eager with_kvcache)   # 原样,未动
spec_decode  -> prepare_batch  + (新 varlen 图 or eager varlen)         # 新增
其余(含 prefill 混批) -> prepare_batch + eager varlen                   # 原样,未动
```

`is_pure_decode` / `use_cudagraph` / `use_varlen_cudagraph` 是**三个独立判断**,
严格延续坑 5 的教训("能用快 kernel"与"能用图"正交)。纯 decode 批仍然走
`flash_attn_with_kvcache`,那条路的快 kernel 一点没丢。

### 静态缓冲区

设 `k = num_speculative_tokens`,`Q = k+1`,桶 `bs`,`T = bs*Q`,seq 槽 `nslot = 2*bs`
(`bs` 个真实槽 + 至多 `bs` 个 padding 槽)。

| 缓冲 | 形状 | padding 填法 |
|---|---|---|
| `input_ids` / `positions` | `[max_T]` | 0 |
| `cu_seqlens_q` | `[2*max_bs+1]` | 剩余行按每槽至多 `Q` 行切给 padding 槽,切完后其余槽 q 长度 0 |
| `cu_seqlens_k` | `[2*max_bs+1]` | padding 槽 **k 长度 = 它自己的 q 长度**(不是 0,见坑 10) |
| `slot_mapping` | `[max_T]` | **-1**(`store_kvcache_kernel` 跳过,`attention.py:23`) |
| `block_tables` | `[2*max_bs, max_num_blocks]` | padding 槽第 0 列填 **0**(读 block 0 的垃圾 KV,输出丢弃) |
| `outputs` | `[max_T, hidden]` | 图写出 |
| `pad_ramp` | `[2*max_bs+1]` | `arange * Q`,replay 时一次算出全部 padding 槽的 `cu_seqlens` |

烧死的 host 标量:`max_seqlen_q = Q`、`max_seqlen_k = max_model_len`。

**padding 槽的两条约束都不能松**(坑 10 的结论):

- **q 长度 ≤ Q ≤ 64**:图的 m 维 grid 烧死为 `ceil(max_seqlen_q/64) = 1`,
  一个槽里超过 64 行的部分永远不会被 kernel 写到,会留下未初始化显存;
- **k 长度 ≥ 1**:k 长度为 0 会让 FA 走 early-exit 分支,那条分支**写越界**。

---

## 二、踩过的坑

### 坑 1 · ulp 指标用错了,差点把 Q2 判成"不可行"

**现象**:Q2 第一次跑出 330~451 ulp 的偏差。按决策点表,这属于"有值被烧死",
方案该停在这里。

**排查**:写了 `tests/phase25_probes/diag_ulp.py`,不只报 ulp,还报 max|Δ|、
相对 Frobenius 误差、有多少分量不同、这些分量本身多大:

```
max|Δ| = 9.766e-04   张量尺度 max|b| = 3.574e-01   1ulp@scale = 1.953e-03
按整体尺度算的 ulp = 0.50
差异分量的量级: 中位 2.405e-02  最大 2.217e-01  最小 2.193e-05
逐分量相对误差:   中位 6.289e-03  最大 1.528e+00     <-- 就是它把指标顶上去的
```

**根因**:我按**逐元素**量级算 ulp。attention 输出必然有接近 0 的分量
(输出是 v 的凸组合,16384 个分量里总有几个碰巧接近 0),而 kernel 内部的累加是在
**张量尺度**(~0.35)上做的。一次 bf16 舍入落到一个量级 2e-5 的分量上,
逐元素算出来就是几百 ulp —— 那是指标的假象,不是误差。

05 报告的 ulp 判据用在 **logprob**(O(1)~O(10),良态)上,逐元素算没问题;
搬到原始 attention 输出上必须改成按整体尺度算。

**决定性证据**:拿 fp32 手算 attention 当参照,传真实值和传 4096 **与真值等距**
(相对 Frobenius 2.114e-03 vs 2.176e-03)。既然两个都在 bf16 舍入范围内,
就不存在"谁更对"。

**教训**:换了被测对象就要重新审视判据是否还成立。差点因为一个指标 bug 把可行的方案毙掉。

### 坑 2 · Q2 的第一版测试是空的

第一版 sweep 的两个 case 是手写的 `context_lens`,最大值**恰好都是 4096**
(= `max_model_len`)。于是"传真实值 vs 传 max_model_len"传的是同一个数,
bitwise 全 True,看起来完美 —— 实际上什么都没测。

改成按档位构造(`lens[0] = m` 保证 `real_max_k == m`)才真的覆盖到 32~2048。

**教训**:比对类测试要先确认"两边确实不同",否则 PASS 毫无信息量。

### 坑 3 · `torch.randn(bs, ...)` 同 seed 下不同 bs 的前几行并不相同

Q3 的 padded/unpadded 比对一开始报 343 ulp。原因是我用
`torch.randn(bs, H, D, generator=g)` 生成 q,bs=5 和 bs=8 各生成一次 ——
CUDA 的 randn 按**元素总数**做 Philox 分块,前 5 行根本不是同一批数。
比的是两组不同的输入。

改成恒按 `MAX_BS` 生成再切片,偏差立刻降到 0.0625 ulp。

### 坑 4 · 微基准不可信:量出了 0.489× 这种不可能的比值

第一版 kernel 计时给出"烧死 `max_seqlen_q=3` 比 `=1` 快 2 倍"。同样的输入、
只差一个 host 标量,不可能快 2 倍。

根因:每次测量都新建/销毁一张 CUDA graph,显存池和时钟状态一直在变。
重写为:所有图**一次性**建好 → 全部 warmup → A/B **交替**测 12 轮 → 取**最小值**。
之后 min 与 median 贴合(9.112 / 9.123),数据才可信。

顺带发现一个真事实:`max_seqlen_q == 1` 会让 FA 走**另一条 kernel 路径**。
后来在 vLLM 源码里找到了佐证(`gpu_model_runner.py:6004-6006` 的注释:
`max_query_len=1` → "Flashdecode + an optimization for GQA/MQA")。
这也是本方案**保留**老 decode 图、不用 varlen 图去接管纯 decode 批的原因之一。

### 坑 5 · 6 ulp 超阈值 —— 不是放宽阈值,是把偏差来源拆开

实现完成后,逐 step 对拍报出 **2/2505 行 argmax 不一致、最大 6.00 ulp**,超过 4 ulp 判据。

**没有动阈值**,而是把对拍改成**三方**,把两个混在一起的变量分开:

| 对比 | 差在哪 | 结果 |
|---|---|---|
| **g vs p** 图 vs eager,**同为 padded 形状** | 只差"图 replay vs 直接执行" | **2505/2505 逐位相同,0.00 ulp** |
| **p vs e** padded vs 真实形状,都是 eager | 只差"多算了几行 padding" | 1947/2505 相同,2 行 argmax 不同,6.00 ulp |

再加一个对照组,把因果钉死 —— 按"这一步有没有 padding 行"把 p vs e 拆开:

| 分组 | step 数 | 逐位相同 | argmax 不一致 | 最大 ulp |
|---|---:|---|---:|---:|
| **无 padding 行**(恰好落桶) | 42 | **702/702 (100%)** | **0** | **0.00** |
| 有 padding 行 | 103 | 1245/1803 (69.1%) | 2 | 6.00 |

**结论**:varlen 图本身**零误差**;全部偏差来自"缓冲区按最坏情况多算了几行",
即 05 报告坑 3 记的同一个机制(cuBLAS 按 M 维分块变化改变归约顺序)——
那里是"CUDA graph 把 batch 从 6 padding 到桶大小 8",这里是把 T 从 17 padding 到 24,
**一模一样的成因**,只是比例大一点。

再按坑 4 的方法核对分歧点性质:2 个分歧行的 top1−top2 间隙都是 **2.0 ulp**,
而 05 报告实测的位置中位间隙是 **56 ulp**、只有 4.6% 的位置间隙 ≤2 ulp。
分歧全部落在最并列的那几个百分点里 —— 逻辑 bug 不会只挑这种位置发作。

### 坑 6 · 端到端逐 token 比对在投机下**无法判定**

`check_equal_or_noise` 报"6 条真分歧",但每条的理由都是"无 logprobs,无法判定"。
根因是 05 报告遗留项 3:**投机路径下 logprobs 恒为 None**,判据拿不到 logprob
就没法把浮点噪声和真分歧分开,只能一律报成真分歧。

这正是任务书要求"不要用跑 128 步看 token 是否相同"当判据的原因。
处理方式:把它降级为 `[INFO]` 打印现象供人工复核,**并写明降级理由**,
判据由逐 step 三方对拍承担。没有为了让数字好看去删这项或改判据。

### 坑 7 · 抢占测试第一版报了 4/6 的**假失败**

把 05 报告 Phase 3 第 5 项(抢占 + 投机,6/6 逐 token 一致)照搬过来时,我顺手把
`eager=True` 去掉了 —— 想着"顺便把图也覆盖上"。结果报 4/6。

这不是 bug,是**判据被我改坏了**。原测试两边都 eager 是有道理的:它检验的是
"重算路径没有状态残留"这个**逻辑**性质,而逻辑性质只有在浮点噪声不存在时才可能
逐 token 全等。开着图时批的组成随抢占变化(桶、padding 行数都变),bf16 噪声必然出现,
即便代码完全正确也不会 6/6。去掉 `eager=True` 等于把"抢占逻辑"和"图的 padding 噪声"
两个变量搅在一起,谁也测不出来。

改法是**拆成两项**,而不是放宽任何一项:
- 抢占**逻辑**:两边 eager,要求 6/6 —— 恢复原判据,结果 **6/6 PASS**;
- 图在抢占下**是否正确**:用逐 step 三方对拍(不受自回归放大影响),
  结果 **305/305 逐位相同,0.00 ulp**。

### 坑 8 · 测试套件的顺序会把自己饿死

`suite_inprocess` 在本进程里建 LLM 并一直占着显存,之后再起 `gen.py` 子进程时
`allocate_kv_cache` 直接 `assert max_blocks > 0` 挂掉(两份 0.35 显存预算叠不上)。
把它挪到最后执行即可。

### 坑 10 · 独立 review 抓到的**写越界** —— 我自己的测试全绿也没发现

改完、25 项测试全过、基准也跑完之后,起了一个独立 agent 审代码。它报了一条 HIGH,
**是真的**,而且是我这套测试结构上看不见的那类问题。

**它的论断**:padding sink 的形态是 `q_len > 0` 且 `k_len == 0`。这会让 FA 走
`flash_fwd_kernel.h:537` 的 early-exit 分支,而那条分支算 LSE 偏移用的是
**padded** 公式(`:545`):

```cpp
const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh)
                                     * params.seqlen_q + m_block * kBlockM;
```

它**没有理会 `params.unpadded_lse`** —— 而正常路径(`:1023`)是理会的。
varlen 的 `softmax_lse` 偏偏是按 unpadded 布局分配的:
`torch::empty({num_heads, total_q})`(`flash_api.cpp:652`)+ 传 `/*unpadded_lse*/true`(`:688`)。
于是 `bidb = num_seqs` 这一格算出的偏移会超出缓冲区。

**我的核查过程(没有直接采信,也没有直接否定)**:

1. 读 FA 2.8.3.post1 的源码,`:545` / `:1023` / `flash_api.cpp:652,688` 四处**逐条对上**。
2. 算术:生产形态 `bs=8, k=2, h=16` → 缓冲 `h*total_q = 384` floats,
   sink 的偏移 `(8*16+bidh)*3 + row` = **384…429**,确实越界。
3. **第一次 compute-sanitizer 跑出 `ERROR SUMMARY: 0 errors`** —— 差点据此认为 review 说错了。
   原因是 PyTorch 的 caching allocator 只做一次大 `cudaMalloc` 再自己切分,
   **memcheck 看不见落在池内、但在子分配之外的越界**。
4. 加上 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 再跑,立刻出来:

```
========= Invalid __global__ write of size 4 bytes
=========     Address 0x7c90df630e54 is out of bounds
=========     and is 69 bytes after the nearest allocation at 0x7c90df630e00 of size 16 bytes
```

**为什么我原来的 25 项测试一项都没报**:nano-vllm 从不读 `softmax_lse`
(`flash_attn_varlen_func` 只返回 `out`),而同一分支里 `out` 的偏移是**对的**、
而且按行做了 mask,所以输出逐位正确。越界写落在图私有内存池里的某个临时块上,
是否可见完全取决于分配器当时的布局 —— **"逐位相同"证明不了内存安全**。
第一阶段 Q3/Q4 也没能覆盖:Q4a 给 sink 的 `k_len = pad_rows`(非 0),
根本没进 early-exit;唯一进了的是 Q4c(`k_len=0`),而它当时只检查了"有没有 trap / NaN",
**"没有 trap"不等于"没越界"**。

**修法**(不 patch flash-attn,只改调用侧):不再用一个大 sink,而是把剩余行
**拆给多个 padding 槽**,每槽至多 `q_max` 行,并且 **k 长度 = 它自己的 q 长度**、
`block_tables` 第 0 列指向 block 0。这样 `n_block_max ≥ 1 > n_block_min`,
**根本不进 early-exit 分支**,走正常路径(用 unpadded 偏移)。
槽数上限 `nslot = 2*bs`(真实 + padding),`cu_seqlens` 用一个预置的 `pad_ramp`
一次算出来,不引入 host 同步。

**验证**:`tests/phase25_probes/probe_sink_memcheck.py`,三种模式:

| 模式 | 形态 | memcheck(关掉 caching allocator) |
|---|---|---|
| `sink` | 旧设计,一个 sink,k 长度 0 | **Invalid __global__ write**(复现) |
| `padslots` | 新设计 | **0 errors** |
| `prod` | 复刻生产下标算法,扫 k∈{1,2,4,8} × num_seqs∈{1..32} × 命中/未命中/参差 = **108 个构型** | **0 errors**,且真实行与 padding 行全部有限 |

**顺带修掉的第二个问题**(review 的 S2):旧设计里 sink 的 `q_len` 可以远大于 64,
而图的 m 维 grid 烧死为 1,超过 64 的那些行**永远不会被 kernel 写**,留着上一次
replay 的残留(可能是 Inf/NaN)。当时它不构成正确性 bug(模型里所有算子都是逐行的,
padding 行的脏值传不到真实行,`compute_logits` 也只取 `[:num_tokens]`),
但那是个只靠"恰好没有跨行算子"撑着的性质。新设计每个 padding 槽 ≤ `q_max ≤ 64` 行,
全部会被写到,这个隐患一并没了。

**教训**:自己设计的判据只覆盖自己想得到的失效模式。这套测试对**数值**极其严格
(逐位相同、0 ulp),但对**内存安全**零覆盖 —— 而且第一次用 sanitizer 还被
caching allocator 骗了一次。以后凡是"往固定缓冲区里按自己算的下标写 / 让第三方 kernel
读写自己摆的形状",要把 `compute-sanitizer + PYTORCH_NO_CUDA_MEMORY_CACHING=1`
当成必跑项,而不是可选项。

### review 的其余意见与处理

| 编号 | 意见 | 处理 |
|---|---|---|
| S1 | padding sink 写越界(HIGH) | **已确认并修复**,见上 |
| S2 | sink 超过 64 行的部分从不被写,留未初始化显存(MEDIUM) | **已随 S1 的修法一并消除** |
| S3 | `max_seqlen_k` 烧死为 `max_model_len`,真实 k 超过它会被静默截断(LOW) | 采纳:在 `run_varlen_graph` 里加了一条 `assert context.max_seqlen_k <= max_model_len`,host 侧整数比较,零开销,把静默错误变成响的错误 |
| — | `exec_stats` 把 `warmup_model()` 那次前向记成了 eager 步 | 采纳:`__init__` 末尾把计数清零,让计数只覆盖真实 step |
| — | 报告写的行数 `+122 −2` 与实际不符 | 已按 `git diff --numstat` 更正 |
| — | 报告写"静态缓冲 `max_T = max_num_seqs × (k+1)`",代码用的是 `graph_bs[-1] × q_max` | 已更正措辞 |
| — | review 认为**正确的**部分:缓冲区下标覆盖完整无残留、`total ≤ bs*(1+k)` 有双重保证、block_table 越界列从不被读、图内存池共享安全、三条路径互斥且穷尽、KV 写路径不会污染 prefix cache、`compute_logits` 取行正确、teardown 无残留、TP>1 无新增破绽 | 与我自己的推演一致,无需改动 |

### 坑 11 · `varlen_graph_ready` 必须在 `warmup_model` 之前存在

`warmup_model()` 会先跑一次 `run()`,那时 `capture_varlen_cudagraph` 还没执行。
选路里若直接访问 `self.varlen_graphs` 会 AttributeError。
用一个在 `__init__` 开头就置 False 的 `varlen_graph_ready` 标志挡住。
(warmup 的 seq 是 `is_prefill=True`,`is_spec_decode` 本来也会返回 False,双保险。)

---

## 三、每一步的效果(全部实测)

### 正确性(`tests/test_phase25_varlen.py`,**25 项全过**)

**主判据 —— 逐 step 三方对拍**(8 条并发、k=2、256 输出 token、145 个投机 step、2505 行 logits):

| 对比 | 差在哪 | 结果 |
|---|---|---|
| **g vs p** 图 vs eager,**同为 padded 形状** | 只差"图 replay vs 直接执行" | **2505/2505 逐位相同,0.00 ulp** |
| **对照组** 无 padding 行的 step | 完全相同的计算 | **702/702 逐位相同,0.00 ulp** |
| p vs e 有 padding 行的 step | 多算了几行 | 1245/1803,2 行 argmax 不同,6.00 ulp |

**抢占场景下重跑同一套对拍**(k=4,KV block 卡死 7 块,触发 4 次抢占,26 个投机 step):

| 对比 | 结果 |
|---|---|
| **g vs p** 图 vs eager(同 padded 形状) | **305/305 逐位相同,0.00 ulp** |

抢占会让批的组成在 step 之间剧烈变化(桶、padding 行数都跟着变),是对图路径最不友好的场景。
图仍然零误差。

**其余各项**:

| 检查项 | 结果 |
|---|---|
| 投机接受判定两条路径一致(greedy 由 argmax 全等直接推出) | PASS(normal + preempt 两个场景) |
| 接受数 Step B vs Step A(端到端 128 步) | **616 vs 616,完全相同** |
| 路径1 纯 decode + 老 decode 图 | PASS(`graph_varlen=0`,老图照常用) |
| 路径2 纯 decode + eager(`enforce_eager`) | PASS(不录任何图) |
| 路径1 vs 路径2 单步 logprob | argmax 12/12 一致,**0.0 ulp** |
| 路径3 混批仍发生且走 eager | PASS(mixed=5) |
| 坑5 复核:kernel 判断与 graph 判断仍分开 | PASS(三个独立方法) |
| 关掉 `varlen_cudagraph` 确实退回 Step A | PASS |
| **prefix cache 未被 varlen 图的 padding 行污染** | argmax **6/6 一致,2.0 ulp** |
| **抢占 + 投机(eager)输出与不抢占一致** | **6/6 逐 token 完全一致**(触发 4 次抢占) |
| k=1 / 4 / 8 均能走 varlen 图 | PASS |
| bs=1 / bs=6(不落桶)均能走 varlen 图 | PASS |

> **污染专项的构造**:第一阶段 k=4 **开着 varlen 图**跑 64 步,第二阶段用
> `prompt + 前 32 个生成 token` 当新 prompt 且**关掉投机**,必然命中第一阶段写入的 block。
> 这是唯一能暴露"图的 `slot_mapping` padding 填错"的通道 —— 填错就会往 prefix cache
> 写垃圾 KV,而且当场不报错。原有的污染测试跑的是 eager,覆盖不到图路径。
>
> **抢占那一项两边都用 eager**,和 05 报告 Phase 3 第 5 项的构造完全一致。
> 该判据检验的是"重算路径没有状态残留"这个**逻辑**性质,而逻辑性质只有在浮点噪声不存在时
> 才可能逐 token 全等;开着图时批的组成随抢占变化,bf16 噪声必然出现,即便代码完全正确也
> 不会 6/6 —— 那时它测的就不是抢占逻辑了。图路径在抢占下是否正确,由上面那张
> 305/305 的逐 step 对拍表回答。这不是放宽判据,是把两个被混在一起的变量分开
> (第一版就是没分开,报了 4/6 的假失败)。

### 已有套件回归

| 套件 | 结果 |
|---|---|
| `tests/test_phase3_spec.py`(投机解码,14 项) | **14/14 通过** |
| `tests/test_m3_moe_local.py`(MoE EP=1 vs HF 逐位对拍,5 项) | **5/5 通过** |

### 走图占比(硬性要求 1)

k=2、8 条并发、稳态 decode:

```
exec_stats = {'graph_decode': 0, 'graph_varlen': 145, 'eager': 2}
scheduler  = {'steps': 146, 'prefill_only': 1, 'decode_only': 145, 'mixed': 0}
总占比      = 145/147 = 98.6%
稳态 decode = 145/145 = 100.0%      (要求 ≥ 90%)
```

基准脚本里同样打印(`走图%` 一列):generic 99.6%,repeat 99.7%。

### 性能(`tests/bench_varlen_graph.py`,8 条并发,每条 448 输出 token,greedy)

以下为**修完坑 10 之后**重测的最终数据(padding 槽方案,`nslot = 2*bs`)。

| 负载 | 配置 | 耗时 s | tok/s | tok/步 | TBT p50 | TBT p99 | 走图% | padding% | 接受率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generic | spec-off + decode 图 | 1.58 | 2263.3 | 8.00 | 3.51 | 3.76 | 99.8% | — | — |
| generic | k=2 **Step A**(投机走 eager) | 5.05 | 710.2 | 13.47 | 18.99 | 19.97 | 0.0% | — | 62.3% |
| generic | k=2 **Step B**(varlen 图) | **0.92** | **3883.4** | 15.32 | **3.91** | **4.15** | 99.6% | 13.7% | 67.2% |
| generic | k=2 全 eager | 5.05 | 709.9 | 13.47 | 19.03 | 20.36 | 0.0% | — | 62.3% |
| repeat | spec-off + decode 图 | 1.79 | 2006.4 | 8.00 | 3.91 | 4.19 | 99.8% | — | — |
| repeat | k=2 **Step A** | 5.90 | 607.0 | 10.54 | 18.63 | 19.77 | 9.4% | — | 70.7% |
| repeat | k=2 **Step B** | **1.30** | **2757.6** | 11.95 | **4.38** | **5.12** | 99.7% | 18.2% | 71.9% |
| repeat | k=2 全 eager | 6.94 | 516.2 | 10.54 | 18.77 | 20.80 | 0.0% | — | 70.7% |

三个关键读法:

1. **Step B vs Step A**:generic **5.47×**、repeat **4.54×**;
   TBT p50 18.99 → 3.91 ms,p99 19.97 → 4.15 ms。
2. **更要紧的一条**:开着图时,投机原本是**负收益** ——
   `spec-off + decode图 2263.3` vs `k=2 StepA 710.2`,**慢 3.2 倍**。
   因为每个投机 step 都掉出图路径,而图路径本身就值 3 倍多。
   Step B 之后 3883.4 tok/s,相对此前最快配置净赚 **1.72×**(repeat **1.37×**)。
   **这才是这次改动真正解决的问题。**
3. **padding 的代价**(硬性要求 2):缓冲区里 13.7%~18.2% 的行是 padding。
   代价上界可以这样卡:Step B 单步 3.91 ms(24 行缓冲、48 个 seq 槽)vs
   spec-off decode 图单步 3.51 ms(8 行)—— 差 **+0.40 ms(约 +11%)**,
   而且这 0.40 ms 里还包含了"varlen kernel 换掉 GQA 特化 decode kernel"
   和"seq 槽从 8 涨到 16"两部分,所以**纯 padding 行的代价比 11% 更小**。
   换来的是每步 15.32 tok 而不是 8.00 tok。**净收益为正,而且是大正。**

修坑 10 引入的槽数翻倍(`bs+1 → 2*bs`)本身的代价:
generic Step B 3926.2 → 3883.4 tok/s(**−1.1%**),repeat 2784.6 → 2757.6(−1.0%)。
换来的是消除一处写越界,值。

### MoE 有没有受影响(专项核查)

`tests/phase25_probes/probe_moe_graph.py`,模型 `~/huggingface/tiny-qwen3-moe`,EP=1:

| 档 | 配置 | 结果 |
|---|---|---|
| 1 | MoE + eager + spec=0 | ✓ 基线 |
| 2 | MoE + eager + **spec=2** | ✓ 与基线 **4/4 条逐 token 一致** |
| 3 | MoE + eager + spec=2 + `varlen_cudagraph=False` | ✓ 与基线 **4/4 条逐 token 一致** |
| 4 | MoE + **graph** + spec=0 | ✗ 捕获失败 |
| 5 | MoE + **graph** + spec=2 | ✗ 捕获失败 |

**第 4 档是归因的关键**:它 `spec=0`,而 `spec=0` 时 varlen 图**一张都不录**,
本次改动的代码一行都不执行。它照样失败,说明 **MoE 本来就进不了 CUDA graph**。

用 `git stash` 退回改动前的代码复核(沿用 05 报告坑 3/坑 5 的对照法):

```
PRE-CHANGE MoE + graph: FAILED -> AcceleratorError CUDA error:
                                  operation failed due to a previous error during capture
```

**改动前后表现完全一致**,而且失败点在 `capture_cudagraph`(旧的 decode 图捕获,
`model_runner.py:599`),不在新增的 `capture_varlen_cudagraph`。

根因在 `qwen3_moe.py:113-119` 的 `forward_local`:

```python
tok, k = torch.where(topk_idx == self.expert_start + e)   # 形状依赖数据
if tok.numel() == 0: continue                             # 拿设备数据做 host 分支
```

`torch.where` 产出的张量形状由数据决定,`tok.numel()` 又是一次设备→主机同步 ——
两者都是 CUDA graph 捕获不允许的。这是 MoE 层的固有属性,与 Step B 无关。

**结论:MoE 不受本次改动影响。** MoE 只能跑 `enforce_eager=True`,
而 `enforce_eager=True` 时 `varlen_graph_ready` 恒为 `False`,`run_varlen_graph`
永远不会被调用 —— 本次改动对 MoE 路径是**可证的空操作**。
另外 `tests/test_m3_moe_local.py`(MoE vs HF 逐位对拍)重跑 **5/5 通过**,无回归。

还有一条旁证:`test_m3_moe_local.py` 每次跑都会重写
`tests/baselines/greedy_moe_ep1.json`(M4/M6 的 EP=2 对拍基线)。
把重写后的文件与 HEAD 版本逐字段比:

```
token_ids 完全一致: True (6/6 条)
config 新增键: {'no_varlen_cudagraph'}   变化的 config 项: {'master_port'}
```

**生成的 token 一个都没变**,变的只有随机端口和 `gen.py` 新增的那个命令行开关名
(它进 `vars(args)` 才出现在 config 里)。该文件已 `git checkout` 还原,不计入本次改动。

> 顺带一个观察:第 2 档说明**投机解码本身在 MoE 上是工作的**(eager 路径),
> 只是享受不到图加速。要让 MoE 也能进图,得先把 `forward_local` 改成定长形态
> (例如按专家做 dense 的 masked GEMM,或预分配最大容量的 gather/scatter),
> 那是另一件事,不在本任务范围内。

### 冷启动代价

多录一族图(桶与老图相同,共用 `graph_pool`)。静态缓冲主项
`outputs [max_T, hidden]`,其中 `max_T = graph_bs[-1] × (k+1)` ——
注意是**最大的桶**,不是 `max_num_seqs`(两者在 `max_num_seqs` 不是桶值时不相等,
例如 `max_num_seqs=20` 时 `graph_bs[-1]=16`)。
测试用 `max_num_seqs=32`、k=2 时 `max_T=96`,缓冲不到 1 MB。

---

## 四、遗留与注意事项

1. **padding 引入的 bf16 噪声无法消除**。它是"缓冲区按最坏情况开"的固有产物,
   与 05 报告坑 3 记的老 decode 图 padding 同源。想进一步减小只能把桶切得更细
   (按 T 分桶而不是按 bs 分桶),代价是图数量上升 —— 收益(少量 ulp)不值这个复杂度,**没做**。
2. **`max_seqlen_k` 烧死为 `max_model_len`**,短上下文下 attention kernel 最多慢 1.45×。
   端到端已经是大幅净收益,所以没做"按 `max_seqlen_k` 分桶"的退路。
   若将来 `max_model_len` 调得很大而实际上下文很短,这一项值得重新量。
   反方向(真实 k 长度**超过** `max_model_len`)会被 FA 静默截断,已在 `run_varlen_graph`
   加了 `assert` 挡住(review 的 S3)。
2.5. **`probe_sink_memcheck.py` 是靠"复刻"而不是"调用"生产代码的**,所以它和
   `run_varlen_graph` 的下标算法有漂移的可能。根因是引擎跑不了
   `PYTORCH_NO_CUDA_MEMORY_CACHING=1`(CUDA graph 捕获要用 caching allocator 的图内存池)。
   更好的做法是把填 buffer 的那段逻辑抽成一个纯函数,两边共用 —— **没做**,
   因为抽出来会让生产路径多一层调用,而这条路径每个 step 都跑。
3. **投机 + logprobs 仍不能同时用**(05 报告遗留项 3,本次未动)。
   直接后果是端到端逐 token 比对在投机下无法判定分歧性质(坑 6)。
4. **TP > 1 未实测**(本机单卡)。worker 侧会走同样的新分支,设计上自洽,但没有真机验证。
   沿用 05 报告的同名遗留项。
5. **超桶行为**:`len(seqs) > graph_bs[-1]` 时落回 eager,与改动前一致。
6. **EP / MoE 模式不受影响**:`config.py` 已断言 EP 必须 `enforce_eager`,两族图都不录。
   MoE(EP=1)因 `forward_local` 的数据依赖形状,**本来就进不了 CUDA graph**
   (改动前后一致,已用 `git stash` 复核),只能 `enforce_eager=True` 跑,
   于是本次改动对它是可证的空操作。详见上面的 "MoE 有没有受影响" 一节。
7. **本次没有发现新的过时用户注释**。改动集中在新增函数里,
   `tools/check_comments.py` 核对结果:**原始注释 640 行,缺失 0 行**。

---

## 五、测试与复现

```bash
cd nano-vllm

# 第一阶段探针(不依赖项目代码,只要 torch + flash_attn)
.venv/bin/python tests/phase25_probes/phase1_varlen_graph_probe.py   # Q1/Q2/Q3
.venv/bin/python tests/phase25_probes/phase1_q4_ragged_q.py          # Q4 参差 q
.venv/bin/python tests/phase25_probes/phase1_timing.py               # 干净的 kernel 计时
.venv/bin/python tests/phase25_probes/diag_ulp.py                    # 坑 1 的排查脚本

# 第二阶段
.venv/bin/python tests/test_phase25_varlen.py     # 正确性 25 项
.venv/bin/python tests/bench_varlen_graph.py      # 性能对照表

# 内存安全(坑 10)。两个都要:少了 PYTORCH_NO_CUDA_MEMORY_CACHING=1,
# PyTorch 的 caching allocator 会把池内越界完全挡住,memcheck 报 0 errors。
PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck \
  .venv/bin/python tests/phase25_probes/probe_sink_memcheck.py prod    # 应为 0 errors
PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck \
  .venv/bin/python tests/phase25_probes/probe_sink_memcheck.py sink    # 旧设计,应报越界

# MoE 专项(改动是否影响 MoE)
.venv/bin/python tests/phase25_probes/probe_moe_graph.py
.venv/bin/python tests/test_m3_moe_local.py       # MoE vs HF 逐位对拍,5/5

# 已有套件回归
.venv/bin/python tests/test_phase3_spec.py        # 14/14
```

> 注:引擎本身没法直接跑在 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 下(CUDA graph 捕获依赖
> caching allocator 的图内存池)。所以 `prod` 模式**复刻了 `run_varlen_graph` 的下标算法**
> 而不 import nanovllm,扫 k∈{1,2,4,8} × num_seqs∈{1…32} × 命中/未命中/参差 = 108 个构型。
> 改动 `run_varlen_graph` 的填法时,这个探针要跟着改。

`tests/test_phase25_varlen.py` 的判据分层:

| 判据 | 用途 | 严格程度 |
|---|---|---|
| 逐 step 三方对拍(g vs p / p vs e / 对照组) | **主判据**,normal + preempt 两个场景 | 图本身必须**逐位相同**,0 ulp |
| 单步 logprob 对拍 | 老路径等价、prefix cache 污染专项 | argmax 全等 + ≤4 ulp |
| 逐 token 全等 | 抢占**逻辑**(两边 eager) | 6/6,不允许任何差异 |
| 接受数比对 | 投机语义 | 完全相等 |
| 端到端逐 token(开图) | 仅 `[INFO]` | 不作判据(投机下无 logprobs,见坑 6) |

`suite_inprocess` 会一直占着显存,所以它被放进子进程跑(`--inproc normal` / `--inproc preempt`),
否则后续的 `gen.py` 子进程会因显存不足而 `assert max_blocks > 0` 挂掉(坑 8)。
