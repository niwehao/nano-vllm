# Phase 3B · 草稿模型路径 —— 动手前的方案

写于第一阶段(E1~E4)全部有结论之后、改任何项目文件之前。
第一阶段的实测数据见 `tests/phase3b_probes/`,结论汇总在最终报告 `08-...-report.md`。

> ⚠ **本文是"动手前"的记录,保持原样不回填。有一处在实现中被推翻:**
> 1.2 / 2.2 节里的 `draft_kv_lag ∈ {0,1}`(1 bit 表示"落后几格")表达力不够 ——
> 某条 seq 连着几轮不被打草稿时,真实落后量会超过 1,1 bit 装不下且会静默错位。
> 最终实现改成存**绝对值** `Sequence.draft_num_cached_tokens`,并在要补的 chunk
> 超过 2 时让草稿第一步退回 eager(自愈)。原委见报告的"坑 3"。

## 0. 第一阶段结论(决定进第二阶段的依据)

| 问题 | 结论 | 依据 |
|---|---|---|
| E1 配对 | **可用**。Qwen3-8B(16.4GB 权重)+ Qwen3-0.6B(1.2GB),`tokenizer.json` 逐字节相同,词表都是 151936,919 token 语料 round-trip 逐元素相同 | `tests/phase3b_probes/e1_pairing.py` |
| E2 显存 | **放得下**。两套 KV 每 block 64 MiB(目标 36 + 草稿 28),0.90 利用率下 385 block = 98,560 token = 24 条 4096 满长并发 | `e23_budget_and_cost.py` |
| E3 划算吗 | **图化后划算,不图化直接亏**。bs=8/ctx=512:草稿 2 步 6.36ms / 目标 1 步 23.60ms = 0.269,回本需接受率 α>0.221;草稿不图化时比值 1.326,**必亏** | 同上 |
| E4 草稿 KV | **不需要回滚**。唯一缺口是 a==k 时第 k 个草稿 token 自己的 KV 没人算,下一轮第一次前向吃 2 个 token 补上 | `e4_draft_kv_bookkeeping.py` + vLLM `config/speculative.py:1453-1455` |

**决定:进第二阶段。**

E3 同时给出一条必须写进报告的负面结论:比值随 (batch × 上下文) 恶化,
bs=32/ctx=4096 时到 0.926,回本需 α>0.584。原因是 Qwen3-0.6B 有 28 层 × 8 个 KV 头,
它的 KV 只比 Qwen3-8B(36 层 × 8 头)小 22% —— "草稿便宜"完全来自权重,不来自 KV。

---

## 1. 改哪些文件

只动 4 个文件,`attention.py` / `block_manager.py` / `sampler.py` / `llm_engine.py` 不动。

### 1.1 `nanovllm/config.py`(+6 行)

```python
draft_sample_method: str = "greedy"   # "greedy" | "random"
draft_cudagraph: bool = True          # 关掉用来实测"不图化"的代价
```

`__post_init__` 加断言:`draft_sample_method in ("greedy","random")`;
`speculative_method=="model"` 时要求 `tensor_parallel_size==1 and ep_size==1`。

**为什么要 `draft_sample_method`**:vLLM 的默认就是 greedy 草稿采样
(`config/speculative.py:290` `draft_sample_method: DraftSampleMethod = "greedy"`),
此时草稿分布 q 是 one-hot δ_d,通用接受规则 `min(1, p/q)` 精确退化成 `p(d)` ——
也就是现在 n-gram 那条简化式。vLLM 的 kernel 里这条路叫 `NO_DRAFT_PROBS`,
`draft_prob = 1`(`v1/sample/rejection_sampler.py:817-819`)。
所以"换成真实草稿模型就必须回到通用形式"这句话只对 `random` 成立。
两条都实现,默认 greedy(与 vLLM 一致,且不需要 [B,k,V] 的显存)。

### 1.2 `nanovllm/engine/sequence.py`(+1 个字段)

```python
self.draft_kv_lag = 0   # 草稿 KV 比目标 KV 落后几格(只可能是 0 或 1)
```

**为什么非加不可**:草稿侧 "已算好 KV 的位置数" 与目标侧不总是相等
(E4 的 a==k 情形),这是每条 seq 的状态,没有别处可以推出来。
用 1 bit 的 lag 而不是存一个绝对位置,是为了让 `allocate()` 的 prefix cache 命中
(`num_cached_tokens` 直接跳到 `c*block_size`)不需要同步第二个游标。

### 1.3 `nanovllm/engine/scheduler.py`(+~8 行)

- `schedule()` 的 decode 段:`method=="model"` 时草稿要跑 GPU,只能在 ModelRunner 里出,
  所以这里按最坏情况预留 `k` 个位置(`ensure_capacity(seq, k)`、
  `num_scheduled_tokens = 1+k`、预算扣 `1+k`),`seq.draft_tokens` 留空由 `run()` 回填。
  这与 vLLM 的 `input_budget -= num_new_tokens + draft_slots`
  (`v1/core/sched/scheduler.py:701`)是同一件事。
- `preempt()`:加 `seq.draft_kv_lag = 0`(挨着已有的 `seq.draft_tokens = []`)。
  被抢占的 seq 全量重 prefill,两套 cache 一起归零。

### 1.4 `nanovllm/engine/model_runner.py`(主体)

| 新增/改动 | 做什么 |
|---|---|
| `__init__` | 建第二个模型;断言 vocab/词表一致;录草稿两族图 |
| `allocate_kv_cache` | `block_bytes = 目标 + 草稿`,分配两个 tensor,block 数严格相同 |
| `_capture_decode_family` / `_capture_varlen_family` | 把现有两个 capture 的函数体抽成按 (model, hidden, q_max) 参数化的版本 |
| `run_draft(seqs)` | prefill 同步 + k 步草稿循环,回填 `seq.draft_tokens` |
| `sample_speculative` | 接受规则从"q=δ 特化"扩成通用式,q=None 时行为与现在逐位相同 |
| `run()` | 开头调 `run_draft` |

---

## 2. 关键设计与理由

### 2.1 两套 KV cache:共用 block_table(跟 04 计划,也跟 vLLM)

不建第二套 BlockManager。同一个 `seq.block_table` / 同一份 `slot_mapping` 同时索引
两个物理 tensor,block 数严格相同。

vLLM 是同一个做法,而且更彻底:草稿模型的 attention 层直接注册进全局
`vllm_config`,`get_kv_cache_spec()`(`v1/worker/gpu_model_runner.py:7899-7937`)
一视同仁地遍历到它们,于是草稿层只是"同一个 KVCacheGroup 里多出来的几层"。
`SpecDecodeBaseProposer.validate_same_kv_cache_group`
(`v1/spec_decode/llm_base_proposer.py:1704-1726`)断言所有草稿层落在同一个 group;
`prepare_inputs`(同文件 :1267-1270)给草稿构造 metadata 时直接复用目标的
`block_table_tensor`,`slot_mapping` 取目标 slot_mapping 的子集。

### 2.2 草稿 KV 的推进:不回滚,补一格

E4 已经证清楚。一轮里:

```
第 1 次前向  q = 1 + draft_kv_lag  吃 [len-1-lag, len)     → 出 d_1
第 2..k 次   q = 1                 吃 d_1..d_{k-1}          → 出 d_2..d_k
目标验证     q = 1 + k             吃 [last_token, d_1..d_k]
接受 a 个 → 新长度 len+a+1,draft_kv_lag' = (a == k)
```

被拒绝位置的草稿 KV 是垃圾,不擦:slot 由 position 唯一决定,下一轮真实 token 原地覆盖;
attention 的读取被 `cu_seqlens_k` 挡住。与 04 文档对目标侧残留 KV 的论证同构。

**已知的、有界的不变量松弛**(必须写进报告):`a == k` 那一轮结束时,
`hash_blocks` 登记的范围到位置 `len+k-1`,而草稿只写到 `len+k-2`。
若 `len+k-1` 恰好是某个 block 的最后一格,该 block 会带着一格"过期的草稿 KV"
进 prefix cache 索引,窗口是一轮。后果**只影响命中该前缀的请求的草稿质量(接受率),
不影响输出 token** —— 目标模型的 KV 是权威,接受与否由目标决定。

### 2.3 CUDA graph:草稿两族

- **草稿 decode 图**(q=1,`flash_attn_with_kvcache`):供第 2..k 步。形态与现有
  `capture_cudagraph` 同构,只是换模型换 cache。
- **草稿 varlen 图**(`q_max=2`):供第 1 步(q∈{1,2} 参差)。形态与 Step B 的
  `capture_varlen_cudagraph` 同构,`q_max` 从 `1+k` 换成 `2`。

E4 实测第一次前向的 q 上界就是 2,所以 `q_max=2` 是紧的。

必须图化的依据是 E3:bs=8/ctx=512 时草稿不图化的成本比是 1.326 —— 直接亏。
vLLM 那边草稿只能吃 PIECEWISE 图,吃不了 FULL
(`v1/spec_decode/llm_base_proposer.py:419-434`,注释 "Only supports PIECEWISE
cudagraphs"),nano-vllm 没有 piecewise,所以只能自己录整图 —— 这是我的设计决定,
不是照抄 vLLM。

### 2.4 接受规则:通用式,q=None 退化

```python
# q 为 None(greedy 草稿 / n-gram) → q(d)=1、resid = p 挖掉 d 再归一化
# q 给了真实分布           → 接受概率 min(1, p(d)/q(d)),resid = norm(clamp(p-q,0))
```

`temperature=0` 仍然走 token 比较,不比概率。

**top_k / top_p 与草稿分布怎么交互** —— vLLM 的答案是 **草稿的 q 只过 temperature,
不过 top_k/top_p**,注释写在 `v1/spec_decode/llm_base_proposer.py:1878-1881`:

> Currently, we ignore most of the sampling parameters in generating the draft
> tokens. We only use the temperature. While this could degrade the acceptance
> rate, it does not affect the distribution of the generated tokens after
> rejection sampling.

数学上成立:拒绝采样对**任意** q 都产出精确的 p,q 只影响接受率。
目标侧的 p 则必须是"实际用于采样的分布",即过完 temperature+top_k+top_p
(vLLM 的 `apply_sampling_constraints`,`v1/sample/rejection_sampler.py:510-565`)。
本实现跟这个做法。

### 2.5 草稿循环放在 `run()` 开头,不是结尾 —— 与 vLLM 不同

vLLM 在一步的**结尾**打草稿,存进 `request.spec_token_ids`,下一步的 scheduler
再读出来(`v1/core/sched/scheduler.py:705-720` 消费,`:2254` 写入)。
代价是 scheduler 必须为"下一轮的草稿写 KV"额外预留 slot
(`max_num_new_slots_for_drafting`)。

本实现放在 `run()` 开头。理由:nano-vllm 的 `ensure_capacity(seq, k)` 在 schedule
时按**当轮**的 k 预留,草稿写的位置 `len-1-lag .. len+k-2` 全在这段里,
不需要再引入一个"跨 step 携带草稿"的状态。**这是我的设计决定,不是从 vLLM 抄的。**

副作用:`Sequence.__getstate__` 在 pickle 时草稿还没填,TP worker 拿到的
`scheduled_token_ids` 是错的 —— 所以 `speculative_method="model"` 断言
`tensor_parallel_size == 1`。TP 本来就没实测过(05 报告遗留项 4)。

---

## 3. 可预见的副作用与受影响的调用方

| 影响 | 说明 |
|---|---|
| `allocate_kv_cache` 的 block 数 | 开草稿后腰斩到 56.2%,并发上限跟着降。已有 `assert max_blocks > 0` 太松,换成"至少够 max_num_seqs 条最短序列" |
| `scheduler.stats` / `spec_proposed` | `method="model"` 时按 k 记提议数(恒定),不再是 n-gram 的"命中才记" |
| `exec_stats` | 新增 `draft_graph_decode` / `draft_graph_varlen` / `draft_eager` 三个计数 |
| `enforce_eager` | 草稿两族图跟着一起关 |
| `is_pure_decode` / `use_cudagraph` | **不动**。目标侧的三条路径原样保留(坑 5) |
| n-gram 路径 | **不动**。`method=="ngram"` 时 `draft_model is None`,`run_draft` 整段不进 |
| MoE | `Qwen3MoeForCausalLM` 进不了 CUDA graph(07 报告已证),草稿模型限定 dense;MoE 仍走 `enforce_eager` |

---

## 4. 验收(不放宽)

1. **greedy 逐 token 全等**:`temperature=0`,`k=0` vs `k=2 model` 输出必须逐 token 完全一致。
2. **接受规则数学正确性**:固定 seed 构造已知 p、q,蒙特卡洛验接受率与残差分布符合理论值。
3. **逐 step 三方对拍**:草稿图 replay vs 同形状 eager 必须逐位相同(0 ulp)。
4. **回归**:`test_phase3_spec.py`(14)、`test_phase25_varlen.py`(25)、`test_m3_moe_local.py`(5)。
5. **内存安全**:新增的"按自己算的下标往固定缓冲区写"全部过
   `PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck`。
