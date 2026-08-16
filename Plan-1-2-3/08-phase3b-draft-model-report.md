# Phase 3B · 把投机解码的草稿来源换成真实的小模型

> 一句话结论:**打通了,8 条并发下 1.44×。**
> Qwen3-8B(目标)× Qwen3-0.6B(草稿),k=2,接受率 57%,
> **473.0 tok/s vs 关投机 327.5 tok/s = 1.444×**(通用负载);
> 重复性负载 1.407×。同一个目标模型上 n-gram 只有 1.023×。
>
> **最关键的一个数字:草稿不图化是真的净亏。** 第一阶段预测成本比 1.326 > 1,
> 端到端实测 **0.779×** —— 比关投机还慢两成,而接受率和 tok/步 与图化版一模一样,
> 唯一差别就是草稿那 k 次前向有没有走图。所以"给草稿模型单独录图"不是优化项,
> 是这条路能不能成立的前提 —— 这也是本任务里最容易踩空的地方。
>
> **有一条负面结论**:这笔交易随 (batch × 上下文) 恶化,bs=32 / ctx=4096 时
> 成本比到 0.926,需要接受率 > 0.584 才回本。根因是 Qwen3-0.6B 的 KV 有
> Qwen3-8B 的 78%(28×8×128 vs 36×8×128)—— "草稿便宜"完全来自权重,不来自 KV。
>
> 所有数字都是在这台 L40S 上真跑出来的;凡是估算的都标了"估算"。

基于分支 `phase2.5-stepb-varlen-cudagraph`(Step B 让投机批能 replay varlen 图)。
在 main 上做的话每个投机 step 都掉出图路径,测出来的数字没有意义。

---

## 一、第一阶段:可行性(改任何项目文件之前)

探针脚本全部在 `tests/phase3b_probes/`,可独立复跑。

### E1 · 本机有没有可配对的模型

`tests/phase3b_probes/e1_pairing.py`

配对的硬条件是**同一个 tokenizer + 同一个词表**。只比 `vocab_size` 不够:
vocab_size 相同但 merges 不同的两个模型会给出静默错位的 id。所以三层都查了。

```
目录                架构                          词表    层  hidden   kv头 head_dim     权重GiB  tokenizer.json sha
Qwen3-0.6B        Qwen3ForCausalLM        151936   28    1024     8      128      1.40  aeb13307a71acd8f
Qwen3-8B          Qwen3ForCausalLM        151936   36    4096     8      128     15.26  aeb13307a71acd8f
tiny-qwen3-moe    Qwen3MoeForCausalLM     151936    4    2048     8      128      1.35  be75606093db2094

  [✓] vocab_size 相等: 151936 vs 151936
  [✓] tokenizer.json         aeb13307a71acd8f vs aeb13307a71acd8f
  [✓] vocab.json             ca10d7e9fb3ed185 vs ca10d7e9fb3ed185
  [✓] merges.txt             8831e4f1a0444713 vs 8831e4f1a0444713
  [✓] 真实语料 round-trip: 919 vs 919 个 token,逐元素相同
  [✓] eos_token_id: 151645 vs 151645
```

起点上本机只有 Qwen3-0.6B 和 tiny-qwen3-moe(后者是 `tools/make_tiny_qwen3_moe.py`
造的对拍用权重,随机初始化,拿它当草稿测出来的接受率没有意义)。
**Qwen3-8B 是本次下载的**:16 GiB,`hf download` 实测 **2 分 11 秒**,磁盘余量 230 GiB。
下载后与 Qwen3-0.6B 的 `tokenizer.json` **逐字节相同**,配对成立。

### E2 · 显存放得下吗

`tests/phase3b_probes/e23_budget_and_cost.py`

```
项                                                 字节       GiB
权重 target-8B                          16,404,557,824    15.278
权重 draft-0.6B                          1,213,085,696     1.130
权重合计                                  17,617,643,520    16.408

KV 每 block(256 tok) target-8B             37,748,736     36.00 MiB
KV 每 block(256 tok) draft-0.6B            29,360,128     28.00 MiB
KV 每 block 合计                             67,108,864     64.00 MiB

可用 = 46068MiB*0.90 - 权重               25,857,575,731    24.082   (未扣激活峰值,是上界)
  只放目标 KV              block 数    684  =   175,104 token  =   42 条 4096 长的并发
  两套 KV                block 数    385  =    98,560 token  =   24 条 4096 长的并发
  加草稿 KV 后 block 数缩水到 56.2%(草稿 KV 占 43.8%)
```

**这里有个反直觉的点必须写下来**:草稿模型的权重只有目标的 7.4%,但它的 KV
占到两套合计的 **43.8%**。因为 KV 大小只看 `层数 × KV 头数 × head_dim`,
Qwen3-0.6B 是 28×8×128,Qwen3-8B 是 36×8×128 —— 草稿的 KV 是目标的 **78%**。
"草稿便宜"这件事完全来自权重,一点都不来自 KV。这直接决定了 E3 的结论形状。

### E3 · 这笔交易划算吗

同一个脚本。计时方法沿用 07 报告坑 4 的结论:图只建一次、全部先热身、
重复 12×5 取最小值。一测一建图会量出不可能的比值。

关键行(完整 48 行表在 `tests/phase3b_probes/e23_results.json`):

| 形态 | bs | ctx | 目标 8B (ms) | 草稿 0.6B (ms) | 草稿/目标 |
|---|---|---|---|---|---|
| graph_q1 | 1 | 512 | 21.600 | 2.505 | 0.116 |
| graph_q1 | 8 | 512 | 23.600 | 3.179 | 0.135 |
| graph_q1 | 8 | 4096 | 29.417 | 7.746 | 0.263 |
| graph_q1 | 32 | 4096 | 50.026 | 23.160 | 0.463 |
| **eager**_q1 | 8 | 512 | 24.078 | **15.644** | 0.650 |
| **eager**_q1 | 1 | 512 | 21.980 | **13.955** | 0.635 |

交易表(k=2,几何接受模型 `每步产出 = 1 + Σ_{i=1..k} α^i`):

| bs | ctx | 草稿 k 步 | 目标 1 步 | 比值 r | 回本需 tok/步 | 需 α | **不图化时比值** |
|---|---|---|---|---|---|---|---|
| 1 | 512 | 5.009 | 21.600 | 0.232 | 1.232 | 0.194 | **1.292** |
| 8 | 512 | 6.358 | 23.600 | 0.269 | 1.269 | 0.221 | **1.326** |
| 8 | 4096 | 15.492 | 29.417 | 0.527 | 1.527 | 0.381 | 1.067 |
| 32 | 1024 | 15.431 | 30.231 | 0.510 | 1.510 | 0.372 | 1.034 |
| 32 | 4096 | 46.320 | 50.026 | 0.926 | 1.926 | 0.584 | 0.941 |

三条结论:

1. **草稿必须图化,否则这条路根本不成立。** 最后一列 ≥ 1 意味着"草稿那 k 步比目标
   一整步还贵",接受率再高也回不了本。bs=1/ctx=512 下,0.6B 的单步 eager 是
   13.955 ms、走图是 2.505 ms —— 差出来的 **11.45 ms 全是 kernel launch 开销**,
   是真正算力/带宽开销(2.505 ms)的 **4.6 倍**。同一个尺度上目标模型的
   eager 与图只差 0.38 ms(21.980 vs 21.600),因为它那 19 ms 的权重读取把
   launch 开销完全盖住了。**模型越小,不图化的相对代价越大** —— 这正是草稿模型
   这条路最容易踩空的地方。
2. **小 batch × 短上下文是主场**:bs=8/ctx=512 只需 α > 0.221 就回本。
   同族模型的 greedy 接受率通常远高于此(本次实测 0.55~0.60,见第五节)。
3. **大 batch × 长上下文会塌**:bs=32/ctx=4096 需要 α > 0.584。原因就是 E2 那条 ——
   上下文一长,KV 读取压过权重读取,而草稿的 KV 有目标的 78%,"草稿便宜"就没了。
   **这是负面结论,如实写在这里。**

### E4 · 草稿模型的 KV 怎么跟着走

`tests/phase3b_probes/e4_draft_kv_bookkeeping.py`(纯整数模拟,无 GPU)

把一轮投机的位置账目当纯整数问题,k∈{1,2,4,8} × 3 seed × 20000 轮,断言不变量:

```
  k=1: 第一次前向 q 长度分布 {1: 10064, 2: 9936}   全接受 9936 次,其中落后一格 9936 次   ✓
  k=2: 第一次前向 q 长度分布 {1: 13320, 2: 6680}   全接受 6681 次,其中落后一格 6681 次   ✓
  k=4: 第一次前向 q 长度分布 {1: 15986, 2: 4014}   全接受 4015 次,其中落后一格 4015 次   ✓
  k=8: 第一次前向 q 长度分布 {1: 17804, 2: 2196}   全接受 2196 次,其中落后一格 2196 次   ✓

  A/B/C 三条不变量全部成立
  D 第一次前向的 q 长度上界 = 2  → 草稿那族 varlen 图 q_max 取 2 即可
```

结论:

- **不需要任何"回滚"。** 被拒绝位置的草稿 KV 是垃圾,但下一轮真实 token 会原地
  覆盖它们(slot 由 position 唯一决定),而 attention 的读取被 `cu_seqlens_k`
  (= 真实长度)挡住,读不到残留区 —— 与 04 计划文档对**目标侧**残留 KV 的论证同构。
- **唯一要补的是 `a == k`(全接受)时缺的那一格**:草稿一轮跑 k 次前向,
  写位置到 `len+k-2`,却提议到 `d_k` —— 第 k 个草稿 token 自己的 KV 没人算。
  做法是下一轮草稿的第一次前向吃 2 个 token 而不是 1 个。
  成本是每 `1/α^k` 轮多**一行 q**,不是多跑一次前向。
- 这与 vLLM 的结论一致:`draft_model` 每条请求多留 **1** 个 slot,而
  ngram / EAGLE3 / MTP 是 0(见下一节)。

**决策点:E1~E4 全部为正,进第二阶段。**

---

## 二、vLLM 是怎么做的

clone 到 `https://github.com/vllm-project/vllm`,commit **`fe1c317157d4478fdc0e02096447e61305b871e9`**(2026-08-16)。
**以下每一条都来自源码,不是印象;写不出行号的会明确标注为推断。**

**先确认读的是活代码,不是历史遗留。** 任务提醒 V0/V1 两代的投机解码差别很大,
而且 V1 里 EAGLE/MTP 是主力、纯 draft-model 路径的状态要自己确认。查了三件事:

1. **V0 整代已经不在这个 commit 上了**:全仓库 `grep -rn "SpecDecodeWorker" --include=*.py`
   **0 个命中**,`vllm/spec_decode/` 和 `vllm/worker/` 两个目录都不存在。
   所以不存在"照抄一段已废弃代码"的风险。
2. **draft-model 是 V1 里的一等公民,不是 EAGLE 的副产品**:
   `vllm/v1/spec_decode/draft_model.py:19` `class DraftModelProposer(SpecDecodeBaseProposer)`,
   与 EAGLE/MTP/ngram 并列继承同一个基类;`vllm/config/speculative.py:73` 把
   `"draft_model"` 列在合法 method 里,`:750` 是自动推断分支
   (识别不出别的类型就当作 `draft_model`)。
3. **它有自己的专属分支,不是走 EAGLE 的代码**:`llm_base_proposer.py:113-119`
   算出 `draft_model` 的 `extra_slots_per_request = 1`、`needs_extra_input_slots = True`,
   于是 `set_inputs_first_pass`(`:829-846`)走的是**非 EAGLE** 的那一支
   —— EAGLE 那支要把输入整体移位一格,draft_model 不移位。

以下每一条都能追到 `文件:行号`。

| 问题 | vLLM 的答案 | 出处 |
|---|---|---|
| 草稿与目标的 KV cache 怎么组织?共用还是各管各? | **共用一套 block 管理。** 草稿模型的 attention 层直接注册进全局 `vllm_config`,`get_kv_cache_spec()` 一视同仁地遍历到它们,于是草稿层只是"同一个 KVCacheGroup 里多出来的几层"。给草稿构造 attention metadata 时**直接复用目标的 block table**,slot_mapping 取目标的子集。 | `v1/worker/gpu_model_runner.py:7899-7937`;`v1/spec_decode/llm_base_proposer.py:1704-1726`(`validate_same_kv_cache_group`,断言所有草稿层同一 group);同文件 `:1267-1270`(`block_table_tensor=common_attn_metadata.block_table_tensor`、`slot_mapping=...[token_indices]`) |
| 一轮验证之后草稿怎么把 KV 回滚到接受位置? | **不回滚,重算。** `prepare_inputs` 按 `num_rejected_tokens = k+1-len(sampled)` 把每条请求的 query 长度改成 `q_i - n_i`(= 接受数+1)、`seq_lens` 改成 `s_i - n_i + 1`,草稿下一轮就在这段上跑。被拒绝位置的 KV 不擦,靠 position→slot 的固定映射被后来的真实 token 覆盖。 | `v1/spec_decode/llm_base_proposer.py:1172-1276`,函数头 15 行注释画出了 `query_start_loc` / `seq_lens` / `token_indices` 的变换 |
| **草稿要多留几个 slot?** | **`draft_model` 是 1,ngram/EAGLE3/MTP 是 0。** 注释:"The autoregressive draft-model input retains one unsliced token." 表格里 `Draft model / draft_model / Parallel=No / Additional slots=1`。scheduler 把它扣进 input budget。 | `config/speculative.py:1421-1459`(表格在 :1427-1438)、`:1455-1457`(注释 + `return 1`);`v1/core/sched/scheduler.py:498` `draft_slots = spec.max_num_new_slots_for_drafting`、`:701` `input_budget -= num_new_tokens + draft_slots` |
| 显存在两个模型之间怎么切分?谁先分配? | **不切分。两个模型的权重都先加载完,再 profile,剩下多少就是多少给 KV;两套 KV 只是"同一个 group 里更多的层",block 数天然相同。** `gpu_worker.load_model()` → `model_runner.load_model()`(内部 `self.drafter.load_model(self.model)`)→ 之后才是 `determine_available_memory()` → `profile_run()`。 | `v1/worker/gpu_worker.py:450-457`(load)、`:475-492`(determine/profile);`v1/worker/gpu_model_runner.py:5443-5445`(`self.drafter.load_model(self.model)`) |
| 草稿的 decode 循环有没有单独的 CUDA graph? | **只有 PIECEWISE,拿不到 FULL。** 注释原文 "Only supports PIECEWISE cudagraphs (via mixed_mode)";代码里无论 `mixed_mode()` 是 PIECEWISE 还是 FULL,草稿一律降到 `CUDAGraphMode.PIECEWISE`,否则 `NONE`。 | `v1/spec_decode/llm_base_proposer.py:419-434` |
| 接受/拒绝规则在哪?怎么处理 temperature=0? | Triton kernel,greedy 与 random 两个 kernel,按 `is_greedy` 各自 early-return。greedy:`rejected = draft_token_id != target_argmax_id`,**比 token 不比概率**;random:`accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob`。 | `v1/sample/rejection_sampler.py:715-770`(greedy,`:756-757` 是判定)、`:774-845`(random,`:829` 是判定,`:827-828` 是 woosuk 关于 draft_prob=0 的注释) |
| 草稿分布 q 怎么存?没有 q 的时候呢? | `NO_DRAFT_PROBS` 常量分支里 **`draft_prob = 1`** —— 也就是 q=δ,接受概率退化成 `p(d)`;残差采样对应地把 draft token mask 掉再从 p 里采。而 **`draft_sample_method` 的默认值就是 `"greedy"`**,所以 vLLM 默认走的就是这条 q=δ 的路。 | `rejection_sampler.py:815-817`(`if NO_DRAFT_PROBS: draft_prob = 1`)、`:913-918`(`vocab_offset != draft_token_id` 的 mask);`config/speculative.py:290` `draft_sample_method: DraftSampleMethod = "greedy"` |
| 残差分布怎么算? | `prob = tl.maximum(target_prob - draft_prob, 0.0)`,**不归一化** —— 注释说明因为后面用 Gumbel-max 取 argmax,归一化常数不影响结果。 | `rejection_sampler.py:930-932` |
| **top_k / top_p 与草稿分布怎么交互?** | **目标侧 p 过 temperature+top_k+top_p;草稿侧 q 只过 temperature。** 注释原文:"Currently, we ignore most of the sampling parameters in generating the draft tokens. We only use the temperature. While this could degrade the acceptance rate, it does not affect the distribution of the generated tokens after rejection sampling." | p 侧:`rejection_sampler.py:510-565`(`apply_sampling_constraints`);q 侧:`v1/spec_decode/llm_base_proposer.py:1854-1892`,注释在 `:1878-1881` |
| 词表/tokenizer 一致性怎么校验?不一致报什么? | 只查 `vocab_size` 严格相等,不一致抛 `ValueError`,文案里点明后果是"out-of-bounds errors"。异构词表要显式开 `use_heterogeneous_vocab`,那时建一张 `VocabMapping` 把 id 映射到交集。 | `config/speculative.py:1403-1418`;`v1/spec_decode/draft_model.py:34-58`、`:60-61` |
| 草稿提议失败 / 长度不齐怎么退化? | 三处:① prefill chunk 的请求直接清空草稿(`request.spec_token_ids = []`);② 调度时按本轮实际能排下的数量截断 `spec_token_ids[:num_scheduled_spec_tokens]`;③ kernel 里 **`draft_token_id < 0` 一律判拒绝**(注释:"-1 is used for padded draft token ids that should be rejected")。 | `v1/core/sched/scheduler.py:2244-2248`、`:714-715`;`rejection_sampler.py:810-811`(random kernel)与 `:751-752`(greedy kernel) |

### 本实现跟了哪些、没跟哪些

跟的:

- 两套 KV **共用 block_table**(与 04 计划文档一致,也与 vLLM 一致)。
- 被拒绝位置的 KV **不擦**,靠 position→slot 覆盖。
- "草稿比 ngram/EAGLE 多欠 **1** 格"这个结论本身(E4 用整数模拟独立推出来,
  和 vLLM 的表格对上)。**但两边表现形式不同,别混淆**:vLLM 是在 scheduler 里
  真的多**预留一个 slot**(因为它的草稿跑在下一个 step);本实现不多留 ——
  草稿跑在同一个 step 内、写的位置全在 `ensure_capacity(seq, k)` 已经留好的范围里,
  那一格表现为**下一轮第一次前向的 q 从 1 变成 2**。
- 草稿的 q **只过 temperature,不过 top_k/top_p**;目标的 p 过全套。
- `draft_sample_method` 默认 `"greedy"`,此时 q=δ,走与 n-gram 相同的简化式。
- 词表校验必须硬失败(本实现比 vLLM **更严**:除 `vocab_size` 外还逐字节比
  `tokenizer.json`/`vocab.json`/`merges.txt`。理由:vocab_size 相同而 merges 不同
  的两个模型会静默错位,vLLM 那条断言拦不住)。

**没跟、这是我的设计决定**:

1. **草稿循环放在一步的开头,不是结尾。** vLLM 在一步的结尾打草稿存进
   `request.spec_token_ids`,下一步 scheduler 再读出来,代价是必须为"下一轮的草稿"
   额外预留 slot(`max_num_new_slots_for_drafting`)。nano-vllm 的
   `ensure_capacity(seq, k)` 是在 schedule 时按**当轮**的 k 预留的,草稿要写的位置
   `draft_num_cached .. len+k-2` 全在这段里,放开头就不需要再引入一个"跨 step 携带草稿"
   的状态。副作用:`Sequence.__getstate__` 在 pickle 时草稿还没填,所以
   `speculative_method="model"` 断言 `tensor_parallel_size == ep_size == 1`。
2. **给草稿录整图(FULL),不是 PIECEWISE。** vLLM 的草稿只能吃 piecewise 图,
   因为它依赖 torch.compile 的切图基础设施;nano-vllm 没有 piecewise,只有
   "整图 or 全 eager"两档。E3 已经证明全 eager 那档直接亏,所以只能整图。
   代价是要为草稿多录两族图(见第三节)。
3. **草稿 KV 的推进用"绝对已算位置数"而不是重算。** vLLM 每轮用
   `prepare_inputs` 把接受段整体重跑一遍(query 长度 = 接受数+1);本实现让草稿
   在自己那 k 步里就把 KV 写好,下一轮只补最多 1 格(E4 证明上界就是 1)。
   两者的 KV 内容等价,本实现少跑 `a` 行 q。之所以能这么做,是因为 nano-vllm 的
   草稿循环和目标验证在**同一个 step 内**,vLLM 的隔了一个 step 且中间可能被
   preemption/chunked prefill 打断,必须用更通用的重算。

---

## 三、改了哪些部分

`nanovllm/` 只动了 4 个文件。`attention.py` / `block_manager.py` / `sampler.py` /
`llm_engine.py` / `context.py` 一行没动。

`git diff --numstat nanovllm/` 实测 **+593 / −57**(4 个文件,含 review 之后的修改):

| 文件 | 增/删 | 改动 |
|---|---|---|
| `config.py` | +15 / −0 | +2 个字段 `draft_sample_method` / `draft_cudagraph`,+3 条断言 |
| `engine/sequence.py` | +11 / −0 | +1 个字段 `draft_num_cached_tokens` |
| `engine/scheduler.py` | +24 / −5 | decode 段按 `num_spec` 预留;`preempt` 归零;`postprocess` 更新草稿游标 |
| `engine/model_runner.py` | +543 / −52 | 主体 |

`model_runner.py` 那 −52 行里绝大部分是"被搬走"而不是"被删掉" ——
四个函数体抽成参数化版本时,原行在 diff 里同时算一次删除和一次新增。

### 3.1 `config.py`

```python
draft_sample_method: str = "greedy"       # "greedy" | "random"
draft_cudagraph: bool = True
```

`__post_init__` 加:`draft_sample_method` 取值断言;`method=="model"` 时断言
`tensor_parallel_size == 1 and ep_size == 1`(原因见上一节设计决定 1)。

### 3.2 `engine/sequence.py`

```python
self.draft_num_cached_tokens = 0
```

语义与 `num_cached_tokens` 平行:位置 `0..draft_num_cached_tokens-1` 的**草稿** KV 有效。
稳态下两者相等,只有"上一轮草稿全被接受"时少 1。

存**绝对值**而不是"落后几格"是踩了坑之后改的(见坑 3)。

### 3.3 `engine/scheduler.py`

- decode 段:`method=="model"` 时草稿要跑 GPU、这里拿不到,于是按最坏情况预留
  `k` 个位置(`ensure_capacity(seq, k)`、`num_scheduled_tokens = 1+k`、预算扣 `1+k`),
  `seq.draft_tokens` 留空由 `run()` 回填。
- `preempt()`:`seq.draft_num_cached_tokens = 0`(挨着已有的 `seq.draft_tokens = []`)。
- `postprocess()`:`draft_cached = min(L+a, L+k-1)`,其中 L 是 append 之前的长度。
  `a<k` → `L+a` = 新长度-1(齐平);`a==k` → `L+k-1` = 新长度-2(欠一格)。

### 3.4 `engine/model_runner.py`

新增:

| 函数 | 作用 |
|---|---|
| `build_draft_model` | 加载草稿模型 + 三层配对校验(vocab_size / 分词器文件逐字节 / dtype) |
| `kv_block_bytes` / `bind_kv_cache` | 从 `allocate_kv_cache` 里抽出来,现在要给两个模型各算一份 |
| `slots_for` | position → 物理槽位 |
| `prepare_draft_first` | 草稿第 1 次前向的变长 context(q ∈ {1,2}) |
| `prepare_draft_decode` | 草稿第 2..k 次前向的 decode context(q=1) |
| `sample_draft` | 从草稿 logits 采 token,`random` 时同时返回整行 q |
| `run_draft` / `sync_draft_prefill` / `propose_draft_model` | 草稿主流程 |
| `sync_draft_decode` | 这一轮不打草稿的 decode seq 也要把草稿 KV 追平(review 之后加的,见第七节 #2) |
| `capture_draft_cudagraph` | 录草稿的两族图 |

改动(**都是把已有函数体按参数抽出来,原注释随代码原样搬过来,一字未动**):

| 原 | 现 |
|---|---|
| `capture_cudagraph` 的函数体 | → `capture_decode_family(model, hidden_size)`,原函数变成 3 行的调用方 |
| `capture_varlen_cudagraph` 的函数体 | → `capture_varlen_family(model, hidden_size, q_max)` |
| `run_model` 的 else 分支 | → `replay_decode(graphs, graph_vars, ...)`,返回隐状态 |
| `run_varlen_graph` 的图刷新+replay | → `replay_varlen(graphs, gv, ...)`,返回隐状态 |
| `allocate_kv_cache` | block 字节数 = 目标 + 草稿;`assert max_blocks > 0` 换成可诊断的断言 |
| `sample_speculative` | 加 `draft_probs` 形参与通用接受式分支(**n-gram 那条分支一个字没动**) |
| `run` | 开头调 `run_draft` |

抽出来而不是整段抄,是因为现在有 **4 族图**(目标 decode / 目标 varlen q_max=1+k /
草稿 decode / 草稿 varlen q_max=2),抄 4 遍不可维护。
`tools/check_comments.py` 复核:**原始注释 841 行,缺失 0 行**。

### 3.5 一轮的完整时序

```
run(seqs):
  run_draft(seqs)
    ├─ prefill 子集 → sync_draft_prefill:草稿对同一 chunk 再跑一遍,只写 KV,logits 丢弃
    └─ decode 子集(且 num_scheduled_tokens == 1+k)→ propose_draft_model:
         第 1 步  q = len - draft_num_cached ∈ {1,2}  → 草稿 varlen 图(q_max=2)
         第 2..k 步 q = 1                              → 草稿 decode 图
         k 步只做 1 次 D2H(每步 .tolist() 会串上 k 次同步)
  prepare_batch(seqs)          # 此时 seq.draft_tokens 已被填成长度 k
  run_varlen_graph             # 目标验证前向,q=1+k,走 Step B 的图
  sample_speculative           # 接受判定
```

### 3.6 接受规则

```
q 为 None(greedy 草稿 / n-gram)   →  接受概率 p(d);resid = p 挖掉 d 再归一化
q 是真实分布(random 草稿)          →  接受概率 min(1, p(d)/q(d));resid = norm(clamp(p-q,0))
temperature == 0                    →  比 token 不比概率(两条路都一样)
```

退化情形逐个查过:

- `q(d) == 0`:`q_d.clamp_min(1e-10)` 让比值变成一个巨大的正数,`rand < ratio`
  恒真 → 必接受。这是对的:草稿提了一个 q 说不可能出现的 token,说明 q 已经
  不是真的提议分布了,按 `min(1, p/q) = 1` 接受不破坏分布。
- **残差全零**(`clamp(p-q,0).sum() == 0`,即 `p == q` 逐元素相等):
  这一支**不可达**。`p == q` 蕴含 `p(d)/q(d)` 在 IEEE 下**恰好**等于 1.0,
  而 `torch.rand` 取值在 `[0,1)`,所以 `rand < 1.0` 恒真 —— 必接受,
  残差根本不会被消费。只要 p 和 q 有任何一个元素不等,残差就是合法分布。
- `p(d)/q(d) > 1`:直接和 `rand ∈ [0,1)` 比,恒真 → 必接受,
  等价于 `min(1, ·)`,不需要显式 clamp。

两条分支分开写,不是合成一条。q=δ 时通用式确实**精确**退化成简化式
(`q(d)=1`;`clamp(p-δ_d,0)` 就是"把 d 挖掉的 p",因为 `p(d)≤1`),但简化式用
`scatter` 少一次 `[R,V]` 的减法和一次 `[B·k,V]` 的物化 —— n-gram 那条路没道理为
草稿模型的功能付这个钱。**n-gram 分支一个字没动。**

---

## 四、踩过的坑

### 坑 1 · "greedy 逐 token 全等"在这个项目里拿不到,而且不是本次改动引起的

第一次跑 8B+0.6B 对比不投机的基线,**6 条里只有 3 条逐 token 一致**。
按任务书的硬判据这是不合格的。

**先怀疑代码,不怀疑判据。** 做了三个对照实验:

1. **自草稿**(草稿模型 = 目标模型 = Qwen3-0.6B)。如果机械部分有任何错位,
   草稿的预测就会偏离目标,接受率立刻掉下来。实测 **`spec_accepted: 192 /
   spec_proposed: 192` = 100%**,48 个 token 用 16 个 decode step 产出
   = 恰好 3.0 tok/step = 1+k 的理论上限。
2. **自草稿 vs 已验证过的 n-gram 路径**:**6/6 逐 token 全等**。
3. **两条路各自 vs 关投机**:分歧落在**完全相同的两个位置、完全相同的两对 token、
   完全相同的 ulp 差**:

```
c_ngram   seq[0] @ 24: off=382   ngram=13   差 0.1250 = 2.0 ulp (最并列的 3.5%)
c_self    seq[0] @ 24: off=382   self=13    差 0.1250 = 2.0 ulp (最并列的 3.5%)
c_ngram   seq[2] @ 33: off=11211 ngram=198  差 0.1250 = 2.0 ulp (最并列的 7.3%)
c_self    seq[2] @ 33: off=11211 self=198   差 0.1250 = 2.0 ulp (最并列的 7.3%)
```

结论:分歧的来源是**不投机走 `flash_attn_with_kvcache`、投机的验证前向走变长
kernel** 这条边界,两条 kernel 路径的归约顺序不同,bf16 logits 有 ulp 级漂移,
在本来就近似并列的位置上 argmax 翻转。这是 Phase 3 就存在的性质,
`tests/test_phase3_spec.py:3-8` 已经写明并留下了判据设计。

8B 上的三处分歧,gap 分别是 **0.0 / 0.0 / 2.0 ulp**(前两处是 bf16 下**完全并列**,
argmax 靠下标先后决出),全部落在最并列的 3.1%~6.8%,`0 条为真分歧`。

**判据没有放宽。** 处理方式是把"必须 0 差异"放到它真正成立的地方:
自草稿 vs n-gram 路径(两侧 kernel 路径相同)必须 6/6 全等,这是测试 A1;
图 vs 同形状 eager 必须 0 ulp,这是测试 E。

### 坑 2 · 单元测试差点测错了东西

写"q=δ 时通用式必须与简化式给出同一个 token 序列"的测试,固定同一个种子跑两条分支,
**失败**。查下去发现两条分支物化的张量大小不同(通用式只在草稿行上算残差,
`[B·k, V]`;简化式在全部行上算,`[B·(k+1), V]`),`sample_from_probs` 里
`exponential_` 抽的随机数个数因此不同,随机流必然错开 —— 我在测一件与正确性无关的事。

改成两段,都不含随机流依赖:(a) 确定性角(把 p 做成尖峰,接受/拒绝与修正 token
唯一确定,两条分支必须完全相同);(b) 统计角(4 万次,两条分支的输出分布与接受率
必须互相吻合且都吻合理论值)。实测 `max|Δ互相|=0.0034`,接受率 `0.0204/0.0204`
对理论 `p(d)=0.0207`。

这是**加强**判据不是放宽:原来的写法在随机流碰巧对齐时会假通过。

### 坑 3 · 1 bit 的"落后几格"表达不了真实状态

最初把草稿的进度存成 `draft_kv_lag ∈ {0,1}`。E4 的模拟证明稳态下确实只有 0/1
两种取值,所以看起来够用。

**审自己的代码时发现一个反例**:`tests/gen.py --pollution` 的第二阶段会在运行期
把 `llm.scheduler.num_spec_tokens` 改成 0。那之后 decode seq 的
`num_scheduled_tokens` 只有 1,但 `run_draft` 还是会往 `draft_tokens` 里塞 k 个
token —— `scheduled_token_ids` 里 `n = num_scheduled - k` 变成负数,
送进模型的是"草稿 token 顶掉了 last_token"的一批垃圾。

第一版修法是在 `run_draft` 里过滤掉 `num_scheduled_tokens != 1+k` 的 seq。
但这只堵住了一半:被跳过的那些 seq,目标每步长 1 个 token 而草稿不动,
真实的落后量会一路涨上去,1 bit 表达不了,恢复打草稿时就会静默错位。

最终改成存**绝对值** `draft_num_cached_tokens`,第一次前向的 chunk 长度
`q = len - draft_num_cached_tokens` 自己算出来;`q > 2` 时那一步退回 eager
(变长路径本来就吃得下任意 q),后面 k-1 步的 decode 图照走。**自愈,不静默。**

这个坑是 `--pollution` 这条既有测试路径踩出来的,而它恰好在我新写的 C3 用例里。

### 坑 4 · 注释搬家掉了一个行尾空格

把 `capture_cudagraph` 的函数体抽成 `capture_decode_family` 时,
`tools/check_comments.py` 报 `[缺失] ...而是自己 get_context() ` —— 原注释行尾有一个空格,
我在搬运时被编辑器吃掉了。按 CLAUDE.md 的约定"重构或移动代码时,注释跟着它所属的
代码一起走,内容一字不变",一个空格也算。复原后 **841 行原始注释缺失 0 行**。

### 坑 5 · `max_seqlen_q` 也是烧进图的 host 标量,原来没断言

`replay_varlen` 原本只断言了 `max_seqlen_k <= max_model_len`。抽成参数化版本、
让草稿那族用 `q_max=2` 之后才意识到 `max_seqlen_q` 同样被烧死,而且超了**不会报错**
—— FA 的 m 维 grid 是 `ceil(max_seqlen_q/64)`,q_max=2 时 grid=1 仍能覆盖 64 行,
所以 kernel 照跑,只是 padding 槽的 cu_seqlens 算术(每槽至多 q_max 行)对不上,
**静默错位**。补了断言。

### 坑 6 · 抢占测试的块数是算出来的,不是随手填的

写抢占用例时给 `num_kvcache_blocks` 随手填了 32,结果 `preempted: 0` —— 用例 FAIL。

**先怀疑代码**:算了一遍需求。`build_preempt_prompts` 是 6 条 × 255 token,
prefill 后每条占 1 块;k=2 时 `ensure_capacity(seq, 2)` 要覆盖到位置 256,
于是每条立刻要第 2 块 —— **峰值需求 12 块**。32 > 12,压根不该有抢占。
所以是用例配错了,不是调度器的问题;数字对得上,不是"我觉得"。

改成 `test_phase3_spec.py` 用的 7 块(< 12),抢占才会发生。
顺带把判据从"抢占次数 > 0"加强成"抢占次数 > 0 **且** 6 条输出都跑满 48 个 token"
—— 只看抢占次数的话,抢占后卡死也会通过。

### 坑 7 · 给"退化 q"写回归用例,第一版把前提写没了

独立 review 指出通用接受式在 q 带 NaN / 残差全零时会吐出 token 0(见第七节),
修完之后要补回归用例。第一版写成"固定 `d=4`,给 q 灌 NaN,断言输出仍服从 p" ——
**FAIL**,而且失败方式是 `max|Δp|=0.8366`、`P(token 0)=0.0000`。

两个错都在用例里,不在代码里:

1. **前提没了。** Leviathan 定理要求 `d ~ q`。我把 d 钉死成 4,输出当然不是 p ——
   实测输出恒为 token 4,`max|Δp|` 正好等于 `1 - p(4)`。
   (`test_general_rule_preserves_distribution` 里我是对的:`torch.multinomial(q_row, ...)`。)
2. **NaN 根本没走到要测的那条路。** 我把 NaN 放在第 0 列,而 `q(d)` 是好的,
   于是 `ratio = p(d)/q(d) = 1` → **必接受** → 残差分支一次都没进。
   要逼进去得让它必然拒绝:把 `q(d)` 设成 1.0,`ratio = p(d)` 就很小了。

改对之后三格都是可判的:NaN 情形接受率 0.1642 ≈ `p(d)=0.1634`、输出对理论混合分布
`max|Δ|=0.0053`、`P(token 0)=0.1440`(旧代码这里会是 ~0.84);`q(d)=0` 情形接受率
**恰为 0**、残差对理论 `max|Δ|=0.0033`;`q==p` 情形接受率**恰为 1.0000**,
把"残差全零那一支不可达"这件事钉在了测试里。

跟坑 2、坑 6 是同一类错误:**判据写错了会以"代码有 bug"的样子出现**。
三次都是先把失败数字算清楚,才敢说是用例的问题。

### 坑 8 · `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 和 CUDA graph 录制**天然互斥**

内存安全检查的第一版写成"起一个真的 LLM,整个跑一遍",结果:

```
========= ERROR SUMMARY: 4 errors
torch.AcceleratorError: CUDA error: operation failed due to a previous error during capture
  ... in capture_decode_family:  with torch.cuda.graph(graph, self.graph_pool):
```

差点当成"新代码有 4 处越界"。**其实一处都不是**:关掉 caching allocator 之后每次
分配都是裸 `cudaMalloc`,而 `cudaMalloc` 在 stream capture 期间是非法操作,
于是录图必然失败,那 4 个 error 是这个失败本身。

也就是说,**这个组合下"整个引擎跑一遍"这条路根本走不通** —— 而少了那个环境变量的
memcheck 又什么都查不到(07 报告坑 10)。两头都堵死。

07 报告其实已经给出了正确做法,我一开始没照做:它的 `probe_sink_memcheck.py`
不建模型、不录图,**直接复刻生产代码的下标算法再调一次 FA kernel**。
照这个思路把探针拆成两个 mode:

| mode | 覆盖什么 | 为什么能跑 |
|---|---|---|
| `varlen` | `replay_varlen` 对**草稿族**(q_max=2)的 cu_seqlens / block_tables 算术,27 个构型(9 种 num_seqs × 3 种 q 长度分布) | 不建模型不录图,只调 FA |
| `eager` | `prepare_draft_first` / `prepare_draft_decode` 算出的 slot_mapping 真的喂给 `store_kvcache` 的 triton kernel 写 KV cache | `enforce_eager=True`,一张图都不录,没有 capture |

两个合起来盖住本次新增的全部"按自己算的下标往固定缓冲区写"。
**代价要说清楚**:图 replay 本身(`graphs[bs].replay()`)在这个组合下没法直接查;
但 replay 只是重放录制时就固定下来的 kernel 序列,而那些 kernel 的下标全部来自
`varlen` mode 复刻的那套 cu_seqlens/block_tables —— 这是我的论证,不是实测。

---

## 五、每一步的效果(全部实测)

### 5.1 正确性

`tests/test_phase3b_draft_model.py`,共 24 项,**24/24 通过**。

**第 1 层 · 数学(不允许任何误差)** —— 直接调真代码
(`ModelRunner.sample_speculative` 不依赖 `self`,可以当函数调,沿用
`test_phase3_spec.py` 的做法):

```
[PASS] 通用式保分布 (max|Δp|=0.0013 < 0.012, 接受率 0.2149 ≈ Σmin(p,q)=0.2149)
[PASS] q=δ 时通用式 ≡ n-gram 简化式(确定性角精确相等;4 万次 max|Δ互相|=0.0034,
       max|Δ理论|=0.0022, 接受率 0.0204/0.0204 ≈ p(d)=0.0207)
[PASS] 拒绝时的修正 token 服从 norm(clamp(p-q,0)) (拒绝 59962/60000 次, max|Δ|=0.0053 < 0.012)
[PASS] temperature=0 行不受 draft_probs 影响(仍是 token 比较)
```

第一条构造了**故意错开**的 p 和 q(q 用 `-p_logits` 的 softmax,把概率压在 p 的
低概率区),接受率只有 0.215,残差项被充分激活 —— 不是"q≈p 时怎么写都对"的假通过。

**第 2 层 · 机械(最灵敏的探针)**

把草稿模型换成目标模型自己。草稿的每一个提议都应当恰好等于目标的 argmax,
所以接受率必须是 **100%**。草稿 KV 错位一格、position 算错、
`draft_num_cached_tokens` 漏更新 —— 任何一个都会让它掉下来。

```
spec_accepted: 192 / spec_proposed: 192 = 100.0%
48 个输出 token / 16 个 decode step = 3.00 tok/step   (= 1+k 的理论上限)
```

而且 100% 接受意味着**每一轮都是 `a == k`**,也就是每一轮都走 `q=2` 的补算路径 ——
这条路被压满了 16 轮,不是偶尔碰一下。k=4 时同样是 240/240,10 个 decode step
产出 48 个 token = **4.80 tok/step**(理论上限 5.0)。

一个例外值得记下来:**抢占场景下自草稿是 188/192 = 97.9%,不是 100%**。
原因不是簿记错了(C4/C5 都过),而是抢占触发全量重 prefill,草稿这一次是
"整段一个 chunk"算 KV,而抢占前是"prefill chunk + 若干次 q=1 decode"逐步算的。
两条路数学等价但归约顺序不同,KV 有 ulp 级差异(05 报告坑 3 实测这类变换的单步
扰动是 2.0~2.7 ulp),于是草稿在近似并列的位置上 argmax 偶尔翻一下。
**输出 token 不受影响** —— 接受与否由目标决定,C5 的三方判据是 0 条真分歧。

**第 3 层 · 逐位(0 ulp,不允许任何偏差)**

图 replay vs **同形状** eager。参照用的是图刚刚刷新好的那份 padded 缓冲区原样再跑
一遍(07 报告坑 5 的方法:直接和"真实形状的 eager"比,量到的是
图误差 + padding 误差之和,分不开)。

```
[PASS] draft_varlen:  13/13 逐位相同, 最大偏差 0.00 ulp
[PASS] draft_decode:  13/13 逐位相同, 最大偏差 0.00 ulp
[PASS] target_varlen: 13/13 逐位相同, 最大偏差 0.00 ulp
[--]   target_decode: 本配置下没走到
```

`target_decode` 那行是**故意打印出来**的:自草稿 k=2 的配置下每个 step 都是投机批,
目标那侧一次都不会走纯 decode 的图。测试如实报"没走到"而不是把它算成通过 ——
否则一条从未执行的路径会伪装成一次成功验证。

**第 4 层 · 端到端**

| 项 | 结果 |
|---|---|
| A1 自草稿 vs n-gram 路径 | **6/6 逐 token 全等**(两侧 kernel 路径相同,这条不允许任何差异) |
| A2 两条路相对关投机的分歧集合 | **完全相同**(同位置、同 token、同 ulp) |
| A3 自草稿 vs 关投机 | 4/6 全等,2 条噪声(≤2.0 ulp),**0 条真分歧** |
| A4 同配置重复运行 | 6/6 全等(确定性) |
| A5 自草稿接受率 | 192/192 = 100% |
| B1 8B+0.6B vs 关投机 | 3/6 全等,3 条噪声(0.0/0.0/2.0 ulp),**0 条真分歧** |
| B2 接受率 / step 数 | 0.561,decode step 31 → 19,1.68 tok/step |
| B3 草稿前向走图占比 | 38/39 = **97.4%**(那 1 次 eager 是 prefill 同步) |
| C1 k=1 / 2 / 4 | 各 4/6 全等 + 2 条噪声,0 条真分歧 |
| C2 混批 | 6 个混批 step,草稿 prefill 同步 7 次,18 条全部跑满 |
| C3 prefix cache 污染 | 2/6 全等,4 条噪声,**0 条真分歧** |
| C3b 每个 decode step 都推进草稿 KV | 草稿首次前向 **63 次 = decode step 63 次** |
| C4 抢占 | 抢占 4 次,6 条全部跑满 48 token |
| C5 抢占 + 草稿模型 | 3/6 全等,3 条噪声,**0 条真分歧** |
| C6 `temperature=0.8, top_k=50, top_p=0.9` | greedy 草稿接受率 0.640;**random 草稿 0.909** |
| C7 关掉草稿图 | 6/6 逐 token 全等(只慢,不错) |
| D1 分词器不同的配对 | 带诊断信息硬失败 |

C6 的两个数字值得单独说:同一个自草稿配置下,`random` 的接受率(0.909)明显高于
`greedy`(0.640)。这正是通用接受式在起作用 —— q 按温度采样时更贴近 p,
`min(1, p/q)` 更接近 1。两者产出的 token 分布**都**严格等于 p,只是接受率不同。

**C3b 是独立 review 逼出来的用例**,值得单独说。它指出 C3 本身测不到污染:
`--pollution` 的第二阶段是最后一阶段,被污染的 block 再没人读回来。
所以 C3b 改成直接查那条不变量的机械前提 ——「每个 decode step 都必须有一次
草稿的第一次前向」。

第二阶段把 `num_spec_tokens` 置 0,那些 step 不打草稿,但仍然必须把草稿 KV 追平,
否则 `hash_blocks` 会把**一整段草稿 KV 从没写过**的 block 登记进 prefix cache 索引。
修之前 `exec_draft_graph_varlen` 是 **16**(只有第一阶段那 16 步),
加了 `sync_draft_decode` 之后是 **63**,恰好等于 63 个 decode step。
这个用例在修改前会直接 FAIL(16 ≠ 63),是一条真的回归防线,不是走过场。

**第 5 层 · 内存安全**

`PYTORCH_NO_CUDA_MEMORY_CACHING=1` + `compute-sanitizer --tool memcheck`,两个 mode:

```
===== mode=varlen =====
[varlen] 27 个构型全部跑完(q_max=2),真实行与 padding 行均为有限值
========= ERROR SUMMARY: 0 errors

===== mode=eager =====
[eager] exec={... 'draft_eager': 7}   accept=24/24
========= ERROR SUMMARY: 0 errors
```

`eager` 那一档 `accept=24/24` 是 100%,意味着每一轮都是 `a==k`、
每一轮的草稿第一次前向都走 `q=2` 的补格路径 —— 要查的那条路被压满了,不是碰巧绕过去。

(第一版探针把整个引擎塞进 memcheck 跑,报了 4 个 error,**一个都不是越界** ——
是 `cudaMalloc` 在 stream capture 期间非法导致的录图失败。见坑 8。)

**回归**

| 套件 | 结果 |
|---|---|
| `tests/test_phase3b_draft_model.py` | **26/26**(含 review 逼出来的 2 条新用例) |
| `tests/test_phase25_varlen.py` | 全过(rc=0) |
| `tests/test_phase3_spec.py` | **14/14** |
| `tests/test_m3_moe_local.py` | **5/5** |
| `tools/check_comments.py` | 原始注释 841 行,**缺失 0 行** |

### 5.2 性能

`tests/bench_draft_model.py`。目标 Qwen3-8B、草稿 Qwen3-0.6B、**8 条并发**、
每条 256 个输出 token。n-gram 的对照跑在**同一个目标模型**上,否则两组数字没法比。
两种负载:`generic` 普通文本;`repeat` 是 prompt 里同一段出现两次(n-gram 的理想场景)。

| 负载 | 配置 | tok/s | tok/步 | 接受率 | 走图% | 草稿走图% | TBT p50/p99 (ms) | vs 关投机 |
|---|---|---|---|---|---|---|---|---|
| generic | 投机关闭 + decode 图 | 327.5 | 8.00 | — | 99.6% | — | 23.89 / 24.13 | 1.000× |
| generic | n-gram k=2 + varlen 图 | 335.0 | 8.68 | 18.7% | 99.6% | — | 25.43 / 26.11 | 1.023× |
| generic | **小模型 k=2 + 全图** | **473.0** | 15.40 | 57.2% | 99.2% | 99.6% | 31.47 / 31.99 | **1.444×** |
| generic | 小模型 k=4 + 全图 | 458.0 | 18.29 | 40.1% | 99.1% | 99.8% | 38.90 / 39.73 | 1.399× |
| generic | 小模型 k=8 + 全图 | 371.4 | 19.50 | 25.9% | 99.0% | 99.9% | 51.89 / 53.64 | 1.134× |
| generic | 小模型 k=2 **草稿不图化** | 255.1 | 15.40 | 57.5% | 99.2% | 0% | 59.29 / 62.05 | **0.779×** |
| repeat | 投机关闭 + decode 图 | 311.5 | 8.00 | — | 99.6% | — | 24.41 / 24.62 | 1.000× |
| repeat | n-gram k=2 + varlen 图 | 322.2 | 8.98 | 18.2% | 99.6% | — | 26.55 / 26.97 | 1.034× |
| repeat | **小模型 k=2 + 全图** | **438.3** | 15.52 | 59.5% | 99.2% | 99.6% | 32.69 / 34.12 | **1.407×** |
| repeat | 小模型 k=4 + 全图 | 429.1 | 18.79 | 43.5% | 99.1% | 99.8% | 40.85 / 42.60 | 1.378× |
| repeat | 小模型 k=8 + 全图 | 345.3 | 20.28 | 26.8% | 99.0% | 99.9% | 55.75 / 77.81 | 1.109× |
| repeat | 小模型 k=2 **草稿不图化** | 257.7 | 16.38 | 59.3% | 99.2% | 0% | 60.68 / 64.78 | **0.827×** |

四条结论:

1. **草稿不图化是真的净亏,0.779× / 0.827×。** 这是第一阶段 E3 那个预测
   (成本比 1.326 > 1)的端到端确认 —— 接受率一模一样(57.5% / 59.3%),
   tok/步 一模一样(15.40 / 16.38),**唯一的差别就是草稿那 k 次前向有没有走图**,
   结果比关投机还慢两成。这条路的成败确实压在"给草稿单独录图"上。
2. **最优点在 k=2**,1.444× / 1.407×。k 再往上,接受率掉得比每步产出涨得快:
   k=4 时 tok/步 从 15.40 涨到 18.29(+19%)但接受率从 57.2% 掉到 40.1%,
   净收益反而降到 1.399×;k=8 只剩 1.134×。
3. **n-gram 在 8B 上几乎没用了(1.023× / 1.034×),连它的主场 `repeat` 也一样。**
   07 报告在 Qwen3-0.6B 上测到 n-gram 最佳 1.66×,那里 `repeat` 的接受率是 71.9%;
   同样的负载换到 8B,接受率只剩 18.2%。原因是 8B 不再亦步亦趋地照抄 prompt 里
   重复的那一段,n-gram 猜的东西它不认。**"n-gram 吃文本重复性"这句话的前提是
   模型本身也照抄;模型越强,这个前提越不成立。** 草稿模型则两种负载都是 ~1.4×。
4. **TBT 变大但每 token 延迟变小。** k=2 的 TBT p50 从 23.89 涨到 31.47 ms,
   因为一步里多做了 2 次草稿前向;但一步产出 1.925 个 token(8 条并发下 15.40/8),
   摊到每个 token 是 16.3 ms,比关投机的 23.89 ms **好 1.46×**。
   看 TBT 会以为变慢了,要看 TBT/每步产出。

---

## 六、遗留与注意事项

### 6.1 已知的不变量松弛:一格,**不一定**只持续一轮

> ⚠ 这一节最初写成"窗口是一轮,下一轮就补上了"。**独立 review 指出那是错的**,
> 下面是更正后的版本。经过见第七节的 review 转述。

两套 cache 的同步不变量是"每次往目标 cache 写某个位置,草稿 cache 的同一位置也被写过"。
它在**一处**不成立:

`a == k`(全部草稿被接受)那一轮结束时,`hash_blocks` 登记的范围到位置 `len+k-1`,
而草稿只写到 `len+k-2`。若 `len+k-1` 恰好是某个 block 的最后一格,该 block 会带着
**一格过期的草稿 KV** 进 prefix cache 索引。

- **这条 seq 还能再跑一轮**:下一轮草稿的第一次前向(q=2)就把它补上,窗口一轮。
- **这条 seq 当场结束或被抢占**(`find_stop` 在同一次 postprocess 里命中 EOS/
  max_tokens,或紧接着被 preempt):补写**永远不会发生**。而
  `BlockManager.deallocate` 只把 block 还回空闲队列,**故意不删** `hash_to_block_id`
  的索引项(`block_manager.py:48-49` 只在该 block 被 `_allocate_block` 取走复用时
  才删)。于是这个 block 带着那一格过期草稿 KV **长期留在 prefix cache 索引里**,
  直到被回收覆盖。后来命中这段前缀的请求,`sync_draft_prefill` 会把
  `draft_num_cached_tokens` 直接设到那一格之后,于是它**整条生命周期**都带着
  那一格错的草稿 KV,不是一轮。

触发要同时满足:`a == k`(概率 ≈ α^k)∧ `(len+k) % block_size == 0`(≈ 1/256)
∧ 该轮恰好终止/被抢占 ∧ 之后有请求命中这段前缀 ∧ block 还没被回收。很罕见,
但**一旦发生就是持久的**,而且在长跑的服务里会累积。

**后果仍然只影响接受率,不影响输出 token。** 理由不变:目标模型的 KV 是权威,
接受与否完全由目标的 logits 决定;草稿只负责"提议"。

**为什么没有修**,两条路都试算过:

1. 让 `hash_blocks` 的登记范围跟草稿走 —— 会把那个 block **永久**排除出 prefix
   cache(下一轮 `start = num_cached // bs` 已经越过它),等于用目标侧的命中率
   去换草稿侧的一格,不划算。
2. 让草稿多跑一次前向把 `d_k` 的 KV 算出来 —— 每轮 k+1 次而不是 k 次前向,
   按 E3 的口径是草稿成本 +1/k(k=2 时 +50%),直接吃掉大半收益。
   这个缺口是结构性的:`d_k` 由第 k 次前向**产出**,它自己的 KV 只能由第 k+1 次
   前向来写。vLLM 有完全相同的结构(所以它才要多留 1 个 slot)。

所以选择记录在案,并把它列为**首要遗留问题**。真要修,方案 2 更干净,代价是明确的。

### 6.2 只支持单卡

`speculative_method="model"` 断言 `tensor_parallel_size == ep_size == 1`。
原因:草稿在 `ModelRunner.run()` 开头才产出,而 `Sequence.__getstate__` 在那之前
就被 pickle 给 worker 了,worker 侧拿到的 `scheduled_token_ids` 会缺草稿那几个 token。
要支持 TP 得把草稿循环的结果广播给 worker,或者把草稿挪到每个 rank 上各跑一遍。
TP > 1 本来也没实测过(05 报告遗留项 4)。

### 6.3 `draft_sample_method="random"` 的显存

通用接受式要保留 k 份 `[B, V]` 的 q(fp32)。`B=8, k=2, V=151936` 时 9.7 MB,
与 `compute_probs` 已经物化的 `[R, V]` 同量级;但 `B=512, k=8` 会到 2.5 GB。
本版没有做 micro-batch 拆分,大批量下要么用默认的 `greedy`(不需要 q),
要么自己压 `max_num_seqs`。默认值就是 `greedy`,与 vLLM 一致。

独立 review 补了一条:`sample_draft` 里那三个 `[B,V]` 的临时量
(`torch.zeros_like(probs)` + `scatter_` + `torch.where`,用来把 temperature=0
的行的 q 换成 one-hot)是**无条件**物化的,哪怕批里一行 greedy 都没有,
于是 `random` 模式下的瞬时占用还要再乘约 3 倍。要绕开就得加一次 host 同步
(`(temperatures == 0).any()`)或者原地改写 `probs`。`random` 不是默认模式,
本版先记着不动。

### 6.4 / 6.5 没有计入显存预算的两笔,加草稿模型后都变大了

`allocate_kv_cache` 的预算 = `total × util − used − peak + current`,其中 `peak`
来自 `warmup_model()`。有两笔不在里面,而且这次都翻了倍:

1. **草稿模型的激活峰值**。`warmup_model()` 在 `draft_ready` 置位**之前**跑,
   所以只跑了目标模型。我原本的论证是"草稿层数更少、hidden 更窄
   (28×1024 vs 36×4096),同样 token 数下激活严格更小,已被目标覆盖" ——
   独立 review 指出这个论证漏了最大的一项:`sync_draft_prefill` 会把最多
   `max_num_batched_tokens`(默认 **16384**)个 token 推过草稿模型,
   它估算是 0.2~0.5 GB。**这一条我接受**,论证原来是不完整的。
2. **4 族图的静态缓冲 + 图池**(目标 decode / 目标 varlen / 草稿 decode /
   草稿 varlen),都在 `allocate_kv_cache` **之后**分配。Step B 就有这个问题,
   草稿模型让它从 2 族变 4 族;review 估 0.3~0.5 GB。

两笔都是**推断/它的估算,没有实测**(它没有 GPU;我也没有单独量)。表现形式是:
`gpu_memory_utilization` 调得很高时,会在**录图阶段** OOM,而不是在分配 KV 时
干净地报"放不下"。`tests/bench_draft_model.py` 用的就是 0.90,实测没有撞上,
但余量比预算表上看到的小。

要根治就是在 `warmup_model` 里也跑一遍草稿的最坏 chunk,并把图的缓冲预扣掉。
本版为了控制改动面没做。

### 6.6 大 batch × 长上下文会亏

见 E3。`bs=32 / ctx=4096` 时草稿 2 步的成本已经是目标 1 步的 0.926,
需要 α > 0.584 才回本。**这条路的适用范围是"并发不太高、上下文不太长"。**
根因是 Qwen3-0.6B 的 KV 有 Qwen3-8B 的 78%(28×8×128 vs 36×8×128)——
换一个 KV 更小的草稿模型(更少层或更少 KV 头)会显著改善,本机没有这样的模型。

### 6.7 已经过时的注释(按约定只指出,没有改动)

`nanovllm/engine/model_runner.py` 的 `run_varlen_graph` docstring 里这两行:

> 缓冲区按最坏情况 bs*(1+k) 行开,多出来的行挂给一条 sink seq:
> 它的 k 长度为 0(FA 对 seqlen_k=0 的行输出恰为全 0,第一阶段 Q3 实测),

**与现在的代码不符,而且是被 07 报告的坑 10 推翻掉的那个说法。** 现状是:
padding 行分散到**多个** padding 槽,每槽 `k 长度 = q 长度 ≥ 1`,
正因为 `k 长度为 0` 会触发 flash-attn 的 early-exit 分支写越界。
`replay_varlen` 函数体里那段长注释写的是正确的现状,两处互相矛盾。

同一条 docstring 里"方案 C 的要点:草稿长度保持 ngram 给出的原样"现在也窄了 ——
这条路现在同时服务 n-gram 和草稿模型两种来源。

是否更新由你决定,我没有动它。

### 6.8 `test_m3_moe_local.py` 会改写自己的基线文件

跑完 MoE 那 5 项之后 `tests/baselines/greedy_moe_ep1.json` 会被重写。核对过:
**6 条 `token_ids` 逐条完全相同**,变的只有 `config.master_port`(本来就是每次随机挑
空闲端口)和本阶段新加的三个 argparse 键(`draft_sample_method` /
`no_draft_cudagraph` / `no_varlen_cudagraph`)。已 `git checkout` 还原,
工作区里没有留下这个改动。这是既有行为,不是本次引入的。

### 6.9 投机 + logprobs 仍然不能同时用

Phase 3 的遗留项 3,本次没有改变:投机路径返回 `logprobs=None`。
这也是为什么端到端等价性判据要靠"关投机那一侧的 logprobs"来做噪声分析。

---

## 七、独立 agent 的代码审查

改完之后起了一个独立 agent 做对抗式审查(明确让它"找 bug,不要验证"),重点是
任务书点名的四块:接受规则的数学正确性、两套 KV cache 的同步不变量、显存预算边界、
并发/回滚安全。它读代码、不跑 GPU。**下面如实转述,包括我不同意的部分。**

### 采纳并已修的

| # | 它说的 | 我的判断 | 处置 |
|---|---|---|---|
| 1 | §6.1 的"窗口是一轮"是错的:seq 在同一次 postprocess 里终止或被抢占时,补写永远不会发生,而 `deallocate` 故意保留 `hash_to_block_id`,于是污染是**持久**的 | **它是对的,我写弱了。** 我沿着 `block_manager.py:48-49` 复核了一遍,确实只有 `_allocate_block` 复用时才删索引 | 改报告 §6.1,并列为首要遗留问题;两条修法都算了代价,选择记录而不是修(理由见 §6.1) |
| 2 | "被跳过的 decode seq"那条路上,草稿 KV 停在原地而 `hash_blocks` 照常登记,整段 256 token 的 block 一格草稿 KV 都没有;而且我的 C3 用例**测不到**(第二阶段是最后一阶段,污染的 block 再没人读) | **它是对的。** 我的坑 3 修法只堵住了 `scheduled_token_ids` 被顶掉,没维护不变量;而我在报告里把不变量写成无条件成立 | **改代码**:新增 `sync_draft_decode`,这些 seq 也跑一次草稿前向把 KV 追平。实测 `--pollution` 的 `exec_draft_graph_varlen` 从 16 变成 63,恰好等于 63 个 decode step。新增 C3b 用例直接查这个计数(修前 16≠63 会 FAIL) |
| 3 | 通用接受式没有 NaN / 残差全零的保护:NaN → `rand < NaN` 恒假 → 必拒绝 → 残差整行 NaN → `argmax` 返回下标 0 → **token 0 被当成真 token 吐出去** | **它是对的**,而且它很克制地标注了"没能确认 NaN 真的会出现"。但配合 1/2 两条(未写过的草稿 KV 来自 `torch.empty`)这条路是通的 | **改代码**:残差退化(`rsum<=0` 或 NaN)时回退到从 p 采样;`ratio` 的 NaN 归零。新增单元用例,实测 NaN 情形下 P(token 0)=0.144 而不是 ~0.84 |
| 4 | 重构把 decode 图族的 `max_bs` 从 `min(max_num_seqs,512)` 换成 `graph_bs[-1]`,`max_num_seqs ∈ {3,5,6,7}` 时两者不等,目标路径的 bf16 结果会**静默**改变 | **它是对的,这条最尖锐**,我没意识到旧代码里 `outputs[:8]` 在 5 行缓冲区上只切 5 行 —— "8 号桶"其实录的是 5 行图 | **改代码**:`max_bs` 改由调用方传入,原样恢复旧口径 |
| 5 | 新的块数下限断言写在 `if num_kvcache_blocks < 0` **之外**,显式指定小块数(测试用来复现抢占)时也会被它挡 | 对 | **改代码**:挪进自动定块数的分支 |
| 7 | `q_d.clamp_min(1e-10)` 只用在比值上、没用在残差上,两者口径不一致(TV 误差 ≤ V·1e-10 ≈ 1.5e-5) | 数值上无所谓,但口径不一致确实是缺陷。它顺带指出 vLLM 是"`q(d)==0` 直接判拒绝"而不是 clamp | **改代码**:与 vLLM 对齐,`q(d)==0` 判拒绝 |
| 8 | `p_d_full` 里存的是 `ratio` 不是 `p_d`,名字误导 | 对 | **改代码**:改名 `ratio_full` 并加注释 |

### 转述但没改的

| # | 它说的 | 我的处置 |
|---|---|---|
| 6 | `min(max_num_seqs, 8)` 这个下限**不保证不活锁**,并给了反例(8 块、一条长 2047 的 seq、k=2:`can_allocate` 过得去,`ensure_capacity` 过不去,自己抢自己,重 prefill 后再撞同一堵墙)。它认为真正能保证前进的下限是 `ceil(max_model_len / kvcache_block_size)`,但那会让现有 `gpu_memory_utilization=0.35` 的测试全起不来 | **接受它的判断**。反例我验算过,成立。保留这个下限但**不再宣称它防活锁** —— 代码注释和本报告都改成"只是别把 1 个 block 也当成能跑的冒烟门槛",并把它推荐的真下限和代价一起写进注释 |
| 9 | `sample_draft` 在 `random` 模式下无条件物化 3 个 `[B,V]` fp32,`B=512,k=2` 时比 §6.3 记的 622MB 还多两倍 | 属实,但要绕开就得加一次 host 同步,或者原地改写 `probs`。`random` 不是默认模式,先记在遗留里不动 |
| 10 | 未计入预算的显存现在翻倍了,而且 `sync_draft_prefill` 会把最多 `max_num_batched_tokens`(默认 16384)个 token 推过草稿模型,这一项比我估的大 | 接受,已补进 §6.4 / §6.5。它也说明了自己没法在没有 GPU 的情况下量化 |
| 11 | `sync_draft_prefill` 在"草稿根本不会提议"的场景下也照跑,浪费 | **不同意。** prefill 的 KV 同步正是维持不变量的东西,而且引擎无法预知"以后也不会有人读它" —— 恰恰是findings 1/2 那类问题的来源。跑它是正确的,不是浪费 |
| 8(前半) | "`:534` 算了 `p_d` 但从没用过" | **不同意,这条它看错了。** `p_d` 就在下一行喂给 `ratio = p_d / q_d...`。同一条的后半(`p_d_full` 命名)是对的,已改 |

### 它查过、没找到问题的地方

行索引对齐(它手工推了一个 `[D1(k=2), D2(k=2), P1(prefill)]` 的混批,逐行核对
`rows` 与 `stack(draft_probs,dim=1).reshape(-1,V)` 的行序)、
`min(L+a, L+k-1)` 的簿记算术(独立重推了一遍,含 k=1)、
`slots_for` 不会越过 `ensure_capacity` 的预留、
`BlockManager.truncate` 不会回收刚被 `hash_blocks` 登记的 block(登记下标与弹出下标不相交)、
两套 cache 的 block 数由构造保证相同、
四族图共用一个 graph pool 是安全的、
以及 —— 值得单独记一笔 —— **它复核了 `q_len=0` 的 padding 槽是安全的**:
FA2 的 `compute_attn_1rowblock` 在 `m_block*kBlockM >= actual_seqlen_q` 处就返回了,
根本到不了 07 报告坑 10 那个写 LSE 的地方;只有 `q_len>0 ∧ k_len==0` 才危险,
而 `dst_k = dst_q - num_tokens` 这个构造让它不可能出现。

---

## 八、测试与复现命令

```bash
cd nano-vllm

# ── 第一阶段(不需要改任何项目文件就能跑)────────────────────
.venv/bin/python tests/phase3b_probes/e1_pairing.py             # E1 配对
.venv/bin/python tests/phase3b_probes/e23_budget_and_cost.py    # E2 显存 + E3 成本比(~12 分钟)
.venv/bin/python tests/phase3b_probes/e4_draft_kv_bookkeeping.py # E4 簿记模拟(秒级)

# ── 第二阶段:正确性 ────────────────────────────────────────
.venv/bin/python tests/test_phase3b_draft_model.py    # 本阶段新增
.venv/bin/python tests/test_phase3_spec.py            # 回归 14 项
.venv/bin/python tests/test_phase25_varlen.py         # 回归 25 项
.venv/bin/python tests/test_m3_moe_local.py           # 回归 5 项
.venv/bin/python tools/check_comments.py              # 注释一行都不能少

# ── 内存安全(环境变量和 memcheck 两个都不能少)────────────────
# 少了 PYTORCH_NO_CUDA_MEMORY_CACHING=1,caching allocator 会把池内越界完全挡住
# —— 07 报告坑 10 就是这么差点漏掉一处真实的写越界。
# 两个 mode 都要跑;不能把整个引擎塞进去跑(会撞 capture 冲突,见坑 8)。
for m in varlen eager; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    ~/cuda-12.8/bin/compute-sanitizer --tool memcheck \
    .venv/bin/python tests/phase3b_probes/memcheck_draft.py $m
done

# ── 性能 ───────────────────────────────────────────────────
.venv/bin/python tests/bench_draft_model.py           # 6 配置 × 2 负载

# ── 单跑一个配置 ────────────────────────────────────────────
.venv/bin/python tests/gen.py --out /tmp/x.json \
    --model ~/huggingface/Qwen3-8B \
    --speculative-model ~/huggingface/Qwen3-0.6B \
    --num-speculative-tokens 2 --speculative-method model \
    --gpu-util 0.90 --max-model-len 2048 --max-num-seqs 16
```

新增文件:

```
nanovllm/                      改 4 个文件(config / sequence / scheduler / model_runner)
tests/phase3b_probes/          e1_pairing.py  e23_budget_and_cost.py
                               e4_draft_kv_bookkeeping.py  memcheck_draft.py
tests/test_phase3b_draft_model.py
tests/bench_draft_model.py
Plan-1-2-3/08-phase3b-draft-model-plan.md      动手前写的方案
Plan-1-2-3/08-phase3b-draft-model-report.md    本文件
var.md                         登记 draft_sample_method / draft_cudagraph
```
