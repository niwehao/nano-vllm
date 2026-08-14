# 实现总结报告

**环境**:NVIDIA L40S 46GB / torch 2.8.0+cu128 / flash-attn 2.8.3 / triton 3.4.0
**模型**:Qwen3-0.6B(`~/huggingface/Qwen3-0.6B`,bf16),全部结果为真实模型跑出
**代码**:`nanovllm/` 10 个文件 +529 / −85 行;新增 `tests/` 测试套件

**测试结果:4 个套件,42 项检查,全部通过。**

```
[PASS] Phase 1 单元 · Sampler                12/12
[PASS] Phase 1 端到端 · 采样/logprobs/基线     7/7
[PASS] Phase 2 · 统一调度器(混批)             9/9
[PASS] Phase 3 · 投机解码                     14/14
```

---

## 一、改了哪些部分

### Phase 1 · Sampler 重写

| 文件 | 改动 |
|---|---|
| `sampling_params.py` | 新增 `top_p` / `top_k` / `logprobs`;删掉 `assert temperature > 1e-10`,改为 `temperature == 0` 表示 greedy |
| `layers/sampler.py` | 重写。拆出三个可复用函数:`apply_top_k_top_p`、`compute_probs`、`sample_from_probs`;`Sampler.forward` 支持 greedy/采样混批 + logprobs |
| `engine/sequence.py` | 存 `top_p`/`top_k`/`num_logprobs`,累积 `completion_logprobs` |
| `engine/model_runner.py` | `prepare_sample` 产出四元组;新增 `unpack_logprobs` |
| `engine/scheduler.py` | `postprocess` 接收并落盘 logprobs |
| `engine/llm_engine.py` | `generate()` 输出增加 `logprobs` 字段 |

关键设计:
- greedy 与采样**同批共存**,靠 `torch.where(temperatures == 0, greedy_tokens, sample_tokens)` 按行合并,不拆 batch。除法前用 `safe_t` 避开除零。
- `logprobs` 取"温度缩放后、top_k/top_p 截断前"的 `log_softmax`(与 vLLM 一致)。
- 全 batch 都没开 top_k/top_p 时 `prepare_sample` 传 `None`,`apply_top_k_top_p` 整段跳过那次 `[B, 152k]` 的 sort。判断在 CPU 上做,不引入 GPU 同步。
- 摘掉了原来的 `@torch.compile`(引入 sort/topk/条件分支后先保正确性)。

### Phase 2 · 统一调度器(prefill/decode 混批)

| 文件 | 改动 |
|---|---|
| `engine/scheduler.py` | `schedule()` 重写:**decode 优先拿预算,剩余给 prefill**,返回 `list[Sequence]` 而不是 `(seqs, is_prefill)`;新增调度统计 |
| `engine/model_runner.py` | `prepare_prefill` → `prepare_batch`(prefill/decode 统一);`prepare_decode` 降级为纯 decode 快路径专用;`is_pure_decode` / `use_cudagraph` 拆成两个判断 |
| `engine/sequence.py` | 新增 `token_offset` + `scheduled_token_ids`;`__getstate__` 改为只传本轮切片 |
| `engine/llm_engine.py` | `step()` 返回 `(outputs, num_prefill_tokens, num_decode_tokens)` |
| `engine/model_runner.py` | `allocate_kv_cache` 支持显式指定 `num_kvcache_blocks`(测试里稳定复现抢占) |

计划里的判断被验证是对的:**`prepare_decode` 本来就是 `prepare_prefill` 在 `start = len-1` 时的退化情形**(postprocess 后恒有 `num_cached_tokens == len(seq) - 1`),所以两者可以真正合一,attention 层一行没改 —— 原来 `if context.block_tables is not None` 那个分支本来就是通用的 paged 读取,统一路径下自动生效。

### Phase 2.5 · CUDA graph(Step A)

按计划采用**双路径共存**:纯 decode 批走 `prepare_decode` + `flash_attn_with_kvcache`(可 replay 那 36 张图),混批走 `prepare_batch` + `flash_attn_varlen_func`(eager)。36 张图**没有作废**。Step B(用 varlen 形态重录图)未做 —— 它是纯优化,不影响功能,且 Phase 3 的验证前向本来就是变长。

### Phase 3 · 投机解码

| 文件 | 改动 |
|---|---|
| `config.py` | `num_speculative_tokens` / `speculative_method` / `ngram_prompt_lookup_max` / `ngram_lookup_window` |
| `engine/scheduler.py` | `propose_draft()`(n-gram prompt-lookup);decode 段按 k+1 预留容量;`postprocess` 支持一步多 token + 回滚簿记;`find_stop` 逐 token 检查停止条件 |
| `engine/block_manager.py` | 新增 `ensure_capacity`(一次预留 k+1 个位置)、`truncate`(收回多预留的块) |
| `engine/model_runner.py` | `sample_speculative`(rejection sampling);`prepare_batch` 产出 `logits_indices`;`prepare_sample` 按行展开采样参数 |
| `layers/embed_head.py` | LMHead 按 `logits_indices` 取行 |
| `utils/context.py` | 新增 `logits_indices` 字段 |
| `engine/sequence.py` | `draft_tokens`、`truncate_to` |

接受规则的推导:n-gram 是**确定性提议**,等价于草稿分布 `q = δ_d`。代入 Leviathan 的规则:

```
以概率 min(1, p(d)/q(d)) = p(d) 接受 d
拒绝时从 norm(max(p − q, 0)) = "把 d 挖掉再归一化的 p" 采样
```

合起来恰好还原 `p`(单元测试蒙特卡洛验证,max|Δp| = 0.0009)。greedy 单独走一条路:直接比 token 而不是比概率。

**顺带收益**:`logits_indices` 机制把 chunked prefill 中间块的无用行也剔掉了 —— 那些行原来每轮都要过一次 `lm_head`(15 万词表的 GEMM)再丢弃。

---

## 二、踩过的坑(按遇到顺序)

### 坑 1 · bf16 logits 里的并列最大值 —— 唯一一个真 bug

**现象**:Phase 1 端到端第一次跑,7 项里挂了 4 项。最诡异的一条:同一次前向、同一行 logits 上,
```
greedy(argmax)        → token 82
log_softmax 的 top-1  → token 6266
top_k=1               → token 1180
top_p=1e-5            → token 6266
```
四个"最大值"给出三个不同答案,这在数学上不可能。

**排查**:写了 `tests/debug_ties.py`,monkeypatch `Sampler.forward`,打印每行"等于最大值的元素个数":

```
第 1 次: logits dtype=torch.bfloat16 -> torch.float32
  每行并列最大值个数: [1, 1, 1, 1, 3, 1]     ← 第 5 行有 3 个并列
```

**根因**:`lm_head` 输出是 bf16,只有 8 位尾数。词表 15 万个词,logits 量级在 8~32 之间,该区间一个 ulp 是 0.0625 —— **出现完全并列的最大值是常事**。而我的 `apply_top_k_top_p` 用的是"取第 k 大的值当阈值,砍掉所有小于它的",并列时会把并列项**全部保留**,于是 `top_k=1` 保留了 3 个候选,Gumbel 采样在里面随机挑了一个。

**修复**:改成按**排序位置**截断而不是按值:

```python
logits_sort, logits_idx = logits.sort(dim=-1, descending=True, stable=True)
pos = torch.arange(V, device=...).unsqueeze(0)
logits_sort = logits_sort.masked_fill(pos >= top_ks.unsqueeze(1), -inf)   # 恰好保留 k 个
```

`stable=True` 保证并列时保留原始下标最小的那个,与 `torch.argmax`"返回第一个最大值下标"的行为一致。于是 **greedy == top_k=1 == top_p→0 在并列情况下也严格相等**。top_p 相应改成降序 `prev_sum` 形式。

**教训**:这不是测试用例的问题,是实现的真缺陷。`top_k=k` 的语义应该是"至多 k 个候选",按值阈值截断会违反这一点。写了两条回归单测锁住(`test_ties_greedy_topk1_topp_agree`、`test_top_k_exact_count`)。

### 坑 2 · 用随机 token id 当测试 prompt 是错的

最初 `build_prompts()` 用 `random.randint(1000, 5000)` 生成 600 个 token 当长 prompt。看起来"随机=覆盖面广",实际上完全相反:模型看到乱码时输出接近**均匀分布**,logits 挤在一起,bf16 下并列和近似并列遍地都是,greedy 对浮点噪声变得极度敏感。这既不代表真实负载,也让所有等价性比对变成掷硬币。

换成真实自然语言语料(内嵌一段讲 LLM 推理的英文长文,切出 600/512+88/256/257 四种长度),分布变得峰值明显,比对才有意义。

### 坑 3 · "逐 token 一致"这个判据本身站不住

计划里写了 Phase 1 必须逐 token 一致。修完坑 1 后,`top_k=1 == greedy` 之类都 6/6 过了,但 **eager vs cudagraph 只有 4/6**。

我没有直接放宽阈值,而是做了**对照实验**:`git stash` 回到改动前的原始代码,用只依赖原始 API 的脚本(`temperature=1e-9` 近似 greedy)跑同样的比对:

```
原始代码 eager vs graph:        4/6 条一致(分歧在同样的 seq 上)
原始代码 batch vs 串行:          3/6 条一致
改动后   batch vs 串行:          4/6 条一致(反而更好)
```

**结论**:这是改动前就存在的固有现象,来源有两个 —— CUDA graph 把 batch 从 6 padding 到桶大小 8,以及 cuBLAS GEMM 按 M 维分块。两者都改变归约顺序,在 bf16 下产生 ulp 级漂移。

于是重新设计判据,分两层:

1. **单步 logprob 比对**(主判据,锐利):只生成 1 个 token,直接比 top-10 的 logprob。不经自回归放大,能把"数学等价"和"逻辑 bug"清晰分开。判据是**首选 token 必须完全一致** + logprob 偏差 ≤ 4 ulp。
   - chunked prefill:argmax 6/6 一致,偏差 2.7 ulp
   - prefix cache 命中:argmax 2/2 一致,偏差 2.0 ulp
2. **长输出的分歧点性质分析**(辅判据):分歧点上两个候选的 logprob 差必须落在噪声量级。

### 坑 4 · 噪声阈值不能拍脑袋 —— 用数据定

第 3 条里的"噪声量级"最初我随手写了 0.15,结果 Phase 3 有一条 0.25 被判成"真分歧"。与其调阈值让它过,不如先量一下**分歧点到底特殊不特殊**:

```
spec-off 全部 768 个位置的 top1−top2 logprob 差:
  gap <= 0.0625 (1 ulp):   11 个位置 ( 1.4%)
  gap <= 0.1250 (2 ulp):   35 个位置 ( 4.6%)
  gap <= 0.2500 (4 ulp):   66 个位置 ( 8.6%)
  中位数 3.5000 (= 56 ulp), 均值 4.3564
```

而实际观测到的**每一个**分歧点:

```
seq[0] @ 24: gap = 2.0 ulp     seq[0] @ 30: gap = 0.0 ulp
seq[2] @ 30: gap = 2.0 ulp     seq[2] @ 40: gap = 2.0 ulp
seq[5] @ 40: gap = 0.0 ulp     seq[1] @ 62: gap = 2.0 ulp
```

全部落在最并列的 4.6% 里,而位置的中位 gap 是 56 ulp。**逻辑 bug 会在随机位置发作,不可能只挑最并列的那几个百分点出现。** 据此把阈值定为 4 ulp,并让比对工具打印每个分歧点在整体分布中的分位数供人工复核。

另一个佐证:Phase 3 里出现的分歧位置和 token 对(`seq[0]@24: 382 vs 13`、`seq[0]@30: 785 vs 16141`、`seq[2]@30: 3460 vs 2696`),与**原始代码 batch vs 串行**时出现的完全是同一批。

### 坑 5 · Phase 2 的性能回退 −18%,原因是两个判断被合成了一个

统一调度做完后基准显示稳态 decode 吞吐从 **987.7 掉到 809.3 tok/s**,单步 16.20ms → 19.44ms。

根因:我写了一个 `use_cudagraph()` 同时决定两件事 —— 用哪个 attention kernel、要不要 replay 图。`enforce_eager=True` 时它返回 False,于是 eager 下的**纯 decode 批也被赶去走变长路径**。而 `flash_attn_with_kvcache` 是专为 q=1 做 split-K 优化的,通用变长 kernel 在 q=1 上明显吃亏。

拆成 `is_pure_decode()`(决定 kernel/prepare)和 `use_cudagraph()`(决定 replay)之后:

```
decode 吞吐  809.3 → 967.3 tok/s(与原版 987.7 差 2%,在噪声内)
场景总耗时   3.88 → 3.32 s(反而比原版 3.62s 快 8.3%)
```

**教训**:"是否能用快 kernel"和"是否能用 CUDA graph"是两个正交条件,合并会悄悄丢性能。

### 坑 6 · TP pickle 在混批后会越界

`Sequence.__getstate__` 原来是 `last_state = self.last_token if not self.is_prefill else self.token_ids` —— decode 时只传一个 `last_token`,`token_ids` 在 worker 上是空的。统一调度后 worker 的 `prepare_batch` 对 prefill/decode 一视同仁地取 `scheduled_token_ids`,decode seq 上会取到空切片。

修法:引入 `token_offset`,约定 `token_ids[i]` 的绝对位置是 `token_offset + i`。driver 侧恒为 0,worker 侧等于 `num_cached_tokens`。`__getstate__` 一律只传本轮被调度的那一段(通信量和原来 decode 路径一样是 O(本轮 token 数),不传整条历史)。

### 坑 7 · chunked prefill 的"只允许第一条截断"判据失效

原代码:`if remaining < num_tokens and scheduled_seqs: break`。统一调度后 `scheduled_seqs` 里通常已经躺着一批 decode,这个条件会让**只要有请求在 decode,长 prompt 就永远排不进来**。

改成 `len(scheduled_seqs) > num_decode_seqs` —— 判断的是"本轮已经排了别的 prefill 没有"。

### 坑 8 · 抢占场景构造不出来

想验证抢占路径,先试 `gpu_memory_utilization=0.062`,结果 `preempted: 0`。原因:显存不够时 `can_allocate` 会先把 prefill 挡住,请求排队进来,根本轮不到抢占 —— 抢占只在**已在 running 的 seq 追加不了新块**时才发生。

正确构造:6 条**恰好 255 token** 的 prompt(每条 1 块装得下,互不共享前缀),KV block 总数卡死在 **7**。生成到第 257 个 token 时每条都需要第 2 块,只剩 1 块空闲 → 必然抢占。为此给 `allocate_kv_cache` 加了"显式指定 block 数"的支持。

结果:触发 3 次抢占,**抢占重算后输出与不抢占逐 token 完全一致 6/6**(这是个很强的信号 —— 重算路径没有任何状态残留)。

### 坑 9 · 基准测试被一次性 kernel 自动调优开销淹没

第一版性能基准显示 TBT max 旧 795ms / 新 746ms,几乎没差别,而且新版总耗时反而更长。检查发现 746ms 是**单个 step** 的耗时 —— 一个 2000-token 的 prefill 在 0.6B 模型上不该超过 20ms。真相是 flash-attn / cuBLAS 首次遇到新形状时的 kernel 选择开销,一次性的,把信号完全盖住了。

重写基准:先跑两轮 throwaway 负载把这些开销付掉,再分三个场景独立测量。信号立刻清晰(见下表)。

同时第一版场景设计也不对:4 条长 prompt、`max_num_batched_tokens=2048`,每条 prompt 一个 step 就 prefill 完,"连续阻塞"根本没形成。改成 `max_num_batched_tokens=512` + 3 条 4000-token prompt,旧调度下每条要切 8 个 chunk,连续 19 个 step 不出词 —— 这才是要测的东西。

### 坑 10 · `hash_blocks` 的登记范围(计划里预判到的那个)

`num_scheduled_tokens` 必须写**实际接受数 1+a**,而不是提议数 1+k。写成 1+k 的话,一旦 `L+k` 跨过 block 边界而 `L+a` 没跨,`hash_blocks` 就会把一个含"被拒绝位置垃圾 KV"的 block 登记进 prefix cache 索引。实现时严格按 1+a 写,并写了专项测试(见下)。

被拒绝位置的残留 KV **不需要主动擦除**:slot 由 position 唯一决定,将来会被真实 token 原地覆盖;attention 读取受 `cu_seqlens_k` 限制,读不到残留区。唯一能把残留暴露出去的通道就是 hash 登记,堵住即可。

另外 `ensure_capacity` 可能为 `L+a+1..L+k-1` 多开了块,验证后必须 `truncate` —— 否则 `block_table[-1]` 指错,下一轮的 slot 计算和 `last_block_num_tokens` 全错。

### 坑 11 · 一次产出多个 token 时的停止条件

投机一步可能吐出 a+1 个 token,其中任何一个都可能撞上 eos 或 max_tokens。原来的 `postprocess` 只检查一个 token。新增 `find_stop()` 逐个检查,并用 `seq.truncate_to()` 裁掉停止点之后的部分。

顺序很关键:**先 append 全部、再 hash、最后裁**。因为 `hash_blocks` 按 block 的实际内容算 hash,而 KV 里存的就是裁剪前的内容 —— 先裁再 hash 会让 hash 和 KV 对不上。裁剪后这条 seq 立即 FINISHED 并释放全部 block,不会有不一致残留。

### 坑 12 · 小工具坑

- 用字符串替换给 `context.py` 加字段时,`_CONTEXT = Context()` 在文件里出现了两次,`str.replace` 全替了,产生了缩进错误。之后所有批量改动都跟一次 `ast.parse` 语法检查。
- `LLM` 是 `LLMEngine` 的子类(`class LLM(LLMEngine): pass`),没有 `.llm_engine` 属性,测试脚本里写错过一次。

---

## 三、每一步的效果

### Phase 1 · Sampler

功能从"只有 temperature"扩到 **greedy / top_k / top_p / logprobs 四样齐全**。

| 检查项 | 结果 |
|---|---|
| greedy == argmax | PASS |
| top_k=1 == greedy(端到端 128 token) | 6/6 逐 token 一致 |
| top_p→0 == greedy(端到端 128 token) | 6/6 逐 token 一致 |
| 并列最大值下三者仍相等 | PASS(修坑 1 前失败) |
| 全并列时 top_k 恰好保留 k 个 | PASS(修坑 1 前保留全部) |
| 采样频率收敛到理论分布 | max\|Δp\| = 0.0034 |
| logprobs == log_softmax 对应项 | PASS |
| greedy 重复运行确定性 | 6/6 |

### Phase 2 · 统一调度器

基准配置:eager,`max_num_batched_tokens=512`,8 条请求在 decode 稳态中,第 10 步灌入 3 条 4000-token 长 prompt(每条要切 8 个 chunk)。

| 指标 | 原始调度 | 统一调度(中间版) | 统一调度(最终) | 最终 vs 原始 |
|---|---:|---:|---:|---|
| **最长 decode 停摆** | 350.4 ms | 18.4 ms | **18.6 ms** | **−94.7%(18.8×)** |
| **连续 prefill-only 步数** | 19 | 1 | **1** | **19 → 1** |
| **TBT p99** | 34.97 ms | 19.78 ms | **19.11 ms** | **−45.4%** |
| **TBT max** | 367.0 ms | 37.9 ms | **34.97 ms** | **−90.5%** |
| TBT p50 | 16.34 ms | 19.49 ms | 16.39 ms | +0.3% |
| 场景总耗时 | 3.62 s | 3.88 s | **3.32 s** | **−8.3%** |
| 稳态 decode 吞吐 | 987.7 tok/s | 809.3 tok/s | 967.3 tok/s | −2.1%(噪声内) |
| prefill 吞吐 | 26976 tok/s | 27078 tok/s | 26728 tok/s | −0.9% |
| 混批 step 数 | 0 | 19 | 19 | — |

一句话:**长 prompt 到达时,已在跑的请求的最长卡顿从 350ms 降到 19ms,而稳态吞吐没有损失,总耗时还快了 8%。**

正确性:

| 检查项 | 结果 |
|---|---|
| chunked prefill 单步等价 | argmax 6/6 一致,logprob 偏差 2.7 ulp |
| prefix cache 命中单步等价 | argmax 2/2 一致,偏差 2.0 ulp |
| 抢占重算后输出 | **6/6 逐 token 完全一致**(触发 3 次抢占) |
| 混批 vs 非混批 64 步输出 | 0 条真分歧 |
| Phase 1 基线回归 | **6/6 完全一致** |

### Phase 3 · 投机解码

正确性(数学层,不允许任何误差):

| 检查项 | 结果 |
|---|---|
| rejection sampling 保分布(4 万次蒙特卡洛) | max\|Δp\| = **0.0009** |
| 劣质草稿的接受率 | 实测 **0.0002**,理论 p(d) = **0.0002** |
| greedy 接受规则 | 严格"接受当且仅当草稿==argmax" |
| 投机 greedy 重复运行确定性 | 6/6 逐 token 一致 |
| **prefix cache 污染专项** | 首步 argmax **6/6 一致**,偏差 2.0 ulp |
| 抢占 + 投机 | **6/6 逐 token 完全一致**(触发 4 次抢占) |
| k=1/2/4/8 开关投机 | 0 条真分歧,全部分歧落在最并列的 1.4%~6.8% 位置 |

污染专项的构造值得一提:第一阶段用 k=8 跑 64 步(必然跨 256 边界、且大量草稿被拒绝),第二阶段用 `prompt + 前 32 个生成 token` 当新 prompt,必然命中第一阶段写入的 block,**且第二阶段关掉投机** —— 于是任何差异都只能来自"读到的缓存 KV"。如果把被拒绝位置的垃圾 KV 错误登记了,首步 logits 会天差地别,绝不可能只差 2 ulp。

性能(8 条请求,每条 448 输出 token,greedy):

| 负载 | k | 耗时 | 步数 | tok/步 | tok/s | 接受率 | 加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| generic | 0 | 7.28s | 448 | 8.00 | 492.3 | — | 1.00× |
| generic | 1 | 5.68s | 305 | 11.75 | 631.4 | 71.1% | 1.28× |
| generic | **2** | 5.00s | 266 | 13.47 | **717.3** | 62.3% | **1.46×** |
| generic | 4 | 6.19s | 302 | 11.87 | 579.3 | 43.6% | 1.18× |
| generic | 8 | 5.11s | 239 | 15.00 | 701.7 | 28.8% | 1.43× |
| repeat | 0 | 7.34s | 448 | 8.00 | 488.0 | — | 1.00× |
| repeat | 1 | 6.03s | 324 | 11.06 | 594.6 | 75.7% | 1.22× |
| repeat | 2 | 6.90s | 340 | 10.54 | 519.3 | 70.7% | 1.06× |
| repeat | **4** | 4.41s | 232 | 15.45 | **812.5** | 65.1% | **1.66×** |
| repeat | 8 | 4.44s | 232 | 15.45 | 806.8 | 47.0% | 1.65× |

`repeat` 是 prompt 里同一段文字出现两次的负载(n-gram 的理想场景)。**最佳 1.66×,不需要第二个模型。** k 增大时接受率下降但每步产出增加,最优点在 k=2~4。

---

## 四、遗留与注意事项

### 未做的部分

1. **Phase 2.5 Step B**(用 varlen q=1 形态重录 CUDA graph,消灭双路径)。Step A 的双路径已经工作且没有性能损失,Step B 是纯代码整洁性优化,风险集中在 `max_seqlen_k` 是 host 标量会被烧进图。
2. **草稿模型路径**(`speculative_method="model"`)。配置项和断言已就位,`sample_speculative` 的接受规则对真实的 `q` 分布同样适用(只需把 `q = δ_d` 换成草稿模型的分布),但两套 KV cache 的分配与草稿模型的 prefill 同步未实现。本机只有 Qwen3-0.6B 一个模型,也没有合适的草稿模型可搭配。
3. **投机 + logprobs 不能同时用**。投机一步产出多个 token,logprobs 的语义需要另行设计,目前投机路径直接返回 `None`。
4. **TP > 1 未实测**(本机单卡)。相关改动(`__getstate__` 只传本轮切片、worker 侧 `token_offset`)在设计上是自洽的,但没有真机验证。

### 已经过时的用户注释(按约定只指出,没有改动)

下面这些注释是你写的,代码改动后它们的描述已经和现状不符,是否更新由你决定:

| 位置 | 注释内容 | 现状 |
|---|---|---|
| `layers/attention.py:65` | `# prefix cache` | 这个分支现在是**通用的 paged 读取**,不再只服务 prefix cache;统一调度后所有非 warmup 的前向都走它 |
| `utils/context.py:41` | "prefill 时只有命中 prefix cache 才需要" | 现在一律需要 `block_tables` |
| `utils/context.py:8-9` | `is_prefill` 的说明 | 语义已漂移成"走变长统一路径";纯 decode 快路径才是 False |
| `engine/llm_engine.py:81` | "num_tokens: 正数表示 prefill,负数表示 decode" | 协议已改成分别返回 `num_prefill_tokens` / `num_decode_tokens` |
| `engine/llm_engine.py:80` | "output: 元素是 (seq_id, 该 seq 生成的全部 token)" | 现在元素是 `(seq_id, seq 对象)` |
| `engine/llm_engine.py:91-92` | "每轮只更新其中一个" | 混批时两个可能同时更新 |
| `layers/sampler.py:78` | "sampler -> token_ids [T] 每条 seq 挑出一个词" | 投机时一步可能产出多个词 |
| `engine/scheduler.py:188` | "chunked prefill 没喂完就到此为止" | 描述仍准确,但触发条件里的 `is_prefill` 前缀已去掉(等价) |

---

## 五、测试套件说明

```
nano-vllm/tests/
├── run_all.py             一键跑完四个套件
├── common.py              真实语料 prompt 集(短/长/共享 512 前缀/block 边界/边界+1)
├── harness.py             子进程运行 + 三种比对判据
├── gen.py                 通用生成运行器(一种配置一个进程)
├── test_sampler.py        Phase 1 单元测试(12 项,不加载模型)
├── test_phase1_e2e.py     Phase 1 端到端 + 建 greedy 基线(7 项)
├── test_phase2_mixed.py   Phase 2 混批/分块/抢占/prefix cache(9 项)
├── test_phase3_spec.py    Phase 3 投机解码(4 单元 + 10 端到端)
├── bench.py               Phase 2 性能基准(新旧代码都能跑)
├── bench_spec.py          Phase 3 加速比基准
├── debug_ties.py          坑 1 的排查脚本(保留)
├── minimal_run.py         对照实验脚本(只用原始 API,git stash 后可直接跑)
└── baselines/             greedy 回归基线 JSON
```

三种比对判据:

| 判据 | 用途 | 严格程度 |
|---|---|---|
| `check_equal` | 抢占、确定性、top_k=1==greedy | 逐 token 全等,不允许任何差异 |
| `compare_logprobs` | chunked prefill、prefix cache、污染专项 | 单步:argmax 必须一致 + logprob 偏差 ≤ 4 ulp |
| `check_equal_or_noise` | 长输出的跨 kernel 路径比对 | 分歧点必须落在近似并列位置,并打印分位数 |

`bench.py` 和 `minimal_run.py` 刻意只依赖原始 API,`git stash` 回到改动前可以直接跑,用于做对照实验 —— 坑 3 和坑 5 都是靠它们定位的。

### 复现

```bash
cd nano-vllm
.venv/bin/python tests/run_all.py        # 全部测试,约 25 分钟
.venv/bin/python tests/bench.py tests/out/bench_new.json new    # Phase 2 基准
.venv/bin/python tests/bench_spec.py                            # Phase 3 加速比
```
