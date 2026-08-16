# 可配置参数登记表

`Config` 的全部字段都在 `nanovllm/config.py`,通过 `LLM(model, **kwargs)` 传入
(`llm_engine.py:17-20` 只挑 `Config` 里存在的字段,其余 kwargs 被忽略)。

## Phase 3B 草稿模型 新增

| 名字 | 位置 | 默认 | 作用 |
|---|---|---|---|
| `draft_sample_method` | `nanovllm/config.py:36` | `"greedy"` | 草稿模型怎么采下一个草稿 token,决定接受规则走哪一支。`"greedy"` = 取 argmax,草稿分布 q 是 one-hot δ_d,通用式 `min(1,p/q)` 精确退化成 `p(d)`,不需要保存 `[B,k,V]` 的 q(vLLM 的默认也是这个,`config/speculative.py:290`)。`"random"` = 按 q 采样,走通用接受式,显存多花 `k·B·V·4B`。两者产出的 token 分布**都**严格等于目标分布 p,只影响接受率。 |
| `draft_cudagraph` | `nanovllm/config.py:37` | `True` | 是否给草稿模型录两族 CUDA graph(q=1 的 decode 族 + `q_max=2` 的 varlen 族)。`False` 时草稿的 k 次前向全走 eager。**只该用于 A/B 对照** —— 第一阶段 E3 实测 bs=8/ctx=512 时草稿不图化的成本比是 1.326,直接亏。只在 `speculative_method="model"` 且 `enforce_eager=False` 时起作用。 |

对应的测试开关:`tests/gen.py --draft-sample-method`、`--no-draft-cudagraph`(`tests/gen.py:42-45`)。

已有但本阶段才真正生效的两项:`speculative_method="model"`、`speculative_model=<路径>`。
开启时会额外断言:两个模型的 `vocab_size` 相等且 `tokenizer.json`/`vocab.json`/`merges.txt`
逐字节相同(`model_runner.py:build_draft_model`);且 `tensor_parallel_size == ep_size == 1`
(`config.py:__post_init__`,原因见 08 报告"遗留")。

运行期的观测量:

| 名字 | 位置 | 作用 |
|---|---|---|
| `Sequence.draft_num_cached_tokens` | `nanovllm/engine/sequence.py:54` | 草稿模型已经算好 KV 的 token 数,语义与 `num_cached_tokens` 平行。稳态下两者相等,只有「上一轮草稿全被接受」时会少 1(第 k 个草稿 token 自己的 KV 这一轮没人算),下一轮草稿的第一次前向吃 2 个 token 补上。差值大于 2 时草稿第一步自动退回 eager,不会静默错位。 |
| `ModelRunner.exec_stats` 的 `draft_*` 三项 | `nanovllm/engine/model_runner.py:44-46` | `draft_graph_varlen` / `draft_graph_decode` / `draft_eager`,草稿那 k 次前向分别走了哪条路。`draft_eager` 里含每次 prefill 同步(那一次必然 eager)。 |

---

## Phase 2.5 Step B 新增

| 名字 | 位置 | 默认 | 作用 |
|---|---|---|---|
| `varlen_cudagraph` | `nanovllm/config.py:28` | `True` | 是否给**投机批**(无 prefill、批内 q 长度 1..1+k 参差)再录一族 varlen 形态的 CUDA graph。`False` 时投机批一律走 eager,即 Phase 2.5 Step A 的行为。只在 `num_speculative_tokens > 0` 且 `enforce_eager=False` 时才起作用 —— 投机关掉时 q 恒为 1,这族图一张都不录。用于 A/B 对照和出问题时一键回退。 |

对应的测试开关:`tests/gen.py --no-varlen-cudagraph`(`tests/gen.py:40`)。

运行期的观测量(不是配置,但配套):

| 名字 | 位置 | 作用 |
|---|---|---|
| `ModelRunner.exec_stats` | `nanovllm/engine/model_runner.py:43` | `{"graph_decode", "graph_varlen", "eager"}` 三个 step 计数器,用来算"走图 step 占比"。`tests/gen.py` 会把它并进输出 JSON 的 `stats`,前缀 `exec_`。 |

---

## 已有参数(便于对照,均在 `nanovllm/config.py`)

### 调度与容量

| 名字 | 默认 | 作用 |
|---|---|---|
| `max_num_batched_tokens` | 16384 | 一个 step 里所有 seq 加起来最多算多少 token。调小会强制 chunked prefill 与混批。 |
| `max_num_seqs` | 512 | 一个 step 最多排多少条 seq。也是 CUDA graph 分桶的上界。 |
| `max_model_len` | 4096 | 单条 seq 的最大长度,会被 `min()` 到模型的 `max_position_embeddings`。 |
| `kvcache_block_size` | 256 | 一个 KV block 装多少 token,必须是 256 的倍数。 |
| `num_kvcache_blocks` | -1 | KV block 总数。-1 = 按显存自动算;显式指定用于稳定复现抢占路径(05 报告坑 8)。 |
| `gpu_memory_utilization` | 0.9 | KV cache 能占显存的比例。 |

### 执行

| 名字 | 默认 | 作用 |
|---|---|---|
| `enforce_eager` | `False` | `True` 则一张 CUDA graph 都不录,全部 eager。注意它**不影响**用哪个 attention kernel —— 纯 decode 批在 eager 下仍走 `flash_attn_with_kvcache` 快路径(05 报告坑 5)。 |
| `model` | 必填 | 模型目录(HF 格式,本地路径)。 |

### 投机解码

| 名字 | 默认 | 作用 |
|---|---|---|
| `num_speculative_tokens` | 0 | 每步提议几个草稿 token。0 = 关闭投机。 |
| `speculative_method` | `"ngram"` | `"ngram"` = prompt-lookup,不需要第二个模型;`"model"` = 草稿模型(Phase 3B 已实现,见上)。 |
| `speculative_model` | `None` | `method="model"` 时的草稿模型路径。必须与目标模型同分词器同词表。 |
| `ngram_prompt_lookup_max` | 4 | n-gram 匹配时尝试的最长模式。 |
| `ngram_lookup_window` | 2048 | 只在最近这么多 token 里回溯,控制 CPU 开销。 |

### 并行

| 名字 | 默认 | 作用 |
|---|---|---|
| `tensor_parallel_size` | 1 | 单机 TP 卡数(1~8)。与 `ep_size` 最多开一个。 |
| `ep_size` | 1 | 专家并行的节点数,每机一进程一卡。>1 时强制要求 `enforce_eager=True`。 |
| `node_rank` | 0 | 本进程的全局 rank。 |
| `master_addr` / `master_port` | `localhost` / 2333 | 进程组的会合地址。EP 模式填直连口 IP。 |
| `ep_transport` | `"nccl"` | `"nccl"` \| `"nvshmem"`。 |
| `eos` | -1 | 由 `LLMEngine` 从 tokenizer 填入,不用手传。 |
