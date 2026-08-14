# nano-vllm 增强计划总览

三个阶段,严格按依赖顺序推进。每个阶段都以 **greedy 回归基线** 作为等价性验收标准。

| 阶段 | 内容 | 依赖 | 风险 |
|---|---|---|---|
| Phase 1 | Sampler 重写:top_k / top_p / greedy / logprobs + 建立回归基线 | 无 | 低,完全独立 |
| Phase 2 | 统一调度器:prefill/decode 混批 | Phase 1 的基线 | 中 |
| Phase 2.5 | CUDA graph 适配混批 | Phase 2 | 高,单列 |
| Phase 3 | 投机解码 | Phase 2(混批让 q_len=k+1 的验证前向"免费") | 高,难在正确性 |

## 分阶段文档

- [01-phase1-sampler.md](01-phase1-sampler.md)
- [02-phase2-unified-scheduler.md](02-phase2-unified-scheduler.md)
- [03-phase2.5-cudagraph.md](03-phase2.5-cudagraph.md)
- [04-phase3-speculative-decoding.md](04-phase3-speculative-decoding.md)
- **[05-implementation-report.md](05-implementation-report.md) — 实现总结:改了什么 / 踩了哪些坑 / 提升了多少(已完成,4 套件 42 项测试全通过)**

## 全局验证策略:greedy 回归基线

Phase 1 一开始就建立(甚至可以在改 sampler 之前先建),之后每个阶段收尾都跑一遍:

1. 固定一组 prompt(短/长/跨 block 边界 256 的、有公共前缀能触发 prefix cache 的),`temperature=0, max_tokens=128`。
2. 跑 `enforce_eager=True` 和 `False` 两种配置,把每条的 token_ids 存成 JSON(`tests/baselines/greedy_v<phase>.json`)。
3. 后续阶段用比对脚本逐 token 比对;发现分歧时输出**第一个分歧位置**和该位置 top-5 logprobs,便于定位是数值漂移还是逻辑 bug。

**关于"逐 token 一致"的现实预期**:

- Phase 1(只改 sampler,greedy=argmax):必须逐 token 一致,不一致就是 bug。
- Phase 2(换 attention kernel 路径、批次组成变化):kernel 归约顺序不同,logits 有 1e-3 量级浮点漂移是正常的,argmax 绝大多数位置不变;若个别位置分歧,需人工确认该位置 top-2 logit 差值是否本来就在噪声范围内(比对脚本要自动打印这个信息)。不能无脑判 fail,但也不能放过真 bug。
- Phase 3(greedy 下开关投机):这是最硬的验收——同样 kernel 路径下,开/关投机的输出**必须逐 token 一致**,因为 greedy 验证的接受规则就是 argmax 相等,数学上是恒等变换。

## 关键代码位置速查(现状)

| 文件 | 关键点 |
|---|---|
| `nanovllm/sampling_params.py` | 只有 temperature/max_tokens/ignore_eos,且 `assert temperature > 1e-10` 禁 greedy |
| `nanovllm/layers/sampler.py:11` | Gumbel-max 直接出 token,probs 算完即丢 |
| `nanovllm/layers/attention.py:64-70` | prefill 的 prefix-cache 分支已是通用变长形式(varlen + block_table) |
| `nanovllm/layers/embed_head.py:59` | LMHead 用 `cu_seqlens_q[1:]-1` 取每条 seq 最后位置 |
| `nanovllm/engine/scheduler.py:25-79` | schedule() 返回 (seqs, is_prefill),prefill/decode 互斥 |
| `nanovllm/engine/model_runner.py:129/172` | prepare_prefill / prepare_decode 两套 |
| `nanovllm/engine/model_runner.py:223-265` | 按 bs 分桶的 36 张 CUDA graph,烧死了 flash_attn_with_kvcache |
| `nanovllm/engine/block_manager.py:124-135` | hash_blocks 用 num_cached_tokens + num_scheduled_tokens 定登记范围 |
| `nanovllm/engine/sequence.py:82-93` | __getstate__ 在 decode 时只传 last_token(TP 混批的坑) |
