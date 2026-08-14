# Phase 2 · 统一调度器(prefill/decode 混批)

目标:一次 step 里 decode seq 和 prefill chunk 同批前向,消除 prefill 阻塞 decode。

## 为什么几乎不用改 attention

attention.py:64-70 的 prefix-cache 分支已经是通用形式:
`flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, block_table=...)`。
decode 只是 `seqlen_q=1, seqlen_k=len(seq)` 的特例。

而且 prepare 侧的统一形式现成:对任意 seq,
`start = num_cached_tokens, end = start + num_scheduled_tokens`。
postprocess 后恒有 `num_cached_tokens == len(seq) - 1`(scheduler.py:90 先加 scheduled 再 append_token),
所以 decode 代入得 `start = len-1, end = len`,input 恰是 last_token、position 恰是 len-1、seqlen_k 恰是 len ——
**prepare_decode 本来就是 prepare_prefill 的退化情形**,可以直接删掉合并。

LMHead 同理:embed_head.py:59 的 `cu_seqlens_q[1:]-1` 对 q_len=1 的 decode 一样取到最后位置,
统一后 LMHead 只走这一条路,`context.is_prefill` 分支可删。

## 改动清单

### 1. `nanovllm/engine/scheduler.py` — schedule() 重写

策略:**decode 优先,剩余 token 预算给 prefill**(decode 每条只花 1 个预算,先保住在跑请求的延迟;prefill 用剩余预算做 chunked)。

```python
def schedule(self) -> list[Sequence]:          # 不再返回 is_prefill
    scheduled = []
    num_batched_tokens = 0

    # ---- 1. decode:running 里的 seq 逐条安排,每条 1 token ----
    #      保留现有的 can_append / preempt 逻辑(scheduler.py:62-76 原样搬)
    #      被抢占的 seq 回 waiting 队首,本轮不再参与
    num_decodes = len(scheduled)

    # ---- 2. prefill:waiting 严格 FIFO,用剩余预算 ----
    #      照搬 scheduler.py:30-56,差别只有两处:
    #      a) remaining = max_num_batched_tokens - num_batched_tokens(已含 decode 的份)
    #      b) len(scheduled) < max_num_seqs 的上限对两类一起数
    #      chunked 规则不变:只允许第一条 prefill 截断

    return scheduled     # 顺序:[decode..., prefill...]
```

排序约定:**decode 在前、prefill 在后**,写进注释。两个理由:
- Phase 2.5 判断"纯 decode 批"只看 `scheduled[num_decodes:]` 是否为空;
- 未来若做 decode 段 graph 化,decode 必须是连续前缀。

原 schedule() 里 "if scheduled_seqs: return ..., True"(scheduler.py:58-59)的短路逻辑删除——那就是 prefill 阻塞 decode 的根源。

注意一个新交互:**decode 先占了预算,可能出现 running 很多时 prefill 长期分不到预算**。可接受(vLLM 同款行为),饥饿由 max_num_seqs 上限自然缓解,不额外做公平性。

`postprocess` 去掉 `is_prefill` 参数,改成逐 seq 判断:

```python
def postprocess(self, seqs, token_ids, logprobs):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if seq.num_cached_tokens < seq.num_tokens:   # chunked prefill 没喂完,丢弃
            continue
        seq.append_token(token_id)
        ...
```

(原来的 `is_prefill and ...` 条件里,decode 时 `num_cached < num_tokens` 不可能成立,所以直接去掉 is_prefill 前缀是等价的。)

### 2. `nanovllm/engine/model_runner.py` — prepare 合一

`prepare_prefill` / `prepare_decode`(model_runner.py:129-188)合并成一个 `prepare_batch(seqs)`:

- 主循环就是现在 prepare_prefill 的循环体,对所有 seq 无差别执行(decode 自动退化为 q=1);
- slot_mapping 的 block 展开逻辑(model_runner.py:151-161)原样保留,decode 时 start=len-1、end=len,自然只落一个 slot,替代 prepare_decode:181 的单点公式;
- **block_tables 永远构建并传入**(不再以 `cu_seqlens_k > cu_seqlens_q` 为条件)——统一后 attention 一律从 paged cache 读 K/V;
- context_lens 不再需要(那是 flash_attn_with_kvcache 的参数,Phase 2.5 的 graph 快路径还要用,见下);
- warmup 时 block_table 为空的 continue 分支(model_runner.py:149)保留。

`run()` 简化:

```python
def run(self, seqs, ...):
    input_ids, positions = self.prepare_batch(seqs)
    ...
```

`is_prefill` 参数从 `run` / `run_model` 的对外语义里退役,但 **graph 快路径需要知道"本批是否纯 decode"**,由 runner 自己判断:

```python
is_pure_decode = all(not seq.is_prefill for seq in seqs)
# 等价判断:len(input_ids) == len(seqs)
```

`run_model`(model_runner.py:196-212)的分支条件改为:
`纯 decode 且 not enforce_eager 且 bs<=512` → 走 graph(此时仍按老方式 set_context(False, ...) 构造 with_kvcache 需要的 context_lens 等,见 Phase 2.5);否则 eager 走统一 varlen。

即 Phase 2 阶段 **prepare_decode 并不物理删除**,降级为"纯 decode 批的 graph 快路径专用",混批/prefill 走 prepare_batch。这样 36 张图暂不作废,Phase 2.5 再处理。

### 3. `nanovllm/layers/attention.py`

```python
def forward(self, q, k, v):
    context = get_context()
    if k_cache.numel():
        store_kvcache(...)                      # 不变
    if context.is_prefill:                      # 语义改为"varlen 统一路径"
        if k_cache.numel():                     # 正常运行:一律读 paged cache
            o = flash_attn_varlen_func(q, k_cache, v_cache, ..., block_table=context.block_tables)
        else:                                   # warmup:cache 未分配,用本轮 k/v
            o = flash_attn_varlen_func(q, k, v, ..., block_table=None)
    else:                                       # 纯 decode 的 graph 快路径,原样保留
        o = flash_attn_with_kvcache(...)
```

行为差异说明:原来"无 prefix cache 命中的纯 prefill"直接用本轮算出的连续 k/v(attention.py:67),统一后改成从 paged cache 经 block_table 读。数值等价(KV 刚被 store_kvcache 写进去),性能上多一层页表间接寻址,prefill 吞吐可能掉几个百分点——基线跑分时记录一下,可接受。

`context.is_prefill` 建议改名 `use_varlen`(改名放在 Phase 2.5 一起做,避免本阶段 diff 混入无关重命名;本阶段只在 context.py 加注释说明语义漂移)。

### 4. `nanovllm/engine/sequence.py` — TP pickle 的坑

`__getstate__`(sequence.py:82-84)现在 decode 只传 `last_token`、token_ids 传空。
混批后 worker 侧 prepare_batch 要取 `seq[start:end]`,decode seq 会 IndexError。

改法:last_state 一律传**本轮被调度的切片**:

```python
def __getstate__(self):
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table,
            self.token_ids[self.num_cached_tokens : self.num_cached_tokens + self.num_scheduled_tokens])

def __setstate__(self, state):
    *..., sched_slice = state
    # token_ids 只在 [num_cached_tokens, +scheduled) 区间有效,worker 只会读这个区间
    self.token_ids = [0] * self.num_cached_tokens + sched_slice   # 或改 prepare 侧不用绝对下标
```

更干净的替代(推荐):prepare_batch 里不用 `seq[start:end]` 绝对切片,给 Sequence 加一个
`scheduled_token_ids` 属性,driver 侧返回真切片、worker 侧返回 `__setstate__` 存下的切片。
避免用 `[0]*n` 填充这种容易埋雷的写法。
(worker 侧 hash_blocks 不存在——block_manager 只活在 rank 0 driver,所以 worker 不需要完整 token_ids,确认过 model_runner 里 worker 只跑 run。)

顺带:`is_prefill` 需要进 pickle 元组吗?worker 的 prepare_batch 不再按它分支(统一循环),
graph 快路径判断用 `num_scheduled_tokens==1 && num_cached_tokens+1==num_tokens` 可推导,
但为了可读性直接把 `is_prefill` 加进 state 元组,一个 bool 成本为零。

### 5. `nanovllm/engine/llm_engine.py`

`step()`(llm_engine.py:49-55):

```python
seqs = self.scheduler.schedule()
result = self.model_runner.call("run", seqs)
self.scheduler.postprocess(seqs, *result)
num_prefill_tokens = sum(s.num_scheduled_tokens for s in seqs if s.is_prefill)   # 注意要在 postprocess 前取
num_decode_tokens  = sum(1 for s in seqs if not s.is_prefill)
```

`generate()` 的吞吐显示(llm_engine.py:81-93)相应改成两个数同轮都可能非零;
`num_tokens` 正负号的旧协议废弃。

## 验证

1. **强制单类批**:先构造只有 prefill / 只有 decode 的负载,统一路径的输出对齐 Phase 1 基线(容忍 kernel 路径改变带来的浮点漂移,按总览的分歧分析流程处理),重存 `greedy_p2.json`。
2. **混批正确性**:长 prompt(持续 chunked prefill)+ 一批已在 decode 的短请求同时进,结束后每条输出与"单独串行跑"一致。
3. **抢占路径**:调小 gpu_memory_utilization 逼出 preempt,确认被抢占 seq 恢复后输出不变(prefix cache 命中路径 + 混批)。
4. **TP=2** 跑一遍(pickle 改动的回归)。
5. 性能:记录混批前后 decode P50 延迟(prefill 阻塞消除应有明显改善)和纯 prefill 吞吐(页表间接寻址的代价)。

## 风险

- flash_attn_varlen_func + block_table 要求 flash-attn >= 2.5,现有 prefix-cache 路径已在用,无新增依赖风险。
- 混批下 `max_seqlen_q` 由最长 prefill chunk 决定,decode 行的 q=1 padding 在 varlen 里天然无 padding(cu_seqlens 精确),无浪费。
- 主要风险集中在 CUDA graph,见 Phase 2.5。
