# Phase 3 · 投机解码

前提:Phase 2 完成。此时"q_len=k+1 的验证前向"就是统一 varlen 路径里一个普通的小 chunk,
调度和 attention 侧几乎零改动——难点全部在正确性。

## 总体设计

- 草稿来源:**独立小模型**(同 tokenizer,如 Qwen3-0.6B 给大模型打草稿)。
  建议先用 **n-gram prompt-lookup** 作为第 0 个草稿源(无需第二个模型,从 seq 自身找 n-gram 匹配提议 token)
  把验证/回滚/接受机制先调通,再接草稿模型——机制代码完全复用,只换"proposals 从哪来"。
- 每 step 每 seq 提议 k 个草稿 token,目标模型一次前向验证,接受 a 个(0≤a≤k),
  外加 1 个修正/奖励 token,单 step 净产出 a+1 个 token。
- 只对 decode 阶段的 seq 投机;prefill chunk 照旧。

## 配置与开关

`config.py`:

```python
speculative_model: str | None = None    # 草稿模型路径;None 且 ngram 关闭 = 不投机
speculative_method: str = "model"       # "model" | "ngram"
num_speculative_tokens: int = 4         # k
```

## 显存与 KV cache:草稿模型共用 block_table

关键决定:**草稿模型不建第二套 BlockManager**。两个模型看到完全相同的 token 序列,
同一个 `seq.block_table` / 同一套 slot_mapping 可以同时索引两套物理 cache:

- `allocate_kv_cache`(model_runner.py:103)改为:
  `block_bytes_total = block_bytes_target + block_bytes_draft`,
  `num_kvcache_blocks = 可用显存 // block_bytes_total`;
  分配两个大 tensor `self.kv_cache`(目标)和 `self.draft_kv_cache`,block 数相同、编号一一对应;
  草稿模型的 attention 层绑定到 draft cache(同样的 modules() 遍历)。
- prefix cache 一致性:hash 命中复用 block i 时,draft cache 的 block i 里存的就是同一段前缀的
  草稿模型 KV——**前提是草稿模型对每个写入目标 cache 的 chunk 都同步做了 prefill**(见下),
  这个不变量必须在代码注释里写明并在 debug 模式下可断言。

## 执行流程(model_runner 新增 `run_speculative`)

设 seq 当前长度 L(last_token 是第 L-1 个,其 KV 未算,`num_cached_tokens == L-1`)。

### 0. prefill 同步

统一批里的 prefill chunk,目标模型跑完后,**草稿模型对同一 chunk、同一 context(slot_mapping/block_tables 不变,换 cache)再跑一遍**,只为写 draft KV,logits 丢弃。
代价 ≈ 草稿模型大小占比,可接受;这是维持"两套 cache 同步"不变量的最简单方式。

### 1. 提前扩容 block

投机会临时写到位置 L+k-1,`may_append` 的"每次最多加 1 块"不够用。
BlockManager 新增:

```python
def ensure_capacity(self, seq, num_new_tokens) -> bool:
    need = ceil((len(seq) + num_new_tokens) / block_size) - len(seq.block_table)
    # 空闲不足时沿用 preempt 逻辑;成功则 append need 块
```

scheduler 的 decode 段:`can_append` 换成按 k+1 容量检查,每条投机 seq 的 token 预算按 **k+1** 计。

### 2. 草稿阶段:k 次小模型 decode

```python
draft_tokens[B, k], draft_probs[B, k, V]
输入 last_token,循环 k 次:
  logits_d = draft_model(...)        # q_len=1 的统一 varlen decode,写 draft cache
  用与目标相同的 (temperature, top_k, top_p) 处理 → q_i 分布
  d_i ~ q_i(greedy 时 argmax),记录 q_i(d_i) 与整行 q_i
```

草稿 KV 写入位置 L-1 .. L+k-2(slot 由 position 决定,block 已在步骤 1 保证存在)。
draft_probs 只需保留 `q_i` 整行(rejection 的修正分布要用 p−q),显存 [B,k,V]·fp32,
B=256、k=4、V=152k ≈ 2.4GB——太大,**只存 top 候选不行(修正分布需要全行)**,
改为:逐位置流式处理,验证时按位置 i 对齐消费,峰值只留 [B, V] 两三份(实现注意点,见"验证阶段")。
更简单的折中:k 很小(≤8),先按 [B,k,V] fp16 存,B 大时分 micro-batch;首版用简单方案 + 显存注释。

### 3. 验证阶段:目标模型一次前向

每条投机 seq 构造 chunk:`input = [last_token, d_1..d_k]`,q_len=k+1,positions L-1..L+k-1。
在统一 varlen 里这就是一个普通 chunk,目标 KV 写入位置 L-1..L+k-1(其中被拒绝的尾部是垃圾,后述)。

**LMHead 改动**(本阶段唯一的模型侧改动):现在 embed_head.py:59 只取每 seq 最后位置,
验证需要全部 k+1 个位置的 logits。给 context 加 `logits_indices: Tensor | None`:

```python
# prepare 时算好:普通 seq → 该 seq q 段最后一个下标;投机验证 seq → q 段全部 k+1 个下标
x = x[context.logits_indices].contiguous()
```

顺带收益:chunked prefill 中间块可以不进 logits_indices,省掉一直在浪费的 lm_head 计算与采样
(Phase 1 里"照算再丢弃"的 TODO 在这里自然解决)。
sampler 侧需要知道 logits 的行→seq 映射,prepare 返回一个 per-seq 的 (offset, count)。

### 4. 接受判定(rejection sampling)

对每条 seq、位置 i = 1..k(p_i = 目标分布,经同样的 temperature/top_k/top_p 处理并归一化;q_i = 草稿分布):

```python
r ~ U(0,1)
if r < p_i(d_i) / q_i(d_i):    接受 d_i,继续
else:                           拒绝;从 norm(clamp(p_i - q_i, min=0)) 采修正 token,停止
全部接受:                       从 p_{k+1} 采奖励 token
```

- 该算法保证输出分布与不投机时**逐分布相同**(Leviathan 2023),前提是 p、q 都是"实际用于采样的分布",
  所以 top_k/top_p 截断必须在算 p_i、q_i 之前做完——sampler 需要导出一个
  `compute_processed_probs(logits, params) -> probs` 的复用函数(Phase 1 的 apply_top_k_top_p 直接复用)。
- **greedy 特化**:temperature=0 时退化为 `argmax(p_i) == d_i` 则接受、否则取 argmax(p_i) 为修正 token。
  这条路径与不投机的 greedy **数学恒等**,是硬验收的基础。实现里 greedy 不走概率比较,
  直接 token 比较,避免浮点边界问题。
- batch 内各 seq 接受数不同:用张量算出每 seq 的 `accept_len`(第一个拒绝位置),
  修正/奖励 token 用 gather 批量采,不逐 seq 进 python 循环。

### 5. 回滚与簿记(正确性核心)

接受 a 个草稿 + 1 个修正/奖励 token:

```python
# 顺序严格如下:
seq.append_token(d_1) ... seq.append_token(d_a)      # ① 先把接受的草稿落到 token_ids
seq.append_token(t_corr)                              # ② 修正/奖励 token
seq.num_scheduled_tokens = a + 1                      # ③ 本轮"KV 已算好"的 token 数:
                                                      #    last_token + d_1..d_a,共 a+1 个
scheduler.postprocess → hash_blocks → num_cached_tokens += a+1
# 结果:num_cached_tokens = L-1 + a+1 = L+a = 新len-1,不变量恢复 ✔
```

三个必须处理的污染/泄漏点:

1. **hash_blocks 范围(block_manager.py:124-135)**:登记范围由
   `num_cached_tokens + num_scheduled_tokens` 决定,所以 ③ 必须写**实际接受数 a+1**,
   而不是投机长度 k+1。写成 k+1 的话,若 L+k 跨过 block 边界而 L+a 没跨,
   会把一个**含垃圾 KV(被拒绝位置)的 block 登记进 prefix cache 索引**,
   后续命中该前缀的请求直接读到错误 KV——这就是 prefix cache 污染,必须加一条
   针对性测试(构造 L 使拒绝恰好发生在 block 边界前后)。
2. **多余 block 回收**:步骤 1 里 ensure_capacity 可能为位置 L+a+1..L+k-1 多开了 block。
   接受数确定后,`len(seq.block_table)` 必须收缩回 `seq.num_blocks`
   (否则 slot 计算用 `block_table[-1]`,model_runner 老公式和 last_block_num_tokens 全错)。
   BlockManager 加 `truncate(seq)`:pop 尾部多余 block_id,ref_count-- 并按 deallocate 的规则归还。
3. **被拒绝位置的残留 KV**(两套 cache 都有,位置 L+a..L+k-1):不用主动擦除。
   slot 由 position 唯一决定,这些位置将来被真实 token 重新计算时**原地覆盖**;
   attention 读取受 cu_seqlens_k(= 有效长度)限制,读不到残留区。
   唯一能把残留暴露出去的通道就是上面第 1 点的 hash 登记,堵住即可。
   ——这段推理要写成 block_manager 里的注释。

抢占交互:被抢占的投机 seq 走 preempt → deallocate → 全量重 prefill,天然安全,无需特判。

## 各文件改动汇总

| 文件 | 改动 |
|---|---|
| config.py | 3 个投机配置项 |
| model_runner.py | 加载草稿模型;draft kv_cache 分配;run_speculative(草稿循环 + 验证前向 + 接受判定);prefill 的草稿同步 |
| sampler.py | 抽出 compute_processed_probs 复用;新增 rejection 采样工具函数(批量 accept_len、修正分布采样) |
| embed_head.py | logits_indices 机制 |
| context.py | 加 logits_indices 字段 |
| scheduler.py | decode 段按 k+1 记预算;can_append→按容量检查;postprocess 接收 per-seq 多 token(接口从 `token_ids: list[int]` 变为 `list[list[int]]`,非投机时每项长度 1) |
| block_manager.py | ensure_capacity / truncate;hash_blocks 不改代码但依赖 num_scheduled_tokens 语义,加注释与断言 `num_scheduled_tokens <= 实际已算 KV 数` |
| sequence.py | 无结构性改动;append 多 token 的循环在 postprocess 里做 |
| llm_engine.py | step 透传;统计接受率(accepted / proposed)挂到 tqdm postfix |

CUDA graph:草稿模型的 q_len=1 decode 循环是主要开销来源,可给**草稿模型单独录图**(形态与 Phase 2.5 Step B 相同);目标模型验证前向 q_len=k+1 固定,也可按 (bs, k+1) 录图。都属于后续优化,首版全 eager。

## 验证(按顺序)

1. **单元:rejection sampler**。随机构造 p、q,蒙特卡洛验证输出分布收敛到 p(KL < 阈值);
   greedy 特化下与 argmax 逐 token 相等。
2. **硬验收:greedy 开关投机逐 token 一致**。
   同一批 prompt、temperature=0:`speculative_model=None` 跑一遍存输出,
   开投机(先 ngram 后草稿模型)再跑,`greedy_p3.json` 必须与关投机版本**完全一致**。
   任何一个 token 不同都是 bug(数学上恒等;若出现分歧,优先排查:验证前向的 KV 写入范围、
   accept_len 边界、修正 token 取错行)。
3. **污染专项**:构造接受/拒绝恰好落在 256 边界两侧的用例 + 后续同前缀请求,
   验证 prefix cache 命中后的输出仍与串行一致。
4. **抢占 + 投机**并发压力测试(小显存配置),跑完全部输出与串行一致。
5. **收益度量**:接受率、每 step 净 token 数、端到端 tokens/s vs 不投机——k 取 2/4/8 扫一遍。
