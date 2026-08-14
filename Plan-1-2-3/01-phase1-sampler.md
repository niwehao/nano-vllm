# Phase 1 · Sampler 重写

产出四样:**greedy、top_k、top_p、logprobs**,外加 greedy 回归基线脚本。

## 现状

- `sampling_params.py`:只有 temperature/max_tokens/ignore_eos,`__post_init__` 里 `assert temperature > 1e-10` 明确禁止 greedy。
- `sampler.py:8-12`:`logits / T → softmax → Gumbel-max argmax`,probs 算完就丢,没有任何截断,也不输出 logprob。
- `model_runner.py:190-193 prepare_sample`:只搬 temperatures 一个 tensor 上 GPU。
- `model_runner.py:219`:`token_ids = self.sampler(logits, temperatures).tolist()`,返回值就是 `list[int]`,一路传到 `scheduler.postprocess`。

## 改动清单

### 1. `nanovllm/sampling_params.py`

```python
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0      # 0.0 = greedy
    top_p: float = 1.0            # 1.0 = 关闭
    top_k: int = -1               # -1 = 关闭
    max_tokens: int = 64
    ignore_eos: bool = False
    logprobs: int | None = None   # None = 不要;N = 采样 token 的 logprob + top-N 候选

    def __post_init__(self):
        assert self.temperature >= 0.0          # 删掉禁 greedy 的 assert
        assert 0.0 < self.top_p <= 1.0
        assert self.top_k == -1 or self.top_k >= 1
        assert self.logprobs is None or 0 <= self.logprobs <= 20
```

语义约定(和 vLLM 对齐):
- `temperature == 0` → greedy,此时 top_p/top_k 全部忽略(argmax 与截断无关)。
- 截断顺序:先 top_k 后 top_p(vLLM 的顺序)。

### 2. `nanovllm/engine/sequence.py`

`__init__` 里从 sampling_params 多存三个标量,并加 logprobs 的累积容器:

```python
self.temperature = sampling_params.temperature
self.top_p = sampling_params.top_p
self.top_k = sampling_params.top_k
self.num_logprobs = sampling_params.logprobs   # None 或 int
self.completion_logprobs: list = []   # 每生成一个 token 追加一项
```

每项的结构建议:`{"token_id": int, "logprob": float, "top": [(token_id, logprob), ...]}`。
只在 rank 0 的 driver 进程里用,不进 `__getstate__`(worker 不需要)。

### 3. `nanovllm/layers/sampler.py` — 核心重写

新的 forward 签名:

```python
def forward(self, logits, temperatures, top_ps, top_ks, max_logprobs: int):
    # logits: [B, V] float;返回 (tokens [B], logprob 载荷或 None)
```

内部流程(全 batch 张量化,不逐行 python 循环):

```python
logits = logits.float()
greedy_tokens = logits.argmax(dim=-1)                       # ① greedy 结果先算好

safe_t = torch.where(temperatures == 0, 1.0, temperatures)  # ② 防除零
logits = logits / safe_t.unsqueeze(1)

logprobs = torch.log_softmax(logits, dim=-1)                # ③ logprobs 在截断"前"算
                                                            #    (温度缩放后、top-k/p 前,和 vLLM 一致;
                                                            #     greedy 行因 safe_t=1 等价于原始 logits)

logits = apply_top_k_top_p(logits, top_ks, top_ps)          # ④ 截断
probs = torch.softmax(logits, dim=-1)

eps = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
sampled = probs.div_(eps).argmax(dim=-1)                    # ⑤ 沿用 Gumbel-max

tokens = torch.where(temperatures == 0, greedy_tokens, sampled)  # ⑥ 按行合并

if max_logprobs >= 0:                                       # ⑦ 有人要 logprobs 才做
    token_lp = logprobs.gather(1, tokens.unsqueeze(1))      # 采样出的 token 的 logprob
    top_lp, top_ids = logprobs.topk(max(max_logprobs, 1), dim=-1)
    payload = (token_lp, top_ids, top_lp)                   # 都留在 GPU,run() 里一次 .tolist()
else:
    payload = None
return tokens, payload
```

`apply_top_k_top_p` 用 vLLM 的升序排序实现(一次 sort 同时服务 k 和 p):

```python
def apply_top_k_top_p(logits, top_ks, top_ps):
    # top_ks 里 -1 先替换成 V(等于不截断)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    # top-k:升序下,保留最后 k 个 → 阈值 = 第 (V-k) 个位置的值
    boundary = logits_sort.gather(1, (V - top_ks).unsqueeze(1))
    logits_sort.masked_fill_(logits_sort < boundary, -inf)
    # top-p:升序 cumsum,累积概率 <= 1-p 的尾巴砍掉;最后一列永不砍(至少留 1 个)
    probs_sort = logits_sort.softmax(-1)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum <= (1 - top_ps.unsqueeze(1))
    mask[:, -1] = False
    logits_sort.masked_fill_(mask, -inf)
    # 散射回原始词序
    return torch.empty_like(logits_sort).scatter_(1, logits_idx, logits_sort)
```

关于 `@torch.compile`:重写后引入了 sort/topk/条件分支,先**摘掉装饰器**保证正确,基线过了之后再试着加回来测(可能因 `max_logprobs` 分支产生 recompile;可以拆成 compile 的采样主体 + eager 的 logprobs 尾巴)。

快速路径:`top_ks==V 且 top_ps==1` 全批成立时跳过 sort(一个 `.all()` 判断,代价是一次同步,可先不做,留 TODO)。

### 4. `nanovllm/engine/model_runner.py`

`prepare_sample`(model_runner.py:190)扩成:

```python
def prepare_sample(self, seqs):
    temperatures = [seq.temperature for seq in seqs]
    top_ps = [seq.top_p for seq in seqs]
    top_ks = [seq.top_k if seq.top_k != -1 else vocab_size for seq in seqs]
    max_logprobs = max((seq.num_logprobs or -1 for seq in seqs), default=-1)
    # 三个 list → pin_memory tensor → cuda(non_blocking),照抄现有 temperatures 的写法
    return temperatures, top_ps, top_ks, max_logprobs
```

`run()`(model_runner.py:214-221)的采样行改为:

```python
tokens, lp_payload = self.sampler(logits, *sample_args) if self.rank == 0 else (None, None)
return (tokens.tolist(), unpack(lp_payload)) if self.rank == 0 else None
```

`unpack` 把 GPU 张量转成 per-seq 的 python 结构(一次性 `.tolist()`,避免逐 seq 同步)。
注意:TP 时只有 rank 0 采样、rank 0 又是 driver 进程,返回值不过 shared memory,**结构变了不影响 worker**。worker 侧 `run` 返回 None 的分支保持不变。

`warmup_model`(model_runner.py:100)里的 `self.run(seqs, True)` 不受影响(warmup seq 的默认 SamplingParams 走 temperature=1)。但默认 SamplingParams 现在允许构造,不用改。

### 5. `nanovllm/engine/scheduler.py` — postprocess

签名从 `postprocess(seqs, token_ids, is_prefill)` 扩为接收 `(token_ids, logprobs)`。
在 scheduler.py:92 的 chunked-prefill 未完成分支(`continue` 那里),**logprob 和 token 一起丢弃**;
在 `seq.append_token(token_id)` 旁边同步 `seq.completion_logprobs.append(lp)`(仅当 `seq.num_logprobs is not None`,且按 seq 自己的 N 截取 top 列表——batch 里传的是全批最大 N)。

### 6. `nanovllm/engine/llm_engine.py`

- `step()`(llm_engine.py:52):`token_ids` 变成 `(token_ids, logprobs)`,透传给 postprocess。
- `generate()`(llm_engine.py:102):输出 dict 增加 `"logprobs": seq.completion_logprobs`(用户没要就是空 list / None)。
- 完成的 seq 现在在 step 里只回传 `(seq_id, completion_token_ids)`,补上 logprobs;或者干脆回传 seq 对象,generate 里取字段——**选后者**,改动最小且 Phase 3 还要用。

## 回归基线(本阶段就建)

新增 `tests/greedy_baseline.py`:

1. 固定 8~10 条 prompt:短(<10 token)、长(>1000 token,跨多个 256 block)、两条共享长前缀(触发 prefix cache)、一条正好 256 整数倍长度(块边界)。
2. `SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)`(ignore_eos 保证长度确定)。
3. `--save tests/baselines/greedy_p1.json` / `--check <file>` 两种模式;check 模式逐 token 比对,分歧时打印位置 i、两边 token、该位置 top-5 logprob 及 top-2 差值。
4. eager 和 CUDA graph 两种配置各存一份。

**顺序建议:先在改动前用「temperature=1e-9 近似 greedy」跑一版留档参考,改完 sampler 后用真 greedy 存正式基线**(近似版只用于自查改动前后行为没有意外漂移,不作为正式基线)。

另加 `tests/test_sampler.py` 纯单元测试(不用 GPU 模型,随机 logits):
- greedy 行 == argmax;
- top_k=k 时采样结果的支撑集 ⊆ 每行 top-k 集合(采样 1000 次验证);
- top_p 同理验证支撑集;
- logprobs 数值 == 手算 log_softmax 的对应项;
- temperature=0 与 top_k/top_p 任意组合不崩。

## 边界与坑

- **温度除零**:老代码 assert 挡住了,新代码靠 `safe_t` + `torch.where`,别漏。
- **chunked prefill 中间块**:sampler 照算(浪费但结构简单,维持现状),token 和 logprob 都在 postprocess 丢弃。
- **`prepare_sample` 只在 rank 0 调**(model_runner.py:217 已如此),扩展后维持。
- vocab_size 从 `config.hf_config.vocab_size` 拿,注意 logits 是 gather 后的完整词表宽度(embed_head.py:65),没有 TP 切分问题。
