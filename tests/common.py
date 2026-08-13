"""测试共用的模型路径、prompt 集合与比对工具。"""
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.expanduser(os.environ.get("NANOVLLM_MODEL", "~/huggingface/Qwen3-0.6B/"))
BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines")


# 用真实自然语言而不是随机 token id 当 prompt:随机 token 会让模型输出接近均匀分布,
# bf16 的 logits 里立刻出现大量并列/近似并列的最大值,greedy 就变得对浮点噪声极度敏感,
# 既不代表真实负载,也会让 eager / cudagraph 的比对变成掷硬币。
CORPUS = """Large language model inference is dominated by two very different phases.
The first phase, called prefill, processes the entire prompt at once. Every token in the
prompt can attend to every earlier token, so the work is one large matrix multiplication
per layer and the graphics processor runs close to its peak arithmetic throughput. The
second phase, called decode, produces one token at a time. Each step reads the entire
key and value cache but performs only a single row of arithmetic, so the step is limited
by memory bandwidth rather than by arithmetic. A serving engine that treats these two
phases identically will leave a great deal of performance unused.

The key and value cache is the central data structure of a modern inference engine. For
every token that has already been processed, each attention layer stores one key vector
and one value vector per attention head. The cache grows linearly with sequence length
and with the number of concurrent requests, and it quickly becomes the factor that limits
how many requests can be served at the same time. Storing each sequence in one contiguous
buffer wastes an enormous amount of memory, because the engine must reserve room for the
longest output the request might produce, and most requests stop long before that.

Paged attention solves this problem by borrowing an idea from operating systems. The cache
is divided into fixed size blocks, and each sequence keeps a small table that maps its
logical block numbers onto physical block numbers. Blocks are allocated only when they are
actually needed, so a request that stops early never occupies memory it did not use.
Because the mapping is indirect, two sequences that begin with the same text can point at
the same physical blocks and share the work that was already done. This is called prefix
caching, and it is extremely effective for chat workloads where a long system prompt is
repeated on every single request.

Continuous batching is the scheduling counterpart of paged attention. Instead of waiting
for every request in a batch to finish, the scheduler rebuilds the batch before every
forward pass. A request that finishes leaves the batch immediately, and a request that has
been waiting joins as soon as there is room. The result is that the accelerator stays busy
even when the requests have wildly different output lengths, which is the normal situation
in production.

Chunked prefill goes one step further. A very long prompt is split into pieces, and each
piece is processed in a separate forward pass. This bounds the amount of work in any single
step, which in turn bounds how long a decode step can be delayed by an arriving prompt. When
prefill chunks and decode steps are mixed into the same batch, the engine no longer has to
choose between throughput and latency, because it gets both at once.

Speculative decoding attacks the memory bandwidth problem from a different direction. A
small and cheap draft model proposes several tokens ahead, and the large target model then
verifies all of the proposals in a single forward pass. Because the verification pass reads
the weights only once, several tokens can be produced for roughly the cost of one ordinary
decode step. A carefully designed acceptance rule guarantees that the tokens which survive
verification follow exactly the same probability distribution as tokens drawn from the
target model alone, so the speedup is free of any quality cost.

Sampling is the last stage of every step. The model produces a score for each word in the
vocabulary, and the sampler converts those scores into one concrete choice. Dividing the
scores by a temperature makes the distribution sharper or flatter. Keeping only the highest
scoring candidates, or only the smallest group of candidates whose probabilities add up to
a threshold, removes the long tail of unlikely words that would otherwise derail the text.
Setting the temperature to zero disables randomness entirely and simply takes the single
highest scoring word, which makes the output reproducible and is the basis of every
regression test in this project.

Correctness in an inference engine is subtle because almost every optimization changes the
order in which floating point numbers are added together. Two implementations can be
mathematically identical and still produce slightly different scores. The usual discipline
is to fix the random seed, disable temperature, and compare the generated tokens one by one
against a stored reference. When a difference appears, the engineer looks at the gap between
the top two scores at that position. A tiny gap means the difference is numerical noise,
while a large gap means a genuine defect has been introduced somewhere in the pipeline.
"""


@functools.lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)


@functools.lru_cache(maxsize=1)
def corpus_tokens() -> tuple:
    tok = get_tokenizer()
    ids = tok.encode(CORPUS)
    # 语料不够长就整篇重复拼接,保证能切出 600 / 256 / 257 长度的片段
    while len(ids) < 2200:
        ids = ids + tok.encode(CORPUS)
    return tuple(ids)


def build_prompts() -> list[list[int]]:
    """覆盖:短 prompt、长 prompt(跨多个 256 block)、共享 512 token 前缀(必中 prefix cache)、
    恰好落在 block 边界 和 边界 +1 的长度。返回 list[list[int]]。"""
    tok = get_tokenizer()
    c = list(corpus_tokens())

    short = [tok.encode("What is the capital of France? Answer in one word."),
             tok.encode("Write a short sentence about the sun.")]

    long_a = c[:600]                       # 跨 3 个 block
    long_b = c[:512] + c[900:988]          # 与 long_a 共享整 2 个 block 的前缀
    exact = c[1200:1456]                   # 恰好 256,踩 block 边界
    edge = c[1500:1757]                    # 256 + 1

    prompts = short + [long_a, long_b, exact, edge]
    assert [len(p) for p in prompts[2:]] == [600, 600, 256, 257], [len(p) for p in prompts]
    return prompts


def build_preempt_prompts(n: int = 6) -> list[list[int]]:
    """专门用来逼出抢占的 prompt 集:每条 255 token(恰好 1 个 block 装得下),
    互不共享前缀(避免 prefix cache 干扰块数计算)。

    生成第 1 个 token 后长度 256,仍是 1 块;第 2 个 token 让长度变成 257,
    此时每条都需要第 2 块。只要把 KV block 总数卡在 n+1,就必然有人被抢占。
    """
    c = list(corpus_tokens())
    out = []
    for i in range(n):
        s = 30 * i
        out.append(c[s:s + 255])
    assert all(len(p) == 255 for p in out)
    return out


def report(title: str, ok: bool, detail: str = ""):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {title}")
    if detail:
        print(detail)
    return ok
