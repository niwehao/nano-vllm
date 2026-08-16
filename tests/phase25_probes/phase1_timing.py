"""干净的 kernel 计时:回答 Q2(烧死 max_seqlen_k 的性能代价)和
Q4d(烧死 max_seqlen_q 的性能代价)。

前两版计时不可信(出现 0.489× 这种不可能的比值),原因是每次测量都新建/销毁
CUDA graph,显存池和时钟状态在变。这版:
  - 所有图**一次性**建好,先全部 warmup
  - A/B 交替测多轮,取**最小值**(最小值最不受降频/抖动污染)
  - 每个图内部连录 REP 次同一 kernel,摊掉 launch 开销
"""
import math
import statistics

import torch
from flash_attn import flash_attn_varlen_func

NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 8, 128
BLOCK_SIZE, MAX_MODEL_LEN, NUM_BLOCKS = 256, 4096, 512
MAX_NUM_BLOCKS = MAX_MODEL_LEN // BLOCK_SIZE
DTYPE, DEV = torch.bfloat16, "cuda"
SCALE = HEAD_DIM ** -0.5
REP = 100
ROUNDS = 12

torch.manual_seed(0)
k_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
v_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)


def build(qlens, klens, seed):
    n = len(qlens)
    T = sum(qlens)
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(T, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV, generator=g)
    cu_q, cu_k = [0], [0]
    for i in range(n):
        cu_q.append(cu_q[-1] + qlens[i])
        cu_k.append(cu_k[-1] + klens[i])
    cu_q = torch.tensor(cu_q, dtype=torch.int32, device=DEV)
    cu_k = torch.tensor(cu_k, dtype=torch.int32, device=DEV)
    bt = torch.full((n, MAX_NUM_BLOCKS), -1, dtype=torch.int32, device=DEV)
    rng = torch.Generator().manual_seed(seed + 999)
    for i in range(n):
        nb = (klens[i] + BLOCK_SIZE - 1) // BLOCK_SIZE
        if nb:
            bt[i, :nb] = torch.randperm(NUM_BLOCKS, generator=rng)[:nb].to(torch.int32).to(DEV)
    return q, cu_q, cu_k, bt


class Bench:
    """一个待测配置 = 一张内含 REP 次同 kernel 的图。"""

    def __init__(self, q, cu_q, cu_k, bt, max_q, max_k):
        self.keep = []
        for _ in range(3):
            flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                   max_seqlen_q=max_q, max_seqlen_k=max_k,
                                   softmax_scale=SCALE, causal=True, block_table=bt)
        torch.cuda.synchronize()
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            for _ in range(REP):
                self.keep.append(flash_attn_varlen_func(
                    q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=max_q, max_seqlen_k=max_k,
                    softmax_scale=SCALE, causal=True, block_table=bt))
        torch.cuda.synchronize()

    def time_once(self):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        self.g.replay()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / REP * 1000      # us / 次


def ab(name, a: Bench, b: Bench, label_a, label_b):
    for _ in range(3):
        a.time_once(); b.time_once()
    ta, tb = [], []
    for _ in range(ROUNDS):
        ta.append(a.time_once())
        tb.append(b.time_once())
    ma, mb = min(ta), min(tb)
    print(f"  {name:<30} {label_a}={ma:7.3f}us  {label_b}={mb:7.3f}us  "
          f"比值={mb/ma:6.3f}×   (中位 {statistics.median(ta):.3f}/{statistics.median(tb):.3f})")
    return ma, mb, mb / ma


print(f"{torch.cuda.get_device_name(0)}  REP={REP} ROUNDS={ROUNDS}  取最小值")
print()
print("=" * 96)
print("Q2 · 烧死 max_seqlen_k = 4096 的性能代价(batch=8,全部 q_len=1)")
print("=" * 96)
q2res = {}
for m in [32, 128, 512, 1024, 2048, 4096]:
    lens = [max(1, m - 7 * i) for i in range(8)]
    lens[0] = m
    q, cq, ck, bt = build([1] * 8, lens, seed=100 + m)
    A = Bench(q, cq, ck, bt, 1, m)                 # 真实 max_seqlen_k
    B = Bench(q, cq, ck, bt, 1, MAX_MODEL_LEN)     # 烧死 4096
    q2res[m] = ab(f"real_max_k={m}", A, B, "real", "big ")
    del A, B

print()
print("=" * 96)
print("Q4d · 烧死 max_seqlen_q = 3 但实际 q_len 全为 1 的性能代价")
print("=" * 96)
for ctx in [256, 1000, 2048, 4096]:
    q, cq, ck, bt = build([1] * 8, [ctx] * 8, seed=7)
    A = Bench(q, cq, ck, bt, 1, MAX_MODEL_LEN)     # max_seqlen_q = 1
    B = Bench(q, cq, ck, bt, 3, MAX_MODEL_LEN)     # max_seqlen_q = 3(烧死)
    ab(f"ctx={ctx} 均匀", A, B, "maxq=1", "maxq=3")
    del A, B

print()
print("=" * 96)
print("投机的真实代价:T=8(全未命中)vs T=24(全命中)—— 同一张图必须按最坏 T 分桶")
print("=" * 96)
for ctx in [256, 1000, 2048]:
    q1, cq1, ck1, bt1 = build([1] * 8, [ctx] * 8, seed=7)
    q3, cq3, ck3, bt3 = build([3] * 8, [ctx] * 8, seed=7)
    A = Bench(q1, cq1, ck1, bt1, 3, MAX_MODEL_LEN)
    B = Bench(q3, cq3, ck3, bt3, 3, MAX_MODEL_LEN)
    ab(f"ctx={ctx}", A, B, "T=8 ", "T=24")
    del A, B

print()
print("=" * 96)
print("方案 A 的白算代价:q 恒为 3(padding 草稿)vs 只算真实需要的行")
print("  —— 若一张图按 T=24 捕获,全未命中时仍只送 8 行(cu_seqlens_q 决定),不白算")
print("=" * 96)
ctx = 1000
q_full, cq_full, ck_full, bt_full = build([3] * 8, [ctx] * 8, seed=7)
# 同样 T=24 的缓冲区,但 cu_seqlens_q 只标 8 行有效、其余给 padding sink
qlens = [1] * 8 + [16]
klens = [ctx] * 8 + [16]
q_pad, cq_pad, ck_pad, bt_pad = build(qlens, klens, seed=7)
A = Bench(q_pad, cq_pad, ck_pad, bt_pad, 3, MAX_MODEL_LEN)
B = Bench(q_full, cq_full, ck_full, bt_full, 3, MAX_MODEL_LEN)
ab(f"ctx={ctx} T=24 缓冲", A, B, "8真+16pad", "24真     ")
