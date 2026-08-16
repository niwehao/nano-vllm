"""Q4(任务书三问之外,但决定第二阶段设计):
varlen 图能不能容纳"批内 q 长度参差不齐"?

这是投机解码的核心矛盾。关键观察:
  cu_seqlens_q 是**设备张量**,和 cu_seqlens_k 一样可以 replay 前刷新。
  真正被烧进图的 host 标量只有 max_seqlen_q / max_seqlen_k,
  以及缓冲区切片决定的 (总 q 行数 T, seq 数 B)。
若 FA 的 grid 只由 max_seqlen_q 定上界、每个 (m_block, seq) 自己按 cu_seqlens_q 早退,
那么**一张图就能同时跑 q_len=1 和 q_len=k+1 的混合批**,不需要强制统一草稿长度。

逐条验证:
  4a  捕获 max_seqlen_q=K 的图,replay 批内 q 长度参差(1..K),对拍 eager
  4b  q_len=0 的 seq(用来填没排满的 seq 槽位)是否安全
  4c  q_len > k_len 的 padding seq 会怎样(causal 下 q 比 k 长)
  4d  烧死 max_seqlen_q=K 但实际全是 q_len=1 时,性能损失多少
      —— 这决定能不能用同一族图同时服务纯 decode 和投机 decode
"""
import math

import torch
from flash_attn import flash_attn_varlen_func

NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 8, 128
BLOCK_SIZE, MAX_MODEL_LEN, NUM_BLOCKS = 256, 4096, 512
MAX_NUM_BLOCKS = MAX_MODEL_LEN // BLOCK_SIZE
DTYPE, DEV = torch.bfloat16, "cuda"
SCALE = HEAD_DIM ** -0.5

torch.manual_seed(0)
k_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
v_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)

results = {}
fails = []


def ulp_at_scale(a, b):
    a32, b32 = a.float(), b.float()
    scale = torch.maximum(a32.abs().max(), b32.abs().max()).item()
    if scale <= 0:
        return 0.0
    u = 2.0 ** (math.floor(math.log2(scale)) - 7)
    return ((a32 - b32).abs().max() / u).item()


def run(q, cu_q, cu_k, bt, max_q, max_k):
    return flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                  max_seqlen_q=max_q, max_seqlen_k=max_k,
                                  softmax_scale=SCALE, causal=True, block_table=bt)


def build(qlens, klens, seed, T=None, B=None):
    """qlens/klens 等长。T/B 给定时把 cu_seqlens 垫到固定形状。"""
    assert len(qlens) == len(klens)
    n = len(qlens)
    T = T or sum(qlens)
    B = B or n
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(T, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV, generator=g)
    cu_q, cu_k = [0], [0]
    for i in range(n):
        cu_q.append(cu_q[-1] + qlens[i])
        cu_k.append(cu_k[-1] + klens[i])
    # 补足到 B 个 seq:q_len=0、k_len=0
    while len(cu_q) < B + 1:
        cu_q.append(cu_q[-1])
        cu_k.append(cu_k[-1])
    cu_q = torch.tensor(cu_q, dtype=torch.int32, device=DEV)
    cu_k = torch.tensor(cu_k, dtype=torch.int32, device=DEV)
    bt = torch.full((B, MAX_NUM_BLOCKS), -1, dtype=torch.int32, device=DEV)
    rng = torch.Generator().manual_seed(seed + 999)
    for i in range(n):
        nb = (klens[i] + BLOCK_SIZE - 1) // BLOCK_SIZE
        if nb:
            bt[i, :nb] = torch.randperm(NUM_BLOCKS, generator=rng)[:nb].to(torch.int32).to(DEV)
    return q, cu_q, cu_k, bt


print("=" * 78)
print("4a · 一张 max_seqlen_q=K 的图,跑批内参差 q 长度")
print("=" * 78)
K = 3                      # num_speculative_tokens=2 -> q_len 最大 1+2 = 3
BS = 8                     # 8 条并发(真实 seq 槽)
B = BS + 1                 # 多留 1 个槽当 padding sink,吸收没排满的 q 行
T = BS * K                 # 总 q 行数按最坏情况(全部命中)固定 = 24

# 静态缓冲区
s_q = torch.zeros(T, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
s_cu_q = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
s_cu_k = torch.zeros(B + 1, dtype=torch.int32, device=DEV)
s_bt = torch.zeros(B, MAX_NUM_BLOCKS, dtype=torch.int32, device=DEV)

# 捕获前先填一份合法数据(全 0 的 cu_seqlens 会让 warmup 什么都不算)
init_q, init_cuq, init_cuk, init_bt = build([K] * BS, [1000] * BS, seed=1, T=T, B=B)
s_q.copy_(init_q); s_cu_q.copy_(init_cuq); s_cu_k.copy_(init_cuk); s_bt.copy_(init_bt)

for _ in range(3):
    run(s_q, s_cu_q, s_cu_k, s_bt, K, MAX_MODEL_LEN)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    s_out = run(s_q, s_cu_q, s_cu_k, s_bt, K, MAX_MODEL_LEN)
torch.cuda.synchronize()
print(f"  捕获成功(max_seqlen_q={K}, T={T}, B={B})")

# 各种参差组合。sum(qlens) 必须 == T,不足的行给"padding seq"吸收。
# 这里先用"最后补一条 padding seq 吃掉剩余 q 行"的办法,k 长度设 = q 长度。
CASES = {
    "全部命中 (q=3×8)":        ([3] * 8,               [1000, 900, 800, 700, 600, 500, 400, 300]),
    "全部未命中 (q=1×8)":      ([1] * 8,               [1000, 900, 800, 700, 600, 500, 400, 300]),
    "一半命中":                 ([3, 1, 3, 1, 3, 1, 3, 1], [1000, 900, 800, 700, 600, 500, 400, 300]),
    "参差 1/2/3":               ([1, 2, 3, 2, 1, 3, 2, 1], [1000, 900, 800, 700, 600, 500, 400, 300]),
    "短上下文参差":             ([3, 1, 2, 1, 1, 3, 1, 2], [37, 256, 257, 300, 511, 512, 600, 1]),
}

for name, (qlens, klens) in CASES.items():
    n_real = sum(qlens)
    pad_rows = T - n_real
    # padding 行挂到一条额外的 padding seq 上(k 长度 = q 长度,行为良定义)
    ql = list(qlens); kl = list(klens)
    if pad_rows:
        ql.append(pad_rows)      # padding sink 吸收剩余 q 行
        kl.append(pad_rows)      # k 长度 = q 长度,causal 行为良定义
    assert len(ql) <= B, f"{len(ql)} > B={B}"
    q, cu_q, cu_k, bt = build(ql, kl, seed=42, T=T, B=B)
    # eager 参照:传真实 max_seqlen_q / max_seqlen_k
    o_eager = run(q, cu_q, cu_k, bt, max(ql), max(kl))
    # graph replay
    s_q.copy_(q); s_cu_q.copy_(cu_q); s_cu_k.copy_(cu_k); s_bt.copy_(bt)
    g.replay()
    torch.cuda.synchronize()
    o_graph = s_out.clone()
    # 只比真实行(padding seq 的行丢弃)
    og, oe = o_graph[:n_real], o_eager[:n_real]
    u = ulp_at_scale(og, oe)
    bw = torch.equal(og, oe)
    ok = u <= 4
    print(f"  {name}: pad_rows={pad_rows:>2}  bitwise={str(bw):>5}  ulp@scale={u:.2f}  "
          f"{'OK' if ok else 'FAIL'}")
    results[f"4a_{name}"] = dict(bitwise=bw, ulp=u)
    if not ok:
        fails.append(f"4a_{name}")

print()
print("=" * 78)
print("4b · q_len=0 的 seq(填没排满的 seq 槽位)")
print("=" * 78)
# 6 条真实 seq(q 长度参差)+ 2 条 q_len=0 的空槽,总 q 行数用 padding seq 补齐
qlens = [3, 1, 2, 3, 1, 2]      # 和 = 12
klens = [1000, 900, 800, 700, 600, 500]
pad_rows = T - sum(qlens)        # 12
ql = qlens + [pad_rows] + [0]    # 一条 padding seq 吃 12 行,一条 q_len=0 空槽
kl = klens + [pad_rows] + [0]
try:
    q, cu_q, cu_k, bt = build(ql, kl, seed=77, T=T, B=B)
    o_eager = run(q, cu_q, cu_k, bt, max(ql), max(kl))
    s_q.copy_(q); s_cu_q.copy_(cu_q); s_cu_k.copy_(cu_k); s_bt.copy_(bt)
    g.replay(); torch.cuda.synchronize()
    o_graph = s_out.clone()
    n_real = sum(qlens)
    u = ulp_at_scale(o_graph[:n_real], o_eager[:n_real])
    print(f"  未 trap。真实行 ulp@scale={u:.2f}  bitwise={torch.equal(o_graph[:n_real], o_eager[:n_real])}")
    print(f"  全部输出 finite: {torch.isfinite(o_graph).all().item()}  "
          f"NaN 数={int(torch.isnan(o_graph).sum())}")
    results["4b"] = dict(ulp=u, finite=bool(torch.isfinite(o_graph).all().item()))
    if u > 4:
        fails.append("4b")
except Exception as ex:
    print(f"  出错: {type(ex).__name__}: {ex}")
    fails.append("4b")

print()
print("=" * 78)
print("4c · padding seq 的 q_len > k_len(causal 下 q 比 k 长)")
print("=" * 78)
for kl_pad in [0, 1]:
    try:
        ql = [3, 1, 2, 3, 1, 2] + [T - 12]     # padding seq q_len=12
        kl = [1000, 900, 800, 700, 600, 500] + [kl_pad]
        q, cu_q, cu_k, bt = build(ql, kl, seed=88, T=T, B=B)
        o = run(q, cu_q, cu_k, bt, max(ql), MAX_MODEL_LEN)
        torch.cuda.synchronize()
        pad = o[12:]
        real = o[:12]
        print(f"  padding seq k_len={kl_pad}: 未 trap  "
              f"padding 行 NaN={torch.isnan(pad).any().item()} "
              f"全零={bool((pad==0).all().item())}  "
              f"真实行 finite={torch.isfinite(real).all().item()}")
        results[f"4c_klen{kl_pad}"] = dict(
            nan=bool(torch.isnan(pad).any().item()),
            allzero=bool((pad == 0).all().item()))
    except Exception as ex:
        print(f"  padding seq k_len={kl_pad}: 出错 {type(ex).__name__}: {ex}")
        results[f"4c_klen{kl_pad}"] = "ERROR"

print()
print("=" * 78)
print("4d · 烧死 max_seqlen_q=K 但实际全是 q_len=1 时,性能损失")
print("=" * 78)
REP = 50


def kernel_time(q, cu_q, cu_k, bt, max_q, max_k):
    keep = []
    for _ in range(3):
        run(q, cu_q, cu_k, bt, max_q, max_k)
    torch.cuda.synchronize()
    gg = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gg):
        for _ in range(REP):
            keep.append(run(q, cu_q, cu_k, bt, max_q, max_k))
    torch.cuda.synchronize()
    for _ in range(5):
        gg.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(20):
        gg.replay()
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) / 20 / REP * 1000
    del gg, keep
    return t


print(f"  {'场景':<34} {'max_q=1':>10} {'max_q=3':>10} {'比值':>8}")
for tag, klens in [("ctx=1000 均匀", [1000] * 8), ("ctx=256 均匀", [256] * 8),
                   ("ctx=2048 均匀", [2048] * 8)]:
    # 全部 q_len=1,T=8
    q1, cq1, ck1, bt1 = build([1] * 8, klens, seed=5)
    t1 = kernel_time(q1, cq1, ck1, bt1, 1, MAX_MODEL_LEN)
    t3 = kernel_time(q1, cq1, ck1, bt1, K, MAX_MODEL_LEN)
    print(f"  {tag:<34} {t1:>9.2f}u {t3:>9.2f}u {t3/t1:>7.3f}×")
    results[f"4d_{tag}"] = dict(t_maxq1=t1, t_maxq3=t3, ratio=t3 / t1)

print()
print("  参考:T=24(全部命中,q_len=3)时的 kernel 时间")
for tag, klens in [("ctx=1000 均匀", [1000] * 8)]:
    q3, cq3, ck3, bt3 = build([3] * 8, klens, seed=5)
    t = kernel_time(q3, cq3, ck3, bt3, K, MAX_MODEL_LEN)
    q1, cq1, ck1, bt1 = build([1] * 8, klens, seed=5)
    t1 = kernel_time(q1, cq1, ck1, bt1, 1, MAX_MODEL_LEN)
    print(f"  {tag}: q_len=1 (T=8) {t1:.2f}us  vs  q_len=3 (T=24) {t:.2f}us  "
          f"= {t/t1:.3f}×")
    results["4d_T24_vs_T8"] = dict(t_T8=t1, t_T24=t, ratio=t / t1)

print()
print("=" * 78)
print("FAIL 项:", fails if fails else "无")
print("=" * 78)
import json
print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
