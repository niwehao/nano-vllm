"""诊断:Q2/Q3 里 330-451 ulp 的差异到底是什么。

三个候选解释:
  (a) max_seqlen_k / batch size 真的改变了 kernel 的数值结果(split-K 归约顺序)
  (b) block_table 的 -1 padding 被越界读了
  (c) 我的 ulp 指标本身有问题 —— 对接近 0 的分量,相对指标会爆炸

分别量:max|Δ|、相对 Frobenius 误差、按"张量整体尺度"算的 ulp、以及差异分量的量级分布。
"""
import torch
from flash_attn import flash_attn_varlen_func

NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 8, 128
BLOCK_SIZE, MAX_MODEL_LEN, NUM_BLOCKS = 256, 4096, 512
MAX_NUM_BLOCKS = MAX_MODEL_LEN // BLOCK_SIZE
DTYPE, DEV = torch.bfloat16, "cuda"
SCALE = HEAD_DIM ** -0.5
MAX_BS = 8

torch.manual_seed(0)
k_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
v_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)


def make(context_lens, seed, bs_pad=None, bt_pad=-1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    real_bs = len(context_lens)
    bs = bs_pad or real_bs
    q = torch.randn(MAX_BS, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV, generator=g)[:bs].contiguous()
    cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=DEV)
    klens = list(context_lens) + [0] * (bs - real_bs)
    cu_k = [0]
    for L in klens:
        cu_k.append(cu_k[-1] + L)
    cu_k = torch.tensor(cu_k, dtype=torch.int32, device=DEV)
    bt = torch.full((bs, MAX_NUM_BLOCKS), bt_pad, dtype=torch.int32, device=DEV)
    rng = torch.Generator().manual_seed(seed + 999)
    for i in range(real_bs):
        nblk = (context_lens[i] + BLOCK_SIZE - 1) // BLOCK_SIZE
        bt[i, :nblk] = torch.randperm(NUM_BLOCKS, generator=rng)[:nblk].to(torch.int32).to(DEV)
    return q, cu_q, cu_k, bt, real_bs


def run(q, cu_q, cu_k, bt, max_k):
    return flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                  max_seqlen_q=1, max_seqlen_k=max_k,
                                  softmax_scale=SCALE, causal=True, block_table=bt)


def report(tag, a, b):
    a32, b32 = a.float(), b.float()
    diff = (a32 - b32).abs()
    scale = b32.abs().max().item()
    # bf16 在 "scale" 这个量级上的 1 ulp
    import math
    ulp_at_scale = 2.0 ** (math.floor(math.log2(scale)) - 7) if scale > 0 else 1.0
    rel_fro = (diff.norm() / b32.norm().clamp_min(1e-30)).item()
    nz = diff > 0
    print(f"  {tag}")
    print(f"    max|Δ| = {diff.max().item():.3e}   张量尺度 max|b| = {scale:.3e}   "
          f"1ulp@scale = {ulp_at_scale:.3e}")
    print(f"    按整体尺度算的 ulp = {diff.max().item()/ulp_at_scale:.2f}   "
          f"相对 Frobenius 误差 = {rel_fro:.3e}")
    print(f"    不同的分量数 = {int(nz.sum())}/{a.numel()} ({100*nz.float().mean():.2f}%)")
    if nz.any():
        # 这些不同的分量,它们本身有多大?
        mags = torch.maximum(a32.abs(), b32.abs())[nz]
        print(f"    差异分量的量级: 中位 {mags.median().item():.3e}  最大 {mags.max().item():.3e}  "
              f"最小 {mags.min().item():.3e}")
        # 每个差异分量,Δ 相对它自身量级
        rel_each = (diff[nz] / mags.clamp_min(1e-30))
        print(f"    逐分量相对误差: 中位 {rel_each.median().item():.3e}  最大 {rel_each.max().item():.3e}")
        # 差异最大的那个分量,值是多少
        i = diff.argmax()
        print(f"    max|Δ| 处: a={a32.flatten()[i].item():.6e}  b={b32.flatten()[i].item():.6e}")


print("=" * 78)
print("A. 同一 case,只改 max_seqlen_k (real=512 vs 4096)")
print("=" * 78)
lens = [512 - 7 * i for i in range(MAX_BS)]
lens[0] = 512
q, cu_q, cu_k, bt, _ = make(lens, seed=612)
o_real = run(q, cu_q, cu_k, bt, 512)
o_big = run(q, cu_q, cu_k, bt, MAX_MODEL_LEN)
report("real=512 vs big=4096", o_real, o_big)

print("\n  determinism 检查(同参数跑两次):")
print(f"    real 两次 bitwise 相同: {torch.equal(run(q,cu_q,cu_k,bt,512), o_real)}")
print(f"    big  两次 bitwise 相同: {torch.equal(run(q,cu_q,cu_k,bt,MAX_MODEL_LEN), o_big)}")

print("\n  逐行看差异(每行的 k 长度不同):")
for i in range(MAX_BS):
    d = (o_real[i].float() - o_big[i].float()).abs().max().item()
    print(f"    row{i} klen={lens[i]:>5}  max|Δ|={d:.3e}")

print()
print("=" * 78)
print("B. block_table 的 padding 值:-1 vs 0(排除越界读)")
print("=" * 78)
q2, cu_q2, cu_k2, bt_m1, _ = make(lens, seed=612, bt_pad=-1)
q3, cu_q3, cu_k3, bt_0, _ = make(lens, seed=612, bt_pad=0)
print(f"  bt padding=-1 与 =0 的 block_table 是否只在 padding 位不同: "
      f"{torch.equal(bt_m1[bt_m1>=0], bt_0[bt_m1>=0])}")
o_m1 = run(q2, cu_q2, cu_k2, bt_m1, MAX_MODEL_LEN)
o_0 = run(q3, cu_q3, cu_k3, bt_0, MAX_MODEL_LEN)
report("bt_pad=-1 vs bt_pad=0 (都用 max_k=4096)", o_m1, o_0)

print()
print("=" * 78)
print("C. batch size:bs=5(不垫) vs bs=8(垫 3 行 klen=0),比前 5 行")
print("=" * 78)
L5 = [100, 512, 33, 2000, 777]
qa, cqa, cka, bta, _ = make(L5, seed=5)
qb, cqb, ckb, btb, rb = make(L5, seed=5, bs_pad=8)
print(f"  前 5 行 q 是否相同: {torch.equal(qa, qb[:5])}")
print(f"  前 6 项 cu_k 是否相同: {torch.equal(cka, ckb[:6])}")
print(f"  前 5 行 bt 是否相同: {torch.equal(bta, btb[:5])}")
oa = run(qa, cqa, cka, bta, MAX_MODEL_LEN)
ob = run(qb, cqb, ckb, btb, MAX_MODEL_LEN)
report("bs=5 vs bs=8 前 5 行", oa, ob[:5])
print("\n  逐行:")
for i in range(5):
    d = (oa[i].float() - ob[i].float()).abs().max().item()
    print(f"    row{i} klen={L5[i]:>5}  max|Δ|={d:.3e}")

print()
print("=" * 78)
print("D. 参照系:纯 fp32 手算 attention 作为 ground truth,看谁更准")
print("=" * 78)
# 只对 row1 (klen=512) 手算
row, klen = 1, lens[1]
nblk = (klen + BLOCK_SIZE - 1) // BLOCK_SIZE
blocks = bt[row, :nblk].tolist()
kk = torch.cat([k_cache[b] for b in blocks], dim=0)[:klen]      # [klen, kvh, hd]
vv = torch.cat([v_cache[b] for b in blocks], dim=0)[:klen]
qq = q[row].float()                                             # [qh, hd]
# GQA: 16 q heads / 8 kv heads -> 每 2 个 q head 共用 1 个 kv head
rep = NUM_Q_HEADS // NUM_KV_HEADS
kk = kk.float().repeat_interleave(rep, dim=1)                   # [klen, qh, hd]
vv = vv.float().repeat_interleave(rep, dim=1)
logits = torch.einsum("hd,khd->hk", qq, kk) * SCALE
ref = torch.einsum("hk,khd->hd", logits.softmax(dim=-1), vv)
for tag, o in [("real=512", o_real), ("big=4096", o_big)]:
    d = (o[row].float() - ref).abs()
    print(f"  {tag}: vs fp32 参照 max|Δ| = {d.max().item():.3e}  "
          f"相对 Fro = {(d.norm()/ref.norm()).item():.3e}")
print(f"  参照本身的尺度 max|ref| = {ref.abs().max().item():.3e}")
print(f"  bf16 在该尺度的 1 ulp ≈ {2.0**(int(torch.floor(torch.log2(ref.abs().max()))) - 7):.3e}")
