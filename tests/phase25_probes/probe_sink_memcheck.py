"""复现/验证 review 指出的 S1:padding sink(q_len>0 且 k_len==0)会让 FA 走
flash_fwd_kernel.h:537 的 early-exit 分支,而该分支算 LSE 偏移用的是 **padded** 公式
(:545),忽略了 params.unpadded_lse —— 而 varlen 路径的 softmax_lse 是按
{num_heads, total_q} 的 **unpadded** 布局分配的(flash_api.cpp:652 + :688 传 true)。
于是 sink 那一格会写到 softmax_lse 缓冲区之外。

用法:
  compute-sanitizer --tool memcheck python probe_sink_memcheck.py sink     # 应报 invalid write
  compute-sanitizer --tool memcheck python probe_sink_memcheck.py padslots # 修复方案,应干净
  python probe_sink_memcheck.py formula                                    # 不用 sanitizer,
                                                                           # 直接证明用的是 padded 公式
"""
import sys

import torch
from flash_attn import flash_attn_varlen_func

H, HKV, D = 16, 8, 128
BLOCK, NBLOCK = 256, 64
DTYPE, DEV = torch.bfloat16, "cuda"
SCALE = D ** -0.5

torch.manual_seed(0)
k_cache = torch.randn(NBLOCK, BLOCK, HKV, D, dtype=DTYPE, device=DEV)
v_cache = torch.randn(NBLOCK, BLOCK, HKV, D, dtype=DTYPE, device=DEV)


def run(qlens, klens, max_q, max_k, total):
    n = len(qlens)
    q = torch.randn(total, H, D, dtype=DTYPE, device=DEV)
    cq, ck = [0], [0]
    for i in range(n):
        cq.append(cq[-1] + qlens[i])
        ck.append(ck[-1] + klens[i])
    cq = torch.tensor(cq, dtype=torch.int32, device=DEV)
    ck = torch.tensor(ck, dtype=torch.int32, device=DEV)
    bt = torch.full((n, 16), -1, dtype=torch.int32, device=DEV)
    for i in range(n):
        nb = max(1, (klens[i] + BLOCK - 1) // BLOCK)
        bt[i, :nb] = torch.arange(nb, dtype=torch.int32, device=DEV)
    o = flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cq, cu_seqlens_k=ck,
                               max_seqlen_q=max_q, max_seqlen_k=max_k,
                               softmax_scale=SCALE, causal=True, block_table=bt)
    torch.cuda.synchronize()
    return o


mode = sys.argv[1] if len(sys.argv) > 1 else "sink"

# 生产形态:bucket bs=8, k=2 -> q_max=3, total=24;8 条真实 seq + 1 个 sink
BS, QMAX = 8, 3
TOTAL = BS * QMAX                       # 24
REAL_Q = [3, 3, 3, 3, 3, 1, 1, 1]       # 5 命中 3 未命中 -> num_tokens = 17
REAL_K = [1000, 900, 800, 700, 600, 500, 400, 300]
NT = sum(REAL_Q)

if mode == "sink":
    # 现状:一个 sink 吃掉全部剩余行,k 长度 0  <-- 这就是被指出的写越界
    qlens = REAL_Q + [TOTAL - NT]
    klens = REAL_K + [0]
    print(f"[sink] nslot={len(qlens)} sink_q={TOTAL-NT} sink_k=0  "
          f"softmax_lse 尺寸 = H*total_q = {H*TOTAL}")
    print(f"       early-exit 分支会写到 (bidb*H+bidh)*max_seqlen_q + row, "
          f"bidb={len(qlens)-1} -> 最大约 "
          f"{((len(qlens)-1)*H + H-1)*QMAX + min(64, TOTAL-NT) - 1}")
    run(qlens, klens, QMAX, 4096, TOTAL)

elif mode == "padslots":
    # 修复:padding 行拆给多个 slot,每个 q_len<=q_max 且 k_len=q_len(指向真实 block),
    # 于是走**正常**分支(用 unpadded 偏移),不碰 early-exit。
    P = TOTAL - NT
    qlens, klens = list(REAL_Q), list(REAL_K)
    while P > 0:
        take = min(P, QMAX)
        qlens.append(take)
        klens.append(take)          # k_len = q_len -> n_block_max >= 1,不进 early-exit
        P -= take
    print(f"[padslots] nslot={len(qlens)} padding slots={len(qlens)-len(REAL_Q)} "
          f"每个 q_len<=q_max={QMAX}, k_len=q_len")
    run(qlens, klens, QMAX, 4096, TOTAL)

elif mode == "formula":
    # 不用 sanitizer 也能证明"用的是 padded 公式":
    # 挑一组形状让 padded 偏移**落在缓冲区内**,然后看 -inf 出现在哪。
    # 直接调底层 op 才拿得到 softmax_lse(公开 wrapper 只回 out)。
    h, max_q, nslot = H, 3, 4
    total = 60                                   # 缓冲 = h*total = 960,padded 最大偏移 189
    qlens = [20, 20, 17, 3]                      # 最后一格是 sink
    klens = [1000, 900, 800, 0]                  # sink k_len = 0
    n = len(qlens)
    q = torch.randn(total, h, D, dtype=DTYPE, device=DEV)
    cq, ck = [0], [0]
    for i in range(n):
        cq.append(cq[-1] + qlens[i]); ck.append(ck[-1] + klens[i])
    cq = torch.tensor(cq, dtype=torch.int32, device=DEV)
    ck = torch.tensor(ck, dtype=torch.int32, device=DEV)
    bt = torch.full((n, 16), -1, dtype=torch.int32, device=DEV)
    for i in range(n):
        nb = max(1, (klens[i] + BLOCK - 1) // BLOCK)
        bt[i, :nb] = torch.arange(nb, dtype=torch.int32, device=DEV)
    out, lse, _, _ = torch.ops.flash_attn._flash_attn_varlen_forward(
        q, k_cache, v_cache, cq, ck, max_q, 4096, 0.0, SCALE, True,
        -1, -1, 0.0, None, False, bt, None, None, False)
    torch.cuda.synchronize()
    lse = lse.reshape(-1)                       # 布局应是 [h, total_q] = [16, 60]
    print(f"softmax_lse 形状 {tuple(lse.shape)} = h*total_q = {h*total}")
    sink_b = n - 1
    for bidh in (0, 1, h - 1):
        padded = (sink_b * h + bidh) * max_q                 # kernel:545 的公式
        unpadded = bidh * total + cq[sink_b].item()          # 正确的 varlen 公式
        print(f"  bidh={bidh:2d}: padded 偏移={padded:4d} 值={lse[padded].item():>10.3g}   "
              f"unpadded 偏移={unpadded:4d} 值={lse[unpadded].item():>10.3g}")
    print("\n  -inf 落在 padded 偏移上 => early-exit 分支用的确实是 padded 公式")
    ninf = torch.isinf(lse) & (lse < 0)
    print(f"  全部 -inf 的下标: {ninf.nonzero().flatten().tolist()[:20]}")
    exp_pad = sorted((sink_b * h + b) * max_q + r for b in range(h) for r in range(qlens[-1]))
    print(f"  按 padded 公式预测的下标: {exp_pad[:20]}")

elif mode == "prod":
    # 直接复刻 ModelRunner.run_varlen_graph 的下标算法(不 import nanovllm),
    # 扫一批 (num_seqs, bucket, k, 命中组合),每个都真的调一次 FA。
    # 配合 compute-sanitizer + PYTORCH_NO_CUDA_MEMORY_CACHING=1 用。
    import itertools
    GRAPH_BS = [1, 2, 4, 8, 16, 32]
    ncase = 0
    for k in (1, 2, 4, 8):
        q_max = k + 1
        for num_seqs in (1, 2, 3, 5, 8, 9, 16, 17, 32):
            bs = next((x for x in GRAPH_BS if x >= num_seqs), None)
            if bs is None:
                continue
            total, nslot = bs * q_max, 2 * bs
            for pattern in ("allhit", "allmiss", "mixed"):
                if pattern == "allhit":
                    ql = [q_max] * num_seqs
                elif pattern == "allmiss":
                    ql = [1] * num_seqs
                else:
                    ql = [(q_max if i % 2 == 0 else 1) for i in range(num_seqs)]
                nt = sum(ql)
                kl = [300 + 17 * i for i in range(num_seqs)]
                # ---- 复刻生产代码的填法 ----
                cq = [0]
                ck = [0]
                for i in range(num_seqs):
                    cq.append(cq[-1] + ql[i])
                    ck.append(ck[-1] + kl[i])
                last_k = ck[-1]
                for t in range(1, nslot + 1 - num_seqs):
                    cq.append(min(nt + t * q_max, total))
                    ck.append(last_k + (cq[-1] - nt))
                assert len(cq) == nslot + 1 == len(ck), (len(cq), nslot + 1)
                assert cq[-1] == total, (cq[-1], total)
                q = torch.randn(total, H, D, dtype=DTYPE, device=DEV)
                cqt = torch.tensor(cq, dtype=torch.int32, device=DEV)
                ckt = torch.tensor(ck, dtype=torch.int32, device=DEV)
                bt = torch.full((nslot, 16), -1, dtype=torch.int32, device=DEV)
                for i in range(num_seqs):
                    nb = (kl[i] + BLOCK - 1) // BLOCK
                    bt[i, :nb] = torch.arange(nb, dtype=torch.int32, device=DEV)
                bt[num_seqs:nslot, 0] = 0
                o = flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cqt,
                                           cu_seqlens_k=ckt, max_seqlen_q=q_max,
                                           max_seqlen_k=4096, softmax_scale=SCALE,
                                           causal=True, block_table=bt)
                torch.cuda.synchronize()
                assert torch.isfinite(o[:nt]).all(), (k, num_seqs, pattern)
                # padding 行也必须被写过(非未初始化):这里只要求有限
                assert torch.isfinite(o).all(), f"padding 行出现非有限值 {k=} {num_seqs=} {pattern=}"
                ncase += 1
    print(f"[prod] {ncase} 个构型全部跑完,真实行与 padding 行均为有限值")

print("done")
