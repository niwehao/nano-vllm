"""Phase 1 可行性验证:flash_attn_varlen_func 能不能进 CUDA graph。

独立脚本,不 import nanovllm,不改项目任何文件。
只依赖 torch + flash_attn,形状参数照抄 Qwen3-0.6B / nano-vllm 的默认配置。

回答三个问题:
  Q1  varlen kernel 能否被 torch.cuda.graph 捕获并正确 replay
  Q2  max_seqlen_k 传偏大值(max_model_len)是否仍正确、性能是否可接受
  Q3  padding 行(k 长度 0)会不会 NaN / trap,退路(k 长度 1 指向 block 0)是否可用

用法: .venv/bin/python phase1_varlen_graph_probe.py
"""
import json
import os
import sys
import time

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# ---------------------------------------------------------------- 配置
# 照抄 Qwen3-0.6B + nanovllm/config.py 的默认值
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 256          # config.kvcache_block_size
MAX_MODEL_LEN = 4096      # config.max_model_len (min(4096, 40960))
MAX_NUM_BLOCKS = (MAX_MODEL_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE   # 16
NUM_BLOCKS = 512          # 假 paged cache 的总块数
DTYPE = torch.bfloat16
DEV = "cuda"
SCALE = HEAD_DIM ** -0.5

MAX_BS = 8                # 本探针固定 batch=8(任务书 Q1 指定)

results = {}
_fail = []


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def record(key, value, ok=None):
    results[key] = value
    if ok is False:
        _fail.append(key)


def ulp_diff(a, b, dtype=torch.bfloat16):
    """max|Δ| 换算成"张量整体尺度上的 bf16 ulp"。

    注意:不能按**逐元素**量级算 ulp。attention 输出里必然有接近 0 的分量,
    而 kernel 内部的累加是在张量尺度(~0.35)上做的,一次 bf16 舍入落到一个
    量级 2e-5 的分量上,逐元素算出来就是几百 ulp —— 那是指标的假象,不是误差。
    项目 05-implementation-report 的 ulp 判据用在 logprob(O(1)~O(10),良态)上,
    逐元素算没问题;搬到原始 attention 输出上必须改成按整体尺度算。
    """
    import math
    a32, b32 = a.float(), b.float()
    scale = torch.maximum(a32.abs().max(), b32.abs().max()).item()
    if scale <= 0:
        return 0.0
    ulp_at_scale = 2.0 ** (math.floor(math.log2(scale)) - 7)   # bf16: 8 位尾数
    return ((a32 - b32).abs().max() / ulp_at_scale).item()


def rel_fro(a, b):
    a32, b32 = a.float(), b.float()
    return ((a32 - b32).norm() / b32.norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------- 假 paged KV cache
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

k_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
v_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)


def make_case(context_lens, seed=0, bs_pad=None, pad_mode="zero_len", pad_block=0):
    """构造一批 decode 输入(每条 q_len=1)。

    context_lens : list[int]  每条 seq 的真实 k 长度
    bs_pad       : 垫到多少行(None = 不垫)
    pad_mode     : "zero_len"  padding 段 cu_seqlens_k 与前一项相等(k 长度 0)
                   "one_len"   padding 行 k 长度设 1、block_tables 指向 pad_block
    返回 dict,含 q / cu_seqlens_q / cu_seqlens_k / block_tables / real_max_k
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    real_bs = len(context_lens)
    bs = bs_pad if bs_pad is not None else real_bs

    # q 恒按 MAX_BS 生成再切片:否则同 seed 下 bs=5 和 bs=8 的前 5 行并不相同
    # (CUDA randn 按元素总数做 Philox 分块),padded/unpadded 比对会假失败。
    q = torch.randn(MAX_BS, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV, generator=g)[:bs].contiguous()

    cu_q = torch.arange(0, bs + 1, dtype=torch.int32, device=DEV)   # 每条 q_len=1

    klens = list(context_lens)
    for _ in range(bs - real_bs):
        klens.append(0 if pad_mode == "zero_len" else 1)

    cu_k = [0]
    for L in klens:
        cu_k.append(cu_k[-1] + L)
    cu_k = torch.tensor(cu_k, dtype=torch.int32, device=DEV)

    # block table:真实行给连续的物理块,padding 行按 pad_mode 决定
    bt = torch.full((bs, MAX_NUM_BLOCKS), -1, dtype=torch.int32, device=DEV)
    rng = torch.Generator().manual_seed(seed + 999)
    for i in range(real_bs):
        nblk = (context_lens[i] + BLOCK_SIZE - 1) // BLOCK_SIZE
        # 打散块号,确保 kernel 真的走了页表翻译而不是顺序读
        blocks = torch.randperm(NUM_BLOCKS, generator=rng)[:nblk].to(torch.int32)
        bt[i, :nblk] = blocks.to(DEV)
    for i in range(real_bs, bs):
        if pad_mode == "one_len":
            bt[i, 0] = pad_block
        # zero_len 时整行保持 -1(= nano-vllm prepare_block_tables 的 padding 值)

    return dict(q=q, cu_q=cu_q, cu_k=cu_k, bt=bt, klens=klens,
                real_bs=real_bs, bs=bs, real_max_k=max(context_lens))


def run_varlen(case, max_seqlen_k):
    return flash_attn_varlen_func(
        case["q"], k_cache, v_cache,
        cu_seqlens_q=case["cu_q"], cu_seqlens_k=case["cu_k"],
        max_seqlen_q=1, max_seqlen_k=max_seqlen_k,
        softmax_scale=SCALE, causal=True, block_table=case["bt"])


def bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print(f"torch {torch.__version__} / flash_attn {__import__('flash_attn').__version__} / {torch.cuda.get_device_name(0)}")
print(f"q_heads={NUM_Q_HEADS} kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM} "
      f"block_size={BLOCK_SIZE} max_model_len={MAX_MODEL_LEN} bs={MAX_BS}")

# ================================================================= 预备:varlen(q=1) 与 with_kvcache 是否等价
# 这一步不在任务书的三问里,但它是后面所有比对的地基:
# 如果 varlen 在 q=1 时本来就和现有 decode kernel 不等价,那 Step B 从一开始就不成立。
section("预备 · varlen(q_len=1) vs flash_attn_with_kvcache(现有 decode 路径)")
ctx0 = [37, 256, 257, 1000, 511, 512, 4096, 1]
case0 = make_case(ctx0, seed=1)
o_varlen = run_varlen(case0, case0["real_max_k"])
o_kvcache = flash_attn_with_kvcache(
    case0["q"].unsqueeze(1), k_cache, v_cache,
    cache_seqlens=torch.tensor(ctx0, dtype=torch.int32, device=DEV),
    block_table=case0["bt"], softmax_scale=SCALE, causal=True).squeeze(1)
d = ulp_diff(o_varlen, o_kvcache)
same = torch.equal(o_varlen, o_kvcache)
print(f"  context_lens = {ctx0}")
print(f"  bitwise 相同: {same}   最大偏差: {d:.2f} ulp")
record("pre_varlen_vs_kvcache_ulp", d, ok=(d <= 4))
print(f"  -> {'一致(≤4 ulp)' if d <= 4 else '不一致!'}")

# ================================================================= Q1
section("Q1 · flash_attn_varlen_func 能否被 torch.cuda.graph 捕获")

# 静态缓冲区:地址固定,replay 前往里写数据
s_q = torch.zeros(MAX_BS, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
s_cu_q = torch.arange(0, MAX_BS + 1, dtype=torch.int32, device=DEV)   # 恒为 arange,捕获时就填死
s_cu_k = torch.zeros(MAX_BS + 1, dtype=torch.int32, device=DEV)
s_bt = torch.zeros(MAX_BS, MAX_NUM_BLOCKS, dtype=torch.int32, device=DEV)

CAPTURE_MAX_K = MAX_MODEL_LEN     # Q2 的"最坏情况"取值,捕获时烧死


def graph_body():
    return flash_attn_varlen_func(
        s_q, k_cache, v_cache,
        cu_seqlens_q=s_cu_q, cu_seqlens_k=s_cu_k,
        max_seqlen_q=1, max_seqlen_k=CAPTURE_MAX_K,
        softmax_scale=SCALE, causal=True, block_table=s_bt)


def load(case):
    """把一批输入拷进静态缓冲区。"""
    s_q.copy_(case["q"])
    s_cu_k.copy_(case["cu_k"])
    s_bt.fill_(-1)
    s_bt[:, :case["bt"].size(1)].copy_(case["bt"])


# 先用 case0 填一遍再 warmup,避免捕获时缓冲区还是全 0(cu_k 全 0 = 所有 seq 长度 0)
load(case0)

capture_ok, capture_err = False, None
mem_before = torch.cuda.memory_allocated()
try:
    # warmup:照 nano-vllm capture_cudagraph 的做法,先 eager 跑一次
    for _ in range(3):
        graph_body()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s_out = graph_body()
    torch.cuda.synchronize()
    capture_ok = True
    print("  捕获: 成功")
except Exception as ex:
    capture_err = f"{type(ex).__name__}: {ex}"
    print(f"  捕获: 失败 -> {capture_err}")

record("q1_capture_ok", capture_ok, ok=capture_ok)
record("q1_capture_err", capture_err)

if not capture_ok:
    print("\n  Q1 失败,按决策点表:停止。Q2/Q3 无意义。")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    sys.exit(1)

# replay 并与 eager 比对
load(case0)
graph.replay()
torch.cuda.synchronize()
o_graph = s_out.clone()
o_eager = run_varlen(case0, CAPTURE_MAX_K)
d = ulp_diff(o_graph, o_eager)
bitwise = torch.equal(o_graph, o_eager)
print(f"  replay vs eager(同一批输入,同一 max_seqlen_k={CAPTURE_MAX_K}):")
print(f"    bitwise 相同: {bitwise}   最大偏差: {d:.2f} ulp")
record("q1_replay_vs_eager_bitwise", bitwise)
record("q1_replay_vs_eager_ulp", d, ok=(d <= 4))

# 换一批数据再 replay:确认结果跟着变(不是把第一批答案烧死了)
case1 = make_case([100, 4096, 33, 2000, 777, 256, 1, 3000], seed=7)
load(case1)
graph.replay()
torch.cuda.synchronize()
o_graph1 = s_out.clone()
o_eager1 = run_varlen(case1, CAPTURE_MAX_K)
d1 = ulp_diff(o_graph1, o_eager1)
changed = not torch.equal(o_graph1, o_graph)
print(f"  换一批数据 replay:")
print(f"    输出跟着变: {changed}   vs eager 偏差: {d1:.2f} ulp   bitwise: {torch.equal(o_graph1, o_eager1)}")
record("q1_replay_data_follows", changed, ok=changed)
record("q1_replay2_vs_eager_ulp", d1, ok=(d1 <= 4))

# 隐性显存分配 / host 同步检查
mem_after = torch.cuda.memory_allocated()
torch.cuda.synchronize()
before = torch.cuda.memory_allocated()
for _ in range(20):
    graph.replay()
torch.cuda.synchronize()
after = torch.cuda.memory_allocated()
print(f"  20 次 replay 前后 memory_allocated 变化: {after - before} B "
      f"({'无动态分配' if after == before else '有分配!'})")
record("q1_replay_alloc_delta", after - before, ok=(after == before))

# ================================================================= Q2
section("Q2 · max_seqlen_k 传偏大值是否仍正确 / 性能如何")

# 注意:real_max_k 必须明显小于 MAX_MODEL_LEN,否则这一问是空的。
# 每档构造 8 条 seq,长度围绕该档的 max 波动,max 恰为档位值。
SWEEP = [32, 128, 512, 1024, 2048, 4096]
sweep_cases = {}
for m in SWEEP:
    lens = [max(1, m - 7 * i) for i in range(MAX_BS)]
    lens[0] = m                                   # 保证 real_max_k == m
    sweep_cases[m] = make_case(lens, seed=100 + m)

print("  [2a] eager 下:真实 max_seqlen_k vs max_model_len,输出比对")
print(f"    {'real_max_k':>10} {'bitwise':>8} {'ulp':>6}")
for m in SWEEP:
    case = sweep_cases[m]
    o_real = run_varlen(case, m)
    o_big = run_varlen(case, MAX_MODEL_LEN)
    d = ulp_diff(o_real, o_big)
    bw = torch.equal(o_real, o_big)
    print(f"    {m:>10} {str(bw):>8} {d:>6.2f}")
    record(f"q2_sweep{m}_real_vs_big_bitwise", bw)
    record(f"q2_sweep{m}_real_vs_big_ulp", d, ok=(d <= 4))

print(f"  [2b] eager 下:耗时对比(真实值 vs max_model_len={MAX_MODEL_LEN})")
print(f"    {'real_max_k':>10} {'t_real/us':>10} {'t_big/us':>10} {'big/real':>9}")
for m in SWEEP:
    case = sweep_cases[m]
    t_real = bench(lambda c=case, mm=m: run_varlen(c, mm))
    t_big = bench(lambda c=case: run_varlen(c, MAX_MODEL_LEN))
    ratio = t_big / t_real
    print(f"    {m:>10} {t_real*1000:>10.2f} {t_big*1000:>10.2f} {ratio:>8.3f}×")
    record(f"q2_sweep{m}_t_real_us", t_real * 1000)
    record(f"q2_sweep{m}_t_big_us", t_big * 1000)
    record(f"q2_sweep{m}_slowdown", ratio)

case_short = sweep_cases[32]

# [2b'] eager 的逐次调用被 Python/launch 开销(~80us)稀释,ratio 不敏感。
# 把同一个 kernel 在一张图里连录 REP 次,replay 一次再除 —— 得到干净的 GPU 时间。
REP = 50


def kernel_time(case, max_k):
    pool = []
    for _ in range(3):
        run_varlen(case, max_k)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REP):
            pool.append(run_varlen(case, max_k))
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(20):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) / 20 / REP * 1000    # us / 次
    del g, pool
    return t


print(f"  [2b'] 纯 GPU kernel 时间(图内连录 {REP} 次摊掉 launch 开销)")
print(f"    {'real_max_k':>10} {'k_real/us':>10} {'k_big/us':>10} {'big/real':>9}")
for m in SWEEP:
    case = sweep_cases[m]
    t_real = kernel_time(case, m)
    t_big = kernel_time(case, MAX_MODEL_LEN)
    ratio = t_big / t_real
    print(f"    {m:>10} {t_real:>10.2f} {t_big:>10.2f} {ratio:>8.3f}×")
    record(f"q2_sweep{m}_kernel_real_us", t_real)
    record(f"q2_sweep{m}_kernel_big_us", t_big)
    record(f"q2_sweep{m}_kernel_slowdown", ratio)

print("  [2c] graph replay(烧死 max_seqlen_k=4096) vs eager(传真实 max_seqlen_k)")
for name, case in [("case0", case0), ("case1", case1), ("case_short", case_short)]:
    load(case)
    graph.replay()
    torch.cuda.synchronize()
    og = s_out.clone()
    oe = run_varlen(case, case["real_max_k"])
    d = ulp_diff(og, oe)
    print(f"    {name}: bitwise={torch.equal(og, oe)}  偏差={d:.2f} ulp")
    record(f"q2_graph_vs_eagerreal_{name}_ulp", d, ok=(d <= 4))

print("  [2d] graph replay vs eager 的耗时(端到端 attention 单次调用)")
load(case0)
t_graph = bench(lambda: graph.replay())
t_eager = bench(lambda: run_varlen(case0, case0["real_max_k"]))
print(f"    graph.replay={t_graph*1000:7.2f} us   eager varlen={t_eager*1000:7.2f} us")
record("q2_t_graph_replay_us", t_graph * 1000)
record("q2_t_eager_us", t_eager * 1000)

# ================================================================= Q3
section("Q3 · padding 行怎么填才安全")

print("  [3a] pad_mode=zero_len(cu_seqlens_k padding 段与前一项相等,k 长度 0),"
      "block_tables padding 行 = -1")
case_pad0 = make_case([100, 512, 33, 2000, 777], seed=5, bs_pad=MAX_BS, pad_mode="zero_len")
err0 = None
try:
    o = run_varlen(case_pad0, MAX_MODEL_LEN)
    torch.cuda.synchronize()
    real = o[:case_pad0["real_bs"]]
    pad = o[case_pad0["real_bs"]:]
    print(f"    未 trap。真实行 finite={torch.isfinite(real).all().item()}  "
          f"padding 行: NaN={torch.isnan(pad).any().item()} "
          f"Inf={torch.isinf(pad).any().item()} "
          f"全零={bool((pad == 0).all().item())}")
    # 真实行是否受 padding 影响?与不垫齐时逐行比对
    case_nopad = make_case([100, 512, 33, 2000, 777], seed=5)
    o_nopad = run_varlen(case_nopad, MAX_MODEL_LEN)
    d = ulp_diff(real, o_nopad)
    print(f"    真实行 vs 不垫齐时: bitwise={torch.equal(real, o_nopad)}  偏差={d:.2f} ulp")
    record("q3_zerolen_ok", True)
    record("q3_zerolen_pad_nan", bool(torch.isnan(pad).any().item()))
    record("q3_zerolen_pad_allzero", bool((pad == 0).all().item()))
    record("q3_zerolen_real_ulp", d, ok=(d <= 4))
except Exception as ex:
    err0 = f"{type(ex).__name__}: {ex}"
    print(f"    出错: {err0}")
    record("q3_zerolen_ok", False)
    record("q3_zerolen_err", err0)

print("  [3b] pad_mode=one_len(padding 行 k 长度 1、block_tables 指向 block 0)—— 退路方案")
err1 = None
try:
    case_pad1 = make_case([100, 512, 33, 2000, 777], seed=5, bs_pad=MAX_BS,
                          pad_mode="one_len", pad_block=0)
    o = run_varlen(case_pad1, MAX_MODEL_LEN)
    torch.cuda.synchronize()
    real = o[:case_pad1["real_bs"]]
    pad = o[case_pad1["real_bs"]:]
    print(f"    未 trap。padding 行: NaN={torch.isnan(pad).any().item()} "
          f"Inf={torch.isinf(pad).any().item()} finite={torch.isfinite(pad).all().item()}")
    case_nopad = make_case([100, 512, 33, 2000, 777], seed=5)
    o_nopad = run_varlen(case_nopad, MAX_MODEL_LEN)
    d = ulp_diff(real, o_nopad)
    print(f"    真实行 vs 不垫齐时: bitwise={torch.equal(real, o_nopad)}  偏差={d:.2f} ulp")
    record("q3_onelen_ok", True)
    record("q3_onelen_pad_nan", bool(torch.isnan(pad).any().item()))
    record("q3_onelen_real_ulp", d, ok=(d <= 4))
except Exception as ex:
    err1 = f"{type(ex).__name__}: {ex}"
    print(f"    出错: {err1}")
    record("q3_onelen_ok", False)
    record("q3_onelen_err", err1)

print("  [3c] 在 CUDA graph 里 replay 带 padding 的批(用 3a 的填法)")
try:
    load(case_pad0)
    graph.replay()
    torch.cuda.synchronize()
    og = s_out.clone()
    real_g = og[:case_pad0["real_bs"]]
    oe = run_varlen(make_case([100, 512, 33, 2000, 777], seed=5), MAX_MODEL_LEN)
    d = ulp_diff(real_g, oe)
    pad_g = og[case_pad0["real_bs"]:]
    print(f"    replay 未 trap。真实行 vs eager 偏差={d:.2f} ulp  "
          f"padding 行 NaN={torch.isnan(pad_g).any().item()}")
    record("q3_graph_pad_ok", True)
    record("q3_graph_pad_real_ulp", d, ok=(d <= 4))
    record("q3_graph_pad_nan", bool(torch.isnan(pad_g).any().item()))
except Exception as ex:
    print(f"    出错: {type(ex).__name__}: {ex}")
    record("q3_graph_pad_ok", False)
    record("q3_graph_pad_err", f"{type(ex).__name__}: {ex}")

# ================================================================= 汇总
section("汇总")
print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
print()
if _fail:
    print(f"FAIL 项: {_fail}")
else:
    print("全部检查通过。")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase1_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"结果写入 {out_path}")
