"""Phase 1 · Sampler 单元测试。不加载模型,直接喂随机 logits。"""
import torch

import common  # noqa: F401  (插入 sys.path)
from common import report
from nanovllm.layers.sampler import Sampler, apply_top_k_top_p, compute_probs

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V = 512
B = 8


def t(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=DEV)


def test_greedy_is_argmax():
    torch.manual_seed(0)
    logits = torch.randn(B, V, device=DEV)
    sampler = Sampler()
    temps = t([0.0] * B)
    tokens, _ = sampler(logits, temps)
    expect = logits.argmax(dim=-1)
    ok = torch.equal(tokens, expect)
    return report("greedy(temperature=0) == argmax", ok,
                  "" if ok else f"  got {tokens.tolist()}\n  want {expect.tolist()}")


def test_greedy_unaffected_by_truncation():
    """greedy 行即使同时设了 top_k/top_p,结果仍必须是原始 logits 的 argmax。"""
    torch.manual_seed(1)
    logits = torch.randn(B, V, device=DEV)
    sampler = Sampler()
    temps = t([0.0] * B)
    tokens, _ = sampler(logits, temps, top_ks=t([3] * B, torch.int64), top_ps=t([0.5] * B))
    ok = torch.equal(tokens, logits.argmax(dim=-1))
    return report("greedy + top_k/top_p 组合仍等于 argmax", ok)


def test_top_k_support():
    """采样 N 次,所有采到的 token 必须落在该行的 top-k 集合里。"""
    torch.manual_seed(2)
    logits = torch.randn(B, V, device=DEV)
    k = 5
    sampler = Sampler()
    temps = t([1.0] * B)
    top_ks = t([k] * B, torch.int64)
    allowed = [set(logits[i].topk(k).indices.tolist()) for i in range(B)]
    seen = [set() for _ in range(B)]
    for _ in range(300):
        tokens, _ = sampler(logits, temps, top_ks=top_ks)
        for i, tok in enumerate(tokens.tolist()):
            seen[i].add(tok)
    bad = [(i, seen[i] - allowed[i]) for i in range(B) if seen[i] - allowed[i]]
    # 300 次采样,k=5 时每个候选都该出现过(概率上几乎必然)
    thin = [i for i in range(B) if len(seen[i]) < 2]
    ok = not bad and not thin
    detail = ""
    if bad:
        detail += f"  越界采样: {bad}\n"
    if thin:
        detail += f"  采样多样性过低(可能没真正随机): 行 {thin}\n"
    return report(f"top_k={k} 采样支撑集 ⊆ top-k 集合", ok, detail)


def test_top_p_support():
    torch.manual_seed(3)
    logits = torch.randn(B, V, device=DEV)
    p = 0.8
    sampler = Sampler()
    temps = t([1.0] * B)
    top_ps = t([p] * B)
    # 手算每行的 nucleus 集合:按概率降序累加,直到累计 >= p
    allowed = []
    probs = logits.softmax(dim=-1)
    for i in range(B):
        sp, si = probs[i].sort(descending=True)
        c = sp.cumsum(0)
        cut = int((c < p).sum().item()) + 1     # 含使累计首次 >= p 的那个
        allowed.append(set(si[:cut].tolist()))
    seen = [set() for _ in range(B)]
    for _ in range(300):
        tokens, _ = sampler(logits, temps, top_ps=top_ps)
        for i, tok in enumerate(tokens.tolist()):
            seen[i].add(tok)
    bad = [(i, len(seen[i] - allowed[i])) for i in range(B) if seen[i] - allowed[i]]
    ok = not bad
    return report(f"top_p={p} 采样支撑集 ⊆ nucleus 集合", ok, "" if ok else f"  越界: {bad}")


def test_top_p_keeps_at_least_one():
    """极端 top_p(比最大概率还小)时不能把整行砍空。"""
    logits = torch.zeros(2, V, device=DEV)
    logits[:, 0] = 100.0     # 第 0 个 token 概率 ~1
    out = apply_top_k_top_p(logits, None, t([1e-6, 1e-6]))
    finite = torch.isfinite(out).sum(dim=-1)
    ok = bool((finite >= 1).all())
    probs = out.softmax(dim=-1)
    ok = ok and bool(torch.isfinite(probs).all()) and bool((probs.sum(dim=-1) > 0.99).all())
    return report("top_p 极小时仍保留至少一个候选", ok, "" if ok else f"  剩余候选数 {finite.tolist()}")


def test_ties_greedy_topk1_topp_agree():
    """并列最大值下,greedy / top_k=1 / top_p→0 必须给出同一个 token。

    回归用例:lm_head 输出是 bf16,只有 8 位尾数,15 万词表里出现并列最大值是常事。
    早先按"第 k 大的值"做阈值截断时,top_k=1 会把并列的全部留下再随机挑,和 greedy 对不上。
    """
    logits = torch.full((4, V), -5.0, device=DEV)
    # 每行制造 3 个完全并列的最大值,下标故意不连续
    for i in range(4):
        for j in (7 + i, 100 + i, 300 + i):
            logits[i, j] = 9.0
    sampler = Sampler()
    greedy, _ = sampler(logits, t([0.0] * 4))
    ok = True
    for _ in range(30):
        k1, _ = sampler(logits, t([1.0] * 4), top_ks=t([1] * 4, torch.int64))
        p0, _ = sampler(logits, t([1.0] * 4), top_ps=t([1e-6] * 4))
        ok = ok and torch.equal(k1, greedy) and torch.equal(p0, greedy)
    # top_k=1 只能留一个候选
    trunc = apply_top_k_top_p(logits, t([1] * 4, torch.int64), None)
    nnz = torch.isfinite(trunc).sum(dim=-1)
    ok = ok and bool((nnz == 1).all())
    return report("并列最大值下 greedy == top_k=1 == top_p→0,且 top_k 恰好保留 k 个", ok,
                  "" if ok else f"  greedy={greedy.tolist()} 保留候选数={nnz.tolist()}")


def test_top_k_exact_count():
    """top_k 必须恰好保留 k 个候选,即使存在并列。"""
    logits = torch.zeros(3, V, device=DEV)   # 全部并列
    for k in (1, 4, 17):
        trunc = apply_top_k_top_p(logits, t([k] * 3, torch.int64), None)
        nnz = torch.isfinite(trunc).sum(dim=-1)
        if not bool((nnz == k).all()):
            return report("全并列时 top_k 仍恰好保留 k 个", False, f"  k={k} 实际保留 {nnz.tolist()}")
    return report("全并列时 top_k 仍恰好保留 k 个", True)


def test_logprobs_values():
    torch.manual_seed(4)
    logits = torch.randn(B, V, device=DEV)
    sampler = Sampler()
    temps = t([1.0] * B)
    n = 5
    tokens, payload = sampler(logits, temps, max_logprobs=n)
    token_lp, top_ids, top_lp = payload
    ref = torch.log_softmax(logits.float(), dim=-1)
    ok = torch.allclose(token_lp, ref.gather(1, tokens.unsqueeze(1)).squeeze(1), atol=1e-5)
    ref_lp, ref_ids = ref.topk(n, dim=-1)
    ok = ok and torch.allclose(top_lp, ref_lp, atol=1e-5) and torch.equal(top_ids, ref_ids)
    return report(f"logprobs 数值 == log_softmax(logits) 的对应项 (top-{n})", ok)


def test_logprobs_with_temperature():
    """logprobs 取的是温度缩放后、截断前的分布。"""
    torch.manual_seed(5)
    logits = torch.randn(B, V, device=DEV)
    sampler = Sampler()
    temp = 0.7
    temps = t([temp] * B)
    _, payload = sampler(logits, temps, top_ks=t([10] * B, torch.int64), max_logprobs=3)
    _, top_ids, top_lp = payload
    ref = torch.log_softmax(logits.float() / temp, dim=-1)
    ref_lp, ref_ids = ref.topk(3, dim=-1)
    ok = torch.allclose(top_lp, ref_lp, atol=1e-5) and torch.equal(top_ids, ref_ids)
    return report("logprobs 在温度缩放后、top_k 截断前计算", ok)


def test_mixed_batch():
    """同一 batch 内 greedy / 采样 / 不同 top_k 混排,各行互不干扰。"""
    torch.manual_seed(6)
    logits = torch.randn(4, V, device=DEV)
    sampler = Sampler()
    temps = t([0.0, 1.0, 0.0, 0.5])
    top_ks = t([V, 4, V, 8], torch.int64)
    greedy_ref = logits.argmax(dim=-1)
    ok = True
    for _ in range(50):
        tokens, _ = sampler(logits, temps, top_ks=top_ks)
        ok = ok and tokens[0].item() == greedy_ref[0].item()
        ok = ok and tokens[2].item() == greedy_ref[2].item()
        ok = ok and tokens[1].item() in set(logits[1].topk(4).indices.tolist())
        ok = ok and tokens[3].item() in set(logits[3].topk(8).indices.tolist())
    return report("混合 batch(greedy + 不同 top_k)各行独立正确", ok)


def test_compute_probs_matches_sampler():
    """compute_probs 必须和 Sampler 内部用的分布完全一致(Phase 3 rejection sampling 依赖)。"""
    torch.manual_seed(7)
    logits = torch.randn(B, V, device=DEV)
    temps = t([0.9] * B)
    top_ks = t([16] * B, torch.int64)
    top_ps = t([0.9] * B)
    probs = compute_probs(logits, temps, top_ks, top_ps)
    manual = apply_top_k_top_p(logits.float() / 0.9, top_ks, top_ps).softmax(dim=-1)
    ok = torch.allclose(probs, manual, atol=1e-6)
    ok = ok and torch.allclose(probs.sum(dim=-1), torch.ones(B, device=DEV), atol=1e-5)
    nnz = (probs > 0).sum(dim=-1)
    ok = ok and bool((nnz <= 16).all())
    return report("compute_probs 与 Sampler 处理链一致且已归一化", ok,
                  "" if ok else f"  非零候选数 {nnz.tolist()}")


def test_distribution_correctness():
    """采样频率应逼近截断后的理论分布。"""
    torch.manual_seed(8)
    logits = torch.randn(1, 16, device=DEV) * 2
    temps = t([1.0])
    sampler = Sampler()
    N = 20000
    counts = torch.zeros(16, device=DEV)
    for _ in range(N // 100):
        rep = logits.expand(100, -1).contiguous()
        tokens, _ = sampler(rep, temps.expand(100).contiguous())
        counts += torch.bincount(tokens, minlength=16).float()
    emp = counts / counts.sum()
    ref = logits[0].softmax(dim=-1)
    max_dev = (emp - ref).abs().max().item()
    ok = max_dev < 0.02
    return report("采样频率收敛到理论分布 (max|Δp| < 0.02)", ok, f"  max|Δp| = {max_dev:.4f}")


if __name__ == "__main__":
    tests = [
        test_greedy_is_argmax,
        test_greedy_unaffected_by_truncation,
        test_top_k_support,
        test_top_p_support,
        test_top_p_keeps_at_least_one,
        test_ties_greedy_topk1_topp_agree,
        test_top_k_exact_count,
        test_logprobs_values,
        test_logprobs_with_temperature,
        test_mixed_batch,
        test_compute_probs_matches_sampler,
        test_distribution_correctness,
    ]
    print(f"=== Phase 1 Sampler 单元测试 (device={DEV}) ===")
    results = [fn() for fn in tests]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    raise SystemExit(0 if passed == len(results) else 1)
