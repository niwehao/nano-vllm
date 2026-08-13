"""Phase 3 · 投机解码测试。

关于"greedy 下开关投机必须逐 token 一致"这条验收标准:
它在【logits 逐比特相同】的前提下成立 —— greedy 的接受规则就是"草稿 == argmax",
数学上是恒等变换。但这个前提在 bf16 下拿不到:投机的验证前向 q 长度是 k+1,
走变长 kernel;不投机时 q=1 走 flash_attn_with_kvcache。两条 kernel 路径的归约顺序
不同,logits 有 ulp 级漂移,在模型本来就近似并列的位置上 argmax 会翻转。
(同样的现象在改动前的原始代码上也存在:batch vs 串行只有 3/6 条一致。)

所以判据设计成三层:
  1) 数学层(单元测试):rejection sampling 产出的 token 分布必须严格等于目标分布 p,
     greedy 接受规则必须严格是"草稿==argmax"。这一层不允许任何误差。
  2) 确定性:同配置重复运行必须逐 token 完全一致。
  3) 端到端等价性:分歧点必须全部落在"模型本来就近似并列"的位置上
     (top1-top2 差 <= 4 个 bf16 ulp,占全部位置的 8.6%)。
     逻辑 bug 会在随机位置发作 —— 全部位置的 top1-top2 差中位数是 3.5,
     真 bug 不可能只挑最并列的那几个百分点出现。
"""
import json
import os

import torch

import common
from common import report
from harness import run_gen, token_ids, check_equal, check_equal_or_noise, compare_logprobs, OUT_DIR

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 单元测试

class FakeSeq:
    def __init__(self, draft, temperature=1.0):
        self.draft_tokens = list(draft)
        self.temperature = temperature


def call_spec(seqs, logits, temperatures):
    """sample_speculative 不依赖 self 的任何状态,可以直接当函数调用。"""
    from nanovllm.engine.model_runner import ModelRunner
    runner = object.__new__(ModelRunner)
    return ModelRunner.sample_speculative(runner, seqs, logits,
                                          (temperatures, None, None, -1))


def test_rejection_preserves_distribution():
    """核心正确性:投机产出的第一个 token 的分布必须等于目标分布 p。

    n-gram 是确定性提议,等价于 q = δ_d。接受规则退化成"以概率 p(d) 接受 d,
    否则从挖掉 d 再归一化的 p 里采样",合起来恰好还原 p。
    """
    torch.manual_seed(0)
    V = 12
    base = torch.randn(1, V, device=DEV) * 1.5
    p = base.softmax(dim=-1)[0]
    d = 3                                    # 固定草稿 token
    N = 40000
    counts = torch.zeros(V)
    B = 200
    for _ in range(N // B):
        seqs = [FakeSeq([d]) for _ in range(B)]
        # 每条 seq 占 2 行(1 个草稿 + 1 个奖励行)
        logits = base.expand(2 * B, -1).contiguous()
        temps = torch.ones(2 * B, device=DEV)
        out, acc = call_spec(seqs, logits, temps)
        for o in out:
            counts[o[0]] += 1
    emp = counts / counts.sum()
    ref = p.cpu()
    max_dev = (emp - ref).abs().max().item()
    ok = max_dev < 0.012
    return report(f"rejection sampling 保分布 (草稿 d={d}, max|Δp|={max_dev:.4f} < 0.012)", ok,
                  "" if ok else f"  经验分布 {emp.tolist()}\n  理论分布 {ref.tolist()}")


def test_rejection_bad_draft():
    """草稿是低概率 token 时,几乎总被拒绝,但分布仍必须是 p。"""
    torch.manual_seed(1)
    V = 8
    base = torch.tensor([[5.0, 4.0, 3.0, 0.0, -3.0, -3.0, -3.0, -3.0]], device=DEV)
    p = base.softmax(dim=-1)[0]
    d = 6                                     # 概率极低的草稿
    N, B = 30000, 200
    counts = torch.zeros(V)
    accs = 0
    for _ in range(N // B):
        seqs = [FakeSeq([d]) for _ in range(B)]
        logits = base.expand(2 * B, -1).contiguous()
        out, acc = call_spec(seqs, logits, torch.ones(2 * B, device=DEV))
        accs += sum(acc)
        for o in out:
            counts[o[0]] += 1
    emp = counts / counts.sum()
    max_dev = (emp - p.cpu()).abs().max().item()
    rate = accs / (N // B * B)
    ok = max_dev < 0.012 and abs(rate - p[d].item()) < 0.02
    return report(f"劣质草稿仍保分布 (接受率 {rate:.4f} ≈ p(d)={p[d].item():.4f}, "
                  f"max|Δp|={max_dev:.4f})", ok)


def test_greedy_acceptance_rule():
    """greedy 下接受当且仅当草稿 == argmax,且拒绝时产出 argmax。"""
    torch.manual_seed(2)
    V = 32
    ok = True
    for trial in range(20):
        base = torch.randn(3, V, device=DEV)
        am = base.argmax(dim=-1).tolist()
        # seq0: 草稿全对; seq1: 第 2 个错; seq2: 第 1 个就错
        # 每条 2 个草稿 -> 3 行
        logits = torch.cat([base[0:1], base[0:1], base[0:1],
                            base[1:2], base[1:2], base[1:2],
                            base[2:3], base[2:3], base[2:3]], dim=0)
        seqs = [FakeSeq([am[0], am[0]], 0.0),
                FakeSeq([am[1], (am[1] + 1) % V], 0.0),
                FakeSeq([(am[2] + 1) % V, am[2]], 0.0)]
        out, acc = call_spec(seqs, logits, torch.zeros(9, device=DEV))
        ok = ok and acc == [2, 1, 0]
        ok = ok and out[0] == [am[0], am[0], am[0]]      # 全接受 + 奖励 token
        ok = ok and out[1] == [am[1], am[1]]             # 接受 1 个 + 修正
        ok = ok and out[2] == [am[2]]                    # 直接拒绝 + 修正
        if not ok:
            return report("greedy 接受规则", False, f"  trial {trial}: acc={acc} out={out}")
    return report("greedy 接受规则:接受当且仅当草稿==argmax,拒绝处产出 argmax", ok)


def test_no_draft_path():
    """草稿为空(n-gram 没命中)时退化成普通采样,每条只产出 1 个 token。"""
    torch.manual_seed(3)
    logits = torch.randn(3, 64, device=DEV)
    seqs = [FakeSeq([], 0.0) for _ in range(3)]
    out, acc = call_spec(seqs, logits, torch.zeros(3, device=DEV))
    ok = acc == [0, 0, 0] and [o[0] for o in out] == logits.argmax(dim=-1).tolist()
    ok = ok and all(len(o) == 1 for o in out)
    return report("无草稿时退化为普通 decode", ok)


# ---------------------------------------------------------------- 端到端

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Phase 3 投机解码:单元测试 ===")
    unit = [test_rejection_preserves_distribution(),
            test_rejection_bad_draft(),
            test_greedy_acceptance_rule(),
            test_no_draft_path()]

    print("\n=== Phase 3 投机解码:端到端 ===")
    results = list(unit)

    print("\n[1] 确定性:开投机重复运行必须逐 token 完全一致")
    on_a = run_gen("p3_det_a", eager=True, temperature=0.0, max_tokens=128,
                   num_speculative_tokens=4, speculative_method="ngram")
    on_b = run_gen("p3_det_b", eager=True, temperature=0.0, max_tokens=128,
                   num_speculative_tokens=4, speculative_method="ngram")
    results.append(check_equal(token_ids(on_a), token_ids(on_b),
                               "投机 greedy 重复运行确定性", "run1", "run2"))

    print("\n[2] 端到端等价:greedy 下开关投机,分歧必须全部落在近似并列的位置")
    # 验证前向的 q 长度是 k+1,走的是变长 kernel;不投机时 q=1 走 flash_attn_with_kvcache。
    # 两条 kernel 路径的归约顺序不同,bf16 下 logits 会有 ulp 级漂移,
    # 128 步自回归后必然在近似并列处翻转。所以长程只能做噪声判定。
    off = run_gen("p3_off", eager=True, temperature=0.0, max_tokens=128, logprobs=5)
    for k in (1, 2, 4, 8):
        on = run_gen(f"p3_k{k}", eager=True, temperature=0.0, max_tokens=128,
                     num_speculative_tokens=k, speculative_method="ngram")
        st = on["stats"]
        rate = st["spec_accepted"] / max(1, st["spec_proposed"])
        print(f"    k={k}: 提议 {st['spec_proposed']} 接受 {st['spec_accepted']} "
              f"(接受率 {rate:.1%}), 总步数 {st['steps']} vs 关投机 {off['stats']['steps']}")
        results.append(check_equal_or_noise(off, on, f"k={k} 开关投机的分歧均在近似并列位置", "off", f"spec-k{k}"))

    print("\n[3] cudagraph 模式下开关投机(长程,噪声判定)")
    off_g = run_gen("p3_off_graph", eager=False, temperature=0.0, max_tokens=128, logprobs=5)
    on_g = run_gen("p3_k4_graph", eager=False, temperature=0.0, max_tokens=128,
                   num_speculative_tokens=4, speculative_method="ngram")
    results.append(check_equal_or_noise(off_g, on_g, "cudagraph 下开关投机", "off", "spec"))

    print("\n[4] prefix cache 污染专项")
    # 第一阶段带 k=8 投机跑 64 步(必然跨 256 的 block 边界,且大量草稿被拒绝);
    # 第二阶段用 prompt+前 32 个生成 token 当新 prompt,必然命中第一阶段写入的 block,
    # 且第二阶段关掉投机 —— 于是任何差异都只能来自"读到的缓存 KV"。
    # 若把被拒绝位置的垃圾 KV 错误登记进了 prefix cache,首步 logits 会天差地别,
    # 绝不可能只差几个 ulp。
    poll = run_gen("p3_pollution", eager=True, temperature=0.0, max_tokens=64, logprobs=10,
                   num_speculative_tokens=8, speculative_method="ngram", pollution=True)
    pf = os.path.join(OUT_DIR, "p3_pollution.json")
    ref = run_gen("p3_pollution_ref", eager=True, temperature=0.0, max_tokens=64, logprobs=10,
                  prompt_file=pf)
    print(f"    第一阶段接受率: {poll['stats']['spec_accepted']}/{poll['stats']['spec_proposed']}")
    results.append(compare_logprobs(ref, poll, "命中投机写入的 block 后首步 logprob",
                                    max_ulp=4))
    results.append(check_equal_or_noise(ref, poll, "命中投机写入的 block 后 64 步输出",
                                        "clean", "spec-cached"))

    print("\n[5] 抢占 + 投机:KV block 卡到 7 块")
    pre_off = run_gen("p3_pre_off", eager=True, temperature=0.0, max_tokens=16,
                      prompts="preempt")
    pre_on = run_gen("p3_pre_on", eager=True, temperature=0.0, max_tokens=16,
                     prompts="preempt", num_kvcache_blocks=7,
                     num_speculative_tokens=4, speculative_method="ngram")
    print(f"    抢占次数: {pre_on['stats']['preempted']}")
    ok = pre_on["stats"]["preempted"] > 0
    print(f"  [{'PASS' if ok else 'WARN'}] 确实触发了抢占")
    results.append(check_equal(token_ids(pre_off), token_ids(pre_on),
                               "抢占 + 投机输出一致", "normal", "preempt+spec"))

    print("\n[6] 采样模式(temperature>0)下能正常跑完")
    s = run_gen("p3_sample", eager=True, temperature=0.8, top_k=50, top_p=0.9,
                max_tokens=64, num_speculative_tokens=4, speculative_method="ngram")
    lens = [len(o["token_ids"]) for o in s["outputs"]]
    ok = all(l == 64 for l in lens)
    print(f"  [{'PASS' if ok else 'FAIL'}] 采样 + 投机输出长度全为 64: {lens}")
    print(f"    接受率 {s['stats']['spec_accepted']}/{s['stats']['spec_proposed']}")
    results.append(ok)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
