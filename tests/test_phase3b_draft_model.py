"""Phase 3B · 草稿模型路径测试。

关于"greedy 下开关投机必须逐 token 一致"这条硬判据 —— 先把话说清楚:

它在【目标模型的 logits 逐比特相同】的前提下才成立。这个前提在本项目里拿不到,
原因与 Phase 3 完全一样(见 test_phase3_spec.py:3-8):不投机时 q=1 走
flash_attn_with_kvcache,投机的验证前向 q=k+1 走变长 kernel,两条 kernel 路径归约
顺序不同,bf16 logits 有 ulp 级漂移,在本来就近似并列的位置上 argmax 会翻转。

**这不是本次改动引入的**,下面 A1/A2 用两条独立证据把它钉死:
  A1  把草稿模型换成目标模型自己(自草稿),输出与已验证过的 n-gram 路径
      **逐 token 全等**。两侧 kernel 路径完全相同,所以这一条不允许任何差异 ——
      它是"新代码有没有算错"的判据。
  A2  自草稿 vs 关投机的分歧位置/token/ulp 与 n-gram vs 关投机**完全重合**。
      同一个位置、同一对 token、同样的 ulp 差 —— 说明分歧的来源是那条 kernel 边界,
      与草稿从哪来无关。

判据因此分成四层,一层都没放宽:
  1) 数学层:通用接受式(q 是真实分布)产出的 token 分布必须严格等于 p;
     q=δ 时通用式与简化式必须逐位相同(证明 n-gram 那条路没被改坏)。
  2) 机械层:自草稿的接受率必须是 100%。草稿 KV 错位一格、position 算错、
     draft_num_cached_tokens 漏更新 —— 任何一个都会让它掉下来,这是最灵敏的探针。
  3) 逐位层:草稿图 replay vs 同形状 eager 必须 0 ulp。
  4) 端到端:分歧必须全部落在 <=4 ulp 的近似并列位置(harness 的既有判据)。
"""
import json
import os
import subprocess
import sys

import torch

import common
from common import report, MODEL_PATH
from harness import (run_gen, token_ids, check_equal, check_equal_or_noise,
                     PYTHON, OUT_DIR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_8B = os.path.expanduser("~/huggingface/Qwen3-8B")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-0.6B")


# ================================================================ 单元:接受规则

class FakeSeq:
    def __init__(self, draft, temperature=1.0):
        self.draft_tokens = list(draft)
        self.temperature = temperature


def call_spec(seqs, logits, temperatures, draft_probs=None):
    """sample_speculative 不依赖 self 的任何状态,可以直接当函数调用
    (沿用 test_phase3_spec.py 的做法,测的是真代码不是复刻品)。"""
    from nanovllm.engine.model_runner import ModelRunner
    runner = object.__new__(ModelRunner)
    return ModelRunner.sample_speculative(runner, seqs, logits,
                                          (temperatures, None, None, -1), draft_probs)


def test_general_rule_preserves_distribution():
    """通用式(q 是真实分布)产出的第一个 token 分布必须等于目标分布 p。

    这是整个任务里数学上不许有任何误差的地方。构造一对**故意错开**的 p 和 q
    (q 把概率压在 p 的低概率区),使接受率远离 1,残差项被充分激活。
    """
    torch.manual_seed(10)
    V = 16
    p_logits = torch.randn(1, V, device=DEV) * 1.5
    p = p_logits.softmax(dim=-1)[0]
    # q 用反号 logits:p 高的地方 q 低,最大化 (p-q)+ 的作用
    q_row = (-p_logits).softmax(dim=-1)[0]
    N, B = 60000, 300
    counts = torch.zeros(V)
    acc_total = 0
    for _ in range(N // B):
        d = torch.multinomial(q_row, B, replacement=True)          # 草稿按 q 采
        seqs = [FakeSeq([int(x)]) for x in d.tolist()]
        logits = p_logits.expand(2 * B, -1).contiguous()
        temps = torch.ones(2 * B, device=DEV)
        out, acc = call_spec(seqs, logits, temps, [q_row.expand(B, V).contiguous()])
        acc_total += sum(acc)
        for o in out:
            counts[o[0]] += 1
    emp = counts / counts.sum()
    max_dev = (emp - p.cpu()).abs().max().item()
    # 理论接受率 = Σ_x q(x)·min(1, p(x)/q(x)) = Σ_x min(p,q)
    theo_acc = torch.minimum(p, q_row).sum().item()
    emp_acc = acc_total / (N // B * B)
    ok = max_dev < 0.012 and abs(emp_acc - theo_acc) < 0.02
    return report(
        f"通用式保分布 (max|Δp|={max_dev:.4f} < 0.012, "
        f"接受率 {emp_acc:.4f} ≈ Σmin(p,q)={theo_acc:.4f})", ok,
        "" if ok else f"  经验 {emp.tolist()}\n  理论 {p.cpu().tolist()}")


def test_general_rule_matches_simplified_when_delta():
    """q = δ_d 时,通用式与 n-gram 的简化式必须是**同一个数学对象**。

    判据不能写成"同种子下逐 token 相同":两条分支物化的张量大小不同
    (通用式只在草稿行上算残差,[B·k, V];简化式在全部行上算,[B·(k+1), V]),
    sample_from_probs 里的 exponential_ 抽样个数因此不同,随机流必然错开。
    要求随机流一致是在测一件与正确性无关的事。

    真正要证的是两条分支定义的**分布**相同,分两段测,都不含随机流依赖:
      (a) 确定性角:把 p 做成尖峰,接受/拒绝与修正 token 都唯一确定,
          两条分支必须给出完全相同的结果 —— 这是精确等式,不是统计。
      (b) 统计角:普通 p 上各跑 4 万次,输出分布与接受率必须互相吻合,
          且都吻合理论值 p(d)。
    """
    V = 24
    ok, detail = True, ""

    # ---- (a) 确定性角 ----
    for peak_on_draft in (True, False):
        d, other = 5, 11
        logits = torch.full((1, V), -30.0, device=DEV)
        logits[0, d if peak_on_draft else other] = 30.0     # softmax 后几乎是 one-hot
        rows = logits.expand(6, V).contiguous()             # 3 条 seq × (1 草稿 + 1 奖励)
        onehot = torch.zeros(3, V, device=DEV)
        onehot[:, d] = 1.0
        seqs_s = [FakeSeq([d]) for _ in range(3)]
        seqs_g = [FakeSeq([d]) for _ in range(3)]
        temps = torch.ones(6, device=DEV)
        out_s, acc_s = call_spec(seqs_s, rows, temps, None)
        out_g, acc_g = call_spec(seqs_g, rows, temps, [onehot])
        want_acc = [1, 1, 1] if peak_on_draft else [0, 0, 0]
        if acc_s != want_acc or acc_g != want_acc or out_s != out_g:
            ok = False
            detail = (f"  确定性角 peak_on_draft={peak_on_draft}: "
                      f"简化 {out_s}/{acc_s} vs 通用 {out_g}/{acc_g}, 期望 acc={want_acc}")
            break
    if not ok:
        return report("q=δ 时通用式 ≡ n-gram 简化式", False, detail)

    # ---- (b) 统计角 ----
    torch.manual_seed(21)
    base = torch.randn(1, V, device=DEV) * 1.2
    p = base.softmax(dim=-1)[0]
    d = 7
    N, B = 40000, 400
    emp = {}
    acc_rate = {}
    for tag in ("simplified", "general"):
        counts = torch.zeros(V)
        accs = 0
        onehot = torch.zeros(B, V, device=DEV)
        onehot[:, d] = 1.0
        torch.manual_seed(4242)
        for _ in range(N // B):
            seqs = [FakeSeq([d]) for _ in range(B)]
            logits = base.expand(2 * B, -1).contiguous()
            out, acc = call_spec(seqs, logits, torch.ones(2 * B, device=DEV),
                                 None if tag == "simplified" else [onehot])
            accs += sum(acc)
            for o in out:
                counts[o[0]] += 1
        emp[tag] = counts / counts.sum()
        acc_rate[tag] = accs / N
    cross = (emp["simplified"] - emp["general"]).abs().max().item()
    to_p = max((emp[t] - p.cpu()).abs().max().item() for t in emp)
    d_acc = abs(acc_rate["simplified"] - acc_rate["general"])
    ok = cross < 0.012 and to_p < 0.012 and d_acc < 0.02 \
        and abs(acc_rate["general"] - p[d].item()) < 0.02
    return report(
        f"q=δ 时通用式 ≡ n-gram 简化式(确定性角精确相等;"
        f"4 万次 max|Δ互相|={cross:.4f}, max|Δ理论|={to_p:.4f}, "
        f"接受率 {acc_rate['simplified']:.4f}/{acc_rate['general']:.4f} ≈ p(d)={p[d].item():.4f})",
        ok, "" if ok else f"  emp {emp}\n  acc {acc_rate}")


def test_general_rule_residual_shape():
    """拒绝时的修正 token 必须服从 norm(clamp(p-q,0)),不是别的分布。

    把草稿固定成一个 q 极高、p 极低的 token,使其几乎必被拒绝,
    于是产出的 token 分布就直接暴露残差分布本身。
    """
    torch.manual_seed(11)
    V = 10
    p_logits = torch.tensor([[3.0, 2.5, 2.0, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0, -4.0]],
                            device=DEV)
    p = p_logits.softmax(dim=-1)[0]
    q_row = torch.full((V,), 0.02 / (V - 1), device=DEV)
    d = 9                                    # p 最低的位置
    q_row[d] = 0.98                          # q 几乎全押在这里 -> 几乎必被拒
    resid = (p - q_row).clamp_min(0)
    resid = resid / resid.sum()
    N, B = 60000, 300
    counts = torch.zeros(V)
    rejects = 0
    for _ in range(N // B):
        seqs = [FakeSeq([d]) for _ in range(B)]
        logits = p_logits.expand(2 * B, -1).contiguous()
        out, acc = call_spec(seqs, logits, torch.ones(2 * B, device=DEV),
                             [q_row.expand(B, V).contiguous()])
        rejects += sum(1 for a in acc if a == 0)
        for o, a in zip(out, acc):
            if a == 0:
                counts[o[0]] += 1
    emp = counts / counts.sum()
    max_dev = (emp - resid.cpu()).abs().max().item()
    ok = max_dev < 0.012
    return report(f"拒绝时的修正 token 服从 norm(clamp(p-q,0)) "
                  f"(拒绝 {rejects}/{N} 次, max|Δ|={max_dev:.4f} < 0.012)", ok,
                  "" if ok else f"  经验 {emp.tolist()}\n  理论 {resid.cpu().tolist()}")


def test_general_rule_degenerate_q():
    """q 退化时不许吐出 token 0 —— 独立 review 抓出来的坑,补的回归。

    三种退化,每一种都曾经会让 sample_from_probs 对一行全零/全 NaN 取 argmax、
    返回下标 0,于是 **token id 0 被当成真 token 写进输出**:
      (a) q 里带 NaN(草稿 KV 读到未初始化显存时会这样);
      (b) q 与 p 逐元素相等 -> 残差恒为 0;
      (c) q(d) == 0 却提议了 d。
    判据不是"别崩",而是"产出的 token 仍然服从 p":拿 token 0 的经验频率
    和它在 p 里的真实概率比,差太多就说明退化分支在乱吐 0。
    """
    torch.manual_seed(30)
    V = 12
    p_logits = torch.randn(1, V, device=DEV) * 1.5
    p = p_logits.softmax(dim=-1)[0]
    N, B = 30000, 300
    d = 4
    ok, detail = True, []

    def run(q_row):
        counts = torch.zeros(V)
        accs = 0
        for _ in range(N // B):
            seqs = [FakeSeq([d]) for _ in range(B)]
            logits = p_logits.expand(2 * B, -1).contiguous()
            out, acc = call_spec(seqs, logits, torch.ones(2 * B, device=DEV),
                                 [q_row.expand(B, V).contiguous()])
            accs += sum(acc)
            for o in out:
                counts[o[0]] += 1
        return counts / counts.sum(), accs / N

    # (a) q 里有 NaN,且构造成必然走到拒绝分支(q(d)=1 让 ratio=p(d) 很小)。
    #     旧代码:残差整行 NaN -> argmax 返回下标 0 -> 每次都吐 token 0。
    #     新代码:rsum 是 NaN -> 回退到 p 采样。
    q_nan = torch.zeros(V, device=DEV)
    q_nan[d] = 1.0
    q_nan[0 if d != 0 else 1] = float("nan")
    emp, rate = run(q_nan)
    # 输出 = 以 p(d) 接受 d,否则从 p 采 -> p(d)·δ_d + (1-p(d))·p
    pc = p.cpu()
    want = (1 - pc[d]) * pc
    want[d] += pc[d]
    dev = (emp - want).abs().max().item()
    good = dev < 0.02 and abs(rate - p[d].item()) < 0.02
    ok &= good
    detail.append(f"    NaN: max|Δ理论|={dev:.4f}, 接受率 {rate:.4f}≈p(d)={p[d].item():.4f}, "
                  f"P(token 0)={emp[0].item():.4f}")

    # (b) q(d)==0:必须**判拒绝**(与 vLLM 一致),然后从 norm((p-q)^+) 采。
    #     q 取"除 d 外均匀",这样残差是个真分布而不是又一个 δ。
    q_zero = torch.full((V,), 1.0 / (V - 1), device=DEV)
    q_zero[d] = 0.0
    emp, rate = run(q_zero)
    resid = (p - q_zero).clamp_min(0)
    resid = (resid / resid.sum()).cpu()
    dev = (emp - resid).abs().max().item()
    good = rate == 0.0 and dev < 0.02
    ok &= good
    detail.append(f"    q(d)=0: 接受率 {rate:.4f}(必须恰为 0), "
                  f"残差 max|Δ理论|={dev:.4f}, P(token 0)={emp[0].item():.4f}")

    # (c) q == p:ratio 恒为 1,rand∈[0,1) 必接受,残差那一支**不可达**。
    #     留着这一格是为了把"不可达"这件事钉在测试里:一旦哪天它变成可达的,
    #     接受率就不再是 1.0,这里会立刻响。
    emp, rate = run(p.clone())
    good = rate == 1.0 and abs(emp[d].item() - 1.0) < 1e-9
    ok &= good
    detail.append(f"    q==p: 接受率 {rate:.4f}(必须恰为 1,证明 rsum==0 那支不可达)")

    return report("q 退化(NaN / q(d)=0 / q==p)时不吐 token 0,分布仍正确", ok,
                  "\n".join(detail))


def test_general_rule_greedy_unaffected():
    """temperature=0 的行必须仍然走 token 比较,不受 draft_probs 影响。"""
    torch.manual_seed(12)
    V = 32
    base = torch.randn(2, V, device=DEV)
    am = base.argmax(dim=-1).tolist()
    logits = torch.cat([base[0:1].expand(3, V), base[1:2].expand(3, V)], dim=0)
    seqs = [FakeSeq([am[0], am[0]], 0.0), FakeSeq([(am[1] + 1) % V, am[1]], 0.0)]
    # 给一份"乱写"的 q:greedy 行根本不该看它
    junk = torch.rand(2, V, device=DEV)
    junk = junk / junk.sum(dim=-1, keepdim=True)
    out, acc = call_spec(seqs, logits, torch.zeros(6, device=DEV), [junk, junk])
    ok = acc == [2, 0] and out[0] == [am[0], am[0], am[0]] and out[1] == [am[1]]
    return report("temperature=0 行不受 draft_probs 影响(仍是 token 比较)", ok,
                  "" if ok else f"  acc={acc} out={out}")


# ================================================================ 端到端

def suite_equivalence():
    """A:等价性。"""
    print("\n[A] 等价性")
    res = []
    common_kw = dict(max_tokens=48, gpu_util=0.55, temperature=0.0)

    off = run_gen("p3b_off", num_speculative_tokens=0, logprobs=20,
                  max_tokens=48, gpu_util=0.35, temperature=0.0)
    ngram = run_gen("p3b_ngram", num_speculative_tokens=2, speculative_method="ngram",
                    max_tokens=48, gpu_util=0.35, temperature=0.0)
    self_draft = run_gen("p3b_self", num_speculative_tokens=2, speculative_method="model",
                         speculative_model=MODEL_PATH, **common_kw)

    # A1 —— 唯一"必须 0 差异"的端到端判据:两侧 kernel 路径相同
    res.append(check_equal(token_ids(ngram), token_ids(self_draft),
                           "A1 自草稿(model 路径)与 n-gram 路径逐 token 全等",
                           "ngram-k2", "model-k2-self"))

    # A2 —— 与关投机的分歧必须与 n-gram 的分歧完全重合
    def diverge_set(p):
        s = {}
        for i, (x, y) in enumerate(zip(token_ids(off), token_ids(p))):
            if x != y:
                j = next(k for k in range(min(len(x), len(y))) if x[k] != y[k])
                s[i] = (j, x[j], y[j])
        return s
    dn, ds = diverge_set(ngram), diverge_set(self_draft)
    ok = dn == ds
    res.append(report(f"A2 自草稿与 n-gram 相对关投机的分歧集合完全相同 "
                      f"({len(dn)} 处: {sorted(dn.items())})", ok,
                      "" if ok else f"  ngram {dn}\n  model {ds}"))

    # A3 —— 既有的噪声判据(<=4 ulp)
    res.append(check_equal_or_noise(off, self_draft,
                                    "A3 自草稿 vs 关投机的分歧全在近似并列位置",
                                    "off", "model-k2-self"))

    # A4 —— 确定性
    r2 = run_gen("p3b_self2", num_speculative_tokens=2, speculative_method="model",
                 speculative_model=MODEL_PATH, **common_kw)
    res.append(check_equal(token_ids(self_draft), token_ids(r2),
                           "A4 同配置重复运行逐 token 全等", "run1", "run2"))

    # A5 —— 机械层最灵敏的探针
    st = self_draft["stats"]
    rate = st["spec_accepted"] / max(1, st["spec_proposed"])
    ok = st["spec_accepted"] == st["spec_proposed"]
    res.append(report(f"A5 自草稿接受率 = 100% "
                      f"({st['spec_accepted']}/{st['spec_proposed']} = {rate:.4f})", ok,
                      "" if ok else "  草稿 KV 错位/position 算错/draft_num_cached_tokens 漏更新都会让它掉下来"))
    return res


def suite_pairing():
    """B:真实配对 8B + 0.6B。"""
    print("\n[B] 真实配对 Qwen3-8B(目标) × Qwen3-0.6B(草稿)")
    res = []
    if not os.path.isdir(TARGET_8B):
        return [report("B 真实配对(缺 Qwen3-8B,跳过)", True)]
    kw = dict(model=TARGET_8B, gpu_util=0.90, max_model_len=2048,
              max_num_seqs=16, max_tokens=32, temperature=0.0)
    off = run_gen("p3b_8b_off", num_speculative_tokens=0, logprobs=20, **kw)
    on = run_gen("p3b_8b_model", num_speculative_tokens=2, speculative_method="model",
                 speculative_model=DRAFT, **kw)
    res.append(check_equal_or_noise(off, on, "B1 8B+0.6B vs 关投机的分歧全在近似并列位置",
                                    "off", "model-k2"))
    st = on["stats"]
    rate = st["spec_accepted"] / max(1, st["spec_proposed"])
    steps_off, steps_on = off["stats"]["decode_only"], st["decode_only"]
    res.append(report(f"B2 接受率 {rate:.3f},decode step {steps_off} → {steps_on} "
                      f"(每步净产出 {32 / max(1, steps_on):.2f} tok)",
                      rate > 0.0))
    # 走图占比:草稿那 k 次前向必须真的在 replay,否则 E3 算出来的收益全没了
    g = st["exec_draft_graph_varlen"] + st["exec_draft_graph_decode"]
    e = st["exec_draft_eager"]
    ratio = g / max(1, g + e)
    res.append(report(f"B3 草稿前向走图占比 {ratio:.3f} "
                      f"(图 {g} / eager {e},eager 的那次是 prefill 同步)", ratio >= 0.90))
    return res


def suite_edge():
    """C:边界与簿记。"""
    print("\n[C] 边界与簿记")
    res = []
    base = dict(speculative_method="model", speculative_model=MODEL_PATH,
                gpu_util=0.55, temperature=0.0, max_tokens=48)
    ref = run_gen("p3b_c_off", num_speculative_tokens=0, gpu_util=0.35,
                  temperature=0.0, max_tokens=48, logprobs=20)
    for k in (1, 2, 4):
        r = run_gen(f"p3b_c_k{k}", num_speculative_tokens=k, **base)
        res.append(check_equal_or_noise(ref, r, f"C1 k={k} 跑通且分歧全在近似并列位置",
                                        "off", f"model-k{k}"))
    # 混批:prefill 与 decode 同批,草稿要对 prefill chunk 做同步
    mix = run_gen("p3b_c_mix", num_speculative_tokens=2, repeat=3,
                  max_num_batched_tokens=512, **base)
    res.append(report(f"C2 混批(prefill+decode 同批){mix['stats']['mixed']} 个混批 step,"
                      f"草稿 prefill 同步 {mix['stats']['exec_draft_eager']} 次",
                      mix["stats"]["mixed"] > 0))
    # prefix cache 污染:第二遍必然命中第一遍(投机)写入的 block
    poll = run_gen("p3b_c_poll", num_speculative_tokens=2, pollution=True, **base)
    prompts_file = os.path.join(OUT_DIR, "p3b_c_poll.json")
    ref2 = run_gen("p3b_c_poll_ref", num_speculative_tokens=0, gpu_util=0.35,
                   temperature=0.0, max_tokens=48, logprobs=20,
                   prompt_file=prompts_file)
    res.append(check_equal_or_noise(ref2, poll,
                                    "C3 命中草稿模型写入的 block 后输出仍等价",
                                    "serial", "spec-poll"))
    # C3b —— 独立 review 指出 C3 本身**测不到**污染:--pollution 的第二阶段是最后一阶段,
    # 那时被污染的 block 再没人读回来。所以这里改成直接查那条不变量的**机械前提**:
    # 「每一个 decode step 都必须有一次草稿的第一次前向」。
    # 第二阶段把 num_spec_tokens 置 0,那些 step 不打草稿,但仍然必须调 sync_draft_decode
    # 把草稿 KV 追平 —— 否则 hash_blocks 会把一整段草稿 KV 从没写过的 block 登记进
    # prefix cache 索引。计数对不上就说明那条同步路径没跑。
    st = poll["stats"]
    first_fwd = st["exec_draft_graph_varlen"] + st["exec_draft_eager"] - st["prefill_only"]
    res.append(report(
        f"C3b 每个 decode step 都推进了草稿 KV "
        f"(草稿首次前向 {first_fwd} 次 vs decode step {st['decode_only']} 次;"
        f"其中第二阶段 num_spec_tokens=0,靠 sync_draft_decode 兜底)",
        first_fwd == st["decode_only"]))
    # 抢占:小 block 数逼出抢占,草稿必须跟着重 prefill。
    # 块数必须按需求算,不能随手填一个数:preempt prompt 是 6 条 × 255 token,
    # 每条 prefill 后占 1 块;k=2 时 ensure_capacity(seq,2) 要覆盖到位置 256,
    # 于是每条立刻要第 2 块 —— 峰值需求 12 块。填 32 就一次也压不出抢占
    # (第一次跑就是这么 FAIL 的)。沿用 test_phase3_spec.py 的 7 块。
    pre = run_gen("p3b_c_preempt", num_speculative_tokens=2, prompts="preempt",
                  num_kvcache_blocks=7, speculative_method="model",
                  speculative_model=MODEL_PATH, gpu_util=0.55, temperature=0.0,
                  max_tokens=48)
    pre_ref = run_gen("p3b_c_preempt_ref", num_speculative_tokens=0, prompts="preempt",
                      gpu_util=0.35, temperature=0.0, max_tokens=48, logprobs=20)
    res.append(report(f"C4 抢占 {pre['stats']['preempted']} 次仍跑完,"
                      f"输出长度 {[len(o['token_ids']) for o in pre['outputs']]}",
                      pre["stats"]["preempted"] > 0
                      and all(len(o["token_ids"]) == 48 for o in pre["outputs"])))
    res.append(check_equal_or_noise(pre_ref, pre, "C5 抢占 + 草稿模型输出等价",
                                    "normal", "preempt"))
    # 采样模式 × 两种草稿采样法。random 那条才会真的把通用接受式跑到端到端。
    for m in ("greedy", "random"):
        rnd = run_gen(f"p3b_c_{m}", num_speculative_tokens=2, temperature=0.8,
                      top_k=50, top_p=0.9, draft_sample_method=m,
                      speculative_method="model", speculative_model=MODEL_PATH,
                      gpu_util=0.55, max_tokens=48)
        st = rnd["stats"]
        lens = [len(o["token_ids"]) for o in rnd["outputs"]]
        res.append(report(f"C6 temperature>0 + draft_sample_method={m} 跑通 "
                          f"(接受 {st['spec_accepted']}/{st['spec_proposed']} = "
                          f"{st['spec_accepted'] / max(1, st['spec_proposed']):.3f}, 长度 {lens})",
                          st["spec_proposed"] > 0 and all(x == 48 for x in lens)))
    # 关掉草稿图必须仍然正确(只是慢),这是 E3 对照组的正确性前提
    ng = run_gen("p3b_c_nograph", num_speculative_tokens=2, no_draft_cudagraph=True,
                 speculative_method="model", speculative_model=MODEL_PATH,
                 gpu_util=0.55, temperature=0.0, max_tokens=48)
    with_g = run_gen("p3b_c_withgraph", num_speculative_tokens=2,
                     speculative_method="model", speculative_model=MODEL_PATH,
                     gpu_util=0.55, temperature=0.0, max_tokens=48)
    res.append(check_equal(token_ids(with_g), token_ids(ng),
                           "C7 关掉草稿图输出不变(自草稿,两侧目标 kernel 路径相同)",
                           "draft-graph", "draft-eager"))
    return res


def suite_guard():
    """D:词表不一致必须报错,不能静默算错。"""
    print("\n[D] 配对校验")
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from nanovllm import LLM\n"
        "LLM(%r, speculative_model=%r, num_speculative_tokens=2,\n"
        "    speculative_method='model', enforce_eager=True, gpu_memory_utilization=0.3)\n"
        % (ROOT, MODEL_PATH, os.path.expanduser("~/huggingface/tiny-qwen3-moe"))
    )
    r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, cwd=ROOT)
    msg = (r.stderr or "")
    ok = r.returncode != 0 and ("分词器" in msg or "词表" in msg or "tokenizer" in msg)
    line = next((l for l in msg.strip().splitlines() if "ValueError" in l), msg[-200:])
    return [report("D1 分词器不同的配对必须带诊断信息报错", ok, f"    {line.strip()[:180]}")]


# ================================================================ 逐 step 三方对拍

def suite_inprocess():
    """E:草稿图 replay vs 同形状 eager 必须逐位相同(0 ulp)。放子进程跑。"""
    print("\n[E] 逐 step 对拍(子进程)")
    r = subprocess.run([PYTHON, os.path.abspath(__file__), "--inproc"],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
    return [report("E 草稿图 replay 与 eager 逐位相同", r.returncode == 0)]


def inproc_main():
    """图 replay vs **同形状** eager,逐位比。

    07 报告坑 5 的教训:直接拿"图输出"和"真实形状的 eager 输出"比,量到的是
    【图的误差 + padding 引入的误差】之和,分不开。要证明"图本身没引入误差",
    参照必须用图刚刚刷新好的那份 **padded** 缓冲区原样再跑一遍 eager ——
    形状、cu_seqlens、slot_mapping、block_tables 全是同一份,差异只可能来自图。
    这一层的判据是逐位相同(0 ulp),不允许任何偏差。

    ulp 按**张量尺度**算,不是逐元素(坑 1:隐状态里有近零分量,逐元素算会把
    一次正常的 bf16 舍入放大成几百 ulp)。
    """
    sys.path.insert(0, ROOT)
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner
    import nanovllm.utils.context as C
    from nanovllm.utils.context import set_context
    from common import build_prompts

    stats = {"draft_varlen": [0, 0, 0.0], "draft_decode": [0, 0, 0.0],
             "target_varlen": [0, 0, 0.0], "target_decode": [0, 0, 0.0]}

    def ulp(a, b):
        d = (a.float() - b.float()).abs().max()
        scale = torch.finfo(torch.bfloat16).eps * a.float().abs().max().clamp_min(1e-6)
        return (d / scale).item()

    orig_v = ModelRunner.replay_varlen
    orig_d = ModelRunner.replay_decode

    def record(key, g, p):
        s = stats[key]
        s[0] += 1
        s[1] += int(torch.equal(g, p))
        s[2] = max(s[2], ulp(g, p))

    def patched_v(self, graphs, gv, input_ids, positions, num_seqs):
        out = orig_v(self, graphs, gv, input_ids, positions, num_seqs)
        is_draft = graphs is getattr(self, "draft_varlen_graphs", None)
        model = self.draft_model if is_draft else self.model
        g = out.clone()
        bs = next(x for x in self.graph_bs if x >= num_seqs)
        q_max = gv["q_max"]
        total, nslot = bs * q_max, 2 * bs
        saved = C._CONTEXT
        set_context(True, gv["cu_seqlens_q"][:nslot + 1], gv["cu_seqlens_k"][:nslot + 1],
                    q_max, self.config.max_model_len, gv["slot_mapping"][:total], None,
                    gv["block_tables"][:nslot], None)
        p = model(gv["input_ids"][:total], gv["positions"][:total])[:g.size(0)]
        C._CONTEXT = saved
        record("draft_varlen" if is_draft else "target_varlen", g, p)
        return out

    def patched_d(self, graphs, graph_vars, input_ids, positions):
        out = orig_d(self, graphs, graph_vars, input_ids, positions)
        is_draft = graphs is getattr(self, "draft_graphs", None)
        model = self.draft_model if is_draft else self.model
        g = out.clone()
        n = g.size(0)
        bs = next(x for x in self.graph_bs if x >= n)
        saved = C._CONTEXT
        set_context(False, slot_mapping=graph_vars["slot_mapping"][:bs],
                    context_lens=graph_vars["context_lens"][:bs],
                    block_tables=graph_vars["block_tables"][:bs])
        p = model(graph_vars["input_ids"][:bs], graph_vars["positions"][:bs])[:n]
        C._CONTEXT = saved
        record("draft_decode" if is_draft else "target_decode", g, p)
        return out

    ModelRunner.replay_varlen = patched_v
    ModelRunner.replay_decode = patched_d

    llm = LLM(MODEL_PATH, num_speculative_tokens=2, speculative_method="model",
              speculative_model=MODEL_PATH, gpu_memory_utilization=0.55,
              max_model_len=2048, max_num_seqs=16)
    sp = SamplingParams(temperature=0.0, max_tokens=40, ignore_eos=True)
    llm.generate(build_prompts(), sp, use_tqdm=False)

    ok = True
    for name, (tot, same, mx) in stats.items():
        if tot == 0:
            print(f"  [--] {name}: 本配置下没走到")
            continue
        good = same == tot
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {same}/{tot} 逐位相同, "
              f"最大偏差 {mx:.2f} ulp")
    sys.exit(0 if ok else 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Phase 3B 草稿模型:单元测试(接受规则) ===")
    unit = [test_general_rule_preserves_distribution(),
            test_general_rule_matches_simplified_when_delta(),
            test_general_rule_residual_shape(),
            test_general_rule_degenerate_q(),
            test_general_rule_greedy_unaffected()]
    print("\n=== Phase 3B 草稿模型:端到端 ===")
    results = unit + suite_equivalence() + suite_pairing() + suite_edge() \
        + suite_guard() + suite_inprocess()
    n = sum(results)
    print(f"\n=== 汇总: {n}/{len(results)} 通过 ===")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--inproc":
        inproc_main()
    sys.exit(main())
