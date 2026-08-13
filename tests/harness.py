"""跑 gen.py 的子进程封装 + 结果比对。"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYTHON = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable
OUT_DIR = os.path.join(HERE, "out")


def run_gen(name: str, **kwargs) -> dict:
    """跑一次 gen.py,返回解析后的 JSON。kwargs 用下划线,自动转成 --xx-yy。"""
    out = os.path.join(OUT_DIR, f"{name}.json")
    cmd = [PYTHON, os.path.join(HERE, "gen.py"), "--out", out]
    for k, v in kwargs.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif v is not None:
            cmd += [flag, str(v)]
    print(f"  $ {' '.join(cmd[2:])}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-8000:], file=sys.stderr)
        raise RuntimeError(f"gen.py failed for {name} (rc={r.returncode})")
    tail = [l for l in r.stdout.strip().splitlines() if l.startswith("wrote") or "spec stats" in l]
    for l in tail:
        print(f"    {l}")
    with open(out) as f:
        return json.load(f)


def token_ids(payload) -> list[list[int]]:
    return [o["token_ids"] for o in payload["outputs"]]


def diff_report(a, b, name_a="A", name_b="B", max_show=3):
    """返回 (相同条数, 总条数, 报告行列表)"""
    lines = []
    same = 0
    for i, (xa, xb) in enumerate(zip(a, b)):
        if xa == xb:
            same += 1
            continue
        n = min(len(xa), len(xb))
        pos = next((j for j in range(n) if xa[j] != xb[j]), n)
        if len(lines) < max_show:
            ctx_a = xa[max(0, pos - 3):pos + 3]
            ctx_b = xb[max(0, pos - 3):pos + 3]
            lines.append(
                f"    seq[{i}] 首个分歧 @ {pos}/{len(xa)}: "
                f"{name_a}={xa[pos] if pos < len(xa) else '-'} {name_b}={xb[pos] if pos < len(xb) else '-'}\n"
                f"      {name_a} ctx: {ctx_a}\n      {name_b} ctx: {ctx_b}"
            )
    return same, len(a), lines


def gap_distribution(payload):
    """把参照运行里每个位置的 top1-top2 logprob 差收集起来,用来给分歧定位分位数。"""
    gaps = []
    for o in payload["outputs"]:
        for it in (o.get("logprobs") or []):
            t = it["top_logprobs"]
            if len(t) >= 2:
                gaps.append(t[0][1] - t[1][1])
    return sorted(gaps)


def check_equal_or_noise(pa, pb, title, name_a="A", name_b="B", max_ulp=4):
    """比对两个 gen.py 结果;分歧时用 logprobs 判断是不是数值噪声。

    判定依据:在首个分歧位置,如果 A 选中的 token 和 B 选中的 token 在 A 的 top-k 里
    logprob 差距很小,说明这两个候选本来就几乎并列,argmax 谁赢取决于 kernel 的
    归约顺序,属于浮点噪声;差距大则是真 bug。

    阈值取 4 个 bf16 ulp(0.25),依据是实测数据而不是拍脑袋:
      - 已知数学等价的变换(prefill 分块、prefix cache 命中)实测单步扰动 2.0~2.7 ulp;
      - 在本测试集上统计 768 个位置的 top1-top2 差,中位数是 3.5(56 ulp),
        只有 8.6% 的位置差距 <= 4 ulp;
      - 而实际观测到的每一个分歧点差距都 <= 4 ulp,全落在这条退化尾巴里。
    逻辑 bug 会在随机位置发作,不可能只挑最并列的那几个百分点出现。
    所以除了阈值,这里还会打印分歧点在整体分布里的分位数,便于人工复核。
    """
    gap_thresh = max_ulp * BF16_ULP
    dist = gap_distribution(pa)
    def pctile(g):
        if not dist:
            return float("nan")
        return 100.0 * sum(1 for x in dist if x <= g) / len(dist)

    a, b = token_ids(pa), token_ids(pb)
    same = 0
    noise = 0
    real_bugs = []
    for i, (xa, xb) in enumerate(zip(a, b)):
        if xa == xb:
            same += 1
            continue
        n = min(len(xa), len(xb))
        pos = next((j for j in range(n) if xa[j] != xb[j]), n)
        lp = pa["outputs"][i].get("logprobs")
        if not lp or pos >= len(lp):
            real_bugs.append(f"    seq[{i}] @ {pos}: 无 logprobs,无法判定")
            continue
        top = dict(lp[pos]["top_logprobs"])
        la = top.get(xa[pos])
        lb = top.get(xb[pos])
        if la is None or lb is None:
            real_bugs.append(
                f"    seq[{i}] @ {pos}: {name_b} 选的 token {xb[pos]} 不在 {name_a} 的 top-{len(top)} 里 → 真分歧")
            continue
        gap = la - lb
        msg = (f"    seq[{i}] @ {pos}: {name_a}={xa[pos]} {name_b}={xb[pos]}, "
               f"差 {gap:.4f} = {gap / BF16_ULP:.1f} ulp (处于全部位置最并列的 {pctile(gap):.1f}%)")
        if gap <= gap_thresh:
            noise += 1
            print(msg + " → 数值噪声")
        else:
            real_bugs.append(msg + f" > {max_ulp} ulp → 真分歧")
    ok = not real_bugs
    print(f"  [{'PASS' if ok else 'FAIL'}] {title}: {same}/{len(a)} 完全一致, "
          f"{noise} 条为浮点噪声, {len(real_bugs)} 条为真分歧")
    for l in real_bugs:
        print(l)
    return ok


# lm_head 输出是 bf16。词表 logits 的量级通常在 8~32 之间,该区间一个 ulp 是
# 2^-8 * 16 = 0.0625。28 层累积下来,几个 ulp 的偏差属于正常的归约顺序差异。
BF16_ULP = 0.0625


def compare_logprobs(pa, pb, title, max_ulp=4, step=0):
    """比对两次运行在同一位置的 logprob 分布。

    比"跑 64 步再看 token"锐利得多:token 比对会把一次 bf16 级别的扰动放大成整段文本
    不同,而单步 logprob 直接反映这一步的 logits 本身差了多少,没有自回归放大。

    prefill 分块、prefix cache 命中、batch 组成变化,数学上都不该改变 logits,
    所以这里的判据是:
      1) 首选 token 必须完全一致(硬判据,逻辑 bug 一定会在这里露出来);
      2) 共同候选上的 logprob 偏差必须落在几个 bf16 ulp 之内。
    """
    tol = max_ulp * BF16_ULP
    max_dev = 0.0
    bad = []
    top1_mismatch = []
    for i, (oa, ob) in enumerate(zip(pa["outputs"], pb["outputs"])):
        la, lb = oa["logprobs"], ob["logprobs"]
        if not la or not lb or step >= len(la) or step >= len(lb):
            bad.append(f"    seq[{i}]: 缺少第 {step} 步的 logprobs")
            continue
        if la[step]["token_id"] != lb[step]["token_id"]:
            top1_mismatch.append(
                f"    seq[{i}]: 首选 token 不同 {la[step]['token_id']} vs {lb[step]['token_id']}")
        da = dict(la[step]["top_logprobs"])
        db = dict(lb[step]["top_logprobs"])
        shared = set(da) & set(db)
        if not shared:
            bad.append(f"    seq[{i}]: top 候选集合完全不相交 → 真分歧")
            continue
        dev = max(abs(da[t] - db[t]) for t in shared)
        max_dev = max(max_dev, dev)
        jaccard = len(shared) / len(set(da) | set(db))
        if dev > tol or jaccard < 0.6:
            bad.append(f"    seq[{i}]: logprob 偏差 {dev:.4f} = {dev / BF16_ULP:.1f} ulp "
                       f"(上限 {max_ulp} ulp), 候选集合重合度 {jaccard:.2f}")
    ok = not bad and not top1_mismatch
    print(f"  [{'PASS' if ok else 'FAIL'}] {title}: 第 {step} 步 argmax "
          f"{len(pa['outputs']) - len(top1_mismatch)}/{len(pa['outputs'])} 一致, "
          f"logprob 最大偏差 {max_dev:.4f} = {max_dev / BF16_ULP:.1f} ulp (上限 {max_ulp})")
    for l in top1_mismatch + bad:
        print(l)
    return ok


def check_equal(a, b, title, name_a="A", name_b="B", strict=True):
    same, total, lines = diff_report(a, b, name_a, name_b)
    ok = (same == total)
    print(f"  [{'PASS' if ok else ('FAIL' if strict else 'WARN')}] {title}: {same}/{total} 条逐 token 一致")
    for l in lines:
        print(l)
    return ok
