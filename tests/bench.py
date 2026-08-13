"""性能基准:量化统一调度(prefill/decode 混批)的效果。

对旧版(改动前)和新版都能跑:只用 add_request / step / is_finished,
并兼容 step() 的两种返回形状:
  旧版 (outputs, num_tokens)              >0 是 prefill 轮的 token 数,<0 是 decode 轮的 seq 数
  新版 (outputs, num_prefill, num_decode) 两者可同时 >0(混批)

三个场景:
  A 稳态 decode 吞吐    —— 检查统一路径有没有拖慢 decode
  B prefill 吞吐        —— 检查"prefill 改走 paged cache"的代价
  C 长 prompt 灌入      —— 主场景。用 chunked prefill 让一条长 prompt 横跨多个 step,
                           旧调度下这期间 decode 完全停摆,新调度下每步都还在出词。
"""
import json
import sys
from time import perf_counter

import common
from common import MODEL_PATH, corpus_tokens
from nanovllm import LLM, SamplingParams


def step_counts(res):
    """把 step() 的返回值归一成 (prefill_tokens, decode_tokens)。"""
    counts = res[1:]
    if len(counts) == 1:              # 旧版协议
        n = counts[0]
        return (n, 0) if n > 0 else (0, -n)
    return (counts[0], counts[1])     # 新版协议


def drive(llm, inject_at=None, inject=None):
    """跑到全部完成,返回每步的 (dt, prefill_tokens, decode_tokens)。"""
    rec = []
    i = 0
    while not llm.is_finished():
        if inject_at is not None and i == inject_at:
            for p, sp in inject:
                llm.add_request(p, sp)
            inject_at = None
        t0 = perf_counter()
        res = llm.step()
        rec.append((perf_counter() - t0,) + step_counts(res))
        i += 1
    return rec


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tests/out/bench.json"
    label = sys.argv[2] if len(sys.argv) > 2 else "new"

    c = list(corpus_tokens())
    while len(c) < 9000:
        c = c + c

    llm = LLM(MODEL_PATH, enforce_eager=True, gpu_memory_utilization=0.35,
              max_num_batched_tokens=512, max_model_len=4096)
    sp = lambda n: SamplingParams(temperature=1e-9, max_tokens=n, ignore_eos=True)
    r = {"label": label}

    # ---- 预热:把 flash-attn / cuBLAS 的首次 kernel 选择开销先付掉 ----
    for i in range(8):
        llm.add_request(c[13 * i: 13 * i + 200], sp(32))
    drive(llm)
    for i in range(2):
        llm.add_request(c[11 * i: 11 * i + 3000], sp(8))
    drive(llm)

    # ---- A 稳态 decode 吞吐 ----
    for i in range(16):
        llm.add_request(c[29 * i: 29 * i + 128], sp(128))
    rec = drive(llm)
    dec = [(dt, d) for dt, p, d in rec if d > 0 and p == 0]
    r["decode_tokens"] = sum(d for _, d in dec)
    r["decode_time_s"] = sum(dt for dt, _ in dec)
    r["decode_tok_s"] = r["decode_tokens"] / r["decode_time_s"]
    r["decode_step_ms"] = 1000 * r["decode_time_s"] / len(dec)

    # ---- B prefill 吞吐(单条 3000 token 的 prompt,无并发 decode)----
    llm.add_request(c[500:3500], sp(1))
    rec = drive(llm)
    pre = [(dt, p) for dt, p, d in rec if p > 0]
    r["prefill_tokens"] = sum(p for _, p in pre)
    r["prefill_time_s"] = sum(dt for dt, _ in pre)
    r["prefill_tok_s"] = r["prefill_tokens"] / r["prefill_time_s"]

    # ---- C 主场景:8 条在跑,中途灌入 3 条 4000-token 长 prompt ----
    # max_num_batched_tokens=512 -> 每条长 prompt 要切 8 个 chunk。
    # 旧调度:只要 waiting 里有 prefill,整轮就只做 prefill,decode 连续停摆 8+ 步。
    for i in range(8):
        llm.add_request(c[37 * i: 37 * i + 64], sp(200))
    inject = [(c[3 * i: 3 * i + 4000], sp(32)) for i in range(3)]
    t0 = perf_counter()
    rec = drive(llm, inject_at=10, inject=inject)
    r["scenario_total_s"] = perf_counter() - t0
    r["steps"] = len(rec)
    r["mixed_steps"] = sum(1 for _, p, d in rec if p > 0 and d > 0)
    r["prefill_only_steps"] = sum(1 for _, p, d in rec if p > 0 and d == 0)
    r["decode_only_steps"] = sum(1 for _, p, d in rec if p == 0 and d > 0)

    # 出词间隔:两次"产出 decode token"的 step 之间隔了多久 —— 已在跑的请求感受到的卡顿
    gaps, acc = [], 0.0
    for dt, p, d in rec:
        acc += dt
        if d > 0:
            gaps.append(acc)
            acc = 0.0
    gaps_sorted = sorted(gaps)
    pct = lambda q: gaps_sorted[min(len(gaps_sorted) - 1, int(len(gaps_sorted) * q))] if gaps_sorted else 0.0
    r["tbt_p50_ms"] = 1000 * pct(0.50)
    r["tbt_p95_ms"] = 1000 * pct(0.95)
    r["tbt_p99_ms"] = 1000 * pct(0.99)
    r["tbt_max_ms"] = 1000 * max(gaps) if gaps else 0.0
    # 最长的一段"完全没有 decode 产出"的连续时间
    stall, best = 0.0, 0.0
    for dt, p, d in rec:
        if d == 0:
            stall += dt
            best = max(best, stall)
        else:
            stall = 0.0
    r["max_decode_stall_ms"] = 1000 * best
    r["max_consecutive_prefill_only_steps"] = max(
        [len(list(g)) for k, g in __import__("itertools").groupby(
            (d == 0 for _, p, d in rec)) if k] or [0])

    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
