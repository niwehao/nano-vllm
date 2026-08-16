"""Phase 2.5 Step B 基准:varlen 图(Step B) vs 投机批走 eager(Step A)。

要回答的是硬性要求 2:padding 行带来的额外前向开销是多少,净收益是正还是负。

每个配置单独起子进程(LLMEngine 会 init_process_group 并吃掉大块显存,
同进程反复创建容易互相干扰 —— 沿用 bench_spec.py 的做法)。
"""
import json
import os
import statistics
import subprocess
import sys
from time import perf_counter

import common
from common import MODEL_PATH, corpus_tokens
from nanovllm import LLM, SamplingParams

HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, ".venv", "bin", "python")


def build_workload():
    c = list(corpus_tokens())
    while len(c) < 4000:
        c = c + c
    generic = [c[100 * i: 100 * i + 300] for i in range(8)]      # 8 条并发
    seg = c[200:500]
    repeat = [seg + seg + c[500 + 20 * i: 520 + 20 * i] for i in range(8)]
    return {"generic": generic, "repeat": repeat}


def measure(k, eager, varlen, prompts, out_tokens):
    llm = LLM(MODEL_PATH, enforce_eager=eager, gpu_memory_utilization=0.35,
              max_model_len=4096, num_speculative_tokens=k,
              speculative_method="ngram", varlen_cudagraph=varlen)

    # padding 统计:不改生产代码,在这里挂一层记账
    pad = {"real_rows": 0, "buf_rows": 0, "steps": 0}
    import nanovllm.engine.model_runner as mr
    orig = mr.ModelRunner.run_varlen_graph

    def patched(self, input_ids, positions, num_seqs):
        q_max = 1 + self.config.num_speculative_tokens
        bs = next(x for x in self.graph_bs if x >= num_seqs)
        pad["real_rows"] += input_ids.size(0)
        pad["buf_rows"] += bs * q_max
        pad["steps"] += 1
        return orig(self, input_ids, positions, num_seqs)
    mr.ModelRunner.run_varlen_graph = patched

    warm = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    llm.generate([p[:64] for p in prompts[:2]], warm, use_tqdm=False)   # 付掉一次性调优开销
    llm.generate([p[:64] for p in prompts[:2]], warm, use_tqdm=False)

    sp = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)
    for p in prompts:
        llm.add_request(p, sp)
    for d in (pad,):
        d.update(real_rows=0, buf_rows=0, steps=0)
    st0 = dict(llm.scheduler.stats)
    ex0 = dict(llm.model_runner.exec_stats)
    a0, p0 = llm.scheduler.spec_accepted, llm.scheduler.spec_proposed

    gaps, n_out = [], 0
    t0 = perf_counter()
    while not llm.is_finished():
        t = perf_counter()
        outs, npre, ndec = llm.step()
        dt = perf_counter() - t
        if ndec:                       # 只统计产出 token 的 step 的 TBT
            gaps.append(dt)
        n_out += sum(len(s.completion_token_ids) for _, s in outs)
    total_s = perf_counter() - t0
    mr.ModelRunner.run_varlen_graph = orig

    ex = {kk: llm.model_runner.exec_stats[kk] - ex0[kk] for kk in ex0}
    steps = llm.scheduler.stats["steps"] - st0["steps"]
    acc = llm.scheduler.spec_accepted - a0
    prop = llm.scheduler.spec_proposed - p0
    g = sorted(gaps)

    def pct(q):
        return g[min(len(g) - 1, int(q * len(g)))] if g else 0.0

    graph_steps = ex["graph_decode"] + ex["graph_varlen"]
    return {
        "k": k, "eager": eager, "varlen": varlen,
        "time_s": total_s, "steps": steps, "out_tokens": n_out,
        "tok_s": n_out / total_s,
        "tokens_per_step": n_out / max(1, steps),
        "tbt_p50_ms": 1000 * pct(0.50), "tbt_p99_ms": 1000 * pct(0.99),
        "tbt_max_ms": 1000 * max(g) if g else 0.0,
        "accepted": acc, "proposed": prop,
        "accept_rate": acc / max(1, prop),
        "exec": ex,
        "graph_ratio": graph_steps / max(1, steps),
        "pad_real_rows": pad["real_rows"], "pad_buf_rows": pad["buf_rows"],
        "pad_ratio": (pad["buf_rows"] - pad["real_rows"]) / max(1, pad["buf_rows"]),
    }


CONFIGS = [
    # (标签, k, enforce_eager, varlen_cudagraph)
    ("spec-off + decode图",   0, False, True),
    ("k=2 StepA(投机走eager)", 2, False, False),
    ("k=2 StepB(varlen图)",   2, False, True),
    ("k=2 全 eager",           2, True,  False),
]


def main():
    wl = build_workload()
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        idx, which = int(sys.argv[2]), sys.argv[3]
        label, k, eager, varlen = CONFIGS[idx]
        r = measure(k, eager, varlen, wl[which], 448)
        r["label"] = label
        r["workload"] = which
        print("RESULT " + json.dumps(r))
        return

    rows = []
    for which in ("generic", "repeat"):
        for i in range(len(CONFIGS)):
            out = subprocess.run([PY, HERE, "--child", str(i), which],
                                 capture_output=True, text=True, cwd=ROOT)
            line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
            if not line:
                print(out.stdout[-3000:]); print(out.stderr[-4000:]); raise SystemExit(1)
            rows.append(json.loads(line[0][7:]))
            print(f"  done {which} / {CONFIGS[i][0]}")

    print(f"\n{'负载':8} {'配置':24} {'耗时s':>7} {'tok/s':>8} {'tok/步':>7} "
          f"{'TBTp50':>7} {'TBTp99':>7} {'走图%':>6} {'padding%':>8} {'接受率':>7}")
    for r in rows:
        print(f"{r['workload']:8} {r['label']:24} {r['time_s']:7.2f} {r['tok_s']:8.1f} "
              f"{r['tokens_per_step']:7.2f} {r['tbt_p50_ms']:7.2f} {r['tbt_p99_ms']:7.2f} "
              f"{100*r['graph_ratio']:5.1f}% {100*r['pad_ratio']:7.1f}% "
              f"{r['accept_rate']:6.1%}")

    print("\n净收益(Step B vs Step A,同负载):")
    for which in ("generic", "repeat"):
        a = next(r for r in rows if r["workload"] == which and "StepA" in r["label"])
        b = next(r for r in rows if r["workload"] == which and "StepB" in r["label"])
        print(f"  {which}: tok/s {a['tok_s']:.1f} -> {b['tok_s']:.1f} "
              f"({b['tok_s']/a['tok_s']:.3f}×)   "
              f"TBT p50 {a['tbt_p50_ms']:.2f} -> {b['tbt_p50_ms']:.2f} ms   "
              f"p99 {a['tbt_p99_ms']:.2f} -> {b['tbt_p99_ms']:.2f} ms")

    os.makedirs(os.path.join(ROOT, "tests", "out"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "tests", "out", "bench_varlen_graph.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
