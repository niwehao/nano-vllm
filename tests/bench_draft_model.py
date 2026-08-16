"""Phase 3B 基准:草稿模型 vs n-gram vs 关投机。

目标模型 Qwen3-8B,草稿模型 Qwen3-0.6B —— 任务要求的"8B 级目标 + 0.6B 级草稿"。
n-gram 的对照必须跑在**同一个目标模型**上,否则两组数字没法比。

两种负载:
  generic —— 普通文本,n-gram 基本命中不了,是草稿模型的主场;
  repeat  —— prompt 里同一段出现两次,n-gram 的理想场景。
"谁在什么负载上赢"本身就是结论,所以两种都跑。

每个配置单独起子进程(沿用 bench_varlen_graph.py 的做法:LLMEngine 会
init_process_group 并吃掉大块显存,同进程反复创建容易互相干扰)。
"""
import json
import os
import subprocess
import sys
from time import perf_counter

import common
from common import corpus_tokens
from nanovllm import LLM, SamplingParams

HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
PY = os.path.join(ROOT, ".venv", "bin", "python")
TARGET = os.path.expanduser("~/huggingface/Qwen3-8B")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def build_workload():
    c = list(corpus_tokens())
    while len(c) < 4000:
        c = c + c
    generic = [c[100 * i: 100 * i + 300] for i in range(8)]      # 8 条并发
    seg = c[200:500]
    repeat = [seg + seg + c[500 + 20 * i: 520 + 20 * i] for i in range(8)]
    return {"generic": generic, "repeat": repeat}


def measure(k, method, draft_graph, prompts, out_tokens):
    kw = dict(enforce_eager=False, gpu_memory_utilization=0.90,
              max_model_len=2048, max_num_seqs=16, draft_cudagraph=draft_graph)
    if k:
        kw.update(num_speculative_tokens=k, speculative_method=method)
        if method == "model":
            kw["speculative_model"] = DRAFT
    llm = LLM(TARGET, **kw)

    warm = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    llm.generate([p[:64] for p in prompts[:2]], warm, use_tqdm=False)   # 付掉一次性调优开销
    llm.generate([p[:64] for p in prompts[:2]], warm, use_tqdm=False)

    sp = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)
    for p in prompts:
        llm.add_request(p, sp)
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

    ex = {kk: llm.model_runner.exec_stats[kk] - ex0[kk] for kk in ex0}
    steps = llm.scheduler.stats["steps"] - st0["steps"]
    acc = llm.scheduler.spec_accepted - a0
    prop = llm.scheduler.spec_proposed - p0
    g = sorted(gaps)

    def pct(q):
        return g[min(len(g) - 1, int(q * len(g)))] if g else 0.0

    graph_steps = ex["graph_decode"] + ex["graph_varlen"]
    draft_g = ex["draft_graph_varlen"] + ex["draft_graph_decode"]
    return {
        "k": k, "method": method, "draft_graph": draft_graph,
        "time_s": total_s, "steps": steps, "out_tokens": n_out,
        "tok_s": n_out / total_s,
        "tokens_per_step": n_out / max(1, steps),
        "tbt_p50_ms": 1000 * pct(0.50), "tbt_p99_ms": 1000 * pct(0.99),
        "accepted": acc, "proposed": prop,
        "accept_rate": acc / max(1, prop),
        "exec": ex,
        "graph_ratio": graph_steps / max(1, steps),
        "draft_graph_ratio": draft_g / max(1, draft_g + ex["draft_eager"]),
    }


CONFIGS = [
    # (标签, k, method, draft_cudagraph)
    ("spec-off + decode图",      0, "ngram", True),
    ("n-gram k=2 + varlen图",    2, "ngram", True),
    ("小模型 k=2 + 全图",         2, "model", True),
    ("小模型 k=4 + 全图",         4, "model", True),
    ("小模型 k=8 + 全图",         8, "model", True),
    ("小模型 k=2 草稿不图化",      2, "model", False),
]


def main():
    wl = build_workload()
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        idx, which = int(sys.argv[2]), sys.argv[3]
        label, k, method, dg = CONFIGS[idx]
        r = measure(k, method, dg, wl[which], 256)
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

    print(f"\n目标 Qwen3-8B / 草稿 Qwen3-0.6B / 8 条并发 / 每条 256 个输出 token\n")
    print(f"{'负载':8} {'配置':24} {'耗时s':>7} {'tok/s':>8} {'tok/步':>7} "
          f"{'接受率':>7} {'走图%':>6} {'草稿走图%':>9} {'TBTp50':>7} {'TBTp99':>7}")
    for r in rows:
        print(f"{r['workload']:8} {r['label']:24} {r['time_s']:7.2f} {r['tok_s']:8.1f} "
              f"{r['tokens_per_step']:7.2f} {r['accept_rate']:6.1%} "
              f"{100*r['graph_ratio']:5.1f}% {100*r['draft_graph_ratio']:8.1f}% "
              f"{r['tbt_p50_ms']:7.2f} {r['tbt_p99_ms']:7.2f}")

    print("\n相对基线(同负载):")
    for which in ("generic", "repeat"):
        base = next(r for r in rows if r["workload"] == which and r["k"] == 0)
        for r in rows:
            if r["workload"] != which or r["k"] == 0:
                continue
            print(f"  {which:8} {r['label']:24} {r['tok_s']/base['tok_s']:6.3f}× "
                  f"(vs spec-off {base['tok_s']:.1f} tok/s)")

    os.makedirs(os.path.join(ROOT, "tests", "out"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "tests", "out", "bench_draft_model.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
