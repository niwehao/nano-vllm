"""投机解码性能基准:接受率 与 端到端加速比。

两种负载:
  generic  —— 普通续写,n-gram 命中一般
  repeat   —— 让模型接着抄一段已经出现过的文本,n-gram 命中率高(投机的理想场景)
"""
import json
from time import perf_counter

import common
from common import MODEL_PATH, corpus_tokens
from nanovllm import LLM, SamplingParams


def measure(k, prompts, out_tokens, label):
    llm = LLM(MODEL_PATH, enforce_eager=True, gpu_memory_utilization=0.35,
              max_model_len=4096, num_speculative_tokens=k, speculative_method="ngram")
    sp = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)
    # 预热
    llm.generate([p[:64] for p in prompts[:2]], SamplingParams(temperature=0.0, max_tokens=8,
                                                              ignore_eos=True), use_tqdm=False)
    st0 = dict(llm.scheduler.stats)
    a0, p0 = llm.scheduler.spec_accepted, llm.scheduler.spec_proposed
    t = perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = perf_counter() - t
    steps = llm.scheduler.stats["steps"] - st0["steps"]
    acc = llm.scheduler.spec_accepted - a0
    prop = llm.scheduler.spec_proposed - p0
    total_out = sum(len(o["token_ids"]) for o in outs)
    return {
        "label": label, "k": k, "time_s": dt, "steps": steps,
        "out_tokens": total_out, "tok_s": total_out / dt,
        "tokens_per_step": total_out / max(1, steps),
        "proposed": prop, "accepted": acc,
        "accept_rate": acc / max(1, prop),
    }


def main():
    c = list(corpus_tokens())
    while len(c) < 4000:
        c = c + c
    # generic:普通续写
    generic = [c[100 * i: 100 * i + 300] for i in range(8)]
    # repeat:prompt 里同一段文字出现两次,模型大概率继续照抄 -> n-gram 命中率高
    seg = c[200:500]
    repeat = [seg + seg + c[500 + 20 * i: 520 + 20 * i] for i in range(8)]

    rows = []
    import subprocess, sys, os
    # 每个配置单独起进程,避免 KV cache / 进程组冲突
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        k = int(sys.argv[2]); which = sys.argv[3]
        prompts = generic if which == "generic" else repeat
        r = measure(k, prompts, 448, which)
        print("RESULT " + json.dumps(r))
        return
    here = os.path.abspath(__file__)
    py = os.path.join(os.path.dirname(os.path.dirname(here)), ".venv", "bin", "python")
    for which in ("generic", "repeat"):
        for k in (0, 1, 2, 4, 8):
            out = subprocess.run([py, here, "--child", str(k), which],
                                 capture_output=True, text=True)
            line = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
            if not line:
                print(out.stdout[-2000:]); print(out.stderr[-3000:]); raise SystemExit(1)
            rows.append(json.loads(line[0][7:]))

    base = {r["label"]: r for r in rows if r["k"] == 0}
    print(f"\n{'负载':8} {'k':>2} {'耗时s':>7} {'步数':>6} {'tok/步':>7} "
          f"{'tok/s':>8} {'接受率':>7} {'加速':>6}")
    for r in rows:
        b = base[r["label"]]
        print(f"{r['label']:8} {r['k']:2d} {r['time_s']:7.2f} {r['steps']:6d} "
              f"{r['tokens_per_step']:7.2f} {r['tok_s']:8.1f} "
              f"{r['accept_rate']:6.1%} {r['tok_s'] / b['tok_s']:5.2f}x")
    json.dump(rows, open("tests/out/bench_spec.json", "w"), indent=2)


if __name__ == "__main__":
    main()
