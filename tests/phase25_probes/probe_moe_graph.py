"""MoE + CUDA graph 的兼容性检查(回答"Step B 有没有影响 MoE")。

分四档,每档单独一个子进程:
  1. MoE + eager + spec=0        —— 基线
  2. MoE + graph + spec=0        —— **不涉及本次改动**(spec=0 时 varlen 图一张都不录),
                                     用来判断"MoE 能不能进 CUDA graph"本来是什么状态
  3. MoE + graph + spec=2        —— 本次新增的路径
  4. MoE + graph + spec=2 + 关掉 varlen 图 —— Step A 行为

第 2 档的结果决定怎么归因:它要是本来就不行,那第 3 档不行就与本次改动无关。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tests"))
PY = os.path.join(ROOT, ".venv", "bin", "python")
MOE = os.path.expanduser("~/huggingface/tiny-qwen3-moe")

CASES = [
    ("1 eager  spec=0", dict(eager=True,  k=0, varlen=True)),
    ("2 eager  spec=2", dict(eager=True,  k=2, varlen=True)),
    ("3 eager  spec=2 varlen=off", dict(eager=True, k=2, varlen=False)),
    ("4 graph  spec=0", dict(eager=False, k=0, varlen=True)),
    ("5 graph  spec=2", dict(eager=False, k=2, varlen=True)),
]


def child(idx):
    sys.path.insert(0, ROOT)
    from nanovllm import LLM, SamplingParams
    _, cfg = CASES[idx]
    kw = dict(enforce_eager=cfg["eager"], gpu_memory_utilization=0.35,
              max_model_len=1024, max_num_seqs=16, varlen_cudagraph=cfg["varlen"])
    if cfg["k"]:
        kw.update(num_speculative_tokens=cfg["k"], speculative_method="ngram")
    llm = LLM(MOE, **kw)
    prompts = [list(range(100, 100 + 40 + 7 * i)) for i in range(4)]
    sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    print("RESULT " + json.dumps({
        "tokens": [o["token_ids"] for o in outs],
        "exec": dict(llm.model_runner.exec_stats),
    }))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        child(int(sys.argv[2]))
        sys.exit(0)

    results = {}
    for i, (label, _) in enumerate(CASES):
        r = subprocess.run([PY, os.path.abspath(__file__), "--child", str(i)],
                           capture_output=True, text=True, cwd=ROOT)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            err = (r.stderr or r.stdout).strip().splitlines()
            msg = next((l for l in reversed(err) if l.strip()), "?")
            print(f"[{label}]  ✗ 失败 (rc={r.returncode})")
            print(f"    {msg[:220]}")
            results[label] = None
            continue
        d = json.loads(line[0][7:])
        results[label] = d
        print(f"[{label}]  ✓  exec={d['exec']}")

    base = results.get("1 eager  spec=0")
    print()
    if base:
        for label, d in results.items():
            if d is None or label.startswith("1 "):
                continue
            same = sum(1 for a, b in zip(base["tokens"], d["tokens"]) if a == b)
            print(f"  {label} vs eager 基线: {same}/{len(base['tokens'])} 条逐 token 一致")
    print("\n归因:第 4 档(graph+spec=0)与本次改动无关 —— spec=0 时 varlen 图一张都不录。\n它失败就说明 MoE 本来就进不了 CUDA graph(forward_local 里 torch.where 的形状依赖数据,\n且 tok.numel()==0 是拿设备数据做 host 分支),与 Step B 无关。")
