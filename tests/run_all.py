"""一键跑完三个阶段的全部测试。"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
if not os.path.exists(PY):
    PY = sys.executable

SUITES = [
    ("Phase 1 单元 · Sampler", "test_sampler.py"),
    ("Phase 1 端到端 · 采样/logprobs/基线", "test_phase1_e2e.py"),
    ("Phase 2 · 统一调度器(混批)", "test_phase2_mixed.py"),
    ("Phase 3 · 投机解码", "test_phase3_spec.py"),
]

if __name__ == "__main__":
    summary = []
    for name, script in SUITES:
        print("=" * 70)
        print(f">>> {name}  ({script})")
        print("=" * 70, flush=True)
        r = subprocess.run([PY, os.path.join(HERE, script)], cwd=ROOT)
        summary.append((name, r.returncode == 0))
    print("\n" + "=" * 70)
    for name, ok in summary:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    n = sum(1 for _, ok in summary if ok)
    print(f"\n{n}/{len(summary)} 个测试套件通过")
    raise SystemExit(0 if n == len(summary) else 1)
