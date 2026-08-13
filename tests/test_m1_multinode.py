"""M1 · 多机运行时验收（dense Qwen3-0.6B，不涉及 MoE）。

这一步只验证"跨机复制计算"的地基：rank0 当 driver，rank1 当 worker，两边跑同一批、
TP=1 所以前向里没有任何集合通信。因此 rank0 的输出必须与**单机 world=1 基线逐 token
全等**——任何分歧都是搬运 bug，不是浮点噪声。这是本文件里唯一的硬判据。

只能在 rank0（gpu-02）上跑。
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_PATH, report                              # noqa: E402
from harness import OUT_DIR, ROOT, check_equal, run_gen, token_ids  # noqa: E402

EP_PORT = int(os.environ.get("EP_PORT", "29500"))
LAUNCH = os.path.join(ROOT, "scripts", "launch_both.sh")
DENSE = "~/huggingface/Qwen3-0.6B"


def run_gen_ep(name, extra_env=None, timeout=1200, **kwargs):
    out = os.path.join(OUT_DIR, f"{name}.json")
    gen_args = [os.path.join(ROOT, "tests", "gen.py"), "--out", out,
                "--model", DENSE, "--eager", "--ep-size", "2",
                "--master-addr", "192.168.100.2", "--master-port", str(EP_PORT)]
    for k, v in kwargs.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                gen_args.append(flag)
        elif v is not None:
            gen_args += [flag, str(v)]
    env = dict(os.environ, MODEL=os.path.expanduser(DENSE), MNBT="512",
               KVBLOCKS="-1", MASTER_PORT=str(EP_PORT), NOSYNC="1",
               EP_TIMEOUT=str(timeout), **(extra_env or {}))
    print(f"  $ launch_both.sh gen.py {' '.join(gen_args[3:])}")
    r = subprocess.run(["bash", LAUNCH] + gen_args, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return r, out


def test_1_smoke(results):
    """判据 1：组建立冒烟（nccl all_reduce + gloo broadcast），并记控制面延迟。"""
    print("\n[1] 双机进程组冒烟 + 控制面延迟")
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "run2.sh"),
                        "scripts/m0_nccl_test.py"], cwd=ROOT,
                       capture_output=True, text=True,
                       env=dict(os.environ, RUN2_TIMEOUT="300"))
    lines = [l for l in r.stdout.splitlines()
             if l.startswith("[L3") or l.strip().startswith(("1048576", "268435456"))]
    for l in lines:
        print("    " + l.strip())
    ok = r.returncode == 0 and any("all_reduce  OK" in l for l in lines)
    results.append(report("双机 nccl all_reduce + gloo broadcast", ok))


def test_2_dense_equivalence(results):
    """判据 2（硬判据）：EP=2 的 rank0 输出与单机 world=1 逐 token 全等。"""
    print("\n[2] dense 等价：EP=2 rank0 vs 单机 world=1（逐 token 全等）")
    single = run_gen("m1_single", model=DENSE, eager=True, max_tokens=128, logprobs=10,
                     gpu_util=0.4, max_num_batched_tokens=512)
    r, out = run_gen_ep("m1_ep2", max_tokens=128, logprobs=10, gpu_util=0.4,
                        max_num_batched_tokens=512)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:], file=sys.stderr)
        results.append(report("dense EP=2 vs 单机逐 token 全等", False, "EP 运行失败"))
        return
    with open(out) as f:
        ep2 = json.load(f)
    ok = check_equal(token_ids(single), token_ids(ep2),
                     "dense: 单机 world=1 vs 双机 EP=2", name_a="single", name_b="EP2")
    results.append(report("dense EP=2 vs 单机逐 token 全等", ok))


def test_3_worker_probe(results):
    """判据 3：worker 侧一致性探针。NANOVLLM_EP_CHECK=1 时每步比 logits 校验和。

    跑两遍，因为这两台机器上"位级一致"只有关掉 torch.compile 才成立：

      * TORCHDYNAMO_DISABLE=1（硬判据）：Triton 不参与，两边必须**每步位级一致**。
        这一遍才是真正在测搬运——seqs 的 pickle 广播、KV 块号、context 是否原样到达。
      * 默认（记录项，不判 PASS/FAIL）：torch.compile 会在运行时生成并自动调优 Triton
        kernel，两机驱动版本不同（595.71.05 vs 610.57.04），选出的 kernel 配置/归约顺序
        不同 → 逐层累积出 ulp 级差异。这与 EP 代码无关：把同一步 forward 分别在两台机器
        上**单机**跑（ep_size=1，完全不走 EP 代码），logprob 也差 0.176（≈2.8 ulp）；
        同样的对照实验加上 TORCHDYNAMO_DISABLE=1 就变成位级全等。
        证据见 Plan-4/08-implementation-report.md 的坑 6。
    """
    print("\n[3] worker 一致性探针（NANOVLLM_EP_CHECK=1，64 步）")
    r, _ = run_gen_ep("m1_epcheck_nodynamo",
                      extra_env={"NANOVLLM_EP_CHECK": "1", "TORCHDYNAMO_DISABLE": "1"},
                      prompts="single", max_tokens=64, gpu_util=0.4,
                      max_num_batched_tokens=512)
    hits = [l for l in (r.stdout + r.stderr).splitlines() if "[EP_CHECK]" in l]
    ok = r.returncode == 0 and not hits
    print(f"    TORCHDYNAMO_DISABLE=1：不一致的步数 {len(hits)}（硬判据，必须为 0）")
    for l in hits[:3]:
        print("    " + l)

    r2, _ = run_gen_ep("m1_epcheck_default", extra_env={"NANOVLLM_EP_CHECK": "1"},
                       prompts="single", max_tokens=64, gpu_util=0.4,
                       max_num_batched_tokens=512)
    hits2 = [l for l in (r2.stdout + r2.stderr).splitlines() if "[EP_CHECK]" in l]
    print(f"    默认（torch.compile 开）：不一致的步数 {len(hits2)}（记录项，不判定）")
    if hits2:
        print("    " + hits2[0])
    results.append(report("worker 复制计算与 rank0 位级一致（关 torch.compile）", ok))


def test_4_failure_behavior(results):
    """判据 5：kill rank1 → rank0 在超时窗口内报错退出，不无限挂。"""
    print("\n[4] 故障行为：中途 kill worker")
    env = dict(os.environ, MODEL=os.path.expanduser(DENSE), MNBT="512", KVBLOCKS="-1",
               MASTER_PORT=str(EP_PORT), NOSYNC="1", EP_TIMEOUT="150",
               NANOVLLM_GLOO_TIMEOUT="30")
    out = os.path.join(OUT_DIR, "m1_kill.json")
    args = ["bash", LAUNCH, os.path.join(ROOT, "tests", "gen.py"), "--out", out,
            "--model", DENSE, "--eager", "--ep-size", "2",
            "--master-addr", "192.168.100.2", "--master-port", str(EP_PORT),
            # 负载要足够长、kill 要落在生成中途。第一版用 6 条 × 512 token 只跑了 45s，
            # sleep 45 之后再 kill，任务其实已经跑完了（rc=0，什么都没测到）。
            # 现在 16 条短 prompt × 3072 token ≈ 3000 步 ≈ 70s，30s 时下刀正好在中间。
            "--prompts", "single", "--repeat", "16",
            "--max-tokens", "3072", "--gpu-util", "0.4",
            "--max-num-batched-tokens", "512"]
    t0 = time.time()
    p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    time.sleep(30)                      # 等模型加载完、真正开始跑
    subprocess.run(["ssh", "-o", "BatchMode=yes", "192.168.100.1",
                    "pkill -f 'nanovllm[.]entry_worker'"], capture_output=True)
    try:
        p.communicate(timeout=180)
        dt = time.time() - t0
        ok = p.returncode != 0 and dt < 175
        print(f"    rank0 在 {dt:.0f}s 后以 rc={p.returncode} 退出")
    except subprocess.TimeoutExpired:
        p.kill(); p.communicate()
        ok = False
        print("    rank0 无限挂住（超过 180s 没退出）")
    results.append(report("kill worker 后 rank0 有限时间内报错退出", ok))


def main():
    print("=" * 70)
    print(">>> M1 · 多机运行时（dense Qwen3-0.6B）")
    print("=" * 70)
    subprocess.run(["bash", os.path.join(ROOT, "scripts", "sync.sh")],
                   cwd=ROOT, capture_output=True, text=True)
    results = []
    test_1_smoke(results)
    test_2_dense_equivalence(results)
    test_3_worker_probe(results)
    test_4_failure_behavior(results)
    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} passed")
    print("（判据 4「单机 42 项回归」由 tests/run_all.py 覆盖）")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
