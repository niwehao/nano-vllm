"""M6 判据 1 · 两个 EP 后端的端到端对拍（nccl vs nvshmem/IBGDA）。

计划（07-m6 任务 1）原本写的是「**必须逐 token 全等**」，理由是 M5 已证两后端的
combine 位级一致。**实测下来这个前提过强了**，原因值得记一笔：

    两后端的差别不只是"搬运方式"，还包括 **packed_recv_x 的行序**。
    nccl 后端把各 rank 的段按 rank 升序首尾相接；IBGDA 内核用
    atomicAdd(packed_recv_count+l, n) 取 begin，段序是到达序。
    同一个 token 因此落在**不同的行下标**上。

    而 MoE 的专家 GEMM 是 `torch.bmm(recv_x[L, R*M, H], w13.T)` —— cuBLAS 会按 M 维
    分块，行下标不同就落进不同的 tile，split-k 的 fp32 归约顺序随之不同 → 1 ulp。
    combine 逻辑本身是位级一致的，它只是忠实地把这 1 ulp 传下去。

所以判据改成**证据链**，而不是放宽阈值。四条一起看才有说服力：

  1. 单步 logits（无自回归放大）：argmax 一致 + 偏差 ≤ 4 ulp   ← 硬判据
  2. IBGDA 后端**自身**跑两遍逐 token 全等                      ← 硬判据（排除随机性）
  3. greedy 128 的分歧点必须全部落在近似并列位置               ← 硬判据
  4. 短负载（64 token，放大不足）应当逐 token 全等             ← 硬判据

第 2 条最关键：如果两后端的差异来自"通信不可靠/buffer 复用错乱"，IBGDA 自身重跑就不会
稳定。它稳定 → 差异是确定性的数值差异，而不是搬运 bug。

    .venv/bin/python tests/test_m6_backends.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import report                                                       # noqa: E402
from harness import (OUT_DIR, ROOT, check_equal, check_equal_or_noise,          # noqa: E402
                     token_ids, ulp_for)

MOE = "~/huggingface/tiny-qwen3-moe"
EP_PORT = int(os.environ.get("EP_PORT", "29500"))
LAUNCH = os.path.join(ROOT, "scripts", "launch_both.sh")


def run_ep(name, transport, kvblocks=-1, **kwargs):
    out = os.path.join(OUT_DIR, f"{name}.json")
    args = [os.path.join(ROOT, "tests", "gen.py"), "--out", out, "--model", MOE, "--eager",
            "--ep-size", "2", "--ep-transport", transport,
            "--master-addr", "192.168.100.2", "--master-port", str(EP_PORT),
            "--max-num-batched-tokens", "512", "--num-kvcache-blocks", str(kvblocks)]
    for k, v in kwargs.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(flag)
        elif v is not None:
            args += [flag, str(v)]
    env = dict(os.environ, MODEL=os.path.expanduser(MOE), MNBT="512",
               KVBLOCKS=str(kvblocks), MASTER_PORT=str(EP_PORT), NOSYNC="1",
               TRANSPORT=transport, EP_TIMEOUT="900")
    print(f"  $ [{transport}] {' '.join(args[3:])}")
    r = subprocess.run(["bash", LAUNCH] + args, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"{name} 失败 (rc={r.returncode})")
    for l in r.stdout.splitlines():
        if l.startswith("wrote"):
            print(f"    {l}")
    with open(out) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print(">>> M6 · EP=2(nccl) vs EP=2(nvshmem/IBGDA) 端到端对拍")
    print("=" * 70)
    subprocess.run(["bash", os.path.join(ROOT, "scripts", "sync.sh")],
                   cwd=ROOT, capture_output=True, text=True)
    results = []
    hf = json.load(open(os.path.join(OUT_DIR, "m3_hf_step.json")))
    ulp = ulp_for(max(o["logit_absmax"] for o in hf["outputs"]))

    # ---- 判据 1：单步 logits（最锐利，没有自回归放大）----
    print(f"\n[1] 单步 logits 对拍（1 ulp = {ulp:.4f}）")
    a = run_ep("m6_step_nccl", "nccl", max_tokens=1, logprobs=10, gpu_util=0.4)
    b = run_ep("m6_step_nvshmem", "nvshmem", max_tokens=1, logprobs=10, gpu_util=0.4)
    worst, mism, ties = 0.0, [], 0
    for i, (x, y) in enumerate(zip(a["outputs"], b["outputs"])):
        ta, tb = x["logprobs"][0], y["logprobs"][0]
        da, db = dict(ta["top_logprobs"]), dict(tb["top_logprobs"])
        sh = set(da) & set(db)
        worst = max(worst, max(abs(da[t] - db[t]) for t in sh) if sh else float("inf"))
        if ta["token_id"] != tb["token_id"]:
            gap = da.get(ta["token_id"], 0) - da.get(tb["token_id"], -99)
            if gap <= 4 * ulp:
                ties += 1
                print(f"    seq[{i}]: argmax {ta['token_id']} vs {tb['token_id']}，"
                      f"参照分布里差 {gap / ulp:.1f} ulp → 并列翻面")
            else:
                mism.append(f"    seq[{i}]: argmax 差 {gap / ulp:.1f} ulp → 真分歧")
    ok = not mism and worst <= 4 * ulp
    print(f"    argmax {len(a['outputs']) - ties - len(mism)}/{len(a['outputs'])} 一致"
          f"（另 {ties} 条并列翻面）；最大偏差 {worst:.6f} = {worst / ulp:.2f} ulp（上限 4）")
    for m in mism:
        print(m)
    results.append(report("单步 logits：两后端 <= 4 ulp 且无真分歧", ok))

    # ---- 判据 2：IBGDA 后端自身可复现（排除"通信不稳定"）----
    print("\n[2] IBGDA 后端自身跑两遍（硬判据：必须逐 token 全等）")
    r1 = run_ep("m6_rep1", "nvshmem", max_tokens=128, gpu_util=0.4)
    r2 = run_ep("m6_rep2", "nvshmem", max_tokens=128, gpu_util=0.4)
    ok = check_equal(token_ids(r1), token_ids(r2), "IBGDA 后端重复运行",
                     name_a="run1", name_b="run2")
    results.append(report("IBGDA 后端自身逐 token 可复现", ok))

    # ---- 判据 3：greedy 128 的分歧必须全在近似并列位置 ----
    print("\n[3] greedy 128 token：分歧点必须全部落在近似并列位置")
    ga = run_ep("m6_nccl_greedy", "nccl", max_tokens=128, logprobs=10, gpu_util=0.4)
    gb = run_ep("m6_nvshmem_greedy", "nvshmem", max_tokens=128, logprobs=10, gpu_util=0.4)
    ok = check_equal_or_noise(ga, gb, "greedy 128: nccl vs ibgda",
                              name_a="nccl", name_b="ibgda", max_ulp=4)
    results.append(report("greedy 128 分歧全部为浮点噪声", ok))

    # ---- 判据 4：短负载（放大不足）应逐 token 全等 ----
    print("\n[4] 短负载（64 token，自回归放大不足）：应逐 token 全等")
    kw = dict(prompts="pair", max_tokens=64, gpu_util=0.4)
    x = run_ep("m6_nccl_chunked", "nccl", **kw)
    y = run_ep("m6_nvshmem_chunked", "nvshmem", **kw)
    ok = check_equal(token_ids(x), token_ids(y), "chunked: nccl vs ibgda",
                     name_a="nccl", name_b="ibgda")
    results.append(report("chunked（64 token）两后端逐 token 全等", ok))

    # ---- 判据 5：投机路径单独处理 ----
    # 投机下**不能**要求跨后端逐 token 全等，原因与判据 3 不同、也更直接：
    # 普通 decode 里 1 ulp 要靠自回归慢慢放大（实测分歧点在第 33~91 个 token）；
    # 但投机的接受判据是 "草稿 token == argmax"，1 ulp 一旦让 argmax 翻面，
    # **当场**就从"接受"变成"拒绝"，不需要任何放大 —— 实测分歧就在第 4 个 token。
    # 所以这里改验两件事：
    #   a) IBGDA + 投机自身可复现（排除通信不稳定）
    #   b) IBGDA 下开关投机等价 —— 这条已由 M4 判据 5 覆盖并通过
    print("\n[5] 投机路径：IBGDA + 投机自身可复现（跨后端逐 token 全等在此不适用，理由见注释）")
    kw = dict(prompts="pair", max_tokens=64, gpu_util=0.4,
              num_speculative_tokens=2, speculative_method="ngram")
    s1 = run_ep("m6_specrep1", "nvshmem", **kw)
    s2 = run_ep("m6_specrep2", "nvshmem", **kw)
    ok = check_equal(token_ids(s1), token_ids(s2), "IBGDA + 投机 重复运行",
                     name_a="run1", name_b="run2")
    results.append(report("IBGDA + 投机 自身逐 token 可复现", ok))

    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
