"""M4 · EP 集成验收：tiny-qwen3-moe 在 gpu-02(expert 0/1) + gpu-01(expert 2/3) 上端到端。

每个配置都要重起一次 worker（进程组是一次性的），所以统一走 scripts/launch_both.sh。
本文件只能在 rank0（gpu-02）上跑。
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                       # noqa: E402
from common import report                                           # noqa: E402
from harness import (OUT_DIR, PYTHON, ROOT, check_equal_or_noise,   # noqa: E402
                     compare_logprobs, run_gen, token_ids, ulp_for)

MOE = "~/huggingface/tiny-qwen3-moe"
EP_PORT = int(os.environ.get("EP_PORT", "29500"))
# M6 切后端时只改这一个环境变量：EP_TRANSPORT=nvshmem .venv/bin/python tests/test_m4_ep2.py
EP_TRANSPORT = os.environ.get("EP_TRANSPORT", "nccl")
LAUNCH = os.path.join(ROOT, "scripts", "launch_both.sh")
GPU01 = "192.168.100.1"


def run_gen_ep(name, mnbt=512, kvblocks=-1, **kwargs):
    """双机跑一次 gen.py。worker 的 max_num_batched_tokens / num_kvcache_blocks
    必须与 driver 一致，否则 EP buffer 的 M 或 KV 块数对不上。"""
    out = os.path.join(OUT_DIR, f"{name}.json")
    gen_args = [os.path.join(ROOT, "tests", "gen.py"), "--out", out,
                "--model", MOE, "--eager", "--ep-size", "2",
                "--master-addr", "192.168.100.2", "--master-port", str(EP_PORT),
                "--max-num-batched-tokens", str(mnbt),
                "--num-kvcache-blocks", str(kvblocks),
                "--ep-transport", EP_TRANSPORT]
    for k, v in kwargs.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                gen_args.append(flag)
        elif v is not None:
            gen_args += [flag, str(v)]
    env = dict(os.environ, MODEL=os.path.expanduser(MOE), MNBT=str(mnbt),
               KVBLOCKS=str(kvblocks), MASTER_PORT=str(EP_PORT), NOSYNC="1",
               TRANSPORT=EP_TRANSPORT)
    print(f"  $ launch_both.sh gen.py {' '.join(gen_args[3:])}")
    r = subprocess.run(["bash", LAUNCH] + gen_args, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-5000:]); print(r.stderr[-6000:], file=sys.stderr)
        raise RuntimeError(f"EP run failed for {name} (rc={r.returncode})")
    for l in r.stdout.splitlines():
        if l.startswith("wrote") or "spec stats" in l:
            print(f"    {l}")
    with open(out) as f:
        return json.load(f)


def rdma_counters():
    """两机 RoCE 口的发送字节数 + bond0 的发送字节数。
    port_xmit_data 的单位是 4 字节（IB 规范的 lane word），要 ×4 才是字节。"""
    def local():
        with open("/sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_data") as f:
            ib = int(f.read()) * 4
        with open("/sys/class/net/bond0/statistics/tx_bytes") as f:
            return ib, int(f.read())
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", GPU01,
                        "cat /sys/class/infiniband/rocep66s0f0/ports/1/counters/port_xmit_data "
                        "/sys/class/net/bond0/statistics/tx_bytes"],
                       capture_output=True, text=True)
    a, b = r.stdout.split()
    return local() + (int(a) * 4, int(b))


def test_1_2_single_step_and_greedy(results):
    print("\n[1] 单步 logits：EP=2 vs EP=1")
    ep1 = run_gen("m4_ep1_step", model=MOE, eager=True, max_tokens=1, logprobs=10,
                  gpu_util=0.4, max_num_batched_tokens=512)
    ep2 = run_gen_ep("m4_ep2_step", max_tokens=1, logprobs=10, gpu_util=0.4)
    # ulp 按 tiny-MoE 的 logits 量级（~4）现算，不用 dense 的 0.0625
    hf = json.load(open(os.path.join(OUT_DIR, "m3_hf_step.json")))
    ulp = ulp_for(max(o["logit_absmax"] for o in hf["outputs"]))
    ok = _cmp_logprobs(ep1, ep2, "EP=1 vs EP=2 单步", ulp)
    results.append(report("单步 logits EP=2 vs EP=1", ok))

    print("\n[2] greedy 128 token：EP=2 vs EP=1 基线")
    base = json.load(open(os.path.join(common.BASELINE_DIR, "greedy_moe_ep1.json")))
    ep2g = run_gen_ep("m4_ep2_greedy", max_tokens=128, logprobs=10, gpu_util=0.4)
    ok = check_equal_or_noise(base, ep2g, "greedy 128: EP=1 基线 vs EP=2",
                              name_a="EP1", name_b="EP2", max_ulp=4)
    results.append(report("greedy 128 token EP=2 vs EP=1 基线", ok))
    return ep2g


def _cmp_logprobs(pa, pb, title, ulp, max_ulp=4):
    """与 harness.compare_logprobs 同一套判据，只是 ulp 按本模型量级现算。

    argmax 不同不直接判失败，先看这两个候选在参照分布里差多少：tiny-MoE 是随机权重，
    top1/top2 精确并列很常见（M3 的判据 1 里 6 条 prompt 就有 2 条 gap 恰好为 0），
    并列时谁赢取决于归约顺序，是噪声不是 bug。差距超过阈值才算真分歧。
    """
    tol, worst, bad, noise, mism = max_ulp * ulp, 0.0, [], 0, 0
    for i, (oa, ob) in enumerate(zip(pa["outputs"], pb["outputs"])):
        la, lb = oa["logprobs"][0], ob["logprobs"][0]
        da, db = dict(la["top_logprobs"]), dict(lb["top_logprobs"])
        if la["token_id"] != lb["token_id"]:
            mism += 1
            ga, gb = da.get(la["token_id"]), da.get(lb["token_id"])
            if ga is None or gb is None:
                bad.append(f"    seq[{i}]: argmax {la['token_id']} vs {lb['token_id']}"
                           f" —— 对方选的 token 不在参照 top-10 里 → 真分歧")
            elif ga - gb <= tol:
                noise += 1
                print(f"    seq[{i}]: argmax {la['token_id']} vs {lb['token_id']}，"
                      f"参照分布里差 {ga - gb:.4f} = {(ga - gb) / ulp:.1f} ulp → 并列，数值噪声")
            else:
                bad.append(f"    seq[{i}]: argmax {la['token_id']} vs {lb['token_id']}，"
                           f"差 {ga - gb:.4f} = {(ga - gb) / ulp:.1f} ulp > {max_ulp} → 真分歧")
        sh = set(da) & set(db)
        if sh:
            worst = max(worst, max(abs(da[t] - db[t]) for t in sh))
    ok = not bad and worst <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {title}: argmax "
          f"{len(pa['outputs']) - mism}/{len(pa['outputs'])} 一致（另 {noise} 条为并列翻面），"
          f"logprob 最大偏差 {worst:.5f} = {worst / ulp:.1f} ulp（上限 {max_ulp}）")
    for l in bad:
        print(l)
    return ok


def test_3_chunked_prefill(results):
    """判据 3：chunked prefill + EP。max_num_batched_tokens=512，长 prompt 被切成多块，
    每块都要过 MoE（路由与 lm_head 的 logits_indices 无关，中间块一样要 dispatch）。"""
    print("\n[3] chunked prefill + EP（600 token prompt / 512 预算 → 切 2 块）")
    ep1 = run_gen("m4_ep1_chunk", model=MOE, eager=True, prompts="pair", max_tokens=64,
                  logprobs=10, gpu_util=0.4, max_num_batched_tokens=512)
    ep2 = run_gen_ep("m4_ep2_chunk", prompts="pair", max_tokens=64, logprobs=10, gpu_util=0.4)
    print(f"    EP=2 调度统计: {ep2['stats']}")
    ok = check_equal_or_noise(ep1, ep2, "chunked prefill: EP=1 vs EP=2",
                              name_a="EP1", name_b="EP2", max_ulp=4)
    results.append(report("chunked prefill + EP 输出一致", ok))


def test_4_preemption(results):
    """判据 4：抢占 + EP。沿用 Plan-2 坑 8 的构造法：6 条 255-token prompt + KV 块卡 7。"""
    print("\n[4] 抢占 + EP")
    ep1 = run_gen("m4_ep1_preempt", model=MOE, eager=True, prompts="preempt", max_tokens=32,
                  logprobs=10, gpu_util=0.4, max_num_batched_tokens=512, num_kvcache_blocks=7)
    ep2 = run_gen_ep("m4_ep2_preempt", kvblocks=7, prompts="preempt", max_tokens=32,
                     logprobs=10, gpu_util=0.4)
    n1, n2 = ep1["stats"].get("preempted", 0), ep2["stats"].get("preempted", 0)
    print(f"    抢占次数 EP=1: {n1}, EP=2: {n2}")
    ok = n2 > 0 and check_equal_or_noise(ep1, ep2, "抢占: EP=1 vs EP=2",
                                         name_a="EP1", name_b="EP2", max_ulp=4)
    if n2 == 0:
        print("    [FAIL] EP=2 下没触发抢占，这条判据没测到东西")
    results.append(report("抢占 + EP 输出一致且确实发生抢占", ok))


def test_5_speculative(results):
    """判据 5：投机 + EP。验证前向的 k+1 行同样要走路由，形状护栏也要顶得住。"""
    print("\n[5] 投机解码 + EP（ngram k=2，greedy 下开关投机）")
    off = run_gen_ep("m4_ep2_spec_off", prompts="pair", max_tokens=64, logprobs=10, gpu_util=0.4)
    on = run_gen_ep("m4_ep2_spec_on", prompts="pair", max_tokens=64, logprobs=10, gpu_util=0.4,
                    num_speculative_tokens=2, speculative_method="ngram")
    print(f"    投机统计: 提出 {on['stats'].get('spec_proposed')}, "
          f"接受 {on['stats'].get('spec_accepted')}, 投机步数 {on['stats'].get('spec_steps')}")
    ok = check_equal_or_noise(off, on, "EP=2 下开关投机",
                              name_a="spec-off", name_b="spec-on", max_ulp=4)
    results.append(report("投机 + EP 输出一致", ok))


def test_6_gpudirect(results):
    """判据 6：压测期间 RoCE 口计数增长，bond0 不增长 → 数据确实走直连口的 RDMA。"""
    print("\n[6] GPUDirect / RoCE 流量证据")
    before = rdma_counters()
    run_gen_ep("m4_ep2_traffic", prompts="pair", max_tokens=64, gpu_util=0.4)
    after = rdma_counters()
    d = [a - b for a, b in zip(after, before)]
    names = ["gpu-02 RoCE tx", "gpu-02 bond0 tx", "gpu-01 RoCE tx", "gpu-01 bond0 tx"]
    for n, v in zip(names, d):
        print(f"    {n:<16} +{v / 1e6:10.2f} MB")

    # 从第一性原理估通信量，而不是拍一个绝对门槛（计划的原话是"与计数器对得上数量级即过"）：
    #   本次负载 = 2 条 600-token prompt + 64 步 decode（每步 2 个 token）
    #   每个 token 有 K=2 个副本，其中约一半的目的专家在对端 → 过网
    #   每个副本 H*2B = 4KB；dispatch 发一次、combine 原路返回一次 → ×2
    #   ×4 层
    H, K, L = 2048, 2, 4
    tokens = 600 * 2 + 64 * 2
    est = tokens * K * 0.5 * H * 2 * 2 * L
    lo, hi = 0.5 * est, 3.0 * est
    ratio02 = d[0] / max(d[1], 1)
    ratio01 = d[2] / max(d[3], 1)
    ok = (lo < d[0] < hi) and (lo < d[2] < hi) and ratio02 > 20 and ratio01 > 20
    print(f"    估算 {est/1e6:.1f} MB（{tokens} token × K=2 × 半数过网 × 4KB × 收发 2 次 × 4 层）")
    print(f"    判据：两机 RoCE 增量落在估算的 0.5~3 倍内，且 RoCE/bond0 > 20")
    print(f"    实测 RoCE/bond0 比值：gpu-02 {ratio02:.0f}×，gpu-01 {ratio01:.0f}×")
    results.append(report("流量走 RoCE 直连口而非 bond0，量级与估算相符", ok))


def main():
    print("=" * 70)
    print(f">>> M4 · EP 集成（双机 4 专家，transport={EP_TRANSPORT}）")
    print("=" * 70)
    subprocess.run(["bash", os.path.join(ROOT, "scripts", "sync.sh")],
                   cwd=ROOT, capture_output=True, text=True)
    results = []
    test_1_2_single_step_and_greedy(results)
    test_3_chunked_prefill(results)
    test_4_preemption(results)
    test_5_speculative(results)
    test_6_gpudirect(results)
    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
