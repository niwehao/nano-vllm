"""M3 · MoE 模型层（EP=1 本地版）验收。

主判据是与 HF transformers 的 Qwen3MoeForCausalLM 逐位对拍。HF 侧必须
attn_implementation="eager" + bf16，否则 sdpa/flash 的数值路径差异会污染归因。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import report                                            # noqa: E402
from harness import (OUT_DIR, PYTHON, ROOT, check_equal_or_noise,    # noqa: E402
                     run_gen, token_ids, ulp_for)

MOE = "~/huggingface/tiny-qwen3-moe"
BASELINE = os.path.join(common.BASELINE_DIR, "greedy_moe_ep1.json")


def run_hf(name, **kwargs):
    import subprocess
    out = os.path.join(OUT_DIR, f"{name}.json")
    cmd = [PYTHON, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_ref.py"),
           "--out", out, "--model", MOE]
    for k, v in kwargs.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        elif v is not None:
            cmd += [flag, str(v)]
    print(f"  $ hf_ref.py {' '.join(cmd[4:])}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-6000:], file=sys.stderr)
        raise RuntimeError(f"hf_ref.py failed for {name}")
    print("    " + [l for l in r.stdout.strip().splitlines() if l.startswith("wrote")][0])
    with open(out) as f:
        return json.load(f)


def test_1_single_step_vs_hf(results):
    """判据 1（主判据）：8 条 prompt 的单步 logits，argmax 全一致 + top-10 logprob 偏差小。

    比"跑 128 步再看 token"锐利得多——单步没有自回归放大，直接反映这一层前向本身
    差了多少。ulp 按本模型 logits 的实际量级现算（tiny-MoE 是随机权重，|logit|~4，
    ulp=0.0156；dense Qwen3 的 0.0625 在这里松了 4 倍）。
    """
    print("\n[1] 单步 logits vs HF（主判据）")
    hf = run_hf("m3_hf_step", logprobs=10)
    na = run_gen("m3_nano_step", model=MOE, eager=True, max_tokens=1, logprobs=10,
                 gpu_util=0.4, max_num_batched_tokens=4096)
    absmax = max(o["logit_absmax"] for o in hf["outputs"])
    ulp = ulp_for(absmax)
    n = len(hf["outputs"])
    mismatch, devs = [], []
    for i, (a, b) in enumerate(zip(hf["outputs"], na["outputs"])):
        ta, tb = a["logprobs"][0], b["logprobs"][0]
        if ta["token_id"] != tb["token_id"]:
            mismatch.append(f"    seq[{i}]: argmax HF={ta['token_id']} nano={tb['token_id']}")
        da, db = dict(ta["top_logprobs"]), dict(tb["top_logprobs"])
        shared = set(da) & set(db)
        devs.append(max(abs(da[t] - db[t]) for t in shared) if shared else float("inf"))
    worst = max(devs)
    ok = not mismatch and worst <= 4 * ulp
    print(f"    HF logits |max|={absmax:.2f} → 1 ulp = {ulp:.4f}")
    print(f"    argmax {n - len(mismatch)}/{n} 一致；top-10 logprob 最大偏差 "
          f"{worst:.5f} = {worst / ulp:.1f} ulp（上限 4）")
    for m in mismatch:
        print(m)
    results.append(report("单步 logits vs HF：argmax 全一致 + <=4 ulp", ok))


def test_2_greedy_vs_hf_and_baseline(results):
    """判据 2：greedy 128 token。与 HF 比 + 存成 EP=2 的对拍基线。"""
    print("\n[2] greedy 128 token vs HF，并存 EP 基线")
    hf = run_hf("m3_hf_gen", max_tokens=128)
    na = run_gen("m3_nano_gen", model=MOE, eager=True, max_tokens=128, logprobs=10,
                 gpu_util=0.4, max_num_batched_tokens=4096)
    ok = check_equal_or_noise(na, hf, "greedy 128 token: nano(EP=1) vs HF",
                              name_a="nano", name_b="HF", max_ulp=4)
    os.makedirs(common.BASELINE_DIR, exist_ok=True)
    with open(BASELINE, "w") as f:
        json.dump(na, f)
    print(f"    基线已写入 {os.path.relpath(BASELINE, ROOT)}（M4/M6 的 EP=2 对拍用）")
    results.append(report("greedy 128 token vs HF", ok))


def test_3_router_consistency(results):
    """判据 3：逐层路由一致性。

    随机权重下 router 的 top-k 边界并列极其常见，nano(flash-attn) 与 HF(eager attn)
    的 hidden states 差几个 ulp 就足以让边界翻面。所以判据不是"零分歧"，而是
    "每一个分歧都必须落在最并列的尾巴里"——逻辑 bug 会在随机位置发作，不可能只挑
    最并列的那几个百分点出现（与 Plan-1-2-3 的 4-ulp 判据同一套论证方式）。
    """
    print("\n[3] 逐层路由一致性（600 token 单条 prompt，4 层）")
    na = run_gen("m3_route_nano", model=MOE, eager=True, prompts="long", max_tokens=1,
                 gpu_util=0.4, max_num_batched_tokens=4096, router_probe=True)
    hf = run_hf("m3_route_hf", prompts="long", router_probe=True)
    rn, rh = na["routes"], hf["routes"][0]
    assert len(rn) == len(rh), f"层数不一致 {len(rn)} vs {len(rh)}"

    gaps, diff_set, order_only = [], [], 0
    for li, (a, b) in enumerate(zip(rn, rh)):
        for i, (x, y, v) in enumerate(zip(a["idx"], b["idx"], a["topv"])):
            gaps.append(v[-2] - v[-1])              # 第 k 名与第 k+1 名的概率差
            if x == y:
                continue
            if set(x) == set(y):
                order_only += 1                     # 只是顺序不同，对输出零影响（加法可交换）
            else:
                diff_set.append((li, i, v[-2] - v[-1]))
    gaps_sorted = sorted(gaps)

    def pct(g):
        return 100.0 * sum(1 for x in gaps_sorted if x <= g) / len(gaps_sorted)

    worst_pct = max((pct(g) for _, _, g in diff_set), default=0.0)
    median = gaps_sorted[len(gaps_sorted) // 2]
    ok = worst_pct <= 10.0                          # 全部分歧都在最并列的 10% 内
    print(f"    共 {len(gaps)} 个 (token, 层) 位置；顺序不同 {order_only} 个（对输出无影响），"
          f"选中专家不同 {len(diff_set)} 个")
    print(f"    第k名-第k+1名 概率差中位数 {median:.5f}；分歧点最高分位数 {worst_pct:.2f}%（上限 10%）")
    for li, i, g in diff_set[:5]:
        print(f"      layer{li} token{i}: gap={g:.3e} → 最并列的 {pct(g):.2f}%")
    results.append(report("逐层路由分歧全部落在最并列尾部", ok))
    return na


def test_4_expert_utilization(results, na):
    """判据 4：专家利用率 sanity（打印，不设阈值）。"""
    print("\n[4] 专家利用率直方图")
    for li, layer in enumerate(na["routes"]):
        hist = {}
        for row in layer["idx"]:
            for e in row:
                hist[e] = hist.get(e, 0) + 1
        tot = sum(hist.values())
        share = {e: f"{100.0 * c / tot:.1f}%" for e, c in sorted(hist.items())}
        print(f"    layer{li}: {share}")
    results.append(report("专家利用率直方图已打印（无阈值）", True))


def test_5_ep_filter(results):
    """判据 5：EP 过滤单测。单进程里 mock ep_rank=1/ep_size=2，只有 expert 2/3 该落权重。"""
    print("\n[5] EP 过滤 + 权重落位（单进程 mock ep_rank=1, ep_size=2）")
    import torch
    from safetensors import safe_open
    from glob import glob
    from nanovllm.utils import parallel
    from nanovllm.utils.loader import _EXPERT_RE

    # 正则边界：router 的 mlp.gate.weight 不能被 experts 分支捕获
    assert _EXPERT_RE.match("model.layers.0.mlp.gate.weight") is None
    m = _EXPERT_RE.match("model.layers.2.mlp.experts.3.gate_proj.weight")
    assert m and m.group(1) == "model.layers.2.mlp" and int(m.group(2)) == 3 \
        and m.group(3) == "gate_proj", m

    old = parallel.get_ep_size, parallel.get_ep_rank
    parallel.get_ep_size, parallel.get_ep_rank = (lambda: 2), (lambda: 1)
    try:
        from nanovllm.models.qwen3_moe import FusedExpertsEP
        ex = FusedExpertsEP(num_experts=4, hidden_size=2048, intermediate_size=512)
        assert ex.num_local == 2 and ex.expert_start == 2, (ex.num_local, ex.expert_start)
        ex.w13.data.zero_(); ex.w2.data.zero_()

        path = os.path.expanduser(MOE)
        src = {}
        with safe_open(sorted(glob(os.path.join(path, "*.safetensors")))[0], "pt", "cpu") as f:
            for e in range(4):
                for shard in ("gate_proj", "up_proj", "down_proj"):
                    src[(e, shard)] = f.get_tensor(f"model.layers.0.mlp.experts.{e}.{shard}.weight")
        for (e, shard), t in src.items():
            ex.load_expert_weight(t, e, shard)

        checks = [
            ("w13[0] 前半 == 全局 expert2.gate_proj", torch.equal(ex.w13[0, :512], src[(2, "gate_proj")])),
            ("w13[0] 后半 == 全局 expert2.up_proj", torch.equal(ex.w13[0, 512:], src[(2, "up_proj")])),
            ("w2[0] == 全局 expert2.down_proj", torch.equal(ex.w2[0], src[(2, "down_proj")])),
            ("w13[1] 前半 == 全局 expert3.gate_proj", torch.equal(ex.w13[1, :512], src[(3, "gate_proj")])),
            ("w2[1] == 全局 expert3.down_proj", torch.equal(ex.w2[1], src[(3, "down_proj")])),
        ]
        # expert 0/1 的权重一个字节都不该出现（它们属于 rank 0）
        for e in (0, 1):
            for shard in ("gate_proj", "up_proj"):
                hit = any(torch.equal(ex.w13[l, :512] if shard == "gate_proj" else ex.w13[l, 512:],
                                      src[(e, shard)]) for l in range(2))
                checks.append((f"expert{e}.{shard} 未落进本 rank", not hit))
        ok = all(v for _, v in checks)
        for name, v in checks:
            print(f"    {'OK ' if v else 'BAD'} {name}")
    finally:
        parallel.get_ep_size, parallel.get_ep_rank = old
    results.append(report("EP 过滤：只加载本 rank 的专家，落位正确", ok))


def main():
    print("=" * 70)
    print(">>> M3 · MoE 模型层（EP=1）vs HF 对拍")
    print("=" * 70)
    results = []
    test_1_single_step_vs_hf(results)
    test_2_greedy_vs_hf_and_baseline(results)
    na = test_3_router_consistency(results)
    test_4_expert_utilization(results, na)
    test_5_ep_filter(results)
    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} passed")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
