"""Phase 1 · 真模型端到端测试 + 建立 greedy 回归基线。"""
import json
import os
import shutil
import sys

import common
from common import BASELINE_DIR
from harness import run_gen, token_ids, check_equal, check_equal_or_noise, OUT_DIR

PHASE = os.environ.get("NANOVLLM_PHASE", "p1")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(BASELINE_DIR, exist_ok=True)
    results = []

    print("=== Phase 1 端到端测试 (真模型 Qwen3-0.6B) ===\n")

    print("[1] greedy 基线 (eager / cuda graph),带 logprobs 以便分析分歧")
    g_eager = run_gen(f"{PHASE}_greedy_eager", eager=True, temperature=0.0, max_tokens=128, logprobs=5)
    g_graph = run_gen(f"{PHASE}_greedy_graph", eager=False, temperature=0.0, max_tokens=128, logprobs=5)

    # eager 走 flash_attn_with_kvcache 的直接调用,graph 走把 batch padding 到桶大小后的 replay,
    # 归约顺序不同 => logits 有 bf16 级别的漂移。原始代码(改动前)在同样的 prompt 上也有同样的分歧,
    # 所以这里判定的是"分歧点是否落在噪声范围内",而不是逐 token 全等。
    results.append(check_equal_or_noise(g_eager, g_graph,
                                        "eager vs cudagraph 分歧均为浮点噪声", "eager", "graph"))

    print("\n[3] greedy 可复现性 (同配置跑两次)")
    g_eager2 = run_gen(f"{PHASE}_greedy_eager_rerun", eager=True, temperature=0.0, max_tokens=128)
    results.append(check_equal(token_ids(g_eager), token_ids(g_eager2),
                               "greedy 重复运行确定性", "run1", "run2"))

    print("\n[4] top_k=1 必须等价于 greedy(温度任意,截断到唯一候选)")
    k1 = run_gen(f"{PHASE}_topk1", eager=True, temperature=1.0, top_k=1, max_tokens=128)
    results.append(check_equal(token_ids(g_eager), token_ids(k1),
                               "top_k=1 == greedy", "greedy", "top_k=1"))

    print("\n[5] top_p 极小必须等价于 greedy(nucleus 只剩最大概率那个)")
    p_tiny = run_gen(f"{PHASE}_toppmin", eager=True, temperature=1.0, top_p=1e-5, max_tokens=128)
    results.append(check_equal(token_ids(g_eager), token_ids(p_tiny),
                               "top_p→0 == greedy", "greedy", "top_p=1e-5"))

    print("\n[6] logprobs 结构与数值 (复用 [1] 的 eager 结果)")
    lp = g_eager
    ok = True
    detail = []
    ties = 0
    for i, o in enumerate(lp["outputs"]):
        lps = o["logprobs"]
        if lps is None or len(lps) != len(o["token_ids"]):
            ok = False
            detail.append(f"    seq[{i}]: logprobs 条数 {None if lps is None else len(lps)} != token 数 {len(o['token_ids'])}")
            continue
        for j, item in enumerate(lps):
            if item["token_id"] != o["token_ids"][j]:
                ok = False
                detail.append(f"    seq[{i}][{j}]: logprob.token_id {item['token_id']} != token {o['token_ids'][j]}")
                break
            top = item["top_logprobs"]
            if len(top) != 5:
                ok = False
                detail.append(f"    seq[{i}][{j}]: top_logprobs 长度 {len(top)} != 5")
                break
            # greedy 下,采样出的 token 必须取到 top-1 的 logprob 值。
            # 不能直接比 token id:bf16 logits 常出现并列最大值,argmax 返回下标最小的那个,
            # 而 topk 在并列时的顺序未定义,两者可能给出不同 id 但 logprob 完全相等。
            if top[0][0] != item["token_id"]:
                if abs(item["logprob"] - top[0][1]) > 1e-6:
                    ok = False
                    detail.append(f"    seq[{i}][{j}]: greedy token {item['token_id']} "
                                  f"(logprob {item['logprob']:.6f}) 严格劣于 top-1 {top[0][0]} "
                                  f"(logprob {top[0][1]:.6f})")
                    break
                ties += 1
            # top_logprobs 必须降序,且 logprob <= 0
            vals = [v for _, v in top]
            if vals != sorted(vals, reverse=True) or max(vals) > 1e-6:
                ok = False
                detail.append(f"    seq[{i}][{j}]: top_logprobs 非降序或为正: {vals}")
                break
            if abs(item["logprob"] - top[0][1]) > 1e-5:
                ok = False
                detail.append(f"    seq[{i}][{j}]: 采样 token logprob {item['logprob']} != top-1 {top[0][1]}")
                break
    print(f"  [{'PASS' if ok else 'FAIL'}] logprobs 结构/数值/降序/与 greedy 自洽"
          f"({ties} 处并列最大值,已按 logprob 相等判定)")
    for d in detail[:5]:
        print(d)
    results.append(ok)

    print("\n[7] 采样路径(temperature>0)能正常跑完且长度正确")
    s = run_gen(f"{PHASE}_sample", eager=True, temperature=0.8, top_k=50, top_p=0.9, max_tokens=64)
    lens = [len(o["token_ids"]) for o in s["outputs"]]
    ok = all(l == 64 for l in lens)
    print(f"  [{'PASS' if ok else 'FAIL'}] top_k=50/top_p=0.9/T=0.8 采样输出长度全为 64: {lens}")
    results.append(ok)
    # 采样结果不应与 greedy 完全相同(否则说明随机性没生效)
    diff_any = any(a != b for a, b in zip(token_ids(s), [t[:64] for t in token_ids(g_eager)]))
    print(f"  [{'PASS' if diff_any else 'FAIL'}] 采样输出与 greedy 不同(随机性生效)")
    results.append(diff_any)

    # 保存基线
    base = os.path.join(BASELINE_DIR, f"greedy_{PHASE}.json")
    with open(base, "w") as f:
        json.dump({"eager": token_ids(g_eager), "graph": token_ids(g_graph)}, f)
    print(f"\n基线已保存: {base}")

    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
