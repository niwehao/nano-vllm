"""Phase 2 · 统一调度器(prefill/decode 混批)正确性测试。

测试设计说明:
把请求混进同一个 batch、把 prompt 切成 chunk、命中 prefix cache,这三件事在数学上
都不该改变任何一条请求的 logits。但在 bf16 下它们都会改变 GEMM 的分块和归约顺序,
于是 logits 会有 ulp 级别的漂移,greedy 的 argmax 在近似并列处会翻转,
128 步自回归之后整段文本就不同了。原始代码(改动前)同样如此:
  - eager vs cudagraph        4/6 条一致
  - 批量 vs 串行(max_num_seqs=1) 3/6 条一致
所以"逐 token 全等"不是这里能用的判据。本文件用两层判据:
  1) 单步 logprob 比对(主判据,锐利):只生成 1 个 token,直接比 top-10 的 logprob 值。
     不经自回归放大,能把"数学等价"和"逻辑 bug"清晰分开。
  2) 长输出的 token 比对 + 分歧点噪声分析(辅判据):分歧点上两个候选的 logprob 差
     必须在 bf16 噪声量级内。
"""
import json
import os

import common
from common import BASELINE_DIR
from harness import (run_gen, token_ids, check_equal, check_equal_or_noise,
                     compare_logprobs, diff_report, OUT_DIR)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    print("=== Phase 2 统一调度器测试 ===\n")

    # ---------- 基准运行:预算充足,一次性 prefill,不产生混批 ----------
    print("[0] 基准运行(预算充足,无混批)")
    ref1 = run_gen("p2_ref_step1", eager=True, temperature=0.0, max_tokens=1, logprobs=10)
    ref = run_gen("p2_ref", eager=True, temperature=0.0, max_tokens=64, logprobs=5)
    print(f"    调度统计: {ref['stats']}")

    print("\n[1] chunked prefill 单步等价:prompt 切成 128 token 的 chunk,首个 token 的 logits 不变")
    chunk1 = run_gen("p2_chunk_step1", eager=True, temperature=0.0, max_tokens=1, logprobs=10,
                     max_num_batched_tokens=128)
    print(f"    调度统计: {chunk1['stats']}")
    results.append(compare_logprobs(ref1, chunk1, "全量 prefill vs 分块 prefill 的首步 logprob"))

    print("\n[2] 混批必须真的发生")
    # 预算 128 -> 600 token 的 prompt 要切 5 个 chunk,期间先完成的请求已在 decode
    mixed = run_gen("p2_mixed", eager=True, temperature=0.0, max_tokens=64, logprobs=5,
                    max_num_batched_tokens=128)
    st = mixed["stats"]
    ok = st.get("mixed", 0) > 0
    print(f"    调度统计: {st}")
    print(f"  [{'PASS' if ok else 'FAIL'}] 存在 prefill+decode 同批的 step: "
          f"{st.get('mixed')} / {st.get('steps')} 步")
    results.append(ok)

    print("\n[3] 混批 64 步输出:分歧必须都是浮点噪声")
    results.append(check_equal_or_noise(ref, mixed, "混批 vs 非混批", "nomix", "mixed"))

    print("\n[4] prefix cache 单步等价:共享 512 token 前缀的请求,命中缓存后 logits 不变")
    # pair = [long_a, long_b],long_b 与 long_a 共享整 2 个 block。
    # 对照组是同样两条请求在完整 batch 里跑(此时它们同批 prefill,缓存来不及建立)。
    pair1 = run_gen("p2_pair_step1", eager=True, temperature=0.0, max_tokens=1, logprobs=10,
                    prompts="pair")
    sub = {"outputs": [ref1["outputs"][2], ref1["outputs"][3]]}
    results.append(compare_logprobs(sub, pair1, "prefix cache 命中 vs 未命中的首步 logprob"))

    print("\n[5] 抢占路径:6 条 255-token 请求,只给 7 个 KV block")
    # 每条 prefill 后占 1 块(6 块);生成到第 257 个 token 时每条都要第 2 块,
    # 但只剩 1 块空闲 -> 必然发生抢占重算。
    pre_ref = run_gen("p2_preempt_ref", eager=True, temperature=0.0, max_tokens=8, logprobs=5,
                      prompts="preempt")
    pre = run_gen("p2_preempt", eager=True, temperature=0.0, max_tokens=8, logprobs=5,
                  prompts="preempt", num_kvcache_blocks=7)
    st = pre["stats"]
    print(f"    受限调度统计: {st}")
    ok = st.get("preempted", 0) > 0
    print(f"  [{'PASS' if ok else 'FAIL'}] 确实触发了抢占: {st.get('preempted')} 次")
    results.append(ok)
    results.append(check_equal(token_ids(pre_ref), token_ids(pre),
                               "抢占重算后输出与不抢占完全一致", "normal", "preempt"))

    print("\n[6] 串行 vs 并发(纯 batch 组成差异,原始代码也有,只做噪声判定)")
    serial = run_gen("p2_serial", eager=True, temperature=0.0, max_tokens=64, logprobs=5,
                     max_num_seqs=1)
    results.append(check_equal_or_noise(ref, serial, "并发 vs 串行", "batch", "serial"))

    print("\n[7] cudagraph 快路径(纯 decode 批)仍可用")
    graph = run_gen("p2_graph", eager=False, temperature=0.0, max_tokens=64, logprobs=5)
    results.append(check_equal_or_noise(ref, graph, "eager vs cudagraph", "eager", "graph"))

    print("\n[8] Phase 1 基线回归")
    with open(os.path.join(BASELINE_DIR, "greedy_p1.json")) as f:
        base = json.load(f)
    p1_128 = {"outputs": [{"token_ids": t[:64]} for t in base["eager"]]}
    results.append(check_equal_or_noise(ref, p1_128, "Phase 2 vs Phase 1 基线", "p2", "p1"))

    # 保存 Phase 2 基线
    os.makedirs(BASELINE_DIR, exist_ok=True)
    with open(os.path.join(BASELINE_DIR, "greedy_p2.json"), "w") as f:
        json.dump({"eager": token_ids(ref), "graph": token_ids(graph)}, f)

    passed = sum(results)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
