"""E4:草稿模型的 KV 该怎么跟着走 —— 先把簿记推清楚,再写 GPU 代码。

不碰 GPU。把"一轮投机"里两套 cache 的位置账目当成纯整数问题模拟出来,
用随机接受序列跑几万轮,断言不变量。写错的方案会在这里当场炸,
而不是在真机上表现成"接受率莫名其妙低了一点"。

约定(与 nano-vllm 现有语义一致):
  len            = seq.num_tokens
  target_cached  = seq.num_cached_tokens,恒等于 len-1(postprocess 之后)
                   含义:位置 0..target_cached-1 的目标 KV 已算好
  draft_cached   = 位置 0..draft_cached-1 的草稿 KV 已算好

一轮的动作:
  1) 草稿模型第一次前向吃 [draft_cached, len) 这一段(q = len - draft_cached),
     写草稿 KV 到这些位置,并产出 d_1;
  2) 再跑 k-1 次 q=1,分别吃 d_1..d_{k-1},写位置 len..len+k-2,产出 d_2..d_k;
     —— 注意 d_k 自己的 KV 这一轮**没人算**,这就是缺口的来源;
  3) 目标模型吃 [last_token, d_1..d_k],q=k+1,写目标 KV 到 len-1..len+k-1;
  4) 接受 a 个(0<=a<=k)+ 1 个修正/奖励 token,新长度 len' = len+a+1。

要验的三条:
  A. target_cached' == len'-1               (现有不变量,不能被破坏)
  B. draft_cached'  == len'-1 或 len'-2      (最多落后一格)
  C. 落后恰好一格 <=> a == k                 (即"全接受"是唯一的缺口来源)
并顺带量出 D:第一次前向的 q 长度分布 —— 它决定草稿那族 varlen 图的 q_max。
"""
import random
import sys
from collections import Counter


def simulate(k, rounds, seed, prompt_len=17):
    rng = random.Random(seed)
    length = prompt_len
    # prefill 同步之后两套 cache 齐平:都算到 len-1
    target_cached = length - 1
    draft_cached = length - 1
    q_hist = Counter()
    gap_when_full = 0
    full_accepts = 0

    for _ in range(rounds):
        # 1) 草稿第一次前向
        q_first = length - draft_cached
        q_hist[q_first] += 1
        assert q_first >= 1, "草稿 cache 跑到 last_token 前面去了"
        draft_written_upto = length - 1          # 位置 0..len-1 的草稿 KV 齐了
        # 2) 再跑 k-1 次 q=1,写位置 len..len+k-2
        if k >= 1:
            draft_written_upto = length + k - 2
        # 3) 目标一次前向,写 len-1..len+k-1
        # 4) 接受 a 个
        a = rng.randint(0, k)
        new_len = length + a + 1
        # 目标侧簿记(scheduler.postprocess)
        target_cached = target_cached + (a + 1)
        # 草稿侧:被接受的位置才算数 —— 位置 len-1..len+a-1 的草稿 KV 有效,
        # 但只有当草稿真的写到了那里才成立
        draft_cached = min(length + a, draft_written_upto + 1)

        assert target_cached == new_len - 1, "A 破了:目标不变量"
        lag = (new_len - 1) - draft_cached
        assert lag in (0, 1), f"B 破了:草稿落后 {lag} 格"
        if a == k:
            full_accepts += 1
            gap_when_full += (lag == 1)
        else:
            assert lag == 0, f"C 破了:a={a}<k={k} 却落后了 {lag} 格"
        length = new_len

    ok_c = (gap_when_full == full_accepts)
    return dict(k=k, q_hist=dict(sorted(q_hist.items())), full=full_accepts,
                gap=gap_when_full, ok_c=ok_c, final_len=length)


def main():
    print("=" * 88)
    print("E4 草稿 KV 簿记模拟(纯整数,无 GPU)")
    print("=" * 88)
    all_ok = True
    max_q = 0
    for k in (1, 2, 4, 8):
        for seed in (0, 1, 2):
            r = simulate(k, rounds=20000, seed=seed)
            all_ok &= r["ok_c"]
            max_q = max(max_q, max(r["q_hist"]))
            if seed == 0:
                print(f"  k={k}: 第一次前向 q 长度分布 {r['q_hist']}"
                      f"   全接受 {r['full']} 次,其中落后一格 {r['gap']} 次"
                      f"   {'✓' if r['ok_c'] else '✗'}")
    print()
    print(f"  A/B/C 三条不变量在 k∈{{1,2,4,8}} × 3 seed × 20000 轮下"
          f"{'全部成立' if all_ok else '有反例'}")
    print(f"  D 第一次前向的 q 长度上界 = {max_q}  → 草稿那族 varlen 图 q_max 取 2 即可")
    print()
    print("结论:")
    print("  · 不需要任何'回滚'。被拒绝位置的草稿 KV 是垃圾,但下一轮真实 token 会")
    print("    原地覆盖它们(slot 由 position 唯一决定),而 attention 的读取被")
    print("    cu_seqlens_k(= 真实长度)挡住,读不到残留区 —— 与 04 计划文档对目标侧")
    print("    残留 KV 的论证同构。")
    print("  · 唯一要补的是 a==k(全接受)时缺的那一格:第 k 个草稿 token 自己的 KV")
    print("    这一轮没人算。做法是下一轮草稿的第一次前向吃 2 个 token 而不是 1 个,")
    print("    成本是每 (1/α^k) 轮多一行 q —— 不是多跑一次前向。")
    print("  · 这与 vLLM 的结论一致:draft_model 每条请求多留 **1** 个 slot")
    print("    (vllm/config/speculative.py:1421-1459 的表格 + :1453-1455 的注释")
    print("     'The autoregressive draft-model input retains one unsliced token.'),")
    print("    而 ngram / EAGLE3 / MTP 是 0。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
