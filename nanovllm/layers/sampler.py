import torch
from torch import nn


NEG_INF = -float("inf")


def apply_top_k_top_p(logits: torch.Tensor, top_ks: torch.Tensor | None, top_ps: torch.Tensor | None):
    # 一次降序 stable sort 同时服务 top_k 和 top_p。
    #
    # 用 stable=True 的降序排序,而不是"按第 k 大的值做阈值",是因为 lm_head 出来的
    # logits 是 bf16,只有 8 位尾数,词表 15 万个词里出现"并列最大值"是常事
    # (实测一次前向里就有一行有 3 个并列)。阈值法在并列时会把并列的全部留下,
    # top_k=1 就退化成在并列项里随机挑一个,和 greedy 对不上。
    # 按排序位置截断则恒好保留 k 个;stable 保证并列时保留原始下标最小的那个,
    # 与 torch.argmax "返回第一个最大值下标" 的行为一致,于是
    # greedy == top_k=1 == top_p→0 在并列情况下也严格相等。
    if top_ks is None and top_ps is None:
        return logits
    logits_sort, logits_idx = logits.sort(dim=-1, descending=True, stable=True)

    if top_ks is not None:
        pos = torch.arange(logits.size(-1), device=logits.device).unsqueeze(dim=0)
        logits_sort = logits_sort.masked_fill(pos >= top_ks.unsqueeze(dim=1), NEG_INF)

    if top_ps is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        # nucleus = 使累计概率首次 >= p 的最短前缀。
        # 位置 i 该保留 <=> 它前面(不含自己)的累计概率还没到 p。
        prev_sum = probs_sum - probs_sort
        mask = prev_sum >= top_ps.unsqueeze(dim=1)
        mask[:, 0] = False    # 至少保留概率最大的那个 token
        logits_sort = logits_sort.masked_fill(mask, NEG_INF)

    return torch.empty_like(logits_sort).scatter_(dim=-1, index=logits_idx, src=logits_sort)


def compute_probs(logits: torch.Tensor, temperatures: torch.Tensor,
                  top_ks: torch.Tensor | None, top_ps: torch.Tensor | None):
    # 返回"实际用于采样"的概率分布。投机解码的 rejection sampling 要求
    # p 和 q 都是截断后的分布,所以这一步必须和 Sampler.forward 走完全相同的处理。
    logits = logits.float()
    safe_t = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
    logits = logits / safe_t.unsqueeze(dim=1)
    logits = apply_top_k_top_p(logits, top_ks, top_ps)
    return torch.softmax(logits, dim=-1)


def sample_from_probs(probs: torch.Tensor):
    # Gumbel-max:除以 Exp(1) 再取 argmax,等价于按 probs 多项式采样,且无需 CPU 同步
    q = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return probs.div(q).argmax(dim=-1)


class Sampler(nn.Module):#把模型输出的 logits 变成具体的 token id —— 决定"下一个词是哪个"。

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor,
                top_ks: torch.Tensor | None = None, top_ps: torch.Tensor | None = None,
                max_logprobs: int = -1):
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)

        # temperature == 0 表示 greedy。除法要先避开 0,再用 where 按行挑回 argmax 结果,
        # 这样 greedy 行和采样行可以待在同一个 batch 里,不用拆分。
        safe_t = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
        logits = logits / safe_t.unsqueeze(dim=1)

        # logprobs 取"温度缩放后、top_k/top_p 截断前"的分布(与 vLLM 一致);
        # greedy 行 safe_t == 1,等价于原始 logits 的 log_softmax。
        logprobs = torch.log_softmax(logits, dim=-1) if max_logprobs >= 0 else None

        logits = apply_top_k_top_p(logits, top_ks, top_ps)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = sample_from_probs(probs)
        tokens = torch.where(temperatures == 0, greedy_tokens, sample_tokens)

        if logprobs is None:
            return tokens, None
        token_logprobs = logprobs.gather(dim=-1, index=tokens.unsqueeze(dim=1)).squeeze(dim=1)
        if max_logprobs > 0:
            top_logprobs, top_ids = logprobs.topk(max_logprobs, dim=-1)
        else:
            top_logprobs = top_ids = None
        return tokens, (token_logprobs, top_ids, top_logprobs)
# [OLD ↓ 下面 5 行] 原注释保留原文不动;其中的形状和"每条 seq 一个词"已随 Phase 2/3 改动而过时
# 位置
# 28 层前向  ->  hidden_states  [T, 4096]
# lm_head    ->  logits         [T, 151936]     每个词一个分数
# sampler    ->  token_ids      [T]             每条 seq 挑出一个词
# logits 是词表里 15 万个词各自的原始分数,sampler 负责从中选一个。选出来的 id 随后被 postprocess 追加到 seq 上(scheduler.py:88),成为下一轮的输入。
# [NEW] 现状:
#   1) lm_head 不再对全部 T 行算 logits。prepare_batch 会算好 context.logits_indices,
#      普通 seq 只取它那段 q 的最后一个位置,投机 seq 取全部 k+1 个位置;
#      chunked prefill 的中间块干脆不进这张表(原来那些行算完就丢)。
#      所以 logits 的行数是 R = Σ(每条 seq 需要的行数),不是 T。
#   2) 投机解码时一条 seq 一步可能产出多个 token(被接受的草稿 + 修正/奖励 token),
#      走的是 ModelRunner.sample_speculative 而不是这里的 Sampler.forward。
#   3) postprocess 的位置也变了(现在在 scheduler.py 的 postprocess 里,一次可 append 多个)。
