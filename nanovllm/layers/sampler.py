import torch
from torch import nn


class Sampler(nn.Module):#把模型输出的 logits 变成具体的 token id —— 决定"下一个词是哪个"。

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
# 位置
# 28 层前向  ->  hidden_states  [T, 4096]
# lm_head    ->  logits         [T, 151936]     每个词一个分数
# sampler    ->  token_ids      [T]             每条 seq 挑出一个词
# logits 是词表里 15 万个词各自的原始分数,sampler 负责从中选一个。选出来的 id 随后被 postprocess 追加到 seq 上(scheduler.py:88),成为下一轮的输入。