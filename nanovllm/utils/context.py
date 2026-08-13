from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    # [OLD ↓ 下面 2 行] 原注释保留原文不动;Phase 2 统一调度后语义已经漂移
    # 本轮是 prefill 还是 decode。attention 层据此选 kernel:
    # True  -> flash_attn_varlen_func    False -> flash_attn_with_kvcache
    # [NEW] 现状:这个字段已经不是"prefill 还是 decode",而是"走不走变长统一路径":
    #   True  -> prepare_batch 的变长路径,flash_attn_varlen_func。
    #            prefill chunk、prefill+decode 混批、投机的 k+1 验证前向都走这里。
    #   False -> 只剩一种情况:纯 decode 批的快路径(prepare_decode +
    #            flash_attn_with_kvcache,可 replay CUDA graph)。那个 kernel 专为
    #            q 长度=1 做了 split-K 优化,比通用变长 kernel 快(实测单步 16.3ms vs 19.4ms),
    #            所以纯 decode 批仍然单独留了一条路。
    #   选路在 ModelRunner.is_pure_decode() / use_cudagraph()。

    cu_seqlens_q: torch.Tensor | None = None
    # cumulative sequence lengths of query,长度 batch+1 的前缀和,如 [0, 488, 556]
    # 把扁平拼接的 token 序列切回各条 seq 的边界。q 侧 = 本轮实际要计算的 token 数
    # ParallelLMHead 也用它取每条 seq 的最后一个位置(embed_head.py:59)

    cu_seqlens_k: torch.Tensor | None = None
    # key 侧的前缀和,含 prefix cache 命中的历史部分,所以 >= cu_seqlens_q
    # 两者的差额就是省掉的计算量

    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 本轮 batch 里最长的 q / k 长度,flash attention 用来决定内部分块策略

    slot_mapping: torch.Tensor | None = None#slot_mapping 是一个数组,每个 token 一个元素,不是单个值。
    # 每个待计算 token 的 KV 该写到显存哪个绝对槽位 = block_id * block_size + 块内偏移
    # store_kvcache 的 Triton kernel 按它逐 token 写入(attention.py:22) 
    # 值为 -1 表示跳过写入,给 CUDA graph 的 padding 位用

    #为什么写用,读需要block_table
#     写入  本轮 N 个 token,N 是几十到几千     表很小,CPU 展开划算
#       而且每个 token 的目的地各不相同,没有可压缩的结构
    # 读取  每条 seq 全部历史,总量几十万        必须用压缩表示
    #       而且天然按 block 连续,页表就是最自然的压缩形式

    context_lens: torch.Tensor | None = None
    # decode 专用:每条 seq 的历史总长度(含当前 token)
    # 告诉 kernel 从 cache 里读多少个 KV

    block_tables: torch.Tensor | None = None
    # 每条 seq 的页表,形状 [batch, max_blocks],padding 位填 -1
    # [OLD ↓ 下面 1 行] 原注释保留原文不动;后半句"只有命中 prefix cache 才需要"已过时
    # kernel 靠它把逻辑位置翻译成物理 block。prefill 时只有命中 prefix cache 才需要
    # [NEW] 现状:统一路径下一律需要 block_tables。
    #       因为 decode 行的 seqlen_k 远大于 seqlen_q,历史 KV 只能从 paged cache 里读,
    #       没有页表就取不到。prepare_batch 现在的条件是 any(seq.block_table),
    #       只有 warmup(KV cache 尚未分配、block_table 为空)时才保持 None —— 那时
    #       attention 直接用本轮算出的 k/v。

    logits_indices: torch.Tensor | None = None
    # 需要送进 lm_head 的行下标。普通 seq 只要它那段 q 的最后一个位置;
    # 投机解码的验证前向要全部 k+1 个位置的 logits。为 None 时 lm_head 用全部行
    # (纯 decode 的 CUDA graph 快路径就是这种情况,每行本来就是一条 seq)。

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, logits_indices=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, logits_indices)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
