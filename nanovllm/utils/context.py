from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    # 本轮是 prefill 还是 decode。attention 层据此选 kernel:
    # True  -> flash_attn_varlen_func    False -> flash_attn_with_kvcache

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
    # kernel 靠它把逻辑位置翻译成物理 block。prefill 时只有命中 prefix cache 才需要

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
