"""NanoEPBuffer —— 按 transport 分发的门面，以及两后端共用的 handle 定义。"""

from typing import NamedTuple

import torch
import torch.distributed as dist


def pack2(count: int, begin: int) -> int:
    """照抄 internode_ll.cu:411 的 pack2<int, int64_t>：高 32 位 begin、低 32 位 count。"""
    return (begin << 32) | (count & 0xFFFFFFFF)


def unpack2(v: int) -> tuple[int, int]:
    return int(v & 0xFFFFFFFF), int(v >> 32)


class EPHandle(NamedTuple):
    """前 5 项与 deep_ep 的 handle 五元组同序同义，可直接解包：

        src_info, layout_range, M, hidden, num_experts = handle[:5]

    后面几项是 nano 的后端私有数据（nccl 后端的置换信息），nvshmem 后端置 None。
    """
    src_info: torch.Tensor          # [L, R*M] int32，槽位 -> 源 rank 上的 token 下标
    layout_range: torch.Tensor      # [L, R] int64，pack2(count, begin)
    M: int                          # num_max_dispatch_tokens_per_rank
    hidden: int
    num_experts: int
    # ---- 以下为 nccl 后端私有 ----
    send_splits: list | None = None     # 本 rank 发往各 rank 的行数
    recv_splits: list | None = None     # 本 rank 从各 rank 收到的行数
    recv_cnt: list | None = None        # recv_cnt[r][l]：rank r 发给我本地专家 l 的行数
    order_tok: torch.Tensor | None = None   # [S] 发送序对应的源 token 下标
    order_k: torch.Tensor | None = None     # [S] 发送序对应的 k 下标
    num_tokens: int = 0                 # T，combine 的输出行数


class NanoEPBuffer:
    """LL 语义的 dispatch/combine。

    形状约定（L = num_experts // ep_size，R = ep_size，M = 每 rank 单步 token 上限）：
        low_latency_dispatch(x[T,H], topk_idx[T,K])
            -> packed_recv_x[L, R*M, H], recv_count[L], handle
        low_latency_combine(x[L, R*M, H], topk_idx[T,K], topk_weights[T,K], handle)
            -> combined_x[T, H]

    packed_recv_x 的有效行是**前 recv_count[l] 行**（压实在段首），rank r 的那段落在
    layout_range[l][r] 给出的 [begin, begin+count)。这与 DeepEP 一致：内核接收侧用
    `atomicAdd(packed_recv_count+l, n)` 拿 begin，所以 begin 是 R*M 维上的绝对下标、
    各 rank 的段首尾相接。（Plan-4/03 里写的"begin 恒 0、每段从 r*M 起"是笔误，那是
    RDMA 中转缓冲的布局，不是 packed_recv_x 的布局——见报告坑 3。）

    与 DeepEP 的唯一实质差异：DeepEP 的 begin 由 atomicAdd 竞争决定，rank 段的先后
    是**到达序**（不确定）；nano 恒按 **rank 升序**排。这是 DeepEP 语义的一个合法实例，
    换来"同输入必位级同输出"（验收 4 依赖）。
    """

    def __init__(self, group: dist.ProcessGroup, num_max_dispatch_tokens_per_rank: int,
                 hidden: int, num_experts: int, transport: str = "nccl"):
        self.group = group
        self.rank = dist.get_rank(group)
        self.R = dist.get_world_size(group)
        assert num_experts % self.R == 0, f"{num_experts=} 必须被 {self.R=} 整除"
        self.E = num_experts
        self.L = num_experts // self.R
        self.M = num_max_dispatch_tokens_per_rank
        self.H = hidden
        self.transport = transport
        self.expert_start = self.rank * self.L

        if transport == "nccl":
            from .nccl_backend import NcclBackend
            self._impl = NcclBackend(self)
        elif transport == "nvshmem":
            from .nvshmem_backend import NvshmemBackend
            self._impl = NvshmemBackend(self)
        else:
            raise ValueError(f"未知 transport: {transport}")

    # ---- 对外 API ----

    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.H, f"{x.shape=} 与 {self.H=} 不符"
        assert x.size(0) <= self.M, f"T={x.size(0)} 超过 M={self.M}（对齐 buffer.hpp:1481 的 host 检查）"
        assert topk_idx.dtype == torch.int64, "topk_idx 用 int64（deep_ep.topk_idx_t 默认 64 位）"
        assert topk_idx.size(0) == x.size(0)
        return self._impl.dispatch(x, topk_idx)

    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor,
                            topk_weights: torch.Tensor, handle: EPHandle):
        assert x.dim() == 3 and x.shape[0] == self.L and x.shape[2] == self.H, f"{x.shape=}"
        assert topk_weights.dtype == torch.float32, "topk_weights 用 fp32"
        return self._impl.combine(x, topk_idx, topk_weights, handle)

    def destroy(self):
        if hasattr(self._impl, "destroy"):
            self._impl.destroy()

    def __repr__(self):
        return (f"NanoEPBuffer(transport={self.transport}, rank={self.rank}/{self.R}, "
                f"E={self.E}, L={self.L}, M={self.M}, H={self.H})")
