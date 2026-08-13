"""nano-deepEP —— DeepEP legacy(V1) low-latency 路径的极简移植。

为什么只取 low-latency：本仓库 DeepEP 是 V2 主干（要求 SM90 / torch>=2.10 /
NCCL>=2.30.4，三项全不满足），但完整保留了 legacy V1。V1 三条路径里
intranode(NVLink) 和 normal internode(按 rdma_rank=rank/8 分组，2 ranks 会退化成
"同节点") 都不适用"两机各一卡"的形态；只有 low-latency 模式下
`nvshmem_rank = rank, num_nvshmem_ranks = num_ranks`，每个 rank 就是一个独立的
RDMA PE，任意 rank 数都支持 —— 所以选它。

LL 名义上是给 decode 用的（`num_max_dispatch_tokens_per_rank` 是静态上限），但
nano-vllm 的统一调度保证每步 token 数 <= max_num_batched_tokens，把 M 设成它，
一条 LL 路径就同时覆盖 prefill 和 decode。

两个 transport，API 与数据布局完全一致，上层无感切换：

    "nccl"    —— torch.distributed.all_to_all_single 实现。数据面走 NCCL over
                 RoCE 的 GPUDirect RDMA。立即可跑，且是 nvshmem 后端的对拍 oracle。
    "nvshmem" —— 忠实 DeepEP 的 IBGDA 内核，GPU 发起 RDMA。需要
                 PeerMappingOverride=1 或 gdrcopy（见 Plan-4/01-m0-environment.md）。

与 deep_ep.Buffer 的取舍对照：

| deep_ep 参数/返回                              | nano                      | 理由                       |
|-----------------------------------------------|---------------------------|----------------------------|
| use_fp8 / round_scale / use_ue8m0              | 首版恒 bf16（nvshmem 后端 | 内核原生支持，先减一维变量 |
|                                                | 保留 use_fp8 开关）       |                            |
| async_finish / return_recv_hook                | 不暴露，恒同步            | nano 无重叠需求            |
| cumulative_*_stats / mask / shrink             | 不要                      | 容错/监控超出 nano 范围    |
| handle 五元组 (src_info, layout_range, M, H, E)| 同构保留（前 5 项同序）   | combine 必需 + 可与 DeepEP |
|                                                |                           | 互换                       |
| packed_recv_x 布局 [L, R*M, H]                 | 逐字节同语义              | 换后端零改动               |
| layout_range[l][r] = pack2(count, begin)       | 同（begin 是 R*M 维上的   | 同上                       |
|                                                | 绝对下标）                |                            |
"""

from .buffer import NanoEPBuffer, EPHandle, pack2, unpack2

_BUFFER: NanoEPBuffer | None = None


def init_ep_buffer(group, num_max_dispatch_tokens_per_rank: int, hidden: int,
                   num_experts: int, transport: str = "nccl") -> "NanoEPBuffer":
    """进程级单例。MoE 层每层共用同一个 buffer——LL 语义下 dispatch/combine
    成对调用即可复用（DeepEP 的 decode 示例同款用法）。"""
    global _BUFFER
    _BUFFER = NanoEPBuffer(group, num_max_dispatch_tokens_per_rank, hidden,
                           num_experts, transport)
    return _BUFFER


def get_ep_buffer() -> "NanoEPBuffer":
    assert _BUFFER is not None, "init_ep_buffer() 没被调过"
    return _BUFFER


def destroy_ep_buffer():
    global _BUFFER
    if _BUFFER is not None:
        _BUFFER.destroy()
        _BUFFER = None


__all__ = ["NanoEPBuffer", "EPHandle", "pack2", "unpack2",
           "init_ep_buffer", "get_ep_buffer", "destroy_ep_buffer"]
