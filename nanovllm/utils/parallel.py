"""parallel_state —— 进程组的唯一事实源。

在这之前，所有并行层都拿 `dist.get_world_size()` 当 TP size。单机 TP 时这没问题
（world 就是 TP），但 EP 模式下 world=2 而 TP=1，再按 world 切权重会把每台机器的
权重错切一半。所以引入这一层间接：模型侧一律问 `get_tp_size()`，EP 侧问
`get_ep_size()`。

并行语义定死（nano 只取最简形态，不做 TP×EP 交叉）：

    单机 TP:  world = tensor_parallel_size, TP 组 = WORLD,      EP 组 = None
    跨机 EP:  world = ep_size = 节点数,      TP 组 = {自己},     EP 组 = WORLD, TP=1

vLLM 里 EP 组是 DP×TP 的融合；nano 把它简化成"每机一卡、一个 rank 就是一个 EP rank"。
"""

import os
from datetime import timedelta

import torch.distributed as dist

_WORLD_SIZE = 1
_RANK = 0
_TP_GROUP = None
_TP_RANKS: list[int] = [0]
_EP_GROUP = None
_EP_SIZE = 1
_CPU_GROUP = None


def init_distributed(config, rank: int):
    """建 nccl 主组 + gloo 控制面组 + TP/EP 子组。

    dist.new_group 必须**所有 rank 以相同顺序调用**，哪怕自己不在那个组里，
    所以下面建单员 TP 组时是全员循环 world 遍、只留自己那个。
    """
    global _WORLD_SIZE, _RANK, _TP_GROUP, _TP_RANKS, _EP_GROUP, _EP_SIZE, _CPU_GROUP
    world = max(config.tensor_parallel_size, config.ep_size)
    _WORLD_SIZE, _RANK, _EP_SIZE = world, rank, config.ep_size

    # 原来这里是写死的 tcp://localhost:2333；master_addr 的默认值仍是 localhost:2333，
    # 所以单机路径的行为一个字节都没变，只有 EP 模式才会填成 192.168.100.2。
    dist.init_process_group("nccl", f"tcp://{config.master_addr}:{config.master_port}",
                            world_size=world, rank=rank)

    # 控制面走 gloo（CPU），同机跨机同一条代码路径
    # 给个显式超时：对端挂了要在有限时间内报错，而不是无限等下去。
    # gloo 默认 30 分钟，跨机调试时那等于挂死。
    timeout = timedelta(seconds=int(os.environ.get("NANOVLLM_GLOO_TIMEOUT", "180")))
    _CPU_GROUP = dist.new_group(backend="gloo", timeout=timeout) if world > 1 else None

    if config.ep_size > 1:
        for r in range(world):                       # 全员按同序建组，只留自己的
            g = dist.new_group([r])
            if r == rank:
                _TP_GROUP, _TP_RANKS = g, [r]
        _EP_GROUP = dist.group.WORLD
    else:
        _TP_GROUP, _TP_RANKS = dist.group.WORLD, list(range(world))
        _EP_GROUP = None


def get_rank() -> int:
    return _RANK


def get_world_size() -> int:
    return _WORLD_SIZE


def get_tp_rank() -> int:
    return _TP_RANKS.index(_RANK)


def get_tp_size() -> int:
    return len(_TP_RANKS)


def get_tp_group():
    return _TP_GROUP


def get_tp_src_rank() -> int:
    """TP 组内 rank0 的**全局** rank —— dist.gather 的 dst 要的是全局编号。"""
    return _TP_RANKS[0]


def get_ep_rank() -> int:
    return _RANK if _EP_GROUP is not None else 0


def get_ep_size() -> int:
    return _EP_SIZE


def get_ep_group():
    return _EP_GROUP


def get_cpu_group():
    return _CPU_GROUP
