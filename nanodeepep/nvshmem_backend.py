"""IBGDA/NVSHMEM 后端 —— DeepEP legacy internode_ll 内核的 SM89 移植。

内核代码在 `csrc/legacy/`，是从 `DeepEP/csrc/kernels/legacy/` 复制过来的，手术处都带
`[nano-deepEP]` 注释。手术清单（详见 Plan-4/08-implementation-report.md）：

  * dispatch 内核 **零手术**：FP8 cvt、bar.sync 命名屏障、cg::this_grid().sync()、
    atomicMax(ull)、st.release/ld.acquire.sys 在 SM89 上全部可用。
  * combine 内核发送侧：TMA 三段流水 → UNROLLED_WARP_COPY。
  * combine 内核接收侧：TMA producer/consumer 双角色 warp 组 → 每 warp 一个 token 的
    朴素 fp32 归约（按 k 升序，与 nccl 后端同序，两个后端才能位级对拍）。
  * LogFMT 整体删除（它调用 tma_store_fence，那是 SM90 专有；函数模板里的非依赖名
    在定义处就要查找，留着不用也编译不过）。
  * launch.cuh：cooperative 保留（两个内核都有 grid 级同步）、cluster attr 去掉。
  * utils.cuh / compiled.cuh：靠 -DDISABLE_SM90_FEATURES 拿 elect_one_sync 的 lane0
    回退和 TMA 段的条件编译；但 FP8 分支要恢复（SM89 原生支持）。

环境变量在 buffer.py 的 `_set_nvshmem_env()` 里设，必须在 import _C 之前生效。
"""

import os

import torch
import torch.distributed as dist

from .buffer import EPHandle


def set_nvshmem_env(num_local_experts: int, m: int):
    """照抄 deep_ep/buffers/legacy.py:104-122，另外修正 3.x 的通道选择开关。"""
    # ⚠ DeepEP 设的 NVSHMEM_IB_ENABLE_IBGDA=1 是 NVSHMEM **2.x** 的开关。本机装的是
    # 3.7.2，选通道要用 NVSHMEM_REMOTE_TRANSPORT=ibgda —— 只设老开关的话 NVSHMEM 会
    # 默默选 ibrc（CPU 代理），功能全对但根本不是 IBGDA。用 NVSHMEM_DEBUG=INFO 看
    # "Successfully initialized the transport:" 那行才能确认。
    os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "ibgda")
    os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"
    # QP 数 = 本地专家数（legacy.py:110 的要求）
    os.environ.setdefault("NVSHMEM_IBGDA_NUM_RC_PER_PE", str(num_local_experts))
    # 断言是 >= (M+1)*2，M=512 时默认的 1024 必炸（legacy.py:609）
    os.environ.setdefault("NVSHMEM_QP_DEPTH", str(max(1024, 2 * (m + 1))))
    os.environ.setdefault("NVSHMEM_DISABLE_P2P", "1")       # 每机单卡，无 NVLink peer
    os.environ.setdefault("NVSHMEM_CUMEM_GRANULARITY", str(2 ** 29))   # legacy.py:122（init 至少要 256MiB）
    os.environ.setdefault("NVSHMEM_MAX_TEAMS", "7")        # legacy.py:118，6 个默认 team + 1
    os.environ.setdefault("NVSHMEM_DISABLE_NVLS", "1")     # legacy.py:120，关掉 NVLink SHARP


class NvshmemBackend:

    def __init__(self, buf):
        self.b = buf
        set_nvshmem_env(buf.L, buf.M)
        from . import _C
        self._C = _C

        # 初始化协议照抄 legacy.py:104-136：rank0 生成 unique id → 广播 → 各 rank init。
        # 用 buffer 的 group 做广播；它是 nccl 组，broadcast_object_list 会走 gloo 的
        # 默认组，所以这里显式用 CPU 组（没有就退回默认组）。
        payload = [_C.get_unique_id() if buf.rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        _C.init(payload[0], buf.rank, buf.R)
        _C.buffer_create(buf.rank, buf.R, buf.E, buf.H, buf.M)
        self.nbytes = _C.buffer_bytes()
        _C.barrier()

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor):
        packed, count, src_info, layout_range = self._C.buffer_dispatch(x, topk_idx)
        handle = EPHandle(src_info, layout_range, self.b.M, self.b.H, self.b.E,
                          num_tokens=x.size(0))
        return packed, count, handle

    def combine(self, y: torch.Tensor, topk_idx: torch.Tensor,
                topk_weights: torch.Tensor, handle: EPHandle):
        return self._C.buffer_combine(y, topk_idx, topk_weights,
                                      handle.src_info, handle.layout_range)

    def destroy(self):
        self._C.buffer_destroy()
