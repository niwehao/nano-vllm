"""NCCL 参考后端 —— 纯 torch.distributed 实现的 LL dispatch/combine。

三重身份：
  1. M4 能立即集成的可用实现（数据面 = NCCL over RoCE 的 GPUDirect RDMA）；
  2. M5 ibgda 内核的逐元素对拍 oracle；
  3. LL 语义（packed 布局 / recv_count / layout_range）的可读文档。

与 LL 内核的设计差异（有意为之，注释在此说明）：
  * LL 内核用**静态 M 上限**规避掉所有 CPU 同步——收多少 token 只写在 GPU 的
    recv_count 里，host 永远不读。本后端要用 all_to_all_single 的 splits 参数，
    而 splits 必须是 python list，所以**每次 dispatch 有且仅有一次 D2H 同步**
    （把 send/recv 计数一起搬下来那一下）。这是 NCCL 后端的固有代价。
  * 逐 k 的加权归约写成 scatter 到 [T,K,H] 再沿 k 求和，而不是 index_add_ ——
    index_add_ 在 CUDA 上走原子加，同一行的多个副本累加顺序不定，fp32 舍入会飘；
    scatter+sum 每个 (token,k) 槽只写一次、求和顺序固定，同输入必位级同输出。
"""

import torch
import torch.distributed as dist

from .buffer import EPHandle, pack2, unpack2


class NcclBackend:

    def __init__(self, buf):
        self.b = buf
        b = buf
        dev = torch.cuda.current_device()
        # 静态预分配，语义对齐 LL 的常驻 RDMA buffer（每层复用同一块）。
        # 零初始化一次：垃圾行不会是 NaN，调试时看得懂；之后不再清（对齐 DeepEP
        # 只在 clean_low_latency_buffer 时清的做法）。
        self.packed_recv_x = torch.zeros(b.L, b.R * b.M, b.H, dtype=torch.bfloat16, device=dev)
        self.src_info = torch.full((b.L, b.R * b.M), -1, dtype=torch.int32, device=dev)

    # ------------------------------------------------------------------ dispatch

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor):
        b = self.b
        L, R, M, H, E = b.L, b.R, b.M, b.H, b.E
        T, K = topk_idx.shape
        dev = x.device

        flat = topk_idx.reshape(-1)                                   # [T*K]
        valid = flat >= 0
        expert = flat[valid]                                          # [S] 全局专家号
        arange_t = torch.arange(T, device=dev, dtype=torch.int64)
        src_tok = arange_t.repeat_interleave(K)[valid]                # [S]
        k_idx = torch.arange(K, device=dev, dtype=torch.int64).repeat(T)[valid]

        # 排序键 = expert*(T+1) + src_tok。expert = dst_rank*L + dst_local，所以按
        # expert 排天然先按目的 rank 分组、组内再按本地专家分组；末位 src_tok 让同一
        # (rank, local_expert) 段内按源 token 升序 —— 确定性布局就来自这里。
        # 一个 token 不会重复选中同一专家，故键唯一，argsort 结果与实现无关。
        order = torch.argsort(expert * (T + 1) + src_tok, stable=True)
        send_tok = src_tok[order]
        send_k = k_idx[order]
        send_x = x[send_tok]                                          # [S, H]
        send_meta = send_tok.to(torch.int32)                          # 源 token 下标，回程用

        # 计数交换：把 [E] 的"我发给每个全局专家多少"等分成 R 份 all_to_all，
        # 换回来就是 recv_cnt_flat[r*L+l] = rank r 发给我本地专家 l 多少。
        send_cnt_e = torch.bincount(expert, minlength=E)               # [E] int64
        recv_cnt_e = torch.empty_like(send_cnt_e)
        dist.all_to_all_single(recv_cnt_e, send_cnt_e, group=b.group)
        both = torch.stack([send_cnt_e, recv_cnt_e]).cpu()             # ← 唯一的 D2H 同步
        send_cnt_flat, recv_cnt_flat = both[0].tolist(), both[1].tolist()

        send_splits = [sum(send_cnt_flat[r * L:(r + 1) * L]) for r in range(R)]
        recv_splits = [sum(recv_cnt_flat[r * L:(r + 1) * L]) for r in range(R)]
        recv_cnt = [[recv_cnt_flat[r * L + l] for l in range(L)] for r in range(R)]

        # 数据与元信息同序到达
        n_recv = sum(recv_splits)
        recv_x = x.new_empty(n_recv, H)
        recv_meta = torch.empty(n_recv, dtype=torch.int32, device=dev)
        dist.all_to_all_single(recv_x, send_x, recv_splits, send_splits, group=b.group)
        dist.all_to_all_single(recv_meta, send_meta, recv_splits, send_splits, group=b.group)

        # 落 packed 布局。rank r 送来的整块已按 (local_expert, src_tok) 排好序，
        # 所以本地专家 l 的子块就是块内的一段连续切片——纯切片，无掩码、无额外同步。
        packed, src_info = self.packed_recv_x, self.src_info
        layout_cpu = [[0] * R for _ in range(L)]
        begin = [0] * L
        off = 0
        for r in range(R):                                   # R=2、L=2 的双层小循环是
            sub = 0                                          # 刻意的可读性取舍，别过早张量化
            for l in range(L):
                n = recv_cnt[r][l]
                layout_cpu[l][r] = pack2(n, begin[l])
                if n:
                    packed[l, begin[l]:begin[l] + n] = recv_x[off + sub:off + sub + n]
                    src_info[l, begin[l]:begin[l] + n] = recv_meta[off + sub:off + sub + n]
                    begin[l] += n
                    sub += n
            off += recv_splits[r]

        layout_range = torch.tensor(layout_cpu, dtype=torch.int64, device=dev)
        recv_count = torch.tensor(begin, dtype=torch.int32, device=dev)
        handle = EPHandle(src_info, layout_range, M, H, E,
                          send_splits, recv_splits, recv_cnt, send_tok, send_k, T)
        return packed, recv_count, handle

    # ------------------------------------------------------------------- combine

    def combine(self, y: torch.Tensor, topk_idx: torch.Tensor,
                topk_weights: torch.Tensor, handle: EPHandle):
        b = self.b
        L, R, H = b.L, b.R, b.H
        T, K = topk_idx.shape
        dev = y.device
        recv_cnt, send_splits, recv_splits = handle.recv_cnt, handle.send_splits, handle.recv_splits

        # 1) 逆着 dispatch 的落位把专家输出收集回"到达序"
        n_recv = sum(recv_splits)
        send_back = y.new_empty(n_recv, H)
        off = 0
        for r in range(R):
            sub = 0
            for l in range(L):
                n = recv_cnt[r][l]
                if n:
                    # begin 直接由 recv_cnt 重算，不去读 GPU 上的 layout_range
                    # （读它会引入一次 D2H 同步）。段按 rank 升序首尾相接是本后端的
                    # 布局约定，两处必须保持一致。
                    beg = sum(recv_cnt[rr][l] for rr in range(r))
                    send_back[off + sub:off + sub + n] = y[l, beg:beg + n]
                    sub += n
            off += recv_splits[r]

        # 2) 沿原路回传（splits 互换）
        n_send = sum(send_splits)
        back = y.new_empty(n_send, H)
        dist.all_to_all_single(back, send_back, send_splits, recv_splits, group=b.group)

        # 3) 加权归约。权重乘在**接收侧**——DeepEP 语义：combine 发送的是专家原始
        #    输出，权重在归约时才乘（internode_ll.cu 的 decode_and_accumulate）。
        tok, kk = handle.order_tok, handle.order_k
        w = topk_weights[tok, kk]                                  # [S] fp32
        slots = torch.zeros(T, K, H, dtype=torch.float32, device=dev)
        if n_send:
            slots[tok, kk] = back.float() * w.unsqueeze(1)         # 每槽至多写一次
        return slots.sum(dim=1).to(y.dtype)                        # 沿 k 升序求和，确定性

    def destroy(self):
        self.packed_recv_x = None
        self.src_info = None
