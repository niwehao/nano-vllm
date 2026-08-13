"""nanodeepep 双机验收（M2 判据 1-6）。也可用 world=1 单进程跑退化版。

用法（双机，各起一个进程）：
    gpu-02: RANK=0 WORLD_SIZE=2 MASTER_ADDR=192.168.100.2 MASTER_PORT=29500 \
            .venv/bin/python nanodeepep/tests/test_2rank.py
    gpu-01: RANK=1 ...（同上）
单机：
    .venv/bin/python nanodeepep/tests/test_2rank.py        # 自动 world=1
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanodeepep import NanoEPBuffer                                    # noqa: E402
from nanodeepep.tests.common import check_all, make_case               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", default="nccl")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--num-topk", type=int, default=2)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 7, 128, 512])
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--det-rounds", type=int, default=20)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(0)
    if world > 1:
        dist.init_process_group("nccl", init_method="env://", world_size=world, rank=rank)
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29777")
        dist.init_process_group("nccl", init_method="env://", world_size=1, rank=0)
    group = dist.group.WORLD

    E = args.num_experts
    assert E % world == 0
    buf = NanoEPBuffer(group, args.M, args.hidden, E, transport=args.transport)
    if rank == 0:
        print(f"=== nanodeepep 验收: {buf} ===")

    # ---- 判据 1/2/3/5：各 token 规模跑一遍 ----
    if rank == 0:
        print("[1/2/3/5] recv_count / 数据搬运 / 加权恒等式 / -1 与空手")
    for T in args.tokens:
        assert T <= args.M
        x, idx, w = make_case(T, args.hidden, E, args.num_topk, rank, seed=T)
        check_all(buf, x, idx, w, group, rank, world)

    # 一个 rank 空手（T=0），对端有数据 —— splits 全 0 也必须参与集合调用
    if world > 1:
        T = 0 if rank == 0 else 64
        x, idx, w = make_case(T, args.hidden, E, args.num_topk, rank, seed=99)
        # 注：两 rank 的 T 不同，check_all 里的 all_gather_into_tensor 需要等形状，
        # 所以这里只做"跑通 + 输出为 0"的弱检查
        packed, cnt, h = buf.low_latency_dispatch(x, idx)
        out = buf.low_latency_combine(packed.clone(), idx, w, h)
        assert out.shape == (T, args.hidden)
        dist.barrier()
        if rank == 0:
            print(f"    不等长 T（rank0=0, rank1=64）通过，rank0 输出 {tuple(out.shape)}")

    # ---- 判据 4：确定性 ----
    # 判据 4：确定性。
    # ⚠ 两个后端的确定性**语义不同**，这不是 bug：
    #   nccl 后端：packed_recv_x 里各 rank 的段按 **rank 升序**首尾相接（我们自己定的），
    #              所以连 packed 的哈希都稳定；
    #   nvshmem 后端：DeepEP 内核用 atomicAdd(packed_recv_count+l, n) 取 begin，
    #              段的先后是**到达序**，跑两遍可能不同 —— 但 layout_range 会如实记录，
    #              所以 **combined_x 仍然位级稳定**。实测正是如此：packed_hash 会飘，
    #              combined_hash 一模一样。
    # 因此这里只对 combined_hash 做硬判据，packed_hash 仅在 nccl 后端下检查。
    check_packed = args.transport == "nccl"
    if rank == 0:
        print(f"[4] 确定性：同 seed 连跑 {args.det_rounds} 遍"
              f"（combined_x 必须位级稳定；packed_recv_x 的段序"
              f"{'也稳定' if check_packed else '由 atomicAdd 竞争决定，不检查'}）")
    x, idx, w = make_case(args.tokens[-1], args.hidden, E, args.num_topk, rank, seed=7)
    ref = None
    for i in range(args.det_rounds):
        ph, ch = check_all(buf, x, idx, w, group, rank, world, verbose=False)
        key = (ph, ch) if check_packed else (ch,)
        if ref is None:
            ref = key
        assert key == ref, f"第 {i} 轮哈希漂移: {key} != {ref}"
    if rank == 0:
        tag = f"packed_hash={ref[0]:x} combined_hash={ref[1]:x}" if check_packed else f"combined_hash={ref[0]:x}"
        print(f"    {tag}  {args.det_rounds} 轮一致")

    # ---- 判据 6：带宽记录（非门槛，供 M6 对比）----
    if rank == 0:
        print("[6] 微基准（cuda event，20 次平均）")
        print(f"    {'T':>5} {'dispatch us':>13} {'combine us':>12} {'发送 MB':>9}")
    for T in args.tokens:
        x, idx, w = make_case(T, args.hidden, E, args.num_topk, rank, seed=T)
        packed, cnt, h = buf.low_latency_dispatch(x, idx)
        y = packed.clone()
        for _ in range(5):                       # 预热（首次建链/autotune）
            p, c, hh = buf.low_latency_dispatch(x, idx)
            buf.low_latency_combine(y, idx, w, hh)
        torch.cuda.synchronize()
        dist.barrier()

        def timeit(fn, iters=20):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize(); s.record()
            for _ in range(iters):
                fn()
            e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e) / iters * 1e3       # us

        td = timeit(lambda: buf.low_latency_dispatch(x, idx))
        p, c, hh = buf.low_latency_dispatch(x, idx)
        tc = timeit(lambda: buf.low_latency_combine(y, idx, w, hh))
        nbytes = T * args.num_topk * args.hidden * 2 / 1e6
        if rank == 0:
            print(f"    {T:>5} {td:>13.1f} {tc:>12.1f} {nbytes:>9.2f}")

    dist.barrier()
    if rank == 0:
        print("=== 全部通过 ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
