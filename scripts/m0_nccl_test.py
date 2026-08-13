"""scripts/m0_nccl_test.py —— M0 的 L3：torch 栈上的双机 NCCL 实测。

两机各起一个进程（env:// 初始化）：
    gpu-02: RANK=0  gpu-01: RANK=1
校验 all_reduce 数值 + 扫 all_to_all_single 带宽 + gloo 控制面延迟。
判据: all_reduce 正确、大消息 busbw >= 10 GB/s、NCCL_DEBUG=INFO 日志里出现 GDRDMA。
"""
import os
import time

import torch
import torch.distributed as dist


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="env://", world_size=world, rank=rank)
    cpu_group = dist.new_group(backend="gloo")

    # 1) 数值正确性: 每 rank 填 rank+1，求和应为 world*(world+1)/2
    x = torch.full((1024,), float(rank + 1), device="cuda")
    dist.all_reduce(x)
    expect = world * (world + 1) / 2
    ok = bool((x == expect).all().item())
    if rank == 0:
        print(f"[L3-1] all_reduce  {'OK' if ok else 'FAIL'}  (got {x[0].item()}, want {expect})")

    # 2) all_to_all_single 带宽扫描
    if rank == 0:
        print(f"{'bytes':>12} {'ms':>9} {'algbw GB/s':>12} {'busbw GB/s':>12}")
    for mb in (1, 4, 16, 64, 256):
        n = mb * 1024 * 1024 // 2 // world * world      # bf16, 可整除
        send = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        recv = torch.empty_like(send)
        dt = bench(lambda: dist.all_to_all_single(recv, send))
        nbytes = send.numel() * 2
        algbw = nbytes / dt / 1e9
        busbw = algbw * (world - 1) / world             # a2a 的有效跨机流量
        if rank == 0:
            print(f"{nbytes:>12} {dt*1e3:>9.3f} {algbw:>12.2f} {busbw:>12.2f}")

    # 3) gloo 控制面延迟（M1 的每步广播用它，要知道代价）
    payload = [("run", ("x" * 512,))]
    buf = [None]
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(200):
        if rank == 0:
            dist.broadcast_object_list(list(payload), src=0, group=cpu_group)
        else:
            dist.broadcast_object_list(buf, src=0, group=cpu_group)
    dt = (time.perf_counter() - t0) / 200
    if rank == 0:
        print(f"[L3-3] gloo broadcast_object_list(~0.5KB): {dt*1e3:.3f} ms/次")

    dist.barrier()
    if rank == 0:
        print("[L3] done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
