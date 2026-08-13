"""M5 第一道关：NVSHMEM/IBGDA 环境冒烟（**不涉及任何 DeepEP 内核移植**）。

刻意与后面的内核对拍分开：第一次跑不通是常态，这一层能把
"NVSHMEM 在 RoCE + L40S + 开放内核驱动上能不能用" 和 "我们移植 internode_ll 有没有错"
两件事分开归因。这条过了，后面出问题就一定是移植的锅。

    RUN2_TIMEOUT=300 ./scripts/run2.sh nanodeepep/tests/test_ibgda_smoke.py

初始化协议照抄 deep_ep/buffers/legacy.py:104-136：
rank0 生成 unique id → 经 gloo 组广播 → 各 rank 用它调 nvshmemx_init_attr。
"""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def set_ibgda_env():
    """照抄 legacy.py:104-122 的 env 设置。必须在 import _C / init 之前设好。

    ⚠ DeepEP 只设 NVSHMEM_IB_ENABLE_IBGDA=1，那是 NVSHMEM **2.x** 的开关。
    本机装的是 3.7.2，选通道要用 NVSHMEM_REMOTE_TRANSPORT=ibgda —— 只设老开关的话
    NVSHMEM 会默默选 ibrc（CPU 代理通道），冒烟照样"通过"，但测的根本不是 IBGDA。
    用 NVSHMEM_DEBUG=INFO 看日志里的 "Selected remote transport:" 那行才能确认。
    """
    os.environ["NVSHMEM_REMOTE_TRANSPORT"] = os.environ.get("NVSHMEM_REMOTE_TRANSPORT", "ibgda")
    os.environ["NVSHMEM_IB_ENABLE_IBGDA"] = "1"          # 2.x 兼容，留着无害
    os.environ["NVSHMEM_IBGDA_NUM_RC_PER_PE"] = os.environ.get(
        "NVSHMEM_IBGDA_NUM_RC_PER_PE", "2")          # = num_local_experts
    # QP 深度断言是 >= (M+1)*2，M=512 时默认 1024 不够，必炸（legacy.py:609）
    os.environ["NVSHMEM_QP_DEPTH"] = os.environ.get("NVSHMEM_QP_DEPTH", "2048")
    os.environ["NVSHMEM_DISABLE_P2P"] = "1"          # 每机单卡，无 NVLink peer
    os.environ["NVSHMEM_CUMEM_GRANULARITY"] = str(2 ** 29)   # legacy.py:122 照抄


def main():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    set_ibgda_env()
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", init_method="env://", world_size=world, rank=rank)

    from nanodeepep import _C

    # rank0 生成 unique id，广播给所有人
    payload = [_C.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    uid = payload[0]

    if rank == 0:
        print(f"=== NVSHMEM 冒烟：world={world} ===")
        print(f"    unique_id {len(uid)} 字节")
        print(f"    REMOTE_TRANSPORT={os.environ['NVSHMEM_REMOTE_TRANSPORT']} "
              f"QP_DEPTH={os.environ['NVSHMEM_QP_DEPTH']} "
              f"HCA_LIST={os.environ.get('NVSHMEM_HCA_LIST')} "
              f"GID={os.environ.get('NVSHMEM_IB_GID_INDEX')}")

    pe = _C.init(uid, rank, world)
    print(f"[rank{rank}] nvshmem init OK: my_pe={pe}/{_C.n_pes()}", flush=True)
    _C.barrier()

    ok_all = True
    for nelem in (1, 1024, 256 * 1024):
        ok = _C.put_test(nelem)
        ok_all &= ok
        if rank == 0:
            print(f"    device 侧 put {nelem * 4 / 1024:>9.1f} KB : {'OK' if ok else 'FAIL'}")
    _C.barrier()

    if rank == 0:
        print("=== 全部通过 ===" if ok_all else "=== 有失败 ===")
    _C.finalize()
    dist.destroy_process_group()
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
