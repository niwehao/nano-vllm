"""IBGDA/NVSHMEM 后端（M5）—— 忠实 DeepEP legacy internode_ll 内核的 SM89 移植。

当前状态：**未启用**。前置闸门（Plan-4/01-m0-environment.md 的 L4）在本环境未通过：

    PeerMappingOverride  未设（/proc/driver/nvidia/params 的 RegistryDwords 为空）
    gdrdrv               未加载，/dev/gdrdrv 不存在
    sudo                 无

IBGDA 的两种开启方式（DeepEP docs/nvshmem.md:36-61）都要 root：
  A. 两机 /etc/modprobe.d/nvidia.conf 加
     `options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"`
     + update-initramfs + 重启；
  B. 两机装 gdrcopy（加载 gdrdrv 模块），NVSHMEM 走 CPU 辅助 IBGDA
     （NVSHMEM_IBGDA_NIC_HANDLER=cpu，DeepEP 的 ibgda_device.cuh 里
      use_async_postsend 分支就是这个模式）。

环境就绪后要做的事全部写在 Plan-4/06-m5-ibgda-port.md：复制清单（哪些文件原样搬、
哪些手术）、combine 内核去 TMA 的逐段方案、setup.py、运行时环境变量表、排障预案。
布局与语义已经在 nccl_backend 里逐字节对齐，届时只需实现本类的 dispatch/combine，
上层（qwen3_moe.FusedExpertsEP / model_runner）一行不用改。
"""


class NvshmemBackend:

    def __init__(self, buf):
        raise NotImplementedError(
            "nvshmem/IBGDA 后端未启用：本机 PeerMappingOverride 未设、gdrdrv 未装、无 sudo。\n"
            "详见 Plan-4/artifacts/m0-report.md 的 IBGDA 结论，与 06-m5-ibgda-port.md 的移植方案。\n"
            "当前请用 transport='nccl'（数据面同样走 RoCE 上的 GPUDirect RDMA）。")

    def dispatch(self, x, topk_idx):
        raise NotImplementedError

    def combine(self, y, topk_idx, topk_weights, handle):
        raise NotImplementedError
