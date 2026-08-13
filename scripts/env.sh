# scripts/env.sh —— 两机共用的运行时环境变量，按 hostname 分支处理设备名差异。
# 用法: source scripts/env.sh
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/hosts.sh"

# ---- 控制面/数据面都锁在直连口，别绕 bond0 ----
export NCCL_SOCKET_IFNAME=$IFNAME
export GLOO_SOCKET_IFNAME=$IFNAME
export TP_SOCKET_IFNAME=$IFNAME

# ---- RoCE ----
export NCCL_IB_GID_INDEX=$GID_INDEX
export NCCL_IB_DISABLE=0
# GPU 与活跃网卡跨了 host bridge，NCCL 算出的距离是 8，超过 NCCL_NET_GDR_LEVEL 的
# 默认阈值 5(PXN) 就自动把 GDR 关了。放宽到 SYS 让它无视拓扑距离启用 GDR。
export NCCL_NET_GDR_LEVEL=SYS

case "$(hostname -s)" in
  inet-p4lab-gpu-02) HCA=$HCA_GPU02 ;;
  inet-p4lab-gpu-01) HCA=$HCA_GPU01 ;;
  *) echo "env.sh: 未知主机 $(hostname -s)" >&2 ;;
esac
export NCCL_IB_HCA=$HCA
export NVSHMEM_HCA_LIST="${HCA}:1"
export NVSHMEM_IB_GID_INDEX=$GID_INDEX
export NVSHMEM_DISABLE_P2P=1          # 每机单卡，没有 NVLink peer

# ---- 挂死变报错（跨机调试的必需品）----
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120

export MASTER_ADDR MASTER_PORT

# ---- 用户态 CUDA 12.8（M0 装的，M5 编 nanodeepep 扩展要用）----
# 机器上原本只有 /usr/local/cuda-13.3 的 runtime 库，连 nvcc 都没有；
# 且 13.x 与 torch 的 cu12.8 major 不匹配，本来也编不了扩展。
export CUDA_HOME=$HOME/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
