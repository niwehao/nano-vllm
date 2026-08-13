# scripts/hosts.sh —— 双机拓扑的唯一事实源，其余脚本一律 source 本文件。
# 实测依据见 Plan-4/00-overview.md 的「环境事实清单」。

# 远端（rank 1）的 ssh 目标。直连口(100GbE)，比管理网 bond0(2GbE) 快一个量级；
# 管理网名 inet-p4lab-gpu-01.mpi-inf.mpg.de 仍可用，只是同步 7.5G venv 会慢很多。
GPU01_SSH=192.168.100.1

# 直连私网（无交换机路径干扰）。MASTER_ADDR 必须用这个而不是 hostname，
# 否则会解析到 bond0 的 139.19.59.x 公网口。
GPU01_IP=192.168.100.1
GPU02_IP=192.168.100.2          # 本机 = rank 0 = driver = MASTER

# 两机的直连网卡名恰好相同
IFNAME=ens5f0np0

# verbs 设备名两机不同（udev 命名策略差异），任何按名字选卡的 env 都要分支
HCA_GPU02=mlx5_0
HCA_GPU01=rocep66s0f0

# RoCE v2 + IPv4-mapped GID 的下标（两机实测均为 3）
GID_INDEX=3

# 两机上的仓库/权重/venv 路径（保持完全一致，sync.sh 单向覆盖）
REPO=$HOME/CodeRead/vllm/nano-vllm
VENV=$HOME/CodeRead/nano-vllm/.venv          # 仓库里的 .venv 是指向它的符号链接
UVPY=$HOME/.local/share/uv/python            # venv 依赖的解释器实体
HFDIR=$HOME/huggingface

MASTER_ADDR=$GPU02_IP
MASTER_PORT=29500
