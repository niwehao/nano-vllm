#!/usr/bin/env bash
# scripts/m0_nettest.sh —— M0 的 L1/L2 分层实测（RDMA host / RDMA GPU-direct）。
# 在 gpu-02 上跑，脚本自己 ssh 到 gpu-01 起服务端。
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/hosts.sh"
SSH="ssh -o BatchMode=yes $GPU01_SSH"

run_pair () {                       # $1=工具 $2=服务端额外参数 $3=客户端额外参数 $4=标题
  echo "=============== $4 ==============="
  # pkill 一定要用 -x（按进程名精确匹配）。用 -f 匹配整条命令行的话，ssh 起的
  # `bash -c "pkill -f ib_write_bw; ... ib_write_bw ..."` 会匹配到自己，
  # 服务端还没起来就先把自己的父 shell 杀了 —— 这个坑排查了一轮。
  # setsid + </dev/null 让服务端脱离 ssh 会话，别跟着断开一起死。
  $SSH "pkill -x $1 >/dev/null 2>&1; sleep 0.3; setsid nohup $1 -d $HCA_GPU01 -x $GID_INDEX $2 > /tmp/m0_srv.log 2>&1 < /dev/null &"
  sleep 2
  timeout 90 $1 -d $HCA_GPU02 -x $GID_INDEX $3 $GPU01_IP 2>&1 | tail -12
  $SSH "pkill -x $1 >/dev/null 2>&1" ; sleep 0.5
}

# L1: 纯 RDMA，host 内存。判据 BW >= 90 Gb/s
run_pair ib_write_bw "--report_gbits -D 5" "--report_gbits -D 5" "L1 RDMA write (host mem)"
run_pair ib_send_bw  "--report_gbits -D 5" "--report_gbits -D 5" "L1 RDMA send  (host mem)"

# L1': RDMA atomic —— M5 的 ibgda 计数通知(MLX5_OPCODE_ATOMIC_MASKED_FA)能否工作的前置信号
run_pair ib_atomic_bw "-A FETCH_AND_ADD" "-A FETCH_AND_ADD" "L1' RDMA atomic fetch-add"

# L2: GPU 显存 GDR。判据 >= 80 Gb/s
# --use_cuda_dmabuf 不能省：不加它 perftest 走老的 ibv_reg_mr(CUDA VA) 路线，
# 那条路需要 nvidia_peermem 模块，本机没有 → "failed to create mr"。
# dmabuf 路线由内核 6.18 + 驱动 595/610 提供，不需要任何额外内核模块。
run_pair ib_write_bw "--report_gbits -D 5 --use_cuda=0 --use_cuda_dmabuf" \
                     "--report_gbits -D 5 --use_cuda=0 --use_cuda_dmabuf" \
         "L2 RDMA write (GPU mem, GPUDirect via dmabuf)"

echo "=============== NIC 型号 ==============="
lspci | grep -i mellanox | head -4
