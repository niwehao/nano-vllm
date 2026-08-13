#!/usr/bin/env bash
# scripts/sync.sh —— gpu-02 → gpu-01 单向同步。约定：gpu-01 上不手工改任何东西。
#   ./sync.sh          代码 + 权重（默认，秒级）
#   ./sync.sh --venv   连 .venv 和 uv 解释器一起（7.5G，首次/换包时才需要）
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/hosts.sh"

# 走直连口传(100GbE)，ssh 目标已在 hosts.sh 设成 192.168.100.1
# 用函数而不是变量：变量里的 "-e ssh -o BatchMode=yes" 会被词法拆开，
# rsync 只吃到 -e ssh，剩下的 -o BatchMode=yes 被当成源路径。
SSH() { ssh -o BatchMode=yes "$@"; }
R() { rsync -a --info=stats1 -e "ssh -o BatchMode=yes" "$@"; }

echo ">>> 代码"
SSH $GPU01_SSH "mkdir -p $REPO $HFDIR"
R --delete --exclude .git --exclude .venv --exclude 'tests/out' \
   --exclude '__pycache__' --exclude '*.pyc' \
   "$REPO/" "$GPU01_SSH:$REPO/"
# 仓库里的 .venv 是符号链接，rsync 排除后在远端重建（指向同一绝对路径）
SSH $GPU01_SSH "ln -sfn $VENV $REPO/.venv"

echo ">>> 权重"
for m in Qwen3-0.6B tiny-qwen3-moe; do
  [ -d "$HFDIR/$m" ] && R --delete "$HFDIR/$m/" "$GPU01_SSH:$HFDIR/$m/" || true
done

if [ "${1:-}" = "--venv" ]; then
  echo ">>> uv 解释器（venv 的 home 指向它）"
  SSH $GPU01_SSH "mkdir -p $UVPY"
  R --delete "$UVPY/" "$GPU01_SSH:$UVPY/"
  echo ">>> .venv（7.5G，耐心等）"
  SSH $GPU01_SSH "mkdir -p $VENV"
  R --delete "$VENV/" "$GPU01_SSH:$VENV/"
fi

echo ">>> 冒烟"
SSH $GPU01_SSH "$VENV/bin/python -c \"
import torch, transformers
print('gpu-01:', torch.__version__, torch.cuda.get_device_name(0), 'nccl', torch.cuda.nccl.version(), 'tf', transformers.__version__)\""
