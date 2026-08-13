#!/usr/bin/env bash
# scripts/run2.sh —— 双机跑同一条 python 命令（env:// 两 rank），rank1 在 gpu-01。
#   用法: scripts/run2.sh nanodeepep/tests/test_2rank.py --transport nccl
#   本机是 rank 0（driver，输出打在这里）；远端 rank 1 的日志落 /tmp/nano_rank1.log，
#   结束后自动回显尾部。trap 里兜底清理远端进程，别留僵尸占着显存。
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
SSH="ssh -o BatchMode=yes $GPU01_SSH"

# 清理用一个只出现在 worker 命令行里的标记串。别用 pkill -f 匹配脚本名或 "python"：
# ssh 起进程的那条 `bash -c "...python ..."` 命令行里也含同样的字，pkill 会连自己的
# 父 shell 一起杀掉（M0 排查 perftest 时就栽在这上面）。
# 用脚本路径当模式，首字母套中括号防自匹配（见 launch_both.sh 的同款注释）
PAT="[${1:0:1}]${1:1}"
cleanup() { $SSH "pkill -f '$PAT' >/dev/null 2>&1" >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

cleanup; sleep 0.5
$SSH "cd $REPO && source scripts/env.sh && \
  setsid nohup env RANK=1 WORLD_SIZE=2 \
    MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT PYTHONPATH=$REPO \
    ${RUN2_LAUNCHER:-} .venv/bin/python $* > /tmp/nano_rank1.log 2>&1 < /dev/null &" >/dev/null 2>&1 &
# 末尾这个 & 把**本地的 ssh 进程**也放后台，不能省：远端命令里的 & 只让远端
# 那条 bash 立刻返回，ssh 自己还会一直等远端会话的 fd 全部关闭才退出，
# 结果就是 driver 根本走不到下一行（排查过一轮：远端 worker 起来了、本机却卡住）。
sleep 3

export RANK=0 WORLD_SIZE=2 PYTHONPATH=$REPO
cd "$REPO"
# RUN2_LAUNCHER 可以塞 compute-sanitizer 之类的包装器（两端都会用上）
timeout "${RUN2_TIMEOUT:-900}" ${RUN2_LAUNCHER:-} .venv/bin/python "$@"
rc=$?

echo "----- rank1 (gpu-01) 日志尾部 -----"
$SSH "tail -25 /tmp/nano_rank1.log" 2>&1 || true
exit $rc
