#!/usr/bin/env bash
# scripts/launch_both.sh —— EP 双机一键跑：sync → 远端起 worker → 本机跑负载 → 收尾清理。
#   scripts/launch_both.sh examples/ep_generate.py --max-tokens 32
#   MODEL=~/huggingface/tiny-qwen3-moe scripts/launch_both.sh tests/gen.py --out ... --ep-size 2 ...
#
# 本机 = rank 0 = driver（scheduler/采样都在这里），gpu-01 = rank 1 = worker
# （收 seqs → prepare → forward → 丢弃 logits，只在 MoE 层内参与 EP）。
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
SSH="ssh -o BatchMode=yes $GPU01_SSH"

MODEL=${MODEL:-$HFDIR/tiny-qwen3-moe}
EP_SIZE=${EP_SIZE:-2}
MNBT=${MNBT:-512}                       # max_num_batched_tokens，同时是 EP buffer 的 M
TRANSPORT=${TRANSPORT:-nccl}
KVBLOCKS=${KVBLOCKS:--1}
NOSYNC=${NOSYNC:-0}

# pkill 用 -x/-f 都要小心自匹配：这里用一个只出现在 worker 命令行里的标记串。
# 匹配 worker 用命令行里真实出现的模块名，中括号是防自匹配的老把戏：
# 正则 nanovllm[.]entry_worker 匹配得到 "nanovllm.entry_worker"，但 pkill 自己那条
# `bash -c "pkill -f 'nanovllm[.]entry_worker'"` 的命令行里写的是带中括号的版本，
# 匹配不上，于是不会把自己的父 shell 一起杀掉。
# 别用 `env MARK=1 python ...` 里的 MARK 当模式：那是**环境变量**，
# 而 pkill -f 匹配的是 /proc/pid/cmdline，env 赋值根本不在里面 —— 这个坑让
# cleanup 和"kill worker 看 driver 反应"的测试双双静默失效。
PAT='nanovllm[.]entry_worker'

# 调试开关必须**原样传给 worker**。只在 driver 侧开 NANOVLLM_EP_CHECK 的话，
# 只有 rank0 会去调那次 all_gather，rank1 直接走到下一个集合调用 —— 两边的集合
# 调用序列错开一位，双双卡到 gloo 超时才报错。凡是会改变集合调用次数的 env，
# 一律在这里透传。
FWD=""
for v in NANOVLLM_EP_CHECK NANOVLLM_EP_TIMING NANOVLLM_GLOO_TIMEOUT \
         TORCHDYNAMO_DISABLE TORCH_COMPILE_DISABLE; do
  [ -n "${!v:-}" ] && FWD="$FWD $v=${!v}"
done
cleanup() { $SSH "pkill -f '$PAT' >/dev/null 2>&1" >/dev/null 2>&1; }
trap cleanup EXIT INT TERM

[ "$NOSYNC" = "1" ] || "$here/sync.sh" >/dev/null

cleanup; sleep 0.5
$SSH "cd $REPO && source scripts/env.sh && \
  setsid nohup env $FWD .venv/bin/python -m nanovllm.entry_worker \
    --model $MODEL --ep-size $EP_SIZE --node-rank 1 \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --ep-transport $TRANSPORT --enforce-eager \
    --max-num-batched-tokens $MNBT --num-kvcache-blocks $KVBLOCKS \
    > /tmp/nano_ep_worker.log 2>&1 < /dev/null &" >/dev/null 2>&1 &
# 末尾这个 & 把**本地的 ssh 进程**也放后台，不能省：远端命令里的 & 只让远端
# 那条 bash 立刻返回，ssh 自己还会一直等远端会话的 fd 全部关闭才退出，
# 结果就是 driver 根本走不到下一行（排查过一轮：远端 worker 起来了、本机却卡住）。
sleep 3

cd "$REPO"
timeout "${EP_TIMEOUT:-1200}" .venv/bin/python "$@"
rc=$?

echo "----- rank1 (gpu-01) 日志尾部 -----"
$SSH "tail -20 /tmp/nano_ep_worker.log" 2>&1 || true
exit $rc
