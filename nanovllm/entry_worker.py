"""非 0 rank 的进程入口（跨机 EP 用）。

单机 TP 时 worker 是 llm_engine 用 mp.Process 在**本机** spawn 出来的；跨机没有这条
路，所以每台机器各自跑一个入口脚本，靠 --node-rank 区分身份。

worker 只做一件事：构造 ModelRunner。构造函数末尾发现 rank != 0 就自己进 loop()，
永不返回——之后一切听 rank0 通过 gloo 广播过来的指令。worker 不碰 tokenizer、
不碰 scheduler、拿到 logits 直接丢弃（model_runner.run 里 rank != 0 的分支）。

    .venv/bin/python -m nanovllm.entry_worker --model ~/huggingface/tiny-qwen3-moe \
        --ep-size 2 --node-rank 1 --master-addr 192.168.100.2 --enforce-eager
"""

import argparse
import os

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ep-size", type=int, default=2)
    ap.add_argument("--node-rank", type=int, required=True)
    ap.add_argument("--master-addr", default="localhost")
    ap.add_argument("--master-port", type=int, default=2333)
    ap.add_argument("--ep-transport", default="nccl")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--num-kvcache-blocks", type=int, default=-1)
    args = ap.parse_args()

    config = Config(
        os.path.expanduser(args.model),
        ep_size=args.ep_size,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        ep_transport=args.ep_transport,
        enforce_eager=args.enforce_eager,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
    )
    Sequence.block_size = config.kvcache_block_size     # 与 llm_engine 里的设置保持一致
    print(f"[worker rank={args.node_rank}] 启动，等 rank0 的指令…", flush=True)
    ModelRunner(config, args.node_rank)                 # 内部进 loop()，正常情况下不返回
    print(f"[worker rank={args.node_rank}] 收到 exit，退出", flush=True)


if __name__ == "__main__":
    main()
