"""rank0 侧的 EP 端到端示例（rank1 跑 nanovllm.entry_worker）。

    scripts/launch_both.sh examples/ep_generate.py --max-tokens 32

两台机器复制运行同一批（attention 重复计算），MoE 层内做 EP：各自把**相同的**
[T,H] token 按路由 dispatch，专家算完 combine 回各自的源 rank，两边得到相同的
combined_x，前向继续。正确性与"专家收到 2 份同 token"无冲突——每份都 combine 回
自己的源 rank。冗余代价见 Plan-4 的报告。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/huggingface/tiny-qwen3-moe")
    ap.add_argument("--ep-size", type=int, default=2)
    ap.add_argument("--node-rank", type=int, default=0)
    ap.add_argument("--master-addr", default="192.168.100.2")
    ap.add_argument("--master-port", type=int, default=2333)
    ap.add_argument("--ep-transport", default="nccl")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    args = ap.parse_args()

    llm = LLM(os.path.expanduser(args.model),
              ep_size=args.ep_size,
              node_rank=args.node_rank,
              master_addr=args.master_addr,
              master_port=args.master_port,
              ep_transport=args.ep_transport,
              enforce_eager=True,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_model_len=4096,
              gpu_memory_utilization=0.5)

    prompts = ["The key idea behind expert parallelism is",
               "GPUDirect RDMA lets the network card"]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    for p, o in zip(prompts, outs):
        print(f"\nPrompt: {p!r}\nOutput: {o['text']!r}")
    print(f"\nscheduler stats: {llm.scheduler.stats}")


if __name__ == "__main__":
    main()
