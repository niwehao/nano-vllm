"""通用生成运行器:一个进程跑一种配置,结果落 JSON。

每种配置单独起进程,是因为 LLMEngine 会 init_process_group 并吃掉大块显存,
同进程内反复创建/销毁容易互相干扰。
"""
import argparse
import json
import os
import sys

import common
from common import MODEL_PATH, build_prompts, build_preempt_prompts
from nanovllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=-1)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--logprobs", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.35)
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--num-kvcache-blocks", type=int, default=-1)
    ap.add_argument("--speculative-model", type=str, default=None)
    ap.add_argument("--num-speculative-tokens", type=int, default=0)
    ap.add_argument("--speculative-method", type=str, default="model")
    ap.add_argument("--prompts", type=str, default="default",
                    help="default | single | pair | preempt")
    ap.add_argument("--repeat", type=int, default=1, help="把 prompt 集重复几遍(制造并发)")
    ap.add_argument("--pollution", action="store_true",
                    help="两阶段:先跑一遍(可能带投机),再用 prompt+前 32 个生成 token 当新 prompt "
                         "跑第二遍。第二遍必然命中第一遍写入的 block,用来查 prefix cache 污染")
    ap.add_argument("--prompt-file", type=str, default=None,
                    help="从 JSON 文件读取 prompt(配合 --pollution 产出的 phase2_prompts)")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    prompts = build_prompts()
    if args.prompts == "single":
        prompts = prompts[:1]
    elif args.prompts == "pair":
        prompts = prompts[2:4]      # long_a / long_b,共享 512 token 前缀
    elif args.prompts == "preempt":
        prompts = build_preempt_prompts()
    prompts = prompts * args.repeat
    if args.prompt_file:
        prompts = json.load(open(args.prompt_file))["phase2_prompts"]

    kwargs = dict(
        enforce_eager=args.eager,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        num_kvcache_blocks=args.num_kvcache_blocks,
    )
    if args.num_speculative_tokens > 0:
        kwargs["num_speculative_tokens"] = args.num_speculative_tokens
        kwargs["speculative_method"] = args.speculative_method
        if args.speculative_model:
            kwargs["speculative_model"] = os.path.expanduser(args.speculative_model)

    llm = LLM(MODEL_PATH, **kwargs)
    sp = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        logprobs=args.logprobs,
    )
    outputs = llm.generate(prompts, sp, use_tqdm=False)

    phase2_prompts = None
    if args.pollution:
        # 第二阶段:新 prompt = 原 prompt + 第一阶段前 32 个生成 token。
        # 这些位置的 KV 是第一阶段(可能由投机)写进 block 的,第二阶段必然命中它们。
        # 只要投机把"被拒绝位置的垃圾 KV"错误登记进了 prefix cache,这里就会读到错的 KV。
        phase2_prompts = [list(p) + o["token_ids"][:32] for p, o in zip(prompts, outputs)]
        # 第二阶段关掉投机:这样第二阶段的输出差异只可能来自"读到的缓存 KV",
        # 不会掺进投机自己那条 kernel 路径的浮点噪声,判据锐利得多。
        llm.scheduler.num_spec_tokens = 0
        outputs = llm.generate(phase2_prompts, sp, use_tqdm=False)

    stats = dict(getattr(llm.scheduler, "stats", {}))
    accepted = getattr(llm.scheduler, "spec_accepted", None)
    if accepted is not None:
        stats["spec_accepted"] = llm.scheduler.spec_accepted
        stats["spec_proposed"] = llm.scheduler.spec_proposed
        stats["spec_steps"] = llm.scheduler.spec_steps

    payload = {
        "config": vars(args),
        "stats": stats,
        "phase2_prompts": phase2_prompts,
        "outputs": [
            {"token_ids": o["token_ids"], "logprobs": o.get("logprobs")}
            for o in outputs
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {args.out}: {len(outputs)} seqs, "
          f"lens={[len(o['token_ids']) for o in outputs]}")
    if stats:
        print(f"  spec stats: {stats}")


if __name__ == "__main__":
    main()
