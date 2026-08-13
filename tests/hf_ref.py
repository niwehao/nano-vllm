"""HF transformers 参考实现的单步 logits / greedy 生成，产出与 gen.py 同结构的 JSON。

单开一个进程跑，理由与 gen.py 一样：HF 模型和 nano 引擎都吃大块显存，同进程里
一起活着会互相挤，而且 nano 侧 init_process_group 之后的全局状态不好清。

对拍要求 attn_implementation="eager" + bf16：sdpa/flash 的数值路径与 nano 的
flash-attn 不同，混进来会污染 4-ulp 判据的归因。
"""

import argparse
import json
import os

import torch

import common
from common import build_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=0, help=">0 时做 greedy 生成")
    ap.add_argument("--logprobs", type=int, default=0, help=">0 时输出下一步 top-k logprob")
    ap.add_argument("--prompts", default="default")
    ap.add_argument("--router-probe", action="store_true",
                    help="用 hook 抓每层的 topk_idx，落进 JSON 供路由一致性比对")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    path = os.path.expanduser(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()

    prompts = build_prompts()
    if args.prompts == "single":
        prompts = prompts[:1]
    elif args.prompts == "pair":
        prompts = prompts[2:4]
    elif args.prompts == "long":
        prompts = prompts[2:3]

    # 路由探针：挂在每个 Qwen3MoeTopKRouter 上，抓它返回的 router_indices
    routes: list[list] = []
    if args.router_probe:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
        cur: list = []

        def hook(mod, inp, out):
            # out = (router_logits, router_scores, router_indices)
            probs = torch.softmax(out[0], dim=-1, dtype=torch.float32)
            v3, _ = torch.topk(probs, min(mod.top_k + 1, probs.size(-1)), dim=-1)
            cur.append({"idx": out[2].tolist(), "topv": v3.tolist()})
        for m in model.modules():
            if isinstance(m, Qwen3MoeTopKRouter):
                m.register_forward_hook(hook)

    outputs = []
    for p in prompts:
        ids = torch.tensor([p], device="cuda")
        if args.router_probe:
            cur.clear()
        with torch.inference_mode():
            logits = model(ids).logits[0, -1].float()      # 只要最后一个位置的下一步分布
        if args.router_probe:
            routes.append(list(cur))
        lp = torch.log_softmax(logits, dim=-1)
        k = args.logprobs or 10
        top = torch.topk(lp, k)
        # 结构与 gen.py 的 payload 对齐，好让 harness.compare_logprobs 直接吃
        entry = {
            "logprobs": [{
                "token_id": int(logits.argmax().item()),
                "top_logprobs": [[int(i), float(v)]
                                 for i, v in zip(top.indices.tolist(), top.values.tolist())],
            }],
            "token_ids": [],
            "logit_absmax": float(logits.abs().max().item()),   # 用来算这个模型的 bf16 ulp
        }
        if args.max_tokens > 0:
            with torch.inference_mode():
                gen = model.generate(ids, max_new_tokens=args.max_tokens, do_sample=False,
                                     temperature=None, top_p=None, top_k=None,
                                     pad_token_id=0)
            entry["token_ids"] = gen[0, len(p):].tolist()
        outputs.append(entry)

    payload = {"config": vars(args), "stats": {}, "routes": routes, "outputs": outputs}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {args.out}: {len(outputs)} seqs, "
          f"lens={[len(o['token_ids']) for o in outputs]}")


if __name__ == "__main__":
    main()
