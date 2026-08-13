"""生成对拍用的 tiny Qwen3-MoE 权重（一次入库，之后不再重生成）。

seed 写死 42：所有 EP 对拍基线都建立在"两边加载的是同一份权重"之上，权重每变一次
基线就得全部重跑，所以生成一次就固定下来。

尺寸选择的理由：
  hidden_size=2048   —— DeepEP LL 内核的 SWITCH_HIDDEN 有现成的 2048 分支
                        (launch.cuh:115)，且 2048 % 512 == 0 满足 LL FP8 的 host
                        断言 (buffer.hpp:1530)。换别的宽度 M5 就得改内核模板。
  num_experts=4, top-2, ep_size=2 —— 每台机器 2 个本地专家，最小的"真跨机 EP"。
  num_hidden_layers=4 —— 每步 4 次 dispatch + 4 次 combine，够看出通信开销，又不慢。
  vocab_size=151936  —— 与 Qwen3-0.6B 同分词器，可以直接复用 tests/common.py 的语料。

参数量 ~0.72B，大头是 embed + lm_head 各 151936×2048=0.31B，MoE 主干很小。
"""

import os
import shutil

import torch
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

OUT = os.path.expanduser("~/huggingface/tiny-qwen3-moe")
TOKENIZER_SRC = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def main():
    cfg = Qwen3MoeConfig(
        hidden_size=2048,
        num_hidden_layers=4,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=4096,          # dense MLP 尺寸；mlp_only_layers 为空时用不到
        moe_intermediate_size=512,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],              # 每层都是 MoE
        norm_topk_prob=True,
        vocab_size=151936,
        tie_word_embeddings=False,       # 显式关掉，别走 qwen3.py 的 tie 分支
        attention_bias=False,            # → qkv_bias=False，q_norm/k_norm 生效
        max_position_embeddings=4096,
        rope_theta=1000000.0,
        dtype="bfloat16",
    )
    torch.manual_seed(42)
    model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n/1e9:.3f}B")

    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)

    # 分词器直接从 Qwen3-0.6B 拷（同家族），worker 侧用不到，rank0 编码 prompt 要用
    tok = AutoTokenizer.from_pretrained(TOKENIZER_SRC)
    tok.save_pretrained(OUT)

    print(f"已写入 {OUT}")
    from safetensors import safe_open
    from glob import glob
    for f in sorted(glob(os.path.join(OUT, "*.safetensors"))):
        with safe_open(f, "pt", "cpu") as h:
            keys = list(h.keys())
        print(f"{os.path.basename(f)}: {len(keys)} 个张量")
        for k in keys:
            if "layers.0." in k or "layers." not in k:
                print("   ", k)
        break


if __name__ == "__main__":
    main()
