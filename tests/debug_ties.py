"""排查:同一行 logits 上 argmax / topk / top_k=1 给出不同结果,怀疑 bf16 并列最大值。"""
import torch
import common
from common import MODEL_PATH, build_prompts
from nanovllm import LLM, SamplingParams
from nanovllm.layers import sampler as sampler_mod

captured = []
orig = sampler_mod.Sampler.forward


def patched(self, logits, temperatures, top_ks=None, top_ps=None, max_logprobs=-1):
    lf = logits.float()
    mx = lf.max(dim=-1, keepdim=True).values
    n_tied = (lf == mx).sum(dim=-1)
    captured.append((n_tied.tolist(), lf.dtype, logits.dtype,
                     lf.argmax(dim=-1).tolist(),
                     lf.topk(1, dim=-1).indices.squeeze(1).tolist()))
    return orig(self, logits, temperatures, top_ks, top_ps, max_logprobs)


sampler_mod.Sampler.forward = patched

llm = LLM(MODEL_PATH, enforce_eager=True, gpu_memory_utilization=0.35)
prompts = build_prompts()
llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=3, ignore_eos=True), use_tqdm=False)

print(f"\n捕获 {len(captured)} 次 sampler 调用")
for i, (n_tied, fdt, odt, am, tk) in enumerate(captured[:4]):
    print(f"\n第 {i} 次: logits dtype={odt} -> {fdt}")
    print(f"  每行并列最大值个数: {n_tied}")
    print(f"  argmax : {am}")
    print(f"  topk(1): {tk}")
    print(f"  一致?   {am == tk}")
