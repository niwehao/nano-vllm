"""最小生成脚本:只用原始代码就有的 API(temperature=1e-9 近似 greedy),
用来对照"改动前后"同一现象是否存在。
用法: minimal_run.py <out.json> [key=value ...]
"""
import json
import sys

import common
from common import MODEL_PATH, build_prompts
from nanovllm import LLM, SamplingParams

out = sys.argv[1]
kw = {}
for a in sys.argv[2:]:
    k, v = a.split("=", 1)
    kw[k] = eval(v)
kw.setdefault("gpu_memory_utilization", 0.35)
llm = LLM(MODEL_PATH, **kw)
sp = SamplingParams(temperature=1e-9, max_tokens=64, ignore_eos=True)
outs = llm.generate(build_prompts(), sp, use_tqdm=False)
json.dump([o["token_ids"] for o in outs], open(out, "w"))
print(f"wrote {out} with {kw}")
