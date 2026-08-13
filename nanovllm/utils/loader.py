import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# MoE 的专家权重。transformers 落盘时把内存里的 [E, 2I, H] 拆成每专家一份，
# 所以 checkpoint 里是 model.layers.N.mlp.experts.3.gate_proj.weight 这种老格式。
_EXPERT_RE = re.compile(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # experts 分支必须排在 packed mapping **之前**：packed mapping 按子串
                # 匹配，"experts.3.gate_proj" 里也含 "gate_proj"，会被误当成 dense MLP
                # 的 gate_up_proj 去 get_parameter，直接报找不到参数。
                m = _EXPERT_RE.match(weight_name)
                if m is not None:
                    prefix, expert_id, shard = m.group(1), int(m.group(2)), m.group(3)
                    experts = model.get_submodule(prefix + ".experts")
                    # EP 过滤在 load_expert_weight 内部：不属于本 rank 的专家直接
                    # return，权重根本不落显存。
                    experts.load_expert_weight(f.get_tensor(weight_name), expert_id, shard)
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
