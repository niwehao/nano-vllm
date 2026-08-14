# M3 · MoE 模型层（qwen3_moe.py，先做 EP=1 本地版）

目标：nano-vllm 能加载并运行 **Qwen3-MoE 架构**（`Qwen3MoeForCausalLM`），4 专家 top-2，单机 EP=1 下与 HF transformers 逐位对拍。EP 切分的钩子（每 rank 只持有本地专家权重）在本里程碑就位，但 dispatch/combine 到 M4 才接。

不依赖 M1/M2，可与它们并行。

## 现状

- `models/qwen3.py:91` 只有 dense `Qwen3MLP`；`model_runner.py:9` 硬编码 `Qwen3ForCausalLM`。
- venv 里 transformers 5.14.1，`Qwen3MoeConfig/Qwen3MoeForCausalLM` 可用（已验证 import）。
- attention/RMSNorm/rope/embed/lm_head 全部可原样复用（Qwen3-MoE 的 attention 与 Qwen3 相同，含 q_norm/k_norm）。

## 任务 1 · 生成 tiny 权重：`tools/make_tiny_qwen3_moe.py`

```python
cfg = Qwen3MoeConfig(
    hidden_size=2048,            # ← 关键：SWITCH_HIDDEN 已有 2048 case（launch.cuh:115）
                                 #    且 %512==0 满足 LL FP8 的 host 断言（buffer.hpp:1530）
    num_hidden_layers=4,
    num_attention_heads=16, num_key_value_heads=8, head_dim=128,
    intermediate_size=4096,      # dense MLP 尺寸（mlp_only_layers 为空时用不到）
    moe_intermediate_size=512,
    num_experts=4, num_experts_per_tok=2,
    decoder_sparse_step=1, mlp_only_layers=[],      # 每层都是 MoE
    norm_topk_prob=True,
    vocab_size=151936, tie_word_embeddings=False,
    max_position_embeddings=4096, rope_theta=1000000.0,
    torch_dtype="bfloat16",
)
torch.manual_seed(42)
model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
model.save_pretrained("~/huggingface/tiny-qwen3-moe")
# tokenizer 直接从 ~/huggingface/Qwen3-0.6B 拷 tokenizer*.json / vocab 相关文件（同家族分词器）
```

权重规模 ≈ 0.7B 参数中大头是 151936×2048 的 embed+lm_head（各 0.31B），MoE 主干很小——单卡随便放，双机 rsync 快。**seed 固定**写进脚本，权重生成一次入库路径，之后不再重生成（对拍基线的稳定前提）。

## 任务 2 · `nanovllm/models/qwen3_moe.py`

复用 qwen3.py 的 `Qwen3Attention/Qwen3DecoderLayer/Qwen3Model` 骨架（import 或浅拷贝，倾向 import + 组合），只替换 MLP：

```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts)   # 复制，不切
        self.experts = FusedExpertsEP(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size)

    def forward(self, x):                       # x: [T, H]（nano 全程扁平 token 维）
        router_logits = self.gate(x)            # [T, E]
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)   # ← fp32，对齐 HF
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)          # [T, K]
        if self.norm_topk_prob:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)            # ← 先 topk 后归一，对齐 HF
        return self.experts(x, topk_idx.to(torch.int64), topk_w.to(torch.float32))
```

HF 对齐点（transformers `modeling_qwen3_moe.Qwen3MoeSparseMoeBlock`，对拍失败先查这里）：softmax 在 **fp32**；`norm_topk_prob` 是 **topk 之后**除以 topk 和；**权重乘法与累加的 dtype 以本机 transformers 5.14.1 源码实测为准**——HF 在版本间对"routing_weights cast 回 bf16 再乘、累加张量用输入 dtype 还是 fp32"摇摆过，动手前先读 `modeling_qwen3_moe.py` 把三个 dtype 决策抄下来逐点对齐，并各锁一条单测（nano 侧 forward_local 的 fp32 index_add 若与 HF 的 bf16 累加不同，top-2 只有两个加数，偏差应 ≤1-2 ulp，仍在 4-ulp 判据内；但 argmax 级对拍最好先对齐再放宽）。

```python
class FusedExpertsEP(nn.Module):
    """EP 切分的专家组。rank r 持有全局专家 [r*L, (r+1)*L)。"""
    def __init__(self, num_experts, hidden_size, intermediate_size):
        self.ep_size = parallel.get_ep_size() or 1
        self.ep_rank = parallel.get_ep_rank() or 0
        self.num_local = num_experts // self.ep_size
        self.expert_start = self.ep_rank * self.num_local
        # 合并权重：gate/up 拼一起（对应 dense 的 MergedColumnParallelLinear 习惯）
        self.w13 = nn.Parameter(torch.empty(self.num_local, 2*intermediate_size, hidden_size))
        self.w2  = nn.Parameter(torch.empty(self.num_local, hidden_size, intermediate_size))
        self.w13.weight_loader = self.weight_loader_w13   # (param, loaded, expert_id, shard)
        self.w2.weight_loader  = self.weight_loader_w2

    def forward_local(self, x, topk_idx, topk_w):        # EP=1 路径（本里程碑）
        out = torch.zeros_like(x, dtype=torch.float32)
        for e in range(self.num_local):
            tok, k = torch.where(topk_idx == self.expert_start + e)     # 命中该专家的 (行, k)
            if tok.numel() == 0: continue
            h = x[tok] @ self.w13[e].T                    # [n, 2I]
            h = SiluAndMul()(h) @ self.w2[e].T            # [n, H]
            out.index_add_(0, tok, h.float() * topk_w[tok, k].unsqueeze(1))
        return out.to(x.dtype)

    def forward(self, x, topk_idx, topk_w):
        if self.ep_size == 1: return self.forward_local(x, topk_idx, topk_w)
        return self.forward_ep(x, topk_idx, topk_w)       # M4 实现（nanodeepep）
```

`Qwen3MoeDecoderLayer`：与 qwen3.py:120 的 DecoderLayer 相同，`self.mlp = Qwen3MoeSparseMoeBlock(config)`（本 tiny 配置每层都是 MoE，`mlp_only_layers/decoder_sparse_step` 的通用分支写上但不必测）。`Qwen3MoeForCausalLM`：照抄 Qwen3ForCausalLM（:186-216），换 config 类型与 DecoderLayer。

## 任务 3 · 权重加载（loader 扩展）

HF 权重名：`model.layers.N.mlp.gate.weight`（router）、`model.layers.N.mlp.experts.E.{gate_proj,up_proj,down_proj}.weight`。现有 loader（utils/loader.py:14-28）按子串匹配 `packed_modules_mapping`，两个问题：

1. `experts.E.gate_proj` 会被现有 `"gate_proj" → ("gate_up_proj", 0)` 的映射误捕获。
2. expert 编号要解析出来做 **EP 过滤**（不属于本 rank 的专家权重直接跳过，不落显存）。

改法——`load_model` 循环里，在 packed mapping 匹配**之前**插入 experts 分支：

```python
# loader.py 新增（伪码）
m = re.match(r"(.*)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", weight_name)
if m:
    prefix, expert_id, shard = m.group(1), int(m.group(2)), m.group(3)
    experts_mod = model.get_submodule(prefix + ".experts")     # FusedExpertsEP
    experts_mod.load_expert_weight(f.get_tensor(weight_name), expert_id, shard)  # 内部做 EP 过滤
    continue
```

`load_expert_weight`：`expert_id` 不在 `[expert_start, expert_start+num_local)` → return；否则写入 `w13[local][:I]`（gate_proj）/ `w13[local][I:]`（up_proj）/ `w2[local]`（down_proj）。`mlp.gate.weight` 不含 "gate_proj" 子串，落到默认分支由 `ReplicatedLinear.weight_loader` 处理，无冲突（确认过匹配顺序后写一条单测锁住）。

## 任务 4 · 模型分发

`model_runner.py:9/31` 改为按架构选择：

```python
_MODEL_REGISTRY = {"Qwen3ForCausalLM": Qwen3ForCausalLM, "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM}
self.model = _MODEL_REGISTRY[hf_config.architectures[0]](hf_config)
```

`allocate_kv_cache`（:103-121）与 attention 无需任何改动（MoE 只换了 MLP）。CUDA graph：tiny-moe 首版强制 `enforce_eager=True`（config 断言已在 M1 加）。

## 验收

新增 `tests/test_m3_moe_local.py`：

1. **单步 logits vs HF（主判据）**：8 条 prompt（复用 tests/common.py 语料，含 600 token 长文与 256 边界），HF `Qwen3MoeForCausalLM`（bf16, cuda, eager attention）与 nano-vllm（world=1）各算 1 个 token 的 logits：**argmax 全一致 + top-10 logprob 偏差 ≤ 4 ulp**（复用 harness.compare_logprobs 与分位数打印）。
2. **greedy 128 token vs HF `generate(do_sample=False)`** 逐 token 全等；同时存 `tests/baselines/greedy_moe_ep1.json`——这是 M4/M6 的 EP=2 对拍基线。
3. **路由一致性探针**：同一 prompt 下，nano 与 HF 每层的 `topk_idx` 逐元素一致（hook 抓取）；bf16 router logits 并列名次问题若出现，按 Plan-1 坑 1 的结论处理（topk 稳定性：`torch.topk` 两边实现一致即可，先实测再定）。
4. **专家利用率 sanity**：随机权重下 4 专家命中直方图大致均匀（打印，不设阈值）。
5. **EP 过滤单测**：mock `get_ep_rank()=1, ep_size=2` 单进程加载 → 只有 expert 2/3 落权重、`w13[0]` 对应全局 expert 2；`gate.weight` 加载不受 experts 分支干扰。
6. **回归**：dense Qwen3-0.6B 的 42 项测试不回退（registry 改动的回归）。

## 边界与坑

- **chunked prefill 中间块也过 MoE**：MoE 在每个 decoder layer 内，所有调度进来的 token 都会路由（与 lm_head 的 logits_indices 无关）——语义正确，无需特判；这也决定了 M4 的 `M = max_num_batched_tokens`。
- 投机解码（Plan-3）与 MoE 正交：验证前向的 k+1 行同样走路由。M4 的验收里带一条组合用例。
- HF 侧对拍要 `attn_implementation="eager"` 且 bf16，避免 sdpa/flash 数值路径差异干扰 4-ulp 判据；prompt 长度 >512 时 HF 显存注意（tiny 模型无压力）。
- `tie_word_embeddings=False` 显式设置（qwen3.py:202 的 tie 分支不触发）；`attention_bias` 默认 False → qkv_bias=False → q_norm/k_norm 生效（与 Qwen3 相同路径，qwen3.py:68-70）。
- w13/w2 常驻显存的 dtype 跟 hf_config.dtype（bf16）；`forward_local` 中 `@ w.T` 走 cuBLAS bf16，累加在 fp32 输出张量上——**乘权重在 fp32**，对齐 HF。
