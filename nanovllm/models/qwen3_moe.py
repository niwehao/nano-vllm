"""Qwen3-MoE —— 与 qwen3.py 只差一个 MLP。

attention / RMSNorm / rope / embed / lm_head 全部原样复用 qwen3.py（Qwen3-MoE 的
attention 与 Qwen3 完全相同，含 q_norm/k_norm），所以这里只写 MoE 块和串起来的骨架。

EP 切分的钩子在 FusedExpertsEP 里：rank r 只持有全局专家 [r*L, (r+1)*L)，不属于自己
的专家权重在加载时就跳过、根本不落显存。ep_size == 1 时走 forward_local（本地全算），
ep_size > 1 时走 forward_ep（nano-deepEP 的 dispatch/combine）。
"""

import os

import torch
from torch import nn
from transformers import Qwen3MoeConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP
from nanovllm.utils import parallel

# EP 每层延迟剖面（M6 基准用）。默认关；开了每层每步多 4 个 cuda event。
EP_TIMING = os.environ.get("NANOVLLM_EP_TIMING") == "1"
EP_EVENTS: list = []


def reset_ep_timing():
    """丢掉预热阶段的打点。首次调用某个形状时混着 NCCL 建链、cuBLAS 选算法、
    显存首触，一次就能有上百毫秒——平均进去会把该形状的剖面整个带偏
    （T=512 的 GEMM 曾因此显示 2187us，而那个 bmm 与 T 无关、稳态只有 146us）。"""
    EP_EVENTS.clear()


def report_ep_timing():
    """把累计的 event 对换算成 dispatch / MoE-GEMM / combine 三段耗时（微秒）。
    按本步的 T 分组——decode 步 T 很小、prefill 步 T 接近 M，两者的通信剖面完全不同。"""
    import collections
    torch.cuda.synchronize()
    agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for T, t0, t1, t2, t3 in EP_EVENTS:
        a = agg[T]
        a[0] += t0.elapsed_time(t1) * 1e3
        a[1] += t1.elapsed_time(t2) * 1e3
        a[2] += t2.elapsed_time(t3) * 1e3
        a[3] += 1
    return {T: {"dispatch_us": v[0] / v[3], "gemm_us": v[1] / v[3],
                "combine_us": v[2] / v[3], "calls": v[3]}
            for T, v in sorted(agg.items())}


class FusedExpertsEP(nn.Module):
    """EP 切分的专家组。rank r 持有全局专家 [r*L, (r+1)*L)。

    权重按 gate/up 合并存放（对应 dense 侧 MergedColumnParallelLinear 的习惯）：
        w13[e] = [2I, H]，前 I 行是 gate_proj、后 I 行是 up_proj
        w2[e]  = [H, I]
    这与 HF 在内存里的 Qwen3MoeExperts.gate_up_proj/[E,2I,H] 同布局；但 checkpoint
    落盘时 transformers 会拆成 experts.{e}.{gate,up,down}_proj.weight 的老格式，
    所以 loader 里按老格式解析（见 utils/loader.py 的 experts 分支）。
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.ep_size = parallel.get_ep_size()
        self.ep_rank = parallel.get_ep_rank()
        assert num_experts % self.ep_size == 0, f"{num_experts=} 必须被 {self.ep_size=} 整除"
        self.num_experts = num_experts
        self.num_local = num_experts // self.ep_size
        self.expert_start = self.ep_rank * self.num_local
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.w13 = nn.Parameter(torch.empty(self.num_local, 2 * intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.num_local, hidden_size, intermediate_size))
        self.w13.weight_loader = self._noop_loader          # 不走默认 loader，见 load_expert_weight
        self.w2.weight_loader = self._noop_loader
        self.act_fn = SiluAndMul()

    @staticmethod
    def _noop_loader(param, loaded_weight, *args):
        raise RuntimeError("专家权重必须经 load_expert_weight 加载（带 EP 过滤）")

    def load_expert_weight(self, loaded: torch.Tensor, expert_id: int, shard: str):
        """EP 过滤 + 落位。不属于本 rank 的专家直接 return，权重不进显存。"""
        if not (self.expert_start <= expert_id < self.expert_start + self.num_local):
            return
        local = expert_id - self.expert_start
        i = self.intermediate_size
        if shard == "gate_proj":
            self.w13.data[local, :i].copy_(loaded)
        elif shard == "up_proj":
            self.w13.data[local, i:].copy_(loaded)
        elif shard == "down_proj":
            self.w2.data[local].copy_(loaded)
        else:
            raise ValueError(f"未知的 expert shard: {shard}")

    # ------------------------------------------------------------------ EP=1

    def forward_local(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor):
        """本地全算。累加在 fp32、权重也用 fp32 —— 与 EP 路径 combine 的归约 dtype
        保持一致，这样 EP=1 与 EP=2 的差别只剩"归约顺序/搬运"，不掺 dtype 策略差异。

        与 HF 的已知差异（transformers 5.14.1 实测）：HF 在
        Qwen3MoeTopKRouter 末尾把 routing_weights cast 回 bf16 再乘，
        Qwen3MoeExperts 的累加张量也是 bf16。top-2 只有两个加数，实测差异在 1 ulp 量级，
        落在 4-ulp 判据内（tests/test_m3_moe_local.py 判据 1 会把分位数打出来）。
        """
        if EP_TIMING:
            t0 = torch.cuda.Event(enable_timing=True); t0.record()
        out = torch.zeros(x.size(0), self.hidden_size, dtype=torch.float32, device=x.device)
        for e in range(self.num_local):
            tok, k = torch.where(topk_idx == self.expert_start + e)     # 命中该专家的 (行, k)
            if tok.numel() == 0:
                continue
            h = torch.nn.functional.linear(x[tok], self.w13[e])          # [n, 2I]
            h = torch.nn.functional.linear(self.act_fn(h), self.w2[e])   # [n, H]
            out.index_add_(0, tok, h.float() * topk_w[tok, k].unsqueeze(1))
        if EP_TIMING:
            # EP=1 没有 dispatch/combine，整段都记在 gemm 一栏，好与 EP=2 的
            # dispatch+gemm+combine 三段合计直接对比（同一个"MoE 层花了多少"口径）
            t1 = torch.cuda.Event(enable_timing=True); t1.record()
            EP_EVENTS.append((x.size(0), t0, t0, t1, t1))
        return out.to(x.dtype)

    # ------------------------------------------------------------------ EP>1

    def forward_ep(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor):
        from nanodeepep import get_ep_buffer
        buf = get_ep_buffer()
        assert x.size(0) <= buf.M, f"T={x.size(0)} 超过 buffer 的 M={buf.M}"
        if EP_TIMING:
            return self._forward_ep_timed(buf, x, topk_idx, topk_w)

        recv_x, recv_count, handle = buf.low_latency_dispatch(x, topk_idx)
        # 本地专家 GEMM：对 [L, R*M, H] **全量**算，不读 recv_count。
        # 读它要 D2H 同步，而整条 EP 路径的价值就在零 CPU 同步（LL 内核用静态 M 上限
        # 规避同步，本后端沿用同一设计）。垃圾行（本步没被写过的 slot）算出的结果不会
        # 进 combine —— combine 只按 handle 标注的有效区间取数，行与行独立，NaN 也不传染。
        # 备选：用 recv_count.max() 截断能省掉垃圾行的算力，代价是引入一次 D2H，
        # tiny 模型上不划算（单层全量 bmm 才 ~4 GFLOP×2，L40S 上微秒级）。
        h = torch.bmm(recv_x, self.w13.transpose(1, 2))                  # [L, R*M, 2I]
        h = self.act_fn(h)
        y = torch.bmm(h, self.w2.transpose(1, 2))                        # [L, R*M, H]
        return buf.low_latency_combine(y.to(x.dtype), topk_idx, topk_w, handle)

    def _forward_ep_timed(self, buf, x, topk_idx, topk_w):
        """NANOVLLM_EP_TIMING=1 时用的带 cuda event 打点版本，给 M6 的每层延迟剖面。
        事件只 record 不 synchronize，等 report_ep_timing() 统一算，尽量不扰动被测对象。"""
        def ev():
            e = torch.cuda.Event(enable_timing=True); e.record(); return e
        t0 = ev()
        recv_x, recv_count, handle = buf.low_latency_dispatch(x, topk_idx)
        t1 = ev()
        h = self.act_fn(torch.bmm(recv_x, self.w13.transpose(1, 2)))
        y = torch.bmm(h, self.w2.transpose(1, 2))
        t2 = ev()
        out = buf.low_latency_combine(y.to(x.dtype), topk_idx, topk_w, handle)
        t3 = ev()
        EP_EVENTS.append((x.size(0), t0, t1, t2, t3))
        return out

    def forward(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor):
        if self.ep_size == 1:
            return self.forward_local(x, topk_idx, topk_w)
        return self.forward_ep(x, topk_idx, topk_w)


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        # router 复制到每个 rank，不切 —— 它只有 [E, H]，切了反而要通信
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts)
        self.experts = FusedExpertsEP(config.num_experts, config.hidden_size,
                                      config.moe_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: [T, H]（nano 全程扁平 token 维）
        router_logits = self.gate(x)                                     # [T, E] bf16
        # 三个 dtype 决策逐点对齐 HF transformers 5.14.1 的 Qwen3MoeTopKRouter：
        #   ① softmax 在 fp32；② norm_topk_prob 是 topk **之后**除以 topk 和，也在 fp32。
        #   ③ HF 随后把权重 cast 回 bf16 再乘，nano 保持 fp32（理由见 forward_local 的注释）。
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        return self.experts(x, topk_idx.to(torch.int64), topk_w.to(torch.float32))


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # 与 HF Qwen3MoeDecoderLayer:314-319 同一套判定：既不在 mlp_only_layers 里、
        # 又落在 decoder_sparse_step 的整除点上，才是 MoE 层。tiny 配置每层都是 MoE，
        # 通用分支写上但没测过。
        if (layer_idx not in config.mlp_only_layers) and (
                config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3MoeDecoderLayer(config, i)
                                     for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    # 注意：这里**不能**有 gate_proj/up_proj 的映射。experts.{e}.gate_proj 会被
    # 子串匹配误捕获成 dense 的 gate_up_proj —— loader 里 experts 分支必须排在
    # packed mapping 之前（见 utils/loader.py）。dense 分支（mlp_only_layers）用的是
    # Qwen3MLP，它的 gate_up_proj 由下面的映射处理。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
