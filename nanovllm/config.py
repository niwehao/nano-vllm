import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # 投机解码。num_speculative_tokens = 0 表示关闭。
    num_speculative_tokens: int = 0
    speculative_method: str = "ngram"        # "ngram" = prompt-lookup,不需要第二个模型
    speculative_model: str | None = None     # method="model" 时的草稿模型路径
    ngram_prompt_lookup_max: int = 4         # n-gram 匹配时尝试的最长模式
    ngram_lookup_window: int = 2048          # 只在最近这么多 token 里回溯,控制 CPU 开销
    # Phase 2.5 Step B:给"投机批"(q 长度 = 1+接受前的草稿数,批内参差)再录一族
    # varlen 形态的 CUDA graph。False 则投机批一律走 eager,即 Step A 的行为。
    # 只在 num_speculative_tokens > 0 且非 enforce_eager 时才有意义。
    varlen_cudagraph: bool = True
    # 多机 / 专家并行(EP)。ep_size = 节点数,每机一进程一卡；1 表示单机原行为。
    ep_size: int = 1
    node_rank: int = 0                       # 本进程的全局 rank
    master_addr: str = "localhost"           # EP 模式填直连口 IP(如 192.168.100.2)
    master_port: int = 2333
    ep_transport: str = "nccl"               # "nccl" | "nvshmem"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.num_speculative_tokens >= 0
        assert self.speculative_method in ("ngram", "model")
        if self.speculative_method == "model" and self.num_speculative_tokens > 0:
            assert self.speculative_model and os.path.isdir(self.speculative_model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.ep_size >= 1
        assert self.ep_transport in ("nccl", "nvshmem")
        # 不做 TP×EP 交叉:两者最多开一个
        assert self.tensor_parallel_size == 1 or self.ep_size == 1, \
            "TP 和 EP 不能同时 >1(nano 只取每机一卡的最简形态)"
        assert 0 <= self.node_rank < max(self.tensor_parallel_size, self.ep_size)
        if self.ep_size > 1:
            # EP 首版不进 CUDA graph:MoE 层里有 all_to_all,且形状随路由变化
            assert self.enforce_eager, "EP 模式必须 enforce_eager=True"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
