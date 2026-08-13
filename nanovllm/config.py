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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.num_speculative_tokens >= 0
        assert self.speculative_method in ("ngram", "model")
        if self.speculative_method == "model" and self.num_speculative_tokens > 0:
            assert self.speculative_model and os.path.isdir(self.speculative_model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
