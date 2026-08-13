from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 64
    ignore_eos: bool = False
    logprobs: int | None = None

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be non-negative (0 = greedy)"
        assert 0.0 < self.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.top_k == -1 or self.top_k >= 1, "top_k must be -1 (disabled) or >= 1"
        assert self.logprobs is None or 0 <= self.logprobs <= 20, "logprobs must be in [0, 20]"

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0
