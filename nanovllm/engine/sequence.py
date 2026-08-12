from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:#引擎处理这条请求过程中需要记住的所有信息,都存在这一个对象里,不散落在别处。
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)# 全局递增的编号,用来最后还原输出顺序,为什么还原数据
#你调用时传的是 prompts = [A, B, C],期望拿回 [A的回答, B的回答, C的回答] —— 第 i 个输出对应第 i 个输入。这是调用方的天然假设。

# 但引擎内部不是这个顺序完成的。三条 prompt 并发跑,谁先生成完 eos 或撞上 max_tokens,谁就先被标 FINISHED、先从 step 返回(llm_engine.py:53)。
# 假如 B 只生成 10 个 token 就结束,A 生成了 200 个,那 B 会在第 10 轮就返回,A 要等到第 200 轮。
# 实际到达顺序可能是 B → C → A。被抢占重跑的 seq 还会更晚。
        self.status = SequenceStatus.WAITING
        #status — 初始 WAITING,后面在 WAITING / RUNNING / FINISHED 之间转
        self.token_ids = copy(token_ids)
        # token_ids — prompt 的拷贝(copy,避免和调用方共享 list),之后生成的 token 会不断 append 到同一个 list 里。
        # 所以它同时装着 prompt 和 completion,靠 num_prompt_tokens 这个分界线切开
        self.last_token = token_ids[-1]

        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0#已经算完的KV
        #chunked prefill 用的游标:已经算完 KV 的有多少、这一轮打算算多少
        self.num_scheduled_tokens = 0 #本轮安排给这条 seq 计算多少个 token,是个每轮重置的临时值
        self.is_prefill = True
        self.block_table = []#这条 seq 占用的 KV cache block 编号,入队时是空的,等 scheduler 真正调度到它才分配
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
