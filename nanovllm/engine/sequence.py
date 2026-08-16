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
        # token_ids[i] 对应的绝对位置是 token_offset + i。
        # driver 侧恒为 0(存着完整序列);TP worker 侧只收到本轮被调度的那一段切片,
        # 于是 token_offset = num_cached_tokens。有了它,scheduled_token_ids 两侧写法一致。
        self.token_offset = 0
        # 投机解码:本轮提议的草稿 token(还没被目标模型验证,不算进 token_ids)
        self.draft_tokens: list[int] = []
        # Phase 3B:草稿模型已经算好 KV 的 token 数,语义与 num_cached_tokens 平行
        # (位置 0..draft_num_cached_tokens-1 的**草稿** KV 有效)。
        # 稳态下它等于 num_cached_tokens,只有"上一轮草稿全被接受"时会少 1:
        # 草稿一轮跑 k 次前向,写位置 .. len+k-2,却提议到 d_k —— 最后那个 d_k
        # 自己的 KV 没人算。下一轮草稿的第一次前向吃 2 个 token 而不是 1 个补上。
        # vLLM 对 draft_model 也是每条请求多留 1 个 slot,同一件事
        # (vllm/config/speculative.py:1453-1455)。
        # 存绝对值而不是"落后几格":万一某几轮没给这条 seq 打草稿(比如运行期把
        # num_spec_tokens 改成 0),差值会越拉越大,绝对值能自己算出要补多长的
        # chunk,补不进图就退回 eager,不会静默错位。
        self.draft_num_cached_tokens = 0
        self.block_table = []#这条 seq 占用的 KV cache block 编号,入队时是空的,等 scheduler 真正调度到它才分配
        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.num_logprobs = sampling_params.logprobs
        self.completion_logprobs: list[dict] = []    # 只在 rank 0 的 driver 进程里累积,不进 __getstate__

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

    @property
    def scheduled_token_ids(self) -> list[int]:
        # 本轮要送进模型的那一段 token。prefill 是一个 chunk,decode 就是最后那 1 个,
        # 统一调度后 prepare_batch 对两者用同一个取法。
        # 投机解码时后面再接上 k 个草稿 token —— 它们还没被验证,不在 token_ids 里,
        # 但这一轮要和 last_token 一起送进目标模型做验证前向(q 长度 = k+1)。
        start = self.num_cached_tokens - self.token_offset
        n = self.num_scheduled_tokens - len(self.draft_tokens)
        return self.token_ids[start: start + n] + self.draft_tokens

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def truncate_to(self, n: int):
        # 投机一步可能吐出多个 token,如果其中某个撞上 eos / max_tokens,
        # 后面的要丢掉。注意 KV 和 hash 登记按"实际算出来的"来做,这里只裁 token 序列;
        # 这条 seq 紧接着就 FINISHED 并释放全部 block,不会有不一致残留。
        del self.token_ids[n - self.token_offset:]
        self.num_tokens = n
        self.last_token = self.token_ids[-1]

    def __getstate__(self):
        # 统一调度后 worker 侧的 prepare_batch 对 prefill / decode 一视同仁,都要取
        # scheduled_token_ids,所以不能再像以前那样 decode 只传一个 last_token
        # (那样 worker 上取切片会越界)。改成一律只传本轮被调度的那一段,
        # 通信量和以前的 decode 路径一样是 O(本轮 token 数),不传整条历史。
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, self.is_prefill,
                self.draft_tokens, self.scheduled_token_ids)

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
         self.num_scheduled_tokens, self.block_table, self.is_prefill,
         self.draft_tokens, scheduled) = state
        # worker 侧 token_ids 只有本轮切片(含草稿),token_offset 记下它的起点
        self.token_ids = scheduled[:len(scheduled) - len(self.draft_tokens)]
        self.token_offset = self.num_cached_tokens
        self.last_token = scheduled[-1] if scheduled else -1
