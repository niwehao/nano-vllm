from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # 调度统计,用来验证混批确实发生了,以及给性能报告用
        self.stats = {"steps": 0, "mixed": 0, "prefill_only": 0, "decode_only": 0, "preempted": 0}
        # 投机解码
        self.num_spec_tokens = config.num_speculative_tokens
        self.spec_method = config.speculative_method
        self.ngram_max = config.ngram_prompt_lookup_max
        self.ngram_window = config.ngram_lookup_window
        self.spec_proposed = 0      # 累计提议了多少草稿 token
        self.spec_accepted = 0      # 其中被接受了多少
        self.spec_steps = 0         # 有多少个 step 用上了投机

    def propose_draft(self, seq: Sequence) -> list[int]:
        """prompt-lookup(n-gram)草稿:在自己已有的 token 里找最近一次出现的相同后缀,
        把它后面跟着的 k 个 token 拿来当提议。不需要第二个模型。

        对"照抄输入""重复结构"这类文本命中率很高;命中不了就返回空,
        这一步退化成普通 decode,不会有额外开销以外的影响。
        """
        if self.num_spec_tokens == 0 or self.spec_method != "ngram":
            return []
        ids = seq.token_ids
        L = len(ids)
        lo = max(0, L - self.ngram_window)
        k = self.num_spec_tokens
        for n in range(min(self.ngram_max, L - 1), 0, -1):
            pat = ids[L - n:]
            # 从近到远回溯,取最近一次匹配
            for i in range(L - n - 1, lo - 1, -1):
                if ids[i:i + n] == pat:
                    cand = ids[i + n: i + n + k]
                    if cand:
                        return cand[:k]
                    break
        return []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        # 统一调度:一次 step 里 decode 和 prefill 混在同一个 batch 前向。
        # 顺序固定为 [decode..., prefill...],两个理由:
        #   1) 判断"纯 decode 批"(能走 CUDA graph 快路径)只需看后半段是不是空;
        #   2) 将来若要对 decode 段单独图化,decode 必须是连续前缀。
        # decode 优先拿预算:每条只花 1 个 token,先保住已在跑的请求的出词延迟,
        # 剩下的预算再给 prefill 做 chunk。这样 prefill 不再阻塞 decode。
        scheduled_seqs = []#本轮被选中要跑这一次前向的 seq 列表。
        num_batched_tokens = 0#本轮 batch 到目前为止已经排进去多少个 token

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.running.popleft()#当前这条 seq 在进入内层 while 之前已经被 self.running.popleft() 摘出来了,它不在 running 里,所以不会自己踢自己
            # 投机解码:本轮除了 last_token,还要一起验证 k 个草稿 token,
            # 所以要一次预留 k+1 个位置的 KV 空间,而不是 may_append 的一次一块。
            draft = self.propose_draft(seq)
            if len(scheduled_seqs) and num_batched_tokens + 1 + len(draft) > self.max_num_batched_tokens:
                self.running.appendleft(seq)
                break
            while not self.block_manager.ensure_capacity(seq, len(draft)):#现在有没有地方写这条 seq 的下一个 token 的 KV"
                if self.running:#如果自己的 running 里还有别的 seq,就抢它们的 block。抢完后回到 while 重新检查,还不够就再抢下一个。
                    self.preempt(self.running.pop())
                    # if self.running: — running 里还有别的 seq,就抢它们。self.running.pop() 取队尾(最晚加入的那条),preempt 把它的全部 block 释放掉、状态改回 WAITING、塞进 waiting 队首(scheduler.py:75-79)。
                    # 腾出显存后回到 while 重新检查,还不够就再抢下一个。
                else:
                    self.preempt(seq)
                    break
            else:
                seq.draft_tokens = draft
                seq.num_scheduled_tokens = 1 + len(draft)
                seq.is_prefill = False
                scheduled_seqs.append(seq)
                num_batched_tokens += seq.num_scheduled_tokens
                if draft:
                    self.spec_proposed += len(draft)
                    self.spec_steps += 1
        num_decode_seqs = len(scheduled_seqs)
        self.running.extendleft(reversed(scheduled_seqs))#把这一轮 popleft 出来的 seq 按原顺序放回 running 队首。

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:#循环每次只看 waiting[0] —— 严格 FIFO,不做任何优先级重排。
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:#block_table只是一个 Python 的 list[int],躺在 CPU 内存里,内容形如 [7, 12, 3],意思是"这条 seq 的第 0、1、2 块 KV 分别存在大 tensor 的第 7、12、3 号 block 上"。它是一张地址表,不含任何 KV 数据。
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # block_table 空,说明还没给它分配过 KV cache block。两种情况会进来:第一次被调度;或者被抢占过
                # 。反之非空就是上一轮 chunked prefill 没喂完的,走 else 分支。
                if num_cached_blocks == -1:#显存不足,放弃本轮 prefill
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 这里判断的是"本轮已经排了别的 prefill 没有",不能再用 scheduled_seqs 是否为空:
            # 统一调度后 scheduled_seqs 里通常已经躺着一批 decode,用它判断会让长 prompt
            # 只要有请求在 decode 就永远排不进来。
            if remaining < num_tokens and len(scheduled_seqs) > num_decode_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)#决定这一轮实际喂给模型多少个 token
#             num_tokens — 这条 seq 还需要计算的 token 数(总长减去 prefix cache 命中的、或已经算完的部分)
# remaining — 本轮 batch 的 token 预算还剩多少(max_num_batched_tokens - num_batched_tokens)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING#prompt 已经全部安排完,进入逐 token 生成阶段
                self.waiting.popleft()
                self.running.append(seq)#加入running区域
            scheduled_seqs.append(seq)

        assert scheduled_seqs
        num_prefill_seqs = len(scheduled_seqs) - num_decode_seqs
        self.stats["steps"] += 1
        if num_decode_seqs and num_prefill_seqs:
            self.stats["mixed"] += 1
        elif num_prefill_seqs:
            self.stats["prefill_only"] += 1
        else:
            self.stats["decode_only"] += 1
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        self.stats["preempted"] += 1
        # 被抢占的 seq 会走全量重 prefill,本轮的草稿作废
        seq.draft_tokens = []
        seq.num_scheduled_tokens = 0
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)#把block标记删除
        self.waiting.appendleft(seq)

    def find_stop(self, seq: Sequence, num_new: int):
        """本轮可能一次产出多个 token(投机接受了若干个),要逐个检查停止条件。
        返回触发停止时序列该有的总长度,没触发则返回 None。"""
        total = seq.num_tokens
        for j in range(num_new):
            idx = total - num_new + j                      # 该 token 的绝对下标
            token_id = seq.token_ids[idx - seq.token_offset]
            completion = idx - seq.num_prompt_tokens + 1   # 含它在内已生成多少个
            if (not seq.ignore_eos and token_id == self.eos) or completion == seq.max_tokens:
                #两个停止条件:撞上 eos,或生成数达到 max_tokens。ignore_eos 是 benchmark 用的开关,强制跑满长度好测稳定吞吐
                return idx + 1
        return None

    def postprocess(self, seqs: list[Sequence], token_ids: list[list[int]],
                    logprobs: list | None, accepted: list[int] | None = None):
        # 不再需要 is_prefill 参数:decode 时 num_cached_tokens 加完恒等于 num_tokens,
        # 下面这个条件对 decode 自动为假,和原来带 is_prefill 前缀的写法完全等价。
        # token_ids 现在是每条 seq 一个 list:普通情况长度 1,投机时是
        # [被接受的草稿... , 修正/奖励 token]。
        for i, (seq, new_tokens) in enumerate(zip(seqs, token_ids)):#一条 seq 配一个刚采样出的 token
            if seq.draft_tokens:
                num_accepted = accepted[i]
                self.spec_accepted += num_accepted
                # 先把被接受的草稿落进 token_ids:hash_blocks 要按 block 的实际内容算 hash,
                # 而这些位置的 KV 这一轮确实已经算出来了。
                for t in new_tokens[:num_accepted]:
                    seq.append_token(t)
                # 关键:num_scheduled_tokens 必须写"实际算出有效 KV 的位置数" = 1+接受数,
                # 而不是提议数 1+k。写成 1+k 的话,一旦 L+k 跨过 block 边界而 L+a 没跨,
                # hash_blocks 就会把一个含"被拒绝位置的垃圾 KV"的 block 登记进 prefix cache
                # 索引,之后命中该前缀的请求会直接读到错误的 KV —— 这就是 prefix cache 污染。
                seq.num_scheduled_tokens = 1 + num_accepted
                seq.draft_tokens = []
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_cached_tokens < seq.num_tokens:
                # [OLD ↓ 下一行] 原注释保留原文不动;结论仍然成立,只有括号里那句需要补充
                #chunked prefill 没喂完就到此为止,不追加 token。因为这一 chunk 的最后位置不是 prompt 的真正结尾,预测出来的东西没意义(模型算了,但要丢弃)。这条 seq 还留在 waiting 队首,下一轮接着喂。
                # [NEW] 两点补充:
                #   1) 判断条件去掉了原来的 is_prefill 前缀。decode 时 num_cached_tokens
                #      加完恒等于 num_tokens,这个条件对 decode 自动为假,所以是等价改写。
                #   2) "模型算了,但要丢弃"现在只对 attention/MLP 成立 —— Phase 3 加的
                #      logits_indices 已经让 lm_head 不再为这些中间块算 logits,
                #      省掉了每轮一次 15 万词表的 GEMM。
                continue
            seq.append_token(new_tokens[-1])
            if logprobs is not None and logprobs[i] is not None:
                seq.completion_logprobs.append(logprobs[i])
            # 被拒绝的位置多预留的 block 要还回去,否则 block_table[-1] 会指错
            self.block_manager.truncate(seq)

            stop_len = self.find_stop(seq, len(new_tokens))
            if stop_len is not None:
                seq.truncate_to(stop_len)
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)#释放全部 bloc
                self.running.remove(seq)
