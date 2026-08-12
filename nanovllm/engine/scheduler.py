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

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []#本轮被选中要跑这一次前向的 seq 列表。
        num_batched_tokens = 0#本轮 batch 到目前为止已经排进去多少个 token

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
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
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

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()#当前这条 seq 在进入内层 while 之前已经被 self.running.popleft() 摘出来了,它不在 running 里,所以不会自己踢自己
            while not self.block_manager.can_append(seq):#现在有没有地方写这条 seq 的下一个 token 的 KV"
                if self.running:#如果自己的 running 里还有别的 seq,就抢它们的 block。抢完后回到 while 重新检查,还不够就再抢下一个。
                    self.preempt(self.running.pop())
                    # if self.running: — running 里还有别的 seq,就抢它们。self.running.pop() 取队尾(最晚加入的那条),preempt 把它的全部 block 释放掉、状态改回 WAITING、塞进 waiting 队首(scheduler.py:75-79)。
                    # 腾出显存后回到 while 重新检查,还不够就再抢下一个。
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))#把这一轮 popleft 出来的 seq 按原顺序放回 running 队首。
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)#把block标记删除
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):#一条 seq 配一个刚采样出的 token
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                #chunked prefill 没喂完就到此为止,不追加 token。因为这一 chunk 的最后位置不是 prompt 的真正结尾,预测出来的东西没意义(模型算了,但要丢弃)。这条 seq 还留在 waiting 队首,下一轮接着喂。
                continue
            seq.append_token(token_id)
            
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                #两个停止条件:撞上 eos,或生成数达到 max_tokens。ignore_eos 是 benchmark 用的开关,强制跑满长度好测稳定吞吐
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)#释放全部 bloc
                self.running.remove(seq)
