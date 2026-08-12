from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:#从空闲池里拿一块全新的(内容作废的)block 出来用。
        block_id = self.free_block_ids.popleft()
        #popleft() — 从 free 队列头部取。配合归还时的 append(队尾),形成 FIFO。这个顺序是刻意的:最早释放的先被拿去覆盖,刚释放的排在队尾能活最久,给 prefix cache 留最大的命中窗口。换成 LIFO 的话刚释放就被覆盖,缓存基本失效。
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:#淘汰旧的缓存索引
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id
# 第 1 步:seqA 的第 0 块内容是 "你好世界",算完 KV 存进 block 7。登记索引 hash_to_block_id[H] = 7,同时 block 7 自己也记下 block.hash = H。

# 第 2 步:seqA 结束,deallocate。block 7 的 ref_count 归零,还回 free 队列。但注意——索引表项 H → 7 没删,block 7 里的 KV 数据也没擦。这是故意的,就是为了以后有请求前缀也是 "你好世界" 时能直接命中复用。

# 第 3 步:seqC 来了,内容完全不同,需要一块空白 block。_allocate_block 从 free 队首取,恰好取到 block 7。接下来 seqC 的 KV 就要写进去,把 "你好世界" 的 KV 覆盖掉。

# 第 4 步,关键:如果这时不删索引表项,后面 seqD 来了、前缀也是 "你好世界",它算出 H、查表得到 7,以为命中,直接把 block 7 填进自己的 block_table 跳过计算。但 block 7 里装的已经是 seqC 的 KV 了——seqD 拿着一堆毫不相干的 KV 去做 attention,输出直接是垃圾。

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0#num_new_blocks 先按最坏情况估:整条 seq 的 block 数全都要新分配。
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
#             token_ids —— 词表编号,CPU 上的 int list。
# block_id —— 显存大 tensor 的第几块。地址
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)#第 i 个 hash 代表的是"从开头到第 i 块的整段前缀",不是这一块孤立的内容,关键在于把上一轮的 h 当 prefix 混进去
            block_id = self.hash_to_block_id.get(h, -1)#prefix cache 的索引表:"这段前缀的 KV 之前算过吗?算过的话存在哪个 block?"
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):#复用命中的前缀 block
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:# 这块正被别的 seq 用着,那就共享
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
                #用两个容器而不是在 Block 上放一个 is_free 标志位,是为了 _allocate_block 能 O(1) 拿到一块空闲的——否则得遍历找。代价就是要维护两边的一致性。
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):#新分配剩下的
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):#一次性全部释放完整,但是只是标记可以回收
        for block_id in reversed(seq.block_table):#反转释放,这样下次可以复用到刚释放的 block,增加 prefix cache 命中概率
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
#     len(seq)=256 → 最新 token 索引 255 → 第 0 块 → 已有,不用新块   (256%256=0)
# len(seq)=257 → 最新 token 索引 256 → 第 1 块 → 需要新块 ✓      (257%256=1)
# len(seq)=258 → 最新 token 索引 257 → 第 1 块 → 上轮已开,不用   (258%256=2)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):#把本轮新填满的 block 登记进 hash_to_block_id
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        #两边都是整除,range(start, end) 只覆盖本轮跨过 256 边界的那些块。比如这轮算了 300 个 token,end = 300 // 256 = 1,只登记 block 0(满 256);block 1 里那 44 个 token 不登记。
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
