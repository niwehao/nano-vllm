import os

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
from nanovllm.layers.sampler import Sampler, compute_probs, sample_from_probs
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils import parallel


_MODEL_REGISTRY = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Qwen3MoeForCausalLM": Qwen3MoeForCausalLM,
}


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event] | None = None):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        # [NEW] world 现在是 TP 与 EP 的较大者。单机 TP 时 = tensor_parallel_size(原语义),
        #       跨机 EP 时 = ep_size = 节点数。
        self.world_size = max(config.tensor_parallel_size, config.ep_size)
        self.rank = rank
        # 调试开关：每步把 logits 校验和跨 rank 比一遍，证明"复制计算"两边真的算了
        # 同样的东西。默认关（每步一次 gloo all_gather，不该进热路径）。
        self._ep_check = os.environ.get("NANOVLLM_EP_CHECK") == "1"
        self.event = event                     # [OLD] 单机 TP 的 mp.Event,已退役,只为签名兼容留着
        # 每个 step 走了哪条执行路径。硬性要求:必须能算出"走图的 step 占比",
        # 否则没法证明新录的 varlen 图真的在被 replay 而不是摆设。
        #   graph_decode = 纯 decode 批 replay 老的 flash_attn_with_kvcache 图
        #   graph_varlen = 投机批 replay 新的 varlen 图(Step B)
        #   eager        = 其余(混批、超桶、enforce_eager)
        #   draft_*      = Phase 3B 草稿模型自己那 k 次前向走了哪条路
        self.exec_stats = {"graph_decode": 0, "graph_varlen": 0, "eager": 0,
                           "draft_graph_varlen": 0, "draft_graph_decode": 0,
                           "draft_eager": 0}
        # warmup_model 会先跑一次 run(),那时 varlen 图还没录 —— 这个标志必须先存在
        self.varlen_graph_ready = False
        # 草稿模型的 KV cache 要等 allocate_kv_cache 才绑上,在那之前 run_draft 必须
        # 整段跳过(warmup 那一次就在那之前)。图同理。
        self.draft_model = None
        self.draft_ready = False
        self.draft_graph_ready = False

        # [OLD ↓] 原来这里是写死的 dist.init_process_group("nccl", "tcp://localhost:2333", ...)
        #         + torch.cuda.set_device(rank) —— localhost 让它只能单机,set_device(rank)
        #         在"每机一卡"的多机形态下会去选不存在的 cuda:1。
        # [NEW] 地址来自 config(单机默认仍是 localhost:2333,行为不变);EP 模式每台机器
        #       只有一张卡,恒选 device 0。
        parallel.init_distributed(config, rank)
        torch.cuda.set_device(0 if config.ep_size > 1 else rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # dense 模型没有 MoE 层，也就没有 dispatch/combine —— EP=2 跑 dense 时
        # 两 rank 只是复制计算（M1 的等价性验收正是靠这一点），不建 buffer。
        if config.ep_size > 1 and getattr(hf_config, "num_experts", 0) > 0:
            # 进程级单例，MoE 层每层共用同一个 buffer —— LL 语义下 dispatch/combine
            # 成对调用就能复用（DeepEP 的 decode 示例同款用法）。
            # M 取 max_num_batched_tokens：scheduler 的预算保证每步 token 数不超过它，
            # 所以一条 LL 路径同时覆盖 prefill 和 decode。
            # 放在建模型之前：它占的显存要计进 allocate_kv_cache 的 used，别让 KV 超发。
            import nanodeepep
            nanodeepep.init_ep_buffer(
                group=parallel.get_ep_group(),
                num_max_dispatch_tokens_per_rank=config.max_num_batched_tokens,
                hidden=hf_config.hidden_size,
                num_experts=hf_config.num_experts,
                transport=config.ep_transport)
        # [NEW] 按 hf_config.architectures[0] 分发，不再写死 Qwen3ForCausalLM
        self.model = _MODEL_REGISTRY[hf_config.architectures[0]](hf_config)#权重是未初始化的空张量,但形状已是切分后的
        load_model(self.model, config.model)## 从 safetensors 填入
        # [NEW] Phase 3B:草稿模型。必须建在 allocate_kv_cache **之前** —— 它的权重
        # 要算进 allocate_kv_cache 里的 used/peak,否则 KV 会超发。
        # vLLM 的顺序也是这样:gpu_worker.load_model()(:450-457)先把目标和草稿都
        # 加载完,再走 determine_available_memory()(:475)profile。
        self.draft_hf_config = None
        if config.speculative_method == "model" and config.num_speculative_tokens > 0:
            self.draft_hf_config = self.build_draft_model(config)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
            # 投机关掉时 q 恒为 1,投机批根本不会出现,这族图一张都不用录。
            if config.varlen_cudagraph and config.num_speculative_tokens > 0:
                self.capture_varlen_cudagraph()
            # 草稿的 k 次前向不图化直接亏(第一阶段 E3:bs=8/ctx=512 时不图化的
            # 成本比 1.326 > 1),所以只要开了草稿模型就必须录。
            if config.draft_cudagraph and self.draft_ready:
                self.capture_draft_cudagraph()
        # warmup_model() 自己会跑一次 run(),那一次会被记成 eager。它不是真实请求,
        # 计进去会把"走图占比"算低。这里清零,让计数只覆盖真正的 step。
        for key in self.exec_stats:
            self.exec_stats[key] = 0
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # [OLD ↑ 下一行行尾] 原注释保留原文不动;它说的"TP"现在不止 TP
        # [NEW] 现状:同一段代码同时服务单机 TP 和跨机 EP,通道从 SharedMemory 换成了
        #       gloo broadcast(见 send_cmd/recv_cmd)。"让 worker 进入待命状态"没变。
        if self.world_size > 1:#给 TP 建立控制通道,并让 worker 进入待命状态。
            # [OLD ↓] 下面两行原本描述的是 SharedMemory 控制面,那条路已整体换成 gloo
            #         broadcast(见 send_cmd/recv_cmd),这两行注释保留原文备查。
            #     self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)#创建一块 1MB 的共享内存,取名 "nanovllm"。然后 barrier 等所有 rank 到齐。
            #     self.shm = SharedMemory(name="nanovllm")#按名字 attach 到同一块内存
            # [NEW] 控制通道 = gloo 组的 broadcast_object_list,同机/跨机同一条代码路径;
            #       barrier 的作用不变(等所有 rank 到齐再放 worker 进 loop)。
            dist.barrier()
            if rank != 0:
                self.loop()

    def exit(self):
        if self.world_size > 1:
            dist.barrier()
        if not self.enforce_eager:
            if self.draft_graph_ready:
                del self.draft_graphs, self.draft_varlen_graphs
            if self.varlen_graph_ready:
                del self.varlen_graphs
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.recv_cmd()#阻塞等 rank 0 的指令,收到就执行,执行完继续等。
            self.call(method_name, *args)#一次迭代 = 一次完整前向+生成 batch 个token
            if method_name == "exit":
                break

    # [OLD ↓] read_shm/write_shm 这对函数已删除,它们靠同机 SharedMemory + mp.Event
    #         传指令,跨机不可用。下面的 recv_cmd/send_cmd 是等价替换:
    #         gloo 组的 broadcast_object_list,底层同样是 pickle,同机跨机一条路径。
    #         代价实测 0.26-0.28 ms/次(scripts/m0_nccl_test.py 的 L3-3),nano 可接受。
    def recv_cmd(self):
        assert self.world_size > 1 and self.rank > 0
        buf = [None]
        dist.broadcast_object_list(buf, src=0, group=parallel.get_cpu_group())
        return buf[0][0], buf[0][1]

    def send_cmd(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        dist.broadcast_object_list([(method_name, args)], src=0, group=parallel.get_cpu_group())

    # [OLD ↓ 下面 2 行行尾] 两条原注释保留原文不动,均已与现状不符:
    #   "8 张卡各算各的那份权重分片,靠 NCCL all-reduce 汇总" —— 那是单机 TP 的图景;
    #   "共享内存" —— 传指令的通道已经换掉了。
    # [NEW] 现状:
    #   EP 模式是 2 台机器各跑**同一批**(TP=1,权重不切,前向里没有 all-reduce),
    #   只在 MoE 层内部做 EP(dispatch/combine);单机 TP 的原语义仍然成立。
    #   指令通道 = gloo 组的 broadcast_object_list,不再是 SharedMemory。
    def call(self, method_name, *args):#效果是 8 张卡同时进入 run,各算各的那份权重分片,靠 NCCL all-reduce 汇总
        if self.world_size > 1 and self.rank == 0:
            self.send_cmd(method_name, *args)#共享内存
        method = getattr(self, method_name, None)
        return method(*args)# model_runner.py:196

    def check_logits_across_ranks(self, input_ids, positions, logits):
        """NANOVLLM_EP_CHECK=1 时用：两机同型号同库版本，复制计算的 logits 应当位级一致。
        输入(input_ids/positions)一起比，好把"输入就不同"和"输入相同但算出来不同"分开——
        前者是广播/pickle 的搬运 bug，后者才是数值问题。"""
        ctx = get_context()
        h = {
            "in": [list(input_ids.shape), int(input_ids.sum()), int(positions.sum())],
            "slot": int(ctx.slot_mapping.sum()) if ctx.slot_mapping is not None else -1,
            "clen": int(ctx.context_lens.sum()) if ctx.context_lens is not None else -1,
            "lidx": int(ctx.logits_indices.sum()) if ctx.logits_indices is not None else -1,
            "out": [list(logits.shape), float(logits.float().sum()),
                    float(logits.float().abs().max())],
        }
        buf = [None] * self.world_size
        dist.all_gather_object(buf, (self.rank, h), group=parallel.get_cpu_group())
        if self.rank == 0:
            ref = buf[0][1]
            for r, v in buf[1:]:
                bad = [k for k in ref if v[k] != ref[k]]
                if bad:
                    print(f"[EP_CHECK] rank{r} 与 rank0 不一致的项 {bad}: "
                          f"{ {k: (ref[k], v[k]) for k in bad} }", flush=True)
            self._ep_check_rounds = getattr(self, "_ep_check_rounds", 0) + 1

    def build_draft_model(self, config: Config):
        """加载草稿模型。返回它的 hf_config(allocate_kv_cache 要按它算 block 字节数)。

        两边的词表必须严格相等 —— 接受判定比的是 token id,对不上就毫无意义。
        vLLM 也只查这一条,报错文案见 config/speculative.py:1403-1418
        ("Using models with different tokenizers can cause out-of-bounds errors
        during speculative decoding")。这里连 tokenizer 文件本身也一起比,
        比只比 vocab_size 更严:vocab_size 相同但 merges 不同的两个模型
        (同架构不同训练)会静默地给出错位的 id。
        """
        from transformers import AutoConfig
        draft_hf = AutoConfig.from_pretrained(config.speculative_model)
        tgt_hf = config.hf_config
        if draft_hf.vocab_size != tgt_hf.vocab_size:
            raise ValueError(
                f"草稿模型与目标模型的词表大小不同:目标 {tgt_hf.vocab_size}, "
                f"草稿 {draft_hf.vocab_size}。投机解码要求两边 token id 一一对应,"
                f"否则接受判定没有意义、且会越界索引。"
                f"\n  目标 = {config.model}\n  草稿 = {config.speculative_model}")
        for fn in ("tokenizer.json", "vocab.json", "merges.txt"):
            a = os.path.join(config.model, fn)
            b = os.path.join(config.speculative_model, fn)
            if not (os.path.exists(a) and os.path.exists(b)):
                continue
            if os.path.getsize(a) != os.path.getsize(b) or \
                    open(a, "rb").read() != open(b, "rb").read():
                raise ValueError(
                    f"草稿模型与目标模型的 {fn} 不同。vocab_size 相同不代表 id 映射相同,"
                    f"分词器不同会让'草稿 token d 是否等于目标 argmax'这句话失去意义。"
                    f"\n  目标 = {a}\n  草稿 = {b}")
        # 两个模型共用一个 torch 默认 dtype(建模型和建 KV cache 都靠它),
        # dtype 不同会让草稿权重按目标的 dtype 建出来、KV 字节数也按错的 itemsize 估。
        if draft_hf.dtype != tgt_hf.dtype:
            raise ValueError(f"草稿模型 dtype {draft_hf.dtype} 与目标 {tgt_hf.dtype} 不同,"
                             f"本实现只支持两边同 dtype")
        arch = draft_hf.architectures[0]
        if arch not in _MODEL_REGISTRY:
            raise ValueError(f"草稿模型架构 {arch} 不在 _MODEL_REGISTRY 里")
        self.draft_model = _MODEL_REGISTRY[arch](draft_hf)
        load_model(self.draft_model, config.speculative_model)
        return draft_hf

    def kv_block_bytes(self, hf_config):
        # [NEW] 抽出来是因为现在要给两个模型各算一份:草稿和目标的层数、KV 头数
        # 都不一样(Qwen3-8B 36 层 vs Qwen3-0.6B 28 层),不能共用一个数。
        num_kv_heads = hf_config.num_key_value_heads // parallel.get_tp_size()
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size
                       * num_kv_heads * head_dim * hf_config.dtype.itemsize)
        return num_kv_heads, head_dim, block_bytes

    def bind_kv_cache(self, model, hf_config, num_blocks, num_kv_heads, head_dim):
        cache = torch.empty(2, hf_config.num_hidden_layers, num_blocks,
                            self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = cache[0, layer_id]
                module.v_cache = cache[1, layer_id]
                layer_id += 1
        assert layer_id == hf_config.num_hidden_layers
        return cache

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # [NEW] 按 TP size 除,不是 world —— EP 模式下 world=2 但 TP=1,
        #       按 world 除会把每机的 KV 头数错切一半。
        num_kv_heads, head_dim, block_bytes = self.kv_block_bytes(hf_config)
        # [NEW] Phase 3B:开了草稿模型就要两套 cache。block 数必须**严格相同** ——
        # 两套共用同一个 seq.block_table,block id 是同一套编号,数量不同就会指错。
        # 所以不是"分别按各自预算算"，而是把每 block 的字节数加起来一起算,
        # 天然得到相同的 block 数。这也是 04 计划文档 25-46 行的设计。
        draft_bytes = 0
        if self.draft_hf_config is not None:
            d_kv_heads, d_head_dim, draft_bytes = self.kv_block_bytes(self.draft_hf_config)
        max_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // (block_bytes + draft_bytes)
        if config.num_kvcache_blocks < 0:
            # [NEW] 原来只 assert > 0,那个门槛太松:1 个 block 也能过,然后在第一次
            # 抢占循环里死掉。加草稿模型后可用 block 直接腰斩,这个边界很容易撞上,
            # 所以加一条带诊断信息的下限。
            #
            # 两点要说清楚,免得高估它:
            #   1) 它只是"别把 1 个 block 也当成能跑"的冒烟门槛,**不保证不活锁**。
            #      反例:8 个 block、block_size=256、k=2、一条长 2047 的 seq ——
            #      can_allocate 要 8 块,过;decode 时 ensure_capacity(seq,2) 要 9 块,
            #      不够,running 空了就 preempt 自己,重 prefill 后再撞同一堵墙。
            #      真正能保证前进的下限是"一条 max_model_len 的 seq 装得下",
            #      即 ceil(max_model_len / kvcache_block_size);但那会让现有
            #      gpu_memory_utilization=0.35 的测试配置全部起不来,所以没有采用。
            #   2) 只在自动定块数时检查。显式指定 num_kvcache_blocks(测试里用来
            #      稳定复现抢占)时不该被这条挡住 —— 那种配置本来就是故意压到很小。
            need = min(config.max_num_seqs, 8)
            assert max_blocks >= need, (
                f"KV cache 放不下:算出 {max_blocks} 个 block,至少需要 {need} 个"
                f"(max_num_seqs={config.max_num_seqs} 取 min 8)。"
                f"\n  每 block {(block_bytes + draft_bytes) / 2**20:.1f} MiB"
                f"(目标 {block_bytes / 2**20:.1f} + 草稿 {draft_bytes / 2**20:.1f})"
                f"\n  权重+激活峰值已占 {(used + peak - current) / 2**30:.2f} GiB,"
                f"预算 {total * config.gpu_memory_utilization / 2**30:.2f} GiB"
                f"\n  可以调大 gpu_memory_utilization、调小 kvcache_block_size,"
                f"或者换更小的草稿模型")
            # [NEW] 多 rank 时块数必须全体一致:block id 由 rank0 的 scheduler 统一分配,
            #       某个 rank 的 cache 更小就会越界。两机显存同规格但驱动版本不同
            #       (595 vs 610),mem_get_info 的余量可能差几 MB —— 以 rank0 为准广播。
            if self.world_size > 1:
                buf = [max_blocks]
                dist.broadcast_object_list(buf, src=0, group=parallel.get_cpu_group())
                max_blocks = buf[0]
            config.num_kvcache_blocks = max_blocks
        else:
            # 显式指定 block 数(测试里用来稳定复现抢占路径),但不能超过显存放得下的上限
            assert config.num_kvcache_blocks <= max_blocks, \
                f"num_kvcache_blocks={config.num_kvcache_blocks} 超过显存上限 {max_blocks}"
        self.kv_cache = self.bind_kv_cache(self.model, hf_config,
                                           config.num_kvcache_blocks, num_kv_heads, head_dim)
        if self.draft_hf_config is not None:
            # 同样的 block 数、同样的编号。seq.block_table[i] 在两套 cache 里指的是
            # 同一段 token 的 KV,只是一份是目标算的、一份是草稿算的。
            self.draft_kv_cache = self.bind_kv_cache(
                self.draft_model, self.draft_hf_config,
                config.num_kvcache_blocks, d_kv_heads, d_head_dim)
            self.draft_ready = True

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_batch(self, seqs: list[Sequence]):
        # 统一路径:prefill chunk 和 decode 用同一套变长表示。
        # decode 只是 start = len-1、seqlen_q = 1 的退化情形 —— postprocess 之后恒有
        # num_cached_tokens == len(seq) - 1,代入下面的循环就自动得到
        # input = 最后一个 token、position = len-1、seqlen_k = len,和原来的
        # prepare_decode 完全一致,所以不需要为 decode 单独写一份。
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq.scheduled_token_ids)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # [OLD ↓ 下面 1 行] 原来的判断条件是 `if cu_seqlens_k[-1] > cu_seqlens_q[-1]:`,
        #                   行尾带这条注释,条件已重写,注释原文保留在此
        # prefix cache
        # [NEW] 统一路径下一律走 paged 读取:只要 KV cache 已分配就必须给 block_table,
        # 否则 decode 行(seqlen_k >> seqlen_q)根本没法从本轮的 k/v 里取到历史。
        # 只有 warmup(cache 还没分配、block_table 为空)才保持 None。
        if any(seq.block_table for seq in seqs):
            block_tables = self.prepare_block_tables(seqs)
        # lm_head 只对需要的行算 logits:普通 seq 取它那段 q 的最后一个位置,
        # 投机 seq 的验证前向要全部 k+1 个位置(每个位置都要判断草稿接受与否)。
        # 顺带把 chunked prefill 中间块的无用行也剔掉了 —— 那些行原来算完就丢。
        logits_indices = []
        offset = 0
        for seq in seqs:
            q = seq.num_scheduled_tokens
            if seq.draft_tokens:
                logits_indices.extend(range(offset, offset + q))
            else:
                logits_indices.append(offset + q - 1)
            offset += q
        logits_indices = torch.tensor(logits_indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None,
                    block_tables, logits_indices)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        # 只剩"纯 decode 批走 CUDA graph 快路径"这一个用途(Phase 2.5 Step A)。
        # 混批和 prefill 一律走 prepare_batch。图里烧死的是 flash_attn_with_kvcache 的
        # decode 形态,所以喂给它的 context 必须还是老样子(context_lens + is_prefill=False)。
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        vocab_size = self.config.hf_config.vocab_size
        # 投机 seq 在 logits 里占 1+k 行,其余占 1 行 —— 采样参数要按行展开
        rows = [1 + len(seq.draft_tokens) for seq in seqs]
        def expand(vals):
            out = []
            for v, n in zip(vals, rows):
                out.extend([v] * n)
            return out

        temperatures = expand([seq.temperature for seq in seqs])
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        # 全 batch 都没开 top_k / top_p 时传 None,apply_top_k_top_p 整段跳过那次 [B, V] 的 sort。
        # 判断在 CPU 上做(这些值本来就在 CPU),不引入 GPU 同步。
        raw_ks = [seq.top_k if seq.top_k != -1 else vocab_size for seq in seqs]
        if any(k < vocab_size for k in raw_ks):
            top_ks = torch.tensor(expand(raw_ks), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        else:
            top_ks = None
        raw_ps = [seq.top_p for seq in seqs]
        if any(p < 1.0 for p in raw_ps):
            top_ps = torch.tensor(expand(raw_ps), dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        else:
            top_ps = None
        max_logprobs = max((seq.num_logprobs if seq.num_logprobs is not None else -1) for seq in seqs)
        return temperatures, top_ks, top_ps, max_logprobs

    def unpack_logprobs(self, seqs: list[Sequence], token_ids: list[int], payload):
        # payload 里的张量一次性 .tolist(),避免逐 seq 触发 GPU 同步
        if payload is None:
            return None
        token_logprobs, top_ids, top_logprobs = payload
        token_logprobs = token_logprobs.tolist()
        top_ids = top_ids.tolist() if top_ids is not None else None
        top_logprobs = top_logprobs.tolist() if top_logprobs is not None else None
        result = []
        for i, seq in enumerate(seqs):
            n = seq.num_logprobs
            if n is None:
                result.append(None)
                continue
            item = {"token_id": token_ids[i], "logprob": token_logprobs[i]}
            if n > 0:
                # batch 里传的是全批最大的 N,这里按每条 seq 自己的 N 截取
                item["top_logprobs"] = list(zip(top_ids[i][:n], top_logprobs[i][:n]))
            result.append(item)
        return result

    def sample_speculative(self, seqs: list[Sequence], logits: torch.Tensor, sample_args,
                           draft_probs: list[torch.Tensor] | None = None):
        """投机解码的验证 + 接受判定。

        n-gram 草稿是确定性提议,等价于草稿分布 q = δ_d(在 d 上是 1 的 one-hot)。
        代入 Leviathan 的接受规则:
            以概率 min(1, p(d)/q(d)) = p(d) 接受 d;
            拒绝时从 norm(max(p - q, 0)) = "把 d 挖掉再归一化的 p" 采样。
        这样产出的 token 分布与不投机时逐分布相同。

        temperature=0(greedy)单独走一条路:直接比 token 而不是比概率,
        避免浮点边界问题,也让"开关投机输出必须逐 token 一致"成为严格恒等。

        [NEW] Phase 3B:draft_probs 不为 None 时(草稿模型 + draft_sample_method=
        "random"),q 是真实分布,走**通用式** —— 接受概率 min(1, p(d)/q(d)),
        拒绝时从 norm(clamp(p-q, 0)) 采样。两条路分开写而不是合成一条:
        q=δ 时通用式确实精确退化成简化式(q(d)=1;clamp(p-δ_d,0) 就是"把 d 挖掉的 p",
        因为 p(d)≤1),但简化式用 scatter 少一次 [R,V] 的减法和一次 [B*k,V] 的物化,
        n-gram 那条路没道理为草稿模型的功能付这个钱。**n-gram 分支一个字没动。**

        p 这一侧两条路都必须是"实际用于采样的分布"(过完 temperature+top_k+top_p),
        这是 Leviathan 定理的前提;q 那一侧只过 temperature 就够 —— 见 sample_draft
        的注释和它引的 vLLM 出处。
        """
        temperatures, top_ks, top_ps, _ = sample_args
        probs = compute_probs(logits, temperatures, top_ks, top_ps)
        greedy_tok = logits.float().argmax(dim=-1)
        sampled = sample_from_probs(probs)
        tokens = torch.where(temperatures == 0, greedy_tok, sampled)

        # 每一行对应的草稿 token(非草稿行填 0,后面不会用到)
        flat_draft = []
        for seq in seqs:
            flat_draft.extend(seq.draft_tokens)
            flat_draft.append(0)          # 每条 seq 的最后一行是"奖励 token"行,没有草稿
        draft_ids = torch.tensor(flat_draft, dtype=torch.int64, device=logits.device)

        any_sampling = bool((temperatures != 0).any().item())
        if any_sampling and draft_probs is None:
            p_d = probs.gather(dim=-1, index=draft_ids.unsqueeze(dim=1)).squeeze(dim=1)
            resid = probs.scatter(dim=-1, index=draft_ids.unsqueeze(dim=1),
                                  src=torch.zeros_like(p_d).unsqueeze(dim=1))
            resid = resid / resid.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            resid_tok = sample_from_probs(resid)
            rand = torch.rand(logits.size(0), device=logits.device)
            p_d_l, resid_tok_l, rand_l = p_d.tolist(), resid_tok.tolist(), rand.tolist()
        elif any_sampling:
            # 通用式。只在"草稿行"上算 —— 每条 seq 的最后一行是奖励 token 行,没有草稿,
            # 不参与接受判定。rows 按 (seq, 第几个草稿) 的顺序排,与
            # stack(draft_probs, dim=1).reshape(-1, V) 的行序一致。
            rows = []
            r = 0
            n_drafted = 0
            for seq in seqs:
                kk = len(seq.draft_tokens)
                rows.extend(range(r, r + kk))
                r += kk + 1
                n_drafted += kk > 0
            assert all(q.size(0) == n_drafted for q in draft_probs), \
                f"draft_probs 的 batch 维 {[q.size(0) for q in draft_probs]} 与带草稿的 seq 数 {n_drafted} 对不上"
            rows_t = torch.tensor(rows, dtype=torch.int64, device=logits.device)
            # [B, k, V] -> [B*k, V],(seq, step) 主序。显存 = B*k*V*4B;
            # bs=8/k=2/V=151936 时 9.7MB,与 compute_probs 已经物化的 [R,V] 同量级。
            qs = torch.stack(draft_probs, dim=1).reshape(-1, probs.size(-1))
            p_rows = probs.index_select(0, rows_t)
            d_rows = draft_ids.index_select(0, rows_t)
            q_d = qs.gather(dim=-1, index=d_rows.unsqueeze(dim=1)).squeeze(dim=1)
            p_d = p_rows.gather(dim=-1, index=d_rows.unsqueeze(dim=1)).squeeze(dim=1)
            # 接受概率 min(1, p/q);下面拿它和 U(0,1) 比,>1 时必接受,不用显式 clamp。
            # q(d)==0 直接判拒绝,不靠 clamp 把比值放大成一个巨数 —— 与 vLLM 一致
            # (v1/sample/rejection_sampler.py:829 的 `draft_prob > 0 and ...`)。
            # q(d)=0 却提议了 d,说明 q 已经不是真正的提议分布,接受它会让 d 超发。
            ratio = torch.where(q_d > 0, p_d / q_d.clamp_min(1e-10),
                                torch.zeros_like(p_d))
            resid = (p_rows - qs).clamp_min_(0)
            rsum = resid.sum(dim=-1, keepdim=True)
            # [NEW] 残差退化时回退到 p,不做除法。两种退化:
            #   rsum == 0  —— p 与 q 逐元素相等。数学上这时 p(d)/q(d) 恒为 1、
            #                 rand ∈ [0,1) 必接受,残差根本用不到;但不写这一支的话,
            #                 一旦浮点误差让它落到拒绝分支,0/1e-10 会得到全零行,
            #                 sample_from_probs 对全零行取 argmax 返回下标 0,
            #                 **token 0 就被当成真 token 吐出去了**。
            #   rsum 是 NaN —— 草稿 KV 里混进未初始化显存时 q 可能带 NaN,
            #                 NaN 会一路传下去,同样落到"吐 token 0"。
            # 回退到 p 是正确的:拒绝后本该从 norm((p-q)^+) 采,而该分布退化时
            # 它的极限就是 p 本身。
            safe = rsum > 0                       # NaN > 0 为 False,一并接住
            resid = torch.where(safe, resid / rsum.clamp_min(1e-10), p_rows)
            rtok = sample_from_probs(resid)
            # 同理:ratio 里的 NaN 会让 `rand < ratio` 恒假(必拒绝),
            # 拒绝后走上面回退到 p 的那条路,输出仍然服从 p。
            ratio = torch.nan_to_num(ratio, nan=0.0)
            # 回填成按 logits 行号索引的全长数组,下面的 python 循环两条路共用。
            # 名字叫 ratio_full 而不是 p_d_full:这里存的是 p(d)/q(d),
            # 只有在 q=δ 的简化式里它才恰好等于 p(d)。
            ratio_full = torch.zeros(probs.size(0), device=logits.device)
            ratio_full[rows_t] = ratio
            resid_full = torch.zeros(probs.size(0), dtype=torch.int64, device=logits.device)
            resid_full[rows_t] = rtok
            rand = torch.rand(logits.size(0), device=logits.device)
            p_d_l = ratio_full.tolist()
            resid_tok_l = resid_full.tolist()
            rand_l = rand.tolist()
        else:
            p_d_l = resid_tok_l = rand_l = None

        greedy_l = greedy_tok.tolist()
        tokens_l = tokens.tolist()

        out_tokens, accepted = [], []
        r = 0
        for seq in seqs:
            k = len(seq.draft_tokens)
            if k == 0:
                out_tokens.append([tokens_l[r]])
                accepted.append(0)
                r += 1
                continue
            is_greedy = seq.temperature == 0
            emitted, a = [], 0
            for i, d in enumerate(seq.draft_tokens):
                row = r + i
                if is_greedy:
                    ok = greedy_l[row] == d
                else:
                    ok = rand_l[row] < p_d_l[row]
                if ok:
                    emitted.append(d)
                    a += 1
                else:
                    emitted.append(greedy_l[row] if is_greedy else resid_tok_l[row])
                    break
            else:
                # 全部接受,再白赚一个"奖励 token"(第 k+1 行本来就是免费算出来的)
                emitted.append(tokens_l[r + k])
            out_tokens.append(emitted)
            accepted.append(a)
            r += k + 1
        return out_tokens, accepted

    # ------------------------------------------------------------------
    # Phase 3B:草稿模型
    # ------------------------------------------------------------------

    def slots_for(self, seq: Sequence, positions):
        # position -> 物理槽位。block_table 已由 scheduler 的 ensure_capacity(seq, k)
        # 保证覆盖到 len+k-1,草稿写的最大位置是 len+k-2,在里面。
        bs = self.block_size
        return [seq.block_table[p // bs] * bs + p % bs for p in positions]

    def prepare_draft_first(self, seqs: list[Sequence]):
        """草稿第一次前向:每条 seq 吃 [draft_num_cached_tokens, len) 这一段。

        q=1 是常态(吃 last_token,和普通 decode 一样);
        q=2 只在上一轮"全部接受"之后出现 —— 那一轮草稿写到位置 len+k-2,
        而第 k 个草稿 token 自己的 KV 没人算,这里补上。
        走变长路径(is_prefill=True 那条分支),因为批内 q 参差。
        返回值第三项是本批最大的 q,调用方据此决定进不进 q_max=2 的那族图。
        """
        input_ids, positions, slot_mapping = [], [], []
        cu_q, cu_k, logits_indices = [0], [0], []
        max_q = max_k = 0
        for seq in seqs:
            L = len(seq)
            q = L - seq.draft_num_cached_tokens
            assert 1 <= q <= L, (
                f"草稿要补的 chunk 长度 {q} 不合法(len={L}, "
                f"draft_num_cached={seq.draft_num_cached_tokens})")
            pos = list(range(L - q, L))
            off = seq.token_offset
            input_ids.extend(seq.token_ids[L - q - off: L - off])
            positions.extend(pos)
            slot_mapping.extend(self.slots_for(seq, pos))
            cu_q.append(cu_q[-1] + q)
            cu_k.append(cu_k[-1] + L)
            logits_indices.append(cu_q[-1] - 1)     # 只要每条 seq 最后一行的 logits
            max_q, max_k = max(max_q, q), max(max_k, L)

        def t(x, dt):
            return torch.tensor(x, dtype=dt, pin_memory=True).cuda(non_blocking=True)

        block_tables = self.prepare_block_tables(seqs)
        set_context(True, t(cu_q, torch.int32), t(cu_k, torch.int32), max_q, max_k,
                    t(slot_mapping, torch.int32), None, block_tables,
                    t(logits_indices, torch.int64))
        return t(input_ids, torch.int64), t(positions, torch.int64), max_q

    def prepare_draft_decode(self, seqs: list[Sequence], step: int):
        """草稿第 2..k 次前向:q 恒为 1,走 flash_attn_with_kvcache 快路径。

        第 step 次(step 从 1 数)吃的是上一轮采出来的 d_step,它的位置是 len-1+step,
        此刻 cache 里有效的 KV 有 len+step 个。input_ids 由调用方以 GPU 张量直接给,
        不经 CPU —— 草稿循环里每步都 .tolist() 会引入 k 次同步。
        """
        positions, context_lens, slot_mapping = [], [], []
        for seq in seqs:
            p = len(seq) - 1 + step
            positions.append(p)
            context_lens.append(p + 1)
            slot_mapping.append(self.slots_for(seq, [p])[0])

        def t(x, dt):
            return torch.tensor(x, dtype=dt, pin_memory=True).cuda(non_blocking=True)

        set_context(False, slot_mapping=t(slot_mapping, torch.int32),
                    context_lens=t(context_lens, torch.int32),
                    block_tables=self.prepare_block_tables(seqs))
        return t(positions, torch.int64)

    def sample_draft(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """从草稿 logits 采一个 token,并(需要时)返回整行的 q 分布。

        跟 vLLM 一样,草稿这一侧**只过 temperature,不过 top_k / top_p**:
        拒绝采样对任意 q 都精确产出 p,q 只影响接受率不影响正确性。
        vLLM 的原注释在 v1/spec_decode/llm_base_proposer.py:1878-1881:
          "Currently, we ignore most of the sampling parameters in generating the
           draft tokens. We only use the temperature. While this could degrade the
           acceptance rate, it does not affect the distribution of the generated
           tokens after rejection sampling."

        draft_sample_method="greedy"(默认,与 vLLM 的默认一致)时直接 argmax,
        q 退化成 one-hot δ_d,接受规则精确等于简化式 p(d),连 [B,V] 都不用存。
        """
        greedy = logits.float().argmax(dim=-1)
        if self.config.draft_sample_method == "greedy":
            return greedy, None
        safe_t = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits.float() / safe_t.unsqueeze(dim=1), dim=-1)
        sampled = sample_from_probs(probs)
        tokens = torch.where(temperatures == 0, greedy, sampled)
        # temperature==0 的行草稿也是确定性的,q 仍是 δ。把那些行的 q 换成 one-hot,
        # 通用接受式代进去就自动退化成 token 比较,不用在下游再分叉。
        onehot = torch.zeros_like(probs).scatter_(
            dim=-1, index=tokens.unsqueeze(dim=1), src=torch.ones_like(probs[:, :1]))
        q = torch.where((temperatures == 0).unsqueeze(dim=1), onehot, probs)
        return tokens, q

    @torch.inference_mode()
    def run_draft(self, seqs: list[Sequence]):
        """在目标模型的验证前向**之前**把草稿跑出来,回填 seq.draft_tokens。

        放在前面而不是像 vLLM 那样放在一步的结尾,是因为 nano-vllm 的
        ensure_capacity(seq, k) 是在 schedule 时按**当轮**的 k 预留的,草稿要写的位置
        len-1-lag .. len+k-2 全在这段里;放到结尾就得再为"下一轮的草稿"多留一份
        (vLLM 正是这么做的:config/speculative.py:1421 的 max_num_new_slots_for_drafting
        + core/sched/scheduler.py:701 的 input_budget -= ... + draft_slots)。
        这是本实现的设计选择,不是照抄 vLLM。

        返回 (draft_probs, None) —— draft_probs 是长度 k 的列表,每项 [B, V];
        greedy 草稿时为 None(q=δ,下游走简化式)。
        """
        if not self.draft_ready:
            return None
        k = self.config.num_speculative_tokens
        prefill = [s for s in seqs if s.is_prefill]
        # 只给 scheduler 真的预留了 k 个草稿位的 seq 打草稿。这个条件不能省:
        # scheduler.num_spec_tokens 是可以被运行期改掉的(tests/gen.py 的 --pollution
        # 第二阶段就把它置 0),那时 num_scheduled_tokens 只有 1,再往 draft_tokens 里
        # 塞 k 个 token,scheduled_token_ids 里 n = num_scheduled - k 会变成负数,
        # 送进模型的就是"草稿 token 顶掉了 last_token"的一批垃圾。
        decode = [s for s in seqs if not s.is_prefill]
        drafted = [s for s in decode if s.num_scheduled_tokens == 1 + k]
        # [NEW] 但"不给它打草稿"不等于"可以不管它的草稿 KV"。目标每轮照样往前走 1 个
        # token、hash_blocks 照样把填满的 block 登记进 prefix cache 索引;草稿要是停在
        # 原地,登记进去的就是一段**草稿 KV 从没写过**的 block,后来命中这个前缀的请求
        # 会一直读到未初始化的显存。所以这些 seq 也要跑一次草稿前向,只写 KV、丢弃 logits,
        # 把 draft_num_cached_tokens 追平。
        # (独立 review 指出的问题:原来只挡住了 scheduled_token_ids 被顶掉,
        #  没有维护"两套 cache 逐位置对齐"这条不变量。)
        sync_only = [s for s in decode if s.num_scheduled_tokens != 1 + k]
        if prefill:
            self.sync_draft_prefill(prefill)
        if sync_only:
            self.sync_draft_decode(sync_only)
        if not drafted:
            return None
        return self.propose_draft_model(drafted)

    def sync_draft_decode(self, seqs: list[Sequence]):
        """这一轮不给它们打草稿,但草稿 KV 必须跟上目标,否则 prefix cache 会被污染。

        走的是与草稿第一次前向完全相同的形态(变长,q = len - draft_num_cached),
        只是不采样、不产出草稿 token。q 通常是 1,但如果连着几轮没打草稿会更长,
        那时 prepare_draft_first 算出的 max_q > 2,自动落到 eager。
        """
        input_ids, positions, max_q = self.prepare_draft_first(seqs)
        use_graph = self.draft_graph_ready and max_q <= 2 and len(seqs) <= self.graph_bs[-1]
        if use_graph:
            self.exec_stats["draft_graph_varlen"] += 1
            self.replay_varlen(self.draft_varlen_graphs, self.draft_varlen_vars,
                               input_ids, positions, len(seqs))
        else:
            self.exec_stats["draft_eager"] += 1
            self.draft_model(input_ids, positions)
        reset_context()
        for seq in seqs:
            seq.draft_num_cached_tokens = len(seq)

    def sync_draft_prefill(self, seqs: list[Sequence]):
        """prefill chunk:目标算完之后草稿必须对同一段再算一遍,只写 KV,logits 丢弃。

        这是"两套 cache 内容对齐"这个不变量的最简单维持方式(04 计划文档 43-46 行)。
        对齐必须成立,否则 prefix cache 命中时草稿会读到别的 seq 的 KV:
        hash 命中复用 block i 时,草稿 cache 的 block i 里存的必须也是同一段前缀的
        草稿 KV。slot_mapping / block_tables 与目标完全相同,所以直接复用 prepare_batch。
        """
        input_ids, positions = self.prepare_batch(seqs)
        self.exec_stats["draft_eager"] += 1
        self.draft_model(input_ids, positions)
        reset_context()
        for seq in seqs:
            # 目标写到哪,草稿就写到哪,两边齐平。命中 prefix cache 被跳过的那段前缀
            # 也算数 —— 那些 block 里的草稿 KV 由"每次写目标就同步写草稿"这条不变量
            # 保证是同一段前缀的草稿 KV。
            seq.draft_num_cached_tokens = seq.num_cached_tokens + seq.num_scheduled_tokens

    def propose_draft_model(self, seqs: list[Sequence]):
        k = self.config.num_speculative_tokens
        temps = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32,
                             pin_memory=True).cuda(non_blocking=True)
        use_graph = self.draft_graph_ready and len(seqs) <= self.graph_bs[-1]
        drafts, probs_per_step = [], []

        # ---- 第 1 步:varlen,q ∈ {1,2} ----
        input_ids, positions, max_q = self.prepare_draft_first(seqs)
        # 草稿那族 varlen 图的 q_max 烧死在 2。稳态下 max_q 就是 1 或 2,但如果某条 seq
        # 连着几轮没被打草稿(运行期把 num_spec_tokens 改小之类),要补的 chunk 会更长,
        # 那时**只有第一步**退回 eager —— 变长路径本来就吃得下任意 q,只是慢一点,
        # 不会算错。后面 k-1 步的 q 恒为 1,那族 decode 图照走。
        if use_graph and max_q <= 2:
            self.exec_stats["draft_graph_varlen"] += 1
            hidden = self.replay_varlen(self.draft_varlen_graphs, self.draft_varlen_vars,
                                        input_ids, positions, len(seqs))
        else:
            self.exec_stats["draft_eager"] += 1
            hidden = self.draft_model(input_ids, positions)
        logits = self.draft_model.compute_logits(hidden)
        reset_context()
        tok, q = self.sample_draft(logits, temps)
        drafts.append(tok)
        probs_per_step.append(q)

        # ---- 第 2..k 步:q=1 纯 decode ----
        for step in range(1, k):
            positions = self.prepare_draft_decode(seqs, step)
            if use_graph:
                self.exec_stats["draft_graph_decode"] += 1
                hidden = self.replay_decode(self.draft_graphs, self.draft_graph_vars,
                                            tok, positions)
            else:
                self.exec_stats["draft_eager"] += 1
                hidden = self.draft_model(tok, positions)
            logits = self.draft_model.compute_logits(hidden)
            reset_context()
            tok, q = self.sample_draft(logits, temps)
            drafts.append(tok)
            probs_per_step.append(q)

        # k 步一共只做一次 D2H —— 每步都 .tolist() 会串上 k 次同步
        flat = torch.stack(drafts, dim=1).tolist()      # [B, k]
        for seq, row in zip(seqs, flat):
            seq.draft_tokens = row
            assert len(row) == k
        return None if probs_per_step[0] is None else probs_per_step

    def is_pure_decode(self, seqs: list[Sequence]) -> bool:
        # 纯 decode 批(每条 q 长度都是 1)可以走 flash_attn_with_kvcache —— 那个 kernel
        # 是专为 q=1 做 split-K 优化的,比通用变长 kernel 快一截(实测 decode step
        # 16.2ms vs 19.4ms)。所以"用哪个 kernel"和"要不要 replay CUDA graph"必须分开判断:
        # 早先把两者合成一个条件,导致 eager 模式下的纯 decode 也被赶去走变长路径,
        # decode 吞吐白掉了 18%。
        # 投机 seq 的 q 长度是 k+1,同样不能走 q=1 的快路径
        return not any(seq.is_prefill or seq.num_scheduled_tokens != 1 for seq in seqs)

    def use_cudagraph(self, seqs: list[Sequence]) -> bool:
        # 图里烧死的是 q 长度恒为 1 的 decode 形态,所以只有纯 decode 批能 replay。
        # 混批里 prefill chunk 的 q 长度是变的,图化不了,只能走 eager 变长路径。
        if self.enforce_eager or not self.is_pure_decode(seqs):
            return False
        return len(seqs) <= self.graph_bs[-1]

    def is_spec_decode(self, seqs: list[Sequence]) -> bool:
        # "投机批":没有 prefill,但至少一条 seq 带草稿(q 长度 1+draft > 1)。
        # 批内 q 长度可以参差不齐 —— 这正是 varlen 图能吃下、而老 decode 图吃不下的形态。
        # 上界 1+k 是图的容量:缓冲区按最坏情况 bs*(1+k) 行开的,超了放不下。
        q_max = 1 + self.config.num_speculative_tokens
        if any(seq.is_prefill for seq in seqs):
            return False
        if not any(seq.num_scheduled_tokens > 1 for seq in seqs):
            return False
        return all(1 <= seq.num_scheduled_tokens <= q_max for seq in seqs)

    def use_varlen_cudagraph(self, seqs: list[Sequence]) -> bool:
        # 与 use_cudagraph 严格并列的第二个"要不要 replay"判断,同样和"用哪个 kernel"分开
        # (坑 5 的教训:两件事合成一个条件会悄悄丢性能)。
        if self.enforce_eager or not self.varlen_graph_ready:
            return False
        if not self.is_spec_decode(seqs):
            return False
        return len(seqs) <= self.graph_bs[-1]

    @torch.inference_mode()
    def run_varlen_graph(self, input_ids: torch.Tensor, positions: torch.Tensor, num_seqs: int):
        """投机批走 varlen 图。context 必须是 prepare_batch 刚产出的那份(变长形态)。

        方案 C 的要点:草稿长度保持 ngram 给出的原样,不伪造任何 draft token。
        批内 q 长度参差不齐由 cu_seqlens_q(设备张量,replay 前刷新)表达,
        图里烧死的只有 max_seqlen_q=1+k 和 max_seqlen_k=max_model_len 两个 host 标量。
        缓冲区按最坏情况 bs*(1+k) 行开,多出来的行挂给一条 sink seq:
        它的 k 长度为 0(FA 对 seqlen_k=0 的行输出恰为全 0,第一阶段 Q3 实测),
        slot_mapping 为 -1(store_kvcache_kernel 跳过,attention.py:23),输出丢弃。
        """
        out = self.replay_varlen(self.varlen_graphs, self.graph_varlen_vars,
                                 input_ids, positions, num_seqs)
        # compute_logits 在图外,用的是 prepare_batch 产出的真实 logits_indices,
        # 它只指向真实行,padding 槽的行天然被丢弃。
        return self.model.compute_logits(out)

    @torch.inference_mode()
    def replay_varlen(self, graphs, gv, input_ids, positions, num_seqs):
        """[NEW] 从 run_varlen_graph 里抽出来的图刷新 + replay,按 (graphs, gv) 参数化,
        好让草稿模型那族 varlen 图复用同一份 padding 槽逻辑。原注释随代码原样搬过来。
        返回图输出的隐状态,lm_head 由调用方按自己的模型算。
        """
        context = get_context()
        q_max = gv["q_max"]
        bs = next(x for x in self.graph_bs if x >= num_seqs)
        num_tokens = input_ids.size(0)          # 本轮真实 token 数
        total = bs * q_max                      # 该桶的缓冲区行数
        nslot = 2 * bs                          # bs 个真实槽 + 至多 bs 个 padding 槽
        # max_seqlen_k 是烧进图的 host 标量(= max_model_len)。真实 k 长度超过它时
        # FA 会按 n_blocks_per_split 截断,悄悄少读一段 KV —— 宁可在这里响。
        assert context.max_seqlen_k <= self.config.max_model_len, \
            f"max_seqlen_k={context.max_seqlen_k} 超过图里烧死的 {self.config.max_model_len}"
        # [NEW] max_seqlen_q 同样是烧进图的 host 标量。超了不会报错,只会让下面
        # padding 槽的 cu_seqlens 算术(每槽至多 q_max 行)对不上,静默错位。
        assert context.max_seqlen_q <= q_max, \
            f"max_seqlen_q={context.max_seqlen_q} 超过这族图的 q_max={q_max}"

        gv["input_ids"][:num_tokens] = input_ids
        gv["input_ids"][num_tokens:total] = 0
        gv["positions"][:num_tokens] = positions
        gv["positions"][num_tokens:total] = 0
        gv["slot_mapping"][:total].fill_(-1)
        gv["slot_mapping"][:num_tokens] = context.slot_mapping
        gv["cu_seqlens_q"][:num_seqs + 1] = context.cu_seqlens_q
        gv["cu_seqlens_k"][:num_seqs + 1] = context.cu_seqlens_k
        # 剩下的 total-num_tokens 行分给若干 padding 槽,每槽至多 q_max 行,
        # 且 k 长度 = q 长度(指向 block 0)。**两个约束都不能松**:
        #   q 长度 ≤ q_max ≤ 64:图的 m 维 grid 烧死为 ceil(max_seqlen_q/64)=1,
        #                        一格里超过 64 行的部分永远不会被写,会留下未初始化显存;
        #   k 长度 ≥ 1        :k 长度为 0 会让 FA 走 flash_fwd_kernel.h:537 的 early-exit
        #                        分支,而那个分支算 LSE 偏移用的是 **padded** 公式(:545),
        #                        没有理会 params.unpadded_lse —— 但 varlen 的 softmax_lse
        #                        是按 {num_heads, total_q} 的 unpadded 布局分配的
        #                        (flash_api.cpp:652 + :688)。于是最后一格会写出缓冲区。
        #                        实测(compute-sanitizer + PYTORCH_NO_CUDA_MEMORY_CACHING=1):
        #                        k 长度 0 -> "Invalid __global__ write of size 4 bytes";
        #                        k 长度 = q 长度 -> 0 errors。见 tests/phase25_probes/probe_sink_memcheck.py
        n = nslot + 1 - num_seqs
        dst_q = gv["cu_seqlens_q"][num_seqs:nslot + 1]
        torch.add(gv["pad_ramp"][:n], num_tokens, out=dst_q)
        dst_q.clamp_(max=total)
        dst_k = gv["cu_seqlens_k"][num_seqs:nslot + 1]
        torch.sub(dst_q, num_tokens, out=dst_k)
        dst_k += context.cu_seqlens_k[num_seqs]
        # padding 槽读 block 0 的垃圾 KV(输出丢弃)。q_max ≤ block_size,一块就够。
        gv["block_tables"][num_seqs:nslot, 0] = 0
        gv["block_tables"][:num_seqs, :context.block_tables.size(1)] = context.block_tables

        graphs[bs].replay()
        return gv["outputs"][:num_tokens]

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, use_graph: bool):
        if not use_graph:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:# decode 要用 CUDA graph
            out = self.replay_decode(self.graphs, self.graph_vars, input_ids, positions)
            return self.model.compute_logits(out)

    @torch.inference_mode()
    def replay_decode(self, graphs, graph_vars, input_ids, positions):
        """[NEW] 从 run_model 的 else 分支抽出来,按 (graphs, graph_vars) 参数化,
        好让草稿模型那族 q=1 图复用。原注释随代码原样搬过来。返回隐状态。
        """
        bs = input_ids.size(0)
        context = get_context()
        graph = graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return graph_vars["outputs"][:bs]

    def run(self, seqs: list[Sequence]):
        # [NEW] Phase 3B:草稿模型的 k 次前向必须排在最前面 —— 后面 prepare_batch /
        # prepare_sample 都要读 seq.draft_tokens,而 method="model" 时 scheduler
        # 只预留了位置、没填内容,内容由这里回填。
        draft_probs = self.run_draft(seqs)
        pure_decode = self.is_pure_decode(seqs)
        use_graph = pure_decode and self.use_cudagraph(seqs)
        # 第三条路:投机批(无 prefill、批内 q 长度 1..1+k 参差)走新录的 varlen 图。
        # 与 use_graph 互斥 —— 纯 decode 批仍归老图,那条路的快 kernel 不能丢(坑 5)。
        use_varlen_graph = not pure_decode and self.use_varlen_cudagraph(seqs)
        input_ids, positions = self.prepare_decode(seqs) if pure_decode else self.prepare_batch(seqs)
        #这里context是真数据,之前只是一个烧录地址的动作
        sample_args = self.prepare_sample(seqs) if self.rank == 0 else None
        if use_varlen_graph:
            self.exec_stats["graph_varlen"] += 1
            logits = self.run_varlen_graph(input_ids, positions, len(seqs))
        else:
            self.exec_stats["graph_decode" if use_graph else "eager"] += 1
            logits = self.run_model(input_ids, positions, use_graph)
        if self._ep_check and self.world_size > 1:
            self.check_logits_across_ranks(input_ids, positions, logits)
        if self.rank != 0:
            reset_context()
            return None
        if any(seq.draft_tokens for seq in seqs):
            token_ids, accepted = self.sample_speculative(seqs, logits, sample_args,
                                                          draft_probs)
            reset_context()
            return token_ids, None, accepted
        tokens, payload = self.sampler(logits, *sample_args)
        flat = tokens.tolist()
        logprobs = self.unpack_logprobs(seqs, flat, payload)
        reset_context()
        # 统一成"每条 seq 一个 token 列表",投机时这个列表会更长
        return [[t] for t in flat], logprobs, [0] * len(seqs)

    @torch.inference_mode()
    def capture_cudagraph(self):#捕获计算图
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graph_pool = None
        self.graphs, self.graph_vars = self.capture_decode_family(
            self.model, hf_config.hidden_size, max_bs)

    @torch.inference_mode()
    def capture_decode_family(self, model, hidden_size, max_bs):
        """录一族 q=1 的 decode 图(flash_attn_with_kvcache 形态)。

        [NEW] 函数体原样来自 capture_cudagraph,只是把"哪个模型、hidden 多宽"
        提成参数 —— Phase 3B 的草稿模型要录同构的一族,不参数化就得整段抄一遍。
        原有注释随代码原样搬过来,一字未动。

        max_bs 也必须由调用方传进来,不能就地写成 self.graph_bs[-1]:
        max_num_seqs ∈ {3,5,6,7} 时 graph_bs[-1]=8 > max_num_seqs,两者不等。
        原代码用的是 min(max_num_seqs, 512),于是 outputs[:8] 在一个 5 行的缓冲区上
        只切出 5 行 —— "8 号桶"其实录的是一张 5 行的图。改用 graph_bs[-1] 会让它变成
        真正的 8 行图,bs=5 的批就多出 3 行 padding,cuBLAS 的 M 维分块跟着变,
        目标路径的 bf16 结果**静默地**不一样了(07 报告坑 5 就是这个机制)。
        没有测试配置落在那几个值上,但既有路径不许有静默变化,所以原样保留旧口径。
        """
        config = self.config
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hidden_size)
        #上面这六个就是后来的 graph_vars —— replay 时往里写数据的固定地址
        graphs = {}

        for bs in reversed(self.graph_bs):#这个循环跑 36 次,每次为一个 batch size 造一张图。
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])#把静态张量的切片塞进全局 context。因为模型内部的 attention 层不是从参数拿这些,而是自己 get_context() 
            outputs[:bs] = model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()#指定用哪个显存池
            graphs[bs] = graph# 一张录好的图,里面烧死了
            torch.cuda.synchronize()
            reset_context()

#             输入:  bs = 16
# 产出:  self.graphs[16] = 一张录好的图,里面烧死了
#          input_ids[:16] / positions[:16] / slot_mapping[:16] /
#          context_lens[:16] / block_tables[:16] / outputs[:16]
#          这六块的显存地址,以及 28 层全部 kernel 的调用序列

        return graphs, dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_varlen_cudagraph(self):
        """Step B:给投机批再录一族 varlen 形态的图。

        与 capture_cudagraph 的三点不同:
          1) attention 走 flash_attn_varlen_func —— 即 set_context 的第一个参数为 True
             那条分支(attention.py:64),不是 flash_attn_with_kvcache;
          2) 每张图的 q 行数按最坏情况 bs*(1+k) 开,**实际算几行由 cu_seqlens_q 决定**,
             所以同一张图能吃下批内参差的 q 长度(第一阶段 Q4 实测 5 种组合全部 bitwise 相同);
          3) seq 槽比桶多一个,最后一个是吸收 padding 行的 sink。

        两个被烧进图的 host 标量:
          max_seqlen_q = 1+k     每条 seq 的 q 长度上界,批内实际值可以更小;
          max_seqlen_k = max_model_len   最坏情况(第一阶段 Q2:结果仍正确,
                         与传真实值的差异 ≤ 0.5 ulp 且与 fp32 参照等距)。
        """
        hf_config = self.config.hf_config
        q_max = 1 + self.config.num_speculative_tokens
        self.varlen_graphs, self.graph_varlen_vars = self.capture_varlen_family(
            self.model, hf_config.hidden_size, q_max)
        self.varlen_graph_ready = True

    @torch.inference_mode()
    def capture_varlen_family(self, model, hidden_size, q_max):
        """[NEW] 函数体原样来自 capture_varlen_cudagraph,把 (模型, hidden, q_max)
        提成参数。草稿模型第一次前向的 q 只可能是 1 或 2(第一阶段 E4 实测上界就是 2),
        用同一份代码换个 q_max 录一族就够,不必整段抄。原注释随代码原样搬过来。
        """
        config = self.config
        # padding 槽每格只放 q_max 行,所以 q_max 不能超过一块能装的 token 数
        # (每格的 k 长度 = q 长度,block_tables 只填了第 0 列)。
        assert q_max <= self.block_size, \
            f"1+num_speculative_tokens={q_max} 超过 kvcache_block_size={self.block_size}"
        max_bs = self.graph_bs[-1]
        max_total = max_bs * q_max
        max_nslot = 2 * max_bs                 # 真实槽 + padding 槽
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_total, dtype=torch.int64)
        positions = torch.zeros(max_total, dtype=torch.int64)
        slot_mapping = torch.full((max_total,), -1, dtype=torch.int32)
        cu_seqlens_q = torch.zeros(max_nslot + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_nslot + 1, dtype=torch.int32)
        block_tables = torch.full((max_nslot, max_num_blocks), -1, dtype=torch.int32)
        outputs = torch.zeros(max_total, hidden_size)
        # replay 时用来一次算出全部 padding 槽的 cu_seqlens:0, q_max, 2*q_max, ...
        pad_ramp = torch.arange(0, max_nslot + 1, dtype=torch.int32) * q_max
        # 捕获时的 warmup 那一次会真的读 KV:让每条 seq 都指向 block 0,读一条合法的垃圾 KV。
        # (replay 时这一列会被真实 block_table 覆盖;sink 行 k 长度为 0,从不被读 ——
        #  第一阶段实测 block_tables 的 -1 padding 与填 0 逐位相同,即根本没读。)
        block_tables[:, 0] = 0

        graphs = {}
        for bs in reversed(self.graph_bs):
            total = bs * q_max
            nslot = 2 * bs                      # bs 个真实槽 + 至多 bs 个 padding 槽
            # 捕获前填一份合法数据:bs 条 seq 各 q_len=k_len=q_max,padding 槽长度 0。
            # 缓冲区里的值不影响录下来的 kernel 序列(FA 的 grid 只由 host 标量定),
            # 但 warmup 那一次得跑得通。
            cu_seqlens_q[:bs + 1] = pad_ramp[:bs + 1]
            cu_seqlens_q[bs + 1:nslot + 1] = total
            cu_seqlens_k[:bs + 1] = pad_ramp[:bs + 1]
            cu_seqlens_k[bs + 1:nslot + 1] = total
            graph = torch.cuda.CUDAGraph()
            set_context(True, cu_seqlens_q[:nslot + 1], cu_seqlens_k[:nslot + 1],
                        q_max, config.max_model_len, slot_mapping[:total], None,
                        block_tables[:nslot], None)
            outputs[:total] = model(input_ids[:total], positions[:total])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:total] = model(input_ids[:total], positions[:total])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        return graphs, dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            outputs=outputs,
            pad_ramp=pad_ramp,
            q_max=q_max,
        )

    @torch.inference_mode()
    def capture_draft_cudagraph(self):
        """Phase 3B:给草稿模型录两族图。

        草稿一轮跑 k 次前向,形态是两种:
          第 1 次   q = len - draft_num_cached_tokens ∈ {1,2},批内参差 → varlen 图,q_max=2。
                    上界 2 是第一阶段 E4 用 4 个 k 值 × 3 seed × 20000 轮模拟出来的,
                    不是拍脑袋:一轮里草稿写到位置 len+k-2 而提议到 d_k,
                    全接受时恰好欠一格,所以最多补 1 格。
          第 2..k 次 q = 1,纯 decode → 老的 flash_attn_with_kvcache 图,那个 kernel
                    专为 q=1 做了 split-K 优化(坑 5:"能用快 kernel"和"能用图"
                    是两个正交条件,这里两个都满足)。

        为什么非录不可:第一阶段 E3 实测 bs=8/ctx=512 时,草稿 2 步走图 6.36ms、
        目标 1 步 23.60ms,比值 0.269;草稿**不图化**时比值 1.326 —— 直接亏。
        0.6B 模型 28 层的 launch 开销比它自己的算力开销还大。
        (vLLM 那边草稿只吃 PIECEWISE 图、吃不了 FULL,见
         vllm/v1/spec_decode/llm_base_proposer.py:419-434;nano-vllm 没有 piecewise,
         只能自己录整图 —— 这是本实现的设计选择,不是照抄 vLLM。)
        """
        hidden = self.draft_hf_config.hidden_size
        self.draft_graphs, self.draft_graph_vars = self.capture_decode_family(
            self.draft_model, hidden, min(self.config.max_num_seqs, 512))
        self.draft_varlen_graphs, self.draft_varlen_vars = self.capture_varlen_family(
            self.draft_model, hidden, 2)
        self.draft_graph_ready = True
