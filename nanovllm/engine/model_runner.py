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
        self.exec_stats = {"graph_decode": 0, "graph_varlen": 0, "eager": 0}
        # warmup_model 会先跑一次 run(),那时 varlen 图还没录 —— 这个标志必须先存在
        self.varlen_graph_ready = False

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
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
            # 投机关掉时 q 恒为 1,投机批根本不会出现,这族图一张都不用录。
            if config.varlen_cudagraph and config.num_speculative_tokens > 0:
                self.capture_varlen_cudagraph()
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
        num_kv_heads = hf_config.num_key_value_heads // parallel.get_tp_size()
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        max_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert max_blocks > 0
        if config.num_kvcache_blocks < 0:
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
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

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

    def sample_speculative(self, seqs: list[Sequence], logits: torch.Tensor, sample_args):
        """投机解码的验证 + 接受判定。

        n-gram 草稿是确定性提议,等价于草稿分布 q = δ_d(在 d 上是 1 的 one-hot)。
        代入 Leviathan 的接受规则:
            以概率 min(1, p(d)/q(d)) = p(d) 接受 d;
            拒绝时从 norm(max(p - q, 0)) = "把 d 挖掉再归一化的 p" 采样。
        这样产出的 token 分布与不投机时逐分布相同。

        temperature=0(greedy)单独走一条路:直接比 token 而不是比概率,
        避免浮点边界问题,也让"开关投机输出必须逐 token 一致"成为严格恒等。
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
        if any_sampling:
            p_d = probs.gather(dim=-1, index=draft_ids.unsqueeze(dim=1)).squeeze(dim=1)
            resid = probs.scatter(dim=-1, index=draft_ids.unsqueeze(dim=1),
                                  src=torch.zeros_like(p_d).unsqueeze(dim=1))
            resid = resid / resid.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            resid_tok = sample_from_probs(resid)
            rand = torch.rand(logits.size(0), device=logits.device)
            p_d_l, resid_tok_l, rand_l = p_d.tolist(), resid_tok.tolist(), rand.tolist()
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
        context = get_context()
        gv = self.graph_varlen_vars
        q_max = 1 + self.config.num_speculative_tokens
        bs = next(x for x in self.graph_bs if x >= num_seqs)
        num_tokens = input_ids.size(0)          # 本轮真实 token 数
        total = bs * q_max                      # 该桶的缓冲区行数
        nslot = 2 * bs                          # bs 个真实槽 + 至多 bs 个 padding 槽
        # max_seqlen_k 是烧进图的 host 标量(= max_model_len)。真实 k 长度超过它时
        # FA 会按 n_blocks_per_split 截断,悄悄少读一段 KV —— 宁可在这里响。
        assert context.max_seqlen_k <= self.config.max_model_len, \
            f"max_seqlen_k={context.max_seqlen_k} 超过图里烧死的 {self.config.max_model_len}"

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

        self.varlen_graphs[bs].replay()
        # compute_logits 在图外,用的是 prepare_batch 产出的真实 logits_indices,
        # 它只指向真实行,padding 槽的行天然被丢弃。
        return self.model.compute_logits(gv["outputs"][:num_tokens])

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, use_graph: bool):
        if not use_graph:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:# decode 要用 CUDA graph
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence]):
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
            token_ids, accepted = self.sample_speculative(seqs, logits, sample_args)
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
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        #上面这六个就是后来的 graph_vars —— replay 时往里写数据的固定地址
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):#这个循环跑 36 次,每次为一个 batch size 造一张图。
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])#把静态张量的切片塞进全局 context。因为模型内部的 attention 层不是从参数拿这些,而是自己 get_context() 
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()#指定用哪个显存池
            self.graphs[bs] = graph# 一张录好的图,里面烧死了
            torch.cuda.synchronize()
            reset_context()

#             输入:  bs = 16
# 产出:  self.graphs[16] = 一张录好的图,里面烧死了
#          input_ids[:16] / positions[:16] / slot_mapping[:16] /
#          context_lens[:16] / block_tables[:16] / outputs[:16]
#          这六块的显存地址,以及 28 层全部 kernel 的调用序列

        self.graph_vars = dict(
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
        config = self.config
        hf_config = config.hf_config
        q_max = 1 + config.num_speculative_tokens
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
        outputs = torch.zeros(max_total, hf_config.hidden_size)
        # replay 时用来一次算出全部 padding 槽的 cu_seqlens:0, q_max, 2*q_max, ...
        pad_ramp = torch.arange(0, max_nslot + 1, dtype=torch.int32) * q_max
        # 捕获时的 warmup 那一次会真的读 KV:让每条 seq 都指向 block 0,读一条合法的垃圾 KV。
        # (replay 时这一列会被真实 block_table 覆盖;sink 行 k 长度为 0,从不被读 ——
        #  第一阶段实测 block_tables 的 -1 padding 与填 0 逐位相同,即根本没读。)
        block_tables[:, 0] = 0

        self.varlen_graphs = {}
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
            outputs[:total] = self.model(input_ids[:total], positions[:total])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:total] = self.model(input_ids[:total], positions[:total])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.varlen_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_varlen_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables,
            outputs=outputs,
            pad_ramp=pad_ramp,
        )
        self.varlen_graph_ready = True
