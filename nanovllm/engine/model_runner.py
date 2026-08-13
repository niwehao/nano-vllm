import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler, compute_probs, sample_from_probs
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)#权重是未初始化的空张量,但形状已是切分后的
        load_model(self.model, config.model)## 从 safetensors 填入
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:#给 TP 建立控制通道,并让 worker 进入待命状态。
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)#创建一块 1MB 的共享内存,取名 "nanovllm"。然后 barrier 等所有 rank 到齐。
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")#按名字 attach 到同一块内存
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()#阻塞等 rank 0 的指令,收到就执行,执行完继续等。
            self.call(method_name, *args)#一次迭代 = 一次完整前向+生成 batch 个token
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):#效果是 8 张卡同时进入 run,各算各的那份权重分片,靠 NCCL all-reduce 汇总
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)#共享内存
        method = getattr(self, method_name, None)
        return method(*args)# model_runner.py:196

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
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        max_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert max_blocks > 0
        if config.num_kvcache_blocks < 0:
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
        input_ids, positions = self.prepare_decode(seqs) if pure_decode else self.prepare_batch(seqs)
        #这里context是真数据,之前只是一个烧录地址的动作
        sample_args = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, use_graph)
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
