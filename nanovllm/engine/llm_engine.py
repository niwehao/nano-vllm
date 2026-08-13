import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        # 统一调度后一个 batch 里可能同时有 prefill chunk 和 decode,
        # 所以不再用"num_tokens 的正负号"区分两类,而是分别统计后一起返回。
        seqs = self.scheduler.schedule()
        num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        num_decode_tokens = sum(seq.num_scheduled_tokens for seq in seqs if not seq.is_prefill)
        token_ids, logprobs, accepted = self.model_runner.call("run", seqs)#所有 GPU 一起跑一次前向并采样,拿回每条 seq 的下一个 token。跳转到model_runner.py:run() 
        self.scheduler.postprocess(seqs, token_ids, logprobs, accepted)
        outputs = [(seq.seq_id, seq) for seq in seqs if seq.is_finished]
        return outputs, num_prefill_tokens, num_decode_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():          # 只要 scheduler 的 waiting / running 还有 seq 就继续
            t = perf_counter()                 # 记下本轮起始时刻,用于算吞吐

            # 推进一轮:调度一批 seq -> 跑一次模型前向 -> 采样出 token -> 后处理
            # [OLD ↓ 下面 2 行] 原注释保留原文不动;它描述的是改动前的 step() 协议,现已过时
            # output    : 本轮"刚刚结束"的 seq,元素是 (seq_id, 该 seq 生成的全部 token)
            # num_tokens: 本轮处理的 token 数,正数表示这轮是 prefill,负数表示是 decode
            # [NEW] 现状(Phase 2 统一调度后):
            #   output            : 元素改成 (seq_id, seq 对象) —— generate() 要从中同时取
            #                       completion_token_ids 和 completion_logprobs
            #   num_prefill_tokens: 本轮 prefill chunk 的 token 总数
            #   num_decode_tokens : 本轮 decode 侧的 token 数(投机时一条 seq 可能不止 1 个)
            #   一个 batch 里 prefill 和 decode 可以同时存在,所以不能再用一个数的正负号区分
            output, num_prefill_tokens, num_decode_tokens = self.step()

            # [OLD ↓ 下面 2 行] 这两行原注释属于改动前的 if/else 两个分支,分支已重写,
            #                   注释按原文原缩进保留在此,内容未动
                # prefill 轮:num_tokens 是本轮实际计算的 prompt token 总数
                # decode 轮:num_tokens = -(本轮 seq 条数),每条恰好产 1 个 token,故取反即 token 数
            # [NEW] 统一调度后 prefill 和 decode 可能出现在同一轮里,两个吞吐各自独立更新。
            #       不再是 if/else 二选一,而是两个独立的 if。
            dt = perf_counter() - t
            if num_prefill_tokens > 0:
                prefill_throughput = num_prefill_tokens / dt
            if num_decode_tokens > 0:
                decode_throughput = num_decode_tokens / dt

            # [OLD ↓ 下面 2 行] 原注释保留原文不动;"每轮只更新其中一个"已过时
            # 挂在进度条尾部显示。两个变量在循环外初始化,每轮只更新其中一个,
            # 另一个保留上一次的旧值,所以显示的是各阶段"最近一次"的瞬时吞吐
            # [NEW] 现状:混批的那些轮里 prefill 和 decode 两个吞吐会同时更新;
            #       只有纯 prefill / 纯 decode 的轮才仍然是"只更新一个,另一个留旧值"。
            #       "显示各阶段最近一次的瞬时吞吐"这个语义没变。
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 收集本轮完成的结果。完成顺序由各请求的生成长度决定,与输入顺序无关,
            # 所以先按 seq_id 存进 dict,循环结束后再排序还原成输入顺序
            for seq_id, finished_seq in output:
                outputs[seq_id] = finished_seq
                pbar.update(1)                 # 进度条按"完成的请求数"推进,不是按 token 数
        pbar.close()
        finished = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        return [{"text": self.tokenizer.decode(seq.completion_token_ids),
                 "token_ids": seq.completion_token_ids,
                 "logprobs": seq.completion_logprobs if seq.num_logprobs is not None else None}
                for seq in finished]
