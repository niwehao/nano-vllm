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
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)#所有 GPU 一起跑一次前向并采样,拿回每条 seq 的下一个 token。跳转到model_runner.py:run() 
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

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
            # output    : 本轮"刚刚结束"的 seq,元素是 (seq_id, 该 seq 生成的全部 token)
            # num_tokens: 本轮处理的 token 数,正数表示这轮是 prefill,负数表示是 decode
            output, num_tokens = self.step()

            if num_tokens > 0:
                # prefill 轮:num_tokens 是本轮实际计算的 prompt token 总数
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                # decode 轮:num_tokens = -(本轮 seq 条数),每条恰好产 1 个 token,故取反即 token 数
                decode_throughput = -num_tokens / (perf_counter() - t)

            # 挂在进度条尾部显示。两个变量在循环外初始化,每轮只更新其中一个,
            # 另一个保留上一次的旧值,所以显示的是各阶段"最近一次"的瞬时吞吐
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })

            # 收集本轮完成的结果。完成顺序由各请求的生成长度决定,与输入顺序无关,
            # 所以先按 seq_id 存进 dict,循环结束后再排序还原成输入顺序
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)                 # 进度条按"完成的请求数"推进,不是按 token 数
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
