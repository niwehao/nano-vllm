"""M6 · EP 三配置基准（EP=1 单机 / EP=2 + nccl / EP=2 + ibgda）。

风格对齐 tests/bench.py：先跑两轮丢弃的预热再测——首轮里有 kernel autotune、
NCCL 建链、cuBLAS 选算法，全是一次性开销，混进来会把结论带偏（Plan-1-2-3 的坑 9
就是被这 ~700ms 骗过一次）。

场景 A：8 条请求稳态 decode 448 token —— 纯 decode，每步 T=8，最能体现小消息通信延迟。
场景 B：A 的基础上在第 10 步注入 3 条 4000-token prompt（max_num_batched_tokens=512）
        —— 混批，prefill 被切成 8 块，每块都要过 MoE，看 TBT 尾延迟被拖成什么样。

用法（EP=1）：  .venv/bin/python tests/bench_ep.py --out tests/out/bench_ep1.json
用法（EP=2）：  scripts/launch_both.sh tests/bench_ep.py --ep-size 2 --out ...
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common                                                        # noqa: E402
from common import corpus_tokens                                     # noqa: E402
from nanovllm import LLM, SamplingParams                             # noqa: E402


def ib_counters():
    """本机 RoCE 口发送字节（port_xmit_data 单位是 4 字节）。"""
    try:
        with open("/sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_data") as f:
            return int(f.read()) * 4
    except OSError:
        return 0


def make_llm(args):
    return LLM(os.path.expanduser(args.model),
               ep_size=args.ep_size,
               node_rank=0,
               master_addr=args.master_addr,
               master_port=args.master_port,
               ep_transport=args.ep_transport,
               enforce_eager=True,
               max_num_batched_tokens=args.mnbt,
               max_model_len=8192,
               max_num_seqs=args.max_num_seqs,
               gpu_memory_utilization=args.gpu_util)


def drive(llm, prompts, sp, inject=None, inject_at=None, inject_sp=None):
    """手动驱动 step 循环，逐步记 TBT 与吞吐。inject 在第 inject_at 步加进去。"""
    for p in prompts:
        llm.add_request(p, sp)
    tbt, n_dec, n_pre = [], 0, 0
    step = 0
    t_start = time.perf_counter()
    while not llm.is_finished():
        if inject is not None and step == inject_at:
            for p in inject:
                llm.add_request(p, inject_sp)
        t0 = time.perf_counter()
        _, npre, ndec = llm.step()
        dt = time.perf_counter() - t0
        n_pre += npre
        n_dec += ndec
        if ndec:                       # 只有产出 token 的步才计 TBT
            tbt.append(dt * 1e3)
        step += 1
    total = time.perf_counter() - t_start
    tbt.sort()
    def p(q):
        return tbt[min(int(len(tbt) * q), len(tbt) - 1)] if tbt else float("nan")
    return {"steps": step, "wall_s": total,
            "prefill_tokens": n_pre, "decode_tokens": n_dec,
            "decode_tok_s": n_dec / total, "prefill_tok_s": n_pre / total if n_pre else 0.0,
            "tbt_p50_ms": p(0.50), "tbt_p99_ms": p(0.99)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="~/huggingface/tiny-qwen3-moe")
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--master-addr", default="192.168.100.2")
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--ep-transport", default="nccl")
    ap.add_argument("--mnbt", type=int, default=512)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-util", type=float, default=0.5)
    ap.add_argument("--decode-tokens", type=int, default=448)
    args = ap.parse_args()

    c = list(corpus_tokens())
    short = [c[i * 17: i * 17 + 48] for i in range(8)]        # 8 条短 prompt → 稳态 decode
    longp = [c[:2000] + c[:2000] for _ in range(3)]           # 3 条 4000 token
    llm = make_llm(args)

    sp_dec = SamplingParams(temperature=0.0, max_tokens=args.decode_tokens, ignore_eos=True)
    sp_pre = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)

    # 预热两轮，丢弃
    for _ in range(2):
        drive(llm, [c[:64] for _ in range(4)],
              SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True))

    if os.environ.get("NANOVLLM_EP_TIMING") == "1":
        from nanovllm.models.qwen3_moe import reset_ep_timing
        reset_ep_timing()                      # 预热的打点全丢掉，只留稳态

    res = {"config": vars(args)}
    ib0 = ib_counters()
    res["A_decode"] = drive(llm, short, sp_dec)
    res["A_roce_MB"] = (ib_counters() - ib0) / 1e6

    ib0 = ib_counters()
    res["B_mixed"] = drive(llm, short, sp_dec, inject=longp, inject_at=10, inject_sp=sp_pre)
    res["B_roce_MB"] = (ib_counters() - ib0) / 1e6
    res["sched_stats"] = dict(llm.scheduler.stats)

    if os.environ.get("NANOVLLM_EP_TIMING") == "1":
        from nanovllm.models.qwen3_moe import report_ep_timing
        res["ep_layer_timing"] = {str(k): v for k, v in report_ep_timing().items()}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
