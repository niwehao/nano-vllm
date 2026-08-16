"""E2(显存预算)+ E3(草稿/目标成本比)。

不走 LLMEngine,直接按 model_runner 的做法搭最小环境(建模型 → 分配 paged KV →
录 decode 图),这样能把"权重占多少""一步 decode 多久"单独量出来,不被调度器、
采样器、tokenizer 的开销污染。

一个进程只测一个模型(8B + 0.6B 同时放得下,但两者交替测会互相影响显存峰值和
时钟),结果打到 stdout 的 RESULT 行,由 __main__ 汇总。

计时方法沿用 07 报告坑 4 的结论:图只建一次、全部先热身、A/B 交错重复取最小值。
一测一建图会量出 0.489× 这种不可能的比值。
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

TARGET = os.path.expanduser("~/huggingface/Qwen3-8B")
DRAFT = os.path.expanduser("~/huggingface/Qwen3-0.6B")
PY = os.path.join(ROOT, ".venv", "bin", "python")

BLOCK = 256
MAX_MODEL_LEN = 4096
BS_LIST = [1, 8, 32]
# 上下文长度必须扫:草稿模型(28 层 × 8 kv 头)的 KV 只比目标(36 层 × 8 kv 头)小
# 22%,所以"草稿便宜"这件事完全来自权重(1.2GB vs 16.4GB),上下文一长、KV 读取
# 压过权重读取,成本比就会塌掉。只在一个长度上测会给出误导性的结论。
CTX_LIST = [512, 1024, 2048, 4096]
K = 2                      # E3 的成本比按 num_speculative_tokens=2 算


def build(path):
    import torch
    from transformers import AutoConfig
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    from nanovllm.utils.loader import load_model

    hf = AutoConfig.from_pretrained(path)
    torch.set_default_dtype(hf.dtype)
    torch.set_default_device("cuda")
    before = torch.cuda.memory_allocated()
    model = Qwen3ForCausalLM(hf)
    load_model(model, path)
    weights = torch.cuda.memory_allocated() - before
    return hf, model, weights


def kv_block_bytes(hf):
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    return (2 * hf.num_hidden_layers * BLOCK
            * hf.num_key_value_heads * head_dim * hf.dtype.itemsize)


def child(path, tag):
    import torch
    from nanovllm.utils.context import set_context, reset_context

    hf, model, weights = build(path)
    blk = kv_block_bytes(hf)

    # 只要够跑 bs=32 × 4096 上下文即可,不吃满显存 —— E2 的预算是算出来的,不是测出来的
    num_blocks = 32 * (MAX_MODEL_LEN // BLOCK) + 8
    head_dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
    kv = torch.empty(2, hf.num_hidden_layers, num_blocks, BLOCK,
                     hf.num_key_value_heads, head_dim)
    lid = 0
    for m in model.modules():
        if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
            m.k_cache, m.v_cache = kv[0, lid], kv[1, lid]
            lid += 1
    assert lid == hf.num_hidden_layers

    max_bs = max(BS_LIST)
    nb = MAX_MODEL_LEN // BLOCK
    block_tables = torch.arange(max_bs * nb, dtype=torch.int32).view(max_bs, nb)
    slot = (block_tables[:, -1] * BLOCK + BLOCK - 1).to(torch.int32)
    ids = torch.zeros(max_bs * (K + 1), dtype=torch.int64)
    pos = torch.full((max_bs * (K + 1),), MAX_MODEL_LEN - 1, dtype=torch.int64)

    def decode_once(bs, ctx):
        set_context(False, slot_mapping=slot[:bs],
                    context_lens=torch.full((bs,), ctx, dtype=torch.int32),
                    block_tables=block_tables[:bs])
        h = model(ids[:bs], pos[:bs])
        logits = model.compute_logits(h)
        tok = logits.argmax(dim=-1)
        reset_context()
        return tok

    # ---- 录 decode 图(bs 每档一张),形态与 model_runner.capture_cudagraph 一致 ----
    g_ids = torch.zeros(max_bs, dtype=torch.int64)
    g_pos = torch.zeros(max_bs, dtype=torch.int64)
    g_slot = torch.zeros(max_bs, dtype=torch.int32)
    g_ctx = torch.zeros(max_bs, dtype=torch.int32)
    g_bt = torch.zeros(max_bs, nb, dtype=torch.int32)
    g_out = torch.zeros(max_bs, hf.hidden_size)
    graphs, pool = {}, None
    with torch.inference_mode():
        for bs in reversed(BS_LIST):
            set_context(False, slot_mapping=g_slot[:bs], context_lens=g_ctx[:bs],
                        block_tables=g_bt[:bs])
            g_out[:bs] = model(g_ids[:bs], g_pos[:bs])
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool):
                g_out[:bs] = model(g_ids[:bs], g_pos[:bs])
            pool = pool or g.pool()
            graphs[bs] = g
            torch.cuda.synchronize()
            reset_context()

    def graph_step(bs, ctx):
        g_ids[:bs] = ids[:bs]
        g_pos[:bs] = pos[:bs]
        g_slot.fill_(-1)
        g_slot[:bs] = slot[:bs]
        g_ctx.zero_()
        g_ctx[:bs] = ctx
        g_bt[:bs, :nb] = block_tables[:bs]
        graphs[bs].replay()
        return model.compute_logits(g_out[:bs]).argmax(dim=-1)

    def varlen_step(bs, q, ctx):
        n = bs * q
        cq = torch.arange(0, bs + 1, dtype=torch.int32) * q
        ck = torch.arange(0, bs + 1, dtype=torch.int32) * ctx
        sm = torch.zeros(n, dtype=torch.int32)
        set_context(True, cq, ck, q, ctx, sm, None, block_tables[:bs],
                    torch.arange(n, dtype=torch.int64))
        h = model(ids[:n], pos[:n])
        out = model.compute_logits(h).argmax(dim=-1)
        reset_context()
        return out

    def timeit(fn, rounds=12, inner=5):
        import time
        best = float("inf")
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        for _ in range(rounds):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(inner):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / inner)
        return best * 1e3          # ms

    res = {"tag": tag, "path": path, "weights": weights, "block_bytes": blk,
           "layers": hf.num_hidden_layers, "hidden": hf.hidden_size,
           "vocab": hf.vocab_size, "times": {}}
    with torch.inference_mode():
        for bs in BS_LIST:
            for ctx in CTX_LIST:
                t = res["times"]
                t[f"graph_q1|bs{bs}|ctx{ctx}"] = timeit(
                    lambda bs=bs, c=ctx: graph_step(bs, c))
                t[f"eager_q1|bs{bs}|ctx{ctx}"] = timeit(
                    lambda bs=bs, c=ctx: decode_once(bs, c))
                t[f"eager_varlen_q{K+1}|bs{bs}|ctx{ctx}"] = timeit(
                    lambda bs=bs, c=ctx: varlen_step(bs, K + 1, c))
                t[f"eager_varlen_q1|bs{bs}|ctx{ctx}"] = timeit(
                    lambda bs=bs, c=ctx: varlen_step(bs, 1, c))
    print("RESULT " + json.dumps(res))


def run_child(path, tag):
    r = subprocess.run([PY, os.path.abspath(__file__), "--child", path, tag],
                       capture_output=True, text=True, cwd=ROOT)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[7:])
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])
    raise SystemExit(f"{tag} 子进程失败 rc={r.returncode}")


def main():
    t = run_child(TARGET, "target-8B")
    d = run_child(DRAFT, "draft-0.6B")
    GiB = 2 ** 30
    MiB = 2 ** 20
    total_gpu = 46068 * 2 ** 20          # L40S 标称
    util = 0.90

    print("=" * 96)
    print("E2 显存预算(L40S 46068 MiB, gpu_memory_utilization=0.90)")
    print("=" * 96)
    print(f"{'项':<34}{'字节':>18}{'GiB':>10}")
    for r in (t, d):
        print(f"{'权重 ' + r['tag']:<34}{r['weights']:>18,}{r['weights']/GiB:>10.3f}")
    wsum = t["weights"] + d["weights"]
    print(f"{'权重合计':<34}{wsum:>18,}{wsum/GiB:>10.3f}")
    print()
    for r in (t, d):
        print(f"{'KV 每 block(256 tok) ' + r['tag']:<34}"
              f"{r['block_bytes']:>18,}{r['block_bytes']/MiB:>10.2f} MiB")
    bsum = t["block_bytes"] + d["block_bytes"]
    print(f"{'KV 每 block 合计':<34}{bsum:>18,}{bsum/MiB:>10.2f} MiB")
    print()
    budget = int(total_gpu * util) - wsum
    print(f"{'可用 = 46068MiB*0.90 - 权重':<34}{budget:>18,}{budget/GiB:>10.3f}"
          "   (未扣激活峰值,是上界)")
    for label, per in (("只放目标 KV", t["block_bytes"]), ("两套 KV", bsum)):
        nblk = budget // per
        print(f"  {label:<20} block 数 {nblk:>6}  = {nblk*BLOCK:>9,} token"
              f"  = {nblk*BLOCK//MAX_MODEL_LEN:>4} 条 {MAX_MODEL_LEN} 长的并发")
    print(f"  加草稿 KV 后 block 数缩水到 "
          f"{t['block_bytes']/bsum*100:.1f}%(草稿 KV 占 {d['block_bytes']/bsum*100:.1f}%)")

    print()
    print("=" * 96)
    print("E3 单步成本(ms,min of 12×5)")
    print("=" * 96)
    print(f"{'形态':<18}{'bs':>4}{'ctx':>6}{'目标8B':>10}{'草稿0.6B':>11}{'草稿/目标':>11}")
    for k in t["times"]:
        shape, bs, ctx = k.split("|")
        a, b = t["times"][k], d["times"][k]
        print(f"{shape:<18}{bs[2:]:>4}{ctx[3:]:>6}{a:>10.3f}{b:>11.3f}{b/a:>11.3f}")

    print()
    print("=" * 96)
    print(f"E3 交易是否划算(k={K})")
    print("=" * 96)
    print("  草稿一轮 = k 次前向。乐观口径:全部走图,按 graph_q1 计价(实现里第一次是")
    print("  q<=2 的 varlen 图,略贵一点,见报告)。目标一步按 graph_q1 计价。")
    print()
    print(f"{'bs':>4}{'ctx':>6}{'草稿k步':>10}{'目标1步':>10}{'比值r':>9}"
          f"{'回本需tok/步':>13}{'均匀近似α':>11}{'不图化时比值':>13}")
    for bs in BS_LIST:
        for ctx in CTX_LIST:
            tg = t["times"][f"graph_q1|bs{bs}|ctx{ctx}"]
            dg = d["times"][f"graph_q1|bs{bs}|ctx{ctx}"]
            de = d["times"][f"eager_q1|bs{bs}|ctx{ctx}"]
            r = K * dg / tg
            need = 1 + r
            # 每步净产出 = 1 + Σ_{i=1..k} α^i(几何接受模型);解出回本所需 α
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                y = sum(mid ** i for i in range(1, K + 1)) + 1
                lo, hi = (mid, hi) if y < need else (lo, mid)
            alpha = (lo + hi) / 2
            print(f"{bs:>4}{ctx:>6}{K*dg:>10.3f}{tg:>10.3f}{r:>9.3f}"
                  f"{need:>13.3f}{alpha:>11.3f}{K*de/tg:>13.3f}")
    json.dump({"target": t, "draft": d},
              open(os.path.join(HERE, "e23_results.json"), "w"), indent=1)
    print(f"\n原始数据 -> {os.path.join(HERE, 'e23_results.json')}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        main()
