"""Phase 3B 的内存安全检查。

跑法(**环境变量和 memcheck 两个都不能少**):

    PYTORCH_NO_CUDA_MEMORY_CACHING=1 ~/cuda-12.8/bin/compute-sanitizer --tool memcheck \\
        .venv/bin/python tests/phase3b_probes/memcheck_draft.py <mode>

少了 PYTORCH_NO_CUDA_MEMORY_CACHING=1,PyTorch 的 caching allocator 会做一次大
cudaMalloc 再自己切分,池内越界 memcheck 完全看不见 —— 07 报告坑 10 就是这么
差点漏掉一处真实的写越界(第一次跑报的是 ERROR SUMMARY: 0 errors)。

## 为什么不能直接把整个 LLM 塞进 memcheck 跑

第一版就是那么写的,结果是 `CUDA error: operation failed due to a previous error
during capture` —— **关掉 caching allocator 之后每次分配都是裸 cudaMalloc,
而 cudaMalloc 在 stream capture 期间是非法操作**,于是录图必然失败。
这不是我们要找的越界,是这套组合本身不兼容。

所以拆成两个 mode,合起来覆盖本次新增的全部"按自己算的下标往固定缓冲区写":

  varlen —— 不建模型、不录图,直接复刻 ModelRunner.replay_varlen 对**草稿那族图**
            (q_max=2,与目标族的 1+k 不同,padding 槽的 cu_seqlens 算术要重算)
            的下标算法,每个构型真的调一次 flash_attn_varlen_func。
            这正是 07 报告 probe_sink_memcheck.py 的 "prod" 模式的做法。
  eager  —— 起一个真的 LLM 但 enforce_eager=True(一张图都不录,于是没有 capture),
            让 prepare_draft_first / prepare_draft_decode 算出来的 slot_mapping
            真的喂给 store_kvcache 的 triton kernel 写 KV cache。
            slot 越界会直接写坏别的 seq 的 KV,这一层只有真跑才查得到。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B")
MODE = sys.argv[1] if len(sys.argv) > 1 else "varlen"

if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1":
    print("[warn] 没设 PYTORCH_NO_CUDA_MEMORY_CACHING=1,池内越界会被 allocator 挡住")


def run_varlen():
    """复刻草稿那族 varlen 图的下标算法(q_max=2),直接调 FA。"""
    import torch
    from flash_attn import flash_attn_varlen_func

    DEV, DTYPE = "cuda", torch.bfloat16
    H, D, BLOCK = 8, 128, 256
    SCALE = D ** -0.5
    NBLK = 64
    k_cache = torch.randn(NBLK, BLOCK, H, D, dtype=DTYPE, device=DEV)
    v_cache = torch.randn(NBLK, BLOCK, H, D, dtype=DTYPE, device=DEV)

    # 与 model_runner.capture_cudagraph 的分桶一致
    GRAPH_BS = [1, 2, 4, 8, 16, 32]
    q_max = 2                      # 草稿族烧死的 q_max(第一阶段 E4 证明上界就是 2)
    ncase = 0
    for num_seqs in (1, 2, 3, 5, 8, 9, 16, 17, 32):
        bs = next((x for x in GRAPH_BS if x >= num_seqs), None)
        if bs is None:
            continue
        total, nslot = bs * q_max, 2 * bs
        # q 长度的三种极端:全 1(没人补格)、全 2(每条都在补格)、交替
        for pattern in ("all1", "all2", "mixed"):
            if pattern == "all1":
                ql = [1] * num_seqs
            elif pattern == "all2":
                ql = [2] * num_seqs
            else:
                ql = [(2 if i % 2 == 0 else 1) for i in range(num_seqs)]
            nt = sum(ql)
            kl = [300 + 17 * i for i in range(num_seqs)]     # 参差的 k 长度
            # ---- 复刻 replay_varlen 的填法 ----
            cq, ck = [0], [0]
            for i in range(num_seqs):
                cq.append(cq[-1] + ql[i])
                ck.append(ck[-1] + kl[i])
            last_k = ck[-1]
            for t in range(1, nslot + 1 - num_seqs):
                cq.append(min(nt + t * q_max, total))
                ck.append(last_k + (cq[-1] - nt))
            assert len(cq) == nslot + 1 == len(ck), (len(cq), nslot + 1)
            assert cq[-1] == total, (cq[-1], total)
            # 两条不能松的约束,在这里直接断言出来
            for t in range(num_seqs, nslot):
                qlen, klen = cq[t + 1] - cq[t], ck[t + 1] - ck[t]
                assert qlen == klen, (t, qlen, klen)         # k 长度 = q 长度
                assert 0 <= qlen <= q_max, (t, qlen)         # 不超过 q_max
            q = torch.randn(total, H, D, dtype=DTYPE, device=DEV)
            cqt = torch.tensor(cq, dtype=torch.int32, device=DEV)
            ckt = torch.tensor(ck, dtype=torch.int32, device=DEV)
            bt = torch.full((nslot, 16), -1, dtype=torch.int32, device=DEV)
            for i in range(num_seqs):
                nb = (kl[i] + BLOCK - 1) // BLOCK
                bt[i, :nb] = torch.arange(nb, dtype=torch.int32, device=DEV)
            bt[num_seqs:nslot, 0] = 0
            o = flash_attn_varlen_func(q, k_cache, v_cache, cu_seqlens_q=cqt,
                                       cu_seqlens_k=ckt, max_seqlen_q=q_max,
                                       max_seqlen_k=4096, softmax_scale=SCALE,
                                       causal=True, block_table=bt)
            torch.cuda.synchronize()
            assert torch.isfinite(o).all(), f"出现非有限值 {num_seqs=} {pattern=}"
            ncase += 1
    print(f"[varlen] {ncase} 个构型全部跑完(q_max=2),真实行与 padding 行均为有限值")


def run_eager():
    """真 LLM,enforce_eager=True 所以没有 capture;查 slot_mapping 的下标算术。"""
    from nanovllm import LLM, SamplingParams
    llm = LLM(MODEL, num_speculative_tokens=2, speculative_method="model",
              speculative_model=MODEL, gpu_memory_utilization=0.45,
              max_model_len=1024, max_num_seqs=8, max_num_batched_tokens=1024,
              enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True)
    # 长度参差:让批内 q 长度、cu_seqlens_k、block_table 长度都不整齐
    prompts = [list(range(100, 100 + 37 + 11 * i)) for i in range(4)]
    outs = llm.generate(prompts, sp, use_tqdm=False)
    ex = dict(llm.model_runner.exec_stats)
    acc, prop = llm.scheduler.spec_accepted, llm.scheduler.spec_proposed
    print(f"[eager] lens={[len(o['token_ids']) for o in outs]}")
    print(f"[eager] exec={ex}")
    print(f"[eager] accept={acc}/{prop}")
    assert ex["draft_eager"] > 0, "草稿一次都没跑,这次检查没意义"
    # 自草稿 100% 接受 => 每轮都是 a==k => 每轮第一次前向都走 q=2 的补格路径
    assert acc == prop, f"自草稿接受率不是 100%({acc}/{prop}),q=2 那条路可能没被覆盖"


if __name__ == "__main__":
    if MODE == "varlen":
        run_varlen()
    elif MODE == "eager":
        run_eager()
    else:
        raise SystemExit(f"unknown mode {MODE!r}; use 'varlen' or 'eager'")
    print("done")
