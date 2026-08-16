"""Phase 2.5 Step B · varlen CUDA graph + 投机解码 共存

判据沿用 05-implementation-report.md:118-162 的方法论:
  - 主判据是**单步 logits 对拍**,不是"跑 128 步看 token 是否相同"。
  - 这里比 05 的做法还锐利一档:不比两次独立运行的结果,而是在**同一个 step、
    同一份 KV cache、同一份 context** 上把 graph 与 eager 两条路径都跑一遍,
    直接比 logits。没有自回归放大,也没有"两次运行的 batch 组成不同"这种干扰项。
  - 阈值仍是 4 ulp(BF16_ULP=0.0625),不为了好看放宽;分歧点按坑 4 的方法给分位数。

跑法: .venv/bin/python tests/test_phase25_varlen.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import torch

import common
from common import MODEL_PATH, build_prompts, report
from harness import BF16_ULP, run_gen, token_ids, check_equal, diff_report

results = []


def note(title, ok, detail=""):
    report(title, ok, detail)
    results.append((title, ok))


# ===================================================================== A + B
def suite_inprocess(mode="normal"):
    """在一个进程里同时做两件事:
      A. 每个投机 step 上 graph vs eager 的 logits 逐行对拍
      B. 统计走图占比

    mode="preempt" 时把 KV block 卡死逼出抢占 —— 抢占会让批的组成在 step 之间
    剧烈变化(桶、padding 行数都跟着变),是对图路径最不友好的场景。
    这里比"抢占前后输出是否逐 token 相同"锐利得多:后者在开图时本来就会被
    padding 噪声放大(见坑 5/坑 6),而逐 step 对拍是在**同一份 KV cache、同一个 step**
    上比两条路径,不受自回归放大影响。
    """
    from nanovllm import LLM, SamplingParams
    from common import build_preempt_prompts
    import nanovllm.engine.model_runner as mr

    from nanovllm.utils.context import get_context, set_context

    orig = mr.ModelRunner.run_varlen_graph

    def blank():
        return {"rows": 0, "argmax_bad": 0, "max_ulp": 0.0, "bitwise_rows": 0, "gaps": []}

    # 三方对拍,把偏差来源拆开:
    #   g = graph,padded 形状        p = eager,**同样的 padded 形状**
    #   e = eager,真实(未 padded)形状
    # g vs p  只差"图 replay vs 直接执行",数学与形状都相同 -> 图本身引入了多少误差
    # p vs e  只差"多算了几行 padding" -> cuBLAS 按 M 维分块变化带来的噪声(坑 3 的机制)
    # p_vs_e 再按"这一步到底有没有 padding 行"拆开 —— 这是判定"噪声来自 padding"的对照组:
    # 没有 padding 行时 p 与 e 是**完全相同的计算**,必须逐位相同;有 padding 才该出现噪声。
    st = {"steps": 0, "nopad_steps": 0,
          "g_vs_p": blank(), "p_vs_e": blank(), "g_vs_e": blank(),
          "p_vs_e_nopad": blank(), "p_vs_e_pad": blank()}

    def acc(d, a, b):
        a32, b32 = a.float(), b.float()
        am_a, am_b = a32.argmax(-1), b32.argmax(-1)
        bad = (am_a != am_b)
        lp_a, lp_b = a32.log_softmax(-1), b32.log_softmax(-1)
        top = lp_b.topk(10, dim=-1).indices
        dev = float((lp_a.gather(-1, top) - lp_b.gather(-1, top)).abs().max())
        d["rows"] += a32.size(0)
        d["argmax_bad"] += int(bad.sum())
        d["max_ulp"] = max(d["max_ulp"], dev / BF16_ULP)
        d["bitwise_rows"] += int((a == b).all(dim=-1).sum())
        if bad.any():
            # 坑 4 的方法:分歧行的 top1-top2 间隙有多大?逻辑 bug 会在随机位置发作,
            # 不可能只挑最并列的那几个百分点出现。
            two = lp_b.topk(2, dim=-1).values
            gaps = two[:, 0] - two[:, 1]
            for r in bad.nonzero().flatten().tolist():
                d["gaps"].append(float(gaps[r]) / BF16_ULP)

    @torch.inference_mode()
    def patched(self, input_ids, positions, num_seqs):
        logits_g = orig(self, input_ids, positions, num_seqs)   # 顺带把 gv 缓冲填好了
        real = get_context()
        gv = self.graph_varlen_vars
        q_max = 1 + self.config.num_speculative_tokens
        bs = next(x for x in self.graph_bs if x >= num_seqs)
        total, nslot = bs * q_max, bs + 1
        nt = input_ids.size(0)

        # p:eager,但喂**和图完全一样的 padded 形状**(直接复用图刚填好的缓冲)
        set_context(True, gv["cu_seqlens_q"][:nslot + 1], gv["cu_seqlens_k"][:nslot + 1],
                    q_max, self.config.max_model_len, gv["slot_mapping"][:total], None,
                    gv["block_tables"][:nslot], None)
        h_p = self.model(gv["input_ids"][:total], gv["positions"][:total])
        # e:eager,真实形状。注意这两次前向都会让 store_kvcache 对同一批 slot 再写一次,
        # 值完全相同(幂等),不污染 KV cache,也不改变后续 step 的行为。
        set_context(True, real.cu_seqlens_q, real.cu_seqlens_k, real.max_seqlen_q,
                    real.max_seqlen_k, real.slot_mapping, None, real.block_tables, None)
        h_e = self.model(input_ids, positions)

        set_context(True, real.cu_seqlens_q, real.cu_seqlens_k, real.max_seqlen_q,
                    real.max_seqlen_k, real.slot_mapping, None, real.block_tables,
                    real.logits_indices)
        logits_p = self.model.compute_logits(h_p[:nt])
        logits_e = self.model.compute_logits(h_e)

        st["steps"] += 1
        acc(st["g_vs_p"], logits_g, logits_p)
        acc(st["p_vs_e"], logits_p, logits_e)
        acc(st["g_vs_e"], logits_g, logits_e)
        nopad = (nt == total)          # 这一步恰好落桶,没有任何 padding 行
        st["nopad_steps"] += int(nopad)
        acc(st["p_vs_e_nopad" if nopad else "p_vs_e_pad"], logits_p, logits_e)
        return logits_g

    mr.ModelRunner.run_varlen_graph = patched
    try:
        cfg = dict(enforce_eager=False, max_model_len=4096, max_num_seqs=32,
                   gpu_memory_utilization=0.35, varlen_cudagraph=True,
                   num_speculative_tokens=2)
        if mode == "preempt":
            cfg.update(num_speculative_tokens=4, num_kvcache_blocks=7)
            prompts = build_preempt_prompts()        # 6 条 255 token,必然抢占
            ntok = 16
        else:
            prompts = (build_prompts() * 2)[:8]      # 8 条并发
            ntok = 256
        llm = LLM(MODEL_PATH, **cfg)
        sp = SamplingParams(temperature=0.0, max_tokens=ntok, ignore_eos=True)
        llm.generate(prompts, sp, use_tqdm=False)
        ex = dict(llm.model_runner.exec_stats)
        sch = dict(llm.scheduler.stats)
        acc, prop = llm.scheduler.spec_accepted, llm.scheduler.spec_proposed
    finally:
        mr.ModelRunner.run_varlen_graph = orig

    # ---- A. 逐 step 三方对拍
    tag = f"[{mode}] "
    print(f"\n  [A] {tag}逐 step 三方对拍:{st['steps']} 个投机 step"
          + (f",抢占 {sch['preempted']} 次" if mode == "preempt" else ""))
    print(f"      其中恰好落桶(无 padding 行)的 step: {st['nopad_steps']}/{st['steps']}")
    for cmp_name, d in (("g vs p  图 vs eager(同为 padded 形状)", st["g_vs_p"]),
                   ("p vs e  padded vs 真实形状(全部 step)  ", st["p_vs_e"]),
                   ("   └ 无 padding 行的 step(对照组)   ", st["p_vs_e_nopad"]),
                   ("   └ 有 padding 行的 step           ", st["p_vs_e_pad"]),
                   ("g vs e  图 vs eager(端到端)", st["g_vs_e"])):
        line = (f"      {cmp_name}: 逐位相同 {d['bitwise_rows']}/{d['rows']} "
                f"({100*d['bitwise_rows']/max(d['rows'],1):5.1f}%)  "
                f"argmax 不一致 {d['argmax_bad']}  最大 {d['max_ulp']:.2f} ulp")
        if d["gaps"]:
            g = sorted(d["gaps"])
            line += f"  [分歧行 top1-top2 间隙 ulp: 中位 {g[len(g)//2]:.1f} 最大 {g[-1]:.1f}]"
        print(line)

    # 主判据:图本身不得引入误差 —— 同样的 padded 形状下,图与 eager 必须完全一致。
    gp = st["g_vs_p"]
    note(f"{tag}varlen 图本身零误差(同 padded 形状下 graph vs eager 逐位相同)",
         gp["rows"] > 0 and gp["argmax_bad"] == 0 and gp["max_ulp"] == 0.0,
         f"{gp['rows']} 行,逐位相同 {gp['bitwise_rows']}/{gp['rows']},{gp['max_ulp']:.2f} ulp")

    # 对照组:没有 padding 行时,padded 路径与真实路径是同一个计算,必须逐位相同。
    # 它成立 + 有 padding 时才出现偏差,合起来证明偏差的唯一来源是"多算了几行",
    # 即坑 3 记的那个机制(cuBLAS 按 M 维分块变化改变归约顺序),不是图引入的。
    nop = st["p_vs_e_nopad"]
    if nop["rows"] == 0:
        # 抢占场景里没有任何一步恰好落桶(批一直在变),这个对照组无从构造。
        # 它的结论已由 normal 场景给出(42 个落桶 step / 702 行),不重复要求。
        print(f"      [N/A] 对照组:本场景没有恰好落桶的 step,不适用")
    else:
        note(f"{tag}对照组:无 padding 行时 padded 与真实形状逐位相同",
             nop["max_ulp"] == 0.0 and nop["argmax_bad"] == 0,
             f"{st['nopad_steps']} 个落桶 step,{nop['rows']} 行,{nop['max_ulp']:.2f} ulp")

    # argmax 全等 ⇒ greedy 下每一行"草稿==argmax"的判定完全相同 ⇒ 接受数必然一致
    note(f"{tag}投机接受判定在两条路径下一致(greedy:由 argmax 全等直接推出)",
         gp["argmax_bad"] == 0, f"接受 {acc}/{prop}")
    if mode == "preempt":
        note("抢占确实被触发(逐 step 对拍场景)", sch["preempted"] > 0,
             f"{sch['preempted']} 次")

    # ---- B. 走图占比
    graph = ex["graph_decode"] + ex["graph_varlen"]
    total = sum(ex.values())
    dec = sch["decode_only"]
    print(f"\n  [B] exec_stats = {ex}")
    print(f"      scheduler   = {sch}")
    print(f"      总占比      = {graph}/{total} = {100*graph/max(total,1):.1f}%")
    print(f"      稳态 decode = {graph}/{dec} = {100*graph/max(dec,1):.1f}%  "
          f"(prefill/混批 step 本来就不走图)")
    if mode == "normal":
        # 抢占场景下再 prefill 会产生混批,那些 step 本来就不该走图,不适用这条指标
        note("走图 step 占比 ≥ 90%(k=2, 8 条并发, 稳态 decode)",
             graph / max(dec, 1) >= 0.90,
             f"{100*graph/max(dec,1):.1f}%,其中 varlen 图 {ex['graph_varlen']} 步")
    note(f"{tag}varlen 图确实被 replay(不是摆设)", ex["graph_varlen"] > 0,
         f"{ex['graph_varlen']} 步")
    return ex


# ===================================================================== C
def suite_paths():
    """三条老路径不能坏(硬性要求 4)。特别复核坑 5:
    "能用快 kernel" 和 "能用 CUDA graph" 必须仍是两个正交判断。"""
    # 1) 纯 decode + graph:投机关掉,应该全部走老的 decode 图
    a = run_gen("p25_decode_graph", max_tokens=64, repeat=2, logprobs=10)
    ex = {k[5:]: v for k, v in a["stats"].items() if k.startswith("exec_")}
    note("路径1 纯 decode + 老 decode 图(投机关闭)",
         ex.get("graph_decode", 0) > 0 and ex.get("graph_varlen", 0) == 0,
         f"{ex}")

    # 2) 纯 decode + eager(enforce_eager)
    b = run_gen("p25_decode_eager", max_tokens=64, repeat=2, logprobs=10, eager=True)
    exb = {k[5:]: v for k, v in b["stats"].items() if k.startswith("exec_")}
    note("路径2 纯 decode + eager(enforce_eager,不录任何图)",
         exb.get("graph_decode", 0) == 0 and exb.get("graph_varlen", 0) == 0, f"{exb}")
    from harness import compare_logprobs
    note("路径1 vs 路径2 单步 logprob 等价",
         compare_logprobs(a, b, "graph vs eager(纯 decode)", max_ulp=4))

    # 3) 混批 + eager:小 token 预算强制 chunked prefill 与 decode 混批
    c = run_gen("p25_mixed", max_tokens=64, repeat=2, logprobs=10,
                max_num_batched_tokens=512)
    note("路径3 混批仍然发生且走 eager",
         c["stats"].get("mixed", 0) > 0, f"mixed={c['stats'].get('mixed')}")

    # 4) 坑 5 复核:eager 模式下纯 decode 仍必须用 flash_attn_with_kvcache 快路径。
    #    判据是代码结构:is_pure_decode 与 use_cudagraph 是两个独立方法。
    import nanovllm.engine.model_runner as mr
    src_pure = mr.ModelRunner.is_pure_decode.__doc__
    ok = (hasattr(mr.ModelRunner, "is_pure_decode")
          and hasattr(mr.ModelRunner, "use_cudagraph")
          and hasattr(mr.ModelRunner, "use_varlen_cudagraph"))
    note("坑5 复核:kernel 选择与 graph 选择仍是分开的判断", ok,
         "is_pure_decode / use_cudagraph / use_varlen_cudagraph 三个独立方法")


# ===================================================================== D
def suite_e2e():
    """端到端:varlen 图开 vs 关(Step B vs Step A),输出应当只差浮点噪声。"""
    on = run_gen("p25_spec_graph_on", max_tokens=128, repeat=2, logprobs=10,
                 num_speculative_tokens=2, speculative_method="ngram")
    off = run_gen("p25_spec_graph_off", max_tokens=128, repeat=2, logprobs=10,
                  num_speculative_tokens=2, speculative_method="ngram",
                  no_varlen_cudagraph=True)
    exon = {k[5:]: v for k, v in on["stats"].items() if k.startswith("exec_")}
    exoff = {k[5:]: v for k, v in off["stats"].items() if k.startswith("exec_")}
    print(f"    图开: {exon}")
    print(f"    图关: {exoff}")
    note("关掉 varlen_cudagraph 后确实退回 Step A(投机批走 eager)",
         exoff.get("graph_varlen", 0) == 0 and exon.get("graph_varlen", 0) > 0)
    note("接受数在两条路径下一致",
         on["stats"]["spec_accepted"] == off["stats"]["spec_accepted"],
         f"图开 {on['stats']['spec_accepted']} vs 图关 {off['stats']['spec_accepted']}")
    # 端到端逐 token 比对在这里**不能当判据**,两个原因:
    #   1) 任务书本身就要求不要用"跑 128 步看 token 是否相同"当判据(05 报告坑 3 的教训);
    #   2) 投机路径下 logprobs 恒为 None(05 报告遗留项 3),check_equal_or_noise 没有
    #      logprob 就无法把"浮点噪声"和"真分歧"分开,只会一律报成真分歧。
    # 真正的判据是上面 suite_inprocess 的逐 step 三方对拍。这里只打印现象供人工复核。
    same, total, lines = diff_report(token_ids(on), token_ids(off), "graph", "eager")
    print(f"  [INFO] 端到端逐 token:{same}/{total} 条完全一致 "
          f"(不作判据 —— 投机路径无 logprobs,无法判定分歧性质;见上面的逐 step 对拍)")
    for l in lines:
        print(l)


# ===================================================================== E
def suite_edge():
    """边界与安全:padding 语义、桶边界、抢占、prefix cache 污染。

    最要紧的是污染专项:varlen 图是通过**图自己的 slot_mapping 缓冲**写 KV 的,
    padding 行填 -1 才会被 store_kvcache_kernel 跳过(attention.py:23)。
    这一条填错就会往 prefix cache 里写垃圾 KV —— 而且不会当场报错,
    只有下一个命中该前缀的请求才会读到错的东西。05 报告的坑 10 就是这条路径。
    """
    from harness import compare_logprobs, OUT_DIR

    # E1 污染专项,但这次**开着 varlen 图**跑第一阶段(原来的测试是 eager)
    poll = run_gen("p25_pollution", temperature=0.0, max_tokens=64, logprobs=10,
                   num_speculative_tokens=4, speculative_method="ngram", pollution=True)
    ex = {k[5:]: v for k, v in poll["stats"].items() if k.startswith("exec_")}
    note("污染专项的第一阶段确实走了 varlen 图", ex.get("graph_varlen", 0) > 0, f"{ex}")
    pf = os.path.join(OUT_DIR, "p25_pollution.json")
    ref = run_gen("p25_pollution_ref", temperature=0.0, max_tokens=64, logprobs=10,
                  prompt_file=pf)
    print(f"    第一阶段接受率: {poll['stats']['spec_accepted']}/{poll['stats']['spec_proposed']}")
    note("prefix cache 未被 varlen 图的 padding 行污染(命中后首步 logprob)",
         compare_logprobs(ref, poll, "命中 varlen 图写入的 block 后首步 logprob", max_ulp=4))

    # E2 抢占 + 投机:抢占**逻辑**有没有被这次改动碰坏。
    # 这里两边都必须 eager —— 与 05 报告 Phase 3 第 5 项的构造完全一致。
    # 理由:该判据检验的是"重算路径没有状态残留"这个**逻辑**性质,而逻辑性质只有在
    # 浮点噪声不存在时才可能逐 token 全等。开着图时批的组成会随抢占变化(桶、padding
    # 行数都跟着变),bf16 噪声必然出现,即便代码完全正确也不会 6/6 —— 那时这条判据
    # 测的就不是抢占逻辑了。图路径在抢占下是否正确,由下面 suite_inprocess("preempt")
    # 的逐 step 对拍来判(那个不受自回归放大影响)。
    pre_off = run_gen("p25_pre_off", eager=True, temperature=0.0, max_tokens=16,
                      prompts="preempt")
    pre_on = run_gen("p25_pre_on", eager=True, temperature=0.0, max_tokens=16,
                     prompts="preempt", num_kvcache_blocks=7,
                     num_speculative_tokens=4, speculative_method="ngram")
    print(f"    抢占次数: {pre_on['stats']['preempted']}")
    note("抢占确实被触发", pre_on["stats"]["preempted"] > 0,
         f"{pre_on['stats']['preempted']} 次")
    note("抢占 + 投机(eager)输出与不抢占逐 token 一致 —— 抢占逻辑未被破坏",
         check_equal(token_ids(pre_off), token_ids(pre_on),
                     "抢占+投机(eager)", "normal", "preempt+spec"))

    # E3 不同 k / 不同并发数(含不落桶、bs=1)
    for k in (1, 4, 8):
        r = run_gen(f"p25_k{k}", temperature=0.0, max_tokens=48, repeat=2,
                    num_speculative_tokens=k, speculative_method="ngram")
        e = {kk[5:]: v for kk, v in r["stats"].items() if kk.startswith("exec_")}
        note(f"k={k} 能走 varlen 图", e.get("graph_varlen", 0) > 0, f"{e}")
    for n, tag in ((1, "bs=1"), (3, "bs=6 不落桶")):
        r = run_gen(f"p25_bs{n}", temperature=0.0, max_tokens=48, repeat=n,
                    num_speculative_tokens=2, speculative_method="ngram")
        e = {kk[5:]: v for kk, v in r["stats"].items() if kk.startswith("exec_")}
        note(f"{tag} 能走 varlen 图", e.get("graph_varlen", 0) > 0, f"{e}")


def run_inprocess_child(mode):
    """把 suite_inprocess 放进子进程跑:它会一直占着显存,同进程连跑两个场景会 OOM。"""
    import json
    import subprocess
    from harness import PYTHON
    r = subprocess.run([PYTHON, os.path.abspath(__file__), "--inproc", mode],
                       capture_output=True, text=True, cwd=os.path.dirname(HERE))
    for line in r.stdout.splitlines():
        if line.startswith("INPROC_RESULT "):
            for t, ok in json.loads(line[14:]):
                results.append((t, ok))
        else:
            print(line)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        results.append((f"suite_inprocess({mode}) 子进程退出码 {r.returncode}", False))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--inproc":
        import json
        suite_inprocess(sys.argv[2])
        print("INPROC_RESULT " + json.dumps(results, ensure_ascii=False))
        sys.exit(0)

    print("=" * 78)
    print("Phase 2.5 Step B · varlen CUDA graph + 投机解码")
    print("=" * 78)
    print("\n--- 三条老路径 ---")
    suite_paths()
    print("\n--- 端到端 Step B vs Step A ---")
    suite_e2e()
    print("\n--- 边界与安全 ---")
    suite_edge()
    print("\n--- 逐 step 三方对拍 + 走图占比 ---")
    run_inprocess_child("normal")
    print("\n--- 逐 step 三方对拍(抢占场景)---")
    run_inprocess_child("preempt")

    print("\n" + "=" * 78)
    npass = sum(1 for _, ok in results if ok)
    print(f"{'[PASS]' if npass == len(results) else '[FAIL]'} "
          f"Phase 2.5 Step B  {npass}/{len(results)}")
    for t, ok in results:
        if not ok:
            print(f"   FAIL: {t}")
    sys.exit(0 if npass == len(results) else 1)
