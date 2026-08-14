# Phase 2.5 · CUDA graph 适配

单列的硬骨头。现状:model_runner.py:223-265 按 bs 分桶捕获 36 张图
(`graph_bs = [1,2,4,8] + range(16, 512+1, 16)`),图里烧死的是 **flash_attn_with_kvcache 的 decode 形态**
(set_context(False, ...),model_runner.py:242)。

## 问题拆解

混批后 step 分三类:

| 批类型 | 出现时机 | 能否用现有图 |
|---|---|---|
| 纯 decode | 稳态,占大头 | ✅ 形态没变 |
| 纯 prefill | 冷启动 | 本来就不走图(model_runner.py:197) |
| 混合批 | prefill 进行中 | ❌ q 长度可变,图化不了 |

所以"36 张图全部作废"只在**追求单一执行路径**时成立。分两步走:

## Step A(先做,低风险):双路径共存

Phase 2 已经铺好:`run_model` 里判断纯 decode 批 → 走老 graph 路径(prepare_decode + flash_attn_with_kvcache 的图);混合批 → eager 走统一 varlen。

需要的改动只有:
1. `run()` 里按 `is_pure_decode` 选 prepare_decode / prepare_batch(Phase 2 已描述);
2. `capture_cudagraph` 原样保留;
3. attention.forward 保留 else 分支(flash_attn_with_kvcache)。

代价:代码里长期存在两条 attention 路径、两套 prepare。稳态负载(prefill 已排空)下性能与改造前持平;prefill 与 decode 交叠的窗口期 decode 掉速(eager)。**这一步做完就可以发版**,Step B 是纯优化。

## Step B(后做,消灭双路径):用 varlen 形态重录图

目标:图里烧 `flash_attn_varlen_func(q, k_cache, v_cache, ..., block_table)`、每 seq q_len=1,
从而 prepare_decode / flash_attn_with_kvcache / context_lens 全部退役,attention 只剩一条路。

### 静态缓冲区设计(capture_cudagraph 重写)

| 缓冲 | 形状 | replay 前是否要刷新值 |
|---|---|---|
| input_ids / positions | [max_bs] | 是 |
| cu_seqlens_q | [max_bs+1] | **否**,恒为 arange(0, bs+1)(每 seq q_len=1),捕获时就填死 |
| cu_seqlens_k | [max_bs+1] | 是,= context_lens 的前缀和,CPU 算好 copy 进去 |
| slot_mapping | [max_bs] | 是,padding 位填 -1(store_kvcache_kernel 已处理 -1 跳过,attention.py:23) |
| block_tables | [max_bs, max_num_blocks] | 是 |
| outputs | [max_bs, hidden] | 图写出 |

host 侧标量参数:
- `max_seqlen_q = 1`:每 seq 恒 1,捕获时烧死,安全;
- `max_seqlen_k`:**这是最大的坑**。它是 host int,capture 时烧进 kernel 启动参数。对 q 侧 grid 无影响
  (grid 由 max_seqlen_q 决定),但 FA 内部可能用它做 num_splits 等调度启发。
  方案:capture 时传 `max_model_len`(最坏情况)。
  需要验证两件事:(a) 正确性——传偏大的 max_seqlen_k 结果仍正确(kernel 实际按 cu_seqlens_k 读);
  (b) 性能——偏大值是否让 kernel 调度变差。验证方法:eager 下分别传真实值和 max_model_len,
  比对输出与耗时。如果性能损失大,退路是按 max_seqlen_k 也分桶(图数量 ×桶数,先不做)。

### padding 行的处理

bs 不在桶上时按桶垫齐(现有做法)。垫的行:
- slot_mapping = -1:不写 cache,安全;
- cu_seqlens_k 的 padding 段:设为与前一项相等(k 长度 0)。**风险点**:FA varlen 对 seqlen_k=0 的行
  输出可能是 NaN/未定义——输出反正被丢弃(compute_logits 只取 :bs,model_runner.py:212),
  但要确认 NaN 不会 trap。若有问题,退路:padding 行 k 长度设 1、block_tables 指向 block 0
  (读一条垃圾 KV,输出仍被丢弃,但行为定义良好)。capture 前用 eager 实测选定其一。

### 捕获与 replay 的代码改动

- `capture_cudagraph`:set_context 改为传 varlen 形态
  (`use_varlen=True, cu_seqlens_q=arange 切片, cu_seqlens_k=buffer 切片, max_seqlen_q=1, max_seqlen_k=max_model_len, slot_mapping, block_tables`);
  注意 context 里的 tensor 必须是**静态缓冲区的切片**,和现在的做法一致(model_runner.py:242)。
- `run_model` graph 分支:多刷一个 cu_seqlens_k 缓冲(由 prepare_batch 顺手产出 per-seq context len,
  CPU cumsum 后 copy);`graph_vars` 增加 cu_seqlens_k。
- LMHead 在图外(compute_logits 单独调,model_runner.py:212),不受影响;
  但注意 replay 后 `compute_logits(graph_vars["outputs"][:bs])` 走的是 hidden 直接取行,
  而统一 varlen 的 eager 路径里 LMHead 是按 cu_seqlens_q 取最后位置——decode 时两者等价
  (q_len=1,最后位置=本身)。确认 `ParallelLMHead.forward` 在"图路径"下拿到的 context 是 replay 用的
  context(cu_seqlens_q=arange),取 `arange[1:]-1 = 0..bs-1` 即逐行,正确。
- 删除:prepare_decode、attention 的 flash_attn_with_kvcache 分支、context.context_lens 字段、
  context.is_prefill(改名 use_varlen 或直接删掉,因为永远 True)。

### 验证

1. 图路径 vs eager 路径,同一批输入 outputs 允许浮点噪声、argmax 一致;greedy 基线重跑存 `greedy_p25.json`。
2. bs 恰好落桶/不落桶(padding 行生效)各测;bs=1 和 bs=512 边界。
3. `nvidia-smi`/torch profiler 确认 replay 没有隐性 host 同步(cu_seqlens_k 的 CPU cumsum 要用 pinned buffer + non_blocking copy)。
4. 与 Step A 相比 decode 吞吐不回退。

### 兜底

如果 FA 的 varlen kernel 在 capture 下有任何不兼容(实测才知道),Step A 的双路径就是长期方案,
Step B 整体放弃也不影响 Phase 3——投机解码的验证前向本来就是变长(q_len=k+1),走 eager。
