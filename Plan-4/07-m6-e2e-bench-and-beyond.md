# M6 · 切换 IBGDA 后端端到端 + 三配置基准 + 收尾报告（附 M7 延伸清单）

目标：`ep_transport="nvshmem"` 一键切换后，M4 的全部端到端验收在忠实 DeepEP 内核上复跑通过；产出三配置性能对比与《实现总结报告》（对齐 Plan-1-2-3/05 的报告格式）。

## 任务 1 · 切换与复验

改动仅一行配置（M4 的 `init_ep_buffer(transport=config.ep_transport)` 已留好口）。复跑 M4 验收表全部 7 项，判据不变：

- 单步 logits EP=2(nvshmem) vs EP=1：argmax 一致 + ≤4 ulp；
- **关键新增**：EP=2(nvshmem) vs EP=2(nccl) 同 prompt greedy 128 token——因 M5 验收 3 已证两后端 combine 位级一致，这里**必须逐 token 全等**；任何分歧 = 集成层 bug（buffer 复用/双缓冲翻转/QP 状态），不是数值噪声。
- chunked prefill / 抢占 / 投机组合用例同 M4。
- 双缓冲专项：LL 有奇偶双缓冲（buffer.hpp:1504-1505 `low_latency_buffer_idx ^= 1`），4 层×每步 2 次调用 → 翻转 8 次/步；构造连续 3 step 的 decode 验证无串扰（这是 clean_meta 链路在真实负载下的回归）。

## 任务 2 · 基准（`tests/bench_ep.py`，风格对齐 tests/bench.py：throwaway 预热两轮再测）

负载：tiny-qwen3-moe；场景 A = 8 请求 decode 稳态 448 token；场景 B = 场景 A 第 10 步注入 3×4000-token prefill（max_num_batched_tokens=512）。

| 指标 | EP=1 单机 | EP=2 + nccl | EP=2 + ibgda |
|---|---|---|---|
| decode tokens/s（稳态） | | | |
| 每 step 时延分解：attention / MoE-GEMM / dispatch / combine（torch profiler + cuda event 打点） | | | |
| 每层 dispatch 延迟 µs（T=8 decode 形态） | — | | |
| 每层 combine 延迟 µs | — | | |
| prefill 吞吐 tok/s | | | |
| TBT p50/p99（场景 B） | | | |
| RoCE 口位流量（ethtool 计数增量/步） | — | | |

预期叙事（写报告时验证或推翻）：tiny 模型上 EP=2 总吞吐**低于** EP=1（复制 attention + 每层 2 次跨机通信 vs 零通信），本计划的价值在机制与延迟剖面——ibgda 相对 nccl 在小 T decode 的每层通信延迟应有数倍以上优势；把这条曲线画出来就是本线的核心成果。

## 任务 3 · 《实现总结报告》 `Plan-4/08-implementation-report.md`

对齐 05-implementation-report.md 的结构：环境 / 改了哪些部分（M0-M6 逐里程碑表格）/ 踩过的坑（编号+现象+根因+修法）/ 每一步的效果（验收表全填实测值）/ 遗留与注意事项 / 测试套件说明与复现命令。**过时的用户注释按项目约定只列出不修改**。

## M7 · 延伸清单（不承诺，按价值排序，各给入口点）

1. **DP attention（消除复制计算）**：scheduler 把 seqs 按 rank 切半（driver 仍在 rank0，广播时带 per-rank 子集），各 rank 只 prefill/decode 自己的一半；MoE 处 dispatch 的 T 每 rank 不同（LL 天然支持 T_r ≤ M）；难点=两 rank 每步都必须进 MoE（空批也要 dispatch T=0）与完成时序同步、KV/block_manager 按 rank 分账。入口：scheduler.schedule() 的返回按 rank 分组 + model_runner 广播协议加 rank 维。
2. **通信重叠**：暴露 `return_recv_hook`（内核已支持 SEND/RECV 分相位，internode_ll.cu:189/353），dispatch send 后先算 shared 部分再 hook 收——tiny 模型无 shared expert，重叠对象可选双 micro-batch（docs/legacy.md:288 的图）。
3. **CUDA graph**：LL 内核 CUDA-graph 兼容（docs/legacy.md:263 注释），静态形状（全量 bmm 方案）已具备；把 EP decode 步整图捕获，对齐 Phase 2.5 的桶策略。入口：解除 config 里 EP+eager 的强制断言。
4. **FP8 dispatch 默认开**：M5 验收 4 已测通路，把 MoE 前向切 fp8 收包 + GEMM 前反量化（或直接 fp8 GEMM，L40S 支持），对拍阈值放宽到 9e-4。
5. **cached handle / decode 网格复用**：decode 期 topk 不变时跳过重复 layout（README decode 示例的 cached_handle 模式），降 CPU 开销。
6. **真模型可行性**：Qwen3-30B-A3B bf16 ≈ 60GB > 2×46GB 且专家 128 个——需权重量化（fp8/awq）+ M 上限重算；结论倾向不做，nano 的教学目标已由 tiny 模型达成。

## 完成定义（这条线什么时候算"做完"）

- [ ] M0 报告 + IBGDA 结论存档
- [ ] M1-M4：NCCL 后端端到端全绿（这是**保底交付**，不依赖管理员）
- [ ] M5-M6：ibgda 后端全绿 + 三配置基准表（**目标交付**，依赖 M0 闸门）
- [ ] 08-implementation-report.md 完成
- [ ] dense 单机 42 项回归始终全绿
