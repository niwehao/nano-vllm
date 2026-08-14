# M4 · EP 集成：双机 4 专家端到端（NCCL 后端先行）

目标：M1（多机运行时）+ M2（nano-deepEP/nccl）+ M3（MoE 层）汇合——tiny-qwen3-moe 在 **gpu-02(rank0, expert 0/1) + gpu-01(rank1, expert 2/3)** 上端到端生成，输出与单机 EP=1 逐 token 一致。此时数据面已经走 RoCE GPUDirect（NCCL），DeepEP 忠实内核在 M5 替入。

## 执行模型回顾（M1 已定）

两 rank 复制运行同一批（attention 重复计算），MoE 层内 EP：rank 各自把 **相同的 [T,H] token** 按路由 dispatch，专家算完 combine 回各自 rank，两边得到相同的 combined_x，前向继续。正确性与"专家收到 2 份同 token"无冲突（每份都 combine 回自己的源 rank）；冗余代价在 M6 报告量化、M7 讨论消除。

## 改动清单

### 1. `FusedExpertsEP.forward_ep`（qwen3_moe.py，M3 留的口）

```python
def forward_ep(self, x, topk_idx, topk_w):
    buf = get_ep_buffer()                          # 见下,进程级单例
    recv_x, recv_count, handle = buf.low_latency_dispatch(x, topk_idx)
    # 本地专家 GEMM —— 对 [L, R*M, H] 全量算,不读 recv_count(避免 D2H 同步):
    #   垃圾行(未被本步写过的 slot)算出的结果不会进 combine —— combine 只按
    #   handle.layout_range 标注的有效区间取数,行与行独立,NaN 也不传染。
    h = torch.bmm(recv_x, self.w13.transpose(1, 2))          # [L, R*M, 2I]
    h = SiluAndMul()(h)
    y = torch.bmm(h, self.w2.transpose(1, 2))                # [L, R*M, H]
    return buf.low_latency_combine(y.to(torch.bfloat16), topk_idx, topk_w, handle)
```

全量 bmm 的账：L=2 × R·M=1024 行 × (2048→1024→2048)，单层 ~4.3 GFLOP·2，L40S 上微秒级——换来**整条 EP 路径零 CPU 同步**（NCCL 后端内部仍有 split 同步，M5 的 ibgda 后端则真正零同步）。若想省垃圾行，备选方案 `recv_count.max()` 截断（引入一次 D2H），写成注释保留。

### 2. buffer 生命周期（`nanodeepep/__init__.py` + model_runner）

```python
# model_runner.__init__，在构建模型之前:
if config.ep_size > 1:
    nanodeepep.init_ep_buffer(group=parallel.get_ep_group(),
                              num_max_dispatch_tokens_per_rank=config.max_num_batched_tokens,
                              hidden=hf_config.hidden_size,
                              num_experts=hf_config.num_experts,
                              transport=config.ep_transport)     # "nccl"（M6 切 "nvshmem"）
```

- 单例挂在模块级（`get_ep_buffer()`），MoE 层 forward 里取用——与 nano 的 context 全局风格一致。
- **每层共用同一个 buffer**：LL 语义下 dispatch/combine 成对调用即可复用（DeepEP 同款用法，见 docs/legacy.md 的 decode 示例）。
- warmup（model_runner.py:91-100）自动覆盖 EP 路径：warmup 的 [max_num_batched_tokens] 假 batch 会把最大形状的 dispatch/combine 各跑一遍——顺带把 NCCL communicator 的首次建链开销付掉。

### 3. 形状护栏

- `prepare_batch` 产出的 T = Σ num_scheduled_tokens ≤ `max_num_batched_tokens`（scheduler 预算保证，Plan-2）→ 恰好等于 buffer 的 M 上限。在 `forward_ep` 加 `assert x.size(0) <= buf.M`。
- 投机解码的验证前向 q_len=k+1 也计入预算（Plan-3 的 decode 段按 k+1 记账）→ 无额外风险。
- `config.max_num_batched_tokens` 测试配置定为 512（与 Plan-2 基准一致），buffer ~34MB。

### 4. 启动与配置样例

```python
# examples/ep_generate.py（rank0 侧脚本；rank1 用 entry_worker）
llm = LLM("~/huggingface/tiny-qwen3-moe",
          ep_size=2, node_rank=args.node_rank,
          master_addr="192.168.100.2", enforce_eager=True,
          max_num_batched_tokens=512, max_model_len=4096)
```

`scripts/launch_both.sh` 一键：sync → 远端起 worker → 本机跑 examples/ep_generate.py → 收尾清理。

## 验收（`tests/test_m4_ep2.py`，判据沿用总览全局策略）

| # | 检查项 | 判据 |
|---|---|---|
| 1 | 单步 logits：EP=2 vs EP=1 | 同权重同 prompt 8 条，argmax 全一致 + top-10 logprob ≤ 4 ulp。理论预期：combine 的 fp32 归约顺序与 forward_local 的 index_add 顺序不同 → 允许 ulp 级漂移，走分位数分析流程 |
| 2 | greedy 128 token：EP=2 vs `greedy_moe_ep1.json` | 逐 token 比对；分歧点必须落在近并列位置（check_equal_or_noise 判据） |
| 3 | chunked prefill + EP | 3 条 4000-token prompt（max_num_batched_tokens=512 → 每条切 8 chunk）+ 8 条 decode 稳态混批，输出与 EP=1 一致 |
| 4 | 抢占 + EP | Plan-2 坑 8 的构造法（255-token prompt × 6、KV 块数卡 7）在 EP=2 下触发抢占，输出与不抢占一致 |
| 5 | 投机 + EP | ngram 投机 k=2 开启，greedy 下开/关投机逐 token 一致（Plan-3 硬验收在 EP 下复验） |
| 6 | GPUDirect 证据 | 压测期间两机采样 `/sys/class/infiniband/<dev>/ports/1/counters/port_xmit_data` 增量 ≈ 通信量估算值（token 数 × 副本数 × 4KB/token 量级），且 bond0 流量无增长 |
| 7 | 回归 | dense 单机 42 项 + M3 的 moe 本地测试全部不回退 |

通信量估算参考（验收 6 用）：每层每 step 每 rank dispatch 发送 ≈ T×K×H×2B（T=512,K=2,H=2048 → 4MB），combine 同量级；×4 层×2 rank。与计数器对得上数量级即过。

## 边界与坑

- **两 rank 的 topk_idx 必须一致**：复制计算下两边独立算 router——bf16 下 topk 出现并列时理论上可能选出不同专家（两机同硬件同库应位级一致，但这是个隐患点）。防御：debug 开关 `NANOVLLM_EP_CHECK=1` 时对 topk_idx 做跨 rank checksum 比对（gloo，每层）；若实测出现分歧，改为"rank0 广播 topk_idx"的一致化路径（一行 broadcast，代价小，作为兜底开关实现好放着）。
- dispatch/combine 在 decoder 层循环里逐层调用（4 层×每步 2 次集合通信×NCCL 后端），NCCL 小消息延迟 ~50-100µs/次——tiny 模型 decode 步会明显变慢，这是 NCCL 后端的固有代价，记录进 M6 基准（ibgda 后端的意义所在）。
- `torch.bmm` 对 w13/w2 的 transpose 会物化拷贝？——`transpose(1,2)` 是 view，bmm 支持非连续 → 不拷贝；实现时 profile 确认。
- worker（rank1）在 M1 里丢弃 logits，但 MoE 路径它必须**全程参与**每层 dispatch/combine——两边层数/调用次数天然一致（同一模型同一 batch），不存在集合调用错配；但 rank0 的采样在 lm_head 之后，worker 没有采样步——确认 worker 的 `run()` 在 forward 结束后直接返回（现状如此，model_runner.py:385-387）。
- 若 M1 的 dense 验收发现两机存在 ulp 漂移（库版本细微差异），先修环境（rsync .venv）再进本里程碑——否则判据 1/2 的归因会被污染。
