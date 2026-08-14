<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM（扩展版）

Fork 自 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，
在原版「~1200 行可读实现」的基础上补了三块 vLLM 的核心能力，外加一条**跨机专家并行**线。

所有改动都有配套的**计划文档**（动手前写的）与**实现报告**（做完后写的，含全部实测数据与踩过的坑）：

| | 计划 | 报告 |
|---|---|---|
| 采样器 / 统一调度 / 投机解码 | [Plan-1-2-3/](Plan-1-2-3/) | [05-implementation-report.md](Plan-1-2-3/05-implementation-report.md) |
| MoE + 跨机 EP + IBGDA | [Plan-4/](Plan-4/) | [08-implementation-report.md](Plan-4/08-implementation-report.md) |

---

## 加了什么

### 一、采样器重写（Plan-1）

原版 `sampler.py` 只有 Gumbel-max，`SamplingParams` 只有 `temperature/max_tokens/ignore_eos`。

* **`top_k` / `top_p` / greedy / `logprobs`** 四样补齐；`temperature=0` 即 greedy
* 修掉一个真 bug：bf16 的 152k 词表里**并列最大值很常见**，原来的阈值截断会让
  `argmax`、`top_k=1`、`top_p→0` 三者选出不同的 token。改成
  `sort(stable=True)` + 位置掩码后三者恒等（`tests/test_sampler.py` 有两条回归用例锁住）

```python
SamplingParams(temperature=0.8, top_k=50, top_p=0.9, logprobs=10)
```

### 二、统一调度器：prefill/decode 混批（Plan-2 / 2.5）

原版一步要么全 prefill、要么全 decode，长 prompt 进来会**阻塞所有 decode**。

* `schedule()` 改成**先给每条 decode 留 1 token 预算，再用余量塞 prefill**
* `prepare_prefill` → `prepare_batch`：变长统一路径，decode 只是 `q_len=1` 的特例
* `logits_indices`：lm_head 只算需要采样的行，中间 chunk 不算
* **CUDA graph 共存**：`is_pure_decode()` 决定走哪个 kernel、`use_cudagraph()` 决定是否
  replay，36 张分桶图在纯 decode 步继续生效

效果：在"3 条 4000-token prompt 撞上稳态 decode"的场景下，原调度器有 19 个连续的
prefill-only 步（decode 完全饿死），新调度器把它们变成混批。

### 三、投机解码（Plan-3）

n-gram prompt-lookup，不需要第二个模型。

* 逐位置 rejection sampling（Leviathan）；greedy 下退化为"接受当且仅当草稿 == argmax"
* KV 回滚：`block_manager.truncate()` 按**实际接受数**回收块
* prefix cache 不被污染：`hash_blocks` 的范围必须按 `1 + num_accepted` 算，不是 `1 + k`

```python
LLM(model, num_speculative_tokens=2, speculative_method="ngram")
```

### 四、MoE 模型层（Plan-4 / M3）

* `models/qwen3_moe.py`：`Qwen3MoeSparseMoeBlock` + `FusedExpertsEP`
* loader 支持 `experts.{N}.{gate,up,down}_proj` 并做 **EP 过滤**（不属于本 rank 的专家
  权重根本不进显存）
* `tools/make_tiny_qwen3_moe.py` 生成对拍用的 tiny 权重（seed 固定）

与 HF `Qwen3MoeForCausalLM` 单步对拍：**argmax 6/6 一致，偏差 2.0 ulp**。

### 五、多机运行时（Plan-4 / M1）

原版 `dist.init_process_group("nccl", "tcp://localhost:2333")` 写死了单机。

* `utils/parallel.py`：TP 组与 EP 组分离，模型侧一律问 `get_tp_size()`
* 控制面从 SharedMemory + Event 换成 **gloo `broadcast_object_list`**（同机跨机一条路径，
  实测 0.26 ms/次）
* `nanovllm/entry_worker.py` + `scripts/launch_both.sh` 一键双机

硬验收：dense Qwen3-0.6B 在 `ep_size=2` 下，rank0 输出与单机**逐 token 全等**。

### 六、nano-deepEP：跨机专家并行（Plan-4 / M2、M4、M5）

`nanodeepep/` 是 [DeepEP](https://github.com/deepseek-ai/DeepEP) legacy(V1) low-latency
路径的移植，**一套 API、两个可切换后端**：

```
              NanoEPBuffer.low_latency_dispatch / low_latency_combine
                      │
        ┌─────────────┴─────────────┐
   transport="nccl"            transport="nvshmem"
   torch.distributed            DeepEP 的 IBGDA 内核（SM89 移植）
```

* **`nccl` 后端**：~180 行纯 `torch.distributed`，兼作 IBGDA 后端的对拍 oracle
* **`nvshmem` 后端**：`csrc/legacy/` 是从 DeepEP 复制的内核，做了 SM89 手术
  —— dispatch **零手术**，combine 的 TMA 流水（SM90 专有）改写成 warp copy +
  每 warp 一个 token 的 fp32 归约，LogFMT 删除，launch 配置改成
  "cooperative 但不带 cluster"

两个后端的 `combined_x` **位级一致**——这不是巧合，是刻意让两边都按 k 升序做 fp32 归约。

```python
LLM("~/huggingface/tiny-qwen3-moe",
    ep_size=2, node_rank=0, master_addr="192.168.100.2",
    ep_transport="nvshmem",     # 或 "nccl"
    enforce_eager=True, max_num_batched_tokens=512)
```

---

## 核心性能数字

**每层 dispatch 延迟（双机 2×L40S + ConnectX-6 Dx 100GbE RoCE）**

| T | nccl | **ibgda** | 加速 |
|---|---|---|---|
| 1 | 400.8 µs | **11.8** | **34.0×** |
| 8（decode 形态） | 563.8 | **17.3** | **32.6×** |
| 128 | 571.7 | **83.8** | 6.8× |
| 512（prefill 块） | 731.8 | **308.5** | 2.4× |

NCCL 的开销**与数据量无关**（T ×60，耗时只涨 30%）——那是一次 D2H 同步 + 3 次集合调用的
固定延迟。IBGDA 把它整个消掉，耗时才开始随数据量走。

**端到端（tiny-qwen3-moe，8 请求稳态 decode）**

| | EP=1 单机 | EP=2 + nccl | **EP=2 + ibgda** |
|---|---|---|---|
| decode tok/s | 1034.5 | 888.4 | **1485.0** |
| TBT p50 ms | 7.60 | 8.98 | **5.37** |

跨机 EP 反超单机 44%。原因是 EP=1 的本地路径每层要 `torch.where` 找命中行，
输出尺寸数据依赖 → **每步 8 次 D2H 同步**；IBGDA 路径一次 CPU 同步都没有。

---

## 测试

判据方法论统一：**单步 logprob 对拍 + ulp 噪声分析**，而不是"跑 128 步看 token 是否相同"
（后者会把一次 bf16 级扰动放大成整段文本不同）。分歧点还要看它在
"top1−top2 差距分布"里的分位数——逻辑 bug 会在随机位置发作，不可能只挑最并列的几个百分点。

```bash
# 单机 42 项（采样器 / 统一调度 / 投机解码）
.venv/bin/python tests/run_all.py

# MoE 层 vs HF transformers
.venv/bin/python tests/test_m3_moe_local.py

# 双机（需要先配好 scripts/hosts.sh）
.venv/bin/python tests/test_m1_multinode.py                 # 多机运行时
./scripts/run2.sh nanodeepep/tests/test_2rank.py            # 通信层恒等式
.venv/bin/python tests/test_m4_ep2.py                       # EP 端到端（nccl）
EP_TRANSPORT=nvshmem .venv/bin/python tests/test_m4_ep2.py  # EP 端到端（ibgda）
.venv/bin/python tests/test_m6_backends.py                  # 两后端对拍
```

`tools/check_comments.py` 用来确认改代码时没有误删注释（逐条比对基准版本）。

---

## IBGDA 环境要求

IBGDA（GPU kernel 自己发起 RDMA）需要把网卡的 BAR 映射进 GPU 地址空间，
NVIDIA 驱动默认禁止。开启方式：

```bash
./scripts/m0_ibgda_check.sh          # 两机只读检查
sudo ./scripts/enable_ibgda.sh --apply   # 两机各跑一次；--revert 可完全还原
```

**不需要重启** —— `PeerMappingOverride` 虽是加载期参数，但可以卸载 nvidia 模块再带参数
装回去（前提是没有显示服务和 CUDA 进程占用）。脚本有前置检查、硬断言和可逆开关。

另外两条容易踩的：
* NVSHMEM **3.x** 选通道要用 `NVSHMEM_REMOTE_TRANSPORT=ibgda`；只设
  `NVSHMEM_IB_ENABLE_IBGDA=1`（2.x 的开关）会**静默退回 ibrc（CPU 代理）**，
  功能全对但性能是另一回事。用 `NVSHMEM_DEBUG=INFO` 确认日志里有
  `Successfully initialized the transport: IBGDA`
* NCCL 会按拓扑距离**默认关掉 GDR**（`distance 8 > 5`），需要
  `NCCL_NET_GDR_LEVEL=SYS`

以上都已写进 `scripts/env.sh`。

---

## 原版说明

<details>
<summary>展开</summary>

A lightweight vLLM implementation built from scratch.

### Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

### Model Download

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### Quick Start

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
outputs[0]["text"]
```

### Benchmark

RTX 4070 Laptop (8GB) / Qwen3-0.6B / 256 sequences：

| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

</details>
