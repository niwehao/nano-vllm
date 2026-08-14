# M1 · 多机通信层（打破 localhost 硬编码）

目标：nano-vllm 能以 **2 节点 × 1 GPU** 的形态起进程组：rank 0（gpu-02）继续当 driver（scheduler/采样都在它上面），rank 1（gpu-01）当 worker。本里程碑**不引入 MoE**——用 dense Qwen3-0.6B 验证"跨机复制计算"与单机逐 token 一致，把通信地基做实。

## 现状

- `model_runner.py:26`：`dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)`，`world_size = config.tensor_parallel_size`。
- `model_runner.py:27`：`torch.cuda.set_device(rank)`——多机后每机只有 cuda:0。
- `model_runner.py:41-89`：TP 控制面 = 同机 SharedMemory("nanovllm") + mp.Event；`llm_engine.py:24-31` 用 `mp.Process` 在**本机** spawn worker。跨机两者都不可用。
- 所有并行层拿 `dist.get_world_size()` 当 TP size（qwen3.py:29、linear.py:23-24、embed_head.py:17-18）；`model_runner.py:110` 的 KV head 数也按 world 除。EP 模式下 world=2 但 TP=1，**不改会把权重错切一半**。
- `sequence.py` 的 `__getstate__` 已经只传本轮调度切片（Plan-2 的 token_offset 设计）——跨机 pickle 直接复用，通信量 O(本轮 token 数)。

## 设计决策

1. **并行语义定死**：EP 模式下 `world_size == ep_size == 节点数`，**TP=1**（`assert tensor_parallel_size == 1 or ep_size == 1`，不做 TP×EP 交叉——vLLM 里 EP 组 = DP×TP 的融合，nano 只取每机一卡的最简形态，注释里写明这个对应关系）。
2. **rank 1 与 rank 0 跑同一批**（复制 attention）：worker 侧执行流与今天的 TP worker 完全一致（收到 seqs → prepare → forward → 丢弃 logits），唯一区别在 M4 的 MoE 层内部走 EP。代价（每专家收到两份相同 token）在总览与 M7 讨论。
3. **控制面统一换 gloo broadcast**：`world_size > 1` 时用 CPU gloo 组的 `broadcast_object_list` 替代 SharedMemory——同机、跨机一条代码路径，SHM/Event 相关代码全部退役（read_shm/write_shm/loop 的 event 部分，model_runner.py:61-83）。每步 ~0.1-0.5ms 的控制延迟对 nano 可接受，实测记录。

## 改动清单

### 1. `nanovllm/config.py`

```python
# 多机 / EP
ep_size: int = 1                 # =节点数; 1 表示单机原行为
node_rank: int = 0               # 本进程的全局 rank（每机一进程一卡）
master_addr: str = "localhost"   # EP 模式填 192.168.100.2
master_port: int = 29500
ep_transport: str = "nccl"       # "nccl" | "nvshmem"（M5 后可选）

def __post_init__(self):
    ...
    assert self.tensor_parallel_size == 1 or self.ep_size == 1
    if self.ep_size > 1:
        assert self.enforce_eager, "EP 首版不进 CUDA graph（M7 再解）"
```

### 2. 新增 `nanovllm/utils/parallel.py`（parallel_state）

```python
import torch.distributed as dist

_WORLD_SIZE = 1; _RANK = 0
_TP_GROUP = None; _EP_GROUP = None; _CPU_GROUP = None

def init_distributed(config, rank: int):
    global ...
    world = max(config.tensor_parallel_size, config.ep_size)
    dist.init_process_group("nccl", f"tcp://{config.master_addr}:{config.master_port}",
                            world_size=world, rank=rank)
    _CPU_GROUP = dist.new_group(backend="gloo")     # 控制面（跨机 pickle 广播）
    if config.ep_size > 1:
        _TP_GROUP = dist.new_group([rank])           # 每机自成 TP=1 组
        _EP_GROUP = dist.group.WORLD
    else:
        _TP_GROUP = dist.group.WORLD                 # 单机 TP 原语义
        _EP_GROUP = None

def get_tp_rank()/get_tp_size()/get_tp_group()
def get_ep_rank()/get_ep_size()/get_ep_group()
def get_cpu_group()
```

注意 `dist.new_group` 必须**所有 rank 以相同顺序调用**（gloo 组、每个 rank 的单员 TP 组都要全员参与创建循环：`for r in range(world): g = dist.new_group([r]); r==rank 时留存`）。

### 3. 模型侧改用 TP 组（机械替换，行为在单机下不变）

- `layers/linear.py:23-24`、`layers/embed_head.py:17-18`、`models/qwen3.py:29`：`dist.get_world_size()/get_rank()` → `parallel.get_tp_size()/get_tp_rank()`。
- `linear.py:155` `dist.all_reduce(y)` → `if get_tp_size() > 1: dist.all_reduce(y, group=get_tp_group())`（EP 模式 tp=1 自动跳过）。
- `embed_head.py:69` `dist.gather(logits, all_logits, 0)` → 加 `group=get_tp_group()`，且 `dst` 要用**组内 rank0 的全局 rank**（单机时即 0；EP 模式 tp=1 不进该分支）。
- `model_runner.py:110`：`num_kv_heads // self.world_size` → `// get_tp_size()`。

### 4. `nanovllm/engine/model_runner.py` —— init 与控制面重写

```python
def __init__(self, config, rank, event=None):        # event 参数退役
    ...
    parallel.init_distributed(config, rank)          # 替换 :26
    torch.cuda.set_device(config.ep_size > 1 and 0 or rank)   # 每机单卡恒 0；单机 TP 保持 rank
    ...模型构建/加载/warmup/kv cache 不变...
    if self.world_size > 1:
        dist.barrier()
        if rank != 0:
            self.loop()

def loop(self):
    while True:
        method_name, args = self.recv_cmd()
        self.call(method_name, *args)
        if method_name == "exit": break

# 控制面：gloo broadcast 取代 shm（同机/跨机同路径）
def send_cmd(self, method_name, *args):              # rank 0
    dist.broadcast_object_list([(method_name, args)], src=0, group=parallel.get_cpu_group())
def recv_cmd(self):                                  # rank > 0
    buf = [None]
    dist.broadcast_object_list(buf, src=0, group=parallel.get_cpu_group())
    return buf[0][0], buf[0][1]

def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.send_cmd(method_name, *args)
    return getattr(self, method_name)(*args)
```

删除：`read_shm/write_shm`、SharedMemory 创建/清理（:41-59 相应行）、Event 链路。`exit()` 保留 barrier + destroy_process_group。

### 5. `nanovllm/engine/llm_engine.py` —— 进程模型

- `ep_size > 1` 时**不再本机 spawn**（:24-31 的 mp.Process 循环加条件跳过）；每台机器各自运行入口脚本，靠 `node_rank` 区分。
- 新增入口 `nanovllm/entry_worker.py`：非 0 rank 的进程只构造 `ModelRunner(config, rank=node_rank)`（构造函数内部进 loop，永不返回）；rank 0 照常构造 `LLM(...)` 并跑上层脚本。
- tokenizer/eos：仅 rank 0 需要（worker 不碰 tokenizer）。

### 6. 启动脚本

```bash
# scripts/run_ep_node1.sh（gpu-01, 先起）
source scripts/env.sh
.venv/bin/python -m nanovllm.entry_worker --model ~/huggingface/Qwen3-0.6B \
    --ep-size 2 --node-rank 1 --master-addr 192.168.100.2 --enforce-eager

# scripts/run_ep_node0.sh（gpu-02, 后起, 跑实际负载脚本）
.venv/bin/python <workload>.py --ep-size 2 --node-rank 0 --master-addr 192.168.100.2 ...

# scripts/launch_both.sh：sync.sh → ssh gpu-01 'nohup run_ep_node1.sh > /tmp/ep_w.log 2>&1 &' → 本机跑 node0 → 完毕后 ssh pkill；trap 里兜底清理远端进程
```

`scripts/env.sh` 统一设置：`NCCL_SOCKET_IFNAME=ens5f0np0`、`NCCL_IB_GID_INDEX=3`、`GLOO_SOCKET_IFNAME=ens5f0np0`、按 hostname 设 `NCCL_IB_HCA`、`TORCH_NCCL_BLOCKING_WAIT=1` 与 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120`（挂死变报错）。

## 验收

1. **组建立冒烟**：双机 init 成功，nccl all_reduce 与 gloo broadcast 各跑 100 次无卡顿；每步控制面广播耗时打点（预期 <1ms，写进报告）。
2. **dense 等价（硬验收）**：Qwen3-0.6B，Plan-1-2-3 的基线 prompt 集，greedy 128 token：`ep_size=2` 下 rank 0 输出与**单机 world=1 基线逐 token 全等**。理由：rank 0 的计算图与单机完全相同（tp=1 无通信参与前向），必须 bit 一致；任何分歧都是搬运 bug。
3. **worker 侧一致性探针**：临时在 worker 的 `run()` 里对 logits 求 checksum 经 gloo 回传比对 rank 0（debug 开关 `NANOVLLM_EP_CHECK=1`），连续 64 步 checksum 一致——证明"复制计算"两边真的算了同样的东西（两机同 GPU 型号同库版本，应位级一致；若有 ulp 级差异记录并分析，不阻塞，M4 有自己的判据）。
4. **回归**：单机 world=1 跑 `tests/run_all.py` 42 项全过（控制面改动不影响单进程路径）。
5. **故障行为**：kill rank 1 → rank 0 在超时窗口内报错退出（不无限挂）；launch_both.sh 的清理分支有效。

## 边界与坑

- gloo 也要指定 `GLOO_SOCKET_IFNAME`，否则可能选 bond0/docker0 导致 broadcast 慢或失败。
- `broadcast_object_list` 底层 pickle：seqs 列表要复用现有 `__getstate__` 切片协议（sequence.py:82-93），**不要**让完整 token_ids 进 pickle——加一条断言测试：广播 payload 字节数 ≈ O(本轮调度 token 数)。
- 两机模型路径必须一致（sync.sh 已覆盖）；`config.eos` 在 worker 侧没有 tokenizer，依赖 driver 广播的调度结果，不需要本地 eos——确认 worker 路径不读它。
- warmup（model_runner.py:91-100）在两 rank 各自执行，天然同步（同一 batch 形状）；allocate_kv_cache 的显存探测两机独立，块数取 min？——不需要：两机显存同规格，且 `config.num_kvcache_blocks` 由 rank 0 决定后通过首次广播的 config 固化（实现时在 rank0 算好显式写入 config 再启动 worker；测试配置直接显式指定块数，Plan-2 已支持）。
- NCCL 初始化本身在 EP 模式的前向里没有集合通信（tp=1），但保留 nccl 组给 M2/M4 的 EP all_to_all 用——init 时就建好，避免首次 MoE 前向时 lazy 建组的抖动。
