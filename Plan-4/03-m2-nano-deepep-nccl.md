# M2 · nano-deepEP 包：API 定义 + NCCL 参考后端

目标：定下 nano-deepEP 的 **API（= deep_ep legacy Buffer 的 low-latency 子集）**，并用纯 `torch.distributed` 写出第一个后端。这个后端有三重身份：M4 能立即集成的可用实现（GPUDirect via NCCL/RoCE）、M5 ibgda 后端的**逐元素对拍 oracle**、以及 LL 语义（packed 布局 / recv_count / handle）的可读文档。

本里程碑不依赖 M1：测试脚本用裸 `torchrun`/env:// 起双机进程组。

## 包结构

```
nano-vllm/nanodeepep/
├── __init__.py            # NanoEPBuffer 工厂（按 transport 分发）
├── nccl_backend.py        # 本里程碑
├── nvshmem_backend.py     # M5（占位）
├── csrc/                  # M5（DeepEP 拷贝件）
└── tests/
    ├── test_local.py      # 单进程逻辑单测（world=1 自发自收）
    └── test_2rank.py      # 双机恒等式验收（改编 DeepEP tests/legacy/test_low_latency.py）
```

## API（蓝本：`DeepEP/deep_ep/buffers/legacy.py:553-621 / 624-670`）

```python
class NanoEPBuffer:
    def __init__(self, group: dist.ProcessGroup,
                 num_max_dispatch_tokens_per_rank: int,   # M，本项目 = config.max_num_batched_tokens
                 hidden: int, num_experts: int,
                 transport: str = "nccl"): ...

    def low_latency_dispatch(self, x: Tensor,            # [T, H] bf16, T <= M
                             topk_idx: Tensor            # [T, K] int64, -1 = 不选
                             ) -> tuple[Tensor,          # packed_recv_x [L, R*M, H] bf16
                                        Tensor,          # recv_count [L] int32
                                        object]:         # handle（combine 用，语义见下）

    def low_latency_combine(self, x: Tensor,             # [L, R*M, H] bf16（专家算完的结果，原位布局）
                            topk_idx: Tensor,            # dispatch 时的同一份 [T, K]
                            topk_weights: Tensor,        # [T, K] float32
                            handle) -> Tensor:           # combined_x [T, H] bf16
```

其中 `L = num_experts // ep_size`（本项目 =2），`R = ep_size`（=2），`M = num_max_dispatch_tokens_per_rank`。

与 deep_ep 的取舍对照表（写进 `__init__.py` docstring）：

| deep_ep 参数/返回 | nano | 理由 |
|---|---|---|
| `use_fp8/round_scale/use_ue8m0` | 首版恒 bf16；nvshmem 后端保留 use_fp8 开关（M5/M7） | 内核原生支持，先减一维变量 |
| `async_finish/return_recv_hook` | 不暴露，恒同步 | nano 无重叠需求（M7 再说） |
| `cumulative_..._stats/mask/shrink` | 不要 | 容错/监控超出 nano 范围 |
| `handle` 五元组（src_info, layout_range, M, hidden, E） | 同构保留 | combine 必需 + 与 DeepEP 可互换 |
| `packed_recv_x` 布局 `[L, R*M, H]`，第 r 段 = 来自 rank r 的 token | **逐字节同语义** | M5 换后端零改动 |
| `layout_range[l][r] = pack2(count, begin)`（int64 高 32 位 begin、低 32 位 count，见 internode_ll.cu:411 `pack2<int,int64_t>(num_recv_tokens, recv_token_begin_idx)`） | 同 | 同上 |

关键布局约定（照抄 LL 内核语义，internode_ll.cu:260-262/365-374）：expert l 收到的来自 rank r 的第 s 个 token 存放在 `packed_recv_x[l, r*M + s_r_begin + s]`；NCCL 后端令 `begin = 0`（每个 (l,r) 段从段首连续放），这与 ibgda 后端的实际排布一致（每段独立计数从 0 开始，begin 由 layout_range 给出）。

## NCCL 后端算法（`nccl_backend.py`，目标 ~200 行）

dispatch（无 GPU 内核，全部张量原语；有一次不可避免的 D2H 同步取 split size——这正是 LL 内核用静态 M 上限规避掉的东西，注释里写明这一设计差异）：

```python
def low_latency_dispatch(self, x, topk_idx):
    T, K = topk_idx.shape
    flat = topk_idx.reshape(-1)                       # [T*K]
    valid = flat >= 0
    dst_rank  = flat[valid] // self.L                 # 每份 token 副本的目的 rank
    dst_local = flat[valid] %  self.L
    src_tok   = torch.arange(T, device=..).repeat_interleave(K)[valid]

    order = torch.argsort(dst_rank, stable=True)      # 稳定排序 → 确定性布局
    send_x    = x[src_tok[order]]                     # [S, H] 发送副本
    send_meta = torch.stack([src_tok[order], dst_local[order]], 1).int()   # [S, 2]

    send_splits = torch.bincount(dst_rank, minlength=R).tolist()           # D2H 同步点①
    recv_splits = all_to_all(send_splits)             # 先交换计数（int64 tensor all_to_all + .tolist()，同步点②）

    recv_x    = all_to_all_single(send_x,    out_splits=recv_splits, in_splits=send_splits, group=ep)
    recv_meta = all_to_all_single(send_meta, ...)     # 与数据同序到达

    # 落 packed 布局：段内到达序即 slot 序
    packed = x.new_empty(L, R*M, H); count = zeros(L, int32); src_info = new_full((L, R*M), -1, int32)
    off = 0
    for r in range(R):                                # R=2，python 循环可接受
        m = recv_meta[off:off+recv_splits[r]]         # 该源 rank 的份
        for l in range(L):                            # L=2
            rows = m[:, 1] == l
            n = int(rows.sum())                       # 同步点③（首版接受；优化留 TODO）
            packed[l, r*M : r*M+n] = recv_x[off:off+recv_splits[r]][rows]
            src_info[l, r*M : r*M+n] = m[rows, 0]     # 源 token 下标（combine 回程用）
            count[l] += n; layout_range[l, r] = pack2(n, 0)
        off += recv_splits[r]
    handle = (src_info, layout_range, M, H, self.num_experts, send_splits, recv_splits, order_meta...)
    return packed, count, handle
```

combine 是 dispatch 的逆置换 + 加权归约（**fp32 累加、按 k 升序**，与 internode_ll.cu 接收侧 `decode_and_accumulate` 的顺序一致，保证 M5 可位级对拍）：

```python
def low_latency_combine(self, x, topk_idx, topk_weights, handle):
    # 1) 按 handle 把 x[l, r*M+s] gather 回 "发送序" → 沿原路 all_to_all_single 逆向回传
    # 2) 回到源 rank 后得到每份副本的专家输出 y_copy [S_local, H] 与 (tok, k) 对应关系
    # 3) out = zeros(T, H, fp32); 对 k = 0..K-1 依序 index_add_(out, tok_k, y_k * w_k)
    #    （逐 k 循环而不是一把 index_add，固定加法顺序）
    # 4) return out.to(bf16)；topk_idx == -1 的 k 跳过（权重视为 0）
```

实现注意：
- `all_to_all_single` 走 EP 的 nccl 组 → 数据面 GPUDirect RDMA（M0 已验证）。
- 全流程对 `T=0`（本 rank 本步无 token）健壮：splits 全 0 也要参与集合调用（对端可能有数据）。
- 确定性：稳定排序 + 逐 k 归约顺序固定 → 同输入必位级同输出（验收 4 依赖这一点）。

## 验收（`tests/test_2rank.py`，双机 torchrun）

构造照抄 `DeepEP/tests/legacy/test_low_latency.py:56-77`：`x = 常数(rank-128)`、最后 128 列写 token 序号、随机 scores→topk、随机戳 10 个 `-1`；参数 `R=2, E=4, K=2, H=2048, T ∈ {1, 7, 128, 512}`（含边界）。

| # | 检查项 | 判据 |
|---|---|---|
| 1 | recv_count | `count[l] == (all_topk_idx == rank*L+l).sum()`（all_gather topk_idx 后全局数，photocopy test_low_latency.py:120-124） |
| 2 | 数据搬运正确 | 收到行的前 H-128 列全等于源 rank 常数；`src_info` 与最后 128 列的 token 序号一致（:129-137 同款） |
| 3 | **加权恒等式** | 假 GEMM=恒等：`combine(dispatch(x)) == x * Σ_k w_k·[idx_k≠-1]`，bf16 `calc_diff < 1e-5`（:178-181；calc_diff 从 deep_ep/utils/math.py 拷来） |
| 4 | 确定性 | 同 seed 连跑 20 遍，combined_x 与 packed_recv_x 的哈希不变（:297-307 同款） |
| 5 | T=0 / 全 -1 行 | 一个 rank 空手、某 token 全不选 → 不挂、对应输出行为 0 |
| 6 | 带宽记录 | T=512 时 dispatch/combine 各自耗时（cuda event），换算有效带宽写进报告（非门槛，供 M6 对比） |

单进程 `test_local.py`（world=1，R=1）：跑判据 1-5 的退化版，进 CI 习惯性回归（不需要第二台机器）。

## 边界与坑

- `topk_idx` dtype 与 deep_ep 对齐用 int64（`deep_ep.topk_idx_t` 默认 64 位，compiled.cuh `EP_NUM_TOPK_IDX_BITS=64`）。
- `T <= M` 由上层保证，但 buffer 侧加 assert（对齐 buffer.hpp:1481 的 host 检查）。
- 权重乘在 **combine 接收侧**（不是发送侧）——DeepEP 语义：combine 发送的是专家原始输出，权重在归约时乘（internode_ll.cu:1088/1105 `topk_weight` 进 `decode_and_accumulate`）。写单测锁住：权重非均匀时对拍。
- python 双层小循环（R=2×L=2）是刻意的可读性取舍；不要过早张量化，M5 后它只剩 oracle 用途。
