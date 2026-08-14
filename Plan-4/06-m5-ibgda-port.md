# M5 · nvshmem/IBGDA 后端：忠实移植 DeepEP legacy internode_ll 到 SM89

目标：把 DeepEP legacy 的 low-latency 内核（GPU 直发 RDMA：GPU 上写 mlx5 WQE、按门铃，数据 NIC↔GPU 显存零 host 拷贝）复制进 `nanodeepep/csrc`，做 **SM89（L40S）最小手术**，在两机上通过 standalone 验收并与 NCCL 后端逐元素对拍。这是本计划"忠实原码"的主体。

**前置闸门**：M0 的 IBGDA 结论为方案 A（PeerMappingOverride）或 B（gdrcopy + `NVSHMEM_IBGDA_NIC_HANDLER=cpu`）之一已生效。否则本里程碑冻结，M6 按 NCCL 后端交付。

## 复制清单（源 → `nanodeepep/csrc/`，按 include 闭包）

| 源文件（DeepEP/） | 行数 | 处置 |
|---|---|---|
| `csrc/kernels/legacy/ibgda_device.cuh` | 496 | **原样复制**（纯 PTX 字节序/原子/内存序 + mlx5 WQE 构造，无 SM90 指令；`use_async_postsend` 分支即 CPU 辅助模式，方案 B 直接可用） |
| `csrc/kernels/legacy/internode_ll.cu` | 1290 | 复制 + 手术（见下节）：dispatch 原样，combine 去 TMA，mask/shrink 相关删除 |
| `csrc/kernels/legacy/utils.cuh` | ~430 | 复制；`elect_one_sync` 恒走 :332 的 `lane_id==0` fallback；TMA/mbarrier 段（:336-427）整段删除；其余 ld/st/warp_reduce/UNROLLED_WARP_COPY 保留（激进 PTX 有 `DISABLE_AGGRESSIVE_PTX_INSTRS` 的常规回退） |
| `csrc/kernels/legacy/launch.cuh` | 134 | 复制；`SETUP_LAUNCH_CONFIG`（:7-17）去掉 cluster attr（SM90 专有），**保留 cooperative=1**（两个内核的 `cg::this_grid().sync()` 依赖它：dispatch:359、combine:976）；`SWITCH_HIDDEN` 精简为 `{2048}` + 注释保留原表 |
| `csrc/kernels/legacy/buffer.cuh` `compiled.cuh` `api.cuh` | 小 | 原样；`api.cuh` 裁到只剩 internode_ll 的声明 |
| `deep_ep/include/deep_ep/common/compiled.cuh` | ~100 | 复制；`DISABLE_SM90_FEATURES` 的 FP8 stub 分支（:32-40）删除，恒 `#include <cuda_fp8.h>`（SM89 原生 FP8 cvt） |
| `deep_ep/include/deep_ep/common/exception.cuh` | 小 | 原样（EP_HOST_ASSERT/EP_DEVICE_ASSERT/EPException） |
| `csrc/legacy/config.hpp` | :102-188 | 只留 `LowLatencyBuffer/LowLatencyLayout/get_low_latency_rdma_size_hint` |
| `csrc/legacy/buffer.hpp` | 1794→~500 | **裁剪**：留构造（:84-169，`num_nvl_bytes` 恒 0 → IPC/barrier 段不进）、`sync`（:227-290，只走 NVSHMEM 分支）、`destroy`、`get_local_nvshmem_unique_id`、`clean_low_latency_buffer`、`low_latency_dispatch/combine`（:1456-1715）、size hint 转发；删 intranode/internode/layout/mask/shrink/fabric/get_next_low_latency_combine_buffer（zero_copy 不支持） |
| `csrc/kernels/backend/nvshmem.cu` + `backend/api.cuh` 中 nvshmem 部分 | 87 | 原样（unique_id/init/alloc/barrier/finalize；2 ranks < team_split_stride=8 → 不建子 team） |
| `csrc/utils/event.hpp` | 小 | 原样（EventHandle；同步语义下只用 stream_wait） |
| `csrc/python_api.cpp` | 39→~25 | 重写：只注册裁剪后的 Buffer + `get_low_latency_rdma_size_hint`（照抄 buffer.hpp:1751-1792 的子集） |
| `deep_ep/buffers/legacy.py` | 713→~150 | 拷为 `nvshmem_backend.py`：保 `__init__`（:66-136，删 `check_nvlink_connections`——L40S 名字不含 "PCIE" 本来就跳过，但依赖删干净）、`low_latency_dispatch/combine`（:553-670，砍掉 stats/logfmt/zero_copy 参数）、`clean_low_latency_buffer`；`get_dispatch_config` 等全删 |
| `third-party/fmt` | — | 上游 `utils/format.hpp` 依赖 fmt；裁剪版若仅剩 EP_HOST_ASSERT 的 sprintf 路径则去掉 fmt 依赖，否则拷 include（子模块已 checkout，二选一在实现时定，倾向去依赖） |

拷贝原则：**文件内容能不动就不动**（diff 最小化 = 忠实性可审计），所有手术处加 `// [nano-deepEP] 修改原因：...` 注释；不改 DeepEP 原目录任何文件。

## combine 内核手术方案（唯一的大改）

现状：combine（internode_ll.cu:714-1138）发送侧用 TMA 三段流水（:805-905 的 `tma_load_1d/tma_store_1d/mbarrier`），接收侧是 TMA producer/consumer 双角色 warp 组（:978-1137）。TMA=cp.async.bulk 是 SM90 指令，SM89 无。手术=退回 DeepEP 引入 TMA 前的结构（dispatch 内核里现成的 warp-copy 风格）：

**发送侧**（替换 :849-906 的流水，保留外层循环与 RDMA 调用不动）：

```cuda
// 原: TMA 载入 smem → (LogFMT) → TMA 存到 buf/p2p → tma_store_wait
// 改: 直接 int4 warp 拷贝(与 dispatch:271 同款)
if (not zero_copy or dst_p2p_ptr != 0) {   // zero_copy 恒 false,分支简化
    const auto src_int4 = x_int4;
    const auto dst_int4 = dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr)
                                           : reinterpret_cast<int4*>(dst_p2p_ptr);
    UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4, src_int4,
                       ld_nc_global, st_na_global);
}
__syncwarp();
if (dst_p2p_ptr == 0)
    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank,
                                local_expert_idx, lane_id, token_idx - offset);   // :911 原样
```

同时删除：smem 声明/mbarrier init 与 inval（:810-823, :934-939）、LogFMT 全部（`logfmt_encode/logfmt_check_amaxmin` :556-670、模板参数 `kUseLogFMT` 定死 false、host 侧 `use_logfmt` assert false）。消息布局 `num_bytes_per_slot = kHidden*2 + kNumMetaBytes`（:769）**保持不变**（meta 区浪费 hidden/128×4B/slot，换与 DeepEP 逐字节同布局，便于对照调试）。

**接收侧**（:978-1137 整段重写为"每 warp 一个 token"的朴素归约）：

```cuda
// 等待 flag 的段(:948-975)原样保留(ld_acquire_sys_global 轮询 + grid sync)
// 之后:
const int num_recv_warps = num_threads / 32;
for (int token_idx = sm_id * num_recv_warps + warp_id;
     token_idx < num_combined_tokens; token_idx += num_sms * num_recv_warps) {
    // 每 lane 负责 hidden/32 = 64 个 bf16 = 32 个 bf162,分 4 段×16 float 累加,
    // 控制寄存器压力(fp32 accum 峰值 16 个/段)
    for (int seg = 0; seg < kHidden / (32 * kNumElemsPerSeg); ++seg) {   // kNumElemsPerSeg=16
        float accum[kNumElemsPerSeg] = {0};
        #pragma unroll
        for (int i = 0; i < num_topk; ++i) {                    // k 升序 → 与 NCCL 后端同序
            const auto idx = __ldg(topk_idx + token_idx * num_topk + i);
            if (idx < 0) continue;
            const auto w = __ldg(topk_weights + token_idx * num_topk + i);
            // 注意偏移：BF16 模式数据在 slot 偏移 0（原代码 :1064 `kUseLogFMT ? kNumMetaBytes : 0`，
            // meta 区只有 LogFMT 用；发送侧 BF16 也是写偏移 0、只发 hidden*2 字节，:847/:891）
            const auto src = rdma_recv_x + (idx * M + token_idx) * num_bytes_per_slot
                             + seg/lane 偏移;
            // ld_nc_global 逐 bf162 读、fp32 乘加(照抄 decode_and_accumulate 的 else 分支 :704-711)
        }
        // cast bf16 写 combined_x 对应段(普通 st,不再 tma_store)
    }
}
```

寄存器/占用核算（写进实现前的自查）：16 float accum + 少量指针 ≈ 40 regs/thread，1024 线程/block 无压力；`num_sms` 的 host 计算（:1167-1174）保留但 cap 到 `min(..., num_device_sms)`——cooperative launch 要求全部 block 同时驻留（L40S 142 SM，1024 线程/块 → 每 SM 1 块 → grid ≤142；T=512 时原公式给 128 ✓，仍加 cap 防御）。

**dispatch 内核零手术**確认单（读码结论，实现时复核编译）：FP8 cast `__nv_cvt_float2_to_fp8x2`（:243）SM89 原生；`bar.sync` 命名屏障（:251/419）、`cg::this_grid().sync()`（:359，配 cooperative launch）、`atomicMax(ull)`、`st.release/ld.acquire.sys`（sm70+）全部可用；`EP_DEVICE_ASSERT(num_sms > 1)`（:281）在 E=4→4 SM 下满足；`num_warp_groups=1, num_warps_per_group=32` 满足 :382/:497 的断言。

## 构建（`nanodeepep/setup.py`，仿 DeepEP setup.py 裁剪）

```python
sources = ['csrc/python_api.cpp', 'csrc/internode_ll.cu', 'csrc/nvshmem_backend_glue.cu']
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'
nvcc_flags = ['-O3', '--extended-lambda', '-rdc=true',
              '-DDISABLE_AGGRESSIVE_PTX_INSTRS']          # Hopper 专有 UB-PTX 关闭(上游对非 9.0 也强制,setup.py:147-150)
nvcc_dlink = ['-dlink', f'-L{nvshmem_root}/lib', '-lnvshmem_device']
extra_link  = ['-lcuda', f'-l:{nvshmem_host_so}', '-l:libnvshmem_device.a',
               f'-Wl,-rpath,{nvshmem_root}/lib']          # 版本号 so 的处理照抄上游 :26-47
```

- `CUDA_HOME=$HOME/cuda-12.8`（M0 装好）；NVSHMEM 从 `.venv` 的 pip wheel 定位（find_pkgs.py 拷来）。
- 产物 `nanodeepep/_C.so`；两机各编一次或编完 rsync（同环境，直接 rsync 即可）。
- 冒烟：`python -c "import nanodeepep._C"`。

## 运行时环境变量（`scripts/env.sh` 增补，按 hostname 分支）

| 变量 | 值 | 说明 |
|---|---|---|
| `NVSHMEM_IB_ENABLE_IBGDA` | 1 | Python 侧代码设（照抄 legacy.py:109） |
| `NVSHMEM_IBGDA_NUM_RC_PER_PE` | 2 | = num_local_experts（QP 数=本地专家数，legacy.py:110 注释的要求） |
| `NVSHMEM_QP_DEPTH` | **2048** | 断言 `>= (M+1)*2`（legacy.py:609），M=512 时默认 1024 **必炸**，提前设好 |
| `NVSHMEM_HCA_LIST` | gpu-02: `mlx5_0:1` / gpu-01: `rocep66s0f0:1` | 两机 verbs 名不同 |
| `NVSHMEM_IB_GID_INDEX` | 3 | RoCE v2（M0 实测的 index） |
| `NVSHMEM_DISABLE_P2P` | 1 | 每机单卡，无 NVLink peer（对应 allow_nvlink_for_low_latency_mode=False） |
| `NVSHMEM_IBGDA_NIC_HANDLER` | `gpu`（方案 A）/ `cpu`（方案 B） | 按 M0 结论 |
| `NVSHMEM_DEBUG` | INFO（排障时） | 初始化失败第一现场 |

Python 侧 Buffer 初始化协议照抄 legacy.py:104-136：rank 0 `get_local_nvshmem_unique_id()` → 经 M1 的 gloo 组 `all_gather_object` 散布 → 各 rank `runtime.sync(device_ids, ipc_handles, root_unique_id)`（ipc 传空即可，num_nvl_bytes=0 分支不读，buffer.hpp:233 已确认）。

## 验收（`nanodeepep/tests/test_2rank.py --transport nvshmem`，双机）

M2 的测试参数化 transport 后直接复用，另加本后端专属项：

| # | 检查项 | 判据 |
|---|---|---|
| 1 | Buffer 初始化 | 双机 nvshmem init + `nvshmem_barrier_all` 通过；34MB RDMA buffer 分配成功 |
| 2 | M2 判据 1-5 | 全过（recv_count / 数据 / **加权恒等 diff<1e-5** / 确定性×20 / 空手与 -1） |
| 3 | **与 NCCL 后端对拍** | 同输入喂两个后端：packed_recv_x 有效区间逐元素相等（bf16 位级——两后端都是搬运不算数）；combined_x 位级相等（fp32 归约都按 k 升序） |
| 4 | FP8 dispatch（可选开关） | `use_fp8=True` 时恒等式 diff < 9e-4（test_low_latency.py:181 的阈值） |
| 5 | 压力 | 随机 seed 连跑 1000 轮 dispatch+combine 不挂、不超时（内核内置 200G cycles 超时会 trap，compiled.cuh:18） |
| 6 | 微基准 | T∈{8,64,128,512}: dispatch/combine 延迟（cuda event + kineto 双口径）与有效带宽；对照 NCCL 后端同表——**预期 decode 小 T 时延迟优势一个量级**（NCCL ~百µs vs IBGDA ~几十µs，H800/CX7 参考值 docs/legacy.md:34-39） |
| 7 | GDR 证据 | RDMA 计数器增量核对 + `nvidia-smi` 无 host staging 迹象（CPU 利用率旁证） |

## 排障预案（第一次跑不通是常态，先备好梯子）

| 症状 | 首查 |
|---|---|
| nvshmem init 失败/找不到设备 | `NVSHMEM_DEBUG=INFO`；HCA_LIST 名字是否用对（两机不同）；GID index；`nvshmem-info -a` |
| init 报 IBGDA 不可用 | M0 闸门是否真生效（regkey 需重启后 `/proc/driver/nvidia/params` 复核；方案 B 查 `/dev/gdrdrv` 权限） |
| dispatch 挂死在 recv_count 轮询 | ① RDMA atomic 是否可用（M0 的 ib_atomic_bw 结论）——不行则启用兜底：`nvshmemi_ibgda_amo_nonfetch_add`（dispatch:336 与 combine:925 两处调用）替换为 `nvshmemi_ibgda_rma_p` 单写（count 协议本就是"写 -cnt-1"与"加"等价于单生产者场景，两处都是每 (expert,src_rank) 恰好一次通知 → 语义等价）；② QP_DEPTH |
| 数据错但 count 对 | put_nbi 的 chunk 切分（ibgda_get_lkey_and_rkey 跨 cumem granularity）——buffer 对齐 `NVSHMEM_CUMEM_GRANULARITY=2^29`（legacy.py:122 照抄） |
| 编译 sm_89 报 PTX 指令不识别 | 漏删的 TMA/elect.sync 残留；grep 复查 `cp.async.bulk|elect.sync|mbarrier|cluster` 必须零命中 |
| cooperative launch 返回 too many blocks | combine 的 num_sms cap（本文手术节）没生效 |

## 风险

- 最大不确定性 = **NVSHMEM IBGDA 在"RoCE + L40S + 开放内核驱动"组合的成熟度**（NVSHMEM 官方主要在 IB+数据中心卡上验证）。对策：M0 先用 NVSHMEM 自带 perftest（`shmem_put_bw` device 侧）单独验 IBGDA 链路，再上 DeepEP 内核——把"NVSHMEM 环境问题"与"我们的移植问题"分开归因。
- combine 手术引入的性能回退（无 TMA 双缓冲）：接收侧变为纯 ld/st 归约——tiny 模型消息小，预计瓶颈在网络 RTT 不在 SM 拷贝；基准表里如实记录与上游 H800 数据的差距及原因。
