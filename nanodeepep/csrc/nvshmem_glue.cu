// NVSHMEM bootstrap glue —— 照抄 DeepEP 的 csrc/kernels/backend/nvshmem.cu，
// 只裁掉 team split（本项目 2 ranks < team_split_stride=8，本来就不建子 team）。
//
// 初始化协议与 deep_ep/buffers/legacy.py:104-136 一致：
//   rank0 调 get_unique_id() → 经 gloo 组 broadcast 给所有 rank → 各 rank 调 init()。
//
// 另外带一个 device-initiated put 的冒烟测试：这是 M5 的第一道关。
// 刻意先单独验 NVSHMEM 环境、再验内核移植——第一次跑不通时，这一层能把
// "NVSHMEM/IBGDA 环境问题" 和 "我们移植 internode_ll 的问题" 分开归因。

#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include "nano_ep.cuh"
// compiled.cuh 要排在前面：utils.cuh 用它定义的 LEGACY_* 宏
#include "legacy/compiled.cuh"
#include "legacy/ibgda_device.cuh"

namespace nanoep {

std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return result;
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks) {
    nvshmemx_uniqueid_t root_unique_id;
    nvshmemx_init_attr_t attr;
    NANO_HOST_ASSERT(root_unique_id_val.size() == sizeof(nvshmemx_uniqueid_t));
    std::memcpy(&root_unique_id, root_unique_id_val.data(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &root_unique_id, &attr);
    NANO_HOST_ASSERT(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr) == 0);
    // 等所有 GPU 就绪
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    CUDA_CHECK(cudaDeviceSynchronize());
    return nvshmem_my_pe();
}

void* alloc(size_t size, size_t alignment) { return nvshmem_align(alignment, size); }

void dealloc(void* ptr) { nvshmem_free(ptr); }

void barrier() {
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void finalize() {
    barrier();
    nvshmem_finalize();
}

int my_pe() { return nvshmem_my_pe(); }
int n_pes() { return nvshmem_n_pes(); }

// ---------------------------------------------------------------- 冒烟测试

// device 侧发起的 RDMA put：这条路径走的就是 IBGDA（GPU 自己写 WQE、按门铃）。
// 用 block 版本而不是单元素版，好顺带验证大消息的分片逻辑。
//
// 注意 src 必须也在**对称堆**里。NVSHMEM 的 API 语义上允许源是本地非对称内存，
// 但 IBGDA 下 GPU 直接发 RDMA，源地址必须是注册过的 RDMA 内存 —— 拿普通
// cudaMalloc 的缓冲当源，内核会静默挂死（第一版就是这么挂的：init/barrier 都过了，
// put 一进去就再没出来）。
__global__ void put_kernel(int* dst, const int* src, size_t nbytes, int peer) {
    nvshmemx_putmem_block(dst, src, nbytes, peer);
    // 等本 PE 发出的写全部落地，再让 host 侧的 barrier 去同步两端
    if (threadIdx.x == 0)
        nvshmem_quiet();
}

// 每个 PE 往对端的对称缓冲写 (100 + 自己的 PE 号)，barrier 后各自读自己的缓冲，
// 应该读到对端写进来的值。
bool put_test(int64_t nelem) {
    const int me = nvshmem_my_pe(), npes = nvshmem_n_pes();
    const int peer = (me + 1) % npes;
    const size_t nbytes = nelem * sizeof(int);

    int* sym = static_cast<int*>(nvshmem_align(256, nbytes));
    int* local = static_cast<int*>(nvshmem_align(256, nbytes));   // 源也要在对称堆里
    NANO_HOST_ASSERT(sym != nullptr and local != nullptr);

    // 源缓冲填 100+me；目的缓冲填 0xFF（没被写到的话一眼看得出来）
    std::vector<int> host(nelem, 100 + me);
    CUDA_CHECK(cudaMemcpy(local, host.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sym, 0xFF, nbytes));
    barrier();

    put_kernel<<<1, 256>>>(sym, local, nbytes, peer);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    barrier();

    std::vector<int> got(nelem);
    CUDA_CHECK(cudaMemcpy(got.data(), sym, nbytes, cudaMemcpyDeviceToHost));
    const int want = 100 + peer;
    bool ok = true;
    for (int64_t i = 0; i < nelem; ++i)
        if (got[i] != want) { ok = false; break; }

    nvshmem_free(local);
    nvshmem_free(sym);
    return ok;
}

// ---------------------------------------------- IBGDA 手写 WQE 路径的探针
//
// 前面那个 put_test 用的是 NVSHMEM **官方 API**（nvshmemx_putmem_block），它验证的是
// "IBGDA 传输层能不能用"。DeepEP 的内核走的是另一条路：自己往 mlx5 的发送队列写 WQE、
// 自己按门铃（nvshmemi_ibgda_put_nbi_warp，ibgda_device.cuh:337）。两条路的前提不同，
// 官方 API 通过**不代表**手写路径通过 —— M5 的 R=2 崩溃就卡在这个区别上。
//
// 这个探针做两件事：把设备侧看到的 IBGDA 状态打出来（状态没被填的话 rcs 会是空指针，
// 那是最常见的"链接进来了但没初始化"故障），然后走一次手写 WQE 的 put。

using deep_ep::legacy::get_lane_id;
using deep_ep::legacy::ibgda_get_state;
using deep_ep::legacy::nvshmemi_ibgda_put_nbi_warp;

__global__ void ibgda_probe_kernel(uint64_t dst, uint64_t src, size_t nbytes, int dst_pe, int qp_id) {
    const auto lane_id = get_lane_id();
    if (threadIdx.x == 0 and blockIdx.x == 0) {
        auto st = ibgda_get_state();
        // 注意：CUDA 的设备侧 printf **不支持 %zu**，size_t 要显式转成 unsigned long long 用 %llu。
        // （第一版写 %zu，输出里原样打了 "%zu" 三个字符，白跑一轮。）
        const auto heap = reinterpret_cast<uint64_t>(nvshmemi_device_state_d.heap_base);
        const auto g = st->log2_cumem_granularity;
        const uint64_t lidx = ((src - heap) >> g) * st->num_devices_initialized;
        const uint64_t roff = dst - heap;
        const uint64_t ridx = ((roff >> g) * nvshmemi_device_state_d.npes) * st->num_devices_initialized
                              + dst_pe * st->num_devices_initialized;
        printf("[ibgda-probe] pe=%d num_rc_per_pe=%u ndev=%d rcs=%p nic_gpumem=%d batch=%u\n"
               "              log2_cumem_granularity=%llu heap_base=0x%llx npes=%d\n"
               "              src=0x%llx dst=0x%llx  lkey_idx=%llu rkey_idx=%llu (MAX_CONST_RKEYS=%d)\n"
               "              peer_heap_base_remote[%d]=%p  lkeys[lidx].key=0x%x\n",
               nvshmem_my_pe(), st->num_rc_per_pe, st->num_devices_initialized,
               (void*)st->globalmem.rcs, (int)st->nic_buf_on_gpumem, st->num_requests_in_batch,
               (unsigned long long)g, (unsigned long long)heap, nvshmemi_device_state_d.npes,
               (unsigned long long)src, (unsigned long long)dst,
               (unsigned long long)lidx, (unsigned long long)ridx, (int)NVSHMEMI_IBGDA_MAX_CONST_RKEYS,
               dst_pe, (void*)nvshmemi_device_state_d.peer_heap_base_remote[dst_pe],
               (unsigned)st->constmem.lkeys[lidx].key);
    }
    __syncthreads();
    if (nbytes > 0 and threadIdx.x < 32)
        // 模板参数必须给 true！ibgda_submit_requests 默认是**批量**提交：
        // 只有 (message_idx+1) % 4 == 0 时才按门铃（ibgda_post_send）。单发一条消息时
        // message_idx=0 → 永远不按门铃 → 网卡不发 → 对端收不到 → 挂死。
        // dispatch 内核里靠 slot_idx 递增自然凑够 4 条，最后由 amo_nonfetch_add
        // （它用的是 ibgda_submit_requests<true>）兜底刷出去。
        nvshmemi_ibgda_put_nbi_warp<true>(dst, src, nbytes, dst_pe, qp_id, lane_id, 0);
}

// 只打印状态、不发数据（发之前先确认状态是不是有效的）
bool ibgda_probe(int64_t nelem, bool do_put) {
    const int me = nvshmem_my_pe(), npes = nvshmem_n_pes();
    const int peer = (me + 1) % npes;
    const size_t nbytes = nelem * sizeof(int);
    int* sym = static_cast<int*>(nvshmem_align(256, nbytes));
    int* local = static_cast<int*>(nvshmem_align(256, nbytes));
    NANO_HOST_ASSERT(sym and local);
    std::vector<int> host(nelem, 100 + me);
    CUDA_CHECK(cudaMemcpy(local, host.data(), nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sym, 0xFF, nbytes));
    barrier();

    ibgda_probe_kernel<<<1, 32>>>(reinterpret_cast<uint64_t>(sym),
                                  reinterpret_cast<uint64_t>(local),
                                  do_put ? nbytes : 0, peer, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    barrier();

    bool ok = true;
    if (do_put) {
        std::vector<int> got(nelem);
        CUDA_CHECK(cudaMemcpy(got.data(), sym, nbytes, cudaMemcpyDeviceToHost));
        const int want = 100 + peer;
        for (int64_t i = 0; i < nelem; ++i)
            if (got[i] != want) { ok = false; break; }
    }
    nvshmem_free(local);
    nvshmem_free(sym);
    return ok;
}

}  // namespace nanoep
