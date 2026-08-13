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

}  // namespace nanoep
