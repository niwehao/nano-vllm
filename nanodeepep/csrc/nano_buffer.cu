// nano-deepEP 的 Buffer 宿主侧：对应 DeepEP 的 csrc/legacy/buffer.hpp（1794 行）裁到只剩
// low-latency 用得上的部分，加上 config.hpp:102-188 的 LowLatencyLayout。
//
// 裁掉的：intranode / normal internode / layout / mask / shrink / fabric /
// zero_copy(get_next_low_latency_combine_buffer) / IPC 与 NVL barrier
// （num_nvl_bytes 恒 0，那些分支根本进不去）/ async 与 recv_hook / FP8 / LogFMT /
// 各类 stats。
//
// 保留的语义与上游逐字节一致：双缓冲奇偶翻转、next_clean 的自清理协议、
// packed 布局与 layout_range 的 pack2(count, begin) 编码。

#include <algorithm>
#include <vector>

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <nvshmem.h>

#include "legacy/compiled.cuh"
#include "nano_ep.cuh"

namespace deep_ep::legacy::internode_ll {
// 内核的宿主侧启动器（定义在 legacy/internode_ll.cu）
void dispatch(void* packed_recv_x, void* packed_recv_x_scales, int* packed_recv_src_info,
              int64_t* packed_recv_layout_range, int* packed_recv_count, int* mask_buffer_ptr,
              int* cumulative_local_expert_recv_stats, int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x, const void* x,
              const topk_idx_t* topk_idx, int* next_clean, int num_next_clean_int, int num_tokens,
              int hidden, int num_max_dispatch_tokens_per_rank, int num_topk, int num_experts,
              int rank, int num_ranks, bool use_fp8, bool round_scale, bool use_ue8m0,
              void* workspace, int num_device_sms, cudaStream_t stream, int phases);

void combine(void* combined_x, void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const topk_idx_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range, int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats, int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks, bool use_logfmt,
             void* workspace, int num_device_sms, cudaStream_t stream, int phases, bool zero_copy);
}  // namespace deep_ep::legacy::internode_ll

namespace nanoep {

using deep_ep::topk_idx_t;

// ---- config.hpp:102-188 的 LowLatencyLayout，原样搬（只删了注释里与 nano 无关的 TODO）----

struct LowLatencyBuffer {
    int num_clean_int = 0;
    void* dispatch_rdma_send_buffer = nullptr;
    void* dispatch_rdma_recv_data_buffer = nullptr;
    int* dispatch_rdma_recv_count_buffer = nullptr;
    void* combine_rdma_send_buffer = nullptr;
    void* combine_rdma_recv_data_buffer = nullptr;
    int* combine_rdma_recv_flag_buffer = nullptr;

    std::pair<int*, int> clean_meta() const {
        NANO_HOST_ASSERT(dispatch_rdma_recv_count_buffer == combine_rdma_recv_flag_buffer);
        return {dispatch_rdma_recv_count_buffer, num_clean_int};
    }
};

struct LowLatencyLayout {
    size_t total_bytes = 0;
    LowLatencyBuffer buffers[2];

    template <typename out_t = void*, typename cnt_t = uint8_t*, typename in_t = void*>
    static out_t advance(const in_t& ptr, size_t count) {
        return reinterpret_cast<out_t>(reinterpret_cast<cnt_t>(ptr) + count);
    }

    LowLatencyLayout(void* rdma_buffer, int m, int hidden, int num_ranks, int num_experts) {
        const int num_scales = hidden / 128;
        // 双缓冲：奇偶各一套 send / recv / signaling
        size_t bytes_per_dispatch_msg =
            sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float));
        size_t bytes_per_combine_msg = num_scales * sizeof(nv_bfloat162) + hidden * sizeof(nv_bfloat16);

        size_t send_bytes = std::max(static_cast<size_t>(m) * bytes_per_dispatch_msg,
                                     static_cast<size_t>(num_experts) * m * bytes_per_combine_msg);
        total_bytes += send_bytes * 2;
        size_t recv_bytes = std::max(static_cast<size_t>(num_experts) * m * bytes_per_dispatch_msg,
                                     static_cast<size_t>(num_experts) * m * bytes_per_combine_msg);
        total_bytes += recv_bytes * 2;
        size_t sig_bytes = num_experts * sizeof(int);
        size_t sig_aligned = ((sig_bytes + 127) / 128) * 128;
        total_bytes += sig_aligned * 2;

        for (int i = 0; i < 2; ++i) {
            buffers[i] = {static_cast<int>(sig_bytes / sizeof(int)),
                          advance(rdma_buffer, sig_aligned * 2 + send_bytes * i),
                          advance(rdma_buffer, sig_aligned * 2 + send_bytes * 2 + recv_bytes * i),
                          advance<int*>(rdma_buffer, sig_aligned * i),
                          advance(rdma_buffer, sig_aligned * 2 + send_bytes * i),
                          advance(rdma_buffer, sig_aligned * 2 + send_bytes * 2 + recv_bytes * i),
                          advance<int*>(rdma_buffer, sig_aligned * i)};
        }
    }
};

size_t rdma_size_hint(int m, int hidden, int num_ranks, int num_experts) {
    auto n = LowLatencyLayout(nullptr, m, hidden, num_ranks, num_experts).total_bytes;
    return ((n + LEGACY_NUM_BUFFER_ALIGNMENT_BYTES) / LEGACY_NUM_BUFFER_ALIGNMENT_BYTES) *
           LEGACY_NUM_BUFFER_ALIGNMENT_BYTES;
}

// ---------------------------------------------------------------- Buffer

struct NanoBuffer {
    int rank, num_ranks, num_experts, hidden, m;
    int num_device_sms;
    size_t num_rdma_bytes;
    void* rdma_buffer = nullptr;
    void* workspace = nullptr;
    int low_latency_buffer_idx = 0;

    NanoBuffer(int rank_, int num_ranks_, int num_experts_, int hidden_, int m_)
        : rank(rank_), num_ranks(num_ranks_), num_experts(num_experts_), hidden(hidden_), m(m_) {
        NANO_HOST_ASSERT(num_experts % num_ranks == 0);
        NANO_HOST_ASSERT(hidden % 128 == 0);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        num_device_sms = prop.multiProcessorCount;

        num_rdma_bytes = rdma_size_hint(m, hidden, num_ranks, num_experts);
        rdma_buffer = nvshmem_align(LEGACY_NUM_BUFFER_ALIGNMENT_BYTES, num_rdma_bytes);
        NANO_HOST_ASSERT(rdma_buffer != nullptr);
        // 首次使用前必须清零：LL 内核靠 recv_count/flag 从 0 变非 0 来判断到达，
        // 之后每一步由内核自己清理**下一个**缓冲（next_clean 协议）。
        CUDA_CHECK(cudaMemset(rdma_buffer, 0, num_rdma_bytes));

        CUDA_CHECK(cudaMalloc(&workspace, LEGACY_NUM_WORKSPACE_BYTES));
        CUDA_CHECK(cudaMemset(workspace, 0, LEGACY_NUM_WORKSPACE_BYTES));

        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~NanoBuffer() {
        if (workspace) cudaFree(workspace);
        if (rdma_buffer) nvshmem_free(rdma_buffer);
    }

    // 返回 (packed_recv_x, recv_count, src_info, layout_range)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx) {
        NANO_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
        NANO_HOST_ASSERT(x.size(1) == hidden);
        NANO_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
        NANO_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
        NANO_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= m);

        const int num_tokens = static_cast<int>(x.size(0));
        const int num_topk = static_cast<int>(topk_idx.size(1));
        const int num_local_experts = num_experts / num_ranks;

        LowLatencyLayout layout(rdma_buffer, m, hidden, num_ranks, num_experts);
        NANO_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
        auto buffer = layout.buffers[low_latency_buffer_idx];
        auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];   // 奇偶翻转，与上游一致

        auto opts_i32 = torch::dtype(torch::kInt32).device(torch::kCUDA);
        auto packed_recv_x = torch::empty({num_local_experts, num_ranks * m, hidden}, x.options());
        auto src_info = torch::empty({num_local_experts, num_ranks * m}, opts_i32);
        auto layout_range =
            torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
        auto recv_count = torch::empty({num_local_experts}, opts_i32);

        auto clean = next_buffer.clean_meta();
        deep_ep::legacy::internode_ll::dispatch(
            packed_recv_x.data_ptr(), nullptr, src_info.data_ptr<int>(), layout_range.data_ptr<int64_t>(),
            recv_count.data_ptr<int>(), nullptr, nullptr, nullptr,
            buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
            buffer.dispatch_rdma_send_buffer, x.data_ptr(),
            reinterpret_cast<const topk_idx_t*>(topk_idx.data_ptr()), clean.first, clean.second,
            num_tokens, hidden, m, num_topk, num_experts, rank, num_ranks,
            /*use_fp8=*/false, /*round_scale=*/false, /*use_ue8m0=*/false, workspace, num_device_sms,
            at::cuda::getCurrentCUDAStream(),
            LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE);
        return {packed_recv_x, recv_count, src_info, layout_range};
    }

    torch::Tensor combine(const torch::Tensor& x, const torch::Tensor& topk_idx,
                          const torch::Tensor& topk_weights, const torch::Tensor& src_info,
                          const torch::Tensor& layout_range) {
        NANO_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
        NANO_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
        const int num_combined_tokens = static_cast<int>(topk_idx.size(0));
        const int num_topk = static_cast<int>(topk_idx.size(1));

        LowLatencyLayout layout(rdma_buffer, m, hidden, num_ranks, num_experts);
        auto buffer = layout.buffers[low_latency_buffer_idx];
        auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

        auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
        auto clean = next_buffer.clean_meta();
        deep_ep::legacy::internode_ll::combine(
            combined_x.data_ptr(), buffer.combine_rdma_recv_data_buffer,
            buffer.combine_rdma_recv_flag_buffer, buffer.combine_rdma_send_buffer, x.data_ptr(),
            reinterpret_cast<const topk_idx_t*>(topk_idx.data_ptr()), topk_weights.data_ptr<float>(),
            src_info.data_ptr<int>(), layout_range.data_ptr<int64_t>(), nullptr, nullptr,
            clean.first, clean.second, num_combined_tokens, hidden, m, num_topk, num_experts, rank,
            num_ranks, /*use_logfmt=*/false, workspace, num_device_sms,
            at::cuda::getCurrentCUDAStream(),
            LEGACY_LOW_LATENCY_SEND_PHASE | LEGACY_LOW_LATENCY_RECV_PHASE, /*zero_copy=*/false);
        return combined_x;
    }
};

static NanoBuffer* g_buffer = nullptr;

void buffer_create(int rank, int num_ranks, int num_experts, int hidden, int m) {
    NANO_HOST_ASSERT(g_buffer == nullptr);
    g_buffer = new NanoBuffer(rank, num_ranks, num_experts, hidden, m);
}

void buffer_destroy() { delete g_buffer; g_buffer = nullptr; }

size_t buffer_bytes() { NANO_HOST_ASSERT(g_buffer); return g_buffer->num_rdma_bytes; }

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
buffer_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx) {
    NANO_HOST_ASSERT(g_buffer);
    return g_buffer->dispatch(x, topk_idx);
}

torch::Tensor buffer_combine(const torch::Tensor& x, const torch::Tensor& topk_idx,
                             const torch::Tensor& topk_weights, const torch::Tensor& src_info,
                             const torch::Tensor& layout_range) {
    NANO_HOST_ASSERT(g_buffer);
    return g_buffer->combine(x, topk_idx, topk_weights, src_info, layout_range);
}

}  // namespace nanoep
