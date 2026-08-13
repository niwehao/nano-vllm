#include "compiled.cuh"
#include "ibgda_device.cuh"
#include "launch.cuh"

namespace deep_ep::legacy {

namespace internode_ll {

template <bool use_warp_sync = false>
__forceinline__ __device__ bool is_rank_masked(int* mask_buffer_ptr, int rank) {
    if (mask_buffer_ptr == nullptr) {
        return false;
    }
    if constexpr (use_warp_sync) {
        return __shfl_sync(0xffffffff, ld_acquire_global(mask_buffer_ptr + rank), 0) != 0;
    } else {
        return ld_acquire_global(mask_buffer_ptr + rank) != 0;
    }
}

template <int kNumThreads>
__forceinline__ __device__ void barrier(int thread_id, int rank, int num_ranks, int* mask_buffer_ptr, int* sync_buffer_ptr) {
    EP_DEVICE_ASSERT(kNumThreads >= num_ranks);

    // Quiet all QPs
    auto qps_per_rank = ibgda_get_state()->num_rc_per_pe * ibgda_get_state()->num_devices_initialized;

    for (int i = thread_id; i < qps_per_rank * (num_ranks - 1); i += kNumThreads) {
        auto dst_rank = (rank + 1 + i / qps_per_rank) % num_ranks;
        auto qp_id = i % qps_per_rank;
        nvshmemi_ibgda_quiet(dst_rank, qp_id);
    }

    // Update local counter
    if (thread_id == 0)
        atomicAdd(sync_buffer_ptr + rank, -1);
    __syncthreads();

    int cnt = sync_buffer_ptr[rank];
    // Update remote counter and wait for local counter to be updated
    if (thread_id < num_ranks && thread_id != rank) {
        const auto dst_rank = thread_id;
        const auto dst_ptr = reinterpret_cast<uint64_t>(sync_buffer_ptr + rank);
        const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);

        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_rma_p(reinterpret_cast<int*>(dst_ptr), cnt, dst_rank, 0);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), cnt);
            }

            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            while (ld_acquire_sys_global(sync_buffer_ptr + dst_rank) != cnt            // remote is not ready
                   && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES  // not timeout
            )
                ;
            // Mask rank if timeout
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for barrier, rank %d, dst_rank %d\n", rank, dst_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + dst_rank, 1);
            }
        }
    }
    __syncthreads();
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_low_latency_buffer(int* clean_0,
                                                                           int num_clean_int_0,
                                                                           int* clean_1,
                                                                           int num_clean_int_1,
                                                                           int rank,
                                                                           int num_ranks,
                                                                           int* mask_buffer_ptr,
                                                                           int* sync_buffer_ptr) {
    auto thread_id = static_cast<int>(threadIdx.x);

    // Barrier before cleaning (in case of unfinished chunked EP)
    if (sync_buffer_ptr == nullptr)
        nvshmemx_barrier_all_block();
    else
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr);

    // Clean
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure the low-latency mode works fine)
    if (sync_buffer_ptr == nullptr)
        nvshmemx_barrier_all_block();
    else
        barrier<kNumThreads>(thread_id, rank, num_ranks, mask_buffer_ptr, sync_buffer_ptr);
}

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer_ptr,
                              int* sync_buffer_ptr,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);

    LAUNCH_KERNEL(&cfg,
                  clean_low_latency_buffer<kNumThreads>,
                  clean_0,
                  num_clean_int_0,
                  clean_1,
                  num_clean_int_1,
                  rank,
                  num_ranks,
                  mask_buffer_ptr,
                  sync_buffer_ptr);
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(void* packed_recv_x,
                                                    void* packed_recv_x_scales,
                                                    int* packed_recv_src_info,
                                                    int64_t* packed_recv_layout_range,
                                                    int* packed_recv_count,
                                                    int* mask_buffer_ptr,
                                                    int* cumulative_local_expert_recv_stats,
                                                    int64_t* dispatch_wait_recv_cost_stats,
                                                    void* rdma_recv_x,
                                                    int* rdma_recv_count,
                                                    void* rdma_x,
                                                    const void* x,
                                                    const topk_idx_t* topk_idx,
                                                    int* atomic_counter_per_expert,
                                                    int* atomic_finish_counter_per_expert,
                                                    int* next_clean,
                                                    int num_next_clean_int,
                                                    int num_tokens,
                                                    int num_max_dispatch_tokens_per_rank,
                                                    int num_topk,
                                                    int num_experts,
                                                    int rank,
                                                    int num_ranks,
                                                    int num_warp_groups,
                                                    int num_warps_per_group,
                                                    bool round_scale,
                                                    int phases) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: index at source (int), 3 reserved int fields, hidden data, FP8 scales
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int shared_num_tokens_sent_per_expert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = static_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(static_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            // Overlap top-k index read and source token index writes
            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            // FP8 cast
            EP_STATIC_ASSERT(hidden_bf16_int4 % 32 == 0, "Must use the full warp to reduce");
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                // Read
                auto int4_value = __ldg(x_int4 + i);

                if constexpr (kUseFP8) {
                    // Calculate local amax
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }

                    // Reduce amax and scale
                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    amax = warp_reduce_max<16>(amax);
                    calculate_fp8_scales(amax, scale, scale_inv, round_scale);
                    if (lane_id == 0 or lane_id == 16)
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                    // Cast into send buffer
                    vec_t int2_value;
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                    }
                    rdma_x_vec[i] = int2_value;
                } else {
                    // Reinterpret-cast is for C++14 compatibility
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            asm volatile("bar.sync 1, %0;" ::"r"(num_threads));

            // Issue IBGDA sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                    rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg + slot_idx * num_bytes_per_msg;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {
                    if (dst_p2p_ptr == 0) {
                        nvshmemi_ibgda_put_nbi_warp(dst_ptr, src_ptr, num_bytes_per_msg, dst_rank, dst_expert_local_idx, lane_id, slot_idx);
                    } else {
                        // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
                        const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                        const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_p2p_ptr);
                        UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                    }
                }

                // Increase counter after finishing
                __syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // The first SM is also responsible for checking QPs
            EP_DEVICE_ASSERT(ibgda_get_state()->num_rc_per_pe >= num_local_experts);

            // The first SM is also responsible for cleaning the next buffer
            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;

            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, LEGACY_FINISHED_SUM_TAG);
        }

        // This SM should be responsible for some destination experts, read `topk_idx` for them
        int expert_count[kNumMaxWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * num_warp_groups;
        const auto expert_end_idx = min(expert_begin_idx + num_warp_groups, num_experts);

        // Per lane count
        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx]++;
        }

        // Warp reduce
        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, LEGACY_FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * num_warp_groups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != LEGACY_FINISHED_SUM_TAG * 2)
            ;
        auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_count + dst_expert_local_idx * num_ranks + rank);
        auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
        if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
            if (dst_p2p_ptr == 0) {
                nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), -num_tokens_sent - 1, dst_rank, dst_expert_local_idx);
            } else {
                st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), -num_tokens_sent - 1);
            }
        }

        // Clean workspace for next use
        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        // Clean `packed_recv_count`
        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LEGACY_LOW_LATENCY_SEND_PHASE)
        cg::this_grid().sync();

    // Receiving and packing
    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = static_cast<uint8_t*>(rdma_recv_x) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
            src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 =
            static_cast<int4*>(packed_recv_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;
        const auto num_aligned_scales = align_up<int>(num_scales, sizeof(float) / sizeof(scale_t));
        const auto recv_x_scales = static_cast<scale_t*>(packed_recv_x_scales) +
            local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_aligned_scales;

        // Shared between sub-warps in warp groups
        __shared__ int shared_num_recv_tokens[kNumMaxWarpGroups], shared_recv_token_begin_idx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int num_recv_tokens = 0, recv_token_begin_idx;
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 15);
        if (sub_warp_id == 1 and lane_id == 0) {
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) ==
                           0                                                               // data not arrived
                       && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES  // not timeout
                )
                    ;
            }
            // Do not receive tokens if rank timeout or masked
            if (num_recv_tokens == 0)
                num_recv_tokens = -1;
            // Mask rank if timeout
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for dispatch receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       local_expert_idx,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
            }

            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);

            // Add stats for diagnosis
            if (cumulative_local_expert_recv_stats != nullptr)
                atomicAdd(cumulative_local_expert_recv_stats + local_expert_idx, num_recv_tokens);
            if (dispatch_wait_recv_cost_stats != nullptr)
                atomicAdd(reinterpret_cast<unsigned long long*>(dispatch_wait_recv_cost_stats + src_rank), wait_recv_cost);
        }
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 2), "r"(num_warps_per_group * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        // Copy tokens
        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += num_warps_per_group) {
            // Copy source info
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();

            // Copy data
            // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            // Copy scales
            if constexpr (kUseFP8) {
                // Equivalent CuTe layout:
                //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto num_elems_per_pack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
                const auto token_idx = recv_token_begin_idx + i;
                const auto token_stride = num_elems_per_pack;
                const auto pack_stride = num_ranks * num_max_dispatch_tokens_per_rank * num_elems_per_pack;
                if (lane_id < num_scales) {
                    const auto pack_idx = lane_id / num_elems_per_pack;
                    const auto elem_idx = lane_id % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
                if (lane_id + 32 < num_scales) {
                    const auto pack_idx = (lane_id + 32) / num_elems_per_pack;
                    const auto elem_idx = (lane_id + 32) % num_elems_per_pack;
                    auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(src_scales + lane_id + 32));
                    recv_x_scales[token_idx * token_stride + pack_idx * pack_stride + elem_idx] = scale;
                }
            }
        }
    }
}

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer_ptr,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,
              cudaStream_t stream,
              int phases) {
    constexpr int kNumMaxTopK = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= num_warp_groups * num_warps_per_group);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    const auto num_sms = ceil_div(num_experts, num_warp_groups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    // Workspace checks
    auto atomic_counter_per_expert = static_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= LEGACY_NUM_WORKSPACE_BYTES);

    // FP8 checks
    if (use_ue8m0)
        EP_HOST_ASSERT(round_scale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden)                         \
    {                                                        \
        auto dispatch_func = dispatch<false, false, hidden>; \
        if (use_fp8 and not use_ue8m0)                       \
            dispatch_func = dispatch<true, false, hidden>;   \
        if (use_fp8 and use_ue8m0)                           \
            dispatch_func = dispatch<true, true, hidden>;    \
        LAUNCH_KERNEL(&cfg,                                  \
                      dispatch_func,                         \
                      packed_recv_x,                         \
                      packed_recv_x_scales,                  \
                      packed_recv_src_info,                  \
                      packed_recv_layout_range,              \
                      packed_recv_count,                     \
                      mask_buffer_ptr,                       \
                      cumulative_local_expert_recv_stats,    \
                      dispatch_wait_recv_cost_stats,         \
                      rdma_recv_x,                           \
                      rdma_recv_count,                       \
                      rdma_x,                                \
                      x,                                     \
                      topk_idx,                              \
                      atomic_counter_per_expert,             \
                      atomic_finish_counter_per_expert,      \
                      next_clean,                            \
                      num_next_clean_int,                    \
                      num_tokens,                            \
                      num_max_dispatch_tokens_per_rank,      \
                      num_topk,                              \
                      num_experts,                           \
                      rank,                                  \
                      num_ranks,                             \
                      num_warp_groups,                       \
                      num_warps_per_group,                   \
                      round_scale,                           \
                      phases);                               \
    }                                                        \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

// [nano-deepEP] 手术：删除 logfmt_encode / logfmt_check_amaxmin（原 :556-670）。
// LogFMT 是一种 10 bit 的动态对数量化格式，本项目恒走 BF16、用不上；更重要的是它们
// 直接调用 tma_store_fence()，而那是 SM90 专有、在 DISABLE_SM90_FEATURES 下根本不存在——
// 函数模板里的**非依赖名**在定义处就要查找，即便永不实例化也会编译失败，所以必须删掉
// 而不是留着不用。下面的 decode_and_accumulate 原样保留（它的 enable_cast 分支同样
// 永远走不到，但那是纯算术、不引用任何 SM90 指令）。

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decode_and_accumulate(
    uint32_t* ld_buffer, float* accum, const float& log_amax, const float& log_amin, const bool& enable_cast, const float& weight) {
    if (enable_cast) {
        constexpr int kNumBits = 10;
        constexpr int kNumValues = 1 << (kNumBits - 1);

        const auto step = (log_amax - log_amin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t& encoded, const uint32_t& sign) {
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + log_amin);
            return sign ? -decoded : decoded;
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++i) {
            uint32_t concat[6];
            concat[0] = ld_buffer[i * 5];
            #pragma unroll
            for (int k = 1; k < 5; ++k)
                concat[k] = (ld_buffer[i * 5 + k - 1] >> (32 - k * 5)) | (ld_buffer[i * 5 + k] << (k * 5));
            concat[5] = ld_buffer[i * 5 + 4] >> 7;

            const uint32_t local_signs = ld_buffer[i * 5 + 4] >> 16;
            #pragma unroll
            for (int k = 0; k < 5; ++k) {
                accum[i * 16 + k * 3 + 0] += decode((concat[k] >> 0) & 0x1ff, (local_signs >> (k * 3 + 0)) & 1) * weight;
                accum[i * 16 + k * 3 + 1] += decode((concat[k] >> 9) & 0x1ff, (local_signs >> (k * 3 + 1)) & 1) * weight;
                accum[i * 16 + k * 3 + 2] += decode((concat[k] >> 18) & 0x1ff, (local_signs >> (k * 3 + 2)) & 1) * weight;
            }
            accum[i * 16 + 15] += decode(concat[5] & 0x1ff, (local_signs >> 15) & 1) * weight;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++k) {
            auto bf16_pack = *reinterpret_cast<__nv_bfloat162*>(ld_buffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16_pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16_pack.y) * weight;
        }
    }
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls>
__global__ __launch_bounds__(1024, 1) void combine(void* combined_x,
                                                   void* rdma_recv_x,
                                                   int* rdma_recv_flag,
                                                   void* rdma_send_x,
                                                   const void* x,
                                                   const topk_idx_t* topk_idx,
                                                   const float* topk_weights,
                                                   const int* src_info,
                                                   const int64_t* layout_range,
                                                   int* mask_buffer_ptr,
                                                   int64_t* combine_wait_recv_cost_stats,
                                                   int* next_clean,
                                                   int num_next_clean_int,
                                                   int* atomic_clean_flag,
                                                   int num_combined_tokens,
                                                   int hidden,
                                                   int num_topk,
                                                   int num_max_dispatch_tokens_per_rank,
                                                   int num_experts,
                                                   int rank,
                                                   int num_ranks,
                                                   int num_warp_groups,
                                                   int num_warps_per_group,
                                                   int phases,
                                                   bool zero_copy) {
    const auto sm_id = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto num_sms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warp_id = __shfl_sync(0xffffffff, thread_id / 32, 0), lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / num_warps_per_group;
    const auto sub_warp_id = warp_id % num_warps_per_group;
    const auto responsible_expert_idx = sm_id * num_warp_groups + warp_group_id;

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hidden_bf16_int4_pad = align_up(static_cast<int>(hidden_bf16_int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hidden_bf16_int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LEGACY_LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        // Notify before executing `int_p`
        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    // Issue IBGDA sends
    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x =
            static_cast<const int4*>(x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec =
            static_cast<uint8_t*>(rdma_send_x) + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        // Unpack layout
        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        // [nano-deepEP] 手术：TMA(cp.async.bulk)是 SM90 专有指令，Ada 没有。
        // 这里整段删除上游的 TMA 三段流水（smem 缓冲、mbarrier、prefetch/store 的
        // stage 轮转），下面的拷贝改成与 dispatch 内核 :271 同款的 warp copy。
        // 代价：没有 L2→smem 的异步预取，但 LL 的消息很小（hidden*2 = 4KB/token），
        // 瓶颈在网络 RTT 不在 SM 拷贝。

        // Issue IBGDA send
        if (not is_rank_masked<true>(mask_buffer_ptr, dst_rank)) {
            for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += num_warps_per_group) {
                const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
                const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
                const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

                // Copy directly to local rank, or copy to buffer and issue RDMA
                const auto src_idx = __shfl_sync(0xffffffff, __ldg(local_src_info + token_idx), 0);
                const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                    (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
                const auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
                int num_send_bytes = hidden * sizeof(nv_bfloat16);

                // [nano-deepEP] 手术：原本是 TMA 载入 smem →(LogFMT)→ TMA 存到
                // buf/p2p → tma_store_wait 的三段流水。zero_copy 恒 false、LogFMT 恒关，
                // 所以退化成一次直白的 warp copy：从 x 读、写进 RDMA 发送槽（或对端 p2p）。
                // 注意目的偏移是 **0**：BF16 模式下数据就放在槽首，meta 区（kNumMetaBytes）
                // 只有 LogFMT 用得着，这里空着不动 —— 保持与上游逐字节同布局，
                // 便于将来和 DeepEP 原版对照调试。
                {
                    const auto cpy_src_int4_ptr = x_int4;
                    const auto cpy_dst_int4_ptr =
                        dst_p2p_ptr == 0 ? reinterpret_cast<int4*>(buf_ptr) : reinterpret_cast<int4*>(dst_p2p_ptr);
                    UNROLLED_WARP_COPY(kNumSendUnrolls,
                                       lane_id,
                                       hidden_bf16_int4,
                                       cpy_dst_int4_ptr,
                                       cpy_src_int4_ptr,
                                       ld_nc_global,
                                       st_na_global);
                    __syncwarp();
                }

                // Issue RDMA
                // NOTES: for zero-copy mode, we assume the data is already in the send buffer
                if (dst_p2p_ptr == 0)
                    nvshmemi_ibgda_put_nbi_warp(dst_ptr, buf_ptr, num_send_bytes, dst_rank, local_expert_idx, lane_id, token_idx - offset);
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(num_warps_per_group > 1 and num_warp_groups < 16);
        asm volatile("bar.sync %0, %1;" ::"r"(warp_group_id + 1), "r"(num_warps_per_group * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0)
                ;
            auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_flag + global_expert_idx);
            auto dst_p2p_ptr = nvshmemi_get_p2p_ptr(dst_ptr, rank, dst_rank);
            if (not is_rank_masked(mask_buffer_ptr, dst_rank)) {
                if (dst_p2p_ptr == 0) {
                    nvshmemi_ibgda_amo_nonfetch_add(reinterpret_cast<int*>(dst_ptr), 1, dst_rank, local_expert_idx);
                } else {
                    st_release_sys_global(reinterpret_cast<int*>(dst_p2p_ptr), 1);
                }
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();

        // [nano-deepEP] 手术：没有 mbarrier 了，不需要销毁
    }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
    if ((phases & LEGACY_LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsible_expert_idx < num_experts) {
        EP_DEVICE_ASSERT(num_warps_per_group > 1);
        if (sub_warp_id == 0 and lane_id == 0) {
            const auto src_rank = responsible_expert_idx / num_local_experts;
            auto start_time = clock64();
            uint64_t wait_recv_cost = 0;
            if (not is_rank_masked(mask_buffer_ptr, src_rank)) {
                while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0  // recv not ready
                       && (wait_recv_cost = clock64() - start_time) <= LEGACY_NUM_TIMEOUT_CYCLES   // not timeout
                )
                    ;
            }
            // Mask rank if timeout
            if (wait_recv_cost > LEGACY_NUM_TIMEOUT_CYCLES) {
                printf("Warning: DeepEP timeout for combine receive, rank %d, local_expert_idx %d, src_rank %d\n",
                       rank,
                       responsible_expert_idx % num_local_experts,
                       src_rank);
                if (mask_buffer_ptr == nullptr)
                    trap();
                atomicExch(mask_buffer_ptr + src_rank, 1);
            }

            if (combine_wait_recv_cost_stats != nullptr) {
                atomicAdd(reinterpret_cast<unsigned long long*>(combine_wait_recv_cost_stats + src_rank), wait_recv_cost);
            }
        }
    }
    cg::this_grid().sync();

    // [nano-deepEP] 手术：接收侧整段重写。
    //
    // 上游是 "1 个 TMA 载入 warp + N 个归约 warp" 的 producer/consumer 结构，靠 mbarrier
    // 在 smem 上做三级流水（原 :978-1138）。TMA(cp.async.bulk) 与 mbarrier 都是 SM90 专有，
    // Ada 一个都没有，所以退回最朴素的形式：**每个 warp 负责一个 token**，直接从
    // rdma_recv_x 读、fp32 累加、bf16 写回。没有 smem、没有屏障、没有角色分工。
    //
    // 两个必须守住的语义：
    //   1) 归约按 **k 升序**，与 nccl_backend 的 scatter+sum 一致 —— 两个后端才能位级对拍；
    //   2) 数据在槽的 **偏移 0**（meta 区只有 LogFMT 用得着），与发送侧的 warp copy 对应。
    //
    // 读用 ld_nc_global，与上游 dispatch 接收侧 (:436) 同款：等 rdma_recv_flag 那次
    // ld_acquire_sys_global 加上 cg::this_grid().sync() 已经建立了可见性顺序。
    //
    // 寄存器核算：每 lane 每轮 kNumRecvUnrolls(=2) 个 int4 = 16 个 fp32 累加器，
    // 加少量指针，远低于 1024 线程/block 的预算。
    {
        const int num_recv_warps = num_threads / 32;
        for (int token_idx = sm_id * num_recv_warps + warp_id; token_idx < num_combined_tokens;
             token_idx += num_sms * num_recv_warps) {
            int topk_idx_by_lane = -1;
            float topk_weights_by_lane = 0.0f;
            if (lane_id < num_topk) {
                topk_idx_by_lane = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + lane_id));
                topk_weights_by_lane = __ldg(topk_weights + token_idx * num_topk + lane_id);
            }
            __syncwarp();

            for (int base = lane_id * kNumRecvUnrolls; base < hidden_bf16_int4; base += 32 * kNumRecvUnrolls) {
                float accum[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};

                for (int i = 0; i < num_topk; ++i) {
                    const int expert_idx = __shfl_sync(0xffffffff, topk_idx_by_lane, i);
                    if (expert_idx < 0)
                        continue;
                    if (is_rank_masked(mask_buffer_ptr, expert_idx / num_local_experts))
                        continue;
                    const float weight = __shfl_sync(0xffffffff, topk_weights_by_lane, i);
                    const auto slot = reinterpret_cast<const int4*>(
                        static_cast<const uint8_t*>(rdma_recv_x) +
                        (static_cast<size_t>(expert_idx) * num_max_dispatch_tokens_per_rank + token_idx) *
                            num_bytes_per_slot);

                    #pragma unroll
                    for (int u = 0; u < kNumRecvUnrolls; ++u) {
                        const auto v = ld_nc_global(slot + base + u);
                        const auto bf = reinterpret_cast<const __nv_bfloat162*>(&v);
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            accum[u * 8 + k * 2 + 0] += static_cast<float>(bf[k].x) * weight;
                            accum[u * 8 + k * 2 + 1] += static_cast<float>(bf[k].y) * weight;
                        }
                    }
                }

                auto dst = static_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4 + base;
                #pragma unroll
                for (int u = 0; u < kNumRecvUnrolls; ++u) {
                    int4 packed;
                    auto out = reinterpret_cast<__nv_bfloat162*>(&packed);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k)
                        out[k] = __nv_bfloat162(accum[u * 8 + k * 2 + 0], accum[u * 8 + k * 2 + 1]);
                    st_na_global(dst + u, packed);
                }
            }
        }
    }
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer_ptr,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy) {
    constexpr int kNumMaxTopk = 11;
    const int num_warp_groups = ceil_div(num_experts, num_device_sms);
    const int num_warps_per_group = 32 / num_warp_groups;
    const int num_recv_per_sm = ceil_div(num_combined_tokens, num_device_sms);
    EP_HOST_ASSERT(num_warp_groups > 0 and num_warps_per_group > 0 and num_recv_per_sm >= 0);

    const auto num_warps = num_warp_groups * num_warps_per_group;
    // [nano-deepEP] 加了个 min(..., num_device_sms) 的封顶：内核用 cooperative launch
    // （combine 里有 cg::this_grid().sync()），要求**所有 block 同时驻留**，grid 大于
    // 常驻上限时 cudaLaunchKernelEx 会返回 too-many-blocks。L40S 有 142 个 SM、
    // 1024 线程/block → 每 SM 一个 block，所以 grid 不能超过 SM 数。
    // 上游公式在 T=512 时给 128，本来就没超，但这里显式封住防御。
    const auto num_sms = min(
        max(ceil_div(num_experts, num_warp_groups), num_recv_per_sm == 0 ? 1 : ceil_div(num_combined_tokens, num_recv_per_sm)),
        num_device_sms);

    // Check workspace
    auto atomic_clean_flag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= LEGACY_NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

    // [nano-deepEP] 手术：LogFMT 与 zero_copy 两条路都不支持，直接在入口挡住
    EP_HOST_ASSERT(not use_logfmt and "nano-deepEP 不支持 LogFMT（内核里已删除）");
    EP_HOST_ASSERT(not zero_copy and "nano-deepEP 不支持 zero_copy（get_next_low_latency_combine_buffer 未移植）");

    constexpr int kNumMaxUnrolls = 4;

    // [nano-deepEP] 手术：TMA 全部删掉了，一个字节动态 smem 都不需要
    // （上游这里按 kNumStages 的 TMA 缓冲 + LogFMT 的 meta 区算 smem_size）。

#define COMBINE_LAUNCH_CASE(hidden)                                                                                                \
    {                                                                                                                              \
        auto combine_func =                                                                                                        \
            combine<false, hidden, kNumMaxTopk, kNumMaxUnrolls>; /* [nano-deepEP] LogFMT 恒关 */ \
        SET_SHARED_MEMORY_FOR_TMA(combine_func);                                                                                   \
        LAUNCH_KERNEL(&cfg,                                                                                                        \
                      combine_func,                                                                                                \
                      combined_x,                                                                                                  \
                      rdma_recv_x,                                                                                                 \
                      rdma_recv_flag,                                                                                              \
                      rdma_send_x,                                                                                                 \
                      x,                                                                                                           \
                      topk_idx,                                                                                                    \
                      topk_weights,                                                                                                \
                      src_info,                                                                                                    \
                      layout_range,                                                                                                \
                      mask_buffer_ptr,                                                                                             \
                      combine_wait_recv_cost_stats,                                                                                \
                      next_clean,                                                                                                  \
                      num_next_clean_int,                                                                                          \
                      atomic_clean_flag,                                                                                           \
                      num_combined_tokens,                                                                                         \
                      hidden,                                                                                                      \
                      num_topk,                                                                                                    \
                      num_max_dispatch_tokens_per_rank,                                                                            \
                      num_experts,                                                                                                 \
                      rank,                                                                                                        \
                      num_ranks,                                                                                                   \
                      num_warp_groups,                                                                                             \
                      num_warps_per_group,                                                                                         \
                      phases,                                                                                                      \
                      zero_copy);                                                                                                  \
    }                                                                                                                              \
    break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor) {
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_threads = num_sms * kNumThreads;
    const auto thread_id = sm_id * kNumThreads + static_cast<int>(threadIdx.x);
    for (int rank_id = thread_id; rank_id < num_ranks; rank_id += num_threads)
        mask_tensor[rank_id] = mask_buffer_ptr[rank_id];
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* mask_tensor, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 1024;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, query_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks, mask_tensor);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    if (sm_id == 0 and thread_id == 0)
        atomicExch(mask_buffer_ptr + rank_to_mask, mask ? 1 : 0);
}

void update_mask_buffer(int* mask_buffer_ptr, int rank, bool mask, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, update_mask_buffer<kNumThreads>, mask_buffer_ptr, rank, mask);
}

template <int kNumThreads>
__launch_bounds__(kNumThreads, 1) __global__ void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks) {
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_ranks; i += kNumThreads)
        mask_buffer_ptr[i] = 0;
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    constexpr int num_sms = 1;
    constexpr int kNumThreads = 32;
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_mask_buffer<kNumThreads>, mask_buffer_ptr, num_ranks);
}

}  // namespace internode_ll

}  // namespace deep_ep::legacy
