#pragma once

#include "compiled.cuh"

// [nano-deepEP] 修改原因：上游有两条路径 —— 不带 DISABLE_SM90_FEATURES 时是
// "cooperative + cluster"，带上时退化成普通 <<<>>> 启动。两条都不合用：
//   * cluster（cudaLaunchAttributeClusterDimension）是 SM90 专有，Ada 不支持；
//   * 但 cooperative 必须保留 —— dispatch:359 和 combine:976 都有 cg::this_grid().sync()，
//     没有 cudaLaunchCooperativeKernel 语义的话 grid 级同步是未定义行为。
// 所以这里给出第三种组合：cooperative 但不带 cluster。
#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream)                       \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[1];                                                \
    attr[0].id = cudaLaunchAttributeCooperative;                                \
    attr[0].val.cooperative = 1;                                                \
    cfg.attrs = attr;                                                           \
    cfg.numAttrs = 1
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_RUNTIME_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif

// [nano-deepEP] TMA 全部删掉了，不需要动态 smem
#ifndef SET_SHARED_MEMORY_FOR_TMA
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#endif

#define SWITCH_RANKS(case_macro)                           \
    switch (num_ranks) {                                   \
        case 2:                                            \
            case_macro(2);                                 \
        case 4:                                            \
            case_macro(4);                                 \
        case 8:                                            \
            case_macro(8);                                 \
        default:                                           \
            EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                      \
    while (false)

#define SWITCH_RDMA_RANKS(case_macro)                           \
    switch (num_ranks / LEGACY_NUM_MAX_NVL_PEERS) {             \
        case 2:                                                 \
            case_macro(2);                                      \
        case 3:                                                 \
            case_macro(3);                                      \
        case 4:                                                 \
            case_macro(4);                                      \
        case 6:                                                 \
            case_macro(6);                                      \
        case 8:                                                 \
            case_macro(8);                                      \
        case 12:                                                \
            case_macro(12);                                     \
        case 16:                                                \
            case_macro(16);                                     \
        case 18:                                                \
            case_macro(18);                                     \
        case 20:                                                \
            case_macro(20);                                     \
        default:                                                \
            EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    }                                                           \
    while (false)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro)         \
    switch (num_ranks) {                                   \
        case 2:                                            \
            case_macro(dtype, 2);                          \
        case 4:                                            \
            case_macro(dtype, 4);                          \
        case 8:                                            \
            case_macro(dtype, 8);                          \
        default:                                           \
            EP_HOST_ASSERT(false and "Unsupported ranks"); \
    }                                                      \
    while (false)

#define SWITCH_TYPES(case_macro)                          \
    switch (type) {                                       \
        case CUDA_R_16BF:                                 \
            case_macro(nv_bfloat16);                      \
        default:                                          \
            EP_HOST_ASSERT(false and "Unsupported type"); \
    }                                                     \
    while (false)

// [nano-deepEP] 修改原因：每多一个 hidden 就多实例化一整套内核模板，编译时间和 .so
// 体积都翻倍。本项目 tiny-qwen3-moe 只有 hidden=2048，其余全部裁掉。
// 上游原表（要支持别的模型时照着加回来）：
//   2048 / 2560 / 3072(gpt-oss) / 4096 / 5120 / 6144(qwen3-coder) / 7168 / 8192
#define SWITCH_HIDDEN(case_macro)                           \
    switch (hidden) {                                       \
        case 2048:                                          \
            case_macro(2048);                               \
        default:                                            \
            EP_HOST_ASSERT(false and "Unsupported hidden"); \
    }                                                       \
    while (false)
