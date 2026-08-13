#pragma once

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900
#define __CUDACC_RDC__
#define __CUDACC__
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// [nano-deepEP] 修改原因：上游这里在 DISABLE_SM90_FEATURES 下把 FP8 桩掉，因为那条分支
// 是给 Ampere(SM80) 准备的。SM89(Ada) **原生支持** FP8 转换指令
// (__nv_cvt_float2_to_fp8x2，dispatch 内核 :243 要用)，所以恒包含真头文件。
// 我们靠 DISABLE_SM90_FEATURES 拿到 elect_one_sync 的 lane0 回退和 TMA 段的条件编译，
// 但不想连 FP8 一起被关掉。
#include <cuda_fp8.h>

// Compatibility: 256 bits LD/ST instructions
#if defined(CUDART_VERSION) and CUDART_VERSION >= 13000
using longlong4_t = longlong4_32a;
#define make_longlong4_t make_longlong4_32a
#else
struct alignas(32) longlong4_t { long long x, y, z, w; };
__device__ __forceinline__ longlong4_t make_longlong4_t(
    const long long& x, const long long& y, const long long& z, const long long& w) {
    return {x, y, z, w};
}
#endif

#ifndef EP_NUM_TOPK_IDX_BITS
#define EP_NUM_TOPK_IDX_BITS 64
#endif

namespace deep_ep {

#ifndef DISABLE_SM90_FEATURES
constexpr bool kEnableSM90Features = true;
#else
constexpr bool kEnableSM90Features = false;
#endif

template <int kNumBits> struct int_with_bits;
template <> struct int_with_bits<8>  { using type = int8_t;  };
template <> struct int_with_bits<16> { using type = int16_t; };
template <> struct int_with_bits<32> { using type = int32_t; };
template <> struct int_with_bits<64> { using type = int64_t; };

using topk_idx_t = int_with_bits<EP_NUM_TOPK_IDX_BITS>::type;

union sf_pack_t {
    float fp32;
    int ue8m0x4;
};

constexpr int kNumTMAAlignedBytes = 16;
constexpr int kNumAlignedSFPacks = 16 / sizeof(sf_pack_t);

// Some communication channel settings
constexpr int kNumMaxChannels = 1024;
constexpr int kGinQPDepth = 1024;
constexpr int kGinQPFlushDepth = 768;

} // namespace deep_ep
