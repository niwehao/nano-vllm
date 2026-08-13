// 公共声明与断言宏。对应 DeepEP 的 deep_ep/common/exception.cuh + api.cuh 的子集。
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define NANO_HOST_ASSERT(cond)                                                        \
    do {                                                                              \
        if (not(cond)) {                                                              \
            throw std::runtime_error(std::string("nano-deepEP assert failed: " #cond) \
                                     + " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                             \
    } while (0)

#define CUDA_CHECK(expr)                                                              \
    do {                                                                              \
        cudaError_t e = (expr);                                                       \
        if (e != cudaSuccess) {                                                       \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e) \
                                     + " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                             \
    } while (0)

namespace nanoep {

std::vector<uint8_t> get_unique_id();
int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks);
void* alloc(size_t size, size_t alignment);
void dealloc(void* ptr);
void barrier();
void finalize();
int my_pe();
int n_pes();
bool put_test(int64_t nelem);

// Buffer（nano_buffer.cu）
void buffer_create(int rank, int num_ranks, int num_experts, int hidden, int m);
void buffer_destroy();
size_t buffer_bytes();

}  // namespace nanoep
