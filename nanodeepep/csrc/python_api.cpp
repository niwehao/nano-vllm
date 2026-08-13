// pybind 入口。对应 DeepEP 的 csrc/python_api.cpp，裁到只剩 nano 用得上的。
#include <torch/extension.h>

#include "nano_ep.cuh"

namespace nanoep {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
buffer_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx);
torch::Tensor buffer_combine(const torch::Tensor& x, const torch::Tensor& topk_idx,
                             const torch::Tensor& topk_weights, const torch::Tensor& src_info,
                             const torch::Tensor& layout_range);
}  // namespace nanoep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "nano-deepEP 的 NVSHMEM/IBGDA 后端";
    m.def("get_unique_id", &nanoep::get_unique_id,
          "rank0 生成，经 gloo 广播给所有 rank（协议同 deep_ep legacy.py:104-136）");
    m.def("init", &nanoep::init, "用 unique id 初始化 NVSHMEM，返回本进程的 PE 号");
    m.def("barrier", &nanoep::barrier);
    m.def("finalize", &nanoep::finalize);
    m.def("my_pe", &nanoep::my_pe);
    m.def("n_pes", &nanoep::n_pes);
    m.def("put_test", &nanoep::put_test,
          "device 侧发起的 RDMA put 冒烟测试（走的就是 IBGDA 路径）");

    m.def("ibgda_probe", &nanoep::ibgda_probe,
          "打印设备侧 IBGDA 状态，并走一次 DeepEP 手写 WQE 的 put");

    m.def("buffer_create", &nanoep::buffer_create,
          "分配 LL 的 RDMA 对称缓冲（双缓冲布局同 config.hpp:102-188）");
    m.def("buffer_destroy", &nanoep::buffer_destroy);
    m.def("buffer_bytes", &nanoep::buffer_bytes);
    m.def("buffer_dispatch", &nanoep::buffer_dispatch,
          "LL dispatch：返回 (packed_recv_x, recv_count, src_info, layout_range)");
    m.def("buffer_combine", &nanoep::buffer_combine, "LL combine：返回 combined_x");
}
