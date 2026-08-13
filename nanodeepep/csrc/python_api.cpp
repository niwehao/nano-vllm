// pybind 入口。对应 DeepEP 的 csrc/python_api.cpp，裁到只剩 nano 用得上的。
#include <torch/extension.h>

#include "nano_ep.cuh"

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
}
