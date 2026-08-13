"""nanodeepep 的扩展构建，仿 DeepEP setup.py 裁剪到 SM89 + 只留 nvshmem 后端需要的部分。

    CUDA_HOME=$HOME/cuda-12.8 .venv/bin/python nanodeepep/setup.py build_ext --inplace

在哪个目录调用都可以：脚本自己会切到仓库根目录，因为 --inplace 的落点是按 **cwd**
算的（扩展名叫 nanodeepep._C，落点就是 <cwd>/nanodeepep/_C*.so）。在 nanodeepep/ 里
直接跑会去找 nanodeepep/nanodeepep/ 而失败。

关键点（都对着 DeepEP setup.py 的对应行）：
  * `-rdc=true` + `nvcc_dlink`：NVSHMEM 的 device 库是静态库，必须做单独的设备端链接
    （setup.py:115 的 `['-dlink', '-L.../lib', '-lnvshmem_device']`）。
  * pip wheel 的 `libnvshmem_host.so.3` **只有带版本号的名字**，没有 `.so` 软链，
    所以 `-l:libnvshmem_host.so` 解析不了 —— 构建时把真实文件名找出来传给 linker
    （setup.py:26-47 专门处理了这件事）。
  * `-DDISABLE_AGGRESSIVE_PTX_INSTRS`：`ld.global.nc.L1::no_allocate` 之类只在 Hopper
    验证过，上游对非 9.0 架构本来就强制关掉（setup.py:147-150）。
  * `TORCH_CUDA_ARCH_LIST=8.9`：上游的 `DISABLE_SM90_FEATURES` 路径直接
    `assert False, 'Not implemented'`，所以 SM89 的移植必须自己做。
"""

import os
from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent


def find_nvshmem_root() -> Path:
    """从 venv 的 pip wheel 里找 NVSHMEM（对应 DeepEP 的 find_pkgs.find_nvshmem_root）。"""
    import nvidia
    for base in nvidia.__path__:
        p = Path(base) / "nvshmem"
        if (p / "include" / "nvshmem.h").exists():
            return p
    raise ModuleNotFoundError("找不到 nvidia-nvshmem-cu12，先 `uv pip install nvidia-nvshmem-cu12`")


def versioned_so(lib_dir: Path, prefix: str) -> str:
    """pip wheel 只发 SONAME 文件名（libnvshmem_host.so.3），没有无版本号的软链。"""
    plain = lib_dir / f"{prefix}.so"
    if plain.exists():
        return plain.name
    for f in sorted(lib_dir.glob(f"{prefix}.so.*")):
        return f.name
    raise ModuleNotFoundError(f"{prefix}.so 在 {lib_dir} 下找不到")


if __name__ == "__main__":
    os.chdir(HERE.parent)          # 见文档字符串：--inplace 的落点按 cwd 算
    nvshmem = find_nvshmem_root()
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.getenv("TORCH_CUDA_ARCH_LIST", "8.9")

    cxx_flags = ["-O3", "-Wno-deprecated-declarations", "-Wno-unused-variable",
                 "-Wno-sign-compare", "-DDISABLE_AGGRESSIVE_PTX_INSTRS"]
    nvcc_flags = ["-O3", "-Xcompiler", "-O3", "--extended-lambda",
                  "--diag-suppress=128,2417", "-rdc=true",
                  "-DDISABLE_AGGRESSIVE_PTX_INSTRS",
                  # torch 的 cpp_extension 默认加 -D__CUDA_NO_HALF_OPERATORS__ 等一串，
                  # 把 __half/__nv_bfloat16 的算术运算符全禁掉；而 NVSHMEM 的设备端归约
                  # 模板 (include/non_abi/device/coll/reduce.cuh:95-101) 正好要用 + * < >，
                  # 于是报 "no operator + matches these operands"。这些 -U 排在 torch 的
                  # -D 之后，撤销即可。（flash-attn 等项目同款处理。）
                  "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                  "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__"]
    sources = [str(HERE / "csrc" / "python_api.cpp"), str(HERE / "csrc" / "nvshmem_glue.cu"),
               str(HERE / "csrc" / "nano_buffer.cu")]
    ll = HERE / "csrc" / "legacy" / "internode_ll.cu"
    if os.getenv("NANOEP_WITH_LL", "1") == "1" and ll.exists():
        sources.append(str(ll))
        # DISABLE_SM90_FEATURES 是白拿的手术刀：utils.cuh 里 elect_one_sync 自动走 lane0
        # 回退、整段 TMA 定义被条件编译掉。我们只需要额外覆盖 launch.cuh 的启动宏
        # （要 cooperative、不要 cluster）与 compiled.cuh 的 FP8 分支（SM89 原生支持）。
        nvcc_flags.append("-DDISABLE_SM90_FEATURES")
        cxx_flags.append("-DDISABLE_SM90_FEATURES")

    host_lib = versioned_so(nvshmem / "lib", "libnvshmem_host")
    print(f" > NVSHMEM: {nvshmem}")
    print(f" > host lib: {host_lib}")
    print(f" > sources : {sources}")

    setuptools.setup(
        name="nanodeepep_C",
        ext_modules=[CUDAExtension(
            name="nanodeepep._C",
            sources=sources,
            include_dirs=[str(HERE / "csrc"), str(HERE / "csrc" / "legacy"),
                          str(nvshmem / "include")],
            library_dirs=[str(nvshmem / "lib")],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
                "nvcc_dlink": ["-dlink", f"-L{nvshmem / 'lib'}", "-lnvshmem_device"],
            },
            extra_link_args=["-lcuda", f"-l:{host_lib}", "-l:libnvshmem_device.a",
                             f"-Wl,-rpath,{nvshmem / 'lib'}"],
        )],
        cmdclass={"build_ext": BuildExtension},
    )
