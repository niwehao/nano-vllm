#!/usr/bin/env bash
# scripts/patch_cuda_glibc241.sh —— 修 CUDA 12.8 头文件与 glibc >= 2.41 的冲突。
# 幂等，可重复跑；重装 toolkit 后再跑一遍即可。
#
# 症状（编任何 CUDA 扩展都会撞）：
#   /usr/include/x86_64-linux-gnu/bits/mathcalls.h(79): error:
#     exception specification is incompatible with that of previous function "cospi"
#     (declared at line 2601 of .../cuda-12.8/include/crt/math_functions.h)
#
# 根因：glibc 2.41 起把 C23 的 sinpi/cospi/tanpi 加进了 <math.h>，声明带 __THROW
# （C++ 里展开成 noexcept(true)）。CUDA 12.8 的 crt/math_functions.h 写在那之前，
# 这几个声明**没带** __THROW —— 而 C++ 里异常规格属于声明的一部分，两次声明不一致就是硬错误。
# 同一个文件里 fabs/fmin/fmax 等都是带 __THROW 的，只有 pi 系列漏了。
#
# 修法：给它们补上 __THROW，与上游 CUDA 12.9 的修法一致（也可以直接换 CUDA >= 12.9）。
set -euo pipefail
H="${CUDA_HOME:-$HOME/cuda-12.8}/include/crt/math_functions.h"
[ -f "$H" ] || { echo "找不到 $H（设 CUDA_HOME）"; exit 1; }

echo "glibc: $(ldd --version | head -1)"
echo "目标 : $H"
[ -f "$H.orig" ] || cp -a "$H" "$H.orig"

python3 - "$H" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
n = 0
# 只动"声明行"：extern ... <name>(...);  且还没有 __THROW 的
for fn in ("cospi", "cospif", "sinpi", "sinpif", "tanpi", "tanpif"):
    pat = re.compile(rf"^(extern[^\n;]*\b{fn}\s*\([^\n;]*\))\s*;", re.M)
    def add(m):
        global n
        if "__THROW" in m.group(1):
            return m.group(0)
        n += 1
        return m.group(1) + " __THROW;"
    s = pat.sub(add, s)
open(p, "w").write(s)
print(f"  补了 {n} 处 __THROW")
PY

echo "验证：编一个最小 CUDA 文件"
tmp=$(mktemp -d)
printf '#include <cuda_runtime.h>\n#include <cmath>\n__global__ void k(){}\n' > "$tmp/t.cu"
if "${CUDA_HOME:-$HOME/cuda-12.8}/bin/nvcc" -arch=sm_89 -std=c++17 -c "$tmp/t.cu" -o /dev/null 2>&1 | grep -q "error:"; then
  echo "❌ 仍有错误："
  "${CUDA_HOME:-$HOME/cuda-12.8}/bin/nvcc" -arch=sm_89 -std=c++17 -c "$tmp/t.cu" -o /dev/null 2>&1 | grep -m4 "error:"
  rm -rf "$tmp"; exit 1
fi
rm -rf "$tmp"
echo "✅ 通过（原文件已备份为 $H.orig）"
