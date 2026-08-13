#!/usr/bin/env bash
# scripts/m0_ibgda_check.sh —— IBGDA 前置条件检查（两机各查一遍，只读，不改任何东西）。
#   ./scripts/m0_ibgda_check.sh
# 判定规则见 Plan-4/01-m0-environment.md 的 L4：
#   方案 A = PeerMappingOverride=1（GPU 直接按门铃，性能最好）
#   方案 B = gdrdrv 已加载（CPU 辅助按门铃，NVSHMEM_IBGDA_NIC_HANDLER=cpu）
# 两个都没有 → M5 冻结。
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/hosts.sh"

probe() {
cat <<'PROBE'
MODINFO=$(command -v modinfo || echo /sbin/modinfo)
rd=$(grep -oP 'RegistryDwords:\s*"\K[^"]*' /proc/driver/nvidia/params 2>/dev/null)
echo "  驱动           : $(sed -n 's/.*NVRM version: //p;q' /proc/driver/nvidia/version | cut -c1-60)"
echo "  RegistryDwords : \"${rd}\""
if echo "$rd" | grep -q 'PeerMappingOverride=1'; then echo "  方案A(PeerMappingOverride) : ✅ 已开"; A=1
else echo "  方案A(PeerMappingOverride) : ❌ 未开"; A=0; fi
if lsmod | grep -q '^gdrdrv' && [ -e /dev/gdrdrv ]; then echo "  方案B(gdrcopy/gdrdrv)      : ✅ 已装"; B=1
else echo "  方案B(gdrcopy/gdrdrv)      : ❌ 未装 (lsmod:$(lsmod|grep -c '^gdrdrv') /dev/gdrdrv:$([ -e /dev/gdrdrv ] && echo yes || echo no))"; B=0; fi
echo "  模块支持该参数 : $($MODINFO -p nvidia 2>/dev/null | grep -c NVreg_RegistryDwords) 处"
echo "  热重载可行性   : DM=$(systemctl is-active display-manager 2>/dev/null) \
persistenced=$(systemctl is-active nvidia-persistenced 2>/dev/null) \
holders=[$(ls /sys/module/nvidia/holders/ 2>/dev/null | tr '\n' ' ')] \
CUDA进程=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)"
[ $((A+B)) -gt 0 ] && echo "  ==> 闸门: 通过" || echo "  ==> 闸门: 未过 (M5 冻结)"
PROBE
}

echo "=============== gpu-02 (本机, rank0) ==============="
bash -c "$(probe)"
echo "=============== gpu-01 (远端, rank1) ==============="
ssh -o BatchMode=yes "$GPU01_SSH" "bash -s" <<< "$(probe)"
