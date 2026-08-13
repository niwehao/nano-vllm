#!/usr/bin/env bash
# scripts/enable_ibgda.sh —— 开启 IBGDA 方案 A（PeerMappingOverride），**不需要重启**。
#
#   sudo ./scripts/enable_ibgda.sh            # 只检查前置条件，不动任何东西
#   sudo ./scripts/enable_ibgda.sh --apply    # 真的做
#   sudo ./scripts/enable_ibgda.sh --revert   # 撤销
#
# 原理：PeerMappingOverride 是 nvidia 内核模块的加载期参数（NVreg_RegistryDwords），
# 不能运行时改，但**可以卸载模块再带参数装回去** —— 前提是没人在用 GPU。
# 本机实测：无显示服务、无 CUDA 进程、nvidia 的 holders 只有 modeset/uvm，
# 唯一占用是 nvidia-persistenced，停掉即可。所以不用重启。
#
# 它解禁的是什么：IBGDA 要让 CUDA kernel 自己写网卡的门铃寄存器，就得把网卡的
# BAR(MMIO) 映射进 GPU 地址空间。NVIDIA 驱动默认禁止把别的 PCIe 设备的 BAR 映射给
# GPU，PeerMappingOverride=1 就是放行这件事。
#
# 风险与可逆性：卸载驱动期间 GPU 不可用（秒级）；配置写在 /etc/modprobe.d/，
# --revert 删掉再重载即可完全还原。两台机器都要各跑一遍。

set -uo pipefail
CONF=/etc/modprobe.d/nvidia-ibgda.conf
MODINFO=$(command -v modinfo || echo /sbin/modinfo)
MODE=${1:-}

need_root() { [ "$(id -u)" = 0 ] || { echo "要 root：sudo $0 $MODE"; exit 1; }; }

echo "=== 前置检查 ==="
ok=1
p=$($MODINFO -p nvidia 2>/dev/null | grep -c NVreg_RegistryDwords)
echo "  模块认 NVreg_RegistryDwords : $([ "$p" -gt 0 ] && echo 是 || { echo 否; ok=0; })"
dm=$(systemctl is-active display-manager 2>/dev/null)
echo "  显示服务                    : $dm $([ "$dm" = active ] && { echo '← 得先停，否则 rmmod 会失败'; ok=0; })"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
echo "  在跑的 CUDA 进程            : $n $([ "$n" -gt 0 ] && { echo '← 得先停'; ok=0; })"
echo "  nvidia 模块 holders         : [$(ls /sys/module/nvidia/holders/ 2>/dev/null | tr '\n' ' ')]"
echo "  nvidia-persistenced         : $(systemctl is-active nvidia-persistenced 2>/dev/null)（脚本会自动停/起）"
echo "  当前 RegistryDwords         : \"$(grep -oP 'RegistryDwords:\s*"\K[^"]*' /proc/driver/nvidia/params 2>/dev/null)\""
[ "$ok" = 1 ] && echo "  ==> 可以热重载" || { echo "  ==> 有阻塞项，先处理"; [ "$MODE" = --apply ] && exit 1; }

reload() {
  systemctl stop nvidia-persistenced 2>/dev/null
  sleep 1
  # 顺序要紧：先卸依赖 nvidia 的，最后卸 nvidia 自己
  for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do
    lsmod | grep -q "^$m " && { rmmod $m || { echo "rmmod $m 失败，查占用：fuser -v /dev/nvidia*"; return 1; }; }
  done
  modprobe nvidia && modprobe nvidia_uvm
  modprobe nvidia_drm 2>/dev/null       # headless 用不到，装回去只为还原原状态
  systemctl start nvidia-persistenced 2>/dev/null
  nvidia-smi -L
}

case "$MODE" in
  --apply)
    need_root
    echo "=== 写配置 $CONF ==="
    cat > "$CONF" <<'EOF'
# IBGDA(GPUDirect Async kernel-initiated)所需：允许把 NIC 的 BAR 映射进 GPU 地址空间，
# 让 CUDA kernel 能直接写网卡门铃。由 nano-vllm/scripts/enable_ibgda.sh 写入。
options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;" NVreg_EnableStreamMemOPs=1
EOF
    cat "$CONF"
    echo "=== 热重载驱动 ==="
    reload || exit 1
    # 让**下次重启**也生效。仅当 nvidia 真的被打进 initramfs 时才动它——这一步会重建
    # 引导镜像，失败了是要影响开机的，所以不吞错误、结果打出来给人看。
    # 本机是 headless 无早期 KMS，通常 initramfs 里没有 nvidia，跳过即可。
    if command -v lsinitramfs >/dev/null && lsinitramfs /boot/initrd.img-$(uname -r) 2>/dev/null | grep -q 'nvidia\.ko'; then
      echo "  nvidia 在 initramfs 里，重建以便下次开机生效："
      update-initramfs -u || echo "  ⚠ update-initramfs 失败——本次热加载不受影响，但**重启后会失效**，请先修好再重启"
    else
      echo "  nvidia 不在 initramfs 里（headless 无早期 KMS），无需重建；"
      echo "  $CONF 会在开机时由 udev 加载模块时读到。"
    fi
    echo "=== 结果 ==="
    grep RegistryDwords /proc/driver/nvidia/params
    grep -q 'PeerMappingOverride=1' /proc/driver/nvidia/params \
      && echo "✅ 已生效" || { echo "❌ 没生效，检查 $CONF 与 dmesg"; exit 1; }
    ;;
  --revert)
    need_root
    rm -f "$CONF"
    if command -v lsinitramfs >/dev/null && lsinitramfs /boot/initrd.img-$(uname -r) 2>/dev/null | grep -q 'nvidia\.ko'; then
      update-initramfs -u || echo "  ⚠ update-initramfs 失败"
    fi
    reload || exit 1
    grep RegistryDwords /proc/driver/nvidia/params
    echo "已还原"
    ;;
  *)
    echo
    echo "只做了检查。要真的执行：sudo $0 --apply   （撤销：sudo $0 --revert）"
    echo "两台机器都要跑。之后用 scripts/m0_ibgda_check.sh 复核。"
    ;;
esac
