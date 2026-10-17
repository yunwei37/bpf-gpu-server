#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# bpf-gpu-server over a real netdevice (veth pair across two netns) instead of `lo`.
#
# The two netns force every datagram through the veth driver + the tc/clsact
# ingress hook on bgsv1 — the same hook a NIC's RX path would hit. It is still a
# virtual device (no hardware DMA/RX ring), but it is not the loopback shortcut
# where the packet never leaves the device driver in an skb form.
#
#   ./bench/run_veth.sh [seconds]
set -eu
SECS=${1:-10}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/bench/results/veth_$(date -u +%Y%m%dT%H%M%SZ).txt"
A=bgsns1; B=bgsns2; AIP=10.99.0.1; BIP=10.99.0.2

cd "$ROOT"
cleanup() {
  pkill -x executor 2>/dev/null || true
  ip netns del $A 2>/dev/null || true
  ip netns del $B 2>/dev/null || true
}
trap cleanup EXIT

cleanup
ip netns add $A; ip netns add $B
ip link add bgsv0 type veth peer name bgsv1
ip link set bgsv0 netns $A; ip link set bgsv1 netns $B
ip -n $A addr add $AIP/24 dev bgsv0; ip -n $A link set bgsv0 up; ip -n $A link set lo up
ip -n $B addr add $BIP/24 dev bgsv1; ip -n $B link set bgsv1 up; ip -n $B link set lo up

ip netns exec $B bash -c '
  mount -t bpf bpf /sys/fs/bpf 2>/dev/null || true
  ./build/bpf_gpu_server_load attach bgsv1 >/dev/null
  BGS_BIND='"$BIP"' ./build/executor '"$SECS"' 4 8 6 7 0 >/tmp/bgs_veth_exec.log 2>&1 &
  sleep 3
  {
    echo "=== bpf-gpu-server over veth ($A -> $B via bgsv1 tc ingress), $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    ip netns exec '"$A"' env BGS_DEST='"$BIP"' ./build/client verify 0 0 39400 4
    ip netns exec '"$A"' env BGS_DEST='"$BIP"' ./build/client own 2000 0 39400 4
    ip netns exec '"$A"' env BGS_DEST='"$BIP"' ./build/client burst 400 8 39400 4
    echo
    ./build/bpf_gpu_server_load stats
    echo
    echo "=== executor ==="
    cat /tmp/bgs_veth_exec.log
  }
' | tee "$OUT"

echo "saved $OUT"
