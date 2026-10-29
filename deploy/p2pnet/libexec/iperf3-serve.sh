#!/usr/bin/env bash
# One iperf3 listener per overlay address of this node: the wg0 overlay IPv4 and, on dual-WAN nodes, the plane-2
# IPv4 on wg1. A dual-WAN server must answer MP_JOIN on its plane-2 address too; with only the wg0 address bound a
# single-WAN client's second subflow is refused (ECONNREFUSED) and the pair never shows two paths. Every address is
# bound explicitly, so nothing listens on a WAN NIC: no wider than the single-address listener it replaces.
set -Eeuo pipefail

serve() {
  # Retry: the plane-2 address only exists once `p2pnet mptcp install` has brought wg1 up.
  while :; do
    /opt/p2pnet/libexec/mptcp-exec /usr/bin/iperf3 -s -B "$1" -p 5201 || true
    sleep 2
  done
}

serve "${P2P_OVL4:?P2P_OVL4 is not set (EnvironmentFile=/etc/p2pnet/generated/node.env)}" &
[[ -z ${P2P_PLANE2_IP:-} ]] || serve "$P2P_PLANE2_IP" &
wait
