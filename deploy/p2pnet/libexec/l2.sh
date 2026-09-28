#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck disable=SC2034
# shellcheck disable=SC1091
source /opt/p2pnet/libexec/common.sh

l2_peer_fields() {
    inv peers --node "$P2P_NODE"
}

l2_vxlan_mismatch() {
    /usr/bin/python3 - "$P2P_OVL4" "$P2P_VXLAN_VNI" "$P2P_VXLAN_PORT" <<'PY'
import json, subprocess, sys
local, vni, port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    raw = subprocess.check_output(['ip', '-d', '-j', 'link', 'show', 'dev', 'vxlan0'], text=True)
    link = json.loads(raw)[0]
    data = link.get('linkinfo', {}).get('info_data', {})
    # iproute2 6.x JSON names the UDP port "port"; older releases used "dstport".
    bad = link.get('linkinfo', {}).get('info_kind') != 'vxlan' or int(data.get('id', -1)) != vni or data.get('local') != local or int(data.get('port', data.get('dstport', -1))) != port
except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError, json.JSONDecodeError):
    bad = True
sys.exit(0 if bad else 1)
PY
}

l2_fdb_sets() {
    /usr/bin/python3 - <<'PY'
import json, subprocess
entries = json.loads(subprocess.check_output(['bridge', '-j', 'fdb', 'show', 'dev', 'vxlan0'], text=True))
print('\n'.join(sorted({e.get('dst', '') for e in entries if e.get('mac', '').lower() == '00:00:00:00:00:00' and e.get('dst')})))
PY
}

l2_apply() {
    require_root
    load_env
    need_cmd ip bridge
    if ! ip link show dev "$P2P_BRIDGE" >/dev/null 2>&1; then
        CHANGED=$((CHANGED+1))
        ip link add name "$P2P_BRIDGE" type bridge
    elif ! ip -d -j link show dev "$P2P_BRIDGE" | /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin)[0]; sys.exit(0 if d.get("linkinfo",{}).get("info_kind")=="bridge" else 1)'; then
        die "$P2P_BRIDGE exists and is not a bridge"
    fi
    local bridge_mtu bridge_flags vxlan_mtu vxlan_flags vxlan_master
    bridge_mtu=$(ip -o link show dev "$P2P_BRIDGE" | sed -n 's/.* mtu \([0-9][0-9]*\).*/\1/p')
    bridge_flags=$(ip -o link show dev "$P2P_BRIDGE" | sed -n 's/.*<\([^>]*\)>.*/\1/p')
    if [[ $bridge_mtu != "$P2P_VXLAN_MTU" || ,$bridge_flags, != *,UP,* ]]; then CHANGED=$((CHANGED+1)); fi
    ip link set dev "$P2P_BRIDGE" mtu "$P2P_VXLAN_MTU"
    ip link set dev "$P2P_BRIDGE" type bridge stp_state 0
    ip link set dev "$P2P_BRIDGE" up
    local vxlan_changed=0
    if ip link show dev vxlan0 >/dev/null 2>&1; then
        if l2_vxlan_mismatch; then
            vxlan_changed=1
            ip link del dev vxlan0
        else
            vxlan_mtu=$(ip -o link show dev vxlan0 | sed -n 's/.* mtu \([0-9][0-9]*\).*/\1/p')
            vxlan_flags=$(ip -o link show dev vxlan0 | sed -n 's/.*<\([^>]*\)>.*/\1/p')
            vxlan_master=$(ip -o link show dev vxlan0 | sed -n 's/.* master \([^ ]*\).*/\1/p')
            [[ $vxlan_mtu == "$P2P_VXLAN_MTU" && ,$vxlan_flags, == *,UP,* && $vxlan_master == "$P2P_BRIDGE" ]] || vxlan_changed=1
        fi
    else
        vxlan_changed=1
    fi
    (( vxlan_changed == 0 )) || CHANGED=$((CHANGED+1))
    if ! ip link show dev vxlan0 >/dev/null 2>&1; then
        ip link add vxlan0 type vxlan id "$P2P_VXLAN_VNI" local "$P2P_OVL4" dstport "$P2P_VXLAN_PORT" ttl 64
    fi
    ip link set dev vxlan0 mtu "$P2P_VXLAN_MTU"
    ip link set dev vxlan0 master "$P2P_BRIDGE"
    ip link set dev vxlan0 up
    local name ovl4 current desired ip
    desired=$(while IFS=$'\t' read -r name ovl4 _ _ _ _ _ _ _ _ _ _; do [[ -z $name ]] || printf '%s\n' "$ovl4"; done < <(l2_peer_fields) | sort -u)
    current=$(l2_fdb_sets || true)
    while IFS= read -r ip; do
        [[ -z $ip ]] && continue
        if ! grep -Fxq "$ip" <<<"$current"; then CHANGED=$((CHANGED+1)); bridge fdb append 00:00:00:00:00:00 dev vxlan0 dst "$ip"; fi
    done <<<"$desired"
    while IFS= read -r ip; do
        [[ -z $ip ]] && continue
        if ! grep -Fxq "$ip" <<<"$desired"; then CHANGED=$((CHANGED+1)); bridge fdb del 00:00:00:00:00:00 dev vxlan0 dst "$ip"; fi
    done <<<"$current"
}

l2_install() {
    require_root
    load_env
    l2_apply
    install_unit /opt/p2pnet/systemd/p2pnet-l2.service
    reload_units
    systemctl enable --now p2pnet-l2.service
    mark_installed l2
    printf 'p2pnet[l2] install complete: changed=%s\n' "$CHANGED"
}

l2_check_fdb() {
    local desired current name ovl4
    desired=$(while IFS=$'\t' read -r name ovl4 _ _ _ _ _ _ _ _ _ _; do [[ -z $name ]] || printf '%s\n' "$ovl4"; done < <(l2_peer_fields) | sort -u)
    current=$(l2_fdb_sets 2>/dev/null || true)
    [[ $desired == "$current" ]]
}

l2_verify() {
    load_env
    if ! is_installed l2; then check SKIP l2 "not installed"; verify_done "${P2P_VERIFY_FAILED:-0}"; return; fi
    need_cmd ip bridge
    local bridge_mtu vxlan_mtu
    if ip -o link show dev "$P2P_BRIDGE" 2>/dev/null | grep -q '<[^>]*UP'; then check PASS l2.bridge "${P2P_BRIDGE} up"; else check FAIL l2.bridge "${P2P_BRIDGE} missing or down"; fi
    if ip -o link show dev vxlan0 2>/dev/null | grep -q '<[^>]*UP'; then check PASS l2.vxlan "vxlan0 up"; else check FAIL l2.vxlan "vxlan0 missing or down"; fi
    bridge_mtu=$(ip -o link show dev "$P2P_BRIDGE" 2>/dev/null | sed -n 's/.* mtu \([0-9][0-9]*\).*/\1/p')
    vxlan_mtu=$(ip -o link show dev vxlan0 2>/dev/null | sed -n 's/.* mtu \([0-9][0-9]*\).*/\1/p')
    if [[ $bridge_mtu == "$P2P_VXLAN_MTU" ]]; then check PASS l2.bridge_mtu "$bridge_mtu"; else check FAIL l2.bridge_mtu "actual=${bridge_mtu:-missing} expected=$P2P_VXLAN_MTU"; fi
    if [[ $vxlan_mtu == "$P2P_VXLAN_MTU" ]]; then check PASS l2.vxlan_mtu "$vxlan_mtu"; else check FAIL l2.vxlan_mtu "actual=${vxlan_mtu:-missing} expected=$P2P_VXLAN_MTU"; fi
    if l2_check_fdb; then check PASS l2.fdb "flood list matches inventory"; else check FAIL l2.fdb "flood list differs from inventory"; fi
    if [[ -r /proc/sys/net/bridge/bridge-nf-call-iptables ]] && [[ $(< /proc/sys/net/bridge/bridge-nf-call-iptables) == 1 ]]; then
        check WARN l2.bridge_nf "bridged traffic may be dropped; remediation: iptables -I DOCKER-USER -i $P2P_BRIDGE -o $P2P_BRIDGE -j ACCEPT"
    fi
    verify_done "${P2P_VERIFY_FAILED:-0}"
}

l2_cleanup_selftest() {
    ip netns del p2pnet-st 2>/dev/null || true
    ip link del p2pst-h 2>/dev/null || true
}

l2_selftest() {
    load_env
    require_root
    need_cmd ip ping
    local peer='' wait_s=60 arg name peer_index target_ip
    while (($#)); do
        arg=$1; shift
        case $arg in
            --peer) (($#)) || die 'usage: l2 selftest --peer NODE [--wait 60]'; peer=$1; shift ;;
            --wait) (($#)) || die 'usage: l2 selftest --peer NODE [--wait 60]'; wait_s=$1; shift ;;
            *) die 'usage: l2 selftest --peer NODE [--wait 60]' ;;
        esac
    done
    [[ -n $peer && $wait_s =~ ^[0-9]+$ ]] || die 'usage: l2 selftest --peer NODE [--wait 60]'
    peer_index=$(while IFS=$'\t' read -r name _ _ _ _ _ _ peer_index _ _ _ _; do [[ $name == "$peer" ]] && { printf '%s' "$peer_index"; break; }; done < <(l2_peer_fields))
    [[ $peer_index =~ ^[0-9]+$ ]] || die "unknown peer $peer"
    target_ip="169.254.77.$((peer_index+1))"
    l2_cleanup_selftest
    trap l2_cleanup_selftest EXIT
    ip netns add p2pnet-st
    ip link add p2pst-h type veth peer name p2pst-n
    ip link set p2pst-h mtu "$P2P_VXLAN_MTU"
    ip link set p2pst-n mtu "$P2P_VXLAN_MTU"
    ip link set p2pst-h master "$P2P_BRIDGE"
    ip link set p2pst-h up
    ip link set p2pst-n netns p2pnet-st
    ip -n p2pnet-st link set lo up
    ip -n p2pnet-st link set p2pst-n up
    ip -n p2pnet-st addr add "169.254.77.$((P2P_NODE_INDEX+1))/24" dev p2pst-n
    local deadline=$((SECONDS+wait_s)) reached=0
    while (( SECONDS <= deadline )); do
        if ip netns exec p2pnet-st ping -c 1 -W 1 "$target_ip" >/dev/null 2>&1; then reached=1; break; fi
        sleep 1
    done
    (( reached )) || die "peer $peer did not answer selftest ping within ${wait_s}s"
# shellcheck disable=SC1010
    ip netns exec p2pnet-st ping -M do -s "$((P2P_VXLAN_MTU-28))" -c 2 -W 2 "$target_ip" >/dev/null || die "peer $peer failed DF MTU ping"
    printf 'PASS l2.selftest peer=%s test_ip=%s mtu=%s\n' "$peer" "$target_ip" "$P2P_VXLAN_MTU"
    trap - EXIT
    l2_cleanup_selftest
}

l2_uninstall() {
    require_root
    load_env
    need_cmd ip bridge
    local force=0 arg members
    for arg in "$@"; do
        case $arg in --force) force=1 ;; --purge) ;; *) die 'usage: l2 uninstall [--force|--purge]' ;; esac
    done
    if ip link show dev "$P2P_BRIDGE" >/dev/null 2>&1; then
        members=$(ip -o link show master "$P2P_BRIDGE" | /usr/bin/python3 -c 'import re,sys; print(" ".join(x for x in re.findall(r": ([^:@]+)(?:@[^:]+)?:",sys.stdin.read()) if x != "vxlan0"))')
        [[ -z $members || $force == 1 ]] || die "refusing to remove $P2P_BRIDGE with member ports: $members (use --force)"
    fi
    systemctl disable --now p2pnet-l2.service 2>/dev/null || true
    ip link del dev vxlan0 2>/dev/null || true
    ip link del dev "$P2P_BRIDGE" 2>/dev/null || true
    rm -f /etc/systemd/system/p2pnet-l2.service
    export UNITS_CHANGED=1
    reload_units
    mark_uninstalled l2
}

case "${1:-}" in
    install) l2_install ;;
    apply) shift; (($# == 0)) || die 'usage: p2pnet l2 apply'; l2_apply ;;
    verify) l2_verify ;;
    selftest) shift; l2_selftest "$@" ;;
    uninstall) shift; l2_uninstall "$@" ;;
    *) die 'usage: p2pnet l2 {install|apply|verify|selftest|uninstall [--force]}' ;;
esac
