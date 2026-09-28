#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck disable=SC2034
# Executed by bin/p2pnet after component dispatch.
# shellcheck disable=SC1091
source /opt/p2pnet/libexec/common.sh


wg_render() {
    inv render-wg0 --node "$(self_node)"
}

wg_install() {
    require_root
    load_env
    need_cmd wg systemctl
    local key=/etc/wireguard/wg0.key pub expected active=0 before_config=$CHANGED config_changed=0 config_tmp
    [[ -r $key ]] || die "missing $key — run p2pnet keys"
    pub=$(wg pubkey < "$key")
    expected=$P2P_WG_PUBKEY
    [[ -n $expected && $pub == "$expected" ]] || die "wg0 key mismatch — run p2pnet keys and update the inventory"
    config_tmp=$(mktemp)
    if ! wg_render >"$config_tmp"; then rm -f "$config_tmp"; return 1; fi
    install_file "$config_tmp" /etc/wireguard/wg0.conf 0600
    rm -f "$config_tmp"
    (( CHANGED == before_config )) || config_changed=1
    install_unit /opt/p2pnet/systemd/p2pnet-wg-reresolve.service
    install_unit /opt/p2pnet/systemd/p2pnet-wg-reresolve.timer
    reload_units
    if systemctl is-active --quiet wg-quick@wg0.service; then
        active=1
    fi
    if (( active )); then
        if (( config_changed )); then
            systemctl restart wg-quick@wg0.service
            CHANGED=$((CHANGED+1))
        fi
    else
        systemctl enable --now wg-quick@wg0.service
        CHANGED=$((CHANGED+1))
    fi
    if ! systemctl is-enabled --quiet p2pnet-wg-reresolve.timer; then CHANGED=$((CHANGED+1)); fi
    systemctl enable --now p2pnet-wg-reresolve.timer
    if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q '^Status: active'; then
        log WARN "ufw is active; allow ${P2P_WG_PORT}/udp and traffic in on wg0 (p2pnet does not modify firewalls)"
    fi
    mark_installed wg
    printf 'p2pnet[wg] install complete: changed=%s\n' "${CHANGED:-0}"
}

wg_reresolve() {
    load_env
    need_cmd wg
    local name endpoint pub latest current host port
    while IFS=$'\t' read -r name _ _ endpoint pub _ _ _ _ _ _ _; do
        [[ -n $name ]] || continue
        [[ $endpoint != - && -n $endpoint ]] || continue
        # Split host:port, preserving bracketed IPv6 literals.
        if [[ $endpoint == \[*\]:* ]]; then
            host=${endpoint#\[}; host=${host%%\]*}; port=${endpoint##*:}
            [[ $host == *:* ]] && continue
        else
            host=${endpoint%:*}; port=${endpoint##*:}
            [[ -n $host && $host != *:* ]] || continue
        fi
        [[ $host =~ [A-Za-z] ]] || continue
        latest=$(wg show wg0 latest-handshakes 2>/dev/null | awk -v k="$pub" '$1 == k {print $2; exit}')
        current=$(wg show wg0 endpoints 2>/dev/null | awk -v k="$pub" '$1 == k {$1=""; sub(/^ /,""); print; exit}')
        if [[ ! $latest =~ ^[0-9]+$ ]] || (( latest == 0 || $(date +%s) - latest > 135 )); then
            wg set wg0 peer "$pub" endpoint "$host:$port"
        fi
    done < <(inv peers --node "$P2P_NODE")
}

wg_verify() {
    load_env
    if ! is_installed wg; then check SKIP wg "not installed"; verify_done "${P2P_VERIFY_FAILED:-0}"; return; fi
    need_cmd wg ip ping
    local mtu name ovl4 endpoint pub latest current first_peer='' now age
    if systemctl is-active --quiet wg-quick@wg0.service; then check PASS wg.unit "wg-quick@wg0 active"; else check FAIL wg.unit "wg-quick@wg0 inactive"; fi
    mtu=$(ip -o link show dev wg0 2>/dev/null | sed -n 's/.* mtu \([0-9][0-9]*\).*/\1/p')
    if [[ $mtu == "$P2P_WG_MTU" ]]; then check PASS wg.mtu "$mtu"; else check FAIL wg.mtu "actual=${mtu:-missing} expected=$P2P_WG_MTU"; fi
    now=$(date +%s)
    while IFS=$'\t' read -r name ovl4 _ endpoint pub _ _ _ _ _ _ _; do
        [[ -n $name ]] || continue
        [[ -n $first_peer ]] || first_peer=$ovl4
        latest=$(wg show wg0 latest-handshakes 2>/dev/null | awk -v k="$pub" '$1 == k {print $2; exit}')
        if [[ $latest =~ ^[0-9]+$ ]] && (( latest > 0 )); then
            age=$((now-latest))
            if (( age <= 180 )); then check PASS "wg.handshake.$name" "${age}s ago"; else check FAIL "wg.handshake.$name" "stale ${age}s"; fi
        else
            check FAIL "wg.handshake.$name" "never handshaken"
        fi
        current=$(wg show wg0 endpoints 2>/dev/null | awk -v k="$pub" '$1 == k {$1=""; sub(/^ /,""); print; exit}')
        if [[ ( $endpoint =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+$ || $endpoint =~ ^\[[0-9A-Fa-f:.]+\]:[0-9]+$ ) && -n $current && $current != "$endpoint" ]]; then
            check WARN "wg.endpoint.$name" "current=$current inventory=$endpoint"
        fi
        if ping -c 3 -W 2 "$ovl4" >/dev/null 2>&1; then check PASS "wg.ping.$name" "$ovl4 replies"; else check FAIL "wg.ping.$name" "$ovl4 unreachable"; fi
    done < <(inv peers --node "$P2P_NODE")
    first_peer=''
    while IFS=$'\t' read -r name ovl4 _ _ _ _ _ _ _ _ _ _; do
        [[ -n $name ]] || continue
        if ping -c 1 -W 2 "$ovl4" >/dev/null 2>&1; then first_peer=$ovl4; break; fi
    done < <(inv peers --node "$P2P_NODE")
    if [[ -n $first_peer ]]; then
# shellcheck disable=SC1010
        if ping -M do -s "$((P2P_WG_MTU-28))" -c 1 -W 2 "$first_peer" >/dev/null 2>&1; then check PASS wg.mtu_probe "DF payload=$((P2P_WG_MTU-28))"; else check FAIL wg.mtu_probe "DF probe failed at MTU $P2P_WG_MTU"; fi
    else
        check SKIP wg.mtu_probe "no reachable peer for DF probe"
    fi
    verify_done "${P2P_VERIFY_FAILED:-0}"
}

wg_uninstall() {
    require_root
    local purge=0 arg
    for arg in "$@"; do
        if [[ $arg == --purge ]]; then purge=1; else die "usage: wg uninstall [--purge]"; fi
    done
    systemctl disable --now wg-quick@wg0.service p2pnet-wg-reresolve.timer 2>/dev/null || true
    systemctl stop p2pnet-wg-reresolve.service 2>/dev/null || true
    rm -f /etc/wireguard/wg0.conf /etc/systemd/system/p2pnet-wg-reresolve.service /etc/systemd/system/p2pnet-wg-reresolve.timer
    export UNITS_CHANGED=1
    (( purge == 0 )) || rm -f /etc/wireguard/wg0.key /etc/wireguard/wg0.pub
    reload_units
    mark_uninstalled wg
}

case "${1:-}" in
    install) wg_install ;;
    verify) wg_verify ;;
    uninstall) shift; wg_uninstall "$@" ;;
    reresolve) wg_reresolve ;;
    *) die "usage: p2pnet wg {install|verify|uninstall [--purge]|reresolve}" ;;
esac
