#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck disable=SC1091
source /opt/p2pnet/libexec/common.sh

MPTCP_COMPONENT=mptcp
MPTCP_SYSCTL_CONF=/etc/sysctl.d/91-p2pnet-mptcp.conf
MPTCP_RT_CONF=/etc/iproute2/rt_tables.d/p2pnet.conf
MPTCP_SYSCTL_KEYS=(net.mptcp.enabled net.mptcp.pm_type net.ipv4.conf.all.rp_filter net.ipv4.conf.default.rp_filter)

_mptcp_usage() {
  printf 'usage: p2pnet mptcp {install|apply|verify [--peer NODE]|uninstall [--purge]}\n' >&2
  exit 2
}

_mptcp_dualwan() { [[ -n ${P2P_WAN2_IF:-} ]]; }

_mptcp_install() {
  require_root
  load_env
  need_cmd sysctl ip systemctl
  [[ -r /proc/sys/net/mptcp/enabled ]] || die 'required kernel MPTCP support unavailable: net/mptcp'
  local key current desired tmp routes_before='' rules_before=''
  declare -A before
  for key in "${MPTCP_SYSCTL_KEYS[@]}"; do
    current=$(sysctl -n "$key" 2>/dev/null) || die "required MPTCP sysctl unavailable: $key"
    before["$key"]=$current
    save_prev_once "$MPTCP_COMPONENT" "$key" "$current"
  done
  install_file "$P2P_PREFIX/conf/91-p2pnet-mptcp.conf" "$MPTCP_SYSCTL_CONF" 0644
  tmp=$(mktemp)
  sed -E 's/[[:space:]]+#.*$//' "$MPTCP_SYSCTL_CONF" >"$tmp"
  if sysctl -p "$tmp"; then rm -f "$tmp"; else rm -f "$tmp"; die "failed to apply sysctl settings from $MPTCP_SYSCTL_CONF"; fi
  for key in "${MPTCP_SYSCTL_KEYS[@]}"; do
    desired=$(sysctl -n "$key")
    [[ ${before[$key]} == "$desired" ]] || CHANGED=$((CHANGED + 1))
  done
  if _mptcp_dualwan; then
    install_file - "$MPTCP_RT_CONF" 0644 <<'EOF'
201 p2p-wan1
202 p2p-wan2
203 p2p-plane2
EOF
    tmp=$(mktemp)
    inv render-wg1 --node "$P2P_NODE" >"$tmp"
    install_file "$tmp" /etc/wireguard/wg1.conf 0600
    rm -f "$tmp"
    if ! systemctl is-enabled --quiet wg-quick@wg1.service || ! systemctl is-active --quiet wg-quick@wg1.service; then CHANGED=$((CHANGED + 1)); fi
    systemctl enable --now wg-quick@wg1.service
    routes_before=$(ip -4 route show table all)
    rules_before=$(ip -4 rule show)
  fi
  _mptcp_apply
  if _mptcp_dualwan; then
    [[ $routes_before == "$(ip -4 route show table all)" ]] || CHANGED=$((CHANGED + 1))
    [[ $rules_before == "$(ip -4 rule show)" ]] || CHANGED=$((CHANGED + 1))
  fi
  install_unit "$P2P_PREFIX/systemd/p2pnet-mptcp.service"
  reload_units
  if ! systemctl is-enabled --quiet p2pnet-mptcp.service || ! systemctl is-active --quiet p2pnet-mptcp.service; then CHANGED=$((CHANGED + 1)); fi
  systemctl enable --now p2pnet-mptcp.service
  mark_installed "$MPTCP_COMPONENT"
  printf 'p2pnet[mptcp] install complete: changed=%s\n' "$CHANGED"
}

_mptcp_endpoint_present() {
  ip mptcp endpoint show | grep -Eq "(^|[[:space:]])${P2P_PLANE2_IP//./\\.}([[:space:]]|$).*id 51|id 51.*${P2P_PLANE2_IP//./\\.}"
}
_mptcp_limits_ok() {
  ip mptcp limits show 2>/dev/null | grep -Eq 'subflows[[:space:]]+8.*add_addr_accepted[[:space:]]+8|add_addr_accepted[[:space:]]+8.*subflows[[:space:]]+8'
}
_mptcp_apply() {
  require_root
  load_env
  local endpoint_before=0
  if ! _mptcp_limits_ok; then
    # A single `ip mptcp limits set` can apply subflows but leave
    # add_addr_accepted at 0 when MPTCP sockets from a previous run are still
    # closing, so confirm the result and retry instead of assuming success.
    local attempt
    for attempt in 1 2 3 4 5; do
      ip mptcp limits set subflows 8 add_addr_accepted 8 || true
      _mptcp_limits_ok && break
      sleep 1
    done
    _mptcp_limits_ok || die "could not set MPTCP limits to subflows 8 add_addr_accepted 8 after $attempt attempts (now: $(ip mptcp limits show 2>/dev/null))"
    CHANGED=$((CHANGED + 1))
  fi
  if _mptcp_dualwan; then
    /opt/p2pnet/libexec/plane2-routes up
    if _mptcp_endpoint_present; then endpoint_before=1; fi
    if (( ! endpoint_before )); then
      ip mptcp endpoint delete id 51 2>/dev/null || true
      ip mptcp endpoint add "$P2P_PLANE2_IP" dev wg1 id 51 subflow signal
      CHANGED=$((CHANGED + 1))
    fi
  fi
}

_mptcp_verify() {
  if ! is_installed "$MPTCP_COMPONENT"; then
    printf 'SKIP mptcp not installed\n'
    return 0
  fi
  load_env
  local fail=0 key expected actual limits name pub handshake handshake_rows peer_arg=${1:-}
  for key in "${MPTCP_SYSCTL_KEYS[@]}"; do
    expected=$(sed -nE "s/^[[:space:]]*${key//./\\.}[[:space:]]*=[[:space:]]*([^#]+).*/\\1/p" "$MPTCP_SYSCTL_CONF" | xargs)
    actual=$(sysctl -n "$key" 2>/dev/null || true)
    if [[ $actual == "$expected" ]]; then check PASS "sysctl.$key" "$actual"; else check FAIL "sysctl.$key" "actual='$actual' expected='$expected'"; fail=1; fi
  done
  limits=$(ip mptcp limits show 2>/dev/null || true)
  if grep -Eq 'subflows[[:space:]]+8.*add_addr_accepted[[:space:]]+8|add_addr_accepted[[:space:]]+8.*subflows[[:space:]]+8' <<<"$limits"; then
    check PASS mptcp.limits 'subflows 8 add_addr_accepted 8'
  else check FAIL mptcp.limits "unexpected limits: $limits"; fail=1; fi
  if _mptcp_dualwan; then
    if [[ -s /etc/wireguard/wg1.conf ]] && ip -o link show dev wg1 2>/dev/null | grep -q '<[^>]*UP'; then check PASS wg1.interface 'up'; else check FAIL wg1.interface 'missing or down'; fail=1; fi
    local rule table
    for rule in 10101 10102 10103; do
      if ip -4 rule show | grep -Fq "$rule:"; then check PASS "rule.$rule" present; else check FAIL "rule.$rule" missing; fail=1; fi
    done
    for table in p2p-wan1 p2p-wan2 p2p-plane2; do
      if [[ -n $(ip -4 route show table "$table") ]]; then check PASS "route.$table" present; else check FAIL "route.$table" missing; fail=1; fi
    done
    if _mptcp_endpoint_present; then check PASS mptcp.endpoint 'plane2 id 51 subflow signal'; else check FAIL mptcp.endpoint 'plane2 endpoint id 51 missing'; fail=1; fi
    local fresh=0 now
    now=$(date +%s)
    handshake_rows=$(wg show wg1 latest-handshakes 2>/dev/null || true)
    while IFS=$'\t' read -r name _ _ _ pub _ _ _ _ _ _ _; do
      [[ -n $name ]] || continue
      handshake=$(awk -v key="$pub" '$1 == key {print $2; exit}' <<<"$handshake_rows")
      if [[ $handshake =~ ^[0-9]+$ ]] && (( handshake > 0 && now - handshake <= 180 )); then fresh=$((fresh+1)); else check FAIL "wg1.handshake.$name" "stale or missing ($handshake)"; fail=1; fi
    done < <(inv peers --node "$P2P_NODE")
    if (( fresh > 0 )); then check PASS wg1.handshakes "$fresh fresh"; elif (($(inv peers --node "$P2P_NODE" | wc -l) > 0)); then check FAIL wg1.handshakes 'no fresh plane2 handshakes'; fail=1; fi
    if [[ -n $peer_arg ]]; then _mptcp_peer_verify "$peer_arg" || fail=1; fi
  elif [[ -n $peer_arg ]]; then
    _mptcp_peer_verify "$peer_arg" || fail=1
  fi
  (( fail == 0 )) || return 1
}

_mptcp_peer_verify() {
  local peer=$1 name address plane2 output pid subflows=0 dual=0
  while IFS=$'\t' read -r name address _ _ _ plane2 _ _ _ _ _ _; do
    [[ $name == "$peer" ]] && break
    name=''
  done < <(inv peers --node "$P2P_NODE")
  [[ -n $name ]] || { check FAIL "mptcp.$peer" 'unknown peer'; return 1; }
  if [[ -n ${P2P_WAN2_IF:-} || $plane2 != - ]]; then dual=1; fi
  local tmp
  tmp=$(mktemp)
  /opt/p2pnet/libexec/mptcp-exec /usr/bin/iperf3 -c "$address" -t 5 >"$tmp" 2>&1 &
  pid=$!
  sleep 2
  output=$(ss -Mni dst "$address" 2>/dev/null || true)
  if grep -Eq 'tcp-ulp-mptcp|MPTCP|subflows:' <<<"$output"; then
    subflows=$(sed -nE 's/.*subflows:([0-9]+).*/\1/p' <<<"$output" | head -n1)
    subflows=${subflows:-0}
    if (( dual == 0 || subflows >= 1 )); then
      if (( dual )); then check PASS "mptcp.$peer" "MPTCP socket; subflows=$subflows"; else check PASS "mptcp.$peer" "MPTCP socket; single path (subflows=$subflows)"; fi
    else
      check FAIL "mptcp.$peer" "expected >=2 paths; subflows=$subflows"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      rm -f "$tmp"
      return 1
    fi
  else
    check FAIL "mptcp.$peer" 'no MPTCP socket observed'
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    rm -f "$tmp"
    return 1
  fi
  wait "$pid" || { cat "$tmp" >&2; rm -f "$tmp"; return 1; }
  rm -f "$tmp"
}

_mptcp_restore_sysctl() { sysctl -w "$1=$2"; }

_mptcp_uninstall() {
  case ${1:-} in
    ''|--purge) (($# <= 1)) || _mptcp_usage ;;
    *) _mptcp_usage ;;
  esac
  require_root
  load_env
  need_cmd ip systemctl
  ip mptcp endpoint delete id 51 2>/dev/null || true
  ip mptcp limits set subflows 2 add_addr_accepted 0 2>/dev/null || true
  /opt/p2pnet/libexec/plane2-routes down 2>/dev/null || true
  systemctl disable --now p2pnet-mptcp.service 2>/dev/null || true
  if _mptcp_dualwan; then systemctl disable --now wg-quick@wg1.service 2>/dev/null || true; fi
  restore_prev "$MPTCP_COMPONENT" _mptcp_restore_sysctl
  rm -f "$MPTCP_SYSCTL_CONF" "$MPTCP_RT_CONF" /etc/wireguard/wg1.conf /etc/systemd/system/p2pnet-mptcp.service
  if [[ ${1:-} == --purge ]]; then rm -f /etc/wireguard/wg1.key /etc/wireguard/wg1.pub; fi
  export UNITS_CHANGED=1
  reload_units
  mark_uninstalled "$MPTCP_COMPONENT"
  printf 'p2pnet[mptcp] uninstall complete\n'
}

mptcp_main() {
  local action=${1:-}; (($#)) && shift
  case $action in
    install) (($# == 0)) || _mptcp_usage; _mptcp_install ;;
    apply) (($# == 0)) || _mptcp_usage; _mptcp_apply ;;
    verify) if (($# == 0)); then _mptcp_verify; elif (($# == 2)) && [[ $1 == --peer ]]; then _mptcp_verify "$2"; else _mptcp_usage; fi ;;
    uninstall) _mptcp_uninstall "$@" ;;
    *) _mptcp_usage ;;
  esac
}

mptcp_main "$@"
