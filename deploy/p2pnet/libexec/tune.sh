#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck disable=SC1091
source /opt/p2pnet/libexec/common.sh

TUNE_COMPONENT=tune
TUNE_SYSCTL_CONF=/etc/sysctl.d/90-p2pnet-net.conf
TUNE_MODULE_CONF=/etc/modules-load.d/p2pnet.conf
TUNE_KEYS=(
  net.core.default_qdisc
  net.ipv4.tcp_congestion_control
  net.core.rmem_max
  net.core.wmem_max
  net.ipv4.tcp_rmem
  net.ipv4.tcp_wmem
  net.ipv4.tcp_notsent_lowat
  net.ipv4.tcp_mtu_probing
  net.ipv4.tcp_slow_start_after_idle
  net.core.netdev_max_backlog
)
TUNE_MODULES=(tcp_bbr sch_fq sch_fq_codel sch_htb cls_u32 vxlan bridge)

_tune_usage() {
  printf 'usage: p2pnet tune {install|verify|uninstall|apply-qdisc}\n' >&2
  exit 2
}

_tune_write_configs() {
  install_file "$P2P_PREFIX/conf/90-p2pnet-net.conf" "$TUNE_SYSCTL_CONF" 0644
  install_file "$P2P_PREFIX/conf/modules-load-p2pnet.conf" "$TUNE_MODULE_CONF" 0644
}

_tune_qdisc_for() {
  local dev=$1 queues state
  [[ -d "/sys/class/net/$dev" ]] || die "WAN interface does not exist: $dev"
  queues=$(find "/sys/class/net/$dev/queues" -maxdepth 1 -type d -name 'tx-*' -print 2>/dev/null | wc -l)
  state=$(tc -j qdisc show dev "$dev")
  if (( queues > 1 )); then
    if jq -e 'any(.[]; .root == true and .kind == "mq")' <<<"$state" >/dev/null &&
      jq -e 'all(.[] | select(.parent? != null); .kind == "fq")' <<<"$state" >/dev/null; then
      return 0
    fi
    tc qdisc replace dev "$dev" root mq
  else
    if jq -e 'any(.[]; .root == true and .kind == "fq")' <<<"$state" >/dev/null; then
      return 0
    fi
    tc qdisc replace dev "$dev" root fq
  fi
  CHANGED=$((CHANGED + 1))
}

tune_apply_qdisc() {
  load_env
  need_cmd tc jq
  local dev
  for dev in "${P2P_WAN_IF:-}" "${P2P_WAN2_IF:-}"; do
    [[ -n $dev ]] || continue
    _tune_qdisc_for "$dev"
  done
}

_tune_install() {
  require_root
  load_env
  need_cmd sysctl modprobe tc systemctl jq
  local key value module desired apply_conf
  declare -A before
  for key in "${TUNE_KEYS[@]}"; do
    value=$(sysctl -n "$key" 2>/dev/null) || die "cannot read current sysctl: $key"
    before["$key"]=$value
    save_prev_once "$TUNE_COMPONENT" "$key" "$value"
  done
  _tune_write_configs
  for module in "${TUNE_MODULES[@]}"; do
    modprobe "$module" || die "required kernel module unavailable: $module"
  done
  apply_conf=$(mktemp)
  sed -E 's/[[:space:]]+#.*$//' "$TUNE_SYSCTL_CONF" >"$apply_conf"
  if sysctl -p "$apply_conf"; then rm -f "$apply_conf"; else rm -f "$apply_conf"; die "failed to apply sysctl settings from $TUNE_SYSCTL_CONF"; fi
  for key in "${TUNE_KEYS[@]}"; do
    desired=$(sysctl -n "$key")
    [[ ${before[$key]} == "$desired" ]] || CHANGED=$((CHANGED + 1))
  done
  tune_apply_qdisc
  install_unit "$P2P_PREFIX/systemd/p2pnet-tune.service"
  reload_units
  if ! systemctl is-enabled --quiet p2pnet-tune.service || ! systemctl is-active --quiet p2pnet-tune.service; then CHANGED=$((CHANGED + 1)); fi
  systemctl enable --now p2pnet-tune.service
  mark_installed "$TUNE_COMPONENT"
  printf 'p2pnet[tune] install complete: changed=%s\n' "$CHANGED"
}


_tune_verify() {
  if ! is_installed "$TUNE_COMPONENT"; then
    printf 'SKIP tune not installed\n'
    return 0
  fi
  load_env
  local key expected actual status qdisc dev bad=0
  for key in "${TUNE_KEYS[@]}"; do
    expected=$(sed -nE "s/^[[:space:]]*${key//./\\.}[[:space:]]*=[[:space:]]*([^#]+).*/\\1/p" "$TUNE_SYSCTL_CONF" | xargs)
    actual=$(sysctl -n "$key" 2>/dev/null | xargs || true)
    if [[ $actual == "$expected" ]]; then status=PASS; else status=FAIL; bad=1; fi
    check "$status" "sysctl.$key" "actual='$actual' expected='$expected'"
  done
  if grep -qw bbr /proc/sys/net/ipv4/tcp_available_congestion_control; then
    check PASS tune.bbr 'bbr available'
  else
    check FAIL tune.bbr 'bbr unavailable'; bad=1
  fi
  for dev in "${P2P_WAN_IF:-}" "${P2P_WAN2_IF:-}"; do
    [[ -n $dev ]] || continue
    qdisc=$(tc -j qdisc show dev "$dev" 2>/dev/null || true)
    if [[ ! -d "/sys/class/net/$dev" ]]; then
      check FAIL "qdisc.$dev" 'WAN interface missing'; bad=1
    elif (( $(find "/sys/class/net/$dev/queues" -maxdepth 1 -type d -name 'tx-*' -print 2>/dev/null | wc -l) > 1 )); then
      if jq -e 'any(.[]; .root == true and .kind == "mq")' <<<"$qdisc" >/dev/null && jq -e 'all(.[] | select(.parent? != null); .kind == "fq")' <<<"$qdisc" >/dev/null; then
        check PASS "qdisc.$dev" 'mq root with fq children'
      else check FAIL "qdisc.$dev" 'expected mq root with fq children'; bad=1; fi
    elif jq -e 'any(.[]; .root == true and .kind == "fq")' <<<"$qdisc" >/dev/null; then
      check PASS "qdisc.$dev" 'fq root'
    else check FAIL "qdisc.$dev" 'expected fq root'; bad=1; fi
  done
  (( bad == 0 )) || return 1
}

_tune_restore_sysctl() {
  sysctl -w "$1=$2"
}

_tune_uninstall() {
  case ${1:-} in
    ''|--purge) (($# <= 1)) || _tune_usage ;;
    *) _tune_usage ;;
  esac
  require_root
  load_env
  local dev
  restore_prev "$TUNE_COMPONENT" _tune_restore_sysctl
  for dev in "${P2P_WAN_IF:-}" "${P2P_WAN2_IF:-}"; do
    if [[ -n $dev && -d "/sys/class/net/$dev" ]]; then
      tc qdisc del dev "$dev" root 2>/dev/null || true
    fi
  done
  systemctl disable --now p2pnet-tune.service 2>/dev/null || true
  rm -f "$TUNE_SYSCTL_CONF" "$TUNE_MODULE_CONF" /etc/systemd/system/p2pnet-tune.service
  export UNITS_CHANGED=1
  reload_units
  mark_uninstalled "$TUNE_COMPONENT"
  printf 'p2pnet[tune] uninstall complete\n'
}

tune_main() {
  local action=${1:-}
  [[ $# -gt 0 ]] && shift
  case $action in
    install) (($# == 0)) || _tune_usage; _tune_install ;;
    verify) (($# == 0)) || _tune_usage; _tune_verify ;;
    uninstall) _tune_uninstall "$@" ;;
    apply-qdisc) (($# == 0)) || _tune_usage; tune_apply_qdisc ;;
    *) _tune_usage ;;
  esac
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then tune_main "$@"; fi
