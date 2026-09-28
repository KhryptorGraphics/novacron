#!/usr/bin/env bash
set -Eeuo pipefail

if ! declare -F load_env >/dev/null 2>&1; then
  _self_dir=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  # shellcheck source=common.sh
  source "${P2P_COMMON:-${_self_dir}/common.sh}"
fi

_sshd_unit=p2pnet-sshd.service
P2P_COMPONENT=sshd
_sshd_config=/etc/p2pnet/sshd_config
_sshd_keys=/etc/p2pnet/authorized_keys
_sshd_known=/etc/p2pnet/ssh/known_hosts
sshd_usage() { log ERROR "$*"; exit 2; }
_sshd_tmpl=${P2P_PREFIX:-/opt/p2pnet}/conf/sshd_config.tmpl
_sshd_service=${P2P_PREFIX:-/opt/p2pnet}/systemd/p2pnet-sshd.service

sshd_install() {
  require_root
  load_env
  need_cmd sshd systemctl
  ensure_dir /etc/p2pnet/ssh 0755 root:root
  ensure_dir "$_sshd_keys" 0755 root:root
  ensure_dir /var/lib/p2pnet/virt/.ssh 0700 p2pvirt:p2pvirt
  ensure_dir /var/lib/p2pnet/public 0755 root:root
  ensure_dir /var/lib/p2pnet/public/torrents 0755 p2pbulk:p2pbulk
  ensure_dir /var/lib/p2pnet/public/restic 0700 p2pbulk:p2pbulk

  local tempdir rendered
  tempdir=$(mktemp -d)
  SSHD_TMPDIR=$tempdir
  trap 'rm -rf -- "$SSHD_TMPDIR"' EXIT
  rendered=$tempdir/sshd_config
  # shellcheck disable=SC2153
  sed -e "s|@P2P_OVL4@|$P2P_OVL4|g" \
      -e "s|@P2P_OVL6@|${P2P_OVL6:-}|g" \
      -e "s|@P2P_PLANE2_IP@|${P2P_PLANE2_IP:-}|g" "$_sshd_tmpl" > "$rendered"
  if [[ -z ${P2P_OVL6:-} ]]; then sed -i '/^ListenAddress $/d' "$rendered"; fi
  if [[ -z ${P2P_PLANE2_IP:-} ]]; then sed -i '/^ListenAddress $/d' "$rendered"; fi
  local reload_before=$CHANGED
  inv render-known-hosts > "$tempdir/known_hosts"
  inv render-authorized-keys --node "$P2P_NODE" --user p2prepl > "$tempdir/p2prepl"
  inv render-authorized-keys --node "$P2P_NODE" --user p2pbulk > "$tempdir/p2pbulk"
  inv render-authorized-keys --node "$P2P_NODE" --user p2pvirt > "$tempdir/p2pvirt"
  install_file "$rendered" "$_sshd_config" 0644 root:root
  install_file "$tempdir/known_hosts" "$_sshd_known" 0644 root:root
  install_file "$tempdir/p2prepl" "$_sshd_keys/p2prepl" 0644 root:root
  install_file "$tempdir/p2pbulk" "$_sshd_keys/p2pbulk" 0644 root:root
  install_file "$tempdir/p2pvirt" /var/lib/p2pnet/virt/.ssh/authorized_keys 0600 p2pvirt:p2pvirt
  install_unit "$_sshd_service"
  reload_units
  sshd -t -f "$_sshd_config"
  if systemctl is-active --quiet "$_sshd_unit"; then
    if ! systemctl is-enabled --quiet "$_sshd_unit"; then systemctl enable "$_sshd_unit"; CHANGED=$((CHANGED + 1)); fi
    if (( CHANGED != reload_before )); then
      systemctl reload-or-restart "$_sshd_unit"
      CHANGED=$((CHANGED + 1))
    fi
  else
    systemctl enable --now "$_sshd_unit"
    CHANGED=$((CHANGED + 1))
  fi
  if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -q '^Status: active'; then
    check WARN sshd.ufw "run: ufw allow ${P2P_WG_PORT}/udp; ufw allow in on wg0"
  fi
  mark_installed sshd
  printf 'p2pnet[sshd] install complete: changed=%s\n' "${CHANGED:-0}"
}

sshd_verify() {
  if ! is_installed sshd; then check SKIP sshd 'not installed'; verify_done "${P2P_VERIFY_FAILED:-0}"; return; fi
  load_env
  if systemctl is-active --quiet "$_sshd_unit"; then check PASS sshd.service active; else check FAIL sshd.service inactive; fi
  local listeners peer ovl4 replica
  listeners=$(ss -Hltn 2>/dev/null || true)
  for peer in "$P2P_OVL4:2222" "$P2P_OVL4:2223"; do
    if grep -Fq "$peer" <<< "$listeners"; then check PASS "sshd.listen.$peer" listening; else check FAIL "sshd.listen.$peer" missing; fi
  done
  while IFS=$'\t' read -r peer ovl4 _ _ _ _ _ _ _ _ _ replica; do
    [[ -n $peer && $peer != "$P2P_NODE" ]] || continue
    if timeout 15 /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$ovl4" "state $replica/$P2P_NODE/_probe" 2>/dev/null | sed -n '1p' | grep -Fxq 'token=-'; then
      check PASS "sshd.replication.$peer" guard-state
    else check FAIL "sshd.replication.$peer" state-probe-failed; fi
    if timeout 15 /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$ovl4" id >/dev/null 2>&1; then
      check FAIL "sshd.guard.$peer" unexpected-command-allowed
    else check PASS "sshd.guard.$peer" id-denied; fi
    # shellcheck disable=SC2016
    if timeout 15 runuser -u p2pbulk -- env P2PNET_SSH_KEY=/var/lib/p2pnet/bulk/.ssh/id_ed25519 bash -c 'printf "ls /torrents\\n" | sftp -S /opt/p2pnet/libexec/ssh-p2p -P 2223 -b - "p2pbulk@$1"' _ "$ovl4" >/dev/null 2>&1; then
      check PASS "sshd.bulk.$peer" sftp-list
    else check FAIL "sshd.bulk.$peer" sftp-failed; fi
  done < <(inv peers --node "$P2P_NODE")
  verify_done "${P2P_VERIFY_FAILED:-0}"
}

sshd_uninstall() {
  require_root
  local purge=0
  while (($#)); do
    case $1 in --purge) purge=1; shift ;; *) sshd_usage "unknown sshd uninstall option: $1" ;; esac
  done
  systemctl disable --now "$_sshd_unit" >/dev/null 2>&1 || true
  rm -f "/etc/systemd/system/$_sshd_unit" "$_sshd_config" "$_sshd_known" "$_sshd_keys/p2prepl" "$_sshd_keys/p2pbulk" /var/lib/p2pnet/virt/.ssh/authorized_keys
  UNITS_CHANGED=1
  reload_units
  if (( purge )); then rm -rf /var/lib/p2pnet/public; fi
  mark_uninstalled sshd
}

log INFO "action=${1:-missing}"
case ${1:-} in
  install) (($# == 1)) || sshd_usage 'install takes no options'; sshd_install ;;
  verify) (($# == 1)) || sshd_usage 'verify takes no options'; sshd_verify ;;
  uninstall) shift; sshd_uninstall "$@" ;;
  *) sshd_usage 'usage: p2pnet sshd {install|verify|uninstall [--purge]}' ;;
esac
