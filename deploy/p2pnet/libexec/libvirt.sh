#!/usr/bin/env bash
set -Eeuo pipefail
# Libvirt host configuration for p2pnet.
if ! declare -F load_env >/dev/null 2>&1; then
  _self_dir=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  # shellcheck source=common.sh
  source "${P2P_COMMON:-${_self_dir}/common.sh}"
fi

p2pnet_libvirt_block() {
  cat <<EOF
# BEGIN p2pnet (managed by \`p2pnet libvirt\`; edit the inventory instead)
migration_address = "${P2P_OVL4}"
migration_host = "${P2P_OVL4}"
migration_port_min = ${P2P_MIG_PORT_MIN}
migration_port_max = ${P2P_MIG_PORT_MAX}   # 64-port aligned block → one tc u32 mask; NBD for --copy-storage uses it too
# END p2pnet
EOF
}

p2pnet_libvirt_render_qemu() {
  local config=/etc/libvirt/qemu.conf block tmp
  block=$(p2pnet_libvirt_block)
  if [[ -f $config ]]; then
    /usr/bin/python3 - "$config" "$block" <<'PY'
import pathlib, re, sys
path, block = pathlib.Path(sys.argv[1]), sys.argv[2]
text = path.read_text()
start, end = '# BEGIN p2pnet (managed by `p2pnet libvirt`; edit the inventory instead)', '# END p2pnet'
if text.count(start) != text.count(end) or text.count(start) > 1:
    raise SystemExit('invalid p2pnet managed block in /etc/libvirt/qemu.conf')
pattern = re.compile(r'(?m)^\s*(migration_address|migration_host|migration_port_min|migration_port_max)\s*=')
if start in text:
    text = re.sub(r'(?ms)^' + re.escape(start) + r'.*?^' + re.escape(end) + r'\n?', '', text)
if pattern.search(text):
    raise SystemExit('refusing: migration_* key exists outside the p2pnet managed block')
text = text.rstrip() + '\n\n' + block + '\n'
sys.stdout.write(text)
PY
  else
    printf '%s\n' "$block"
  fi
}

p2pnet_libvirt_install() {
  require_root
  load_env
  need_cmd virsh usermod
  local tmp qemu_before=$CHANGED qemu_changed=0 groups pool_info net_info
  tmp=$(mktemp)
  p2pnet_libvirt_render_qemu >"$tmp" || { rm -f "$tmp"; return 1; }
  install_file "$tmp" /etc/libvirt/qemu.conf 0644
  rm -f "$tmp"
  (( CHANGED != qemu_before )) && qemu_changed=1
  groups=$(id -nG p2pvirt 2>/dev/null || true)
  if [[ " $groups " != *" libvirt "* ]]; then usermod -aG libvirt p2pvirt; CHANGED=$((CHANGED + 1)); fi
  if ! pool_info=$(virsh pool-info default 2>/dev/null); then
    virsh pool-define-as default dir --target /var/lib/libvirt/images
    virsh pool-build default
    CHANGED=$((CHANGED + 1))
    pool_info=$(virsh pool-info default)
  fi
  if ! grep -q 'State:.*running' <<<"$pool_info"; then virsh pool-start default; CHANGED=$((CHANGED + 1)); fi
  if ! grep -q 'Autostart:.*yes' <<<"$pool_info"; then virsh pool-autostart default; CHANGED=$((CHANGED + 1)); fi
  tmp=$(mktemp)
  cat >"$tmp" <<EOF
<network><name>p2p-l2</name><forward mode='bridge'/><bridge name='${P2P_BRIDGE}'/></network>
EOF
  local net_matches=0
  if virsh net-info p2p-l2 >/dev/null 2>&1; then
    if /usr/bin/python3 - "$(virsh net-dumpxml p2p-l2)" "$P2P_BRIDGE" <<'PY'
import sys, xml.etree.ElementTree as ET
r=ET.fromstring(sys.argv[1])
name=r.findtext('name')
forward=r.find('forward')
bridge=r.find('bridge')
sys.exit(0 if name=='p2p-l2' and forward is not None and forward.get('mode')=='bridge' and bridge is not None and bridge.get('name')==sys.argv[2] else 1)
PY
    then net_matches=1; fi
  fi
  if (( net_matches == 0 )); then
    if virsh net-info p2p-l2 >/dev/null 2>&1; then virsh net-destroy p2p-l2 || true; virsh net-undefine p2p-l2; fi
    virsh net-define "$tmp"
    CHANGED=$((CHANGED + 1))
  fi
  rm -f "$tmp"
  net_info=$(virsh net-info p2p-l2 2>/dev/null || true)
  if ! grep -q 'Active:.*yes' <<<"$net_info"; then virsh net-start p2p-l2; CHANGED=$((CHANGED + 1)); fi
  if ! grep -q 'Autostart:.*yes' <<<"$net_info"; then virsh net-autostart p2p-l2; CHANGED=$((CHANGED + 1)); fi
  tmp=$(mktemp)
  inv render-authorized-keys --node "$(self_node)" --user p2pvirt >"$tmp"
  install_file "$tmp" /var/lib/p2pnet/virt/.ssh/authorized_keys 0600 p2pvirt:p2pvirt
  rm -f "$tmp"
  (( qemu_changed == 0 )) || systemctl try-restart libvirtd.service
  mark_installed libvirt
  printf 'p2pnet[libvirt] install complete: changed=%s\n' "$CHANGED"
}

p2pnet_libvirt_verify() {
  load_env
  if ! is_installed libvirt; then check SKIP libvirt 'not installed'; return 0; fi
  local expected actual info uri peer ovl failed=0
  expected=$(p2pnet_libvirt_block)
  actual=$(/usr/bin/python3 - /etc/libvirt/qemu.conf <<'PY'
import pathlib,re,sys
s=pathlib.Path(sys.argv[1]).read_text()
m=re.search(r'(?ms)^# BEGIN p2pnet .*?^# END p2pnet\n?',s)
if m: print(m.group(0).rstrip())
PY
)
  if [[ $actual == "$expected" ]]; then check PASS libvirt.config 'managed block matches inventory'; else check FAIL libvirt.config 'managed block differs'; failed=1; fi
  info=$(virsh pool-info default 2>&1) || info=''
  if grep -q 'State:.*running' <<<"$info"; then check PASS libvirt.pool default-active; else check FAIL libvirt.pool 'default pool is not active'; failed=1; fi
  info=$(virsh net-info p2p-l2 2>&1) || info=''
  if grep -q 'Active:.*yes' <<<"$info" && grep -q 'Autostart:.*yes' <<<"$info"; then check PASS libvirt.network 'p2p-l2 active and autostarted'; else check FAIL libvirt.network 'p2p-l2 inactive or not autostarted'; failed=1; fi
  if runuser -u p2pvirt -- virsh -c qemu:///system list >/dev/null 2>&1; then check PASS libvirt.local-access 'p2pvirt can access qemu:///system'; else check FAIL libvirt.local-access 'p2pvirt virsh access failed'; failed=1; fi
  while IFS=$'\t' read -r peer ovl _; do
    [[ -n $peer ]] || continue
    uri="qemu+ssh://p2pvirt@${ovl}/system?command=/opt/p2pnet/libexec/ssh-p2p&no_tty=1"
    if timeout 15 virsh -c "$uri" version >/dev/null 2>&1; then check PASS "libvirt.peer.$peer" 'remote libvirt reachable'; else check FAIL "libvirt.peer.$peer" 'remote libvirt connection failed'; failed=1; fi
  done < <(inv peers --node "$(self_node)" | cut -f1,2)
  verify_done "$failed"
}

p2pnet_libvirt_uninstall() {
  require_root
  local force=0
  case ${1:-} in '') ;; --purge) ;; --force) force=1 ;; *) die 'usage: p2pnet libvirt uninstall [--force|--purge]' ;; esac
  if virsh net-info p2p-l2 >/dev/null 2>&1; then
    local domains
    domains=$(virsh list --all --name)
    while IFS= read -r d; do
      [[ -n $d ]] || continue
      if virsh dumpxml "$d" 2>/dev/null | /usr/bin/python3 -c 'import sys,xml.etree.ElementTree as E; r=E.fromstring(sys.stdin.read()); raise SystemExit(0 if any(s.get("network")=="p2p-l2" for s in r.findall("./devices/interface/source")) else 1)' && (( force == 0 )); then die "refusing to remove p2p-l2: domain $d references it (use --force)"; fi
    done <<<"$domains"
    virsh net-destroy p2p-l2 2>/dev/null || true
    virsh net-undefine p2p-l2
  fi
  if [[ -f /etc/libvirt/qemu.conf ]]; then
    local tmp
    tmp=$(mktemp)
    /usr/bin/python3 - /etc/libvirt/qemu.conf >"$tmp" <<'PY'
import pathlib,re,sys
s=pathlib.Path(sys.argv[1]).read_text()
start='# BEGIN p2pnet (managed by `p2pnet libvirt`; edit the inventory instead)'
end='# END p2pnet'
if s.count(start)!=s.count(end) or s.count(start)>1: raise SystemExit('invalid managed block')
if start in s: s=re.sub(r'(?ms)^'+re.escape(start)+r'.*?^'+re.escape(end)+r'\n?', '', s)
sys.stdout.write(s)
PY
    install_file "$tmp" /etc/libvirt/qemu.conf 0644
    rm -f "$tmp"
  fi
  rm -f /var/lib/p2pnet/virt/.ssh/authorized_keys
  systemctl try-restart libvirtd.service
  mark_uninstalled libvirt
  printf 'p2pnet[libvirt] uninstall complete: changed=%s\n' "$CHANGED"
}

p2pnet_libvirt_main() {
  local action=${1:-}; shift || true
  case $action in install) p2pnet_libvirt_install "$@";; verify) p2pnet_libvirt_verify;; uninstall) p2pnet_libvirt_uninstall "$@";; *) die 'usage: p2pnet libvirt {install|verify|uninstall [--force]}' ;; esac
}

p2pnet_libvirt_main "$@"
