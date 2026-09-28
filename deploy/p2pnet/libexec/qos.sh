#!/usr/bin/env bash
set -Eeuo pipefail

if ! declare -F load_env >/dev/null 2>&1; then
  _self_dir=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  # shellcheck source=common.sh
  source "${P2P_COMMON:-${_self_dir}/common.sh}"
fi
_qos_unit=p2pnet-qos.service
_qos_service=${P2P_PREFIX:-/opt/p2pnet}/systemd/p2pnet-qos.service
P2P_COMPONENT=qos

_qos_cmd() { run tc "$@"; }
_qos_value() { local key=$1; printf '%s' "${!key:-}"; }
_qos_burst() {
  local rate=$1 bytes
  bytes=$(((rate * 1000 + 7) / 8 / 1000))
  if (( bytes < 65536 )); then bytes=65536; fi
  printf '%s' "$bytes"
}
_qos_class_data() {
  local output
  output=$(tc -s class show dev "$1" 2>/dev/null) || return 1
  /usr/bin/python3 - "$output" <<'PY'
import json
import re
import sys

def to_kbit(value):
    match = re.fullmatch(r"([0-9.]+)(Gbit|Mbit|Kbit|bit)", value)
    if match is None:
        raise ValueError(f"unsupported tc rate: {value}")
    scale = {"Gbit": 1_000_000, "Mbit": 1_000, "Kbit": 1, "bit": 0.001}
    return round(float(match.group(1)) * scale[match.group(2)])

classes = {}
for line in sys.argv[1].splitlines():
    match = re.search(
        r"\bclass htb (1:\S+).*?\brate ([^ ]+) ceil ([^ ]+)", line
    )
    if match:
        classes[match.group(1)] = {
            "rate": to_kbit(match.group(2)),
            "ceil": to_kbit(match.group(3)),
        }
print(json.dumps(classes))
PY
}

qos_usage() { log ERROR "$*"; exit 2; }
_qos_tree_matches() {
  local dev=$1 root=$2 ctrl=$3 mig=$4 repl=$5 def=$6 bulk=$7 night=$8 day=$9 ipv6=${10}
  local classes filters
  classes=$(_qos_class_data "$dev") || return 1
  local filter_data
  filter_data=$(tc -j filter show dev "$dev" parent 1: 2>/dev/null || true)
  [[ -n $filter_data ]] || filter_data='[]'
  filters=$(printf '%s\n' "$filter_data" | /usr/bin/python3 -c 'import json,sys; print(sum(1 for row in json.load(sys.stdin) if row.get("options",{}).get("order") is not None))') || return 1
  /usr/bin/python3 - "$classes" "$filters" "$root" "$ctrl" "$mig" "$repl" "$def" "$bulk" "$night" "$day" "$ipv6" <<'PY'
import json, sys
rows = json.loads(sys.argv[1])
root, ctrl, mig, repl, default, bulk, night, day = map(int, sys.argv[3:11])
expected_filters = 31 if sys.argv[11] == "1" else 16
if int(sys.argv[2]) != expected_filters:
    raise SystemExit(1)
want = {"1:1": (root, (root,)), "1:10": (ctrl, (root,)),
        "1:20": (mig, (root,)), "1:25": (repl, (root,)),
        "1:30": (default, (root,)), "1:40": (bulk, (night, day))}
for cid, (rate, ceilings) in want.items():
    row = rows.get(cid)
    if row is None:
        raise SystemExit(1)
    actual_rate = int(row.get("rate", 0))
    actual_ceil = int(row.get("ceil", 0))
    if abs(actual_rate-rate) * 100 > max(1, rate):
        raise SystemExit(1)
    if not any(abs(actual_ceil-target) * 100 <= max(1, target) for target in ceilings):
        raise SystemExit(1)
PY
}

qos_apply_device() {
  local dev=$1 profile=$2 prefix=P2P_QOS key root ctrl mig repl def bulk ceil b i p dir
  [[ $dev == wg1 ]] && prefix=P2P_QOS2
  key=${prefix}_ROOT_KBIT; root=$(_qos_value "$key")
  key=${prefix}_CTRL_KBIT; ctrl=$(_qos_value "$key")
  key=${prefix}_MIG_KBIT; mig=$(_qos_value "$key")
  key=${prefix}_REPL_KBIT; repl=$(_qos_value "$key")
  key=${prefix}_DEF_KBIT; def=$(_qos_value "$key")
  key=${prefix}_BULK_KBIT; bulk=$(_qos_value "$key")
  if [[ $profile == auto ]]; then
    local now_h now_m start_h start_m end_h end_m now_min start_min end_min
    IFS=: read -r now_h now_m <<< "$(date +%H:%M)"
    IFS=: read -r start_h start_m <<< "${P2P_NIGHT_START:?}"
    IFS=: read -r end_h end_m <<< "${P2P_NIGHT_END:?}"
    now_min=$((10#$now_h * 60 + 10#$now_m))
    start_min=$((10#$start_h * 60 + 10#$start_m))
    end_min=$((10#$end_h * 60 + 10#$end_m))
    if (( start_min < end_min )); then
      (( now_min >= start_min && now_min < end_min )) && profile=night || profile=day
    else
      (( now_min >= start_min || now_min < end_min )) && profile=night || profile=day
    fi
  fi
  if [[ $profile == night ]]; then key=${prefix}_BULK_CEIL_NIGHT_KBIT
  elif [[ $profile == day ]]; then key=${prefix}_BULK_CEIL_DAY_KBIT
  else qos_usage "unknown QoS profile: $profile"; fi
  ceil=$(_qos_value "$key")
  for b in "$root" "$ctrl" "$mig" "$repl" "$def" "$bulk" "$ceil"; do [[ $b =~ ^[0-9]+$ ]] || die "missing QoS rate for $dev"; done

  if tc qdisc show dev "$dev" 2>/dev/null | grep -q 'qdisc htb 1:' \
      && _qos_tree_matches "$dev" "$root" "$ctrl" "$mig" "$repl" "$def" "$bulk" \
        "$(_qos_value "${prefix}_BULK_CEIL_NIGHT_KBIT")" "$(_qos_value "${prefix}_BULK_CEIL_DAY_KBIT")" \
        "$([[ -n ${P2P_OVL6:-} ]] && printf 1 || printf 0)"; then
    local oldceil
    oldceil=$(_qos_class_data "$dev" 2>/dev/null | /usr/bin/python3 -c 'import json,sys; print(json.load(sys.stdin).get("1:40", {}).get("ceil", 0))' || true)
    if [[ $oldceil != "$ceil" ]]; then
      _qos_cmd class change dev "$dev" parent 1:1 classid 1:40 htb rate "${bulk}kbit" ceil "${ceil}kbit" prio 4 burst "$(_qos_burst "$bulk")" cburst "$(_qos_burst "$bulk")" quantum 60000
      if [[ ${DRY_RUN:-0} != 1 ]]; then CHANGED=$((CHANGED + 1)); fi
    fi
    return
  fi
  _qos_cmd qdisc replace dev "$dev" root handle 1: htb default 30
  _qos_cmd class replace dev "$dev" parent 1: classid 1:1 htb rate "${root}kbit" ceil "${root}kbit" burst "$(_qos_burst "$root")" cburst "$(_qos_burst "$root")" quantum 60000
  local classes=(10 20 25 30 40) rates=("$ctrl" "$mig" "$repl" "$def" "$bulk") prios=(0 1 2 3 4)
  for i in {0..4}; do
    b=$(_qos_burst "${rates[$i]}")
    if (( i == 4 )); then
      _qos_cmd class replace dev "$dev" parent 1:1 classid "1:${classes[$i]}" htb rate "${rates[$i]}kbit" ceil "${ceil}kbit" prio "${prios[$i]}" burst "$b" cburst "$b" quantum 60000
    else
      _qos_cmd class replace dev "$dev" parent 1:1 classid "1:${classes[$i]}" htb rate "${rates[$i]}kbit" ceil "${root}kbit" prio "${prios[$i]}" burst "$b" cburst "$b" quantum 60000
    fi
    _qos_cmd qdisc replace dev "$dev" parent "1:${classes[$i]}" handle "${classes[$i]}:" fq_codel
  done
  _qos_cmd filter del dev "$dev" parent 1: || true
  _qos_cmd filter add dev "$dev" parent 1: protocol ip prio 1 u32 match ip protocol 1 0xff match u16 0x0000 0xfe00 at 2 flowid 1:10
  _qos_cmd filter add dev "$dev" parent 1: protocol ip prio 2 u32 match ip protocol 6 0xff match u8 0x05 0x0f at 0 match u16 0x0000 0xffc0 at 2 match u8 0x10 0xff at 33 flowid 1:10
  for p in 53 123; do _qos_cmd filter add dev "$dev" parent 1: protocol ip prio 3 u32 match ip protocol 17 0xff match ip dport "$p" 0xffff flowid 1:10; done
  for p in 22 8090 9000; do
    _qos_cmd filter add dev "$dev" parent 1: protocol ip prio 4 u32 match ip protocol 6 0xff match ip dport "$p" 0xffff match u16 0x0000 0xfe00 at 2 flowid 1:10
    _qos_cmd filter add dev "$dev" parent 1: protocol ip prio 4 u32 match ip protocol 6 0xff match ip sport "$p" 0xffff match u16 0x0000 0xfe00 at 2 flowid 1:10
  done
  _qos_cmd filter add dev "$dev" parent 1: protocol ip prio 5 u32 match ip protocol 6 0xff match ip dport "$P2P_MIG_PORT_MIN" "$P2P_MIG_PORT_MASK" flowid 1:20
  _qos_cmd filter add dev "$dev" parent 1: protocol ip prio 6 u32 match ip protocol 6 0xff match ip dport 2222 0xffff flowid 1:25
  for dir in dport sport; do for p in 6881 2223; do _qos_cmd filter add dev "$dev" parent 1: protocol ip prio 7 u32 match ip protocol 6 0xff match ip "$dir" "$p" 0xffff flowid 1:40; done; done
  if [[ -n ${P2P_OVL6:-} ]]; then
    _qos_cmd filter add dev "$dev" parent 1: protocol ipv6 prio 1 u32 match ip6 protocol 58 0xff flowid 1:10
    for p in 53 123; do _qos_cmd filter add dev "$dev" parent 1: protocol ipv6 prio 3 u32 match ip6 protocol 17 0xff match ip6 dport "$p" 0xffff flowid 1:10; done
    for p in 22 8090 9000; do
      _qos_cmd filter add dev "$dev" parent 1: protocol ipv6 prio 4 u32 match ip6 protocol 6 0xff match ip6 dport "$p" 0xffff flowid 1:10
      _qos_cmd filter add dev "$dev" parent 1: protocol ipv6 prio 4 u32 match ip6 protocol 6 0xff match ip6 sport "$p" 0xffff flowid 1:10
    done
    _qos_cmd filter add dev "$dev" parent 1: protocol ipv6 prio 5 u32 match ip6 protocol 6 0xff match ip6 dport "$P2P_MIG_PORT_MIN" "$P2P_MIG_PORT_MASK" flowid 1:20
    _qos_cmd filter add dev "$dev" parent 1: protocol ipv6 prio 6 u32 match ip6 protocol 6 0xff match ip6 dport 2222 0xffff flowid 1:25
    for dir in dport sport; do for p in 6881 2223; do _qos_cmd filter add dev "$dev" parent 1: protocol ipv6 prio 7 u32 match ip6 protocol 6 0xff match ip6 "$dir" "$p" 0xffff flowid 1:40; done; done
  fi
  if [[ ${DRY_RUN:-0} != 1 ]]; then
    local object_count=28
    [[ -n ${P2P_OVL6:-} ]] && object_count=43
    CHANGED=$((CHANGED + object_count))
  fi
}

qos_apply() {
  load_env; need_cmd tc
  local profile=auto
  while (($#)); do
    case $1 in
      --profile) (($# >= 2)) || qos_usage 'apply --profile requires night, day, or auto'; profile=$2; shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;
      *) qos_usage "unknown qos apply option: $1" ;;
    esac
  done
  [[ ${DRY_RUN:-0} == 1 ]] || require_root
  qos_apply_device wg0 "$profile"
  if command -v ip >/dev/null 2>&1 && ip link show wg1 >/dev/null 2>&1 && [[ -n ${P2P_QOS2_ROOT_KBIT:-} ]]; then qos_apply_device wg1 "$profile"; fi
}
qos_install() {
  require_root; load_env
  local active=0 enabled=0
  systemctl is-active --quiet "$_qos_unit" && active=1
  systemctl is-enabled --quiet "$_qos_unit" && enabled=1
  install_unit "$_qos_service"; reload_units
  qos_apply --profile auto
  systemctl enable --now "$_qos_unit"
  if (( ! active || ! enabled )); then CHANGED=$((CHANGED + 1)); fi
  mark_installed qos
  printf 'p2pnet[qos] install complete: changed=%s\n' "$CHANGED"
}
qos_status() { load_env; tc -s class show dev wg0; if ip link show wg1 >/dev/null 2>&1; then tc -s class show dev wg1; fi; }

qos_verify() {
  if ! is_installed qos; then check SKIP qos 'not installed'; verify_done "${P2P_VERIFY_FAILED:-0}"; return; fi
  load_env
  local dev data count pre class name expected
  for dev in wg0 wg1; do
    [[ $dev == wg0 || ( -n ${P2P_QOS2_ROOT_KBIT:-} && -e /sys/class/net/wg1 ) ]] || continue
    data=$(_qos_class_data "$dev" 2>/dev/null || printf '{}')
    if tc qdisc show dev "$dev" 2>/dev/null | grep -q 'qdisc htb 1:'; then check PASS "qos.$dev.root" htb; else check FAIL "qos.$dev.root" missing-htb; continue; fi
    pre=P2P_QOS; [[ $dev == wg1 ]] && pre=P2P_QOS2
    expected=$(/usr/bin/python3 -c 'import json,sys; print(" ".join(json.loads(sys.argv[1]).keys()) )' "$data")
    for pair in '1:10 CTRL' '1:20 MIG' '1:25 REPL' '1:30 DEF' '1:40 BULK'; do
      read -r class name <<< "$pair"
      if [[ " $expected " == *" $class "* ]]; then check PASS "qos.$dev.$class" "${name,,}-present"; else check FAIL "qos.$dev.$class" missing; fi
    done
    local status
    status=$(/usr/bin/python3 - "$data" "$(_qos_value "${pre}_ROOT_KBIT")" "$(_qos_value "${pre}_CTRL_KBIT")" "$(_qos_value "${pre}_MIG_KBIT")" "$(_qos_value "${pre}_REPL_KBIT")" "$(_qos_value "${pre}_DEF_KBIT")" "$(_qos_value "${pre}_BULK_KBIT")" "$(_qos_value "${pre}_BULK_CEIL_NIGHT_KBIT")" "$(_qos_value "${pre}_BULK_CEIL_DAY_KBIT")" <<'PY'
import json, sys
rows = json.loads(sys.argv[1])
root, ctrl, mig, repl, default, bulk, night, day = map(int, sys.argv[2:])
want = {"1:1": (root, root), "1:10": (ctrl, root), "1:20": (mig, root),
        "1:25": (repl, root), "1:30": (default, root), "1:40": (bulk, None)}
bad = []
for cid, (rate, ceil) in want.items():
    row = rows.get(cid)
    if row is None:
        bad.append(cid + ":missing")
        continue
    actual_rate = int(row.get("rate", 0))
    actual_ceil = int(row.get("ceil", 0))
    if abs(actual_rate-rate) * 100 > max(1, rate):
        bad.append(f"{cid}:rate={actual_rate}/{rate}")
    ceilings = (night, day) if ceil is None else (ceil,)
    if not any(abs(actual_ceil-target) * 100 <= max(1, target) for target in ceilings):
        bad.append(f"{cid}:ceil={actual_ceil}/{ceilings}")
print(";".join(bad))
PY
    )
    if [[ -z $status ]]; then check PASS "qos.$dev.rates" within-1-percent; else check FAIL "qos.$dev.rates" "$status"; fi
    count=$(tc -j filter show dev "$dev" parent 1: 2>/dev/null | /usr/bin/python3 -c 'import json,sys; print(sum(1 for row in json.load(sys.stdin) if row.get("options",{}).get("order") is not None))' 2>/dev/null || printf 0)
    local expected_filters
    expected_filters=16; [[ -n ${P2P_OVL6:-} ]] && expected_filters=31
    if (( count == expected_filters )); then check PASS "qos.$dev.filters" "$count-installed"; else check FAIL "qos.$dev.filters" "expected-$expected_filters-got-$count"; fi
    local drops
    drops=$(tc -s class show dev "$dev" | /usr/bin/python3 -c 'import re,sys; s=sys.stdin.read(); m=re.search(r"class htb 1:10.*?dropped (\d+)",s,re.S); print(m.group(1) if m else 0)')
    if (( drops > 0 )); then check WARN "qos.$dev.control-drops" "$drops"; else check PASS "qos.$dev.control-drops" zero; fi
  done
  verify_done "${P2P_VERIFY_FAILED:-0}"
}
qos_snapshot() { tc -s class show dev wg0 | /usr/bin/python3 -c 'import re,sys; s=sys.stdin.read(); print(" ".join(f"{c}={p}" for c,p in re.findall(r"class htb (1:[0-9]+).*?Sent \d+ bytes (\d+) pkt",s,re.S)))'; }
qos_counter() { local snap=$1 class=$2; /usr/bin/python3 - "$snap" "$class" <<'PY'
import sys
for item in sys.argv[1].split():
    key, value = item.split('=', 1)
    if key == sys.argv[2]:
        print(value)
        break
else:
    print(0)
PY
}
qos_probe_port() {
  local peer=$1 port=$2 expected=$3 before after b a
  before=$(qos_snapshot); b=$(qos_counter "$before" "$expected")
# shellcheck disable=SC2016
  for _ in {1..20}; do timeout 1 bash -c 'exec 3<>/dev/tcp/$1/$2' _ "$peer" "$port" >/dev/null 2>&1 || true; done
  after=$(qos_snapshot); a=$(qos_counter "$after" "$expected")
  (( a - b >= 20 )) || { check FAIL "qos.probe.$port" "expected-$expected delta=$((a-b))"; return 1; }
  check PASS "qos.probe.$port" "$expected packets=$((a-b))"
}
qos_selftest() {
  load_env; local peer='' waitval=60
  while (($#)); do
    case $1 in
      --peer|--wait) (($# >= 2)) || qos_usage "$1 requires a value"; if [[ $1 == --peer ]]; then peer=$2; else waitval=$2; fi; shift 2 ;;
      *) qos_usage "unknown qos selftest option: $1" ;;
    esac
  done
  [[ -n $peer ]] || qos_usage 'usage: p2pnet qos selftest --peer NODE [--wait 60]'
  local node=$peer
  peer=$(inv env --node "$node" | sed -n 's/^P2P_OVL4="\([^"]*\)"$/\1/p')
  [[ -n $peer ]] || die "cannot resolve overlay address for $node"
  [[ $waitval =~ ^[0-9]+$ ]] || qos_usage '--wait must be an integer number of seconds'
  local deadline=$((SECONDS + waitval))
  until ping -c 1 -W 1 "$peer" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then check FAIL qos.peer "unreachable-$peer"; return 1; fi
    sleep 1
  done
  local failed=0 port cls before after b a migration_before
  for port in 49152 49215 49216 2222 6881 2223 5999 22; do
    case $port in 49152|49215) cls=1:20 ;; 2222) cls=1:25 ;; 6881|2223) cls=1:40 ;; 22) cls=1:10 ;; *) cls=1:30 ;; esac
    if [[ $port == 49216 ]]; then migration_before=$(qos_snapshot); b=$(qos_counter "$migration_before" 1:20); fi
    qos_probe_port "$peer" "$port" "$cls" || failed=1
    if [[ $port == 49216 ]]; then after=$(qos_snapshot); a=$(qos_counter "$after" 1:20); (( a == b )) || { check FAIL qos.probe.49216-migration 'unexpected-migration-packets'; failed=1; }; fi
  done
  before=$(qos_snapshot); local i; for i in {1..20}; do ping -c 1 -W 1 "$peer" >/dev/null 2>&1 || true; done; after=$(qos_snapshot)
  b=$(qos_counter "$before" 1:10); a=$(qos_counter "$after" 1:10)
  if (( a - b >= 20 )); then check PASS qos.probe.icmp "control packets=$((a-b))"; else check FAIL qos.probe.icmp "control delta=$((a-b))"; failed=1; fi
  (( failed == 0 )) || return 1
  verify_done "${P2P_VERIFY_FAILED:-0}"
}
qos_uninstall() {
  require_root; local dev
  while (($#)); do case $1 in --purge) shift ;; *) qos_usage "unknown qos uninstall option: $1" ;; esac; done
  for dev in wg0 wg1; do
    if [[ -e /sys/class/net/$dev ]]; then tc qdisc del dev "$dev" root >/dev/null 2>&1 || true; fi
  done
  systemctl disable --now "$_qos_unit" >/dev/null 2>&1 || true
  rm -f "/etc/systemd/system/$_qos_unit"
  UNITS_CHANGED=1; reload_units; mark_uninstalled qos
}

log INFO "action=${1:-missing}"
case ${1:-} in
  install) shift; (($# == 0)) || qos_usage 'install takes no options'; qos_install ;;
  apply) shift; qos_apply "$@" ;;
  verify) shift; (($# == 0)) || qos_usage 'verify takes no options'; qos_verify ;;
  status) shift; (($# == 0)) || qos_usage 'status takes no options'; qos_status ;;
  selftest) shift; qos_selftest "$@" ;;
  uninstall) shift; qos_uninstall "$@" ;;
  *) qos_usage 'usage: p2pnet qos {install|apply|verify|status|selftest|uninstall [--purge]}' ;;
esac
