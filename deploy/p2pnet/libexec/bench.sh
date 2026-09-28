#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck disable=SC1091
source /opt/p2pnet/libexec/common.sh

BENCH_COMPONENT=bench
BENCH_UNIT=p2pnet-iperf3.service

_bench_usage() {
  printf 'usage: p2pnet bench {install|verify|uninstall|run PEER [--duration SEC] [--json] [--perf]}\n' >&2
  exit 2
}

_bench_install() {
  require_root
  load_env
  need_cmd systemctl iperf3 ss
  install_unit "$P2P_PREFIX/systemd/$BENCH_UNIT"
  reload_units
  if ! systemctl is-enabled --quiet "$BENCH_UNIT" || ! systemctl is-active --quiet "$BENCH_UNIT"; then CHANGED=$((CHANGED + 1)); fi
  systemctl enable --now "$BENCH_UNIT"
  mark_installed "$BENCH_COMPONENT"
  printf 'p2pnet[bench] install complete: changed=%s\n' "$CHANGED"

}

_bench_verify() {
  if ! is_installed "$BENCH_COMPONENT"; then
    printf 'SKIP bench not installed\n'
    return 0
  fi
  load_env
  local fail=0
  if systemctl is-active --quiet "$BENCH_UNIT"; then check PASS bench.service active; else check FAIL bench.service inactive; fail=1; fi
  if ss -Hltn | grep -Fq "$P2P_OVL4:5201"; then check PASS bench.listener "$P2P_OVL4:5201"; else check FAIL bench.listener "not listening on $P2P_OVL4:5201"; fail=1; fi
  (( fail == 0 )) || return 1
}

_bench_uninstall() {
  case ${1:-} in
    ''|--purge) (($# <= 1)) || _bench_usage ;;
    *) _bench_usage ;;
  esac
  require_root
  systemctl disable --now "$BENCH_UNIT" 2>/dev/null || true
  rm -f "/etc/systemd/system/$BENCH_UNIT"
  # shellcheck disable=SC2034
  UNITS_CHANGED=1
  reload_units
  mark_uninstalled "$BENCH_COMPONENT"
  printf 'p2pnet[bench] uninstall complete\n'
}

_bench_peer_fields() {
  local peer=$1 line
  while IFS= read -r line; do
    [[ -n $line ]] || continue
    IFS=$'\t' read -r BENCH_PEER BENCH_OVL4 _ BENCH_ENDPOINT _ _ _ _ _ _ BENCH_IPERF _ <<<"$line"
    if [[ $BENCH_PEER == "$peer" ]]; then return 0; fi
  done < <(inv peers --node "$P2P_NODE")
  die "unknown peer: $peer"
}

_bench_measure() {
  local host=$1 parallel=$2 duration=$3 outfile=$4
  local -a args=(-c "$host" -t "$duration" -J)
  (( parallel > 1 )) && args+=(-P "$parallel")
  /usr/bin/iperf3 "${args[@]}" >"$outfile"
  jq -er '((.end.sum_received.bits_per_second // .end.sum.bits_per_second) / 1000000)' "$outfile"
}

_bench_rtt() {
  local host=$1
  ping -n -c 10 -W 2 "$host" | sed -nE 's/.* = [^/]+\/([^/]+)\/.*/\1/p'
}

_bench_cpu() {
  local file=$1
  /usr/bin/python3 - "$file" <<'PY'
import sys
rows=[]
for line in open(sys.argv[1], encoding="utf-8"):
    fields=line.split()
    if len(fields)>=3 and fields[0]=="Average:" and fields[1] not in ("all", "CPU"):
        try: rows.append((int(fields[1]), 100.0-float(fields[-1])))
        except ValueError: pass
if not rows: raise SystemExit("mpstat did not report per-core average idle values")
busy=max(rows, key=lambda row:row[1])
print(f"{sum(row[1] for row in rows)/len(rows):.2f} {busy[1]:.2f} {busy[0]}")
PY
}

_bench_run() {
  local peer=$1 duration=10 json=0 perf=0 arg
  shift
  while (($#)); do
    arg=$1; shift
    case $arg in
      --duration) (($#)) || _bench_usage; duration=$1; shift; [[ $duration =~ ^[1-9][0-9]*$ ]] || _bench_usage ;;
      --json) json=1 ;;
      --perf) perf=1 ;;
      *) _bench_usage ;;
    esac
  done
  load_env
  _bench_peer_fields "$peer"
  need_cmd ping mpstat jq
  local tmp rtt o1 o8 u1='' u8='' cpu_mean cpu_max cpu_core
  tmp=$(mktemp -d)
  BENCH_TMP=$tmp
  trap 'rm -rf -- "$BENCH_TMP"' EXIT
  rtt=$(_bench_rtt "$BENCH_OVL4") || die "overlay ping failed to $peer"
  [[ -n $rtt ]] || die "no RTT result from $peer"
  o1=$(_bench_measure "$BENCH_OVL4" 1 "$duration" "$tmp/o1.json") || die "single-stream overlay iperf failed"
  mpstat -P ALL 1 "$duration" >"$tmp/mpstat.txt" &
  local mpstat_pid=$!
  o8=$(_bench_measure "$BENCH_OVL4" 8 "$duration" "$tmp/o8.json") || { kill "$mpstat_pid" 2>/dev/null || true; die 'eight-stream overlay iperf failed'; }
  wait "$mpstat_pid" || die 'mpstat sampling failed'
  read -r cpu_mean cpu_max cpu_core < <(_bench_cpu "$tmp/mpstat.txt") || die 'cannot parse per-core CPU utilization'
  if [[ $BENCH_IPERF == true || $BENCH_IPERF == 1 ]]; then
    local endpoint_host=${BENCH_ENDPOINT%:*}
    [[ $BENCH_ENDPOINT == \[*\]:* ]] && endpoint_host=${BENCH_ENDPOINT%%]*} && endpoint_host=${endpoint_host#[}
    u1=$(_bench_measure "$endpoint_host" 1 "$duration" "$tmp/u1.json") || die 'single-stream underlay iperf failed'
    u8=$(_bench_measure "$endpoint_host" 8 "$duration" "$tmp/u8.json") || die 'eight-stream underlay iperf failed'
  fi
  if (( perf )); then
    require_root
    need_cmd perf
    perf record -a -g -F 99 -o /tmp/p2pnet-bench.perf -- sleep 5 &
    local perf_pid=$!
    _bench_measure "$BENCH_OVL4" 8 5 "$tmp/perf.json" >/dev/null || { kill "$perf_pid" 2>/dev/null || true; die 'perf iperf run failed'; }
    wait "$perf_pid" || die 'perf record failed'
    if (( json )); then
      perf report --stdio --no-children --sort symbol -i /tmp/p2pnet-bench.perf | sed -n '1,25p' >&2
    else
      perf report --stdio --no-children --sort symbol -i /tmp/p2pnet-bench.perf | sed -n '1,25p'
    fi
  fi
  # inventory.py validates each emitted value for safe shell sourcing.
  # shellcheck disable=SC1090
  source <(inv path --from "$P2P_NODE" --to "$peer")
  local cc rmem retrans
  cc=$(sysctl -n net.ipv4.tcp_congestion_control)
  read -r _ _ rmem <<<"$(sysctl -n net.ipv4.tcp_rmem)"
  retrans=$(jq -r '.end.sum_sent.retransmits // .end.sum.retransmits // 0' "$tmp/o8.json")
  BENCH_RTT=$rtt BENCH_O1=$o1 BENCH_O8=$o8 BENCH_U1=$u1 BENCH_U8=$u8 BENCH_CPU_MEAN=$cpu_mean BENCH_CPU_MAX=$cpu_max BENCH_CPU_CORE=$cpu_core BENCH_CEIL=$P2P_PATH_OVERLAY_CEIL_MBIT BENCH_EXPECTED=$P2P_PATH_OVERLAY_TARGET_MBIT BENCH_RATE=$P2P_PATH_KBIT BENCH_CC=$cc BENCH_RMEM=$rmem BENCH_RETRANS=$retrans "$P2P_PY" - "$json" <<'PY'
import json, os, sys
f=lambda k: float(os.environ[k])
rtt=f("BENCH_RTT"); o1=f("BENCH_O1"); o8=f("BENCH_O8"); ceil=f("BENCH_CEIL")
busy=f("BENCH_CPU_MAX"); cpu_mean=f("BENCH_CPU_MEAN"); core=int(os.environ["BENCH_CPU_CORE"])
if busy >= 95:
    verdict=f"CPU-bound: core {core} at {busy:.1f}% (WireGuard crypto/softirq or iperf3 itself)"
elif o8 >= 1.25*o1:
    bdp=f("BENCH_RATE")*rtt/8000000
    rmem=int(os.environ["BENCH_RMEM"])
    advice="; run p2pnet tune install" if os.environ["BENCH_CC"]!="bbr" or rmem < 2*bdp*1000000 else ""
    verdict=f"per-flow window/BDP-bound: 1 stream {o1:.1f} vs 8 streams {o8:.1f}; BDP={bdp:.2f} MB vs tcp_rmem max={rmem/1048576:.1f} MiB; cc={os.environ['BENCH_CC']}{advice}"
elif o8 >= .9*ceil:
    verdict=f"link-bound: {o8/ceil*100:.1f}% of expected overlay ceiling {ceil:.1f} Mbit"
else:
    retrans=int(os.environ["BENCH_RETRANS"])
    verdict=f"path-limited below expected ({o8/ceil*100:.1f}%): retransmits {retrans} — ISP shaping, loss, MTU or the peer's downlink"
result={"rtt_ms":rtt,"overlay_mbit":{"one_stream":o1,"eight_streams":o8},"overlay_retransmits":int(os.environ["BENCH_RETRANS"]),"cpu":{"mean_busy_pct":cpu_mean,"max_busy_pct":busy,"max_core":core},"expected_overlay_ceiling_mbit":ceil,"expected_overlay_target_mbit":f("BENCH_EXPECTED"),"verdict":verdict}
if os.environ.get("BENCH_U1"):
    u1=f("BENCH_U1"); u8=f("BENCH_U8"); ratio=o8/u8 if u8 else 0
    result["underlay_mbit"]={"one_stream":u1,"eight_streams":u8}
    result["overlay_underlay_ratio"]=ratio
    result["wireguard_overhead_abnormal"]=ratio<.85
    result["overlay_underlay_expected_ratio"]=(int(os.environ.get("P2P_WG_MTU","1440"))-52)/1448
if len(sys.argv)>1 and sys.argv[1]=="1":
    print(json.dumps(result,sort_keys=True))
else:
    print(f"RTT {rtt:.2f} ms; overlay O1 {o1:.1f} Mbit, O8 {o8:.1f} Mbit; CPU mean {cpu_mean:.1f}%, max core {core} {busy:.1f}%")
    print("U1/U8 {0:.1f}/{1:.1f} Mbit; overlay/underlay={2:.3f} (expected ≈ {3:.3f})".format(result["underlay_mbit"]["one_stream"],result["underlay_mbit"]["eight_streams"],result["overlay_underlay_ratio"],result["overlay_underlay_expected_ratio"]) if "underlay_mbit" in result else "")
    print(verdict)
PY
}

bench_main() {
  local action=${1:-}; (($#)) && shift
  case $action in
    install) (($# == 0)) || _bench_usage; _bench_install ;;
    verify) (($# == 0)) || _bench_usage; _bench_verify ;;
    uninstall) _bench_uninstall "$@" ;;
    run) (($# >= 1)) || _bench_usage; local peer=$1; shift; _bench_run "$peer" "$@" ;;
    *) _bench_usage ;;
  esac
}

bench_main "$@"
