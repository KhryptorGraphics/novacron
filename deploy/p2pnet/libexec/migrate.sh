#!/usr/bin/env bash
set -Eeuo pipefail
# libvirt/virsh migration planner and runner; QEMU migration is not mptcp-wrapped.
if ! declare -F load_env >/dev/null 2>&1; then
  _self_dir=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  # shellcheck disable=SC1090
  # shellcheck source=common.sh
  source "${P2P_COMMON:-${_self_dir}/common.sh}"
fi

p2pnet_migrate_dest() {
  local wanted=$1 peer ovl
  while IFS=$'\t' read -r peer ovl _; do
    if [[ $peer == "$wanted" ]]; then printf '%s\n' "$ovl"; return 0; fi
  done < <(inv peers --node "$(self_node)")
  die "unknown migration destination: $wanted"
}

p2pnet_migrate_path() {
  local dest=$1
  # inventory env output is constrained to shell-safe values by validate.
  # shellcheck disable=SC1090
  source <(inv path --from "$(self_node)" --to "$dest")
  P2P_PATH_MIG_BW_MBPS=${P2P_PATH_MIG_BW_MBPS:?inventory omitted migration bandwidth}
  P2P_PATH_DOWNTIME_MS=${P2P_PATH_DOWNTIME_MS:?inventory omitted migration downtime}
  P2P_PATH_MIG_CHANNELS=${P2P_PATH_MIG_CHANNELS:?inventory omitted migration channel count}
}

p2pnet_migrate_dom_value() {
  local domain=$1 key=$2 output
  output=$(virsh dominfo "$domain") || die "cannot read domain info: $domain"
  sed -n "s/^${key}:[[:space:]]*//p" <<<"$output" | awk '{print $1}'
}

p2pnet_migrate_dirty_rate() {
  local domain=$1 output status value i
  if ! virsh domdirtyrate-calc "$domain" --seconds 5 --mode page-sampling >/dev/null 2>&1; then return 1; fi
  for ((i=0; i<15; i++)); do
    output=$(virsh domstats "$domain" --dirtyrate 2>/dev/null || true)
    status=$(sed -n 's/.*dirtyrate.calc_status=//p' <<<"$output" | head -n1)
    if [[ $status == 2 ]]; then
      value=$(sed -n 's/.*dirtyrate.megabytes_per_second=//p' <<<"$output" | head -n1)
      [[ $value =~ ^[0-9]+([.][0-9]+)?$ ]] || return 1
      printf '%s\n' "$value"
      return 0
    fi
    sleep 1
  done
  return 1
}

p2pnet_migrate_plan_values() {
  local domain=$1 dest=$2 ram_kib ram dirty='' ratio profile reason state
  p2pnet_migrate_path "$dest"
  state=$(virsh domstate "$domain" 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
  [[ $state == running ]] || die "domain must be running: $domain (state=${state:-unknown})"
  ram_kib=$(p2pnet_migrate_dom_value "$domain" 'Used memory')
  [[ $ram_kib =~ ^[0-9]+$ ]] || die "domain has no numeric Used memory: $domain"
  ram=$((ram_kib / 1024))
  if dirty=$(p2pnet_migrate_dirty_rate "$domain"); then
    ratio=$(awk -v d="$dirty" -v b="$P2P_PATH_MIG_BW_MBPS" 'BEGIN { if (b == 0) print 999; else printf "%.6f", d/b }')
    if awk -v r="$ratio" 'BEGIN {exit !(r < 0.5)}'; then profile=multifd; reason='pre-copy converges geometrically'
    elif awk -v r="$ratio" 'BEGIN {exit !(r < 0.9)}'; then profile=xbzrle; reason='delta-encode re-dirtied pages + auto-converge throttling'
    else profile=postcopy; reason='pre-copy cannot converge'; fi
  else
    dirty=''; ratio=''; profile=multifd; reason='dirty rate unavailable'
  fi
  local timeout_s est_first
  timeout_s=$(awk -v r="$ram" -v b="$P2P_PATH_MIG_BW_MBPS" 'BEGIN { if (b<=0) print 300; else { n=int(3*r/b); if (n<3*r/b) n++; if(n<300)n=300; print n } }')
  est_first=$(awk -v r="$ram" -v b="$P2P_PATH_MIG_BW_MBPS" 'BEGIN {if(b<=0)print "inf"; else printf "%.2f",r/b}')
  MIG_PLAN_DOMAIN=$domain MIG_PLAN_DEST=$dest MIG_PLAN_RAM=$ram MIG_PLAN_DIRTY=$dirty MIG_PLAN_BW=$P2P_PATH_MIG_BW_MBPS MIG_PLAN_RATIO=$ratio MIG_PLAN_PROFILE=$profile MIG_PLAN_DOWNTIME=$P2P_PATH_DOWNTIME_MS MIG_PLAN_TIMEOUT=$timeout_s MIG_PLAN_FIRST=$est_first MIG_PLAN_REASON=$reason
}

p2pnet_migrate_plan() {
  local domain=${1:-} dest=${2:-} json=0
  [[ -n $domain && -n $dest ]] || die 'usage: p2pnet migrate plan DOMAIN DEST [--json]'
  shift 2
  [[ ${1:-} == --json ]] && json=1
  load_env
  p2pnet_migrate_plan_values "$domain" "$dest"
  if (( json )); then
    /usr/bin/python3 - "$MIG_PLAN_DOMAIN" "$MIG_PLAN_DEST" "$MIG_PLAN_RAM" "$MIG_PLAN_DIRTY" "$MIG_PLAN_BW" "$MIG_PLAN_RATIO" "$MIG_PLAN_PROFILE" "$MIG_PLAN_DOWNTIME" "$MIG_PLAN_TIMEOUT" "$MIG_PLAN_FIRST" "$MIG_PLAN_REASON" <<'PY'
import json,sys
keys=('domain','dest','ram_mib','dirty_rate_mbps','bw_mbps','ratio','profile','downtime_ms','timeout_s','est_first_pass_s','reason')
v=sys.argv[1:]
v[2]=int(v[2]); v[3]=float(v[3]) if v[3] else None
v[4]=int(v[4]); v[5]=float(v[5]) if v[5] else None
v[7]=int(v[7]); v[8]=int(v[8]); v[9]=float(v[9]) if v[9]!='inf' else None
print(json.dumps(dict(zip(keys,v)),separators=(',',':')))
PY
  else
    printf 'domain=%s dest=%s ram_mib=%s dirty_rate_mbps=%s bw_mbps=%s ratio=%s profile=%s downtime_ms=%s timeout_s=%s est_first_pass_s=%s reason=%s\n' \
      "$MIG_PLAN_DOMAIN" "$MIG_PLAN_DEST" "$MIG_PLAN_RAM" "${MIG_PLAN_DIRTY:-unavailable}" "$MIG_PLAN_BW" "${MIG_PLAN_RATIO:-n/a}" "$MIG_PLAN_PROFILE" "$MIG_PLAN_DOWNTIME" "$MIG_PLAN_TIMEOUT" "$MIG_PLAN_FIRST" "$MIG_PLAN_REASON"
  fi
}

p2pnet_migrate_json_result() {
  local domain=$1 dest=$2 profile=$3 dirty=$4 bw=$5 downtime=$6 elapsed=$7 processed=$8 ok=$9 error=${10:-}
  mkdir -p /var/lib/p2pnet/migrate /var/log/p2pnet
  /usr/bin/python3 - "$domain" "$dest" "$profile" "$dirty" "$bw" "$downtime" "$elapsed" "$processed" "$ok" "$error" <<'PY' > /var/lib/p2pnet/migrate/last.json.tmp
import datetime,json,re,sys
keys=('domain','dest','profile','dirty_rate_mbps','bw_mbps','downtime_ms','elapsed_ms','data_processed','ok','error')
v=sys.argv[1:]
for i in (3,4): v[i]=float(v[i]) if v[i] else None
for i in (5,): v[i]=int(v[i]) if v[i] else None
m=re.fullmatch(r'([0-9]+(?:[.][0-9]+)?)(?: *(B|KiB|MiB|GiB))?',v[7])
v[7]=int(float(m[1])*{'B':1,'KiB':1024,'MiB':1024**2,'GiB':1024**3}.get(m[2] or 'B',1)) if m else None
m=re.fullmatch(r'([0-9]+(?:[.][0-9]+)?) *(ms|s)',v[6])
v[6]=int(float(m[1])*(1000 if m[2]=='s' else 1)) if m else None
v[8]=v[8]=='true'
print(json.dumps(dict(zip(keys,v)) | {'time':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')},separators=(',',':')))
PY
  chmod 0644 /var/lib/p2pnet/migrate/last.json.tmp
  mv -f /var/lib/p2pnet/migrate/last.json.tmp /var/lib/p2pnet/migrate/last.json
}

p2pnet_migrate_metric() {
  local text=$1 field=$2
  sed -n "s/^[[:space:]]*${field}:[[:space:]]*//p" <<<"$text" | head -n1 | sed 's/[[:space:]]*$//'
}

p2pnet_migrate_run() {
  local domain=${1:-} dest=${2:-} profile=auto allow_postcopy=0 copy_storage=0 dry=0
  [[ -n $domain && -n $dest ]] || die 'usage: p2pnet migrate run DOMAIN DEST [options]'
  shift 2
  while (($#)); do
    case $1 in
      --profile) (($# >= 2)) || die '--profile requires a value'; profile=$2; shift 2;;
      --allow-postcopy) allow_postcopy=1; shift;;
      --copy-storage) copy_storage=1; shift;;
      --dry-run) dry=1; shift;;
      *) die "unknown migrate run option: $1";;
    esac
  done
  load_env
  local dest_ip dirty='' bw downtime channels ram_kib ram timeout_s xbzrle_cache uri logf jobinfo jobtype rc=0 nvram=0 state
  p2pnet_migrate_dest "$dest" >/dev/null
  p2pnet_migrate_path "$dest"
  dest_ip=$(p2pnet_migrate_dest "$dest")
  bw=$P2P_PATH_MIG_BW_MBPS; downtime=$P2P_PATH_DOWNTIME_MS; channels=$P2P_PATH_MIG_CHANNELS
  state=$(virsh domstate "$domain" 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
  [[ $state == running ]] || die "domain must be running: $domain (state=${state:-unknown})"
  ram_kib=$(p2pnet_migrate_dom_value "$domain" 'Used memory')
  [[ $ram_kib =~ ^[0-9]+$ ]] || die "domain has no numeric Used memory: $domain"
  ram=$((ram_kib / 1024))
  if [[ $profile == auto ]]; then
    p2pnet_migrate_plan_values "$domain" "$dest"
    profile=$MIG_PLAN_PROFILE; dirty=$MIG_PLAN_DIRTY; timeout_s=$MIG_PLAN_TIMEOUT
  else
    case $profile in multifd|xbzrle|postcopy) ;; *) die "invalid migration profile: $profile";; esac
    dirty=''
    if dirty=$(p2pnet_migrate_dirty_rate "$domain"); then :; else dirty=''; fi
    timeout_s=$(awk -v r="$ram" -v b="$bw" 'BEGIN { if (b<=0) print 300; else { n=int(3*r/b); if(n<3*r/b)n++; if(n<300)n=300; print n } }')
  fi
  if [[ $profile == postcopy ]] && (( allow_postcopy == 0 )); then
    printf 'refused: post-copy can split guest state across hosts if the link fails; rerun with --allow-postcopy. Recovery: virsh migrate --postcopy-resume %s\n' "$domain" >&2
    return 3
  fi
  uri="qemu+ssh://p2pvirt@${dest_ip}/system?command=/opt/p2pnet/libexec/ssh-p2p&no_tty=1"
  if (( dry == 0 )); then timeout 15 virsh -c "$uri" version >/dev/null || die "destination libvirt preflight failed: $dest"; fi
  local -a cmd=(virsh migrate --live --persistent --verbose --abort-on-error --listen-address "$dest_ip" --migrateuri "tcp://${dest_ip}" )
  (( copy_storage )) && cmd+=(--copy-storage-all)
  case $profile in
    multifd) cmd+=(--parallel --parallel-connections "$channels" --compressed --comp-methods zstd --comp-zstd-level 1 --auto-converge --auto-converge-initial 20 --auto-converge-increment 10 --timeout "$timeout_s" --timeout-suspend);;
    xbzrle)
      xbzrle_cache=$(awk -v r="$ram" 'BEGIN {v=r/4; if(v<256)v=256; if(v>2048)v=2048; printf "%.0f",v*1024*1024}')
      cmd+=(--compressed --comp-methods xbzrle --comp-xbzrle-cache "$xbzrle_cache" --auto-converge --auto-converge-initial 20 --auto-converge-increment 10 --timeout "$timeout_s" --timeout-suspend);;
    postcopy) cmd+=(--postcopy --postcopy-after-precopy);;
  esac
  cmd+=("$domain" "$uri")
  (( dry )) && { printf 'downtime_ms=%s\n' "$downtime"; printf '%q ' "${cmd[@]}"; printf '\n'; return 0; }
  mkdir -p /var/log/p2pnet /var/lib/p2pnet/migrate
  logf=/var/log/p2pnet/migrate.log
  log INFO "starting $profile migration of $domain to $dest"
  "${cmd[@]}" >>"$logf" 2>&1 &
  local pid=$! i
  for ((i=0; i<60; i++)); do
    jobinfo=$(virsh domjobinfo "$domain" 2>/dev/null || true)
    jobtype=$(sed -n 's/^[[:space:]]*Job type:[[:space:]]*//p' <<<"$jobinfo" | head -n1 | tr -d '[:space:]')
    [[ -n $jobtype && $jobtype != None ]] && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if [[ $profile != postcopy && -n ${jobtype:-} && $jobtype != None ]]; then
    virsh migrate-setmaxdowntime "$domain" "$downtime" >>"$logf" 2>&1 || log WARN 'could not set migration downtime limit while job was active'
  fi
  wait "$pid" || rc=$?
  if (( rc != 0 )); then
    local err="virsh migrate exited $rc"
    p2pnet_migrate_json_result "$domain" "$dest" "$profile" "$dirty" "$bw" '' '' '' false "$err"
    if [[ $profile == postcopy ]]; then printf 'post-copy recovery command: virsh migrate --postcopy-resume %q\n' "$domain" >&2; fi
    log ERROR "$err (see $logf)"
    return 1
  fi
  jobinfo=$(virsh domjobinfo "$domain" --completed 2>/dev/null || true)
  local elapsed processed total_downtime downtime_num
  elapsed=$(p2pnet_migrate_metric "$jobinfo" 'Time elapsed')
  processed=$(p2pnet_migrate_metric "$jobinfo" 'Data processed')
  total_downtime=$(p2pnet_migrate_metric "$jobinfo" 'Total downtime')
  downtime_num=$(awk -v x="$total_downtime" 'BEGIN {gsub(/[^0-9.]/,"",x); if(x=="") print ""; else printf "%.0f",x}')
  [[ -n $downtime_num ]] && downtime=$downtime_num
  local xml
  xml=$(virsh dumpxml "$domain" 2>/dev/null || true)
  grep -q '<nvram' <<<"$xml" && nvram=1
  if (( nvram )); then virsh undefine "$domain" --nvram; else virsh undefine "$domain"; fi
  p2pnet_migrate_json_result "$domain" "$dest" "$profile" "$dirty" "$bw" "$downtime" "$elapsed" "$processed" true ''
  printf 'mptcp: n/a (libvirt-managed QEMU); multifd connections provide multi-stream\n'
  log INFO "migration of $domain to $dest completed; see $logf"
}

p2pnet_migrate_main() {
  local action=${1:-}; shift || true
  case $action in
    plan) p2pnet_migrate_plan "$@";;
    run) p2pnet_migrate_run "$@";;
    *) die 'usage: p2pnet migrate {plan DOMAIN DEST [--json]|run DOMAIN DEST [options]}' ;;
  esac
}
p2pnet_migrate_main "$@"
