#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=common.sh
source /opt/p2pnet/libexec/common.sh
P2P_COMPONENT=health

P2PNET_BIN=/opt/p2pnet/bin/p2pnet
MPTCP_EXEC=/opt/p2pnet/libexec/mptcp-exec
JSON_MODE=0
RESULTS_FILE=''
HEALTH_FAILED=0
PEER_ROWS=''

health_usage() { printf 'usage: p2pnet health [--json] [--quick]\n' >&2; exit 2; }

# One line per check: STATUS CHECK VALUE TARGET DETAIL. VALUE and TARGET are single tokens; DETAIL is the
# rest of the line. Tabs and line breaks never reach the output, so the text and --json forms stay parseable.
emit() {
    local status=$1 name=$2 value=${3:-} target=${4:-} detail=${5:-}
    value=${value//[[:space:]]/_}; target=${target//[[:space:]]/_}; detail=${detail//[$'\t\r\n']/ }
    [[ -n $value ]] || value=-
    [[ -n $target ]] || target=-
    [[ -n $detail ]] || detail=-
    if [[ $status == FAIL ]]; then HEALTH_FAILED=1; fi
    if (( JSON_MODE )); then
        printf '%s\t%s\t%s\t%s\t%s\n' "$status" "$name" "$value" "$target" "$detail" >>"$RESULTS_FILE"
    else
        printf '%s %s %s %s %s\n' "$status" "$name" "$value" "$target" "$detail"
    fi
}

# Runs a p2pnet subcommand with the 20 s check timeout; sets RUN_OUT (stdout+stderr) and RUN_RC.
run_check() {
    RUN_RC=0
    RUN_OUT=$(timeout 20 "$@" 2>&1 </dev/null) || RUN_RC=$?
}

# Lines of RUN_OUT matching an extended regex, joined with "; ".
matching_lines() { grep -E -- "$1" <<<"$RUN_OUT" | paste -sd ';' - || true; }

check_handshake() {
    local dev=$1 name=$2 pubkey=$3 peer_ip=$4 inventory_endpoint=$5 latest age current
    latest=$(timeout 20 wg show "$dev" latest-handshakes 2>/dev/null | awk -v key="$pubkey" '$1 == key {print $2; exit}' || true)
    if [[ ! $latest =~ ^[0-9]+$ ]] || (( latest == 0 )); then
        emit FAIL "$name" never '<=180s' "no handshake with $peer_ip on $dev"
        return 0
    fi
    age=$(( $(date +%s) - latest ))
    if (( age <= 180 )); then
        emit PASS "$name" "${age}s" '<=180s' "$dev $peer_ip"
    else
        emit FAIL "$name" "${age}s" '<=180s' "stale WireGuard handshake on $dev; check NAT traversal and endpoint reachability"
    fi
    # Drift is only meaningful against a literal-IP inventory endpoint (DNS names legitimately resolve anew).
    [[ $inventory_endpoint =~ ^([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+|\[[0-9A-Fa-f:.]+\]):[0-9]+$ ]] || return 0
    current=$(timeout 20 wg show "$dev" endpoints 2>/dev/null | awk -v key="$pubkey" '$1 == key {print $2; exit}' || true)
    if [[ -n $current && $current != '(none)' && $current != "$inventory_endpoint" ]]; then
        emit WARN "${name/.handshake./.endpoint.}" "$current" "$inventory_endpoint" 'endpoint drift (NAT rebinding or roaming)'
    fi
}

check_iperf() {
    local peer=$1 ovl4=$2 target=$3 rtt json errfile rc=0 mbit reason
    if [[ ! $target =~ ^[0-9]+$ ]]; then
        emit FAIL "iperf.overlay.$peer" unavailable - 'no inventory path target'
        return 0
    fi
    rtt=$(timeout 20 ping -n -c 1 -W 2 "$ovl4" 2>/dev/null | sed -n 's/.* = [^/]*\/\([^/]*\)\/.*/\1/p' || true)
    errfile=$(mktemp)
    # -O 1 drops the slow-start second: the target is 95 % of the steady-state overlay ceiling.
    json=$(timeout 20 "$MPTCP_EXEC" /usr/bin/iperf3 -c "$ovl4" -P 4 -O 1 -t 3 -J 2>"$errfile" </dev/null) || rc=$?
    mbit=$(jq -r '.end.sum_received.bits_per_second // empty | . / 1000000 | floor' <<<"$json" 2>/dev/null || true)
    if (( rc != 0 )) || [[ ! $mbit =~ ^[0-9]+$ ]]; then
        reason=$(jq -r '.error // empty' <<<"$json" 2>/dev/null || true)
        [[ -n $reason ]] || reason=$(grep -v '^p2pnet: MPTCP unavailable' "$errfile" | paste -sd ';' - || true)
        rm -f "$errfile"
        emit FAIL "iperf.overlay.$peer" unavailable ">=${target}Mbit" "${reason:-iperf3 exited $rc}"
        return 0
    fi
    rm -f "$errfile"
    if (( mbit >= target )); then
        emit PASS "iperf.overlay.$peer" "${mbit}Mbit" ">=${target}Mbit" "RTT=${rtt:-unknown}ms"
    else
        emit FAIL "iperf.overlay.$peer" "${mbit}Mbit" ">=${target}Mbit" "RTT=${rtt:-unknown}ms; overlay throughput below the inventory-derived target"
    fi
}

check_mptcp() {
    local peer=$1 dual=$2 line subflows
    if [[ $dual != 1 ]]; then
        emit SKIP "mptcp.$peer" N/A - 'single-WAN pair'
        return 0
    fi
    run_check "$P2PNET_BIN" mptcp verify --peer "$peer"
    if grep -q '^SKIP mptcp ' <<<"$RUN_OUT"; then
        emit SKIP "mptcp.$peer" N/A 'subflows>=1' 'mptcp component not installed'
        return 0
    fi
    line=$(matching_lines "^(PASS|FAIL) mptcp\\.${peer} ")
    subflows=$(sed -nE 's/.*subflows=([0-9]+).*/\1/p' <<<"$line" | head -n1)
    if (( RUN_RC == 0 )) && [[ $line == PASS* ]]; then
        emit PASS "mptcp.$peer" "subflows=${subflows:-?}" 'subflows>=1' "$line"
    else
        line=$(matching_lines '^FAIL ')
        [[ -n $line ]] || line=$(tail -n 3 <<<"$RUN_OUT" | paste -sd ';' -)
        emit FAIL "mptcp.$peer" "subflows=${subflows:-?}" 'subflows>=1' "$line"
    fi
}

check_peers() {
    local quick=$1 peer ovl4 endpoint pubkey plane2 plane2key path target dual plane2_name
    while IFS=$'\t' read -r -u 3 peer ovl4 _ endpoint pubkey plane2 plane2key _; do
        [[ -n $peer ]] || continue
        check_handshake wg0 "wg.handshake.$peer" "$pubkey" "$ovl4" "$endpoint"
        # This node's wg1 carries each peer's wg0 identity.
        if [[ -n ${P2P_PLANE2_IP:-} ]]; then
            check_handshake wg1 "wg.plane2.handshake.$peer" "$pubkey" "$ovl4" "$endpoint"
        fi
        # The peer's wg1 identity is an extra peer on this node's wg0; its wan2 endpoint is not in the peer table.
        if [[ $plane2 != - ]]; then
            plane2_name="wg.plane2.handshake.$peer"
            [[ -z ${P2P_PLANE2_IP:-} ]] || plane2_name="wg.plane2.handshake.$peer.wan2"
            check_handshake wg0 "$plane2_name" "$plane2key" "$plane2" -
        fi
        if (( quick )); then
            emit SKIP "iperf.overlay.$peer" N/A - 'skipped with --quick'
            emit SKIP "mptcp.$peer" N/A - 'skipped with --quick'
            continue
        fi
        path=$(inv path --from "$P2P_NODE" --to "$peer" 2>/dev/null </dev/null || true)
        target=$(sed -n 's/^P2P_PATH_OVERLAY_TARGET_MBIT=//p' <<<"$path")
        dual=$(sed -n 's/^P2P_PATH_DUALWAN=//p' <<<"$path")
        check_iperf "$peer" "$ovl4" "$target"
        check_mptcp "$peer" "$dual"
    done 3<<<"$PEER_ROWS"
}

check_zrepl() {
    local job dataset targets interval target_node name limit peer_row target_ip target_root state rc latest stamp epoch lag
    local -a target_nodes
    while IFS=$'\t' read -r -u 3 job dataset targets interval _; do
        [[ -n $job ]] || continue
        [[ $interval =~ ^[0-9]+$ ]] || interval=15
        limit=$(( (interval * 2 + 5) * 60 ))
        IFS=, read -r -a target_nodes <<<"$targets"
        for target_node in "${target_nodes[@]}"; do
            name="zrepl.$job.$target_node"
            peer_row=$(awk -F '\t' -v n="$target_node" '$1 == n {print; exit}' <<<"$PEER_ROWS")
            target_ip=$(cut -f2 <<<"$peer_row")
            target_root=$(cut -f12 <<<"$peer_row")
            if [[ -z $target_ip || -z $target_root || $target_root == - ]]; then
                emit FAIL "$name" unknown "<=${limit}s" 'target has no overlay address or replica root in the inventory'
                continue
            fi
            # Same receiver path as zrepl: <replica_root>/<source node>/<dataset without its pool>.
            rc=0
            state=$(timeout 20 "$MPTCP_EXEC" /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$target_ip" state "$target_root/$P2P_NODE/${dataset#*/}" 2>&1 </dev/null) || rc=$?
            if (( rc != 0 )); then
                emit FAIL "$name" unavailable "<=${limit}s" "state query failed: $state"
                continue
            fi
            latest=$(sed -n 's/^latest=//p' <<<"$state" | head -n1)
            stamp=${latest#p2pnet-}
            epoch=''
            if [[ $stamp =~ ^([0-9]{8})T([0-9]{2})([0-9]{2})([0-9]{2})Z$ ]]; then
                epoch=$(date -u -d "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}:${BASH_REMATCH[3]}:${BASH_REMATCH[4]}" +%s 2>/dev/null || true)
            fi
            if [[ ! $epoch =~ ^[0-9]+$ ]]; then
                emit FAIL "$name" never "<=${limit}s" "no p2pnet snapshot on $target_node (latest=${latest:--})"
                continue
            fi
            lag=$(( $(date +%s) - epoch ))
            if (( lag <= limit )); then
                emit PASS "$name" "${lag}s" "<=${limit}s" "$latest"
            else
                emit FAIL "$name" "${lag}s" "<=${limit}s" "replication snapshot $latest is stale"
            fi
        done
    done 3< <(inv jobs --node "$P2P_NODE")
}

check_migrate() {
    local file=/var/lib/p2pnet/migrate/last.json ok downtime dest error path limit=1000 limit_note=''
    if [[ ! -e $file ]]; then
        emit INFO migrate.last N/A - 'no migration recorded'
        return 0
    fi
    ok=$(jq -r '.ok == true' "$file" 2>/dev/null || true)
    downtime=$(jq -r '.downtime_ms // empty' "$file" 2>/dev/null || true)
    dest=$(jq -r '.dest // empty' "$file" 2>/dev/null || true)
    error=$(jq -r '.error // empty' "$file" 2>/dev/null || true)
    # The limit is the one the migration ran with: the inventory path's planned downtime to that destination.
    path=''
    [[ -z $dest ]] || path=$(inv path --from "$P2P_NODE" --to "$dest" 2>/dev/null </dev/null || true)
    if [[ $(sed -n 's/^P2P_PATH_DOWNTIME_MS=//p' <<<"$path") =~ ^[0-9]+$ ]]; then
        limit=$(sed -n 's/^P2P_PATH_DOWNTIME_MS=//p' <<<"$path")
    else
        limit_note="; destination ${dest:-unknown} not in inventory, default limit"
    fi
    if [[ $ok != true ]]; then
        emit FAIL migrate.last "${downtime:+${downtime}ms}" "<=${limit}ms" "migration to ${dest:-unknown} failed: ${error:-unreadable $file}"
    elif [[ ! $downtime =~ ^[0-9]+$ ]]; then
        emit FAIL migrate.last unknown "<=${limit}ms" "migration to ${dest:-unknown} recorded no downtime_ms"
    elif (( downtime <= limit )); then
        emit PASS migrate.last "${downtime}ms" "<=${limit}ms" "to ${dest:-unknown}${limit_note}"
    else
        emit FAIL migrate.last "${downtime}ms" "<=${limit}ms" "downtime to ${dest:-unknown} exceeded the planned limit${limit_note}"
    fi
}

# Per-class bytes/drops, e.g. "1:10=1234B/0drop 1:20=...".
qos_class_summary() {
    tc -s class show dev "$1" 2>/dev/null | awk '
        $1 == "class" && $2 == "htb" { cls = $3; next }
        $1 == "Sent" && cls != "" { drops = $7; sub(/,$/, "", drops); printf "%s%s=%sB/%sdrop", sep, cls, $2, drops; sep = " "; cls = "" }' || true
}

check_qos() {
    local dev failures
    run_check "$P2PNET_BIN" qos verify
    for dev in wg0 ${P2P_WAN2_IF:+wg1}; do
        if grep -q '^SKIP qos ' <<<"$RUN_OUT"; then
            emit SKIP "qos.$dev" N/A - 'qos component not installed'
            continue
        fi
        failures=$(matching_lines "^FAIL qos\\.${dev}\\.")
        if [[ -z $failures ]] && ! grep -q "^PASS qos\\.${dev}\\.root " <<<"$RUN_OUT"; then
            failures="qos verify did not check $dev: $(tail -n 3 <<<"$RUN_OUT" | paste -sd ';' -)"
        fi
        if [[ -n $failures ]]; then
            emit FAIL "qos.$dev" failed 'htb-rates-within-1%' "$failures"
        else
            emit PASS "qos.$dev" ok 'htb-rates-within-1%' "$(qos_class_summary "$dev")"
        fi
    done
}

check_l2() {
    local fdb_fail mtu_fail mtu other=''
    run_check "$P2PNET_BIN" l2 verify
    if grep -q '^SKIP l2 ' <<<"$RUN_OUT"; then
        emit SKIP l2.fdb N/A - 'l2 component not installed'
        emit SKIP l2.mtu N/A - 'l2 component not installed'
        return 0
    fi
    fdb_fail=$(matching_lines '^FAIL l2\.(fdb|bridge|vxlan) ')
    mtu_fail=$(matching_lines '^FAIL l2\.(bridge_mtu|vxlan_mtu) ')
    if (( RUN_RC != 0 )) && [[ -z $fdb_fail$mtu_fail ]]; then other=$(tail -n 3 <<<"$RUN_OUT" | paste -sd ';' -); fi
    if [[ -n $fdb_fail$other ]]; then emit FAIL l2.fdb failed 'flood-list==peers' "$fdb_fail$other"
    else emit PASS l2.fdb ok 'flood-list==peers' "$(matching_lines '^PASS l2\.fdb ')"; fi
    mtu=$(sed -n 's/^PASS l2\.vxlan_mtu //p' <<<"$RUN_OUT" | head -n1)
    if [[ -n $mtu_fail$other ]]; then emit FAIL l2.mtu failed "${P2P_VXLAN_MTU:-?}" "$mtu_fail$other"
    else emit PASS l2.mtu "${mtu:-$P2P_VXLAN_MTU}" "$P2P_VXLAN_MTU" 'bridge and vxlan0'; fi
}

check_tune() {
    local cc qdisc
    run_check "$P2PNET_BIN" tune verify
    cc=$(sysctl -n net.ipv4.tcp_congestion_control 2>/dev/null || printf '?')
    qdisc=$(sysctl -n net.core.default_qdisc 2>/dev/null || printf '?')
    if grep -q '^SKIP tune ' <<<"$RUN_OUT"; then
        emit SKIP tune.cc "$cc/$qdisc" bbr/fq 'tune component not installed'
    elif (( RUN_RC == 0 )); then
        emit PASS tune.cc "$cc/$qdisc" bbr/fq "$(matching_lines '^PASS qdisc\.')"
    else
        emit FAIL tune.cc "$cc/$qdisc" bbr/fq "$(matching_lines '^FAIL ')"
    fi
}

check_dedup() {
    local out ratio detail
    if out=$(timeout 20 "$P2PNET_BIN" dedup stats --json 2>/dev/null </dev/null) \
        && ratio=$(jq -er 'if .ratio == null then "N/A" else (.ratio * 1000 | floor / 1000 | tostring) end' <<<"$out" 2>/dev/null); then
        detail=$(jq -r '"logical=\(.logical_bytes) stored=\(.stored_bytes)"' <<<"$out" 2>/dev/null || true)
        emit INFO dedup.ratio "$ratio" logical/stored "$detail"
    else
        emit INFO dedup.ratio N/A logical/stored 'repository not initialized or unreadable'
    fi
}

write_json() {
    "$P2P_PY" - "$RESULTS_FILE" <<'PY'
import json
import sys
rows = []
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        status, name, value, target, detail = line.rstrip("\n").split("\t", 4)
        rows.append({"status": status, "check": name, "value": value, "target": target, "detail": detail})
print(json.dumps({"ok": all(row["status"] != "FAIL" for row in rows), "checks": rows}, separators=(",", ":")))
PY
}

run_health() {
    local quick=0 arg
    for arg in "$@"; do
        case $arg in
            --quick) quick=1 ;;
            --json) JSON_MODE=1 ;;
            *) health_usage ;;
        esac
    done
    load_env
    need_cmd wg jq tc timeout
    if (( JSON_MODE )); then
        RESULTS_FILE=$(mktemp)
        trap 'rm -f -- "$RESULTS_FILE"' EXIT
    fi
    PEER_ROWS=$(inv peers --node "$P2P_NODE")
    check_peers "$quick"
    check_zrepl
    check_migrate
    check_qos
    check_l2
    check_tune
    check_dedup
    if (( JSON_MODE )); then write_json; fi
    (( HEALTH_FAILED == 0 ))
}

run_health "$@"
