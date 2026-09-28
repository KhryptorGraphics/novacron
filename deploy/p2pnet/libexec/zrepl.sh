#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck disable=SC1090
source "${P2PNET_COMMON:-/opt/p2pnet/libexec/common.sh}"

ZREPL_SERVICE=/etc/systemd/system/p2pnet-zrepl@.service
ZREPL_TIMER=/etc/systemd/system/p2pnet-zrepl@.timer
ZREPL_LOG=/var/log/p2pnet/zrepl.log
ZREPL_STATUS_DIR=/var/lib/p2pnet/zrepl

zrepl_log() {
    local message=$1
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$message" | tee -a "$ZREPL_LOG" >&2
    logger -t p2pnet-zrepl -- "$message" || true
}

zrepl_peer_values() {
    local wanted=$1 name ovl4 ovl6 replica
    while IFS=$'\t' read -r name ovl4 ovl6 _ _ _ _ _ _ _ _ replica; do
        [[ $name == "$wanted" ]] && { printf '%s\t%s\t%s\n' "$ovl4" "$replica" "$ovl6"; return 0; }
    done < <(inv peers --node "${P2P_NODE}")
    return 1
}

zrepl_jobs() { inv jobs --node "${P2P_NODE}"; }
zrepl_valid_dataset() {
    local value=$1 component
    [[ $value =~ ^[A-Za-z0-9_.:-]+(/[A-Za-z0-9_.:-]+)*$ ]] || return 1
    local IFS=/
    local -a components
    read -r -a components <<< "$value"
    for component in "${components[@]}"; do
        [[ $component != . && $component != .. ]] || return 1
    done
}

zrepl_install() {
    require_root
    load_env
    local changed=0 root job targets interval target peer_info target_replica
    local -a target_array
    root=${P2P_REPLICA_ROOT:-}
    if [[ -n $root ]]; then
        zrepl_valid_dataset "$root" || die "unsafe replica root $root"
        need_cmd zfs
        if ! zfs list -H -o name -- "$root" >/dev/null 2>&1; then
            zfs create -p -o canmount=off -o readonly=on -- "$root"
            ((changed+=1))
        fi
        if ! zfs allow "$root" 2>/dev/null | grep -Fq p2prepl; then
            ((changed+=1))
        fi
        zfs allow -u p2prepl create,receive,mount,destroy,hold,release "$root"
    fi
    # shellcheck disable=SC2153
    local units_before=$CHANGED
    install_unit "$P2P_PREFIX/systemd/p2pnet-zrepl@.service"
    install_unit "$P2P_PREFIX/systemd/p2pnet-zrepl@.timer"
    while IFS=$'\t' read -r job _ targets interval _ _; do
        [[ -n $job ]] || continue
        [[ $interval =~ ^[0-9]+$ && $interval -ge 1 && $interval -le 1440 ]] || die "invalid interval for zrepl job $job"
        need_cmd zfs
        local dropin_dir="/etc/systemd/system/p2pnet-zrepl@${job}.timer.d"
        install_file - "$dropin_dir/10-interval.conf" 0644 <<EOF
[Timer]
OnUnitInactiveSec=
OnUnitInactiveSec=${interval}min
EOF
        IFS=',' read -r -a target_array <<< "$targets"
        for target in "${target_array[@]}"; do
            peer_info=$(zrepl_peer_values "$target") || die "unknown replication target $target"
            IFS=$'\t' read -r _ target_replica _ <<< "$peer_info"
            zrepl_valid_dataset "$target_replica" || die "unsafe target replica root $target_replica"
        done
        (( CHANGED == units_before )) || UNITS_CHANGED=1
        reload_units
        if ! systemctl is-enabled --quiet "p2pnet-zrepl@${job}.timer" || ! systemctl is-active --quiet "p2pnet-zrepl@${job}.timer"; then
            ((changed+=1))
        fi
        systemctl enable --now "p2pnet-zrepl@${job}.timer"
        # A re-armed definition only takes effect for an active timer after restart.
        (( CHANGED == units_before )) || systemctl restart "p2pnet-zrepl@${job}.timer"
    done < <(zrepl_jobs)
    reload_units
    mark_installed zrepl
# shellcheck disable=SC2153
    printf 'p2pnet[zrepl] install complete: changed=%s\n' "$((CHANGED + changed))"
}

zrepl_run_job() {
    local job_name=$1 dataset keep_last keep_daily name job_dataset job_targets
    local source=$P2P_NODE found=0
    while IFS=$'\t' read -r name job_dataset job_targets _ keep_last keep_daily; do
        if [[ $name == "$job_name" ]]; then
            dataset=$job_dataset; found=1; break
        fi
    done < <(zrepl_jobs)
    (( found )) || die "unknown zrepl job $job_name"
    zrepl_valid_dataset "$dataset" || die "unsafe source dataset $dataset"
    with_lock "zrepl-$job_name" zrepl_run_locked "$job_name" "$dataset" "$job_targets" "$keep_last" "$keep_daily" "$source"
}

zrepl_run_locked() {
    local job_name=$1 dataset=$2 targets=$3 keep_last=$4 keep_daily=$5 source=$6
    local snapshot target ovl4 replica ovl6 peer_info d rel remote_path latest token action from estimated start elapsed rc status state_output n send_error
    snapshot="p2pnet-$(date -u +%Y%m%dT%H%M%SZ)"
    local failed=0 status_file="$ZREPL_STATUS_DIR/$job_name.json" previous_success=null
    local -a send_cmd=() pipe_status=()
    mkdir -p "$ZREPL_STATUS_DIR"
    if [[ -r $status_file ]]; then
        previous_success=$(/usr/bin/python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])).get("last_success")))' "$status_file") || previous_success=null
    fi
    local -A target_ok=() target_latest=() target_error=() dataset_latest=()
    if ! zfs snapshot -r "$dataset@$snapshot"; then
        zrepl_log "job=$job_name dataset=$dataset action=snapshot result=error"
        alert zrepl "\"job\":\"$job_name\",\"error\":\"snapshot creation failed\""
        printf '{"job":"%s","last_run":"%s","last_success":%s,"ok":false,"targets":{}}\n' "$job_name" "$(date -u +%FT%TZ)" "$previous_success" > "$status_file"
        return 1
    fi
    local -a target_list
    IFS=',' read -r -a target_list <<< "$targets"
    for target in "${target_list[@]}"; do
        if ! peer_info=$(zrepl_peer_values "$target"); then
            target_ok[$target]=false; target_error[$target]='unknown target'; failed=1; continue
        fi
        IFS=$'\t' read -r ovl4 replica ovl6 <<< "$peer_info"
        [[ -n $replica && $replica != - ]] || { target_ok[$target]=false; target_error[$target]='target has no replica root'; failed=1; continue; }
        target_ok[$target]=true; target_error[$target]=''; target_latest[$target]='-'
        while IFS= read -r -u 3 d; do
            [[ -n $d ]] || continue
            rel=${d#*/}
            remote_path="$replica/$source/$rel"
            if ! zrepl_safe_receiver_path "$remote_path" "$replica" "$source"; then
                target_ok[$target]=false; target_error[$target]="unsafe receiver path $remote_path"; failed=1; break
            fi
            if ! state_output=$(/opt/p2pnet/libexec/mptcp-exec /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$ovl4" state "$remote_path" 2>/dev/null); then
                target_ok[$target]=false; target_error[$target]="state query failed for $remote_path"; failed=1; break
            fi
            token=-; latest=-
            while IFS= read -r state_line; do
                case $state_line in token=*) token=${state_line#token=};; latest=*) latest=${state_line#latest=};; esac
            done <<< "$state_output"
            dataset_latest["$target|$d"]=$latest
            action=incremental; from=-
            if [[ $latest == "$snapshot" ]]; then
                action=skip; target_latest[$target]=$snapshot
                dataset_latest["$target|$d"]=$snapshot
                zrepl_log "job=$job_name target=$target dataset=$d action=skip from=$snapshot to=$snapshot est_bytes=0 seconds=0 result=ok"
                continue
            elif [[ $token != - && -n $token ]]; then
                action=resume; from=$latest
                send_cmd=(zfs send -w -t "$token")
            elif [[ $latest == - ]]; then
                action=full; send_cmd=(zfs send -w "$d@$snapshot")
            elif zfs list -H -t snapshot -o name -- "$d@$latest" >/dev/null 2>&1; then
                from=$latest; send_cmd=(zfs send -w -i "$d@$latest" "$d@$snapshot")
            else
                target_ok[$target]=false
                target_error[$target]="no common snapshot with $target; reseed: zfs destroy -r $remote_path on $target"
                failed=1
                zrepl_log "job=$job_name target=$target dataset=$d action=none from=$latest to=$snapshot result=error: no common snapshot; replica preserved"
                break
            fi
            estimated=$(zfs send -nP "${send_cmd[@]:2}" 2>/dev/null | awk '$1 == "size" {print $2; exit}' || true)
            send_error=$(mktemp "${P2P_RUN}/zrepl-send.XXXXXX")
            start=$(date +%s)
            set +e
            "${send_cmd[@]}" 2>"$send_error" | mbuffer -q -s 128k -m 1G | /opt/p2pnet/libexec/mptcp-exec /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$ovl4" "recv $remote_path"
            pipe_status=("${PIPESTATUS[@]}")
            set -e
            rc=0
            for status in "${pipe_status[@]}"; do (( status == 0 )) || rc=$status; done
            if (( rc != 0 )) && [[ $action == resume ]] && grep -Fqi 'no longer exists' "$send_error"; then
                cat "$send_error" >&2
                zrepl_log "job=$job_name target=$target dataset=$d resume token expired; aborting receive and retrying incrementally"
                if ! /opt/p2pnet/libexec/mptcp-exec /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$ovl4" abort "$remote_path"; then
                    target_ok[$target]=false
                    target_error[$target]="expired resume token; receiver abort failed for $remote_path"
                    rm -f "$send_error"; failed=1; break
                fi
                if ! state_output=$(/opt/p2pnet/libexec/mptcp-exec /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$ovl4" state "$remote_path" 2>/dev/null); then
                    target_ok[$target]=false
                    target_error[$target]="expired resume token; state query failed for $remote_path"
                    rm -f "$send_error"; failed=1; break
                fi
                latest=-
                while IFS= read -r state_line; do case $state_line in latest=*) latest=${state_line#latest=};; esac; done <<< "$state_output"
                if [[ $latest != - ]] && zfs list -H -t snapshot -o name -- "$d@$latest" >/dev/null 2>&1; then
                    action=incremental; from=$latest
                    send_cmd=(zfs send -w -i "$d@$latest" "$d@$snapshot")
                    : > "$send_error"
                    set +e
                    "${send_cmd[@]}" 2>"$send_error" | mbuffer -q -s 128k -m 1G | /opt/p2pnet/libexec/mptcp-exec /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$ovl4" "recv $remote_path"
                    pipe_status=("${PIPESTATUS[@]}")
                    set -e
                    rc=0
                    for status in "${pipe_status[@]}"; do (( status == 0 )) || rc=$status; done
                else
                    target_ok[$target]=false
                    target_error[$target]="expired resume token and no common snapshot with $target; replica preserved"
                    rm -f "$send_error"; failed=1; break
                fi
            fi
            cat "$send_error" >&2
            rm -f "$send_error"
            elapsed=$(( $(date +%s) - start ))
            if (( rc == 0 )); then
                target_latest[$target]=$snapshot
                dataset_latest["$target|$d"]=$snapshot
                zrepl_log "job=$job_name target=$target dataset=$d action=$action from=$from to=$snapshot est_bytes=${estimated:-unknown} seconds=$elapsed result=ok"
            else
                target_ok[$target]=false; target_error[$target]="send/receive pipeline failed (status ${pipe_status[*]})"; failed=1
                zrepl_log "job=$job_name target=$target dataset=$d action=$action from=$from to=$snapshot est_bytes=${estimated:-unknown} seconds=$elapsed result=error:${pipe_status[*]}"
                break
            fi
        done 3< <(zfs list -H -o name -r -t filesystem,volume "$dataset" | sort)
        if [[ ${target_ok[$target]} == true ]]; then
            zrepl_remote_prune "$ovl4" "$replica/$source/${dataset#*/}" "$keep_last" "$keep_daily" || {
                target_ok[$target]=false; target_error[$target]='remote prune failed'; failed=1
            }
        fi
    done
    local -a source_snapshots=() destroy_names=()
    local source_dataset
    while IFS= read -r -u 3 source_dataset; do
        [[ -n $source_dataset ]] || continue
        local -a protect_names=()
        for target in "${target_list[@]}"; do
            local target_snapshot=${dataset_latest["$target|$source_dataset"]:-}
            [[ -n $target_snapshot && $target_snapshot != - ]] && protect_names+=(--protect "$source_dataset@$target_snapshot")
        done
        mapfile -t source_snapshots < <(zfs list -H -t snapshot -o name -s creation "$source_dataset" | while IFS= read -r n; do [[ $n == *@p2pnet-* ]] && printf '%s\n' "$n"; done)
        mapfile -t destroy_names < <(/usr/bin/python3 /opt/p2pnet/libexec/zrepl_retention.py "${source_snapshots[@]}" --keep-last "$keep_last" --keep-daily "$keep_daily" "${protect_names[@]}")
        for source_snap in "${destroy_names[@]}"; do zfs destroy "$source_snap" || failed=1; done
    done 3< <(zfs list -H -o name -r -t filesystem,volume "$dataset" | sort)
    local now json_targets='' entry
    now=$(date -u +%FT%TZ)
    for target in "${target_list[@]}"; do
        [[ -n $json_targets ]] && json_targets+=,
        printf -v entry '{"ok":%s,"latest":"%s","error":"%s"}' "${target_ok[$target]:-false}" "${target_latest[$target]:--}" "${target_error[$target]:-}"
        json_targets+="\"$target\":$entry"
        if [[ ${target_ok[$target]:-false} != true ]]; then alert zrepl "\"job\":\"$job_name\",\"target\":\"$target\",\"error\":\"${target_error[$target]:-failed}\""; fi
    done
    if (( failed != 0 )); then alert zrepl "\"job\":\"$job_name\",\"error\":\"replication or retention failed\""; fi
    if (( failed == 0 )); then
        printf '{"job":"%s","last_run":"%s","last_success":"%s","ok":true,"targets":{%s}}\n' "$job_name" "$now" "$now" "$json_targets" > "$status_file"
    else
        printf '{"job":"%s","last_run":"%s","last_success":%s,"ok":false,"targets":{%s}}\n' "$job_name" "$now" "$previous_success" "$json_targets" > "$status_file"
    fi
    return "$failed"
}

zrepl_safe_receiver_path() {
    local path=$1 replica=$2 source_node=$3 suffix component
    zrepl_valid_dataset "$replica" || return 1
    [[ $source_node =~ ^[a-z0-9][a-z0-9-]{0,30}$ ]] || return 1
    [[ $path == "$replica/$source_node/"* && $path != *'@'* ]] || return 1
    suffix=${path#"$replica/$source_node/"}
    [[ -n $suffix && $suffix != */ && $suffix != /* ]] || return 1
    local IFS=/
    local -a components
    read -r -a components <<< "$suffix"
    for component in "${components[@]}"; do
        [[ $component =~ ^[A-Za-z0-9_.:-]+$ && $component != . && $component != .. ]] || return 1
    done
}

zrepl_remote_prune() {
    local ip=$1 replica=$2 keep_last=$3 keep_daily=$4
    /opt/p2pnet/libexec/mptcp-exec /opt/p2pnet/libexec/ssh-p2p -p 2222 "p2prepl@$ip" prune "$replica" "$keep_last" "$keep_daily"
}

zrepl_status() {
    load_env
    shopt -s nullglob
    local file
    for file in "$ZREPL_STATUS_DIR"/*.json; do cat "$file"; done
}

zrepl_verify() {
    load_env
    is_installed zrepl || { check SKIP zrepl 'not installed'; verify_done "${P2P_VERIFY_FAILED:-0}"; }
    local job root
    root=${P2P_REPLICA_ROOT:-}
    if [[ -n $root ]]; then
        if zfs allow "$root" 2>/dev/null | grep -q 'p2prepl'; then check PASS "allow.$root" p2prepl; else check FAIL "allow.$root" missing-p2prepl; fi
    fi
    while IFS=$'\t' read -r job _ _ _ _ _; do
        [[ -n $job ]] || continue
        if systemctl is-enabled --quiet "p2pnet-zrepl@${job}.timer"; then check PASS "timer.$job" enabled; else check FAIL "timer.$job" not-enabled; fi
        if [[ -r "$ZREPL_STATUS_DIR/$job.json" ]]; then
            if /usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("ok") else 1)' "$ZREPL_STATUS_DIR/$job.json"; then check PASS "last.$job" successful; else check FAIL "last.$job" failed; fi
        else check WARN "last.$job" never-ran; fi
    done < <(zrepl_jobs)
    verify_done "${P2P_VERIFY_FAILED:-0}"
}

zrepl_uninstall() {
    require_root
    load_env
    local job root
    while IFS=$'\t' read -r job _ _ _ _ _; do
        [[ -n $job ]] || continue
        systemctl disable --now "p2pnet-zrepl@${job}.timer" 2>/dev/null || true
        rm -f "/etc/systemd/system/p2pnet-zrepl@${job}.timer.d/10-interval.conf"
        rmdir "/etc/systemd/system/p2pnet-zrepl@${job}.timer.d" 2>/dev/null || true
    done < <(zrepl_jobs)
    root=${P2P_REPLICA_ROOT:-}
    if [[ -n $root ]] && command -v zfs >/dev/null 2>&1 && zfs allow "$root" 2>/dev/null | grep -q p2prepl; then
        zfs unallow -u p2prepl "$root"
    fi
    rm -f "$ZREPL_SERVICE" "$ZREPL_TIMER"
    export UNITS_CHANGED=1
    reload_units
    mark_uninstalled zrepl
    # Snapshots and replicas are intentionally preserved, including with --purge.
}

zrepl_usage() {
    printf 'usage: p2pnet zrepl {install|verify|uninstall [--purge]|run JOB|status [--json]}\n' >&2
    exit 2
}

zrepl_main() {
    local action=${1:-}; [[ $# -gt 0 ]] && shift
    case $action in
        install) (($# == 0)) || zrepl_usage; zrepl_install;;
        verify) (($# == 0)) || zrepl_usage; zrepl_verify;;
        uninstall) [[ $# -eq 0 || ( $# -eq 1 && $1 == --purge ) ]] || zrepl_usage; zrepl_uninstall;;
        run) (($# == 1)) || zrepl_usage; load_env; zrepl_run_job "$1";;
        status) [[ $# -eq 0 || ( $# -eq 1 && $1 == --json ) ]] || zrepl_usage; zrepl_status;;
        *) zrepl_usage;;
    esac
}
zrepl_main "$@"
