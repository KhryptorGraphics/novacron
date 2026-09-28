#!/usr/bin/env bash
set -Eeuo pipefail

P2P_PREFIX=/opt/p2pnet
P2P_ETC=/etc/p2pnet
P2P_STATE=/var/lib/p2pnet
P2P_LOGDIR=/var/log/p2pnet
P2P_RUN=/run/p2pnet
P2P_PY=/usr/bin/python3
export P2P_LOGDIR
P2P_INVENTORY=${P2PNET_INVENTORY:-/etc/p2pnet/inventory.yaml}
CHANGED=0
UNITS_CHANGED=0
export CHANGED UNITS_CHANGED

inv() { "$P2P_PY" "$P2P_PREFIX/libexec/inventory.py" --inventory "$P2P_INVENTORY" "$@"; }

log() {
    local level=$1; shift
    local message=$*
    printf '%s p2pnet: %s\n' "$level" "$message" >&2
    if command -v logger >/dev/null 2>&1; then logger -t "p2pnet-${P2P_COMPONENT:-core}" -- "$level $message" || true; fi
}

die() { log ERROR "$*"; exit 1; }
require_root() { [[ ${EUID} -eq 0 ]] || die "must run as root"; }
need_cmd() {
    local command
    for command in "$@"; do command -v "$command" >/dev/null 2>&1 || die "missing $command — run: p2pnet deps install"; done
}

self_node() {
    if [[ -n ${P2PNET_NODE:-} ]]; then printf '%s\n' "$P2PNET_NODE"; return; fi
    [[ -r $P2P_ETC/node-name ]] || die "run p2pnet init --node NAME"
    local node
    IFS= read -r node < "$P2P_ETC/node-name" || true
    [[ -n $node ]] || die "run p2pnet init --node NAME"
    printf '%s\n' "$node"
}

load_env() {
    local node dest tmp
    if [[ -n ${P2PNET_NODE_ENV:-} ]]; then
        [[ -r $P2PNET_NODE_ENV ]] || die "node env fixture is unreadable: $P2PNET_NODE_ENV"
        set -a
        # shellcheck disable=SC1090
        . "$P2PNET_NODE_ENV"
        set +a
        return
    fi
    node=$(self_node)
    dest=$P2P_ETC/generated/node.env
    mkdir -p "${dest%/*}"
    tmp=$(mktemp "${dest}.XXXXXX")
    if ! inv env --node "$node" > "$tmp"; then rm -f "$tmp"; die "cannot generate node environment for $node"; fi
    install_file "$tmp" "$dest" 0644
    rm -f "$tmp"
    set -a
    # shellcheck disable=SC1090
    . "$dest"
    set +a
}

install_file() {
    local src=$1 dest=$2 mode=$3 owner=${4:-root:root} tmp mode_norm=${3#0}
    mkdir -p "${dest%/*}"
    tmp=$(mktemp "${dest}.XXXXXX")
    if [[ $src == - ]]; then cat > "$tmp"; else cat -- "$src" > "$tmp"; fi
    if [[ -f $dest ]] && cmp -s "$tmp" "$dest" && [[ $(stat -c '%a:%U:%G' "$dest") == "$mode_norm:${owner%:*}:${owner#*:}" ]]; then
        rm -f "$tmp"; return 0
    fi
    chmod "$mode" "$tmp"
    chown "$owner" "$tmp"
    mv -f -- "$tmp" "$dest"
    CHANGED=$((CHANGED + 1))
}
ensure_dir() {
    local path=$1 mode=$2 owner=${3:-root:root} changed=0 expected_mode=${2#0}
    if [[ ! -d $path ]]; then changed=1
    elif [[ $(stat -c '%a:%U:%G' "$path") != "$expected_mode:${owner%:*}:${owner#*:}" ]]; then changed=1
    fi
    install -d -o "${owner%:*}" -g "${owner#*:}" -m "$mode" "$path"
    if (( changed )); then CHANGED=$((CHANGED + 1)); fi
}

install_unit() {
    local file=$1 before=$CHANGED
    install_file "$file" "/etc/systemd/system/${file##*/}" 0644
    if (( CHANGED != before )); then UNITS_CHANGED=1; fi
}

reload_units() {
    if (( UNITS_CHANGED )); then systemctl daemon-reload; UNITS_CHANGED=0; fi
}

mark_installed() {
    local component=$1 marker="$P2P_STATE/state/installed/$1"
    mkdir -p "${marker%/*}"
    if [[ ! -e $marker ]]; then : > "$marker"; CHANGED=$((CHANGED + 1)); fi
}
mark_uninstalled() {
    local component=$1 marker="$P2P_STATE/state/installed/$1"
    if [[ -e $marker ]]; then rm -f -- "$marker"; CHANGED=$((CHANGED + 1)); fi
}
is_installed() { [[ -e $P2P_STATE/state/installed/$1 ]]; }

save_prev_once() {
    local component=$1 key=$2 value=$3 file="$P2P_STATE/state/$1.prev"
    mkdir -p "${file%/*}"
    touch "$file"
    if ! grep -Fq -- "${key}=" "$file"; then printf '%s=%s\n' "$key" "$value" >> "$file"; fi
}

restore_prev() {
    local component=$1 callback=$2 file="$P2P_STATE/state/$1.prev" key value
    [[ -r $file ]] || return 0
    while IFS='=' read -r key value; do
        [[ -n $key ]] || continue
        "$callback" "$key" "$value"
    done < "$file"
    rm -f -- "$file"
}

check() {
    printf '%s %s %s\n' "$1" "$2" "$3"
    if [[ $1 == FAIL ]]; then P2P_VERIFY_FAILED=1; fi
}
verify_done() {
    local status=${1:-${P2P_VERIFY_FAILED:-0}}
    [[ $status -eq 0 ]] || exit 1
}

with_lock() {
    local name=$1 status; shift
    mkdir -p "$P2P_RUN"
    exec {P2P_LOCK_FD}>"$P2P_RUN/$name.lock"
    if ! flock -n "$P2P_LOCK_FD"; then log INFO "already running: $name"; return 0; fi
    if "$@"; then status=0; else status=$?; fi
    flock -u "$P2P_LOCK_FD"
    return "$status"
}

alert() {
    local component=$1; shift
    [[ -n ${P2P_ALERT_WEBHOOK:-} ]] || return 0
    local fields=$* payload
    payload=$(printf '{"cluster":"%s","node":"%s","component":"%s","time":"%s",%s}' \
        "${P2P_CLUSTER:-}" "${P2P_NODE:-}" "$component" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$fields")
    if ! printf '%s' "$payload" | curl -fsS -m 10 -H 'Content-Type: application/json' --data @- "$P2P_ALERT_WEBHOOK" >/dev/null; then
        log WARN "alert delivery failed for $component"
    fi
}

run() {
    if [[ ${DRY_RUN:-0} == 1 ]]; then printf '+' >&2; printf ' %q' "$@" >&2; printf '\n' >&2; return 0; fi
    "$@"
}
