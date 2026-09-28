#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "${P2P_COMMON:-/opt/p2pnet/libexec/common.sh}"

DEDUP_ROOT=/var/lib/p2pnet/public/restic
DEDUP_DIR=/var/lib/p2pnet/dedup
RESTIC_PASSWORD=/etc/p2pnet/secrets/restic.pass
BULK_KEY=/var/lib/p2pnet/bulk/.ssh/id_ed25519

# restic as p2pbulk. P2PNET_SSH_KEY is only read by ssh-p2p when an sftp.command option is given.
restic_bulk() {
    runuser -u p2pbulk -- env P2PNET_SSH_KEY="$BULK_KEY" RESTIC_PASSWORD_FILE="$RESTIC_PASSWORD" RESTIC_CACHE_DIR=/var/cache/p2pnet/restic restic "$@"
}

restic_local() { restic_bulk -r "$DEDUP_ROOT" "$@"; }

# total_size of `stats --mode MODE` over every snapshot in the local repository.
local_total_size() {
    restic_local stats --mode "$1" --json | "$P2P_PY" -c 'import json,sys; print(json.load(sys.stdin)["total_size"])'
}

# chunker_polynomial of the repository selected by the given restic global options.
chunker_polynomial() {
    restic_bulk "$@" cat config | "$P2P_PY" -c 'import json,sys; print(json.load(sys.stdin)["chunker_polynomial"])'
}

restic_remote_args() {
    local address=$1
    REMOTE_REPO="sftp:p2pbulk@$address:/restic"
    REMOTE_OPTIONS="sftp.command=/opt/p2pnet/libexec/mptcp-exec /opt/p2pnet/libexec/ssh-p2p -p 2223 p2pbulk@$address -s sftp"
}

peer_address() {
    local wanted=$1 peer address
    while IFS=$'\t' read -r peer address _; do
        if [[ $peer == "$wanted" ]]; then
            [[ -n $address && $address != - ]] || die "peer $wanted has no overlay IPv4"
            printf '%s' "$address"
            return
        fi
    done < <(inv peers --node "$(self_node)")
    die "unknown peer: $wanted"
}

dedup_install() {
    require_root
    load_env
    need_cmd restic runuser
    [[ -r $RESTIC_PASSWORD ]] || die "missing $RESTIC_PASSWORD — run p2pnet keys --restic-password and distribute it to every node"
    ensure_dir "$DEDUP_ROOT" 0700 p2pbulk:p2pbulk
    ensure_dir "$DEDUP_DIR" 0750 root:p2pbulk
    ensure_dir /var/cache/p2pnet/restic 0750 p2pbulk:p2pbulk
    mark_installed dedup
    printf 'p2pnet[dedup] install complete: changed=%s\n' "$CHANGED"
}

dedup_verify() {
    if ! is_installed dedup; then check SKIP dedup 'not installed'; verify_done "${P2P_VERIFY_FAILED:-0}"; return; fi
    if [[ -f $RESTIC_PASSWORD ]] && [[ $(stat -c '%U:%G:%a' "$RESTIC_PASSWORD") == root:p2pbulk:640 ]]; then check PASS dedup.password "$RESTIC_PASSWORD root:p2pbulk 0640"; else check FAIL dedup.password 'password file missing or permissions differ'; fi
    if restic_local snapshots --json >/dev/null 2>&1; then check PASS dedup.repository 'snapshots command succeeded'; else check FAIL dedup.repository 'repository unavailable or not initialized'; fi
    verify_done "${P2P_VERIFY_FAILED:-0}"
}

# Re-running init against an existing repository is a no-op. For --from, the existing repository must
# already share the source's chunker parameters, otherwise identical data would chunk differently.
dedup_init() {
    require_root
    case ${1:-} in
        --seed)
            (($# == 1)) || die 'usage: p2pnet dedup init --seed'
            if [[ -e $DEDUP_ROOT/config ]]; then
                restic_local cat config >/dev/null || die "$DEDUP_ROOT holds a repository that $RESTIC_PASSWORD cannot open"
                printf 'p2pnet[dedup] repository already initialized: %s\n' "$DEDUP_ROOT"
                return
            fi
            restic_local init
            ;;
        --from)
            (($# == 2)) || die 'usage: p2pnet dedup init --from NODE'
            load_env
            local address local_poly source_poly
            address=$(peer_address "$2")
            restic_remote_args "$address"
            if [[ -e $DEDUP_ROOT/config ]]; then
                local_poly=$(chunker_polynomial -r "$DEDUP_ROOT")
                source_poly=$(chunker_polynomial -r "$REMOTE_REPO" -o "$REMOTE_OPTIONS")
                [[ $local_poly == "$source_poly" ]] || die "$DEDUP_ROOT uses chunker polynomial $local_poly but $2 uses $source_poly; run p2pnet dedup uninstall --purge, dedup install, then init --from $2"
                printf 'p2pnet[dedup] repository already initialized with the chunker parameters of %s: %s\n' "$2" "$DEDUP_ROOT"
                return
            fi
            restic_local -o "$REMOTE_OPTIONS" init --copy-chunker-params --from-repo "$REMOTE_REPO" --from-password-file "$RESTIC_PASSWORD"
            ;;
        *) die 'usage: p2pnet dedup init --seed|--from NODE';;
    esac
}

dedup_ingest() {
    (($# >= 1)) || die 'usage: p2pnet dedup ingest PATH [--name NAME]'
    local path=$1; shift; local name
    name=$(basename -- "$path")
    while (($#)); do case $1 in --name) (($# >= 2)) || die '--name requires value'; name=$2; shift 2;; *) die "unknown ingest option: $1";; esac; done
    [[ -f $path ]] || die "not a file: $path"
    [[ $name =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid name: $name"
    mkdir -p "$DEDUP_DIR"
    local output=$DEDUP_DIR/.ingest-$$.json size
    size=$(stat -c %s "$path")
    if ! cat -- "$path" | restic_local backup --stdin --stdin-filename "$name" --tag p2pnet-image --tag "name=$name" --json >"$output"; then rm -f "$output"; return 1; fi
    /usr/bin/python3 - "$output" "$DEDUP_DIR/ingest-$name.json" "$name" "$size" <<'PY'
import json, os, sys, tempfile
source, target, name, size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
records = [json.loads(line) for line in open(source, encoding="utf-8") if line.strip()]
summary = next((r for r in reversed(records) if r.get("message_type") == "summary"), {})
value = {"name": name, "image_bytes": size, "data_added": summary.get("data_added", 0)}
directory = os.path.dirname(target)
fd, tmp = tempfile.mkstemp(dir=directory, prefix=".ingest-")
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(value, stream, sort_keys=True); stream.write("\n")
os.replace(tmp, target)
print("new unique bytes: %s" % value["data_added"])
PY
    rm -f "$output"
}

dedup_fetch() {
    (($# >= 3)) || die 'usage: p2pnet dedup fetch NAME --from NODE [--output PATH]'
    local name=$1; shift
    [[ $1 == --from && $# -ge 2 ]] || die 'fetch requires --from NODE'
    local peer=$2; shift 2
    load_env
    local output=$P2P_IMAGES_DIR/$name
    while (($#)); do case $1 in --output) (($# >= 2)) || die '--output requires path'; output=$2; shift 2;; *) die "unknown fetch option: $1";; esac; done
    [[ $name =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid name: $name"
    local address before after transferred size tmp
    address=$(peer_address "$peer"); restic_remote_args "$address"
    before=$(local_total_size raw-data)
    restic_local -o "$REMOTE_OPTIONS" copy --from-repo "$REMOTE_REPO" --from-password-file "$RESTIC_PASSWORD" --tag "name=$name" latest
    after=$(local_total_size raw-data)
    transferred=$((after-before)); mkdir -p "$(dirname -- "$output")" "$DEDUP_DIR"
    # Dump into a fresh root-owned file and rename it into place: never leaves a truncated image behind,
    # and fs.protected_regular=2 forbids O_CREAT on an existing p2pbulk-owned file in sticky dirs like /tmp.
    tmp=$(mktemp "$output.XXXXXX")
    if ! restic_local dump --tag "name=$name" latest "/$name" >"$tmp"; then rm -f -- "$tmp"; die "restic dump of $name failed"; fi
    chmod 0644 "$tmp"
    chown p2pbulk:p2pbulk "$tmp"
    mv -f -- "$tmp" "$output"
    size=$(stat -c %s "$output")
    /usr/bin/python3 - "$DEDUP_DIR/last-fetch.json" "$name" "$peer" "$size" "$transferred" <<'PY'
import json, os, sys, tempfile
path, name, peer, size, moved = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
value = {"name": name, "from": peer, "image_bytes": size, "transferred_bytes": moved, "savings_pct": (1 - moved / size) * 100 if size else 0}
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".last-fetch-")
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(value, stream, sort_keys=True); stream.write("\n")
os.replace(tmp, path)
print(json.dumps(value, sort_keys=True))
PY
}

dedup_stats() {
    local json=0
    [[ ${1:-} != --json ]] || json=1
    local logical stored
    logical=$(local_total_size restore-size)
    stored=$(local_total_size raw-data)
    /usr/bin/python3 - "$logical" "$stored" "$json" <<'PY'
import json, sys
logical, stored, as_json = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
value = {"logical_bytes": logical, "stored_bytes": stored, "ratio": logical / stored if stored else None}
if as_json: print(json.dumps(value, sort_keys=True))
else: print("logical=%d stored=%d ratio=%s" % (logical, stored, "n/a" if value["ratio"] is None else "%.3f" % value["ratio"]))
PY
}

dedup_uninstall() {
    require_root
    local purge=0; [[ ${1:-} != --purge ]] || purge=1
    mark_uninstalled dedup
    if ((purge)); then rm -rf -- "$DEDUP_ROOT" /var/cache/p2pnet/restic "$DEDUP_DIR"; fi
    printf 'p2pnet[dedup] uninstall complete\n'
}

dedup_main() {
    (($#)) || die 'usage: p2pnet dedup install|verify|uninstall|init|ingest|fetch|stats'
    local action=$1; shift
    case $action in
        install) dedup_install;; verify) dedup_verify;; uninstall) dedup_uninstall "${1:-}";; init) dedup_init "$@";; ingest) dedup_ingest "$@";; fetch) dedup_fetch "$@";; stats) dedup_stats "${1:-}";; *) die "unknown dedup action: $action";;
    esac
}

dedup_main "$@"
