#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "${P2P_COMMON:-/opt/p2pnet/libexec/common.sh}"

SWARM_UNIT=/opt/p2pnet/systemd/p2pnet-swarmd.service
SWARM_STATE=/var/lib/p2pnet/swarm
SWARM_TORRENTS=/var/lib/p2pnet/public/torrents
SWARM_QUEUE=$SWARM_STATE/queue

swarm_peers() {
    inv peers --node "$(self_node)"
}

swarm_status() {
    local json=${1:-0}
    if [[ ! -r /run/p2pnet-swarm/status.json ]]; then
        if [[ $json == 1 ]]; then printf '{"time":null,"torrents":[]}\n'; else printf 'no swarm status available\n'; fi
        return
    fi
    if [[ $json == 1 ]]; then cat /run/p2pnet-swarm/status.json; else /usr/bin/python3 -c 'import json,sys; d=json.load(sys.stdin); print("NAME STATE PROGRESS PEERS DOWN_BPS UP_BPS ERROR"); [print("%s %s %.1f%% %s %s %s %s"%(x["name"],x["state"],x["progress"]*100,x["num_peers"],x["download_rate"],x["upload_rate"],x["error"])) for x in d["torrents"]]' </run/p2pnet-swarm/status.json; fi
}

swarm_install() {
    require_root
    load_env
    need_cmd systemctl runuser
    install_unit "$SWARM_UNIT"
    reload_units
    systemctl enable --now p2pnet-swarmd.service
    mark_installed swarm
    printf 'p2pnet[swarm] install complete: changed=%s\n' "${CHANGED:-0}"
}

swarm_verify() {
    if ! is_installed swarm; then check SKIP swarm 'not installed'; verify_done "${P2P_VERIFY_FAILED:-0}"; return; fi
    load_env
    if systemctl is-active --quiet p2pnet-swarmd.service; then check PASS swarm.unit active; else check FAIL swarm.unit inactive; fi
    # libtorrent may bind with a device scope, e.g. 10.77.0.3%wg0:6881.
    swarm_listening() { ss -Hltn 2>/dev/null | awk '{print $4}' | grep -Eq "^${P2P_OVL4//./\\.}(%[^:]+)?:6881$"; }
    # swarmd needs a few seconds after (re)start to import libtorrent and bind.
    local i
    for ((i=0; i<20; i++)); do
        swarm_listening && [[ -r /run/p2pnet-swarm/status.json ]] && break
        sleep 1
    done
    if swarm_listening; then check PASS swarm.listen "${P2P_OVL4}:6881"; else check FAIL swarm.listen "not listening on ${P2P_OVL4}:6881"; fi
    if [[ -r /run/p2pnet-swarm/status.json ]] && find /run/p2pnet-swarm/status.json -mmin -0.5 -print -quit | grep -q .; then check PASS swarm.status 'updated within 30 seconds'; else check FAIL swarm.status 'missing or older than 30 seconds'; fi
    verify_done "${P2P_VERIFY_FAILED:-0}"
}

swarm_publish() {
    require_root
    load_env
    (($# >= 1)) || die 'usage: p2pnet swarm publish PATH [--name NAME]'
    local source=$1; shift
    local name='' peer address tmp torrent
    while (($#)); do
        case $1 in --name) (($# >= 2)) || die '--name requires a value'; name=$2; shift 2;; *) die "unknown publish option: $1";; esac
    done
    [[ -f $source ]] || die "not a file: $source"
    name=${name:-$(basename -- "$source")}
    [[ $name =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid image name: $name"
    mkdir -p "$P2P_IMAGES_DIR" "$SWARM_TORRENTS"
    local data=$P2P_IMAGES_DIR/$name
    if [[ $(realpath -m -- "$source") != $(realpath -m -- "$data") ]]; then cp --reflink=auto --sparse=always -- "$source" "$data"; fi
    chown p2pbulk:p2pbulk "$data"
    torrent=$SWARM_TORRENTS/$name.torrent
    runuser -u p2pbulk -- /usr/bin/python3 /opt/p2pnet/libexec/swarm_make.py "$data" "$torrent" "$name"
    chown p2pbulk:p2pbulk "$torrent"
    local failed=0
    while IFS=$'\t' read -r peer address _; do
        [[ -n $peer && $address != - && -n $address ]] || continue
        tmp=$(mktemp)
        printf 'put %s /torrents/%s.torrent.tmp\nrename /torrents/%s.torrent.tmp /torrents/%s.torrent\n' "$torrent" "$name" "$name" "$name" >"$tmp"
        chmod 0644 "$tmp"
        if ! runuser -u p2pbulk -- env P2PNET_SSH_KEY=/var/lib/p2pnet/bulk/.ssh/id_ed25519 sftp -S /opt/p2pnet/libexec/ssh-p2p -P 2223 -b "$tmp" "p2pbulk@$address"; then log ERROR "failed publishing $name to $peer"; failed=1; fi
        rm -f -- "$tmp"
    done < <(swarm_peers)
    ((failed == 0)) || return 1
    printf 'published %s to all peers\n' "$name"
}

swarm_fetch() {
    require_root
    load_env
    (($# >= 1)) || die 'usage: p2pnet swarm fetch NAME [--wait] [--timeout S]'
    local name=$1; shift
    [[ $name =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid image name: $name"
    local wait=0 timeout_s=3600
    while (($#)); do case $1 in --wait) wait=1; shift;; --timeout) (($# >= 2)) || die '--timeout requires seconds'; timeout_s=$2; shift 2;; *) die "unknown fetch option: $1";; esac; done
    [[ $timeout_s =~ ^[1-9][0-9]*$ ]] || die 'timeout must be positive seconds'
    local torrent=$SWARM_TORRENTS/$name.torrent data=$P2P_IMAGES_DIR/$name
    [[ -f $torrent ]] || die "torrent not found: $torrent"
    if [[ -f $data && ! -e $SWARM_QUEUE/$name.torrent ]]; then printf 'already present: %s\n' "$data"; return; fi
    mkdir -p "$P2P_IMAGES_DIR" "$SWARM_QUEUE"
    install -o p2pbulk -g p2pbulk -m 0644 "$torrent" "$SWARM_QUEUE/$name.torrent"
    ((wait == 1)) || { printf 'queued %s\n' "$name"; return; }
    local start=$SECONDS row progress rate
    while ((SECONDS - start < timeout_s)); do
        if [[ -r /run/p2pnet-swarm/status.json ]]; then
            row=$(/usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); x=next((x for x in d["torrents"] if x["name"]==sys.argv[2]),None); print(str(x["progress"])+"\t"+str(x["download_rate"]) if x else "\t")' /run/p2pnet-swarm/status.json "$name")
            IFS=$'\t' read -r progress rate <<<"$row"
            if [[ -n $progress ]]; then printf '\r%s elapsed=%ss progress=%.1f%% rate=%s B/s' "$name" "$((SECONDS-start))" "$(/usr/bin/awk -v p="$progress" 'BEGIN {print p*100}')" "$rate"; fi
        fi
        # swarmd removes the queue entry on torrent_finished; the data file exists (preallocated) long before.
        [[ -e $SWARM_QUEUE/$name.torrent ]] || { printf '\ncomplete %s\n' "$data"; return; }
        sleep 2
    done
    printf '\ntimeout waiting for %s\n' "$name" >&2
    return 1
}

swarm_preseed() {
    load_env
    local name
    while IFS= read -r name; do [[ -n $name && ! -f $P2P_IMAGES_DIR/$name ]] && swarm_fetch "$name"; done < <(P2PNET_INVENTORY="$P2P_INVENTORY" /usr/bin/python3 -c 'import yaml,os; d=yaml.safe_load(open(os.environ["P2PNET_INVENTORY"])); [print(x) for x in d.get("swarm",{}).get("base_images",[])]')
}

swarm_uninstall() {
    require_root
    load_env
    local purge=0
    [[ ${1:-} != --purge ]] || purge=1
    systemctl disable --now p2pnet-swarmd.service 2>/dev/null || true
    rm -f /etc/systemd/system/p2pnet-swarmd.service
    UNITS_CHANGED=1
    reload_units
    mark_uninstalled swarm
    if ((purge)); then rm -rf -- "$P2P_IMAGES_DIR" /var/lib/p2pnet/public/torrents "$SWARM_QUEUE" "$SWARM_STATE/resume"; fi
    printf 'p2pnet[swarm] uninstall complete\n'
}

swarm_main() {
    (($#)) || die 'usage: p2pnet swarm install|verify|uninstall|publish|fetch|preseed|status|unpublish'
    local action=$1; shift
    case $action in
        install) swarm_install;; verify) swarm_verify;; uninstall) swarm_uninstall "${1:-}";; publish) swarm_publish "$@";; fetch) swarm_fetch "$@";; preseed) swarm_preseed;;
        status) local json=0; [[ ${1:-} != --json ]] || json=1; swarm_status "$json";;
        unpublish) (($# == 1)) || die 'usage: p2pnet swarm unpublish NAME'; [[ $1 =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid image name: $1"; rm -f -- "$SWARM_TORRENTS/$1.torrent";;
        *) die "unknown swarm action: $action";;
    esac
}

swarm_main "$@"
