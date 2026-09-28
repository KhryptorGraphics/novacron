#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=common.sh
source /opt/p2pnet/libexec/common.sh
P2P_COMPONENT=init

init_user() {
    local user=$1 home=$2 shell=$3 password_hash
    if ! id "$user" >/dev/null 2>&1; then
        useradd --system --create-home --home-dir "$home" --shell "$shell" --password x "$user"
        CHANGED=$((CHANGED + 1))
        return
    fi
    password_hash=$(getent shadow "$user" | cut -d: -f2)
    if [[ $password_hash != x ]]; then
        # OpenSSH rejects passwd -l accounts before public-key auth. "x" is not
        # a usable password hash but keeps the system account key-authenticatable.
        usermod --password x "$user"
        CHANGED=$((CHANGED + 1))
    fi
}

init_main() {
    local action=${1:-}
    if [[ $action == --node ]]; then action=install; else shift || true; fi
    case $action in
        install)
            require_root
            local node=${1:-}
            if [[ $node == --node ]]; then node=${2:-}; fi
            [[ -n $node ]] || die "usage: p2pnet init --node NAME"
            [[ $node =~ ^[a-z0-9][a-z0-9-]{0,30}$ ]] || die "invalid node name: $node"
            ensure_dir "$P2P_ETC" 0755 root:root
            ensure_dir "$P2P_STATE" 0755 root:root
            local old=""
            if [[ -r $P2P_ETC/node-name ]]; then IFS= read -r old < "$P2P_ETC/node-name" || true; fi
            if [[ $old != "$node" ]]; then printf '%s\n' "$node" | install_file - "$P2P_ETC/node-name" 0644; fi
            init_user p2prepl /var/lib/p2pnet/repl /bin/sh
            init_user p2pbulk /var/lib/p2pnet/bulk /usr/sbin/nologin
            init_user p2pvirt /var/lib/p2pnet/virt /bin/sh
            local path
            for path in "$P2P_ETC" "$P2P_ETC/ssh" "$P2P_ETC/authorized_keys" "$P2P_ETC/generated" \
                "$P2P_STATE" "$P2P_STATE/state/installed" "$P2P_STATE/zrepl" "$P2P_STATE/migrate" \
                "$P2P_STATE/swarm" "$P2P_STATE/public" "$P2P_LOGDIR" "$P2P_RUN" /var/lib/p2pnet; do
                ensure_dir "$path" 0755 root:root
            done
            ensure_dir "$P2P_ETC/secrets" 0750 root:p2pbulk
            ensure_dir /var/lib/p2pnet/bulk/.ssh 0700 p2pbulk:p2pbulk
            ensure_dir /var/lib/p2pnet/virt/.ssh 0700 p2pvirt:p2pvirt
            ensure_dir "$P2P_STATE/dedup" 0750 root:p2pbulk
            ensure_dir /var/cache/p2pnet/restic 0750 p2pbulk:p2pbulk
            ensure_dir "$P2P_STATE/swarm/queue" 0755 p2pbulk:p2pbulk
            ensure_dir "$P2P_STATE/swarm/resume" 0755 p2pbulk:p2pbulk
            ensure_dir /var/lib/p2pnet/public 0755 root:root
            ensure_dir /var/lib/p2pnet/public/torrents 0755 p2pbulk:p2pbulk
            ensure_dir /var/lib/p2pnet/public/restic 0700 p2pbulk:p2pbulk
            local images_dir=/var/lib/p2pnet/images
            if [[ -r $P2P_INVENTORY ]]; then
                inv validate || die "inventory validation failed"
                images_dir=$(inv env --node "$node" | /usr/bin/python3 -c 'import json,sys; print(json.loads(next(line.split("=",1)[1] for line in sys.stdin if line.startswith("P2P_IMAGES_DIR="))))')
            fi
            ensure_dir "$images_dir" 0755 p2pbulk:p2pbulk
            mark_installed init
            install_file - /etc/logrotate.d/p2pnet 0644 <<'EOF'
/var/log/p2pnet/*.log {
    weekly
    rotate 8
    compress
    missingok
    notifempty
    create 0640 root adm
}
EOF
            printf 'p2pnet[init] install complete: changed=%s\n' "$CHANGED"
            ;;
        verify)
            if ! is_installed init; then check SKIP init "not installed"; return 0; fi
            local failed=0
            if [[ -r $P2P_ETC/node-name ]]; then check PASS init.node-name "$(cat "$P2P_ETC/node-name")"; else check FAIL init.node-name "missing"; failed=1; fi
            for user in p2prepl p2pbulk p2pvirt; do if id "$user" >/dev/null 2>&1; then check PASS "user.$user" "exists"; else check FAIL "user.$user" "missing"; failed=1; fi; done
            verify_done "$failed"
            ;;
        uninstall)
            mark_uninstalled init
            log INFO "node identity and data directories are retained"
            ;;
        *) printf 'usage: p2pnet init --node NAME | init {verify|uninstall}\n' >&2; return 2 ;;
    esac
}
init_main "$@"
