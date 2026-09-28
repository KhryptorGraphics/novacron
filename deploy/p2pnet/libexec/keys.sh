#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=common.sh
source /opt/p2pnet/libexec/common.sh
P2P_COMPONENT=keys

create_wg_key() {
    local key=$1 pub=$2 tmp
    if [[ ! -s $key ]]; then
        install -d -o root -g root -m 0700 "${key%/*}"
        tmp=$(mktemp "${key}.XXXXXX"); chmod 0600 "$tmp"
        wg genkey > "$tmp"
        install -o root -g root -m 0600 "$tmp" "$key"; rm -f "$tmp"
        CHANGED=$((CHANGED + 1))
    fi
    wg pubkey < "$key" | install_file - "$pub" 0644
}

create_ssh_key() {
    local key=$1 owner=$2 comment=$3
    if [[ ! -s $key ]]; then
        install -d -o "${owner%:*}" -g "${owner#*:}" -m 0700 "${key%/*}"
        ssh-keygen -q -t ed25519 -N '' -C "$comment" -f "$key"
        chown "$owner" "$key"; chmod 0600 "$key"
        CHANGED=$((CHANGED + 1))
    fi
    if [[ ! -s $key.pub ]]; then
        ssh-keygen -y -f "$key" | awk -v comment="$comment" '{print $1 " " $2 " " comment}' | install_file - "$key.pub" 0644 "$owner"
    fi
}

keys_main() {
    local action=generate
    if [[ ${1:-} == install || ${1:-} == verify || ${1:-} == uninstall ]]; then action=$1; shift; fi
    local wan2=0 restic=0 purge=0 arg node
    for arg in "$@"; do
        case $arg in
            --wan2) wan2=1 ;;
            --restic-password) restic=1 ;;
            --purge) purge=1 ;;
            *) printf 'unknown keys argument: %s\n' "$arg" >&2; return 2 ;;
        esac
    done
    case $action in
        generate|install)
            require_root
            node=$(self_node)
            need_cmd wg ssh-keygen
            create_wg_key /etc/wireguard/wg0.key /etc/wireguard/wg0.pub
            if (( wan2 )); then create_wg_key /etc/wireguard/wg1.key /etc/wireguard/wg1.pub; fi
            create_ssh_key "$P2P_ETC/ssh/id_ed25519" root:root "p2pnet@$node"
            create_ssh_key /var/lib/p2pnet/bulk/.ssh/id_ed25519 p2pbulk:p2pbulk "p2pnet-bulk@$node"
            if (( restic )) && [[ ! -s $P2P_ETC/secrets/restic.pass ]]; then
                ensure_dir "$P2P_ETC/secrets" 0750 root:p2pbulk
                umask 077
                head -c 32 /dev/urandom | base64 | tr -d '\n' | install_file - "$P2P_ETC/secrets/restic.pass" 0640 root:p2pbulk
            fi
            ssh-keygen -A
            local wg0_pub wg1_pub ssh_pub bulk_pub host_pub
            wg0_pub=$(cat /etc/wireguard/wg0.pub)
            wg1_pub=""; [[ ! -r /etc/wireguard/wg1.pub ]] || wg1_pub=$(cat /etc/wireguard/wg1.pub)
            ssh_pub=$(cat "$P2P_ETC/ssh/id_ed25519.pub")
            bulk_pub=$(cat /var/lib/p2pnet/bulk/.ssh/id_ed25519.pub)
            [[ -r /etc/ssh/ssh_host_ed25519_key.pub ]] || die "OpenSSH host key missing after ssh-keygen -A"
            host_pub=$(awk '{print $1 " " $2}' /etc/ssh/ssh_host_ed25519_key.pub)
            printf 'node key snippet (copy public values into inventory):\n'
            printf 'wg_pubkey: "%s"\nssh_pubkey: "%s"\nssh_bulk_pubkey: "%s"\nssh_hostkey: "%s"\n' "$wg0_pub" "$ssh_pub" "$bulk_pub" "$host_pub"
            if (( wan2 )); then printf 'wan2:\n  wg1_pubkey: "%s"\n' "$wg1_pub"; fi
            mark_installed keys
            printf 'p2pnet[keys] install complete: changed=%s\n' "$CHANGED"
            ;;
        verify)
            if ! is_installed keys; then check SKIP keys "not installed"; return 0; fi
            local failed=0
            for key in /etc/wireguard/wg0.key "$P2P_ETC/ssh/id_ed25519" /var/lib/p2pnet/bulk/.ssh/id_ed25519; do
                if [[ -s $key && -s $key.pub ]]; then check PASS "key.${key##*/}" "present"; else check FAIL "key.${key##*/}" "missing"; failed=1; fi
            done
            verify_done "$failed"
            ;;
        uninstall)
            if (( purge )); then
                for key in /etc/wireguard/wg0.key /etc/wireguard/wg0.pub /etc/wireguard/wg1.key /etc/wireguard/wg1.pub \
                    "$P2P_ETC/ssh/id_ed25519" "$P2P_ETC/ssh/id_ed25519.pub" /var/lib/p2pnet/bulk/.ssh/id_ed25519 \
                    /var/lib/p2pnet/bulk/.ssh/id_ed25519.pub "$P2P_ETC/secrets/restic.pass"; do
                    if [[ -e $key ]]; then rm -f -- "$key"; CHANGED=$((CHANGED + 1)); fi
                done
            fi
            mark_uninstalled keys
            ;;
        *) printf 'usage: p2pnet keys [--wan2] [--restic-password] | keys {verify|uninstall [--purge]}\n' >&2; return 2 ;;
    esac
}
keys_main "$@"
