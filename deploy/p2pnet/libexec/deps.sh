#!/usr/bin/env bash
set -Eeuo pipefail
# shellcheck source=common.sh
source /opt/p2pnet/libexec/common.sh
P2P_COMPONENT=deps

deps_main() {
    local action=${1:-}; shift || true
    case $action in
        install)
            require_root
            need_cmd apt-get
            DEBIAN_FRONTEND=noninteractive apt-get update
            DEBIAN_FRONTEND=noninteractive apt-get install -y wireguard-tools iproute2 iperf3 sysstat jq rsync curl openssh-server mptcpd zfsutils-linux mbuffer libvirt-daemon-system libvirt-clients python3-yaml python3-libtorrent restic linux-tools-common
            case $(dpkg --print-architecture) in
                amd64) DEBIAN_FRONTEND=noninteractive apt-get install -y qemu-system-x86 ;;
                arm64) DEBIAN_FRONTEND=noninteractive apt-get install -y qemu-system-arm ;;
                *) die "unsupported architecture: $(dpkg --print-architecture)" ;;
            esac
            if ! DEBIAN_FRONTEND=noninteractive apt-get install -y "linux-tools-$(uname -r)"; then log WARN "linux-tools-$(uname -r) unavailable; bench --perf will refuse"; fi
            if ! DEBIAN_FRONTEND=noninteractive apt-get install -y "linux-modules-extra-$(uname -r)"; then log WARN "linux-modules-extra-$(uname -r) unavailable; sch_netem and ifb may be absent"; fi
            if systemctl list-unit-files mptcpd.service --no-legend 2>/dev/null | grep -q '^mptcpd\.service'; then systemctl disable --now mptcpd.service; fi
            mark_installed deps
            printf 'p2pnet[deps] install complete: changed=%s\n' "$CHANGED"
            ;;
        verify)
            if ! is_installed deps; then check SKIP deps "not installed"; return 0; fi
            local failed=0 command
            for command in wireguard ip iperf3 mpstat jq rsync curl ssh mptcpize zfs mbuffer virsh restic; do
                if command -v "$command" >/dev/null 2>&1; then check PASS "command.$command" "available"; else check FAIL "command.$command" "missing"; failed=1; fi
            done
            verify_done "$failed"
            ;;
        uninstall)
            mark_uninstalled deps
            log INFO "dependency packages are retained because they may be shared by system services"
            ;;
        *) printf 'usage: p2pnet deps {install|verify|uninstall}\n' >&2; return 2 ;;
    esac
}
deps_main "$@"
