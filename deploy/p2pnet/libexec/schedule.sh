#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "${P2P_COMMON:-/opt/p2pnet/libexec/common.sh}"

P2P_COMPONENT=schedule
UNIT_DIR=/etc/systemd/system
TIMERS=(p2pnet-qos-night.timer p2pnet-qos-day.timer p2pnet-swarm-preseed.timer)

render_service() {
    local name=$1 command=$2
    cat <<EOF
[Unit]
Description=p2pnet scheduled action: ${name}

[Service]
Type=oneshot
ExecStart=/opt/p2pnet/bin/p2pnet ${command}
EOF
}

# No Persistent=: after downtime spanning both QoS switch times, both catch-up runs would fire in
# undefined order and could leave the night profile in place by day. Boot state is already set by
# p2pnet-qos.service (`qos apply --profile auto`), and a missed preseed must wait for the next night.
render_timer() {
    local calendar=$1 service=$2
    cat <<EOF
[Unit]
Description=p2pnet schedule timer for ${service}

[Timer]
OnCalendar=${calendar}
Unit=${service}

[Install]
WantedBy=timers.target
EOF
}

# HH:MM + MINUTES, wrapping at midnight. Plain arithmetic: GNU date parses "02:00 +5 minutes" as
# 02:00 in UTC+5 plus one minute.
clock_add() {
    local clock=$1 add=$2 total
    [[ $clock =~ ^([01][0-9]|2[0-3]):([0-5][0-9])$ ]] || die "invalid HH:MM time: $clock"
    total=$(( (10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]} + add) % 1440 ))
    printf '%02d:%02d\n' $((total / 60)) $((total % 60))
}

schedule_install() {
    require_root
    load_env
    local before=$CHANGED preseed_time
    preseed_time=$(clock_add "$P2P_NIGHT_START" 5)
    # Redirects, not pipes: install_file must run in this shell so its CHANGED increments survive.
    install_file - "$UNIT_DIR/p2pnet-qos-night.service" 0644 < <(render_service qos-night 'qos apply --profile night')
    install_file - "$UNIT_DIR/p2pnet-qos-day.service" 0644 < <(render_service qos-day 'qos apply --profile day')
    install_file - "$UNIT_DIR/p2pnet-swarm-preseed.service" 0644 < <(render_service swarm-preseed 'swarm preseed')
    install_file - "$UNIT_DIR/p2pnet-qos-night.timer" 0644 < <(render_timer "*-*-* ${P2P_NIGHT_START}:00" p2pnet-qos-night.service)
    install_file - "$UNIT_DIR/p2pnet-qos-day.timer" 0644 < <(render_timer "*-*-* ${P2P_NIGHT_END}:00" p2pnet-qos-day.service)
    install_file - "$UNIT_DIR/p2pnet-swarm-preseed.timer" 0644 < <(render_timer "*-*-* ${preseed_time}:00" p2pnet-swarm-preseed.service)
    if (( CHANGED != before )); then UNITS_CHANGED=1; fi
    reload_units
    local unit
    for unit in "${TIMERS[@]}"; do
        if ! systemctl is-enabled --quiet "$unit" || ! systemctl is-active --quiet "$unit"; then
            systemctl enable --now "$unit"
            CHANGED=$((CHANGED + 1))
        fi
    done
    mark_installed schedule
    log INFO "install complete: changed=$CHANGED"
    printf 'p2pnet[schedule] install complete: changed=%s\n' "$CHANGED"
}

schedule_verify() {
    if ! is_installed schedule; then
        check SKIP schedule 'not installed'
        verify_done "${P2P_VERIFY_FAILED:-0}"
        return
    fi
    local unit next
    for unit in "${TIMERS[@]}"; do
        if systemctl is-enabled --quiet "$unit" && systemctl is-active --quiet "$unit"; then
            next=$(systemctl show "$unit" -p NextElapseUSecRealtime --value 2>/dev/null || true)
            check PASS "$unit" "enabled; next=${next:-unknown}"
        else
            check FAIL "$unit" 'timer is not enabled and active'
        fi
    done
    verify_done "${P2P_VERIFY_FAILED:-0}"
}

schedule_uninstall() {
    require_root
    systemctl disable --now "${TIMERS[@]}" 2>/dev/null || true
    rm -f "$UNIT_DIR/p2pnet-qos-night.timer" "$UNIT_DIR/p2pnet-qos-day.timer" \
        "$UNIT_DIR/p2pnet-swarm-preseed.timer" "$UNIT_DIR/p2pnet-qos-night.service" \
        "$UNIT_DIR/p2pnet-qos-day.service" "$UNIT_DIR/p2pnet-swarm-preseed.service"
    UNITS_CHANGED=1
    reload_units
    mark_uninstalled schedule
}

case "${1:-}" in
    install) shift; (($# == 0)) || die 'usage: p2pnet schedule install'; schedule_install ;;
    verify) shift; (($# == 0)) || die 'usage: p2pnet schedule verify'; schedule_verify ;;
    uninstall)
        shift
        if (($# > 1)); then
            die 'usage: p2pnet schedule uninstall [--purge]'
        elif (($# == 1)) && [[ $1 != --purge ]]; then
            die 'usage: p2pnet schedule uninstall [--purge]'
        fi
        schedule_uninstall
        ;;
    *) die 'usage: p2pnet schedule {install|verify|uninstall [--purge]}' ;;
esac
