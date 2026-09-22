#!/usr/bin/env bash
# NovaCron soak canary: rerun the two-node fabric harness and hold it to the
# soak exit criteria.
#
# The harness (scripts/fabric/two-node-fabric-test.sh) is the only end-to-end
# proof the fabric has: two real api-server processes, one inside a network
# namespace behind a shaped veth, real postgres, real qemu, and a signed join
# between them. A single green run says "this commit works once"; the soak
# reruns it, because the fabric's failure modes (heartbeat races, admission
# queue state, migration reconcile after a restart) are the kind that pass by
# luck. See deploy/README.md for the exit criteria in full.
#
# Each iteration runs the harness in FABRIC_KEEP=1 mode: a failed or even a
# passing run leaves its two nodes, netns, veth pair and databases up for
# inspection. Ports are shifted per iteration so the preserved nodes of the
# previous iteration do not collide with the next one, and the leftovers are
# listed (with the commands that remove them) at the end.
#
# Usage:  deploy/scripts/soak-test.sh [--iterations N] [--min-pass N]
#                                    [--consecutive N] [--log-dir DIR]
#                                    [--clean] [--harness PATH]
# Env:    FABRIC_SHAPE_MBIT, FABRIC_DELAY_MS  passed straight to the harness
# Exit:   0 = exit criteria met, 1 = a run failed the criteria, 2 = usage or
#         missing prerequisite (the harness itself is never started in that case).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$REPO_ROOT/scripts/fabric/two-node-fabric-test.sh"
LOG_DIR="/var/log/novacron/soak"
ITERATIONS=0
MIN_PASS=14
CONSECUTIVE=2
KEEP=1
PORT_BASE=18190

log()  { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die()  { printf '%s ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 2; }

usage() {
  sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --iterations N    how many harness runs to attempt (default: the --consecutive
                    requirement, i.e. stop as soon as it is met)
  --min-pass N      assertions a clean run must report (default 14)
  --consecutive N   clean runs required in a row (default 2)
  --log-dir DIR     where per-iteration logs go (default /var/log/novacron/soak)
  --clean           run the harness with FABRIC_KEEP=0 (tears down after each
                    run; leaves nothing to inspect but nothing behind either)
  --harness PATH    harness to run (default scripts/fabric/two-node-fabric-test.sh)
  -h, --help        this text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --iterations)  ITERATIONS="${2:?--iterations needs a value}"; shift 2 ;;
    --min-pass)    MIN_PASS="${2:?--min-pass needs a value}"; shift 2 ;;
    --consecutive) CONSECUTIVE="${2:?--consecutive needs a value}"; shift 2 ;;
    --log-dir)     LOG_DIR="${2:?--log-dir needs a value}"; shift 2 ;;
    --clean)       KEEP=0; shift ;;
    --harness)     HARNESS="${2:?--harness needs a value}"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *)             usage >&2; die "unknown argument: $1" ;;
  esac
done

case "$CONSECUTIVE" in ''|*[!0-9]*) die "--consecutive must be a number";; esac
case "$MIN_PASS" in ''|*[!0-9]*) die "--min-pass must be a number";; esac
[ "$CONSECUTIVE" -ge 1 ] || die "--consecutive must be >= 1"
[ "$MIN_PASS" -ge 1 ] || die "--min-pass must be >= 1"
[ "$ITERATIONS" -eq 0 ] && ITERATIONS="$CONSECUTIVE"
case "$ITERATIONS" in ''|*[!0-9]*) die "--iterations must be a number";; esac
[ -x "$HARNESS" ] || die "harness not found or not executable: $HARNESS"

# The harness needs passwordless sudo for the netns/veth/tc work and a local
# postgres; check both here so a misconfigured host fails before a run starts.
command -v sudo >/dev/null 2>&1 || die "sudo is required (netns, veth, tc)"
sudo -n true 2>/dev/null || die "passwordless sudo is required: the harness builds a netns, a veth pair and tc qdiscs. Add a sudoers entry for this user or run from a root shell."
command -v psql >/dev/null 2>&1 || die "psql client is required (the harness creates two databases)"
PGPASSWORD="${FABRIC_PGPASSWORD:-postgres}" psql -h "${FABRIC_PGHOST:-127.0.0.1}" -p "${FABRIC_PGPORT:-5432}" \
  -U "${FABRIC_PGUSER:-postgres}" -tAqc 'SELECT 1' >/dev/null 2>&1 \
  || die "no local postgres reachable at ${FABRIC_PGHOST:-127.0.0.1}:${FABRIC_PGPORT:-5432} as ${FABRIC_PGUSER:-postgres}"

mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR (run with sudo)"
[ -w "$LOG_DIR" ] || die "$LOG_DIR is not writable by $(id -un) — run with sudo"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)-$$"
CLEAN_RUNS=0
ATTEMPT=0
RESULT=1
RUN_LOGS=()

log "soak start: harness=$HARNESS min_pass=$MIN_PASS consecutive=$CONSECUTIVE keep=$KEEP logs=$LOG_DIR"

while [ "$ATTEMPT" -lt "$ITERATIONS" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  PORT_A=$((PORT_BASE + ATTEMPT * 10))
  PORT_B=$((PORT_A + 1))
  RUN_LOG="$LOG_DIR/soak-$STAMP-iter$ATTEMPT.log"
  RUN_LOGS+=("$RUN_LOG")

  log "iteration $ATTEMPT/$ITERATIONS (FABRIC_PORT_A=$PORT_A FABRIC_PORT_B=$PORT_B FABRIC_KEEP=$KEEP)"
  set +e
  FABRIC_KEEP="$KEEP" FABRIC_REQUIRE_ALL=1 \
  FABRIC_PORT_A="$PORT_A" FABRIC_PORT_B="$PORT_B" \
    "$HARNESS" 2>&1 | tee "$RUN_LOG"
  HARNESS_RC="${PIPESTATUS[0]}"
  set -e

  PASS_N="$(sed -n 's/.*pass=\([0-9]*\) fail=[0-9]* skip=[0-9]*.*/\1/p' "$RUN_LOG" | tail -n 1)"
  FAIL_N="$(sed -n 's/.*pass=[0-9]* fail=\([0-9]*\) skip=[0-9]*.*/\1/p' "$RUN_LOG" | tail -n 1)"
  SKIP_N="$(sed -n 's/.*pass=[0-9]* fail=[0-9]* skip=\([0-9]*\).*/\1/p' "$RUN_LOG" | tail -n 1)"

  if [ -z "$PASS_N" ] || [ -z "$FAIL_N" ] || [ -z "$SKIP_N" ]; then
    log "iteration $ATTEMPT: FAIL — no pass=/fail=/skip= summary in the harness output (harness exit $HARNESS_RC); log: $RUN_LOG"
    RESULT=1
    break
  fi

  log "iteration $ATTEMPT: harness exit=$HARNESS_RC pass=$PASS_N fail=$FAIL_N skip=$SKIP_N"

  if [ "$FAIL_N" != "0" ] || [ "$SKIP_N" != "0" ] || [ "$PASS_N" -lt "$MIN_PASS" ]; then
    log "iteration $ATTEMPT: DIRTY — exit criteria need fail=0, skip=0 and pass>=$MIN_PASS"
    RESULT=1
    break
  fi

  CLEAN_RUNS=$((CLEAN_RUNS + 1))
  log "iteration $ATTEMPT: CLEAN ($CLEAN_RUNS/$CONSECUTIVE consecutive)"
  if [ "$CLEAN_RUNS" -ge "$CONSECUTIVE" ]; then
    RESULT=0
    break
  fi
done

if [ "$KEEP" = "1" ]; then
  # The harness says exactly what it kept; surface those lines plus the
  # commands that remove them, so a loop cannot leak silently.
  log "preserved state (FABRIC_KEEP=1):"
  for l in "${RUN_LOGS[@]}"; do
    grep -h "FABRIC_KEEP=1 — leaving" "$l" 2>/dev/null | sed 's/^/  /' || true
  done
  for l in "${RUN_LOGS[@]}"; do
    [ -f "$l" ] || continue
    ns="$(sed -n 's/.*netns (\([^)]*\)).*/\1/p' "$l" | tail -n 1)"
    dbs="$(sed -n 's/.*databases (\([^)]*\)).*/\1/p' "$l" | tail -n 1)"
    [ -n "$ns" ] || continue
    run_id="${ns##*-}"
    log "  clean up: sudo ip netns del $ns; sudo ip link del veth-fh-$run_id; sudo pkill -f fabric-test-$run_id"
    if [ -n "$dbs" ]; then
      log "  clean up: psql -h ${FABRIC_PGHOST:-127.0.0.1} -U ${FABRIC_PGUSER:-postgres} -c 'DROP DATABASE ${dbs%%,*}'; psql ... -c 'DROP DATABASE ${dbs##*, }'"
    fi
  done
fi

if [ "$RESULT" = "0" ]; then
  log "SOAK: PASS — $CLEAN_RUNS consecutive clean run(s) (pass>=$MIN_PASS, fail=0, skip=0)"
else
  if [ "$CLEAN_RUNS" -gt 0 ]; then
    log "SOAK: FAIL — $CLEAN_RUNS clean run(s) but $CONSECUTIVE required in a row; ran $ATTEMPT of $ITERATIONS iteration(s). Raise --iterations or lower --consecutive."
  else
    log "SOAK: FAIL — no clean run in $ATTEMPT iteration(s)"
  fi
  log "logs: $LOG_DIR/soak-$STAMP-iter*.log"
fi
exit "$RESULT"