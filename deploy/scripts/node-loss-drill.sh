#!/usr/bin/env bash
# NovaCron node-loss drill: run the drill documented in deploy/README.md §7
# against a REAL two-node fabric and report what actually happened.
#
# The drill itself lives inside the acceptance harness
# (scripts/fabric/two-node-fabric-test.sh, the FABRIC_DRILL=1 section), because
# it needs a LIVE two-node fabric: the harness's cleanup kills the node
# processes even under FABRIC_KEEP=1, so a standalone script could not reuse
# them. Running it in place also means the drill inherits every proven helper
# the harness already has - start_node, the api helper, the pidfile liveness
# rule and the psql access to both databases.
#
# What the drill asserts is the DOCUMENTED behaviour, including the parts
# that are deliberately not automatic:
#   1. detection            - the survivor reports the dead peer unreachable
#   2. control-plane state  - the dead node's VM row is NOT rewritten by the
#                             survivor (there is no failover)
#   3. no silent migration  - the guest does not run on the survivor
#   4. reconcile-on-restart - when the node comes back it repairs its own
#                             bookkeeping (a 'running' row with no live qemu
#                             becomes 'stopped')
#   5. no auto-restart      - the guest is NOT restarted somewhere else; a
#                             guest on a dead node is a single point of failure
#
# Usage:  deploy/scripts/node-loss-drill.sh [--report FILE] [--keep]
# Env:    FABRIC_PGPORT, FABRIC_PGHOST, FABRIC_PGUSER, FABRIC_PGPASSWORD
# Exit:   0 = every documented outcome held, 1 = a check failed, 2 = usage or
#         missing prerequisite (nothing is started in that case).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARNESS="$REPO_ROOT/scripts/fabric/two-node-fabric-test.sh"
REPORT=""
KEEP=0

usage() {
  sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --report FILE  write the drill report to FILE (default: stdout only)
  --keep         leave the drill's netns/veth pair/databases/logs up for
                 inspection (the node processes are always stopped -- the
                 harness's cleanup kills them before it returns)
  -h, --help     this text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --report) REPORT="${2:?--report needs a value}"; shift 2 ;;
    --keep)   KEEP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)        usage >&2; printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[ -x "$HARNESS" ] || { printf 'ERROR: harness not found or not executable: %s\n' "$HARNESS" >&2; exit 2; }

# FABRIC_DRILL=1 turns the drill section on; FABRIC_KEEP=1 leaves the netns,
# veth pair, databases and logs up afterwards so the operator can inspect what
# the drill left behind (the node processes are always stopped).
export FABRIC_DRILL=1
[ "$KEEP" = "1" ] && export FABRIC_KEEP=1

if [ -n "$REPORT" ]; then
  mkdir -p "$(dirname "$REPORT")"
  # The harness prints its own transcript; tee it into a TEMP file so a failed
  # run can never truncate the committed evidence artifact (tee straight into
  # $REPORT would empty it at startup, leaving criterion-7 evidence holding
  # only the header block of a failed drill). The report is copied into place
  # only after the drill passes; on failure the transcript is left at the
  # temp path and the old report is preserved untouched.
  TMP_REPORT="$(mktemp "${TMPDIR:-/tmp}/node-loss-drill.XXXXXX.txt")"
  set +e
  {
    # Provenance header: tie the transcript to the revision that produced it,
    # so a reader never has to infer which binary ran.
    printf '=== drill provenance\n'
    printf '  revision: %s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
      printf '  working tree: DIRTY\n'
    else
      printf '  working tree: clean\n'
    fi
    printf '  generated: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '=== end provenance\n'
    bash "$HARNESS" 2>&1
  } | tee "$TMP_REPORT"
  RC="${PIPESTATUS[0]}"
  set -e
  if [ "$RC" -eq 0 ]; then
    cp "$TMP_REPORT" "$REPORT"
    printf 'drill report written to %s\n' "$REPORT"
  else
    printf 'drill FAILED (exit %s) - %s left untouched; transcript: %s\n' \
      "$RC" "$REPORT" "$TMP_REPORT" >&2
  fi
  exit "$RC"
fi

exec bash "$HARNESS"
