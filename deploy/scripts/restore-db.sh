#!/usr/bin/env bash
# NovaCron control-plane database restore (pg_restore of a backup-db.sh dump).
#
# Restoring replaces the fabric's authoritative state — membership, VM
# ownership, jobs, transfers, usage ledger. Run it against a stopped service:
# an api-server writing rows while they are being dropped and recreated will
# either fail the restore or leave the cluster inconsistent. This script
# therefore refuses to run while a NovaCron unit is active unless --force.
#
# Usage:  deploy/scripts/restore-db.sh --from FILE.dump [--db-url URL]
#                                    [--env-file PATH] [--jobs N]
#                                    [--no-owner] [--force] [--dry-run]
# Env:    DB_URL             (used when --db-url is not given)
# Exit:   0 = restore applied, 1 = failed, 2 = usage error, 3 = refused
#         (service running or target not empty) without --force.

set -euo pipefail

ENV_FILE="/etc/novacron/novacron.env"
DB_URL_ARG=""
DUMP=""
JOBS=1
NO_OWNER=0
FORCE=0
DRY_RUN=0

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { printf '%s ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }
# Refusals are their own exit code: a caller can tell "I refused to touch your
# database" from "the restore ran and failed".
refuse() { printf '%s REFUSED: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 3; }

usage() {
  sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --from FILE.dump  dump to restore (required)
  --db-url URL      postgres URL to restore into (else $DB_URL, else --env-file)
  --env-file PATH   file to read DB_URL from (default /etc/novacron/novacron.env)
  --jobs N          parallel restore jobs (default 1; any N > 1 disables
                    --single-transaction and --clean, so use it on an empty DB)
  --no-owner        restore without object ownership (cross-cluster restores)
  --force           proceed even if a NovaCron unit is active / the DB is not empty
  --dry-run         verify the dump and print the plan, change nothing
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --from)     DUMP="${2:?--from needs a value}"; shift 2 ;;
    --db-url)   DB_URL_ARG="${2:?--db-url needs a value}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?--env-file needs a value}"; shift 2 ;;
    --jobs)     JOBS="${2:?--jobs needs a value}"; shift 2 ;;
    --no-owner) NO_OWNER=1; shift ;;
    --force)    FORCE=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *)          usage >&2; die "unknown argument: $1" ;;
  esac
done

[ -n "$DUMP" ] || { usage >&2; die "--from is required"; }
[ -f "$DUMP" ] || die "no such dump: $DUMP"
case "$JOBS" in ''|*[!0-9]*) die "--jobs must be a positive integer, got '$JOBS'";; esac
[ "$JOBS" -ge 1 ] || die "--jobs must be >= 1"

read_env_db_url() {
  local file="$1"
  [ -r "$file" ] || return 1
  sed -n 's/^[[:space:]]*DB_URL=//p' "$file" | tail -n 1 | sed 's/^"\(.*\)"$/\1/'
}

if [ -z "$DB_URL_ARG" ]; then
  if [ -n "${DB_URL:-}" ]; then
    DB_URL_ARG="$DB_URL"
  elif DB_URL_ARG="$(read_env_db_url "$ENV_FILE")" && [ -n "$DB_URL_ARG" ]; then
    :
  else
    die "no database URL: pass --db-url, set \$DB_URL, or add a DB_URL= line to $ENV_FILE"
  fi
fi

for bin in pg_restore psql sha256sum; do
  command -v "$bin" >/dev/null 2>&1 || die "missing '$bin' — install the postgres client package (postgresql-client on Debian/Ubuntu)"
done

if [ -f "$DUMP.sha256" ]; then
  # Compare the recorded hash against THIS file: `sha256sum --check` would
  # verify whatever name the .sha256 file lists, so a stale or swapped
  # checksum file could vouch for a dump that is not the one being restored.
  expected="$(awk 'NF { print $1; exit }' "$DUMP.sha256")"
  actual="$(sha256sum "$DUMP" | awk '{print $1}')"
  [ -n "$expected" ] || die "$DUMP.sha256 has no hash in it"
  if [ "$expected" != "$actual" ]; then
    refuse "checksum mismatch for $DUMP (recorded $expected, actual $actual) — refusing to restore a corrupt or tampered dump"
  fi
  log "checksum ok: $DUMP"
else
  log "WARNING: no $DUMP.sha256 alongside the dump — integrity not verified"
fi

pg_restore --list "$DUMP" >/dev/null 2>&1 || die "$DUMP is not a pg_restore archive (pg_dump -Fc output)"

DB_NAME="$(psql "$DB_URL_ARG" -tAqc 'SELECT current_database()')" \
  || die "cannot connect to $DB_URL_ARG (check the server is up and the URL credentials)"

# An active control plane both races the restore and will be serving rows that
# no longer exist mid-restore.
if command -v systemctl >/dev/null 2>&1; then
  for unit in novacron-api-server.service novacron-fabric-peer.service novacron-api.service; do
    if systemctl is-active --quiet "$unit" 2>/dev/null; then
      if [ "$FORCE" != "1" ]; then
        refuse "$unit is running — 'sudo systemctl stop $unit' first, or pass --force (unsafe: the service will write to $DB_NAME while objects are dropped and recreated)"
      fi
      log "WARNING: $unit is active and --force was given; the restore is racing a live control plane"
    fi
  done
fi

EXISTING="$(psql "$DB_URL_ARG" -tAqc \
  "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")" || EXISTING="0"
if [ "$EXISTING" -gt 0 ] && [ "$FORCE" != "1" ]; then
  refuse "$DB_NAME already has $EXISTING public tables — restore would overwrite them. Re-run with --force to drop and recreate the objects in the dump."
fi

RESTORE_ARGS=(--dbname="$DB_URL_ARG" --exit-on-error)
if [ "$JOBS" -gt 1 ]; then
  RESTORE_ARGS+=(--jobs="$JOBS")
else
  # One transaction: a failed restore leaves the previous state intact instead
  # of a half-applied schema.
  RESTORE_ARGS+=(--single-transaction --clean --if-exists)
fi
[ "$NO_OWNER" = "1" ] && RESTORE_ARGS+=(--no-owner)

if [ "$DRY_RUN" = "1" ]; then
  log "dry-run: would restore $DUMP into $DB_NAME (${EXISTING} existing public tables)"
  log "dry-run: pg_restore ${RESTORE_ARGS[*]}"
  exit 0
fi

log "restoring $DUMP into $DB_NAME (jobs=$JOBS)"
pg_restore "${RESTORE_ARGS[@]}" "$DUMP" \
  || die "pg_restore failed — with --single-transaction the database is unchanged; fix the reported error and retry"

TABLES="$(psql "$DB_URL_ARG" -tAqc \
  "SELECT count(*) FROM information_schema.tables WHERE table_schema='public'")"
log "restore ok: $DB_NAME now has $TABLES public tables"
log "reminder: start the units again ('sudo systemctl start novacron-api-server') and check /health"