#!/usr/bin/env bash
# NovaCron control-plane database backup (pg_dump custom format).
#
# The fabric's authoritative state is postgres: cluster_peers membership, vms
# rows (which node owns which VM), jobs, transfers and the usage_events ledger.
# Everything else on a node is reconstructible from these rows plus the VM
# disks, so a dump is the one artifact an operator cannot re-derive.
#
# Usage:  deploy/scripts/backup-db.sh [--db-url URL] [--env-file PATH]
#                                    [--out DIR] [--keep N] [--quiet]
# Env:    DB_URL             (used when --db-url is not given)
# Exit:   0 = dump written and verified, 1 = failed, 2 = usage error.

set -euo pipefail

ENV_FILE="/etc/novacron/novacron.env"
OUT_DIR="/var/backups/novacron"
KEEP=0
DB_URL_ARG=""
QUIET=0

log()  { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die()  { printf '%s ERROR: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; exit 1; }

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --db-url URL      postgres URL to dump (else $DB_URL, else --env-file)
  --env-file PATH   file to read DB_URL from (default /etc/novacron/novacron.env)
  --out DIR         directory for the dump (default /var/backups/novacron)
  --keep N          keep only the newest N dumps in DIR (default 0 = keep all)
  --quiet           suppress progress output (errors still go to stderr)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --db-url)   DB_URL_ARG="${2:?--db-url needs a value}"; shift 2 ;;
    --env-file) ENV_FILE="${2:?--env-file needs a value}"; shift 2 ;;
    --out)      OUT_DIR="${2:?--out needs a value}"; shift 2 ;;
    --keep)     KEEP="${2:?--keep needs a value}"; shift 2 ;;
    --quiet)    QUIET=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *)          usage >&2; die "unknown argument: $1" ;;
  esac
done

# DB_URL from the same env file the api-server unit loads, so a backup taken
# here always targets the database the running control plane uses.
read_env_db_url() {
  local file="$1"
  [ -r "$file" ] || return 1
  # Last assignment wins, matching systemd EnvironmentFile semantics.
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

for bin in pg_dump pg_restore psql sha256sum; do
  command -v "$bin" >/dev/null 2>&1 || die "missing '$bin' — install the postgres client package (postgresql-client on Debian/Ubuntu)"
done

mkdir -p "$OUT_DIR" || die "cannot create $OUT_DIR (need root/sudo)"
[ -w "$OUT_DIR" ] || die "$OUT_DIR is not writable by $(id -un) — run with sudo"

DB_NAME="$(psql "$DB_URL_ARG" -tAqc 'SELECT current_database()')" \
  || die "cannot connect to $DB_URL_ARG (check the server is up and the URL credentials)"
[ -n "$DB_NAME" ] || die "connected to $DB_URL_ARG but could not read current_database()"

# golang-migrate's bookkeeping table; recorded so a restore can be compared
# against the schema the dump was taken at. Absent on a never-migrated DB.
SCHEMA_VERSION="$(psql "$DB_URL_ARG" -tAqc \
  "SELECT version FROM schema_migrations LIMIT 1" 2>/dev/null || true)"
SCHEMA_DIRTY="$(psql "$DB_URL_ARG" -tAqc \
  "SELECT dirty FROM schema_migrations LIMIT 1" 2>/dev/null || true)"
[ -n "$SCHEMA_VERSION" ] || SCHEMA_VERSION="unknown"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BASE="$OUT_DIR/$DB_NAME-$STAMP"
DUMP="$BASE.dump"

[ "$QUIET" = "1" ] || log "dumping $DB_NAME (schema version $SCHEMA_VERSION) -> $DUMP"

# -Fc: custom format — compressed, and the only format pg_restore can select
# tables/objects from. --no-owner is deliberately NOT set: a restore on this
# cluster should reproduce the original object ownership (use pg_restore
# --no-owner explicitly when restoring onto a differently-owned cluster).
pg_dump --format=custom --compress=6 --file="$DUMP" "$DB_URL_ARG" \
  || { rm -f "$DUMP"; die "pg_dump failed (see the pg_dump error above)"; }

# A dump whose pg_restore --list fails is not a backup; catch it now, not on
# the day it is needed.
pg_restore --list "$DUMP" >/dev/null 2>&1 \
  || { rm -f "$DUMP"; die "dump at $DUMP is unreadable by pg_restore — backup discarded"; }

# Recorded with the basename so the file can be verified after the dump is
# moved:  cd <dir> && sha256sum -c <file>.dump.sha256
( cd "$OUT_DIR" && sha256sum "$(basename "$DUMP")" > "$(basename "$DUMP").sha256" )
cat > "$BASE.meta" <<EOF
database=$DB_NAME
taken_at=$STAMP
schema_version=$SCHEMA_VERSION
schema_dirty=${SCHEMA_DIRTY:-unknown}
host=$(hostname -f 2>/dev/null || hostname)
size_bytes=$(wc -c < "$DUMP")
EOF

if [ "$KEEP" -gt 0 ]; then
  # shellcheck disable=SC2012,SC2086 # dump names are operator-controlled, no spaces
  ls -1t "$OUT_DIR"/*.dump 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do
    [ "$QUIET" = "1" ] || log "retention: removing $old"
    rm -f "$old" "$old.sha256" "${old%.dump}.meta"
  done
fi

[ "$QUIET" = "1" ] || log "backup ok: $DUMP ($(wc -c < "$DUMP") bytes, sha256 in $DUMP.sha256)"