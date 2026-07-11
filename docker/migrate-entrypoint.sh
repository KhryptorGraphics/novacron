#!/bin/sh
# NovaCron Migration Tool Entrypoint
#
# Publishes the migrate-db binary (built from database/migrate.go) onto a
# shared volume — consumed defensively by docker/api-entrypoint.sh — then
# runs it against $DB_URL with whatever flags were passed as CMD/args.
set -e

if [ -d /shared/bin ]; then
    if cp /usr/local/bin/migrate-db /shared/bin/migrate-db 2>/dev/null; then
        echo "Published migrate-db to /shared/bin/migrate-db"
    else
        echo "WARNING: could not publish migrate-db to /shared/bin (non-fatal, no shared volume mounted?)"
    fi
fi

if [ -z "$DB_URL" ] && [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DB_URL (or DATABASE_URL) is required to run migrations" >&2
    exit 1
fi

echo "Running database migrations ($*)..."
exec /usr/local/bin/migrate-db "$@"
