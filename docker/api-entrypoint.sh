#!/bin/sh
# NovaCron API Entrypoint Script
set -e

# The canonical api-server reads *only* environment variables via
# backend/pkg/config/config.go's Load(). It does NOT parse a config file.
# Earlier this entrypoint wrote /etc/novacron/config.yaml, which crashed
# read-only rootfs containers and could not be read by the binary anyway.
# This entrypoint only does runtime-side setup: optional DB readiness poll
# and idempotent schema migration driven by the shared migrate-db binary.

# Wait for the database to be ready
if [ -n "$DB_URL" ]; then
    echo "Waiting for database to be ready..."
    DB_HOST=$(echo "$DB_URL" | sed -n 's/.*@\(.*\):.*/\1/p')
    DB_PORT=$(echo "$DB_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')

    if [ -z "$DB_PORT" ]; then
        DB_PORT=5432
    fi

    RETRIES=30
    until nc -z "$DB_HOST" "$DB_PORT" || [ "$RETRIES" -eq 0 ]; do
        echo "Waiting for database at $DB_HOST:$DB_PORT, $RETRIES retries left..."
        RETRIES=$((RETRIES-1))
        sleep 1
    done

    if [ "$RETRIES" -eq 0 ]; then
        echo "WARNING: Could not connect to database, continuing startup anyway"
    else
        echo "Database is ready"
    fi
fi

# Apply database schema migrations (database/migrate.go — its own Go module,
# built by docker/migrate.Dockerfile). In docker-compose this is normally
# already applied by the dedicated `migrate` init service before this
# container even starts (see docker-compose.yml: api depends_on migrate
# with condition service_completed_successfully); this is a defensive
# re-check for any deployment path that starts this container without that
# ordering guarantee (e.g. a bare `docker run`). golang-migrate's Up() is
# idempotent and no-ops when the schema is already current.
MIGRATE_BIN="${MIGRATE_BIN_PATH:-/shared/bin/migrate-db}"
if [ -x "$MIGRATE_BIN" ]; then
    if [ -n "$DB_URL" ]; then
        echo "Applying database migrations via $MIGRATE_BIN..."
        if "$MIGRATE_BIN" -direction up; then
            echo "Database migrations are up to date."
        else
            echo "ERROR: database migration failed" >&2
            exit 1
        fi
    else
        echo "WARNING: DB_URL not set, skipping migration step"
    fi
else
    echo "No migrate-db binary at $MIGRATE_BIN — assuming migrations were already applied by the dedicated migrate init step (docker-compose 'migrate' service / k8s migrate Job)."
fi

# Check hypervisor connectivity
echo "Checking hypervisor connectivity..."
for HYPERVISOR in $(echo "${HYPERVISOR_ADDRS:-novacron-hypervisor:9000}" | tr ',' ' '); do
    HOST=$(echo "$HYPERVISOR" | cut -d: -f1)
    PORT=$(echo "$HYPERVISOR" | cut -d: -f2)

    if nc -z "$HOST" "$PORT" 2>/dev/null; then
        echo "Hypervisor at $HOST:$PORT is reachable"
    else
        echo "WARNING: Hypervisor at $HOST:$PORT is not reachable"
    fi
done

# Print startup message
echo "Starting NovaCron API Service..."
echo "Log Level: ${LOG_LEVEL:-info}"
echo "API Port: ${API_PORT:-8090}"
echo "Database URL: ${DB_URL:-postgresql://postgres:postgres@postgres:5432/novacron}"
echo "Hypervisor Addresses: ${HYPERVISOR_ADDRS:-novacron-hypervisor:9000}"

# Start the application
if [ "$1" = "novacron-api" ]; then
    echo "Starting Go API service..."
    exec "$@"
elif [ "$1" = "python" ]; then
    echo "Starting Python service..."
    exec "$@"
else
    echo "Starting default service..."
    exec "$@"
fi
