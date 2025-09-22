#!/bin/bash
# PostgreSQL 15 Replica Initialization Script

set -e

PGDATA="${PGDATA:-/var/lib/postgresql/data}"
PRIMARY_HOST="${PRIMARY_HOST:-postgres-primary}"
PRIMARY_PORT="${PRIMARY_PORT:-5432}"
REPLICATION_USER="${REPLICATION_USER:-replicator}"
REPLICATION_PASSWORD_FILE="${REPLICATION_PASSWORD_FILE:-/run/secrets/replication_password}"

# Function to wait for primary
wait_for_primary() {
    echo "Waiting for primary server at ${PRIMARY_HOST}:${PRIMARY_PORT}..."
    until pg_isready -h "${PRIMARY_HOST}" -p "${PRIMARY_PORT}" -U postgres; do
        echo "Primary is not ready yet. Waiting..."
        sleep 2
    done
    echo "Primary server is ready!"
}

# Initialize replica if data directory is empty
if [ ! -s "${PGDATA}/PG_VERSION" ]; then
    echo "Initializing PostgreSQL replica..."

    # Wait for primary to be available
    wait_for_primary

    # Read replication password
    if [ -f "${REPLICATION_PASSWORD_FILE}" ]; then
        REPLICATION_PASSWORD=$(cat "${REPLICATION_PASSWORD_FILE}")
    else
        echo "ERROR: Replication password file not found!"
        exit 1
    fi

    # Perform base backup from primary
    echo "Creating base backup from primary..."
    PGPASSWORD="${REPLICATION_PASSWORD}" pg_basebackup \
        -h "${PRIMARY_HOST}" \
        -p "${PRIMARY_PORT}" \
        -D "${PGDATA}" \
        -U "${REPLICATION_USER}" \
        -v -P -w \
        -X stream \
        -c fast \
        -S replica_slot

    # Create standby.signal file (PostgreSQL 12+ requirement)
    touch "${PGDATA}/standby.signal"

    # Configure replication settings in postgresql.auto.conf
    cat >> "${PGDATA}/postgresql.auto.conf" <<EOF
# Replication Configuration
primary_conninfo = 'host=${PRIMARY_HOST} port=${PRIMARY_PORT} user=${REPLICATION_USER} password=${REPLICATION_PASSWORD} application_name=replica1'
primary_slot_name = 'replica_slot'
recovery_target_timeline = 'latest'
restore_command = 'cp /var/lib/postgresql/archive/%f %p'
EOF

    # Set proper permissions
    chown -R postgres:postgres "${PGDATA}"
    chmod 700 "${PGDATA}"

    echo "Replica initialization complete!"
else
    echo "Data directory already exists. Skipping initialization."
fi

# Start PostgreSQL
echo "Starting PostgreSQL in replica mode..."
exec postgres