#!/bin/bash
# Set replication password after database initialization
# This script runs as part of docker-entrypoint-initdb.d

set -e

if [ -f "/run/secrets/replication_password" ]; then
    REPLICATION_PASSWORD=$(cat /run/secrets/replication_password)

    # Set the replication password for the replicator user
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        ALTER ROLE replicator WITH PASSWORD '$REPLICATION_PASSWORD';
EOSQL

    echo "Replication password set successfully for user 'replicator'"
else
    echo "WARNING: Replication password file not found at /run/secrets/replication_password"
    echo "The replicator user was created without a password!"
fi