#!/bin/bash
# PostgreSQL 15 Primary Entrypoint Script
# Sets up replication password and initializes primary configuration

set -e

# Source the original entrypoint functions
source /usr/local/bin/docker-entrypoint.sh

# Function to set replication password
setup_replication_password() {
    if [ -f "/run/secrets/replication_password" ]; then
        REPLICATION_PASSWORD=$(cat /run/secrets/replication_password)

        # Wait for PostgreSQL to be ready
        until pg_isready -U postgres; do
            echo "Waiting for PostgreSQL to be ready..."
            sleep 2
        done

        # Set the replication password
        psql -U postgres -c "ALTER ROLE replicator WITH PASSWORD '$REPLICATION_PASSWORD';"
        echo "Replication password set successfully"
    else
        echo "WARNING: Replication password file not found at /run/secrets/replication_password"
    fi
}

# Custom initialization function
docker_setup_primary() {
    # Copy the initialization SQL to docker-entrypoint-initdb.d
    if [ -d "/postgres-init" ]; then
        cp -r /postgres-init/* /docker-entrypoint-initdb.d/ 2>/dev/null || true
    fi

    # Run the original postgres initialization
    docker_setup_env
    docker_create_db_directories

    if [ "$(id -u)" = '0' ]; then
        exec gosu postgres "$BASH_SOURCE" "$@"
    fi

    if [ -z "$DATABASE_ALREADY_EXISTS" ]; then
        docker_verify_minimum_env
        docker_init_database_dir
        pg_setup_hba_conf

        # Start temporary server for setup
        docker_temp_server_start "$@"

        docker_setup_db
        docker_process_init_files /docker-entrypoint-initdb.d/*

        # Setup replication password after database initialization
        setup_replication_password

        docker_temp_server_stop
    fi
}

# Override configuration files if provided
if [ -f "/etc/novacron/postgres/postgresql.conf" ]; then
    cp /etc/novacron/postgres/postgresql.conf "$PGDATA/postgresql.conf"
    echo "Applied custom postgresql.conf"
fi

if [ -f "/etc/novacron/postgres/pg_hba.conf" ]; then
    cp /etc/novacron/postgres/pg_hba.conf "$PGDATA/pg_hba.conf"
    echo "Applied custom pg_hba.conf"
fi

# Execute the main entrypoint
exec docker-entrypoint.sh "$@"