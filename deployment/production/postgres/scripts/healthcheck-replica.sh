#!/bin/bash
# PostgreSQL 15 Replica Healthcheck Script

set -e

# Basic connectivity check
pg_isready -U postgres || exit 1

# Check if this is a replica (in recovery mode)
IS_REPLICA=$(psql -U postgres -tAc "SELECT pg_is_in_recovery();" 2>/dev/null)

if [ "$IS_REPLICA" != "t" ]; then
    echo "ERROR: Expected replica but server is not in recovery mode"
    exit 1
fi

# Check replication lag (optional, informational)
LAG_BYTES=$(psql -U postgres -tAc "
    SELECT CASE
        WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn() THEN 0
        ELSE pg_wal_lsn_diff(pg_last_wal_receive_lsn(), pg_last_wal_replay_lsn())
    END AS lag_bytes;" 2>/dev/null || echo "unknown")

if [ "$LAG_BYTES" = "unknown" ]; then
    echo "Replica is healthy (lag unknown)"
elif [ "$LAG_BYTES" -eq 0 ]; then
    echo "Replica is healthy (no lag)"
else
    # Convert bytes to MB for readability
    LAG_MB=$((LAG_BYTES / 1048576))
    echo "Replica is healthy (lag: ${LAG_MB}MB)"

    # Fail health check if lag is too high (>100MB)
    if [ "$LAG_MB" -gt 100 ]; then
        echo "WARNING: Replication lag is too high"
        exit 1
    fi
fi

exit 0