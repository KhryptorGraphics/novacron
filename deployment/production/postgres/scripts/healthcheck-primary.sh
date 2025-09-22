#!/bin/bash
# PostgreSQL 15 Primary Healthcheck Script

set -e

# Basic connectivity check
pg_isready -U postgres -d novacron || exit 1

# Check if this is the primary (not in recovery mode)
IS_PRIMARY=$(psql -U postgres -tAc "SELECT NOT pg_is_in_recovery();" 2>/dev/null)

if [ "$IS_PRIMARY" != "t" ]; then
    echo "ERROR: Expected primary but server is in recovery mode"
    exit 1
fi

# Check replication status if replicas are expected
REPLICATION_STATUS=$(psql -U postgres -tAc "SELECT count(*) FROM pg_stat_replication;" 2>/dev/null || echo "0")

# Log replication status (informational)
if [ "$REPLICATION_STATUS" -gt 0 ]; then
    echo "Primary is healthy with $REPLICATION_STATUS connected replica(s)"
else
    echo "Primary is healthy (no replicas connected)"
fi

exit 0