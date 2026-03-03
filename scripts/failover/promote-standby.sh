#!/bin/bash
# Database Failover Script for NovaCron
# Promote standby to primary

set -e

# Configuration
STANDBY_HOST="${STANDBY_HOST:-standby.novacron.io}"
STANDBY_PORT="${STANDBY_PORT:-5432}"
STANDBY_USER="${STANDBY_USER:-postgres}"

echo "==================================="
echo "  Database Failover"
echo "==================================="
echo "Standby: $STANDBY_HOST:$STANDBY_PORT"
echo "Starting at $(date)"
echo ""

# Check standby status
echo "Checking standby status..."
STANDBY_STATUS=$(PGPASSWORD="$STANDBY_PASSWORD" psql \
  --host="$STANDBY_HOST" \
  --port="$STANDBY_PORT" \
  --username="$STANDBY_USER" \
  --dbname="postgres" \
  --tuples-only \
  --command="SELECT pg_is_in_recovery();")

if [ "$STANDBY_STATUS" != " t" ]; then
    echo "Error: Server is not a standby (pg_is_in_recovery = $STANDBY_STATUS)"
    exit 1
fi

echo "Standby confirmed, promoting to primary..."

# Promote standby
PGPASSWORD="$STANDBY_PASSWORD" pg_ctl promote \
  -D /var/lib/postgresql/data

# Wait for promotion
echo "Waiting for promotion to complete..."
for i in {1..30}; do
    RECOVERY_STATUS=$(PGPASSWORD="$STANDBY_PASSWORD" psql \
      --host="$STANDBY_HOST" \
      --port="$STANDBY_PORT" \
      --username="$STANDBY_USER" \
      --dbname="postgres" \
      --tuples-only \
      --command="SELECT pg_is_in_recovery();")

    if [ "$RECOVERY_STATUS" = " f" ]; then
        echo "Promotion successful!"
        break
    fi

    echo "Still in recovery mode... (attempt $i/30)"
    sleep 2
done

# Verify new primary
echo "Verifying new primary..."
PGPASSWORD="$STANDBY_PASSWORD" psql \
  --host="$STANDBY_HOST" \
  --port="$STANDBY_PORT" \
  --username="$STANDBY_USER" \
  --dbname="postgres" \
  --command="SELECT current_timestamp, pg_is_in_recovery();"

# Update DNS/Load Balancer
echo "Updating DNS to point to new primary..."
# Add your DNS update logic here

echo "Failover completed successfully at $(date)"
echo "New primary: $STANDBY_HOST:$STANDBY_PORT"
