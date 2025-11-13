#!/bin/bash
# Database Restore Script for NovaCron
# Restore PostgreSQL database from backup

set -e

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-novacron}"
DB_USER="${DB_USER:-novacron}"
BACKUP_FILE="${1}"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 /backup/database/novacron_backup_20251112_120000.sql.gz"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "==================================="
echo "  Database Restore"
echo "==================================="
echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
echo "Backup file: $BACKUP_FILE"
echo ""

# Verify backup integrity
echo "Verifying backup integrity..."
if [[ "$BACKUP_FILE" == *.gz ]]; then
    gunzip -t "$BACKUP_FILE" || { echo "Backup verification failed!"; exit 1; }
fi

# Confirm restore
read -p "WARNING: This will OVERWRITE the database. Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Restore cancelled"
    exit 0
fi

# Drop existing connections
echo "Dropping existing database connections..."
PGPASSWORD="$DB_PASSWORD" psql \
  --host="$DB_HOST" \
  --port="$DB_PORT" \
  --username="$DB_USER" \
  --dbname="postgres" \
  --command="SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='$DB_NAME' AND pid <> pg_backend_pid();"

# Drop and recreate database
echo "Recreating database..."
PGPASSWORD="$DB_PASSWORD" psql \
  --host="$DB_HOST" \
  --port="$DB_PORT" \
  --username="$DB_USER" \
  --dbname="postgres" \
  --command="DROP DATABASE IF EXISTS $DB_NAME;"

PGPASSWORD="$DB_PASSWORD" psql \
  --host="$DB_HOST" \
  --port="$DB_PORT" \
  --username="$DB_USER" \
  --dbname="postgres" \
  --command="CREATE DATABASE $DB_NAME OWNER $DB_USER;"

# Restore from backup
echo "Restoring database..."
if [[ "$BACKUP_FILE" == *.custom ]]; then
    # Custom format restore
    PGPASSWORD="$DB_PASSWORD" pg_restore \
      --host="$DB_HOST" \
      --port="$DB_PORT" \
      --username="$DB_USER" \
      --dbname="$DB_NAME" \
      --verbose \
      --no-owner \
      --no-acl \
      "$BACKUP_FILE"
else
    # SQL dump restore
    gunzip -c "$BACKUP_FILE" | PGPASSWORD="$DB_PASSWORD" psql \
      --host="$DB_HOST" \
      --port="$DB_PORT" \
      --username="$DB_USER" \
      --dbname="$DB_NAME"
fi

# Verify restoration
echo "Verifying restoration..."
TABLE_COUNT=$(PGPASSWORD="$DB_PASSWORD" psql \
  --host="$DB_HOST" \
  --port="$DB_PORT" \
  --username="$DB_USER" \
  --dbname="$DB_NAME" \
  --tuples-only \
  --command="SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")

echo "Database restored successfully"
echo "Tables restored: $TABLE_COUNT"
echo "Completed at $(date)"
