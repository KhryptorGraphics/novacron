#!/bin/bash
# Database Backup Script for NovaCron
# Automated PostgreSQL backups with WAL archiving

set -e

# Configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-novacron}"
DB_USER="${DB_USER:-novacron}"
BACKUP_DIR="${BACKUP_DIR:-/backup/database}"
S3_BUCKET="${S3_BUCKET:-novacron-backups}"
RETENTION_DAYS=30

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="novacron_backup_${TIMESTAMP}.sql.gz"

echo "Starting database backup at $(date)"
echo "Database: $DB_HOST:$DB_PORT/$DB_NAME"
echo "Backup file: $BACKUP_FILE"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Perform backup with pg_dump
echo "Creating backup..."
PGPASSWORD="$DB_PASSWORD" pg_dump \
  --host="$DB_HOST" \
  --port="$DB_PORT" \
  --username="$DB_USER" \
  --dbname="$DB_NAME" \
  --format=custom \
  --compress=9 \
  --verbose \
  --file="$BACKUP_DIR/$BACKUP_FILE.custom" 2>&1 | tee "$BACKUP_DIR/backup_${TIMESTAMP}.log"

# Create SQL dump for easy restore
PGPASSWORD="$DB_PASSWORD" pg_dump \
  --host="$DB_HOST" \
  --port="$DB_PORT" \
  --username="$DB_USER" \
  --dbname="$DB_NAME" \
  --format=plain \
  --no-owner \
  --no-acl | gzip > "$BACKUP_DIR/$BACKUP_FILE"

# Calculate checksum
md5sum "$BACKUP_DIR/$BACKUP_FILE" > "$BACKUP_DIR/$BACKUP_FILE.md5"
md5sum "$BACKUP_DIR/$BACKUP_FILE.custom" > "$BACKUP_DIR/$BACKUP_FILE.custom.md5"

# Upload to S3
if [ -n "$S3_BUCKET" ]; then
    echo "Uploading to S3..."
    aws s3 cp "$BACKUP_DIR/$BACKUP_FILE" "s3://$S3_BUCKET/database/"
    aws s3 cp "$BACKUP_DIR/$BACKUP_FILE.custom" "s3://$S3_BUCKET/database/"
    aws s3 cp "$BACKUP_DIR/$BACKUP_FILE.md5" "s3://$S3_BUCKET/database/"
fi

# Clean up old backups
echo "Cleaning up old backups (retention: $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "novacron_backup_*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "novacron_backup_*.custom" -mtime +$RETENTION_DAYS -delete

# Backup verification
echo "Verifying backup integrity..."
gunzip -t "$BACKUP_DIR/$BACKUP_FILE" || { echo "Backup verification failed!"; exit 1; }

BACKUP_SIZE=$(du -h "$BACKUP_DIR/$BACKUP_FILE" | awk '{print $1}')
echo "Backup completed successfully"
echo "Backup size: $BACKUP_SIZE"
echo "Backup location: $BACKUP_DIR/$BACKUP_FILE"
echo "S3 location: s3://$S3_BUCKET/database/$BACKUP_FILE"
echo "Completed at $(date)"
