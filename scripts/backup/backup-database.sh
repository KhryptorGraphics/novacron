#!/bin/bash

# NovaCron Database Backup Script
# Version: 1.0.0
# Description: Comprehensive database backup with encryption and cloud storage

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/novacron}"
LOG_FILE="${LOG_DIR:-/var/log/novacron}/backup.log"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-9}"
ENCRYPTION_ENABLED="${ENCRYPTION_ENABLED:-true}"

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-novacron}"
DB_USER="${DB_USER:-novacron}"
DB_PASSWORD="${DB_PASSWORD:-}"
DATABASE_URL="${DATABASE_URL:-}"

# Cloud storage configuration
S3_ENABLED="${S3_ENABLED:-false}"
S3_BUCKET="${S3_BUCKET:-}"
S3_PREFIX="${S3_PREFIX:-novacron/database}"
AWS_REGION="${AWS_REGION:-us-west-2}"

# Notification configuration
WEBHOOK_URL="${BACKUP_WEBHOOK_URL:-}"
EMAIL_ENABLED="${EMAIL_ENABLED:-false}"
EMAIL_TO="${EMAIL_TO:-}"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Error handler
error_exit() {
    log "ERROR" "Backup failed: $1"
    send_notification "error" "Database backup failed: $1"
    exit 1
}

# Success handler
success_exit() {
    log "INFO" "Backup completed successfully: $1"
    send_notification "success" "Database backup completed: $1"
    exit 0
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    # Webhook notification
    if [ -n "$WEBHOOK_URL" ]; then
        local color="good"
        if [ "$status" = "error" ]; then
            color="danger"
        elif [ "$status" = "warning" ]; then
            color="warning"
        fi
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"$color\",
                    \"title\": \"NovaCron Database Backup\",
                    \"text\": \"$message\",
                    \"fields\": [{
                        \"title\": \"Host\",
                        \"value\": \"$(hostname)\",
                        \"short\": true
                    }, {
                        \"title\": \"Database\",
                        \"value\": \"$DB_NAME\",
                        \"short\": true
                    }, {
                        \"title\": \"Timestamp\",
                        \"value\": \"$(date)\",
                        \"short\": true
                    }]
                }]
            }" "$WEBHOOK_URL" 2>/dev/null || log "WARN" "Failed to send webhook notification"
    fi
    
    # Email notification
    if [ "$EMAIL_ENABLED" = "true" ] && [ -n "$EMAIL_TO" ]; then
        local subject="NovaCron Database Backup - $status"
        echo "$message" | mail -s "$subject" "$EMAIL_TO" 2>/dev/null || log "WARN" "Failed to send email notification"
    fi
}

# Check dependencies
check_dependencies() {
    log "INFO" "Checking dependencies..."
    
    local deps=("pg_dump" "gzip" "date")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error_exit "$dep is required but not installed"
        fi
    done
    
    if [ "$ENCRYPTION_ENABLED" = "true" ] && ! command -v "gpg" &> /dev/null; then
        error_exit "gpg is required for encryption but not installed"
    fi
    
    if [ "$S3_ENABLED" = "true" ] && ! command -v "aws" &> /dev/null; then
        error_exit "aws CLI is required for S3 upload but not installed"
    fi
    
    log "INFO" "All dependencies satisfied"
}

# Create backup directory
create_backup_dir() {
    log "INFO" "Creating backup directory..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Set appropriate permissions
    chmod 750 "$BACKUP_DIR"
    chmod 640 "$LOG_FILE" 2>/dev/null || touch "$LOG_FILE"
    
    log "INFO" "Backup directory created: $BACKUP_DIR"
}

# Test database connection
test_database_connection() {
    log "INFO" "Testing database connection..."
    
    if [ -n "$DATABASE_URL" ]; then
        export PGPASSWORD=""
        if ! pg_isready -d "$DATABASE_URL" -t 10; then
            error_exit "Cannot connect to database using DATABASE_URL"
        fi
    else
        export PGPASSWORD="$DB_PASSWORD"
        if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t 10; then
            error_exit "Cannot connect to database at $DB_HOST:$DB_PORT"
        fi
    fi
    
    log "INFO" "Database connection successful"
}

# Create database backup
create_backup() {
    log "INFO" "Starting database backup..."
    
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_name="novacron_backup_${timestamp}"
    local backup_file="$BACKUP_DIR/${backup_name}.sql"
    
    # Create backup with pg_dump
    log "INFO" "Creating SQL dump..."
    
    local pg_dump_args=(
        "--verbose"
        "--clean"
        "--if-exists"
        "--create"
        "--no-owner"
        "--no-privileges"
        "--format=custom"
        "--compress=$COMPRESSION_LEVEL"
    )
    
    if [ -n "$DATABASE_URL" ]; then
        pg_dump "${pg_dump_args[@]}" "$DATABASE_URL" -f "$backup_file" 2>&1 | tee -a "$LOG_FILE"
    else
        PGPASSWORD="$DB_PASSWORD" pg_dump "${pg_dump_args[@]}" \
            -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
            -f "$backup_file" 2>&1 | tee -a "$LOG_FILE"
    fi
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        error_exit "pg_dump failed"
    fi
    
    # Get backup size
    local backup_size=$(du -h "$backup_file" | cut -f1)
    log "INFO" "SQL dump created: $backup_file ($backup_size)"
    
    # Create additional compressed backup
    local compressed_backup="$backup_file.gz"
    log "INFO" "Creating compressed backup..."
    
    if gzip -c "$backup_file" > "$compressed_backup"; then
        local compressed_size=$(du -h "$compressed_backup" | cut -f1)
        log "INFO" "Compressed backup created: $compressed_backup ($compressed_size)"
        rm "$backup_file" # Remove uncompressed version
        backup_file="$compressed_backup"
    else
        log "WARN" "Failed to create compressed backup, keeping original"
    fi
    
    # Encrypt backup if enabled
    if [ "$ENCRYPTION_ENABLED" = "true" ]; then
        log "INFO" "Encrypting backup..."
        
        local encrypted_backup="${backup_file}.gpg"
        if [ -n "${GPG_RECIPIENT:-}" ]; then
            gpg --trust-model always --encrypt -r "$GPG_RECIPIENT" --cipher-algo AES256 \
                --output "$encrypted_backup" "$backup_file"
        else
            # Use symmetric encryption with passphrase
            local passphrase="${GPG_PASSPHRASE:-$(openssl rand -base64 32)}"
            echo "$passphrase" | gpg --batch --yes --passphrase-fd 0 \
                --symmetric --cipher-algo AES256 --output "$encrypted_backup" "$backup_file"
            
            # Store passphrase securely (in production, use proper secret management)
            echo "$passphrase" > "${encrypted_backup}.key"
            chmod 600 "${encrypted_backup}.key"
            log "INFO" "Encryption passphrase stored in ${encrypted_backup}.key"
        fi
        
        if [ -f "$encrypted_backup" ]; then
            rm "$backup_file" # Remove unencrypted version
            backup_file="$encrypted_backup"
            log "INFO" "Backup encrypted: $backup_file"
        else
            log "WARN" "Encryption failed, keeping unencrypted backup"
        fi
    fi
    
    # Create metadata file
    local metadata_file="${backup_file}.meta"
    cat > "$metadata_file" << EOF
{
    "backup_name": "$backup_name",
    "database": "$DB_NAME",
    "host": "$DB_HOST",
    "port": $DB_PORT,
    "user": "$DB_USER",
    "timestamp": "$timestamp",
    "date": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "size": "$(du -b "$backup_file" | cut -f1)",
    "compressed": true,
    "encrypted": $ENCRYPTION_ENABLED,
    "format": "custom",
    "pg_version": "$(pg_dump --version | head -n1)",
    "checksum": "$(sha256sum "$backup_file" | cut -d' ' -f1)"
}
EOF
    
    log "INFO" "Metadata file created: $metadata_file"
    
    # Upload to cloud storage if enabled
    if [ "$S3_ENABLED" = "true" ] && [ -n "$S3_BUCKET" ]; then
        upload_to_s3 "$backup_file" "$metadata_file"
    fi
    
    echo "$backup_file"
}

# Upload backup to S3
upload_to_s3() {
    local backup_file=$1
    local metadata_file=$2
    
    log "INFO" "Uploading backup to S3..."
    
    local s3_backup_key="$S3_PREFIX/$(basename "$backup_file")"
    local s3_metadata_key="$S3_PREFIX/$(basename "$metadata_file")"
    
    # Upload backup file
    if aws s3 cp "$backup_file" "s3://$S3_BUCKET/$s3_backup_key" \
        --region "$AWS_REGION" \
        --storage-class STANDARD_IA \
        --metadata "backup-type=database,database-name=$DB_NAME,hostname=$(hostname)" \
        2>&1 | tee -a "$LOG_FILE"; then
        log "INFO" "Backup uploaded to s3://$S3_BUCKET/$s3_backup_key"
    else
        log "ERROR" "Failed to upload backup to S3"
        return 1
    fi
    
    # Upload metadata file
    if aws s3 cp "$metadata_file" "s3://$S3_BUCKET/$s3_metadata_key" \
        --region "$AWS_REGION" \
        --content-type "application/json" \
        2>&1 | tee -a "$LOG_FILE"; then
        log "INFO" "Metadata uploaded to s3://$S3_BUCKET/$s3_metadata_key"
    else
        log "WARN" "Failed to upload metadata to S3"
    fi
    
    # Set lifecycle policy if not exists
    local lifecycle_config="{
        \"Rules\": [{
            \"ID\": \"NovaCronBackupLifecycle\",
            \"Status\": \"Enabled\",
            \"Filter\": {\"Prefix\": \"$S3_PREFIX/\"},
            \"Transitions\": [{
                \"Days\": 30,
                \"StorageClass\": \"GLACIER\"
            }, {
                \"Days\": 90,
                \"StorageClass\": \"DEEP_ARCHIVE\"
            }],
            \"Expiration\": {
                \"Days\": $((RETENTION_DAYS + 365))
            }
        }]
    }"
    
    echo "$lifecycle_config" | aws s3api put-bucket-lifecycle-configuration \
        --bucket "$S3_BUCKET" \
        --lifecycle-configuration file:///dev/stdin \
        2>/dev/null || log "WARN" "Could not set S3 lifecycle policy"
}

# Cleanup old backups
cleanup_old_backups() {
    log "INFO" "Cleaning up backups older than $RETENTION_DAYS days..."
    
    local deleted_count=0
    
    # Local cleanup
    while IFS= read -r -d '' file; do
        if [[ "$file" =~ novacron_backup_[0-9]{8}_[0-9]{6} ]]; then
            rm -f "$file"
            ((deleted_count++))
            log "INFO" "Deleted old backup: $(basename "$file")"
        fi
    done < <(find "$BACKUP_DIR" -name "novacron_backup_*" -type f -mtime +$RETENTION_DAYS -print0)
    
    # S3 cleanup (if enabled)
    if [ "$S3_ENABLED" = "true" ] && [ -n "$S3_BUCKET" ]; then
        local cutoff_date=$(date -d "$RETENTION_DAYS days ago" '+%Y-%m-%d')
        
        aws s3api list-objects-v2 \
            --bucket "$S3_BUCKET" \
            --prefix "$S3_PREFIX/" \
            --query "Contents[?LastModified<='$cutoff_date'].Key" \
            --output text | while read -r key; do
                if [ -n "$key" ] && [ "$key" != "None" ]; then
                    aws s3 rm "s3://$S3_BUCKET/$key" 2>/dev/null && \
                        log "INFO" "Deleted old S3 backup: $key"
                fi
            done
    fi
    
    log "INFO" "Cleanup completed. Deleted $deleted_count local files"
}

# Verify backup integrity
verify_backup() {
    local backup_file=$1
    
    log "INFO" "Verifying backup integrity..."
    
    # Check if file exists and is readable
    if [ ! -r "$backup_file" ]; then
        error_exit "Backup file is not readable: $backup_file"
    fi
    
    # Check file size (must be > 1MB for a real database backup)
    local file_size=$(du -b "$backup_file" | cut -f1)
    if [ "$file_size" -lt 1048576 ]; then
        log "WARN" "Backup file seems too small: $file_size bytes"
    fi
    
    # Verify checksum if metadata exists
    local metadata_file="${backup_file}.meta"
    if [ -f "$metadata_file" ]; then
        local stored_checksum=$(grep '"checksum"' "$metadata_file" | cut -d'"' -f4)
        local actual_checksum=$(sha256sum "$backup_file" | cut -d' ' -f1)
        
        if [ "$stored_checksum" = "$actual_checksum" ]; then
            log "INFO" "Backup checksum verification passed"
        else
            error_exit "Backup checksum verification failed"
        fi
    fi
    
    # Test backup by attempting to read it (for unencrypted backups)
    if [[ "$backup_file" != *.gpg ]]; then
        if [[ "$backup_file" == *.gz ]]; then
            if ! gzip -t "$backup_file"; then
                error_exit "Backup file integrity check failed (gzip test)"
            fi
        else
            # For PostgreSQL custom format, we can use pg_restore to verify
            if command -v pg_restore &> /dev/null; then
                if ! pg_restore --list "$backup_file" >/dev/null 2>&1; then
                    log "WARN" "pg_restore verification failed, but file may still be valid"
                fi
            fi
        fi
    fi
    
    log "INFO" "Backup verification completed successfully"
}

# Create backup report
create_backup_report() {
    local backup_file=$1
    local start_time=$2
    local end_time=$3
    
    local duration=$((end_time - start_time))
    local backup_size=$(du -h "$backup_file" | cut -f1)
    local backup_size_bytes=$(du -b "$backup_file" | cut -f1)
    
    local report_file="$BACKUP_DIR/backup_report_$(date +%Y%m%d_%H%M%S).json"
    
    cat > "$report_file" << EOF
{
    "backup_summary": {
        "status": "success",
        "backup_file": "$backup_file",
        "database": "$DB_NAME",
        "host": "$DB_HOST",
        "start_time": "$(date -d @$start_time -Iseconds)",
        "end_time": "$(date -d @$end_time -Iseconds)",
        "duration_seconds": $duration,
        "backup_size": "$backup_size",
        "backup_size_bytes": $backup_size_bytes,
        "encryption_enabled": $ENCRYPTION_ENABLED,
        "cloud_upload": $S3_ENABLED,
        "retention_days": $RETENTION_DAYS
    },
    "environment": {
        "hostname": "$(hostname)",
        "pg_version": "$(pg_dump --version | head -n1)",
        "script_version": "1.0.0"
    }
}
EOF
    
    log "INFO" "Backup report created: $report_file"
}

# Main backup function
main() {
    local start_time=$(date +%s)
    
    log "INFO" "NovaCron Database Backup v1.0.0 started"
    log "INFO" "Database: $DB_NAME at $DB_HOST:$DB_PORT"
    log "INFO" "Backup directory: $BACKUP_DIR"
    log "INFO" "Retention period: $RETENTION_DAYS days"
    
    check_dependencies
    create_backup_dir
    test_database_connection
    
    local backup_file
    backup_file=$(create_backup)
    
    verify_backup "$backup_file"
    cleanup_old_backups
    
    local end_time=$(date +%s)
    create_backup_report "$backup_file" "$start_time" "$end_time"
    
    local duration=$((end_time - start_time))
    local backup_size=$(du -h "$backup_file" | cut -f1)
    
    success_exit "Backup completed in ${duration}s, size: $backup_size"
}

# Display usage information
usage() {
    cat << EOF
NovaCron Database Backup Script

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    -v, --verbose       Enable verbose output
    -n, --dry-run       Show what would be done without executing

Environment Variables:
    DB_HOST             Database host (default: localhost)
    DB_PORT             Database port (default: 5432)
    DB_NAME             Database name (default: novacron)
    DB_USER             Database user (default: novacron)
    DB_PASSWORD         Database password
    DATABASE_URL        Full database URL (overrides individual settings)
    
    BACKUP_DIR          Backup directory (default: /var/backups/novacron)
    LOG_DIR             Log directory (default: /var/log/novacron)
    RETENTION_DAYS      Backup retention in days (default: 30)
    
    ENCRYPTION_ENABLED  Enable GPG encryption (default: true)
    GPG_RECIPIENT       GPG recipient for encryption
    GPG_PASSPHRASE      GPG passphrase for symmetric encryption
    
    S3_ENABLED          Enable S3 upload (default: false)
    S3_BUCKET           S3 bucket name
    S3_PREFIX           S3 prefix (default: novacron/database)
    AWS_REGION          AWS region (default: us-west-2)
    
    BACKUP_WEBHOOK_URL  Webhook URL for notifications
    EMAIL_ENABLED       Enable email notifications (default: false)
    EMAIL_TO            Email recipient for notifications

Examples:
    $0                                      # Basic backup
    ENCRYPTION_ENABLED=false $0             # Backup without encryption
    S3_ENABLED=true S3_BUCKET=my-backups $0 # Backup with S3 upload

EOF
}

# Trap signals for graceful shutdown
trap 'error_exit "Interrupted by signal"' INT TERM

# Parse command line arguments
VERBOSE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set verbose logging if requested
if [ "$VERBOSE" = "true" ]; then
    set -x
fi

# Execute dry run or main function
if [ "$DRY_RUN" = "true" ]; then
    log "INFO" "DRY RUN - No actual backup will be performed"
    log "INFO" "Would backup database: $DB_NAME at $DB_HOST:$DB_PORT"
    log "INFO" "Would store backup in: $BACKUP_DIR"
    log "INFO" "Would retain backups for: $RETENTION_DAYS days"
    [ "$ENCRYPTION_ENABLED" = "true" ] && log "INFO" "Would encrypt backup with GPG"
    [ "$S3_ENABLED" = "true" ] && log "INFO" "Would upload to S3 bucket: $S3_BUCKET"
    exit 0
else
    main "$@"
fi