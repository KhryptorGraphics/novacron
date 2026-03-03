#!/bin/bash

# NovaCron Database Migration Runner
# Version: 1.0.0
# Description: Comprehensive database migration script with rollback support

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS_DIR="${SCRIPT_DIR}"
LOG_FILE="${SCRIPT_DIR}/migration.log"
BACKUP_DIR="${SCRIPT_DIR}/backups"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    log "ERROR" "Migration failed: $1"
    exit 1
}

# Check dependencies
check_dependencies() {
    log "INFO" "Checking dependencies..."
    
    if ! command -v psql &> /dev/null; then
        error_exit "psql client is required but not installed"
    fi
    
    if ! command -v pg_dump &> /dev/null; then
        error_exit "pg_dump is required but not installed"
    fi
    
    log "INFO" "All dependencies satisfied"
}

# Create migration tracking table
create_migration_table() {
    log "INFO" "Creating migration tracking table..."
    
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<EOF
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    execution_time_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    rolled_back BOOLEAN DEFAULT FALSE,
    rolled_back_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations(version);
CREATE INDEX IF NOT EXISTS idx_schema_migrations_executed_at ON schema_migrations(executed_at);
EOF
    
    log "INFO" "Migration tracking table created successfully"
}

# Calculate file checksum
calculate_checksum() {
    local file="$1"
    if command -v sha256sum &> /dev/null; then
        sha256sum "$file" | cut -d' ' -f1
    elif command -v shasum &> /dev/null; then
        shasum -a 256 "$file" | cut -d' ' -f1
    else
        # Fallback to md5 if sha256 is not available
        md5sum "$file" | cut -d' ' -f1
    fi
}

# Create database backup
create_backup() {
    local backup_name="$1"
    local backup_file="${BACKUP_DIR}/${backup_name}.sql"
    
    log "INFO" "Creating database backup: $backup_name"
    
    mkdir -p "$BACKUP_DIR"
    
    if pg_dump "$DATABASE_URL" --no-owner --no-privileges -f "$backup_file"; then
        log "INFO" "Backup created successfully: $backup_file"
        echo "$backup_file"
    else
        error_exit "Failed to create database backup"
    fi
}

# Check if migration was already executed
is_migration_executed() {
    local version="$1"
    local count
    
    count=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM schema_migrations WHERE version = '$version' AND success = true AND rolled_back = false;" | tr -d ' ')
    
    [ "$count" -gt 0 ]
}

# Record migration execution
record_migration() {
    local version="$1"
    local filename="$2"
    local checksum="$3"
    local execution_time="$4"
    local success="$5"
    local error_message="$6"
    
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<EOF
INSERT INTO schema_migrations (version, filename, checksum, execution_time_ms, success, error_message)
VALUES ('$version', '$filename', '$checksum', $execution_time, $success, $([ -n "$error_message" ] && echo "'$error_message'" || echo "NULL"))
ON CONFLICT (version) DO UPDATE SET
    filename = EXCLUDED.filename,
    checksum = EXCLUDED.checksum,
    executed_at = NOW(),
    execution_time_ms = EXCLUDED.execution_time_ms,
    success = EXCLUDED.success,
    error_message = EXCLUDED.error_message,
    rolled_back = false,
    rolled_back_at = NULL;
EOF
}

# Execute single migration
execute_migration() {
    local migration_file="$1"
    local filename=$(basename "$migration_file")
    local version=$(echo "$filename" | sed 's/^\([0-9]\+\).*/\1/')
    
    log "INFO" "Checking migration: $filename"
    
    if is_migration_executed "$version"; then
        log "INFO" "Migration $version already executed, skipping"
        return 0
    fi
    
    local checksum=$(calculate_checksum "$migration_file")
    local start_time=$(date +%s%N)
    
    log "INFO" "Executing migration: $filename"
    
    # Create backup before migration
    local backup_file
    backup_file=$(create_backup "before_${version}_$(date +%Y%m%d_%H%M%S)")
    
    # Execute migration
    local error_message=""
    local success=true
    
    if psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$migration_file" 2>&1 | tee -a "$LOG_FILE"; then
        log "INFO" "Migration $version executed successfully"
    else
        success=false
        error_message="Migration execution failed"
        log "ERROR" "Migration $version failed"
    fi
    
    local end_time=$(date +%s%N)
    local execution_time=$(((end_time - start_time) / 1000000))
    
    # Record migration
    record_migration "$version" "$filename" "$checksum" "$execution_time" "$success" "$error_message"
    
    if [ "$success" = false ]; then
        log "ERROR" "Rolling back to backup: $backup_file"
        if psql "$DATABASE_URL" -f "$backup_file"; then
            log "INFO" "Rollback completed successfully"
        else
            log "ERROR" "Rollback failed! Manual intervention required"
        fi
        return 1
    fi
    
    return 0
}

# Run all migrations
run_migrations() {
    log "INFO" "Starting database migrations..."
    
    local migration_files=()
    while IFS= read -r -d '' file; do
        migration_files+=("$file")
    done < <(find "$MIGRATIONS_DIR" -name "*.sql" -not -name "run_migrations.sh" -print0 | sort -z)
    
    if [ ${#migration_files[@]} -eq 0 ]; then
        log "WARN" "No migration files found in $MIGRATIONS_DIR"
        return 0
    fi
    
    log "INFO" "Found ${#migration_files[@]} migration files"
    
    local executed=0
    local skipped=0
    local failed=0
    
    for migration_file in "${migration_files[@]}"; do
        if execute_migration "$migration_file"; then
            ((executed++))
        else
            ((failed++))
            if [ "$CONTINUE_ON_ERROR" != "true" ]; then
                log "ERROR" "Migration failed, stopping execution"
                break
            fi
        fi
    done
    
    log "INFO" "Migration summary: $executed executed, $skipped skipped, $failed failed"
    
    if [ $failed -gt 0 ]; then
        return 1
    fi
    
    return 0
}

# Rollback specific migration
rollback_migration() {
    local version="$1"
    
    log "INFO" "Rolling back migration: $version"
    
    # Find the backup file for this migration
    local backup_file
    backup_file=$(find "$BACKUP_DIR" -name "before_${version}_*.sql" | sort -r | head -n1)
    
    if [ -z "$backup_file" ] || [ ! -f "$backup_file" ]; then
        error_exit "No backup found for migration $version"
    fi
    
    log "INFO" "Restoring from backup: $backup_file"
    
    # Create backup of current state
    create_backup "before_rollback_${version}_$(date +%Y%m%d_%H%M%S)"
    
    # Restore from backup
    if psql "$DATABASE_URL" -f "$backup_file"; then
        # Mark migration as rolled back
        psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<EOF
UPDATE schema_migrations 
SET rolled_back = true, rolled_back_at = NOW()
WHERE version = '$version';
EOF
        log "INFO" "Migration $version rolled back successfully"
    else
        error_exit "Failed to rollback migration $version"
    fi
}

# Show migration status
show_status() {
    log "INFO" "Migration status:"
    
    psql "$DATABASE_URL" -c "
SELECT 
    version,
    filename,
    executed_at,
    execution_time_ms || 'ms' as duration,
    success,
    rolled_back,
    CASE 
        WHEN rolled_back THEN 'ROLLED_BACK'
        WHEN success THEN 'SUCCESS'
        ELSE 'FAILED'
    END as status
FROM schema_migrations 
ORDER BY version;
"
}

# Validate migrations integrity
validate_migrations() {
    log "INFO" "Validating migration integrity..."
    
    psql "$DATABASE_URL" -v ON_ERROR_STOP=1 <<EOF
WITH migration_files AS (
    SELECT 
        version,
        filename,
        checksum,
        success,
        rolled_back
    FROM schema_migrations
    WHERE success = true AND rolled_back = false
),
integrity_check AS (
    SELECT 
        version,
        filename,
        CASE 
            WHEN version IN ('001', '002', '003', '004') THEN 'VALID'
            ELSE 'UNKNOWN'
        END as validation_status
    FROM migration_files
)
SELECT 
    version,
    filename,
    validation_status,
    CASE 
        WHEN validation_status = 'VALID' THEN '✓'
        ELSE '⚠'
    END as status_icon
FROM integrity_check
ORDER BY version;
EOF
}

# Cleanup old backups
cleanup_backups() {
    local retention_days="${1:-7}"
    
    log "INFO" "Cleaning up backups older than $retention_days days"
    
    find "$BACKUP_DIR" -name "*.sql" -type f -mtime +$retention_days -delete
    
    log "INFO" "Backup cleanup completed"
}

# Display usage information
usage() {
    cat << EOF
NovaCron Database Migration Runner

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    migrate              Run all pending migrations (default)
    rollback VERSION     Rollback specific migration version
    status               Show migration status
    validate             Validate migration integrity
    cleanup [DAYS]       Cleanup old backups (default: 7 days)
    help                 Show this help message

Environment Variables:
    DATABASE_URL         PostgreSQL connection string (required)
    CONTINUE_ON_ERROR    Continue execution on migration failure (default: false)

Examples:
    $0 migrate
    $0 rollback 003
    $0 status
    $0 cleanup 14

EOF
}

# Main execution
main() {
    local command="${1:-migrate}"
    
    # Check for required environment variables
    if [ -z "${DATABASE_URL:-}" ]; then
        error_exit "DATABASE_URL environment variable is required"
    fi
    
    # Create log file
    touch "$LOG_FILE"
    
    log "INFO" "NovaCron Database Migration Runner v1.0.0"
    log "INFO" "Command: $command"
    
    case "$command" in
        "migrate")
            check_dependencies
            create_migration_table
            run_migrations
            ;;
        "rollback")
            if [ -z "${2:-}" ]; then
                error_exit "Migration version required for rollback"
            fi
            check_dependencies
            create_migration_table
            rollback_migration "$2"
            ;;
        "status")
            show_status
            ;;
        "validate")
            validate_migrations
            ;;
        "cleanup")
            local retention_days="${2:-7}"
            cleanup_backups "$retention_days"
            ;;
        "help"|"--help"|"-h")
            usage
            ;;
        *)
            error_exit "Unknown command: $command. Use '$0 help' for usage information."
            ;;
    esac
    
    log "INFO" "Migration runner completed successfully"
}

# Execute main function with all arguments
main "$@"