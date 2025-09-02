#!/bin/bash

# NovaCron Backup Service Entrypoint
# Runs backup service with cron scheduler

set -euo pipefail

# Configuration
LOG_FILE="/var/log/backup/backup-service.log"
HEALTH_FILE="/tmp/backup-health"
BACKUP_SCRIPT="/scripts/backup-database.sh"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Health check function
update_health() {
    echo "$(date -Iseconds)" > "$HEALTH_FILE"
}

# Initialize
initialize() {
    log "Initializing backup service..."
    
    # Create log directory if not exists
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Initial health update
    update_health
    
    # Verify backup script exists and is executable
    if [ ! -x "$BACKUP_SCRIPT" ]; then
        log "ERROR: Backup script not found or not executable: $BACKUP_SCRIPT"
        exit 1
    fi
    
    # Test database connectivity
    if [ -n "${DATABASE_URL:-}" ]; then
        if pg_isready -d "$DATABASE_URL" -t 10; then
            log "Database connectivity verified"
        else
            log "WARNING: Cannot connect to database"
        fi
    else
        log "WARNING: DATABASE_URL not configured"
    fi
    
    # Test cloud connectivity (if configured)
    if [ "${S3_ENABLED:-false}" = "true" ] && [ -n "${S3_BUCKET:-}" ]; then
        if aws s3 ls "s3://$S3_BUCKET/" &>/dev/null; then
            log "S3 connectivity verified"
        else
            log "WARNING: Cannot connect to S3 bucket: $S3_BUCKET"
        fi
    fi
    
    log "Backup service initialized successfully"
}

# Run manual backup
run_manual_backup() {
    log "Running manual backup..."
    
    if "$BACKUP_SCRIPT"; then
        log "Manual backup completed successfully"
        update_health
        return 0
    else
        log "ERROR: Manual backup failed"
        return 1
    fi
}

# Start cron service
start_cron() {
    log "Starting cron service..."
    
    # Install crontab for backup user
    crontab /etc/crontabs/backup
    
    # Start crond in foreground
    exec crond -f -l 2
}

# Main execution
main() {
    log "NovaCron Backup Service starting..."
    
    # Parse command line arguments
    case "${1:-cron}" in
        "cron")
            initialize
            start_cron
            ;;
        "manual")
            initialize
            run_manual_backup
            ;;
        "test")
            initialize
            log "Test completed successfully"
            ;;
        *)
            log "ERROR: Unknown command: $1"
            echo "Usage: $0 [cron|manual|test]"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"