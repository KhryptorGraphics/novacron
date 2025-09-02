#!/bin/bash

# NovaCron Backup Creation Script
# Usage: ./create-backup.sh [backup_name]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Configuration
BACKUP_NAME="${1:-backup-$(date +%Y%m%d-%H%M%S)}"
BACKUP_DIR="${BACKUP_ROOT:-/opt/novacron/backups}"
LOG_FILE="/var/log/novacron/backup-$(date +%Y%m%d).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"; exit 1; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}" | tee -a "$LOG_FILE"; }

# Ensure directories exist
mkdir -p "$BACKUP_DIR" "$(dirname "$LOG_FILE")"

log "Starting backup: $BACKUP_NAME"

# Load environment configuration
load_config() {
    local env_file="$PROJECT_ROOT/deployment/configs/.env.production"
    if [[ -f "$env_file" ]]; then
        set -a
        source "$env_file"
        set +a
        log "Loaded production configuration"
    else
        warn "Production config not found, using defaults"
    fi
}

# Check if services are running
check_services() {
    log "Checking service status..."
    
    # Check if Docker Swarm is active
    if docker info | grep -q "Swarm: active"; then
        DEPLOYMENT_TYPE="swarm"
        log "Detected Docker Swarm deployment"
    # Check if Kubernetes is available
    elif kubectl cluster-info &>/dev/null; then
        DEPLOYMENT_TYPE="kubernetes"
        log "Detected Kubernetes deployment"
    else
        DEPLOYMENT_TYPE="compose"
        log "Detected Docker Compose deployment"
    fi
}

# Database backup
backup_database() {
    log "Creating database backup..."
    
    local db_backup_file="$BACKUP_DIR/$BACKUP_NAME/database.sql"
    mkdir -p "$(dirname "$db_backup_file")"
    
    case "$DEPLOYMENT_TYPE" in
        "swarm")
            # Get database container name in swarm
            local db_container=$(docker service ps novacron_postgres --format "{{.Name}}.{{.ID}}" | head -1)
            docker exec "$db_container" pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > "$db_backup_file"
            ;;
        "kubernetes")
            # Use kubectl to access database pod
            local db_pod=$(kubectl get pods -n novacron -l app=postgres -o jsonpath='{.items[0].metadata.name}')
            kubectl exec -n novacron "$db_pod" -- pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > "$db_backup_file"
            ;;
        "compose")
            # Standard Docker Compose
            docker-compose -f "$PROJECT_ROOT/docker-compose.yml" exec -T postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > "$db_backup_file"
            ;;
    esac
    
    # Compress database backup
    gzip "$db_backup_file"
    success "Database backup created: ${db_backup_file}.gz"
}

# Configuration backup
backup_configurations() {
    log "Backing up configurations..."
    
    local config_backup_dir="$BACKUP_DIR/$BACKUP_NAME/configs"
    mkdir -p "$config_backup_dir"
    
    # Copy configuration files
    cp -r "$PROJECT_ROOT/deployment/configs" "$config_backup_dir/"
    cp -r "$PROJECT_ROOT/configs" "$config_backup_dir/app-configs" 2>/dev/null || true
    
    # Backup environment files (without secrets)
    for env_file in "$PROJECT_ROOT"/deployment/configs/.env.*; do
        if [[ -f "$env_file" ]]; then
            # Remove sensitive information
            grep -v -E "(PASSWORD|SECRET|KEY)" "$env_file" > "$config_backup_dir/$(basename "$env_file").sanitized" || true
        fi
    done
    
    success "Configuration backup created"
}

# Volume backup
backup_volumes() {
    log "Backing up persistent volumes..."
    
    local volumes_backup_dir="$BACKUP_DIR/$BACKUP_NAME/volumes"
    mkdir -p "$volumes_backup_dir"
    
    case "$DEPLOYMENT_TYPE" in
        "swarm"|"compose")
            # Backup Docker volumes
            local volumes=("prometheus_data" "grafana_data" "redis_data")
            for volume in "${volumes[@]}"; do
                if docker volume inspect "$volume" &>/dev/null; then
                    log "Backing up volume: $volume"
                    docker run --rm -v "$volume":/source -v "$volumes_backup_dir":/backup alpine tar czf "/backup/$volume.tar.gz" -C /source .
                fi
            done
            ;;
        "kubernetes")
            # Backup PVCs
            local pvcs=("prometheus-pvc" "grafana-pvc" "redis-pvc")
            for pvc in "${pvcs[@]}"; do
                if kubectl get pvc "$pvc" -n novacron &>/dev/null; then
                    log "Backing up PVC: $pvc"
                    # Create a backup pod to access PVC data
                    kubectl run backup-pod-"$pvc" --image=alpine --rm -i --restart=Never \
                        --overrides="{\"spec\":{\"containers\":[{\"name\":\"backup\",\"image\":\"alpine\",\"command\":[\"tar\",\"czf\",\"/backup/$pvc.tar.gz\",\"-C\",\"/data\",\".\"],\"volumeMounts\":[{\"name\":\"data\",\"mountPath\":\"/data\"},{\"name\":\"backup\",\"mountPath\":\"/backup\"}]}],\"volumes\":[{\"name\":\"data\",\"persistentVolumeClaim\":{\"claimName\":\"$pvc\"}},{\"name\":\"backup\",\"hostPath\":{\"path\":\"$volumes_backup_dir\"}}]}}" \
                        -n novacron
                fi
            done
            ;;
    esac
    
    success "Volume backup created"
}

# SSL certificates backup
backup_certificates() {
    log "Backing up SSL certificates..."
    
    local cert_backup_dir="$BACKUP_DIR/$BACKUP_NAME/certificates"
    mkdir -p "$cert_backup_dir"
    
    # Copy certificates if they exist
    if [[ -d "$PROJECT_ROOT/deployment/ssl" ]]; then
        cp -r "$PROJECT_ROOT/deployment/ssl" "$cert_backup_dir/"
    fi
    
    # Backup Let's Encrypt certificates
    if [[ -d "/etc/letsencrypt" ]]; then
        sudo cp -r /etc/letsencrypt "$cert_backup_dir/" 2>/dev/null || warn "Could not backup Let's Encrypt certificates"
    fi
    
    success "Certificate backup created"
}

# Create backup manifest
create_manifest() {
    log "Creating backup manifest..."
    
    local manifest_file="$BACKUP_DIR/$BACKUP_NAME/manifest.json"
    
    cat > "$manifest_file" << EOF
{
  "backup_name": "$BACKUP_NAME",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "deployment_type": "$DEPLOYMENT_TYPE",
  "environment": "${ENVIRONMENT:-production}",
  "version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "database": {
    "type": "postgresql",
    "version": "$(docker exec $(docker ps --format "{{.Names}}" | grep postgres | head -1) psql --version 2>/dev/null | awk '{print $3}' || echo 'unknown')"
  },
  "components": [
    "database",
    "configurations", 
    "volumes",
    "certificates"
  ],
  "size_bytes": $(du -sb "$BACKUP_DIR/$BACKUP_NAME" | cut -f1),
  "checksum": "$(find "$BACKUP_DIR/$BACKUP_NAME" -type f -exec sha256sum {} \; | sha256sum | cut -d' ' -f1)"
}
EOF
    
    success "Backup manifest created"
}

# Upload to cloud storage (if configured)
upload_to_cloud() {
    if [[ -n "${BACKUP_S3_BUCKET:-}" ]]; then
        log "Uploading backup to S3..."
        
        # Create archive
        local archive_file="$BACKUP_DIR/$BACKUP_NAME.tar.gz"
        tar czf "$archive_file" -C "$BACKUP_DIR" "$BACKUP_NAME"
        
        # Upload to S3
        if command -v aws &> /dev/null; then
            aws s3 cp "$archive_file" "s3://$BACKUP_S3_BUCKET/novacron/$BACKUP_NAME.tar.gz" \
                --region "${BACKUP_S3_REGION:-us-east-1}" \
                --server-side-encryption AES256
            
            # Clean up local archive
            rm "$archive_file"
            success "Backup uploaded to S3"
        else
            warn "AWS CLI not found, skipping S3 upload"
        fi
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    local retention_days="${BACKUP_RETENTION_DAYS:-30}"
    
    # Remove local backups older than retention period
    find "$BACKUP_DIR" -name "backup-*" -type d -mtime +$retention_days -exec rm -rf {} \; 2>/dev/null || true
    
    # Clean up S3 backups if configured
    if [[ -n "${BACKUP_S3_BUCKET:-}" ]] && command -v aws &> /dev/null; then
        aws s3api list-objects-v2 \
            --bucket "$BACKUP_S3_BUCKET" \
            --prefix "novacron/" \
            --query "Contents[?LastModified<='$(date -d "$retention_days days ago" -u +%Y-%m-%dT%H:%M:%SZ)'].Key" \
            --output text | \
        while read -r key; do
            if [[ -n "$key" && "$key" != "None" ]]; then
                aws s3 rm "s3://$BACKUP_S3_BUCKET/$key"
            fi
        done
    fi
    
    success "Old backups cleaned up"
}

# Main execution
main() {
    log "=== Starting NovaCron Backup Process ==="
    
    load_config
    check_services
    backup_database
    backup_configurations
    backup_volumes
    backup_certificates
    create_manifest
    upload_to_cloud
    cleanup_old_backups
    
    success "=== Backup Process Completed Successfully ==="
    
    # Display backup information
    echo ""
    echo "=== Backup Summary ==="
    echo "Name: $BACKUP_NAME"
    echo "Location: $BACKUP_DIR/$BACKUP_NAME"
    echo "Size: $(du -sh "$BACKUP_DIR/$BACKUP_NAME" | cut -f1)"
    echo "Created: $(date)"
    echo ""
    
    log "Backup process completed successfully"
}

# Error handling
trap 'error "Backup failed at line $LINENO"' ERR

# Run main function
main