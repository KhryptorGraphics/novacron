#!/bin/bash

# NovaCron Production Rollback Script
# Usage: ./rollback.sh [docker-swarm|kubernetes] [version|previous]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
DEPLOYMENT_TYPE="${1:-}"
ROLLBACK_TARGET="${2:-previous}"
BACKUP_BEFORE_ROLLBACK="${BACKUP_BEFORE_ROLLBACK:-true}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"; exit 1; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"; }

# Validate arguments
if [[ -z "$DEPLOYMENT_TYPE" ]]; then
    error "Usage: $0 [docker-swarm|kubernetes] [version|previous]"
fi

if [[ "$DEPLOYMENT_TYPE" != "docker-swarm" && "$DEPLOYMENT_TYPE" != "kubernetes" ]]; then
    error "Invalid deployment type. Use 'docker-swarm' or 'kubernetes'"
fi

log "Starting NovaCron rollback: $DEPLOYMENT_TYPE to $ROLLBACK_TARGET"

# Confirmation prompt
confirm_rollback() {
    echo ""
    warn "⚠️  ROLLBACK OPERATION ⚠️"
    warn "This will rollback the NovaCron deployment"
    warn "Deployment Type: $DEPLOYMENT_TYPE"
    warn "Rollback Target: $ROLLBACK_TARGET"
    echo ""
    
    if [[ "$BACKUP_BEFORE_ROLLBACK" == "true" ]]; then
        warn "A backup will be created before rollback"
    fi
    
    echo ""
    read -p "Are you sure you want to proceed? (type 'yes' to confirm): " -r
    if [[ ! "$REPLY" == "yes" ]]; then
        log "Rollback cancelled by user"
        exit 0
    fi
    
    log "Rollback confirmed, proceeding..."
}

# Create backup before rollback
create_pre_rollback_backup() {
    if [[ "$BACKUP_BEFORE_ROLLBACK" == "true" ]]; then
        log "Creating pre-rollback backup..."
        if [[ -f "$SCRIPT_DIR/backup/create-backup.sh" ]]; then
            "$SCRIPT_DIR/backup/create-backup.sh" "pre-rollback-$(date +%Y%m%d-%H%M%S)"
            success "Pre-rollback backup created"
        else
            warn "Backup script not found, skipping backup"
        fi
    fi
}

# Docker Swarm rollback
rollback_docker_swarm() {
    log "Performing Docker Swarm rollback..."
    
    # Get list of services
    local services=$(docker service ls --format "{{.Name}}" | grep "^novacron_")
    
    if [[ -z "$services" ]]; then
        error "No NovaCron services found in Docker Swarm"
    fi
    
    # Rollback each service
    for service in $services; do
        log "Rolling back service: $service"
        
        if [[ "$ROLLBACK_TARGET" == "previous" ]]; then
            # Rollback to previous version
            if docker service rollback "$service" --detach=false; then
                success "Rolled back service: $service"
            else
                error "Failed to rollback service: $service"
            fi
        else
            # Rollback to specific version
            log "Rolling back $service to version: $ROLLBACK_TARGET"
            local image_name=$(docker service inspect "$service" --format "{{.Spec.TaskTemplate.ContainerSpec.Image}}" | cut -d':' -f1)
            
            if docker service update --image "$image_name:$ROLLBACK_TARGET" "$service" --detach=false; then
                success "Updated $service to version: $ROLLBACK_TARGET"
            else
                error "Failed to update $service to version: $ROLLBACK_TARGET"
            fi
        fi
        
        # Wait for service to stabilize
        sleep 10
        
        # Verify service health
        local replicas=$(docker service ls --filter "name=$service" --format "{{.Replicas}}")
        if [[ "$replicas" =~ ^[0-9]+/[0-9]+$ ]] && [[ "${replicas%/*}" == "${replicas#*/}" ]]; then
            success "Service $service is healthy after rollback"
        else
            warn "Service $service may not be fully healthy: $replicas"
        fi
    done
    
    success "Docker Swarm rollback completed"
}

# Kubernetes rollback
rollback_kubernetes() {
    log "Performing Kubernetes rollback..."
    
    # Check cluster connectivity
    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Get deployments
    local deployments=$(kubectl get deployments -n novacron -o name | sed 's/deployment\.//')
    
    if [[ -z "$deployments" ]]; then
        error "No NovaCron deployments found in Kubernetes"
    fi
    
    # Rollback each deployment
    for deployment in $deployments; do
        log "Rolling back deployment: $deployment"
        
        if [[ "$ROLLBACK_TARGET" == "previous" ]]; then
            # Rollback to previous revision
            if kubectl rollout undo deployment/"$deployment" -n novacron; then
                log "Initiated rollback for deployment: $deployment"
                
                # Wait for rollback to complete
                if kubectl rollout status deployment/"$deployment" -n novacron --timeout=600s; then
                    success "Rolled back deployment: $deployment"
                else
                    error "Rollback failed for deployment: $deployment"
                fi
            else
                error "Failed to initiate rollback for deployment: $deployment"
            fi
        else
            # Rollback to specific revision
            log "Rolling back $deployment to revision: $ROLLBACK_TARGET"
            
            if kubectl rollout undo deployment/"$deployment" --to-revision="$ROLLBACK_TARGET" -n novacron; then
                log "Initiated rollback for deployment: $deployment"
                
                # Wait for rollback to complete
                if kubectl rollout status deployment/"$deployment" -n novacron --timeout=600s; then
                    success "Rolled back deployment: $deployment"
                else
                    error "Rollback failed for deployment: $deployment"
                fi
            else
                error "Failed to initiate rollback for deployment: $deployment"
            fi
        fi
    done
    
    success "Kubernetes rollback completed"
}

# Verify rollback
verify_rollback() {
    log "Verifying rollback..."
    
    # Wait for services to stabilize
    sleep 30
    
    # Run health checks
    if [[ -f "$SCRIPT_DIR/validation/health-check.sh" ]]; then
        log "Running health checks..."
        if "$SCRIPT_DIR/validation/health-check.sh" production; then
            success "Health checks passed after rollback"
        else
            error "Health checks failed after rollback"
        fi
    else
        warn "Health check script not found, skipping verification"
    fi
    
    # Run smoke tests
    if [[ -f "$SCRIPT_DIR/validation/smoke-tests.sh" ]]; then
        log "Running smoke tests..."
        if "$SCRIPT_DIR/validation/smoke-tests.sh" production; then
            success "Smoke tests passed after rollback"
        else
            warn "Some smoke tests failed after rollback"
        fi
    else
        warn "Smoke test script not found, skipping smoke tests"
    fi
}

# Get deployment history
show_deployment_history() {
    log "Deployment history:"
    
    case "$DEPLOYMENT_TYPE" in
        "docker-swarm")
            echo "Docker Swarm services:"
            docker service ls --format "table {{.Name}}\t{{.Image}}\t{{.Replicas}}\t{{.Ports}}"
            ;;
        "kubernetes")
            echo "Kubernetes deployments:"
            kubectl get deployments -n novacron -o wide
            
            echo ""
            echo "Rollout history:"
            local deployments=$(kubectl get deployments -n novacron -o name | sed 's/deployment\.//')
            for deployment in $deployments; do
                echo "--- $deployment ---"
                kubectl rollout history deployment/"$deployment" -n novacron
            done
            ;;
    esac
}

# Restore from backup (emergency procedure)
emergency_restore() {
    local backup_name="$1"
    
    warn "Performing emergency restore from backup: $backup_name"
    
    if [[ -f "$SCRIPT_DIR/backup/restore-backup.sh" ]]; then
        "$SCRIPT_DIR/backup/restore-backup.sh" "$backup_name"
        success "Emergency restore completed"
    else
        error "Backup restore script not found"
    fi
}

# Main execution
main() {
    log "=== NovaCron Rollback Process Started ==="
    
    # Show current deployment state
    show_deployment_history
    
    # Confirm rollback
    confirm_rollback
    
    # Create backup
    create_pre_rollback_backup
    
    # Perform rollback based on deployment type
    case "$DEPLOYMENT_TYPE" in
        "docker-swarm")
            rollback_docker_swarm
            ;;
        "kubernetes")
            rollback_kubernetes
            ;;
    esac
    
    # Verify rollback
    verify_rollback
    
    success "=== NovaCron Rollback Process Completed ==="
    
    echo ""
    echo "=== Post-Rollback Status ==="
    show_deployment_history
    echo ""
    
    log "Rollback completed successfully"
    log "Monitor the application closely for any issues"
    
    # Provide troubleshooting information
    echo ""
    echo "=== Troubleshooting ==="
    echo "If issues persist:"
    echo "1. Check logs: docker service logs <service> (Swarm) or kubectl logs -n novacron <pod> (K8s)"
    echo "2. Run health check: $SCRIPT_DIR/validation/health-check.sh production"
    echo "3. Emergency restore: $0 emergency-restore <backup-name>"
    echo ""
}

# Handle special commands
case "${1:-}" in
    "emergency-restore")
        if [[ $# -ne 2 ]]; then
            error "Usage: $0 emergency-restore <backup-name>"
        fi
        emergency_restore "$2"
        exit 0
        ;;
    "history")
        show_deployment_history
        exit 0
        ;;
    *)
        # Continue with normal rollback
        ;;
esac

# Error handling
trap 'error "Rollback failed at line $LINENO"' ERR

# Run main function
main