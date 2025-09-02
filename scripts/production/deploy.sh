#!/bin/bash

# NovaCron Production Deployment Master Script
# Usage: ./deploy.sh [docker-swarm|kubernetes] [staging|production]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="/var/log/novacron/deployment-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Validate arguments
if [ $# -ne 2 ]; then
    error "Usage: $0 [docker-swarm|kubernetes] [staging|production]"
fi

DEPLOYMENT_TYPE="$1"
ENVIRONMENT="$2"

# Validate deployment type
if [[ "$DEPLOYMENT_TYPE" != "docker-swarm" && "$DEPLOYMENT_TYPE" != "kubernetes" ]]; then
    error "Invalid deployment type. Use 'docker-swarm' or 'kubernetes'"
fi

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    error "Invalid environment. Use 'staging' or 'production'"
fi

log "Starting NovaCron deployment: $DEPLOYMENT_TYPE to $ENVIRONMENT"

# Pre-deployment checks
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root for production
    if [[ "$ENVIRONMENT" == "production" && $EUID -ne 0 ]]; then
        error "Production deployment must be run as root"
    fi
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq" "openssl")
    if [[ "$DEPLOYMENT_TYPE" == "kubernetes" ]]; then
        required_commands+=("kubectl" "helm")
    fi
    
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command not found: $cmd"
        fi
    done
    
    # Check environment file
    local env_file="$PROJECT_ROOT/deployment/configs/.env.$ENVIRONMENT"
    if [[ ! -f "$env_file" ]]; then
        error "Environment file not found: $env_file"
    fi
    
    success "Prerequisites check passed"
}

# Environment validation
validate_environment() {
    log "Validating environment configuration..."
    
    source "$PROJECT_ROOT/deployment/configs/.env.$ENVIRONMENT"
    
    # Required variables
    local required_vars=(
        "POSTGRES_PASSWORD"
        "JWT_SECRET"
        "GRAFANA_ADMIN_PASSWORD"
        "REDIS_PASSWORD"
    )
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable not set: $var"
        fi
    done
    
    # Validate secret strength
    if [[ ${#JWT_SECRET} -lt 32 ]]; then
        error "JWT_SECRET must be at least 32 characters long"
    fi
    
    if [[ ${#POSTGRES_PASSWORD} -lt 12 ]]; then
        error "POSTGRES_PASSWORD must be at least 12 characters long"
    fi
    
    success "Environment validation passed"
}

# Pre-deployment backup (production only)
backup_current_deployment() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Creating pre-deployment backup..."
        "$SCRIPT_DIR/backup/create-backup.sh" "pre-deployment-$(date +%Y%m%d-%H%M%S)"
        success "Pre-deployment backup completed"
    fi
}

# Deploy based on type
deploy() {
    log "Starting deployment process..."
    
    case "$DEPLOYMENT_TYPE" in
        "docker-swarm")
            "$SCRIPT_DIR/docker-swarm/deploy-swarm.sh" "$ENVIRONMENT"
            ;;
        "kubernetes")
            "$SCRIPT_DIR/kubernetes/deploy-k8s.sh" "$ENVIRONMENT"
            ;;
    esac
    
    success "Deployment process completed"
}

# Post-deployment validation
validate_deployment() {
    log "Validating deployment..."
    
    # Wait for services to be ready
    sleep 30
    
    # Run health checks
    "$SCRIPT_DIR/validation/health-check.sh" "$ENVIRONMENT"
    
    # Run smoke tests
    "$SCRIPT_DIR/validation/smoke-tests.sh" "$ENVIRONMENT"
    
    success "Deployment validation passed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring and alerting..."
    "$SCRIPT_DIR/monitoring/setup-monitoring.sh" "$ENVIRONMENT"
    success "Monitoring setup completed"
}

# Main execution
main() {
    log "=== NovaCron Production Deployment Started ==="
    
    check_prerequisites
    validate_environment
    backup_current_deployment
    deploy
    validate_deployment
    setup_monitoring
    
    success "=== NovaCron Production Deployment Completed Successfully ==="
    log "Deployment logs saved to: $LOG_FILE"
    
    # Print deployment summary
    echo ""
    echo "=== Deployment Summary ==="
    echo "Type: $DEPLOYMENT_TYPE"
    echo "Environment: $ENVIRONMENT"
    echo "Deployed at: $(date)"
    echo "Log file: $LOG_FILE"
    echo ""
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "🚀 Production deployment completed successfully!"
        echo "Services should be available at your configured domain."
    else
        echo "🧪 Staging deployment completed successfully!"
        echo "Please run additional tests before promoting to production."
    fi
}

# Error handling
trap 'error "Deployment failed at line $LINENO"' ERR

# Run main function
main "$@"