#!/bin/bash

# Kubernetes Production Deployment Script
# Usage: ./deploy-k8s.sh [staging|production]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENVIRONMENT="${1:-staging}"

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

log "Starting Kubernetes deployment for $ENVIRONMENT"

# Load environment variables
ENV_FILE="$PROJECT_ROOT/deployment/configs/.env.$ENVIRONMENT"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
    log "Loaded environment configuration from $ENV_FILE"
else
    error "Environment file not found: $ENV_FILE"
fi

# Check cluster connectivity
check_cluster() {
    log "Checking Kubernetes cluster connectivity..."
    
    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if we have proper permissions
    if ! kubectl auth can-i create deployments &>/dev/null; then
        error "Insufficient permissions to deploy to cluster"
    fi
    
    success "Cluster connectivity verified"
}

# Create namespace and RBAC
create_namespace() {
    log "Creating namespace and RBAC..."
    
    # Create namespace
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/namespace.yaml"
    
    # Create service account and RBAC
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/rbac.yaml"
    
    success "Namespace and RBAC created"
}

# Create secrets
create_secrets() {
    log "Creating Kubernetes secrets..."
    
    # Database secrets
    kubectl create secret generic postgres-credentials \
        --from-literal=username="$POSTGRES_USER" \
        --from-literal=password="$POSTGRES_PASSWORD" \
        --namespace=novacron \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # JWT secret
    kubectl create secret generic jwt-secret \
        --from-literal=secret="$JWT_SECRET" \
        --namespace=novacron \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Redis secret
    kubectl create secret generic redis-credentials \
        --from-literal=password="$REDIS_PASSWORD" \
        --namespace=novacron \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Grafana admin secret
    kubectl create secret generic grafana-credentials \
        --from-literal=admin-user="$GRAFANA_ADMIN_USER" \
        --from-literal=admin-password="$GRAFANA_ADMIN_PASSWORD" \
        --namespace=novacron \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # TLS secrets (if certificates exist)
    if [[ -f "$PROJECT_ROOT/deployment/ssl/cert.pem" && -f "$PROJECT_ROOT/deployment/ssl/key.pem" ]]; then
        kubectl create secret tls novacron-tls \
            --cert="$PROJECT_ROOT/deployment/ssl/cert.pem" \
            --key="$PROJECT_ROOT/deployment/ssl/key.pem" \
            --namespace=novacron \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    success "Secrets created"
}

# Deploy persistent volumes
deploy_storage() {
    log "Deploying persistent storage..."
    
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/storage.yaml"
    
    # Wait for storage to be ready
    kubectl wait --for=condition=Bound pvc/postgres-pvc -n novacron --timeout=300s
    kubectl wait --for=condition=Bound pvc/grafana-pvc -n novacron --timeout=300s
    kubectl wait --for=condition=Bound pvc/prometheus-pvc -n novacron --timeout=300s
    
    success "Storage deployed"
}

# Deploy configuration
deploy_config() {
    log "Deploying configuration..."
    
    # Create ConfigMaps
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/configmap.yaml"
    
    success "Configuration deployed"
}

# Deploy databases and dependencies
deploy_dependencies() {
    log "Deploying databases and dependencies..."
    
    # Deploy PostgreSQL
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/postgres.yaml"
    
    # Deploy Redis
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/redis.yaml"
    
    # Wait for databases to be ready
    kubectl wait --for=condition=ready pod -l app=postgres -n novacron --timeout=600s
    kubectl wait --for=condition=ready pod -l app=redis -n novacron --timeout=300s
    
    success "Dependencies deployed"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Create migration job
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/migration-job.yaml"
    
    # Wait for migration to complete
    kubectl wait --for=condition=complete job/novacron-migration -n novacron --timeout=600s
    
    # Check if migration was successful
    if kubectl get job novacron-migration -n novacron -o jsonpath='{.status.conditions[?(@.type=="Failed")]}' | grep -q Failed; then
        error "Database migration failed"
    fi
    
    success "Database migrations completed"
}

# Deploy main applications
deploy_applications() {
    log "Deploying main applications..."
    
    # Deploy API server
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/api-deployment.yaml"
    
    # Deploy frontend
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/frontend-deployment.yaml"
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available deployment/novacron-api -n novacron --timeout=600s
    kubectl wait --for=condition=available deployment/novacron-frontend -n novacron --timeout=600s
    
    success "Applications deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring stack..."
    
    # Deploy Prometheus
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/prometheus.yaml"
    
    # Deploy Grafana
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/grafana.yaml"
    
    # Wait for monitoring to be ready
    kubectl wait --for=condition=available deployment/prometheus -n novacron --timeout=600s
    kubectl wait --for=condition=available deployment/grafana -n novacron --timeout=600s
    
    success "Monitoring stack deployed"
}

# Deploy services and ingress
deploy_networking() {
    log "Deploying services and ingress..."
    
    # Deploy services
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/services.yaml"
    
    # Deploy ingress
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/ingress.yaml"
    
    success "Networking deployed"
}

# Deploy operators (production only)
deploy_operators() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Deploying operators for production..."
        
        # Deploy NovaCron operator
        if [[ -f "$PROJECT_ROOT/deployment/kubernetes/operator.yaml" ]]; then
            kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/operator.yaml"
            success "Operators deployed"
        else
            warn "Operator manifest not found, skipping"
        fi
    fi
}

# Setup autoscaling
setup_autoscaling() {
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Setting up autoscaling..."
        
        # Deploy HPA for API
        kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/hpa.yaml"
        
        # Deploy VPA (if available)
        if kubectl get crd verticalpodautoscalers.autoscaling.k8s.io &>/dev/null; then
            kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/vpa.yaml" || warn "VPA not available"
        fi
        
        success "Autoscaling configured"
    fi
}

# Deploy backup jobs
deploy_backup_jobs() {
    log "Deploying backup jobs..."
    
    kubectl apply -f "$PROJECT_ROOT/deployment/kubernetes/backup-cronjob.yaml"
    
    success "Backup jobs deployed"
}

# Validate deployment
validate_deployment() {
    log "Validating deployment..."
    
    # Check all pods are running
    local failed_pods=$(kubectl get pods -n novacron --field-selector=status.phase!=Running,status.phase!=Succeeded -o name)
    if [[ -n "$failed_pods" ]]; then
        error "Some pods are not running: $failed_pods"
    fi
    
    # Check services have endpoints
    local services=("novacron-api" "novacron-frontend" "postgres" "redis")
    for service in "${services[@]}"; do
        local endpoints=$(kubectl get endpoints "$service" -n novacron -o jsonpath='{.subsets[0].addresses}')
        if [[ -z "$endpoints" || "$endpoints" == "null" ]]; then
            error "Service $service has no endpoints"
        fi
    done
    
    success "Deployment validation passed"
}

# Print deployment summary
print_summary() {
    echo ""
    echo "=== Deployment Summary ==="
    echo "Environment: $ENVIRONMENT"
    echo "Namespace: novacron"
    echo "Deployed at: $(date)"
    echo ""
    
    log "Pod status:"
    kubectl get pods -n novacron
    
    echo ""
    log "Service status:"
    kubectl get services -n novacron
    
    echo ""
    log "Ingress status:"
    kubectl get ingress -n novacron
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo ""
        echo "🚀 Production deployment completed!"
        echo "Access the application at: https://${DOMAIN_NAME:-novacron.local}"
    else
        echo ""
        echo "🧪 Staging deployment completed!"
        echo "Use port-forward to access services:"
        echo "kubectl port-forward -n novacron service/novacron-frontend 8080:80"
        echo "kubectl port-forward -n novacron service/novacron-api 8090:8090"
    fi
}

# Main execution
main() {
    log "=== Starting Kubernetes Deployment ==="
    
    check_cluster
    create_namespace
    create_secrets
    deploy_storage
    deploy_config
    deploy_dependencies
    run_migrations
    deploy_applications
    deploy_monitoring
    deploy_networking
    deploy_operators
    setup_autoscaling
    deploy_backup_jobs
    validate_deployment
    print_summary
    
    success "=== Kubernetes Deployment Completed ==="
}

# Rollback function
rollback() {
    local deployment_name="$1"
    log "Rolling back deployment: $deployment_name"
    
    kubectl rollout undo deployment "$deployment_name" -n novacron
    kubectl rollout status deployment "$deployment_name" -n novacron --timeout=600s
    
    success "Rollback completed for $deployment_name"
}

# Error handling with rollback option
error_handler() {
    error "Kubernetes deployment failed at line $LINENO"
    
    read -p "Do you want to rollback? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Initiating rollback..."
        rollback novacron-api || true
        rollback novacron-frontend || true
    fi
    exit 1
}

trap error_handler ERR

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        if [[ $# -ne 2 ]]; then
            error "Usage: $0 rollback <deployment-name>"
        fi
        rollback "$2"
        ;;
    *)
        error "Unknown command: $1"
        ;;
esac