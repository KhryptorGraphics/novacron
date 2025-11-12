#!/bin/bash
#
# DWCP v3 Deployment Script
# Automated deployment with validation, health checks, and rollback capability
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/dwcp-v3-deploy-${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="dwcp-v3"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_TESTS="${SKIP_TESTS:-false}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✅ $*${NC}" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}❌ $*${NC}" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $*${NC}" | tee -a "$LOG_FILE"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy DWCP v3 to Kubernetes cluster

OPTIONS:
    -e, --environment ENV    Target environment (staging|production) [default: staging]
    -t, --tag TAG           Docker image tag [default: latest]
    -n, --namespace NS      Kubernetes namespace [default: dwcp-v3]
    --dry-run              Perform a dry run without actual deployment
    --skip-tests           Skip pre-deployment tests
    --no-rollback          Disable automatic rollback on failure
    -h, --help             Show this help message

EXAMPLES:
    # Deploy to staging
    $0 --environment staging --tag v3.0.0

    # Deploy to production with specific tag
    $0 -e production -t v3.0.0

    # Dry run deployment
    $0 --dry-run --environment production

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    exit 1
fi

log "Starting DWCP v3 deployment"
log "Environment: $ENVIRONMENT"
log "Namespace: $NAMESPACE"
log "Image Tag: $IMAGE_TAG"
log "Dry Run: $DRY_RUN"

# Pre-deployment checks
check_prerequisites() {
    log "Checking prerequisites..."

    local missing_tools=()

    for tool in kubectl helm docker; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi

    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist, creating..."
        if [[ "$DRY_RUN" == "false" ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi

    log_success "Prerequisites check passed"
}

# Backup current deployment
backup_deployment() {
    log "Creating deployment backup..."

    local backup_dir="$PROJECT_ROOT/backups/deployments"
    mkdir -p "$backup_dir"

    local backup_file="$backup_dir/dwcp-v3-${ENVIRONMENT}-${TIMESTAMP}.yaml"

    if kubectl get deployment dwcp-v3 -n "$NAMESPACE" &> /dev/null; then
        kubectl get deployment dwcp-v3 -n "$NAMESPACE" -o yaml > "$backup_file"
        log_success "Backup saved to: $backup_file"
        echo "$backup_file" > /tmp/dwcp-v3-last-backup.txt
    else
        log_warning "No existing deployment to backup"
    fi
}

# Run pre-deployment tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests (--skip-tests flag set)"
        return 0
    fi

    log "Running pre-deployment tests..."

    cd "$PROJECT_ROOT"

    # Run unit tests
    if [[ "$DRY_RUN" == "false" ]]; then
        if ! npm run test:unit; then
            log_error "Unit tests failed"
            return 1
        fi
    fi

    log_success "Tests passed"
}

# Validate deployment manifests
validate_manifests() {
    log "Validating Kubernetes manifests..."

    local manifest_dir="$PROJECT_ROOT/deployments/k8s"

    if ! kubectl apply --dry-run=client -f "$manifest_dir/dwcp-v3-deployment.yaml" &> /dev/null; then
        log_error "Invalid deployment manifest"
        return 1
    fi

    log_success "Manifests validated"
}

# Deploy application
deploy() {
    log "Deploying DWCP v3..."

    local manifest_dir="$PROJECT_ROOT/deployments/k8s"
    local dry_run_flag=""

    if [[ "$DRY_RUN" == "true" ]]; then
        dry_run_flag="--dry-run=client"
        log_warning "Dry run mode - no actual changes will be made"
    fi

    # Apply ConfigMap
    kubectl apply $dry_run_flag -f "$manifest_dir/dwcp-v3-deployment.yaml" -n "$NAMESPACE"

    if [[ "$DRY_RUN" == "false" ]]; then
        # Update image
        kubectl set image deployment/dwcp-v3 \
            dwcp-v3="ghcr.io/novacron/dwcp-v3:${IMAGE_TAG}" \
            -n "$NAMESPACE"

        # Wait for rollout
        log "Waiting for deployment rollout..."
        if ! kubectl rollout status deployment/dwcp-v3 -n "$NAMESPACE" --timeout=10m; then
            log_error "Deployment rollout failed"
            return 1
        fi
    fi

    log_success "Deployment completed"
}

# Health checks
health_checks() {
    log "Running health checks..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "Skipping health checks in dry-run mode"
        return 0
    fi

    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)
        local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 -o jsonpath='{.items[*].metadata.name}' | wc -w)

        if [ "$ready_pods" -eq "$total_pods" ] && [ "$total_pods" -gt 0 ]; then
            log_success "All pods are ready ($ready_pods/$total_pods)"
            break
        fi

        log "Waiting for pods... ($ready_pods/$total_pods ready)"
        sleep 10
        ((attempt++))
    done

    if [ $attempt -eq $max_attempts ]; then
        log_error "Health checks timed out"
        return 1
    fi

    # Test endpoint
    log "Testing application endpoint..."
    local service_url=$(kubectl get service dwcp-v3 -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

    if [ -z "$service_url" ]; then
        service_url=$(kubectl get service dwcp-v3 -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    fi

    if [ -n "$service_url" ]; then
        if curl -sf "http://${service_url}/health" > /dev/null; then
            log_success "Health endpoint is responding"
        else
            log_warning "Health endpoint not accessible externally"
        fi
    fi
}

# Rollback deployment
rollback() {
    log_error "Initiating rollback..."

    if [[ "$ROLLBACK_ON_FAILURE" == "false" ]]; then
        log_warning "Automatic rollback disabled"
        return 0
    fi

    if [ -f /tmp/dwcp-v3-last-backup.txt ]; then
        local backup_file=$(cat /tmp/dwcp-v3-last-backup.txt)
        if [ -f "$backup_file" ]; then
            log "Restoring from backup: $backup_file"
            kubectl apply -f "$backup_file" -n "$NAMESPACE"
            kubectl rollout status deployment/dwcp-v3 -n "$NAMESPACE" --timeout=5m
            log_success "Rollback completed"
            return 0
        fi
    fi

    log "Performing rollback using kubectl..."
    kubectl rollout undo deployment/dwcp-v3 -n "$NAMESPACE"
    kubectl rollout status deployment/dwcp-v3 -n "$NAMESPACE" --timeout=5m
    log_success "Rollback completed"
}

# Post-deployment tasks
post_deployment() {
    log "Running post-deployment tasks..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "Skipping post-deployment tasks in dry-run mode"
        return 0
    fi

    # Update Claude Flow hooks
    npx claude-flow@alpha hooks post-task --task-id "dwcp-v3-deployment-${IMAGE_TAG}" || true

    # Generate deployment report
    cat > "/tmp/deployment-report-${TIMESTAMP}.txt" << EOF
DWCP v3 Deployment Report
========================
Timestamp: ${TIMESTAMP}
Environment: ${ENVIRONMENT}
Namespace: ${NAMESPACE}
Image Tag: ${IMAGE_TAG}
Status: Success

Deployed Pods:
$(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 -o wide)

Services:
$(kubectl get services -n "$NAMESPACE")

Ingress:
$(kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "No ingress configured")
EOF

    log_success "Deployment report saved to: /tmp/deployment-report-${TIMESTAMP}.txt"
}

# Cleanup
cleanup() {
    log "Cleaning up..."
    rm -f /tmp/dwcp-v3-last-backup.txt
}

# Main execution
main() {
    trap cleanup EXIT

    check_prerequisites
    backup_deployment
    validate_manifests

    if ! run_tests; then
        log_error "Tests failed, aborting deployment"
        exit 1
    fi

    if ! deploy; then
        log_error "Deployment failed"
        rollback
        exit 1
    fi

    if ! health_checks; then
        log_error "Health checks failed"
        rollback
        exit 1
    fi

    post_deployment

    log_success "🎉 DWCP v3 deployment completed successfully!"
    log "Log file: $LOG_FILE"
}

# Run main function
main
