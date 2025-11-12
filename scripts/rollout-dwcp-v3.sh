#!/bin/bash
#
# DWCP v3 Gradual Rollout Script
# Feature flag-based gradual deployment with monitoring
#

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
IMAGE=""
NAMESPACE="production"
INITIAL_PERCENTAGE=10
STAGES=(10 50 100)
WAIT_TIME=300  # 5 minutes between stages
ERROR_THRESHOLD=5  # Max error rate percentage

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}✅ $*${NC}"
}

log_error() {
    echo -e "${RED}❌ $*${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $*${NC}"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Gradual rollout of DWCP v3 with feature flags

OPTIONS:
    --image IMAGE           Docker image to deploy (required)
    --namespace NS          Kubernetes namespace [default: production]
    --percentage PCT        Target rollout percentage (10|50|100) [default: gradual]
    --wait-time SEC         Wait time between stages in seconds [default: 300]
    --error-threshold PCT   Maximum error rate threshold [default: 5]
    -h, --help             Show this help message

EXAMPLES:
    # Gradual rollout (10% -> 50% -> 100%)
    $0 --image ghcr.io/novacron/dwcp-v3:v3.0.0

    # Direct rollout to 50%
    $0 --image ghcr.io/novacron/dwcp-v3:v3.0.0 --percentage 50

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image)
            IMAGE="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --percentage)
            STAGES=("$2")
            shift 2
            ;;
        --wait-time)
            WAIT_TIME="$2"
            shift 2
            ;;
        --error-threshold)
            ERROR_THRESHOLD="$2"
            shift 2
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

# Validate
if [ -z "$IMAGE" ]; then
    log_error "Image is required"
    usage
fi

log "Starting DWCP v3 gradual rollout"
log "Image: $IMAGE"
log "Namespace: $NAMESPACE"
log "Stages: ${STAGES[*]}"
log "Wait time: ${WAIT_TIME}s"

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Get current metrics
get_metrics() {
    local pod_selector="app=dwcp-v3"

    # Get error rate from Prometheus
    local error_rate=$(kubectl exec -n "$NAMESPACE" \
        $(kubectl get pods -n "$NAMESPACE" -l app=prometheus -o jsonpath='{.items[0].metadata.name}') \
        -- wget -qO- 'http://localhost:9090/api/v1/query?query=rate(dwcp_errors_total[5m])' 2>/dev/null | \
        grep -o '"value":\[[^]]*\]' | grep -o '[0-9.]*' | tail -1 || echo "0")

    echo "$error_rate"
}

# Check metrics health
check_metrics_health() {
    log "Checking application metrics..."

    local error_rate=$(get_metrics)
    local error_pct=$(echo "$error_rate * 100" | bc)

    log "Current error rate: ${error_pct}%"

    if (( $(echo "$error_pct > $ERROR_THRESHOLD" | bc -l) )); then
        log_error "Error rate ${error_pct}% exceeds threshold ${ERROR_THRESHOLD}%"
        return 1
    fi

    log_success "Metrics are healthy"
    return 0
}

# Set feature flag percentage
set_feature_flag() {
    local percentage=$1

    log "Setting feature flag to ${percentage}%..."

    # Update ConfigMap with rollout percentage
    kubectl create configmap dwcp-v3-rollout \
        --from-literal=percentage="$percentage" \
        -n "$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Restart pods to pick up new configuration
    kubectl rollout restart deployment/dwcp-v3 -n "$NAMESPACE"
    kubectl rollout status deployment/dwcp-v3 -n "$NAMESPACE" --timeout=5m

    log_success "Feature flag set to ${percentage}%"
}

# Deploy new version
deploy_version() {
    log "Deploying new version: $IMAGE"

    kubectl set image deployment/dwcp-v3 \
        dwcp-v3="$IMAGE" \
        -n "$NAMESPACE"

    # Wait for rollout
    if ! kubectl rollout status deployment/dwcp-v3 -n "$NAMESPACE" --timeout=10m; then
        log_error "Deployment rollout failed"
        return 1
    fi

    log_success "Version deployed"
}

# Monitor stage
monitor_stage() {
    local percentage=$1
    local duration=$2

    log "Monitoring ${percentage}% rollout for ${duration}s..."

    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    local check_interval=30

    while [ $(date +%s) -lt $end_time ]; do
        if ! check_metrics_health; then
            log_error "Metrics check failed during monitoring"
            return 1
        fi

        local remaining=$((end_time - $(date +%s)))
        log "Monitoring... ${remaining}s remaining"

        sleep "$check_interval"
    done

    log_success "Monitoring completed successfully"
    return 0
}

# Rollback
rollback() {
    log_error "Initiating rollback..."

    # Set feature flag to 0%
    set_feature_flag 0

    # Rollback deployment
    kubectl rollout undo deployment/dwcp-v3 -n "$NAMESPACE"
    kubectl rollout status deployment/dwcp-v3 -n "$NAMESPACE" --timeout=5m

    # Notify
    npx claude-flow@alpha hooks notify --message "⚠️  DWCP v3 rollout rolled back" || true

    log_success "Rollback completed"
}

# Execute rollout stage
execute_stage() {
    local percentage=$1

    log "=========================================="
    log "Starting ${percentage}% rollout stage"
    log "=========================================="

    # Set feature flag
    if ! set_feature_flag "$percentage"; then
        return 1
    fi

    # Wait for traffic to stabilize
    sleep 30

    # Monitor stage
    if ! monitor_stage "$percentage" "$WAIT_TIME"; then
        log_error "Stage ${percentage}% failed health checks"
        return 1
    fi

    log_success "Stage ${percentage}% completed successfully"
    return 0
}

# Main rollout
main() {
    check_prerequisites

    # Deploy new version (but at 0% traffic)
    log "Deploying new version at 0% traffic..."
    set_feature_flag 0

    if ! deploy_version; then
        log_error "Initial deployment failed"
        exit 1
    fi

    # Execute each stage
    for stage in "${STAGES[@]}"; do
        if ! execute_stage "$stage"; then
            log_error "Rollout failed at ${stage}%"
            rollback
            exit 1
        fi

        if [ "$stage" -ne 100 ]; then
            log "Waiting before next stage..."
            sleep "$WAIT_TIME"
        fi
    done

    # Notify success
    npx claude-flow@alpha hooks notify --message "✅ DWCP v3 rollout completed successfully to 100%" || true

    log_success "🎉 DWCP v3 rollout completed successfully!"
    log "All stages passed health checks"
    log "Deployment is now at 100%"
}

# Run main function
main
