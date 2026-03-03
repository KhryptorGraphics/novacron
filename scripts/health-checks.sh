#!/bin/bash
#
# DWCP v3 Health Check Script
# Comprehensive health checks for deployed application
#

set -euo pipefail

NAMESPACE="${1:-production}"
SERVICE_NAME="dwcp-v3"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log_success() {
    echo -e "${GREEN}✅ $*${NC}"
}

log_error() {
    echo -e "${RED}❌ $*${NC}"
}

check_pods() {
    log "Checking pod health..."

    local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)
    local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" -o jsonpath='{.items[*].metadata.name}' | wc -w)

    if [ "$ready_pods" -eq "$total_pods" ] && [ "$total_pods" -gt 0 ]; then
        log_success "All pods healthy ($ready_pods/$total_pods)"
        return 0
    else
        log_error "Pod health check failed ($ready_pods/$total_pods ready)"
        kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME"
        return 1
    fi
}

check_endpoints() {
    log "Checking service endpoints..."

    local endpoints=$(kubectl get endpoints -n "$NAMESPACE" "$SERVICE_NAME" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)

    if [ "$endpoints" -gt 0 ]; then
        log_success "Service has $endpoints endpoints"
        return 0
    else
        log_error "No service endpoints available"
        return 1
    fi
}

check_health_endpoint() {
    log "Testing health endpoint..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" -o jsonpath='{.items[0].metadata.name}')

    if kubectl exec -n "$NAMESPACE" "$pod_name" -- curl -sf http://localhost:8080/health > /dev/null; then
        log_success "Health endpoint responding"
        return 0
    else
        log_error "Health endpoint not responding"
        return 1
    fi
}

check_metrics() {
    log "Checking metrics endpoint..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" -o jsonpath='{.items[0].metadata.name}')

    if kubectl exec -n "$NAMESPACE" "$pod_name" -- curl -sf http://localhost:9090/metrics > /dev/null; then
        log_success "Metrics endpoint responding"
        return 0
    else
        log_error "Metrics endpoint not responding"
        return 1
    fi
}

check_redis() {
    log "Checking Redis connectivity..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app="$SERVICE_NAME" -o jsonpath='{.items[0].metadata.name}')

    if kubectl exec -n "$NAMESPACE" "$pod_name" -- sh -c "timeout 5 redis-cli -h dwcp-v3-redis ping" 2>/dev/null | grep -q "PONG"; then
        log_success "Redis connectivity OK"
        return 0
    else
        log_error "Cannot connect to Redis"
        return 1
    fi
}

main() {
    log "Running DWCP v3 health checks for namespace: $NAMESPACE"

    local failed=0

    check_pods || ((failed++))
    check_endpoints || ((failed++))
    check_health_endpoint || ((failed++))
    check_metrics || ((failed++))
    check_redis || ((failed++))

    echo ""
    if [ "$failed" -eq 0 ]; then
        log_success "All health checks passed!"
        exit 0
    else
        log_error "$failed health check(s) failed"
        exit 1
    fi
}

main
