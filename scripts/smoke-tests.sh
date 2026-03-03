#!/bin/bash
#
# DWCP v3 Smoke Tests
# Quick validation of deployed application
#

set -euo pipefail

ENVIRONMENT="${1:-staging}"
NAMESPACE="dwcp-v3"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log_success() {
    echo -e "${GREEN}✅ $*${NC}"
}

log_error() {
    echo -e "${RED}❌ $*${NC}"
}

test_basic_connectivity() {
    log "Testing basic connectivity..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 -o jsonpath='{.items[0].metadata.name}')

    if kubectl exec -n "$NAMESPACE" "$pod_name" -- curl -sf http://localhost:8080/health > /dev/null; then
        log_success "Basic connectivity OK"
        return 0
    else
        log_error "Basic connectivity failed"
        return 1
    fi
}

test_codec_initialization() {
    log "Testing DWCP v3 codec initialization..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 -o jsonpath='{.items[0].metadata.name}')

    # Check logs for successful codec initialization
    if kubectl logs -n "$NAMESPACE" "$pod_name" | grep -q "DWCP v3 codec initialized"; then
        log_success "Codec initialized successfully"
        return 0
    else
        log_error "Codec initialization check failed"
        return 1
    fi
}

test_metrics_collection() {
    log "Testing metrics collection..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 -o jsonpath='{.items[0].metadata.name}')

    local metrics=$(kubectl exec -n "$NAMESPACE" "$pod_name" -- curl -s http://localhost:9090/metrics | grep "dwcp_" | wc -l)

    if [ "$metrics" -gt 0 ]; then
        log_success "Metrics collection working ($metrics metrics found)"
        return 0
    else
        log_error "No DWCP metrics found"
        return 1
    fi
}

test_redis_connection() {
    log "Testing Redis connection..."

    local pod_name=$(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 -o jsonpath='{.items[0].metadata.name}')

    if kubectl exec -n "$NAMESPACE" "$pod_name" -- redis-cli -h dwcp-v3-redis ping 2>/dev/null | grep -q "PONG"; then
        log_success "Redis connection OK"
        return 0
    else
        log_error "Redis connection failed"
        return 1
    fi
}

main() {
    log "Running DWCP v3 smoke tests for $ENVIRONMENT"

    local failed=0

    test_basic_connectivity || ((failed++))
    test_codec_initialization || ((failed++))
    test_metrics_collection || ((failed++))
    test_redis_connection || ((failed++))

    echo ""
    if [ "$failed" -eq 0 ]; then
        log_success "All smoke tests passed!"
        exit 0
    else
        log_error "$failed smoke test(s) failed"
        exit 1
    fi
}

main
