#!/bin/bash
#
# DWCP v3 Metrics Verification
# Verify application metrics are within acceptable ranges
#

set -euo pipefail

NAMESPACE="${1:-production}"
PROMETHEUS_POD=""

# Thresholds
ERROR_RATE_THRESHOLD=5
LATENCY_THRESHOLD=1.0
THROUGHPUT_THRESHOLD=1048576  # 1MB/s

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
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

log_warning() {
    echo -e "${YELLOW}⚠️  $*${NC}"
}

get_prometheus_pod() {
    PROMETHEUS_POD=$(kubectl get pods -n "$NAMESPACE" -l app=prometheus -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [ -z "$PROMETHEUS_POD" ]; then
        log_warning "Prometheus not found, skipping metrics verification"
        return 1
    fi
    return 0
}

query_prometheus() {
    local query="$1"

    kubectl exec -n "$NAMESPACE" "$PROMETHEUS_POD" -- \
        wget -qO- "http://localhost:9090/api/v1/query?query=${query}" 2>/dev/null | \
        grep -o '"value":\[[^]]*\]' | \
        grep -o '[0-9.]*' | \
        tail -1 || echo "0"
}

check_error_rate() {
    log "Checking error rate..."

    local error_rate=$(query_prometheus "rate(dwcp_errors_total[5m])")
    local error_pct=$(echo "$error_rate * 100" | bc -l)

    log "Current error rate: ${error_pct}%"

    if (( $(echo "$error_pct > $ERROR_RATE_THRESHOLD" | bc -l) )); then
        log_error "Error rate ${error_pct}% exceeds threshold ${ERROR_RATE_THRESHOLD}%"
        return 1
    else
        log_success "Error rate within acceptable range"
        return 0
    fi
}

check_latency() {
    log "Checking latency..."

    local latency=$(query_prometheus "histogram_quantile(0.95, dwcp_latency_seconds_bucket)")

    log "Current P95 latency: ${latency}s"

    if (( $(echo "$latency > $LATENCY_THRESHOLD" | bc -l) )); then
        log_warning "Latency ${latency}s above threshold ${LATENCY_THRESHOLD}s"
        return 1
    else
        log_success "Latency within acceptable range"
        return 0
    fi
}

check_throughput() {
    log "Checking throughput..."

    local throughput=$(query_prometheus "rate(dwcp_bytes_transferred_total[5m])")
    local throughput_mb=$(echo "scale=2; $throughput / 1048576" | bc)

    log "Current throughput: ${throughput_mb} MB/s"

    if (( $(echo "$throughput < $THROUGHPUT_THRESHOLD" | bc -l) )); then
        log_warning "Throughput ${throughput_mb}MB/s below threshold 1MB/s"
        return 1
    else
        log_success "Throughput within acceptable range"
        return 0
    fi
}

check_active_connections() {
    log "Checking active connections..."

    local connections=$(query_prometheus "dwcp_active_connections")

    log "Active connections: $connections"

    if [ "$(echo "$connections" | cut -d. -f1)" -gt 0 ]; then
        log_success "Application has active connections"
        return 0
    else
        log_warning "No active connections"
        return 1
    fi
}

main() {
    log "Verifying DWCP v3 metrics for namespace: $NAMESPACE"

    if ! get_prometheus_pod; then
        exit 0
    fi

    local warnings=0
    local errors=0

    check_error_rate || ((errors++))
    check_latency || ((warnings++))
    check_throughput || ((warnings++))
    check_active_connections || ((warnings++))

    echo ""
    if [ "$errors" -eq 0 ] && [ "$warnings" -eq 0 ]; then
        log_success "All metrics verified successfully!"
        exit 0
    elif [ "$errors" -eq 0 ]; then
        log_warning "$warnings metric(s) have warnings"
        exit 0
    else
        log_error "$errors metric(s) failed verification"
        exit 1
    fi
}

main
