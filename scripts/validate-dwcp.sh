#!/bin/bash
# DWCP Validation Script
# Validates DWCP deployment and functionality

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
API_ENDPOINT="${API_ENDPOINT:-http://localhost:8080}"
METRICS_ENDPOINT="${METRICS_ENDPOINT:-http://localhost:9090}"
VALIDATION_LOG="/tmp/dwcp-validation-$(date +%Y%m%d-%H%M%S).log"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$VALIDATION_LOG"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" | tee -a "$VALIDATION_LOG"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*" | tee -a "$VALIDATION_LOG"
}

# Validation checks
check_api_health() {
    log "Checking API health..."

    if response=$(curl -sf "$API_ENDPOINT/health"); then
        log "✓ API health check passed"
        echo "$response" | jq '.' >> "$VALIDATION_LOG" 2>/dev/null || echo "$response" >> "$VALIDATION_LOG"
        return 0
    else
        error "API health check failed"
        return 1
    fi
}

check_dwcp_enabled() {
    log "Checking if DWCP is enabled..."

    if response=$(curl -sf "$API_ENDPOINT/api/v1/dwcp/status"); then
        enabled=$(echo "$response" | jq -r '.enabled' 2>/dev/null || echo "unknown")
        if [[ "$enabled" == "true" ]]; then
            log "✓ DWCP is enabled"
            return 0
        else
            warn "DWCP is disabled"
            return 1
        fi
    else
        error "Failed to check DWCP status"
        return 1
    fi
}

check_amst_streams() {
    log "Checking AMST streams..."

    if metrics=$(curl -sf "$METRICS_ENDPOINT/metrics" | grep "dwcp_amst_active_streams"); then
        stream_count=$(echo "$metrics" | awk '{print $2}')
        log "✓ AMST streams active: $stream_count"

        if (( $(echo "$stream_count > 0" | bc -l) )); then
            return 0
        else
            error "No active AMST streams"
            return 1
        fi
    else
        error "Failed to get AMST stream metrics"
        return 1
    fi
}

check_hde_compression() {
    log "Checking HDE compression..."

    if metrics=$(curl -sf "$METRICS_ENDPOINT/metrics" | grep "dwcp_hde_compression_ratio"); then
        ratio=$(echo "$metrics" | awk '{print $2}')
        log "✓ HDE compression ratio: ${ratio}x"

        if (( $(echo "$ratio > 1.0" | bc -l) )); then
            return 0
        else
            warn "Compression ratio below 1.0"
            return 1
        fi
    else
        error "Failed to get HDE compression metrics"
        return 1
    fi
}

check_baseline_sync() {
    log "Checking baseline synchronization..."

    if metrics=$(curl -sf "$METRICS_ENDPOINT/metrics" | grep "dwcp_hde_baselines_synced"); then
        baselines=$(echo "$metrics" | awk '{print $2}')
        log "✓ Synchronized baselines: $baselines"

        if (( $(echo "$baselines > 0" | bc -l) )); then
            return 0
        else
            warn "No synchronized baselines"
            return 1
        fi
    else
        error "Failed to get baseline sync metrics"
        return 1
    fi
}

check_error_rate() {
    log "Checking error rate..."

    if error_metrics=$(curl -sf "$METRICS_ENDPOINT/metrics" | grep "dwcp_errors_total"); then
        errors=$(echo "$error_metrics" | awk '{print $2}')

        if total_metrics=$(curl -sf "$METRICS_ENDPOINT/metrics" | grep "dwcp_requests_total"); then
            total=$(echo "$total_metrics" | awk '{print $2}')

            if (( $(echo "$total > 0" | bc -l) )); then
                error_rate=$(echo "scale=2; ($errors / $total) * 100" | bc)
                log "✓ Error rate: ${error_rate}%"

                if (( $(echo "$error_rate < 5.0" | bc -l) )); then
                    return 0
                else
                    error "Error rate too high: ${error_rate}%"
                    return 1
                fi
            else
                log "✓ No requests yet, error rate: 0%"
                return 0
            fi
        fi
    else
        warn "Error metrics not available yet"
        return 0
    fi
}

check_prometheus_metrics() {
    log "Checking Prometheus metrics..."

    expected_metrics=(
        "dwcp_amst_active_streams"
        "dwcp_amst_bytes_sent"
        "dwcp_amst_bytes_received"
        "dwcp_hde_compression_ratio"
        "dwcp_hde_baselines_synced"
        "dwcp_requests_total"
        "dwcp_errors_total"
    )

    missing_metrics=()

    for metric in "${expected_metrics[@]}"; do
        if curl -sf "$METRICS_ENDPOINT/metrics" | grep -q "$metric"; then
            log "  ✓ $metric"
        else
            warn "  ✗ $metric missing"
            missing_metrics+=("$metric")
        fi
    done

    if [[ ${#missing_metrics[@]} -eq 0 ]]; then
        log "✓ All expected metrics present"
        return 0
    else
        error "${#missing_metrics[@]} metrics missing"
        return 1
    fi
}

performance_test() {
    log "Running basic performance test..."

    if ! command -v ab &> /dev/null; then
        warn "Apache Bench (ab) not installed, skipping performance test"
        return 0
    fi

    # Send 1000 requests with 10 concurrent connections
    if ab -n 1000 -c 10 "$API_ENDPOINT/health" > /tmp/dwcp-perf-test.txt 2>&1; then
        requests_per_sec=$(grep "Requests per second" /tmp/dwcp-perf-test.txt | awk '{print $4}')
        log "✓ Performance test complete: ${requests_per_sec} req/sec"
        return 0
    else
        error "Performance test failed"
        return 1
    fi
}

generate_report() {
    log "Generating validation report..."

    cat > /tmp/dwcp-validation-report.txt <<EOF
DWCP Phase 1 Validation Report
Generated: $(date)

Environment: $API_ENDPOINT
Metrics: $METRICS_ENDPOINT

Validation Results:
==================

EOF

    # Run all checks and collect results
    checks=(
        "API Health:check_api_health"
        "DWCP Enabled:check_dwcp_enabled"
        "AMST Streams:check_amst_streams"
        "HDE Compression:check_hde_compression"
        "Baseline Sync:check_baseline_sync"
        "Error Rate:check_error_rate"
        "Prometheus Metrics:check_prometheus_metrics"
    )

    passed=0
    failed=0

    for check in "${checks[@]}"; do
        name="${check%%:*}"
        func="${check##*:}"

        if $func > /dev/null 2>&1; then
            echo "✓ $name: PASSED" >> /tmp/dwcp-validation-report.txt
            ((passed++))
        else
            echo "✗ $name: FAILED" >> /tmp/dwcp-validation-report.txt
            ((failed++))
        fi
    done

    cat >> /tmp/dwcp-validation-report.txt <<EOF

Summary:
========
Passed: $passed
Failed: $failed
Total: $((passed + failed))

Success Rate: $(echo "scale=2; ($passed / ($passed + $failed)) * 100" | bc)%

Detailed Log: $VALIDATION_LOG
EOF

    cat /tmp/dwcp-validation-report.txt
    log "Full report saved to: /tmp/dwcp-validation-report.txt"
}

# Main validation
main() {
    log "=== DWCP Phase 1 Validation Starting ==="

    failed_checks=0

    # Run all validation checks
    check_api_health || ((failed_checks++))
    check_dwcp_enabled || ((failed_checks++))
    check_amst_streams || ((failed_checks++))
    check_hde_compression || ((failed_checks++))
    check_baseline_sync || ((failed_checks++))
    check_error_rate || ((failed_checks++))
    check_prometheus_metrics || ((failed_checks++))
    performance_test || ((failed_checks++))

    # Generate report
    generate_report

    log ""
    log "=== Validation Complete ==="
    log "Failed checks: $failed_checks"
    log "Validation log: $VALIDATION_LOG"

    if [[ $failed_checks -eq 0 ]]; then
        log "✓ All validation checks passed!"
        exit 0
    else
        error "$failed_checks validation check(s) failed"
        exit 1
    fi
}

main "$@"
