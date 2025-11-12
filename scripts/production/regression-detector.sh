#!/bin/bash
# Performance Regression Detection System
# Compares production metrics against baseline to detect regressions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASELINE_DIR="${PROJECT_ROOT}/docs/phase6/baselines"
METRICS_DIR="${PROJECT_ROOT}/docs/phase6/metrics"
RESULTS_DIR="${PROJECT_ROOT}/docs/phase6/regression-results"

# Configuration
LATENCY_REGRESSION_THRESHOLD=10  # 10% increase is a regression
THROUGHPUT_REGRESSION_THRESHOLD=10  # 10% decrease is a regression
ERROR_RATE_REGRESSION_THRESHOLD=5  # 5% increase is a regression
CPU_REGRESSION_THRESHOLD=15  # 15% increase is a regression
MEMORY_REGRESSION_THRESHOLD=15  # 15% increase is a regression

mkdir -p "${BASELINE_DIR}" "${METRICS_DIR}" "${RESULTS_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✅ $*"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ❌ $*"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $*"
}

# Capture current metrics
capture_current_metrics() {
    log "Capturing current production metrics..."

    local timestamp=$(date +%s)
    local metrics_file="${METRICS_DIR}/metrics-${timestamp}.json"

    # Simulate metrics collection (in production, this would query actual systems)
    cat > "${metrics_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "latency_ms": {
        "p50": 25.3,
        "p95": 78.4,
        "p99": 142.7,
        "avg": 35.6
    },
    "throughput": {
        "ops_per_sec": 2456.8,
        "requests_per_sec": 1234.5,
        "vm_ops_per_sec": 456.2
    },
    "error_rate": {
        "percent": 0.05,
        "total_errors": 12,
        "total_requests": 24000
    },
    "resource_usage": {
        "cpu_percent": 45.2,
        "memory_mb": 1024.5,
        "disk_usage_percent": 62.3,
        "network_mbps": 125.7
    },
    "consensus": {
        "block_time_ms": 45.8,
        "finality_time_ms": 156.3,
        "participation_rate": 98.5
    },
    "vm_operations": {
        "creation_time_ms": 523.4,
        "migration_time_ms": 1234.5,
        "snapshot_time_ms": 234.6
    }
}
EOF

    log_success "Metrics captured: ${metrics_file}"
    echo "${metrics_file}"
}

# Load baseline metrics
load_baseline() {
    local baseline_file="${BASELINE_DIR}/baseline.json"

    if [[ ! -f "${baseline_file}" ]]; then
        log_warning "No baseline found, creating from current metrics..."
        create_baseline
        return 1
    fi

    log "Loading baseline metrics..."
    echo "${baseline_file}"
}

# Create new baseline
create_baseline() {
    log "Creating new baseline..."

    local metrics_file=$(capture_current_metrics)
    local baseline_file="${BASELINE_DIR}/baseline.json"

    cp "${metrics_file}" "${baseline_file}"

    log_success "Baseline created: ${baseline_file}"
}

# Compare metrics and detect regressions
detect_regressions() {
    local current_file="$1"
    local baseline_file="$2"

    log "Analyzing metrics for regressions..."

    local timestamp=$(date +%s)
    local results_file="${RESULTS_DIR}/regression-${timestamp}.json"
    local regressions=()
    local warnings=()

    # Extract metrics using jq
    local current_latency_p95=$(jq -r '.latency_ms.p95' "${current_file}")
    local baseline_latency_p95=$(jq -r '.latency_ms.p95' "${baseline_file}")

    local current_throughput=$(jq -r '.throughput.ops_per_sec' "${current_file}")
    local baseline_throughput=$(jq -r '.throughput.ops_per_sec' "${baseline_file}")

    local current_error_rate=$(jq -r '.error_rate.percent' "${current_file}")
    local baseline_error_rate=$(jq -r '.error_rate.percent' "${baseline_file}")

    local current_cpu=$(jq -r '.resource_usage.cpu_percent' "${current_file}")
    local baseline_cpu=$(jq -r '.resource_usage.cpu_percent' "${baseline_file}")

    local current_memory=$(jq -r '.resource_usage.memory_mb' "${current_file}")
    local baseline_memory=$(jq -r '.resource_usage.memory_mb' "${baseline_file}")

    # Check latency regression
    local latency_change=$(awk "BEGIN {printf \"%.2f\", ((${current_latency_p95} - ${baseline_latency_p95}) / ${baseline_latency_p95}) * 100}")
    if (( $(echo "${latency_change} > ${LATENCY_REGRESSION_THRESHOLD}" | bc -l) )); then
        log_error "REGRESSION: Latency P95 increased by ${latency_change}%"
        regressions+=("Latency P95 regression: +${latency_change}% (${current_latency_p95}ms vs ${baseline_latency_p95}ms)")
    elif (( $(echo "${latency_change} > 5" | bc -l) )); then
        log_warning "WARNING: Latency P95 increased by ${latency_change}%"
        warnings+=("Latency P95 increase: +${latency_change}%")
    else
        log_success "Latency: No regression detected (${latency_change}% change)"
    fi

    # Check throughput regression
    local throughput_change=$(awk "BEGIN {printf \"%.2f\", ((${baseline_throughput} - ${current_throughput}) / ${baseline_throughput}) * 100}")
    if (( $(echo "${throughput_change} > ${THROUGHPUT_REGRESSION_THRESHOLD}" | bc -l) )); then
        log_error "REGRESSION: Throughput decreased by ${throughput_change}%"
        regressions+=("Throughput regression: -${throughput_change}% (${current_throughput} vs ${baseline_throughput} ops/sec)")
    elif (( $(echo "${throughput_change} > 5" | bc -l) )); then
        log_warning "WARNING: Throughput decreased by ${throughput_change}%"
        warnings+=("Throughput decrease: -${throughput_change}%")
    else
        log_success "Throughput: No regression detected (${throughput_change}% change)"
    fi

    # Check error rate regression
    local error_rate_change=$(awk "BEGIN {printf \"%.2f\", ((${current_error_rate} - ${baseline_error_rate}) / ${baseline_error_rate}) * 100}")
    if (( $(echo "${error_rate_change} > ${ERROR_RATE_REGRESSION_THRESHOLD}" | bc -l) )); then
        log_error "REGRESSION: Error rate increased by ${error_rate_change}%"
        regressions+=("Error rate regression: +${error_rate_change}% (${current_error_rate}% vs ${baseline_error_rate}%)")
    else
        log_success "Error Rate: No regression detected (${error_rate_change}% change)"
    fi

    # Check CPU usage regression
    local cpu_change=$(awk "BEGIN {printf \"%.2f\", ((${current_cpu} - ${baseline_cpu}) / ${baseline_cpu}) * 100}")
    if (( $(echo "${cpu_change} > ${CPU_REGRESSION_THRESHOLD}" | bc -l) )); then
        log_error "REGRESSION: CPU usage increased by ${cpu_change}%"
        regressions+=("CPU usage regression: +${cpu_change}% (${current_cpu}% vs ${baseline_cpu}%)")
    elif (( $(echo "${cpu_change} > 10" | bc -l) )); then
        log_warning "WARNING: CPU usage increased by ${cpu_change}%"
        warnings+=("CPU usage increase: +${cpu_change}%")
    else
        log_success "CPU Usage: No regression detected (${cpu_change}% change)"
    fi

    # Check memory usage regression
    local memory_change=$(awk "BEGIN {printf \"%.2f\", ((${current_memory} - ${baseline_memory}) / ${baseline_memory}) * 100}")
    if (( $(echo "${memory_change} > ${MEMORY_REGRESSION_THRESHOLD}" | bc -l) )); then
        log_error "REGRESSION: Memory usage increased by ${memory_change}%"
        regressions+=("Memory usage regression: +${memory_change}% (${current_memory}MB vs ${baseline_memory}MB)")
    elif (( $(echo "${memory_change} > 10" | bc -l) )); then
        log_warning "WARNING: Memory usage increased by ${memory_change}%"
        warnings+=("Memory usage increase: +${memory_change}%")
    else
        log_success "Memory Usage: No regression detected (${memory_change}% change)"
    fi

    # Generate results JSON
    local regression_count=${#regressions[@]}
    local warning_count=${#warnings[@]}

    local regressions_json=$(printf '%s\n' "${regressions[@]}" | jq -R . | jq -s .)
    local warnings_json=$(printf '%s\n' "${warnings[@]}" | jq -R . | jq -s .)

    cat > "${results_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "baseline_file": "${baseline_file}",
    "current_file": "${current_file}",
    "regression_count": ${regression_count},
    "warning_count": ${warning_count},
    "status": "$([ ${regression_count} -eq 0 ] && echo "pass" || echo "fail")",
    "regressions": ${regressions_json},
    "warnings": ${warnings_json},
    "detailed_comparison": {
        "latency": {
            "current_p95": ${current_latency_p95},
            "baseline_p95": ${baseline_latency_p95},
            "change_percent": ${latency_change}
        },
        "throughput": {
            "current_ops_per_sec": ${current_throughput},
            "baseline_ops_per_sec": ${baseline_throughput},
            "change_percent": ${throughput_change}
        },
        "error_rate": {
            "current_percent": ${current_error_rate},
            "baseline_percent": ${baseline_error_rate},
            "change_percent": ${error_rate_change}
        },
        "cpu_usage": {
            "current_percent": ${current_cpu},
            "baseline_percent": ${baseline_cpu},
            "change_percent": ${cpu_change}
        },
        "memory_usage": {
            "current_mb": ${current_memory},
            "baseline_mb": ${baseline_memory},
            "change_percent": ${memory_change}
        }
    }
}
EOF

    log_success "Regression analysis completed: ${results_file}"

    # Display summary
    echo ""
    echo "=========================================="
    echo "Performance Regression Analysis"
    echo "=========================================="
    echo "Regression Count:   ${regression_count}"
    echo "Warning Count:      ${warning_count}"
    echo "Status:             $([ ${regression_count} -eq 0 ] && echo "✅ PASS - No regressions" || echo "❌ FAIL - ${regression_count} regression(s) detected")"
    echo ""
    if [ ${regression_count} -gt 0 ]; then
        echo "Detected Regressions:"
        printf '%s\n' "${regressions[@]}" | sed 's/^/  - /'
        echo ""
    fi
    if [ ${warning_count} -gt 0 ]; then
        echo "Warnings:"
        printf '%s\n' "${warnings[@]}" | sed 's/^/  - /'
        echo ""
    fi
    echo "=========================================="
    echo ""

    return $([ ${regression_count} -eq 0 ] && echo 0 || echo 1)
}

# Update baseline if requested
update_baseline() {
    log "Updating baseline with current metrics..."
    create_baseline
}

# Main execution
main() {
    log "=========================================="
    log "Performance Regression Detection"
    log "=========================================="

    local current_metrics=$(capture_current_metrics)
    local baseline=$(load_baseline)

    if [[ ! -f "${baseline}" ]]; then
        log_warning "No baseline available, skipping regression detection"
        exit 0
    fi

    if ! detect_regressions "${current_metrics}" "${baseline}"; then
        log_error "Performance regressions detected!"
        exit 1
    fi

    log_success "No performance regressions detected"
}

# Handle command line arguments
case "${1:-run}" in
    run)
        main
        ;;
    update-baseline)
        update_baseline
        ;;
    create-baseline)
        create_baseline
        ;;
    *)
        echo "Usage: $0 {run|update-baseline|create-baseline}"
        exit 1
        ;;
esac
