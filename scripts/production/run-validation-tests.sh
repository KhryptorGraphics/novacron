#!/bin/bash
# Production Validation Test Runner
# Runs comprehensive validation tests every hour in production

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/docs/phase6/validation-results}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/validation}"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_RETRIES="${MAX_RETRIES:-3}"
TEST_TIMEOUT="${TEST_TIMEOUT:-300}" # 5 minutes
RETRY_DELAY="${RETRY_DELAY:-60}"
ALERT_ON_FAILURE="${ALERT_ON_FAILURE:-true}"
SAVE_DETAILED_LOGS="${SAVE_DETAILED_LOGS:-true}"
ENABLE_METRICS="${ENABLE_METRICS:-true}"

# Create directories
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_DIR}/validation.log"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✅ $*" | tee -a "${LOG_DIR}/validation.log"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ❌ $*" | tee -a "${LOG_DIR}/validation.log"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $*" | tee -a "${LOG_DIR}/validation.log"
}

# Send alert via webhook
send_alert() {
    local severity="$1"
    local message="$2"

    if [[ -z "${ALERT_WEBHOOK}" ]]; then
        log_warning "No alert webhook configured, skipping alert"
        return
    fi

    local payload=$(cat <<EOF
{
    "severity": "${severity}",
    "message": "${message}",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "production",
    "service": "dwcp-validation"
}
EOF
)

    curl -X POST "${ALERT_WEBHOOK}" \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        --max-time 10 \
        --silent \
        --show-error || log_warning "Failed to send alert"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    if ! command -v go &> /dev/null; then
        log_error "Go is not installed"
        return 1
    fi

    if [[ ! -d "${PROJECT_ROOT}/backend/core/network/dwcp/v3" ]]; then
        log_error "DWCP v3 directory not found"
        return 1
    fi

    log_success "Prerequisites check passed"
    return 0
}

# Run validation tests
run_validation_tests() {
    local attempt
    local test_result="failure"
    local test_log=""
    local results_file=""

    cd "${PROJECT_ROOT}/backend/core/network/dwcp/v3/tests"

    for attempt in $(seq 1 "${MAX_RETRIES}"); do
        log "Running production validation tests (attempt ${attempt}/${MAX_RETRIES})..."

        local timestamp=$(date +%Y%m%d-%H%M%S)
        test_log="${LOG_DIR}/test-${timestamp}.log"
        results_file="${RESULTS_DIR}/validation-${timestamp}.json"

        # Run tests with timeout
        if timeout "${TEST_TIMEOUT}" go test -v \
            -run TestProductionValidationComplete \
            -timeout "${TEST_TIMEOUT}s" \
            ./... 2>&1 | tee "${test_log}"; then
            test_result="success"
            log_success "Validation tests passed"
            break
        fi

        log_error "Validation tests failed"
        if [[ ${attempt} -lt ${MAX_RETRIES} ]]; then
            log "Retrying in ${RETRY_DELAY} seconds..."
            sleep "${RETRY_DELAY}"
        fi
    done

    # Parse results
    parse_test_results "${test_log}" "${results_file}"

    # Send alerts if needed
    if [[ "${test_result}" == "failure" ]] && [[ "${ALERT_ON_FAILURE}" == "true" ]]; then
        send_alert "critical" "Production validation tests failed after ${MAX_RETRIES} attempts"
    fi

    return $([ "${test_result}" == "success" ] && echo 0 || echo 1)
}

# Parse test results
parse_test_results() {
    local test_log="$1"
    local results_file="$2"

    log "Parsing test results..."

    local total_tests=$(grep -c -- "=== RUN" "${test_log}" || true)
    local passed_tests=$(grep -c -- "--- PASS:" "${test_log}" || true)
    local failed_tests=$(grep -c -- "--- FAIL:" "${test_log}" || true)
    local skipped_tests=$(grep -c -- "--- SKIP:" "${test_log}" || true)

    local pass_rate=0
    if [[ ${total_tests} -gt 0 ]]; then
        pass_rate=$(awk "BEGIN {printf \"%.2f\", (${passed_tests}/${total_tests})*100}")
    fi

    # Create JSON results
    cat > "${results_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total_tests": ${total_tests},
    "passed_tests": ${passed_tests},
    "failed_tests": ${failed_tests},
    "skipped_tests": ${skipped_tests},
    "pass_rate": ${pass_rate},
    "test_log": "${test_log}",
    "environment": "production"
}
EOF

    log_success "Test results: ${passed_tests}/${total_tests} passed (${pass_rate}%)"

    # Display summary
    echo ""
    echo "=========================================="
    echo "Production Validation Test Summary"
    echo "=========================================="
    echo "Total Tests:    ${total_tests}"
    echo "Passed:         ${passed_tests}"
    echo "Failed:         ${failed_tests}"
    echo "Skipped:        ${skipped_tests}"
    echo "Pass Rate:      ${pass_rate}%"
    echo "Results File:   ${results_file}"
    echo "=========================================="
    echo ""
}

# Collect system metrics
collect_metrics() {
    if [[ "${ENABLE_METRICS}" != "true" ]]; then
        return
    fi

    log "Collecting system metrics..."

    local metrics_file="${RESULTS_DIR}/metrics-$(date +%Y%m%d-%H%M%S).json"
    local cpu_usage
    local memory_usage
    local disk_usage
    local load_average
    local active_connections

    cpu_usage=$(top -bn1 | awk -F'id,' '/Cpu\(s\)/ { split($1, parts, ","); sub(/^.* /, "", parts[length(parts)]); printf "%.2f", 100 - parts[length(parts)] }' || echo "0")
    memory_usage=$(free -m | awk 'NR==2{printf "%.2f", $3*100/$2 }' || echo "0")
    disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//' || echo "0")
    load_average=$(uptime | awk -F'load average:' '{print $2}' | sed 's/^ *//' || echo "0")
    if command -v ss >/dev/null 2>&1; then
        active_connections=$(ss -tan state established | awk 'NR>1 {count++} END {print count+0}')
    elif command -v netstat >/dev/null 2>&1; then
        active_connections=$(netstat -an | grep -c -- "ESTABLISHED" || echo "0")
    else
        active_connections="0"
    fi

    cat > "${metrics_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "cpu_usage": "${cpu_usage}",
    "memory_usage": "${memory_usage}",
    "disk_usage": "${disk_usage}",
    "load_average": "${load_average}",
    "active_connections": "${active_connections}"
}
EOF

    log_success "Metrics collected: ${metrics_file}"
}

# Generate validation report
generate_report() {
    log "Generating validation report..."

    local report_file="${RESULTS_DIR}/validation-report-$(date +%Y%m%d-%H%M%S).md"
    local test_summary="No test results available"
    local metrics_summary="No metrics available"
    local recent_results
    local recommendations

    if compgen -G "${RESULTS_DIR}/validation-*.json" >/dev/null; then
        test_summary=$(jq -s -r '
            "- Total Test Runs: \(length)",
            "- Average Pass Rate: \(map(.pass_rate) | add / length)%",
            "- Total Tests Executed: \(map(.total_tests) | add)",
            "- Total Failures: \(map(.failed_tests) | add)"
        ' "${RESULTS_DIR}"/validation-*.json 2>/dev/null || echo "No test results available")
    fi

    local latest_metrics
    latest_metrics=$(ls -t "${RESULTS_DIR}"/metrics-*.json 2>/dev/null | head -1 || true)
    if [[ -n "${latest_metrics}" ]]; then
        metrics_summary=$(jq -r '
            "- CPU Usage: \(.cpu_usage)%",
            "- Memory Usage: \(.memory_usage)%",
            "- Disk Usage: \(.disk_usage)%",
            "- Active Connections: \(.active_connections)"
        ' "${latest_metrics}" 2>/dev/null || echo "No metrics available")
    fi

    recent_results=$(tail -20 "${LOG_DIR}/validation.log" 2>/dev/null || echo "No recent logs")
    if grep -q "FAIL" "${LOG_DIR}/validation.log" 2>/dev/null; then
        recommendations="- ⚠️ Investigate failed tests immediately
- Review error logs for root cause analysis
- Consider rolling back recent changes if failures persist"
    else
        recommendations="- ✅ All validation tests passing
- Continue monitoring system metrics
- Maintain current deployment configuration"
    fi

    cat > "${report_file}" <<EOF
# Production Validation Report

**Generated:** $(date +'%Y-%m-%d %H:%M:%S')
**Environment:** Production
**Phase:** 6 - Continuous Validation

## Test Execution Summary

${test_summary}

## System Metrics

${metrics_summary}

## Recent Test Results

\`\`\`
${recent_results}
\`\`\`

## Recommendations

${recommendations}

---
*Report generated by DWCP v3 Production Validation System*
EOF

    log_success "Report generated: ${report_file}"
}

# Cleanup old results
cleanup_old_results() {
    log "Cleaning up old validation results (older than 7 days)..."

    find "${RESULTS_DIR}" -type f -name "*.json" -mtime +7 -delete
    find "${RESULTS_DIR}" -type f -name "*.md" -mtime +7 -delete
    find "${LOG_DIR}" -type f -name "*.log" -mtime +7 -delete

    log_success "Cleanup completed"
}

# Main execution
main() {
    log "=========================================="
    log "Production Validation Test Runner"
    log "=========================================="

    if ! check_prerequisites; then
        log_error "Prerequisites check failed"
        exit 1
    fi

    if ! run_validation_tests; then
        log_error "Validation tests failed"
        exit 1
    fi

    collect_metrics
    generate_report
    cleanup_old_results

    log_success "Production validation completed successfully"
}

# Run main function
main "$@"
