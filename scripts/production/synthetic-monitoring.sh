#!/bin/bash
# Synthetic Transaction Monitoring
# Simulates real workloads and measures end-to-end performance

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/docs/phase6/synthetic-results"
LOG_DIR="${PROJECT_ROOT}/logs/synthetic"

# Configuration
DWCP_ENDPOINT="${DWCP_ENDPOINT:-http://localhost:8080}"
TEST_INTERVAL=300 # 5 minutes
LATENCY_THRESHOLD_MS=100
SUCCESS_RATE_THRESHOLD=99.0
ENABLE_ALERTS=true

# Create directories
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOG_DIR}/synthetic.log"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ✅ $*" | tee -a "${LOG_DIR}/synthetic.log"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ❌ $*" | tee -a "${LOG_DIR}/synthetic.log"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ⚠️  $*" | tee -a "${LOG_DIR}/synthetic.log"
}

# Synthetic test: VM creation
synthetic_vm_creation() {
    local start_time=$(date +%s%3N)
    local vm_id="synthetic-vm-$(date +%s)"

    log "Testing VM creation..."

    local response=$(curl -s -w "\n%{http_code}\n%{time_total}" -X POST \
        "${DWCP_ENDPOINT}/api/v3/vm/create" \
        -H "Content-Type: application/json" \
        -d "{
            \"vm_id\": \"${vm_id}\",
            \"cpu\": 2,
            \"memory\": 4096,
            \"storage\": 20480
        }" 2>/dev/null || echo -e "\n500\n0")

    local body=$(echo "${response}" | head -n -2)
    local http_code=$(echo "${response}" | tail -n 2 | head -n 1)
    local time_total=$(echo "${response}" | tail -n 1)

    local end_time=$(date +%s%3N)
    local latency=$((end_time - start_time))

    if [[ "${http_code}" == "200" || "${http_code}" == "201" ]]; then
        log_success "VM creation: ${latency}ms (HTTP ${http_code})"
        echo "${vm_id}|${latency}|success"
    else
        log_error "VM creation failed: HTTP ${http_code}"
        echo "${vm_id}|${latency}|failure"
    fi
}

# Synthetic test: Consensus operation
synthetic_consensus_operation() {
    local start_time=$(date +%s%3N)

    log "Testing consensus operation..."

    local response=$(curl -s -w "\n%{http_code}\n%{time_total}" -X POST \
        "${DWCP_ENDPOINT}/api/v3/consensus/propose" \
        -H "Content-Type: application/json" \
        -d "{
            \"proposal_id\": \"synthetic-proposal-$(date +%s)\",
            \"data\": \"test-data\"
        }" 2>/dev/null || echo -e "\n500\n0")

    local http_code=$(echo "${response}" | tail -n 2 | head -n 1)
    local end_time=$(date +%s%3N)
    local latency=$((end_time - start_time))

    if [[ "${http_code}" == "200" || "${http_code}" == "201" ]]; then
        log_success "Consensus operation: ${latency}ms (HTTP ${http_code})"
        echo "${latency}|success"
    else
        log_error "Consensus operation failed: HTTP ${http_code}"
        echo "${latency}|failure"
    fi
}

# Synthetic test: Network communication
synthetic_network_communication() {
    local start_time=$(date +%s%3N)

    log "Testing network communication..."

    local response=$(curl -s -w "\n%{http_code}" -X GET \
        "${DWCP_ENDPOINT}/api/v3/network/peers" \
        -H "Accept: application/json" 2>/dev/null || echo -e "\n500")

    local http_code=$(echo "${response}" | tail -n 1)
    local end_time=$(date +%s%3N)
    local latency=$((end_time - start_time))

    if [[ "${http_code}" == "200" ]]; then
        log_success "Network communication: ${latency}ms (HTTP ${http_code})"
        echo "${latency}|success"
    else
        log_error "Network communication failed: HTTP ${http_code}"
        echo "${latency}|failure"
    fi
}

# Synthetic test: Data replication
synthetic_data_replication() {
    local start_time=$(date +%s%3N)
    local key="synthetic-key-$(date +%s)"
    local value="synthetic-value-$(openssl rand -hex 32)"

    log "Testing data replication..."

    # Write data
    local write_response=$(curl -s -w "\n%{http_code}" -X PUT \
        "${DWCP_ENDPOINT}/api/v3/data/${key}" \
        -H "Content-Type: application/json" \
        -d "{\"value\": \"${value}\"}" 2>/dev/null || echo -e "\n500")

    local write_code=$(echo "${write_response}" | tail -n 1)

    if [[ "${write_code}" != "200" && "${write_code}" != "201" ]]; then
        log_error "Data write failed: HTTP ${write_code}"
        echo "0|failure"
        return
    fi

    # Wait for replication
    sleep 2

    # Read data
    local read_response=$(curl -s -w "\n%{http_code}" -X GET \
        "${DWCP_ENDPOINT}/api/v3/data/${key}" \
        -H "Accept: application/json" 2>/dev/null || echo -e "\n500")

    local read_code=$(echo "${read_response}" | tail -n 1)
    local read_body=$(echo "${read_response}" | head -n -1)

    local end_time=$(date +%s%3N)
    local latency=$((end_time - start_time))

    if [[ "${read_code}" == "200" ]] && echo "${read_body}" | grep -q "${value}"; then
        log_success "Data replication: ${latency}ms (HTTP ${read_code})"
        echo "${latency}|success"
    else
        log_error "Data replication failed: HTTP ${read_code}"
        echo "${latency}|failure"
    fi
}

# Synthetic test: End-to-end workflow
synthetic_end_to_end_workflow() {
    local start_time=$(date +%s%3N)

    log "Testing end-to-end workflow..."

    # Step 1: Create VM
    local vm_result=$(synthetic_vm_creation)
    local vm_status=$(echo "${vm_result}" | cut -d'|' -f3)

    if [[ "${vm_status}" != "success" ]]; then
        log_error "End-to-end workflow failed at VM creation"
        echo "0|failure"
        return
    fi

    # Step 2: Consensus operation
    local consensus_result=$(synthetic_consensus_operation)
    local consensus_status=$(echo "${consensus_result}" | cut -d'|' -f2)

    if [[ "${consensus_status}" != "success" ]]; then
        log_error "End-to-end workflow failed at consensus"
        echo "0|failure"
        return
    fi

    # Step 3: Network communication
    local network_result=$(synthetic_network_communication)
    local network_status=$(echo "${network_result}" | cut -d'|' -f2)

    if [[ "${network_status}" != "success" ]]; then
        log_error "End-to-end workflow failed at network communication"
        echo "0|failure"
        return
    fi

    local end_time=$(date +%s%3N)
    local latency=$((end_time - start_time))

    log_success "End-to-end workflow: ${latency}ms"
    echo "${latency}|success"
}

# Run all synthetic tests
run_synthetic_tests() {
    log "=========================================="
    log "Running Synthetic Transaction Tests"
    log "=========================================="

    local timestamp=$(date +%s)
    local results_file="${RESULTS_DIR}/synthetic-${timestamp}.json"

    local total_tests=0
    local successful_tests=0
    local total_latency=0

    # Test 1: VM Creation (3 iterations)
    for i in {1..3}; do
        local result=$(synthetic_vm_creation)
        local latency=$(echo "${result}" | cut -d'|' -f2)
        local status=$(echo "${result}" | cut -d'|' -f3)

        total_tests=$((total_tests + 1))
        total_latency=$((total_latency + latency))
        [[ "${status}" == "success" ]] && successful_tests=$((successful_tests + 1))

        sleep 1
    done

    # Test 2: Consensus Operation (3 iterations)
    for i in {1..3}; do
        local result=$(synthetic_consensus_operation)
        local latency=$(echo "${result}" | cut -d'|' -f1)
        local status=$(echo "${result}" | cut -d'|' -f2)

        total_tests=$((total_tests + 1))
        total_latency=$((total_latency + latency))
        [[ "${status}" == "success" ]] && successful_tests=$((successful_tests + 1))

        sleep 1
    done

    # Test 3: Network Communication (3 iterations)
    for i in {1..3}; do
        local result=$(synthetic_network_communication)
        local latency=$(echo "${result}" | cut -d'|' -f1)
        local status=$(echo "${result}" | cut -d'|' -f2)

        total_tests=$((total_tests + 1))
        total_latency=$((total_latency + latency))
        [[ "${status}" == "success" ]] && successful_tests=$((successful_tests + 1))

        sleep 1
    done

    # Test 4: Data Replication (3 iterations)
    for i in {1..3}; do
        local result=$(synthetic_data_replication)
        local latency=$(echo "${result}" | cut -d'|' -f1)
        local status=$(echo "${result}" | cut -d'|' -f2)

        total_tests=$((total_tests + 1))
        total_latency=$((total_latency + latency))
        [[ "${status}" == "success" ]] && successful_tests=$((successful_tests + 1))

        sleep 1
    done

    # Test 5: End-to-end Workflow (2 iterations)
    for i in {1..2}; do
        local result=$(synthetic_end_to_end_workflow)
        local latency=$(echo "${result}" | cut -d'|' -f1)
        local status=$(echo "${result}" | cut -d'|' -f2)

        total_tests=$((total_tests + 1))
        total_latency=$((total_latency + latency))
        [[ "${status}" == "success" ]] && successful_tests=$((successful_tests + 1))

        sleep 2
    done

    # Calculate metrics
    local success_rate=$(awk "BEGIN {printf \"%.2f\", (${successful_tests}/${total_tests})*100}")
    local avg_latency=$((total_latency / total_tests))

    # Save results
    cat > "${results_file}" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total_tests": ${total_tests},
    "successful_tests": ${successful_tests},
    "failed_tests": $((total_tests - successful_tests)),
    "success_rate": ${success_rate},
    "average_latency_ms": ${avg_latency},
    "latency_threshold_ms": ${LATENCY_THRESHOLD_MS},
    "success_rate_threshold": ${SUCCESS_RATE_THRESHOLD},
    "status": "$([ "${success_rate%.*}" -ge "${SUCCESS_RATE_THRESHOLD%.*}" ] && echo "pass" || echo "fail")"
}
EOF

    # Display summary
    echo ""
    echo "=========================================="
    echo "Synthetic Test Summary"
    echo "=========================================="
    echo "Total Tests:        ${total_tests}"
    echo "Successful:         ${successful_tests}"
    echo "Failed:             $((total_tests - successful_tests))"
    echo "Success Rate:       ${success_rate}%"
    echo "Average Latency:    ${avg_latency}ms"
    echo "Status:             $([ "${success_rate%.*}" -ge "${SUCCESS_RATE_THRESHOLD%.*}" ] && echo "✅ PASS" || echo "❌ FAIL")"
    echo "=========================================="
    echo ""

    # Check thresholds and alert
    if [ "${success_rate%.*}" -lt "${SUCCESS_RATE_THRESHOLD%.*}" ]; then
        log_error "Success rate ${success_rate}% below threshold ${SUCCESS_RATE_THRESHOLD}%"
        [[ "${ENABLE_ALERTS}" == "true" ]] && send_alert "critical" "Synthetic tests success rate below threshold"
        return 1
    fi

    if [ "${avg_latency}" -gt "${LATENCY_THRESHOLD_MS}" ]; then
        log_warning "Average latency ${avg_latency}ms exceeds threshold ${LATENCY_THRESHOLD_MS}ms"
        [[ "${ENABLE_ALERTS}" == "true" ]] && send_alert "warning" "Synthetic tests latency above threshold"
    fi

    log_success "Synthetic monitoring completed successfully"
    return 0
}

send_alert() {
    local severity="$1"
    local message="$2"

    log "Sending alert: ${severity} - ${message}"
    # Alert implementation would go here
}

# Main execution
main() {
    log "Starting synthetic transaction monitoring..."

    if ! run_synthetic_tests; then
        log_error "Synthetic tests failed"
        exit 1
    fi

    log_success "Synthetic monitoring completed"
}

main "$@"
