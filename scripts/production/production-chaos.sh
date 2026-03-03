#!/bin/bash

################################################################################
# DWCP v3 Production Chaos Engineering
# Continuous chaos testing in production environment
################################################################################

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
CHAOS_LOG="/var/log/novacron/chaos-production.log"
CHAOS_RESULTS="/var/lib/novacron/chaos-results.json"
DWCP_API="${DWCP_API:-http://localhost:8080/api/v3}"
PRODUCTION_SAFE_MODE="${PRODUCTION_SAFE_MODE:-true}"  # Extra safety for production

# Chaos scenarios
declare -A CHAOS_SCENARIOS=(
    ["latency_injection"]="Inject network latency"
    ["packet_loss"]="Introduce packet loss"
    ["cpu_stress"]="CPU stress test"
    ["memory_stress"]="Memory pressure test"
    ["disk_io_stress"]="Disk I/O saturation"
    ["connection_limit"]="Connection pool exhaustion"
    ["cache_invalidation"]="Cache clearing"
    ["gc_pressure"]="Garbage collection pressure"
)

# Safety limits for production
MAX_LATENCY_MS=50
MAX_PACKET_LOSS_PERCENT=1
MAX_CPU_STRESS_PERCENT=15
MAX_MEMORY_STRESS_PERCENT=10
MAX_CHAOS_DURATION=300  # 5 minutes max

################################################################################
# Logging
################################################################################

log_chaos() {
    echo -e "${MAGENTA}[CHAOS]${NC} $1" | tee -a "$CHAOS_LOG"
}

log_result() {
    echo -e "${BLUE}[RESULT]${NC} $1" | tee -a "$CHAOS_LOG"
}

################################################################################
# Safety Checks
################################################################################

verify_production_safety() {
    log_chaos "Verifying production safety prerequisites..."

    # Check if rollback system is running
    if ! pgrep -f "auto-rollback.sh" > /dev/null; then
        log_chaos "ERROR: Rollback system not running - chaos tests aborted"
        return 1
    fi

    # Check if health checker is running
    if ! pgrep -f "health-checker.sh" > /dev/null; then
        log_chaos "ERROR: Health checker not running - chaos tests aborted"
        return 1
    fi

    # Check if incident response is running
    if ! pgrep -f "incident-response.sh" > /dev/null; then
        log_chaos "ERROR: Incident response not running - chaos tests aborted"
        return 1
    fi

    # Check current system health
    local health_score=$(curl -sf "${DWCP_API}/health/score" | jq -r '.score // 0')
    if [[ $health_score -lt 95 ]]; then
        log_chaos "ERROR: System health too low ($health_score/100) - chaos tests aborted"
        return 1
    fi

    # Check if there are ongoing incidents
    local active_incidents=$(curl -sf "${DWCP_API}/incidents/active" | jq -r '.count // 0')
    if [[ $active_incidents -gt 0 ]]; then
        log_chaos "ERROR: Active incidents detected ($active_incidents) - chaos tests aborted"
        return 1
    fi

    # Check current load
    local request_rate=$(curl -sf "${DWCP_API}/metrics/requests" | jq -r '.rate_1m // 0')
    local peak_rate=$(curl -sf "${DWCP_API}/metrics/requests" | jq -r '.peak_rate_1h // 1000')
    local load_ratio=$(echo "scale=2; $request_rate / $peak_rate" | bc)

    if [[ $(echo "$load_ratio > 0.8" | bc -l) -eq 1 ]]; then
        log_chaos "WARNING: High traffic detected (${load_ratio}x peak) - chaos tests may be postponed"
        if [[ "$PRODUCTION_SAFE_MODE" == "true" ]]; then
            return 1
        fi
    fi

    log_chaos "Production safety verification: PASSED"
    return 0
}

check_blast_radius() {
    local scenario=$1
    local affected_nodes=$2

    # In production, limit blast radius to single node
    if [[ $affected_nodes -gt 1 ]] && [[ "$PRODUCTION_SAFE_MODE" == "true" ]]; then
        log_chaos "ERROR: Blast radius too large ($affected_nodes nodes) for production"
        return 1
    fi

    # Check cluster can tolerate the failure
    local total_nodes=$(curl -sf "${DWCP_API}/cluster/size" | jq -r '.total // 7')
    local quorum_size=$(( (total_nodes / 2) + 1 ))
    local remaining_nodes=$((total_nodes - affected_nodes))

    if [[ $remaining_nodes -lt $quorum_size ]]; then
        log_chaos "ERROR: Would break quorum ($remaining_nodes < $quorum_size)"
        return 1
    fi

    log_chaos "Blast radius check: PASSED (affecting $affected_nodes/$total_nodes nodes)"
    return 0
}

################################################################################
# Chaos Scenarios
################################################################################

scenario_latency_injection() {
    local target_node=${1:-"node-6"}  # Target least critical node
    local latency_ms=${2:-$MAX_LATENCY_MS}
    local duration=${3:-60}

    log_chaos "Injecting ${latency_ms}ms latency on $target_node for ${duration}s"

    # Apply traffic control rules
    local node_ip=$(curl -sf "${DWCP_API}/cluster/nodes/${target_node}/ip" | jq -r '.ip')

    ssh "$node_ip" "sudo tc qdisc add dev eth0 root netem delay ${latency_ms}ms" 2>/dev/null || true

    # Monitor impact
    local start_time=$(date +%s)
    local max_observed_latency=0

    while [[ $(($(date +%s) - start_time)) -lt $duration ]]; do
        local current_latency=$(curl -sf -w "%{time_total}" -o /dev/null "${DWCP_API}/ping" | awk '{print int($1 * 1000)}')
        if [[ $current_latency -gt $max_observed_latency ]]; then
            max_observed_latency=$current_latency
        fi
        sleep 5
    done

    # Remove traffic control
    ssh "$node_ip" "sudo tc qdisc del dev eth0 root" 2>/dev/null || true

    log_result "Latency injection: Max observed ${max_observed_latency}ms (injected ${latency_ms}ms)"

    # Verify recovery
    sleep 10
    local post_latency=$(curl -sf -w "%{time_total}" -o /dev/null "${DWCP_API}/ping" | awk '{print int($1 * 1000)}')
    log_result "Post-chaos latency: ${post_latency}ms"

    return 0
}

scenario_packet_loss() {
    local target_node=${1:-"node-6"}
    local loss_percent=${2:-$MAX_PACKET_LOSS_PERCENT}
    local duration=${3:-60}

    log_chaos "Injecting ${loss_percent}% packet loss on $target_node for ${duration}s"

    local node_ip=$(curl -sf "${DWCP_API}/cluster/nodes/${target_node}/ip" | jq -r '.ip')

    # Apply packet loss
    ssh "$node_ip" "sudo tc qdisc add dev eth0 root netem loss ${loss_percent}%" 2>/dev/null || true

    # Monitor impact
    local error_count_start=$(curl -sf "${DWCP_API}/metrics/errors" | jq -r '.count_total // 0')

    sleep "$duration"

    local error_count_end=$(curl -sf "${DWCP_API}/metrics/errors" | jq -r '.count_total // 0')
    local errors_during_chaos=$((error_count_end - error_count_start))

    # Remove packet loss
    ssh "$node_ip" "sudo tc qdisc del dev eth0 root" 2>/dev/null || true

    log_result "Packet loss: ${errors_during_chaos} errors observed during chaos"

    return 0
}

scenario_cpu_stress() {
    local target_node=${1:-"node-6"}
    local stress_percent=${2:-$MAX_CPU_STRESS_PERCENT}
    local duration=${3:-60}

    log_chaos "Applying ${stress_percent}% CPU stress on $target_node for ${duration}s"

    local node_ip=$(curl -sf "${DWCP_API}/cluster/nodes/${target_node}/ip" | jq -r '.ip')

    # Calculate number of workers
    local cpu_cores=$(ssh "$node_ip" "nproc")
    local workers=$(( (cpu_cores * stress_percent) / 100 ))
    if [[ $workers -lt 1 ]]; then workers=1; fi

    # Apply CPU stress
    ssh "$node_ip" "stress-ng --cpu $workers --timeout ${duration}s &" 2>/dev/null || true

    # Monitor impact
    local start_time=$(date +%s)
    local max_cpu=0

    while [[ $(($(date +%s) - start_time)) -lt $duration ]]; do
        local current_cpu=$(curl -sf "${DWCP_API}/metrics/nodes/${target_node}/cpu" | jq -r '.percent // 0')
        if [[ $(echo "$current_cpu > $max_cpu" | bc -l) -eq 1 ]]; then
            max_cpu=$current_cpu
        fi
        sleep 5
    done

    log_result "CPU stress: Max CPU ${max_cpu}% (target ${stress_percent}%)"

    # Verify recovery
    sleep 10
    local post_cpu=$(curl -sf "${DWCP_API}/metrics/nodes/${target_node}/cpu" | jq -r '.percent // 0')
    log_result "Post-chaos CPU: ${post_cpu}%"

    return 0
}

scenario_memory_stress() {
    local target_node=${1:-"node-6"}
    local stress_percent=${2:-$MAX_MEMORY_STRESS_PERCENT}
    local duration=${3:-60}

    log_chaos "Applying ${stress_percent}% memory stress on $target_node for ${duration}s"

    local node_ip=$(curl -sf "${DWCP_API}/cluster/nodes/${target_node}/ip" | jq -r '.ip')

    # Calculate memory to consume
    local total_mem=$(ssh "$node_ip" "free -m | awk 'NR==2 {print \$2}'")
    local stress_mem=$(( (total_mem * stress_percent) / 100 ))

    # Apply memory stress
    ssh "$node_ip" "stress-ng --vm 1 --vm-bytes ${stress_mem}M --timeout ${duration}s &" 2>/dev/null || true

    # Monitor impact
    sleep "$duration"

    log_result "Memory stress: Applied ${stress_mem}MB stress"

    return 0
}

scenario_cache_invalidation() {
    log_chaos "Performing cache invalidation"

    # Flush all caches
    local components=("consensus" "network" "api" "scheduler")

    for component in "${components[@]}"; do
        log_chaos "Clearing $component cache"
        curl -X POST "${DWCP_API}/cache/${component}/clear" 2>/dev/null || true
    done

    # Monitor cache miss rate
    sleep 30

    for component in "${components[@]}"; do
        local miss_rate=$(curl -sf "${DWCP_API}/cache/${component}/stats" | jq -r '.miss_rate_1m // 0')
        log_result "$component cache miss rate: ${miss_rate}%"
    done

    return 0
}

scenario_gc_pressure() {
    local target_node=${1:-"node-6"}
    local duration=${3:-60}

    log_chaos "Applying GC pressure on $target_node for ${duration}s"

    # Trigger aggressive garbage collection
    curl -X POST "${DWCP_API}/nodes/${target_node}/gc/force" 2>/dev/null || true

    # Monitor GC activity
    local gc_count_start=$(curl -sf "${DWCP_API}/metrics/nodes/${target_node}/gc" | jq -r '.count_total // 0')

    sleep "$duration"

    local gc_count_end=$(curl -sf "${DWCP_API}/metrics/nodes/${target_node}/gc" | jq -r '.count_total // 0')
    local gc_cycles=$((gc_count_end - gc_count_start))

    log_result "GC pressure: $gc_cycles GC cycles during chaos"

    return 0
}

################################################################################
# Chaos Execution Framework
################################################################################

execute_chaos_scenario() {
    local scenario=$1
    shift
    local args=("$@")

    log_chaos "=========================================="
    log_chaos "Executing scenario: $scenario"
    log_chaos "Arguments: ${args[*]}"
    log_chaos "=========================================="

    # Pre-chaos health check
    local pre_health=$(curl -sf "${DWCP_API}/health/score" | jq -r '.score // 0')
    log_chaos "Pre-chaos health: $pre_health/100"

    # Verify safety
    if ! verify_production_safety; then
        log_chaos "Safety verification failed - scenario aborted"
        return 1
    fi

    # Verify blast radius
    if ! check_blast_radius "$scenario" 1; then
        log_chaos "Blast radius check failed - scenario aborted"
        return 1
    fi

    # Execute scenario
    local start_time=$(date +%s)

    case "$scenario" in
        "latency_injection") scenario_latency_injection "${args[@]}" ;;
        "packet_loss") scenario_packet_loss "${args[@]}" ;;
        "cpu_stress") scenario_cpu_stress "${args[@]}" ;;
        "memory_stress") scenario_memory_stress "${args[@]}" ;;
        "cache_invalidation") scenario_cache_invalidation ;;
        "gc_pressure") scenario_gc_pressure "${args[@]}" ;;
        *)
            log_chaos "Unknown scenario: $scenario"
            return 1
            ;;
    esac

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Post-chaos health check
    sleep 30  # Allow system to stabilize

    local post_health=$(curl -sf "${DWCP_API}/health/score" | jq -r '.score // 0')
    log_chaos "Post-chaos health: $post_health/100"

    # Calculate recovery
    local health_delta=$((pre_health - post_health))
    local recovery_success="true"

    if [[ $post_health -lt 95 ]]; then
        recovery_success="false"
        log_chaos "WARNING: System did not fully recover (health: $post_health/100)"
    fi

    # Record results
    record_chaos_result "$scenario" "$pre_health" "$post_health" "$duration" "$recovery_success"

    log_chaos "=========================================="
    log_chaos "Scenario completed in ${duration}s"
    log_chaos "Health delta: ${health_delta} points"
    log_chaos "Recovery: $recovery_success"
    log_chaos "=========================================="

    return 0
}

record_chaos_result() {
    local scenario=$1
    local pre_health=$2
    local post_health=$3
    local duration=$4
    local recovery=$5

    if [[ ! -f "$CHAOS_RESULTS" ]]; then
        echo '{"scenarios":[]}' > "$CHAOS_RESULTS"
    fi

    local result=$(cat <<EOF
{
    "scenario": "$scenario",
    "timestamp": $(date +%s),
    "pre_health": $pre_health,
    "post_health": $post_health,
    "duration": $duration,
    "recovery_success": $recovery,
    "health_delta": $((pre_health - post_health))
}
EOF
)

    jq --argjson result "$result" '.scenarios += [$result]' "$CHAOS_RESULTS" > "${CHAOS_RESULTS}.tmp" && mv "${CHAOS_RESULTS}.tmp" "$CHAOS_RESULTS"
}

################################################################################
# Continuous Chaos Loop
################################################################################

continuous_chaos() {
    local interval=${1:-3600}  # Default: 1 hour between chaos tests

    log_chaos "Starting continuous chaos engineering (interval: ${interval}s)"

    while true; do
        # Randomly select a scenario (weighted towards less impactful ones)
        local scenarios=("cache_invalidation" "gc_pressure" "latency_injection" "cpu_stress")
        local random_index=$((RANDOM % ${#scenarios[@]}))
        local selected_scenario=${scenarios[$random_index]}

        log_chaos "Selected scenario for this cycle: $selected_scenario"

        # Execute scenario
        execute_chaos_scenario "$selected_scenario"

        # Wait before next chaos injection
        log_chaos "Waiting ${interval}s before next chaos injection..."
        sleep "$interval"
    done
}

################################################################################
# Generate Report
################################################################################

generate_chaos_report() {
    log_chaos "Generating production chaos engineering report..."

    if [[ ! -f "$CHAOS_RESULTS" ]]; then
        echo "No chaos results available"
        return 1
    fi

    local total_scenarios=$(jq '.scenarios | length' "$CHAOS_RESULTS")
    local successful_recoveries=$(jq '[.scenarios[] | select(.recovery_success == "true")] | length' "$CHAOS_RESULTS")
    local avg_health_delta=$(jq '[.scenarios[].health_delta] | add / length' "$CHAOS_RESULTS")

    echo "=========================================="
    echo "Production Chaos Engineering Report"
    echo "=========================================="
    echo "Total Scenarios Executed: $total_scenarios"
    echo "Successful Recoveries: $successful_recoveries"
    echo "Recovery Rate: $(echo "scale=2; $successful_recoveries * 100 / $total_scenarios" | bc)%"
    echo "Average Health Delta: $avg_health_delta points"
    echo "=========================================="

    # Show recent scenarios
    echo ""
    echo "Recent Scenarios:"
    jq -r '.scenarios[-5:] | .[] | "\(.scenario): health \(.pre_health) -> \(.post_health), recovery: \(.recovery_success)"' "$CHAOS_RESULTS"

    return 0
}

################################################################################
# Main
################################################################################

main() {
    log_chaos "DWCP v3 Production Chaos Engineering System"

    # Create necessary directories
    mkdir -p "$(dirname "$CHAOS_LOG")" "$(dirname "$CHAOS_RESULTS")"

    # Initialize results file
    if [[ ! -f "$CHAOS_RESULTS" ]]; then
        echo '{"scenarios":[],"start_time":'$(date +%s)'}' > "$CHAOS_RESULTS"
    fi

    # Start continuous chaos
    continuous_chaos 3600  # 1-hour interval
}

# Handle termination
trap 'log_chaos "Production chaos system shutting down"; exit 0' SIGTERM SIGINT

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        continuous)
            # Continuous chaos mode
            main
            ;;
        once)
            # Single scenario execution
            if [[ -z "${2:-}" ]]; then
                echo "Usage: $0 once <scenario> [args...]"
                echo "Available scenarios: ${!CHAOS_SCENARIOS[@]}"
                exit 1
            fi
            execute_chaos_scenario "$2" "${@:3}"
            ;;
        report)
            # Generate report
            generate_chaos_report
            ;;
        list)
            # List available scenarios
            echo "Available chaos scenarios:"
            for scenario in "${!CHAOS_SCENARIOS[@]}"; do
                echo "  - $scenario: ${CHAOS_SCENARIOS[$scenario]}"
            done
            ;;
        *)
            echo "Usage: $0 {continuous|once|report|list}"
            exit 1
            ;;
    esac
fi