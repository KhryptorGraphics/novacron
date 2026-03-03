#!/bin/bash

################################################################################
# DWCP v3 Automated Health Checker
# Comprehensive health checking system with self-healing capabilities
################################################################################

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
HEALTH_LOG="/var/log/novacron/health.log"
HEALTH_STATE="/var/lib/novacron/health-state.json"
DWCP_API="${DWCP_API:-http://localhost:8080/api/v3}"
CHECK_INTERVAL=30
FAILURE_THRESHOLD=3
RECOVERY_WAIT=10

# Health check thresholds
CPU_THRESHOLD=85
MEMORY_THRESHOLD=90
LATENCY_THRESHOLD=100
ERROR_RATE_THRESHOLD=1.0

################################################################################
# Logging
################################################################################

log_health() {
    echo -e "${GREEN}[HEALTH]${NC} $1" | tee -a "$HEALTH_LOG"
}

log_degraded() {
    echo -e "${YELLOW}[DEGRADED]${NC} $1" | tee -a "$HEALTH_LOG"
}

log_unhealthy() {
    echo -e "${RED}[UNHEALTHY]${NC} $1" | tee -a "$HEALTH_LOG"
}

################################################################################
# Component Health Checks
################################################################################

check_consensus_layer() {
    local component="consensus"
    local health_score=100

    # Check leader election
    local leader=$(curl -sf "${DWCP_API}/consensus/leader" | jq -r '.node_id // "none"')
    if [[ "$leader" == "none" ]]; then
        log_unhealthy "No consensus leader elected"
        health_score=$((health_score - 50))
    fi

    # Check quorum
    local quorum=$(curl -sf "${DWCP_API}/consensus/quorum" | jq -r '.status // "unknown"')
    if [[ "$quorum" != "healthy" ]]; then
        log_unhealthy "Quorum not healthy: $quorum"
        health_score=$((health_score - 40))
    fi

    # Check log replication lag
    local max_lag=$(curl -sf "${DWCP_API}/consensus/replication-lag" | jq -r '.max_lag_ms // 0')
    if [[ $max_lag -gt 1000 ]]; then
        log_degraded "High replication lag: ${max_lag}ms"
        health_score=$((health_score - 20))
    fi

    # Check Byzantine detection
    local byzantine_count=$(curl -sf "${DWCP_API}/consensus/byzantine-nodes" | jq -r '.count // 0')
    if [[ $byzantine_count -gt 0 ]]; then
        log_unhealthy "Byzantine nodes detected: $byzantine_count"
        health_score=$((health_score - 30))
    fi

    echo "$health_score"
}

check_network_layer() {
    local component="network"
    local health_score=100

    # Check connectivity
    local connected_nodes=$(curl -sf "${DWCP_API}/network/peers" | jq -r '.connected // 0')
    local expected_nodes=$(curl -sf "${DWCP_API}/network/peers" | jq -r '.expected // 7')

    if [[ $connected_nodes -lt $expected_nodes ]]; then
        local missing=$((expected_nodes - connected_nodes))
        log_degraded "Missing $missing network peers"
        health_score=$((health_score - (missing * 10)))
    fi

    # Check packet loss
    local packet_loss=$(curl -sf "${DWCP_API}/network/stats" | jq -r '.packet_loss_percent // 0')
    if [[ $(echo "$packet_loss > 5" | bc -l) -eq 1 ]]; then
        log_degraded "High packet loss: ${packet_loss}%"
        health_score=$((health_score - 20))
    fi

    # Check bandwidth utilization
    local bandwidth_util=$(curl -sf "${DWCP_API}/network/stats" | jq -r '.bandwidth_utilization_percent // 0')
    if [[ $(echo "$bandwidth_util > 90" | bc -l) -eq 1 ]]; then
        log_degraded "High bandwidth utilization: ${bandwidth_util}%"
        health_score=$((health_score - 15))
    fi

    # Check partition detection
    local partition_detected=$(curl -sf "${DWCP_API}/network/partition-status" | jq -r '.partition_detected // false')
    if [[ "$partition_detected" == "true" ]]; then
        log_unhealthy "Network partition detected"
        health_score=0
    fi

    echo "$health_score"
}

check_storage_layer() {
    local component="storage"
    local health_score=100

    # Check disk usage
    local disk_usage=$(df -h /var/lib/novacron | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        log_unhealthy "Disk usage critical: ${disk_usage}%"
        health_score=$((health_score - 40))
    elif [[ $disk_usage -gt 80 ]]; then
        log_degraded "Disk usage high: ${disk_usage}%"
        health_score=$((health_score - 20))
    fi

    # Check database health
    local db_status=$(curl -sf "${DWCP_API}/database/health" | jq -r '.status // "unknown"')
    if [[ "$db_status" != "healthy" ]]; then
        log_unhealthy "Database unhealthy: $db_status"
        health_score=$((health_score - 50))
    fi

    # Check replication lag
    local replication_lag=$(curl -sf "${DWCP_API}/database/replication-lag" | jq -r '.lag_ms // 0')
    if [[ $replication_lag -gt 500 ]]; then
        log_degraded "High database replication lag: ${replication_lag}ms"
        health_score=$((health_score - 15))
    fi

    # Check I/O wait
    local io_wait=$(iostat -x 1 2 | awk '/avg/ {print $4}' | tail -1 | cut -d. -f1)
    if [[ $io_wait -gt 30 ]]; then
        log_degraded "High I/O wait: ${io_wait}%"
        health_score=$((health_score - 10))
    fi

    echo "$health_score"
}

check_api_layer() {
    local component="api"
    local health_score=100

    # Check response time
    local response_time=$(curl -sf -w "%{time_total}" -o /dev/null "${DWCP_API}/ping" | awk '{print int($1 * 1000)}')
    if [[ $response_time -gt $LATENCY_THRESHOLD ]]; then
        log_degraded "High API latency: ${response_time}ms"
        health_score=$((health_score - 20))
    fi

    # Check error rate
    local error_rate=$(curl -sf "${DWCP_API}/metrics/errors" | jq -r '.rate_1m // 0')
    if [[ $(echo "$error_rate > $ERROR_RATE_THRESHOLD" | bc -l) -eq 1 ]]; then
        log_degraded "High error rate: ${error_rate}%"
        health_score=$((health_score - 25))
    fi

    # Check request queue depth
    local queue_depth=$(curl -sf "${DWCP_API}/metrics/queue" | jq -r '.depth // 0')
    if [[ $queue_depth -gt 1000 ]]; then
        log_degraded "High request queue depth: $queue_depth"
        health_score=$((health_score - 15))
    fi

    # Check concurrent connections
    local connections=$(curl -sf "${DWCP_API}/metrics/connections" | jq -r '.active // 0')
    local max_connections=$(curl -sf "${DWCP_API}/metrics/connections" | jq -r '.max // 10000')
    local connection_ratio=$((connections * 100 / max_connections))
    if [[ $connection_ratio -gt 90 ]]; then
        log_degraded "High connection usage: ${connection_ratio}%"
        health_score=$((health_score - 10))
    fi

    echo "$health_score"
}

check_scheduler_layer() {
    local component="scheduler"
    local health_score=100

    # Check pending jobs
    local pending_jobs=$(curl -sf "${DWCP_API}/scheduler/stats" | jq -r '.pending // 0')
    if [[ $pending_jobs -gt 1000 ]]; then
        log_degraded "High pending job count: $pending_jobs"
        health_score=$((health_score - 20))
    fi

    # Check failed jobs
    local failed_jobs=$(curl -sf "${DWCP_API}/scheduler/stats" | jq -r '.failed_1h // 0')
    if [[ $failed_jobs -gt 10 ]]; then
        log_degraded "High failed job count: $failed_jobs"
        health_score=$((health_score - 25))
    fi

    # Check scheduler latency
    local scheduler_latency=$(curl -sf "${DWCP_API}/scheduler/metrics" | jq -r '.avg_latency_ms // 0')
    if [[ $scheduler_latency -gt 500 ]]; then
        log_degraded "High scheduler latency: ${scheduler_latency}ms"
        health_score=$((health_score - 15))
    fi

    echo "$health_score"
}

check_vm_layer() {
    local component="vm"
    local health_score=100

    # Check active VMs
    local active_vms=$(curl -sf "${DWCP_API}/vms/count" | jq -r '.active // 0')
    local capacity=$(curl -sf "${DWCP_API}/vms/capacity" | jq -r '.total // 100')
    local vm_utilization=$((active_vms * 100 / capacity))

    if [[ $vm_utilization -gt 95 ]]; then
        log_unhealthy "VM capacity critical: ${vm_utilization}%"
        health_score=$((health_score - 30))
    elif [[ $vm_utilization -gt 85 ]]; then
        log_degraded "VM capacity high: ${vm_utilization}%"
        health_score=$((health_score - 15))
    fi

    # Check failed VM starts
    local failed_starts=$(curl -sf "${DWCP_API}/vms/stats" | jq -r '.failed_starts_1h // 0')
    if [[ $failed_starts -gt 5 ]]; then
        log_degraded "High VM start failures: $failed_starts"
        health_score=$((health_score - 20))
    fi

    # Check VM migration failures
    local failed_migrations=$(curl -sf "${DWCP_API}/vms/migrations" | jq -r '.failed_1h // 0')
    if [[ $failed_migrations -gt 3 ]]; then
        log_degraded "High VM migration failures: $failed_migrations"
        health_score=$((health_score - 15))
    fi

    echo "$health_score"
}

check_security_layer() {
    local component="security"
    local health_score=100

    # Check certificate expiration
    local cert_days=$(curl -sf "${DWCP_API}/security/certificates" | jq -r '.days_until_expiry // 999')
    if [[ $cert_days -lt 7 ]]; then
        log_unhealthy "Certificate expiring soon: $cert_days days"
        health_score=$((health_score - 40))
    elif [[ $cert_days -lt 30 ]]; then
        log_degraded "Certificate expiring: $cert_days days"
        health_score=$((health_score - 20))
    fi

    # Check authentication failures
    local auth_failures=$(curl -sf "${DWCP_API}/security/auth-failures" | jq -r '.count_1h // 0')
    if [[ $auth_failures -gt 100 ]]; then
        log_degraded "High authentication failures: $auth_failures"
        health_score=$((health_score - 15))
    fi

    # Check security policy violations
    local policy_violations=$(curl -sf "${DWCP_API}/security/violations" | jq -r '.count_1h // 0')
    if [[ $policy_violations -gt 0 ]]; then
        log_unhealthy "Security policy violations: $policy_violations"
        health_score=$((health_score - 30))
    fi

    echo "$health_score"
}

################################################################################
# System Resource Checks
################################################################################

check_system_resources() {
    local health_score=100

    # CPU check
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}' | cut -d. -f1)
    if [[ $cpu_usage -gt $CPU_THRESHOLD ]]; then
        log_degraded "High CPU usage: ${cpu_usage}%"
        health_score=$((health_score - 20))
    fi

    # Memory check
    local mem_usage=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    if [[ $mem_usage -gt $MEMORY_THRESHOLD ]]; then
        log_degraded "High memory usage: ${mem_usage}%"
        health_score=$((health_score - 20))
    fi

    # Load average check
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_cores=$(nproc)
    local load_per_core=$(echo "$load_avg / $cpu_cores" | bc -l | cut -d. -f1)

    if [[ $load_per_core -gt 2 ]]; then
        log_degraded "High load average: $load_avg (${load_per_core}x cores)"
        health_score=$((health_score - 15))
    fi

    # Network connectivity check
    if ! ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
        log_unhealthy "No external network connectivity"
        health_score=$((health_score - 30))
    fi

    echo "$health_score"
}

################################################################################
# Self-Healing Actions
################################################################################

self_heal_transient_failure() {
    local component=$1
    local health_score=$2

    log_health "Attempting self-healing for $component (score: $health_score)"

    # Clear caches
    if systemctl is-active "novacron-${component}" > /dev/null 2>&1; then
        curl -X POST "${DWCP_API}/cache/${component}/clear" 2>/dev/null || true
    fi

    # Restart if severely degraded
    if [[ $health_score -lt 50 ]]; then
        log_health "Restarting $component service"
        systemctl restart "novacron-${component}" 2>/dev/null || true
        sleep $RECOVERY_WAIT
    fi

    # Trigger garbage collection
    curl -X POST "${DWCP_API}/gc/${component}" 2>/dev/null || true

    return 0
}

################################################################################
# Health State Management
################################################################################

save_health_state() {
    local component=$1
    local health_score=$2
    local status=$3

    if [[ ! -f "$HEALTH_STATE" ]]; then
        echo '{}' > "$HEALTH_STATE"
    fi

    jq --arg component "$component" \
       --arg score "$health_score" \
       --arg status "$status" \
       --arg timestamp "$(date +%s)" \
       '.[$component] = {"score": ($score | tonumber), "status": $status, "timestamp": ($timestamp | tonumber), "failures": ((.[$component].failures // 0) + (if $status != "healthy" then 1 else -(.[$component].failures // 0) end))}' \
       "$HEALTH_STATE" > "${HEALTH_STATE}.tmp" && mv "${HEALTH_STATE}.tmp" "$HEALTH_STATE"
}

get_failure_count() {
    local component=$1

    if [[ -f "$HEALTH_STATE" ]]; then
        jq -r --arg component "$component" '.[$component].failures // 0' "$HEALTH_STATE"
    else
        echo "0"
    fi
}

################################################################################
# Comprehensive Health Check
################################################################################

perform_health_checks() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    log_health "==================== Health Check: $timestamp ===================="

    local overall_score=100
    local component_count=0
    local unhealthy_components=()

    # Check each component
    local components=("consensus" "network" "storage" "api" "scheduler" "vm" "security" "system")

    for component in "${components[@]}"; do
        local score=0

        case "$component" in
            "consensus") score=$(check_consensus_layer) ;;
            "network") score=$(check_network_layer) ;;
            "storage") score=$(check_storage_layer) ;;
            "api") score=$(check_api_layer) ;;
            "scheduler") score=$(check_scheduler_layer) ;;
            "vm") score=$(check_vm_layer) ;;
            "security") score=$(check_security_layer) ;;
            "system") score=$(check_system_resources) ;;
        esac

        # Determine status
        local status="healthy"
        if [[ $score -lt 50 ]]; then
            status="unhealthy"
            unhealthy_components+=("$component")
        elif [[ $score -lt 80 ]]; then
            status="degraded"
        fi

        # Log status
        if [[ "$status" == "healthy" ]]; then
            log_health "$component: $score/100 - ${GREEN}HEALTHY${NC}"
        elif [[ "$status" == "degraded" ]]; then
            log_degraded "$component: $score/100 - ${YELLOW}DEGRADED${NC}"
        else
            log_unhealthy "$component: $score/100 - ${RED}UNHEALTHY${NC}"
        fi

        # Save state
        save_health_state "$component" "$score" "$status"

        # Check failure threshold
        local failures=$(get_failure_count "$component")
        if [[ $failures -ge $FAILURE_THRESHOLD ]] && [[ "$status" != "healthy" ]]; then
            # Attempt self-healing
            self_heal_transient_failure "$component" "$score"
        fi

        # Accumulate overall score
        overall_score=$((overall_score + score))
        component_count=$((component_count + 1))
    done

    # Calculate average health
    local avg_health=$((overall_score / component_count))

    log_health "==================== Overall Health: $avg_health/100 ===================="

    # Trigger incident response for unhealthy components
    if [[ ${#unhealthy_components[@]} -gt 0 ]]; then
        log_unhealthy "Unhealthy components detected: ${unhealthy_components[*]}"

        # Trigger incident response system
        if [[ -x /home/kp/novacron/scripts/production/incident-response.sh ]]; then
            for component in "${unhealthy_components[@]}"; do
                /home/kp/novacron/scripts/production/incident-response.sh handle "$component" &
            done
        fi
    fi

    return 0
}

################################################################################
# Main Loop
################################################################################

main() {
    log_health "Starting DWCP v3 Automated Health Checker"

    # Create necessary directories
    mkdir -p "$(dirname "$HEALTH_LOG")" "$(dirname "$HEALTH_STATE")"

    # Initialize health state
    if [[ ! -f "$HEALTH_STATE" ]]; then
        echo '{}' > "$HEALTH_STATE"
    fi

    # Continuous health checking
    while true; do
        perform_health_checks
        sleep $CHECK_INTERVAL
    done
}

# Handle termination
trap 'log_health "Health Checker shutting down"; exit 0' SIGTERM SIGINT

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        once)
            # Single check
            perform_health_checks
            ;;
        status)
            # Show current health state
            if [[ -f "$HEALTH_STATE" ]]; then
                jq . "$HEALTH_STATE"
            else
                echo "No health state available"
            fi
            ;;
        *)
            main
            ;;
    esac
fi