#!/bin/bash

################################################################################
# DWCP v3 Automatic Rollback System
# Intelligent rollback decision engine with validation
################################################################################

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ROLLBACK_LOG="/var/log/novacron/rollback.log"
ROLLBACK_STATE="/var/lib/novacron/rollback-state.json"
DEPLOYMENT_STATE="/var/lib/novacron/deployment-state.json"
DWCP_API="${DWCP_API:-http://localhost:8080/api/v3}"

# Rollback thresholds
ERROR_RATE_THRESHOLD=1.0          # % - Automatic rollback if exceeded
LATENCY_P99_THRESHOLD=100         # ms - Automatic rollback if exceeded
CPU_THRESHOLD=85                  # % - Warning threshold
MEMORY_THRESHOLD=90               # % - Warning threshold
SUSTAINED_DEGRADATION_TIME=120    # seconds - How long to observe before rollback
VALIDATION_TIME=60                # seconds - Post-rollback validation time

# Rollback modes
ROLLBACK_MODE_AUTO="auto"
ROLLBACK_MODE_MANUAL="manual"
ROLLBACK_MODE_EMERGENCY="emergency"

################################################################################
# Logging
################################################################################

log_info() {
    echo -e "${GREEN}[ROLLBACK]${NC} $1" | tee -a "$ROLLBACK_LOG"
}

log_warn() {
    echo -e "${YELLOW}[ROLLBACK]${NC} $1" | tee -a "$ROLLBACK_LOG"
}

log_error() {
    echo -e "${RED}[ROLLBACK]${NC} $1" | tee -a "$ROLLBACK_LOG"
}

################################################################################
# State Management
################################################################################

get_current_version() {
    if [[ -f "$DEPLOYMENT_STATE" ]]; then
        jq -r '.current_version // "unknown"' "$DEPLOYMENT_STATE"
    else
        echo "unknown"
    fi
}

get_previous_version() {
    if [[ -f "$DEPLOYMENT_STATE" ]]; then
        jq -r '.previous_version // "unknown"' "$DEPLOYMENT_STATE"
    else
        echo "unknown"
    fi
}

get_deployment_start_time() {
    if [[ -f "$DEPLOYMENT_STATE" ]]; then
        jq -r '.deployment_start_time // 0' "$DEPLOYMENT_STATE"
    else
        echo "0"
    fi
}

save_rollback_decision() {
    local decision=$1
    local reason=$2
    local metrics=$3

    local rollback_data=$(cat <<EOF
{
    "decision": "$decision",
    "reason": "$reason",
    "metrics": $metrics,
    "timestamp": $(date +%s),
    "current_version": "$(get_current_version)",
    "target_version": "$(get_previous_version)"
}
EOF
)

    echo "$rollback_data" >> "$ROLLBACK_LOG"
    echo "$rollback_data" > "$ROLLBACK_STATE"
}

################################################################################
# Metrics Collection
################################################################################

collect_current_metrics() {
    local error_rate=$(curl -sf "${DWCP_API}/metrics/errors" | jq -r '.rate_1m // 0')
    local latency_p99=$(curl -sf "${DWCP_API}/metrics/latency" | jq -r '.p99_ms // 0')
    local latency_p50=$(curl -sf "${DWCP_API}/metrics/latency" | jq -r '.p50_ms // 0')
    local cpu_usage=$(curl -sf "${DWCP_API}/metrics/system" | jq -r '.cpu_percent // 0')
    local memory_usage=$(curl -sf "${DWCP_API}/metrics/system" | jq -r '.memory_percent // 0')
    local request_rate=$(curl -sf "${DWCP_API}/metrics/requests" | jq -r '.rate_1m // 0')
    local success_rate=$(curl -sf "${DWCP_API}/metrics/requests" | jq -r '.success_rate_1m // 0')

    cat <<EOF
{
    "error_rate": $error_rate,
    "latency_p99": $latency_p99,
    "latency_p50": $latency_p50,
    "cpu_usage": $cpu_usage,
    "memory_usage": $memory_usage,
    "request_rate": $request_rate,
    "success_rate": $success_rate,
    "timestamp": $(date +%s)
}
EOF
}

################################################################################
# Decision Engine
################################################################################

evaluate_error_rate() {
    local error_rate=$1

    if [[ $(echo "$error_rate > $ERROR_RATE_THRESHOLD" | bc -l) -eq 1 ]]; then
        echo "true"
    else
        echo "false"
    fi
}

evaluate_latency() {
    local latency_p99=$1

    if [[ $latency_p99 -gt $LATENCY_P99_THRESHOLD ]]; then
        echo "true"
    else
        echo "false"
    fi
}

evaluate_sustained_degradation() {
    local component=$1
    local current_time=$(date +%s)
    local deployment_start=$(get_deployment_start_time)

    # Must be deployed for at least the degradation window
    if [[ $((current_time - deployment_start)) -lt $SUSTAINED_DEGRADATION_TIME ]]; then
        echo "false"
        return
    fi

    # Check historical metrics for sustained issues
    local degraded_samples=0
    local total_samples=0

    # Sample metrics over the degradation window
    for i in $(seq 1 4); do
        sleep $((SUSTAINED_DEGRADATION_TIME / 4))

        local metrics=$(collect_current_metrics)
        local error_rate=$(echo "$metrics" | jq -r '.error_rate')
        local latency=$(echo "$metrics" | jq -r '.latency_p99')

        total_samples=$((total_samples + 1))

        if [[ $(evaluate_error_rate "$error_rate") == "true" ]] || [[ $(evaluate_latency "$latency") == "true" ]]; then
            degraded_samples=$((degraded_samples + 1))
        fi
    done

    # If 75% of samples show degradation, it's sustained
    if [[ $((degraded_samples * 100 / total_samples)) -ge 75 ]]; then
        echo "true"
    else
        echo "false"
    fi
}

should_rollback_automatic() {
    local metrics=$1

    local error_rate=$(echo "$metrics" | jq -r '.error_rate')
    local latency_p99=$(echo "$metrics" | jq -r '.latency_p99')
    local success_rate=$(echo "$metrics" | jq -r '.success_rate')

    local reasons=()

    # Check error rate
    if [[ $(evaluate_error_rate "$error_rate") == "true" ]]; then
        reasons+=("High error rate: ${error_rate}%")
    fi

    # Check latency
    if [[ $(evaluate_latency "$latency_p99") == "true" ]]; then
        reasons+=("High latency: ${latency_p99}ms")
    fi

    # Check success rate
    if [[ $(echo "$success_rate < 99.0" | bc -l) -eq 1 ]]; then
        reasons+=("Low success rate: ${success_rate}%")
    fi

    # Return decision
    if [[ ${#reasons[@]} -gt 0 ]]; then
        echo "true|$(IFS='; '; echo "${reasons[*]}")"
    else
        echo "false|"
    fi
}

################################################################################
# Rollback Execution
################################################################################

execute_rollback() {
    local mode=$1
    local reason=$2

    local current_version=$(get_current_version)
    local target_version=$(get_previous_version)

    log_info "==================== ROLLBACK INITIATED ===================="
    log_info "Mode: $mode"
    log_info "Reason: $reason"
    log_info "Current Version: $current_version"
    log_info "Target Version: $target_version"
    log_info "============================================================"

    local rollback_start=$(date +%s)

    # Step 1: Pause new deployments
    log_info "Step 1: Pausing new deployments"
    curl -X POST "${DWCP_API}/deployment/pause" 2>/dev/null || true

    # Step 2: Drain traffic gracefully
    log_info "Step 2: Draining traffic"
    curl -X POST "${DWCP_API}/traffic/drain" \
        -H "Content-Type: application/json" \
        -d '{"timeout_seconds": 30}' 2>/dev/null || true
    sleep 30

    # Step 3: Stop current version services
    log_info "Step 3: Stopping current version services"
    systemctl stop novacron-dwcp-v3 2>/dev/null || true

    # Step 4: Restore previous version
    log_info "Step 4: Restoring previous version ($target_version)"

    # Restore binaries
    if [[ -d "/opt/novacron/versions/${target_version}" ]]; then
        rm -f /opt/novacron/current
        ln -s "/opt/novacron/versions/${target_version}" /opt/novacron/current
    else
        log_error "Previous version binaries not found!"
        return 1
    fi

    # Restore configuration
    if [[ -f "/etc/novacron/config.${target_version}.yaml" ]]; then
        cp "/etc/novacron/config.${target_version}.yaml" /etc/novacron/config.yaml
    fi

    # Step 5: Start previous version services
    log_info "Step 5: Starting previous version services"
    systemctl start novacron-dwcp-v2 2>/dev/null || true
    sleep 10

    # Step 6: Verify service health
    log_info "Step 6: Verifying service health"
    local health_check_attempts=0
    local max_health_checks=12  # 1 minute with 5-second intervals

    while [[ $health_check_attempts -lt $max_health_checks ]]; do
        if curl -sf "${DWCP_API}/health" > /dev/null 2>&1; then
            log_info "Service health check passed"
            break
        fi

        health_check_attempts=$((health_check_attempts + 1))
        sleep 5
    done

    if [[ $health_check_attempts -ge $max_health_checks ]]; then
        log_error "Service health check failed after rollback!"
        return 1
    fi

    # Step 7: Restore traffic
    log_info "Step 7: Restoring traffic"
    curl -X POST "${DWCP_API}/traffic/restore" 2>/dev/null || true

    # Step 8: Resume deployments
    log_info "Step 8: Resuming deployment capability"
    curl -X POST "${DWCP_API}/deployment/resume" 2>/dev/null || true

    local rollback_end=$(date +%s)
    local rollback_duration=$((rollback_end - rollback_start))

    log_info "==================== ROLLBACK COMPLETED ===================="
    log_info "Duration: ${rollback_duration} seconds"
    log_info "Target Version: $target_version"
    log_info "============================================================"

    # Update deployment state
    if [[ -f "$DEPLOYMENT_STATE" ]]; then
        jq --arg version "$target_version" \
           --arg timestamp "$(date +%s)" \
           '.current_version = $version | .rollback_time = ($timestamp | tonumber) | .rollback_count = ((.rollback_count // 0) + 1)' \
           "$DEPLOYMENT_STATE" > "${DEPLOYMENT_STATE}.tmp" && mv "${DEPLOYMENT_STATE}.tmp" "$DEPLOYMENT_STATE"
    fi

    return 0
}

validate_rollback() {
    log_info "Validating rollback success..."

    sleep $VALIDATION_TIME

    # Collect post-rollback metrics
    local metrics=$(collect_current_metrics)

    local error_rate=$(echo "$metrics" | jq -r '.error_rate')
    local latency_p99=$(echo "$metrics" | jq -r '.latency_p99')
    local success_rate=$(echo "$metrics" | jq -r '.success_rate')

    log_info "Post-rollback metrics:"
    log_info "  Error Rate: ${error_rate}%"
    log_info "  Latency P99: ${latency_p99}ms"
    log_info "  Success Rate: ${success_rate}%"

    # Validate metrics are within acceptable ranges
    local validation_passed=true

    if [[ $(echo "$error_rate > $ERROR_RATE_THRESHOLD" | bc -l) -eq 1 ]]; then
        log_error "Validation failed: Error rate still high"
        validation_passed=false
    fi

    if [[ $latency_p99 -gt $LATENCY_P99_THRESHOLD ]]; then
        log_error "Validation failed: Latency still high"
        validation_passed=false
    fi

    if [[ $(echo "$success_rate < 99.0" | bc -l) -eq 1 ]]; then
        log_error "Validation failed: Success rate still low"
        validation_passed=false
    fi

    if [[ "$validation_passed" == "true" ]]; then
        log_info "Rollback validation: PASSED"
        return 0
    else
        log_error "Rollback validation: FAILED"
        return 1
    fi
}

################################################################################
# Continuous Monitoring
################################################################################

monitor_and_decide() {
    log_info "Starting rollback monitoring..."

    while true; do
        # Check if deployment is in progress
        if [[ ! -f "$DEPLOYMENT_STATE" ]]; then
            sleep 30
            continue
        fi

        local deployment_status=$(jq -r '.status // "none"' "$DEPLOYMENT_STATE")

        if [[ "$deployment_status" != "in_progress" ]]; then
            sleep 30
            continue
        fi

        # Collect current metrics
        local metrics=$(collect_current_metrics)

        # Evaluate rollback decision
        local decision_result=$(should_rollback_automatic "$metrics")
        local should_rollback=$(echo "$decision_result" | cut -d'|' -f1)
        local reasons=$(echo "$decision_result" | cut -d'|' -f2)

        if [[ "$should_rollback" == "true" ]]; then
            log_warn "Automatic rollback triggered: $reasons"

            # Check if this is sustained degradation
            if [[ $(evaluate_sustained_degradation "deployment") == "true" ]]; then
                log_error "SUSTAINED DEGRADATION DETECTED - INITIATING ROLLBACK"

                # Save decision
                save_rollback_decision "automatic" "$reasons" "$metrics"

                # Execute rollback
                if execute_rollback "$ROLLBACK_MODE_AUTO" "$reasons"; then
                    # Validate rollback
                    if validate_rollback; then
                        log_info "Automatic rollback completed successfully"
                    else
                        log_error "Rollback validation failed - manual intervention required"
                    fi
                else
                    log_error "Rollback execution failed - manual intervention required"
                fi

                # Exit monitoring loop after rollback
                break
            else
                log_warn "Degradation detected but not sustained yet, continuing monitoring"
            fi
        fi

        sleep 30
    done
}

################################################################################
# Manual Rollback
################################################################################

manual_rollback() {
    local reason="${1:-Manual rollback requested}"

    log_info "Manual rollback requested: $reason"

    # Confirm with operator
    echo -e "${YELLOW}Manual rollback confirmation required${NC}"
    echo "Current version: $(get_current_version)"
    echo "Target version: $(get_previous_version)"
    echo "Reason: $reason"
    echo ""
    read -p "Confirm rollback? (yes/no): " confirmation

    if [[ "$confirmation" != "yes" ]]; then
        log_info "Rollback cancelled by operator"
        return 1
    fi

    # Collect current metrics
    local metrics=$(collect_current_metrics)

    # Save decision
    save_rollback_decision "manual" "$reason" "$metrics"

    # Execute rollback
    if execute_rollback "$ROLLBACK_MODE_MANUAL" "$reason"; then
        # Validate rollback
        if validate_rollback; then
            log_info "Manual rollback completed successfully"
            return 0
        else
            log_error "Rollback validation failed"
            return 1
        fi
    else
        log_error "Rollback execution failed"
        return 1
    fi
}

################################################################################
# Emergency Rollback
################################################################################

emergency_rollback() {
    local reason="${1:-Emergency rollback - critical failure}"

    log_error "EMERGENCY ROLLBACK INITIATED: $reason"

    # No confirmation required for emergency
    local metrics=$(collect_current_metrics)

    # Save decision
    save_rollback_decision "emergency" "$reason" "$metrics"

    # Execute rollback immediately
    if execute_rollback "$ROLLBACK_MODE_EMERGENCY" "$reason"; then
        log_info "Emergency rollback completed"
        return 0
    else
        log_error "Emergency rollback failed"
        return 1
    fi
}

################################################################################
# Main
################################################################################

main() {
    log_info "Starting DWCP v3 Automatic Rollback System"

    # Create necessary directories
    mkdir -p "$(dirname "$ROLLBACK_LOG")" "$(dirname "$ROLLBACK_STATE")"

    # Start monitoring
    monitor_and_decide
}

# Handle termination
trap 'log_info "Rollback system shutting down"; exit 0' SIGTERM SIGINT

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        auto)
            # Automatic monitoring mode
            main
            ;;
        manual)
            # Manual rollback
            manual_rollback "${2:-Manual rollback requested}"
            ;;
        emergency)
            # Emergency rollback
            emergency_rollback "${2:-Emergency rollback}"
            ;;
        validate)
            # Validate current deployment
            validate_rollback
            ;;
        status)
            # Show rollback state
            if [[ -f "$ROLLBACK_STATE" ]]; then
                jq . "$ROLLBACK_STATE"
            else
                echo "No rollback state available"
            fi
            ;;
        *)
            echo "Usage: $0 {auto|manual|emergency|validate|status} [reason]"
            exit 1
            ;;
    esac
fi