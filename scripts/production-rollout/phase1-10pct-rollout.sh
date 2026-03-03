#!/usr/bin/env bash
# DWCP v3 Phase 5: Production Rollout Phase 1 (10% Traffic)
# Automated gradual rollout with health monitoring and automatic rollback
# Usage: ./phase1-10pct-rollout.sh

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
ENVIRONMENT="production"
NAMESPACE="production"
ROLLOUT_PERCENTAGE=10
MONITORING_DURATION=600  # 10 minutes
HEALTH_CHECK_INTERVAL=30
ERROR_THRESHOLD=1.0  # 1% error rate threshold
LATENCY_THRESHOLD=100  # 100ms P99 latency threshold

# Rollback configuration
AUTO_ROLLBACK=true
ROLLBACK_TRIGGERS=()

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

section() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN} $*${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════${NC}\n"
}

check_prerequisites() {
    section "Pre-Rollout Validation"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
        exit 1
    fi

    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    local context=$(kubectl config current-context)
    warning "Production rollout to context: $context"

    # Verify this is production
    if [[ ! "$context" =~ production ]]; then
        warning "Context does not appear to be production: $context"
        read -p "Continue anyway? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            error "Rollout cancelled"
            exit 1
        fi
    fi

    # Check deployment exists
    if ! kubectl get deployment dwcp-v3 -n "$NAMESPACE" &> /dev/null; then
        error "DWCP v3 deployment not found in $NAMESPACE"
        exit 1
    fi

    # Check current rollout percentage
    local current_rollout=$(kubectl get configmap dwcp-v3-config -n "$NAMESPACE" \
        -o jsonpath='{.data.FEATURE_FLAG_V3_ROLLOUT}' 2>/dev/null || echo "0")

    log "Current rollout percentage: ${current_rollout}%"

    if [ "$current_rollout" -ge "$ROLLOUT_PERCENTAGE" ]; then
        warning "Current rollout (${current_rollout}%) is already at or above target (${ROLLOUT_PERCENTAGE}%)"
        read -p "Continue anyway? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            error "Rollout cancelled"
            exit 1
        fi
    fi

    success "Pre-rollout checks passed"
}

capture_baseline_metrics() {
    section "Capturing Baseline Metrics"

    # Setup port forwarding for metrics
    kubectl port-forward -n "$NAMESPACE" svc/dwcp-v3 9090:9090 > /dev/null 2>&1 &
    PORTFWD_METRICS_PID=$!
    sleep 3

    # Capture baseline metrics
    log "Collecting baseline metrics..."

    BASELINE_ERROR_RATE=$(curl -s http://localhost:9090/metrics 2>/dev/null | \
        grep 'dwcp_error_rate' | tail -1 | awk '{print $2}' || echo "0")

    BASELINE_P99_LATENCY=$(curl -s http://localhost:9090/metrics 2>/dev/null | \
        grep 'dwcp_latency_p99_ms' | tail -1 | awk '{print $2}' || echo "0")

    BASELINE_THROUGHPUT=$(curl -s http://localhost:9090/metrics 2>/dev/null | \
        grep 'dwcp_throughput_bytes_total' | tail -1 | awk '{print $2}' || echo "0")

    BASELINE_CPU=$(kubectl top pods -n "$NAMESPACE" -l app=dwcp-v3 --no-headers 2>/dev/null | \
        awk '{sum+=$2} END {print sum}' || echo "0")

    BASELINE_MEMORY=$(kubectl top pods -n "$NAMESPACE" -l app=dwcp-v3 --no-headers 2>/dev/null | \
        awk '{sum+=$3} END {print sum}' || echo "0")

    log "Baseline Error Rate: ${BASELINE_ERROR_RATE}%"
    log "Baseline P99 Latency: ${BASELINE_P99_LATENCY}ms"
    log "Baseline Throughput: ${BASELINE_THROUGHPUT} bytes"
    log "Baseline CPU: ${BASELINE_CPU}"
    log "Baseline Memory: ${BASELINE_MEMORY}"

    success "Baseline metrics captured"
}

update_feature_flag() {
    section "Updating Feature Flag to ${ROLLOUT_PERCENTAGE}%"

    log "Creating ConfigMap patch..."

    # Update feature flag
    kubectl patch configmap dwcp-v3-config -n "$NAMESPACE" --type merge -p \
        "{\"data\":{\"FEATURE_FLAG_V3_ROLLOUT\":\"${ROLLOUT_PERCENTAGE}\"}}"

    if [ $? -eq 0 ]; then
        success "Feature flag updated to ${ROLLOUT_PERCENTAGE}%"
    else
        error "Failed to update feature flag"
        exit 1
    fi

    # Restart pods to pick up new configuration
    log "Rolling restart of DWCP v3 pods..."

    kubectl rollout restart deployment/dwcp-v3 -n "$NAMESPACE"
    kubectl rollout status deployment/dwcp-v3 -n "$NAMESPACE" --timeout=5m

    if [ $? -eq 0 ]; then
        success "Pods restarted successfully"
    else
        error "Pod restart failed"
        trigger_rollback "Pod restart failed"
    fi

    # Wait for pods to stabilize
    log "Waiting for pods to stabilize..."
    sleep 30
}

monitor_health() {
    section "Monitoring Health (${MONITORING_DURATION}s)"

    local start_time=$(date +%s)
    local end_time=$((start_time + MONITORING_DURATION))
    local check_count=0

    while [ $(date +%s) -lt $end_time ]; do
        check_count=$((check_count + 1))
        local elapsed=$(($(date +%s) - start_time))
        local remaining=$((end_time - $(date +%s)))

        log "Health check #${check_count} (${elapsed}s elapsed, ${remaining}s remaining)"

        # Check error rate
        local current_error_rate=$(curl -s http://localhost:9090/metrics 2>/dev/null | \
            grep 'dwcp_error_rate' | tail -1 | awk '{print $2}' || echo "0")

        log "  Error Rate: ${current_error_rate}% (threshold: ${ERROR_THRESHOLD}%)"

        if (( $(echo "$current_error_rate > $ERROR_THRESHOLD" | bc -l) )); then
            error "  ⚠️  Error rate exceeded threshold!"
            ROLLBACK_TRIGGERS+=("Error rate: ${current_error_rate}% > ${ERROR_THRESHOLD}%")
        fi

        # Check P99 latency
        local current_p99=$(curl -s http://localhost:9090/metrics 2>/dev/null | \
            grep 'dwcp_latency_p99_ms' | tail -1 | awk '{print $2}' || echo "0")

        log "  P99 Latency: ${current_p99}ms (threshold: ${LATENCY_THRESHOLD}ms)"

        if (( $(echo "$current_p99 > $LATENCY_THRESHOLD" | bc -l) )); then
            warning "  ⚠️  P99 latency exceeded threshold!"
            ROLLBACK_TRIGGERS+=("P99 latency: ${current_p99}ms > ${LATENCY_THRESHOLD}ms")
        fi

        # Check pod health
        local ready_pods=$(kubectl get deployment dwcp-v3 -n "$NAMESPACE" \
            -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        local desired_pods=$(kubectl get deployment dwcp-v3 -n "$NAMESPACE" \
            -o jsonpath='{.spec.replicas}')

        log "  Pods: ${ready_pods}/${desired_pods} ready"

        if [ "$ready_pods" -lt "$desired_pods" ]; then
            error "  ⚠️  Not all pods are ready!"
            ROLLBACK_TRIGGERS+=("Pods not ready: ${ready_pods}/${desired_pods}")
        fi

        # Check for pod crashes
        local crash_count=$(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 \
            --field-selector=status.phase=Failed --no-headers 2>/dev/null | wc -l)

        if [ "$crash_count" -gt 0 ]; then
            error "  ⚠️  ${crash_count} pods have crashed!"
            ROLLBACK_TRIGGERS+=("${crash_count} pods crashed")
        fi

        # Trigger rollback if too many issues
        if [ ${#ROLLBACK_TRIGGERS[@]} -ge 3 ]; then
            error "Multiple health check failures detected"
            trigger_rollback "Multiple health check failures"
        fi

        # Wait before next check
        if [ $remaining -gt 0 ]; then
            sleep $HEALTH_CHECK_INTERVAL
        fi
    done

    success "Health monitoring completed"
}

validate_rollout_success() {
    section "Validating Rollout Success"

    # Compare final metrics to baseline
    local final_error_rate=$(curl -s http://localhost:9090/metrics 2>/dev/null | \
        grep 'dwcp_error_rate' | tail -1 | awk '{print $2}' || echo "0")

    local final_p99=$(curl -s http://localhost:9090/metrics 2>/dev/null | \
        grep 'dwcp_latency_p99_ms' | tail -1 | awk '{print $2}' || echo "0")

    log "Final Error Rate: ${final_error_rate}% (baseline: ${BASELINE_ERROR_RATE}%)"
    log "Final P99 Latency: ${final_p99}ms (baseline: ${BASELINE_P99_LATENCY}ms)"

    # Check if metrics degraded significantly
    local error_increase=$(echo "$final_error_rate - $BASELINE_ERROR_RATE" | bc -l)
    local latency_increase=$(echo "$final_p99 - $BASELINE_P99_LATENCY" | bc -l)

    if (( $(echo "$error_increase > 0.5" | bc -l) )); then
        warning "Error rate increased by ${error_increase}%"
    fi

    if (( $(echo "$latency_increase > 20" | bc -l) )); then
        warning "P99 latency increased by ${latency_increase}ms"
    fi

    # Final decision
    if [ ${#ROLLBACK_TRIGGERS[@]} -eq 0 ]; then
        success "✅ Rollout validated successfully"
        return 0
    else
        error "⚠️  Rollout validation found ${#ROLLBACK_TRIGGERS[@]} issues"
        return 1
    fi
}

trigger_rollback() {
    local reason="$1"

    section "TRIGGERING AUTOMATIC ROLLBACK"

    error "Rollback reason: $reason"

    if [ "$AUTO_ROLLBACK" != "true" ]; then
        warning "Auto-rollback is disabled. Manual intervention required."
        exit 1
    fi

    log "Rolling back feature flag to 0%..."

    # Revert feature flag
    kubectl patch configmap dwcp-v3-config -n "$NAMESPACE" --type merge -p \
        '{"data":{"FEATURE_FLAG_V3_ROLLOUT":"0"}}'

    # Restart pods
    kubectl rollout restart deployment/dwcp-v3 -n "$NAMESPACE"
    kubectl rollout status deployment/dwcp-v3 -n "$NAMESPACE" --timeout=5m

    # Notify
    if command -v npx &> /dev/null; then
        npx claude-flow@alpha hooks notify \
            --message "⚠️  DWCP v3 rollback triggered: $reason" \
            2>/dev/null || true
    fi

    # Create incident report
    create_incident_report "$reason"

    error "Rollback completed. Please investigate before retrying."
    exit 1
}

create_incident_report() {
    local reason="$1"
    local report_file="/tmp/dwcp-v3-rollback-$(date +%Y%m%d-%H%M%S).md"

    cat > "$report_file" <<EOF
# DWCP v3 Rollback Incident Report

**Timestamp:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Environment:** $ENVIRONMENT
**Rollout Percentage:** ${ROLLOUT_PERCENTAGE}%
**Rollback Reason:** $reason

## Rollback Triggers

$(for trigger in "${ROLLBACK_TRIGGERS[@]}"; do
    echo "- $trigger"
done)

## Baseline Metrics

- Error Rate: ${BASELINE_ERROR_RATE}%
- P99 Latency: ${BASELINE_P99_LATENCY}ms
- Throughput: ${BASELINE_THROUGHPUT} bytes

## Investigation Required

1. Review application logs: \`kubectl logs -n $NAMESPACE -l app=dwcp-v3 --tail=100\`
2. Check pod events: \`kubectl describe pods -n $NAMESPACE -l app=dwcp-v3\`
3. Review metrics dashboard
4. Check for infrastructure issues

## Remediation Steps

1. Identify root cause
2. Fix issue in staging first
3. Re-run staging validation
4. Retry production rollout with fixes

---

*Incident report generated automatically by DWCP v3 rollout script*
EOF

    log "Incident report saved to: $report_file"
}

generate_rollout_report() {
    section "Generating Rollout Report"

    local report_file="/home/kp/novacron/docs/DWCP_V3_PHASE5_ROLLOUT_PHASE1_REPORT.md"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    cat > "$report_file" <<EOF
# DWCP v3 Phase 5: Production Rollout Phase 1 Report

**Generated:** $timestamp
**Environment:** $ENVIRONMENT
**Rollout Percentage:** ${ROLLOUT_PERCENTAGE}%
**Monitoring Duration:** ${MONITORING_DURATION}s

## Rollout Status

$(if [ ${#ROLLBACK_TRIGGERS[@]} -eq 0 ]; then
    echo "**Status:** ✅ SUCCESS"
    echo ""
    echo "Phase 1 rollout completed successfully. ${ROLLOUT_PERCENTAGE}% of traffic now using DWCP v3."
else
    echo "**Status:** ⚠️  ROLLBACK TRIGGERED"
    echo ""
    echo "Rollout was automatically rolled back due to health check failures."
fi)

## Baseline Metrics

| Metric | Baseline | Final | Delta |
|--------|----------|-------|-------|
| Error Rate | ${BASELINE_ERROR_RATE}% | ${final_error_rate}% | ${error_increase}% |
| P99 Latency | ${BASELINE_P99_LATENCY}ms | ${final_p99}ms | ${latency_increase}ms |
| Throughput | ${BASELINE_THROUGHPUT} | - | - |

## Health Check Results

- Total Checks: $check_count
- Monitoring Duration: ${MONITORING_DURATION}s
- Check Interval: ${HEALTH_CHECK_INTERVAL}s

## Rollback Triggers

$(if [ ${#ROLLBACK_TRIGGERS[@]} -eq 0 ]; then
    echo "None - rollout successful"
else
    for trigger in "${ROLLBACK_TRIGGERS[@]}"; do
        echo "- $trigger"
    done
fi)

## Next Steps

$(if [ ${#ROLLBACK_TRIGGERS[@]} -eq 0 ]; then
    echo "1. Continue monitoring for 24 hours"
    echo "2. Review metrics and user feedback"
    echo "3. Proceed to Phase 2 (50% rollout) if stable"
else
    echo "1. Investigate rollback triggers"
    echo "2. Fix identified issues"
    echo "3. Re-validate in staging"
    echo "4. Retry Phase 1 rollout"
fi)

## Deployment Information

- Kubernetes Context: $(kubectl config current-context)
- Namespace: $NAMESPACE
- Pods Running: $(kubectl get pods -n "$NAMESPACE" -l app=dwcp-v3 --no-headers | wc -l)
- Ready Replicas: $(kubectl get deployment dwcp-v3 -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')

---

*Report generated by DWCP v3 Phase 1 rollout script*
EOF

    success "Rollout report saved to: $report_file"
    cat "$report_file"
}

cleanup() {
    log "Cleaning up..."

    if [ -n "${PORTFWD_METRICS_PID:-}" ]; then
        kill $PORTFWD_METRICS_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

main() {
    log "===== DWCP v3 Phase 5: Production Rollout Phase 1 (${ROLLOUT_PERCENTAGE}%) ====="
    log "Environment: $ENVIRONMENT"
    log "Auto-rollback: $AUTO_ROLLBACK"
    log ""

    # Final confirmation
    warning "You are about to rollout DWCP v3 to ${ROLLOUT_PERCENTAGE}% of production traffic"
    read -p "Type 'ROLLOUT' to confirm: " confirmation

    if [ "$confirmation" != "ROLLOUT" ]; then
        error "Rollout cancelled"
        exit 1
    fi

    check_prerequisites
    capture_baseline_metrics
    update_feature_flag
    monitor_health

    if validate_rollout_success; then
        generate_rollout_report

        # Notify success
        if command -v npx &> /dev/null; then
            npx claude-flow@alpha hooks notify \
                --message "✅ DWCP v3 Phase 1 rollout successful (${ROLLOUT_PERCENTAGE}%)" \
                2>/dev/null || true

            npx claude-flow@alpha hooks post-task \
                --task-id "phase5-production-rollout-phase1" \
                2>/dev/null || true
        fi

        echo ""
        success "===== Phase 1 Rollout Complete ====="
        log "DWCP v3 is now serving ${ROLLOUT_PERCENTAGE}% of production traffic"
        log "Continue monitoring for 24 hours before Phase 2 (50% rollout)"
    else
        trigger_rollback "Validation failed"
    fi
}

# Run main function
main "$@"
