#!/bin/bash
# DWCP v3 Production Monitoring Validation Script
# Validates all monitoring components are functioning correctly

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MONITORING_NAMESPACE="dwcp-v3-monitoring"
PASSED=0
FAILED=0

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

echo ""
echo "=========================================="
echo "  DWCP v3 Monitoring Validation"
echo "=========================================="
echo ""

# Test 1: Prometheus is scraping metrics
log_test "Checking Prometheus is scraping DWCP v3 metrics..."
TARGETS=$(kubectl exec -n $MONITORING_NAMESPACE prometheus-prometheus-operator-kube-prom-prometheus-0 -- \
    wget -qO- http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | length')
if [ "$TARGETS" -gt 0 ]; then
    log_pass "Prometheus has $TARGETS active targets"
else
    log_fail "Prometheus has no active targets"
fi

# Test 2: DWCP v3 metrics are being collected
log_test "Checking DWCP v3 metrics collection..."
METRIC_COUNT=$(kubectl exec -n $MONITORING_NAMESPACE prometheus-prometheus-operator-kube-prom-prometheus-0 -- \
    wget -qO- 'http://localhost:9090/api/v1/query?query=dwcp_v3_mode_latency_seconds_bucket' | jq -r '.data.result | length')
if [ "$METRIC_COUNT" -gt 0 ]; then
    log_pass "DWCP v3 metrics are being collected ($METRIC_COUNT series)"
else
    log_fail "DWCP v3 metrics are not being collected"
fi

# Test 3: Grafana is accessible
log_test "Checking Grafana accessibility..."
GRAFANA_STATUS=$(kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].status.phase}')
if [ "$GRAFANA_STATUS" = "Running" ]; then
    log_pass "Grafana is running"
else
    log_fail "Grafana is not running (status: $GRAFANA_STATUS)"
fi

# Test 4: Dashboards are loaded
log_test "Checking Grafana dashboards..."
DASHBOARD_COUNT=$(kubectl get configmaps -n $MONITORING_NAMESPACE -l grafana_dashboard=1 -o name | wc -l)
if [ "$DASHBOARD_COUNT" -ge 3 ]; then
    log_pass "All dashboards loaded ($DASHBOARD_COUNT dashboards)"
else
    log_fail "Missing dashboards (found $DASHBOARD_COUNT, expected 3)"
fi

# Test 5: Alertmanager is running
log_test "Checking Alertmanager..."
ALERTMANAGER_REPLICAS=$(kubectl get statefulset -n $MONITORING_NAMESPACE alertmanager-prometheus-operator-kube-prom-alertmanager -o jsonpath='{.status.readyReplicas}')
if [ "$ALERTMANAGER_REPLICAS" -ge 2 ]; then
    log_pass "Alertmanager has $ALERTMANAGER_REPLICAS replicas ready"
else
    log_fail "Alertmanager not ready ($ALERTMANAGER_REPLICAS replicas)"
fi

# Test 6: Alert rules are loaded
log_test "Checking alert rules..."
ALERT_RULES=$(kubectl exec -n $MONITORING_NAMESPACE prometheus-prometheus-operator-kube-prom-prometheus-0 -- \
    wget -qO- http://localhost:9090/api/v1/rules | jq -r '.data.groups | length')
if [ "$ALERT_RULES" -gt 0 ]; then
    log_pass "Alert rules loaded ($ALERT_RULES rule groups)"
else
    log_fail "No alert rules loaded"
fi

# Test 7: OpenTelemetry Collector is running
log_test "Checking OpenTelemetry Collector..."
OTEL_STATUS=$(kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=opentelemetry-collector -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "NotFound")
if [ "$OTEL_STATUS" = "Running" ]; then
    log_pass "OpenTelemetry Collector is running"
else
    log_fail "OpenTelemetry Collector is not running (status: $OTEL_STATUS)"
fi

# Test 8: Jaeger is accessible
log_test "Checking Jaeger..."
JAEGER_STATUS=$(kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=jaeger -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "NotFound")
if [ "$JAEGER_STATUS" = "Running" ]; then
    log_pass "Jaeger is running"
else
    log_fail "Jaeger is not running (status: $JAEGER_STATUS)"
fi

# Test 9: Loki is running
log_test "Checking Loki..."
LOKI_PODS=$(kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=loki -o name | wc -l)
if [ "$LOKI_PODS" -gt 0 ]; then
    log_pass "Loki is running ($LOKI_PODS pods)"
else
    log_fail "Loki is not running"
fi

# Test 10: Promtail is collecting logs
log_test "Checking Promtail log collection..."
PROMTAIL_PODS=$(kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=promtail -o name | wc -l)
if [ "$PROMTAIL_PODS" -gt 0 ]; then
    log_pass "Promtail is running ($PROMTAIL_PODS pods)"
else
    log_fail "Promtail is not running"
fi

# Test 11: SLA recording rules are working
log_test "Checking SLA recording rules..."
SLA_METRICS=$(kubectl exec -n $MONITORING_NAMESPACE prometheus-prometheus-operator-kube-prom-prometheus-0 -- \
    wget -qO- 'http://localhost:9090/api/v1/query?query=sla:availability:5m' | jq -r '.data.result | length')
if [ "$SLA_METRICS" -gt 0 ]; then
    log_pass "SLA recording rules are working"
else
    log_fail "SLA recording rules not found"
fi

# Test 12: Thanos is running (if deployed)
log_test "Checking Thanos (optional)..."
THANOS_PODS=$(kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=thanos -o name 2>/dev/null | wc -l || echo "0")
if [ "$THANOS_PODS" -gt 0 ]; then
    log_pass "Thanos is running ($THANOS_PODS pods)"
else
    log_fail "Thanos is not deployed (optional component)"
fi

echo ""
echo "=========================================="
echo "  Validation Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All validation tests passed!${NC}"
    echo ""
    echo "Monitoring stack is ready for production."
    echo "Next steps:"
    echo "1. Configure alert notification channels"
    echo "2. Test alert firing and routing"
    echo "3. Verify dashboard refresh rates"
    echo "4. Set up log retention policies"
    echo "5. Configure backup schedules"
    exit 0
else
    echo -e "${RED}Some validation tests failed.${NC}"
    echo ""
    echo "Please review failed tests and check logs:"
    echo "kubectl logs -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=<component>"
    exit 1
fi
