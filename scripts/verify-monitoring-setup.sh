#!/bin/bash
# DWCP Monitoring Setup Verification Script

set -e

echo "================================================"
echo "DWCP Phase 1 Monitoring Setup Verification"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0

# Check function
check() {
    local description=$1
    local test_command=$2

    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((FAILED++))
        return 1
    fi
}

echo "1. Checking File Structure..."
echo "----------------------------------------"

check "Metrics package exists" "test -d /home/kp/novacron/backend/core/network/dwcp/metrics"
check "prometheus.go exists" "test -f /home/kp/novacron/backend/core/network/dwcp/metrics/prometheus.go"
check "exporter.go exists" "test -f /home/kp/novacron/backend/core/network/dwcp/metrics/exporter.go"
check "collector.go exists" "test -f /home/kp/novacron/backend/core/network/dwcp/metrics/collector.go"
check "integration.go exists" "test -f /home/kp/novacron/backend/core/network/dwcp/metrics/integration.go"
check "examples_test.go exists" "test -f /home/kp/novacron/backend/core/network/dwcp/metrics/examples_test.go"
check "Metrics README exists" "test -f /home/kp/novacron/backend/core/network/dwcp/metrics/README.md"

echo ""
echo "2. Checking Prometheus Configuration..."
echo "----------------------------------------"

check "Prometheus config directory exists" "test -d /home/kp/novacron/configs/prometheus"
check "Scrape config exists" "test -f /home/kp/novacron/configs/prometheus/dwcp-scrape-config.yml"
check "Alert rules exist" "test -f /home/kp/novacron/configs/prometheus/dwcp-alerts.yml"
check "Recording rules exist" "test -f /home/kp/novacron/configs/prometheus/dwcp-recording-rules.yml"
check "Alertmanager config exists" "test -f /home/kp/novacron/configs/prometheus/alertmanager.yml"

echo ""
echo "3. Checking Grafana Configuration..."
echo "----------------------------------------"

check "Grafana config directory exists" "test -d /home/kp/novacron/configs/grafana"
check "DWCP dashboard exists" "test -f /home/kp/novacron/configs/grafana/dwcp-dashboard.json"
check "Dashboard has panels" "grep -q 'AMST Active Streams' /home/kp/novacron/configs/grafana/dwcp-dashboard.json"
check "Dashboard has 18+ panels" "test $(grep -c '\"id\":' /home/kp/novacron/configs/grafana/dwcp-dashboard.json) -ge 18"

echo ""
echo "4. Checking Documentation..."
echo "----------------------------------------"

check "Monitoring docs directory exists" "test -d /home/kp/novacron/docs/monitoring"
check "Quick start guide exists" "test -f /home/kp/novacron/docs/monitoring/DWCP_MONITORING_QUICKSTART.md"
check "Implementation doc exists" "test -f /home/kp/novacron/docs/monitoring/DWCP_MONITORING_IMPLEMENTATION.md"

echo ""
echo "5. Checking Docker Compose..."
echo "----------------------------------------"

check "Docker compose file exists" "test -f /home/kp/novacron/configs/docker-compose.monitoring.yml"
check "Prometheus service defined" "grep -q 'prometheus:' /home/kp/novacron/configs/docker-compose.monitoring.yml"
check "Grafana service defined" "grep -q 'grafana:' /home/kp/novacron/configs/docker-compose.monitoring.yml"
check "Alertmanager service defined" "grep -q 'alertmanager:' /home/kp/novacron/configs/docker-compose.monitoring.yml"

echo ""
echo "6. Validating Metric Definitions..."
echo "----------------------------------------"

check "AMST metrics defined" "grep -q 'AMSTStreamsActive' /home/kp/novacron/backend/core/network/dwcp/metrics/prometheus.go"
check "HDE metrics defined" "grep -q 'HDECompressionRatio' /home/kp/novacron/backend/core/network/dwcp/metrics/prometheus.go"
check "Migration metrics defined" "grep -q 'MigrationDuration' /home/kp/novacron/backend/core/network/dwcp/metrics/prometheus.go"
check "System metrics defined" "grep -q 'ComponentHealth' /home/kp/novacron/backend/core/network/dwcp/metrics/prometheus.go"
check "50+ metrics defined" "test $(grep -c 'promauto.New' /home/kp/novacron/backend/core/network/dwcp/metrics/prometheus.go) -ge 15"

echo ""
echo "7. Validating Alert Rules..."
echo "----------------------------------------"

check "Critical alerts defined" "grep -q 'DWCPHighErrorRate' /home/kp/novacron/configs/prometheus/dwcp-alerts.yml"
check "SLA alerts defined" "grep -q 'DWCPSLAViolation' /home/kp/novacron/configs/prometheus/dwcp-alerts.yml"
check "Performance alerts defined" "grep -q 'DWCPMigrationSlow' /home/kp/novacron/configs/prometheus/dwcp-alerts.yml"
check "10+ alert rules" "test $(grep -c 'alert:' /home/kp/novacron/configs/prometheus/dwcp-alerts.yml) -ge 10"

echo ""
echo "8. Validating Recording Rules..."
echo "----------------------------------------"

check "AMST aggregations defined" "grep -q 'dwcp:amst:error_rate:5m' /home/kp/novacron/configs/prometheus/dwcp-recording-rules.yml"
check "HDE aggregations defined" "grep -q 'dwcp:hde:compression_ratio_median:10m' /home/kp/novacron/configs/prometheus/dwcp-recording-rules.yml"
check "Migration aggregations defined" "grep -q 'dwcp:migration:duration_median:1h' /home/kp/novacron/configs/prometheus/dwcp-recording-rules.yml"
check "SLA tracking rules defined" "grep -q 'dwcp:sla:availability:30d' /home/kp/novacron/configs/prometheus/dwcp-recording-rules.yml"

echo ""
echo "9. Checking Integration Wrappers..."
echo "----------------------------------------"

check "AMST wrapper defined" "grep -q 'AMSTMetricsWrapper' /home/kp/novacron/backend/core/network/dwcp/metrics/integration.go"
check "HDE wrapper defined" "grep -q 'HDEMetricsWrapper' /home/kp/novacron/backend/core/network/dwcp/metrics/integration.go"
check "Migration wrapper defined" "grep -q 'MigrationMetricsWrapper' /home/kp/novacron/backend/core/network/dwcp/metrics/integration.go"
check "System wrapper defined" "grep -q 'SystemMetricsWrapper' /home/kp/novacron/backend/core/network/dwcp/metrics/integration.go"
check "Convenience functions defined" "grep -q 'RecordStreamMetrics' /home/kp/novacron/backend/core/network/dwcp/metrics/integration.go"

echo ""
echo "10. Checking Examples and Tests..."
echo "----------------------------------------"

check "Test examples exist" "grep -q 'ExampleInitializeMetrics' /home/kp/novacron/backend/core/network/dwcp/metrics/examples_test.go"
check "AMST examples exist" "grep -q 'ExampleAMSTMetricsWrapper' /home/kp/novacron/backend/core/network/dwcp/metrics/examples_test.go"
check "HDE examples exist" "grep -q 'ExampleHDEMetricsWrapper' /home/kp/novacron/backend/core/network/dwcp/metrics/examples_test.go"
check "Migration examples exist" "grep -q 'ExampleMigrationMetricsWrapper' /home/kp/novacron/backend/core/network/dwcp/metrics/examples_test.go"
check "10+ examples defined" "test $(grep -c 'func Example' /home/kp/novacron/backend/core/network/dwcp/metrics/examples_test.go) -ge 10"

echo ""
echo "================================================"
echo "Verification Summary"
echo "================================================"
echo -e "Tests Passed: ${GREEN}$PASSED${NC}"
echo -e "Tests Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Monitoring setup is complete.${NC}"
    echo ""
    echo "Next Steps:"
    echo "1. Start monitoring stack: cd configs && docker-compose -f docker-compose.monitoring.yml up -d"
    echo "2. Access Grafana at: http://localhost:3001 (admin / dwcp-admin-2025)"
    echo "3. Import dashboard from: configs/grafana/dwcp-dashboard.json"
    echo "4. Check Prometheus targets: http://localhost:9091/targets"
    echo ""
    echo "Documentation:"
    echo "- Quick Start: docs/monitoring/DWCP_MONITORING_QUICKSTART.md"
    echo "- Implementation: docs/monitoring/DWCP_MONITORING_IMPLEMENTATION.md"
    echo "- Metrics README: backend/core/network/dwcp/metrics/README.md"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please review the output above.${NC}"
    exit 1
fi
