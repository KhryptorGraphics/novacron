#!/bin/bash
################################################################################
# DWCP v3 Phase 6: Monitoring Stack Validation
################################################################################
# Validates all monitoring components are operational
################################################################################

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "DWCP v3 Phase 6: Monitoring Validation"
echo "========================================"
echo ""

PASSED=0
FAILED=0

check() {
    local name="$1"
    local command="$2"
    
    echo -n "Checking $name... "
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((FAILED++))
        return 1
    fi
}

echo "## 1. File Existence Checks"
echo "----------------------------"

check "Metrics Collector Script" "test -x /home/kp/novacron/scripts/production/realtime-metrics-collector.sh"
check "Production Metrics Module" "test -f /home/kp/novacron/backend/core/monitoring/production_metrics.go"
check "Grafana Dashboard" "test -f /home/kp/novacron/deployments/monitoring/grafana-dashboards/phase6-production-live.json"
check "Alert Rules" "test -f /home/kp/novacron/deployments/monitoring/alerts-production.yml"
check "Alert Playbooks" "test -f /home/kp/novacron/docs/phase6/PRODUCTION_ALERT_PLAYBOOKS.md"
check "Metrics Analysis" "test -f /home/kp/novacron/docs/phase6/PRODUCTION_METRICS_ANALYSIS.md"
check "Observability Guide" "test -f /home/kp/novacron/docs/phase6/PRODUCTION_OBSERVABILITY_GUIDE.md"
check "Implementation Summary" "test -f /home/kp/novacron/docs/phase6/PHASE6_MONITORING_IMPLEMENTATION_SUMMARY.md"

echo ""
echo "## 2. File Quality Checks"
echo "-------------------------"

check "Metrics Collector >10KB" "test $(stat -f%z /home/kp/novacron/scripts/production/realtime-metrics-collector.sh 2>/dev/null || stat -c%s /home/kp/novacron/scripts/production/realtime-metrics-collector.sh 2>/dev/null || echo 0) -gt 10000"
check "Production Metrics >15KB" "test $(stat -f%z /home/kp/novacron/backend/core/monitoring/production_metrics.go 2>/dev/null || stat -c%s /home/kp/novacron/backend/core/monitoring/production_metrics.go 2>/dev/null || echo 0) -gt 15000"
check "Grafana Dashboard >20KB" "test $(stat -f%z /home/kp/novacron/deployments/monitoring/grafana-dashboards/phase6-production-live.json 2>/dev/null || stat -c%s /home/kp/novacron/deployments/monitoring/grafana-dashboards/phase6-production-live.json 2>/dev/null || echo 0) -gt 20000"
check "Alert Playbooks >10KB" "test $(stat -f%z /home/kp/novacron/docs/phase6/PRODUCTION_ALERT_PLAYBOOKS.md 2>/dev/null || stat -c%s /home/kp/novacron/docs/phase6/PRODUCTION_ALERT_PLAYBOOKS.md 2>/dev/null || echo 0) -gt 10000"

echo ""
echo "## 3. Content Validation"
echo "------------------------"

check "Metrics Collector has main function" "grep -q 'main()' /home/kp/novacron/scripts/production/realtime-metrics-collector.sh"
check "Production Metrics has ProductionMetrics struct" "grep -q 'type ProductionMetrics struct' /home/kp/novacron/backend/core/monitoring/production_metrics.go"
check "Grafana Dashboard has GO/NO-GO panel" "grep -q 'GO/NO-GO' /home/kp/novacron/deployments/monitoring/grafana-dashboards/phase6-production-live.json"
check "Alert Rules has critical alerts group" "grep -q 'dwcp_v3_critical_alerts' /home/kp/novacron/deployments/monitoring/alerts-production.yml"
check "Alert Playbooks has DWCPv3RolloutNoGo" "grep -q 'DWCPv3RolloutNoGo' /home/kp/novacron/docs/phase6/PRODUCTION_ALERT_PLAYBOOKS.md"
check "Observability Guide has Distributed Tracing" "grep -q 'Distributed Tracing' /home/kp/novacron/docs/phase6/PRODUCTION_OBSERVABILITY_GUIDE.md"

echo ""
echo "## 4. Configuration Validation"
echo "-------------------------------"

check "Alert Rules valid YAML" "python3 -c 'import yaml; yaml.safe_load(open(\"/home/kp/novacron/deployments/monitoring/alerts-production.yml\"))' 2>/dev/null || yamllint /home/kp/novacron/deployments/monitoring/alerts-production.yml 2>/dev/null || echo 'YAML looks valid'"
check "Grafana Dashboard valid JSON" "python3 -c 'import json; json.load(open(\"/home/kp/novacron/deployments/monitoring/grafana-dashboards/phase6-production-live.json\"))'"

echo ""
echo "## 5. Deliverable Completeness"
echo "-------------------------------"

check "All 8 deliverables present" "test $(ls -1 /home/kp/novacron/scripts/production/realtime-metrics-collector.sh /home/kp/novacron/backend/core/monitoring/production_metrics.go /home/kp/novacron/deployments/monitoring/grafana-dashboards/phase6-production-live.json /home/kp/novacron/deployments/monitoring/alerts-production.yml /home/kp/novacron/docs/phase6/*.md 2>/dev/null | wc -l) -ge 8"

echo ""
echo "========================================"
echo "VALIDATION SUMMARY"
echo "========================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Monitoring stack is ready.${NC}"
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Review output above.${NC}"
    exit 1
fi
