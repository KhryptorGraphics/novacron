#!/bin/bash
# Automated Rollback Script for NovaCron
# Triggers rollback on health check failures or high error rates

set -e

ROLLBACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
NAMESPACE="${NAMESPACE:-novacron}"
ROLLBACK_VERSION="${1:-previous}"
REASON="${2:-Manual rollback}"

echo -e "${RED}========================================${NC}"
echo -e "${RED}  NovaCron Automated Rollback${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo -e "Namespace: ${GREEN}$NAMESPACE${NC}"
echo -e "Rolling back to: ${GREEN}$ROLLBACK_VERSION${NC}"
echo -e "Reason: ${YELLOW}$REASON${NC}"
echo ""

# Confirm rollback (skip if automated)
if [ "$AUTOMATED" != "true" ]; then
    read -p "Are you sure you want to rollback? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Rollback cancelled"
        exit 0
    fi
fi

echo "Starting rollback..."

# Step 1: Rollback deployments
echo "Step 1: Rolling back deployments..."

if [ "$ROLLBACK_VERSION" == "previous" ]; then
    kubectl rollout undo deployment/novacron-api -n $NAMESPACE
    kubectl rollout undo deployment/novacron-core -n $NAMESPACE
    kubectl rollout undo deployment/novacron-frontend -n $NAMESPACE
else
    kubectl rollout undo deployment/novacron-api -n $NAMESPACE --to-revision=$ROLLBACK_VERSION
    kubectl rollout undo deployment/novacron-core -n $NAMESPACE --to-revision=$ROLLBACK_VERSION
    kubectl rollout undo deployment/novacron-frontend -n $NAMESPACE --to-revision=$ROLLBACK_VERSION
fi

# Wait for rollback to complete
echo "Waiting for rollback to complete..."
kubectl rollout status deployment/novacron-api -n $NAMESPACE --timeout=5m
kubectl rollout status deployment/novacron-core -n $NAMESPACE --timeout=5m
kubectl rollout status deployment/novacron-frontend -n $NAMESPACE --timeout=5m

echo -e "${GREEN}✓ Rollback completed${NC}"

# Step 2: Health checks
echo ""
echo "Step 2: Running health checks..."

API_ENDPOINT=$(kubectl get service -n $NAMESPACE novacron-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

HEALTHY=false
for i in {1..10}; do
    echo "Health check attempt $i/10..."

    if curl -f -s "http://$API_ENDPOINT:8080/health" > /dev/null; then
        HEALTHY=true
        echo -e "${GREEN}✓ Health check passed${NC}"
        break
    fi

    if [ $i -lt 10 ]; then
        echo "Waiting for service to be healthy..."
        sleep 10
    fi
done

if [ "$HEALTHY" != "true" ]; then
    echo -e "${RED}✗ Health checks failed after rollback${NC}"
    echo "Manual intervention required"
    exit 1
fi

# Step 3: Verify metrics
echo ""
echo "Step 3: Verifying metrics..."
sleep 30  # Wait for metrics to stabilize

ERROR_RATE=$(curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(novacron_api_errors_total[1m]))/sum(rate(novacron_api_requests_total[1m]))" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

echo "Error rate: $ERROR_RATE"

if (( $(echo "$ERROR_RATE < 0.01" | bc -l) )); then
    echo -e "${GREEN}✓ Error rate is acceptable${NC}"
else
    echo -e "${YELLOW}⚠ Error rate is still elevated${NC}"
fi

# Step 4: Record rollback event
echo ""
echo "Step 4: Recording rollback event..."

kubectl annotate deployment novacron-api -n $NAMESPACE \
  "rollback.novacron.io/date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "rollback.novacron.io/reason=$REASON" \
  "rollback.novacron.io/version=$ROLLBACK_VERSION"

echo -e "${GREEN}✓ Rollback event recorded${NC}"

# Success
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Rollback Completed Successfully${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "System has been rolled back to: $ROLLBACK_VERSION"
echo "Reason: $REASON"
echo ""
echo "Next steps:"
echo "1. Investigate the issue that caused the rollback"
echo "2. Fix the issue in the codebase"
echo "3. Test thoroughly before next deployment"
echo "4. Review rollback event: kubectl describe deployment/novacron-api -n $NAMESPACE"
echo ""
