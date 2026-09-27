#!/bin/bash
# Rollback script for NovaCron on Kubernetes.
#
# Rolls the `novacron-api` and `novacron-frontend` Deployments (the only
# application Deployments in deployment/kubernetes) back to the previous
# revision, or to an explicit revision number, waits for the rollouts and
# verifies the result with deployment/smoke-tests.sh.
#
# Usage: auto-rollback.sh [previous|<revision>] [reason]
#
# Environment:
#   NAMESPACE        target namespace (default novacron)
#   AUTOMATED        "true" skips the interactive confirmation
#   ROLLOUT_TIMEOUT  kubectl rollout status timeout (default 5m)
#   API_URL / FRONTEND_URL  passed through to deployment/smoke-tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SMOKE_TESTS="$PROJECT_ROOT/deployment/smoke-tests.sh"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

NAMESPACE="${NAMESPACE:-novacron}"
ROLLBACK_VERSION="${1:-previous}"
REASON="${2:-Manual rollback}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-5m}"
DEPLOYMENTS="novacron-api novacron-frontend"

echo -e "${RED}========================================${NC}"
echo -e "${RED}  NovaCron Rollback${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo -e "Namespace: ${GREEN}$NAMESPACE${NC}"
echo -e "Rolling back to: ${GREEN}$ROLLBACK_VERSION${NC}"
echo -e "Reason: ${YELLOW}$REASON${NC}"
echo ""

if [ "${AUTOMATED:-}" != "true" ]; then
    read -r -p "Proceed with rollback? (yes/no): " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        echo "Rollback cancelled"
        exit 0
    fi
fi

# Step 1: roll back
echo "Step 1: Rolling back deployments..."
for deployment in $DEPLOYMENTS; do
    if [ "$ROLLBACK_VERSION" == "previous" ]; then
        kubectl rollout undo "deployment/$deployment" -n "$NAMESPACE"
    else
        kubectl rollout undo "deployment/$deployment" -n "$NAMESPACE" --to-revision="$ROLLBACK_VERSION"
    fi
done

echo "Waiting for rollback to complete..."
for deployment in $DEPLOYMENTS; do
    kubectl rollout status "deployment/$deployment" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
done
echo -e "${GREEN}✓ Rollback completed${NC}"

# Step 2: verify with the same smoke tests a deployment must pass
echo ""
echo "Step 2: Running smoke tests..."
if ! "$SMOKE_TESTS"; then
    echo -e "${RED}✗ Smoke tests failed after rollback — investigate immediately${NC}"
    exit 1
fi

# Step 3: record the rollback on the api Deployment
echo ""
echo "Step 3: Recording rollback event..."
kubectl annotate deployment novacron-api -n "$NAMESPACE" --overwrite \
    "rollback.novacron.io/date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "rollback.novacron.io/reason=$REASON" \
    "rollback.novacron.io/version=$ROLLBACK_VERSION"
echo -e "${GREEN}✓ Rollback event recorded${NC}"

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
echo "4. Review history: kubectl rollout history deployment/novacron-api -n $NAMESPACE"
