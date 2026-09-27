#!/bin/bash
# Production deployment for NovaCron on Kubernetes.
#
# The manifests in deployment/kubernetes define ONE `novacron-api` and ONE
# `novacron-frontend` Deployment (RollingUpdate, maxSurge=1, maxUnavailable=0),
# with no `version` label on any pod, so there are no blue/green Deployment
# pairs to flip a Service selector between. Zero-downtime and instant
# rollback come from the Deployment rollout itself:
#
#   kubectl set image  -> kubectl rollout status --timeout
#   -> deployment/smoke-tests.sh -> kubectl rollout undo on failure
#
# Environment:
#   NAMESPACE        target namespace (default novacron)
#   IMAGE_TAG        tag to deploy (default latest)
#   API_IMAGE        api-server image repository (default novacron/api-server)
#   MIGRATE_IMAGE    migrate-tool image repository (default novacron/migrate)
#   FRONTEND_IMAGE   frontend image repository (default novacron/frontend)
#   ROLLOUT_TIMEOUT  kubectl rollout status timeout (default 10m)
#   API_URL / FRONTEND_URL  passed through to deployment/smoke-tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SMOKE_TESTS="$PROJECT_ROOT/deployment/smoke-tests.sh"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

NAMESPACE="${NAMESPACE:-novacron}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
API_IMAGE="${API_IMAGE:-novacron/api-server}"
MIGRATE_IMAGE="${MIGRATE_IMAGE:-novacron/migrate}"
FRONTEND_IMAGE="${FRONTEND_IMAGE:-novacron/frontend}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10m}"

# Container names as declared in deployment/kubernetes/deployments.yaml.
API_DEPLOYMENT="novacron-api"
API_CONTAINER="api-server"
FRONTEND_DEPLOYMENT="novacron-frontend"
FRONTEND_CONTAINER="frontend"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NovaCron Production Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Namespace: ${GREEN}$NAMESPACE${NC}"
echo -e "API image: ${GREEN}$API_IMAGE:$IMAGE_TAG${NC}"
echo -e "Frontend image: ${GREEN}$FRONTEND_IMAGE:$IMAGE_TAG${NC}"
echo ""

rollback() {
    echo -e "${RED}Rolling back $API_DEPLOYMENT and $FRONTEND_DEPLOYMENT...${NC}"
    kubectl rollout undo "deployment/$API_DEPLOYMENT" -n "$NAMESPACE" || true
    kubectl rollout undo "deployment/$FRONTEND_DEPLOYMENT" -n "$NAMESPACE" || true
    kubectl rollout status "deployment/$API_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT" || true
    kubectl rollout status "deployment/$FRONTEND_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT" || true
    echo -e "${YELLOW}Rolled back to the previous revision${NC}"
}

# Step 1: roll the new images out
echo -e "${YELLOW}Step 1: Updating images...${NC}"
# `migration` is the schema initContainer (kubectl set image matches init
# containers by name too), so the migrate tool rolls with the api-server.
kubectl set image "deployment/$API_DEPLOYMENT" \
    "$API_CONTAINER=$API_IMAGE:$IMAGE_TAG" "migration=$MIGRATE_IMAGE:$IMAGE_TAG" -n "$NAMESPACE"
kubectl set image "deployment/$FRONTEND_DEPLOYMENT" \
    "$FRONTEND_CONTAINER=$FRONTEND_IMAGE:$IMAGE_TAG" -n "$NAMESPACE"
kubectl annotate "deployment/$API_DEPLOYMENT" -n "$NAMESPACE" --overwrite \
    "kubernetes.io/change-cause=deploy $IMAGE_TAG ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
kubectl annotate "deployment/$FRONTEND_DEPLOYMENT" -n "$NAMESPACE" --overwrite \
    "kubernetes.io/change-cause=deploy $IMAGE_TAG ($(date -u +%Y-%m-%dT%H:%M:%SZ))"

# Step 2: wait for the rollouts (readiness probes gate every replica)
echo -e "${YELLOW}Step 2: Waiting for rollouts...${NC}"
if ! kubectl rollout status "deployment/$API_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"; then
    echo -e "${RED}✗ $API_DEPLOYMENT rollout failed${NC}"
    rollback
    exit 1
fi
if ! kubectl rollout status "deployment/$FRONTEND_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"; then
    echo -e "${RED}✗ $FRONTEND_DEPLOYMENT rollout failed${NC}"
    rollback
    exit 1
fi
echo -e "${GREEN}✓ Rollouts complete${NC}"
echo ""

# Step 3: smoke tests against the live endpoints
echo -e "${YELLOW}Step 3: Smoke tests...${NC}"
if ! "$SMOKE_TESTS"; then
    echo -e "${RED}✗ Smoke tests failed${NC}"
    rollback
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment of $IMAGE_TAG successful${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Rollback command (if needed):"
echo "  kubectl rollout undo deployment/$API_DEPLOYMENT deployment/$FRONTEND_DEPLOYMENT -n $NAMESPACE"
