#!/bin/bash
# Canary deployment for the NovaCron api-server on Kubernetes.
#
# There is no service mesh, no canary Deployment and no ingress canary
# annotation in deployment/kubernetes, and the api-server exports no
# Prometheus error-rate/latency metrics, so this canary is built on the
# Deployment rollout itself:
#
#   1. set the new image with maxSurge=1,maxUnavailable=0
#   2. pause the rollout as soon as the first new pod is Ready
#   3. smoke-test that pod directly via port-forward (deployment/smoke-tests.sh)
#   4. resume and wait for the full rollout, or undo on failure
#
# Environment:
#   NAMESPACE        target namespace (default novacron)
#   IMAGE_TAG        tag to deploy (default latest)
#   API_IMAGE        api-server image repository (default novacron/api-server)
#   MIGRATE_IMAGE    migrate-tool image repository (default novacron/migrate)
#   ROLLOUT_TIMEOUT  kubectl rollout status timeout (default 10m)
#   CANARY_PORT      local port for the port-forward (default 18090)

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
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-10m}"
CANARY_PORT="${CANARY_PORT:-18090}"

DEPLOYMENT="novacron-api"
CONTAINER="api-server"
POD_SELECTOR="app.kubernetes.io/name=novacron,app.kubernetes.io/component=api"

PF_PID=""
cleanup() {
    if [ -n "$PF_PID" ]; then
        kill "$PF_PID" 2>/dev/null || true
        wait "$PF_PID" 2>/dev/null || true
        PF_PID=""
    fi
}
trap cleanup EXIT

abort() {
    echo -e "${RED}$1${NC}"
    cleanup
    kubectl rollout resume "deployment/$DEPLOYMENT" -n "$NAMESPACE" 2>/dev/null || true
    kubectl rollout undo "deployment/$DEPLOYMENT" -n "$NAMESPACE"
    kubectl rollout status "deployment/$DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT" || true
    echo -e "${YELLOW}Rolled back to the previous revision${NC}"
    exit 1
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NovaCron Canary Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Namespace: ${GREEN}$NAMESPACE${NC}"
echo -e "API image: ${GREEN}$API_IMAGE:$IMAGE_TAG${NC}"
echo ""

# Step 1: one new pod at a time, never below the current replica count
echo -e "${YELLOW}Step 1: Starting canary rollout...${NC}"
kubectl patch "deployment/$DEPLOYMENT" -n "$NAMESPACE" --type=merge \
    -p '{"spec":{"strategy":{"type":"RollingUpdate","rollingUpdate":{"maxSurge":1,"maxUnavailable":0}}}}'
# `migration` is the schema initContainer; it rolls with the api-server.
kubectl set image "deployment/$DEPLOYMENT" \
    "$CONTAINER=$API_IMAGE:$IMAGE_TAG" "migration=$MIGRATE_IMAGE:$IMAGE_TAG" -n "$NAMESPACE"
kubectl annotate "deployment/$DEPLOYMENT" -n "$NAMESPACE" --overwrite \
    "kubernetes.io/change-cause=canary $IMAGE_TAG ($(date -u +%Y-%m-%dT%H:%M:%SZ))"

# Step 2: wait for the first pod running the new image, then pause
echo -e "${YELLOW}Step 2: Waiting for the first canary pod...${NC}"
CANARY_POD=""
for _ in $(seq 1 60); do
    CANARY_POD=$(kubectl get pods -n "$NAMESPACE" -l "$POD_SELECTOR" \
        -o jsonpath="{range .items[*]}{.metadata.name}{' '}{.spec.containers[?(@.name=='$CONTAINER')].image}{'\n'}{end}" \
        | awk -v img="$API_IMAGE:$IMAGE_TAG" '$2 == img {print $1; exit}')
    [ -n "$CANARY_POD" ] && break
    sleep 5
done
[ -n "$CANARY_POD" ] || abort "✗ No pod with image $API_IMAGE:$IMAGE_TAG appeared"

kubectl rollout pause "deployment/$DEPLOYMENT" -n "$NAMESPACE"
echo -e "Canary pod: ${GREEN}$CANARY_POD${NC} (rollout paused)"
kubectl wait --for=condition=Ready "pod/$CANARY_POD" -n "$NAMESPACE" --timeout=5m \
    || abort "✗ Canary pod never became Ready"

# Step 3: smoke-test the canary pod directly
echo -e "${YELLOW}Step 3: Smoke-testing the canary pod...${NC}"
kubectl port-forward -n "$NAMESPACE" "pod/$CANARY_POD" "$CANARY_PORT:8090" >/dev/null 2>&1 &
PF_PID=$!
sleep 3
if ! API_URL="http://127.0.0.1:$CANARY_PORT" FRONTEND_URL="" "$SMOKE_TESTS"; then
    abort "✗ Canary smoke tests failed"
fi
cleanup
echo -e "${GREEN}✓ Canary healthy${NC}"

# Step 4: promote by resuming the rollout
echo -e "${YELLOW}Step 4: Promoting canary (resuming rollout)...${NC}"
kubectl rollout resume "deployment/$DEPLOYMENT" -n "$NAMESPACE"
kubectl rollout status "deployment/$DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT" \
    || abort "✗ Full rollout failed"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Canary $IMAGE_TAG promoted${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Rollback command (if needed):"
echo "  kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE"
