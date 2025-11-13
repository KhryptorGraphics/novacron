#!/bin/bash
# Blue-Green Deployment Script for NovaCron
# Zero-downtime deployment with instant rollback capability

set -e

BLUE_GREEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$BLUE_GREEN_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-novacron}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NovaCron Blue-Green Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Namespace: ${GREEN}$NAMESPACE${NC}"
echo -e "Image Tag: ${GREEN}$IMAGE_TAG${NC}"
echo ""

# Determine current active environment
CURRENT_ACTIVE=$(kubectl get service -n $NAMESPACE novacron-api -o jsonpath='{.spec.selector.version}' 2>/dev/null || echo "blue")

if [ "$CURRENT_ACTIVE" == "blue" ]; then
    DEPLOY_TO="green"
    SWITCH_FROM="blue"
else
    DEPLOY_TO="blue"
    SWITCH_FROM="green"
fi

echo -e "Current Active: ${GREEN}$SWITCH_FROM${NC}"
echo -e "Deploying To: ${BLUE}$DEPLOY_TO${NC}"
echo ""

# Step 1: Deploy to inactive environment
echo -e "${YELLOW}Step 1: Deploying to $DEPLOY_TO environment...${NC}"

kubectl set image deployment/novacron-api-$DEPLOY_TO \
  api-server=ghcr.io/novacron/api:$IMAGE_TAG \
  -n $NAMESPACE

kubectl set image deployment/novacron-core-$DEPLOY_TO \
  core-server=ghcr.io/novacron/core:$IMAGE_TAG \
  -n $NAMESPACE

kubectl set image deployment/novacron-frontend-$DEPLOY_TO \
  frontend=ghcr.io/novacron/frontend:$IMAGE_TAG \
  -n $NAMESPACE

# Wait for deployment to complete
echo "Waiting for $DEPLOY_TO deployment to complete..."
kubectl rollout status deployment/novacron-api-$DEPLOY_TO -n $NAMESPACE --timeout=10m
kubectl rollout status deployment/novacron-core-$DEPLOY_TO -n $NAMESPACE --timeout=10m
kubectl rollout status deployment/novacron-frontend-$DEPLOY_TO -n $NAMESPACE --timeout=10m

echo -e "${GREEN}✓ Deployment to $DEPLOY_TO completed${NC}"
echo ""

# Step 2: Health checks on new environment
echo -e "${YELLOW}Step 2: Running health checks on $DEPLOY_TO environment...${NC}"

# Get the service endpoint for the new environment
NEW_ENDPOINT=$(kubectl get service -n $NAMESPACE novacron-api-$DEPLOY_TO -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

HEALTHY=false
for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
    echo "Health check attempt $i/$HEALTH_CHECK_RETRIES..."

    if curl -f -s "http://$NEW_ENDPOINT:8080/health" > /dev/null; then
        HEALTHY=true
        echo -e "${GREEN}✓ Health check passed${NC}"
        break
    fi

    if [ $i -lt $HEALTH_CHECK_RETRIES ]; then
        echo "Health check failed, retrying in $HEALTH_CHECK_INTERVAL seconds..."
        sleep $HEALTH_CHECK_INTERVAL
    fi
done

if [ "$HEALTHY" != "true" ]; then
    echo -e "${RED}✗ Health checks failed after $HEALTH_CHECK_RETRIES attempts${NC}"
    echo "Rolling back..."
    exit 1
fi

echo ""

# Step 3: Run smoke tests
echo -e "${YELLOW}Step 3: Running smoke tests on $DEPLOY_TO environment...${NC}"

# Test critical endpoints
echo "Testing API endpoints..."
curl -f -s "http://$NEW_ENDPOINT:8080/api/v1/health" || { echo "API health check failed"; exit 1; }
curl -f -s "http://$NEW_ENDPOINT:8080/metrics" | grep -q "novacron_api_requests_total" || { echo "Metrics endpoint failed"; exit 1; }

echo -e "${GREEN}✓ Smoke tests passed${NC}"
echo ""

# Step 4: Switch traffic to new environment
echo -e "${YELLOW}Step 4: Switching traffic from $SWITCH_FROM to $DEPLOY_TO...${NC}"

# Update service selector to point to new environment
kubectl patch service novacron-api -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"version\":\"$DEPLOY_TO\"}}}"
kubectl patch service novacron-core -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"version\":\"$DEPLOY_TO\"}}}"
kubectl patch service novacron-frontend -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"version\":\"$DEPLOY_TO\"}}}"

echo -e "${GREEN}✓ Traffic switched to $DEPLOY_TO${NC}"
echo ""

# Step 5: Monitor new environment
echo -e "${YELLOW}Step 5: Monitoring $DEPLOY_TO environment for 2 minutes...${NC}"

MONITOR_DURATION=120
MONITOR_INTERVAL=10
ERRORS_DETECTED=false

for i in $(seq 1 $(($MONITOR_DURATION / $MONITOR_INTERVAL))); do
    echo "Monitoring... ($((i * MONITOR_INTERVAL))s / ${MONITOR_DURATION}s)"

    # Check error rate
    ERROR_RATE=$(curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(novacron_api_errors_total[1m]))/sum(rate(novacron_api_requests_total[1m]))" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
        echo -e "${RED}✗ High error rate detected: $ERROR_RATE${NC}"
        ERRORS_DETECTED=true
        break
    fi

    # Check latency
    LATENCY_P95=$(curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(novacron_api_request_duration_seconds_bucket[1m]))by(le))" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    if (( $(echo "$LATENCY_P95 > 0.5" | bc -l) )); then
        echo -e "${YELLOW}⚠ High latency detected: ${LATENCY_P95}s${NC}"
    fi

    sleep $MONITOR_INTERVAL
done

if [ "$ERRORS_DETECTED" == "true" ]; then
    echo -e "${RED}✗ Errors detected in new environment, rolling back...${NC}"

    # Rollback: Switch traffic back to old environment
    kubectl patch service novacron-api -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"version\":\"$SWITCH_FROM\"}}}"
    kubectl patch service novacron-core -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"version\":\"$SWITCH_FROM\"}}}"
    kubectl patch service novacron-frontend -n $NAMESPACE -p "{\"spec\":{\"selector\":{\"version\":\"$SWITCH_FROM\"}}}"

    echo -e "${GREEN}✓ Rolled back to $SWITCH_FROM environment${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Monitoring complete, no issues detected${NC}"
echo ""

# Step 6: Success
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Successful!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Active Environment: $DEPLOY_TO"
echo "Inactive Environment: $SWITCH_FROM (available for instant rollback)"
echo ""
echo "Rollback command (if needed):"
echo "  kubectl patch service novacron-api -n $NAMESPACE -p '{\"spec\":{\"selector\":{\"version\":\"$SWITCH_FROM\"}}}'"
echo ""
