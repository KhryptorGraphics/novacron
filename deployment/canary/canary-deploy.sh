#!/bin/bash
# Canary Deployment Script for NovaCron
# Progressive rollout with automated rollback on errors

set -e

CANARY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$CANARY_DIR/../.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
NAMESPACE="${NAMESPACE:-novacron}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CANARY_STAGES=(1 10 50 100)  # Percentage of traffic
STAGE_DURATION=900  # 15 minutes per stage
MONITOR_INTERVAL=60  # 1 minute
ERROR_THRESHOLD=0.01  # 1% error rate
LATENCY_THRESHOLD=0.5  # 500ms p95 latency

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  NovaCron Canary Deployment${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo -e "Namespace: ${GREEN}$NAMESPACE${NC}"
echo -e "Image Tag: ${GREEN}$IMAGE_TAG${NC}"
echo -e "Canary Stages: ${GREEN}${CANARY_STAGES[@]}%${NC}"
echo -e "Stage Duration: ${GREEN}${STAGE_DURATION}s (15 minutes)${NC}"
echo ""

# Deploy canary version
echo -e "${YELLOW}Step 1: Deploying canary version...${NC}"

kubectl set image deployment/novacron-api-canary \
  api-server=ghcr.io/novacron/api:$IMAGE_TAG \
  -n $NAMESPACE

kubectl set image deployment/novacron-core-canary \
  core-server=ghcr.io/novacron/core:$IMAGE_TAG \
  -n $NAMESPACE

# Wait for canary deployment
echo "Waiting for canary deployment to complete..."
kubectl rollout status deployment/novacron-api-canary -n $NAMESPACE --timeout=10m
kubectl rollout status deployment/novacron-core-canary -n $NAMESPACE --timeout=10m

echo -e "${GREEN}✓ Canary deployment completed${NC}"
echo ""

# Function to check metrics
check_metrics() {
    local stage=$1

    echo "Checking canary metrics..."

    # Get error rate
    local error_rate=$(curl -s "http://prometheus:9090/api/v1/query?query=sum(rate(novacron_api_errors_total{version=\"canary\"}[5m]))/sum(rate(novacron_api_requests_total{version=\"canary\"}[5m]))" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    # Get latency
    local latency_p95=$(curl -s "http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(novacron_api_request_duration_seconds_bucket{version=\"canary\"}[5m]))by(le))" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0")

    echo "  Error Rate: $error_rate (threshold: $ERROR_THRESHOLD)"
    echo "  P95 Latency: ${latency_p95}s (threshold: ${LATENCY_THRESHOLD}s)"

    # Check thresholds
    if (( $(echo "$error_rate > $ERROR_THRESHOLD" | bc -l) )); then
        echo -e "${RED}✗ Error rate exceeds threshold${NC}"
        return 1
    fi

    if (( $(echo "$latency_p95 > $LATENCY_THRESHOLD" | bc -l) )); then
        echo -e "${RED}✗ Latency exceeds threshold${NC}"
        return 1
    fi

    echo -e "${GREEN}✓ Metrics within acceptable range${NC}"
    return 0
}

# Function to set canary traffic weight
set_traffic_weight() {
    local weight=$1

    echo "Setting canary traffic to $weight%..."

    # Update Istio VirtualService or Nginx Ingress
    kubectl patch virtualservice novacron-api -n $NAMESPACE --type=json -p="[{\"op\": \"replace\", \"path\": \"/spec/http/0/route/1/weight\", \"value\": $weight}]" 2>/dev/null || \
    kubectl annotate ingress novacron-api -n $NAMESPACE nginx.ingress.kubernetes.io/canary-weight="$weight" --overwrite

    echo -e "${GREEN}✓ Traffic weight updated to $weight%${NC}"
}

# Progressive rollout
for stage in "${CANARY_STAGES[@]}"; do
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  Stage: ${stage}% Traffic${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""

    # Set traffic weight
    set_traffic_weight $stage

    # Monitor for stage duration
    echo "Monitoring for $((STAGE_DURATION / 60)) minutes..."

    ITERATIONS=$((STAGE_DURATION / MONITOR_INTERVAL))
    for i in $(seq 1 $ITERATIONS); do
        echo ""
        echo "Monitoring iteration $i/$ITERATIONS ($(($i * MONITOR_INTERVAL))s / ${STAGE_DURATION}s)..."

        if ! check_metrics $stage; then
            echo -e "${RED}✗ Canary deployment failed at ${stage}% stage${NC}"
            echo "Rolling back..."

            # Rollback: Set traffic to 0%
            set_traffic_weight 0

            # Scale down canary
            kubectl scale deployment/novacron-api-canary -n $NAMESPACE --replicas=0
            kubectl scale deployment/novacron-core-canary -n $NAMESPACE --replicas=0

            echo -e "${GREEN}✓ Rolled back to stable version${NC}"
            exit 1
        fi

        if [ $i -lt $ITERATIONS ]; then
            sleep $MONITOR_INTERVAL
        fi
    done

    echo -e "${GREEN}✓ Stage ${stage}% completed successfully${NC}"

    # Don't wait after 100% stage
    if [ "$stage" != "100" ]; then
        echo "Waiting 2 minutes before next stage..."
        sleep 120
    fi
done

# Success - Promote canary to stable
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Canary Deployment Successful!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo "Promoting canary to stable..."

# Update stable deployment
kubectl set image deployment/novacron-api \
  api-server=ghcr.io/novacron/api:$IMAGE_TAG \
  -n $NAMESPACE

kubectl set image deployment/novacron-core \
  core-server=ghcr.io/novacron/core:$IMAGE_TAG \
  -n $NAMESPACE

kubectl rollout status deployment/novacron-api -n $NAMESPACE --timeout=10m
kubectl rollout status deployment/novacron-core -n $NAMESPACE --timeout=10m

# Set traffic back to 100% stable
set_traffic_weight 0

# Scale down canary
kubectl scale deployment/novacron-api-canary -n $NAMESPACE --replicas=1
kubectl scale deployment/novacron-core-canary -n $NAMESPACE --replicas=1

echo -e "${GREEN}✓ Canary promoted to stable${NC}"
echo ""
echo "Deployment completed successfully!"
echo "Total time: $((${#CANARY_STAGES[@]} * STAGE_DURATION / 60)) minutes"
echo ""
