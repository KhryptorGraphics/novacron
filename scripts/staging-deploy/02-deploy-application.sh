#!/usr/bin/env bash
# DWCP v3 Phase 5: Staging Application Deployment
# Deploys DWCP v3 application to staging Kubernetes cluster
# Usage: ./02-deploy-application.sh

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
K8S_MANIFESTS_DIR="/home/kp/novacron/deployments/k8s"
DOCKER_DIR="/home/kp/novacron/deployments/docker"
NAMESPACE="staging"
IMAGE_TAG="${IMAGE_TAG:-staging-$(date +%Y%m%d-%H%M%S)}"
REGISTRY="${REGISTRY:-ghcr.io/novacron}"
IMAGE_NAME="dwcp-v3"

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

check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
        exit 1
    fi

    # Check kubectl connectivity
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster. Please configure kubectl."
        exit 1
    fi

    local context=$(kubectl config current-context)
    log "Kubernetes context: $context"

    success "Prerequisites check passed"
}

build_docker_image() {
    log "Building Docker image..."

    cd "$DOCKER_DIR"

    # Build image
    docker build \
        -t "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
        -t "${REGISTRY}/${IMAGE_NAME}:staging-latest" \
        -f Dockerfile.dwcp-v3 \
        --build-arg VERSION="${IMAGE_TAG}" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        .

    if [ $? -eq 0 ]; then
        success "Docker image built: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    else
        error "Docker build failed"
        exit 1
    fi
}

push_docker_image() {
    log "Pushing Docker image to registry..."

    # Check if logged in to registry
    if ! docker info | grep -q "Registry:"; then
        warning "Not logged in to Docker registry"
        log "Run: echo \$GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin"
        read -p "Press Enter after logging in..."
    fi

    # Push both tags
    docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    docker push "${REGISTRY}/${IMAGE_NAME}:staging-latest"

    if [ $? -eq 0 ]; then
        success "Docker image pushed successfully"
    else
        error "Docker push failed"
        exit 1
    fi
}

create_namespace() {
    log "Creating Kubernetes namespace: $NAMESPACE"

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        kubectl label namespace "$NAMESPACE" \
            environment=staging \
            app=dwcp-v3 \
            phase=5
        success "Namespace created"
    fi
}

deploy_secrets() {
    log "Deploying secrets..."

    # Create Redis password
    local redis_password=$(openssl rand -base64 32)

    # Create API key
    local api_key=$(openssl rand -hex 32)

    kubectl create secret generic dwcp-v3-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=REDIS_PASSWORD="$redis_password" \
        --from-literal=API_KEY="$api_key" \
        --dry-run=client -o yaml | kubectl apply -f -

    success "Secrets deployed"
}

deploy_configmap() {
    log "Deploying ConfigMap..."

    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: dwcp-v3-config
  namespace: $NAMESPACE
data:
  NODE_ENV: "staging"
  PORT: "8080"
  LOG_LEVEL: "debug"
  DWCP_VERSION: "3.0.0"
  DWCP_MODE: "hybrid"
  REDIS_URL: "redis://dwcp-v3-redis:6379"
  PROMETHEUS_PORT: "9090"
  METRICS_ENABLED: "true"
  FEATURE_FLAG_V3_ROLLOUT: "0"
  DATACENTER_THROUGHPUT_TARGET: "2.4GB/s"
  INTERNET_COMPRESSION_TARGET: "80%"
EOF

    success "ConfigMap deployed"
}

deploy_redis() {
    log "Deploying Redis..."

    # Deploy Redis from K8s manifest
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dwcp-v3-redis
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dwcp-v3-redis
  template:
    metadata:
      labels:
        app: dwcp-v3-redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: dwcp-v3-redis
  namespace: $NAMESPACE
spec:
  type: ClusterIP
  ports:
  - port: 6379
    targetPort: 6379
  selector:
    app: dwcp-v3-redis
EOF

    # Wait for Redis to be ready
    kubectl wait --for=condition=available --timeout=60s \
        deployment/dwcp-v3-redis -n "$NAMESPACE"

    success "Redis deployed and ready"
}

deploy_application() {
    log "Deploying DWCP v3 application..."

    # Update image in manifest
    cat "$K8S_MANIFESTS_DIR/dwcp-v3-deployment.yaml" | \
        sed "s|namespace: dwcp-v3|namespace: $NAMESPACE|g" | \
        sed "s|image: ghcr.io/novacron/dwcp-v3:latest|image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" | \
        sed "s|replicas: 3|replicas: 2|g" | \
        kubectl apply -f -

    success "Application deployment created"
}

wait_for_deployment() {
    log "Waiting for deployment to be ready..."

    kubectl rollout status deployment/dwcp-v3 \
        -n "$NAMESPACE" \
        --timeout=5m

    if [ $? -eq 0 ]; then
        success "Deployment is ready"
    else
        error "Deployment failed to become ready"
        kubectl get pods -n "$NAMESPACE"
        kubectl describe deployment dwcp-v3 -n "$NAMESPACE"
        exit 1
    fi
}

deploy_service() {
    log "Deploying Service and Ingress..."

    # Deploy service
    cat "$K8S_MANIFESTS_DIR/dwcp-v3-deployment.yaml" | \
        grep -A 50 "kind: Service" | \
        sed "s|namespace: dwcp-v3|namespace: $NAMESPACE|g" | \
        kubectl apply -f -

    # Get service details
    kubectl get svc -n "$NAMESPACE"

    success "Service deployed"
}

run_smoke_tests() {
    log "Running smoke tests..."

    # Port forward to test locally
    kubectl port-forward -n "$NAMESPACE" svc/dwcp-v3 8080:80 &
    local pf_pid=$!
    sleep 5

    # Test health endpoint
    if curl -f http://localhost:8080/health &> /dev/null; then
        success "Health check passed"
    else
        error "Health check failed"
        kill $pf_pid
        exit 1
    fi

    # Test metrics endpoint
    if curl -f http://localhost:9090/metrics &> /dev/null; then
        success "Metrics endpoint accessible"
    else
        warning "Metrics endpoint not accessible"
    fi

    # Clean up port forward
    kill $pf_pid

    success "Smoke tests passed"
}

save_deployment_info() {
    log "Saving deployment information..."

    local deployment_info="/home/kp/novacron/deployments/staging-deployment-info.json"

    cat > "$deployment_info" <<EOF
{
  "environment": "$NAMESPACE",
  "image": "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}",
  "deployed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "replicas": $(kubectl get deployment dwcp-v3 -n "$NAMESPACE" -o jsonpath='{.spec.replicas}'),
  "ready_replicas": $(kubectl get deployment dwcp-v3 -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo 0),
  "service_ip": "$(kubectl get svc dwcp-v3 -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')"
}
EOF

    cat "$deployment_info"
    success "Deployment info saved to: $deployment_info"
}

notify_completion() {
    log "Sending deployment notification..."

    if command -v npx &> /dev/null; then
        npx claude-flow@alpha hooks notify \
            --message "DWCP v3 deployed to staging: ${IMAGE_TAG}" \
            2>/dev/null || true
    fi

    # Save to memory
    if command -v npx &> /dev/null; then
        npx claude-flow@alpha hooks post-edit \
            --file "/home/kp/novacron/deployments/staging-deployment-info.json" \
            --memory-key "swarm/phase5/staging/deployment" \
            2>/dev/null || true
    fi
}

main() {
    log "===== DWCP v3 Phase 5: Staging Application Deployment ====="
    log "Namespace: $NAMESPACE"
    log "Image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    log ""

    check_prerequisites
    build_docker_image

    # Skip push if using local registry
    if [[ ! "$REGISTRY" =~ ^localhost ]]; then
        push_docker_image
    fi

    create_namespace
    deploy_secrets
    deploy_configmap
    deploy_redis
    deploy_application
    wait_for_deployment
    deploy_service
    run_smoke_tests
    save_deployment_info
    notify_completion

    echo ""
    success "===== Application Deployment Complete ====="
    log "Next steps:"
    log "  1. Run validation tests: ../validation/run-validation-suite.sh"
    log "  2. Monitor metrics: kubectl port-forward -n staging svc/dwcp-v3 9090:9090"
    log "  3. View logs: kubectl logs -n staging -l app=dwcp-v3 -f"
}

# Run main function
main "$@"
