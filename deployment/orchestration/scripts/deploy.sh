#!/bin/bash

# NovaCron Orchestration Deployment Script
# This script deploys the NovaCron orchestration system to Kubernetes

set -euo pipefail

# Configuration
NAMESPACE="novacron-orchestration"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_SECRETS="${SKIP_SECRETS:-false}"
MONITORING_ENABLED="${MONITORING_ENABLED:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if helm is available (optional)
    if command -v helm &> /dev/null; then
        log_info "Helm is available"
        HELM_AVAILABLE=true
    else
        log_warning "Helm is not available, using plain kubectl"
        HELM_AVAILABLE=false
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        log_info "Namespace $NAMESPACE will be created"
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace $NAMESPACE..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create namespace $NAMESPACE"
        return
    fi
    
    kubectl apply -f kubernetes/namespace.yaml
    
    # Wait for namespace to be ready
    kubectl wait --for=condition=Ready namespace/$NAMESPACE --timeout=60s
    
    log_success "Namespace $NAMESPACE created/updated"
}

# Deploy secrets
deploy_secrets() {
    if [[ "$SKIP_SECRETS" == "true" ]]; then
        log_warning "Skipping secrets deployment"
        return
    fi
    
    log_info "Deploying secrets..."
    
    # Check if secrets file exists
    if [[ ! -f "kubernetes/secrets.yaml" ]]; then
        log_error "Secrets file not found. Please create kubernetes/secrets.yaml"
        exit 1
    fi
    
    # Validate that secrets don't contain placeholder values
    if grep -q "CHANGE_ME" kubernetes/secrets.yaml; then
        log_error "Secrets file contains placeholder values. Please update with real secrets."
        exit 1
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy secrets"
        return
    fi
    
    kubectl apply -f kubernetes/secrets.yaml
    
    log_success "Secrets deployed"
}

# Deploy storage
deploy_storage() {
    log_info "Deploying storage resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy storage"
        return
    fi
    
    kubectl apply -f kubernetes/pvc.yaml
    
    # Wait for PVCs to be bound
    log_info "Waiting for persistent volumes to be bound..."
    kubectl wait --for=condition=Bound pvc/ml-models-pvc -n $NAMESPACE --timeout=300s || log_warning "ML models PVC binding timeout"
    kubectl wait --for=condition=Bound pvc/orchestration-data-pvc -n $NAMESPACE --timeout=300s || log_warning "Orchestration data PVC binding timeout"
    kubectl wait --for=condition=Bound pvc/training-data-pvc -n $NAMESPACE --timeout=300s || log_warning "Training data PVC binding timeout"
    
    log_success "Storage deployed"
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy configuration"
        return
    fi
    
    kubectl apply -f kubernetes/configmap.yaml
    
    log_success "Configuration deployed"
}

# Deploy RBAC
deploy_rbac() {
    log_info "Deploying RBAC resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy RBAC"
        return
    fi
    
    # Create service account and RBAC if it doesn't exist
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: orchestration-service
  namespace: $NAMESPACE
  labels:
    app.kubernetes.io/name: novacron-orchestration
    app.kubernetes.io/component: service-account
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: orchestration-operator
  labels:
    app.kubernetes.io/name: novacron-orchestration
rules:
- apiGroups: [""]
  resources: ["pods", "nodes", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["pods", "nodes"]
  verbs: ["get", "list"]
- apiGroups: ["policy"]
  resources: ["poddisruptionbudgets"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: orchestration-operator
  labels:
    app.kubernetes.io/name: novacron-orchestration
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: orchestration-operator
subjects:
- kind: ServiceAccount
  name: orchestration-service
  namespace: $NAMESPACE
EOF
    
    log_success "RBAC deployed"
}

# Deploy applications
deploy_applications() {
    log_info "Deploying applications..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy applications"
        return
    fi
    
    # Deploy the main applications
    kubectl apply -f kubernetes/deployment.yaml
    kubectl apply -f kubernetes/service.yaml
    
    # Wait for deployments to be ready
    log_info "Waiting for deployments to be ready..."
    kubectl wait --for=condition=Available deployment/orchestration-engine -n $NAMESPACE --timeout=600s
    kubectl wait --for=condition=Available deployment/ml-training-service -n $NAMESPACE --timeout=600s || log_warning "ML training service deployment timeout"
    
    log_success "Applications deployed"
}

# Deploy monitoring
deploy_monitoring() {
    if [[ "$MONITORING_ENABLED" != "true" ]]; then
        log_warning "Monitoring disabled, skipping"
        return
    fi
    
    log_info "Deploying monitoring resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would deploy monitoring"
        return
    fi
    
    # Check if Prometheus operator is installed
    if kubectl get crd servicemonitors.monitoring.coreos.com &> /dev/null; then
        kubectl apply -f monitoring/prometheus.yaml
        log_success "Prometheus monitoring deployed"
    else
        log_warning "Prometheus operator not found, skipping ServiceMonitor and PrometheusRule deployment"
    fi
    
    # Deploy Grafana dashboard (requires manual import)
    if [[ -f "monitoring/grafana-dashboard.json" ]]; then
        log_info "Grafana dashboard available at monitoring/grafana-dashboard.json"
        log_info "Please import this dashboard manually into your Grafana instance"
    fi
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n $NAMESPACE -o wide
    
    # Check service status
    log_info "Checking service status..."
    kubectl get services -n $NAMESPACE
    
    # Check ingress status
    log_info "Checking ingress status..."
    kubectl get ingress -n $NAMESPACE || log_warning "No ingress found"
    
    # Check if endpoints are healthy
    log_info "Checking endpoint health..."
    if kubectl get pods -n $NAMESPACE -l app.kubernetes.io/component=orchestration-engine -o jsonpath='{.items[0].status.phase}' | grep -q "Running"; then
        local pod_name=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/component=orchestration-engine -o jsonpath='{.items[0].metadata.name}')
        
        # Port forward temporarily to check health
        kubectl port-forward -n $NAMESPACE pod/$pod_name 8080:8080 &
        local port_forward_pid=$!
        sleep 5
        
        if curl -s http://localhost:8080/health > /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed"
        fi
        
        kill $port_forward_pid 2>/dev/null || true
    else
        log_warning "Orchestration engine pod is not running"
    fi
    
    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log_info "Performing cleanup..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    
    log_info "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting NovaCron Orchestration deployment"
    log_info "Environment: $DEPLOYMENT_ENV"
    log_info "Namespace: $NAMESPACE"
    log_info "Dry run: $DRY_RUN"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Change to script directory
    cd "$(dirname "$0")/.."
    
    # Run deployment steps
    check_prerequisites
    create_namespace
    deploy_secrets
    deploy_storage
    deploy_config
    deploy_rbac
    deploy_applications
    deploy_monitoring
    verify_deployment
    
    log_success "Deployment completed successfully!"
    
    # Print useful information
    echo ""
    log_info "Useful commands:"
    echo "  Monitor pods:     kubectl get pods -n $NAMESPACE -w"
    echo "  View logs:        kubectl logs -n $NAMESPACE -l app.kubernetes.io/component=orchestration-engine -f"
    echo "  Port forward:     kubectl port-forward -n $NAMESPACE svc/orchestration-engine 8080:80"
    echo "  Delete deployment: ./scripts/cleanup.sh"
    echo ""
    
    # Show service endpoints
    log_info "Service endpoints:"
    kubectl get services -n $NAMESPACE
    
    if kubectl get ingress -n $NAMESPACE &> /dev/null; then
        echo ""
        log_info "Ingress endpoints:"
        kubectl get ingress -n $NAMESPACE
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi