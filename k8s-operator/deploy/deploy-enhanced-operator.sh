#!/bin/bash
set -e

# Enhanced NovaCron Kubernetes Operator Deployment Script
# This script deploys the enhanced operator with multi-cloud and AI capabilities

OPERATOR_VERSION="v2.0.0-enhanced"
NAMESPACE="novacron-system"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if we can create resources
    if ! kubectl auth can-i create namespace &> /dev/null; then
        log_error "Insufficient permissions to create resources"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace ${NAMESPACE}..."
    
    if kubectl get namespace ${NAMESPACE} &> /dev/null; then
        log_warning "Namespace ${NAMESPACE} already exists"
    else
        kubectl create namespace ${NAMESPACE}
        log_success "Namespace ${NAMESPACE} created"
    fi
}

# Install CRDs
install_crds() {
    log_info "Installing Custom Resource Definitions..."
    
    local crd_files=(
        "crds/novacron.io_virtualmachines.yaml"
        "crds/novacron.io_vmclusters.yaml" 
        "crds/novacron.io_multicloudvms.yaml"
        "crds/novacron.io_federatedclusters.yaml"
        "crds/novacron.io_aischedulingpolicies.yaml"
        "crds/novacron.io_cacheintegrations.yaml"
    )
    
    for crd_file in "${crd_files[@]}"; do
        local crd_path="${SCRIPT_DIR}/${crd_file}"
        if [[ -f "$crd_path" ]]; then
            log_info "Installing CRD: $(basename $crd_file)"
            kubectl apply -f "$crd_path"
        else
            log_warning "CRD file not found: $crd_path"
        fi
    done
    
    # Wait for CRDs to be established
    log_info "Waiting for CRDs to be established..."
    sleep 10
    
    log_success "CRDs installed successfully"
}

# Install RBAC
install_rbac() {
    log_info "Installing RBAC configuration..."
    
    local rbac_files=(
        "rbac/service_account.yaml"
        "rbac/role.yaml"
    )
    
    for rbac_file in "${rbac_files[@]}"; do
        local rbac_path="${SCRIPT_DIR}/${rbac_file}"
        if [[ -f "$rbac_path" ]]; then
            kubectl apply -f "$rbac_path"
        else
            log_warning "RBAC file not found: $rbac_path"
        fi
    done
    
    log_success "RBAC configuration installed"
}

# Create default secrets
create_secrets() {
    log_info "Creating default secrets..."
    
    # Create API token secret (empty by default)
    if ! kubectl get secret novacron-api-token -n ${NAMESPACE} &> /dev/null; then
        kubectl create secret generic novacron-api-token \
            --from-literal=token="" \
            -n ${NAMESPACE}
        log_info "Created empty API token secret (update with actual token later)"
    fi
    
    # Create placeholder cloud provider secrets
    local providers=("aws-credentials" "azure-credentials" "gcp-credentials")
    for provider in "${providers[@]}"; do
        if ! kubectl get secret ${provider} -n ${NAMESPACE} &> /dev/null; then
            kubectl create secret generic ${provider} \
                --from-literal=placeholder="update-with-actual-credentials" \
                -n ${NAMESPACE}
            log_info "Created placeholder secret: ${provider}"
        fi
    done
    
    # Create Redis credentials secret
    if ! kubectl get secret redis-credentials -n ${NAMESPACE} &> /dev/null; then
        kubectl create secret generic redis-credentials \
            --from-literal=username="redis" \
            --from-literal=password="change-this-password" \
            -n ${NAMESPACE}
        log_info "Created Redis credentials secret"
    fi
    
    log_success "Default secrets created"
}

# Deploy operator
deploy_operator() {
    log_info "Deploying NovaCron Enhanced Operator..."
    
    local operator_path="${SCRIPT_DIR}/operator.yaml"
    if [[ -f "$operator_path" ]]; then
        kubectl apply -f "$operator_path"
        
        # Wait for deployment to be ready
        log_info "Waiting for operator deployment to be ready..."
        kubectl wait --for=condition=available --timeout=300s deployment/novacron-operator -n ${NAMESPACE}
        
        log_success "Operator deployed successfully"
    else
        log_error "Operator deployment file not found: $operator_path"
        exit 1
    fi
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pods
    local pod_count=$(kubectl get pods -n ${NAMESPACE} -l app=novacron-operator --no-headers | wc -l)
    if [[ $pod_count -eq 0 ]]; then
        log_error "No operator pods found"
        return 1
    fi
    
    # Check pod status
    local ready_pods=$(kubectl get pods -n ${NAMESPACE} -l app=novacron-operator --no-headers | grep "1/1" | wc -l)
    if [[ $ready_pods -eq 0 ]]; then
        log_error "Operator pods are not ready"
        kubectl get pods -n ${NAMESPACE} -l app=novacron-operator
        return 1
    fi
    
    # Check CRDs
    local expected_crds=("virtualmachines" "vmclusters" "multicloudvms" "federatedclusters" "aischedulingpolicies" "cacheintegrations")
    for crd in "${expected_crds[@]}"; do
        if ! kubectl get crd ${crd}.novacron.io &> /dev/null; then
            log_error "CRD not found: ${crd}.novacron.io"
            return 1
        fi
    done
    
    log_success "Deployment verification passed"
}

# Show status
show_status() {
    log_info "Deployment Status:"
    echo
    
    # Operator pods
    echo "Operator Pods:"
    kubectl get pods -n ${NAMESPACE} -l app=novacron-operator -o wide
    echo
    
    # Services
    echo "Services:"
    kubectl get services -n ${NAMESPACE}
    echo
    
    # CRDs
    echo "Custom Resource Definitions:"
    kubectl get crds | grep novacron.io
    echo
    
    # Operator logs (last 10 lines)
    echo "Recent Operator Logs:"
    kubectl logs -n ${NAMESPACE} -l app=novacron-operator --tail=10 || log_warning "Could not retrieve logs"
}

# Deploy examples (optional)
deploy_examples() {
    if [[ "${DEPLOY_EXAMPLES:-false}" == "true" ]]; then
        log_info "Deploying example resources..."
        
        # Create a basic VM template first
        cat <<EOF | kubectl apply -f -
apiVersion: novacron.io/v1
kind: VMTemplate
metadata:
  name: web-server-template
  namespace: default
spec:
  description: "Basic web server template"
  config:
    resources:
      cpu:
        request: "500m"
        limit: "1000m"
      memory:
        request: "512Mi"
        limit: "1Gi"
      disk:
        request: "10Gi"
    image: "ubuntu:20.04"
    command: ["bash"]
    args: ["-c", "sleep infinity"]
EOF
        
        # Deploy examples
        local example_files=(
            "examples/multicloud-vm.yaml"
            "examples/ai-scheduling-policy.yaml" 
            "examples/cache-integration.yaml"
        )
        
        for example_file in "${example_files[@]}"; do
            local example_path="${SCRIPT_DIR}/${example_file}"
            if [[ -f "$example_path" ]]; then
                log_info "Deploying example: $(basename $example_file)"
                kubectl apply -f "$example_path" || log_warning "Failed to deploy example: $example_file"
            fi
        done
        
        log_success "Example resources deployed"
    fi
}

# Main deployment function
main() {
    log_info "Starting NovaCron Enhanced Operator deployment"
    log_info "Version: ${OPERATOR_VERSION}"
    log_info "Namespace: ${NAMESPACE}"
    echo
    
    check_prerequisites
    create_namespace
    install_crds
    install_rbac
    create_secrets
    deploy_operator
    verify_deployment
    deploy_examples
    show_status
    
    echo
    log_success "NovaCron Enhanced Operator deployed successfully!"
    echo
    echo "Next steps:"
    echo "1. Update cloud provider credentials in secrets:"
    echo "   kubectl edit secret aws-credentials -n ${NAMESPACE}"
    echo "   kubectl edit secret azure-credentials -n ${NAMESPACE}" 
    echo "   kubectl edit secret gcp-credentials -n ${NAMESPACE}"
    echo
    echo "2. Update NovaCron API token:"
    echo "   kubectl edit secret novacron-api-token -n ${NAMESPACE}"
    echo
    echo "3. Monitor operator logs:"
    echo "   kubectl logs -f -n ${NAMESPACE} -l app=novacron-operator"
    echo
    echo "4. Test with example resources:"
    echo "   kubectl get multicloudvms"
    echo "   kubectl get aischedulingpolicies"
    echo "   kubectl get cacheintegrations"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --examples)
            DEPLOY_EXAMPLES=true
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --version)
            OPERATOR_VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --examples              Deploy example resources"
            echo "  --namespace NAMESPACE   Override namespace (default: novacron-system)"
            echo "  --version VERSION       Override operator version"
            echo "  --help                  Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main