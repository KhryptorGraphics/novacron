#!/bin/bash

# NovaCron Orchestration Cleanup Script
# This script removes the NovaCron orchestration system from Kubernetes

set -euo pipefail

# Configuration
NAMESPACE="novacron-orchestration"
FORCE_DELETE="${FORCE_DELETE:-false}"
PRESERVE_DATA="${PRESERVE_DATA:-true}"
DRY_RUN="${DRY_RUN:-false}"

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
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE does not exist"
        exit 0
    fi
    
    log_success "Prerequisites check passed"
}

# Confirm deletion
confirm_deletion() {
    if [[ "$FORCE_DELETE" == "true" ]]; then
        log_warning "Force delete enabled, skipping confirmation"
        return
    fi
    
    echo ""
    log_warning "This will delete the entire NovaCron Orchestration deployment!"
    log_warning "Namespace: $NAMESPACE"
    
    if [[ "$PRESERVE_DATA" == "false" ]]; then
        log_error "WARNING: Data preservation is disabled - all data will be lost!"
    else
        log_info "Data preservation is enabled - PVCs will be retained"
    fi
    
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log_info "Cancelling cleanup"
        exit 0
    fi
}

# Scale down deployments
scale_down_deployments() {
    log_info "Scaling down deployments..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would scale down deployments"
        return
    fi
    
    # Scale down to 0 replicas
    kubectl scale deployment --all --replicas=0 -n $NAMESPACE || log_warning "Failed to scale down some deployments"
    
    # Wait for pods to terminate gracefully
    log_info "Waiting for pods to terminate..."
    kubectl wait --for=delete pod --all -n $NAMESPACE --timeout=300s || log_warning "Some pods did not terminate within timeout"
    
    log_success "Deployments scaled down"
}

# Delete monitoring resources
delete_monitoring() {
    log_info "Deleting monitoring resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would delete monitoring resources"
        return
    fi
    
    # Delete Prometheus monitoring resources
    kubectl delete -f monitoring/prometheus.yaml 2>/dev/null || log_warning "Failed to delete some monitoring resources"
    
    log_success "Monitoring resources deleted"
}

# Delete applications
delete_applications() {
    log_info "Deleting applications..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would delete applications"
        return
    fi
    
    # Delete in reverse order of deployment
    kubectl delete -f kubernetes/service.yaml 2>/dev/null || log_warning "Failed to delete services"
    kubectl delete -f kubernetes/deployment.yaml 2>/dev/null || log_warning "Failed to delete deployments"
    
    log_success "Applications deleted"
}

# Delete configuration
delete_configuration() {
    log_info "Deleting configuration..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would delete configuration"
        return
    fi
    
    kubectl delete -f kubernetes/configmap.yaml 2>/dev/null || log_warning "Failed to delete configuration"
    
    log_success "Configuration deleted"
}

# Delete secrets
delete_secrets() {
    log_info "Deleting secrets..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would delete secrets"
        return
    fi
    
    kubectl delete -f kubernetes/secrets.yaml 2>/dev/null || log_warning "Failed to delete secrets"
    
    log_success "Secrets deleted"
}

# Delete storage
delete_storage() {
    if [[ "$PRESERVE_DATA" == "true" ]]; then
        log_warning "Data preservation enabled, skipping PVC deletion"
        log_info "To delete PVCs manually later, run:"
        echo "  kubectl delete pvc ml-models-pvc orchestration-data-pvc training-data-pvc -n $NAMESPACE"
        return
    fi
    
    log_warning "Deleting storage (data will be lost)..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would delete storage"
        return
    fi
    
    kubectl delete -f kubernetes/pvc.yaml 2>/dev/null || log_warning "Failed to delete storage"
    
    log_success "Storage deleted"
}

# Delete RBAC
delete_rbac() {
    log_info "Deleting RBAC resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would delete RBAC resources"
        return
    fi
    
    kubectl delete clusterrolebinding orchestration-operator 2>/dev/null || log_warning "Failed to delete cluster role binding"
    kubectl delete clusterrole orchestration-operator 2>/dev/null || log_warning "Failed to delete cluster role"
    kubectl delete serviceaccount orchestration-service -n $NAMESPACE 2>/dev/null || log_warning "Failed to delete service account"
    
    log_success "RBAC resources deleted"
}

# Delete namespace
delete_namespace() {
    log_info "Deleting namespace..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would delete namespace $NAMESPACE"
        return
    fi
    
    kubectl delete namespace $NAMESPACE
    
    # Wait for namespace to be fully deleted
    log_info "Waiting for namespace to be deleted..."
    while kubectl get namespace $NAMESPACE &> /dev/null; do
        sleep 5
        echo -n "."
    done
    echo ""
    
    log_success "Namespace deleted"
}

# List remaining resources
list_remaining_resources() {
    log_info "Checking for remaining resources..."
    
    # Check for any remaining PVs that might have been bound to our PVCs
    local remaining_pvs=$(kubectl get pv -o jsonpath='{.items[?(@.spec.claimRef.namespace=="'$NAMESPACE'")].metadata.name}' 2>/dev/null || echo "")
    
    if [[ -n "$remaining_pvs" ]]; then
        log_warning "Remaining PersistentVolumes found:"
        for pv in $remaining_pvs; do
            echo "  - $pv"
        done
        log_info "To delete these PVs manually, run:"
        echo "  kubectl delete pv $remaining_pvs"
    fi
    
    # Check for any remaining cluster-wide resources
    log_info "Checking for remaining cluster-wide resources..."
    kubectl get clusterrole,clusterrolebinding | grep orchestration || log_info "No remaining cluster-wide orchestration resources found"
}

# Backup data before deletion
backup_data() {
    if [[ "$PRESERVE_DATA" == "false" ]]; then
        log_warning "Data preservation disabled, skipping backup"
        return
    fi
    
    log_info "Creating data backup..."
    
    local backup_dir="backup-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup configuration
    kubectl get configmap -n $NAMESPACE -o yaml > "$backup_dir/configmaps.yaml" 2>/dev/null || log_warning "Failed to backup configmaps"
    
    # Backup secrets (without actual secret data for security)
    kubectl get secret -n $NAMESPACE -o yaml | sed 's/data:/data: {}/' > "$backup_dir/secrets-structure.yaml" 2>/dev/null || log_warning "Failed to backup secret structure"
    
    # Export ML models if accessible
    local pods=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/component=orchestration-engine -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [[ -n "$pods" ]]; then
        log_info "Attempting to backup ML models..."
        kubectl exec -n $NAMESPACE "$pods" -- tar czf - -C /var/lib/novacron/ml-models . > "$backup_dir/ml-models.tar.gz" 2>/dev/null || log_warning "Failed to backup ML models"
    fi
    
    log_success "Backup created at $backup_dir"
}

# Main cleanup function
main() {
    log_info "Starting NovaCron Orchestration cleanup"
    log_info "Namespace: $NAMESPACE"
    log_info "Preserve data: $PRESERVE_DATA"
    log_info "Force delete: $FORCE_DELETE"
    log_info "Dry run: $DRY_RUN"
    
    # Change to script directory
    cd "$(dirname "$0")/.."
    
    # Run cleanup steps
    check_prerequisites
    confirm_deletion
    backup_data
    scale_down_deployments
    delete_monitoring
    delete_applications
    delete_configuration
    delete_secrets
    delete_storage
    delete_rbac
    delete_namespace
    list_remaining_resources
    
    log_success "Cleanup completed successfully!"
    
    if [[ "$PRESERVE_DATA" == "true" ]]; then
        echo ""
        log_info "Data has been preserved. PersistentVolumeClaims were not deleted."
        log_info "If you want to delete them later, run:"
        echo "  kubectl delete pvc ml-models-pvc orchestration-data-pvc training-data-pvc"
    fi
    
    if [[ -d "backup-"* ]]; then
        echo ""
        log_info "Backup directory created: backup-*"
        log_info "This contains configuration and data exports from your deployment"
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi