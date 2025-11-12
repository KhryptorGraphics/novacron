#!/bin/bash
# DWCP v3 Production Monitoring Stack Deployment Script
# Deploys complete monitoring infrastructure with HA and production readiness

set -e  # Exit on error
set -o pipefail  # Exit on pipe failure

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="dwcp-v3-production"
MONITORING_NAMESPACE="dwcp-v3-monitoring"
BACKUP_DIR="/backup/monitoring-$(date +%Y%m%d-%H%M%S)"

# Function to print colored messages
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

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed"
        exit 1
    fi

    # Check cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot access Kubernetes cluster"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Function to create namespaces
create_namespaces() {
    log_info "Creating namespaces..."

    kubectl create namespace $MONITORING_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace $MONITORING_NAMESPACE monitoring=enabled --overwrite

    log_success "Namespaces created"
}

# Function to deploy Prometheus Operator
deploy_prometheus_operator() {
    log_info "Deploying Prometheus Operator..."

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    helm upgrade --install prometheus-operator prometheus-community/kube-prometheus-stack \
        --namespace $MONITORING_NAMESPACE \
        --set prometheus.prometheusSpec.replicas=3 \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.retentionSize=100GB \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=120Gi \
        --set alertmanager.alertmanagerSpec.replicas=3 \
        --set grafana.replicas=2 \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=10Gi \
        --wait

    log_success "Prometheus Operator deployed"
}

# Function to deploy production Prometheus configuration
deploy_prometheus_config() {
    log_info "Deploying production Prometheus configuration..."

    # Create ConfigMap from prometheus-production.yml
    kubectl create configmap prometheus-production-config \
        --from-file=prometheus.yml=prometheus-production.yml \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    # Create alert rules ConfigMap
    kubectl create configmap prometheus-alerts-production \
        --from-file=alerts.yml=prometheus-alerts-production.yml \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    # Apply Alertmanager configuration
    kubectl create secret generic alertmanager-production-config \
        --from-file=alertmanager.yml=alertmanager-production.yml \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    log_success "Prometheus configuration deployed"
}

# Function to deploy Thanos for long-term storage
deploy_thanos() {
    log_info "Deploying Thanos for long-term storage..."

    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update

    # Create S3 credentials secret
    kubectl create secret generic thanos-objstore-config \
        --from-literal=objstore.yml="
type: S3
config:
  bucket: dwcp-v3-metrics
  endpoint: s3.amazonaws.com
  region: us-east-1
  access_key: \${AWS_ACCESS_KEY_ID}
  secret_key: \${AWS_SECRET_ACCESS_KEY}
" \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    helm upgrade --install thanos bitnami/thanos \
        --namespace $MONITORING_NAMESPACE \
        --set query.enabled=true \
        --set query.replicaCount=2 \
        --set queryFrontend.enabled=true \
        --set storegateway.enabled=true \
        --set storegateway.replicaCount=2 \
        --set compactor.enabled=true \
        --set compactor.retentionResolutionRaw=30d \
        --set compactor.retentionResolution5m=180d \
        --set compactor.retentionResolution1h=1y \
        --set receive.enabled=true \
        --set receive.replicaCount=3 \
        --set objstoreConfig="$(kubectl get secret thanos-objstore-config -n $MONITORING_NAMESPACE -o jsonpath='{.data.objstore\.yml}' | base64 -d)" \
        --wait

    log_success "Thanos deployed"
}

# Function to deploy Grafana dashboards
deploy_grafana_dashboards() {
    log_info "Deploying Grafana dashboards..."

    # Create ConfigMaps for dashboards
    kubectl create configmap grafana-dashboard-rollout \
        --from-file=production-rollout.json=grafana/dashboards/production-rollout.json \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    kubectl create configmap grafana-dashboard-realtime \
        --from-file=real-time-performance.json=grafana/dashboards/real-time-performance.json \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    kubectl create configmap grafana-dashboard-sla \
        --from-file=sla-compliance.json=grafana/dashboards/sla-compliance.json \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    # Label ConfigMaps for Grafana auto-discovery
    kubectl label configmap grafana-dashboard-rollout grafana_dashboard=1 -n $MONITORING_NAMESPACE --overwrite
    kubectl label configmap grafana-dashboard-realtime grafana_dashboard=1 -n $MONITORING_NAMESPACE --overwrite
    kubectl label configmap grafana-dashboard-sla grafana_dashboard=1 -n $MONITORING_NAMESPACE --overwrite

    # Restart Grafana to pick up dashboards
    kubectl rollout restart deployment prometheus-operator-grafana -n $MONITORING_NAMESPACE

    log_success "Grafana dashboards deployed"
}

# Function to deploy OpenTelemetry Collector
deploy_otel_collector() {
    log_info "Deploying OpenTelemetry Collector..."

    helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
    helm repo update

    # Create ConfigMap from otel-collector-config.yml
    kubectl create configmap otel-collector-config \
        --from-file=otel-collector-config.yml \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    helm upgrade --install otel-collector open-telemetry/opentelemetry-collector \
        --namespace $MONITORING_NAMESPACE \
        --set mode=deployment \
        --set replicaCount=2 \
        --set config="$(cat otel-collector-config.yml)" \
        --set resources.limits.memory=4Gi \
        --set resources.limits.cpu=2 \
        --wait

    log_success "OpenTelemetry Collector deployed"
}

# Function to deploy Jaeger
deploy_jaeger() {
    log_info "Deploying Jaeger for distributed tracing..."

    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm repo update

    helm upgrade --install jaeger jaegertracing/jaeger \
        --namespace $MONITORING_NAMESPACE \
        --set provisionDataStore.cassandra=true \
        --set cassandra.config.cluster_size=3 \
        --set cassandra.config.max_heap_size=2048M \
        --set cassandra.config.heap_new_size=512M \
        --set collector.replicaCount=2 \
        --set query.replicaCount=2 \
        --set agent.daemonset.enabled=true \
        --set storage.type=cassandra \
        --wait

    log_success "Jaeger deployed"
}

# Function to deploy Loki
deploy_loki() {
    log_info "Deploying Loki for log aggregation..."

    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update

    # Create ConfigMap from loki-config.yml
    kubectl create configmap loki-config \
        --from-file=loki.yaml=loki-config.yml \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    helm upgrade --install loki grafana/loki-distributed \
        --namespace $MONITORING_NAMESPACE \
        --set loki.config="$(cat loki-config.yml)" \
        --set ingester.replicas=3 \
        --set querier.replicas=2 \
        --set distributor.replicas=2 \
        --set queryFrontend.replicas=2 \
        --set gateway.enabled=true \
        --set memcached.enabled=true \
        --set memcachedChunks.enabled=true \
        --set memcachedFrontend.enabled=true \
        --set memcachedIndexQueries.enabled=true \
        --wait

    log_success "Loki deployed"
}

# Function to deploy Promtail
deploy_promtail() {
    log_info "Deploying Promtail log collection agents..."

    # Create ConfigMap from promtail-config.yml
    kubectl create configmap promtail-config \
        --from-file=promtail.yaml=promtail-config.yml \
        --namespace $MONITORING_NAMESPACE \
        --dry-run=client -o yaml | kubectl apply -f -

    helm upgrade --install promtail grafana/promtail \
        --namespace $MONITORING_NAMESPACE \
        --set config.file="$(cat promtail-config.yml)" \
        --set daemonset.enabled=true \
        --wait

    log_success "Promtail deployed"
}

# Function to configure ServiceMonitors for DWCP v3
configure_service_monitors() {
    log_info "Configuring ServiceMonitors for DWCP v3..."

    cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dwcp-v3-core
  namespace: $MONITORING_NAMESPACE
  labels:
    app: dwcp-v3
spec:
  selector:
    matchLabels:
      app: dwcp-v3
  namespaceSelector:
    matchNames:
    - $NAMESPACE
  endpoints:
  - port: metrics
    interval: 5s
    path: /metrics
EOF

    log_success "ServiceMonitors configured"
}

# Function to verify deployment
verify_deployment() {
    log_info "Verifying deployment..."

    local all_ok=true

    # Check Prometheus
    if kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=prometheus | grep -q Running; then
        log_success "Prometheus is running"
    else
        log_error "Prometheus is not running"
        all_ok=false
    fi

    # Check Grafana
    if kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=grafana | grep -q Running; then
        log_success "Grafana is running"
    else
        log_error "Grafana is not running"
        all_ok=false
    fi

    # Check Alertmanager
    if kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=alertmanager | grep -q Running; then
        log_success "Alertmanager is running"
    else
        log_error "Alertmanager is not running"
        all_ok=false
    fi

    # Check OpenTelemetry Collector
    if kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=opentelemetry-collector | grep -q Running; then
        log_success "OpenTelemetry Collector is running"
    else
        log_error "OpenTelemetry Collector is not running"
        all_ok=false
    fi

    # Check Jaeger
    if kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=jaeger | grep -q Running; then
        log_success "Jaeger is running"
    else
        log_error "Jaeger is not running"
        all_ok=false
    fi

    # Check Loki
    if kubectl get pods -n $MONITORING_NAMESPACE -l app.kubernetes.io/name=loki | grep -q Running; then
        log_success "Loki is running"
    else
        log_error "Loki is not running"
        all_ok=false
    fi

    if [ "$all_ok" = true ]; then
        log_success "All monitoring components are running"
        return 0
    else
        log_error "Some monitoring components failed to start"
        return 1
    fi
}

# Function to print access information
print_access_info() {
    log_info "Monitoring Stack Access Information"
    echo ""

    # Get Grafana admin password
    GRAFANA_PASSWORD=$(kubectl get secret --namespace $MONITORING_NAMESPACE prometheus-operator-grafana -o jsonpath="{.data.admin-password}" | base64 --decode)

    echo -e "${GREEN}Grafana:${NC}"
    echo "  URL: http://$(kubectl get svc prometheus-operator-grafana -n $MONITORING_NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):3000"
    echo "  Username: admin"
    echo "  Password: $GRAFANA_PASSWORD"
    echo ""

    echo -e "${GREEN}Prometheus:${NC}"
    echo "  URL: http://$(kubectl get svc prometheus-operator-kube-prom-prometheus -n $MONITORING_NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9090"
    echo ""

    echo -e "${GREEN}Alertmanager:${NC}"
    echo "  URL: http://$(kubectl get svc prometheus-operator-kube-prom-alertmanager -n $MONITORING_NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9093"
    echo ""

    echo -e "${GREEN}Jaeger:${NC}"
    echo "  URL: http://$(kubectl get svc jaeger-query -n $MONITORING_NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):16686"
    echo ""

    echo -e "${GREEN}Dashboards:${NC}"
    echo "  - Production Rollout: https://grafana.novacron.io/d/dwcp-v3-production-rollout"
    echo "  - Real-Time Performance: https://grafana.novacron.io/d/dwcp-v3-real-time"
    echo "  - SLA Compliance: https://grafana.novacron.io/d/dwcp-v3-sla"
    echo ""
}

# Function to create backup
create_backup() {
    log_info "Creating backup of existing configuration..."

    mkdir -p $BACKUP_DIR

    # Backup Prometheus config
    kubectl get configmap -n $MONITORING_NAMESPACE -o yaml > $BACKUP_DIR/configmaps.yaml

    # Backup Grafana dashboards
    kubectl get secret -n $MONITORING_NAMESPACE -o yaml > $BACKUP_DIR/secrets.yaml

    log_success "Backup created at $BACKUP_DIR"
}

# Main deployment function
main() {
    echo ""
    echo "=================================================="
    echo "  DWCP v3 Production Monitoring Stack Deployment"
    echo "=================================================="
    echo ""

    log_info "Starting deployment at $(date)"

    # Check if running in production
    read -p "Are you deploying to PRODUCTION? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        log_warning "Deployment cancelled"
        exit 0
    fi

    # Run deployment steps
    check_prerequisites
    create_backup
    create_namespaces
    deploy_prometheus_operator
    deploy_prometheus_config
    deploy_thanos
    deploy_grafana_dashboards
    deploy_otel_collector
    deploy_jaeger
    deploy_loki
    deploy_promtail
    configure_service_monitors

    # Wait for all pods to be ready
    log_info "Waiting for all pods to be ready..."
    sleep 30

    # Verify deployment
    if verify_deployment; then
        log_success "Deployment completed successfully!"
        print_access_info
    else
        log_error "Deployment completed with errors. Check logs above."
        exit 1
    fi

    log_info "Deployment finished at $(date)"
    echo ""
    echo "Next steps:"
    echo "1. Access Grafana and verify dashboards"
    echo "2. Check Prometheus targets are being scraped"
    echo "3. Verify alerts are configured in Alertmanager"
    echo "4. Test trace collection in Jaeger"
    echo "5. Query logs in Grafana using Loki datasource"
    echo ""
    echo "See docs/DWCP_V3_PHASE5_MONITORING_OPERATIONS.md for operational procedures"
    echo ""
}

# Run main function
main "$@"
