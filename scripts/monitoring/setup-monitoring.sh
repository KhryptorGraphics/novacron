#!/bin/bash

# NovaCron Monitoring Setup Script
# Version: 1.0.0
# Description: Setup comprehensive monitoring stack (Prometheus, Grafana, Alertmanager)

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
STORAGE_CLASS="${STORAGE_CLASS:-fast-ssd}"
PROMETHEUS_RETENTION="${PROMETHEUS_RETENTION:-30d}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin123}"

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - $message"
            ;;
        "DEBUG")
            echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message"
            ;;
    esac
}

# Error handler
error_exit() {
    log "ERROR" "$1"
    exit 1
}

# Check dependencies
check_dependencies() {
    log "INFO" "Checking dependencies..."
    
    local deps=("kubectl" "helm" "curl")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            error_exit "$dep is required but not installed"
        fi
    done
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    log "INFO" "All dependencies satisfied"
}

# Create monitoring namespace
create_namespace() {
    log "INFO" "Creating monitoring namespace..."
    
    kubectl create namespace "$MONITORING_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace for network policies
    kubectl label namespace "$MONITORING_NAMESPACE" name=monitoring --overwrite
    
    log "INFO" "Monitoring namespace created/updated: $MONITORING_NAMESPACE"
}

# Add Helm repositories
setup_helm_repos() {
    log "INFO" "Setting up Helm repositories..."
    
    # Add Prometheus community repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    
    # Add Grafana repo
    helm repo add grafana https://grafana.github.io/helm-charts
    
    # Update repos
    helm repo update
    
    log "INFO" "Helm repositories configured"
}

# Install Prometheus Operator
install_prometheus_operator() {
    log "INFO" "Installing Prometheus Operator..."
    
    local values_file="$SCRIPT_DIR/prometheus-operator-values.yaml"
    
    cat > "$values_file" << EOF
# Prometheus Operator Values
fullnameOverride: "prometheus-operator"

prometheusOperator:
  enabled: true
  admissionWebhooks:
    enabled: true
    patch:
      enabled: true

prometheus:
  enabled: true
  prometheusSpec:
    retention: $PROMETHEUS_RETENTION
    retentionSize: "50GB"
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: $STORAGE_CLASS
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
    resources:
      requests:
        memory: "2Gi"
        cpu: "500m"
      limits:
        memory: "4Gi"
        cpu: "2"
    nodeSelector:
      kubernetes.io/os: linux
    serviceMonitorSelectorNilUsesHelmValues: false
    ruleSelectorNilUsesHelmValues: false
    
    # External labels
    externalLabels:
      cluster: novacron
      environment: production
    
    # Remote write (optional)
    # remoteWrite:
    #   - url: "https://prometheus-remote-write-url"
    #     writeRelabelConfigs:
    #       - sourceLabels: [__name__]
    #         regex: 'go_.*'
    #         action: drop

alertmanager:
  enabled: true
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          storageClassName: $STORAGE_CLASS
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 10Gi
    resources:
      requests:
        memory: "256Mi"
        cpu: "100m"
      limits:
        memory: "512Mi"
        cpu: "500m"
    nodeSelector:
      kubernetes.io/os: linux
      
grafana:
  enabled: true
  adminPassword: $GRAFANA_ADMIN_PASSWORD
  persistence:
    enabled: true
    storageClassName: $STORAGE_CLASS
    size: 10Gi
  resources:
    requests:
      memory: "256Mi"
      cpu: "100m"
    limits:
      memory: "512Mi"
      cpu: "500m"
  nodeSelector:
    kubernetes.io/os: linux
  
  # Grafana configuration
  grafana.ini:
    server:
      root_url: https://grafana.novacron.local
    auth:
      disable_login_form: false
    security:
      admin_user: admin
      admin_password: $GRAFANA_ADMIN_PASSWORD
    database:
      type: postgres
      host: novacron-postgres.novacron.svc.cluster.local:5432
      name: grafana
      user: grafana
      password: $GRAFANA_ADMIN_PASSWORD
      ssl_mode: require
  
  # Data sources
  sidecar:
    datasources:
      enabled: true
      defaultDatasourceEnabled: true
    dashboards:
      enabled: true
      folder: /tmp/dashboards
      folderAnnotation: grafana_folder
      provider:
        foldersFromFilesStructure: true

nodeExporter:
  enabled: true

kubeStateMetrics:
  enabled: true

kubeEtcd:
  enabled: false

kubeControllerManager:
  enabled: false

kubeScheduler:
  enabled: false

kubelet:
  enabled: true
  serviceMonitor:
    https: true

coreDns:
  enabled: true

kubeApiServer:
  enabled: true

kubeProxy:
  enabled: true
EOF
    
    # Install or upgrade
    helm upgrade --install kube-prometheus-stack \
        prometheus-community/kube-prometheus-stack \
        --namespace "$MONITORING_NAMESPACE" \
        --values "$values_file" \
        --wait \
        --timeout 600s
    
    log "INFO" "Prometheus Operator installed successfully"
}

# Create Alertmanager configuration
create_alertmanager_config() {
    log "INFO" "Creating Alertmanager configuration..."
    
    local config_file="$SCRIPT_DIR/alertmanager-config.yaml"
    
    cat > "$config_file" << EOF
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-kube-prometheus-stack-alertmanager
  namespace: $MONITORING_NAMESPACE
  labels:
    app.kubernetes.io/name: alertmanager
type: Opaque
stringData:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'smtp.gmail.com:587'
      smtp_from: 'alerts@novacron.local'
      smtp_auth_username: 'alerts@novacron.local'
      smtp_auth_password: 'your-smtp-password'
      slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    
    route:
      group_by: ['alertname', 'cluster', 'service']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'default'
      routes:
        - match:
            severity: critical
          receiver: 'critical-alerts'
        - match:
            severity: warning
          receiver: 'warning-alerts'
        - match:
            alertname: Watchdog
          receiver: 'null'
    
    receivers:
      - name: 'null'
      
      - name: 'default'
        slack_configs:
          - channel: '#novacron-alerts'
            title: 'NovaCron Alert'
            text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
            color: 'good'
            
      - name: 'critical-alerts'
        email_configs:
          - to: 'ops@novacron.local'
            subject: 'CRITICAL: {{ .GroupLabels.alertname }} - NovaCron'
            body: |
              {{ range .Alerts }}
              Alert: {{ .Annotations.summary }}
              Description: {{ .Annotations.description }}
              Severity: {{ .Labels.severity }}
              Instance: {{ .Labels.instance }}
              {{ end }}
        slack_configs:
          - channel: '#novacron-critical'
            title: 'CRITICAL ALERT - NovaCron'
            text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
            color: 'danger'
            
      - name: 'warning-alerts'
        slack_configs:
          - channel: '#novacron-alerts'
            title: 'Warning - NovaCron'
            text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
            color: 'warning'
    
    inhibit_rules:
      - source_match:
          severity: 'critical'
        target_match:
          severity: 'warning'
        equal: ['alertname', 'instance']
EOF
    
    kubectl apply -f "$config_file"
    log "INFO" "Alertmanager configuration applied"
}

# Create Prometheus rules
create_prometheus_rules() {
    log "INFO" "Creating Prometheus alert rules..."
    
    local rules_file="$SCRIPT_DIR/prometheus-rules.yaml"
    
    cat > "$rules_file" << EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: novacron-alerts
  namespace: $MONITORING_NAMESPACE
  labels:
    app: novacron
    prometheus: kube-prometheus-stack-prometheus
    role: alert-rules
spec:
  groups:
    - name: novacron.infrastructure
      rules:
        - alert: NodeDown
          expr: up{job="node-exporter"} == 0
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Node {{ \$labels.instance }} is down"
            description: "Node {{ \$labels.instance }} has been down for more than 5 minutes"
            
        - alert: HighCPUUsage
          expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[2m])) * 100) > 80
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High CPU usage on {{ \$labels.instance }}"
            description: "CPU usage is above 80% for more than 5 minutes"
            
        - alert: HighMemoryUsage
          expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High memory usage on {{ \$labels.instance }}"
            description: "Memory usage is above 85% for more than 5 minutes"
            
        - alert: DiskSpaceLow
          expr: (1 - (node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"})) * 100 > 85
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "Low disk space on {{ \$labels.instance }}"
            description: "Disk usage is above 85% on {{ \$labels.mountpoint }}"
            
    - name: novacron.application
      rules:
        - alert: NovaCronAPIDown
          expr: up{job="novacron-api"} == 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "NovaCron API is down"
            description: "NovaCron API has been down for more than 1 minute"
            
        - alert: HighAPIResponseTime
          expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="novacron-api"}[5m])) by (le)) > 1
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High API response time"
            description: "95th percentile response time is above 1 second"
            
        - alert: HighAPIErrorRate
          expr: sum(rate(http_requests_total{job="novacron-api",code=~"5.."}[5m])) / sum(rate(http_requests_total{job="novacron-api"}[5m])) > 0.1
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "High API error rate"
            description: "API error rate is above 10%"
            
    - name: novacron.database
      rules:
        - alert: PostgreSQLDown
          expr: up{job="postgres-exporter"} == 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "PostgreSQL is down"
            description: "PostgreSQL database has been down for more than 1 minute"
            
        - alert: PostgreSQLHighConnections
          expr: pg_stat_database_numbackends / pg_settings_max_connections * 100 > 80
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High PostgreSQL connections"
            description: "PostgreSQL connection usage is above 80%"
            
        - alert: PostgreSQLSlowQueries
          expr: rate(pg_stat_activity_max_tx_duration[5m]) > 60
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "PostgreSQL slow queries detected"
            description: "Long running queries detected in PostgreSQL"
            
    - name: novacron.vms
      rules:
        - alert: VMCreationFailure
          expr: increase(novacron_vm_creation_failures_total[5m]) > 3
          for: 1m
          labels:
            severity: warning
          annotations:
            summary: "VM creation failures detected"
            description: "{{ \$value }} VM creation failures in the last 5 minutes"
            
        - alert: HighVMResourceUsage
          expr: avg by (vm_id) (novacron_vm_cpu_usage_percent) > 90
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "High VM resource usage"
            description: "VM {{ \$labels.vm_id }} CPU usage is above 90%"
EOF
    
    kubectl apply -f "$rules_file"
    log "INFO" "Prometheus alert rules applied"
}

# Install ServiceMonitors for NovaCron
create_service_monitors() {
    log "INFO" "Creating ServiceMonitors for NovaCron..."
    
    local monitors_file="$SCRIPT_DIR/service-monitors.yaml"
    
    cat > "$monitors_file" << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: novacron-api
  namespace: $MONITORING_NAMESPACE
  labels:
    app: novacron
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: novacron
      app.kubernetes.io/component: api
  namespaceSelector:
    matchNames:
      - novacron
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
      honorLabels: true

---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: postgres-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: postgres-exporter
spec:
  selector:
    matchLabels:
      app: postgres-exporter
  namespaceSelector:
    matchNames:
      - novacron
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics

---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: redis-exporter
  namespace: $MONITORING_NAMESPACE
  labels:
    app: redis-exporter
spec:
  selector:
    matchLabels:
      app: redis-exporter
  namespaceSelector:
    matchNames:
      - novacron
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
EOF
    
    kubectl apply -f "$monitors_file"
    log "INFO" "ServiceMonitors created"
}

# Install Grafana dashboards
install_grafana_dashboards() {
    log "INFO" "Installing Grafana dashboards..."
    
    local dashboards_dir="$SCRIPT_DIR/dashboards"
    mkdir -p "$dashboards_dir"
    
    # NovaCron Overview Dashboard
    cat > "$dashboards_dir/novacron-overview.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "NovaCron Overview",
    "tags": ["novacron"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Requests per Second",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{job=\"novacron-api\"}[5m]))"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job=\"novacron-api\"}[5m])) by (le))"
          }
        ]
      },
      {
        "title": "Active VMs",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(novacron_vms_total{state=\"running\"})"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(pg_stat_database_numbackends)"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF
    
    # Create ConfigMap for dashboards
    kubectl create configmap grafana-dashboard-novacron-overview \
        --from-file="$dashboards_dir/novacron-overview.json" \
        --namespace="$MONITORING_NAMESPACE" \
        --dry-run=client -o yaml | \
    kubectl label --local -f - grafana_dashboard=1 -o yaml | \
    kubectl apply -f -
    
    log "INFO" "Grafana dashboards installed"
}

# Setup ingress for monitoring services
setup_monitoring_ingress() {
    log "INFO" "Setting up monitoring ingress..."
    
    local ingress_file="$SCRIPT_DIR/monitoring-ingress.yaml"
    
    cat > "$ingress_file" << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: monitoring-ingress
  namespace: $MONITORING_NAMESPACE
  annotations:
    kubernetes.io/ingress.class: nginx-internal
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/auth-type: basic
    nginx.ingress.kubernetes.io/auth-secret: monitoring-basic-auth
    nginx.ingress.kubernetes.io/auth-realm: "NovaCron Monitoring"
    cert-manager.io/cluster-issuer: "internal-ca-issuer"
spec:
  tls:
    - hosts:
        - prometheus.novacron.local
        - grafana.novacron.local
        - alertmanager.novacron.local
      secretName: monitoring-tls-cert
  rules:
    - host: prometheus.novacron.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: kube-prometheus-stack-prometheus
                port:
                  number: 9090
    - host: grafana.novacron.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: kube-prometheus-stack-grafana
                port:
                  number: 80
    - host: alertmanager.novacron.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: kube-prometheus-stack-alertmanager
                port:
                  number: 9093

---
apiVersion: v1
kind: Secret
metadata:
  name: monitoring-basic-auth
  namespace: $MONITORING_NAMESPACE
type: Opaque
data:
  auth: YWRtaW46JGFwcjEkOHJXZVNWVnokUXk4UDhXNFZLcWpVZVZOczRZc2pFLgo= # admin:monitor123
EOF
    
    kubectl apply -f "$ingress_file"
    log "INFO" "Monitoring ingress configured"
}

# Verify monitoring stack
verify_monitoring() {
    log "INFO" "Verifying monitoring stack deployment..."
    
    # Check if all pods are running
    log "INFO" "Waiting for all monitoring pods to be ready..."
    kubectl wait --for=condition=ready pod -l "app.kubernetes.io/name=prometheus" -n "$MONITORING_NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l "app.kubernetes.io/name=grafana" -n "$MONITORING_NAMESPACE" --timeout=300s
    kubectl wait --for=condition=ready pod -l "app.kubernetes.io/name=alertmanager" -n "$MONITORING_NAMESPACE" --timeout=300s
    
    # Test Prometheus
    log "INFO" "Testing Prometheus connectivity..."
    kubectl port-forward svc/kube-prometheus-stack-prometheus 9090:9090 -n "$MONITORING_NAMESPACE" &
    local prometheus_pid=$!
    sleep 5
    
    if curl -s http://localhost:9090/-/ready | grep -q "Prometheus is Ready"; then
        log "INFO" "Prometheus is ready"
    else
        log "WARN" "Prometheus may not be fully ready"
    fi
    
    kill $prometheus_pid 2>/dev/null || true
    
    # Test Grafana
    log "INFO" "Testing Grafana connectivity..."
    kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n "$MONITORING_NAMESPACE" &
    local grafana_pid=$!
    sleep 5
    
    if curl -s http://admin:$GRAFANA_ADMIN_PASSWORD@localhost:3000/api/health | grep -q "ok"; then
        log "INFO" "Grafana is ready"
    else
        log "WARN" "Grafana may not be fully ready"
    fi
    
    kill $grafana_pid 2>/dev/null || true
    
    log "INFO" "Monitoring stack verification completed"
}

# Create monitoring documentation
create_monitoring_docs() {
    log "INFO" "Creating monitoring documentation..."
    
    local docs_file="$SCRIPT_DIR/MONITORING_GUIDE.md"
    
    cat > "$docs_file" << EOF
# NovaCron Monitoring Guide

## Overview
This document describes the monitoring setup for NovaCron, including Prometheus, Grafana, and Alertmanager.

## Components
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notifications
- **Node Exporter**: System metrics
- **Kube State Metrics**: Kubernetes cluster metrics

## Access URLs
- Prometheus: https://prometheus.novacron.local
- Grafana: https://grafana.novacron.local
- Alertmanager: https://alertmanager.novacron.local

Default credentials:
- Grafana: admin / $GRAFANA_ADMIN_PASSWORD
- Basic Auth: admin / monitor123

## Key Metrics

### Application Metrics
- \`novacron_vms_total\` - Total number of VMs by state
- \`novacron_api_requests_total\` - Total API requests
- \`novacron_api_request_duration_seconds\` - API response times
- \`novacron_vm_operations_total\` - VM operations counter

### System Metrics
- \`up\` - Service availability
- \`node_cpu_seconds_total\` - CPU usage
- \`node_memory_MemAvailable_bytes\` - Available memory
- \`node_filesystem_avail_bytes\` - Disk space

### Database Metrics
- \`pg_up\` - PostgreSQL availability
- \`pg_stat_database_numbackends\` - Active connections
- \`pg_stat_database_tup_inserted\` - Insert rate

## Alerts

### Critical Alerts
- API service down
- Database down
- High error rate (>10%)

### Warning Alerts
- High CPU/Memory usage (>80%)
- High response time (>1s)
- VM creation failures

## Grafana Dashboards
- **NovaCron Overview**: High-level metrics
- **Node Exporter Full**: System metrics
- **PostgreSQL**: Database metrics
- **Kubernetes Cluster**: K8s metrics

## Maintenance

### Updating Configuration
Edit the Helm values and upgrade:
\`\`\`bash
helm upgrade kube-prometheus-stack prometheus-community/kube-prometheus-stack \\
    --namespace monitoring \\
    --values prometheus-operator-values.yaml
\`\`\`

### Adding New Alerts
Create or modify PrometheusRule resources:
\`\`\`yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: my-alerts
  namespace: monitoring
spec:
  groups:
    - name: my-group
      rules:
        - alert: MyAlert
          expr: up == 0
          for: 5m
\`\`\`

### Backup Grafana Dashboards
\`\`\`bash
kubectl get configmaps -n monitoring -l grafana_dashboard=1 -o yaml > grafana-dashboards-backup.yaml
\`\`\`

## Troubleshooting

### Prometheus Issues
\`\`\`bash
kubectl logs -n monitoring deployment/kube-prometheus-stack-prometheus
kubectl describe prometheus -n monitoring
\`\`\`

### Grafana Issues
\`\`\`bash
kubectl logs -n monitoring deployment/kube-prometheus-stack-grafana
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
\`\`\`

### Storage Issues
Check PVC status:
\`\`\`bash
kubectl get pvc -n monitoring
kubectl describe pvc -n monitoring
\`\`\`

## Best Practices
1. Regular backup of Grafana dashboards
2. Monitor storage usage and cleanup old data
3. Review and update alert thresholds regularly
4. Test alert notifications periodically
5. Keep Prometheus retention period appropriate for storage

Generated on: $(date)
Version: 1.0.0
EOF
    
    log "INFO" "Monitoring documentation created: $docs_file"
}

# Main function
main() {
    log "INFO" "NovaCron Monitoring Setup v1.0.0"
    log "INFO" "Setting up monitoring in namespace: $MONITORING_NAMESPACE"
    
    check_dependencies
    create_namespace
    setup_helm_repos
    install_prometheus_operator
    create_alertmanager_config
    create_prometheus_rules
    create_service_monitors
    install_grafana_dashboards
    setup_monitoring_ingress
    verify_monitoring
    create_monitoring_docs
    
    log "INFO" "Monitoring setup completed successfully!"
    log "INFO" "Access URLs:"
    log "INFO" "  Prometheus: https://prometheus.novacron.local"
    log "INFO" "  Grafana: https://grafana.novacron.local (admin/$GRAFANA_ADMIN_PASSWORD)"
    log "INFO" "  Alertmanager: https://alertmanager.novacron.local"
}

# Display usage information
usage() {
    cat << EOF
NovaCron Monitoring Setup Script

Usage: $0 [OPTIONS]

Options:
    -n, --namespace NAME    Monitoring namespace (default: monitoring)
    -s, --storage-class     Storage class name (default: fast-ssd)
    -r, --retention PERIOD Prometheus retention period (default: 30d)
    -p, --password PASSWORD Grafana admin password
    -h, --help             Show this help message

Examples:
    $0                                     # Default setup
    $0 -n monitoring-prod -r 60d          # Custom namespace and retention
    $0 -p mySecurePassword                 # Custom Grafana password

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            MONITORING_NAMESPACE="$2"
            shift 2
            ;;
        -s|--storage-class)
            STORAGE_CLASS="$2"
            shift 2
            ;;
        -r|--retention)
            PROMETHEUS_RETENTION="$2"
            shift 2
            ;;
        -p|--password)
            GRAFANA_ADMIN_PASSWORD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"