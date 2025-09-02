#!/bin/bash

# NovaCron Monitoring Setup Script
# Usage: ./setup-monitoring.sh [staging|production]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENVIRONMENT="${1:-production}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%H:%M:%S')] ERROR: $1${NC}"; exit 1; }
success() { echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS: $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARN: $1${NC}"; }

log "Setting up monitoring for $ENVIRONMENT environment"

# Load environment configuration
ENV_FILE="$PROJECT_ROOT/deployment/configs/.env.$ENVIRONMENT"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Create monitoring directories
create_directories() {
    log "Creating monitoring directories..."
    
    local monitoring_dirs=(
        "/opt/novacron/monitoring/prometheus"
        "/opt/novacron/monitoring/grafana/dashboards"
        "/opt/novacron/monitoring/grafana/provisioning/datasources"
        "/opt/novacron/monitoring/grafana/provisioning/dashboards"
        "/opt/novacron/monitoring/alertmanager"
        "/var/log/novacron/monitoring"
    )
    
    for dir in "${monitoring_dirs[@]}"; do
        sudo mkdir -p "$dir"
        sudo chown $(id -u):$(id -g) "$dir"
    done
    
    success "Monitoring directories created"
}

# Configure Prometheus
configure_prometheus() {
    log "Configuring Prometheus..."
    
    local prometheus_config="/opt/novacron/monitoring/prometheus/prometheus.yml"
    
    cat > "$prometheus_config" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: '$ENVIRONMENT'
    cluster: 'novacron'

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # NovaCron API Server
  - job_name: 'novacron-api'
    static_configs:
      - targets: ['api:8090']
    metrics_path: /metrics
    scrape_interval: 10s
    scrape_timeout: 5s

  # Node Exporter
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # Postgres Exporter
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  # Redis Exporter
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Grafana
  - job_name: 'grafana'
    static_configs:
      - targets: ['grafana:3000']
    metrics_path: /metrics

  # Container metrics (cAdvisor)
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
    scrape_interval: 30s
EOF
    
    success "Prometheus configuration created"
}

# Configure alert rules
configure_alert_rules() {
    log "Configuring alert rules..."
    
    local alert_rules="/opt/novacron/monitoring/prometheus/alert_rules.yml"
    
    cat > "$alert_rules" << EOF
groups:
  - name: novacron.rules
    rules:
      # Service availability alerts
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ \$labels.instance }} is down"
          description: "{{ \$labels.job }} service on {{ \$labels.instance }} has been down for more than 1 minute."

      # API response time alerts  
      - alert: APIHighResponseTime
        expr: http_request_duration_seconds{quantile="0.95"} > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API response time"
          description: "95th percentile response time is {{ \$value }}s for {{ \$labels.method }} {{ \$labels.endpoint }}"

      # Database connection alerts
      - alert: DatabaseConnectionHigh
        expr: postgres_stat_activity_count > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High database connections"
          description: "Database has {{ \$value }} active connections"

      # Memory usage alerts
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ \$value }}%"

      # Disk usage alerts
      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes * 100 > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High disk usage"
          description: "Disk usage is {{ \$value }}% on {{ \$labels.mountpoint }}"

      # CPU usage alerts
      - alert: HighCPUUsage
        expr: 100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ \$value }}%"

      # Redis alerts
      - alert: RedisDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis instance {{ \$labels.instance }} is down"

      - alert: RedisHighMemoryUsage
        expr: redis_memory_used_bytes / redis_memory_max_bytes * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis high memory usage"
          description: "Redis memory usage is {{ \$value }}%"

      # SSL certificate alerts
      - alert: SSLCertExpiringSoon
        expr: probe_ssl_earliest_cert_expiry - time() < 86400 * 14
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "SSL certificate expiring soon"
          description: "SSL certificate for {{ \$labels.instance }} expires in {{ \$value | humanizeDuration }}"
EOF
    
    success "Alert rules configured"
}

# Configure Alertmanager
configure_alertmanager() {
    log "Configuring Alertmanager..."
    
    local alertmanager_config="/opt/novacron/monitoring/alertmanager/alertmanager.yml"
    
    cat > "$alertmanager_config" << EOF
global:
  smtp_smarthost: '${SMTP_HOST:-localhost:587}'
  smtp_from: '${ALERT_EMAIL:-alerts@novacron.local}'

route:
  group_by: ['alertname']
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

receivers:
  - name: 'default'
    email_configs:
      - to: '${ALERT_EMAIL:-alerts@novacron.local}'
        subject: 'NovaCron Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Instance: {{ .Labels.instance }}
          Severity: {{ .Labels.severity }}
          {{ end }}

  - name: 'critical-alerts'
    email_configs:
      - to: '${ALERT_EMAIL:-alerts@novacron.local}'
        subject: '🚨 CRITICAL: NovaCron Alert'
        body: |
          {{ range .Alerts }}
          CRITICAL ALERT: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Instance: {{ .Labels.instance }}
          Time: {{ .StartsAt }}
          {{ end }}
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL:-}'
        channel: '#alerts'
        title: '🚨 Critical Alert'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ .Annotations.description }}
          {{ end }}

  - name: 'warning-alerts'
    email_configs:
      - to: '${ALERT_EMAIL:-alerts@novacron.local}'
        subject: '⚠️ WARNING: NovaCron Alert'
        body: |
          {{ range .Alerts }}
          WARNING: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Instance: {{ .Labels.instance }}
          Time: {{ .StartsAt }}
          {{ end }}

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
EOF
    
    success "Alertmanager configuration created"
}

# Configure Grafana datasources
configure_grafana_datasources() {
    log "Configuring Grafana datasources..."
    
    local datasources_config="/opt/novacron/monitoring/grafana/provisioning/datasources/prometheus.yml"
    
    cat > "$datasources_config" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
    jsonData:
      timeInterval: 15s
      httpMethod: POST
      queryTimeout: 60s
EOF
    
    success "Grafana datasources configured"
}

# Create Grafana dashboards
create_grafana_dashboards() {
    log "Creating Grafana dashboards..."
    
    local dashboards_config="/opt/novacron/monitoring/grafana/provisioning/dashboards/dashboards.yml"
    
    cat > "$dashboards_config" << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
EOF
    
    # Create main NovaCron dashboard
    local main_dashboard="/opt/novacron/monitoring/grafana/dashboards/novacron-overview.json"
    
    cat > "$main_dashboard" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "NovaCron Overview",
    "description": "Main overview dashboard for NovaCron infrastructure",
    "tags": ["novacron"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Service Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "{{job}} - {{instance}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "Memory Usage %"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "postgres_stat_activity_count",
            "legendFormat": "Active Connections"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF
    
    success "Grafana dashboards created"
}

# Deploy monitoring stack
deploy_monitoring_stack() {
    log "Deploying monitoring stack..."
    
    local monitoring_compose="$PROJECT_ROOT/deployment/docker/docker-compose.monitoring.yml"
    
    cat > "$monitoring_compose" << EOF
version: '3.8'

services:
  # Node Exporter
  node-exporter:
    image: prom/node-exporter:latest
    restart: unless-stopped
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring

  # Postgres Exporter
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    restart: unless-stopped
    environment:
      DATA_SOURCE_NAME: "postgresql://\${POSTGRES_USER}:\${POSTGRES_PASSWORD}@postgres:5432/\${POSTGRES_DB}?sslmode=disable"
    ports:
      - "9187:9187"
    depends_on:
      - postgres
    networks:
      - monitoring
      - database

  # Redis Exporter
  redis-exporter:
    image: oliver006/redis_exporter:latest
    restart: unless-stopped
    environment:
      REDIS_ADDR: redis://redis-master:6379
      REDIS_PASSWORD: \${REDIS_PASSWORD}
    ports:
      - "9121:9121"
    depends_on:
      - redis-master
    networks:
      - monitoring
      - cache

  # cAdvisor
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    networks:
      - monitoring

  # Alertmanager
  alertmanager:
    image: prom/alertmanager:latest
    restart: unless-stopped
    ports:
      - "9093:9093"
    volumes:
      - /opt/novacron/monitoring/alertmanager:/etc/alertmanager:ro
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
    networks:
      - monitoring

networks:
  monitoring:
    external: true
    name: novacron_monitoring
  database:
    external: true
    name: novacron_database
  cache:
    external: true
    name: novacron_cache
EOF
    
    # Deploy the monitoring stack
    docker-compose -f "$monitoring_compose" up -d
    
    success "Monitoring stack deployed"
}

# Configure log aggregation
setup_log_aggregation() {
    log "Setting up log aggregation..."
    
    # Configure rsyslog for centralized logging
    local rsyslog_config="/etc/rsyslog.d/50-novacron.conf"
    
    sudo tee "$rsyslog_config" > /dev/null << EOF
# NovaCron log aggregation
\$template NovaFormat,"%timestamp:::date-rfc3339% %HOSTNAME% %syslogtag%%msg:::sp-if-no-1st-sp%%msg:::drop-last-lf%\n"

# Log all novacron messages to dedicated file
:programname, isequal, "novacron" /var/log/novacron/application.log;NovaFormat
& stop

# Docker container logs
\$template DockerFormat,"%timestamp:::date-rfc3339% %HOSTNAME% docker/%syslogtag%%msg:::sp-if-no-1st-sp%%msg:::drop-last-lf%\n"
:syslogtag, startswith, "docker/" /var/log/novacron/containers.log;DockerFormat
EOF
    
    # Restart rsyslog
    sudo systemctl restart rsyslog
    
    # Configure log rotation
    local logrotate_config="/etc/logrotate.d/novacron"
    
    sudo tee "$logrotate_config" > /dev/null << EOF
/var/log/novacron/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 syslog syslog
    postrotate
        /bin/kill -HUP \$(cat /var/run/rsyslogd.pid 2> /dev/null) 2> /dev/null || true
    endscript
}
EOF
    
    success "Log aggregation configured"
}

# Setup monitoring alerts
setup_monitoring_alerts() {
    log "Setting up monitoring alerts..."
    
    # Create alert testing script
    local alert_test_script="/usr/local/bin/test-novacron-alerts.sh"
    
    sudo tee "$alert_test_script" > /dev/null << EOF
#!/bin/bash
# Test NovaCron monitoring alerts

set -euo pipefail

echo "Testing NovaCron monitoring alerts..."

# Test Prometheus connectivity
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✓ Prometheus is healthy"
else
    echo "✗ Prometheus is not accessible"
    exit 1
fi

# Test Alertmanager connectivity
if curl -s http://localhost:9093/-/healthy > /dev/null; then
    echo "✓ Alertmanager is healthy"
else
    echo "✗ Alertmanager is not accessible"
    exit 1
fi

# Check for active alerts
ALERTS=\$(curl -s http://localhost:9093/api/v1/alerts | jq -r '.data | length')
echo "Active alerts: \$ALERTS"

echo "Alert testing completed"
EOF
    
    sudo chmod +x "$alert_test_script"
    
    success "Monitoring alerts configured"
}

# Main execution
main() {
    log "=== Setting up NovaCron Monitoring ==="
    
    create_directories
    configure_prometheus
    configure_alert_rules
    configure_alertmanager
    configure_grafana_datasources
    create_grafana_dashboards
    deploy_monitoring_stack
    setup_log_aggregation
    setup_monitoring_alerts
    
    success "=== Monitoring Setup Completed ==="
    
    echo ""
    echo "=== Monitoring Access Information ==="
    echo "Prometheus: http://localhost:9090"
    echo "Alertmanager: http://localhost:9093"
    echo "Grafana: http://localhost:3001"
    echo "Node Exporter: http://localhost:9100"
    echo "cAdvisor: http://localhost:8080"
    echo ""
    echo "Test alerts: /usr/local/bin/test-novacron-alerts.sh"
    echo ""
}

# Error handling
trap 'error "Monitoring setup failed at line $LINENO"' ERR

# Run main function
main