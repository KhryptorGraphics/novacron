#!/bin/bash
# NovaCron Setup Script for Linux/macOS
set -e

# Print banner
echo "=========================================================="
echo "                 NovaCron Setup Script                    "
echo "=========================================================="
echo

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker before running this script."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose before running this script."
    exit 1
fi

# Define directories
CONFIG_DIR="./configs"
PROMETHEUS_CONFIG_DIR="$CONFIG_DIR/prometheus"
GRAFANA_CONFIG_DIR="$CONFIG_DIR/grafana/provisioning"
GRAFANA_DASHBOARDS_DIR="$GRAFANA_CONFIG_DIR/dashboards"
GRAFANA_DATASOURCES_DIR="$GRAFANA_CONFIG_DIR/datasources"

# Create config directories
echo "Creating configuration directories..."
mkdir -p "$PROMETHEUS_CONFIG_DIR"
mkdir -p "$GRAFANA_DASHBOARDS_DIR"
mkdir -p "$GRAFANA_DATASOURCES_DIR"

# Create Prometheus configuration
echo "Creating Prometheus configuration..."
cat > "$PROMETHEUS_CONFIG_DIR/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'novacron-api'
    static_configs:
      - targets: ['api:9090']

  - job_name: 'novacron-hypervisor'
    static_configs:
      - targets: ['hypervisor:9000']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

# Create Grafana datasource configuration
echo "Creating Grafana datasource configuration..."
cat > "$GRAFANA_DATASOURCES_DIR/datasource.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create Grafana dashboard configuration
echo "Creating Grafana dashboard configuration..."
cat > "$GRAFANA_DASHBOARDS_DIR/dashboards.yml" << EOF
apiVersion: 1

providers:
  - name: 'Default'
    folder: 'Novacron'
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Create default dashboard for Grafana
echo "Creating default dashboard for Grafana..."
cat > "$GRAFANA_DASHBOARDS_DIR/novacron-dashboard.json" << EOF
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 9,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "process_cpu_seconds_total{job=\"novacron-api\"}",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "API CPU Usage",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "schemaVersion": 25,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {
    "refresh_intervals": [
      "5s",
      "10s",
      "30s",
      "1m",
      "5m",
      "15m",
      "30m",
      "1h",
      "2h",
      "1d"
    ]
  },
  "timezone": "",
  "title": "NovaCron Dashboard",
  "uid": "novacron",
  "version": 1
}
EOF

# Generate random auth secret if not set
if [ -z "$AUTH_SECRET" ]; then
    export AUTH_SECRET=$(openssl rand -base64 32)
    echo "Generated random AUTH_SECRET for API service"
fi

# Create .env file for storing secrets
echo "Creating .env file..."
cat > .env << EOF
# NovaCron environment variables
AUTH_SECRET=$AUTH_SECRET
GRAFANA_PASSWORD=admin
EOF

echo "Configuration files created successfully!"

# Ask if user wants to build and start containers
read -p "Do you want to build and start the containers now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building and starting containers..."
    docker-compose up -d
    
    echo
    echo "NovaCron services are starting!"
    echo "Access the web interface at: http://localhost:3000"
    echo "API is available at: http://localhost:8080"
    echo "Grafana is available at: http://localhost:3001 (admin/admin)"
else
    echo
    echo "Setup completed. You can start NovaCron manually with:"
    echo "docker-compose up -d"
fi

echo
echo "Setup completed successfully!"
