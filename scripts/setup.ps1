# NovaCron Setup Script for Windows
# Run this script as administrator

# Print banner
Write-Host "==========================================================="
Write-Host "                 NovaCron Setup Script                     "
Write-Host "==========================================================="
Write-Host ""

# Check if Docker is installed
try {
    docker --version | Out-Null
} catch {
    Write-Host "Error: Docker is not installed or not in PATH. Please install Docker Desktop before running this script." -ForegroundColor Red
    exit 1
}

# Check if Docker Compose is installed
try {
    docker-compose --version | Out-Null
} catch {
    Write-Host "Error: Docker Compose is not installed or not in PATH. Please ensure Docker Desktop is properly installed." -ForegroundColor Red
    exit 1
}

# Define directories
$ConfigDir = ".\configs"
$PrometheusConfigDir = "$ConfigDir\prometheus"
$GrafanaConfigDir = "$ConfigDir\grafana\provisioning"
$GrafanaDashboardsDir = "$GrafanaConfigDir\dashboards"
$GrafanaDatasourcesDir = "$GrafanaConfigDir\datasources"

# Create config directories
Write-Host "Creating configuration directories..."
New-Item -ItemType Directory -Force -Path $PrometheusConfigDir | Out-Null
New-Item -ItemType Directory -Force -Path $GrafanaDashboardsDir | Out-Null
New-Item -ItemType Directory -Force -Path $GrafanaDatasourcesDir | Out-Null

# Create Prometheus configuration
Write-Host "Creating Prometheus configuration..."
@"
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
"@ | Out-File -FilePath "$PrometheusConfigDir\prometheus.yml" -Encoding utf8

# Create Grafana datasource configuration
Write-Host "Creating Grafana datasource configuration..."
@"
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
"@ | Out-File -FilePath "$GrafanaDatasourcesDir\datasource.yml" -Encoding utf8

# Create Grafana dashboard configuration
Write-Host "Creating Grafana dashboard configuration..."
@"
apiVersion: 1

providers:
  - name: 'Default'
    folder: 'Novacron'
    type: file
    options:
      path: /etc/grafana/provisioning/dashboards
"@ | Out-File -FilePath "$GrafanaDashboardsDir\dashboards.yml" -Encoding utf8

# Create default dashboard for Grafana
Write-Host "Creating default dashboard for Grafana..."
@"
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
"@ | Out-File -FilePath "$GrafanaDashboardsDir\novacron-dashboard.json" -Encoding utf8

# Generate random auth secret if not set
if (-not $env:AUTH_SECRET) {
    $random = [byte[]]::new(32)
    [Security.Cryptography.RNGCryptoServiceProvider]::Create().GetBytes($random)
    $env:AUTH_SECRET = [Convert]::ToBase64String($random)
    Write-Host "Generated random AUTH_SECRET for API service"
}

# Create .env file for storing secrets
Write-Host "Creating .env file..."
@"
# NovaCron environment variables
AUTH_SECRET=$env:AUTH_SECRET
GRAFANA_PASSWORD=admin
"@ | Out-File -FilePath ".env" -Encoding utf8

Write-Host "Configuration files created successfully!"

# Ask if user wants to build and start containers
$StartContainers = Read-Host "Do you want to build and start the containers now? (y/n)"
if ($StartContainers -eq "y" -or $StartContainers -eq "Y") {
    Write-Host "Building and starting containers..."
    docker-compose up -d
    
    Write-Host ""
    Write-Host "NovaCron services are starting!"
    Write-Host "Access the web interface at: http://localhost:8092"
    Write-Host "API is available at: http://localhost:8090"
    Write-Host "Grafana is available at: http://localhost:3001 (admin/admin)"
} else {
    Write-Host ""
    Write-Host "Setup completed. You can start NovaCron manually with:"
    Write-Host "docker-compose up -d"
}

Write-Host ""
Write-Host "Setup completed successfully!"
