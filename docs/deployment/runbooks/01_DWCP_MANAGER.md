# DWCP Manager Deployment Runbook

**System:** DWCP Manager v2.0
**Purpose:** Core coordinator for all DWCP components
**Status:** Production Ready
**Performance:** 10s health checks, auto-recovery, circuit breaker protection

## Overview

The DWCP Manager is the central coordinator for all Distributed Workload Coordination Protocol components. It manages:

- **Transport Layer:** AMST multi-stream TCP with RDMA support
- **Compression Layer:** HDE compression coordination
- **Health Monitoring:** 10-second interval health checks
- **Auto-Recovery:** Exponential backoff component recovery
- **Circuit Breaker:** 5-failure threshold with 30s timeout
- **Metrics Collection:** 5-second interval metrics gathering

### Key Features

- ✅ Health monitoring with automatic recovery
- ✅ Circuit breaker for fault protection
- ✅ RDMA transport with TCP fallback
- ✅ BBR congestion control
- ✅ Component lifecycle management
- ✅ Real-time metrics collection

### Performance Metrics

- **Health Check Interval:** 10 seconds
- **Metrics Collection:** 5 seconds
- **Recovery Timeout:** 1s → 2s → 4s (exponential backoff)
- **Circuit Breaker:** 5 failures, 30s timeout
- **Transport:** RDMA (when available) or TCP with BBR

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: 8 cores
- RAM: 16 GB
- Network: 1 Gbps
- Storage: 100 GB SSD

**Recommended:**
- CPU: 16 cores
- RAM: 32 GB
- Network: 10 Gbps (for RDMA)
- Storage: 256 GB NVMe SSD

**RDMA Support (Optional but Recommended):**
- Mellanox ConnectX-5 or newer
- InfiniBand or RoCE v2
- RDMA-capable network switch

### Software Requirements

```bash
# Go 1.21 or higher
go version
# Expected: go version go1.21.0 linux/amd64

# Verify Go modules
cd /home/kp/repos/novacron/backend/core/network/dwcp
go mod download
go mod verify

# Build binary
go build -o dwcp-manager ./cmd/manager
```

### Dependencies

```bash
# Install required system packages
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  libibverbs-dev \
  librdmacm-dev \
  rdma-core

# Verify RDMA (if using)
ibv_devices  # Should list RDMA devices

# Install monitoring tools
sudo apt-get install -y prometheus-node-exporter
```

### Access Requirements

- Sudo access for system service setup
- Network access to all DWCP component nodes
- Port 8080 (HTTP) and 8443 (HTTPS) available
- Firewall rules configured (see Network Configuration)

## Deployment Steps

### 1. Pre-Deployment Validation

```bash
# Verify system resources
./scripts/check-system-resources.sh

# Check network connectivity
ping -c 3 consensus-node-01
ping -c 3 compression-api-01

# Verify RDMA (if applicable)
rdma link show

# Check port availability
sudo netstat -tulpn | grep -E ':(8080|8443|9000)'
```

### 2. Configuration

Create configuration file: `/etc/dwcp/config.yaml`

```yaml
# DWCP Manager Configuration
enabled: true
version: "2.0"

# Transport configuration
transport:
  min_streams: 16
  max_streams: 256
  initial_streams: 32
  stream_scaling_factor: 1.5
  congestion_algorithm: "bbr"
  enable_ecn: true
  send_buffer_size: 16777216  # 16 MB
  recv_buffer_size: 16777216  # 16 MB
  connect_timeout: "30s"
  read_timeout: "60s"
  write_timeout: "60s"

  # RDMA settings (enable if hardware available)
  enable_rdma: false  # Set to true if RDMA hardware available
  rdma_device: "mlx5_0"
  rdma_port: 1

  # Packet pacing
  enable_pacing: true
  pacing_rate: 1000000000  # 1 Gbps

# Compression configuration
compression:
  enabled: true
  algorithm: "zstd"
  level: "balanced"
  enable_delta_encoding: true
  baseline_interval: "5m"
  max_delta_chain: 10
  delta_algorithm: "auto"
  enable_dictionary: true
  dictionary_update_interval: "24h"
  enable_adaptive: true
  adaptive_threshold: 15.0
  min_compression_ratio: 1.1
  enable_baseline_sync: false
  baseline_sync_interval: "5s"
  enable_pruning: true
  pruning_interval: "10m"

# Prediction configuration (Phase 2)
prediction:
  enabled: false
  model_type: "lstm"
  prediction_horizon: "5m"
  update_interval: "30s"
  history_window: "1h"
  confidence_level: 0.95

# Synchronization configuration (Phase 3)
sync:
  enabled: false
  sync_interval: "1s"
  max_staleness: "5s"
  conflict_resolution: "lww"
  enable_versioning: true

# Consensus configuration (Phase 3)
consensus:
  enabled: false
  algorithm: "raft"
  quorum_size: 3
  election_timeout: "150ms"
  heartbeat_interval: "50ms"
  adaptive_mode: false
```

Validate configuration:
```bash
# Test configuration loading
./dwcp-manager --config /etc/dwcp/config.yaml --validate

# Check for syntax errors
yamllint /etc/dwcp/config.yaml
```

### 3. Deployment

**Option A: Systemd Service (Recommended)**

Create service file: `/etc/systemd/system/dwcp-manager.service`

```ini
[Unit]
Description=DWCP Manager Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=dwcp
Group=dwcp
WorkingDirectory=/opt/dwcp
ExecStart=/opt/dwcp/bin/dwcp-manager --config /etc/dwcp/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dwcp-manager

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/dwcp

[Install]
WantedBy=multi-user.target
```

Deploy:
```bash
# Create user
sudo useradd -r -s /bin/false dwcp

# Create directories
sudo mkdir -p /opt/dwcp/{bin,logs}
sudo mkdir -p /var/lib/dwcp
sudo chown -R dwcp:dwcp /opt/dwcp /var/lib/dwcp

# Copy binary
sudo cp dwcp-manager /opt/dwcp/bin/
sudo chmod +x /opt/dwcp/bin/dwcp-manager

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable dwcp-manager
sudo systemctl start dwcp-manager
```

**Option B: Docker Container**

```bash
# Build Docker image
docker build -t dwcp-manager:2.0 -f deployments/docker/dwcp-manager.Dockerfile .

# Run container
docker run -d \
  --name dwcp-manager \
  --network host \
  --restart unless-stopped \
  -v /etc/dwcp/config.yaml:/etc/dwcp/config.yaml:ro \
  -v /var/lib/dwcp:/var/lib/dwcp \
  dwcp-manager:2.0
```

**Option C: Kubernetes Deployment**

```bash
# Apply manifests
kubectl apply -f deployments/kubernetes/dwcp-manager/

# Verify deployment
kubectl rollout status deployment/dwcp-manager -n dwcp
```

### 4. Validation

```bash
# Check service status
sudo systemctl status dwcp-manager

# Verify health endpoint
curl http://localhost:8080/health
# Expected: {"status":"healthy","version":"2.0","enabled":true}

# Check metrics endpoint
curl http://localhost:8080/metrics | grep dwcp_

# Verify logs
journalctl -u dwcp-manager -n 50 --no-pager

# Test circuit breaker state
curl http://localhost:8080/circuit-breaker/state
# Expected: "closed" (healthy)
```

### 5. Monitoring Setup

```bash
# Configure Prometheus scrape target
cat >> /etc/prometheus/prometheus.yml <<EOF
  - job_name: 'dwcp-manager'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
EOF

# Reload Prometheus
sudo systemctl reload prometheus

# Verify scraping
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="dwcp-manager")'
```

## Configuration Parameters

### Transport Layer

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `min_streams` | 16 | 1-256 | Minimum concurrent TCP streams |
| `max_streams` | 256 | 16-1024 | Maximum concurrent TCP streams |
| `congestion_algorithm` | bbr | bbr/cubic/reno | TCP congestion control |
| `enable_rdma` | false | true/false | Enable RDMA transport |
| `rdma_device` | mlx5_0 | - | RDMA device name |
| `pacing_rate` | 1Gbps | - | Packet pacing rate (bytes/sec) |

### Health Monitoring

| Parameter | Default | Description |
|-----------|---------|-------------|
| Health Check Interval | 10s | Component health check frequency |
| Metrics Collection | 5s | Metrics gathering frequency |
| Recovery Max Retries | 3 | Maximum recovery attempts |
| Recovery Backoff | 1s→2s→4s | Exponential backoff timing |

### Circuit Breaker

| Parameter | Default | Description |
|-----------|---------|-------------|
| Failure Threshold | 5 | Failures before opening circuit |
| Timeout Duration | 30s | Half-open state timeout |
| Success Threshold | 2 | Successes to close circuit |

## Health Checks

### Endpoint URLs

```bash
# Main health check
GET http://localhost:8080/health

# Expected Response:
{
  "status": "healthy",
  "version": "2.0",
  "enabled": true,
  "components": {
    "transport": "healthy",
    "compression": "healthy",
    "circuit_breaker": "closed"
  }
}

# Metrics endpoint
GET http://localhost:8080/metrics

# Circuit breaker state
GET http://localhost:8080/circuit-breaker/state
```

### Expected Responses

**Healthy:**
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "components_healthy": true
}
```

**Degraded:**
```json
{
  "status": "degraded",
  "unhealthy_components": ["compression"],
  "recovery_in_progress": true
}
```

**Unhealthy:**
```json
{
  "status": "unhealthy",
  "error": "transport layer failed",
  "circuit_breaker": "open"
}
```

### Check Frequency

- **Production:** Every 10 seconds
- **Staging:** Every 30 seconds
- **Development:** Every 60 seconds

## Monitoring

### Key Metrics to Track

**System Health:**
```promql
# Manager uptime
dwcp_manager_uptime_seconds

# Component health status
dwcp_manager_component_healthy{component="transport"}
dwcp_manager_component_healthy{component="compression"}

# Circuit breaker state (0=closed, 1=half-open, 2=open)
dwcp_manager_circuit_breaker_state
```

**Transport Layer:**
```promql
# Stream count
dwcp_transport_active_streams

# Throughput
rate(dwcp_transport_bytes_sent_total[1m])
rate(dwcp_transport_bytes_received_total[1m])

# Latency
dwcp_transport_latency_milliseconds
```

**Recovery Operations:**
```promql
# Recovery attempts
rate(dwcp_manager_recovery_attempts_total[5m])

# Recovery success rate
rate(dwcp_manager_recovery_success_total[5m]) / rate(dwcp_manager_recovery_attempts_total[5m])
```

### Alert Thresholds

**Critical (P0):**
- Manager down for >30 seconds
- Circuit breaker open for >5 minutes
- Transport layer failure

**Warning (P1):**
- CPU usage >80% for 5 minutes
- Memory usage >85% for 5 minutes
- Recovery attempts >10 per minute

**Info (P2):**
- Circuit breaker half-open state
- Component recovery in progress
- Metrics collection delayed

### Dashboard Links

- Grafana: `http://grafana/d/dwcp-manager`
- Prometheus: `http://prometheus:9090/graph?g0.expr=dwcp_manager_`

## Troubleshooting

### Common Issues

#### Issue: Manager won't start

**Symptoms:**
```
Failed to start DWCP manager
Error: invalid configuration
```

**Diagnosis:**
```bash
# Check configuration
./dwcp-manager --config /etc/dwcp/config.yaml --validate

# Verify file permissions
ls -la /etc/dwcp/config.yaml

# Check logs
journalctl -u dwcp-manager -n 100
```

**Resolution:**
```bash
# Fix configuration syntax
yamllint /etc/dwcp/config.yaml

# Ensure proper permissions
sudo chown dwcp:dwcp /etc/dwcp/config.yaml
sudo chmod 640 /etc/dwcp/config.yaml

# Restart service
sudo systemctl restart dwcp-manager
```

#### Issue: Transport layer unhealthy

**Symptoms:**
```
Transport layer health check failed
RDMA device not found
```

**Diagnosis:**
```bash
# Check RDMA devices
ibv_devices

# Verify network connectivity
ping -c 3 peer-node-01

# Check port availability
sudo netstat -tulpn | grep 9000
```

**Resolution:**
```bash
# If RDMA unavailable, disable in config
sed -i 's/enable_rdma: true/enable_rdma: false/' /etc/dwcp/config.yaml

# Restart manager
sudo systemctl restart dwcp-manager

# Verify TCP fallback working
curl http://localhost:8080/health | jq '.components.transport'
```

#### Issue: Circuit breaker stuck open

**Symptoms:**
```
Circuit breaker state: open
Requests being rejected
```

**Diagnosis:**
```bash
# Check circuit breaker state
curl http://localhost:8080/circuit-breaker/state

# Review error logs
journalctl -u dwcp-manager | grep "circuit breaker"

# Check failure count
curl http://localhost:8080/metrics | grep circuit_breaker_failures
```

**Resolution:**
```bash
# Manual circuit breaker reset
curl -X POST http://localhost:8080/circuit-breaker/reset

# Check underlying component health
./scripts/diagnose-components.sh

# Restart if necessary
sudo systemctl restart dwcp-manager
```

#### Issue: High memory usage

**Symptoms:**
```
Memory usage >85%
OOM killer warnings
```

**Diagnosis:**
```bash
# Check process memory
ps aux | grep dwcp-manager

# Review Go heap profile
curl http://localhost:8080/debug/pprof/heap > heap.prof
go tool pprof heap.prof

# Check for memory leaks
curl http://localhost:8080/metrics | grep go_memstats
```

**Resolution:**
```bash
# Reduce buffer sizes in config
sed -i 's/send_buffer_size: 16777216/send_buffer_size: 8388608/' /etc/dwcp/config.yaml

# Set GOGC for more aggressive GC
echo 'Environment="GOGC=50"' | sudo tee -a /etc/systemd/system/dwcp-manager.service

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart dwcp-manager
```

### Diagnostic Commands

```bash
# Full system status
./scripts/dwcp-status.sh

# Component health check
curl http://localhost:8080/health | jq

# Performance metrics
curl http://localhost:8080/metrics | grep -E "(latency|throughput|errors)"

# Active connections
ss -tuln | grep 8080

# Resource usage
top -p $(pgrep dwcp-manager)
```

### Resolution Steps

1. **Identify the issue** using health checks and logs
2. **Check configuration** for correctness
3. **Verify dependencies** (network, RDMA, ports)
4. **Attempt automatic recovery** (wait for health monitor)
5. **Manual intervention** if auto-recovery fails
6. **Escalate** if issue persists >15 minutes

## Rollback Procedure

### Conditions for Rollback

- Manager fails to start after 3 attempts
- Circuit breaker open for >10 minutes
- Data corruption detected
- Critical security vulnerability

### Rollback Steps

```bash
# 1. Stop current version
sudo systemctl stop dwcp-manager

# 2. Backup current state
sudo tar czf /backup/dwcp-state-$(date +%s).tar.gz /var/lib/dwcp

# 3. Restore previous version
sudo cp /backup/dwcp-manager-v1.9 /opt/dwcp/bin/dwcp-manager

# 4. Restore previous config
sudo cp /etc/dwcp/config.yaml.backup /etc/dwcp/config.yaml

# 5. Start previous version
sudo systemctl start dwcp-manager

# 6. Verify rollback
curl http://localhost:8080/health
journalctl -u dwcp-manager -n 50
```

### Data Preservation

```bash
# Before rollback, preserve:
# - Transaction logs
sudo cp -r /var/lib/dwcp/logs /backup/logs-$(date +%s)

# - State snapshots
sudo cp -r /var/lib/dwcp/snapshots /backup/snapshots-$(date +%s)

# - Metrics history
curl http://localhost:8080/metrics > /backup/metrics-$(date +%s).txt
```

## Performance Tuning

### Optimization Parameters

**For High Throughput (>1 Gbps):**
```yaml
transport:
  max_streams: 512
  send_buffer_size: 33554432  # 32 MB
  recv_buffer_size: 33554432
  enable_rdma: true
  pacing_rate: 10000000000  # 10 Gbps
```

**For Low Latency (<10ms):**
```yaml
transport:
  min_streams: 8
  congestion_algorithm: "reno"
  enable_pacing: false
  enable_rdma: true
```

**For Resource Constrained:**
```yaml
transport:
  max_streams: 64
  send_buffer_size: 4194304  # 4 MB
  recv_buffer_size: 4194304
compression:
  algorithm: "lz4"  # Faster than zstd
  level: "fast"
```

### Tuning Guidelines

1. **Monitor baseline performance** for 24 hours
2. **Adjust one parameter** at a time
3. **Measure impact** for 6 hours minimum
4. **Document changes** and results
5. **Rollback if degradation** observed

### Benchmarking

```bash
# Run performance benchmark
./scripts/benchmark-dwcp-manager.sh

# Expected results:
# - Health check latency: <5ms
# - Throughput: >1 Gbps (RDMA), >800 Mbps (TCP)
# - CPU usage: 20-40% at full load
# - Memory usage: 30-50% steady state
```

## References

- **Architecture:** `/home/kp/repos/novacron/backend/core/network/dwcp/README.md`
- **Source Code:** `/home/kp/repos/novacron/backend/core/network/dwcp/dwcp_manager.go`
- **Configuration:** `/home/kp/repos/novacron/backend/core/network/dwcp/config.go`
- **Circuit Breaker:** `/home/kp/repos/novacron/backend/core/network/dwcp/circuit_breaker.go`
- **Transport:** `/home/kp/repos/novacron/backend/core/network/dwcp/transport/`

---

**Runbook Version:** 1.0
**Last Updated:** 2025-11-14
**Next Review:** 2025-12-14
**Owner:** Platform Engineering Team
**On-Call:** ops-oncall@example.com
