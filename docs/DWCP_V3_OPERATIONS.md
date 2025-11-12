# DWCP v3 Operational Runbooks

**Version:** 3.0.0
**Last Updated:** 2025-11-10

## Table of Contents

1. [Production Deployment](#production-deployment)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Performance Tuning](#performance-tuning)
4. [Security Best Practices](#security-best-practices)
5. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
6. [Scaling Strategies](#scaling-strategies)
7. [Common Issues and Resolutions](#common-issues-and-resolutions)

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] System requirements met (CPU, RAM, disk, network)
- [ ] Dependencies installed (Go, zap, zstd, lz4, ONNX runtime)
- [ ] RDMA drivers installed (for datacenter mode)
- [ ] Network connectivity verified
- [ ] Firewall rules configured
- [ ] SSL/TLS certificates ready
- [ ] Monitoring stack deployed (Prometheus, Grafana)
- [ ] Backup system configured
- [ ] Disaster recovery plan reviewed

### Deployment Steps

#### 1. Install DWCP v3

```bash
# Download release
wget https://releases.novacron.io/dwcp-v3.0.0-linux-amd64.tar.gz

# Extract
tar xzf dwcp-v3.0.0-linux-amd64.tar.gz
cd dwcp-v3.0.0

# Install binaries
sudo cp bin/* /usr/local/bin/
sudo chmod +x /usr/local/bin/dwcp*
```

#### 2. Configure System

```bash
# Create configuration directory
sudo mkdir -p /etc/dwcp/v3

# Deploy configuration
sudo cp configs/production.yaml /etc/dwcp/v3/config.yaml

# Set permissions
sudo chown -R dwcp:dwcp /etc/dwcp
sudo chmod 640 /etc/dwcp/v3/config.yaml
```

#### 3. Create Systemd Service

```bash
sudo cat > /etc/systemd/system/dwcp.service << EOF
[Unit]
Description=DWCP v3 Distributed Compute Protocol
After=network.target

[Service]
Type=simple
User=dwcp
Group=dwcp
ExecStart=/usr/local/bin/dwcp server --config=/etc/dwcp/v3/config.yaml
Restart=always
RestartSec=10s
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable dwcp
sudo systemctl start dwcp
```

#### 4. Verify Deployment

```bash
# Check service status
sudo systemctl status dwcp

# Check health
curl http://localhost:8080/health

# Check metrics
curl http://localhost:8080/metrics
```

---

## Monitoring and Alerting

### Prometheus Metrics

DWCP v3 exposes metrics at `http://localhost:8080/metrics`:

**Transport Metrics:**
```
dwcp_amst_bytes_sent_total
dwcp_amst_bytes_received_total
dwcp_amst_active_streams
dwcp_amst_mode_transitions_total
dwcp_amst_send_latency_seconds
```

**Compression Metrics:**
```
dwcp_hde_compression_ratio
dwcp_hde_bytes_original_total
dwcp_hde_bytes_compressed_total
dwcp_hde_delta_hit_rate
dwcp_hde_ml_algorithm_usage
```

**Prediction Metrics:**
```
dwcp_pba_prediction_latency_seconds
dwcp_pba_prediction_accuracy
dwcp_pba_datacenter_predictions_total
dwcp_pba_internet_predictions_total
```

### Grafana Dashboard

Import the included Grafana dashboard:

```bash
# Import dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @dashboards/dwcp-v3-overview.json
```

**Key Panels:**
- Network mode distribution
- Transport throughput
- Compression ratio over time
- Prediction accuracy
- Sync latency
- Consensus latency
- Placement success rate

### Alerting Rules

```yaml
# /etc/prometheus/rules/dwcp.yml
groups:
- name: dwcp_alerts
  interval: 30s
  rules:
  
  # High error rate
  - alert: DWCPHighErrorRate
    expr: rate(dwcp_errors_total[5m]) > 0.01
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High DWCP error rate"
      description: "Error rate is {{ $value }} errors/second"
  
  # Mode detection issues
  - alert: DWCPModeFlapping
    expr: rate(dwcp_amst_mode_transitions_total[10m]) > 0.5
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "DWCP mode flapping"
      description: "Mode changing {{ $value }} times per minute"
  
  # Low prediction accuracy
  - alert: DWCPLowPredictionAccuracy
    expr: dwcp_pba_prediction_accuracy < 0.7
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "Low prediction accuracy"
      description: "Accuracy is {{ $value }}"
  
  # High sync latency
  - alert: DWCPHighSyncLatency
    expr: dwcp_ass_sync_latency_seconds > 1.0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High sync latency"
      description: "Sync latency is {{ $value }} seconds"
```

### Log Aggregation

```bash
# Centralized logging with Loki
sudo cat > /etc/promtail/config.yaml << EOF
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: dwcp
    static_configs:
    - targets:
        - localhost
      labels:
        job: dwcp
        __path__: /var/log/dwcp/*.log
EOF
```

---

## Performance Tuning

### Per-Mode Optimization

#### Datacenter Mode

```yaml
# /etc/dwcp/v3/config.yaml
mode: datacenter

transport:
  rdma_device: mlx5_0
  streams: 512              # Max streams for throughput
  congestion_algorithm: cubic

compression:
  algorithm: lz4            # Speed over size
  level: 1

prediction:
  model: datacenter
  sequence_length: 10       # Short history

sync:
  mechanism: raft
  timeout: 100ms            # Low latency target

consensus:
  mechanism: raft
  election_timeout: 150ms
```

#### Internet Mode

```yaml
mode: internet

transport:
  tcp_streams: 8            # Fewer streams
  congestion_algorithm: bbr # Better for high latency
  pacing_enabled: true

compression:
  algorithm: zstd-max       # Size over speed
  ml_enabled: true

prediction:
  model: internet
  sequence_length: 60       # Long history

sync:
  mechanism: crdt
  convergence_timeout: 30s  # Longer timeout

consensus:
  mechanism: pbft
  replica_count: 4          # Byzantine tolerance
```

### Resource Limits

```yaml
# /etc/dwcp/v3/resources.yaml
resources:
  cpu:
    limit: 16               # CPU cores
    reservation: 8
  
  memory:
    limit: 32GB
    reservation: 16GB
    ml_cache: 4GB
    compression_buffer: 2GB
  
  network:
    bandwidth_limit: 10Gbps
    connections_max: 10000
```

---

## Security Best Practices

### TLS Configuration

```yaml
# /etc/dwcp/v3/security.yaml
tls:
  enabled: true
  cert_file: /etc/dwcp/certs/server.crt
  key_file: /etc/dwcp/certs/server.key
  ca_file: /etc/dwcp/certs/ca.crt
  min_version: "1.3"
  cipher_suites:
    - TLS_AES_256_GCM_SHA384
    - TLS_AES_128_GCM_SHA256
```

### Authentication

```yaml
authentication:
  mechanism: mutual_tls
  client_cert_required: true
  allowed_subjects:
    - "CN=node-*.datacenter.local"
    - "CN=edge-*.cloud.local"
```

### Network Policies

```bash
# Firewall rules (iptables)
sudo iptables -A INPUT -p tcp --dport 8080 -j ACCEPT  # HTTP API
sudo iptables -A INPUT -p tcp --dport 8443 -j ACCEPT  # HTTPS API
sudo iptables -A INPUT -p tcp --dport 4789 -j ACCEPT  # RDMA
sudo iptables -A INPUT -p tcp --dport 5000 -j ACCEPT  # Raft
sudo iptables -A INPUT -p tcp --dport 6000 -j ACCEPT  # PBFT
```

---

## Backup and Disaster Recovery

### Backup Strategy

```bash
#!/bin/bash
# /usr/local/bin/backup-dwcp.sh

BACKUP_DIR="/backup/dwcp/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configuration
tar czf $BACKUP_DIR/config.tar.gz /etc/dwcp/

# Backup state
dwcp backup state --output=$BACKUP_DIR/state.db

# Backup models
tar czf $BACKUP_DIR/models.tar.gz /var/lib/dwcp/models/

# Backup metadata
tar czf $BACKUP_DIR/metadata.tar.gz /var/lib/dwcp/metadata/

echo "Backup complete: $BACKUP_DIR"
```

### Disaster Recovery

```bash
#!/bin/bash
# /usr/local/bin/restore-dwcp.sh

BACKUP_DIR=$1

# Stop service
sudo systemctl stop dwcp

# Restore configuration
sudo tar xzf $BACKUP_DIR/config.tar.gz -C /

# Restore state
sudo dwcp restore state --input=$BACKUP_DIR/state.db

# Restore models
sudo tar xzf $BACKUP_DIR/models.tar.gz -C /var/lib/dwcp/

# Restore metadata
sudo tar xzf $BACKUP_DIR/metadata.tar.gz -C /var/lib/dwcp/

# Start service
sudo systemctl start dwcp

echo "Restore complete"
```

---

## Scaling Strategies

### Horizontal Scaling

```bash
# Add new node to cluster
dwcp cluster join \
  --node-id=node-5 \
  --address=10.0.1.55:8080 \
  --region=us-west-2

# Rebalance VMs
dwcp rebalance --strategy=least-loaded
```

### Vertical Scaling

```bash
# Increase resource limits
dwcp config set resources.cpu.limit=32
dwcp config set resources.memory.limit=64GB

# Restart to apply
sudo systemctl restart dwcp
```

---

## Common Issues and Resolutions

### Issue 1: High Latency

**Symptom**: Sync latency > 1 second

**Diagnosis**:
```bash
dwcp debug latency --trace
```

**Resolution**:
1. Check network: `ping <peer-node>`
2. Check mode: `dwcp status | grep Mode`
3. Force appropriate mode: `dwcp config set mode=datacenter`

### Issue 2: Low Compression Ratio

**Symptom**: Compression ratio < 2x

**Diagnosis**:
```bash
dwcp stats compression --detailed
```

**Resolution**:
1. Enable ML compression: `dwcp config set hde.ml_enabled=true`
2. Retrain model: `dwcp ml retrain --component=compression`
3. Force better algorithm: `dwcp config set hde.algorithm=zstd-max`

### Issue 3: RDMA Not Working

**Symptom**: Datacenter mode falls back to TCP

**Diagnosis**:
```bash
ibv_devices
ibv_devinfo
dwcp debug transport --mode=rdma
```

**Resolution**:
1. Install drivers: `sudo apt-get install rdma-core`
2. Load modules: `sudo modprobe ib_uverbs mlx5_ib`
3. Restart DWCP: `sudo systemctl restart dwcp`

---

## Emergency Procedures

### Graceful Shutdown

```bash
# Drain VMs (5 minute timeout)
dwcp shutdown --drain --timeout=5m

# Stop service
sudo systemctl stop dwcp
```

### Emergency Rollback

```bash
# Rollback to v1
sudo cp /usr/local/bin/dwcp.v1 /usr/local/bin/dwcp
sudo systemctl restart dwcp
```

---

**For more information, see:**
- Architecture: `/docs/DWCP_V3_ARCHITECTURE.md`
- API Reference: `/docs/DWCP_V3_API_REFERENCE.md`
- Performance Tuning: `/docs/DWCP_V3_PERFORMANCE_TUNING.md`
