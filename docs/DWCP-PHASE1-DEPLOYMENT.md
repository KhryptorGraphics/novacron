# DWCP Phase 1 Deployment Guide

## Overview

This guide covers the deployment of DWCP (Distributed WAN Communication Protocol) Phase 1 to staging and production environments. Phase 1 includes AMST (Adaptive Multi-Stream Transport) and HDE (Hybrid Delta Encoding) with comprehensive monitoring and rollback capabilities.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Configuration Guide](#configuration-guide)
3. [Deployment Steps](#deployment-steps)
4. [Validation Procedures](#validation-procedures)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Troubleshooting](#troubleshooting)
7. [Rollback Procedures](#rollback-procedures)

---

## Pre-Deployment Checklist

### System Requirements

- [ ] **Go 1.21 or later** installed
- [ ] **Prometheus** running (for metrics collection)
- [ ] **Grafana** (optional, for dashboards)
- [ ] **yamllint** (optional, for config validation)
- [ ] **Apache Bench (ab)** (optional, for performance testing)

### Infrastructure Requirements

- [ ] Minimum 4 CPU cores
- [ ] 8GB RAM minimum
- [ ] 100GB disk space
- [ ] Network bandwidth: 1 Gbps minimum
- [ ] RDMA hardware (optional, auto-detected)

### Access Requirements

- [ ] SSH access to deployment servers
- [ ] Sudo privileges
- [ ] Access to configuration repositories
- [ ] Prometheus metrics endpoint access

### Environment Preparation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/novacron.git
cd novacron

# 2. Verify Go installation
go version  # Should be 1.21 or later

# 3. Install optional tools
pip install yamllint  # For config validation
sudo apt-get install apache2-utils  # For ab (performance testing)

# 4. Verify Prometheus
curl http://localhost:9090/metrics  # Should return metrics
```

---

## Configuration Guide

### Configuration Files

DWCP uses a hierarchical configuration system:

1. **Base Configuration**: `/configs/dwcp.yaml` (default settings)
2. **Environment Overrides**: `/configs/dwcp.{environment}.yaml`
3. **Runtime Config**: Loaded at startup

### Key Configuration Parameters

#### AMST Transport Settings

```yaml
transport:
  min_streams: 16           # Minimum concurrent streams
  max_streams: 256          # Maximum concurrent streams
  initial_streams: 32       # Starting stream count
  stream_scaling_factor: 1.5
  congestion_algorithm: "bbr"  # bbr, cubic, or reno
  enable_rdma: false        # Auto-detect RDMA hardware
```

#### HDE Compression Settings

```yaml
compression:
  enabled: true
  algorithm: "zstd"         # zstd, lz4, or snappy
  level: 3                  # 1-22 (3 = balanced)
  enable_delta_encoding: true
  baseline_interval: 5m
  max_delta_chain: 10
  adaptive_threshold: 3.0   # Switch to delta if ratio > 3.0
```

#### Monitoring Settings

```yaml
monitoring:
  prometheus_enabled: true
  metrics_port: 9090
  metrics_path: "/metrics"
  log_level: "info"         # debug, info, warn, error
  health_check_interval: 30s
```

### Environment-Specific Configurations

#### Staging Configuration (`dwcp.staging.yaml`)

```yaml
dwcp:
  enabled: true              # Enable DWCP in staging

  transport:
    initial_streams: 16      # Conservative settings
    max_streams: 128
    enable_rdma: false       # Disable for testing

  monitoring:
    log_level: "debug"       # Verbose logging
    enable_tracing: true
    tracing_sample_rate: 0.5  # 50% sampling
```

#### Production Configuration (`dwcp.production.yaml`)

```yaml
dwcp:
  enabled: false             # Disabled until Phase 2

  transport:
    initial_streams: 64      # Aggressive settings
    max_streams: 256
    enable_rdma: true        # Enable if available
    pacing_rate: 10000000000 # 10 Gbps

  monitoring:
    log_level: "info"
    tracing_sample_rate: 0.01  # 1% sampling

  security:
    enable_tls: true
    enable_mtls: true
```

### Configuration Validation

```bash
# Validate YAML syntax
yamllint -d relaxed configs/dwcp.yaml
yamllint -d relaxed configs/dwcp.staging.yaml

# Test configuration loading
go run backend/cmd/config-validator/main.go configs/dwcp.staging.yaml
```

---

## Deployment Steps

### Automated Deployment (Recommended)

#### Using GitHub Actions

Push to the `dwcp-phase1` branch to trigger automated deployment:

```bash
git checkout -b dwcp-phase1
git push origin dwcp-phase1
```

The CI/CD pipeline will:
1. Run all tests
2. Run benchmarks
3. Perform security scans
4. Deploy to staging
5. Run validation checks
6. Setup monitoring

#### Manual Deployment Script

For manual deployment:

```bash
# 1. Make scripts executable
chmod +x scripts/deploy-dwcp-phase1.sh
chmod +x scripts/validate-dwcp.sh

# 2. Run deployment (staging)
sudo ./scripts/deploy-dwcp-phase1.sh staging

# 3. Validate deployment
./scripts/validate-dwcp.sh
```

### Manual Deployment (Step-by-Step)

#### 1. Pre-Deployment Checks

```bash
# Check Go version
go version

# Check RDMA hardware (optional)
ibv_devices

# Verify Prometheus
curl http://localhost:9090/metrics
```

#### 2. Build Binaries

```bash
cd backend

# Clean previous builds
rm -rf bin/
mkdir -p bin/

# Build with version info
go build -o bin/api-server \
  -ldflags "-X main.Version=1.0.0-dwcp-phase1 -X main.BuildTime=$(date -u +%Y%m%dT%H%M%S)" \
  ./cmd/api-server/main.go
```

#### 3. Run Tests

```bash
# Unit tests
go test -v -race -coverprofile=coverage.out ./...

# DWCP-specific tests
go test -v -race ./core/network/dwcp/...

# Check coverage
go tool cover -func=coverage.out | grep total
```

#### 4. Run Benchmarks

```bash
# AMST benchmarks
go test -bench=BenchmarkAMST -benchmem -run=^$ ./core/network/dwcp/amst/...

# HDE benchmarks
go test -bench=BenchmarkHDE -benchmem -run=^$ ./core/network/dwcp/hde/...
```

#### 5. Deploy Configuration

```bash
# Create config directory
sudo mkdir -p /etc/dwcp/

# Copy configurations
sudo cp configs/dwcp.yaml /etc/dwcp/
sudo cp configs/dwcp.staging.yaml /etc/dwcp/
```

#### 6. Deploy Binary

```bash
# Install binary
sudo cp bin/api-server /usr/local/bin/dwcp-api-server
sudo chmod +x /usr/local/bin/dwcp-api-server
```

#### 7. Start Service

```bash
# Start with staging config
DWCP_CONFIG=/etc/dwcp/dwcp.staging.yaml \
  /usr/local/bin/dwcp-api-server &

# Save PID
echo $! > /tmp/dwcp-service.pid
```

---

## Validation Procedures

### Health Check Validation

```bash
# Run validation script
./scripts/validate-dwcp.sh

# Expected output:
# ✓ API Health: PASSED
# ✓ DWCP Enabled: PASSED
# ✓ AMST Streams: PASSED
# ✓ HDE Compression: PASSED
# ✓ Baseline Sync: PASSED
# ✓ Error Rate: PASSED
# ✓ Prometheus Metrics: PASSED
```

### Manual Health Checks

#### 1. API Health

```bash
curl http://localhost:8080/health

# Expected response:
# {
#   "status": "healthy",
#   "components": {
#     "amst_transport": {"status": "healthy"},
#     "hde_compression": {"status": "healthy"}
#   }
# }
```

#### 2. DWCP Status

```bash
curl http://localhost:8080/api/v1/dwcp/status

# Expected response:
# {
#   "enabled": true,
#   "version": "1.0.0",
#   "phase": "phase1"
# }
```

#### 3. AMST Streams

```bash
curl http://localhost:9090/metrics | grep dwcp_amst_active_streams

# Expected output:
# dwcp_amst_active_streams 32
```

#### 4. HDE Compression

```bash
curl http://localhost:9090/metrics | grep dwcp_hde_compression_ratio

# Expected output:
# dwcp_hde_compression_ratio 5.2
```

#### 5. Error Rate

```bash
curl http://localhost:9090/metrics | grep -E "dwcp_(errors|requests)_total"

# Calculate error rate:
# error_rate = (errors_total / requests_total) * 100
# Should be < 5%
```

### Performance Validation

```bash
# Run performance test
ab -n 1000 -c 10 http://localhost:8080/health

# Check results:
# Requests per second should be > 100
# No failed requests
```

---

## Monitoring and Observability

### Prometheus Metrics

DWCP exposes the following metrics:

#### AMST Metrics

- `dwcp_amst_active_streams` - Number of active streams
- `dwcp_amst_bytes_sent` - Total bytes sent
- `dwcp_amst_bytes_received` - Total bytes received
- `dwcp_amst_stream_errors` - Stream-level errors
- `dwcp_amst_connection_duration_seconds` - Connection duration histogram

#### HDE Metrics

- `dwcp_hde_compression_ratio` - Average compression ratio
- `dwcp_hde_baselines_synced` - Number of synchronized baselines
- `dwcp_hde_delta_encoding_ratio` - Delta encoding efficiency
- `dwcp_hde_dictionary_size_bytes` - Compression dictionary size

#### General Metrics

- `dwcp_requests_total` - Total requests processed
- `dwcp_errors_total` - Total errors encountered
- `dwcp_latency_seconds` - Request latency histogram
- `dwcp_enabled` - DWCP enabled status (0 or 1)
- `dwcp_config_loaded` - Configuration loaded status

### Grafana Dashboards

Import the staging dashboard:

```bash
# Import dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @configs/grafana/dwcp-staging-dashboard.json
```

Dashboard includes:
- **Deployment Status Timeline** - Track deployment progress
- **AMST Active Streams** - Monitor stream scaling
- **HDE Compression Ratio** - Compression effectiveness
- **Pre/Post Performance Comparison** - Before/after metrics
- **Error Rate Monitoring** - Error tracking with alerts
- **Throughput Graphs** - Bytes sent/received
- **Health Check Status** - Component health matrix

### Logging

DWCP logs are structured JSON:

```bash
# View logs
tail -f /var/log/dwcp.log

# Example log entry:
# {
#   "timestamp": "2025-11-08T18:00:00Z",
#   "level": "info",
#   "component": "amst",
#   "message": "Stream scaling triggered",
#   "active_streams": 32,
#   "target_streams": 48
# }
```

### Alerting

Configure Prometheus alerts:

```yaml
# /etc/prometheus/rules/dwcp.yml
groups:
  - name: dwcp_alerts
    interval: 30s
    rules:
      - alert: DWCPHighErrorRate
        expr: rate(dwcp_errors_total[5m]) / rate(dwcp_requests_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "DWCP error rate above 5%"

      - alert: DWCPNoActiveStreams
        expr: dwcp_amst_active_streams == 0
        for: 2m
        annotations:
          summary: "No active AMST streams"

      - alert: DWCPLowCompressionRatio
        expr: dwcp_hde_compression_ratio < 2.0
        for: 10m
        annotations:
          summary: "HDE compression ratio below 2.0"
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Service Fails to Start

**Symptoms:**
```
Health check failed after 30 seconds
```

**Diagnosis:**
```bash
# Check if service is running
ps aux | grep dwcp-api-server

# Check logs
tail -f /var/log/dwcp.log

# Check configuration
cat /etc/dwcp/dwcp.staging.yaml
```

**Solution:**
```bash
# Validate configuration
yamllint /etc/dwcp/dwcp.staging.yaml

# Check port availability
sudo lsof -i :8080
sudo lsof -i :9090

# Restart service
killall dwcp-api-server
DWCP_CONFIG=/etc/dwcp/dwcp.staging.yaml /usr/local/bin/dwcp-api-server &
```

#### Issue 2: No Active AMST Streams

**Symptoms:**
```
dwcp_amst_active_streams 0
```

**Diagnosis:**
```bash
# Check AMST configuration
curl http://localhost:8080/api/v1/dwcp/amst/config

# Check for connection errors
curl http://localhost:9090/metrics | grep amst_stream_errors
```

**Solution:**
```bash
# Check network connectivity
ping remote-host

# Verify firewall rules
sudo iptables -L

# Check RDMA (if enabled)
ibv_devices
ibv_devinfo
```

#### Issue 3: Low Compression Ratio

**Symptoms:**
```
dwcp_hde_compression_ratio < 2.0
```

**Diagnosis:**
```bash
# Check HDE configuration
curl http://localhost:8080/api/v1/dwcp/hde/config

# Check baseline synchronization
curl http://localhost:9090/metrics | grep baselines_synced

# Check data characteristics
curl http://localhost:8080/api/v1/dwcp/hde/stats
```

**Solution:**
```bash
# Increase compression level (edit config)
# compression.level: 5  # Instead of 3

# Enable dictionary compression
# compression.enable_dictionary: true

# Restart service to apply changes
```

#### Issue 4: High Error Rate

**Symptoms:**
```
Error rate > 5%
```

**Diagnosis:**
```bash
# Check error types
curl http://localhost:9090/metrics | grep dwcp_errors

# Check logs for error details
grep -i error /var/log/dwcp.log | tail -50

# Check resource usage
top
df -h
```

**Solution:**
```bash
# If memory errors, increase limits
ulimit -n 65536  # File descriptors
ulimit -m unlimited  # Memory

# If timeout errors, adjust timeouts in config
# transport.read_timeout: 120s  # Instead of 60s

# Monitor resource usage
watch -n 1 'free -h && df -h'
```

### Debug Mode

Enable debug logging:

```bash
# Edit configuration
# monitoring.log_level: "debug"

# Restart service
killall dwcp-api-server
DWCP_CONFIG=/etc/dwcp/dwcp.staging.yaml \
  DWCP_LOG_LEVEL=debug \
  /usr/local/bin/dwcp-api-server &

# View debug logs
tail -f /var/log/dwcp.log
```

### Health Check Deep Dive

```bash
# Get detailed health status
curl http://localhost:8080/health?verbose=true

# Response includes:
# - Component health
# - Recent errors
# - Configuration status
# - Performance metrics
```

---

## Rollback Procedures

### Automated Rollback

The deployment script automatically rolls back on failure:

```bash
# If deployment fails, automatic rollback triggers
# Logs will show: "Deployment failed, initiating rollback..."
```

### Manual Rollback

#### Quick Rollback

```bash
# 1. Stop DWCP service
killall dwcp-api-server

# 2. Disable DWCP in configuration
sudo sed -i 's/enabled: true/enabled: false/' /etc/dwcp/dwcp.staging.yaml

# 3. Restart service with DWCP disabled
DWCP_CONFIG=/etc/dwcp/dwcp.staging.yaml /usr/local/bin/dwcp-api-server &

# 4. Verify standard path working
curl http://localhost:8080/health
```

#### Full Rollback

```bash
# 1. Stop service
killall dwcp-api-server

# 2. Restore from backup
BACKUP_DIR=$(cat /tmp/dwcp-latest-backup)

# 3. Restore configuration
sudo cp -r "$BACKUP_DIR/configs/"* /etc/dwcp/

# 4. Restore binary
sudo cp "$BACKUP_DIR/api-server" /usr/local/bin/dwcp-api-server

# 5. Restart service
/usr/local/bin/dwcp-api-server &

# 6. Validate
./scripts/validate-dwcp.sh
```

### Rollback Validation

```bash
# Verify service is healthy
curl http://localhost:8080/health

# Verify DWCP is disabled (if disabled)
curl http://localhost:8080/api/v1/dwcp/status
# Should show: "enabled": false

# Check error rate
curl http://localhost:9090/metrics | grep -E "dwcp_(errors|requests)_total"
# Error rate should be 0% or very low
```

### Post-Rollback Actions

1. **Document the Issue**: Record why rollback was necessary
2. **Analyze Logs**: Review logs to identify root cause
3. **Fix and Re-test**: Address issues in development environment
4. **Plan Re-deployment**: Schedule new deployment attempt

---

## Success Criteria

Deployment is considered successful when:

- [x] All health checks pass
- [x] AMST streams active (> 0)
- [x] HDE compression ratio > 2.0
- [x] Error rate < 5%
- [x] All Prometheus metrics available
- [x] Grafana dashboard functional
- [x] Performance meets or exceeds baseline
- [x] No critical alerts firing
- [x] Rollback capability verified

---

## Next Steps

After successful Phase 1 deployment:

1. **Monitor for 48 hours** in staging
2. **Collect performance data** and compare with baseline
3. **Address any degraded components**
4. **Plan Phase 2** (additional DWCP features)
5. **Schedule production deployment** (requires approval)

---

## Support and Resources

- **Documentation**: `/docs/architecture/distributed-wan-communication-protocol.md`
- **Configuration Reference**: `/configs/dwcp.yaml`
- **Deployment Scripts**: `/scripts/deploy-dwcp-phase1.sh`
- **Validation Tools**: `/scripts/validate-dwcp.sh`
- **CI/CD Pipeline**: `/.github/workflows/dwcp-phase1-deploy.yml`

For issues or questions, contact the DWCP development team.
