# DWCP v3 Quick Start Guide

**Version:** 3.0.0
**Last Updated:** 2025-11-10
**Time to Complete:** 5-10 minutes

## What is DWCP v3?

DWCP v3 is a **hybrid datacenter + internet distributed compute protocol** that automatically adapts to network conditions. It extends v1's datacenter-only RDMA implementation with internet-optimized components while maintaining 100% backward compatibility.

---

## Prerequisites

- Linux 4.15+ (Ubuntu 18.04+, RHEL 8+)
- 4+ CPU cores, 8+ GB RAM
- Network connectivity
- Go 1.21+ (for building from source)

---

## Quick Install (5 minutes)

### Option 1: Binary Installation (Recommended)

```bash
# Download release
wget https://github.com/khryptorgraphics/novacron/releases/download/v3.0.0/dwcp-v3.0.0-linux-amd64.tar.gz

# Extract
tar xzf dwcp-v3.0.0-linux-amd64.tar.gz
cd dwcp-v3.0.0

# Install
sudo cp bin/dwcp /usr/local/bin/
sudo chmod +x /usr/local/bin/dwcp

# Verify
dwcp version
# Output: DWCP v3.0.0
```

### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/khryptorgraphics/novacron.git
cd novacron/backend/core/network/dwcp/v3

# Build
go build -o dwcp ./cmd/dwcp

# Install
sudo cp dwcp /usr/local/bin/
```

---

## Quick Start: Datacenter Mode (v1 Compatible)

Use this for existing datacenter deployments with RDMA.

```bash
# Create config
cat > config.yaml << EOF
version: v3
mode: datacenter

transport:
  rdma_device: mlx5_0
  streams: 64

compression:
  algorithm: lz4

features:
  enable_hybrid_mode: false  # v1 compatible
EOF

# Start DWCP
dwcp server --config=config.yaml
```

**Expected Output:**
```
INFO  Starting DWCP v3.0.0
INFO  Mode: datacenter (v1 compatible)
INFO  AMST: RDMA transport initialized (mlx5_0)
INFO  HDE: LZ4 compression enabled
INFO  Ready on :8080
```

---

## Quick Start: Internet Mode

Use this for cloud, edge, or volunteer compute nodes.

```bash
# Create config
cat > config.yaml << EOF
version: v3
mode: internet

transport:
  tcp_streams: 8
  congestion_algorithm: bbr

compression:
  algorithm: zstd-max
  ml_enabled: true

sync:
  mechanism: crdt

consensus:
  mechanism: pbft
  replica_count: 4
EOF

# Start DWCP
dwcp server --config=config.yaml
```

**Expected Output:**
```
INFO  Starting DWCP v3.0.0
INFO  Mode: internet
INFO  AMST: TCP transport initialized (BBR)
INFO  HDE: ML compression enabled
INFO  ASS: CRDT sync initialized
INFO  ACP: PBFT consensus initialized (f=1)
INFO  Ready on :8080
```

---

## Quick Start: Hybrid Mode (Recommended)

Auto-detects network conditions and selects optimal mode.

```bash
# Create config
cat > config.yaml << EOF
version: v3
mode: hybrid
auto_mode_detection: true

transport:
  datacenter_streams: 512
  internet_streams: 8
  congestion_algorithm: bbr

compression:
  ml_enabled: true

prediction:
  datacenter_model: /var/lib/dwcp/models/dc_predictor.onnx
  internet_model: /var/lib/dwcp/models/inet_predictor.onnx
EOF

# Start DWCP
dwcp server --config=config.yaml
```

**Expected Output:**
```
INFO  Starting DWCP v3.0.0
INFO  Mode: hybrid (auto-detection enabled)
INFO  Mode Detector: Analyzing network...
INFO  Detected: datacenter (latency: 5ms, bandwidth: 10Gbps)
INFO  AMST: RDMA transport active
INFO  Ready on :8080
```

---

## Basic Usage

### 1. Check Status

```bash
# System status
dwcp status

# Output:
# Status: Running
# Mode: datacenter
# Uptime: 2m 15s
# VMs: 42 active
# Throughput: 12.3 Gbps
```

### 2. Check Health

```bash
# Health check
curl http://localhost:8080/health

# Output:
# {"status":"healthy","mode":"datacenter","components":{"amst":"ok","hde":"ok","pba":"ok","ass":"ok","acp":"ok","itp":"ok"}}
```

### 3. View Metrics

```bash
# Prometheus metrics
curl http://localhost:8080/metrics | grep dwcp_

# Key metrics:
# dwcp_amst_throughput_gbps 12.3
# dwcp_hde_compression_ratio 2.82
# dwcp_pba_prediction_accuracy 0.87
```

### 4. Send Data (API Example)

```bash
# Send VM state
curl -X POST http://localhost:8080/api/v3/send \
  -H "Content-Type: application/octet-stream" \
  --data-binary @vm-state.bin

# Output:
# {"success":true,"bytes_sent":1048576,"compression_ratio":2.8,"latency_ms":45}
```

---

## Testing Your Setup

### Basic Connectivity Test

```bash
# Test datacenter mode
dwcp test connectivity --mode=datacenter --peer=10.0.1.50

# Test internet mode
dwcp test connectivity --mode=internet --peer=203.0.113.10
```

### Performance Benchmark

```bash
# Quick benchmark (1 minute)
dwcp benchmark quick

# Output:
# Transport throughput: 42.3 Gbps
# Compression ratio: 2.82x
# Prediction latency: 18ms
# Overall score: 9.2/10
```

---

## Common Configurations

### Configuration 1: Pure Datacenter (v1 Compatible)

```yaml
version: v3
mode: datacenter
features:
  enable_hybrid_mode: false
  enable_internet_transport: false

transport:
  rdma_device: mlx5_0
  streams: 512

compression:
  algorithm: lz4
```

### Configuration 2: Pure Internet

```yaml
version: v3
mode: internet

transport:
  tcp_streams: 8
  congestion_algorithm: bbr

compression:
  algorithm: zstd-max
  ml_enabled: true

consensus:
  mechanism: pbft
  replica_count: 4
```

### Configuration 3: Hybrid with Auto-Detection

```yaml
version: v3
mode: hybrid
auto_mode_detection: true

mode_detection:
  interval: 10s
  thresholds:
    datacenter:
      latency_max: 10ms
      bandwidth_min: 1Gbps
    internet:
      latency_min: 50ms
```

---

## Next Steps

### For Datacenter Deployments
1. ✅ Verify RDMA working: `ibv_devices`
2. ✅ Tune for performance: [DWCP_V3_PERFORMANCE_TUNING.md](/docs/DWCP_V3_PERFORMANCE_TUNING.md)
3. ✅ Set up monitoring: [DWCP_V3_OPERATIONS.md](/docs/DWCP_V3_OPERATIONS.md)

### For Internet Deployments
1. ✅ Enable BBR: `sudo sysctl net.ipv4.tcp_congestion_control=bbr`
2. ✅ Configure PBFT: See [DWCP_V3_API_REFERENCE.md](/docs/DWCP_V3_API_REFERENCE.md)
3. ✅ Train ML models: [DWCP_V3_PERFORMANCE_TUNING.md](/docs/DWCP_V3_PERFORMANCE_TUNING.md)

### For Hybrid Deployments
1. ✅ Calibrate mode detection: [DWCP_V3_ARCHITECTURE.md](/docs/DWCP_V3_ARCHITECTURE.md)
2. ✅ Enable gradual rollout: [UPGRADE_GUIDE_V1_TO_V3.md](/docs/UPGRADE_GUIDE_V1_TO_V3.md)
3. ✅ Monitor mode transitions: [DWCP_V3_OPERATIONS.md](/docs/DWCP_V3_OPERATIONS.md)

---

## Troubleshooting

### Issue: "RDMA device not found"

```bash
# Install RDMA drivers
sudo apt-get install rdma-core libibverbs-dev

# Verify
ibv_devices
```

### Issue: "Mode detection not working"

```bash
# Check network metrics
dwcp debug network-metrics

# Adjust thresholds
dwcp config set mode-detection.latency-threshold=50ms
```

### Issue: "Low compression ratio"

```bash
# Enable ML compression
dwcp config set hde.ml_enabled=true

# Retrain model
dwcp ml retrain --component=compression-selector
```

---

## Support and Documentation

- **Architecture**: [DWCP_V3_ARCHITECTURE.md](/docs/DWCP_V3_ARCHITECTURE.md)
- **API Reference**: [DWCP_V3_API_REFERENCE.md](/docs/DWCP_V3_API_REFERENCE.md)
- **Operations**: [DWCP_V3_OPERATIONS.md](/docs/DWCP_V3_OPERATIONS.md)
- **Performance**: [DWCP_V3_PERFORMANCE_TUNING.md](/docs/DWCP_V3_PERFORMANCE_TUNING.md)
- **Upgrade Guide**: [UPGRADE_GUIDE_V1_TO_V3.md](/docs/UPGRADE_GUIDE_V1_TO_V3.md)

---

**Congratulations! Your DWCP v3 system is ready.** 🎉

You can now deploy VMs across datacenters, clouds, edge nodes, and volunteer compute with automatic optimization for each environment.
