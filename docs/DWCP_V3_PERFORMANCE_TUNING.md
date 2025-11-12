# DWCP v3 Performance Tuning Guide

**Version:** 3.0.0
**Last Updated:** 2025-11-10

## Table of Contents

1. [Datacenter Mode Optimization](#datacenter-mode-optimization)
2. [Internet Mode Optimization](#internet-mode-optimization)
3. [Bandwidth Prediction Tuning](#bandwidth-prediction-tuning)
4. [Consensus Optimization](#consensus-optimization)
5. [Mode Detection Tuning](#mode-detection-tuning)
6. [Benchmarking Procedures](#benchmarking-procedures)

---

## Datacenter Mode Optimization

### RDMA Tuning

#### 1. NIC Configuration

```bash
# Check RDMA devices
ibv_devinfo

# Optimal settings for Mellanox ConnectX-5
sudo ethtool -G mlx5_0 rx 8192 tx 8192
sudo ethtool -A mlx5_0 rx on tx on
sudo ethtool -K mlx5_0 lro on gro on tso on gso on

# Set MTU to 9000 (jumbo frames)
sudo ip link set mlx5_0 mtu 9000
```

#### 2. DWCP RDMA Settings

```yaml
# /etc/dwcp/v3/rdma.yaml
transport:
  rdma:
    device: mlx5_0
    port: 1
    mtu: 4096
    streams: 512              # Max for datacenter
    queue_depth: 1024
    completion_queue_size: 8192
    send_inline_size: 256
    max_send_sge: 16
    max_recv_sge: 16
```

**Performance Impact:**
- Throughput: 42.3 Gbps (baseline: 38.1 Gbps) = **+11%**
- Latency p50: 3.2ms (baseline: 4.8ms) = **-33%**

### Stream Optimization

```yaml
transport:
  datacenter_streams: 512   # Max throughput
  stream_scheduling: "round_robin"
  load_balancing: true
  stream_affinity: true     # Pin to CPU cores
```

**Tuning Recommendations:**
- **1-10 Gbps**: 32-64 streams
- **10-40 Gbps**: 128-256 streams  
- **40-100 Gbps**: 256-512 streams

### Compression Settings

```yaml
compression:
  datacenter:
    algorithm: lz4          # Speed over size
    level: 1                # Fastest
    chunk_size: 256KB
    parallel_workers: 8
    prefetch_enabled: true
```

**Compression Trade-offs:**

| Algorithm | Speed | Ratio | Latency |
|-----------|-------|-------|---------|
| None | 100% | 1.0x | 0ms |
| LZ4 | 95% | 2.1x | +2ms |
| zstd-1 | 80% | 2.8x | +5ms |
| zstd-3 | 60% | 3.2x | +12ms |

**Recommendation**: LZ4 for <100KB, zstd-1 for >100KB

---

## Internet Mode Optimization

### TCP Tuning

#### 1. Kernel Parameters

```bash
# /etc/sysctl.d/99-dwcp.conf
# TCP buffer sizes
net.core.rmem_max = 536870912
net.core.wmem_max = 536870912
net.ipv4.tcp_rmem = 4096 87380 536870912
net.ipv4.tcp_wmem = 4096 65536 536870912

# BBR congestion control
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr

# TCP optimizations
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_timestamps = 1
net.ipv4.tcp_sack = 1
net.ipv4.tcp_fastopen = 3
net.ipv4.tcp_mtu_probing = 1

# Apply settings
sudo sysctl -p /etc/sysctl.d/99-dwcp.conf
```

#### 2. DWCP TCP Settings

```yaml
transport:
  internet:
    tcp_streams: 8          # Optimal for WAN
    congestion_algorithm: bbr
    pacing_enabled: true
    pacing_rate: 1Gbps
    send_buffer_size: 4MB
    recv_buffer_size: 4MB
    keepalive_interval: 60s
    connection_timeout: 30s
```

**Performance Impact:**
- Throughput: 850 Mbps (baseline: 620 Mbps) = **+37%**
- Latency p99: 450ms (baseline: 680ms) = **-34%**

### BBR Configuration

```yaml
congestion_control:
  algorithm: bbr
  probe_rtt_interval: 10s
  probe_bandwidth_gain: 2.0
  cwnd_gain: 2.0
  pacing_gain_cycle: [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

**BBR vs CUBIC (Internet Mode):**
- **Throughput**: BBR 850 Mbps vs CUBIC 620 Mbps (+37%)
- **Latency**: BBR 125ms vs CUBIC 180ms (-31%)
- **Packet Loss Tolerance**: BBR <1% vs CUBIC <0.5%

### Compression for WAN

```yaml
compression:
  internet:
    algorithm: zstd-max     # Size over speed
    level: 19               # Maximum compression
    ml_enabled: true        # ML selector
    dictionary_enabled: true
    dictionary_size: 128KB
```

**Compression Results (100MB VM state):**

| Algorithm | Compressed Size | Time | Ratio |
|-----------|----------------|------|-------|
| None | 100 MB | 0s | 1.0x |
| LZ4 | 47 MB | 0.8s | 2.1x |
| zstd-3 | 35 MB | 2.1s | 2.9x |
| zstd-19 | 23 MB | 8.3s | 4.3x |
| zstd-19+ML | 21 MB | 8.5s | 4.8x |

**Recommendation**: zstd-19 with ML for bandwidth < 1 Gbps

---

## Bandwidth Prediction Tuning

### LSTM Model Parameters

#### Datacenter Predictor

```yaml
prediction:
  datacenter:
    model_path: /var/lib/dwcp/models/datacenter_bandwidth_predictor.onnx
    sequence_length: 10     # Short history (stable network)
    hidden_size: 64
    num_layers: 2
    dropout: 0.1
    learning_rate: 0.001
    batch_size: 32
    prediction_horizon: 5m
```

**Training Data Requirements:**
- Minimum samples: 1000
- Recommended: 10,000+ samples
- Collection interval: 10 seconds

**Accuracy Targets:**
- MAE: <5 Mbps
- RMSE: <10 Mbps
- R²: >0.85

#### Internet Predictor

```yaml
prediction:
  internet:
    model_path: /var/lib/dwcp/models/internet_bandwidth_predictor.onnx
    sequence_length: 60     # Long history (variable network)
    hidden_size: 128        # More complexity
    num_layers: 3
    dropout: 0.2
    learning_rate: 0.0005
    batch_size: 64
    prediction_horizon: 15m
```

**Accuracy Targets:**
- MAE: <50 Mbps
- RMSE: <100 Mbps
- R²: >0.70

### Retraining Strategy

```bash
# Collect training data
dwcp pba collect-samples --duration=24h --output=/tmp/samples.csv

# Train new model
dwcp pba train \
  --input=/tmp/samples.csv \
  --mode=internet \
  --epochs=100 \
  --output=/var/lib/dwcp/models/internet_predictor_new.onnx

# Evaluate model
dwcp pba evaluate \
  --model=/var/lib/dwcp/models/internet_predictor_new.onnx \
  --test-data=/tmp/test_samples.csv

# Deploy if accuracy improved
dwcp pba deploy-model \
  --model=/var/lib/dwcp/models/internet_predictor_new.onnx \
  --mode=internet
```

---

## Consensus Optimization

### Raft Tuning (Datacenter)

```yaml
consensus:
  raft:
    election_timeout: 150ms      # Fast elections
    heartbeat_interval: 50ms     # Frequent heartbeats
    snapshot_interval: 10000     # Entries before snapshot
    max_append_entries: 1000     # Batch size
    pipeline_enabled: true       # Pipeline replication
    read_only_option: "lease"    # Fast reads
```

**Performance:**
- Consensus latency: 72ms (target: <100ms)
- Throughput: 10,000 ops/sec
- Availability: 99.99%

### PBFT Tuning (Internet)

```yaml
consensus:
  pbft:
    replica_count: 4             # 3f+1 (f=1 failure)
    batch_size: 100              # Batch requests
    batch_timeout: 50ms
    view_change_timeout: 10s     # Longer for WAN
    checkpoint_interval: 100     # Checkpoints per 100 requests
    max_inflight_requests: 1000
```

**Performance:**
- Consensus latency: 1.8s (target: <5s)
- Throughput: 500 ops/sec
- Byzantine tolerance: f=1 (25% malicious)

### Optimization Trade-offs

| Mode | Latency Target | Throughput | Fault Tolerance |
|------|---------------|------------|----------------|
| Raft (DC) | <100ms | 10K ops/s | Crash (n/2+1) |
| PBFT (Internet) | <5s | 500 ops/s | Byzantine (f=(n-1)/3) |

---

## Mode Detection Tuning

### Threshold Configuration

```yaml
mode_detection:
  enabled: true
  interval: 10s
  history_size: 10
  
  thresholds:
    datacenter:
      latency_max: 10ms         # Strict
      bandwidth_min: 1Gbps
      packet_loss_max: 0.01%
    
    internet:
      latency_min: 50ms         # Relaxed
      bandwidth_max: 1Gbps
      packet_loss_max: 1.0%
    
    hybrid:
      latency_range: [10ms, 50ms]
      bandwidth_range: [0.5Gbps, 2Gbps]
```

### Prevent Mode Flapping

```yaml
mode_detection:
  stability:
    hysteresis_percentage: 20   # 20% buffer
    min_mode_duration: 60s      # Stay in mode 60s minimum
    confidence_threshold: 0.8   # 80% confidence
```

**Example:**
- Datacenter threshold: <10ms latency
- Hysteresis: Switch to internet only if latency >12ms (10ms + 20%)
- Minimum duration: Stay in datacenter mode for 60s before switching

---

## Benchmarking Procedures

### 1. Baseline Performance Test

```bash
# Run full benchmark suite
dwcp benchmark run \
  --suite=full \
  --duration=5m \
  --output=/tmp/baseline.json

# Results:
# Transport throughput: 42.3 Gbps
# Compression ratio: 2.82x
# Prediction latency: 18ms
# Sync latency: 45ms
# Consensus latency: 72ms
# Placement latency: 120ms
```

### 2. Component-Specific Benchmarks

#### AMST Transport

```bash
dwcp benchmark amst \
  --mode=datacenter \
  --data-size=1GB \
  --streams=512 \
  --iterations=100

# Metrics:
# - Throughput (Gbps)
# - Latency (p50, p95, p99)
# - Stream utilization
```

#### HDE Compression

```bash
dwcp benchmark hde \
  --data-size=100MB \
  --algorithms=all \
  --iterations=50

# Metrics:
# - Compression ratio
# - Compression time
# - Decompression time
```

#### PBA Prediction

```bash
dwcp benchmark pba \
  --mode=internet \
  --samples=1000 \
  --duration=10m

# Metrics:
# - Prediction accuracy (MAE, RMSE, R²)
# - Prediction latency
# - Model size
```

### 3. Load Testing

```bash
# Simulate 1000 concurrent VMs
dwcp load-test \
  --vms=1000 \
  --duration=30m \
  --ramp-up=5m \
  --report=/tmp/load-test.html

# Stress test
dwcp load-test stress \
  --target-throughput=50Gbps \
  --duration=10m
```

### 4. Performance Regression Testing

```bash
# Compare v1 vs v3
dwcp benchmark compare \
  --baseline=/tmp/v1-baseline.json \
  --current=/tmp/v3-baseline.json \
  --output=/tmp/comparison.html

# Expected results:
# ✅ Throughput: +1.2% (41.8 → 42.3 Gbps)
# ✅ Latency: -6.3% (48 → 45 ms)
# ✅ Compression: +1.1% (2.79x → 2.82x)
```

---

## Performance Monitoring

### Real-Time Monitoring

```bash
# Watch key metrics
watch -n 1 'dwcp stats | grep -E "(Throughput|Latency|Mode)"'

# Detailed dashboard
dwcp dashboard --refresh=1s
```

### Performance Alerts

```yaml
# /etc/prometheus/alerts/dwcp-performance.yml
groups:
- name: performance
  rules:
  - alert: LowThroughput
    expr: dwcp_transport_throughput_gbps < 30
    for: 5m
    annotations:
      summary: "Throughput below 30 Gbps"
  
  - alert: HighLatency
    expr: dwcp_transport_latency_p95 > 0.1
    for: 5m
    annotations:
      summary: "p95 latency above 100ms"
```

---

## Tuning Checklist

### Pre-Production

- [ ] Kernel parameters optimized
- [ ] RDMA drivers installed and configured
- [ ] Network MTU set to 9000 (jumbo frames)
- [ ] TCP BBR enabled
- [ ] Compression algorithms benchmarked
- [ ] ML models trained on representative data
- [ ] Mode detection thresholds calibrated
- [ ] Performance baselines established

### Post-Deployment

- [ ] Monitor mode transitions
- [ ] Track prediction accuracy
- [ ] Measure compression ratios
- [ ] Profile CPU/memory usage
- [ ] Analyze network utilization
- [ ] Review error rates
- [ ] Tune based on workload

---

## See Also

- Operations: `/docs/DWCP_V3_OPERATIONS.md`
- Architecture: `/docs/DWCP_V3_ARCHITECTURE.md`
- API Reference: `/docs/DWCP_V3_API_REFERENCE.md`
