# DWCP v1 → v3 Upgrade Guide

Comprehensive guide for upgrading from DWCP v1 to v3 with zero downtime.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Compatibility Matrix](#compatibility-matrix)
- [Upgrade Paths](#upgrade-paths)
- [Step-by-Step Instructions](#step-by-step-instructions)
- [Feature Flag Rollout](#feature-flag-rollout)
- [Rollback Procedures](#rollback-procedures)
- [Troubleshooting](#troubleshooting)
- [Migration Checklist](#migration-checklist)

## Overview

DWCP v3 introduces hybrid datacenter + internet capabilities while maintaining backward compatibility with v1. This guide provides a safe, incremental upgrade path using feature flags.

### What's New in v3

- **Hybrid Architecture**: Automatic mode detection and switching
- **Internet Optimization**: TCP transport with BBR, pacing, multi-streaming
- **ML-Based Compression**: Intelligent algorithm selection
- **CRDT Synchronization**: Conflict-free eventual consistency
- **Adaptive Consensus**: PBFT for internet, Raft for datacenter
- **Enhanced Prediction**: Separate models for datacenter/internet
- **Intelligent Placement**: Mode-aware VM placement with DQN

### Backward Compatibility

- ✅ v1 RDMA transport fully supported in datacenter mode
- ✅ v1 Delta encoding reused in HDE v3
- ✅ v1 LSTM predictor reused in PBA v3
- ✅ Existing APIs preserved with v3 extensions
- ✅ Wire protocol compatible with feature negotiation

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Go Version | 1.21 | 1.22+ |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Disk | 50 GB | 100+ GB SSD |
| Network | 1 Gbps | 10+ Gbps |

### Software Dependencies

```bash
# Required
go get go.uber.org/zap@latest
go get github.com/klauspost/compress/zstd@latest
go get github.com/pierrec/lz4/v4@latest

# Optional (for RDMA in datacenter mode)
# Install RDMA drivers and libraries
sudo apt-get install libibverbs-dev librdmacm-dev  # Debian/Ubuntu
sudo yum install libibverbs-devel librdmacm-devel  # RHEL/CentOS
```

### Pre-Upgrade Checklist

- [ ] DWCP v1 running in production
- [ ] Full backup of current system state
- [ ] Test environment matches production
- [ ] Monitoring and alerting configured
- [ ] Rollback plan documented
- [ ] Team trained on v3 features
- [ ] Downtime window scheduled (if needed)

## Compatibility Matrix

| v1 Feature | v3 Equivalent | Compatibility | Notes |
|------------|---------------|---------------|-------|
| RDMA Transport | AMST v3 Datacenter | ✅ Full | Reuses v1 implementation |
| Delta Encoding | HDE v3 Delta | ✅ Full | Extended with CRDT |
| LSTM Predictor | PBA v3 Datacenter | ✅ Full | Enhanced with internet model |
| Simple Partition | ITP v3 DQN | ⚠️ API Change | Upgrade to DQN-based placement |
| Basic Sync | ASS v3 Raft | ✅ Full | Extended with CRDT option |

**Legend:**
- ✅ Full: Drop-in compatible
- ⚠️ API Change: Requires code changes
- ❌ Deprecated: No longer supported

## Upgrade Paths

### Path 1: In-Place Upgrade (Recommended)

**Best for:** Single datacenter deployments with <1000 nodes

1. Deploy v3 binaries with v1 compatibility mode
2. Enable feature flags incrementally (0% → 10% → 50% → 100%)
3. Monitor performance and stability
4. Complete rollout over 2-4 weeks

**Downtime:** Zero (rolling upgrade)

### Path 2: Blue-Green Deployment

**Best for:** Multi-datacenter deployments with strict SLA

1. Set up v3 environment (green)
2. Replicate data from v1 (blue)
3. Switch traffic incrementally
4. Keep v1 as fallback for 30 days

**Downtime:** ~1 hour (DNS cutover)

### Path 3: Parallel Migration

**Best for:** Internet-scale deployments with geographic distribution

1. Deploy v3 in new regions/zones
2. Onboard new workloads to v3
3. Migrate existing workloads gradually
4. Decommission v1 after 90 days

**Downtime:** Zero (gradual migration)

## Step-by-Step Instructions

### Phase 1: Preparation (Week 1)

#### 1.1 Backup Current System

```bash
# Backup DWCP v1 configuration
tar -czf dwcp-v1-config-$(date +%Y%m%d).tar.gz /etc/dwcp/

# Backup state databases
dwcp-cli export --output dwcp-v1-state-$(date +%Y%m%d).json

# Backup VM snapshots
dwcp-cli snapshot-all --output /backup/vm-snapshots/
```

#### 1.2 Deploy v3 in Test Environment

```bash
# Clone v3 repository
git clone https://github.com/khryptorgraphics/novacron.git
cd novacron/backend/core/network/dwcp/v3

# Build v3 components
make build

# Run integration tests
make test-integration

# Deploy to test environment
./scripts/deploy-test.sh
```

#### 1.3 Validate v3 Functionality

```bash
# Test mode detection
dwcp-v3-cli mode-detect

# Test transport layer
dwcp-v3-cli transport-test --mode hybrid

# Test encoding layer
dwcp-v3-cli encoding-test --enable-ml

# Test prediction
dwcp-v3-cli prediction-test --mode internet

# Run full validation suite
./scripts/validate-v3.sh
```

### Phase 2: Staged Rollout (Weeks 2-4)

#### 2.1 Deploy v3 with Feature Flags Disabled (Week 2)

```bash
# Deploy v3 binaries with v1 compatibility
kubectl apply -f deployments/dwcp-v3-compat.yaml

# Verify v3 is running in v1-compatible mode
dwcp-v3-cli status --check-compatibility

# Monitor for any regressions
kubectl logs -l app=dwcp-v3 -f
```

**Configuration:**

```yaml
# /etc/dwcp/v3-config.yaml
feature_flags:
  hybrid_mode: false           # Disabled - use v1 mode only
  ml_compression: false        # Disabled - use v1 compression
  crdt_sync: false            # Disabled - use v1 sync
  pbft_consensus: false       # Disabled - use v1 consensus
  internet_optimization: false # Disabled - datacenter only
```

#### 2.2 Enable Hybrid Mode at 10% (Week 2)

```bash
# Update feature flags to 10% rollout
dwcp-v3-cli feature-flags set hybrid_mode --percent 10

# Monitor performance
dwcp-v3-cli metrics --component amst --watch

# Check for mode transitions
dwcp-v3-cli events --filter mode_transition
```

**Expected Impact:**
- 10% of traffic uses mode detection
- 90% remains on v1 RDMA
- Latency increase: <5%
- Throughput change: <2%

**Rollback Criteria:**
- Latency increase >10%
- Error rate >0.5%
- Mode detection failure rate >5%

#### 2.3 Increase to 50% (Week 3)

```bash
# Increase hybrid mode to 50%
dwcp-v3-cli feature-flags set hybrid_mode --percent 50

# Enable ML compression at 10%
dwcp-v3-cli feature-flags set ml_compression --percent 10

# Monitor compression ratios
dwcp-v3-cli metrics --component hde --watch
```

**Expected Impact:**
- 50% traffic adaptive mode switching
- Compression ratio improvement: 15-30%
- Bandwidth savings: 10-25%

**Validation Checks:**
```bash
# Check compression effectiveness
dwcp-v3-cli compression-stats

# Verify no data corruption
dwcp-v3-cli data-integrity-check

# Monitor delta encoding hit rate
dwcp-v3-cli metrics hde.delta_hit_rate
```

#### 2.4 Full Rollout at 100% (Week 4)

```bash
# Enable all v3 features at 100%
dwcp-v3-cli feature-flags set hybrid_mode --percent 100
dwcp-v3-cli feature-flags set ml_compression --percent 100
dwcp-v3-cli feature-flags set crdt_sync --percent 100
dwcp-v3-cli feature-flags set pbft_consensus --percent 100
dwcp-v3-cli feature-flags set internet_optimization --percent 100

# Verify all features active
dwcp-v3-cli status --verbose
```

**Final Configuration:**

```yaml
# /etc/dwcp/v3-config.yaml
feature_flags:
  hybrid_mode: true
  ml_compression: true
  crdt_sync: true
  pbft_consensus: true
  internet_optimization: true

network_mode: hybrid  # Auto-detect and switch

amst_v3:
  enable_datacenter: true
  enable_internet: true
  auto_mode: true
  datacenter_streams: 64
  internet_streams: 8
  congestion_algorithm: bbr
  pacing_enabled: true

hde_v3:
  enable_ml_compression: true
  enable_delta_encoding: true
  enable_crdt: true

pba_v3:
  enable_hybrid_mode: true
  datacenter_accuracy_target: 0.85
  internet_accuracy_target: 0.70

ass_v3:
  enable_adaptive_sync: true
  datacenter_target_latency: 100ms
  internet_target_latency: 5s

acp_v3:
  enable_adaptive_consensus: true
  datacenter_algorithm: raft
  internet_algorithm: pbft

itp_v3:
  enable_dqn_placement: true
  enable_geographic_optimization: true
```

### Phase 3: Validation and Optimization (Week 5)

#### 3.1 Performance Validation

```bash
# Run comprehensive benchmarks
dwcp-v3-cli benchmark --duration 1h --workload production

# Compare v1 vs v3 performance
dwcp-v3-cli compare-baseline --v1-metrics v1-baseline.json

# Generate performance report
dwcp-v3-cli report --output performance-v3-$(date +%Y%m%d).pdf
```

**Key Metrics to Validate:**

| Metric | v1 Baseline | v3 Target | Threshold |
|--------|-------------|-----------|-----------|
| Datacenter Latency | 10-50ms | <50ms | <100ms |
| Internet Latency | 100-500ms | <500ms | <1s |
| Bandwidth (DC) | 10-40 Gbps | >30 Gbps | >20 Gbps |
| Bandwidth (Internet) | 100-500 Mbps | >300 Mbps | >150 Mbps |
| Compression Ratio | 2-3x | >3x | >2.5x |
| Prediction Accuracy (DC) | 80-85% | >85% | >80% |
| Prediction Accuracy (Int) | 60-70% | >70% | >65% |
| Consensus Time (DC) | 50-100ms | <100ms | <150ms |
| Consensus Time (Int) | 1-3s | <3s | <5s |

#### 3.2 Stability Validation

```bash
# Run 7-day soak test
dwcp-v3-cli soak-test --duration 168h --monitor-interval 5m

# Check for memory leaks
dwcp-v3-cli memory-profile --duration 24h

# Verify no resource leaks
dwcp-v3-cli resource-check --components all

# Test failure scenarios
dwcp-v3-cli chaos-test --scenarios network-partition,node-failure
```

#### 3.3 Cleanup v1 Legacy Code

```bash
# Remove v1 compatibility shims (after 30 days of v3 stability)
dwcp-v3-cli remove-v1-compat

# Clean up old baselines
dwcp-v3-cli cleanup-baselines --older-than 30d

# Archive v1 metrics
dwcp-v3-cli archive-metrics --version v1 --output /archive/
```

## Feature Flag Rollout

### Feature Flag Configuration

```go
// Feature flags are controlled via configuration
type FeatureFlags struct {
    HybridMode           FeatureFlagConfig `json:"hybrid_mode"`
    MLCompression        FeatureFlagConfig `json:"ml_compression"`
    CRDTSync             FeatureFlagConfig `json:"crdt_sync"`
    PBFTConsensus        FeatureFlagConfig `json:"pbft_consensus"`
    InternetOptimization FeatureFlagConfig `json:"internet_optimization"`
}

type FeatureFlagConfig struct {
    Enabled        bool    `json:"enabled"`
    RolloutPercent float64 `json:"rollout_percent"`  // 0-100
    AllowList      []string `json:"allow_list"`     // Node IDs
    DenyList       []string `json:"deny_list"`      // Node IDs
}
```

### Rollout Schedule

| Week | Hybrid Mode | ML Compression | CRDT Sync | PBFT Consensus | Internet Opt |
|------|-------------|----------------|-----------|----------------|--------------|
| 1    | 0%          | 0%             | 0%        | 0%             | 0%           |
| 2    | 10%         | 0%             | 0%        | 0%             | 0%           |
| 3    | 50%         | 10%            | 10%       | 0%             | 10%          |
| 4    | 100%        | 50%            | 50%       | 10%            | 50%          |
| 5    | 100%        | 100%           | 100%      | 50%            | 100%         |
| 6    | 100%        | 100%           | 100%      | 100%           | 100%         |

### Per-Node Rollout

```bash
# Enable for specific nodes first (canary)
dwcp-v3-cli feature-flags set hybrid_mode \
    --enabled true \
    --allow-list node-1,node-2,node-3

# Monitor canary nodes
dwcp-v3-cli metrics --nodes node-1,node-2,node-3 --watch

# Expand to percentage rollout
dwcp-v3-cli feature-flags set hybrid_mode --percent 10
```

## Rollback Procedures

### Automatic Rollback Triggers

v3 includes automatic rollback triggers for safety:

```yaml
rollback_triggers:
  max_error_rate: 1.0          # Rollback if error rate >1%
  max_latency_increase: 50.0   # Rollback if latency increases >50%
  min_throughput_ratio: 0.8    # Rollback if throughput <80% of baseline
  max_mode_switch_rate: 10     # Rollback if mode switches >10/min
```

### Manual Rollback Steps

#### Immediate Rollback (Emergency)

```bash
# Disable all v3 features immediately
dwcp-v3-cli feature-flags disable-all

# Revert to v1 compatibility mode
dwcp-v3-cli set-mode v1-compat

# Verify v1 mode active
dwcp-v3-cli status --check-mode

# Expected: All components in v1-compatible mode
```

**Expected Recovery Time:** <5 minutes

#### Gradual Rollback (Controlled)

```bash
# Reduce feature flag percentages gradually
dwcp-v3-cli feature-flags set hybrid_mode --percent 50  # From 100%
# Wait 10 minutes, monitor
dwcp-v3-cli feature-flags set hybrid_mode --percent 10  # From 50%
# Wait 10 minutes, monitor
dwcp-v3-cli feature-flags set hybrid_mode --percent 0   # Disable

# Disable other features
dwcp-v3-cli feature-flags set ml_compression --percent 0
dwcp-v3-cli feature-flags set crdt_sync --percent 0
```

**Expected Recovery Time:** 30-60 minutes

#### Full Downgrade to v1

```bash
# Export v3 state
dwcp-v3-cli export-state --output v3-state-backup.json

# Stop v3 services
systemctl stop dwcp-v3

# Restore v1 binaries
./scripts/restore-v1-binaries.sh

# Import compatible state
dwcp-cli import-state --input v3-state-compat.json

# Start v1 services
systemctl start dwcp-v1

# Verify v1 operational
dwcp-cli status
```

**Expected Recovery Time:** 1-2 hours

### Rollback Validation

```bash
# After rollback, verify:

# 1. All nodes operational
dwcp-cli node-status --all

# 2. VMs running correctly
dwcp-cli vm-status --all

# 3. Performance restored
dwcp-cli metrics --baseline v1-baseline.json

# 4. No data loss
dwcp-cli data-integrity-check
```

## Troubleshooting

### Common Issues

#### Issue 1: Mode Detection Incorrect

**Symptoms:**
- Frequent mode transitions
- Performance degradation
- High CPU usage

**Diagnosis:**
```bash
# Check mode detection logs
dwcp-v3-cli logs --component mode-detector --level debug

# Check network conditions
dwcp-v3-cli network-probe --verbose

# Check RDMA availability
ibv_devinfo
```

**Solution:**
```bash
# Adjust mode detection thresholds
dwcp-v3-cli config set mode_detector.latency_threshold 100ms
dwcp-v3-cli config set mode_detector.jitter_threshold 10ms

# Or disable auto-mode and set manually
dwcp-v3-cli config set amst.auto_mode false
dwcp-v3-cli config set amst.mode datacenter
```

#### Issue 2: Compression Ratio Lower Than Expected

**Symptoms:**
- Compression ratio <2x
- High bandwidth usage
- Slow transfers

**Diagnosis:**
```bash
# Check compression algorithm selection
dwcp-v3-cli metrics hde.algorithm_usage

# Check delta encoding effectiveness
dwcp-v3-cli metrics hde.delta_hit_rate

# Check ML selector performance
dwcp-v3-cli compression-analyze --sample 1000
```

**Solution:**
```bash
# Enable delta encoding
dwcp-v3-cli config set hde.enable_delta_encoding true

# Increase baseline refresh interval
dwcp-v3-cli config set hde.baseline_refresh_interval 10m

# Tune ML selector
dwcp-v3-cli config set hde.selector.learning_rate 0.01
```

#### Issue 3: Consensus Timeouts

**Symptoms:**
- Consensus failures
- State inconsistencies
- Slow migrations

**Diagnosis:**
```bash
# Check consensus metrics
dwcp-v3-cli metrics acp.consensus_time

# Check PBFT replica status
dwcp-v3-cli pbft-status --verbose

# Check network latency between replicas
dwcp-v3-cli network-matrix --nodes all
```

**Solution:**
```bash
# Increase consensus timeout
dwcp-v3-cli config set acp.consensus_timeout 10s

# Adjust PBFT batch size
dwcp-v3-cli config set acp.pbft.batch_size 100

# Switch to Raft if in trusted environment
dwcp-v3-cli config set acp.force_raft true
```

#### Issue 4: High Memory Usage

**Symptoms:**
- OOM errors
- Slow performance
- Memory leaks

**Diagnosis:**
```bash
# Check memory usage by component
dwcp-v3-cli memory-usage --breakdown

# Profile memory allocation
dwcp-v3-cli pprof --type heap --duration 5m

# Check for baseline accumulation
dwcp-v3-cli baselines-count
```

**Solution:**
```bash
# Reduce maximum baselines
dwcp-v3-cli config set hde.max_baselines 500

# Reduce history buffer sizes
dwcp-v3-cli config set pba.datacenter_sequence_length 5
dwcp-v3-cli config set pba.internet_sequence_length 30

# Enable aggressive cleanup
dwcp-v3-cli config set hde.baseline_cleanup_aggressive true
```

### Support Resources

- **GitHub Issues**: https://github.com/khryptorgraphics/novacron/issues
- **Documentation**: `/docs`
- **Logs**: `/var/log/dwcp/v3/`
- **Metrics**: Prometheus endpoint `:9090/metrics`

## Migration Checklist

Use this checklist to track your upgrade progress:

### Pre-Migration

- [ ] Review upgrade guide completely
- [ ] Check system meets prerequisites
- [ ] Backup all v1 data and configuration
- [ ] Set up test environment matching production
- [ ] Test v3 in test environment
- [ ] Train team on v3 features and operations
- [ ] Document rollback procedures
- [ ] Schedule upgrade window
- [ ] Notify stakeholders

### Week 1: Preparation

- [ ] Deploy v3 binaries in v1-compat mode
- [ ] Verify v3 running with all features disabled
- [ ] Configure monitoring and alerting
- [ ] Run baseline performance tests
- [ ] Validate rollback procedures work

### Week 2: Initial Rollout

- [ ] Enable hybrid mode at 10%
- [ ] Monitor for 48 hours
- [ ] Check metrics against baselines
- [ ] Resolve any issues found
- [ ] Document lessons learned

### Week 3: Expanded Rollout

- [ ] Increase hybrid mode to 50%
- [ ] Enable ML compression at 10%
- [ ] Monitor for 72 hours
- [ ] Validate compression improvements
- [ ] Check for any regressions

### Week 4: Full Rollout

- [ ] Enable all features at 100%
- [ ] Run comprehensive performance tests
- [ ] Validate all metrics meet targets
- [ ] Check stability over 7 days
- [ ] Document final configuration

### Week 5+: Validation

- [ ] Run soak tests for 7 days
- [ ] Validate no memory leaks
- [ ] Test failure scenarios
- [ ] Generate upgrade report
- [ ] Archive v1 data (after 30 days)
- [ ] Remove v1 compatibility code (after 60 days)
- [ ] Mark upgrade complete

## Expected Downtime and Impact

### Downtime Summary

| Upgrade Path | Downtime | Traffic Impact | Data Loss Risk |
|--------------|----------|----------------|----------------|
| In-Place (Recommended) | 0 minutes | None | None |
| Blue-Green | 60 minutes | 5-10% during cutover | None |
| Parallel Migration | 0 minutes | None | None |

### Performance Impact During Rollout

**Week 2 (10% hybrid mode):**
- Latency: +3-5%
- Throughput: -1-2%
- CPU Usage: +5-8%

**Week 3 (50% hybrid mode, 10% ML compression):**
- Latency: +5-10%
- Throughput: +5-10% (compression savings)
- CPU Usage: +10-15%

**Week 4+ (100% all features):**
- Latency: -10-20% (optimizations)
- Throughput: +20-35% (compression + optimization)
- CPU Usage: +15-20% (ML models)

## Conclusion

Following this guide ensures a safe, incremental upgrade from DWCP v1 to v3 with:
- Zero downtime for most deployments
- Automatic rollback on issues
- Comprehensive monitoring and validation
- Clear success criteria

For questions or issues, refer to the troubleshooting section or contact support.
