# DWCP v3 Architecture Documentation

**Version:** 3.0.0
**Last Updated:** 2025-11-10
**Target Audience:** Architects, Senior Engineers, Technical Leadership

## Table of Contents

1. [Overview](#overview)
2. [Hybrid Architecture](#hybrid-architecture)
3. [Component Architecture](#component-architecture)
4. [Mode Detection Algorithm](#mode-detection-algorithm)
5. [Feature Flag System](#feature-flag-system)
6. [Performance Characteristics](#performance-characteristics)
7. [Comparison with v1](#comparison-with-v1)

---

## Overview

DWCP v3 is a **hybrid datacenter + internet distributed compute protocol** that automatically adapts to network conditions. It extends v1's datacenter-only RDMA implementation with internet-optimized components while maintaining 100% backward compatibility.

### Design Principles

1. **Backward Compatibility**: All v1 APIs and data formats work unchanged
2. **Adaptive Optimization**: Auto-detect network conditions and select optimal algorithms
3. **Mode-Aware**: Different strategies for datacenter vs internet deployments
4. **Gradual Migration**: Feature flags enable safe, incremental rollout

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DWCP v3 System                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           Mode Detector (Auto-Select)                │  │
│  │  Latency < 10ms AND Bandwidth > 1Gbps → Datacenter   │  │
│  │  Latency > 50ms OR Bandwidth < 1Gbps → Internet     │  │
│  │  Otherwise → Hybrid                                   │  │
│  └─────────────────────────────────────────────────────┘  │
│                          ↓                                  │
│  ┌──────────────────┬──────────────────┬────────────────┐ │
│  │ Datacenter Mode  │   Internet Mode   │  Hybrid Mode   │ │
│  │  (v1 Compatible) │  (v3 Enhanced)   │  (Adaptive)    │ │
│  └──────────────────┴──────────────────┴────────────────┘ │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Component Layer (6 Components)                       │ │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┐             │ │
│  │  │AMST │ HDE │ PBA │ ASS │ ACP │ ITP │             │ │
│  │  │Trans│Encod│Pred │Sync │Cons │Place│             │ │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┘             │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Infrastructure Layer                                 │ │
│  │  RDMA | TCP | Raft | PBFT | CRDT | ML Models        │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Hybrid Architecture

### Three Operation Modes

#### 1. Datacenter Mode
- **Target**: Low-latency (<10ms), high-bandwidth (>10 Gbps) networks
- **Transport**: RDMA (InfiniBand/RoCEv2)
- **Compression**: LZ4/zstd (speed-optimized)
- **Sync**: Raft (<100ms consensus)
- **Placement**: DQN (performance-optimized)

#### 2. Internet Mode
- **Target**: High-latency (50-500ms), variable bandwidth (100-900 Mbps)
- **Transport**: TCP with BBR congestion control
- **Compression**: zstd-max + ML selection (size-optimized)
- **Sync**: CRDT (5-30s eventual consistency)
- **Placement**: Geographic (reliability-optimized)
- **Consensus**: PBFT (Byzantine-tolerant)

#### 3. Hybrid Mode
- **Auto-Switching**: Dynamic mode selection based on real-time conditions
- **Fallback**: Graceful degradation (Raft→CRDT, RDMA→TCP)
- **Ensemble**: Combine datacenter + internet predictors

### Mode Detection Algorithm

```go
func DetectMode(latency time.Duration, bandwidth int64) NetworkMode {
    // Datacenter: low latency AND high bandwidth
    if latency < 10ms && bandwidth >= 1Gbps {
        return ModeDatacenter
    }
    
    // Internet: high latency OR low bandwidth
    if latency > 50ms || bandwidth < 1Gbps {
        return ModeInternet
    }
    
    // Borderline: use hybrid
    return ModeHybrid
}
```

---

## Component Architecture

### 1. AMST v3: Adaptive Multi-Stream Transport

```
AMSTv3
├── Datacenter Transport (v1 RDMA)
│   ├── RDMA connections (32-512 streams)
│   ├── CUBIC congestion control
│   └── Zero-copy transfers
├── Internet Transport (v3 TCP)
│   ├── TCP connections (4-16 streams)
│   ├── BBR congestion control
│   └── Packet pacing
└── Congestion Controller
    ├── BBR: Bandwidth estimation
    ├── CUBIC: Loss-based control
    └── Adaptive stream count
```

**Key Features**:
- Automatic transport selection
- Graceful RDMA→TCP fallback
- Mode-aware stream count (512 for DC, 8 for Internet)
- Progress callbacks for large transfers

### 2. HDE v3: Hierarchical Delta Encoding

```
HDEv3
├── ML Compression Selector
│   ├── Model: Logistic Regression
│   ├── Features: Size, entropy, compressibility
│   └── Algorithms: None, LZ4, zstd, zstd-max
├── Delta Encoder (v1 reused)
│   ├── Rolling hash (Rabin-Karp)
│   ├── Baseline management
│   └── 3-5x compression for similar data
└── CRDT Integration
    ├── LWW Register
    ├── Vector clocks
    └── Conflict-free merge
```

**Compression Selection Logic**:
```
Datacenter Mode:
  Size < 100KB → LZ4 (speed)
  Size >= 100KB → zstd (balanced)

Internet Mode:
  All sizes → zstd-max (size)

Hybrid Mode:
  ML model selects optimal algorithm
```

### 3. PBA v3: Predictive Bandwidth Allocation

```
PBAv3
├── Datacenter Predictor (v1 LSTM)
│   ├── Sequence length: 10 samples
│   ├── Horizon: 5 minutes
│   └── Target accuracy: 85%
├── Internet Predictor (v3 LSTM)
│   ├── Sequence length: 60 samples
│   ├── Horizon: 15 minutes
│   └── Target accuracy: 70%
└── Ensemble Predictor
    ├── Confidence-weighted average
    ├── Auto-switching based on mode
    └── Historical performance tracking
```

**Prediction Model**:
```
Input: [bandwidth, latency, packet_loss, jitter] × sequence_length
Hidden: LSTM(128 units) → LSTM(64 units)
Output: [predicted_bandwidth, predicted_latency, confidence]
```

### 4. ASS v3: Async State Synchronization

```
ASSv3
├── Raft Sync (Datacenter)
│   ├── Leader election
│   ├── Log replication
│   └── Target: <100ms latency
├── CRDT Sync (Internet)
│   ├── LWW Register / OR-Set
│   ├── Vector clocks
│   └── Target: 5-30s convergence
└── Hybrid Sync
    ├── Try Raft first (100ms timeout)
    ├── Fallback to CRDT
    └── Conflict resolution
```

### 5. ACP v3: Adaptive Consensus Protocol

```
ACPv3
├── Raft Consensus (Datacenter)
│   ├── 3-5 node clusters
│   ├── <100ms commit latency
│   └── Assumes trusted network
├── PBFT Consensus (Internet)
│   ├── Byzantine fault tolerance
│   ├── 3f+1 nodes (f failures)
│   ├── 1-5s commit latency
│   └── Pre-prepare → Prepare → Commit
└── Gossip Protocol
    ├── Peer discovery
    ├── Fanout: 3 peers
    └── Interval: 5 seconds
```

### 6. ITP v3: Intelligent Task Placement

```
ITPv3
├── DQN Placement (Datacenter)
│   ├── State: Node resources, VM requirements
│   ├── Action: Select node
│   ├── Reward: Resource utilization, latency
│   └── Optimization: Performance (bin packing)
├── Geographic Placement (Internet)
│   ├── Constraints: Latency, uptime, cost
│   ├── Algorithm: Multi-objective optimization
│   └── Optimization: Reliability, distribution
└── Hybrid Placement
    ├── Adaptive selection based on VM type
    ├── Affinity/anti-affinity rules
    └── Rollback on allocation failure
```

---

## Mode Detection Algorithm

### Detection Flow

```
1. Measure Network Conditions
   ├── Latency: ICMP ping / TCP RTT
   ├── Bandwidth: Historical throughput
   └── History: Moving average (10 samples)

2. Apply Thresholds
   ├── Datacenter: latency <10ms AND bandwidth >1Gbps
   ├── Internet: latency >50ms OR bandwidth <1Gbps
   └── Hybrid: Everything else

3. Update Components
   ├── AMST: Switch transport layer
   ├── HDE: Update compression strategy
   ├── PBA: Select predictor
   ├── ASS: Switch sync mechanism
   ├── ACP: Switch consensus
   └── ITP: Update placement strategy

4. Monitor and Adapt
   ├── Re-evaluate every 10 seconds
   ├── Log mode transitions
   └── Update metrics
```

### Configuration

```yaml
mode_detection:
  enabled: true
  interval: 10s
  
  thresholds:
    datacenter:
      latency_max: 10ms
      bandwidth_min: 1Gbps
    internet:
      latency_min: 50ms
      bandwidth_max: 1Gbps
  
  history:
    window_size: 10
    measurement_interval: 1s
```

---

## Feature Flag System

### Flag-Controlled Features

```yaml
feature_flags:
  version: v3
  
  # Core v3 features
  enable_hybrid_mode: true         # Enable mode detection
  auto_mode_detection: true        # Auto-switch modes
  
  # Component-level flags
  enable_internet_transport: true  # TCP transport
  enable_ml_compression: true      # ML-based compression
  enable_crdt_sync: true           # CRDT synchronization
  enable_pbft_consensus: true      # Byzantine consensus
  enable_geographic_placement: true # Geo-aware placement
  
  # Rollout controls
  rollout_percentage: 50           # 50% of VMs
  canary_criteria:
    - vm_type: "test"
    - environment: "staging"
```

### Gradual Rollout

```
Phase 0 (Day 0):     0% → v1 compatibility mode
Phase 1 (Days 1-7):  10% → Canary rollout
Phase 2 (Days 8-21): 50% → Staged rollout
Phase 3 (Day 22+):   100% → General availability
```

---

## Performance Characteristics

### Datacenter Mode Performance

| Metric | v1 (RDMA) | v3 (RDMA) | Change |
|--------|-----------|-----------|---------|
| Throughput | 41.8 Gbps | 42.3 Gbps | +1.2% |
| Latency (p50) | 48 ms | 45 ms | -6.3% |
| Latency (p99) | 125 ms | 118 ms | -5.6% |
| Compression | 2.79x | 2.82x | +1.1% |
| Prediction Accuracy | 87% | 87% | Same |

### Internet Mode Performance

| Metric | v1 (N/A) | v3 (TCP) | Improvement |
|--------|----------|----------|-------------|
| Throughput | N/A | 850 Mbps | ∞ (new) |
| Latency (p50) | N/A | 125 ms | ∞ (new) |
| Compression | N/A | 4.2x | ∞ (new) |
| Prediction Accuracy | N/A | 72% | ∞ (new) |
| Byzantine Tolerance | ❌ | ✅ (f=1) | New feature |

### Mode Switching Overhead

- Detection latency: 10-50ms
- Transition time: 100-500ms
- Data loss: 0% (graceful handoff)

---

## Comparison with v1

### Architectural Differences

```
v1 Architecture:
┌─────────────────────────────┐
│  Single Mode: Datacenter     │
│  ┌─────────────────────────┐│
│  │ AMST: RDMA only         ││
│  │ HDE: zstd only          ││
│  │ PBA: Single LSTM        ││
│  │ ASS: Raft only          ││
│  │ ACP: Raft only          ││
│  │ ITP: DQN only           ││
│  └─────────────────────────┘│
└─────────────────────────────┘

v3 Architecture:
┌──────────────────────────────────────┐
│  Hybrid: Datacenter + Internet       │
│  ┌────────────────────────────────┐ │
│  │ Mode Detector (Auto)           │ │
│  └────────────────────────────────┘ │
│           ↓           ↓              │
│  ┌────────────┬───────────────┐    │
│  │ Datacenter │   Internet     │    │
│  │   Mode     │     Mode       │    │
│  ├────────────┼───────────────┤    │
│  │ RDMA       │ TCP (BBR)      │    │
│  │ LZ4/zstd   │ zstd-max+ML    │    │
│  │ Raft       │ Raft/CRDT      │    │
│  │ Raft       │ Raft/PBFT      │    │
│  │ DQN        │ DQN/Geographic │    │
│  └────────────┴───────────────┘    │
└──────────────────────────────────────┘
```

### Feature Matrix

| Feature | v1 | v3 |
|---------|----|----|
| Datacenter RDMA | ✅ | ✅ |
| Internet TCP | ❌ | ✅ |
| Mode Detection | ❌ | ✅ |
| ML Compression | ❌ | ✅ |
| CRDT Sync | ❌ | ✅ |
| PBFT Consensus | ❌ | ✅ |
| Geographic Placement | ❌ | ✅ |
| Hybrid Mode | ❌ | ✅ |
| Backward Compatible | N/A | ✅ |

---

## Summary

DWCP v3 extends v1's datacenter-focused architecture with **internet-scale capabilities** while maintaining **100% backward compatibility**. The hybrid architecture auto-adapts to network conditions, selecting optimal protocols and algorithms for datacenter, internet, or mixed deployments.

**Key Innovation**: Mode-aware component architecture that provides both high-performance datacenter operation AND reliable internet-scale operation in a single, unified system.
