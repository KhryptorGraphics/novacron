# DWCP v1.0 → v3.0 Migration Strategy
## Backward-Compatible Upgrade with Dual-Mode Operation

**Date:** 2025-11-10
**Status:** Implementation Ready
**Version:** v1.0

---

## Executive Summary

This migration strategy ensures **zero-downtime**, **backward-compatible** upgrade from DWCP v1.0 to v3.0 with **dual-mode operation** where v1 and v3 run simultaneously.

### Key Principles
1. **No Breaking Changes:** DWCP v1.0 continues to work
2. **Gradual Rollout:** Feature flags enable phased deployment
3. **Automatic Rollback:** Instant revert to v1.0 if issues arise
4. **Mode Detection:** Automatic network mode selection

---

## Migration Phases

### Phase 1: Dual-Mode Infrastructure (Week 1)

#### 1.1 Mode Detection System
**File:** `backend/core/network/dwcp/upgrade/mode_detector.go`

```go
package upgrade

import (
    "context"
    "time"
)

// NetworkMode represents DWCP operation mode
type NetworkMode int

const (
    ModeDatacenter NetworkMode = iota  // v1: RDMA, 10-100 Gbps, <10ms
    ModeInternet                        // v3: TCP, 100-900 Mbps, 50-500ms
    ModeHybrid                          // Adaptive switching
)

// ModeDetector automatically detects network mode
type ModeDetector struct {
    currentMode      NetworkMode
    latencyThreshold time.Duration  // 10ms threshold
    bandwidthThreshold int64         // 1 Gbps threshold
}

// DetectMode analyzes network conditions and returns optimal mode
func (md *ModeDetector) DetectMode(ctx context.Context) NetworkMode {
    latency := md.measureLatency(ctx)
    bandwidth := md.measureBandwidth(ctx)

    // Datacenter mode: low latency + high bandwidth
    if latency < 10*time.Millisecond && bandwidth > 1e9 {
        return ModeDatacenter
    }

    // Internet mode: high latency or low bandwidth
    if latency > 50*time.Millisecond || bandwidth < 1e9 {
        return ModeInternet
    }

    // Hybrid mode: borderline conditions
    return ModeHybrid
}

func (md *ModeDetector) measureLatency(ctx context.Context) time.Duration {
    // Measure RTT to peer nodes
    // Implementation: ICMP ping or TCP handshake timing
    return 5 * time.Millisecond // Placeholder
}

func (md *ModeDetector) measureBandwidth(ctx context.Context) int64 {
    // Measure available bandwidth
    // Implementation: iperf-style bandwidth test
    return 10e9 // 10 Gbps placeholder
}
```

#### 1.2 Feature Flag System
**File:** `backend/core/network/dwcp/upgrade/feature_flags.go`

```go
package upgrade

import (
    "sync"
)

// DWCPFeatureFlags controls v3 feature rollout
type DWCPFeatureFlags struct {
    mu sync.RWMutex

    // Component-level flags
    EnableV3Transport      bool  // AMST v3
    EnableV3Compression    bool  // HDE v3
    EnableV3Prediction     bool  // PBA v3
    EnableV3StateSync      bool  // ASS v3
    EnableV3Consensus      bool  // ACP v3
    EnableV3Placement      bool  // ITP v3

    // Rollout control
    V3RolloutPercentage    int   // 0-100%

    // Emergency killswitch
    ForceV1Mode            bool  // Override everything, use v1
}

var globalFlags = &DWCPFeatureFlags{
    EnableV3Transport:   false,
    EnableV3Compression: false,
    EnableV3Prediction:  false,
    EnableV3StateSync:   false,
    EnableV3Consensus:   false,
    EnableV3Placement:   false,
    V3RolloutPercentage: 0,
    ForceV1Mode:         false,
}

// GetFeatureFlags returns current feature flags
func GetFeatureFlags() *DWCPFeatureFlags {
    globalFlags.mu.RLock()
    defer globalFlags.mu.RUnlock()

    // Return copy to prevent mutation
    return &DWCPFeatureFlags{
        EnableV3Transport:   globalFlags.EnableV3Transport,
        EnableV3Compression: globalFlags.EnableV3Compression,
        EnableV3Prediction:  globalFlags.EnableV3Prediction,
        EnableV3StateSync:   globalFlags.EnableV3StateSync,
        EnableV3Consensus:   globalFlags.EnableV3Consensus,
        EnableV3Placement:   globalFlags.EnableV3Placement,
        V3RolloutPercentage: globalFlags.V3RolloutPercentage,
        ForceV1Mode:         globalFlags.ForceV1Mode,
    }
}

// UpdateFeatureFlags updates feature flags (hot-reload)
func UpdateFeatureFlags(flags *DWCPFeatureFlags) {
    globalFlags.mu.Lock()
    defer globalFlags.mu.Unlock()

    globalFlags.EnableV3Transport = flags.EnableV3Transport
    globalFlags.EnableV3Compression = flags.EnableV3Compression
    globalFlags.EnableV3Prediction = flags.EnableV3Prediction
    globalFlags.EnableV3StateSync = flags.EnableV3StateSync
    globalFlags.EnableV3Consensus = flags.EnableV3Consensus
    globalFlags.EnableV3Placement = flags.EnableV3Placement
    globalFlags.V3RolloutPercentage = flags.V3RolloutPercentage
    globalFlags.ForceV1Mode = flags.ForceV1Mode
}

// ShouldUseV3 determines if v3 should be used based on rollout percentage
func ShouldUseV3(nodeID string) bool {
    flags := GetFeatureFlags()

    // Emergency killswitch
    if flags.ForceV1Mode {
        return false
    }

    // Gradual rollout based on node ID hash
    if flags.V3RolloutPercentage == 0 {
        return false
    }

    if flags.V3RolloutPercentage == 100 {
        return true
    }

    // Consistent hashing for gradual rollout
    hash := hashString(nodeID)
    return (hash % 100) < uint32(flags.V3RolloutPercentage)
}

func hashString(s string) uint32 {
    // FNV-1a hash
    const offset32 = 2166136261
    const prime32 = 16777619

    hash := uint32(offset32)
    for i := 0; i < len(s); i++ {
        hash ^= uint32(s[i])
        hash *= prime32
    }
    return hash
}
```

#### 1.3 Version Compatibility Layer
**File:** `backend/core/network/dwcp/upgrade/compatibility.go`

```go
package upgrade

import (
    "context"
    "fmt"

    dwcpv1 "github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
    dwcpv3 "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3"
)

// DWCPManager manages both v1 and v3 implementations
type DWCPManager struct {
    v1Manager    *dwcpv1.DWCPManager
    v3Manager    *dwcpv3.DWCPManager
    modeDetector *ModeDetector
    flags        *DWCPFeatureFlags
}

// NewDWCPManager creates dual-mode DWCP manager
func NewDWCPManager(ctx context.Context) (*DWCPManager, error) {
    v1Mgr, err := dwcpv1.NewDWCPManager(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to create v1 manager: %w", err)
    }

    v3Mgr, err := dwcpv3.NewDWCPManager(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to create v3 manager: %w", err)
    }

    return &DWCPManager{
        v1Manager:    v1Mgr,
        v3Manager:    v3Mgr,
        modeDetector: &ModeDetector{
            latencyThreshold:   10 * time.Millisecond,
            bandwidthThreshold: 1e9, // 1 Gbps
        },
        flags: GetFeatureFlags(),
    }, nil
}

// SelectManager chooses v1 or v3 based on mode and feature flags
func (dm *DWCPManager) SelectManager(ctx context.Context, nodeID string) interface{} {
    // Emergency killswitch
    if dm.flags.ForceV1Mode {
        return dm.v1Manager
    }

    // Check gradual rollout
    if !ShouldUseV3(nodeID) {
        return dm.v1Manager
    }

    // Detect network mode
    mode := dm.modeDetector.DetectMode(ctx)

    switch mode {
    case ModeDatacenter:
        // Datacenter mode: Use v1 (optimized for RDMA)
        if dm.flags.EnableV3Transport {
            return dm.v3Manager // v3 has datacenter mode too
        }
        return dm.v1Manager

    case ModeInternet:
        // Internet mode: Use v3 (optimized for WAN)
        if dm.flags.EnableV3Transport {
            return dm.v3Manager
        }
        return dm.v1Manager // Fallback

    case ModeHybrid:
        // Hybrid mode: Adaptive selection
        if dm.flags.EnableV3Transport {
            return dm.v3Manager // v3 handles hybrid mode
        }
        return dm.v1Manager

    default:
        return dm.v1Manager
    }
}

// Close shuts down both v1 and v3 managers gracefully
func (dm *DWCPManager) Close(ctx context.Context) error {
    v1Err := dm.v1Manager.Close(ctx)
    v3Err := dm.v3Manager.Close(ctx)

    if v1Err != nil {
        return fmt.Errorf("v1 shutdown error: %w", v1Err)
    }
    if v3Err != nil {
        return fmt.Errorf("v3 shutdown error: %w", v3Err)
    }

    return nil
}
```

---

### Phase 2: Component Migration (Weeks 2-6)

#### Component Upgrade Order

1. **AMST (Week 3)** - Transport layer first
2. **HDE (Week 4)** - Compression layer
3. **PBA (Week 4)** - Prediction layer (parallel with HDE)
4. **ASS (Week 5)** - State synchronization
5. **ACP (Week 5)** - Consensus (parallel with ASS)
6. **ITP (Week 6)** - Placement layer

#### Component Migration Pattern

Each component follows this pattern:

```go
// Example: AMST v1 → v3 migration
package dwcp

import (
    "context"

    amst_v1 "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
    amst_v3 "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/transport"
    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// AMSTManager manages both v1 and v3 AMST
type AMSTManager struct {
    v1 *amst_v1.AMST
    v3 *amst_v3.AMSTv3
    flags *upgrade.DWCPFeatureFlags
}

// SendData uses v1 or v3 based on flags
func (am *AMSTManager) SendData(ctx context.Context, data []byte, nodeID string) error {
    flags := upgrade.GetFeatureFlags()

    // Use v3 if enabled and rolled out to this node
    if flags.EnableV3Transport && upgrade.ShouldUseV3(nodeID) {
        return am.v3.SendData(ctx, data)
    }

    // Fall back to v1
    return am.v1.SendData(ctx, data)
}
```

---

### Phase 3: Gradual Rollout (Weeks 7-10)

#### Rollout Schedule

**Week 7: 10% Rollout**
```go
upgrade.UpdateFeatureFlags(&upgrade.DWCPFeatureFlags{
    EnableV3Transport:   true,
    EnableV3Compression: true,
    V3RolloutPercentage: 10,
})
```
- Monitor: Error rates, latency, throughput
- Validate: 10% of nodes using v3
- Rollback if: Error rate > 1% or latency > 2x baseline

**Week 8: 50% Rollout**
```go
upgrade.UpdateFeatureFlags(&upgrade.DWCPFeatureFlags{
    EnableV3Transport:   true,
    EnableV3Compression: true,
    EnableV3Prediction:  true,
    EnableV3StateSync:   true,
    V3RolloutPercentage: 50,
})
```
- Monitor: Performance metrics, resource usage
- Validate: v1 and v3 operating side-by-side
- Rollback if: Performance degradation > 20%

**Week 9: 100% Rollout**
```go
upgrade.UpdateFeatureFlags(&upgrade.DWCPFeatureFlags{
    EnableV3Transport:   true,
    EnableV3Compression: true,
    EnableV3Prediction:  true,
    EnableV3StateSync:   true,
    EnableV3Consensus:   true,
    EnableV3Placement:   true,
    V3RolloutPercentage: 100,
})
```
- Monitor: Full production metrics
- Validate: All nodes on v3
- Keep v1 code for emergency rollback

**Week 10: Stabilization**
- Monitor production stability
- Fix any issues discovered
- Prepare for v1 code removal (future)

---

### Phase 4: Rollback Procedures

#### Emergency Rollback (Immediate)
```go
// Instant rollback to v1 (no restart required)
upgrade.UpdateFeatureFlags(&upgrade.DWCPFeatureFlags{
    ForceV1Mode: true,
})
```
- **Time to Rollback:** <5 seconds
- **Data Loss:** None
- **Downtime:** None

#### Component-Level Rollback
```go
// Disable specific v3 component
upgrade.UpdateFeatureFlags(&upgrade.DWCPFeatureFlags{
    EnableV3Transport:   false,  // Disable AMST v3
    EnableV3Compression: true,   // Keep HDE v3
    V3RolloutPercentage: 50,     // Keep 50% rollout
})
```

#### Partial Rollback
```go
// Reduce rollout percentage
upgrade.UpdateFeatureFlags(&upgrade.DWCPFeatureFlags{
    EnableV3Transport:   true,
    V3RolloutPercentage: 25,  // Reduce from 50% to 25%
})
```

---

## Backward Compatibility Guarantees

### API Compatibility
- ✅ **No breaking changes** to existing DWCP APIs
- ✅ **Existing code** continues to work without modification
- ✅ **v1 and v3** can coexist in same deployment

### Data Compatibility
- ✅ **No data migration** required
- ✅ **v1 data format** still supported
- ✅ **v3 data format** backward-compatible with v1

### Protocol Compatibility
- ✅ **v1 protocol** still supported
- ✅ **v3 protocol** can fall back to v1
- ✅ **Automatic protocol negotiation**

---

## Monitoring & Validation

### Key Metrics to Track

**Performance Metrics:**
- Migration downtime (v1 baseline: <500ms)
- Bandwidth utilization (v1 baseline: 90%+)
- Consensus latency (v1 baseline: <100ms)
- Compression ratio (v3 target: 70-85%)

**Reliability Metrics:**
- Error rate (target: <0.1%)
- Node failure rate
- Byzantine node detection rate (v3 only)
- Rollback frequency

**Resource Metrics:**
- CPU usage (v1 baseline)
- Memory usage (v1 baseline)
- Network bandwidth usage
- Disk I/O

### Validation Checklist

Before each rollout increase:
- [ ] All v1 tests passing
- [ ] All v3 tests passing
- [ ] Backward compatibility tests passing
- [ ] Performance benchmarks meet targets
- [ ] No critical issues in production
- [ ] Rollback procedure tested and ready

---

## Testing Strategy

### Backward Compatibility Tests
**File:** `backend/core/network/dwcp/upgrade/compatibility/backward_compat_test.go`

```go
package compatibility

import (
    "context"
    "testing"
)

func TestDualModeOperation(t *testing.T) {
    ctx := context.Background()

    // Create dual-mode manager
    mgr, err := NewDWCPManager(ctx)
    if err != nil {
        t.Fatalf("Failed to create manager: %v", err)
    }
    defer mgr.Close(ctx)

    // Test v1 mode
    UpdateFeatureFlags(&DWCPFeatureFlags{
        ForceV1Mode: true,
    })

    // Run v1 tests
    testV1Functionality(t, mgr)

    // Test v3 mode
    UpdateFeatureFlags(&DWCPFeatureFlags{
        EnableV3Transport:   true,
        V3RolloutPercentage: 100,
    })

    // Run v3 tests
    testV3Functionality(t, mgr)

    // Test hybrid mode
    UpdateFeatureFlags(&DWCPFeatureFlags{
        EnableV3Transport:   true,
        V3RolloutPercentage: 50,
    })

    // Run hybrid tests
    testHybridMode(t, mgr)
}

func TestV1StillWorks(t *testing.T) {
    // Verify all v1 functionality after v3 upgrade
    // This is the most critical test
}

func TestRollback(t *testing.T) {
    // Verify rollback from v3 to v1 works instantly
}
```

---

## Success Criteria

### Technical Success
- ✅ All v1 tests still passing
- ✅ All v3 tests passing
- ✅ Backward compatibility tests passing
- ✅ Performance targets met
- ✅ Zero production incidents during rollout

### Business Success
- ✅ Zero downtime during migration
- ✅ No customer-facing issues
- ✅ Performance improvement in internet mode
- ✅ Successful 100% rollout

---

## Timeline Summary

| Week | Phase | Activities | Validation |
|------|-------|-----------|------------|
| 1 | Infrastructure | Mode detection, feature flags, compatibility layer | Unit tests |
| 2-6 | Components | AMST, HDE, PBA, ASS, ACP, ITP upgrades | Component tests |
| 7 | 10% Rollout | Enable v3 for 10% of nodes | Monitor metrics |
| 8 | 50% Rollout | Enable v3 for 50% of nodes | Performance validation |
| 9 | 100% Rollout | Enable v3 for all nodes | Full production validation |
| 10 | Stabilization | Fix issues, optimize, finalize | Production sign-off |

**Total Duration:** 10 weeks

---

## Conclusion

This migration strategy provides a **safe**, **gradual**, and **reversible** path from DWCP v1.0 to v3.0 with **zero downtime** and **full backward compatibility**.

**Key Advantages:**
1. **No Breaking Changes:** v1 continues to work
2. **Gradual Rollout:** 10% → 50% → 100%
3. **Instant Rollback:** <5 seconds to revert
4. **Mode Detection:** Automatic optimization

**Next Steps:**
1. Implement dual-mode infrastructure (Week 1)
2. Begin component upgrades (Weeks 2-6)
3. Start gradual rollout (Weeks 7-9)
4. Stabilize and finalize (Week 10)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-10
**Status:** Ready for Implementation
