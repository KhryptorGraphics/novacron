# DWCP Conflict Resolution - Quick Start Guide

## 5-Minute Integration

### 1. Import Package

```go
import (
    "context"
    "time"
    "github.com/novacron/backend/core/network/dwcp/conflict"
)
```

### 2. Initialize System

```go
// Create detector
detectorConfig := conflict.DefaultDetectorConfig()
detector := conflict.NewConflictDetector(detectorConfig)

// Create strategy registry
registry := conflict.NewStrategyRegistry()

// Create merge engine
mergeConfig := conflict.DefaultMergeConfig()
mergeEngine := conflict.NewMergeEngine(mergeConfig)

// Create audit log
auditLog := conflict.NewAuditLog(10000, 24*time.Hour)

// Create policy manager
policyManager := conflict.NewPolicyManager(detector, registry, mergeEngine)

// Create recovery manager
recoveryConfig := conflict.DefaultRecoveryConfig()
recoveryManager := conflict.NewRecoveryManager(detector, policyManager, auditLog, recoveryConfig)

// Create VM handler
vmHandler := conflict.NewVMStateConflictHandler(policyManager, detector)

// Create CRDT integration
crdtIntegration := conflict.NewCRDTIntegration(mergeEngine, detector)

// Create metrics collector
metricsCollector := conflict.NewMetricsCollector(detector, policyManager, auditLog)

// Create performance monitor
perfTargets := conflict.DefaultPerformanceTargets()
perfMonitor := conflict.NewPerformanceMonitor(perfTargets)

// Create dashboard
dashboard := conflict.NewDashboard(metricsCollector, perfMonitor, detector, auditLog)
```

### 3. Configure Production Policy

```go
productionPolicy := conflict.NewPolicyBuilder("production").
    WithDefaultStrategy(conflict.StrategyLastWriteWins).
    WithFieldStrategy("power_state", conflict.StrategyHighestPriority).
    WithFieldStrategy("network_config", conflict.StrategySemanticMerge).
    WithRule("safe_concurrent",
        conflict.And(
            conflict.IsType(conflict.ConflictTypeConcurrentUpdate),
            conflict.ComplexityBelow(0.5),
        ),
        conflict.StrategyMultiValueRegister,
        50).
    WithEscalation("critical_conflicts",
        conflict.Or(
            conflict.IsSeverity(conflict.SeverityCritical),
            conflict.IsType(conflict.ConflictTypeInvariantViolation),
        ),
        conflict.EscalationActionManual,
        100).
    WithMaxRetries(5).
    WithManualThreshold(0.7).
    WithTimeout(30 * time.Second).
    WithFallback(conflict.StrategyLastWriteWins).
    Build()

policyManager.RegisterPolicy(productionPolicy)
```

### 4. Detect and Resolve Conflicts

```go
func handleVMUpdate(ctx context.Context, vmID string, localState, remoteState *VMState) error {
    // Convert to versions
    localVersion := &conflict.Version{
        VectorClock: localClock,
        Timestamp:   time.Now(),
        NodeID:      currentNodeID,
        Data:        localState,
    }

    remoteVersion := &conflict.Version{
        VectorClock: remoteClock,
        Timestamp:   time.Now(),
        NodeID:      remoteNodeID,
        Data:        remoteState,
    }

    // Detect conflict
    detectedConflict, err := detector.DetectConflict(ctx, vmID, localVersion, remoteVersion)
    if err != nil {
        return err
    }

    if detectedConflict == nil {
        // No conflict, use remote state
        return applyState(remoteState)
    }

    // Resolve using VM handler
    resolved, err := vmHandler.ResolveVMStateConflict(ctx, vmID, localState, remoteState)
    if err != nil {
        return err
    }

    return applyState(resolved)
}
```

### 5. Monitor Health

```go
func monitorHealth(ctx context.Context) {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            data := dashboard.GetDashboardData(ctx)

            log.Printf("Conflict Resolution Health: %s (%.1f/100)",
                data.Health.Overall, data.Health.Score)
            log.Printf("Success Rate: %.2f%%",
                data.MetricsSummary.SuccessRate * 100)
            log.Printf("Pending Conflicts: %d",
                data.MetricsSummary.PendingConflicts)

            if data.Health.Overall != "healthy" {
                for _, issue := range data.Health.Issues {
                    log.Printf("Issue: %s", issue)
                }
                for _, rec := range data.Health.Recommendations {
                    log.Printf("Recommendation: %s", rec)
                }
            }
        }
    }
}
```

### 6. Enable Automatic Recovery

```go
func enableAutoRecovery(ctx context.Context) {
    // Recovery manager already configured in step 2

    // Handle split-brain detection
    go func() {
        ticker := time.NewTicker(10 * time.Second)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                nodes := getClusterNodes()
                isSplitBrain, _ := recoveryManager.DetectSplitBrain(ctx, nodes)

                if isSplitBrain {
                    log.Println("Split-brain detected! Attempting recovery...")
                    partitions := getPartitions(nodes)
                    err := recoveryManager.ResolveSplitBrain(ctx, partitions)
                    if err != nil {
                        log.Printf("Split-brain resolution failed: %v", err)
                    } else {
                        log.Println("Split-brain resolved successfully")
                    }
                }
            }
        }
    }()

    // Create periodic checkpoints
    go func() {
        ticker := time.NewTicker(5 * time.Minute)
        defer ticker.Stop()

        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                resources := getActiveResources()
                for _, resource := range resources {
                    state := getResourceState(resource.ID)
                    vc := getVectorClock(resource.ID)
                    _, err := recoveryManager.CreateCheckpoint(resource.ID, state, vc)
                    if err != nil {
                        log.Printf("Checkpoint creation failed for %s: %v", resource.ID, err)
                    }
                }
            }
        }
    }()
}
```

## Common Use Cases

### Use Case 1: VM Power State Conflict

```go
// VM running on two hosts (split-brain)
localVM := &conflict.VMState{
    ID: "vm-123",
    PowerState: "running",
    NodeID: "host1",
}

remoteVM := &conflict.VMState{
    ID: "vm-123",
    PowerState: "running",
    NodeID: "host2",
}

// Handler will detect critical split-brain and escalate
resolved, err := vmHandler.ResolveVMStateConflict(ctx, "vm-123", localVM, remoteVM)
// Result: Manual intervention required
```

### Use Case 2: CRDT Counter Merge

```go
// Merge G-Counters
localCounter := map[string]uint64{
    "node1": 100,
    "node2": 50,
}

remoteCounter := map[string]uint64{
    "node1": 95,
    "node2": 75,
}

merged, err := crdtIntegration.MergeCRDT(ctx,
    conflict.CRDTTypeGCounter,
    localCounter,
    remoteCounter)

// Result: {"node1": 100, "node2": 75}
```

### Use Case 3: Custom Semantic Merge

```go
// Register custom merge rule
semanticStrategy := conflict.NewSemanticMergeStrategy()
semanticStrategy.RegisterRule("vm_config", func(local, remote interface{}) (interface{}, error) {
    localConfig := local.(*VMConfig)
    remoteConfig := remote.(*VMConfig)

    return &VMConfig{
        CPU:    max(localConfig.CPU, remoteConfig.CPU),
        Memory: max(localConfig.Memory, remoteConfig.Memory),
        Disks:  unionMerge(localConfig.Disks, remoteConfig.Disks),
    }, nil
})

registry.Register(semanticStrategy)
```

## Prometheus Metrics

Add to your scrape configuration:

```yaml
scrape_configs:
  - job_name: 'dwcp_conflict_resolution'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

Key metrics to monitor:
- `dwcp_conflict_resolution_success_rate` (target: >0.95)
- `dwcp_average_resolution_time_ms` (target: <10)
- `dwcp_pending_conflicts_count` (target: <50)
- `dwcp_manual_intervention_rate` (target: <0.05)

## Grafana Dashboard

Import the conflict resolution dashboard:

```json
{
  "dashboard": {
    "title": "DWCP Conflict Resolution",
    "panels": [
      {
        "title": "Success Rate",
        "targets": [{"expr": "dwcp_conflict_resolution_success_rate"}]
      },
      {
        "title": "Resolution Latency",
        "targets": [{"expr": "dwcp_average_resolution_time_ms"}]
      },
      {
        "title": "Pending Conflicts",
        "targets": [{"expr": "dwcp_pending_conflicts_count"}]
      }
    ]
  }
}
```

## Troubleshooting

### Problem: Low Success Rate

```bash
# Check metrics
curl localhost:9090/metrics | grep success_rate

# Review audit log
curl localhost:8080/api/conflicts/audit

# Analyze patterns
curl localhost:8080/api/conflicts/patterns
```

**Solution**: Adjust policy thresholds or add custom merge rules

### Problem: High Latency

```bash
# Check performance metrics
curl localhost:9090/metrics | grep latency

# Profile resolution
curl localhost:8080/api/conflicts/profile
```

**Solution**: Enable caching, optimize merge algorithms, or increase workers

### Problem: Split-Brain Detection

```bash
# Check cluster status
curl localhost:8080/api/cluster/status

# View partition information
curl localhost:8080/api/cluster/partitions
```

**Solution**: Review network connectivity, adjust quorum settings

## Best Practices

1. **Start Conservative**: Use LWW for simple fields, manual intervention for critical data
2. **Monitor Continuously**: Set up alerts for success rate, latency, and pending conflicts
3. **Test Policies**: Validate policies in staging before production
4. **Create Checkpoints**: Enable automatic checkpoint creation
5. **Review Audit Logs**: Regularly analyze conflict patterns

## Next Steps

1. Configure your production policy
2. Set up monitoring and alerting
3. Enable automatic recovery
4. Test failure scenarios
5. Review the full documentation at `/docs/DWCP_CONFLICT_RESOLUTION.md`

---

**Quick Start Complete!** Your conflict resolution system is now operational.

For detailed information, see the comprehensive guide at:
`/home/kp/novacron/docs/DWCP_CONFLICT_RESOLUTION.md`
