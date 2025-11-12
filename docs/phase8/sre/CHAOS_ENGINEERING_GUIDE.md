# Chaos Engineering Guide - NovaCron Phase 8

**Version:** 1.0.0
**Last Updated:** 2025-11-10
**Safety Level:** Production-ready with comprehensive safeguards

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Chaos Experiment Types](#chaos-experiment-types)
4. [Safety Framework](#safety-framework)
5. [Game Days](#game-days)
6. [Recovery Validation](#recovery-validation)
7. [Integration Guide](#integration-guide)
8. [Best Practices](#best-practices)

## Overview

### Purpose

Chaos engineering validates system resilience through controlled failure injection, ensuring:

- **Resilience Validation**: Verify system behavior under failure conditions
- **Weakness Discovery**: Identify single points of failure
- **Recovery Testing**: Validate automated recovery mechanisms
- **Confidence Building**: Increase confidence in system reliability

### Philosophy

> "Chaos Engineering is the discipline of experimenting on a system in order to build confidence in the system's capability to withstand turbulent conditions in production." - Principles of Chaos Engineering

### Key Principles

1. **Build a Hypothesis**: Start with a steady-state hypothesis
2. **Vary Real-World Events**: Simulate realistic failure scenarios
3. **Run in Production**: Test where it matters most (with safeguards)
4. **Automate**: Make chaos a continuous process
5. **Minimize Blast Radius**: Start small, expand gradually

## Architecture

### System Overview

```
┌───────────────────────────────────────────────────────────────┐
│                   Chaos Engineering Platform                   │
├───────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │   Chaos    │  │   Safety   │  │  Recovery  │              │
│  │ Orchestrator│  │   Guard    │  │ Validator  │              │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘              │
│        │               │               │                       │
│        └───────────────┼───────────────┘                       │
│                        │                                       │
│                        ▼                                       │
│         ┌──────────────────────────────┐                      │
│         │     Chaos Injectors          │                      │
│         ├──────────────────────────────┤                      │
│         │  • Network Chaos             │                      │
│         │  • Resource Chaos            │                      │
│         │  • Application Chaos         │                      │
│         │  • Dependency Chaos          │                      │
│         └──────────────────────────────┘                      │
│                                                                 │
│         ┌──────────────────────────────┐                      │
│         │    Monitoring & Alerting     │                      │
│         ├──────────────────────────────┤                      │
│         │  • Impact Metrics            │                      │
│         │  • Recovery Metrics          │                      │
│         │  • Safety Violations         │                      │
│         └──────────────────────────────┘                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**File:** `backend/core/chaos/chaos_framework.go` (1,400+ lines)

### Core Components

1. **ChaosOrchestrator**: Manages experiment lifecycle
2. **ChaosInjectors**: Implement specific chaos types
3. **SafetyGuard**: Enforces safety constraints
4. **RecoveryValidator**: Validates system recovery
5. **ChaosMonitor**: Monitors experiment impact

## Chaos Experiment Types

### 1. Network Chaos

#### Latency Injection

Introduces network latency to test timeout handling and user experience.

```go
experiment := &ChaosExperiment{
    Name: "API Latency Test",
    Type: ChaosNetworkLatency,
    Target: ChaosTarget{
        Type:       "service",
        Selector:   map[string]string{"app": "api-gateway"},
        Percentage: 50,  // Affect 50% of traffic
    },
    Parameters: map[string]interface{}{
        "latency": "500ms",
        "jitter":  "100ms",
    },
    Duration: 5 * time.Minute,
    Safety: SafetyConfig{
        Level:             SafetyCanary,
        MaxBlastRadius:    10,
        AutoRollback:      true,
        RollbackThreshold: 0.1,  // 10% error rate
    },
}
```

**Use Cases:**
- Test client timeout handling
- Validate retry logic
- Assess user experience degradation

#### Packet Loss

Simulates unreliable network conditions.

```go
experiment := &ChaosExperiment{
    Name: "Network Packet Loss",
    Type: ChaosNetworkPacketLoss,
    Parameters: map[string]interface{}{
        "loss_rate": "5%",  // 5% packet loss
        "correlation": "25%",  // 25% correlation between losses
    },
    Duration: 10 * time.Minute,
}
```

#### Network Partition

Tests split-brain scenarios and partition tolerance.

```go
experiment := &ChaosExperiment{
    Name: "Zone Partition",
    Type: ChaosNetworkPartition,
    Parameters: map[string]interface{}{
        "partition_type": "zone",
        "zones": []string{"us-west-2a", "us-west-2b"},
        "direction": "bidirectional",
    },
    Duration: 2 * time.Minute,
}
```

### 2. Resource Chaos

#### CPU Pressure

Simulates high CPU usage.

```go
experiment := &ChaosExperiment{
    Name: "CPU Stress Test",
    Type: ChaosCPUPressure,
    Parameters: map[string]interface{}{
        "cpu_cores": 2,
        "load_percentage": 80,
    },
    Duration: 5 * time.Minute,
}
```

#### Memory Pressure

Tests behavior under memory constraints.

```go
experiment := &ChaosExperiment{
    Name: "Memory Exhaustion",
    Type: ChaosMemoryPressure,
    Parameters: map[string]interface{}{
        "memory_mb": 2048,
        "rate": "100MB/s",
    },
    Duration: 3 * time.Minute,
}
```

#### Disk Pressure

Simulates disk I/O saturation or full disk conditions.

```go
experiment := &ChaosExperiment{
    Name: "Disk I/O Saturation",
    Type: ChaosDiskPressure,
    Parameters: map[string]interface{}{
        "io_ops": 1000,
        "write_size": "4KB",
    },
    Duration: 5 * time.Minute,
}
```

### 3. Application Chaos

#### Service Failure

Kills or crashes services to test redundancy.

```go
experiment := &ChaosExperiment{
    Name: "API Service Crash",
    Type: ChaosServiceFailure,
    Target: ChaosTarget{
        Selector: map[string]string{"service": "api"},
        Percentage: 30,  // Kill 30% of instances
    },
    Duration: 1 * time.Minute,
}
```

#### Dependency Failure

Simulates downstream dependency failures.

```go
experiment := &ChaosExperiment{
    Name: "Database Outage",
    Type: ChaosDependencyFailure,
    Parameters: map[string]interface{}{
        "dependency": "postgres",
        "failure_mode": "timeout",  // or "connection_refused"
        "error_rate": 100,
    },
    Duration: 2 * time.Minute,
}
```

#### Cascading Failure

Tests behavior under cascading failures.

```go
experiment := &ChaosExperiment{
    Name: "Cascading Failure",
    Type: ChaosCascadingFailure,
    Parameters: map[string]interface{}{
        "initial_service": "api-gateway",
        "propagation_delay": "30s",
        "affected_services": []string{"cache", "database", "queue"},
    },
    Duration: 5 * time.Minute,
}
```

## Safety Framework

### Safety Levels

```go
const (
    SafetyDryRun       // No actual chaos, just simulation
    SafetyDev          // Development environment only
    SafetyStaging      // Staging environment
    SafetyCanary       // Production canary (limited scope)
    SafetyProduction   // Full production (with safeguards)
)
```

### Safety Configuration

```go
type SafetyConfig struct {
    Level              SafetyLevel
    MaxBlastRadius     int        // Max affected instances
    AutoRollback       bool       // Auto-rollback on threshold
    RollbackThreshold  float64    // Error rate threshold
    RequireApproval    bool       // Require manual approval
    Approvers          []string   // List of approvers
    MonitoringEnabled  bool       // Enable impact monitoring
    AlertChannels      []string   // Alert channels
    SafetyChecks       []SafetyCheck
    EmergencyStopKey   string     // Emergency stop key
}
```

### Pre-Flight Safety Checks

```go
type SafetyCheck struct {
    Name        string
    Type        string
    Condition   string
    Required    bool
    LastChecked time.Time
    Passed      bool
    Message     string
}

// Example safety checks
safetyChecks := []SafetyCheck{
    {
        Name: "System Load",
        Condition: "cpu_usage < 70%",
        Required: true,
    },
    {
        Name: "Recent Incidents",
        Condition: "incidents_last_24h == 0",
        Required: true,
    },
    {
        Name: "On-Call Available",
        Condition: "oncall_engineer_available == true",
        Required: true,
    },
}
```

### Auto-Rollback Conditions

Experiments automatically rollback when:

1. Error rate exceeds threshold
2. Blast radius exceeds limit
3. Customer impact detected
4. Safety check fails
5. Emergency stop triggered

```go
func (o *ChaosOrchestrator) shouldRollback(
    experiment *ChaosExperiment,
    metrics ImpactMetrics,
) bool {
    if !experiment.Safety.AutoRollback {
        return false
    }

    // Check error rate threshold
    if metrics.ErrorRateIncrease > experiment.Safety.RollbackThreshold {
        return true
    }

    // Check customer impact
    if metrics.CustomerImpact > 0.1 {  // 10% customer impact
        return true
    }

    // Check blast radius
    if metrics.BlastRadius > experiment.Safety.MaxBlastRadius {
        return true
    }

    return false
}
```

### Emergency Stop

```bash
# Trigger emergency stop
curl -X POST http://localhost:8080/api/v1/chaos/emergency-stop \
  -H "Authorization: Bearer ${EMERGENCY_STOP_KEY}" \
  -d '{"experiment_id": "exp-001"}'
```

## Game Days

### What are Game Days?

Scheduled chaos engineering exercises that:
- Test incident response procedures
- Validate runbooks
- Train team on failure scenarios
- Build confidence in system resilience

### Planning a Game Day

#### 1. Define Objectives

```yaml
game_day:
  name: "Database Failure Recovery"
  date: "2025-11-15T14:00:00Z"
  duration: 2h
  objectives:
    - Test database failover
    - Validate automated recovery
    - Measure MTTR
    - Train on-call team
```

#### 2. Create Experiment Plan

```yaml
experiments:
  - name: "Primary Database Failure"
    type: dependency_failure
    target: "postgres-primary"
    duration: 5m
    expected_outcome: "Automatic failover to replica"

  - name: "Cache Invalidation"
    type: service_failure
    target: "redis-cluster"
    duration: 3m
    expected_outcome: "Direct database queries with degraded performance"

  - name: "Network Partition"
    type: network_partition
    zones: ["us-west-2a", "us-west-2b"]
    duration: 2m
    expected_outcome: "Multi-region traffic routing"
```

#### 3. Assemble Team

- **Chaos Engineer**: Executes experiments
- **SRE Team**: Monitors and responds
- **Dev Team**: Standby for code issues
- **Product Manager**: Observes customer impact
- **Executive Observer**: Optional

#### 4. Prepare Environment

```bash
# Pre-Game Day Checklist

# 1. Verify safety checks
./scripts/chaos/verify-safety-checks.sh

# 2. Notify stakeholders
./scripts/chaos/notify-game-day.sh

# 3. Set up monitoring
./scripts/chaos/setup-monitoring.sh

# 4. Brief team
./scripts/chaos/team-brief.sh

# 5. Enable dry-run mode (optional)
export CHAOS_DRY_RUN=true
```

### Running a Game Day

#### Execution Flow

```
┌──────────────┐
│  Pre-Game    │
│  Briefing    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Environment │
│  Preparation │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Execute     │
│  Experiments │ ───────▶ [Monitor Impact]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Validate    │
│  Recovery    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Post-Game   │
│  Debrief     │
└──────────────┘
```

#### Example Game Day Script

```go
func RunGameDay(ctx context.Context, plan *GameDayPlan) error {
    logger.Info("Starting game day", zap.String("name", plan.Name))

    // Pre-game
    if err := preGameChecks(plan); err != nil {
        return err
    }

    // Execute experiments
    for _, exp := range plan.Experiments {
        logger.Info("Running experiment", zap.String("name", exp.Name))

        // Run experiment
        execution := orchestrator.RunExperiment(ctx, exp.ID)

        // Record results
        results = append(results, execution.Results)

        // Wait between experiments
        time.Sleep(plan.ExperimentInterval)
    }

    // Post-game
    debrief := generateDebrief(results)
    sendDebriefReport(debrief)

    return nil
}
```

### Post-Game Day Analysis

#### Metrics to Review

1. **MTTR**: Mean time to recovery
2. **MTTD**: Mean time to detection
3. **Error Rate**: Increase during chaos
4. **Customer Impact**: Affected users/requests
5. **Recovery Success**: Automatic vs manual
6. **Team Response**: Response time and effectiveness

#### Debrief Template

```markdown
# Game Day Debrief: Database Failure Recovery

## Date: 2025-11-15
## Duration: 2 hours

### Experiments Executed

1. **Primary Database Failure**
   - Result: ✅ Success
   - MTTR: 45 seconds
   - Recovery: Automatic
   - Issues: None

2. **Cache Invalidation**
   - Result: ⚠️ Partial
   - MTTR: 3 minutes
   - Recovery: Manual intervention required
   - Issues: Connection pool exhaustion

3. **Network Partition**
   - Result: ✅ Success
   - MTTR: 12 seconds
   - Recovery: Automatic
   - Issues: None

### Key Findings

- ✅ Database failover working as expected
- ⚠️ Cache recovery needs improvement
- ✅ Multi-region routing effective
- ❌ Connection pool sizing inadequate

### Action Items

1. Increase connection pool size (Owner: @sre-team)
2. Improve cache recovery automation (Owner: @dev-team)
3. Add alerting for connection pool exhaustion (Owner: @sre-team)
4. Update runbooks with new procedures (Owner: @sre-team)

### Next Game Day

- Date: 2025-12-15
- Focus: Multi-region failover
```

## Recovery Validation

### Validation Framework

```go
type RecoveryValidator struct {
    checks    []RecoveryCheck
    timeout   time.Duration
}

type RecoveryCheck interface {
    Validate(ctx context.Context, experiment *ChaosExperiment) (*ValidationResult, error)
    GetName() string
}
```

### Built-in Validation Checks

#### 1. Health Check Validation

```go
type HealthCheckValidator struct {
    endpoint string
    expectedStatus int
}

func (v *HealthCheckValidator) Validate(
    ctx context.Context,
    experiment *ChaosExperiment,
) (*ValidationResult, error) {
    resp, err := http.Get(v.endpoint)
    if err != nil {
        return &ValidationResult{
            Check:   "health_check",
            Passed:  false,
            Message: fmt.Sprintf("Health check failed: %v", err),
        }, nil
    }

    passed := resp.StatusCode == v.expectedStatus

    return &ValidationResult{
        Check:   "health_check",
        Passed:  passed,
        Message: fmt.Sprintf("Status: %d", resp.StatusCode),
    }, nil
}
```

#### 2. Metric Validation

```go
type MetricValidator struct {
    metric    string
    threshold float64
    operator  string
}

func (v *MetricValidator) Validate(
    ctx context.Context,
    experiment *ChaosExperiment,
) (*ValidationResult, error) {
    value := getMetricValue(v.metric)

    var passed bool
    switch v.operator {
    case "<":
        passed = value < v.threshold
    case ">":
        passed = value > v.threshold
    }

    return &ValidationResult{
        Check:   "metric_validation",
        Passed:  passed,
        Message: fmt.Sprintf("%s %s %f (actual: %f)",
            v.metric, v.operator, v.threshold, value),
    }, nil
}
```

#### 3. Log Validation

```go
type LogValidator struct {
    query string
    maxErrors int
}

func (v *LogValidator) Validate(
    ctx context.Context,
    experiment *ChaosExperiment,
) (*ValidationResult, error) {
    errors := queryLogs(v.query, experiment.StartedAt, time.Now())

    passed := len(errors) <= v.maxErrors

    return &ValidationResult{
        Check:   "log_validation",
        Passed:  passed,
        Message: fmt.Sprintf("Found %d errors (max: %d)",
            len(errors), v.maxErrors),
    }, nil
}
```

### Custom Validation

```go
// Register custom validator
validator := &CustomValidator{
    name: "database_consistency",
    validateFunc: func(ctx context.Context, exp *ChaosExperiment) (*ValidationResult, error) {
        // Check database consistency
        consistent := checkDatabaseConsistency()

        return &ValidationResult{
            Check:   "database_consistency",
            Passed:  consistent,
            Message: "Database consistency check",
        }, nil
    },
}

orchestrator.validator.AddCheck(validator)
```

## Integration Guide

### Kubernetes Integration

```yaml
# chaos-experiment-crd.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-latency
spec:
  selector:
    namespaces:
      - production
    labelSelectors:
      app: api-gateway
  mode: one
  action: delay
  delay:
    latency: "500ms"
    jitter: "100ms"
  duration: "5m"
```

### Prometheus Integration

```yaml
# chaos-alerts.yaml
groups:
  - name: chaos
    rules:
      - alert: ChaosExperimentHighImpact
        expr: chaos_error_rate_increase > 0.2
        for: 1m
        annotations:
          summary: "Chaos experiment causing high error rate"

      - alert: ChaosExperimentFailed
        expr: chaos_experiment_status{status="failed"} == 1
        annotations:
          summary: "Chaos experiment failed"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Chaos Engineering",
    "panels": [
      {
        "title": "Active Experiments",
        "targets": [
          {
            "expr": "chaos_experiments_running"
          }
        ]
      },
      {
        "title": "Blast Radius",
        "targets": [
          {
            "expr": "chaos_blast_radius"
          }
        ]
      },
      {
        "title": "Recovery Time",
        "targets": [
          {
            "expr": "chaos_recovery_time_seconds"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

### 1. Start Small

```go
// ❌ Don't start with production-wide chaos
experiment := &ChaosExperiment{
    Target: ChaosTarget{
        Percentage: 100,  // Affects all instances
    },
}

// ✅ Start with canary deployments
experiment := &ChaosExperiment{
    Target: ChaosTarget{
        Percentage: 5,    // Affects only 5%
        MaxTargets: 2,    // Max 2 instances
    },
    Safety: SafetyConfig{
        Level: SafetyCanary,
    },
}
```

### 2. Always Enable Auto-Rollback

```go
experiment.Safety = SafetyConfig{
    AutoRollback:      true,
    RollbackThreshold: 0.05,  // 5% error rate
}
```

### 3. Monitor Continuously

```go
// Set up monitoring before experiment
monitor := &ChaosMonitor{
    metrics: []string{
        "error_rate",
        "latency_p95",
        "throughput",
        "customer_impact",
    },
    interval: 5 * time.Second,
}

orchestrator.AddObserver(monitor)
```

### 4. Document Everything

- Document expected behavior
- Record actual behavior
- Note any surprises
- Update runbooks

### 5. Schedule Regular Game Days

```yaml
game_days:
  frequency: monthly
  duration: 2h
  rotation: true  # Rotate team members
  scenarios:
    - database_failure
    - network_partition
    - cascading_failure
    - resource_exhaustion
```

### 6. Gradually Increase Scope

```
Week 1: Dry run in dev
Week 2: Staging environment
Week 3: Production canary (5%)
Week 4: Production canary (20%)
Week 5: Production (50%)
```

## Configuration

### Complete Configuration Example

```yaml
# config/chaos/chaos.yaml

chaos_engineering:
  enabled: true
  default_safety_level: canary
  max_concurrent: 3
  recovery_timeout: 5m
  monitoring_interval: 10s
  dry_run: false
  auto_recovery: true

  # Safety
  safety:
    require_approval: true
    approvers:
      - sre-team@company.com
    emergency_stop_enabled: true
    pre_flight_checks:
      - system_load
      - recent_incidents
      - oncall_available

  # Network chaos
  network:
    enabled: true
    allowed_latency_max: 5s
    allowed_packet_loss_max: 20%

  # Resource chaos
  resource:
    enabled: true
    max_cpu_pressure: 90%
    max_memory_pressure: 90%

  # Application chaos
  application:
    enabled: true
    max_instances_affected: 5

  # Integrations
  integrations:
    slack_webhook: ${SLACK_WEBHOOK}
    pagerduty_key: ${PAGERDUTY_KEY}
```

## Troubleshooting

### Common Issues

1. **Experiment Won't Start**
   - Check safety pre-flight checks
   - Verify approval if required
   - Check for concurrent experiment limit

2. **Unexpected Impact**
   - Review blast radius configuration
   - Check target selector specificity
   - Verify monitoring thresholds

3. **Failed Recovery**
   - Review recovery validation logs
   - Check rollback actions
   - Verify system health

---

**Document Version:** 1.0.0
**Lines:** 720+
**Last Updated:** 2025-11-10