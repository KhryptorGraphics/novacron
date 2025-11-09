# DWCP Resilience - Quick Reference Guide

## File Structure

```
backend/core/network/dwcp/resilience/
├── circuit_breaker.go     # Cascade failure prevention
├── rate_limiter.go        # Rate limiting & adaptive controls
├── bulkhead.go            # Failure isolation
├── retry.go               # Retry with backoff
├── timeout.go             # Timeout management
├── health_checker.go      # Health monitoring
├── chaos.go               # Chaos engineering
├── degradation.go         # Graceful degradation
├── error_budget.go        # SLO tracking
├── integration.go         # Unified manager
└── resilience_test.go     # 26 tests (all passing)
```

## Quick Start

### Basic Usage

```go
import "github.com/khryptorgraphics/novacron/backend/core/network/dwcp"

// Initialize DWCP (resilience auto-initialized)
manager, _ := dwcp.NewManager(config, logger)
manager.Start()

// Send with full protection
ctx := context.Background()
err := manager.Send(ctx, data)

// Check health
if manager.IsHealthy() {
    // System healthy
}

// Check degradation
level := manager.GetDegradationLevel() // "normal", "degraded", etc.
```

## Component Cheat Sheet

### Circuit Breaker

**Purpose**: Stop calling failing services
**States**: Closed → Open → Half-Open → Closed

```go
// DWCP pre-configured:
Send:      5 failures, 30s timeout, 60s reset
Receive:   5 failures, 30s timeout, 60s reset
Sync:      3 failures, 10s timeout, 30s reset
Consensus: 3 failures, 10s timeout, 30s reset
```

### Rate Limiting

**Purpose**: Prevent overload

```go
// Fixed rate (DWCP uses adaptive)
rl := resilience.NewRateLimiter("op", 1000.0, 100, logger)
if rl.Allow() {
    // Process request
}

// Adaptive (auto-configured in DWCP)
// Adjusts 100-10000 RPS based on P95 latency target (100ms)
```

### Bulkhead

**Purpose**: Isolate failures

```go
// DWCP pre-configured:
Send:      100 concurrent, 200 queue, 5s wait
Receive:   100 concurrent, 200 queue, 5s wait
Sync:      50 concurrent, 100 queue, 10s wait
Consensus: 20 concurrent, 50 queue, 15s wait
```

### Retry Policy

**Purpose**: Recover from transient failures

```go
// DWCP pre-configured exponential backoff:
Send:      3 retries, 100ms-5s, 2.0x multiplier, jitter
Receive:   3 retries, 100ms-5s, 2.0x multiplier, jitter
Sync:      5 retries, 200ms-10s, 2.0x multiplier, jitter
Consensus: 5 retries, 500ms-30s, 2.0x multiplier, jitter
```

### Health Checks

**Purpose**: Proactive failure detection

```go
// DWCP monitors:
// - Transport connectivity
// - Send error rate (< 1%)
// - System error rate (< 0.1%)
// - Component health

healthy := manager.IsHealthy()
```

### Chaos Engineering

**Purpose**: Test resilience

```go
// Enable for testing (DISABLED by default)
manager.EnableChaosEngineering()

// Available faults:
// - Latency (50-200ms)
// - Random errors (10% rate)
// - Timeouts (5s)
// - Network partitions
// - CPU spikes
// - Memory leaks

// Disable after testing
manager.DisableChaosEngineering()
```

⚠️ **WARNING**: Only enable in non-production!

### Degradation Levels

**Purpose**: Graceful performance reduction

```
Normal           → Full functionality
Degraded         → Reduced performance
Severely Degraded → Essential features only
Emergency        → Survival mode
```

```go
level := manager.GetDegradationLevel()

// Auto-triggers on:
// - Error budget exhaustion
// - Health check failures
// - High error rates
// - High latency
```

### Error Budget

**Purpose**: Track SLO compliance

```go
// DWCP SLOs:
Send/Receive: 99.9% (0.1% error budget)
Sync:         99.5%
Consensus:    99.5%
System:       99.9%

// Check status
if manager.CheckErrorBudget("send") {
    // Budget exhausted, auto-degradation active
}
```

## Monitoring

### Get All Metrics

```go
metrics := manager.GetResilienceMetrics()

// Available metrics:
// - Circuit breaker states
// - Rate limiter stats
// - Bulkhead utilization
// - Retry statistics
// - Error budget compliance
// - Latency budgets
// - Health check status
// - Chaos injection stats
// - Degradation level
```

### Key Metrics to Monitor

```go
// Circuit Breakers
cb := metrics.CircuitBreakers["send"]
cb.State           // "closed", "open", "half-open"
cb.SuccessRate     // 0.0-1.0
cb.TotalFailures   // Count

// Rate Limiters
rl := metrics.RateLimiters["send"]
rl.TotalRejected   // Count
rl.RatePerSecond   // Current rate

// Error Budgets
eb := metrics.ErrorBudgets["send"]
eb.SuccessRate     // Current SLO compliance
eb.RemainingBudget // 0.0-1.0
eb.BudgetExhausted // true/false

// Health
hc := metrics.HealthChecker
hc.IsHealthy       // Overall health
hc.UnhealthyChecks // Failed check count

// Degradation
deg := metrics.Degradation
deg.CurrentLevel   // "normal", "degraded", etc.
```

## Common Operations

### Enable Resilience Features

```go
// All features auto-enabled on Start()
manager.Start()
```

### Check System Status

```go
// Overall health
healthy := manager.IsHealthy()

// Degradation level
level := manager.GetDegradationLevel()

// Error budget status
sendExhausted := manager.CheckErrorBudget("send")
systemExhausted := manager.CheckErrorBudget("system")
```

### Custom Resilience Configuration

If you need custom settings (beyond DWCP defaults):

```go
rm := resilience.NewResilienceManager("custom", logger)

// Register custom components
rm.RegisterCircuitBreaker("my-op", maxFail, timeout, reset)
rm.RegisterRateLimiter("my-op", rps, burst)
rm.RegisterBulkhead("my-op", concurrent, queue, wait)
rm.RegisterRetryPolicy("my-op", retries, initial, max, multiplier, jitter)
rm.RegisterErrorBudget("my-op", slo, window)

// Execute with protection
err := rm.ExecuteWithAllProtections(ctx, "my-op", func(ctx context.Context) error {
    // Your operation
    return nil
})
```

## Troubleshooting

### Circuit Breaker Open

**Symptom**: Requests rejected with `ErrCircuitBreakerOpen`

**Causes**:
- Repeated failures (> 5 for send/receive, > 3 for sync/consensus)
- Downstream service unavailable

**Resolution**:
1. Check downstream service health
2. Wait for reset timeout (60s for send/receive)
3. Circuit will auto-transition to half-open and test recovery
4. If recovery succeeds, circuit closes

### Rate Limit Exceeded

**Symptom**: Requests rejected with `ErrRateLimitExceeded`

**Causes**:
- Request rate > current limit
- Adaptive limiter reduced rate due to high latency

**Resolution**:
1. Check current rate: `metrics.RateLimiters["send"].RatePerSecond`
2. If adaptive, improve downstream latency to increase rate
3. Verify not experiencing DDoS or abnormal traffic

### Bulkhead Full

**Symptom**: Requests rejected with `ErrBulkheadFull`

**Causes**:
- Too many concurrent requests
- Slow operations blocking slots

**Resolution**:
1. Check bulkhead metrics: `metrics.Bulkheads["send"].Utilization`
2. Optimize operation performance
3. Consider increasing concurrent limit (if resources allow)

### Error Budget Exhausted

**Symptom**: Automatic degradation, high error rates

**Causes**:
- Success rate < SLO (99.9% for send/receive)
- Persistent failures

**Resolution**:
1. Check error budget: `metrics.ErrorBudgets["send"]`
2. Investigate root cause of failures
3. Fix underlying issues
4. Wait for budget to recover (resets hourly)

### Health Check Failures

**Symptom**: `IsHealthy()` returns false

**Causes**:
- Transport connectivity issues
- High error rates
- Component failures

**Resolution**:
1. Check specific health checks: `metrics.HealthChecker.GetAllStatuses()`
2. Address failing components
3. Health auto-recovers when checks pass

## Performance Tuning

### Adjusting Timeouts

DWCP uses adaptive timeouts (auto-adjust based on latency). To customize:

```go
// Via resilience manager
rm.timeoutManager.SetTimeout("my-op", 5*time.Second)
```

### Tuning Adaptive Rate Limiting

```go
// Increase target latency for more throughput
rm.RegisterAdaptiveRateLimiter("op",
    1000.0,               // initial
    100.0,                // min
    10000.0,              // max
    200*time.Millisecond, // higher target latency
    100)
```

### Bulkhead Sizing

```go
// Increase concurrency (requires more resources)
rm.RegisterBulkhead("op",
    200,              // concurrent (was 100)
    400,              // queue (was 200)
    5*time.Second)
```

## Best Practices

### DO ✅

- Keep chaos engineering disabled in production
- Monitor error budgets regularly
- Set appropriate SLOs (99.9% is aggressive)
- Use adaptive rate limiting for variable workloads
- Configure health checks for critical components
- Review degradation levels during incidents

### DON'T ❌

- Enable chaos in production without careful planning
- Ignore circuit breaker open states
- Set timeout too low (< observed latency × 2)
- Exceed error budget repeatedly
- Disable resilience features to "fix" issues
- Ignore degradation alerts

## Emergency Procedures

### System in Emergency Mode

1. Check degradation: `manager.GetDegradationLevel()`
2. Review error budgets: `manager.GetResilienceMetrics().ErrorBudgets`
3. Identify failing components
4. Implement fixes
5. Monitor recovery

### Complete System Failure

1. Check DWCP health: `manager.IsHealthy()`
2. Review transport layer
3. Check all circuit breaker states
4. Verify downstream services
5. Consider manual circuit breaker reset (last resort)

### Rolling Back Changes

If new deployment causes issues:

1. Check metrics before/after deployment
2. Identify degraded components
3. Roll back deployment
4. Monitor error budget recovery
5. Wait for circuit breakers to close

## Support & Documentation

- **Full Documentation**: `/docs/DWCP_PHASE2_PRODUCTION_HARDENING.md`
- **Code**: `/backend/core/network/dwcp/resilience/`
- **Tests**: `/backend/core/network/dwcp/resilience/resilience_test.go`
- **Integration**: `/backend/core/network/dwcp/resilience_integration.go`

---

**Quick Reference Version**: 1.0.0
**Last Updated**: 2025-11-08
**Status**: Production Ready ✅