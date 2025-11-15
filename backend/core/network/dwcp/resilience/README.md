# DWCP Resilience Package

Comprehensive error recovery and fault tolerance system for Distributed Weighted Consensus Protocol (DWCP).

## Overview

The resilience package provides production-grade fault tolerance patterns including:

- **Circuit Breaker**: Prevents cascading failures with automatic recovery
- **Retry Logic**: Exponential backoff with jitter to handle transient failures
- **Health Monitoring**: Continuous health checks with automatic alerting
- **Rate Limiting**: Protect resources from overload
- **Bulkhead**: Isolate failures and limit concurrent operations
- **Error Budgets**: Track SLO compliance and error rates
- **Prometheus Metrics**: Full observability integration

## Quick Start

### Basic Usage

```go
import (
    "context"
    "github.com/yourusername/novacron/backend/core/network/dwcp/resilience"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()

    // Initialize metrics
    resilience.InitMetrics("dwcp")

    // Create resilience manager
    rm := resilience.NewResilienceManager("dwcp", logger)

    // Register circuit breaker
    rm.RegisterCircuitBreaker("network-ops", 5, 10*time.Second, 60*time.Second)

    // Execute protected operation
    ctx := context.Background()
    err := rm.ExecuteWithCircuitBreaker("network-ops", func() error {
        // Your operation here
        return sendNetworkMessage()
    })
}
```

### Full Protection Stack

```go
// Execute with all resilience patterns
err := rm.ExecuteWithAllProtections(ctx, "operation-name",
    func(ctx context.Context) error {
        // Your operation with:
        // - Circuit breaker protection
        // - Retry logic
        // - Rate limiting
        // - Bulkhead isolation
        // - Metrics collection
        return performOperation(ctx)
    })
```

## Circuit Breaker

### States

The circuit breaker has three states:

1. **Closed (0)**: Normal operation, all requests pass through
2. **Half-Open (1)**: Testing recovery, limited requests allowed
3. **Open (2)**: Failure mode, all requests rejected

### State Transitions

```
Closed --[failures >= threshold]--> Open
Open --[timeout elapsed]--> Half-Open
Half-Open --[success count >= threshold]--> Closed
Half-Open --[any failure]--> Open
```

### Configuration

```go
cb := resilience.NewCircuitBreaker(
    "operation-name",
    5,                  // Max failures before opening
    10*time.Second,     // Operation timeout
    60*time.Second,     // Reset timeout (open -> half-open)
    logger,
)
```

### Prometheus Metrics

```prometheus
# Circuit breaker state (0=closed, 1=half-open, 2=open)
dwcp_circuit_breaker_state{name="network-ops"} 0

# Total requests by result
dwcp_circuit_breaker_requests_total{name="network-ops",result="success"} 1000
dwcp_circuit_breaker_requests_total{name="network-ops",result="failure"} 50
dwcp_circuit_breaker_requests_total{name="network-ops",result="rejected"} 10

# State changes
dwcp_circuit_breaker_state_changes_total{name="network-ops",from_state="closed",to_state="open"} 2

# Request duration
dwcp_circuit_breaker_request_duration_seconds{name="network-ops"} 0.05
```

## Retry Logic

### Exponential Backoff

```go
retry := resilience.NewExponentialBackoff(
    "network-retry",
    3,                      // Max retries
    100*time.Millisecond,   // Initial delay
    10*time.Second,         // Max delay
    logger,
)

err := retry.ExecuteWithContext(ctx, func() error {
    return performNetworkCall()
})
```

### Custom Retry Policy

```go
policy := resilience.NewRetryPolicy(
    "custom-retry",
    5,                      // Max retries
    500*time.Millisecond,   // Initial delay
    30*time.Second,         // Max delay
    2.0,                    // Multiplier
    true,                   // Enable jitter
    logger,
)

// Set retryable errors
policy.SetRetryableErrors(
    ErrNetworkTimeout,
    ErrTemporaryFailure,
)

// Or custom check
policy.SetRetryableCheck(func(err error) bool {
    return isTransientError(err)
})
```

### Prometheus Metrics

```prometheus
# Retry attempts
dwcp_retry_attempts_total{policy="network-retry",attempt="1"} 100
dwcp_retry_attempts_total{policy="network-retry",attempt="2"} 30
dwcp_retry_attempts_total{policy="network-retry",attempt="3"} 5

# Backoff delays
dwcp_retry_backoff_delay_seconds{policy="network-retry"} 0.15
```

## Health Monitoring

### Register Health Checks

```go
hm := resilience.NewHealthMonitor("dwcp", 30*time.Second, 5*time.Second, logger)

// Ping check
hm.RegisterHealthCheck(resilience.NewPingHealthCheck("database",
    func(ctx context.Context) error {
        return db.Ping(ctx)
    }))

// Threshold check
hm.RegisterHealthCheck(resilience.NewThresholdHealthCheck("cpu-usage",
    func() float64 {
        return getCPUUsage()
    },
    0, 80)) // Alert if CPU > 80%

// Start monitoring
hm.StartMonitoring()
```

### Health Status

```go
// Check specific component
status, exists := hm.GetStatus("database")
if exists && !status.Healthy {
    log.Warn("Database unhealthy", "error", status.Error)
}

// Overall health
if hm.IsHealthy() {
    log.Info("All systems healthy")
}

// All statuses
statuses := hm.GetAllStatuses()
for name, status := range statuses {
    log.Info(name, "healthy", status.Healthy, "successRate", status.SuccessRate)
}
```

### Prometheus Metrics

```prometheus
# Health status (1=healthy, 0=unhealthy)
dwcp_health_check_status{checker="dwcp",check="database"} 1

# Check duration
dwcp_health_check_duration_seconds{checker="dwcp",check="database"} 0.002

# Total checks
dwcp_health_check_total{checker="dwcp",check="database",result="success"} 1000
```

## Rate Limiting

### Basic Rate Limiter

```go
rl := resilience.NewRateLimiter(
    "api-requests",
    1000,  // 1000 requests per second
    100,   // Burst size of 100
    logger,
)

if rl.Allow() {
    processRequest()
} else {
    return ErrRateLimitExceeded
}
```

### Adaptive Rate Limiter

```go
arl := resilience.NewAdaptiveRateLimiter(
    "adaptive-api",
    1000,               // Initial rate
    100,                // Min rate
    5000,               // Max rate
    100*time.Millisecond, // Target latency
    100,                // Window size
    logger,
)

// Records latency and auto-adjusts rate
arl.RecordLatency(150*time.Millisecond)
```

## Bulkhead Pattern

### Isolate Concurrent Operations

```go
bh := resilience.NewBulkhead(
    "database-pool",
    100,            // Max concurrent
    50,             // Queue size
    5*time.Second,  // Max wait time
    logger,
)

err := bh.Execute(func() error {
    return db.Query(ctx, query)
})
```

### Prometheus Metrics

```prometheus
# Current concurrent executions
dwcp_bulkhead_concurrent{bulkhead="database-pool"} 75

# Queued requests
dwcp_bulkhead_queued{bulkhead="database-pool"} 10

# Rejected requests
dwcp_bulkhead_rejected_total{bulkhead="database-pool"} 5
```

## Error Budgets

### Track SLO Compliance

```go
eb := resilience.NewErrorBudget(
    "availability-slo",
    0.999,          // 99.9% SLO
    24*time.Hour,   // 24-hour window
    logger,
)

// Record requests
eb.RecordRequest(true)  // Success
eb.RecordRequest(false) // Failure

// Check budget
if eb.BudgetExhausted() {
    log.Warn("Error budget exhausted, reduce deployment velocity")
}

metrics := eb.GetMetrics()
log.Info("Error budget",
    "remaining", metrics.RemainingBudget,
    "consumed", metrics.ConsumedBudget)
```

### Prometheus Metrics

```prometheus
# Remaining budget (0-1)
dwcp_error_budget_remaining{budget="availability-slo"} 0.85

# Consumed budget
dwcp_error_budget_consumed{budget="availability-slo"} 0.15

# Exhaustion events
dwcp_error_budget_exhausted_total{budget="availability-slo"} 2
```

## Integration Example

### Complete DWCP Resilient Client

```go
package main

import (
    "context"
    "time"

    "github.com/yourusername/novacron/backend/core/network/dwcp/resilience"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()
    defer logger.Sync()

    // Create configuration
    config := resilience.DefaultDWCPResilienceConfig()

    // Create resilient client
    client := resilience.NewDWCPResilientClient(config, logger)
    defer client.Shutdown()

    ctx := context.Background()

    // Send message with full protection
    err := client.SendMessage(ctx, []byte("Hello DWCP"))
    if err != nil {
        logger.Error("Send failed", zap.Error(err))
    }

    // Receive with circuit breaker
    msg, err := client.ReceiveMessage(ctx)
    if err != nil {
        logger.Error("Receive failed", zap.Error(err))
    }

    // Discover peers with retry
    peers, err := client.DiscoverPeers()
    if err != nil {
        logger.Error("Discovery failed", zap.Error(err))
    }

    // Check health
    if client.GetHealthStatus() {
        logger.Info("System healthy")
    }

    // Get metrics
    metrics := client.GetMetrics()
    logger.Info("Metrics", zap.Any("resilience", metrics))
}
```

## Testing

### Run Tests

```bash
# All tests
go test ./backend/core/network/dwcp/resilience/...

# Verbose output
go test -v ./backend/core/network/dwcp/resilience/...

# Specific test
go test -run TestCircuitBreakerOpensAfterFailures

# With coverage
go test -cover ./backend/core/network/dwcp/resilience/...
```

### Run Benchmarks

```bash
# All benchmarks
go test -bench=. ./backend/core/network/dwcp/resilience/...

# Specific benchmark
go test -bench=BenchmarkCircuitBreakerConcurrent

# With memory profiling
go test -bench=. -benchmem -memprofile=mem.prof

# With CPU profiling
go test -bench=. -cpuprofile=cpu.prof
```

## Performance Characteristics

### Circuit Breaker

- **Overhead**: ~200ns per operation (closed state)
- **Memory**: ~500 bytes per instance
- **Concurrency**: Lock-free reads, optimistic locking for state transitions

### Retry Logic

- **Overhead**: ~100ns per attempt calculation
- **Memory**: ~200 bytes per policy
- **Jitter**: 0-10% randomization to prevent thundering herd

### Health Monitoring

- **Check Interval**: Configurable (default 10s)
- **Timeout**: Configurable (default 5s)
- **Concurrent**: All checks run in parallel

## Best Practices

### 1. Circuit Breaker Thresholds

```go
// Conservative (production)
maxFailures: 5
timeout: 30*time.Second
resetTimeout: 60*time.Second

// Aggressive (development)
maxFailures: 3
timeout: 10*time.Second
resetTimeout: 30*time.Second
```

### 2. Retry Configuration

```go
// Network operations
maxRetries: 3
initialDelay: 100*time.Millisecond
maxDelay: 10*time.Second
multiplier: 2.0
jitter: true

// Database operations
maxRetries: 5
initialDelay: 50*time.Millisecond
maxDelay: 5*time.Second
multiplier: 1.5
jitter: true
```

### 3. Error Budget Targets

```go
// High availability service
SLO: 0.9999  // 99.99%
window: 30*24*time.Hour  // 30 days

// Standard service
SLO: 0.999   // 99.9%
window: 24*time.Hour  // 24 hours

// Best effort
SLO: 0.99    // 99%
window: time.Hour  // 1 hour
```

### 4. Monitoring and Alerting

```yaml
# Prometheus alerts
groups:
  - name: dwcp_resilience
    rules:
      # Circuit breaker is open
      - alert: CircuitBreakerOpen
        expr: dwcp_circuit_breaker_state{name="network-ops"} == 2
        for: 5m
        annotations:
          summary: "Circuit breaker {{ $labels.name }} is open"

      # High failure rate
      - alert: HighFailureRate
        expr: |
          rate(dwcp_circuit_breaker_requests_total{result="failure"}[5m])
          / rate(dwcp_circuit_breaker_requests_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High failure rate on {{ $labels.name }}"

      # Health check failing
      - alert: HealthCheckFailing
        expr: dwcp_health_check_status == 0
        for: 5m
        annotations:
          summary: "Health check {{ $labels.check }} is failing"

      # Error budget exhausted
      - alert: ErrorBudgetExhausted
        expr: dwcp_error_budget_remaining < 0.1
        for: 10m
        annotations:
          summary: "Error budget {{ $labels.budget }} nearly exhausted"
```

## Architecture

### Component Hierarchy

```
ResilienceManager
├── CircuitBreakers (map[string]*CircuitBreaker)
│   ├── network-send
│   ├── network-receive
│   └── peer-discovery
├── RetryPolicies (map[string]*RetryPolicy)
│   └── network-retry
├── RateLimiters (map[string]*RateLimiter)
│   └── outbound-messages
├── Bulkheads (map[string]*Bulkhead)
│   └── concurrent-connections
├── HealthChecker (*HealthChecker)
│   ├── network-connectivity
│   └── connection-pool
├── ErrorBudgets (map[string]*ErrorBudget)
│   ├── slo-availability
│   └── slo-latency
└── PrometheusMetrics (*PrometheusMetrics)
```

## Troubleshooting

### Circuit Breaker Stuck Open

**Problem**: Circuit breaker remains open despite service recovery

**Solution**:
```go
// Manually reset circuit breaker
cb.Reset()

// Or adjust reset timeout
cb := NewCircuitBreaker(name, maxFailures, timeout, 30*time.Second, logger)
```

### High Retry Overhead

**Problem**: Too many retries causing performance degradation

**Solution**:
```go
// Reduce max retries
policy.maxRetries = 2

// Increase initial delay
policy.initialDelay = 500*time.Millisecond

// Add selective retry logic
policy.SetRetryableCheck(func(err error) bool {
    return isTransientError(err) && !isPermanentFailure(err)
})
```

### Memory Leak in Metrics

**Problem**: Prometheus metrics growing unbounded

**Solution**:
```go
// Use label limits
metrics := NewPrometheusMetrics("dwcp")

// Clean up unused metrics periodically
// Or use shorter metric retention
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

See [LICENSE](../../LICENSE) for details.

## References

- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Exponential Backoff](https://en.wikipedia.org/wiki/Exponential_backoff)
- [SRE Book - Error Budgets](https://sre.google/sre-book/embracing-risk/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
