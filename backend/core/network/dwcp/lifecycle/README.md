# DWCP Component Lifecycle Management

## Overview

The lifecycle package provides comprehensive lifecycle management for all DWCP components. It ensures proper initialization, startup, shutdown, health monitoring, and recovery of distributed system components.

## Quick Start

### 1. Implement ComponentLifecycle Interface

```go
package mycomponent

import (
    "context"
    "time"

    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/lifecycle"
    "go.uber.org/zap"
)

type MyComponent struct {
    *lifecycle.BaseLifecycle
    config *Config
    // component fields
}

func NewMyComponent(logger *zap.Logger) *MyComponent {
    return &MyComponent{
        BaseLifecycle: lifecycle.NewBaseLifecycle("mycomponent", logger),
    }
}

func (c *MyComponent) Init(ctx context.Context, config interface{}) error {
    c.config = config.(*Config)
    return c.TransitionTo(lifecycle.StateInitialized)
}

func (c *MyComponent) Start(ctx context.Context) error {
    if err := c.TransitionTo(lifecycle.StateStarting); err != nil {
        return err
    }

    // Start component logic

    return c.TransitionTo(lifecycle.StateRunning)
}

func (c *MyComponent) Stop(ctx context.Context) error {
    if err := c.TransitionTo(lifecycle.StateStopping); err != nil {
        return err
    }

    // Stop component logic

    return c.TransitionTo(lifecycle.StateStopped)
}

func (c *MyComponent) HealthCheck(ctx context.Context) error {
    if err := c.BaseLifecycle.HealthCheck(ctx); err != nil {
        return err
    }

    // Component-specific health checks

    return nil
}
```

### 2. Register and Manage Components

```go
// Create lifecycle manager
config := lifecycle.DefaultManagerConfig()
manager := lifecycle.NewManager(config, logger)

// Create and register components
transport := NewTransportComponent(logger)
compression := NewCompressionComponent(logger)
compression.SetDependencies([]string{"transport"})

manager.Register(transport)
manager.Register(compression)

// Start all components (handles dependency order automatically)
ctx := context.Background()
if err := manager.StartAll(ctx); err != nil {
    log.Fatal(err)
}

// Components are running...

// Stop all components (reverse dependency order)
manager.StopAll(ctx)
```

## Features

### State Machine

Components transition through well-defined states:

```
Uninitialized → Initialized → Starting → Running → Stopping → Stopped
                                 ↓
                              Failed (with recovery)
```

### Dependency Management

Automatic dependency resolution ensures components start in the correct order:

```go
component.SetDependencies([]string{"transport", "database"})
```

### Health Monitoring

Continuous health checks with automatic recovery:

```go
config := lifecycle.DefaultManagerConfig()
config.HealthCheckInterval = 10 * time.Second
config.RecoveryEnabled = true
```

### Graceful Shutdown

Configurable graceful shutdown with timeouts:

```go
shutdownConfig := lifecycle.DefaultShutdownConfig()
shutdownConfig.DrainTimeout = 30 * time.Second
shutdownConfig.CleanupTimeout = 10 * time.Second
component.SetShutdownConfig(shutdownConfig)
```

### Recovery Strategies

Automatic recovery from failures:

```go
type RecoverableComponent struct {
    *lifecycle.BaseLifecycle
}

func (c *RecoverableComponent) Recover(ctx context.Context) error {
    // Recovery logic
}

func (c *RecoverableComponent) GetRecoveryStrategy() lifecycle.RecoveryStrategy {
    return lifecycle.RecoveryStrategy{
        MaxRetries: 3,
        RetryBackoff: 1 * time.Second,
        ExponentialBackoff: true,
    }
}
```

### Metrics Collection

Built-in metrics for all components:

```go
metrics := component.GetMetrics()
fmt.Printf("Uptime: %v\n", metrics.Uptime)
fmt.Printf("State: %s\n", metrics.State)
fmt.Printf("Failures: %d\n", metrics.FailureCount)
```

### Observer Pattern

Subscribe to lifecycle events:

```go
type MyObserver struct{}

func (o *MyObserver) OnStateChange(component string, old, new lifecycle.State) {
    log.Printf("%s transitioned from %s to %s", component, old, new)
}

func (o *MyObserver) OnHealthCheckFailed(component string, err error) {
    log.Printf("%s health check failed: %v", component, err)
}

// ... implement other Observer methods

manager.RegisterObserver(&MyObserver{})
```

## Architecture

### Components

- **interface.go**: Core lifecycle interfaces
- **states.go**: State machine implementation
- **base.go**: Base lifecycle implementation
- **manager.go**: Lifecycle coordinator
- **health.go**: Health monitoring and recovery
- **dependency.go**: Dependency graph management
- **examples.go**: Implementation examples

### Key Concepts

1. **ComponentLifecycle**: Interface all components must implement
2. **StateMachine**: Validates and tracks state transitions
3. **BaseLifecycle**: Provides common lifecycle functionality
4. **Manager**: Coordinates multiple components
5. **HealthMonitor**: Continuous health checking
6. **DependencyGraph**: Manages component dependencies

## Configuration

### Manager Configuration

```go
config := &lifecycle.ManagerConfig{
    HealthCheckInterval:  10 * time.Second,   // Health check frequency
    HealthCheckTimeout:   5 * time.Second,    // Health check timeout
    RecoveryEnabled:      true,               // Enable auto-recovery
    MaxConcurrentStartup: 5,                  // Concurrent starts
    ShutdownTimeout:      45 * time.Second,   // Total shutdown timeout
    ParallelShutdown:     true,               // Parallel shutdown
}
```

### Shutdown Configuration

```go
shutdownConfig := lifecycle.ShutdownConfig{
    DrainTimeout:   30 * time.Second,  // Wait for in-flight ops
    CleanupTimeout: 10 * time.Second,  // Resource cleanup
    ForceTimeout:   5 * time.Second,   // Force termination
    TotalTimeout:   45 * time.Second,  // Overall timeout
}
```

## Best Practices

### 1. Always Use BaseLifecycle

Embed `BaseLifecycle` to get standard functionality:

```go
type MyComponent struct {
    *lifecycle.BaseLifecycle
    // your fields
}
```

### 2. Validate Dependencies

Ensure dependencies are satisfied before starting:

```go
func (c *Component) Start(ctx context.Context) error {
    if c.dependency.GetState() != lifecycle.StateRunning {
        return fmt.Errorf("dependency not ready")
    }
    // continue startup
}
```

### 3. Implement Proper Cleanup

Always clean up resources in Stop():

```go
func (c *Component) Stop(ctx context.Context) error {
    c.TransitionTo(lifecycle.StateStopping)

    // Close connections
    // Release locks
    // Flush buffers

    c.TransitionTo(lifecycle.StateStopped)
}
```

### 4. Use Context for Cancellation

Respect context cancellation in long operations:

```go
func (c *Component) Start(ctx context.Context) error {
    select {
    case <-ctx.Done():
        return ctx.Err()
    case <-time.After(initTime):
        // continue
    }
}
```

### 5. Fast Health Checks

Keep health checks fast (< 100ms):

```go
func (c *Component) HealthCheck(ctx context.Context) error {
    // Quick checks only
    if c.GetState() != lifecycle.StateRunning {
        return fmt.Errorf("not running")
    }
    // Basic connectivity check
    return nil
}
```

## Testing

### Unit Tests

```go
func TestComponentLifecycle(t *testing.T) {
    component := NewMyComponent(logger)

    // Test Init
    err := component.Init(context.Background(), config)
    assert.NoError(t, err)
    assert.Equal(t, lifecycle.StateInitialized, component.GetState())

    // Test Start
    err = component.Start(context.Background())
    assert.NoError(t, err)
    assert.Equal(t, lifecycle.StateRunning, component.GetState())

    // Test Stop
    err = component.Stop(context.Background())
    assert.NoError(t, err)
    assert.Equal(t, lifecycle.StateStopped, component.GetState())
}
```

### Integration Tests

```go
func TestMultiComponentLifecycle(t *testing.T) {
    manager := lifecycle.NewManager(config, logger)

    // Register multiple components
    manager.Register(comp1)
    manager.Register(comp2)

    // Test coordinated startup
    err := manager.StartAll(context.Background())
    assert.NoError(t, err)

    // Test coordinated shutdown
    err = manager.StopAll(context.Background())
    assert.NoError(t, err)
}
```

## Performance

### Benchmarks

- State transition: ~100ns
- Health check: < 1ms
- Component startup: < 500ms (target)
- Component shutdown: < 1s (target)

### Resource Limits

- Max goroutines per component: 100
- Max memory per component: 512MB
- Max concurrent health checks: 50

## Migration Guide

### From Manual Lifecycle to Lifecycle Package

1. **Wrap existing component**:

```go
type LegacyComponent struct {
    *lifecycle.BaseLifecycle
    legacy *OldComponent
}
```

2. **Implement interface methods**:

```go
func (c *LegacyComponent) Start(ctx context.Context) error {
    c.TransitionTo(lifecycle.StateStarting)
    err := c.legacy.OldStart()
    if err != nil {
        return err
    }
    return c.TransitionTo(lifecycle.StateRunning)
}
```

3. **Register with manager**:

```go
manager.Register(NewLegacyComponent(old, logger))
```

## Troubleshooting

### Component Won't Start

- Check dependencies are satisfied
- Verify configuration is valid
- Check logs for initialization errors

### Health Checks Failing

- Ensure component is in Running state
- Check resource availability
- Verify network connectivity

### Shutdown Hangs

- Increase DrainTimeout
- Check for deadlocks in Stop()
- Enable debug logging

### Dependency Cycle Detected

- Review dependency graph
- Remove circular dependencies
- Split components if needed

## References

- [Architecture Documentation](../../../docs/architecture/DWCP_COMPONENT_LIFECYCLE.md)
- [Examples](./examples.go)
- [Tests](./lifecycle_test.go)

---

**Version**: 1.0
**Last Updated**: 2025-11-14
