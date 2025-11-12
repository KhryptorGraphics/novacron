// Package orchestrator provides component initialization orchestration for NovaCron
package orchestrator

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Component represents an initializable component
type Component interface {
	// Name returns the component name
	Name() string

	// Initialize initializes the component
	Initialize(ctx context.Context) error

	// Shutdown gracefully shuts down the component
	Shutdown(ctx context.Context) error

	// HealthCheck checks if the component is healthy
	HealthCheck(ctx context.Context) error

	// Dependencies returns list of component names this component depends on
	Dependencies() []string
}

// ComponentStatus represents the status of a component
type ComponentStatus string

const (
	StatusPending     ComponentStatus = "pending"
	StatusInitializing ComponentStatus = "initializing"
	StatusReady       ComponentStatus = "ready"
	StatusFailed      ComponentStatus = "failed"
	StatusShuttingDown ComponentStatus = "shutting_down"
	StatusShutdown    ComponentStatus = "shutdown"
)

// ComponentInfo holds component metadata and status
type ComponentInfo struct {
	Component Component
	Status    ComponentStatus
	Error     error
	StartTime time.Time
	ReadyTime time.Time
}

// Orchestrator manages the lifecycle of all system components
type Orchestrator struct {
	mu         sync.RWMutex
	components map[string]*ComponentInfo
	order      []string // Initialization order
	logger     Logger
	metrics    Metrics
}

// Logger interface for logging
type Logger interface {
	Info(msg string, keysAndValues ...interface{})
	Error(msg string, err error, keysAndValues ...interface{})
	Debug(msg string, keysAndValues ...interface{})
	Warn(msg string, keysAndValues ...interface{})
}

// Metrics interface for monitoring
type Metrics interface {
	RecordComponentInit(name string, duration time.Duration, success bool)
	RecordComponentShutdown(name string, duration time.Duration, success bool)
	SetComponentStatus(name string, status string)
}

// NewOrchestrator creates a new component orchestrator
func NewOrchestrator(logger Logger, metrics Metrics) *Orchestrator {
	return &Orchestrator{
		components: make(map[string]*ComponentInfo),
		order:      make([]string, 0),
		logger:     logger,
		metrics:    metrics,
	}
}

// Register registers a component for initialization
func (o *Orchestrator) Register(component Component) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	name := component.Name()
	if _, exists := o.components[name]; exists {
		return fmt.Errorf("component already registered: %s", name)
	}

	o.components[name] = &ComponentInfo{
		Component: component,
		Status:    StatusPending,
	}

	o.logger.Info("Component registered", "name", name, "dependencies", component.Dependencies())
	return nil
}

// Initialize initializes all components in dependency order
func (o *Orchestrator) Initialize(ctx context.Context) error {
	o.logger.Info("Starting component initialization")

	// Build dependency graph and determine order
	order, err := o.buildInitOrder()
	if err != nil {
		return fmt.Errorf("failed to build initialization order: %w", err)
	}
	o.order = order

	// Initialize components in order
	for _, name := range order {
		if err := o.initializeComponent(ctx, name); err != nil {
			o.logger.Error("Component initialization failed", err, "component", name)

			// Attempt to shutdown already initialized components
			o.shutdownInitialized(ctx)

			return fmt.Errorf("failed to initialize component %s: %w", name, err)
		}
	}

	o.logger.Info("All components initialized successfully", "count", len(order))
	return nil
}

// InitializeParallel initializes independent components in parallel
func (o *Orchestrator) InitializeParallel(ctx context.Context, maxConcurrency int) error {
	o.logger.Info("Starting parallel component initialization", "max_concurrency", maxConcurrency)

	// Build dependency graph
	order, err := o.buildInitOrder()
	if err != nil {
		return fmt.Errorf("failed to build initialization order: %w", err)
	}
	o.order = order

	// Group components by dependency level
	levels := o.groupByLevel()

	// Initialize each level in parallel
	for levelNum, level := range levels {
		o.logger.Info("Initializing component level", "level", levelNum, "components", len(level))

		if err := o.initializeLevel(ctx, level, maxConcurrency); err != nil {
			o.logger.Error("Level initialization failed", err, "level", levelNum)
			o.shutdownInitialized(ctx)
			return fmt.Errorf("failed to initialize level %d: %w", levelNum, err)
		}
	}

	o.logger.Info("All components initialized successfully", "count", len(order))
	return nil
}

// initializeComponent initializes a single component
func (o *Orchestrator) initializeComponent(ctx context.Context, name string) error {
	o.mu.Lock()
	info := o.components[name]
	info.Status = StatusInitializing
	info.StartTime = time.Now()
	o.mu.Unlock()

	o.logger.Info("Initializing component", "name", name)
	o.metrics.SetComponentStatus(name, string(StatusInitializing))

	// Check dependencies are ready
	for _, depName := range info.Component.Dependencies() {
		if err := o.checkDependency(depName); err != nil {
			o.updateStatus(name, StatusFailed, err)
			return err
		}
	}

	// Initialize the component
	startTime := time.Now()
	err := info.Component.Initialize(ctx)
	duration := time.Since(startTime)

	if err != nil {
		o.updateStatus(name, StatusFailed, err)
		o.metrics.RecordComponentInit(name, duration, false)
		return fmt.Errorf("component initialization failed: %w", err)
	}

	// Verify health
	if err := info.Component.HealthCheck(ctx); err != nil {
		o.updateStatus(name, StatusFailed, err)
		o.metrics.RecordComponentInit(name, duration, false)
		return fmt.Errorf("component health check failed: %w", err)
	}

	o.mu.Lock()
	info.Status = StatusReady
	info.ReadyTime = time.Now()
	o.mu.Unlock()

	o.logger.Info("Component initialized successfully", "name", name, "duration", duration)
	o.metrics.RecordComponentInit(name, duration, true)
	o.metrics.SetComponentStatus(name, string(StatusReady))

	return nil
}

// initializeLevel initializes components at the same dependency level in parallel
func (o *Orchestrator) initializeLevel(ctx context.Context, components []string, maxConcurrency int) error {
	semaphore := make(chan struct{}, maxConcurrency)
	errChan := make(chan error, len(components))
	var wg sync.WaitGroup

	for _, name := range components {
		wg.Add(1)
		go func(compName string) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire
			defer func() { <-semaphore }() // Release

			if err := o.initializeComponent(ctx, compName); err != nil {
				errChan <- fmt.Errorf("component %s: %w", compName, err)
			}
		}(name)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("initialization errors: %v", errors)
	}

	return nil
}

// Shutdown gracefully shuts down all components in reverse order
func (o *Orchestrator) Shutdown(ctx context.Context) error {
	o.logger.Info("Starting graceful shutdown")

	var errors []error

	// Shutdown in reverse order
	for i := len(o.order) - 1; i >= 0; i-- {
		name := o.order[i]

		o.mu.RLock()
		info := o.components[name]
		o.mu.RUnlock()

		if info.Status != StatusReady {
			continue // Skip components that aren't running
		}

		o.mu.Lock()
		info.Status = StatusShuttingDown
		o.mu.Unlock()

		o.logger.Info("Shutting down component", "name", name)
		o.metrics.SetComponentStatus(name, string(StatusShuttingDown))

		startTime := time.Now()
		err := info.Component.Shutdown(ctx)
		duration := time.Since(startTime)

		if err != nil {
			o.logger.Error("Component shutdown failed", err, "name", name)
			errors = append(errors, fmt.Errorf("component %s: %w", name, err))
			o.metrics.RecordComponentShutdown(name, duration, false)
		} else {
			o.logger.Info("Component shutdown successfully", "name", name, "duration", duration)
			o.metrics.RecordComponentShutdown(name, duration, true)
		}

		o.mu.Lock()
		info.Status = StatusShutdown
		o.mu.Unlock()
		o.metrics.SetComponentStatus(name, string(StatusShutdown))
	}

	if len(errors) > 0 {
		return fmt.Errorf("shutdown errors: %v", errors)
	}

	o.logger.Info("Graceful shutdown completed")
	return nil
}

// shutdownInitialized shuts down all initialized components (used during failed initialization)
func (o *Orchestrator) shutdownInitialized(ctx context.Context) {
	o.logger.Warn("Shutting down initialized components due to initialization failure")

	for i := len(o.order) - 1; i >= 0; i-- {
		name := o.order[i]

		o.mu.RLock()
		info := o.components[name]
		o.mu.RUnlock()

		if info.Status != StatusReady {
			continue
		}

		if err := info.Component.Shutdown(ctx); err != nil {
			o.logger.Error("Failed to shutdown component during rollback", err, "name", name)
		}
	}
}

// GetStatus returns the status of all components
func (o *Orchestrator) GetStatus() map[string]ComponentStatus {
	o.mu.RLock()
	defer o.mu.RUnlock()

	status := make(map[string]ComponentStatus)
	for name, info := range o.components {
		status[name] = info.Status
	}
	return status
}

// GetComponentInfo returns detailed info about a component
func (o *Orchestrator) GetComponentInfo(name string) (*ComponentInfo, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()

	info, exists := o.components[name]
	if !exists {
		return nil, fmt.Errorf("component not found: %s", name)
	}

	return info, nil
}

// HealthCheck checks the health of all components
func (o *Orchestrator) HealthCheck(ctx context.Context) error {
	o.mu.RLock()
	defer o.mu.RUnlock()

	var errors []error

	for name, info := range o.components {
		if info.Status != StatusReady {
			errors = append(errors, fmt.Errorf("component %s not ready: %s", name, info.Status))
			continue
		}

		if err := info.Component.HealthCheck(ctx); err != nil {
			errors = append(errors, fmt.Errorf("component %s health check failed: %w", name, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("health check errors: %v", errors)
	}

	return nil
}

// buildInitOrder builds initialization order using topological sort
func (o *Orchestrator) buildInitOrder() ([]string, error) {
	// Build dependency graph
	graph := make(map[string][]string)
	inDegree := make(map[string]int)

	for name, info := range o.components {
		if _, exists := inDegree[name]; !exists {
			inDegree[name] = 0
		}

		deps := info.Component.Dependencies()
		graph[name] = deps

		for _, dep := range deps {
			// Verify dependency exists
			if _, exists := o.components[dep]; !exists {
				return nil, fmt.Errorf("component %s depends on unknown component: %s", name, dep)
			}
			inDegree[dep]++
		}
	}

	// Topological sort using Kahn's algorithm
	var order []string
	queue := make([]string, 0)

	// Find all nodes with no incoming edges
	for name, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, name)
		}
	}

	for len(queue) > 0 {
		// Pop from queue
		current := queue[0]
		queue = queue[1:]
		order = append(order, current)

		// Process dependencies
		for _, neighbor := range graph[current] {
			inDegree[neighbor]--
			if inDegree[neighbor] == 0 {
				queue = append(queue, neighbor)
			}
		}
	}

	// Check for cycles
	if len(order) != len(o.components) {
		return nil, fmt.Errorf("circular dependency detected")
	}

	return order, nil
}

// groupByLevel groups components by dependency level for parallel initialization
func (o *Orchestrator) groupByLevel() [][]string {
	levels := make([][]string, 0)
	processed := make(map[string]bool)

	for len(processed) < len(o.components) {
		level := make([]string, 0)

		for name, info := range o.components {
			if processed[name] {
				continue
			}

			// Check if all dependencies are processed
			allDepsReady := true
			for _, dep := range info.Component.Dependencies() {
				if !processed[dep] {
					allDepsReady = false
					break
				}
			}

			if allDepsReady {
				level = append(level, name)
			}
		}

		if len(level) == 0 {
			break // Should not happen if no circular dependencies
		}

		// Mark as processed
		for _, name := range level {
			processed[name] = true
		}

		levels = append(levels, level)
	}

	return levels
}

// checkDependency checks if a dependency is ready
func (o *Orchestrator) checkDependency(name string) error {
	o.mu.RLock()
	info, exists := o.components[name]
	o.mu.RUnlock()

	if !exists {
		return fmt.Errorf("dependency not found: %s", name)
	}

	if info.Status != StatusReady {
		return fmt.Errorf("dependency not ready: %s (status: %s)", name, info.Status)
	}

	return nil
}

// updateStatus updates component status
func (o *Orchestrator) updateStatus(name string, status ComponentStatus, err error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if info, exists := o.components[name]; exists {
		info.Status = status
		info.Error = err
	}
}
