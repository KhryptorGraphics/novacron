package lifecycle

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// Manager coordinates lifecycle of multiple components
type Manager struct {
	components map[string]ComponentLifecycle
	mu         sync.RWMutex

	// Dependency graph
	dependencyGraph *DependencyGraph

	// Health monitoring
	healthMonitor *HealthMonitor

	// Configuration
	config *ManagerConfig

	logger *zap.Logger

	// Observers
	observers      []Observer
	observersMutex sync.RWMutex
}

// ManagerConfig configures the lifecycle manager
type ManagerConfig struct {
	// HealthCheckInterval is how often to check component health
	HealthCheckInterval time.Duration

	// HealthCheckTimeout is timeout for health checks
	HealthCheckTimeout time.Duration

	// RecoveryEnabled enables automatic recovery
	RecoveryEnabled bool

	// MaxConcurrentStartup limits concurrent component starts
	MaxConcurrentStartup int

	// ShutdownTimeout is overall shutdown timeout
	ShutdownTimeout time.Duration

	// ParallelShutdown enables parallel component shutdown
	ParallelShutdown bool
}

// DefaultManagerConfig returns default configuration
func DefaultManagerConfig() *ManagerConfig {
	return &ManagerConfig{
		HealthCheckInterval:  10 * time.Second,
		HealthCheckTimeout:   5 * time.Second,
		RecoveryEnabled:      true,
		MaxConcurrentStartup: 5,
		ShutdownTimeout:      45 * time.Second,
		ParallelShutdown:     true,
	}
}

// NewManager creates a new lifecycle manager
func NewManager(config *ManagerConfig, logger *zap.Logger) *Manager {
	if config == nil {
		config = DefaultManagerConfig()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	m := &Manager{
		components:      make(map[string]ComponentLifecycle),
		dependencyGraph: NewDependencyGraph(),
		config:          config,
		logger:          logger,
		observers:       make([]Observer, 0),
	}

	m.healthMonitor = NewHealthMonitor(m, config.HealthCheckInterval, config.HealthCheckTimeout, logger)

	return m
}

// Register registers a component with the manager
func (m *Manager) Register(component ComponentLifecycle) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	name := component.GetName()

	if _, exists := m.components[name]; exists {
		return fmt.Errorf("component %s already registered", name)
	}

	// Register component
	m.components[name] = component

	// Add to dependency graph
	if err := m.dependencyGraph.AddNode(name, component.GetDependencies()); err != nil {
		delete(m.components, name)
		return fmt.Errorf("failed to add to dependency graph: %w", err)
	}

	m.logger.Info("Component registered",
		zap.String("component", name),
		zap.Strings("dependencies", component.GetDependencies()))

	return nil
}

// Unregister removes a component from the manager
func (m *Manager) Unregister(name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.components[name]; !exists {
		return fmt.Errorf("component %s not registered", name)
	}

	// Check if other components depend on this
	dependents := m.dependencyGraph.GetDependents(name)
	if len(dependents) > 0 {
		return fmt.Errorf("component %s has dependents: %v", name, dependents)
	}

	delete(m.components, name)
	m.dependencyGraph.RemoveNode(name)

	m.logger.Info("Component unregistered", zap.String("component", name))

	return nil
}

// Get retrieves a component by name
func (m *Manager) Get(name string) (ComponentLifecycle, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	component, exists := m.components[name]
	if !exists {
		return nil, fmt.Errorf("component %s not found", name)
	}

	return component, nil
}

// StartAll starts all components in dependency order
func (m *Manager) StartAll(ctx context.Context) error {
	m.logger.Info("Starting all components")

	// Get start order based on dependencies
	startOrder, err := m.dependencyGraph.GetStartOrder()
	if err != nil {
		return fmt.Errorf("failed to determine start order: %w", err)
	}

	m.logger.Info("Component start order determined",
		zap.Strings("order", startOrder))

	// Initialize all components first
	for _, name := range startOrder {
		component, err := m.Get(name)
		if err != nil {
			return err
		}

		m.logger.Info("Initializing component", zap.String("component", name))

		if err := component.Init(ctx, nil); err != nil {
			return fmt.Errorf("failed to initialize %s: %w", name, err)
		}
	}

	// Start components in dependency order with concurrency control
	semaphore := make(chan struct{}, m.config.MaxConcurrentStartup)
	errors := make(chan error, len(startOrder))
	var wg sync.WaitGroup

	for _, name := range startOrder {
		wg.Add(1)
		semaphore <- struct{}{} // Acquire semaphore

		go func(componentName string) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release semaphore

			component, err := m.Get(componentName)
			if err != nil {
				errors <- fmt.Errorf("failed to get component %s: %w", componentName, err)
				return
			}

			m.logger.Info("Starting component", zap.String("component", componentName))
			startTime := time.Now()

			if err := component.Start(ctx); err != nil {
				errors <- fmt.Errorf("failed to start %s: %w", componentName, err)
				return
			}

			duration := time.Since(startTime)
			m.logger.Info("Component started",
				zap.String("component", componentName),
				zap.Duration("duration", duration))
		}(name)
	}

	// Wait for all components to start
	wg.Wait()
	close(errors)

	// Check for errors
	var startErrors []error
	for err := range errors {
		startErrors = append(startErrors, err)
	}

	if len(startErrors) > 0 {
		return fmt.Errorf("failed to start some components: %v", startErrors)
	}

	// Start health monitoring
	if m.config.RecoveryEnabled {
		m.healthMonitor.Start(ctx)
	}

	m.logger.Info("All components started successfully")

	return nil
}

// StopAll stops all components in reverse dependency order
func (m *Manager) StopAll(ctx context.Context) error {
	m.logger.Info("Stopping all components")

	// Stop health monitoring first
	m.healthMonitor.Stop()

	// Get stop order (reverse of start order)
	stopOrder, err := m.dependencyGraph.GetStopOrder()
	if err != nil {
		return fmt.Errorf("failed to determine stop order: %w", err)
	}

	m.logger.Info("Component stop order determined",
		zap.Strings("order", stopOrder))

	// Create shutdown context with timeout
	shutdownCtx, cancel := context.WithTimeout(ctx, m.config.ShutdownTimeout)
	defer cancel()

	if m.config.ParallelShutdown {
		return m.stopAllParallel(shutdownCtx, stopOrder)
	}

	return m.stopAllSequential(shutdownCtx, stopOrder)
}

// stopAllSequential stops components one by one
func (m *Manager) stopAllSequential(ctx context.Context, stopOrder []string) error {
	var stopErrors []error

	for _, name := range stopOrder {
		component, err := m.Get(name)
		if err != nil {
			m.logger.Warn("Component not found during shutdown",
				zap.String("component", name))
			continue
		}

		m.logger.Info("Stopping component", zap.String("component", name))
		startTime := time.Now()

		if err := component.Stop(ctx); err != nil {
			m.logger.Error("Failed to stop component",
				zap.String("component", name),
				zap.Error(err))
			stopErrors = append(stopErrors, err)

			// Try force shutdown
			_ = component.Shutdown(ctx, 5*time.Second)
		}

		duration := time.Since(startTime)
		m.logger.Info("Component stopped",
			zap.String("component", name),
			zap.Duration("duration", duration))
	}

	if len(stopErrors) > 0 {
		return fmt.Errorf("errors stopping components: %v", stopErrors)
	}

	m.logger.Info("All components stopped successfully")
	return nil
}

// stopAllParallel stops components in parallel within each dependency level
func (m *Manager) stopAllParallel(ctx context.Context, stopOrder []string) error {
	// Group components by dependency level
	levels := m.dependencyGraph.GetDependencyLevels()

	// Process levels in reverse order
	for i := len(levels) - 1; i >= 0; i-- {
		level := levels[i]

		m.logger.Info("Stopping dependency level",
			zap.Int("level", i),
			zap.Strings("components", level))

		var wg sync.WaitGroup
		errors := make(chan error, len(level))

		for _, name := range level {
			wg.Add(1)
			go func(componentName string) {
				defer wg.Done()

				component, err := m.Get(componentName)
				if err != nil {
					errors <- fmt.Errorf("component not found: %s", componentName)
					return
				}

				m.logger.Info("Stopping component", zap.String("component", componentName))
				startTime := time.Now()

				if err := component.Stop(ctx); err != nil {
					m.logger.Error("Failed to stop component",
						zap.String("component", componentName),
						zap.Error(err))
					errors <- err

					// Try force shutdown
					_ = component.Shutdown(ctx, 5*time.Second)
					return
				}

				duration := time.Since(startTime)
				m.logger.Info("Component stopped",
					zap.String("component", componentName),
					zap.Duration("duration", duration))
			}(name)
		}

		wg.Wait()
		close(errors)

		// Check for errors in this level
		var levelErrors []error
		for err := range errors {
			levelErrors = append(levelErrors, err)
		}

		if len(levelErrors) > 0 {
			m.logger.Warn("Errors in dependency level",
				zap.Int("level", i),
				zap.Int("error_count", len(levelErrors)))
		}
	}

	m.logger.Info("All components stopped")
	return nil
}

// GetComponent returns a component by name
func (m *Manager) GetComponent(name string) (ComponentLifecycle, error) {
	return m.Get(name)
}

// GetAllComponents returns all registered components
func (m *Manager) GetAllComponents() map[string]ComponentLifecycle {
	m.mu.RLock()
	defer m.mu.RUnlock()

	components := make(map[string]ComponentLifecycle, len(m.components))
	for k, v := range m.components {
		components[k] = v
	}

	return components
}

// GetMetrics returns metrics for all components
func (m *Manager) GetMetrics() map[string]ComponentMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	metrics := make(map[string]ComponentMetrics, len(m.components))

	for name, component := range m.components {
		if metricsProvider, ok := component.(MetricsProvider); ok {
			metrics[name] = metricsProvider.GetMetrics()
		}
	}

	return metrics
}

// RegisterObserver registers an observer for all components
func (m *Manager) RegisterObserver(observer Observer) {
	m.observersMutex.Lock()
	defer m.observersMutex.Unlock()

	m.observers = append(m.observers, observer)

	// Register with all existing components
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, component := range m.components {
		if observable, ok := component.(Observable); ok {
			observable.RegisterObserver(observer)
		}
	}
}

// GetHealthStatus returns health status of all components
func (m *Manager) GetHealthStatus() map[string]error {
	return m.healthMonitor.GetHealthStatus()
}
