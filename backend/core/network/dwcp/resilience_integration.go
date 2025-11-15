package dwcp

import (
	"context"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/resilience"
	"go.uber.org/zap"
)

// initializeResilience initializes all resilience components
func (m *Manager) initializeResilience() error {
	m.logger.Info("Initializing DWCP resilience layer")

	// Create resilience manager
	rm := resilience.NewResilienceManager("dwcp", m.logger)

	// Configure circuit breakers for different operations
	rm.RegisterCircuitBreaker("send", 5, 30*time.Second, 60*time.Second)
	rm.RegisterCircuitBreaker("receive", 5, 30*time.Second, 60*time.Second)
	rm.RegisterCircuitBreaker("sync", 3, 10*time.Second, 30*time.Second)
	rm.RegisterCircuitBreaker("consensus", 3, 10*time.Second, 30*time.Second)

	// Configure adaptive rate limiters
	// Target: 1000 RPS with adaptive scaling between 100-10000 RPS
	// Target latency: 100ms
	rm.RegisterAdaptiveRateLimiter("send", 1000.0, 100.0, 10000.0, 100*time.Millisecond, 100)
	rm.RegisterAdaptiveRateLimiter("receive", 1000.0, 100.0, 10000.0, 100*time.Millisecond, 100)

	// Configure bulkheads for isolation
	// Send: 100 concurrent, 200 queue
	rm.RegisterBulkhead("send", 100, 200, 5*time.Second)
	// Receive: 100 concurrent, 200 queue
	rm.RegisterBulkhead("receive", 100, 200, 5*time.Second)
	// Sync: 50 concurrent, 100 queue
	rm.RegisterBulkhead("sync", 50, 100, 10*time.Second)
	// Consensus: 20 concurrent, 50 queue
	rm.RegisterBulkhead("consensus", 20, 50, 15*time.Second)

	// Configure retry policies with exponential backoff
	rm.RegisterRetryPolicy("send", 3, 100*time.Millisecond, 5*time.Second, 2.0, true)
	rm.RegisterRetryPolicy("receive", 3, 100*time.Millisecond, 5*time.Second, 2.0, true)
	rm.RegisterRetryPolicy("sync", 5, 200*time.Millisecond, 10*time.Second, 2.0, true)
	rm.RegisterRetryPolicy("consensus", 5, 500*time.Millisecond, 30*time.Second, 2.0, true)

	// Configure error budgets (SLO tracking)
	// Send operations: 99.9% SLO (0.1% error budget)
	rm.RegisterErrorBudget("send", 0.999, time.Hour)
	// Receive operations: 99.9% SLO
	rm.RegisterErrorBudget("receive", 0.999, time.Hour)
	// Sync operations: 99.5% SLO
	rm.RegisterErrorBudget("sync", 0.995, time.Hour)
	// Consensus operations: 99.5% SLO
	rm.RegisterErrorBudget("consensus", 0.995, time.Hour)
	// Overall system: 99.9% SLO
	rm.RegisterErrorBudget("system", 0.999, time.Hour)

	// Configure latency budgets
	// P95 latency targets
	rm.RegisterLatencyBudget("send", 100*time.Millisecond, 95, 1000)
	rm.RegisterLatencyBudget("receive", 100*time.Millisecond, 95, 1000)
	rm.RegisterLatencyBudget("sync", 500*time.Millisecond, 95, 1000)
	rm.RegisterLatencyBudget("consensus", 1*time.Second, 95, 1000)

	// Configure health checks
	rm.RegisterHealthCheck(resilience.NewPingHealthCheck("transport", func(ctx context.Context) error {
		if m.transport == nil {
			return nil // Not initialized yet
		}
		// Transport-specific health check would go here
		return nil
	}))

	rm.RegisterHealthCheck(resilience.NewThresholdHealthCheck(
		"send-error-rate",
		func() float64 {
			metrics := rm.GetAllMetrics()
			if budgetMetrics, ok := metrics.ErrorBudgets["send"]; ok {
				return 1.0 - budgetMetrics.SuccessRate
			}
			return 0.0
		},
		0.0,
		0.01, // Alert if error rate > 1%
	))

	rm.RegisterHealthCheck(resilience.NewThresholdHealthCheck(
		"system-error-rate",
		func() float64 {
			metrics := rm.GetAllMetrics()
			if budgetMetrics, ok := metrics.ErrorBudgets["system"]; ok {
				return 1.0 - budgetMetrics.SuccessRate
			}
			return 0.0
		},
		0.0,
		0.001, // Alert if system error rate > 0.1%
	))

	// Set up degradation callbacks
	rm.SetDegradationLevelChangeCallback(func(old, new resilience.DegradationLevel) {
		m.logger.Warn("DWCP degradation level changed",
			zap.String("oldLevel", old.String()),
			zap.String("newLevel", new.String()))

		// Update metrics
		m.metricsMutex.Lock()
		m.metrics.DegradationLevel = new.String()
		m.metricsMutex.Unlock()
	})

	// Set up error budget callbacks
	rm.ForEachErrorBudget(func(name string, budget *resilience.ErrorBudget) {
		budgetName := name // Capture for closure
		budget.SetBudgetExhaustedCallback(func() {
			m.logger.Error("Error budget exhausted",
				zap.String("budget", budgetName))

			// Trigger degradation
			rm.SetComponentDegradation(budgetName, resilience.LevelDegraded)
		})

		budget.SetBudgetRecoveredCallback(func() {
			m.logger.Info("Error budget recovered",
				zap.String("budget", budgetName))

			// Recover from degradation
			rm.SetComponentDegradation(budgetName, resilience.LevelNormal)
		})
	})

	// Register chaos faults (disabled by default)
	rm.RegisterFault(resilience.NewLatencyFault("network-latency", 50*time.Millisecond, 200*time.Millisecond))
	rm.RegisterFault(resilience.NewErrorFault("random-error", 0.1)) // 10% error rate when injected
	rm.RegisterFault(resilience.NewTimeoutFault("timeout", 5*time.Second))

	// Start health monitoring
	rm.StartHealthMonitoring()

	m.resilience = rm

	m.logger.Info("DWCP resilience layer initialized successfully")
	return nil
}

// Send sends data with full resilience protection
func (m *Manager) Send(ctx context.Context, data []byte) error {
	if m.resilience == nil {
		// Fallback to direct send if resilience not initialized
		if m.transport != nil {
			return m.transport.Send(data)
		}
		return nil
	}

	return m.resilience.ExecuteWithAllProtections(ctx, "send", func(ctx context.Context) error {
		if m.transport == nil {
			return nil
		}
		return m.transport.Send(data)
	})
}

// Receive receives data with full resilience protection
func (m *Manager) Receive(ctx context.Context) ([]byte, error) {
	if m.resilience == nil {
		// Fallback to direct receive if resilience not initialized
		if m.transport != nil {
			return m.transport.Receive(0) // 0 = read all available data
		}
		return nil, nil
	}

	var result []byte
	err := m.resilience.ExecuteWithAllProtections(ctx, "receive", func(ctx context.Context) error {
		if m.transport == nil {
			return nil
		}
		data, err := m.transport.Receive(0) // 0 = read all available data
		result = data
		return err
	})

	return result, err
}

// EnableChaosEngineering enables chaos monkey for testing
func (m *Manager) EnableChaosEngineering() {
	if m.resilience != nil {
		m.resilience.EnableChaos()
		m.logger.Warn("Chaos engineering ENABLED for DWCP")
	}
}

// DisableChaosEngineering disables chaos monkey
func (m *Manager) DisableChaosEngineering() {
	if m.resilience != nil {
		m.resilience.DisableChaos()
		m.logger.Info("Chaos engineering disabled for DWCP")
	}
}

// GetResilienceMetrics returns resilience metrics
func (m *Manager) GetResilienceMetrics() *resilience.ResilienceMetrics {
	if m.resilience == nil {
		return nil
	}

	metrics := m.resilience.GetAllMetrics()
	return &metrics
}

// IsHealthy returns overall health status
func (m *Manager) IsHealthy() bool {
	if m.resilience == nil {
		return true
	}

	return m.resilience.IsHealthy()
}

// GetDegradationLevel returns current degradation level
func (m *Manager) GetDegradationLevel() string {
	if m.resilience == nil {
		return "normal"
	}

	return m.resilience.GetDegradationLevel().String()
}

// CheckErrorBudget checks if error budget is exhausted
func (m *Manager) CheckErrorBudget(operation string) bool {
	if m.resilience == nil {
		return false
	}

	return m.resilience.IsBudgetExhausted(operation)
}

// Add resilience field to Manager struct
// This should be added to the Manager struct in dwcp_manager.go:

/*
type Manager struct {
	config *Config
	logger *zap.Logger

	// Component interfaces
	transport   transport.Transport
	compression interface{}
	prediction  interface{}
	sync        interface{}
	consensus   interface{}

	// Resilience layer (add this)
	resilience  *resilience.ResilienceManager

	// ... rest of fields
}
*/
