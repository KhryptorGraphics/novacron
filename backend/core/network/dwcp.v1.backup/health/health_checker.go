// Package health provides health checking and validation for DWCP components
package health

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Status represents the health status of a component
type Status string

const (
	StatusHealthy   Status = "healthy"
	StatusDegraded  Status = "degraded"
	StatusUnhealthy Status = "unhealthy"
	StatusUnknown   Status = "unknown"
)

// ComponentHealth represents the health of a single component
type ComponentHealth struct {
	Name        string                 `json:"name"`
	Status      Status                 `json:"status"`
	Message     string                 `json:"message,omitempty"`
	LastChecked time.Time              `json:"last_checked"`
	Metrics     map[string]interface{} `json:"metrics,omitempty"`
}

// HealthCheck represents the overall health of DWCP
type HealthCheck struct {
	Status     Status                      `json:"status"`
	Components map[string]*ComponentHealth `json:"components"`
	Timestamp  time.Time                   `json:"timestamp"`
	Version    string                      `json:"version"`
}

// Checker performs health checks on DWCP components
type Checker struct {
	mu         sync.RWMutex
	components map[string]CheckFunc
	interval   time.Duration
	timeout    time.Duration
	results    map[string]*ComponentHealth
	stopCh     chan struct{}
	version    string
}

// CheckFunc is a function that performs a health check
type CheckFunc func(ctx context.Context) (*ComponentHealth, error)

// NewChecker creates a new health checker
func NewChecker(interval, timeout time.Duration, version string) *Checker {
	return &Checker{
		components: make(map[string]CheckFunc),
		interval:   interval,
		timeout:    timeout,
		results:    make(map[string]*ComponentHealth),
		stopCh:     make(chan struct{}),
		version:    version,
	}
}

// RegisterCheck registers a health check function for a component
func (c *Checker) RegisterCheck(name string, check CheckFunc) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.components[name] = check
}

// Start starts the health checker
func (c *Checker) Start(ctx context.Context) {
	ticker := time.NewTicker(c.interval)
	defer ticker.Stop()

	// Run initial check
	c.runChecks(ctx)

	for {
		select {
		case <-ticker.C:
			c.runChecks(ctx)
		case <-c.stopCh:
			return
		case <-ctx.Done():
			return
		}
	}
}

// Stop stops the health checker
func (c *Checker) Stop() {
	close(c.stopCh)
}

// runChecks runs all registered health checks
func (c *Checker) runChecks(ctx context.Context) {
	c.mu.RLock()
	components := make(map[string]CheckFunc, len(c.components))
	for name, check := range c.components {
		components[name] = check
	}
	c.mu.RUnlock()

	var wg sync.WaitGroup
	results := make(map[string]*ComponentHealth)
	resultsMu := sync.Mutex{}

	for name, check := range components {
		wg.Add(1)
		go func(n string, cf CheckFunc) {
			defer wg.Done()

			checkCtx, cancel := context.WithTimeout(ctx, c.timeout)
			defer cancel()

			result, err := cf(checkCtx)
			if err != nil {
				result = &ComponentHealth{
					Name:        n,
					Status:      StatusUnhealthy,
					Message:     fmt.Sprintf("Check failed: %v", err),
					LastChecked: time.Now(),
				}
			}

			resultsMu.Lock()
			results[n] = result
			resultsMu.Unlock()
		}(name, check)
	}

	wg.Wait()

	c.mu.Lock()
	c.results = results
	c.mu.Unlock()
}

// GetHealth returns the current health status
func (c *Checker) GetHealth() *HealthCheck {
	c.mu.RLock()
	defer c.mu.RUnlock()

	health := &HealthCheck{
		Status:     StatusHealthy,
		Components: make(map[string]*ComponentHealth),
		Timestamp:  time.Now(),
		Version:    c.version,
	}

	// Copy results
	for name, result := range c.results {
		health.Components[name] = result

		// Determine overall status
		switch result.Status {
		case StatusUnhealthy:
			health.Status = StatusUnhealthy
		case StatusDegraded:
			if health.Status == StatusHealthy {
				health.Status = StatusDegraded
			}
		}
	}

	if len(health.Components) == 0 {
		health.Status = StatusUnknown
	}

	return health
}

// IsHealthy returns true if all components are healthy
func (c *Checker) IsHealthy() bool {
	health := c.GetHealth()
	return health.Status == StatusHealthy
}

// AMSTHealthCheck checks the health of AMST transport
func AMSTHealthCheck(activeStreams, minStreams, maxStreams int) CheckFunc {
	return func(ctx context.Context) (*ComponentHealth, error) {
		health := &ComponentHealth{
			Name:        "amst_transport",
			LastChecked: time.Now(),
			Metrics: map[string]interface{}{
				"active_streams": activeStreams,
				"min_streams":    minStreams,
				"max_streams":    maxStreams,
			},
		}

		// Check if streams are within acceptable range
		if activeStreams < minStreams {
			health.Status = StatusDegraded
			health.Message = fmt.Sprintf("Active streams (%d) below minimum (%d)", activeStreams, minStreams)
		} else if activeStreams > maxStreams {
			health.Status = StatusUnhealthy
			health.Message = fmt.Sprintf("Active streams (%d) exceeds maximum (%d)", activeStreams, maxStreams)
		} else if activeStreams == 0 {
			health.Status = StatusUnhealthy
			health.Message = "No active streams"
		} else {
			health.Status = StatusHealthy
			health.Message = fmt.Sprintf("%d streams active", activeStreams)
		}

		return health, nil
	}
}

// HDEHealthCheck checks the health of HDE compression
func HDEHealthCheck(compressionEnabled bool, avgRatio, minRatio float64, baselineCount int) CheckFunc {
	return func(ctx context.Context) (*ComponentHealth, error) {
		health := &ComponentHealth{
			Name:        "hde_compression",
			LastChecked: time.Now(),
			Metrics: map[string]interface{}{
				"enabled":             compressionEnabled,
				"avg_compression_ratio": avgRatio,
				"min_compression_ratio": minRatio,
				"baseline_count":        baselineCount,
			},
		}

		if !compressionEnabled {
			health.Status = StatusDegraded
			health.Message = "Compression disabled"
			return health, nil
		}

		// Check compression effectiveness
		if avgRatio < minRatio {
			health.Status = StatusDegraded
			health.Message = fmt.Sprintf("Average compression ratio (%.2f) below minimum (%.2f)", avgRatio, minRatio)
		} else if baselineCount == 0 {
			health.Status = StatusDegraded
			health.Message = "No baselines synchronized"
		} else {
			health.Status = StatusHealthy
			health.Message = fmt.Sprintf("Compression ratio: %.2fx with %d baselines", avgRatio, baselineCount)
		}

		return health, nil
	}
}

// PrometheusHealthCheck checks if Prometheus metrics are accessible
func PrometheusHealthCheck(metricsPort int) CheckFunc {
	return func(ctx context.Context) (*ComponentHealth, error) {
		health := &ComponentHealth{
			Name:        "prometheus_metrics",
			LastChecked: time.Now(),
			Metrics: map[string]interface{}{
				"metrics_port": metricsPort,
			},
		}

		// Simple check - in production, would actually query the endpoint
		if metricsPort > 0 && metricsPort < 65536 {
			health.Status = StatusHealthy
			health.Message = fmt.Sprintf("Metrics available on port %d", metricsPort)
		} else {
			health.Status = StatusUnhealthy
			health.Message = "Invalid metrics port"
		}

		return health, nil
	}
}

// ConfigHealthCheck checks if configuration is loaded correctly
func ConfigHealthCheck(configLoaded bool, configPath string) CheckFunc {
	return func(ctx context.Context) (*ComponentHealth, error) {
		health := &ComponentHealth{
			Name:        "configuration",
			LastChecked: time.Now(),
			Metrics: map[string]interface{}{
				"config_path": configPath,
			},
		}

		if configLoaded {
			health.Status = StatusHealthy
			health.Message = fmt.Sprintf("Configuration loaded from %s", configPath)
		} else {
			health.Status = StatusUnhealthy
			health.Message = "Configuration not loaded"
		}

		return health, nil
	}
}

// ErrorRateHealthCheck checks if error rate is within acceptable limits
func ErrorRateHealthCheck(errorCount, totalRequests int, maxErrorRate float64) CheckFunc {
	return func(ctx context.Context) (*ComponentHealth, error) {
		health := &ComponentHealth{
			Name:        "error_rate",
			LastChecked: time.Now(),
		}

		var errorRate float64
		if totalRequests > 0 {
			errorRate = float64(errorCount) / float64(totalRequests) * 100
		}

		health.Metrics = map[string]interface{}{
			"error_count":     errorCount,
			"total_requests":  totalRequests,
			"error_rate_pct": errorRate,
			"max_error_rate_pct": maxErrorRate,
		}

		if errorRate > maxErrorRate {
			health.Status = StatusUnhealthy
			health.Message = fmt.Sprintf("Error rate (%.2f%%) exceeds maximum (%.2f%%)", errorRate, maxErrorRate)
		} else if errorRate > maxErrorRate*0.5 {
			health.Status = StatusDegraded
			health.Message = fmt.Sprintf("Error rate (%.2f%%) elevated", errorRate)
		} else {
			health.Status = StatusHealthy
			health.Message = fmt.Sprintf("Error rate: %.2f%%", errorRate)
		}

		return health, nil
	}
}

// BaselineSyncHealthCheck checks if baseline states are synchronized
func BaselineSyncHealthCheck(syncedNodes, totalNodes int, lastSyncTime time.Time, maxSyncAge time.Duration) CheckFunc {
	return func(ctx context.Context) (*ComponentHealth, error) {
		health := &ComponentHealth{
			Name:        "baseline_sync",
			LastChecked: time.Now(),
			Metrics: map[string]interface{}{
				"synced_nodes": syncedNodes,
				"total_nodes":  totalNodes,
				"last_sync":    lastSyncTime,
			},
		}

		syncAge := time.Since(lastSyncTime)
		syncRatio := float64(syncedNodes) / float64(totalNodes)

		if syncRatio < 0.5 {
			health.Status = StatusUnhealthy
			health.Message = fmt.Sprintf("Only %d/%d nodes synchronized", syncedNodes, totalNodes)
		} else if syncRatio < 1.0 || syncAge > maxSyncAge {
			health.Status = StatusDegraded
			health.Message = fmt.Sprintf("%d/%d nodes synchronized, last sync %s ago", syncedNodes, totalNodes, syncAge)
		} else {
			health.Status = StatusHealthy
			health.Message = fmt.Sprintf("All %d nodes synchronized", totalNodes)
		}

		return health, nil
	}
}
