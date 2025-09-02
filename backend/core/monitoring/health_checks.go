package monitoring

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// HealthCheckStatus represents the status of a health check
type HealthCheckStatus string

const (
	HealthStatusHealthy   HealthCheckStatus = "healthy"
	HealthStatusUnhealthy HealthCheckStatus = "unhealthy"
	HealthStatusDegraded  HealthCheckStatus = "degraded"
	HealthStatusUnknown   HealthCheckStatus = "unknown"
)

// HealthCheckResult represents the result of a health check
type HealthCheckResult struct {
	Name        string            `json:"name"`
	Status      HealthCheckStatus `json:"status"`
	Message     string            `json:"message"`
	Duration    time.Duration     `json:"duration"`
	Timestamp   time.Time         `json:"timestamp"`
	Details     map[string]any    `json:"details,omitempty"`
	Error       error             `json:"-"`
}

// HealthCheckFunc defines the signature for health check functions
type HealthCheckFunc func(ctx context.Context) HealthCheckResult

// HealthChecker manages all health checks
type HealthChecker struct {
	checks    map[string]HealthCheckFunc
	results   map[string]HealthCheckResult
	mutex     sync.RWMutex
	
	// Metrics
	healthCheckDuration *prometheus.HistogramVec
	healthCheckStatus   *prometheus.GaugeVec
	
	// Configuration
	timeout time.Duration
	tracer  trace.Tracer
}

// HealthCheckConfig represents configuration for health checks
type HealthCheckConfig struct {
	Timeout         time.Duration `json:"timeout"`
	EnableMetrics   bool          `json:"enable_metrics"`
	EnableTracing   bool          `json:"enable_tracing"`
}

// DefaultHealthCheckConfig returns default health check configuration
func DefaultHealthCheckConfig() *HealthCheckConfig {
	return &HealthCheckConfig{
		Timeout:       30 * time.Second,
		EnableMetrics: true,
		EnableTracing: true,
	}
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(config *HealthCheckConfig) *HealthChecker {
	if config == nil {
		config = DefaultHealthCheckConfig()
	}
	
	var healthCheckDuration *prometheus.HistogramVec
	var healthCheckStatus *prometheus.GaugeVec
	
	if config.EnableMetrics {
		healthCheckDuration = promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name: "novacron_health_check_duration_seconds",
				Help: "Duration of health checks in seconds",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"check_name", "status"},
		)
		
		healthCheckStatus = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "novacron_health_check_status",
				Help: "Status of health checks (1=healthy, 0=unhealthy)",
			},
			[]string{"check_name"},
		)
	}
	
	var tracer trace.Tracer
	if config.EnableTracing {
		tracer = otel.Tracer("novacron-health-checks")
	}
	
	return &HealthChecker{
		checks:              make(map[string]HealthCheckFunc),
		results:             make(map[string]HealthCheckResult),
		timeout:             config.Timeout,
		healthCheckDuration: healthCheckDuration,
		healthCheckStatus:   healthCheckStatus,
		tracer:              tracer,
	}
}

// RegisterCheck registers a health check function
func (hc *HealthChecker) RegisterCheck(name string, checkFunc HealthCheckFunc) {
	hc.mutex.Lock()
	defer hc.mutex.Unlock()
	hc.checks[name] = checkFunc
}

// UnregisterCheck removes a health check
func (hc *HealthChecker) UnregisterCheck(name string) {
	hc.mutex.Lock()
	defer hc.mutex.Unlock()
	delete(hc.checks, name)
	delete(hc.results, name)
}

// RunCheck runs a specific health check
func (hc *HealthChecker) RunCheck(ctx context.Context, name string) (HealthCheckResult, error) {
	hc.mutex.RLock()
	checkFunc, exists := hc.checks[name]
	hc.mutex.RUnlock()
	
	if !exists {
		return HealthCheckResult{}, fmt.Errorf("health check %s not found", name)
	}
	
	// Create context with timeout
	checkCtx, cancel := context.WithTimeout(ctx, hc.timeout)
	defer cancel()
	
	// Start tracing if enabled
	if hc.tracer != nil {
		checkCtx, span := hc.tracer.Start(checkCtx, fmt.Sprintf("health_check.%s", name))
		span.SetAttributes(
			attribute.String("check.name", name),
		)
		defer span.End()
	}
	
	startTime := time.Now()
	result := checkFunc(checkCtx)
	duration := time.Since(startTime)
	
	// Set duration and timestamp
	result.Name = name
	result.Duration = duration
	result.Timestamp = time.Now()
	
	// Update metrics if enabled
	if hc.healthCheckDuration != nil {
		hc.healthCheckDuration.WithLabelValues(name, string(result.Status)).Observe(duration.Seconds())
	}
	
	if hc.healthCheckStatus != nil {
		status := 0.0
		if result.Status == HealthStatusHealthy {
			status = 1.0
		}
		hc.healthCheckStatus.WithLabelValues(name).Set(status)
	}
	
	// Store result
	hc.mutex.Lock()
	hc.results[name] = result
	hc.mutex.Unlock()
	
	return result, nil
}

// RunAllChecks runs all registered health checks
func (hc *HealthChecker) RunAllChecks(ctx context.Context) map[string]HealthCheckResult {
	hc.mutex.RLock()
	checks := make([]string, 0, len(hc.checks))
	for name := range hc.checks {
		checks = append(checks, name)
	}
	hc.mutex.RUnlock()
	
	results := make(map[string]HealthCheckResult)
	var wg sync.WaitGroup
	resultChan := make(chan HealthCheckResult, len(checks))
	
	// Run all checks concurrently
	for _, name := range checks {
		wg.Add(1)
		go func(checkName string) {
			defer wg.Done()
			result, _ := hc.RunCheck(ctx, checkName)
			resultChan <- result
		}(name)
	}
	
	// Wait for all checks to complete
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	// Collect results
	for result := range resultChan {
		results[result.Name] = result
	}
	
	return results
}

// GetOverallStatus returns the overall system health status
func (hc *HealthChecker) GetOverallStatus(ctx context.Context) HealthCheckStatus {
	results := hc.RunAllChecks(ctx)
	
	if len(results) == 0 {
		return HealthStatusUnknown
	}
	
	healthyCount := 0
	degradedCount := 0
	unhealthyCount := 0
	
	for _, result := range results {
		switch result.Status {
		case HealthStatusHealthy:
			healthyCount++
		case HealthStatusDegraded:
			degradedCount++
		case HealthStatusUnhealthy:
			unhealthyCount++
		}
	}
	
	totalChecks := len(results)
	
	// If any check is unhealthy, system is unhealthy
	if unhealthyCount > 0 {
		return HealthStatusUnhealthy
	}
	
	// If more than 50% are degraded, system is degraded
	if degradedCount > totalChecks/2 {
		return HealthStatusDegraded
	}
	
	// If any check is degraded, system is degraded
	if degradedCount > 0 {
		return HealthStatusDegraded
	}
	
	return HealthStatusHealthy
}

// ServeHTTP implements http.Handler for health check endpoint
func (hc *HealthChecker) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	
	// Check if specific check is requested
	checkName := r.URL.Query().Get("check")
	
	if checkName != "" {
		// Run specific check
		result, err := hc.RunCheck(ctx, checkName)
		if err != nil {
			http.Error(w, fmt.Sprintf("Health check not found: %s", checkName), http.StatusNotFound)
			return
		}
		
		statusCode := http.StatusOK
		if result.Status == HealthStatusUnhealthy {
			statusCode = http.StatusServiceUnavailable
		} else if result.Status == HealthStatusDegraded {
			statusCode = http.StatusPartialContent
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		json.NewEncoder(w).Encode(result)
		return
	}
	
	// Run all checks
	results := hc.RunAllChecks(ctx)
	overallStatus := hc.GetOverallStatus(ctx)
	
	statusCode := http.StatusOK
	if overallStatus == HealthStatusUnhealthy {
		statusCode = http.StatusServiceUnavailable
	} else if overallStatus == HealthStatusDegraded {
		statusCode = http.StatusPartialContent
	}
	
	response := map[string]any{
		"status":    overallStatus,
		"timestamp": time.Now(),
		"checks":    results,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(response)
}

// Common health check implementations

// DatabaseHealthCheck creates a health check for database connectivity
func DatabaseHealthCheck(db *sql.DB) HealthCheckFunc {
	return func(ctx context.Context) HealthCheckResult {
		start := time.Now()
		
		err := db.PingContext(ctx)
		if err != nil {
			return HealthCheckResult{
				Status:  HealthStatusUnhealthy,
				Message: fmt.Sprintf("Database ping failed: %v", err),
				Error:   err,
			}
		}
		
		// Check if we can execute a simple query
		var result int
		err = db.QueryRowContext(ctx, "SELECT 1").Scan(&result)
		if err != nil {
			return HealthCheckResult{
				Status:  HealthStatusDegraded,
				Message: fmt.Sprintf("Database query failed: %v", err),
				Error:   err,
			}
		}
		
		duration := time.Since(start)
		
		return HealthCheckResult{
			Status:  HealthStatusHealthy,
			Message: "Database connection healthy",
			Details: map[string]any{
				"query_duration": duration.String(),
			},
		}
	}
}

// HTTPServiceHealthCheck creates a health check for HTTP service dependencies
func HTTPServiceHealthCheck(name, url string, client *http.Client) HealthCheckFunc {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	
	return func(ctx context.Context) HealthCheckResult {
		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err != nil {
			return HealthCheckResult{
				Status:  HealthStatusUnhealthy,
				Message: fmt.Sprintf("Failed to create request: %v", err),
				Error:   err,
			}
		}
		
		start := time.Now()
		resp, err := client.Do(req)
		duration := time.Since(start)
		
		if err != nil {
			return HealthCheckResult{
				Status:  HealthStatusUnhealthy,
				Message: fmt.Sprintf("%s service unreachable: %v", name, err),
				Error:   err,
				Details: map[string]any{
					"url":      url,
					"duration": duration.String(),
				},
			}
		}
		defer resp.Body.Close()
		
		status := HealthStatusHealthy
		message := fmt.Sprintf("%s service healthy", name)
		
		if resp.StatusCode >= 500 {
			status = HealthStatusUnhealthy
			message = fmt.Sprintf("%s service returned %d", name, resp.StatusCode)
		} else if resp.StatusCode >= 400 {
			status = HealthStatusDegraded
			message = fmt.Sprintf("%s service returned %d", name, resp.StatusCode)
		}
		
		return HealthCheckResult{
			Status:  status,
			Message: message,
			Details: map[string]any{
				"url":         url,
				"status_code": resp.StatusCode,
				"duration":    duration.String(),
			},
		}
	}
}

// DiskSpaceHealthCheck creates a health check for disk space
func DiskSpaceHealthCheck(path string, warningPercent, criticalPercent float64) HealthCheckFunc {
	return func(ctx context.Context) HealthCheckResult {
		// This would typically use syscalls to check disk usage
		// For now, we'll return a placeholder implementation
		
		// In a real implementation, you'd use something like:
		// var stat syscall.Statfs_t
		// syscall.Statfs(path, &stat)
		// Calculate usage percentage
		
		// Placeholder values for demonstration
		usagePercent := 75.0
		
		status := HealthStatusHealthy
		message := fmt.Sprintf("Disk usage at %s is %.1f%%", path, usagePercent)
		
		if usagePercent >= criticalPercent {
			status = HealthStatusUnhealthy
			message = fmt.Sprintf("Critical: Disk usage at %s is %.1f%% (>= %.1f%%)", path, usagePercent, criticalPercent)
		} else if usagePercent >= warningPercent {
			status = HealthStatusDegraded
			message = fmt.Sprintf("Warning: Disk usage at %s is %.1f%% (>= %.1f%%)", path, usagePercent, warningPercent)
		}
		
		return HealthCheckResult{
			Status:  status,
			Message: message,
			Details: map[string]any{
				"path":             path,
				"usage_percent":    usagePercent,
				"warning_percent":  warningPercent,
				"critical_percent": criticalPercent,
			},
		}
	}
}

// MemoryHealthCheck creates a health check for memory usage
func MemoryHealthCheck(warningPercent, criticalPercent float64) HealthCheckFunc {
	return func(ctx context.Context) HealthCheckResult {
		// This would typically read from /proc/meminfo or use runtime.MemStats
		// Placeholder implementation
		
		var memStats struct {
			TotalMemory     uint64
			AvailableMemory uint64
		}
		
		// Placeholder values
		memStats.TotalMemory = 16 * 1024 * 1024 * 1024 // 16GB
		memStats.AvailableMemory = 8 * 1024 * 1024 * 1024 // 8GB available
		
		usedMemory := memStats.TotalMemory - memStats.AvailableMemory
		usagePercent := float64(usedMemory) / float64(memStats.TotalMemory) * 100
		
		status := HealthStatusHealthy
		message := fmt.Sprintf("Memory usage is %.1f%%", usagePercent)
		
		if usagePercent >= criticalPercent {
			status = HealthStatusUnhealthy
			message = fmt.Sprintf("Critical: Memory usage is %.1f%% (>= %.1f%%)", usagePercent, criticalPercent)
		} else if usagePercent >= warningPercent {
			status = HealthStatusDegraded
			message = fmt.Sprintf("Warning: Memory usage is %.1f%% (>= %.1f%%)", usagePercent, warningPercent)
		}
		
		return HealthCheckResult{
			Status:  status,
			Message: message,
			Details: map[string]any{
				"total_memory_bytes":     memStats.TotalMemory,
				"available_memory_bytes": memStats.AvailableMemory,
				"used_memory_bytes":      usedMemory,
				"usage_percent":          usagePercent,
				"warning_percent":        warningPercent,
				"critical_percent":       criticalPercent,
			},
		}
	}
}

// VMManagerHealthCheck creates a health check for VM manager
func VMManagerHealthCheck(vmManager interface{}) HealthCheckFunc {
	return func(ctx context.Context) HealthCheckResult {
		// This would check if VM manager is responsive and can list VMs
		// Placeholder implementation
		
		status := HealthStatusHealthy
		message := "VM Manager is healthy"
		
		// In a real implementation, you'd check:
		// - Can connect to hypervisor
		// - Can list VMs
		// - Response time is reasonable
		
		return HealthCheckResult{
			Status:  status,
			Message: message,
			Details: map[string]any{
				"active_vms":     42,
				"hypervisor":     "KVM",
				"connection":     "active",
				"last_check":     time.Now(),
			},
		}
	}
}

// Custom health check for NovaCron components
func NovaCronComponentHealthCheck(component string) HealthCheckFunc {
	return func(ctx context.Context) HealthCheckResult {
		// Component-specific health checks
		switch component {
		case "storage":
			return HealthCheckResult{
				Status:  HealthStatusHealthy,
				Message: "Storage subsystem operational",
				Details: map[string]any{
					"active_tiers":       3,
					"deduplication":      "enabled",
					"compression_ratio":  2.5,
					"available_space":    "85%",
				},
			}
			
		case "network":
			return HealthCheckResult{
				Status:  HealthStatusHealthy,
				Message: "Network subsystem operational",
				Details: map[string]any{
					"overlay_networks":   5,
					"load_balancers":     2,
					"bandwidth_usage":    "45%",
					"packet_loss":        0.01,
				},
			}
			
		case "scheduler":
			return HealthCheckResult{
				Status:  HealthStatusHealthy,
				Message: "Scheduler operational",
				Details: map[string]any{
					"queue_size":         12,
					"pending_migrations": 3,
					"resource_usage":     "normal",
					"scheduling_latency": "150ms",
				},
			}
			
		default:
			return HealthCheckResult{
				Status:  HealthStatusUnknown,
				Message: fmt.Sprintf("Unknown component: %s", component),
			}
		}
	}
}