package loadbalancer

import (
	"bytes"
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// AdvancedHealthChecker provides comprehensive health monitoring
type AdvancedHealthChecker struct {
	// Configuration
	ID                    string                 `json:"id"`
	Name                  string                 `json:"name"`
	ServiceID             string                 `json:"service_id"`
	BackendID             string                 `json:"backend_id"`
	Config                AdvancedHealthConfig   `json:"config"`
	
	// State management
	Status                HealthStatus           `json:"status"`
	ConsecutivePasses     int32                  `json:"consecutive_passes"`
	ConsecutiveFails      int32                  `json:"consecutive_fails"`
	TotalChecks           int64                  `json:"total_checks"`
	TotalPasses           int64                  `json:"total_passes"`
	TotalFails            int64                  `json:"total_fails"`
	
	// Circuit breaker state
	CircuitState          CircuitBreakerState    `json:"circuit_state"`
	CircuitFailures       int32                  `json:"circuit_failures"`
	CircuitLastFailure    time.Time              `json:"circuit_last_failure"`
	CircuitResetTimeout   time.Duration          `json:"circuit_reset_timeout"`
	
	// Timing and metrics
	LastCheck             time.Time              `json:"last_check"`
	LastSuccess           time.Time              `json:"last_success"`
	LastFailure           time.Time              `json:"last_failure"`
	LastError             string                 `json:"last_error"`
	AverageLatency        float64                `json:"average_latency_ms"`
	LatencyP95            float64                `json:"latency_p95_ms"`
	LatencyP99            float64                `json:"latency_p99_ms"`
	
	// Passive monitoring
	PassiveMetrics        PassiveHealthMetrics   `json:"passive_metrics"`
	
	// Control channels
	ticker                *time.Ticker
	stopCh                chan struct{}
	ctx                   context.Context
	cancel                context.CancelFunc
	
	// Synchronization
	mu                    sync.RWMutex
	latencyHistory        []float64
	passiveMu             sync.RWMutex
}

// AdvancedHealthConfig holds comprehensive health check configuration
type AdvancedHealthConfig struct {
	// Basic health check settings
	Type                  HealthCheckType        `json:"type"`
	Interval              time.Duration          `json:"interval"`
	Timeout               time.Duration          `json:"timeout"`
	HealthyThreshold      int                    `json:"healthy_threshold"`
	UnhealthyThreshold    int                    `json:"unhealthy_threshold"`
	
	// Active check configuration
	Path                  string                 `json:"path,omitempty"`
	Method                string                 `json:"method,omitempty"`
	ExpectedStatus        []int                  `json:"expected_status,omitempty"`
	ExpectedBody          string                 `json:"expected_body,omitempty"`
	ExpectedHeaders       map[string]string      `json:"expected_headers,omitempty"`
	RequestHeaders        map[string]string      `json:"request_headers,omitempty"`
	RequestBody           string                 `json:"request_body,omitempty"`
	
	// TLS configuration
	TLSConfig             *TLSHealthCheckConfig  `json:"tls_config,omitempty"`
	
	// Circuit breaker settings
	EnableCircuitBreaker  bool                   `json:"enable_circuit_breaker"`
	CircuitFailureThreshold int                  `json:"circuit_failure_threshold"`
	CircuitResetTimeout   time.Duration          `json:"circuit_reset_timeout"`
	CircuitHalfOpenRequests int                  `json:"circuit_half_open_requests"`
	
	// Passive monitoring settings
	EnablePassiveMonitoring bool                 `json:"enable_passive_monitoring"`
	PassiveErrorThreshold   float64              `json:"passive_error_threshold"`
	PassiveLatencyThreshold time.Duration        `json:"passive_latency_threshold"`
	PassiveWindowSize       time.Duration        `json:"passive_window_size"`
	
	// Advanced settings
	JitterPercent         int                    `json:"jitter_percent"`
	RetryCount            int                    `json:"retry_count"`
	RetryDelay            time.Duration          `json:"retry_delay"`
	DNSLookup             bool                   `json:"dns_lookup"`
	KeepAlive             bool                   `json:"keep_alive"`
}

// TLSHealthCheckConfig holds TLS-specific health check configuration
type TLSHealthCheckConfig struct {
	SkipVerify            bool                   `json:"skip_verify"`
	ServerName            string                 `json:"server_name,omitempty"`
	ClientCertPath        string                 `json:"client_cert_path,omitempty"`
	ClientKeyPath         string                 `json:"client_key_path,omitempty"`
	CACertPath            string                 `json:"ca_cert_path,omitempty"`
	MinVersion            uint16                 `json:"min_version,omitempty"`
	MaxVersion            uint16                 `json:"max_version,omitempty"`
	CipherSuites          []uint16               `json:"cipher_suites,omitempty"`
}

// PassiveHealthMetrics holds passive monitoring metrics
type PassiveHealthMetrics struct {
	RequestCount          int64                  `json:"request_count"`
	ErrorCount            int64                  `json:"error_count"`
	ErrorRate             float64                `json:"error_rate"`
	AverageLatency        float64                `json:"average_latency_ms"`
	P95Latency            float64                `json:"p95_latency_ms"`
	P99Latency            float64                `json:"p99_latency_ms"`
	LastUpdated           time.Time              `json:"last_updated"`
	WindowStart           time.Time              `json:"window_start"`
}

// CircuitBreakerState represents circuit breaker states
type CircuitBreakerState string

const (
	CircuitStateClosed     CircuitBreakerState = "closed"
	CircuitStateOpen       CircuitBreakerState = "open"
	CircuitStateHalfOpen   CircuitBreakerState = "half_open"
)

// NewAdvancedHealthChecker creates a new advanced health checker
func NewAdvancedHealthChecker(serviceID, backendID string, config AdvancedHealthConfig) *AdvancedHealthChecker {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &AdvancedHealthChecker{
		ID:                    uuid.New().String(),
		Name:                  fmt.Sprintf("health-checker-%s-%s", serviceID, backendID),
		ServiceID:             serviceID,
		BackendID:             backendID,
		Config:                config,
		Status: HealthStatus{
			Status:      "unknown",
			LastChecked: time.Now(),
			Details:     make(map[string]interface{}),
		},
		CircuitState:          CircuitStateClosed,
		CircuitResetTimeout:   config.CircuitResetTimeout,
		PassiveMetrics: PassiveHealthMetrics{
			LastUpdated: time.Now(),
			WindowStart: time.Now(),
		},
		stopCh:                make(chan struct{}),
		ctx:                   ctx,
		cancel:                cancel,
		latencyHistory:        make([]float64, 0, 1000), // Keep last 1000 measurements
	}
}

// Start begins health checking
func (hc *AdvancedHealthChecker) Start(backend *Backend) error {
	// Set defaults
	if hc.Config.Interval == 0 {
		hc.Config.Interval = 30 * time.Second
	}
	if hc.Config.Timeout == 0 {
		hc.Config.Timeout = 5 * time.Second
	}
	if hc.Config.HealthyThreshold == 0 {
		hc.Config.HealthyThreshold = 2
	}
	if hc.Config.UnhealthyThreshold == 0 {
		hc.Config.UnhealthyThreshold = 3
	}
	if hc.Config.CircuitFailureThreshold == 0 {
		hc.Config.CircuitFailureThreshold = 5
	}
	if hc.Config.CircuitResetTimeout == 0 {
		hc.Config.CircuitResetTimeout = 60 * time.Second
	}
	if hc.Config.PassiveErrorThreshold == 0 {
		hc.Config.PassiveErrorThreshold = 0.1 // 10% error rate
	}
	if hc.Config.PassiveWindowSize == 0 {
		hc.Config.PassiveWindowSize = 5 * time.Minute
	}
	
	// Add jitter to interval to prevent thundering herd
	interval := hc.Config.Interval
	if hc.Config.JitterPercent > 0 {
		jitter := time.Duration(float64(interval) * float64(hc.Config.JitterPercent) / 100.0)
		interval = interval + time.Duration(time.Now().UnixNano()%int64(jitter))
	}
	
	hc.ticker = time.NewTicker(interval)
	
	// Start active health checking loop
	go hc.activeHealthCheckLoop(backend)
	
	// Start passive monitoring if enabled
	if hc.Config.EnablePassiveMonitoring {
		go hc.passiveMonitoringLoop(backend)
	}
	
	// Start circuit breaker reset loop if enabled
	if hc.Config.EnableCircuitBreaker {
		go hc.circuitBreakerResetLoop(backend)
	}
	
	return nil
}

// Stop stops health checking
func (hc *AdvancedHealthChecker) Stop() {
	if hc.ticker != nil {
		hc.ticker.Stop()
	}
	close(hc.stopCh)
	hc.cancel()
}

// activeHealthCheckLoop performs periodic active health checks
func (hc *AdvancedHealthChecker) activeHealthCheckLoop(backend *Backend) {
	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-hc.stopCh:
			return
		case <-hc.ticker.C:
			// Skip active check if circuit breaker is open
			if hc.Config.EnableCircuitBreaker && hc.CircuitState == CircuitStateOpen {
				continue
			}
			
			hc.performActiveHealthCheck(backend)
		}
	}
}

// performActiveHealthCheck performs a single active health check
func (hc *AdvancedHealthChecker) performActiveHealthCheck(backend *Backend) {
	startTime := time.Now()
	hc.LastCheck = startTime
	
	var healthy bool
	var err error
	var latency time.Duration
	
	// Perform health check based on type
	switch hc.Config.Type {
	case HealthCheckTypeHTTP, HealthCheckTypeHTTPS:
		healthy, latency, err = hc.performHTTPHealthCheck(backend)
	case HealthCheckTypeTCP:
		healthy, latency, err = hc.performTCPHealthCheck(backend)
	case HealthCheckTypeUDP:
		healthy, latency, err = hc.performUDPHealthCheck(backend)
	case HealthCheckTypeICMP:
		healthy, latency, err = hc.performICMPHealthCheck(backend)
	default:
		healthy, latency, err = hc.performTCPHealthCheck(backend)
	}
	
	// Update statistics
	hc.mu.Lock()
	atomic.AddInt64(&hc.TotalChecks, 1)
	
	if healthy {
		atomic.AddInt64(&hc.TotalPasses, 1)
		atomic.AddInt32(&hc.ConsecutivePasses, 1)
		atomic.StoreInt32(&hc.ConsecutiveFails, 0)
		hc.LastSuccess = startTime
		hc.LastError = ""
		
		// Reset circuit breaker on success
		if hc.Config.EnableCircuitBreaker && hc.CircuitState != CircuitStateClosed {
			hc.resetCircuitBreaker()
		}
	} else {
		atomic.AddInt64(&hc.TotalFails, 1)
		atomic.AddInt32(&hc.ConsecutiveFails, 1)
		atomic.StoreInt32(&hc.ConsecutivePasses, 0)
		hc.LastFailure = startTime
		if err != nil {
			hc.LastError = err.Error()
		}
		
		// Update circuit breaker on failure
		if hc.Config.EnableCircuitBreaker {
			hc.recordCircuitBreakerFailure()
		}
	}
	
	// Update latency metrics
	if latency > 0 {
		latencyMs := float64(latency.Nanoseconds()) / 1e6
		hc.updateLatencyMetrics(latencyMs)
	}
	
	// Determine health status
	hc.updateHealthStatus(backend)
	
	hc.mu.Unlock()
}

// performHTTPHealthCheck performs HTTP/HTTPS health check
func (hc *AdvancedHealthChecker) performHTTPHealthCheck(backend *Backend) (bool, time.Duration, error) {
	scheme := "http"
	if hc.Config.Type == HealthCheckTypeHTTPS {
		scheme = "https"
	}
	
	path := hc.Config.Path
	if path == "" {
		path = "/"
	}
	
	url := fmt.Sprintf("%s://%s:%d%s", scheme, backend.Address, backend.Port, path)
	
	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: hc.Config.Timeout,
	}
	
	// Configure TLS if needed
	if hc.Config.Type == HealthCheckTypeHTTPS && hc.Config.TLSConfig != nil {
		tlsConfig := &tls.Config{
			InsecureSkipVerify: hc.Config.TLSConfig.SkipVerify,
			ServerName:         hc.Config.TLSConfig.ServerName,
			MinVersion:         hc.Config.TLSConfig.MinVersion,
			MaxVersion:         hc.Config.TLSConfig.MaxVersion,
			CipherSuites:       hc.Config.TLSConfig.CipherSuites,
		}
		
		// Load client certificate if provided
		if hc.Config.TLSConfig.ClientCertPath != "" && hc.Config.TLSConfig.ClientKeyPath != "" {
			cert, err := tls.LoadX509KeyPair(hc.Config.TLSConfig.ClientCertPath, hc.Config.TLSConfig.ClientKeyPath)
			if err != nil {
				return false, 0, fmt.Errorf("failed to load client certificate: %w", err)
			}
			tlsConfig.Certificates = []tls.Certificate{cert}
		}
		
		client.Transport = &http.Transport{
			TLSClientConfig: tlsConfig,
		}
	}
	
	// Create request
	method := hc.Config.Method
	if method == "" {
		method = "GET"
	}
	
	var reqBody *bytes.Reader
	if hc.Config.RequestBody != "" {
		reqBody = bytes.NewReader([]byte(hc.Config.RequestBody))
	}
	
	var req *http.Request
	var err error
	
	if reqBody != nil {
		req, err = http.NewRequestWithContext(hc.ctx, method, url, reqBody)
	} else {
		req, err = http.NewRequestWithContext(hc.ctx, method, url, nil)
	}
	
	if err != nil {
		return false, 0, fmt.Errorf("failed to create request: %w", err)
	}
	
	// Add request headers
	for key, value := range hc.Config.RequestHeaders {
		req.Header.Set(key, value)
	}
	
	// Set default User-Agent if not provided
	if req.Header.Get("User-Agent") == "" {
		req.Header.Set("User-Agent", "NovaCron-HealthChecker/1.0")
	}
	
	// Perform request with retry logic
	var resp *http.Response
	var latency time.Duration
	
	for attempt := 0; attempt <= hc.Config.RetryCount; attempt++ {
		if attempt > 0 && hc.Config.RetryDelay > 0 {
			time.Sleep(hc.Config.RetryDelay)
		}
		
		start := time.Now()
		resp, err = client.Do(req)
		latency = time.Since(start)
		
		if err == nil {
			break
		}
		
		if attempt == hc.Config.RetryCount {
			return false, latency, fmt.Errorf("all %d attempts failed, last error: %w", attempt+1, err)
		}
	}
	
	defer resp.Body.Close()
	
	// Check status code
	expectedStatuses := hc.Config.ExpectedStatus
	if len(expectedStatuses) == 0 {
		expectedStatuses = []int{200}
	}
	
	statusOK := false
	for _, expected := range expectedStatuses {
		if resp.StatusCode == expected {
			statusOK = true
			break
		}
	}
	
	if !statusOK {
		return false, latency, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	// Check response headers if specified
	for key, expectedValue := range hc.Config.ExpectedHeaders {
		actualValue := resp.Header.Get(key)
		if actualValue != expectedValue {
			return false, latency, fmt.Errorf("header %s: expected %s, got %s", key, expectedValue, actualValue)
		}
	}
	
	// Check response body if specified
	if hc.Config.ExpectedBody != "" {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return false, latency, fmt.Errorf("failed to read response body: %w", err)
		}
		
		if string(body) != hc.Config.ExpectedBody {
			return false, latency, fmt.Errorf("unexpected response body")
		}
	}
	
	return true, latency, nil
}

// performTCPHealthCheck performs TCP health check
func (hc *AdvancedHealthChecker) performTCPHealthCheck(backend *Backend) (bool, time.Duration, error) {
	address := fmt.Sprintf("%s:%d", backend.Address, backend.Port)
	
	start := time.Now()
	conn, err := net.DialTimeout("tcp", address, hc.Config.Timeout)
	latency := time.Since(start)
	
	if err != nil {
		return false, latency, err
	}
	
	conn.Close()
	return true, latency, nil
}

// performUDPHealthCheck performs UDP health check
func (hc *AdvancedHealthChecker) performUDPHealthCheck(backend *Backend) (bool, time.Duration, error) {
	address := fmt.Sprintf("%s:%d", backend.Address, backend.Port)
	
	start := time.Now()
	conn, err := net.DialTimeout("udp", address, hc.Config.Timeout)
	latency := time.Since(start)
	
	if err != nil {
		return false, latency, err
	}
	
	// Send a simple probe packet
	_, err = conn.Write([]byte("health-check"))
	if err != nil {
		conn.Close()
		return false, latency, err
	}
	
	conn.Close()
	return true, latency, nil
}

// performICMPHealthCheck performs ICMP (ping) health check
func (hc *AdvancedHealthChecker) performICMPHealthCheck(backend *Backend) (bool, time.Duration, error) {
	// Note: ICMP requires root privileges or special permissions
	// This is a simplified implementation that uses TCP fallback
	return hc.performTCPHealthCheck(backend)
}

// updateLatencyMetrics updates latency statistics
func (hc *AdvancedHealthChecker) updateLatencyMetrics(latencyMs float64) {
	// Add to history
	hc.latencyHistory = append(hc.latencyHistory, latencyMs)
	
	// Keep only last 1000 measurements
	if len(hc.latencyHistory) > 1000 {
		hc.latencyHistory = hc.latencyHistory[len(hc.latencyHistory)-1000:]
	}
	
	// Calculate average
	var sum float64
	for _, l := range hc.latencyHistory {
		sum += l
	}
	hc.AverageLatency = sum / float64(len(hc.latencyHistory))
	
	// Calculate percentiles
	if len(hc.latencyHistory) >= 20 { // Minimum samples for percentiles
		sorted := make([]float64, len(hc.latencyHistory))
		copy(sorted, hc.latencyHistory)
		
		// Simple sort for percentiles
		for i := 0; i < len(sorted)-1; i++ {
			for j := i + 1; j < len(sorted); j++ {
				if sorted[i] > sorted[j] {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}
		
		p95Index := int(float64(len(sorted)) * 0.95)
		p99Index := int(float64(len(sorted)) * 0.99)
		
		if p95Index < len(sorted) {
			hc.LatencyP95 = sorted[p95Index]
		}
		if p99Index < len(sorted) {
			hc.LatencyP99 = sorted[p99Index]
		}
	}
}

// updateHealthStatus updates the health status based on consecutive results
func (hc *AdvancedHealthChecker) updateHealthStatus(backend *Backend) {
	consecutivePasses := atomic.LoadInt32(&hc.ConsecutivePasses)
	consecutiveFails := atomic.LoadInt32(&hc.ConsecutiveFails)
	
	// Check circuit breaker state
	if hc.Config.EnableCircuitBreaker && hc.CircuitState == CircuitStateOpen {
		hc.Status = HealthStatus{Status: "unhealthy", LastChecked: time.Now(), Details: make(map[string]interface{})}
		backend.HealthStatus = HealthStatusUnhealthy
		return
	}
	
	// Check passive monitoring results
	if hc.Config.EnablePassiveMonitoring {
		hc.passiveMu.RLock()
		passiveErrorRate := hc.PassiveMetrics.ErrorRate
		passiveLatency := hc.PassiveMetrics.AverageLatency
		hc.passiveMu.RUnlock()
		
		if passiveErrorRate > hc.Config.PassiveErrorThreshold ||
			(hc.Config.PassiveLatencyThreshold > 0 && time.Duration(passiveLatency)*time.Millisecond > hc.Config.PassiveLatencyThreshold) {
			hc.Status = HealthStatus{Status: "unhealthy", LastChecked: time.Now(), Details: make(map[string]interface{})}
			backend.HealthStatus = HealthStatusUnhealthy
			return
		}
	}
	
	// Update status based on active checks
	if consecutivePasses >= int32(hc.Config.HealthyThreshold) {
		hc.Status = HealthStatus{Status: "healthy", LastChecked: time.Now(), Details: make(map[string]interface{})}
		backend.HealthStatus = HealthStatusHealthy
	} else if consecutiveFails >= int32(hc.Config.UnhealthyThreshold) {
		hc.Status = HealthStatus{Status: "unhealthy", LastChecked: time.Now(), Details: make(map[string]interface{})}
		backend.HealthStatus = HealthStatusUnhealthy
	}
}

// Circuit breaker implementation

// recordCircuitBreakerFailure records a failure for circuit breaker
func (hc *AdvancedHealthChecker) recordCircuitBreakerFailure() {
	atomic.AddInt32(&hc.CircuitFailures, 1)
	hc.CircuitLastFailure = time.Now()
	
	failures := atomic.LoadInt32(&hc.CircuitFailures)
	if failures >= int32(hc.Config.CircuitFailureThreshold) {
		hc.CircuitState = CircuitStateOpen
	}
}

// resetCircuitBreaker resets the circuit breaker to closed state
func (hc *AdvancedHealthChecker) resetCircuitBreaker() {
	atomic.StoreInt32(&hc.CircuitFailures, 0)
	hc.CircuitState = CircuitStateClosed
}

// circuitBreakerResetLoop handles circuit breaker reset timing
func (hc *AdvancedHealthChecker) circuitBreakerResetLoop(backend *Backend) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-hc.stopCh:
			return
		case <-ticker.C:
			if hc.CircuitState == CircuitStateOpen {
				if time.Since(hc.CircuitLastFailure) > hc.CircuitResetTimeout {
					hc.CircuitState = CircuitStateHalfOpen
				}
			}
		}
	}
}

// Passive monitoring implementation

// recordPassiveMetrics records metrics from actual traffic
func (hc *AdvancedHealthChecker) RecordPassiveMetrics(success bool, latency time.Duration) {
	if !hc.Config.EnablePassiveMonitoring {
		return
	}
	
	hc.passiveMu.Lock()
	defer hc.passiveMu.Unlock()
	
	// Reset window if needed
	if time.Since(hc.PassiveMetrics.WindowStart) > hc.Config.PassiveWindowSize {
		hc.PassiveMetrics = PassiveHealthMetrics{
			LastUpdated: time.Now(),
			WindowStart: time.Now(),
		}
	}
	
	// Update metrics
	hc.PassiveMetrics.RequestCount++
	if !success {
		hc.PassiveMetrics.ErrorCount++
	}
	
	// Update error rate
	if hc.PassiveMetrics.RequestCount > 0 {
		hc.PassiveMetrics.ErrorRate = float64(hc.PassiveMetrics.ErrorCount) / float64(hc.PassiveMetrics.RequestCount)
	}
	
	// Update latency
	latencyMs := float64(latency.Nanoseconds()) / 1e6
	if hc.PassiveMetrics.AverageLatency == 0 {
		hc.PassiveMetrics.AverageLatency = latencyMs
	} else {
		// Exponential moving average
		hc.PassiveMetrics.AverageLatency = 0.9*hc.PassiveMetrics.AverageLatency + 0.1*latencyMs
	}
	
	hc.PassiveMetrics.LastUpdated = time.Now()
}

// passiveMonitoringLoop performs periodic passive monitoring evaluation
func (hc *AdvancedHealthChecker) passiveMonitoringLoop(backend *Backend) {
	ticker := time.NewTicker(10 * time.Second) // Check every 10 seconds
	defer ticker.Stop()
	
	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-hc.stopCh:
			return
		case <-ticker.C:
			hc.evaluatePassiveHealth(backend)
		}
	}
}

// evaluatePassiveHealth evaluates health based on passive monitoring
func (hc *AdvancedHealthChecker) evaluatePassiveHealth(backend *Backend) {
	hc.passiveMu.RLock()
	errorRate := hc.PassiveMetrics.ErrorRate
	avgLatency := hc.PassiveMetrics.AverageLatency
	requestCount := hc.PassiveMetrics.RequestCount
	hc.passiveMu.RUnlock()
	
	// Only evaluate if we have sufficient data
	if requestCount < 10 {
		return
	}
	
	// Check error rate threshold
	if errorRate > hc.Config.PassiveErrorThreshold {
		hc.mu.Lock()
		atomic.AddInt32(&hc.ConsecutiveFails, 1)
		atomic.StoreInt32(&hc.ConsecutivePasses, 0)
		hc.LastError = fmt.Sprintf("passive monitoring: error rate %.2f%% exceeds threshold %.2f%%", errorRate*100, hc.Config.PassiveErrorThreshold*100)
		hc.updateHealthStatus(backend)
		hc.mu.Unlock()
	}
	
	// Check latency threshold
	if hc.Config.PassiveLatencyThreshold > 0 && time.Duration(avgLatency)*time.Millisecond > hc.Config.PassiveLatencyThreshold {
		hc.mu.Lock()
		atomic.AddInt32(&hc.ConsecutiveFails, 1)
		atomic.StoreInt32(&hc.ConsecutivePasses, 0)
		hc.LastError = fmt.Sprintf("passive monitoring: average latency %.2fms exceeds threshold %v", avgLatency, hc.Config.PassiveLatencyThreshold)
		hc.updateHealthStatus(backend)
		hc.mu.Unlock()
	}
}

// GetMetrics returns comprehensive health check metrics
func (hc *AdvancedHealthChecker) GetMetrics() map[string]interface{} {
	hc.mu.RLock()
	hc.passiveMu.RLock()
	defer hc.mu.RUnlock()
	defer hc.passiveMu.RUnlock()
	
	totalChecks := atomic.LoadInt64(&hc.TotalChecks)
	totalPasses := atomic.LoadInt64(&hc.TotalPasses)
	totalFails := atomic.LoadInt64(&hc.TotalFails)
	consecutivePasses := atomic.LoadInt32(&hc.ConsecutivePasses)
	consecutiveFails := atomic.LoadInt32(&hc.ConsecutiveFails)
	circuitFailures := atomic.LoadInt32(&hc.CircuitFailures)
	
	successRate := 0.0
	if totalChecks > 0 {
		successRate = float64(totalPasses) / float64(totalChecks)
	}
	
	return map[string]interface{}{
		"id":                    hc.ID,
		"name":                  hc.Name,
		"service_id":            hc.ServiceID,
		"backend_id":            hc.BackendID,
		"status":                hc.Status,
		"circuit_state":         hc.CircuitState,
		"total_checks":          totalChecks,
		"total_passes":          totalPasses,
		"total_fails":           totalFails,
		"success_rate":          successRate,
		"consecutive_passes":    consecutivePasses,
		"consecutive_fails":     consecutiveFails,
		"circuit_failures":      circuitFailures,
		"last_check":            hc.LastCheck,
		"last_success":          hc.LastSuccess,
		"last_failure":          hc.LastFailure,
		"last_error":            hc.LastError,
		"average_latency_ms":    hc.AverageLatency,
		"latency_p95_ms":        hc.LatencyP95,
		"latency_p99_ms":        hc.LatencyP99,
		"passive_metrics":       hc.PassiveMetrics,
	}
}

// DefaultAdvancedHealthConfig returns default advanced health check configuration
func DefaultAdvancedHealthConfig() AdvancedHealthConfig {
	return AdvancedHealthConfig{
		Type:                     HealthCheckTypeHTTP,
		Interval:                 30 * time.Second,
		Timeout:                  5 * time.Second,
		HealthyThreshold:         2,
		UnhealthyThreshold:       3,
		Method:                   "GET",
		ExpectedStatus:           []int{200},
		EnableCircuitBreaker:     true,
		CircuitFailureThreshold:  5,
		CircuitResetTimeout:      60 * time.Second,
		CircuitHalfOpenRequests:  3,
		EnablePassiveMonitoring:  true,
		PassiveErrorThreshold:    0.1, // 10%
		PassiveLatencyThreshold:  1 * time.Second,
		PassiveWindowSize:        5 * time.Minute,
		JitterPercent:            10,
		RetryCount:               2,
		RetryDelay:               1 * time.Second,
		DNSLookup:                true,
		KeepAlive:                true,
	}
}