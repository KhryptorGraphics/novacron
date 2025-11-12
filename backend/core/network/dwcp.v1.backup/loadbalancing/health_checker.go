package loadbalancing

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

// HealthChecker performs health checks on backend servers
type HealthChecker struct {
	pool    *ServerPool
	config  *LoadBalancerConfig
	hcConfig *HealthCheckConfig
	client  *http.Client
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(pool *ServerPool, config *LoadBalancerConfig, hcConfig *HealthCheckConfig) *HealthChecker {
	ctx, cancel := context.WithCancel(context.Background())

	return &HealthChecker{
		pool:     pool,
		config:   config,
		hcConfig: hcConfig,
		client: &http.Client{
			Timeout: hcConfig.Timeout,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start begins health checking
func (hc *HealthChecker) Start() {
	hc.wg.Add(2)

	// Active health checks
	go hc.runActiveHealthChecks()

	// Circuit breaker recovery
	go hc.runCircuitRecovery()
}

// Stop stops health checking
func (hc *HealthChecker) Stop() {
	hc.cancel()
	hc.wg.Wait()
}

// runActiveHealthChecks performs periodic active health checks
func (hc *HealthChecker) runActiveHealthChecks() {
	defer hc.wg.Done()

	ticker := time.NewTicker(hc.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			hc.checkAllServers()
		}
	}
}

// checkAllServers checks health of all servers
func (hc *HealthChecker) checkAllServers() {
	hc.pool.mu.RLock()
	servers := make([]*Server, 0, len(hc.pool.servers))
	for _, server := range hc.pool.servers {
		servers = append(servers, server)
	}
	hc.pool.mu.RUnlock()

	// Check servers concurrently
	var wg sync.WaitGroup
	for _, server := range servers {
		wg.Add(1)
		go func(s *Server) {
			defer wg.Done()
			hc.checkServer(s)
		}(server)
	}
	wg.Wait()
}

// checkServer performs health check on a single server
func (hc *HealthChecker) checkServer(server *Server) {
	// Skip draining servers
	server.mu.RLock()
	if server.Health == ServerDraining {
		server.mu.RUnlock()
		return
	}
	server.mu.RUnlock()

	start := time.Now()
	var healthy bool

	switch hc.hcConfig.Type {
	case HealthCheckTCP:
		healthy = hc.checkTCP(server)
	case HealthCheckHTTP, HealthCheckHTTPS:
		healthy = hc.checkHTTP(server)
	case HealthCheckCustom:
		healthy = hc.checkCustom(server)
	default:
		healthy = hc.checkTCP(server)
	}

	latency := time.Since(start)

	// Update health state
	hc.pool.UpdateServerHealth(server.ID, healthy)

	// Update metrics
	if healthy {
		hc.pool.UpdateServerMetrics(server.ID, latency, true)
	}
}

// checkTCP performs TCP connection check
func (hc *HealthChecker) checkTCP(server *Server) bool {
	address := fmt.Sprintf("%s:%d", server.Address, server.Port)
	conn, err := net.DialTimeout("tcp", address, hc.hcConfig.Timeout)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// checkHTTP performs HTTP health check
func (hc *HealthChecker) checkHTTP(server *Server) bool {
	scheme := "http"
	if hc.hcConfig.Type == HealthCheckHTTPS {
		scheme = "https"
	}

	url := fmt.Sprintf("%s://%s:%d%s", scheme, server.Address, server.Port, hc.hcConfig.Endpoint)

	req, err := http.NewRequestWithContext(hc.ctx, "GET", url, nil)
	if err != nil {
		return false
	}

	// Add custom headers
	for key, value := range hc.hcConfig.Headers {
		req.Header.Set(key, value)
	}

	resp, err := hc.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == hc.hcConfig.ExpectedCode
}

// checkCustom performs custom health check
func (hc *HealthChecker) checkCustom(server *Server) bool {
	// Custom health check logic can be implemented here
	// For now, fall back to TCP check
	return hc.checkTCP(server)
}

// runCircuitRecovery attempts to recover servers with open circuits
func (hc *HealthChecker) runCircuitRecovery() {
	defer hc.wg.Done()

	ticker := time.NewTicker(hc.config.CircuitBreakerTimeout / 2)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			hc.pool.TryCircuitRecovery()
		}
	}
}

// PassiveHealthCheck performs passive health monitoring based on request results
func (hc *HealthChecker) PassiveHealthCheck(serverID string, success bool, responseTime time.Duration) {
	if success {
		// Reset failure count on successful requests
		server, err := hc.pool.GetServer(serverID)
		if err != nil {
			return
		}

		server.mu.Lock()
		if server.ConsecutiveFailures > 0 {
			server.ConsecutiveFailures--
		}
		server.mu.Unlock()

		hc.pool.UpdateServerMetrics(serverID, responseTime, true)
	} else {
		// Increment failure count
		hc.pool.UpdateServerHealth(serverID, false)
		hc.pool.UpdateServerMetrics(serverID, responseTime, false)
	}
}

// GetHealthStatus returns the health status of all servers
func (hc *HealthChecker) GetHealthStatus() map[string]ServerHealth {
	hc.pool.mu.RLock()
	defer hc.pool.mu.RUnlock()

	status := make(map[string]ServerHealth, len(hc.pool.servers))
	for id, server := range hc.pool.servers {
		server.mu.RLock()
		status[id] = server.Health
		server.mu.RUnlock()
	}

	return status
}
