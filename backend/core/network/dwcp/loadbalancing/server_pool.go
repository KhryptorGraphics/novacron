package loadbalancing

import (
	"sync"
	"sync/atomic"
	"time"
)

// ServerPool manages a pool of backend servers across multiple regions
type ServerPool struct {
	servers map[string]*Server
	regions map[string][]*Server
	mu      sync.RWMutex
	config  *LoadBalancerConfig
	stats   *LoadBalancerStats
}

// NewServerPool creates a new server pool
func NewServerPool(config *LoadBalancerConfig) *ServerPool {
	return &ServerPool{
		servers: make(map[string]*Server),
		regions: make(map[string][]*Server),
		config:  config,
		stats:   &LoadBalancerStats{},
	}
}

// AddServer adds a server to the pool
func (sp *ServerPool) AddServer(server *Server) error {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	if _, exists := sp.servers[server.ID]; exists {
		return ErrServerAlreadyExists
	}

	server.Health = ServerHealthy
	server.HealthScore = 1.0
	server.CircuitBreakerState = CircuitClosed
	server.LastHealthCheck = time.Now()

	sp.servers[server.ID] = server

	// Add to region index
	sp.regions[server.Region] = append(sp.regions[server.Region], server)

	return nil
}

// RemoveServer removes a server from the pool with graceful draining
func (sp *ServerPool) RemoveServer(serverID string) error {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	server, exists := sp.servers[serverID]
	if !exists {
		return ErrServerNotFound
	}

	// Mark as draining
	server.mu.Lock()
	server.Health = ServerDraining
	server.mu.Unlock()

	// Wait for connections to drain
	go sp.drainServer(server)

	return nil
}

// drainServer drains connections from a server before removal
func (sp *ServerPool) drainServer(server *Server) {
	timeout := time.NewTimer(sp.config.DrainTimeout)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	defer timeout.Stop()

	for {
		select {
		case <-timeout.C:
			// Force removal after timeout
			sp.forceRemoveServer(server.ID)
			return
		case <-ticker.C:
			if atomic.LoadInt32(&server.ActiveConnections) == 0 {
				sp.forceRemoveServer(server.ID)
				return
			}
		}
	}
}

// forceRemoveServer forcefully removes a server
func (sp *ServerPool) forceRemoveServer(serverID string) {
	sp.mu.Lock()
	defer sp.mu.Unlock()

	server, exists := sp.servers[serverID]
	if !exists {
		return
	}

	// Remove from region index
	regionServers := sp.regions[server.Region]
	for i, s := range regionServers {
		if s.ID == serverID {
			sp.regions[server.Region] = append(regionServers[:i], regionServers[i+1:]...)
			break
		}
	}

	delete(sp.servers, serverID)
}

// GetServer retrieves a server by ID
func (sp *ServerPool) GetServer(serverID string) (*Server, error) {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	server, exists := sp.servers[serverID]
	if !exists {
		return nil, ErrServerNotFound
	}

	return server, nil
}

// GetHealthyServers returns all healthy servers
func (sp *ServerPool) GetHealthyServers() []*Server {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	var healthy []*Server
	for _, server := range sp.servers {
		server.mu.RLock()
		if server.Health == ServerHealthy && server.CircuitBreakerState == CircuitClosed {
			healthy = append(healthy, server)
		}
		server.mu.RUnlock()
	}

	return healthy
}

// GetServersByRegion returns servers in a specific region
func (sp *ServerPool) GetServersByRegion(region string) []*Server {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	servers := sp.regions[region]
	var healthy []*Server
	for _, server := range servers {
		server.mu.RLock()
		if server.Health == ServerHealthy && server.CircuitBreakerState == CircuitClosed {
			healthy = append(healthy, server)
		}
		server.mu.RUnlock()
	}

	return healthy
}

// UpdateServerHealth updates the health state of a server
func (sp *ServerPool) UpdateServerHealth(serverID string, healthy bool) error {
	server, err := sp.GetServer(serverID)
	if err != nil {
		return err
	}

	server.mu.Lock()
	defer server.mu.Unlock()

	if healthy {
		server.ConsecutiveFailures = 0
		server.ConsecutiveSuccesses++

		if server.ConsecutiveSuccesses >= sp.config.HealthyThreshold {
			server.Health = ServerHealthy
			server.HealthScore = 1.0
			server.CircuitBreakerState = CircuitClosed
		}
	} else {
		server.ConsecutiveSuccesses = 0
		server.ConsecutiveFailures++

		if server.ConsecutiveFailures >= sp.config.UnhealthyThreshold {
			server.Health = ServerUnhealthy
			server.HealthScore = 0.0

			// Open circuit breaker
			if float64(server.TotalFailures)/float64(server.TotalRequests) > sp.config.CircuitBreakerThreshold {
				server.CircuitBreakerState = CircuitOpen
				server.CircuitOpenedAt = time.Now()
			}
		} else if server.ConsecutiveFailures > 0 {
			server.Health = ServerDegraded
			server.HealthScore = 1.0 - (float64(server.ConsecutiveFailures) / float64(sp.config.UnhealthyThreshold))
		}
	}

	server.LastHealthCheck = time.Now()

	return nil
}

// UpdateServerMetrics updates performance metrics for a server
func (sp *ServerPool) UpdateServerMetrics(serverID string, responseTime time.Duration, success bool) error {
	server, err := sp.GetServer(serverID)
	if err != nil {
		return err
	}

	server.mu.Lock()
	defer server.mu.Unlock()

	atomic.AddUint64(&server.TotalRequests, 1)
	if !success {
		atomic.AddUint64(&server.TotalFailures, 1)
	}

	// Update average response time using exponential moving average
	alpha := 0.2
	server.AvgResponseTime = time.Duration(float64(server.AvgResponseTime)*(1-alpha) + float64(responseTime)*alpha)

	// Adjust weight based on performance
	sp.adjustServerWeight(server)

	return nil
}

// adjustServerWeight adjusts server weight based on performance metrics
func (sp *ServerPool) adjustServerWeight(server *Server) {
	// Base weight on health score, response time, and resource utilization
	healthFactor := server.HealthScore

	// Penalize high response times
	responseFactor := 1.0
	if server.AvgResponseTime > 100*time.Millisecond {
		responseFactor = 100.0 / float64(server.AvgResponseTime.Milliseconds())
	}

	// Penalize high resource utilization
	resourceFactor := (1.0 - server.CPUUtilization) * (1.0 - server.MemoryUtilization)

	// Calculate final weight (scale 1-100)
	weight := int(healthFactor * responseFactor * resourceFactor * 100)
	if weight < 1 {
		weight = 1
	}
	if weight > 100 {
		weight = 100
	}

	server.Weight = weight
}

// IncrementConnections increments the active connection count
func (sp *ServerPool) IncrementConnections(serverID string) error {
	server, err := sp.GetServer(serverID)
	if err != nil {
		return err
	}

	atomic.AddInt32(&server.ActiveConnections, 1)
	return nil
}

// DecrementConnections decrements the active connection count
func (sp *ServerPool) DecrementConnections(serverID string) error {
	server, err := sp.GetServer(serverID)
	if err != nil {
		return err
	}

	atomic.AddInt32(&server.ActiveConnections, -1)
	return nil
}

// GetStats returns the current statistics
func (sp *ServerPool) GetStats() *LoadBalancerStats {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	sp.stats.mu.Lock()
	defer sp.stats.mu.Unlock()

	// Update server counts
	sp.stats.HealthyServers = 0
	sp.stats.DegradedServers = 0
	sp.stats.UnhealthyServers = 0
	sp.stats.DrainingServers = 0
	sp.stats.TotalConnections = 0

	for _, server := range sp.servers {
		server.mu.RLock()
		switch server.Health {
		case ServerHealthy:
			sp.stats.HealthyServers++
		case ServerDegraded:
			sp.stats.DegradedServers++
		case ServerUnhealthy:
			sp.stats.UnhealthyServers++
		case ServerDraining:
			sp.stats.DrainingServers++
		}
		sp.stats.TotalConnections += atomic.LoadInt32(&server.ActiveConnections)
		server.mu.RUnlock()
	}

	return sp.stats
}

// TryCircuitRecovery attempts to recover servers with open circuits
func (sp *ServerPool) TryCircuitRecovery() {
	sp.mu.RLock()
	defer sp.mu.RUnlock()

	now := time.Now()
	for _, server := range sp.servers {
		server.mu.Lock()
		if server.CircuitBreakerState == CircuitOpen {
			if now.Sub(server.CircuitOpenedAt) > sp.config.CircuitBreakerTimeout {
				// Move to half-open to test recovery
				server.CircuitBreakerState = CircuitHalfOpen
				server.ConsecutiveFailures = 0
				server.ConsecutiveSuccesses = 0
			}
		}
		server.mu.Unlock()
	}
}
