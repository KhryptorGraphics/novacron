package loadbalancing

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// GeoLoadBalancer implements geographic-aware load balancing
type GeoLoadBalancer struct {
	config          *LoadBalancerConfig
	pool            *ServerPool
	geoRouter       *GeoRouter
	sessionManager  *SessionAffinityManager
	healthChecker   *HealthChecker
	metrics         *MetricsCollector
	rrIndex         uint32 // Round-robin index
	ctx             context.Context
	cancel          context.CancelFunc
	mu              sync.RWMutex
}

// NewGeoLoadBalancer creates a new geographic load balancer
func NewGeoLoadBalancer(config *LoadBalancerConfig) (*GeoLoadBalancer, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(context.Background())

	pool := NewServerPool(config)
	geoRouter, err := NewGeoRouter(config)
	if err != nil {
		cancel()
		return nil, err
	}

	sessionManager := NewSessionAffinityManager(config)
	healthChecker := NewHealthChecker(pool, config, DefaultHealthCheckConfig())
	metrics := NewMetricsCollector(config)

	lb := &GeoLoadBalancer{
		config:         config,
		pool:           pool,
		geoRouter:      geoRouter,
		sessionManager: sessionManager,
		healthChecker:  healthChecker,
		metrics:        metrics,
		rrIndex:        0,
		ctx:            ctx,
		cancel:         cancel,
	}

	return lb, nil
}

// Start starts the load balancer
func (lb *GeoLoadBalancer) Start() error {
	lb.healthChecker.Start()
	lb.metrics.Start()
	return nil
}

// Stop stops the load balancer
func (lb *GeoLoadBalancer) Stop() error {
	lb.cancel()
	lb.healthChecker.Stop()
	lb.metrics.Stop()
	return nil
}

// AddServer adds a backend server to the pool
func (lb *GeoLoadBalancer) AddServer(server *Server) error {
	if err := lb.pool.AddServer(server); err != nil {
		return err
	}

	// Add to session affinity ring
	if lb.config.EnableSessionAffinity {
		lb.sessionManager.AddServerToRing(server.ID)
	}

	return nil
}

// RemoveServer removes a backend server from the pool
func (lb *GeoLoadBalancer) RemoveServer(serverID string) error {
	if err := lb.pool.RemoveServer(serverID); err != nil {
		return err
	}

	// Remove from session affinity ring
	if lb.config.EnableSessionAffinity {
		lb.sessionManager.RemoveServerFromRing(serverID)
	}

	return nil
}

// SelectServer selects the best server for a request
func (lb *GeoLoadBalancer) SelectServer(clientIP string, sessionID string) (*RoutingDecision, error) {
	start := time.Now()

	var decision *RoutingDecision
	var err error

	// Check for session affinity first
	if lb.config.EnableSessionAffinity && sessionID != "" {
		decision, err = lb.selectWithAffinity(sessionID, clientIP)
		if err == nil {
			decision.Latency = time.Since(start)
			lb.metrics.RecordRoutingDecision(decision)
			return decision, nil
		}
	}

	// Select based on algorithm
	switch lb.config.Algorithm {
	case AlgorithmGeoProximity:
		decision, err = lb.selectByGeoProximity(clientIP)
	case AlgorithmLeastLatency:
		decision, err = lb.selectByLeastLatency(clientIP)
	case AlgorithmLeastConnections:
		decision, err = lb.selectByLeastConnections()
	case AlgorithmWeightedRoundRobin:
		decision, err = lb.selectByWeightedRoundRobin()
	case AlgorithmIPHash:
		decision, err = lb.selectByIPHash(clientIP)
	case AlgorithmRoundRobin:
		fallthrough
	default:
		decision, err = lb.selectByRoundRobin()
	}

	if err != nil {
		lb.metrics.RecordFailure()
		return nil, err
	}

	decision.Latency = time.Since(start)
	lb.metrics.RecordRoutingDecision(decision)

	// Create session if enabled
	if lb.config.EnableSessionAffinity && sessionID != "" {
		lb.sessionManager.CreateSession(sessionID, decision.Server.ID)
	}

	return decision, nil
}

// selectWithAffinity selects server based on session affinity
func (lb *GeoLoadBalancer) selectWithAffinity(sessionID, clientIP string) (*RoutingDecision, error) {
	session, err := lb.sessionManager.GetSession(sessionID)
	if err != nil {
		return nil, err
	}

	// Check if affinity server is healthy
	server, err := lb.pool.GetServer(session.ServerID)
	if err != nil {
		return nil, err
	}

	server.mu.RLock()
	healthy := server.Health == ServerHealthy && server.CircuitBreakerState == CircuitClosed
	server.mu.RUnlock()

	if !healthy {
		// Server is down, need to failover
		return lb.handleAffinityFailover(sessionID, clientIP)
	}

	// Update session
	lb.sessionManager.UpdateSession(sessionID)

	return &RoutingDecision{
		Server:     server,
		Algorithm:  "session-affinity",
		Timestamp:  time.Now(),
		ReasonCode: "affinity-match",
		IsFailover: false,
	}, nil
}

// handleAffinityFailover handles failover when affinity server is down
func (lb *GeoLoadBalancer) handleAffinityFailover(sessionID, clientIP string) (*RoutingDecision, error) {
	start := time.Now()

	// Select new server using geo-proximity
	decision, err := lb.selectByGeoProximity(clientIP)
	if err != nil {
		return nil, err
	}

	// Migrate session to new server
	lb.sessionManager.MigrateSession(sessionID, decision.Server.ID)

	decision.IsFailover = true
	decision.ReasonCode = "affinity-failover"

	failoverTime := time.Since(start)
	lb.metrics.RecordFailover(failoverTime)

	return decision, nil
}

// selectByGeoProximity selects server based on geographic proximity
func (lb *GeoLoadBalancer) selectByGeoProximity(clientIP string) (*RoutingDecision, error) {
	// Get client location
	clientLoc, err := lb.geoRouter.GetClientLocation(clientIP)
	if err != nil {
		// Fallback to round-robin if geo lookup fails
		return lb.selectByRoundRobin()
	}

	// Get healthy servers
	servers := lb.pool.GetHealthyServers()
	if len(servers) == 0 {
		return nil, ErrNoHealthyServers
	}

	// Find nearest server
	nearestServer := lb.geoRouter.FindNearestServer(clientLoc, servers)
	if nearestServer == nil {
		return nil, ErrNoHealthyServers
	}

	return &RoutingDecision{
		Server:     nearestServer,
		Algorithm:  AlgorithmGeoProximity,
		Timestamp:  time.Now(),
		ReasonCode: fmt.Sprintf("nearest-to-%s", clientLoc.Region),
		IsFailover: false,
	}, nil
}

// selectByLeastLatency selects server with lowest response time
func (lb *GeoLoadBalancer) selectByLeastLatency(clientIP string) (*RoutingDecision, error) {
	servers := lb.pool.GetHealthyServers()
	if len(servers) == 0 {
		return nil, ErrNoHealthyServers
	}

	var bestServer *Server
	minLatency := time.Duration(1<<63 - 1) // Max duration

	for _, server := range servers {
		server.mu.RLock()
		latency := server.AvgResponseTime
		server.mu.RUnlock()

		if latency < minLatency {
			minLatency = latency
			bestServer = server
		}
	}

	if bestServer == nil {
		return lb.selectByRoundRobin()
	}

	return &RoutingDecision{
		Server:     bestServer,
		Algorithm:  AlgorithmLeastLatency,
		Timestamp:  time.Now(),
		ReasonCode: fmt.Sprintf("latency-%dms", minLatency.Milliseconds()),
		IsFailover: false,
	}, nil
}

// selectByLeastConnections selects server with fewest active connections
func (lb *GeoLoadBalancer) selectByLeastConnections() (*RoutingDecision, error) {
	servers := lb.pool.GetHealthyServers()
	if len(servers) == 0 {
		return nil, ErrNoHealthyServers
	}

	var bestServer *Server
	minConns := int32(1<<31 - 1) // Max int32

	for _, server := range servers {
		conns := atomic.LoadInt32(&server.ActiveConnections)
		if conns < minConns {
			minConns = conns
			bestServer = server
		}
	}

	if bestServer == nil {
		return lb.selectByRoundRobin()
	}

	return &RoutingDecision{
		Server:     bestServer,
		Algorithm:  AlgorithmLeastConnections,
		Timestamp:  time.Now(),
		ReasonCode: fmt.Sprintf("connections-%d", minConns),
		IsFailover: false,
	}, nil
}

// selectByWeightedRoundRobin selects server using weighted round-robin
func (lb *GeoLoadBalancer) selectByWeightedRoundRobin() (*RoutingDecision, error) {
	servers := lb.pool.GetHealthyServers()
	if len(servers) == 0 {
		return nil, ErrNoHealthyServers
	}

	// Calculate total weight
	totalWeight := 0
	for _, server := range servers {
		server.mu.RLock()
		totalWeight += server.Weight
		server.mu.RUnlock()
	}

	if totalWeight == 0 {
		return lb.selectByRoundRobin()
	}

	// Select server based on weight
	index := atomic.AddUint32(&lb.rrIndex, 1) % uint32(totalWeight)
	currentWeight := uint32(0)

	for _, server := range servers {
		server.mu.RLock()
		weight := uint32(server.Weight)
		server.mu.RUnlock()

		currentWeight += weight
		if index < currentWeight {
			return &RoutingDecision{
				Server:     server,
				Algorithm:  AlgorithmWeightedRoundRobin,
				Timestamp:  time.Now(),
				ReasonCode: fmt.Sprintf("weight-%d", weight),
				IsFailover: false,
			}, nil
		}
	}

	// Fallback
	return lb.selectByRoundRobin()
}

// selectByRoundRobin selects server using simple round-robin
func (lb *GeoLoadBalancer) selectByRoundRobin() (*RoutingDecision, error) {
	servers := lb.pool.GetHealthyServers()
	if len(servers) == 0 {
		return nil, ErrNoHealthyServers
	}

	index := atomic.AddUint32(&lb.rrIndex, 1) % uint32(len(servers))
	server := servers[index]

	return &RoutingDecision{
		Server:     server,
		Algorithm:  AlgorithmRoundRobin,
		Timestamp:  time.Now(),
		ReasonCode: "round-robin",
		IsFailover: false,
	}, nil
}

// selectByIPHash selects server using IP hash for consistent routing
func (lb *GeoLoadBalancer) selectByIPHash(clientIP string) (*RoutingDecision, error) {
	serverID, err := lb.sessionManager.GetServerByIP(clientIP)
	if err != nil {
		return lb.selectByRoundRobin()
	}

	server, err := lb.pool.GetServer(serverID)
	if err != nil {
		return lb.selectByRoundRobin()
	}

	server.mu.RLock()
	healthy := server.Health == ServerHealthy && server.CircuitBreakerState == CircuitClosed
	server.mu.RUnlock()

	if !healthy {
		// Fallback to geo-proximity
		return lb.selectByGeoProximity(clientIP)
	}

	return &RoutingDecision{
		Server:     server,
		Algorithm:  AlgorithmIPHash,
		Timestamp:  time.Now(),
		ReasonCode: "ip-hash-match",
		IsFailover: false,
	}, nil
}

// RecordResponse records the response from a server
func (lb *GeoLoadBalancer) RecordResponse(serverID string, responseTime time.Duration, success bool) error {
	// Update passive health check
	lb.healthChecker.PassiveHealthCheck(serverID, success, responseTime)

	// Record metrics
	lb.metrics.RecordResponse(responseTime, success)

	return nil
}

// GetStats returns current load balancer statistics
func (lb *GeoLoadBalancer) GetStats() *LoadBalancerStats {
	return lb.pool.GetStats()
}

// GetHealthStatus returns health status of all servers
func (lb *GeoLoadBalancer) GetHealthStatus() map[string]ServerHealth {
	return lb.healthChecker.GetHealthStatus()
}
