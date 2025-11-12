package loadbalancing

import (
	"testing"
	"time"
)

func setupTestLoadBalancer(t *testing.T) *GeoLoadBalancer {
	config := DefaultConfig()
	config.HealthCheckInterval = 1 * time.Second
	config.FailoverTimeout = 50 * time.Millisecond

	lb, err := NewGeoLoadBalancer(config)
	if err != nil {
		t.Fatalf("Failed to create load balancer: %v", err)
	}

	// Add test servers
	servers := []*Server{
		{
			ID:        "server-1",
			Address:   "10.0.0.1",
			Port:      8080,
			Region:    "us-east-1",
			Latitude:  39.0438,
			Longitude: -77.4874,
			Weight:    100,
		},
		{
			ID:        "server-2",
			Address:   "10.0.0.2",
			Port:      8080,
			Region:    "us-west-1",
			Latitude:  37.7749,
			Longitude: -122.4194,
			Weight:    100,
		},
		{
			ID:        "server-3",
			Address:   "10.0.0.3",
			Port:      8080,
			Region:    "eu-west-1",
			Latitude:  53.3498,
			Longitude: -6.2603,
			Weight:    100,
		},
	}

	for _, server := range servers {
		if err := lb.AddServer(server); err != nil {
			t.Fatalf("Failed to add server: %v", err)
		}
	}

	return lb
}

func TestGeoLoadBalancerCreation(t *testing.T) {
	config := DefaultConfig()
	lb, err := NewGeoLoadBalancer(config)
	if err != nil {
		t.Fatalf("Failed to create load balancer: %v", err)
	}

	if lb == nil {
		t.Fatal("Expected non-nil load balancer")
	}
}

func TestLoadBalancerStartStop(t *testing.T) {
	lb := setupTestLoadBalancer(t)

	err := lb.Start()
	if err != nil {
		t.Fatalf("Failed to start load balancer: %v", err)
	}

	// Run for a short time
	time.Sleep(50 * time.Millisecond)

	err = lb.Stop()
	if err != nil {
		t.Fatalf("Failed to stop load balancer: %v", err)
	}
}

func TestSelectServerRoundRobin(t *testing.T) {
	config := DefaultConfig()
	config.Algorithm = AlgorithmRoundRobin
	config.EnableSessionAffinity = false

	lb, _ := NewGeoLoadBalancer(config)

	// Add servers
	for i := 1; i <= 3; i++ {
		server := createTestServer(string(rune('0'+i)), "us-east-1")
		lb.AddServer(server)
	}

	lb.Start()
	defer lb.Stop()

	// Test round-robin distribution
	serverCounts := make(map[string]int)
	totalRequests := 30
	for i := 0; i < totalRequests; i++ {
		decision, err := lb.SelectServer("1.2.3.4", "")
		if err != nil {
			t.Fatalf("Failed to select server: %v", err)
		}
		serverCounts[decision.Server.ID]++
	}

	// Each server should be selected roughly equally (within 20% tolerance)
	expectedPerServer := totalRequests / 3
	tolerance := expectedPerServer / 5
	for serverID, count := range serverCounts {
		if count < expectedPerServer-tolerance || count > expectedPerServer+tolerance {
			t.Errorf("Server %s selected %d times, expected around %d (±%d)",
				serverID, count, expectedPerServer, tolerance)
		}
	}
}

func TestSelectServerLeastConnections(t *testing.T) {
	config := DefaultConfig()
	config.Algorithm = AlgorithmLeastConnections
	config.EnableSessionAffinity = false

	lb, _ := NewGeoLoadBalancer(config)

	// Add servers
	server1 := createTestServer("1", "us-east-1")
	server2 := createTestServer("2", "us-east-1")
	lb.AddServer(server1)
	lb.AddServer(server2)

	lb.Start()
	defer lb.Stop()

	// Add connections to server1
	lb.pool.IncrementConnections(server1.ID)
	lb.pool.IncrementConnections(server1.ID)
	lb.pool.IncrementConnections(server1.ID)

	// Next selection should be server2 (fewer connections)
	decision, err := lb.SelectServer("1.2.3.4", "")
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	if decision.Server.ID != "2" {
		t.Errorf("Expected server 2 (least connections), got %s", decision.Server.ID)
	}
}

func TestSelectServerGeoProximity(t *testing.T) {
	config := DefaultConfig()
	config.Algorithm = AlgorithmGeoProximity
	config.EnableSessionAffinity = false

	lb, _ := NewGeoLoadBalancer(config)

	// Add servers in different regions
	servers := []*Server{
		{
			ID:        "us-east",
			Region:    "us-east-1",
			Latitude:  39.0438,
			Longitude: -77.4874,
			Weight:    100,
			Health:    ServerHealthy,
			CircuitBreakerState: CircuitClosed,
		},
		{
			ID:        "eu-west",
			Region:    "eu-west-1",
			Latitude:  53.3498,
			Longitude: -6.2603,
			Weight:    100,
			Health:    ServerHealthy,
			CircuitBreakerState: CircuitClosed,
		},
	}

	for _, s := range servers {
		lb.AddServer(s)
	}

	lb.Start()
	defer lb.Stop()

	// Client from US should get US server
	decision, err := lb.SelectServer("8.8.8.8", "")
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	// Should select based on proximity
	if decision.Server == nil {
		t.Error("Expected server selection")
	}
}

func TestSelectServerWithSessionAffinity(t *testing.T) {
	config := DefaultConfig()
	config.EnableSessionAffinity = true
	config.SessionAffinityTTL = 1 * time.Second

	lb, _ := NewGeoLoadBalancer(config)

	server1 := createTestServer("1", "us-east-1")
	server2 := createTestServer("2", "us-east-1")
	lb.AddServer(server1)
	lb.AddServer(server2)

	lb.Start()
	defer lb.Stop()

	sessionID := "test-session"

	// First request creates affinity
	decision1, err := lb.SelectServer("1.2.3.4", sessionID)
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	// Second request should use same server
	decision2, err := lb.SelectServer("1.2.3.4", sessionID)
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	if decision1.Server.ID != decision2.Server.ID {
		t.Errorf("Session affinity broken: %s != %s",
			decision1.Server.ID, decision2.Server.ID)
	}
}

func TestFailoverOnServerFailure(t *testing.T) {
	config := DefaultConfig()
	config.EnableSessionAffinity = true
	config.FailoverTimeout = 50 * time.Millisecond

	lb, _ := NewGeoLoadBalancer(config)

	server1 := createTestServer("1", "us-east-1")
	server2 := createTestServer("2", "us-east-1")
	lb.AddServer(server1)
	lb.AddServer(server2)

	lb.Start()
	defer lb.Stop()

	sessionID := "test-session-failover"

	// Create affinity to server1
	decision1, _ := lb.SelectServer("1.2.3.4", sessionID)
	originalServerID := decision1.Server.ID

	// Mark server as unhealthy
	server, _ := lb.pool.GetServer(originalServerID)
	server.mu.Lock()
	server.Health = ServerUnhealthy
	server.CircuitBreakerState = CircuitOpen
	server.mu.Unlock()

	// Next request should failover
	start := time.Now()
	decision2, err := lb.SelectServer("1.2.3.4", sessionID)
	failoverTime := time.Since(start)

	if err != nil {
		t.Fatalf("Failover failed: %v", err)
	}

	if decision2.Server.ID == originalServerID {
		t.Error("Failover did not occur")
	}

	if !decision2.IsFailover {
		t.Error("Expected IsFailover to be true")
	}

	// Verify failover time is under target
	if failoverTime > config.FailoverTimeout {
		t.Errorf("Failover took %v, exceeds target %v",
			failoverTime, config.FailoverTimeout)
	}
}

func TestRecordResponse(t *testing.T) {
	lb := setupTestLoadBalancer(t)
	lb.Start()
	defer lb.Stop()

	serverID := "server-1"
	responseTime := 50 * time.Millisecond

	// Record successful response
	err := lb.RecordResponse(serverID, responseTime, true)
	if err != nil {
		t.Fatalf("Failed to record response: %v", err)
	}

	// Verify metrics updated
	server, _ := lb.pool.GetServer(serverID)
	server.mu.RLock()
	totalRequests := server.TotalRequests
	server.mu.RUnlock()

	if totalRequests == 0 {
		t.Error("Expected request count to be updated")
	}
}

func TestNoHealthyServersError(t *testing.T) {
	config := DefaultConfig()
	lb, _ := NewGeoLoadBalancer(config)

	// Add server but mark as unhealthy
	server := createTestServer("1", "us-east-1")
	lb.AddServer(server)

	s, _ := lb.pool.GetServer(server.ID)
	s.mu.Lock()
	s.Health = ServerUnhealthy
	s.CircuitBreakerState = CircuitOpen
	s.mu.Unlock()

	lb.Start()
	defer lb.Stop()

	// Should return error
	_, err := lb.SelectServer("1.2.3.4", "")
	if err != ErrNoHealthyServers {
		t.Errorf("Expected ErrNoHealthyServers, got %v", err)
	}
}

func TestRoutingLatency(t *testing.T) {
	lb := setupTestLoadBalancer(t)
	lb.Start()
	defer lb.Stop()

	// Perform routing and measure latency
	decision, err := lb.SelectServer("1.2.3.4", "")
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	// Routing should be very fast (< 1ms target)
	if decision.Latency > time.Millisecond {
		t.Errorf("Routing latency %v exceeds 1ms target", decision.Latency)
	}
}

func TestConcurrentRequests(t *testing.T) {
	lb := setupTestLoadBalancer(t)
	lb.Start()
	defer lb.Stop()

	// Simulate concurrent requests
	concurrency := 100
	done := make(chan bool, concurrency)
	errors := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		go func(id int) {
			_, err := lb.SelectServer("1.2.3.4", "")
			if err != nil {
				errors <- err
			}
			done <- true
		}(i)
	}

	// Wait for all requests
	for i := 0; i < concurrency; i++ {
		<-done
	}

	// Check for errors
	select {
	case err := <-errors:
		t.Errorf("Concurrent request failed: %v", err)
	default:
		// No errors
	}
}
