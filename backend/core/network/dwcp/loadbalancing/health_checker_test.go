package loadbalancing

import (
	"testing"
	"time"
)

func TestHealthCheckerCreation(t *testing.T) {
	pool := NewServerPool(DefaultConfig())
	hc := NewHealthChecker(pool, DefaultConfig(), DefaultHealthCheckConfig())

	if hc == nil {
		t.Fatal("Expected non-nil health checker")
	}
}

func TestHealthCheckerStartStop(t *testing.T) {
	pool := NewServerPool(DefaultConfig())
	config := DefaultConfig()
	config.HealthCheckInterval = 50 * time.Millisecond

	hc := NewHealthChecker(pool, config, DefaultHealthCheckConfig())

	hc.Start()
	time.Sleep(100 * time.Millisecond)
	hc.Stop()
}

func TestPassiveHealthCheck(t *testing.T) {
	pool := NewServerPool(DefaultConfig())
	server := createTestServer("1", "us-east-1")
	pool.AddServer(server)

	hc := NewHealthChecker(pool, DefaultConfig(), DefaultHealthCheckConfig())

	// Record successful responses
	for i := 0; i < 5; i++ {
		hc.PassiveHealthCheck(server.ID, true, 10*time.Millisecond)
	}

	// Server should remain healthy
	s, _ := pool.GetServer(server.ID)
	s.mu.RLock()
	health := s.Health
	s.mu.RUnlock()

	if health != ServerHealthy {
		t.Errorf("Expected server to be healthy, got %s", health)
	}

	// Record failures
	for i := 0; i < 3; i++ {
		hc.PassiveHealthCheck(server.ID, false, 10*time.Millisecond)
	}

	// Server should be marked unhealthy
	s, _ = pool.GetServer(server.ID)
	s.mu.RLock()
	health = s.Health
	s.mu.RUnlock()

	if health != ServerUnhealthy {
		t.Errorf("Expected server to be unhealthy, got %s", health)
	}
}

func TestGetHealthStatus(t *testing.T) {
	pool := NewServerPool(DefaultConfig())
	hc := NewHealthChecker(pool, DefaultConfig(), DefaultHealthCheckConfig())

	// Add servers
	server1 := createTestServer("1", "us-east-1")
	server2 := createTestServer("2", "us-east-1")
	pool.AddServer(server1)
	pool.AddServer(server2)

	// Mark server2 as unhealthy
	s2, _ := pool.GetServer(server2.ID)
	s2.mu.Lock()
	s2.Health = ServerUnhealthy
	s2.mu.Unlock()

	// Get status
	status := hc.GetHealthStatus()

	if len(status) != 2 {
		t.Errorf("Expected 2 servers, got %d", len(status))
	}

	if status[server1.ID] != ServerHealthy {
		t.Errorf("Expected server1 to be healthy")
	}

	if status[server2.ID] != ServerUnhealthy {
		t.Errorf("Expected server2 to be unhealthy")
	}
}

func TestCircuitBreakerRecovery(t *testing.T) {
	config := DefaultConfig()
	config.CircuitBreakerTimeout = 100 * time.Millisecond

	pool := NewServerPool(config)
	server := createTestServer("1", "us-east-1")
	pool.AddServer(server)

	// Open circuit breaker
	s, _ := pool.GetServer(server.ID)
	s.mu.Lock()
	s.CircuitBreakerState = CircuitOpen
	s.CircuitOpenedAt = time.Now()
	s.mu.Unlock()

	// Wait for recovery timeout
	time.Sleep(150 * time.Millisecond)

	// Trigger recovery check
	pool.TryCircuitRecovery()

	// Circuit should be half-open
	s, _ = pool.GetServer(server.ID)
	s.mu.RLock()
	state := s.CircuitBreakerState
	s.mu.RUnlock()

	if state != CircuitHalfOpen {
		t.Errorf("Expected circuit to be half-open, got %s", state)
	}
}
