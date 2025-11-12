package loadbalancing

import (
	"sync/atomic"
	"testing"
	"time"
)

func createTestServer(id, region string) *Server {
	return &Server{
		ID:        id,
		Address:   "10.0.0." + id,
		Port:      8080,
		Region:    region,
		Latitude:  39.0438,
		Longitude: -77.4874,
		Weight:    100,
	}
}

func TestServerPoolAddServer(t *testing.T) {
	pool := NewServerPool(DefaultConfig())

	server := createTestServer("1", "us-east-1")
	err := pool.AddServer(server)
	if err != nil {
		t.Fatalf("Failed to add server: %v", err)
	}

	// Check server exists
	retrieved, err := pool.GetServer(server.ID)
	if err != nil {
		t.Fatalf("Failed to get server: %v", err)
	}

	if retrieved.ID != server.ID {
		t.Errorf("Expected server ID %s, got %s", server.ID, retrieved.ID)
	}

	// Try adding duplicate
	err = pool.AddServer(server)
	if err != ErrServerAlreadyExists {
		t.Errorf("Expected ErrServerAlreadyExists, got %v", err)
	}
}

func TestServerPoolGetHealthyServers(t *testing.T) {
	pool := NewServerPool(DefaultConfig())

	// Add healthy servers
	for i := 1; i <= 5; i++ {
		server := createTestServer(string(rune('0'+i)), "us-east-1")
		pool.AddServer(server)
	}

	// Mark one as unhealthy
	server, _ := pool.GetServer("1")
	server.mu.Lock()
	server.Health = ServerUnhealthy
	server.mu.Unlock()

	healthy := pool.GetHealthyServers()
	if len(healthy) != 4 {
		t.Errorf("Expected 4 healthy servers, got %d", len(healthy))
	}
}

func TestServerPoolUpdateServerHealth(t *testing.T) {
	config := DefaultConfig()
	config.UnhealthyThreshold = 3
	config.HealthyThreshold = 2

	pool := NewServerPool(config)
	server := createTestServer("1", "us-east-1")
	pool.AddServer(server)

	// Mark as failed 3 times
	for i := 0; i < 3; i++ {
		pool.UpdateServerHealth(server.ID, false)
	}

	retrieved, _ := pool.GetServer(server.ID)
	retrieved.mu.RLock()
	health := retrieved.Health
	retrieved.mu.RUnlock()

	if health != ServerUnhealthy {
		t.Errorf("Expected server to be unhealthy, got %s", health)
	}

	// Mark as healthy 2 times
	for i := 0; i < 2; i++ {
		pool.UpdateServerHealth(server.ID, true)
	}

	retrieved, _ = pool.GetServer(server.ID)
	retrieved.mu.RLock()
	health = retrieved.Health
	retrieved.mu.RUnlock()

	if health != ServerHealthy {
		t.Errorf("Expected server to be healthy, got %s", health)
	}
}

func TestServerPoolConnectionTracking(t *testing.T) {
	pool := NewServerPool(DefaultConfig())
	server := createTestServer("1", "us-east-1")
	pool.AddServer(server)

	// Increment connections
	for i := 0; i < 10; i++ {
		pool.IncrementConnections(server.ID)
	}

	retrieved, _ := pool.GetServer(server.ID)
	conns := atomic.LoadInt32(&retrieved.ActiveConnections)
	if conns != 10 {
		t.Errorf("Expected 10 connections, got %d", conns)
	}

	// Decrement connections
	for i := 0; i < 5; i++ {
		pool.DecrementConnections(server.ID)
	}

	conns = atomic.LoadInt32(&retrieved.ActiveConnections)
	if conns != 5 {
		t.Errorf("Expected 5 connections, got %d", conns)
	}
}

func TestServerPoolGetServersByRegion(t *testing.T) {
	pool := NewServerPool(DefaultConfig())

	// Add servers in different regions
	pool.AddServer(createTestServer("1", "us-east-1"))
	pool.AddServer(createTestServer("2", "us-east-1"))
	pool.AddServer(createTestServer("3", "us-west-1"))

	usEast := pool.GetServersByRegion("us-east-1")
	if len(usEast) != 2 {
		t.Errorf("Expected 2 servers in us-east-1, got %d", len(usEast))
	}

	usWest := pool.GetServersByRegion("us-west-1")
	if len(usWest) != 1 {
		t.Errorf("Expected 1 server in us-west-1, got %d", len(usWest))
	}
}

func TestServerPoolRemoveServer(t *testing.T) {
	pool := NewServerPool(DefaultConfig())
	server := createTestServer("1", "us-east-1")
	pool.AddServer(server)

	// Remove server
	err := pool.RemoveServer(server.ID)
	if err != nil {
		t.Fatalf("Failed to remove server: %v", err)
	}

	// Wait for drain timeout
	time.Sleep(200 * time.Millisecond)

	// Server should be marked as draining or removed
	retrieved, err := pool.GetServer(server.ID)
	if err == nil {
		retrieved.mu.RLock()
		if retrieved.Health != ServerDraining {
			t.Error("Server should be draining or removed")
		}
		retrieved.mu.RUnlock()
	}
}
