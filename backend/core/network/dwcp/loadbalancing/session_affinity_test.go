package loadbalancing

import (
	"testing"
	"time"
)

func TestSessionAffinityManagerCreation(t *testing.T) {
	config := DefaultConfig()
	sam := NewSessionAffinityManager(config)

	if sam == nil {
		t.Fatal("Expected non-nil session affinity manager")
	}
}

func TestCreateAndGetSession(t *testing.T) {
	config := DefaultConfig()
	sam := NewSessionAffinityManager(config)

	sessionID := "test-session-123"
	serverID := "server-1"

	// Create session
	session := sam.CreateSession(sessionID, serverID)
	if session == nil {
		t.Fatal("Expected non-nil session")
	}

	if session.SessionID != sessionID {
		t.Errorf("Expected session ID %s, got %s", sessionID, session.SessionID)
	}

	// Retrieve session
	retrieved, err := sam.GetSession(sessionID)
	if err != nil {
		t.Fatalf("Failed to get session: %v", err)
	}

	if retrieved.ServerID != serverID {
		t.Errorf("Expected server ID %s, got %s", serverID, retrieved.ServerID)
	}
}

func TestSessionExpiration(t *testing.T) {
	config := DefaultConfig()
	config.SessionAffinityTTL = 100 * time.Millisecond
	sam := NewSessionAffinityManager(config)

	sessionID := "test-session-expire"
	sam.CreateSession(sessionID, "server-1")

	// Wait for expiration
	time.Sleep(150 * time.Millisecond)

	// Try to get expired session
	_, err := sam.GetSession(sessionID)
	if err != ErrSessionNotFound {
		t.Errorf("Expected ErrSessionNotFound, got %v", err)
	}
}

func TestUpdateSession(t *testing.T) {
	config := DefaultConfig()
	sam := NewSessionAffinityManager(config)

	sessionID := "test-session-update"
	session := sam.CreateSession(sessionID, "server-1")

	initialExpiry := session.ExpiresAt
	initialCount := session.RequestCount

	// Wait a bit
	time.Sleep(10 * time.Millisecond)

	// Update session
	err := sam.UpdateSession(sessionID)
	if err != nil {
		t.Fatalf("Failed to update session: %v", err)
	}

	// Retrieve updated session
	updated, _ := sam.GetSession(sessionID)

	// Expiry should be extended
	if !updated.ExpiresAt.After(initialExpiry) {
		t.Error("Expected expiry to be extended")
	}

	// Request count should be incremented
	if updated.RequestCount != initialCount+1 {
		t.Errorf("Expected request count %d, got %d", initialCount+1, updated.RequestCount)
	}
}

func TestMigrateSession(t *testing.T) {
	config := DefaultConfig()
	sam := NewSessionAffinityManager(config)

	sessionID := "test-session-migrate"
	sam.CreateSession(sessionID, "server-1")

	// Migrate to new server
	newServerID := "server-2"
	err := sam.MigrateSession(sessionID, newServerID)
	if err != nil {
		t.Fatalf("Failed to migrate session: %v", err)
	}

	// Verify migration
	session, _ := sam.GetSession(sessionID)
	if session.ServerID != newServerID {
		t.Errorf("Expected server ID %s, got %s", newServerID, session.ServerID)
	}
}

func TestDeleteSession(t *testing.T) {
	config := DefaultConfig()
	sam := NewSessionAffinityManager(config)

	sessionID := "test-session-delete"
	sam.CreateSession(sessionID, "server-1")

	// Delete session
	sam.DeleteSession(sessionID)

	// Verify deletion
	_, err := sam.GetSession(sessionID)
	if err != ErrSessionNotFound {
		t.Errorf("Expected ErrSessionNotFound, got %v", err)
	}
}

func TestConsistentHashRing(t *testing.T) {
	ring := NewConsistentHashRing(150)

	// Add servers
	ring.AddServer("server-1")
	ring.AddServer("server-2")
	ring.AddServer("server-3")

	// Test consistent routing
	key := "user-12345"
	server1, _ := ring.GetServer(key)

	// Same key should always map to same server
	server2, _ := ring.GetServer(key)
	if server1 != server2 {
		t.Errorf("Inconsistent routing: %s != %s", server1, server2)
	}

	// Different keys should distribute across servers
	servers := make(map[string]int)
	for i := 0; i < 1000; i++ {
		key := string(rune(i))
		server, _ := ring.GetServer(key)
		servers[server]++
	}

	// All servers should receive some traffic (roughly balanced)
	if len(servers) != 3 {
		t.Errorf("Expected 3 servers to receive traffic, got %d", len(servers))
	}

	// Each server should receive some traffic (basic distribution check)
	// With consistent hashing, distribution may not be perfectly even
	expectedPerServer := 1000 / 3  // ~333
	minExpected := expectedPerServer / 3  // At least 33% of expected

	for server, count := range servers {
		if count < minExpected {
			t.Errorf("Server %s received %d requests (expected at least %d)",
				server, count, minExpected)
		}
	}
}

func TestConsistentHashRingRemoval(t *testing.T) {
	ring := NewConsistentHashRing(150)

	ring.AddServer("server-1")
	ring.AddServer("server-2")
	ring.AddServer("server-3")

	// Track which keys map to server-2
	keysToServer2 := make([]string, 0)
	for i := 0; i < 100; i++ {
		key := string(rune(i))
		server, _ := ring.GetServer(key)
		if server == "server-2" {
			keysToServer2 = append(keysToServer2, key)
		}
	}

	// Remove server-2
	ring.RemoveServer("server-2")

	// Keys that mapped to server-2 should now map to other servers
	for _, key := range keysToServer2 {
		server, _ := ring.GetServer(key)
		if server == "server-2" {
			t.Errorf("Key %s still maps to removed server", key)
		}
	}
}

func TestGenerateSessionID(t *testing.T) {
	sessionID1 := GenerateSessionID("1.2.3.4", "Mozilla/5.0")
	sessionID2 := GenerateSessionID("1.2.3.4", "Mozilla/5.0")

	// Should generate different IDs (due to timestamp)
	if sessionID1 == sessionID2 {
		t.Error("Expected different session IDs")
	}

	// Should be non-empty
	if sessionID1 == "" {
		t.Error("Expected non-empty session ID")
	}
}
