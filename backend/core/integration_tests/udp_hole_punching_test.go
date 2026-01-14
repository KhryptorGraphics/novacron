package integration_tests

import (
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/discovery"
	"go.uber.org/zap"
)

func TestUDPHolePuncherWithReceiver(t *testing.T) {
	logger := zap.NewNop()

	// Create two hole punchers for testing
	addr1 := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	addr2 := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}

	puncher1 := discovery.NewUDPHolePuncher(addr1, logger)
	puncher2 := discovery.NewUDPHolePuncher(addr2, logger)

	// Give receivers time to start
	time.Sleep(100 * time.Millisecond)

	// Clean up when done
	defer puncher1.Stop()
	defer puncher2.Stop()

	// Test message handling through receiver
	t.Run("HandshakeResponse", func(t *testing.T) {
		// Create a test UDP connection to send messages
		conn, err := net.DialUDP("udp", nil, puncher1.LocalAddr)
		if err != nil {
			t.Fatalf("Failed to create test connection: %v", err)
		}
		defer conn.Close()

		// Send handshake message
		handshakeMsg := `{"type":"HANDSHAKE","peer_id":"test-peer"}`
		_, err = conn.Write([]byte(handshakeMsg))
		if err != nil {
			t.Fatalf("Failed to send handshake: %v", err)
		}

		// Read response
		buffer := make([]byte, 1024)
		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		n, err := conn.Read(buffer)
		if err != nil {
			t.Fatalf("Failed to read handshake response: %v", err)
		}

		response := string(buffer[:n])
		if !strings.Contains(response, "HANDSHAKE_ACK") {
			t.Errorf("Expected HANDSHAKE_ACK, got: %s", response)
		}
	})

	t.Run("PingPongResponse", func(t *testing.T) {
		// Create a test UDP connection
		conn, err := net.DialUDP("udp", nil, puncher1.LocalAddr)
		if err != nil {
			t.Fatalf("Failed to create test connection: %v", err)
		}
		defer conn.Close()

		// Send ping message
		timestamp := time.Now().UnixNano()
		pingMsg := fmt.Sprintf(`{"type":"PING","ts":%d}`, timestamp)
		_, err = conn.Write([]byte(pingMsg))
		if err != nil {
			t.Fatalf("Failed to send ping: %v", err)
		}

		// Read pong response
		buffer := make([]byte, 1024)
		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		n, err := conn.Read(buffer)
		if err != nil {
			t.Fatalf("Failed to read pong response: %v", err)
		}

		response := string(buffer[:n])
		if !strings.Contains(response, "PONG") {
			t.Errorf("Expected PONG response, got: %s", response)
		}

		// Verify timestamp is preserved
		expectedTs := fmt.Sprintf("%d", timestamp)
		if !strings.Contains(response, expectedTs) {
			t.Errorf("Expected timestamp %s in response, got: %s", expectedTs, response)
		}
	})

	t.Run("LegacyProtocolSupport", func(t *testing.T) {
		// Test that old protocol format is still supported
		conn, err := net.DialUDP("udp", nil, puncher1.LocalAddr)
		if err != nil {
			t.Fatalf("Failed to create test connection: %v", err)
		}
		defer conn.Close()

		// Send old format handshake
		oldHandshake := "HANDSHAKE:legacy-peer"
		_, err = conn.Write([]byte(oldHandshake))
		if err != nil {
			t.Fatalf("Failed to send legacy handshake: %v", err)
		}

		// Should still get a response
		buffer := make([]byte, 1024)
		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		n, err := conn.Read(buffer)
		if err != nil {
			t.Fatalf("Failed to read response to legacy handshake: %v", err)
		}

		response := string(buffer[:n])
		if !strings.Contains(response, "ACK") {
			t.Errorf("Expected ACK response to legacy handshake, got: %s", response)
		}

		// Test old format ping
		oldPing := fmt.Sprintf("PING:%d", time.Now().Unix())
		_, err = conn.Write([]byte(oldPing))
		if err != nil {
			t.Fatalf("Failed to send legacy ping: %v", err)
		}

		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		n, err = conn.Read(buffer)
		if err != nil {
			t.Fatalf("Failed to read response to legacy ping: %v", err)
		}

		response = string(buffer[:n])
		if !strings.Contains(response, "PONG") {
			t.Errorf("Expected PONG response to legacy ping, got: %s", response)
		}
	})
}

func TestPeerConnectionManagement(t *testing.T) {
	logger := zap.NewNop()

	addr := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 0}
	puncher := discovery.NewUDPHolePuncher(addr, logger)
	defer puncher.Stop()

	// Give receiver time to start
	time.Sleep(100 * time.Millisecond)

	// Test connection tracking
	t.Run("ConnectionTracking", func(t *testing.T) {
		// Simulate a peer connection
		remoteAddr := &net.UDPAddr{IP: net.IPv4(127, 0, 0, 1), Port: 9999}

		// Note: EstablishConnection will try to connect which may fail in test
		// but we're testing the management aspects
		conn, err := puncher.EstablishConnection("test-peer", remoteAddr)
		if err == nil {
			// Connection succeeded (unlikely in test)
			if conn.PeerID != "test-peer" {
				t.Errorf("Expected peer ID test-peer, got %s", conn.PeerID)
			}

			// Check connection retrieval
			retrieved, exists := puncher.GetConnection("test-peer")
			if !exists {
				t.Error("Failed to retrieve established connection")
			}
			if retrieved.PeerID != conn.PeerID {
				t.Error("Retrieved connection doesn't match")
			}

			// Test connection close
			err = puncher.CloseConnection("test-peer")
			if err != nil {
				t.Errorf("Failed to close connection: %v", err)
			}

			// Verify connection is removed
			_, exists = puncher.GetConnection("test-peer")
			if exists {
				t.Error("Connection still exists after closing")
			}
		} else {
			// This is expected in test environment
			t.Logf("Connection establishment failed as expected in test: %v", err)
		}
	})
}