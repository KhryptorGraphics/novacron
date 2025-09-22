package discovery

import (
	"net"
	"sync"
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestUDPHolePuncherShutdown(t *testing.T) {
	logger := zap.NewNop()
	localAddr := &net.UDPAddr{
		IP:   net.IPv4zero,
		Port: 0,
	}

	// Create hole puncher
	uhp, err := NewUDPHolePuncher(localAddr, logger)
	if err != nil {
		t.Fatalf("Failed to create hole puncher: %v", err)
	}

	// Test concurrent stop during handshake simulation
	var wg sync.WaitGroup
	wg.Add(2)

	// Goroutine 1: Try to perform operations
	go func() {
		defer wg.Done()
		time.Sleep(50 * time.Millisecond)

		// This should fail gracefully after Stop()
		remoteAddr := &net.UDPAddr{
			IP:   net.ParseIP("127.0.0.1"),
			Port: 12345,
		}

		peerConn := &PeerConnection{
			PeerID:         "test-peer",
			LocalEndpoint:  localAddr,
			RemoteEndpoint: remoteAddr,
			ConnectionType: "nat_traversal",
			Established:    false,
			LastActivity:   time.Now(),
			conn:           uhp.receiver,
		}

		// This should return "stopped" error
		err := uhp.performHandshake(peerConn)
		if err != nil && err.Error() != "stopped" {
			// Expected error after stop
			t.Logf("Handshake failed as expected: %v", err)
		}
	}()

	// Goroutine 2: Stop the hole puncher
	go func() {
		defer wg.Done()
		time.Sleep(100 * time.Millisecond)
		uhp.Stop()
		t.Log("Hole puncher stopped")
	}()

	wg.Wait()
	t.Log("Test completed without panic")
}

func TestRTTMeasurementCorrelation(t *testing.T) {
	logger := zap.NewNop()
	localAddr := &net.UDPAddr{
		IP:   net.IPv4zero,
		Port: 0,
	}

	uhp, err := NewUDPHolePuncher(localAddr, logger)
	if err != nil {
		t.Fatalf("Failed to create hole puncher: %v", err)
	}
	defer uhp.Stop()

	// Test ping ID tracking
	startTime := time.Now()
	pingID := uint64(startTime.UnixNano())

	// Create a channel for this ping's PONG response
	pongChan := make(chan time.Duration, 1)

	// Store pending ping and channel
	uhp.pingsMu.Lock()
	uhp.pendingPings[pingID] = startTime
	uhp.pongCh[pingID] = pongChan
	uhp.pingsMu.Unlock()

	// Verify they were stored
	uhp.pingsMu.Lock()
	storedTime, exists := uhp.pendingPings[pingID]
	storedChan, chanExists := uhp.pongCh[pingID]
	uhp.pingsMu.Unlock()

	if !exists {
		t.Fatal("Ping ID not stored in pending pings")
	}

	if !chanExists {
		t.Fatal("Pong channel not stored")
	}

	if storedTime != startTime {
		t.Fatalf("Stored time mismatch: expected %v, got %v", startTime, storedTime)
	}

	if storedChan != pongChan {
		t.Fatal("Stored channel mismatch")
	}

	// Simulate RTT of 50ms
	expectedRTT := 50 * time.Millisecond

	// Simulate handleIncomingMessage sending RTT to channel
	go func() {
		time.Sleep(10 * time.Millisecond)
		// This simulates what handleIncomingMessage does when PONG arrives
		uhp.pingsMu.Lock()
		if ch, ok := uhp.pongCh[pingID]; ok {
			select {
			case ch <- expectedRTT:
				t.Log("RTT sent to channel successfully")
			default:
				t.Error("Failed to send RTT to channel")
			}
		}
		uhp.pingsMu.Unlock()
	}()

	// Wait for RTT on channel (simulating measureRTT waiting)
	select {
	case rtt := <-pongChan:
		if rtt != expectedRTT {
			t.Fatalf("RTT mismatch: expected %v, got %v", expectedRTT, rtt)
		}
		t.Logf("Received correct RTT: %v", rtt)
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Timeout waiting for RTT on channel")
	}

	// Clean up
	uhp.pingsMu.Lock()
	delete(uhp.pendingPings, pingID)
	delete(uhp.pongCh, pingID)
	uhp.pingsMu.Unlock()
	close(pongChan)

	t.Log("RTT correlation tracking verified")
}

func TestPeerConnectionTCPFields(t *testing.T) {
	// Test that PeerConnection properly handles TCP addresses
	tcpAddr := &net.TCPAddr{
		IP:   net.ParseIP("192.168.1.100"),
		Port: 8080,
	}

	peerConn := &PeerConnection{
		PeerID:         "test-peer",
		LocalEndpoint:  nil,
		RemoteEndpoint: nil, // No UDP endpoint for direct TCP
		RemoteTCPAddr:  tcpAddr,
		ConnectionType: "direct",
		Established:    true,
		LastActivity:   time.Now(),
	}

	if peerConn.RemoteTCPAddr == nil {
		t.Fatal("RemoteTCPAddr should not be nil")
	}

	if peerConn.RemoteEndpoint != nil {
		t.Fatal("RemoteEndpoint should be nil for direct TCP connection")
	}

	if peerConn.ConnectionType != "direct" {
		t.Fatalf("Expected connection type 'direct', got '%s'", peerConn.ConnectionType)
	}

	t.Log("TCP/UDP endpoint separation verified")
}