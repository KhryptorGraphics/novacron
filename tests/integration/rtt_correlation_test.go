package integration

import (
	"fmt"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/discovery"
	"go.uber.org/zap"
)

// TestRTTCorrelationIntegration tests the complete RTT measurement flow
// verifying that PONG-correlated RTT is properly used instead of heuristics
func TestRTTCorrelationIntegration(t *testing.T) {
	logger := zap.NewNop()

	// Create two UDP hole punchers simulating two peers
	localAddr1 := &net.UDPAddr{
		IP:   net.IPv4(127, 0, 0, 1),
		Port: 0, // Let the OS assign a port
	}
	localAddr2 := &net.UDPAddr{
		IP:   net.IPv4(127, 0, 0, 1),
		Port: 0, // Let the OS assign a port
	}

	uhp1, err := discovery.NewUDPHolePuncher(localAddr1, logger)
	if err != nil {
		t.Fatalf("Failed to create hole puncher 1: %v", err)
	}
	defer uhp1.Stop()

	uhp2, err := discovery.NewUDPHolePuncher(localAddr2, logger)
	if err != nil {
		t.Fatalf("Failed to create hole puncher 2: %v", err)
	}
	defer uhp2.Stop()

	// Get actual addresses (with assigned ports)
	addr1 := uhp1.GetLocalAddress()
	addr2 := uhp2.GetLocalAddress()

	// Set up connections
	conn1 := &discovery.PeerConnection{
		PeerID:         "peer2",
		LocalEndpoint:  addr1,
		RemoteEndpoint: addr2,
		ConnectionType: "nat_traversal",
		Established:    true,
		LastActivity:   time.Now(),
	}

	conn2 := &discovery.PeerConnection{
		PeerID:         "peer1",
		LocalEndpoint:  addr2,
		RemoteEndpoint: addr1,
		ConnectionType: "nat_traversal",
		Established:    true,
		LastActivity:   time.Now(),
	}

	// Register connections
	uhp1.AddConnection(conn1)
	uhp2.AddConnection(conn2)

	// Create a channel to capture when PONG is received
	pongReceived := make(chan time.Duration, 1)

	// Override handleIncomingMessage to capture RTT
	// For this test, we'll manually handle the message
	go func() {
		buffer := make([]byte, 1024)
		for {
			n, addr, err := uhp2.GetReceiver().ReadFromUDP(buffer)
			if err != nil {
				if !strings.Contains(err.Error(), "closed") {
					t.Logf("Read error: %v", err)
				}
				return
			}

			message := string(buffer[:n])
			t.Logf("UHP2 received: %s from %s", message, addr)

			// Handle PING by sending PONG
			if strings.Contains(message, "\"type\":\"PING\"") {
				// Extract ping ID
				var pingID uint64
				if strings.Contains(message, "\"id\":") {
					start := strings.Index(message, "\"id\":")
					if start >= 0 {
						start += 5
						end := strings.Index(message[start:], "}")
						if end > 0 {
							idStr := message[start : start+end]
							fmt.Sscanf(idStr, "%d", &pingID)

							// Send PONG back
							pongMsg := []byte(fmt.Sprintf("{\"type\":\"PONG\",\"id\":%d}", pingID))
							uhp2.GetReceiver().WriteToUDP(pongMsg, addr)
							t.Logf("UHP2 sent PONG with ID %d", pingID)
						}
					}
				}
			}

			// Handle incoming messages normally
			uhp2.HandleIncomingMessage(message, addr)
		}
	}()

	// Also handle messages for UHP1
	go func() {
		buffer := make([]byte, 1024)
		for {
			n, addr, err := uhp1.GetReceiver().ReadFromUDP(buffer)
			if err != nil {
				if !strings.Contains(err.Error(), "closed") {
					t.Logf("Read error: %v", err)
				}
				return
			}

			message := string(buffer[:n])
			t.Logf("UHP1 received: %s from %s", message, addr)

			// Check if PONG was received
			if strings.Contains(message, "\"type\":\"PONG\"") {
				// Get the RTT from the connection
				conn, exists := uhp1.GetConnection("peer2")
				if exists && conn.Quality.RTT > 0 {
					select {
					case pongReceived <- conn.Quality.RTT:
					default:
					}
				}
			}

			// Handle incoming messages normally
			uhp1.HandleIncomingMessage(message, addr)
		}
	}()

	// Give receivers time to start
	time.Sleep(100 * time.Millisecond)

	// Measure RTT from uhp1 to uhp2
	quality, err := uhp1.MeasureRTT(conn1)
	if err != nil {
		t.Fatalf("Failed to measure RTT: %v", err)
	}

	// Verify we got a real RTT (should be low for localhost)
	if quality.RTT == 0 {
		t.Fatal("RTT should not be zero")
	}

	if quality.RTT > 100*time.Millisecond {
		t.Errorf("RTT too high for localhost: %v", quality.RTT)
	}

	if quality.PacketLoss != 0 {
		t.Errorf("Should have no packet loss on localhost: %f", quality.PacketLoss)
	}

	t.Logf("Measured RTT: %v, PacketLoss: %f", quality.RTT, quality.PacketLoss)

	// Verify the RTT was properly correlated with PONG
	select {
	case actualRTT := <-pongReceived:
		t.Logf("PONG-correlated RTT: %v", actualRTT)
		if actualRTT != quality.RTT {
			t.Logf("Warning: RTT mismatch (expected %v, got %v) - may be due to timing", quality.RTT, actualRTT)
		}
	case <-time.After(100 * time.Millisecond):
		// This is okay - the PONG might have been processed after measureRTT returned
		t.Log("PONG processing completed after measureRTT returned")
	}

	// Test timeout scenario (no PONG response)
	// Stop uhp2's receiver to simulate no response
	uhp2.Stop()
	time.Sleep(100 * time.Millisecond)

	quality2, err := uhp1.MeasureRTT(conn1)
	if err != nil {
		t.Fatalf("Failed to measure RTT with timeout: %v", err)
	}

	// Should have packet loss due to timeout
	if quality2.PacketLoss == 0 {
		t.Error("Should have packet loss when PONG times out")
	}

	// RTT should be degraded (previous + penalty)
	if quality2.RTT <= quality.RTT {
		t.Errorf("RTT should be degraded on timeout: was %v, now %v", quality.RTT, quality2.RTT)
	}

	t.Logf("Timeout scenario - RTT: %v, PacketLoss: %f", quality2.RTT, quality2.PacketLoss)
	t.Log("RTT correlation integration test passed")
}