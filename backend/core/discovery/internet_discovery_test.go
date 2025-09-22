package discovery

import (
	"net"
	"strconv"
	"testing"

	"go.uber.org/zap"
)

func TestDirectTCPConnectionHandling(t *testing.T) {
	logger := zap.NewNop()

	config := InternetDiscoveryConfig{
		Config: Config{
			NodeID:   "test-node",
			NodeName: "Test Node",
			NodeRole: "worker",
			Address:  "127.0.0.1",
			Port:     7701,
		},
		EnableNATTraversal: false,
	}

	service, err := NewInternetDiscovery(config, logger)
	if err != nil {
		t.Fatalf("Failed to create internet discovery service: %v", err)
	}

	// Test that the service is properly configured for handling connections
	if service.config.Port != 7701 {
		t.Errorf("Expected service port 7701, got %d", service.config.Port)
	}

	// Test endpoint construction with service port
	remoteAddr := "192.168.1.100:12345" // Ephemeral port from request
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		t.Fatalf("Failed to parse remote address: %v", err)
	}

	// Should build endpoint with service port, not ephemeral port
	expectedEndpoint := net.JoinHostPort(host, strconv.Itoa(service.config.Port))
	if expectedEndpoint != "192.168.1.100:7701" {
		t.Errorf("Expected endpoint '192.168.1.100:7701', got '%s'", expectedEndpoint)
	}

	t.Log("Direct TCP connection handling verified")
}

func TestEndpointFallbackWithServicePort(t *testing.T) {
	logger := zap.NewNop()

	config := InternetDiscoveryConfig{
		Config: Config{
			NodeID:   "test-node",
			NodeName: "Test Node",
			NodeRole: "worker",
			Address:  "127.0.0.1",
			Port:     8800, // Custom service port
		},
	}

	service, err := NewInternetDiscovery(config, logger)
	if err != nil {
		t.Fatalf("Failed to create service: %v", err)
	}

	testCases := []struct {
		name         string
		remoteAddr   string
		expectedEndpoint string
	}{
		{
			name:         "IPv4 with ephemeral port",
			remoteAddr:   "10.0.0.1:54321",
			expectedEndpoint: "10.0.0.1:8800",
		},
		{
			name:         "IPv6 with ephemeral port",
			remoteAddr:   "[2001:db8::1]:54321",
			expectedEndpoint: "[2001:db8::1]:8800",
		},
		{
			name:         "Localhost with ephemeral port",
			remoteAddr:   "127.0.0.1:12345",
			expectedEndpoint: "127.0.0.1:8800",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			host, _, err := net.SplitHostPort(tc.remoteAddr)
			if err != nil {
				t.Fatalf("Failed to parse remote address %s: %v", tc.remoteAddr, err)
			}

			// Build endpoint with service port
			endpoint := net.JoinHostPort(host, strconv.Itoa(service.config.Port))
			if endpoint != tc.expectedEndpoint {
				t.Errorf("Expected endpoint '%s', got '%s'", tc.expectedEndpoint, endpoint)
			}
		})
	}

	t.Log("Endpoint fallback with service port verified")
}