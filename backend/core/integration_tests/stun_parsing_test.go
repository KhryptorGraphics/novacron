package integration_tests

import (
	"encoding/binary"
	"net"
	"os"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/discovery"
	"go.uber.org/zap"
)

func TestSTUNAddressParsing(t *testing.T) {
	// Skip STUN tests in CI environment to avoid network dependencies
	if os.Getenv("CI") != "" || os.Getenv("SKIP_STUN_TESTS") != "" {
		t.Skip("Skipping STUN tests in CI environment (set SKIP_STUN_TESTS=false to force run)")
	}

	logger := zap.NewNop()
	client := discovery.NewSTUNClient([]discovery.STUNServer{}, logger)

	// Test IPv4 MAPPED-ADDRESS parsing
	t.Run("IPv4_MAPPED_ADDRESS", func(t *testing.T) {
		// Create a test attribute
		// Format: 0x00 (reserved), 0x01 (family IPv4), port (2 bytes), IP (4 bytes)
		attr := &discovery.STUNAttribute{
			Type:   discovery.ATTR_MAPPED_ADDRESS,
			Length: 8,
			Value:  make([]byte, 8),
		}
		
		// Set family to IPv4 (0x01)
		attr.Value[0] = 0x00 // Reserved
		attr.Value[1] = 0x01 // Family
		
		// Set port to 12345
		binary.BigEndian.PutUint16(attr.Value[2:4], 12345)
		
		// Set IP to 192.168.1.1
		attr.Value[4] = 192
		attr.Value[5] = 168
		attr.Value[6] = 1
		attr.Value[7] = 1
		
		var transactionID [12]byte
		endpoint, err := client.ParseAddressAttribute(attr, false, transactionID)
		if err != nil {
			t.Fatalf("Failed to parse IPv4 MAPPED-ADDRESS: %v", err)
		}
		
		if endpoint.Port != 12345 {
			t.Errorf("Expected port 12345, got %d", endpoint.Port)
		}
		
		expectedIP := net.IPv4(192, 168, 1, 1)
		if !endpoint.IP.Equal(expectedIP) {
			t.Errorf("Expected IP %s, got %s", expectedIP, endpoint.IP)
		}
	})

	// Test IPv4 XOR-MAPPED-ADDRESS parsing
	t.Run("IPv4_XOR_MAPPED_ADDRESS", func(t *testing.T) {
		attr := &discovery.STUNAttribute{
			Type:   discovery.ATTR_XOR_MAPPED_ADDRESS,
			Length: 8,
			Value:  make([]byte, 8),
		}
		
		// Set family to IPv4 (0x01)
		attr.Value[0] = 0x00 // Reserved
		attr.Value[1] = 0x01 // Family
		
		// Port is XORed with magic cookie upper 16 bits
		port := uint16(12345)
		xorPort := port ^ uint16(discovery.STUN_MAGIC_COOKIE>>16)
		binary.BigEndian.PutUint16(attr.Value[2:4], xorPort)
		
		// IP is XORed with magic cookie
		ip := net.IPv4(192, 168, 1, 1)
		magicBytes := make([]byte, 4)
		binary.BigEndian.PutUint32(magicBytes, discovery.STUN_MAGIC_COOKIE)
		
		for i := 0; i < 4; i++ {
			attr.Value[4+i] = ip[i] ^ magicBytes[i]
		}
		
		var transactionID [12]byte
		endpoint, err := client.ParseAddressAttribute(attr, true, transactionID)
		if err != nil {
			t.Fatalf("Failed to parse IPv4 XOR-MAPPED-ADDRESS: %v", err)
		}
		
		if endpoint.Port != 12345 {
			t.Errorf("Expected port 12345, got %d", endpoint.Port)
		}
		
		expectedIP := net.IPv4(192, 168, 1, 1)
		if !endpoint.IP.Equal(expectedIP) {
			t.Errorf("Expected IP %s, got %s", expectedIP, endpoint.IP)
		}
	})

	// Test IPv6 parsing
	t.Run("IPv6_XOR_MAPPED_ADDRESS", func(t *testing.T) {
		attr := &discovery.STUNAttribute{
			Type:   discovery.ATTR_XOR_MAPPED_ADDRESS,
			Length: 20,
			Value:  make([]byte, 20),
		}
		
		// Set family to IPv6 (0x02)
		attr.Value[0] = 0x00 // Reserved
		attr.Value[1] = 0x02 // Family
		
		// Port is XORed with magic cookie upper 16 bits
		port := uint16(8080)
		xorPort := port ^ uint16(discovery.STUN_MAGIC_COOKIE>>16)
		binary.BigEndian.PutUint16(attr.Value[2:4], xorPort)
		
		// IPv6 address (simplified test with zeros)
		ipv6 := net.ParseIP("2001:db8::1")
		if ipv6 == nil {
			t.Fatal("Failed to parse test IPv6 address")
		}
		
		// XOR first 4 bytes with magic cookie
		magicBytes := make([]byte, 4)
		binary.BigEndian.PutUint32(magicBytes, discovery.STUN_MAGIC_COOKIE)
		
		for i := 0; i < 4; i++ {
			attr.Value[4+i] = ipv6[i] ^ magicBytes[i]
		}
		
		// XOR remaining 12 bytes with transaction ID
		transactionID := [12]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
		for i := 4; i < 16; i++ {
			attr.Value[4+i] = ipv6[i] ^ transactionID[i-4]
		}
		
		endpoint, err := client.ParseAddressAttribute(attr, true, transactionID)
		if err != nil {
			t.Fatalf("Failed to parse IPv6 XOR-MAPPED-ADDRESS: %v", err)
		}
		
		if endpoint.Port != 8080 {
			t.Errorf("Expected port 8080, got %d", endpoint.Port)
		}
		
		t.Logf("Successfully parsed IPv6 address: %s:%d", endpoint.IP, endpoint.Port)
	})
}

func TestInvalidSTUNAttributes(t *testing.T) {
	// Skip STUN tests in CI environment to avoid network dependencies
	if os.Getenv("CI") != "" || os.Getenv("SKIP_STUN_TESTS") != "" {
		t.Skip("Skipping STUN tests in CI environment (set SKIP_STUN_TESTS=false to force run)")
	}

	logger := zap.NewNop()
	client := discovery.NewSTUNClient([]discovery.STUNServer{}, logger)

	// Test attribute too short
	t.Run("AttributeTooShort", func(t *testing.T) {
		attr := &discovery.STUNAttribute{
			Type:   discovery.ATTR_MAPPED_ADDRESS,
			Length: 4, // Too short
			Value:  make([]byte, 4),
		}
		
		var transactionID [12]byte
		_, err := client.ParseAddressAttribute(attr, false, transactionID)
		if err == nil {
			t.Error("Expected error for short attribute, got nil")
		}
	})

	// Test invalid family
	t.Run("InvalidFamily", func(t *testing.T) {
		attr := &discovery.STUNAttribute{
			Type:   discovery.ATTR_MAPPED_ADDRESS,
			Length: 8,
			Value:  make([]byte, 8),
		}
		
		// Set invalid family (0x03)
		attr.Value[0] = 0x00
		attr.Value[1] = 0x03
		
		var transactionID [12]byte
		_, err := client.ParseAddressAttribute(attr, false, transactionID)
		if err == nil {
			t.Error("Expected error for invalid family, got nil")
		}
	})
}