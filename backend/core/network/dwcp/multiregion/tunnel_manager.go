package multiregion

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"sync"
	"time"
)

// TunnelType defines the type of VPN tunnel
type TunnelType int

const (
	TunnelIPSec TunnelType = iota
	TunnelWireGuard
	TunnelGRE
	TunnelVXLAN
)

// TunnelProtocol defines the transport protocol
type TunnelProtocol int

const (
	ProtocolUDP TunnelProtocol = iota
	ProtocolTCP
	ProtocolSCTP
)

// TunnelStatus represents the operational status
type TunnelStatus int

const (
	TunnelStatusDown TunnelStatus = iota
	TunnelStatusUp
	TunnelStatusDegraded
	TunnelStatusConfiguring
)

// EncryptionConfig defines encryption settings
type EncryptionConfig struct {
	Algorithm   string
	KeySize     int
	Cipher      string
	PublicKey   string
	PrivateKey  string
	PresharedKey string
}

// VPNTunnel represents a VPN tunnel between regions
type VPNTunnel struct {
	ID          string
	Type        TunnelType
	Source      NetworkEndpoint
	Destination NetworkEndpoint
	Protocol    TunnelProtocol
	Encryption  EncryptionConfig
	Status      TunnelStatus
	CreatedAt   time.Time
	UpdatedAt   time.Time
	Metrics     *TunnelMetrics
	mu          sync.RWMutex
}

// TunnelMetrics tracks tunnel performance
type TunnelMetrics struct {
	BytesSent      uint64
	BytesReceived  uint64
	PacketsSent    uint64
	PacketsReceived uint64
	Errors         uint64
	Latency        time.Duration
	Uptime         time.Duration
	LastHandshake  time.Time
}

// TunnelMonitor monitors tunnel health
type TunnelMonitor struct {
	tunnels     map[string]*VPNTunnel
	healthCheck time.Duration
	mu          sync.RWMutex
}

// TunnelManager manages VPN tunnels
type TunnelManager struct {
	tunnels  map[string]*VPNTunnel
	monitor  *TunnelMonitor
	topology *GlobalTopology
	mu       sync.RWMutex
}

// WireGuardConfig represents WireGuard configuration
type WireGuardConfig struct {
	PrivateKey          string
	PublicKey           string
	ListenPort          int
	Peers               []WireGuardPeer
	MTU                 int
	PersistentKeepalive int
}

// WireGuardPeer represents a WireGuard peer
type WireGuardPeer struct {
	PublicKey           string
	Endpoint            string
	AllowedIPs          []string
	PersistentKeepalive int
}

// NewTunnelManager creates a new tunnel manager
func NewTunnelManager(topology *GlobalTopology) *TunnelManager {
	tm := &TunnelManager{
		tunnels:  make(map[string]*VPNTunnel),
		topology: topology,
		monitor: &TunnelMonitor{
			tunnels:     make(map[string]*VPNTunnel),
			healthCheck: 30 * time.Second,
		},
	}

	// Start health monitoring
	go tm.monitor.start()

	return tm
}

// EstablishTunnel establishes a VPN tunnel between two regions
func (tm *TunnelManager) EstablishTunnel(srcRegion, dstRegion *Region) (*VPNTunnel, error) {
	if len(srcRegion.Endpoints) == 0 || len(dstRegion.Endpoints) == 0 {
		return nil, fmt.Errorf("regions must have network endpoints")
	}

	tunnelID := fmt.Sprintf("tunnel-%s-%s", srcRegion.ID, dstRegion.ID)

	// Check if tunnel already exists
	tm.mu.RLock()
	if existing, exists := tm.tunnels[tunnelID]; exists {
		tm.mu.RUnlock()
		return existing, nil
	}
	tm.mu.RUnlock()

	// Create new tunnel
	tunnel := &VPNTunnel{
		ID:          tunnelID,
		Type:        TunnelWireGuard, // Default to WireGuard
		Source:      srcRegion.Endpoints[0],
		Destination: dstRegion.Endpoints[0],
		Protocol:    ProtocolUDP,
		Status:      TunnelStatusConfiguring,
		CreatedAt:   time.Now(),
		Metrics:     &TunnelMetrics{},
	}

	// Generate encryption keys
	if err := tm.generateEncryptionKeys(tunnel); err != nil {
		return nil, fmt.Errorf("failed to generate keys: %w", err)
	}

	// Configure tunnel based on type
	switch tunnel.Type {
	case TunnelWireGuard:
		if err := tm.configureWireGuard(tunnel); err != nil {
			return nil, fmt.Errorf("failed to configure WireGuard: %w", err)
		}
	case TunnelIPSec:
		if err := tm.configureIPSec(tunnel); err != nil {
			return nil, fmt.Errorf("failed to configure IPSec: %w", err)
		}
	case TunnelVXLAN:
		if err := tm.configureVXLAN(tunnel); err != nil {
			return nil, fmt.Errorf("failed to configure VXLAN: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported tunnel type: %v", tunnel.Type)
	}

	// Store tunnel
	tm.mu.Lock()
	tm.tunnels[tunnelID] = tunnel
	tm.mu.Unlock()

	// Start monitoring
	tm.monitor.StartMonitoring(tunnel)

	// Update status
	tunnel.mu.Lock()
	tunnel.Status = TunnelStatusUp
	tunnel.UpdatedAt = time.Now()
	tunnel.mu.Unlock()

	return tunnel, nil
}

// generateEncryptionKeys generates encryption keys for the tunnel
func (tm *TunnelManager) generateEncryptionKeys(tunnel *VPNTunnel) error {
	// Generate WireGuard-style keys (32 bytes)
	privateKey := make([]byte, 32)
	if _, err := rand.Read(privateKey); err != nil {
		return err
	}

	// For WireGuard, public key is derived from private key
	// In production, use proper WireGuard key derivation
	publicKey := make([]byte, 32)
	if _, err := rand.Read(publicKey); err != nil {
		return err
	}

	presharedKey := make([]byte, 32)
	if _, err := rand.Read(presharedKey); err != nil {
		return err
	}

	tunnel.Encryption = EncryptionConfig{
		Algorithm:    "ChaCha20-Poly1305",
		KeySize:      256,
		Cipher:       "AEAD",
		PrivateKey:   base64.StdEncoding.EncodeToString(privateKey),
		PublicKey:    base64.StdEncoding.EncodeToString(publicKey),
		PresharedKey: base64.StdEncoding.EncodeToString(presharedKey),
	}

	return nil
}

// configureWireGuard configures a WireGuard tunnel
func (tm *TunnelManager) configureWireGuard(tunnel *VPNTunnel) error {
	config := WireGuardConfig{
		PrivateKey: tunnel.Encryption.PrivateKey,
		PublicKey:  tunnel.Encryption.PublicKey,
		ListenPort: tunnel.Source.Port,
		MTU:        1420, // Standard WireGuard MTU
		Peers: []WireGuardPeer{
			{
				PublicKey:           tunnel.Encryption.PublicKey,
				Endpoint:            fmt.Sprintf("%s:%d", tunnel.Destination.Address, tunnel.Destination.Port),
				AllowedIPs:          []string{"10.0.0.0/8", "172.16.0.0/12"},
				PersistentKeepalive: 25,
			},
		},
	}

	// In production, this would actually configure the WireGuard interface
	// using netlink or wgctrl library
	_ = config

	return nil
}

// configureIPSec configures an IPSec tunnel
func (tm *TunnelManager) configureIPSec(tunnel *VPNTunnel) error {
	// IPSec configuration would go here
	// This would involve setting up IKEv2, ESP, etc.
	return nil
}

// configureVXLAN configures a VXLAN tunnel
func (tm *TunnelManager) configureVXLAN(tunnel *VPNTunnel) error {
	// VXLAN configuration would go here
	// This would set up VXLAN interfaces with proper VNI
	return nil
}

// TeardownTunnel tears down a VPN tunnel
func (tm *TunnelManager) TeardownTunnel(tunnelID string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tunnel, exists := tm.tunnels[tunnelID]
	if !exists {
		return fmt.Errorf("tunnel %s not found", tunnelID)
	}

	// Stop monitoring
	tm.monitor.StopMonitoring(tunnelID)

	// Update status
	tunnel.mu.Lock()
	tunnel.Status = TunnelStatusDown
	tunnel.UpdatedAt = time.Now()
	tunnel.mu.Unlock()

	// Remove tunnel
	delete(tm.tunnels, tunnelID)

	return nil
}

// GetTunnel retrieves a tunnel by ID
func (tm *TunnelManager) GetTunnel(tunnelID string) (*VPNTunnel, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	tunnel, exists := tm.tunnels[tunnelID]
	if !exists {
		return nil, fmt.Errorf("tunnel %s not found", tunnelID)
	}

	return tunnel, nil
}

// ListTunnels returns all active tunnels
func (tm *TunnelManager) ListTunnels() []*VPNTunnel {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	tunnels := make([]*VPNTunnel, 0, len(tm.tunnels))
	for _, tunnel := range tm.tunnels {
		tunnels = append(tunnels, tunnel)
	}

	return tunnels
}

// TunnelMonitor methods

// StartMonitoring starts monitoring a tunnel
func (mon *TunnelMonitor) StartMonitoring(tunnel *VPNTunnel) {
	mon.mu.Lock()
	defer mon.mu.Unlock()

	mon.tunnels[tunnel.ID] = tunnel
}

// StopMonitoring stops monitoring a tunnel
func (mon *TunnelMonitor) StopMonitoring(tunnelID string) {
	mon.mu.Lock()
	defer mon.mu.Unlock()

	delete(mon.tunnels, tunnelID)
}

// start begins the monitoring loop
func (mon *TunnelMonitor) start() {
	ticker := time.NewTicker(mon.healthCheck)
	defer ticker.Stop()

	for range ticker.C {
		mon.checkHealth()
	}
}

// checkHealth performs health checks on all tunnels
func (mon *TunnelMonitor) checkHealth() {
	mon.mu.RLock()
	tunnels := make([]*VPNTunnel, 0, len(mon.tunnels))
	for _, tunnel := range mon.tunnels {
		tunnels = append(tunnels, tunnel)
	}
	mon.mu.RUnlock()

	for _, tunnel := range tunnels {
		mon.probeTunnel(tunnel)
	}
}

// probeTunnel sends a probe through the tunnel
func (mon *TunnelMonitor) probeTunnel(tunnel *VPNTunnel) {
	// In production, this would send actual probe packets
	// For now, we'll simulate it

	tunnel.mu.Lock()
	defer tunnel.mu.Unlock()

	// Update last handshake time (simulated)
	tunnel.Metrics.LastHandshake = time.Now()

	// Update uptime
	if tunnel.Status == TunnelStatusUp {
		tunnel.Metrics.Uptime += mon.healthCheck
	}
}

// IsHealthy checks if a tunnel is healthy
func (mon *TunnelMonitor) IsHealthy(tunnel *VPNTunnel) bool {
	tunnel.mu.RLock()
	defer tunnel.mu.RUnlock()

	if tunnel.Status != TunnelStatusUp {
		return false
	}

	// Check if we've had a recent handshake
	timeSinceHandshake := time.Since(tunnel.Metrics.LastHandshake)
	return timeSinceHandshake < 2*mon.healthCheck
}

// String methods for enums

func (tt TunnelType) String() string {
	switch tt {
	case TunnelIPSec:
		return "IPSec"
	case TunnelWireGuard:
		return "WireGuard"
	case TunnelGRE:
		return "GRE"
	case TunnelVXLAN:
		return "VXLAN"
	default:
		return "Unknown"
	}
}

func (ts TunnelStatus) String() string {
	switch ts {
	case TunnelStatusDown:
		return "DOWN"
	case TunnelStatusUp:
		return "UP"
	case TunnelStatusDegraded:
		return "DEGRADED"
	case TunnelStatusConfiguring:
		return "CONFIGURING"
	default:
		return "UNKNOWN"
	}
}

func (tp TunnelProtocol) String() string {
	switch tp {
	case ProtocolUDP:
		return "UDP"
	case ProtocolTCP:
		return "TCP"
	case ProtocolSCTP:
		return "SCTP"
	default:
		return "Unknown"
	}
}
