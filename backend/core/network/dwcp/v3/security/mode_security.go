package security

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ModeAwareSecurity provides adaptive security based on deployment mode
// Datacenter: Trusted nodes, minimal overhead, fast consensus
// Internet: Untrusted nodes, full Byzantine tolerance, TLS/mTLS
// Hybrid: Adaptive security based on network conditions
type ModeAwareSecurity struct {
	mu sync.RWMutex

	nodeID string
	logger *zap.Logger

	// Current mode
	currentMode SecurityMode
	lastModeChange time.Time

	// Components
	byzantineDetector *ByzantineDetector
	reputationSystem  *ReputationSystem

	// TLS configuration
	tlsConfig     *tls.Config
	certManager   *CertificateManager
	tlsEnabled    bool

	// Mode-specific settings
	datacenterConfig *DatacenterSecurityConfig
	internetConfig   *InternetSecurityConfig
	hybridConfig     *HybridSecurityConfig

	// Adaptive thresholds
	networkTrust      float64 // 0-1, based on observed behavior
	adaptiveThreshold float64 // Threshold to switch modes

	ctx    context.Context
	cancel context.CancelFunc
}

// SecurityMode defines the security operating mode
type SecurityMode int

const (
	ModeDatacenter SecurityMode = iota // Trusted datacenter environment
	ModeInternet                       // Untrusted internet environment
	ModeHybrid                         // Adaptive based on conditions
)

// DatacenterSecurityConfig for trusted datacenter deployments
type DatacenterSecurityConfig struct {
	// Minimal security overhead
	SkipSignatureValidation bool
	SkipByzantineDetection  bool
	FastConsensusPath       bool

	// Basic checks only
	ValidateMessageFormat   bool
	CheckNodeIdentity       bool

	// Timeouts (shorter for low latency)
	MessageTimeout    time.Duration
	ConsensusTimeout  time.Duration
}

// InternetSecurityConfig for untrusted internet deployments
type InternetSecurityConfig struct {
	// Full security
	RequireTLS              bool
	RequireMutualTLS        bool
	RequireSignatures       bool
	EnableByzantineDetection bool
	EnableReputationSystem  bool

	// Strict validation
	ValidateAllMessages    bool
	StrictConsensusChecks  bool
	QuarantineAggressive   bool

	// Timeouts (longer for high latency)
	MessageTimeout       time.Duration
	ConsensusTimeout     time.Duration
	HandshakeTimeout     time.Duration

	// TLS settings
	MinTLSVersion        uint16
	RequireClientCerts   bool
	AllowedCipherSuites  []uint16
	CertValidityPeriod   time.Duration
}

// HybridSecurityConfig for adaptive security
type HybridSecurityConfig struct {
	// Thresholds for mode switching
	TrustThreshold          float64 // > this = datacenter mode
	UntrustThreshold        float64 // < this = internet mode

	// Monitoring
	MonitoringWindow        time.Duration
	AdaptiveCheckInterval   time.Duration

	// Gradual security adjustment
	GradualTransition       bool
	TransitionSteps         int

	// Fallback
	DefaultMode             SecurityMode
	FallbackOnAmbiguous     bool
}

// CertificateManager manages TLS certificates
type CertificateManager struct {
	mu sync.RWMutex

	nodeID string
	logger *zap.Logger

	// Certificates
	serverCert   *tls.Certificate
	clientCerts  map[string]*x509.Certificate
	caCertPool   *x509.CertPool

	// Rotation
	certValidUntil   time.Time
	rotationInterval time.Duration
	autoRotate       bool

	// Validation
	strictValidation bool
	checkCRL         bool
	checkOCSP        bool

	ctx    context.Context
	cancel context.CancelFunc
}

// NewModeAwareSecurity creates a new mode-aware security system
func NewModeAwareSecurity(nodeID string, mode SecurityMode, detector *ByzantineDetector, reputation *ReputationSystem, logger *zap.Logger) *ModeAwareSecurity {
	ctx, cancel := context.WithCancel(context.Background())

	mas := &ModeAwareSecurity{
		nodeID:            nodeID,
		logger:            logger,
		currentMode:       mode,
		lastModeChange:    time.Now(),
		byzantineDetector: detector,
		reputationSystem:  reputation,
		datacenterConfig:  DefaultDatacenterConfig(),
		internetConfig:    DefaultInternetConfig(),
		hybridConfig:      DefaultHybridConfig(),
		networkTrust:      0.5, // Neutral start
		adaptiveThreshold: 0.7,
		ctx:               ctx,
		cancel:            cancel,
	}

	// Initialize TLS for internet mode
	if mode == ModeInternet {
		mas.initializeTLS()
	}

	// Start adaptive monitoring for hybrid mode
	if mode == ModeHybrid {
		go mas.adaptiveMonitoring()
	}

	return mas
}

// DefaultDatacenterConfig returns default datacenter security config
func DefaultDatacenterConfig() *DatacenterSecurityConfig {
	return &DatacenterSecurityConfig{
		SkipSignatureValidation: true,
		SkipByzantineDetection:  true,
		FastConsensusPath:       true,
		ValidateMessageFormat:   true,
		CheckNodeIdentity:       true,
		MessageTimeout:          100 * time.Millisecond,
		ConsensusTimeout:        500 * time.Millisecond,
	}
}

// DefaultInternetConfig returns default internet security config
func DefaultInternetConfig() *InternetSecurityConfig {
	return &InternetSecurityConfig{
		RequireTLS:               true,
		RequireMutualTLS:         true,
		RequireSignatures:        true,
		EnableByzantineDetection: true,
		EnableReputationSystem:   true,
		ValidateAllMessages:      true,
		StrictConsensusChecks:    true,
		QuarantineAggressive:     true,
		MessageTimeout:           5 * time.Second,
		ConsensusTimeout:         30 * time.Second,
		HandshakeTimeout:         10 * time.Second,
		MinTLSVersion:            tls.VersionTLS13,
		RequireClientCerts:       true,
		AllowedCipherSuites: []uint16{
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_CHACHA20_POLY1305_SHA256,
		},
		CertValidityPeriod: 90 * 24 * time.Hour,
	}
}

// DefaultHybridConfig returns default hybrid security config
func DefaultHybridConfig() *HybridSecurityConfig {
	return &HybridSecurityConfig{
		TrustThreshold:        0.8,
		UntrustThreshold:      0.4,
		MonitoringWindow:      5 * time.Minute,
		AdaptiveCheckInterval: 30 * time.Second,
		GradualTransition:     true,
		TransitionSteps:       5,
		DefaultMode:           ModeInternet, // Default to secure
		FallbackOnAmbiguous:   true,
	}
}

// GetCurrentMode returns the current security mode
func (mas *ModeAwareSecurity) GetCurrentMode() SecurityMode {
	mas.mu.RLock()
	defer mas.mu.RUnlock()
	return mas.currentMode
}

// SwitchMode switches to a different security mode
func (mas *ModeAwareSecurity) SwitchMode(newMode SecurityMode, reason string) error {
	mas.mu.Lock()
	defer mas.mu.Unlock()

	if mas.currentMode == newMode {
		return nil // Already in this mode
	}

	oldMode := mas.currentMode
	mas.currentMode = newMode
	mas.lastModeChange = time.Now()

	mas.logger.Info("Security mode switched",
		zap.String("old_mode", mas.modeString(oldMode)),
		zap.String("new_mode", mas.modeString(newMode)),
		zap.String("reason", reason),
	)

	// Reconfigure based on new mode
	switch newMode {
	case ModeDatacenter:
		mas.configureDatacenterMode()
	case ModeInternet:
		mas.configureInternetMode()
	case ModeHybrid:
		mas.configureHybridMode()
	}

	return nil
}

// ValidateMessage validates a message based on current security mode
func (mas *ModeAwareSecurity) ValidateMessage(senderID string, messageType string, message interface{}, signature string) error {
	mas.mu.RLock()
	mode := mas.currentMode
	mas.mu.RUnlock()

	switch mode {
	case ModeDatacenter:
		return mas.validateDatacenterMessage(senderID, messageType, message)
	case ModeInternet:
		return mas.validateInternetMessage(senderID, messageType, message, signature)
	case ModeHybrid:
		return mas.validateHybridMessage(senderID, messageType, message, signature)
	default:
		return fmt.Errorf("unknown security mode: %v", mode)
	}
}

// validateDatacenterMessage validates message in datacenter mode (minimal checks)
func (mas *ModeAwareSecurity) validateDatacenterMessage(senderID string, messageType string, message interface{}) error {
	config := mas.datacenterConfig

	// Basic format validation
	if config.ValidateMessageFormat {
		if message == nil {
			return fmt.Errorf("nil message")
		}
	}

	// Basic identity check
	if config.CheckNodeIdentity {
		if senderID == "" {
			return fmt.Errorf("empty sender ID")
		}
	}

	return nil
}

// validateInternetMessage validates message in internet mode (full security)
func (mas *ModeAwareSecurity) validateInternetMessage(senderID string, messageType string, message interface{}, signature string) error {
	config := mas.internetConfig

	// Check if sender is quarantined
	if mas.reputationSystem != nil && mas.reputationSystem.IsQuarantined(senderID) {
		return fmt.Errorf("sender is quarantined: %s", senderID)
	}

	// Check reputation
	if mas.reputationSystem != nil {
		score := mas.reputationSystem.GetScore(senderID)
		if score < 20.0 { // Very low reputation
			return fmt.Errorf("sender has very low reputation: %.2f", score)
		}
	}

	// Validate signature
	if config.RequireSignatures {
		if signature == "" {
			return fmt.Errorf("signature required but not provided")
		}
		// Record for Byzantine detection
		if mas.byzantineDetector != nil {
			mas.byzantineDetector.RecordMessage(senderID, messageType, message, signature)
			// Actual signature verification would happen here
			// For now, we'll simulate it
			valid := len(signature) > 0 // Simplified check
			mas.byzantineDetector.ValidateSignature(fmt.Sprintf("%s-%s-%d", senderID, messageType, time.Now().UnixNano()), valid)
		}
	}

	// Validate message structure
	if config.ValidateAllMessages {
		if message == nil {
			mas.reputationSystem.RecordMessageFailure(senderID)
			return fmt.Errorf("nil message")
		}
	}

	// Check Byzantine detector
	if config.EnableByzantineDetection && mas.byzantineDetector != nil {
		if mas.byzantineDetector.IsByzantine(senderID) {
			return fmt.Errorf("sender is confirmed Byzantine: %s", senderID)
		}
	}

	return nil
}

// validateHybridMessage validates message in hybrid mode (adaptive)
func (mas *ModeAwareSecurity) validateHybridMessage(senderID string, messageType string, message interface{}, signature string) error {
	// Use network trust to determine validation level
	if mas.networkTrust > mas.hybridConfig.TrustThreshold {
		return mas.validateDatacenterMessage(senderID, messageType, message)
	}
	return mas.validateInternetMessage(senderID, messageType, message, signature)
}

// GetTLSConfig returns TLS configuration for current mode
func (mas *ModeAwareSecurity) GetTLSConfig() (*tls.Config, error) {
	mas.mu.RLock()
	defer mas.mu.RUnlock()

	if mas.currentMode == ModeDatacenter {
		return nil, fmt.Errorf("TLS not required in datacenter mode")
	}

	if mas.tlsConfig == nil {
		return nil, fmt.Errorf("TLS not initialized")
	}

	return mas.tlsConfig, nil
}

// initializeTLS initializes TLS configuration
func (mas *ModeAwareSecurity) initializeTLS() error {
	config := mas.internetConfig

	// Create certificate manager
	mas.certManager = NewCertificateManager(mas.nodeID, mas.logger)

	// Generate self-signed cert (in production, use proper CA)
	cert, err := mas.certManager.GenerateSelfSignedCert(config.CertValidityPeriod)
	if err != nil {
		return fmt.Errorf("failed to generate certificate: %w", err)
	}

	// Create TLS config
	mas.tlsConfig = &tls.Config{
		Certificates: []tls.Certificate{*cert},
		MinVersion:   config.MinTLSVersion,
		CipherSuites: config.AllowedCipherSuites,
	}

	if config.RequireMutualTLS {
		mas.tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
		if mas.certManager.caCertPool != nil {
			mas.tlsConfig.ClientCAs = mas.certManager.caCertPool
		}
	}

	mas.tlsEnabled = true

	mas.logger.Info("TLS initialized",
		zap.String("node_id", mas.nodeID),
		zap.Bool("mutual_tls", config.RequireMutualTLS),
		zap.String("min_version", "TLS 1.3"),
	)

	return nil
}

// configureDatacenterMode configures for datacenter mode
func (mas *ModeAwareSecurity) configureDatacenterMode() {
	mas.tlsEnabled = false

	// Disable intensive security checks
	if mas.datacenterConfig.SkipByzantineDetection && mas.byzantineDetector != nil {
		// Byzantine detector stays running but in passive mode
	}

	mas.logger.Info("Configured for datacenter mode",
		zap.String("node_id", mas.nodeID),
		zap.Bool("tls_enabled", false),
		zap.Bool("byzantine_detection", !mas.datacenterConfig.SkipByzantineDetection),
	)
}

// configureInternetMode configures for internet mode
func (mas *ModeAwareSecurity) configureInternetMode() {
	if !mas.tlsEnabled {
		mas.initializeTLS()
	}

	mas.logger.Info("Configured for internet mode",
		zap.String("node_id", mas.nodeID),
		zap.Bool("tls_enabled", true),
		zap.Bool("byzantine_detection", true),
		zap.Bool("reputation_system", true),
	)
}

// configureHybridMode configures for hybrid mode
func (mas *ModeAwareSecurity) configureHybridMode() {
	// Initialize TLS but don't enforce yet
	if !mas.tlsEnabled {
		mas.initializeTLS()
	}

	go mas.adaptiveMonitoring()

	mas.logger.Info("Configured for hybrid mode",
		zap.String("node_id", mas.nodeID),
		zap.Float64("network_trust", mas.networkTrust),
		zap.Float64("trust_threshold", mas.hybridConfig.TrustThreshold),
	)
}

// adaptiveMonitoring monitors network conditions and adjusts security
func (mas *ModeAwareSecurity) adaptiveMonitoring() {
	ticker := time.NewTicker(mas.hybridConfig.AdaptiveCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-mas.ctx.Done():
			return
		case <-ticker.C:
			mas.updateNetworkTrust()
			mas.adjustSecurityLevel()
		}
	}
}

// updateNetworkTrust updates network trust based on reputation and Byzantine detection
func (mas *ModeAwareSecurity) updateNetworkTrust() {
	if mas.reputationSystem == nil {
		return
	}

	stats := mas.reputationSystem.GetStats()
	totalNodes := stats["total_nodes"].(int)
	if totalNodes == 0 {
		return
	}

	trustedNodes := stats["trusted_nodes"].(int)
	quarantinedNodes := stats["quarantined"].(int)

	// Calculate trust ratio
	trustRatio := float64(trustedNodes) / float64(totalNodes)
	quarantineRatio := float64(quarantinedNodes) / float64(totalNodes)

	// Network trust = trust ratio - quarantine penalty
	mas.networkTrust = trustRatio - (quarantineRatio * 2.0)
	if mas.networkTrust < 0 {
		mas.networkTrust = 0
	}
	if mas.networkTrust > 1 {
		mas.networkTrust = 1
	}

	mas.logger.Debug("Network trust updated",
		zap.Float64("trust", mas.networkTrust),
		zap.Int("trusted", trustedNodes),
		zap.Int("quarantined", quarantinedNodes),
		zap.Int("total", totalNodes),
	)
}

// adjustSecurityLevel adjusts security based on network trust
func (mas *ModeAwareSecurity) adjustSecurityLevel() {
	mas.mu.Lock()
	defer mas.mu.Unlock()

	if mas.currentMode != ModeHybrid {
		return
	}

	config := mas.hybridConfig

	// Switch to datacenter mode if trust is high
	if mas.networkTrust > config.TrustThreshold {
		if time.Since(mas.lastModeChange) > config.MonitoringWindow {
			mas.SwitchMode(ModeDatacenter, fmt.Sprintf("High network trust: %.2f", mas.networkTrust))
		}
	}

	// Switch to internet mode if trust is low
	if mas.networkTrust < config.UntrustThreshold {
		if time.Since(mas.lastModeChange) > config.MonitoringWindow {
			mas.SwitchMode(ModeInternet, fmt.Sprintf("Low network trust: %.2f", mas.networkTrust))
		}
	}
}

// modeString returns string representation of mode
func (mas *ModeAwareSecurity) modeString(mode SecurityMode) string {
	switch mode {
	case ModeDatacenter:
		return "datacenter"
	case ModeInternet:
		return "internet"
	case ModeHybrid:
		return "hybrid"
	default:
		return "unknown"
	}
}

// Stop stops the mode-aware security system
func (mas *ModeAwareSecurity) Stop() {
	mas.cancel()
	if mas.certManager != nil {
		mas.certManager.Stop()
	}
}

// GetStats returns security statistics
func (mas *ModeAwareSecurity) GetStats() map[string]interface{} {
	mas.mu.RLock()
	defer mas.mu.RUnlock()

	return map[string]interface{}{
		"current_mode":    mas.modeString(mas.currentMode),
		"tls_enabled":     mas.tlsEnabled,
		"network_trust":   mas.networkTrust,
		"last_mode_change": mas.lastModeChange,
	}
}

// NewCertificateManager creates a new certificate manager
func NewCertificateManager(nodeID string, logger *zap.Logger) *CertificateManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &CertificateManager{
		nodeID:           nodeID,
		logger:           logger,
		clientCerts:      make(map[string]*x509.Certificate),
		rotationInterval: 30 * 24 * time.Hour, // 30 days
		autoRotate:       true,
		strictValidation: true,
		ctx:              ctx,
		cancel:           cancel,
	}
}

// GenerateSelfSignedCert generates a self-signed certificate
func (cm *CertificateManager) GenerateSelfSignedCert(validity time.Duration) (*tls.Certificate, error) {
	// In production, use proper certificate generation
	// This is a placeholder for the actual implementation

	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.certValidUntil = time.Now().Add(validity)

	cm.logger.Info("Certificate generated",
		zap.String("node_id", cm.nodeID),
		zap.Time("valid_until", cm.certValidUntil),
	)

	// Return placeholder cert
	// In production, generate actual certificate using crypto/x509
	cert := &tls.Certificate{}
	cm.serverCert = cert

	return cert, nil
}

// RotateCertificate rotates the node's certificate
func (cm *CertificateManager) RotateCertificate() error {
	cm.logger.Info("Rotating certificate", zap.String("node_id", cm.nodeID))

	// Generate new certificate
	newCert, err := cm.GenerateSelfSignedCert(90 * 24 * time.Hour)
	if err != nil {
		return fmt.Errorf("failed to generate new certificate: %w", err)
	}

	cm.mu.Lock()
	cm.serverCert = newCert
	cm.mu.Unlock()

	return nil
}

// Stop stops the certificate manager
func (cm *CertificateManager) Stop() {
	cm.cancel()
}
