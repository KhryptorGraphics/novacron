package security

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"sync"
	"time"

	"go.uber.org/zap"
)

// SecurityConfig holds security configuration
type SecurityConfig struct {
	TLSEnabled           bool
	MinVersion           string
	CertFile             string
	KeyFile              string
	CAFile               string
	MTLSEnabled          bool
	RequireClientCert    bool
	VerifyPeer           bool
	SessionCacheSize     int
	CipherSuites         []string
	CurvePreferences     []string
	RenegotiationSupport string
}

// TLSManager manages TLS configuration and certificates
type TLSManager struct {
	config          *tls.Config
	securityConfig  SecurityConfig
	certManager     *CertificateManager
	caPool          *x509.CertPool
	clientCAPool    *x509.CertPool
	mu              sync.RWMutex
	logger          *zap.Logger
	lastReload      time.Time
	reloadCallbacks []func(*tls.Config)
}

// NewTLSManager creates a new TLS manager with secure defaults
func NewTLSManager(config SecurityConfig, logger *zap.Logger) (*TLSManager, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	// Create TLS configuration with TLS 1.3 enforcement
	tlsConfig := &tls.Config{
		MinVersion:               tls.VersionTLS13,
		MaxVersion:               tls.VersionTLS13,
		PreferServerCipherSuites: true,
		CipherSuites: []uint16{
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_AES_128_GCM_SHA256,
			tls.TLS_CHACHA20_POLY1305_SHA256,
		},
		CurvePreferences: []tls.CurveID{
			tls.X25519,
			tls.CurveP384,
			tls.CurveP256,
		},
		SessionTicketsDisabled: false,
		ClientSessionCache:     tls.NewLRUClientSessionCache(config.SessionCacheSize),
		Renegotiation:          tls.RenegotiateNever,
	}

	// Override min version if specified (for testing only)
	if config.MinVersion != "" {
		version, err := parseTLSVersion(config.MinVersion)
		if err != nil {
			logger.Warn("Invalid TLS version specified, using TLS 1.3",
				zap.String("version", config.MinVersion),
				zap.Error(err))
		} else if version < tls.VersionTLS13 {
			logger.Warn("TLS version below 1.3 not recommended for production",
				zap.String("version", config.MinVersion))
			tlsConfig.MinVersion = version
		}
	}

	tm := &TLSManager{
		config:         tlsConfig,
		securityConfig: config,
		logger:         logger,
		lastReload:     time.Now(),
	}

	// Load CA certificates if provided
	if config.CAFile != "" {
		if err := tm.loadCACertificates(config.CAFile); err != nil {
			return nil, fmt.Errorf("failed to load CA certificates: %w", err)
		}
	}

	// Load server certificate if provided
	if config.CertFile != "" && config.KeyFile != "" {
		if err := tm.loadServerCertificate(config.CertFile, config.KeyFile); err != nil {
			return nil, fmt.Errorf("failed to load server certificate: %w", err)
		}
	}

	// Configure mTLS if enabled
	if config.MTLSEnabled {
		if err := tm.configureMTLS(); err != nil {
			return nil, fmt.Errorf("failed to configure mTLS: %w", err)
		}
	}

	// Set up custom certificate verification
	tlsConfig.VerifyPeerCertificate = tm.VerifyPeerCertificate

	// Set up GetCertificate callback for dynamic certificate selection
	tlsConfig.GetCertificate = tm.GetCertificate

	logger.Info("TLS manager initialized",
		zap.String("min_version", tlsVersionName(tlsConfig.MinVersion)),
		zap.Bool("mtls_enabled", config.MTLSEnabled),
		zap.Int("cipher_suites", len(tlsConfig.CipherSuites)))

	return tm, nil
}

// ConfigureMTLS sets up mutual TLS authentication
func (tm *TLSManager) ConfigureMTLS(clientCert, clientKey, caCert string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Load client certificate for client mode
	if clientCert != "" && clientKey != "" {
		cert, err := tls.LoadX509KeyPair(clientCert, clientKey)
		if err != nil {
			return fmt.Errorf("failed to load client certificate: %w", err)
		}
		tm.config.Certificates = []tls.Certificate{cert}
		tm.logger.Info("Client certificate loaded for mTLS",
			zap.String("cert_file", clientCert))
	}

	// Load CA certificate for server verification (client mode)
	if caCert != "" {
		caCertPool := x509.NewCertPool()
		caCertBytes, err := os.ReadFile(caCert)
		if err != nil {
			return fmt.Errorf("failed to read CA certificate: %w", err)
		}
		if !caCertPool.AppendCertsFromPEM(caCertBytes) {
			return errors.New("failed to parse CA certificate")
		}
		tm.config.RootCAs = caCertPool
		tm.caPool = caCertPool
		tm.logger.Info("CA certificate pool configured for server verification")
	}

	// Configure server mode mTLS
	if tm.securityConfig.RequireClientCert {
		tm.config.ClientAuth = tls.RequireAndVerifyClientCert
		if tm.clientCAPool != nil {
			tm.config.ClientCAs = tm.clientCAPool
		}
		tm.logger.Info("mTLS configured: requiring and verifying client certificates")
	} else if tm.securityConfig.MTLSEnabled {
		tm.config.ClientAuth = tls.VerifyClientCertIfGiven
		tm.logger.Info("mTLS configured: verifying client certificates if provided")
	}

	return nil
}

// configureMTLS sets up mTLS from security config
func (tm *TLSManager) configureMTLS() error {
	if tm.securityConfig.RequireClientCert {
		tm.config.ClientAuth = tls.RequireAndVerifyClientCert
	} else {
		tm.config.ClientAuth = tls.VerifyClientCertIfGiven
	}

	if tm.clientCAPool != nil {
		tm.config.ClientCAs = tm.clientCAPool
	}

	tm.logger.Info("mTLS configured from security config",
		zap.Bool("require_client_cert", tm.securityConfig.RequireClientCert))

	return nil
}

// VerifyPeerCertificate performs custom certificate verification
func (tm *TLSManager) VerifyPeerCertificate(rawCerts [][]byte, verifiedChains [][]*x509.Certificate) error {
	if !tm.securityConfig.VerifyPeer {
		return nil
	}

	if len(verifiedChains) == 0 {
		return errors.New("no verified certificate chains")
	}

	cert := verifiedChains[0][0]

	// Check certificate expiration
	now := time.Now()
	if now.Before(cert.NotBefore) {
		tm.logger.Error("Certificate not yet valid",
			zap.Time("not_before", cert.NotBefore),
			zap.Time("now", now))
		return errors.New("certificate not yet valid")
	}
	if now.After(cert.NotAfter) {
		tm.logger.Error("Certificate expired",
			zap.Time("not_after", cert.NotAfter),
			zap.Time("now", now))
		return errors.New("certificate expired")
	}

	// Warn if certificate expires soon
	if time.Until(cert.NotAfter) < 30*24*time.Hour {
		tm.logger.Warn("Certificate expires soon",
			zap.String("subject", cert.Subject.CommonName),
			zap.Time("not_after", cert.NotAfter),
			zap.Duration("time_until_expiry", time.Until(cert.NotAfter)))
	}

	// Check for revocation if certificate manager is available
	if tm.certManager != nil && tm.certManager.IsRevoked(cert) {
		tm.logger.Error("Certificate revoked",
			zap.String("serial", cert.SerialNumber.String()),
			zap.String("subject", cert.Subject.CommonName))
		return errors.New("certificate revoked")
	}

	// Verify organizational unit (optional strict check)
	// Uncomment for production with specific organizational requirements
	/*
	if len(cert.Subject.OrganizationalUnit) == 0 ||
	   !contains(cert.Subject.OrganizationalUnit, "NovaCron") {
		tm.logger.Error("Invalid organizational unit",
			zap.Strings("ou", cert.Subject.OrganizationalUnit))
		return errors.New("invalid organizational unit")
	}
	*/

	tm.logger.Debug("Peer certificate verified",
		zap.String("subject", cert.Subject.CommonName),
		zap.String("issuer", cert.Issuer.CommonName),
		zap.Time("not_after", cert.NotAfter))

	return nil
}

// GetCertificate returns the appropriate certificate for the connection
func (tm *TLSManager) GetCertificate(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	// If certificate manager with rotation is available, use it
	if tm.certManager != nil {
		cert, err := tm.certManager.GetCertificate(hello)
		if err == nil {
			return cert, nil
		}
		tm.logger.Warn("Certificate manager failed, using default certificate",
			zap.Error(err))
	}

	// Return first available certificate
	if len(tm.config.Certificates) > 0 {
		return &tm.config.Certificates[0], nil
	}

	return nil, errors.New("no certificates available")
}

// GetTLSConfig returns the current TLS configuration (thread-safe)
func (tm *TLSManager) GetTLSConfig() *tls.Config {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return tm.config.Clone()
}

// loadServerCertificate loads server certificate and key
func (tm *TLSManager) loadServerCertificate(certFile, keyFile string) error {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return fmt.Errorf("failed to load certificate pair: %w", err)
	}

	// Parse certificate to log information
	if len(cert.Certificate) > 0 {
		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err == nil {
			tm.logger.Info("Server certificate loaded",
				zap.String("subject", x509Cert.Subject.CommonName),
				zap.Time("not_after", x509Cert.NotAfter),
				zap.Strings("dns_names", x509Cert.DNSNames))
		}
	}

	tm.config.Certificates = []tls.Certificate{cert}
	return nil
}

// loadCACertificates loads CA certificates for client verification
func (tm *TLSManager) loadCACertificates(caFile string) error {
	caCertBytes, err := os.ReadFile(caFile)
	if err != nil {
		return fmt.Errorf("failed to read CA file: %w", err)
	}

	caPool := x509.NewCertPool()
	if !caPool.AppendCertsFromPEM(caCertBytes) {
		return errors.New("failed to parse CA certificates")
	}

	tm.clientCAPool = caPool
	tm.logger.Info("CA certificates loaded for client verification",
		zap.String("ca_file", caFile))

	return nil
}

// ReloadCertificates reloads certificates from disk
func (tm *TLSManager) ReloadCertificates() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tm.securityConfig.CertFile != "" && tm.securityConfig.KeyFile != "" {
		cert, err := tls.LoadX509KeyPair(tm.securityConfig.CertFile, tm.securityConfig.KeyFile)
		if err != nil {
			return fmt.Errorf("failed to reload certificate: %w", err)
		}
		tm.config.Certificates = []tls.Certificate{cert}
		tm.lastReload = time.Now()
		tm.logger.Info("Certificates reloaded successfully")

		// Execute reload callbacks
		for _, callback := range tm.reloadCallbacks {
			callback(tm.config)
		}
	}

	return nil
}

// SetCertificateManager sets the certificate manager for advanced features
func (tm *TLSManager) SetCertificateManager(cm *CertificateManager) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.certManager = cm
	tm.logger.Info("Certificate manager attached to TLS manager")
}

// RegisterReloadCallback registers a callback to be called when certificates are reloaded
func (tm *TLSManager) RegisterReloadCallback(callback func(*tls.Config)) {
	tm.mu.Lock()
	defer tm.mu.Unlock()
	tm.reloadCallbacks = append(tm.reloadCallbacks, callback)
}

// GetLastReloadTime returns the last time certificates were reloaded
func (tm *TLSManager) GetLastReloadTime() time.Time {
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	return tm.lastReload
}

// parseTLSVersion converts string version to tls constant
func parseTLSVersion(version string) (uint16, error) {
	switch version {
	case "1.0":
		return tls.VersionTLS10, nil
	case "1.1":
		return tls.VersionTLS11, nil
	case "1.2":
		return tls.VersionTLS12, nil
	case "1.3":
		return tls.VersionTLS13, nil
	default:
		return 0, fmt.Errorf("unsupported TLS version: %s", version)
	}
}

// tlsVersionName returns the name of a TLS version
func tlsVersionName(version uint16) string {
	switch version {
	case tls.VersionTLS10:
		return "TLS 1.0"
	case tls.VersionTLS11:
		return "TLS 1.1"
	case tls.VersionTLS12:
		return "TLS 1.2"
	case tls.VersionTLS13:
		return "TLS 1.3"
	default:
		return fmt.Sprintf("Unknown (0x%04x)", version)
	}
}

// cipherSuiteName returns the name of a cipher suite
func cipherSuiteName(suite uint16) string {
	switch suite {
	case tls.TLS_AES_128_GCM_SHA256:
		return "TLS_AES_128_GCM_SHA256"
	case tls.TLS_AES_256_GCM_SHA384:
		return "TLS_AES_256_GCM_SHA384"
	case tls.TLS_CHACHA20_POLY1305_SHA256:
		return "TLS_CHACHA20_POLY1305_SHA256"
	default:
		return fmt.Sprintf("Unknown (0x%04x)", suite)
	}
}

// contains checks if a string slice contains a value
func contains(slice []string, value string) bool {
	for _, item := range slice {
		if item == value {
			return true
		}
	}
	return false
}

// extractCertSubjects extracts subject names from certificates
func extractCertSubjects(certs []*x509.Certificate) []string {
	subjects := make([]string, len(certs))
	for i, cert := range certs {
		subjects[i] = cert.Subject.CommonName
	}
	return subjects
}
