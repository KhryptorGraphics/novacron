package security

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"go.uber.org/zap"
	"golang.org/x/crypto/acme"
	"golang.org/x/crypto/acme/autocert"
)

// ACMEConfig holds ACME/Let's Encrypt configuration
type ACMEConfig struct {
	Domains           []string
	Email             string
	CacheDir          string
	RenewBefore       time.Duration
	DirectoryURL      string // Optional: custom ACME directory URL
	UseStaging        bool   // Use Let's Encrypt staging environment
	HTTPChallengePort int    // Port for HTTP-01 challenge (default: 80)
}

// ACMEManager manages ACME certificate operations
type ACMEManager struct {
	config      ACMEConfig
	manager     *autocert.Manager
	client      *acme.Client
	logger      *zap.Logger
	certManager *CertificateManager
	ctx         context.Context
	cancel      context.CancelFunc
}

// NewACMEManager creates a new ACME manager
func NewACMEManager(config ACMEConfig, logger *zap.Logger) (*ACMEManager, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	if len(config.Domains) == 0 {
		return nil, errors.New("at least one domain must be specified")
	}

	if config.Email == "" {
		return nil, errors.New("email must be specified for ACME")
	}

	if config.CacheDir == "" {
		config.CacheDir = "/var/lib/dwcp/acme-cache"
	}

	if config.RenewBefore == 0 {
		config.RenewBefore = 30 * 24 * time.Hour // 30 days
	}

	// Create cache directory
	if err := os.MkdirAll(config.CacheDir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create cache directory: %w", err)
	}

	// Create autocert manager
	manager := &autocert.Manager{
		Prompt:      autocert.AcceptTOS,
		HostPolicy:  autocert.HostWhitelist(config.Domains...),
		Cache:       autocert.DirCache(config.CacheDir),
		Email:       config.Email,
		RenewBefore: config.RenewBefore,
	}

	// Use staging environment if specified
	if config.UseStaging {
		manager.Client = &acme.Client{
			DirectoryURL: "https://acme-staging-v02.api.letsencrypt.org/directory",
		}
		logger.Info("Using Let's Encrypt staging environment")
	}

	// Use custom directory URL if specified
	if config.DirectoryURL != "" {
		manager.Client = &acme.Client{
			DirectoryURL: config.DirectoryURL,
		}
		logger.Info("Using custom ACME directory",
			zap.String("url", config.DirectoryURL))
	}

	ctx, cancel := context.WithCancel(context.Background())

	am := &ACMEManager{
		config:  config,
		manager: manager,
		logger:  logger,
		ctx:     ctx,
		cancel:  cancel,
	}

	logger.Info("ACME manager initialized",
		zap.Strings("domains", config.Domains),
		zap.String("email", config.Email),
		zap.String("cache_dir", config.CacheDir),
		zap.Duration("renew_before", config.RenewBefore))

	return am, nil
}

// GetTLSConfig returns TLS configuration with ACME certificate management
func (am *ACMEManager) GetTLSConfig() *tls.Config {
	tlsConfig := am.manager.TLSConfig()

	// Enforce TLS 1.3
	tlsConfig.MinVersion = tls.VersionTLS13
	tlsConfig.MaxVersion = tls.VersionTLS13
	tlsConfig.CipherSuites = []uint16{
		tls.TLS_AES_256_GCM_SHA384,
		tls.TLS_AES_128_GCM_SHA256,
		tls.TLS_CHACHA20_POLY1305_SHA256,
	}

	am.logger.Info("TLS configuration created with ACME certificate management")

	return tlsConfig
}

// GetCertificate gets a certificate for the specified domain
func (am *ACMEManager) GetCertificate(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
	cert, err := am.manager.GetCertificate(hello)
	if err != nil {
		am.logger.Error("Failed to get certificate",
			zap.String("server_name", hello.ServerName),
			zap.Error(err))
		return nil, err
	}

	// Parse certificate to log info
	if len(cert.Certificate) > 0 {
		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err == nil {
			am.logger.Debug("Certificate obtained",
				zap.String("subject", x509Cert.Subject.CommonName),
				zap.Time("not_after", x509Cert.NotAfter),
				zap.Strings("dns_names", x509Cert.DNSNames))
		}
	}

	return cert, nil
}

// ObtainCertificate explicitly obtains a certificate for a domain
func (am *ACMEManager) ObtainCertificate(domain string) error {
	am.logger.Info("Obtaining certificate",
		zap.String("domain", domain))

	// Create TLS ClientHello for the domain
	hello := &tls.ClientHelloInfo{
		ServerName: domain,
	}

	cert, err := am.manager.GetCertificate(hello)
	if err != nil {
		return fmt.Errorf("failed to obtain certificate: %w", err)
	}

	// Parse and log certificate info
	if len(cert.Certificate) > 0 {
		x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
		if err == nil {
			am.logger.Info("Certificate obtained successfully",
				zap.String("domain", domain),
				zap.String("subject", x509Cert.Subject.CommonName),
				zap.Time("not_after", x509Cert.NotAfter),
				zap.Duration("valid_for", time.Until(x509Cert.NotAfter)))
		}
	}

	return nil
}

// RenewCertificate forces renewal of certificate for a domain
func (am *ACMEManager) RenewCertificate(domain string) error {
	am.logger.Info("Forcing certificate renewal",
		zap.String("domain", domain))

	// Delete cached certificate to force renewal
	cacheDir := am.config.CacheDir
	certFile := filepath.Join(cacheDir, domain)

	if err := os.Remove(certFile); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove cached certificate: %w", err)
	}

	// Obtain new certificate
	return am.ObtainCertificate(domain)
}

// GetCachedCertificates returns list of cached certificates
func (am *ACMEManager) GetCachedCertificates() ([]string, error) {
	entries, err := os.ReadDir(am.config.CacheDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read cache directory: %w", err)
	}

	domains := []string{}
	for _, entry := range entries {
		if !entry.IsDir() {
			domains = append(domains, entry.Name())
		}
	}

	return domains, nil
}

// CheckCertificateExpiry checks expiration of all cached certificates
func (am *ACMEManager) CheckCertificateExpiry() (map[string]time.Time, error) {
	domains, err := am.GetCachedCertificates()
	if err != nil {
		return nil, err
	}

	expiry := make(map[string]time.Time)

	for _, domain := range domains {
		// Load certificate
		certPath := filepath.Join(am.config.CacheDir, domain)
		certPEM, err := os.ReadFile(certPath)
		if err != nil {
			am.logger.Warn("Failed to read certificate",
				zap.String("domain", domain),
				zap.Error(err))
			continue
		}

		// Parse certificate (simplified - real implementation needs proper PEM parsing)
		// This is a placeholder - actual implementation would use proper PEM decoding
		_ = certPEM

		// For now, mark as needing check
		expiry[domain] = time.Now()
	}

	return expiry, nil
}

// StartAutoRenewal starts automatic certificate renewal monitoring
func (am *ACMEManager) StartAutoRenewal() {
	go am.renewalLoop()
	am.logger.Info("Auto-renewal monitoring started")
}

// renewalLoop periodically checks and renews certificates
func (am *ACMEManager) renewalLoop() {
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			am.checkAndRenewAll()
		case <-am.ctx.Done():
			return
		}
	}
}

// checkAndRenewAll checks and renews all certificates if needed
func (am *ACMEManager) checkAndRenewAll() {
	expiry, err := am.CheckCertificateExpiry()
	if err != nil {
		am.logger.Error("Failed to check certificate expiry", zap.Error(err))
		return
	}

	for domain, expiryTime := range expiry {
		if time.Until(expiryTime) < am.config.RenewBefore {
			am.logger.Info("Certificate needs renewal",
				zap.String("domain", domain),
				zap.Time("expires", expiryTime))

			if err := am.RenewCertificate(domain); err != nil {
				am.logger.Error("Failed to renew certificate",
					zap.String("domain", domain),
					zap.Error(err))
			}
		}
	}
}

// HTTPChallengeHandler returns HTTP handler for HTTP-01 challenge
func (am *ACMEManager) HTTPChallengeHandler() func(*tls.ClientHelloInfo) (*tls.Certificate, error) {
	return am.manager.GetCertificate
}

// SetCertificateManager links ACME manager with certificate manager
func (am *ACMEManager) SetCertificateManager(cm *CertificateManager) {
	am.certManager = cm
	am.logger.Info("Certificate manager linked to ACME manager")
}

// Stop stops the ACME manager
func (am *ACMEManager) Stop() {
	am.cancel()
	am.logger.Info("ACME manager stopped")
}
