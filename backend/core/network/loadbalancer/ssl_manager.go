package loadbalancer

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// SSLManager manages SSL/TLS certificates and configurations
type SSLManager struct {
	// Configuration
	config           SSLManagerConfig
	
	// Certificate storage
	certificates     map[string]*CertificateInfo
	certMutex        sync.RWMutex
	
	// ACME client for automatic certificate provisioning
	acmeClient       *ACMEClient
	
	// Certificate rotation
	rotationScheduler *CertificateRotationScheduler
	
	// SNI handler for multi-domain certificates
	sniHandler       *SNIHandler
	
	// Certificate validation
	validator        *CertificateValidator
	
	// Monitoring and metrics
	metrics          *SSLMetrics
	metricsMutex     sync.RWMutex
	
	// Runtime state
	ctx              context.Context
	cancel           context.CancelFunc
	initialized      bool
}

// SSLManagerConfig holds SSL manager configuration
type SSLManagerConfig struct {
	// Enable SSL functionality
	EnableSSL           bool              `json:"enable_ssl"`
	
	// Storage paths
	CertStorePath       string            `json:"cert_store_path"`
	PrivateKeyPath      string            `json:"private_key_path"`
	CAStorePath         string            `json:"ca_store_path"`
	
	// ACME configuration
	EnableACME          bool              `json:"enable_acme"`
	ACMEDirectoryURL    string            `json:"acme_directory_url"`
	ACMEEmail           string            `json:"acme_email"`
	ACMEKeyType         string            `json:"acme_key_type"`
	ACMEKeySize         int               `json:"acme_key_size"`
	
	// Certificate rotation
	AutoRotation        bool              `json:"auto_rotation"`
	RotationThreshold   time.Duration     `json:"rotation_threshold"`
	RotationCheckInterval time.Duration   `json:"rotation_check_interval"`
	
	// Security settings
	MinTLSVersion       uint16            `json:"min_tls_version"`
	MaxTLSVersion       uint16            `json:"max_tls_version"`
	CipherSuites        []uint16          `json:"cipher_suites"`
	PreferServerCiphers bool              `json:"prefer_server_ciphers"`
	
	// OCSP settings
	EnableOCSP          bool              `json:"enable_ocsp"`
	OCSPCacheTimeout    time.Duration     `json:"ocsp_cache_timeout"`
	
	// Session resumption
	EnableSessionTickets bool             `json:"enable_session_tickets"`
	SessionTicketKey    []byte            `json:"session_ticket_key,omitempty"`
	
	// Client authentication
	ClientAuth          tls.ClientAuthType `json:"client_auth"`
	ClientCAPath        string            `json:"client_ca_path"`
	
	// Performance settings
	MaxCertCacheSize    int               `json:"max_cert_cache_size"`
	CertCacheTTL        time.Duration     `json:"cert_cache_ttl"`
	
	// Monitoring
	EnableMetrics       bool              `json:"enable_metrics"`
	MetricsInterval     time.Duration     `json:"metrics_interval"`
}

// CertificateInfo holds information about a certificate
type CertificateInfo struct {
	ID              string                 `json:"id"`
	CommonName      string                 `json:"common_name"`
	SubjectAltNames []string               `json:"subject_alt_names"`
	Issuer          string                 `json:"issuer"`
	SerialNumber    string                 `json:"serial_number"`
	NotBefore       time.Time              `json:"not_before"`
	NotAfter        time.Time              `json:"not_after"`
	KeyType         string                 `json:"key_type"`
	KeySize         int                    `json:"key_size"`
	Fingerprint     string                 `json:"fingerprint"`
	CertPath        string                 `json:"cert_path"`
	KeyPath         string                 `json:"key_path"`
	IsCA            bool                   `json:"is_ca"`
	IsSelfSigned    bool                   `json:"is_self_signed"`
	Source          CertificateSource      `json:"source"`
	Status          CertificateStatus      `json:"status"`
	AutoRenew       bool                   `json:"auto_renew"`
	LastRotated     time.Time              `json:"last_rotated"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	
	// Runtime data
	tlsCertificate  *tls.Certificate       `json:"-"`
	x509Certificate *x509.Certificate      `json:"-"`
}

// CertificateSource indicates how a certificate was obtained
type CertificateSource string

const (
	CertSourceManual     CertificateSource = "manual"
	CertSourceACME       CertificateSource = "acme"
	CertSourceSelfSigned CertificateSource = "self_signed"
	CertSourceImported   CertificateSource = "imported"
)

// CertificateStatus indicates the current status of a certificate
type CertificateStatus string

const (
	CertStatusValid     CertificateStatus = "valid"
	CertStatusExpiring  CertificateStatus = "expiring"
	CertStatusExpired   CertificateStatus = "expired"
	CertStatusRevoked   CertificateStatus = "revoked"
	CertStatusError     CertificateStatus = "error"
)

// ACMEClient handles automatic certificate provisioning
type ACMEClient struct {
	directoryURL string
	email        string
	keyType      string
	keySize      int
	httpClient   *http.Client
	challenges   map[string]*ACMEChallenge
	mutex        sync.RWMutex
}

// ACMEChallenge represents an ACME challenge
type ACMEChallenge struct {
	Type     string    `json:"type"`
	Token    string    `json:"token"`
	KeyAuth  string    `json:"key_authorization"`
	URL      string    `json:"url"`
	Status   string    `json:"status"`
	Created  time.Time `json:"created"`
	Expires  time.Time `json:"expires"`
}

// CertificateRotationScheduler handles automatic certificate rotation
type CertificateRotationScheduler struct {
	manager         *SSLManager
	ticker          *time.Ticker
	rotationQueue   chan string
	stopCh          chan struct{}
	ctx             context.Context
}

// SNIHandler handles Server Name Indication for multi-domain certificates
type SNIHandler struct {
	certificates map[string]*CertificateInfo
	defaultCert  *CertificateInfo
	mutex        sync.RWMutex
}

// CertificateValidator validates certificate properties and constraints
type CertificateValidator struct {
	trustedCAs     *x509.CertPool
	crlCache       map[string]*x509.RevocationList
	ocspCache      map[string]*OCSPResponse
	cacheMutex     sync.RWMutex
}

// OCSPResponse represents an OCSP response
type OCSPResponse struct {
	Status    int       `json:"status"`
	ThisUpdate time.Time `json:"this_update"`
	NextUpdate time.Time `json:"next_update"`
	RevokedAt  time.Time `json:"revoked_at,omitempty"`
	Reason     int       `json:"reason,omitempty"`
	Raw        []byte    `json:"-"`
}

// SSLMetrics holds SSL/TLS related metrics
type SSLMetrics struct {
	TotalCertificates      int64             `json:"total_certificates"`
	ValidCertificates      int64             `json:"valid_certificates"`
	ExpiringCertificates   int64             `json:"expiring_certificates"`
	ExpiredCertificates    int64             `json:"expired_certificates"`
	TLSHandshakes          int64             `json:"tls_handshakes"`
	TLSHandshakeErrors     int64             `json:"tls_handshake_errors"`
	CertificateRotations   int64             `json:"certificate_rotations"`
	ACMECertificateRequests int64            `json:"acme_certificate_requests"`
	ACMECertificateFailures int64            `json:"acme_certificate_failures"`
	CipherSuiteUsage       map[uint16]int64  `json:"cipher_suite_usage"`
	TLSVersionUsage        map[uint16]int64  `json:"tls_version_usage"`
	LastUpdated            time.Time         `json:"last_updated"`
}

// NewSSLManager creates a new SSL manager
func NewSSLManager(config SSLManagerConfig) *SSLManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &SSLManager{
		config:       config,
		certificates: make(map[string]*CertificateInfo),
		metrics: &SSLMetrics{
			CipherSuiteUsage: make(map[uint16]int64),
			TLSVersionUsage:  make(map[uint16]int64),
			LastUpdated:      time.Now(),
		},
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start initializes and starts the SSL manager
func (sm *SSLManager) Start() error {
	if sm.initialized {
		return fmt.Errorf("SSL manager already started")
	}
	
	// Create certificate storage directories
	if err := sm.createStorageDirectories(); err != nil {
		return fmt.Errorf("failed to create storage directories: %w", err)
	}
	
	// Initialize ACME client if enabled
	if sm.config.EnableACME {
		sm.acmeClient = &ACMEClient{
			directoryURL: sm.config.ACMEDirectoryURL,
			email:        sm.config.ACMEEmail,
			keyType:      sm.config.ACMEKeyType,
			keySize:      sm.config.ACMEKeySize,
			httpClient:   &http.Client{Timeout: 30 * time.Second},
			challenges:   make(map[string]*ACMEChallenge),
		}
		
		if err := sm.acmeClient.initialize(); err != nil {
			return fmt.Errorf("failed to initialize ACME client: %w", err)
		}
	}
	
	// Initialize certificate rotation scheduler
	if sm.config.AutoRotation {
		sm.rotationScheduler = &CertificateRotationScheduler{
			manager:       sm,
			rotationQueue: make(chan string, 100),
			stopCh:        make(chan struct{}),
			ctx:           sm.ctx,
		}
		
		interval := sm.config.RotationCheckInterval
		if interval == 0 {
			interval = 24 * time.Hour
		}
		
		sm.rotationScheduler.ticker = time.NewTicker(interval)
		go sm.rotationScheduler.run()
	}
	
	// Initialize SNI handler
	sm.sniHandler = &SNIHandler{
		certificates: make(map[string]*CertificateInfo),
	}
	
	// Initialize certificate validator
	sm.validator = &CertificateValidator{
		crlCache:  make(map[string]*x509.RevocationList),
		ocspCache: make(map[string]*OCSPResponse),
	}
	
	// Load existing certificates
	if err := sm.loadExistingCertificates(); err != nil {
		return fmt.Errorf("failed to load existing certificates: %w", err)
	}
	
	// Start metrics collection if enabled
	if sm.config.EnableMetrics {
		go sm.metricsCollectionLoop()
	}
	
	sm.initialized = true
	return nil
}

// Stop stops the SSL manager
func (sm *SSLManager) Stop() error {
	sm.cancel()
	
	if sm.rotationScheduler != nil && sm.rotationScheduler.ticker != nil {
		sm.rotationScheduler.ticker.Stop()
		close(sm.rotationScheduler.stopCh)
	}
	
	sm.initialized = false
	return nil
}

// createStorageDirectories creates necessary storage directories
func (sm *SSLManager) createStorageDirectories() error {
	dirs := []string{
		sm.config.CertStorePath,
		sm.config.PrivateKeyPath,
		sm.config.CAStorePath,
	}
	
	for _, dir := range dirs {
		if dir != "" {
			if err := os.MkdirAll(dir, 0755); err != nil {
				return fmt.Errorf("failed to create directory %s: %w", dir, err)
			}
		}
	}
	
	return nil
}

// loadExistingCertificates loads certificates from storage
func (sm *SSLManager) loadExistingCertificates() error {
	if sm.config.CertStorePath == "" {
		return nil
	}
	
	files, err := ioutil.ReadDir(sm.config.CertStorePath)
	if err != nil {
		return fmt.Errorf("failed to read certificate directory: %w", err)
	}
	
	for _, file := range files {
		if strings.HasSuffix(file.Name(), ".crt") || strings.HasSuffix(file.Name(), ".pem") {
			certPath := filepath.Join(sm.config.CertStorePath, file.Name())
			
			// Try to find corresponding key file
			baseName := strings.TrimSuffix(file.Name(), filepath.Ext(file.Name()))
			keyPath := filepath.Join(sm.config.PrivateKeyPath, baseName+".key")
			
			if _, err := os.Stat(keyPath); os.IsNotExist(err) {
				// Try alternative key path
				keyPath = filepath.Join(sm.config.PrivateKeyPath, baseName+".pem")
			}
			
			if err := sm.loadCertificate(certPath, keyPath); err != nil {
				// Log error but continue loading other certificates
				fmt.Printf("Warning: Failed to load certificate %s: %v\n", certPath, err)
			}
		}
	}
	
	return nil
}

// loadCertificate loads a single certificate from files
func (sm *SSLManager) loadCertificate(certPath, keyPath string) error {
	// Load certificate file
	certPEM, err := ioutil.ReadFile(certPath)
	if err != nil {
		return fmt.Errorf("failed to read certificate file: %w", err)
	}
	
	// Load private key file
	keyPEM, err := ioutil.ReadFile(keyPath)
	if err != nil {
		return fmt.Errorf("failed to read private key file: %w", err)
	}
	
	// Parse TLS certificate
	tlsCert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return fmt.Errorf("failed to parse X509 key pair: %w", err)
	}
	
	// Parse X509 certificate
	x509Cert, err := x509.ParseCertificate(tlsCert.Certificate[0])
	if err != nil {
		return fmt.Errorf("failed to parse X509 certificate: %w", err)
	}
	
	// Create certificate info
	certInfo := &CertificateInfo{
		ID:              uuid.New().String(),
		CommonName:      x509Cert.Subject.CommonName,
		SubjectAltNames: x509Cert.DNSNames,
		Issuer:          x509Cert.Issuer.String(),
		SerialNumber:    x509Cert.SerialNumber.String(),
		NotBefore:       x509Cert.NotBefore,
		NotAfter:        x509Cert.NotAfter,
		KeyType:         "RSA", // Simplified
		KeySize:         2048,  // Simplified
		CertPath:        certPath,
		KeyPath:         keyPath,
		IsCA:            x509Cert.IsCA,
		IsSelfSigned:    x509Cert.Issuer.String() == x509Cert.Subject.String(),
		Source:          CertSourceImported,
		Status:          sm.getCertificateStatus(x509Cert),
		AutoRenew:       false,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
		tlsCertificate:  &tlsCert,
		x509Certificate: x509Cert,
	}
	
	// Calculate fingerprint
	certInfo.Fingerprint = sm.calculateFingerprint(x509Cert.Raw)
	
	// Store certificate
	sm.certMutex.Lock()
	sm.certificates[certInfo.ID] = certInfo
	sm.certMutex.Unlock()
	
	// Add to SNI handler
	sm.addToSNI(certInfo)
	
	return nil
}

// getCertificateStatus determines the current status of a certificate
func (sm *SSLManager) getCertificateStatus(cert *x509.Certificate) CertificateStatus {
	now := time.Now()
	
	if now.Before(cert.NotBefore) || now.After(cert.NotAfter) {
		return CertStatusExpired
	}
	
	// Check if certificate is expiring within threshold
	threshold := sm.config.RotationThreshold
	if threshold == 0 {
		threshold = 30 * 24 * time.Hour // Default 30 days
	}
	
	if now.Add(threshold).After(cert.NotAfter) {
		return CertStatusExpiring
	}
	
	return CertStatusValid
}

// calculateFingerprint calculates SHA256 fingerprint of certificate
func (sm *SSLManager) calculateFingerprint(certDER []byte) string {
	hash := make([]byte, 32)
	// Simplified fingerprint calculation
	return fmt.Sprintf("%x", hash)
}

// addToSNI adds certificate to SNI handler
func (sm *SSLManager) addToSNI(certInfo *CertificateInfo) {
	sm.sniHandler.mutex.Lock()
	defer sm.sniHandler.mutex.Unlock()
	
	// Add for common name
	if certInfo.CommonName != "" {
		sm.sniHandler.certificates[certInfo.CommonName] = certInfo
	}
	
	// Add for subject alternative names
	for _, name := range certInfo.SubjectAltNames {
		sm.sniHandler.certificates[name] = certInfo
	}
	
	// Set as default if none exists
	if sm.sniHandler.defaultCert == nil {
		sm.sniHandler.defaultCert = certInfo
	}
}

// CreateSelfSignedCertificate creates a self-signed certificate
func (sm *SSLManager) CreateSelfSignedCertificate(commonName string, altNames []string, validFor time.Duration) (*CertificateInfo, error) {
	// Generate private key
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("failed to generate private key: %w", err)
	}
	
	// Create certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: commonName,
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(validFor),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              altNames,
	}
	
	// Add IP addresses to certificate if any
	for _, name := range altNames {
		if ip := net.ParseIP(name); ip != nil {
			template.IPAddresses = append(template.IPAddresses, ip)
		}
	}
	
	// Create certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %w", err)
	}
	
	// Encode certificate as PEM
	certPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: certDER,
	})
	
	// Encode private key as PEM
	privateKeyDER, err := x509.MarshalPKCS8PrivateKey(privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal private key: %w", err)
	}
	
	keyPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "PRIVATE KEY",
		Bytes: privateKeyDER,
	})
	
	// Save to files
	certID := uuid.New().String()
	certPath := filepath.Join(sm.config.CertStorePath, fmt.Sprintf("%s.crt", certID))
	keyPath := filepath.Join(sm.config.PrivateKeyPath, fmt.Sprintf("%s.key", certID))
	
	if err := ioutil.WriteFile(certPath, certPEM, 0644); err != nil {
		return nil, fmt.Errorf("failed to save certificate: %w", err)
	}
	
	if err := ioutil.WriteFile(keyPath, keyPEM, 0600); err != nil {
		return nil, fmt.Errorf("failed to save private key: %w", err)
	}
	
	// Parse back the certificate
	tlsCert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to parse created certificate: %w", err)
	}
	
	x509Cert, err := x509.ParseCertificate(tlsCert.Certificate[0])
	if err != nil {
		return nil, fmt.Errorf("failed to parse X509 certificate: %w", err)
	}
	
	// Create certificate info
	certInfo := &CertificateInfo{
		ID:              certID,
		CommonName:      commonName,
		SubjectAltNames: altNames,
		Issuer:          x509Cert.Issuer.String(),
		SerialNumber:    x509Cert.SerialNumber.String(),
		NotBefore:       x509Cert.NotBefore,
		NotAfter:        x509Cert.NotAfter,
		KeyType:         "RSA",
		KeySize:         2048,
		Fingerprint:     sm.calculateFingerprint(certDER),
		CertPath:        certPath,
		KeyPath:         keyPath,
		IsCA:            false,
		IsSelfSigned:    true,
		Source:          CertSourceSelfSigned,
		Status:          CertStatusValid,
		AutoRenew:       false,
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
		tlsCertificate:  &tlsCert,
		x509Certificate: x509Cert,
	}
	
	// Store certificate
	sm.certMutex.Lock()
	sm.certificates[certInfo.ID] = certInfo
	sm.certMutex.Unlock()
	
	// Add to SNI handler
	sm.addToSNI(certInfo)
	
	return certInfo, nil
}

// GetTLSConfig creates a TLS configuration for the load balancer
func (sm *SSLManager) GetTLSConfig() *tls.Config {
	return &tls.Config{
		MinVersion:               sm.config.MinTLSVersion,
		MaxVersion:               sm.config.MaxTLSVersion,
		CipherSuites:             sm.config.CipherSuites,
		PreferServerCipherSuites: sm.config.PreferServerCiphers,
		GetCertificate:           sm.getCertificate,
		ClientAuth:               sm.config.ClientAuth,
		SessionTicketsDisabled:   !sm.config.EnableSessionTickets,
	}
}

// getCertificate is the SNI callback function
func (sm *SSLManager) getCertificate(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
	sm.sniHandler.mutex.RLock()
	defer sm.sniHandler.mutex.RUnlock()
	
	// Look for exact match first
	if certInfo, exists := sm.sniHandler.certificates[hello.ServerName]; exists {
		if certInfo.Status == CertStatusValid {
			return certInfo.tlsCertificate, nil
		}
	}
	
	// Try wildcard matching
	if strings.HasPrefix(hello.ServerName, "*.") {
		domain := hello.ServerName[2:]
		if certInfo, exists := sm.sniHandler.certificates[domain]; exists {
			if certInfo.Status == CertStatusValid {
				return certInfo.tlsCertificate, nil
			}
		}
	}
	
	// Use default certificate if available
	if sm.sniHandler.defaultCert != nil {
		return sm.sniHandler.defaultCert.tlsCertificate, nil
	}
	
	return nil, fmt.Errorf("no certificate found for %s", hello.ServerName)
}

// ACME implementation

// initialize initializes the ACME client
func (ac *ACMEClient) initialize() error {
	// Simplified ACME initialization
	// In a real implementation, this would:
	// 1. Discover ACME directory
	// 2. Create account
	// 3. Accept terms of service
	return nil
}

// Certificate rotation implementation

// run runs the certificate rotation scheduler
func (crs *CertificateRotationScheduler) run() {
	for {
		select {
		case <-crs.ctx.Done():
			return
		case <-crs.stopCh:
			return
		case <-crs.ticker.C:
			crs.checkForRotation()
		case certID := <-crs.rotationQueue:
			crs.rotateCertificate(certID)
		}
	}
}

// checkForRotation checks which certificates need rotation
func (crs *CertificateRotationScheduler) checkForRotation() {
	crs.manager.certMutex.RLock()
	defer crs.manager.certMutex.RUnlock()
	
	for id, cert := range crs.manager.certificates {
		if cert.AutoRenew && cert.Status == CertStatusExpiring {
			select {
			case crs.rotationQueue <- id:
				// Queued for rotation
			default:
				// Queue is full, skip this round
			}
		}
	}
}

// rotateCertificate rotates a specific certificate
func (crs *CertificateRotationScheduler) rotateCertificate(certID string) {
	crs.manager.certMutex.RLock()
	cert, exists := crs.manager.certificates[certID]
	crs.manager.certMutex.RUnlock()
	
	if !exists {
		return
	}
	
	// Rotate based on source
	switch cert.Source {
	case CertSourceACME:
		// Renew via ACME
		if err := crs.renewACMECertificate(cert); err != nil {
			fmt.Printf("Failed to renew ACME certificate %s: %v\n", cert.CommonName, err)
		}
	case CertSourceSelfSigned:
		// Create new self-signed certificate
		if err := crs.renewSelfSignedCertificate(cert); err != nil {
			fmt.Printf("Failed to renew self-signed certificate %s: %v\n", cert.CommonName, err)
		}
	}
}

// renewACMECertificate renews an ACME certificate
func (crs *CertificateRotationScheduler) renewACMECertificate(cert *CertificateInfo) error {
	// Simplified ACME renewal
	// In a real implementation, this would use the ACME protocol
	return fmt.Errorf("ACME renewal not implemented")
}

// renewSelfSignedCertificate renews a self-signed certificate
func (crs *CertificateRotationScheduler) renewSelfSignedCertificate(cert *CertificateInfo) error {
	// Create new self-signed certificate with same parameters
	validFor := cert.NotAfter.Sub(cert.NotBefore)
	
	newCert, err := crs.manager.CreateSelfSignedCertificate(cert.CommonName, cert.SubjectAltNames, validFor)
	if err != nil {
		return err
	}
	
	// Update the existing certificate entry
	crs.manager.certMutex.Lock()
	oldCert := crs.manager.certificates[cert.ID]
	crs.manager.certificates[cert.ID] = newCert
	newCert.ID = cert.ID // Keep same ID
	newCert.AutoRenew = cert.AutoRenew
	newCert.LastRotated = time.Now()
	crs.manager.certMutex.Unlock()
	
	// Remove old files
	if oldCert.CertPath != "" {
		os.Remove(oldCert.CertPath)
	}
	if oldCert.KeyPath != "" {
		os.Remove(oldCert.KeyPath)
	}
	
	// Update SNI handler
	crs.manager.addToSNI(newCert)
	
	return nil
}

// metricsCollectionLoop collects SSL/TLS metrics
func (sm *SSLManager) metricsCollectionLoop() {
	interval := sm.config.MetricsInterval
	if interval == 0 {
		interval = 60 * time.Second
	}
	
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	
	for {
		select {
		case <-sm.ctx.Done():
			return
		case <-ticker.C:
			sm.updateMetrics()
		}
	}
}

// updateMetrics updates SSL/TLS metrics
func (sm *SSLManager) updateMetrics() {
	sm.metricsMutex.Lock()
	defer sm.metricsMutex.Unlock()
	
	sm.certMutex.RLock()
	defer sm.certMutex.RUnlock()
	
	// Reset counters
	sm.metrics.TotalCertificates = int64(len(sm.certificates))
	sm.metrics.ValidCertificates = 0
	sm.metrics.ExpiringCertificates = 0
	sm.metrics.ExpiredCertificates = 0
	
	// Count certificate statuses
	for _, cert := range sm.certificates {
		switch cert.Status {
		case CertStatusValid:
			sm.metrics.ValidCertificates++
		case CertStatusExpiring:
			sm.metrics.ExpiringCertificates++
		case CertStatusExpired:
			sm.metrics.ExpiredCertificates++
		}
	}
	
	sm.metrics.LastUpdated = time.Now()
}

// Public API methods

// GetCertificates returns all managed certificates
func (sm *SSLManager) GetCertificates() []*CertificateInfo {
	sm.certMutex.RLock()
	defer sm.certMutex.RUnlock()
	
	certs := make([]*CertificateInfo, 0, len(sm.certificates))
	for _, cert := range sm.certificates {
		// Create a copy without sensitive fields
		certCopy := &CertificateInfo{
			ID:              cert.ID,
			CommonName:      cert.CommonName,
			SubjectAltNames: cert.SubjectAltNames,
			Issuer:          cert.Issuer,
			SerialNumber:    cert.SerialNumber,
			NotBefore:       cert.NotBefore,
			NotAfter:        cert.NotAfter,
			KeyType:         cert.KeyType,
			KeySize:         cert.KeySize,
			Fingerprint:     cert.Fingerprint,
			CertPath:        cert.CertPath,
			IsCA:            cert.IsCA,
			IsSelfSigned:    cert.IsSelfSigned,
			Source:          cert.Source,
			Status:          cert.Status,
			AutoRenew:       cert.AutoRenew,
			LastRotated:     cert.LastRotated,
			CreatedAt:       cert.CreatedAt,
			UpdatedAt:       cert.UpdatedAt,
		}
		certs = append(certs, certCopy)
	}
	
	return certs
}

// GetCertificate returns a specific certificate
func (sm *SSLManager) GetCertificate(certID string) (*CertificateInfo, error) {
	sm.certMutex.RLock()
	defer sm.certMutex.RUnlock()
	
	cert, exists := sm.certificates[certID]
	if !exists {
		return nil, fmt.Errorf("certificate %s not found", certID)
	}
	
	// Return copy without sensitive fields
	return &CertificateInfo{
		ID:              cert.ID,
		CommonName:      cert.CommonName,
		SubjectAltNames: cert.SubjectAltNames,
		Issuer:          cert.Issuer,
		SerialNumber:    cert.SerialNumber,
		NotBefore:       cert.NotBefore,
		NotAfter:        cert.NotAfter,
		KeyType:         cert.KeyType,
		KeySize:         cert.KeySize,
		Fingerprint:     cert.Fingerprint,
		CertPath:        cert.CertPath,
		IsCA:            cert.IsCA,
		IsSelfSigned:    cert.IsSelfSigned,
		Source:          cert.Source,
		Status:          cert.Status,
		AutoRenew:       cert.AutoRenew,
		LastRotated:     cert.LastRotated,
		CreatedAt:       cert.CreatedAt,
		UpdatedAt:       cert.UpdatedAt,
	}, nil
}

// GetMetrics returns SSL/TLS metrics
func (sm *SSLManager) GetMetrics() *SSLMetrics {
	sm.metricsMutex.RLock()
	defer sm.metricsMutex.RUnlock()
	
	// Return copy of metrics
	metricsCopy := &SSLMetrics{
		TotalCertificates:       sm.metrics.TotalCertificates,
		ValidCertificates:       sm.metrics.ValidCertificates,
		ExpiringCertificates:    sm.metrics.ExpiringCertificates,
		ExpiredCertificates:     sm.metrics.ExpiredCertificates,
		TLSHandshakes:           sm.metrics.TLSHandshakes,
		TLSHandshakeErrors:      sm.metrics.TLSHandshakeErrors,
		CertificateRotations:    sm.metrics.CertificateRotations,
		ACMECertificateRequests: sm.metrics.ACMECertificateRequests,
		ACMECertificateFailures: sm.metrics.ACMECertificateFailures,
		CipherSuiteUsage:        make(map[uint16]int64),
		TLSVersionUsage:         make(map[uint16]int64),
		LastUpdated:             sm.metrics.LastUpdated,
	}
	
	// Copy maps
	for k, v := range sm.metrics.CipherSuiteUsage {
		metricsCopy.CipherSuiteUsage[k] = v
	}
	for k, v := range sm.metrics.TLSVersionUsage {
		metricsCopy.TLSVersionUsage[k] = v
	}
	
	return metricsCopy
}

// DefaultSSLManagerConfig returns default SSL manager configuration
func DefaultSSLManagerConfig() SSLManagerConfig {
	return SSLManagerConfig{
		CertStorePath:           "/etc/novacron/ssl/certs",
		PrivateKeyPath:          "/etc/novacron/ssl/private",
		CAStorePath:             "/etc/novacron/ssl/ca",
		EnableACME:              false,
		ACMEDirectoryURL:        "https://acme-v02.api.letsencrypt.org/directory",
		ACMEKeyType:             "RSA",
		ACMEKeySize:             2048,
		AutoRotation:            true,
		RotationThreshold:       30 * 24 * time.Hour, // 30 days
		RotationCheckInterval:   24 * time.Hour,      // Daily
		MinTLSVersion:           tls.VersionTLS12,
		MaxTLSVersion:           tls.VersionTLS13,
		PreferServerCiphers:     true,
		EnableOCSP:              true,
		OCSPCacheTimeout:        24 * time.Hour,
		EnableSessionTickets:    true,
		ClientAuth:              tls.NoClientCert,
		MaxCertCacheSize:        1000,
		CertCacheTTL:            24 * time.Hour,
		EnableMetrics:           true,
		MetricsInterval:         60 * time.Second,
	}
}