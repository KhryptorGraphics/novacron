package security

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// MutualTLSManager handles mutual TLS authentication and certificate management
type MutualTLSManager struct {
	config          MutualTLSConfig
	caManager       *CertificateAuthorityManager
	certStore       *CertificateStore
	validator       *CertificateValidator
	revokedCerts    *RevokedCertificateStore
	auditLogger     AuditLogger
	metrics         *MutualTLSMetrics
	mu              sync.RWMutex
}

// MutualTLSConfig configuration for mutual TLS
type MutualTLSConfig struct {
	EnableMutualTLS         bool          `json:"enable_mutual_tls"`
	CAKeyBits               int           `json:"ca_key_bits"`
	ClientCertKeyBits       int           `json:"client_cert_key_bits"`
	ServerCertKeyBits       int           `json:"server_cert_key_bits"`
	CertificateValidityDays int           `json:"certificate_validity_days"`
	AutoRenewalEnabled      bool          `json:"auto_renewal_enabled"`
	RenewalThresholdDays    int           `json:"renewal_threshold_days"`
	RequireClientCerts      bool          `json:"require_client_certs"`
	AllowedClientCNs        []string      `json:"allowed_client_cns"`
	AllowedServerCNs        []string      `json:"allowed_server_cns"`
	CRLDistributionPoints   []string      `json:"crl_distribution_points"`
	OCSPServers            []string      `json:"ocsp_servers"`
	PinningEnabled         bool          `json:"pinning_enabled"`
	PinnedCertificates     []string      `json:"pinned_certificates"`
	TLSVersionMin          uint16        `json:"tls_version_min"`
	TLSVersionMax          uint16        `json:"tls_version_max"`
	CipherSuites           []uint16      `json:"cipher_suites"`
	CurvePreferences       []tls.CurveID `json:"curve_preferences"`
}

// CertificateAuthorityManager manages the certificate authority
type CertificateAuthorityManager struct {
	rootCA         *CertificateAuthority
	intermediateCA *CertificateAuthority
	caChain        []*x509.Certificate
	mu             sync.RWMutex
}

// CertificateAuthority represents a certificate authority
type CertificateAuthority struct {
	ID                string                 `json:"id"`
	Type              CAType                 `json:"type"` // root, intermediate
	CommonName        string                 `json:"common_name"`
	Organization      string                 `json:"organization"`
	Country           string                 `json:"country"`
	ValidFrom         time.Time              `json:"valid_from"`
	ValidTo           time.Time              `json:"valid_to"`
	Certificate       *x509.Certificate      `json:"-"`
	PrivateKey        *rsa.PrivateKey        `json:"-"`
	CertificatePEM    []byte                 `json:"certificate_pem"`
	PrivateKeyPEM     []byte                 `json:"private_key_pem"`
	SerialNumber      *big.Int               `json:"serial_number"`
	IsRevoked         bool                   `json:"is_revoked"`
	RevocationReason  string                 `json:"revocation_reason,omitempty"`
	RevokedAt         *time.Time             `json:"revoked_at,omitempty"`
	IssuedCerts       map[string]*IssuedCert `json:"issued_certs"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// CAType defines certificate authority types
type CAType string

const (
	CATypeRoot         CAType = "root"
	CATypeIntermediate CAType = "intermediate"
)

// IssuedCert tracks certificates issued by the CA
type IssuedCert struct {
	ID           string    `json:"id"`
	SerialNumber *big.Int  `json:"serial_number"`
	CommonName   string    `json:"common_name"`
	ValidFrom    time.Time `json:"valid_from"`
	ValidTo      time.Time `json:"valid_to"`
	IsRevoked    bool      `json:"is_revoked"`
	RevokedAt    *time.Time `json:"revoked_at,omitempty"`
	CertType     CertType  `json:"cert_type"`
}

// CertType defines certificate types
type CertType string

const (
	CertTypeServer CertType = "server"
	CertTypeClient CertType = "client"
	CertTypePeer   CertType = "peer"
)

// CertificateStore manages certificate storage and retrieval
type CertificateStore struct {
	certificates   map[string]*ManagedCertificate
	certsBySubject map[string][]*ManagedCertificate
	certsBySerial  map[string]*ManagedCertificate
	mu             sync.RWMutex
}

// ManagedCertificate represents a managed certificate
type ManagedCertificate struct {
	ID            string                 `json:"id"`
	CommonName    string                 `json:"common_name"`
	SANs          []string               `json:"sans"`
	CertType      CertType               `json:"cert_type"`
	Certificate   *x509.Certificate      `json:"-"`
	PrivateKey    *rsa.PrivateKey        `json:"-"`
	CertPEM       []byte                 `json:"cert_pem"`
	PrivateKeyPEM []byte                 `json:"private_key_pem"`
	ChainPEM      [][]byte               `json:"chain_pem"`
	ValidFrom     time.Time              `json:"valid_from"`
	ValidTo       time.Time              `json:"valid_to"`
	SerialNumber  *big.Int               `json:"serial_number"`
	Fingerprint   string                 `json:"fingerprint"`
	Status        CertificateStatus      `json:"status"`
	IssuerCA      string                 `json:"issuer_ca"`
	AutoRenew     bool                   `json:"auto_renew"`
	UsageCount    int64                  `json:"usage_count"`
	LastUsed      time.Time              `json:"last_used"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
	Tags          []string               `json:"tags"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// CertificateStatus defines certificate status
type CertificateStatus string

const (
	CertStatusActive    CertificateStatus = "active"
	CertStatusExpiring  CertificateStatus = "expiring"
	CertStatusExpired   CertificateStatus = "expired"
	CertStatusRevoked   CertificateStatus = "revoked"
	CertStatusSuspended CertificateStatus = "suspended"
)

// CertificateValidator validates certificates
type CertificateValidator struct {
	rootCAs      *x509.CertPool
	intermediateCAs *x509.CertPool
	crlStore     *CRLStore
	ocspClients  map[string]*OCSPClient
	mu           sync.RWMutex
}

// RevokedCertificateStore manages revoked certificates
type RevokedCertificateStore struct {
	revokedCerts map[string]*RevokedCertificate
	crlData      []byte
	crlLastUpdated time.Time
	mu           sync.RWMutex
}

// RevokedCertificate represents a revoked certificate
type RevokedCertificate struct {
	SerialNumber     *big.Int              `json:"serial_number"`
	RevocationTime   time.Time             `json:"revocation_time"`
	RevocationReason x509.RevocationReason `json:"revocation_reason"`
	Extensions       []x509.Extension      `json:"extensions"`
}

// CRLStore manages Certificate Revocation Lists
type CRLStore struct {
	crls       map[string]*x509.RevocationList
	lastUpdate time.Time
	mu         sync.RWMutex
}

// OCSPClient handles OCSP requests
type OCSPClient struct {
	serverURL string
	client    *http.Client
}

// MutualTLSMetrics tracks mutual TLS metrics
type MutualTLSMetrics struct {
	certificatesIssued    prometheus.Counter
	certificatesRevoked   prometheus.Counter
	tlsHandshakesDuration prometheus.Histogram
	tlsHandshakesTotal    prometheus.Counter
	tlsHandshakesFailed   prometheus.Counter
}

// NewMutualTLSManager creates a new mutual TLS manager
func NewMutualTLSManager(config MutualTLSConfig, auditLogger AuditLogger) (*MutualTLSManager, error) {
	caManager := &CertificateAuthorityManager{
		caChain: make([]*x509.Certificate, 0),
	}

	certStore := &CertificateStore{
		certificates:   make(map[string]*ManagedCertificate),
		certsBySubject: make(map[string][]*ManagedCertificate),
		certsBySerial:  make(map[string]*ManagedCertificate),
	}

	validator := &CertificateValidator{
		rootCAs:         x509.NewCertPool(),
		intermediateCAs: x509.NewCertPool(),
		crlStore:        &CRLStore{crls: make(map[string]*x509.RevocationList)},
		ocspClients:     make(map[string]*OCSPClient),
	}

	revokedCerts := &RevokedCertificateStore{
		revokedCerts: make(map[string]*RevokedCertificate),
	}

	metrics := &MutualTLSMetrics{
		certificatesIssued: promauto.NewCounter(prometheus.CounterOpts{
			Name: "mtls_certificates_issued_total",
			Help: "Total number of certificates issued",
		}),
		certificatesRevoked: promauto.NewCounter(prometheus.CounterOpts{
			Name: "mtls_certificates_revoked_total",
			Help: "Total number of certificates revoked",
		}),
		tlsHandshakesDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name: "mtls_handshakes_duration_seconds",
			Help: "TLS handshake duration in seconds",
		}),
		tlsHandshakesTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "mtls_handshakes_total",
			Help: "Total number of TLS handshakes",
		}),
		tlsHandshakesFailed: promauto.NewCounter(prometheus.CounterOpts{
			Name: "mtls_handshakes_failed_total",
			Help: "Total number of failed TLS handshakes",
		}),
	}

	manager := &MutualTLSManager{
		config:       config,
		caManager:    caManager,
		certStore:    certStore,
		validator:    validator,
		revokedCerts: revokedCerts,
		auditLogger:  auditLogger,
		metrics:      metrics,
	}

	// Initialize CA if needed
	if err := manager.initializeCertificateAuthority(); err != nil {
		return nil, fmt.Errorf("failed to initialize CA: %w", err)
	}

	// Start background processes
	if config.AutoRenewalEnabled {
		go manager.startCertificateRenewalWorker()
	}

	go manager.startCRLUpdateWorker()
	go manager.startMetricsCollector()

	return manager, nil
}

// initializeCertificateAuthority initializes the certificate authority
func (m *MutualTLSManager) initializeCertificateAuthority() error {
	m.caManager.mu.Lock()
	defer m.caManager.mu.Unlock()

	// Check if root CA already exists
	if m.caManager.rootCA != nil {
		return nil
	}

	// Create root CA
	rootCA, err := m.createRootCA()
	if err != nil {
		return fmt.Errorf("failed to create root CA: %w", err)
	}

	m.caManager.rootCA = rootCA

	// Add root CA to validator
	m.validator.rootCAs.AddCert(rootCA.Certificate)

	// Create intermediate CA (optional for higher security)
	intermediateCA, err := m.createIntermediateCA(rootCA)
	if err != nil {
		return fmt.Errorf("failed to create intermediate CA: %w", err)
	}

	m.caManager.intermediateCA = intermediateCA

	// Add intermediate CA to validator
	m.validator.intermediateCAs.AddCert(intermediateCA.Certificate)

	// Build CA chain
	m.caManager.caChain = []*x509.Certificate{
		intermediateCA.Certificate,
		rootCA.Certificate,
	}

	return nil
}

// createRootCA creates a root certificate authority
func (m *MutualTLSManager) createRootCA() (*CertificateAuthority, error) {
	// Generate private key
	privateKey, err := rsa.GenerateKey(rand.Reader, m.config.CAKeyBits)
	if err != nil {
		return nil, fmt.Errorf("failed to generate root CA private key: %w", err)
	}

	// Create certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName:    "NovaCron Root CA",
			Organization:  []string{"NovaCron"},
			Country:       []string{"US"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(10 * 365 * 24 * time.Hour), // 10 years
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign | x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            1,
		MaxPathLenZero:        false,
	}

	// Self-sign the certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create root CA certificate: %w", err)
	}

	// Parse the certificate
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, fmt.Errorf("failed to parse root CA certificate: %w", err)
	}

	// Encode to PEM
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})

	rootCA := &CertificateAuthority{
		ID:             uuid.New().String(),
		Type:           CATypeRoot,
		CommonName:     "NovaCron Root CA",
		Organization:   "NovaCron",
		Country:        "US",
		ValidFrom:      template.NotBefore,
		ValidTo:        template.NotAfter,
		Certificate:    cert,
		PrivateKey:     privateKey,
		CertificatePEM: certPEM,
		PrivateKeyPEM:  keyPEM,
		SerialNumber:   template.SerialNumber,
		IssuedCerts:    make(map[string]*IssuedCert),
		Metadata:       make(map[string]interface{}),
	}

	// Audit log
	m.auditLogger.LogEvent(context.Background(), &AuditEvent{
		EventType: "CERTIFICATE_AUTHORITY_CREATED",
		Actor:     "system",
		Resource:  "root_ca",
		Action:    "CREATE",
		Result:    "SUCCESS",
		Details: map[string]interface{}{
			"ca_id":       rootCA.ID,
			"common_name": rootCA.CommonName,
			"valid_until": rootCA.ValidTo,
		},
	})

	return rootCA, nil
}

// createIntermediateCA creates an intermediate certificate authority
func (m *MutualTLSManager) createIntermediateCA(rootCA *CertificateAuthority) (*CertificateAuthority, error) {
	// Generate private key
	privateKey, err := rsa.GenerateKey(rand.Reader, m.config.CAKeyBits)
	if err != nil {
		return nil, fmt.Errorf("failed to generate intermediate CA private key: %w", err)
	}

	// Create certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			CommonName:    "NovaCron Intermediate CA",
			Organization:  []string{"NovaCron"},
			Country:       []string{"US"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(5 * 365 * 24 * time.Hour), // 5 years
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            0,
		MaxPathLenZero:        true,
	}

	// Sign with root CA
	certDER, err := x509.CreateCertificate(rand.Reader, &template, rootCA.Certificate, &privateKey.PublicKey, rootCA.PrivateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create intermediate CA certificate: %w", err)
	}

	// Parse the certificate
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, fmt.Errorf("failed to parse intermediate CA certificate: %w", err)
	}

	// Encode to PEM
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})

	intermediateCA := &CertificateAuthority{
		ID:             uuid.New().String(),
		Type:           CATypeIntermediate,
		CommonName:     "NovaCron Intermediate CA",
		Organization:   "NovaCron",
		Country:        "US",
		ValidFrom:      template.NotBefore,
		ValidTo:        template.NotAfter,
		Certificate:    cert,
		PrivateKey:     privateKey,
		CertificatePEM: certPEM,
		PrivateKeyPEM:  keyPEM,
		SerialNumber:   template.SerialNumber,
		IssuedCerts:    make(map[string]*IssuedCert),
		Metadata:       make(map[string]interface{}),
	}

	// Track in root CA
	rootCA.IssuedCerts[intermediateCA.ID] = &IssuedCert{
		ID:           intermediateCA.ID,
		SerialNumber: template.SerialNumber,
		CommonName:   intermediateCA.CommonName,
		ValidFrom:    template.NotBefore,
		ValidTo:      template.NotAfter,
		CertType:     CertTypePeer,
	}

	// Audit log
	m.auditLogger.LogEvent(context.Background(), &AuditEvent{
		EventType: "CERTIFICATE_AUTHORITY_CREATED",
		Actor:     "system",
		Resource:  "intermediate_ca",
		Action:    "CREATE",
		Result:    "SUCCESS",
		Details: map[string]interface{}{
			"ca_id":       intermediateCA.ID,
			"common_name": intermediateCA.CommonName,
			"parent_ca":   rootCA.ID,
			"valid_until": intermediateCA.ValidTo,
		},
	})

	return intermediateCA, nil
}

// IssueCertificate issues a new certificate
func (m *MutualTLSManager) IssueCertificate(ctx context.Context, req *CertificateRequest) (*ManagedCertificate, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Validate request
	if err := m.validateCertificateRequest(req); err != nil {
		return nil, fmt.Errorf("invalid certificate request: %w", err)
	}

	// Generate private key
	privateKey, err := rsa.GenerateKey(rand.Reader, m.getKeyBitsForType(req.CertType))
	if err != nil {
		return nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	// Create certificate template
	template, err := m.createCertificateTemplate(req)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate template: %w", err)
	}

	// Sign certificate with intermediate CA
	signingCA := m.caManager.intermediateCA
	certDER, err := x509.CreateCertificate(rand.Reader, template, signingCA.Certificate, &privateKey.PublicKey, signingCA.PrivateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign certificate: %w", err)
	}

	// Parse the certificate
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, fmt.Errorf("failed to parse certificate: %w", err)
	}

	// Encode to PEM
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})

	// Build certificate chain
	chainPEM := [][]byte{
		m.caManager.intermediateCA.CertificatePEM,
		m.caManager.rootCA.CertificatePEM,
	}

	// Create managed certificate
	managedCert := &ManagedCertificate{
		ID:            uuid.New().String(),
		CommonName:    req.CommonName,
		SANs:          req.SANs,
		CertType:      req.CertType,
		Certificate:   cert,
		PrivateKey:    privateKey,
		CertPEM:       certPEM,
		PrivateKeyPEM: keyPEM,
		ChainPEM:      chainPEM,
		ValidFrom:     cert.NotBefore,
		ValidTo:       cert.NotAfter,
		SerialNumber:  cert.SerialNumber,
		Fingerprint:   generateCertFingerprint(cert),
		Status:        CertStatusActive,
		IssuerCA:      signingCA.ID,
		AutoRenew:     req.AutoRenew,
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
		Tags:          req.Tags,
		Metadata:      req.Metadata,
	}

	// Store certificate
	m.certStore.certificates[managedCert.ID] = managedCert
	m.certStore.certsBySerial[cert.SerialNumber.String()] = managedCert

	// Index by subject
	if _, exists := m.certStore.certsBySubject[req.CommonName]; !exists {
		m.certStore.certsBySubject[req.CommonName] = make([]*ManagedCertificate, 0)
	}
	m.certStore.certsBySubject[req.CommonName] = append(m.certStore.certsBySubject[req.CommonName], managedCert)

	// Track in CA
	signingCA.IssuedCerts[managedCert.ID] = &IssuedCert{
		ID:           managedCert.ID,
		SerialNumber: cert.SerialNumber,
		CommonName:   req.CommonName,
		ValidFrom:    cert.NotBefore,
		ValidTo:      cert.NotAfter,
		CertType:     req.CertType,
	}

	// Update metrics
	m.metrics.certificatesIssued.Inc()

	// Audit log
	m.auditLogger.LogEvent(ctx, &AuditEvent{
		EventType: "CERTIFICATE_ISSUED",
		Actor:     getActorFromContext(ctx),
		Resource:  "certificate",
		Action:    "CREATE",
		Result:    "SUCCESS",
		Details: map[string]interface{}{
			"cert_id":     managedCert.ID,
			"common_name": req.CommonName,
			"cert_type":   req.CertType,
			"valid_until": managedCert.ValidTo,
			"serial":      cert.SerialNumber.String(),
		},
	})

	return managedCert, nil
}

// CertificateRequest represents a certificate issuance request
type CertificateRequest struct {
	CommonName    string                 `json:"common_name"`
	SANs          []string               `json:"sans"`
	IPAddresses   []net.IP               `json:"ip_addresses"`
	EmailAddresses []string              `json:"email_addresses"`
	URIs          []string               `json:"uris"`
	CertType      CertType               `json:"cert_type"`
	KeyUsage      []string               `json:"key_usage"`
	ExtKeyUsage   []string               `json:"ext_key_usage"`
	ValidityDays  int                    `json:"validity_days"`
	AutoRenew     bool                   `json:"auto_renew"`
	Tags          []string               `json:"tags"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// GetTLSConfigForServer returns TLS config for server use
func (m *MutualTLSManager) GetTLSConfigForServer(certID string) (*tls.Config, error) {
	m.certStore.mu.RLock()
	cert, exists := m.certStore.certificates[certID]
	m.certStore.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("certificate not found: %s", certID)
	}

	if cert.Status != CertStatusActive {
		return nil, fmt.Errorf("certificate is not active: %s", certID)
	}

	// Create TLS certificate
	tlsCert, err := tls.X509KeyPair(cert.CertPEM, cert.PrivateKeyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to create TLS certificate: %w", err)
	}

	config := &tls.Config{
		Certificates: []tls.Certificate{tlsCert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		ClientCAs:    m.validator.rootCAs,
		MinVersion:   m.config.TLSVersionMin,
		MaxVersion:   m.config.TLSVersionMax,
		CipherSuites: m.config.CipherSuites,
		CurvePreferences: m.config.CurvePreferences,
		VerifyConnection: m.createVerifyConnectionFunc(),
	}

	return config, nil
}

// GetTLSConfigForClient returns TLS config for client use
func (m *MutualTLSManager) GetTLSConfigForClient(certID string) (*tls.Config, error) {
	m.certStore.mu.RLock()
	cert, exists := m.certStore.certificates[certID]
	m.certStore.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("certificate not found: %s", certID)
	}

	if cert.Status != CertStatusActive {
		return nil, fmt.Errorf("certificate is not active: %s", certID)
	}

	// Create TLS certificate
	tlsCert, err := tls.X509KeyPair(cert.CertPEM, cert.PrivateKeyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to create TLS certificate: %w", err)
	}

	config := &tls.Config{
		Certificates: []tls.Certificate{tlsCert},
		RootCAs:      m.validator.rootCAs,
		MinVersion:   m.config.TLSVersionMin,
		MaxVersion:   m.config.TLSVersionMax,
		CipherSuites: m.config.CipherSuites,
		CurvePreferences: m.config.CurvePreferences,
		VerifyConnection: m.createVerifyConnectionFunc(),
	}

	return config, nil
}

// createVerifyConnectionFunc creates a connection verification function
func (m *MutualTLSManager) createVerifyConnectionFunc() func(cs tls.ConnectionState) error {
	return func(cs tls.ConnectionState) error {
		start := time.Now()
		defer func() {
			m.metrics.tlsHandshakesDuration.Observe(time.Since(start).Seconds())
			m.metrics.tlsHandshakesTotal.Inc()
		}()

		if len(cs.PeerCertificates) == 0 {
			m.metrics.tlsHandshakesFailed.Inc()
			return fmt.Errorf("no peer certificates provided")
		}

		peerCert := cs.PeerCertificates[0]

		// Check if certificate is revoked
		if m.isCertificateRevoked(peerCert.SerialNumber) {
			m.metrics.tlsHandshakesFailed.Inc()
			return fmt.Errorf("certificate is revoked")
		}

		// Check certificate pinning if enabled
		if m.config.PinningEnabled {
			if !m.isCertificatePinned(peerCert) {
				m.metrics.tlsHandshakesFailed.Inc()
				return fmt.Errorf("certificate not pinned")
			}
		}

		// Check allowed CNs if configured
		if len(m.config.AllowedClientCNs) > 0 {
			allowed := false
			for _, allowedCN := range m.config.AllowedClientCNs {
				if peerCert.Subject.CommonName == allowedCN {
					allowed = true
					break
				}
			}
			if !allowed {
				m.metrics.tlsHandshakesFailed.Inc()
				return fmt.Errorf("client CN not allowed: %s", peerCert.Subject.CommonName)
			}
		}

		// Update certificate usage
		m.updateCertificateUsage(peerCert.SerialNumber)

		return nil
	}
}

// Helper methods and background workers
func (m *MutualTLSManager) validateCertificateRequest(req *CertificateRequest) error {
	if req.CommonName == "" {
		return fmt.Errorf("common name is required")
	}
	if req.ValidityDays <= 0 {
		req.ValidityDays = m.config.CertificateValidityDays
	}
	if req.ValidityDays > m.config.CertificateValidityDays {
		return fmt.Errorf("validity days exceeds maximum allowed")
	}
	return nil
}

func (m *MutualTLSManager) createCertificateTemplate(req *CertificateRequest) (*x509.Certificate, error) {
	serialNumber, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return nil, fmt.Errorf("failed to generate serial number: %w", err)
	}

	template := &x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			CommonName:   req.CommonName,
			Organization: []string{"NovaCron"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(time.Duration(req.ValidityDays) * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
	}

	// Set key usage based on certificate type
	switch req.CertType {
	case CertTypeServer:
		template.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
	case CertTypeClient:
		template.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth}
	case CertTypePeer:
		template.ExtKeyUsage = []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth}
	}

	// Add SANs
	for _, san := range req.SANs {
		template.DNSNames = append(template.DNSNames, san)
	}

	// Add IP addresses
	template.IPAddresses = req.IPAddresses

	// Add email addresses
	template.EmailAddresses = req.EmailAddresses

	return template, nil
}

func (m *MutualTLSManager) getKeyBitsForType(certType CertType) int {
	switch certType {
	case CertTypeServer:
		return m.config.ServerCertKeyBits
	case CertTypeClient:
		return m.config.ClientCertKeyBits
	default:
		return m.config.ClientCertKeyBits
	}
}

func (m *MutualTLSManager) isCertificateRevoked(serialNumber *big.Int) bool {
	m.revokedCerts.mu.RLock()
	defer m.revokedCerts.mu.RUnlock()

	_, revoked := m.revokedCerts.revokedCerts[serialNumber.String()]
	return revoked
}

func (m *MutualTLSManager) isCertificatePinned(cert *x509.Certificate) bool {
	fingerprint := generateCertFingerprint(cert)
	for _, pinnedFingerprint := range m.config.PinnedCertificates {
		if fingerprint == pinnedFingerprint {
			return true
		}
	}
	return false
}

func (m *MutualTLSManager) updateCertificateUsage(serialNumber *big.Int) {
	m.certStore.mu.Lock()
	defer m.certStore.mu.Unlock()

	if cert, exists := m.certStore.certsBySerial[serialNumber.String()]; exists {
		cert.UsageCount++
		cert.LastUsed = time.Now()
		cert.UpdatedAt = time.Now()
	}
}

// Background workers
func (m *MutualTLSManager) startCertificateRenewalWorker() {
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		m.renewExpiringCertificates()
	}
}

func (m *MutualTLSManager) renewExpiringCertificates() {
	threshold := time.Duration(m.config.RenewalThresholdDays) * 24 * time.Hour
	renewalTime := time.Now().Add(threshold)

	m.certStore.mu.RLock()
	expiringCerts := make([]*ManagedCertificate, 0)
	for _, cert := range m.certStore.certificates {
		if cert.AutoRenew && cert.Status == CertStatusActive && cert.ValidTo.Before(renewalTime) {
			expiringCerts = append(expiringCerts, cert)
		}
	}
	m.certStore.mu.RUnlock()

	for _, cert := range expiringCerts {
		// Create renewal request
		req := &CertificateRequest{
			CommonName:   cert.CommonName,
			SANs:         cert.SANs,
			CertType:     cert.CertType,
			ValidityDays: m.config.CertificateValidityDays,
			AutoRenew:    true,
			Tags:         cert.Tags,
			Metadata:     cert.Metadata,
		}

		newCert, err := m.IssueCertificate(context.Background(), req)
		if err != nil {
			// Log error and continue
			continue
		}

		// Mark old certificate as expired
		cert.Status = CertStatusExpired
		cert.UpdatedAt = time.Now()

		// Log renewal
		m.auditLogger.LogEvent(context.Background(), &AuditEvent{
			EventType: "CERTIFICATE_RENEWED",
			Actor:     "system",
			Resource:  "certificate",
			Action:    "UPDATE",
			Result:    "SUCCESS",
			Details: map[string]interface{}{
				"old_cert_id": cert.ID,
				"new_cert_id": newCert.ID,
				"common_name": cert.CommonName,
			},
		})
	}
}

func (m *MutualTLSManager) startCRLUpdateWorker() {
	ticker := time.NewTicker(6 * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		m.updateCRLs()
	}
}

func (m *MutualTLSManager) updateCRLs() {
	// Update CRLs from distribution points
	// Implementation would fetch and validate CRLs
}

func (m *MutualTLSManager) startMetricsCollector() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		m.collectMetrics()
	}
}

func (m *MutualTLSManager) collectMetrics() {
	// Collect and report certificate metrics
	m.certStore.mu.RLock()
	activeCerts := 0
	expiringCerts := 0
	expiredCerts := 0
	revokedCerts := 0

	for _, cert := range m.certStore.certificates {
		switch cert.Status {
		case CertStatusActive:
			activeCerts++
			if time.Until(cert.ValidTo) < 30*24*time.Hour {
				expiringCerts++
			}
		case CertStatusExpired:
			expiredCerts++
		case CertStatusRevoked:
			revokedCerts++
		}
	}
	m.certStore.mu.RUnlock()

	// Report metrics (would integrate with monitoring system)
}

// Helper functions
func generateCertFingerprint(cert *x509.Certificate) string {
	// Generate SHA-256 fingerprint
	hash := sha256.Sum256(cert.Raw)
	return fmt.Sprintf("%x", hash)
}

func getActorFromContext(ctx context.Context) string {
	if actor := ctx.Value("actor"); actor != nil {
		return actor.(string)
	}
	return "system"
}

// RevokeCertificate revokes a certificate
func (m *MutualTLSManager) RevokeCertificate(ctx context.Context, certID string, reason x509.RevocationReason) error {
	m.certStore.mu.Lock()
	defer m.certStore.mu.Unlock()

	cert, exists := m.certStore.certificates[certID]
	if !exists {
		return fmt.Errorf("certificate not found: %s", certID)
	}

	if cert.Status == CertStatusRevoked {
		return fmt.Errorf("certificate already revoked: %s", certID)
	}

	// Mark certificate as revoked
	cert.Status = CertStatusRevoked
	cert.UpdatedAt = time.Now()

	// Add to revoked certificates store
	revokedTime := time.Now()
	m.revokedCerts.mu.Lock()
	m.revokedCerts.revokedCerts[cert.SerialNumber.String()] = &RevokedCertificate{
		SerialNumber:     cert.SerialNumber,
		RevocationTime:   revokedTime,
		RevocationReason: reason,
	}
	m.revokedCerts.mu.Unlock()

	// Update metrics
	m.metrics.certificatesRevoked.Inc()

	// Audit log
	m.auditLogger.LogEvent(ctx, &AuditEvent{
		EventType: "CERTIFICATE_REVOKED",
		Actor:     getActorFromContext(ctx),
		Resource:  "certificate",
		Action:    "UPDATE",
		Result:    "SUCCESS",
		Details: map[string]interface{}{
			"cert_id":     certID,
			"common_name": cert.CommonName,
			"reason":      reason,
			"serial":      cert.SerialNumber.String(),
		},
	})

	return nil
}

// GetCertificate retrieves a certificate by ID
func (m *MutualTLSManager) GetCertificate(certID string) (*ManagedCertificate, error) {
	m.certStore.mu.RLock()
	defer m.certStore.mu.RUnlock()

	cert, exists := m.certStore.certificates[certID]
	if !exists {
		return nil, fmt.Errorf("certificate not found: %s", certID)
	}

	return cert, nil
}

// ListCertificates lists all certificates
func (m *MutualTLSManager) ListCertificates() []*ManagedCertificate {
	m.certStore.mu.RLock()
	defer m.certStore.mu.RUnlock()

	certs := make([]*ManagedCertificate, 0, len(m.certStore.certificates))
	for _, cert := range m.certStore.certificates {
		certs = append(certs, cert)
	}

	return certs
}

// GetCAChain returns the CA certificate chain
func (m *MutualTLSManager) GetCAChain() ([]*x509.Certificate, error) {
	m.caManager.mu.RLock()
	defer m.caManager.mu.RUnlock()

	return m.caManager.caChain, nil
}