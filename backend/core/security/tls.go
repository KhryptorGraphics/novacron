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
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// TLSConfig manages TLS certificates and configuration
type TLSConfig struct {
	CertFile       string
	KeyFile        string
	CAFile         string
	ClientAuth     tls.ClientAuthType
	MinVersion     uint16
	CipherSuites   []uint16
	AutoTLS        bool
	ACMEEmail      string
	ACMEDomains    []string
}

// NewTLSConfig creates a secure TLS configuration
func NewTLSConfig(certFile, keyFile string) *TLSConfig {
	return &TLSConfig{
		CertFile:   certFile,
		KeyFile:    keyFile,
		MinVersion: tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
		},
	}
}

// GetTLSConfig returns a configured tls.Config
func (tc *TLSConfig) GetTLSConfig() (*tls.Config, error) {
	cert, err := tls.LoadX509KeyPair(tc.CertFile, tc.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load certificate: %w", err)
	}
	
	config := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tc.MinVersion,
		CipherSuites: tc.CipherSuites,
		CurvePreferences: []tls.CurveID{
			tls.X25519,
			tls.CurveP256,
		},
		PreferServerCipherSuites: true,
	}
	
	// Add client authentication if specified
	if tc.ClientAuth != tls.NoClientCert && tc.CAFile != "" {
		caCert, err := os.ReadFile(tc.CAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read CA file: %w", err)
		}
		
		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to parse CA certificate")
		}
		
		config.ClientAuth = tc.ClientAuth
		config.ClientCAs = caCertPool
	}
	
	return config, nil
}

// GenerateSelfSignedCert generates a self-signed certificate for development
func GenerateSelfSignedCert(hosts []string, certPath, keyPath string) error {
	priv, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return fmt.Errorf("failed to generate private key: %w", err)
	}
	
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization:  []string{"NovaCron Development"},
			Country:       []string{"US"},
			Province:      []string{""},
			Locality:      []string{""},
			StreetAddress: []string{""},
			PostalCode:    []string{""},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}
	
	// Add hosts and IPs
	for _, h := range hosts {
		if ip := net.ParseIP(h); ip != nil {
			template.IPAddresses = append(template.IPAddresses, ip)
		} else {
			template.DNSNames = append(template.DNSNames, h)
		}
	}
	
	// Generate certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return fmt.Errorf("failed to create certificate: %w", err)
	}
	
	// Create directories if needed
	if err := os.MkdirAll(filepath.Dir(certPath), 0755); err != nil {
		return fmt.Errorf("failed to create cert directory: %w", err)
	}
	
	// Write certificate
	certOut, err := os.Create(certPath)
	if err != nil {
		return fmt.Errorf("failed to open cert file: %w", err)
	}
	defer certOut.Close()
	
	if err := pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: certDER}); err != nil {
		return fmt.Errorf("failed to write certificate: %w", err)
	}
	
	// Write private key
	keyOut, err := os.OpenFile(keyPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
	if err != nil {
		return fmt.Errorf("failed to open key file: %w", err)
	}
	defer keyOut.Close()
	
	privKeyPEM := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(priv),
	}
	
	if err := pem.Encode(keyOut, privKeyPEM); err != nil {
		return fmt.Errorf("failed to write private key: %w", err)
	}
	
	return nil
}

// HTTPSRedirectHandler redirects HTTP to HTTPS
type HTTPSRedirectHandler struct {
	HTTPSPort string
}

// ServeHTTP redirects all HTTP requests to HTTPS
func (h *HTTPSRedirectHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	host := r.Host
	// Remove port if present
	if colonPos := strings.LastIndex(host, ":"); colonPos != -1 {
		host = host[:colonPos]
	}
	
	target := "https://" + host
	if h.HTTPSPort != "443" && h.HTTPSPort != "" {
		target = fmt.Sprintf("https://%s:%s", host, h.HTTPSPort)
	}
	target += r.URL.Path
	if r.URL.RawQuery != "" {
		target += "?" + r.URL.RawQuery
	}
	
	http.Redirect(w, r, target, http.StatusMovedPermanently)
}


// TLSServer wraps a server with TLS configuration
type TLSServer struct {
	server    *http.Server
	tlsConfig *TLSConfig
}

// NewTLSServer creates a new TLS-enabled server
func NewTLSServer(addr string, handler http.Handler, tlsConfig *TLSConfig) (*TLSServer, error) {
	config, err := tlsConfig.GetTLSConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get TLS config: %w", err)
	}
	
	server := &http.Server{
		Addr:      addr,
		Handler:   handler,
		TLSConfig: config,
		// Security headers middleware should be added to handler
		ReadTimeout:    15 * time.Second,
		WriteTimeout:   15 * time.Second,
		IdleTimeout:    60 * time.Second,
		MaxHeaderBytes: 1 << 20, // 1 MB
	}
	
	return &TLSServer{
		server:    server,
		tlsConfig: tlsConfig,
	}, nil
}

// Start starts the TLS server
func (ts *TLSServer) Start() error {
	return ts.server.ListenAndServeTLS(ts.tlsConfig.CertFile, ts.tlsConfig.KeyFile)
}

// Stop gracefully stops the TLS server
func (ts *TLSServer) Stop(ctx context.Context) error {
	return ts.server.Shutdown(ctx)
}

// generateSecureToken generates a cryptographically secure random token
func generateSecureToken(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, length)
	for i := range b {
		n, _ := rand.Int(rand.Reader, big.NewInt(int64(len(charset))))
		b[i] = charset[n.Int64()]
	}
	return string(b)
}