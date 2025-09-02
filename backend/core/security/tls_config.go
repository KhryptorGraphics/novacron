package security

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"os"
)

// TLSConfiguration holds TLS configuration settings
type TLSConfiguration struct {
	CertFile      string
	KeyFile       string
	CAFile        string
	MinVersion    uint16
	CipherSuites  []uint16
	ClientAuth    tls.ClientAuthType
	ServerName    string
}

// DefaultTLSConfig returns a secure default TLS configuration
func DefaultTLSConfig() *TLSConfiguration {
	return &TLSConfiguration{
		CertFile:   os.Getenv("TLS_CERT_FILE"),
		KeyFile:    os.Getenv("TLS_KEY_FILE"),
		CAFile:     os.Getenv("TLS_CA_FILE"),
		MinVersion: tls.VersionTLS13, // TLS 1.3 minimum
		CipherSuites: []uint16{
			// TLS 1.3 cipher suites (preferred)
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_AES_128_GCM_SHA256,
			tls.TLS_CHACHA20_POLY1305_SHA256,
			// TLS 1.2 cipher suites (fallback)
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
		},
		ClientAuth: tls.NoClientCert,
		ServerName: os.Getenv("APP_URL"),
	}
}

// LoadTLSConfig creates a tls.Config from TLSConfiguration
func (tc *TLSConfiguration) LoadTLSConfig() (*tls.Config, error) {
	// Load certificate and key
	cert, err := tls.LoadX509KeyPair(tc.CertFile, tc.KeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load certificate and key: %w", err)
	}

	// Create base TLS config
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tc.MinVersion,
		CipherSuites: tc.CipherSuites,
		ClientAuth:   tc.ClientAuth,
		ServerName:   tc.ServerName,
		// Security settings
		PreferServerCipherSuites: true,
		CurvePreferences: []tls.CurveID{
			tls.CurveP256,
			tls.X25519,
		},
		// Enable session tickets for performance
		SessionTicketsDisabled: false,
		// Set reasonable timeouts
		Renegotiation: tls.RenegotiateNever,
	}

	// Load CA certificate if mutual TLS is required
	if tc.CAFile != "" && tc.ClientAuth != tls.NoClientCert {
		caCert, err := ioutil.ReadFile(tc.CAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read CA certificate: %w", err)
		}

		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to parse CA certificate")
		}

		tlsConfig.ClientCAs = caCertPool
		tlsConfig.ClientAuth = tc.ClientAuth
	}

	return tlsConfig, nil
}

// LoadClientTLSConfig creates a client TLS configuration
func (tc *TLSConfiguration) LoadClientTLSConfig() (*tls.Config, error) {
	tlsConfig := &tls.Config{
		MinVersion:   tc.MinVersion,
		CipherSuites: tc.CipherSuites,
		ServerName:   tc.ServerName,
	}

	// Load client certificate if provided
	if tc.CertFile != "" && tc.KeyFile != "" {
		cert, err := tls.LoadX509KeyPair(tc.CertFile, tc.KeyFile)
		if err != nil {
			return nil, fmt.Errorf("failed to load client certificate: %w", err)
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}

	// Load CA certificate for server verification
	if tc.CAFile != "" {
		caCert, err := ioutil.ReadFile(tc.CAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read CA certificate: %w", err)
		}

		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to parse CA certificate")
		}

		tlsConfig.RootCAs = caCertPool
	}

	return tlsConfig, nil
}

// ValidateTLSConfig checks if the TLS configuration is valid
func (tc *TLSConfiguration) ValidateTLSConfig() error {
	// Check if TLS is enabled
	if os.Getenv("TLS_ENABLED") != "true" {
		return fmt.Errorf("TLS is not enabled")
	}

	// Check certificate files exist
	if tc.CertFile == "" {
		return fmt.Errorf("TLS certificate file not specified")
	}
	if _, err := os.Stat(tc.CertFile); os.IsNotExist(err) {
		return fmt.Errorf("TLS certificate file does not exist: %s", tc.CertFile)
	}

	if tc.KeyFile == "" {
		return fmt.Errorf("TLS key file not specified")
	}
	if _, err := os.Stat(tc.KeyFile); os.IsNotExist(err) {
		return fmt.Errorf("TLS key file does not exist: %s", tc.KeyFile)
	}

	// Validate minimum TLS version
	if tc.MinVersion < tls.VersionTLS12 {
		return fmt.Errorf("TLS version must be 1.2 or higher")
	}

	// Try to load the configuration
	_, err := tc.LoadTLSConfig()
	if err != nil {
		return fmt.Errorf("failed to load TLS configuration: %w", err)
	}

	return nil
}

// MutualTLSConfig returns a configuration for mutual TLS
func MutualTLSConfig() *TLSConfiguration {
	config := DefaultTLSConfig()
	config.ClientAuth = tls.RequireAndVerifyClientCert
	return config
}

// GetTLSVersionString returns a human-readable TLS version string
func GetTLSVersionString(version uint16) string {
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
		return "Unknown"
	}
}