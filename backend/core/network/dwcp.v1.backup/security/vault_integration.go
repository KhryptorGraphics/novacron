package security

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"fmt"
	"time"

	"github.com/hashicorp/vault/api"
	"go.uber.org/zap"
)

// VaultClient manages HashiCorp Vault PKI integration
type VaultClient struct {
	client      *api.Client
	pkiPath     string
	role        string
	logger      *zap.Logger
	tokenTTL    time.Duration
	lastRenewal time.Time
}

// VaultConfig holds Vault configuration
type VaultConfig struct {
	Address  string
	Token    string
	PKIPath  string
	Role     string
	TokenTTL time.Duration
	CACert   string
	TLSConfig *tls.Config
}

// NewVaultClient creates a new Vault client for PKI operations
func NewVaultClient(config VaultConfig, logger *zap.Logger) (*VaultClient, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	// Create Vault client configuration
	vaultConfig := api.DefaultConfig()
	vaultConfig.Address = config.Address

	// Configure TLS if provided
	if config.TLSConfig != nil {
		tlsConfig := &api.TLSConfig{
			Insecure: false,
		}
		if config.CACert != "" {
			tlsConfig.CACert = config.CACert
		}
		vaultConfig.ConfigureTLS(tlsConfig)
	}

	client, err := api.NewClient(vaultConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Vault client: %w", err)
	}

	// Set authentication token
	if config.Token != "" {
		client.SetToken(config.Token)
	}

	vc := &VaultClient{
		client:      client,
		pkiPath:     config.PKIPath,
		role:        config.Role,
		logger:      logger,
		tokenTTL:    config.TokenTTL,
		lastRenewal: time.Now(),
	}

	// Verify connection
	if err := vc.verifyConnection(); err != nil {
		return nil, fmt.Errorf("failed to verify Vault connection: %w", err)
	}

	logger.Info("Vault client initialized",
		zap.String("address", config.Address),
		zap.String("pki_path", config.PKIPath),
		zap.String("role", config.Role))

	return vc, nil
}

// verifyConnection verifies connection to Vault
func (vc *VaultClient) verifyConnection() error {
	// Try to read PKI mount
	path := fmt.Sprintf("sys/mounts/%s", vc.pkiPath)
	_, err := vc.client.Logical().Read(path)
	if err != nil {
		return fmt.Errorf("failed to verify PKI mount: %w", err)
	}
	return nil
}

// IssueCertificate requests a new certificate from Vault PKI
func (vc *VaultClient) IssueCertificate(commonName string, ttl string, altNames []string, ipSANs []string) (*tls.Certificate, *x509.Certificate, error) {
	vc.logger.Info("Requesting certificate from Vault",
		zap.String("common_name", commonName),
		zap.String("ttl", ttl))

	// Prepare request data
	data := map[string]interface{}{
		"common_name": commonName,
		"ttl":         ttl,
	}

	if len(altNames) > 0 {
		data["alt_names"] = joinStrings(altNames, ",")
	}

	if len(ipSANs) > 0 {
		data["ip_sans"] = joinStrings(ipSANs, ",")
	}

	// Issue certificate
	path := fmt.Sprintf("%s/issue/%s", vc.pkiPath, vc.role)
	secret, err := vc.client.Logical().Write(path, data)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to issue certificate: %w", err)
	}

	if secret == nil || secret.Data == nil {
		return nil, nil, errors.New("empty response from Vault")
	}

	// Extract certificate and key
	certPEM, ok := secret.Data["certificate"].(string)
	if !ok {
		return nil, nil, errors.New("certificate not found in response")
	}

	keyPEM, ok := secret.Data["private_key"].(string)
	if !ok {
		return nil, nil, errors.New("private_key not found in response")
	}

	caPEM, ok := secret.Data["issuing_ca"].(string)
	if !ok {
		return nil, nil, errors.New("issuing_ca not found in response")
	}

	// Combine certificate with CA chain
	fullChainPEM := certPEM + "\n" + caPEM

	// Parse into tls.Certificate
	cert, err := tls.X509KeyPair([]byte(fullChainPEM), []byte(keyPEM))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse certificate: %w", err)
	}

	// Parse x509 certificate for metadata
	x509Cert, err := x509.ParseCertificate(cert.Certificate[0])
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse x509 certificate: %w", err)
	}

	vc.logger.Info("Certificate issued successfully",
		zap.String("subject", x509Cert.Subject.CommonName),
		zap.Time("not_after", x509Cert.NotAfter),
		zap.Duration("valid_for", time.Until(x509Cert.NotAfter)))

	return &cert, x509Cert, nil
}

// RevokeCertificate revokes a certificate via Vault
func (vc *VaultClient) RevokeCertificate(serialNumber string) error {
	vc.logger.Info("Revoking certificate",
		zap.String("serial_number", serialNumber))

	path := fmt.Sprintf("%s/revoke", vc.pkiPath)
	_, err := vc.client.Logical().Write(path, map[string]interface{}{
		"serial_number": serialNumber,
	})

	if err != nil {
		return fmt.Errorf("failed to revoke certificate: %w", err)
	}

	vc.logger.Info("Certificate revoked successfully",
		zap.String("serial_number", serialNumber))

	return nil
}

// GetCACertificate retrieves the CA certificate
func (vc *VaultClient) GetCACertificate() (*x509.Certificate, error) {
	path := fmt.Sprintf("%s/ca/pem", vc.pkiPath)
	secret, err := vc.client.Logical().Read(path)
	if err != nil {
		return nil, fmt.Errorf("failed to get CA certificate: %w", err)
	}

	if secret == nil || secret.Data == nil {
		return nil, errors.New("empty CA certificate response")
	}

	caPEM, ok := secret.Data["certificate"].(string)
	if !ok {
		return nil, errors.New("CA certificate not found in response")
	}

	// Parse PEM
	block, _ := pem.Decode([]byte(caPEM))
	if block == nil {
		return nil, errors.New("failed to decode CA certificate PEM")
	}

	// Parse x509 certificate
	caCert, err := x509.ParseCertificate(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CA certificate: %w", err)
	}

	return caCert, nil
}

// GetCRL retrieves the certificate revocation list
func (vc *VaultClient) GetCRL() (*x509.RevocationList, error) {
	path := fmt.Sprintf("%s/crl", vc.pkiPath)
	secret, err := vc.client.Logical().Read(path)
	if err != nil {
		return nil, fmt.Errorf("failed to get CRL: %w", err)
	}

	if secret == nil || secret.Data == nil {
		return nil, errors.New("empty CRL response")
	}

	crlPEM, ok := secret.Data["certificate"].(string)
	if !ok {
		return nil, errors.New("CRL not found in response")
	}

	// Parse PEM
	block, _ := pem.Decode([]byte(crlPEM))
	if block == nil {
		return nil, errors.New("failed to decode CRL PEM")
	}

	// Parse CRL
	crl, err := x509.ParseRevocationList(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CRL: %w", err)
	}

	return crl, nil
}

// RenewToken renews the Vault authentication token
func (vc *VaultClient) RenewToken() error {
	secret, err := vc.client.Auth().Token().RenewSelf(int(vc.tokenTTL.Seconds()))
	if err != nil {
		return fmt.Errorf("failed to renew token: %w", err)
	}

	vc.lastRenewal = time.Now()
	vc.logger.Info("Vault token renewed",
		zap.Duration("ttl", time.Duration(secret.Auth.LeaseDuration)*time.Second))

	return nil
}

// StartTokenRenewal starts automatic token renewal
func (vc *VaultClient) StartTokenRenewal(stopChan <-chan struct{}) {
	ticker := time.NewTicker(vc.tokenTTL / 2)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := vc.RenewToken(); err != nil {
				vc.logger.Error("Failed to renew Vault token", zap.Error(err))
			}
		case <-stopChan:
			return
		}
	}
}

// CreateRole creates or updates a PKI role
func (vc *VaultClient) CreateRole(roleName string, config map[string]interface{}) error {
	path := fmt.Sprintf("%s/roles/%s", vc.pkiPath, roleName)
	_, err := vc.client.Logical().Write(path, config)
	if err != nil {
		return fmt.Errorf("failed to create role: %w", err)
	}

	vc.logger.Info("PKI role created/updated",
		zap.String("role", roleName))

	return nil
}

// joinStrings joins string slice with separator
func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	result := strs[0]
	for i := 1; i < len(strs); i++ {
		result += sep + strs[i]
	}
	return result
}
