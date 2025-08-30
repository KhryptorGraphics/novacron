package auth

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"time"

	"golang.org/x/crypto/chacha20poly1305"
	"golang.org/x/crypto/scrypt"
)

// EncryptionConfig defines encryption configuration
type EncryptionConfig struct {
	// DefaultAlgorithm for symmetric encryption (AES-256-GCM, ChaCha20Poly1305)
	DefaultAlgorithm string
	// KeyDerivationFunction for password-based encryption (scrypt, argon2)
	KeyDerivationFunction string
	// KeyRotationInterval for automatic key rotation
	KeyRotationInterval time.Duration
	// KeySize in bytes (32 for 256-bit)
	KeySize int
	// ScryptN CPU/memory cost parameter
	ScryptN int
	// ScryptR block size parameter
	ScryptR int
	// ScryptP parallelization parameter
	ScryptP int
	// CertificateValidityPeriod for TLS certificates
	CertificateValidityPeriod time.Duration
	// KeyUsageRotationPeriod for key usage limits
	KeyUsageRotationPeriod time.Duration
}

// EncryptionKey represents an encryption key with metadata
type EncryptionKey struct {
	ID        string    `json:"id"`
	KeyData   []byte    `json:"key_data"`
	Algorithm string    `json:"algorithm"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
	UsageCount int64    `json:"usage_count"`
	MaxUsage  int64     `json:"max_usage"`
	Active    bool      `json:"active"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// EncryptedData represents encrypted data with metadata
type EncryptedData struct {
	Data      string    `json:"data"`
	KeyID     string    `json:"key_id"`
	Algorithm string    `json:"algorithm"`
	Nonce     string    `json:"nonce"`
	CreatedAt time.Time `json:"created_at"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// TLSCertificate represents a TLS certificate with private key
type TLSCertificate struct {
	Certificate *x509.Certificate `json:"-"`
	PrivateKey  interface{}       `json:"-"`
	CertPEM     []byte            `json:"cert_pem"`
	KeyPEM      []byte            `json:"key_pem"`
	CreatedAt   time.Time         `json:"created_at"`
	ExpiresAt   time.Time         `json:"expires_at"`
	Domains     []string          `json:"domains"`
	KeyUsage    x509.KeyUsage     `json:"key_usage"`
}

// EncryptionService provides comprehensive encryption capabilities
type EncryptionService struct {
	config EncryptionConfig
	keys   map[string]*EncryptionKey
	certs  map[string]*TLSCertificate
}

// NewEncryptionService creates a new encryption service
func NewEncryptionService(config EncryptionConfig) *EncryptionService {
	if config.DefaultAlgorithm == "" {
		config.DefaultAlgorithm = "AES-256-GCM"
	}
	if config.KeyDerivationFunction == "" {
		config.KeyDerivationFunction = "scrypt"
	}
	if config.KeyRotationInterval == 0 {
		config.KeyRotationInterval = 30 * 24 * time.Hour // 30 days
	}
	if config.KeySize == 0 {
		config.KeySize = 32 // 256 bits
	}
	if config.ScryptN == 0 {
		config.ScryptN = 32768 // 2^15
	}
	if config.ScryptR == 0 {
		config.ScryptR = 8
	}
	if config.ScryptP == 0 {
		config.ScryptP = 1
	}
	if config.CertificateValidityPeriod == 0 {
		config.CertificateValidityPeriod = 365 * 24 * time.Hour // 1 year
	}
	if config.KeyUsageRotationPeriod == 0 {
		config.KeyUsageRotationPeriod = 7 * 24 * time.Hour // 7 days
	}

	return &EncryptionService{
		config: config,
		keys:   make(map[string]*EncryptionKey),
		certs:  make(map[string]*TLSCertificate),
	}
}

// GenerateKey generates a new encryption key
func (e *EncryptionService) GenerateKey(algorithm string) (*EncryptionKey, error) {
	if algorithm == "" {
		algorithm = e.config.DefaultAlgorithm
	}

	keyData := make([]byte, e.config.KeySize)
	_, err := rand.Read(keyData)
	if err != nil {
		return nil, fmt.Errorf("failed to generate key: %w", err)
	}

	now := time.Now()
	key := &EncryptionKey{
		ID:         e.generateKeyID(),
		KeyData:    keyData,
		Algorithm:  algorithm,
		CreatedAt:  now,
		ExpiresAt:  now.Add(e.config.KeyRotationInterval),
		UsageCount: 0,
		MaxUsage:   1000000, // 1 million operations before rotation
		Active:     true,
		Metadata:   make(map[string]interface{}),
	}

	e.keys[key.ID] = key
	return key, nil
}

// EncryptData encrypts data using the specified algorithm
func (e *EncryptionService) EncryptData(data []byte, keyID string) (*EncryptedData, error) {
	key, exists := e.keys[keyID]
	if !exists || !key.Active {
		return nil, fmt.Errorf("encryption key not found or inactive: %s", keyID)
	}

	if time.Now().After(key.ExpiresAt) {
		return nil, fmt.Errorf("encryption key expired: %s", keyID)
	}

	if key.UsageCount >= key.MaxUsage {
		return nil, fmt.Errorf("encryption key usage limit exceeded: %s", keyID)
	}

	var encrypted []byte
	var nonce []byte
	var err error

	switch key.Algorithm {
	case "AES-256-GCM":
		encrypted, nonce, err = e.encryptAESGCM(data, key.KeyData)
	case "ChaCha20Poly1305":
		encrypted, nonce, err = e.encryptChaCha20Poly1305(data, key.KeyData)
	default:
		return nil, fmt.Errorf("unsupported encryption algorithm: %s", key.Algorithm)
	}

	if err != nil {
		return nil, fmt.Errorf("encryption failed: %w", err)
	}

	// Increment usage counter
	key.UsageCount++

	return &EncryptedData{
		Data:      base64.StdEncoding.EncodeToString(encrypted),
		KeyID:     keyID,
		Algorithm: key.Algorithm,
		Nonce:     base64.StdEncoding.EncodeToString(nonce),
		CreatedAt: time.Now(),
		Metadata:  make(map[string]interface{}),
	}, nil
}

// DecryptData decrypts encrypted data
func (e *EncryptionService) DecryptData(encryptedData *EncryptedData) ([]byte, error) {
	key, exists := e.keys[encryptedData.KeyID]
	if !exists {
		return nil, fmt.Errorf("decryption key not found: %s", encryptedData.KeyID)
	}

	data, err := base64.StdEncoding.DecodeString(encryptedData.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode encrypted data: %w", err)
	}

	nonce, err := base64.StdEncoding.DecodeString(encryptedData.Nonce)
	if err != nil {
		return nil, fmt.Errorf("failed to decode nonce: %w", err)
	}

	var decrypted []byte
	switch encryptedData.Algorithm {
	case "AES-256-GCM":
		decrypted, err = e.decryptAESGCM(data, key.KeyData, nonce)
	case "ChaCha20Poly1305":
		decrypted, err = e.decryptChaCha20Poly1305(data, key.KeyData, nonce)
	default:
		return nil, fmt.Errorf("unsupported decryption algorithm: %s", encryptedData.Algorithm)
	}

	if err != nil {
		return nil, fmt.Errorf("decryption failed: %w", err)
	}

	return decrypted, nil
}

// EncryptString encrypts a string and returns base64 encoded result
func (e *EncryptionService) EncryptString(plaintext, keyID string) (string, error) {
	encrypted, err := e.EncryptData([]byte(plaintext), keyID)
	if err != nil {
		return "", err
	}

	// Serialize encrypted data to JSON-like format for storage
	result := fmt.Sprintf("%s.%s.%s.%s", encrypted.KeyID, encrypted.Algorithm, encrypted.Nonce, encrypted.Data)
	return base64.StdEncoding.EncodeToString([]byte(result)), nil
}

// DecryptString decrypts a base64 encoded encrypted string
func (e *EncryptionService) DecryptString(encryptedString string) (string, error) {
	decodedBytes, err := base64.StdEncoding.DecodeString(encryptedString)
	if err != nil {
		return "", fmt.Errorf("failed to decode encrypted string: %w", err)
	}

	parts := string(decodedBytes)
	components := splitString(parts, ".", 4)
	if len(components) != 4 {
		return "", fmt.Errorf("invalid encrypted string format")
	}

	encryptedData := &EncryptedData{
		KeyID:     components[0],
		Algorithm: components[1],
		Nonce:     components[2],
		Data:      components[3],
	}

	decrypted, err := e.DecryptData(encryptedData)
	if err != nil {
		return "", err
	}

	return string(decrypted), nil
}

// DeriveKeyFromPassword derives an encryption key from a password
func (e *EncryptionService) DeriveKeyFromPassword(password, salt string) ([]byte, error) {
	saltBytes, err := base64.StdEncoding.DecodeString(salt)
	if err != nil {
		return nil, fmt.Errorf("invalid salt: %w", err)
	}

	switch e.config.KeyDerivationFunction {
	case "scrypt":
		return scrypt.Key([]byte(password), saltBytes, e.config.ScryptN, e.config.ScryptR, e.config.ScryptP, e.config.KeySize)
	default:
		return nil, fmt.Errorf("unsupported key derivation function: %s", e.config.KeyDerivationFunction)
	}
}

// GenerateTLSCertificate generates a self-signed TLS certificate
func (e *EncryptionService) GenerateTLSCertificate(domains []string, organization string) (*TLSCertificate, error) {
	// Generate private key
	privateKey, err := ecdsa.GenerateKey(elliptic.P384(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	// Create certificate template
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			Organization: []string{organization},
			Country:      []string{"US"},
			Province:     []string{"CA"},
			Locality:     []string{"San Francisco"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(e.config.CertificateValidityPeriod),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	// Add domains
	for _, domain := range domains {
		if ip := net.ParseIP(domain); ip != nil {
			template.IPAddresses = append(template.IPAddresses, ip)
		} else {
			template.DNSNames = append(template.DNSNames, domain)
		}
	}

	// Create certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %w", err)
	}

	// Parse certificate
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, fmt.Errorf("failed to parse certificate: %w", err)
	}

	// Encode to PEM
	certPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: certDER,
	})

	privateKeyBytes, err := x509.MarshalECPrivateKey(privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal private key: %w", err)
	}

	keyPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "EC PRIVATE KEY",
		Bytes: privateKeyBytes,
	})

	tlsCert := &TLSCertificate{
		Certificate: cert,
		PrivateKey:  privateKey,
		CertPEM:     certPEM,
		KeyPEM:      keyPEM,
		CreatedAt:   cert.NotBefore,
		ExpiresAt:   cert.NotAfter,
		Domains:     domains,
		KeyUsage:    cert.KeyUsage,
	}

	certID := e.generateKeyID()
	e.certs[certID] = tlsCert

	return tlsCert, nil
}

// GetTLSConfig returns a TLS configuration for secure connections
func (e *EncryptionService) GetTLSConfig(certID string) (*tls.Config, error) {
	cert, exists := e.certs[certID]
	if !exists {
		return nil, fmt.Errorf("certificate not found: %s", certID)
	}

	tlsCert, err := tls.X509KeyPair(cert.CertPEM, cert.KeyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to load certificate: %w", err)
	}

	return &tls.Config{
		Certificates: []tls.Certificate{tlsCert},
		MinVersion:   tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
		},
		PreferServerCipherSuites: true,
		ClientAuth:               tls.RequireAndVerifyClientCert,
	}, nil
}

// RotateKeys rotates expired encryption keys
func (e *EncryptionService) RotateKeys() error {
	now := time.Now()
	for keyID, key := range e.keys {
		if now.After(key.ExpiresAt) || key.UsageCount >= key.MaxUsage {
			// Deactivate old key
			key.Active = false
			
			// Generate new key
			newKey, err := e.GenerateKey(key.Algorithm)
			if err != nil {
				return fmt.Errorf("failed to rotate key %s: %w", keyID, err)
			}
			
			// Copy metadata
			newKey.Metadata = key.Metadata
		}
	}
	return nil
}

// GetActiveKeys returns all active encryption keys
func (e *EncryptionService) GetActiveKeys() []*EncryptionKey {
	var activeKeys []*EncryptionKey
	for _, key := range e.keys {
		if key.Active && time.Now().Before(key.ExpiresAt) {
			activeKeys = append(activeKeys, key)
		}
	}
	return activeKeys
}

// encryptAESGCM encrypts data using AES-256-GCM
func (e *EncryptionService) encryptAESGCM(data, key []byte) ([]byte, []byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, nil, err
	}

	nonce := make([]byte, gcm.NonceSize())
	_, err = io.ReadFull(rand.Reader, nonce)
	if err != nil {
		return nil, nil, err
	}

	ciphertext := gcm.Seal(nil, nonce, data, nil)
	return ciphertext, nonce, nil
}

// decryptAESGCM decrypts data using AES-256-GCM
func (e *EncryptionService) decryptAESGCM(ciphertext, key, nonce []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

// encryptChaCha20Poly1305 encrypts data using ChaCha20-Poly1305
func (e *EncryptionService) encryptChaCha20Poly1305(data, key []byte) ([]byte, []byte, error) {
	aead, err := chacha20poly1305.New(key)
	if err != nil {
		return nil, nil, err
	}

	nonce := make([]byte, aead.NonceSize())
	_, err = io.ReadFull(rand.Reader, nonce)
	if err != nil {
		return nil, nil, err
	}

	ciphertext := aead.Seal(nil, nonce, data, nil)
	return ciphertext, nonce, nil
}

// decryptChaCha20Poly1305 decrypts data using ChaCha20-Poly1305
func (e *EncryptionService) decryptChaCha20Poly1305(ciphertext, key, nonce []byte) ([]byte, error) {
	aead, err := chacha20poly1305.New(key)
	if err != nil {
		return nil, err
	}

	plaintext, err := aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

// generateKeyID generates a unique key identifier
func (e *EncryptionService) generateKeyID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("key-%x", b)
}

// generateSalt generates a cryptographic salt
func (e *EncryptionService) GenerateSalt() (string, error) {
	salt := make([]byte, 32)
	_, err := rand.Read(salt)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(salt), nil
}

// HashData creates a SHA-256 hash of data
func (e *EncryptionService) HashData(data []byte) string {
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash)
}

// splitString splits a string by delimiter with limit
func splitString(s, delimiter string, limit int) []string {
	parts := make([]string, 0, limit)
	start := 0

	for i := 0; i < limit-1; i++ {
		idx := strings.Index(s[start:], delimiter)
		if idx == -1 {
			break
		}
		parts = append(parts, s[start:start+idx])
		start = start + idx + len(delimiter)
	}

	if start < len(s) {
		parts = append(parts, s[start:])
	}

	return parts
}

// DefaultEncryptionConfig returns secure default encryption configuration
func DefaultEncryptionConfig() EncryptionConfig {
	return EncryptionConfig{
		DefaultAlgorithm:              "AES-256-GCM",
		KeyDerivationFunction:         "scrypt",
		KeyRotationInterval:           30 * 24 * time.Hour,
		KeySize:                       32,
		ScryptN:                       32768,
		ScryptR:                       8,
		ScryptP:                       1,
		CertificateValidityPeriod:     365 * 24 * time.Hour,
		KeyUsageRotationPeriod:        7 * 24 * time.Hour,
	}
}