package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"io"
	"sync"
	"time"
)

// EncryptionManager provides comprehensive encryption services
type EncryptionManager struct {
	config           EncryptionConfig
	keyManager       *KeyManager
	tlsConfig        *tls.Config
	certManager      *CertificateManager
	fieldEncryption  *FieldEncryption
	mu               sync.RWMutex
}

// KeyManager handles cryptographic key management
type KeyManager struct {
	keys             map[string]*EncryptionKey
	activeKeyID      string
	rotationSchedule map[string]time.Time
	keyDerivation    *KeyDerivationService
	keyStorage       *KeyStorage
	mu               sync.RWMutex
}

// EncryptionKey represents a cryptographic key
type EncryptionKey struct {
	ID          string                 `json:"id"`
	Algorithm   string                 `json:"algorithm"`
	KeySize     int                    `json:"key_size"`
	Purpose     string                 `json:"purpose"` // data, auth, signing
	Status      KeyStatus              `json:"status"`
	KeyData     []byte                 `json:"-"` // Never serialize
	PublicKey   []byte                 `json:"public_key,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	ExpiresAt   *time.Time             `json:"expires_at,omitempty"`
	RotatedFrom string                 `json:"rotated_from,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type KeyStatus string

const (
	KeyStatusActive   KeyStatus = "active"
	KeyStatusInactive KeyStatus = "inactive"
	KeyStatusRotating KeyStatus = "rotating"
	KeyStatusExpired  KeyStatus = "expired"
	KeyStatusRevoked  KeyStatus = "revoked"
)

// CertificateManager handles TLS certificate management
type CertificateManager struct {
	certificates map[string]*CertificateInfo
	caPool       *x509.CertPool
	autoRenewal  bool
	mu           sync.RWMutex
}

// CertificateInfo holds certificate information
type CertificateInfo struct {
	ID          string      `json:"id"`
	CommonName  string      `json:"common_name"`
	SANs        []string    `json:"sans"`
	Certificate []byte      `json:"certificate"`
	PrivateKey  []byte      `json:"private_key"`
	Chain       [][]byte    `json:"chain,omitempty"`
	IssuedAt    time.Time   `json:"issued_at"`
	ExpiresAt   time.Time   `json:"expires_at"`
	Issuer      string      `json:"issuer"`
	KeyUsage    []string    `json:"key_usage"`
	Status      KeyStatus   `json:"status"`
}

// FieldEncryption provides field-level encryption
type FieldEncryption struct {
	encryptionKeys map[string]*EncryptionKey
	fieldMappings  map[string]*FieldMapping
	mu             sync.RWMutex
}

// FieldMapping defines field encryption configuration
type FieldMapping struct {
	TableName     string `json:"table_name"`
	ColumnName    string `json:"column_name"`
	KeyID         string `json:"key_id"`
	Algorithm     string `json:"algorithm"`
	Tokenization  bool   `json:"tokenization"`
	SearchEnabled bool   `json:"search_enabled"`
}

// KeyStorage provides secure key storage
type KeyStorage struct {
	storageType   string // vault, hsm, kms
	vaultClient   VaultClient
	hsmClient     HSMClient
	kmsClient     KMSClient
	encryptionKey []byte
}

// KeyDerivationService provides key derivation functions
type KeyDerivationService struct {
	masterKey []byte
	salt      []byte
}

// EncryptedData represents encrypted data with metadata
type EncryptedData struct {
	Data          []byte            `json:"data"`
	Algorithm     string            `json:"algorithm"`
	KeyID         string            `json:"key_id"`
	IV            []byte            `json:"iv,omitempty"`
	AuthTag       []byte            `json:"auth_tag,omitempty"`
	Metadata      map[string]string `json:"metadata,omitempty"`
	EncryptedAt   time.Time         `json:"encrypted_at"`
}

// NewEncryptionManager creates a new encryption manager
func NewEncryptionManager(config EncryptionConfig) *EncryptionManager {
	keyManager := &KeyManager{
		keys:             make(map[string]*EncryptionKey),
		rotationSchedule: make(map[string]time.Time),
		keyDerivation:    &KeyDerivationService{},
		keyStorage:       &KeyStorage{storageType: "vault"},
	}

	certManager := &CertificateManager{
		certificates: make(map[string]*CertificateInfo),
		caPool:       x509.NewCertPool(),
		autoRenewal:  true,
	}

	fieldEncryption := &FieldEncryption{
		encryptionKeys: make(map[string]*EncryptionKey),
		fieldMappings:  make(map[string]*FieldMapping),
	}

	em := &EncryptionManager{
		config:          config,
		keyManager:      keyManager,
		certManager:     certManager,
		fieldEncryption: fieldEncryption,
	}

	// Initialize TLS configuration
	em.initializeTLS()

	// Initialize default encryption keys
	em.initializeDefaultKeys()

	// Start key rotation scheduler
	go em.startKeyRotationScheduler()

	// Start certificate renewal scheduler
	go em.startCertRenewalScheduler()

	return em
}

// initializeTLS sets up TLS configuration
func (em *EncryptionManager) initializeTLS() {
	em.tlsConfig = &tls.Config{
		MinVersion:               tls.VersionTLS13,
		CurvePreferences:         []tls.CurveID{tls.X25519, tls.CurveP384, tls.CurveP256},
		PreferServerCipherSuites: true,
		CipherSuites: []uint16{
			tls.TLS_AES_256_GCM_SHA384,
			tls.TLS_AES_128_GCM_SHA256,
			tls.TLS_CHACHA20_POLY1305_SHA256,
		},
		ClientAuth:               tls.RequireAndVerifyClientCert,
		InsecureSkipVerify:      false,
		GetCertificate:          em.getCertificate,
		GetClientCertificate:    em.getClientCertificate,
	}
}

// initializeDefaultKeys creates default encryption keys
func (em *EncryptionManager) initializeDefaultKeys() {
	// Create master data encryption key
	dataKey, err := em.GenerateKey("AES-256-GCM", 32, "data")
	if err == nil {
		em.keyManager.keys[dataKey.ID] = dataKey
		em.keyManager.activeKeyID = dataKey.ID
	}

	// Create authentication signing key
	authKey, err := em.GenerateKey("RSA-2048", 2048, "auth")
	if err == nil {
		em.keyManager.keys[authKey.ID] = authKey
	}

	// Create field encryption keys
	fieldKey, err := em.GenerateKey("AES-256-GCM", 32, "field")
	if err == nil {
		em.fieldEncryption.encryptionKeys[fieldKey.ID] = fieldKey
	}
}

// GenerateKey generates a new encryption key
func (em *EncryptionManager) GenerateKey(algorithm string, keySize int, purpose string) (*EncryptionKey, error) {
	keyID := generateKeyID()
	var keyData []byte
	var publicKey []byte
	var err error

	switch algorithm {
	case "AES-256-GCM", "AES-128-GCM":
		keyData = make([]byte, keySize)
		if _, err = rand.Read(keyData); err != nil {
			return nil, fmt.Errorf("failed to generate AES key: %w", err)
		}

	case "RSA-2048", "RSA-4096":
		rsaKey, err := rsa.GenerateKey(rand.Reader, keySize)
		if err != nil {
			return nil, fmt.Errorf("failed to generate RSA key: %w", err)
		}
		keyData = x509.MarshalPKCS1PrivateKey(rsaKey)
		publicKeyBytes, err := x509.MarshalPKIXPublicKey(&rsaKey.PublicKey)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal public key: %w", err)
		}
		publicKey = publicKeyBytes

	default:
		return nil, fmt.Errorf("unsupported algorithm: %s", algorithm)
	}

	key := &EncryptionKey{
		ID:        keyID,
		Algorithm: algorithm,
		KeySize:   keySize,
		Purpose:   purpose,
		Status:    KeyStatusActive,
		KeyData:   keyData,
		PublicKey: publicKey,
		CreatedAt: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	// Set expiration based on key type
	if purpose == "data" {
		expiresAt := time.Now().Add(time.Duration(em.config.KeyManagement.KeyRotation) * 24 * time.Hour)
		key.ExpiresAt = &expiresAt
	}

	// Store key securely
	if err := em.keyManager.keyStorage.StoreKey(key); err != nil {
		return nil, fmt.Errorf("failed to store key: %w", err)
	}

	return key, nil
}

// Encrypt encrypts data using the active encryption key
func (em *EncryptionManager) Encrypt(plaintext []byte, keyID ...string) (*EncryptedData, error) {
	em.keyManager.mu.RLock()
	
	var key *EncryptionKey
	var selectedKeyID string
	
	if len(keyID) > 0 && keyID[0] != "" {
		selectedKeyID = keyID[0]
		var exists bool
		key, exists = em.keyManager.keys[selectedKeyID]
		if !exists {
			em.keyManager.mu.RUnlock()
			return nil, fmt.Errorf("key not found: %s", selectedKeyID)
		}
	} else {
		selectedKeyID = em.keyManager.activeKeyID
		key = em.keyManager.keys[selectedKeyID]
		if key == nil {
			em.keyManager.mu.RUnlock()
			return nil, fmt.Errorf("no active encryption key available")
		}
	}
	
	em.keyManager.mu.RUnlock()

	if key.Status != KeyStatusActive {
		return nil, fmt.Errorf("key is not active: %s", selectedKeyID)
	}

	switch key.Algorithm {
	case "AES-256-GCM", "AES-128-GCM":
		return em.encryptAESGCM(plaintext, key)
	default:
		return nil, fmt.Errorf("unsupported encryption algorithm: %s", key.Algorithm)
	}
}

// Decrypt decrypts data using the specified key
func (em *EncryptionManager) Decrypt(encryptedData *EncryptedData) ([]byte, error) {
	em.keyManager.mu.RLock()
	key, exists := em.keyManager.keys[encryptedData.KeyID]
	em.keyManager.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("encryption key not found: %s", encryptedData.KeyID)
	}

	switch encryptedData.Algorithm {
	case "AES-256-GCM", "AES-128-GCM":
		return em.decryptAESGCM(encryptedData, key)
	default:
		return nil, fmt.Errorf("unsupported encryption algorithm: %s", encryptedData.Algorithm)
	}
}

// encryptAESGCM performs AES-GCM encryption
func (em *EncryptionManager) encryptAESGCM(plaintext []byte, key *EncryptionKey) (*EncryptedData, error) {
	block, err := aes.NewCipher(key.KeyData)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create AEAD: %w", err)
	}

	// Generate random nonce
	nonce := make([]byte, aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	// Encrypt data
	ciphertext := aead.Seal(nil, nonce, plaintext, nil)

	return &EncryptedData{
		Data:        ciphertext,
		Algorithm:   key.Algorithm,
		KeyID:       key.ID,
		IV:          nonce,
		EncryptedAt: time.Now(),
		Metadata:    map[string]string{"version": "1.0"},
	}, nil
}

// decryptAESGCM performs AES-GCM decryption
func (em *EncryptionManager) decryptAESGCM(encryptedData *EncryptedData, key *EncryptionKey) ([]byte, error) {
	block, err := aes.NewCipher(key.KeyData)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create AEAD: %w", err)
	}

	// Decrypt data
	plaintext, err := aead.Open(nil, encryptedData.IV, encryptedData.Data, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt data: %w", err)
	}

	return plaintext, nil
}

// EncryptField encrypts a database field
func (em *EncryptionManager) EncryptField(tableName, columnName string, value []byte) (*EncryptedData, error) {
	em.fieldEncryption.mu.RLock()
	
	mappingKey := fmt.Sprintf("%s.%s", tableName, columnName)
	mapping, exists := em.fieldEncryption.fieldMappings[mappingKey]
	
	em.fieldEncryption.mu.RUnlock()

	if !exists {
		// Use default field encryption key
		return em.Encrypt(value)
	}

	key, exists := em.fieldEncryption.encryptionKeys[mapping.KeyID]
	if !exists {
		return nil, fmt.Errorf("field encryption key not found: %s", mapping.KeyID)
	}

	if mapping.Tokenization {
		return em.tokenizeField(value, key)
	}

	return em.Encrypt(value, key.ID)
}

// DecryptField decrypts a database field
func (em *EncryptionManager) DecryptField(encryptedData *EncryptedData) ([]byte, error) {
	return em.Decrypt(encryptedData)
}

// tokenizeField performs format-preserving tokenization
func (em *EncryptionManager) tokenizeField(value []byte, key *EncryptionKey) (*EncryptedData, error) {
	// Simple tokenization implementation
	// In production, use format-preserving encryption (FPE)
	hash := sha256.Sum256(append(key.KeyData, value...))
	token := base64.URLEncoding.EncodeToString(hash[:])

	return &EncryptedData{
		Data:        []byte(token),
		Algorithm:   "FPE-SHA256",
		KeyID:       key.ID,
		EncryptedAt: time.Now(),
		Metadata:    map[string]string{"tokenized": "true"},
	}, nil
}

// RotateKey performs key rotation
func (em *EncryptionManager) RotateKey(keyID string) (*EncryptionKey, error) {
	em.keyManager.mu.Lock()
	defer em.keyManager.mu.Unlock()

	oldKey, exists := em.keyManager.keys[keyID]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", keyID)
	}

	// Generate new key with same parameters
	newKey, err := em.GenerateKey(oldKey.Algorithm, oldKey.KeySize, oldKey.Purpose)
	if err != nil {
		return nil, fmt.Errorf("failed to generate new key: %w", err)
	}

	// Set rotation metadata
	newKey.RotatedFrom = oldKey.ID
	oldKey.Status = KeyStatusInactive

	// Update active key if this was the active key
	if em.keyManager.activeKeyID == keyID {
		em.keyManager.activeKeyID = newKey.ID
	}

	// Store new key
	em.keyManager.keys[newKey.ID] = newKey

	return newKey, nil
}

// startKeyRotationScheduler starts automatic key rotation
func (em *EncryptionManager) startKeyRotationScheduler() {
	ticker := time.NewTicker(24 * time.Hour) // Check daily
	defer ticker.Stop()

	for range ticker.C {
		em.keyManager.mu.RLock()
		for keyID, key := range em.keyManager.keys {
			if key.Status == KeyStatusActive && key.ExpiresAt != nil && time.Now().After(*key.ExpiresAt) {
				em.keyManager.mu.RUnlock()
				
				// Rotate expired key
				if _, err := em.RotateKey(keyID); err != nil {
					fmt.Printf("Failed to rotate key %s: %v\n", keyID, err)
				} else {
					fmt.Printf("Successfully rotated key %s\n", keyID)
				}
				
				em.keyManager.mu.RLock()
			}
		}
		em.keyManager.mu.RUnlock()
	}
}

// Certificate Management

// GenerateCertificate generates a new TLS certificate
func (em *EncryptionManager) GenerateCertificate(commonName string, sans []string) (*CertificateInfo, error) {
	// Generate private key
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	// Create certificate template
	template := x509.Certificate{
		SerialNumber:          generateSerial(),
		Subject:               x509.Name{CommonName: commonName},
		NotBefore:            time.Now(),
		NotAfter:             time.Now().Add(365 * 24 * time.Hour), // 1 year
		KeyUsage:             x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:          []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	// Add SANs
	for _, san := range sans {
		template.DNSNames = append(template.DNSNames, san)
	}

	// Self-sign the certificate (in production, use proper CA)
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %w", err)
	}

	// Encode certificate and key
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})

	certInfo := &CertificateInfo{
		ID:          generateCertID(),
		CommonName:  commonName,
		SANs:        sans,
		Certificate: certPEM,
		PrivateKey:  keyPEM,
		IssuedAt:    time.Now(),
		ExpiresAt:   time.Now().Add(365 * 24 * time.Hour),
		Issuer:      "NovaCron-Self-Signed",
		KeyUsage:    []string{"digital_signature", "key_encipherment", "server_auth"},
		Status:      KeyStatusActive,
	}

	em.certManager.mu.Lock()
	em.certManager.certificates[certInfo.ID] = certInfo
	em.certManager.mu.Unlock()

	return certInfo, nil
}

// getCertificate returns certificate for TLS handshake
func (em *EncryptionManager) getCertificate(hello *tls.ClientHelloInfo) (*tls.Certificate, error) {
	em.certManager.mu.RLock()
	defer em.certManager.mu.RUnlock()

	// Find matching certificate
	for _, certInfo := range em.certManager.certificates {
		if certInfo.Status != KeyStatusActive {
			continue
		}

		// Check if certificate matches the requested server name
		if hello.ServerName == certInfo.CommonName {
			cert, err := tls.X509KeyPair(certInfo.Certificate, certInfo.PrivateKey)
			if err != nil {
				continue
			}
			return &cert, nil
		}

		// Check SANs
		for _, san := range certInfo.SANs {
			if hello.ServerName == san {
				cert, err := tls.X509KeyPair(certInfo.Certificate, certInfo.PrivateKey)
				if err != nil {
					continue
				}
				return &cert, nil
			}
		}
	}

	return nil, fmt.Errorf("no matching certificate found for %s", hello.ServerName)
}

// getClientCertificate returns client certificate for mutual TLS
func (em *EncryptionManager) getClientCertificate(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
	em.certManager.mu.RLock()
	defer em.certManager.mu.RUnlock()

	// Return first available client certificate
	for _, certInfo := range em.certManager.certificates {
		if certInfo.Status == KeyStatusActive {
			cert, err := tls.X509KeyPair(certInfo.Certificate, certInfo.PrivateKey)
			if err != nil {
				continue
			}
			return &cert, nil
		}
	}

	return nil, fmt.Errorf("no client certificate available")
}

// startCertRenewalScheduler starts automatic certificate renewal
func (em *EncryptionManager) startCertRenewalScheduler() {
	ticker := time.NewTicker(24 * time.Hour) // Check daily
	defer ticker.Stop()

	for range ticker.C {
		if !em.certManager.autoRenewal {
			continue
		}

		em.certManager.mu.RLock()
		for _, certInfo := range em.certManager.certificates {
			// Renew certificates that expire within 30 days
			if certInfo.Status == KeyStatusActive && time.Until(certInfo.ExpiresAt) < 30*24*time.Hour {
				em.certManager.mu.RUnlock()
				
				// Renew certificate
				newCert, err := em.GenerateCertificate(certInfo.CommonName, certInfo.SANs)
				if err != nil {
					fmt.Printf("Failed to renew certificate %s: %v\n", certInfo.ID, err)
				} else {
					// Mark old certificate as inactive
					em.certManager.mu.Lock()
					certInfo.Status = KeyStatusInactive
					em.certManager.mu.Unlock()
					fmt.Printf("Successfully renewed certificate %s\n", newCert.ID)
				}
				
				em.certManager.mu.RLock()
			}
		}
		em.certManager.mu.RUnlock()
	}
}

// GetTLSConfig returns the TLS configuration
func (em *EncryptionManager) GetTLSConfig() *tls.Config {
	return em.tlsConfig
}

// Storage interface implementations (placeholder)
type VaultClient interface {
	StoreKey(key *EncryptionKey) error
	RetrieveKey(keyID string) (*EncryptionKey, error)
	DeleteKey(keyID string) error
}

type HSMClient interface {
	GenerateKey(algorithm string, keySize int) (*EncryptionKey, error)
	Encrypt(keyID string, plaintext []byte) ([]byte, error)
	Decrypt(keyID string, ciphertext []byte) ([]byte, error)
}

type KMSClient interface {
	CreateKey(keySpec string) (string, error)
	Encrypt(keyID string, plaintext []byte) ([]byte, error)
	Decrypt(keyID string, ciphertext []byte) ([]byte, error)
}

// StoreKey stores an encryption key securely
func (ks *KeyStorage) StoreKey(key *EncryptionKey) error {
	switch ks.storageType {
	case "vault":
		if ks.vaultClient != nil {
			return ks.vaultClient.StoreKey(key)
		}
		// Fallback to local encrypted storage
		return ks.storeKeyLocally(key)
	case "hsm":
		// HSM storage implementation
		return fmt.Errorf("HSM storage not implemented")
	case "kms":
		// KMS storage implementation
		return fmt.Errorf("KMS storage not implemented")
	default:
		return ks.storeKeyLocally(key)
	}
}

// storeKeyLocally stores key with local encryption
func (ks *KeyStorage) storeKeyLocally(key *EncryptionKey) error {
	// In production, encrypt key data before storage
	// For now, this is a placeholder
	return nil
}

// Helper functions
func generateKeyID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("key-%x", b)
}

func generateCertID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("cert-%x", b)
}

func generateSerial() *big.Int {
	max := new(big.Int)
	max.Exp(big.NewInt(2), big.NewInt(130), nil).Sub(max, big.NewInt(1))
	n, _ := rand.Int(rand.Reader, max)
	return n
}

// Import for big.Int
import "math/big"