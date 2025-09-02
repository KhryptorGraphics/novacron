package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"errors"
	"fmt"
	"time"
	"io"
	"sync"
)

const (
	// DefaultKeySize is the default key size for RSA keys
	DefaultKeySize = 2048

	// DefaultAESKeySize is the default key size for AES keys
	DefaultAESKeySize = 32 // 256 bits
)

// EncryptionType defines the type of encryption
type EncryptionType string

const (
	// AESEncryption represents AES encryption
	AESEncryption EncryptionType = "aes"

	// RSAEncryption represents RSA encryption
	RSAEncryption EncryptionType = "rsa"
)

// EncryptionKey represents an encryption key
type EncryptionKey struct {
	// ID is the unique identifier of the key
	ID string

	// Name is the human-readable name of the key
	Name string

	// Type is the type of encryption this key is used for
	Type EncryptionType

	// Data is the key data
	Data []byte

	// RSAKey is the RSA key if Type is RSAEncryption
	RSAKey *rsa.PrivateKey

	// Created is the time the key was created
	Created int64

	// Expires is the time the key expires
	Expires int64

	// Metadata is additional metadata for the key
	Metadata map[string]string
}

// EncryptionManager manages encryption and decryption operations
type EncryptionManager struct {
	// keys is a map of key ID to key
	keys map[string]*EncryptionKey

	// defaultKeyID is the ID of the default key
	defaultKeyID string

	// mutex protects the keys map and defaultKeyID
	mutex sync.RWMutex
}

// NewEncryptionManager creates a new encryption manager
func NewEncryptionManager() *EncryptionManager {
	return &EncryptionManager{
		keys: make(map[string]*EncryptionKey),
	}
}

// AddKey adds a key to the manager
func (m *EncryptionManager) AddKey(key *EncryptionKey) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if key already exists
	if _, exists := m.keys[key.ID]; exists {
		return fmt.Errorf("key with ID %s already exists", key.ID)
	}

	// Add key
	m.keys[key.ID] = key

	// If this is the first key, set it as default
	if len(m.keys) == 1 {
		m.defaultKeyID = key.ID
	}

	return nil
}

// RemoveKey removes a key from the manager
func (m *EncryptionManager) RemoveKey(keyID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if key exists
	if _, exists := m.keys[keyID]; !exists {
		return fmt.Errorf("key with ID %s does not exist", keyID)
	}

	// Remove key
	delete(m.keys, keyID)

	// If this was the default key, set a new default
	if m.defaultKeyID == keyID {
		if len(m.keys) > 0 {
			// Set any key as default
			for id := range m.keys {
				m.defaultKeyID = id
				break
			}
		} else {
			m.defaultKeyID = ""
		}
	}

	return nil
}

// GetKey gets a key by ID
func (m *EncryptionManager) GetKey(keyID string) (*EncryptionKey, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if key exists
	key, exists := m.keys[keyID]
	if !exists {
		return nil, fmt.Errorf("key with ID %s does not exist", keyID)
	}

	return key, nil
}

// SetDefaultKey sets the default key
func (m *EncryptionManager) SetDefaultKey(keyID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if key exists
	if _, exists := m.keys[keyID]; !exists {
		return fmt.Errorf("key with ID %s does not exist", keyID)
	}

	// Set default key
	m.defaultKeyID = keyID

	return nil
}

// GetDefaultKey gets the default key
func (m *EncryptionManager) GetDefaultKey() (*EncryptionKey, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if default key is set
	if m.defaultKeyID == "" {
		return nil, errors.New("no default key set")
	}

	// Check if key exists
	key, exists := m.keys[m.defaultKeyID]
	if !exists {
		return nil, fmt.Errorf("default key with ID %s does not exist", m.defaultKeyID)
	}

	return key, nil
}

// CreateAESKey creates a new AES key
func (m *EncryptionManager) CreateAESKey(id, name string, keySize int) (*EncryptionKey, error) {
	if keySize <= 0 {
		keySize = DefaultAESKeySize
	}

	// Generate key
	key := make([]byte, keySize)
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		return nil, fmt.Errorf("failed to generate AES key: %w", err)
	}

	// Create encryption key
	encKey := &EncryptionKey{
		ID:       id,
		Name:     name,
		Type:     AESEncryption,
		Data:     key,
		Created:  time.Now().Unix(),
		Expires:  0, // No expiration
		Metadata: make(map[string]string),
	}

	// Add key
	if err := m.AddKey(encKey); err != nil {
		return nil, err
	}

	return encKey, nil
}

// CreateRSAKey creates a new RSA key
func (m *EncryptionManager) CreateRSAKey(id, name string, keySize int) (*EncryptionKey, error) {
	if keySize <= 0 {
		keySize = DefaultKeySize
	}

	// Generate key
	privateKey, err := rsa.GenerateKey(rand.Reader, keySize)
	if err != nil {
		return nil, fmt.Errorf("failed to generate RSA key: %w", err)
	}

	// Create encryption key
	encKey := &EncryptionKey{
		ID:       id,
		Name:     name,
		Type:     RSAEncryption,
		RSAKey:   privateKey,
		Created:  time.Now().Unix(),
		Expires:  0, // No expiration
		Metadata: make(map[string]string),
	}

	// Add key
	if err := m.AddKey(encKey); err != nil {
		return nil, err
	}

	return encKey, nil
}

// EncryptWithAES encrypts data with AES-GCM
func (m *EncryptionManager) EncryptWithAES(data []byte, keyID string) ([]byte, error) {
	// Get key
	var key *EncryptionKey
	var err error
	if keyID == "" {
		key, err = m.GetDefaultKey()
	} else {
		key, err = m.GetKey(keyID)
	}
	if err != nil {
		return nil, err
	}

	// Check key type
	if key.Type != AESEncryption {
		return nil, fmt.Errorf("key %s is not an AES key", key.ID)
	}

	// Create AES cipher
	block, err := aes.NewCipher(key.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM mode: %w", err)
	}

	// Create nonce
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to create nonce: %w", err)
	}

	// Encrypt data
	ciphertext := gcm.Seal(nonce, nonce, data, nil)

	return ciphertext, nil
}

// DecryptWithAES decrypts data with AES-GCM
func (m *EncryptionManager) DecryptWithAES(data []byte, keyID string) ([]byte, error) {
	// Get key
	var key *EncryptionKey
	var err error
	if keyID == "" {
		key, err = m.GetDefaultKey()
	} else {
		key, err = m.GetKey(keyID)
	}
	if err != nil {
		return nil, err
	}

	// Check key type
	if key.Type != AESEncryption {
		return nil, fmt.Errorf("key %s is not an AES key", key.ID)
	}

	// Create AES cipher
	block, err := aes.NewCipher(key.Data)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM mode: %w", err)
	}

	// Check data length
	if len(data) < gcm.NonceSize() {
		return nil, errors.New("ciphertext too short")
	}

	// Extract nonce and ciphertext
	nonce, ciphertext := data[:gcm.NonceSize()], data[gcm.NonceSize():]

	// Decrypt data
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt data: %w", err)
	}

	return plaintext, nil
}

// EncryptWithRSA encrypts data with RSA-OAEP
func (m *EncryptionManager) EncryptWithRSA(data []byte, keyID string) ([]byte, error) {
	// Get key
	var key *EncryptionKey
	var err error
	if keyID == "" {
		key, err = m.GetDefaultKey()
	} else {
		key, err = m.GetKey(keyID)
	}
	if err != nil {
		return nil, err
	}

	// Check key type
	if key.Type != RSAEncryption {
		return nil, fmt.Errorf("key %s is not an RSA key", key.ID)
	}

	// Encrypt data
	ciphertext, err := rsa.EncryptOAEP(sha256.New(), rand.Reader, &key.RSAKey.PublicKey, data, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt data: %w", err)
	}

	return ciphertext, nil
}

// DecryptWithRSA decrypts data with RSA-OAEP
func (m *EncryptionManager) DecryptWithRSA(data []byte, keyID string) ([]byte, error) {
	// Get key
	var key *EncryptionKey
	var err error
	if keyID == "" {
		key, err = m.GetDefaultKey()
	} else {
		key, err = m.GetKey(keyID)
	}
	if err != nil {
		return nil, err
	}

	// Check key type
	if key.Type != RSAEncryption {
		return nil, fmt.Errorf("key %s is not an RSA key", key.ID)
	}

	// Decrypt data
	plaintext, err := rsa.DecryptOAEP(sha256.New(), rand.Reader, key.RSAKey, data, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt data: %w", err)
	}

	return plaintext, nil
}

// EncryptField encrypts a field with the appropriate encryption method
func (m *EncryptionManager) EncryptField(data []byte, keyID string) ([]byte, error) {
	// Get key
	var key *EncryptionKey
	var err error
	if keyID == "" {
		key, err = m.GetDefaultKey()
	} else {
		key, err = m.GetKey(keyID)
	}
	if err != nil {
		return nil, err
	}

	// Encrypt based on key type
	switch key.Type {
	case AESEncryption:
		return m.EncryptWithAES(data, key.ID)
	case RSAEncryption:
		return m.EncryptWithRSA(data, key.ID)
	default:
		return nil, fmt.Errorf("unsupported encryption type: %s", key.Type)
	}
}

// DecryptField decrypts a field with the appropriate encryption method
func (m *EncryptionManager) DecryptField(data []byte, keyID string) ([]byte, error) {
	// Get key
	var key *EncryptionKey
	var err error
	if keyID == "" {
		key, err = m.GetDefaultKey()
	} else {
		key, err = m.GetKey(keyID)
	}
	if err != nil {
		return nil, err
	}

	// Decrypt based on key type
	switch key.Type {
	case AESEncryption:
		return m.DecryptWithAES(data, key.ID)
	case RSAEncryption:
		return m.DecryptWithRSA(data, key.ID)
	default:
		return nil, fmt.Errorf("unsupported encryption type: %s", key.Type)
	}
}

// ListKeys lists all keys
func (m *EncryptionManager) ListKeys() []*EncryptionKey {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	keys := make([]*EncryptionKey, 0, len(m.keys))
	for _, key := range m.keys {
		keys = append(keys, key)
	}

	return keys
}

// GenerateRandomBytes generates random bytes
func GenerateRandomBytes(length int) ([]byte, error) {
	bytes := make([]byte, length)
	if _, err := io.ReadFull(rand.Reader, bytes); err != nil {
		return nil, err
	}
	return bytes, nil
}
