package encryption

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
)

// EncryptionAlgorithm represents the algorithm used for encryption
type EncryptionAlgorithm string

const (
	// EncryptionNone indicates no encryption should be used
	EncryptionNone EncryptionAlgorithm = "none"

	// EncryptionAES256 uses AES-256 encryption
	EncryptionAES256 EncryptionAlgorithm = "aes256"

	// EncryptionAES192 uses AES-192 encryption
	EncryptionAES192 EncryptionAlgorithm = "aes192"

	// EncryptionAES128 uses AES-128 encryption
	EncryptionAES128 EncryptionAlgorithm = "aes128"
)

// EncryptionMode represents the encryption mode
type EncryptionMode string

const (
	// EncryptionModeGCM uses Galois/Counter Mode
	EncryptionModeGCM EncryptionMode = "gcm"

	// EncryptionModeCBC uses Cipher Block Chaining
	EncryptionModeCBC EncryptionMode = "cbc"

	// EncryptionModeCTR uses Counter Mode
	EncryptionModeCTR EncryptionMode = "ctr"
)

// EncryptionConfig contains configuration for data encryption
type EncryptionConfig struct {
	// Algorithm to use for encryption
	Algorithm EncryptionAlgorithm `json:"algorithm"`

	// Mode to use for encryption
	Mode EncryptionMode `json:"mode"`

	// Master key for encryption (will be used to derive per-volume keys)
	MasterKey string `json:"master_key"`

	// Per-volume salt prefix to prevent rainbow table attacks
	SaltPrefix string `json:"salt_prefix"`

	// Whether to authenticate encrypted data
	Authenticate bool `json:"authenticate"`

	// Minimum size in bytes before encryption is applied
	MinSizeBytes int `json:"min_size_bytes"`

	// Whether to include key verification data
	IncludeVerification bool `json:"include_verification"`

	// Whether to store metadata alongside the encrypted data
	StoreMetadata bool `json:"store_metadata"`
}

// DefaultEncryptionConfig returns a default encryption configuration
func DefaultEncryptionConfig() EncryptionConfig {
	return EncryptionConfig{
		Algorithm:           EncryptionAES256,
		Mode:                EncryptionModeGCM,
		MasterKey:           "", // Must be provided by the user
		SaltPrefix:          "novacron",
		Authenticate:        true,
		MinSizeBytes:        1,
		IncludeVerification: true,
		StoreMetadata:       true,
	}
}

// EncryptedData represents data that has been encrypted
type EncryptedData struct {
	// The encrypted data bytes
	Data []byte `json:"data"`

	// The algorithm used for encryption
	Algorithm EncryptionAlgorithm `json:"algorithm"`

	// The mode used for encryption
	Mode EncryptionMode `json:"mode"`

	// The initialization vector used for encryption
	IV []byte `json:"iv"`

	// The authentication tag (for authenticated encryption modes)
	AuthTag []byte `json:"auth_tag,omitempty"`

	// The original size of the data before encryption
	OriginalSize int `json:"original_size"`

	// Verification data to confirm decryption key correctness
	Verification []byte `json:"verification,omitempty"`
}

// Encryptor provides methods for encrypting and decrypting data
type Encryptor struct {
	config EncryptionConfig
}

// NewEncryptor creates a new Encryptor with the provided configuration
func NewEncryptor(config EncryptionConfig) (*Encryptor, error) {
	if config.Algorithm != EncryptionNone && config.MasterKey == "" {
		return nil, errors.New("master key is required for encryption")
	}

	return &Encryptor{
		config: config,
	}, nil
}

// GenerateKey generates a derived key for a specific volume
func (e *Encryptor) GenerateKey(volumeID string) ([]byte, error) {
	if e.config.Algorithm == EncryptionNone {
		return nil, nil
	}

	// Use the salt prefix, volume ID, and master key to derive a unique key
	salt := fmt.Sprintf("%s-%s", e.config.SaltPrefix, volumeID)
	combined := salt + e.config.MasterKey

	// Use SHA-256 to derive a key of appropriate length
	hash := sha256.Sum256([]byte(combined))

	// Return a key of appropriate length based on the algorithm
	switch e.config.Algorithm {
	case EncryptionAES256:
		return hash[:32], nil // 256 bits = 32 bytes
	case EncryptionAES192:
		return hash[:24], nil // 192 bits = 24 bytes
	case EncryptionAES128:
		return hash[:16], nil // 128 bits = 16 bytes
	default:
		return nil, fmt.Errorf("unsupported encryption algorithm: %s", e.config.Algorithm)
	}
}

// Encrypt encrypts the provided data using the configured algorithm and mode
func (e *Encryptor) Encrypt(data []byte, volumeID string) (*EncryptedData, error) {
	if e.config.Algorithm == EncryptionNone {
		return &EncryptedData{
			Data:         data,
			Algorithm:    EncryptionNone,
			OriginalSize: len(data),
		}, nil
	}

	// Check if data meets minimum size requirements
	if len(data) < e.config.MinSizeBytes {
		return &EncryptedData{
			Data:         data,
			Algorithm:    EncryptionNone,
			OriginalSize: len(data),
		}, nil
	}

	// Generate a key for this volume
	key, err := e.GenerateKey(volumeID)
	if err != nil {
		return nil, err
	}

	// Create a new cipher block
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	// Create encrypted data object
	encData := &EncryptedData{
		Algorithm:    e.config.Algorithm,
		Mode:         e.config.Mode,
		OriginalSize: len(data),
	}

	// Encrypt based on the configured mode
	switch e.config.Mode {
	case EncryptionModeGCM:
		return e.encryptGCM(data, block, encData)
	case EncryptionModeCBC:
		return e.encryptCBC(data, block, encData)
	case EncryptionModeCTR:
		return e.encryptCTR(data, block, encData)
	default:
		return nil, fmt.Errorf("unsupported encryption mode: %s", e.config.Mode)
	}
}

// encryptGCM encrypts data using GCM mode
func (e *Encryptor) encryptGCM(data []byte, block cipher.Block, encData *EncryptedData) (*EncryptedData, error) {
	// Create a new GCM cipher
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Create a nonce
	nonce := make([]byte, gcm.NonceSize())
	if _, err = io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	// Encrypt the data
	encData.Data = gcm.Seal(nil, nonce, data, nil)
	encData.IV = nonce

	// Add verification if configured
	if e.config.IncludeVerification {
		// Use a fixed plaintext as verification
		verification := []byte("VERIFIED")
		encData.Verification = gcm.Seal(nil, nonce, verification, nil)
	}

	return encData, nil
}

// encryptCBC encrypts data using CBC mode
func (e *Encryptor) encryptCBC(data []byte, block cipher.Block, encData *EncryptedData) (*EncryptedData, error) {
	// Create IV
	iv := make([]byte, aes.BlockSize)
	if _, err := io.ReadFull(rand.Reader, iv); err != nil {
		return nil, err
	}

	// Pad data to a multiple of the block size
	paddedData := padData(data, aes.BlockSize)

	// Encrypt the data
	ciphertext := make([]byte, len(paddedData))
	mode := cipher.NewCBCEncrypter(block, iv)
	mode.CryptBlocks(ciphertext, paddedData)

	encData.Data = ciphertext
	encData.IV = iv

	// Add verification if configured
	if e.config.IncludeVerification {
		// Use a fixed plaintext as verification
		verification := []byte("VERIFIED")
		paddedVerification := padData(verification, aes.BlockSize)
		verificationCiphertext := make([]byte, len(paddedVerification))
		mode.CryptBlocks(verificationCiphertext, paddedVerification)
		encData.Verification = verificationCiphertext
	}

	return encData, nil
}

// encryptCTR encrypts data using CTR mode
func (e *Encryptor) encryptCTR(data []byte, block cipher.Block, encData *EncryptedData) (*EncryptedData, error) {
	// Create IV
	iv := make([]byte, aes.BlockSize)
	if _, err := io.ReadFull(rand.Reader, iv); err != nil {
		return nil, err
	}

	// Encrypt the data
	ciphertext := make([]byte, len(data))
	stream := cipher.NewCTR(block, iv)
	stream.XORKeyStream(ciphertext, data)

	encData.Data = ciphertext
	encData.IV = iv

	// Add verification if configured
	if e.config.IncludeVerification {
		// Use a fixed plaintext as verification
		verification := []byte("VERIFIED")
		verificationCiphertext := make([]byte, len(verification))
		verificationStream := cipher.NewCTR(block, iv)
		verificationStream.XORKeyStream(verificationCiphertext, verification)
		encData.Verification = verificationCiphertext
	}

	return encData, nil
}

// Decrypt decrypts the provided encrypted data
func (e *Encryptor) Decrypt(encData *EncryptedData, volumeID string) ([]byte, error) {
	// If data wasn't encrypted, return as is
	if encData.Algorithm == EncryptionNone {
		return encData.Data, nil
	}

	// Generate a key for this volume
	key, err := e.GenerateKey(volumeID)
	if err != nil {
		return nil, err
	}

	// Create a new cipher block
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	// Decrypt based on the mode
	switch encData.Mode {
	case EncryptionModeGCM:
		return e.decryptGCM(encData, block)
	case EncryptionModeCBC:
		return e.decryptCBC(encData, block)
	case EncryptionModeCTR:
		return e.decryptCTR(encData, block)
	default:
		return nil, fmt.Errorf("unsupported encryption mode: %s", encData.Mode)
	}
}

// decryptGCM decrypts data using GCM mode
func (e *Encryptor) decryptGCM(encData *EncryptedData, block cipher.Block) ([]byte, error) {
	// Create a new GCM cipher
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Check nonce size
	if len(encData.IV) != gcm.NonceSize() {
		return nil, fmt.Errorf("invalid nonce size")
	}

	// Verify the key if verification data is included
	if e.config.IncludeVerification && len(encData.Verification) > 0 {
		// Try to decrypt the verification data
		_, err := gcm.Open(nil, encData.IV, encData.Verification, nil)
		if err != nil {
			return nil, fmt.Errorf("key verification failed: %w", err)
		}
	}

	// Decrypt the data
	plaintext, err := gcm.Open(nil, encData.IV, encData.Data, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

// decryptCBC decrypts data using CBC mode
func (e *Encryptor) decryptCBC(encData *EncryptedData, block cipher.Block) ([]byte, error) {
	// Check IV size
	if len(encData.IV) != aes.BlockSize {
		return nil, fmt.Errorf("invalid IV size")
	}

	// Verify the key if verification data is included
	if e.config.IncludeVerification && len(encData.Verification) > 0 {
		// Create a new CBC decrypter for verification
		mode := cipher.NewCBCDecrypter(block, encData.IV)
		verificationPlaintext := make([]byte, len(encData.Verification))
		mode.CryptBlocks(verificationPlaintext, encData.Verification)
		verificationPlaintext = unpadData(verificationPlaintext)
		if string(verificationPlaintext) != "VERIFIED" {
			return nil, fmt.Errorf("key verification failed")
		}
	}

	// Create a new CBC decrypter
	mode := cipher.NewCBCDecrypter(block, encData.IV)
	plaintext := make([]byte, len(encData.Data))
	mode.CryptBlocks(plaintext, encData.Data)

	// Unpad the data
	plaintext = unpadData(plaintext)

	return plaintext, nil
}

// decryptCTR decrypts data using CTR mode
func (e *Encryptor) decryptCTR(encData *EncryptedData, block cipher.Block) ([]byte, error) {
	// Check IV size
	if len(encData.IV) != aes.BlockSize {
		return nil, fmt.Errorf("invalid IV size")
	}

	// Verify the key if verification data is included
	if e.config.IncludeVerification && len(encData.Verification) > 0 {
		// Create a new CTR stream for verification
		verificationStream := cipher.NewCTR(block, encData.IV)
		verificationPlaintext := make([]byte, len(encData.Verification))
		verificationStream.XORKeyStream(verificationPlaintext, encData.Verification)
		if string(verificationPlaintext) != "VERIFIED" {
			return nil, fmt.Errorf("key verification failed")
		}
	}

	// Create a new CTR stream
	stream := cipher.NewCTR(block, encData.IV)
	plaintext := make([]byte, len(encData.Data))
	stream.XORKeyStream(plaintext, encData.Data)

	return plaintext, nil
}

// padData pads data to a multiple of blockSize using PKCS#7 padding
func padData(data []byte, blockSize int) []byte {
	padLength := blockSize - (len(data) % blockSize)
	padding := make([]byte, padLength)
	for i := range padding {
		padding[i] = byte(padLength)
	}
	return append(data, padding...)
}

// unpadData removes PKCS#7 padding from data
func unpadData(data []byte) []byte {
	if len(data) == 0 {
		return data
	}

	padLength := int(data[len(data)-1])
	if padLength > len(data) {
		// Invalid padding, return data as is
		return data
	}

	return data[:len(data)-padLength]
}

// GenerateRandomKey generates a random key of the specified length
func GenerateRandomKey(length int) (string, error) {
	key := make([]byte, length)
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		return "", err
	}

	// Convert to a hex string for easier storage
	return fmt.Sprintf("%x", key), nil
}

// IsValidKey checks if a key is valid for the specified algorithm
func IsValidKey(key string, algorithm EncryptionAlgorithm) bool {
	// Convert hex string to bytes
	keyBytes := make([]byte, len(key)/2)
	for i := 0; i < len(key); i += 2 {
		if i+1 >= len(key) {
			return false
		}
		var value int
		if _, err := fmt.Sscanf(key[i:i+2], "%x", &value); err != nil {
			return false
		}
		keyBytes[i/2] = byte(value)
	}

	// Check key length
	switch algorithm {
	case EncryptionAES256:
		return len(keyBytes) >= 32
	case EncryptionAES192:
		return len(keyBytes) >= 24
	case EncryptionAES128:
		return len(keyBytes) >= 16
	case EncryptionNone:
		return true
	default:
		return false
	}
}
