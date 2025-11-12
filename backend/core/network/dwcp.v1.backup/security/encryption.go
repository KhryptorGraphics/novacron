package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"

	"go.uber.org/zap"
	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/pbkdf2"
)

// KeyDerivationMethod defines the key derivation algorithm
type KeyDerivationMethod string

const (
	// Argon2ID is the recommended key derivation function
	Argon2ID KeyDerivationMethod = "argon2id"
	// PBKDF2 is a fallback key derivation function
	PBKDF2 KeyDerivationMethod = "pbkdf2"
)

// EncryptionConfig holds encryption configuration
type EncryptionConfig struct {
	Algorithm       string
	KeyDerivation   KeyDerivationMethod
	KeyLength       int
	Argon2Time      uint32
	Argon2Memory    uint32
	Argon2Threads   uint8
	PBKDF2Iterations int
}

// DataEncryptor handles data encryption/decryption
type DataEncryptor struct {
	key           []byte
	gcm           cipher.AEAD
	config        EncryptionConfig
	logger        *zap.Logger
	keyDerivation KeyDerivationMethod
}

// DefaultEncryptionConfig returns secure default encryption configuration
func DefaultEncryptionConfig() EncryptionConfig {
	return EncryptionConfig{
		Algorithm:        "AES-256-GCM",
		KeyDerivation:    Argon2ID,
		KeyLength:        32, // 256 bits
		Argon2Time:       1,
		Argon2Memory:     64 * 1024, // 64 MB
		Argon2Threads:    4,
		PBKDF2Iterations: 100000,
	}
}

// NewDataEncryptor creates a new data encryptor with derived key
func NewDataEncryptor(password string, salt []byte, config EncryptionConfig, logger *zap.Logger) (*DataEncryptor, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	if len(salt) < 16 {
		return nil, errors.New("salt must be at least 16 bytes")
	}

	// Derive key using specified method
	var key []byte
	switch config.KeyDerivation {
	case Argon2ID:
		key = argon2.IDKey(
			[]byte(password),
			salt,
			config.Argon2Time,
			config.Argon2Memory,
			config.Argon2Threads,
			uint32(config.KeyLength),
		)
		logger.Info("Key derived using Argon2id",
			zap.Int("key_length", config.KeyLength))
	case PBKDF2:
		key = pbkdf2.Key(
			[]byte(password),
			salt,
			config.PBKDF2Iterations,
			config.KeyLength,
			sha256.New,
		)
		logger.Info("Key derived using PBKDF2",
			zap.Int("key_length", config.KeyLength),
			zap.Int("iterations", config.PBKDF2Iterations))
	default:
		return nil, fmt.Errorf("unsupported key derivation method: %s", config.KeyDerivation)
	}

	// Create AES cipher
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	return &DataEncryptor{
		key:           key,
		gcm:           gcm,
		config:        config,
		logger:        logger,
		keyDerivation: config.KeyDerivation,
	}, nil
}

// NewDataEncryptorWithKey creates a new data encryptor with direct key
func NewDataEncryptorWithKey(key []byte, logger *zap.Logger) (*DataEncryptor, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	if len(key) != 16 && len(key) != 24 && len(key) != 32 {
		return nil, errors.New("key must be 16, 24, or 32 bytes")
	}

	// Create AES cipher
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	return &DataEncryptor{
		key:    key,
		gcm:    gcm,
		logger: logger,
	}, nil
}

// Encrypt encrypts plaintext data
func (de *DataEncryptor) Encrypt(plaintext []byte) ([]byte, error) {
	if len(plaintext) == 0 {
		return nil, errors.New("plaintext cannot be empty")
	}

	// Generate random nonce
	nonce := make([]byte, de.gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	// Encrypt and authenticate
	ciphertext := de.gcm.Seal(nonce, nonce, plaintext, nil)

	de.logger.Debug("Data encrypted",
		zap.Int("plaintext_size", len(plaintext)),
		zap.Int("ciphertext_size", len(ciphertext)))

	return ciphertext, nil
}

// Decrypt decrypts ciphertext data
func (de *DataEncryptor) Decrypt(ciphertext []byte) ([]byte, error) {
	nonceSize := de.gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		return nil, errors.New("ciphertext too short")
	}

	// Extract nonce and ciphertext
	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]

	// Decrypt and verify
	plaintext, err := de.gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: %w", err)
	}

	de.logger.Debug("Data decrypted",
		zap.Int("ciphertext_size", len(ciphertext)+nonceSize),
		zap.Int("plaintext_size", len(plaintext)))

	return plaintext, nil
}

// EncryptString encrypts a string and returns base64 encoded result
func (de *DataEncryptor) EncryptString(plaintext string) (string, error) {
	ciphertext, err := de.Encrypt([]byte(plaintext))
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(ciphertext), nil
}

// DecryptString decrypts a base64 encoded string
func (de *DataEncryptor) DecryptString(ciphertext string) (string, error) {
	data, err := base64.StdEncoding.DecodeString(ciphertext)
	if err != nil {
		return "", fmt.Errorf("failed to decode base64: %w", err)
	}

	plaintext, err := de.Decrypt(data)
	if err != nil {
		return "", err
	}

	return string(plaintext), nil
}

// RotateKey rotates the encryption key
func (de *DataEncryptor) RotateKey(newPassword string, salt []byte) error {
	// Derive new key
	var newKey []byte
	switch de.keyDerivation {
	case Argon2ID:
		newKey = argon2.IDKey(
			[]byte(newPassword),
			salt,
			de.config.Argon2Time,
			de.config.Argon2Memory,
			de.config.Argon2Threads,
			uint32(de.config.KeyLength),
		)
	case PBKDF2:
		newKey = pbkdf2.Key(
			[]byte(newPassword),
			salt,
			de.config.PBKDF2Iterations,
			de.config.KeyLength,
			sha256.New,
		)
	default:
		return fmt.Errorf("unsupported key derivation method: %s", de.keyDerivation)
	}

	// Create new cipher
	block, err := aes.NewCipher(newKey)
	if err != nil {
		return fmt.Errorf("failed to create new cipher: %w", err)
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return fmt.Errorf("failed to create new GCM: %w", err)
	}

	// Update encryptor
	de.key = newKey
	de.gcm = gcm

	de.logger.Info("Encryption key rotated successfully")
	return nil
}

// GenerateSalt generates a random salt
func GenerateSalt() ([]byte, error) {
	salt := make([]byte, 32)
	if _, err := rand.Read(salt); err != nil {
		return nil, fmt.Errorf("failed to generate salt: %w", err)
	}
	return salt, nil
}

// StreamEncryptor handles streaming encryption for large data
type StreamEncryptor struct {
	writer cipher.StreamWriter
	logger *zap.Logger
}

// NewStreamEncryptor creates a new stream encryptor
func NewStreamEncryptor(key []byte, w io.Writer, logger *zap.Logger) (*StreamEncryptor, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	// Generate IV
	iv := make([]byte, aes.BlockSize)
	if _, err := rand.Read(iv); err != nil {
		return nil, fmt.Errorf("failed to generate IV: %w", err)
	}

	// Write IV to output
	if _, err := w.Write(iv); err != nil {
		return nil, fmt.Errorf("failed to write IV: %w", err)
	}

	stream := cipher.NewCTR(block, iv)
	streamWriter := cipher.StreamWriter{S: stream, W: w}

	return &StreamEncryptor{
		writer: streamWriter,
		logger: logger,
	}, nil
}

// Write encrypts and writes data
func (se *StreamEncryptor) Write(p []byte) (n int, err error) {
	return se.writer.Write(p)
}

// StreamDecryptor handles streaming decryption for large data
type StreamDecryptor struct {
	reader cipher.StreamReader
	logger *zap.Logger
}

// NewStreamDecryptor creates a new stream decryptor
func NewStreamDecryptor(key []byte, r io.Reader, logger *zap.Logger) (*StreamDecryptor, error) {
	if logger == nil {
		logger = zap.NewNop()
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	// Read IV from input
	iv := make([]byte, aes.BlockSize)
	if _, err := io.ReadFull(r, iv); err != nil {
		return nil, fmt.Errorf("failed to read IV: %w", err)
	}

	stream := cipher.NewCTR(block, iv)
	streamReader := cipher.StreamReader{S: stream, R: r}

	return &StreamDecryptor{
		reader: streamReader,
		logger: logger,
	}, nil
}

// Read decrypts and reads data
func (sd *StreamDecryptor) Read(p []byte) (n int, err error) {
	return sd.reader.Read(p)
}

// SecureEraseKey securely erases encryption key from memory
func (de *DataEncryptor) SecureEraseKey() {
	for i := range de.key {
		de.key[i] = 0
	}
	de.logger.Info("Encryption key securely erased from memory")
}
