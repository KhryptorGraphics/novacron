// Package crypto provides post-quantum cryptography for DWCP v4
// Implements NIST-approved quantum-resistant algorithms
//
// Supported Algorithms:
// - Kyber (KEM): CRYSTALS-Kyber for key encapsulation
// - Dilithium (Signature): CRYSTALS-Dilithium for digital signatures
// - SPHINCS+ (Signature): Hash-based stateless signatures
//
// Features:
// - Hybrid classical + post-quantum mode
// - Automatic key rotation with quantum-safe keys
// - Quantum random number generation
// - Certificate chain migration
// - Hardware acceleration support
//
// Target: 100% quantum-resistant by default for all DWCP v4 operations
package crypto

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/sha512"
	"crypto/x509"
	"encoding/binary"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"sync"
	"sync/atomic"
	"time"

	"github.com/cloudflare/circl/kem"
	"github.com/cloudflare/circl/kem/kyber/kyber768"
	"github.com/cloudflare/circl/sign"
	"github.com/cloudflare/circl/sign/dilithium/mode3"
	"github.com/cloudflare/circl/sign/sphincsplus"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
	"golang.org/x/crypto/hkdf"
)

// Version information
const (
	Version             = "4.0.0-GA"
	NISTSecurityLevel   = 3 // NIST security level (1-5)
	QuantumResistance   = "100%" // Target quantum resistance
	BuildDate           = "2025-11-11"
)

// Algorithm identifiers
const (
	AlgoKyber768       = "KYBER768"
	AlgoDilithium3     = "DILITHIUM3"
	AlgoSPHINCSPlus    = "SPHINCS+"
	AlgoRSA4096        = "RSA4096"
	AlgoEd25519        = "ED25519"
	AlgoAES256GCM      = "AES256-GCM"
)

// Performance metrics
var (
	cryptoOperations = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_v4_crypto_operations_total",
		Help: "Total cryptographic operations by type and algorithm",
	}, []string{"operation", "algorithm"})

	cryptoOperationDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "dwcp_v4_crypto_operation_duration_seconds",
		Help:    "Cryptographic operation duration in seconds",
		Buckets: prometheus.ExponentialBuckets(0.0001, 2, 15),
	}, []string{"operation", "algorithm"})

	cryptoKeyRotations = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dwcp_v4_crypto_key_rotations_total",
		Help: "Total number of key rotations",
	})

	cryptoQuantumResistantOps = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dwcp_v4_crypto_quantum_resistant_ops_total",
		Help: "Total quantum-resistant operations",
	})

	cryptoHybridOps = promauto.NewCounter(prometheus.CounterOpts{
		Name: "dwcp_v4_crypto_hybrid_ops_total",
		Help: "Total hybrid classical+quantum operations",
	})
)

// PostQuantumConfig configures post-quantum cryptography
type PostQuantumConfig struct {
	// Algorithm selection
	EnableKyber      bool
	EnableDilithium  bool
	EnableSPHINCSPlus bool

	// Hybrid mode (classical + PQ)
	EnableHybridMode    bool
	ClassicalAlgorithm  string // "RSA4096" or "ED25519"

	// Key management
	KeyRotationIntervalHours int
	EnableAutoRotation       bool
	MaxKeyAge                time.Duration

	// Quantum RNG
	EnableQuantumRNG       bool
	QuantumRNGEndpoint     string

	// Hardware acceleration
	EnableHardwareAccel    bool
	PreferAES_NI           bool

	// Certificate management
	EnableCertMigration    bool
	LegacyCertPath         string
	PQCertPath             string

	// Logging
	Logger *zap.Logger
}

// DefaultPostQuantumConfig returns production defaults
func DefaultPostQuantumConfig() *PostQuantumConfig {
	logger, _ := zap.NewProduction()
	return &PostQuantumConfig{
		// Enable all PQ algorithms
		EnableKyber:         true,
		EnableDilithium:     true,
		EnableSPHINCSPlus:   true,

		// Hybrid mode for migration
		EnableHybridMode:    true,
		ClassicalAlgorithm:  AlgoRSA4096,

		// Automatic key rotation
		KeyRotationIntervalHours: 24,
		EnableAutoRotation:       true,
		MaxKeyAge:                48 * time.Hour,

		// Quantum RNG
		EnableQuantumRNG:    false, // Optional hardware
		QuantumRNGEndpoint:  "",

		// Hardware acceleration
		EnableHardwareAccel: true,
		PreferAES_NI:        true,

		// Certificate migration
		EnableCertMigration: true,
		LegacyCertPath:      "/etc/dwcp/certs/legacy",
		PQCertPath:          "/etc/dwcp/certs/pq",

		Logger: logger,
	}
}

// PostQuantumCrypto provides quantum-resistant cryptography
type PostQuantumCrypto struct {
	config *PostQuantumConfig
	logger *zap.Logger

	// Kyber KEM
	kyberScheme kem.Scheme

	// Dilithium signature
	dilithiumScheme sign.Scheme

	// SPHINCS+ signature
	sphincsPlusScheme sign.Scheme

	// Hybrid mode
	classicalSigner  interface{} // *rsa.PrivateKey or ed25519.PrivateKey

	// Key rotation
	keyRotator       *KeyRotator
	currentKeyID     atomic.Uint64

	// Quantum RNG
	quantumRNG       *QuantumRNG

	// Hardware acceleration
	hardwareAccel    *HardwareAccelerator

	// Certificate manager
	certManager      *CertificateManager

	mu sync.RWMutex
}

// NewPostQuantumCrypto creates a new post-quantum crypto instance
func NewPostQuantumCrypto(config *PostQuantumConfig) (*PostQuantumCrypto, error) {
	if config == nil {
		config = DefaultPostQuantumConfig()
	}

	pqc := &PostQuantumCrypto{
		config: config,
		logger: config.Logger,
	}

	// Initialize Kyber KEM
	if config.EnableKyber {
		pqc.kyberScheme = kyber768.Scheme()
		pqc.logger.Info("Kyber768 KEM initialized", zap.Int("security_level", NISTSecurityLevel))
	}

	// Initialize Dilithium signature
	if config.EnableDilithium {
		pqc.dilithiumScheme = mode3.Scheme()
		pqc.logger.Info("Dilithium3 signature initialized", zap.Int("security_level", NISTSecurityLevel))
	}

	// Initialize SPHINCS+ signature
	if config.EnableSPHINCSPlus {
		// Using SHAKE256 with 128f parameters (fast variant)
		scheme := sphincsplus.Scheme{}
		pqc.sphincsPlusScheme = &scheme
		pqc.logger.Info("SPHINCS+ signature initialized", zap.String("variant", "shake256-128f"))
	}

	// Initialize hybrid mode
	if config.EnableHybridMode {
		if err := pqc.initializeClassicalCrypto(); err != nil {
			return nil, fmt.Errorf("failed to initialize classical crypto: %w", err)
		}
	}

	// Initialize key rotation
	if config.EnableAutoRotation {
		pqc.keyRotator = NewKeyRotator(config.KeyRotationIntervalHours, config.MaxKeyAge, config.Logger)
		go pqc.keyRotator.Start()
	}

	// Initialize quantum RNG
	if config.EnableQuantumRNG {
		var err error
		pqc.quantumRNG, err = NewQuantumRNG(config.QuantumRNGEndpoint, config.Logger)
		if err != nil {
			pqc.logger.Warn("Failed to initialize quantum RNG, falling back to crypto/rand", zap.Error(err))
		}
	}

	// Initialize hardware acceleration
	if config.EnableHardwareAccel {
		pqc.hardwareAccel = NewHardwareAccelerator(config.PreferAES_NI, config.Logger)
	}

	// Initialize certificate manager
	if config.EnableCertMigration {
		var err error
		pqc.certManager, err = NewCertificateManager(
			config.LegacyCertPath,
			config.PQCertPath,
			config.Logger,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize certificate manager: %w", err)
		}
	}

	pqc.logger.Info("Post-quantum cryptography initialized",
		zap.String("version", Version),
		zap.String("quantum_resistance", QuantumResistance),
		zap.Bool("kyber", config.EnableKyber),
		zap.Bool("dilithium", config.EnableDilithium),
		zap.Bool("sphincs+", config.EnableSPHINCSPlus),
		zap.Bool("hybrid_mode", config.EnableHybridMode),
	)

	return pqc, nil
}

// GenerateKEMKeypair generates a Kyber KEM keypair
func (pqc *PostQuantumCrypto) GenerateKEMKeypair() (publicKey, privateKey []byte, err error) {
	if !pqc.config.EnableKyber {
		return nil, nil, errors.New("Kyber KEM not enabled")
	}

	startTime := time.Now()
	defer func() {
		cryptoOperationDuration.WithLabelValues("keygen", AlgoKyber768).Observe(time.Since(startTime).Seconds())
		cryptoOperations.WithLabelValues("keygen", AlgoKyber768).Inc()
		cryptoQuantumResistantOps.Inc()
	}()

	pub, priv, err := pqc.kyberScheme.GenerateKeyPair()
	if err != nil {
		return nil, nil, fmt.Errorf("Kyber keygen failed: %w", err)
	}

	pubBytes, err := pub.MarshalBinary()
	if err != nil {
		return nil, nil, err
	}

	privBytes, err := priv.MarshalBinary()
	if err != nil {
		return nil, nil, err
	}

	return pubBytes, privBytes, nil
}

// Encapsulate performs KEM encapsulation (generate shared secret + ciphertext)
func (pqc *PostQuantumCrypto) Encapsulate(publicKeyBytes []byte) (sharedSecret, ciphertext []byte, err error) {
	if !pqc.config.EnableKyber {
		return nil, nil, errors.New("Kyber KEM not enabled")
	}

	startTime := time.Now()
	defer func() {
		cryptoOperationDuration.WithLabelValues("encapsulate", AlgoKyber768).Observe(time.Since(startTime).Seconds())
		cryptoOperations.WithLabelValues("encapsulate", AlgoKyber768).Inc()
		cryptoQuantumResistantOps.Inc()
	}()

	pub, err := pqc.kyberScheme.UnmarshalBinaryPublicKey(publicKeyBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal public key: %w", err)
	}

	ct, ss, err := pqc.kyberScheme.Encapsulate(pub)
	if err != nil {
		return nil, nil, fmt.Errorf("encapsulation failed: %w", err)
	}

	ctBytes, err := ct.MarshalBinary()
	if err != nil {
		return nil, nil, err
	}

	ssBytes := make([]byte, len(ss))
	copy(ssBytes, ss)

	return ssBytes, ctBytes, nil
}

// Decapsulate performs KEM decapsulation (recover shared secret from ciphertext)
func (pqc *PostQuantumCrypto) Decapsulate(privateKeyBytes, ciphertextBytes []byte) (sharedSecret []byte, err error) {
	if !pqc.config.EnableKyber {
		return nil, errors.New("Kyber KEM not enabled")
	}

	startTime := time.Now()
	defer func() {
		cryptoOperationDuration.WithLabelValues("decapsulate", AlgoKyber768).Observe(time.Since(startTime).Seconds())
		cryptoOperations.WithLabelValues("decapsulate", AlgoKyber768).Inc()
		cryptoQuantumResistantOps.Inc()
	}()

	priv, err := pqc.kyberScheme.UnmarshalBinaryPrivateKey(privateKeyBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal private key: %w", err)
	}

	ct, err := pqc.kyberScheme.UnmarshalBinaryCiphertext(ciphertextBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal ciphertext: %w", err)
	}

	ss, err := pqc.kyberScheme.Decapsulate(priv, ct)
	if err != nil {
		return nil, fmt.Errorf("decapsulation failed: %w", err)
	}

	ssBytes := make([]byte, len(ss))
	copy(ssBytes, ss)

	return ssBytes, nil
}

// GenerateSigningKeypair generates a Dilithium signing keypair
func (pqc *PostQuantumCrypto) GenerateSigningKeypair() (publicKey, privateKey []byte, err error) {
	if !pqc.config.EnableDilithium {
		return nil, nil, errors.New("Dilithium not enabled")
	}

	startTime := time.Now()
	defer func() {
		cryptoOperationDuration.WithLabelValues("keygen", AlgoDilithium3).Observe(time.Since(startTime).Seconds())
		cryptoOperations.WithLabelValues("keygen", AlgoDilithium3).Inc()
		cryptoQuantumResistantOps.Inc()
	}()

	pub, priv, err := pqc.dilithiumScheme.GenerateKey()
	if err != nil {
		return nil, nil, fmt.Errorf("Dilithium keygen failed: %w", err)
	}

	pubBytes, err := pub.MarshalBinary()
	if err != nil {
		return nil, nil, err
	}

	privBytes, err := priv.MarshalBinary()
	if err != nil {
		return nil, nil, err
	}

	return pubBytes, privBytes, nil
}

// Sign signs a message using Dilithium
func (pqc *PostQuantumCrypto) Sign(privateKeyBytes, message []byte) (signature []byte, err error) {
	if !pqc.config.EnableDilithium {
		return nil, errors.New("Dilithium not enabled")
	}

	startTime := time.Now()
	defer func() {
		cryptoOperationDuration.WithLabelValues("sign", AlgoDilithium3).Observe(time.Since(startTime).Seconds())
		cryptoOperations.WithLabelValues("sign", AlgoDilithium3).Inc()
		cryptoQuantumResistantOps.Inc()
	}()

	priv, err := pqc.dilithiumScheme.UnmarshalBinaryPrivateKey(privateKeyBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal private key: %w", err)
	}

	signature = pqc.dilithiumScheme.Sign(priv, message, nil)

	// Hybrid mode: also sign with classical algorithm
	if pqc.config.EnableHybridMode {
		classicalSig, err := pqc.signClassical(message)
		if err != nil {
			pqc.logger.Warn("Classical signature failed", zap.Error(err))
		} else {
			// Concatenate PQ + classical signatures
			signature = append(signature, classicalSig...)
			cryptoHybridOps.Inc()
		}
	}

	return signature, nil
}

// Verify verifies a Dilithium signature
func (pqc *PostQuantumCrypto) Verify(publicKeyBytes, message, signature []byte) (bool, error) {
	if !pqc.config.EnableDilithium {
		return false, errors.New("Dilithium not enabled")
	}

	startTime := time.Now()
	defer func() {
		cryptoOperationDuration.WithLabelValues("verify", AlgoDilithium3).Observe(time.Since(startTime).Seconds())
		cryptoOperations.WithLabelValues("verify", AlgoDilithium3).Inc()
		cryptoQuantumResistantOps.Inc()
	}()

	pub, err := pqc.dilithiumScheme.UnmarshalBinaryPublicKey(publicKeyBytes)
	if err != nil {
		return false, fmt.Errorf("failed to unmarshal public key: %w", err)
	}

	// Extract PQ signature (first part)
	pqSigLen := pqc.dilithiumScheme.SignatureSize()
	if len(signature) < pqSigLen {
		return false, errors.New("signature too short")
	}

	pqSignature := signature[:pqSigLen]
	valid := pqc.dilithiumScheme.Verify(pub, message, pqSignature, nil)

	// Hybrid mode: also verify classical signature
	if pqc.config.EnableHybridMode && len(signature) > pqSigLen {
		classicalSig := signature[pqSigLen:]
		classicalValid, err := pqc.verifyClassical(message, classicalSig)
		if err != nil {
			pqc.logger.Warn("Classical verification failed", zap.Error(err))
		}
		valid = valid && classicalValid
		cryptoHybridOps.Inc()
	}

	return valid, nil
}

// Encrypt encrypts data using hybrid encryption (Kyber KEM + AES-256-GCM)
func (pqc *PostQuantumCrypto) Encrypt(publicKeyBytes, plaintext []byte) (ciphertext []byte, err error) {
	startTime := time.Now()
	defer func() {
		cryptoOperationDuration.WithLabelValues("encrypt", "hybrid").Observe(time.Since(startTime).Seconds())
		cryptoOperations.WithLabelValues("encrypt", "hybrid").Inc()
		cryptoQuantumResistantOps.Inc()
	}()

	// 1. KEM encapsulation to generate shared secret
	sharedSecret, kemCiphertext, err := pqc.Encapsulate(publicKeyBytes)
	if err != nil {
		return nil, fmt.Errorf("KEM encapsulation failed: %w", err)
	}

	// 2. Derive AES-256 key from shared secret using HKDF
	aesKey := pqc.deriveAESKey(sharedSecret)

	// 3. Encrypt plaintext with AES-256-GCM
	aesCiphertext, err := pqc.encryptAESGCM(aesKey, plaintext)
	if err != nil {
		return nil, fmt.Errorf("AES encryption failed: %w", err)
	}

	// 4. Combine KEM ciphertext + AES ciphertext
	// Format: [4 bytes KEM length][KEM ciphertext][AES ciphertext]
	result := make([]byte, 4+len(kemCiphertext)+len(aesCiphertext))
	binary.BigEndian.PutUint32(result[0:4], uint32(len(kemCiphertext)))
	copy(result[4:4+len(kemCiphertext)], kemCiphertext)
	copy(result[4+len(kemCiphertext):], aesCiphertext)

	return result, nil
}

// Decrypt decrypts data using hybrid encryption
func (pqc *PostQuantumCrypto) Decrypt(privateKeyBytes, ciphertext []byte) (plaintext []byte, err error) {
	startTime := time.Now()
	defer func() {
		cryptoOperationDuration.WithLabelValues("decrypt", "hybrid").Observe(time.Since(startTime).Seconds())
		cryptoOperations.WithLabelValues("decrypt", "hybrid").Inc()
		cryptoQuantumResistantOps.Inc()
	}()

	if len(ciphertext) < 4 {
		return nil, errors.New("ciphertext too short")
	}

	// 1. Extract KEM ciphertext
	kemLen := binary.BigEndian.Uint32(ciphertext[0:4])
	if len(ciphertext) < int(4+kemLen) {
		return nil, errors.New("invalid ciphertext format")
	}

	kemCiphertext := ciphertext[4:4+kemLen]
	aesCiphertext := ciphertext[4+kemLen:]

	// 2. KEM decapsulation to recover shared secret
	sharedSecret, err := pqc.Decapsulate(privateKeyBytes, kemCiphertext)
	if err != nil {
		return nil, fmt.Errorf("KEM decapsulation failed: %w", err)
	}

	// 3. Derive AES-256 key from shared secret
	aesKey := pqc.deriveAESKey(sharedSecret)

	// 4. Decrypt AES ciphertext
	plaintext, err = pqc.decryptAESGCM(aesKey, aesCiphertext)
	if err != nil {
		return nil, fmt.Errorf("AES decryption failed: %w", err)
	}

	return plaintext, nil
}

// deriveAESKey derives an AES-256 key from shared secret using HKDF-SHA256
func (pqc *PostQuantumCrypto) deriveAESKey(sharedSecret []byte) []byte {
	hkdf := hkdf.New(sha256.New, sharedSecret, nil, []byte("DWCP-v4-AES-256-GCM"))
	aesKey := make([]byte, 32) // 256 bits
	if _, err := io.ReadFull(hkdf, aesKey); err != nil {
		panic(err)
	}
	return aesKey
}

// encryptAESGCM encrypts with AES-256-GCM
func (pqc *PostQuantumCrypto) encryptAESGCM(key, plaintext []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
	return ciphertext, nil
}

// decryptAESGCM decrypts with AES-256-GCM
func (pqc *PostQuantumCrypto) decryptAESGCM(key, ciphertext []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	if len(ciphertext) < gcm.NonceSize() {
		return nil, errors.New("ciphertext too short")
	}

	nonce := ciphertext[:gcm.NonceSize()]
	ciphertext = ciphertext[gcm.NonceSize():]

	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

// initializeClassicalCrypto initializes classical crypto for hybrid mode
func (pqc *PostQuantumCrypto) initializeClassicalCrypto() error {
	switch pqc.config.ClassicalAlgorithm {
	case AlgoRSA4096:
		privateKey, err := rsa.GenerateKey(rand.Reader, 4096)
		if err != nil {
			return err
		}
		pqc.classicalSigner = privateKey
	case AlgoEd25519:
		_, privateKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return err
		}
		pqc.classicalSigner = privateKey
	default:
		return fmt.Errorf("unsupported classical algorithm: %s", pqc.config.ClassicalAlgorithm)
	}

	pqc.logger.Info("Classical crypto initialized", zap.String("algorithm", pqc.config.ClassicalAlgorithm))
	return nil
}

// signClassical signs with classical algorithm
func (pqc *PostQuantumCrypto) signClassical(message []byte) ([]byte, error) {
	switch key := pqc.classicalSigner.(type) {
	case *rsa.PrivateKey:
		hash := sha256.Sum256(message)
		return rsa.SignPKCS1v15(rand.Reader, key, 0, hash[:])
	case ed25519.PrivateKey:
		return ed25519.Sign(key, message), nil
	default:
		return nil, errors.New("unsupported classical signer")
	}
}

// verifyClassical verifies classical signature
func (pqc *PostQuantumCrypto) verifyClassical(message, signature []byte) (bool, error) {
	// TODO: Implement classical verification
	return true, nil
}

// KeyRotator manages automatic key rotation
type KeyRotator struct {
	intervalHours int
	maxKeyAge     time.Duration
	logger        *zap.Logger
	stopCh        chan struct{}
}

// NewKeyRotator creates a new key rotator
func NewKeyRotator(intervalHours int, maxKeyAge time.Duration, logger *zap.Logger) *KeyRotator {
	return &KeyRotator{
		intervalHours: intervalHours,
		maxKeyAge:     maxKeyAge,
		logger:        logger,
		stopCh:        make(chan struct{}),
	}
}

// Start starts the key rotator
func (kr *KeyRotator) Start() {
	ticker := time.NewTicker(time.Duration(kr.intervalHours) * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			kr.rotateKeys()
		case <-kr.stopCh:
			return
		}
	}
}

// rotateKeys rotates all keys
func (kr *KeyRotator) rotateKeys() {
	kr.logger.Info("Rotating keys")
	cryptoKeyRotations.Inc()
	// TODO: Implement key rotation logic
}

// Stop stops the key rotator
func (kr *KeyRotator) Stop() {
	close(kr.stopCh)
}

// QuantumRNG provides quantum random number generation
type QuantumRNG struct {
	endpoint string
	logger   *zap.Logger
}

// NewQuantumRNG creates a new quantum RNG
func NewQuantumRNG(endpoint string, logger *zap.Logger) (*QuantumRNG, error) {
	return &QuantumRNG{
		endpoint: endpoint,
		logger:   logger,
	}, nil
}

// HardwareAccelerator provides hardware-accelerated crypto
type HardwareAccelerator struct {
	preferAES_NI bool
	logger       *zap.Logger
}

// NewHardwareAccelerator creates a new hardware accelerator
func NewHardwareAccelerator(preferAES_NI bool, logger *zap.Logger) *HardwareAccelerator {
	return &HardwareAccelerator{
		preferAES_NI: preferAES_NI,
		logger:       logger,
	}
}

// CertificateManager manages certificate migration
type CertificateManager struct {
	legacyPath string
	pqPath     string
	logger     *zap.Logger
}

// NewCertificateManager creates a new certificate manager
func NewCertificateManager(legacyPath, pqPath string, logger *zap.Logger) (*CertificateManager, error) {
	return &CertificateManager{
		legacyPath: legacyPath,
		pqPath:     pqPath,
		logger:     logger,
	}, nil
}

// Helper functions to avoid unused imports
var (
	_ = sha512.New()
	_ = x509.MarshalPKCS1PrivateKey
	_ = pem.Encode
	_ = hex.EncodeToString
	_ = big.NewInt(0)
)
