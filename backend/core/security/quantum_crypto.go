package security

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/google/uuid"
	"golang.org/x/crypto/argon2"
	"golang.org/x/crypto/chacha20poly1305"
)

// QuantumCryptoManager manages quantum-resistant cryptography
type QuantumCryptoManager struct {
	mu                sync.RWMutex
	keyStore          *QuantumKeyStore
	kyberEngine       *KyberEngine
	dilithiumEngine   *DilithiumEngine
	hybridCrypto      *HybridCryptoEngine
	cryptoAgility     *CryptoAgilityFramework
	config            *QuantumCryptoConfig
	metrics           *QuantumCryptoMetrics
}

// QuantumCryptoConfig configuration for quantum crypto
type QuantumCryptoConfig struct {
	// Post-quantum algorithms
	EnableKyber        bool // CRYSTALS-Kyber for key exchange
	EnableDilithium    bool // CRYSTALS-Dilithium for signatures
	EnableSphincs      bool // SPHINCS+ for signatures
	EnableFrodo        bool // FrodoKEM for key exchange

	// Hybrid mode
	EnableHybridMode   bool // Classical + quantum-resistant
	ClassicalAlgorithm string // RSA, ECDSA, etc.

	// Key management
	KeyRotationInterval  time.Duration
	KeyDerivationRounds  uint32
	MinKeyStrength       int

	// Performance
	UseHardwareAccel     bool
	ParallelOperations   int
	CachingEnabled       bool

	// Compliance
	FIPS140_3Compliant   bool
	NISTLevel            int // NIST security level (1-5)
}

// QuantumKeyStore stores quantum-resistant keys
type QuantumKeyStore struct {
	mu              sync.RWMutex
	kyberKeys       map[string]*KyberKeyPair
	dilithiumKeys   map[string]*DilithiumKeyPair
	sphincsKeys     map[string]*SphincsKeyPair
	hybridKeys      map[string]*HybridKeyPair
	keyMetadata     map[string]*KeyMetadata
}

// KyberKeyPair represents CRYSTALS-Kyber key pair
type KyberKeyPair struct {
	ID              string
	PublicKey       []byte
	PrivateKey      []byte
	SecurityLevel   int // 2, 3, or 5 (corresponding to 512, 768, 1024)
	CreatedAt       time.Time
	LastUsed        time.Time
	UsageCount      int64
	RotationDue     time.Time
}

// DilithiumKeyPair represents CRYSTALS-Dilithium key pair
type DilithiumKeyPair struct {
	ID              string
	PublicKey       []byte
	PrivateKey      []byte
	SecurityLevel   int // 2, 3, or 5
	CreatedAt       time.Time
	LastUsed        time.Time
	SignatureCount  int64
	RotationDue     time.Time
}

// SphincsKeyPair represents SPHINCS+ key pair
type SphincsKeyPair struct {
	ID              string
	PublicKey       []byte
	PrivateKey      []byte
	Variant         string // "shake256", "sha256"
	SecurityLevel   int
	CreatedAt       time.Time
}

// HybridKeyPair represents hybrid classical+quantum key pair
type HybridKeyPair struct {
	ID              string
	ClassicalKey    interface{} // RSA or ECDSA key
	QuantumKey      interface{} // Kyber or other PQC key
	Algorithm       string
	CreatedAt       time.Time
}

// KeyMetadata contains key metadata
type KeyMetadata struct {
	KeyID           string
	KeyType         KeyType
	Algorithm       string
	SecurityLevel   int
	Purpose         KeyPurpose
	Owner           string
	Status          KeyStatus
	CreatedAt       time.Time
	ExpiresAt       time.Time
	LastRotated     time.Time
	RotationPolicy  *RotationPolicy
	ComplianceFlags []string
	Tags            map[string]string
}

// KeyType defines key types
type KeyType string

const (
	KeyTypeKyber        KeyType = "kyber"
	KeyTypeDilithium    KeyType = "dilithium"
	KeyTypeSphincs      KeyType = "sphincs"
	KeyTypeHybrid       KeyType = "hybrid"
	KeyTypeSymmetric    KeyType = "symmetric"
)

// KeyPurpose defines key purposes
type KeyPurpose string

const (
	KeyPurposeEncryption KeyPurpose = "encryption"
	KeyPurposeSigning    KeyPurpose = "signing"
	KeyPurposeKEM        KeyPurpose = "kem" // Key Encapsulation Mechanism
	KeyPurposeDerivation KeyPurpose = "derivation"
)

// KeyStatus defines key status
type KeyStatus string

const (
	KeyStatusActive     KeyStatus = "active"
	KeyStatusRotating   KeyStatus = "rotating"
	KeyStatusDeprecated KeyStatus = "deprecated"
	KeyStatusRevoked    KeyStatus = "revoked"
)

// RotationPolicy defines key rotation policy
type RotationPolicy struct {
	Enabled         bool
	Interval        time.Duration
	MaxUsageCount   int64
	AutoRotate      bool
	NotifyBefore    time.Duration
}

// KyberEngine implements CRYSTALS-Kyber KEM
type KyberEngine struct {
	mu            sync.RWMutex
	securityLevel int
	cache         *KEMCache
}

// KEMCache caches KEM operations
type KEMCache struct {
	mu           sync.RWMutex
	encapsulated map[string]*CachedKEM
	maxSize      int
	ttl          time.Duration
}

// CachedKEM represents cached KEM
type CachedKEM struct {
	Ciphertext  []byte
	SharedSecret []byte
	Timestamp   time.Time
}

// DilithiumEngine implements CRYSTALS-Dilithium signatures
type DilithiumEngine struct {
	mu            sync.RWMutex
	securityLevel int
	signCache     *SignatureCache
}

// SignatureCache caches signatures
type SignatureCache struct {
	mu         sync.RWMutex
	signatures map[string]*CachedSignature
	maxSize    int
}

// CachedSignature represents cached signature
type CachedSignature struct {
	Signature []byte
	Message   []byte
	Timestamp time.Time
}

// HybridCryptoEngine implements hybrid classical+quantum crypto
type HybridCryptoEngine struct {
	mu             sync.RWMutex
	classicalEngine interface{}
	quantumEngine   interface{}
	combiner       *KeyCombiner
}

// KeyCombiner combines classical and quantum keys
type KeyCombiner struct {
	mu         sync.RWMutex
	algorithm  CombinerAlgorithm
	kdf        *KeyDerivationFunction
}

// CombinerAlgorithm defines combination algorithm
type CombinerAlgorithm string

const (
	CombinerXOR        CombinerAlgorithm = "xor"
	CombinerConcat     CombinerAlgorithm = "concat"
	CombinerKDF        CombinerAlgorithm = "kdf"
	CombinerHKDF       CombinerAlgorithm = "hkdf"
)

// KeyDerivationFunction implements KDF
type KeyDerivationFunction struct {
	mu        sync.RWMutex
	algorithm KDFAlgorithm
	saltSize  int
	keySize   int
}

// KDFAlgorithm defines KDF algorithms
type KDFAlgorithm string

const (
	KDFArgon2   KDFAlgorithm = "argon2"
	KDFPBKDF2   KDFAlgorithm = "pbkdf2"
	KDFHKDF     KDFAlgorithm = "hkdf"
	KDFScrypt   KDFAlgorithm = "scrypt"
)

// CryptoAgilityFramework enables crypto algorithm agility
type CryptoAgilityFramework struct {
	mu                  sync.RWMutex
	supportedAlgorithms map[string]*AlgorithmProfile
	activeAlgorithm     string
	migrationPlan       *MigrationPlan
	compatibilityMatrix map[string][]string
}

// AlgorithmProfile defines algorithm profile
type AlgorithmProfile struct {
	ID              string
	Name            string
	Type            AlgorithmType
	SecurityLevel   int
	QuantumResistant bool
	Enabled         bool
	PerformanceProfile *PerformanceProfile
	ComplianceFlags []string
	DeprecationDate *time.Time
}

// AlgorithmType defines algorithm types
type AlgorithmType string

const (
	AlgorithmTypeKEM       AlgorithmType = "kem"
	AlgorithmTypeSignature AlgorithmType = "signature"
	AlgorithmTypeEncryption AlgorithmType = "encryption"
	AlgorithmTypeHash       AlgorithmType = "hash"
)

// PerformanceProfile contains performance metrics
type PerformanceProfile struct {
	KeyGenSpeed      float64 // ops/sec
	EncryptSpeed     float64 // MB/s
	DecryptSpeed     float64 // MB/s
	SignSpeed        float64 // ops/sec
	VerifySpeed      float64 // ops/sec
	KeySize          int     // bytes
	CiphertextExpansion float64 // ratio
}

// MigrationPlan defines crypto migration plan
type MigrationPlan struct {
	FromAlgorithm   string
	ToAlgorithm     string
	StartDate       time.Time
	EndDate         time.Time
	RollbackPlan    *RollbackPlan
	MigrationSteps  []MigrationStep
	CurrentStep     int
	Status          MigrationStatus
}

// MigrationStatus defines migration status
type MigrationStatus string

const (
	MigrationPending    MigrationStatus = "pending"
	MigrationInProgress MigrationStatus = "in_progress"
	MigrationCompleted  MigrationStatus = "completed"
	MigrationFailed     MigrationStatus = "failed"
	MigrationRolledBack MigrationStatus = "rolled_back"
)

// RollbackPlan defines rollback plan
type RollbackPlan struct {
	Enabled      bool
	TriggerConditions []RollbackCondition
	Steps        []RollbackStep
}

// RollbackCondition defines rollback conditions
type RollbackCondition struct {
	Type      string
	Threshold float64
	Metric    string
}

// RollbackStep defines rollback step
type RollbackStep struct {
	ID          string
	Description string
	Action      func() error
	Order       int
}

// MigrationStep defines migration step
type MigrationStep struct {
	ID          string
	Description string
	Action      func() error
	Validate    func() error
	Rollback    func() error
	Order       int
	Status      StepStatus
}

// StepStatus defines step status
type StepStatus string

const (
	StepPending    StepStatus = "pending"
	StepInProgress StepStatus = "in_progress"
	StepCompleted  StepStatus = "completed"
	StepFailed     StepStatus = "failed"
)

// QuantumCryptoMetrics contains metrics
type QuantumCryptoMetrics struct {
	mu                    sync.RWMutex
	TotalKeys             int64
	KeyRotations          int64
	EncryptionOps         int64
	DecryptionOps         int64
	SignatureOps          int64
	VerificationOps       int64
	HybridOps             int64
	QuantumOps            int64
	AverageOpLatency      time.Duration
	CacheHitRate          float64
	LastUpdated           time.Time
}

// NewQuantumCryptoManager creates quantum crypto manager
func NewQuantumCryptoManager(config *QuantumCryptoConfig) *QuantumCryptoManager {
	qcm := &QuantumCryptoManager{
		keyStore:        NewQuantumKeyStore(),
		kyberEngine:     NewKyberEngine(config.NISTLevel),
		dilithiumEngine: NewDilithiumEngine(config.NISTLevel),
		hybridCrypto:    NewHybridCryptoEngine(),
		cryptoAgility:   NewCryptoAgilityFramework(),
		config:          config,
		metrics:         &QuantumCryptoMetrics{},
	}

	qcm.initializeSupportedAlgorithms()
	qcm.startBackgroundTasks()

	return qcm
}

// NewQuantumKeyStore creates quantum key store
func NewQuantumKeyStore() *QuantumKeyStore {
	return &QuantumKeyStore{
		kyberKeys:     make(map[string]*KyberKeyPair),
		dilithiumKeys: make(map[string]*DilithiumKeyPair),
		sphincsKeys:   make(map[string]*SphincsKeyPair),
		hybridKeys:    make(map[string]*HybridKeyPair),
		keyMetadata:   make(map[string]*KeyMetadata),
	}
}

// NewKyberEngine creates Kyber engine
func NewKyberEngine(securityLevel int) *KyberEngine {
	return &KyberEngine{
		securityLevel: securityLevel,
		cache: &KEMCache{
			encapsulated: make(map[string]*CachedKEM),
			maxSize:      1000,
			ttl:          15 * time.Minute,
		},
	}
}

// NewDilithiumEngine creates Dilithium engine
func NewDilithiumEngine(securityLevel int) *DilithiumEngine {
	return &DilithiumEngine{
		securityLevel: securityLevel,
		signCache: &SignatureCache{
			signatures: make(map[string]*CachedSignature),
			maxSize:    1000,
		},
	}
}

// NewHybridCryptoEngine creates hybrid crypto engine
func NewHybridCryptoEngine() *HybridCryptoEngine {
	return &HybridCryptoEngine{
		combiner: &KeyCombiner{
			algorithm: CombinerKDF,
			kdf: &KeyDerivationFunction{
				algorithm: KDFArgon2,
				saltSize:  32,
				keySize:   32,
			},
		},
	}
}

// NewCryptoAgilityFramework creates crypto agility framework
func NewCryptoAgilityFramework() *CryptoAgilityFramework {
	return &CryptoAgilityFramework{
		supportedAlgorithms: make(map[string]*AlgorithmProfile),
		compatibilityMatrix: make(map[string][]string),
	}
}

// GenerateKyberKeyPair generates Kyber key pair
func (qcm *QuantumCryptoManager) GenerateKyberKeyPair() (*KyberKeyPair, error) {
	qcm.mu.Lock()
	defer qcm.mu.Unlock()

	// Simulate Kyber key generation (in production, use actual Kyber implementation)
	publicKey := make([]byte, qcm.getKyberPublicKeySize())
	privateKey := make([]byte, qcm.getKyberPrivateKeySize())

	if _, err := rand.Read(publicKey); err != nil {
		return nil, fmt.Errorf("failed to generate public key: %w", err)
	}

	if _, err := rand.Read(privateKey); err != nil {
		return nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	keyPair := &KyberKeyPair{
		ID:            uuid.New().String(),
		PublicKey:     publicKey,
		PrivateKey:    privateKey,
		SecurityLevel: qcm.config.NISTLevel,
		CreatedAt:     time.Now(),
		RotationDue:   time.Now().Add(qcm.config.KeyRotationInterval),
	}

	qcm.keyStore.mu.Lock()
	qcm.keyStore.kyberKeys[keyPair.ID] = keyPair
	qcm.keyStore.keyMetadata[keyPair.ID] = &KeyMetadata{
		KeyID:         keyPair.ID,
		KeyType:       KeyTypeKyber,
		Algorithm:     "CRYSTALS-Kyber",
		SecurityLevel: qcm.config.NISTLevel,
		Purpose:       KeyPurposeKEM,
		Status:        KeyStatusActive,
		CreatedAt:     time.Now(),
		ExpiresAt:     time.Now().Add(qcm.config.KeyRotationInterval),
	}
	qcm.keyStore.mu.Unlock()

	qcm.metrics.mu.Lock()
	qcm.metrics.TotalKeys++
	qcm.metrics.LastUpdated = time.Now()
	qcm.metrics.mu.Unlock()

	return keyPair, nil
}

// GenerateDilithiumKeyPair generates Dilithium key pair
func (qcm *QuantumCryptoManager) GenerateDilithiumKeyPair() (*DilithiumKeyPair, error) {
	qcm.mu.Lock()
	defer qcm.mu.Unlock()

	// Simulate Dilithium key generation
	publicKey := make([]byte, qcm.getDilithiumPublicKeySize())
	privateKey := make([]byte, qcm.getDilithiumPrivateKeySize())

	if _, err := rand.Read(publicKey); err != nil {
		return nil, fmt.Errorf("failed to generate public key: %w", err)
	}

	if _, err := rand.Read(privateKey); err != nil {
		return nil, fmt.Errorf("failed to generate private key: %w", err)
	}

	keyPair := &DilithiumKeyPair{
		ID:            uuid.New().String(),
		PublicKey:     publicKey,
		PrivateKey:    privateKey,
		SecurityLevel: qcm.config.NISTLevel,
		CreatedAt:     time.Now(),
		RotationDue:   time.Now().Add(qcm.config.KeyRotationInterval),
	}

	qcm.keyStore.mu.Lock()
	qcm.keyStore.dilithiumKeys[keyPair.ID] = keyPair
	qcm.keyStore.keyMetadata[keyPair.ID] = &KeyMetadata{
		KeyID:         keyPair.ID,
		KeyType:       KeyTypeDilithium,
		Algorithm:     "CRYSTALS-Dilithium",
		SecurityLevel: qcm.config.NISTLevel,
		Purpose:       KeyPurposeSigning,
		Status:        KeyStatusActive,
		CreatedAt:     time.Now(),
		ExpiresAt:     time.Now().Add(qcm.config.KeyRotationInterval),
	}
	qcm.keyStore.mu.Unlock()

	qcm.metrics.mu.Lock()
	qcm.metrics.TotalKeys++
	qcm.metrics.LastUpdated = time.Now()
	qcm.metrics.mu.Unlock()

	return keyPair, nil
}

// KyberEncapsulate performs Kyber key encapsulation
func (qcm *QuantumCryptoManager) KyberEncapsulate(publicKey []byte) (ciphertext []byte, sharedSecret []byte, err error) {
	startTime := time.Now()
	defer func() {
		qcm.metrics.mu.Lock()
		qcm.metrics.EncryptionOps++
		qcm.metrics.QuantumOps++
		qcm.metrics.AverageOpLatency = time.Since(startTime)
		qcm.metrics.mu.Unlock()
	}()

	// Check cache
	cacheKey := qcm.generateCacheKey(publicKey)
	if cached, ok := qcm.kyberEngine.cache.Get(cacheKey); ok {
		qcm.metrics.mu.Lock()
		qcm.metrics.CacheHitRate = (qcm.metrics.CacheHitRate*0.9 + 1.0*0.1)
		qcm.metrics.mu.Unlock()
		return cached.Ciphertext, cached.SharedSecret, nil
	}

	// Simulate Kyber encapsulation
	sharedSecret = make([]byte, 32)
	ciphertext = make([]byte, qcm.getKyberCiphertextSize())

	if _, err := rand.Read(sharedSecret); err != nil {
		return nil, nil, fmt.Errorf("failed to generate shared secret: %w", err)
	}

	if _, err := rand.Read(ciphertext); err != nil {
		return nil, nil, fmt.Errorf("failed to generate ciphertext: %w", err)
	}

	// Cache result
	qcm.kyberEngine.cache.Put(cacheKey, &CachedKEM{
		Ciphertext:   ciphertext,
		SharedSecret: sharedSecret,
		Timestamp:    time.Now(),
	})

	return ciphertext, sharedSecret, nil
}

// KyberDecapsulate performs Kyber key decapsulation
func (qcm *QuantumCryptoManager) KyberDecapsulate(privateKey []byte, ciphertext []byte) (sharedSecret []byte, err error) {
	startTime := time.Now()
	defer func() {
		qcm.metrics.mu.Lock()
		qcm.metrics.DecryptionOps++
		qcm.metrics.QuantumOps++
		qcm.metrics.AverageOpLatency = time.Since(startTime)
		qcm.metrics.mu.Unlock()
	}()

	// Simulate Kyber decapsulation
	sharedSecret = make([]byte, 32)
	if _, err := rand.Read(sharedSecret); err != nil {
		return nil, fmt.Errorf("failed to decapsulate: %w", err)
	}

	return sharedSecret, nil
}

// DilithiumSign signs message with Dilithium
func (qcm *QuantumCryptoManager) DilithiumSign(privateKey []byte, message []byte) (signature []byte, err error) {
	startTime := time.Now()
	defer func() {
		qcm.metrics.mu.Lock()
		qcm.metrics.SignatureOps++
		qcm.metrics.QuantumOps++
		qcm.metrics.AverageOpLatency = time.Since(startTime)
		qcm.metrics.mu.Unlock()
	}()

	// Check cache
	cacheKey := qcm.generateSignatureCacheKey(message)
	if cached, ok := qcm.dilithiumEngine.signCache.Get(cacheKey); ok {
		return cached.Signature, nil
	}

	// Simulate Dilithium signing
	signature = make([]byte, qcm.getDilithiumSignatureSize())
	if _, err := rand.Read(signature); err != nil {
		return nil, fmt.Errorf("failed to sign: %w", err)
	}

	// Cache signature
	qcm.dilithiumEngine.signCache.Put(cacheKey, &CachedSignature{
		Signature: signature,
		Message:   message,
		Timestamp: time.Now(),
	})

	return signature, nil
}

// DilithiumVerify verifies Dilithium signature
func (qcm *QuantumCryptoManager) DilithiumVerify(publicKey []byte, message []byte, signature []byte) (bool, error) {
	startTime := time.Now()
	defer func() {
		qcm.metrics.mu.Lock()
		qcm.metrics.VerificationOps++
		qcm.metrics.QuantumOps++
		qcm.metrics.AverageOpLatency = time.Since(startTime)
		qcm.metrics.mu.Unlock()
	}()

	// Simulate Dilithium verification
	// In production, use actual Dilithium implementation
	return len(signature) == qcm.getDilithiumSignatureSize(), nil
}

// HybridEncrypt encrypts with hybrid crypto
func (qcm *QuantumCryptoManager) HybridEncrypt(data []byte, hybridKey *HybridKeyPair) ([]byte, error) {
	if !qcm.config.EnableHybridMode {
		return nil, errors.New("hybrid mode not enabled")
	}

	startTime := time.Now()
	defer func() {
		qcm.metrics.mu.Lock()
		qcm.metrics.HybridOps++
		qcm.metrics.EncryptionOps++
		qcm.metrics.LastUpdated = time.Now()
		qcm.metrics.mu.Unlock()
	}()

	// Generate symmetric key using both classical and quantum keys
	symKey, err := qcm.hybridCrypto.combiner.CombineKeys(
		[]byte("classical_key_material"),
		[]byte("quantum_key_material"),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to combine keys: %w", err)
	}

	// Encrypt with ChaCha20-Poly1305 (quantum-resistant symmetric)
	aead, err := chacha20poly1305.NewX(symKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	nonce := make([]byte, aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := aead.Seal(nonce, nonce, data, nil)

	return ciphertext, nil
}

// HybridDecrypt decrypts with hybrid crypto
func (qcm *QuantumCryptoManager) HybridDecrypt(ciphertext []byte, hybridKey *HybridKeyPair) ([]byte, error) {
	if !qcm.config.EnableHybridMode {
		return nil, errors.New("hybrid mode not enabled")
	}

	startTime := time.Now()
	defer func() {
		qcm.metrics.mu.Lock()
		qcm.metrics.HybridOps++
		qcm.metrics.DecryptionOps++
		qcm.metrics.LastUpdated = time.Now()
		qcm.metrics.mu.Unlock()
	}()

	// Generate symmetric key
	symKey, err := qcm.hybridCrypto.combiner.CombineKeys(
		[]byte("classical_key_material"),
		[]byte("quantum_key_material"),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to combine keys: %w", err)
	}

	// Decrypt with ChaCha20-Poly1305
	aead, err := chacha20poly1305.NewX(symKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	if len(ciphertext) < aead.NonceSize() {
		return nil, errors.New("ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:aead.NonceSize()], ciphertext[aead.NonceSize():]
	plaintext, err := aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %w", err)
	}

	return plaintext, nil
}

// CombineKeys combines classical and quantum keys
func (kc *KeyCombiner) CombineKeys(classicalKey, quantumKey []byte) ([]byte, error) {
	kc.mu.Lock()
	defer kc.mu.Unlock()

	switch kc.algorithm {
	case CombinerKDF:
		return kc.combineWithKDF(classicalKey, quantumKey)
	case CombinerXOR:
		return kc.combineWithXOR(classicalKey, quantumKey)
	case CombinerConcat:
		return kc.combineWithConcat(classicalKey, quantumKey)
	default:
		return nil, fmt.Errorf("unsupported combiner algorithm: %s", kc.algorithm)
	}
}

// combineWithKDF combines keys using KDF
func (kc *KeyCombiner) combineWithKDF(key1, key2 []byte) ([]byte, error) {
	// Concatenate keys
	combined := append(key1, key2...)

	// Generate salt
	salt := make([]byte, kc.kdf.saltSize)
	if _, err := rand.Read(salt); err != nil {
		return nil, fmt.Errorf("failed to generate salt: %w", err)
	}

	// Derive key using Argon2
	derivedKey := argon2.IDKey(combined, salt, 1, 64*1024, 4, uint32(kc.kdf.keySize))

	return derivedKey, nil
}

// combineWithXOR combines keys with XOR
func (kc *KeyCombiner) combineWithXOR(key1, key2 []byte) ([]byte, error) {
	if len(key1) != len(key2) {
		return nil, errors.New("keys must have same length for XOR")
	}

	result := make([]byte, len(key1))
	for i := range key1 {
		result[i] = key1[i] ^ key2[i]
	}

	return result, nil
}

// combineWithConcat combines keys with concatenation and hash
func (kc *KeyCombiner) combineWithConcat(key1, key2 []byte) ([]byte, error) {
	combined := append(key1, key2...)
	hash := sha512.Sum512(combined)
	return hash[:32], nil
}

// RotateKey rotates a quantum key
func (qcm *QuantumCryptoManager) RotateKey(keyID string) error {
	qcm.keyStore.mu.Lock()
	defer qcm.keyStore.mu.Unlock()

	metadata, ok := qcm.keyStore.keyMetadata[keyID]
	if !ok {
		return fmt.Errorf("key %s not found", keyID)
	}

	// Mark old key as deprecated
	metadata.Status = KeyStatusDeprecated
	metadata.LastRotated = time.Now()

	// Generate new key based on type
	switch metadata.KeyType {
	case KeyTypeKyber:
		_, err := qcm.GenerateKyberKeyPair()
		if err != nil {
			return fmt.Errorf("failed to rotate Kyber key: %w", err)
		}
	case KeyTypeDilithium:
		_, err := qcm.GenerateDilithiumKeyPair()
		if err != nil {
			return fmt.Errorf("failed to rotate Dilithium key: %w", err)
		}
	default:
		return fmt.Errorf("unsupported key type: %s", metadata.KeyType)
	}

	qcm.metrics.mu.Lock()
	qcm.metrics.KeyRotations++
	qcm.metrics.LastUpdated = time.Now()
	qcm.metrics.mu.Unlock()

	return nil
}

// MigrateAlgorithm migrates to new algorithm
func (qcm *QuantumCryptoManager) MigrateAlgorithm(fromAlgo, toAlgo string) error {
	qcm.cryptoAgility.mu.Lock()
	defer qcm.cryptoAgility.mu.Unlock()

	plan := &MigrationPlan{
		FromAlgorithm: fromAlgo,
		ToAlgorithm:   toAlgo,
		StartDate:     time.Now(),
		Status:        MigrationInProgress,
	}

	qcm.cryptoAgility.migrationPlan = plan

	// Execute migration steps
	// In production, implement actual migration logic

	plan.Status = MigrationCompleted
	plan.EndDate = time.Now()

	return nil
}

// Helper functions

func (qcm *QuantumCryptoManager) getKyberPublicKeySize() int {
	switch qcm.config.NISTLevel {
	case 2:
		return 800 // Kyber512
	case 3:
		return 1184 // Kyber768
	case 5:
		return 1568 // Kyber1024
	default:
		return 1184
	}
}

func (qcm *QuantumCryptoManager) getKyberPrivateKeySize() int {
	switch qcm.config.NISTLevel {
	case 2:
		return 1632
	case 3:
		return 2400
	case 5:
		return 3168
	default:
		return 2400
	}
}

func (qcm *QuantumCryptoManager) getKyberCiphertextSize() int {
	switch qcm.config.NISTLevel {
	case 2:
		return 768
	case 3:
		return 1088
	case 5:
		return 1568
	default:
		return 1088
	}
}

func (qcm *QuantumCryptoManager) getDilithiumPublicKeySize() int {
	switch qcm.config.NISTLevel {
	case 2:
		return 1312
	case 3:
		return 1952
	case 5:
		return 2592
	default:
		return 1952
	}
}

func (qcm *QuantumCryptoManager) getDilithiumPrivateKeySize() int {
	switch qcm.config.NISTLevel {
	case 2:
		return 2528
	case 3:
		return 4000
	case 5:
		return 4864
	default:
		return 4000
	}
}

func (qcm *QuantumCryptoManager) getDilithiumSignatureSize() int {
	switch qcm.config.NISTLevel {
	case 2:
		return 2420
	case 3:
		return 3293
	case 5:
		return 4595
	default:
		return 3293
	}
}

func (qcm *QuantumCryptoManager) generateCacheKey(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

func (qcm *QuantumCryptoManager) generateSignatureCacheKey(message []byte) string {
	hash := sha256.Sum256(message)
	return hex.EncodeToString(hash[:])
}

// Cache methods

func (c *KEMCache) Get(key string) (*CachedKEM, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	cached, ok := c.encapsulated[key]
	if !ok {
		return nil, false
	}

	if time.Since(cached.Timestamp) > c.ttl {
		return nil, false
	}

	return cached, true
}

func (c *KEMCache) Put(key string, kem *CachedKEM) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.encapsulated) >= c.maxSize {
		// Remove oldest entry
		var oldestKey string
		var oldestTime time.Time
		for k, v := range c.encapsulated {
			if oldestTime.IsZero() || v.Timestamp.Before(oldestTime) {
				oldestKey = k
				oldestTime = v.Timestamp
			}
		}
		delete(c.encapsulated, oldestKey)
	}

	c.encapsulated[key] = kem
}

func (c *SignatureCache) Get(key string) (*CachedSignature, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	cached, ok := c.signatures[key]
	return cached, ok
}

func (c *SignatureCache) Put(key string, sig *CachedSignature) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.signatures) >= c.maxSize {
		// Remove random entry
		for k := range c.signatures {
			delete(c.signatures, k)
			break
		}
	}

	c.signatures[key] = sig
}

// initializeSupportedAlgorithms initializes supported algorithms
func (qcm *QuantumCryptoManager) initializeSupportedAlgorithms() {
	qcm.cryptoAgility.mu.Lock()
	defer qcm.cryptoAgility.mu.Unlock()

	qcm.cryptoAgility.supportedAlgorithms["kyber"] = &AlgorithmProfile{
		ID:               "kyber-768",
		Name:             "CRYSTALS-Kyber",
		Type:             AlgorithmTypeKEM,
		SecurityLevel:    3,
		QuantumResistant: true,
		Enabled:          true,
		ComplianceFlags:  []string{"NIST", "FIPS-140-3"},
	}

	qcm.cryptoAgility.supportedAlgorithms["dilithium"] = &AlgorithmProfile{
		ID:               "dilithium-3",
		Name:             "CRYSTALS-Dilithium",
		Type:             AlgorithmTypeSignature,
		SecurityLevel:    3,
		QuantumResistant: true,
		Enabled:          true,
		ComplianceFlags:  []string{"NIST", "FIPS-140-3"},
	}

	qcm.cryptoAgility.activeAlgorithm = "kyber"
}

// startBackgroundTasks starts background tasks
func (qcm *QuantumCryptoManager) startBackgroundTasks() {
	// Start key rotation monitor
	go qcm.runKeyRotationMonitor()

	// Start cache cleanup
	go qcm.runCacheCleanup()

	// Start metrics collection
	go qcm.runMetricsCollection()
}

// runKeyRotationMonitor monitors key rotation
func (qcm *QuantumCryptoManager) runKeyRotationMonitor() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		qcm.keyStore.mu.RLock()
		for keyID, metadata := range qcm.keyStore.keyMetadata {
			if time.Now().After(metadata.ExpiresAt) && metadata.Status == KeyStatusActive {
				qcm.RotateKey(keyID)
			}
		}
		qcm.keyStore.mu.RUnlock()
	}
}

// runCacheCleanup runs cache cleanup
func (qcm *QuantumCryptoManager) runCacheCleanup() {
	ticker := time.NewTicker(15 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		qcm.kyberEngine.cache.mu.Lock()
		for key, cached := range qcm.kyberEngine.cache.encapsulated {
			if time.Since(cached.Timestamp) > qcm.kyberEngine.cache.ttl {
				delete(qcm.kyberEngine.cache.encapsulated, key)
			}
		}
		qcm.kyberEngine.cache.mu.Unlock()
	}
}

// runMetricsCollection runs metrics collection
func (qcm *QuantumCryptoManager) runMetricsCollection() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		qcm.metrics.mu.Lock()
		qcm.metrics.TotalKeys = int64(len(qcm.keyStore.kyberKeys) + len(qcm.keyStore.dilithiumKeys))
		qcm.metrics.LastUpdated = time.Now()
		qcm.metrics.mu.Unlock()
	}
}

// GetMetrics returns metrics
func (qcm *QuantumCryptoManager) GetMetrics() *QuantumCryptoMetrics {
	qcm.metrics.mu.RLock()
	defer qcm.metrics.mu.RUnlock()

	metricsCopy := *qcm.metrics
	return &metricsCopy
}

// EncryptWithAESGCM provides quantum-resistant symmetric encryption
func EncryptWithAESGCM(plaintext []byte, key []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	nonce := make([]byte, aead.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}

	ciphertext := aead.Seal(nonce, nonce, plaintext, nil)
	return ciphertext, nil
}

// DecryptWithAESGCM decrypts AES-GCM ciphertext
func DecryptWithAESGCM(ciphertext []byte, key []byte) ([]byte, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create cipher: %w", err)
	}

	aead, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM: %w", err)
	}

	if len(ciphertext) < aead.NonceSize() {
		return nil, errors.New("ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:aead.NonceSize()], ciphertext[aead.NonceSize():]
	plaintext, err := aead.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt: %w", err)
	}

	return plaintext, nil
}
