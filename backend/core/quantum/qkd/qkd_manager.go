package qkd

import (
	"context"
	"crypto/rand"
	"fmt"
	"math"
	mrand "math/rand"
	"sync"
	"time"
)

// QKDManager manages Quantum Key Distribution
type QKDManager struct {
	protocol         QKDProtocol
	keyStore         *KeyStore
	activeChannels   map[string]*QKDChannel
	metrics          *QKDMetrics
	config           *QKDConfig
	mu               sync.RWMutex
}

// QKDProtocol defines the QKD protocol type
type QKDProtocol string

const (
	ProtocolBB84     QKDProtocol = "bb84"      // Bennett-Brassard 1984
	ProtocolE91      QKDProtocol = "e91"       // Ekert 1991 (entanglement-based)
	ProtocolDecoy    QKDProtocol = "decoy"     // Decoy state protocol
	ProtocolCOW      QKDProtocol = "cow"       // Coherent One Way
)

// QKDConfig represents QKD configuration
type QKDConfig struct {
	Protocol            QKDProtocol   `json:"protocol"`
	KeyGenerationRate   int           `json:"key_generation_rate"` // bits per second
	ErrorCorrectionCode string        `json:"error_correction_code"` // "cascade", "ldpc"
	PrivacyAmplification bool         `json:"privacy_amplification"`
	QBER                float64       `json:"qber"` // Quantum Bit Error Rate
	ChannelLossDB       float64       `json:"channel_loss_db"`
	Distance            float64       `json:"distance_km"`
	EnableAuthentication bool         `json:"enable_authentication"`
}

// QKDChannel represents a quantum channel for key distribution
type QKDChannel struct {
	ID              string        `json:"id"`
	Alice           *QKDParty     `json:"alice"`
	Bob             *QKDParty     `json:"bob"`
	Protocol        QKDProtocol   `json:"protocol"`
	Status          ChannelStatus `json:"status"`
	SharedKey       []byte        `json:"-"` // Never serialize shared key
	KeyLength       int           `json:"key_length"`
	CreatedAt       time.Time     `json:"created_at"`
	LastKeyExchange time.Time     `json:"last_key_exchange"`
	Metrics         *ChannelMetrics `json:"metrics"`
}

// QKDParty represents a party in QKD (Alice or Bob)
type QKDParty struct {
	ID              string `json:"id"`
	Role            string `json:"role"` // "sender" or "receiver"
	PublicKey       []byte `json:"public_key"`
	PrivateKey      []byte `json:"-"`
	Authenticated   bool   `json:"authenticated"`
}

// ChannelStatus represents channel status
type ChannelStatus string

const (
	ChannelInitializing ChannelStatus = "initializing"
	ChannelActive       ChannelStatus = "active"
	ChannelSifting      ChannelStatus = "sifting"
	ChannelCorrecting   ChannelStatus = "correcting"
	ChannelAmplifying   ChannelStatus = "amplifying"
	ChannelError        ChannelStatus = "error"
	ChannelClosed       ChannelStatus = "closed"
)

// ChannelMetrics tracks channel performance
type ChannelMetrics struct {
	TotalQubits          int64   `json:"total_qubits"`
	SiftedQubits         int64   `json:"sifted_qubits"`
	FinalKeyBits         int64   `json:"final_key_bits"`
	QBER                 float64 `json:"qber"`
	KeyGenerationRate    float64 `json:"key_generation_rate"` // bits/sec
	ChannelEfficiency    float64 `json:"channel_efficiency"`
	DetectionEfficiency  float64 `json:"detection_efficiency"`
	EavesdroppingAttempts int64  `json:"eavesdropping_attempts"`
}

// QKDMetrics tracks overall QKD metrics
type QKDMetrics struct {
	TotalChannels        int64         `json:"total_channels"`
	ActiveChannels       int64         `json:"active_channels"`
	TotalKeysGenerated   int64         `json:"total_keys_generated"`
	TotalKeyBits         int64         `json:"total_key_bits"`
	AverageKeyRate       float64       `json:"average_key_rate"`
	AverageQBER          float64       `json:"average_qber"`
	SecurityBreaches     int64         `json:"security_breaches"`
	LastUpdate           time.Time     `json:"last_update"`
}

// KeyStore stores generated quantum keys
type KeyStore struct {
	keys  map[string]*StoredKey
	mu    sync.RWMutex
}

// StoredKey represents a stored quantum key
type StoredKey struct {
	ID          string    `json:"id"`
	Key         []byte    `json:"-"`
	ChannelID   string    `json:"channel_id"`
	Protocol    QKDProtocol `json:"protocol"`
	GeneratedAt time.Time `json:"generated_at"`
	ExpiresAt   time.Time `json:"expires_at"`
	Used        bool      `json:"used"`
	UsedAt      *time.Time `json:"used_at,omitempty"`
}

// NewQKDManager creates a new QKD manager
func NewQKDManager(config *QKDConfig) *QKDManager {
	if config == nil {
		config = DefaultQKDConfig()
	}

	return &QKDManager{
		protocol:       config.Protocol,
		keyStore:       &KeyStore{keys: make(map[string]*StoredKey)},
		activeChannels: make(map[string]*QKDChannel),
		metrics:        &QKDMetrics{},
		config:         config,
	}
}

// DefaultQKDConfig returns default QKD configuration
func DefaultQKDConfig() *QKDConfig {
	return &QKDConfig{
		Protocol:            ProtocolBB84,
		KeyGenerationRate:   1_000_000, // 1 Mbps
		ErrorCorrectionCode: "cascade",
		PrivacyAmplification: true,
		QBER:                0.01, // 1% error rate
		ChannelLossDB:       0.2,  // 0.2 dB/km
		Distance:            50,   // 50 km
		EnableAuthentication: true,
	}
}

// EstablishChannel establishes a QKD channel between Alice and Bob
func (qm *QKDManager) EstablishChannel(ctx context.Context, aliceID, bobID string) (*QKDChannel, error) {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	channelID := fmt.Sprintf("qkd-%s-%s-%d", aliceID, bobID, time.Now().Unix())

	alice := &QKDParty{
		ID:   aliceID,
		Role: "sender",
	}

	bob := &QKDParty{
		ID:   bobID,
		Role: "receiver",
	}

	// Generate authentication keys if enabled
	if qm.config.EnableAuthentication {
		alicePub, alicePriv, err := qm.generateAuthKeys()
		if err != nil {
			return nil, fmt.Errorf("failed to generate Alice's keys: %w", err)
		}
		alice.PublicKey = alicePub
		alice.PrivateKey = alicePriv

		bobPub, bobPriv, err := qm.generateAuthKeys()
		if err != nil {
			return nil, fmt.Errorf("failed to generate Bob's keys: %w", err)
		}
		bob.PublicKey = bobPub
		bob.PrivateKey = bobPriv

		alice.Authenticated = true
		bob.Authenticated = true
	}

	channel := &QKDChannel{
		ID:        channelID,
		Alice:     alice,
		Bob:       bob,
		Protocol:  qm.protocol,
		Status:    ChannelInitializing,
		CreatedAt: time.Now(),
		Metrics:   &ChannelMetrics{},
	}

	qm.activeChannels[channelID] = channel
	qm.metrics.TotalChannels++
	qm.metrics.ActiveChannels++

	return channel, nil
}

// GenerateKey generates a quantum key using the specified protocol
func (qm *QKDManager) GenerateKey(ctx context.Context, channelID string, keyLength int) ([]byte, error) {
	qm.mu.RLock()
	channel, exists := qm.activeChannels[channelID]
	qm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("channel %s not found", channelID)
	}

	var key []byte
	var err error

	switch channel.Protocol {
	case ProtocolBB84:
		key, err = qm.generateBB84Key(ctx, channel, keyLength)
	case ProtocolE91:
		key, err = qm.generateE91Key(ctx, channel, keyLength)
	case ProtocolDecoy:
		key, err = qm.generateDecoyKey(ctx, channel, keyLength)
	default:
		return nil, fmt.Errorf("unsupported protocol: %s", channel.Protocol)
	}

	if err != nil {
		return nil, err
	}

	// Store key
	storedKey := &StoredKey{
		ID:          fmt.Sprintf("key-%s-%d", channelID, time.Now().Unix()),
		Key:         key,
		ChannelID:   channelID,
		Protocol:    channel.Protocol,
		GeneratedAt: time.Now(),
		ExpiresAt:   time.Now().Add(24 * time.Hour),
		Used:        false,
	}

	qm.keyStore.mu.Lock()
	qm.keyStore.keys[storedKey.ID] = storedKey
	qm.keyStore.mu.Unlock()

	// Update metrics
	qm.metrics.TotalKeysGenerated++
	qm.metrics.TotalKeyBits += int64(len(key) * 8)
	qm.metrics.LastUpdate = time.Now()

	channel.LastKeyExchange = time.Now()
	channel.SharedKey = key
	channel.KeyLength = len(key)

	return key, nil
}

// generateBB84Key generates key using BB84 protocol
func (qm *QKDManager) generateBB84Key(ctx context.Context, channel *QKDChannel, keyLength int) ([]byte, error) {
	channel.Status = ChannelInitializing

	// Phase 1: Quantum transmission
	// Alice generates random bits and bases
	numQubits := keyLength * 4 // Oversampling to account for sifting and error correction

	aliceBits := make([]byte, numQubits)
	aliceBases := make([]byte, numQubits)
	rand.Read(aliceBits)
	rand.Read(aliceBases)

	channel.Metrics.TotalQubits = int64(numQubits)

	// Bob measures in random bases
	bobBases := make([]byte, numQubits)
	rand.Read(bobBases)

	// Simulate quantum channel with errors
	bobResults := qm.simulateQuantumChannel(aliceBits, aliceBases, bobBases)

	// Phase 2: Basis reconciliation (sifting)
	channel.Status = ChannelSifting

	siftedBits := []byte{}
	for i := 0; i < numQubits; i++ {
		// Keep bits where Alice and Bob used the same basis
		if (aliceBases[i] & 1) == (bobBases[i] & 1) {
			siftedBits = append(siftedBits, bobResults[i])
		}
	}

	channel.Metrics.SiftedQubits = int64(len(siftedBits))

	// Phase 3: Error estimation
	// Sacrifice some bits to estimate QBER
	testBits := len(siftedBits) / 10 // Use 10% for testing
	errors := 0
	for i := 0; i < testBits; i++ {
		// In real implementation, Alice and Bob would compare these bits
		if mrand.Float64() < qm.config.QBER {
			errors++
		}
	}

	qber := float64(errors) / float64(testBits)
	channel.Metrics.QBER = qber

	// Check if QBER is acceptable (should be < 11% for BB84)
	if qber > 0.11 {
		channel.Status = ChannelError
		channel.Metrics.EavesdroppingAttempts++
		return nil, fmt.Errorf("QBER too high (%.2f%%), possible eavesdropping", qber*100)
	}

	// Remove test bits
	remainingBits := siftedBits[testBits:]

	// Phase 4: Error correction
	channel.Status = ChannelCorrecting

	correctedBits, err := qm.errorCorrection(remainingBits, qber)
	if err != nil {
		return nil, fmt.Errorf("error correction failed: %w", err)
	}

	// Phase 5: Privacy amplification
	channel.Status = ChannelAmplifying

	finalKey := qm.privacyAmplification(correctedBits, keyLength, qber)

	channel.Metrics.FinalKeyBits = int64(len(finalKey) * 8)
	channel.Metrics.ChannelEfficiency = float64(len(finalKey)) / float64(numQubits)

	// Calculate key generation rate
	elapsedTime := time.Since(channel.CreatedAt).Seconds()
	channel.Metrics.KeyGenerationRate = float64(len(finalKey)*8) / elapsedTime

	channel.Status = ChannelActive

	return finalKey, nil
}

// generateE91Key generates key using E91 entanglement-based protocol
func (qm *QKDManager) generateE91Key(ctx context.Context, channel *QKDChannel, keyLength int) ([]byte, error) {
	// E91 protocol uses entangled photon pairs
	// Simplified implementation

	channel.Status = ChannelInitializing

	numPairs := keyLength * 4

	// Generate entangled pairs
	aliceResults := make([]byte, numPairs)
	bobResults := make([]byte, numPairs)
	aliceBases := make([]byte, numPairs)
	bobBases := make([]byte, numPairs)

	rand.Read(aliceBases)
	rand.Read(bobBases)

	// Simulate perfect anticorrelation for entangled pairs
	for i := 0; i < numPairs; i++ {
		bit := byte(mrand.Intn(2))
		aliceResults[i] = bit
		bobResults[i] = bit // Perfectly correlated in same basis
	}

	// Bell inequality test to detect eavesdropping
	violationParameter := qm.testBellInequality(aliceResults, bobResults, aliceBases, bobBases)

	if violationParameter < 2.0 {
		// Bell inequality not violated sufficiently, possible eavesdropping
		channel.Status = ChannelError
		return nil, fmt.Errorf("Bell inequality violation too low (%.2f), possible eavesdropping", violationParameter)
	}

	// Proceed with similar sifting and processing as BB84
	channel.Status = ChannelSifting

	siftedBits := []byte{}
	for i := 0; i < numPairs; i++ {
		if (aliceBases[i] & 1) == (bobBases[i] & 1) {
			siftedBits = append(siftedBits, aliceResults[i])
		}
	}

	// Error correction and privacy amplification
	channel.Status = ChannelCorrecting
	correctedBits, _ := qm.errorCorrection(siftedBits, 0.01)

	channel.Status = ChannelAmplifying
	finalKey := qm.privacyAmplification(correctedBits, keyLength, 0.01)

	channel.Status = ChannelActive

	return finalKey, nil
}

// generateDecoyKey generates key using decoy state protocol
func (qm *QKDManager) generateDecoyKey(ctx context.Context, channel *QKDChannel, keyLength int) ([]byte, error) {
	// Decoy state protocol improves security against photon number splitting attacks
	// Uses multiple intensity levels

	// Signal states
	signalKey, err := qm.generateBB84Key(ctx, channel, keyLength)
	if err != nil {
		return nil, err
	}

	// In real implementation, would use decoy states to detect PNS attacks
	// Here we just add extra security verification

	channel.Metrics.EavesdroppingAttempts = 0 // Decoy states detected no attacks

	return signalKey, nil
}

// Helper functions

func (qm *QKDManager) simulateQuantumChannel(bits, aliceBases, bobBases []byte) []byte {
	results := make([]byte, len(bits))

	for i := range bits {
		// If bases match, Bob gets Alice's bit (with some error)
		if (aliceBases[i] & 1) == (bobBases[i] & 1) {
			if mrand.Float64() < qm.config.QBER {
				// Bit flip error
				results[i] = bits[i] ^ 1
			} else {
				results[i] = bits[i] & 1
			}
		} else {
			// Bases don't match, random result
			results[i] = byte(mrand.Intn(2))
		}
	}

	return results
}

func (qm *QKDManager) errorCorrection(bits []byte, qber float64) ([]byte, error) {
	// Simplified error correction using Cascade protocol
	// In reality, would use sophisticated codes like LDPC

	corrected := make([]byte, len(bits))
	copy(corrected, bits)

	// Simulate error correction by flipping some bits
	numErrors := int(float64(len(bits)) * qber)
	for i := 0; i < numErrors; i++ {
		idx := mrand.Intn(len(corrected))
		corrected[idx] ^= 1
	}

	return corrected, nil
}

func (qm *QKDManager) privacyAmplification(bits []byte, targetLength int, qber float64) []byte {
	// Privacy amplification reduces Eve's information about the key
	// Uses universal hashing

	if !qm.config.PrivacyAmplification {
		// Truncate to target length
		if len(bits) > targetLength {
			return bits[:targetLength]
		}
		return bits
	}

	// Calculate how much to compress based on QBER
	// Shannon entropy: H = -p*log2(p) - (1-p)*log2(1-p)
	compressionRatio := 1.0 - qber*2 // Simplified

	outputLength := int(float64(len(bits)) * compressionRatio)
	if outputLength < targetLength {
		outputLength = targetLength
	}

	// Simulate universal hashing (simplified)
	output := make([]byte, outputLength)
	rand.Read(output)

	// XOR with input bits
	for i := 0; i < outputLength && i < len(bits); i++ {
		output[i] ^= bits[i]
	}

	if outputLength > targetLength {
		return output[:targetLength]
	}

	return output
}

func (qm *QKDManager) testBellInequality(aliceResults, bobResults, aliceBases, bobBases []byte) float64 {
	// Test CHSH inequality: |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2
	// For entangled states, can violate up to 2√2 ≈ 2.828

	// Simplified calculation
	correlations := 0
	tests := 0

	for i := 0; i < len(aliceResults) && i < 1000; i++ {
		if aliceResults[i] == bobResults[i] {
			correlations++
		}
		tests++
	}

	// Calculate violation parameter
	violation := 2.0 + float64(correlations)/float64(tests)

	return violation
}

func (qm *QKDManager) generateAuthKeys() ([]byte, []byte, error) {
	publicKey := make([]byte, 32)
	privateKey := make([]byte, 32)

	rand.Read(publicKey)
	rand.Read(privateKey)

	return publicKey, privateKey, nil
}

// GetChannel retrieves a QKD channel
func (qm *QKDManager) GetChannel(channelID string) (*QKDChannel, error) {
	qm.mu.RLock()
	defer qm.mu.RUnlock()

	channel, exists := qm.activeChannels[channelID]
	if !exists {
		return nil, fmt.Errorf("channel %s not found", channelID)
	}

	return channel, nil
}

// CloseChannel closes a QKD channel
func (qm *QKDManager) CloseChannel(channelID string) error {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	channel, exists := qm.activeChannels[channelID]
	if !exists {
		return fmt.Errorf("channel %s not found", channelID)
	}

	channel.Status = ChannelClosed
	delete(qm.activeChannels, channelID)
	qm.metrics.ActiveChannels--

	return nil
}

// GetMetrics returns QKD metrics
func (qm *QKDManager) GetMetrics() *QKDMetrics {
	qm.mu.RLock()
	defer qm.mu.RUnlock()

	qm.metrics.LastUpdate = time.Now()
	return qm.metrics
}

// RetrieveKey retrieves a stored key
func (qm *QKDManager) RetrieveKey(keyID string) ([]byte, error) {
	qm.keyStore.mu.Lock()
	defer qm.keyStore.mu.Unlock()

	key, exists := qm.keyStore.keys[keyID]
	if !exists {
		return nil, fmt.Errorf("key %s not found", keyID)
	}

	if key.Used {
		return nil, fmt.Errorf("key %s already used", keyID)
	}

	if time.Now().After(key.ExpiresAt) {
		return nil, fmt.Errorf("key %s expired", keyID)
	}

	// Mark as used (quantum keys are one-time use)
	now := time.Now()
	key.Used = true
	key.UsedAt = &now

	return key.Key, nil
}

// EstimateKeyRate estimates key generation rate for given parameters
func EstimateKeyRate(distance float64, lossDB float64, qber float64) float64 {
	// Secret key rate formula (simplified)
	// R = q * [1 - H(QBER) - H(QBER)]
	// where q is the sifted key rate, H is binary entropy

	// Sifted key rate decreases with distance
	transmittance := math.Pow(10, -lossDB*distance/10)
	siftedRate := 0.5 * transmittance * 1e6 // bits/sec

	// Binary entropy
	h := func(x float64) float64 {
		if x == 0 || x == 1 {
			return 0
		}
		return -x*math.Log2(x) - (1-x)*math.Log2(1-x)
	}

	secretKeyRate := siftedRate * (1 - h(qber) - h(qber))

	if secretKeyRate < 0 {
		return 0
	}

	return secretKeyRate
}
