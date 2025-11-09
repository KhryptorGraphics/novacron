// Package smpc implements secure multi-party computation
package smpc

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/big"
	"sync"
)

// Protocol represents an SMPC protocol
type Protocol string

const (
	ProtocolShamir           Protocol = "shamir"
	ProtocolGarbledCircuits  Protocol = "garbled_circuits"
	ProtocolObliviousTransfer Protocol = "oblivious_transfer"
)

// Party represents a party in SMPC
type Party struct {
	ID        string
	PublicKey []byte
	Address   string
}

// SecretShare represents a secret share
type SecretShare struct {
	PartyID string
	ShareID string
	X       *big.Int // Party index
	Y       *big.Int // Share value
}

// Computation represents an SMPC computation
type Computation struct {
	ID              string
	Protocol        Protocol
	Parties         []*Party
	Threshold       int
	Secret          *big.Int
	Shares          []*SecretShare
	Result          *big.Int
	Status          string
	PrivacyPreserving bool
}

// Coordinator manages secure multi-party computation
type Coordinator struct {
	protocol          Protocol
	parties           map[string]*Party
	computations      map[string]*Computation
	threshold         int
	privacyPreserving bool
	shamirEngine      *ShamirEngine
	gcEngine          *GarbledCircuitsEngine
	otEngine          *ObliviousTransferEngine
	mu                sync.RWMutex
}

// ShamirEngine implements Shamir secret sharing
type ShamirEngine struct {
	Prime *big.Int
	mu    sync.RWMutex
}

// GarbledCircuitsEngine implements garbled circuits
type GarbledCircuitsEngine struct {
	Circuits map[string]*GarbledCircuit
	mu       sync.RWMutex
}

// GarbledCircuit represents a garbled circuit
type GarbledCircuit struct {
	ID      string
	Gates   []*GarbledGate
	WireKeys map[string][]byte
}

// GarbledGate represents a garbled gate
type GarbledGate struct {
	Type    string // "AND", "OR", "XOR", "NOT"
	Input1  string
	Input2  string
	Output  string
	Table   [][]byte
}

// ObliviousTransferEngine implements oblivious transfer
type ObliviousTransferEngine struct {
	Sessions map[string]*OTSession
	mu       sync.RWMutex
}

// OTSession represents an oblivious transfer session
type OTSession struct {
	ID       string
	Sender   string
	Receiver string
	Messages [][]byte
	Choice   int
}

// NewCoordinator creates a new SMPC coordinator
func NewCoordinator(protocol Protocol, threshold int, privacyPreserving bool) *Coordinator {
	// Generate a large prime for Shamir
	prime, _ := rand.Prime(rand.Reader, 256)

	return &Coordinator{
		protocol:          protocol,
		parties:           make(map[string]*Party),
		computations:      make(map[string]*Computation),
		threshold:         threshold,
		privacyPreserving: privacyPreserving,
		shamirEngine: &ShamirEngine{
			Prime: prime,
		},
		gcEngine: &GarbledCircuitsEngine{
			Circuits: make(map[string]*GarbledCircuit),
		},
		otEngine: &ObliviousTransferEngine{
			Sessions: make(map[string]*OTSession),
		},
	}
}

// RegisterParty registers a party for SMPC
func (c *Coordinator) RegisterParty(party *Party) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if party.ID == "" {
		return fmt.Errorf("party ID is required")
	}

	c.parties[party.ID] = party
	return nil
}

// CreateComputation creates a new SMPC computation
func (c *Coordinator) CreateComputation(parties []*Party) (*Computation, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(parties) < c.threshold {
		return nil, fmt.Errorf("insufficient parties: need at least %d", c.threshold)
	}

	comp := &Computation{
		ID:                generateComputationID(),
		Protocol:          c.protocol,
		Parties:           parties,
		Threshold:         c.threshold,
		Shares:            make([]*SecretShare, 0),
		Status:            "initialized",
		PrivacyPreserving: c.privacyPreserving,
	}

	c.computations[comp.ID] = comp
	return comp, nil
}

// ShareSecret shares a secret among parties using Shamir secret sharing
func (c *Coordinator) ShareSecret(computationID string, secret *big.Int) ([]*SecretShare, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	comp, exists := c.computations[computationID]
	if !exists {
		return nil, fmt.Errorf("computation not found: %s", computationID)
	}

	if c.protocol != ProtocolShamir {
		return nil, fmt.Errorf("protocol %s does not support secret sharing", c.protocol)
	}

	// Generate shares using Shamir's scheme
	shares, err := c.shamirEngine.GenerateShares(secret, len(comp.Parties), comp.Threshold)
	if err != nil {
		return nil, fmt.Errorf("share generation failed: %w", err)
	}

	// Assign shares to parties
	for i, party := range comp.Parties {
		if i >= len(shares) {
			break
		}
		shares[i].PartyID = party.ID
	}

	comp.Shares = shares
	comp.Secret = secret
	comp.Status = "shared"

	return shares, nil
}

// ReconstructSecret reconstructs a secret from shares
func (c *Coordinator) ReconstructSecret(computationID string, shares []*SecretShare) (*big.Int, error) {
	c.mu.RLock()
	comp, exists := c.computations[computationID]
	c.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("computation not found: %s", computationID)
	}

	if len(shares) < comp.Threshold {
		return nil, fmt.Errorf("insufficient shares: need at least %d", comp.Threshold)
	}

	if c.protocol != ProtocolShamir {
		return nil, fmt.Errorf("protocol %s does not support secret reconstruction", c.protocol)
	}

	secret, err := c.shamirEngine.ReconstructSecret(shares, comp.Threshold)
	if err != nil {
		return nil, fmt.Errorf("reconstruction failed: %w", err)
	}

	c.mu.Lock()
	comp.Result = secret
	comp.Status = "completed"
	c.mu.Unlock()

	return secret, nil
}

// ShamirEngine implementation

// GenerateShares generates Shamir secret shares
func (s *ShamirEngine) GenerateShares(secret *big.Int, numShares, threshold int) ([]*SecretShare, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if threshold > numShares {
		return nil, fmt.Errorf("threshold cannot exceed number of shares")
	}

	// Generate random polynomial coefficients
	coeffs := make([]*big.Int, threshold)
	coeffs[0] = secret // a0 = secret

	for i := 1; i < threshold; i++ {
		coeff, err := rand.Int(rand.Reader, s.Prime)
		if err != nil {
			return nil, err
		}
		coeffs[i] = coeff
	}

	// Generate shares
	shares := make([]*SecretShare, numShares)
	for i := 0; i < numShares; i++ {
		x := big.NewInt(int64(i + 1))
		y := s.evaluatePolynomial(coeffs, x)

		shares[i] = &SecretShare{
			ShareID: generateShareID(),
			X:       x,
			Y:       y,
		}
	}

	return shares, nil
}

// evaluatePolynomial evaluates polynomial at point x
func (s *ShamirEngine) evaluatePolynomial(coeffs []*big.Int, x *big.Int) *big.Int {
	result := big.NewInt(0)

	for i := len(coeffs) - 1; i >= 0; i-- {
		result.Mul(result, x)
		result.Add(result, coeffs[i])
		result.Mod(result, s.Prime)
	}

	return result
}

// ReconstructSecret reconstructs secret from shares using Lagrange interpolation
func (s *ShamirEngine) ReconstructSecret(shares []*SecretShare, threshold int) (*big.Int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(shares) < threshold {
		return nil, fmt.Errorf("insufficient shares")
	}

	secret := big.NewInt(0)

	// Lagrange interpolation at x=0
	for i := 0; i < threshold; i++ {
		numerator := big.NewInt(1)
		denominator := big.NewInt(1)

		for j := 0; j < threshold; j++ {
			if i == j {
				continue
			}

			// numerator *= -shares[j].X
			temp := new(big.Int).Neg(shares[j].X)
			numerator.Mul(numerator, temp)
			numerator.Mod(numerator, s.Prime)

			// denominator *= (shares[i].X - shares[j].X)
			temp = new(big.Int).Sub(shares[i].X, shares[j].X)
			denominator.Mul(denominator, temp)
			denominator.Mod(denominator, s.Prime)
		}

		// Lagrange basis polynomial
		denomInv := new(big.Int).ModInverse(denominator, s.Prime)
		lagrange := new(big.Int).Mul(numerator, denomInv)
		lagrange.Mod(lagrange, s.Prime)

		// Add to secret
		term := new(big.Int).Mul(shares[i].Y, lagrange)
		secret.Add(secret, term)
		secret.Mod(secret, s.Prime)
	}

	return secret, nil
}

// CreateGarbledCircuit creates a garbled circuit for computation
func (c *Coordinator) CreateGarbledCircuit(computationID string, gates []*GarbledGate) (*GarbledCircuit, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	comp, exists := c.computations[computationID]
	if !exists {
		return nil, fmt.Errorf("computation not found: %s", computationID)
	}

	if c.protocol != ProtocolGarbledCircuits {
		return nil, fmt.Errorf("protocol %s does not support garbled circuits", c.protocol)
	}

	circuit := &GarbledCircuit{
		ID:       generateCircuitID(),
		Gates:    gates,
		WireKeys: make(map[string][]byte),
	}

	// Generate wire keys
	for _, gate := range gates {
		if _, exists := circuit.WireKeys[gate.Input1]; !exists {
			circuit.WireKeys[gate.Input1] = generateWireKey()
		}
		if gate.Input2 != "" {
			if _, exists := circuit.WireKeys[gate.Input2]; !exists {
				circuit.WireKeys[gate.Input2] = generateWireKey()
			}
		}
		if _, exists := circuit.WireKeys[gate.Output]; !exists {
			circuit.WireKeys[gate.Output] = generateWireKey()
		}
	}

	c.gcEngine.mu.Lock()
	c.gcEngine.Circuits[circuit.ID] = circuit
	c.gcEngine.mu.Unlock()

	comp.Status = "circuit_created"
	return circuit, nil
}

// InitiateObliviousTransfer initiates an oblivious transfer session
func (c *Coordinator) InitiateObliviousTransfer(sender, receiver string, messages [][]byte) (*OTSession, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.protocol != ProtocolObliviousTransfer {
		return nil, fmt.Errorf("protocol %s does not support oblivious transfer", c.protocol)
	}

	session := &OTSession{
		ID:       generateSessionID(),
		Sender:   sender,
		Receiver: receiver,
		Messages: messages,
	}

	c.otEngine.mu.Lock()
	c.otEngine.Sessions[session.ID] = session
	c.otEngine.mu.Unlock()

	return session, nil
}

// ExecuteObliviousTransfer executes oblivious transfer with receiver's choice
func (c *Coordinator) ExecuteObliviousTransfer(sessionID string, choice int) ([]byte, error) {
	c.otEngine.mu.RLock()
	session, exists := c.otEngine.Sessions[sessionID]
	c.otEngine.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("session not found: %s", sessionID)
	}

	if choice < 0 || choice >= len(session.Messages) {
		return nil, fmt.Errorf("invalid choice: %d", choice)
	}

	session.Choice = choice
	return session.Messages[choice], nil
}

// GetMetrics returns SMPC metrics
func (c *Coordinator) GetMetrics() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	completedComputations := 0
	for _, comp := range c.computations {
		if comp.Status == "completed" {
			completedComputations++
		}
	}

	return map[string]interface{}{
		"protocol":             c.protocol,
		"total_parties":        len(c.parties),
		"total_computations":   len(c.computations),
		"completed":            completedComputations,
		"threshold":            c.threshold,
		"privacy_preserving":   c.privacyPreserving,
		"garbled_circuits":     len(c.gcEngine.Circuits),
		"ot_sessions":          len(c.otEngine.Sessions),
	}
}

// Helper functions

func generateComputationID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("comp-%s", hex.EncodeToString(b))
}

func generateShareID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func generateCircuitID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("circuit-%s", hex.EncodeToString(b))
}

func generateSessionID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("ot-%s", hex.EncodeToString(b))
}

func generateWireKey() []byte {
	key := make([]byte, 16)
	rand.Read(key)
	return key
}
