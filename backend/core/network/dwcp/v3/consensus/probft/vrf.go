// Package probft implements Probabilistic Byzantine Fault Tolerance consensus
// with Verifiable Random Functions for leader election
package probft

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
)

// VRF implements a Verifiable Random Function using Ed25519
// VRF provides cryptographically secure randomness with public verifiability
type VRF struct {
	privateKey ed25519.PrivateKey
	publicKey  ed25519.PublicKey
}

// VRFProof contains the proof and output of a VRF evaluation
type VRFProof struct {
	Proof  []byte // Cryptographic proof of correct VRF evaluation
	Output []byte // Deterministic pseudorandom output
}

// NewVRF creates a new VRF instance with a fresh keypair
func NewVRF() (*VRF, error) {
	publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate VRF keypair: %w", err)
	}

	return &VRF{
		privateKey: privateKey,
		publicKey:  publicKey,
	}, nil
}

// NewVRFFromKeys creates a VRF instance from existing keys
func NewVRFFromKeys(privateKey ed25519.PrivateKey, publicKey ed25519.PublicKey) (*VRF, error) {
	if len(privateKey) != ed25519.PrivateKeySize {
		return nil, errors.New("invalid private key size")
	}
	if len(publicKey) != ed25519.PublicKeySize {
		return nil, errors.New("invalid public key size")
	}

	return &VRF{
		privateKey: privateKey,
		publicKey:  publicKey,
	}, nil
}

// Prove generates a VRF proof and output for the given input
// The output is deterministic but unpredictable without the private key
func (v *VRF) Prove(input []byte) (*VRFProof, error) {
	if v.privateKey == nil {
		return nil, errors.New("VRF private key not set")
	}

	// Create deterministic hash of input
	h := sha256.New()
	h.Write(input)
	inputHash := h.Sum(nil)

	// Sign the input hash to create proof
	proof := ed25519.Sign(v.privateKey, inputHash)

	// Generate output by hashing proof
	outputHash := sha256.New()
	outputHash.Write(proof)
	outputHash.Write(inputHash)
	output := outputHash.Sum(nil)

	return &VRFProof{
		Proof:  proof,
		Output: output,
	}, nil
}

// Verify verifies a VRF proof and output using the public key
func (v *VRF) Verify(input []byte, proof *VRFProof) bool {
	return VerifyVRF(v.publicKey, input, proof)
}

// VerifyVRF verifies a VRF proof using any public key
func VerifyVRF(publicKey ed25519.PublicKey, input []byte, proof *VRFProof) bool {
	if len(publicKey) != ed25519.PublicKeySize {
		return false
	}

	// Recreate input hash
	h := sha256.New()
	h.Write(input)
	inputHash := h.Sum(nil)

	// Verify signature
	if !ed25519.Verify(publicKey, inputHash, proof.Proof) {
		return false
	}

	// Recreate output
	outputHash := sha256.New()
	outputHash.Write(proof.Proof)
	outputHash.Write(inputHash)
	expectedOutput := outputHash.Sum(nil)

	// Compare outputs
	if len(proof.Output) != len(expectedOutput) {
		return false
	}

	for i := range proof.Output {
		if proof.Output[i] != expectedOutput[i] {
			return false
		}
	}

	return true
}

// GetPublicKey returns the VRF public key
func (v *VRF) GetPublicKey() ed25519.PublicKey {
	return v.publicKey
}

// OutputToUint64 converts VRF output to uint64 for leader election
func OutputToUint64(output []byte) uint64 {
	if len(output) < 8 {
		// Pad if necessary
		padded := make([]byte, 8)
		copy(padded, output)
		return binary.BigEndian.Uint64(padded)
	}
	return binary.BigEndian.Uint64(output[:8])
}

// SelectLeader uses VRF output to select a leader from a set of validators
func SelectLeader(vrfOutput []byte, validatorCount int) int {
	if validatorCount <= 0 {
		return 0
	}

	value := OutputToUint64(vrfOutput)
	return int(value % uint64(validatorCount))
}

// CompareVRFOutputs compares two VRF outputs, returns -1, 0, or 1
// Used for deterministic leader selection when multiple nodes produce VRFs
func CompareVRFOutputs(a, b []byte) int {
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	for i := 0; i < minLen; i++ {
		if a[i] < b[i] {
			return -1
		}
		if a[i] > b[i] {
			return 1
		}
	}

	if len(a) < len(b) {
		return -1
	}
	if len(a) > len(b) {
		return 1
	}
	return 0
}
