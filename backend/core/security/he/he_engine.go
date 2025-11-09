// Package he implements homomorphic encryption
package he

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/big"
	"sync"
)

// Scheme represents a homomorphic encryption scheme
type Scheme string

const (
	SchemePHE  Scheme = "phe"  // Partially Homomorphic Encryption
	SchemeSHE  Scheme = "she"  // Somewhat Homomorphic Encryption
	SchemeLFHE Scheme = "lfhe" // Leveled Fully Homomorphic Encryption
)

// HEKeyPair represents a homomorphic encryption key pair
type HEKeyPair struct {
	PublicKey  *PublicKey
	PrivateKey *PrivateKey
	Scheme     Scheme
	KeySize    int
}

// PublicKey represents a public key
type PublicKey struct {
	N *big.Int // Modulus
	G *big.Int // Generator
}

// PrivateKey represents a private key
type PrivateKey struct {
	Lambda *big.Int
	Mu     *big.Int
}

// Ciphertext represents encrypted data
type Ciphertext struct {
	Value  *big.Int
	Level  int // Noise level for leveled FHE
	Scheme Scheme
}

// Engine implements homomorphic encryption
type Engine struct {
	scheme        Scheme
	securityLevel int
	keySize       int
	keyPair       *HEKeyPair
	mu            sync.RWMutex
	operations    int64
}

// NewEngine creates a new homomorphic encryption engine
func NewEngine(scheme Scheme, securityLevel, keySize int) *Engine {
	return &Engine{
		scheme:        scheme,
		securityLevel: securityLevel,
		keySize:       keySize,
	}
}

// GenerateKeyPair generates a new key pair
func (e *Engine) GenerateKeyPair() (*HEKeyPair, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	var keyPair *HEKeyPair
	var err error

	switch e.scheme {
	case SchemePHE:
		keyPair, err = e.generatePaillierKeyPair()
	case SchemeSHE, SchemeLFHE:
		keyPair, err = e.generateBGVKeyPair()
	default:
		return nil, fmt.Errorf("unsupported scheme: %s", e.scheme)
	}

	if err != nil {
		return nil, fmt.Errorf("key generation failed: %w", err)
	}

	e.keyPair = keyPair
	return keyPair, nil
}

// generatePaillierKeyPair generates a Paillier key pair (PHE)
func (e *Engine) generatePaillierKeyPair() (*HEKeyPair, error) {
	// Generate two large primes
	p, err := rand.Prime(rand.Reader, e.keySize/2)
	if err != nil {
		return nil, err
	}

	q, err := rand.Prime(rand.Reader, e.keySize/2)
	if err != nil {
		return nil, err
	}

	// n = p * q
	n := new(big.Int).Mul(p, q)

	// g = n + 1 (simplified)
	g := new(big.Int).Add(n, big.NewInt(1))

	// lambda = lcm(p-1, q-1)
	pMinus1 := new(big.Int).Sub(p, big.NewInt(1))
	qMinus1 := new(big.Int).Sub(q, big.NewInt(1))
	lambda := lcm(pMinus1, qMinus1)

	// mu = (L(g^lambda mod n^2))^-1 mod n
	// Simplified calculation
	mu := new(big.Int).ModInverse(lambda, n)

	return &HEKeyPair{
		PublicKey: &PublicKey{
			N: n,
			G: g,
		},
		PrivateKey: &PrivateKey{
			Lambda: lambda,
			Mu:     mu,
		},
		Scheme:  SchemePHE,
		KeySize: e.keySize,
	}, nil
}

// generateBGVKeyPair generates a BGV-style key pair (SHE/LFHE)
func (e *Engine) generateBGVKeyPair() (*HEKeyPair, error) {
	// Simplified BGV key generation
	n, err := rand.Prime(rand.Reader, e.keySize)
	if err != nil {
		return nil, err
	}

	g := big.NewInt(2)

	s, err := rand.Prime(rand.Reader, e.keySize/2)
	if err != nil {
		return nil, err
	}

	return &HEKeyPair{
		PublicKey: &PublicKey{
			N: n,
			G: g,
		},
		PrivateKey: &PrivateKey{
			Lambda: s,
			Mu:     s,
		},
		Scheme:  e.scheme,
		KeySize: e.keySize,
	}, nil
}

// Encrypt encrypts a plaintext value
func (e *Engine) Encrypt(plaintext *big.Int) (*Ciphertext, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.keyPair == nil {
		return nil, fmt.Errorf("key pair not generated")
	}

	switch e.scheme {
	case SchemePHE:
		return e.encryptPaillier(plaintext)
	case SchemeSHE, SchemeLFHE:
		return e.encryptBGV(plaintext)
	default:
		return nil, fmt.Errorf("unsupported scheme: %s", e.scheme)
	}
}

// encryptPaillier encrypts using Paillier (PHE)
func (e *Engine) encryptPaillier(plaintext *big.Int) (*Ciphertext, error) {
	pub := e.keyPair.PublicKey

	// r random in Z*n
	r, err := rand.Int(rand.Reader, pub.N)
	if err != nil {
		return nil, err
	}

	// n^2
	nSquared := new(big.Int).Mul(pub.N, pub.N)

	// g^m mod n^2
	gm := new(big.Int).Exp(pub.G, plaintext, nSquared)

	// r^n mod n^2
	rn := new(big.Int).Exp(r, pub.N, nSquared)

	// c = g^m * r^n mod n^2
	c := new(big.Int).Mul(gm, rn)
	c.Mod(c, nSquared)

	return &Ciphertext{
		Value:  c,
		Level:  0,
		Scheme: SchemePHE,
	}, nil
}

// encryptBGV encrypts using BGV-style (SHE/LFHE)
func (e *Engine) encryptBGV(plaintext *big.Int) (*Ciphertext, error) {
	pub := e.keyPair.PublicKey

	// Simplified BGV encryption
	r, err := rand.Int(rand.Reader, pub.N)
	if err != nil {
		return nil, err
	}

	// c = (g^m * r) mod n
	gm := new(big.Int).Exp(pub.G, plaintext, pub.N)
	c := new(big.Int).Mul(gm, r)
	c.Mod(c, pub.N)

	level := 0
	if e.scheme == SchemeLFHE {
		level = 10 // Start with max noise budget
	}

	return &Ciphertext{
		Value:  c,
		Level:  level,
		Scheme: e.scheme,
	}, nil
}

// Decrypt decrypts a ciphertext
func (e *Engine) Decrypt(ciphertext *Ciphertext) (*big.Int, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.keyPair == nil {
		return nil, fmt.Errorf("key pair not generated")
	}

	switch ciphertext.Scheme {
	case SchemePHE:
		return e.decryptPaillier(ciphertext)
	case SchemeSHE, SchemeLFHE:
		return e.decryptBGV(ciphertext)
	default:
		return nil, fmt.Errorf("unsupported scheme: %s", ciphertext.Scheme)
	}
}

// decryptPaillier decrypts using Paillier
func (e *Engine) decryptPaillier(ciphertext *Ciphertext) (*big.Int, error) {
	priv := e.keyPair.PrivateKey
	pub := e.keyPair.PublicKey

	nSquared := new(big.Int).Mul(pub.N, pub.N)

	// c^lambda mod n^2
	cLambda := new(big.Int).Exp(ciphertext.Value, priv.Lambda, nSquared)

	// L(c^lambda mod n^2)
	l := lFunction(cLambda, pub.N)

	// m = L(c^lambda mod n^2) * mu mod n
	m := new(big.Int).Mul(l, priv.Mu)
	m.Mod(m, pub.N)

	return m, nil
}

// decryptBGV decrypts using BGV-style
func (e *Engine) decryptBGV(ciphertext *Ciphertext) (*big.Int, error) {
	priv := e.keyPair.PrivateKey
	pub := e.keyPair.PublicKey

	// Simplified BGV decryption
	// m = (c * s^-1) mod n
	sInv := new(big.Int).ModInverse(priv.Lambda, pub.N)
	m := new(big.Int).Mul(ciphertext.Value, sInv)
	m.Mod(m, pub.N)

	// Extract lower bits (simplified)
	mask := big.NewInt((1 << 32) - 1)
	m.And(m, mask)

	return m, nil
}

// Add performs homomorphic addition
func (e *Engine) Add(c1, c2 *Ciphertext) (*Ciphertext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if c1.Scheme != c2.Scheme {
		return nil, fmt.Errorf("ciphertexts must use the same scheme")
	}

	e.operations++

	switch c1.Scheme {
	case SchemePHE, SchemeSHE, SchemeLFHE:
		pub := e.keyPair.PublicKey
		if c1.Scheme == SchemePHE {
			// For Paillier: c1 + c2 = c1 * c2 mod n^2
			nSquared := new(big.Int).Mul(pub.N, pub.N)
			result := new(big.Int).Mul(c1.Value, c2.Value)
			result.Mod(result, nSquared)

			return &Ciphertext{
				Value:  result,
				Level:  c1.Level,
				Scheme: c1.Scheme,
			}, nil
		} else {
			// For BGV: c1 + c2 = (c1 + c2) mod n
			result := new(big.Int).Add(c1.Value, c2.Value)
			result.Mod(result, pub.N)

			newLevel := c1.Level
			if c2.Level < newLevel {
				newLevel = c2.Level
			}
			if newLevel > 0 {
				newLevel--
			}

			return &Ciphertext{
				Value:  result,
				Level:  newLevel,
				Scheme: c1.Scheme,
			}, nil
		}
	default:
		return nil, fmt.Errorf("unsupported scheme: %s", c1.Scheme)
	}
}

// Multiply performs homomorphic multiplication
func (e *Engine) Multiply(c1, c2 *Ciphertext) (*Ciphertext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if c1.Scheme != c2.Scheme {
		return nil, fmt.Errorf("ciphertexts must use the same scheme")
	}

	e.operations++

	switch c1.Scheme {
	case SchemeSHE, SchemeLFHE:
		pub := e.keyPair.PublicKey
		// c1 * c2 = (c1 * c2) mod n
		result := new(big.Int).Mul(c1.Value, c2.Value)
		result.Mod(result, pub.N)

		newLevel := c1.Level
		if c2.Level < newLevel {
			newLevel = c2.Level
		}
		if newLevel > 1 {
			newLevel -= 2 // Multiplication consumes more noise budget
		} else {
			newLevel = 0
		}

		if newLevel < 0 && c1.Scheme == SchemeLFHE {
			return nil, fmt.Errorf("noise budget exhausted")
		}

		return &Ciphertext{
			Value:  result,
			Level:  newLevel,
			Scheme: c1.Scheme,
		}, nil
	default:
		return nil, fmt.Errorf("multiplication not supported for scheme: %s", c1.Scheme)
	}
}

// ScalarMultiply performs scalar multiplication on encrypted data
func (e *Engine) ScalarMultiply(c *Ciphertext, scalar *big.Int) (*Ciphertext, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.operations++

	pub := e.keyPair.PublicKey

	if c.Scheme == SchemePHE {
		// For Paillier: c * k = c^k mod n^2
		nSquared := new(big.Int).Mul(pub.N, pub.N)
		result := new(big.Int).Exp(c.Value, scalar, nSquared)

		return &Ciphertext{
			Value:  result,
			Level:  c.Level,
			Scheme: c.Scheme,
		}, nil
	} else {
		// For BGV: c * k = (c * k) mod n
		result := new(big.Int).Mul(c.Value, scalar)
		result.Mod(result, pub.N)

		return &Ciphertext{
			Value:  result,
			Level:  c.Level,
			Scheme: c.Scheme,
		}, nil
	}
}

// GetMetrics returns homomorphic encryption metrics
func (e *Engine) GetMetrics() map[string]interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	hasKeyPair := e.keyPair != nil

	return map[string]interface{}{
		"scheme":          e.scheme,
		"security_level":  e.securityLevel,
		"key_size":        e.keySize,
		"key_pair_exists": hasKeyPair,
		"operations":      e.operations,
	}
}

// Helper functions

func lcm(a, b *big.Int) *big.Int {
	gcd := new(big.Int).GCD(nil, nil, a, b)
	product := new(big.Int).Mul(a, b)
	return new(big.Int).Div(product, gcd)
}

func lFunction(u, n *big.Int) *big.Int {
	// L(u) = (u - 1) / n
	result := new(big.Int).Sub(u, big.NewInt(1))
	return result.Div(result, n)
}

// EncryptBytes encrypts byte data
func (e *Engine) EncryptBytes(data []byte) (*Ciphertext, error) {
	// Convert bytes to big.Int
	plaintext := new(big.Int).SetBytes(data)
	return e.Encrypt(plaintext)
}

// DecryptBytes decrypts to byte data
func (e *Engine) DecryptBytes(ciphertext *Ciphertext) ([]byte, error) {
	plaintext, err := e.Decrypt(ciphertext)
	if err != nil {
		return nil, err
	}
	return plaintext.Bytes(), nil
}

// GetKeyID returns the key pair ID
func (e *Engine) GetKeyID() string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if e.keyPair == nil {
		return ""
	}

	hash := sha256.Sum256(e.keyPair.PublicKey.N.Bytes())
	return hex.EncodeToString(hash[:])
}
