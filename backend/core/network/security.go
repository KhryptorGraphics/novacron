package network

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
// 	"encoding/binary"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"math/big"
	"sync"
	"time"
)

// Security constants
const (
	// EncryptionKeySize is the size of the AES encryption key
	EncryptionKeySize = 32 // 256 bits
	
	// NonceSize is the size of the AES-GCM nonce
	NonceSize = 12
	
	// SignatureSize is the size of the ECDSA signature
	SignatureSize = 64
)

// SecurityConfig contains configuration for the security layer
type SecurityConfig struct {
	// EnableEncryption enables message encryption
	EnableEncryption bool
	
	// EnableSignature enables message signing
	EnableSignature bool
	
	// TLSCertFile is the path to the TLS certificate file
	TLSCertFile string
	
	// TLSKeyFile is the path to the TLS key file
	TLSKeyFile string
	
	// AutoGenerateTLSCert automatically generates a TLS certificate if none is provided
	AutoGenerateTLSCert bool
	
	// TLSCertValidDuration is the duration for which the auto-generated TLS certificate is valid
	TLSCertValidDuration time.Duration
}

// DefaultSecurityConfig returns a default security configuration
func DefaultSecurityConfig() SecurityConfig {
	return SecurityConfig{
		EnableEncryption:     true,
		EnableSignature:      true,
		AutoGenerateTLSCert:  true,
		TLSCertValidDuration: 365 * 24 * time.Hour, // 1 year
	}
}

// SecurityContext contains the security context for a peer
type SecurityContext struct {
	// PrivateKey is the private key used for signing
	PrivateKey *ecdsa.PrivateKey
	
	// PublicKey is the public key used for verification
	PublicKey *ecdsa.PublicKey
	
	// PeerPublicKeys maps peer IDs to their public keys
	PeerPublicKeys map[string]*ecdsa.PublicKey
	
	// EncryptionKeys maps peer IDs to their encryption keys
	EncryptionKeys map[string][]byte
	
	// TLSConfig is the TLS configuration
	TLSConfig *tls.Config
	
	// mutex protects the maps
	mutex sync.RWMutex
}

// NewSecurityContext creates a new security context
func NewSecurityContext() (*SecurityContext, error) {
	// Generate a new ECDSA key pair
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("failed to generate ECDSA key pair: %w", err)
	}
	
	return &SecurityContext{
		PrivateKey:     privateKey,
		PublicKey:      &privateKey.PublicKey,
		PeerPublicKeys: make(map[string]*ecdsa.PublicKey),
		EncryptionKeys: make(map[string][]byte),
	}, nil
}

// SetupTLS sets up TLS configuration from a certificate and key file
func (s *SecurityContext) SetupTLS(config SecurityConfig) error {
	var cert tls.Certificate
	var err error
	
	if config.TLSCertFile != "" && config.TLSKeyFile != "" {
		// Load certificate and key from files
		cert, err = tls.LoadX509KeyPair(config.TLSCertFile, config.TLSKeyFile)
		if err != nil {
			return fmt.Errorf("failed to load TLS certificate and key: %w", err)
		}
	} else if config.AutoGenerateTLSCert {
		// Generate a self-signed certificate
		cert, err = s.generateSelfSignedCert(config.TLSCertValidDuration)
		if err != nil {
			return fmt.Errorf("failed to generate self-signed certificate: %w", err)
		}
	} else {
		return errors.New("TLS certificate and key files or auto-generation must be specified")
	}
	
	// Create TLS configuration
	s.TLSConfig = &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
		},
	}
	
	return nil
}

// generateSelfSignedCert generates a self-signed certificate
func (s *SecurityContext) generateSelfSignedCert(validFor time.Duration) (tls.Certificate, error) {
	// Set up certificate template
	notBefore := time.Now()
	notAfter := notBefore.Add(validFor)
	
	serialNumber, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("failed to generate serial number: %w", err)
	}
	
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			Organization: []string{"NovaCron"},
			CommonName:   "NovaCron Node",
		},
		NotBefore:             notBefore,
		NotAfter:              notAfter,
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	
	// Generate the certificate
	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &s.PrivateKey.PublicKey, s.PrivateKey)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("failed to create certificate: %w", err)
	}
	
	// Encode private key and certificate to PEM
	var keyBuf bytes.Buffer
	privateKeyBytes, err := x509.MarshalPKCS8PrivateKey(s.PrivateKey)
	if err != nil {
		return tls.Certificate{}, fmt.Errorf("failed to marshal private key: %w", err)
	}
	
	if err := pem.Encode(&keyBuf, &pem.Block{Type: "PRIVATE KEY", Bytes: privateKeyBytes}); err != nil {
		return tls.Certificate{}, fmt.Errorf("failed to encode private key: %w", err)
	}
	
	var certBuf bytes.Buffer
	if err := pem.Encode(&certBuf, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes}); err != nil {
		return tls.Certificate{}, fmt.Errorf("failed to encode certificate: %w", err)
	}
	
	// Create certificate from PEM data
	return tls.X509KeyPair(certBuf.Bytes(), keyBuf.Bytes())
}

// AddPeerPublicKey adds a peer's public key
func (s *SecurityContext) AddPeerPublicKey(peerID string, publicKey *ecdsa.PublicKey) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	s.PeerPublicKeys[peerID] = publicKey
}

// GetPeerPublicKey gets a peer's public key
func (s *SecurityContext) GetPeerPublicKey(peerID string) (*ecdsa.PublicKey, bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	key, ok := s.PeerPublicKeys[peerID]
	return key, ok
}

// SetEncryptionKey sets the encryption key for a peer
func (s *SecurityContext) SetEncryptionKey(peerID string, key []byte) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	
	s.EncryptionKeys[peerID] = key
}

// GetEncryptionKey gets the encryption key for a peer
func (s *SecurityContext) GetEncryptionKey(peerID string) ([]byte, bool) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	
	key, ok := s.EncryptionKeys[peerID]
	return key, ok
}

// GenerateEncryptionKey generates a new encryption key for a peer
func (s *SecurityContext) GenerateEncryptionKey(peerID string) ([]byte, error) {
	// Generate a random key
	key := make([]byte, EncryptionKeySize)
	if _, err := io.ReadFull(rand.Reader, key); err != nil {
		return nil, fmt.Errorf("failed to generate encryption key: %w", err)
	}
	
	// Store the key
	s.SetEncryptionKey(peerID, key)
	
	return key, nil
}

// EncryptMessage encrypts a message for a peer
func (s *SecurityContext) EncryptMessage(msg *Message, peerID string) (*Message, error) {
	// Get the encryption key
	key, ok := s.GetEncryptionKey(peerID)
	if !ok {
		return nil, fmt.Errorf("no encryption key for peer %s", peerID)
	}
	
	// Serialize the message
	data := msg.Serialize()
	
	// Create a new AES-GCM cipher
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}
	
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM cipher: %w", err)
	}
	
	// Generate a nonce
	nonce := make([]byte, NonceSize)
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, fmt.Errorf("failed to generate nonce: %w", err)
	}
	
	// Encrypt the message
	ciphertext := aesGCM.Seal(nil, nonce, data, nil)
	
	// Create a new message with the encrypted data
	encryptedData := make([]byte, NonceSize+len(ciphertext))
	copy(encryptedData[:NonceSize], nonce)
	copy(encryptedData[NonceSize:], ciphertext)
	
	// Create a new message with the encrypted payload
	encryptedMsg := NewMessage(msg.Header.Type, encryptedData, msg.Header.Flags|FlagEncrypted, msg.Header.SequenceID)
	
	return encryptedMsg, nil
}

// DecryptMessage decrypts a message from a peer
func (s *SecurityContext) DecryptMessage(msg *Message, peerID string) (*Message, error) {
	// Check if the message is encrypted
	if !msg.Header.IsFlag(FlagEncrypted) {
		return msg, nil
	}
	
	// Get the encryption key
	key, ok := s.GetEncryptionKey(peerID)
	if !ok {
		return nil, fmt.Errorf("no encryption key for peer %s", peerID)
	}
	
	// Extract the nonce and ciphertext
	if len(msg.Payload) < NonceSize {
		return nil, errors.New("payload too short for encrypted message")
	}
	
	nonce := msg.Payload[:NonceSize]
	ciphertext := msg.Payload[NonceSize:]
	
	// Create a new AES-GCM cipher
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %w", err)
	}
	
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create GCM cipher: %w", err)
	}
	
	// Decrypt the message
	plaintext, err := aesGCM.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt message: %w", err)
	}
	
	// Deserialize the original message
	decryptedMsg, err := Deserialize(plaintext)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize decrypted message: %w", err)
	}
	
	return decryptedMsg, nil
}

// SignMessage signs a message
func (s *SecurityContext) SignMessage(msg *Message) (*Message, error) {
	// Serialize the message without the signature
	data := msg.Serialize()
	
	// Create a SHA-256 hash of the message
	hash := sha256.Sum256(data)
	
	// Sign the hash
	r, ss, err := ecdsa.Sign(rand.Reader, s.PrivateKey, hash[:])
	if err != nil {
		return nil, fmt.Errorf("failed to sign message: %w", err)
	}
	
	// Convert the signature to a byte array
	signature := make([]byte, SignatureSize)
	
	// Convert r and s to big-endian format
	rBytes := r.Bytes()
	sBytes := ss.Bytes()
	
	// Pad r and s to 32 bytes
	copy(signature[32-len(rBytes):32], rBytes)
	copy(signature[64-len(sBytes):64], sBytes)
	
	// Create a new message with the signature appended to the payload
	signedPayload := append(msg.Payload, signature...)
	
	// Create a new message with the signed payload
	signedMsg := NewMessage(msg.Header.Type, signedPayload, msg.Header.Flags, msg.Header.SequenceID)
	
	return signedMsg, nil
}

// VerifyMessage verifies a signed message
func (s *SecurityContext) VerifyMessage(msg *Message, peerID string) (*Message, bool, error) {
	// Get the peer's public key
	publicKey, ok := s.GetPeerPublicKey(peerID)
	if !ok {
		return nil, false, fmt.Errorf("no public key for peer %s", peerID)
	}
	
	// Extract the signature and original payload
	if len(msg.Payload) < SignatureSize {
		return nil, false, errors.New("payload too short for signed message")
	}
	
	originalPayloadSize := len(msg.Payload) - SignatureSize
	originalPayload := msg.Payload[:originalPayloadSize]
	signature := msg.Payload[originalPayloadSize:]
	
	// Recreate the original message
	originalMsg := NewMessage(msg.Header.Type, originalPayload, msg.Header.Flags, msg.Header.SequenceID)
	
	// Serialize the original message
	data := originalMsg.Serialize()
	
	// Create a SHA-256 hash of the message
	hash := sha256.Sum256(data)
	
	// Extract r and s from the signature
	r := new(big.Int).SetBytes(signature[:32])
	s := new(big.Int).SetBytes(signature[32:64])
	
	// Verify the signature
	valid := ecdsa.Verify(publicKey, hash[:], r, s)
	
	return originalMsg, valid, nil
}

// ExportPublicKey exports the public key as a base64-encoded string
func (s *SecurityContext) ExportPublicKey() (string, error) {
	// Encode public key bytes
	pubKeyBytes := elliptic.Marshal(elliptic.P256(), s.PublicKey.X, s.PublicKey.Y)
	
	// Base64 encode
	encoded := base64.StdEncoding.EncodeToString(pubKeyBytes)
	
	return encoded, nil
}

// ImportPublicKey imports a public key from a base64-encoded string
func (s *SecurityContext) ImportPublicKey(encoded string) (*ecdsa.PublicKey, error) {
	// Base64 decode
	pubKeyBytes, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("failed to decode public key: %w", err)
	}
	
	// Parse public key
	x, y := elliptic.Unmarshal(elliptic.P256(), pubKeyBytes)
	if x == nil || y == nil {
		return nil, errors.New("failed to unmarshal public key")
	}
	
	return &ecdsa.PublicKey{
		Curve: elliptic.P256(),
		X:     x,
		Y:     y,
	}, nil
}

// DeriveSharedEncryptionKey derives a shared encryption key using ECDH
func (s *SecurityContext) DeriveSharedEncryptionKey(peerID string, peerPublicKey *ecdsa.PublicKey) ([]byte, error) {
	// Compute shared secret using ECDH
	x, _ := s.PrivateKey.ScalarMult(peerPublicKey.X, peerPublicKey.Y, s.PrivateKey.D.Bytes())
	
	// Convert shared secret to bytes
	sharedSecret := x.Bytes()
	
	// Hash the shared secret to get the encryption key
	h := sha256.New()
	h.Write(sharedSecret)
	key := h.Sum(nil)
	
	// Store the key
	s.SetEncryptionKey(peerID, key)
	
	return key, nil
}

// EncodePublicKeyMessage encodes a public key message
func (s *SecurityContext) EncodePublicKeyMessage() ([]byte, error) {
	// Export the public key
	pubKeyStr, err := s.ExportPublicKey()
	if err != nil {
		return nil, err
	}
	
	// Encode as a message payload
	return []byte(pubKeyStr), nil
}

// DecodePublicKeyMessage decodes a public key message
func (s *SecurityContext) DecodePublicKeyMessage(payload []byte) (*ecdsa.PublicKey, error) {
	// Convert payload to string
	pubKeyStr := string(payload)
	
	// Import the public key
	return s.ImportPublicKey(pubKeyStr)
}

// ProtectMessage applies encryption and/or signing to a message
func (s *SecurityContext) ProtectMessage(msg *Message, peerID string, encrypt bool, sign bool) (*Message, error) {
	result := msg
	var err error
	
	// Sign the message if requested
	if sign {
		result, err = s.SignMessage(result)
		if err != nil {
			return nil, fmt.Errorf("failed to sign message: %w", err)
		}
	}
	
	// Encrypt the message if requested
	if encrypt {
		result, err = s.EncryptMessage(result, peerID)
		if err != nil {
			return nil, fmt.Errorf("failed to encrypt message: %w", err)
		}
	}
	
	return result, nil
}

// UnprotectMessage applies decryption and/or signature verification to a message
func (s *SecurityContext) UnprotectMessage(msg *Message, peerID string) (*Message, bool, error) {
	result := msg
	var err error
	valid := true
	
	// Decrypt the message if it's encrypted
	if msg.Header.IsFlag(FlagEncrypted) {
		result, err = s.DecryptMessage(msg, peerID)
		if err != nil {
			return nil, false, fmt.Errorf("failed to decrypt message: %w", err)
		}
	}
	
	// Verify the signature if it's signed
	if len(result.Payload) >= SignatureSize {
		var signatureValid bool
		result, signatureValid, err = s.VerifyMessage(result, peerID)
		if err != nil {
			// Not a signed message or verification error
			// Just continue with the original message
			valid = false
		} else {
			valid = signatureValid
		}
	}
	
	return result, valid, nil
}
