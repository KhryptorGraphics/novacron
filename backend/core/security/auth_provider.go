package security

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v4"
)

// AuthProvider defines the interface for authentication providers
type AuthProvider interface {
	// Authenticate authenticates a user and returns a token
	Authenticate(username, password string) (string, error)

	// ValidateToken validates a token and returns the claims
	ValidateToken(token string) (*Claims, error)

	// RefreshToken refreshes a token and returns a new token
	RefreshToken(token string) (string, error)

	// RevokeToken revokes a token
	RevokeToken(token string) error

	// GetName returns the provider name
	GetName() string
}

// BaseAuthProvider is a base implementation of AuthProvider
type BaseAuthProvider struct {
	// Name is the name of the provider
	Name string

	// JWTSecret is the secret used to sign JWT tokens
	JWTSecret []byte

	// PrivateKey is the RSA private key used to sign JWT tokens
	PrivateKey *rsa.PrivateKey

	// PublicKey is the RSA public key used to verify JWT tokens
	PublicKey *rsa.PublicKey

	// Expiration is the token expiration time
	Expiration time.Duration

	// RefreshExpiration is the refresh token expiration time
	RefreshExpiration time.Duration

	// RevokedTokens is a map of revoked tokens
	RevokedTokens map[string]time.Time

	// mutex protects the revoked tokens map
	mutex sync.RWMutex
}

// Claims represents the JWT claims
type Claims struct {
	UserID       string   `json:"user_id"`
	Username     string   `json:"username"`
	Email        string   `json:"email"`
	Roles        []string `json:"roles"`
	Permissions  []string `json:"permissions"`
	TenantID     string   `json:"tenant_id"`
	SessionID    string   `json:"session_id"`
	TokenType    string   `json:"token_type"`
	ClientIP     string   `json:"client_ip,omitempty"`
	UserAgent    string   `json:"user_agent,omitempty"`
	RefreshToken string   `json:"refresh_token,omitempty"`
	jwt.RegisteredClaims
}

// NewBaseAuthProvider creates a new base auth provider
func NewBaseAuthProvider(name string, expiration, refreshExpiration time.Duration) (*BaseAuthProvider, error) {
	// Generate a random JWT secret if none is provided
	jwtSecret := make([]byte, 32)
	_, err := rand.Read(jwtSecret)
	if err != nil {
		return nil, fmt.Errorf("failed to generate JWT secret: %w", err)
	}

	// Generate RSA key pair
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, fmt.Errorf("failed to generate RSA key pair: %w", err)
	}

	return &BaseAuthProvider{
		Name:              name,
		JWTSecret:         jwtSecret,
		PrivateKey:        privateKey,
		PublicKey:         &privateKey.PublicKey,
		Expiration:        expiration,
		RefreshExpiration: refreshExpiration,
		RevokedTokens:     make(map[string]time.Time),
	}, nil
}

// Authenticate authenticates a user and returns a token
func (p *BaseAuthProvider) Authenticate(username, password string) (string, error) {
	// This is a base implementation that should be overridden by concrete providers
	return "", errors.New("not implemented in base provider")
}

// ValidateToken validates a token and returns the claims
func (p *BaseAuthProvider) ValidateToken(tokenString string) (*Claims, error) {
	// Check if token is revoked
	p.mutex.RLock()
	if _, revoked := p.RevokedTokens[tokenString]; revoked {
		p.mutex.RUnlock()
		return nil, errors.New("token is revoked")
	}
	p.mutex.RUnlock()

	// Parse token
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		// Check signing method
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return p.PublicKey, nil
	})

	if err != nil {
		return nil, fmt.Errorf("invalid token: %w", err)
	}

	// Extract claims
	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		// Validate token type
		if claims.TokenType != "access" {
			return nil, errors.New("invalid token type")
		}
		return claims, nil
	}

	return nil, errors.New("invalid token claims")
}

// RefreshToken refreshes a token and returns a new token
func (p *BaseAuthProvider) RefreshToken(tokenString string) (string, error) {
	// Check if token is revoked
	p.mutex.RLock()
	if _, revoked := p.RevokedTokens[tokenString]; revoked {
		p.mutex.RUnlock()
		return "", errors.New("token is revoked")
	}
	p.mutex.RUnlock()

	// Parse token
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		// Check signing method
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return p.PublicKey, nil
	})

	if err != nil {
		return "", fmt.Errorf("invalid token: %w", err)
	}

	// Extract claims
	claims, ok := token.Claims.(*Claims)
	if !ok || !token.Valid {
		return "", errors.New("invalid token claims")
	}

	// Validate token type
	if claims.TokenType != "refresh" {
		return "", errors.New("not a refresh token")
	}

	// Create new access token
	newClaims := &Claims{
		UserID:      claims.UserID,
		Username:    claims.Username,
		Email:       claims.Email,
		Roles:       claims.Roles,
		Permissions: claims.Permissions,
		TenantID:    claims.TenantID,
		SessionID:   claims.SessionID,
		TokenType:   "access",
		ClientIP:    claims.ClientIP,
		UserAgent:   claims.UserAgent,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(p.Expiration)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    p.Name,
		},
	}

	// Create token
	newToken := jwt.NewWithClaims(jwt.SigningMethodRS256, newClaims)
	signedToken, err := newToken.SignedString(p.PrivateKey)
	if err != nil {
		return "", fmt.Errorf("failed to sign token: %w", err)
	}

	// Revoke old refresh token
	p.RevokeToken(tokenString)

	return signedToken, nil
}

// RevokeToken revokes a token
func (p *BaseAuthProvider) RevokeToken(tokenString string) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// Add token to revoked tokens
	p.RevokedTokens[tokenString] = time.Now()

	// Clean up old revoked tokens (optional, could be done in a background task)
	for token, revokedTime := range p.RevokedTokens {
		if time.Since(revokedTime) > p.RefreshExpiration {
			delete(p.RevokedTokens, token)
		}
	}

	return nil
}

// GetName returns the provider name
func (p *BaseAuthProvider) GetName() string {
	return p.Name
}

// GenerateKeyPair generates a new RSA key pair
func GenerateKeyPair(bits int) (*rsa.PrivateKey, *rsa.PublicKey, error) {
	privateKey, err := rsa.GenerateKey(rand.Reader, bits)
	if err != nil {
		return nil, nil, err
	}

	return privateKey, &privateKey.PublicKey, nil
}

// EncodePrivateKeyToPEM encodes a private key to PEM format
func EncodePrivateKeyToPEM(privateKey *rsa.PrivateKey) string {
	privateKeyBytes := x509.MarshalPKCS1PrivateKey(privateKey)
	privateKeyPEM := pem.EncodeToMemory(
		&pem.Block{
			Type:  "RSA PRIVATE KEY",
			Bytes: privateKeyBytes,
		},
	)
	return string(privateKeyPEM)
}

// EncodePublicKeyToPEM encodes a public key to PEM format
func EncodePublicKeyToPEM(publicKey *rsa.PublicKey) (string, error) {
	publicKeyBytes, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return "", err
	}
	publicKeyPEM := pem.EncodeToMemory(
		&pem.Block{
			Type:  "RSA PUBLIC KEY",
			Bytes: publicKeyBytes,
		},
	)
	return string(publicKeyPEM), nil
}

// DecodePrivateKeyFromPEM decodes a private key from PEM format
func DecodePrivateKeyFromPEM(privateKeyPEM string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(privateKeyPEM))
	if block == nil || block.Type != "RSA PRIVATE KEY" {
		return nil, errors.New("failed to decode PEM block containing private key")
	}

	privateKey, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	return privateKey, nil
}

// DecodePublicKeyFromPEM decodes a public key from PEM format
func DecodePublicKeyFromPEM(publicKeyPEM string) (*rsa.PublicKey, error) {
	block, _ := pem.Decode([]byte(publicKeyPEM))
	if block == nil || block.Type != "RSA PUBLIC KEY" {
		return nil, errors.New("failed to decode PEM block containing public key")
	}

	publicKeyInterface, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}

	publicKey, ok := publicKeyInterface.(*rsa.PublicKey)
	if !ok {
		return nil, errors.New("not an RSA public key")
	}

	return publicKey, nil
}

// EncodeToBase64 encodes a byte slice to base64
func EncodeToBase64(data []byte) string {
	return base64.StdEncoding.EncodeToString(data)
}

// DecodeFromBase64 decodes a base64 string to a byte slice
func DecodeFromBase64(encodedData string) ([]byte, error) {
	return base64.StdEncoding.DecodeString(encodedData)
}

// ValidatePassword validates a password against a hashed password
func ValidatePassword(password, hashedPassword string) bool {
	// This is a simple implementation that should be replaced with a proper password hashing function
	parts := strings.Split(hashedPassword, ":")
	if len(parts) != 2 {
		return false
	}

	salt, _ := parts[0], parts[1]
	expected := HashPassword(password, salt)
	return expected == hashedPassword
}

// HashPassword hashes a password with a salt
func HashPassword(password, salt string) string {
	// This is a simple implementation that should be replaced with a proper password hashing function
	return salt + ":" + base64.StdEncoding.EncodeToString([]byte(password+salt))
}

// GenerateSalt generates a random salt
func GenerateSalt(length int) (string, error) {
	salt := make([]byte, length)
	_, err := rand.Read(salt)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(salt), nil
}
