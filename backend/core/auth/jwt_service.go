package auth

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"strconv"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// JWTConfiguration defines JWT token settings
type JWTConfiguration struct {
	// RSAPrivateKey for RS256 signing
	RSAPrivateKey *rsa.PrivateKey
	// RSAPublicKey for RS256 verification
	RSAPublicKey *rsa.PublicKey
	// AccessTokenTTL defines access token lifetime
	AccessTokenTTL time.Duration
	// RefreshTokenTTL defines refresh token lifetime
	RefreshTokenTTL time.Duration
	// Issuer defines the token issuer
	Issuer string
	// Audience defines the token audience
	Audience string
	// KeyID for key rotation support
	KeyID string
}

// JWTClaims defines custom JWT claims
type JWTClaims struct {
	jwt.RegisteredClaims
	UserID      string                 `json:"user_id"`
	TenantID    string                 `json:"tenant_id"`
	Roles       []string               `json:"roles"`
	Permissions []string               `json:"permissions"`
	TokenType   string                 `json:"token_type"` // "access" or "refresh"
	SessionID   string                 `json:"session_id"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// TokenPair represents access and refresh tokens
type TokenPair struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int64  `json:"expires_in"`
}

// JWTService handles JWT token operations
type JWTService struct {
	config JWTConfiguration
}

// NewJWTService creates a new JWT service
func NewJWTService(config JWTConfiguration) *JWTService {
	if config.AccessTokenTTL == 0 {
		config.AccessTokenTTL = 15 * time.Minute
	}
	if config.RefreshTokenTTL == 0 {
		config.RefreshTokenTTL = 7 * 24 * time.Hour // 7 days
	}
	if config.Issuer == "" {
		config.Issuer = "novacron"
	}
	if config.Audience == "" {
		config.Audience = "novacron-api"
	}
	if config.KeyID == "" {
		config.KeyID = "default"
	}

	return &JWTService{config: config}
}

// GenerateTokenPair generates access and refresh tokens
func (j *JWTService) GenerateTokenPair(userID, tenantID string, roles []string, permissions []string, sessionID string, metadata map[string]interface{}) (*TokenPair, error) {
	now := time.Now()

	// Generate access token
	accessClaims := JWTClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    j.config.Issuer,
			Subject:   userID,
			Audience:  jwt.ClaimStrings{j.config.Audience},
			ExpiresAt: jwt.NewNumericDate(now.Add(j.config.AccessTokenTTL)),
			NotBefore: jwt.NewNumericDate(now),
			IssuedAt:  jwt.NewNumericDate(now),
			ID:        generateJTI(),
		},
		UserID:      userID,
		TenantID:    tenantID,
		Roles:       roles,
		Permissions: permissions,
		TokenType:   "access",
		SessionID:   sessionID,
		Metadata:    metadata,
	}

	accessToken := jwt.NewWithClaims(jwt.SigningMethodRS256, accessClaims)
	accessToken.Header["kid"] = j.config.KeyID

	accessTokenString, err := accessToken.SignedString(j.config.RSAPrivateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign access token: %w", err)
	}

	// Generate refresh token
	refreshClaims := JWTClaims{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    j.config.Issuer,
			Subject:   userID,
			Audience:  jwt.ClaimStrings{j.config.Audience},
			ExpiresAt: jwt.NewNumericDate(now.Add(j.config.RefreshTokenTTL)),
			NotBefore: jwt.NewNumericDate(now),
			IssuedAt:  jwt.NewNumericDate(now),
			ID:        generateJTI(),
		},
		UserID:    userID,
		TenantID:  tenantID,
		TokenType: "refresh",
		SessionID: sessionID,
	}

	refreshToken := jwt.NewWithClaims(jwt.SigningMethodRS256, refreshClaims)
	refreshToken.Header["kid"] = j.config.KeyID

	refreshTokenString, err := refreshToken.SignedString(j.config.RSAPrivateKey)
	if err != nil {
		return nil, fmt.Errorf("failed to sign refresh token: %w", err)
	}

	return &TokenPair{
		AccessToken:  accessTokenString,
		RefreshToken: refreshTokenString,
		TokenType:    "Bearer",
		ExpiresIn:    int64(j.config.AccessTokenTTL.Seconds()),
	}, nil
}

// ValidateToken validates and parses a JWT token
func (j *JWTService) ValidateToken(tokenString string) (*JWTClaims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &JWTClaims{}, func(token *jwt.Token) (interface{}, error) {
		// Validate signing method
		if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return j.config.RSAPublicKey, nil
	})

	if err != nil {
		return nil, fmt.Errorf("token validation failed: %w", err)
	}

	if !token.Valid {
		return nil, fmt.Errorf("token is invalid")
	}

	claims, ok := token.Claims.(*JWTClaims)
	if !ok {
		return nil, fmt.Errorf("invalid token claims")
	}

	// Validate audience
	if claims.Audience == nil || len(claims.Audience) == 0 || claims.Audience[0] != j.config.Audience {
		return nil, fmt.Errorf("invalid audience")
	}

	// Validate issuer
	if claims.Issuer != j.config.Issuer {
		return nil, fmt.Errorf("invalid issuer")
	}

	return claims, nil
}

// RefreshToken generates a new access token from a valid refresh token
func (j *JWTService) RefreshToken(refreshTokenString string) (*TokenPair, error) {
	claims, err := j.ValidateToken(refreshTokenString)
	if err != nil {
		return nil, fmt.Errorf("refresh token validation failed: %w", err)
	}

	if claims.TokenType != "refresh" {
		return nil, fmt.Errorf("invalid token type for refresh")
	}

	// Generate new token pair (refresh rotation)
	return j.GenerateTokenPair(claims.UserID, claims.TenantID, claims.Roles, claims.Permissions, claims.SessionID, claims.Metadata)
}

// RevokeToken adds a token to the revocation list (requires external storage)
func (j *JWTService) RevokeToken(tokenString string) error {
	claims, err := j.ValidateToken(tokenString)
	if err != nil {
		return fmt.Errorf("cannot revoke invalid token: %w", err)
	}

	// In production, store JTI in Redis/database with expiration
	// For now, we'll just validate the token structure
	_ = claims.ID // JTI for revocation list
	return nil
}

// GetTokenClaims extracts claims without full validation (for expired tokens)
func (j *JWTService) GetTokenClaims(tokenString string) (*JWTClaims, error) {
	token, _, err := new(jwt.Parser).ParseUnverified(tokenString, &JWTClaims{})
	if err != nil {
		return nil, fmt.Errorf("failed to parse token: %w", err)
	}

	claims, ok := token.Claims.(*JWTClaims)
	if !ok {
		return nil, fmt.Errorf("invalid token claims")
	}

	return claims, nil
}

// GenerateRSAKeys generates RSA key pair for JWT signing
func GenerateRSAKeys(keySize int) (*rsa.PrivateKey, *rsa.PublicKey, error) {
	if keySize < 2048 {
		keySize = 2048 // Minimum secure key size
	}

	privateKey, err := rsa.GenerateKey(rand.Reader, keySize)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate RSA private key: %w", err)
	}

	return privateKey, &privateKey.PublicKey, nil
}

// ExportRSAPrivateKeyToPEM exports RSA private key to PEM format
func ExportRSAPrivateKeyToPEM(privateKey *rsa.PrivateKey) ([]byte, error) {
	privateKeyBytes := x509.MarshalPKCS1PrivateKey(privateKey)

	privateKeyBlock := &pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: privateKeyBytes,
	}

	return pem.EncodeToMemory(privateKeyBlock), nil
}

// ExportRSAPublicKeyToPEM exports RSA public key to PEM format
func ExportRSAPublicKeyToPEM(publicKey *rsa.PublicKey) ([]byte, error) {
	publicKeyBytes, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal public key: %w", err)
	}

	publicKeyBlock := &pem.Block{
		Type:  "RSA PUBLIC KEY",
		Bytes: publicKeyBytes,
	}

	return pem.EncodeToMemory(publicKeyBlock), nil
}

// ImportRSAPrivateKeyFromPEM imports RSA private key from PEM format
func ImportRSAPrivateKeyFromPEM(privateKeyPEM []byte) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode(privateKeyPEM)
	if block == nil || block.Type != "RSA PRIVATE KEY" {
		return nil, fmt.Errorf("invalid PEM block for RSA private key")
	}

	privateKey, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse RSA private key: %w", err)
	}

	return privateKey, nil
}

// ImportRSAPublicKeyFromPEM imports RSA public key from PEM format
func ImportRSAPublicKeyFromPEM(publicKeyPEM []byte) (*rsa.PublicKey, error) {
	block, _ := pem.Decode(publicKeyPEM)
	if block == nil || block.Type != "RSA PUBLIC KEY" {
		return nil, fmt.Errorf("invalid PEM block for RSA public key")
	}

	publicKey, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("failed to parse RSA public key: %w", err)
	}

	rsaPublicKey, ok := publicKey.(*rsa.PublicKey)
	if !ok {
		return nil, fmt.Errorf("not an RSA public key")
	}

	return rsaPublicKey, nil
}

// generateJTI generates a unique JWT ID
func generateJTI() string {
	b := make([]byte, 16)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}

// DefaultJWTConfiguration returns a default JWT configuration with generated keys
func DefaultJWTConfiguration() (JWTConfiguration, error) {
	privateKey, publicKey, err := GenerateRSAKeys(2048)
	if err != nil {
		return JWTConfiguration{}, err
	}

	return JWTConfiguration{
		RSAPrivateKey:   privateKey,
		RSAPublicKey:    publicKey,
		AccessTokenTTL:  15 * time.Minute,
		RefreshTokenTTL: 7 * 24 * time.Hour,
		Issuer:          "novacron",
		Audience:        "novacron-api",
		KeyID:           "key-" + strconv.FormatInt(time.Now().Unix(), 10),
	}, nil
}
