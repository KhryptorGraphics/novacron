package auth

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// TokenRevocationService manages JWT token blacklisting
type TokenRevocationService interface {
	// RevokeToken adds a token JTI to the blacklist
	RevokeToken(jti string, expiresAt time.Time) error

	// RevokeTokenWithReason adds a token to blacklist with a reason
	RevokeTokenWithReason(jti string, expiresAt time.Time, reason string) error

	// IsRevoked checks if a token JTI is in the blacklist
	IsRevoked(jti string) (bool, error)

	// RevokeAllUserTokens revokes all tokens for a user (logout all devices)
	RevokeAllUserTokens(userID string, reason string) error

	// GetRevocationReason gets the reason a token was revoked
	GetRevocationReason(jti string) (string, error)

	// CleanupExpired removes expired entries (for non-Redis implementations)
	CleanupExpired() error
}

// RedisTokenRevocation implements TokenRevocationService using Redis
type RedisTokenRevocation struct {
	client *redis.Client
	ctx    context.Context
}

// NewRedisTokenRevocation creates a new Redis-backed token revocation service
func NewRedisTokenRevocation(client *redis.Client) *RedisTokenRevocation {
	return &RedisTokenRevocation{
		client: client,
		ctx:    context.Background(),
	}
}

// NewRedisTokenRevocationFromURL creates a token revocation service from Redis URL
func NewRedisTokenRevocationFromURL(redisURL string) (*RedisTokenRevocation, error) {
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	client := redis.NewClient(opt)

	// Test connection
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return NewRedisTokenRevocation(client), nil
}

// revokedKey returns the Redis key for a revoked token
func revokedKey(jti string) string {
	return fmt.Sprintf("revoked:%s", jti)
}

// userRevokedKey returns the Redis key for tracking user's revoked tokens
func userRevokedKey(userID string) string {
	return fmt.Sprintf("revoked:user:%s", userID)
}

// RevokeToken adds a token JTI to the blacklist
func (r *RedisTokenRevocation) RevokeToken(jti string, expiresAt time.Time) error {
	return r.RevokeTokenWithReason(jti, expiresAt, "token_revoked")
}

// RevokeTokenWithReason adds a token to blacklist with a reason
func (r *RedisTokenRevocation) RevokeTokenWithReason(jti string, expiresAt time.Time, reason string) error {
	// Calculate TTL - token only needs to be blacklisted until it would naturally expire
	ttl := time.Until(expiresAt)
	if ttl <= 0 {
		// Token already expired, no need to blacklist
		return nil
	}

	// Add a small buffer to account for clock skew
	ttl += 5 * time.Minute

	// Store the revocation with reason
	err := r.client.Set(r.ctx, revokedKey(jti), reason, ttl).Err()
	if err != nil {
		return fmt.Errorf("failed to revoke token: %w", err)
	}

	return nil
}

// IsRevoked checks if a token JTI is in the blacklist
func (r *RedisTokenRevocation) IsRevoked(jti string) (bool, error) {
	exists, err := r.client.Exists(r.ctx, revokedKey(jti)).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check token revocation: %w", err)
	}

	return exists > 0, nil
}

// RevokeAllUserTokens revokes all tokens for a user
// This works by storing a timestamp - any token issued before this time is considered revoked
func (r *RedisTokenRevocation) RevokeAllUserTokens(userID string, reason string) error {
	// Store the revocation timestamp with 7-day TTL (matches max refresh token lifetime)
	ttl := 7 * 24 * time.Hour

	data := fmt.Sprintf("%d|%s", time.Now().Unix(), reason)
	err := r.client.Set(r.ctx, userRevokedKey(userID), data, ttl).Err()
	if err != nil {
		return fmt.Errorf("failed to revoke all user tokens: %w", err)
	}

	return nil
}

// IsUserTokenRevoked checks if a token was issued before the user's revocation timestamp
func (r *RedisTokenRevocation) IsUserTokenRevoked(userID string, issuedAt time.Time) (bool, error) {
	data, err := r.client.Get(r.ctx, userRevokedKey(userID)).Result()
	if err == redis.Nil {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("failed to check user token revocation: %w", err)
	}

	// Parse timestamp from stored data
	var revokedAt int64
	_, err = fmt.Sscanf(data, "%d|", &revokedAt)
	if err != nil {
		// Old format or corrupted data, treat as not revoked
		return false, nil
	}

	// Token is revoked if it was issued before the revocation timestamp
	return issuedAt.Unix() <= revokedAt, nil
}

// GetRevocationReason gets the reason a token was revoked
func (r *RedisTokenRevocation) GetRevocationReason(jti string) (string, error) {
	reason, err := r.client.Get(r.ctx, revokedKey(jti)).Result()
	if err == redis.Nil {
		return "", fmt.Errorf("token not revoked: %s", jti)
	}
	if err != nil {
		return "", fmt.Errorf("failed to get revocation reason: %w", err)
	}

	return reason, nil
}

// CleanupExpired is a no-op for Redis as TTL handles cleanup automatically
func (r *RedisTokenRevocation) CleanupExpired() error {
	// Redis handles expiration automatically via TTL
	return nil
}

// Close closes the Redis connection
func (r *RedisTokenRevocation) Close() error {
	return r.client.Close()
}

// RevokeRefreshToken is a convenience method that uses the standard refresh token TTL
func (r *RedisTokenRevocation) RevokeRefreshToken(jti string) error {
	// Refresh tokens have 7-day TTL
	expiresAt := time.Now().Add(7 * 24 * time.Hour)
	return r.RevokeTokenWithReason(jti, expiresAt, "refresh_token_revoked")
}

// RevokeAccessToken is a convenience method that uses the standard access token TTL
func (r *RedisTokenRevocation) RevokeAccessToken(jti string) error {
	// Access tokens have 15-minute TTL
	expiresAt := time.Now().Add(15 * time.Minute)
	return r.RevokeTokenWithReason(jti, expiresAt, "access_token_revoked")
}

// RevocationInfo contains information about a revoked token
type RevocationInfo struct {
	JTI       string    `json:"jti"`
	Reason    string    `json:"reason"`
	RevokedAt time.Time `json:"revoked_at"`
	ExpiresAt time.Time `json:"expires_at"`
}

// Ensure RedisTokenRevocation implements TokenRevocationService
var _ TokenRevocationService = (*RedisTokenRevocation)(nil)

// InMemoryTokenRevocation is a simple in-memory implementation for testing
type InMemoryTokenRevocation struct {
	revoked map[string]RevocationInfo
}

// NewInMemoryTokenRevocation creates a new in-memory token revocation service
func NewInMemoryTokenRevocation() *InMemoryTokenRevocation {
	return &InMemoryTokenRevocation{
		revoked: make(map[string]RevocationInfo),
	}
}

// RevokeToken adds a token JTI to the blacklist
func (m *InMemoryTokenRevocation) RevokeToken(jti string, expiresAt time.Time) error {
	return m.RevokeTokenWithReason(jti, expiresAt, "token_revoked")
}

// RevokeTokenWithReason adds a token to blacklist with a reason
func (m *InMemoryTokenRevocation) RevokeTokenWithReason(jti string, expiresAt time.Time, reason string) error {
	m.revoked[jti] = RevocationInfo{
		JTI:       jti,
		Reason:    reason,
		RevokedAt: time.Now(),
		ExpiresAt: expiresAt,
	}
	return nil
}

// IsRevoked checks if a token JTI is in the blacklist
func (m *InMemoryTokenRevocation) IsRevoked(jti string) (bool, error) {
	info, exists := m.revoked[jti]
	if !exists {
		return false, nil
	}

	// Check if expired
	if time.Now().After(info.ExpiresAt) {
		delete(m.revoked, jti)
		return false, nil
	}

	return true, nil
}

// RevokeAllUserTokens is not fully supported in memory implementation
func (m *InMemoryTokenRevocation) RevokeAllUserTokens(userID string, reason string) error {
	// In-memory implementation doesn't track user -> token mappings
	// This would require additional data structure
	return nil
}

// GetRevocationReason gets the reason a token was revoked
func (m *InMemoryTokenRevocation) GetRevocationReason(jti string) (string, error) {
	info, exists := m.revoked[jti]
	if !exists {
		return "", fmt.Errorf("token not revoked: %s", jti)
	}
	return info.Reason, nil
}

// CleanupExpired removes expired entries
func (m *InMemoryTokenRevocation) CleanupExpired() error {
	now := time.Now()
	for jti, info := range m.revoked {
		if now.After(info.ExpiresAt) {
			delete(m.revoked, jti)
		}
	}
	return nil
}

// Ensure InMemoryTokenRevocation implements TokenRevocationService
var _ TokenRevocationService = (*InMemoryTokenRevocation)(nil)
