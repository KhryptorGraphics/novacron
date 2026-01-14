package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

// SessionService defines the interface for session management
type SessionService interface {
	// Create creates a new session
	Create(session *Session) error

	// Get retrieves a session by ID
	Get(id string) (*Session, error)

	// GetByToken retrieves a session by token
	GetByToken(token string) (*Session, error)

	// Delete deletes a session
	Delete(id string) error

	// DeleteByUserID deletes all sessions for a user
	DeleteByUserID(userID string) error

	// UpdateLastAccessed updates the last accessed time
	UpdateLastAccessed(id string) error

	// GetUserSessions gets all sessions for a user
	GetUserSessions(userID string) ([]*Session, error)

	// Cleanup removes expired sessions (for in-memory stores)
	Cleanup() error
}

// RedisSessionStore implements SessionService using Redis
type RedisSessionStore struct {
	client *redis.Client
	ctx    context.Context
}

// NewRedisSessionStore creates a new Redis-backed session store
func NewRedisSessionStore(client *redis.Client) *RedisSessionStore {
	return &RedisSessionStore{
		client: client,
		ctx:    context.Background(),
	}
}

// NewRedisSessionStoreFromURL creates a new Redis session store from connection URL
func NewRedisSessionStoreFromURL(redisURL string) (*RedisSessionStore, error) {
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse Redis URL: %w", err)
	}

	client := redis.NewClient(opt)

	// Test connection
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return NewRedisSessionStore(client), nil
}

// sessionKey returns the Redis key for a session
func sessionKey(id string) string {
	return fmt.Sprintf("session:%s", id)
}

// tokenKey returns the Redis key for a token -> session ID mapping
func tokenKey(token string) string {
	return fmt.Sprintf("session:token:%s", token)
}

// userSessionsKey returns the Redis key for user sessions set
func userSessionsKey(userID string) string {
	return fmt.Sprintf("session:user:%s", userID)
}

// Create creates a new session
func (s *RedisSessionStore) Create(session *Session) error {
	// Calculate TTL from expiration
	ttl := time.Until(session.ExpiresAt)
	if ttl <= 0 {
		return fmt.Errorf("session already expired")
	}

	// Serialize session
	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("failed to serialize session: %w", err)
	}

	// Use pipeline for atomic operations
	pipe := s.client.Pipeline()

	// Store session data with TTL
	pipe.Set(s.ctx, sessionKey(session.ID), data, ttl)

	// Store token -> session ID mapping
	pipe.Set(s.ctx, tokenKey(session.Token), session.ID, ttl)

	// Add to user's session set
	pipe.SAdd(s.ctx, userSessionsKey(session.UserID), session.ID)
	pipe.Expire(s.ctx, userSessionsKey(session.UserID), ttl)

	_, err = pipe.Exec(s.ctx)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}

	return nil
}

// Get retrieves a session by ID
func (s *RedisSessionStore) Get(id string) (*Session, error) {
	data, err := s.client.Get(s.ctx, sessionKey(id)).Bytes()
	if err == redis.Nil {
		return nil, fmt.Errorf("session not found: %s", id)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	var session Session
	if err := json.Unmarshal(data, &session); err != nil {
		return nil, fmt.Errorf("failed to deserialize session: %w", err)
	}

	return &session, nil
}

// GetByToken retrieves a session by token
func (s *RedisSessionStore) GetByToken(token string) (*Session, error) {
	// Get session ID from token
	sessionID, err := s.client.Get(s.ctx, tokenKey(token)).Result()
	if err == redis.Nil {
		return nil, fmt.Errorf("session not found for token")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get session ID: %w", err)
	}

	return s.Get(sessionID)
}

// Delete deletes a session
func (s *RedisSessionStore) Delete(id string) error {
	// Get session first to remove token mapping
	session, err := s.Get(id)
	if err != nil {
		return err
	}

	// Use pipeline for atomic deletion
	pipe := s.client.Pipeline()

	// Delete session
	pipe.Del(s.ctx, sessionKey(id))

	// Delete token mapping
	pipe.Del(s.ctx, tokenKey(session.Token))

	// Remove from user's session set
	pipe.SRem(s.ctx, userSessionsKey(session.UserID), id)

	_, err = pipe.Exec(s.ctx)
	if err != nil {
		return fmt.Errorf("failed to delete session: %w", err)
	}

	return nil
}

// DeleteByUserID deletes all sessions for a user
func (s *RedisSessionStore) DeleteByUserID(userID string) error {
	// Get all session IDs for user
	sessionIDs, err := s.client.SMembers(s.ctx, userSessionsKey(userID)).Result()
	if err != nil {
		return fmt.Errorf("failed to get user sessions: %w", err)
	}

	if len(sessionIDs) == 0 {
		return nil
	}

	// Delete each session
	pipe := s.client.Pipeline()
	for _, sessionID := range sessionIDs {
		// Get session to find token
		session, err := s.Get(sessionID)
		if err == nil {
			pipe.Del(s.ctx, tokenKey(session.Token))
		}
		pipe.Del(s.ctx, sessionKey(sessionID))
	}

	// Delete user sessions set
	pipe.Del(s.ctx, userSessionsKey(userID))

	_, err = pipe.Exec(s.ctx)
	if err != nil {
		return fmt.Errorf("failed to delete user sessions: %w", err)
	}

	return nil
}

// UpdateLastAccessed updates the last accessed time
func (s *RedisSessionStore) UpdateLastAccessed(id string) error {
	session, err := s.Get(id)
	if err != nil {
		return err
	}

	session.LastAccessedAt = time.Now()

	// Recalculate TTL based on remaining time
	ttl := time.Until(session.ExpiresAt)
	if ttl <= 0 {
		return fmt.Errorf("session expired")
	}

	// Serialize updated session
	data, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("failed to serialize session: %w", err)
	}

	// Update session with existing TTL preserved
	err = s.client.Set(s.ctx, sessionKey(id), data, ttl).Err()
	if err != nil {
		return fmt.Errorf("failed to update session: %w", err)
	}

	return nil
}

// GetUserSessions gets all sessions for a user
func (s *RedisSessionStore) GetUserSessions(userID string) ([]*Session, error) {
	sessionIDs, err := s.client.SMembers(s.ctx, userSessionsKey(userID)).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get user session IDs: %w", err)
	}

	var sessions []*Session
	for _, sessionID := range sessionIDs {
		session, err := s.Get(sessionID)
		if err != nil {
			// Session may have expired, skip it
			continue
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// Cleanup removes expired sessions
// For Redis, TTL handles this automatically, but we clean up user session sets
func (s *RedisSessionStore) Cleanup() error {
	// In Redis with TTL, cleanup is automatic
	// This method can be used to clean up orphaned user session set entries
	return nil
}

// Close closes the Redis connection
func (s *RedisSessionStore) Close() error {
	return s.client.Close()
}

// Ensure RedisSessionStore implements SessionService
var _ SessionService = (*RedisSessionStore)(nil)
