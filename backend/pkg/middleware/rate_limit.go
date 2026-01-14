package middleware

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"golang.org/x/time/rate"
)

// RateLimitConfig defines rate limiting configuration
type RateLimitConfig struct {
	// RequestsPerMinute is the number of requests allowed per minute
	RequestsPerMinute int
	// BurstSize is the maximum burst size (usually equal to or slightly larger than RequestsPerMinute)
	BurstSize int
	// KeyPrefix is the Redis key prefix for this rate limiter
	KeyPrefix string
	// UseRedis enables distributed rate limiting with Redis
	UseRedis bool
	// RedisClient is the Redis client for distributed rate limiting
	RedisClient *redis.Client
	// ExcludedIPs are IPs exempt from rate limiting (e.g., internal services)
	ExcludedIPs []string
	// SkipPaths are paths exempt from rate limiting
	SkipPaths []string
	// IdentifyByToken if true, uses JWT user ID instead of IP for rate limiting
	IdentifyByToken bool
}

// RateLimiter handles rate limiting for API endpoints
type RateLimiter struct {
	config        RateLimitConfig
	localLimiters sync.Map // map[string]*rate.Limiter for local rate limiting
	ctx           context.Context
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(config RateLimitConfig) *RateLimiter {
	if config.RequestsPerMinute == 0 {
		config.RequestsPerMinute = 100
	}
	if config.BurstSize == 0 {
		config.BurstSize = config.RequestsPerMinute
	}
	if config.KeyPrefix == "" {
		config.KeyPrefix = "ratelimit"
	}

	return &RateLimiter{
		config: config,
		ctx:    context.Background(),
	}
}

// Limit is the rate limiting middleware
func (rl *RateLimiter) Limit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if path should be skipped
		for _, path := range rl.config.SkipPaths {
			if strings.HasPrefix(r.URL.Path, path) {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Get client identifier
		clientID := rl.getClientID(r)

		// Check if IP is excluded
		clientIP := getClientIP(r)
		for _, excludedIP := range rl.config.ExcludedIPs {
			if clientIP == excludedIP {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Check rate limit
		allowed, remaining, resetAt := rl.checkLimit(clientID)
		if !allowed {
			rl.writeRateLimitError(w, remaining, resetAt)
			return
		}

		// Set rate limit headers
		w.Header().Set("X-RateLimit-Limit", strconv.Itoa(rl.config.RequestsPerMinute))
		w.Header().Set("X-RateLimit-Remaining", strconv.Itoa(remaining))
		w.Header().Set("X-RateLimit-Reset", strconv.FormatInt(resetAt.Unix(), 10))

		next.ServeHTTP(w, r)
	})
}

// getClientID returns the identifier for rate limiting
func (rl *RateLimiter) getClientID(r *http.Request) string {
	// If token-based identification is enabled, use user ID
	if rl.config.IdentifyByToken {
		if userID, ok := r.Context().Value(UserIDKey).(string); ok && userID != "" {
			return "user:" + userID
		}
	}

	// Fall back to IP-based identification
	return "ip:" + getClientIP(r)
}

// checkLimit checks if the client is within rate limits
func (rl *RateLimiter) checkLimit(clientID string) (bool, int, time.Time) {
	if rl.config.UseRedis && rl.config.RedisClient != nil {
		return rl.checkRedisLimit(clientID)
	}
	return rl.checkLocalLimit(clientID)
}

// checkLocalLimit uses in-memory rate limiting
func (rl *RateLimiter) checkLocalLimit(clientID string) (bool, int, time.Time) {
	limiter := rl.getLocalLimiter(clientID)

	// Check if request is allowed
	now := time.Now()
	resetAt := now.Add(time.Minute)

	if !limiter.Allow() {
		return false, 0, resetAt
	}

	// Calculate approximate remaining tokens
	remaining := int(limiter.Tokens())
	return true, remaining, resetAt
}

// getLocalLimiter gets or creates a local rate limiter for a client
func (rl *RateLimiter) getLocalLimiter(clientID string) *rate.Limiter {
	if limiter, ok := rl.localLimiters.Load(clientID); ok {
		return limiter.(*rate.Limiter)
	}

	// Create new limiter: tokens per second = RequestsPerMinute / 60
	tokensPerSecond := rate.Limit(float64(rl.config.RequestsPerMinute) / 60.0)
	limiter := rate.NewLimiter(tokensPerSecond, rl.config.BurstSize)

	// Store limiter
	actual, loaded := rl.localLimiters.LoadOrStore(clientID, limiter)
	if loaded {
		return actual.(*rate.Limiter)
	}
	return limiter
}

// checkRedisLimit uses Redis for distributed rate limiting
func (rl *RateLimiter) checkRedisLimit(clientID string) (bool, int, time.Time) {
	key := fmt.Sprintf("%s:%s", rl.config.KeyPrefix, clientID)
	now := time.Now()
	windowStart := now.Truncate(time.Minute)
	resetAt := windowStart.Add(time.Minute)

	// Use sliding window counter algorithm
	pipe := rl.config.RedisClient.Pipeline()

	// Increment counter
	incrCmd := pipe.Incr(rl.ctx, key)

	// Set expiration (only if key is new)
	pipe.Expire(rl.ctx, key, time.Minute)

	_, err := pipe.Exec(rl.ctx)
	if err != nil {
		// On Redis error, allow the request (fail open)
		return true, rl.config.RequestsPerMinute, resetAt
	}

	count := int(incrCmd.Val())
	remaining := rl.config.RequestsPerMinute - count
	if remaining < 0 {
		remaining = 0
	}

	return count <= rl.config.RequestsPerMinute, remaining, resetAt
}

// writeRateLimitError writes a 429 Too Many Requests error
func (rl *RateLimiter) writeRateLimitError(w http.ResponseWriter, remaining int, resetAt time.Time) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("X-RateLimit-Limit", strconv.Itoa(rl.config.RequestsPerMinute))
	w.Header().Set("X-RateLimit-Remaining", "0")
	w.Header().Set("X-RateLimit-Reset", strconv.FormatInt(resetAt.Unix(), 10))
	w.Header().Set("Retry-After", strconv.FormatInt(int64(time.Until(resetAt).Seconds()), 10))
	w.WriteHeader(http.StatusTooManyRequests)

	response := map[string]interface{}{
		"error": map[string]interface{}{
			"code":       "rate_limit_exceeded",
			"message":    "Too many requests. Please try again later.",
			"retry_after": int64(time.Until(resetAt).Seconds()),
		},
	}
	_ = json.NewEncoder(w).Encode(response)
}

// getClientIP extracts the client IP from the request
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (for proxied requests)
	xff := r.Header.Get("X-Forwarded-For")
	if xff != "" {
		// Get the first IP in the list
		ips := strings.Split(xff, ",")
		if len(ips) > 0 {
			ip := strings.TrimSpace(ips[0])
			if ip != "" {
				return ip
			}
		}
	}

	// Check X-Real-IP header
	xri := r.Header.Get("X-Real-IP")
	if xri != "" {
		return xri
	}

	// Fall back to RemoteAddr
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}

// PresetRateLimiters provides common rate limiter configurations

// LoginRateLimiter creates a rate limiter for login endpoints (5/min)
func LoginRateLimiter(redisClient *redis.Client) *RateLimiter {
	return NewRateLimiter(RateLimitConfig{
		RequestsPerMinute: 5,
		BurstSize:         5,
		KeyPrefix:         "ratelimit:login",
		UseRedis:          redisClient != nil,
		RedisClient:       redisClient,
	})
}

// RegisterRateLimiter creates a rate limiter for registration endpoints (3/min)
func RegisterRateLimiter(redisClient *redis.Client) *RateLimiter {
	return NewRateLimiter(RateLimitConfig{
		RequestsPerMinute: 3,
		BurstSize:         3,
		KeyPrefix:         "ratelimit:register",
		UseRedis:          redisClient != nil,
		RedisClient:       redisClient,
	})
}

// APIRateLimiter creates a rate limiter for general API endpoints (100/min)
func APIRateLimiter(redisClient *redis.Client) *RateLimiter {
	return NewRateLimiter(RateLimitConfig{
		RequestsPerMinute: 100,
		BurstSize:         100,
		KeyPrefix:         "ratelimit:api",
		UseRedis:          redisClient != nil,
		RedisClient:       redisClient,
		IdentifyByToken:   true, // Use user ID when available
	})
}

// PasswordResetRateLimiter creates a rate limiter for password reset (2/min)
func PasswordResetRateLimiter(redisClient *redis.Client) *RateLimiter {
	return NewRateLimiter(RateLimitConfig{
		RequestsPerMinute: 2,
		BurstSize:         2,
		KeyPrefix:         "ratelimit:password_reset",
		UseRedis:          redisClient != nil,
		RedisClient:       redisClient,
	})
}

// CleanupLocalLimiters removes stale local limiters (call periodically)
func (rl *RateLimiter) CleanupLocalLimiters() {
	// In a production system, you'd want to track last access time
	// and remove limiters that haven't been used recently
	// For now, this is a placeholder
}
