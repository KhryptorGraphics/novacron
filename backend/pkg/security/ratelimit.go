package security

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/pkg/errors"
)

// RateLimiter provides rate limiting functionality
type RateLimiter struct {
	clients map[string]*ClientInfo
	mutex   sync.RWMutex
	config  RateLimitConfig
}

// ClientInfo holds rate limiting information for a client
type ClientInfo struct {
	Requests  int
	ResetTime time.Time
	Blocked   bool
	BlockedUntil time.Time
}

// RateLimitConfig holds rate limiting configuration
type RateLimitConfig struct {
	RequestsPerMinute   int           `json:"requests_per_minute"`
	BurstSize          int           `json:"burst_size"`
	BlockDuration      time.Duration `json:"block_duration"`
	CleanupInterval    time.Duration `json:"cleanup_interval"`
	WhitelistedIPs     []string      `json:"whitelisted_ips"`
	TrustedProxies     []string      `json:"trusted_proxies"`
}

// DefaultRateLimitConfig returns default rate limiting configuration
func DefaultRateLimitConfig() RateLimitConfig {
	return RateLimitConfig{
		RequestsPerMinute: 60,
		BurstSize:        10,
		BlockDuration:    15 * time.Minute,
		CleanupInterval:  5 * time.Minute,
		WhitelistedIPs:   []string{"127.0.0.1", "::1"},
		TrustedProxies:   []string{"127.0.0.1", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"},
	}
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(config RateLimitConfig) *RateLimiter {
	rl := &RateLimiter{
		clients: make(map[string]*ClientInfo),
		config:  config,
	}
	
	// Start cleanup goroutine
	go rl.startCleanup()
	
	return rl
}

// Allow checks if a request should be allowed
func (rl *RateLimiter) Allow(clientIP string) (bool, *errors.AppError) {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()
	
	// Check if IP is whitelisted
	if rl.isWhitelisted(clientIP) {
		return true, nil
	}
	
	now := time.Now()
	
	// Get or create client info
	client, exists := rl.clients[clientIP]
	if !exists {
		client = &ClientInfo{
			Requests:  0,
			ResetTime: now.Add(time.Minute),
		}
		rl.clients[clientIP] = client
	}
	
	// Check if client is currently blocked
	if client.Blocked && now.Before(client.BlockedUntil) {
		return false, errors.NewAppErrorWithDetails(
			errors.ErrResourceExhausted,
			"Rate limit exceeded - client blocked",
			fmt.Sprintf("Client blocked until %s", client.BlockedUntil.Format(time.RFC3339)),
		).WithMetadata(map[string]interface{}{
			"client_ip": clientIP,
			"blocked_until": client.BlockedUntil,
		})
	}
	
	// Reset counter if time window has passed
	if now.After(client.ResetTime) {
		client.Requests = 0
		client.ResetTime = now.Add(time.Minute)
		client.Blocked = false
	}
	
	// Check rate limit
	if client.Requests >= rl.config.RequestsPerMinute {
		// Block client
		client.Blocked = true
		client.BlockedUntil = now.Add(rl.config.BlockDuration)
		
		return false, errors.NewAppErrorWithDetails(
			errors.ErrResourceExhausted,
			"Rate limit exceeded",
			fmt.Sprintf("Maximum %d requests per minute exceeded", rl.config.RequestsPerMinute),
		).WithMetadata(map[string]interface{}{
			"client_ip": clientIP,
			"requests": client.Requests,
			"limit": rl.config.RequestsPerMinute,
		})
	}
	
	// Increment request count
	client.Requests++
	
	return true, nil
}

// GetClientInfo returns rate limiting information for a client
func (rl *RateLimiter) GetClientInfo(clientIP string) *ClientInfo {
	rl.mutex.RLock()
	defer rl.mutex.RUnlock()
	
	if client, exists := rl.clients[clientIP]; exists {
		// Return a copy to prevent external modification
		return &ClientInfo{
			Requests:     client.Requests,
			ResetTime:    client.ResetTime,
			Blocked:      client.Blocked,
			BlockedUntil: client.BlockedUntil,
		}
	}
	
	return nil
}

// isWhitelisted checks if an IP is whitelisted
func (rl *RateLimiter) isWhitelisted(clientIP string) bool {
	for _, whitelistedIP := range rl.config.WhitelistedIPs {
		if clientIP == whitelistedIP {
			return true
		}
		
		// Check CIDR ranges
		if _, cidr, err := net.ParseCIDR(whitelistedIP); err == nil {
			if ip := net.ParseIP(clientIP); ip != nil && cidr.Contains(ip) {
				return true
			}
		}
	}
	return false
}

// startCleanup starts the cleanup goroutine to remove stale entries
func (rl *RateLimiter) startCleanup() {
	ticker := time.NewTicker(rl.config.CleanupInterval)
	defer ticker.Stop()
	
	for range ticker.C {
		rl.cleanup()
	}
}

// cleanup removes stale client entries
func (rl *RateLimiter) cleanup() {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()
	
	now := time.Now()
	staleThreshold := now.Add(-2 * time.Hour) // Keep entries for 2 hours
	
	for clientIP, client := range rl.clients {
		// Remove entries that are old and not blocked
		if !client.Blocked && client.ResetTime.Before(staleThreshold) {
			delete(rl.clients, clientIP)
		}
		// Remove entries where block has expired and reset time has passed
		if client.Blocked && client.BlockedUntil.Before(now) && client.ResetTime.Before(staleThreshold) {
			delete(rl.clients, clientIP)
		}
	}
}

// RateLimitMiddleware creates HTTP middleware for rate limiting
func RateLimitMiddleware(rateLimiter *RateLimiter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Get client IP
			clientIP := GetClientIP(r, rateLimiter.config.TrustedProxies)
			
			// Check rate limit
			allowed, rateLimitErr := rateLimiter.Allow(clientIP)
			if !allowed {
				// Add rate limit headers
				client := rateLimiter.GetClientInfo(clientIP)
				if client != nil {
					w.Header().Set("X-RateLimit-Limit", fmt.Sprintf("%d", rateLimiter.config.RequestsPerMinute))
					w.Header().Set("X-RateLimit-Remaining", "0")
					w.Header().Set("X-RateLimit-Reset", fmt.Sprintf("%d", client.ResetTime.Unix()))
					
					if client.Blocked {
						w.Header().Set("Retry-After", fmt.Sprintf("%.0f", time.Until(client.BlockedUntil).Seconds()))
					}
				}
				
				// Return error response
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(rateLimitErr.HTTPStatusCode())
				
				if err := rateLimitErr.MarshalJSON(); err == nil {
					w.Write(err)
				}
				return
			}
			
			// Add rate limit headers for successful requests
			client := rateLimiter.GetClientInfo(clientIP)
			if client != nil {
				remaining := rateLimiter.config.RequestsPerMinute - client.Requests
				if remaining < 0 {
					remaining = 0
				}
				
				w.Header().Set("X-RateLimit-Limit", fmt.Sprintf("%d", rateLimiter.config.RequestsPerMinute))
				w.Header().Set("X-RateLimit-Remaining", fmt.Sprintf("%d", remaining))
				w.Header().Set("X-RateLimit-Reset", fmt.Sprintf("%d", client.ResetTime.Unix()))
			}
			
			// Continue with next handler
			next.ServeHTTP(w, r)
		})
	}
}

// GetClientIP extracts the client IP address from the request
func GetClientIP(r *http.Request, trustedProxies []string) string {
	// Check X-Forwarded-For header (from trusted proxies)
	if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
		// Get the first IP in the chain
		ips := strings.Split(forwarded, ",")
		if len(ips) > 0 {
			clientIP := strings.TrimSpace(ips[0])
			
			// Verify the request comes from a trusted proxy
			remoteIP, _, _ := net.SplitHostPort(r.RemoteAddr)
			if isTrustedProxy(remoteIP, trustedProxies) {
				return clientIP
			}
		}
	}
	
	// Check X-Real-IP header (from trusted proxies)
	if realIP := r.Header.Get("X-Real-IP"); realIP != "" {
		remoteIP, _, _ := net.SplitHostPort(r.RemoteAddr)
		if isTrustedProxy(remoteIP, trustedProxies) {
			return realIP
		}
	}
	
	// Fall back to RemoteAddr
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return ip
}

// isTrustedProxy checks if an IP is a trusted proxy
func isTrustedProxy(ip string, trustedProxies []string) bool {
	for _, proxy := range trustedProxies {
		if ip == proxy {
			return true
		}
		
		// Check CIDR ranges
		if _, cidr, err := net.ParseCIDR(proxy); err == nil {
			if clientIP := net.ParseIP(ip); clientIP != nil && cidr.Contains(clientIP) {
				return true
			}
		}
	}
	return false
}

// Advanced rate limiting with different tiers
type TieredRateLimiter struct {
	tiers  map[string]*RateLimiter
	config TieredRateLimitConfig
}

type TieredRateLimitConfig struct {
	Default    RateLimitConfig            `json:"default"`
	Tiers      map[string]RateLimitConfig `json:"tiers"`
	UserTiers  map[string]string          `json:"user_tiers"`
	PathLimits map[string]RateLimitConfig `json:"path_limits"`
}

// NewTieredRateLimiter creates a tiered rate limiter
func NewTieredRateLimiter(config TieredRateLimitConfig) *TieredRateLimiter {
	trl := &TieredRateLimiter{
		tiers:  make(map[string]*RateLimiter),
		config: config,
	}
	
	// Create rate limiters for each tier
	trl.tiers["default"] = NewRateLimiter(config.Default)
	for tierName, tierConfig := range config.Tiers {
		trl.tiers[tierName] = NewRateLimiter(tierConfig)
	}
	
	// Create path-specific rate limiters
	for path, pathConfig := range config.PathLimits {
		trl.tiers["path_"+path] = NewRateLimiter(pathConfig)
	}
	
	return trl
}

// Allow checks rate limits using tiered configuration
func (trl *TieredRateLimiter) Allow(clientIP, userID, path string) (bool, *errors.AppError) {
	// Determine which rate limiter to use
	var rateLimiter *RateLimiter
	
	// Check for path-specific limits first
	if pathLimiter, exists := trl.tiers["path_"+path]; exists {
		rateLimiter = pathLimiter
	} else if userID != "" {
		// Check user tier
		if tier, exists := trl.config.UserTiers[userID]; exists {
			if tierLimiter, exists := trl.tiers[tier]; exists {
				rateLimiter = tierLimiter
			}
		}
	}
	
	// Fall back to default
	if rateLimiter == nil {
		rateLimiter = trl.tiers["default"]
	}
	
	return rateLimiter.Allow(clientIP)
}

// DDoS protection middleware
func DDoSProtectionMiddleware(rateLimiter *RateLimiter, threshold int) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			clientIP := GetClientIP(r, rateLimiter.config.TrustedProxies)
			client := rateLimiter.GetClientInfo(clientIP)
			
			// If client is making too many requests, apply stricter limits
			if client != nil && client.Requests > threshold {
				// Block for longer duration
				rateLimiter.clients[clientIP].BlockedUntil = time.Now().Add(time.Hour)
				
				http.Error(w, "DDoS protection activated", http.StatusTooManyRequests)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	}
}