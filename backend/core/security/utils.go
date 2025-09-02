package security

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
	
	"golang.org/x/crypto/bcrypt"
)

// PasswordHasher handles secure password hashing
type PasswordHasher struct {
	cost int
}

// NewPasswordHasher creates a new password hasher
func NewPasswordHasher() *PasswordHasher {
	return &PasswordHasher{
		cost: bcrypt.DefaultCost,
	}
}

// HashPassword creates a secure hash of a password
func (ph *PasswordHasher) HashPassword(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), ph.cost)
	if err != nil {
		return "", fmt.Errorf("failed to hash password: %w", err)
	}
	return string(bytes), nil
}

// VerifyPassword checks if a password matches a hash
func (ph *PasswordHasher) VerifyPassword(password, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
	return err == nil
}

// SecureCompare performs constant-time comparison
func SecureCompare(a, b string) bool {
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}

// SanitizeInput removes potentially dangerous characters from user input
func SanitizeInput(input string) string {
	// Remove null bytes
	input = strings.ReplaceAll(input, "\x00", "")
	
	// Trim whitespace
	input = strings.TrimSpace(input)
	
	// Remove control characters
	var result strings.Builder
	for _, r := range input {
		if r >= 32 && r != 127 { // Printable characters only
			result.WriteRune(r)
		}
	}
	
	return result.String()
}

// ValidateEmail performs basic email validation
func ValidateEmail(email string) bool {
	email = strings.ToLower(strings.TrimSpace(email))
	
	// Basic validation
	if len(email) < 3 || len(email) > 254 {
		return false
	}
	
	// Must contain @ and .
	if !strings.Contains(email, "@") || !strings.Contains(email, ".") {
		return false
	}
	
	// Split into local and domain parts
	parts := strings.Split(email, "@")
	if len(parts) != 2 {
		return false
	}
	
	local := parts[0]
	domain := parts[1]
	
	// Validate local part
	if len(local) < 1 || len(local) > 64 {
		return false
	}
	
	// Validate domain part
	if len(domain) < 3 || len(domain) > 253 {
		return false
	}
	
	// Domain must contain at least one dot
	if !strings.Contains(domain, ".") {
		return false
	}
	
	return true
}

// SecurityHeaders middleware adds security headers to responses
func SecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Security headers
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
		w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
		w.Header().Set("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
		
		next.ServeHTTP(w, r)
	})
}

// RateLimiter implements basic rate limiting
type RateLimiter struct {
	requests map[string][]int64
	limit    int
	window   int64 // in seconds
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(limit int, windowSeconds int64) *RateLimiter {
	return &RateLimiter{
		requests: make(map[string][]int64),
		limit:    limit,
		window:   windowSeconds,
	}
}

// Allow checks if a request should be allowed
func (rl *RateLimiter) Allow(identifier string) bool {
	now := time.Now().Unix()
	cutoff := now - rl.window
	
	// Clean old requests
	if requests, ok := rl.requests[identifier]; ok {
		var cleaned []int64
		for _, timestamp := range requests {
			if timestamp > cutoff {
				cleaned = append(cleaned, timestamp)
			}
		}
		rl.requests[identifier] = cleaned
	}
	
	// Check limit
	if len(rl.requests[identifier]) >= rl.limit {
		return false
	}
	
	// Add new request
	rl.requests[identifier] = append(rl.requests[identifier], now)
	return true
}

// InputValidator provides input validation utilities
type InputValidator struct{}

// NewInputValidator creates a new input validator
func NewInputValidator() *InputValidator {
	return &InputValidator{}
}

// ValidateVMName validates a VM name
func (iv *InputValidator) ValidateVMName(name string) error {
	if len(name) < 1 || len(name) > 63 {
		return fmt.Errorf("VM name must be between 1 and 63 characters")
	}
	
	// Allow alphanumeric, hyphens, and underscores
	for _, r := range name {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || 
		     (r >= '0' && r <= '9') || r == '-' || r == '_') {
			return fmt.Errorf("VM name contains invalid character: %c", r)
		}
	}
	
	return nil
}

// ValidateNodeID validates a node identifier
func (iv *InputValidator) ValidateNodeID(nodeID string) error {
	if len(nodeID) < 1 || len(nodeID) > 36 {
		return fmt.Errorf("node ID must be between 1 and 36 characters")
	}
	
	// Allow UUID format or simple alphanumeric
	for _, r := range nodeID {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || 
		     (r >= '0' && r <= '9') || r == '-') {
			return fmt.Errorf("node ID contains invalid character: %c", r)
		}
	}
	
	return nil
}

// ValidateResourceLimit validates resource specifications
func (iv *InputValidator) ValidateResourceLimit(value int64, min, max int64) error {
	if value < min || value > max {
		return fmt.Errorf("value must be between %d and %d", min, max)
	}
	return nil
}

// SecureRandomString generates a secure random string
func SecureRandomString(length int) (string, error) {
	bytes := make([]byte, length)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("failed to generate random string: %w", err)
	}
	return base64.URLEncoding.EncodeToString(bytes)[:length], nil
}

// AuditLogger logs security-relevant events
type AuditLogger struct {
	logger *log.Logger
}

// NewAuditLogger creates a new audit logger
func NewAuditLogger(output io.Writer) *AuditLogger {
	return &AuditLogger{
		logger: log.New(output, "[AUDIT] ", log.LstdFlags|log.LUTC),
	}
}

// LogAuthentication logs authentication attempts
func (al *AuditLogger) LogAuthentication(ctx context.Context, email string, success bool, ip string) {
	status := "SUCCESS"
	if !success {
		status = "FAILURE"
	}
	al.logger.Printf("AUTH %s email=%s ip=%s", status, email, ip)
}

// LogAccessControl logs access control decisions
func (al *AuditLogger) LogAccessControl(ctx context.Context, user, resource, action string, allowed bool) {
	status := "ALLOWED"
	if !allowed {
		status = "DENIED"
	}
	al.logger.Printf("ACCESS %s user=%s resource=%s action=%s", status, user, resource, action)
}

// LogDataAccess logs data access events
func (al *AuditLogger) LogDataAccess(ctx context.Context, user, dataType, operation string) {
	al.logger.Printf("DATA user=%s type=%s operation=%s", user, dataType, operation)
}

// LogSecurityEvent logs general security events
func (al *AuditLogger) LogSecurityEvent(ctx context.Context, eventType, description string) {
	al.logger.Printf("SECURITY event=%s description=%s", eventType, description)
}