package auth

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
)

// SecurityConfig defines security middleware configuration
type SecurityConfig struct {
	// RateLimiting configuration
	RateLimitEnabled   bool
	RateLimitRequests  int           // Requests per window
	RateLimitWindow    time.Duration // Time window
	RateLimitBurstSize int           // Burst allowance
	RateLimitByIP      bool          // Rate limit by IP
	RateLimitByUser    bool          // Rate limit by user
	RateLimitByTenant  bool          // Rate limit by tenant

	// Input validation
	InputValidationEnabled bool
	MaxRequestSize         int64 // Maximum request body size
	AllowedContentTypes    []string
	SQLInjectionCheck      bool
	XSSProtection          bool
	CSRFProtection         bool

	// Security headers
	SecurityHeadersEnabled bool
	HSTSMaxAge             int
	ContentSecurityPolicy  string
	ReferrerPolicy         string

	// IP restrictions
	IPWhitelistEnabled bool
	IPWhitelist        []string // CIDR blocks
	IPBlacklistEnabled bool
	IPBlacklist        []string // CIDR blocks

	// Geolocation restrictions
	GeoBlockEnabled  bool
	AllowedCountries []string
	BlockedCountries []string

	// Bot protection
	BotProtectionEnabled bool
	UserAgentBlacklist   []string
	HoneypotEnabled      bool

	// Audit logging
	AuditLogging     bool
	LogSensitiveData bool
}

// RateLimitEntry represents a rate limiting entry
type RateLimitEntry struct {
	Count        int
	Window       time.Time
	LastAccess   time.Time
	Blocked      bool
	BlockedUntil time.Time
}

// SecurityContext contains security information for a request
type SecurityContext struct {
	UserID      string
	TenantID    string
	SessionID   string
	ClientIP    string
	UserAgent   string
	Country     string
	ThreatLevel int // 0-100 threat score
	RiskFactors []string
	Permissions []string
}

// SecurityMiddleware provides comprehensive API security
type SecurityMiddleware struct {
	config               SecurityConfig
	rateLimits           map[string]*RateLimitEntry
	mu                   sync.RWMutex
	sqlInjectionPatterns []*regexp.Regexp
	xssPatterns          []*regexp.Regexp
	botPatterns          []*regexp.Regexp
	ipWhitelist          []*net.IPNet
	ipBlacklist          []*net.IPNet
	auditService         AuditService
	encryptionService    *EncryptionService
}

// NewSecurityMiddleware creates a new security middleware
func NewSecurityMiddleware(config SecurityConfig, auditService AuditService, encryptionService *EncryptionService) *SecurityMiddleware {
	// Set defaults
	if config.RateLimitRequests == 0 {
		config.RateLimitRequests = 1000
	}
	if config.RateLimitWindow == 0 {
		config.RateLimitWindow = time.Hour
	}
	if config.RateLimitBurstSize == 0 {
		config.RateLimitBurstSize = 10
	}
	if config.MaxRequestSize == 0 {
		config.MaxRequestSize = 10 * 1024 * 1024 // 10MB
	}
	if len(config.AllowedContentTypes) == 0 {
		config.AllowedContentTypes = []string{
			"application/json",
			"application/x-www-form-urlencoded",
			"multipart/form-data",
			"text/plain",
		}
	}
	if config.HSTSMaxAge == 0 {
		config.HSTSMaxAge = 31536000 // 1 year
	}
	if config.ContentSecurityPolicy == "" {
		config.ContentSecurityPolicy = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
	}
	if config.ReferrerPolicy == "" {
		config.ReferrerPolicy = "strict-origin-when-cross-origin"
	}

	middleware := &SecurityMiddleware{
		config:            config,
		rateLimits:        make(map[string]*RateLimitEntry),
		auditService:      auditService,
		encryptionService: encryptionService,
	}

	// Compile patterns
	middleware.compileSQLInjectionPatterns()
	middleware.compileXSSPatterns()
	middleware.compileBotPatterns()

	// Parse IP ranges
	middleware.parseIPRanges()

	return middleware
}

// Middleware returns the HTTP middleware handler
func (s *SecurityMiddleware) Middleware(authService AuthService) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			ctx := r.Context()
			secCtx := &SecurityContext{
				ClientIP:    s.getClientIP(r),
				UserAgent:   r.UserAgent(),
				ThreatLevel: 0,
				RiskFactors: make([]string, 0),
			}

			// IP-based restrictions
			if blocked, reason := s.checkIPRestrictions(secCtx.ClientIP); blocked {
				s.logSecurityEvent("ip_blocked", secCtx, reason, r)
				http.Error(w, "Access denied", http.StatusForbidden)
				return
			}

			// Bot protection
			if s.config.BotProtectionEnabled && s.isBot(secCtx.UserAgent) {
				secCtx.ThreatLevel += 20
				secCtx.RiskFactors = append(secCtx.RiskFactors, "bot_detected")
			}

			// Rate limiting
			if s.config.RateLimitEnabled {
				if limited, reason := s.checkRateLimit(secCtx, r); limited {
					s.logSecurityEvent("rate_limited", secCtx, reason, r)
					w.Header().Set("Retry-After", "60")
					http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
					return
				}
			}

			// Authentication
			authToken := s.extractAuthToken(r)
			if authToken != "" {
				if session, err := authService.ValidateSession(authToken, authToken); err == nil {
					secCtx.UserID = session.UserID
					secCtx.TenantID = session.TenantID
					secCtx.SessionID = session.ID

					// Get user permissions
					if roles, err := authService.GetUserRoles(session.UserID); err == nil {
						for _, role := range roles {
							for _, permission := range role.Permissions {
								permStr := fmt.Sprintf("%s:%s", permission.Resource, permission.Action)
								secCtx.Permissions = append(secCtx.Permissions, permStr)
							}
						}
					}
				}
			}

			// Input validation
			if s.config.InputValidationEnabled {
				if valid, reason := s.validateInput(r); !valid {
					secCtx.ThreatLevel += 50
					secCtx.RiskFactors = append(secCtx.RiskFactors, reason)
					s.logSecurityEvent("input_validation_failed", secCtx, reason, r)
					http.Error(w, "Invalid input detected", http.StatusBadRequest)
					return
				}
			}

			// Security headers
			if s.config.SecurityHeadersEnabled {
				s.setSecurityHeaders(w)
			}

			// Add security context to request
			ctx = context.WithValue(ctx, "security_context", secCtx)
			r = r.WithContext(ctx)

			// Continue to next handler
			next.ServeHTTP(w, r)

			// Log successful request
			if s.config.AuditLogging {
				s.logSecurityEvent("request_completed", secCtx, "", r)
			}
		})
	}
}

// checkRateLimit checks if the request should be rate limited
func (s *SecurityMiddleware) checkRateLimit(secCtx *SecurityContext, r *http.Request) (bool, string) {
	var keys []string

	if s.config.RateLimitByIP {
		keys = append(keys, "ip:"+secCtx.ClientIP)
	}
	if s.config.RateLimitByUser && secCtx.UserID != "" {
		keys = append(keys, "user:"+secCtx.UserID)
	}
	if s.config.RateLimitByTenant && secCtx.TenantID != "" {
		keys = append(keys, "tenant:"+secCtx.TenantID)
	}

	if len(keys) == 0 {
		keys = []string{"global"}
	}

	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, key := range keys {
		entry, exists := s.rateLimits[key]
		if !exists {
			entry = &RateLimitEntry{
				Count:      0,
				Window:     now,
				LastAccess: now,
			}
			s.rateLimits[key] = entry
		}

		// Check if blocked
		if entry.Blocked && now.Before(entry.BlockedUntil) {
			return true, fmt.Sprintf("blocked until %v", entry.BlockedUntil)
		}

		// Reset window if expired
		if now.Sub(entry.Window) > s.config.RateLimitWindow {
			entry.Count = 0
			entry.Window = now
			entry.Blocked = false
		}

		// Check rate limit
		entry.Count++
		entry.LastAccess = now

		if entry.Count > s.config.RateLimitRequests {
			// Block for double the window period
			entry.Blocked = true
			entry.BlockedUntil = now.Add(s.config.RateLimitWindow * 2)
			return true, fmt.Sprintf("rate limit exceeded: %d/%d requests", entry.Count, s.config.RateLimitRequests)
		}
	}

	return false, ""
}

// checkIPRestrictions checks IP whitelist/blacklist
func (s *SecurityMiddleware) checkIPRestrictions(clientIP string) (bool, string) {
	ip := net.ParseIP(clientIP)
	if ip == nil {
		return true, "invalid IP address"
	}

	// Check blacklist first
	if s.config.IPBlacklistEnabled {
		for _, blacklistNet := range s.ipBlacklist {
			if blacklistNet.Contains(ip) {
				return true, "IP in blacklist"
			}
		}
	}

	// Check whitelist
	if s.config.IPWhitelistEnabled {
		allowed := false
		for _, whitelistNet := range s.ipWhitelist {
			if whitelistNet.Contains(ip) {
				allowed = true
				break
			}
		}
		if !allowed {
			return true, "IP not in whitelist"
		}
	}

	return false, ""
}

// validateInput validates request input for security threats
func (s *SecurityMiddleware) validateInput(r *http.Request) (bool, string) {
	// Check content type
	contentType := r.Header.Get("Content-Type")
	if contentType != "" {
		allowed := false
		for _, allowedType := range s.config.AllowedContentTypes {
			if strings.HasPrefix(contentType, allowedType) {
				allowed = true
				break
			}
		}
		if !allowed {
			return false, "unsupported content type"
		}
	}

	// Check request size
	if r.ContentLength > s.config.MaxRequestSize {
		return false, "request too large"
	}

	// Check for SQL injection in URL parameters
	if s.config.SQLInjectionCheck {
		for _, values := range r.URL.Query() {
			for _, value := range values {
				if s.containsSQLInjection(value) {
					return false, "SQL injection detected in URL parameters"
				}
			}
		}
	}

	// Check for XSS in URL parameters
	if s.config.XSSProtection {
		for _, values := range r.URL.Query() {
			for _, value := range values {
				if s.containsXSS(value) {
					return false, "XSS attempt detected in URL parameters"
				}
			}
		}
	}

	// For POST/PUT requests, check body (simplified - in production you'd want to preserve the body)
	if r.Method == "POST" || r.Method == "PUT" {
		if err := r.ParseForm(); err == nil {
			for _, values := range r.PostForm {
				for _, value := range values {
					if s.config.SQLInjectionCheck && s.containsSQLInjection(value) {
						return false, "SQL injection detected in form data"
					}
					if s.config.XSSProtection && s.containsXSS(value) {
						return false, "XSS attempt detected in form data"
					}
				}
			}
		}
	}

	return true, ""
}

// isBot checks if the user agent indicates a bot
func (s *SecurityMiddleware) isBot(userAgent string) bool {
	if userAgent == "" {
		return true // Empty user agent is suspicious
	}

	userAgentLower := strings.ToLower(userAgent)

	// Check against bot patterns
	for _, pattern := range s.botPatterns {
		if pattern.MatchString(userAgentLower) {
			return true
		}
	}

	// Check user agent blacklist
	for _, blacklisted := range s.config.UserAgentBlacklist {
		if strings.Contains(userAgentLower, strings.ToLower(blacklisted)) {
			return true
		}
	}

	return false
}

// containsSQLInjection checks for SQL injection patterns
func (s *SecurityMiddleware) containsSQLInjection(input string) bool {
	inputLower := strings.ToLower(input)
	for _, pattern := range s.sqlInjectionPatterns {
		if pattern.MatchString(inputLower) {
			return true
		}
	}
	return false
}

// containsXSS checks for XSS patterns
func (s *SecurityMiddleware) containsXSS(input string) bool {
	inputLower := strings.ToLower(input)
	for _, pattern := range s.xssPatterns {
		if pattern.MatchString(inputLower) {
			return true
		}
	}
	return false
}

// setSecurityHeaders sets security-related HTTP headers
func (s *SecurityMiddleware) setSecurityHeaders(w http.ResponseWriter) {
	header := w.Header()

	// HSTS
	header.Set("Strict-Transport-Security", fmt.Sprintf("max-age=%d; includeSubDomains; preload", s.config.HSTSMaxAge))

	// CSP
	header.Set("Content-Security-Policy", s.config.ContentSecurityPolicy)

	// Other security headers
	header.Set("X-Content-Type-Options", "nosniff")
	header.Set("X-Frame-Options", "DENY")
	header.Set("X-XSS-Protection", "1; mode=block")
	header.Set("Referrer-Policy", s.config.ReferrerPolicy)
	header.Set("Permissions-Policy", "camera=(), microphone=(), geolocation=(), interest-cohort=()")

	// Remove server info
	header.Set("Server", "NovaCron")
}

// getClientIP extracts the real client IP from the request
func (s *SecurityMiddleware) getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// Take the first IP in the chain
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}

	// Check X-Real-IP header
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// Fall back to RemoteAddr
	ip, _, _ := net.SplitHostPort(r.RemoteAddr)
	return ip
}

// extractAuthToken extracts authentication token from the request
func (s *SecurityMiddleware) extractAuthToken(r *http.Request) string {
	// Check Authorization header
	authHeader := r.Header.Get("Authorization")
	if strings.HasPrefix(authHeader, "Bearer ") {
		return strings.TrimPrefix(authHeader, "Bearer ")
	}

	// Check cookie
	if cookie, err := r.Cookie("auth_token"); err == nil {
		return cookie.Value
	}

	// Check query parameter (less secure, but sometimes necessary)
	return r.URL.Query().Get("token")
}

// logSecurityEvent logs a security event
func (s *SecurityMiddleware) logSecurityEvent(event string, secCtx *SecurityContext, reason string, r *http.Request) {
	if !s.config.AuditLogging || s.auditService == nil {
		return
	}

	auditEntry := &AuditEntry{
		UserID:       secCtx.UserID,
		TenantID:     secCtx.TenantID,
		ResourceType: "api",
		ResourceID:   r.URL.Path,
		Action:       event,
		Success:      event == "request_completed",
		Reason:       reason,
		Timestamp:    time.Now(),
		IPAddress:    secCtx.ClientIP,
		UserAgent:    secCtx.UserAgent,
		AdditionalData: map[string]interface{}{
			"method":       r.Method,
			"url":          r.URL.String(),
			"threat_level": secCtx.ThreatLevel,
			"risk_factors": secCtx.RiskFactors,
		},
	}

	s.auditService.LogAccess(auditEntry)
}

// compileSQLInjectionPatterns compiles SQL injection detection patterns
func (s *SecurityMiddleware) compileSQLInjectionPatterns() {
	patterns := []string{
		`\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b`,
		`\b(or|and)\s+\d+\s*=\s*\d+`,
		`['"](\s|;|$)`,
		`--`,
		`/\*.*\*/`,
		`\bxp_cmdshell\b`,
		`\bsp_executesql\b`,
	}

	for _, pattern := range patterns {
		if re, err := regexp.Compile(pattern); err == nil {
			s.sqlInjectionPatterns = append(s.sqlInjectionPatterns, re)
		}
	}
}

// compileXSSPatterns compiles XSS detection patterns
func (s *SecurityMiddleware) compileXSSPatterns() {
	patterns := []string{
		`<script[^>]*>.*?</script>`,
		`javascript:`,
		`on(load|error|click|mouseover)\s*=`,
		`<iframe[^>]*>.*?</iframe>`,
		`<object[^>]*>.*?</object>`,
		`<embed[^>]*>`,
		`eval\s*\(`,
		`expression\s*\(`,
	}

	for _, pattern := range patterns {
		if re, err := regexp.Compile(pattern); err == nil {
			s.xssPatterns = append(s.xssPatterns, re)
		}
	}
}

// compileBotPatterns compiles bot detection patterns
func (s *SecurityMiddleware) compileBotPatterns() {
	patterns := []string{
		`bot|crawler|spider|scraper`,
		`curl|wget|python|java|go-http`,
		`headless|phantom|selenium`,
		`scanner|penetration|security`,
	}

	for _, pattern := range patterns {
		if re, err := regexp.Compile(pattern); err == nil {
			s.botPatterns = append(s.botPatterns, re)
		}
	}
}

// parseIPRanges parses IP whitelist and blacklist CIDR ranges
func (s *SecurityMiddleware) parseIPRanges() {
	// Parse whitelist
	for _, cidr := range s.config.IPWhitelist {
		if _, network, err := net.ParseCIDR(cidr); err == nil {
			s.ipWhitelist = append(s.ipWhitelist, network)
		}
	}

	// Parse blacklist
	for _, cidr := range s.config.IPBlacklist {
		if _, network, err := net.ParseCIDR(cidr); err == nil {
			s.ipBlacklist = append(s.ipBlacklist, network)
		}
	}
}

// CleanupRateLimits removes old rate limit entries
func (s *SecurityMiddleware) CleanupRateLimits() {
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()

	for key, entry := range s.rateLimits {
		// Remove entries that haven't been accessed for 2x the window period
		if now.Sub(entry.LastAccess) > s.config.RateLimitWindow*2 {
			delete(s.rateLimits, key)
		}
	}
}

// GetSecurityContext retrieves security context from request context
func GetSecurityContext(ctx context.Context) (*SecurityContext, bool) {
	if secCtx, ok := ctx.Value("security_context").(*SecurityContext); ok {
		return secCtx, true
	}
	return nil, false
}

// DefaultSecurityConfig returns secure default configuration
func DefaultSecurityConfig() SecurityConfig {
	return SecurityConfig{
		RateLimitEnabled:       true,
		RateLimitRequests:      1000,
		RateLimitWindow:        time.Hour,
		RateLimitBurstSize:     10,
		RateLimitByIP:          true,
		RateLimitByUser:        true,
		RateLimitByTenant:      true,
		InputValidationEnabled: true,
		MaxRequestSize:         10 * 1024 * 1024,
		AllowedContentTypes: []string{
			"application/json",
			"application/x-www-form-urlencoded",
			"multipart/form-data",
			"text/plain",
		},
		SQLInjectionCheck:      true,
		XSSProtection:          true,
		CSRFProtection:         true,
		SecurityHeadersEnabled: true,
		HSTSMaxAge:             31536000,
		ContentSecurityPolicy:  "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
		ReferrerPolicy:         "strict-origin-when-cross-origin",
		BotProtectionEnabled:   true,
		UserAgentBlacklist: []string{
			"sqlmap",
			"nikto",
			"nessus",
			"openvas",
			"nmap",
		},
		HoneypotEnabled:  true,
		AuditLogging:     true,
		LogSensitiveData: false,
	}
}
