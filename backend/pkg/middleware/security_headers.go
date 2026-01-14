package middleware

import (
	"net/http"
	"strconv"
	"strings"
)

// SecurityHeadersConfig defines the configuration for security headers
type SecurityHeadersConfig struct {
	// ContentSecurityPolicy defines the CSP header value
	ContentSecurityPolicy string
	// XFrameOptions defines whether the page can be framed (DENY, SAMEORIGIN, or ALLOW-FROM uri)
	XFrameOptions string
	// XContentTypeOptions prevents MIME type sniffing
	XContentTypeOptions string
	// XXSSProtection enables XSS filtering
	XXSSProtection string
	// StrictTransportSecurity enables HSTS
	StrictTransportSecurity string
	// ReferrerPolicy controls the Referer header
	ReferrerPolicy string
	// PermissionsPolicy defines feature policies
	PermissionsPolicy string
	// CacheControl for preventing caching of sensitive data
	CacheControl string
}

// DefaultSecurityHeadersConfig returns a secure default configuration
func DefaultSecurityHeadersConfig() SecurityHeadersConfig {
	return SecurityHeadersConfig{
		// Content Security Policy - restricts resource loading
		ContentSecurityPolicy: "default-src 'self'; " +
			"script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
			"style-src 'self' 'unsafe-inline'; " +
			"img-src 'self' data: https:; " +
			"font-src 'self' data:; " +
			"connect-src 'self' ws: wss:; " +
			"frame-ancestors 'self'; " +
			"base-uri 'self'; " +
			"form-action 'self'",
		// Prevent clickjacking
		XFrameOptions: "SAMEORIGIN",
		// Prevent MIME type sniffing
		XContentTypeOptions: "nosniff",
		// Enable XSS filter (legacy but still useful)
		XXSSProtection: "1; mode=block",
		// Enable HSTS with 1 year max-age
		StrictTransportSecurity: "max-age=31536000; includeSubDomains",
		// Control referrer information
		ReferrerPolicy: "strict-origin-when-cross-origin",
		// Restrict browser features
		PermissionsPolicy: "geolocation=(), microphone=(), camera=()",
		// Prevent caching of sensitive pages
		CacheControl: "no-store, no-cache, must-revalidate, proxy-revalidate",
	}
}

// SecurityHeaders adds security headers to all responses
type SecurityHeaders struct {
	config SecurityHeadersConfig
}

// NewSecurityHeaders creates a new security headers middleware
func NewSecurityHeaders(config SecurityHeadersConfig) *SecurityHeaders {
	return &SecurityHeaders{config: config}
}

// Handler applies security headers to responses
func (sh *SecurityHeaders) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Content Security Policy
		if sh.config.ContentSecurityPolicy != "" {
			w.Header().Set("Content-Security-Policy", sh.config.ContentSecurityPolicy)
		}

		// X-Frame-Options (clickjacking protection)
		if sh.config.XFrameOptions != "" {
			w.Header().Set("X-Frame-Options", sh.config.XFrameOptions)
		}

		// X-Content-Type-Options (MIME sniffing protection)
		if sh.config.XContentTypeOptions != "" {
			w.Header().Set("X-Content-Type-Options", sh.config.XContentTypeOptions)
		}

		// X-XSS-Protection (XSS filter)
		if sh.config.XXSSProtection != "" {
			w.Header().Set("X-XSS-Protection", sh.config.XXSSProtection)
		}

		// Strict-Transport-Security (HSTS)
		if sh.config.StrictTransportSecurity != "" {
			w.Header().Set("Strict-Transport-Security", sh.config.StrictTransportSecurity)
		}

		// Referrer-Policy
		if sh.config.ReferrerPolicy != "" {
			w.Header().Set("Referrer-Policy", sh.config.ReferrerPolicy)
		}

		// Permissions-Policy (formerly Feature-Policy)
		if sh.config.PermissionsPolicy != "" {
			w.Header().Set("Permissions-Policy", sh.config.PermissionsPolicy)
		}

		// Cache-Control for API responses
		if sh.config.CacheControl != "" && strings.HasPrefix(r.URL.Path, "/api/") {
			w.Header().Set("Cache-Control", sh.config.CacheControl)
			w.Header().Set("Pragma", "no-cache")
			w.Header().Set("Expires", "0")
		}

		next.ServeHTTP(w, r)
	})
}

// CORSConfig defines CORS configuration
type CORSConfig struct {
	// AllowedOrigins is a list of allowed origins (can include "*" for any)
	AllowedOrigins []string
	// AllowedMethods is a list of allowed HTTP methods
	AllowedMethods []string
	// AllowedHeaders is a list of allowed request headers
	AllowedHeaders []string
	// ExposedHeaders is a list of headers clients can access
	ExposedHeaders []string
	// AllowCredentials indicates if cookies should be allowed
	AllowCredentials bool
	// MaxAge indicates how long preflight results can be cached
	MaxAge int
}

// DefaultCORSConfig returns a secure default CORS configuration
func DefaultCORSConfig() CORSConfig {
	return CORSConfig{
		AllowedOrigins: []string{"http://localhost:8092", "http://localhost:3000"},
		AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"},
		AllowedHeaders: []string{
			"Authorization",
			"Content-Type",
			"X-Requested-With",
			"X-Request-ID",
			"X-Trace-ID",
		},
		ExposedHeaders: []string{
			"X-Request-ID",
			"X-Trace-ID",
			"X-RateLimit-Limit",
			"X-RateLimit-Remaining",
			"X-RateLimit-Reset",
		},
		AllowCredentials: true,
		MaxAge:           86400, // 24 hours
	}
}

// ProductionCORSConfig returns a CORS configuration for production
func ProductionCORSConfig(frontendOrigin string) CORSConfig {
	return CORSConfig{
		AllowedOrigins:   []string{frontendOrigin},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"},
		AllowedHeaders:   DefaultCORSConfig().AllowedHeaders,
		ExposedHeaders:   DefaultCORSConfig().ExposedHeaders,
		AllowCredentials: true,
		MaxAge:           86400,
	}
}

// CORSHandler is an enhanced CORS middleware with more options
type CORSHandler struct {
	config CORSConfig
}

// NewCORSHandler creates a new enhanced CORS middleware
func NewCORSHandler(config CORSConfig) *CORSHandler {
	return &CORSHandler{config: config}
}

// Handler applies CORS headers to responses
func (c *CORSHandler) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")

		// Check if origin is allowed
		allowed := false
		for _, allowedOrigin := range c.config.AllowedOrigins {
			if allowedOrigin == "*" || allowedOrigin == origin {
				allowed = true
				break
			}
		}

		if allowed && origin != "" {
			w.Header().Set("Access-Control-Allow-Origin", origin)

			if c.config.AllowCredentials {
				w.Header().Set("Access-Control-Allow-Credentials", "true")
			}

			// Expose headers
			if len(c.config.ExposedHeaders) > 0 {
				w.Header().Set("Access-Control-Expose-Headers", strings.Join(c.config.ExposedHeaders, ", "))
			}
		}

		// Handle preflight requests
		if r.Method == http.MethodOptions {
			if allowed {
				w.Header().Set("Access-Control-Allow-Methods", strings.Join(c.config.AllowedMethods, ", "))
				w.Header().Set("Access-Control-Allow-Headers", strings.Join(c.config.AllowedHeaders, ", "))

				if c.config.MaxAge > 0 {
					w.Header().Set("Access-Control-Max-Age", strconv.Itoa(c.config.MaxAge))
				}
			}

			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// Vary ensures proper caching of CORS responses
func (c *CORSHandler) Vary(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Vary", "Origin")
		next.ServeHTTP(w, r)
	})
}

// CombinedSecurityMiddleware chains security headers and CORS
func CombinedSecurityMiddleware(securityConfig SecurityHeadersConfig, corsConfig CORSConfig) func(http.Handler) http.Handler {
	security := NewSecurityHeaders(securityConfig)
	cors := NewCORSHandler(corsConfig)

	return func(next http.Handler) http.Handler {
		return security.Handler(cors.Handler(next))
	}
}

// DefaultSecurityMiddleware returns a middleware with secure defaults
func DefaultSecurityMiddleware() func(http.Handler) http.Handler {
	return CombinedSecurityMiddleware(DefaultSecurityHeadersConfig(), DefaultCORSConfig())
}
