package middleware

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"net/http"
	"time"
)

// RequestContext holds context information for requests
type RequestContext struct {
	RequestID string
	TraceID   string
	StartTime time.Time
	UserID    string
	TenantID  string
}

// ContextKey is a custom type for context keys to avoid collisions
type ContextKey string

const (
	RequestIDKey  ContextKey = "request_id"
	TraceIDKey    ContextKey = "trace_id"
	StartTimeKey  ContextKey = "start_time"
	UserIDKey     ContextKey = "user_id"
	TenantIDKey   ContextKey = "tenant_id"
	RequestCtxKey ContextKey = "request_context"
)

// RequestID middleware generates a unique request ID for each request
func RequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if request ID is already present in headers
		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			// Generate a new request ID
			requestID = generateID("req")
		}
		
		// Check for trace ID
		traceID := r.Header.Get("X-Trace-ID")
		if traceID == "" {
			traceID = generateID("trace")
		}
		
		// Create request context
		reqCtx := &RequestContext{
			RequestID: requestID,
			TraceID:   traceID,
			StartTime: time.Now().UTC(),
		}
		
		// Add to context
		ctx := context.WithValue(r.Context(), RequestIDKey, requestID)
		ctx = context.WithValue(ctx, TraceIDKey, traceID)
		ctx = context.WithValue(ctx, StartTimeKey, reqCtx.StartTime)
		ctx = context.WithValue(ctx, RequestCtxKey, reqCtx)
		
		// Set response headers
		w.Header().Set("X-Request-ID", requestID)
		w.Header().Set("X-Trace-ID", traceID)
		
		// Continue with the request
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// UserContext middleware extracts user information from authenticated requests
func UserContext(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		
		// Extract user ID from context (set by auth middleware)
		userID := getUserIDFromContext(ctx)
		tenantID := getTenantIDFromContext(ctx)
		
		// Update request context
		if reqCtx := GetRequestContext(ctx); reqCtx != nil {
			reqCtx.UserID = userID
			reqCtx.TenantID = tenantID
			
			// Update context
			ctx = context.WithValue(ctx, UserIDKey, userID)
			ctx = context.WithValue(ctx, TenantIDKey, tenantID)
			ctx = context.WithValue(ctx, RequestCtxKey, reqCtx)
		}
		
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// Logging middleware logs request details
func Logging(logger Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Wrap response writer to capture status code
			wrapped := &responseWriter{ResponseWriter: w, statusCode: 200}
			
			// Get start time
			startTime := time.Now()
			
			// Process request
			next.ServeHTTP(wrapped, r)
			
			// Calculate duration
			duration := time.Since(startTime)
			
			// Get context information
			ctx := r.Context()
			requestID := GetRequestIDFromContext(ctx)
			userID := GetUserIDFromContext(ctx)
			
			// Log request details
			logger.Info("HTTP Request",
				"method", r.Method,
				"path", r.URL.Path,
				"status", wrapped.statusCode,
				"duration_ms", duration.Milliseconds(),
				"request_id", requestID,
				"user_id", userID,
				"remote_addr", r.RemoteAddr,
				"user_agent", r.UserAgent(),
			)
		})
	}
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// CORS middleware handles Cross-Origin Resource Sharing
func CORS(allowedOrigins, allowedMethods, allowedHeaders []string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")
			
			// Check if origin is allowed
			allowed := false
			for _, allowedOrigin := range allowedOrigins {
				if allowedOrigin == "*" || allowedOrigin == origin {
					allowed = true
					break
				}
			}
			
			if allowed {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			}
			
			// Set other CORS headers
			w.Header().Set("Access-Control-Allow-Methods", joinStrings(allowedMethods, ", "))
			w.Header().Set("Access-Control-Allow-Headers", joinStrings(allowedHeaders, ", "))
			w.Header().Set("Access-Control-Max-Age", "86400") // 24 hours
			
			// Handle preflight requests
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	}
}

// Helper functions for context extraction

// GetRequestContext extracts the full request context
func GetRequestContext(ctx context.Context) *RequestContext {
	if reqCtx, ok := ctx.Value(RequestCtxKey).(*RequestContext); ok {
		return reqCtx
	}
	return nil
}

// GetRequestIDFromContext extracts request ID from context
func GetRequestIDFromContext(ctx context.Context) string {
	if reqID, ok := ctx.Value(RequestIDKey).(string); ok {
		return reqID
	}
	return ""
}

// GetTraceIDFromContext extracts trace ID from context
func GetTraceIDFromContext(ctx context.Context) string {
	if traceID, ok := ctx.Value(TraceIDKey).(string); ok {
		return traceID
	}
	return ""
}

// GetUserIDFromContext extracts user ID from context
func GetUserIDFromContext(ctx context.Context) string {
	if userID, ok := ctx.Value(UserIDKey).(string); ok {
		return userID
	}
	return ""
}

// GetTenantIDFromContext extracts tenant ID from context
func GetTenantIDFromContext(ctx context.Context) string {
	if tenantID, ok := ctx.Value(TenantIDKey).(string); ok {
		return tenantID
	}
	return ""
}

// generateID generates a random ID with a prefix
func generateID(prefix string) string {
	bytes := make([]byte, 16)
	if _, err := rand.Read(bytes); err != nil {
		// Fallback to timestamp if random generation fails
		return prefix + "_" + hex.EncodeToString([]byte(time.Now().String()))[:16]
	}
	return prefix + "_" + hex.EncodeToString(bytes)[:16]
}

// getUserIDFromContext extracts user ID from authentication context
func getUserIDFromContext(ctx context.Context) string {
	// This would typically be set by an authentication middleware
	// For now, we'll check for common patterns
	if userID, ok := ctx.Value("user_id").(string); ok {
		return userID
	}
	if userID, ok := ctx.Value("uid").(string); ok {
		return userID
	}
	return ""
}

// getTenantIDFromContext extracts tenant ID from authentication context
func getTenantIDFromContext(ctx context.Context) string {
	// This would typically be set by an authentication middleware
	if tenantID, ok := ctx.Value("tenant_id").(string); ok {
		return tenantID
	}
	if tenantID, ok := ctx.Value("tid").(string); ok {
		return tenantID
	}
	return ""
}

// joinStrings joins a slice of strings with a separator
func joinStrings(slice []string, separator string) string {
	if len(slice) == 0 {
		return ""
	}
	
	result := slice[0]
	for i := 1; i < len(slice); i++ {
		result += separator + slice[i]
	}
	return result
}

// Logger interface for structured logging
type Logger interface {
	Info(msg string, keysAndValues ...interface{})
	Error(msg string, keysAndValues ...interface{})
	Debug(msg string, keysAndValues ...interface{})
	Warn(msg string, keysAndValues ...interface{})
}