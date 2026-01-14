package middleware

import (
	"context"
	"crypto/rsa"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/golang-jwt/jwt/v5"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// JWT-specific context keys (reuses ContextKey type from request.go)
const (
	SessionIDKey   ContextKey = "session_id"
	RolesKey       ContextKey = "roles"
	PermissionsKey ContextKey = "permissions"
	TokenClaimsKey ContextKey = "token_claims"
	JTIKey         ContextKey = "jti"
)

// JWTMiddlewareConfig configures the JWT authentication middleware
type JWTMiddlewareConfig struct {
	// RSAPublicKey for RS256 verification
	RSAPublicKey *rsa.PublicKey
	// HMACSecret for HS256 verification (fallback)
	HMACSecret []byte
	// TokenRevocation service for checking blacklisted tokens
	TokenRevocation auth.TokenRevocationService
	// RequireValidToken if true, rejects requests without valid tokens (default: true)
	RequireValidToken bool
	// SkipPaths are paths that don't require authentication
	SkipPaths []string
	// Issuer to validate (optional)
	Issuer string
	// Audience to validate (optional)
	Audience string
}

// JWTMiddleware handles JWT authentication for API endpoints
type JWTMiddleware struct {
	config JWTMiddlewareConfig
}

// NewJWTMiddleware creates a new JWT authentication middleware
func NewJWTMiddleware(config JWTMiddlewareConfig) *JWTMiddleware {
	return &JWTMiddleware{config: config}
}

// Authenticate is middleware that validates JWT tokens
func (m *JWTMiddleware) Authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if path should be skipped
		for _, path := range m.config.SkipPaths {
			if strings.HasPrefix(r.URL.Path, path) {
				next.ServeHTTP(w, r)
				return
			}
		}

		// Extract JWT token from Authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			if m.config.RequireValidToken {
				writeJSONError(w, http.StatusUnauthorized, "authorization_required", "Authorization header required")
				return
			}
			next.ServeHTTP(w, r)
			return
		}

		// Check if the header starts with "Bearer "
		if !strings.HasPrefix(authHeader, "Bearer ") {
			writeJSONError(w, http.StatusUnauthorized, "invalid_header", "Authorization header must start with 'Bearer '")
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")

		// Parse and validate the JWT token
		token, err := jwt.ParseWithClaims(tokenString, &auth.JWTClaims{}, m.keyFunc)
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "invalid_token", "Invalid token: "+err.Error())
			return
		}

		if !token.Valid {
			writeJSONError(w, http.StatusUnauthorized, "token_invalid", "Token is not valid")
			return
		}

		// Extract claims
		claims, ok := token.Claims.(*auth.JWTClaims)
		if !ok {
			writeJSONError(w, http.StatusUnauthorized, "invalid_claims", "Invalid token claims")
			return
		}

		// Check token revocation
		if m.config.TokenRevocation != nil && claims.ID != "" {
			revoked, err := m.config.TokenRevocation.IsRevoked(claims.ID)
			if err != nil {
				writeJSONError(w, http.StatusInternalServerError, "revocation_check_failed", "Failed to check token revocation")
				return
			}
			if revoked {
				writeJSONError(w, http.StatusUnauthorized, "token_revoked", "Token has been revoked")
				return
			}
		}

		// Check user-wide revocation
		if revocationSvc, ok := m.config.TokenRevocation.(*auth.RedisTokenRevocation); ok {
			if claims.UserID != "" && claims.IssuedAt != nil {
				revoked, err := revocationSvc.IsUserTokenRevoked(claims.UserID, claims.IssuedAt.Time)
				if err == nil && revoked {
					writeJSONError(w, http.StatusUnauthorized, "user_tokens_revoked", "All user tokens have been revoked")
					return
				}
			}
		}

		// Validate issuer if configured
		if m.config.Issuer != "" {
			issuer, _ := claims.GetIssuer()
			if issuer != m.config.Issuer {
				writeJSONError(w, http.StatusUnauthorized, "invalid_issuer", "Invalid token issuer")
				return
			}
		}

		// Validate audience if configured
		if m.config.Audience != "" {
			aud, _ := claims.GetAudience()
			validAudience := false
			for _, a := range aud {
				if a == m.config.Audience {
					validAudience = true
					break
				}
			}
			if !validAudience {
				writeJSONError(w, http.StatusUnauthorized, "invalid_audience", "Invalid token audience")
				return
			}
		}

		// Add claims to request context
		ctx := r.Context()
		ctx = context.WithValue(ctx, UserIDKey, claims.UserID)
		ctx = context.WithValue(ctx, TenantIDKey, claims.TenantID)
		ctx = context.WithValue(ctx, SessionIDKey, claims.SessionID)
		ctx = context.WithValue(ctx, RolesKey, claims.Roles)
		ctx = context.WithValue(ctx, PermissionsKey, claims.Permissions)
		ctx = context.WithValue(ctx, TokenClaimsKey, claims)
		ctx = context.WithValue(ctx, JTIKey, claims.ID)

		// Continue to next handler
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// keyFunc returns the key for JWT verification
func (m *JWTMiddleware) keyFunc(token *jwt.Token) (interface{}, error) {
	switch token.Method.(type) {
	case *jwt.SigningMethodRSA:
		if m.config.RSAPublicKey == nil {
			return nil, jwt.ErrTokenSignatureInvalid
		}
		return m.config.RSAPublicKey, nil
	case *jwt.SigningMethodHMAC:
		if len(m.config.HMACSecret) == 0 {
			return nil, jwt.ErrTokenSignatureInvalid
		}
		return m.config.HMACSecret, nil
	default:
		return nil, jwt.ErrTokenSignatureInvalid
	}
}

// RequireRole creates middleware that requires a specific role
func (m *JWTMiddleware) RequireRole(role string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			roles, ok := r.Context().Value(RolesKey).([]string)
			if !ok {
				writeJSONError(w, http.StatusForbidden, "no_roles", "No roles found in token")
				return
			}

			hasRole := false
			for _, userRole := range roles {
				if userRole == role || userRole == "admin" {
					hasRole = true
					break
				}
			}

			if !hasRole {
				writeJSONError(w, http.StatusForbidden, "insufficient_permissions", "Role '"+role+"' required")
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// RequirePermission creates middleware that requires a specific permission
func (m *JWTMiddleware) RequirePermission(permission string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			permissions, ok := r.Context().Value(PermissionsKey).([]string)
			if !ok {
				writeJSONError(w, http.StatusForbidden, "no_permissions", "No permissions found in token")
				return
			}

			hasPermission := false
			for _, perm := range permissions {
				if perm == permission || perm == "*" {
					hasPermission = true
					break
				}
			}

			if !hasPermission {
				writeJSONError(w, http.StatusForbidden, "insufficient_permissions", "Permission '"+permission+"' required")
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// RequireAnyRole creates middleware that requires at least one of the specified roles
func (m *JWTMiddleware) RequireAnyRole(roles ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			userRoles, ok := r.Context().Value(RolesKey).([]string)
			if !ok {
				writeJSONError(w, http.StatusForbidden, "no_roles", "No roles found in token")
				return
			}

			hasRole := false
			for _, userRole := range userRoles {
				if userRole == "admin" {
					hasRole = true
					break
				}
				for _, required := range roles {
					if userRole == required {
						hasRole = true
						break
					}
				}
				if hasRole {
					break
				}
			}

			if !hasRole {
				writeJSONError(w, http.StatusForbidden, "insufficient_permissions", "One of roles required: "+strings.Join(roles, ", "))
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// GetUserID extracts user ID from request context
func GetUserID(r *http.Request) (string, bool) {
	userID, ok := r.Context().Value(UserIDKey).(string)
	return userID, ok
}

// GetTenantID extracts tenant ID from request context
func GetTenantID(r *http.Request) (string, bool) {
	tenantID, ok := r.Context().Value(TenantIDKey).(string)
	return tenantID, ok
}

// GetSessionID extracts session ID from request context
func GetSessionID(r *http.Request) (string, bool) {
	sessionID, ok := r.Context().Value(SessionIDKey).(string)
	return sessionID, ok
}

// GetRoles extracts roles from request context
func GetRoles(r *http.Request) ([]string, bool) {
	roles, ok := r.Context().Value(RolesKey).([]string)
	return roles, ok
}

// GetPermissions extracts permissions from request context
func GetPermissions(r *http.Request) ([]string, bool) {
	permissions, ok := r.Context().Value(PermissionsKey).([]string)
	return permissions, ok
}

// GetClaims extracts all JWT claims from request context
func GetClaims(r *http.Request) (*auth.JWTClaims, bool) {
	claims, ok := r.Context().Value(TokenClaimsKey).(*auth.JWTClaims)
	return claims, ok
}

// writeJSONError writes a JSON error response
func writeJSONError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	response := map[string]interface{}{
		"error": map[string]interface{}{
			"code":    code,
			"message": message,
		},
	}
	_ = json.NewEncoder(w).Encode(response)
}
