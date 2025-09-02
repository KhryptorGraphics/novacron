//go:build experimental

package middleware


import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// AuthMiddleware handles JWT authentication for API endpoints
type AuthMiddleware struct {
	authManager interface {
		GetJWTSecret() string
		GetUser(userID string) (*auth.User, error)
	}
	jwtSecret   string
}

// NewAuthMiddleware creates a new authentication middleware
func NewAuthMiddleware(authManager interface {
	GetJWTSecret() string
	GetUser(userID string) (*auth.User, error)
}) *AuthMiddleware {
	return &AuthMiddleware{
		authManager: authManager,
		jwtSecret:   authManager.GetJWTSecret(),
	}
}

// RequireAuth is middleware that requires valid JWT authentication
func (m *AuthMiddleware) RequireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Extract JWT token from Authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, "Authorization header required", http.StatusUnauthorized)
			return
		}

		// Check if the header starts with "Bearer "
		if !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "Authorization header must start with 'Bearer '", http.StatusUnauthorized)
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")

		// Parse and validate the JWT token
		token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			// Verify signing method
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, jwt.ErrSignatureInvalid
			}
			return []byte(m.jwtSecret), nil
		})

		if err != nil {
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		if !token.Valid {
			http.Error(w, "Token is not valid", http.StatusUnauthorized)
			return
		}

		// Extract claims
		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok {
			http.Error(w, "Invalid token claims", http.StatusUnauthorized)
			return
		}

		// Check token expiration
		if exp, ok := claims["exp"].(float64); ok {
			if time.Now().Unix() > int64(exp) {
				http.Error(w, "Token has expired", http.StatusUnauthorized)
				return
			}
		}

		// Extract user information from claims
		userID, ok := claims["user_id"].(string)
		if !ok {
			http.Error(w, "Invalid token: missing user_id", http.StatusUnauthorized)
			return
		}

		tenantID, _ := claims["tenant_id"].(string)
		if tenantID == "" {
			tenantID = "default"
		}

		// Add user info to request context
		ctx := context.WithValue(r.Context(), "user_id", userID)
		ctx = context.WithValue(ctx, "tenant_id", tenantID)
		ctx = context.WithValue(ctx, "token_claims", claims)

		// Continue to next handler
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// RequireRole is middleware that requires a specific role
func (m *AuthMiddleware) RequireRole(role string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Check if user is authenticated first
			userID := r.Context().Value("user_id")
			if userID == nil {
				http.Error(w, "Authentication required", http.StatusUnauthorized)
				return
			}

			// Get user from database and check role
			user, err := m.authManager.GetUser(userID.(string))
			if err != nil {
				http.Error(w, "User not found", http.StatusUnauthorized)
				return
			}

			hasRole := false
			for _, userRole := range user.Roles {
				if userRole.Name == role {
					hasRole = true
					break
				}
			}

			if !hasRole {
				http.Error(w, "Insufficient permissions", http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// ExtractUserFromContext extracts user ID from request context
func ExtractUserFromContext(r *http.Request) (string, bool) {
	userID, ok := r.Context().Value("user_id").(string)
	return userID, ok
}

// ExtractTenantFromContext extracts tenant ID from request context
func ExtractTenantFromContext(r *http.Request) (string, bool) {
	tenantID, ok := r.Context().Value("tenant_id").(string)
	return tenantID, ok
}