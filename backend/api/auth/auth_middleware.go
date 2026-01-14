package auth

import (
	"context"
	"net/http"
	"strings"
)

// AuthMiddleware handles authentication for protected routes
func AuthMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Get authorization header
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" {
			http.Error(w, "Authorization header required", http.StatusUnauthorized)
			return
		}

		// Check if it's a Bearer token
		if !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "Invalid authorization header format", http.StatusUnauthorized)
			return
		}

		// Extract token
		token := strings.TrimPrefix(authHeader, "Bearer ")

		// In a real implementation, you would validate the token here
		// For now, we'll just pass through with a mock session
		ctx := context.WithValue(r.Context(), "token", token)
		ctx = context.WithValue(ctx, "sessionID", "session-123")
		ctx = context.WithValue(ctx, "userID", "user-123")

		// Call next handler with updated context
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}