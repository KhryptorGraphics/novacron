//go:build !experimental

package middleware

import (
	"context"
	"net/http"
	"strings"
)

// Core mode: allow all requests (no JWT enforcement) unless a simple header is present
// This avoids pulling in full auth core module for now
// Core mode: simple auth/RBAC without JWT dependency. Role is from X-Role header.

type AuthMiddleware struct{}

func NewAuthMiddleware(_ interface{}) *AuthMiddleware { return &AuthMiddleware{} }

func (m *AuthMiddleware) RequireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		role := strings.ToLower(r.Header.Get("X-Role"))
		if role == "" {
			role = "viewer"
		}
		ctx := context.WithValue(r.Context(), "role", role)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func ExtractUserFromContext(r *http.Request) (string, bool)   { return "", false }
func ExtractTenantFromContext(r *http.Request) (string, bool) { return "", false }
