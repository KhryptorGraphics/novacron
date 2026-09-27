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

// Authenticate is an alias for RequireAuth (core mode), matching the
// JWTMiddleware.Authenticate method name used in experimental mode.
func (m *AuthMiddleware) Authenticate(next http.Handler) http.Handler {
	return m.RequireAuth(next)
}

// Authenticate is a package-level convenience middleware used by API routers
// (e.g. backend/api/federation). Core mode: tag role from X-Role header.
func Authenticate(next http.Handler) http.Handler {
	m := &AuthMiddleware{}
	return m.RequireAuth(next)
}

func ExtractUserFromContext(r *http.Request) (string, bool)   { return "", false }
func ExtractTenantFromContext(r *http.Request) (string, bool) { return "", false }
