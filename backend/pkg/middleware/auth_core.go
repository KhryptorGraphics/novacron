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


type AuthMiddleware struct { }

func NewAuthMiddleware(_ interface{}) *AuthMiddleware { return &AuthMiddleware{} }

func (m *AuthMiddleware) RequireAuth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request){
		role := strings.ToLower(r.Header.Get("X-Role"))
		if role == "" { role = "viewer" }
		ctx := context.WithValue(r.Context(), "role", role)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

func (m *AuthMiddleware) RequireRole(role string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request){
			userRole, _ := r.Context().Value("role").(string)
			if userRole == "" { userRole = "viewer" }
			if !roleAllowed(userRole, role) {
				w.Header().Set("Content-Type", "application/json; charset=utf-8")
				w.WriteHeader(http.StatusForbidden)
				_, _ = w.Write([]byte(`{"error":{"code":"forbidden","message":"Insufficient permissions"}}`))
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

func roleAllowed(userRole, required string) bool {
	order := map[string]int{"viewer":1, "operator":2, "admin":3}
	return order[strings.ToLower(userRole)] >= order[strings.ToLower(required)]
}

func ExtractUserFromContext(r *http.Request) (string, bool) { return "", false }
func ExtractTenantFromContext(r *http.Request) (string, bool) { return "", false }

