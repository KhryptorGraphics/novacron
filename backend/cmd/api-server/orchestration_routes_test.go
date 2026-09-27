package main

import (
	"context"
	"database/sql"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
)

// Healing restarts/migrates real VMs and its targets are caller-chosen, so the
// orchestration API must be admin-only: a plain authenticated user must not be
// able to register another tenant's VM as a healing target or trigger healing.
func TestOrchestrationRoutesRequireAdmin(t *testing.T) {
	cfg := &config.Config{}
	cfg.VM.StoragePath = t.TempDir()

	// Unconnected handle: the orchestration handlers never touch the DB.
	db, err := sql.Open("postgres", "postgres://127.0.0.1:1/none?sslmode=disable")
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()

	services, err := initializeCanonicalServices(cfg, db, auth.NewSimpleAuthManager("test-secret", db))
	if err != nil {
		t.Fatalf("initializeCanonicalServices: %v", err)
	}
	defer services.shutdown()

	router := mux.NewRouter()
	apiRouter := router.PathPrefix("/api").Subrouter()
	// Stand-in for requireAuth: the role comes from the (already verified) token.
	apiRouter.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			role := r.Header.Get("X-Test-Role")
			ctx := context.WithValue(r.Context(), "role", role)
			ctx = context.WithValue(ctx, "roles", []string{role})
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	})
	registerOrchestrationRoutes(apiRouter, services.orchestrationAPI)

	do := func(role, method, path, body string) int {
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("X-Test-Role", role)
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec.Code
	}

	target := `{"id":"victim-vm","type":"vm","name":"victim-vm","enabled":true}`
	for _, role := range []string{"user", "viewer", "operator"} {
		if code := do(role, http.MethodPost, "/api/orchestration/healing/targets", target); code != http.StatusForbidden {
			t.Fatalf("%s create healing target: got %d, want 403", role, code)
		}
		if code := do(role, http.MethodPost, "/api/orchestration/healing/targets/victim-vm/heal", `{"reason":"x"}`); code != http.StatusForbidden {
			t.Fatalf("%s trigger healing: got %d, want 403", role, code)
		}
		if code := do(role, http.MethodGet, "/api/orchestration/healing/status", ""); code != http.StatusForbidden {
			t.Fatalf("%s healing status: got %d, want 403", role, code)
		}
	}

	if code := do("admin", http.MethodPost, "/api/orchestration/healing/targets", target); code != http.StatusCreated {
		t.Fatalf("admin create healing target: got %d, want 201", code)
	}
	if code := do("admin", http.MethodGet, "/api/orchestration/healing/targets/victim-vm", ""); code != http.StatusOK {
		t.Fatalf("admin get healing target: got %d, want 200", code)
	}
}
