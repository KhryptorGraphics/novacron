//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/gorilla/mux"
)

// newIncomingMigrationTestRouter mirrors how buildCanonicalServer wires the
// route in production: registerInternalMigrationRoutes on a fresh mux.Router,
// no JWT middleware in front (this is the deliberately node-to-node,
// pre-authentication route -- the shared secret IS the auth). db and
// vmManager are nil: the auth check must run and reject/allow BEFORE either
// is touched, so a nil vmManager surfacing as 503 (not a panic, not 403) is
// itself proof the request passed the auth gate.
func newIncomingMigrationTestRouter() *mux.Router {
	router := mux.NewRouter()
	registerInternalMigrationRoutes(router, nil, nil, "")
	return router
}

func postIncomingMigration(t *testing.T, router *mux.Router, secretHeader string) *httptest.ResponseRecorder {
	t.Helper()
	body := strings.NewReader(`{"vm_id":"vm-1","disk_path":"/tmp/x.qcow2"}`)
	req := httptest.NewRequest(http.MethodPost, "/internal/migrate/incoming", body)
	req.Header.Set("Content-Type", "application/json")
	if secretHeader != "" {
		req.Header.Set("X-Migration-Secret", secretHeader)
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

// TestInternalMigrationIncomingFailsClosedWithoutSecret is THE discriminator
// for novacron-el8. Before the fix, the handler only checked the header when
// `secret != ""`; with no secret configured it fell straight through to the
// vmManager-nil branch, so an attacker with no credentials at all could reach
// the handler (503 with a nil vmManager here, live qemu launch in prod). The
// fix must fail closed: no configured secret means every request is
// forbidden, full stop. Revert `!migrationAuthOK(r)` back to the old
// `secret != "" && header != secret` skip and this test starts seeing 503
// instead of 403, failing.
func TestInternalMigrationIncomingFailsClosedWithoutSecret(t *testing.T) {
	os.Unsetenv("NOVACRON_MIGRATION_SECRET")

	rec := postIncomingMigration(t, newIncomingMigrationTestRouter(), "")
	if rec.Code != http.StatusForbidden {
		t.Fatalf("no secret configured: got HTTP %d, want 403 (fail closed, not pass-through)", rec.Code)
	}

	// Also confirm an attacker can't just supply *some* header value and slip
	// through when no secret is configured server-side.
	rec2 := postIncomingMigration(t, newIncomingMigrationTestRouter(), "anything-attacker-likes")
	if rec2.Code != http.StatusForbidden {
		t.Fatalf("no secret configured, attacker header present: got HTTP %d, want 403", rec2.Code)
	}
}

// TestInternalMigrationIncomingRejectsBadHeader covers the configured-secret
// side: missing header and wrong header must both be forbidden.
func TestInternalMigrationIncomingRejectsBadHeader(t *testing.T) {
	const secret = "unit-test-migration-secret-value"
	os.Setenv("NOVACRON_MIGRATION_SECRET", secret)
	defer os.Unsetenv("NOVACRON_MIGRATION_SECRET")

	if rec := postIncomingMigration(t, newIncomingMigrationTestRouter(), ""); rec.Code != http.StatusForbidden {
		t.Fatalf("secret configured, missing header: got HTTP %d, want 403", rec.Code)
	}
	if rec := postIncomingMigration(t, newIncomingMigrationTestRouter(), secret+"-wrong"); rec.Code != http.StatusForbidden {
		t.Fatalf("secret configured, wrong header: got HTTP %d, want 403", rec.Code)
	}
}

// TestInternalMigrationIncomingAllowsCorrectSecret proves the correct header
// actually clears the auth gate: with vmManager nil, the handler's very next
// check (vmManager == nil) returns 503, which is only reachable if
// migrationAuthOK returned true. A 403 here would mean the fix over-rejects
// valid, correctly-authenticated peers.
func TestInternalMigrationIncomingAllowsCorrectSecret(t *testing.T) {
	const secret = "unit-test-migration-secret-value"
	os.Setenv("NOVACRON_MIGRATION_SECRET", secret)
	defer os.Unsetenv("NOVACRON_MIGRATION_SECRET")

	rec := postIncomingMigration(t, newIncomingMigrationTestRouter(), secret)
	if rec.Code == http.StatusForbidden {
		t.Fatalf("secret configured, correct header: got 403, auth should have passed")
	}
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("secret configured, correct header: got HTTP %d, want 503 (vm manager unavailable -- proves the request reached past the auth gate)", rec.Code)
	}
}
