package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/pquerna/otp/totp"
	"github.com/sirupsen/logrus"

	graphqlapi "github.com/khryptorgraphics/novacron/backend/api/graphql"
	securityapi "github.com/khryptorgraphics/novacron/backend/api/security"
	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"golang.org/x/crypto/bcrypt"
)

// newStubVMManager builds a real VMManager whose KVM driver falls back to the
// CoreStubDriver (bogus qemu_path + NOVACRON_ALLOW_STUB_KVM), so VM lifecycle
// ops exercise the manager/driver path without spawning qemu.
func newStubVMManager(t *testing.T) *core_vm.VMManager {
	t.Helper()
	t.Setenv("NOVACRON_ALLOW_STUB_KVM", "1")

	vmCfg := core_vm.DefaultVMManagerConfig()
	vmCfg.Drivers[core_vm.VMTypeKVM] = core_vm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"qemu_path": "missing-qemu-for-stub-test",
			"vm_path":   t.TempDir(),
		},
	}
	m, err := core_vm.NewVMManager(vmCfg)
	if err != nil {
		t.Fatalf("new stub vm manager: %v", err)
	}
	return m
}

// seedManagerVM creates a VM directly through the manager so handler power
// routes can act on a known id.
func seedManagerVM(t *testing.T, m *core_vm.VMManager, id string) {
	t.Helper()
	if _, err := m.CreateVM(context.Background(), core_vm.CreateVMRequest{
		Name:                  id,
		AllowMissingOwnership: true,
		Spec:                  core_vm.VMConfig{ID: id, Name: id, Type: core_vm.VMTypeKVM, TenantID: "default"},
	}); err != nil {
		t.Fatalf("seed manager vm %s: %v", id, err)
	}
}

// TestCanonicalVMRoutesDriveManagerState proves the VM handlers call the
// injected manager/driver and persist the ACTUAL resulting runtime state — not
// a hardcoded constant. The final case (stop on an unknown id) proves the old
// fake "stopped" write is gone: it errors and touches no rows.
func TestCanonicalVMRoutesDriveManagerState(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()

	router := mux.NewRouter()
	registerSecureAPIRoutes(router, db, manager, t.TempDir())

	// Create: manager.CreateVM(stub) succeeds, then a single INSERT with the
	// manager-derived state (the manager reports "stopped" for a freshly
	// created, not-yet-started VM), never a hardcoded "creating". cpu_cores is
	// the REAL vCPU count now: a request without vcpus persists 1 (was: the
	// 1024 CPUShares scheduling weight).
	mock.ExpectExec("INSERT INTO vms").
		WithArgs(sqlmock.AnyArg(), "vm-a", "stopped", 1, 0, 0, sqlmock.AnyArg(), "", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, mustJSONRequest(t, http.MethodPost, "/vms", map[string]interface{}{"name": "vm-a"}))
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create: expected 201, got %d (%s)", createRec.Code, createRec.Body.String())
	}
	var created map[string]interface{}
	decodeJSONBody(t, createRec, &created)
	vmID, _ := created["id"].(string)
	if vmID == "" {
		t.Fatalf("create: missing id, got %#v", created)
	}
	if created["state"] != "stopped" {
		t.Fatalf("create: expected manager-derived state 'stopped', got %#v", created["state"])
	}

	// Start: manager sets running; handler must persist "running".
	mock.ExpectExec("UPDATE vms SET state").
		WithArgs(vmID, "running").
		WillReturnResult(sqlmock.NewResult(0, 1))

	startRec := httptest.NewRecorder()
	router.ServeHTTP(startRec, mustJSONRequest(t, http.MethodPost, "/vms/"+vmID+"/start", nil))
	if startRec.Code != http.StatusOK {
		t.Fatalf("start: expected 200, got %d (%s)", startRec.Code, startRec.Body.String())
	}
	var started map[string]interface{}
	decodeJSONBody(t, startRec, &started)
	if started["state"] != "running" {
		t.Fatalf("start: expected state 'running' from manager, got %#v", started["state"])
	}

	// Stop: manager sets stopped; handler must persist "stopped".
	mock.ExpectExec("UPDATE vms SET state").
		WithArgs(vmID, "stopped").
		WillReturnResult(sqlmock.NewResult(0, 1))

	stopRec := httptest.NewRecorder()
	router.ServeHTTP(stopRec, mustJSONRequest(t, http.MethodPost, "/vms/"+vmID+"/stop", nil))
	if stopRec.Code != http.StatusOK {
		t.Fatalf("stop: expected 200, got %d (%s)", stopRec.Code, stopRec.Body.String())
	}
	var stopped map[string]interface{}
	decodeJSONBody(t, stopRec, &stopped)
	if stopped["state"] != "stopped" {
		t.Fatalf("stop: expected state 'stopped' from manager, got %#v", stopped["state"])
	}

	// Stop an id the manager does not know: must error WITHOUT any DB write.
	// No sqlmock expectation is registered — a stray UPDATE would fail the run.
	unknownRec := httptest.NewRecorder()
	router.ServeHTTP(unknownRec, mustJSONRequest(t, http.MethodPost, "/vms/does-not-exist/stop", nil))
	if unknownRec.Code != http.StatusNotFound {
		t.Fatalf("stop-unknown: expected 404 (no fake write), got %d (%s)", unknownRec.Code, unknownRec.Body.String())
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func newCanonicalSecurityRouter(t *testing.T, authManager *auth.SimpleAuthManager, handlers *securityapi.SecurityHandlers) *mux.Router {
	t.Helper()

	router := mux.NewRouter()
	registerCanonicalSecurityRoutes(router, authManager, handlers)
	return router
}

func mustJSONRequest(t *testing.T, method, path string, payload interface{}) *http.Request {
	t.Helper()

	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}

	req := httptest.NewRequest(method, path, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	return req
}

func decodeJSONBody(t *testing.T, rec *httptest.ResponseRecorder, target interface{}) {
	t.Helper()
	if err := json.NewDecoder(rec.Body).Decode(target); err != nil {
		t.Fatalf("decode response: %v", err)
	}
}

// TestCanonicalPasswordResetRoutesFailClosedWithoutEmail proves the token
// routes refuse with 503 before touching any database when SMTP is not
// configured (emailService == nil) — the honest replacement for the old fake
// "success without doing anything" handlers.
func TestCanonicalPasswordResetRoutesFailClosedWithoutEmail(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)

	router := mux.NewRouter()
	registerPublicRoutes(router, authManager, nil, nil, nil)

	tests := []struct {
		name       string
		path       string
		payload    map[string]string
		wantStatus int
		wantErr    string
	}{
		{
			name:       "forgot password",
			path:       "/api/auth/forgot-password",
			payload:    map[string]string{"email": "user@example.com"},
			wantStatus: http.StatusServiceUnavailable,
			wantErr:    "email delivery is not configured (set SMTP_HOST)",
		},
		{
			name:       "resend verification",
			path:       "/api/auth/resend-verification",
			payload:    map[string]string{"email": "user@example.com"},
			wantStatus: http.StatusServiceUnavailable,
			wantErr:    "email delivery is not configured (set SMTP_HOST)",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := mustJSONRequest(t, http.MethodPost, tc.path, tc.payload)
			rec := httptest.NewRecorder()

			router.ServeHTTP(rec, req)

			if rec.Code != tc.wantStatus {
				t.Fatalf("expected %d, got %d (%s)", tc.wantStatus, rec.Code, rec.Body.String())
			}

			var payload map[string]interface{}
			decodeJSONBody(t, rec, &payload)
			if payload["error"] != tc.wantErr {
				t.Fatalf("expected error %q, got %#v", tc.wantErr, payload["error"])
			}
		})
	}
}

func TestCanonicalTwoFactorLoginFlow(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", db)
	twoFactorService := auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret()))
	handlers := securityapi.NewSecurityHandlers(twoFactorService, audit.NewSimpleAuditLogger())

	router := mux.NewRouter()
	registerPublicRoutes(router, authManager, db, twoFactorService, nil)
	registerCanonicalSecurityRoutes(router, authManager, handlers)

	setupReq := mustJSONRequest(t, http.MethodPost, "/api/auth/2fa/setup", map[string]interface{}{
		"account_name": "user@example.com",
	})
	setupReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	setupRec := httptest.NewRecorder()
	router.ServeHTTP(setupRec, setupReq)

	if setupRec.Code != http.StatusOK {
		t.Fatalf("expected setup 200, got %d (%s)", setupRec.Code, setupRec.Body.String())
	}

	var setupPayload struct {
		Secret string `json:"secret"`
	}
	decodeJSONBody(t, setupRec, &setupPayload)
	if setupPayload.Secret == "" {
		t.Fatal("expected setup secret")
	}

	enableCode, err := totp.GenerateCode(setupPayload.Secret, time.Now().UTC())
	if err != nil {
		t.Fatalf("generate enable code: %v", err)
	}

	enableReq := mustJSONRequest(t, http.MethodPost, "/api/auth/2fa/enable", map[string]interface{}{
		"code": enableCode,
	})
	enableReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	enableRec := httptest.NewRecorder()
	router.ServeHTTP(enableRec, enableReq)
	if enableRec.Code != http.StatusOK {
		t.Fatalf("expected enable 200, got %d (%s)", enableRec.Code, enableRec.Body.String())
	}

	passwordHash, err := bcrypt.GenerateFromPassword([]byte("correct-horse-battery-staple"), bcrypt.DefaultCost)
	if err != nil {
		t.Fatalf("hash password: %v", err)
	}

	now := time.Now().UTC()
	mock.ExpectQuery(regexp.QuoteMeta(`SELECT username FROM users WHERE email = $1`)).
		WithArgs("user@example.com").
		WillReturnRows(sqlmock.NewRows([]string{"username"}).AddRow("user"))
	mock.ExpectQuery(regexp.QuoteMeta(`
		SELECT id, username, email, password_hash, role, status, created_at, updated_at
		FROM users WHERE username = $1
	`)).
		WithArgs("user").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "password_hash", "role", "status", "created_at", "updated_at"}).
			AddRow("7", "user", "user@example.com", string(passwordHash), "admin", "active", now, now))

	loginReq := mustJSONRequest(t, http.MethodPost, "/api/auth/login", map[string]interface{}{
		"email":    "user@example.com",
		"password": "correct-horse-battery-staple",
	})
	loginRec := httptest.NewRecorder()
	router.ServeHTTP(loginRec, loginReq)

	if loginRec.Code != http.StatusOK {
		t.Fatalf("expected login 200, got %d (%s)", loginRec.Code, loginRec.Body.String())
	}

	var loginPayload struct {
		RequiresTwoFactor bool   `json:"requires_2fa"`
		TempToken         string `json:"temp_token"`
	}
	decodeJSONBody(t, loginRec, &loginPayload)
	if !loginPayload.RequiresTwoFactor {
		t.Fatalf("expected requires_2fa=true, got %#v", loginPayload)
	}
	if loginPayload.TempToken == "" {
		t.Fatal("expected temp_token in 2FA login challenge")
	}

	mock.ExpectQuery(regexp.QuoteMeta(`
		SELECT id, username, email, password_hash, role, status, created_at, updated_at
		FROM users WHERE id = $1
	`)).
		WithArgs("7").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "password_hash", "role", "status", "created_at", "updated_at"}).
			AddRow("7", "user", "user@example.com", string(passwordHash), "admin", "active", now, now))

	verifyCode, err := totp.GenerateCode(setupPayload.Secret, time.Now().UTC())
	if err != nil {
		t.Fatalf("generate verify code: %v", err)
	}

	verifyReq := mustJSONRequest(t, http.MethodPost, "/api/auth/2fa/verify-login", map[string]interface{}{
		"code":       verifyCode,
		"temp_token": loginPayload.TempToken,
	})
	verifyRec := httptest.NewRecorder()
	router.ServeHTTP(verifyRec, verifyReq)

	if verifyRec.Code != http.StatusOK {
		t.Fatalf("expected verify-login 200, got %d (%s)", verifyRec.Code, verifyRec.Body.String())
	}

	var verifyPayload struct {
		Token string `json:"token"`
		User  struct {
			ID string `json:"id"`
		} `json:"user"`
	}
	decodeJSONBody(t, verifyRec, &verifyPayload)
	if verifyPayload.Token == "" {
		t.Fatal("expected session token after successful 2FA verification")
	}
	if verifyPayload.User.ID != "7" {
		t.Fatalf("expected verified user 7, got %q", verifyPayload.User.ID)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestCanonicalTwoFactorVerifyLoginRejectsInvalidOrMismatchedTempToken(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	twoFactorService := auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret()))

	router := mux.NewRouter()
	registerPublicRoutes(router, authManager, nil, twoFactorService, nil)

	invalidTokenReq := mustJSONRequest(t, http.MethodPost, "/api/auth/2fa/verify-login", map[string]interface{}{
		"code":       "123456",
		"temp_token": "not-a-valid-token",
	})
	invalidTokenRec := httptest.NewRecorder()
	router.ServeHTTP(invalidTokenRec, invalidTokenReq)

	if invalidTokenRec.Code != http.StatusUnauthorized {
		t.Fatalf("expected invalid temp token to return 401, got %d (%s)", invalidTokenRec.Code, invalidTokenRec.Body.String())
	}

	pendingToken, err := issuePending2FAToken(authManager.GetJWTSecret(), &auth.User{
		ID:       "7",
		Email:    "user@example.com",
		Username: "user",
		RoleIDs:  []string{"admin"},
		TenantID: "default",
	})
	if err != nil {
		t.Fatalf("issue pending token: %v", err)
	}

	mismatchedUserReq := mustJSONRequest(t, http.MethodPost, "/api/auth/2fa/verify-login", map[string]interface{}{
		"user_id":    "someone-else",
		"code":       "123456",
		"temp_token": pendingToken,
	})
	mismatchedUserRec := httptest.NewRecorder()
	router.ServeHTTP(mismatchedUserRec, mismatchedUserReq)

	if mismatchedUserRec.Code != http.StatusUnauthorized {
		t.Fatalf("expected mismatched user to return 401, got %d (%s)", mismatchedUserRec.Code, mismatchedUserRec.Body.String())
	}
}

func TestCanonicalTwoFactorRoutesUseAuthenticatedPrincipal(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	twoFactorService := auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret()))
	handlers := securityapi.NewSecurityHandlers(twoFactorService, audit.NewSimpleAuditLogger())

	router := mux.NewRouter()
	registerCanonicalSecurityRoutes(router, authManager, handlers)

	setupReq := mustJSONRequest(t, http.MethodPost, "/api/auth/2fa/setup", map[string]interface{}{
		"user_id":      "attacker-selected-user",
		"account_name": "user@example.com",
	})
	setupReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	setupRec := httptest.NewRecorder()
	router.ServeHTTP(setupRec, setupReq)
	if setupRec.Code != http.StatusOK {
		t.Fatalf("expected setup 200, got %d (%s)", setupRec.Code, setupRec.Body.String())
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/api/auth/2fa/status?user_id=someone-else", nil)
	statusReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	statusRec := httptest.NewRecorder()
	router.ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d (%s)", statusRec.Code, statusRec.Body.String())
	}

	var statusPayload map[string]interface{}
	decodeJSONBody(t, statusRec, &statusPayload)
	if setup, _ := statusPayload["setup"].(bool); !setup {
		t.Fatalf("expected authenticated user's 2FA setup to be returned, got %#v", statusPayload)
	}
}

func TestCanonicalSecurityRoutesRunScansAndSurfaceThreats(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	handlers := securityapi.NewSecurityHandlers(auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret())), audit.NewSimpleAuditLogger())
	router := newCanonicalSecurityRouter(t, authManager, handlers)

	targetDir := t.TempDir()
	sensitiveFile := filepath.Join(targetDir, ".env")
	if err := os.WriteFile(sensitiveFile, []byte("API_TOKEN=secret"), 0o600); err != nil {
		t.Fatalf("write scan target: %v", err)
	}

	scanReq := mustJSONRequest(t, http.MethodPost, "/api/security/scan", map[string]interface{}{
		"targets":    []string{targetDir},
		"scan_types": []string{"secrets", "filesystem"},
	})
	scanReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	scanRec := httptest.NewRecorder()
	router.ServeHTTP(scanRec, scanReq)
	if scanRec.Code != http.StatusAccepted {
		t.Fatalf("expected scan start 202, got %d (%s)", scanRec.Code, scanRec.Body.String())
	}

	var scanStart map[string]interface{}
	decodeJSONBody(t, scanRec, &scanStart)
	scanID, _ := scanStart["scan_id"].(string)
	if scanID == "" {
		t.Fatalf("expected scan_id, got %#v", scanStart)
	}

	var scanStatus map[string]interface{}
	for attempt := 0; attempt < 50; attempt++ {
		statusReq := httptest.NewRequest(http.MethodGet, "/api/security/scan/"+scanID, nil)
		statusReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
		statusRec := httptest.NewRecorder()
		router.ServeHTTP(statusRec, statusReq)
		if statusRec.Code != http.StatusOK {
			t.Fatalf("expected scan status 200, got %d (%s)", statusRec.Code, statusRec.Body.String())
		}

		decodeJSONBody(t, statusRec, &scanStatus)
		if scanStatus["status"] == "completed" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if scanStatus["status"] != "completed" {
		t.Fatalf("expected completed scan, got %#v", scanStatus)
	}

	threatReq := httptest.NewRequest(http.MethodGet, "/api/security/threats", nil)
	threatReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	threatRec := httptest.NewRecorder()
	router.ServeHTTP(threatRec, threatReq)
	if threatRec.Code != http.StatusOK {
		t.Fatalf("expected threats 200, got %d (%s)", threatRec.Code, threatRec.Body.String())
	}

	var threatPayload map[string]interface{}
	decodeJSONBody(t, threatRec, &threatPayload)
	threats, ok := threatPayload["threats"].([]interface{})
	if !ok || len(threats) == 0 {
		t.Fatalf("expected at least one surfaced threat, got %#v", threatPayload["threats"])
	}

	vulnReq := httptest.NewRequest(http.MethodGet, "/api/security/vulnerabilities", nil)
	vulnReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	vulnRec := httptest.NewRecorder()
	router.ServeHTTP(vulnRec, vulnReq)
	if vulnRec.Code != http.StatusOK {
		t.Fatalf("expected vulnerabilities 200, got %d (%s)", vulnRec.Code, vulnRec.Body.String())
	}

	var vulnPayload map[string]interface{}
	decodeJSONBody(t, vulnRec, &vulnPayload)
	summary, ok := vulnPayload["summary"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected vulnerability summary, got %#v", vulnPayload["summary"])
	}
	if highCount, _ := summary["high"].(float64); highCount < 1 {
		t.Fatalf("expected at least one high severity finding, got %#v", summary)
	}

	complianceReq := httptest.NewRequest(http.MethodGet, "/api/security/compliance", nil)
	complianceReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	complianceRec := httptest.NewRecorder()
	router.ServeHTTP(complianceRec, complianceReq)
	if complianceRec.Code != http.StatusOK {
		t.Fatalf("expected compliance 200, got %d (%s)", complianceRec.Code, complianceRec.Body.String())
	}

	var compliancePayload map[string]interface{}
	decodeJSONBody(t, complianceRec, &compliancePayload)
	if score, _ := compliancePayload["compliance_score"].(float64); score >= 100 {
		t.Fatalf("expected compliance score to reflect scan findings, got %#v", compliancePayload["compliance_score"])
	}

	incidentReq := httptest.NewRequest(http.MethodGet, "/api/security/incidents", nil)
	incidentReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	incidentRec := httptest.NewRecorder()
	router.ServeHTTP(incidentRec, incidentReq)
	if incidentRec.Code != http.StatusOK {
		t.Fatalf("expected incidents 200, got %d (%s)", incidentRec.Code, incidentRec.Body.String())
	}

	var incidentPayload map[string]interface{}
	decodeJSONBody(t, incidentRec, &incidentPayload)
	if total, _ := incidentPayload["total"].(float64); total < 1 {
		t.Fatalf("expected at least one incident, got %#v", incidentPayload)
	}
}

func TestCanonicalSecurityRoutesSupportReleaseAdminMutations(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	auditLogger := audit.NewSimpleAuditLogger()
	handlers := securityapi.NewSecurityHandlers(
		auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret())),
		auditLogger,
	)
	router := newCanonicalSecurityRouter(t, authManager, handlers)
	authz := signedBearerToken(t, authManager, "7", "default", "admin")

	seedEvent := &audit.AuditEvent{
		ID:        "audit-seed",
		Timestamp: time.Now().Add(-time.Minute).UTC(),
		EventType: audit.EventPermissionDeny,
		Actor:     "seed@example.com",
		UserID:    "7",
		Resource:  "admin_panel",
		Action:    audit.ActionRead,
		Result:    audit.ResultDenied,
		Details: map[string]interface{}{
			"description": "Seed security event",
		},
	}
	if err := auditLogger.LogEvent(context.Background(), seedEvent); err != nil {
		t.Fatalf("seed audit event: %v", err)
	}

	ackReq := mustJSONRequest(t, http.MethodPost, "/api/security/events/audit-seed/acknowledge", map[string]interface{}{
		"note": "triaged",
	})
	ackReq.Header.Set("Authorization", authz)
	ackRec := httptest.NewRecorder()
	router.ServeHTTP(ackRec, ackReq)
	if ackRec.Code != http.StatusOK {
		t.Fatalf("expected acknowledge 200, got %d (%s)", ackRec.Code, ackRec.Body.String())
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/api/security/events", nil)
	eventsReq.Header.Set("Authorization", authz)
	eventsRec := httptest.NewRecorder()
	router.ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("expected events 200, got %d (%s)", eventsRec.Code, eventsRec.Body.String())
	}

	var eventsPayload map[string]interface{}
	decodeJSONBody(t, eventsRec, &eventsPayload)
	events, ok := eventsPayload["events"].([]interface{})
	if !ok || len(events) == 0 {
		t.Fatalf("expected events payload, got %#v", eventsPayload["events"])
	}

	foundAcknowledged := false
	for _, rawEvent := range events {
		event, ok := rawEvent.(map[string]interface{})
		if !ok {
			continue
		}
		if event["id"] == "audit-seed" {
			foundAcknowledged = event["acknowledged"] == true
		}
	}
	if !foundAcknowledged {
		t.Fatalf("expected seeded event to be acknowledged, got %#v", events)
	}

	complianceReq := mustJSONRequest(t, http.MethodPost, "/api/security/compliance/check", map[string]interface{}{
		"requirement_id": "overall-security-posture",
	})
	complianceReq.Header.Set("Authorization", authz)
	complianceRec := httptest.NewRecorder()
	router.ServeHTTP(complianceRec, complianceReq)
	if complianceRec.Code != http.StatusAccepted {
		t.Fatalf("expected compliance check 202, got %d (%s)", complianceRec.Code, complianceRec.Body.String())
	}

	var compliancePayload map[string]interface{}
	decodeJSONBody(t, complianceRec, &compliancePayload)
	if _, ok := compliancePayload["jobId"].(string); !ok {
		t.Fatalf("expected compliance check jobId, got %#v", compliancePayload)
	}

	exportReq := httptest.NewRequest(http.MethodGet, "/api/security/compliance/export?format=csv", nil)
	exportReq.Header.Set("Authorization", authz)
	exportRec := httptest.NewRecorder()
	router.ServeHTTP(exportRec, exportReq)
	if exportRec.Code != http.StatusOK {
		t.Fatalf("expected compliance export 200, got %d (%s)", exportRec.Code, exportRec.Body.String())
	}
	if contentType := exportRec.Header().Get("Content-Type"); !strings.Contains(contentType, "text/csv") {
		t.Fatalf("expected csv export, got %q", contentType)
	}

	incidentReq := mustJSONRequest(t, http.MethodPost, "/api/security/incidents", map[string]interface{}{
		"title":           "Manual investigation",
		"description":     "Operator escalated a suspicious login pattern.",
		"severity":        "high",
		"type":            "manual",
		"affectedSystems": []string{"auth-gateway"},
	})
	incidentReq.Header.Set("Authorization", authz)
	incidentRec := httptest.NewRecorder()
	router.ServeHTTP(incidentRec, incidentReq)
	if incidentRec.Code != http.StatusCreated {
		t.Fatalf("expected incident create 201, got %d (%s)", incidentRec.Code, incidentRec.Body.String())
	}

	incidentsReq := httptest.NewRequest(http.MethodGet, "/api/security/incidents", nil)
	incidentsReq.Header.Set("Authorization", authz)
	incidentsRec := httptest.NewRecorder()
	router.ServeHTTP(incidentsRec, incidentsReq)
	if incidentsRec.Code != http.StatusOK {
		t.Fatalf("expected incidents 200, got %d (%s)", incidentsRec.Code, incidentsRec.Body.String())
	}

	var incidentsPayload map[string]interface{}
	decodeJSONBody(t, incidentsRec, &incidentsPayload)
	incidents, ok := incidentsPayload["incidents"].([]interface{})
	if !ok || len(incidents) == 0 {
		t.Fatalf("expected incidents payload, got %#v", incidentsPayload["incidents"])
	}

	foundManualIncident := false
	for _, rawIncident := range incidents {
		incident, ok := rawIncident.(map[string]interface{})
		if !ok {
			continue
		}
		if incident["title"] == "Manual investigation" {
			foundManualIncident = true
		}
	}
	if !foundManualIncident {
		t.Fatalf("expected manual incident to be surfaced, got %#v", incidents)
	}
}

func TestCanonicalSecurityRoutesRejectNonAdminUsers(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	handlers := securityapi.NewSecurityHandlers(auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret())), audit.NewSimpleAuditLogger())
	router := newCanonicalSecurityRouter(t, authManager, handlers)

	for _, endpoint := range []struct {
		method string
		path   string
		body   interface{}
	}{
		{method: http.MethodGet, path: "/api/security/compliance"},
		{method: http.MethodPost, path: "/api/security/rbac/user/42/roles", body: map[string]interface{}{"roles": []string{"admin"}}},
	} {
		var req *http.Request
		if endpoint.body == nil {
			req = httptest.NewRequest(endpoint.method, endpoint.path, nil)
		} else {
			req = mustJSONRequest(t, endpoint.method, endpoint.path, endpoint.body)
		}
		req.Header.Set("Authorization", signedBearerToken(t, authManager, "11", "default", "user"))

		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusForbidden {
			t.Fatalf("expected %s %s to return 403 for user role, got %d (%s)", endpoint.method, endpoint.path, rec.Code, rec.Body.String())
		}
	}
}

func TestCanonicalSecurityRoutesRejectPendingTwoFactorTokens(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	handlers := securityapi.NewSecurityHandlers(auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret())), audit.NewSimpleAuditLogger())
	router := newCanonicalSecurityRouter(t, authManager, handlers)

	tempToken, err := issuePending2FAToken(authManager.GetJWTSecret(), &auth.User{
		ID:       "7",
		Email:    "user@example.com",
		Username: "user",
		RoleIDs:  []string{"admin"},
		TenantID: "default",
	})
	if err != nil {
		t.Fatalf("issue temp token: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/security/compliance", nil)
	req.Header.Set("Authorization", "Bearer "+tempToken)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected pending 2FA token to return 401, got %d (%s)", rec.Code, rec.Body.String())
	}
}

func TestCanonicalGraphQLRouteServesStorageBackedVolumeOperations(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	volumeStore, err := storage.NewStorageManager(storage.StorageManagerConfig{
		BasePath: filepath.Join(t.TempDir(), "volumes"),
	})
	if err != nil {
		t.Fatalf("create volume store: %v", err)
	}

	resolver := graphqlapi.NewResolverWithVolumeStore(nil, nil, volumeStore)
	handler := graphqlapi.NewVolumeHTTPHandler(resolver)

	router := mux.NewRouter()
	registerCanonicalGraphQLRoute(router, authManager, handler)

	createReq := mustJSONRequest(t, http.MethodPost, "/graphql", map[string]interface{}{
		"query": `mutation CreateVolume($input: CreateVolumeInput!) { createVolume(input: $input) { id name size tier } }`,
		"variables": map[string]interface{}{
			"input": map[string]interface{}{
				"name": "fast-disk",
				"size": 10,
				"tier": "hot",
			},
		},
	})
	createReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusOK {
		t.Fatalf("expected createVolume 200, got %d (%s)", createRec.Code, createRec.Body.String())
	}

	var createPayload struct {
		Data struct {
			CreateVolume struct {
				ID   string `json:"id"`
				Name string `json:"name"`
				Tier string `json:"tier"`
			} `json:"createVolume"`
		} `json:"data"`
	}
	decodeJSONBody(t, createRec, &createPayload)
	if createPayload.Data.CreateVolume.ID == "" {
		t.Fatal("expected created volume id")
	}
	if !strings.EqualFold(createPayload.Data.CreateVolume.Tier, "hot") {
		t.Fatalf("expected created tier hot, got %q", createPayload.Data.CreateVolume.Tier)
	}

	listReq := mustJSONRequest(t, http.MethodPost, "/graphql", map[string]interface{}{
		"query":     `query ListVolumes { volumes { id name tier } }`,
		"variables": map[string]interface{}{},
	})
	listReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	listRec := httptest.NewRecorder()
	router.ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("expected volumes 200, got %d (%s)", listRec.Code, listRec.Body.String())
	}

	var listPayload struct {
		Data struct {
			Volumes []struct {
				ID   string `json:"id"`
				Name string `json:"name"`
				Tier string `json:"tier"`
			} `json:"volumes"`
		} `json:"data"`
	}
	decodeJSONBody(t, listRec, &listPayload)
	if len(listPayload.Data.Volumes) != 1 {
		t.Fatalf("expected 1 volume, got %d", len(listPayload.Data.Volumes))
	}

	changeReq := mustJSONRequest(t, http.MethodPost, "/graphql", map[string]interface{}{
		"query": `mutation ChangeVolumeTier($id: ID!, $tier: String!) { changeVolumeTier(id: $id, tier: $tier) { id tier } }`,
		"variables": map[string]interface{}{
			"id":   createPayload.Data.CreateVolume.ID,
			"tier": "cold",
		},
	})
	changeReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	changeRec := httptest.NewRecorder()
	router.ServeHTTP(changeRec, changeReq)
	if changeRec.Code != http.StatusOK {
		t.Fatalf("expected changeVolumeTier 200, got %d (%s)", changeRec.Code, changeRec.Body.String())
	}

	var changePayload struct {
		Data struct {
			ChangeVolumeTier struct {
				ID   string `json:"id"`
				Tier string `json:"tier"`
			} `json:"changeVolumeTier"`
		} `json:"data"`
	}
	decodeJSONBody(t, changeRec, &changePayload)
	if !strings.EqualFold(changePayload.Data.ChangeVolumeTier.Tier, "cold") {
		t.Fatalf("expected changed tier cold, got %q", changePayload.Data.ChangeVolumeTier.Tier)
	}
}

func TestCanonicalLiveServerSmoke(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", db)
	twoFactorService := auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret()))
	securityHandlers := securityapi.NewSecurityHandlers(twoFactorService, audit.NewSimpleAuditLogger())

	volumeStore, err := storage.NewStorageManager(storage.StorageManagerConfig{
		BasePath: filepath.Join(t.TempDir(), "volumes"),
	})
	if err != nil {
		t.Fatalf("create volume store: %v", err)
	}

	resolver := graphqlapi.NewResolverWithVolumeStore(nil, nil, volumeStore)
	graphqlHandler := graphqlapi.NewVolumeHTTPHandler(resolver)
	wsHandler := websocketapi.NewWebSocketHandler(nil, nil, nil, nil, logrus.New())
	defer wsHandler.Shutdown()

	router := mux.NewRouter()
	registerPublicRoutes(router, authManager, db, twoFactorService, nil)
	registerCanonicalSecurityRoutes(router, authManager, securityHandlers)
	registerCanonicalGraphQLRoute(router, authManager, graphqlHandler)
	wsHandler.RegisterWebSocketRoutes(router, func(required string, next http.HandlerFunc) http.Handler {
		return requireAuth(authManager)(requireRoleHandler(required, next))
	})
	registerSecurityWebSocketAliases(router, authManager, securityHandlers)

	server := httptest.NewServer(router)
	defer server.Close()

	passwordHash, err := bcrypt.GenerateFromPassword([]byte("correct-horse-battery-staple"), bcrypt.DefaultCost)
	if err != nil {
		t.Fatalf("hash password: %v", err)
	}

	now := time.Now().UTC()
	mock.ExpectQuery(regexp.QuoteMeta(`SELECT username FROM users WHERE email = $1`)).
		WithArgs("admin@example.com").
		WillReturnRows(sqlmock.NewRows([]string{"username"}).AddRow("admin"))
	mock.ExpectQuery(regexp.QuoteMeta(`
		SELECT id, username, email, password_hash, role, status, created_at, updated_at
		FROM users WHERE username = $1
	`)).
		WithArgs("admin").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "password_hash", "role", "status", "created_at", "updated_at"}).
			AddRow("7", "admin", "admin@example.com", string(passwordHash), "admin", "active", now, now))

	loginReq, err := http.NewRequest(http.MethodPost, server.URL+"/api/auth/login", strings.NewReader(`{"email":"admin@example.com","password":"correct-horse-battery-staple"}`))
	if err != nil {
		t.Fatalf("build login request: %v", err)
	}
	loginReq.Header.Set("Content-Type", "application/json")

	loginResp, err := http.DefaultClient.Do(loginReq)
	if err != nil {
		t.Fatalf("login request failed: %v", err)
	}
	defer loginResp.Body.Close()

	if loginResp.StatusCode != http.StatusOK {
		t.Fatalf("expected login 200, got %d", loginResp.StatusCode)
	}

	var loginPayload struct {
		Token string `json:"token"`
	}
	if err := json.NewDecoder(loginResp.Body).Decode(&loginPayload); err != nil {
		t.Fatalf("decode login response: %v", err)
	}
	if loginPayload.Token == "" {
		t.Fatal("expected login token from canonical server")
	}

	complianceReq, err := http.NewRequest(http.MethodGet, server.URL+"/api/security/compliance", nil)
	if err != nil {
		t.Fatalf("build compliance request: %v", err)
	}
	complianceReq.Header.Set("Authorization", "Bearer "+loginPayload.Token)

	complianceResp, err := http.DefaultClient.Do(complianceReq)
	if err != nil {
		t.Fatalf("compliance request failed: %v", err)
	}
	defer complianceResp.Body.Close()

	if complianceResp.StatusCode != http.StatusOK {
		t.Fatalf("expected compliance 200, got %d", complianceResp.StatusCode)
	}

	var compliancePayload map[string]interface{}
	if err := json.NewDecoder(complianceResp.Body).Decode(&compliancePayload); err != nil {
		t.Fatalf("decode compliance response: %v", err)
	}
	if _, ok := compliancePayload["compliance_score"]; !ok {
		t.Fatalf("expected compliance_score in response, got %#v", compliancePayload)
	}

	wsHeaders := http.Header{}
	wsHeaders.Set("Authorization", "Bearer "+loginPayload.Token)
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/api/ws/security/events"

	wsConn, _, err := websocket.DefaultDialer.Dial(wsURL, wsHeaders)
	if err != nil {
		t.Fatalf("dial security websocket: %v", err)
	}
	_ = wsConn.Close()

	createReq := mustJSONRequest(t, http.MethodPost, "/graphql", map[string]interface{}{
		"query": `mutation CreateVolume($input: CreateVolumeInput!) { createVolume(input: $input) { id name size tier } }`,
		"variables": map[string]interface{}{
			"input": map[string]interface{}{
				"name": "smoke-disk",
				"size": 5,
				"tier": "hot",
			},
		},
	})
	createReq.URL.Scheme = "http"
	createReq.URL.Host = strings.TrimPrefix(server.URL, "http://")
	createReq.RequestURI = ""
	createReq.Header.Set("Authorization", "Bearer "+loginPayload.Token)

	createResp, err := http.DefaultClient.Do(createReq)
	if err != nil {
		t.Fatalf("createVolume request failed: %v", err)
	}
	defer createResp.Body.Close()

	if createResp.StatusCode != http.StatusOK {
		t.Fatalf("expected createVolume 200, got %d", createResp.StatusCode)
	}

	var createPayload struct {
		Data struct {
			CreateVolume struct {
				ID   string `json:"id"`
				Name string `json:"name"`
				Tier string `json:"tier"`
			} `json:"createVolume"`
		} `json:"data"`
	}
	if err := json.NewDecoder(createResp.Body).Decode(&createPayload); err != nil {
		t.Fatalf("decode createVolume response: %v", err)
	}
	if createPayload.Data.CreateVolume.ID == "" {
		t.Fatal("expected created volume id")
	}

	listReq := mustJSONRequest(t, http.MethodPost, "/graphql", map[string]interface{}{
		"query":     `query ListVolumes { volumes { id name tier } }`,
		"variables": map[string]interface{}{},
	})
	listReq.URL.Scheme = "http"
	listReq.URL.Host = strings.TrimPrefix(server.URL, "http://")
	listReq.RequestURI = ""
	listReq.Header.Set("Authorization", "Bearer "+loginPayload.Token)

	listResp, err := http.DefaultClient.Do(listReq)
	if err != nil {
		t.Fatalf("list volumes request failed: %v", err)
	}
	defer listResp.Body.Close()

	if listResp.StatusCode != http.StatusOK {
		t.Fatalf("expected volumes 200, got %d", listResp.StatusCode)
	}

	var listPayload struct {
		Data struct {
			Volumes []struct {
				ID   string `json:"id"`
				Name string `json:"name"`
				Tier string `json:"tier"`
			} `json:"volumes"`
		} `json:"data"`
	}
	if err := json.NewDecoder(listResp.Body).Decode(&listPayload); err != nil {
		t.Fatalf("decode volumes response: %v", err)
	}
	if len(listPayload.Data.Volumes) != 1 || listPayload.Data.Volumes[0].ID != createPayload.Data.CreateVolume.ID {
		t.Fatalf("expected created volume in list response, got %#v", listPayload.Data.Volumes)
	}

	changeReq := mustJSONRequest(t, http.MethodPost, "/graphql", map[string]interface{}{
		"query": `mutation ChangeVolumeTier($id: ID!, $tier: String!) { changeVolumeTier(id: $id, tier: $tier) { id tier } }`,
		"variables": map[string]interface{}{
			"id":   createPayload.Data.CreateVolume.ID,
			"tier": "cold",
		},
	})
	changeReq.URL.Scheme = "http"
	changeReq.URL.Host = strings.TrimPrefix(server.URL, "http://")
	changeReq.RequestURI = ""
	changeReq.Header.Set("Authorization", "Bearer "+loginPayload.Token)

	changeResp, err := http.DefaultClient.Do(changeReq)
	if err != nil {
		t.Fatalf("changeVolumeTier request failed: %v", err)
	}
	defer changeResp.Body.Close()

	if changeResp.StatusCode != http.StatusOK {
		t.Fatalf("expected changeVolumeTier 200, got %d", changeResp.StatusCode)
	}

	var changePayload struct {
		Data struct {
			ChangeVolumeTier struct {
				ID   string `json:"id"`
				Tier string `json:"tier"`
			} `json:"changeVolumeTier"`
		} `json:"data"`
	}
	if err := json.NewDecoder(changeResp.Body).Decode(&changePayload); err != nil {
		t.Fatalf("decode changeVolumeTier response: %v", err)
	}
	if !strings.EqualFold(changePayload.Data.ChangeVolumeTier.Tier, "cold") {
		t.Fatalf("expected changed tier cold, got %q", changePayload.Data.ChangeVolumeTier.Tier)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestCanonicalAndCompatibilityWebSocketMetricsRoutes(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	wsHandler := websocketapi.NewWebSocketHandler(nil, nil, nil, nil, logrus.New())
	defer wsHandler.Shutdown()

	router := mux.NewRouter()
	wsHandler.RegisterWebSocketRoutes(router, func(required string, next http.HandlerFunc) http.Handler {
		return requireAuth(authManager)(requireRoleHandler(required, next))
	})

	server := httptest.NewServer(router)
	defer server.Close()

	for _, route := range []string{"/api/ws/metrics?interval=1", "/ws/metrics?interval=1"} {
		headers := http.Header{}
		headers.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
		wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + route

		conn, _, err := websocket.DefaultDialer.Dial(wsURL, headers)
		if err != nil {
			t.Fatalf("dial %s: %v", route, err)
		}

		conn.SetReadDeadline(time.Now().Add(3 * time.Second))
		_, message, err := conn.ReadMessage()
		if err != nil {
			conn.Close()
			t.Fatalf("read %s: %v", route, err)
		}

		var payload map[string]interface{}
		if err := json.Unmarshal(message, &payload); err != nil {
			conn.Close()
			t.Fatalf("decode %s message: %v", route, err)
		}
		if payload["type"] != "metrics_update" {
			conn.Close()
			t.Fatalf("expected metrics_update on %s, got %#v", route, payload["type"])
		}

		_ = conn.Close()
	}
}

func TestCanonicalAndCompatibilityWebSocketMetricsRoutesRejectUserRole(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	wsHandler := websocketapi.NewWebSocketHandler(nil, nil, nil, nil, logrus.New())
	defer wsHandler.Shutdown()

	router := mux.NewRouter()
	wsHandler.RegisterWebSocketRoutes(router, func(required string, next http.HandlerFunc) http.Handler {
		return requireAuth(authManager)(requireRoleHandler(required, next))
	})

	server := httptest.NewServer(router)
	defer server.Close()

	headers := http.Header{}
	headers.Set("Authorization", signedBearerToken(t, authManager, "11", "default", "user"))
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/api/ws/metrics?interval=1"

	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, headers)
	if err == nil {
		_ = conn.Close()
		t.Fatal("expected user role websocket dial to fail")
	}
	if resp == nil || resp.StatusCode != http.StatusForbidden {
		t.Fatalf("expected user role websocket rejection with 403, got resp=%v err=%v", resp, err)
	}
}

func TestCanonicalSecurityWebSocketAliasesUpgrade(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	handlers := securityapi.NewSecurityHandlers(auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret())), audit.NewSimpleAuditLogger())

	router := mux.NewRouter()
	registerSecurityWebSocketAliases(router, authManager, handlers)

	server := httptest.NewServer(router)
	defer server.Close()

	for _, route := range []string{"/api/ws/security/events", "/api/security/events/stream"} {
		headers := http.Header{}
		headers.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
		wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + route

		conn, _, err := websocket.DefaultDialer.Dial(wsURL, headers)
		if err != nil {
			t.Fatalf("dial %s: %v", route, err)
		}
		_ = conn.Close()
	}
}

func TestCanonicalSecurityWebSocketAliasesRejectNonAdminUsers(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	handlers := securityapi.NewSecurityHandlers(auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret())), audit.NewSimpleAuditLogger())

	router := mux.NewRouter()
	registerSecurityWebSocketAliases(router, authManager, handlers)

	server := httptest.NewServer(router)
	defer server.Close()

	headers := http.Header{}
	headers.Set("Authorization", signedBearerToken(t, authManager, "11", "default", "user"))
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/api/ws/security/events"

	conn, resp, err := websocket.DefaultDialer.Dial(wsURL, headers)
	if err == nil {
		_ = conn.Close()
		t.Fatal("expected security websocket dial to fail for non-admin user")
	}
	if resp == nil || resp.StatusCode != http.StatusForbidden {
		t.Fatalf("expected 403 for non-admin security websocket, got resp=%v err=%v", resp, err)
	}
}

// TestCanonicalVMCreatePersistsVCPUs proves the API contract residue fix: the
// create request's "vcpus" is persisted as the REAL cpu_cores value (not the
// CPUShares scheduling weight the column used to hold), defaults to 1 without
// vcpus, and the create response reports "vcpus". The KVM driver maps
// VCPUs straight to -smp (vm package tests cover that side).
func TestCanonicalVMCreatePersistsVCPUs(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()

	router := mux.NewRouter()
	registerSecureAPIRoutes(router, db, manager, t.TempDir())

	// Explicit vcpus=2: cpu_cores arg must be 2, and cpu_shares stays in
	// metadata only (arg 5 is os_type; the metadata JSON is arg 9).
	mock.ExpectExec("INSERT INTO vms").
		WithArgs(sqlmock.AnyArg(), "vcpu-vm", "stopped", 2, 512, 1, sqlmock.AnyArg(), "", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/vms", map[string]interface{}{
		"name": "vcpu-vm", "vcpus": 2, "memory_mb": 512, "disk_size_gb": 1,
	}))
	if rec.Code != http.StatusCreated {
		t.Fatalf("create: expected 201, got %d (%s)", rec.Code, rec.Body.String())
	}
	var created map[string]interface{}
	decodeJSONBody(t, rec, &created)
	if created["vcpus"] != float64(2) {
		t.Fatalf("create response: expected vcpus 2, got %#v", created["vcpus"])
	}
	if created["memory_mb"] != float64(512) || created["disk_gb"] != float64(1) {
		t.Fatalf("create response: expected memory_mb 512 disk_gb 1, got %#v", created)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCanonicalVMCreateRejectsVCPUsOver256 proves the create upper bound: a
// vcpus value above 256 is rejected 400 before any DB write (no sqlmock
// expectation registered — a stray INSERT would fail the run).
func TestCanonicalVMCreateRejectsVCPUsOver256(t *testing.T) {
	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()

	router := mux.NewRouter()
	registerSecureAPIRoutes(router, db, manager, t.TempDir())

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/vms", map[string]interface{}{
		"name": "toobig", "vcpus": 257, "memory_mb": 512,
	}))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for vcpus=257, got %d (%s)", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "vcpus must be between 1 and 256") {
		t.Fatalf("expected vcpus bound error, got %s", rec.Body.String())
	}
}

// TestCanonicalVMGetExposesVCPUs proves the get handler SELECTs cpu_cores/
// memory_mb/disk_gb and reports them as vcpus/memory_mb/disk_gb.
func TestCanonicalVMGetExposesVCPUs(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()

	router := mux.NewRouter()
	registerSecureAPIRoutes(router, db, manager, t.TempDir())

	now := time.Now().UTC()
	mock.ExpectQuery(regexp.QuoteMeta(`
			SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at
			FROM vms WHERE id = $1
		`)).
		WithArgs("vm-9").
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "state", "node_id", "organization_id", "cpu_cores", "memory_mb", "disk_gb", "created_at", "updated_at"}).
			AddRow("vm-9", "getter", "stopped", nil, nil, 4, 1024, 20, now, now))

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodGet, "/vms/vm-9", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("get: expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var got map[string]interface{}
	decodeJSONBody(t, rec, &got)
	if got["vcpus"] != float64(4) {
		t.Fatalf("get: expected vcpus 4, got %#v", got["vcpus"])
	}
	if got["memory_mb"] != float64(1024) || got["disk_gb"] != float64(20) {
		t.Fatalf("get: expected memory_mb 1024 disk_gb 20, got %#v", got)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCanonicalAdminUserUpdateRejectsInvalidRole proves strict role validation:
// an unknown label is rejected 400 with the canonical alias hint, and NO UPDATE
// is executed (no sqlmock expectation is registered — a stray UPDATE would
// fail the run). Previously a typo like "ghost" silently became "viewer".
func TestCanonicalAdminUserUpdateRejectsInvalidRole(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	router := mux.NewRouter()
	adminRouter := router.PathPrefix("/api/admin").Subrouter()
	adminRouter.Use(requireAuth(authManager))
	adminRouter.Use(requireAnyRoleMiddleware("admin", "super-admin"))
	registerCanonicalAdminRoutes(router, authManager, db)

	rec := httptest.NewRecorder()
	req := mustJSONRequest(t, http.MethodPut, "/api/admin/users/00000000-0000-0000-0000-000000000001", map[string]interface{}{"role": "ghost"})
	req.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for role ghost, got %d (%s)", rec.Code, rec.Body.String())
	}
	want := "invalid role: ghost (valid: admin, operator, viewer; aliases: super-admin, user, readonly)"
	if !strings.Contains(rec.Body.String(), want) {
		t.Fatalf("expected error %q, got %s", want, rec.Body.String())
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations (an unexpected UPDATE ran): %v", err)
	}
}

// TestCanonicalAdminUserUpdateAcceptsSuperAdminAlias proves the six accepted
// labels still map: super-admin -> admin in the DB write.
func TestCanonicalAdminUserUpdateAcceptsSuperAdminAlias(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	router := mux.NewRouter()
	registerCanonicalAdminRoutes(router, authManager, db)

	now := time.Now().UTC()
	// updateCanonicalAdminUser issues ONE QueryRow("UPDATE users ... RETURNING
	// ...") — sqlmock models it as ExpectQuery, not ExpectExec.
	mock.ExpectQuery(regexp.QuoteMeta(`
			UPDATE users
			SET role = $1, updated_at = NOW()
			WHERE id = $2
			RETURNING id, username, email, role, status, created_at, updated_at
		`)).
		WithArgs("admin", "00000000-0000-0000-0000-000000000001").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "role", "status", "created_at", "updated_at"}).
			AddRow("00000000-0000-0000-0000-000000000001", "u", "u@e.com", "admin", "active", now, now))
	rec := httptest.NewRecorder()
	req := mustJSONRequest(t, http.MethodPut, "/api/admin/users/00000000-0000-0000-0000-000000000001", map[string]interface{}{"role": "super-admin"})
	req.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for role super-admin, got %d (%s)", rec.Code, rec.Body.String())
	}
}

// newAuthTokenTestRouter builds a router whose public auth token routes run
// against a sqlmock DB and an *auth.EmailService pointed at an unreachable
// SMTP host (the send is best-effort; handlers still return their status).
func newAuthTokenTestRouter(t *testing.T, db *sql.DB) (*mux.Router, sqlmock.Sqlmock, *auth.SimpleAuthManager) {
	t.Helper()

	authManager := auth.NewSimpleAuthManager("test-secret", db)
	emailService := auth.NewEmailService(auth.EmailConfig{
		SMTPHost:    "127.0.0.1",
		SMTPPort:    1, // nothing listens here; sends fail fast and are logged
		FromAddress: "test@novacron.local",
		FromName:    "NovaCron",
		FrontendURL: "http://localhost:8092",
	})

	router := mux.NewRouter()
	registerPublicRoutes(router, authManager, db, nil, emailService)
	return router, nil, authManager
}

// sha256HexForTest mirrors authTokenSHA256Hex (same package; kept local for
// readability in expectations).
func sha256HexForTest(t *testing.T, raw string) string {
	t.Helper()
	sum := sha256.Sum256([]byte(raw))
	return hex.EncodeToString(sum[:])
}

// TestCanonicalResetPasswordHappyPath proves a live reset token rotates the
// password, revokes sessions, and marks the token used inside one transaction.
func TestCanonicalResetPasswordHappyPath(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	router, _, _ := newAuthTokenTestRouter(t, db)

	rawToken := "0000000000000000000000000000000000000000000000000000000000000abc"
	tokenHash := sha256HexForTest(t, rawToken)

	mock.ExpectQuery(regexp.QuoteMeta(`
			SELECT user_id FROM auth_tokens
			WHERE token_hash = $1 AND purpose = 'password_reset' AND used_at IS NULL AND expires_at > NOW()
		`)).WithArgs(tokenHash).
		WillReturnRows(sqlmock.NewRows([]string{"user_id"}).AddRow("11111111-1111-1111-1111-111111111111"))
	mock.ExpectBegin()
	mock.ExpectExec(regexp.QuoteMeta(`UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2`)).
		WithArgs(sqlmock.AnyArg(), "11111111-1111-1111-1111-111111111111").
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(regexp.QuoteMeta(`UPDATE auth_tokens SET used_at = NOW() WHERE token_hash = $1 AND purpose = 'password_reset'`)).
		WithArgs(tokenHash).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(regexp.QuoteMeta(`DELETE FROM sessions WHERE user_id = $1`)).
		WithArgs("11111111-1111-1111-1111-111111111111").
		WillReturnResult(sqlmock.NewResult(0, 3))
	mock.ExpectCommit()

	req := mustJSONRequest(t, http.MethodPost, "/api/auth/reset-password", map[string]string{
		"token":    rawToken,
		"password": "NewPassw0rd!",
	})
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload map[string]string
	decodeJSONBody(t, rec, &payload)
	if payload["message"] != "Password reset successfully" {
		t.Fatalf("expected success message, got %#v", payload)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCanonicalResetPasswordExpiredToken proves an unknown/expired token is a
// 400 and never reaches the users/sessions tables.
func TestCanonicalResetPasswordExpiredToken(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	router, _, _ := newAuthTokenTestRouter(t, db)

	tokenHash := sha256HexForTest(t, "stale-or-unknown-token")
	mock.ExpectQuery(regexp.QuoteMeta(`
			SELECT user_id FROM auth_tokens
			WHERE token_hash = $1 AND purpose = 'password_reset' AND used_at IS NULL AND expires_at > NOW()
		`)).WithArgs(tokenHash).
		WillReturnError(sql.ErrNoRows)

	req := mustJSONRequest(t, http.MethodPost, "/api/auth/reset-password", map[string]string{
		"token":    "stale-or-unknown-token",
		"password": "NewPassw0rd!",
	})
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload map[string]string
	decodeJSONBody(t, rec, &payload)
	if payload["error"] != "invalid or expired token" {
		t.Fatalf("expected invalid-or-expired error, got %#v", payload)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCanonicalVerifyEmailHappyPath proves a live verification token flips
// email_verified and promotes a pending user to active.
func TestCanonicalVerifyEmailHappyPath(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	router, _, _ := newAuthTokenTestRouter(t, db)

	rawToken := "abcdef0000000000000000000000000000000000000000000000000000000012"
	tokenHash := sha256HexForTest(t, rawToken)

	mock.ExpectQuery(regexp.QuoteMeta(`
			SELECT user_id FROM auth_tokens
			WHERE token_hash = $1 AND purpose = 'email_verification' AND used_at IS NULL AND expires_at > NOW()
		`)).WithArgs(tokenHash).
		WillReturnRows(sqlmock.NewRows([]string{"user_id"}).AddRow("22222222-2222-2222-2222-222222222222"))
	mock.ExpectBegin()
	mock.ExpectExec(regexp.QuoteMeta(`
			UPDATE users
			SET email_verified = TRUE,
			    status = CASE WHEN status = 'pending' THEN 'active'::user_status ELSE status END,
			    updated_at = NOW()
			WHERE id = $1
		`)).WithArgs("22222222-2222-2222-2222-222222222222").
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(regexp.QuoteMeta(`UPDATE auth_tokens SET used_at = NOW() WHERE token_hash = $1 AND purpose = 'email_verification'`)).
		WithArgs(tokenHash).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectCommit()

	req := mustJSONRequest(t, http.MethodPost, "/api/auth/verify-email", map[string]string{"token": rawToken})
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload map[string]bool
	decodeJSONBody(t, rec, &payload)
	if payload["success"] != true {
		t.Fatalf("expected success true, got %#v", payload)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCanonicalForgotPasswordConfiguredReturns200AfterSendFailure proves the
// configured path responds 200 (no enumeration) even when the SMTP send fails.
func TestCanonicalForgotPasswordConfiguredReturns200AfterSendFailure(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	router, _, _ := newAuthTokenTestRouter(t, db)

	mock.ExpectQuery(regexp.QuoteMeta(`SELECT id, username FROM users WHERE email = $1`)).
		WithArgs("known@novacron.local").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username"}).
			AddRow("33333333-3333-3333-3333-333333333333", "knownuser"))
	mock.ExpectExec(regexp.QuoteMeta(`UPDATE auth_tokens SET used_at = NOW() WHERE user_id = $1 AND purpose = $2 AND used_at IS NULL`)).
		WithArgs("33333333-3333-3333-3333-333333333333", "password_reset").
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(regexp.QuoteMeta(`
		INSERT INTO auth_tokens (user_id, token_hash, purpose, expires_at)
		VALUES ($1, $2, $3, NOW() + make_interval(secs => $4::int))
		`)).
		WithArgs("33333333-3333-3333-3333-333333333333", sqlmock.AnyArg(), "password_reset", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	req := mustJSONRequest(t, http.MethodPost, "/api/auth/forgot-password", map[string]string{"email": "known@novacron.local"})
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 (no enumeration), got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload map[string]string
	decodeJSONBody(t, rec, &payload)
	if payload["message"] != "If an account exists for that email, a reset link has been sent" {
		t.Fatalf("expected generic message, got %#v", payload)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCanonicalForgotPasswordUnknownEmailStillGeneric proves a miss on the
// users lookup still returns the same 200 generic message (and runs no token
// writes) — no account enumeration.
func TestCanonicalForgotPasswordUnknownEmailStillGeneric(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	router, _, _ := newAuthTokenTestRouter(t, db)

	mock.ExpectQuery(regexp.QuoteMeta(`SELECT id, username FROM users WHERE email = $1`)).
		WithArgs("nobody@novacron.local").
		WillReturnError(sql.ErrNoRows)

	req := mustJSONRequest(t, http.MethodPost, "/api/auth/forgot-password", map[string]string{"email": "nobody@novacron.local"})
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCanonicalResendVerificationGenericSuccess proves resend returns
// success:true for a known unverified user and issues a fresh token.
func TestCanonicalResendVerificationGenericSuccess(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	router, _, _ := newAuthTokenTestRouter(t, db)

	mock.ExpectQuery(regexp.QuoteMeta(`SELECT id, username, email_verified FROM users WHERE email = $1`)).
		WithArgs("pending@novacron.local").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email_verified"}).
			AddRow("44444444-4444-4444-4444-444444444444", "pendinguser", false))
	mock.ExpectExec(regexp.QuoteMeta(`UPDATE auth_tokens SET used_at = NOW() WHERE user_id = $1 AND purpose = $2 AND used_at IS NULL`)).
		WithArgs("44444444-4444-4444-4444-444444444444", "email_verification").
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(regexp.QuoteMeta(`
		INSERT INTO auth_tokens (user_id, token_hash, purpose, expires_at)
		VALUES ($1, $2, $3, NOW() + make_interval(secs => $4::int))
		`)).
		WithArgs("44444444-4444-4444-4444-444444444444", sqlmock.AnyArg(), "email_verification", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(1, 1))

	req := mustJSONRequest(t, http.MethodPost, "/api/auth/resend-verification", map[string]string{"email": "pending@novacron.local"})
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload map[string]bool
	decodeJSONBody(t, rec, &payload)
	if payload["success"] != true {
		t.Fatalf("expected success true, got %#v", payload)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// TestCanonicalResetPasswordRejectsWeakPassword proves the reset route applies
// the canonical registration password policy before touching the database.
func TestCanonicalResetPasswordRejectsWeakPassword(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	router, _, _ := newAuthTokenTestRouter(t, db)

	req := mustJSONRequest(t, http.MethodPost, "/api/auth/reset-password", map[string]string{
		"token":    "some-token",
		"password": "short",
	})
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for weak password, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("weak password must not touch the DB: %v", err)
	}
}
