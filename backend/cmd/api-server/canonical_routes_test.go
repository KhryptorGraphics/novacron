package main

import (
	"bytes"
	"context"
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
	"golang.org/x/crypto/bcrypt"
)

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

func TestCanonicalPasswordResetRoutesAreLive(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)

	router := mux.NewRouter()
	registerPublicRoutes(router, authManager, nil, nil)

	tests := []struct {
		name       string
		path       string
		payload    map[string]string
		wantStatus int
		wantMsg    string
	}{
		{
			name:       "forgot password",
			path:       "/api/auth/forgot-password",
			payload:    map[string]string{"email": "user@example.com"},
			wantStatus: http.StatusOK,
			wantMsg:    "Password reset email sent",
		},
		{
			name:       "reset password",
			path:       "/api/auth/reset-password",
			payload:    map[string]string{"token": "reset-token", "password": "SecurePassword123!"},
			wantStatus: http.StatusOK,
			wantMsg:    "Password reset successfully",
		},
		{
			name:       "verify email remains deferred",
			path:       "/api/auth/verify-email",
			payload:    map[string]string{"token": "verify-token"},
			wantStatus: http.StatusNotImplemented,
			wantMsg:    "email verification is not wired in the canonical server yet",
		},
		{
			name:       "resend verification remains deferred",
			path:       "/api/auth/resend-verification",
			payload:    map[string]string{"email": "user@example.com"},
			wantStatus: http.StatusNotImplemented,
			wantMsg:    "email verification is not wired in the canonical server yet",
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
			if payload["message"] != tc.wantMsg {
				t.Fatalf("expected message %q, got %#v", tc.wantMsg, payload["message"])
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
	registerPublicRoutes(router, authManager, db, twoFactorService)
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
		SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at
		FROM users WHERE username = $1
	`)).
		WithArgs("user").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "password_hash", "role", "tenant_id", "created_at", "updated_at"}).
			AddRow("7", "user", "user@example.com", string(passwordHash), "admin", "default", now, now))

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
		SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at
		FROM users WHERE id = $1
	`)).
		WithArgs("7").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "password_hash", "role", "tenant_id", "created_at", "updated_at"}).
			AddRow("7", "user", "user@example.com", string(passwordHash), "admin", "default", now, now))

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
	registerPublicRoutes(router, authManager, nil, twoFactorService)

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
		"title": "Manual investigation",
		"description": "Operator escalated a suspicious login pattern.",
		"severity": "high",
		"type": "manual",
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
	registerPublicRoutes(router, authManager, db, twoFactorService)
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
		SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at
		FROM users WHERE username = $1
	`)).
		WithArgs("admin").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "password_hash", "role", "tenant_id", "created_at", "updated_at"}).
			AddRow("7", "admin", "admin@example.com", string(passwordHash), "admin", "default", now, now))

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
