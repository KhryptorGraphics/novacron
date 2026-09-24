package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"

	graphqlapi "github.com/khryptorgraphics/novacron/backend/api/graphql"
	securityapi "github.com/khryptorgraphics/novacron/backend/api/security"
	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"golang.org/x/crypto/bcrypt"
)

func signedBearerToken(t *testing.T, authManager *auth.SimpleAuthManager, userID string, tenantID string, role string) string {
	t.Helper()

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id":   userID,
		"tenant_id": tenantID,
		"role":      role,
		"roles":     []string{role},
		"exp":       time.Now().Add(time.Hour).Unix(),
		"iat":       time.Now().Unix(),
	})

	tokenString, err := token.SignedString([]byte(authManager.GetJWTSecret()))
	if err != nil {
		t.Fatalf("failed to sign token: %v", err)
	}

	return "Bearer " + tokenString
}

func TestRequireAuthRejectsInvalidToken(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)

	router := mux.NewRouter()
	protected := router.PathPrefix("/api").Subrouter()
	protected.Use(requireAuth(authManager))
	protected.HandleFunc("/v1/vms", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/vms", nil)
	req.Header.Set("Authorization", "Bearer invalid-token")

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for invalid token, got %d", rec.Code)
	}
}

func TestRequireAuthAcceptsValidToken(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)

	router := mux.NewRouter()
	protected := router.PathPrefix("/api").Subrouter()
	protected.Use(requireAuth(authManager))
	protected.HandleFunc("/v1/vms", func(w http.ResponseWriter, r *http.Request) {
		payload := map[string]interface{}{
			"user_id":   r.Context().Value("user_id"),
			"tenant_id": r.Context().Value("tenant_id"),
			"role":      r.Context().Value("role"),
		}
		writeJSON(w, http.StatusOK, payload)
	})

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id":   "42",
		"tenant_id": "default",
		"role":      "admin",
		"roles":     []string{"admin"},
		"exp":       time.Now().Add(time.Hour).Unix(),
		"iat":       time.Now().Unix(),
	})
	tokenString, err := token.SignedString([]byte(authManager.GetJWTSecret()))
	if err != nil {
		t.Fatalf("failed to sign token: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/vms", nil)
	req.Header.Set("Authorization", "Bearer "+tokenString)

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for valid token, got %d (%s)", rec.Code, rec.Body.String())
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if payload["user_id"] != "42" {
		t.Fatalf("expected user_id 42, got %#v", payload["user_id"])
	}
	if payload["tenant_id"] != "default" {
		t.Fatalf("expected tenant_id default, got %#v", payload["tenant_id"])
	}
	if payload["role"] != "admin" {
		t.Fatalf("expected role admin, got %#v", payload["role"])
	}
}

func TestRegisterPublicRoutesSupportsCanonicalEmailLogin(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", db)
	router := mux.NewRouter()
	registerPublicRoutes(router, authManager, db, nil, nil)

	passwordHash, err := bcrypt.GenerateFromPassword([]byte("correct-horse-battery-staple"), bcrypt.DefaultCost)
	if err != nil {
		t.Fatalf("failed to hash password: %v", err)
	}

	now := time.Now()
	mock.ExpectQuery(regexp.QuoteMeta(`SELECT username FROM users WHERE email = $1`)).
		WithArgs("user@example.com").
		WillReturnRows(sqlmock.NewRows([]string{"username"}).AddRow("user"))
	mock.ExpectQuery(regexp.QuoteMeta(`
		SELECT id, username, email, password_hash, role, status, created_at, updated_at, organization_id
		FROM users WHERE username = $1
	`)).
		WithArgs("user").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "password_hash", "role", "status", "created_at", "updated_at", "organization_id"}).
			AddRow("7", "user", "user@example.com", string(passwordHash), "admin", "active", now, now, "00000000-0000-0000-0000-000000000001"))

	req := httptest.NewRequest(http.MethodPost, "/api/auth/login", strings.NewReader(`{"email":"user@example.com","password":"correct-horse-battery-staple"}`))
	req.Header.Set("Content-Type", "application/json")

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}

	var payload struct {
		Token string `json:"token"`
		User  struct {
			ID       string   `json:"id"`
			Email    string   `json:"email"`
			Role     string   `json:"role"`
			Roles    []string `json:"roles"`
			TenantID string   `json:"tenantId"`
		} `json:"user"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if payload.Token == "" {
		t.Fatal("expected token in login response")
	}
	if payload.User.Email != "user@example.com" {
		t.Fatalf("expected user email user@example.com, got %q", payload.User.Email)
	}
	if payload.User.Role != "admin" {
		t.Fatalf("expected role admin, got %q", payload.User.Role)
	}
	if payload.User.TenantID != "00000000-0000-0000-0000-000000000001" {
		t.Fatalf("expected tenantId to be the seeded default org UUID, got %q", payload.User.TenantID)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestRegisterSecureAPIRoutesListsVMsOnCanonicalRoute(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	router := mux.NewRouter()
	apiV1 := router.PathPrefix("/api/v1").Subrouter()
	apiV1.Use(requireAuth(authManager))
	registerSecureAPIRoutes(apiV1, db, nil, t.TempDir())

	now := time.Now().UTC()
	mock.ExpectQuery(`SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at FROM vms ORDER BY created_at DESC`).
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "state", "node_id", "organization_id", "cpu_cores", "memory_mb", "disk_gb", "created_at", "updated_at"}).
			AddRow("vm-1", "alpha", "running", "node-a", nil, 2, 512, 10, now, now))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/vms", nil)
	req.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}

	var payload []map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if len(payload) != 1 {
		t.Fatalf("expected 1 VM, got %d", len(payload))
	}
	if payload[0]["id"] != "vm-1" {
		t.Fatalf("expected vm id vm-1, got %#v", payload[0]["id"])
	}
	if payload[0]["status"] != "running" {
		t.Fatalf("expected status running, got %#v", payload[0]["status"])
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestRegisterSecureAPIRoutesCreatesVMOnCompatibilityRoute(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	router := mux.NewRouter()
	apiCompat := router.PathPrefix("/api").Subrouter()
	apiCompat.Use(requireAuth(authManager))
	// nil manager exercises the metadata-only path (manager unavailable); the
	// row is recorded as "created", never the old fake "creating". node_id is
	// this node's selfNodeID() (VM belongs to the creating node) — AnyArg since
	// it depends on NOVACRON_NODE_ID, matching canonical_routes_test.go.
	registerSecureAPIRoutes(apiCompat, db, nil, t.TempDir())
	mock.ExpectExec(`INSERT INTO vms`).
		WithArgs(
			sqlmock.AnyArg(), // uuid id
			"builder",
			"stopped",
			1,                // cpu_cores: REAL vCPU count; request without vcpus persists 1
			2048,             // memory_mb
			0,                // disk_gb (0 = driver default)
			sqlmock.AnyArg(), // os_type (image)
			sqlmock.AnyArg(), // node_id: selfNodeID(), depends on NOVACRON_NODE_ID
			"",               // owner_id: non-uuid JWT sub is sanitized to '' (NULLIF -> NULL)
			sqlmock.AnyArg(), // requested_owner_id
			"",               // organization_id: JWT tenant_id "default" is not a uuid, so it is dropped
			sqlmock.AnyArg(), // metadata JSON
		).
		WillReturnResult(sqlmock.NewResult(1, 1))

	req := httptest.NewRequest(
		http.MethodPost,
		"/api/vms",
		strings.NewReader(`{"name":"builder","cpu_shares":1000,"memory_mb":2048}`),
	)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d (%s)", rec.Code, rec.Body.String())
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if payload["name"] != "builder" {
		t.Fatalf("expected name builder, got %#v", payload["name"])
	}
	if payload["status"] != "stopped" {
		t.Fatalf("expected status stopped, got %#v", payload["status"])
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestRegisterSecureAPIRoutesSupportsStateTransitionsAndMetrics(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	router := mux.NewRouter()
	apiV1 := router.PathPrefix("/api/v1").Subrouter()
	apiV1.Use(requireAuth(authManager))
	// Start now routes through the real manager; seed vm-42 so it exists.
	manager := newStubVMManager(t)
	defer manager.Stop()
	seedManagerVM(t, manager, "vm-42")
	registerSecureAPIRoutes(apiV1, db, manager, t.TempDir())

	mock.ExpectExec(`UPDATE vms SET state = \$2, updated_at = NOW\(\) WHERE id = \$1`).
		WithArgs("vm-42", "running").
		WillReturnResult(sqlmock.NewResult(0, 1))

	startReq := httptest.NewRequest(http.MethodPost, "/api/v1/vms/vm-42/start", nil)
	startReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	startRec := httptest.NewRecorder()
	router.ServeHTTP(startRec, startReq)

	if startRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for start, got %d (%s)", startRec.Code, startRec.Body.String())
	}

	mock.ExpectQuery(`SELECT COALESCE\(cpu_usage, 0\), COALESCE\(memory_usage, 0\)`).
		WithArgs("vm-42").
		WillReturnRows(sqlmock.NewRows([]string{"cpu_usage", "memory_usage"}).AddRow(32.5, 61.25))

	metricsReq := httptest.NewRequest(http.MethodGet, "/api/v1/vms/vm-42/metrics", nil)
	metricsReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	metricsRec := httptest.NewRecorder()
	router.ServeHTTP(metricsRec, metricsReq)

	if metricsRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for metrics, got %d (%s)", metricsRec.Code, metricsRec.Body.String())
	}

	var metrics map[string]interface{}
	if err := json.NewDecoder(metricsRec.Body).Decode(&metrics); err != nil {
		t.Fatalf("failed to decode metrics response: %v", err)
	}

	if metrics["id"] != "vm-42" {
		t.Fatalf("expected metrics id vm-42, got %#v", metrics["id"])
	}
	if metrics["cpu_usage"] != 32.5 {
		t.Fatalf("expected cpu usage 32.5, got %#v", metrics["cpu_usage"])
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestAPIInfoAdvertisesCanonicalContract(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/info", nil)

	apiInfoHandler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var payload map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	endpoints, ok := payload["endpoints"].([]interface{})
	if !ok {
		t.Fatalf("expected endpoints list, got %#v", payload["endpoints"])
	}

	compatibilityEndpoints, ok := payload["compatibility_endpoints"].([]interface{})
	if !ok {
		t.Fatalf("expected compatibility endpoints list, got %#v", payload["compatibility_endpoints"])
	}

	unsupportedEndpoints, ok := payload["unsupported_endpoints"].([]interface{})
	if !ok {
		t.Fatalf("expected unsupported endpoints list, got %#v", payload["unsupported_endpoints"])
	}

	assertContains := func(label string, values []interface{}, expected string) {
		t.Helper()
		for _, value := range values {
			if value == expected {
				return
			}
		}
		t.Fatalf("expected %s to contain %q, got %#v", label, expected, values)
	}

	assertContains("endpoints", endpoints, "/api/v1/vms")
	assertContains("endpoints", endpoints, "/api/v1/monitoring/metrics")
	assertContains("endpoints", endpoints, "/api/security/threats")
	assertContains("endpoints", endpoints, "/api/security/compliance")
	assertContains("endpoints", endpoints, "/api/security/compliance/check")
	assertContains("endpoints", endpoints, "/api/security/compliance/export")
	assertContains("endpoints", endpoints, "/api/security/incidents")
	assertContains("endpoints", endpoints, "/api/security/events/{eventId}/acknowledge")
	assertContains("endpoints", endpoints, "/api/admin/security/compliance/check")
	assertContains("endpoints", endpoints, "/api/admin/security/compliance/export")
	assertContains("endpoints", endpoints, "/api/admin/security/incidents")
	assertContains("endpoints", endpoints, "/api/admin/security/events/{eventId}/acknowledge")
	assertContains("endpoints", endpoints, "/graphql")
	assertContains("endpoints", endpoints, "/api/ws/console/{vmId}")
	assertContains("compatibility_endpoints", compatibilityEndpoints, "/api/vms")
	assertContains("compatibility_endpoints", compatibilityEndpoints, "/ws/metrics")
	// novacron-8t1: resend-verification and verify-email ARE implemented
	// (registered at main.go:823-824 and covered by canonical_routes_test.go) —
	// they must not be reported as unsupported.
	for _, ep := range unsupportedEndpoints {
		if ep == "/api/auth/resend-verification" || ep == "/api/auth/verify-email" {
			t.Fatalf("%q must not appear in unsupported_endpoints (it is implemented and registered)", ep)
		}
	}
}

func TestRegisterCanonicalSecurityRoutesServesDashboardEndpoints(t *testing.T) {
	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	twoFactorService := auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret()))
	auditLogger := audit.NewSimpleAuditLogger()
	if err := auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
		ID:        "audit-1",
		Timestamp: time.Now().Add(-5 * time.Minute).UTC(),
		EventType: audit.EventPermissionDeny,
		Actor:     "alice@example.com",
		UserID:    "7",
		Resource:  "admin_panel",
		Action:    audit.ActionRead,
		Result:    audit.ResultDenied,
		ClientIP:  "203.0.113.7",
		Details: map[string]interface{}{
			"description": "Blocked admin panel access from suspicious IP",
		},
	}); err != nil {
		t.Fatalf("failed to seed audit log: %v", err)
	}

	handlers := securityapi.NewSecurityHandlers(twoFactorService, auditLogger).WithRBACStore(securityapi.NewPostgresRBACStore(db))
	router := mux.NewRouter()
	registerCanonicalSecurityRoutes(router, authManager, handlers)

	token := signedBearerToken(t, authManager, "7", "default", "admin")
	for _, endpoint := range []string{
		"/api/security/compliance",
		"/api/security/incidents",
		"/api/security/events",
		"/api/security/audit/statistics",
	} {
		req := httptest.NewRequest(http.MethodGet, endpoint, nil)
		req.Header.Set("Authorization", token)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200 from %s, got %d (%s)", endpoint, rec.Code, rec.Body.String())
		}
	}

	complianceReq := httptest.NewRequest(http.MethodGet, "/api/security/compliance", nil)
	complianceReq.Header.Set("Authorization", token)
	complianceRec := httptest.NewRecorder()
	router.ServeHTTP(complianceRec, complianceReq)

	var compliancePayload map[string]interface{}
	if err := json.NewDecoder(complianceRec.Body).Decode(&compliancePayload); err != nil {
		t.Fatalf("failed to decode compliance response: %v", err)
	}
	if _, ok := compliancePayload["compliance_score"]; !ok {
		t.Fatalf("expected compliance_score in response, got %#v", compliancePayload)
	}
	if frameworks, ok := compliancePayload["frameworks"].([]interface{}); !ok || len(frameworks) == 0 {
		t.Fatalf("expected compliance frameworks in response, got %#v", compliancePayload["frameworks"])
	}

	incidentReq := httptest.NewRequest(http.MethodGet, "/api/security/incidents", nil)
	incidentReq.Header.Set("Authorization", token)
	incidentRec := httptest.NewRecorder()
	router.ServeHTTP(incidentRec, incidentReq)

	var incidentPayload map[string]interface{}
	if err := json.NewDecoder(incidentRec.Body).Decode(&incidentPayload); err != nil {
		t.Fatalf("failed to decode incidents response: %v", err)
	}
	incidents, ok := incidentPayload["incidents"].([]interface{})
	if !ok || len(incidents) == 0 {
		t.Fatalf("expected at least one incident, got %#v", incidentPayload["incidents"])
	}

	auditStatsReq := httptest.NewRequest(http.MethodGet, "/api/security/audit/statistics", nil)
	auditStatsReq.Header.Set("Authorization", token)
	auditStatsRec := httptest.NewRecorder()
	router.ServeHTTP(auditStatsRec, auditStatsReq)

	var auditStatsPayload map[string]interface{}
	if err := json.NewDecoder(auditStatsRec.Body).Decode(&auditStatsPayload); err != nil {
		t.Fatalf("failed to decode audit statistics response: %v", err)
	}
	if _, ok := auditStatsPayload["overallScore"]; !ok {
		t.Fatalf("expected overallScore in audit statistics, got %#v", auditStatsPayload)
	}
}

func TestRegisterCanonicalGraphQLRouteSupportsVolumeOperations(t *testing.T) {
	authManager := auth.NewSimpleAuthManager("test-secret", nil)
	volumeStore, err := storage.NewStorageManager(storage.StorageManagerConfig{
		BasePath: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("failed to create storage manager: %v", err)
	}

	router := mux.NewRouter()
	registerCanonicalGraphQLRoute(
		router,
		authManager,
		graphqlapi.NewVolumeHTTPHandler(graphqlapi.NewResolverWithVolumeStore(nil, nil, volumeStore)),
	)

	createReq := httptest.NewRequest(http.MethodPost, "/graphql", bytes.NewBufferString(`{
		"query":"mutation CreateVolume($input: CreateVolumeInput!) { createVolume(input: $input) { id name tier size } }",
		"variables":{"input":{"name":"alpha","size":25,"tier":"hot"}}
	}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)

	if createRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for createVolume, got %d (%s)", createRec.Code, createRec.Body.String())
	}

	var createPayload map[string]interface{}
	if err := json.NewDecoder(createRec.Body).Decode(&createPayload); err != nil {
		t.Fatalf("failed to decode createVolume response: %v", err)
	}

	data, ok := createPayload["data"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected GraphQL data envelope, got %#v", createPayload)
	}
	createdVolume, ok := data["createVolume"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected createVolume payload, got %#v", data["createVolume"])
	}
	if createdVolume["name"] != "alpha" {
		t.Fatalf("expected created volume name alpha, got %#v", createdVolume["name"])
	}

	queryReq := httptest.NewRequest(http.MethodPost, "/graphql", bytes.NewBufferString(`{
		"query":"query Volumes { volumes { id name tier size } }"
	}`))
	queryReq.Header.Set("Content-Type", "application/json")
	queryReq.Header.Set("Authorization", signedBearerToken(t, authManager, "7", "default", "admin"))
	queryRec := httptest.NewRecorder()
	router.ServeHTTP(queryRec, queryReq)

	if queryRec.Code != http.StatusOK {
		t.Fatalf("expected 200 for volumes query, got %d (%s)", queryRec.Code, queryRec.Body.String())
	}

	var queryPayload map[string]interface{}
	if err := json.NewDecoder(queryRec.Body).Decode(&queryPayload); err != nil {
		t.Fatalf("failed to decode volumes response: %v", err)
	}

	queryData, ok := queryPayload["data"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected GraphQL data envelope, got %#v", queryPayload)
	}
	volumes, ok := queryData["volumes"].([]interface{})
	if !ok || len(volumes) != 1 {
		t.Fatalf("expected one volume in GraphQL query, got %#v", queryData["volumes"])
	}

	volume, ok := volumes[0].(map[string]interface{})
	if !ok {
		t.Fatalf("expected volume object, got %#v", volumes[0])
	}
	if volume["name"] != "alpha" {
		t.Fatalf("expected queried volume name alpha, got %#v", volume["name"])
	}
	if tier, _ := volume["tier"].(string); !strings.EqualFold(tier, "hot") {
		t.Fatalf("expected queried volume tier hot, got %#v", volume["tier"])
	}
}

func TestBuildCanonicalServerSupportsLiveStartup(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.MonitorPingsOption(true))
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", db)
	twoFactorService := auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret()))
	securityHandlers := securityapi.NewSecurityHandlers(twoFactorService, audit.NewSimpleAuditLogger())
	volumeStore, err := storage.NewStorageManager(storage.StorageManagerConfig{
		BasePath: filepath.Join(t.TempDir(), "volumes"),
	})
	if err != nil {
		t.Fatalf("failed to create storage manager: %v", err)
	}

	wsHandler := websocketapi.NewWebSocketHandler(nil, nil, nil, nil, logrus.New())
	defer wsHandler.Shutdown()

	services := &canonicalServices{
		twoFactorService: twoFactorService,
		securityHandlers: securityHandlers,
		websocketHandler: wsHandler,
		graphqlHandler: graphqlapi.NewVolumeHTTPHandler(
			graphqlapi.NewResolverWithVolumeStore(nil, nil, volumeStore),
		),
		shutdown: func() {
			wsHandler.Shutdown()
		},
	}

	cfg := &config.Config{
		Server: config.ServerConfig{
			APIPort:         "0",
			ReadTimeout:     5 * time.Second,
			WriteTimeout:    5 * time.Second,
			IdleTimeout:     30 * time.Second,
			ShutdownTimeout: 5 * time.Second,
		},
		VM: config.VMConfig{
			StoragePath:     t.TempDir(),
			HypervisorAddrs: []string{"localhost:9000"},
		},
	}

	server := buildCanonicalServer(cfg, db, authManager, services, nil)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to create listener: %v", err)
	}

	serverErr := make(chan error, 1)
	go func() {
		serverErr <- server.Serve(listener)
	}()

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutdownCancel()
	defer func() {
		if err := server.Shutdown(shutdownCtx); err != nil && err != http.ErrServerClosed {
			t.Fatalf("failed to shutdown server: %v", err)
		}
		if err := <-serverErr; err != nil && err != http.ErrServerClosed {
			t.Fatalf("server exited unexpectedly: %v", err)
		}
	}()

	baseURL := "http://" + listener.Addr().String()
	mock.ExpectPing()

	healthResp, err := http.Get(baseURL + "/health")
	if err != nil {
		t.Fatalf("health request failed: %v", err)
	}
	defer healthResp.Body.Close()

	if healthResp.StatusCode != http.StatusOK {
		t.Fatalf("expected health 200, got %d", healthResp.StatusCode)
	}

	var healthPayload map[string]interface{}
	if err := json.NewDecoder(healthResp.Body).Decode(&healthPayload); err != nil {
		t.Fatalf("failed to decode health response: %v", err)
	}
	if healthPayload["status"] != "healthy" {
		t.Fatalf("expected healthy status, got %#v", healthPayload["status"])
	}

	passwordHash, err := bcrypt.GenerateFromPassword([]byte("correct-horse-battery-staple"), bcrypt.DefaultCost)
	if err != nil {
		t.Fatalf("failed to hash password: %v", err)
	}

	now := time.Now().UTC()
	mock.ExpectQuery(regexp.QuoteMeta(`SELECT username FROM users WHERE email = $1`)).
		WithArgs("admin@example.com").
		WillReturnRows(sqlmock.NewRows([]string{"username"}).AddRow("admin"))
	mock.ExpectQuery(regexp.QuoteMeta(`
		SELECT id, username, email, password_hash, role, status, created_at, updated_at, organization_id
		FROM users WHERE username = $1
	`)).
		WithArgs("admin").
		WillReturnRows(sqlmock.NewRows([]string{"id", "username", "email", "password_hash", "role", "status", "created_at", "updated_at", "organization_id"}).
			AddRow("7", "admin", "admin@example.com", string(passwordHash), "admin", "active", now, now, "00000000-0000-0000-0000-000000000001"))

	loginReq, err := http.NewRequest(http.MethodPost, baseURL+"/api/auth/login", strings.NewReader(`{"email":"admin@example.com","password":"correct-horse-battery-staple"}`))
	if err != nil {
		t.Fatalf("failed to build login request: %v", err)
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
		t.Fatalf("failed to decode login response: %v", err)
	}
	if loginPayload.Token == "" {
		t.Fatal("expected login token")
	}

	complianceReq, err := http.NewRequest(http.MethodGet, baseURL+"/api/security/compliance", nil)
	if err != nil {
		t.Fatalf("failed to build compliance request: %v", err)
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
		t.Fatalf("failed to decode compliance response: %v", err)
	}
	if _, ok := compliancePayload["compliance_score"]; !ok {
		t.Fatalf("expected compliance_score in response, got %#v", compliancePayload)
	}

	wsHeaders := http.Header{}
	wsHeaders.Set("Authorization", "Bearer "+loginPayload.Token)
	wsURL := "ws://" + listener.Addr().String() + "/api/ws/security/events"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, wsHeaders)
	if err != nil {
		t.Fatalf("failed to dial websocket: %v", err)
	}
	_ = conn.Close()

	createReq, err := http.NewRequest(http.MethodPost, baseURL+"/graphql", bytes.NewBufferString(`{
		"query":"mutation CreateVolume($input: CreateVolumeInput!) { createVolume(input: $input) { id name tier size } }",
		"variables":{"input":{"name":"startup-smoke","size":10,"tier":"hot"}}
	}`))
	if err != nil {
		t.Fatalf("failed to build GraphQL create request: %v", err)
	}
	createReq.Header.Set("Content-Type", "application/json")
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
				ID string `json:"id"`
			} `json:"createVolume"`
		} `json:"data"`
	}
	if err := json.NewDecoder(createResp.Body).Decode(&createPayload); err != nil {
		t.Fatalf("failed to decode createVolume response: %v", err)
	}
	if createPayload.Data.CreateVolume.ID == "" {
		t.Fatal("expected created volume id")
	}

	listReq, err := http.NewRequest(http.MethodPost, baseURL+"/graphql", bytes.NewBufferString(`{
		"query":"query Volumes { volumes { id name tier size } }"
	}`))
	if err != nil {
		t.Fatalf("failed to build GraphQL list request: %v", err)
	}
	listReq.Header.Set("Content-Type", "application/json")
	listReq.Header.Set("Authorization", "Bearer "+loginPayload.Token)

	listResp, err := http.DefaultClient.Do(listReq)
	if err != nil {
		t.Fatalf("volumes request failed: %v", err)
	}
	defer listResp.Body.Close()

	if listResp.StatusCode != http.StatusOK {
		t.Fatalf("expected volumes 200, got %d", listResp.StatusCode)
	}

	var listPayload struct {
		Data struct {
			Volumes []struct {
				ID string `json:"id"`
			} `json:"volumes"`
		} `json:"data"`
	}
	if err := json.NewDecoder(listResp.Body).Decode(&listPayload); err != nil {
		t.Fatalf("failed to decode volumes response: %v", err)
	}
	if len(listPayload.Data.Volumes) != 1 || listPayload.Data.Volumes[0].ID != createPayload.Data.CreateVolume.ID {
		t.Fatalf("expected created volume in GraphQL list response, got %#v", listPayload.Data.Volumes)
	}

	changeReq, err := http.NewRequest(http.MethodPost, baseURL+"/graphql", bytes.NewBufferString(fmt.Sprintf(`{
		"query":"mutation ChangeVolumeTier($id: ID!, $tier: String!) { changeVolumeTier(id: $id, tier: $tier) { id tier } }",
		"variables":{"id":"%s","tier":"cold"}
	}`, createPayload.Data.CreateVolume.ID)))
	if err != nil {
		t.Fatalf("failed to build GraphQL change-tier request: %v", err)
	}
	changeReq.Header.Set("Content-Type", "application/json")
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
				Tier string `json:"tier"`
			} `json:"changeVolumeTier"`
		} `json:"data"`
	}
	if err := json.NewDecoder(changeResp.Body).Decode(&changePayload); err != nil {
		t.Fatalf("failed to decode changeVolumeTier response: %v", err)
	}
	if !strings.EqualFold(changePayload.Data.ChangeVolumeTier.Tier, "cold") {
		t.Fatalf("expected changed tier cold, got %q", changePayload.Data.ChangeVolumeTier.Tier)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}


// Login rate limiting --------------------------------------------------------
//
// POST /api/auth/login has no other guard, so these tests pin the properties
// that make the limiter worth having: it blocks past the limit, it does so per
// client IP, and a forged forwarding header cannot mint a fresh bucket.

// newRateLimitedLoginTestRouter builds the public routes over a sqlmock-backed
// database with no expectations, so every login attempt fails its user lookup
// and returns 401 unless the limiter rejects it first. It returns the router as
// well, for tests that need the /auth/login compatibility mount point.
func newRateLimitedLoginTestRouter(t *testing.T) (*mux.Router, func(remoteAddr string) *httptest.ResponseRecorder) {
	t.Helper()

	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("failed to create sqlmock: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	authManager := auth.NewSimpleAuthManager("test-secret", db)
	router := mux.NewRouter()
	registerPublicRoutes(router, authManager, db, nil, nil)

	return router, func(remoteAddr string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, "/api/auth/login",
			strings.NewReader(`{"email":"user@example.com","password":"correct-horse-battery-staple"}`))
		req.Header.Set("Content-Type", "application/json")
		req.RemoteAddr = remoteAddr

		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec
	}
}

func TestLoginRateLimiterSlidingWindow(t *testing.T) {
	limiter := newLoginRateLimiter(2, time.Minute)
	if limiter == nil {
		t.Fatal("newLoginRateLimiter(2, time.Minute) returned nil; want an enabled limiter")
	}

	now := time.Unix(1700000000, 0)
	limiter.now = func() time.Time { return now }

	for attempt := 1; attempt <= 2; attempt++ {
		if allowed, _ := limiter.allow("203.0.113.7"); !allowed {
			t.Fatalf("attempt %d rejected inside the limit of 2", attempt)
		}
	}

	allowed, retryAfter := limiter.allow("203.0.113.7")
	if allowed {
		t.Fatal("attempt past the limit allowed; want rejected")
	}
	if want := time.Minute; retryAfter != want {
		t.Fatalf("retryAfter = %v, want %v (time until the oldest attempt leaves the window)", retryAfter, want)
	}

	// A rejected attempt is not recorded, so a client that keeps hammering
	// cannot push its own block out further than one window.
	now = now.Add(30 * time.Second)
	allowed, retryAfter = limiter.allow("203.0.113.7")
	if allowed {
		t.Fatal("attempt allowed 30s into the block; want rejected")
	}
	if retryAfter != 30*time.Second {
		t.Fatalf("retryAfter = %v, want 30s: rejected attempts must not extend the block", retryAfter)
	}

	// The window slides: once the recorded attempts age out, the IP is free.
	now = now.Add(31 * time.Second)
	if allowed, _ := limiter.allow("203.0.113.7"); !allowed {
		t.Fatal("attempt rejected after the window slid past every recorded attempt")
	}

	if allowed, _ := limiter.allow("203.0.113.8"); !allowed {
		t.Fatal("a second client was rejected; the limiter is not bucketing per IP")
	}
}

func TestLoginRateLimiterBoundsTrackedClients(t *testing.T) {
	limiter := newLoginRateLimiter(1, time.Minute)
	if limiter == nil {
		t.Fatal("limiter disabled")
	}
	limiter.now = func() time.Time { return time.Unix(1700000000, 0) }

	// An unauthenticated caller can present as many source addresses as it
	// likes; the tracked-IP map must not grow with them.
	for i := range loginRateMaxClients + 16 {
		limiter.allow(fmt.Sprintf("198.51.%d.%d", i/256, i%256))
	}

	limiter.mu.Lock()
	tracked := len(limiter.hits)
	limiter.mu.Unlock()

	if tracked > loginRateMaxClients {
		t.Fatalf("limiter tracks %d clients, want at most %d", tracked, loginRateMaxClients)
	}
}

func TestLoginRateLimiterEnvKnobs(t *testing.T) {
	cases := []struct {
		name         string
		limit        string
		windowS      string
		wantDisabled bool
		wantLimit    int
		wantWindow   time.Duration
	}{
		{"unset uses the documented defaults", "", "", false, defaultLoginRateLimit, defaultLoginRateWindow},
		{"explicit values are honored", "3", "120", false, 3, 120 * time.Second},
		{"limit 0 disables the limiter", "0", "", true, 0, 0},
		{"unparsable values fall back to the defaults", "lots", "soon", false, defaultLoginRateLimit, defaultLoginRateWindow},
		{"non-positive window falls back to the default", "5", "0", false, 5, defaultLoginRateWindow},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("NOVACRON_LOGIN_RATE_LIMIT", tc.limit)
			t.Setenv("NOVACRON_LOGIN_RATE_WINDOW_S", tc.windowS)

			limiter := newLoginRateLimiterFromEnv()
			if tc.wantDisabled {
				if limiter != nil {
					t.Fatalf("NOVACRON_LOGIN_RATE_LIMIT=%q: limiter enabled, want nil (disabled)", tc.limit)
				}
				return
			}
			if limiter == nil {
				t.Fatalf("NOVACRON_LOGIN_RATE_LIMIT=%q: limiter nil, want enabled", tc.limit)
			}
			if limiter.limit != tc.wantLimit || limiter.window != tc.wantWindow {
				t.Fatalf("limit/window = %d/%v, want %d/%v", limiter.limit, limiter.window, tc.wantLimit, tc.wantWindow)
			}
		})
	}
}

func TestRegisterPublicRoutesRateLimitsLoginByIP(t *testing.T) {
	t.Setenv("NOVACRON_LOGIN_RATE_LIMIT", "3")
	t.Setenv("NOVACRON_LOGIN_RATE_WINDOW_S", "60")
	t.Setenv("NOVACRON_TRUSTED_PROXIES", "")

	router, login := newRateLimitedLoginTestRouter(t)

	for attempt := 1; attempt <= 3; attempt++ {
		if rec := login("203.0.113.9:34567"); rec.Code != http.StatusUnauthorized {
			t.Fatalf("attempt %d: got HTTP %d, want 401 (the limiter must not block inside the limit)", attempt, rec.Code)
		}
	}

	rec := login("203.0.113.9:34567")
	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("attempt past the limit: got HTTP %d, want 429", rec.Code)
	}
	retryAfter, err := strconv.Atoi(rec.Header().Get("Retry-After"))
	if err != nil {
		t.Fatalf("429 without a numeric Retry-After header: %q", rec.Header().Get("Retry-After"))
	}
	if retryAfter < 1 || retryAfter > 60 {
		t.Fatalf("Retry-After = %d, want 1..60 (the configured window)", retryAfter)
	}

	if rec := login("203.0.113.10:34567"); rec.Code != http.StatusUnauthorized {
		t.Fatalf("another client got HTTP %d, want 401: the limit is not bucketed per IP", rec.Code)
	}

	// /auth/login is the same handler on a compatibility path, so a blocked
	// client stays blocked there instead of getting a second allowance.
	compat := httptest.NewRequest(http.MethodPost, "/auth/login",
		strings.NewReader(`{"email":"user@example.com","password":"correct-horse-battery-staple"}`))
	compat.Header.Set("Content-Type", "application/json")
	compat.RemoteAddr = "203.0.113.9:34567"
	compatRec := httptest.NewRecorder()
	router.ServeHTTP(compatRec, compat)
	if compatRec.Code != http.StatusTooManyRequests {
		t.Fatalf("blocked client on /auth/login got HTTP %d, want 429 (both mount points must share one bucket)", compatRec.Code)
	}
}

func TestRegisterPublicRoutesLoginUnlimitedWhenLimitZero(t *testing.T) {
	t.Setenv("NOVACRON_LOGIN_RATE_LIMIT", "0")
	t.Setenv("NOVACRON_TRUSTED_PROXIES", "")

	_, login := newRateLimitedLoginTestRouter(t)

	for attempt := 1; attempt <= 12; attempt++ {
		if rec := login("203.0.113.9:34567"); rec.Code != http.StatusUnauthorized {
			t.Fatalf("attempt %d with NOVACRON_LOGIN_RATE_LIMIT=0: got HTTP %d, want 401 (limiter must be off)", attempt, rec.Code)
		}
	}
}

func TestLoginRateLimiterClientIPIgnoresForwardedHeadersByDefault(t *testing.T) {
	limiter := newLoginRateLimiter(1, time.Minute)
	if limiter == nil {
		t.Fatal("limiter disabled")
	}

	// A client-supplied X-Forwarded-For must not mint a new bucket: with no
	// trusted proxy configured the limiter keys on the peer address only.
	forged := httptest.NewRequest(http.MethodPost, "/api/auth/login", nil)
	forged.RemoteAddr = "203.0.113.11:34567"
	forged.Header.Set("X-Forwarded-For", "198.51.100.1")
	if got := limiter.clientIP(forged); got != "203.0.113.11" {
		t.Fatalf("clientIP = %q, want the peer address 203.0.113.11", got)
	}

	t.Setenv("NOVACRON_TRUSTED_PROXIES", "127.0.0.1, 10.0.0.0/8")
	trusted := newLoginRateLimiterFromEnv()
	if trusted == nil {
		t.Fatal("limiter disabled with default settings")
	}

	proxied := httptest.NewRequest(http.MethodPost, "/api/auth/login", nil)
	proxied.RemoteAddr = "127.0.0.1:44321"
	proxied.Header.Set("X-Forwarded-For", "198.51.100.7, 127.0.0.1")
	if got := trusted.clientIP(proxied); got != "198.51.100.7" {
		t.Fatalf("clientIP = %q, want the first forwarded entry 198.51.100.7", got)
	}

	// A peer outside the trusted list cannot use the header either.
	untrusted := httptest.NewRequest(http.MethodPost, "/api/auth/login", nil)
	untrusted.RemoteAddr = "203.0.113.12:34567"
	untrusted.Header.Set("X-Forwarded-For", "198.51.100.9")
	if got := trusted.clientIP(untrusted); got != "203.0.113.12" {
		t.Fatalf("clientIP = %q, want the untrusted peer address 203.0.113.12", got)
	}

	// CIDR entries cover a proxy on a fabric/private network.
	cidr := httptest.NewRequest(http.MethodPost, "/api/auth/login", nil)
	cidr.RemoteAddr = "10.96.0.2:34567"
	cidr.Header.Set("X-Forwarded-For", "198.51.100.11")
	if got := trusted.clientIP(cidr); got != "198.51.100.11" {
		t.Fatalf("clientIP = %q, want 198.51.100.11 via the trusted 10.0.0.0/8 prefix", got)
	}
}

func TestLoginRateLimiterConcurrentAttempts(t *testing.T) {
	const limit = 8

	limiter := newLoginRateLimiter(limit, time.Minute)
	if limiter == nil {
		t.Fatal("limiter disabled")
	}

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		allowed int
	)
	for range 64 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if ok, _ := limiter.allow("203.0.113.13"); ok {
				mu.Lock()
				allowed++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	// Concurrent callers must not be able to exceed the limit between them;
	// run under -race, this also pins the lock around the hits map.
	if allowed != limit {
		t.Fatalf("allowed = %d of 64 concurrent attempts, want exactly %d", allowed, limit)
	}
}
