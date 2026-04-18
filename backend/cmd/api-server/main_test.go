package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/mux"

	graphqlapi "github.com/khryptorgraphics/novacron/backend/api/graphql"
	securityapi "github.com/khryptorgraphics/novacron/backend/api/security"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
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
	registerPublicRoutes(router, authManager, db, nil)

	passwordHash, err := bcrypt.GenerateFromPassword([]byte("correct-horse-battery-staple"), bcrypt.DefaultCost)
	if err != nil {
		t.Fatalf("failed to hash password: %v", err)
	}

	now := time.Now()
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
	if payload.User.TenantID != "default" {
		t.Fatalf("expected tenantId default, got %q", payload.User.TenantID)
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
	registerSecureAPIRoutes(apiV1, db)

	now := time.Now().UTC()
	mock.ExpectQuery(`SELECT id, name, state, node_id, tenant_id, created_at, updated_at FROM vms ORDER BY created_at DESC`).
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "state", "node_id", "tenant_id", "created_at", "updated_at"}).
			AddRow("vm-1", "alpha", "running", "node-a", "default", now, now))

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
	registerSecureAPIRoutes(apiCompat, db)

	mock.ExpectExec(`INSERT INTO vms`).
		WithArgs(
			sqlmock.AnyArg(),
			"builder",
			"creating",
			nil,
			7,
			"default",
			sqlmock.AnyArg(),
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
	if payload["status"] != "creating" {
		t.Fatalf("expected status creating, got %#v", payload["status"])
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
	registerSecureAPIRoutes(apiV1, db)

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
	assertContains("endpoints", endpoints, "/api/security/incidents")
	assertContains("endpoints", endpoints, "/graphql")
	assertContains("compatibility_endpoints", compatibilityEndpoints, "/api/vms")
	assertContains("compatibility_endpoints", compatibilityEndpoints, "/ws/metrics")
	assertContains("unsupported_endpoints", unsupportedEndpoints, "/api/ws/console/{vmId}")
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
