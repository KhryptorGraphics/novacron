package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/mux"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
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
	registerPublicRoutes(router, authManager, db)

	passwordHash, err := bcrypt.GenerateFromPassword([]byte("correct-horse-battery-staple"), bcrypt.DefaultCost)
	if err != nil {
		t.Fatalf("failed to hash password: %v", err)
	}

	now := time.Now()
	mock.ExpectQuery(`SELECT username FROM users WHERE email = \\$1`).
		WithArgs("user@example.com").
		WillReturnRows(sqlmock.NewRows([]string{"username"}).AddRow("user"))
	mock.ExpectQuery(`SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at\\s+FROM users WHERE username = \\$1`).
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

func TestRegisterExplicitlyUnsupportedRoutesReturnsNotImplemented(t *testing.T) {
	router := mux.NewRouter()
	registerExplicitlyUnsupportedRoutes(router)

	req := httptest.NewRequest(http.MethodGet, "/api/security/health", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d", rec.Code)
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
	assertContains("compatibility_endpoints", compatibilityEndpoints, "/api/vms")
	assertContains("unsupported_endpoints", unsupportedEndpoints, "/graphql")
	assertContains("unsupported_endpoints", unsupportedEndpoints, "/api/security/*")
}
