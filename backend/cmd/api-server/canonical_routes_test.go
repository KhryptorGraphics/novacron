package main

import (
	"bytes"
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
