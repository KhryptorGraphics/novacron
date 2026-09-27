package main

import (
	"context"
	"database/sql"
	"errors"
	"net/http"
	"net/http/httptest"
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/mux"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

const (
	testSessionID  = "3f0e8d9a-6a2b-4c1e-9f2b-1c2d3e4f5a6b"
	testSessionID2 = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"
	testUserID     = "7"
)

var userRowColumns = []string{"id", "username", "email", "password_hash", "role", "status", "created_at", "updated_at", "organization_id"}

func testUser() *auth.User {
	return &auth.User{
		ID:        testUserID,
		Username:  "user",
		Email:     "user@example.com",
		RoleIDs:   []string{"admin"},
		TenantID:  "00000000-0000-0000-0000-000000000001",
		Status:    auth.UserStatusActive,
		CreatedAt: time.Now().Add(-48 * time.Hour),
		UpdatedAt: time.Now().Add(-48 * time.Hour),
	}
}

// expectSessionInsert mocks createSession's INSERT ... RETURNING.
func expectSessionInsert(mock sqlmock.Sqlmock, userID, sessionID string) {
	now := time.Now().UTC()
	mock.ExpectQuery(regexp.QuoteMeta(sqlCreateSession)).
		WithArgs(userID, sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg(), sqlmock.AnyArg()).
		WillReturnRows(sqlmock.NewRows([]string{"id", "created_at", "expires_at", "last_accessed_at"}).
			AddRow(sessionID, now, now.Add(refreshTokenTTL), now))
}

// expectAuthLookup mocks authenticateToken's folded users/sessions query.
// sessionLive=false returns a NULL s.id (session revoked, swept, or none).
func expectAuthLookup(mock sqlmock.Sqlmock, userID, sessionID string, updatedAt time.Time, status string, sessionLive bool) {
	var matched interface{}
	if sessionLive {
		matched = sessionID
	}
	mock.ExpectQuery(regexp.QuoteMeta(sqlAuthenticateTokenLookup)).
		WithArgs(userID, sessionID).
		WillReturnRows(sqlmock.NewRows([]string{"updated_at", "status", "id"}).AddRow(updatedAt, status, matched))
}

func expectGetUser(mock sqlmock.Sqlmock, user *auth.User) {
	mock.ExpectQuery(regexp.QuoteMeta(`
		SELECT id, username, email, password_hash, role, status, created_at, updated_at, organization_id
		FROM users WHERE id = $1
	`)).
		WithArgs(user.ID).
		WillReturnRows(sqlmock.NewRows(userRowColumns).
			AddRow(user.ID, user.Username, user.Email, "x", user.RoleIDs[0], string(user.Status), user.CreatedAt, user.UpdatedAt, user.TenantID))
}

func expectFetchSession(mock sqlmock.Sqlmock, sessionID, userID string) {
	now := time.Now().UTC()
	mock.ExpectQuery(regexp.QuoteMeta(sqlFetchSession)).
		WithArgs(sessionID).
		WillReturnRows(sqlmock.NewRows([]string{"id", "user_id", "created_at", "expires_at", "last_accessed_at"}).
			AddRow(sessionID, userID, now.Add(-time.Hour), now.Add(refreshTokenTTL), now.Add(-time.Minute)))
}

func expectRefreshLookup(mock sqlmock.Sqlmock, rawToken, sessionID, userID string, revoked bool, isCurrent bool) {
	now := time.Now().UTC()
	var revokedAt interface{}
	if revoked {
		revokedAt = now.Add(-time.Second)
	}
	mock.ExpectQuery(regexp.QuoteMeta(sqlLookupSessionByRefreshHash)).
		WithArgs(authTokenSHA256Hex(rawToken)).
		WillReturnRows(sqlmock.NewRows([]string{"id", "user_id", "created_at", "revoked_at", "expires_at", "is_current"}).
			AddRow(sessionID, userID, now.Add(-time.Hour), revokedAt, now.Add(refreshTokenTTL), isCurrent))
}

func expectRotate(mock sqlmock.Sqlmock, oldRaw, sessionID, userID string) {
	now := time.Now().UTC()
	mock.ExpectQuery(regexp.QuoteMeta(sqlRotateRefreshToken)).
		WithArgs(sqlmock.AnyArg(), sqlmock.AnyArg(), sessionID, authTokenSHA256Hex(oldRaw)).
		WillReturnRows(sqlmock.NewRows([]string{"id", "user_id", "created_at", "expires_at", "last_accessed_at"}).
			AddRow(sessionID, userID, now.Add(-time.Hour), now.Add(refreshTokenTTL), now))
}

func newAuthSessionTestRouter(t *testing.T) (*mux.Router, sqlmock.Sqlmock, *auth.SimpleAuthManager) {
	t.Helper()
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	authManager := auth.NewSimpleAuthManager("test-secret", db)
	router := mux.NewRouter()
	registerAuthSessionRoutes(router, authManager, db, nil, nil)
	return router, mock, authManager
}

func mustSessionToken(t *testing.T, authManager *auth.SimpleAuthManager, user *auth.User, sessionID string) string {
	t.Helper()
	token, err := issueSessionToken(authManager.GetJWTSecret(), user, sessionID)
	if err != nil {
		t.Fatalf("issue session token: %v", err)
	}
	return token
}

type currentUserPayload struct {
	Token        string `json:"token"`
	RefreshToken string `json:"refreshToken"`
	User         struct {
		ID string `json:"id"`
	} `json:"user"`
	Admission       AdmissionResponse       `json:"admission"`
	Memberships     []AdmissionResponse     `json:"memberships"`
	SelectedCluster *ClusterSummaryResponse `json:"selectedCluster"`
	Session         SessionResponse         `json:"session"`
}

func assertCurrentUserShape(t *testing.T, payload currentUserPayload, sessionID string) {
	t.Helper()
	if payload.User.ID != testUserID {
		t.Fatalf("expected user %s, got %q", testUserID, payload.User.ID)
	}
	if !payload.Admission.Admitted || !payload.Admission.Selected || payload.Admission.ClusterID == "" {
		t.Fatalf("expected an admitted, selected local admission, got %#v", payload.Admission)
	}
	if len(payload.Memberships) != 1 || payload.Memberships[0].ClusterID != payload.Admission.ClusterID {
		t.Fatalf("expected exactly the local membership, got %#v", payload.Memberships)
	}
	if payload.SelectedCluster == nil || payload.SelectedCluster.ID != payload.Admission.ClusterID || payload.SelectedCluster.CurrentNodeCount < 1 {
		t.Fatalf("expected selectedCluster to be the local cluster, got %#v", payload.SelectedCluster)
	}
	if payload.Session.ID != sessionID || payload.Session.ExpiresAt == "" || payload.Session.CreatedAt == "" || payload.Session.LastAccessedAt == "" {
		t.Fatalf("expected session %s in payload, got %#v", sessionID, payload.Session)
	}
}

// --- authenticateToken / requireAuth ---

func TestAuthenticateTokenFailsClosed(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	authManager := auth.NewSimpleAuthManager("test-secret", db)
	user := testUser()
	past := time.Now().Add(-time.Hour)

	sign := func(claims jwt.MapClaims) string {
		s, err := jwt.NewWithClaims(jwt.SigningMethodHS256, claims).SignedString([]byte(authManager.GetJWTSecret()))
		if err != nil {
			t.Fatalf("sign: %v", err)
		}
		return s
	}
	base := func() jwt.MapClaims {
		return jwt.MapClaims{"user_id": testUserID, "role": "admin", "roles": []string{"admin"}, "tenant_id": "default",
			"exp": time.Now().Add(time.Hour).Unix(), "iat": time.Now().Unix()}
	}

	t.Run("garbage token", func(t *testing.T) {
		if _, status, err := authenticateToken(context.Background(), authManager, db, "nope"); status != http.StatusUnauthorized || err == nil {
			t.Fatalf("expected 401, got %d %v", status, err)
		}
	})
	t.Run("any purpose claim is rejected", func(t *testing.T) {
		for _, purpose := range []string{"pending_2fa", "password_reset", "whatever"} {
			claims := base()
			claims["purpose"] = purpose
			if _, status, err := authenticateToken(context.Background(), authManager, db, sign(claims)); status != http.StatusUnauthorized || err == nil {
				t.Fatalf("purpose %q: expected 401, got %d %v", purpose, status, err)
			}
		}
	})
	t.Run("missing iat", func(t *testing.T) {
		claims := base()
		delete(claims, "iat")
		if _, status, _ := authenticateToken(context.Background(), authManager, db, sign(claims)); status != http.StatusUnauthorized {
			t.Fatalf("expected 401 without iat, got %d", status)
		}
	})
	t.Run("malformed sid", func(t *testing.T) {
		claims := base()
		claims["sid"] = "not-a-uuid"
		if _, status, _ := authenticateToken(context.Background(), authManager, db, sign(claims)); status != http.StatusUnauthorized {
			t.Fatalf("expected 401 for malformed sid, got %d", status)
		}
	})
	t.Run("user row missing", func(t *testing.T) {
		mock.ExpectQuery(regexp.QuoteMeta(sqlAuthenticateTokenLookup)).WithArgs(testUserID, "").WillReturnError(sql.ErrNoRows)
		if _, status, _ := authenticateToken(context.Background(), authManager, db, sign(base())); status != http.StatusUnauthorized {
			t.Fatalf("expected 401 for missing user, got %d", status)
		}
	})
	t.Run("database error is 503", func(t *testing.T) {
		mock.ExpectQuery(regexp.QuoteMeta(sqlAuthenticateTokenLookup)).WithArgs(testUserID, "").WillReturnError(errors.New("connection refused"))
		if _, status, _ := authenticateToken(context.Background(), authManager, db, sign(base())); status != http.StatusServiceUnavailable {
			t.Fatalf("expected 503 for DB error, got %d", status)
		}
	})
	t.Run("inactive user", func(t *testing.T) {
		expectAuthLookup(mock, testUserID, "", past, "disabled", false)
		if _, status, _ := authenticateToken(context.Background(), authManager, db, sign(base())); status != http.StatusUnauthorized {
			t.Fatalf("expected 401 for disabled user, got %d", status)
		}
	})
	t.Run("user updated after issue", func(t *testing.T) {
		expectAuthLookup(mock, testUserID, "", time.Now().Add(time.Minute), "active", false)
		if _, status, _ := authenticateToken(context.Background(), authManager, db, sign(base())); status != http.StatusUnauthorized {
			t.Fatalf("expected 401 after profile change, got %d", status)
		}
	})
	t.Run("sid without live session row", func(t *testing.T) {
		expectAuthLookup(mock, testUserID, testSessionID, past, "active", false)
		if _, status, _ := authenticateToken(context.Background(), authManager, db, mustSessionToken(t, authManager, user, testSessionID)); status != http.StatusUnauthorized {
			t.Fatalf("expected 401 for revoked/missing session, got %d", status)
		}
	})
	t.Run("sid with live session", func(t *testing.T) {
		expectAuthLookup(mock, testUserID, testSessionID, past, "active", true)
		ctx, status, err := authenticateToken(context.Background(), authManager, db, mustSessionToken(t, authManager, user, testSessionID))
		if err != nil || status != http.StatusOK {
			t.Fatalf("expected success, got %d %v", status, err)
		}
		if ctx.Value("session_id") != testSessionID || ctx.Value("user_id") != testUserID || ctx.Value("role") != "admin" {
			t.Fatalf("unexpected principal context: sid=%v user=%v role=%v", ctx.Value("session_id"), ctx.Value("user_id"), ctx.Value("role"))
		}
	})
	t.Run("legacy token without sid skips the session check", func(t *testing.T) {
		expectAuthLookup(mock, testUserID, "", past, "active", false)
		if _, status, err := authenticateToken(context.Background(), authManager, db, sign(base())); err != nil || status != http.StatusOK {
			t.Fatalf("expected legacy token to pass, got %d %v", status, err)
		}
	})
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestLoginTokenPassesRequireAuthInProcess(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	authManager := auth.NewSimpleAuthManager("test-secret", db)
	user := testUser()

	router := mux.NewRouter()
	protected := router.PathPrefix("/api").Subrouter()
	protected.Use(requireAuth(authManager, db))
	protected.HandleFunc("/v1/whoami", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{"user_id": r.Context().Value("user_id"), "session_id": r.Context().Value("session_id")})
	})

	token := mustSessionToken(t, authManager, user, testSessionID)
	expectAuthLookup(mock, testUserID, testSessionID, user.UpdatedAt, "active", true)

	req := httptest.NewRequest(http.MethodGet, "/api/v1/whoami", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload map[string]string
	decodeJSONBody(t, rec, &payload)
	if payload["user_id"] != testUserID || payload["session_id"] != testSessionID {
		t.Fatalf("unexpected principal: %#v", payload)
	}

	// A DB outage must not let the request through.
	mock.ExpectQuery(regexp.QuoteMeta(sqlAuthenticateTokenLookup)).WithArgs(testUserID, testSessionID).WillReturnError(errors.New("down"))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 on DB error, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// --- /api/auth/me ---

func TestMeHandlerRequiresAuth(t *testing.T) {
	router, _, _ := newAuthSessionTestRouter(t)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/auth/me", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 without a token, got %d", rec.Code)
	}
}

func TestMeHandlerReturnsCurrentUserResponseShape(t *testing.T) {
	router, mock, authManager := newAuthSessionTestRouter(t)
	user := testUser()

	expectAuthLookup(mock, testUserID, testSessionID, user.UpdatedAt, "active", true)
	expectGetUser(mock, user)
	expectFetchSession(mock, testSessionID, testUserID)
	mock.ExpectExec(regexp.QuoteMeta(sqlTouchSessionLastAccessed)).WithArgs(testSessionID).WillReturnResult(sqlmock.NewResult(0, 1))

	req := httptest.NewRequest(http.MethodGet, "/api/auth/me", nil)
	req.Header.Set("Authorization", "Bearer "+mustSessionToken(t, authManager, user, testSessionID))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload currentUserPayload
	decodeJSONBody(t, rec, &payload)
	assertCurrentUserShape(t, payload, testSessionID)
	if payload.Token != "" {
		t.Fatal("/me must not mint tokens")
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestMeHandlerRejectsTokenWithoutSession(t *testing.T) {
	router, mock, authManager := newAuthSessionTestRouter(t)
	// A pre-session token passes requireAuth but has no session to report.
	expectAuthLookup(mock, testUserID, "", time.Now().Add(-time.Hour), "active", false)
	expectGetUser(mock, testUser())

	req := httptest.NewRequest(http.MethodGet, "/api/auth/me", nil)
	req.Header.Set("Authorization", signedBearerToken(t, authManager, testUserID, "default", "admin"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for a token without a session, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// --- /api/auth/refresh ---

func TestAuthRefreshRotatesTokenAndRejectsReuse(t *testing.T) {
	router, mock, authManager := newAuthSessionTestRouter(t)
	user := testUser()
	raw1, err := generateOpaqueToken(32)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Current token rotates.
	expectRefreshLookup(mock, raw1, testSessionID, testUserID, false, true)
	expectGetUser(mock, user)
	expectRotate(mock, raw1, testSessionID, testUserID)
	// buildMeResponsePayload does not touch the DB (nil twoFactorService).

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/refresh", map[string]string{"refreshToken": raw1}))
	if rec.Code != http.StatusOK {
		t.Fatalf("expected refresh 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload currentUserPayload
	decodeJSONBody(t, rec, &payload)
	assertCurrentUserShape(t, payload, testSessionID)
	if payload.RefreshToken == "" || payload.RefreshToken == raw1 {
		t.Fatalf("expected a rotated refresh token, got %q", payload.RefreshToken)
	}
	claims, err := validateJWT(payload.Token, authManager.GetJWTSecret())
	if err != nil || stringClaim(claims, "sid") != testSessionID || stringClaim(claims, "role") != "admin" {
		t.Fatalf("expected access token bound to session %s with DB role, got claims %#v (%v)", testSessionID, claims, err)
	}

	// 2. Presenting the rotated-out token revokes the whole session.
	expectRefreshLookup(mock, raw1, testSessionID, testUserID, false, false)
	mock.ExpectExec(regexp.QuoteMeta(sqlRevokeSession)).WithArgs(testSessionID).WillReturnResult(sqlmock.NewResult(0, 1))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/refresh", map[string]string{"refreshToken": raw1}))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected reuse to be rejected with 401, got %d (%s)", rec.Code, rec.Body.String())
	}

	// 3. The new token is dead too: the session is revoked.
	expectRefreshLookup(mock, payload.RefreshToken, testSessionID, testUserID, true, true)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/refresh", map[string]string{"refreshToken": payload.RefreshToken}))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected refresh on revoked session to be 401, got %d (%s)", rec.Code, rec.Body.String())
	}

	// 4. An unknown token is 401, an empty body 400.
	mock.ExpectQuery(regexp.QuoteMeta(sqlLookupSessionByRefreshHash)).WithArgs(sqlmock.AnyArg()).WillReturnError(sql.ErrNoRows)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/refresh", map[string]string{"refreshToken": "unknown"}))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected unknown token 401, got %d", rec.Code)
	}
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/refresh", map[string]string{}))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected missing token 400, got %d", rec.Code)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestAuthRefreshRejectsDeactivatedOrChangedUser(t *testing.T) {
	router, mock, _ := newAuthSessionTestRouter(t)
	raw, _ := generateOpaqueToken(32)

	t.Run("deactivated", func(t *testing.T) {
		user := testUser()
		user.Status = auth.UserStatusInactive
		expectRefreshLookup(mock, raw, testSessionID, testUserID, false, true)
		expectGetUser(mock, user)
		mock.ExpectExec(regexp.QuoteMeta(sqlRevokeSession)).WithArgs(testSessionID).WillReturnResult(sqlmock.NewResult(0, 1))
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/refresh", map[string]string{"refreshToken": raw}))
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401 for deactivated user, got %d (%s)", rec.Code, rec.Body.String())
		}
	})
	t.Run("credentials changed after session issue", func(t *testing.T) {
		user := testUser()
		user.UpdatedAt = time.Now().Add(time.Minute) // after the session's created_at (now-1h)
		expectRefreshLookup(mock, raw, testSessionID, testUserID, false, true)
		expectGetUser(mock, user)
		mock.ExpectExec(regexp.QuoteMeta(sqlRevokeSession)).WithArgs(testSessionID).WillReturnResult(sqlmock.NewResult(0, 1))
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/refresh", map[string]string{"refreshToken": raw}))
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401 after credential change, got %d (%s)", rec.Code, rec.Body.String())
		}
	})
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// --- /api/auth/logout ---

func TestAuthLogoutRevokesByRefreshTokenOrBearerSession(t *testing.T) {
	router, mock, authManager := newAuthSessionTestRouter(t)
	user := testUser()
	raw, _ := generateOpaqueToken(32)

	t.Run("refresh token alone, even with an expired access token", func(t *testing.T) {
		expired, err := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
			"user_id": testUserID, "sid": testSessionID, "exp": time.Now().Add(-time.Hour).Unix(), "iat": time.Now().Add(-2 * time.Hour).Unix(),
		}).SignedString([]byte(authManager.GetJWTSecret()))
		if err != nil {
			t.Fatal(err)
		}
		mock.ExpectExec(regexp.QuoteMeta(sqlRevokeSessionByRefresh)).WithArgs(authTokenSHA256Hex(raw)).WillReturnResult(sqlmock.NewResult(0, 1))
		req := mustJSONRequest(t, http.MethodPost, "/api/auth/logout", map[string]string{"refreshToken": raw})
		req.Header.Set("Authorization", "Bearer "+expired)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
		}
	})
	t.Run("bearer sid fallback", func(t *testing.T) {
		expectAuthLookup(mock, testUserID, testSessionID, user.UpdatedAt, "active", true)
		mock.ExpectExec(regexp.QuoteMeta(sqlRevokeSession)).WithArgs(testSessionID).WillReturnResult(sqlmock.NewResult(0, 1))
		req := mustJSONRequest(t, http.MethodPost, "/api/auth/logout", map[string]string{})
		req.Header.Set("Authorization", "Bearer "+mustSessionToken(t, authManager, user, testSessionID))
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
		}
	})
	t.Run("nothing to revoke", func(t *testing.T) {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/logout", map[string]string{}))
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401 with no credentials, got %d", rec.Code)
		}
		mock.ExpectExec(regexp.QuoteMeta(sqlRevokeSessionByRefresh)).WithArgs(sqlmock.AnyArg()).WillReturnResult(sqlmock.NewResult(0, 0))
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, mustJSONRequest(t, http.MethodPost, "/api/auth/logout", map[string]string{"refreshToken": "stale"}))
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401 for an unknown refresh token, got %d", rec.Code)
		}
	})
	t.Run("logged-out access token is rejected everywhere", func(t *testing.T) {
		expectAuthLookup(mock, testUserID, testSessionID, user.UpdatedAt, "active", false)
		req := httptest.NewRequest(http.MethodGet, "/api/auth/me", nil)
		req.Header.Set("Authorization", "Bearer "+mustSessionToken(t, authManager, user, testSessionID))
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusUnauthorized {
			t.Fatalf("expected 401 after logout, got %d (%s)", rec.Code, rec.Body.String())
		}
	})
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// --- /api/auth/sessions, /api/cluster/admissions ---

func TestListSessionsHandlerScopedToCaller(t *testing.T) {
	router, mock, authManager := newAuthSessionTestRouter(t)
	now := time.Now().UTC()
	past := now.Add(-time.Hour)

	expectAuthLookup(mock, "alice", "", past, "active", false)
	mock.ExpectQuery(regexp.QuoteMeta(sqlListUserSessions)).WithArgs("alice").
		WillReturnRows(sqlmock.NewRows([]string{"id", "created_at", "expires_at", "last_accessed_at"}).
			AddRow(testSessionID, now, now.Add(time.Hour), now).
			AddRow(testSessionID2, past, now.Add(time.Hour), nil))
	req := httptest.NewRequest(http.MethodGet, "/api/auth/sessions?user_id=bob", nil)
	req.Header.Set("Authorization", signedBearerToken(t, authManager, "alice", "default", "admin"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var sessions []SessionResponse
	decodeJSONBody(t, rec, &sessions)
	if len(sessions) != 2 || sessions[0].ID != testSessionID || sessions[1].LastAccessedAt != sessions[1].CreatedAt {
		t.Fatalf("unexpected sessions: %#v", sessions)
	}

	expectAuthLookup(mock, "bob", "", past, "active", false)
	mock.ExpectQuery(regexp.QuoteMeta(sqlListUserSessions)).WithArgs("bob").
		WillReturnRows(sqlmock.NewRows([]string{"id", "created_at", "expires_at", "last_accessed_at"}))
	req = httptest.NewRequest(http.MethodGet, "/api/auth/sessions", nil)
	req.Header.Set("Authorization", signedBearerToken(t, authManager, "bob", "default", "viewer"))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	sessions = nil
	decodeJSONBody(t, rec, &sessions)
	if len(sessions) != 0 {
		t.Fatalf("expected bob to see no sessions, got %#v", sessions)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestClusterAdmissionsAndSelect(t *testing.T) {
	router, mock, authManager := newAuthSessionTestRouter(t)
	user := testUser()
	token := mustSessionToken(t, authManager, user, testSessionID)

	expectAuthLookup(mock, testUserID, testSessionID, user.UpdatedAt, "active", true)
	expectGetUser(mock, user)
	req := httptest.NewRequest(http.MethodGet, "/api/cluster/admissions", nil)
	req.Header.Set("Authorization", "Bearer "+token)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var admissions []AdmissionResponse
	decodeJSONBody(t, rec, &admissions)
	if len(admissions) != 1 || !admissions[0].Admitted || admissions[0].ClusterID != selfNodeID() || admissions[0].Role != "admin" {
		t.Fatalf("expected the single local admission, got %#v", admissions)
	}

	expectAuthLookup(mock, testUserID, testSessionID, user.UpdatedAt, "active", true)
	req = mustJSONRequest(t, http.MethodPost, "/api/cluster/admissions/select", map[string]string{"clusterId": "elsewhere"})
	req.Header.Set("Authorization", "Bearer "+token)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for an unknown cluster, got %d (%s)", rec.Code, rec.Body.String())
	}

	expectAuthLookup(mock, testUserID, testSessionID, user.UpdatedAt, "active", true)
	expectGetUser(mock, user)
	expectFetchSession(mock, testSessionID, testUserID)
	mock.ExpectExec(regexp.QuoteMeta(sqlTouchSessionLastAccessed)).WithArgs(testSessionID).WillReturnResult(sqlmock.NewResult(0, 1))
	req = mustJSONRequest(t, http.MethodPost, "/api/cluster/admissions/select", map[string]string{"clusterId": selfNodeID()})
	req.Header.Set("Authorization", "Bearer "+token)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var payload currentUserPayload
	decodeJSONBody(t, rec, &payload)
	assertCurrentUserShape(t, payload, testSessionID)
	if payload.Session.SelectedClusterID != selfNodeID() {
		t.Fatalf("expected selectedClusterId %q, got %q", selfNodeID(), payload.Session.SelectedClusterID)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

// --- schema gate and sweeper ---

func TestRequireMigratedSchemaGatesOnMigrationVersion(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cases := []struct {
		name    string
		version int64
		dirty   bool
		wantErr bool
	}{
		{"behind", requiredSchemaVersion - 1, false, true},
		{"dirty", requiredSchemaVersion, true, true},
		{"current", requiredSchemaVersion, false, false},
		{"ahead", requiredSchemaVersion + 3, false, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mock.ExpectQuery(regexp.QuoteMeta(sqlSchemaMigrationState)).
				WillReturnRows(sqlmock.NewRows([]string{"version", "dirty"}).AddRow(tc.version, tc.dirty))
			err := requireMigratedSchema(db)
			if (err != nil) != tc.wantErr {
				t.Fatalf("version=%d dirty=%v: got err=%v wantErr=%v", tc.version, tc.dirty, err, tc.wantErr)
			}
		})
	}
	t.Run("no schema_migrations rows", func(t *testing.T) {
		mock.ExpectQuery(regexp.QuoteMeta(sqlSchemaMigrationState)).WillReturnRows(sqlmock.NewRows([]string{"version", "dirty"}))
		if err := requireMigratedSchema(db); err == nil {
			t.Fatal("expected an error for an empty schema_migrations table")
		}
	})
	t.Run("table missing", func(t *testing.T) {
		mock.ExpectQuery(regexp.QuoteMeta(sqlSchemaMigrationState)).WillReturnError(errors.New(`relation "schema_migrations" does not exist`))
		if err := requireMigratedSchema(db); err == nil {
			t.Fatal("expected an error when schema_migrations is missing")
		}
	})
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet sql expectations: %v", err)
	}
}

func TestSessionSweeperDeletesExpiredRows(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	mock.ExpectExec(regexp.QuoteMeta(sqlSweepExpiredSessions)).WillReturnResult(sqlmock.NewResult(0, 3))
	mock.ExpectExec(regexp.QuoteMeta(sqlSweepExpiredSessions)).WillReturnResult(sqlmock.NewResult(0, 0))
	mock.MatchExpectationsInOrder(true)

	stop := startSessionSweeper(context.Background(), db, 20*time.Millisecond)
	deadline := time.Now().Add(2 * time.Second)
	for mock.ExpectationsWereMet() != nil && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	stop()
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("sweeper did not run as expected: %v", err)
	}
}
