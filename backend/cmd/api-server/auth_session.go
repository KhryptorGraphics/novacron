package main

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/netip"
	"os"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/netutil"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// SQL text lives in consts so the implementation and its sqlmock tests
// reference the exact same string.
const (
	// sqlAuthenticateTokenLookup folds the two per-request revocation checks
	// into one indexed round trip: the user's updated_at (profile/password
	// change after the token was issued) and, when the token names a session
	// ("sid" claim), whether that session row still exists, is unrevoked and
	// unexpired. s.id comes back NULL for a token without a sid (nothing to
	// check) and for a sid whose row is gone/revoked (reject).
	sqlAuthenticateTokenLookup = `
		SELECT u.updated_at, u.status, s.id
		FROM users u
		LEFT JOIN sessions s
		  ON s.user_id = u.id
		 AND s.id = NULLIF($2, '')::uuid
		 AND s.revoked_at IS NULL
		 AND s.expires_at > NOW()
		WHERE u.id = $1
	`
	sqlCreateSession = `
		INSERT INTO sessions (user_id, refresh_token_hash, ip_address, user_agent, expires_at, created_at, last_accessed_at)
		VALUES ($1, $2, NULLIF($3, '')::inet, NULLIF($4, ''), $5, NOW(), NOW())
		RETURNING id, created_at, expires_at, last_accessed_at
	`
	sqlFetchSession = `
		SELECT id, user_id, created_at, expires_at, last_accessed_at
		FROM sessions
		WHERE id = $1 AND revoked_at IS NULL AND expires_at > NOW()
	`
	sqlTouchSessionLastAccessed = `UPDATE sessions SET last_accessed_at = NOW() WHERE id = $1 AND revoked_at IS NULL`
	sqlRevokeSession            = `UPDATE sessions SET revoked_at = NOW() WHERE id = $1 AND revoked_at IS NULL`
	sqlRevokeSessionByRefresh   = `
		UPDATE sessions SET revoked_at = NOW()
		WHERE (refresh_token_hash = $1 OR previous_refresh_token_hash = $1) AND revoked_at IS NULL
	`
	sqlListUserSessions = `
		SELECT id, created_at, expires_at, last_accessed_at
		FROM sessions
		WHERE user_id = $1 AND revoked_at IS NULL AND expires_at > NOW()
		ORDER BY created_at DESC
	`
	sqlLookupSessionByRefreshHash = `
		SELECT id, user_id, created_at, revoked_at, expires_at, (refresh_token_hash = $1) AS is_current
		FROM sessions
		WHERE refresh_token_hash = $1 OR previous_refresh_token_hash = $1
		LIMIT 1
	`
	// Compare-and-swap on the old hash: of two concurrent refreshes with the
	// same token exactly one wins; the loser sees zero rows and fails closed.
	sqlRotateRefreshToken = `
		UPDATE sessions
		SET previous_refresh_token_hash = refresh_token_hash,
		    refresh_token_hash = $1,
		    expires_at = $2,
		    last_accessed_at = NOW()
		WHERE id = $3 AND revoked_at IS NULL AND refresh_token_hash = $4
		RETURNING id, user_id, created_at, expires_at, last_accessed_at
	`
	sqlSweepExpiredSessions = `DELETE FROM sessions WHERE expires_at < NOW()`
)

var errInvalidRefreshToken = errors.New("invalid or expired refresh token")

const (
	// refreshTokenTTL bounds a session: the refresh token (and with it the
	// sessions row) lives this long past its last rotation.
	refreshTokenTTL = 7 * 24 * time.Hour
	// accessTokenTTL mirrors issueSessionToken's exp claim.
	accessTokenTTL = 24 * time.Hour
	// sessionSweepInterval is how often expired session rows are deleted.
	sessionSweepInterval = time.Hour
)

// trustedProxiesFromEnv parses NOVACRON_TRUSTED_PROXIES once per route
// registration (the same variable the login rate limiter reads).
func trustedProxiesFromEnv() []netip.Prefix {
	return netutil.ParseTrustedProxies(os.Getenv("NOVACRON_TRUSTED_PROXIES"))
}

// authenticateToken is the single token-authentication core shared by
// requireAuth (Authorization header) and the WebSocket subprotocol adapter.
// It validates an HS256 access token and returns a context carrying the
// verified principal (user_id, tenant_id, organization_id, role, roles,
// session_id), or an HTTP status plus error. It fails closed:
//   - bad signature / expired / malformed claims → 401;
//   - any non-empty "purpose" claim (e.g. pending_2fa) → 401: only plain
//     access tokens are accepted here, whatever the caller;
//   - missing "iat" → 401 (the revocation rule cannot be evaluated);
//   - user row missing or not active → 401; DB error → 503;
//   - users.updated_at after iat (password reset, role change, admin edit,
//     deactivation — the BEFORE UPDATE trigger bumps it) → 401;
//   - a "sid" claim whose session row is missing, revoked or expired → 401.
//     Tokens without a sid (issued before sessions existed) skip only the
//     session check.
//
// A nil db (unit tests that exercise routing only; main() never builds one)
// skips the database checks.
func authenticateToken(ctx context.Context, authManager *auth.SimpleAuthManager, db *sql.DB, tokenString string) (context.Context, int, error) {
	claims, err := validateJWT(tokenString, authManager.GetJWTSecret())
	if err != nil {
		return nil, http.StatusUnauthorized, errors.New("invalid or expired token")
	}
	if purpose := stringClaim(claims, "purpose"); purpose != "" {
		if purpose == "pending_2fa" {
			return nil, http.StatusUnauthorized, errors.New("two-factor authentication is not complete")
		}
		return nil, http.StatusUnauthorized, errors.New("token is not an access token")
	}

	userID := stringClaim(claims, "user_id", "sub")
	if userID == "" {
		return nil, http.StatusUnauthorized, errors.New("token missing user identity")
	}
	issuedAt, err := claims.GetIssuedAt()
	if err != nil || issuedAt == nil {
		return nil, http.StatusUnauthorized, errors.New("token missing issued-at claim")
	}

	sessionID := stringClaim(claims, "sid")
	if sessionID != "" {
		if _, err := uuid.Parse(sessionID); err != nil {
			return nil, http.StatusUnauthorized, errors.New("token session is invalid")
		}
	}

	if db != nil {
		var updatedAt time.Time
		var status string
		var matchedSession sql.NullString
		err := db.QueryRowContext(ctx, sqlAuthenticateTokenLookup, userID, sessionID).Scan(&updatedAt, &status, &matchedSession)
		switch {
		case errors.Is(err, sql.ErrNoRows):
			return nil, http.StatusUnauthorized, errors.New("token user no longer exists")
		case err != nil:
			return nil, http.StatusServiceUnavailable, fmt.Errorf("authentication lookup failed: %w", err)
		}
		if auth.UserStatus(status) != auth.UserStatusActive {
			return nil, http.StatusUnauthorized, errors.New("user account is not active")
		}
		// iat has second precision; updated_at is a µs timestamp. Truncate the
		// comparison to the second so a row written in the same second the
		// token was minted does not revoke it spuriously.
		if updatedAt.UTC().Truncate(time.Second).After(issuedAt.Time.UTC()) {
			return nil, http.StatusUnauthorized, errors.New("token revoked (session invalidated by profile change)")
		}
		if sessionID != "" && !matchedSession.Valid {
			return nil, http.StatusUnauthorized, errors.New("token revoked (session logged out)")
		}
	}

	ctx = context.WithValue(ctx, "user_id", userID)
	ctx = context.WithValue(ctx, "tenant_id", stringClaim(claims, "tenant_id"))
	ctx = context.WithValue(ctx, "organization_id", stringClaim(claims, "tenant_id"))
	ctx = context.WithValue(ctx, "role", stringClaim(claims, "role"))
	ctx = context.WithValue(ctx, "roles", stringSliceClaim(claims, "roles"))
	ctx = context.WithValue(ctx, "session_id", sessionID)
	return ctx, http.StatusOK, nil
}

// --- Session persistence ---

type sessionMeta struct {
	ID             string
	UserID         string
	CreatedAt      time.Time
	ExpiresAt      time.Time
	LastAccessedAt time.Time
}

func generateOpaqueToken(nBytes int) (string, error) {
	raw := make([]byte, nBytes)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("generate token: %w", err)
	}
	return hex.EncodeToString(raw), nil
}

// createSession mints a DB-backed session for userID: an opaque 32-byte
// refresh token (returned raw, stored only as its sha256 hash) and the
// sessions row the access token's "sid" claim points at. clientIP is a bare
// IP literal from netutil.ClientIPString ("" stores NULL).
func createSession(ctx context.Context, db *sql.DB, userID, clientIP, userAgent string) (sessionMeta, string, error) {
	rawRefresh, err := generateOpaqueToken(32)
	if err != nil {
		return sessionMeta{}, "", err
	}
	refreshHash := authTokenSHA256Hex(rawRefresh)
	refreshExpiresAt := time.Now().UTC().Add(refreshTokenTTL)

	meta := sessionMeta{UserID: userID}
	var lastAccessed sql.NullTime
	err = db.QueryRowContext(ctx, sqlCreateSession, userID, refreshHash, clientIP, truncateUserAgent(userAgent), refreshExpiresAt).
		Scan(&meta.ID, &meta.CreatedAt, &meta.ExpiresAt, &lastAccessed)
	if err != nil {
		return sessionMeta{}, "", fmt.Errorf("create session: %w", err)
	}
	meta.LastAccessedAt = meta.CreatedAt
	if lastAccessed.Valid {
		meta.LastAccessedAt = lastAccessed.Time
	}
	return meta, rawRefresh, nil
}

func truncateUserAgent(ua string) string {
	ua = strings.TrimSpace(ua)
	if len(ua) > 512 {
		return ua[:512]
	}
	return ua
}

// fetchSession loads an active session by id. ok=false with a nil error means
// no such live session (unknown/revoked/expired id, or an empty id from a
// token without a "sid" claim); callers must not fabricate session data.
func fetchSession(ctx context.Context, db *sql.DB, sessionID string) (sessionMeta, bool, error) {
	if db == nil || strings.TrimSpace(sessionID) == "" {
		return sessionMeta{}, false, nil
	}
	var meta sessionMeta
	var lastAccessed sql.NullTime
	err := db.QueryRowContext(ctx, sqlFetchSession, sessionID).
		Scan(&meta.ID, &meta.UserID, &meta.CreatedAt, &meta.ExpiresAt, &lastAccessed)
	if errors.Is(err, sql.ErrNoRows) {
		return sessionMeta{}, false, nil
	}
	if err != nil {
		return sessionMeta{}, false, err
	}
	meta.LastAccessedAt = meta.CreatedAt
	if lastAccessed.Valid {
		meta.LastAccessedAt = lastAccessed.Time
	}
	return meta, true, nil
}

func touchSessionLastAccessed(ctx context.Context, db *sql.DB, sessionID string) {
	if db == nil || strings.TrimSpace(sessionID) == "" {
		return
	}
	if _, err := db.ExecContext(ctx, sqlTouchSessionLastAccessed, sessionID); err != nil {
		log.Printf("session %s: last_accessed_at not updated: %v", sessionID, err)
	}
}

func revokeSession(ctx context.Context, db *sql.DB, sessionID string) error {
	if db == nil || strings.TrimSpace(sessionID) == "" {
		return nil
	}
	_, err := db.ExecContext(ctx, sqlRevokeSession, sessionID)
	return err
}

// revokeSessionByRefreshToken revokes the session holding rawRefreshToken
// (current or immediately-prior generation). Returns whether a row changed.
func revokeSessionByRefreshToken(ctx context.Context, db *sql.DB, rawRefreshToken string) (bool, error) {
	res, err := db.ExecContext(ctx, sqlRevokeSessionByRefresh, authTokenSHA256Hex(rawRefreshToken))
	if err != nil {
		return false, err
	}
	n, err := res.RowsAffected()
	if err != nil {
		// Driver without RowsAffected support: the UPDATE itself succeeded.
		return true, nil
	}
	return n > 0, nil
}

func listUserSessions(ctx context.Context, db *sql.DB, userID string) ([]SessionResponse, error) {
	rows, err := db.QueryContext(ctx, sqlListUserSessions, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	sessions := make([]SessionResponse, 0)
	for rows.Next() {
		var id string
		var createdAt, expiresAt time.Time
		var lastAccessed sql.NullTime
		if err := rows.Scan(&id, &createdAt, &expiresAt, &lastAccessed); err != nil {
			return nil, err
		}
		resp := SessionResponse{
			ID:        id,
			CreatedAt: createdAt.UTC().Format(time.RFC3339),
			ExpiresAt: expiresAt.UTC().Format(time.RFC3339),
		}
		resp.LastAccessedAt = resp.CreatedAt
		if lastAccessed.Valid {
			resp.LastAccessedAt = lastAccessed.Time.UTC().Format(time.RFC3339)
		}
		sessions = append(sessions, resp)
	}
	return sessions, rows.Err()
}

type rotatedSession struct {
	Meta            sessionMeta
	NewRefreshToken string
	User            *auth.User
}

// refreshSession implements refresh-token rotation with one-generation reuse
// detection, re-reading the user row so role and tenant come from the
// database rather than the old token:
//   - presenting the CURRENT refresh token of a live session rotates it
//     (compare-and-swap on the old hash);
//   - presenting the immediately-prior (already rotated-out) token means the
//     token leaked: the whole session is revoked;
//   - a revoked or expired session, an unknown token, a user that is no
//     longer active, or a user row changed since the session was issued all
//     fail with errInvalidRefreshToken (the last two also revoke the session).
func refreshSession(ctx context.Context, db *sql.DB, authManager *auth.SimpleAuthManager, rawRefreshToken string) (rotatedSession, error) {
	hash := authTokenSHA256Hex(rawRefreshToken)

	var sessionID, userID string
	var createdAt, expiresAt time.Time
	var revokedAt sql.NullTime
	var isCurrent bool
	err := db.QueryRowContext(ctx, sqlLookupSessionByRefreshHash, hash).
		Scan(&sessionID, &userID, &createdAt, &revokedAt, &expiresAt, &isCurrent)
	if errors.Is(err, sql.ErrNoRows) {
		return rotatedSession{}, errInvalidRefreshToken
	}
	if err != nil {
		return rotatedSession{}, err
	}
	if revokedAt.Valid {
		return rotatedSession{}, errInvalidRefreshToken
	}
	if !isCurrent {
		// Reuse of a rotated-out token: treat the session as compromised and
		// kill it — both the live refresh token and every access token
		// carrying its sid.
		if err := revokeSession(ctx, db, sessionID); err != nil {
			return rotatedSession{}, err
		}
		return rotatedSession{}, errInvalidRefreshToken
	}
	if time.Now().After(expiresAt) {
		return rotatedSession{}, errInvalidRefreshToken
	}

	user, err := authManager.GetUser(userID)
	if err != nil {
		return rotatedSession{}, errInvalidRefreshToken
	}
	// Same revocation rule as authenticateToken: a deactivated user, or a
	// user row changed after this session was issued (password reset, role
	// change, admin edit), ends the session instead of renewing it.
	if user.Status != auth.UserStatusActive || user.UpdatedAt.UTC().Truncate(time.Second).After(createdAt.UTC()) {
		if err := revokeSession(ctx, db, sessionID); err != nil {
			return rotatedSession{}, err
		}
		return rotatedSession{}, errInvalidRefreshToken
	}

	newRaw, err := generateOpaqueToken(32)
	if err != nil {
		return rotatedSession{}, err
	}
	newExpiresAt := time.Now().UTC().Add(refreshTokenTTL)

	var meta sessionMeta
	var lastAccessed sql.NullTime
	err = db.QueryRowContext(ctx, sqlRotateRefreshToken, authTokenSHA256Hex(newRaw), newExpiresAt, sessionID, hash).
		Scan(&meta.ID, &meta.UserID, &meta.CreatedAt, &meta.ExpiresAt, &lastAccessed)
	if errors.Is(err, sql.ErrNoRows) {
		// Lost the compare-and-swap (concurrent refresh) or revoked meanwhile.
		return rotatedSession{}, errInvalidRefreshToken
	}
	if err != nil {
		return rotatedSession{}, err
	}
	meta.LastAccessedAt = meta.CreatedAt
	if lastAccessed.Valid {
		meta.LastAccessedAt = lastAccessed.Time
	}
	return rotatedSession{Meta: meta, NewRefreshToken: newRaw, User: user}, nil
}

// startSessionSweeper deletes sessions whose refresh token has expired
// (expires_at < now()) every interval until ctx is cancelled or the returned
// stop func runs. Revoked rows are kept until they expire so reuse of a
// rotated token is still recognized as such.
func startSessionSweeper(ctx context.Context, db *sql.DB, interval time.Duration) func() {
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			sweepExpiredSessions(ctx, db)
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
		}
	}()
	return func() {
		cancel()
		<-done
	}
}

func sweepExpiredSessions(ctx context.Context, db *sql.DB) {
	res, err := db.ExecContext(ctx, sqlSweepExpiredSessions)
	if err != nil {
		if ctx.Err() == nil {
			log.Printf("session sweep failed: %v", err)
		}
		return
	}
	if n, err := res.RowsAffected(); err == nil && n > 0 {
		log.Printf("session sweep: removed %d expired session(s)", n)
	}
}

// --- Response shapes (mirror frontend/src/lib/auth.ts field-for-field) ---

// ClusterSummaryResponse describes a cluster for API responses. api-server
// serves exactly one cluster (this node's local fabric, cluster.go's
// selfNodeID()); fields with no measured source here (tier,
// performance/interconnect scoring, growth/federation state, degraded,
// last-evaluated timestamp, max node cap) are omitted rather than invented.
type ClusterSummaryResponse struct {
	ID                        string     `json:"id"`
	Name                      string     `json:"name"`
	Tier                      string     `json:"tier,omitempty"`
	PerformanceScore          *float64   `json:"performanceScore,omitempty"`
	InterconnectLatencyMs     *float64   `json:"interconnectLatencyMs,omitempty"`
	InterconnectBandwidthMbps *float64   `json:"interconnectBandwidthMbps,omitempty"`
	CurrentNodeCount          int        `json:"currentNodeCount"`
	MaxSupportedNodeCount     *int       `json:"maxSupportedNodeCount,omitempty"`
	GrowthState               string     `json:"growthState,omitempty"`
	FederationState           string     `json:"federationState,omitempty"`
	Degraded                  *bool      `json:"degraded,omitempty"`
	LastEvaluatedAt           *time.Time `json:"lastEvaluatedAt,omitempty"`
	EdgeLatencyMs             *float64   `json:"edgeLatencyMs,omitempty"`
	EdgeBandwidthMbps         *float64   `json:"edgeBandwidthMbps,omitempty"`
}

// AdmissionResponse mirrors AdmissionResponse in frontend/src/lib/auth.ts.
// Every active user has exactly one membership: the local fabric.
type AdmissionResponse struct {
	Admitted   bool                    `json:"admitted"`
	State      string                  `json:"state,omitempty"`
	ClusterID  string                  `json:"clusterId,omitempty"`
	Role       string                  `json:"role,omitempty"`
	Source     string                  `json:"source,omitempty"`
	AdmittedAt string                  `json:"admittedAt,omitempty"`
	TenantID   string                  `json:"tenantId,omitempty"`
	Selected   bool                    `json:"selected,omitempty"`
	Cluster    *ClusterSummaryResponse `json:"cluster,omitempty"`
}

// SessionResponse mirrors SessionResponse in frontend/src/lib/auth.ts.
type SessionResponse struct {
	ID                string `json:"id"`
	ExpiresAt         string `json:"expiresAt"`
	CreatedAt         string `json:"createdAt"`
	LastAccessedAt    string `json:"lastAccessedAt"`
	SelectedClusterID string `json:"selectedClusterId,omitempty"`
}

// meResponsePayload mirrors CurrentUserResponse in frontend/src/lib/auth.ts.
type meResponsePayload struct {
	User            map[string]interface{} `json:"user"`
	Admission       AdmissionResponse       `json:"admission"`
	Memberships     []AdmissionResponse     `json:"memberships"`
	SelectedCluster *ClusterSummaryResponse `json:"selectedCluster,omitempty"`
	Session         SessionResponse         `json:"session"`
}

// authResponsePayload mirrors AuthResponse in frontend/src/lib/auth.ts.
type authResponsePayload struct {
	Token                string `json:"token"`
	RefreshToken         string `json:"refreshToken,omitempty"`
	ExpiresAt            string `json:"expiresAt"`
	RemainingBackupCodes *int   `json:"remaining_backup_codes,omitempty"`
	meResponsePayload
}

// localClusterSummary describes api-server's one cluster: this node's local
// fabric. currentNodeCount is the number of nodes this node actually knows
// (itself plus registered migration peers) — no probe is made.
func localClusterSummary(vmManager *core_vm.VMManager) ClusterSummaryResponse {
	nodeCount := 1
	if vmManager != nil {
		nodeCount += len(vmManager.MigrationPeers())
	}
	id := selfNodeID()
	return ClusterSummaryResponse{ID: id, Name: id, CurrentNodeCount: nodeCount}
}

func buildLocalMembership(user *auth.User, clusterSummary ClusterSummaryResponse) AdmissionResponse {
	return AdmissionResponse{
		Admitted:   true,
		State:      "active",
		ClusterID:  clusterSummary.ID,
		Role:       primaryRole(user),
		Source:     "local-fabric",
		AdmittedAt: user.CreatedAt.UTC().Format(time.RFC3339),
		TenantID:   defaultTenantLabel(user.TenantID),
		Selected:   true,
		Cluster:    &clusterSummary,
	}
}

func sessionResponse(sess sessionMeta, selectedClusterID string) SessionResponse {
	return SessionResponse{
		ID:                sess.ID,
		ExpiresAt:         sess.ExpiresAt.UTC().Format(time.RFC3339),
		CreatedAt:         sess.CreatedAt.UTC().Format(time.RFC3339),
		LastAccessedAt:    sess.LastAccessedAt.UTC().Format(time.RFC3339),
		SelectedClusterID: selectedClusterID,
	}
}

func buildMeResponsePayload(user *auth.User, sess sessionMeta, vmManager *core_vm.VMManager, twoFactorService *auth.TwoFactorService) meResponsePayload {
	clusterSummary := localClusterSummary(vmManager)
	membership := buildLocalMembership(user, clusterSummary)
	userPayload := frontendUser(user)
	userPayload["two_factor_enabled"] = userHasEnabledTwoFactor(twoFactorService, user.ID)
	return meResponsePayload{
		User:            userPayload,
		Admission:       membership,
		Memberships:     []AdmissionResponse{membership},
		SelectedCluster: &clusterSummary,
		Session:         sessionResponse(sess, clusterSummary.ID),
	}
}

// issueAuthResponse creates the session row plus access token for user and
// writes the frontend AuthResponse. Used by login and 2FA verify-login.
func issueAuthResponse(w http.ResponseWriter, r *http.Request, authManager *auth.SimpleAuthManager, db *sql.DB, vmManager *core_vm.VMManager, twoFactorService *auth.TwoFactorService, trustedProxies []netip.Prefix, user *auth.User, remainingBackupCodes *int) {
	sess, refreshRaw, err := createSession(r.Context(), db, user.ID, netutil.ClientIPString(r, trustedProxies), r.UserAgent())
	if err != nil {
		log.Printf("login: session for user %s not created: %v", user.ID, err)
		writeJSONError(w, http.StatusInternalServerError, "failed to create session")
		return
	}
	accessToken, err := issueSessionToken(authManager.GetJWTSecret(), user, sess.ID)
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "failed to create session token")
		return
	}
	writeJSON(w, http.StatusOK, authResponsePayload{
		Token:                accessToken,
		RefreshToken:         refreshRaw,
		ExpiresAt:            time.Now().UTC().Add(accessTokenTTL).Format(time.RFC3339),
		RemainingBackupCodes: remainingBackupCodes,
		meResponsePayload:    buildMeResponsePayload(user, sess, vmManager, twoFactorService),
	})
}

// --- Route handlers ---

// registerAuthSessionRoutes mounts the session-backed part of the frontend
// auth contract. /api/auth/refresh and /api/auth/logout are deliberately
// public: refresh is called exactly when the access token is missing or
// expired, and logout must still revoke the server session when only the
// refresh token is still valid.
func registerAuthSessionRoutes(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB, vmManager *core_vm.VMManager, twoFactorService *auth.TwoFactorService) {
	protect := requireAuth(authManager, db)

	router.Handle("/api/auth/me", protect(meHandler(authManager, db, vmManager, twoFactorService))).Methods(http.MethodGet)
	router.Handle("/api/auth/refresh", refreshHandler(authManager, db, vmManager, twoFactorService)).Methods(http.MethodPost)
	router.Handle("/api/auth/logout", logoutHandler(authManager, db)).Methods(http.MethodPost)
	router.Handle("/api/auth/sessions", protect(listSessionsHandler(db))).Methods(http.MethodGet)
	router.Handle("/api/cluster/admissions", protect(admissionsHandler(authManager, vmManager))).Methods(http.MethodGet)
	router.Handle("/api/cluster/admissions/select", protect(selectClusterHandler(authManager, db, vmManager, twoFactorService))).Methods(http.MethodPost)
}

func contextUserID(r *http.Request) string {
	userID, _ := r.Context().Value("user_id").(string)
	return userID
}

func contextSessionID(r *http.Request) string {
	sessionID, _ := r.Context().Value("session_id").(string)
	return sessionID
}

// loadAuthenticatedSession resolves the caller's user and live session for
// the handlers whose contract requires a real session row. A token without a
// sid claim (issued before sessions existed) passes requireAuth everywhere
// else but is rejected here: there is no honest session to report.
func loadAuthenticatedSession(w http.ResponseWriter, r *http.Request, authManager *auth.SimpleAuthManager, db *sql.DB) (*auth.User, sessionMeta, bool) {
	userID := contextUserID(r)
	if userID == "" {
		writeJSONError(w, http.StatusUnauthorized, "authentication required")
		return nil, sessionMeta{}, false
	}
	user, err := authManager.GetUser(userID)
	if err != nil {
		writeJSONError(w, http.StatusUnauthorized, "user not found")
		return nil, sessionMeta{}, false
	}
	sess, ok, err := fetchSession(r.Context(), db, contextSessionID(r))
	if err != nil {
		writeJSONError(w, http.StatusInternalServerError, "failed to load session")
		return nil, sessionMeta{}, false
	}
	if !ok {
		writeJSONError(w, http.StatusUnauthorized, "session not found; sign in again")
		return nil, sessionMeta{}, false
	}
	return user, sess, true
}

func meHandler(authManager *auth.SimpleAuthManager, db *sql.DB, vmManager *core_vm.VMManager, twoFactorService *auth.TwoFactorService) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, sess, ok := loadAuthenticatedSession(w, r, authManager, db)
		if !ok {
			return
		}
		touchSessionLastAccessed(r.Context(), db, sess.ID)
		writeJSON(w, http.StatusOK, buildMeResponsePayload(user, sess, vmManager, twoFactorService))
	})
}

func refreshHandler(authManager *auth.SimpleAuthManager, db *sql.DB, vmManager *core_vm.VMManager, twoFactorService *auth.TwoFactorService) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			RefreshToken string `json:"refreshToken"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		raw := strings.TrimSpace(req.RefreshToken)
		if raw == "" {
			writeJSONError(w, http.StatusBadRequest, "refreshToken is required")
			return
		}
		rotated, err := refreshSession(r.Context(), db, authManager, raw)
		if errors.Is(err, errInvalidRefreshToken) {
			writeJSONError(w, http.StatusUnauthorized, errInvalidRefreshToken.Error())
			return
		}
		if err != nil {
			log.Printf("refresh: %v", err)
			writeJSONError(w, http.StatusServiceUnavailable, "failed to refresh session")
			return
		}
		accessToken, err := issueSessionToken(authManager.GetJWTSecret(), rotated.User, rotated.Meta.ID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to create session token")
			return
		}
		writeJSON(w, http.StatusOK, authResponsePayload{
			Token:             accessToken,
			RefreshToken:      rotated.NewRefreshToken,
			ExpiresAt:         time.Now().UTC().Add(accessTokenTTL).Format(time.RFC3339),
			meResponsePayload: buildMeResponsePayload(rotated.User, rotated.Meta, vmManager, twoFactorService),
		})
	})
}

// logoutHandler revokes the caller's session server-side. The refresh token
// in the body is the primary handle (possessing it is the credential, and it
// outlives the access token); the bearer token's sid is the fallback for
// clients that only hold an access token. Both are revoked when both are
// present. Nothing to revoke → 401 rather than a hollow success.
func logoutHandler(authManager *auth.SimpleAuthManager, db *sql.DB) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			RefreshToken string `json:"refreshToken"`
		}
		if r.Body != nil && r.ContentLength != 0 {
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				writeJSONError(w, http.StatusBadRequest, "invalid request body")
				return
			}
		}

		revoked := false
		if raw := strings.TrimSpace(req.RefreshToken); raw != "" {
			changed, err := revokeSessionByRefreshToken(r.Context(), db, raw)
			if err != nil {
				log.Printf("logout: revoke by refresh token: %v", err)
				writeJSONError(w, http.StatusServiceUnavailable, "failed to revoke session")
				return
			}
			revoked = revoked || changed
		}

		if header := r.Header.Get("Authorization"); header != "" {
			status, err := revokeBearerSession(r.Context(), authManager, db, header)
			switch {
			case err == nil:
				revoked = true
			case status == http.StatusServiceUnavailable:
				writeJSONError(w, status, "failed to revoke session")
				return
			case !revoked:
				// An expired/revoked access token with no refresh token: nothing
				// left to revoke on the server, and nothing to prove who asked.
				writeJSONError(w, http.StatusUnauthorized, err.Error())
				return
			}
		}

		if !revoked {
			writeJSONError(w, http.StatusUnauthorized, "no active session to revoke")
			return
		}
		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})
}

// revokeBearerSession revokes the session named by a valid bearer access
// token's sid claim. A token without a sid (pre-session issue) has nothing
// to revoke server-side and reports 401.
func revokeBearerSession(ctx context.Context, authManager *auth.SimpleAuthManager, db *sql.DB, header string) (int, error) {
	tokenString, err := extractBearerToken(header)
	if err != nil {
		return http.StatusUnauthorized, err
	}
	authCtx, status, err := authenticateToken(ctx, authManager, db, tokenString)
	if err != nil {
		return status, err
	}
	sid, _ := authCtx.Value("session_id").(string)
	if sid == "" {
		return http.StatusUnauthorized, errors.New("token carries no server session")
	}
	if err := revokeSession(ctx, db, sid); err != nil {
		return http.StatusServiceUnavailable, err
	}
	return http.StatusOK, nil
}

func listSessionsHandler(db *sql.DB) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		userID := contextUserID(r)
		if userID == "" {
			writeJSONError(w, http.StatusUnauthorized, "authentication required")
			return
		}
		sessions, err := listUserSessions(r.Context(), db, userID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to list sessions")
			return
		}
		writeJSON(w, http.StatusOK, sessions)
	})
}

func admissionsHandler(authManager *auth.SimpleAuthManager, vmManager *core_vm.VMManager) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		user, err := authManager.GetUser(contextUserID(r))
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "user not found")
			return
		}
		writeJSON(w, http.StatusOK, []AdmissionResponse{buildLocalMembership(user, localClusterSummary(vmManager))})
	})
}

func selectClusterHandler(authManager *auth.SimpleAuthManager, db *sql.DB, vmManager *core_vm.VMManager, twoFactorService *auth.TwoFactorService) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			ClusterID string `json:"clusterId"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if strings.TrimSpace(req.ClusterID) != localClusterSummary(vmManager).ID {
			writeJSONError(w, http.StatusNotFound, "cluster not found")
			return
		}
		user, sess, ok := loadAuthenticatedSession(w, r, authManager, db)
		if !ok {
			return
		}
		touchSessionLastAccessed(r.Context(), db, sess.ID)
		writeJSON(w, http.StatusOK, buildMeResponsePayload(user, sess, vmManager, twoFactorService))
	})
}
