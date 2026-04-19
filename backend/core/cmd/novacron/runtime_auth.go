package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/gorilla/mux"
	coreauth "github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/google/uuid"
)

type runtimeAuthRuntime struct {
	config          runtimeAuthConfig
	authService     *coreauth.AuthServiceImpl
	securityManager *coreauth.SecurityManager
	userStore       coreauth.UserService
	tenantStore     *runtimeTenantStore
	roleStore       *coreauth.RoleMemoryStore
	persistence     *runtimeAuthPersistence
}

type runtimeAuthRegisterRequest struct {
	Email     string `json:"email"`
	Password  string `json:"password"`
	FirstName string `json:"firstName"`
	LastName  string `json:"lastName"`
	TenantID  string `json:"tenantId,omitempty"`
}

type runtimeAuthLoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type runtimeRefreshRequest struct {
	RefreshToken string `json:"refreshToken"`
}

type runtimeSelectClusterRequest struct {
	ClusterID string `json:"clusterId"`
}

type runtimeEdgeMetricsRequest struct {
	ClusterID     string  `json:"clusterId"`
	LatencyMS     float64 `json:"latencyMs"`
	BandwidthMBPS float64 `json:"bandwidthMbps"`
}

type runtimeAuthUserResponse struct {
	ID        string   `json:"id"`
	Email     string   `json:"email"`
	FirstName string   `json:"firstName,omitempty"`
	LastName  string   `json:"lastName,omitempty"`
	TenantID  string   `json:"tenantId"`
	Status    string   `json:"status"`
	Roles     []string `json:"roles,omitempty"`
	Role      string   `json:"role,omitempty"`
}

type runtimeClusterSummaryResponse struct {
	ID                         string    `json:"id"`
	Name                       string    `json:"name"`
	Tier                       string    `json:"tier"`
	PerformanceScore           float64   `json:"performanceScore"`
	InterconnectLatencyMS      float64   `json:"interconnectLatencyMs"`
	InterconnectBandwidthMBPS  float64   `json:"interconnectBandwidthMbps"`
	CurrentNodeCount           int       `json:"currentNodeCount"`
	MaxSupportedNodeCount      int       `json:"maxSupportedNodeCount"`
	GrowthState                string    `json:"growthState"`
	FederationState            string    `json:"federationState"`
	Degraded                   bool      `json:"degraded"`
	LastEvaluatedAt            time.Time `json:"lastEvaluatedAt"`
	EdgeLatencyMS              float64   `json:"edgeLatencyMs,omitempty"`
	EdgeBandwidthMBPS          float64   `json:"edgeBandwidthMbps,omitempty"`
}

type runtimeAdmissionResponse struct {
	Admitted   bool                        `json:"admitted"`
	State      string                      `json:"state,omitempty"`
	ClusterID  string                      `json:"clusterId,omitempty"`
	Role       string                      `json:"role,omitempty"`
	Source     string                      `json:"source,omitempty"`
	AdmittedAt time.Time                   `json:"admittedAt,omitempty"`
	TenantID   string                      `json:"tenantId,omitempty"`
	Selected   bool                        `json:"selected,omitempty"`
	Cluster    *runtimeClusterSummaryResponse `json:"cluster,omitempty"`
}

type runtimeSessionResponse struct {
	ID                string    `json:"id"`
	ExpiresAt         time.Time `json:"expiresAt"`
	CreatedAt         time.Time `json:"createdAt"`
	LastAccessedAt    time.Time `json:"lastAccessedAt"`
	SelectedClusterID string    `json:"selectedClusterId,omitempty"`
}

type runtimeAuthResponse struct {
	Token           string                    `json:"token"`
	RefreshToken    string                    `json:"refreshToken,omitempty"`
	ExpiresAt       time.Time                 `json:"expiresAt"`
	User            runtimeAuthUserResponse   `json:"user"`
	Admission       runtimeAdmissionResponse  `json:"admission"`
	Memberships     []runtimeAdmissionResponse `json:"memberships"`
	SelectedCluster *runtimeClusterSummaryResponse `json:"selectedCluster,omitempty"`
	Session         runtimeSessionResponse    `json:"session"`
}

type runtimeAuthProviderURLResponse struct {
	Provider         string `json:"provider"`
	AuthorizationURL string `json:"authorizationUrl"`
}

type runtimeAuthCurrentUserResponse struct {
	User            runtimeAuthUserResponse    `json:"user"`
	Admission       runtimeAdmissionResponse   `json:"admission"`
	Memberships     []runtimeAdmissionResponse `json:"memberships"`
	SelectedCluster *runtimeClusterSummaryResponse `json:"selectedCluster,omitempty"`
	Session         runtimeSessionResponse     `json:"session"`
}

type runtimePrincipal struct {
	Token           string
	Claims          *coreauth.JWTClaims
	User            *coreauth.User
	Session         *runtimePersistedSession
	Memberships     []runtimeClusterMembership
	SelectedCluster *runtimeClusterMembership
}

type runtimePrincipalContextKey string

const runtimePrincipalKey runtimePrincipalContextKey = "runtime_auth_principal"

func initializeRuntimeAuth(config runtimeAuthConfig) (*runtimeAuthRuntime, error) {
	runtime := &runtimeAuthRuntime{config: config}
	if !config.Enabled {
		return runtime, nil
	}

	applyRuntimeAuthDefaults(&config)
	if err := validateRuntimeAuthConfig(config); err != nil {
		return nil, err
	}
	runtime.config = config

	persistence, err := newRuntimeAuthPersistence(config)
	if err != nil {
		return nil, err
	}

	roleStore := newRuntimeRoleStore()

	securityConfig, err := coreauth.DefaultSecurityConfiguration()
	if err != nil {
		return nil, fmt.Errorf("build security configuration: %w", err)
	}
	securityConfig.Compliance = false
	securityConfig.ZeroTrust = false
	securityConfig.OAuth2 = make(map[string]coreauth.OAuth2Config)
	if githubConfigured(config) {
		githubProvider := coreauth.GetProviderConfigs()["github"]
		githubProvider.ClientID = config.OAuth.GitHub.ClientID
		githubProvider.ClientSecret = config.OAuth.GitHub.ClientSecret
		githubProvider.RedirectURL = config.OAuth.GitHub.RedirectURL
		securityConfig.OAuth2["github"] = githubProvider
	}

	authService := coreauth.NewAuthService(
		coreauth.DefaultAuthConfiguration(),
		persistence.users,
		roleStore,
		persistence.tenants,
		coreauth.NewInMemoryAuditService(),
	)
	securityManager, err := coreauth.NewSecurityManager(securityConfig, authService)
	if err != nil {
		return nil, fmt.Errorf("create security manager: %w", err)
	}

	runtime.authService = authService
	runtime.securityManager = securityManager
	runtime.userStore = persistence.users
	runtime.tenantStore = persistence.tenants
	runtime.roleStore = roleStore
	runtime.persistence = persistence

	return runtime, nil
}

func newRuntimeRoleStore() *coreauth.RoleMemoryStore {
	store := coreauth.NewRoleMemoryStore()
	_ = store.Create(&coreauth.Role{
		ID:          "viewer",
		Name:        "Viewer",
		Description: "Read-only cluster access",
		IsSystem:    true,
		Permissions: []coreauth.Permission{
			{Resource: "cluster", Action: "read", Effect: "allow"},
			{Resource: "cluster", Action: "select", Effect: "allow"},
		},
	})
	_ = store.Create(&coreauth.Role{
		ID:          "operator",
		Name:        "Operator",
		Description: "Cluster operator access",
		IsSystem:    true,
		Permissions: []coreauth.Permission{
			{Resource: "cluster", Action: "read", Effect: "allow"},
			{Resource: "cluster", Action: "select", Effect: "allow"},
			{Resource: "federation", Action: "read", Effect: "allow"},
		},
	})
	return store
}

func validateRuntimeAuthConfig(config runtimeAuthConfig) error {
	if strings.TrimSpace(config.OAuth.GitHub.ClientID) == "" && strings.TrimSpace(config.OAuth.GitHub.ClientSecret) == "" {
		return nil
	}
	if !githubConfigured(config) {
		return fmt.Errorf("github oauth requires both client_id and client_secret")
	}
	return nil
}

func registerRuntimeAuthRoutes(router *mux.Router, runtimeAuth *runtimeAuthRuntime) {
	if router == nil || runtimeAuth == nil {
		return
	}

	authRouter := router.PathPrefix("/api/auth").Subrouter()
	authRouter.HandleFunc("/register", runtimeAuth.handleRegister).Methods(http.MethodPost, http.MethodOptions)
	authRouter.HandleFunc("/login", runtimeAuth.handleLogin).Methods(http.MethodPost, http.MethodOptions)
	authRouter.HandleFunc("/refresh", runtimeAuth.handleRefresh).Methods(http.MethodPost, http.MethodOptions)
	authRouter.Handle("/logout", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleLogout))).Methods(http.MethodPost, http.MethodOptions)
	authRouter.HandleFunc("/check-email", runtimeAuth.handleCheckEmail).Methods(http.MethodGet, http.MethodOptions)
	authRouter.HandleFunc("/oauth/github/url", runtimeAuth.handleGitHubAuthorizationURL).Methods(http.MethodGet, http.MethodOptions)
	authRouter.HandleFunc("/oauth/github/login", runtimeAuth.handleGitHubLoginRedirect).Methods(http.MethodGet, http.MethodOptions)
	authRouter.HandleFunc("/oauth/github/callback", runtimeAuth.handleGitHubCallback).Methods(http.MethodGet, http.MethodOptions)
	authRouter.Handle("/me", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleCurrentUser))).Methods(http.MethodGet, http.MethodOptions)
	authRouter.Handle("/sessions", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleSessions))).Methods(http.MethodGet, http.MethodOptions)

	router.Handle("/api/cluster/admission", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleClusterAdmission))).Methods(http.MethodGet, http.MethodOptions)
	router.Handle("/api/cluster/admissions", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleClusterAdmissions))).Methods(http.MethodGet, http.MethodOptions)
	router.Handle("/api/cluster/admissions/select", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleSelectCluster))).Methods(http.MethodPost, http.MethodOptions)
	router.Handle("/api/cluster/edge-metrics", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleEdgeMetrics))).Methods(http.MethodPost, http.MethodOptions)
}

func (r *runtimeAuthRuntime) handleRegister(w http.ResponseWriter, req *http.Request) {
	if !r.enabled() {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime auth is disabled"})
		return
	}

	var request runtimeAuthRegisterRequest
	if err := json.NewDecoder(req.Body).Decode(&request); err != nil {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}
	if strings.TrimSpace(request.Email) == "" || strings.TrimSpace(request.Password) == "" || strings.TrimSpace(request.FirstName) == "" || strings.TrimSpace(request.LastName) == "" {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "email, password, firstName, and lastName are required"})
		return
	}

	email := strings.ToLower(strings.TrimSpace(request.Email))
	if _, err := r.userStore.GetByEmail(email); err == nil {
		respondRuntimeJSON(w, http.StatusConflict, map[string]string{"error": "email already in use"})
		return
	}

	tenantID := strings.TrimSpace(request.TenantID)
	if tenantID == "" {
		tenantID = r.config.DefaultTenantID
	}

	user := &coreauth.User{
		ID:        uuid.NewString(),
		Username:  email,
		Email:     email,
		FirstName: strings.TrimSpace(request.FirstName),
		LastName:  strings.TrimSpace(request.LastName),
		Status:    coreauth.UserStatusActive,
		TenantID:  tenantID,
		RoleIDs:   []string{"viewer"},
		Metadata:  map[string]interface{}{"auth_source": "password"},
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}
	if err := r.authService.CreateUser(user, request.Password); err != nil {
		status := http.StatusBadRequest
		if strings.Contains(strings.ToLower(err.Error()), "already") {
			status = http.StatusConflict
		}
		respondRuntimeJSON(w, status, map[string]string{"error": err.Error()})
		return
	}

	respondRuntimeJSON(w, http.StatusCreated, r.userResponse(user))
}

func (r *runtimeAuthRuntime) handleLogin(w http.ResponseWriter, req *http.Request) {
	if !r.enabled() {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime auth is disabled"})
		return
	}

	var request runtimeAuthLoginRequest
	if err := json.NewDecoder(req.Body).Decode(&request); err != nil {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}
	tokenPair, session, user, err := r.authenticatePasswordLogin(strings.ToLower(strings.TrimSpace(request.Email)), request.Password, req)
	if err != nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid email or password"})
		return
	}
	response, err := r.buildAuthResponse(tokenPair, session, user, "password_login")
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	r.writeSessionCookies(w, tokenPair)
	respondRuntimeJSON(w, http.StatusOK, response)
}

func (r *runtimeAuthRuntime) authenticatePasswordLogin(email string, password string, req *http.Request) (*coreauth.TokenPair, *runtimePersistedSession, *coreauth.User, error) {
	tokenPair, session, err := r.securityManager.AuthenticateWithJWT(email, password)
	if err != nil {
		return nil, nil, nil, err
	}
	user, err := r.userStore.GetByEmail(email)
	if err != nil {
		return nil, nil, nil, err
	}
	persisted := &runtimePersistedSession{
		ID:             session.ID,
		UserID:         user.ID,
		TenantID:       user.TenantID,
		Token:          tokenPair.AccessToken,
		RefreshToken:   tokenPair.RefreshToken,
		ExpiresAt:      time.Now().UTC().Add(time.Duration(tokenPair.ExpiresIn) * time.Second),
		CreatedAt:      session.CreatedAt,
		LastAccessedAt: time.Now().UTC(),
		ClientIP:       clientIP(req),
		UserAgent:      req.UserAgent(),
		Metadata:       runtimeSessionMetadata(user, ""),
	}
	if err := r.persistence.sessions.Upsert(persisted); err != nil {
		return nil, nil, nil, err
	}
	return tokenPair, persisted, user, nil
}

func (r *runtimeAuthRuntime) handleRefresh(w http.ResponseWriter, req *http.Request) {
	if !r.enabled() {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime auth is disabled"})
		return
	}

	refreshToken, err := r.extractRefreshToken(req)
	if err != nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": err.Error()})
		return
	}
	claims, err := r.securityManager.ValidateJWT(refreshToken)
	if err != nil || claims.TokenType != "refresh" {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid refresh token"})
		return
	}
	if revoked, _ := r.persistence.revocations.IsRevoked(claims.ID); revoked {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "refresh token has been revoked"})
		return
	}

	session, err := r.persistence.sessions.GetByRefreshToken(refreshToken)
	if err != nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "session not found"})
		return
	}
	if session.RevokedAt != nil || time.Now().UTC().After(session.ExpiresAt) {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "session has expired"})
		return
	}
	user, err := r.userStore.Get(session.UserID)
	if err != nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "user not found"})
		return
	}

	oldAccessClaims, _ := r.securityManager.GetTokenClaims(session.Token)
	if oldAccessClaims != nil {
		_ = r.persistence.revocations.RevokeToken(oldAccessClaims.ID, oldAccessClaims.ExpiresAt.Time)
	}
	_ = r.persistence.revocations.RevokeToken(claims.ID, claims.ExpiresAt.Time)

	tokenPair, err := r.securityManager.IssueJWTForUser(user, session.ID, runtimeSessionMetadata(user, session.SelectedClusterID))
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to issue refreshed tokens"})
		return
	}
	session.Token = tokenPair.AccessToken
	session.RefreshToken = tokenPair.RefreshToken
	session.ExpiresAt = time.Now().UTC().Add(time.Duration(tokenPair.ExpiresIn) * time.Second)
	session.LastAccessedAt = time.Now().UTC()
	session.Metadata = runtimeSessionMetadata(user, session.SelectedClusterID)
	if err := r.persistence.sessions.Upsert(session); err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to persist refreshed session"})
		return
	}

	response, err := r.buildAuthResponse(tokenPair, session, user, "session_refresh")
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	r.writeSessionCookies(w, tokenPair)
	respondRuntimeJSON(w, http.StatusOK, response)
}

func (r *runtimeAuthRuntime) handleLogout(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.Session == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}
	if principal.Claims != nil {
		_ = r.persistence.revocations.RevokeToken(principal.Claims.ID, principal.Claims.ExpiresAt.Time)
	}
	refreshClaims, _ := r.securityManager.GetTokenClaims(principal.Session.RefreshToken)
	if refreshClaims != nil {
		_ = r.persistence.revocations.RevokeToken(refreshClaims.ID, refreshClaims.ExpiresAt.Time)
	}
	_ = r.persistence.sessions.Revoke(principal.Session.ID)
	_ = r.authService.Logout(principal.Session.ID)
	r.clearSessionCookies(w)
	respondRuntimeJSON(w, http.StatusOK, map[string]bool{"success": true})
}

func (r *runtimeAuthRuntime) handleCheckEmail(w http.ResponseWriter, req *http.Request) {
	email := strings.ToLower(strings.TrimSpace(req.URL.Query().Get("email")))
	if email == "" {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "email query parameter is required"})
		return
	}
	_, err := r.userStore.GetByEmail(email)
	respondRuntimeJSON(w, http.StatusOK, map[string]bool{"available": err != nil})
}

func (r *runtimeAuthRuntime) handleGitHubAuthorizationURL(w http.ResponseWriter, req *http.Request) {
	if !githubConfigured(r.config) {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "github oauth is not configured"})
		return
	}
	redirectTo := sanitizeRuntimeRedirectPath(req.URL.Query().Get("redirect_to"))
	authorizationURL, _, err := r.securityManager.OAuth2Login("github", r.config.DefaultTenantID, redirectTo)
	if err != nil {
		respondRuntimeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	respondRuntimeJSON(w, http.StatusOK, runtimeAuthProviderURLResponse{Provider: "github", AuthorizationURL: authorizationURL})
}

func (r *runtimeAuthRuntime) handleGitHubLoginRedirect(w http.ResponseWriter, req *http.Request) {
	if !githubConfigured(r.config) {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "github oauth is not configured"})
		return
	}
	redirectTo := sanitizeRuntimeRedirectPath(req.URL.Query().Get("redirect_to"))
	authorizationURL, _, err := r.securityManager.OAuth2Login("github", r.config.DefaultTenantID, redirectTo)
	if err != nil {
		respondRuntimeJSON(w, http.StatusBadGateway, map[string]string{"error": err.Error()})
		return
	}
	http.Redirect(w, req, authorizationURL, http.StatusFound)
}

func (r *runtimeAuthRuntime) handleGitHubCallback(w http.ResponseWriter, req *http.Request) {
	if !githubConfigured(r.config) {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "github oauth is not configured"})
		return
	}

	code := strings.TrimSpace(req.URL.Query().Get("code"))
	state := strings.TrimSpace(req.URL.Query().Get("state"))
	if code == "" || state == "" {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "code and state are required"})
		return
	}

	_, user, err := r.securityManager.OAuth2Callback(req.Context(), "github", code, state)
	if err != nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": err.Error()})
		return
	}
	user, err = r.userStore.GetByEmail(user.Email)
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "github user lookup failed"})
		return
	}

	session := &runtimePersistedSession{
		ID:             uuid.NewString(),
		UserID:         user.ID,
		TenantID:       user.TenantID,
		CreatedAt:      time.Now().UTC(),
		LastAccessedAt: time.Now().UTC(),
		ClientIP:       clientIP(req),
		UserAgent:      req.UserAgent(),
	}
	tokenPair, err := r.securityManager.IssueJWTForUser(user, session.ID, runtimeSessionMetadata(user, ""))
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "token generation failed"})
		return
	}
	session.Token = tokenPair.AccessToken
	session.RefreshToken = tokenPair.RefreshToken
	session.ExpiresAt = time.Now().UTC().Add(time.Duration(tokenPair.ExpiresIn) * time.Second)
	session.Metadata = runtimeSessionMetadata(user, "")
	if err := r.persistence.sessions.Upsert(session); err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "session persistence failed"})
		return
	}

	response, err := r.buildAuthResponse(tokenPair, session, user, "github_oauth")
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	r.writeSessionCookies(w, tokenPair)

	redirectTo := defaultRuntimeOAuthRedirectRoute
	if user.Metadata != nil {
		if requested, ok := user.Metadata["oauth_redirect_to"].(string); ok && requested != "" {
			redirectTo = sanitizeRuntimeRedirectPath(requested)
		}
	}
	callbackURL, err := r.frontendOAuthCallbackURL(response, redirectTo)
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	http.Redirect(w, req, callbackURL, http.StatusFound)
}

func (r *runtimeAuthRuntime) handleCurrentUser(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.User == nil || principal.Session == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}
	membershipResponses, selected := r.membershipResponses(principal.User.ID, principal.Session.SelectedClusterID)
	respondRuntimeJSON(w, http.StatusOK, runtimeAuthCurrentUserResponse{
		User:            r.userResponse(principal.User),
		Admission:       selectedAdmissionResponse(membershipResponses),
		Memberships:     membershipResponses,
		SelectedCluster: selected,
		Session:         runtimeSessionResponseFromPersisted(principal.Session),
	})
}

func (r *runtimeAuthRuntime) handleSessions(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.User == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}
	sessions, err := r.persistence.sessions.ListByUser(principal.User.ID)
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to list sessions"})
		return
	}
	response := make([]runtimeSessionResponse, 0, len(sessions))
	for _, session := range sessions {
		if session.RevokedAt != nil {
			continue
		}
		response = append(response, runtimeSessionResponseFromPersisted(&session))
	}
	respondRuntimeJSON(w, http.StatusOK, response)
}

func (r *runtimeAuthRuntime) handleClusterAdmission(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.User == nil || principal.Session == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}
	membershipResponses, _ := r.membershipResponses(principal.User.ID, principal.Session.SelectedClusterID)
	respondRuntimeJSON(w, http.StatusOK, selectedAdmissionResponse(membershipResponses))
}

func (r *runtimeAuthRuntime) handleClusterAdmissions(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.User == nil || principal.Session == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}
	membershipResponses, _ := r.membershipResponses(principal.User.ID, principal.Session.SelectedClusterID)
	respondRuntimeJSON(w, http.StatusOK, membershipResponses)
}

func (r *runtimeAuthRuntime) handleSelectCluster(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.User == nil || principal.Session == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}
	var request runtimeSelectClusterRequest
	if err := json.NewDecoder(req.Body).Decode(&request); err != nil {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}
	request.ClusterID = strings.TrimSpace(request.ClusterID)
	if request.ClusterID == "" {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "clusterId is required"})
		return
	}
	memberships, err := r.persistence.memberships.ListByUser(principal.User.ID)
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to load memberships"})
		return
	}
	allowed := false
	for _, membership := range memberships {
		if membership.ClusterID == request.ClusterID && membership.State == "active" {
			allowed = true
			break
		}
	}
	if !allowed {
		respondRuntimeJSON(w, http.StatusForbidden, map[string]string{"error": "active membership required for target cluster"})
		return
	}
	if err := r.persistence.sessions.SetSelectedCluster(principal.Session.ID, request.ClusterID); err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to persist selected cluster"})
		return
	}
	session, err := r.persistence.sessions.Get(principal.Session.ID)
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to reload session"})
		return
	}
	membershipResponses, selected := r.membershipResponses(principal.User.ID, request.ClusterID)
	respondRuntimeJSON(w, http.StatusOK, runtimeAuthCurrentUserResponse{
		User:            r.userResponse(principal.User),
		Admission:       selectedAdmissionResponse(membershipResponses),
		Memberships:     membershipResponses,
		SelectedCluster: selected,
		Session:         runtimeSessionResponseFromPersisted(session),
	})
}

func (r *runtimeAuthRuntime) handleEdgeMetrics(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.User == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}
	var request runtimeEdgeMetricsRequest
	if err := json.NewDecoder(req.Body).Decode(&request); err != nil {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}
	if strings.TrimSpace(request.ClusterID) == "" {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "clusterId is required"})
		return
	}
	if err := r.persistence.edgeMetrics.Upsert(runtimeEdgeMetric{
		UserID:        principal.User.ID,
		ClusterID:     request.ClusterID,
		LatencyMS:     request.LatencyMS,
		BandwidthMBPS: request.BandwidthMBPS,
	}); err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to persist cluster edge metrics"})
		return
	}
	membershipResponses, _ := r.membershipResponses(principal.User.ID, principal.Session.SelectedClusterID)
	respondRuntimeJSON(w, http.StatusOK, membershipResponses)
}

func (r *runtimeAuthRuntime) requireAuthenticated(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		token, claims, err := r.validateAccessToken(req)
		if err != nil {
			respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": err.Error()})
			return
		}
		session, err := r.persistence.sessions.Get(claims.SessionID)
		if err != nil || session.RevokedAt != nil || time.Now().UTC().After(session.ExpiresAt) {
			respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "session is no longer active"})
			return
		}
		if session.Token != token {
			respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "access token is no longer current"})
			return
		}
		user, err := r.userStore.Get(claims.UserID)
		if err != nil {
			respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "user not found"})
			return
		}
		memberships, _ := r.persistence.memberships.ListByUser(user.ID)
		selected := selectMembershipByClusterID(memberships, session.SelectedClusterID)
		ctx := context.WithValue(req.Context(), runtimePrincipalKey, &runtimePrincipal{
			Token:           token,
			Claims:          claims,
			User:            user,
			Session:         session,
			Memberships:     memberships,
			SelectedCluster: selected,
		})
		next.ServeHTTP(w, req.WithContext(ctx))
	})
}

func (r *runtimeAuthRuntime) validateAccessToken(req *http.Request) (string, *coreauth.JWTClaims, error) {
	token := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(req.Header.Get("Authorization")), "Bearer "))
	if token == "" {
		if cookie, err := req.Cookie(r.config.Session.Cookie.AccessTokenName); err == nil {
			token = strings.TrimSpace(cookie.Value)
		}
	}
	if token == "" {
		return "", nil, fmt.Errorf("authorization required")
	}
	claims, err := r.securityManager.ValidateJWT(token)
	if err != nil || claims.TokenType != "access" {
		return "", nil, fmt.Errorf("invalid access token")
	}
	if revoked, _ := r.persistence.revocations.IsRevoked(claims.ID); revoked {
		return "", nil, fmt.Errorf("access token has been revoked")
	}
	return token, claims, nil
}

func (r *runtimeAuthRuntime) extractRefreshToken(req *http.Request) (string, error) {
	var request runtimeRefreshRequest
	if req.Body != nil {
		_ = json.NewDecoder(req.Body).Decode(&request)
	}
	token := strings.TrimSpace(request.RefreshToken)
	if token == "" {
		if cookie, err := req.Cookie(r.config.Session.Cookie.RefreshTokenName); err == nil {
			token = strings.TrimSpace(cookie.Value)
		}
	}
	if token == "" {
		return "", fmt.Errorf("refresh token is required")
	}
	return token, nil
}

func (r *runtimeAuthRuntime) buildAuthResponse(tokenPair *coreauth.TokenPair, session *runtimePersistedSession, user *coreauth.User, source string) (runtimeAuthResponse, error) {
	if source != "" {
		if _, _, err := r.ensureMemberships(user, source); err != nil {
			return runtimeAuthResponse{}, err
		}
	}
	membershipResponses, selected := r.membershipResponses(user.ID, session.SelectedClusterID)
	if session.SelectedClusterID == "" && selected != nil {
		session.SelectedClusterID = selected.ID
		if err := r.persistence.sessions.SetSelectedCluster(session.ID, selected.ID); err == nil {
			session.Metadata = runtimeSessionMetadata(user, selected.ID)
			_ = r.persistence.sessions.Upsert(session)
		}
	}
	admission := selectedAdmissionResponse(membershipResponses)
	return runtimeAuthResponse{
		Token:           tokenPair.AccessToken,
		RefreshToken:    tokenPair.RefreshToken,
		ExpiresAt:       time.Now().UTC().Add(time.Duration(tokenPair.ExpiresIn) * time.Second),
		User:            r.userResponse(user),
		Admission:       admission,
		Memberships:     membershipResponses,
		SelectedCluster: selected,
		Session:         runtimeSessionResponseFromPersisted(session),
	}, nil
}

func (r *runtimeAuthRuntime) ensureMemberships(user *coreauth.User, source string) ([]runtimeClusterMembership, *runtimeClusterMembership, error) {
	clusters, err := r.persistence.clusters.List()
	if err != nil {
		return nil, nil, err
	}
	if len(clusters) == 0 {
		if err := r.persistence.clusters.EnsureDefaultCluster(r.config.DefaultClusterID); err != nil {
			return nil, nil, err
		}
		clusters, err = r.persistence.clusters.List()
		if err != nil {
			return nil, nil, err
		}
	}

	for _, cluster := range clusters {
		state := r.config.Membership.DefaultState
		if r.config.Membership.AutoAdmit {
			state = "active"
		}
		if _, err := r.persistence.memberships.Upsert(runtimeClusterMembership{
			UserID:    user.ID,
			TenantID:  user.TenantID,
			ClusterID: cluster.ID,
			State:     state,
			Role:      "member",
			Source:    source,
		}); err != nil {
			return nil, nil, err
		}
	}

	memberships, err := r.persistence.memberships.ListByUser(user.ID)
	if err != nil {
		return nil, nil, err
	}
	selected := selectBestMembership(memberships, clusters, nil)
	return memberships, selected, nil
}

func (r *runtimeAuthRuntime) membershipResponses(userID string, selectedClusterID string) ([]runtimeAdmissionResponse, *runtimeClusterSummaryResponse) {
	memberships, err := r.persistence.memberships.ListByUser(userID)
	if err != nil {
		return nil, nil
	}
	clusters, err := r.persistence.clusters.List()
	if err != nil {
		return nil, nil
	}
	edges, _ := r.persistence.edgeMetrics.ListByUser(userID)
	clusterByID := make(map[string]runtimeClusterRecord, len(clusters))
	for _, cluster := range clusters {
		clusterByID[cluster.ID] = cluster
	}
	edgeByCluster := make(map[string]runtimeEdgeMetric, len(edges))
	for _, edge := range edges {
		edgeByCluster[edge.ClusterID] = edge
	}

	sort.SliceStable(memberships, func(i, j int) bool {
		left := clusterByID[memberships[i].ClusterID]
		right := clusterByID[memberships[j].ClusterID]
		leftScore := left.PerformanceScore
		rightScore := right.PerformanceScore
		if edge, ok := edgeByCluster[left.ID]; ok {
			leftScore += edge.BandwidthMBPS/1000 - edge.LatencyMS/10
		}
		if edge, ok := edgeByCluster[right.ID]; ok {
			rightScore += edge.BandwidthMBPS/1000 - edge.LatencyMS/10
		}
		return leftScore > rightScore
	})

	response := make([]runtimeAdmissionResponse, 0, len(memberships))
	var selected *runtimeClusterSummaryResponse
	for _, membership := range memberships {
		cluster := clusterByID[membership.ClusterID]
		summary := runtimeClusterSummaryResponseFromRecord(&cluster)
		if edge, ok := edgeByCluster[membership.ClusterID]; ok {
			summary.EdgeLatencyMS = edge.LatencyMS
			summary.EdgeBandwidthMBPS = edge.BandwidthMBPS
		}
		selectedMembership := membership.ClusterID == selectedClusterID || (selectedClusterID == "" && selected == nil && membership.State == "active")
		admission := runtimeAdmissionResponse{
			Admitted:   membership.State == "active",
			State:      membership.State,
			ClusterID:  membership.ClusterID,
			Role:       membership.Role,
			Source:     membership.Source,
			AdmittedAt: membership.CreatedAt,
			TenantID:   membership.TenantID,
			Selected:   selectedMembership,
			Cluster:    &summary,
		}
		response = append(response, admission)
		if selectedMembership {
			selected = &summary
		}
	}
	return response, selected
}

func (r *runtimeAuthRuntime) userResponse(user *coreauth.User) runtimeAuthUserResponse {
	response := runtimeAuthUserResponse{
		ID:        user.ID,
		Email:     user.Email,
		FirstName: user.FirstName,
		LastName:  user.LastName,
		TenantID:  user.TenantID,
		Status:    string(user.Status),
		Roles:     append([]string(nil), user.RoleIDs...),
	}
	if len(response.Roles) > 0 {
		response.Role = response.Roles[0]
	}
	return response
}

func (r *runtimeAuthRuntime) writeSessionCookies(w http.ResponseWriter, tokenPair *coreauth.TokenPair) {
	if r.config.Session.Transport == "bearer" {
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     r.config.Session.Cookie.AccessTokenName,
		Value:    tokenPair.AccessToken,
		Path:     r.config.Session.Cookie.Path,
		Domain:   r.config.Session.Cookie.Domain,
		HttpOnly: true,
		Secure:   r.config.Session.Cookie.Secure,
		SameSite: runtimeCookieSameSite(r.config.Session.Cookie.SameSite),
		Expires:  time.Now().UTC().Add(time.Duration(tokenPair.ExpiresIn) * time.Second),
	})
	http.SetCookie(w, &http.Cookie{
		Name:     r.config.Session.Cookie.RefreshTokenName,
		Value:    tokenPair.RefreshToken,
		Path:     r.config.Session.Cookie.Path,
		Domain:   r.config.Session.Cookie.Domain,
		HttpOnly: true,
		Secure:   r.config.Session.Cookie.Secure,
		SameSite: runtimeCookieSameSite(r.config.Session.Cookie.SameSite),
		Expires:  time.Now().UTC().Add(7 * 24 * time.Hour),
	})
}

func (r *runtimeAuthRuntime) clearSessionCookies(w http.ResponseWriter) {
	for _, name := range []string{r.config.Session.Cookie.AccessTokenName, r.config.Session.Cookie.RefreshTokenName} {
		http.SetCookie(w, &http.Cookie{
			Name:     name,
			Value:    "",
			Path:     r.config.Session.Cookie.Path,
			Domain:   r.config.Session.Cookie.Domain,
			HttpOnly: true,
			Secure:   r.config.Session.Cookie.Secure,
			SameSite: runtimeCookieSameSite(r.config.Session.Cookie.SameSite),
			MaxAge:   -1,
			Expires:  time.Unix(0, 0),
		})
	}
}

func (r *runtimeAuthRuntime) frontendOAuthCallbackURL(response runtimeAuthResponse, redirectTo string) (string, error) {
	userPayload, err := encodeRuntimeFragmentPayload(response.User)
	if err != nil {
		return "", err
	}
	admissionPayload, err := encodeRuntimeFragmentPayload(response.Admission)
	if err != nil {
		return "", err
	}
	membershipsPayload, err := encodeRuntimeFragmentPayload(response.Memberships)
	if err != nil {
		return "", err
	}
	selectedClusterPayload := ""
	if response.SelectedCluster != nil {
		selectedClusterPayload, err = encodeRuntimeFragmentPayload(response.SelectedCluster)
		if err != nil {
			return "", err
		}
	}
	sessionPayload, err := encodeRuntimeFragmentPayload(response.Session)
	if err != nil {
		return "", err
	}
	callbackBase := strings.TrimRight(r.config.FrontendURL, "/") + "/auth/github/callback"
	fragment := url.Values{}
	fragment.Set("token", response.Token)
	fragment.Set("refresh_token", response.RefreshToken)
	fragment.Set("expires_at", response.ExpiresAt.Format(time.RFC3339))
	fragment.Set("user", userPayload)
	fragment.Set("admission", admissionPayload)
	fragment.Set("memberships", membershipsPayload)
	if selectedClusterPayload != "" {
		fragment.Set("selected_cluster", selectedClusterPayload)
	}
	fragment.Set("session", sessionPayload)
	fragment.Set("redirect_to", sanitizeRuntimeRedirectPath(redirectTo))
	return callbackBase + "#" + fragment.Encode(), nil
}

func (r *runtimeAuthRuntime) enabled() bool {
	return r != nil && r.config.Enabled && r.securityManager != nil && r.persistence != nil
}

func runtimePrincipalFromContext(ctx context.Context) (*runtimePrincipal, bool) {
	principal, ok := ctx.Value(runtimePrincipalKey).(*runtimePrincipal)
	return principal, ok
}

func runtimeAdmissionResponseFromMembership(membership *runtimeClusterMembership, cluster *runtimeClusterSummaryResponse) runtimeAdmissionResponse {
	if membership == nil {
		return runtimeAdmissionResponse{Admitted: false}
	}
	return runtimeAdmissionResponse{
		Admitted:   membership.State == "active",
		State:      membership.State,
		ClusterID:  membership.ClusterID,
		Role:       membership.Role,
		Source:     membership.Source,
		AdmittedAt: membership.CreatedAt,
		TenantID:   membership.TenantID,
		Selected:   membership.Selected,
		Cluster:    cluster,
	}
}

func selectedAdmissionResponse(memberships []runtimeAdmissionResponse) runtimeAdmissionResponse {
	for _, membership := range memberships {
		if membership.Selected {
			return membership
		}
	}
	for _, membership := range memberships {
		if membership.Admitted {
			return membership
		}
	}
	return runtimeAdmissionResponse{Admitted: false}
}

func runtimeClusterSummaryResponseFromRecord(record *runtimeClusterRecord) runtimeClusterSummaryResponse {
	if record == nil {
		return runtimeClusterSummaryResponse{}
	}
	return runtimeClusterSummaryResponse{
		ID:                        record.ID,
		Name:                      record.Name,
		Tier:                      record.Tier,
		PerformanceScore:          record.PerformanceScore,
		InterconnectLatencyMS:     record.InterconnectLatencyMS,
		InterconnectBandwidthMBPS: record.InterconnectBandwidthMBPS,
		CurrentNodeCount:          record.CurrentNodeCount,
		MaxSupportedNodeCount:     record.MaxSupportedNodeCount,
		GrowthState:               record.GrowthState,
		FederationState:           record.FederationState,
		Degraded:                  record.Degraded,
		LastEvaluatedAt:           record.LastEvaluatedAt,
	}
}

func runtimeSessionResponseFromPersisted(session *runtimePersistedSession) runtimeSessionResponse {
	if session == nil {
		return runtimeSessionResponse{}
	}
	return runtimeSessionResponse{
		ID:                session.ID,
		ExpiresAt:         session.ExpiresAt,
		CreatedAt:         session.CreatedAt,
		LastAccessedAt:    session.LastAccessedAt,
		SelectedClusterID: session.SelectedClusterID,
	}
}

func runtimeSessionMetadata(user *coreauth.User, selectedClusterID string) map[string]interface{} {
	metadata := map[string]interface{}{
		"email":       user.Email,
		"first_name":  user.FirstName,
		"last_name":   user.LastName,
		"status":      user.Status,
	}
	if selectedClusterID != "" {
		metadata["selected_cluster_id"] = selectedClusterID
	}
	return metadata
}

func selectBestMembership(memberships []runtimeClusterMembership, clusters []runtimeClusterRecord, edges []runtimeEdgeMetric) *runtimeClusterMembership {
	if len(memberships) == 0 {
		return nil
	}
	clusterByID := make(map[string]runtimeClusterRecord, len(clusters))
	for _, cluster := range clusters {
		clusterByID[cluster.ID] = cluster
	}
	edgeByCluster := make(map[string]runtimeEdgeMetric, len(edges))
	for _, edge := range edges {
		edgeByCluster[edge.ClusterID] = edge
	}
	bestIdx := -1
	bestScore := -1e9
	for idx, membership := range memberships {
		if membership.State != "active" {
			continue
		}
		score := clusterByID[membership.ClusterID].PerformanceScore
		if edge, ok := edgeByCluster[membership.ClusterID]; ok {
			score += edge.BandwidthMBPS/1000 - edge.LatencyMS/10
		}
		if score > bestScore {
			bestScore = score
			bestIdx = idx
		}
	}
	if bestIdx == -1 {
		return nil
	}
	selected := memberships[bestIdx]
	return &selected
}

func selectMembershipByClusterID(memberships []runtimeClusterMembership, clusterID string) *runtimeClusterMembership {
	for _, membership := range memberships {
		if membership.ClusterID == clusterID {
			copy := membership
			return &copy
		}
	}
	return nil
}

func encodeRuntimeFragmentPayload(payload interface{}) (string, error) {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(encoded), nil
}

func githubConfigured(config runtimeAuthConfig) bool {
	return strings.TrimSpace(config.OAuth.GitHub.ClientID) != "" && strings.TrimSpace(config.OAuth.GitHub.ClientSecret) != ""
}

func sanitizeRuntimeRedirectPath(requested string) string {
	requested = strings.TrimSpace(requested)
	if requested == "" {
		return defaultRuntimeOAuthRedirectRoute
	}
	if strings.HasPrefix(requested, "/") && !strings.HasPrefix(requested, "//") {
		return requested
	}
	return defaultRuntimeOAuthRedirectRoute
}

func runtimeCORSMiddleware(config runtimeAuthConfig) mux.MiddlewareFunc {
	allowedOrigins := runtimeAllowedOrigins(config)
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			origin := strings.TrimSpace(req.Header.Get("Origin"))
			if origin != "" && allowedOrigins[origin] {
				w.Header().Set("Access-Control-Allow-Origin", origin)
				w.Header().Set("Access-Control-Allow-Credentials", "true")
				w.Header().Set("Vary", "Origin")
				w.Header().Set("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
				w.Header().Set("Access-Control-Allow-Headers", "Authorization,Content-Type")
			}
			if req.Method == http.MethodOptions {
				w.WriteHeader(http.StatusNoContent)
				return
			}
			next.ServeHTTP(w, req)
		})
	}
}

func runtimeAllowedOrigins(config runtimeAuthConfig) map[string]bool {
	origins := map[string]bool{
		defaultRuntimeFrontendURL: true,
		"http://127.0.0.1:3000":   true,
		"http://localhost:8090":   true,
	}
	if config.FrontendURL != "" {
		origins[strings.TrimRight(config.FrontendURL, "/")] = true
	}
	return origins
}

func runtimeCookieSameSite(value string) http.SameSite {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "strict":
		return http.SameSiteStrictMode
	case "none":
		return http.SameSiteNoneMode
	default:
		return http.SameSiteLaxMode
	}
}

func getenvFirst(keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return ""
}

func clientIP(req *http.Request) string {
	if req == nil {
		return ""
	}
	if forwarded := strings.TrimSpace(req.Header.Get("X-Forwarded-For")); forwarded != "" {
		return strings.TrimSpace(strings.Split(forwarded, ",")[0])
	}
	return strings.TrimSpace(req.RemoteAddr)
}
