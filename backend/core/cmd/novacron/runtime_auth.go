package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	coreauth "github.com/khryptorgraphics/novacron/backend/core/auth"
)

const (
	defaultRuntimeFrontendURL        = "http://localhost:3000"
	defaultRuntimeTenantID           = "default"
	defaultRuntimeClusterID          = "cluster-local"
	defaultRuntimeGitHubRedirectURL  = "http://localhost:8090/api/auth/oauth/github/callback"
	defaultRuntimeOAuthRedirectRoute = "/dashboard"
)

type runtimeAuthConfig struct {
	Enabled          bool               `yaml:"enabled"`
	FrontendURL      string             `yaml:"frontend_url"`
	DefaultTenantID  string             `yaml:"default_tenant_id"`
	DefaultClusterID string             `yaml:"default_cluster_id"`
	OAuth            runtimeOAuthConfig `yaml:"oauth"`
}

type runtimeOAuthConfig struct {
	GitHub runtimeGitHubOAuthConfig `yaml:"github"`
}

type runtimeGitHubOAuthConfig struct {
	ClientID     string `yaml:"client_id"`
	ClientSecret string `yaml:"client_secret"`
	RedirectURL  string `yaml:"redirect_url"`
}

type authConfigFile struct {
	Enabled          *bool                   `yaml:"enabled"`
	FrontendURL      string                  `yaml:"frontend_url"`
	DefaultTenantID  string                  `yaml:"default_tenant_id"`
	DefaultClusterID string                  `yaml:"default_cluster_id"`
	OAuth            *runtimeOAuthConfigFile `yaml:"oauth"`
}

type runtimeOAuthConfigFile struct {
	GitHub *runtimeGitHubOAuthConfig `yaml:"github"`
}

type runtimeAuthRuntime struct {
	config          runtimeAuthConfig
	authService     *coreauth.AuthServiceImpl
	securityManager *coreauth.SecurityManager
	userStore       *coreauth.UserMemoryStore
	tenantStore     *coreauth.TenantMemoryStore
	roleStore       *coreauth.RoleMemoryStore
	admissions      *runtimeAdmissionRegistry
}

type runtimeAdmissionRegistry struct {
	mu               sync.RWMutex
	defaultClusterID string
	byUserID         map[string]runtimeClusterAdmission
}

type runtimeClusterAdmission struct {
	UserID     string    `json:"user_id"`
	TenantID   string    `json:"tenant_id"`
	ClusterID  string    `json:"cluster_id"`
	Role       string    `json:"role"`
	Source     string    `json:"source"`
	AdmittedAt time.Time `json:"admitted_at"`
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

type runtimeAdmissionResponse struct {
	Admitted   bool      `json:"admitted"`
	ClusterID  string    `json:"clusterId,omitempty"`
	Role       string    `json:"role,omitempty"`
	Source     string    `json:"source,omitempty"`
	AdmittedAt time.Time `json:"admittedAt,omitempty"`
	TenantID   string    `json:"tenantId,omitempty"`
}

type runtimeAuthResponse struct {
	Token        string                   `json:"token"`
	RefreshToken string                   `json:"refreshToken,omitempty"`
	ExpiresAt    time.Time                `json:"expiresAt"`
	User         runtimeAuthUserResponse  `json:"user"`
	Admission    runtimeAdmissionResponse `json:"admission"`
}

type runtimeAuthProviderURLResponse struct {
	Provider         string `json:"provider"`
	AuthorizationURL string `json:"authorizationUrl"`
}

type runtimeAuthCurrentUserResponse struct {
	User      runtimeAuthUserResponse  `json:"user"`
	Admission runtimeAdmissionResponse `json:"admission"`
}

type runtimePrincipal struct {
	Claims    *coreauth.JWTClaims
	User      *coreauth.User
	Admission *runtimeClusterAdmission
}

type runtimePrincipalContextKey string

const runtimePrincipalKey runtimePrincipalContextKey = "runtime_auth_principal"

func defaultRuntimeAuthConfig() runtimeAuthConfig {
	config := runtimeAuthConfig{
		Enabled:          true,
		FrontendURL:      getenvFirst("NOVACRON_AUTH_FRONTEND_URL", "NOVACRON_FRONTEND_URL"),
		DefaultTenantID:  getenvFirst("NOVACRON_AUTH_DEFAULT_TENANT_ID"),
		DefaultClusterID: getenvFirst("NOVACRON_CLUSTER_ID"),
		OAuth: runtimeOAuthConfig{
			GitHub: runtimeGitHubOAuthConfig{
				ClientID:     getenvFirst("NOVACRON_GITHUB_CLIENT_ID"),
				ClientSecret: getenvFirst("NOVACRON_GITHUB_CLIENT_SECRET"),
				RedirectURL:  getenvFirst("NOVACRON_GITHUB_REDIRECT_URL"),
			},
		},
	}

	applyRuntimeAuthDefaults(&config)
	return config
}

func mergeRuntimeAuthConfig(config *runtimeAuthConfig, fileConfig *authConfigFile) {
	if config == nil || fileConfig == nil {
		return
	}

	if fileConfig.Enabled != nil {
		config.Enabled = *fileConfig.Enabled
	}
	if fileConfig.FrontendURL != "" {
		config.FrontendURL = fileConfig.FrontendURL
	}
	if fileConfig.DefaultTenantID != "" {
		config.DefaultTenantID = fileConfig.DefaultTenantID
	}
	if fileConfig.DefaultClusterID != "" {
		config.DefaultClusterID = fileConfig.DefaultClusterID
	}
	if fileConfig.OAuth != nil && fileConfig.OAuth.GitHub != nil {
		if fileConfig.OAuth.GitHub.ClientID != "" {
			config.OAuth.GitHub.ClientID = fileConfig.OAuth.GitHub.ClientID
		}
		if fileConfig.OAuth.GitHub.ClientSecret != "" {
			config.OAuth.GitHub.ClientSecret = fileConfig.OAuth.GitHub.ClientSecret
		}
		if fileConfig.OAuth.GitHub.RedirectURL != "" {
			config.OAuth.GitHub.RedirectURL = fileConfig.OAuth.GitHub.RedirectURL
		}
	}
}

func applyRuntimeAuthDefaults(config *runtimeAuthConfig) {
	if config == nil {
		return
	}
	if config.FrontendURL == "" {
		config.FrontendURL = defaultRuntimeFrontendURL
	}
	if config.DefaultTenantID == "" {
		config.DefaultTenantID = defaultRuntimeTenantID
	}
	if config.DefaultClusterID == "" {
		config.DefaultClusterID = defaultRuntimeClusterID
	}
	if config.OAuth.GitHub.ClientID == "" {
		config.OAuth.GitHub.ClientID = getenvFirst("NOVACRON_GITHUB_CLIENT_ID")
	}
	if config.OAuth.GitHub.ClientSecret == "" {
		config.OAuth.GitHub.ClientSecret = getenvFirst("NOVACRON_GITHUB_CLIENT_SECRET")
	}
	if config.OAuth.GitHub.RedirectURL == "" {
		config.OAuth.GitHub.RedirectURL = getenvFirst("NOVACRON_GITHUB_REDIRECT_URL")
	}
	if config.OAuth.GitHub.RedirectURL == "" {
		config.OAuth.GitHub.RedirectURL = defaultRuntimeGitHubRedirectURL
	}
}

func initializeRuntimeAuth(config runtimeAuthConfig) (*runtimeAuthRuntime, error) {
	runtime := &runtimeAuthRuntime{
		config:     config,
		admissions: newRuntimeAdmissionRegistry(config.DefaultClusterID),
	}
	if !config.Enabled {
		return runtime, nil
	}

	securityConfig, err := coreauth.DefaultSecurityConfiguration()
	if err != nil {
		return nil, fmt.Errorf("build security configuration: %w", err)
	}

	// The runtime bootstrap only needs auth, JWTs, and OAuth2. Leave the broader
	// compliance and zero-trust subsystems out of this startup path.
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

	runtime.userStore = coreauth.NewUserMemoryStore()
	runtime.roleStore = coreauth.NewRoleMemoryStore()
	runtime.tenantStore = coreauth.NewTenantMemoryStore()
	auditService := coreauth.NewInMemoryAuditService()
	runtime.authService = coreauth.NewAuthService(
		coreauth.DefaultAuthConfiguration(),
		runtime.userStore,
		runtime.roleStore,
		runtime.tenantStore,
		auditService,
	)

	securityManager, err := coreauth.NewSecurityManager(securityConfig, runtime.authService)
	if err != nil {
		return nil, fmt.Errorf("create security manager: %w", err)
	}
	runtime.securityManager = securityManager

	return runtime, nil
}

func registerRuntimeAuthRoutes(router *mux.Router, runtimeAuth *runtimeAuthRuntime) {
	if router == nil || runtimeAuth == nil {
		return
	}

	authRouter := router.PathPrefix("/api/auth").Subrouter()
	authRouter.HandleFunc("/register", runtimeAuth.handleRegister).Methods(http.MethodPost, http.MethodOptions)
	authRouter.HandleFunc("/login", runtimeAuth.handleLogin).Methods(http.MethodPost, http.MethodOptions)
	authRouter.HandleFunc("/check-email", runtimeAuth.handleCheckEmail).Methods(http.MethodGet, http.MethodOptions)
	authRouter.HandleFunc("/oauth/github/url", runtimeAuth.handleGitHubAuthorizationURL).Methods(http.MethodGet, http.MethodOptions)
	authRouter.HandleFunc("/oauth/github/login", runtimeAuth.handleGitHubLoginRedirect).Methods(http.MethodGet, http.MethodOptions)
	authRouter.HandleFunc("/oauth/github/callback", runtimeAuth.handleGitHubCallback).Methods(http.MethodGet, http.MethodOptions)
	authRouter.Handle("/me", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleCurrentUser))).Methods(http.MethodGet, http.MethodOptions)

	router.Handle("/api/cluster/admission", runtimeAuth.requireAuthenticated(http.HandlerFunc(runtimeAuth.handleClusterAdmission))).Methods(http.MethodGet, http.MethodOptions)
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
		ID:        fmt.Sprintf("user-%d", time.Now().UnixNano()),
		Username:  email,
		Email:     email,
		FirstName: strings.TrimSpace(request.FirstName),
		LastName:  strings.TrimSpace(request.LastName),
		Status:    coreauth.UserStatusActive,
		TenantID:  tenantID,
		RoleIDs:   []string{"user"},
		Metadata:  map[string]interface{}{"auth_source": "password"},
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}

	if err := r.authService.CreateUser(user, request.Password); err != nil {
		status := http.StatusBadRequest
		if strings.Contains(err.Error(), "already") {
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
	if strings.TrimSpace(request.Email) == "" || strings.TrimSpace(request.Password) == "" {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "email and password are required"})
		return
	}

	tokenPair, _, err := r.securityManager.AuthenticateWithJWT(strings.ToLower(strings.TrimSpace(request.Email)), request.Password)
	if err != nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid email or password"})
		return
	}

	user, err := r.userStore.GetByEmail(strings.ToLower(strings.TrimSpace(request.Email)))
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "authenticated user lookup failed"})
		return
	}

	admission := r.admitUser(user, "password_login")
	respondRuntimeJSON(w, http.StatusOK, r.authResponse(tokenPair, user, admission))
}

func (r *runtimeAuthRuntime) handleCheckEmail(w http.ResponseWriter, req *http.Request) {
	if !r.enabled() {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime auth is disabled"})
		return
	}

	email := strings.ToLower(strings.TrimSpace(req.URL.Query().Get("email")))
	if email == "" {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "email query parameter is required"})
		return
	}

	_, err := r.userStore.GetByEmail(email)
	respondRuntimeJSON(w, http.StatusOK, map[string]bool{"available": err != nil})
}

func (r *runtimeAuthRuntime) handleGitHubAuthorizationURL(w http.ResponseWriter, req *http.Request) {
	if !r.enabled() {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime auth is disabled"})
		return
	}
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

	respondRuntimeJSON(w, http.StatusOK, runtimeAuthProviderURLResponse{
		Provider:         "github",
		AuthorizationURL: authorizationURL,
	})
}

func (r *runtimeAuthRuntime) handleGitHubLoginRedirect(w http.ResponseWriter, req *http.Request) {
	if !r.enabled() {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime auth is disabled"})
		return
	}
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
	if !r.enabled() {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime auth is disabled"})
		return
	}
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

	tokenPair, user, err := r.securityManager.OAuth2Callback(req.Context(), "github", code, state)
	if err != nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": err.Error()})
		return
	}

	admission := r.admitUser(user, "github_oauth")
	redirectTo := defaultRuntimeOAuthRedirectRoute
	if user.Metadata != nil {
		if requestedRedirect, ok := user.Metadata["oauth_redirect_to"].(string); ok && requestedRedirect != "" {
			redirectTo = sanitizeRuntimeRedirectPath(requestedRedirect)
		}
	}
	callbackURL, err := r.frontendOAuthCallbackURL(tokenPair, user, admission, redirectTo)
	if err != nil {
		respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	http.Redirect(w, req, callbackURL, http.StatusFound)
}

func (r *runtimeAuthRuntime) handleCurrentUser(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.User == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}

	response := runtimeAuthCurrentUserResponse{
		User:      r.userResponse(principal.User),
		Admission: runtimeAdmissionToResponse(principal.Admission),
	}
	respondRuntimeJSON(w, http.StatusOK, response)
}

func (r *runtimeAuthRuntime) handleClusterAdmission(w http.ResponseWriter, req *http.Request) {
	principal, ok := runtimePrincipalFromContext(req.Context())
	if !ok || principal.User == nil {
		respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "authentication required"})
		return
	}

	respondRuntimeJSON(w, http.StatusOK, runtimeAdmissionToResponse(principal.Admission))
}

func (r *runtimeAuthRuntime) requireAuthenticated(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if !r.enabled() {
			respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "runtime auth is disabled"})
			return
		}

		claims, err := r.validateBearerToken(req)
		if err != nil {
			respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": err.Error()})
			return
		}

		user, err := r.userStore.Get(claims.UserID)
		if err != nil {
			respondRuntimeJSON(w, http.StatusUnauthorized, map[string]string{"error": "user not found"})
			return
		}

		admission, _ := r.admissions.Get(user.ID)
		ctx := context.WithValue(req.Context(), runtimePrincipalKey, &runtimePrincipal{
			Claims:    claims,
			User:      user,
			Admission: admission,
		})

		next.ServeHTTP(w, req.WithContext(ctx))
	})
}

func (r *runtimeAuthRuntime) validateBearerToken(req *http.Request) (*coreauth.JWTClaims, error) {
	authHeader := strings.TrimSpace(req.Header.Get("Authorization"))
	if authHeader == "" {
		return nil, fmt.Errorf("authorization header required")
	}
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return nil, fmt.Errorf("authorization header must start with Bearer")
	}

	return r.securityManager.ValidateJWT(strings.TrimSpace(strings.TrimPrefix(authHeader, "Bearer ")))
}

func (r *runtimeAuthRuntime) admitUser(user *coreauth.User, source string) runtimeClusterAdmission {
	admission := runtimeClusterAdmission{
		UserID:     user.ID,
		TenantID:   user.TenantID,
		ClusterID:  r.config.DefaultClusterID,
		Role:       "member",
		Source:     source,
		AdmittedAt: time.Now().UTC(),
	}
	r.admissions.Upsert(admission)
	return admission
}

func (r *runtimeAuthRuntime) authResponse(tokenPair *coreauth.TokenPair, user *coreauth.User, admission runtimeClusterAdmission) runtimeAuthResponse {
	expiresAt := time.Now().UTC().Add(time.Duration(tokenPair.ExpiresIn) * time.Second)
	return runtimeAuthResponse{
		Token:        tokenPair.AccessToken,
		RefreshToken: tokenPair.RefreshToken,
		ExpiresAt:    expiresAt,
		User:         r.userResponse(user),
		Admission:    runtimeAdmissionToResponse(&admission),
	}
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

func (r *runtimeAuthRuntime) frontendOAuthCallbackURL(tokenPair *coreauth.TokenPair, user *coreauth.User, admission runtimeClusterAdmission, redirectTo string) (string, error) {
	userPayload, err := encodeRuntimeFragmentPayload(r.userResponse(user))
	if err != nil {
		return "", fmt.Errorf("encode frontend user payload: %w", err)
	}
	admissionPayload, err := encodeRuntimeFragmentPayload(runtimeAdmissionToResponse(&admission))
	if err != nil {
		return "", fmt.Errorf("encode frontend admission payload: %w", err)
	}

	callbackBase := strings.TrimRight(r.config.FrontendURL, "/") + "/auth/github/callback"
	fragment := url.Values{}
	fragment.Set("token", tokenPair.AccessToken)
	fragment.Set("refresh_token", tokenPair.RefreshToken)
	fragment.Set("expires_at", time.Now().UTC().Add(time.Duration(tokenPair.ExpiresIn)*time.Second).Format(time.RFC3339))
	fragment.Set("user", userPayload)
	fragment.Set("admission", admissionPayload)
	fragment.Set("redirect_to", sanitizeRuntimeRedirectPath(redirectTo))

	return callbackBase + "#" + fragment.Encode(), nil
}

func (r *runtimeAuthRuntime) enabled() bool {
	return r != nil && r.config.Enabled && r.securityManager != nil
}

func runtimePrincipalFromContext(ctx context.Context) (*runtimePrincipal, bool) {
	principal, ok := ctx.Value(runtimePrincipalKey).(*runtimePrincipal)
	return principal, ok
}

func newRuntimeAdmissionRegistry(defaultClusterID string) *runtimeAdmissionRegistry {
	return &runtimeAdmissionRegistry{
		defaultClusterID: defaultClusterID,
		byUserID:         make(map[string]runtimeClusterAdmission),
	}
}

func (r *runtimeAdmissionRegistry) Upsert(admission runtimeClusterAdmission) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.byUserID[admission.UserID] = admission
}

func (r *runtimeAdmissionRegistry) Get(userID string) (*runtimeClusterAdmission, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	admission, ok := r.byUserID[userID]
	if !ok {
		return nil, false
	}
	copy := admission
	return &copy, true
}

func runtimeAdmissionToResponse(admission *runtimeClusterAdmission) runtimeAdmissionResponse {
	if admission == nil {
		return runtimeAdmissionResponse{Admitted: false}
	}
	return runtimeAdmissionResponse{
		Admitted:   true,
		ClusterID:  admission.ClusterID,
		Role:       admission.Role,
		Source:     admission.Source,
		AdmittedAt: admission.AdmittedAt,
		TenantID:   admission.TenantID,
	}
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

func getenvFirst(keys ...string) string {
	for _, key := range keys {
		if value := strings.TrimSpace(os.Getenv(key)); value != "" {
			return value
		}
	}
	return ""
}
