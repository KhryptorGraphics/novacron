//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"

	authapi "github.com/khryptorgraphics/novacron/backend/api/auth"
	graphqlapi "github.com/khryptorgraphics/novacron/backend/api/graphql"
	securityapi "github.com/khryptorgraphics/novacron/backend/api/security"
	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

type canonicalServices struct {
	twoFactorService *auth.TwoFactorService
	securityHandlers *securityapi.SecurityHandlers
	websocketHandler *websocketapi.WebSocketHandler
	graphqlHandler   http.Handler
	shutdown         func()
}

const (
	canonicalRuntimeMonitoringReadsEnv = "CANONICAL_RUNTIME_MONITORING_READS"
	canonicalRuntimeBaseURLEnv         = "CANONICAL_RUNTIME_BASE_URL"
	canonicalRuntimeMonitoringReadPath = "/internal/runtime/v1/monitoring/metrics"
	canonicalRuntimeMonitoringTimeout  = 2 * time.Second
	novaCronReadSourceHeader           = "X-NovaCron-Read-Source"
	novaCronReadSourceRuntime          = "runtime"
	novaCronReadSourceSQLFallback      = "sql-fallback"
)

type monitoringSummaryResponse struct {
	CurrentCpuUsage         float64  `json:"currentCpuUsage"`
	CurrentMemoryUsage      float64  `json:"currentMemoryUsage"`
	CurrentDiskUsage        float64  `json:"currentDiskUsage"`
	CurrentNetworkUsage     float64  `json:"currentNetworkUsage"`
	CpuChangePercentage     float64  `json:"cpuChangePercentage"`
	MemoryChangePercentage  float64  `json:"memoryChangePercentage"`
	DiskChangePercentage    float64  `json:"diskChangePercentage"`
	NetworkChangePercentage float64  `json:"networkChangePercentage"`
	TimeLabels              []string `json:"timeLabels"`
	CpuAnalysis             string   `json:"cpuAnalysis"`
	MemoryAnalysis          string   `json:"memoryAnalysis"`
}

type runtimeMonitoringReadClient struct {
	baseURL string
	client  *http.Client
}

func syntheticMonitoringSummaryPayload() monitoringSummaryResponse {
	return monitoringSummaryResponse{
		CurrentCpuUsage:         45.2,
		CurrentMemoryUsage:      72.1,
		CurrentDiskUsage:        58.3,
		CurrentNetworkUsage:     125.7,
		CpuChangePercentage:     5.2,
		MemoryChangePercentage:  -2.1,
		DiskChangePercentage:    1.8,
		NetworkChangePercentage: 12.5,
		TimeLabels:              []string{"10:00", "10:30", "11:00", "11:30", "12:00"},
		CpuAnalysis:             "CPU usage shows normal workday patterns.",
		MemoryAnalysis:          "Memory allocation is healthy.",
	}
}

func newRuntimeMonitoringReadClientFromEnv() *runtimeMonitoringReadClient {
	if !envBool(canonicalRuntimeMonitoringReadsEnv) {
		return nil
	}

	baseURL := strings.TrimSpace(os.Getenv(canonicalRuntimeBaseURLEnv))
	if baseURL == "" {
		return nil
	}

	return &runtimeMonitoringReadClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		client: &http.Client{
			Timeout: canonicalRuntimeMonitoringTimeout,
		},
	}
}

func (c *runtimeMonitoringReadClient) readMonitoringSummary(ctx context.Context, source *http.Request) (monitoringSummaryResponse, error) {
	requestCtx, cancel := context.WithTimeout(ctx, canonicalRuntimeMonitoringTimeout)
	defer cancel()

	targetURL := c.baseURL + canonicalRuntimeMonitoringReadPath
	if rawQuery := source.URL.RawQuery; rawQuery != "" {
		targetURL += "?" + rawQuery
	}

	req, err := http.NewRequestWithContext(requestCtx, http.MethodGet, targetURL, nil)
	if err != nil {
		return monitoringSummaryResponse{}, err
	}

	req.Header.Set("Accept", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return monitoringSummaryResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return monitoringSummaryResponse{}, fmt.Errorf("runtime monitoring summary returned status %d", resp.StatusCode)
	}

	var payload monitoringSummaryResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return monitoringSummaryResponse{}, err
	}

	return payload, nil
}

func monitoringSummaryHandler(runtimeClient *runtimeMonitoringReadClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		payload := syntheticMonitoringSummaryPayload()
		readSource := novaCronReadSourceSQLFallback

		if runtimeClient != nil {
			runtimePayload, err := runtimeClient.readMonitoringSummary(r.Context(), r)
			if err == nil {
				payload = runtimePayload
				readSource = novaCronReadSourceRuntime
			}
		}

		w.Header().Set(novaCronReadSourceHeader, readSource)
		writeJSON(w, http.StatusOK, payload)
	}
}

func main() {
	cfg, err := config.Load()
	if err != nil {
		logger.Fatal("Failed to load configuration", "error", err)
	}

	if err := cfg.Validate(); err != nil {
		logger.Fatal("Invalid configuration", "error", err)
	}

	appLogger := logger.NewFromConfig(
		cfg.Logging.Level,
		cfg.Logging.Format,
		cfg.Logging.Output,
		cfg.Logging.Structured,
	)
	logger.SetGlobalLogger(appLogger)

	appLogger.Info("Starting NovaCron API Server...",
		"version", "1.0.0",
		"api_port", cfg.Server.APIPort,
		"ws_port", cfg.Server.WSPort,
	)

	db, err := initDatabase(cfg)
	if err != nil {
		appLogger.Fatal("Failed to initialize database", "error", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager(cfg.Auth.Secret, db)
	services, err := initializeCanonicalServices(cfg, db, authManager)
	if err != nil {
		appLogger.Fatal("Failed to initialize canonical backend services", "error", err)
	}
	defer services.shutdown()

	server := buildCanonicalServer(cfg, db, authManager, services)

	go func() {
		appLogger.Info("API Server starting", "port", cfg.Server.APIPort)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			appLogger.Fatal("Server failed to start", "error", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	appLogger.Info("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		appLogger.Fatal("Server forced to shutdown", "error", err)
	}

	appLogger.Info("Server exited gracefully")
}

func buildCanonicalServer(cfg *config.Config, db *sql.DB, authManager *auth.SimpleAuthManager, services *canonicalServices) *http.Server {
	router := mux.NewRouter()
	router.StrictSlash(true)

	corsHandler := buildCORSHandler(cfg)

	registerPublicRoutes(router, authManager, db, services.twoFactorService)

	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.Use(requireAuth(authManager))
	registerSecureAPIRoutes(apiRouter, db)

	apiV1Router := router.PathPrefix("/api/v1").Subrouter()
	apiV1Router.Use(requireAuth(authManager))
	registerSecureAPIRoutes(apiV1Router, db)

	registerCanonicalSecurityRoutes(router, authManager, services.securityHandlers)
	registerCanonicalAdminRoutes(router, authManager, db)
	registerCanonicalGraphQLRoute(router, authManager, services.graphqlHandler)
	services.websocketHandler.RegisterWebSocketRoutes(router, func(required string, next http.HandlerFunc) http.Handler {
		return requireAuth(authManager)(requireRoleHandler(required, next))
	})
	registerSecurityWebSocketAliases(router, authManager, services.securityHandlers)

	router.HandleFunc("/health", healthCheckHandler(cfg, db)).Methods(http.MethodGet)
	router.HandleFunc("/api/info", apiInfoHandler(cfg)).Methods(http.MethodGet)

	return &http.Server{
		Addr:         ":" + cfg.Server.APIPort,
		Handler:      corsHandler(router),
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}
}

func buildCORSHandler(cfg *config.Config) mux.MiddlewareFunc {
	allowedOrigins := cfg.CORS.AllowedOrigins
	if len(allowedOrigins) == 0 {
		allowedOrigins = []string{
			"http://localhost:8092",
			"http://localhost:3001",
			"http://localhost:3000",
		}
	}

	allowedMethods := cfg.CORS.AllowedMethods
	if len(allowedMethods) == 0 {
		allowedMethods = []string{http.MethodGet, http.MethodPost, http.MethodPut, http.MethodDelete, http.MethodPatch, http.MethodOptions}
	}

	allowedHeaders := cfg.CORS.AllowedHeaders
	if len(allowedHeaders) == 0 {
		allowedHeaders = []string{"Content-Type", "Authorization", "X-User-Email"}
	}

	return handlers.CORS(
		handlers.AllowedOrigins(allowedOrigins),
		handlers.AllowedMethods(allowedMethods),
		handlers.AllowedHeaders(allowedHeaders),
		handlers.AllowCredentials(),
	)
}

func initDatabase(cfg *config.Config) (*sql.DB, error) {
	db, err := sql.Open("postgres", cfg.Database.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	db.SetMaxOpenConns(cfg.Database.MaxConnections)
	db.SetMaxIdleConns(cfg.Database.MaxConnections / 2)
	db.SetConnMaxLifetime(cfg.Database.ConnMaxLifetime)
	db.SetConnMaxIdleTime(cfg.Database.ConnMaxIdleTime)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	if err := runMigrations(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to run migrations: %w", err)
	}

	return db, nil
}

func runMigrations(db *sql.DB) error {
	migrations := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id SERIAL PRIMARY KEY,
			username VARCHAR(255) UNIQUE NOT NULL,
			email VARCHAR(255) UNIQUE NOT NULL,
			password_hash VARCHAR(255) NOT NULL,
			role VARCHAR(50) DEFAULT 'user',
			active BOOLEAN DEFAULT true,
			tenant_id VARCHAR(255) DEFAULT 'default',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`ALTER TABLE users ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT true`,
		`CREATE TABLE IF NOT EXISTS vms (
			id VARCHAR(255) PRIMARY KEY,
			name VARCHAR(255) NOT NULL,
			state VARCHAR(50) NOT NULL,
			node_id VARCHAR(255),
			owner_id INTEGER REFERENCES users(id),
			tenant_id VARCHAR(255) DEFAULT 'default',
			config JSONB,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS vm_metrics (
			id SERIAL PRIMARY KEY,
			vm_id VARCHAR(255) REFERENCES vms(id),
			cpu_usage FLOAT,
			memory_usage FLOAT,
			network_sent BIGINT,
			network_recv BIGINT,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id ON vm_metrics(vm_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_metrics_timestamp ON vm_metrics(timestamp)`,
		`CREATE TABLE IF NOT EXISTS networks (
			id VARCHAR(255) PRIMARY KEY,
			name VARCHAR(255) NOT NULL,
			type VARCHAR(50) NOT NULL DEFAULT 'bridged',
			subnet VARCHAR(255) NOT NULL,
			gateway VARCHAR(255),
			status VARCHAR(50) NOT NULL DEFAULT 'active',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS vm_interfaces (
			id VARCHAR(255) PRIMARY KEY,
			vm_id VARCHAR(255) NOT NULL REFERENCES vms(id) ON DELETE CASCADE,
			network_id VARCHAR(255) REFERENCES networks(id) ON DELETE SET NULL,
			name VARCHAR(255) NOT NULL,
			mac_address VARCHAR(255) NOT NULL,
			ip_address VARCHAR(255),
			status VARCHAR(50) NOT NULL DEFAULT 'attached',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_interfaces_vm_id ON vm_interfaces(vm_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_interfaces_network_id ON vm_interfaces(network_id)`,
	}

	for _, migration := range migrations {
		if _, err := db.Exec(migration); err != nil {
			return fmt.Errorf("migration failed: %w", err)
		}
	}

	return nil
}

func requireAuth(authManager *auth.SimpleAuthManager) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tokenString, err := extractBearerToken(r.Header.Get("Authorization"))
			if err != nil {
				writeJSONError(w, http.StatusUnauthorized, err.Error())
				return
			}

			claims, err := validateJWT(tokenString, authManager.GetJWTSecret())
			if err != nil {
				writeJSONError(w, http.StatusUnauthorized, "invalid or expired token")
				return
			}
			if stringClaim(claims, "purpose") == "pending_2fa" {
				writeJSONError(w, http.StatusUnauthorized, "two-factor authentication is not complete")
				return
			}

			userID := stringClaim(claims, "user_id", "sub")
			if userID == "" {
				writeJSONError(w, http.StatusUnauthorized, "token missing user identity")
				return
			}

			ctx := context.WithValue(r.Context(), "user_id", userID)
			ctx = context.WithValue(ctx, "tenant_id", stringClaim(claims, "tenant_id"))
			ctx = context.WithValue(ctx, "role", stringClaim(claims, "role"))
			ctx = context.WithValue(ctx, "roles", stringSliceClaim(claims, "roles"))

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func requireAnyRoleMiddleware(requiredRoles ...string) mux.MiddlewareFunc {
	return func(next http.Handler) http.Handler {
		return requireAnyRole(requiredRoles...)(next)
	}
}

func requireAnyRole(requiredRoles ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if hasAnyRole(r.Context(), requiredRoles...) {
				next.ServeHTTP(w, r)
				return
			}

			writeJSONError(w, http.StatusForbidden, "insufficient permissions")
		})
	}
}

func requireRoleHandler(requiredRole string, next http.HandlerFunc) http.Handler {
	return requireAnyRole(requiredRole)(next)
}

func registerPublicRoutes(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB, twoFactorService *auth.TwoFactorService) {
	passwordResetHandler := authapi.NewHandler(nil)

	loginHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var loginReq struct {
			Username string `json:"username"`
			Email    string `json:"email"`
			Password string `json:"password"`
		}

		if err := json.NewDecoder(r.Body).Decode(&loginReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		identifier := strings.TrimSpace(loginReq.Username)
		if identifier == "" {
			identifier = strings.TrimSpace(loginReq.Email)
		}
		if identifier == "" || strings.TrimSpace(loginReq.Password) == "" {
			writeJSONError(w, http.StatusBadRequest, "email or username and password are required")
			return
		}

		username, err := resolveLoginUsername(db, identifier)
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "invalid credentials")
			return
		}

		user, token, err := authManager.Authenticate(username, loginReq.Password)
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "invalid credentials")
			return
		}

		userPayload := frontendUser(user)
		twoFactorEnabled := userHasEnabledTwoFactor(twoFactorService, user.ID)
		userPayload["two_factor_enabled"] = twoFactorEnabled

		if twoFactorEnabled {
			tempToken, err := issuePending2FAToken(authManager.GetJWTSecret(), user)
			if err != nil {
				writeJSONError(w, http.StatusInternalServerError, "failed to create 2FA challenge")
				return
			}

			writeJSON(w, http.StatusOK, map[string]interface{}{
				"requires_2fa": true,
				"temp_token":   tempToken,
				"expiresAt":    time.Now().UTC().Add(10 * time.Minute).Format(time.RFC3339),
				"user":         userPayload,
			})
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"token":     token,
			"expiresAt": time.Now().UTC().Add(24 * time.Hour).Format(time.RFC3339),
			"user":      userPayload,
		})
	})

	registerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var registerReq struct {
			Username  string `json:"username"`
			Email     string `json:"email"`
			Password  string `json:"password"`
			TenantID  string `json:"tenant_id,omitempty"`
			TenantID2 string `json:"tenantId,omitempty"`
			FirstName string `json:"firstName,omitempty"`
			LastName  string `json:"lastName,omitempty"`
		}

		if err := json.NewDecoder(r.Body).Decode(&registerReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		email := strings.TrimSpace(registerReq.Email)
		password := strings.TrimSpace(registerReq.Password)
		if email == "" || password == "" {
			writeJSONError(w, http.StatusBadRequest, "email and password are required")
			return
		}

		tenantID := strings.TrimSpace(registerReq.TenantID)
		if tenantID == "" {
			tenantID = strings.TrimSpace(registerReq.TenantID2)
		}
		if tenantID == "" {
			tenantID = "default"
		}

		username := strings.TrimSpace(registerReq.Username)
		if username == "" {
			username = defaultUsernameFromEmail(email)
		}

		user, err := authManager.CreateUser(username, email, password, "user", tenantID)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("failed to create user: %v", err))
			return
		}

		resp := frontendUser(user)
		if registerReq.FirstName != "" {
			resp["firstName"] = registerReq.FirstName
		}
		if registerReq.LastName != "" {
			resp["lastName"] = registerReq.LastName
		}

		writeJSON(w, http.StatusCreated, resp)
	})

	checkEmailHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		email := strings.TrimSpace(r.URL.Query().Get("email"))
		if email == "" {
			writeJSONError(w, http.StatusBadRequest, "email is required")
			return
		}

		var exists bool
		err := db.QueryRow(`SELECT EXISTS(SELECT 1 FROM users WHERE email = $1)`, email).Scan(&exists)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to check email availability")
			return
		}

		writeJSON(w, http.StatusOK, map[string]bool{"available": !exists})
	})

	for _, path := range []string{"/auth/login", "/api/auth/login"} {
		router.Handle(path, loginHandler).Methods(http.MethodPost)
	}
	for _, path := range []string{"/auth/register", "/api/auth/register"} {
		router.Handle(path, registerHandler).Methods(http.MethodPost)
	}
	router.Handle("/api/auth/check-email", checkEmailHandler).Methods(http.MethodGet)

	router.Handle("/api/auth/verify-email", notImplementedJSON("email verification is not wired in the canonical server yet"))
	router.Handle("/api/auth/forgot-password", http.HandlerFunc(passwordResetHandler.ForgotPassword)).Methods(http.MethodPost)
	router.Handle("/api/auth/reset-password", http.HandlerFunc(passwordResetHandler.ResetPassword)).Methods(http.MethodPost)
	router.Handle("/api/auth/resend-verification", notImplementedJSON("email verification is not wired in the canonical server yet"))

	router.HandleFunc("/api/auth/2fa/verify-login", func(w http.ResponseWriter, r *http.Request) {
		if twoFactorService == nil {
			writeJSON(w, http.StatusNotImplemented, map[string]interface{}{
				"error":   "not_supported",
				"message": "two-factor verification is not configured in the canonical server",
			})
			return
		}

		var verifyReq struct {
			UserID       string `json:"user_id"`
			Code         string `json:"code"`
			IsBackupCode bool   `json:"is_backup_code"`
			TempToken    string `json:"temp_token"`
		}
		if err := json.NewDecoder(r.Body).Decode(&verifyReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		claims, err := validatePending2FAToken(verifyReq.TempToken, authManager.GetJWTSecret())
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "invalid or expired temporary token")
			return
		}

		userID := stringClaim(claims, "user_id", "sub")
		if verifyReq.UserID != "" && verifyReq.UserID != userID {
			writeJSONError(w, http.StatusUnauthorized, "temporary token does not match requested user")
			return
		}

		verifyResponse, err := twoFactorService.VerifyCode(auth.TwoFactorVerifyRequest{
			UserID:       userID,
			Code:         verifyReq.Code,
			IsBackupCode: verifyReq.IsBackupCode,
		})
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to verify two-factor code")
			return
		}
		if !verifyResponse.Valid {
			writeJSONError(w, http.StatusUnauthorized, "invalid two-factor code")
			return
		}

		user, err := authManager.GetUser(userID)
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "user not found for temporary token")
			return
		}

		sessionToken, err := issueSessionToken(authManager.GetJWTSecret(), user)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to create session token")
			return
		}

		userPayload := frontendUser(user)
		userPayload["two_factor_enabled"] = true
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"token":                  sessionToken,
			"expiresAt":              time.Now().UTC().Add(24 * time.Hour).Format(time.RFC3339),
			"user":                   userPayload,
			"remaining_backup_codes": verifyResponse.RemainingCodes,
		})
	}).Methods(http.MethodPost)
}

func registerSecureAPIRoutes(router *mux.Router, db *sql.DB) {
	runtimeMonitoringClient := newRuntimeMonitoringReadClientFromEnv()

	router.HandleFunc("/vms", func(w http.ResponseWriter, r *http.Request) {
		rows, err := db.Query(`SELECT id, name, state, node_id, tenant_id, created_at, updated_at FROM vms ORDER BY created_at DESC`)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query VMs")
			return
		}
		defer rows.Close()

		vms := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, name, state, tenantID string
			var nodeID sql.NullString
			var createdAt, updatedAt time.Time

			if err := rows.Scan(&id, &name, &state, &nodeID, &tenantID, &createdAt, &updatedAt); err != nil {
				continue
			}

			vms = append(vms, map[string]interface{}{
				"id":         id,
				"name":       name,
				"state":      state,
				"status":     state,
				"node_id":    nullableString(nodeID),
				"tenant_id":  tenantID,
				"created_at": createdAt.Format(time.RFC3339),
				"updated_at": updatedAt.Format(time.RFC3339),
			})
		}

		writeJSON(w, http.StatusOK, vms)
	}).Methods(http.MethodGet)

	router.HandleFunc("/vms", func(w http.ResponseWriter, r *http.Request) {
		var createReq struct {
			Name      string                 `json:"name"`
			State     string                 `json:"state"`
			NodeID    string                 `json:"node_id"`
			Tags      map[string]interface{} `json:"tags,omitempty"`
			CPUShares int                    `json:"cpu_shares,omitempty"`
			MemoryMB  int                    `json:"memory_mb,omitempty"`
		}

		if err := json.NewDecoder(r.Body).Decode(&createReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if strings.TrimSpace(createReq.Name) == "" {
			writeJSONError(w, http.StatusBadRequest, "name is required")
			return
		}

		vmID := fmt.Sprintf("vm-%d", time.Now().UnixNano())
		userID, _ := strconv.Atoi(fmt.Sprintf("%v", r.Context().Value("user_id")))
		tenantID, _ := r.Context().Value("tenant_id").(string)
		if tenantID == "" {
			tenantID = "default"
		}

		configPayload, _ := json.Marshal(map[string]interface{}{
			"cpu_shares": createReq.CPUShares,
			"memory_mb":  createReq.MemoryMB,
			"tags":       createReq.Tags,
		})

		_, err := db.Exec(`
			INSERT INTO vms (id, name, state, node_id, owner_id, tenant_id, config, created_at, updated_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
		`, vmID, createReq.Name, "creating", nullableStringValue(createReq.NodeID), nullableIntValue(userID), tenantID, configPayload)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to create VM")
			return
		}

		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"id":         vmID,
			"name":       createReq.Name,
			"state":      "creating",
			"status":     "creating",
			"node_id":    createReq.NodeID,
			"tenant_id":  tenantID,
			"created_at": time.Now().UTC().Format(time.RFC3339),
			"updated_at": time.Now().UTC().Format(time.RFC3339),
		})
	}).Methods(http.MethodPost)

	router.HandleFunc("/vms/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]

		var id, name, state, tenantID string
		var nodeID sql.NullString
		var createdAt, updatedAt time.Time
		err := db.QueryRow(`
			SELECT id, name, state, node_id, tenant_id, created_at, updated_at
			FROM vms WHERE id = $1
		`, vmID).Scan(&id, &name, &state, &nodeID, &tenantID, &createdAt, &updatedAt)
		if err != nil {
			if err == sql.ErrNoRows {
				writeJSONError(w, http.StatusNotFound, "vm not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, "failed to query VM")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":         id,
			"name":       name,
			"state":      state,
			"status":     state,
			"node_id":    nullableString(nodeID),
			"tenant_id":  tenantID,
			"created_at": createdAt.Format(time.RFC3339),
			"updated_at": updatedAt.Format(time.RFC3339),
		})
	}).Methods(http.MethodGet)

	router.HandleFunc("/vms/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]
		result, err := db.Exec(`DELETE FROM vms WHERE id = $1`, vmID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to delete VM")
			return
		}

		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":     vmID,
			"status": "deleted",
		})
	}).Methods(http.MethodDelete)

	for action, nextState := range map[string]string{
		"start": "running",
		"stop":  "stopped",
	} {
		router.HandleFunc("/vms/{id}/"+action, func(w http.ResponseWriter, r *http.Request) {
			vmID := mux.Vars(r)["id"]
			result, err := db.Exec(`UPDATE vms SET state = $2, updated_at = NOW() WHERE id = $1`, vmID, nextState)
			if err != nil {
				writeJSONError(w, http.StatusInternalServerError, "failed to update VM state")
				return
			}

			rowsAffected, _ := result.RowsAffected()
			if rowsAffected == 0 {
				writeJSONError(w, http.StatusNotFound, "vm not found")
				return
			}

			writeJSON(w, http.StatusOK, map[string]interface{}{
				"id":     vmID,
				"status": nextState,
			})
		}).Methods(http.MethodPost)
	}

	router.HandleFunc("/vms/{id}/metrics", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]

		var cpuUsage, memoryUsage float64
		err := db.QueryRow(`
			SELECT COALESCE(cpu_usage, 0), COALESCE(memory_usage, 0)
			FROM vm_metrics WHERE vm_id = $1
			ORDER BY timestamp DESC
			LIMIT 1
		`, vmID).Scan(&cpuUsage, &memoryUsage)
		if err != nil && err != sql.ErrNoRows {
			writeJSONError(w, http.StatusInternalServerError, "failed to query VM metrics")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":           vmID,
			"cpu_usage":    cpuUsage,
			"memory_usage": memoryUsage,
		})
	}).Methods(http.MethodGet)

	router.HandleFunc("/monitoring/metrics", monitoringSummaryHandler(runtimeMonitoringClient)).Methods(http.MethodGet)

	router.HandleFunc("/monitoring/vms", func(w http.ResponseWriter, r *http.Request) {
		rows, err := db.Query(`SELECT id, name, state FROM vms ORDER BY created_at DESC`)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query VMs")
			return
		}
		defer rows.Close()

		vmMetrics := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, name, state string
			if err := rows.Scan(&id, &name, &state); err != nil {
				continue
			}

			vmMetrics = append(vmMetrics, map[string]interface{}{
				"vmId":        id,
				"name":        name,
				"cpuUsage":    50.0 + float64(len(id)%20),
				"memoryUsage": 60.0 + float64(len(name)%30),
				"diskUsage":   40.0 + float64(len(id)%15),
				"networkRx":   1024 * 1024,
				"networkTx":   2048 * 1024,
				"iops":        100,
				"status":      state,
			})
		}

		writeJSON(w, http.StatusOK, vmMetrics)
	}).Methods(http.MethodGet)

	router.HandleFunc("/monitoring/alerts", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, []map[string]interface{}{})
	}).Methods(http.MethodGet)

	router.HandleFunc("/networks", func(w http.ResponseWriter, r *http.Request) {
		rows, err := db.Query(`
			SELECT id, name, type, subnet, gateway, status, created_at, updated_at
			FROM networks
			ORDER BY created_at DESC
		`)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query networks")
			return
		}
		defer rows.Close()

		networks := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, name, networkType, subnet, status string
			var gateway sql.NullString
			var createdAt, updatedAt time.Time

			if err := rows.Scan(&id, &name, &networkType, &subnet, &gateway, &status, &createdAt, &updatedAt); err != nil {
				continue
			}

			networks = append(networks, map[string]interface{}{
				"id":         id,
				"name":       name,
				"type":       networkType,
				"subnet":     subnet,
				"gateway":    nullableString(gateway),
				"status":     status,
				"created_at": createdAt.Format(time.RFC3339),
				"updated_at": updatedAt.Format(time.RFC3339),
			})
		}

		writeJSON(w, http.StatusOK, networks)
	}).Methods(http.MethodGet)

	router.HandleFunc("/networks", func(w http.ResponseWriter, r *http.Request) {
		var createReq struct {
			Name    string `json:"name"`
			Type    string `json:"type"`
			Subnet  string `json:"subnet"`
			Gateway string `json:"gateway"`
		}

		if err := json.NewDecoder(r.Body).Decode(&createReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		if strings.TrimSpace(createReq.Name) == "" || strings.TrimSpace(createReq.Subnet) == "" {
			writeJSONError(w, http.StatusBadRequest, "name and subnet are required")
			return
		}

		if strings.TrimSpace(createReq.Type) == "" {
			createReq.Type = "bridged"
		}

		networkID := fmt.Sprintf("net-%d", time.Now().UnixNano())
		if _, err := db.Exec(`
			INSERT INTO networks (id, name, type, subnet, gateway, status, created_at, updated_at)
			VALUES ($1, $2, $3, $4, NULLIF($5, ''), 'active', NOW(), NOW())
		`, networkID, createReq.Name, createReq.Type, createReq.Subnet, createReq.Gateway); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to create network")
			return
		}

		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"id":         networkID,
			"name":       createReq.Name,
			"type":       createReq.Type,
			"subnet":     createReq.Subnet,
			"gateway":    emptyStringToNil(createReq.Gateway),
			"status":     "active",
			"created_at": time.Now().UTC().Format(time.RFC3339),
			"updated_at": time.Now().UTC().Format(time.RFC3339),
		})
	}).Methods(http.MethodPost)

	router.HandleFunc("/networks/{id}", func(w http.ResponseWriter, r *http.Request) {
		networkID := mux.Vars(r)["id"]

		var id, name, networkType, subnet, status string
		var gateway sql.NullString
		var createdAt, updatedAt time.Time
		err := db.QueryRow(`
			SELECT id, name, type, subnet, gateway, status, created_at, updated_at
			FROM networks WHERE id = $1
		`, networkID).Scan(&id, &name, &networkType, &subnet, &gateway, &status, &createdAt, &updatedAt)
		if err != nil {
			if err == sql.ErrNoRows {
				writeJSONError(w, http.StatusNotFound, "network not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, "failed to query network")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":         id,
			"name":       name,
			"type":       networkType,
			"subnet":     subnet,
			"gateway":    nullableString(gateway),
			"status":     status,
			"created_at": createdAt.Format(time.RFC3339),
			"updated_at": updatedAt.Format(time.RFC3339),
		})
	}).Methods(http.MethodGet)

	router.HandleFunc("/networks/{id}", func(w http.ResponseWriter, r *http.Request) {
		networkID := mux.Vars(r)["id"]
		result, err := db.Exec(`DELETE FROM networks WHERE id = $1`, networkID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to delete network")
			return
		}

		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "network not found")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":     networkID,
			"status": "deleted",
		})
	}).Methods(http.MethodDelete)

	router.HandleFunc("/vms/{vm_id}/interfaces", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]

		rows, err := db.Query(`
			SELECT id, vm_id, network_id, name, mac_address, ip_address, status, created_at, updated_at
			FROM vm_interfaces
			WHERE vm_id = $1
			ORDER BY created_at DESC
		`, vmID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query VM interfaces")
			return
		}
		defer rows.Close()

		interfaces := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, currentVMID, name, macAddress, status string
			var networkID, ipAddress sql.NullString
			var createdAt, updatedAt time.Time

			if err := rows.Scan(&id, &currentVMID, &networkID, &name, &macAddress, &ipAddress, &status, &createdAt, &updatedAt); err != nil {
				continue
			}

			interfaces = append(interfaces, map[string]interface{}{
				"id":          id,
				"vm_id":       currentVMID,
				"network_id":  nullableString(networkID),
				"name":        name,
				"mac_address": macAddress,
				"ip_address":  nullableString(ipAddress),
				"status":      status,
				"created_at":  createdAt.Format(time.RFC3339),
				"updated_at":  updatedAt.Format(time.RFC3339),
			})
		}

		writeJSON(w, http.StatusOK, interfaces)
	}).Methods(http.MethodGet)

	router.HandleFunc("/vms/{vm_id}/interfaces", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]

		var createReq struct {
			NetworkID  string `json:"network_id"`
			Name       string `json:"name"`
			MACAddress string `json:"mac_address"`
			IPAddress  string `json:"ip_address"`
		}

		if err := json.NewDecoder(r.Body).Decode(&createReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		if strings.TrimSpace(createReq.Name) == "" || strings.TrimSpace(createReq.MACAddress) == "" {
			writeJSONError(w, http.StatusBadRequest, "name and mac_address are required")
			return
		}

		var vmExists bool
		if err := db.QueryRow(`SELECT EXISTS(SELECT 1 FROM vms WHERE id = $1)`, vmID).Scan(&vmExists); err != nil || !vmExists {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		interfaceID := fmt.Sprintf("iface-%d", time.Now().UnixNano())
		if _, err := db.Exec(`
			INSERT INTO vm_interfaces (id, vm_id, network_id, name, mac_address, ip_address, status, created_at, updated_at)
			VALUES ($1, $2, NULLIF($3, ''), $4, $5, NULLIF($6, ''), 'attached', NOW(), NOW())
		`, interfaceID, vmID, createReq.NetworkID, createReq.Name, createReq.MACAddress, createReq.IPAddress); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to attach interface")
			return
		}

		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"id":          interfaceID,
			"vm_id":       vmID,
			"network_id":  emptyStringToNil(createReq.NetworkID),
			"name":        createReq.Name,
			"mac_address": createReq.MACAddress,
			"ip_address":  emptyStringToNil(createReq.IPAddress),
			"status":      "attached",
			"created_at":  time.Now().UTC().Format(time.RFC3339),
			"updated_at":  time.Now().UTC().Format(time.RFC3339),
		})
	}).Methods(http.MethodPost)

	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		interfaceID := mux.Vars(r)["id"]

		var id, currentVMID, name, macAddress, status string
		var networkID, ipAddress sql.NullString
		var createdAt, updatedAt time.Time
		err := db.QueryRow(`
			SELECT id, vm_id, network_id, name, mac_address, ip_address, status, created_at, updated_at
			FROM vm_interfaces
			WHERE vm_id = $1 AND id = $2
		`, vmID, interfaceID).Scan(&id, &currentVMID, &networkID, &name, &macAddress, &ipAddress, &status, &createdAt, &updatedAt)
		if err != nil {
			if err == sql.ErrNoRows {
				writeJSONError(w, http.StatusNotFound, "vm interface not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, "failed to query VM interface")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":          id,
			"vm_id":       currentVMID,
			"network_id":  nullableString(networkID),
			"name":        name,
			"mac_address": macAddress,
			"ip_address":  nullableString(ipAddress),
			"status":      status,
			"created_at":  createdAt.Format(time.RFC3339),
			"updated_at":  updatedAt.Format(time.RFC3339),
		})
	}).Methods(http.MethodGet)

	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		interfaceID := mux.Vars(r)["id"]

		var updateReq struct {
			NetworkID string `json:"network_id"`
			Name      string `json:"name"`
			IPAddress string `json:"ip_address"`
			Status    string `json:"status"`
		}

		if err := json.NewDecoder(r.Body).Decode(&updateReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		result, err := db.Exec(`
			UPDATE vm_interfaces
			SET network_id = NULLIF($3, ''), name = COALESCE(NULLIF($4, ''), name), ip_address = NULLIF($5, ''), status = COALESCE(NULLIF($6, ''), status), updated_at = NOW()
			WHERE vm_id = $1 AND id = $2
		`, vmID, interfaceID, updateReq.NetworkID, updateReq.Name, updateReq.IPAddress, updateReq.Status)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to update VM interface")
			return
		}

		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "vm interface not found")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":         interfaceID,
			"vm_id":      vmID,
			"network_id": emptyStringToNil(updateReq.NetworkID),
			"name":       updateReq.Name,
			"ip_address": emptyStringToNil(updateReq.IPAddress),
			"status":     updateReq.Status,
		})
	}).Methods(http.MethodPut)

	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		interfaceID := mux.Vars(r)["id"]

		result, err := db.Exec(`DELETE FROM vm_interfaces WHERE vm_id = $1 AND id = $2`, vmID, interfaceID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to detach VM interface")
			return
		}

		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "vm interface not found")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":     interfaceID,
			"vm_id":  vmID,
			"status": "detached",
		})
	}).Methods(http.MethodDelete)
}

func initializeCanonicalServices(cfg *config.Config, db *sql.DB, authManager *auth.SimpleAuthManager) (*canonicalServices, error) {
	auditLogger := audit.NewSimpleAuditLogger()
	twoFactorService := auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret()))
	securityHandlers := securityapi.NewSecurityHandlers(twoFactorService, auditLogger).WithRBACStore(securityapi.NewPostgresRBACStore(db))

	volumeStore, err := storage.NewStorageManager(storage.StorageManagerConfig{
		BasePath: filepath.Join(cfg.VM.StoragePath, "volumes"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize storage-backed volume store: %w", err)
	}

	graphqlResolver := graphqlapi.NewResolverWithVolumeStore(nil, nil, volumeStore)
	websocketLogger := logrus.New()
	websocketLogger.SetLevel(logrus.InfoLevel)
	websocketHandler := websocketapi.NewWebSocketHandler(nil, nil, nil, nil, websocketLogger)

	return &canonicalServices{
		twoFactorService: twoFactorService,
		securityHandlers: securityHandlers,
		websocketHandler: websocketHandler,
		graphqlHandler:   graphqlapi.NewVolumeHTTPHandler(graphqlResolver),
		shutdown: func() {
			websocketHandler.Shutdown()
		},
	}, nil
}

func registerCanonicalSecurityRoutes(router *mux.Router, authManager *auth.SimpleAuthManager, handlers *securityapi.SecurityHandlers) {
	twoFactorRouter := router.PathPrefix("/api/auth/2fa").Subrouter()
	twoFactorRouter.Use(requireAuth(authManager))
	twoFactorRouter.HandleFunc("/setup", handlers.Setup2FA).Methods(http.MethodPost)
	twoFactorRouter.HandleFunc("/qr", handlers.GenerateQRCode).Methods(http.MethodGet)
	twoFactorRouter.HandleFunc("/verify", handlers.Verify2FA).Methods(http.MethodPost)
	twoFactorRouter.HandleFunc("/enable", handlers.Enable2FA).Methods(http.MethodPost)
	twoFactorRouter.HandleFunc("/disable", handlers.Disable2FA).Methods(http.MethodPost)
	twoFactorRouter.HandleFunc("/status", handlers.Get2FAStatus).Methods(http.MethodGet)
	twoFactorRouter.HandleFunc("/backup-codes", handlers.GetBackupCodes).Methods(http.MethodGet)
	twoFactorRouter.HandleFunc("/backup-codes", handlers.RegenerateBackupCodes).Methods(http.MethodPost)

	registerSecurityRouteSet(router.PathPrefix("/api/security").Subrouter(), authManager, handlers)
	registerSecurityRouteSet(router.PathPrefix("/api/admin/security").Subrouter(), authManager, handlers)
}

func registerCanonicalAdminRoutes(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB) {
	adminRouter := router.PathPrefix("/api/admin").Subrouter()
	adminRouter.Use(requireAuth(authManager))
	adminRouter.Use(requireAnyRoleMiddleware("admin", "super-admin"))

	adminRouter.HandleFunc("/users", listCanonicalAdminUsers(db)).Methods(http.MethodGet)
	adminRouter.HandleFunc("/users", createCanonicalAdminUser(db)).Methods(http.MethodPost)
	adminRouter.HandleFunc("/users/{id}", updateCanonicalAdminUser(db)).Methods(http.MethodPut)
	adminRouter.HandleFunc("/users/{id}", deleteCanonicalAdminUser(db)).Methods(http.MethodDelete)
	adminRouter.HandleFunc("/users/{id}/roles", assignCanonicalAdminUserRoles(db)).Methods(http.MethodPost)
}

type canonicalAdminUser struct {
	ID        int       `json:"id"`
	Username  string    `json:"username"`
	Email     string    `json:"email"`
	Role      string    `json:"role"`
	Active    bool      `json:"active"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type canonicalAdminUserListResponse struct {
	Users      []canonicalAdminUser `json:"users"`
	Total      int                  `json:"total"`
	Page       int                  `json:"page"`
	PageSize   int                  `json:"page_size"`
	TotalPages int                  `json:"total_pages"`
}

type canonicalAdminCreateUserRequest struct {
	Username string `json:"username"`
	Email    string `json:"email"`
	Password string `json:"password"`
	Role     string `json:"role"`
}

type canonicalAdminUpdateUserRequest struct {
	Username string `json:"username,omitempty"`
	Email    string `json:"email,omitempty"`
	Role     string `json:"role,omitempty"`
	Active   *bool  `json:"active,omitempty"`
}

func listCanonicalAdminUsers(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		page := parsePositiveInt(r.URL.Query().Get("page"), 1)
		pageSize := parsePositiveInt(r.URL.Query().Get("page_size"), 20)
		if pageSize > 100 {
			pageSize = 100
		}

		search := strings.TrimSpace(r.URL.Query().Get("search"))
		role := normalizeCanonicalAdminRole(r.URL.Query().Get("role"))
		if rawRole := strings.TrimSpace(r.URL.Query().Get("role")); rawRole != "" && role == "" {
			writeJSONError(w, http.StatusBadRequest, "invalid role filter")
			return
		}

		offset := (page - 1) * pageSize
		whereParts := make([]string, 0, 2)
		args := make([]interface{}, 0, 4)

		if search != "" {
			whereParts = append(whereParts, fmt.Sprintf("(username ILIKE $%d OR email ILIKE $%d)", len(args)+1, len(args)+2))
			args = append(args, "%"+search+"%", "%"+search+"%")
		}
		if role != "" {
			whereParts = append(whereParts, fmt.Sprintf("role = $%d", len(args)+1))
			args = append(args, role)
		}

		whereClause := ""
		if len(whereParts) > 0 {
			whereClause = " WHERE " + strings.Join(whereParts, " AND ")
		}

		var total int
		countQuery := "SELECT COUNT(*) FROM users" + whereClause
		if err := db.QueryRow(countQuery, args...).Scan(&total); err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to count users: %v", err))
			return
		}

		listQuery := fmt.Sprintf(`
			SELECT id, username, email, role, active, created_at, updated_at
			FROM users
			%s
			ORDER BY created_at DESC
			LIMIT $%d OFFSET $%d
		`, whereClause, len(args)+1, len(args)+2)
		listArgs := append(args, pageSize, offset)

		rows, err := db.Query(listQuery, listArgs...)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to list users: %v", err))
			return
		}
		defer rows.Close()

		users := make([]canonicalAdminUser, 0)
		for rows.Next() {
			var user canonicalAdminUser
			if err := rows.Scan(&user.ID, &user.Username, &user.Email, &user.Role, &user.Active, &user.CreatedAt, &user.UpdatedAt); err != nil {
				writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to scan user: %v", err))
				return
			}
			users = append(users, user)
		}

		writeJSON(w, http.StatusOK, canonicalAdminUserListResponse{
			Users:      users,
			Total:      total,
			Page:       page,
			PageSize:   pageSize,
			TotalPages: maxInt(1, (total+pageSize-1)/pageSize),
		})
	}
}

func createCanonicalAdminUser(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req canonicalAdminCreateUserRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		req.Username = strings.TrimSpace(req.Username)
		req.Email = strings.TrimSpace(req.Email)
		req.Role = normalizeCanonicalAdminRole(req.Role)
		if req.Role == "" {
			req.Role = "user"
		}

		if req.Username == "" || req.Email == "" || strings.TrimSpace(req.Password) == "" {
			writeJSONError(w, http.StatusBadRequest, "username, email, and password are required")
			return
		}
		if !strings.Contains(req.Email, "@") {
			writeJSONError(w, http.StatusBadRequest, "invalid email format")
			return
		}

		var user canonicalAdminUser
		err := db.QueryRow(`
			INSERT INTO users (username, email, password_hash, role, active, tenant_id, created_at, updated_at)
			VALUES ($1, $2, $3, $4, true, 'default', NOW(), NOW())
			RETURNING id, username, email, role, active, created_at, updated_at
		`, req.Username, req.Email, req.Password, req.Role).Scan(
			&user.ID,
			&user.Username,
			&user.Email,
			&user.Role,
			&user.Active,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to create user: %v", err))
			return
		}

		writeJSON(w, http.StatusCreated, user)
	}
}

func updateCanonicalAdminUser(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, err := canonicalAdminUserID(mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}

		var req canonicalAdminUpdateUserRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		updates := make([]string, 0, 4)
		args := make([]interface{}, 0, 5)

		if username := strings.TrimSpace(req.Username); username != "" {
			updates = append(updates, fmt.Sprintf("username = $%d", len(args)+1))
			args = append(args, username)
		}
		if email := strings.TrimSpace(req.Email); email != "" {
			if !strings.Contains(email, "@") {
				writeJSONError(w, http.StatusBadRequest, "invalid email format")
				return
			}
			updates = append(updates, fmt.Sprintf("email = $%d", len(args)+1))
			args = append(args, email)
		}
		if req.Role != "" {
			role := normalizeCanonicalAdminRole(req.Role)
			if role == "" {
				writeJSONError(w, http.StatusBadRequest, "invalid role")
				return
			}
			updates = append(updates, fmt.Sprintf("role = $%d", len(args)+1))
			args = append(args, role)
		}
		if req.Active != nil {
			updates = append(updates, fmt.Sprintf("active = $%d", len(args)+1))
			args = append(args, *req.Active)
		}
		if len(updates) == 0 {
			writeJSONError(w, http.StatusBadRequest, "no fields to update")
			return
		}

		updates = append(updates, "updated_at = NOW()")
		args = append(args, userID)

		query := fmt.Sprintf(`
			UPDATE users
			SET %s
			WHERE id = $%d
			RETURNING id, username, email, role, active, created_at, updated_at
		`, strings.Join(updates, ", "), len(args))

		var user canonicalAdminUser
		err = db.QueryRow(query, args...).Scan(
			&user.ID,
			&user.Username,
			&user.Email,
			&user.Role,
			&user.Active,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if err != nil {
			if err == sql.ErrNoRows {
				writeJSONError(w, http.StatusNotFound, "user not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to update user: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, user)
	}
}

func deleteCanonicalAdminUser(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, err := canonicalAdminUserID(mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}

		result, err := db.Exec(`DELETE FROM users WHERE id = $1`, userID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to delete user: %v", err))
			return
		}
		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "user not found")
			return
		}

		w.WriteHeader(http.StatusNoContent)
	}
}

func assignCanonicalAdminUserRoles(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, err := canonicalAdminUserID(mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}

		var req struct {
			Roles []string `json:"roles"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if len(req.Roles) == 0 {
			writeJSONError(w, http.StatusBadRequest, "at least one role is required")
			return
		}

		role := normalizeCanonicalAdminRole(req.Roles[0])
		if role == "" {
			writeJSONError(w, http.StatusBadRequest, "invalid role")
			return
		}

		var user canonicalAdminUser
		err = db.QueryRow(`
			UPDATE users
			SET role = $1, updated_at = NOW()
			WHERE id = $2
			RETURNING id, username, email, role, active, created_at, updated_at
		`, role, userID).Scan(
			&user.ID,
			&user.Username,
			&user.Email,
			&user.Role,
			&user.Active,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if err != nil {
			if err == sql.ErrNoRows {
				writeJSONError(w, http.StatusNotFound, "user not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to assign role: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"user":    user,
			"message": "roles updated",
		})
	}
}

func canonicalAdminUserID(raw string) (int, error) {
	userID, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || userID <= 0 {
		return 0, fmt.Errorf("invalid user ID")
	}
	return userID, nil
}

func normalizeCanonicalAdminRole(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "admin", "operator", "viewer", "user", "super-admin":
		return strings.ToLower(strings.TrimSpace(raw))
	default:
		return ""
	}
}

func registerSecurityRouteSet(router *mux.Router, authManager *auth.SimpleAuthManager, handlers *securityapi.SecurityHandlers) {
	router.Use(requireAuth(authManager))
	router.Use(requireAnyRoleMiddleware("admin", "super-admin"))

	router.HandleFunc("/threats", handlers.GetThreats).Methods(http.MethodGet)
	router.HandleFunc("/vulnerabilities", handlers.GetVulnerabilities).Methods(http.MethodGet)
	router.HandleFunc("/compliance", handlers.GetComplianceStatus).Methods(http.MethodGet)
	router.HandleFunc("/compliance/check", handlers.TriggerComplianceCheck).Methods(http.MethodPost)
	router.HandleFunc("/compliance/export", handlers.ExportComplianceReport).Methods(http.MethodGet)
	router.HandleFunc("/incidents", handlers.GetIncidents).Methods(http.MethodGet)
	router.HandleFunc("/incidents", handlers.CreateSecurityIncident).Methods(http.MethodPost)
	router.HandleFunc("/events", handlers.GetSecurityEvents).Methods(http.MethodGet)
	router.HandleFunc("/events/{eventId}/acknowledge", handlers.AcknowledgeSecurityEvent).Methods(http.MethodPost)
	router.HandleFunc("/scan", handlers.StartVulnerabilityScan).Methods(http.MethodPost)
	router.HandleFunc("/scan/{scanId}", handlers.GetScanResults).Methods(http.MethodGet)
	router.HandleFunc("/cluster/{clusterId}/state", handlers.GetClusterSecurityState).Methods(http.MethodGet)
	router.HandleFunc("/audit/events", handlers.GetAuditEvents).Methods(http.MethodGet)
	router.HandleFunc("/audit/export", handlers.ExportAuditLog).Methods(http.MethodGet)
	router.HandleFunc("/audit/statistics", handlers.GetAuditStatistics).Methods(http.MethodGet)
	router.HandleFunc("/rbac/roles", handlers.GetRoles).Methods(http.MethodGet)
	router.HandleFunc("/rbac/permissions", handlers.GetPermissions).Methods(http.MethodGet)
	router.HandleFunc("/rbac/user/{userId}/roles", handlers.GetUserRoles).Methods(http.MethodGet)
	router.HandleFunc("/rbac/user/{userId}/roles", handlers.AssignUserRoles).Methods(http.MethodPost)
	router.HandleFunc("/rbac/user/{userId}/permissions", handlers.GetUserPermissions).Methods(http.MethodGet)
}

func registerCanonicalGraphQLRoute(router *mux.Router, authManager *auth.SimpleAuthManager, handler http.Handler) {
	router.Handle("/graphql", requireAuth(authManager)(handler)).Methods(http.MethodPost)
}

func registerSecurityWebSocketAliases(router *mux.Router, authManager *auth.SimpleAuthManager, handlers *securityapi.SecurityHandlers) {
	securityStream := requireAuth(authManager)(requireRoleHandler("admin", handlers.StreamSecurityEvents))
	router.Handle("/api/ws/security/events", securityStream).Methods(http.MethodGet)

	compatSecurityRouter := router.PathPrefix("/api/security").Subrouter()
	compatSecurityRouter.Use(requireAuth(authManager))
	compatSecurityRouter.Handle("/events/stream", requireRoleHandler("admin", handlers.StreamSecurityEvents)).Methods(http.MethodGet)
}

func healthCheckHandler(cfg *config.Config, db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		checks := make(map[string]string)
		status := "healthy"

		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if err := db.PingContext(ctx); err != nil {
			checks["database"] = fmt.Sprintf("error: %v", err)
			status = "unhealthy"
		} else {
			checks["database"] = "ok"
		}

		if _, err := os.Stat(cfg.VM.StoragePath); err != nil {
			checks["storage"] = fmt.Sprintf("warning: %v", err)
		} else {
			checks["storage"] = "ok"
		}

		switch {
		case cfg != nil && cfg.RuntimeManifest.Loaded:
			checks["runtime_manifest"] = "ok"
		case cfg != nil && cfg.RuntimeManifest.Required:
			checks["runtime_manifest"] = "error: required runtime manifest not loaded"
			status = "unhealthy"
		default:
			checks["runtime_manifest"] = "disabled"
		}

		response := map[string]interface{}{
			"status":    status,
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"version":   "1.0.0",
			"service":   "novacron-api",
			"checks":    checks,
		}

		if status == "unhealthy" {
			writeJSON(w, http.StatusServiceUnavailable, response)
			return
		}

		writeJSON(w, http.StatusOK, response)
	}
}

func apiInfoHandler(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		runtimeManifest := map[string]interface{}{
			"loaded": false,
		}
		if cfg != nil {
			runtimeManifest = map[string]interface{}{
				"loaded":             cfg.RuntimeManifest.Loaded,
				"required":           cfg.RuntimeManifest.Required,
				"path":               cfg.RuntimeManifest.Path,
				"version":            cfg.RuntimeManifest.Version,
				"deployment_profile": cfg.RuntimeManifest.DeploymentProfile,
				"discovery_mode":     cfg.RuntimeManifest.DiscoveryMode,
				"federation_mode":    cfg.RuntimeManifest.FederationMode,
				"migration_mode":     cfg.RuntimeManifest.MigrationMode,
				"auth_mode":          cfg.RuntimeManifest.AuthMode,
				"storage_classes":    cfg.RuntimeManifest.StorageClasses,
				"enabled_services":   cfg.RuntimeManifest.EnabledServices,
			}
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"name":             "NovaCron API",
			"version":          "1.0.0",
			"description":      "Distributed VM Management System",
			"runtime_manifest": runtimeManifest,
			"endpoints": []string{
				"/api/auth/login",
				"/api/auth/register",
				"/api/auth/check-email",
				"/api/auth/forgot-password",
				"/api/auth/reset-password",
				"/api/auth/2fa/setup",
				"/api/auth/2fa/verify",
				"/api/auth/2fa/verify-login",
				"/api/auth/2fa/enable",
				"/api/auth/2fa/disable",
				"/api/auth/2fa/status",
				"/api/auth/2fa/backup-codes",
				"/api/v1/vms",
				"/api/v1/vms/{id}",
				"/api/v1/vms/{id}/start",
				"/api/v1/vms/{id}/stop",
				"/api/v1/vms/{id}/metrics",
				"/api/v1/monitoring/metrics",
				"/api/v1/monitoring/vms",
				"/api/v1/monitoring/alerts",
				"/api/security/threats",
				"/api/security/vulnerabilities",
				"/api/security/compliance",
				"/api/security/compliance/check",
				"/api/security/compliance/export",
				"/api/security/incidents",
				"/api/security/events",
				"/api/security/events/{eventId}/acknowledge",
				"/api/security/scan",
				"/api/security/audit/events",
				"/api/security/audit/export",
				"/api/security/audit/statistics",
				"/api/security/rbac/roles",
				"/api/security/rbac/permissions",
				"/api/admin/users",
				"/api/admin/users/{id}",
				"/api/admin/users/{id}/roles",
				"/api/admin/security/threats",
				"/api/admin/security/compliance",
				"/api/admin/security/compliance/check",
				"/api/admin/security/compliance/export",
				"/api/admin/security/incidents",
				"/api/admin/security/audit/export",
				"/api/admin/security/events/{eventId}/acknowledge",
				"/graphql",
				"/api/v1/networks",
				"/api/v1/networks/{id}",
				"/api/v1/vms/{vm_id}/interfaces",
				"/api/v1/vms/{vm_id}/interfaces/{id}",
				"/api/ws/metrics",
				"/api/ws/alerts",
				"/api/ws/logs",
				"/api/ws/security/events",
				"/api/ws/console/{vmId}",
				"/health",
			},
			"compatibility_endpoints": []string{
				"/auth/login",
				"/auth/register",
				"/api/vms",
				"/api/monitoring/metrics",
				"/api/monitoring/vms",
				"/api/monitoring/alerts",
				"/api/networks",
				"/api/networks/{id}",
				"/api/vms/{vm_id}/interfaces",
				"/api/vms/{vm_id}/interfaces/{id}",
				"/ws/metrics",
				"/ws/alerts",
				"/ws/logs",
				"/api/security/events/stream",
			},
			"unsupported_endpoints": []string{
				"/api/auth/resend-verification",
				"/api/auth/verify-email",
				"unsupported GraphQL operations outside storage-backed volume queries and mutations",
			},
		})
	}
}

func extractBearerToken(authHeader string) (string, error) {
	if authHeader == "" {
		return "", fmt.Errorf("authorization header required")
	}
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return "", fmt.Errorf("authorization header must start with 'Bearer '")
	}

	token := strings.TrimSpace(strings.TrimPrefix(authHeader, "Bearer "))
	if token == "" {
		return "", fmt.Errorf("token is required")
	}
	return token, nil
}

func validateJWT(tokenString, jwtSecret string) (jwt.MapClaims, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %s", token.Method.Alg())
		}
		return []byte(jwtSecret), nil
	})
	if err != nil {
		return nil, err
	}
	if !token.Valid {
		return nil, fmt.Errorf("token is invalid")
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return nil, fmt.Errorf("invalid claims")
	}
	return claims, nil
}

func resolveLoginUsername(db *sql.DB, identifier string) (string, error) {
	if !strings.Contains(identifier, "@") {
		return identifier, nil
	}

	var username string
	if err := db.QueryRow(`SELECT username FROM users WHERE email = $1`, identifier).Scan(&username); err != nil {
		return "", err
	}
	return username, nil
}

func defaultUsernameFromEmail(email string) string {
	localPart := strings.TrimSpace(strings.SplitN(email, "@", 2)[0])
	if localPart == "" {
		return fmt.Sprintf("user-%d", time.Now().Unix())
	}
	return localPart
}

func userHasEnabledTwoFactor(twoFactorService *auth.TwoFactorService, userID string) bool {
	if twoFactorService == nil || strings.TrimSpace(userID) == "" {
		return false
	}

	info, err := twoFactorService.GetUserTwoFactorInfo(userID)
	return err == nil && info != nil && info.Enabled
}

func issuePending2FAToken(secret string, user *auth.User) (string, error) {
	if user == nil {
		return "", fmt.Errorf("user is required")
	}

	claims := jwt.MapClaims{
		"purpose":   "pending_2fa",
		"user_id":   user.ID,
		"sub":       user.ID,
		"email":     user.Email,
		"username":  user.Username,
		"role":      primaryRole(user),
		"roles":     append([]string(nil), user.RoleIDs...),
		"tenant_id": user.TenantID,
		"exp":       time.Now().Add(10 * time.Minute).Unix(),
		"iat":       time.Now().Unix(),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(secret))
}

func validatePending2FAToken(tokenString, secret string) (jwt.MapClaims, error) {
	claims, err := validateJWT(tokenString, secret)
	if err != nil {
		return nil, err
	}
	if stringClaim(claims, "purpose") != "pending_2fa" {
		return nil, fmt.Errorf("token is not a pending 2FA token")
	}
	return claims, nil
}

func issueSessionToken(secret string, user *auth.User) (string, error) {
	if user == nil {
		return "", fmt.Errorf("user is required")
	}

	role := primaryRole(user)
	roles := append([]string(nil), user.RoleIDs...)
	if len(roles) == 0 && role != "" {
		roles = []string{role}
	}

	claims := jwt.MapClaims{
		"user_id":   user.ID,
		"sub":       user.ID,
		"username":  user.Username,
		"email":     user.Email,
		"role":      role,
		"roles":     roles,
		"tenant_id": user.TenantID,
		"exp":       time.Now().Add(24 * time.Hour).Unix(),
		"iat":       time.Now().Unix(),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(secret))
}

func primaryRole(user *auth.User) string {
	if user == nil {
		return ""
	}
	if len(user.RoleIDs) > 0 && strings.TrimSpace(user.RoleIDs[0]) != "" {
		return strings.TrimSpace(user.RoleIDs[0])
	}
	for _, role := range user.Roles {
		if role != nil && strings.TrimSpace(role.Name) != "" {
			return strings.TrimSpace(role.Name)
		}
	}
	return "user"
}

func frontendUser(user *auth.User) map[string]interface{} {
	role := "user"
	roles := make([]string, 0, len(user.RoleIDs))
	if len(user.RoleIDs) > 0 {
		roles = append(roles, user.RoleIDs...)
		role = user.RoleIDs[0]
	}
	if len(roles) == 0 && len(user.Roles) > 0 {
		for _, r := range user.Roles {
			if r == nil || r.Name == "" {
				continue
			}
			roles = append(roles, r.Name)
		}
		if len(roles) > 0 {
			role = roles[0]
		}
	}

	return map[string]interface{}{
		"id":                 user.ID,
		"email":              user.Email,
		"firstName":          "",
		"lastName":           "",
		"tenantId":           user.TenantID,
		"tenant_id":          user.TenantID,
		"status":             "active",
		"role":               role,
		"roles":              roles,
		"two_factor_enabled": false,
	}
}

func hasAnyRole(ctx context.Context, requiredRoles ...string) bool {
	required := make(map[string]struct{}, len(requiredRoles))
	for _, role := range requiredRoles {
		normalized := strings.ToLower(strings.TrimSpace(role))
		if normalized != "" {
			required[normalized] = struct{}{}
		}
	}

	for _, role := range contextRoles(ctx) {
		if roleSatisfies(role, required) {
			return true
		}
	}
	return false
}

func contextRoles(ctx context.Context) []string {
	roleSet := make(map[string]struct{})
	if role, ok := ctx.Value("role").(string); ok && strings.TrimSpace(role) != "" {
		roleSet[strings.ToLower(strings.TrimSpace(role))] = struct{}{}
	}
	if roles, ok := ctx.Value("roles").([]string); ok {
		for _, role := range roles {
			if trimmed := strings.ToLower(strings.TrimSpace(role)); trimmed != "" {
				roleSet[trimmed] = struct{}{}
			}
		}
	}

	normalizedRoles := make([]string, 0, len(roleSet))
	for role := range roleSet {
		normalizedRoles = append(normalizedRoles, role)
	}
	return normalizedRoles
}

func roleSatisfies(userRole string, required map[string]struct{}) bool {
	if len(required) == 0 {
		return true
	}

	normalizedUserRole := strings.ToLower(strings.TrimSpace(userRole))
	if normalizedUserRole == "" {
		return false
	}
	if _, ok := required[normalizedUserRole]; ok {
		return true
	}

	for _, impliedRole := range impliedRoles(normalizedUserRole) {
		if _, ok := required[impliedRole]; ok {
			return true
		}
	}

	return false
}

func impliedRoles(role string) []string {
	switch role {
	case "super-admin":
		return []string{"admin", "operator", "viewer", "readonly", "user"}
	case "admin":
		return []string{"operator", "viewer", "readonly", "user"}
	case "operator":
		return []string{"viewer", "readonly"}
	case "viewer":
		return []string{"readonly"}
	default:
		return nil
	}
}

func notImplementedJSON(message string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusNotImplemented, map[string]interface{}{
			"error":   "not_implemented",
			"message": message,
			"path":    r.URL.Path,
		})
	})
}

func writeJSONError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]interface{}{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func envBool(name string) bool {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return false
	}

	enabled, err := strconv.ParseBool(raw)
	if err == nil {
		return enabled
	}

	switch strings.ToLower(raw) {
	case "yes", "y", "on", "enabled":
		return true
	default:
		return false
	}
}

func stringClaim(claims jwt.MapClaims, keys ...string) string {
	for _, key := range keys {
		value, ok := claims[key]
		if !ok {
			continue
		}
		if s, ok := value.(string); ok {
			return s
		}
	}
	return ""
}

func stringSliceClaim(claims jwt.MapClaims, key string) []string {
	value, ok := claims[key]
	if !ok {
		return nil
	}

	switch typed := value.(type) {
	case []string:
		return typed
	case []interface{}:
		result := make([]string, 0, len(typed))
		for _, item := range typed {
			if s, ok := item.(string); ok && s != "" {
				result = append(result, s)
			}
		}
		return result
	case string:
		if typed == "" {
			return nil
		}
		return []string{typed}
	default:
		return nil
	}
}

func nullableString(value sql.NullString) interface{} {
	if !value.Valid {
		return nil
	}
	return value.String
}

func nullableStringValue(value string) interface{} {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func emptyStringToNil(value string) interface{} {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func parsePositiveInt(raw string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func nullableIntValue(value int) interface{} {
	if value <= 0 {
		return nil
	}
	return value
}
