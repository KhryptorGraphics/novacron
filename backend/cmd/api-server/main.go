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
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	_ "github.com/lib/pq"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

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

	router := mux.NewRouter()
	router.StrictSlash(true)

	corsHandler := buildCORSHandler(cfg)

	registerPublicRoutes(router, authManager, db)

	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.Use(requireAuth(authManager))
	registerSecureAPIRoutes(apiRouter, db)

	apiV1Router := router.PathPrefix("/api/v1").Subrouter()
	apiV1Router.Use(requireAuth(authManager))
	registerSecureAPIRoutes(apiV1Router, db)

	registerExplicitlyUnsupportedRoutes(router)

	router.HandleFunc("/health", healthCheckHandler(cfg, db)).Methods(http.MethodGet)
	router.HandleFunc("/api/info", apiInfoHandler()).Methods(http.MethodGet)

	server := &http.Server{
		Addr:         ":" + cfg.Server.APIPort,
		Handler:      corsHandler(router),
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

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
			tenant_id VARCHAR(255) DEFAULT 'default',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
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

func registerPublicRoutes(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB) {
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

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"token":     token,
			"expiresAt": time.Now().UTC().Add(24 * time.Hour).Format(time.RFC3339),
			"user":      frontendUser(user),
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

	for _, path := range []string{
		"/api/auth/forgot-password",
		"/api/auth/reset-password",
		"/api/auth/resend-verification",
		"/api/auth/verify-email",
		"/api/auth/2fa/setup",
		"/api/auth/2fa/verify",
		"/api/auth/2fa/verify-login",
		"/api/auth/2fa/enable",
		"/api/auth/2fa/disable",
		"/api/auth/2fa/status",
		"/api/auth/2fa/backup-codes",
	} {
		router.Handle(path, notImplementedJSON("auth capability is not wired in the canonical server yet"))
	}
}

func registerSecureAPIRoutes(router *mux.Router, db *sql.DB) {
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

	router.HandleFunc("/monitoring/metrics", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"currentCpuUsage":         45.2,
			"currentMemoryUsage":      72.1,
			"currentDiskUsage":        58.3,
			"currentNetworkUsage":     125.7,
			"cpuChangePercentage":     5.2,
			"memoryChangePercentage":  -2.1,
			"diskChangePercentage":    1.8,
			"networkChangePercentage": 12.5,
			"timeLabels":              []string{"10:00", "10:30", "11:00", "11:30", "12:00"},
			"cpuAnalysis":             "CPU usage shows normal workday patterns.",
			"memoryAnalysis":          "Memory allocation is healthy.",
		})
	}).Methods(http.MethodGet)

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
}

func registerExplicitlyUnsupportedRoutes(router *mux.Router) {
	router.Handle("/graphql", notImplementedJSON("GraphQL is not wired into the canonical API server yet"))

	for _, prefix := range []string{
		"/api/security/",
		"/api/admin/security/",
		"/api/ws/",
	} {
		router.PathPrefix(prefix).Handler(notImplementedJSON("this API surface is not wired in the canonical server yet"))
	}
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

func apiInfoHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"name":        "NovaCron API",
			"version":     "1.0.0",
			"description": "Distributed VM Management System",
			"endpoints": []string{
				"/api/auth/login",
				"/api/auth/register",
				"/api/auth/check-email",
				"/api/v1/vms",
				"/api/v1/vms/{id}",
				"/api/v1/vms/{id}/start",
				"/api/v1/vms/{id}/stop",
				"/api/v1/vms/{id}/metrics",
				"/api/v1/monitoring/metrics",
				"/api/v1/monitoring/vms",
				"/api/v1/monitoring/alerts",
				"/health",
			},
			"compatibility_endpoints": []string{
				"/auth/login",
				"/auth/register",
				"/api/vms",
				"/api/monitoring/metrics",
				"/api/monitoring/vms",
				"/api/monitoring/alerts",
			},
			"unsupported_endpoints": []string{
				"/graphql",
				"/api/security/*",
				"/api/admin/security/*",
				"/api/ws/*",
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

func nullableIntValue(value int) interface{} {
	if value <= 0 {
		return nil
	}
	return value
}
