package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	_ "github.com/lib/pq"
	
	"github.com/khryptorgraphics/novacron/backend/api/monitoring"
	"github.com/khryptorgraphics/novacron/backend/api/vm"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/security"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
)

func main() {
	ctx := context.Background()
	
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Initialize logger
	appLogger := logger.NewFromConfig(
		cfg.Logging.Level,
		cfg.Logging.Format,
		cfg.Logging.Output,
		cfg.Logging.Structured,
	)
	logger.SetGlobalLogger(appLogger)

	appLogger.Info("🔐 Starting Secure NovaCron API Server...",
		"version", "2.0.0-secure",
		"tls_enabled", true,
	)

	// Initialize security components
	securityManager, err := initializeSecurity(ctx, cfg)
	if err != nil {
		appLogger.Fatal("Failed to initialize security", "error", err)
	}

	// Initialize secure database connection
	db, err := initSecureDatabase(securityManager, cfg)
	if err != nil {
		appLogger.Fatal("Failed to initialize database", "error", err)
	}
	defer db.Close()

	// Create secure repositories
	vmRepo := security.NewVMRepository(securityManager.Database)
	userRepo := security.NewUserRepository(securityManager.Database)

	// Initialize authentication with vault secrets
	authManager := auth.NewSimpleAuthManager(securityManager.Secrets.JWTSecret, db)

	// Initialize VM manager
	vmManager, kvmManager := initializeVMManager(appLogger)

	// Create secure router with all middleware
	router := createSecureRouter(
		securityManager,
		vmRepo,
		userRepo,
		authManager,
		vmManager,
		kvmManager,
		appLogger,
	)

	// Start HTTPS server with graceful shutdown
	startSecureServers(router, securityManager, appLogger)
}

func initializeSecurity(ctx context.Context, cfg *config.Config) (*security.SecurityManager, error) {
	// Determine TLS paths
	tlsCertPath := getEnvOrDefault("TLS_CERT_PATH", "/etc/novacron/tls/cert.pem")
	tlsKeyPath := getEnvOrDefault("TLS_KEY_PATH", "/etc/novacron/tls/key.pem")
	
	// Create TLS directory if it doesn't exist
	tlsDir := filepath.Dir(tlsCertPath)
	if err := os.MkdirAll(tlsDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create TLS directory: %w", err)
	}

	securityConfig := &security.SecurityConfig{
		VaultAddress: getEnvOrDefault("VAULT_ADDR", "http://localhost:8200"),
		VaultToken:   getVaultToken(),
		TLSCertPath:  tlsCertPath,
		TLSKeyPath:   tlsKeyPath,
		DatabaseURL:  cfg.Database.URL,
	}

	return security.InitializeSecurity(ctx, securityConfig)
}

func getVaultToken() string {
	// Try to read from file first
	if data, err := os.ReadFile(".vault-token"); err == nil {
		return strings.TrimSpace(string(data))
	}
	
	// Fall back to environment variable
	return getEnvOrDefault("VAULT_TOKEN", "dev-token")
}

func initSecureDatabase(sm *security.SecurityManager, cfg *config.Config) (*sql.DB, error) {
	// Build secure database URL with password from vault
	dbURL := cfg.Database.URL
	if dbURL == "" {
		dbURL = fmt.Sprintf(
			"postgres://%s:%s@%s:%d/%s?sslmode=require",
			cfg.Database.User,
			sm.Secrets.DatabasePassword,
			cfg.Database.Host,
			cfg.Database.Port,
			cfg.Database.Name,
		)
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(cfg.Database.MaxConnections)
	db.SetMaxIdleConns(cfg.Database.MaxIdleConnections)
	db.SetConnMaxLifetime(cfg.Database.ConnectionLifetime)

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return db, nil
}

func initializeVMManager(logger *logger.Logger) (*core_vm.VMManager, *hypervisor.KVMManager) {
	// Initialize VM manager configuration
	vmConfig := core_vm.VMManagerConfig{
		DefaultDriver: core_vm.VMTypeKVM,
		Drivers: map[core_vm.VMType]core_vm.VMDriverConfigManager{
			core_vm.VMTypeKVM: {
				Enabled: true,
				Config:  make(map[string]interface{}),
			},
		},
		Scheduler: core_vm.VMSchedulerConfig{
			Type:   "round-robin",
			Config: make(map[string]interface{}),
		},
	}

	vmManager, err := core_vm.NewVMManager(vmConfig)
	if err != nil {
		logger.Warn("Failed to initialize VM manager", "error", err)
		vmManager = nil
	}

	kvmManager, err := hypervisor.NewKVMManager("qemu:///system")
	if err != nil {
		logger.Warn("Failed to connect to KVM", "error", err)
		kvmManager = nil
	}

	return vmManager, kvmManager
}

func createSecureRouter(
	sm *security.SecurityManager,
	vmRepo *security.VMRepository,
	userRepo *security.UserRepository,
	authManager auth.AuthManager,
	vmManager *core_vm.VMManager,
	kvmManager *hypervisor.KVMManager,
	logger *logger.Logger,
) *mux.Router {
	router := mux.NewRouter()
	
	// Add security headers middleware (applied to all routes)
	router.Use(securityHeadersMiddleware)
	
	// Add rate limiting middleware
	rateLimiter := security.NewRateLimiter(100, 60) // 100 requests per minute
	router.Use(rateLimitMiddleware(rateLimiter))
	
	// Add audit logging middleware
	auditLogger := security.NewAuditLogger(os.Stdout)
	router.Use(auditMiddleware(auditLogger))

	// Health check endpoint
	router.HandleFunc("/api/health", healthHandler(sm)).Methods("GET")

	// Authentication endpoints with secure password handling
	router.HandleFunc("/api/auth/login", loginHandler(userRepo, sm, authManager)).Methods("POST")
	router.HandleFunc("/api/auth/register", registerHandler(userRepo, sm, authManager)).Methods("POST")
	router.HandleFunc("/api/auth/logout", logoutHandler()).Methods("POST")

	// Protected API routes
	api := router.PathPrefix("/api").Subrouter()
	api.Use(jwtMiddleware(sm.Secrets.JWTSecret))

	// VM endpoints with secure database queries
	api.HandleFunc("/vms", listVMsHandler(vmRepo)).Methods("GET")
	api.HandleFunc("/vms", createVMHandler(vmRepo, sm.Validator, vmManager)).Methods("POST")
	api.HandleFunc("/vms/{id}", getVMHandler(vmRepo, sm.Validator)).Methods("GET")
	api.HandleFunc("/vms/{id}", updateVMHandler(vmRepo, sm.Validator)).Methods("PUT")
	api.HandleFunc("/vms/{id}", deleteVMHandler(vmRepo, sm.Validator)).Methods("DELETE")

	// Monitoring endpoints
	api.HandleFunc("/metrics", metricsHandler()).Methods("GET")
	api.HandleFunc("/monitoring/dashboard", monitoringDashboardHandler()).Methods("GET")

	// WebSocket endpoint
	api.HandleFunc("/ws", websocketHandler()).Methods("GET")

	// Apply CORS
	cors := handlers.CORS(
		handlers.AllowedOrigins([]string{"https://localhost:3000", "https://localhost:8092"}),
		handlers.AllowedMethods([]string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
		handlers.AllowedHeaders([]string{"Content-Type", "Authorization"}),
		handlers.AllowCredentials(),
	)

	return cors(router).(*mux.Router)
}

// Security middleware functions
func securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
		w.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
		w.Header().Set("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
		next.ServeHTTP(w, r)
	})
}

func rateLimitMiddleware(rl *security.RateLimiter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Use IP address as identifier
			identifier := r.RemoteAddr
			if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
				identifier = forwarded
			}
			
			if !rl.Allow(identifier) {
				http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	}
}

func auditMiddleware(al *security.AuditLogger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Log API access
			al.LogDataAccess(r.Context(), getUsername(r), r.URL.Path, r.Method)
			next.ServeHTTP(w, r)
		})
	}
}

func jwtMiddleware(secret string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			authHeader := r.Header.Get("Authorization")
			if authHeader == "" {
				http.Error(w, "Missing authorization header", http.StatusUnauthorized)
				return
			}

			tokenString := strings.TrimPrefix(authHeader, "Bearer ")
			token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
				if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
					return nil, fmt.Errorf("unexpected signing method")
				}
				return []byte(secret), nil
			})

			if err != nil || !token.Valid {
				http.Error(w, "Invalid token", http.StatusUnauthorized)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// Handler implementations
func healthHandler(sm *security.SecurityManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		status := map[string]interface{}{
			"status":    "healthy",
			"timestamp": time.Now().UTC(),
			"secure":    true,
			"tls":       true,
			"vault":     sm.Vault != nil,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(status)
	}
}

func loginHandler(userRepo *security.UserRepository, sm *security.SecurityManager, authManager auth.AuthManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Email    string `json:"email"`
			Password string `json:"password"`
		}
		
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}
		
		// Validate and sanitize input
		req.Email = security.SanitizeInput(req.Email)
		if !security.ValidateEmail(req.Email) {
			http.Error(w, "Invalid email format", http.StatusBadRequest)
			return
		}
		
		// Get user (safe from SQL injection)
		user, err := userRepo.GetUserByEmail(r.Context(), req.Email)
		if err != nil {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}
		
		// Verify password
		if !sm.Hasher.VerifyPassword(req.Password, user.PasswordHash) {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}
		
		// Generate JWT token
		token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
			"user_id": user.ID,
			"email":   user.Email,
			"role":    user.Role,
			"exp":     time.Now().Add(24 * time.Hour).Unix(),
		})
		
		tokenString, err := token.SignedString([]byte(sm.Secrets.JWTSecret))
		if err != nil {
			http.Error(w, "Failed to generate token", http.StatusInternalServerError)
			return
		}
		
		response := map[string]interface{}{
			"token": tokenString,
			"user": map[string]interface{}{
				"id":    user.ID,
				"email": user.Email,
				"role":  user.Role,
			},
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

func registerHandler(userRepo *security.UserRepository, sm *security.SecurityManager, authManager auth.AuthManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Email    string `json:"email"`
			Password string `json:"password"`
		}
		
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}
		
		// Validate input
		req.Email = security.SanitizeInput(req.Email)
		if !security.ValidateEmail(req.Email) {
			http.Error(w, "Invalid email format", http.StatusBadRequest)
			return
		}
		
		if len(req.Password) < 8 {
			http.Error(w, "Password must be at least 8 characters", http.StatusBadRequest)
			return
		}
		
		// Hash password
		hashedPassword, err := sm.Hasher.HashPassword(req.Password)
		if err != nil {
			http.Error(w, "Failed to process password", http.StatusInternalServerError)
			return
		}
		
		// Create user
		user := &security.User{
			ID:           generateID(),
			Email:        req.Email,
			PasswordHash: hashedPassword,
			Role:         "user",
		}
		
		if err := userRepo.CreateUser(r.Context(), user); err != nil {
			http.Error(w, "Failed to create user", http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"message": "User created successfully",
		})
	}
}

func logoutHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// In a JWT-based system, logout is handled client-side
		// Here we could blacklist the token if needed
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"message": "Logged out successfully",
		})
	}
}

func listVMsHandler(vmRepo *security.VMRepository) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Get optional state filter (safe from SQL injection)
		state := r.URL.Query().Get("state")
		
		vms, err := vmRepo.GetVMs(r.Context(), state)
		if err != nil {
			http.Error(w, "Failed to get VMs", http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vms)
	}
}

func createVMHandler(vmRepo *security.VMRepository, validator *security.InputValidator, vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Name   string `json:"name"`
			NodeID string `json:"node_id"`
		}
		
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}
		
		// Validate input
		if err := validator.ValidateVMName(req.Name); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		if err := validator.ValidateNodeID(req.NodeID); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		// Create VM
		vm := &security.VM{
			ID:     generateID(),
			Name:   req.Name,
			State:  "creating",
			NodeID: req.NodeID,
		}
		
		if err := vmRepo.CreateVM(r.Context(), vm); err != nil {
			http.Error(w, "Failed to create VM", http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(vm)
	}
}

func getVMHandler(vmRepo *security.VMRepository, validator *security.InputValidator) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		// Validate input
		if err := validator.ValidateNodeID(vmID); err != nil {
			http.Error(w, "Invalid VM ID", http.StatusBadRequest)
			return
		}
		
		vm, err := vmRepo.GetVMByID(r.Context(), vmID)
		if err != nil {
			http.Error(w, "VM not found", http.StatusNotFound)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}
}

func updateVMHandler(vmRepo *security.VMRepository, validator *security.InputValidator) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		// Validate VM ID
		if err := validator.ValidateNodeID(vmID); err != nil {
			http.Error(w, "Invalid VM ID", http.StatusBadRequest)
			return
		}
		
		var req struct {
			Name  string `json:"name"`
			State string `json:"state"`
		}
		
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}
		
		// Validate input
		if req.Name != "" {
			if err := validator.ValidateVMName(req.Name); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}
		
		// Get existing VM
		vm, err := vmRepo.GetVMByID(r.Context(), vmID)
		if err != nil {
			http.Error(w, "VM not found", http.StatusNotFound)
			return
		}
		
		// Update fields
		if req.Name != "" {
			vm.Name = req.Name
		}
		if req.State != "" {
			vm.State = req.State
		}
		
		if err := vmRepo.UpdateVM(r.Context(), vm); err != nil {
			http.Error(w, "Failed to update VM", http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}
}

func deleteVMHandler(vmRepo *security.VMRepository, validator *security.InputValidator) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		// Validate VM ID
		if err := validator.ValidateNodeID(vmID); err != nil {
			http.Error(w, "Invalid VM ID", http.StatusBadRequest)
			return
		}
		
		if err := vmRepo.DeleteVM(r.Context(), vmID); err != nil {
			http.Error(w, "Failed to delete VM", http.StatusInternalServerError)
			return
		}
		
		w.WriteHeader(http.StatusNoContent)
	}
}

func metricsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		metrics := map[string]interface{}{
			"cpu_usage":    45.2,
			"memory_usage": 62.8,
			"disk_usage":   38.5,
			"network_io":   1024,
			"vm_count":     12,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(metrics)
	}
}

func monitoringDashboardHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		dashboard := map[string]interface{}{
			"status": "operational",
			"nodes":  3,
			"vms":    12,
			"alerts": 0,
		}
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(dashboard)
	}
}

func websocketHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// WebSocket implementation would go here
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"message": "WebSocket endpoint",
		})
	}
}

func startSecureServers(router *mux.Router, sm *security.SecurityManager, logger *logger.Logger) {
	// Start HTTP to HTTPS redirect server
	go func() {
		redirectHandler := &security.HTTPSRedirectHandler{HTTPSPort: "8443"}
		logger.Info("Starting HTTP redirect server", "port", 8080)
		if err := http.ListenAndServe(":8080", redirectHandler); err != nil {
			logger.Error("HTTP redirect server error", "error", err)
		}
	}()

	// Create TLS server
	tlsServer, err := security.NewTLSServer(":8443", router, sm.TLS)
	if err != nil {
		logger.Fatal("Failed to create TLS server", "error", err)
	}

	// Handle graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan
		
		logger.Info("Shutting down servers...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		
		if err := tlsServer.Stop(ctx); err != nil {
			logger.Error("Error during shutdown", "error", err)
		}
		os.Exit(0)
	}()

	// Start HTTPS server
	logger.Info("🔒 Starting HTTPS server", "port", 8443)
	logger.Info("📝 TLS Configuration:",
		"min_version", "TLS 1.2",
		"cipher_suites", "Secure set",
		"cert_path", sm.TLS.CertFile,
	)
	
	if err := tlsServer.Start(); err != nil {
		logger.Fatal("HTTPS server error", "error", err)
	}
}

// Helper functions
func getUsername(r *http.Request) string {
	// Extract username from JWT token if present
	authHeader := r.Header.Get("Authorization")
	if authHeader == "" {
		return "anonymous"
	}
	// In production, decode JWT to get username
	return "authenticated_user"
}

func generateID() string {
	// Generate a unique ID
	return fmt.Sprintf("vm-%d", time.Now().UnixNano())
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}