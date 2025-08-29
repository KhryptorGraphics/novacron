package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/lib/pq"
	"github.com/khryptorgraphics/novacron/backend/api/admin"
	"github.com/khryptorgraphics/novacron/backend/api/monitoring"
	"github.com/khryptorgraphics/novacron/backend/api/vm"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		logger.Fatal("Failed to load configuration", "error", err)
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		logger.Fatal("Invalid configuration", "error", err)
	}

	// Initialize logger
	appLogger := logger.NewFromConfig(
		cfg.Logging.Level,
		cfg.Logging.Format,
		cfg.Logging.Output,
		cfg.Logging.Structured,
	)
	logger.SetGlobalLogger(appLogger)

	appLogger.Info("Starting Enhanced NovaCron API Server...",
		"version", "2.0.0",
		"api_port", cfg.Server.APIPort,
		"ws_port", cfg.Server.WSPort,
	)

	// Initialize database connection
	db, err := initDatabase(cfg)
	if err != nil {
		appLogger.Fatal("Failed to initialize database", "error", err)
	}
	defer db.Close()

	// Run database migrations
	if err := runMigrations(db); err != nil {
		appLogger.Fatal("Failed to run database migrations", "error", err)
	}

	// Initialize authentication manager
	authManager := auth.NewSimpleAuthManager(cfg.Auth.Secret, db)

	// Initialize VM manager
	vmConfig := core_vm.VMManagerConfig{
		DefaultDriver: core_vm.VMTypeKVM,
		Drivers: make(map[core_vm.VMType]core_vm.VMDriverConfigManager),
		Scheduler: core_vm.VMSchedulerConfig{
			Type:   "round-robin",
			Config: make(map[string]interface{}),
		},
	}

	// Enable supported drivers
	vmConfig.Drivers[core_vm.VMTypeKVM] = core_vm.VMDriverConfigManager{
		Enabled: true,
		Config:  make(map[string]interface{}),
	}

	vmConfig.Drivers[core_vm.VMTypeContainer] = core_vm.VMDriverConfigManager{
		Enabled: true,
		Config:  make(map[string]interface{}),
	}

	vmManager, err := core_vm.NewVMManager(vmConfig)
	if err != nil {
		appLogger.Warn("Failed to initialize VM manager", "error", err)
		vmManager = nil
	}

	// Initialize KVM manager (optional for development)
	kvmManager, err := hypervisor.NewKVMManager("qemu:///system")
	if err != nil {
		appLogger.Warn("Failed to connect to KVM", "error", err)
		kvmManager = nil
	}

	// Initialize API handlers
	userHandlers := admin.NewUserManagementHandlers(db)
	databaseHandlers := admin.NewDatabaseHandlers(db)
	securityHandlers := admin.NewSecurityHandlers(db)
	configHandlers := admin.NewConfigHandlers("/etc/novacron")
	
	var monitoringHandlers *monitoring.MonitoringHandlers
	if kvmManager != nil {
		monitoringHandlers = monitoring.NewMonitoringHandlers(kvmManager)
	} else {
		monitoringHandlers = monitoring.NewMonitoringHandlersWithVMManager(vmManager)
	}

	// Create router
	router := mux.NewRouter()

	// Add CORS middleware
	corsHandler := handlers.CORS(
		handlers.AllowedOrigins([]string{"*"}),
		handlers.AllowedMethods([]string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
		handlers.AllowedHeaders([]string{"*"}),
		handlers.AllowCredentials(),
	)

	// Add logging middleware
	router.Use(handlers.LoggingHandler(os.Stdout, http.DefaultServeMux))

	// Health check endpoint
	router.HandleFunc("/health", healthCheckHandler(cfg, db, kvmManager)).Methods("GET")
	router.HandleFunc("/api/info", apiInfoHandler()).Methods("GET")

	// Public routes (no authentication required)
	router.HandleFunc("/api/auth/login", authLoginHandler(authManager)).Methods("POST", "OPTIONS")
	router.HandleFunc("/api/auth/register", authRegisterHandler(authManager)).Methods("POST", "OPTIONS")
	
	// Create API router with authentication middleware
	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.Use(middleware.AuthMiddleware(authManager))

	// VM Management APIs
	apiRouter.HandleFunc("/vm/vms", getVMsHandler(vmManager)).Methods("GET")
	apiRouter.HandleFunc("/vm/vms", createVMHandler(vmManager)).Methods("POST")
	apiRouter.HandleFunc("/vm/vms/{id}", getVMHandler(vmManager)).Methods("GET")
	apiRouter.HandleFunc("/vm/vms/{id}", updateVMHandler(vmManager)).Methods("PUT")
	apiRouter.HandleFunc("/vm/vms/{id}", deleteVMHandler(vmManager)).Methods("DELETE")
	apiRouter.HandleFunc("/vm/vms/{id}/start", startVMHandler(vmManager)).Methods("POST")
	apiRouter.HandleFunc("/vm/vms/{id}/stop", stopVMHandler(vmManager)).Methods("POST")
	apiRouter.HandleFunc("/vm/vms/{id}/restart", restartVMHandler(vmManager)).Methods("POST")
	apiRouter.HandleFunc("/vm/vms/{id}/migrate", migrateVMHandler(vmManager)).Methods("POST")

	// Monitoring APIs
	if monitoringHandlers != nil {
		apiRouter.HandleFunc("/monitoring/metrics", monitoringHandlers.GetSystemMetrics).Methods("GET")
		apiRouter.HandleFunc("/monitoring/vms", monitoringHandlers.GetVMMetrics).Methods("GET")
		apiRouter.HandleFunc("/monitoring/alerts", monitoringHandlers.GetAlerts).Methods("GET")
		apiRouter.HandleFunc("/monitoring/alerts/{id}/acknowledge", monitoringHandlers.AcknowledgeAlert).Methods("POST")
		
		// WebSocket endpoint for real-time monitoring
		router.HandleFunc("/ws/monitoring", monitoringHandlers.HandleWebSocket).Methods("GET")
	}

	// Admin APIs (require admin role)
	adminRouter := apiRouter.PathPrefix("/admin").Subrouter()
	adminRouter.Use(middleware.AdminOnlyMiddleware)

	// User Management APIs
	adminRouter.HandleFunc("/users", userHandlers.ListUsers).Methods("GET")
	adminRouter.HandleFunc("/users", userHandlers.CreateUser).Methods("POST")
	adminRouter.HandleFunc("/users/{id}", userHandlers.UpdateUser).Methods("PUT")
	adminRouter.HandleFunc("/users/{id}", userHandlers.DeleteUser).Methods("DELETE")
	adminRouter.HandleFunc("/users/{id}/roles", userHandlers.AssignRoles).Methods("POST")

	// Database Admin APIs
	adminRouter.HandleFunc("/database/tables", databaseHandlers.ListTables).Methods("GET")
	adminRouter.HandleFunc("/database/tables/{table}", databaseHandlers.GetTableDetails).Methods("GET")
	adminRouter.HandleFunc("/database/query", databaseHandlers.ExecuteQuery).Methods("POST")
	adminRouter.HandleFunc("/database/execute", databaseHandlers.ExecuteStatement).Methods("POST")

	// Security APIs
	adminRouter.HandleFunc("/security/metrics", securityHandlers.GetSecurityMetrics).Methods("GET")
	adminRouter.HandleFunc("/security/alerts", securityHandlers.GetSecurityAlerts).Methods("GET")
	adminRouter.HandleFunc("/security/audit", securityHandlers.GetAuditLogs).Methods("GET")
	adminRouter.HandleFunc("/security/policies", securityHandlers.GetSecurityPolicies).Methods("GET")
	adminRouter.HandleFunc("/security/policies/{id}", securityHandlers.UpdateSecurityPolicy).Methods("PUT")

	// System Configuration APIs
	adminRouter.HandleFunc("/config", configHandlers.GetConfig).Methods("GET")
	adminRouter.HandleFunc("/config", configHandlers.UpdateConfig).Methods("PUT")
	adminRouter.HandleFunc("/config/validate", configHandlers.ValidateConfig).Methods("POST")
	adminRouter.HandleFunc("/config/backup", configHandlers.CreateBackup).Methods("POST")
	adminRouter.HandleFunc("/config/backups", configHandlers.ListBackups).Methods("GET")
	adminRouter.HandleFunc("/config/restore/{id}", configHandlers.RestoreBackup).Methods("POST")

	// Apply CORS to all routes
	handler := corsHandler(router)

	// Create HTTP server
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Server.APIPort),
		Handler:      handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in a goroutine
	go func() {
		appLogger.Info("API server starting", "port", cfg.Server.APIPort)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			appLogger.Fatal("Failed to start server", "error", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	appLogger.Info("Shutting down server...")

	// Give outstanding requests 30 seconds to complete
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		appLogger.Fatal("Server forced to shutdown", "error", err)
	}

	appLogger.Info("Server exited")
}

func initDatabase(cfg *config.Config) (*sql.DB, error) {
	dbURL := cfg.Database.URL
	if dbURL == "" {
		dbURL = fmt.Sprintf("postgres://%s:%s@%s:%d/%s?sslmode=%s",
			cfg.Database.Username,
			cfg.Database.Password,
			cfg.Database.Host,
			cfg.Database.Port,
			cfg.Database.Name,
			cfg.Database.SSLMode,
		)
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(cfg.Database.MaxOpenConnections)
	db.SetMaxIdleConns(cfg.Database.MaxIdleConnections)
	db.SetConnMaxLifetime(time.Duration(cfg.Database.ConnMaxLifetimeMinutes) * time.Minute)

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	return db, nil
}

func runMigrations(db *sql.DB) error {
	migrations := []string{
		// Users table (if not exists)
		`CREATE TABLE IF NOT EXISTS users (
			id SERIAL PRIMARY KEY,
			username VARCHAR(50) UNIQUE NOT NULL,
			email VARCHAR(255) UNIQUE NOT NULL,
			password_hash VARCHAR(255) NOT NULL,
			role VARCHAR(20) DEFAULT 'user',
			active BOOLEAN DEFAULT true,
			created_at TIMESTAMP DEFAULT NOW(),
			updated_at TIMESTAMP DEFAULT NOW()
		)`,

		// Security alerts table
		`CREATE TABLE IF NOT EXISTS security_alerts (
			id SERIAL PRIMARY KEY,
			type VARCHAR(50) NOT NULL,
			severity VARCHAR(20) NOT NULL,
			title VARCHAR(255) NOT NULL,
			description TEXT,
			source VARCHAR(100),
			ip INET,
			user_agent TEXT,
			status VARCHAR(20) DEFAULT 'open',
			created_at TIMESTAMP DEFAULT NOW(),
			updated_at TIMESTAMP DEFAULT NOW()
		)`,

		// Audit logs table
		`CREATE TABLE IF NOT EXISTS audit_logs (
			id SERIAL PRIMARY KEY,
			user_id INTEGER REFERENCES users(id),
			action VARCHAR(50) NOT NULL,
			resource VARCHAR(100),
			details JSONB,
			ip INET,
			user_agent TEXT,
			success BOOLEAN DEFAULT true,
			created_at TIMESTAMP DEFAULT NOW()
		)`,

		// VM instances table
		`CREATE TABLE IF NOT EXISTS vm_instances (
			id SERIAL PRIMARY KEY,
			name VARCHAR(100) NOT NULL,
			driver VARCHAR(20) NOT NULL,
			cpu_cores INTEGER DEFAULT 1,
			memory_mb INTEGER DEFAULT 1024,
			disk_gb INTEGER DEFAULT 20,
			status VARCHAR(20) DEFAULT 'stopped',
			node_id VARCHAR(50),
			config JSONB,
			created_at TIMESTAMP DEFAULT NOW(),
			updated_at TIMESTAMP DEFAULT NOW()
		)`,

		// Create indexes for performance
		`CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_security_alerts_severity ON security_alerts(severity)`,
		`CREATE INDEX IF NOT EXISTS idx_security_alerts_created_at ON security_alerts(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_instances_status ON vm_instances(status)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_instances_node_id ON vm_instances(node_id)`,
	}

	for _, migration := range migrations {
		if _, err := db.Exec(migration); err != nil {
			return fmt.Errorf("failed to run migration: %w", err)
		}
	}

	logger.Info("Database migrations completed successfully")
	return nil
}

func healthCheckHandler(cfg *config.Config, db *sql.DB, kvmManager *hypervisor.KVMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		health := map[string]interface{}{
			"status":    "ok",
			"timestamp": time.Now(),
			"version":   "2.0.0",
			"services":  map[string]string{},
		}

		services := health["services"].(map[string]string)

		// Check database
		if err := db.Ping(); err != nil {
			services["database"] = "error: " + err.Error()
			health["status"] = "degraded"
		} else {
			services["database"] = "ok"
		}

		// Check KVM (optional)
		if kvmManager != nil {
			services["kvm"] = "ok"
		} else {
			services["kvm"] = "unavailable"
		}

		w.Header().Set("Content-Type", "application/json")
		if health["status"] == "ok" {
			w.WriteHeader(http.StatusOK)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
		}

		json.NewEncoder(w).Encode(health)
	}
}

func apiInfoHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		info := map[string]interface{}{
			"name":        "NovaCron Enhanced API",
			"version":     "2.0.0",
			"description": "Enhanced NovaCron Distributed VM Management System",
			"features": []string{
				"VM Management",
				"User Administration", 
				"Security Dashboard",
				"Database Administration",
				"System Configuration",
				"Real-time Monitoring",
				"Audit Logging",
			},
			"endpoints": map[string][]string{
				"vm":        {"/api/vm/vms", "/api/vm/vms/{id}"},
				"admin":     {"/api/admin/users", "/api/admin/database", "/api/admin/security", "/api/admin/config"},
				"monitor":   {"/api/monitoring/metrics", "/api/monitoring/vms", "/api/monitoring/alerts"},
				"auth":      {"/api/auth/login", "/api/auth/register"},
				"websocket": {"/ws/monitoring"},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(info)
	}
}

// Placeholder handlers for VM operations
func getVMsHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Mock VM data for now
		vms := []map[string]interface{}{
			{
				"id":       1,
				"name":     "web-server-01",
				"driver":   "kvm",
				"status":   "running",
				"cpu":      2,
				"memory":   2048,
				"disk":     20,
				"node":     "node1",
			},
			{
				"id":       2,
				"name":     "database-01",
				"driver":   "container",
				"status":   "stopped",
				"cpu":      4,
				"memory":   4096,
				"disk":     50,
				"node":     "node2",
			},
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"vms":   vms,
			"count": len(vms),
		})
	}
}

// Additional placeholder handlers would go here...
func createVMHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "VM creation not yet implemented",
			"status":  "pending",
		})
	}
}

func getVMHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      vmID,
			"message": "VM details not yet implemented",
		})
	}
}

func updateVMHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "VM update not yet implemented",
		})
	}
}

func deleteVMHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}
}

func startVMHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "VM start not yet implemented",
			"status":  "starting",
		})
	}
}

func stopVMHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "VM stop not yet implemented",
			"status":  "stopping",
		})
	}
}

func restartVMHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "VM restart not yet implemented",
			"status":  "restarting",
		})
	}
}

func migrateVMHandler(vmManager *core_vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "VM migration not yet implemented",
			"status":  "migrating",
		})
	}
}

func authLoginHandler(authManager *auth.SimpleAuthManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		var req struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}
		
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}
		
		// Mock authentication for now
		token := "mock-jwt-token"
		
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"token": token,
			"user": map[string]interface{}{
				"id":       1,
				"username": req.Username,
				"role":     "admin",
			},
		})
	}
}

func authRegisterHandler(authManager *auth.SimpleAuthManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "Registration not yet implemented",
		})
	}
}

