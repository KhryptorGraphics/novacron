package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/lib/pq"
	"github.com/khryptorgraphics/novacron/backend/api/monitoring"
	"github.com/khryptorgraphics/novacron/backend/api/vm"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
	api_orch "github.com/khryptorgraphics/novacron/backend/api/orchestration"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
	"github.com/sirupsen/logrus"

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

	appLogger.Info("Starting NovaCron API Server...",
		"version", "1.0.0",
		"api_port", cfg.Server.APIPort,
		"ws_port", cfg.Server.WSPort,
	)

	// Initialize database connection
	db, err := initDatabase(cfg)
	if err != nil {
		appLogger.Fatal("Failed to initialize database", "error", err)
	}
	defer db.Close()

	// Initialize authentication manager
	authManager := auth.NewSimpleAuthManager(cfg.Auth.Secret, db)

	// Initialize VM manager with default configuration
	vmConfig := core_vm.VMManagerConfig{
		DefaultDriver: core_vm.VMTypeKVM,
		Drivers: make(map[core_vm.VMType]core_vm.VMDriverConfigManager),
		Scheduler: core_vm.VMSchedulerConfig{
			Type: "round-robin",
			Config: make(map[string]interface{}),
		},
	}

	// Enable KVM driver if available
	vmConfig.Drivers[core_vm.VMTypeKVM] = core_vm.VMDriverConfigManager{
		Enabled: true,
		Config:  make(map[string]interface{}),
	}

	vmManager, err := core_vm.NewVMManager(vmConfig)
	if err != nil {
		appLogger.Warn("Failed to initialize VM manager", "error", err)
		// Create a minimal VM manager for development
		vmManager = nil
	}

	// Initialize KVM manager (optional for development)
	kvmManager, err := hypervisor.NewKVMManager("qemu:///system")
	if err != nil {
		appLogger.Warn("Failed to connect to KVM", "error", err)
		appLogger.Info("Continuing with limited KVM functionality")
		// Create a nil KVM manager but VM manager can still work
		kvmManager = nil
	}

	// Create router
	router := mux.NewRouter()

	// Add CORS middleware
	corsHandler := handlers.CORS(
		handlers.AllowedOrigins([]string{"http://localhost:8092", "http://localhost:3001"}),
		handlers.AllowedMethods([]string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
		handlers.AllowedHeaders([]string{"Content-Type", "Authorization"}),
	)

	// Add authentication middleware
	authMiddleware := middleware.NewAuthMiddleware(authManager)

	// Create API router with middleware
	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.Use(authMiddleware.RequireAuth)

	// Register VM management routes
	if vmManager != nil {
		vmHandlers := vm.NewHandler(vmManager)
		vmHandlers.RegisterRoutes(apiRouter.PathPrefix("/vm").Subrouter())
	} else {
		// Register mock VM handlers for development
		registerMockVMHandlers(apiRouter.PathPrefix("/vm").Subrouter())
	}

	// Initialize orchestration engine (core-only mode)
	orchLogger := appLogger.StandardLogger()
	orchEngine := orchestration.NewDefaultOrchestrationEngine(orchLogger)
	// Compose adapters using the VM manager and engine's placement engine
	if vmManager != nil {
		adapters := &orchestration.OrchestrationAdapters{VMManager: vmManager, PlacementEngine: orchEngine.Placement()}
		// Build default evacuation handler
		evacHandler := orchestration.NewDefaultEvacuationHandler(adapters.ListVMsByNodeAdapter, adapters.SelectTargetAdapter, adapters.MigrateAdapter, orchLogger)
		orchEngine.SetEvacuationHandler(evacHandler)
	}


	// WebSocket orchestration events route (core-compatible)
	wsManager := api_orch.NewWebSocketManager(logrus.New(), orchEngine.EventBus())
	router.HandleFunc("/ws/events/v1", wsManager.HandleWebSocket)


	// Register monitoring routes
	if kvmManager != nil {
		monitoringHandlers := monitoring.NewMonitoringHandlers(kvmManager)
		monitoringHandlers.RegisterRoutes(router)
	} else {
		// Register mock monitoring handlers for development
		registerMockMonitoringHandlers(router)
	}

	// Public routes (no auth required)
	registerPublicRoutes(router, authManager)

	// Health check endpoint with database connectivity check
	router.HandleFunc("/health", healthCheckHandler(cfg, db, kvmManager)).Methods("GET")

	// API info endpoint
	router.HandleFunc("/api/info", apiInfoHandler()).Methods("GET")

	// Create HTTP server with configuration
	server := &http.Server{
		Addr:         ":" + cfg.Server.APIPort,
		Handler:      corsHandler(router),
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	// Start server in a goroutine
	go func() {
		appLogger.Info("API Server starting", "port", cfg.Server.APIPort)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			appLogger.Fatal("Server failed to start", "error", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	appLogger.Info("Shutting down server...")

	// Graceful shutdown with configured timeout
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		appLogger.Fatal("Server forced to shutdown", "error", err)
	}

	// Close connections
	if kvmManager != nil {
		if err := kvmManager.Close(); err != nil {
			appLogger.Error("Error closing KVM manager", "error", err)
		}
	}

	if vmManager != nil {
		if err := vmManager.Close(); err != nil {
			appLogger.Error("Error closing VM manager", "error", err)
		}
	}

	appLogger.Info("Server exited gracefully")
}

// initDatabase initializes the PostgreSQL database connection
func initDatabase(cfg *config.Config) (*sql.DB, error) {
	db, err := sql.Open("postgres", cfg.Database.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(cfg.Database.MaxConnections)
	db.SetMaxIdleConns(cfg.Database.MaxConnections / 2)
	db.SetConnMaxLifetime(cfg.Database.ConnMaxLifetime)
	db.SetConnMaxIdleTime(cfg.Database.ConnMaxIdleTime)

	// Test the connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	// Run database migrations
	if err := runMigrations(db); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to run migrations: %w", err)
	}

	return db, nil
}

// runMigrations executes database migrations
func runMigrations(db *sql.DB) error {
	// Create basic schema if it doesn't exist
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

// registerMockMonitoringHandlers provides mock monitoring endpoints for development
func registerMockMonitoringHandlers(router *mux.Router) {
	appLogger := logger.GlobalLogger
	appLogger.Info("Registering mock monitoring handlers for development...")

	// Mock system metrics
	router.HandleFunc("/api/monitoring/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `{
			"currentCpuUsage": 45.2,
			"currentMemoryUsage": 72.1,
			"currentDiskUsage": 58.3,
			"currentNetworkUsage": 125.7,
			"cpuChangePercentage": 5.2,
			"memoryChangePercentage": -2.1,
			"diskChangePercentage": 1.8,
			"networkChangePercentage": 12.5,
			"cpuTimeseriesData": [40, 42, 45, 48, 52, 49, 46, 44, 47, 45],
			"memoryTimeseriesData": [68, 70, 72, 74, 71, 69, 73, 75, 72, 72],
			"diskTimeseriesData": [55, 56, 58, 60, 59, 57, 58, 59, 58, 58],
			"networkTimeseriesData": [100, 110, 125, 130, 120, 115, 125, 130, 125, 126],
			"timeLabels": ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30"],
			"cpuAnalysis": "CPU usage shows normal workday patterns with peaks during business hours.",
			"memoryAnalysis": "Memory allocation is healthy with sufficient available memory for operations.",
			"memoryInUse": 65.0,
			"memoryAvailable": 20.0,
			"memoryReserved": 10.0,
			"memoryCached": 5.0
		}`
		w.Write([]byte(response))
	}).Methods("GET")

	// Mock VM metrics
	router.HandleFunc("/api/monitoring/vms", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `[
			{
				"vmId": "vm-001",
				"name": "web-server-01",
				"cpuUsage": 78.5,
				"memoryUsage": 65.2,
				"diskUsage": 45.8,
				"networkRx": 1048576,
				"networkTx": 2097152,
				"iops": 150,
				"status": "running"
			},
			{
				"vmId": "vm-002",
				"name": "database-01",
				"cpuUsage": 92.1,
				"memoryUsage": 88.7,
				"diskUsage": 72.3,
				"networkRx": 524288,
				"networkTx": 1048576,
				"iops": 320,
				"status": "running"
			}
		]`
		w.Write([]byte(response))
	}).Methods("GET")

	// Mock alerts
	router.HandleFunc("/api/monitoring/alerts", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `[
			{
				"id": "alert-001",
				"name": "High CPU Usage",
				"description": "VM database-01 CPU usage exceeds 90%",
				"severity": "warning",
				"status": "firing",
				"startTime": "2025-04-11T14:30:00Z",
				"labels": {"vm": "database-01", "metric": "cpu"},
				"value": 92.1,
				"resource": "VM database-01"
			}
		]`
		w.Write([]byte(response))
	}).Methods("GET")

	// WebSocket endpoint mock
	router.HandleFunc("/ws/monitoring", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("WebSocket endpoint - use a WebSocket client to connect"))
	})
}

// registerMockVMHandlers provides mock VM management endpoints for development
func registerMockVMHandlers(router *mux.Router) {
	appLogger := logger.GlobalLogger
	appLogger.Info("Registering mock VM handlers for development...")

	// List VMs
	router.HandleFunc("/vms", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `[
			{
				"id": "vm-001",
				"name": "web-server-01",
				"state": "running",
				"node_id": "node-01",
				"created_at": "2025-01-01T10:00:00Z",
				"updated_at": "2025-01-01T10:00:00Z"
			}
		]`
		w.Write([]byte(response))
	}).Methods("GET")

	// Create VM
	router.HandleFunc("/vms", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		response := `{
			"id": "vm-new",
			"name": "new-vm",
			"state": "creating",
			"node_id": "node-01",
			"created_at": "` + time.Now().Format(time.RFC3339) + `",
			"updated_at": "` + time.Now().Format(time.RFC3339) + `"
		}`
		w.Write([]byte(response))
	}).Methods("POST")

	// Get VM
	router.HandleFunc("/vms/{id}", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		w.Header().Set("Content-Type", "application/json")
		response := `{
			"id": "` + vmID + `",
			"name": "mock-vm-` + vmID + `",
			"state": "running",
			"node_id": "node-01",
			"created_at": "2025-01-01T10:00:00Z",
			"updated_at": "2025-01-01T10:00:00Z"
		}`
		w.Write([]byte(response))
	}).Methods("GET")

	// VM lifecycle operations
	router.HandleFunc("/vms/{id}/start", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status": "started"}`))
	}).Methods("POST")

	router.HandleFunc("/vms/{id}/stop", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status": "stopped"}`))
	}).Methods("POST")

	router.HandleFunc("/vms/{id}/restart", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status": "restarted"}`))
	}).Methods("POST")
}

// registerPublicRoutes registers routes that don't require authentication
func registerPublicRoutes(router *mux.Router, authManager *auth.SimpleAuthManager) {
	// Authentication routes
	router.HandleFunc("/auth/login", func(w http.ResponseWriter, r *http.Request) {
		var loginReq struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}

		if err := json.NewDecoder(r.Body).Decode(&loginReq); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		user, token, err := authManager.Authenticate(loginReq.Username, loginReq.Password)
		if err != nil {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"token": token,
			"user": map[string]interface{}{
				"id":        user.ID,
				"username":  user.Username,
				"email":     user.Email,
				"tenant_id": user.TenantID,
			},
		}

		// Add role information if available
		if len(user.Roles) > 0 {
			response["user"].(map[string]interface{})["role"] = user.Roles[0].Name
		}

		json.NewEncoder(w).Encode(response)
	}).Methods("POST")

	// User registration endpoint
	router.HandleFunc("/auth/register", func(w http.ResponseWriter, r *http.Request) {
		var registerReq struct {
			Username string `json:"username"`
			Email    string `json:"email"`
			Password string `json:"password"`
			TenantID string `json:"tenant_id,omitempty"`
		}

		if err := json.NewDecoder(r.Body).Decode(&registerReq); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if registerReq.TenantID == "" {
			registerReq.TenantID = "default"
		}

		user, err := authManager.CreateUser(registerReq.Username, registerReq.Email, registerReq.Password, "user", registerReq.TenantID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to create user: %v", err), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		response := map[string]interface{}{
			"user": map[string]interface{}{
				"id":        user.ID,
				"username":  user.Username,
				"email":     user.Email,
				"tenant_id": user.TenantID,
			},
		}

		// Add role information if available
		if len(user.Roles) > 0 {
			response["user"].(map[string]interface{})["role"] = user.Roles[0].Name
		}

		json.NewEncoder(w).Encode(response)
	}).Methods("POST")

	// Logout endpoint (can be public since it just returns success)
	router.HandleFunc("/auth/logout", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"message": "Logged out successfully",
		})
	}).Methods("POST")

	// Token validation endpoint
	router.HandleFunc("/auth/validate", func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "Invalid or missing token", http.StatusUnauthorized)
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")

		// Parse and validate the JWT token
		token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, jwt.ErrSignatureInvalid
			}
			return []byte(authManager.GetJWTSecret()), nil
		})

		if err != nil || !token.Valid {
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok {
			http.Error(w, "Invalid token claims", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"valid": true,
			"user": map[string]interface{}{
				"id":        claims["user_id"],
				"username":  claims["username"],
				"email":     claims["email"],
				"role":      claims["role"],
				"tenant_id": claims["tenant_id"],
			},
		})
	}).Methods("GET")
}

// registerMockHandlers registers mock handlers for development when KVM is not available
// This function is kept for backward compatibility but should not be used
func registerMockHandlers(router *mux.Router) {
	log.Println("Registering mock handlers for development...")

	// Mock system metrics
	router.HandleFunc("/api/monitoring/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `{
			"currentCpuUsage": 45.2,
			"currentMemoryUsage": 72.1,
			"currentDiskUsage": 58.3,
			"currentNetworkUsage": 125.7,
			"cpuChangePercentage": 5.2,
			"memoryChangePercentage": -2.1,
			"diskChangePercentage": 1.8,
			"networkChangePercentage": 12.5,
			"cpuTimeseriesData": [40, 42, 45, 48, 52, 49, 46, 44, 47, 45],
			"memoryTimeseriesData": [68, 70, 72, 74, 71, 69, 73, 75, 72, 72],
			"diskTimeseriesData": [55, 56, 58, 60, 59, 57, 58, 59, 58, 58],
			"networkTimeseriesData": [100, 110, 125, 130, 120, 115, 125, 130, 125, 126],
			"timeLabels": ["10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:00", "14:30"],
			"cpuAnalysis": "CPU usage shows normal workday patterns with peaks during business hours.",
			"memoryAnalysis": "Memory allocation is healthy with sufficient available memory for operations.",
			"memoryInUse": 65.0,
			"memoryAvailable": 20.0,
			"memoryReserved": 10.0,
			"memoryCached": 5.0
		}`
		w.Write([]byte(response))
	}).Methods("GET")

	// Mock VM metrics
	router.HandleFunc("/api/monitoring/vms", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `[
			{
				"vmId": "vm-001",
				"name": "web-server-01",
				"cpuUsage": 78.5,
				"memoryUsage": 65.2,
				"diskUsage": 45.8,
				"networkRx": 1048576,
				"networkTx": 2097152,
				"iops": 150,
				"status": "running"
			},
			{
				"vmId": "vm-002",
				"name": "database-01",
				"cpuUsage": 92.1,
				"memoryUsage": 88.7,
				"diskUsage": 72.3,
				"networkRx": 524288,
				"networkTx": 1048576,
				"iops": 320,
				"status": "running"
			},
			{
				"vmId": "vm-003",
				"name": "app-server-01",
				"cpuUsage": 45.3,
				"memoryUsage": 52.1,
				"diskUsage": 38.9,
				"networkRx": 262144,
				"networkTx": 524288,
				"iops": 85,
				"status": "running"
			},
			{
				"vmId": "vm-004",
				"name": "backup-server",
				"cpuUsage": 12.7,
				"memoryUsage": 28.4,
				"diskUsage": 89.2,
				"networkRx": 131072,
				"networkTx": 65536,
				"iops": 45,
				"status": "stopped"
			},
			{
				"vmId": "vm-005",
				"name": "test-environment",
				"cpuUsage": 0.0,
				"memoryUsage": 0.0,
				"diskUsage": 15.6,
				"networkRx": 0,
				"networkTx": 0,
				"iops": 0,
				"status": "error"
			}
		]`
		w.Write([]byte(response))
	}).Methods("GET")

	// Mock alerts
	router.HandleFunc("/api/monitoring/alerts", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `[
			{
				"id": "alert-001",
				"name": "High CPU Usage",
				"description": "VM database-01 CPU usage exceeds 90%",
				"severity": "warning",
				"status": "firing",
				"startTime": "2025-04-11T14:30:00Z",
				"labels": {"vm": "database-01", "metric": "cpu"},
				"value": 92.1,
				"resource": "VM database-01"
			},
			{
				"id": "alert-002",
				"name": "Disk Space Critical",
				"description": "Backup server disk usage exceeds 85%",
				"severity": "critical",
				"status": "firing",
				"startTime": "2025-04-11T13:45:00Z",
				"labels": {"vm": "backup-server", "metric": "disk"},
				"value": 89.2,
				"resource": "VM backup-server"
			},
			{
				"id": "alert-003",
				"name": "VM Unresponsive",
				"description": "Test environment VM is not responding",
				"severity": "error",
				"status": "firing",
				"startTime": "2025-04-11T14:15:00Z",
				"labels": {"vm": "test-environment", "metric": "health"},
				"value": 0,
				"resource": "VM test-environment"
			}
		]`
		w.Write([]byte(response))
	}).Methods("GET")

	// Mock alert acknowledgment
	router.HandleFunc("/api/monitoring/alerts/{id}/acknowledge", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "acknowledged"}`))
	}).Methods("POST")

	// Mock WebSocket endpoint
	router.HandleFunc("/ws/monitoring", func(w http.ResponseWriter, r *http.Request) {
		// For development, just return a simple response
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("WebSocket endpoint - use a WebSocket client to connect"))
	})
}

// healthCheckHandler returns a structured health check response with actual checks
func healthCheckHandler(cfg *config.Config, db *sql.DB, kvmManager *hypervisor.KVMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		checks := make(map[string]string)
		status := "healthy"

		// Check database connectivity
		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if err := db.PingContext(ctx); err != nil {
			checks["database"] = fmt.Sprintf("error: %v", err)
			status = "unhealthy"
		} else {
			checks["database"] = "ok"
		}

		// Check KVM connectivity if available
		if kvmManager != nil {
			if _, err := kvmManager.GetHypervisorMetrics(ctx); err != nil {
				checks["kvm"] = fmt.Sprintf("warning: %v", err)
				// Don't mark as unhealthy for KVM issues in development
			} else {
				checks["kvm"] = "ok"
			}
		} else {
			checks["kvm"] = "not configured"
		}

		// Check storage directory access
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

		w.Header().Set("Content-Type", "application/json")
		if status == "unhealthy" {
			w.WriteHeader(http.StatusServiceUnavailable)
		} else {
			w.WriteHeader(http.StatusOK)
		}
		json.NewEncoder(w).Encode(response)
	}
}

// apiInfoHandler returns API information
func apiInfoHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"name":        "NovaCron API",
			"version":     "1.0.0",
			"description": "Distributed VM Management System",
			"endpoints": []string{
				"/api/monitoring/metrics",
				"/api/monitoring/vms",
				"/api/monitoring/alerts",
				"/api/vm/list",
				"/api/vm/create",
				"/ws/monitoring",
			},
			"documentation": "/api/docs",
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}
}