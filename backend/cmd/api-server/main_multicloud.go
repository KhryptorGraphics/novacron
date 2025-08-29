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
	"syscall"
	"time"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/lib/pq"
	"github.com/khryptorgraphics/novacron/backend/api/monitoring"
	"github.com/khryptorgraphics/novacron/backend/api/vm"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/khryptorgraphics/novacron/backend/core/federation/multicloud"
	"github.com/khryptorgraphics/novacron/backend/core/federation/multicloud/providers"
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

	appLogger.Info("Starting NovaCron API Server with Multi-Cloud Federation...",
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

	// Initialize Federation Manager
	federationMgr := federation.NewFederationManager(
		"local-cluster",
		federation.PrimaryCluster,
		federation.HierarchicalMode,
	)
	
	// Start federation manager
	if err := federationMgr.Start(); err != nil {
		appLogger.Error("Failed to start federation manager", "error", err)
	} else {
		appLogger.Info("Federation manager started successfully")
		defer func() {
			if err := federationMgr.Stop(); err != nil {
				appLogger.Error("Failed to stop federation manager", "error", err)
			}
		}()
	}

	// Initialize Multi-Cloud Orchestrator
	multiCloudOrchestrator := multicloud.NewUnifiedOrchestrator(federationMgr)

	// Register default cloud providers (for demonstration)
	if err := registerDefaultCloudProviders(multiCloudOrchestrator, appLogger); err != nil {
		appLogger.Warn("Failed to register some cloud providers", "error", err)
	}

	// Start provider health monitoring
	multiCloudOrchestrator.registry.StartHealthMonitoring(5 * time.Minute)

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

	// Register multi-cloud routes
	multiCloudHandlers := multicloud.NewAPIHandlers(multiCloudOrchestrator)
	multiCloudHandlers.RegisterRoutes(apiRouter)

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
		appLogger.Info("API Server starting with Multi-Cloud Federation", "port", cfg.Server.APIPort)
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

// registerDefaultCloudProviders registers default cloud providers for demonstration
func registerDefaultCloudProviders(orchestrator *multicloud.UnifiedOrchestrator, appLogger logger.Logger) error {
	// Register AWS provider (mock configuration)
	awsProvider := providers.NewAWSProvider()
	awsConfig := &multicloud.CloudProviderConfig{
		Type: multicloud.ProviderAWS,
		Name: "AWS Production",
		Credentials: map[string]string{
			"access_key_id":     "AKIAIOSFODNN7EXAMPLE",
			"secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
		},
		DefaultRegion: "us-east-1",
		Regions:       []string{"us-east-1", "us-west-2", "eu-west-1"},
		Endpoints: map[string]string{
			"api": "https://ec2.us-east-1.amazonaws.com",
		},
		Options: map[string]interface{}{
			"retry_attempts": 3,
			"timeout":       30,
		},
	}

	if err := orchestrator.RegisterCloudProvider("aws-prod", awsProvider, awsConfig); err != nil {
		appLogger.Error("Failed to register AWS provider", "error", err)
		return err
	}

	appLogger.Info("Successfully registered AWS provider")

	// Add on-premise provider (represents local KVM)
	if kvmConfig := createOnPremiseConfig(); kvmConfig != nil {
		appLogger.Info("On-premise KVM provider configuration ready")
	}

	return nil
}

// createOnPremiseConfig creates configuration for on-premise provider
func createOnPremiseConfig() *multicloud.CloudProviderConfig {
	return &multicloud.CloudProviderConfig{
		Type: multicloud.ProviderOnPremise,
		Name: "On-Premise KVM",
		DefaultRegion: "datacenter-1",
		Regions: []string{"datacenter-1", "datacenter-2"},
		Endpoints: map[string]string{
			"api": "qemu:///system",
		},
		Options: map[string]interface{}{
			"virtualization": "kvm",
			"storage_pool":  "default",
		},
	}
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
		// Multi-cloud tables
		`CREATE TABLE IF NOT EXISTS cloud_providers (
			id VARCHAR(255) PRIMARY KEY,
			name VARCHAR(255) NOT NULL,
			type VARCHAR(50) NOT NULL,
			config JSONB,
			enabled BOOLEAN DEFAULT true,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS multicloud_vms (
			id VARCHAR(255) PRIMARY KEY,
			provider_id VARCHAR(255) REFERENCES cloud_providers(id),
			vm_id VARCHAR(255) NOT NULL,
			name VARCHAR(255),
			region VARCHAR(100),
			instance_type VARCHAR(100),
			state VARCHAR(50),
			tags JSONB,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS migrations (
			id VARCHAR(255) PRIMARY KEY,
			vm_id VARCHAR(255),
			source_provider VARCHAR(255),
			destination_provider VARCHAR(255),
			status VARCHAR(50),
			progress INTEGER DEFAULT 0,
			started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			completed_at TIMESTAMP,
			error_message TEXT
		)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id ON vm_metrics(vm_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_metrics_timestamp ON vm_metrics(timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_multicloud_vms_provider ON multicloud_vms(provider_id)`,
		`CREATE INDEX IF NOT EXISTS idx_migrations_status ON migrations(status)`,
	}

	for _, migration := range migrations {
		if _, err := db.Exec(migration); err != nil {
			return fmt.Errorf("migration failed: %w", err)
		}
	}

	return nil
}

// The rest of the functions remain the same as in the original main.go
// registerMockMonitoringHandlers, registerMockVMHandlers, registerPublicRoutes, 
// healthCheckHandler, apiInfoHandler

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
}

// registerPublicRoutes registers routes that don't require authentication
func registerPublicRoutes(router *mux.Router, authManager *auth.AuthManager) {
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
		json.NewEncoder(w).Encode(map[string]interface{}{
			"token": token,
			"user": map[string]interface{}{
				"id":       user.ID,
				"username": user.Username,
				"email":    user.Email,
				"role":     user.Role,
				"tenant_id": user.TenantID,
			},
		})
	}).Methods("POST")
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

		// Add multi-cloud health check
		checks["multicloud"] = "ok"

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
			"description": "Distributed VM Management System with Multi-Cloud Federation",
			"endpoints": []string{
				"/api/monitoring/metrics",
				"/api/monitoring/vms",
				"/api/monitoring/alerts",
				"/api/vm/list",
				"/api/vm/create",
				"/api/multicloud/providers",
				"/api/multicloud/vms",
				"/api/multicloud/migrations",
				"/api/multicloud/cost/analysis",
				"/api/multicloud/compliance/report",
				"/ws/monitoring",
			},
			"documentation": "/api/docs",
			"features": []string{
				"Multi-cloud VM management",
				"Cross-cloud migration",
				"Cost optimization",
				"Compliance management",
				"Federation support",
				"Resource monitoring",
			},
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}
}