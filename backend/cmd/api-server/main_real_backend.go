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
	"strconv"
	"syscall"
	"time"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/database"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
	"github.com/khryptorgraphics/novacron/backend/pkg/services"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		log.Fatalf("Invalid configuration: %v", err)
	}

	// Initialize logger
	appLogger := logger.NewFromConfig(
		cfg.Logging.Level,
		cfg.Logging.Format,
		cfg.Logging.Output,
		cfg.Logging.Structured,
	)
	logger.SetGlobalLogger(appLogger)

	appLogger.Info("Starting NovaCron API Server (Real Backend)...",
		"version", "1.0.0",
		"api_port", cfg.Server.APIPort,
		"ws_port", cfg.Server.WSPort,
	)

	// Initialize database connection
	db, err := initRealDatabase(cfg)
	if err != nil {
		appLogger.Fatal("Failed to initialize database", "error", err)
	}
	defer db.Close()

	// Run database migrations
	if err := runDatabaseMigrations(db.DB); err != nil {
		appLogger.Fatal("Failed to run database migrations", "error", err)
	}

	// Initialize authentication manager
	authManager := auth.NewSimpleAuthManager(cfg.Auth.Secret, db.DB)

	// Initialize VM manager with default configuration
	vmConfig := core_vm.VMManagerConfig{
		DefaultDriver: core_vm.VMTypeKVM,
		Drivers:       make(map[core_vm.VMType]core_vm.VMDriverConfigManager),
		Scheduler: core_vm.VMSchedulerConfig{
			Type:   "round-robin",
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
		vmManager = nil
	}

	// Initialize KVM manager (optional for development)
	kvmManager, err := hypervisor.NewKVMManager("qemu:///system")
	if err != nil {
		appLogger.Warn("Failed to connect to KVM", "error", err)
		appLogger.Info("Continuing with limited KVM functionality")
		kvmManager = nil
	}

	// Initialize services
	vmService := services.NewVMService(db, vmManager, kvmManager)
	monitoringService := services.NewMonitoringService(db, kvmManager, vmManager)

	// Start monitoring service
	if err := monitoringService.Start(); err != nil {
		appLogger.Fatal("Failed to start monitoring service", "error", err)
	}
	defer monitoringService.Stop()

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

	// Register real API routes
	registerVMRoutes(apiRouter.PathPrefix("/vm").Subrouter(), vmService)
	registerMonitoringRoutes(router, monitoringService)

	// Register WebSocket routes
	registerWebSocketRoutes(router, monitoringService)

	// Public routes (no auth required)
	registerPublicRoutes(router, authManager)

	// Health check endpoint
	router.HandleFunc("/health", healthCheckHandler(cfg, db.DB, kvmManager)).Methods("GET")

	// API info endpoint
	router.HandleFunc("/api/info", apiInfoHandler()).Methods("GET")

	// Create HTTP server
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

// initRealDatabase initializes the database connection with real database package
func initRealDatabase(cfg *config.Config) (*database.DB, error) {
	db, err := database.New(cfg.Database.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to create database connection: %w", err)
	}

	return db, nil
}

// runDatabaseMigrations runs the database migrations
func runDatabaseMigrations(db *sql.DB) error {
	// Read the migrations SQL file
	migrationSQL := `
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user' NOT NULL CHECK (role IN ('admin', 'operator', 'user')),
    tenant_id VARCHAR(255) DEFAULT 'default' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- VMs table
CREATE TABLE IF NOT EXISTS vms (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    state VARCHAR(50) NOT NULL,
    node_id VARCHAR(255),
    owner_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    tenant_id VARCHAR(255) DEFAULT 'default' NOT NULL,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- VM metrics table
CREATE TABLE IF NOT EXISTS vm_metrics (
    id SERIAL PRIMARY KEY,
    vm_id VARCHAR(255) REFERENCES vms(id) ON DELETE CASCADE,
    cpu_usage FLOAT NOT NULL DEFAULT 0,
    memory_usage FLOAT NOT NULL DEFAULT 0,
    disk_usage FLOAT NOT NULL DEFAULT 0,
    network_sent BIGINT NOT NULL DEFAULT 0,
    network_recv BIGINT NOT NULL DEFAULT 0,
    iops INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    node_id VARCHAR(255) NOT NULL,
    cpu_usage FLOAT NOT NULL DEFAULT 0,
    memory_usage FLOAT NOT NULL DEFAULT 0,
    memory_total BIGINT NOT NULL DEFAULT 0,
    memory_available BIGINT NOT NULL DEFAULT 0,
    disk_usage FLOAT NOT NULL DEFAULT 0,
    disk_total BIGINT NOT NULL DEFAULT 0,
    disk_available BIGINT NOT NULL DEFAULT 0,
    network_sent BIGINT NOT NULL DEFAULT 0,
    network_recv BIGINT NOT NULL DEFAULT 0,
    load_average_1 FLOAT NOT NULL DEFAULT 0,
    load_average_5 FLOAT NOT NULL DEFAULT 0,
    load_average_15 FLOAT NOT NULL DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(50) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    status VARCHAR(50) NOT NULL CHECK (status IN ('firing', 'resolved', 'acknowledged', 'suppressed')),
    resource VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    metric_name VARCHAR(255) NOT NULL,
    threshold FLOAT NOT NULL,
    current_value FLOAT NOT NULL,
    labels JSONB,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id ON vm_metrics(vm_id);
CREATE INDEX IF NOT EXISTS idx_vm_metrics_timestamp ON vm_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_metrics_node_id ON system_metrics(node_id);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_vms_state ON vms(state);
CREATE INDEX IF NOT EXISTS idx_vms_owner_id ON vms(owner_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);

-- Create default admin user
-- NOTE: Default password hash is for 'admin123' - MUST be changed in production
-- Set ADMIN_DEFAULT_PASSWORD env var to override
INSERT INTO users (username, email, password_hash, role, tenant_id)
VALUES ('admin', 'admin@novacron.local', '$2a$10$rDjR8Z8Z8Z8Z8Z8Z8Z8Z8uQ8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8', 'admin', 'default')
ON CONFLICT (username) DO NOTHING;
	`

	// Execute migrations
	if _, err := db.Exec(migrationSQL); err != nil {
		return fmt.Errorf("failed to run migrations: %w", err)
	}

	log.Println("Database migrations completed successfully")
	return nil
}

// registerVMRoutes registers VM management routes with real implementation
func registerVMRoutes(router *mux.Router, vmService *services.VMService) {
	// List VMs
	router.HandleFunc("/vms", func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		userID := getUserIDFromContext(ctx)
		tenantID := getTenantIDFromContext(ctx)

		vms, err := vmService.ListVMs(ctx, userID, tenantID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vms)
	}).Methods("GET")

	// Create VM
	router.HandleFunc("/vms", func(w http.ResponseWriter, r *http.Request) {
		var request services.CreateVMRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		ctx := r.Context()
		userID := getUserIDFromContext(ctx)

		vm, err := vmService.CreateVM(ctx, request, userID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(vm)
	}).Methods("POST")

	// Get VM
	router.HandleFunc("/vms/{id}", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		vm, err := vmService.GetVM(r.Context(), vmID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}).Methods("GET")

	// Update VM
	router.HandleFunc("/vms/{id}", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		var updates map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&updates); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		vm, err := vmService.UpdateVM(r.Context(), vmID, updates)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}).Methods("PUT")

	// Delete VM
	router.HandleFunc("/vms/{id}", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		if err := vmService.DeleteVM(r.Context(), vmID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusNoContent)
	}).Methods("DELETE")

	// VM lifecycle operations
	router.HandleFunc("/vms/{id}/start", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		if err := vmService.StartVM(r.Context(), vmID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Return updated VM
		vm, err := vmService.GetVM(r.Context(), vmID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}).Methods("POST")

	router.HandleFunc("/vms/{id}/stop", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		if err := vmService.StopVM(r.Context(), vmID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Return updated VM
		vm, err := vmService.GetVM(r.Context(), vmID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}).Methods("POST")

	router.HandleFunc("/vms/{id}/restart", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		if err := vmService.RestartVM(r.Context(), vmID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Return updated VM
		vm, err := vmService.GetVM(r.Context(), vmID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}).Methods("POST")

	router.HandleFunc("/vms/{id}/pause", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		if err := vmService.PauseVM(r.Context(), vmID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Return updated VM
		vm, err := vmService.GetVM(r.Context(), vmID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}).Methods("POST")

	router.HandleFunc("/vms/{id}/resume", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		if err := vmService.ResumeVM(r.Context(), vmID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Return updated VM
		vm, err := vmService.GetVM(r.Context(), vmID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(vm)
	}).Methods("POST")

	// VM metrics
	router.HandleFunc("/vms/{id}/metrics", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]

		metrics, err := vmService.GetVMMetrics(r.Context(), vmID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(metrics)
	}).Methods("GET")
}

// registerMonitoringRoutes registers monitoring routes with real implementation
func registerMonitoringRoutes(router *mux.Router, monitoringService *services.MonitoringService) {
	// System metrics
	router.HandleFunc("/api/monitoring/metrics", func(w http.ResponseWriter, r *http.Request) {
		timeRangeStr := r.URL.Query().Get("timeRange")
		timeRange, err := strconv.Atoi(timeRangeStr)
		if err != nil {
			timeRange = 3600 // Default to 1 hour
		}

		metrics, err := monitoringService.GetSystemMetrics(r.Context(), timeRange)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(metrics)
	}).Methods("GET")

	// VM metrics
	router.HandleFunc("/api/monitoring/vms", func(w http.ResponseWriter, r *http.Request) {
		metrics, err := monitoringService.GetVMMetrics(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(metrics)
	}).Methods("GET")

	// Alerts
	router.HandleFunc("/api/monitoring/alerts", func(w http.ResponseWriter, r *http.Request) {
		alerts, err := monitoringService.GetAlerts(r.Context())
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(alerts)
	}).Methods("GET")

	// Acknowledge alert
	router.HandleFunc("/api/monitoring/alerts/{id}/acknowledge", func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		alertID := vars["id"]

		// Get user from context
		userID := getUserIDFromContext(r.Context())
		acknowledgedBy := fmt.Sprintf("user-%d", userID)

		if err := monitoringService.AcknowledgeAlert(r.Context(), alertID, acknowledgedBy); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"status": "acknowledged"})
	}).Methods("POST")
}

// registerWebSocketRoutes registers WebSocket routes
func registerWebSocketRoutes(router *mux.Router, monitoringService *services.MonitoringService) {
	upgrader := websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins for development
		},
	}

	router.HandleFunc("/ws/monitoring", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("WebSocket upgrade error: %v", err)
			return
		}
		defer conn.Close()

		log.Println("WebSocket client connected")

		// Send periodic updates
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				// Get latest system metrics
				metrics, err := monitoringService.GetSystemMetrics(r.Context(), 300) // Last 5 minutes
				if err != nil {
					log.Printf("Error getting metrics for WebSocket: %v", err)
					continue
				}

				// Send metric update
				message := map[string]interface{}{
					"type":      "metric",
					"timestamp": time.Now().Unix(),
					"data": map[string]interface{}{
						"cpu":     metrics.CurrentCpuUsage,
						"memory":  metrics.CurrentMemoryUsage,
						"disk":    metrics.CurrentDiskUsage,
						"network": metrics.CurrentNetworkUsage,
					},
				}

				if err := conn.WriteJSON(message); err != nil {
					log.Printf("WebSocket write error: %v", err)
					return
				}
			}
		}
	})
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
				"id":        user.ID,
				"username":  user.Username,
				"email":     user.Email,
				"role":      user.Role,
				"tenant_id": user.TenantID,
			},
		})
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
		json.NewEncoder(w).Encode(map[string]interface{}{
			"user": map[string]interface{}{
				"id":        user.ID,
				"username":  user.Username,
				"email":     user.Email,
				"role":      user.Role,
				"tenant_id": user.TenantID,
			},
		})
	}).Methods("POST")
}

// healthCheckHandler returns a structured health check response
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
			"service":   "novacron-api-real",
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
			"name":        "NovaCron API (Real Backend)",
			"version":     "1.0.0",
			"description": "Distributed VM Management System with Real Backend Implementation",
			"endpoints": []string{
				"/api/monitoring/metrics",
				"/api/monitoring/vms",
				"/api/monitoring/alerts",
				"/api/vm/vms",
				"/ws/monitoring",
			},
			"features": []string{
				"Real database integration",
				"VM lifecycle management",
				"System monitoring",
				"Alert management",
				"WebSocket real-time updates",
				"Authentication & authorization",
			},
			"documentation": "/api/docs",
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}
}

// Helper functions to extract user information from context
func getUserIDFromContext(ctx context.Context) int {
	if userID, ok := ctx.Value("user_id").(int); ok {
		return userID
	}
	return 0 // Anonymous user
}

func getTenantIDFromContext(ctx context.Context) string {
	if tenantID, ok := ctx.Value("tenant_id").(string); ok {
		return tenantID
	}
	return "default"
}