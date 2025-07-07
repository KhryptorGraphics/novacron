package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/api/monitoring"
	"github.com/khryptorgraphics/novacron/backend/api/vm"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
)

func main() {
	log.Println("Starting NovaCron API Server...")

	// Initialize KVM manager
	kvmManager, err := hypervisor.NewKVMManager("qemu:///system")
	if err != nil {
		log.Printf("Warning: Failed to connect to KVM: %v", err)
		log.Println("Continuing with limited functionality...")
		// Create a nil manager for development
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

	// Register API routes
	if kvmManager != nil {
		// VM management routes
		vmHandlers := vm.NewVMHandlers(kvmManager)
		vmHandlers.RegisterRoutes(router)

		// Monitoring routes
		monitoringHandlers := monitoring.NewMonitoringHandlers(kvmManager)
		monitoringHandlers.RegisterRoutes(router)
	} else {
		// Register mock handlers for development
		registerMockHandlers(router)
	}

	// Health check endpoint
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status": "healthy", "timestamp": "` + time.Now().Format(time.RFC3339) + `"}`))
	}).Methods("GET")

	// API info endpoint
	router.HandleFunc("/api/info", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{
			"name": "NovaCron API",
			"version": "1.0.0",
			"description": "Distributed VM Management System",
			"endpoints": [
				"/api/monitoring/metrics",
				"/api/monitoring/vms", 
				"/api/monitoring/alerts",
				"/api/vm/list",
				"/api/vm/create",
				"/ws/monitoring"
			]
		}`))
	}).Methods("GET")

	// Create HTTP server
	server := &http.Server{
		Addr:         ":8090",
		Handler:      corsHandler(router),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Start server in a goroutine
	go func() {
		log.Println("API Server listening on :8090")
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server failed to start: %v", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	// Close KVM manager connection
	if kvmManager != nil {
		if err := kvmManager.Close(); err != nil {
			log.Printf("Error closing KVM manager: %v", err)
		}
	}

	log.Println("Server exited")
}

// registerMockHandlers registers mock handlers for development when KVM is not available
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