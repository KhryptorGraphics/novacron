package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func main() {
	// Parse command line flags
	nodeID := flag.String("node-id", "node-1", "Node ID")
	storageDir := flag.String("storage-dir", "/var/lib/novacron", "Storage directory")
	listenAddr := flag.String("listen", ":8080", "HTTP listen address")
	flag.Parse()

	// Create storage directory if it doesn't exist
	if err := os.MkdirAll(*storageDir, 0755); err != nil {
		log.Fatalf("Failed to create storage directory: %v", err)
	}

	// Create VM manager
	vmConfig := vm.DefaultVMManagerConfig()
	vmManager := vm.NewVMManager(vmConfig, *nodeID, *storageDir, nil)

	// Create VM snapshot manager
	snapshotManager := vm.NewVMSnapshotManager(vmManager, *storageDir)
	vmManager.SetSnapshotManager(snapshotManager)

	// Create VM backup manager
	backupStorageProvider := vm.NewLocalBackupStorageProvider(*storageDir)
	backupManager := vm.NewVMBackupManager(vmManager, backupStorageProvider)
	vmManager.SetBackupManager(backupManager)

	// Create VM network manager
	networkManager := vm.NewVMNetworkManager(vmManager)
	vmManager.SetNetworkManager(networkManager)

	// Create VM storage manager
	storageManager := vm.NewVMStorageManager(vmManager)
	vmManager.SetStorageManager(storageManager)

	// Create VM security manager
	certsDir := filepath.Join(*storageDir, "certs")
	securityManager := vm.NewVMSecurityManager(vmManager, certsDir)
	vmManager.SetSecurityManager(securityManager)

	// Create VM health manager
	eventManager := vm.NewVMEventManager(1000)
	healthManager := vm.NewVMHealthManager(vmManager, eventManager, 30*time.Second)
	vmManager.SetHealthManager(healthManager)

	// Create VM monitor
	monitorConfig := vm.MonitoringConfig{
		Enabled:         true,
		IntervalSeconds: 60,
		ResourceThreshold: vm.ResourceThreshold{
			CPUWarningPercent:     80.0,
			CPUCriticalPercent:    95.0,
			MemoryWarningPercent:  80.0,
			MemoryCriticalPercent: 95.0,
			DiskWarningPercent:    80.0,
			DiskCriticalPercent:   95.0,
		},
		AlertConfig: vm.AlertConfig{
			Enabled:         true,
			AlertOnWarning:  true,
			AlertOnCritical: true,
			AlertOnRecover:  true,
		},
	}
	monitor := vm.NewVMMonitor(vmManager, eventManager, monitorConfig)
	vmManager.SetMonitor(monitor)

	// Create VM metrics collector
	metricsConfig := vm.MetricsConfig{
		Enabled:         true,
		IntervalSeconds: 30,
		RetentionHours:  24,
		StoragePath:     filepath.Join(*storageDir, "metrics"),
	}
	metricsCollector := vm.NewVMMetricsCollector(vmManager, metricsConfig)
	vmManager.SetMetricsCollector(metricsCollector)

	// Create VM scheduler
	schedulerConfig := vm.SchedulerConfig{
		Policy:                 vm.SchedulerPolicyRoundRobin,
		EnableResourceChecking: true,
		EnableAntiAffinity:     true,
		EnableNodeLabels:       true,
		MaxVMsPerNode:          100,
		MaxCPUOvercommit:       1.5,
		MaxMemoryOvercommit:    1.2,
	}
	scheduler := vm.NewVMScheduler(schedulerConfig)
	vmManager.SetScheduler(scheduler)

	// Create VM cluster manager
	clusterManager := vm.NewVMClusterManager(vmManager, scheduler)
	vmManager.SetClusterManager(clusterManager)

	// Start health manager
	healthManager.Start()

	// Start monitor
	monitor.Start()

	// Start metrics collector
	metricsCollector.Start()

	// Create HTTP server
	server := createHTTPServer(*listenAddr, vmManager)
	
	// Start HTTP server in goroutine
	go func() {
		log.Printf("Starting HTTP server on %s", *listenAddr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server failed: %v", err)
		}
	}()

	// Wait for interrupt signal
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	// Shutdown
	log.Println("Shutting down...")

	// Stop metrics collector
	metricsCollector.Stop()

	// Stop monitor
	monitor.Stop()

	// Stop health manager
	healthManager.Stop()

	// Cleanup
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Shutdown HTTP server
	if err := server.Shutdown(ctx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	}

	log.Println("Shutdown complete")
}

// createHTTPServer creates and configures the HTTP server
func createHTTPServer(listenAddr string, vmManager *vm.VMManager) *http.Server {
	router := mux.NewRouter()
	
	// API routes
	api := router.PathPrefix("/api/v1").Subrouter()
	
	// VM management endpoints
	api.HandleFunc("/vms", handleListVMs(vmManager)).Methods("GET")
	api.HandleFunc("/vms", handleCreateVM(vmManager)).Methods("POST")
	api.HandleFunc("/vms/{id}", handleGetVM(vmManager)).Methods("GET")
	api.HandleFunc("/vms/{id}", handleDeleteVM(vmManager)).Methods("DELETE")
	api.HandleFunc("/vms/{id}/start", handleStartVM(vmManager)).Methods("POST")
	api.HandleFunc("/vms/{id}/stop", handleStopVM(vmManager)).Methods("POST")
	api.HandleFunc("/vms/{id}/metrics", handleGetVMMetrics(vmManager)).Methods("GET")
	
	// Health check
	api.HandleFunc("/health", handleHealth).Methods("GET")
	
	// Serve static files (frontend)
	router.PathPrefix("/").Handler(http.FileServer(http.Dir("./frontend/dist/")))
	
	// CORS middleware
	corsHandler := handlers.CORS(
		handlers.AllowedOrigins([]string{"*"}),
		handlers.AllowedMethods([]string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
		handlers.AllowedHeaders([]string{"*"}),
	)(router)
	
	return &http.Server{
		Addr:         listenAddr,
		Handler:      corsHandler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
}

// HTTP handlers
func handleListVMs(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vms, err := vmManager.ListVMs(r.Context())
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to list VMs: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"vms": %d, "status": "success"}`, len(vms))
	}
}

func handleCreateVM(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Basic VM creation - would parse JSON body in real implementation
		vmConfig := vm.VMConfig{
			Name:      "test-vm",
			CPUShares: 2,
			MemoryMB:  2048,
			RootFS:    "/var/lib/libvirt/images/test-vm.qcow2",
		}
		
		vmInfo, err := vmManager.CreateVM(r.Context(), vmConfig)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to create VM: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id": "%s", "name": "%s", "status": "created"}`, vmInfo.ID, vmInfo.Name)
	}
}

func handleGetVM(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		state, err := vmManager.GetVMStatus(r.Context(), vmID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to get VM status: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id": "%s", "state": "%s"}`, vmID, state)
	}
}

func handleDeleteVM(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		err := vmManager.DeleteVM(r.Context(), vmID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to delete VM: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id": "%s", "status": "deleted"}`, vmID)
	}
}

func handleStartVM(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		err := vmManager.StartVM(r.Context(), vmID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to start VM: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id": "%s", "status": "started"}`, vmID)
	}
}

func handleStopVM(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		err := vmManager.StopVM(r.Context(), vmID, false)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to stop VM: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id": "%s", "status": "stopped"}`, vmID)
	}
}

func handleGetVMMetrics(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		vars := mux.Vars(r)
		vmID := vars["id"]
		
		metrics, err := vmManager.GetVMMetrics(r.Context(), vmID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to get VM metrics: %v", err), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id": "%s", "cpu_usage": %.2f, "memory_usage": %d}`, 
			vmID, metrics.CPUUsage, metrics.MemoryUsage)
	}
}

func handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"status": "healthy", "timestamp": "%s"}`, time.Now().Format(time.RFC3339))
}
