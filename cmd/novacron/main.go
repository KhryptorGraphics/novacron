package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

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
	// TODO: Implement HTTP server

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

	// TODO: Shutdown HTTP server

	log.Println("Shutdown complete")
}
