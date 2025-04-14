package vm

import (
	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// RegisterRoutes registers all VM API routes
func RegisterRoutes(router *mux.Router, vmManager *vm.VMManager) {
	// Create API handlers
	vmHandler := NewHandler(vmManager)
	migrationHandler := NewMigrationHandler(vmManager)
	
	// Register VM routes
	vmRouter := router.PathPrefix("/api/v1").Subrouter()
	vmHandler.RegisterRoutes(vmRouter)
	migrationHandler.RegisterRoutes(vmRouter)
	
	// Register snapshot routes if snapshot manager is available
	if snapshotManager := vmManager.GetSnapshotManager(); snapshotManager != nil {
		snapshotHandler := NewSnapshotHandler(snapshotManager)
		snapshotHandler.RegisterRoutes(vmRouter)
	}
	
	// Register backup routes if backup manager is available
	if backupManager := vmManager.GetBackupManager(); backupManager != nil {
		backupHandler := NewBackupHandler(backupManager)
		backupHandler.RegisterRoutes(vmRouter)
	}
	
	// Register cluster routes if cluster manager is available
	if clusterManager := vmManager.GetClusterManager(); clusterManager != nil {
		clusterHandler := NewClusterHandler(clusterManager)
		clusterHandler.RegisterRoutes(vmRouter)
	}
	
	// Register network routes if network manager is available
	if networkManager := vmManager.GetNetworkManager(); networkManager != nil {
		networkHandler := NewNetworkHandler(networkManager)
		networkHandler.RegisterRoutes(vmRouter)
	}
	
	// Register storage routes if storage manager is available
	if storageManager := vmManager.GetStorageManager(); storageManager != nil {
		storageHandler := NewStorageHandler(storageManager)
		storageHandler.RegisterRoutes(vmRouter)
	}
	
	// Register security routes if security manager is available
	if securityManager := vmManager.GetSecurityManager(); securityManager != nil {
		securityHandler := NewSecurityHandler(securityManager)
		securityHandler.RegisterRoutes(vmRouter)
	}
	
	// Register health routes if health manager is available
	if healthManager := vmManager.GetHealthManager(); healthManager != nil {
		healthHandler := NewHealthHandler(healthManager)
		healthHandler.RegisterRoutes(vmRouter)
	}
	
	// Register monitor routes if monitor is available
	if monitor := vmManager.GetMonitor(); monitor != nil {
		monitorHandler := NewMonitorHandler(monitor)
		monitorHandler.RegisterRoutes(vmRouter)
	}
	
	// Register metrics routes if metrics collector is available
	if metricsCollector := vmManager.GetMetricsCollector(); metricsCollector != nil {
		metricsHandler := NewMetricsHandler(metricsCollector)
		metricsHandler.RegisterRoutes(vmRouter)
	}
	
	// Register scheduler routes if scheduler is available
	if scheduler := vmManager.GetScheduler(); scheduler != nil {
		schedulerHandler := NewSchedulerHandler(scheduler, vmManager)
		schedulerHandler.RegisterRoutes(vmRouter)
	}
}
