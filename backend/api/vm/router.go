package vm

import (
	"net/http"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
)

// Router handles VM-related API routes
type Router struct {
	vmHandler        *VMHandler
	migrationHandler *MigrationHandler
	snapshotHandler  *SnapshotHandler
	metricsHandler   *MetricsHandler
	healthHandler    *HealthHandler
	clusterHandler   *ClusterHandler
}

// NewRouter creates a new VM router with all handlers
func NewRouter(
	vmManager vm.VMManager,
	migrationManager vm.MigrationManager,
	snapshotManager vm.SnapshotManager,
	metricsCollector vm.MetricsCollector,
	healthChecker vm.HealthChecker,
	clusterManager vm.ClusterManager,
) *Router {
	return &Router{
		vmHandler:        NewVMHandler(vmManager),
		migrationHandler: NewMigrationHandler(migrationManager),
		snapshotHandler:  NewSnapshotHandler(snapshotManager),
		metricsHandler:   NewMetricsHandler(metricsCollector),
		healthHandler:    NewHealthHandler(healthChecker),
		clusterHandler:   NewClusterHandler(clusterManager),
	}
}

// RegisterRoutes registers all VM-related routes
func (r *Router) RegisterRoutes(router *mux.Router, auth middleware.AuthMiddleware) {
	// Apply authentication middleware to all routes
	api := router.PathPrefix("/api/v1").Subrouter()
	api.Use(auth.Authenticate)

	// VM Core Operations
	r.registerVMRoutes(api)

	// Migration Operations
	r.registerMigrationRoutes(api)

	// Snapshot/Backup Operations
	r.registerSnapshotRoutes(api)

	// Metrics Operations
	r.registerMetricsRoutes(api)

	// Health Check Operations
	r.registerHealthRoutes(api)

	// Cluster Operations
	r.registerClusterRoutes(api)
}

// registerVMRoutes registers core VM management routes
func (r *Router) registerVMRoutes(router *mux.Router) {
	// List VMs with optional filters
	router.HandleFunc("/vms", r.vmHandler.ListVMs).
		Methods(http.MethodGet).
		Name("ListVMs")

	// Create new VM
	router.HandleFunc("/vms", r.vmHandler.CreateVM).
		Methods(http.MethodPost).
		Name("CreateVM")

	// Get VM details
	router.HandleFunc("/vms/{vm_id}", r.vmHandler.GetVM).
		Methods(http.MethodGet).
		Name("GetVM")

	// Update VM configuration
	router.HandleFunc("/vms/{vm_id}", r.vmHandler.UpdateVM).
		Methods(http.MethodPut, http.MethodPatch).
		Name("UpdateVM")

	// Delete VM
	router.HandleFunc("/vms/{vm_id}", r.vmHandler.DeleteVM).
		Methods(http.MethodDelete).
		Name("DeleteVM")

	// VM Power Operations
	router.HandleFunc("/vms/{vm_id}/start", r.vmHandler.StartVM).
		Methods(http.MethodPost).
		Name("StartVM")

	router.HandleFunc("/vms/{vm_id}/stop", r.vmHandler.StopVM).
		Methods(http.MethodPost).
		Name("StopVM")

	router.HandleFunc("/vms/{vm_id}/restart", r.vmHandler.RestartVM).
		Methods(http.MethodPost).
		Name("RestartVM")

	router.HandleFunc("/vms/{vm_id}/suspend", r.vmHandler.SuspendVM).
		Methods(http.MethodPost).
		Name("SuspendVM")

	router.HandleFunc("/vms/{vm_id}/resume", r.vmHandler.ResumeVM).
		Methods(http.MethodPost).
		Name("ResumeVM")

	// VM Console Access
	router.HandleFunc("/vms/{vm_id}/console", r.vmHandler.GetConsole).
		Methods(http.MethodGet).
		Name("GetVMConsole")

	// VM Resource Operations
	router.HandleFunc("/vms/{vm_id}/resize", r.vmHandler.ResizeVM).
		Methods(http.MethodPost).
		Name("ResizeVM")

	router.HandleFunc("/vms/{vm_id}/attach-volume", r.vmHandler.AttachVolume).
		Methods(http.MethodPost).
		Name("AttachVolume")

	router.HandleFunc("/vms/{vm_id}/detach-volume", r.vmHandler.DetachVolume).
		Methods(http.MethodPost).
		Name("DetachVolume")

	router.HandleFunc("/vms/{vm_id}/attach-network", r.vmHandler.AttachNetwork).
		Methods(http.MethodPost).
		Name("AttachNetwork")

	router.HandleFunc("/vms/{vm_id}/detach-network", r.vmHandler.DetachNetwork).
		Methods(http.MethodPost).
		Name("DetachNetwork")
}

// registerMigrationRoutes registers VM migration routes
func (r *Router) registerMigrationRoutes(router *mux.Router) {
	// Initiate VM migration
	router.HandleFunc("/vms/{vm_id}/migrate", r.migrationHandler.InitiateMigration).
		Methods(http.MethodPost).
		Name("InitiateMigration")

	// Get migration status
	router.HandleFunc("/migrations/{migration_id}", r.migrationHandler.GetMigrationStatus).
		Methods(http.MethodGet).
		Name("GetMigrationStatus")

	// List all migrations
	router.HandleFunc("/migrations", r.migrationHandler.ListMigrations).
		Methods(http.MethodGet).
		Name("ListMigrations")

	// Cancel migration
	router.HandleFunc("/migrations/{migration_id}/cancel", r.migrationHandler.CancelMigration).
		Methods(http.MethodPost).
		Name("CancelMigration")

	// Pause migration
	router.HandleFunc("/migrations/{migration_id}/pause", r.migrationHandler.PauseMigration).
		Methods(http.MethodPost).
		Name("PauseMigration")

	// Resume migration
	router.HandleFunc("/migrations/{migration_id}/resume", r.migrationHandler.ResumeMigration).
		Methods(http.MethodPost).
		Name("ResumeMigration")

	// Get migration history for a VM
	router.HandleFunc("/vms/{vm_id}/migration-history", r.migrationHandler.GetVMMigrationHistory).
		Methods(http.MethodGet).
		Name("GetVMMigrationHistory")
}

// registerSnapshotRoutes registers snapshot/backup routes
func (r *Router) registerSnapshotRoutes(router *mux.Router) {
	// List VM snapshots
	router.HandleFunc("/vms/{vm_id}/snapshots", r.snapshotHandler.ListSnapshots).
		Methods(http.MethodGet).
		Name("ListSnapshots")

	// Create VM snapshot
	router.HandleFunc("/vms/{vm_id}/snapshots", r.snapshotHandler.CreateSnapshot).
		Methods(http.MethodPost).
		Name("CreateSnapshot")

	// Get snapshot details
	router.HandleFunc("/snapshots/{snapshot_id}", r.snapshotHandler.GetSnapshot).
		Methods(http.MethodGet).
		Name("GetSnapshot")

	// Delete snapshot
	router.HandleFunc("/snapshots/{snapshot_id}", r.snapshotHandler.DeleteSnapshot).
		Methods(http.MethodDelete).
		Name("DeleteSnapshot")

	// Restore from snapshot
	router.HandleFunc("/snapshots/{snapshot_id}/restore", r.snapshotHandler.RestoreSnapshot).
		Methods(http.MethodPost).
		Name("RestoreSnapshot")

	// Export snapshot
	router.HandleFunc("/snapshots/{snapshot_id}/export", r.snapshotHandler.ExportSnapshot).
		Methods(http.MethodPost).
		Name("ExportSnapshot")

	// Import snapshot
	router.HandleFunc("/snapshots/import", r.snapshotHandler.ImportSnapshot).
		Methods(http.MethodPost).
		Name("ImportSnapshot")
}

// registerMetricsRoutes registers metrics routes
func (r *Router) registerMetricsRoutes(router *mux.Router) {
	// Get VM metrics
	router.HandleFunc("/vms/{vm_id}/metrics", r.metricsHandler.GetVMMetrics).
		Methods(http.MethodGet).
		Name("GetVMMetrics")

	// Get aggregated metrics
	router.HandleFunc("/metrics", r.metricsHandler.GetAggregatedMetrics).
		Methods(http.MethodGet).
		Name("GetAggregatedMetrics")

	// Get resource utilization
	router.HandleFunc("/metrics/utilization", r.metricsHandler.GetUtilization).
		Methods(http.MethodGet).
		Name("GetUtilization")

	// Get performance metrics
	router.HandleFunc("/metrics/performance", r.metricsHandler.GetPerformanceMetrics).
		Methods(http.MethodGet).
		Name("GetPerformanceMetrics")

	// Export metrics (Prometheus format)
	router.HandleFunc("/metrics/export", r.metricsHandler.ExportMetrics).
		Methods(http.MethodGet).
		Name("ExportMetrics")
}

// registerHealthRoutes registers health check routes
func (r *Router) registerHealthRoutes(router *mux.Router) {
	// Get VM health status
	router.HandleFunc("/vms/{vm_id}/health", r.healthHandler.GetVMHealth).
		Methods(http.MethodGet).
		Name("GetVMHealth")

	// Get system health
	router.HandleFunc("/health", r.healthHandler.GetSystemHealth).
		Methods(http.MethodGet).
		Name("GetSystemHealth")

	// Get service health
	router.HandleFunc("/health/services", r.healthHandler.GetServicesHealth).
		Methods(http.MethodGet).
		Name("GetServicesHealth")

	// Liveness probe
	router.HandleFunc("/health/live", r.healthHandler.LivenessProbe).
		Methods(http.MethodGet).
		Name("LivenessProbe")

	// Readiness probe
	router.HandleFunc("/health/ready", r.healthHandler.ReadinessProbe).
		Methods(http.MethodGet).
		Name("ReadinessProbe")
}

// registerClusterRoutes registers cluster management routes
func (r *Router) registerClusterRoutes(router *mux.Router) {
	// List cluster nodes
	router.HandleFunc("/cluster/nodes", r.clusterHandler.ListNodes).
		Methods(http.MethodGet).
		Name("ListClusterNodes")

	// Get node details
	router.HandleFunc("/cluster/nodes/{node_id}", r.clusterHandler.GetNode).
		Methods(http.MethodGet).
		Name("GetClusterNode")

	// Add node to cluster
	router.HandleFunc("/cluster/nodes", r.clusterHandler.AddNode).
		Methods(http.MethodPost).
		Name("AddClusterNode")

	// Remove node from cluster
	router.HandleFunc("/cluster/nodes/{node_id}", r.clusterHandler.RemoveNode).
		Methods(http.MethodDelete).
		Name("RemoveClusterNode")

	// Drain node (evacuate VMs)
	router.HandleFunc("/cluster/nodes/{node_id}/drain", r.clusterHandler.DrainNode).
		Methods(http.MethodPost).
		Name("DrainClusterNode")

	// Cordon node (prevent new VMs)
	router.HandleFunc("/cluster/nodes/{node_id}/cordon", r.clusterHandler.CordonNode).
		Methods(http.MethodPost).
		Name("CordonClusterNode")

	// Uncordon node
	router.HandleFunc("/cluster/nodes/{node_id}/uncordon", r.clusterHandler.UncordonNode).
		Methods(http.MethodPost).
		Name("UncordonClusterNode")

	// Get cluster status
	router.HandleFunc("/cluster/status", r.clusterHandler.GetClusterStatus).
		Methods(http.MethodGet).
		Name("GetClusterStatus")

	// Rebalance cluster
	router.HandleFunc("/cluster/rebalance", r.clusterHandler.RebalanceCluster).
		Methods(http.MethodPost).
		Name("RebalanceCluster")
}