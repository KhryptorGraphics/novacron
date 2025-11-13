package prometheus

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// API Metrics
	APIRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_api_requests_total",
			Help: "Total number of API requests",
		},
		[]string{"method", "endpoint", "status"},
	)

	APIRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "novacron_api_request_duration_seconds",
			Help:    "API request duration in seconds",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)

	APIErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_api_errors_total",
			Help: "Total number of API errors",
		},
		[]string{"method", "endpoint", "error_type"},
	)

	// DWCP Protocol Metrics
	DWCPMigrationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_dwcp_migrations_total",
			Help: "Total number of VM migrations",
		},
		[]string{"source", "destination", "status"},
	)

	DWCPMigrationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "novacron_dwcp_migration_duration_seconds",
			Help:    "VM migration duration in seconds",
			Buckets: []float64{1, 5, 10, 30, 60, 120, 300},
		},
		[]string{"source", "destination"},
	)

	DWCPBandwidthUtilization = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_dwcp_bandwidth_utilization_percent",
			Help: "DWCP bandwidth utilization percentage",
		},
		[]string{"link"},
	)

	DWCPCompressionRatio = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_dwcp_compression_ratio",
			Help: "DWCP compression ratio",
		},
		[]string{"algorithm"},
	)

	// Database Metrics
	DatabaseQueriesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_database_queries_total",
			Help: "Total number of database queries",
		},
		[]string{"operation", "table", "status"},
	)

	DatabaseQueryDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "novacron_database_query_duration_seconds",
			Help:    "Database query duration in seconds",
			Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0},
		},
		[]string{"operation", "table"},
	)

	DatabaseConnectionsActive = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_database_connections_active",
			Help: "Number of active database connections",
		},
	)

	DatabaseConnectionPoolSize = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_database_connection_pool_size",
			Help: "Database connection pool size",
		},
	)

	// VM Lifecycle Metrics
	VMsTotal = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_vms_total",
			Help: "Total number of VMs by state",
		},
		[]string{"state"},
	)

	VMOperationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_vm_operations_total",
			Help: "Total number of VM operations",
		},
		[]string{"operation", "status"},
	)

	VMOperationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "novacron_vm_operation_duration_seconds",
			Help:    "VM operation duration in seconds",
			Buckets: []float64{0.1, 0.5, 1, 5, 10, 30, 60},
		},
		[]string{"operation"},
	)

	VMResourceUsage = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_vm_resource_usage",
			Help: "VM resource usage (CPU, memory, disk)",
		},
		[]string{"vm_id", "resource"},
	)

	// Business Metrics
	ActiveUsersTotal = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_active_users_total",
			Help: "Total number of active users",
		},
	)

	MigrationsPerHour = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_migrations_per_hour",
			Help: "Number of migrations per hour",
		},
	)

	APIUsageByUser = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_api_usage_by_user_total",
			Help: "API usage by user",
		},
		[]string{"user_id", "endpoint"},
	)

	// System Resource Metrics
	SystemCPUUsage = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_system_cpu_usage_percent",
			Help: "System CPU usage percentage",
		},
	)

	SystemMemoryUsage = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_system_memory_usage_bytes",
			Help: "System memory usage in bytes",
		},
	)

	SystemDiskUsage = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_system_disk_usage_bytes",
			Help: "System disk usage in bytes",
		},
		[]string{"mount_point"},
	)

	SystemNetworkBytesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_system_network_bytes_total",
			Help: "Total network bytes transmitted/received",
		},
		[]string{"interface", "direction"},
	)

	// Availability and Health
	ServiceAvailability = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_service_availability",
			Help: "Service availability (1=up, 0=down)",
		},
		[]string{"service"},
	)

	HealthCheckStatus = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_health_check_status",
			Help: "Health check status (1=healthy, 0=unhealthy)",
		},
		[]string{"component"},
	)
)

// RecordAPIRequest records an API request with duration and status
func RecordAPIRequest(method, endpoint, status string, duration float64) {
	APIRequestsTotal.WithLabelValues(method, endpoint, status).Inc()
	APIRequestDuration.WithLabelValues(method, endpoint).Observe(duration)
}

// RecordAPIError records an API error
func RecordAPIError(method, endpoint, errorType string) {
	APIErrorsTotal.WithLabelValues(method, endpoint, errorType).Inc()
}

// RecordDWCPMigration records a DWCP migration
func RecordDWCPMigration(source, destination, status string, duration float64) {
	DWCPMigrationsTotal.WithLabelValues(source, destination, status).Inc()
	if duration > 0 {
		DWCPMigrationDuration.WithLabelValues(source, destination).Observe(duration)
	}
}

// UpdateDWCPBandwidth updates DWCP bandwidth utilization
func UpdateDWCPBandwidth(link string, utilization float64) {
	DWCPBandwidthUtilization.WithLabelValues(link).Set(utilization)
}

// RecordDatabaseQuery records a database query
func RecordDatabaseQuery(operation, table, status string, duration float64) {
	DatabaseQueriesTotal.WithLabelValues(operation, table, status).Inc()
	DatabaseQueryDuration.WithLabelValues(operation, table).Observe(duration)
}

// UpdateDatabaseConnections updates database connection metrics
func UpdateDatabaseConnections(active, poolSize int) {
	DatabaseConnectionsActive.Set(float64(active))
	DatabaseConnectionPoolSize.Set(float64(poolSize))
}

// UpdateVMCount updates VM count by state
func UpdateVMCount(state string, count int) {
	VMsTotal.WithLabelValues(state).Set(float64(count))
}

// RecordVMOperation records a VM operation
func RecordVMOperation(operation, status string, duration float64) {
	VMOperationsTotal.WithLabelValues(operation, status).Inc()
	if duration > 0 {
		VMOperationDuration.WithLabelValues(operation).Observe(duration)
	}
}

// UpdateSystemMetrics updates system resource metrics
func UpdateSystemMetrics(cpuPercent, memoryBytes float64) {
	SystemCPUUsage.Set(cpuPercent)
	SystemMemoryUsage.Set(memoryBytes)
}

// UpdateServiceAvailability updates service availability status
func UpdateServiceAvailability(service string, available bool) {
	if available {
		ServiceAvailability.WithLabelValues(service).Set(1)
	} else {
		ServiceAvailability.WithLabelValues(service).Set(0)
	}
}
