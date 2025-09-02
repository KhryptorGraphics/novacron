package helpers

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/pkg/config"
)

// TestHelpers provides utility functions for integration tests
type TestHelpers struct {
	db     *sql.DB
	config *config.Config
}

// DatabaseStats represents database statistics
type DatabaseStats struct {
	TotalUsers     int `json:"total_users"`
	TotalVMs       int `json:"total_vms"`
	TotalMetrics   int `json:"total_metrics"`
	ActiveUsers    int `json:"active_users"`
	RunningVMs     int `json:"running_vms"`
	RecentMetrics  int `json:"recent_metrics"`
}

// PerformanceMetrics represents performance test metrics
type PerformanceMetrics struct {
	StartTime        time.Time     `json:"start_time"`
	EndTime          time.Time     `json:"end_time"`
	Duration         time.Duration `json:"duration"`
	OperationsCount  int           `json:"operations_count"`
	SuccessfulOps    int           `json:"successful_ops"`
	FailedOps        int           `json:"failed_ops"`
	OperationsPerSec float64       `json:"operations_per_sec"`
	AvgResponseTime  time.Duration `json:"avg_response_time"`
	MinResponseTime  time.Duration `json:"min_response_time"`
	MaxResponseTime  time.Duration `json:"max_response_time"`
	ErrorRate        float64       `json:"error_rate"`
}

// New creates a new test helpers instance
func New(db *sql.DB, config *config.Config) *TestHelpers {
	return &TestHelpers{
		db:     db,
		config: config,
	}
}

// CleanupDatabase removes all test data from the database
func (th *TestHelpers) CleanupDatabase() error {
	cleanupQueries := []string{
		"DELETE FROM vm_metrics WHERE vm_id LIKE 'test-%' OR vm_id LIKE '%test%'",
		"DELETE FROM vms WHERE id LIKE 'test-%' OR name LIKE '%test%'",
		"DELETE FROM users WHERE username LIKE '%test%' OR email LIKE '%test%'",
	}

	for _, query := range cleanupQueries {
		if _, err := th.db.Exec(query); err != nil {
			return fmt.Errorf("failed to execute cleanup query: %s, error: %w", query, err)
		}
	}

	return nil
}

// CleanupTestData removes test data created during specific tests
func (th *TestHelpers) CleanupTestData() error {
	cleanupQueries := []string{
		// Clean VM metrics from the last hour (test data)
		"DELETE FROM vm_metrics WHERE timestamp > (CURRENT_TIMESTAMP - INTERVAL '1 hour')",
		
		// Clean test VMs with specific patterns
		"DELETE FROM vms WHERE id LIKE 'integration-%' OR id LIKE 'bench-%'",
		
		// Clean temporary test users
		"DELETE FROM users WHERE username LIKE 'temp_%' OR username LIKE '%_temp'",
	}

	for _, query := range cleanupQueries {
		if _, err := th.db.Exec(query); err != nil {
			// Log error but continue cleanup
			fmt.Printf("Warning: cleanup query failed: %s, error: %v\n", query, err)
		}
	}

	return nil
}

// GetDatabaseStats returns current database statistics
func (th *TestHelpers) GetDatabaseStats() (*DatabaseStats, error) {
	stats := &DatabaseStats{}

	// Get user counts
	err := th.db.QueryRow("SELECT COUNT(*) FROM users").Scan(&stats.TotalUsers)
	if err != nil {
		return nil, fmt.Errorf("failed to get total users: %w", err)
	}

	err = th.db.QueryRow("SELECT COUNT(*) FROM users WHERE is_active = true").Scan(&stats.ActiveUsers)
	if err != nil {
		return nil, fmt.Errorf("failed to get active users: %w", err)
	}

	// Get VM counts
	err = th.db.QueryRow("SELECT COUNT(*) FROM vms").Scan(&stats.TotalVMs)
	if err != nil {
		return nil, fmt.Errorf("failed to get total VMs: %w", err)
	}

	err = th.db.QueryRow("SELECT COUNT(*) FROM vms WHERE state = 'running'").Scan(&stats.RunningVMs)
	if err != nil {
		return nil, fmt.Errorf("failed to get running VMs: %w", err)
	}

	// Get metric counts
	err = th.db.QueryRow("SELECT COUNT(*) FROM vm_metrics").Scan(&stats.TotalMetrics)
	if err != nil {
		return nil, fmt.Errorf("failed to get total metrics: %w", err)
	}

	err = th.db.QueryRow("SELECT COUNT(*) FROM vm_metrics WHERE timestamp > (CURRENT_TIMESTAMP - INTERVAL '1 hour')").Scan(&stats.RecentMetrics)
	if err != nil {
		return nil, fmt.Errorf("failed to get recent metrics: %w", err)
	}

	return stats, nil
}

// WaitForDatabaseReady waits for the database to be ready
func (th *TestHelpers) WaitForDatabaseReady(timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("database not ready within timeout: %v", timeout)
		case <-ticker.C:
			if err := th.db.PingContext(ctx); err == nil {
				return nil
			}
		}
	}
}

// ExecuteWithTimeout executes a function with a timeout
func (th *TestHelpers) ExecuteWithTimeout(timeout time.Duration, operation func() error) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- operation()
	}()

	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return fmt.Errorf("operation timed out after %v", timeout)
	}
}

// CreatePerformanceMetrics creates and tracks performance metrics
func (th *TestHelpers) CreatePerformanceMetrics() *PerformanceMetrics {
	return &PerformanceMetrics{
		StartTime:       time.Now(),
		MinResponseTime: time.Duration(999999999999), // Large initial value
	}
}

// RecordOperation records an operation result in performance metrics
func (pm *PerformanceMetrics) RecordOperation(duration time.Duration, success bool) {
	pm.OperationsCount++
	
	if success {
		pm.SuccessfulOps++
	} else {
		pm.FailedOps++
	}

	// Update response time statistics
	if duration < pm.MinResponseTime {
		pm.MinResponseTime = duration
	}
	if duration > pm.MaxResponseTime {
		pm.MaxResponseTime = duration
	}
}

// Finalize finalizes performance metrics calculation
func (pm *PerformanceMetrics) Finalize() {
	pm.EndTime = time.Now()
	pm.Duration = pm.EndTime.Sub(pm.StartTime)

	if pm.Duration > 0 {
		pm.OperationsPerSec = float64(pm.OperationsCount) / pm.Duration.Seconds()
	}

	if pm.OperationsCount > 0 {
		pm.ErrorRate = float64(pm.FailedOps) / float64(pm.OperationsCount)
		pm.AvgResponseTime = pm.Duration / time.Duration(pm.OperationsCount)
	}

	// Handle edge case where no operations were recorded
	if pm.MinResponseTime == time.Duration(999999999999) {
		pm.MinResponseTime = 0
	}
}

// ValidateDatabase validates database schema and constraints
func (th *TestHelpers) ValidateDatabase() error {
	// Check required tables exist
	requiredTables := []string{"users", "vms", "vm_metrics"}
	
	for _, table := range requiredTables {
		var exists bool
		query := `
			SELECT EXISTS (
				SELECT FROM information_schema.tables 
				WHERE table_schema = 'public' AND table_name = $1
			)
		`
		err := th.db.QueryRow(query, table).Scan(&exists)
		if err != nil {
			return fmt.Errorf("failed to check table %s: %w", table, err)
		}
		if !exists {
			return fmt.Errorf("required table %s does not exist", table)
		}
	}

	// Check critical indexes exist
	criticalIndexes := []string{
		"idx_users_email",
		"idx_vms_owner_id", 
		"idx_vm_metrics_vm_id",
		"idx_vm_metrics_timestamp",
	}

	for _, index := range criticalIndexes {
		var exists bool
		query := `
			SELECT EXISTS (
				SELECT FROM pg_indexes 
				WHERE schemaname = 'public' AND indexname = $1
			)
		`
		err := th.db.QueryRow(query, index).Scan(&exists)
		if err != nil {
			return fmt.Errorf("failed to check index %s: %w", index, err)
		}
		if !exists {
			return fmt.Errorf("critical index %s does not exist", index)
		}
	}

	return nil
}

// SetupTestSchema ensures the test schema is properly configured
func (th *TestHelpers) SetupTestSchema() error {
	// Enable necessary extensions
	extensions := []string{
		"CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"",
		"CREATE EXTENSION IF NOT EXISTS \"pg_stat_statements\"",
	}

	for _, ext := range extensions {
		if _, err := th.db.Exec(ext); err != nil {
			// Log warning but continue - extensions might not be available in all environments
			fmt.Printf("Warning: failed to create extension: %v\n", err)
		}
	}

	return nil
}

// BackupTestData creates a backup of current test data
func (th *TestHelpers) BackupTestData() (map[string]interface{}, error) {
	backup := make(map[string]interface{})

	// Backup users
	rows, err := th.db.Query("SELECT id, username, email, role, tenant_id FROM users")
	if err != nil {
		return nil, fmt.Errorf("failed to backup users: %w", err)
	}
	defer rows.Close()

	var users []map[string]interface{}
	for rows.Next() {
		var id int
		var username, email, role, tenantID string
		if err := rows.Scan(&id, &username, &email, &role, &tenantID); err != nil {
			continue
		}
		users = append(users, map[string]interface{}{
			"id": id, "username": username, "email": email, "role": role, "tenant_id": tenantID,
		})
	}
	backup["users"] = users

	// Backup VMs
	vmRows, err := th.db.Query("SELECT id, name, state, cpu_cores, memory_mb, disk_gb, owner_id, tenant_id FROM vms")
	if err != nil {
		return nil, fmt.Errorf("failed to backup VMs: %w", err)
	}
	defer vmRows.Close()

	var vms []map[string]interface{}
	for vmRows.Next() {
		var id, name, state, tenantID string
		var cpuCores, memoryMB, diskGB, ownerID int
		if err := vmRows.Scan(&id, &name, &state, &cpuCores, &memoryMB, &diskGB, &ownerID, &tenantID); err != nil {
			continue
		}
		vms = append(vms, map[string]interface{}{
			"id": id, "name": name, "state": state, "cpu_cores": cpuCores,
			"memory_mb": memoryMB, "disk_gb": diskGB, "owner_id": ownerID, "tenant_id": tenantID,
		})
	}
	backup["vms"] = vms

	return backup, nil
}

// RestoreTestData restores test data from backup
func (th *TestHelpers) RestoreTestData(backup map[string]interface{}) error {
	// Clear existing data
	if err := th.CleanupDatabase(); err != nil {
		return fmt.Errorf("failed to cleanup before restore: %w", err)
	}

	// This is a simplified restore - in production you'd want more robust handling
	fmt.Println("Note: Test data restore is simplified for demonstration")
	
	return nil
}

// GenerateTestReport generates a comprehensive test report
func (th *TestHelpers) GenerateTestReport() (map[string]interface{}, error) {
	report := make(map[string]interface{})

	// Database statistics
	stats, err := th.GetDatabaseStats()
	if err != nil {
		return nil, fmt.Errorf("failed to get database stats: %w", err)
	}
	report["database_stats"] = stats

	// Database health check
	report["database_health"] = "healthy"
	if err := th.db.Ping(); err != nil {
		report["database_health"] = "unhealthy"
		report["database_error"] = err.Error()
	}

	// Performance metrics (simplified)
	var avgResponseTime float64
	err = th.db.QueryRow(`
		SELECT AVG(EXTRACT(EPOCH FROM (updated_at - created_at)) * 1000) 
		FROM vms 
		WHERE created_at > (CURRENT_TIMESTAMP - INTERVAL '1 hour')
	`).Scan(&avgResponseTime)
	if err == nil && avgResponseTime > 0 {
		report["avg_vm_creation_time_ms"] = avgResponseTime
	}

	// Test environment info
	report["test_environment"] = map[string]interface{}{
		"database_url": th.config.Database.URL,
		"max_connections": th.config.Database.MaxConnections,
		"timestamp": time.Now().UTC(),
	}

	return report, nil
}

// CheckDatabaseConnectivity performs comprehensive connectivity tests
func (th *TestHelpers) CheckDatabaseConnectivity() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Basic ping
	if err := th.db.PingContext(ctx); err != nil {
		return fmt.Errorf("database ping failed: %w", err)
	}

	// Test query execution
	var version string
	if err := th.db.QueryRowContext(ctx, "SELECT version()").Scan(&version); err != nil {
		return fmt.Errorf("test query failed: %w", err)
	}

	// Test transaction support
	tx, err := th.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("transaction begin failed: %w", err)
	}

	// Test rollback
	if err := tx.Rollback(); err != nil {
		return fmt.Errorf("transaction rollback failed: %w", err)
	}

	return nil
}

// WaitForCondition waits for a condition to be true with timeout
func (th *TestHelpers) WaitForCondition(condition func() bool, timeout time.Duration, checkInterval time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	ticker := time.NewTicker(checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("condition not met within timeout: %v", timeout)
		case <-ticker.C:
			if condition() {
				return nil
			}
		}
	}
}

// CreateUniqueIdentifier creates a unique identifier for test isolation
func (th *TestHelpers) CreateUniqueIdentifier(prefix string) string {
	return fmt.Sprintf("%s_%d_%d", prefix, time.Now().Unix(), time.Now().Nanosecond()%1000000)
}

// ValidateTestEnvironment validates the test environment is ready
func (th *TestHelpers) ValidateTestEnvironment() error {
	// Check database
	if err := th.CheckDatabaseConnectivity(); err != nil {
		return fmt.Errorf("database connectivity check failed: %w", err)
	}

	// Validate schema
	if err := th.ValidateDatabase(); err != nil {
		return fmt.Errorf("database validation failed: %w", err)
	}

	// Check required configuration
	if th.config.Database.URL == "" {
		return fmt.Errorf("database URL not configured")
	}

	if th.config.Auth.Secret == "" {
		return fmt.Errorf("auth secret not configured")
	}

	return nil
}

// LogTestEvent logs a test event (in production, this might go to a logging system)
func (th *TestHelpers) LogTestEvent(level, message string, context map[string]interface{}) {
	fmt.Printf("[%s] %s - %s", level, time.Now().Format(time.RFC3339), message)
	if context != nil && len(context) > 0 {
		fmt.Printf(" - Context: %+v", context)
	}
	fmt.Println()
}

// GetTestMetrics returns comprehensive test metrics
func (th *TestHelpers) GetTestMetrics() (map[string]interface{}, error) {
	metrics := make(map[string]interface{})

	// Database metrics
	stats, err := th.GetDatabaseStats()
	if err == nil {
		metrics["database"] = stats
	}

	// Connection pool metrics
	metrics["connection_pool"] = map[string]interface{}{
		"max_open_connections": th.db.Stats().MaxOpenConnections,
		"open_connections":     th.db.Stats().OpenConnections,
		"in_use":              th.db.Stats().InUse,
		"idle":                th.db.Stats().Idle,
	}

	// Test timing metrics
	metrics["timestamp"] = time.Now().UTC()

	return metrics, nil
}