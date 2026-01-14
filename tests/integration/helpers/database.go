package helpers

import (
	"database/sql"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	_ "github.com/lib/pq"
)

// DatabaseHelper provides utilities for database testing
type DatabaseHelper struct {
	DB *sql.DB
	TestDB string
}

// NewDatabaseHelper creates a new database helper
func NewDatabaseHelper(dbURL string) (*DatabaseHelper, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %w", err)
	}
	
	// Test connection
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}
	
	return &DatabaseHelper{
		DB: db,
		TestDB: "novacron_test",
	}, nil
}

// Close closes the database connection
func (h *DatabaseHelper) Close() error {
	if h.DB != nil {
		return h.DB.Close()
	}
	return nil
}

// SetupTestDatabase creates and initializes test database
func (h *DatabaseHelper) SetupTestDatabase(t *testing.T) {
	t.Helper()
	
	// Ensure test database exists
	h.ensureTestDatabaseExists(t)
	
	// Run migrations
	h.runMigrations(t)
	
	// Seed test data
	h.seedTestData(t)
}

// CleanupTestDatabase cleans up test database
func (h *DatabaseHelper) CleanupTestDatabase(t *testing.T) {
	t.Helper()
	
	// Clean up test data but preserve schema for other tests
	h.cleanTestData(t)
}

// CreateTestTransaction creates a test transaction that can be rolled back
func (h *DatabaseHelper) CreateTestTransaction(t *testing.T) *sql.Tx {
	t.Helper()
	
	tx, err := h.DB.Begin()
	require.NoError(t, err, "Failed to begin transaction")
	
	return tx
}

// RollbackTransaction rolls back a test transaction
func (h *DatabaseHelper) RollbackTransaction(t *testing.T, tx *sql.Tx) {
	t.Helper()
	
	if tx != nil {
		err := tx.Rollback()
		if err != nil && err != sql.ErrTxDone {
			t.Errorf("Failed to rollback transaction: %v", err)
		}
	}
}

// ensureTestDatabaseExists creates the test database if it doesn't exist
func (h *DatabaseHelper) ensureTestDatabaseExists(t *testing.T) {
	t.Helper()
	
	// Check if database exists
	var exists bool
	err := h.DB.QueryRow(`
		SELECT EXISTS (
			SELECT FROM pg_database 
			WHERE datname = $1
		)
	`, h.TestDB).Scan(&exists)
	
	require.NoError(t, err, "Failed to check database existence")
	
	if !exists {
		t.Logf("Creating test database: %s", h.TestDB)
		_, err := h.DB.Exec(fmt.Sprintf("CREATE DATABASE %s", h.TestDB))
		require.NoError(t, err, "Failed to create test database")
	}
}

// runMigrations runs database migrations for tests
func (h *DatabaseHelper) runMigrations(t *testing.T) {
	t.Helper()
	
	// Create basic tables needed for tests
	migrations := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id SERIAL PRIMARY KEY,
			email VARCHAR(255) UNIQUE NOT NULL,
			password_hash VARCHAR(255) NOT NULL,
			name VARCHAR(255),
			role VARCHAR(50) DEFAULT 'user',
			tenant_id VARCHAR(255),
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		
		`CREATE TABLE IF NOT EXISTS vms (
			id VARCHAR(255) PRIMARY KEY,
			name VARCHAR(255) NOT NULL,
			state VARCHAR(50) NOT NULL DEFAULT 'created',
			cpu INTEGER NOT NULL DEFAULT 1,
			memory INTEGER NOT NULL DEFAULT 1024,
			disk_size INTEGER NOT NULL DEFAULT 10240,
			image VARCHAR(255),
			node_id VARCHAR(255),
			tenant_id VARCHAR(255),
			user_id INTEGER REFERENCES users(id),
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		
		`CREATE TABLE IF NOT EXISTS storage_tiers (
			id SERIAL PRIMARY KEY,
			name VARCHAR(255) UNIQUE NOT NULL,
			storage_class VARCHAR(100) NOT NULL,
			performance_tier VARCHAR(50) NOT NULL,
			cost_per_gb_month DECIMAL(10,4),
			iops INTEGER,
			throughput_mbps INTEGER,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		
		`CREATE TABLE IF NOT EXISTS quotas (
			id SERIAL PRIMARY KEY,
			tenant_id VARCHAR(255) NOT NULL,
			resource_type VARCHAR(100) NOT NULL,
			limit_value INTEGER NOT NULL,
			current_usage INTEGER DEFAULT 0,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(tenant_id, resource_type)
		)`,
		
		`CREATE TABLE IF NOT EXISTS audit_logs (
			id SERIAL PRIMARY KEY,
			user_id INTEGER,
			tenant_id VARCHAR(255),
			action VARCHAR(100) NOT NULL,
			resource_type VARCHAR(100),
			resource_id VARCHAR(255),
			details JSONB,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		
		`CREATE TABLE IF NOT EXISTS raft_logs (
			id SERIAL PRIMARY KEY,
			term INTEGER NOT NULL,
			index INTEGER NOT NULL,
			log_type VARCHAR(50) NOT NULL,
			data JSONB,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			UNIQUE(term, index)
		)`,
		
		`CREATE INDEX IF NOT EXISTS idx_vms_tenant_id ON vms(tenant_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vms_state ON vms(state)`,
		`CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id)`,
		`CREATE INDEX IF NOT EXISTS idx_quotas_tenant_id ON quotas(tenant_id)`,
		`CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)`,
		`CREATE INDEX IF NOT EXISTS idx_raft_logs_term_index ON raft_logs(term, index)`,
	}
	
	for _, migration := range migrations {
		_, err := h.DB.Exec(migration)
		require.NoError(t, err, "Failed to run migration: %s", migration)
	}
	
	t.Log("Database migrations completed successfully")
}

// seedTestData inserts test data for integration tests
func (h *DatabaseHelper) seedTestData(t *testing.T) {
	t.Helper()
	
	// Insert test users
	testUsers := []map[string]interface{}{
		{
			"email": "admin@test.com",
			"password_hash": "$2a$10$N9qo8uLOickgx2ZMRZoMye1jzHD.djNy...", // hash of "admin123"
			"name": "Test Admin",
			"role": "admin",
			"tenant_id": "tenant-1",
		},
		{
			"email": "user@test.com", 
			"password_hash": "$2a$10$N9qo8uLOickgx2ZMRZoMye2jzHD.djNy...", // hash of "user123"
			"name": "Test User",
			"role": "user",
			"tenant_id": "tenant-1",
		},
		{
			"email": "tenant2@test.com",
			"password_hash": "$2a$10$N9qo8uLOickgx2ZMRZoMye3jzHD.djNy...", // hash of "tenant123"
			"name": "Tenant 2 User",
			"role": "user", 
			"tenant_id": "tenant-2",
		},
	}
	
	for _, user := range testUsers {
		_, err := h.DB.Exec(`
			INSERT INTO users (email, password_hash, name, role, tenant_id)
			VALUES ($1, $2, $3, $4, $5)
			ON CONFLICT (email) DO UPDATE SET
				password_hash = EXCLUDED.password_hash,
				name = EXCLUDED.name,
				role = EXCLUDED.role,
				tenant_id = EXCLUDED.tenant_id
		`, user["email"], user["password_hash"], user["name"], user["role"], user["tenant_id"])
		require.NoError(t, err, "Failed to insert test user")
	}
	
	// Insert test storage tiers
	storageTiers := []map[string]interface{}{
		{
			"name": "standard",
			"storage_class": "HDD",
			"performance_tier": "standard",
			"cost_per_gb_month": 0.045,
			"iops": 100,
			"throughput_mbps": 125,
		},
		{
			"name": "premium",
			"storage_class": "SSD",
			"performance_tier": "high",
			"cost_per_gb_month": 0.125,
			"iops": 500,
			"throughput_mbps": 500,
		},
		{
			"name": "ultra",
			"storage_class": "NVMe",
			"performance_tier": "ultra",
			"cost_per_gb_month": 0.300,
			"iops": 2000,
			"throughput_mbps": 1000,
		},
	}
	
	for _, tier := range storageTiers {
		_, err := h.DB.Exec(`
			INSERT INTO storage_tiers (name, storage_class, performance_tier, cost_per_gb_month, iops, throughput_mbps)
			VALUES ($1, $2, $3, $4, $5, $6)
			ON CONFLICT (name) DO UPDATE SET
				storage_class = EXCLUDED.storage_class,
				performance_tier = EXCLUDED.performance_tier,
				cost_per_gb_month = EXCLUDED.cost_per_gb_month,
				iops = EXCLUDED.iops,
				throughput_mbps = EXCLUDED.throughput_mbps
		`, tier["name"], tier["storage_class"], tier["performance_tier"], tier["cost_per_gb_month"], tier["iops"], tier["throughput_mbps"])
		require.NoError(t, err, "Failed to insert storage tier")
	}
	
	// Insert test quotas
	quotas := []map[string]interface{}{
		{"tenant_id": "tenant-1", "resource_type": "cpu", "limit_value": 100, "current_usage": 10},
		{"tenant_id": "tenant-1", "resource_type": "memory", "limit_value": 102400, "current_usage": 8192},
		{"tenant_id": "tenant-1", "resource_type": "storage", "limit_value": 1000000, "current_usage": 50000},
		{"tenant_id": "tenant-1", "resource_type": "vms", "limit_value": 50, "current_usage": 5},
		{"tenant_id": "tenant-2", "resource_type": "cpu", "limit_value": 50, "current_usage": 5},
		{"tenant_id": "tenant-2", "resource_type": "memory", "limit_value": 51200, "current_usage": 4096},
		{"tenant_id": "tenant-2", "resource_type": "storage", "limit_value": 500000, "current_usage": 25000},
		{"tenant_id": "tenant-2", "resource_type": "vms", "limit_value": 25, "current_usage": 3},
	}
	
	for _, quota := range quotas {
		_, err := h.DB.Exec(`
			INSERT INTO quotas (tenant_id, resource_type, limit_value, current_usage)
			VALUES ($1, $2, $3, $4)
			ON CONFLICT (tenant_id, resource_type) DO UPDATE SET
				limit_value = EXCLUDED.limit_value,
				current_usage = EXCLUDED.current_usage
		`, quota["tenant_id"], quota["resource_type"], quota["limit_value"], quota["current_usage"])
		require.NoError(t, err, "Failed to insert quota")
	}
	
	t.Log("Test data seeded successfully")
}

// cleanTestData removes test data while preserving schema
func (h *DatabaseHelper) cleanTestData(t *testing.T) {
	t.Helper()
	
	tables := []string{
		"audit_logs",
		"raft_logs", 
		"vms",
		"quotas",
		"users",
	}
	
	for _, table := range tables {
		_, err := h.DB.Exec(fmt.Sprintf("TRUNCATE TABLE %s RESTART IDENTITY CASCADE", table))
		if err != nil {
			t.Logf("Warning: Failed to truncate table %s: %v", table, err)
		}
	}
}

// WaitForDatabase waits for database to become available
func WaitForDatabase(dbURL string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	
	for time.Now().Before(deadline) {
		db, err := sql.Open("postgres", dbURL)
		if err != nil {
			time.Sleep(1 * time.Second)
			continue
		}
		
		err = db.Ping()
		db.Close()
		
		if err == nil {
			return nil
		}
		
		time.Sleep(1 * time.Second)
	}
	
	return fmt.Errorf("database not available after %v", timeout)
}