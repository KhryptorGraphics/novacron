package integration

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"github.com/lib/pq"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
	"github.com/khryptorgraphics/novacron/backend/tests/integration/fixtures"
	"github.com/khryptorgraphics/novacron/backend/tests/integration/helpers"
)

// IntegrationTestSuite provides a comprehensive test suite for NovaCron
type IntegrationTestSuite struct {
	suite.Suite
	
	// Core components
	config      *config.Config
	db          *sql.DB
	router      *mux.Router
	server      *httptest.Server
	
	// Managers
	authManager *auth.SimpleAuthManager
	vmManager   *core_vm.VMManager
	kvmManager  *hypervisor.KVMManager
	
	// Test helpers
	fixtures    *fixtures.TestFixtures
	helpers     *helpers.TestHelpers
	
	// Context and cleanup
	ctx         context.Context
	cancel      context.CancelFunc
}

// SetupSuite runs before all tests in the suite
func (suite *IntegrationTestSuite) SetupSuite() {
	suite.T().Log("Setting up integration test suite...")
	
	// Create context for test suite
	suite.ctx, suite.cancel = context.WithCancel(context.Background())
	
	// Load test configuration
	suite.config = suite.loadTestConfig()
	
	// Initialize logger with debug level for tests
	appLogger := logger.NewFromConfig("debug", "text", "stdout", true)
	logger.SetGlobalLogger(appLogger)
	
	// Initialize test database
	suite.initTestDatabase()
	
	// Initialize managers
	suite.initManagers()
	
	// Setup test fixtures and helpers
	suite.fixtures = fixtures.New(suite.db, suite.authManager, suite.vmManager)
	suite.helpers = helpers.New(suite.db, suite.config)
	
	// Setup test server
	suite.setupTestServer()
	
	suite.T().Log("Integration test suite setup completed")
}

// TearDownSuite runs after all tests in the suite
func (suite *IntegrationTestSuite) TearDownSuite() {
	suite.T().Log("Tearing down integration test suite...")
	
	if suite.server != nil {
		suite.server.Close()
	}
	
	if suite.cancel != nil {
		suite.cancel()
	}
	
	if suite.vmManager != nil {
		suite.vmManager.Close()
	}
	
	if suite.kvmManager != nil {
		suite.kvmManager.Close()
	}
	
	if suite.fixtures != nil {
		suite.fixtures.Cleanup()
	}
	
	if suite.db != nil {
		suite.helpers.CleanupDatabase()
		suite.db.Close()
	}
	
	suite.T().Log("Integration test suite teardown completed")
}

// SetupTest runs before each test
func (suite *IntegrationTestSuite) SetupTest() {
	// Clean database state for each test
	suite.helpers.CleanupTestData()
}

// TearDownTest runs after each test
func (suite *IntegrationTestSuite) TearDownTest() {
	// Cleanup any test-specific resources
	suite.helpers.CleanupTestData()
}

// loadTestConfig loads configuration for testing
func (suite *IntegrationTestSuite) loadTestConfig() *config.Config {
	// Set test environment variables
	os.Setenv("DB_URL", suite.getTestDatabaseURL())
	os.Setenv("JWT_SECRET", "test-jwt-secret-key-for-integration-tests")
	os.Setenv("LOG_LEVEL", "debug")
	
	cfg, err := config.Load()
	if err != nil {
		// Fallback to default test config
		cfg = &config.Config{
			Database: config.DatabaseConfig{
				URL:             suite.getTestDatabaseURL(),
				MaxConnections:  10,
				ConnMaxLifetime: 30 * time.Minute,
				ConnMaxIdleTime: 5 * time.Minute,
			},
			Auth: config.AuthConfig{
				Secret: "test-jwt-secret-key-for-integration-tests",
			},
			Server: config.ServerConfig{
				APIPort:         "0", // Use random port for testing
				WSPort:          "0",
				ReadTimeout:     30 * time.Second,
				WriteTimeout:    30 * time.Second,
				IdleTimeout:     60 * time.Second,
				ShutdownTimeout: 30 * time.Second,
			},
			VM: config.VMConfig{
				StoragePath: "/tmp/novacron-test-storage",
			},
			Logging: config.LoggingConfig{
				Level:      "debug",
				Format:     "text",
				Output:     "stdout",
				Structured: true,
			},
		}
	}
	
	return cfg
}

// getTestDatabaseURL returns the test database URL
func (suite *IntegrationTestSuite) getTestDatabaseURL() string {
	testDB := os.Getenv("TEST_DB_URL")
	if testDB != "" {
		return testDB
	}
	
	// Default test database URL
	return "postgres://postgres:password@localhost:5432/novacron_test?sslmode=disable"
}

// initTestDatabase initializes the test database connection
func (suite *IntegrationTestSuite) initTestDatabase() {
	var err error
	suite.db, err = sql.Open("postgres", suite.config.Database.URL)
	suite.Require().NoError(err, "Failed to connect to test database")
	
	// Configure connection pool for testing
	suite.db.SetMaxOpenConns(suite.config.Database.MaxConnections)
	suite.db.SetMaxIdleConns(5)
	suite.db.SetConnMaxLifetime(suite.config.Database.ConnMaxLifetime)
	suite.db.SetConnMaxIdleTime(suite.config.Database.ConnMaxIdleTime)
	
	// Test connection
	ctx, cancel := context.WithTimeout(suite.ctx, 10*time.Second)
	defer cancel()
	
	err = suite.db.PingContext(ctx)
	suite.Require().NoError(err, "Failed to ping test database")
	
	// Run test database migrations
	suite.runTestMigrations()
	
	suite.T().Log("Test database initialized successfully")
}

// runTestMigrations runs database migrations for testing
func (suite *IntegrationTestSuite) runTestMigrations() {
	migrations := []string{
		// Users table
		`DROP TABLE IF EXISTS vm_metrics CASCADE`,
		`DROP TABLE IF EXISTS vms CASCADE`, 
		`DROP TABLE IF EXISTS users CASCADE`,
		
		`CREATE TABLE users (
			id SERIAL PRIMARY KEY,
			username VARCHAR(255) UNIQUE NOT NULL,
			email VARCHAR(255) UNIQUE NOT NULL,
			password_hash VARCHAR(255) NOT NULL,
			role VARCHAR(50) DEFAULT 'user',
			tenant_id VARCHAR(255) DEFAULT 'default',
			is_active BOOLEAN DEFAULT true,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		
		// VMs table
		`CREATE TABLE vms (
			id VARCHAR(255) PRIMARY KEY,
			name VARCHAR(255) NOT NULL,
			state VARCHAR(50) NOT NULL DEFAULT 'stopped',
			node_id VARCHAR(255),
			owner_id INTEGER REFERENCES users(id),
			tenant_id VARCHAR(255) DEFAULT 'default',
			cpu_cores INTEGER DEFAULT 1,
			memory_mb INTEGER DEFAULT 1024,
			disk_gb INTEGER DEFAULT 10,
			config JSONB,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		
		// VM metrics table
		`CREATE TABLE vm_metrics (
			id SERIAL PRIMARY KEY,
			vm_id VARCHAR(255) REFERENCES vms(id) ON DELETE CASCADE,
			cpu_usage FLOAT DEFAULT 0.0,
			memory_usage FLOAT DEFAULT 0.0,
			disk_usage FLOAT DEFAULT 0.0,
			network_sent BIGINT DEFAULT 0,
			network_recv BIGINT DEFAULT 0,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		
		// Indexes for performance
		`CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)`,
		`CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vms_owner_id ON vms(owner_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vms_tenant_id ON vms(tenant_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vms_state ON vms(state)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id ON vm_metrics(vm_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_metrics_timestamp ON vm_metrics(timestamp DESC)`,
		
		// Test data triggers for updated_at
		`CREATE OR REPLACE FUNCTION update_updated_at_column()
		RETURNS TRIGGER AS $$
		BEGIN
			NEW.updated_at = CURRENT_TIMESTAMP;
			RETURN NEW;
		END;
		$$ language 'plpgsql'`,
		
		`CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
			FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()`,
		
		`CREATE TRIGGER update_vms_updated_at BEFORE UPDATE ON vms 
			FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()`,
	}
	
	for _, migration := range migrations {
		_, err := suite.db.Exec(migration)
		suite.Require().NoError(err, "Migration failed: %s", migration)
	}
	
	suite.T().Log("Test database migrations completed successfully")
}

// initManagers initializes the service managers
func (suite *IntegrationTestSuite) initManagers() {
	// Initialize auth manager
	suite.authManager = auth.NewSimpleAuthManager(suite.config.Auth.Secret, suite.db)
	suite.Require().NotNil(suite.authManager, "Failed to create auth manager")
	
	// Initialize VM manager (with basic config for testing)
	vmConfig := core_vm.VMManagerConfig{
		DefaultDriver: core_vm.VMTypeKVM,
		Drivers: make(map[core_vm.VMType]core_vm.VMDriverConfigManager),
		Scheduler: core_vm.VMSchedulerConfig{
			Type:   "round-robin",
			Config: make(map[string]interface{}),
		},
	}
	
	// Enable KVM driver for testing (may not work in all environments)
	vmConfig.Drivers[core_vm.VMTypeKVM] = core_vm.VMDriverConfigManager{
		Enabled: true,
		Config:  make(map[string]interface{}),
	}
	
	var err error
	suite.vmManager, err = core_vm.NewVMManager(vmConfig)
	if err != nil {
		suite.T().Logf("Warning: Failed to initialize VM manager: %v", err)
		suite.vmManager = nil
	}
	
	// Initialize KVM manager (optional for testing)
	suite.kvmManager, err = hypervisor.NewKVMManager("test:///default")
	if err != nil {
		suite.T().Logf("Warning: Failed to initialize KVM manager: %v", err)
		suite.kvmManager = nil
	}
	
	suite.T().Log("Service managers initialized")
}

// setupTestServer creates the test HTTP server
func (suite *IntegrationTestSuite) setupTestServer() {
	suite.router = mux.NewRouter()
	
	// Register routes (this would typically be done by the main application)
	suite.registerTestRoutes()
	
	// Create test server
	suite.server = httptest.NewServer(suite.router)
	suite.Require().NotNil(suite.server, "Failed to create test server")
	
	suite.T().Logf("Test server started at: %s", suite.server.URL)
}

// registerTestRoutes registers routes for testing
func (suite *IntegrationTestSuite) registerTestRoutes() {
	// This would typically be moved to a separate router setup file
	// For now, we'll register basic routes for testing
	
	// Health check
	suite.router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"healthy","service":"novacron-test"}`))
	}).Methods("GET")
	
	// API info
	suite.router.HandleFunc("/api/info", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"name":"NovaCron Test API","version":"test"}`))
	}).Methods("GET")
}

// GetServer returns the test server URL
func (suite *IntegrationTestSuite) GetServer() *httptest.Server {
	return suite.server
}

// GetDatabase returns the test database connection
func (suite *IntegrationTestSuite) GetDatabase() *sql.DB {
	return suite.db
}

// GetAuthManager returns the auth manager
func (suite *IntegrationTestSuite) GetAuthManager() *auth.SimpleAuthManager {
	return suite.authManager
}

// GetVMManager returns the VM manager
func (suite *IntegrationTestSuite) GetVMManager() *core_vm.VMManager {
	return suite.vmManager
}

// GetFixtures returns the test fixtures
func (suite *IntegrationTestSuite) GetFixtures() *fixtures.TestFixtures {
	return suite.fixtures
}

// GetHelpers returns the test helpers
func (suite *IntegrationTestSuite) GetHelpers() *helpers.TestHelpers {
	return suite.helpers
}

// TestIntegrationSuite runs the integration test suite
func TestIntegrationSuite(t *testing.T) {
	// Skip integration tests in short mode
	if testing.Short() {
		t.Skip("Skipping integration tests in short mode")
	}
	
	// Check if test database is available
	testDB := os.Getenv("TEST_DB_URL")
	if testDB == "" {
		testDB = "postgres://postgres:password@localhost:5432/novacron_test?sslmode=disable"
	}
	
	// Try to connect to test database
	db, err := sql.Open("postgres", testDB)
	if err != nil {
		t.Skipf("Skipping integration tests: cannot connect to test database: %v", err)
	}
	defer db.Close()
	
	if err := db.Ping(); err != nil {
		t.Skipf("Skipping integration tests: test database not available: %v", err)
	}
	
	suite.Run(t, new(IntegrationTestSuite))
}