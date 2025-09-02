package integration

import (
	"context"
	"database/sql"
	"fmt"
	"testing"
	"time"

	"github.com/lib/pq"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	
	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// DatabaseValidationSuite validates database integration and data persistence
type DatabaseValidationSuite struct {
	suite.Suite
	db          *sql.DB
	authManager *auth.SimpleAuthManager
	testData    map[string]interface{}
}

// SetupSuite initializes database connection
func (suite *DatabaseValidationSuite) SetupSuite() {
	// Database connection parameters - would be configured from environment
	dbConfig := map[string]string{
		"host":     "localhost",
		"port":     "5432",
		"user":     "novacron_user",
		"password": "novacron_pass",
		"dbname":   "novacron_db",
		"sslmode":  "disable",
	}

	// Build connection string
	connStr := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
		dbConfig["host"], dbConfig["port"], dbConfig["user"], 
		dbConfig["password"], dbConfig["dbname"], dbConfig["sslmode"])

	// Connect to database
	db, err := sql.Open("postgres", connStr)
	if err != nil {
		suite.T().Skip("Database not available for testing - skipping database validation")
		return
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		suite.T().Skip("Database connection failed - skipping database validation: " + err.Error())
		return
	}

	suite.db = db
	suite.authManager = auth.NewSimpleAuthManager("test-secret", db)
	suite.testData = make(map[string]interface{})

	// Run database setup
	suite.setupTestSchema()
}

// TearDownSuite cleans up database connection
func (suite *DatabaseValidationSuite) TearDownSuite() {
	if suite.db != nil {
		suite.cleanupTestData()
		suite.db.Close()
	}
}

// Test 1: Database Connectivity and Health
func (suite *DatabaseValidationSuite) TestDatabaseConnectivity() {
	if suite.db == nil {
		suite.T().Skip("Database not available")
		return
	}

	// Test basic connectivity
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err := suite.db.PingContext(ctx)
	assert.NoError(suite.T(), err, "Database should be reachable")

	// Test connection pooling
	stats := suite.db.Stats()
	assert.GreaterOrEqual(suite.T(), stats.MaxOpenConnections, 1, "Connection pool should be configured")

	// Test query execution
	var version string
	err = suite.db.QueryRowContext(ctx, "SELECT version()").Scan(&version)
	assert.NoError(suite.T(), err, "Should be able to query database version")
	assert.Contains(suite.T(), version, "PostgreSQL", "Should be using PostgreSQL")
}

// Test 2: Schema Validation
func (suite *DatabaseValidationSuite) TestSchemaValidation() {
	if suite.db == nil {
		suite.T().Skip("Database not available")
		return
	}

	requiredTables := []struct {
		name     string
		required bool
	}{
		{"users", true},
		{"vms", true},
		{"vm_metrics", true},
		{"sessions", false},
		{"audit_logs", false},
	}

	for _, table := range requiredTables {
		suite.T().Run(fmt.Sprintf("Table_%s", table.name), func(t *testing.T) {
			var exists bool
			query := `
				SELECT EXISTS (
					SELECT FROM information_schema.tables 
					WHERE table_schema = 'public' AND table_name = $1
				)`
			
			err := suite.db.QueryRow(query, table.name).Scan(&exists)
			require.NoError(t, err)

			if table.required {
				assert.True(t, exists, "Required table %s should exist", table.name)
			}

			if exists {
				// Validate table structure
				suite.validateTableStructure(t, table.name)
			}
		})
	}
}

// Test 3: User Management and Authentication Data
func (suite *DatabaseValidationSuite) TestUserDataPersistence() {
	if suite.db == nil || suite.authManager == nil {
		suite.T().Skip("Database not available")
		return
	}

	// Test user creation
	suite.T().Run("CreateUser", func(t *testing.T) {
		username := fmt.Sprintf("test_user_%d", time.Now().Unix())
		email := fmt.Sprintf("%s@example.com", username)
		password := "test_password_123"

		user, err := suite.authManager.CreateUser(username, email, password, "user", "default")
		require.NoError(t, err, "Should be able to create user")
		require.NotNil(t, user, "Created user should not be nil")

		assert.Equal(t, username, user.Username)
		assert.Equal(t, email, user.Email)
		assert.NotEmpty(t, user.ID)
		assert.NotEmpty(t, user.PasswordHash)

		// Store for later tests
		suite.testData["test_user_id"] = user.ID
		suite.testData["test_username"] = username
		suite.testData["test_password"] = password
	})

	// Test user retrieval
	suite.T().Run("RetrieveUser", func(t *testing.T) {
		if userID, ok := suite.testData["test_user_id"]; ok {
			user, err := suite.authManager.GetUser(userID.(string))
			require.NoError(t, err, "Should be able to retrieve user")
			require.NotNil(t, user, "Retrieved user should not be nil")

			assert.Equal(t, userID.(string), user.ID)
			assert.Equal(t, suite.testData["test_username"], user.Username)
		} else {
			t.Skip("No test user ID available")
		}
	})

	// Test user authentication
	suite.T().Run("AuthenticateUser", func(t *testing.T) {
		if username, ok := suite.testData["test_username"]; ok {
			if password, ok := suite.testData["test_password"]; ok {
				user, token, err := suite.authManager.Authenticate(username.(string), password.(string))
				require.NoError(t, err, "Should be able to authenticate user")
				require.NotNil(t, user, "Authenticated user should not be nil")
				require.NotEmpty(t, token, "JWT token should be generated")

				assert.Equal(t, username.(string), user.Username)
				assert.Contains(t, token, ".", "JWT token should contain dots")

				// Store token for later tests
				suite.testData["test_token"] = token
			}
		} else {
			t.Skip("No test user credentials available")
		}
	})
}

// Test 4: VM Data Persistence
func (suite *DatabaseValidationSuite) TestVMDataPersistence() {
	if suite.db == nil {
		suite.T().Skip("Database not available")
		return
	}

	// Test VM creation
	suite.T().Run("CreateVM", func(t *testing.T) {
		vmID := fmt.Sprintf("test-vm-%d", time.Now().Unix())
		vmName := fmt.Sprintf("Test VM %d", time.Now().Unix())
		
		query := `
			INSERT INTO vms (id, name, state, node_id, config, created_at, updated_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7)`
		
		config := `{"cpu": 2, "memory": 4096, "disk": 20480}`
		now := time.Now()

		_, err := suite.db.Exec(query, vmID, vmName, "creating", "node-1", config, now, now)
		require.NoError(t, err, "Should be able to insert VM record")

		// Store for later tests
		suite.testData["test_vm_id"] = vmID
	})

	// Test VM retrieval
	suite.T().Run("RetrieveVM", func(t *testing.T) {
		if vmID, ok := suite.testData["test_vm_id"]; ok {
			var id, name, state, nodeID string
			var config string
			var createdAt, updatedAt time.Time

			query := `
				SELECT id, name, state, node_id, config, created_at, updated_at
				FROM vms WHERE id = $1`

			err := suite.db.QueryRow(query, vmID).Scan(
				&id, &name, &state, &nodeID, &config, &createdAt, &updatedAt)
			
			require.NoError(t, err, "Should be able to retrieve VM record")
			assert.Equal(t, vmID.(string), id)
			assert.NotEmpty(t, name)
			assert.NotEmpty(t, state)
		} else {
			t.Skip("No test VM ID available")
		}
	})

	// Test VM state update
	suite.T().Run("UpdateVMState", func(t *testing.T) {
		if vmID, ok := suite.testData["test_vm_id"]; ok {
			query := `UPDATE vms SET state = $1, updated_at = $2 WHERE id = $3`
			
			_, err := suite.db.Exec(query, "running", time.Now(), vmID)
			require.NoError(t, err, "Should be able to update VM state")

			// Verify update
			var state string
			err = suite.db.QueryRow("SELECT state FROM vms WHERE id = $1", vmID).Scan(&state)
			require.NoError(t, err)
			assert.Equal(t, "running", state)
		} else {
			t.Skip("No test VM ID available")
		}
	})
}

// Test 5: Metrics Data Persistence
func (suite *DatabaseValidationSuite) TestMetricsDataPersistence() {
	if suite.db == nil {
		suite.T().Skip("Database not available")
		return
	}

	// Test metrics insertion
	suite.T().Run("InsertMetrics", func(t *testing.T) {
		if vmID, ok := suite.testData["test_vm_id"]; ok {
			query := `
				INSERT INTO vm_metrics (vm_id, cpu_usage, memory_usage, network_sent, network_recv, timestamp)
				VALUES ($1, $2, $3, $4, $5, $6)`

			metrics := []struct {
				cpu, memory float64
				netSent, netRecv int64
			}{
				{45.5, 2048.0, 1024000, 512000},
				{52.3, 2156.8, 1124000, 542000},
				{48.1, 2089.4, 1098000, 531000},
			}

			for i, metric := range metrics {
				timestamp := time.Now().Add(time.Duration(-i) * time.Minute)
				_, err := suite.db.Exec(query, vmID, metric.cpu, metric.memory, 
					metric.netSent, metric.netRecv, timestamp)
				require.NoError(t, err, "Should be able to insert metrics record")
			}

			suite.testData["metrics_count"] = len(metrics)
		} else {
			t.Skip("No test VM ID available")
		}
	})

	// Test metrics retrieval
	suite.T().Run("RetrieveMetrics", func(t *testing.T) {
		if vmID, ok := suite.testData["test_vm_id"]; ok {
			query := `
				SELECT COUNT(*), AVG(cpu_usage), AVG(memory_usage)
				FROM vm_metrics WHERE vm_id = $1`

			var count int
			var avgCPU, avgMemory float64

			err := suite.db.QueryRow(query, vmID).Scan(&count, &avgCPU, &avgMemory)
			require.NoError(t, err, "Should be able to retrieve metrics")

			expectedCount := suite.testData["metrics_count"].(int)
			assert.Equal(t, expectedCount, count)
			assert.Greater(t, avgCPU, 0.0)
			assert.Greater(t, avgMemory, 0.0)
		} else {
			t.Skip("No test VM ID available")
		}
	})

	// Test metrics time-based queries
	suite.T().Run("TimeBasedMetrics", func(t *testing.T) {
		if vmID, ok := suite.testData["test_vm_id"]; ok {
			query := `
				SELECT COUNT(*)
				FROM vm_metrics 
				WHERE vm_id = $1 AND timestamp >= $2`

			oneHourAgo := time.Now().Add(-1 * time.Hour)
			
			var count int
			err := suite.db.QueryRow(query, vmID, oneHourAgo).Scan(&count)
			require.NoError(t, err, "Should be able to query metrics by time")
			assert.GreaterOrEqual(t, count, 0)
		} else {
			t.Skip("No test VM ID available")
		}
	})
}

// Test 6: Transaction Support
func (suite *DatabaseValidationSuite) TestTransactionSupport() {
	if suite.db == nil {
		suite.T().Skip("Database not available")
		return
	}

	suite.T().Run("TransactionCommit", func(t *testing.T) {
		tx, err := suite.db.Begin()
		require.NoError(t, err, "Should be able to begin transaction")

		testID := fmt.Sprintf("tx_test_%d", time.Now().Unix())
		
		// Insert user in transaction
		_, err = tx.Exec(`
			INSERT INTO users (username, email, password_hash, role, tenant_id)
			VALUES ($1, $2, $3, $4, $5)`,
			testID, testID+"@example.com", "hash", "user", "default")
		require.NoError(t, err, "Should be able to insert in transaction")

		err = tx.Commit()
		require.NoError(t, err, "Should be able to commit transaction")

		// Verify data exists
		var count int
		err = suite.db.QueryRow("SELECT COUNT(*) FROM users WHERE username = $1", testID).Scan(&count)
		require.NoError(t, err)
		assert.Equal(t, 1, count, "Committed data should exist")
	})

	suite.T().Run("TransactionRollback", func(t *testing.T) {
		tx, err := suite.db.Begin()
		require.NoError(t, err, "Should be able to begin transaction")

		testID := fmt.Sprintf("rollback_test_%d", time.Now().Unix())
		
		// Insert user in transaction
		_, err = tx.Exec(`
			INSERT INTO users (username, email, password_hash, role, tenant_id)
			VALUES ($1, $2, $3, $4, $5)`,
			testID, testID+"@example.com", "hash", "user", "default")
		require.NoError(t, err, "Should be able to insert in transaction")

		err = tx.Rollback()
		require.NoError(t, err, "Should be able to rollback transaction")

		// Verify data does not exist
		var count int
		err = suite.db.QueryRow("SELECT COUNT(*) FROM users WHERE username = $1", testID).Scan(&count)
		require.NoError(t, err)
		assert.Equal(t, 0, count, "Rolled back data should not exist")
	})
}

// Test 7: Performance and Concurrency
func (suite *DatabaseValidationSuite) TestDatabasePerformance() {
	if suite.db == nil {
		suite.T().Skip("Database not available")
		return
	}

	suite.T().Run("ConcurrentConnections", func(t *testing.T) {
		concurrency := 10
		done := make(chan bool, concurrency)

		for i := 0; i < concurrency; i++ {
			go func(id int) {
				defer func() { done <- true }()

				// Each goroutine performs a database operation
				var result int
				err := suite.db.QueryRow("SELECT $1", id).Scan(&result)
				assert.NoError(t, err, "Concurrent query should succeed")
				assert.Equal(t, id, result, "Concurrent query should return correct result")
			}(i)
		}

		// Wait for all goroutines to complete
		for i := 0; i < concurrency; i++ {
			select {
			case <-done:
				// Continue
			case <-time.After(10 * time.Second):
				t.Fatal("Concurrent operations timed out")
			}
		}
	})

	suite.T().Run("BulkOperations", func(t *testing.T) {
		// Test bulk insert performance
		start := time.Now()
		
		tx, err := suite.db.Begin()
		require.NoError(t, err)

		stmt, err := tx.Prepare(`
			INSERT INTO vm_metrics (vm_id, cpu_usage, memory_usage, network_sent, network_recv, timestamp)
			VALUES ($1, $2, $3, $4, $5, $6)`)
		require.NoError(t, err)
		defer stmt.Close()

		bulkSize := 100
		vmID := "bulk-test-vm"

		for i := 0; i < bulkSize; i++ {
			_, err = stmt.Exec(vmID, 
				float64(i%100), float64((i%4096)+1000),
				int64(i*1000), int64(i*500),
				time.Now().Add(time.Duration(i)*time.Second))
			require.NoError(t, err)
		}

		err = tx.Commit()
		require.NoError(t, err)

		duration := time.Since(start)
		
		// Performance assertion - should complete bulk insert reasonably quickly
		assert.Less(t, duration, 5*time.Second, "Bulk operations should complete within reasonable time")

		// Cleanup
		_, err = suite.db.Exec("DELETE FROM vm_metrics WHERE vm_id = $1", vmID)
		require.NoError(t, err)
	})
}

// Helper methods

func (suite *DatabaseValidationSuite) setupTestSchema() {
	if suite.db == nil {
		return
	}

	// Run basic schema creation (if needed)
	migrations := []string{
		`CREATE TABLE IF NOT EXISTS users (
			id SERIAL PRIMARY KEY,
			username VARCHAR(255) UNIQUE NOT NULL,
			email VARCHAR(255) UNIQUE NOT NULL,
			password_hash VARCHAR(255) NOT NULL,
			role VARCHAR(50) DEFAULT 'user',
			tenant_id VARCHAR(255) DEFAULT 'default',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS vms (
			id VARCHAR(255) PRIMARY KEY,
			name VARCHAR(255) NOT NULL,
			state VARCHAR(50) NOT NULL,
			node_id VARCHAR(255),
			config JSONB,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS vm_metrics (
			id SERIAL PRIMARY KEY,
			vm_id VARCHAR(255) NOT NULL,
			cpu_usage FLOAT,
			memory_usage FLOAT,
			network_sent BIGINT,
			network_recv BIGINT,
			timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id ON vm_metrics(vm_id)`,
		`CREATE INDEX IF NOT EXISTS idx_vm_metrics_timestamp ON vm_metrics(timestamp)`,
	}

	for _, migration := range migrations {
		if _, err := suite.db.Exec(migration); err != nil {
			// Log but don't fail - schema might already exist
			suite.T().Logf("Migration warning: %v", err)
		}
	}
}

func (suite *DatabaseValidationSuite) validateTableStructure(t *testing.T, tableName string) {
	query := `
		SELECT column_name, data_type, is_nullable, column_default
		FROM information_schema.columns
		WHERE table_schema = 'public' AND table_name = $1
		ORDER BY ordinal_position`

	rows, err := suite.db.Query(query, tableName)
	require.NoError(t, err)
	defer rows.Close()

	columnCount := 0
	for rows.Next() {
		var columnName, dataType, isNullable string
		var columnDefault sql.NullString

		err := rows.Scan(&columnName, &dataType, &isNullable, &columnDefault)
		require.NoError(t, err)

		columnCount++
		
		// Basic validation - all tables should have some columns
		assert.NotEmpty(t, columnName, "Column name should not be empty")
		assert.NotEmpty(t, dataType, "Data type should not be empty")
	}

	require.NoError(t, rows.Err())
	assert.Greater(t, columnCount, 0, "Table %s should have columns", tableName)
}

func (suite *DatabaseValidationSuite) cleanupTestData() {
	if suite.db == nil {
		return
	}

	// Clean up test data
	cleanupQueries := []string{
		"DELETE FROM vm_metrics WHERE vm_id LIKE 'test-%' OR vm_id LIKE 'bulk-test-%'",
		"DELETE FROM vms WHERE id LIKE 'test-vm-%'",
		"DELETE FROM users WHERE username LIKE 'test_user_%' OR username LIKE 'tx_test_%' OR username LIKE 'rollback_test_%'",
	}

	for _, query := range cleanupQueries {
		if _, err := suite.db.Exec(query); err != nil {
			suite.T().Logf("Cleanup warning: %v", err)
		}
	}
}

// Run the database validation suite
func TestDatabaseValidationSuite(t *testing.T) {
	suite.Run(t, new(DatabaseValidationSuite))
}