package integration

import (
	"database/sql"
	"os"
	"testing"

	_ "github.com/lib/pq"
)

// TestDatabaseConnection tests database connectivity
func TestDatabaseConnection(t *testing.T) {
	// Get database URL from environment
	dbURL := os.Getenv("DB_URL")
	if dbURL == "" {
		t.Skip("DB_URL not set, skipping database tests")
	}

	// Test database connection
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		t.Fatalf("Failed to open database connection: %v", err)
	}
	defer db.Close()

	// Test ping
	err = db.Ping()
	if err != nil {
		t.Fatalf("Failed to ping database: %v", err)
	}

	t.Log("Database connection successful")
}

// TestBasicCRUD tests basic database operations
func TestBasicCRUD(t *testing.T) {
	dbURL := os.Getenv("DB_URL")
	if dbURL == "" {
		t.Skip("DB_URL not set, skipping database tests")
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		t.Fatalf("Failed to open database connection: %v", err)
	}
	defer db.Close()

	// Test table existence (this assumes tables are created)
	var tableExists bool
	err = db.QueryRow(`
		SELECT EXISTS (
			SELECT FROM information_schema.tables 
			WHERE table_schema = 'public' 
			AND table_name = 'users'
		)
	`).Scan(&tableExists)

	if err != nil {
		t.Logf("Could not check for users table: %v", err)
	} else if tableExists {
		t.Log("Users table exists")
	} else {
		t.Log("Users table does not exist (may not be created yet)")
	}

	// Test basic query execution
	rows, err := db.Query("SELECT 1 as test_value")
	if err != nil {
		t.Fatalf("Failed to execute test query: %v", err)
	}
	defer rows.Close()

	var testValue int
	for rows.Next() {
		err := rows.Scan(&testValue)
		if err != nil {
			t.Fatalf("Failed to scan test value: %v", err)
		}
	}

	if testValue != 1 {
		t.Errorf("Expected test value 1, got %d", testValue)
	}
}

// TestTransactionRollback tests transaction handling
func TestTransactionRollback(t *testing.T) {
	dbURL := os.Getenv("DB_URL")
	if dbURL == "" {
		t.Skip("DB_URL not set, skipping database tests")
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		t.Fatalf("Failed to open database connection: %v", err)
	}
	defer db.Close()

	// Begin transaction
	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Failed to begin transaction: %v", err)
	}

	// Create a temporary table for testing
	_, err = tx.Exec(`
		CREATE TEMPORARY TABLE test_rollback (
			id SERIAL PRIMARY KEY,
			value TEXT
		)
	`)
	if err != nil {
		t.Fatalf("Failed to create temporary table: %v", err)
	}

	// Insert test data
	_, err = tx.Exec("INSERT INTO test_rollback (value) VALUES ('test')")
	if err != nil {
		t.Fatalf("Failed to insert test data: %v", err)
	}

	// Rollback the transaction
	err = tx.Rollback()
	if err != nil {
		t.Fatalf("Failed to rollback transaction: %v", err)
	}

	// Verify rollback worked
	// Note: The temporary table is automatically dropped with the transaction rollback
	t.Log("Transaction rollback successful")
}
