package auth

import (
	"database/sql"
	"os"
	"testing"
	"time"

	_ "github.com/lib/pq"
)

// getTestDB returns a test database connection
// Set NOVACRON_TEST_DB_URL environment variable to run these tests
func getTestDB(t *testing.T) *sql.DB {
	dbURL := os.Getenv("NOVACRON_TEST_DB_URL")
	if dbURL == "" {
		// Default to local test database on non-standard port
		dbURL = "postgres://postgres:novacron_secure_pwd@localhost:15432/novacron_test?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		t.Skipf("Skipping test: cannot connect to database: %v", err)
		return nil
	}

	if err := db.Ping(); err != nil {
		t.Skipf("Skipping test: cannot ping database: %v", err)
		return nil
	}

	return db
}

func TestPostgresUserStore_Create(t *testing.T) {
	db := getTestDB(t)
	if db == nil {
		return
	}
	defer db.Close()

	store := NewPostgresUserStore(db)

	user := &User{
		Username:  "testuser_" + time.Now().Format("20060102150405"),
		Email:     "test_" + time.Now().Format("20060102150405") + "@example.com",
		FirstName: "Test",
		LastName:  "User",
		Status:    UserStatusActive,
		TenantID:  "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11", // Default test org
		RoleIDs:   []string{"user"},
	}

	err := store.Create(user, "password123")
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}

	if user.ID == "" {
		t.Error("User ID should be set after creation")
	}

	// Cleanup
	_ = store.Delete(user.ID)
}

func TestPostgresUserStore_Get(t *testing.T) {
	db := getTestDB(t)
	if db == nil {
		return
	}
	defer db.Close()

	store := NewPostgresUserStore(db)

	// Create a user first
	user := &User{
		Username:  "testuser_get_" + time.Now().Format("20060102150405"),
		Email:     "test_get_" + time.Now().Format("20060102150405") + "@example.com",
		FirstName: "Test",
		LastName:  "User",
		Status:    UserStatusActive,
		TenantID:  "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
		RoleIDs:   []string{"admin"},
	}

	err := store.Create(user, "password123")
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}
	defer store.Delete(user.ID)

	// Test Get
	retrieved, err := store.Get(user.ID)
	if err != nil {
		t.Fatalf("Failed to get user: %v", err)
	}

	if retrieved.Username != user.Username {
		t.Errorf("Username mismatch: expected %s, got %s", user.Username, retrieved.Username)
	}

	if retrieved.Email != user.Email {
		t.Errorf("Email mismatch: expected %s, got %s", user.Email, retrieved.Email)
	}
}

func TestPostgresUserStore_VerifyPassword(t *testing.T) {
	db := getTestDB(t)
	if db == nil {
		return
	}
	defer db.Close()

	store := NewPostgresUserStore(db)

	// Create a user
	user := &User{
		Username:  "testuser_pass_" + time.Now().Format("20060102150405"),
		Email:     "test_pass_" + time.Now().Format("20060102150405") + "@example.com",
		FirstName: "Test",
		LastName:  "User",
		Status:    UserStatusActive,
		TenantID:  "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
	}

	password := "securepassword123"
	err := store.Create(user, password)
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}
	defer store.Delete(user.ID)

	// Test correct password
	valid, err := store.VerifyPassword(user.ID, password)
	if err != nil {
		t.Fatalf("Failed to verify password: %v", err)
	}
	if !valid {
		t.Error("Password should be valid")
	}

	// Test incorrect password
	valid, err = store.VerifyPassword(user.ID, "wrongpassword")
	if err != nil {
		t.Fatalf("Failed to verify password: %v", err)
	}
	if valid {
		t.Error("Wrong password should not be valid")
	}
}

func TestPostgresUserStore_Update(t *testing.T) {
	db := getTestDB(t)
	if db == nil {
		return
	}
	defer db.Close()

	store := NewPostgresUserStore(db)

	// Create a user
	user := &User{
		Username:  "testuser_update_" + time.Now().Format("20060102150405"),
		Email:     "test_update_" + time.Now().Format("20060102150405") + "@example.com",
		FirstName: "Test",
		LastName:  "User",
		Status:    UserStatusActive,
		TenantID:  "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
	}

	err := store.Create(user, "password123")
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}
	defer store.Delete(user.ID)

	// Update user
	user.FirstName = "Updated"
	user.LastName = "Name"
	err = store.Update(user)
	if err != nil {
		t.Fatalf("Failed to update user: %v", err)
	}

	// Verify update
	retrieved, err := store.Get(user.ID)
	if err != nil {
		t.Fatalf("Failed to get user: %v", err)
	}

	if retrieved.FirstName != "Updated" {
		t.Errorf("FirstName not updated: expected 'Updated', got '%s'", retrieved.FirstName)
	}
}

func TestPostgresUserStore_GetByUsername(t *testing.T) {
	db := getTestDB(t)
	if db == nil {
		return
	}
	defer db.Close()

	store := NewPostgresUserStore(db)

	username := "testuser_byname_" + time.Now().Format("20060102150405")
	user := &User{
		Username:  username,
		Email:     "test_byname_" + time.Now().Format("20060102150405") + "@example.com",
		FirstName: "Test",
		LastName:  "User",
		Status:    UserStatusActive,
		TenantID:  "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
	}

	err := store.Create(user, "password123")
	if err != nil {
		t.Fatalf("Failed to create user: %v", err)
	}
	defer store.Delete(user.ID)

	// Test GetByUsername
	retrieved, err := store.GetByUsername(username)
	if err != nil {
		t.Fatalf("Failed to get user by username: %v", err)
	}

	if retrieved.ID != user.ID {
		t.Errorf("ID mismatch: expected %s, got %s", user.ID, retrieved.ID)
	}
}

func TestPostgresUserStore_List(t *testing.T) {
	db := getTestDB(t)
	if db == nil {
		return
	}
	defer db.Close()

	store := NewPostgresUserStore(db)

	// List users (should at least return existing seed users)
	users, err := store.List(map[string]interface{}{
		"limit": 10,
	})
	if err != nil {
		t.Fatalf("Failed to list users: %v", err)
	}

	t.Logf("Found %d users", len(users))
}
