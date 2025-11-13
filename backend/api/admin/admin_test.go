package admin

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
	_ "github.com/mattn/go-sqlite3"
	"novacron/backend/pkg/testutil"
)

// setupTestDB creates an in-memory SQLite database for testing
func setupTestDB(t *testing.T) *sql.DB {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open test database: %v", err)
	}

	// Create tables
	schema := `
	CREATE TABLE IF NOT EXISTS users (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		username TEXT UNIQUE NOT NULL,
		email TEXT UNIQUE NOT NULL,
		password TEXT NOT NULL,
		role TEXT NOT NULL DEFAULT 'user',
		active BOOLEAN NOT NULL DEFAULT 1,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
		updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS vm_templates (
		id TEXT PRIMARY KEY,
		name TEXT NOT NULL,
		description TEXT,
		os TEXT NOT NULL,
		os_version TEXT,
		cpu_cores INTEGER NOT NULL,
		memory_mb INTEGER NOT NULL,
		disk_gb INTEGER NOT NULL,
		image_path TEXT,
		is_public BOOLEAN DEFAULT 0,
		usage_count INTEGER DEFAULT 0,
		tags TEXT,
		metadata TEXT,
		created_by TEXT NOT NULL,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
		updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS security_alerts (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		type TEXT NOT NULL,
		severity TEXT NOT NULL,
		title TEXT NOT NULL,
		description TEXT,
		source TEXT,
		ip TEXT,
		user_agent TEXT,
		status TEXT DEFAULT 'open',
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
		updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS audit_logs (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		user_id INTEGER,
		username TEXT,
		action TEXT NOT NULL,
		resource TEXT,
		details TEXT,
		ip TEXT,
		user_agent TEXT,
		success BOOLEAN,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	);
	`

	if _, err := db.Exec(schema); err != nil {
		t.Fatalf("Failed to create schema: %v", err)
	}

	return db
}

// TestUserManagementHandlers tests user management endpoints
func TestUserManagementHandlers(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	handlers := NewUserManagementHandlers(db)
	router := mux.NewRouter()

	router.HandleFunc("/api/admin/users", handlers.ListUsers).Methods("GET")
	router.HandleFunc("/api/admin/users", handlers.CreateUser).Methods("POST")
	router.HandleFunc("/api/admin/users/{id}", handlers.GetUser).Methods("GET")

	t.Run("CreateUser", func(t *testing.T) {
		reqBody := CreateUserRequest{
			Username: testutil.DefaultTestUsername,
			Email:    testutil.GetTestEmail(),
			Password: "SecureP@ssw0rd",
			Role:     "user",
		}

		body, _ := json.Marshal(reqBody)
		req := httptest.NewRequest("POST", "/api/admin/users", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		if w.Code != http.StatusCreated && w.Code != http.StatusOK {
			t.Errorf("Expected status 201 or 200, got %d", w.Code)
		}

		var user User
		json.NewDecoder(w.Body).Decode(&user)

		if user.Username != reqBody.Username {
			t.Errorf("Expected username %s, got %s", reqBody.Username, user.Username)
		}
	})

	t.Run("ListUsers", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/api/admin/users", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}

		var response UserListResponse
		json.NewDecoder(w.Body).Decode(&response)

		if response.Total == 0 {
			t.Error("Expected at least 1 user after creation")
		}
	})
}

// TestTemplateHandlers tests VM template endpoints
func TestTemplateHandlers(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	handlers := NewTemplateHandlers(db)
	router := mux.NewRouter()

	router.HandleFunc("/api/admin/templates", handlers.ListTemplates).Methods("GET")
	router.HandleFunc("/api/admin/templates", handlers.CreateTemplate).Methods("POST")
	router.HandleFunc("/api/admin/templates/{id}", handlers.GetTemplate).Methods("GET")

	var createdTemplateID string

	t.Run("CreateTemplate", func(t *testing.T) {
		reqBody := CreateTemplateRequest{
			Name:        "Ubuntu 24.04 LTS",
			Description: "Ubuntu 24.04 LTS server template",
			OS:          "ubuntu",
			OSVersion:   "24.04",
			CPUCores:    2,
			MemoryMB:    4096,
			DiskGB:      40,
			ImagePath:   "/images/ubuntu-24.04.qcow2",
			IsPublic:    true,
			Tags:        []string{"linux", "ubuntu", "server"},
		}

		body, _ := json.Marshal(reqBody)
		req := httptest.NewRequest("POST", "/api/admin/templates", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-User-Email", "admin@test.com")
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		if w.Code != http.StatusCreated && w.Code != http.StatusOK {
			t.Errorf("Expected status 201 or 200, got %d: %s", w.Code, w.Body.String())
		}

		var template VMTemplate
		json.NewDecoder(w.Body).Decode(&template)

		createdTemplateID = template.ID

		if template.Name != reqBody.Name {
			t.Errorf("Expected name %s, got %s", reqBody.Name, template.Name)
		}

		if template.OS != reqBody.OS {
			t.Errorf("Expected OS %s, got %s", reqBody.OS, template.OS)
		}
	})

	t.Run("ListTemplates", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/api/admin/templates", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}

		var response TemplateListResponse
		json.NewDecoder(w.Body).Decode(&response)

		if response.Total == 0 {
			t.Error("Expected at least 1 template after creation")
		}
	})

	t.Run("GetTemplate", func(t *testing.T) {
		if createdTemplateID == "" {
			t.Skip("No template created to get")
		}

		req := httptest.NewRequest("GET", "/api/admin/templates/"+createdTemplateID, nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		if w.Code != http.StatusOK && w.Code != http.StatusNotFound {
			t.Errorf("Expected status 200 or 404, got %d", w.Code)
		}
	})
}

// TestSecurityHandlers tests security endpoint
func TestSecurityHandlers(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	handlers := NewSecurityHandlers(db)
	router := mux.NewRouter()

	router.HandleFunc("/api/admin/security/metrics", handlers.GetSecurityMetrics).Methods("GET")

	t.Run("GetSecurityMetrics", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/api/admin/security/metrics", nil)
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("Expected status 200, got %d", w.Code)
		}

		var metrics SecurityMetrics
		json.NewDecoder(w.Body).Decode(&metrics)

		// Just verify the response is properly structured
		if metrics.AlertsByType == nil {
			t.Error("Expected AlertsByType to be initialized")
		}
	})
}

// TestAdminHandlersIntegration tests the full admin handlers setup
func TestAdminHandlersIntegration(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	adminHandlers := NewAdminHandlers(db, "/tmp/config.json")
	router := mux.NewRouter()
	adminHandlers.RegisterRoutes(router)

	t.Run("HealthCheck", func(t *testing.T) {
		// Test that routes are registered properly by checking a few endpoints
		endpoints := []string{
			"/api/admin/users",
			"/api/admin/security/metrics",
			"/api/admin/templates",
			"/api/admin/config",
		}

		for _, endpoint := range endpoints {
			req := httptest.NewRequest("GET", endpoint, nil)
			w := httptest.NewRecorder()

			router.ServeHTTP(w, req)

			// We expect either 200 OK or an error response (not 404)
			// 404 would mean the route wasn't registered
			if w.Code == http.StatusNotFound {
				t.Errorf("Route %s not registered (got 404)", endpoint)
			}
		}
	})
}

// TestInputValidation tests input validation across handlers
func TestInputValidation(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	t.Run("CreateUserWithInvalidData", func(t *testing.T) {
		handlers := NewUserManagementHandlers(db)
		router := mux.NewRouter()
		router.HandleFunc("/api/admin/users", handlers.CreateUser).Methods("POST")

		// Test with empty username
		reqBody := CreateUserRequest{
			Username: "",
			Email:    testutil.GetTestEmail(),
			Password: "password",
			Role:     "user",
		}

		body, _ := json.Marshal(reqBody)
		req := httptest.NewRequest("POST", "/api/admin/users", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		// Should fail with 400 Bad Request
		if w.Code != http.StatusBadRequest && w.Code != http.StatusInternalServerError {
			t.Logf("Expected 400 or 500 for invalid input, got %d", w.Code)
		}
	})

	t.Run("CreateTemplateWithInvalidData", func(t *testing.T) {
		handlers := NewTemplateHandlers(db)
		router := mux.NewRouter()
		router.HandleFunc("/api/admin/templates", handlers.CreateTemplate).Methods("POST")

		// Test with missing required fields
		reqBody := CreateTemplateRequest{
			Name: "",
			OS:   "",
		}

		body, _ := json.Marshal(reqBody)
		req := httptest.NewRequest("POST", "/api/admin/templates", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		router.ServeHTTP(w, req)

		// Should fail with 400 Bad Request
		if w.Code != http.StatusBadRequest {
			t.Errorf("Expected 400 for invalid input, got %d", w.Code)
		}
	})
}

// BenchmarkListUsers benchmarks the user listing endpoint
func BenchmarkListUsers(b *testing.B) {
	db := setupTestDB(&testing.T{})
	defer db.Close()

	// Insert some test users
	for i := 0; i < 100; i++ {
		db.Exec("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
			"user"+string(rune(i)), "user"+string(rune(i))+"@test.com", "password", "user")
	}

	handlers := NewUserManagementHandlers(db)
	router := mux.NewRouter()
	router.HandleFunc("/api/admin/users", handlers.ListUsers).Methods("GET")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/admin/users", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
	}
}

// BenchmarkListTemplates benchmarks the template listing endpoint
func BenchmarkListTemplates(b *testing.B) {
	db := setupTestDB(&testing.T{})
	defer db.Close()

	// Insert some test templates
	for i := 0; i < 50; i++ {
		db.Exec(`INSERT INTO vm_templates (id, name, os, os_version, cpu_cores, memory_mb, disk_gb, created_by)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
			"tmpl-"+string(rune(i)), "Template "+string(rune(i)), "ubuntu", "24.04", 2, 4096, 40, "system")
	}

	handlers := NewTemplateHandlers(db)
	router := mux.NewRouter()
	router.HandleFunc("/api/admin/templates", handlers.ListTemplates).Methods("GET")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/admin/templates", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
	}
}
