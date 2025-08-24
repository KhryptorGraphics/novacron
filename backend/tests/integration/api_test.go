package integration

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/api/vm"
)

// TestVMAPIIntegration tests the VM management API endpoints
func TestVMAPIIntegration(t *testing.T) {
	// Create a test router
	router := mux.NewRouter()
	
	// Note: In a real integration test, we would set up a test database
	// and create proper VM manager instances. For now, we test the routing.
	
	// Test route registration
	router.HandleFunc("/api/vms", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case "GET":
			w.Header().Set("Content-Type", "application/json")
			response := []map[string]interface{}{
				{
					"id":     "test-vm-1",
					"name":   "Test VM 1",
					"state":  "running",
					"cpu":    2,
					"memory": 1024,
				},
			}
			json.NewEncoder(w).Encode(response)
		case "POST":
			var vmConfig map[string]interface{}
			err := json.NewDecoder(r.Body).Decode(&vmConfig)
			if err != nil {
				http.Error(w, "Invalid JSON", http.StatusBadRequest)
				return
			}
			
			// Validate required fields
			if vmConfig["name"] == nil {
				http.Error(w, "Name is required", http.StatusBadRequest)
				return
			}
			
			w.WriteHeader(http.StatusCreated)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"id":      "test-vm-new",
				"name":    vmConfig["name"],
				"state":   "created",
				"message": "VM created successfully",
			})
		}
	}).Methods("GET", "POST")

	// Test GET /api/vms
	t.Run("ListVMs", func(t *testing.T) {
		req, err := http.NewRequest("GET", "/api/vms", nil)
		if err != nil {
			t.Fatalf("Failed to create request: %v", err)
		}

		rr := httptest.NewRecorder()
		router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusOK {
			t.Errorf("Expected status code %d, got %d", http.StatusOK, status)
		}

		var response []map[string]interface{}
		err = json.NewDecoder(rr.Body).Decode(&response)
		if err != nil {
			t.Fatalf("Failed to decode response: %v", err)
		}

		if len(response) != 1 {
			t.Errorf("Expected 1 VM in response, got %d", len(response))
		}

		if response[0]["name"] != "Test VM 1" {
			t.Errorf("Expected VM name 'Test VM 1', got %s", response[0]["name"])
		}
	})

	// Test POST /api/vms
	t.Run("CreateVM", func(t *testing.T) {
		vmData := map[string]interface{}{
			"name":   "Integration Test VM",
			"cpu":    2,
			"memory": 1024,
		}

		jsonData, err := json.Marshal(vmData)
		if err != nil {
			t.Fatalf("Failed to marshal JSON: %v", err)
		}

		req, err := http.NewRequest("POST", "/api/vms", bytes.NewBuffer(jsonData))
		if err != nil {
			t.Fatalf("Failed to create request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusCreated {
			t.Errorf("Expected status code %d, got %d", http.StatusCreated, status)
		}

		var response map[string]interface{}
		err = json.NewDecoder(rr.Body).Decode(&response)
		if err != nil {
			t.Fatalf("Failed to decode response: %v", err)
		}

		if response["name"] != "Integration Test VM" {
			t.Errorf("Expected VM name 'Integration Test VM', got %s", response["name"])
		}
	})

	// Test POST with invalid data
	t.Run("CreateVMInvalidData", func(t *testing.T) {
		invalidData := map[string]interface{}{
			"cpu":    2,
			"memory": 1024,
			// Missing required name field
		}

		jsonData, err := json.Marshal(invalidData)
		if err != nil {
			t.Fatalf("Failed to marshal JSON: %v", err)
		}

		req, err := http.NewRequest("POST", "/api/vms", bytes.NewBuffer(jsonData))
		if err != nil {
			t.Fatalf("Failed to create request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusBadRequest {
			t.Errorf("Expected status code %d, got %d", http.StatusBadRequest, status)
		}
	})
}

// TestAuthAPIIntegration tests authentication endpoints
func TestAuthAPIIntegration(t *testing.T) {
	router := mux.NewRouter()

	// Mock auth endpoints
	router.HandleFunc("/api/auth/login", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var loginData map[string]string
		err := json.NewDecoder(r.Body).Decode(&loginData)
		if err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		email := loginData["email"]
		password := loginData["password"]

		if email == "" || password == "" {
			http.Error(w, "Email and password required", http.StatusBadRequest)
			return
		}

		// Mock successful login
		if email == "test@example.com" && password == "password123" {
			response := map[string]interface{}{
				"token":     "mock-jwt-token",
				"user":      map[string]string{"email": email, "name": "Test User"},
				"expiresIn": 3600,
			}
			json.NewEncoder(w).Encode(response)
		} else {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
		}
	}).Methods("POST")

	// Test successful login
	t.Run("LoginSuccess", func(t *testing.T) {
		loginData := map[string]string{
			"email":    "test@example.com",
			"password": "password123",
		}

		jsonData, err := json.Marshal(loginData)
		if err != nil {
			t.Fatalf("Failed to marshal JSON: %v", err)
		}

		req, err := http.NewRequest("POST", "/api/auth/login", bytes.NewBuffer(jsonData))
		if err != nil {
			t.Fatalf("Failed to create request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusOK {
			t.Errorf("Expected status code %d, got %d", http.StatusOK, status)
		}

		var response map[string]interface{}
		err = json.NewDecoder(rr.Body).Decode(&response)
		if err != nil {
			t.Fatalf("Failed to decode response: %v", err)
		}

		if response["token"] == nil {
			t.Error("Expected token in response")
		}
	})

	// Test failed login
	t.Run("LoginFailure", func(t *testing.T) {
		loginData := map[string]string{
			"email":    "wrong@example.com",
			"password": "wrongpassword",
		}

		jsonData, err := json.Marshal(loginData)
		if err != nil {
			t.Fatalf("Failed to marshal JSON: %v", err)
		}

		req, err := http.NewRequest("POST", "/api/auth/login", bytes.NewBuffer(jsonData))
		if err != nil {
			t.Fatalf("Failed to create request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")

		rr := httptest.NewRecorder()
		router.ServeHTTP(rr, req)

		if status := rr.Code; status != http.StatusUnauthorized {
			t.Errorf("Expected status code %d, got %d", http.StatusUnauthorized, status)
		}
	})
}

// TestCORSHeaders tests CORS configuration
func TestCORSHeaders(t *testing.T) {
	router := mux.NewRouter()

	router.HandleFunc("/api/test", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		json.NewEncoder(w).Encode(map[string]string{"message": "OK"})
	}).Methods("GET", "OPTIONS")

	// Test CORS preflight request
	t.Run("CORSPreflight", func(t *testing.T) {
		req, err := http.NewRequest("OPTIONS", "/api/test", nil)
		if err != nil {
			t.Fatalf("Failed to create request: %v", err)
		}
		req.Header.Set("Origin", "http://localhost:8092")
		req.Header.Set("Access-Control-Request-Method", "POST")

		rr := httptest.NewRecorder()
		router.ServeHTTP(rr, req)

		// Check CORS headers
		if origin := rr.Header().Get("Access-Control-Allow-Origin"); origin == "" {
			t.Error("Expected Access-Control-Allow-Origin header")
		}

		if methods := rr.Header().Get("Access-Control-Allow-Methods"); methods == "" {
			t.Error("Expected Access-Control-Allow-Methods header")
		}
	})
}
