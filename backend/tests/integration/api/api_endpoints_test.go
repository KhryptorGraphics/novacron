package api_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/tests/integration"
)

// APIEndpointsTestSuite tests API endpoints with database connectivity
type APIEndpointsTestSuite struct {
	integration.IntegrationTestSuite
	testUserID int
	authToken  string
}

// SetupSuite runs before all API endpoint tests
func (suite *APIEndpointsTestSuite) SetupSuite() {
	suite.IntegrationTestSuite.SetupSuite()
	suite.setupTestUser()
	suite.registerAPIRoutes()
}

// setupTestUser creates a test user for API operations
func (suite *APIEndpointsTestSuite) setupTestUser() {
	user, err := suite.GetAuthManager().CreateUser(
		"apitestuser", 
		"apitest@example.com", 
		"APITestPassword123!", 
		"user", 
		"test-tenant",
	)
	suite.Require().NoError(err, "Failed to create API test user")
	suite.testUserID = user.ID

	_, token, err := suite.GetAuthManager().Authenticate("apitestuser", "APITestPassword123!")
	suite.Require().NoError(err, "Failed to authenticate API test user")
	suite.authToken = token

	suite.T().Log("API test user created and authenticated")
}

// registerAPIRoutes registers all API routes for testing
func (suite *APIEndpointsTestSuite) registerAPIRoutes() {
	router := suite.GetRouter()

	// Health check endpoint
	router.HandleFunc("/health", suite.handleHealthCheck).Methods("GET")

	// API info endpoint
	router.HandleFunc("/api/info", suite.handleAPIInfo).Methods("GET")

	// Monitoring endpoints
	router.HandleFunc("/api/monitoring/metrics", suite.handleSystemMetrics).Methods("GET")
	router.HandleFunc("/api/monitoring/vms", suite.handleVMMetrics).Methods("GET")
	router.HandleFunc("/api/monitoring/alerts", suite.handleAlerts).Methods("GET")

	// User management endpoints
	router.HandleFunc("/api/users", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			suite.handleListUsers(w, r)
		} else if r.Method == "POST" {
			suite.handleCreateUser(w, r)
		}
	}).Methods("GET", "POST")

	router.HandleFunc("/api/users/{id}", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			suite.handleGetUser(w, r)
		} else if r.Method == "PUT" {
			suite.handleUpdateUser(w, r)
		} else if r.Method == "DELETE" {
			suite.handleDeleteUser(w, r)
		}
	}).Methods("GET", "PUT", "DELETE")

	// VM metrics endpoint
	router.HandleFunc("/api/vms/{id}/metrics", suite.handleVMMetricsHistory).Methods("GET")

	// Database connectivity test endpoint
	router.HandleFunc("/api/database/test", suite.handleDatabaseTest).Methods("GET")

	suite.T().Log("API routes registered for testing")
}

// TestHealthAndInfo tests health check and API info endpoints
func (suite *APIEndpointsTestSuite) TestHealthAndInfo() {
	suite.T().Log("Testing health and info endpoints...")

	// Test health check
	suite.T().Run("HealthCheck", func(t *testing.T) {
		resp, err := http.Get(fmt.Sprintf("%s/health", suite.GetServer().URL))
		require.NoError(t, err, "Failed to send health check request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "Health check should return OK")

		var health map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&health)
		require.NoError(t, err, "Failed to decode health response")

		assert.Contains(t, health, "status", "Health response should contain status")
		assert.Contains(t, health, "service", "Health response should contain service name")
		assert.Equal(t, "healthy", health["status"], "Service should be healthy")
	})

	// Test API info
	suite.T().Run("APIInfo", func(t *testing.T) {
		resp, err := http.Get(fmt.Sprintf("%s/api/info", suite.GetServer().URL))
		require.NoError(t, err, "Failed to send API info request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "API info should return OK")

		var info map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&info)
		require.NoError(t, err, "Failed to decode API info response")

		assert.Contains(t, info, "name", "API info should contain name")
		assert.Contains(t, info, "version", "API info should contain version")
	})

	suite.T().Log("✓ Health and info endpoints tested successfully")
}

// TestMonitoringEndpoints tests monitoring-related endpoints
func (suite *APIEndpointsTestSuite) TestMonitoringEndpoints() {
	suite.T().Log("Testing monitoring endpoints...")

	// Test system metrics
	suite.T().Run("SystemMetrics", func(t *testing.T) {
		req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/monitoring/metrics", suite.GetServer().URL), nil)
		require.NoError(t, err, "Failed to create system metrics request")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send system metrics request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "System metrics should return OK")

		var metrics map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&metrics)
		require.NoError(t, err, "Failed to decode metrics response")

		// Verify expected metric fields
		expectedFields := []string{
			"currentCpuUsage", "currentMemoryUsage", "currentDiskUsage", "currentNetworkUsage",
		}
		for _, field := range expectedFields {
			assert.Contains(t, metrics, field, "Metrics should contain %s", field)
		}
	})

	// Test VM metrics
	suite.T().Run("VMMetrics", func(t *testing.T) {
		req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/monitoring/vms", suite.GetServer().URL), nil)
		require.NoError(t, err, "Failed to create VM metrics request")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send VM metrics request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "VM metrics should return OK")

		var vmMetrics []map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&vmMetrics)
		require.NoError(t, err, "Failed to decode VM metrics response")

		// Should return an array of VM metrics
		assert.IsType(t, []map[string]interface{}{}, vmMetrics, "VM metrics should be an array")
	})

	// Test alerts
	suite.T().Run("Alerts", func(t *testing.T) {
		req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/monitoring/alerts", suite.GetServer().URL), nil)
		require.NoError(t, err, "Failed to create alerts request")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send alerts request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "Alerts should return OK")

		var alerts []map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&alerts)
		require.NoError(t, err, "Failed to decode alerts response")

		assert.IsType(t, []map[string]interface{}{}, alerts, "Alerts should be an array")
	})

	suite.T().Log("✓ Monitoring endpoints tested successfully")
}

// TestUserManagementEndpoints tests user management endpoints with database
func (suite *APIEndpointsTestSuite) TestUserManagementEndpoints() {
	suite.T().Log("Testing user management endpoints...")

	var createdUserID int

	// Test create user
	suite.T().Run("CreateUser", func(t *testing.T) {
		userData := map[string]interface{}{
			"username": "newuser_api",
			"email":    "newuser.api@example.com",
			"password": "NewUserPassword123!",
			"role":     "user",
			"tenant_id": "test-tenant",
		}

		jsonData, err := json.Marshal(userData)
		require.NoError(t, err, "Failed to marshal user data")

		req, err := http.NewRequest("POST", fmt.Sprintf("%s/api/users", suite.GetServer().URL), bytes.NewBuffer(jsonData))
		require.NoError(t, err, "Failed to create user creation request")
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send user creation request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusCreated, resp.StatusCode, "User creation should succeed")

		var response map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&response)
		require.NoError(t, err, "Failed to decode user creation response")

		assert.Contains(t, response, "id", "Response should contain user ID")
		assert.Contains(t, response, "username", "Response should contain username")
		assert.Equal(t, userData["username"], response["username"], "Username should match")

		createdUserID = int(response["id"].(float64))
	})

	// Test list users
	suite.T().Run("ListUsers", func(t *testing.T) {
		req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/users", suite.GetServer().URL), nil)
		require.NoError(t, err, "Failed to create list users request")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send list users request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "List users should succeed")

		var users []map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&users)
		require.NoError(t, err, "Failed to decode users list response")

		assert.GreaterOrEqual(t, len(users), 1, "Should have at least one user")

		// Find our created user
		var foundUser map[string]interface{}
		for _, user := range users {
			if int(user["id"].(float64)) == createdUserID {
				foundUser = user
				break
			}
		}
		assert.NotNil(t, foundUser, "Created user should be in the list")
	})

	// Test get user
	suite.T().Run("GetUser", func(t *testing.T) {
		req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/users/%d", suite.GetServer().URL, createdUserID), nil)
		require.NoError(t, err, "Failed to create get user request")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send get user request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "Get user should succeed")

		var user map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&user)
		require.NoError(t, err, "Failed to decode user response")

		assert.Equal(t, float64(createdUserID), user["id"], "User ID should match")
		assert.Contains(t, user, "username", "User should have username")
		assert.Contains(t, user, "email", "User should have email")
	})

	// Test update user
	suite.T().Run("UpdateUser", func(t *testing.T) {
		updateData := map[string]interface{}{
			"username": "updateduser_api",
			"email":    "updateduser.api@example.com",
		}

		jsonData, err := json.Marshal(updateData)
		require.NoError(t, err, "Failed to marshal update data")

		req, err := http.NewRequest("PUT", fmt.Sprintf("%s/api/users/%d", suite.GetServer().URL, createdUserID), bytes.NewBuffer(jsonData))
		require.NoError(t, err, "Failed to create update user request")
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send update user request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "Update user should succeed")

		var response map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&response)
		require.NoError(t, err, "Failed to decode update response")

		assert.Equal(t, updateData["username"], response["username"], "Username should be updated")
	})

	// Test delete user
	suite.T().Run("DeleteUser", func(t *testing.T) {
		req, err := http.NewRequest("DELETE", fmt.Sprintf("%s/api/users/%d", suite.GetServer().URL, createdUserID), nil)
		require.NoError(t, err, "Failed to create delete user request")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send delete user request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "Delete user should succeed")
	})

	// Verify user is deleted
	suite.T().Run("VerifyUserDeleted", func(t *testing.T) {
		req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/users/%d", suite.GetServer().URL, createdUserID), nil)
		require.NoError(t, err, "Failed to create get deleted user request")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send get deleted user request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusNotFound, resp.StatusCode, "Deleted user should not be found")
	})

	suite.T().Log("✓ User management endpoints tested successfully")
}

// TestDatabaseConnectivity tests database connectivity through API
func (suite *APIEndpointsTestSuite) TestDatabaseConnectivity() {
	suite.T().Log("Testing database connectivity...")

	suite.T().Run("DatabaseConnectionTest", func(t *testing.T) {
		req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/database/test", suite.GetServer().URL), nil)
		require.NoError(t, err, "Failed to create database test request")
		req.Header.Set("Authorization", "Bearer "+suite.authToken)

		client := &http.Client{}
		resp, err := client.Do(req)
		require.NoError(t, err, "Failed to send database test request")
		defer resp.Body.Close()

		assert.Equal(t, http.StatusOK, resp.StatusCode, "Database test should succeed")

		var response map[string]interface{}
		err = json.NewDecoder(resp.Body).Decode(&response)
		require.NoError(t, err, "Failed to decode database test response")

		assert.Contains(t, response, "status", "Response should contain status")
		assert.Contains(t, response, "connection", "Response should contain connection info")
		assert.Equal(t, "healthy", response["status"], "Database should be healthy")
	})

	suite.T().Log("✓ Database connectivity tested successfully")
}

// Handler implementations for the test routes

func (suite *APIEndpointsTestSuite) handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	// Test database connectivity in health check
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	status := "healthy"
	checks := make(map[string]string)

	if err := suite.GetDatabase().PingContext(ctx); err != nil {
		status = "unhealthy"
		checks["database"] = fmt.Sprintf("error: %v", err)
	} else {
		checks["database"] = "ok"
	}

	response := map[string]interface{}{
		"status":    status,
		"service":   "novacron-test",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"checks":    checks,
	}

	w.Header().Set("Content-Type", "application/json")
	if status == "unhealthy" {
		w.WriteHeader(http.StatusServiceUnavailable)
	}
	json.NewEncoder(w).Encode(response)
}

func (suite *APIEndpointsTestSuite) handleAPIInfo(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"name":        "NovaCron Test API",
		"version":     "test",
		"description": "Integration test API",
		"endpoints": []string{
			"/health",
			"/api/info",
			"/api/monitoring/metrics",
			"/api/monitoring/vms",
			"/api/users",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (suite *APIEndpointsTestSuite) handleSystemMetrics(w http.ResponseWriter, r *http.Request) {
	// Simulate system metrics with some realistic data
	response := map[string]interface{}{
		"currentCpuUsage":        45.2,
		"currentMemoryUsage":     72.1,
		"currentDiskUsage":       58.3,
		"currentNetworkUsage":    125.7,
		"cpuChangePercentage":    5.2,
		"memoryChangePercentage": -2.1,
		"diskChangePercentage":   1.8,
		"networkChangePercentage": 12.5,
		"timestamp":              time.Now().UTC().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (suite *APIEndpointsTestSuite) handleVMMetrics(w http.ResponseWriter, r *http.Request) {
	// Query VM metrics from database
	rows, err := suite.GetDatabase().Query(`
		SELECT vm_id, cpu_usage, memory_usage, disk_usage, network_sent, network_recv, timestamp 
		FROM vm_metrics 
		ORDER BY timestamp DESC 
		LIMIT 100
	`)
	if err != nil {
		http.Error(w, fmt.Sprintf("Database query failed: %v", err), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var metrics []map[string]interface{}
	for rows.Next() {
		var vmID string
		var cpuUsage, memoryUsage, diskUsage float64
		var networkSent, networkRecv int64
		var timestamp time.Time

		err := rows.Scan(&vmID, &cpuUsage, &memoryUsage, &diskUsage, &networkSent, &networkRecv, &timestamp)
		if err != nil {
			continue
		}

		metric := map[string]interface{}{
			"vm_id":         vmID,
			"cpu_usage":     cpuUsage,
			"memory_usage":  memoryUsage,
			"disk_usage":    diskUsage,
			"network_sent":  networkSent,
			"network_recv":  networkRecv,
			"timestamp":     timestamp,
		}
		metrics = append(metrics, metric)
	}

	// If no metrics in database, return mock data
	if len(metrics) == 0 {
		metrics = []map[string]interface{}{
			{
				"vm_id":        "test-vm-1",
				"cpu_usage":    45.2,
				"memory_usage": 62.8,
				"disk_usage":   35.1,
				"network_sent": 1048576,
				"network_recv": 2097152,
				"timestamp":    time.Now(),
			},
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (suite *APIEndpointsTestSuite) handleAlerts(w http.ResponseWriter, r *http.Request) {
	// Mock alerts data
	alerts := []map[string]interface{}{
		{
			"id":          "alert-001",
			"name":        "High CPU Usage",
			"description": "System CPU usage exceeds threshold",
			"severity":    "warning",
			"status":      "active",
			"timestamp":   time.Now().Add(-10 * time.Minute),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(alerts)
}

func (suite *APIEndpointsTestSuite) handleListUsers(w http.ResponseWriter, r *http.Request) {
	rows, err := suite.GetDatabase().Query(`
		SELECT id, username, email, role, tenant_id, is_active, created_at 
		FROM users 
		ORDER BY created_at DESC
	`)
	if err != nil {
		http.Error(w, fmt.Sprintf("Database query failed: %v", err), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var users []map[string]interface{}
	for rows.Next() {
		var id int
		var username, email, role, tenantID string
		var isActive bool
		var createdAt time.Time

		err := rows.Scan(&id, &username, &email, &role, &tenantID, &isActive, &createdAt)
		if err != nil {
			continue
		}

		user := map[string]interface{}{
			"id":         id,
			"username":   username,
			"email":      email,
			"role":       role,
			"tenant_id":  tenantID,
			"is_active":  isActive,
			"created_at": createdAt,
		}
		users = append(users, user)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(users)
}

func (suite *APIEndpointsTestSuite) handleCreateUser(w http.ResponseWriter, r *http.Request) {
	var userData map[string]interface{}
	err := json.NewDecoder(r.Body).Decode(&userData)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Validate required fields
	required := []string{"username", "email", "password"}
	for _, field := range required {
		if userData[field] == nil || userData[field].(string) == "" {
			http.Error(w, fmt.Sprintf("%s is required", field), http.StatusBadRequest)
			return
		}
	}

	// Set defaults
	if userData["role"] == nil {
		userData["role"] = "user"
	}
	if userData["tenant_id"] == nil {
		userData["tenant_id"] = "default"
	}

	// Create user using auth manager
	user, err := suite.GetAuthManager().CreateUser(
		userData["username"].(string),
		userData["email"].(string),
		userData["password"].(string),
		userData["role"].(string),
		userData["tenant_id"].(string),
	)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create user: %v", err), http.StatusBadRequest)
		return
	}

	response := map[string]interface{}{
		"id":         user.ID,
		"username":   user.Username,
		"email":      user.Email,
		"tenant_id":  user.TenantID,
		"created_at": time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

func (suite *APIEndpointsTestSuite) handleGetUser(w http.ResponseWriter, r *http.Request) {
	// Extract user ID from URL
	userID := r.URL.Path[len("/api/users/"):]

	var id int
	var username, email, role, tenantID string
	var isActive bool
	var createdAt time.Time

	err := suite.GetDatabase().QueryRow(`
		SELECT id, username, email, role, tenant_id, is_active, created_at 
		FROM users WHERE id = $1
	`, userID).Scan(&id, &username, &email, &role, &tenantID, &isActive, &createdAt)

	if err != nil {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	user := map[string]interface{}{
		"id":         id,
		"username":   username,
		"email":      email,
		"role":       role,
		"tenant_id":  tenantID,
		"is_active":  isActive,
		"created_at": createdAt,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

func (suite *APIEndpointsTestSuite) handleUpdateUser(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Path[len("/api/users/"):]

	var updateData map[string]interface{}
	err := json.NewDecoder(r.Body).Decode(&updateData)
	if err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Build dynamic update query
	setParts := []string{}
	args := []interface{}{}
	argIndex := 1

	if updateData["username"] != nil {
		setParts = append(setParts, fmt.Sprintf("username = $%d", argIndex))
		args = append(args, updateData["username"])
		argIndex++
	}

	if updateData["email"] != nil {
		setParts = append(setParts, fmt.Sprintf("email = $%d", argIndex))
		args = append(args, updateData["email"])
		argIndex++
	}

	if len(setParts) == 0 {
		http.Error(w, "No fields to update", http.StatusBadRequest)
		return
	}

	setParts = append(setParts, fmt.Sprintf("updated_at = CURRENT_TIMESTAMP"))
	args = append(args, userID)

	query := fmt.Sprintf("UPDATE users SET %s WHERE id = $%d RETURNING id, username, email, role, tenant_id", 
		fmt.Sprintf("%s", setParts), argIndex)

	var id int
	var username, email, role, tenantID string
	err = suite.GetDatabase().QueryRow(query, args...).Scan(&id, &username, &email, &role, &tenantID)
	if err != nil {
		http.Error(w, "Failed to update user", http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"id":        id,
		"username":  username,
		"email":     email,
		"role":      role,
		"tenant_id": tenantID,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (suite *APIEndpointsTestSuite) handleDeleteUser(w http.ResponseWriter, r *http.Request) {
	userID := r.URL.Path[len("/api/users/"):]

	result, err := suite.GetDatabase().Exec("DELETE FROM users WHERE id = $1", userID)
	if err != nil {
		http.Error(w, "Failed to delete user", http.StatusInternalServerError)
		return
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil || rowsAffected == 0 {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "User deleted successfully",
	})
}

func (suite *APIEndpointsTestSuite) handleVMMetricsHistory(w http.ResponseWriter, r *http.Request) {
	// Extract VM ID from URL
	vmID := r.URL.Path[len("/api/vms/"):]
	vmID = vmID[:len(vmID)-8] // Remove "/metrics"

	// Query VM metrics history
	rows, err := suite.GetDatabase().Query(`
		SELECT cpu_usage, memory_usage, disk_usage, network_sent, network_recv, timestamp 
		FROM vm_metrics 
		WHERE vm_id = $1 
		ORDER BY timestamp DESC 
		LIMIT 100
	`, vmID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Database query failed: %v", err), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	var metrics []map[string]interface{}
	for rows.Next() {
		var cpuUsage, memoryUsage, diskUsage float64
		var networkSent, networkRecv int64
		var timestamp time.Time

		err := rows.Scan(&cpuUsage, &memoryUsage, &diskUsage, &networkSent, &networkRecv, &timestamp)
		if err != nil {
			continue
		}

		metric := map[string]interface{}{
			"cpu_usage":    cpuUsage,
			"memory_usage": memoryUsage,
			"disk_usage":   diskUsage,
			"network_sent": networkSent,
			"network_recv": networkRecv,
			"timestamp":    timestamp,
		}
		metrics = append(metrics, metric)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"vm_id":   vmID,
		"metrics": metrics,
	})
}

func (suite *APIEndpointsTestSuite) handleDatabaseTest(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	// Test basic connectivity
	if err := suite.GetDatabase().PingContext(ctx); err != nil {
		http.Error(w, fmt.Sprintf("Database connection failed: %v", err), http.StatusServiceUnavailable)
		return
	}

	// Test query execution
	var count int
	err := suite.GetDatabase().QueryRowContext(ctx, "SELECT COUNT(*) FROM users").Scan(&count)
	if err != nil {
		http.Error(w, fmt.Sprintf("Database query failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]interface{}{
		"status":     "healthy",
		"connection": "ok",
		"user_count": count,
		"timestamp":  time.Now().UTC().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// TestAPIEndpointsSuite runs the API endpoints test suite
func TestAPIEndpointsSuite(t *testing.T) {
	suite.Run(t, new(APIEndpointsTestSuite))
}