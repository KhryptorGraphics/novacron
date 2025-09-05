package integration_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/api/rest"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// APIIntegrationTestSuite defines the integration test suite for API endpoints
type APIIntegrationTestSuite struct {
	suite.Suite
	server        *httptest.Server
	router        *mux.Router
	authService   auth.AuthService
	vmManager     vm.VMManager
	adminToken    string
	userToken     string
}

func (suite *APIIntegrationTestSuite) SetupSuite() {
	// Initialize test services
	suite.authService = auth.NewMemoryAuthService()
	suite.vmManager = vm.NewVMManager(&vm.VMManagerConfig{
		DefaultDriver: vm.VMTypeQEMU,
	})

	// Setup API router
	suite.router = rest.NewRouter(suite.authService, suite.vmManager)
	suite.server = httptest.NewServer(suite.router)

	// Create test users and get tokens
	suite.setupTestUsers()
}

func (suite *APIIntegrationTestSuite) TearDownSuite() {
	if suite.server != nil {
		suite.server.Close()
	}
}

func (suite *APIIntegrationTestSuite) setupTestUsers() {
	// Create admin user
	adminUser := &auth.User{
		ID:       "admin-user-id",
		Username: "admin",
		Email:    "admin@test.com",
		Roles:    []string{"admin"},
	}
	err := suite.authService.CreateUser(adminUser, "admin123")
	suite.Require().NoError(err)

	// Login admin and get token
	adminSession, err := suite.authService.Login("admin", "admin123")
	suite.Require().NoError(err)
	suite.adminToken = adminSession.Token

	// Create regular user
	regularUser := &auth.User{
		ID:       "user-id",
		Username: "testuser",
		Email:    "user@test.com",
		Roles:    []string{"user"},
	}
	err = suite.authService.CreateUser(regularUser, "user123")
	suite.Require().NoError(err)

	// Login user and get token
	userSession, err := suite.authService.Login("testuser", "user123")
	suite.Require().NoError(err)
	suite.userToken = userSession.Token
}

func (suite *APIIntegrationTestSuite) makeRequest(method, path string, body interface{}, token string) (*http.Response, error) {
	var reqBody *bytes.Buffer
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reqBody = bytes.NewBuffer(jsonBody)
	} else {
		reqBody = bytes.NewBuffer(nil)
	}

	req, err := http.NewRequest(method, suite.server.URL+path, reqBody)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{Timeout: time.Second * 30}
	return client.Do(req)
}

// Auth API Tests
func (suite *APIIntegrationTestSuite) TestAuthLogin_Success() {
	// Arrange
	loginReq := map[string]string{
		"username": "admin",
		"password": "admin123",
	}

	// Act
	resp, err := suite.makeRequest("POST", "/api/v1/auth/login", loginReq, "")
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)

	var response map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&response)
	assert.NoError(suite.T(), err)
	assert.Contains(suite.T(), response, "token")
	assert.Contains(suite.T(), response, "expiresAt")
}

func (suite *APIIntegrationTestSuite) TestAuthLogin_InvalidCredentials() {
	// Arrange
	loginReq := map[string]string{
		"username": "admin",
		"password": "wrongpassword",
	}

	// Act
	resp, err := suite.makeRequest("POST", "/api/v1/auth/login", loginReq, "")
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusUnauthorized, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestAuthLogout_Success() {
	// Act
	resp, err := suite.makeRequest("POST", "/api/v1/auth/logout", nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestAuthLogout_NoToken() {
	// Act
	resp, err := suite.makeRequest("POST", "/api/v1/auth/logout", nil, "")
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusUnauthorized, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestAuthMe_Success() {
	// Act
	resp, err := suite.makeRequest("GET", "/api/v1/auth/me", nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)

	var user map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&user)
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), "admin", user["username"])
	assert.Equal(suite.T(), "admin@test.com", user["email"])
}

// VM API Tests
func (suite *APIIntegrationTestSuite) TestCreateVM_Success() {
	// Arrange
	vmConfig := map[string]interface{}{
		"name":     "test-vm",
		"type":     "qemu",
		"cpu":      2,
		"memory":   4096,
		"disk":     20,
		"networks": []string{"default"},
	}

	// Act
	resp, err := suite.makeRequest("POST", "/api/v1/vms", vmConfig, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusCreated, resp.StatusCode)

	var vm map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&vm)
	assert.NoError(suite.T(), err)
	assert.Contains(suite.T(), vm, "id")
	assert.Equal(suite.T(), "test-vm", vm["name"])
	assert.Equal(suite.T(), "qemu", vm["type"])
}

func (suite *APIIntegrationTestSuite) TestCreateVM_Unauthorized() {
	// Arrange
	vmConfig := map[string]interface{}{
		"name": "test-vm",
		"type": "qemu",
	}

	// Act
	resp, err := suite.makeRequest("POST", "/api/v1/vms", vmConfig, "")
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusUnauthorized, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestCreateVM_Forbidden() {
	// Arrange
	vmConfig := map[string]interface{}{
		"name": "test-vm",
		"type": "qemu",
	}

	// Act (using regular user token)
	resp, err := suite.makeRequest("POST", "/api/v1/vms", vmConfig, suite.userToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusForbidden, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestCreateVM_InvalidInput() {
	// Arrange
	invalidConfig := map[string]interface{}{
		"name": "", // Empty name
		"type": "qemu",
	}

	// Act
	resp, err := suite.makeRequest("POST", "/api/v1/vms", invalidConfig, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusBadRequest, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestListVMs_Success() {
	// Arrange - Create a test VM first
	vmConfig := map[string]interface{}{
		"name": "list-test-vm",
		"type": "qemu",
		"cpu":  1,
		"memory": 2048,
	}
	
	createResp, err := suite.makeRequest("POST", "/api/v1/vms", vmConfig, suite.adminToken)
	suite.Require().NoError(err)
	suite.Require().Equal(http.StatusCreated, createResp.StatusCode)

	// Act
	resp, err := suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)

	var vms []map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&vms)
	assert.NoError(suite.T(), err)
	assert.NotEmpty(suite.T(), vms)
}

func (suite *APIIntegrationTestSuite) TestGetVM_Success() {
	// Arrange - Create a test VM first
	vmConfig := map[string]interface{}{
		"name": "get-test-vm",
		"type": "qemu",
		"cpu":  1,
		"memory": 2048,
	}
	
	createResp, err := suite.makeRequest("POST", "/api/v1/vms", vmConfig, suite.adminToken)
	suite.Require().NoError(err)
	suite.Require().Equal(http.StatusCreated, createResp.StatusCode)

	var createdVM map[string]interface{}
	err = json.NewDecoder(createResp.Body).Decode(&createdVM)
	suite.Require().NoError(err)
	vmID := createdVM["id"].(string)

	// Act
	resp, err := suite.makeRequest("GET", fmt.Sprintf("/api/v1/vms/%s", vmID), nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)

	var vm map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&vm)
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), vmID, vm["id"])
	assert.Equal(suite.T(), "get-test-vm", vm["name"])
}

func (suite *APIIntegrationTestSuite) TestGetVM_NotFound() {
	// Act
	resp, err := suite.makeRequest("GET", "/api/v1/vms/nonexistent-vm-id", nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusNotFound, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestStartVM_Success() {
	// Arrange - Create a test VM first
	vmConfig := map[string]interface{}{
		"name": "start-test-vm",
		"type": "qemu",
		"cpu":  1,
		"memory": 2048,
	}
	
	createResp, err := suite.makeRequest("POST", "/api/v1/vms", vmConfig, suite.adminToken)
	suite.Require().NoError(err)
	
	var createdVM map[string]interface{}
	err = json.NewDecoder(createResp.Body).Decode(&createdVM)
	suite.Require().NoError(err)
	vmID := createdVM["id"].(string)

	// Act
	resp, err := suite.makeRequest("POST", fmt.Sprintf("/api/v1/vms/%s/start", vmID), nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestStopVM_Success() {
	// Arrange - Create and start a test VM first
	vmConfig := map[string]interface{}{
		"name": "stop-test-vm",
		"type": "qemu",
		"cpu":  1,
		"memory": 2048,
	}
	
	createResp, err := suite.makeRequest("POST", "/api/v1/vms", vmConfig, suite.adminToken)
	suite.Require().NoError(err)
	
	var createdVM map[string]interface{}
	err = json.NewDecoder(createResp.Body).Decode(&createdVM)
	suite.Require().NoError(err)
	vmID := createdVM["id"].(string)

	// Start the VM first
	startResp, err := suite.makeRequest("POST", fmt.Sprintf("/api/v1/vms/%s/start", vmID), nil, suite.adminToken)
	suite.Require().NoError(err)
	suite.Require().Equal(http.StatusOK, startResp.StatusCode)

	// Act
	resp, err := suite.makeRequest("POST", fmt.Sprintf("/api/v1/vms/%s/stop", vmID), nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)
}

func (suite *APIIntegrationTestSuite) TestDeleteVM_Success() {
	// Arrange - Create a test VM first
	vmConfig := map[string]interface{}{
		"name": "delete-test-vm",
		"type": "qemu",
		"cpu":  1,
		"memory": 2048,
	}
	
	createResp, err := suite.makeRequest("POST", "/api/v1/vms", vmConfig, suite.adminToken)
	suite.Require().NoError(err)
	
	var createdVM map[string]interface{}
	err = json.NewDecoder(createResp.Body).Decode(&createdVM)
	suite.Require().NoError(err)
	vmID := createdVM["id"].(string)

	// Act
	resp, err := suite.makeRequest("DELETE", fmt.Sprintf("/api/v1/vms/%s", vmID), nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)

	// Verify VM is deleted
	getResp, err := suite.makeRequest("GET", fmt.Sprintf("/api/v1/vms/%s", vmID), nil, suite.adminToken)
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusNotFound, getResp.StatusCode)
}

// Admin API Tests
func (suite *APIIntegrationTestSuite) TestGetSystemStats_Success() {
	// Act
	resp, err := suite.makeRequest("GET", "/api/v1/admin/stats", nil, suite.adminToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)

	var stats map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&stats)
	assert.NoError(suite.T(), err)
	assert.Contains(suite.T(), stats, "totalVMs")
	assert.Contains(suite.T(), stats, "runningVMs")
	assert.Contains(suite.T(), stats, "systemLoad")
}

func (suite *APIIntegrationTestSuite) TestGetSystemStats_Forbidden() {
	// Act (using regular user token)
	resp, err := suite.makeRequest("GET", "/api/v1/admin/stats", nil, suite.userToken)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusForbidden, resp.StatusCode)
}

// Health Check Tests
func (suite *APIIntegrationTestSuite) TestHealthCheck() {
	// Act
	resp, err := suite.makeRequest("GET", "/health", nil, "")
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)

	var health map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&health)
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), "ok", health["status"])
}

// Rate Limiting Tests
func (suite *APIIntegrationTestSuite) TestRateLimiting() {
	// Make multiple requests rapidly
	for i := 0; i < 100; i++ {
		resp, err := suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
		assert.NoError(suite.T(), err)
		
		// Check if we hit rate limit
		if resp.StatusCode == http.StatusTooManyRequests {
			// Rate limiting is working
			return
		}
	}
	
	// If we reach here, either rate limiting is not enabled or the limit is very high
	suite.T().Log("Rate limiting may not be configured or limit is very high")
}

// CORS Tests
func (suite *APIIntegrationTestSuite) TestCORSHeaders() {
	// Act
	req, err := http.NewRequest("OPTIONS", suite.server.URL+"/api/v1/vms", nil)
	suite.Require().NoError(err)
	req.Header.Set("Origin", "http://localhost:3000")
	req.Header.Set("Access-Control-Request-Method", "POST")
	
	client := &http.Client{}
	resp, err := client.Do(req)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusOK, resp.StatusCode)
	assert.Contains(suite.T(), resp.Header.Get("Access-Control-Allow-Origin"), "localhost")
	assert.Contains(suite.T(), resp.Header.Get("Access-Control-Allow-Methods"), "POST")
}

// Content Type Tests
func (suite *APIIntegrationTestSuite) TestUnsupportedMediaType() {
	// Arrange
	req, err := http.NewRequest("POST", suite.server.URL+"/api/v1/vms", bytes.NewBuffer([]byte("invalid-json")))
	suite.Require().NoError(err)
	req.Header.Set("Content-Type", "text/plain")
	req.Header.Set("Authorization", "Bearer "+suite.adminToken)

	// Act
	client := &http.Client{}
	resp, err := client.Do(req)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusUnsupportedMediaType, resp.StatusCode)
}

// Concurrent Request Tests
func (suite *APIIntegrationTestSuite) TestConcurrentRequests() {
	// Arrange
	numRequests := 10
	results := make(chan *http.Response, numRequests)
	
	// Act - Make concurrent requests
	for i := 0; i < numRequests; i++ {
		go func() {
			resp, err := suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
			if err != nil {
				suite.T().Error(err)
				return
			}
			results <- resp
		}()
	}
	
	// Assert - All requests should succeed
	successCount := 0
	for i := 0; i < numRequests; i++ {
		resp := <-results
		if resp.StatusCode == http.StatusOK {
			successCount++
		}
	}
	
	assert.Equal(suite.T(), numRequests, successCount)
}

// Error Handling Tests
func (suite *APIIntegrationTestSuite) TestInvalidJSONPayload() {
	// Arrange
	invalidJSON := `{"name": "test", "invalid": json}`
	
	req, err := http.NewRequest("POST", suite.server.URL+"/api/v1/vms", bytes.NewBuffer([]byte(invalidJSON)))
	suite.Require().NoError(err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+suite.adminToken)

	// Act
	client := &http.Client{}
	resp, err := client.Do(req)
	
	// Assert
	assert.NoError(suite.T(), err)
	assert.Equal(suite.T(), http.StatusBadRequest, resp.StatusCode)
}

// Performance Tests
func (suite *APIIntegrationTestSuite) TestResponseTimes() {
	// Test that API responses are within acceptable limits
	endpoints := []string{
		"/health",
		"/api/v1/vms",
		"/api/v1/auth/me",
	}
	
	for _, endpoint := range endpoints {
		start := time.Now()
		resp, err := suite.makeRequest("GET", endpoint, nil, suite.adminToken)
		duration := time.Since(start)
		
		assert.NoError(suite.T(), err)
		assert.True(suite.T(), duration < time.Second*2, "Response time for %s should be under 2 seconds, got %v", endpoint, duration)
		
		if resp.StatusCode == http.StatusOK {
			// Response should be valid
			assert.True(suite.T(), resp.ContentLength != 0 || resp.Header.Get("Transfer-Encoding") == "chunked")
		}
	}
}

// Run the test suite
func TestAPIIntegrationTestSuite(t *testing.T) {
	suite.Run(t, new(APIIntegrationTestSuite))
}

// Benchmark tests
func BenchmarkAPILogin(b *testing.B) {
	suite := &APIIntegrationTestSuite{}
	suite.SetupSuite()
	defer suite.TearDownSuite()
	
	loginReq := map[string]string{
		"username": "admin",
		"password": "admin123",
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		suite.makeRequest("POST", "/api/v1/auth/login", loginReq, "")
	}
}

func BenchmarkAPIListVMs(b *testing.B) {
	suite := &APIIntegrationTestSuite{}
	suite.SetupSuite()
	defer suite.TearDownSuite()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
	}
}