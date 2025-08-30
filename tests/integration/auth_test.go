package integration

import (
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/tests/integration/helpers"
)

// AuthTestSuite tests authentication and authorization functionality
type AuthTestSuite struct {
	suite.Suite
	env     *helpers.TestEnvironment
	mockGen *helpers.MockDataGenerator
}

// SetupSuite initializes the test suite
func (suite *AuthTestSuite) SetupSuite() {
	suite.env = helpers.NewTestEnvironment(suite.T())
	suite.env.Setup(suite.T())
	suite.mockGen = helpers.NewMockDataGenerator()
}

// TearDownSuite cleans up the test suite
func (suite *AuthTestSuite) TearDownSuite() {
	if suite.env != nil {
		suite.env.Cleanup(suite.T())
	}
}

// TestUserRegistration tests user registration functionality
func (suite *AuthTestSuite) TestUserRegistration() {
	tests := []struct {
		name         string
		registerData map[string]interface{}
		wantCode     int
		wantErr      bool
	}{
		{
			name: "Valid user registration",
			registerData: map[string]interface{}{
				"email":     "newuser@test.com",
				"password":  "SecurePass123!",
				"name":      "New User",
				"tenant_id": "tenant-1",
			},
			wantCode: http.StatusCreated,
			wantErr:  false,
		},
		{
			name: "Registration with weak password",
			registerData: map[string]interface{}{
				"email":     "weakpass@test.com",
				"password":  "123", // Too weak
				"name":      "Weak Pass User",
				"tenant_id": "tenant-1",
			},
			wantCode: http.StatusBadRequest,
			wantErr:  true,
		},
		{
			name: "Registration with invalid email",
			registerData: map[string]interface{}{
				"email":     "invalid-email", // Invalid format
				"password":  "SecurePass123!",
				"name":      "Invalid Email User",
				"tenant_id": "tenant-1",
			},
			wantCode: http.StatusBadRequest,
			wantErr:  true,
		},
		{
			name: "Registration with missing fields",
			registerData: map[string]interface{}{
				"email": "incomplete@test.com",
				// Missing password and other fields
			},
			wantCode: http.StatusBadRequest,
			wantErr:  true,
		},
		{
			name: "Duplicate email registration",
			registerData: map[string]interface{}{
				"email":     "admin@test.com", // Already exists
				"password":  "AnotherPass123!",
				"name":      "Duplicate User",
				"tenant_id": "tenant-1",
			},
			wantCode: http.StatusConflict,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			resp := suite.env.APIClient.POST(t, "/api/auth/register", tt.registerData)
			defer resp.Body.Close()

			assert.Equal(t, tt.wantCode, resp.StatusCode)

			if !tt.wantErr {
				var result map[string]interface{}
				suite.env.APIClient.ParseJSON(t, resp, &result)

				assert.NotEmpty(t, result["id"], "User ID should not be empty")
				assert.Equal(t, tt.registerData["email"], result["email"])
				assert.Equal(t, tt.registerData["name"], result["name"])
				assert.NotContains(t, result, "password", "Password should not be returned")
			}
		})
	}
}

// TestUserLogin tests login functionality
func (suite *AuthTestSuite) TestUserLogin() {
	tests := []struct {
		name      string
		loginData map[string]interface{}
		wantCode  int
		wantErr   bool
	}{
		{
			name: "Valid admin login",
			loginData: map[string]interface{}{
				"email":    "admin@test.com",
				"password": "admin123",
			},
			wantCode: http.StatusOK,
			wantErr:  false,
		},
		{
			name: "Valid user login",
			loginData: map[string]interface{}{
				"email":    "user@test.com",
				"password": "user123",
			},
			wantCode: http.StatusOK,
			wantErr:  false,
		},
		{
			name: "Invalid password",
			loginData: map[string]interface{}{
				"email":    "admin@test.com",
				"password": "wrongpassword",
			},
			wantCode: http.StatusUnauthorized,
			wantErr:  true,
		},
		{
			name: "Non-existent user",
			loginData: map[string]interface{}{
				"email":    "nonexistent@test.com",
				"password": "anypassword",
			},
			wantCode: http.StatusUnauthorized,
			wantErr:  true,
		},
		{
			name: "Missing credentials",
			loginData: map[string]interface{}{
				"email": "admin@test.com",
				// Missing password
			},
			wantCode: http.StatusBadRequest,
			wantErr:  true,
		},
		{
			name: "Empty credentials",
			loginData: map[string]interface{}{
				"email":    "",
				"password": "",
			},
			wantCode: http.StatusBadRequest,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			resp := suite.env.APIClient.POST(t, "/api/auth/login", tt.loginData)
			defer resp.Body.Close()

			assert.Equal(t, tt.wantCode, resp.StatusCode)

			if !tt.wantErr {
				var result map[string]interface{}
				suite.env.APIClient.ParseJSON(t, resp, &result)

				// Verify token structure
				token, ok := result["token"].(string)
				assert.True(t, ok, "Token should be a string")
				assert.NotEmpty(t, token, "Token should not be empty")
				assert.True(t, strings.HasPrefix(token, "eyJ"), "Token should be a JWT")

				// Verify user info
				user, ok := result["user"].(map[string]interface{})
				assert.True(t, ok, "User info should be present")
				assert.Equal(t, tt.loginData["email"], user["email"])
				assert.NotContains(t, user, "password", "Password should not be returned")

				// Verify expiration
				expiresIn, ok := result["expires_in"].(float64)
				assert.True(t, ok, "Expires in should be present")
				assert.Greater(t, expiresIn, float64(0), "Token should have positive expiration")
			}
		})
	}
}

// TestTokenValidation tests JWT token validation
func (suite *AuthTestSuite) TestTokenValidation() {
	// First, login to get a valid token
	token := suite.env.LoginAsAdmin(suite.T())
	
	tests := []struct {
		name     string
		token    string
		endpoint string
		wantCode int
	}{
		{
			name:     "Valid token",
			token:    token,
			endpoint: "/api/user/profile",
			wantCode: http.StatusOK,
		},
		{
			name:     "No token",
			token:    "",
			endpoint: "/api/user/profile",
			wantCode: http.StatusUnauthorized,
		},
		{
			name:     "Invalid token format",
			token:    "invalid-token",
			endpoint: "/api/user/profile",
			wantCode: http.StatusUnauthorized,
		},
		{
			name:     "Expired token (simulated)",
			token:    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZXhwIjoxfQ.invalid",
			endpoint: "/api/user/profile",
			wantCode: http.StatusUnauthorized,
		},
	}

	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			// Create a fresh client for each test
			client := helpers.NewAPIClient(suite.env.Config.APIURL, "test-api-key")
			if tt.token != "" {
				client.SetAuthToken(tt.token)
			}

			resp := client.GET(t, tt.endpoint)
			defer resp.Body.Close()

			assert.Equal(t, tt.wantCode, resp.StatusCode)

			if tt.wantCode == http.StatusOK {
				var profile map[string]interface{}
				client.ParseJSON(t, resp, &profile)
				assert.NotEmpty(t, profile["email"], "Profile should contain email")
			}
		})
	}
}

// TestRoleBasedAccess tests role-based access control
func (suite *AuthTestSuite) TestRoleBasedAccess() {
	// Get tokens for different user roles
	adminToken := suite.env.LoginAsAdmin(suite.T())
	userToken := suite.env.LoginAsUser(suite.T())

	tests := []struct {
		name      string
		token     string
		endpoint  string
		method    string
		body      interface{}
		wantCode  int
		role      string
	}{
		// Admin-only endpoints
		{
			name:     "Admin can access admin endpoints",
			token:    adminToken,
			endpoint: "/api/admin/users",
			method:   "GET",
			wantCode: http.StatusOK,
			role:     "admin",
		},
		{
			name:     "User cannot access admin endpoints",
			token:    userToken,
			endpoint: "/api/admin/users",
			method:   "GET",
			wantCode: http.StatusForbidden,
			role:     "user",
		},
		{
			name:     "Admin can create users",
			token:    adminToken,
			endpoint: "/api/admin/users",
			method:   "POST",
			body: map[string]interface{}{
				"email":     "newadminuser@test.com",
				"password":  "SecurePass123!",
				"name":      "New Admin User",
				"role":      "user",
				"tenant_id": "tenant-1",
			},
			wantCode: http.StatusCreated,
			role:     "admin",
		},
		{
			name:     "User cannot create users",
			token:    userToken,
			endpoint: "/api/admin/users",
			method:   "POST",
			body: map[string]interface{}{
				"email":     "shouldnotwork@test.com",
				"password":  "SecurePass123!",
				"name":      "Should Not Work",
				"role":      "user",
				"tenant_id": "tenant-1",
			},
			wantCode: http.StatusForbidden,
			role:     "user",
		},
		// User-accessible endpoints
		{
			name:     "Admin can access user endpoints",
			token:    adminToken,
			endpoint: "/api/user/profile",
			method:   "GET",
			wantCode: http.StatusOK,
			role:     "admin",
		},
		{
			name:     "User can access own profile",
			token:    userToken,
			endpoint: "/api/user/profile",
			method:   "GET",
			wantCode: http.StatusOK,
			role:     "user",
		},
		// VM operations based on role
		{
			name:     "Admin can delete any VM",
			token:    adminToken,
			endpoint: "/api/vms/test-vm-id",
			method:   "DELETE",
			wantCode: http.StatusNotFound, // VM doesn't exist, but permission is OK
			role:     "admin",
		},
		{
			name:     "User can create VMs",
			token:    userToken,
			endpoint: "/api/vms",
			method:   "POST",
			body: map[string]interface{}{
				"name":      "user-vm",
				"cpu":       2,
				"memory":    1024,
				"disk_size": 10240,
				"image":     "ubuntu:20.04",
				"tenant_id": "tenant-1",
			},
			wantCode: http.StatusCreated,
			role:     "user",
		},
	}

	for _, tt := range tests {
		suite.T().Run(fmt.Sprintf("%s_%s", tt.role, tt.name), func(t *testing.T) {
			client := helpers.NewAPIClient(suite.env.Config.APIURL, "test-api-key")
			client.SetAuthToken(tt.token)

			var resp *http.Response
			switch tt.method {
			case "GET":
				resp = client.GET(t, tt.endpoint)
			case "POST":
				resp = client.POST(t, tt.endpoint, tt.body)
			case "DELETE":
				resp = client.DELETE(t, tt.endpoint)
			}
			defer resp.Body.Close()

			assert.Equal(t, tt.wantCode, resp.StatusCode,
				"Role %s should get status %d for %s", tt.role, tt.wantCode, tt.name)
		})
	}
}

// TestMultiTenantIsolation tests tenant isolation
func (suite *AuthTestSuite) TestMultiTenantIsolation() {
	// Create VMs in different tenants
	adminToken := suite.env.LoginAsAdmin(suite.T())
	client := helpers.NewAPIClient(suite.env.Config.APIURL, "test-api-key")
	client.SetAuthToken(adminToken)

	// Create VM in tenant-1
	vm1Data := map[string]interface{}{
		"name":      "tenant1-vm",
		"cpu":       2,
		"memory":    1024,
		"disk_size": 10240,
		"image":     "ubuntu:20.04",
		"tenant_id": "tenant-1",
	}
	resp1 := client.POST(suite.T(), "/api/vms", vm1Data)
	defer resp1.Body.Close()
	client.ExpectStatus(suite.T(), resp1, http.StatusCreated)

	var vm1Result map[string]interface{}
	client.ParseJSON(suite.T(), resp1, &vm1Result)
	vm1ID := vm1Result["id"].(string)

	// Create VM in tenant-2
	vm2Data := map[string]interface{}{
		"name":      "tenant2-vm",
		"cpu":       2,
		"memory":    1024,
		"disk_size": 10240,
		"image":     "ubuntu:20.04",
		"tenant_id": "tenant-2",
	}
	resp2 := client.POST(suite.T(), "/api/vms", vm2Data)
	defer resp2.Body.Close()
	client.ExpectStatus(suite.T(), resp2, http.StatusCreated)

	var vm2Result map[string]interface{}
	client.ParseJSON(suite.T(), resp2, &vm2Result)
	vm2ID := vm2Result["id"].(string)

	// Test tenant isolation
	tests := []struct {
		name       string
		userEmail  string
		password   string
		expectedVMs []string
		tenantID   string
	}{
		{
			name:       "User from tenant-1 sees only tenant-1 VMs",
			userEmail:  "user@test.com",     // tenant-1 user
			password:   "user123",
			expectedVMs: []string{vm1ID},
			tenantID:   "tenant-1",
		},
		{
			name:       "User from tenant-2 sees only tenant-2 VMs", 
			userEmail:  "tenant2@test.com",  // tenant-2 user
			password:   "tenant123",
			expectedVMs: []string{vm2ID},
			tenantID:   "tenant-2",
		},
	}

	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			// Login as tenant user
			tenantClient := helpers.NewAPIClient(suite.env.Config.APIURL, "test-api-key")
			token := tenantClient.Login(t, tt.userEmail, tt.password)
			tenantClient.SetAuthToken(token)

			// List VMs
			resp := tenantClient.GET(t, "/api/vms")
			defer resp.Body.Close()
			tenantClient.ExpectStatus(t, resp, http.StatusOK)

			var result map[string]interface{}
			tenantClient.ParseJSON(t, resp, &result)

			vms, ok := result["vms"].([]interface{})
			require.True(t, ok, "Response should contain vms array")

			// Extract VM IDs from response
			var vmIDs []string
			for _, vm := range vms {
				vmMap := vm.(map[string]interface{})
				vmIDs = append(vmIDs, vmMap["id"].(string))
				
				// Verify tenant_id matches
				assert.Equal(t, tt.tenantID, vmMap["tenant_id"], 
					"VM should belong to correct tenant")
			}

			// Verify user sees only VMs from their tenant
			for _, expectedVMID := range tt.expectedVMs {
				assert.Contains(t, vmIDs, expectedVMID, 
					"User should see VM %s from their tenant", expectedVMID)
			}
		})
	}

	// Cleanup
	client.DELETE(suite.T(), "/api/vms/"+vm1ID)
	client.DELETE(suite.T(), "/api/vms/"+vm2ID)
}

// TestPasswordSecurity tests password security requirements
func (suite *AuthTestSuite) TestPasswordSecurity() {
	tests := []struct {
		name      string
		password  string
		wantValid bool
	}{
		{"Strong password", "SecurePass123!", true},
		{"With numbers and symbols", "Password1@", true},
		{"Too short", "Pass1!", false},
		{"No uppercase", "password123!", false},
		{"No lowercase", "PASSWORD123!", false},
		{"No numbers", "Password!", false},
		{"No symbols", "Password123", false},
		{"Common password", "password123", false},
		{"Only letters", "PasswordOnly", false},
	}

	for _, tt := range tests {
		suite.T().Run(tt.name, func(t *testing.T) {
			userData := map[string]interface{}{
				"email":     fmt.Sprintf("pwtest-%d@test.com", time.Now().UnixNano()),
				"password":  tt.password,
				"name":      "Password Test User",
				"tenant_id": "tenant-1",
			}

			resp := suite.env.APIClient.POST(t, "/api/auth/register", userData)
			defer resp.Body.Close()

			if tt.wantValid {
				assert.Equal(t, http.StatusCreated, resp.StatusCode,
					"Password '%s' should be accepted", tt.password)
			} else {
				assert.Equal(t, http.StatusBadRequest, resp.StatusCode,
					"Password '%s' should be rejected", tt.password)
			}
		})
	}
}

// TestTokenExpiration tests token expiration handling
func (suite *AuthTestSuite) TestTokenExpiration() {
	if testing.Short() {
		suite.T().Skip("Skipping token expiration test in short mode")
	}

	// This test would require modifying token expiration time for testing
	// or using a mock time service. For now, we'll test the basic structure.
	
	suite.T().Run("Token contains expiration", func(t *testing.T) {
		loginData := map[string]interface{}{
			"email":    "admin@test.com",
			"password": "admin123",
		}

		resp := suite.env.APIClient.POST(t, "/api/auth/login", loginData)
		defer resp.Body.Close()
		suite.env.APIClient.ExpectStatus(t, resp, http.StatusOK)

		var result map[string]interface{}
		suite.env.APIClient.ParseJSON(t, resp, &result)

		expiresIn, ok := result["expires_in"].(float64)
		assert.True(t, ok, "Token should have expires_in field")
		assert.Greater(t, expiresIn, float64(0), "Token expiration should be positive")
		assert.LessOrEqual(t, expiresIn, float64(24*3600), "Token should expire within 24 hours")
	})
}

// TestLogout tests user logout functionality
func (suite *AuthTestSuite) TestLogout() {
	// Login first
	token := suite.env.LoginAsAdmin(suite.T())
	client := helpers.NewAPIClient(suite.env.Config.APIURL, "test-api-key")
	client.SetAuthToken(token)

	// Verify token works
	resp := client.GET(suite.T(), "/api/user/profile")
	defer resp.Body.Close()
	client.ExpectStatus(suite.T(), resp, http.StatusOK)

	// Logout
	suite.T().Run("Successful logout", func(t *testing.T) {
		logoutResp := client.POST(t, "/api/auth/logout", nil)
		defer logoutResp.Body.Close()
		
		// Logout should succeed
		assert.True(t, logoutResp.StatusCode == http.StatusOK || logoutResp.StatusCode == http.StatusNoContent,
			"Logout should succeed")
	})

	// Note: In a real implementation, you might want to test that the token
	// is invalidated after logout, but this depends on your token management strategy
}

// TestAuthTestSuite runs the authentication test suite
func TestAuthTestSuite(t *testing.T) {
	suite.Run(t, new(AuthTestSuite))
}