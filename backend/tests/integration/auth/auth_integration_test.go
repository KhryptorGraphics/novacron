package auth_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/tests/integration"
)

// AuthIntegrationTestSuite tests authentication flow integration
type AuthIntegrationTestSuite struct {
	integration.IntegrationTestSuite
}

// TestAuthenticationFlow tests the complete authentication workflow
func (suite *AuthIntegrationTestSuite) TestAuthenticationFlow() {
	suite.T().Log("Testing complete authentication flow...")

	// Test data
	testUser := map[string]interface{}{
		"username": "testuser_auth",
		"email":    "testuser@example.com",
		"password": "SecurePassword123!",
		"tenant_id": "test-tenant",
	}

	// Step 1: User Registration
	suite.T().Run("UserRegistration", func(t *testing.T) {
		suite.testUserRegistration(t, testUser)
	})

	// Step 2: User Login
	var loginToken string
	suite.T().Run("UserLogin", func(t *testing.T) {
		loginToken = suite.testUserLogin(t, testUser)
	})

	// Step 3: Token Validation
	suite.T().Run("TokenValidation", func(t *testing.T) {
		suite.testTokenValidation(t, loginToken)
	})

	// Step 4: Protected Endpoint Access
	suite.T().Run("ProtectedEndpointAccess", func(t *testing.T) {
		suite.testProtectedEndpointAccess(t, loginToken)
	})

	// Step 5: Token Expiration Handling
	suite.T().Run("TokenExpiration", func(t *testing.T) {
		suite.testTokenExpiration(t)
	})

	// Step 6: User Logout
	suite.T().Run("UserLogout", func(t *testing.T) {
		suite.testUserLogout(t, loginToken)
	})

	suite.T().Log("✓ Complete authentication flow tested successfully")
}

// testUserRegistration tests user registration
func (suite *AuthIntegrationTestSuite) testUserRegistration(t *testing.T, userData map[string]interface{}) {
	// Register authentication route
	suite.GetRouter().HandleFunc("/auth/register", func(w http.ResponseWriter, r *http.Request) {
		var registerReq map[string]string
		err := json.NewDecoder(r.Body).Decode(&registerReq)
		if err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		// Validate required fields
		if registerReq["username"] == "" || registerReq["email"] == "" || registerReq["password"] == "" {
			http.Error(w, "Missing required fields", http.StatusBadRequest)
			return
		}

		// Create user using auth manager
		tenantID := registerReq["tenant_id"]
		if tenantID == "" {
			tenantID = "default"
		}

		user, err := suite.GetAuthManager().CreateUser(
			registerReq["username"],
			registerReq["email"],
			registerReq["password"],
			"user",
			tenantID,
		)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to create user: %v", err), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		response := map[string]interface{}{
			"user": map[string]interface{}{
				"id":        user.ID,
				"username":  user.Username,
				"email":     user.Email,
				"tenant_id": user.TenantID,
			},
			"message": "User created successfully",
		}
		json.NewEncoder(w).Encode(response)
	}).Methods("POST")

	// Prepare registration request
	jsonData, err := json.Marshal(userData)
	require.NoError(t, err, "Failed to marshal user data")

	// Send registration request
	resp, err := http.Post(
		fmt.Sprintf("%s/auth/register", suite.GetServer().URL),
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	require.NoError(t, err, "Failed to send registration request")
	defer resp.Body.Close()

	// Verify registration response
	assert.Equal(t, http.StatusCreated, resp.StatusCode, "Registration should succeed")

	var response map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&response)
	require.NoError(t, err, "Failed to decode registration response")

	// Verify response structure
	assert.Contains(t, response, "user", "Response should contain user data")
	assert.Contains(t, response, "message", "Response should contain success message")

	userInfo := response["user"].(map[string]interface{})
	assert.Equal(t, userData["username"], userInfo["username"], "Username should match")
	assert.Equal(t, userData["email"], userInfo["email"], "Email should match")
	assert.Equal(t, userData["tenant_id"], userInfo["tenant_id"], "Tenant ID should match")

	t.Log("✓ User registration successful")
}

// testUserLogin tests user login
func (suite *AuthIntegrationTestSuite) testUserLogin(t *testing.T, userData map[string]interface{}) string {
	// Register login route
	suite.GetRouter().HandleFunc("/auth/login", func(w http.ResponseWriter, r *http.Request) {
		var loginReq map[string]string
		err := json.NewDecoder(r.Body).Decode(&loginReq)
		if err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		// Authenticate user
		user, token, err := suite.GetAuthManager().Authenticate(loginReq["username"], loginReq["password"])
		if err != nil {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"token": token,
			"user": map[string]interface{}{
				"id":        user.ID,
				"username":  user.Username,
				"email":     user.Email,
				"tenant_id": user.TenantID,
			},
			"expires_in": 3600,
		}

		// Add role information if available
		if len(user.Roles) > 0 {
			response["user"].(map[string]interface{})["role"] = user.Roles[0].Name
		}

		json.NewEncoder(w).Encode(response)
	}).Methods("POST")

	// Prepare login request
	loginData := map[string]string{
		"username": userData["username"].(string),
		"password": userData["password"].(string),
	}

	jsonData, err := json.Marshal(loginData)
	require.NoError(t, err, "Failed to marshal login data")

	// Send login request
	resp, err := http.Post(
		fmt.Sprintf("%s/auth/login", suite.GetServer().URL),
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	require.NoError(t, err, "Failed to send login request")
	defer resp.Body.Close()

	// Verify login response
	assert.Equal(t, http.StatusOK, resp.StatusCode, "Login should succeed")

	var response map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&response)
	require.NoError(t, err, "Failed to decode login response")

	// Verify response structure
	assert.Contains(t, response, "token", "Response should contain JWT token")
	assert.Contains(t, response, "user", "Response should contain user data")
	assert.Contains(t, response, "expires_in", "Response should contain expiration time")

	token := response["token"].(string)
	assert.NotEmpty(t, token, "Token should not be empty")

	// Verify JWT token structure
	parts := strings.Split(token, ".")
	assert.Equal(t, 3, len(parts), "JWT should have 3 parts")

	userInfo := response["user"].(map[string]interface{})
	assert.Equal(t, userData["username"], userInfo["username"], "Username should match")

	t.Log("✓ User login successful")
	return token
}

// testTokenValidation tests JWT token validation
func (suite *AuthIntegrationTestSuite) testTokenValidation(t *testing.T, token string) {
	// Register validation route
	suite.GetRouter().HandleFunc("/auth/validate", func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "Invalid or missing token", http.StatusUnauthorized)
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")

		// Parse and validate JWT token
		jwtToken, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, jwt.ErrSignatureInvalid
			}
			return []byte(suite.GetAuthManager().GetJWTSecret()), nil
		})

		if err != nil || !jwtToken.Valid {
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		claims, ok := jwtToken.Claims.(jwt.MapClaims)
		if !ok {
			http.Error(w, "Invalid token claims", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"valid": true,
			"user": map[string]interface{}{
				"id":        claims["user_id"],
				"username":  claims["username"],
				"email":     claims["email"],
				"role":      claims["role"],
				"tenant_id": claims["tenant_id"],
			},
		})
	}).Methods("GET")

	// Create validation request
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/auth/validate", suite.GetServer().URL), nil)
	require.NoError(t, err, "Failed to create validation request")
	req.Header.Set("Authorization", "Bearer "+token)

	// Send validation request
	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err, "Failed to send validation request")
	defer resp.Body.Close()

	// Verify validation response
	assert.Equal(t, http.StatusOK, resp.StatusCode, "Token validation should succeed")

	var response map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&response)
	require.NoError(t, err, "Failed to decode validation response")

	assert.True(t, response["valid"].(bool), "Token should be valid")
	assert.Contains(t, response, "user", "Response should contain user data")

	t.Log("✓ Token validation successful")
}

// testProtectedEndpointAccess tests accessing protected endpoints
func (suite *AuthIntegrationTestSuite) testProtectedEndpointAccess(t *testing.T, token string) {
	// Register a protected route
	suite.GetRouter().HandleFunc("/api/protected", func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "Access granted to protected resource",
			"data":    "sensitive information",
		})
	}).Methods("GET")

	// Test with valid token
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/api/protected", suite.GetServer().URL), nil)
	require.NoError(t, err, "Failed to create protected request")
	req.Header.Set("Authorization", "Bearer "+token)

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err, "Failed to send protected request")
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode, "Protected endpoint access should succeed with valid token")

	// Test without token
	req2, err := http.NewRequest("GET", fmt.Sprintf("%s/api/protected", suite.GetServer().URL), nil)
	require.NoError(t, err, "Failed to create unauth request")

	resp2, err := client.Do(req2)
	require.NoError(t, err, "Failed to send unauth request")
	defer resp2.Body.Close()

	assert.Equal(t, http.StatusUnauthorized, resp2.StatusCode, "Protected endpoint should reject request without token")

	// Test with invalid token
	req3, err := http.NewRequest("GET", fmt.Sprintf("%s/api/protected", suite.GetServer().URL), nil)
	require.NoError(t, err, "Failed to create invalid token request")
	req3.Header.Set("Authorization", "Bearer invalid-token")

	resp3, err := client.Do(req3)
	require.NoError(t, err, "Failed to send invalid token request")
	defer resp3.Body.Close()

	assert.Equal(t, http.StatusUnauthorized, resp3.StatusCode, "Protected endpoint should reject invalid token")

	t.Log("✓ Protected endpoint access control working")
}

// testTokenExpiration tests token expiration handling
func (suite *AuthIntegrationTestSuite) testTokenExpiration(t *testing.T) {
	// Create a short-lived token for testing
	claims := jwt.MapClaims{
		"user_id":   1,
		"username":  "testuser",
		"email":     "test@example.com",
		"role":      "user",
		"tenant_id": "default",
		"exp":       time.Now().Add(1 * time.Second).Unix(), // Expires in 1 second
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString([]byte(suite.GetAuthManager().GetJWTSecret()))
	require.NoError(t, err, "Failed to create test token")

	// Wait for token to expire
	time.Sleep(2 * time.Second)

	// Register expired token test route
	suite.GetRouter().HandleFunc("/auth/validate-expired", func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "Invalid or missing token", http.StatusUnauthorized)
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")

		// Parse token (will fail due to expiration)
		_, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			return []byte(suite.GetAuthManager().GetJWTSecret()), nil
		})

		if err != nil {
			http.Error(w, "Token expired or invalid", http.StatusUnauthorized)
			return
		}

		w.WriteHeader(http.StatusOK)
	}).Methods("GET")

	// Test expired token
	req, err := http.NewRequest("GET", fmt.Sprintf("%s/auth/validate-expired", suite.GetServer().URL), nil)
	require.NoError(t, err, "Failed to create expired token request")
	req.Header.Set("Authorization", "Bearer "+tokenString)

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err, "Failed to send expired token request")
	defer resp.Body.Close()

	assert.Equal(t, http.StatusUnauthorized, resp.StatusCode, "Expired token should be rejected")

	t.Log("✓ Token expiration handling working")
}

// testUserLogout tests user logout
func (suite *AuthIntegrationTestSuite) testUserLogout(t *testing.T, token string) {
	// Register logout route
	suite.GetRouter().HandleFunc("/auth/logout", func(w http.ResponseWriter, r *http.Request) {
		// In a real implementation, we might invalidate the token in a blacklist
		// For now, we just return success
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"message": "Logged out successfully",
		})
	}).Methods("POST")

	// Send logout request
	req, err := http.NewRequest("POST", fmt.Sprintf("%s/auth/logout", suite.GetServer().URL), nil)
	require.NoError(t, err, "Failed to create logout request")
	req.Header.Set("Authorization", "Bearer "+token)

	client := &http.Client{}
	resp, err := client.Do(req)
	require.NoError(t, err, "Failed to send logout request")
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode, "Logout should succeed")

	var response map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&response)
	require.NoError(t, err, "Failed to decode logout response")

	assert.Contains(t, response, "message", "Logout response should contain message")

	t.Log("✓ User logout successful")
}

// TestInvalidAuthenticationScenarios tests various invalid authentication scenarios
func (suite *AuthIntegrationTestSuite) TestInvalidAuthenticationScenarios() {
	suite.T().Log("Testing invalid authentication scenarios...")

	// Test invalid registration data
	suite.T().Run("InvalidRegistration", func(t *testing.T) {
		invalidData := []map[string]interface{}{
			{"username": "", "email": "test@example.com", "password": "password123"},
			{"username": "test", "email": "", "password": "password123"},
			{"username": "test", "email": "test@example.com", "password": ""},
			{"username": "test", "email": "invalid-email", "password": "password123"},
		}

		for i, data := range invalidData {
			t.Logf("Testing invalid registration case %d", i+1)
			jsonData, _ := json.Marshal(data)

			resp, err := http.Post(
				fmt.Sprintf("%s/auth/register", suite.GetServer().URL),
				"application/json",
				bytes.NewBuffer(jsonData),
			)
			require.NoError(t, err, "Failed to send request")
			defer resp.Body.Close()

			assert.Equal(t, http.StatusBadRequest, resp.StatusCode, 
				"Invalid registration should return bad request")
		}
	})

	// Test invalid login attempts
	suite.T().Run("InvalidLogin", func(t *testing.T) {
		// First create a valid user
		validUser := map[string]string{
			"username": "validuser",
			"email":    "valid@example.com",
			"password": "ValidPassword123!",
		}

		// Register the user first
		suite.GetAuthManager().CreateUser(
			validUser["username"], 
			validUser["email"], 
			validUser["password"], 
			"user", 
			"default",
		)

		// Test invalid login attempts
		invalidLogins := []map[string]string{
			{"username": "nonexistent", "password": "password123"},
			{"username": validUser["username"], "password": "wrongpassword"},
			{"username": "", "password": validUser["password"]},
			{"username": validUser["username"], "password": ""},
		}

		for i, loginData := range invalidLogins {
			t.Logf("Testing invalid login case %d", i+1)
			jsonData, _ := json.Marshal(loginData)

			resp, err := http.Post(
				fmt.Sprintf("%s/auth/login", suite.GetServer().URL),
				"application/json",
				bytes.NewBuffer(jsonData),
			)
			require.NoError(t, err, "Failed to send login request")
			defer resp.Body.Close()

			assert.Equal(t, http.StatusUnauthorized, resp.StatusCode,
				"Invalid login should return unauthorized")
		}
	})

	suite.T().Log("✓ Invalid authentication scenarios tested successfully")
}

// TestAuthIntegrationSuite runs the auth integration test suite
func TestAuthIntegrationSuite(t *testing.T) {
	suite.Run(t, new(AuthIntegrationTestSuite))
}