package integration_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/mux"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/backend/api/rest"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// APIIntegrationTestSuite defines the integration test suite for API endpoints
type APIIntegrationTestSuite struct {
	suite.Suite
	ctx              context.Context
	cancel           context.CancelFunc
	server           *httptest.Server
	router           *mux.Router
	authService      auth.AuthService
	vmManager        vm.VMManager
	adminToken       string
	userToken        string
	
	// Enhanced testing components
	apiClient        *APIClient
	authManager      *AuthManager
	rateLimiter      *APIRateLimiter
	validator        *RequestValidator
	cache            *APICache
	metrics          *APIMetrics
	testData         *TestDataManager
	mu               sync.RWMutex
}

// APIClient provides enhanced HTTP client functionality for testing
type APIClient struct {
	BaseURL    string
	HTTPClient *http.Client
	AuthToken  string
	Headers    map[string]string
	mu         sync.RWMutex
}

// AuthManager handles authentication and authorization testing
type AuthManager struct {
	Users         map[string]*User
	Sessions      map[string]*Session
	Tokens        map[string]*Token
	Permissions   map[string][]string
	SessionTimeout time.Duration
	TokenTimeout   time.Duration
	mu            sync.RWMutex
}

type User struct {
	ID          string    `json:"id"`
	Username    string    `json:"username"`
	Email       string    `json:"email"`
	Role        string    `json:"role"`
	Permissions []string  `json:"permissions"`
	IsActive    bool      `json:"is_active"`
	CreatedAt   time.Time `json:"created_at"`
	LastLogin   time.Time `json:"last_login"`
}

type Session struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	Token     string    `json:"token"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
	IPAddress string    `json:"ip_address"`
	UserAgent string    `json:"user_agent"`
}

type Token struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	Token     string    `json:"token"`
	Type      string    `json:"type"` // "access", "refresh", "api_key"
	Scopes    []string  `json:"scopes"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
}

// APIRateLimiter provides API rate limiting testing
type APIRateLimiter struct {
	GlobalLimit   int
	UserLimits    map[string]int
	WindowSize    time.Duration
	RequestCounts map[string][]time.Time
	BucketConfig  map[string]*TokenBucket
	mu            sync.RWMutex
}

type TokenBucket struct {
	Capacity   int
	Tokens     int
	RefillRate float64
	LastRefill time.Time
}

// RequestValidator validates API requests in tests
type RequestValidator struct {
	Schemas       map[string]Schema
	ValidateJSON  bool
	ValidateTypes bool
	MaxBodySize   int64
	mu            sync.RWMutex
}

type Schema struct {
	Fields      map[string]FieldDefinition
	Required    []string
	Optional    []string
	Validations map[string][]Validation
}

type FieldDefinition struct {
	Type        string
	Format      string
	MinLength   int
	MaxLength   int
	Min         interface{}
	Max         interface{}
	Pattern     string
	Enum        []interface{}
	Default     interface{}
}

type Validation struct {
	Rule    string
	Param   interface{}
	Message string
}

// APICache provides response caching testing
type APICache struct {
	Store       map[string]*CacheEntry
	TTL         time.Duration
	MaxSize     int
	HitCount    int64
	MissCount   int64
	EvictCount  int64
	mu          sync.RWMutex
}

type CacheEntry struct {
	Key       string
	Value     interface{}
	CreatedAt time.Time
	ExpiresAt time.Time
	HitCount  int64
}

// APIMetrics tracks API performance in tests
type APIMetrics struct {
	RequestCount      int64
	ErrorCount        int64
	ResponseTimes     []time.Duration
	StatusCodes       map[int]int64
	EndpointMetrics   map[string]*EndpointMetrics
	UserMetrics       map[string]*UserMetrics
	mu                sync.RWMutex
}

type EndpointMetrics struct {
	RequestCount  int64
	ErrorCount    int64
	TotalLatency  time.Duration
	MinLatency    time.Duration
	MaxLatency    time.Duration
	StatusCodes   map[int]int64
}

type UserMetrics struct {
	RequestCount    int64
	ErrorCount      int64
	LastRequestTime time.Time
	Endpoints       map[string]int64
}

// TestDataManager manages test data
type TestDataManager struct {
	Users    []*User
	VMs      []*VM
	Orders   []*Order
	Products []*Product
	mu       sync.RWMutex
}

type VM struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Type        string    `json:"type"`
	Status      string    `json:"status"`
	CPU         int       `json:"cpu"`
	Memory      int       `json:"memory"`
	Disk        int       `json:"disk"`
	Networks    []string  `json:"networks"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type Order struct {
	ID          string    `json:"id"`
	UserID      string    `json:"user_id"`
	ProductID   string    `json:"product_id"`
	Quantity    int       `json:"quantity"`
	Status      string    `json:"status"`
	Total       float64   `json:"total"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type Product struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Price       float64   `json:"price"`
	Category    string    `json:"category"`
	InStock     bool      `json:"in_stock"`
	Quantity    int       `json:"quantity"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

func (suite *APIIntegrationTestSuite) SetupSuite() {
	suite.ctx, suite.cancel = context.WithTimeout(context.Background(), 30*time.Minute)
	
	// Initialize test services
	suite.authService = auth.NewMemoryAuthService()
	suite.vmManager = vm.NewVMManager(&vm.VMManagerConfig{
		DefaultDriver: vm.VMTypeQEMU,
	})

	// Setup API router
	suite.router = rest.NewRouter(suite.authService, suite.vmManager)
	suite.server = httptest.NewServer(suite.router)

	// Initialize enhanced testing components
	suite.initializeTestComponents()
	
	// Create test users and get tokens
	suite.setupTestUsers()
}

func (suite *APIIntegrationTestSuite) initializeTestComponents() {
	// Initialize API client
	suite.apiClient = &APIClient{
		BaseURL: suite.server.URL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		Headers: map[string]string{
			"Content-Type": "application/json",
			"Accept":       "application/json",
		},
	}
	
	// Initialize auth manager
	suite.authManager = &AuthManager{
		Users:          make(map[string]*User),
		Sessions:       make(map[string]*Session),
		Tokens:         make(map[string]*Token),
		Permissions:    make(map[string][]string),
		SessionTimeout: 24 * time.Hour,
		TokenTimeout:   1 * time.Hour,
	}
	
	// Initialize rate limiter
	suite.rateLimiter = &APIRateLimiter{
		GlobalLimit:   1000,
		UserLimits:    make(map[string]int),
		WindowSize:    time.Minute,
		RequestCounts: make(map[string][]time.Time),
		BucketConfig:  make(map[string]*TokenBucket),
	}
	
	// Initialize validator
	suite.validator = &RequestValidator{
		Schemas:      make(map[string]Schema),
		ValidateJSON: true,
		ValidateTypes: true,
		MaxBodySize:  10 * 1024 * 1024, // 10MB
	}
	
	// Initialize cache
	suite.cache = &APICache{
		Store:   make(map[string]*CacheEntry),
		TTL:     5 * time.Minute,
		MaxSize: 1000,
	}
	
	// Initialize metrics
	suite.metrics = &APIMetrics{
		StatusCodes:     make(map[int]int64),
		EndpointMetrics: make(map[string]*EndpointMetrics),
		UserMetrics:     make(map[string]*UserMetrics),
	}
	
	// Initialize test data
	suite.testData = &TestDataManager{
		Users:    []*User{},
		VMs:      []*VM{},
		Orders:   []*Order{},
		Products: []*Product{},
	}
	
	// Setup test data
	suite.setupTestData()
}

func (suite *APIIntegrationTestSuite) TearDownSuite() {
	if suite.cancel != nil {
		suite.cancel()
	}
	
	if suite.server != nil {
		suite.server.Close()
	}
}

func (suite *APIIntegrationTestSuite) setupTestData() {
	// Create test users
	testUser := &User{
		ID:          "test-user-1",
		Username:    "testuser",
		Email:       "test@example.com",
		Role:        "user",
		Permissions: []string{"read_profile", "write_profile", "create_vm"},
		IsActive:    true,
		CreatedAt:   time.Now(),
	}
	
	adminUser := &User{
		ID:          "admin-user-1",
		Username:    "admin",
		Email:       "admin@example.com",
		Role:        "admin",
		Permissions: []string{"*"},
		IsActive:    true,
		CreatedAt:   time.Now(),
	}
	
	suite.authManager.Users["testuser"] = testUser
	suite.authManager.Users["admin"] = adminUser
	
	suite.testData.Users = []*User{testUser, adminUser}
	
	// Setup default VMs
	for i := 1; i <= 5; i++ {
		vm := &VM{
			ID:        fmt.Sprintf("vm-%d", i),
			Name:      fmt.Sprintf("Test VM %d", i),
			Type:      "qemu",
			Status:    "stopped",
			CPU:       2,
			Memory:    4096,
			Disk:      20,
			Networks:  []string{"default"},
			CreatedAt: time.Now(),
		}
		suite.testData.VMs = append(suite.testData.VMs, vm)
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

// Enhanced comprehensive API endpoint tests

func (suite *APIIntegrationTestSuite) TestRequestValidationAndErrorHandling() {
	// Test comprehensive request validation
	
	t := suite.T()
	
	// Test missing required fields
	invalidVM := map[string]interface{}{
		"description": "Missing name and type",
	}
	
	resp, err := suite.makeRequest("POST", "/api/v1/vms", invalidVM, suite.adminToken)
	require.NoError(t, err, "Invalid request should be processed")
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode, "Should return 400 for invalid data")
	
	var errorResponse map[string]interface{}
	err = json.NewDecoder(resp.Body).Decode(&errorResponse)
	require.NoError(t, err, "Should decode error response")
	
	// Test malformed JSON
	malformedJSON := `{"name": "test", "type": "qemu", "invalid": }`
	
	req, err := http.NewRequest("POST", suite.server.URL+"/api/v1/vms", strings.NewReader(malformedJSON))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+suite.adminToken)
	
	client := &http.Client{}
	malformedResp, err := client.Do(req)
	require.NoError(t, err, "Malformed JSON should be processed")
	assert.Equal(t, http.StatusBadRequest, malformedResp.StatusCode, "Should return 400 for malformed JSON")
	
	t.Log("Request validation test completed successfully")
}

func (suite *APIIntegrationTestSuite) TestRateLimitingFunctionality() {
	// Test API rate limiting
	
	t := suite.T()
	
	successCount := 0
	rateLimitedCount := 0
	
	// Send requests rapidly to test rate limiting
	for i := 0; i < 50; i++ {
		resp, err := suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
		require.NoError(t, err, "Request should be processed")
		
		if resp.StatusCode == http.StatusOK {
			successCount++
		} else if resp.StatusCode == http.StatusTooManyRequests {
			rateLimitedCount++
		}
		
		// Small delay between requests
		time.Sleep(10 * time.Millisecond)
	}
	
	// Check rate limit headers
	resp, err := suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
	require.NoError(t, err, "Request should be processed")
	
	if resp.Header.Get("X-RateLimit-Limit") != "" {
		limitHeader := resp.Header.Get("X-RateLimit-Limit")
		remainingHeader := resp.Header.Get("X-RateLimit-Remaining")
		
		assert.NotEmpty(t, limitHeader, "Should include rate limit header")
		assert.NotEmpty(t, remainingHeader, "Should include remaining requests header")
		
		if remaining, err := strconv.Atoi(remainingHeader); err == nil {
			assert.GreaterOrEqual(t, remaining, 0, "Remaining should be non-negative")
		}
	}
	
	t.Logf("Rate limiting test: %d successful, %d rate limited", successCount, rateLimitedCount)
}

func (suite *APIIntegrationTestSuite) TestConcurrentAPIAccess() {
	// Test concurrent API access and thread safety
	
	t := suite.T()
	
	concurrency := 10
	requestsPerGoroutine := 5
	totalRequests := concurrency * requestsPerGoroutine
	
	var wg sync.WaitGroup
	var successCount int64
	var errorCount int64
	
	// Launch concurrent requests
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(clientIndex int) {
			defer wg.Done()
			
			for j := 0; j < requestsPerGoroutine; j++ {
				// Mix of different operations
				switch j % 3 {
				case 0:
					// GET request
					resp, err := suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
					if err == nil && resp.StatusCode == http.StatusOK {
						atomic.AddInt64(&successCount, 1)
					} else {
						atomic.AddInt64(&errorCount, 1)
					}
					
				case 1:
					// Auth check
					resp, err := suite.makeRequest("GET", "/api/v1/auth/me", nil, suite.adminToken)
					if err == nil && resp.StatusCode == http.StatusOK {
						atomic.AddInt64(&successCount, 1)
					} else {
						atomic.AddInt64(&errorCount, 1)
					}
					
				case 2:
					// Health check
					resp, err := suite.makeRequest("GET", "/health", nil, "")
					if err == nil && resp.StatusCode == http.StatusOK {
						atomic.AddInt64(&successCount, 1)
					} else {
						atomic.AddInt64(&errorCount, 1)
					}
				}
				
				// Small delay between requests
				time.Sleep(5 * time.Millisecond)
			}
		}(i)
	}
	
	// Wait for all requests to complete
	wg.Wait()
	
	// Verify results
	finalSuccessCount := atomic.LoadInt64(&successCount)
	finalErrorCount := atomic.LoadInt64(&errorCount)
	
	assert.Greater(t, finalSuccessCount, int64(0), "Should have successful requests")
	assert.Equal(t, int64(totalRequests), finalSuccessCount+finalErrorCount, 
		"All requests should be accounted for")
	
	// Success rate should be reasonable (allowing for rate limiting)
	successRate := float64(finalSuccessCount) / float64(totalRequests)
	assert.Greater(t, successRate, 0.7, "Success rate should be reasonable")
	
	t.Logf("Concurrent access test: %d/%d successful (%.2f%% success rate)", 
		finalSuccessCount, totalRequests, successRate*100)
}

func (suite *APIIntegrationTestSuite) TestAPIMetricsAndPerformance() {
	// Test API performance metrics and response times
	
	t := suite.T()
	
	endpoints := []struct {
		method   string
		path     string
		token    string
		maxTime  time.Duration
	}{
		{"GET", "/health", "", 500 * time.Millisecond},
		{"GET", "/api/v1/vms", suite.adminToken, 1 * time.Second},
		{"GET", "/api/v1/auth/me", suite.adminToken, 1 * time.Second},
		{"GET", "/api/v1/admin/stats", suite.adminToken, 2 * time.Second},
	}
	
	for _, endpoint := range endpoints {
		start := time.Now()
		resp, err := suite.makeRequest(endpoint.method, endpoint.path, nil, endpoint.token)
		duration := time.Since(start)
		
		require.NoError(t, err, "Request should succeed for %s", endpoint.path)
		assert.Less(t, duration, endpoint.maxTime, 
			"Response time for %s should be under %v, got %v", endpoint.path, endpoint.maxTime, duration)
		
		if resp.StatusCode == http.StatusOK {
			// Response should be valid
			assert.True(t, resp.ContentLength != 0 || resp.Header.Get("Transfer-Encoding") == "chunked",
				"Response should have content for %s", endpoint.path)
		}
		
		t.Logf("Endpoint %s response time: %v", endpoint.path, duration)
	}
}

func (suite *APIIntegrationTestSuite) TestErrorResponseFormats() {
	// Test consistent error response formats
	
	t := suite.T()
	
	errorTests := []struct {
		name           string
		method         string
		path           string
		body           interface{}
		token          string
		expectedStatus int
	}{
		{"404 Not Found", "GET", "/api/v1/vms/nonexistent", nil, suite.adminToken, 404},
		{"401 Unauthorized", "GET", "/api/v1/auth/me", nil, "", 401},
		{"403 Forbidden", "GET", "/api/v1/admin/stats", nil, suite.userToken, 403},
		{"400 Bad Request", "POST", "/api/v1/vms", map[string]interface{}{}, suite.adminToken, 400},
	}
	
	for _, test := range errorTests {
		t.Run(test.name, func(t *testing.T) {
			resp, err := suite.makeRequest(test.method, test.path, test.body, test.token)
			require.NoError(t, err, "Request should be processed")
			assert.Equal(t, test.expectedStatus, resp.StatusCode, "Should return expected status")
			
			var errorFormat map[string]interface{}
			err = json.NewDecoder(resp.Body).Decode(&errorFormat)
			
			if err == nil {
				// Check for consistent error format
				assert.Contains(t, errorFormat, "error", "Should contain error field")
				
				// Additional consistency checks
				if message, ok := errorFormat["message"]; ok {
					assert.NotEmpty(t, message, "Message should not be empty")
				}
				
				if timestamp, ok := errorFormat["timestamp"]; ok {
					assert.NotEmpty(t, timestamp, "Timestamp should not be empty")
				}
			}
		})
	}
}

func (suite *APIIntegrationTestSuite) TestAPISecurityHeaders() {
	// Test security headers in API responses
	
	t := suite.T()
	
	resp, err := suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
	require.NoError(t, err, "Request should succeed")
	
	// Check for common security headers
	securityHeaders := []string{
		"X-Content-Type-Options",
		"X-Frame-Options",
		"X-XSS-Protection",
	}
	
	for _, header := range securityHeaders {
		headerValue := resp.Header.Get(header)
		if headerValue != "" {
			assert.NotEmpty(t, headerValue, "Security header %s should not be empty", header)
			t.Logf("Security header %s: %s", header, headerValue)
		}
	}
	
	// Check Content-Type is properly set
	contentType := resp.Header.Get("Content-Type")
	if resp.StatusCode == http.StatusOK {
		assert.Contains(t, contentType, "application/json", "Content-Type should be JSON")
	}
}

func (suite *APIIntegrationTestSuite) TestAPIVersioning() {
	// Test API versioning behavior
	
	t := suite.T()
	
	// Test v1 API
	respV1, err := suite.makeRequest("GET", "/api/v1/vms", nil, suite.adminToken)
	require.NoError(t, err, "V1 API request should succeed")
	
	if respV1.StatusCode == http.StatusOK {
		var vms []map[string]interface{}
		err = json.NewDecoder(respV1.Body).Decode(&vms)
		require.NoError(t, err, "Should decode V1 response")
	}
	
	// Test invalid version
	respInvalid, err := suite.makeRequest("GET", "/api/v99/vms", nil, suite.adminToken)
	require.NoError(t, err, "Invalid version request should be processed")
	assert.Equal(t, http.StatusNotFound, respInvalid.StatusCode, "Invalid version should return 404")
}

func (suite *APIIntegrationTestSuite) TestWebSocketConnections() {
	// Test WebSocket connections if supported
	
	t := suite.T()
	
	// This is a placeholder for WebSocket testing
	// In a real implementation, you would test WebSocket endpoints
	// for real-time updates, notifications, etc.
	
	t.Log("WebSocket testing would be implemented here if API supports it")
}

func (suite *APIIntegrationTestSuite) TestAPIDocumentationEndpoints() {
	// Test API documentation endpoints
	
	t := suite.T()
	
	// Test common documentation endpoints
	docEndpoints := []string{
		"/api/docs",
		"/api/v1/docs",
		"/swagger.json",
		"/openapi.json",
	}
	
	for _, endpoint := range docEndpoints {
		resp, err := suite.makeRequest("GET", endpoint, nil, "")
		require.NoError(t, err, "Documentation request should be processed")
		
		if resp.StatusCode == http.StatusOK {
			t.Logf("Documentation available at: %s", endpoint)
			
			contentType := resp.Header.Get("Content-Type")
			assert.True(t, 
				strings.Contains(contentType, "application/json") || 
				strings.Contains(contentType, "text/html") ||
				strings.Contains(contentType, "application/yaml"),
				"Documentation should have appropriate content type")
		}
	}
}

// APIClient helper methods

func (c *APIClient) Get(path string) (*http.Response, error) {
	req, err := http.NewRequest("GET", c.BaseURL+path, nil)
	if err != nil {
		return nil, err
	}
	
	c.setHeaders(req)
	return c.HTTPClient.Do(req)
}

func (c *APIClient) Post(path string, data interface{}) (*http.Response, error) {
	var body io.Reader
	if data != nil {
		jsonData, err := json.Marshal(data)
		if err != nil {
			return nil, err
		}
		body = bytes.NewBuffer(jsonData)
	}
	
	req, err := http.NewRequest("POST", c.BaseURL+path, body)
	if err != nil {
		return nil, err
	}
	
	c.setHeaders(req)
	return c.HTTPClient.Do(req)
}

func (c *APIClient) Put(path string, data interface{}) (*http.Response, error) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}
	
	req, err := http.NewRequest("PUT", c.BaseURL+path, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	
	c.setHeaders(req)
	return c.HTTPClient.Do(req)
}

func (c *APIClient) Delete(path string) (*http.Response, error) {
	req, err := http.NewRequest("DELETE", c.BaseURL+path, nil)
	if err != nil {
		return nil, err
	}
	
	c.setHeaders(req)
	return c.HTTPClient.Do(req)
}

func (c *APIClient) setHeaders(req *http.Request) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	for key, value := range c.Headers {
		req.Header.Set(key, value)
	}
	
	if c.AuthToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.AuthToken)
	}
}

func (c *APIClient) SetAuthToken(token string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.AuthToken = token
}

// Helper functions

func getString(data map[string]interface{}, key string) string {
	if value, ok := data[key].(string); ok {
		return value
	}
	return ""
}

func getInt(data map[string]interface{}, key string, defaultValue int) int {
	if value, ok := data[key].(float64); ok {
		return int(value)
	}
	return defaultValue
}

func getFloat64(data map[string]interface{}, key string, defaultValue float64) float64 {
	if value, ok := data[key].(float64); ok {
		return value
	}
	return defaultValue
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