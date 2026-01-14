package helpers

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// TestEnvironment manages the test environment lifecycle
type TestEnvironment struct {
	Config    *TestConfig
	DB        *DatabaseHelper
	APIClient *APIClient
	WSClient  *WebSocketClient
	GQLClient *GraphQLClient
}

// TestConfig holds environment configuration
type TestConfig struct {
	DatabaseURL  string
	APIURL      string
	WebSocketURL string
	RedisURL     string
	FrontendURL  string
	CleanupDB    bool
	UseDocker    bool
}

// NewTestEnvironment creates and initializes a test environment
func NewTestEnvironment(t *testing.T) *TestEnvironment {
	t.Helper()
	
	config := &TestConfig{
		DatabaseURL:  getEnvOrDefault("DB_URL", "postgresql://postgres:postgres@localhost:5432/novacron_test"),
		APIURL:      getEnvOrDefault("NOVACRON_API_URL", "http://localhost:8090"),
		WebSocketURL: getEnvOrDefault("NOVACRON_WS_URL", "ws://localhost:8091/ws"),
		RedisURL:    getEnvOrDefault("REDIS_URL", "redis://localhost:6379"),
		FrontendURL: getEnvOrDefault("NOVACRON_UI_URL", "http://localhost:8092"),
		CleanupDB:   getBoolEnvOrDefault("CLEANUP_DB", true),
		UseDocker:   getBoolEnvOrDefault("USE_DOCKER", false),
	}
	
	env := &TestEnvironment{
		Config: config,
	}
	
	// Initialize components
	env.initializeDatabase(t)
	env.initializeAPIClients(t)
	
	return env
}

// Setup prepares the test environment
func (e *TestEnvironment) Setup(t *testing.T) {
	t.Helper()
	
	// Wait for services to be available
	e.waitForServices(t)
	
	// Setup database
	e.DB.SetupTestDatabase(t)
	
	t.Log("Test environment setup completed")
}

// Cleanup cleans up the test environment
func (e *TestEnvironment) Cleanup(t *testing.T) {
	t.Helper()
	
	// Cleanup database if enabled
	if e.Config.CleanupDB && e.DB != nil {
		e.DB.CleanupTestDatabase(t)
	}
	
	// Close connections
	if e.WSClient != nil {
		e.WSClient.Close()
	}
	
	if e.DB != nil {
		e.DB.Close()
	}
	
	t.Log("Test environment cleanup completed")
}

// initializeDatabase initializes the database helper
func (e *TestEnvironment) initializeDatabase(t *testing.T) {
	t.Helper()
	
	db, err := NewDatabaseHelper(e.Config.DatabaseURL)
	require.NoError(t, err, "Failed to initialize database helper")
	
	e.DB = db
}

// initializeAPIClients initializes API clients
func (e *TestEnvironment) initializeAPIClients(t *testing.T) {
	t.Helper()
	
	// HTTP API client
	e.APIClient = NewAPIClient(e.Config.APIURL, "test-api-key")
	
	// WebSocket client
	e.WSClient = NewWebSocketClient(e.Config.WebSocketURL)
	
	// GraphQL client
	e.GQLClient = NewGraphQLClient(e.Config.APIURL, "test-api-key")
}

// waitForServices waits for required services to be available
func (e *TestEnvironment) waitForServices(t *testing.T) {
	t.Helper()
	
	timeout := 60 * time.Second
	
	// Wait for database
	t.Log("Waiting for database...")
	err := WaitForDatabase(e.Config.DatabaseURL, timeout)
	require.NoError(t, err, "Database not available")
	
	// Wait for API server
	t.Log("Waiting for API server...")
	err = WaitForAPI(e.Config.APIURL, timeout)
	require.NoError(t, err, "API server not available")
	
	// Wait for Redis (optional)
	if e.Config.RedisURL != "" {
		t.Log("Waiting for Redis...")
		err = WaitForRedis(e.Config.RedisURL, timeout)
		if err != nil {
			t.Logf("Warning: Redis not available: %v", err)
		}
	}
	
	t.Log("All services are available")
}

// LoginAsAdmin logs in as admin user and returns auth token
func (e *TestEnvironment) LoginAsAdmin(t *testing.T) string {
	t.Helper()
	return e.APIClient.Login(t, "admin@test.com", "admin123")
}

// LoginAsUser logs in as regular user and returns auth token
func (e *TestEnvironment) LoginAsUser(t *testing.T) string {
	t.Helper()
	return e.APIClient.Login(t, "user@test.com", "user123")
}

// CreateTestVM creates a test VM and returns its ID
func (e *TestEnvironment) CreateTestVM(t *testing.T, name, tenantID string) string {
	t.Helper()
	
	vmData := map[string]interface{}{
		"name":      name,
		"cpu":       2,
		"memory":    1024,
		"disk_size": 10240,
		"image":     "ubuntu:20.04",
		"tenant_id": tenantID,
	}
	
	resp := e.APIClient.POST(t, "/api/vms", vmData)
	e.APIClient.ExpectStatus(t, resp, 201)
	
	var result map[string]interface{}
	e.APIClient.ParseJSON(t, resp, &result)
	
	vmID, ok := result["id"].(string)
	require.True(t, ok, "No VM ID in response")
	require.NotEmpty(t, vmID, "Empty VM ID")
	
	return vmID
}

// WaitForVMState waits for a VM to reach the expected state
func (e *TestEnvironment) WaitForVMState(t *testing.T, vmID, expectedState string, timeout time.Duration) {
	t.Helper()
	
	deadline := time.Now().Add(timeout)
	
	for time.Now().Before(deadline) {
		resp := e.APIClient.GET(t, "/api/vms/"+vmID)
		
		if resp.StatusCode == 200 {
			var vm map[string]interface{}
			e.APIClient.ParseJSON(t, resp, &vm)
			
			if state, ok := vm["state"].(string); ok && state == expectedState {
				return
			}
		}
		
		time.Sleep(2 * time.Second)
	}
	
	t.Fatalf("VM %s did not reach state %s within %v", vmID, expectedState, timeout)
}

// GetVMMetrics retrieves metrics for a VM
func (e *TestEnvironment) GetVMMetrics(t *testing.T, vmID string) map[string]interface{} {
	t.Helper()
	
	resp := e.APIClient.GET(t, "/api/vms/"+vmID+"/metrics")
	e.APIClient.ExpectStatus(t, resp, 200)
	
	var metrics map[string]interface{}
	e.APIClient.ParseJSON(t, resp, &metrics)
	
	return metrics
}

// RunCommand runs a shell command for test setup
func (e *TestEnvironment) RunCommand(t *testing.T, command string, args ...string) string {
	t.Helper()
	
	cmd := exec.Command(command, args...)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		t.Logf("Command failed: %s %v", command, args)
		t.Logf("Output: %s", string(output))
		require.NoError(t, err, "Command execution failed")
	}
	
	return string(output)
}

// StartDockerServices starts required services using docker-compose
func (e *TestEnvironment) StartDockerServices(t *testing.T) {
	t.Helper()
	
	if !e.Config.UseDocker {
		t.Skip("Docker services disabled")
	}
	
	// Start test environment
	e.RunCommand(t, "docker-compose", "-f", "docker-compose.test.yml", "up", "-d")
	
	// Wait for services to be ready
	time.Sleep(15 * time.Second)
}

// StopDockerServices stops docker services
func (e *TestEnvironment) StopDockerServices(t *testing.T) {
	t.Helper()
	
	if !e.Config.UseDocker {
		return
	}
	
	e.RunCommand(t, "docker-compose", "-f", "docker-compose.test.yml", "down", "-v")
}

// Helper functions

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getBoolEnvOrDefault(key string, defaultValue bool) bool {
	value := strings.ToLower(os.Getenv(key))
	switch value {
	case "true", "1", "yes", "on":
		return true
	case "false", "0", "no", "off":
		return false
	default:
		return defaultValue
	}
}

// WaitForRedis waits for Redis to become available
func WaitForRedis(redisURL string, timeout time.Duration) error {
	// Simple Redis availability check
	// In a real implementation, you'd use a Redis client
	deadline := time.Now().Add(timeout)
	
	for time.Now().Before(deadline) {
		// Try to connect to Redis port
		cmd := exec.Command("nc", "-z", "localhost", "6379")
		if err := cmd.Run(); err == nil {
			return nil
		}
		
		time.Sleep(1 * time.Second)
	}
	
	return fmt.Errorf("Redis not available after %v", timeout)
}

// LoadTestConfiguration loads test configuration from files
func LoadTestConfiguration(configPath string) (*TestConfig, error) {
	// This could load from YAML, JSON, or other config formats
	// For now, return default configuration
	return &TestConfig{
		DatabaseURL:  getEnvOrDefault("DB_URL", "postgresql://postgres:postgres@localhost:5432/novacron_test"),
		APIURL:      getEnvOrDefault("NOVACRON_API_URL", "http://localhost:8090"),
		WebSocketURL: getEnvOrDefault("NOVACRON_WS_URL", "ws://localhost:8091/ws"),
		RedisURL:    getEnvOrDefault("REDIS_URL", "redis://localhost:6379"),
		FrontendURL: getEnvOrDefault("NOVACRON_UI_URL", "http://localhost:8092"),
		CleanupDB:   getBoolEnvOrDefault("CLEANUP_DB", true),
		UseDocker:   getBoolEnvOrDefault("USE_DOCKER", false),
	}, nil
}

// TestSuite provides a base for integration test suites
type TestSuite struct {
	Env *TestEnvironment
}

// SetupSuite sets up the test suite
func (s *TestSuite) SetupSuite(t *testing.T) {
	s.Env = NewTestEnvironment(t)
	s.Env.Setup(t)
}

// TearDownSuite tears down the test suite
func (s *TestSuite) TearDownSuite(t *testing.T) {
	if s.Env != nil {
		s.Env.Cleanup(t)
	}
}