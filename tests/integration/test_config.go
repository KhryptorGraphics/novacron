package integration

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"
	"time"
)

// TestConfig holds configuration for integration tests
type TestConfig struct {
	// Database configuration
	DatabaseURL string
	DatabaseTimeout time.Duration

	// API configuration  
	APIBaseURL string
	APITimeout time.Duration
	APIKey     string

	// Redis configuration
	RedisURL string
	RedisTimeout time.Duration

	// Frontend configuration
	FrontendURL string

	// Authentication configuration
	AuthSecret string
	TestUserEmail string
	TestUserPassword string
	
	// Test timeouts
	DefaultTimeout time.Duration
	LongTimeout time.Duration
	
	// Test options
	SkipSlowTests bool
	DebugMode bool
	CleanupAfterTests bool
	
	// Performance test settings
	ConcurrentRequests int
	BenchmarkDuration time.Duration
}

// DefaultTestConfig returns a default test configuration
func DefaultTestConfig() *TestConfig {
	return &TestConfig{
		DatabaseURL:     getEnvWithDefault("DB_URL", "postgresql://postgres:postgres@localhost:5432/novacron_test"),
		DatabaseTimeout: 30 * time.Second,
		
		APIBaseURL: getEnvWithDefault("NOVACRON_API_URL", "http://localhost:8090"),
		APITimeout: 10 * time.Second,
		APIKey:     getEnvWithDefault("NOVACRON_API_KEY", "test-api-key"),
		
		RedisURL:     getEnvWithDefault("REDIS_URL", "redis://localhost:6379"),
		RedisTimeout: 5 * time.Second,
		
		FrontendURL: getEnvWithDefault("NOVACRON_UI_URL", "http://localhost:8092"),
		
		AuthSecret:       getEnvWithDefault("AUTH_SECRET", "test-secret-key"),
		TestUserEmail:    "test@example.com",
		TestUserPassword: "password123",
		
		DefaultTimeout: 30 * time.Second,
		LongTimeout:    5 * time.Minute,
		
		SkipSlowTests:     getBoolEnv("SKIP_SLOW_TESTS", false),
		DebugMode:         getBoolEnv("DEBUG_MODE", false),
		CleanupAfterTests: getBoolEnv("CLEANUP_AFTER_TESTS", true),
		
		ConcurrentRequests: getIntEnv("CONCURRENT_REQUESTS", 10),
		BenchmarkDuration:  getDurationEnv("BENCHMARK_DURATION", 60*time.Second),
	}
}

// LoadTestConfig loads configuration from environment variables
func LoadTestConfig() *TestConfig {
	config := DefaultTestConfig()
	
	if config.DebugMode {
		log.Printf("Test Configuration:")
		log.Printf("  Database URL: %s", maskSensitive(config.DatabaseURL))
		log.Printf("  API Base URL: %s", config.APIBaseURL)
		log.Printf("  Redis URL: %s", config.RedisURL)
		log.Printf("  Frontend URL: %s", config.FrontendURL)
		log.Printf("  Skip Slow Tests: %v", config.SkipSlowTests)
		log.Printf("  Debug Mode: %v", config.DebugMode)
	}
	
	return config
}

// ValidateConfig validates the test configuration
func (c *TestConfig) ValidateConfig() error {
	if c.DatabaseURL == "" {
		return fmt.Errorf("database URL is required")
	}
	
	if c.APIBaseURL == "" {
		return fmt.Errorf("API base URL is required")
	}
	
	if c.RedisURL == "" {
		return fmt.Errorf("Redis URL is required")
	}
	
	return nil
}

// CreateTestContext creates a context with appropriate timeout
func (c *TestConfig) CreateTestContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), c.DefaultTimeout)
}

// CreateLongTestContext creates a context with long timeout for slow operations
func (c *TestConfig) CreateLongTestContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), c.LongTimeout)
}

// Global test configuration instance
var TestCfg *TestConfig

// InitTestConfig initializes the global test configuration
func InitTestConfig() {
	TestCfg = LoadTestConfig()
	
	if err := TestCfg.ValidateConfig(); err != nil {
		log.Fatalf("Invalid test configuration: %v", err)
	}
}

// Helper functions for environment variable parsing

func getEnvWithDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getBoolEnv(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseBool(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getIntEnv(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getDurationEnv(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if parsed, err := time.ParseDuration(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func maskSensitive(url string) string {
	// Simple masking for database URLs containing passwords
	return "***masked***"
}