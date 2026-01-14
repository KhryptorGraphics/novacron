package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds all configuration for the NovaCron system
type Config struct {
	Server   ServerConfig   `json:"server"`
	Database DatabaseConfig `json:"database"`
	Auth     AuthConfig     `json:"auth"`
	VM       VMConfig       `json:"vm"`
	Logging  LoggingConfig  `json:"logging"`
	CORS     CORSConfig     `json:"cors"`
}

// ServerConfig holds HTTP server configuration
type ServerConfig struct {
	APIPort         string        `json:"api_port" env:"API_PORT" default:"8090"`
	WSPort          string        `json:"ws_port" env:"WS_PORT" default:"8091"`
	ReadTimeout     time.Duration `json:"read_timeout" env:"READ_TIMEOUT" default:"15s"`
	WriteTimeout    time.Duration `json:"write_timeout" env:"WRITE_TIMEOUT" default:"15s"`
	IdleTimeout     time.Duration `json:"idle_timeout" env:"IDLE_TIMEOUT" default:"60s"`
	ShutdownTimeout time.Duration `json:"shutdown_timeout" env:"SHUTDOWN_TIMEOUT" default:"30s"`
}

// DatabaseConfig holds database configuration
type DatabaseConfig struct {
	URL             string        `json:"url" env:"DB_URL" default:"postgresql://postgres:postgres@localhost:5432/novacron"`
	MaxConnections  int           `json:"max_connections" env:"DB_MAX_CONNECTIONS" default:"25"`
	ConnMaxLifetime time.Duration `json:"conn_max_lifetime" env:"DB_CONN_MAX_LIFETIME" default:"5m"`
	ConnMaxIdleTime time.Duration `json:"conn_max_idle_time" env:"DB_CONN_MAX_IDLE_TIME" default:"1m"`
}

// AuthConfig holds authentication configuration
type AuthConfig struct {
	Secret                string        `json:"-" env:"AUTH_SECRET" default:"changeme_in_production"`
	SessionExpiry         time.Duration `json:"session_expiry" env:"AUTH_SESSION_EXPIRY" default:"24h"`
	TokenLength           int           `json:"token_length" env:"AUTH_TOKEN_LENGTH" default:"32"`
	MaxSessionsPerUser    int           `json:"max_sessions_per_user" env:"AUTH_MAX_SESSIONS_PER_USER" default:"5"`
	PasswordMinLength     int           `json:"password_min_length" env:"AUTH_PASSWORD_MIN_LENGTH" default:"8"`
	RequirePasswordMixed  bool          `json:"require_password_mixed" env:"AUTH_REQUIRE_PASSWORD_MIXED" default:"true"`
	RequirePasswordNumber bool          `json:"require_password_number" env:"AUTH_REQUIRE_PASSWORD_NUMBER" default:"true"`
	RequirePasswordSymbol bool          `json:"require_password_symbol" env:"AUTH_REQUIRE_PASSWORD_SYMBOL" default:"true"`
}

// VMConfig holds VM management configuration
type VMConfig struct {
	StoragePath         string   `json:"storage_path" env:"STORAGE_PATH" default:"/var/lib/novacron/vms"`
	HypervisorAddrs     []string `json:"hypervisor_addrs" env:"HYPERVISOR_ADDRS" default:"localhost:9000"`
	DefaultCPUShares    int      `json:"default_cpu_shares" env:"VM_DEFAULT_CPU_SHARES" default:"1024"`
	DefaultMemoryMB     int      `json:"default_memory_mb" env:"VM_DEFAULT_MEMORY_MB" default:"512"`
	MaxVMsPerNode       int      `json:"max_vms_per_node" env:"VM_MAX_PER_NODE" default:"100"`
	HealthCheckInterval time.Duration `json:"health_check_interval" env:"VM_HEALTH_CHECK_INTERVAL" default:"30s"`
}

// LoggingConfig holds logging configuration
type LoggingConfig struct {
	Level      string `json:"level" env:"LOG_LEVEL" default:"info"`
	Format     string `json:"format" env:"LOG_FORMAT" default:"json"`
	Output     string `json:"output" env:"LOG_OUTPUT" default:"stdout"`
	Structured bool   `json:"structured" env:"LOG_STRUCTURED" default:"true"`
}

// CORSConfig holds CORS configuration
type CORSConfig struct {
	AllowedOrigins []string `json:"allowed_origins" env:"CORS_ALLOWED_ORIGINS" default:"http://localhost:8092,http://localhost:3001"`
	AllowedMethods []string `json:"allowed_methods" env:"CORS_ALLOWED_METHODS" default:"GET,POST,PUT,DELETE,OPTIONS"`
	AllowedHeaders []string `json:"allowed_headers" env:"CORS_ALLOWED_HEADERS" default:"Content-Type,Authorization"`
}

// Load creates a new Config instance with values loaded from environment variables
func Load() (*Config, error) {
	config := &Config{}
	
	// Load server configuration
	config.Server = ServerConfig{
		APIPort:         getEnvOrDefault("API_PORT", "8090"),
		WSPort:          getEnvOrDefault("WS_PORT", "8091"),
		ReadTimeout:     getEnvDurationOrDefault("READ_TIMEOUT", 15*time.Second),
		WriteTimeout:    getEnvDurationOrDefault("WRITE_TIMEOUT", 15*time.Second),
		IdleTimeout:     getEnvDurationOrDefault("IDLE_TIMEOUT", 60*time.Second),
		ShutdownTimeout: getEnvDurationOrDefault("SHUTDOWN_TIMEOUT", 30*time.Second),
	}
	
	// Load database configuration
	config.Database = DatabaseConfig{
		URL:             getEnvOrDefault("DB_URL", "postgresql://postgres:postgres@localhost:5432/novacron"),
		MaxConnections:  getEnvIntOrDefault("DB_MAX_CONNECTIONS", 25),
		ConnMaxLifetime: getEnvDurationOrDefault("DB_CONN_MAX_LIFETIME", 5*time.Minute),
		ConnMaxIdleTime: getEnvDurationOrDefault("DB_CONN_MAX_IDLE_TIME", 1*time.Minute),
	}
	
	// Load auth configuration
	authSecret := getEnvOrDefault("AUTH_SECRET", "changeme_in_production")
	if authSecret == "changeme_in_production" {
		return nil, fmt.Errorf("AUTH_SECRET must be set to a secure value in production")
	}
	
	config.Auth = AuthConfig{
		Secret:                authSecret,
		SessionExpiry:         getEnvDurationOrDefault("AUTH_SESSION_EXPIRY", 24*time.Hour),
		TokenLength:           getEnvIntOrDefault("AUTH_TOKEN_LENGTH", 32),
		MaxSessionsPerUser:    getEnvIntOrDefault("AUTH_MAX_SESSIONS_PER_USER", 5),
		PasswordMinLength:     getEnvIntOrDefault("AUTH_PASSWORD_MIN_LENGTH", 8),
		RequirePasswordMixed:  getEnvBoolOrDefault("AUTH_REQUIRE_PASSWORD_MIXED", true),
		RequirePasswordNumber: getEnvBoolOrDefault("AUTH_REQUIRE_PASSWORD_NUMBER", true),
		RequirePasswordSymbol: getEnvBoolOrDefault("AUTH_REQUIRE_PASSWORD_SYMBOL", true),
	}
	
	// Load VM configuration
	hypervisorAddrs := strings.Split(getEnvOrDefault("HYPERVISOR_ADDRS", "localhost:9000"), ",")
	for i, addr := range hypervisorAddrs {
		hypervisorAddrs[i] = strings.TrimSpace(addr)
	}
	
	config.VM = VMConfig{
		StoragePath:         getEnvOrDefault("STORAGE_PATH", "/var/lib/novacron/vms"),
		HypervisorAddrs:     hypervisorAddrs,
		DefaultCPUShares:    getEnvIntOrDefault("VM_DEFAULT_CPU_SHARES", 1024),
		DefaultMemoryMB:     getEnvIntOrDefault("VM_DEFAULT_MEMORY_MB", 512),
		MaxVMsPerNode:       getEnvIntOrDefault("VM_MAX_PER_NODE", 100),
		HealthCheckInterval: getEnvDurationOrDefault("VM_HEALTH_CHECK_INTERVAL", 30*time.Second),
	}
	
	// Load logging configuration
	config.Logging = LoggingConfig{
		Level:      getEnvOrDefault("LOG_LEVEL", "info"),
		Format:     getEnvOrDefault("LOG_FORMAT", "json"),
		Output:     getEnvOrDefault("LOG_OUTPUT", "stdout"),
		Structured: getEnvBoolOrDefault("LOG_STRUCTURED", true),
	}
	
	// Load CORS configuration
	allowedOrigins := strings.Split(getEnvOrDefault("CORS_ALLOWED_ORIGINS", "http://localhost:8092,http://localhost:3001"), ",")
	for i, origin := range allowedOrigins {
		allowedOrigins[i] = strings.TrimSpace(origin)
	}
	
	allowedMethods := strings.Split(getEnvOrDefault("CORS_ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS"), ",")
	for i, method := range allowedMethods {
		allowedMethods[i] = strings.TrimSpace(method)
	}
	
	allowedHeaders := strings.Split(getEnvOrDefault("CORS_ALLOWED_HEADERS", "Content-Type,Authorization"), ",")
	for i, header := range allowedHeaders {
		allowedHeaders[i] = strings.TrimSpace(header)
	}
	
	config.CORS = CORSConfig{
		AllowedOrigins: allowedOrigins,
		AllowedMethods: allowedMethods,
		AllowedHeaders: allowedHeaders,
	}
	
	return config, nil
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.Auth.Secret == "" || c.Auth.Secret == "changeme_in_production" {
		return fmt.Errorf("AUTH_SECRET must be set to a secure value")
	}
	
	if c.Auth.PasswordMinLength < 6 {
		return fmt.Errorf("AUTH_PASSWORD_MIN_LENGTH must be at least 6")
	}
	
	if c.Database.URL == "" {
		return fmt.Errorf("DB_URL must be set")
	}
	
	if len(c.VM.HypervisorAddrs) == 0 {
		return fmt.Errorf("at least one hypervisor address must be configured")
	}
	
	validLogLevels := []string{"debug", "info", "warn", "error"}
	validLevel := false
	for _, level := range validLogLevels {
		if c.Logging.Level == level {
			validLevel = true
			break
		}
	}
	if !validLevel {
		return fmt.Errorf("LOG_LEVEL must be one of: %s", strings.Join(validLogLevels, ", "))
	}
	
	return nil
}

// Helper functions for environment variable parsing

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.Atoi(value); err == nil {
			return intVal
		}
	}
	return defaultValue
}

func getEnvBoolOrDefault(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolVal, err := strconv.ParseBool(value); err == nil {
			return boolVal
		}
	}
	return defaultValue
}

func getEnvDurationOrDefault(key string, defaultValue time.Duration) time.Duration {
	if value := os.Getenv(key); value != "" {
		if duration, err := time.ParseDuration(value); err == nil {
			return duration
		}
	}
	return defaultValue
}