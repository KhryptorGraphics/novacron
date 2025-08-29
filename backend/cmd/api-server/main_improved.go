package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gorilla/mux"
)

// Configuration structures
type Config struct {
	Server   ServerConfig
	Database DatabaseConfig
	Auth     AuthConfig
	Logging  LoggingConfig
	CORS     CORSConfig
}

type ServerConfig struct {
	APIPort         string
	WSPort          string
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	IdleTimeout     time.Duration
	ShutdownTimeout time.Duration
}

type DatabaseConfig struct {
	URL string
}

type AuthConfig struct {
	Secret string
}

type LoggingConfig struct {
	Level  string
	Format string
}

type CORSConfig struct {
	AllowedOrigins []string
	AllowedMethods []string
	AllowedHeaders []string
}

// Load configuration from environment variables
func loadConfig() (*Config, error) {
	config := &Config{
		Server: ServerConfig{
			APIPort:         getEnvOrDefault("API_PORT", "8090"),
			WSPort:          getEnvOrDefault("WS_PORT", "8091"),
			ReadTimeout:     getEnvDurationOrDefault("READ_TIMEOUT", 15*time.Second),
			WriteTimeout:    getEnvDurationOrDefault("WRITE_TIMEOUT", 15*time.Second),
			IdleTimeout:     getEnvDurationOrDefault("IDLE_TIMEOUT", 60*time.Second),
			ShutdownTimeout: getEnvDurationOrDefault("SHUTDOWN_TIMEOUT", 30*time.Second),
		},
		Database: DatabaseConfig{
			URL: getEnvOrDefault("DB_URL", "postgresql://postgres:postgres@postgres:5432/novacron"),
		},
		Auth: AuthConfig{
			Secret: getEnvOrDefault("AUTH_SECRET", "development_secret_change_in_production"),
		},
		Logging: LoggingConfig{
			Level:  getEnvOrDefault("LOG_LEVEL", "info"),
			Format: getEnvOrDefault("LOG_FORMAT", "json"),
		},
		CORS: CORSConfig{
			AllowedOrigins: strings.Split(getEnvOrDefault("CORS_ALLOWED_ORIGINS", "http://localhost:8092,http://localhost:3001"), ","),
			AllowedMethods: strings.Split(getEnvOrDefault("CORS_ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS"), ","),
			AllowedHeaders: strings.Split(getEnvOrDefault("CORS_ALLOWED_HEADERS", "Content-Type,Authorization"), ","),
		},
	}

	// Trim whitespace from CORS configuration
	for i, origin := range config.CORS.AllowedOrigins {
		config.CORS.AllowedOrigins[i] = strings.TrimSpace(origin)
	}
	for i, method := range config.CORS.AllowedMethods {
		config.CORS.AllowedMethods[i] = strings.TrimSpace(method)
	}
	for i, header := range config.CORS.AllowedHeaders {
		config.CORS.AllowedHeaders[i] = strings.TrimSpace(header)
	}

	return config, nil
}

// Structured logger
type Logger struct {
	level string
}

func NewLogger(level string) *Logger {
	return &Logger{level: level}
}

func (l *Logger) Info(msg string, keysAndValues ...interface{}) {
	entry := map[string]interface{}{
		"level":     "info",
		"message":   msg,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}
	
	// Add key-value pairs
	for i := 0; i < len(keysAndValues); i += 2 {
		if i+1 < len(keysAndValues) {
			key := fmt.Sprintf("%v", keysAndValues[i])
			entry[key] = keysAndValues[i+1]
		}
	}
	
	data, _ := json.Marshal(entry)
	fmt.Println(string(data))
}

func (l *Logger) Warn(msg string, keysAndValues ...interface{}) {
	entry := map[string]interface{}{
		"level":     "warn",
		"message":   msg,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}
	
	for i := 0; i < len(keysAndValues); i += 2 {
		if i+1 < len(keysAndValues) {
			key := fmt.Sprintf("%v", keysAndValues[i])
			entry[key] = keysAndValues[i+1]
		}
	}
	
	data, _ := json.Marshal(entry)
	fmt.Println(string(data))
}

func (l *Logger) Error(msg string, keysAndValues ...interface{}) {
	entry := map[string]interface{}{
		"level":     "error",
		"message":   msg,
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}
	
	for i := 0; i < len(keysAndValues); i += 2 {
		if i+1 < len(keysAndValues) {
			key := fmt.Sprintf("%v", keysAndValues[i])
			entry[key] = keysAndValues[i+1]
		}
	}
	
	data, _ := json.Marshal(entry)
	fmt.Println(string(data))
}

func (l *Logger) Fatal(msg string, keysAndValues ...interface{}) {
	l.Error(msg, keysAndValues...)
	os.Exit(1)
}

// CORS middleware
func corsMiddleware(allowedOrigins, allowedMethods, allowedHeaders []string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			origin := r.Header.Get("Origin")
			
			// Check if origin is allowed
			allowed := false
			for _, allowedOrigin := range allowedOrigins {
				if allowedOrigin == "*" || allowedOrigin == origin {
					allowed = true
					break
				}
			}
			
			if allowed {
				w.Header().Set("Access-Control-Allow-Origin", origin)
			}
			
			w.Header().Set("Access-Control-Allow-Methods", strings.Join(allowedMethods, ", "))
			w.Header().Set("Access-Control-Allow-Headers", strings.Join(allowedHeaders, ", "))
			w.Header().Set("Access-Control-Max-Age", "86400")
			
			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}
			
			next.ServeHTTP(w, r)
		})
	}
}

// Request ID middleware
func requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			requestID = fmt.Sprintf("req_%d", time.Now().UnixNano())
		}
		
		w.Header().Set("X-Request-ID", requestID)
		next.ServeHTTP(w, r)
	})
}

// Logging middleware
func loggingMiddleware(logger *Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			
			// Wrap response writer to capture status code
			wrapped := &responseWriter{ResponseWriter: w, statusCode: 200}
			
			next.ServeHTTP(wrapped, r)
			
			duration := time.Since(start)
			requestID := w.Header().Get("X-Request-ID")
			
			logger.Info("HTTP Request",
				"method", r.Method,
				"path", r.URL.Path,
				"status", wrapped.statusCode,
				"duration_ms", duration.Milliseconds(),
				"request_id", requestID,
				"remote_addr", r.RemoteAddr,
			)
		})
	}
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Health check handler
func healthCheckHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"status":    "healthy",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"version":   "1.0.0",
			"service":   "novacron-api",
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}
}

// API info handler
func apiInfoHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		response := map[string]interface{}{
			"name":        "NovaCron API",
			"version":     "1.0.0",
			"description": "Distributed VM Management System",
			"endpoints": []string{
				"/health",
				"/api/info",
				"/api/monitoring/metrics",
				"/api/monitoring/vms",
				"/api/monitoring/alerts",
			},
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(response)
	}
}

func main() {
	// Load configuration
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Initialize logger
	logger := NewLogger(cfg.Logging.Level)

	logger.Info("Starting NovaCron API Server",
		"version", "1.0.0",
		"api_port", cfg.Server.APIPort,
		"ws_port", cfg.Server.WSPort,
	)

	// Create router
	router := mux.NewRouter()

	// Add middleware stack
	handler := requestIDMiddleware(
		loggingMiddleware(logger)(
			corsMiddleware(
				cfg.CORS.AllowedOrigins,
				cfg.CORS.AllowedMethods,
				cfg.CORS.AllowedHeaders,
			)(router),
		),
	)

	// Register routes
	router.HandleFunc("/health", healthCheckHandler()).Methods("GET")
	router.HandleFunc("/api/info", apiInfoHandler()).Methods("GET")

	// Register mock monitoring endpoints
	registerMockHandlers(router)

	// Create HTTP server
	server := &http.Server{
		Addr:         ":" + cfg.Server.APIPort,
		Handler:      handler,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
	}

	// Start server in a goroutine
	go func() {
		logger.Info("API Server starting", "addr", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Server failed to start", "error", err)
		}
	}()

	// Wait for interrupt signal to gracefully shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logger.Fatal("Server forced to shutdown", "error", err)
	}

	logger.Info("Server exited gracefully")
}

// registerMockHandlers registers mock handlers for development
func registerMockHandlers(router *mux.Router) {
	// Mock system metrics
	router.HandleFunc("/api/monitoring/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `{
			"currentCpuUsage": 45.2,
			"currentMemoryUsage": 72.1,
			"currentDiskUsage": 58.3,
			"currentNetworkUsage": 125.7,
			"cpuChangePercentage": 5.2,
			"memoryChangePercentage": -2.1,
			"diskChangePercentage": 1.8,
			"networkChangePercentage": 12.5
		}`
		w.Write([]byte(response))
	}).Methods("GET")

	// Mock VM metrics
	router.HandleFunc("/api/monitoring/vms", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `[
			{
				"vmId": "vm-001",
				"name": "web-server-01",
				"cpuUsage": 78.5,
				"memoryUsage": 65.2,
				"diskUsage": 45.8,
				"status": "running"
			}
		]`
		w.Write([]byte(response))
	}).Methods("GET")

	// Mock alerts
	router.HandleFunc("/api/monitoring/alerts", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		response := `[
			{
				"id": "alert-001",
				"name": "High CPU Usage",
				"description": "VM web-server-01 CPU usage exceeds 75%",
				"severity": "warning",
				"status": "firing",
				"startTime": "2025-04-11T14:30:00Z",
				"resource": "VM web-server-01"
			}
		]`
		w.Write([]byte(response))
	}).Methods("GET")
}

// Helper functions
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
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

func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.Atoi(value); err == nil {
			return intVal
		}
	}
	return defaultValue
}