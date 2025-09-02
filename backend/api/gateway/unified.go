package gateway

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"
	"golang.org/x/time/rate"
	
	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// UnifiedGatewayConfig contains configuration for the unified API gateway
type UnifiedGatewayConfig struct {
	// Server configuration
	ListenAddr      string        `json:"listen_addr"`
	TLSEnabled      bool          `json:"tls_enabled"`
	CertFile        string        `json:"cert_file"`
	KeyFile         string        `json:"key_file"`
	ReadTimeout     time.Duration `json:"read_timeout"`
	WriteTimeout    time.Duration `json:"write_timeout"`
	IdleTimeout     time.Duration `json:"idle_timeout"`

	// Rate limiting
	RateLimitEnabled    bool          `json:"rate_limit_enabled"`
	RateLimitRPS        int           `json:"rate_limit_rps"`
	RateLimitBurst      int           `json:"rate_limit_burst"`
	RateLimitWindow     time.Duration `json:"rate_limit_window"`

	// Authentication
	AuthEnabled         bool   `json:"auth_enabled"`
	JWTSecret          string `json:"jwt_secret"`
	APIKeyHeader       string `json:"api_key_header"`
	AuthServiceURL     string `json:"auth_service_url"`

	// Load balancing
	LoadBalancingEnabled bool                    `json:"load_balancing_enabled"`
	HealthCheckInterval  time.Duration          `json:"health_check_interval"`
	HealthCheckTimeout   time.Duration          `json:"health_check_timeout"`

	// Monitoring
	MetricsEnabled      bool `json:"metrics_enabled"`
	TracingEnabled      bool `json:"tracing_enabled"`

	// CORS
	CORSEnabled         bool     `json:"cors_enabled"`
	CORSAllowedOrigins  []string `json:"cors_allowed_origins"`
	CORSAllowedMethods  []string `json:"cors_allowed_methods"`
	CORSAllowedHeaders  []string `json:"cors_allowed_headers"`
}

// DefaultUnifiedGatewayConfig returns default configuration
func DefaultUnifiedGatewayConfig() *UnifiedGatewayConfig {
	return &UnifiedGatewayConfig{
		ListenAddr:           ":8080",
		TLSEnabled:          false,
		ReadTimeout:         30 * time.Second,
		WriteTimeout:        30 * time.Second,
		IdleTimeout:         120 * time.Second,
		RateLimitEnabled:    true,
		RateLimitRPS:        1000,
		RateLimitBurst:      2000,
		RateLimitWindow:     time.Minute,
		AuthEnabled:         true,
		APIKeyHeader:        "X-API-Key",
		LoadBalancingEnabled: true,
		HealthCheckInterval: 30 * time.Second,
		HealthCheckTimeout:  5 * time.Second,
		MetricsEnabled:      true,
		TracingEnabled:      true,
		CORSEnabled:         true,
		CORSAllowedOrigins:  []string{"*"},
		CORSAllowedMethods:  []string{"GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"},
		CORSAllowedHeaders:  []string{"*"},
	}
}

// ServiceEndpoint represents a backend service endpoint
type ServiceEndpoint struct {
	Name        string `json:"name"`
	URL         string `json:"url"`
	Path        string `json:"path"`
	Methods     []string `json:"methods"`
	Healthy     bool   `json:"healthy"`
	LastChecked time.Time `json:"last_checked"`
	Weight      int    `json:"weight"`
}

// RateLimiter manages rate limiting for clients
type RateLimiter struct {
	limiters map[string]*rate.Limiter
	mu       sync.RWMutex
	rps      int
	burst    int
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(rps, burst int) *RateLimiter {
	return &RateLimiter{
		limiters: make(map[string]*rate.Limiter),
		rps:      rps,
		burst:    burst,
	}
}

// Allow checks if the request should be allowed
func (rl *RateLimiter) Allow(clientID string) bool {
	rl.mu.RLock()
	limiter, exists := rl.limiters[clientID]
	rl.mu.RUnlock()

	if !exists {
		rl.mu.Lock()
		// Double-check after acquiring write lock
		if limiter, exists = rl.limiters[clientID]; !exists {
			limiter = rate.NewLimiter(rate.Limit(rl.rps), rl.burst)
			rl.limiters[clientID] = limiter
		}
		rl.mu.Unlock()
	}

	return limiter.Allow()
}

// UnifiedAPIGateway provides a unified API gateway for all NovaCron services
type UnifiedAPIGateway struct {
	config     *UnifiedGatewayConfig
	logger     *logrus.Logger
	monitoring *monitoring.UnifiedMonitoringSystem

	// HTTP server
	server *http.Server
	router *mux.Router

	// Service registry
	services map[string]*ServiceEndpoint
	servicesMu sync.RWMutex

	// Rate limiting
	rateLimiter *RateLimiter

	// Load balancer
	loadBalancer *LoadBalancer

	// Context for lifecycle management
	ctx    context.Context
	cancel context.CancelFunc

	// Metrics
	requestCount   map[string]int64
	responseTime   map[string]time.Duration
	errorCount     map[string]int64
	metricsMu      sync.RWMutex
}

// LoadBalancer implements simple load balancing
type LoadBalancer struct {
	endpoints []*ServiceEndpoint
	current   int
	mu        sync.Mutex
}

// NextEndpoint returns the next available endpoint
func (lb *LoadBalancer) NextEndpoint() *ServiceEndpoint {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	if len(lb.endpoints) == 0 {
		return nil
	}

	// Simple round-robin
	for i := 0; i < len(lb.endpoints); i++ {
		lb.current = (lb.current + 1) % len(lb.endpoints)
		endpoint := lb.endpoints[lb.current]
		if endpoint.Healthy {
			return endpoint
		}
	}

	// Return first endpoint even if unhealthy
	return lb.endpoints[0]
}

// AddEndpoint adds an endpoint to the load balancer
func (lb *LoadBalancer) AddEndpoint(endpoint *ServiceEndpoint) {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	lb.endpoints = append(lb.endpoints, endpoint)
}

// NewUnifiedAPIGateway creates a new unified API gateway
func NewUnifiedAPIGateway(config *UnifiedGatewayConfig, logger *logrus.Logger, monitoring *monitoring.UnifiedMonitoringSystem) *UnifiedAPIGateway {
	if config == nil {
		config = DefaultUnifiedGatewayConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	gateway := &UnifiedAPIGateway{
		config:       config,
		logger:       logger,
		monitoring:   monitoring,
		ctx:          ctx,
		cancel:       cancel,
		services:     make(map[string]*ServiceEndpoint),
		requestCount: make(map[string]int64),
		responseTime: make(map[string]time.Duration),
		errorCount:   make(map[string]int64),
		loadBalancer: &LoadBalancer{},
	}

	if config.RateLimitEnabled {
		gateway.rateLimiter = NewRateLimiter(config.RateLimitRPS, config.RateLimitBurst)
	}

	gateway.setupRoutes()
	gateway.setupServer()

	return gateway
}

// Start begins the API gateway operations
func (gw *UnifiedAPIGateway) Start() error {
	gw.logger.WithField("addr", gw.config.ListenAddr).Info("Starting unified API gateway")

	// Register default services
	gw.registerDefaultServices()

	// Start health checking
	if gw.config.LoadBalancingEnabled {
		go gw.healthCheckLoop()
	}

	// Start metrics collection
	if gw.config.MetricsEnabled {
		go gw.metricsLoop()
	}

	// Start server
	if gw.config.TLSEnabled {
		return gw.server.ListenAndServeTLS(gw.config.CertFile, gw.config.KeyFile)
	}
	return gw.server.ListenAndServe()
}

// Stop gracefully shuts down the API gateway
func (gw *UnifiedAPIGateway) Stop() error {
	gw.logger.Info("Stopping unified API gateway")
	gw.cancel()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	return gw.server.Shutdown(ctx)
}

// RegisterService registers a new backend service
func (gw *UnifiedAPIGateway) RegisterService(service *ServiceEndpoint) error {
	gw.servicesMu.Lock()
	defer gw.servicesMu.Unlock()

	gw.services[service.Name] = service
	gw.loadBalancer.AddEndpoint(service)

	gw.logger.WithFields(logrus.Fields{
		"service": service.Name,
		"url":     service.URL,
		"path":    service.Path,
	}).Info("Service registered")

	return nil
}

// setupRoutes configures API routes
func (gw *UnifiedAPIGateway) setupRoutes() {
	gw.router = mux.NewRouter()

	// Add middleware
	gw.router.Use(gw.corsMiddleware)
	gw.router.Use(gw.loggingMiddleware)
	gw.router.Use(gw.metricsMiddleware)
	
	if gw.config.TracingEnabled && gw.monitoring != nil {
		gw.router.Use(gw.tracingMiddleware)
	}
	
	if gw.config.RateLimitEnabled {
		gw.router.Use(gw.rateLimitMiddleware)
	}
	
	if gw.config.AuthEnabled {
		gw.router.Use(gw.authMiddleware)
	}

	// API routes
	api := gw.router.PathPrefix("/api/v1").Subrouter()

	// VM Management API
	vm := api.PathPrefix("/vms").Subrouter()
	vm.HandleFunc("", gw.proxyHandler("vm-service")).Methods("GET", "POST")
	vm.HandleFunc("/{id}", gw.proxyHandler("vm-service")).Methods("GET", "PUT", "DELETE")
	vm.HandleFunc("/{id}/start", gw.proxyHandler("vm-service")).Methods("POST")
	vm.HandleFunc("/{id}/stop", gw.proxyHandler("vm-service")).Methods("POST")
	vm.HandleFunc("/{id}/migrate", gw.proxyHandler("vm-service")).Methods("POST")

	// Orchestration API
	orchestration := api.PathPrefix("/orchestration").Subrouter()
	orchestration.HandleFunc("/policies", gw.proxyHandler("orchestration-service")).Methods("GET", "POST")
	orchestration.HandleFunc("/policies/{id}", gw.proxyHandler("orchestration-service")).Methods("GET", "PUT", "DELETE")
	orchestration.HandleFunc("/decisions", gw.proxyHandler("orchestration-service")).Methods("POST")
	orchestration.HandleFunc("/status", gw.proxyHandler("orchestration-service")).Methods("GET")

	// ML API
	ml := api.PathPrefix("/ml").Subrouter()
	ml.HandleFunc("/models", gw.proxyHandler("ml-service")).Methods("GET", "POST")
	ml.HandleFunc("/models/{type}/train", gw.proxyHandler("ml-service")).Methods("POST")
	ml.HandleFunc("/models/{type}/predict", gw.proxyHandler("ml-service")).Methods("POST")
	ml.HandleFunc("/models/{type}/performance", gw.proxyHandler("ml-service")).Methods("GET")

	// Federation API
	federation := api.PathPrefix("/federation").Subrouter()
	federation.HandleFunc("/clusters", gw.proxyHandler("federation-service")).Methods("GET", "POST")
	federation.HandleFunc("/clusters/{id}", gw.proxyHandler("federation-service")).Methods("GET", "PUT", "DELETE")
	federation.HandleFunc("/clusters/{id}/sync", gw.proxyHandler("federation-service")).Methods("POST")

	// Backup API
	backup := api.PathPrefix("/backup").Subrouter()
	backup.HandleFunc("/jobs", gw.proxyHandler("backup-service")).Methods("GET", "POST")
	backup.HandleFunc("/jobs/{id}", gw.proxyHandler("backup-service")).Methods("GET", "DELETE")
	backup.HandleFunc("/restore", gw.proxyHandler("backup-service")).Methods("POST")

	// Monitoring API
	monitoring := api.PathPrefix("/monitoring").Subrouter()
	monitoring.HandleFunc("/metrics", gw.monitoringHandler).Methods("GET")
	monitoring.HandleFunc("/health", gw.healthHandler).Methods("GET")
	monitoring.HandleFunc("/status", gw.statusHandler).Methods("GET")
	monitoring.HandleFunc("/alerts", gw.alertsHandler).Methods("GET")

	// Gateway management API
	gateway := api.PathPrefix("/gateway").Subrouter()
	gateway.HandleFunc("/services", gw.listServicesHandler).Methods("GET")
	gateway.HandleFunc("/services/{name}", gw.getServiceHandler).Methods("GET")
	gateway.HandleFunc("/stats", gw.gatewayStatsHandler).Methods("GET")
}

// setupServer configures the HTTP server
func (gw *UnifiedAPIGateway) setupServer() {
	gw.server = &http.Server{
		Addr:         gw.config.ListenAddr,
		Handler:      gw.router,
		ReadTimeout:  gw.config.ReadTimeout,
		WriteTimeout: gw.config.WriteTimeout,
		IdleTimeout:  gw.config.IdleTimeout,
	}

	// Configure TLS if enabled
	if gw.config.TLSEnabled {
		gw.server.TLSConfig = &tls.Config{
			MinVersion: tls.VersionTLS12,
			CipherSuites: []uint16{
				tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
				tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
				tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
			},
		}
	}
}

// registerDefaultServices registers default backend services
func (gw *UnifiedAPIGateway) registerDefaultServices() {
	services := []*ServiceEndpoint{
		{
			Name:    "vm-service",
			URL:     "http://localhost:8081",
			Path:    "/api/v1/vms",
			Methods: []string{"GET", "POST", "PUT", "DELETE"},
			Healthy: true,
			Weight:  100,
		},
		{
			Name:    "orchestration-service",
			URL:     "http://localhost:8082",
			Path:    "/api/v1/orchestration",
			Methods: []string{"GET", "POST", "PUT", "DELETE"},
			Healthy: true,
			Weight:  100,
		},
		{
			Name:    "ml-service",
			URL:     "http://localhost:8083",
			Path:    "/api/v1/ml",
			Methods: []string{"GET", "POST"},
			Healthy: true,
			Weight:  100,
		},
		{
			Name:    "federation-service",
			URL:     "http://localhost:8084",
			Path:    "/api/v1/federation",
			Methods: []string{"GET", "POST", "PUT", "DELETE"},
			Healthy: true,
			Weight:  100,
		},
		{
			Name:    "backup-service",
			URL:     "http://localhost:8085",
			Path:    "/api/v1/backup",
			Methods: []string{"GET", "POST", "DELETE"},
			Healthy: true,
			Weight:  100,
		},
	}

	for _, service := range services {
		gw.RegisterService(service)
	}
}

// Middleware implementations

func (gw *UnifiedAPIGateway) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !gw.config.CORSEnabled {
			next.ServeHTTP(w, r)
			return
		}

		origin := r.Header.Get("Origin")
		if origin != "" && gw.isOriginAllowed(origin) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		}

		w.Header().Set("Access-Control-Allow-Methods", strings.Join(gw.config.CORSAllowedMethods, ", "))
		w.Header().Set("Access-Control-Allow-Headers", strings.Join(gw.config.CORSAllowedHeaders, ", "))
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (gw *UnifiedAPIGateway) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap response writer to capture status code
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		duration := time.Since(start)

		gw.logger.WithFields(logrus.Fields{
			"method":      r.Method,
			"path":        r.URL.Path,
			"status":      wrapped.statusCode,
			"duration_ms": duration.Milliseconds(),
			"remote_addr": r.RemoteAddr,
			"user_agent":  r.Header.Get("User-Agent"),
		}).Info("API request")
	})
}

func (gw *UnifiedAPIGateway) metricsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !gw.config.MetricsEnabled || gw.monitoring == nil {
			next.ServeHTTP(w, r)
			return
		}

		start := time.Now()
		wrapped := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(wrapped, r)

		duration := time.Since(start)
		
		// Record metrics
		labels := map[string]string{
			"method":   r.Method,
			"endpoint": r.URL.Path,
			"status":   strconv.Itoa(wrapped.statusCode),
		}

		gw.monitoring.RecordMetric("gateway_requests_total", 1, labels)
		gw.monitoring.RecordMetric("gateway_request_duration_seconds", duration.Seconds(), labels)

		if wrapped.statusCode >= 400 {
			gw.monitoring.RecordMetric("gateway_errors_total", 1, labels)
		}

		// Update internal metrics
		gw.updateInternalMetrics(r.URL.Path, duration, wrapped.statusCode >= 400)
	})
}

func (gw *UnifiedAPIGateway) tracingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, span := gw.monitoring.StartTrace(r.Context(), fmt.Sprintf("gateway_%s_%s", r.Method, r.URL.Path))
		defer span.End()

		r = r.WithContext(ctx)
		next.ServeHTTP(w, r)
	})
}

func (gw *UnifiedAPIGateway) rateLimitMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if gw.rateLimiter == nil {
			next.ServeHTTP(w, r)
			return
		}

		// Use client IP as identifier
		clientID := gw.getClientID(r)
		
		if !gw.rateLimiter.Allow(clientID) {
			http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
			
			if gw.monitoring != nil {
				gw.monitoring.RecordMetric("gateway_rate_limit_exceeded_total", 1, map[string]string{
					"client_id": clientID,
				})
			}
			return
		}

		next.ServeHTTP(w, r)
	})
}

func (gw *UnifiedAPIGateway) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Skip auth for health check endpoints
		if strings.HasSuffix(r.URL.Path, "/health") || strings.HasSuffix(r.URL.Path, "/metrics") {
			next.ServeHTTP(w, r)
			return
		}

		// Check API key
		apiKey := r.Header.Get(gw.config.APIKeyHeader)
		if apiKey == "" {
			// Check Authorization header
			authHeader := r.Header.Get("Authorization")
			if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
				http.Error(w, "Authentication required", http.StatusUnauthorized)
				return
			}
			
			// Extract token
			token := strings.TrimPrefix(authHeader, "Bearer ")
			if !gw.validateToken(token) {
				http.Error(w, "Invalid token", http.StatusUnauthorized)
				return
			}
		} else if !gw.validateAPIKey(apiKey) {
			http.Error(w, "Invalid API key", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// Handler implementations

func (gw *UnifiedAPIGateway) proxyHandler(serviceName string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		gw.servicesMu.RLock()
		service, exists := gw.services[serviceName]
		gw.servicesMu.RUnlock()

		if !exists {
			http.Error(w, "Service not found", http.StatusNotFound)
			return
		}

		if !service.Healthy && gw.config.LoadBalancingEnabled {
			// Try to find a healthy endpoint
			endpoint := gw.loadBalancer.NextEndpoint()
			if endpoint != nil {
				service = endpoint
			}
		}

		// Parse target URL
		target, err := url.Parse(service.URL)
		if err != nil {
			http.Error(w, "Invalid service URL", http.StatusInternalServerError)
			return
		}

		// Create reverse proxy
		proxy := httputil.NewSingleHostReverseProxy(target)
		proxy.ErrorHandler = gw.proxyErrorHandler

		// Modify request
		r.URL.Host = target.Host
		r.URL.Scheme = target.Scheme
		r.Header.Set("X-Forwarded-Host", r.Header.Get("Host"))
		r.Host = target.Host

		proxy.ServeHTTP(w, r)
	}
}

func (gw *UnifiedAPIGateway) proxyErrorHandler(w http.ResponseWriter, r *http.Request, err error) {
	gw.logger.WithFields(logrus.Fields{
		"error": err,
		"path":  r.URL.Path,
		"host":  r.URL.Host,
	}).Error("Proxy error")

	http.Error(w, "Service unavailable", http.StatusServiceUnavailable)

	if gw.monitoring != nil {
		gw.monitoring.RecordMetric("gateway_proxy_errors_total", 1, map[string]string{
			"error": "proxy_error",
		})
	}
}

func (gw *UnifiedAPIGateway) healthHandler(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now(),
		"services":  gw.getServiceHealth(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

func (gw *UnifiedAPIGateway) statusHandler(w http.ResponseWriter, r *http.Request) {
	gw.metricsMu.RLock()
	status := map[string]interface{}{
		"uptime":        time.Since(gw.server.BaseContext().Value("start_time").(time.Time)),
		"request_count": gw.requestCount,
		"error_count":   gw.errorCount,
		"services":      len(gw.services),
	}
	gw.metricsMu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

func (gw *UnifiedAPIGateway) monitoringHandler(w http.ResponseWriter, r *http.Request) {
	if gw.monitoring == nil {
		http.Error(w, "Monitoring not enabled", http.StatusServiceUnavailable)
		return
	}

	metrics := gw.monitoring.GetSystemMetrics()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (gw *UnifiedAPIGateway) alertsHandler(w http.ResponseWriter, r *http.Request) {
	if gw.monitoring == nil {
		http.Error(w, "Monitoring not enabled", http.StatusServiceUnavailable)
		return
	}

	health := gw.monitoring.GetSystemHealth()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"alerts": health.Alerts,
		"count":  len(health.Alerts),
	})
}

func (gw *UnifiedAPIGateway) listServicesHandler(w http.ResponseWriter, r *http.Request) {
	gw.servicesMu.RLock()
	services := make([]*ServiceEndpoint, 0, len(gw.services))
	for _, service := range gw.services {
		services = append(services, service)
	}
	gw.servicesMu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(services)
}

func (gw *UnifiedAPIGateway) getServiceHandler(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	serviceName := vars["name"]

	gw.servicesMu.RLock()
	service, exists := gw.services[serviceName]
	gw.servicesMu.RUnlock()

	if !exists {
		http.Error(w, "Service not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(service)
}

func (gw *UnifiedAPIGateway) gatewayStatsHandler(w http.ResponseWriter, r *http.Request) {
	gw.metricsMu.RLock()
	stats := map[string]interface{}{
		"total_requests": gw.getTotalRequests(),
		"total_errors":   gw.getTotalErrors(),
		"avg_response_time": gw.getAverageResponseTime(),
		"services_healthy": gw.getHealthyServicesCount(),
		"rate_limit_config": map[string]interface{}{
			"enabled": gw.config.RateLimitEnabled,
			"rps":     gw.config.RateLimitRPS,
			"burst":   gw.config.RateLimitBurst,
		},
	}
	gw.metricsMu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// Helper methods

func (gw *UnifiedAPIGateway) getClientID(r *http.Request) string {
	// Extract client IP
	ip := r.Header.Get("X-Real-IP")
	if ip == "" {
		ip = r.Header.Get("X-Forwarded-For")
		if ip == "" {
			ip = r.RemoteAddr
		}
	}

	// Extract API key if present for more granular limiting
	apiKey := r.Header.Get(gw.config.APIKeyHeader)
	if apiKey != "" {
		return fmt.Sprintf("%s:%s", ip, apiKey)
	}

	return ip
}

func (gw *UnifiedAPIGateway) isOriginAllowed(origin string) bool {
	for _, allowed := range gw.config.CORSAllowedOrigins {
		if allowed == "*" || allowed == origin {
			return true
		}
	}
	return false
}

func (gw *UnifiedAPIGateway) validateAPIKey(apiKey string) bool {
	// Implement API key validation logic
	// This is a simplified implementation
	return apiKey != "" && len(apiKey) >= 32
}

func (gw *UnifiedAPIGateway) validateToken(token string) bool {
	// Implement JWT token validation logic
	// This is a simplified implementation
	return token != "" && len(token) > 20
}

func (gw *UnifiedAPIGateway) updateInternalMetrics(path string, duration time.Duration, isError bool) {
	gw.metricsMu.Lock()
	defer gw.metricsMu.Unlock()

	gw.requestCount[path]++
	gw.responseTime[path] = duration

	if isError {
		gw.errorCount[path]++
	}
}

func (gw *UnifiedAPIGateway) healthCheckLoop() {
	ticker := time.NewTicker(gw.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-gw.ctx.Done():
			return
		case <-ticker.C:
			gw.performHealthChecks()
		}
	}
}

func (gw *UnifiedAPIGateway) performHealthChecks() {
	gw.servicesMu.Lock()
	defer gw.servicesMu.Unlock()

	for _, service := range gw.services {
		go gw.checkServiceHealth(service)
	}
}

func (gw *UnifiedAPIGateway) checkServiceHealth(service *ServiceEndpoint) {
	client := &http.Client{
		Timeout: gw.config.HealthCheckTimeout,
	}

	healthURL := fmt.Sprintf("%s/health", service.URL)
	resp, err := client.Get(healthURL)
	
	healthy := err == nil && resp != nil && resp.StatusCode == http.StatusOK

	if resp != nil {
		resp.Body.Close()
	}

	service.Healthy = healthy
	service.LastChecked = time.Now()

	if gw.monitoring != nil {
		status := "healthy"
		if !healthy {
			status = "unhealthy"
		}

		gw.monitoring.RecordMetric("gateway_service_health", 1, map[string]string{
			"service": service.Name,
			"status":  status,
		})
	}
}

func (gw *UnifiedAPIGateway) metricsLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-gw.ctx.Done():
			return
		case <-ticker.C:
			gw.recordAggregateMetrics()
		}
	}
}

func (gw *UnifiedAPIGateway) recordAggregateMetrics() {
	if gw.monitoring == nil {
		return
	}

	gw.metricsMu.RLock()
	totalRequests := gw.getTotalRequests()
	totalErrors := gw.getTotalErrors()
	avgResponseTime := gw.getAverageResponseTime()
	gw.metricsMu.RUnlock()

	gw.monitoring.RecordMetric("gateway_total_requests", float64(totalRequests), nil)
	gw.monitoring.RecordMetric("gateway_total_errors", float64(totalErrors), nil)
	gw.monitoring.RecordMetric("gateway_avg_response_time_seconds", avgResponseTime.Seconds(), nil)

	healthyServices := gw.getHealthyServicesCount()
	totalServices := len(gw.services)
	
	gw.monitoring.RecordMetric("gateway_services_total", float64(totalServices), nil)
	gw.monitoring.RecordMetric("gateway_services_healthy", float64(healthyServices), nil)
}

func (gw *UnifiedAPIGateway) getServiceHealth() map[string]interface{} {
	gw.servicesMu.RLock()
	defer gw.servicesMu.RUnlock()

	health := make(map[string]interface{})
	for name, service := range gw.services {
		health[name] = map[string]interface{}{
			"healthy":      service.Healthy,
			"last_checked": service.LastChecked,
			"url":          service.URL,
		}
	}

	return health
}

func (gw *UnifiedAPIGateway) getTotalRequests() int64 {
	var total int64
	for _, count := range gw.requestCount {
		total += count
	}
	return total
}

func (gw *UnifiedAPIGateway) getTotalErrors() int64 {
	var total int64
	for _, count := range gw.errorCount {
		total += count
	}
	return total
}

func (gw *UnifiedAPIGateway) getAverageResponseTime() time.Duration {
	if len(gw.responseTime) == 0 {
		return 0
	}

	var total time.Duration
	for _, duration := range gw.responseTime {
		total += duration
	}

	return total / time.Duration(len(gw.responseTime))
}

func (gw *UnifiedAPIGateway) getHealthyServicesCount() int {
	gw.servicesMu.RLock()
	defer gw.servicesMu.RUnlock()

	count := 0
	for _, service := range gw.services {
		if service.Healthy {
			count++
		}
	}
	return count
}

// responseWriter wraps http.ResponseWriter to capture status code
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}