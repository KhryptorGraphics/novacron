package loadbalancer

import (
	"context"
	"crypto/tls"
	"fmt"
	"hash/fnv"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// L4L7LoadBalancer provides Layer 4 and Layer 7 load balancing capabilities
type L4L7LoadBalancer struct {
	// Configuration
	config LoadBalancerConfig
	
	// State management
	services       map[string]*LoadBalancerService
	backends       map[string]*Backend
	healthCheckers map[string]*HealthChecker
	listeners      map[string]*Listener
	servicesMutex  sync.RWMutex
	backendsMutex  sync.RWMutex
	checkersMutex  sync.RWMutex
	listenersMutex sync.RWMutex
	
	// Statistics
	stats          *LoadBalancerStatistics
	statsMutex     sync.RWMutex
	
	// Runtime state
	ctx         context.Context
	cancel      context.CancelFunc
	initialized bool
}

// LoadBalancerConfig holds configuration for the load balancer
type LoadBalancerConfig struct {
	// Basic settings
	ListenAddress           string        `json:"listen_address"`
	DefaultAlgorithm        Algorithm     `json:"default_algorithm"`
	HealthCheckInterval     time.Duration `json:"health_check_interval"`
	HealthCheckTimeout      time.Duration `json:"health_check_timeout"`
	HealthCheckRetries      int           `json:"health_check_retries"`
	
	// Connection settings
	MaxConnections          int           `json:"max_connections"`
	ConnectionTimeout       time.Duration `json:"connection_timeout"`
	IdleTimeout            time.Duration `json:"idle_timeout"`
	KeepAliveTimeout       time.Duration `json:"keep_alive_timeout"`
	
	// SSL/TLS settings
	EnableSSL              bool          `json:"enable_ssl"`
	SSLCertPath            string        `json:"ssl_cert_path"`
	SSLKeyPath             string        `json:"ssl_key_path"`
	SSLMinVersion          uint16        `json:"ssl_min_version"`
	SSLMaxVersion          uint16        `json:"ssl_max_version"`
	SSLCipherSuites        []uint16      `json:"ssl_cipher_suites"`
	
	// Advanced features
	EnableStickySessions   bool          `json:"enable_sticky_sessions"`
	SessionAffinityTimeout time.Duration `json:"session_affinity_timeout"`
	EnableRateLimiting     bool          `json:"enable_rate_limiting"`
	RateLimitRPS           int           `json:"rate_limit_rps"`
	EnableCompression      bool          `json:"enable_compression"`
	CompressionLevel       int           `json:"compression_level"`
	
	// Monitoring
	EnableMetrics          bool          `json:"enable_metrics"`
	MetricsInterval        time.Duration `json:"metrics_interval"`
	EnableLogging          bool          `json:"enable_logging"`
	LogLevel               string        `json:"log_level"`
}

// DefaultLoadBalancerConfig returns default configuration
func DefaultLoadBalancerConfig() LoadBalancerConfig {
	return LoadBalancerConfig{
		ListenAddress:           "0.0.0.0:80",
		DefaultAlgorithm:        AlgorithmRoundRobin,
		HealthCheckInterval:     30 * time.Second,
		HealthCheckTimeout:      5 * time.Second,
		HealthCheckRetries:      3,
		MaxConnections:          10000,
		ConnectionTimeout:       30 * time.Second,
		IdleTimeout:            90 * time.Second,
		KeepAliveTimeout:       30 * time.Second,
		EnableSSL:              false,
		SSLMinVersion:          tls.VersionTLS12,
		SSLMaxVersion:          tls.VersionTLS13,
		EnableStickySessions:   false,
		SessionAffinityTimeout: 3600 * time.Second,
		EnableRateLimiting:     false,
		RateLimitRPS:           1000,
		EnableCompression:      true,
		CompressionLevel:       6,
		EnableMetrics:          true,
		MetricsInterval:        30 * time.Second,
		EnableLogging:          true,
		LogLevel:               "info",
	}
}

// LoadBalancerService represents a load balancer service
type LoadBalancerService struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	Type            ServiceType            `json:"type"`
	Protocol        Protocol               `json:"protocol"`
	ListenPort      int                    `json:"listen_port"`
	TargetPort      int                    `json:"target_port,omitempty"`
	Algorithm       Algorithm              `json:"algorithm"`
	Backends        []string               `json:"backends"`
	HealthCheck     HealthCheckConfig      `json:"health_check"`
	SSLConfig       *SSLConfig             `json:"ssl_config,omitempty"`
	Rules           []*LoadBalancingRule   `json:"rules,omitempty"`
	SessionAffinity *SessionAffinityConfig `json:"session_affinity,omitempty"`
	RateLimiting    *RateLimitConfig       `json:"rate_limiting,omitempty"`
	Statistics      ServiceStatistics      `json:"statistics"`
	Status          ServiceStatus          `json:"status"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// Backend represents a backend server
type Backend struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Address       string                 `json:"address"`
	Port          int                    `json:"port"`
	Weight        int                    `json:"weight"`
	Status        BackendStatus          `json:"status"`
	HealthStatus  L4L7HealthStatus       `json:"health_status"`
	MaxConns      int                    `json:"max_conns"`
	CurrentConns  int64                  `json:"current_conns"`
	Statistics    BackendStatistics      `json:"statistics"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
	LastChecked   time.Time              `json:"last_checked"`
}

// HealthChecker manages health checking for backends
type HealthChecker struct {
	ID              string                 `json:"id"`
	ServiceID       string                 `json:"service_id"`
	BackendID       string                 `json:"backend_id"`
	Config          HealthCheckConfig      `json:"config"`
	Status          L4L7HealthStatus       `json:"status"`
	ConsecutiveFails int                   `json:"consecutive_fails"`
	ConsecutivePasses int                  `json:"consecutive_passes"`
	LastCheck       time.Time              `json:"last_check"`
	LastError       string                 `json:"last_error,omitempty"`
	Statistics      HealthCheckStatistics  `json:"statistics"`
	ticker          *time.Ticker
	stopCh          chan struct{}
}

// Listener handles incoming connections for a service
type Listener struct {
	ID          string                 `json:"id"`
	ServiceID   string                 `json:"service_id"`
	Address     string                 `json:"address"`
	Port        int                    `json:"port"`
	Protocol    Protocol               `json:"protocol"`
	TLSConfig   *tls.Config            `json:"-"`
	listener    net.Listener
	httpServer  *http.Server
	Statistics  ListenerStatistics     `json:"statistics"`
	Status      ListenerStatus         `json:"status"`
	CreatedAt   time.Time              `json:"created_at"`
}

// Enums and types
type ServiceType string
type Protocol string
type Algorithm string
type ServiceStatus string
type BackendStatus string
type L4L7HealthStatus string  // Rename to avoid conflict with common HealthStatus
type ListenerStatus string

const (
	ServiceTypeHTTP     ServiceType = "http"
	ServiceTypeHTTPS    ServiceType = "https"
	ServiceTypeTCP      ServiceType = "tcp"
	ServiceTypeUDP      ServiceType = "udp"
	ServiceTypeGRPC     ServiceType = "grpc"
	
	ProtocolHTTP        Protocol = "http"
	ProtocolHTTPS       Protocol = "https"
	ProtocolTCP         Protocol = "tcp"
	ProtocolUDP         Protocol = "udp"
	ProtocolGRPC        Protocol = "grpc"
	
	AlgorithmRoundRobin      Algorithm = "round_robin"
	AlgorithmWeightedRoundRobin Algorithm = "weighted_round_robin"
	AlgorithmLeastConnections Algorithm = "least_connections"
	AlgorithmWeightedLeastConnections Algorithm = "weighted_least_connections"
	AlgorithmIPHash          Algorithm = "ip_hash"
	AlgorithmURLHash         Algorithm = "url_hash"
	AlgorithmConsistentHash  Algorithm = "consistent_hash"
	AlgorithmRandom          Algorithm = "random"
	AlgorithmWeightedRandom  Algorithm = "weighted_random"
	
	ServiceStatusActive    ServiceStatus = "active"
	ServiceStatusInactive  ServiceStatus = "inactive"
	ServiceStatusError     ServiceStatus = "error"
	
	BackendStatusActive    BackendStatus = "active"
	BackendStatusInactive  BackendStatus = "inactive"
	BackendStatusDraining  BackendStatus = "draining"
	
	HealthStatusHealthy    L4L7HealthStatus = "healthy"
	HealthStatusUnhealthy  L4L7HealthStatus = "unhealthy"
	HealthStatusUnknown    L4L7HealthStatus = "unknown"
	HealthStatusMaintenance L4L7HealthStatus = "maintenance"
	
	ListenerStatusActive   ListenerStatus = "active"
	ListenerStatusInactive ListenerStatus = "inactive"
	ListenerStatusError    ListenerStatus = "error"
)

// Configuration structures
type HealthCheckConfig struct {
	Type           HealthCheckType    `json:"type"`
	Interval       time.Duration      `json:"interval"`
	Timeout        time.Duration      `json:"timeout"`
	HealthyThreshold int              `json:"healthy_threshold"`
	UnhealthyThreshold int            `json:"unhealthy_threshold"`
	Path           string             `json:"path,omitempty"`
	ExpectedStatus int                `json:"expected_status,omitempty"`
	ExpectedBody   string             `json:"expected_body,omitempty"`
	Headers        map[string]string  `json:"headers,omitempty"`
}

type HealthCheckType string

const (
	HealthCheckTypeHTTP HealthCheckType = "http"
	HealthCheckTypeHTTPS HealthCheckType = "https"
	HealthCheckTypeTCP  HealthCheckType = "tcp"
	HealthCheckTypeUDP  HealthCheckType = "udp"
	HealthCheckTypeICMP HealthCheckType = "icmp"
)

type SSLConfig struct {
	CertPath       string   `json:"cert_path"`
	KeyPath        string   `json:"key_path"`
	CAPath         string   `json:"ca_path,omitempty"`
	MinVersion     uint16   `json:"min_version"`
	MaxVersion     uint16   `json:"max_version"`
	CipherSuites   []uint16 `json:"cipher_suites,omitempty"`
	ClientAuth     string   `json:"client_auth,omitempty"`
}

type LoadBalancingRule struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Priority    int                    `json:"priority"`
	Conditions  []RuleCondition        `json:"conditions"`
	Actions     []RuleAction           `json:"actions"`
	Enabled     bool                   `json:"enabled"`
	Statistics  RuleStatistics         `json:"statistics"`
	CreatedAt   time.Time              `json:"created_at"`
}

type RuleCondition struct {
	Type      ConditionType          `json:"type"`
	Field     string                 `json:"field"`
	Operator  ConditionOperator      `json:"operator"`
	Value     string                 `json:"value"`
	CaseSensitive bool               `json:"case_sensitive"`
}

type RuleAction struct {
	Type       L4L7ActionType         `json:"type"`
	Target     string                 `json:"target,omitempty"`
	Value      string                 `json:"value,omitempty"`
	Headers    map[string]string      `json:"headers,omitempty"`
}

type ConditionType string
type ConditionOperator string
type L4L7ActionType string  // Rename to avoid conflict with common ActionType

const (
	ConditionTypeHost         ConditionType = "host"
	ConditionTypePath         ConditionType = "path"
	ConditionTypeHeader       ConditionType = "header"
	ConditionTypeQuery        ConditionType = "query"
	ConditionTypeMethod       ConditionType = "method"
	ConditionTypeSourceIP     ConditionType = "source_ip"
	
	OperatorEquals       ConditionOperator = "equals"
	OperatorContains     ConditionOperator = "contains"
	OperatorStartsWith   ConditionOperator = "starts_with"
	OperatorEndsWith     ConditionOperator = "ends_with"
	OperatorRegex        ConditionOperator = "regex"
	OperatorIn           ConditionOperator = "in"
	
	ActionTypeForward        L4L7ActionType = "forward"
	ActionTypeRedirect       L4L7ActionType = "redirect"
	ActionTypeRewrite        L4L7ActionType = "rewrite"
	ActionTypeAddHeader      L4L7ActionType = "add_header"
	ActionTypeRemoveHeader   L4L7ActionType = "remove_header"
	ActionTypeDeny           L4L7ActionType = "deny"
	ActionTypeRateLimit      L4L7ActionType = "rate_limit"
)

type SessionAffinityConfig struct {
	Method         AffinityMethod    `json:"method"`
	CookieName     string           `json:"cookie_name,omitempty"`
	CookieTimeout  time.Duration    `json:"cookie_timeout,omitempty"`
	HeaderName     string           `json:"header_name,omitempty"`
}

// AffinityMethod is now defined in types.go
const (
	AffinityMethodClientIP   AffinityMethod = AffinityMethodSourceIP  // Alias for compatibility
)

type RateLimitConfig struct {
	RequestsPerSecond int           `json:"requests_per_second"`
	BurstSize         int           `json:"burst_size"`
	WindowSize        time.Duration `json:"window_size"`
	KeyExtractor      string        `json:"key_extractor"`
}

// Statistics structures
type LoadBalancerStatistics struct {
	TotalConnections     int64     `json:"total_connections"`
	ActiveConnections    int64     `json:"active_connections"`
	TotalRequests        int64     `json:"total_requests"`
	TotalErrors          int64     `json:"total_errors"`
	TotalBytesIn         int64     `json:"total_bytes_in"`
	TotalBytesOut        int64     `json:"total_bytes_out"`
	AverageResponseTime  float64   `json:"average_response_time_ms"`
	RequestsPerSecond    float64   `json:"requests_per_second"`
	ErrorRate            float64   `json:"error_rate_percent"`
	LastUpdated          time.Time `json:"last_updated"`
}

type ServiceStatistics struct {
	TotalConnections     int64     `json:"total_connections"`
	ActiveConnections    int64     `json:"active_connections"`
	TotalRequests        int64     `json:"total_requests"`
	TotalErrors          int64     `json:"total_errors"`
	TotalBytesIn         int64     `json:"total_bytes_in"`
	TotalBytesOut        int64     `json:"total_bytes_out"`
	AverageResponseTime  float64   `json:"average_response_time_ms"`
	LastUpdated          time.Time `json:"last_updated"`
}

type BackendStatistics struct {
	TotalConnections     int64     `json:"total_connections"`
	ActiveConnections    int64     `json:"active_connections"`
	TotalRequests        int64     `json:"total_requests"`
	TotalErrors          int64     `json:"total_errors"`
	TotalBytesIn         int64     `json:"total_bytes_in"`
	TotalBytesOut        int64     `json:"total_bytes_out"`
	AverageResponseTime  float64   `json:"average_response_time_ms"`
	HealthChecksPassed   int64     `json:"health_checks_passed"`
	HealthChecksFailed   int64     `json:"health_checks_failed"`
	LastUpdated          time.Time `json:"last_updated"`
}

type ListenerStatistics struct {
	TotalConnections     int64     `json:"total_connections"`
	ActiveConnections    int64     `json:"active_connections"`
	TotalBytesIn         int64     `json:"total_bytes_in"`
	TotalBytesOut        int64     `json:"total_bytes_out"`
	LastUpdated          time.Time `json:"last_updated"`
}

type HealthCheckStatistics struct {
	TotalChecks          int64     `json:"total_checks"`
	PassedChecks         int64     `json:"passed_checks"`
	FailedChecks         int64     `json:"failed_checks"`
	AverageResponseTime  float64   `json:"average_response_time_ms"`
	LastUpdated          time.Time `json:"last_updated"`
}

type RuleStatistics struct {
	TotalMatches         int64     `json:"total_matches"`
	TotalActions         int64     `json:"total_actions"`
	LastUpdated          time.Time `json:"last_updated"`
}

// NewL4L7LoadBalancer creates a new L4/L7 load balancer
func NewL4L7LoadBalancer(config LoadBalancerConfig) *L4L7LoadBalancer {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &L4L7LoadBalancer{
		config:         config,
		services:       make(map[string]*LoadBalancerService),
		backends:       make(map[string]*Backend),
		healthCheckers: make(map[string]*HealthChecker),
		listeners:      make(map[string]*Listener),
		stats:          &LoadBalancerStatistics{LastUpdated: time.Now()},
		ctx:            ctx,
		cancel:         cancel,
		initialized:    false,
	}
}

// Start starts the load balancer
func (lb *L4L7LoadBalancer) Start() error {
	if lb.initialized {
		return fmt.Errorf("load balancer already started")
	}
	
	log.Println("Starting L4/L7 Load Balancer")
	
	// Start metrics collection if enabled
	if lb.config.EnableMetrics {
		go lb.metricsCollectionLoop()
	}
	
	// Start cleanup loop
	go lb.cleanupLoop()
	
	lb.initialized = true
	log.Println("L4/L7 Load Balancer started successfully")
	return nil
}

// Stop stops the load balancer
func (lb *L4L7LoadBalancer) Stop() error {
	log.Println("Stopping L4/L7 Load Balancer")
	
	// Stop all listeners
	lb.listenersMutex.Lock()
	for _, listener := range lb.listeners {
		if err := lb.stopListener(listener); err != nil {
			log.Printf("Warning: Failed to stop listener %s: %v", listener.ID, err)
		}
	}
	lb.listenersMutex.Unlock()
	
	// Stop all health checkers
	lb.checkersMutex.Lock()
	for _, checker := range lb.healthCheckers {
		lb.stopHealthChecker(checker)
	}
	lb.checkersMutex.Unlock()
	
	lb.cancel()
	lb.initialized = false
	return nil
}

// CreateService creates a new load balancer service
func (lb *L4L7LoadBalancer) CreateService(ctx context.Context, service *LoadBalancerService) error {
	if service.ID == "" {
		service.ID = uuid.New().String()
	}
	
	lb.servicesMutex.Lock()
	defer lb.servicesMutex.Unlock()
	
	// Check if service already exists
	if _, exists := lb.services[service.ID]; exists {
		return fmt.Errorf("service %s already exists", service.ID)
	}
	
	// Validate service configuration
	if err := lb.validateServiceConfig(service); err != nil {
		return fmt.Errorf("service validation failed: %w", err)
	}
	
	// Set default values
	if service.Algorithm == "" {
		service.Algorithm = lb.config.DefaultAlgorithm
	}
	
	// Set timestamps
	service.CreatedAt = time.Now()
	service.UpdatedAt = time.Now()
	service.Status = ServiceStatusActive
	
	// Initialize statistics
	service.Statistics = ServiceStatistics{
		LastUpdated: time.Now(),
	}
	
	// Store service
	lb.services[service.ID] = service
	
	// Create listener for the service
	if err := lb.createServiceListener(service); err != nil {
		delete(lb.services, service.ID)
		return fmt.Errorf("failed to create listener: %w", err)
	}
	
	// Start health checking for backends
	for _, backendID := range service.Backends {
		if err := lb.startHealthCheckingForBackend(service.ID, backendID); err != nil {
			log.Printf("Warning: Failed to start health checking for backend %s: %v", backendID, err)
		}
	}
	
	log.Printf("Created load balancer service: %s (port %d)", service.Name, service.ListenPort)
	return nil
}

// validateServiceConfig validates service configuration
func (lb *L4L7LoadBalancer) validateServiceConfig(service *LoadBalancerService) error {
	if service.Name == "" {
		return fmt.Errorf("service name cannot be empty")
	}
	
	if service.ListenPort <= 0 || service.ListenPort > 65535 {
		return fmt.Errorf("invalid listen port: %d", service.ListenPort)
	}
	
	if len(service.Backends) == 0 {
		return fmt.Errorf("service must have at least one backend")
	}
	
	// Validate algorithm
	validAlgorithms := []Algorithm{
		AlgorithmRoundRobin, AlgorithmWeightedRoundRobin,
		AlgorithmLeastConnections, AlgorithmWeightedLeastConnections,
		AlgorithmIPHash, AlgorithmURLHash, AlgorithmConsistentHash,
		AlgorithmRandom, AlgorithmWeightedRandom,
	}
	
	algorithmValid := false
	for _, alg := range validAlgorithms {
		if service.Algorithm == alg {
			algorithmValid = true
			break
		}
	}
	
	if !algorithmValid {
		return fmt.Errorf("invalid algorithm: %s", service.Algorithm)
	}
	
	return nil
}

// createServiceListener creates a listener for a service
func (lb *L4L7LoadBalancer) createServiceListener(service *LoadBalancerService) error {
	listenerID := fmt.Sprintf("listener-%s", service.ID)
	address := fmt.Sprintf("%s:%d", lb.config.ListenAddress, service.ListenPort)
	
	listener := &Listener{
		ID:        listenerID,
		ServiceID: service.ID,
		Address:   address,
		Port:      service.ListenPort,
		Protocol:  service.Protocol,
		Statistics: ListenerStatistics{
			LastUpdated: time.Now(),
		},
		Status:    ListenerStatusActive,
		CreatedAt: time.Now(),
	}
	
	// Setup TLS if needed
	if service.Protocol == ProtocolHTTPS || service.SSLConfig != nil {
		tlsConfig, err := lb.createTLSConfig(service.SSLConfig)
		if err != nil {
			return fmt.Errorf("failed to create TLS config: %w", err)
		}
		listener.TLSConfig = tlsConfig
	}
	
	// Create network listener
	var netListener net.Listener
	var err error
	
	if listener.TLSConfig != nil {
		netListener, err = tls.Listen("tcp", address, listener.TLSConfig)
	} else {
		netListener, err = net.Listen("tcp", address)
	}
	
	if err != nil {
		return fmt.Errorf("failed to create network listener: %w", err)
	}
	
	listener.listener = netListener
	
	// Create HTTP server for HTTP/HTTPS services
	if service.Type == ServiceTypeHTTP || service.Type == ServiceTypeHTTPS {
		handler := lb.createHTTPHandler(service)
		
		httpServer := &http.Server{
			Handler:      handler,
			ReadTimeout:  lb.config.ConnectionTimeout,
			WriteTimeout: lb.config.ConnectionTimeout,
			IdleTimeout:  lb.config.IdleTimeout,
			MaxHeaderBytes: 1 << 20, // 1MB
		}
		
		listener.httpServer = httpServer
		
		// Start HTTP server
		go func() {
			if err := httpServer.Serve(netListener); err != nil && err != http.ErrServerClosed {
				log.Printf("HTTP server error for service %s: %v", service.Name, err)
				listener.Status = ListenerStatusError
			}
		}()
	} else {
		// Handle TCP connections directly
		go lb.handleTCPListener(listener, service)
	}
	
	// Store listener
	lb.listenersMutex.Lock()
	lb.listeners[listenerID] = listener
	lb.listenersMutex.Unlock()
	
	return nil
}

// createTLSConfig creates TLS configuration from SSL config
func (lb *L4L7LoadBalancer) createTLSConfig(sslConfig *SSLConfig) (*tls.Config, error) {
	if sslConfig == nil {
		// Use default SSL config from load balancer config
		if !lb.config.EnableSSL {
			return nil, fmt.Errorf("SSL not enabled")
		}
		
		cert, err := tls.LoadX509KeyPair(lb.config.SSLCertPath, lb.config.SSLKeyPath)
		if err != nil {
			return nil, fmt.Errorf("failed to load certificate: %w", err)
		}
		
		return &tls.Config{
			Certificates: []tls.Certificate{cert},
			MinVersion:   lb.config.SSLMinVersion,
			MaxVersion:   lb.config.SSLMaxVersion,
			CipherSuites: lb.config.SSLCipherSuites,
		}, nil
	}
	
	// Use service-specific SSL config
	cert, err := tls.LoadX509KeyPair(sslConfig.CertPath, sslConfig.KeyPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load certificate: %w", err)
	}
	
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   sslConfig.MinVersion,
		MaxVersion:   sslConfig.MaxVersion,
		CipherSuites: sslConfig.CipherSuites,
	}
	
	return tlsConfig, nil
}

// createHTTPHandler creates HTTP handler for a service
func (lb *L4L7LoadBalancer) createHTTPHandler(service *LoadBalancerService) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		startTime := time.Now()
		
		// Update statistics
		atomic.AddInt64(&lb.stats.TotalRequests, 1)
		atomic.AddInt64(&service.Statistics.TotalRequests, 1)
		atomic.AddInt64(&lb.stats.ActiveConnections, 1)
		atomic.AddInt64(&service.Statistics.ActiveConnections, 1)
		
		defer func() {
			atomic.AddInt64(&lb.stats.ActiveConnections, -1)
			atomic.AddInt64(&service.Statistics.ActiveConnections, -1)
			
			// Update response time
			duration := time.Since(startTime)
			lb.updateResponseTime(duration)
		}()
		
		// Apply load balancing rules if configured
		if len(service.Rules) > 0 {
			if handled := lb.applyLoadBalancingRules(service, w, r); handled {
				return
			}
		}
		
		// Select backend using configured algorithm
		backend, err := lb.selectBackend(service, r)
		if err != nil {
			http.Error(w, "Service Unavailable", http.StatusServiceUnavailable)
			atomic.AddInt64(&lb.stats.TotalErrors, 1)
			atomic.AddInt64(&service.Statistics.TotalErrors, 1)
			return
		}
		
		// Create reverse proxy to backend
		target, err := url.Parse(fmt.Sprintf("http://%s:%d", backend.Address, backend.Port))
		if err != nil {
			http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			atomic.AddInt64(&lb.stats.TotalErrors, 1)
			atomic.AddInt64(&service.Statistics.TotalErrors, 1)
			return
		}
		
		proxy := httputil.NewSingleHostReverseProxy(target)
		
		// Customize proxy behavior
		originalDirector := proxy.Director
		proxy.Director = func(req *http.Request) {
			originalDirector(req)
			
			// Add custom headers
			req.Header.Set("X-Forwarded-For", lb.getClientIP(r))
			req.Header.Set("X-Forwarded-Proto", lb.getScheme(r))
			req.Header.Set("X-Load-Balancer", "NovaCron-LB")
		}
		
		// Handle response modification
		proxy.ModifyResponse = func(resp *http.Response) error {
			// Update backend statistics
			atomic.AddInt64(&backend.Statistics.TotalRequests, 1)
			
			if resp.StatusCode >= 400 {
				atomic.AddInt64(&backend.Statistics.TotalErrors, 1)
			}
			
			return nil
		}
		
		// Handle errors
		proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("Proxy error for backend %s: %v", backend.Name, err)
			
			// Mark backend as potentially unhealthy
			lb.recordBackendError(backend)
			
			// Return error response
			http.Error(w, "Bad Gateway", http.StatusBadGateway)
			atomic.AddInt64(&lb.stats.TotalErrors, 1)
			atomic.AddInt64(&service.Statistics.TotalErrors, 1)
			atomic.AddInt64(&backend.Statistics.TotalErrors, 1)
		}
		
		// Update backend connection count
		atomic.AddInt64(&backend.CurrentConns, 1)
		atomic.AddInt64(&backend.Statistics.ActiveConnections, 1)
		
		defer func() {
			atomic.AddInt64(&backend.CurrentConns, -1)
			atomic.AddInt64(&backend.Statistics.ActiveConnections, -1)
		}()
		
		// Proxy the request
		proxy.ServeHTTP(w, r)
	})
}

// selectBackend selects a backend using the configured algorithm
func (lb *L4L7LoadBalancer) selectBackend(service *LoadBalancerService, r *http.Request) (*Backend, error) {
	// Get healthy backends
	healthyBackends := lb.getHealthyBackends(service.Backends)
	if len(healthyBackends) == 0 {
		return nil, fmt.Errorf("no healthy backends available")
	}
	
	switch service.Algorithm {
	case AlgorithmRoundRobin:
		return lb.selectRoundRobin(healthyBackends), nil
	case AlgorithmWeightedRoundRobin:
		return lb.selectWeightedRoundRobin(healthyBackends), nil
	case AlgorithmLeastConnections:
		return lb.selectLeastConnections(healthyBackends), nil
	case AlgorithmWeightedLeastConnections:
		return lb.selectWeightedLeastConnections(healthyBackends), nil
	case AlgorithmIPHash:
		return lb.selectIPHash(healthyBackends, r), nil
	case AlgorithmURLHash:
		return lb.selectURLHash(healthyBackends, r), nil
	case AlgorithmConsistentHash:
		return lb.selectConsistentHash(healthyBackends, r), nil
	case AlgorithmRandom:
		return lb.selectRandom(healthyBackends), nil
	case AlgorithmWeightedRandom:
		return lb.selectWeightedRandom(healthyBackends), nil
	default:
		return lb.selectRoundRobin(healthyBackends), nil
	}
}

// Backend selection algorithms implementation

func (lb *L4L7LoadBalancer) selectRoundRobin(backends []*Backend) *Backend {
	// Simple round-robin implementation
	// In production, this would use atomic counters per service
	return backends[int(time.Now().UnixNano())%len(backends)]
}

func (lb *L4L7LoadBalancer) selectWeightedRoundRobin(backends []*Backend) *Backend {
	totalWeight := 0
	for _, backend := range backends {
		totalWeight += backend.Weight
	}
	
	if totalWeight == 0 {
		return backends[0]
	}
	
	// Use time-based selection for simplicity
	// In production, this would use proper weighted round-robin
	selection := int(time.Now().UnixNano()) % totalWeight
	
	currentWeight := 0
	for _, backend := range backends {
		currentWeight += backend.Weight
		if selection < currentWeight {
			return backend
		}
	}
	
	return backends[0]
}

func (lb *L4L7LoadBalancer) selectLeastConnections(backends []*Backend) *Backend {
	leastConns := int64(^uint64(0) >> 1) // Max int64
	var selected *Backend
	
	for _, backend := range backends {
		conns := atomic.LoadInt64(&backend.CurrentConns)
		if conns < leastConns {
			leastConns = conns
			selected = backend
		}
	}
	
	if selected == nil {
		return backends[0]
	}
	
	return selected
}

func (lb *L4L7LoadBalancer) selectWeightedLeastConnections(backends []*Backend) *Backend {
	bestRatio := float64(^uint64(0) >> 1) // Max float64
	var selected *Backend
	
	for _, backend := range backends {
		if backend.Weight == 0 {
			continue
		}
		
		conns := atomic.LoadInt64(&backend.CurrentConns)
		ratio := float64(conns) / float64(backend.Weight)
		
		if ratio < bestRatio {
			bestRatio = ratio
			selected = backend
		}
	}
	
	if selected == nil {
		return backends[0]
	}
	
	return selected
}

func (lb *L4L7LoadBalancer) selectIPHash(backends []*Backend, r *http.Request) *Backend {
	clientIP := lb.getClientIP(r)
	hash := lb.hash(clientIP)
	return backends[int(hash)%len(backends)]
}

func (lb *L4L7LoadBalancer) selectURLHash(backends []*Backend, r *http.Request) *Backend {
	url := r.URL.Path
	if r.URL.RawQuery != "" {
		url += "?" + r.URL.RawQuery
	}
	hash := lb.hash(url)
	return backends[int(hash)%len(backends)]
}

func (lb *L4L7LoadBalancer) selectConsistentHash(backends []*Backend, r *http.Request) *Backend {
	// Simplified consistent hashing
	clientIP := lb.getClientIP(r)
	hash := lb.hash(clientIP)
	
	// Sort backends by name for consistency
	sortedBackends := make([]*Backend, len(backends))
	copy(sortedBackends, backends)
	sort.Slice(sortedBackends, func(i, j int) bool {
		return sortedBackends[i].Name < sortedBackends[j].Name
	})
	
	return sortedBackends[int(hash)%len(sortedBackends)]
}

func (lb *L4L7LoadBalancer) selectRandom(backends []*Backend) *Backend {
	return backends[int(time.Now().UnixNano())%len(backends)]
}

func (lb *L4L7LoadBalancer) selectWeightedRandom(backends []*Backend) *Backend {
	return lb.selectWeightedRoundRobin(backends) // Same logic for simplicity
}

// Helper functions

func (lb *L4L7LoadBalancer) hash(s string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(s))
	return h.Sum32()
}

func (lb *L4L7LoadBalancer) getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header first
	xff := r.Header.Get("X-Forwarded-For")
	if xff != "" {
		ips := strings.Split(xff, ",")
		return strings.TrimSpace(ips[0])
	}
	
	// Check X-Real-IP header
	xri := r.Header.Get("X-Real-IP")
	if xri != "" {
		return xri
	}
	
	// Use remote address
	ip, _, _ := net.SplitHostPort(r.RemoteAddr)
	return ip
}

func (lb *L4L7LoadBalancer) getScheme(r *http.Request) string {
	if r.TLS != nil {
		return "https"
	}
	
	if scheme := r.Header.Get("X-Forwarded-Proto"); scheme != "" {
		return scheme
	}
	
	return "http"
}

func (lb *L4L7LoadBalancer) getHealthyBackends(backendIDs []string) []*Backend {
	var healthy []*Backend
	
	lb.backendsMutex.RLock()
	defer lb.backendsMutex.RUnlock()
	
	for _, backendID := range backendIDs {
		if backend, exists := lb.backends[backendID]; exists {
			if backend.Status == BackendStatusActive && backend.HealthStatus == HealthStatusHealthy {
				healthy = append(healthy, backend)
			}
		}
	}
	
	return healthy
}

func (lb *L4L7LoadBalancer) updateResponseTime(duration time.Duration) {
	// Simple moving average - in production would use more sophisticated metrics
	lb.statsMutex.Lock()
	defer lb.statsMutex.Unlock()
	
	if lb.stats.AverageResponseTime == 0 {
		lb.stats.AverageResponseTime = float64(duration.Milliseconds())
	} else {
		// Exponential moving average with alpha = 0.1
		lb.stats.AverageResponseTime = 0.9*lb.stats.AverageResponseTime + 0.1*float64(duration.Milliseconds())
	}
}

func (lb *L4L7LoadBalancer) recordBackendError(backend *Backend) {
	// Record error and potentially mark backend as unhealthy
	atomic.AddInt64(&backend.Statistics.TotalErrors, 1)
	
	// Simple error threshold - mark unhealthy if error rate > 50%
	totalRequests := atomic.LoadInt64(&backend.Statistics.TotalRequests)
	totalErrors := atomic.LoadInt64(&backend.Statistics.TotalErrors)
	
	if totalRequests > 10 && float64(totalErrors)/float64(totalRequests) > 0.5 {
		backend.HealthStatus = HealthStatusUnhealthy
		log.Printf("Marked backend %s as unhealthy due to high error rate", backend.Name)
	}
}

// Additional methods for TCP handling, health checking, etc.

func (lb *L4L7LoadBalancer) handleTCPListener(listener *Listener, service *LoadBalancerService) {
	for {
		conn, err := listener.listener.Accept()
		if err != nil {
			if lb.ctx.Err() != nil {
				return // Context cancelled
			}
			log.Printf("TCP listener error for service %s: %v", service.Name, err)
			continue
		}
		
		go lb.handleTCPConnection(conn, service)
	}
}

func (lb *L4L7LoadBalancer) handleTCPConnection(clientConn net.Conn, service *LoadBalancerService) {
	defer clientConn.Close()
	
	// Select backend
	healthyBackends := lb.getHealthyBackends(service.Backends)
	if len(healthyBackends) == 0 {
		return
	}
	
	backend := lb.selectRoundRobin(healthyBackends) // Simplified
	
	// Connect to backend
	backendAddr := fmt.Sprintf("%s:%d", backend.Address, backend.Port)
	backendConn, err := net.DialTimeout("tcp", backendAddr, lb.config.ConnectionTimeout)
	if err != nil {
		lb.recordBackendError(backend)
		return
	}
	defer backendConn.Close()
	
	// Update statistics
	atomic.AddInt64(&backend.CurrentConns, 1)
	atomic.AddInt64(&backend.Statistics.ActiveConnections, 1)
	
	defer func() {
		atomic.AddInt64(&backend.CurrentConns, -1)
		atomic.AddInt64(&backend.Statistics.ActiveConnections, -1)
	}()
	
	// Proxy data bidirectionally
	go lb.proxyData(clientConn, backendConn)
	lb.proxyData(backendConn, clientConn)
}

func (lb *L4L7LoadBalancer) proxyData(src, dst net.Conn) {
	// Simple data proxying
	buffer := make([]byte, 32*1024) // 32KB buffer
	
	for {
		n, err := src.Read(buffer)
		if err != nil {
			return
		}
		
		_, err = dst.Write(buffer[:n])
		if err != nil {
			return
		}
	}
}

// Load balancing rules implementation

func (lb *L4L7LoadBalancer) applyLoadBalancingRules(service *LoadBalancerService, w http.ResponseWriter, r *http.Request) bool {
	// Sort rules by priority (higher priority first)
	rules := make([]*LoadBalancingRule, len(service.Rules))
	copy(rules, service.Rules)
	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Priority > rules[j].Priority
	})
	
	for _, rule := range rules {
		if !rule.Enabled {
			continue
		}
		
		if lb.evaluateRuleConditions(rule.Conditions, r) {
			return lb.executeRuleActions(rule.Actions, w, r)
		}
	}
	
	return false // No rules matched
}

func (lb *L4L7LoadBalancer) evaluateRuleConditions(conditions []RuleCondition, r *http.Request) bool {
	// All conditions must match (AND logic)
	for _, condition := range conditions {
		if !lb.evaluateCondition(condition, r) {
			return false
		}
	}
	return true
}

func (lb *L4L7LoadBalancer) evaluateCondition(condition RuleCondition, r *http.Request) bool {
	var fieldValue string
	
	switch condition.Type {
	case ConditionTypeHost:
		fieldValue = r.Host
	case ConditionTypePath:
		fieldValue = r.URL.Path
	case ConditionTypeMethod:
		fieldValue = r.Method
	case ConditionTypeHeader:
		fieldValue = r.Header.Get(condition.Field)
	case ConditionTypeQuery:
		fieldValue = r.URL.Query().Get(condition.Field)
	case ConditionTypeSourceIP:
		fieldValue = lb.getClientIP(r)
	default:
		return false
	}
	
	if !condition.CaseSensitive {
		fieldValue = strings.ToLower(fieldValue)
		condition.Value = strings.ToLower(condition.Value)
	}
	
	switch condition.Operator {
	case OperatorEquals:
		return fieldValue == condition.Value
	case OperatorContains:
		return strings.Contains(fieldValue, condition.Value)
	case OperatorStartsWith:
		return strings.HasPrefix(fieldValue, condition.Value)
	case OperatorEndsWith:
		return strings.HasSuffix(fieldValue, condition.Value)
	case OperatorIn:
		values := strings.Split(condition.Value, ",")
		for _, v := range values {
			if strings.TrimSpace(v) == fieldValue {
				return true
			}
		}
		return false
	default:
		return false
	}
}

func (lb *L4L7LoadBalancer) executeRuleActions(actions []RuleAction, w http.ResponseWriter, r *http.Request) bool {
	for _, action := range actions {
		switch action.Type {
		case ActionTypeRedirect:
			http.Redirect(w, r, action.Value, http.StatusFound)
			return true
		case ActionTypeDeny:
			http.Error(w, "Access Denied", http.StatusForbidden)
			return true
		case ActionTypeAddHeader:
			for key, value := range action.Headers {
				w.Header().Set(key, value)
			}
		case ActionTypeRewrite:
			r.URL.Path = action.Value
		}
	}
	
	return false // Continue processing
}

// Health checking implementation

func (lb *L4L7LoadBalancer) startHealthCheckingForBackend(serviceID, backendID string) error {
	// Get service and backend
	service, exists := lb.services[serviceID]
	if !exists {
		return fmt.Errorf("service %s not found", serviceID)
	}
	
	backend, exists := lb.backends[backendID]
	if !exists {
		return fmt.Errorf("backend %s not found", backendID)
	}
	
	// Create health checker
	checkerID := fmt.Sprintf("checker-%s-%s", serviceID, backendID)
	checker := &HealthChecker{
		ID:        checkerID,
		ServiceID: serviceID,
		BackendID: backendID,
		Config:    service.HealthCheck,
		Status:    HealthStatusUnknown,
		Statistics: HealthCheckStatistics{
			LastUpdated: time.Now(),
		},
		stopCh: make(chan struct{}),
	}
	
	// Start health checking
	interval := service.HealthCheck.Interval
	if interval == 0 {
		interval = lb.config.HealthCheckInterval
	}
	
	checker.ticker = time.NewTicker(interval)
	
	go func() {
		for {
			select {
			case <-checker.stopCh:
				return
			case <-checker.ticker.C:
				lb.performHealthCheck(checker, backend)
			}
		}
	}()
	
	// Store health checker
	lb.checkersMutex.Lock()
	lb.healthCheckers[checkerID] = checker
	lb.checkersMutex.Unlock()
	
	return nil
}

func (lb *L4L7LoadBalancer) performHealthCheck(checker *HealthChecker, backend *Backend) {
	startTime := time.Now()
	checker.LastCheck = startTime
	
	var healthy bool
	var err error
	
	switch checker.Config.Type {
	case HealthCheckTypeHTTP:
		healthy, err = lb.performHTTPHealthCheck(checker, backend)
	case HealthCheckTypeHTTPS:
		healthy, err = lb.performHTTPSHealthCheck(checker, backend)
	case HealthCheckTypeTCP:
		healthy, err = lb.performTCPHealthCheck(checker, backend)
	default:
		healthy, err = lb.performTCPHealthCheck(checker, backend)
	}
	
	duration := time.Since(startTime)
	
	// Update statistics
	atomic.AddInt64(&checker.Statistics.TotalChecks, 1)
	if healthy {
		atomic.AddInt64(&checker.Statistics.PassedChecks, 1)
		checker.ConsecutivePasses++
		checker.ConsecutiveFails = 0
		checker.LastError = ""
	} else {
		atomic.AddInt64(&checker.Statistics.FailedChecks, 1)
		checker.ConsecutiveFails++
		checker.ConsecutivePasses = 0
		if err != nil {
			checker.LastError = err.Error()
		}
	}
	
	// Update average response time
	if checker.Statistics.AverageResponseTime == 0 {
		checker.Statistics.AverageResponseTime = float64(duration.Milliseconds())
	} else {
		checker.Statistics.AverageResponseTime = 0.9*checker.Statistics.AverageResponseTime + 0.1*float64(duration.Milliseconds())
	}
	
	// Determine health status based on consecutive results
	healthyThreshold := checker.Config.HealthyThreshold
	if healthyThreshold == 0 {
		healthyThreshold = 2
	}
	
	unhealthyThreshold := checker.Config.UnhealthyThreshold
	if unhealthyThreshold == 0 {
		unhealthyThreshold = 3
	}
	
	if checker.ConsecutivePasses >= healthyThreshold {
		checker.Status = HealthStatusHealthy
		backend.HealthStatus = HealthStatusHealthy
	} else if checker.ConsecutiveFails >= unhealthyThreshold {
		checker.Status = HealthStatusUnhealthy
		backend.HealthStatus = HealthStatusUnhealthy
	}
	
	backend.LastChecked = time.Now()
}

func (lb *L4L7LoadBalancer) performHTTPHealthCheck(checker *HealthChecker, backend *Backend) (bool, error) {
	url := fmt.Sprintf("http://%s:%d%s", backend.Address, backend.Port, checker.Config.Path)
	
	client := &http.Client{
		Timeout: checker.Config.Timeout,
	}
	
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return false, err
	}
	
	// Add custom headers
	for key, value := range checker.Config.Headers {
		req.Header.Set(key, value)
	}
	
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	
	// Check status code
	expectedStatus := checker.Config.ExpectedStatus
	if expectedStatus == 0 {
		expectedStatus = 200
	}
	
	return resp.StatusCode == expectedStatus, nil
}

func (lb *L4L7LoadBalancer) performHTTPSHealthCheck(checker *HealthChecker, backend *Backend) (bool, error) {
	// Similar to HTTP but with TLS
	url := fmt.Sprintf("https://%s:%d%s", backend.Address, backend.Port, checker.Config.Path)
	
	client := &http.Client{
		Timeout: checker.Config.Timeout,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				InsecureSkipVerify: true, // For self-signed certificates
			},
		},
	}
	
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return false, err
	}
	
	resp, err := client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	
	expectedStatus := checker.Config.ExpectedStatus
	if expectedStatus == 0 {
		expectedStatus = 200
	}
	
	return resp.StatusCode == expectedStatus, nil
}

func (lb *L4L7LoadBalancer) performTCPHealthCheck(checker *HealthChecker, backend *Backend) (bool, error) {
	address := fmt.Sprintf("%s:%d", backend.Address, backend.Port)
	
	conn, err := net.DialTimeout("tcp", address, checker.Config.Timeout)
	if err != nil {
		return false, err
	}
	conn.Close()
	
	return true, nil
}

func (lb *L4L7LoadBalancer) stopHealthChecker(checker *HealthChecker) {
	if checker.ticker != nil {
		checker.ticker.Stop()
	}
	close(checker.stopCh)
}

// Lifecycle management and monitoring

func (lb *L4L7LoadBalancer) stopListener(listener *Listener) error {
	if listener.httpServer != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		if err := listener.httpServer.Shutdown(ctx); err != nil {
			return err
		}
	}
	
	if listener.listener != nil {
		return listener.listener.Close()
	}
	
	return nil
}

func (lb *L4L7LoadBalancer) metricsCollectionLoop() {
	ticker := time.NewTicker(lb.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-lb.ctx.Done():
			return
		case <-ticker.C:
			lb.updateMetrics()
		}
	}
}

func (lb *L4L7LoadBalancer) updateMetrics() {
	lb.statsMutex.Lock()
	defer lb.statsMutex.Unlock()
	
	// Update load balancer statistics
	totalRequests := atomic.LoadInt64(&lb.stats.TotalRequests)
	totalErrors := atomic.LoadInt64(&lb.stats.TotalErrors)
	
	if totalRequests > 0 {
		lb.stats.ErrorRate = float64(totalErrors) / float64(totalRequests) * 100.0
	}
	
	// Calculate requests per second (simplified)
	// In production, this would use a sliding window
	lb.stats.RequestsPerSecond = float64(totalRequests) / time.Since(lb.stats.LastUpdated).Seconds()
	
	lb.stats.LastUpdated = time.Now()
}

func (lb *L4L7LoadBalancer) cleanupLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-lb.ctx.Done():
			return
		case <-ticker.C:
			lb.performCleanup()
		}
	}
}

func (lb *L4L7LoadBalancer) performCleanup() {
	// Cleanup inactive connections, expired sessions, etc.
	log.Println("Performing load balancer cleanup")
}

// Public API methods

// GetServices returns all load balancer services
func (lb *L4L7LoadBalancer) GetServices() []*LoadBalancerService {
	lb.servicesMutex.RLock()
	defer lb.servicesMutex.RUnlock()
	
	services := make([]*LoadBalancerService, 0, len(lb.services))
	for _, service := range lb.services {
		serviceCopy := *service
		services = append(services, &serviceCopy)
	}
	
	return services
}

// GetService returns a specific service
func (lb *L4L7LoadBalancer) GetService(serviceID string) (*LoadBalancerService, error) {
	lb.servicesMutex.RLock()
	defer lb.servicesMutex.RUnlock()
	
	service, exists := lb.services[serviceID]
	if !exists {
		return nil, fmt.Errorf("service %s not found", serviceID)
	}
	
	serviceCopy := *service
	return &serviceCopy, nil
}

// CreateBackend creates a new backend server
func (lb *L4L7LoadBalancer) CreateBackend(ctx context.Context, backend *Backend) error {
	if backend.ID == "" {
		backend.ID = uuid.New().String()
	}
	
	lb.backendsMutex.Lock()
	defer lb.backendsMutex.Unlock()
	
	// Check if backend already exists
	if _, exists := lb.backends[backend.ID]; exists {
		return fmt.Errorf("backend %s already exists", backend.ID)
	}
	
	// Set default values
	if backend.Weight == 0 {
		backend.Weight = 1
	}
	if backend.MaxConns == 0 {
		backend.MaxConns = 1000
	}
	
	// Initialize fields
	backend.Status = BackendStatusActive
	backend.HealthStatus = HealthStatusUnknown
	backend.CurrentConns = 0
	backend.Statistics = BackendStatistics{
		LastUpdated: time.Now(),
	}
	backend.CreatedAt = time.Now()
	backend.UpdatedAt = time.Now()
	
	// Store backend
	lb.backends[backend.ID] = backend
	
	log.Printf("Created backend: %s (%s:%d)", backend.Name, backend.Address, backend.Port)
	return nil
}

// GetBackends returns all backends
func (lb *L4L7LoadBalancer) GetBackends() []*Backend {
	lb.backendsMutex.RLock()
	defer lb.backendsMutex.RUnlock()
	
	backends := make([]*Backend, 0, len(lb.backends))
	for _, backend := range lb.backends {
		backendCopy := *backend
		backends = append(backends, &backendCopy)
	}
	
	return backends
}

// GetStatistics returns load balancer statistics
func (lb *L4L7LoadBalancer) GetStatistics() *LoadBalancerStatistics {
	lb.statsMutex.RLock()
	defer lb.statsMutex.RUnlock()
	
	statsCopy := *lb.stats
	return &statsCopy
}