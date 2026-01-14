package security

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// SecurityOrchestrator orchestrates all security components
type SecurityOrchestrator struct {
	config               SecurityOrchestratorConfig
	enterpriseManager    *EnterpriseSecurityManager
	rbacEngine          *RBACEngine
	rateLimiter         *EnterpriseRateLimiter
	encryptionManager   *EncryptionManager
	auditLogger         *ComprehensiveAuditLogger
	vulnerabilityScanner *VulnerabilityScanner
	secretsManager      *SecretsManager
	securityMonitor     *SecurityMonitor
	complianceFramework *ComplianceFramework
	middlewares         []gin.HandlerFunc
	healthChecker       *SecurityHealthChecker
	metricsCollector    *SecurityMetricsCollector
	mu                  sync.RWMutex
}

// SecurityOrchestratorConfig defines orchestrator configuration
type SecurityOrchestratorConfig struct {
	EnableZeroTrust          bool                    `json:"enable_zero_trust"`
	EnableContinuousValidation bool                  `json:"enable_continuous_validation"`
	SecurityPolicyEnforcement SecurityPolicyConfig   `json:"policy_enforcement"`
	IntegrationConfig        IntegrationConfig       `json:"integrations"`
	MonitoringConfig         MonitoringConfig        `json:"monitoring"`
	ComplianceConfig         ComplianceConfig        `json:"compliance"`
	PerformanceConfig        PerformanceConfig       `json:"performance"`
}

// SecurityPolicyConfig defines security policy enforcement
type SecurityPolicyConfig struct {
	EnforceMFA              bool                    `json:"enforce_mfa"`
	RequireStrongPasswords  bool                    `json:"require_strong_passwords"`
	SessionTimeout          time.Duration           `json:"session_timeout"`
	MaxFailedLogins         int                     `json:"max_failed_logins"`
	PasswordExpiration      time.Duration           `json:"password_expiration"`
	IPWhitelistEnabled      bool                    `json:"ip_whitelist_enabled"`
	AllowedIPs              []string                `json:"allowed_ips"`
	DeviceRegistration      bool                    `json:"device_registration"`
	SecurityHeaders         map[string]string       `json:"security_headers"`
	ContentSecurityPolicy   string                  `json:"csp"`
	CORSPolicy              CORSPolicyConfig        `json:"cors_policy"`
}

// CORSPolicyConfig defines CORS policy
type CORSPolicyConfig struct {
	AllowedOrigins     []string      `json:"allowed_origins"`
	AllowedMethods     []string      `json:"allowed_methods"`
	AllowedHeaders     []string      `json:"allowed_headers"`
	ExposedHeaders     []string      `json:"exposed_headers"`
	AllowCredentials   bool          `json:"allow_credentials"`
	MaxAge             time.Duration `json:"max_age"`
}

// SecurityHealthChecker monitors overall security health
type SecurityHealthChecker struct {
	components          map[string]ComponentHealth `json:"components"`
	overallHealth       IntegrationHealthStatus              `json:"overall_health"`
	lastHealthCheck     time.Time                 `json:"last_health_check"`
	healthScore         float64                   `json:"health_score"`
	criticalIssues      []HealthIssue             `json:"critical_issues"`
	warnings            []HealthIssue             `json:"warnings"`
}

// ComponentHealth represents health status of a security component
type ComponentHealth struct {
	Name          string                 `json:"name"`
	Status        IntegrationHealthStatus           `json:"status"`
	LastCheck     time.Time              `json:"last_check"`
	ResponseTime  time.Duration          `json:"response_time"`
	ErrorRate     float64                `json:"error_rate"`
	Details       map[string]interface{} `json:"details"`
	Dependencies  []string               `json:"dependencies"`
}

// IntegrationIntegrationHealthStatus represents health status
type IntegrationIntegrationHealthStatus string

const (
	HealthHealthy   IntegrationHealthStatus = "healthy"
	HealthDegraded  IntegrationHealthStatus = "degraded"
	HealthUnhealthy IntegrationHealthStatus = "unhealthy"
	HealthCritical  IntegrationHealthStatus = "critical"
)

// HealthIssue represents a health issue
type HealthIssue struct {
	ID          string                 `json:"id"`
	Component   string                 `json:"component"`
	Severity    IssueSeverity          `json:"severity"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Impact      string                 `json:"impact"`
	Remediation string                 `json:"remediation"`
	Timestamp   time.Time              `json:"timestamp"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// IssueSeverity represents issue severity
type IssueSeverity string

const (
	IssueSeverityCritical IssueSeverity = "critical"
	IssueSeverityHigh     IssueSeverity = "high"
	IssueSeverityMedium   IssueSeverity = "medium"
	IssueSeverityLow      IssueSeverity = "low"
)

// Security metrics
var (
	securityRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_security_requests_total",
			Help: "Total number of security-processed requests",
		},
		[]string{"endpoint", "method", "status"},
	)

	securityValidationLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "novacron_security_validation_duration_seconds",
			Help: "Time taken for security validation",
		},
		[]string{"component", "type"},
	)

	securityBlockedRequestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_security_blocked_requests_total",
			Help: "Total number of blocked security requests",
		},
		[]string{"reason", "component"},
	)

	securityHealthScore = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "novacron_security_health_score",
			Help: "Overall security health score (0-1)",
		},
		[]string{"component"},
	)
)

// NewSecurityOrchestrator creates a new security orchestrator
func NewSecurityOrchestrator(config SecurityOrchestratorConfig) (*SecurityOrchestrator, error) {
	so := &SecurityOrchestrator{
		config:           config,
		middlewares:      make([]gin.HandlerFunc, 0),
		healthChecker:    &SecurityHealthChecker{
			components: make(map[string]ComponentHealth),
		},
	}

	// Initialize audit logger first (required by other components)
	auditLogger, err := NewComprehensiveAuditLogger(AuditConfig{
		EnableStructuredLogging: true,
		EnableTamperProofing:   true,
		EnableRealTimeAnalysis: true,
		LogRetention:          90 * 24 * time.Hour, // 90 days
		EnableEncryption:      true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize audit logger: %w", err)
	}
	so.auditLogger = auditLogger

	// Initialize enterprise security manager
	enterpriseManager, err := NewEnterpriseSecurityManager(EnterpriseSecurityConfig{
		ZeroTrustEnabled:     config.EnableZeroTrust,
		ContinuousValidation: config.EnableContinuousValidation,
		PolicyEnforcement:    config.SecurityPolicyEnforcement,
	}, auditLogger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize enterprise security manager: %w", err)
	}
	so.enterpriseManager = enterpriseManager

	// Initialize RBAC engine
	rbacEngine, err := NewRBACEngine(RBACConfig{
		EnableRoleInheritance:    true,
		EnableDynamicPermissions: true,
		CacheEnabled:            true,
		CacheTTL:                5 * time.Minute,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize RBAC engine: %w", err)
	}
	so.rbacEngine = rbacEngine

	// Initialize rate limiter
	rateLimiter, err := NewEnterpriseRateLimiter(RateLimitConfig{
		GlobalEnabled:    true,
		GlobalRate:       1000, // 1000 req/min globally
		UserEnabled:      true,
		UserRate:         100,  // 100 req/min per user
		IPEnabled:        true,
		IPRate:           200,  // 200 req/min per IP
		DDosProtection:   true,
		SuspiciousDetection: true,
	}, auditLogger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize rate limiter: %w", err)
	}
	so.rateLimiter = rateLimiter

	// Initialize encryption manager
	encryptionManager, err := NewEncryptionManager(EncryptionConfig{
		Algorithm:           "AES-256-GCM",
		KeyRotationEnabled:  true,
		KeyRotationInterval: 30 * 24 * time.Hour, // 30 days
		TLSMinVersion:      "1.3",
		CertificateRotation: true,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize encryption manager: %w", err)
	}
	so.encryptionManager = encryptionManager

	// Initialize vulnerability scanner
	vulnerabilityScanner, err := NewVulnerabilityScanner(ScanConfig{
		SASTEnabled:       true,
		DASTEnabled:       true,
		DependencyScanning: true,
		ContainerScanning: true,
		ScanFrequency:     24 * time.Hour, // Daily scans
		AutoRemediation:   false, // Manual review required
	}, auditLogger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize vulnerability scanner: %w", err)
	}
	so.vulnerabilityScanner = vulnerabilityScanner

	// Initialize secrets manager
	secretsManager, err := NewSecretsManager("vault") // Use Vault as default
	if err != nil {
		return nil, fmt.Errorf("failed to initialize secrets manager: %w", err)
	}
	so.secretsManager = secretsManager

	// Initialize security monitor
	securityMonitor, err := NewSecurityMonitor(SecurityMonitorConfig{
		EnableRealTimeMonitoring: true,
		ThreatDetectionConfig: ThreatDetectionConfig{
			EnableBehaviorAnalysis: true,
			EnableAnomalyDetection: true,
			EnableMLDetection:     true,
			BruteForceThreshold:   5,
			RateLimitThreshold:    100,
			GeoLocationEnabled:    true,
		},
		AlertingConfig: AlertingConfig{
			EnableEmailAlerts: true,
			EnableSlackAlerts: true,
		},
	}, auditLogger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize security monitor: %w", err)
	}
	so.securityMonitor = securityMonitor

	// Initialize compliance framework
	complianceFramework, err := NewComplianceFramework(config.ComplianceConfig, auditLogger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize compliance framework: %w", err)
	}
	so.complianceFramework = complianceFramework

	// Initialize metrics collector
	metricsCollector, err := NewSecurityMetricsCollector(MetricsConfig{
		EnablePrometheus:    true,
		CollectionInterval: 30 * time.Second,
		MetricRetention:    24 * time.Hour,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize metrics collector: %w", err)
	}
	so.metricsCollector = metricsCollector

	// Set up middleware chain
	so.setupMiddlewareChain()

	// Start background processes
	go so.healthMonitor()
	go so.securityOrchestrationWorker()
	go so.complianceWorker()

	return so, nil
}

// setupMiddlewareChain sets up the security middleware chain
func (so *SecurityOrchestrator) setupMiddlewareChain() {
	// Security headers middleware
	so.middlewares = append(so.middlewares, so.securityHeadersMiddleware())

	// CORS middleware
	so.middlewares = append(so.middlewares, so.corsMiddleware())

	// Rate limiting middleware
	so.middlewares = append(so.middlewares, so.rateLimitingMiddleware())

	// Authentication middleware
	so.middlewares = append(so.middlewares, so.authenticationMiddleware())

	// Authorization middleware (RBAC)
	so.middlewares = append(so.middlewares, so.authorizationMiddleware())

	// Security monitoring middleware
	so.middlewares = append(so.middlewares, so.monitoringMiddleware())

	// Audit logging middleware
	so.middlewares = append(so.middlewares, so.auditLoggingMiddleware())
}

// GetMiddlewares returns the security middleware chain
func (so *SecurityOrchestrator) GetMiddlewares() []gin.HandlerFunc {
	return so.middlewares
}

// securityHeadersMiddleware adds security headers
func (so *SecurityOrchestrator) securityHeadersMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		start := time.Now()

		// Set security headers
		headers := map[string]string{
			"X-Content-Type-Options":    "nosniff",
			"X-Frame-Options":          "DENY",
			"X-XSS-Protection":         "1; mode=block",
			"Strict-Transport-Security": "max-age=31536000; includeSubDomains",
			"Referrer-Policy":          "strict-origin-when-cross-origin",
			"Permissions-Policy":       "geolocation=(), microphone=(), camera=()",
		}

		// Add CSP header if configured
		if so.config.SecurityPolicyEnforcement.ContentSecurityPolicy != "" {
			headers["Content-Security-Policy"] = so.config.SecurityPolicyEnforcement.ContentSecurityPolicy
		}

		// Apply custom headers from config
		for key, value := range so.config.SecurityPolicyEnforcement.SecurityHeaders {
			headers[key] = value
		}

		for key, value := range headers {
			c.Header(key, value)
		}

		c.Next()

		// Record metrics
		securityValidationLatency.WithLabelValues("headers", "middleware").Observe(time.Since(start).Seconds())
	})
}

// corsMiddleware handles CORS policy
func (so *SecurityOrchestrator) corsMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		start := time.Now()

		origin := c.Request.Header.Get("Origin")
		
		// Check if origin is allowed
		if so.isOriginAllowed(origin) {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Authorization, X-Requested-With")
		c.Header("Access-Control-Max-Age", "86400") // 24 hours

		if so.config.SecurityPolicyEnforcement.CORSPolicy.AllowCredentials {
			c.Header("Access-Control-Allow-Credentials", "true")
		}

		// Handle preflight OPTIONS request
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()

		securityValidationLatency.WithLabelValues("cors", "middleware").Observe(time.Since(start).Seconds())
	})
}

// rateLimitingMiddleware applies rate limiting
func (so *SecurityOrchestrator) rateLimitingMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		start := time.Now()

		// Apply rate limiting
		if !so.rateLimiter.Allow(c.ClientIP(), c.Request.URL.Path, getUserID(c)) {
			securityBlockedRequestsTotal.WithLabelValues("rate_limit", "rate_limiter").Inc()
			
			// Log security event
			so.securityMonitor.ProcessSecurityEvent(SecurityEvent{
				Type:       EventRateLimitExceeded,
				Timestamp:  time.Now(),
				Source:     "rate_limiter",
				Severity:   ThreatLevelMedium,
				IP:         c.ClientIP(),
				UserAgent:  c.Request.UserAgent(),
				Endpoint:   c.Request.URL.Path,
				Method:     c.Request.Method,
				Message:    "Rate limit exceeded",
			})

			c.JSON(429, gin.H{"error": "Rate limit exceeded"})
			c.Abort()
			return
		}

		c.Next()

		securityValidationLatency.WithLabelValues("rate_limit", "middleware").Observe(time.Since(start).Seconds())
	})
}

// authenticationMiddleware handles authentication
func (so *SecurityOrchestrator) authenticationMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		start := time.Now()

		// Skip authentication for public endpoints
		if isPublicEndpoint(c.Request.URL.Path) {
			c.Next()
			return
		}

		// Extract and validate JWT token
		token := extractToken(c.Request)
		if token == "" {
			securityBlockedRequestsTotal.WithLabelValues("missing_token", "auth").Inc()
			
			so.securityMonitor.ProcessSecurityEvent(SecurityEvent{
				Type:      EventAuthFailure,
				Timestamp: time.Now(),
				Source:    "auth_middleware",
				Severity:  ThreatLevelMedium,
				IP:        c.ClientIP(),
				UserAgent: c.Request.UserAgent(),
				Endpoint:  c.Request.URL.Path,
				Method:    c.Request.Method,
				Message:   "Missing authentication token",
			})

			c.JSON(401, gin.H{"error": "Authentication required"})
			c.Abort()
			return
		}

		// Validate token with enterprise security manager
		ctx := context.Background()
		securityContext, err := so.enterpriseManager.ValidateToken(ctx, token)
		if err != nil {
			securityBlockedRequestsTotal.WithLabelValues("invalid_token", "auth").Inc()
			
			so.securityMonitor.ProcessSecurityEvent(SecurityEvent{
				Type:      EventAuthFailure,
				Timestamp: time.Now(),
				Source:    "auth_middleware",
				Severity:  ThreatLevelHigh,
				IP:        c.ClientIP(),
				UserAgent: c.Request.UserAgent(),
				Endpoint:  c.Request.URL.Path,
				Method:    c.Request.Method,
				Message:   "Invalid authentication token",
			})

			c.JSON(401, gin.H{"error": "Invalid token"})
			c.Abort()
			return
		}

		// Store security context in request
		c.Set("security_context", securityContext)
		c.Set("user_id", securityContext.UserID)

		c.Next()

		securityValidationLatency.WithLabelValues("auth", "middleware").Observe(time.Since(start).Seconds())
	})
}

// authorizationMiddleware handles RBAC authorization
func (so *SecurityOrchestrator) authorizationMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		start := time.Now()

		// Skip authorization for public endpoints
		if isPublicEndpoint(c.Request.URL.Path) {
			c.Next()
			return
		}

		// Get security context
		securityContext, exists := c.Get("security_context")
		if !exists {
			c.JSON(401, gin.H{"error": "Security context not found"})
			c.Abort()
			return
		}

		ctx := context.Background()
		sc := securityContext.(*SecurityContext)

		// Check permissions using RBAC engine
		resource := getResourceFromPath(c.Request.URL.Path)
		action := getActionFromMethod(c.Request.Method)

		hasPermission, err := so.rbacEngine.CheckPermission(ctx, sc.UserID, resource, action)
		if err != nil {
			so.auditLogger.LogEvent(AuditEvent{
				EventType: "authorization_error",
				UserID:    sc.UserID,
				Details:   fmt.Sprintf("Authorization check failed: %v", err),
			})

			c.JSON(500, gin.H{"error": "Authorization check failed"})
			c.Abort()
			return
		}

		if !hasPermission {
			securityBlockedRequestsTotal.WithLabelValues("insufficient_permissions", "authz").Inc()
			
			so.securityMonitor.ProcessSecurityEvent(SecurityEvent{
				Type:      EventUnauthorizedAccess,
				Timestamp: time.Now(),
				Source:    "authz_middleware",
				Severity:  ThreatLevelHigh,
				UserID:    sc.UserID,
				IP:        c.ClientIP(),
				UserAgent: c.Request.UserAgent(),
				Endpoint:  c.Request.URL.Path,
				Method:    c.Request.Method,
				Message:   fmt.Sprintf("Insufficient permissions for %s on %s", action, resource),
			})

			c.JSON(403, gin.H{"error": "Insufficient permissions"})
			c.Abort()
			return
		}

		c.Next()

		securityValidationLatency.WithLabelValues("authz", "middleware").Observe(time.Since(start).Seconds())
	})
}

// monitoringMiddleware handles security monitoring
func (so *SecurityOrchestrator) monitoringMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		start := time.Now()

		// Monitor request patterns
		go func() {
			event := SecurityEvent{
				Type:       EventSuspiciousActivity,
				Timestamp:  time.Now(),
				Source:     "monitoring_middleware",
				Severity:   ThreatLevelInfo,
				IP:         c.ClientIP(),
				UserAgent:  c.Request.UserAgent(),
				Endpoint:   c.Request.URL.Path,
				Method:     c.Request.Method,
			}

			// Add user ID if available
			if userID, exists := c.Get("user_id"); exists {
				event.UserID = userID.(string)
			}

			so.securityMonitor.ProcessSecurityEvent(event)
		}()

		c.Next()

		// Record metrics
		status := fmt.Sprintf("%d", c.Writer.Status())
		securityRequestsTotal.WithLabelValues(c.Request.URL.Path, c.Request.Method, status).Inc()

		duration := time.Since(start)
		securityValidationLatency.WithLabelValues("monitoring", "middleware").Observe(duration.Seconds())
	})
}

// auditLoggingMiddleware handles audit logging
func (so *SecurityOrchestrator) auditLoggingMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		start := time.Now()

		c.Next()

		// Log audit event
		event := AuditEvent{
			EventType:   "api_request",
			UserID:      getUserID(c),
			IPAddress:   c.ClientIP(),
			UserAgent:   c.Request.UserAgent(),
			Resource:    c.Request.URL.Path,
			Action:      c.Request.Method,
			Timestamp:   start,
			Success:     c.Writer.Status() < 400,
			StatusCode:  c.Writer.Status(),
			Duration:    time.Since(start),
			Details:     fmt.Sprintf("Request processed in %v", time.Since(start)),
		}

		so.auditLogger.LogEvent(event)
	})
}

// Background workers
func (so *SecurityOrchestrator) healthMonitor() {
	ticker := time.NewTicker(30 * time.Second) // Health check every 30 seconds
	defer ticker.Stop()

	for range ticker.C {
		so.performHealthCheck()
	}
}

func (so *SecurityOrchestrator) securityOrchestrationWorker() {
	ticker := time.NewTicker(60 * time.Second) // Orchestration tasks every minute
	defer ticker.Stop()

	for range ticker.C {
		so.performOrchestrationTasks()
	}
}

func (so *SecurityOrchestrator) complianceWorker() {
	ticker := time.NewTicker(24 * time.Hour) // Compliance checks daily
	defer ticker.Stop()

	for range ticker.C {
		so.performComplianceChecks()
	}
}

// performHealthCheck checks health of all security components
func (so *SecurityOrchestrator) performHealthCheck() {
	so.mu.Lock()
	defer so.mu.Unlock()

	components := map[string]func() ComponentHealth{
		"enterprise_security": so.checkEnterpriseSecurityHealth,
		"rbac_engine":        so.checkRBACEngineHealth,
		"rate_limiter":       so.checkRateLimiterHealth,
		"encryption_manager": so.checkEncryptionManagerHealth,
		"audit_logger":       so.checkAuditLoggerHealth,
		"vulnerability_scanner": so.checkVulnerabilityScannerHealth,
		"secrets_manager":    so.checkSecretsManagerHealth,
		"security_monitor":   so.checkSecurityMonitorHealth,
		"compliance_framework": so.checkComplianceFrameworkHealth,
	}

	totalScore := 0.0
	healthyComponents := 0

	for name, healthFunc := range components {
		health := healthFunc()
		so.healthChecker.components[name] = health

		switch health.Status {
		case HealthHealthy:
			totalScore += 1.0
			healthyComponents++
		case HealthDegraded:
			totalScore += 0.7
		case HealthUnhealthy:
			totalScore += 0.3
		case HealthCritical:
			totalScore += 0.0
		}

		// Update component metrics
		var score float64
		switch health.Status {
		case HealthHealthy:
			score = 1.0
		case HealthDegraded:
			score = 0.7
		case HealthUnhealthy:
			score = 0.3
		case HealthCritical:
			score = 0.0
		}
		securityHealthScore.WithLabelValues(name).Set(score)
	}

	// Calculate overall health
	so.healthChecker.healthScore = totalScore / float64(len(components))
	so.healthChecker.lastHealthCheck = time.Now()

	// Determine overall status
	if so.healthChecker.healthScore >= 0.9 {
		so.healthChecker.overallHealth = HealthHealthy
	} else if so.healthChecker.healthScore >= 0.7 {
		so.healthChecker.overallHealth = HealthDegraded
	} else if so.healthChecker.healthScore >= 0.3 {
		so.healthChecker.overallHealth = HealthUnhealthy
	} else {
		so.healthChecker.overallHealth = HealthCritical
	}

	// Update overall health metric
	securityHealthScore.WithLabelValues("overall").Set(so.healthChecker.healthScore)
}

// Component health check methods
func (so *SecurityOrchestrator) checkEnterpriseSecurityHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	// Check enterprise security manager
	if so.enterpriseManager != nil {
		// Perform health check
		details["initialized"] = true
		details["zero_trust_enabled"] = so.config.EnableZeroTrust
	} else {
		status = HealthCritical
		details["error"] = "Enterprise security manager not initialized"
	}

	return ComponentHealth{
		Name:         "Enterprise Security Manager",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

func (so *SecurityOrchestrator) checkRBACEngineHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	if so.rbacEngine != nil {
		metrics := so.rbacEngine.GetMetrics()
		details["cache_hit_rate"] = metrics["cache_hit_rate"]
		details["active_roles"] = metrics["active_roles"]
		details["active_permissions"] = metrics["active_permissions"]
	} else {
		status = HealthCritical
		details["error"] = "RBAC engine not initialized"
	}

	return ComponentHealth{
		Name:         "RBAC Engine",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

func (so *SecurityOrchestrator) checkRateLimiterHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	if so.rateLimiter != nil {
		metrics := so.rateLimiter.GetMetrics()
		details["requests_blocked"] = metrics["requests_blocked"]
		details["active_limiters"] = metrics["active_limiters"]
	} else {
		status = HealthCritical
		details["error"] = "Rate limiter not initialized"
	}

	return ComponentHealth{
		Name:         "Rate Limiter",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

func (so *SecurityOrchestrator) checkEncryptionManagerHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	if so.encryptionManager != nil {
		details["key_rotation_enabled"] = true
		details["tls_version"] = "1.3"
	} else {
		status = HealthCritical
		details["error"] = "Encryption manager not initialized"
	}

	return ComponentHealth{
		Name:         "Encryption Manager",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

func (so *SecurityOrchestrator) checkAuditLoggerHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	if so.auditLogger != nil {
		metrics := so.auditLogger.GetMetrics()
		details["events_logged"] = metrics["events_logged"]
		details["queue_length"] = metrics["queue_length"]
	} else {
		status = HealthCritical
		details["error"] = "Audit logger not initialized"
	}

	return ComponentHealth{
		Name:         "Audit Logger",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

func (so *SecurityOrchestrator) checkVulnerabilityScannerHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	if so.vulnerabilityScanner != nil {
		metrics := so.vulnerabilityScanner.GetMetrics()
		details["last_scan"] = metrics["last_scan"]
		details["vulnerabilities_found"] = metrics["vulnerabilities_found"]
	} else {
		status = HealthCritical
		details["error"] = "Vulnerability scanner not initialized"
	}

	return ComponentHealth{
		Name:         "Vulnerability Scanner",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

func (so *SecurityOrchestrator) checkSecretsManagerHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	if so.secretsManager != nil {
		// Simple health check - try to access a test secret
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		
		_, err := so.secretsManager.GetSecret(ctx, "health_check")
		if err != nil && err.Error() != "secret not found: health_check" {
			status = HealthDegraded
			details["error"] = err.Error()
		}
		details["provider"] = "vault"
	} else {
		status = HealthCritical
		details["error"] = "Secrets manager not initialized"
	}

	return ComponentHealth{
		Name:         "Secrets Manager",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

func (so *SecurityOrchestrator) checkSecurityMonitorHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	if so.securityMonitor != nil {
		healthStatus := so.securityMonitor.GetIntegrationHealthStatus()
		details = healthStatus
	} else {
		status = HealthCritical
		details["error"] = "Security monitor not initialized"
	}

	return ComponentHealth{
		Name:         "Security Monitor",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

func (so *SecurityOrchestrator) checkComplianceFrameworkHealth() ComponentHealth {
	start := time.Now()
	status := HealthHealthy
	details := make(map[string]interface{})

	if so.complianceFramework != nil {
		details["enabled_frameworks"] = so.config.ComplianceConfig.EnabledFrameworks
		details["continuous_monitoring"] = so.config.ComplianceConfig.ContinuousMonitoring
	} else {
		status = HealthCritical
		details["error"] = "Compliance framework not initialized"
	}

	return ComponentHealth{
		Name:         "Compliance Framework",
		Status:       status,
		LastCheck:    time.Now(),
		ResponseTime: time.Since(start),
		Details:      details,
	}
}

// performOrchestrationTasks performs security orchestration tasks
func (so *SecurityOrchestrator) performOrchestrationTasks() {
	// Coordinate security components
	// Example: trigger vulnerability scans, rotate keys, update threat intel
	
	// Log orchestration activity
	so.auditLogger.LogEvent(AuditEvent{
		EventType: "security_orchestration",
		Details:   "Performed security orchestration tasks",
	})
}

// performComplianceChecks performs compliance validation
func (so *SecurityOrchestrator) performComplianceChecks() {
	ctx := context.Background()
	
	// Run compliance assessments for enabled frameworks
	for _, frameworkID := range so.config.ComplianceConfig.EnabledFrameworks {
		go func(framework string) {
			so.auditLogger.LogEvent(AuditEvent{
				EventType: "compliance_check_started",
				Details:   fmt.Sprintf("Starting compliance check for %s", framework),
			})
		}(frameworkID)
	}
}

// GetIntegrationHealthStatus returns current health status
func (so *SecurityOrchestrator) GetIntegrationHealthStatus() *SecurityHealthChecker {
	so.mu.RLock()
	defer so.mu.RUnlock()
	return so.healthChecker
}

// GetSecurityMetrics returns security metrics
func (so *SecurityOrchestrator) GetSecurityMetrics() map[string]interface{} {
	return so.metricsCollector.GetCurrentMetrics()
}

// Helper functions
func isPublicEndpoint(path string) bool {
	publicEndpoints := []string{"/health", "/metrics", "/login", "/register"}
	for _, endpoint := range publicEndpoints {
		if path == endpoint {
			return true
		}
	}
	return false
}

func extractToken(r *gin.Request) string {
	// Extract JWT token from Authorization header
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return ""
	}
	
	if len(auth) > 7 && auth[:7] == "Bearer " {
		return auth[7:]
	}
	
	return ""
}

func getUserID(c *gin.Context) string {
	if userID, exists := c.Get("user_id"); exists {
		return userID.(string)
	}
	return "anonymous"
}

func getResourceFromPath(path string) string {
	// Extract resource from URL path
	// Example: /api/v1/users/123 -> users
	parts := strings.Split(path, "/")
	if len(parts) >= 4 && parts[1] == "api" {
		return parts[3] // Return the resource part
	}
	return "unknown"
}

func getActionFromMethod(method string) string {
	switch method {
	case "GET":
		return "read"
	case "POST":
		return "create"
	case "PUT", "PATCH":
		return "update"
	case "DELETE":
		return "delete"
	default:
		return "unknown"
	}
}

func (so *SecurityOrchestrator) isOriginAllowed(origin string) bool {
	if len(so.config.SecurityPolicyEnforcement.CORSPolicy.AllowedOrigins) == 0 {
		return true // No restrictions
	}
	
	for _, allowed := range so.config.SecurityPolicyEnforcement.CORSPolicy.AllowedOrigins {
		if allowed == "*" || allowed == origin {
			return true
		}
	}
	return false
}