package security

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// DefaultSecurityConfig returns default security configuration for NovaCron
func DefaultSecurityConfig() *SecurityOrchestratorConfig {
	return &SecurityOrchestratorConfig{
		EnableZeroTrust:            true,
		EnableContinuousValidation: true,
		SecurityPolicyEnforcement: SecurityPolicyConfig{
			EnforceMFA:             true,
			RequireStrongPasswords: true,
			SessionTimeout:         30 * time.Minute,
			MaxFailedLogins:        5,
			PasswordExpiration:     90 * 24 * time.Hour, // 90 days
			IPWhitelistEnabled:     false,
			AllowedIPs:            []string{},
			DeviceRegistration:    true,
			SecurityHeaders: map[string]string{
				"X-Content-Type-Options":    "nosniff",
				"X-Frame-Options":          "DENY",
				"X-XSS-Protection":         "1; mode=block",
				"Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
				"Referrer-Policy":          "strict-origin-when-cross-origin",
				"Permissions-Policy":       "geolocation=(), microphone=(), camera=(), payment=(), usb=(), interest-cohort=()",
			},
			ContentSecurityPolicy: "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' ws: wss:; frame-ancestors 'none'; form-action 'self'; base-uri 'self';",
			CORSPolicy: CORSPolicyConfig{
				AllowedOrigins:   []string{"https://app.novacron.com", "https://admin.novacron.com"},
				AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"},
				AllowedHeaders:   []string{"Origin", "Content-Type", "Authorization", "X-Requested-With", "X-API-Key"},
				ExposedHeaders:   []string{"X-Total-Count", "X-Page-Count"},
				AllowCredentials: true,
				MaxAge:           24 * time.Hour,
			},
		},
		IntegrationConfig: IntegrationConfig{
			EnableAWSIntegration:   true,
			EnableGCPIntegration:   false,
			EnableAzureIntegration: false,
			EnableSIEMIntegration:  true,
			SIEMConfig: SIEMConfig{
				Provider: "elasticsearch",
				Endpoint: "https://siem.novacron.com:9200",
				Index:    "novacron-security",
				Username: "elastic",
			},
		},
		MonitoringConfig: MonitoringConfig{
			EnablePrometheus:   true,
			PrometheusPort:     9090,
			EnableGrafana:      true,
			GrafanaPort:        3000,
			EnableAlerts:       true,
			AlertManager: AlertManagerConfig{
				Endpoint: "http://alertmanager:9093",
				Routes: []AlertRoute{
					{
						Receiver: "security-team",
						Matchers: []string{"severity=critical"},
					},
					{
						Receiver: "on-call",
						Matchers: []string{"severity=high"},
					},
				},
			},
			MetricsRetention: 30 * 24 * time.Hour, // 30 days
		},
		ComplianceConfig: ComplianceConfig{
			EnabledFrameworks: []string{"soc2", "iso27001", "gdpr"},
			AssessmentFrequency: map[string]time.Duration{
				"soc2":    90 * 24 * time.Hour, // Quarterly
				"iso27001": 365 * 24 * time.Hour, // Annually
				"gdpr":    30 * 24 * time.Hour, // Monthly
			},
			AutoRemediation:      false, // Manual review required
			ContinuousMonitoring: true,
			EvidenceRetention:    7 * 365 * 24 * time.Hour, // 7 years
			ReportingSchedule: map[string]time.Duration{
				"soc2":    7 * 24 * time.Hour,  // Weekly
				"iso27001": 30 * 24 * time.Hour, // Monthly
				"gdpr":    24 * time.Hour,      // Daily
			},
			NotificationConfig: ComplianceNotificationConfig{
				EnableEmail: true,
				EnableSlack: true,
				Recipients: map[string][]string{
					"soc2":    {"compliance@novacron.com", "ciso@novacron.com"},
					"iso27001": {"compliance@novacron.com", "ciso@novacron.com"},
					"gdpr":    {"privacy@novacron.com", "legal@novacron.com"},
				},
				AlertThresholds: map[string]float64{
					"compliance_score": 0.8, // Alert if below 80%
					"open_findings":    10,   // Alert if more than 10 open findings
				},
			},
		},
		PerformanceConfig: PerformanceConfig{
			MaxConcurrentRequests:    10000,
			RequestTimeout:           30 * time.Second,
			ValidationTimeout:        5 * time.Second,
			CacheSize:               100000,
			CacheTTL:                5 * time.Minute,
			EnableCompression:       true,
			EnableCaching:          true,
			EnableRequestBuffering: true,
			BufferSize:             1024 * 1024, // 1MB
		},
	}
}

// ProductionSecurityConfig returns production-grade security configuration
func ProductionSecurityConfig() *SecurityOrchestratorConfig {
	config := DefaultSecurityConfig()
	
	// Production-specific overrides
	config.SecurityPolicyEnforcement.SessionTimeout = 15 * time.Minute // Shorter sessions
	config.SecurityPolicyEnforcement.MaxFailedLogins = 3 // Stricter login policy
	config.SecurityPolicyEnforcement.IPWhitelistEnabled = true
	config.SecurityPolicyEnforcement.AllowedIPs = []string{
		"10.0.0.0/8",     // Internal network
		"172.16.0.0/12",  // Docker network
		"192.168.0.0/16", // Local network
	}
	
	// Stricter CSP for production
	config.SecurityPolicyEnforcement.ContentSecurityPolicy = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'self';"
	
	// Production CORS policy
	config.SecurityPolicyEnforcement.CORSPolicy.AllowedOrigins = []string{
		"https://app.novacron.com",
		"https://admin.novacron.com",
	}
	
	// Enhanced compliance for production
	config.ComplianceConfig.EnabledFrameworks = []string{"soc2", "iso27001", "gdpr", "hipaa", "pcidss"}
	config.ComplianceConfig.ContinuousMonitoring = true
	config.ComplianceConfig.AutoRemediation = false // Always require manual review in production
	
	return config
}

// DevelopmentSecurityConfig returns development-friendly security configuration
func DevelopmentSecurityConfig() *SecurityOrchestratorConfig {
	config := DefaultSecurityConfig()
	
	// Development-specific overrides
	config.EnableZeroTrust = false // More permissive for development
	config.SecurityPolicyEnforcement.EnforceMFA = false
	config.SecurityPolicyEnforcement.SessionTimeout = 8 * time.Hour // Longer sessions for development
	config.SecurityPolicyEnforcement.MaxFailedLogins = 10 // More forgiving
	config.SecurityPolicyEnforcement.IPWhitelistEnabled = false
	
	// More permissive CSP for development
	config.SecurityPolicyEnforcement.ContentSecurityPolicy = "default-src 'self' 'unsafe-inline' 'unsafe-eval'; img-src 'self' data: https:; connect-src 'self' ws: wss:;"
	
	// Development CORS policy
	config.SecurityPolicyEnforcement.CORSPolicy.AllowedOrigins = []string{
		"http://localhost:3000",
		"http://localhost:8080",
		"http://127.0.0.1:3000",
		"http://127.0.0.1:8080",
	}
	
	// Minimal compliance for development
	config.ComplianceConfig.EnabledFrameworks = []string{"soc2"}
	config.ComplianceConfig.ContinuousMonitoring = false
	config.ComplianceConfig.AssessmentFrequency = map[string]time.Duration{
		"soc2": 7 * 24 * time.Hour, // Weekly for development
	}
	
	return config
}

// LoadSecurityConfig loads security configuration from environment or file
func LoadSecurityConfig() (*SecurityOrchestratorConfig, error) {
	env := os.Getenv("NOVACRON_ENV")
	if env == "" {
		env = "development"
	}
	
	var config *SecurityOrchestratorConfig
	
	// Load base configuration based on environment
	switch env {
	case "production":
		config = ProductionSecurityConfig()
	case "staging":
		config = ProductionSecurityConfig() // Use production config for staging
	case "development":
		config = DevelopmentSecurityConfig()
	default:
		config = DefaultSecurityConfig()
	}
	
	// Override with configuration file if exists
	configFile := os.Getenv("NOVACRON_SECURITY_CONFIG")
	if configFile != "" {
		if err := loadConfigFromFile(config, configFile); err != nil {
			return nil, fmt.Errorf("failed to load config from file %s: %w", configFile, err)
		}
	}
	
	// Override with environment variables
	overrideWithEnvVars(config)
	
	// Validate configuration
	if err := validateSecurityConfig(config); err != nil {
		return nil, fmt.Errorf("invalid security configuration: %w", err)
	}
	
	return config, nil
}

// loadConfigFromFile loads configuration from JSON file
func loadConfigFromFile(config *SecurityOrchestratorConfig, filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}
	
	if err := json.Unmarshal(data, config); err != nil {
		return fmt.Errorf("failed to parse config file: %w", err)
	}
	
	return nil
}

// overrideWithEnvVars overrides configuration with environment variables
func overrideWithEnvVars(config *SecurityOrchestratorConfig) {
	// JWT Secret
	if jwtSecret := os.Getenv("NOVACRON_JWT_SECRET"); jwtSecret != "" {
		// JWT secret would be handled by secrets manager
	}
	
	// Database encryption key
	if encKey := os.Getenv("NOVACRON_ENCRYPTION_KEY"); encKey != "" {
		// Encryption key would be handled by secrets manager
	}
	
	// CORS origins
	if corsOrigins := os.Getenv("NOVACRON_CORS_ORIGINS"); corsOrigins != "" {
		// Parse comma-separated origins
		config.SecurityPolicyEnforcement.CORSPolicy.AllowedOrigins = []string{corsOrigins}
	}
	
	// Session timeout
	if sessionTimeout := os.Getenv("NOVACRON_SESSION_TIMEOUT"); sessionTimeout != "" {
		if duration, err := time.ParseDuration(sessionTimeout); err == nil {
			config.SecurityPolicyEnforcement.SessionTimeout = duration
		}
	}
	
	// MFA enforcement
	if mfa := os.Getenv("NOVACRON_ENFORCE_MFA"); mfa != "" {
		config.SecurityPolicyEnforcement.EnforceMFA = mfa == "true"
	}
	
	// Zero trust
	if zeroTrust := os.Getenv("NOVACRON_ZERO_TRUST"); zeroTrust != "" {
		config.EnableZeroTrust = zeroTrust == "true"
	}
}

// validateSecurityConfig validates the security configuration
func validateSecurityConfig(config *SecurityOrchestratorConfig) error {
	// Validate session timeout
	if config.SecurityPolicyEnforcement.SessionTimeout < time.Minute {
		return fmt.Errorf("session timeout must be at least 1 minute")
	}
	
	if config.SecurityPolicyEnforcement.SessionTimeout > 24*time.Hour {
		return fmt.Errorf("session timeout must not exceed 24 hours")
	}
	
	// Validate max failed logins
	if config.SecurityPolicyEnforcement.MaxFailedLogins < 1 {
		return fmt.Errorf("max failed logins must be at least 1")
	}
	
	if config.SecurityPolicyEnforcement.MaxFailedLogins > 100 {
		return fmt.Errorf("max failed logins must not exceed 100")
	}
	
	// Validate password expiration
	if config.SecurityPolicyEnforcement.PasswordExpiration < 24*time.Hour {
		return fmt.Errorf("password expiration must be at least 24 hours")
	}
	
	// Validate CORS origins
	if len(config.SecurityPolicyEnforcement.CORSPolicy.AllowedOrigins) == 0 {
		return fmt.Errorf("at least one CORS origin must be specified")
	}
	
	// Validate compliance frameworks
	validFrameworks := map[string]bool{
		"soc2": true, "iso27001": true, "gdpr": true, "hipaa": true, "pcidss": true,
	}
	
	for _, framework := range config.ComplianceConfig.EnabledFrameworks {
		if !validFrameworks[framework] {
			return fmt.Errorf("unsupported compliance framework: %s", framework)
		}
	}
	
	// Validate performance settings
	if config.PerformanceConfig.MaxConcurrentRequests < 1 {
		return fmt.Errorf("max concurrent requests must be at least 1")
	}
	
	if config.PerformanceConfig.RequestTimeout < time.Second {
		return fmt.Errorf("request timeout must be at least 1 second")
	}
	
	return nil
}

// Additional configuration types
type IntegrationConfig struct {
	EnableAWSIntegration   bool       `json:"enable_aws_integration"`
	EnableGCPIntegration   bool       `json:"enable_gcp_integration"`
	EnableAzureIntegration bool       `json:"enable_azure_integration"`
	EnableSIEMIntegration  bool       `json:"enable_siem_integration"`
	SIEMConfig             SIEMConfig `json:"siem_config"`
}

type SIEMConfig struct {
	Provider string `json:"provider"`
	Endpoint string `json:"endpoint"`
	Index    string `json:"index"`
	Username string `json:"username"`
	Password string `json:"password"`
}

type MonitoringConfig struct {
	EnablePrometheus bool                `json:"enable_prometheus"`
	PrometheusPort   int                 `json:"prometheus_port"`
	EnableGrafana    bool                `json:"enable_grafana"`
	GrafanaPort      int                 `json:"grafana_port"`
	EnableAlerts     bool                `json:"enable_alerts"`
	AlertManager     AlertManagerConfig  `json:"alert_manager"`
	MetricsRetention time.Duration       `json:"metrics_retention"`
}

type AlertManagerConfig struct {
	Endpoint string       `json:"endpoint"`
	Routes   []AlertRoute `json:"routes"`
}

type AlertRoute struct {
	Receiver string   `json:"receiver"`
	Matchers []string `json:"matchers"`
}

type PerformanceConfig struct {
	MaxConcurrentRequests    int           `json:"max_concurrent_requests"`
	RequestTimeout           time.Duration `json:"request_timeout"`
	ValidationTimeout        time.Duration `json:"validation_timeout"`
	CacheSize               int           `json:"cache_size"`
	CacheTTL                time.Duration `json:"cache_ttl"`
	EnableCompression       bool          `json:"enable_compression"`
	EnableCaching          bool          `json:"enable_caching"`
	EnableRequestBuffering bool          `json:"enable_request_buffering"`
	BufferSize             int           `json:"buffer_size"`
}

// SaveSecurityConfig saves configuration to file
func SaveSecurityConfig(config *SecurityOrchestratorConfig, filename string) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}
	
	if err := os.WriteFile(filename, data, 0600); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}
	
	return nil
}

// GetSecurityConfigTemplate returns a configuration template
func GetSecurityConfigTemplate() string {
	config := DefaultSecurityConfig()
	data, _ := json.MarshalIndent(config, "", "  ")
	return string(data)
}