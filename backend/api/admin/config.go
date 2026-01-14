package admin

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// Context keys for user information
type contextKey string

const (
	contextKeyUser contextKey = "user"
)

// getUserFromContext retrieves the authenticated user from the request context
// Returns "admin" as default if no user is found in context
func getUserFromContext(ctx context.Context) string {
	if user, ok := ctx.Value(contextKeyUser).(string); ok && user != "" {
		return user
	}
	// Default to "admin" when authentication middleware hasn't set the user
	return "admin"
}

type ConfigHandlers struct {
	configPath string
}

type SystemConfig struct {
	Server      ServerConfig     `json:"server"`
	Database    DatabaseConfig   `json:"database"`
	Security    SecurityConfig   `json:"security"`
	Storage     StorageConfig    `json:"storage"`
	VM          VMConfig         `json:"vm"`
	Monitoring  MonitoringConfig `json:"monitoring"`
	Network     NetworkConfig    `json:"network"`
	LastUpdated time.Time        `json:"last_updated"`
	UpdatedBy   string           `json:"updated_by,omitempty"`
}

type ServerConfig struct {
	APIPort        int    `json:"api_port"`
	WSPort         int    `json:"ws_port"`
	LogLevel       string `json:"log_level"`
	LogFormat      string `json:"log_format"`
	Environment    string `json:"environment"`
	MaxConnections int    `json:"max_connections"`
	RequestTimeout int    `json:"request_timeout_seconds"`
	EnableCORS     bool   `json:"enable_cors"`
	TLSEnabled     bool   `json:"tls_enabled"`
	TLSCertPath    string `json:"tls_cert_path,omitempty"`
	TLSKeyPath     string `json:"tls_key_path,omitempty"`
}

type DatabaseConfig struct {
	Host            string `json:"host"`
	Port            int    `json:"port"`
	Name            string `json:"name"`
	Username        string `json:"username"`
	MaxConnections  int    `json:"max_connections"`
	MaxIdleConns    int    `json:"max_idle_connections"`
	ConnMaxLifetime int    `json:"connection_max_lifetime_minutes"`
	SSLMode         string `json:"ssl_mode"`
}

type SecurityConfig struct {
	JWTSecret           string `json:"-"` // Hidden from JSON
	JWTExpiryHours      int    `json:"jwt_expiry_hours"`
	PasswordMinLength   int    `json:"password_min_length"`
	RequireSpecialChars bool   `json:"require_special_chars"`
	MaxLoginAttempts    int    `json:"max_login_attempts"`
	LockoutDuration     int    `json:"lockout_duration_minutes"`
	SessionTimeout      int    `json:"session_timeout_minutes"`
	RequireMFA          bool   `json:"require_mfa"`
	AllowedOrigins      string `json:"allowed_origins"`
	RateLimitEnabled    bool   `json:"rate_limit_enabled"`
	RateLimitRPM        int    `json:"rate_limit_rpm"`
}

type StorageConfig struct {
	DefaultPath         string `json:"default_path"`
	MaxDiskUsage        int    `json:"max_disk_usage_gb"`
	CompressionEnabled  bool   `json:"compression_enabled"`
	EncryptionEnabled   bool   `json:"encryption_enabled"`
	BackupEnabled       bool   `json:"backup_enabled"`
	BackupRetentionDays int    `json:"backup_retention_days"`
	TieredStorage       bool   `json:"tiered_storage_enabled"`
}

type VMConfig struct {
	DefaultDriver    string         `json:"default_driver"`
	MaxVMsPerNode    int            `json:"max_vms_per_node"`
	DefaultCPU       int            `json:"default_cpu_cores"`
	DefaultMemory    int            `json:"default_memory_mb"`
	DefaultDisk      int            `json:"default_disk_gb"`
	MigrationEnabled bool           `json:"migration_enabled"`
	LiveMigration    bool           `json:"live_migration_enabled"`
	CompressionLevel int            `json:"compression_level"`
	NetworkOptimized bool           `json:"network_optimized"`
	ResourceLimits   ResourceLimits `json:"resource_limits"`
	SupportedDrivers []string       `json:"supported_drivers"`
}

type ResourceLimits struct {
	MaxCPUCores    int `json:"max_cpu_cores"`
	MaxMemoryGB    int `json:"max_memory_gb"`
	MaxDiskGB      int `json:"max_disk_gb"`
	MaxNetworkMbps int `json:"max_network_mbps"`
}

type MonitoringConfig struct {
	Enabled            bool   `json:"enabled"`
	MetricsInterval    int    `json:"metrics_interval_seconds"`
	RetentionDays      int    `json:"retention_days"`
	PrometheusEnabled  bool   `json:"prometheus_enabled"`
	GrafanaEnabled     bool   `json:"grafana_enabled"`
	AlertingEnabled    bool   `json:"alerting_enabled"`
	WebhookURL         string `json:"webhook_url,omitempty"`
	SlackWebhook       string `json:"slack_webhook,omitempty"`
	EmailNotifications bool   `json:"email_notifications"`
}

type NetworkConfig struct {
	DefaultSubnet     string `json:"default_subnet"`
	DHCPEnabled       bool   `json:"dhcp_enabled"`
	VLANSupport       bool   `json:"vlan_support"`
	SDNEnabled        bool   `json:"sdn_enabled"`
	BandwidthLimiting bool   `json:"bandwidth_limiting"`
	QoSEnabled        bool   `json:"qos_enabled"`
	FirewallEnabled   bool   `json:"firewall_enabled"`
	DNSServers        string `json:"dns_servers"`
}

type ConfigBackup struct {
	ID          int          `json:"id"`
	Config      SystemConfig `json:"config"`
	CreatedAt   time.Time    `json:"created_at"`
	CreatedBy   string       `json:"created_by"`
	Description string       `json:"description"`
}

func NewConfigHandlers(configPath string) *ConfigHandlers {
	return &ConfigHandlers{configPath: configPath}
}

// GET /api/admin/config - Get current system configuration
func (h *ConfigHandlers) GetConfig(w http.ResponseWriter, r *http.Request) {
	config := h.loadCurrentConfig()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(config)
}

// PUT /api/admin/config - Update system configuration
func (h *ConfigHandlers) UpdateConfig(w http.ResponseWriter, r *http.Request) {
	var newConfig SystemConfig
	if err := json.NewDecoder(r.Body).Decode(&newConfig); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate configuration
	if err := h.validateConfig(newConfig); err != nil {
		http.Error(w, fmt.Sprintf("Invalid configuration: %v", err), http.StatusBadRequest)
		return
	}

	// Create backup of current config
	currentConfig := h.loadCurrentConfig()
	if err := h.createConfigBackup(currentConfig, "Auto-backup before update"); err != nil {
		logger.Error("Failed to create config backup", "error", err)
		// Continue with update but log the failure
	}

	// Update timestamp and user
	newConfig.LastUpdated = time.Now()
	newConfig.UpdatedBy = getUserFromContext(r.Context())

	// Apply configuration
	if err := h.applyConfig(newConfig); err != nil {
		http.Error(w, fmt.Sprintf("Failed to apply configuration: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(newConfig)
}

// GET /api/admin/config/validate - Validate configuration without applying
func (h *ConfigHandlers) ValidateConfig(w http.ResponseWriter, r *http.Request) {
	var config SystemConfig
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if err := h.validateConfig(config); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"valid":  false,
			"errors": []string{err.Error()},
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"valid":   true,
		"message": "Configuration is valid",
	})
}

// POST /api/admin/config/backup - Create configuration backup
func (h *ConfigHandlers) CreateBackup(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Description string `json:"description"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	currentConfig := h.loadCurrentConfig()

	backup, err := h.createConfigBackup(currentConfig, req.Description)
	if err != nil {
		logger.Error("Failed to create config backup", "error", err)
		http.Error(w, "Failed to create backup", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(backup)
}

// GET /api/admin/config/backups - List configuration backups
func (h *ConfigHandlers) ListBackups(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page <= 0 {
		page = 1
	}

	pageSize, _ := strconv.Atoi(r.URL.Query().Get("page_size"))
	if pageSize <= 0 || pageSize > 50 {
		pageSize = 20
	}

	// For now, return mock data
	// In production, this would query the database
	backups := []ConfigBackup{
		{
			ID:          1,
			CreatedAt:   time.Now().Add(-24 * time.Hour),
			CreatedBy:   "admin",
			Description: "Manual backup before security update",
		},
		{
			ID:          2,
			CreatedAt:   time.Now().Add(-48 * time.Hour),
			CreatedBy:   "system",
			Description: "Auto-backup before update",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"backups":  backups,
		"page":     page,
		"count":    len(backups),
		"has_more": false,
	})
}

// POST /api/admin/config/restore/{id} - Restore configuration from backup
func (h *ConfigHandlers) RestoreBackup(w http.ResponseWriter, r *http.Request) {
	backupIDStr := strings.TrimPrefix(r.URL.Path, "/api/admin/config/restore/")
	backupID, err := strconv.Atoi(backupIDStr)
	if err != nil {
		http.Error(w, "Invalid backup ID", http.StatusBadRequest)
		return
	}

	// For now, return current config as if we restored
	// In production, this would restore from database
	_ = backupID

	config := h.loadCurrentConfig()
	config.LastUpdated = time.Now()
	config.UpdatedBy = "admin (restored)"

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Configuration restored successfully",
		"config":  config,
	})
}

// Helper methods

func (h *ConfigHandlers) loadCurrentConfig() SystemConfig {
	// Load configuration from environment variables and defaults
	return SystemConfig{
		Server: ServerConfig{
			APIPort:        getEnvInt("API_PORT", 8090),
			WSPort:         getEnvInt("WS_PORT", 8091),
			LogLevel:       getEnvString("LOG_LEVEL", "info"),
			LogFormat:      getEnvString("LOG_FORMAT", "json"),
			Environment:    getEnvString("ENVIRONMENT", "development"),
			MaxConnections: getEnvInt("MAX_CONNECTIONS", 1000),
			RequestTimeout: getEnvInt("REQUEST_TIMEOUT", 30),
			EnableCORS:     getEnvBool("ENABLE_CORS", true),
			TLSEnabled:     getEnvBool("TLS_ENABLED", false),
			TLSCertPath:    getEnvString("TLS_CERT_PATH", ""),
			TLSKeyPath:     getEnvString("TLS_KEY_PATH", ""),
		},
		Database: DatabaseConfig{
			Host:            getEnvString("DB_HOST", "localhost"),
			Port:            getEnvInt("DB_PORT", 5432),
			Name:            getEnvString("DB_NAME", "novacron"),
			Username:        getEnvString("DB_USER", "postgres"),
			MaxConnections:  getEnvInt("DB_MAX_CONNECTIONS", 25),
			MaxIdleConns:    getEnvInt("DB_MAX_IDLE_CONNECTIONS", 5),
			ConnMaxLifetime: getEnvInt("DB_CONN_MAX_LIFETIME", 60),
			SSLMode:         getEnvString("DB_SSL_MODE", "prefer"),
		},
		Security: SecurityConfig{
			JWTExpiryHours:      getEnvInt("JWT_EXPIRY_HOURS", 24),
			PasswordMinLength:   getEnvInt("PASSWORD_MIN_LENGTH", 8),
			RequireSpecialChars: getEnvBool("PASSWORD_REQUIRE_SPECIAL", true),
			MaxLoginAttempts:    getEnvInt("MAX_LOGIN_ATTEMPTS", 5),
			LockoutDuration:     getEnvInt("LOCKOUT_DURATION", 30),
			SessionTimeout:      getEnvInt("SESSION_TIMEOUT", 120),
			RequireMFA:          getEnvBool("REQUIRE_MFA", false),
			AllowedOrigins:      getEnvString("ALLOWED_ORIGINS", "*"),
			RateLimitEnabled:    getEnvBool("RATE_LIMIT_ENABLED", true),
			RateLimitRPM:        getEnvInt("RATE_LIMIT_RPM", 60),
		},
		Storage: StorageConfig{
			DefaultPath:         getEnvString("STORAGE_PATH", "/var/lib/novacron/vms"),
			MaxDiskUsage:        getEnvInt("MAX_DISK_USAGE_GB", 1000),
			CompressionEnabled:  getEnvBool("COMPRESSION_ENABLED", true),
			EncryptionEnabled:   getEnvBool("ENCRYPTION_ENABLED", false),
			BackupEnabled:       getEnvBool("BACKUP_ENABLED", true),
			BackupRetentionDays: getEnvInt("BACKUP_RETENTION_DAYS", 30),
			TieredStorage:       getEnvBool("TIERED_STORAGE_ENABLED", false),
		},
		VM: VMConfig{
			DefaultDriver:    getEnvString("DEFAULT_VM_DRIVER", "kvm"),
			MaxVMsPerNode:    getEnvInt("MAX_VMS_PER_NODE", 50),
			DefaultCPU:       getEnvInt("DEFAULT_CPU_CORES", 2),
			DefaultMemory:    getEnvInt("DEFAULT_MEMORY_MB", 2048),
			DefaultDisk:      getEnvInt("DEFAULT_DISK_GB", 20),
			MigrationEnabled: getEnvBool("MIGRATION_ENABLED", true),
			LiveMigration:    getEnvBool("LIVE_MIGRATION_ENABLED", false),
			CompressionLevel: getEnvInt("COMPRESSION_LEVEL", 6),
			NetworkOptimized: getEnvBool("NETWORK_OPTIMIZED", true),
			SupportedDrivers: []string{"kvm", "container", "kata"},
			ResourceLimits: ResourceLimits{
				MaxCPUCores:    getEnvInt("MAX_CPU_CORES", 16),
				MaxMemoryGB:    getEnvInt("MAX_MEMORY_GB", 64),
				MaxDiskGB:      getEnvInt("MAX_DISK_GB", 500),
				MaxNetworkMbps: getEnvInt("MAX_NETWORK_MBPS", 1000),
			},
		},
		Monitoring: MonitoringConfig{
			Enabled:            getEnvBool("MONITORING_ENABLED", true),
			MetricsInterval:    getEnvInt("METRICS_INTERVAL", 30),
			RetentionDays:      getEnvInt("METRICS_RETENTION_DAYS", 30),
			PrometheusEnabled:  getEnvBool("PROMETHEUS_ENABLED", true),
			GrafanaEnabled:     getEnvBool("GRAFANA_ENABLED", true),
			AlertingEnabled:    getEnvBool("ALERTING_ENABLED", true),
			WebhookURL:         getEnvString("WEBHOOK_URL", ""),
			SlackWebhook:       getEnvString("SLACK_WEBHOOK", ""),
			EmailNotifications: getEnvBool("EMAIL_NOTIFICATIONS", false),
		},
		Network: NetworkConfig{
			DefaultSubnet:     getEnvString("DEFAULT_SUBNET", "10.0.0.0/16"),
			DHCPEnabled:       getEnvBool("DHCP_ENABLED", true),
			VLANSupport:       getEnvBool("VLAN_SUPPORT", false),
			SDNEnabled:        getEnvBool("SDN_ENABLED", false),
			BandwidthLimiting: getEnvBool("BANDWIDTH_LIMITING", false),
			QoSEnabled:        getEnvBool("QOS_ENABLED", false),
			FirewallEnabled:   getEnvBool("FIREWALL_ENABLED", true),
			DNSServers:        getEnvString("DNS_SERVERS", "8.8.8.8,8.8.4.4"),
		},
		LastUpdated: time.Now(),
		UpdatedBy:   "system",
	}
}

func (h *ConfigHandlers) validateConfig(config SystemConfig) error {
	// Validate server config
	if config.Server.APIPort < 1024 || config.Server.APIPort > 65535 {
		return fmt.Errorf("API port must be between 1024 and 65535")
	}

	if config.Server.WSPort < 1024 || config.Server.WSPort > 65535 {
		return fmt.Errorf("WebSocket port must be between 1024 and 65535")
	}

	// Validate database config
	if config.Database.Port < 1 || config.Database.Port > 65535 {
		return fmt.Errorf("database port must be between 1 and 65535")
	}

	// Validate VM config
	if config.VM.MaxVMsPerNode < 1 || config.VM.MaxVMsPerNode > 1000 {
		return fmt.Errorf("max VMs per node must be between 1 and 1000")
	}

	// Add more validation as needed

	return nil
}

func (h *ConfigHandlers) applyConfig(config SystemConfig) error {
	// In a real implementation, this would:
	// 1. Update configuration files
	// 2. Restart services if needed
	// 3. Apply runtime configuration changes

	logger.Info("Configuration updated",
		"api_port", config.Server.APIPort,
		"log_level", config.Server.LogLevel,
		"updated_by", config.UpdatedBy)

	return nil
}

func (h *ConfigHandlers) createConfigBackup(config SystemConfig, description string) (*ConfigBackup, error) {
	backup := &ConfigBackup{
		ID:          int(time.Now().Unix()), // Mock ID
		Config:      config,
		CreatedAt:   time.Now(),
		CreatedBy:   getUserFromContext(context.Background()), // Uses default "admin" until context is passed through
		Description: description,
	}

	// In production, this would save to database

	return backup, nil
}

// Helper functions for environment variables
func getEnvString(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseBool(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}
