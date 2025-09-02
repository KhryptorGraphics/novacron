package security

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	vault "github.com/hashicorp/vault/api"
	"gopkg.in/yaml.v3"
)

// SecretsConfig represents the configuration from secrets.yaml
type SecretsConfig struct {
	Secrets struct {
		Provider string `yaml:"provider"`
		Cache    struct {
			Enabled    bool `yaml:"enabled"`
			TTLSeconds int  `yaml:"ttl_seconds"`
			MaxEntries int  `yaml:"max_entries"`
		} `yaml:"cache"`
		Vault struct {
			Address    string `yaml:"address"`
			Token      string `yaml:"token"`
			Namespace  string `yaml:"namespace"`
			PathPrefix string `yaml:"path_prefix"`
			TLS        struct {
				Enabled    bool   `yaml:"enabled"`
				CACert     string `yaml:"ca_cert"`
				ClientCert string `yaml:"client_cert"`
				ClientKey  string `yaml:"client_key"`
				SkipVerify bool   `yaml:"skip_verify"`
			} `yaml:"tls"`
			Auth struct {
				Method   string `yaml:"method"`
				AppRole  struct {
					RoleID   string `yaml:"role_id"`
					SecretID string `yaml:"secret_id"`
				} `yaml:"approle"`
				Kubernetes struct {
					Role    string `yaml:"role"`
					JWTPath string `yaml:"jwt_path"`
				} `yaml:"kubernetes"`
			} `yaml:"auth"`
		} `yaml:"vault"`
		AWS struct {
			Region           string `yaml:"region"`
			Prefix           string `yaml:"prefix"`
			Endpoint         string `yaml:"endpoint"`
			AccessKeyID      string `yaml:"access_key_id"`
			SecretAccessKey  string `yaml:"secret_access_key"`
			SessionToken     string `yaml:"session_token"`
		} `yaml:"aws"`
		Env struct {
			Prefix string `yaml:"prefix"`
		} `yaml:"env"`
	} `yaml:"secrets"`
	Audit struct {
		Enabled bool `yaml:"enabled"`
		Storage struct {
			Type string `yaml:"type"`
		} `yaml:"storage"`
	} `yaml:"audit"`
	Rotation struct {
		Enabled       bool `yaml:"enabled"`
		DefaultPolicy struct {
			MaxAgeHours           int  `yaml:"max_age_hours"`
			RotationIntervalHours int  `yaml:"rotation_interval_hours"`
			MinEntropyBits        int  `yaml:"min_entropy_bits"`
			NotifyBeforeHours     int  `yaml:"notify_before_hours"`
			AutoRotate            bool `yaml:"auto_rotate"`
			RequireApproval       bool `yaml:"require_approval"`
		} `yaml:"default_policy"`
	} `yaml:"rotation"`
}

// EnhancedSecretsManager manages secrets with audit and rotation
type EnhancedSecretsManager struct {
	provider        SecretProvider
	auditor         AuditLogger
	rotationManager *SecretRotationManager
	cache           map[string]*cachedSecret
	config          *SecretsConfig
	mu              sync.RWMutex
}

// LoadSecretsConfig loads configuration from file
func LoadSecretsConfig(configPath string) (*SecretsConfig, error) {
	// Default to environment-based config if file doesn't exist
	if configPath == "" {
		configPath = os.Getenv("SECRETS_CONFIG_PATH")
		if configPath == "" {
			configPath = "/etc/novacron/secrets.yaml"
		}
	}

	// Read config file
	data, err := ioutil.ReadFile(configPath)
	if err != nil {
		// Fall back to environment variables
		return loadConfigFromEnv(), nil
	}

	// Parse YAML with environment variable expansion
	config := &SecretsConfig{}
	if err := yaml.Unmarshal(data, config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Expand environment variables
	expandConfigEnvVars(config)

	return config, nil
}

// loadConfigFromEnv creates config from environment variables
func loadConfigFromEnv() *SecretsConfig {
	config := &SecretsConfig{}
	
	// Provider configuration
	config.Secrets.Provider = getEnvOrDefault("SECRETS_PROVIDER", "vault")
	
	// Cache configuration
	config.Secrets.Cache.Enabled = getEnvOrDefault("SECRETS_CACHE_ENABLED", "true") == "true"
	config.Secrets.Cache.TTLSeconds = getEnvIntOrDefault("SECRETS_CACHE_TTL", 300)
	config.Secrets.Cache.MaxEntries = getEnvIntOrDefault("SECRETS_CACHE_MAX_ENTRIES", 1000)
	
	// Vault configuration - NO DEFAULTS for critical settings
	config.Secrets.Vault.Address = os.Getenv("VAULT_ADDR") // No default!
	config.Secrets.Vault.Token = os.Getenv("VAULT_TOKEN")
	config.Secrets.Vault.PathPrefix = getEnvOrDefault("VAULT_PATH_PREFIX", "secret/data/novacron")
	
	// Audit configuration
	config.Audit.Enabled = getEnvOrDefault("AUDIT_ENABLED", "true") == "true"
	config.Audit.Storage.Type = getEnvOrDefault("AUDIT_STORAGE_TYPE", "database")
	
	// Rotation configuration
	config.Rotation.Enabled = getEnvOrDefault("ROTATION_ENABLED", "true") == "true"
	config.Rotation.DefaultPolicy.MaxAgeHours = getEnvIntOrDefault("ROTATION_MAX_AGE_HOURS", 2160)
	config.Rotation.DefaultPolicy.RotationIntervalHours = getEnvIntOrDefault("ROTATION_INTERVAL_HOURS", 1440)
	config.Rotation.DefaultPolicy.MinEntropyBits = getEnvIntOrDefault("ROTATION_MIN_ENTROPY_BITS", 256)
	config.Rotation.DefaultPolicy.NotifyBeforeHours = getEnvIntOrDefault("ROTATION_NOTIFY_HOURS", 168)
	config.Rotation.DefaultPolicy.AutoRotate = getEnvOrDefault("ROTATION_AUTO_ROTATE", "false") == "true"
	config.Rotation.DefaultPolicy.RequireApproval = getEnvOrDefault("ROTATION_REQUIRE_APPROVAL", "true") == "true"
	
	return config
}

// expandConfigEnvVars expands environment variables in config
func expandConfigEnvVars(config *SecretsConfig) {
	// Expand Vault address - REQUIRED, no default
	if config.Secrets.Vault.Address == "${VAULT_ADDR}" || config.Secrets.Vault.Address == "" {
		config.Secrets.Vault.Address = os.Getenv("VAULT_ADDR")
	}
	
	// Expand other variables with defaults where appropriate
	if config.Secrets.Vault.Token == "${VAULT_TOKEN}" {
		config.Secrets.Vault.Token = os.Getenv("VAULT_TOKEN")
	}
}

// NewEnhancedSecretsManager creates an enhanced secrets manager
func NewEnhancedSecretsManager(configPath string, db *sql.DB) (*EnhancedSecretsManager, error) {
	// Load configuration
	config, err := LoadSecretsConfig(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load config: %w", err)
	}

	// Validate critical configuration
	if config.Secrets.Provider == "vault" && config.Secrets.Vault.Address == "" {
		return nil, fmt.Errorf("VAULT_ADDR is required when using Vault provider")
	}

	// Create provider based on configuration
	provider, err := createProvider(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider: %w", err)
	}

	// Create audit logger if enabled
	var auditor AuditLogger
	if config.Audit.Enabled {
		storage := NewDatabaseAuditStorage(db)
		alerting := NewDefaultNotificationService() // Can be enhanced
		compliance := &noopComplianceService{}      // Can be enhanced
		auditor = NewAuditLogger(storage, alerting, compliance)
	} else {
		auditor = &noopAuditLogger{}
	}

	// Create rotation manager if enabled
	var rotationManager *SecretRotationManager
	if config.Rotation.Enabled {
		notifier := NewDefaultNotificationService()
		rotationManager = NewSecretRotationManager(provider, auditor, notifier)
	}

	manager := &EnhancedSecretsManager{
		provider:        provider,
		auditor:         auditor,
		rotationManager: rotationManager,
		cache:           make(map[string]*cachedSecret),
		config:          config,
	}

	// Start rotation scheduler if enabled
	if rotationManager != nil {
		ctx := context.Background()
		if err := rotationManager.Start(ctx); err != nil {
			return nil, fmt.Errorf("failed to start rotation scheduler: %w", err)
		}
	}

	return manager, nil
}

// createProvider creates the appropriate secret provider
func createProvider(config *SecretsConfig) (SecretProvider, error) {
	switch config.Secrets.Provider {
	case "vault":
		return newVaultProviderWithConfig(config)
	case "aws":
		return newAWSProviderWithConfig(config)
	case "env":
		if os.Getenv("NOVACRON_ENV") == "production" {
			return nil, fmt.Errorf("environment variable provider not allowed in production")
		}
		return newEnvProviderWithConfig(config), nil
	default:
		return nil, fmt.Errorf("unsupported provider: %s", config.Secrets.Provider)
	}
}

// newVaultProviderWithConfig creates Vault provider from config
func newVaultProviderWithConfig(config *SecretsConfig) (*VaultProvider, error) {
	if config.Secrets.Vault.Address == "" {
		return nil, fmt.Errorf("Vault address is required")
	}

	vaultConfig := vault.DefaultConfig()
	vaultConfig.Address = config.Secrets.Vault.Address
	
	// Configure TLS if enabled
	if config.Secrets.Vault.TLS.Enabled {
		tlsConfig := &vault.TLSConfig{
			CACert:        config.Secrets.Vault.TLS.CACert,
			ClientCert:    config.Secrets.Vault.TLS.ClientCert,
			ClientKey:     config.Secrets.Vault.TLS.ClientKey,
			Insecure:      config.Secrets.Vault.TLS.SkipVerify,
		}
		if err := vaultConfig.ConfigureTLS(tlsConfig); err != nil {
			return nil, fmt.Errorf("failed to configure TLS: %w", err)
		}
	}

	client, err := vault.NewClient(vaultConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create vault client: %w", err)
	}

	// Configure authentication
	switch config.Secrets.Vault.Auth.Method {
	case "token":
		if config.Secrets.Vault.Token != "" {
			client.SetToken(config.Secrets.Vault.Token)
		}
	case "approle":
		// Implement AppRole auth
		if err := authenticateWithAppRole(client, config); err != nil {
			return nil, fmt.Errorf("AppRole auth failed: %w", err)
		}
	case "kubernetes":
		// Implement Kubernetes auth
		if err := authenticateWithKubernetes(client, config); err != nil {
			return nil, fmt.Errorf("Kubernetes auth failed: %w", err)
		}
	}

	return &VaultProvider{
		client: client,
		path:   config.Secrets.Vault.PathPrefix,
	}, nil
}

// GetSecret retrieves a secret with audit logging
func (m *EnhancedSecretsManager) GetSecret(ctx context.Context, key string) (string, error) {
	// Extract actor from context
	actor := getActorFromContext(ctx)

	// Check cache first if enabled
	if m.config.Secrets.Cache.Enabled {
		m.mu.RLock()
		if cached, ok := m.cache[key]; ok && time.Now().Before(cached.expiresAt) {
			m.mu.RUnlock()
			// Log cache hit
			m.auditor.LogSecretAccess(ctx, actor, key, ActionRead, ResultSuccess, map[string]interface{}{
				"source": "cache",
			})
			return cached.value, nil
		}
		m.mu.RUnlock()
	}

	// Fetch from provider
	value, err := m.provider.GetSecret(ctx, key)
	if err != nil {
		// Log failure
		m.auditor.LogSecretAccess(ctx, actor, key, ActionRead, ResultFailure, map[string]interface{}{
			"error": err.Error(),
		})
		return "", fmt.Errorf("failed to get secret %s: %w", key, err)
	}

	// Log success
	m.auditor.LogSecretAccess(ctx, actor, key, ActionRead, ResultSuccess, map[string]interface{}{
		"source": "provider",
	})

	// Update cache if enabled
	if m.config.Secrets.Cache.Enabled {
		m.mu.Lock()
		m.cache[key] = &cachedSecret{
			value:     value,
			expiresAt: time.Now().Add(time.Duration(m.config.Secrets.Cache.TTLSeconds) * time.Second),
		}
		m.mu.Unlock()
	}

	return value, nil
}

// SetSecret sets a secret with audit logging
func (m *EnhancedSecretsManager) SetSecret(ctx context.Context, key string, value string) error {
	actor := getActorFromContext(ctx)

	// Perform the operation
	err := m.provider.SetSecret(ctx, key, value)
	
	if err != nil {
		// Log failure
		m.auditor.LogSecretModification(ctx, actor, key, ActionWrite, ResultFailure, map[string]interface{}{
			"error": err.Error(),
		})
		return fmt.Errorf("failed to set secret %s: %w", key, err)
	}

	// Log success
	m.auditor.LogSecretModification(ctx, actor, key, ActionWrite, ResultSuccess, nil)

	// Invalidate cache
	if m.config.Secrets.Cache.Enabled {
		m.mu.Lock()
		delete(m.cache, key)
		m.mu.Unlock()
	}

	return nil
}

// RotateSecret rotates a secret
func (m *EnhancedSecretsManager) RotateSecret(ctx context.Context, key string) error {
	if m.rotationManager == nil {
		return fmt.Errorf("rotation is not enabled")
	}

	actor := getActorFromContext(ctx)
	return m.rotationManager.RotateSecret(ctx, key, actor)
}

// RegisterRotationPolicy registers a rotation policy
func (m *EnhancedSecretsManager) RegisterRotationPolicy(key string, policy SecretRotationPolicy) error {
	if m.rotationManager == nil {
		return fmt.Errorf("rotation is not enabled")
	}

	return m.rotationManager.RegisterPolicy(key, policy)
}

// Helper functions

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvIntOrDefault(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		var intVal int
		fmt.Sscanf(value, "%d", &intVal)
		return intVal
	}
	return defaultValue
}

func getActorFromContext(ctx context.Context) string {
	if actor := ctx.Value("user_id"); actor != nil {
		return actor.(string)
	}
	if actor := ctx.Value("service_account"); actor != nil {
		return actor.(string)
	}
	return "system"
}

// Noop implementations for optional services

type noopAuditLogger struct{}

func (n *noopAuditLogger) LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	return nil
}

func (n *noopAuditLogger) LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	return nil
}

func (n *noopAuditLogger) LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error {
	return nil
}

func (n *noopAuditLogger) LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error {
	return nil
}

func (n *noopAuditLogger) LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error {
	return nil
}

func (n *noopAuditLogger) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error) {
	return nil, nil
}

type noopComplianceService struct{}

func (n *noopComplianceService) ReportEvent(ctx context.Context, event AuditEvent) error {
	return nil
}

// Authentication helpers

func authenticateWithAppRole(client *vault.Client, config *SecretsConfig) error {
	data := map[string]interface{}{
		"role_id":   config.Secrets.Vault.Auth.AppRole.RoleID,
		"secret_id": config.Secrets.Vault.Auth.AppRole.SecretID,
	}
	
	resp, err := client.Logical().Write("auth/approle/login", data)
	if err != nil {
		return err
	}
	
	if resp.Auth == nil {
		return fmt.Errorf("no auth info returned")
	}
	
	client.SetToken(resp.Auth.ClientToken)
	return nil
}

func authenticateWithKubernetes(client *vault.Client, config *SecretsConfig) error {
	jwt, err := ioutil.ReadFile(config.Secrets.Vault.Auth.Kubernetes.JWTPath)
	if err != nil {
		return fmt.Errorf("failed to read JWT: %w", err)
	}
	
	data := map[string]interface{}{
		"role": config.Secrets.Vault.Auth.Kubernetes.Role,
		"jwt":  string(jwt),
	}
	
	resp, err := client.Logical().Write("auth/kubernetes/login", data)
	if err != nil {
		return err
	}
	
	if resp.Auth == nil {
		return fmt.Errorf("no auth info returned")
	}
	
	client.SetToken(resp.Auth.ClientToken)
	return nil
}

// newAWSProviderWithConfig creates AWS provider from config
func newAWSProviderWithConfig(config *SecretsConfig) (*AWSSecretsProvider, error) {
	cfg, err := config.LoadDefaultConfig(context.Background(),
		config.WithRegion(config.Secrets.AWS.Region),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	return &AWSSecretsProvider{
		client: secretsmanager.NewFromConfig(cfg),
		prefix: config.Secrets.AWS.Prefix,
	}, nil
}

// newEnvProviderWithConfig creates environment provider from config
func newEnvProviderWithConfig(config *SecretsConfig) *EnvProvider {
	return &EnvProvider{
		values: make(map[string]string),
	}
}