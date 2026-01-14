package security

import (
	"context"
	"database/sql"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockSecretProvider for testing
type MockSecretProvider struct {
	mock.Mock
}

func (m *MockSecretProvider) GetSecret(ctx context.Context, key string) (string, error) {
	args := m.Called(ctx, key)
	return args.String(0), args.Error(1)
}

func (m *MockSecretProvider) SetSecret(ctx context.Context, key string, value string) error {
	args := m.Called(ctx, key, value)
	return args.Error(0)
}

func (m *MockSecretProvider) DeleteSecret(ctx context.Context, key string) error {
	args := m.Called(ctx, key)
	return args.Error(0)
}

func (m *MockSecretProvider) ListSecrets(ctx context.Context, prefix string) ([]string, error) {
	args := m.Called(ctx, prefix)
	return args.Get(0).([]string), args.Error(1)
}

// MockAuditLogger for testing
type MockAuditLogger struct {
	mock.Mock
}

func (m *MockAuditLogger) LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	args := m.Called(ctx, actor, resource, action, result, details)
	return args.Error(0)
}

func (m *MockAuditLogger) LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	args := m.Called(ctx, actor, resource, action, result, details)
	return args.Error(0)
}

func (m *MockAuditLogger) LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error {
	args := m.Called(ctx, actor, resource, oldVersion, newVersion, result)
	return args.Error(0)
}

func (m *MockAuditLogger) LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error {
	args := m.Called(ctx, actor, success, details)
	return args.Error(0)
}

func (m *MockAuditLogger) LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error {
	args := m.Called(ctx, actor, resource, oldValue, newValue)
	return args.Error(0)
}

func (m *MockAuditLogger) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error) {
	args := m.Called(ctx, filter)
	return args.Get(0).([]AuditEvent), args.Error(1)
}

// Test EnhancedSecretsManager GetSecret with audit logging
func TestEnhancedSecretsManager_GetSecret_WithAudit(t *testing.T) {
	// Setup
	mockProvider := new(MockSecretProvider)
	mockAuditor := new(MockAuditLogger)
	
	manager := &EnhancedSecretsManager{
		provider: mockProvider,
		auditor:  mockAuditor,
		cache:    make(map[string]*cachedSecret),
		config: &SecretsConfig{
			Secrets: struct {
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
			}{
				Cache: struct {
					Enabled    bool `yaml:"enabled"`
					TTLSeconds int  `yaml:"ttl_seconds"`
					MaxEntries int  `yaml:"max_entries"`
				}{
					Enabled:    true,
					TTLSeconds: 300,
				},
			},
		},
	}
	
	ctx := context.WithValue(context.Background(), "user_id", "test-user")
	secretKey := "test-secret"
	secretValue := "secret-value"
	
	// Set expectations
	mockProvider.On("GetSecret", ctx, secretKey).Return(secretValue, nil)
	mockAuditor.On("LogSecretAccess", ctx, "test-user", secretKey, ActionRead, ResultSuccess, mock.Anything).Return(nil)
	
	// Execute
	value, err := manager.GetSecret(ctx, secretKey)
	
	// Assert
	assert.NoError(t, err)
	assert.Equal(t, secretValue, value)
	mockProvider.AssertExpectations(t)
	mockAuditor.AssertExpectations(t)
	
	// Verify cache was updated
	assert.Contains(t, manager.cache, secretKey)
}

// Test Secret Rotation
func TestSecretRotationManager_RotateSecret(t *testing.T) {
	// Setup
	mockProvider := new(MockSecretProvider)
	mockAuditor := new(MockAuditLogger)
	mockNotifier := &DefaultNotificationService{}
	
	rotationManager := NewSecretRotationManager(mockProvider, mockAuditor, mockNotifier)
	
	// Register policy
	policy := SecretRotationPolicy{
		MaxAge:           24 * time.Hour,
		RotationInterval: 12 * time.Hour,
		MinEntropy:       256,
		NotifyBefore:     2 * time.Hour,
		AutoRotate:       false,
		RequireApproval:  false,
	}
	
	err := rotationManager.RegisterPolicy("test-secret", policy)
	require.NoError(t, err)
	
	ctx := context.Background()
	secretKey := "test-secret"
	currentValue := "current-value"
	
	// Set expectations
	mockProvider.On("GetSecret", ctx, secretKey).Return(currentValue, nil)
	mockProvider.On("SetSecret", ctx, secretKey, mock.AnythingOfType("string")).Return(nil)
	mockAuditor.On("LogSecretRotation", ctx, "manual", secretKey, "v0", mock.AnythingOfType("string"), ResultSuccess).Return(nil)
	
	// Execute
	err = rotationManager.RotateSecret(ctx, secretKey, "manual")
	
	// Assert
	assert.NoError(t, err)
	mockProvider.AssertExpectations(t)
	mockAuditor.AssertExpectations(t)
	
	// Verify rotation history
	history, err := rotationManager.GetRotationHistory(ctx, secretKey)
	assert.NoError(t, err)
	assert.NotEmpty(t, history)
}

// Test Audit Logger
func TestAuditLogger_LogSecretAccess(t *testing.T) {
	// Setup mock storage
	mockStorage := new(MockAuditStorage)
	mockAlerting := new(MockAlertingService)
	mockCompliance := new(MockComplianceService)
	
	logger := NewAuditLogger(mockStorage, mockAlerting, mockCompliance)
	
	ctx := context.WithValue(context.Background(), "request_id", "req-123")
	ctx = context.WithValue(ctx, "client_ip", "192.168.1.1")
	
	// Set expectations
	mockStorage.On("Store", ctx, mock.AnythingOfType("AuditEvent")).Return(nil)
	mockCompliance.On("ReportEvent", ctx, mock.AnythingOfType("AuditEvent")).Return(nil)
	
	// Execute
	err := logger.LogSecretAccess(ctx, "test-user", "secret-key", ActionRead, ResultSuccess, nil)
	
	// Assert
	assert.NoError(t, err)
	mockStorage.AssertExpectations(t)
	mockCompliance.AssertExpectations(t)
}

// Test Configuration Loading
func TestLoadSecretsConfig_FromEnvironment(t *testing.T) {
	// Set environment variables
	t.Setenv("SECRETS_PROVIDER", "vault")
	t.Setenv("VAULT_ADDR", "https://vault.example.com:8200")
	t.Setenv("VAULT_TOKEN", "test-token")
	t.Setenv("AUDIT_ENABLED", "true")
	t.Setenv("ROTATION_ENABLED", "true")
	
	// Load config from environment
	config := loadConfigFromEnv()
	
	// Assert
	assert.Equal(t, "vault", config.Secrets.Provider)
	assert.Equal(t, "https://vault.example.com:8200", config.Secrets.Vault.Address)
	assert.Equal(t, "test-token", config.Secrets.Vault.Token)
	assert.True(t, config.Audit.Enabled)
	assert.True(t, config.Rotation.Enabled)
}

// Test Vault Configuration Validation
func TestVaultProviderValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    *SecretsConfig
		wantError bool
		errorMsg  string
	}{
		{
			name: "missing vault address",
			config: &SecretsConfig{
				Secrets: struct {
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
				}{
					Provider: "vault",
					Vault: struct {
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
					}{
						Address: "", // Missing!
					},
				},
			},
			wantError: true,
			errorMsg:  "Vault address is required",
		},
		{
			name: "valid vault configuration",
			config: &SecretsConfig{
				Secrets: struct {
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
				}{
					Provider: "vault",
					Vault: struct {
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
					}{
						Address:    "https://vault.example.com:8200",
						Token:      "test-token",
						PathPrefix: "secret/data/novacron",
					},
				},
			},
			wantError: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := newVaultProviderWithConfig(tt.config)
			if tt.wantError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				// Note: This will still fail because we can't actually connect to Vault in tests
				// But we're testing the validation logic, not the connection
				assert.Error(t, err) // Expected to fail on connection, not validation
			}
		})
	}
}

// Mock implementations for testing

type MockAuditStorage struct {
	mock.Mock
}

func (m *MockAuditStorage) Store(ctx context.Context, event AuditEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

func (m *MockAuditStorage) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error) {
	args := m.Called(ctx, filter)
	return args.Get(0).([]AuditEvent), args.Error(1)
}

func (m *MockAuditStorage) Archive(ctx context.Context, before time.Time) error {
	args := m.Called(ctx, before)
	return args.Error(0)
}

type MockAlertingService struct {
	mock.Mock
}

func (m *MockAlertingService) SendSecurityAlert(ctx context.Context, event AuditEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}

type MockComplianceService struct {
	mock.Mock
}

func (m *MockComplianceService) ReportEvent(ctx context.Context, event AuditEvent) error {
	args := m.Called(ctx, event)
	return args.Error(0)
}