package security

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/secretsmanager"
	vault "github.com/hashicorp/vault/api"
)

// SecretProvider defines the interface for secret management
type SecretProvider interface {
	GetSecret(ctx context.Context, key string) (string, error)
	SetSecret(ctx context.Context, key string, value string) error
	DeleteSecret(ctx context.Context, key string) error
	ListSecrets(ctx context.Context, prefix string) ([]string, error)
}

// SecretsManager manages secrets from multiple providers
type SecretsManager struct {
	provider SecretProvider
	cache    map[string]*cachedSecret
	mu       sync.RWMutex
	ttl      time.Duration
}

type cachedSecret struct {
	value     string
	expiresAt time.Time
}

// NewSecretsManager creates a new secrets manager
func NewSecretsManager(providerType string) (*SecretsManager, error) {
	var provider SecretProvider
	var err error

	switch providerType {
	case "vault":
		provider, err = newVaultProvider()
	case "aws":
		provider, err = newAWSSecretsProvider()
	case "env":
		provider = newEnvProvider()
	default:
		provider = newEnvProvider() // Default to environment variables
	}

	if err != nil {
		return nil, fmt.Errorf("failed to initialize provider %s: %w", providerType, err)
	}

	return &SecretsManager{
		provider: provider,
		cache:    make(map[string]*cachedSecret),
		ttl:      5 * time.Minute,
	}, nil
}

// GetSecret retrieves a secret with caching
func (sm *SecretsManager) GetSecret(ctx context.Context, key string) (string, error) {
	// Check cache first
	sm.mu.RLock()
	if cached, ok := sm.cache[key]; ok && time.Now().Before(cached.expiresAt) {
		sm.mu.RUnlock()
		return cached.value, nil
	}
	sm.mu.RUnlock()

	// Fetch from provider
	value, err := sm.provider.GetSecret(ctx, key)
	if err != nil {
		return "", fmt.Errorf("failed to get secret %s: %w", key, err)
	}

	// Update cache
	sm.mu.Lock()
	sm.cache[key] = &cachedSecret{
		value:     value,
		expiresAt: time.Now().Add(sm.ttl),
	}
	sm.mu.Unlock()

	return value, nil
}

// SetSecret sets a secret value
func (sm *SecretsManager) SetSecret(ctx context.Context, key string, value string) error {
	if err := sm.provider.SetSecret(ctx, key, value); err != nil {
		return fmt.Errorf("failed to set secret %s: %w", key, err)
	}

	// Invalidate cache
	sm.mu.Lock()
	delete(sm.cache, key)
	sm.mu.Unlock()

	return nil
}

// DeleteSecret deletes a secret
func (sm *SecretsManager) DeleteSecret(ctx context.Context, key string) error {
	if err := sm.provider.DeleteSecret(ctx, key); err != nil {
		return fmt.Errorf("failed to delete secret %s: %w", key, err)
	}

	// Invalidate cache
	sm.mu.Lock()
	delete(sm.cache, key)
	sm.mu.Unlock()

	return nil
}

// VaultProvider implements SecretProvider for HashiCorp Vault
type VaultProvider struct {
	client *vault.Client
	path   string
}

func newVaultProvider() (*VaultProvider, error) {
	vaultAddr := os.Getenv("VAULT_ADDR")
	if vaultAddr == "" {
		vaultAddr = "http://localhost:8200"
	}

	config := vault.DefaultConfig()
	config.Address = vaultAddr

	client, err := vault.NewClient(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create vault client: %w", err)
	}

	token := os.Getenv("VAULT_TOKEN")
	if token != "" {
		client.SetToken(token)
	}

	return &VaultProvider{
		client: client,
		path:   "secret/data/novacron",
	}, nil
}

func (v *VaultProvider) GetSecret(ctx context.Context, key string) (string, error) {
	secret, err := v.client.Logical().ReadWithContext(ctx, fmt.Sprintf("%s/%s", v.path, key))
	if err != nil {
		return "", err
	}

	if secret == nil || secret.Data == nil {
		return "", fmt.Errorf("secret not found: %s", key)
	}

	data, ok := secret.Data["data"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid secret format")
	}

	value, ok := data["value"].(string)
	if !ok {
		return "", fmt.Errorf("secret value is not a string")
	}

	return value, nil
}

func (v *VaultProvider) SetSecret(ctx context.Context, key string, value string) error {
	data := map[string]interface{}{
		"data": map[string]interface{}{
			"value": value,
		},
	}

	_, err := v.client.Logical().WriteWithContext(ctx, fmt.Sprintf("%s/%s", v.path, key), data)
	return err
}

func (v *VaultProvider) DeleteSecret(ctx context.Context, key string) error {
	_, err := v.client.Logical().DeleteWithContext(ctx, fmt.Sprintf("%s/%s", v.path, key))
	return err
}

func (v *VaultProvider) ListSecrets(ctx context.Context, prefix string) ([]string, error) {
	secret, err := v.client.Logical().ListWithContext(ctx, v.path)
	if err != nil {
		return nil, err
	}

	if secret == nil || secret.Data == nil {
		return []string{}, nil
	}

	keys, ok := secret.Data["keys"].([]interface{})
	if !ok {
		return []string{}, nil
	}

	var result []string
	for _, k := range keys {
		if key, ok := k.(string); ok {
			result = append(result, key)
		}
	}

	return result, nil
}

// AWSSecretsProvider implements SecretProvider for AWS Secrets Manager
type AWSSecretsProvider struct {
	client *secretsmanager.Client
	prefix string
}

func newAWSSecretsProvider() (*AWSSecretsProvider, error) {
	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	return &AWSSecretsProvider{
		client: secretsmanager.NewFromConfig(cfg),
		prefix: "novacron/",
	}, nil
}

func (a *AWSSecretsProvider) GetSecret(ctx context.Context, key string) (string, error) {
	input := &secretsmanager.GetSecretValueInput{
		SecretId: aws.String(a.prefix + key),
	}

	result, err := a.client.GetSecretValue(ctx, input)
	if err != nil {
		return "", err
	}

	if result.SecretString != nil {
		return *result.SecretString, nil
	}

	return "", fmt.Errorf("secret value is not a string")
}

func (a *AWSSecretsProvider) SetSecret(ctx context.Context, key string, value string) error {
	// Try to update first
	updateInput := &secretsmanager.UpdateSecretInput{
		SecretId:     aws.String(a.prefix + key),
		SecretString: aws.String(value),
	}

	_, err := a.client.UpdateSecret(ctx, updateInput)
	if err != nil {
		// If update fails, try to create
		createInput := &secretsmanager.CreateSecretInput{
			Name:         aws.String(a.prefix + key),
			SecretString: aws.String(value),
		}
		_, err = a.client.CreateSecret(ctx, createInput)
	}

	return err
}

func (a *AWSSecretsProvider) DeleteSecret(ctx context.Context, key string) error {
	input := &secretsmanager.DeleteSecretInput{
		SecretId:                   aws.String(a.prefix + key),
		ForceDeleteWithoutRecovery: aws.Bool(true),
	}

	_, err := a.client.DeleteSecret(ctx, input)
	return err
}

func (a *AWSSecretsProvider) ListSecrets(ctx context.Context, prefix string) ([]string, error) {
	input := &secretsmanager.ListSecretsInput{
		Filters: []secretsmanager.Filter{
			{
				Key:    secretsmanager.FilterNameStringTypeName,
				Values: []string{a.prefix + prefix},
			},
		},
	}

	result, err := a.client.ListSecrets(ctx, input)
	if err != nil {
		return nil, err
	}

	var keys []string
	for _, secret := range result.SecretList {
		if secret.Name != nil {
			keys = append(keys, *secret.Name)
		}
	}

	return keys, nil
}

// EnvProvider implements SecretProvider using environment variables (for development)
type EnvProvider struct {
	mu     sync.RWMutex
	values map[string]string
}

func newEnvProvider() *EnvProvider {
	return &EnvProvider{
		values: make(map[string]string),
	}
}

func (e *EnvProvider) GetSecret(ctx context.Context, key string) (string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Check in-memory values first
	if value, ok := e.values[key]; ok {
		return value, nil
	}

	// Fall back to environment variables
	value := os.Getenv(key)
	if value == "" {
		return "", fmt.Errorf("secret not found: %s", key)
	}

	return value, nil
}

func (e *EnvProvider) SetSecret(ctx context.Context, key string, value string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.values[key] = value
	return os.Setenv(key, value)
}

func (e *EnvProvider) DeleteSecret(ctx context.Context, key string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	delete(e.values, key)
	return os.Unsetenv(key)
}

func (e *EnvProvider) ListSecrets(ctx context.Context, prefix string) ([]string, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var keys []string
	for key := range e.values {
		keys = append(keys, key)
	}

	// Also check environment variables
	for _, env := range os.Environ() {
		if len(env) > 0 {
			keys = append(keys, env[:len(env)-1])
		}
	}

	return keys, nil
}

// Config represents application configuration with secrets
type Config struct {
	DatabaseURL      string
	RedisURL         string
	JWTSecret        string
	EncryptionKey    string
	APIKeys          map[string]string
	CloudCredentials map[string]interface{}
}

// LoadConfig loads configuration with secrets
func LoadConfig(ctx context.Context, sm *SecretsManager) (*Config, error) {
	config := &Config{
		APIKeys:          make(map[string]string),
		CloudCredentials: make(map[string]interface{}),
	}

	// Load database credentials
	dbURL, err := sm.GetSecret(ctx, "DATABASE_URL")
	if err != nil {
		return nil, fmt.Errorf("failed to load database URL: %w", err)
	}
	config.DatabaseURL = dbURL

	// Load Redis credentials
	redisURL, err := sm.GetSecret(ctx, "REDIS_URL")
	if err != nil {
		return nil, fmt.Errorf("failed to load Redis URL: %w", err)
	}
	config.RedisURL = redisURL

	// Load JWT secret
	jwtSecret, err := sm.GetSecret(ctx, "JWT_SECRET")
	if err != nil {
		return nil, fmt.Errorf("failed to load JWT secret: %w", err)
	}
	config.JWTSecret = jwtSecret

	// Load encryption key
	encKey, err := sm.GetSecret(ctx, "ENCRYPTION_KEY")
	if err != nil {
		return nil, fmt.Errorf("failed to load encryption key: %w", err)
	}
	config.EncryptionKey = encKey

	// Load cloud credentials
	awsCreds, err := sm.GetSecret(ctx, "AWS_CREDENTIALS")
	if err == nil {
		var creds map[string]interface{}
		if err := json.Unmarshal([]byte(awsCreds), &creds); err == nil {
			config.CloudCredentials["aws"] = creds
		}
	}

	return config, nil
}