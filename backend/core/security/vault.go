package security

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/hashicorp/vault/api"
)

// VaultManager handles secure secret storage and retrieval
type VaultManager struct {
	client *api.Client
	cache  map[string]*vaultCachedSecret
	mu     sync.RWMutex
}

// vaultCachedSecret stores a secret with expiration
type vaultCachedSecret struct {
	value     string
	expiresAt time.Time
}

// NewVaultManager creates a new vault manager
func NewVaultManager(address string, token string) (*VaultManager, error) {
	config := api.DefaultConfig()
	config.Address = address
	
	client, err := api.NewClient(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create vault client: %w", err)
	}
	
	client.SetToken(token)
	
	// Verify connection
	health, err := client.Sys().Health()
	if err != nil {
		// For development, fallback to environment variables
		if os.Getenv("NOVACRON_ENV") == "development" {
			return &VaultManager{
				client: nil, // Use environment fallback
				cache:  make(map[string]*vaultCachedSecret),
			}, nil
		}
		return nil, fmt.Errorf("vault health check failed: %w", err)
	}
	
	if health.Sealed {
		return nil, fmt.Errorf("vault is sealed")
	}
	
	return &VaultManager{
		client: client,
		cache:  make(map[string]*vaultCachedSecret),
	}, nil
}

// GetSecret retrieves a secret from vault or cache
func (vm *VaultManager) GetSecret(ctx context.Context, path string) (string, error) {
	// Check cache first
	vm.mu.RLock()
	if cached, ok := vm.cache[path]; ok {
		if time.Now().Before(cached.expiresAt) {
			vm.mu.RUnlock()
			return cached.value, nil
		}
	}
	vm.mu.RUnlock()
	
	// Development fallback
	if vm.client == nil {
		return vm.getFromEnvironment(path)
	}
	
	// Fetch from vault
	secret, err := vm.client.Logical().ReadWithContext(ctx, path)
	if err != nil {
		return "", fmt.Errorf("failed to read secret: %w", err)
	}
	
	if secret == nil || secret.Data == nil {
		return "", fmt.Errorf("secret not found at path: %s", path)
	}
	
	// Extract value
	value, ok := secret.Data["value"].(string)
	if !ok {
		// Try data.data for KV v2
		if data, ok := secret.Data["data"].(map[string]interface{}); ok {
			if val, ok := data["value"].(string); ok {
				value = val
			}
		}
	}
	
	if value == "" {
		return "", fmt.Errorf("secret value not found")
	}
	
	// Cache the secret
	vm.mu.Lock()
	vm.cache[path] = &vaultCachedSecret{
		value:     value,
		expiresAt: time.Now().Add(5 * time.Minute),
	}
	vm.mu.Unlock()
	
	return value, nil
}

// SetSecret stores a secret in vault
func (vm *VaultManager) SetSecret(ctx context.Context, path string, value string) error {
	if vm.client == nil {
		return fmt.Errorf("vault client not initialized")
	}
	
	data := map[string]interface{}{
		"value": value,
	}
	
	_, err := vm.client.Logical().WriteWithContext(ctx, path, data)
	if err != nil {
		return fmt.Errorf("failed to write secret: %w", err)
	}
	
	// Invalidate cache
	vm.mu.Lock()
	delete(vm.cache, path)
	vm.mu.Unlock()
	
	return nil
}

// RotateSecret generates and stores a new secret value
func (vm *VaultManager) RotateSecret(ctx context.Context, path string) (string, error) {
	newSecret := generateSecureToken(32)
	
	err := vm.SetSecret(ctx, path, newSecret)
	if err != nil {
		return "", fmt.Errorf("failed to rotate secret: %w", err)
	}
	
	return newSecret, nil
}

// getFromEnvironment fallback for development
func (vm *VaultManager) getFromEnvironment(path string) (string, error) {
	// Map vault paths to environment variables
	envMap := map[string]string{
		"secret/data/novacron/auth":     "AUTH_SECRET",
		"secret/data/novacron/db":       "DB_PASSWORD",
		"secret/data/novacron/api-key":  "NOVACRON_API_KEY",
		"secret/data/novacron/jwt":      "JWT_SECRET",
		"secret/data/novacron/google":   "GOOGLE_CLIENT_SECRET",
		"secret/data/novacron/microsoft": "MICROSOFT_CLIENT_SECRET",
	}
	
	envVar, ok := envMap[path]
	if !ok {
		return "", fmt.Errorf("unknown secret path: %s", path)
	}
	
	value := os.Getenv(envVar)
	if value == "" {
		// Generate a secure default for development
		if os.Getenv("NOVACRON_ENV") == "development" {
			value = generateSecureToken(32)
			os.Setenv(envVar, value)
		} else {
			return "", fmt.Errorf("secret not found in environment: %s", envVar)
		}
	}
	
	return value, nil
}

// SecretConfig holds application secrets
type SecretConfig struct {
	AuthSecret          string
	DatabasePassword    string
	APIKey              string
	JWTSecret           string
	GoogleClientSecret  string
	MicrosoftClientSecret string
	TLSCertPath         string
	TLSKeyPath          string
}

// LoadSecrets loads all application secrets
func (vm *VaultManager) LoadSecrets(ctx context.Context) (*SecretConfig, error) {
	config := &SecretConfig{}
	
	// Load each secret
	var err error
	config.AuthSecret, err = vm.GetSecret(ctx, "secret/data/novacron/auth")
	if err != nil {
		return nil, fmt.Errorf("failed to load auth secret: %w", err)
	}
	
	config.DatabasePassword, err = vm.GetSecret(ctx, "secret/data/novacron/db")
	if err != nil {
		return nil, fmt.Errorf("failed to load database password: %w", err)
	}
	
	config.APIKey, err = vm.GetSecret(ctx, "secret/data/novacron/api-key")
	if err != nil {
		return nil, fmt.Errorf("failed to load API key: %w", err)
	}
	
	config.JWTSecret, err = vm.GetSecret(ctx, "secret/data/novacron/jwt")
	if err != nil {
		return nil, fmt.Errorf("failed to load JWT secret: %w", err)
	}
	
	// OAuth secrets (optional)
	config.GoogleClientSecret, _ = vm.GetSecret(ctx, "secret/data/novacron/google")
	config.MicrosoftClientSecret, _ = vm.GetSecret(ctx, "secret/data/novacron/microsoft")
	
	// TLS paths
	config.TLSCertPath, _ = vm.GetSecret(ctx, "secret/data/novacron/tls-cert-path")
	config.TLSKeyPath, _ = vm.GetSecret(ctx, "secret/data/novacron/tls-key-path")
	
	return config, nil
}

// VaultInitializer helps set up vault for first use
type VaultInitializer struct {
	client *api.Client
}

// NewVaultInitializer creates a vault initializer
func NewVaultInitializer(address string, rootToken string) (*VaultInitializer, error) {
	config := api.DefaultConfig()
	config.Address = address
	
	client, err := api.NewClient(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create vault client: %w", err)
	}
	
	client.SetToken(rootToken)
	
	return &VaultInitializer{client: client}, nil
}

// SetupNovaCronSecrets initializes all required secrets
func (vi *VaultInitializer) SetupNovaCronSecrets(ctx context.Context) error {
	secrets := map[string]string{
		"secret/data/novacron/auth":     generateSecureToken(64),
		"secret/data/novacron/db":       generateSecureToken(32),
		"secret/data/novacron/api-key":  generateSecureToken(48),
		"secret/data/novacron/jwt":      generateSecureToken(64),
	}
	
	for path, value := range secrets {
		data := map[string]interface{}{
			"data": map[string]interface{}{
				"value": value,
			},
		}
		
		_, err := vi.client.Logical().WriteWithContext(ctx, path, data)
		if err != nil {
			return fmt.Errorf("failed to write secret %s: %w", path, err)
		}
		
		fmt.Printf("Created secret at %s\n", path)
	}
	
	// Create policies
	policy := `
path "secret/data/novacron/*" {
  capabilities = ["read", "list"]
}

path "secret/metadata/novacron/*" {
  capabilities = ["list", "read"]
}
`
	
	err := vi.client.Sys().PutPolicyWithContext(ctx, "novacron-read", policy)
	if err != nil {
		return fmt.Errorf("failed to create policy: %w", err)
	}
	
	fmt.Println("Created novacron-read policy")
	
	return nil
}

// CreateAppToken creates a token for the application
func (vi *VaultInitializer) CreateAppToken(ctx context.Context) (string, error) {
	req := &api.TokenCreateRequest{
		Policies: []string{"novacron-read"},
		TTL:      "720h", // 30 days
		Renewable: &[]bool{true}[0],
		DisplayName: "novacron-app",
	}
	
	resp, err := vi.client.Auth().Token().CreateWithContext(ctx, req)
	if err != nil {
		return "", fmt.Errorf("failed to create token: %w", err)
	}
	
	return resp.Auth.ClientToken, nil
}