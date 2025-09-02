package security

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	
	_ "github.com/lib/pq"
)

// SecurityConfig holds all security configuration
type SecurityConfig struct {
	VaultAddress string
	VaultToken   string
	TLSCertPath  string
	TLSKeyPath   string
	DatabaseURL  string
}

// InitializeSecurity sets up all security components
func InitializeSecurity(ctx context.Context, config *SecurityConfig) (*SecurityManager, error) {
	// Initialize Vault
	vault, err := NewVaultManager(config.VaultAddress, config.VaultToken)
	if err != nil {
		log.Printf("Warning: Vault initialization failed, using environment fallback: %v", err)
		vault = &VaultManager{
			cache: make(map[string]*cachedSecret),
		}
	}
	
	// Load secrets from vault
	secrets, err := vault.LoadSecrets(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to load secrets: %w", err)
	}
	
	// Initialize database with secure connection
	dbURL := config.DatabaseURL
	if dbURL == "" {
		dbURL = fmt.Sprintf("postgres://novacron:%s@localhost/novacron?sslmode=require",
			secrets.DatabasePassword)
	}
	
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}
	
	// Create secure database wrapper
	secureDB := NewSecureDB(db)
	
	// Initialize TLS configuration
	tlsConfig := NewTLSConfig(config.TLSCertPath, config.TLSKeyPath)
	
	// Generate self-signed cert for development if needed
	if _, err := os.Stat(config.TLSCertPath); os.IsNotExist(err) {
		log.Println("Generating self-signed certificate for development")
		hosts := []string{"localhost", "127.0.0.1", "novacron.local"}
		if err := GenerateSelfSignedCert(hosts, config.TLSCertPath, config.TLSKeyPath); err != nil {
			return nil, fmt.Errorf("failed to generate certificate: %w", err)
		}
	}
	
	return &SecurityManager{
		Vault:     vault,
		Database:  secureDB,
		TLS:       tlsConfig,
		Secrets:   secrets,
		Hasher:    NewPasswordHasher(),
		Validator: NewInputValidator(),
	}, nil
}

// SecurityManager manages all security components
type SecurityManager struct {
	Vault     *VaultManager
	Database  *SecureDB
	TLS       *TLSConfig
	Secrets   *SecretConfig
	Hasher    *PasswordHasher
	Validator *InputValidator
}

// ExampleAPIServer shows how to use the security components
func ExampleAPIServer() {
	ctx := context.Background()
	
	// Initialize security
	securityConfig := &SecurityConfig{
		VaultAddress: getEnvOrDefault("VAULT_ADDR", "http://localhost:8200"),
		VaultToken:   getEnvOrDefault("VAULT_TOKEN", "dev-token"),
		TLSCertPath:  getEnvOrDefault("TLS_CERT_PATH", "/etc/novacron/tls/cert.pem"),
		TLSKeyPath:   getEnvOrDefault("TLS_KEY_PATH", "/etc/novacron/tls/key.pem"),
		DatabaseURL:  os.Getenv("DATABASE_URL"),
	}
	
	security, err := InitializeSecurity(ctx, securityConfig)
	if err != nil {
		log.Fatalf("Failed to initialize security: %v", err)
	}
	
	// Create repositories with secure database
	vmRepo := NewVMRepository(security.Database)
	userRepo := NewUserRepository(security.Database)
	
	// Create HTTP handlers with security
	mux := http.NewServeMux()
	
	// VM endpoints with parameterized queries
	mux.HandleFunc("/api/vms", func(w http.ResponseWriter, r *http.Request) {
		// Get VMs with optional state filter (safe from SQL injection)
		state := r.URL.Query().Get("state")
		vms, err := vmRepo.GetVMs(r.Context(), state)
		if err != nil {
			http.Error(w, "Failed to get VMs", http.StatusInternalServerError)
			return
		}
		// Return VMs as JSON...
		_ = vms
	})
	
	mux.HandleFunc("/api/vms/", func(w http.ResponseWriter, r *http.Request) {
		// Extract VM ID from path
		vmID := strings.TrimPrefix(r.URL.Path, "/api/vms/")
		
		// Validate input
		if err := security.Validator.ValidateNodeID(vmID); err != nil {
			http.Error(w, "Invalid VM ID", http.StatusBadRequest)
			return
		}
		
		// Get VM by ID (safe from SQL injection)
		vm, err := vmRepo.GetVMByID(r.Context(), vmID)
		if err != nil {
			http.Error(w, "VM not found", http.StatusNotFound)
			return
		}
		// Return VM as JSON...
		_ = vm
	})
	
	// Authentication endpoint with secure password handling
	mux.HandleFunc("/api/auth/login", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Email    string `json:"email"`
			Password string `json:"password"`
		}
		// Parse request...
		
		// Validate email
		if !ValidateEmail(req.Email) {
			http.Error(w, "Invalid email", http.StatusBadRequest)
			return
		}
		
		// Get user (safe from SQL injection)
		user, err := userRepo.GetUserByEmail(r.Context(), req.Email)
		if err != nil {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}
		
		// Verify password
		if !security.Hasher.VerifyPassword(req.Password, user.PasswordHash) {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}
		
		// Generate JWT with secret from vault
		// token := generateJWT(user, security.Secrets.JWTSecret)
		// Return token...
	})
	
	// Apply security headers middleware
	handler := SecurityHeaders(mux)
	
	// Create TLS server
	tlsServer, err := NewTLSServer(":8443", handler, security.TLS)
	if err != nil {
		log.Fatalf("Failed to create TLS server: %v", err)
	}
	
	// Start HTTP to HTTPS redirect server
	go func() {
		redirectHandler := &HTTPSRedirectHandler{HTTPSPort: "8443"}
		log.Println("Starting HTTP redirect server on :8080")
		if err := http.ListenAndServe(":8080", redirectHandler); err != nil {
			log.Printf("HTTP redirect server error: %v", err)
		}
	}()
	
	// Start HTTPS server
	log.Println("Starting HTTPS server on :8443")
	if err := tlsServer.Start(); err != nil {
		log.Fatalf("HTTPS server error: %v", err)
	}
}

// getEnvOrDefault gets environment variable or returns default
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}