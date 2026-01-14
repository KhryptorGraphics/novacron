package auth

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

// SecurityConfiguration holds all security service configurations
type SecurityConfiguration struct {
	JWT                JWTConfiguration
	PasswordSecurity   PasswordSecurityConfig
	Encryption         EncryptionConfig
	SecurityMiddleware SecurityConfig
	OAuth2             map[string]OAuth2Config // Provider name -> config
	Compliance         bool
	ZeroTrust          bool
	AuditLogging       bool
}

// SecurityManager manages all security services
type SecurityManager struct {
	jwtService         *JWTService
	passwordService    *PasswordSecurityService
	encryptionService  *EncryptionService
	securityMiddleware *SecurityMiddleware
	oauth2Services     map[string]*OAuth2Service
	complianceService  *ComplianceService
	ztNetworkService   *ZeroTrustNetworkService
	authService        AuthService
	auditService       AuditService
	config             SecurityConfiguration
}

// NewSecurityManager creates a new security manager with all services
func NewSecurityManager(config SecurityConfiguration, authService AuthService) (*SecurityManager, error) {
	// Initialize audit service first (used by other services)
	auditService := NewInMemoryAuditService()

	// Initialize encryption service
	encryptionService := NewEncryptionService(config.Encryption)

	// Initialize JWT service
	jwtService := NewJWTService(config.JWT)

	// Initialize password security service
	passwordService := NewPasswordSecurityService(config.PasswordSecurity)

	// Initialize security middleware
	securityMiddleware := NewSecurityMiddleware(config.SecurityMiddleware, auditService, encryptionService)

	// Initialize OAuth2 services
	oauth2Services := make(map[string]*OAuth2Service)
	for providerName, oauth2Config := range config.OAuth2 {
		oauth2Services[providerName] = NewOAuth2Service(oauth2Config, jwtService)
	}

	// Initialize compliance service (if enabled)
	var complianceService *ComplianceService
	if config.Compliance {
		complianceService = NewComplianceService(auditService, encryptionService)
	}

	// Initialize zero-trust network service (if enabled)
	var ztNetworkService *ZeroTrustNetworkService
	if config.ZeroTrust {
		ztNetworkService = NewZeroTrustNetworkService(auditService, encryptionService)

		// Set up default network policies
		if err := setupDefaultNetworkPolicies(ztNetworkService); err != nil {
			return nil, fmt.Errorf("failed to setup default network policies: %w", err)
		}
	}

	return &SecurityManager{
		jwtService:         jwtService,
		passwordService:    passwordService,
		encryptionService:  encryptionService,
		securityMiddleware: securityMiddleware,
		oauth2Services:     oauth2Services,
		complianceService:  complianceService,
		ztNetworkService:   ztNetworkService,
		authService:        authService,
		auditService:       auditService,
		config:             config,
	}, nil
}

// GetSecurityMiddleware returns the security middleware for HTTP integration
func (sm *SecurityManager) GetSecurityMiddleware() func(http.Handler) http.Handler {
	return sm.securityMiddleware.Middleware(sm.authService)
}

// AuthenticateWithJWT authenticates a user and returns JWT tokens
func (sm *SecurityManager) AuthenticateWithJWT(username, password string) (*TokenPair, *Session, error) {
	// Authenticate with auth service
	session, err := sm.authService.Login(username, password)
	if err != nil {
		return nil, nil, fmt.Errorf("authentication failed: %w", err)
	}

	// Get user roles and permissions
	roles, err := sm.authService.GetUserRoles(session.UserID)
	if err != nil {
		roles = []*Role{} // Empty roles if error
	}

	roleNames := make([]string, len(roles))
	permissions := make([]string, 0)

	for i, role := range roles {
		roleNames[i] = role.Name
		for _, perm := range role.Permissions {
			permissions = append(permissions, fmt.Sprintf("%s:%s", perm.Resource, perm.Action))
		}
	}

	// Generate JWT tokens
	tokens, err := sm.jwtService.GenerateTokenPair(
		session.UserID,
		session.TenantID,
		roleNames,
		permissions,
		session.ID,
		session.Metadata,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("token generation failed: %w", err)
	}

	return tokens, session, nil
}

// ValidateJWT validates a JWT token and returns claims
func (sm *SecurityManager) ValidateJWT(tokenString string) (*JWTClaims, error) {
	return sm.jwtService.ValidateToken(tokenString)
}

// RefreshJWT refreshes JWT tokens
func (sm *SecurityManager) RefreshJWT(refreshToken string) (*TokenPair, error) {
	return sm.jwtService.RefreshToken(refreshToken)
}

// OAuth2Login initiates OAuth2 authentication
func (sm *SecurityManager) OAuth2Login(provider, tenantID, redirectTo string) (string, *OAuth2State, error) {
	oauth2Service, exists := sm.oauth2Services[provider]
	if !exists {
		return "", nil, fmt.Errorf("OAuth2 provider not configured: %s", provider)
	}

	return oauth2Service.GetAuthorizationURL(tenantID, redirectTo)
}

// OAuth2Callback handles OAuth2 callback
func (sm *SecurityManager) OAuth2Callback(ctx context.Context, provider, code, state string) (*TokenPair, *User, error) {
	oauth2Service, exists := sm.oauth2Services[provider]
	if !exists {
		return nil, nil, fmt.Errorf("OAuth2 provider not configured: %s", provider)
	}

	// Exchange code for tokens
	oauth2Token, oauth2State, err := oauth2Service.ExchangeCodeForToken(ctx, code, state)
	if err != nil {
		return nil, nil, fmt.Errorf("token exchange failed: %w", err)
	}

	// Get user info
	userInfo, err := oauth2Service.GetUserInfo(ctx, oauth2Token.AccessToken)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get user info: %w", err)
	}

	// Create or update user
	user, err := oauth2Service.CreateUserFromOAuth2(userInfo, oauth2State.TenantID)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create user: %w", err)
	}

	// Try to find existing user
	existingUser, err := sm.authService.(*AuthServiceImpl).users.GetByEmail(user.Email)
	if err != nil {
		// User doesn't exist, create new one
		err = sm.authService.CreateUser(user, "")
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create user in auth service: %w", err)
		}
	} else {
		// Update existing user
		existingUser.LastLogin = time.Now()
		user = existingUser
	}

	// Generate JWT tokens
	roles, _ := sm.authService.GetUserRoles(user.ID)
	roleNames := make([]string, len(roles))
	permissions := make([]string, 0)

	for i, role := range roles {
		roleNames[i] = role.Name
		for _, perm := range role.Permissions {
			permissions = append(permissions, fmt.Sprintf("%s:%s", perm.Resource, perm.Action))
		}
	}

	tokens, err := sm.jwtService.GenerateTokenPair(
		user.ID,
		user.TenantID,
		roleNames,
		permissions,
		"", // No session ID for OAuth2
		user.Metadata,
	)
	if err != nil {
		return nil, nil, fmt.Errorf("token generation failed: %w", err)
	}

	return tokens, user, nil
}

// EncryptSensitiveData encrypts sensitive data
func (sm *SecurityManager) EncryptSensitiveData(data string) (string, error) {
	// Get or create a data encryption key
	keys := sm.encryptionService.GetActiveKeys()
	if len(keys) == 0 {
		key, err := sm.encryptionService.GenerateKey("")
		if err != nil {
			return "", fmt.Errorf("failed to generate encryption key: %w", err)
		}
		keys = []*EncryptionKey{key}
	}

	return sm.encryptionService.EncryptString(data, keys[0].ID)
}

// DecryptSensitiveData decrypts sensitive data
func (sm *SecurityManager) DecryptSensitiveData(encryptedData string) (string, error) {
	return sm.encryptionService.DecryptString(encryptedData)
}

// ValidatePassword validates password against security policy
func (sm *SecurityManager) ValidatePassword(password string, user *User) error {
	return sm.passwordService.ValidatePassword(password, user)
}

// HashPassword creates a secure password hash
func (sm *SecurityManager) HashPassword(password string) (*PasswordHash, error) {
	return sm.passwordService.HashPassword(password)
}

// VerifyPassword verifies a password against a hash
func (sm *SecurityManager) VerifyPassword(password string, hash *PasswordHash) (bool, error) {
	return sm.passwordService.VerifyPassword(password, hash)
}

// EvaluateNetworkConnection evaluates a network connection against zero-trust policies
func (sm *SecurityManager) EvaluateNetworkConnection(ctx context.Context, conn *NetworkConnection) (NetworkPolicyAction, *NetworkPolicy, error) {
	if sm.ztNetworkService == nil {
		return NetworkPolicyAllow, nil, nil // Zero-trust not enabled
	}

	return sm.ztNetworkService.EvaluateConnection(ctx, conn)
}

// RunComplianceAssessment runs a compliance assessment
func (sm *SecurityManager) RunComplianceAssessment(ctx context.Context, framework ComplianceFramework, tenantID, assessorID string) (*ComplianceAssessment, error) {
	if sm.complianceService == nil {
		return nil, fmt.Errorf("compliance service not enabled")
	}

	assessment, err := sm.complianceService.CreateAssessment(framework, tenantID, assessorID)
	if err != nil {
		return nil, fmt.Errorf("failed to create assessment: %w", err)
	}

	err = sm.complianceService.RunAutomatedTests(ctx, assessment.ID)
	if err != nil {
		return nil, fmt.Errorf("failed to run automated tests: %w", err)
	}

	return assessment, nil
}

// GetSecurityMetrics returns security metrics for monitoring
func (sm *SecurityManager) GetSecurityMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})

	// JWT metrics
	metrics["jwt_enabled"] = true

	// Encryption metrics
	activeKeys := sm.encryptionService.GetActiveKeys()
	metrics["active_encryption_keys"] = len(activeKeys)

	// OAuth2 metrics
	metrics["oauth2_providers"] = len(sm.oauth2Services)
	oauth2Providers := make([]string, 0, len(sm.oauth2Services))
	for provider := range sm.oauth2Services {
		oauth2Providers = append(oauth2Providers, provider)
	}
	metrics["oauth2_provider_names"] = oauth2Providers

	// Zero-trust metrics
	metrics["zero_trust_enabled"] = sm.ztNetworkService != nil
	if sm.ztNetworkService != nil {
		policies := sm.ztNetworkService.GetPolicies()
		microsegments := sm.ztNetworkService.GetMicrosegments()
		connections := sm.ztNetworkService.GetConnections()
		metrics["network_policies"] = len(policies)
		metrics["microsegments"] = len(microsegments)
		metrics["active_connections"] = len(connections)
	}

	// Compliance metrics
	metrics["compliance_enabled"] = sm.complianceService != nil

	// Audit metrics
	metrics["audit_logging_enabled"] = sm.config.AuditLogging

	return metrics
}

// CleanupExpiredTokens performs cleanup of expired tokens and states
func (sm *SecurityManager) CleanupExpiredTokens() {
	// Cleanup OAuth2 states
	for _, oauth2Service := range sm.oauth2Services {
		oauth2Service.CleanupExpiredStates()
	}

	// Cleanup rate limiting entries
	sm.securityMiddleware.CleanupRateLimits()

	// Rotate encryption keys if needed
	sm.encryptionService.RotateKeys()

	// Cleanup old network connections
	if sm.ztNetworkService != nil {
		sm.ztNetworkService.CleanupExpiredConnections(24 * time.Hour)
	}
}

// SetupPeriodicTasks sets up periodic security maintenance tasks
func (sm *SecurityManager) SetupPeriodicTasks() {
	// Run cleanup every hour
	go func() {
		ticker := time.NewTicker(time.Hour)
		defer ticker.Stop()

		for range ticker.C {
			sm.CleanupExpiredTokens()
		}
	}()

	// Run security health checks every 15 minutes
	go func() {
		ticker := time.NewTicker(15 * time.Minute)
		defer ticker.Stop()

		for range ticker.C {
			sm.runSecurityHealthChecks()
		}
	}()
}

// runSecurityHealthChecks performs periodic security health checks
func (sm *SecurityManager) runSecurityHealthChecks() {
	// Check encryption key health
	activeKeys := sm.encryptionService.GetActiveKeys()
	if len(activeKeys) == 0 {
		log.Println("WARNING: No active encryption keys found")
	}

	// Check for keys nearing expiration
	now := time.Now()
	for _, key := range activeKeys {
		if key.ExpiresAt.Sub(now) < 7*24*time.Hour {
			log.Printf("WARNING: Encryption key %s expires in %v", key.ID, key.ExpiresAt.Sub(now))
		}
		if key.UsageCount > key.MaxUsage*80/100 {
			log.Printf("WARNING: Encryption key %s usage at %d/%d (80%% threshold)", key.ID, key.UsageCount, key.MaxUsage)
		}
	}

	// Check OAuth2 service health
	for provider, service := range sm.oauth2Services {
		if len(service.states) > 1000 {
			log.Printf("WARNING: OAuth2 provider %s has %d pending states", provider, len(service.states))
		}
	}

	// Log health check completion
	if sm.auditService != nil {
		sm.auditService.LogAccess(&AuditEntry{
			ResourceType: "security_system",
			ResourceID:   "health_check",
			Action:       "health_check_completed",
			Success:      true,
			Timestamp:    now,
			AdditionalData: map[string]interface{}{
				"active_keys":      len(activeKeys),
				"oauth2_providers": len(sm.oauth2Services),
			},
		})
	}
}

// setupDefaultNetworkPolicies creates default zero-trust network policies
func setupDefaultNetworkPolicies(ztService *ZeroTrustNetworkService) error {
	// Default deny-all policy (lowest priority)
	denyAllPolicy := &NetworkPolicy{
		ID:          "default-deny-all",
		Name:        "Default Deny All",
		Description: "Default policy to deny all traffic unless explicitly allowed",
		Enabled:     true,
		Priority:    1, // Lowest priority
		Source:      NetworkPolicySelector{Any: true},
		Destination: NetworkPolicySelector{Any: true},
		Action:      NetworkPolicyDeny,
		Protocols:   []NetworkProtocol{},
	}

	if err := ztService.CreatePolicy(denyAllPolicy); err != nil {
		return fmt.Errorf("failed to create deny-all policy: %w", err)
	}

	// Allow internal service communication
	allowInternalPolicy := &NetworkPolicy{
		ID:          "allow-internal-services",
		Name:        "Allow Internal Service Communication",
		Description: "Allow communication between internal services",
		Enabled:     true,
		Priority:    50,
		Source: NetworkPolicySelector{
			IPRanges: []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"},
		},
		Destination: NetworkPolicySelector{
			IPRanges: []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"},
		},
		Action: NetworkPolicyAllow,
		Protocols: []NetworkProtocol{
			{Protocol: "TCP", Ports: []int{80, 443, 8080, 8090, 8091}},
			{Protocol: "UDP", Ports: []int{53}}, // DNS
		},
	}

	if err := ztService.CreatePolicy(allowInternalPolicy); err != nil {
		return fmt.Errorf("failed to create internal services policy: %w", err)
	}

	// Require mTLS for sensitive services
	mtlsPolicy := &NetworkPolicy{
		ID:          "require-mtls-sensitive",
		Name:        "Require mTLS for Sensitive Services",
		Description: "Require mutual TLS for sensitive service communication",
		Enabled:     true,
		Priority:    100,
		Source:      NetworkPolicySelector{Any: true},
		Destination: NetworkPolicySelector{
			Services: []string{"database", "auth-service", "key-management"},
		},
		Action: NetworkPolicyRequireMTLS,
		Protocols: []NetworkProtocol{
			{Protocol: "TCP", Ports: []int{5432, 3306, 6379}}, // Database ports
		},
	}

	if err := ztService.CreatePolicy(mtlsPolicy); err != nil {
		return fmt.Errorf("failed to create mTLS policy: %w", err)
	}

	return nil
}

// DefaultSecurityConfiguration returns a secure default configuration
func DefaultSecurityConfiguration() (SecurityConfiguration, error) {
	// Generate JWT keys
	jwtConfig, err := DefaultJWTConfiguration()
	if err != nil {
		return SecurityConfiguration{}, fmt.Errorf("failed to generate JWT config: %w", err)
	}

	// OAuth2 providers
	oauth2Configs := GetProviderConfigs()

	return SecurityConfiguration{
		JWT:                jwtConfig,
		PasswordSecurity:   DefaultPasswordSecurityConfig(),
		Encryption:         DefaultEncryptionConfig(),
		SecurityMiddleware: DefaultSecurityConfig(),
		OAuth2:             oauth2Configs,
		Compliance:         true,
		ZeroTrust:          true,
		AuditLogging:       true,
	}, nil
}

// SecureHTTPHandler wraps an HTTP handler with comprehensive security
func (sm *SecurityManager) SecureHTTPHandler(handler http.Handler) http.Handler {
	return sm.GetSecurityMiddleware()(handler)
}

// ExtractUserFromContext extracts user information from security context
func (sm *SecurityManager) ExtractUserFromContext(ctx context.Context) (*User, error) {
	secCtx, exists := GetSecurityContext(ctx)
	if !exists || secCtx.UserID == "" {
		return nil, fmt.Errorf("no authenticated user in context")
	}

	return sm.authService.(*AuthServiceImpl).users.Get(secCtx.UserID)
}

// RequirePermission creates a middleware that requires specific permissions
func (sm *SecurityManager) RequirePermission(resource, action string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			secCtx, exists := GetSecurityContext(r.Context())
			if !exists || secCtx.UserID == "" {
				http.Error(w, "Authentication required", http.StatusUnauthorized)
				return
			}

			// Check if user has required permission
			permissionStr := fmt.Sprintf("%s:%s", resource, action)
			for _, perm := range secCtx.Permissions {
				if perm == permissionStr || perm == "*:*" {
					next.ServeHTTP(w, r)
					return
				}
			}

			// Check with auth service as fallback
			hasPermission, err := sm.authService.HasPermission(secCtx.UserID, resource, action)
			if err != nil || !hasPermission {
				sm.auditService.LogAccess(&AuditEntry{
					UserID:       secCtx.UserID,
					TenantID:     secCtx.TenantID,
					ResourceType: resource,
					ResourceID:   r.URL.Path,
					Action:       "permission_denied",
					Success:      false,
					Reason:       fmt.Sprintf("Missing permission: %s:%s", resource, action),
					Timestamp:    time.Now(),
					IPAddress:    secCtx.ClientIP,
				})
				http.Error(w, "Insufficient permissions", http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// RequireRole creates a middleware that requires specific roles
func (sm *SecurityManager) RequireRole(requiredRoles ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			secCtx, exists := GetSecurityContext(r.Context())
			if !exists || secCtx.UserID == "" {
				http.Error(w, "Authentication required", http.StatusUnauthorized)
				return
			}

			// Get user roles
			userRoles, err := sm.authService.GetUserRoles(secCtx.UserID)
			if err != nil {
				http.Error(w, "Failed to get user roles", http.StatusInternalServerError)
				return
			}

			// Check if user has any of the required roles
			for _, userRole := range userRoles {
				for _, requiredRole := range requiredRoles {
					if userRole.Name == requiredRole || userRole.Name == "admin" {
						next.ServeHTTP(w, r)
						return
					}
				}
			}

			sm.auditService.LogAccess(&AuditEntry{
				UserID:       secCtx.UserID,
				TenantID:     secCtx.TenantID,
				ResourceType: "role",
				ResourceID:   strings.Join(requiredRoles, ","),
				Action:       "role_denied",
				Success:      false,
				Reason:       fmt.Sprintf("Missing required roles: %v", requiredRoles),
				Timestamp:    time.Now(),
				IPAddress:    secCtx.ClientIP,
			})

			http.Error(w, "Insufficient role privileges", http.StatusForbidden)
		})
	}
}

// LogSecurityEvent logs a security event through the audit service
func (sm *SecurityManager) LogSecurityEvent(userID, tenantID, resourceType, resourceID, action string, success bool, details map[string]interface{}) {
	if sm.auditService == nil {
		return
	}

	entry := &AuditEntry{
		UserID:         userID,
		TenantID:       tenantID,
		ResourceType:   resourceType,
		ResourceID:     resourceID,
		Action:         action,
		Success:        success,
		Timestamp:      time.Now(),
		AdditionalData: details,
	}

	sm.auditService.LogAccess(entry)
}
