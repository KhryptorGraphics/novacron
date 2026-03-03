package auth

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"strings"
	"time"
)

// Session represents a user session
type Session struct {
	// ID is the unique identifier for the session
	ID string `json:"id"`

	// UserID is the ID of the user associated with this session
	UserID string `json:"userId"`

	// TenantID is the ID of the tenant associated with this session
	TenantID string `json:"tenantId"`

	// Token is the session token used for authentication
	Token string `json:"token"`

	// ExpiresAt is the time when the session expires
	ExpiresAt time.Time `json:"expiresAt"`

	// CreatedAt is the time when the session was created
	CreatedAt time.Time `json:"createdAt"`

	// LastAccessedAt is the time when the session was last accessed
	LastAccessedAt time.Time `json:"lastAccessedAt"`

	// ClientIP is the IP address of the client
	ClientIP string `json:"clientIp,omitempty"`

	// UserAgent is the user agent of the client
	UserAgent string `json:"userAgent,omitempty"`

	// Metadata contains additional metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// AuthService provides authentication and authorization services
type AuthService interface {
	// Login authenticates a user and creates a session
	Login(username, password string) (*Session, error)

	// Logout invalidates a session
	Logout(sessionID string) error

	// ValidateSession validates a session
	ValidateSession(sessionID, token string) (*Session, error)

	// RefreshSession refreshes a session
	RefreshSession(sessionID, token string) (*Session, error)

	// HasPermission checks if a user has a specific permission
	HasPermission(userID, resource, action string) (bool, error)

	// HasPermissionInTenant checks if a user has a specific permission in a tenant
	HasPermissionInTenant(userID, tenantID, resource, action string) (bool, error)

	// GetUserRoles gets a user's roles
	GetUserRoles(userID string) ([]*Role, error)

	// CreateUser creates a new user
	CreateUser(user *User, password string) error

	// CreateRole creates a new role
	CreateRole(role *Role) error

	// CreateTenant creates a new tenant
	CreateTenant(tenant *Tenant) error
}

// AuthConfiguration contains configuration for the auth service
type AuthConfiguration struct {
	// SessionExpiryTime is the duration after which a session expires
	SessionExpiryTime time.Duration

	// TokenLength is the length of the session token
	TokenLength int

	// MaxSessionsPerUser is the maximum number of sessions per user
	MaxSessionsPerUser int

	// PasswordMinLength is the minimum length of a password
	PasswordMinLength int

	// RequirePasswordMixedCase requires passwords to have mixed case
	RequirePasswordMixedCase bool

	// RequirePasswordNumbers requires passwords to have numbers
	RequirePasswordNumbers bool

	// RequirePasswordSpecialChars requires passwords to have special characters
	RequirePasswordSpecialChars bool
}

// DefaultAuthConfiguration returns the default auth configuration
func DefaultAuthConfiguration() AuthConfiguration {
	return AuthConfiguration{
		SessionExpiryTime:           24 * time.Hour,
		TokenLength:                 32,
		MaxSessionsPerUser:          5,
		PasswordMinLength:           8,
		RequirePasswordMixedCase:    true,
		RequirePasswordNumbers:      true,
		RequirePasswordSpecialChars: true,
	}
}

// AuthServiceImpl implements AuthService
type AuthServiceImpl struct {
	config       AuthConfiguration
	users        UserService
	roles        RoleService
	tenants      TenantService
	auditLog     AuditLogService
	sessions     map[string]*Session // sessionID -> Session
	userSessions map[string][]string // userID -> []sessionID
}

// NewAuthService creates a new auth service
func NewAuthService(
	config AuthConfiguration,
	users UserService,
	roles RoleService,
	tenants TenantService,
	auditLog AuditLogService,
) *AuthServiceImpl {
	return &AuthServiceImpl{
		config:       config,
		users:        users,
		roles:        roles,
		tenants:      tenants,
		auditLog:     auditLog,
		sessions:     make(map[string]*Session),
		userSessions: make(map[string][]string),
	}
}

// Login authenticates a user and creates a session
func (s *AuthServiceImpl) Login(username, password string) (*Session, error) {
	// Find the user by username
	user, err := s.users.GetByUsername(username)
	if err != nil {
		s.auditLog.Log(AuditEntry{
			Action:      "login_failed",
			Resource:    "user",
			ResourceID:  username,
			Description: fmt.Sprintf("Login failed: %v", err),
		})
		return nil, fmt.Errorf("invalid username or password")
	}

	// Check if the user is active
	if user.Status != UserStatusActive {
		s.auditLog.Log(AuditEntry{
			Action:      "login_failed",
			Resource:    "user",
			ResourceID:  user.ID,
			Description: fmt.Sprintf("Login failed: user status is %s", user.Status),
		})
		return nil, fmt.Errorf("user is not active")
	}

	// Verify the password
	valid, err := s.users.VerifyPassword(user.ID, password)
	if err != nil || !valid {
		s.auditLog.Log(AuditEntry{
			Action:      "login_failed",
			Resource:    "user",
			ResourceID:  user.ID,
			Description: "Login failed: invalid password",
		})
		return nil, fmt.Errorf("invalid username or password")
	}

	// Check if tenant is active
	tenant, err := s.tenants.Get(user.TenantID)
	if err != nil {
		s.auditLog.Log(AuditEntry{
			Action:      "login_failed",
			Resource:    "user",
			ResourceID:  user.ID,
			Description: fmt.Sprintf("Login failed: tenant not found - %v", err),
		})
		return nil, fmt.Errorf("tenant not found")
	}
	if tenant.Status != TenantStatusActive {
		s.auditLog.Log(AuditEntry{
			Action:      "login_failed",
			Resource:    "user",
			ResourceID:  user.ID,
			Description: fmt.Sprintf("Login failed: tenant status is %s", tenant.Status),
		})
		return nil, fmt.Errorf("tenant is not active")
	}

	// Generate a session
	sessionID := fmt.Sprintf("session-%d", time.Now().UnixNano())
	token, err := s.generateToken(s.config.TokenLength)
	if err != nil {
		s.auditLog.Log(AuditEntry{
			Action:      "login_failed",
			Resource:    "user",
			ResourceID:  user.ID,
			Description: fmt.Sprintf("Login failed: failed to generate token - %v", err),
		})
		return nil, fmt.Errorf("failed to generate session token")
	}

	now := time.Now()
	session := &Session{
		ID:             sessionID,
		UserID:         user.ID,
		TenantID:       user.TenantID,
		Token:          token,
		ExpiresAt:      now.Add(s.config.SessionExpiryTime),
		CreatedAt:      now,
		LastAccessedAt: now,
		Metadata:       make(map[string]interface{}),
	}

	// Store the session
	s.sessions[sessionID] = session

	// Add to user sessions
	if _, exists := s.userSessions[user.ID]; !exists {
		s.userSessions[user.ID] = []string{}
	}
	s.userSessions[user.ID] = append(s.userSessions[user.ID], sessionID)

	// Limit the number of sessions per user
	if len(s.userSessions[user.ID]) > s.config.MaxSessionsPerUser {
		// Remove the oldest session
		oldestSessionID := s.userSessions[user.ID][0]
		delete(s.sessions, oldestSessionID)
		s.userSessions[user.ID] = s.userSessions[user.ID][1:]
	}

	// Update last login time
	user.LastLogin = now
	s.users.Update(user)

	s.auditLog.Log(AuditEntry{
		Action:      "login_success",
		Resource:    "user",
		ResourceID:  user.ID,
		Description: "Login successful",
	})

	return session, nil
}

// Logout invalidates a session
func (s *AuthServiceImpl) Logout(sessionID string) error {
	session, exists := s.sessions[sessionID]
	if !exists {
		return fmt.Errorf("session not found")
	}

	// Remove the session
	delete(s.sessions, sessionID)

	// Remove from user sessions
	if sessions, exists := s.userSessions[session.UserID]; exists {
		newSessions := []string{}
		for _, id := range sessions {
			if id != sessionID {
				newSessions = append(newSessions, id)
			}
		}
		s.userSessions[session.UserID] = newSessions
	}

	s.auditLog.Log(AuditEntry{
		Action:      "logout",
		Resource:    "user",
		ResourceID:  session.UserID,
		Description: "Logout successful",
	})

	return nil
}

// ValidateSession validates a session
func (s *AuthServiceImpl) ValidateSession(sessionID, token string) (*Session, error) {
	session, exists := s.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("session not found")
	}

	// Check if session is expired
	if time.Now().After(session.ExpiresAt) {
		// Remove the session
		delete(s.sessions, sessionID)
		// Remove from user sessions
		if sessions, exists := s.userSessions[session.UserID]; exists {
			newSessions := []string{}
			for _, id := range sessions {
				if id != sessionID {
					newSessions = append(newSessions, id)
				}
			}
			s.userSessions[session.UserID] = newSessions
		}
		return nil, fmt.Errorf("session expired")
	}

	// Validate token
	if session.Token != token {
		return nil, fmt.Errorf("invalid token")
	}

	// Update last accessed time
	session.LastAccessedAt = time.Now()

	return session, nil
}

// RefreshSession refreshes a session
func (s *AuthServiceImpl) RefreshSession(sessionID, token string) (*Session, error) {
	// Validate the session first
	session, err := s.ValidateSession(sessionID, token)
	if err != nil {
		return nil, err
	}

	// Update session expiry time
	session.ExpiresAt = time.Now().Add(s.config.SessionExpiryTime)

	s.auditLog.Log(AuditEntry{
		Action:      "session_refresh",
		Resource:    "user",
		ResourceID:  session.UserID,
		Description: "Session refreshed",
	})

	return session, nil
}

// HasPermission checks if a user has a specific permission
func (s *AuthServiceImpl) HasPermission(userID, resource, action string) (bool, error) {
	// Get the user
	user, err := s.users.Get(userID)
	if err != nil {
		return false, fmt.Errorf("user not found: %w", err)
	}

	// Use the user's tenant ID
	return s.HasPermissionInTenant(userID, user.TenantID, resource, action)
}

// HasPermissionInTenant checks if a user has a specific permission in a tenant
func (s *AuthServiceImpl) HasPermissionInTenant(userID, tenantID, resource, action string) (bool, error) {
	// Get the user's roles
	roles, err := s.users.GetRoles(userID)
	if err != nil {
		return false, fmt.Errorf("failed to get user roles: %w", err)
	}

	// Check each role for the permission
	for _, role := range roles {
		// Skip roles that don't belong to this tenant
		if role.TenantID != "" && role.TenantID != tenantID {
			continue
		}

		// Check if the role has the permission
		hasPermission, err := s.roles.HasPermission(role.ID, resource, action)
		if err != nil {
			continue
		}
		if hasPermission {
			return true, nil
		}
	}

	// No matching permission found
	return false, nil
}

// GetUserRoles gets a user's roles
func (s *AuthServiceImpl) GetUserRoles(userID string) ([]*Role, error) {
	return s.users.GetRoles(userID)
}

// CreateUser creates a new user
func (s *AuthServiceImpl) CreateUser(user *User, password string) error {
	// Validate password
	if err := s.validatePassword(password); err != nil {
		return err
	}

	// Check if tenant exists
	_, err := s.tenants.Get(user.TenantID)
	if err != nil {
		return fmt.Errorf("tenant not found: %w", err)
	}

	// Create the user
	err = s.users.Create(user, password)
	if err != nil {
		return err
	}

	s.auditLog.Log(AuditEntry{
		Action:      "user_create",
		Resource:    "user",
		ResourceID:  user.ID,
		Description: "User created",
	})

	return nil
}

// CreateRole creates a new role
func (s *AuthServiceImpl) CreateRole(role *Role) error {
	// Check if tenant exists (if role is tenant-specific)
	if role.TenantID != "" {
		_, err := s.tenants.Get(role.TenantID)
		if err != nil {
			return fmt.Errorf("tenant not found: %w", err)
		}
	}

	// Create the role
	err := s.roles.Create(role)
	if err != nil {
		return err
	}

	s.auditLog.Log(AuditEntry{
		Action:      "role_create",
		Resource:    "role",
		ResourceID:  role.ID,
		Description: "Role created",
	})

	return nil
}

// CreateTenant creates a new tenant
func (s *AuthServiceImpl) CreateTenant(tenant *Tenant) error {
	// Create the tenant
	err := s.tenants.Create(tenant)
	if err != nil {
		return err
	}

	s.auditLog.Log(AuditEntry{
		Action:      "tenant_create",
		Resource:    "tenant",
		ResourceID:  tenant.ID,
		Description: "Tenant created",
	})

	return nil
}

// generateToken generates a random token
func (s *AuthServiceImpl) generateToken(length int) (string, error) {
	b := make([]byte, length)
	_, err := rand.Read(b)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(b), nil
}

// validatePassword validates a password
func (s *AuthServiceImpl) validatePassword(password string) error {
	if len(password) < s.config.PasswordMinLength {
		return fmt.Errorf("password must be at least %d characters long", s.config.PasswordMinLength)
	}

	if s.config.RequirePasswordMixedCase && !hasMixedCase(password) {
		return fmt.Errorf("password must contain both uppercase and lowercase characters")
	}

	if s.config.RequirePasswordNumbers && !hasNumbers(password) {
		return fmt.Errorf("password must contain at least one number")
	}

	if s.config.RequirePasswordSpecialChars && !hasSpecialChars(password) {
		return fmt.Errorf("password must contain at least one special character")
	}

	return nil
}

// hasMixedCase checks if a string has mixed case
func hasMixedCase(s string) bool {
	hasUpper := false
	hasLower := false
	for _, r := range s {
		if r >= 'A' && r <= 'Z' {
			hasUpper = true
		} else if r >= 'a' && r <= 'z' {
			hasLower = true
		}
		if hasUpper && hasLower {
			return true
		}
	}
	return false
}

// hasNumbers checks if a string has numbers
func hasNumbers(s string) bool {
	for _, r := range s {
		if r >= '0' && r <= '9' {
			return true
		}
	}
	return false
}

// hasSpecialChars checks if a string has special characters
func hasSpecialChars(s string) bool {
	specialChars := "!@#$%^&*()_+-=[]{}|;':\",./<>?"
	for _, r := range s {
		if strings.ContainsRune(specialChars, r) {
			return true
		}
	}
	return false
}
