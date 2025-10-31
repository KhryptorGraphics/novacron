package security

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// RBACManager manages Role-Based Access Control
type RBACManager struct {
	mu          sync.RWMutex
	users       map[string]*User
	roles       map[string]*Role
	permissions map[string]*Permission
	auditLog    *AuditLogger
	config      *RBACConfig
}

// User represents a system user
type User struct {
	ID          string
	Username    string
	Email       string
	Roles       []string
	Enabled     bool
	MFAEnabled  bool
	LastLogin   time.Time
	CreatedAt   time.Time
	UpdatedAt   time.Time
	Metadata    map[string]string
}

// Role represents a role with permissions
type Role struct {
	ID          string
	Name        string
	Description string
	Permissions []string
	Inherits    []string // Role inheritance
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// Permission represents a permission
type Permission struct {
	ID          string
	Resource    string // e.g., "vm", "network", "storage"
	Action      string // e.g., "create", "read", "update", "delete"
	Scope       string // e.g., "global", "tenant", "user"
	Conditions  map[string]interface{}
	CreatedAt   time.Time
}

// RBACConfig configuration for RBAC
type RBACConfig struct {
	EnableAuditLog     bool
	EnableMFA          bool
	SessionTimeout     time.Duration
	MaxLoginAttempts   int
	PasswordMinLength  int
	RequireStrongPassword bool
}

// NewRBACManager creates a new RBAC manager
func NewRBACManager(config *RBACConfig) *RBACManager {
	rbac := &RBACManager{
		users:       make(map[string]*User),
		roles:       make(map[string]*Role),
		permissions: make(map[string]*Permission),
		config:      config,
	}
	
	if config.EnableAuditLog {
		rbac.auditLog = NewAuditLogger()
	}
	
	// Initialize default roles
	rbac.initializeDefaultRoles()
	
	return rbac
}

// initializeDefaultRoles creates default system roles
func (rbac *RBACManager) initializeDefaultRoles() {
	// Admin role - full access
	adminRole := &Role{
		ID:          "role-admin",
		Name:        "admin",
		Description: "Full system access",
		Permissions: []string{"*:*:*"}, // All permissions
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}
	rbac.roles[adminRole.ID] = adminRole
	
	// Operator role - manage VMs and resources
	operatorRole := &Role{
		ID:          "role-operator",
		Name:        "operator",
		Description: "Manage VMs and resources",
		Permissions: []string{
			"vm:create:tenant",
			"vm:read:tenant",
			"vm:update:tenant",
			"vm:delete:tenant",
			"network:read:tenant",
			"storage:read:tenant",
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	rbac.roles[operatorRole.ID] = operatorRole
	
	// Viewer role - read-only access
	viewerRole := &Role{
		ID:          "role-viewer",
		Name:        "viewer",
		Description: "Read-only access",
		Permissions: []string{
			"vm:read:tenant",
			"network:read:tenant",
			"storage:read:tenant",
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	rbac.roles[viewerRole.ID] = viewerRole
}

// CreateUser creates a new user
func (rbac *RBACManager) CreateUser(ctx context.Context, user *User) error {
	rbac.mu.Lock()
	defer rbac.mu.Unlock()
	
	if _, exists := rbac.users[user.ID]; exists {
		return fmt.Errorf("user %s already exists", user.ID)
	}
	
	user.CreatedAt = time.Now()
	user.UpdatedAt = time.Now()
	user.Enabled = true
	
	rbac.users[user.ID] = user
	
	// Audit log
	if rbac.auditLog != nil {
		rbac.auditLog.Log(ctx, &AuditEvent{
			Action:    "user.create",
			UserID:    user.ID,
			Resource:  "user",
			Timestamp: time.Now(),
			Success:   true,
		})
	}
	
	return nil
}

// AssignRole assigns a role to a user
func (rbac *RBACManager) AssignRole(ctx context.Context, userID, roleID string) error {
	rbac.mu.Lock()
	defer rbac.mu.Unlock()
	
	user, exists := rbac.users[userID]
	if !exists {
		return fmt.Errorf("user %s not found", userID)
	}
	
	if _, exists := rbac.roles[roleID]; !exists {
		return fmt.Errorf("role %s not found", roleID)
	}
	
	// Check if already assigned
	for _, r := range user.Roles {
		if r == roleID {
			return fmt.Errorf("role already assigned")
		}
	}
	
	user.Roles = append(user.Roles, roleID)
	user.UpdatedAt = time.Now()
	
	// Audit log
	if rbac.auditLog != nil {
		rbac.auditLog.Log(ctx, &AuditEvent{
			Action:    "role.assign",
			UserID:    userID,
			Resource:  "role",
			ResourceID: roleID,
			Timestamp: time.Now(),
			Success:   true,
		})
	}
	
	return nil
}

// CheckPermission checks if a user has permission for an action
func (rbac *RBACManager) CheckPermission(ctx context.Context, userID, resource, action, scope string) (bool, error) {
	rbac.mu.RLock()
	defer rbac.mu.RUnlock()

	user, exists := rbac.users[userID]
	if !exists {
		return false, fmt.Errorf("user %s not found", userID)
	}

	if !user.Enabled {
		return false, fmt.Errorf("user disabled")
	}

	// Check all user roles
	for _, roleID := range user.Roles {
		role, exists := rbac.roles[roleID]
		if !exists {
			continue
		}

		// Check role permissions
		if rbac.hasPermission(role, resource, action, scope) {
			// Audit log
			if rbac.auditLog != nil {
				rbac.auditLog.Log(ctx, &AuditEvent{
					Action:     fmt.Sprintf("%s.%s", resource, action),
					UserID:     userID,
					Resource:   resource,
					Timestamp:  time.Now(),
					Success:    true,
				})
			}
			return true, nil
		}
	}

	// Audit log - permission denied
	if rbac.auditLog != nil {
		rbac.auditLog.Log(ctx, &AuditEvent{
			Action:     fmt.Sprintf("%s.%s", resource, action),
			UserID:     userID,
			Resource:   resource,
			Timestamp:  time.Now(),
			Success:    false,
			Details:    map[string]interface{}{"reason": "permission denied"},
		})
	}

	return false, nil
}

// hasPermission checks if a role has a specific permission
func (rbac *RBACManager) hasPermission(role *Role, resource, action, scope string) bool {
	for _, perm := range role.Permissions {
		if rbac.matchesPermission(perm, resource, action, scope) {
			return true
		}
	}

	// Check inherited roles
	for _, inheritedRoleID := range role.Inherits {
		if inheritedRole, exists := rbac.roles[inheritedRoleID]; exists {
			if rbac.hasPermission(inheritedRole, resource, action, scope) {
				return true
			}
		}
	}

	return false
}

// matchesPermission checks if a permission string matches the request
func (rbac *RBACManager) matchesPermission(perm, resource, action, scope string) bool {
	// Permission format: "resource:action:scope"
	// Wildcards supported: "*:*:*", "vm:*:tenant", etc.

	// Parse permission
	var permResource, permAction, permScope string
	fmt.Sscanf(perm, "%s:%s:%s", &permResource, &permAction, &permScope)

	// Check wildcards
	if permResource == "*" || permResource == resource {
		if permAction == "*" || permAction == action {
			if permScope == "*" || permScope == scope {
				return true
			}
		}
	}

	return false
}

// AuditLogger logs security events
type AuditLogger struct {
	mu     sync.RWMutex
	events []*AuditEvent
}

// AuditEvent represents a security audit event
type AuditEvent struct {
	ID         string
	Action     string
	UserID     string
	Resource   string
	ResourceID string
	Timestamp  time.Time
	Success    bool
	IPAddress  string
	Details    map[string]interface{}
}

// NewAuditLogger creates a new audit logger
func NewAuditLogger() *AuditLogger {
	return &AuditLogger{
		events: make([]*AuditEvent, 0),
	}
}

// Log logs an audit event
func (al *AuditLogger) Log(ctx context.Context, event *AuditEvent) {
	al.mu.Lock()
	defer al.mu.Unlock()

	event.ID = fmt.Sprintf("audit-%d", time.Now().UnixNano())
	al.events = append(al.events, event)

	// In production, write to persistent storage
	// For now, keep in memory with size limit
	if len(al.events) > 10000 {
		al.events = al.events[1:]
	}
}

// GetEvents returns audit events
func (al *AuditLogger) GetEvents(ctx context.Context, filter *AuditFilter) []*AuditEvent {
	al.mu.RLock()
	defer al.mu.RUnlock()

	filtered := make([]*AuditEvent, 0)

	for _, event := range al.events {
		if filter.Matches(event) {
			filtered = append(filtered, event)
		}
	}

	return filtered
}

// AuditFilter filters audit events
type AuditFilter struct {
	UserID    string
	Resource  string
	Action    string
	StartTime time.Time
	EndTime   time.Time
	Success   *bool
}

// Matches checks if an event matches the filter
func (af *AuditFilter) Matches(event *AuditEvent) bool {
	if af.UserID != "" && event.UserID != af.UserID {
		return false
	}

	if af.Resource != "" && event.Resource != af.Resource {
		return false
	}

	if af.Action != "" && event.Action != af.Action {
		return false
	}

	if !af.StartTime.IsZero() && event.Timestamp.Before(af.StartTime) {
		return false
	}

	if !af.EndTime.IsZero() && event.Timestamp.After(af.EndTime) {
		return false
	}

	if af.Success != nil && event.Success != *af.Success {
		return false
	}

	return true
}

// EncryptionManager handles end-to-end encryption
type EncryptionManager struct {
	mu     sync.RWMutex
	keys   map[string][]byte
	config *EncryptionConfig
}

// EncryptionConfig configuration for encryption
type EncryptionConfig struct {
	Algorithm      string // AES-256-GCM
	KeyRotationInterval time.Duration
	EnableAtRest   bool
	EnableInTransit bool
}

// NewEncryptionManager creates a new encryption manager
func NewEncryptionManager(config *EncryptionConfig) *EncryptionManager {
	return &EncryptionManager{
		keys:   make(map[string][]byte),
		config: config,
	}
}

// Encrypt encrypts data
func (em *EncryptionManager) Encrypt(ctx context.Context, data []byte, keyID string) ([]byte, error) {
	// Implement AES-256-GCM encryption
	// This is a placeholder
	return data, nil
}

// Decrypt decrypts data
func (em *EncryptionManager) Decrypt(ctx context.Context, data []byte, keyID string) ([]byte, error) {
	// Implement AES-256-GCM decryption
	// This is a placeholder
	return data, nil
}


