package auth

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"
)

// AuthorizationType defines the type of authorization check
type AuthorizationType string

const (
	// AuthorizationTypeCreate for creating resources
	AuthorizationTypeCreate AuthorizationType = "create"

	// AuthorizationTypeRead for reading resources
	AuthorizationTypeRead AuthorizationType = "read"

	// AuthorizationTypeUpdate for updating resources
	AuthorizationTypeUpdate AuthorizationType = "update"

	// AuthorizationTypeDelete for deleting resources
	AuthorizationTypeDelete AuthorizationType = "delete"

	// AuthorizationTypeExecute for executing operations
	AuthorizationTypeExecute AuthorizationType = "execute"

	// AuthorizationTypeAdmin for administrative operations
	AuthorizationTypeAdmin AuthorizationType = "admin"
)

// ResourceType defines the type of resource for authorization checks
type ResourceType string

const (
	// ResourceTypeVM for virtual machines
	ResourceTypeVM ResourceType = "vm"

	// ResourceTypeNode for physical nodes
	ResourceTypeNode ResourceType = "node"

	// ResourceTypeStorage for storage resources
	ResourceTypeStorage ResourceType = "storage"

	// ResourceTypeNetwork for network resources
	ResourceTypeNetwork ResourceType = "network"

	// ResourceTypeUser for user management
	ResourceTypeUser ResourceType = "user"

	// ResourceTypeRole for role management
	ResourceTypeRole ResourceType = "role"

	// ResourceTypeTenant for tenant management
	ResourceTypeTenant ResourceType = "tenant"

	// ResourceTypeSystem for system-level operations
	ResourceTypeSystem ResourceType = "system"
)

// AuthorizationRequest represents a request for authorization
type AuthorizationRequest struct {
	// UserID is the ID of the user requesting access
	UserID string

	// TenantID is the tenant context for the request
	TenantID string

	// ResourceType is the type of resource being accessed
	ResourceType ResourceType

	// ResourceID is the ID of the specific resource (optional)
	ResourceID string

	// Action is the type of action being performed
	Action AuthorizationType
}

// AuthorizationResult represents the result of an authorization check
type AuthorizationResult struct {
	// Authorized indicates if the request is authorized
	Authorized bool

	// Reason provides the reason for the decision (especially for denials)
	Reason string

	// Timestamp is when the decision was made
	Timestamp time.Time
}

// AuthManagerConfig contains configuration for the auth manager
type AuthManagerConfig struct {
	// EnableAuditLogging enables audit logging for authorization decisions
	EnableAuditLogging bool

	// DefaultTenantID is the default tenant ID to use if none is specified
	DefaultTenantID string

	// DefaultCacheTTL is the default TTL for cached authorization decisions
	DefaultCacheTTL time.Duration

	// RoleCacheSize is the maximum number of role definitions to cache
	RoleCacheSize int

	// PermissionCacheSize is the maximum number of permission decisions to cache
	PermissionCacheSize int
}

// DefaultAuthManagerConfig returns a default configuration
func DefaultAuthManagerConfig() AuthManagerConfig {
	return AuthManagerConfig{
		EnableAuditLogging:  true,
		DefaultTenantID:     "default",
		DefaultCacheTTL:     5 * time.Minute,
		RoleCacheSize:       1000,
		PermissionCacheSize: 10000,
	}
}

// AuthManager manages authentication and authorization
type AuthManager struct {
	config AuthManagerConfig

	// userService manages users
	userService UserService

	// roleService manages roles
	roleService RoleService

	// tenantService manages tenants
	tenantService TenantService

	// auditService records audit logs
	auditService AuditService

	// permissionCache caches authorization decisions
	permissionCache      map[string]*AuthorizationResult
	permissionCacheMutex sync.RWMutex

	// roleCache caches role definitions
	roleCache      map[string]*Role
	roleCacheMutex sync.RWMutex

	ctx    context.Context
	cancel context.CancelFunc
}

// NewAuthManager creates a new auth manager
func NewAuthManager(
	config AuthManagerConfig,
	userService UserService,
	roleService RoleService,
	tenantService TenantService,
	auditService AuditService,
) *AuthManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &AuthManager{
		config:          config,
		userService:     userService,
		roleService:     roleService,
		tenantService:   tenantService,
		auditService:    auditService,
		permissionCache: make(map[string]*AuthorizationResult),
		roleCache:       make(map[string]*Role),
		ctx:             ctx,
		cancel:          cancel,
	}
}

// Start starts the auth manager
func (m *AuthManager) Start() error {
	log.Println("Starting auth manager")

	// Start periodic cache cleanup
	go m.cacheCleaner()

	return nil
}

// Stop stops the auth manager
func (m *AuthManager) Stop() error {
	log.Println("Stopping auth manager")

	m.cancel()

	return nil
}

// cacheCleaner periodically cleans up expired cache entries
func (m *AuthManager) cacheCleaner() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.cleanCaches()
		}
	}
}

// cleanCaches removes expired entries from caches
func (m *AuthManager) cleanCaches() {
	// Clean permission cache
	m.permissionCacheMutex.Lock()
	cutoff := time.Now().Add(-m.config.DefaultCacheTTL)
	for key, result := range m.permissionCache {
		if result.Timestamp.Before(cutoff) {
			delete(m.permissionCache, key)
		}
	}
	m.permissionCacheMutex.Unlock()

	// Role cache doesn't expire, but we could implement cache size management
	// to remove least recently used entries if needed
}

// Authorize checks if a request is authorized
func (m *AuthManager) Authorize(req AuthorizationRequest) (*AuthorizationResult, error) {
	// Check the cache first
	cacheKey := fmt.Sprintf("%s|%s|%s|%s|%s",
		req.UserID, req.TenantID, req.ResourceType, req.ResourceID, req.Action)

	m.permissionCacheMutex.RLock()
	cachedResult, exists := m.permissionCache[cacheKey]
	m.permissionCacheMutex.RUnlock()

	if exists {
		if time.Since(cachedResult.Timestamp) < m.config.DefaultCacheTTL {
			// Cache hit and not expired
			return cachedResult, nil
		}
	}

	// Get the user
	user, err := m.userService.Get(req.UserID)
	if err != nil {
		return &AuthorizationResult{
			Authorized: false,
			Reason:     fmt.Sprintf("User not found: %v", err),
			Timestamp:  time.Now(),
		}, nil
	}

	// Check if the user is active
	if !user.IsActive {
		result := &AuthorizationResult{
			Authorized: false,
			Reason:     "User account is not active",
			Timestamp:  time.Now(),
		}
		m.cacheResult(cacheKey, result)
		m.auditAccess(req, result)
		return result, nil
	}

	// Get the tenant if specified
	tenantID := req.TenantID
	if tenantID == "" {
		tenantID = m.config.DefaultTenantID
	}

	// Check if the user belongs to the tenant
	if !isUserInTenant(user, tenantID) && tenantID != "" && !user.IsSystem {
		result := &AuthorizationResult{
			Authorized: false,
			Reason:     "User does not belong to the specified tenant",
			Timestamp:  time.Now(),
		}
		m.cacheResult(cacheKey, result)
		m.auditAccess(req, result)
		return result, nil
	}

	// For system users, authorize everything
	if user.IsSystem {
		result := &AuthorizationResult{
			Authorized: true,
			Reason:     "System user has full access",
			Timestamp:  time.Now(),
		}
		m.cacheResult(cacheKey, result)
		m.auditAccess(req, result)
		return result, nil
	}

	// Check user roles for the required permissions
	authorized := false
	reason := "No matching permissions found in user roles"

	for _, roleID := range user.Roles {
		// Get the role
		role, err := m.getRole(roleID)
		if err != nil {
			log.Printf("Warning: Could not get role %s: %v", roleID, err)
			continue
		}

		// Skip roles for other tenants
		if role.TenantID != "" && role.TenantID != tenantID {
			continue
		}

		// Check if the role has the required permission
		hasPermission, err := m.roleService.HasPermission(roleID, string(req.ResourceType), string(req.Action))
		if err != nil {
			log.Printf("Warning: Error checking permission for role %s: %v", roleID, err)
			continue
		}

		if hasPermission {
			authorized = true
			reason = fmt.Sprintf("Permission granted by role: %s", role.Name)
			break
		}
	}

	// Create the result
	result := &AuthorizationResult{
		Authorized: authorized,
		Reason:     reason,
		Timestamp:  time.Now(),
	}

	// Cache the result
	m.cacheResult(cacheKey, result)

	// Audit the access
	m.auditAccess(req, result)

	return result, nil
}

// AuthorizeContext is a convenience method that adds authorization to a context
func (m *AuthManager) AuthorizeContext(ctx context.Context, req AuthorizationRequest) (context.Context, error) {
	result, err := m.Authorize(req)
	if err != nil {
		return ctx, err
	}

	if !result.Authorized {
		return ctx, errors.New(result.Reason)
	}

	// Add authorization info to the context
	ctx = context.WithValue(ctx, "auth_user_id", req.UserID)
	ctx = context.WithValue(ctx, "auth_tenant_id", req.TenantID)
	ctx = context.WithValue(ctx, "auth_timestamp", result.Timestamp)

	return ctx, nil
}

// GetUserPermissions gets all permissions for a user
func (m *AuthManager) GetUserPermissions(userID, tenantID string) (map[ResourceType][]AuthorizationType, error) {
	// Get the user
	user, err := m.userService.Get(userID)
	if err != nil {
		return nil, fmt.Errorf("user not found: %w", err)
	}

	// Check if the user belongs to the tenant
	if !isUserInTenant(user, tenantID) && tenantID != "" && !user.IsSystem {
		return nil, fmt.Errorf("user does not belong to the specified tenant")
	}

	// For system users, return all permissions
	if user.IsSystem {
		return map[ResourceType][]AuthorizationType{
			ResourceTypeVM:      {AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate, AuthorizationTypeDelete, AuthorizationTypeExecute},
			ResourceTypeNode:    {AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate, AuthorizationTypeDelete, AuthorizationTypeExecute},
			ResourceTypeStorage: {AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate, AuthorizationTypeDelete, AuthorizationTypeExecute},
			ResourceTypeNetwork: {AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate, AuthorizationTypeDelete, AuthorizationTypeExecute},
			ResourceTypeUser:    {AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate, AuthorizationTypeDelete, AuthorizationTypeAdmin},
			ResourceTypeRole:    {AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate, AuthorizationTypeDelete, AuthorizationTypeAdmin},
			ResourceTypeTenant:  {AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate, AuthorizationTypeDelete, AuthorizationTypeAdmin},
			ResourceTypeSystem:  {AuthorizationTypeRead, AuthorizationTypeUpdate, AuthorizationTypeExecute, AuthorizationTypeAdmin},
		}, nil
	}

	// Collect all permissions from user roles
	permissions := make(map[ResourceType][]AuthorizationType)

	for _, roleID := range user.Roles {
		// Get the role
		role, err := m.getRole(roleID)
		if err != nil {
			log.Printf("Warning: Could not get role %s: %v", roleID, err)
			continue
		}

		// Skip roles for other tenants
		if role.TenantID != "" && role.TenantID != tenantID {
			continue
		}

		// Add permissions from this role
		for _, permission := range role.Permissions {
			resourceType := ResourceType(permission.Resource)
			actionType := AuthorizationType(permission.Action)

			// Handle wildcards
			if permission.Resource == "*" {
				// All resource types
				for _, rt := range []ResourceType{
					ResourceTypeVM, ResourceTypeNode, ResourceTypeStorage,
					ResourceTypeNetwork, ResourceTypeUser, ResourceTypeRole,
					ResourceTypeTenant, ResourceTypeSystem,
				} {
					if permission.Action == "*" {
						// All actions
						addPermissions(permissions, rt, []AuthorizationType{
							AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate,
							AuthorizationTypeDelete, AuthorizationTypeExecute, AuthorizationTypeAdmin,
						})
					} else {
						// Specific action
						addPermissions(permissions, rt, []AuthorizationType{actionType})
					}
				}
			} else if permission.Action == "*" {
				// All actions for a specific resource
				addPermissions(permissions, resourceType, []AuthorizationType{
					AuthorizationTypeCreate, AuthorizationTypeRead, AuthorizationTypeUpdate,
					AuthorizationTypeDelete, AuthorizationTypeExecute, AuthorizationTypeAdmin,
				})
			} else {
				// Specific resource and action
				addPermissions(permissions, resourceType, []AuthorizationType{actionType})
			}
		}
	}

	return permissions, nil
}

// getRole gets a role, using cache if available
func (m *AuthManager) getRole(roleID string) (*Role, error) {
	// Check the cache first
	m.roleCacheMutex.RLock()
	cachedRole, exists := m.roleCache[roleID]
	m.roleCacheMutex.RUnlock()

	if exists {
		return cachedRole, nil
	}

	// Get from service
	role, err := m.roleService.Get(roleID)
	if err != nil {
		return nil, err
	}

	// Cache the role
	m.roleCacheMutex.Lock()
	m.roleCache[roleID] = role
	m.roleCacheMutex.Unlock()

	return role, nil
}

// cacheResult caches an authorization result
func (m *AuthManager) cacheResult(key string, result *AuthorizationResult) {
	m.permissionCacheMutex.Lock()
	defer m.permissionCacheMutex.Unlock()

	// Check if we need to manage cache size
	if len(m.permissionCache) >= m.config.PermissionCacheSize {
		// Very simple strategy: just clear the entire cache
		// A more sophisticated implementation would use LRU
		m.permissionCache = make(map[string]*AuthorizationResult)
	}

	m.permissionCache[key] = result
}

// auditAccess records an audit log entry for an access attempt
func (m *AuthManager) auditAccess(req AuthorizationRequest, result *AuthorizationResult) {
	if !m.config.EnableAuditLogging || m.auditService == nil {
		return
	}

	entry := &AuditEntry{
		UserID:       req.UserID,
		TenantID:     req.TenantID,
		ResourceType: string(req.ResourceType),
		ResourceID:   req.ResourceID,
		Action:       string(req.Action),
		Success:      result.Authorized,
		Reason:       result.Reason,
		Timestamp:    result.Timestamp,
		IPAddress:    "", // Would be populated in a real implementation
		UserAgent:    "", // Would be populated in a real implementation
	}

	// Use non-blocking call to avoid impacting performance
	go func() {
		err := m.auditService.LogAccess(entry)
		if err != nil {
			log.Printf("Warning: Failed to log audit entry: %v", err)
		}
	}()
}

// isUserInTenant checks if a user belongs to a tenant
func isUserInTenant(user *User, tenantID string) bool {
	if tenantID == "" {
		return true // Empty tenant ID matches everything
	}

	for _, userTenant := range user.Tenants {
		if userTenant == tenantID {
			return true
		}
	}

	return false
}

// addPermissions adds permissions to the map if they don't already exist
func addPermissions(permissions map[ResourceType][]AuthorizationType, resourceType ResourceType, actions []AuthorizationType) {
	existing, exists := permissions[resourceType]
	if !exists {
		permissions[resourceType] = actions
		return
	}

	// Check for each action if it already exists
	existingMap := make(map[AuthorizationType]bool)
	for _, action := range existing {
		existingMap[action] = true
	}

	// Add actions that don't already exist
	for _, action := range actions {
		if !existingMap[action] {
			permissions[resourceType] = append(permissions[resourceType], action)
			existingMap[action] = true
		}
	}
}

// GetContextUserID extracts the user ID from a context
func GetContextUserID(ctx context.Context) (string, bool) {
	userID, ok := ctx.Value("auth_user_id").(string)
	return userID, ok
}

// GetContextTenantID extracts the tenant ID from a context
func GetContextTenantID(ctx context.Context) (string, bool) {
	tenantID, ok := ctx.Value("auth_tenant_id").(string)
	return tenantID, ok
}
