package security

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// RBACEngine provides comprehensive Role-Based Access Control
type RBACEngine struct {
	roles       map[string]*Role
	policies    map[string]*Policy
	users       map[string]*UserRBAC
	permissions map[string]*Permission
	cache       *PermissionCache
	mu          sync.RWMutex
}

// Role represents a role in the RBAC system
type Role struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Permissions []string          `json:"permissions"`
	ParentRoles []string          `json:"parent_roles"`
	TenantID    string            `json:"tenant_id,omitempty"`
	IsSystem    bool              `json:"is_system"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Permission represents a permission in the RBAC system
type Permission struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Resource    string            `json:"resource"`
	Action      string            `json:"action"`
	Effect      string            `json:"effect"` // allow, deny
	Conditions  []string          `json:"conditions,omitempty"`
	CreatedAt   time.Time         `json:"created_at"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Policy represents an access policy
type Policy struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Version     int                    `json:"version"`
	Rules       []PolicyRule           `json:"rules"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	IsActive    bool                   `json:"is_active"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// PolicyRule represents a single rule within a policy
type PolicyRule struct {
	ID         string                 `json:"id"`
	Effect     string                 `json:"effect"`     // allow, deny
	Principals []string               `json:"principals"` // users, roles, groups
	Resources  []string               `json:"resources"`  // resource patterns
	Actions    []string               `json:"actions"`    // action patterns
	Conditions []PolicyCondition      `json:"conditions,omitempty"`
	Priority   int                    `json:"priority"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// PolicyCondition represents a condition for policy evaluation
type PolicyCondition struct {
	Key      string      `json:"key"`
	Operator string      `json:"operator"` // eq, ne, in, not_in, gt, lt, contains
	Value    interface{} `json:"value"`
}

// UserRBAC represents user RBAC information
type UserRBAC struct {
	ID          string            `json:"id"`
	Username    string            `json:"username"`
	Roles       []string          `json:"roles"`
	DirectPerms []string          `json:"direct_permissions"`
	TenantID    string            `json:"tenant_id"`
	Groups      []string          `json:"groups,omitempty"`
	Attributes  map[string]string `json:"attributes,omitempty"`
	IsActive    bool              `json:"is_active"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// PermissionCache provides efficient permission caching
type PermissionCache struct {
	cache      map[string]*CacheEntry
	mu         sync.RWMutex
	ttl        time.Duration
	maxEntries int
}

// CacheEntry represents a cached permission decision
type CacheEntry struct {
	UserID      string    `json:"user_id"`
	Resource    string    `json:"resource"`
	Action      string    `json:"action"`
	Decision    bool      `json:"decision"`
	Reason      string    `json:"reason,omitempty"`
	ExpiresAt   time.Time `json:"expires_at"`
	CreatedAt   time.Time `json:"created_at"`
}

// AuthorizationRequest represents an authorization request
type AuthorizationRequest struct {
	UserID     string                 `json:"user_id"`
	Resource   string                 `json:"resource"`
	Action     string                 `json:"action"`
	Context    map[string]interface{} `json:"context,omitempty"`
	TenantID   string                 `json:"tenant_id,omitempty"`
	SessionID  string                 `json:"session_id,omitempty"`
	Attributes map[string]string      `json:"attributes,omitempty"`
}

// AuthorizationResponse represents an authorization response
type AuthorizationResponse struct {
	Decision   bool                   `json:"decision"`
	Reason     string                 `json:"reason"`
	Policies   []string               `json:"policies_applied,omitempty"`
	Obligations []string              `json:"obligations,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	Cached     bool                   `json:"cached"`
	Timestamp  time.Time              `json:"timestamp"`
}

// NewRBACEngine creates a new RBAC engine
func NewRBACEngine() (*RBACEngine, error) {
	cache := &PermissionCache{
		cache:      make(map[string]*CacheEntry),
		ttl:        15 * time.Minute,
		maxEntries: 10000,
	}

	engine := &RBACEngine{
		roles:       make(map[string]*Role),
		policies:    make(map[string]*Policy),
		users:       make(map[string]*UserRBAC),
		permissions: make(map[string]*Permission),
		cache:       cache,
	}

	// Initialize with default system roles and permissions
	if err := engine.initializeSystemRoles(); err != nil {
		return nil, fmt.Errorf("failed to initialize system roles: %w", err)
	}

	// Start cache cleanup goroutine
	go engine.cleanupCache()

	return engine, nil
}

// HasPermission checks if a user has a specific permission
func (rbac *RBACEngine) HasPermission(userID, resource, action string) bool {
	request := &AuthorizationRequest{
		UserID:   userID,
		Resource: resource,
		Action:   action,
	}
	
	response := rbac.Authorize(request)
	return response.Decision
}

// Authorize performs comprehensive authorization check
func (rbac *RBACEngine) Authorize(request *AuthorizationRequest) *AuthorizationResponse {
	// Check cache first
	cacheKey := rbac.generateCacheKey(request.UserID, request.Resource, request.Action)
	if entry := rbac.cache.get(cacheKey); entry != nil {
		return &AuthorizationResponse{
			Decision:  entry.Decision,
			Reason:    entry.Reason,
			Cached:    true,
			Timestamp: time.Now(),
		}
	}

	rbac.mu.RLock()
	defer rbac.mu.RUnlock()

	// Get user RBAC info
	user, exists := rbac.users[request.UserID]
	if !exists || !user.IsActive {
		response := &AuthorizationResponse{
			Decision:  false,
			Reason:    "user not found or inactive",
			Timestamp: time.Now(),
		}
		rbac.cache.set(cacheKey, &CacheEntry{
			UserID:    request.UserID,
			Resource:  request.Resource,
			Action:    request.Action,
			Decision:  false,
			Reason:    response.Reason,
			ExpiresAt: time.Now().Add(rbac.cache.ttl),
			CreatedAt: time.Now(),
		})
		return response
	}

	// Check direct permissions first
	if rbac.hasDirectPermission(user, request.Resource, request.Action) {
		response := &AuthorizationResponse{
			Decision:  true,
			Reason:    "direct permission granted",
			Timestamp: time.Now(),
		}
		rbac.cache.set(cacheKey, &CacheEntry{
			UserID:    request.UserID,
			Resource:  request.Resource,
			Action:    request.Action,
			Decision:  true,
			Reason:    response.Reason,
			ExpiresAt: time.Now().Add(rbac.cache.ttl),
			CreatedAt: time.Now(),
		})
		return response
	}

	// Check role-based permissions
	if rbac.hasRolePermission(user, request.Resource, request.Action) {
		response := &AuthorizationResponse{
			Decision:  true,
			Reason:    "role-based permission granted",
			Timestamp: time.Now(),
		}
		rbac.cache.set(cacheKey, &CacheEntry{
			UserID:    request.UserID,
			Resource:  request.Resource,
			Action:    request.Action,
			Decision:  true,
			Reason:    response.Reason,
			ExpiresAt: time.Now().Add(rbac.cache.ttl),
			CreatedAt: time.Now(),
		})
		return response
	}

	// Check policy-based permissions
	if decision, reason, policies := rbac.evaluatePolicies(user, request); decision {
		response := &AuthorizationResponse{
			Decision:  true,
			Reason:    reason,
			Policies:  policies,
			Timestamp: time.Now(),
		}
		rbac.cache.set(cacheKey, &CacheEntry{
			UserID:    request.UserID,
			Resource:  request.Resource,
			Action:    request.Action,
			Decision:  true,
			Reason:    response.Reason,
			ExpiresAt: time.Now().Add(rbac.cache.ttl),
			CreatedAt: time.Now(),
		})
		return response
	}

	// Default deny
	response := &AuthorizationResponse{
		Decision:  false,
		Reason:    "no matching permissions found",
		Timestamp: time.Now(),
	}
	rbac.cache.set(cacheKey, &CacheEntry{
		UserID:    request.UserID,
		Resource:  request.Resource,
		Action:    request.Action,
		Decision:  false,
		Reason:    response.Reason,
		ExpiresAt: time.Now().Add(rbac.cache.ttl),
		CreatedAt: time.Now(),
	})
	return response
}

// hasDirectPermission checks if user has direct permission
func (rbac *RBACEngine) hasDirectPermission(user *UserRBAC, resource, action string) bool {
	for _, permID := range user.DirectPerms {
		if perm, exists := rbac.permissions[permID]; exists {
			if rbac.matchesPermission(perm, resource, action) {
				return true
			}
		}
	}
	return false
}

// hasRolePermission checks if user has permission through roles
func (rbac *RBACEngine) hasRolePermission(user *UserRBAC, resource, action string) bool {
	// Get all permissions from user roles (including inherited roles)
	allPermissions := rbac.getAllUserPermissions(user)
	
	for permID := range allPermissions {
		if perm, exists := rbac.permissions[permID]; exists {
			if rbac.matchesPermission(perm, resource, action) {
				return true
			}
		}
	}
	return false
}

// getAllUserPermissions gets all permissions for a user including inherited ones
func (rbac *RBACEngine) getAllUserPermissions(user *UserRBAC) map[string]bool {
	permissions := make(map[string]bool)
	visited := make(map[string]bool)
	
	// Get permissions from all roles recursively
	for _, roleID := range user.Roles {
		rbac.getRolePermissionsRecursive(roleID, permissions, visited)
	}
	
	return permissions
}

// getRolePermissionsRecursive gets permissions from a role and its parent roles
func (rbac *RBACEngine) getRolePermissionsRecursive(roleID string, permissions map[string]bool, visited map[string]bool) {
	if visited[roleID] {
		return // Avoid circular references
	}
	visited[roleID] = true
	
	role, exists := rbac.roles[roleID]
	if !exists {
		return
	}
	
	// Add role's direct permissions
	for _, permID := range role.Permissions {
		permissions[permID] = true
	}
	
	// Recursively add parent role permissions
	for _, parentRoleID := range role.ParentRoles {
		rbac.getRolePermissionsRecursive(parentRoleID, permissions, visited)
	}
}

// matchesPermission checks if a permission matches the requested resource and action
func (rbac *RBACEngine) matchesPermission(perm *Permission, resource, action string) bool {
	// Check if effect is allow
	if perm.Effect != "allow" {
		return false
	}
	
	// Match resource pattern
	if !rbac.matchesPattern(perm.Resource, resource) {
		return false
	}
	
	// Match action pattern
	if !rbac.matchesPattern(perm.Action, action) {
		return false
	}
	
	return true
}

// matchesPattern checks if a pattern matches a string (supports wildcards)
func (rbac *RBACEngine) matchesPattern(pattern, str string) bool {
	// Simple wildcard matching
	if pattern == "*" {
		return true
	}
	
	// Exact match
	if pattern == str {
		return true
	}
	
	// Prefix wildcard
	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(str, prefix)
	}
	
	// Suffix wildcard
	if strings.HasPrefix(pattern, "*") {
		suffix := strings.TrimPrefix(pattern, "*")
		return strings.HasSuffix(str, suffix)
	}
	
	return false
}

// evaluatePolicies evaluates all active policies for a user request
func (rbac *RBACEngine) evaluatePolicies(user *UserRBAC, request *AuthorizationRequest) (bool, string, []string) {
	appliedPolicies := []string{}
	
	for policyID, policy := range rbac.policies {
		if !policy.IsActive {
			continue
		}
		
		for _, rule := range policy.Rules {
			if rbac.evaluateRule(rule, user, request) {
				appliedPolicies = append(appliedPolicies, policyID)
				
				if rule.Effect == "allow" {
					return true, "policy-based permission granted", appliedPolicies
				} else if rule.Effect == "deny" {
					return false, "policy-based permission denied", appliedPolicies
				}
			}
		}
	}
	
	return false, "no matching policies", appliedPolicies
}

// evaluateRule evaluates a single policy rule
func (rbac *RBACEngine) evaluateRule(rule PolicyRule, user *UserRBAC, request *AuthorizationRequest) bool {
	// Check principals (users, roles)
	principalMatch := false
	for _, principal := range rule.Principals {
		if principal == user.ID || principal == user.Username {
			principalMatch = true
			break
		}
		// Check if principal is a role
		for _, userRole := range user.Roles {
			if principal == userRole {
				principalMatch = true
				break
			}
		}
		if principalMatch {
			break
		}
	}
	
	if !principalMatch {
		return false
	}
	
	// Check resources
	resourceMatch := false
	for _, resource := range rule.Resources {
		if rbac.matchesPattern(resource, request.Resource) {
			resourceMatch = true
			break
		}
	}
	
	if !resourceMatch {
		return false
	}
	
	// Check actions
	actionMatch := false
	for _, action := range rule.Actions {
		if rbac.matchesPattern(action, request.Action) {
			actionMatch = true
			break
		}
	}
	
	if !actionMatch {
		return false
	}
	
	// Check conditions
	for _, condition := range rule.Conditions {
		if !rbac.evaluateCondition(condition, user, request) {
			return false
		}
	}
	
	return true
}

// evaluateCondition evaluates a single policy condition
func (rbac *RBACEngine) evaluateCondition(condition PolicyCondition, user *UserRBAC, request *AuthorizationRequest) bool {
	var actualValue interface{}
	
	// Get the actual value based on the key
	switch condition.Key {
	case "user.tenant_id":
		actualValue = user.TenantID
	case "user.groups":
		actualValue = user.Groups
	case "request.tenant_id":
		actualValue = request.TenantID
	case "time.hour":
		actualValue = time.Now().Hour()
	default:
		// Check user attributes
		if strings.HasPrefix(condition.Key, "user.attributes.") {
			attrKey := strings.TrimPrefix(condition.Key, "user.attributes.")
			actualValue = user.Attributes[attrKey]
		} else if strings.HasPrefix(condition.Key, "request.context.") {
			ctxKey := strings.TrimPrefix(condition.Key, "request.context.")
			actualValue = request.Context[ctxKey]
		} else {
			return false
		}
	}
	
	// Evaluate based on operator
	switch condition.Operator {
	case "eq":
		return actualValue == condition.Value
	case "ne":
		return actualValue != condition.Value
	case "in":
		if list, ok := condition.Value.([]interface{}); ok {
			for _, item := range list {
				if actualValue == item {
					return true
				}
			}
		}
		return false
	case "not_in":
		if list, ok := condition.Value.([]interface{}); ok {
			for _, item := range list {
				if actualValue == item {
					return false
				}
			}
		}
		return true
	case "gt":
		if actualFloat, ok := actualValue.(float64); ok {
			if condFloat, ok := condition.Value.(float64); ok {
				return actualFloat > condFloat
			}
		}
		return false
	case "lt":
		if actualFloat, ok := actualValue.(float64); ok {
			if condFloat, ok := condition.Value.(float64); ok {
				return actualFloat < condFloat
			}
		}
		return false
	case "contains":
		if actualStr, ok := actualValue.(string); ok {
			if condStr, ok := condition.Value.(string); ok {
				return strings.Contains(actualStr, condStr)
			}
		}
		return false
	default:
		return false
	}
}

// AddUser adds a user to the RBAC system
func (rbac *RBACEngine) AddUser(user *UserRBAC) error {
	rbac.mu.Lock()
	defer rbac.mu.Unlock()
	
	user.CreatedAt = time.Now()
	user.UpdatedAt = time.Now()
	rbac.users[user.ID] = user
	
	// Invalidate cache for this user
	rbac.invalidateUserCache(user.ID)
	
	return nil
}

// AddRole adds a role to the RBAC system
func (rbac *RBACEngine) AddRole(role *Role) error {
	rbac.mu.Lock()
	defer rbac.mu.Unlock()
	
	role.CreatedAt = time.Now()
	role.UpdatedAt = time.Now()
	rbac.roles[role.ID] = role
	
	// Invalidate all cache entries as roles affect multiple users
	rbac.invalidateAllCache()
	
	return nil
}

// AddPermission adds a permission to the RBAC system
func (rbac *RBACEngine) AddPermission(permission *Permission) error {
	rbac.mu.Lock()
	defer rbac.mu.Unlock()
	
	permission.CreatedAt = time.Now()
	rbac.permissions[permission.ID] = permission
	
	// Invalidate all cache entries
	rbac.invalidateAllCache()
	
	return nil
}

// AddPolicy adds a policy to the RBAC system
func (rbac *RBACEngine) AddPolicy(policy *Policy) error {
	rbac.mu.Lock()
	defer rbac.mu.Unlock()
	
	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()
	rbac.policies[policy.ID] = policy
	
	// Invalidate all cache entries
	rbac.invalidateAllCache()
	
	return nil
}

// AssignRoleToUser assigns a role to a user
func (rbac *RBACEngine) AssignRoleToUser(userID, roleID string) error {
	rbac.mu.Lock()
	defer rbac.mu.Unlock()
	
	user, exists := rbac.users[userID]
	if !exists {
		return fmt.Errorf("user not found: %s", userID)
	}
	
	role, exists := rbac.roles[roleID]
	if !exists {
		return fmt.Errorf("role not found: %s", roleID)
	}
	
	// Check if role is already assigned
	for _, existingRole := range user.Roles {
		if existingRole == roleID {
			return nil // Already assigned
		}
	}
	
	user.Roles = append(user.Roles, roleID)
	user.UpdatedAt = time.Now()
	
	// Invalidate cache for this user
	rbac.invalidateUserCache(userID)
	
	return nil
}

// RevokeRoleFromUser revokes a role from a user
func (rbac *RBACEngine) RevokeRoleFromUser(userID, roleID string) error {
	rbac.mu.Lock()
	defer rbac.mu.Unlock()
	
	user, exists := rbac.users[userID]
	if !exists {
		return fmt.Errorf("user not found: %s", userID)
	}
	
	// Remove role from user's roles
	newRoles := []string{}
	for _, role := range user.Roles {
		if role != roleID {
			newRoles = append(newRoles, role)
		}
	}
	
	user.Roles = newRoles
	user.UpdatedAt = time.Now()
	
	// Invalidate cache for this user
	rbac.invalidateUserCache(userID)
	
	return nil
}

// Cache management methods
func (rbac *RBACEngine) generateCacheKey(userID, resource, action string) string {
	return fmt.Sprintf("%s:%s:%s", userID, resource, action)
}

func (pc *PermissionCache) get(key string) *CacheEntry {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	
	entry, exists := pc.cache[key]
	if !exists || time.Now().After(entry.ExpiresAt) {
		return nil
	}
	
	return entry
}

func (pc *PermissionCache) set(key string, entry *CacheEntry) {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	
	// Remove oldest entries if cache is full
	if len(pc.cache) >= pc.maxEntries {
		pc.evictOldest()
	}
	
	pc.cache[key] = entry
}

func (pc *PermissionCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time
	
	for key, entry := range pc.cache {
		if oldestKey == "" || entry.CreatedAt.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.CreatedAt
		}
	}
	
	if oldestKey != "" {
		delete(pc.cache, oldestKey)
	}
}

func (rbac *RBACEngine) invalidateUserCache(userID string) {
	rbac.cache.mu.Lock()
	defer rbac.cache.mu.Unlock()
	
	for key, entry := range rbac.cache.cache {
		if entry.UserID == userID {
			delete(rbac.cache.cache, key)
		}
	}
}

func (rbac *RBACEngine) invalidateAllCache() {
	rbac.cache.mu.Lock()
	defer rbac.cache.mu.Unlock()
	
	rbac.cache.cache = make(map[string]*CacheEntry)
}

func (rbac *RBACEngine) cleanupCache() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		rbac.cache.mu.Lock()
		now := time.Now()
		for key, entry := range rbac.cache.cache {
			if now.After(entry.ExpiresAt) {
				delete(rbac.cache.cache, key)
			}
		}
		rbac.cache.mu.Unlock()
	}
}

// initializeSystemRoles creates default system roles and permissions
func (rbac *RBACEngine) initializeSystemRoles() error {
	// Create default permissions
	permissions := []*Permission{
		{ID: "read", Name: "Read", Description: "Read access", Resource: "*", Action: "read", Effect: "allow"},
		{ID: "write", Name: "Write", Description: "Write access", Resource: "*", Action: "write", Effect: "allow"},
		{ID: "delete", Name: "Delete", Description: "Delete access", Resource: "*", Action: "delete", Effect: "allow"},
		{ID: "admin", Name: "Admin", Description: "Admin access", Resource: "*", Action: "*", Effect: "allow"},
		{ID: "vm_read", Name: "VM Read", Description: "Read VM data", Resource: "/api/vm/*", Action: "read", Effect: "allow"},
		{ID: "vm_write", Name: "VM Write", Description: "Modify VMs", Resource: "/api/vm/*", Action: "write", Effect: "allow"},
		{ID: "vm_delete", Name: "VM Delete", Description: "Delete VMs", Resource: "/api/vm/*", Action: "delete", Effect: "allow"},
		{ID: "user_read", Name: "User Read", Description: "Read users", Resource: "/api/users/*", Action: "read", Effect: "allow"},
		{ID: "user_write", Name: "User Write", Description: "Modify users", Resource: "/api/users/*", Action: "write", Effect: "allow"},
	}
	
	for _, perm := range permissions {
		rbac.permissions[perm.ID] = perm
	}
	
	// Create default roles
	roles := []*Role{
		{
			ID:          "super_admin",
			Name:        "Super Administrator",
			Description: "Full system access",
			Permissions: []string{"admin"},
			IsSystem:    true,
		},
		{
			ID:          "admin",
			Name:        "Administrator",
			Description: "Administrative access",
			Permissions: []string{"read", "write", "delete", "user_read", "user_write"},
			IsSystem:    true,
		},
		{
			ID:          "vm_admin",
			Name:        "VM Administrator",
			Description: "Full VM management access",
			Permissions: []string{"vm_read", "vm_write", "vm_delete"},
			IsSystem:    true,
		},
		{
			ID:          "vm_operator",
			Name:        "VM Operator",
			Description: "VM operation access",
			Permissions: []string{"vm_read", "vm_write"},
			IsSystem:    true,
		},
		{
			ID:          "vm_viewer",
			Name:        "VM Viewer",
			Description: "Read-only VM access",
			Permissions: []string{"vm_read"},
			IsSystem:    true,
		},
		{
			ID:          "user",
			Name:        "User",
			Description: "Basic user access",
			Permissions: []string{"read"},
			IsSystem:    true,
		},
	}
	
	for _, role := range roles {
		rbac.roles[role.ID] = role
	}
	
	return nil
}

// GetUserRoles returns roles for a user
func (rbac *RBACEngine) GetUserRoles(userID string) ([]*Role, error) {
	rbac.mu.RLock()
	defer rbac.mu.RUnlock()
	
	user, exists := rbac.users[userID]
	if !exists {
		return nil, fmt.Errorf("user not found: %s", userID)
	}
	
	roles := []*Role{}
	for _, roleID := range user.Roles {
		if role, exists := rbac.roles[roleID]; exists {
			roles = append(roles, role)
		}
	}
	
	return roles, nil
}

// GetUserPermissions returns all permissions for a user
func (rbac *RBACEngine) GetUserPermissions(userID string) ([]*Permission, error) {
	rbac.mu.RLock()
	defer rbac.mu.RUnlock()
	
	user, exists := rbac.users[userID]
	if !exists {
		return nil, fmt.Errorf("user not found: %s", userID)
	}
	
	permissionMap := rbac.getAllUserPermissions(user)
	permissions := []*Permission{}
	
	for permID := range permissionMap {
		if perm, exists := rbac.permissions[permID]; exists {
			permissions = append(permissions, perm)
		}
	}
	
	return permissions, nil
}