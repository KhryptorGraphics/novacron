package auth

import (
	"fmt"
	"sync"
	"time"
)

// Role represents a role in the system for RBAC
type Role struct {
	// ID is the unique identifier for the role
	ID string `json:"id"`

	// Name is the human-readable name of the role
	Name string `json:"name"`

	// Description is a description of the role
	Description string `json:"description,omitempty"`

	// Permissions are the permissions granted by this role
	Permissions []Permission `json:"permissions"`

	// IsSystem indicates if this is a system role
	IsSystem bool `json:"isSystem"`

	// TenantID is the tenant this role belongs to
	// If empty, it's a global role
	TenantID string `json:"tenantId,omitempty"`

	// Metadata contains additional metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`

	// CreatedAt is the time when the role was created
	CreatedAt time.Time `json:"createdAt"`

	// UpdatedAt is the time when the role was last updated
	UpdatedAt time.Time `json:"updatedAt"`

	// CreatedBy is the ID of the user who created this role
	CreatedBy string `json:"createdBy,omitempty"`

	// UpdatedBy is the ID of the user who last updated this role
	UpdatedBy string `json:"updatedBy,omitempty"`
}

// Permission represents a permission in the system
type Permission struct {
	// Resource is the resource the permission applies to
	Resource string `json:"resource"`

	// Action is the action allowed on the resource
	Action string `json:"action"`

	// Effect is the effect of the permission (allow or deny)
	Effect string `json:"effect"`

	// Conditions are conditions for the permission
	Conditions map[string]interface{} `json:"conditions,omitempty"`
}

// RoleService provides operations for managing roles
type RoleService interface {
	// Create creates a new role
	Create(role *Role) error

	// Get gets a role by ID
	Get(id string) (*Role, error)

	// List lists roles with optional filtering
	List(filter map[string]interface{}) ([]*Role, error)

	// Update updates a role
	Update(role *Role) error

	// Delete deletes a role
	Delete(id string) error

	// AddPermission adds a permission to a role
	AddPermission(roleID string, permission Permission) error

	// RemovePermission removes a permission from a role
	RemovePermission(roleID string, resource string, action string) error

	// HasPermission checks if a role has a specific permission
	HasPermission(roleID string, resource string, action string) (bool, error)
}

// SystemRoles contains the system roles
var SystemRoles = map[string]*Role{
	"admin": {
		ID:          "admin",
		Name:        "Administrator",
		Description: "Administrator role with full access",
		IsSystem:    true,
		Permissions: []Permission{
			{
				Resource: "*",
				Action:   "*",
				Effect:   "allow",
			},
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	},
	"user": {
		ID:          "user",
		Name:        "User",
		Description: "Regular user role with limited access",
		IsSystem:    true,
		Permissions: []Permission{
			{
				Resource: "vm",
				Action:   "read",
				Effect:   "allow",
			},
			{
				Resource: "vm",
				Action:   "start",
				Effect:   "allow",
			},
			{
				Resource: "vm",
				Action:   "stop",
				Effect:   "allow",
			},
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	},
	"readonly": {
		ID:          "readonly",
		Name:        "Read Only",
		Description: "Read-only access to all resources",
		IsSystem:    true,
		Permissions: []Permission{
			{
				Resource: "*",
				Action:   "read",
				Effect:   "allow",
			},
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	},
}

// clonePermissions returns an independent copy of perms, deep-copying each
// permission's Conditions map so a caller cannot mutate a store's internal
// state through a returned slice/element.
func clonePermissions(perms []Permission) []Permission {
	if perms == nil {
		return nil
	}
	out := make([]Permission, len(perms))
	for i, p := range perms {
		out[i] = p
		if p.Conditions != nil {
			out[i].Conditions = make(map[string]interface{}, len(p.Conditions))
			for k, v := range p.Conditions {
				out[i].Conditions[k] = v
			}
		}
	}
	return out
}

// cloneRole returns a deep copy of r (Permissions and Metadata included), so a
// caller mutating the result cannot affect the store's internal state.
func cloneRole(r *Role) *Role {
	if r == nil {
		return nil
	}
	clone := *r
	clone.Permissions = clonePermissions(r.Permissions)
	clone.Metadata = make(map[string]interface{}, len(r.Metadata))
	for k, v := range r.Metadata {
		clone.Metadata[k] = v
	}
	return &clone
}

// RoleMemoryStore is an in-memory implementation of RoleService
type RoleMemoryStore struct {
	mu    sync.RWMutex
	roles map[string]*Role
}

// NewRoleMemoryStore creates a new in-memory role store
func NewRoleMemoryStore() *RoleMemoryStore {
	store := &RoleMemoryStore{
		roles: make(map[string]*Role),
	}

	// Add system roles. Each store instance gets its own independent copy so
	// mutating one store's "admin" role (e.g. via AddPermission) cannot leak
	// into every other store sharing the package-level SystemRoles map.
	for _, role := range SystemRoles {
		store.roles[role.ID] = cloneRole(role)
	}

	return store
}

// Create creates a new role
func (s *RoleMemoryStore) Create(role *Role) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.roles[role.ID]; exists {
		return fmt.Errorf("role already exists: %s", role.ID)
	}

	// Check if this is trying to override a system role
	if systemRole, isSystem := SystemRoles[role.ID]; isSystem {
		return fmt.Errorf("cannot override system role: %s", systemRole.Name)
	}

	s.roles[role.ID] = cloneRole(role)
	return nil
}

// Get gets a role by ID
func (s *RoleMemoryStore) Get(id string) (*Role, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	role, exists := s.roles[id]
	if !exists {
		return nil, fmt.Errorf("role not found: %s", id)
	}
	return cloneRole(role), nil
}

// List lists roles with optional filtering
func (s *RoleMemoryStore) List(filter map[string]interface{}) ([]*Role, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	roles := make([]*Role, 0, len(s.roles))

	for _, role := range s.roles {
		match := true
		for k, v := range filter {
			switch k {
			case "tenantId":
				if role.TenantID != v.(string) {
					match = false
				}
			case "isSystem":
				if role.IsSystem != v.(bool) {
					match = false
				}
			}
		}
		if match {
			roles = append(roles, cloneRole(role))
		}
	}

	return roles, nil
}

// Update updates a role
func (s *RoleMemoryStore) Update(role *Role) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.roles[role.ID]; !exists {
		return fmt.Errorf("role not found: %s", role.ID)
	}

	// Check if this is trying to modify a system role
	if systemRole, isSystem := SystemRoles[role.ID]; isSystem {
		return fmt.Errorf("cannot modify system role: %s", systemRole.Name)
	}

	s.roles[role.ID] = cloneRole(role)
	return nil
}

// Delete deletes a role
func (s *RoleMemoryStore) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.roles[id]; !exists {
		return fmt.Errorf("role not found: %s", id)
	}

	// Check if this is trying to delete a system role
	if _, isSystem := SystemRoles[id]; isSystem {
		return fmt.Errorf("cannot delete system role: %s", id)
	}

	delete(s.roles, id)
	return nil
}

// AddPermission adds a permission to a role
func (s *RoleMemoryStore) AddPermission(roleID string, permission Permission) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	role, exists := s.roles[roleID]
	if !exists {
		return fmt.Errorf("role not found: %s", roleID)
	}

	// Check if this is trying to modify a system role
	if _, isSystem := SystemRoles[roleID]; isSystem {
		return fmt.Errorf("cannot modify system role: %s", roleID)
	}

	// Check if the permission already exists
	for i, p := range role.Permissions {
		if p.Resource == permission.Resource && p.Action == permission.Action {
			// Update the permission in place — indexing into the slice
			// (rather than ranging over a copy) so the change actually
			// persists onto the stored role.
			role.Permissions[i].Effect = permission.Effect
			role.Permissions[i].Conditions = permission.Conditions
			return nil
		}
	}

	// Add the permission
	role.Permissions = append(role.Permissions, permission)
	return nil
}

// RemovePermission removes a permission from a role
func (s *RoleMemoryStore) RemovePermission(roleID string, resource string, action string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	role, exists := s.roles[roleID]
	if !exists {
		return fmt.Errorf("role not found: %s", roleID)
	}

	// Check if this is trying to modify a system role
	if _, isSystem := SystemRoles[roleID]; isSystem {
		return fmt.Errorf("cannot modify system role: %s", roleID)
	}

	for i, p := range role.Permissions {
		if p.Resource == resource && p.Action == action {
			// Remove the permission
			role.Permissions = append(role.Permissions[:i], role.Permissions[i+1:]...)
			return nil
		}
	}

	return nil // Permission not found, but that's OK
}

// HasPermission checks if a role has a specific permission
func (s *RoleMemoryStore) HasPermission(roleID string, resource string, action string) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	role, exists := s.roles[roleID]
	if !exists {
		return false, fmt.Errorf("role not found: %s", roleID)
	}

	// Check for explicit matching permission
	for _, p := range role.Permissions {
		// Check exact match
		if (p.Resource == resource || p.Resource == "*") && (p.Action == action || p.Action == "*") {
			return p.Effect == "allow", nil
		}
	}

	// No matching permission found
	return false, nil
}

// NewRole creates a new role with default values
func NewRole(id, name, description, tenantID string) *Role {
	now := time.Now()
	return &Role{
		ID:          id,
		Name:        name,
		Description: description,
		TenantID:    tenantID,
		Permissions: []Permission{},
		IsSystem:    false,
		CreatedAt:   now,
		UpdatedAt:   now,
		Metadata:    make(map[string]interface{}),
	}
}
