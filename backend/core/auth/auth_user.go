package auth

// EnhancedUser extends the base User with additional fields for advanced auth
type EnhancedUser struct {
	// Base user information
	*User

	// IsActive indicates if the user is active
	IsActive bool `json:"isActive"`

	// IsSystem indicates if this is a system user with elevated privileges
	IsSystem bool `json:"isSystem"`

	// Roles contains the full role objects assigned to this user
	Roles []*Role `json:"roles,omitempty"`

	// Tenants contains the IDs of all tenants this user belongs to
	Tenants []string `json:"tenants"`

	// SystemPermissions contains direct system-level permissions (not tied to roles)
	SystemPermissions []Permission `json:"systemPermissions,omitempty"`
}

// Using the TenantService interface already defined in tenant.go

// EnhancedUserService extends UserService with additional operations
type EnhancedUserService interface {
	UserService

	// GetEnhanced gets an enhanced user by ID
	GetEnhanced(id string) (*EnhancedUser, error)

	// GetEnhancedByUsername gets an enhanced user by username
	GetEnhancedByUsername(username string) (*EnhancedUser, error)

	// SetSystemStatus sets a user's system status
	SetSystemStatus(id string, isSystem bool) error

	// AddTenant adds a tenant to a user
	AddTenant(userID, tenantID string) error

	// RemoveTenant removes a tenant from a user
	RemoveTenant(userID, tenantID string) error

	// GetTenants gets tenants a user belongs to
	GetTenants(userID string) ([]string, error)
}

// EnhancedUserMemoryStore is an in-memory implementation of EnhancedUserService
type EnhancedUserMemoryStore struct {
	*UserMemoryStore
	systemUsers   map[string]bool
	userTenants   map[string][]string
	tenantService TenantService
	roleService   RoleService
}

// NewEnhancedUserMemoryStore creates a new in-memory enhanced user store
func NewEnhancedUserMemoryStore(roleService RoleService, tenantService TenantService) *EnhancedUserMemoryStore {
	return &EnhancedUserMemoryStore{
		UserMemoryStore: NewUserMemoryStore(),
		systemUsers:     make(map[string]bool),
		userTenants:     make(map[string][]string),
		roleService:     roleService,
		tenantService:   tenantService,
	}
}

// Create creates a new user
func (s *EnhancedUserMemoryStore) Create(user *User, password string) error {
	err := s.UserMemoryStore.Create(user, password)
	if err != nil {
		return err
	}

	// Initialize tenants with the user's primary tenant
	if user.TenantID != "" {
		s.userTenants[user.ID] = []string{user.TenantID}
	} else {
		s.userTenants[user.ID] = []string{}
	}

	return nil
}

// Delete deletes a user
func (s *EnhancedUserMemoryStore) Delete(id string) error {
	err := s.UserMemoryStore.Delete(id)
	if err != nil {
		return err
	}

	// Clean up system users and user tenants
	delete(s.systemUsers, id)
	delete(s.userTenants, id)

	return nil
}

// GetEnhanced gets an enhanced user by ID
func (s *EnhancedUserMemoryStore) GetEnhanced(id string) (*EnhancedUser, error) {
	user, err := s.UserMemoryStore.Get(id)
	if err != nil {
		return nil, err
	}

	return s.enhanceUser(user)
}

// GetEnhancedByUsername gets an enhanced user by username
func (s *EnhancedUserMemoryStore) GetEnhancedByUsername(username string) (*EnhancedUser, error) {
	user, err := s.UserMemoryStore.GetByUsername(username)
	if err != nil {
		return nil, err
	}

	return s.enhanceUser(user)
}

// SetSystemStatus sets a user's system status
func (s *EnhancedUserMemoryStore) SetSystemStatus(id string, isSystem bool) error {
	_, err := s.UserMemoryStore.Get(id)
	if err != nil {
		return err
	}

	s.systemUsers[id] = isSystem
	return nil
}

// AddTenant adds a tenant to a user
func (s *EnhancedUserMemoryStore) AddTenant(userID, tenantID string) error {
	_, err := s.UserMemoryStore.Get(userID)
	if err != nil {
		return err
	}

	tenants, exists := s.userTenants[userID]
	if !exists {
		tenants = []string{}
	}

	// Check if tenant already exists
	for _, existingTenantID := range tenants {
		if existingTenantID == tenantID {
			return nil // Already has this tenant
		}
	}

	// Add the tenant
	s.userTenants[userID] = append(tenants, tenantID)
	return nil
}

// RemoveTenant removes a tenant from a user
func (s *EnhancedUserMemoryStore) RemoveTenant(userID, tenantID string) error {
	_, err := s.UserMemoryStore.Get(userID)
	if err != nil {
		return err
	}

	tenants, exists := s.userTenants[userID]
	if !exists {
		return nil // No tenants to remove
	}

	// Find and remove the tenant
	for i, existingTenantID := range tenants {
		if existingTenantID == tenantID {
			s.userTenants[userID] = append(tenants[:i], tenants[i+1:]...)
			return nil
		}
	}

	return nil // Tenant not found, but that's OK
}

// GetTenants gets tenants a user belongs to
func (s *EnhancedUserMemoryStore) GetTenants(userID string) ([]string, error) {
	_, err := s.UserMemoryStore.Get(userID)
	if err != nil {
		return nil, err
	}

	tenants, exists := s.userTenants[userID]
	if !exists {
		return []string{}, nil
	}

	// Return a copy to prevent modification
	result := make([]string, len(tenants))
	copy(result, tenants)
	return result, nil
}

// enhanceUser creates an EnhancedUser from a base User
func (s *EnhancedUserMemoryStore) enhanceUser(user *User) (*EnhancedUser, error) {
	enhanced := &EnhancedUser{
		User:     user,
		IsActive: user.Status == UserStatusActive,
		IsSystem: s.systemUsers[user.ID],
	}

	// Get full roles
	roles, err := s.UserMemoryStore.GetRoles(user.ID)
	if err != nil {
		return nil, err
	}
	enhanced.Roles = roles

	// Get tenants
	tenants, exists := s.userTenants[user.ID]
	if !exists {
		tenants = []string{}
	}

	// Ensure primary tenant is included
	if user.TenantID != "" {
		found := false
		for _, tenantID := range tenants {
			if tenantID == user.TenantID {
				found = true
				break
			}
		}
		if !found {
			tenants = append(tenants, user.TenantID)
		}
	}

	enhanced.Tenants = tenants

	return enhanced, nil
}
