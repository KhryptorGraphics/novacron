package auth

import (
	"fmt"
	"time"
)

// TenantStatus represents the status of a tenant
type TenantStatus string

const (
	// TenantStatusActive indicates the tenant is active
	TenantStatusActive TenantStatus = "active"

	// TenantStatusInactive indicates the tenant is inactive
	TenantStatusInactive TenantStatus = "inactive"

	// TenantStatusSuspended indicates the tenant is suspended
	TenantStatusSuspended TenantStatus = "suspended"

	// TenantStatusPending indicates the tenant is pending activation
	TenantStatusPending TenantStatus = "pending"
)

// Tenant represents a tenant in the system
type Tenant struct {
	// ID is the unique identifier for the tenant
	ID string `json:"id"`

	// Name is the name of the tenant
	Name string `json:"name"`

	// Description is a description of the tenant
	Description string `json:"description,omitempty"`

	// Status is the status of the tenant
	Status TenantStatus `json:"status"`

	// ParentID is the ID of the parent tenant, if any
	ParentID string `json:"parentId,omitempty"`

	// Metadata contains additional metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`

	// ResourceQuotas defines resource quotas for the tenant
	ResourceQuotas map[string]int64 `json:"resourceQuotas,omitempty"`

	// CreatedAt is the time when the tenant was created
	CreatedAt time.Time `json:"createdAt"`

	// UpdatedAt is the time when the tenant was last updated
	UpdatedAt time.Time `json:"updatedAt"`

	// CreatedBy is the ID of the user who created this tenant
	CreatedBy string `json:"createdBy,omitempty"`

	// UpdatedBy is the ID of the user who last updated this tenant
	UpdatedBy string `json:"updatedBy,omitempty"`
}

// TenantService provides operations for managing tenants
type TenantService interface {
	// Create creates a new tenant
	Create(tenant *Tenant) error

	// Get gets a tenant by ID
	Get(id string) (*Tenant, error)

	// List lists tenants with optional filtering
	List(filter map[string]interface{}) ([]*Tenant, error)

	// Update updates a tenant
	Update(tenant *Tenant) error

	// Delete deletes a tenant
	Delete(id string) error

	// UpdateStatus updates a tenant's status
	UpdateStatus(id string, status TenantStatus) error

	// SetResourceQuota sets a resource quota for a tenant
	SetResourceQuota(id string, resource string, quota int64) error

	// GetResourceQuota gets a resource quota for a tenant
	GetResourceQuota(id string, resource string) (int64, error)

	// GetResourceQuotas gets all resource quotas for a tenant
	GetResourceQuotas(id string) (map[string]int64, error)
}

// DefaultResourceQuotas defines default resource quotas for new tenants
var DefaultResourceQuotas = map[string]int64{
	"vm.count":       10,
	"vm.cpu":         20,
	"vm.memory_gb":   40,
	"vm.storage_gb":  100,
	"user.count":     5,
	"api.rate_limit": 100,
}

// TenantMemoryStore is an in-memory implementation of TenantService
type TenantMemoryStore struct {
	tenants map[string]*Tenant
}

// NewTenantMemoryStore creates a new in-memory tenant store
func NewTenantMemoryStore() *TenantMemoryStore {
	// Create a default tenant
	defaultTenant := &Tenant{
		ID:             "default",
		Name:           "Default Tenant",
		Description:    "Default tenant for the system",
		Status:         TenantStatusActive,
		ResourceQuotas: DefaultResourceQuotas,
		CreatedAt:      time.Now(),
		UpdatedAt:      time.Now(),
		Metadata:       make(map[string]interface{}),
	}

	return &TenantMemoryStore{
		tenants: map[string]*Tenant{
			"default": defaultTenant,
		},
	}
}

// Create creates a new tenant
func (s *TenantMemoryStore) Create(tenant *Tenant) error {
	if _, exists := s.tenants[tenant.ID]; exists {
		return fmt.Errorf("tenant already exists: %s", tenant.ID)
	}

	if tenant.ResourceQuotas == nil {
		tenant.ResourceQuotas = make(map[string]int64)
		// Apply default quotas
		for resource, quota := range DefaultResourceQuotas {
			tenant.ResourceQuotas[resource] = quota
		}
	}

	s.tenants[tenant.ID] = tenant
	return nil
}

// Get gets a tenant by ID
func (s *TenantMemoryStore) Get(id string) (*Tenant, error) {
	tenant, exists := s.tenants[id]
	if !exists {
		return nil, fmt.Errorf("tenant not found: %s", id)
	}
	return tenant, nil
}

// List lists tenants with optional filtering
func (s *TenantMemoryStore) List(filter map[string]interface{}) ([]*Tenant, error) {
	tenants := make([]*Tenant, 0, len(s.tenants))

	for _, tenant := range s.tenants {
		match := true
		for k, v := range filter {
			switch k {
			case "status":
				if tenant.Status != v.(TenantStatus) {
					match = false
				}
			case "parentId":
				if tenant.ParentID != v.(string) {
					match = false
				}
			}
		}
		if match {
			tenants = append(tenants, tenant)
		}
	}

	return tenants, nil
}

// Update updates a tenant
func (s *TenantMemoryStore) Update(tenant *Tenant) error {
	if _, exists := s.tenants[tenant.ID]; !exists {
		return fmt.Errorf("tenant not found: %s", tenant.ID)
	}

	// Preserve resource quotas if not provided
	existingTenant := s.tenants[tenant.ID]
	if tenant.ResourceQuotas == nil {
		tenant.ResourceQuotas = existingTenant.ResourceQuotas
	}

	tenant.UpdatedAt = time.Now()
	s.tenants[tenant.ID] = tenant
	return nil
}

// Delete deletes a tenant
func (s *TenantMemoryStore) Delete(id string) error {
	if _, exists := s.tenants[id]; !exists {
		return fmt.Errorf("tenant not found: %s", id)
	}

	if id == "default" {
		return fmt.Errorf("cannot delete default tenant")
	}

	delete(s.tenants, id)
	return nil
}

// UpdateStatus updates a tenant's status
func (s *TenantMemoryStore) UpdateStatus(id string, status TenantStatus) error {
	tenant, exists := s.tenants[id]
	if !exists {
		return fmt.Errorf("tenant not found: %s", id)
	}

	tenant.Status = status
	tenant.UpdatedAt = time.Now()
	return nil
}

// SetResourceQuota sets a resource quota for a tenant
func (s *TenantMemoryStore) SetResourceQuota(id string, resource string, quota int64) error {
	tenant, exists := s.tenants[id]
	if !exists {
		return fmt.Errorf("tenant not found: %s", id)
	}

	if tenant.ResourceQuotas == nil {
		tenant.ResourceQuotas = make(map[string]int64)
	}

	tenant.ResourceQuotas[resource] = quota
	tenant.UpdatedAt = time.Now()
	return nil
}

// GetResourceQuota gets a resource quota for a tenant
func (s *TenantMemoryStore) GetResourceQuota(id string, resource string) (int64, error) {
	tenant, exists := s.tenants[id]
	if !exists {
		return 0, fmt.Errorf("tenant not found: %s", id)
	}

	quota, exists := tenant.ResourceQuotas[resource]
	if !exists {
		return 0, fmt.Errorf("resource quota not found: %s", resource)
	}

	return quota, nil
}

// GetResourceQuotas gets all resource quotas for a tenant
func (s *TenantMemoryStore) GetResourceQuotas(id string) (map[string]int64, error) {
	tenant, exists := s.tenants[id]
	if !exists {
		return nil, fmt.Errorf("tenant not found: %s", id)
	}

	return tenant.ResourceQuotas, nil
}

// NewTenant creates a new tenant with default values
func NewTenant(id, name, description string) *Tenant {
	now := time.Now()
	return &Tenant{
		ID:             id,
		Name:           name,
		Description:    description,
		Status:         TenantStatusPending,
		ResourceQuotas: make(map[string]int64),
		CreatedAt:      now,
		UpdatedAt:      now,
		Metadata:       make(map[string]interface{}),
	}
}
