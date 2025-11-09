package tenancy

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Tenant represents a tenant in the multi-tenant system
type Tenant struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Status      TenantStatus      `json:"status"`
	Quotas      TenantQuotas      `json:"quotas"`
	Policies    []string          `json:"policies"`    // Policy IDs
	Tags        map[string]string `json:"tags"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	BillingInfo BillingInfo       `json:"billing_info"`
}

// TenantStatus represents tenant status
type TenantStatus string

const (
	TenantActive    TenantStatus = "active"
	TenantSuspended TenantStatus = "suspended"
	TenantDeleted   TenantStatus = "deleted"
)

// TenantQuotas defines resource quotas for a tenant
type TenantQuotas struct {
	MaxVMs      int    `json:"max_vms"`
	MaxCPUs     int    `json:"max_cpus"`
	MaxMemoryGB int    `json:"max_memory_gb"`
	MaxStorageGB int   `json:"max_storage_gb"`
	MaxNetworks int    `json:"max_networks"`
	MaxUsers    int    `json:"max_users"`
}

// BillingInfo contains tenant billing information
type BillingInfo struct {
	BillingPlan    string  `json:"billing_plan"` // basic, standard, enterprise
	MonthlyBudget  float64 `json:"monthly_budget"`
	CurrentSpend   float64 `json:"current_spend"`
	BillingContact string  `json:"billing_contact"`
}

// ResourceOwnership tracks resource ownership by tenant
type ResourceOwnership struct {
	TenantID     string    `json:"tenant_id"`
	ResourceType string    `json:"resource_type"` // vm, network, storage
	ResourceID   string    `json:"resource_id"`
	CreatedAt    time.Time `json:"created_at"`
	Tags         map[string]string `json:"tags"`
}

// TenantManager manages multi-tenancy with hard isolation
type TenantManager struct {
	mu                    sync.RWMutex
	tenants               map[string]*Tenant
	resourceOwnership     map[string]*ResourceOwnership // ResourceID -> Ownership
	hardIsolation         bool
	tenantQuotas          bool
	tenantPolicies        bool
	crossTenantValidation bool
	billingPerTenant      bool
	overheadTarget        float64
	resourceTagging       bool
	metrics               *TenancyMetrics
}

// TenancyMetrics tracks multi-tenancy metrics
type TenancyMetrics struct {
	mu                      sync.RWMutex
	TotalTenants            int64
	ActiveTenants           int64
	TotalResources          int64
	ResourcesByTenant       map[string]int64
	CrossTenantViolations   int64
	IsolationBreaches       int64
	QuotaViolations         int64
	OverheadPercentage      float64
}

// NewTenantManager creates a new tenant manager
func NewTenantManager(hardIsolation, tenantQuotas, tenantPolicies, crossTenantValidation bool) *TenantManager {
	return &TenantManager{
		tenants:               make(map[string]*Tenant),
		resourceOwnership:     make(map[string]*ResourceOwnership),
		hardIsolation:         hardIsolation,
		tenantQuotas:          tenantQuotas,
		tenantPolicies:        tenantPolicies,
		crossTenantValidation: crossTenantValidation,
		billingPerTenant:      true,
		overheadTarget:        0.03, // 3%
		resourceTagging:       true,
		metrics: &TenancyMetrics{
			ResourcesByTenant: make(map[string]int64),
		},
	}
}

// CreateTenant creates a new tenant
func (tm *TenantManager) CreateTenant(ctx context.Context, tenant *Tenant) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tenant.ID == "" {
		tenant.ID = fmt.Sprintf("tenant-%d", time.Now().UnixNano())
	}

	// Set default quotas if not specified
	if tenant.Quotas.MaxVMs == 0 {
		tenant.Quotas = TenantQuotas{
			MaxVMs:       100,
			MaxCPUs:      400,
			MaxMemoryGB:  1024,
			MaxStorageGB: 10240,
			MaxNetworks:  10,
			MaxUsers:     50,
		}
	}

	tenant.Status = TenantActive
	tenant.CreatedAt = time.Now()
	tenant.UpdatedAt = time.Now()

	tm.tenants[tenant.ID] = tenant

	tm.metrics.mu.Lock()
	tm.metrics.TotalTenants++
	tm.metrics.ActiveTenants++
	tm.metrics.mu.Unlock()

	return nil
}

// GetTenant retrieves a tenant by ID
func (tm *TenantManager) GetTenant(ctx context.Context, tenantID string) (*Tenant, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return nil, fmt.Errorf("tenant %s not found", tenantID)
	}

	return tenant, nil
}

// UpdateTenant updates a tenant
func (tm *TenantManager) UpdateTenant(ctx context.Context, tenantID string, updates *Tenant) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return fmt.Errorf("tenant %s not found", tenantID)
	}

	// Update fields
	if updates.Name != "" {
		tenant.Name = updates.Name
	}
	if updates.Description != "" {
		tenant.Description = updates.Description
	}
	if updates.Status != "" {
		tenant.Status = updates.Status
	}

	tenant.UpdatedAt = time.Now()

	return nil
}

// AssignResource assigns a resource to a tenant
func (tm *TenantManager) AssignResource(ctx context.Context, tenantID, resourceType, resourceID string, tags map[string]string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Verify tenant exists
	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return fmt.Errorf("tenant %s not found", tenantID)
	}

	// Check if resource already assigned
	if ownership, exists := tm.resourceOwnership[resourceID]; exists {
		if ownership.TenantID != tenantID {
			tm.metrics.mu.Lock()
			tm.metrics.CrossTenantViolations++
			tm.metrics.mu.Unlock()
			return fmt.Errorf("resource %s already assigned to tenant %s", resourceID, ownership.TenantID)
		}
	}

	// Check quota limits
	if tm.tenantQuotas {
		currentCount := tm.countResourcesByType(tenantID, resourceType)
		limit := tm.getQuotaLimit(tenant, resourceType)

		if currentCount >= limit {
			tm.metrics.mu.Lock()
			tm.metrics.QuotaViolations++
			tm.metrics.mu.Unlock()
			return fmt.Errorf("quota exceeded for %s: %d/%d", resourceType, currentCount, limit)
		}
	}

	// Assign resource
	ownership := &ResourceOwnership{
		TenantID:     tenantID,
		ResourceType: resourceType,
		ResourceID:   resourceID,
		CreatedAt:    time.Now(),
		Tags:         tags,
	}

	tm.resourceOwnership[resourceID] = ownership

	tm.metrics.mu.Lock()
	tm.metrics.TotalResources++
	tm.metrics.ResourcesByTenant[tenantID]++
	tm.metrics.mu.Unlock()

	return nil
}

// ValidateResourceAccess validates that a tenant has access to a resource
func (tm *TenantManager) ValidateResourceAccess(ctx context.Context, tenantID, resourceID string) error {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	if !tm.hardIsolation {
		return nil // No isolation checks needed
	}

	ownership, exists := tm.resourceOwnership[resourceID]
	if !exists {
		return fmt.Errorf("resource %s not found", resourceID)
	}

	if ownership.TenantID != tenantID {
		tm.metrics.mu.Lock()
		tm.metrics.IsolationBreaches++
		tm.metrics.mu.Unlock()
		return fmt.Errorf("tenant %s does not have access to resource %s", tenantID, resourceID)
	}

	return nil
}

// ValidateCrossTenantAccess validates cross-tenant access attempts
func (tm *TenantManager) ValidateCrossTenantAccess(ctx context.Context, sourceTenantID, targetResourceID string) error {
	if !tm.crossTenantValidation {
		return nil
	}

	tm.mu.RLock()
	defer tm.mu.RUnlock()

	ownership, exists := tm.resourceOwnership[targetResourceID]
	if !exists {
		return fmt.Errorf("resource %s not found", targetResourceID)
	}

	if ownership.TenantID != sourceTenantID {
		tm.metrics.mu.Lock()
		tm.metrics.CrossTenantViolations++
		tm.metrics.mu.Unlock()
		return fmt.Errorf("cross-tenant access denied: tenant %s cannot access resource owned by tenant %s",
			sourceTenantID, ownership.TenantID)
	}

	return nil
}

// GetTenantResources returns all resources owned by a tenant
func (tm *TenantManager) GetTenantResources(ctx context.Context, tenantID string) ([]*ResourceOwnership, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	resources := make([]*ResourceOwnership, 0)

	for _, ownership := range tm.resourceOwnership {
		if ownership.TenantID == tenantID {
			resources = append(resources, ownership)
		}
	}

	return resources, nil
}

// GetTenantResourceUsage returns resource usage for a tenant
func (tm *TenantManager) GetTenantResourceUsage(ctx context.Context, tenantID string) (*TenantResourceUsage, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return nil, fmt.Errorf("tenant %s not found", tenantID)
	}

	usage := &TenantResourceUsage{
		TenantID:  tenantID,
		Timestamp: time.Now(),
	}

	// Count resources by type
	for _, ownership := range tm.resourceOwnership {
		if ownership.TenantID == tenantID {
			switch ownership.ResourceType {
			case "vm":
				usage.VMCount++
			case "network":
				usage.NetworkCount++
			case "storage":
				usage.StorageCount++
			}
		}
	}

	// Calculate quota utilization
	if tenant.Quotas.MaxVMs > 0 {
		usage.QuotaUtilization.VMs = float64(usage.VMCount) / float64(tenant.Quotas.MaxVMs) * 100
	}
	if tenant.Quotas.MaxNetworks > 0 {
		usage.QuotaUtilization.Networks = float64(usage.NetworkCount) / float64(tenant.Quotas.MaxNetworks) * 100
	}

	return usage, nil
}

// TenantResourceUsage represents resource usage for a tenant
type TenantResourceUsage struct {
	TenantID         string    `json:"tenant_id"`
	VMCount          int       `json:"vm_count"`
	NetworkCount     int       `json:"network_count"`
	StorageCount     int       `json:"storage_count"`
	QuotaUtilization struct {
		VMs      float64 `json:"vms"`
		Networks float64 `json:"networks"`
		Storage  float64 `json:"storage"`
	} `json:"quota_utilization"`
	Timestamp time.Time `json:"timestamp"`
}

// countResourcesByType counts resources by type for a tenant
func (tm *TenantManager) countResourcesByType(tenantID, resourceType string) int {
	count := 0
	for _, ownership := range tm.resourceOwnership {
		if ownership.TenantID == tenantID && ownership.ResourceType == resourceType {
			count++
		}
	}
	return count
}

// getQuotaLimit returns quota limit for resource type
func (tm *TenantManager) getQuotaLimit(tenant *Tenant, resourceType string) int {
	switch resourceType {
	case "vm":
		return tenant.Quotas.MaxVMs
	case "network":
		return tenant.Quotas.MaxNetworks
	case "storage":
		return tenant.Quotas.MaxStorageGB
	default:
		return 0
	}
}

// CalculateOverhead calculates multi-tenancy overhead
func (tm *TenantManager) CalculateOverhead() float64 {
	tm.metrics.mu.Lock()
	defer tm.metrics.mu.Unlock()

	// Simplified overhead calculation
	// In production, this would measure actual resource overhead
	baseResourceCount := float64(tm.metrics.TotalResources)
	tenantManagementOverhead := float64(tm.metrics.TotalTenants) * 0.1

	if baseResourceCount == 0 {
		return 0
	}

	overhead := (tenantManagementOverhead / baseResourceCount) * 100
	tm.metrics.OverheadPercentage = overhead

	return overhead
}

// EnforceTenantIsolation enforces hard isolation between tenants
func (tm *TenantManager) EnforceTenantIsolation(ctx context.Context) error {
	if !tm.hardIsolation {
		return nil
	}

	tm.mu.RLock()
	defer tm.mu.RUnlock()

	// Validate no cross-tenant resource references
	violations := make([]string, 0)

	for resourceID, ownership := range tm.resourceOwnership {
		// In production, this would check network connectivity, storage access, etc.
		// For now, we just verify ownership is set
		if ownership.TenantID == "" {
			violations = append(violations, fmt.Sprintf("Resource %s has no tenant assignment", resourceID))
		}
	}

	if len(violations) > 0 {
		return fmt.Errorf("isolation violations detected: %v", violations)
	}

	return nil
}

// GetMetrics returns tenancy metrics
func (tm *TenantManager) GetMetrics() *TenancyMetrics {
	tm.metrics.mu.RLock()
	defer tm.metrics.mu.RUnlock()

	metrics := &TenancyMetrics{
		TotalTenants:          tm.metrics.TotalTenants,
		ActiveTenants:         tm.metrics.ActiveTenants,
		TotalResources:        tm.metrics.TotalResources,
		ResourcesByTenant:     make(map[string]int64),
		CrossTenantViolations: tm.metrics.CrossTenantViolations,
		IsolationBreaches:     tm.metrics.IsolationBreaches,
		QuotaViolations:       tm.metrics.QuotaViolations,
		OverheadPercentage:    tm.metrics.OverheadPercentage,
	}

	for k, v := range tm.metrics.ResourcesByTenant {
		metrics.ResourcesByTenant[k] = v
	}

	return metrics
}

// ListTenants returns all tenants
func (tm *TenantManager) ListTenants(ctx context.Context) ([]*Tenant, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	tenants := make([]*Tenant, 0, len(tm.tenants))
	for _, tenant := range tm.tenants {
		tenants = append(tenants, tenant)
	}

	return tenants, nil
}

// DeleteTenant deletes a tenant (soft delete)
func (tm *TenantManager) DeleteTenant(ctx context.Context, tenantID string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tenant, exists := tm.tenants[tenantID]
	if !exists {
		return fmt.Errorf("tenant %s not found", tenantID)
	}

	// Check if tenant has resources
	hasResources := false
	for _, ownership := range tm.resourceOwnership {
		if ownership.TenantID == tenantID {
			hasResources = true
			break
		}
	}

	if hasResources {
		return fmt.Errorf("cannot delete tenant %s: tenant still has resources assigned", tenantID)
	}

	tenant.Status = TenantDeleted
	tenant.UpdatedAt = time.Now()

	tm.metrics.mu.Lock()
	tm.metrics.ActiveTenants--
	tm.metrics.mu.Unlock()

	return nil
}
