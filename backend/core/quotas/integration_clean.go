package quotas

import (
	"context"
	"fmt"
	"time"
)

// Clean integration without external package dependencies
// This provides the framework that can be extended when integrated into NovaCron

// IntegrationFramework provides hooks for system integration
type IntegrationFramework struct {
	quotaManager *Manager
	config       *IntegrationConfig
	
	// Event handlers
	vmHandlers      []VMEventHandler
	storageHandlers []StorageEventHandler
	authHandlers    []AuthEventHandler
}

// Event handler interfaces
type VMEventHandler interface {
	HandleVMCreated(ctx context.Context, vmInfo VMInfo, entityID string) error
	HandleVMDeleted(ctx context.Context, vmInfo VMInfo, entityID string) error
}

type StorageEventHandler interface {
	HandleVolumeCreated(ctx context.Context, volumeInfo VolumeInfo, entityID string) error
	HandleVolumeDeleted(ctx context.Context, volumeInfo VolumeInfo, entityID string) error
}

type AuthEventHandler interface {
	HandleTenantCreated(ctx context.Context, tenant Tenant) error
	HandleUserCreated(ctx context.Context, userID, tenantID string) error
}

// NewIntegrationFramework creates a new integration framework
func NewIntegrationFramework(manager *Manager, config *IntegrationConfig) *IntegrationFramework {
	if config == nil {
		config = DefaultIntegrationConfig()
	}

	return &IntegrationFramework{
		quotaManager:    manager,
		config:          config,
		vmHandlers:      make([]VMEventHandler, 0),
		storageHandlers: make([]StorageEventHandler, 0),
		authHandlers:    make([]AuthEventHandler, 0),
	}
}

// RegisterVMHandler registers a VM event handler
func (f *IntegrationFramework) RegisterVMHandler(handler VMEventHandler) {
	f.vmHandlers = append(f.vmHandlers, handler)
}

// RegisterStorageHandler registers a storage event handler
func (f *IntegrationFramework) RegisterStorageHandler(handler StorageEventHandler) {
	f.storageHandlers = append(f.storageHandlers, handler)
}

// RegisterAuthHandler registers an auth event handler
func (f *IntegrationFramework) RegisterAuthHandler(handler AuthEventHandler) {
	f.authHandlers = append(f.authHandlers, handler)
}

// VM event processing
func (f *IntegrationFramework) ProcessVMCreated(ctx context.Context, vmInfo VMInfo, entityID string) error {
	// Process through default quota handler
	if err := f.handleVMCreatedInternal(ctx, vmInfo, entityID); err != nil {
		return err
	}

	// Process through registered handlers
	for _, handler := range f.vmHandlers {
		if err := handler.HandleVMCreated(ctx, vmInfo, entityID); err != nil {
			return err
		}
	}

	return nil
}

func (f *IntegrationFramework) ProcessVMDeleted(ctx context.Context, vmInfo VMInfo, entityID string) error {
	// Process through default quota handler
	if err := f.handleVMDeletedInternal(ctx, vmInfo, entityID); err != nil {
		return err
	}

	// Process through registered handlers
	for _, handler := range f.vmHandlers {
		if err := handler.HandleVMDeleted(ctx, vmInfo, entityID); err != nil {
			return err
		}
	}

	return nil
}

// Default internal VM handling
func (f *IntegrationFramework) handleVMCreatedInternal(ctx context.Context, vmInfo VMInfo, entityID string) error {
	if !f.config.VMResourceTracking {
		return nil
	}

	// Track CPU usage
	if vmInfo.Config.VCPUs > 0 {
		usage := &UsageRecord{
			EntityID:     entityID,
			ResourceType: ResourceTypeCPU,
			Amount:       int64(vmInfo.Config.VCPUs),
			Delta:        int64(vmInfo.Config.VCPUs),
			Timestamp:    time.Now(),
			Source:       "vm_creation",
			Metadata: map[string]interface{}{
				"vm_id":   vmInfo.ID,
				"vm_name": vmInfo.Name,
			},
		}
		
		if err := f.quotaManager.ConsumeResource(ctx, usage); err != nil {
			return fmt.Errorf("failed to record CPU usage: %w", err)
		}
	}

	// Track memory usage
	if vmInfo.Config.MemoryMB > 0 {
		usage := &UsageRecord{
			EntityID:     entityID,
			ResourceType: ResourceTypeMemory,
			Amount:       vmInfo.Config.MemoryMB,
			Delta:        vmInfo.Config.MemoryMB,
			Timestamp:    time.Now(),
			Source:       "vm_creation",
			Metadata: map[string]interface{}{
				"vm_id":   vmInfo.ID,
				"vm_name": vmInfo.Name,
			},
		}
		
		if err := f.quotaManager.ConsumeResource(ctx, usage); err != nil {
			return fmt.Errorf("failed to record memory usage: %w", err)
		}
	}

	// Track instance count
	usage := &UsageRecord{
		EntityID:     entityID,
		ResourceType: ResourceTypeInstances,
		Amount:       1,
		Delta:        1,
		Timestamp:    time.Now(),
		Source:       "vm_creation",
		Metadata: map[string]interface{}{
			"vm_id":   vmInfo.ID,
			"vm_name": vmInfo.Name,
		},
	}
	
	return f.quotaManager.ConsumeResource(ctx, usage)
}

func (f *IntegrationFramework) handleVMDeletedInternal(ctx context.Context, vmInfo VMInfo, entityID string) error {
	if !f.config.VMResourceTracking {
		return nil
	}

	// Release CPU
	if vmInfo.Config.VCPUs > 0 {
		if err := f.quotaManager.ReleaseResource(ctx, entityID, ResourceTypeCPU, int64(vmInfo.Config.VCPUs)); err != nil {
			return fmt.Errorf("failed to release CPU quota: %w", err)
		}
	}

	// Release memory
	if vmInfo.Config.MemoryMB > 0 {
		if err := f.quotaManager.ReleaseResource(ctx, entityID, ResourceTypeMemory, vmInfo.Config.MemoryMB); err != nil {
			return fmt.Errorf("failed to release memory quota: %w", err)
		}
	}

	// Release instance count
	return f.quotaManager.ReleaseResource(ctx, entityID, ResourceTypeInstances, 1)
}

// Storage event processing
func (f *IntegrationFramework) ProcessVolumeCreated(ctx context.Context, volumeInfo VolumeInfo, entityID string) error {
	// Process through default quota handler
	if err := f.handleVolumeCreatedInternal(ctx, volumeInfo, entityID); err != nil {
		return err
	}

	// Process through registered handlers
	for _, handler := range f.storageHandlers {
		if err := handler.HandleVolumeCreated(ctx, volumeInfo, entityID); err != nil {
			return err
		}
	}

	return nil
}

func (f *IntegrationFramework) handleVolumeCreatedInternal(ctx context.Context, volumeInfo VolumeInfo, entityID string) error {
	if !f.config.StorageUsageTracking {
		return nil
	}

	// Record storage usage
	usage := &UsageRecord{
		EntityID:     entityID,
		ResourceType: ResourceTypeStorage,
		Amount:       volumeInfo.Size / 1024 / 1024, // Convert bytes to MB
		Delta:        volumeInfo.Size / 1024 / 1024,
		Timestamp:    time.Now(),
		Source:       "volume_creation",
		Metadata: map[string]interface{}{
			"volume_id":   volumeInfo.ID,
			"volume_name": volumeInfo.Name,
			"volume_type": volumeInfo.Type,
		},
	}
	
	if err := f.quotaManager.ConsumeResource(ctx, usage); err != nil {
		return err
	}

	// Record volume count
	countUsage := &UsageRecord{
		EntityID:     entityID,
		ResourceType: ResourceTypeVolumes,
		Amount:       1,
		Delta:        1,
		Timestamp:    time.Now(),
		Source:       "volume_creation",
		Metadata: map[string]interface{}{
			"volume_id":   volumeInfo.ID,
			"volume_name": volumeInfo.Name,
		},
	}
	
	return f.quotaManager.ConsumeResource(ctx, countUsage)
}

// Auth event processing
func (f *IntegrationFramework) ProcessTenantCreated(ctx context.Context, tenant Tenant) error {
	// Process through default quota handler
	if err := f.handleTenantCreatedInternal(ctx, tenant); err != nil {
		return err
	}

	// Process through registered handlers
	for _, handler := range f.authHandlers {
		if err := handler.HandleTenantCreated(ctx, tenant); err != nil {
			return err
		}
	}

	return nil
}

func (f *IntegrationFramework) handleTenantCreatedInternal(ctx context.Context, tenant Tenant) error {
	if !f.config.TenantQuotaInheritance {
		return nil
	}

	// Create default quotas for the new tenant
	for resourceType, defaultLimit := range f.quotaManager.config.DefaultQuotas {
		quota := &Quota{
			Name:         fmt.Sprintf("default-%s-%s", tenant.ID, resourceType),
			Level:        QuotaLevelTenant,
			EntityID:     tenant.ID,
			ResourceType: resourceType,
			LimitType:    LimitTypeHard,
			Limit:        defaultLimit,
			Status:       QuotaStatusActive,
			Priority:     1,
		}

		// Apply tenant-specific quota modifications if parent exists
		if tenant.ParentID != "" {
			f.applyParentQuotaInheritance(ctx, quota, tenant.ParentID)
		}

		if err := f.quotaManager.CreateQuota(ctx, quota); err != nil {
			return fmt.Errorf("failed to create quota for tenant %s: %w", tenant.ID, err)
		}
	}

	return nil
}

func (f *IntegrationFramework) applyParentQuotaInheritance(ctx context.Context, quota *Quota, parentID string) {
	// Find parent quota and apply inheritance rules
	parentQuotas, err := f.quotaManager.ListQuotas(ctx, QuotaFilter{
		EntityID:     parentID,
		ResourceType: quota.ResourceType,
		Level:        QuotaLevelTenant,
	})
	
	if err != nil || len(parentQuotas) == 0 {
		return
	}

	parentQuota := parentQuotas[0]
	quota.ParentID = parentQuota.ID
	
	// Apply inheritance factor (child gets 50% of parent quota)
	quota.Limit = parentQuota.Limit / 2
}

// Quota enforcement for scheduling
func (f *IntegrationFramework) CheckQuotaForScheduling(ctx context.Context, entityID string, resourceRequirements map[ResourceType]int64) (*QuotaCheckResult, error) {
	// Check quotas for all required resources
	for resourceType, amount := range resourceRequirements {
		result, err := f.quotaManager.CheckQuota(ctx, entityID, resourceType, amount)
		if err != nil {
			return nil, err
		}
		
		if !result.Allowed {
			return result, nil
		}
	}

	return &QuotaCheckResult{Allowed: true}, nil
}

// Resource reservations for scheduling
func (f *IntegrationFramework) ReserveResourcesForScheduling(ctx context.Context, entityID string, resourceRequirements map[ResourceType]int64, duration time.Duration) ([]*ResourceReservation, error) {
	var reservations []*ResourceReservation
	
	startTime := time.Now()
	endTime := startTime.Add(duration)
	
	for resourceType, amount := range resourceRequirements {
		reservation := &ResourceReservation{
			EntityID:     entityID,
			ResourceType: resourceType,
			Amount:       amount,
			StartTime:    startTime,
			EndTime:      endTime,
			Purpose:      "scheduling",
		}
		
		if err := f.quotaManager.ReserveResource(ctx, reservation); err != nil {
			// Rollback previous reservations on failure
			f.rollbackReservations(ctx, reservations)
			return nil, err
		}
		
		reservations = append(reservations, reservation)
	}
	
	return reservations, nil
}

func (f *IntegrationFramework) rollbackReservations(ctx context.Context, reservations []*ResourceReservation) {
	for _, reservation := range reservations {
		f.quotaManager.CancelReservation(ctx, reservation.ID)
	}
}

// GetIntegrationStatus returns the current integration status
func (f *IntegrationFramework) GetIntegrationStatus() map[string]interface{} {
	return map[string]interface{}{
		"vm_integration_enabled":        f.config.VMQuotaEnforcement,
		"vm_resource_tracking":          f.config.VMResourceTracking,
		"storage_integration_enabled":   f.config.StorageQuotaEnforcement,
		"storage_usage_tracking":        f.config.StorageUsageTracking,
		"auth_integration_enabled":      f.config.RBACIntegration,
		"tenant_quota_inheritance":      f.config.TenantQuotaInheritance,
		"user_quota_management":         f.config.UserQuotaManagement,
		"scheduler_quota_aware":         f.config.SchedulerQuotaAware,
		"metrics_collection_enabled":    f.config.MetricsCollection,
		"alerts_integration":            f.config.AlertsIntegration,
		"registered_vm_handlers":        len(f.vmHandlers),
		"registered_storage_handlers":   len(f.storageHandlers),
		"registered_auth_handlers":      len(f.authHandlers),
	}
}