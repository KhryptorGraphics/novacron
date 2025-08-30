package quotas

import (
	"context"
	"fmt"
	"time"
)

// Standalone integration that doesn't depend on external packages
// This provides the same functionality but with embedded interfaces

// StandaloneIntegration provides system integration without external dependencies
type StandaloneIntegration struct {
	// Core system managers
	quotaManager    *Manager
	authManager     AuthManager
	vmManager       VMManager
	storageService  StorageService
	schedulerService SchedulerService
	monitoringService MonitoringService
	
	// Integration configuration
	config *IntegrationConfig
	
	// Metrics collection
	metricsEnabled bool
}

// NewStandaloneIntegration creates integration with embedded interfaces
func NewStandaloneIntegration(quotaManager *Manager, config *IntegrationConfig) *StandaloneIntegration {
	if config == nil {
		config = DefaultIntegrationConfig()
	}

	return &StandaloneIntegration{
		quotaManager:   quotaManager,
		config:         config,
		metricsEnabled: config.MetricsCollection,
	}
}

// SetAuthManager sets the auth manager
func (i *StandaloneIntegration) SetAuthManager(auth AuthManager) {
	i.authManager = auth
}

// SetVMManager sets the VM manager
func (i *StandaloneIntegration) SetVMManager(vm VMManager) {
	i.vmManager = vm
}

// SetStorageService sets the storage service
func (i *StandaloneIntegration) SetStorageService(storage StorageService) {
	i.storageService = storage
}

// SetSchedulerService sets the scheduler service
func (i *StandaloneIntegration) SetSchedulerService(scheduler SchedulerService) {
	i.schedulerService = scheduler
}

// SetMonitoringService sets the monitoring service
func (i *StandaloneIntegration) SetMonitoringService(monitoring MonitoringService) {
	i.monitoringService = monitoring
}

// InitializeIntegration sets up all system integrations
func (i *StandaloneIntegration) InitializeIntegration(ctx context.Context) error {
	// Initialize VM integration
	if err := i.initializeVMIntegration(ctx); err != nil {
		return fmt.Errorf("failed to initialize VM integration: %w", err)
	}

	// Initialize storage integration
	if err := i.initializeStorageIntegration(ctx); err != nil {
		return fmt.Errorf("failed to initialize storage integration: %w", err)
	}

	// Initialize auth integration
	if err := i.initializeAuthIntegration(ctx); err != nil {
		return fmt.Errorf("failed to initialize auth integration: %w", err)
	}

	// Initialize monitoring integration
	if err := i.initializeMonitoringIntegration(ctx); err != nil {
		return fmt.Errorf("failed to initialize monitoring integration: %w", err)
	}

	return nil
}

// VM Integration

func (i *StandaloneIntegration) initializeVMIntegration(ctx context.Context) error {
	if i.vmManager == nil {
		return nil
	}

	// VM integration would register event handlers here
	// For demonstration, we'll show the concept without actual event system

	return nil
}

// HandleVMCreated handles VM creation events
func (i *StandaloneIntegration) HandleVMCreated(ctx context.Context, vmInfo *VMInfo, entityID string) error {
	if !i.config.VMResourceTracking {
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
		
		if err := i.quotaManager.ConsumeResource(ctx, usage); err != nil {
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
		
		if err := i.quotaManager.ConsumeResource(ctx, usage); err != nil {
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
	
	if err := i.quotaManager.ConsumeResource(ctx, usage); err != nil {
		return fmt.Errorf("failed to record instance usage: %w", err)
	}

	return nil
}

// HandleVMDeleted handles VM deletion events
func (i *StandaloneIntegration) HandleVMDeleted(ctx context.Context, vmInfo *VMInfo, entityID string) error {
	if !i.config.VMResourceTracking {
		return nil
	}

	// Release CPU
	if vmInfo.Config.VCPUs > 0 {
		if err := i.quotaManager.ReleaseResource(ctx, entityID, ResourceTypeCPU, int64(vmInfo.Config.VCPUs)); err != nil {
			return fmt.Errorf("failed to release CPU quota: %w", err)
		}
	}

	// Release memory
	if vmInfo.Config.MemoryMB > 0 {
		if err := i.quotaManager.ReleaseResource(ctx, entityID, ResourceTypeMemory, vmInfo.Config.MemoryMB); err != nil {
			return fmt.Errorf("failed to release memory quota: %w", err)
		}
	}

	// Release instance count
	if err := i.quotaManager.ReleaseResource(ctx, entityID, ResourceTypeInstances, 1); err != nil {
		return fmt.Errorf("failed to release instance quota: %w", err)
	}

	return nil
}

// Storage Integration

func (i *StandaloneIntegration) initializeStorageIntegration(ctx context.Context) error {
	if i.storageService == nil {
		return nil
	}

	if i.config.StorageUsageTracking {
		// Register storage event handler
		i.storageService.AddVolumeEventListener(i.HandleVolumeEvent)
	}

	return nil
}

// HandleVolumeEvent handles storage volume events
func (i *StandaloneIntegration) HandleVolumeEvent(event VolumeEvent) {
	ctx := context.Background()
	
	switch event.Type {
	case VolumeEventCreated:
		i.handleVolumeCreated(ctx, event)
	case VolumeEventDeleted:
		i.handleVolumeDeleted(ctx, event)
	case VolumeEventResized:
		i.handleVolumeResized(ctx, event)
	}
}

func (i *StandaloneIntegration) handleVolumeCreated(ctx context.Context, event VolumeEvent) {
	if !i.config.StorageUsageTracking {
		return
	}

	// Extract volume information from event data
	if volumeData, ok := event.Data.(VolumeInfo); ok {
		// Determine entity ID from volume metadata
		entityID := i.getEntityIDFromVolumeData(volumeData)
		
		// Record storage usage
		usage := &UsageRecord{
			EntityID:     entityID,
			ResourceType: ResourceTypeStorage,
			Amount:       volumeData.Size / 1024 / 1024, // Convert bytes to MB
			Delta:        volumeData.Size / 1024 / 1024,
			Timestamp:    time.Now(),
			Source:       "volume_creation",
			Metadata: map[string]interface{}{
				"volume_id":   volumeData.ID,
				"volume_name": volumeData.Name,
				"volume_type": volumeData.Type,
			},
		}
		
		i.quotaManager.ConsumeResource(ctx, usage)

		// Record volume count
		countUsage := &UsageRecord{
			EntityID:     entityID,
			ResourceType: ResourceTypeVolumes,
			Amount:       1,
			Delta:        1,
			Timestamp:    time.Now(),
			Source:       "volume_creation",
			Metadata: map[string]interface{}{
				"volume_id":   volumeData.ID,
				"volume_name": volumeData.Name,
			},
		}
		
		i.quotaManager.ConsumeResource(ctx, countUsage)
	}
}

func (i *StandaloneIntegration) handleVolumeDeleted(ctx context.Context, event VolumeEvent) {
	// Implementation for volume deletion
	// Would release storage quota
}

func (i *StandaloneIntegration) handleVolumeResized(ctx context.Context, event VolumeEvent) {
	// Implementation for volume resize
	// Would update storage quota usage
}

func (i *StandaloneIntegration) getEntityIDFromVolumeData(volumeData VolumeInfo) string {
	// Extract entity ID from volume metadata
	if entityID, exists := volumeData.Metadata["entity_id"]; exists {
		return entityID
	}
	if tenantID, exists := volumeData.Metadata["tenant_id"]; exists {
		return tenantID
	}
	return "default" // Fallback
}

// Auth Integration

func (i *StandaloneIntegration) initializeAuthIntegration(ctx context.Context) error {
	if i.authManager == nil {
		return nil
	}

	// Auth integration setup would go here
	return nil
}

// HandleTenantCreated handles new tenant creation
func (i *StandaloneIntegration) HandleTenantCreated(ctx context.Context, tenant *Tenant) error {
	if !i.config.TenantQuotaInheritance {
		return nil
	}

	// Create default quotas for the new tenant
	for resourceType, defaultLimit := range i.quotaManager.config.DefaultQuotas {
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
			i.applyParentQuotaInheritance(ctx, quota, tenant.ParentID)
		}

		if err := i.quotaManager.CreateQuota(ctx, quota); err != nil {
			return fmt.Errorf("failed to create quota for tenant %s: %w", tenant.ID, err)
		}
	}

	return nil
}

func (i *StandaloneIntegration) applyParentQuotaInheritance(ctx context.Context, quota *Quota, parentID string) {
	// Find parent quota and apply inheritance rules
	parentQuotas, err := i.quotaManager.ListQuotas(ctx, QuotaFilter{
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

// Monitoring Integration

func (i *StandaloneIntegration) initializeMonitoringIntegration(ctx context.Context) error {
	if i.monitoringService == nil || !i.config.MetricsCollection {
		return nil
	}

	// Register quota metrics collector
	// In a real implementation, this would set up periodic metric collection
	
	return nil
}

// CheckQuotaForScheduling checks if resources are available for scheduling
func (i *StandaloneIntegration) CheckQuotaForScheduling(ctx context.Context, entityID string, resourceRequirements map[ResourceType]int64) (*QuotaCheckResult, error) {
	// Check quotas for all required resources
	for resourceType, amount := range resourceRequirements {
		result, err := i.quotaManager.CheckQuota(ctx, entityID, resourceType, amount)
		if err != nil {
			return nil, err
		}
		
		if !result.Allowed {
			return result, nil
		}
	}

	return &QuotaCheckResult{Allowed: true}, nil
}

// ReserveResourcesForScheduling reserves resources for a scheduling decision
func (i *StandaloneIntegration) ReserveResourcesForScheduling(ctx context.Context, entityID string, resourceRequirements map[ResourceType]int64, duration time.Duration) ([]*ResourceReservation, error) {
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
		
		if err := i.quotaManager.ReserveResource(ctx, reservation); err != nil {
			// Rollback previous reservations on failure
			i.rollbackReservations(ctx, reservations)
			return nil, err
		}
		
		reservations = append(reservations, reservation)
	}
	
	return reservations, nil
}

func (i *StandaloneIntegration) rollbackReservations(ctx context.Context, reservations []*ResourceReservation) {
	for _, reservation := range reservations {
		i.quotaManager.CancelReservation(ctx, reservation.ID)
	}
}

// GetIntegrationMetrics returns integration metrics
func (i *StandaloneIntegration) GetIntegrationMetrics() map[string]interface{} {
	if !i.metricsEnabled {
		return nil
	}

	return map[string]interface{}{
		"vm_integration_enabled":      i.config.VMQuotaEnforcement,
		"storage_integration_enabled": i.config.StorageQuotaEnforcement,
		"auth_integration_enabled":    i.config.RBACIntegration,
		"metrics_collection_enabled":  i.config.MetricsCollection,
		"scheduler_integration_enabled": i.config.SchedulerQuotaAware,
	}
}