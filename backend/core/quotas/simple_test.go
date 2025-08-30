package quotas

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test basic quota operations
func TestSimpleQuotaOperations(t *testing.T) {
	manager := NewManager(DefaultManagerConfig())
	require.NoError(t, manager.Start())
	defer manager.Stop()

	ctx := context.Background()

	t.Run("create and get quota", func(t *testing.T) {
		quota := &Quota{
			Name:         "test-cpu-quota",
			Level:        QuotaLevelTenant,
			EntityID:     "tenant-1",
			ResourceType: ResourceTypeCPU,
			LimitType:    LimitTypeHard,
			Limit:        100,
			Status:       QuotaStatusActive,
		}

		err := manager.CreateQuota(ctx, quota)
		assert.NoError(t, err)
		assert.NotEmpty(t, quota.ID)

		retrieved, err := manager.GetQuota(ctx, quota.ID)
		assert.NoError(t, err)
		assert.Equal(t, quota.Name, retrieved.Name)
		assert.Equal(t, quota.Limit, retrieved.Limit)
	})

	t.Run("update quota", func(t *testing.T) {
		quota := &Quota{
			Name:         "test-memory-quota",
			Level:        QuotaLevelTenant,
			EntityID:     "tenant-2",
			ResourceType: ResourceTypeMemory,
			LimitType:    LimitTypeHard,
			Limit:        8192,
			Status:       QuotaStatusActive,
		}

		err := manager.CreateQuota(ctx, quota)
		require.NoError(t, err)

		// Update the quota
		quota.Limit = 16384
		err = manager.UpdateQuota(ctx, quota)
		assert.NoError(t, err)

		// Verify update
		updated, err := manager.GetQuota(ctx, quota.ID)
		assert.NoError(t, err)
		assert.Equal(t, int64(16384), updated.Limit)
	})

	t.Run("list quotas", func(t *testing.T) {
		quotas, err := manager.ListQuotas(ctx, QuotaFilter{})
		assert.NoError(t, err)
		assert.Greater(t, len(quotas), 0)

		// Test filtering
		filteredQuotas, err := manager.ListQuotas(ctx, QuotaFilter{
			EntityID: "tenant-1",
		})
		assert.NoError(t, err)
		for _, quota := range filteredQuotas {
			assert.Equal(t, "tenant-1", quota.EntityID)
		}
	})

	t.Run("check quota", func(t *testing.T) {
		result, err := manager.CheckQuota(ctx, "tenant-1", ResourceTypeCPU, 50)
		assert.NoError(t, err)
		assert.True(t, result.Allowed)
	})

	t.Run("consume and release resources", func(t *testing.T) {
		// Consume resources
		usage := &UsageRecord{
			EntityID:     "tenant-1",
			ResourceType: ResourceTypeCPU,
			Amount:       30,
			Delta:        30,
			Timestamp:    time.Now(),
			Source:       "test",
		}

		err := manager.ConsumeResource(ctx, usage)
		assert.NoError(t, err)

		// Release resources
		err = manager.ReleaseResource(ctx, "tenant-1", ResourceTypeCPU, 10)
		assert.NoError(t, err)
	})

	t.Run("resource reservations", func(t *testing.T) {
		reservation := &ResourceReservation{
			EntityID:     "tenant-1",
			ResourceType: ResourceTypeCPU,
			Amount:       20,
			StartTime:    time.Now(),
			EndTime:      time.Now().Add(1 * time.Hour),
			Purpose:      "test",
		}

		err := manager.ReserveResource(ctx, reservation)
		assert.NoError(t, err)
		assert.NotEmpty(t, reservation.ID)

		// List reservations
		reservations, err := manager.ListReservations(ctx, ReservationFilter{
			EntityID: "tenant-1",
		})
		assert.NoError(t, err)
		assert.Greater(t, len(reservations), 0)

		// Cancel reservation
		err = manager.CancelReservation(ctx, reservation.ID)
		assert.NoError(t, err)
	})

	t.Run("quota utilization", func(t *testing.T) {
		utilization, err := manager.GetQuotaUtilization(ctx, "tenant-1")
		assert.NoError(t, err)
		assert.Equal(t, "tenant-1", utilization.EntityID)
		assert.NotEmpty(t, utilization.ResourceUtilization)
	})
}

// Test dashboard API
func TestSimpleDashboardAPI(t *testing.T) {
	manager := NewManager(DefaultManagerConfig())
	require.NoError(t, manager.Start())
	defer manager.Stop()

	api := NewSimpleDashboardAPI(manager)
	assert.NotNil(t, api)
	assert.Equal(t, manager, api.manager)
}

// Test integration functionality
func TestStandaloneIntegration(t *testing.T) {
	manager := NewManager(DefaultManagerConfig())
	require.NoError(t, manager.Start())
	defer manager.Stop()

	framework := NewIntegrationFramework(manager, DefaultIntegrationConfig())
	ctx := context.Background()

	t.Run("tenant creation", func(t *testing.T) {
		tenant := &Tenant{
			ID:        "new-tenant",
			Name:      "New Tenant",
			Status:    "active",
			CreatedAt: time.Now(),
		}

		err := framework.ProcessTenantCreated(ctx, *tenant)
		assert.NoError(t, err)

		// Check that quotas were created
		quotas, err := manager.ListQuotas(ctx, QuotaFilter{EntityID: "new-tenant"})
		assert.NoError(t, err)
		assert.Greater(t, len(quotas), 0)
	})

	t.Run("vm creation", func(t *testing.T) {
		// Create CPU and memory quotas first
		cpuQuota := &Quota{
			Name:         "vm-cpu-quota",
			Level:        QuotaLevelTenant,
			EntityID:     "vm-tenant",
			ResourceType: ResourceTypeCPU,
			LimitType:    LimitTypeHard,
			Limit:        50,
			Status:       QuotaStatusActive,
		}
		err := manager.CreateQuota(ctx, cpuQuota)
		require.NoError(t, err)

		memoryQuota := &Quota{
			Name:         "vm-memory-quota",
			Level:        QuotaLevelTenant,
			EntityID:     "vm-tenant",
			ResourceType: ResourceTypeMemory,
			LimitType:    LimitTypeHard,
			Limit:        8192, // 8GB
			Status:       QuotaStatusActive,
		}
		err = manager.CreateQuota(ctx, memoryQuota)
		require.NoError(t, err)

		instanceQuota := &Quota{
			Name:         "vm-instance-quota",
			Level:        QuotaLevelTenant,
			EntityID:     "vm-tenant",
			ResourceType: ResourceTypeInstances,
			LimitType:    LimitTypeHard,
			Limit:        10,
			Status:       QuotaStatusActive,
		}
		err = manager.CreateQuota(ctx, instanceQuota)
		require.NoError(t, err)

		vmInfo := VMInfo{
			ID:   "vm-test",
			Name: "Test VM",
			Config: VMConfig{
				VCPUs:    4,
				MemoryMB: 2048,
			},
			State: StateRunning,
		}

		err = framework.ProcessVMCreated(ctx, vmInfo, "vm-tenant")
		assert.NoError(t, err)

		// Check quota utilization
		utilization, err := manager.GetQuotaUtilization(ctx, "vm-tenant")
		assert.NoError(t, err)
		
		cpuUtil := utilization.ResourceUtilization[ResourceTypeCPU]
		if cpuUtil != nil {
			assert.Greater(t, cpuUtil.Used, int64(0))
		}
	})
}