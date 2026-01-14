package quotas

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Manager implements the QuotaService interface
type Manager struct {
	// Storage for quotas and related data
	quotas       map[string]*Quota
	usageRecords map[string][]*UsageRecord
	reservations map[string]*ResourceReservation
	templates    map[string]*QuotaTemplate

	// Configuration and dependencies
	config *ManagerConfig

	// Thread safety
	mutex sync.RWMutex

	// Internal state
	running bool
}

// ManagerConfig contains configuration for the quota manager
type ManagerConfig struct {
	// Default quotas to create for new entities
	DefaultQuotas map[ResourceType]int64 `json:"default_quotas"`

	// Enable features
	EnableHierarchy bool `json:"enable_hierarchy"`
	EnableBurst     bool `json:"enable_burst"`
	EnableCost      bool `json:"enable_cost"`

	// Cache settings
	CacheEnabled bool          `json:"cache_enabled"`
	CacheTTL     time.Duration `json:"cache_ttl"`

	// Metrics collection
	MetricsEnabled bool `json:"metrics_enabled"`
	
	// Compliance features
	ComplianceEnabled bool `json:"compliance_enabled"`
}

// DefaultManagerConfig returns a sensible default configuration
func DefaultManagerConfig() *ManagerConfig {
	return &ManagerConfig{
		DefaultQuotas: map[ResourceType]int64{
			ResourceTypeCPU:          10,     // 10 vCPUs
			ResourceTypeMemory:       8192,   // 8GB
			ResourceTypeStorage:      102400, // 100GB
			ResourceTypeInstances:    5,      // 5 VMs
			ResourceTypeBandwidthIn:  1000,   // 1Gbps
			ResourceTypeSnapshots:    10,     // 10 snapshots
		},
		EnableHierarchy:   true,
		EnableBurst:       true,
		EnableCost:        false, // Disable by default
		CacheEnabled:      true,
		CacheTTL:          5 * time.Minute,
		MetricsEnabled:    true,
		ComplianceEnabled: false,
	}
}

// NewManager creates a new quota manager
func NewManager(config *ManagerConfig) *Manager {
	if config == nil {
		config = DefaultManagerConfig()
	}

	return &Manager{
		quotas:       make(map[string]*Quota),
		usageRecords: make(map[string][]*UsageRecord),
		reservations: make(map[string]*ResourceReservation),
		templates:    make(map[string]*QuotaTemplate),
		config:       config,
	}
}

// Start initializes the quota manager
func (m *Manager) Start() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if m.running {
		return fmt.Errorf("quota manager is already running")
	}

	// Initialize any required resources
	m.running = true
	return nil
}

// Stop shuts down the quota manager
func (m *Manager) Stop() error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if !m.running {
		return fmt.Errorf("quota manager is not running")
	}

	m.running = false
	return nil
}

// IsRunning returns whether the manager is currently running
func (m *Manager) IsRunning() bool {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	return m.running
}

// CreateQuota creates a new quota
func (m *Manager) CreateQuota(ctx context.Context, quota *Quota) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Generate ID if not provided
	if quota.ID == "" {
		quota.ID = generateID()
	}

	// Set creation timestamp
	quota.CreatedAt = time.Now()
	quota.UpdatedAt = time.Now()

	// Validate the quota
	if err := m.validateQuota(quota); err != nil {
		return fmt.Errorf("quota validation failed: %w", err)
	}

	// Initialize computed fields
	quota.Available = quota.Limit - quota.Used - quota.Reserved

	// Store the quota
	m.quotas[quota.ID] = quota

	return nil
}

// GetQuota retrieves a quota by ID
func (m *Manager) GetQuota(ctx context.Context, quotaID string) (*Quota, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	quota, exists := m.quotas[quotaID]
	if !exists {
		return nil, fmt.Errorf("quota not found: %s", quotaID)
	}

	// Return a copy to prevent external modification
	return m.copyQuota(quota), nil
}

// UpdateQuota updates an existing quota
func (m *Manager) UpdateQuota(ctx context.Context, quota *Quota) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	existing, exists := m.quotas[quota.ID]
	if !exists {
		return fmt.Errorf("quota not found: %s", quota.ID)
	}

	// Preserve creation time and update modification time
	quota.CreatedAt = existing.CreatedAt
	quota.UpdatedAt = time.Now()

	// Validate the updated quota
	if err := m.validateQuota(quota); err != nil {
		return fmt.Errorf("quota validation failed: %w", err)
	}

	// Update computed fields
	quota.Available = quota.Limit - quota.Used - quota.Reserved

	// Store the updated quota
	m.quotas[quota.ID] = quota

	return nil
}

// DeleteQuota removes a quota
func (m *Manager) DeleteQuota(ctx context.Context, quotaID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if _, exists := m.quotas[quotaID]; !exists {
		return fmt.Errorf("quota not found: %s", quotaID)
	}

	delete(m.quotas, quotaID)
	return nil
}

// ListQuotas returns quotas matching the filter
func (m *Manager) ListQuotas(ctx context.Context, filter QuotaFilter) ([]*Quota, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var result []*Quota

	for _, quota := range m.quotas {
		if m.matchesFilter(quota, filter) {
			result = append(result, m.copyQuota(quota))
		}
	}

	// Sort by creation time (newest first)
	sort.Slice(result, func(i, j int) bool {
		return result[i].CreatedAt.After(result[j].CreatedAt)
	})

	return result, nil
}

// CheckQuota checks if a resource request can be satisfied
func (m *Manager) CheckQuota(ctx context.Context, entityID string, resourceType ResourceType, amount int64) (*QuotaCheckResult, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Find applicable quotas
	quotas := m.findApplicableQuotas(entityID, resourceType)
	if len(quotas) == 0 {
		return &QuotaCheckResult{
			Allowed:   false,
			Reason:    "No quota found for resource",
			Available: 0,
		}, nil
	}

	// Use the highest priority quota
	quota := quotas[0]

	// Calculate available capacity
	available := quota.Limit - quota.Used - quota.Reserved

	// Check if request can be satisfied
	if amount <= available {
		return &QuotaCheckResult{
			Allowed:   true,
			Available: available,
			Quota:     quota,
		}, nil
	}

	// Check if burst capacity can satisfy the request
	if m.config.EnableBurst && quota.BurstLimit > 0 {
		burstAvailable := quota.BurstLimit - quota.Used - quota.Reserved
		if amount <= burstAvailable {
			return &QuotaCheckResult{
				Allowed:    true,
				Available:  burstAvailable,
				Quota:      quota,
				UsingBurst: true,
			}, nil
		}
	}

	return &QuotaCheckResult{
		Allowed:   false,
		Reason:    fmt.Sprintf("Insufficient quota: requested %d, available %d", amount, available),
		Available: available,
		Quota:     quota,
	}, nil
}

// ConsumeResource records resource usage
func (m *Manager) ConsumeResource(ctx context.Context, usage *UsageRecord) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Find applicable quotas
	quotas := m.findApplicableQuotas(usage.EntityID, usage.ResourceType)
	if len(quotas) == 0 {
		return fmt.Errorf("no quota found for entity %s, resource %s", usage.EntityID, usage.ResourceType)
	}

	// Update quota usage (use highest priority quota)
	quota := quotas[0]
	quota.Used += usage.Delta
	quota.Available = quota.Limit - quota.Used - quota.Reserved
	quota.UpdatedAt = time.Now()

	// Update status based on usage
	m.updateQuotaStatus(quota)

	// Record usage history
	key := fmt.Sprintf("%s:%s", usage.EntityID, usage.ResourceType)
	m.usageRecords[key] = append(m.usageRecords[key], usage)

	// Keep only recent records (last 1000)
	if len(m.usageRecords[key]) > 1000 {
		m.usageRecords[key] = m.usageRecords[key][len(m.usageRecords[key])-1000:]
	}

	return nil
}

// ReleaseResource releases previously consumed resources
func (m *Manager) ReleaseResource(ctx context.Context, entityID string, resourceType ResourceType, amount int64) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Find applicable quotas
	quotas := m.findApplicableQuotas(entityID, resourceType)
	if len(quotas) == 0 {
		return fmt.Errorf("no quota found for entity %s, resource %s", entityID, resourceType)
	}

	// Update quota usage (use highest priority quota)
	quota := quotas[0]
	quota.Used -= amount
	if quota.Used < 0 {
		quota.Used = 0
	}
	quota.Available = quota.Limit - quota.Used - quota.Reserved
	quota.UpdatedAt = time.Now()

	// Update status based on usage
	m.updateQuotaStatus(quota)

	// Record the release
	usage := &UsageRecord{
		EntityID:     entityID,
		ResourceType: resourceType,
		Amount:       quota.Used, // Current total
		Delta:        -amount,    // Negative for release
		Timestamp:    time.Now(),
		Source:       "resource_release",
	}

	key := fmt.Sprintf("%s:%s", entityID, resourceType)
	m.usageRecords[key] = append(m.usageRecords[key], usage)

	return nil
}

// ReserveResource creates a resource reservation
func (m *Manager) ReserveResource(ctx context.Context, reservation *ResourceReservation) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Generate ID if not provided
	if reservation.ID == "" {
		reservation.ID = generateID()
	}

	// Find applicable quotas
	quotas := m.findApplicableQuotas(reservation.EntityID, reservation.ResourceType)
	if len(quotas) == 0 {
		return fmt.Errorf("no quota found for entity %s, resource %s", reservation.EntityID, reservation.ResourceType)
	}

	// Check if reservation can be satisfied
	quota := quotas[0]
	available := quota.Limit - quota.Used - quota.Reserved
	if reservation.Amount > available {
		return fmt.Errorf("insufficient quota for reservation: requested %d, available %d", reservation.Amount, available)
	}

	// Update quota reservation
	quota.Reserved += reservation.Amount
	quota.Available = quota.Limit - quota.Used - quota.Reserved
	quota.UpdatedAt = time.Now()

	// Store the reservation
	m.reservations[reservation.ID] = reservation

	return nil
}

// CancelReservation cancels a resource reservation
func (m *Manager) CancelReservation(ctx context.Context, reservationID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	reservation, exists := m.reservations[reservationID]
	if !exists {
		return fmt.Errorf("reservation not found: %s", reservationID)
	}

	// Find applicable quotas
	quotas := m.findApplicableQuotas(reservation.EntityID, reservation.ResourceType)
	if len(quotas) > 0 {
		quota := quotas[0]
		quota.Reserved -= reservation.Amount
		if quota.Reserved < 0 {
			quota.Reserved = 0
		}
		quota.Available = quota.Limit - quota.Used - quota.Reserved
		quota.UpdatedAt = time.Now()
	}

	// Remove the reservation
	delete(m.reservations, reservationID)

	return nil
}

// ListReservations returns reservations matching the filter
func (m *Manager) ListReservations(ctx context.Context, filter ReservationFilter) ([]*ResourceReservation, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var result []*ResourceReservation

	for _, reservation := range m.reservations {
		if m.matchesReservationFilter(reservation, filter) {
			result = append(result, reservation)
		}
	}

	// Sort by creation time (newest first)
	sort.Slice(result, func(i, j int) bool {
		return result[i].StartTime.After(result[j].StartTime)
	})

	return result, nil
}

// GetQuotaUtilization returns detailed utilization information
func (m *Manager) GetQuotaUtilization(ctx context.Context, entityID string) (*QuotaUtilization, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	utilization := &QuotaUtilization{
		EntityID:            entityID,
		ResourceUtilization: make(map[ResourceType]*ResourceUtilization),
		TopConsumers:        make([]ResourceConsumer, 0),
		Timestamp:           time.Now(),
	}

	// Calculate utilization for each resource type
	for resourceType := range m.config.DefaultQuotas {
		quotas := m.findApplicableQuotas(entityID, resourceType)
		if len(quotas) == 0 {
			continue
		}

		quota := quotas[0]
		resUtil := &ResourceUtilization{
			ResourceType: resourceType,
			Limit:        quota.Limit,
			Used:         quota.Used,
			Reserved:     quota.Reserved,
			Available:    quota.Available,
		}

		if quota.Limit > 0 {
			resUtil.Utilization = float64(quota.Used) / float64(quota.Limit) * 100
		}

		utilization.ResourceUtilization[resourceType] = resUtil

		// Add to top consumers if usage is significant
		if quota.Used > 0 {
			consumer := ResourceConsumer{
				EntityID:     entityID,
				ResourceType: resourceType,
				Amount:       quota.Used,
				Percentage:   resUtil.Utilization,
			}
			utilization.TopConsumers = append(utilization.TopConsumers, consumer)
		}
	}

	// Sort top consumers by usage percentage
	sort.Slice(utilization.TopConsumers, func(i, j int) bool {
		return utilization.TopConsumers[i].Percentage > utilization.TopConsumers[j].Percentage
	})

	// Keep only top 10
	if len(utilization.TopConsumers) > 10 {
		utilization.TopConsumers = utilization.TopConsumers[:10]
	}

	return utilization, nil
}

// GetUsageHistory returns usage history for an entity
func (m *Manager) GetUsageHistory(ctx context.Context, entityID string, resourceType ResourceType, timeRange TimeRange) ([]*UsageRecord, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	key := fmt.Sprintf("%s:%s", entityID, resourceType)
	records, exists := m.usageRecords[key]
	if !exists {
		return []*UsageRecord{}, nil
	}

	// Filter by time range
	var filtered []*UsageRecord
	for _, record := range records {
		if (timeRange.Start.IsZero() || record.Timestamp.After(timeRange.Start)) &&
		   (timeRange.End.IsZero() || record.Timestamp.Before(timeRange.End)) {
			filtered = append(filtered, record)
		}
	}

	return filtered, nil
}

// GetCostAnalysis returns cost analysis (stub implementation)
func (m *Manager) GetCostAnalysis(ctx context.Context, entityID string, timeRange TimeRange) (*CostAnalysis, error) {
	// Simple stub implementation
	return &CostAnalysis{
		EntityID:        entityID,
		StartTime:       timeRange.Start,
		EndTime:         timeRange.End,
		TotalCost:       0.0,
		ResourceCosts:   make(map[ResourceType]float64),
		CostTrends:      make([]*CostTrend, 0),
		Optimizations:   make([]*CostOptimization, 0),
	}, nil
}

// CreateTemplate creates a quota template
func (m *Manager) CreateTemplate(ctx context.Context, template *QuotaTemplate) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if template.ID == "" {
		template.ID = generateID()
	}

	template.CreatedAt = time.Now()
	template.UpdatedAt = time.Now()

	m.templates[template.ID] = template
	return nil
}

// ListTemplates returns all templates
func (m *Manager) ListTemplates(ctx context.Context) ([]*QuotaTemplate, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var result []*QuotaTemplate
	for _, template := range m.templates {
		result = append(result, template)
	}

	return result, nil
}

// ApplyTemplate applies a template to create quotas
func (m *Manager) ApplyTemplate(ctx context.Context, templateID, entityID string, level QuotaLevel) error {
	m.mutex.RLock()
	template, exists := m.templates[templateID]
	m.mutex.RUnlock()

	if !exists {
		return fmt.Errorf("template not found: %s", templateID)
	}

	// Create quotas based on template
	for _, quotaSpec := range template.Quotas {
		quota := &Quota{
			Name:         fmt.Sprintf("%s-%s-%s", template.Name, entityID, quotaSpec.ResourceType),
			Level:        level,
			EntityID:     entityID,
			ResourceType: quotaSpec.ResourceType,
			LimitType:    quotaSpec.LimitType,
			Limit:        quotaSpec.Limit,
			BurstLimit:   quotaSpec.BurstLimit,
			Status:       QuotaStatusActive,
			Priority:     quotaSpec.Priority,
		}

		if err := m.CreateQuota(ctx, quota); err != nil {
			return fmt.Errorf("failed to create quota from template: %w", err)
		}
	}

	return nil
}

// Helper methods

func (m *Manager) validateQuota(quota *Quota) error {
	if quota.Name == "" {
		return fmt.Errorf("quota name is required")
	}
	if quota.EntityID == "" {
		return fmt.Errorf("entity ID is required")
	}
	if quota.Limit < 0 {
		return fmt.Errorf("quota limit cannot be negative")
	}
	if quota.BurstLimit < 0 {
		return fmt.Errorf("burst limit cannot be negative")
	}
	if quota.Priority < 0 {
		return fmt.Errorf("priority cannot be negative")
	}
	return nil
}

func (m *Manager) matchesFilter(quota *Quota, filter QuotaFilter) bool {
	if filter.EntityID != "" && quota.EntityID != filter.EntityID {
		return false
	}
	if filter.ResourceType != "" && quota.ResourceType != filter.ResourceType {
		return false
	}
	if filter.Status != "" && quota.Status != filter.Status {
		return false
	}
	if filter.Level != "" && quota.Level != filter.Level {
		return false
	}
	return true
}

func (m *Manager) matchesReservationFilter(reservation *ResourceReservation, filter ReservationFilter) bool {
	if filter.EntityID != "" && reservation.EntityID != filter.EntityID {
		return false
	}
	if filter.ResourceType != "" && reservation.ResourceType != filter.ResourceType {
		return false
	}
	return true
}

func (m *Manager) findApplicableQuotas(entityID string, resourceType ResourceType) []*Quota {
	var applicable []*Quota

	for _, quota := range m.quotas {
		if quota.EntityID == entityID && quota.ResourceType == resourceType && quota.Status == QuotaStatusActive {
			applicable = append(applicable, quota)
		}
	}

	// Sort by priority (higher priority first)
	sort.Slice(applicable, func(i, j int) bool {
		return applicable[i].Priority > applicable[j].Priority
	})

	return applicable
}

func (m *Manager) updateQuotaStatus(quota *Quota) {
	if quota.Used >= quota.Limit {
		quota.Status = QuotaStatusExceeded
	} else if quota.Status == QuotaStatusExceeded && quota.Used < quota.Limit {
		quota.Status = QuotaStatusActive
	}
}

func (m *Manager) copyQuota(quota *Quota) *Quota {
	copy := *quota
	return &copy
}

// generateID creates a simple ID for testing purposes
// In production, this would use a proper UUID library
func generateID() string {
	return fmt.Sprintf("quota-%d", time.Now().UnixNano())
}