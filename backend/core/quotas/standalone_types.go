package quotas

// Standalone types to avoid external dependencies for core quota functionality
// In the real implementation, these would import from the actual NovaCron packages

import (
	"context"
	"time"
)

// Mock types for standalone operation
// Replace these with actual imports in production

// Auth types
type AuthManager interface {
	GetTenant(id string) (*Tenant, error)
}

type Tenant struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	ParentID    string `json:"parent_id,omitempty"`
	Status      string `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
}

// VM types
type VMManager interface {
	GetVM(vmID string) (*VM, error)
}

type VM struct {
	ID     string     `json:"id"`
	Name   string     `json:"name"`
	Config VMConfig   `json:"config"`
	State  VMState    `json:"state"`
}

type VMInfo struct {
	ID     string    `json:"id"`
	Name   string    `json:"name"`
	Config VMConfig  `json:"config"`
	State  VMState   `json:"state"`
}

type VMConfig struct {
	VCPUs    int   `json:"vcpus"`
	MemoryMB int64 `json:"memory_mb"`
}

type VMState string

const (
	StateRunning VMState = "running"
	StateStopped VMState = "stopped"
)

// Storage types
type StorageService interface {
	CreateVolume(ctx context.Context, opts VolumeCreateOptions) (*VolumeInfo, error)
	AddVolumeEventListener(listener VolumeEventListener)
}

type VolumeInfo struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Size     int64             `json:"size"`
	Type     VolumeType        `json:"type"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type VolumeCreateOptions struct {
	Name string `json:"name"`
	Size int64  `json:"size"`
}

type VolumeType string

const (
	VolumeTypeBlock VolumeType = "block"
	VolumeTypeFile  VolumeType = "file"
)

type VolumeEvent struct {
	Type       VolumeEventType `json:"type"`
	VolumeID   string          `json:"volume_id"`
	VolumeName string          `json:"volume_name"`
	Data       interface{}     `json:"data,omitempty"`
	Timestamp  time.Time       `json:"timestamp"`
}

type VolumeEventType string

const (
	VolumeEventCreated VolumeEventType = "created"
	VolumeEventDeleted VolumeEventType = "deleted"
	VolumeEventResized VolumeEventType = "resized"
)

type VolumeEventListener func(event VolumeEvent)

// Scheduler types
type SchedulerService interface {
	Schedule(ctx context.Context, requirements map[ResourceType]int64) error
}

// Monitoring types
type MonitoringService interface {
	RecordMetric(name string, value float64, tags map[string]string)
	StartService() error
	StopService() error
}

type MetricRegistry struct {
	// Implementation would be in monitoring package
}

func NewMetricRegistry() *MetricRegistry {
	return &MetricRegistry{}
}

type Metric struct {
	Name      string            `json:"name"`
	Value     float64           `json:"value"`
	Tags      map[string]string `json:"tags"`
	Timestamp time.Time         `json:"timestamp"`
}

func NewMetric(name string, metricType string, value float64, tags map[string]string) *Metric {
	return &Metric{
		Name:      name,
		Value:     value,
		Tags:      tags,
		Timestamp: time.Now(),
	}
}

func (m *Metric) WithUnit(unit string) *Metric {
	if m.Tags == nil {
		m.Tags = make(map[string]string)
	}
	m.Tags["unit"] = unit
	return m
}

func (m *Metric) WithSource(source string) *Metric {
	if m.Tags == nil {
		m.Tags = make(map[string]string)
	}
	m.Tags["source"] = source
	return m
}

// Integration types
type IntegrationConfig struct {
	VMQuotaEnforcement      bool          `json:"vm_quota_enforcement"`
	VMResourceTracking      bool          `json:"vm_resource_tracking"`
	StorageQuotaEnforcement bool          `json:"storage_quota_enforcement"`
	StorageUsageTracking    bool          `json:"storage_usage_tracking"`
	RBACIntegration         bool          `json:"rbac_integration"`
	TenantQuotaInheritance  bool          `json:"tenant_quota_inheritance"`
	UserQuotaManagement     bool          `json:"user_quota_management"`
	SchedulerQuotaAware     bool          `json:"scheduler_quota_aware"`
	MetricsCollection       bool          `json:"metrics_collection"`
	AlertsIntegration       bool          `json:"alerts_integration"`
	CacheEnabled            bool          `json:"cache_enabled"`
	CacheTTL                time.Duration `json:"cache_ttl"`
}

func DefaultIntegrationConfig() *IntegrationConfig {
	return &IntegrationConfig{
		VMQuotaEnforcement:      true,
		VMResourceTracking:      true,
		StorageQuotaEnforcement: true,
		StorageUsageTracking:    true,
		RBACIntegration:         true,
		TenantQuotaInheritance:  true,
		UserQuotaManagement:     true,
		SchedulerQuotaAware:     true,
		MetricsCollection:       true,
		AlertsIntegration:       true,
		CacheEnabled:            true,
		CacheTTL:                5 * time.Minute,
	}
}

// Dashboard API types
type MetaInfo struct {
	Timestamp   time.Time `json:"timestamp"`
	ProcessTime string    `json:"process_time,omitempty"`
	Count       int       `json:"count,omitempty"`
}

type DashboardResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
	Meta    *MetaInfo   `json:"meta"`
}

type QuotaOverview struct {
	TotalQuotas           int     `json:"total_quotas"`
	ActiveQuotas          int     `json:"active_quotas"`
	ExceededQuotas        int     `json:"exceeded_quotas"`
	SuspendedQuotas       int     `json:"suspended_quotas"`
	OverallUtilization    float64 `json:"overall_utilization"`
	TotalCost             float64 `json:"total_cost"`
	ProjectedMonthlyCost  float64 `json:"projected_monthly_cost"`
}