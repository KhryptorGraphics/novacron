package providers

import (
	"context"

	novacronv1 "github.com/khryptorgraphics/novacron/k8s-operator/pkg/apis/novacron/v1"
)

// CloudProvider represents a cloud provider interface
type CloudProvider interface {
	// GetName returns the provider name
	GetName() string

	// CreateVM creates a virtual machine
	CreateVM(ctx context.Context, req *VMRequest) (*VMResult, error)

	// GetVM retrieves VM information
	GetVM(ctx context.Context, vmID string) (*VMResult, error)

	// DeleteVM deletes a virtual machine
	DeleteVM(ctx context.Context, vmID string) error

	// ListVMs lists all VMs
	ListVMs(ctx context.Context, filters map[string]string) ([]*VMResult, error)

	// EstimateCost estimates the cost for given resources
	EstimateCost(region string, resources ResourceRequirements) (*novacronv1.ResourceCost, error)

	// GetAvailableRegions returns available regions
	GetAvailableRegions(ctx context.Context) ([]string, error)

	// GetAvailableInstanceTypes returns available instance types for a region
	GetAvailableInstanceTypes(ctx context.Context, region string) ([]InstanceType, error)

	// MigrateVM migrates a VM to another region or instance type
	MigrateVM(ctx context.Context, vmID string, target MigrationTarget) error

	// GetVMMetrics retrieves performance metrics for a VM
	GetVMMetrics(ctx context.Context, vmID string) (*VMMetrics, error)
}

// CloudProviderManager manages multiple cloud providers
type CloudProviderManager interface {
	// RegisterProvider registers a new cloud provider
	RegisterProvider(provider CloudProvider) error

	// GetClient returns a cloud provider client
	GetClient(providerName string) (CloudProvider, error)

	// ListProviders returns all registered providers
	ListProviders() []string

	// EstimateCost estimates cost across providers for comparison
	EstimateCost(providerName, region string, resources ResourceRequirements) (*novacronv1.ResourceCost, error)
}

// VMRequest represents a VM creation request
type VMRequest struct {
	Name         string
	Region       string
	InstanceType string
	Resources    ResourceRequirements
	Template     interface{}
	Tags         map[string]string
	UserData     string
	KeyPair      string
	SecurityGroups []string
	SubnetID     string
}

// VMResult represents a VM instance
type VMResult struct {
	ID           string
	Name         string
	Status       string
	InstanceType string
	Region       string
	IPAddress    string
	PrivateIP    string
	LaunchTime   string
	Tags         map[string]string
}

// ResourceRequirements represents resource requirements
type ResourceRequirements struct {
	CPU     string // e.g., "2", "1000m"
	Memory  string // e.g., "4Gi", "4096Mi"
	Storage string // e.g., "20Gi"
	GPU     int    // Number of GPUs
}

// InstanceType represents available instance types
type InstanceType struct {
	Name        string
	CPU         int
	Memory      int64 // In MB
	Storage     int64 // In GB
	GPU         int
	Network     string // Performance level
	HourlyCost  float64
	Description string
}

// MigrationTarget represents migration destination
type MigrationTarget struct {
	Region       string
	InstanceType string
	TargetNode   string
}

// VMMetrics represents VM performance metrics
type VMMetrics struct {
	Timestamp       string
	CPUUsage        float64 // Percentage
	MemoryUsage     float64 // Percentage
	NetworkIn       float64 // Bytes per second
	NetworkOut      float64 // Bytes per second
	DiskRead        float64 // IOPS
	DiskWrite       float64 // IOPS
	CPUCreditUsage  float64 // For burstable instances
}

// ProviderConfig represents provider configuration
type ProviderConfig struct {
	Name        string
	Type        string // aws, azure, gcp, etc.
	Region      string
	Credentials map[string]string
	Config      map[string]interface{}
}

// CloudProviderFactory creates cloud provider instances
type CloudProviderFactory interface {
	Create(config ProviderConfig) (CloudProvider, error)
	SupportedTypes() []string
}