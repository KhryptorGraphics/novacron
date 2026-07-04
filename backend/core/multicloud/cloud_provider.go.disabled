package multicloud

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// CloudProvider represents a cloud provider interface
type CloudProvider interface {
	// VM Operations
	CreateVM(ctx context.Context, spec *VMSpec) (*VM, error)
	DeleteVM(ctx context.Context, vmID string) error
	StartVM(ctx context.Context, vmID string) error
	StopVM(ctx context.Context, vmID string) error
	GetVM(ctx context.Context, vmID string) (*VM, error)
	ListVMs(ctx context.Context) ([]*VM, error)
	
	// Migration Operations
	ExportVM(ctx context.Context, vmID string) (*VMImage, error)
	ImportVM(ctx context.Context, image *VMImage) (*VM, error)
	
	// Resource Operations
	GetResources(ctx context.Context) (*Resources, error)
	GetPricing(ctx context.Context) (*Pricing, error)
	
	// Provider Info
	GetProviderName() string
	GetRegion() string
	IsAvailable(ctx context.Context) bool
}

// VMSpec defines VM specifications
type VMSpec struct {
	Name        string
	CPU         int
	Memory      int64 // in bytes
	Disk        int64 // in bytes
	Image       string
	Network     string
	Tags        map[string]string
	Metadata    map[string]string
}

// VM represents a virtual machine
type VM struct {
	ID           string
	Name         string
	Provider     string
	Region       string
	Status       VMStatus
	CPU          int
	Memory       int64
	Disk         int64
	PublicIP     string
	PrivateIP    string
	CreatedAt    time.Time
	UpdatedAt    time.Time
	Tags         map[string]string
	Metadata     map[string]string
}

// VMStatus represents VM status
type VMStatus string

const (
	VMStatusPending   VMStatus = "pending"
	VMStatusRunning   VMStatus = "running"
	VMStatusStopped   VMStatus = "stopped"
	VMStatusDeleted   VMStatus = "deleted"
	VMStatusMigrating VMStatus = "migrating"
	VMStatusError     VMStatus = "error"
)

// VMImage represents a VM image for migration
type VMImage struct {
	ID          string
	VMID        string
	Size        int64
	Format      string
	Compressed  bool
	Encrypted   bool
	Checksum    string
	Data        []byte
	Metadata    map[string]string
	CreatedAt   time.Time
}

// Resources represents cloud resources
type Resources struct {
	TotalCPU      int
	AvailableCPU  int
	TotalMemory   int64
	AvailableMemory int64
	TotalDisk     int64
	AvailableDisk int64
	VMCount       int
	MaxVMs        int
}

// Pricing represents cloud pricing
type Pricing struct {
	CPUPerHour    float64
	MemoryPerGB   float64
	DiskPerGB     float64
	NetworkPerGB  float64
	Currency      string
}

// MultiCloudManager manages multiple cloud providers
type MultiCloudManager struct {
	mu         sync.RWMutex
	providers  map[string]CloudProvider
	vmRegistry map[string]*VM // vmID -> VM
	config     *MultiCloudConfig
}

// MultiCloudConfig configuration for multi-cloud
type MultiCloudConfig struct {
	EnableCostOptimization bool
	EnableAutoFailover     bool
	PreferredProvider      string
	MaxProvidersPerVM      int
	SyncInterval           time.Duration
}

// NewMultiCloudManager creates a new multi-cloud manager
func NewMultiCloudManager(config *MultiCloudConfig) *MultiCloudManager {
	return &MultiCloudManager{
		providers:  make(map[string]CloudProvider),
		vmRegistry: make(map[string]*VM),
		config:     config,
	}
}

// RegisterProvider registers a cloud provider
func (mcm *MultiCloudManager) RegisterProvider(provider CloudProvider) error {
	mcm.mu.Lock()
	defer mcm.mu.Unlock()
	
	name := provider.GetProviderName()
	if _, exists := mcm.providers[name]; exists {
		return fmt.Errorf("provider %s already registered", name)
	}
	
	mcm.providers[name] = provider
	return nil
}

// GetProvider returns a cloud provider by name
func (mcm *MultiCloudManager) GetProvider(name string) (CloudProvider, error) {
	mcm.mu.RLock()
	defer mcm.mu.RUnlock()
	
	provider, exists := mcm.providers[name]
	if !exists {
		return nil, fmt.Errorf("provider %s not found", name)
	}
	
	return provider, nil
}

// ListProviders returns all registered providers
func (mcm *MultiCloudManager) ListProviders() []CloudProvider {
	mcm.mu.RLock()
	defer mcm.mu.RUnlock()
	
	providers := make([]CloudProvider, 0, len(mcm.providers))
	for _, provider := range mcm.providers {
		providers = append(providers, provider)
	}
	
	return providers
}

// SelectOptimalProvider selects the best provider based on criteria
func (mcm *MultiCloudManager) SelectOptimalProvider(ctx context.Context, spec *VMSpec) (CloudProvider, error) {
	providers := mcm.ListProviders()
	if len(providers) == 0 {
		return nil, fmt.Errorf("no providers available")
	}
	
	// Score each provider
	type providerScore struct {
		provider CloudProvider
		score    float64
	}
	
	scores := make([]providerScore, 0, len(providers))
	
	for _, provider := range providers {
		if !provider.IsAvailable(ctx) {
			continue
		}
		
		score := mcm.calculateProviderScore(ctx, provider, spec)
		scores = append(scores, providerScore{provider, score})
	}
	
	if len(scores) == 0 {
		return nil, fmt.Errorf("no available providers")
	}
	
	// Find provider with highest score
	best := scores[0]
	for _, ps := range scores[1:] {
		if ps.score > best.score {
			best = ps
		}
	}
	
	return best.provider, nil
}

// calculateProviderScore calculates a score for a provider
func (mcm *MultiCloudManager) calculateProviderScore(ctx context.Context, provider CloudProvider, spec *VMSpec) float64 {
	score := 0.0

	// Get resources
	resources, err := provider.GetResources(ctx)
	if err != nil {
		return 0.0
	}

	// Get pricing
	pricing, err := provider.GetPricing(ctx)
	if err != nil {
		return 0.0
	}

	// Resource availability (40% weight)
	cpuAvailability := float64(resources.AvailableCPU) / float64(resources.TotalCPU)
	memoryAvailability := float64(resources.AvailableMemory) / float64(resources.TotalMemory)
	resourceScore := (cpuAvailability + memoryAvailability) / 2.0
	score += resourceScore * 0.4

	// Cost optimization (40% weight)
	if mcm.config.EnableCostOptimization {
		// Lower cost = higher score
		estimatedCost := pricing.CPUPerHour*float64(spec.CPU) +
			pricing.MemoryPerGB*float64(spec.Memory)/(1024*1024*1024)
		// Normalize cost (assuming max cost of $10/hour)
		costScore := 1.0 - (estimatedCost / 10.0)
		if costScore < 0 {
			costScore = 0
		}
		score += costScore * 0.4
	} else {
		score += 0.4 // Neutral score if cost optimization disabled
	}

	// Preferred provider bonus (20% weight)
	if provider.GetProviderName() == mcm.config.PreferredProvider {
		score += 0.2
	}

	return score
}

// MigrateVMCrossCloud migrates a VM across cloud providers
func (mcm *MultiCloudManager) MigrateVMCrossCloud(ctx context.Context, vmID, targetProvider string) error {
	mcm.mu.Lock()
	vm, exists := mcm.vmRegistry[vmID]
	if !exists {
		mcm.mu.Unlock()
		return fmt.Errorf("VM %s not found", vmID)
	}
	mcm.mu.Unlock()

	// Get source and target providers
	sourceProvider, err := mcm.GetProvider(vm.Provider)
	if err != nil {
		return fmt.Errorf("source provider not found: %w", err)
	}

	targetProv, err := mcm.GetProvider(targetProvider)
	if err != nil {
		return fmt.Errorf("target provider not found: %w", err)
	}

	// Update VM status
	vm.Status = VMStatusMigrating

	// Export VM from source
	image, err := sourceProvider.ExportVM(ctx, vmID)
	if err != nil {
		vm.Status = VMStatusError
		return fmt.Errorf("failed to export VM: %w", err)
	}

	// Import VM to target
	newVM, err := targetProv.ImportVM(ctx, image)
	if err != nil {
		vm.Status = VMStatusError
		return fmt.Errorf("failed to import VM: %w", err)
	}

	// Delete VM from source
	if err := sourceProvider.DeleteVM(ctx, vmID); err != nil {
		// Log error but don't fail migration
		fmt.Printf("Warning: failed to delete source VM: %v\n", err)
	}

	// Update registry
	mcm.mu.Lock()
	mcm.vmRegistry[newVM.ID] = newVM
	delete(mcm.vmRegistry, vmID)
	mcm.mu.Unlock()

	return nil
}

// GetUnifiedResources returns aggregated resources across all providers
func (mcm *MultiCloudManager) GetUnifiedResources(ctx context.Context) (*Resources, error) {
	providers := mcm.ListProviders()

	unified := &Resources{}

	for _, provider := range providers {
		resources, err := provider.GetResources(ctx)
		if err != nil {
			continue
		}

		unified.TotalCPU += resources.TotalCPU
		unified.AvailableCPU += resources.AvailableCPU
		unified.TotalMemory += resources.TotalMemory
		unified.AvailableMemory += resources.AvailableMemory
		unified.TotalDisk += resources.TotalDisk
		unified.AvailableDisk += resources.AvailableDisk
		unified.VMCount += resources.VMCount
		unified.MaxVMs += resources.MaxVMs
	}

	return unified, nil
}

// OptimizeCosts analyzes and optimizes costs across providers
func (mcm *MultiCloudManager) OptimizeCosts(ctx context.Context) ([]string, error) {
	if !mcm.config.EnableCostOptimization {
		return nil, fmt.Errorf("cost optimization disabled")
	}

	recommendations := []string{}

	// Get all VMs
	mcm.mu.RLock()
	vms := make([]*VM, 0, len(mcm.vmRegistry))
	for _, vm := range mcm.vmRegistry {
		vms = append(vms, vm)
	}
	mcm.mu.RUnlock()

	// Analyze each VM
	for _, vm := range vms {
		// Create spec from VM
		spec := &VMSpec{
			CPU:    vm.CPU,
			Memory: vm.Memory,
			Disk:   vm.Disk,
		}

		// Find optimal provider
		optimalProvider, err := mcm.SelectOptimalProvider(ctx, spec)
		if err != nil {
			continue
		}

		// If different from current, recommend migration
		if optimalProvider.GetProviderName() != vm.Provider {
			recommendations = append(recommendations,
				fmt.Sprintf("Migrate VM %s from %s to %s for cost savings",
					vm.ID, vm.Provider, optimalProvider.GetProviderName()))
		}
	}

	return recommendations, nil
}

// SyncVMRegistry synchronizes VM registry with all providers
func (mcm *MultiCloudManager) SyncVMRegistry(ctx context.Context) error {
	providers := mcm.ListProviders()

	newRegistry := make(map[string]*VM)

	for _, provider := range providers {
		vms, err := provider.ListVMs(ctx)
		if err != nil {
			continue
		}

		for _, vm := range vms {
			newRegistry[vm.ID] = vm
		}
	}

	mcm.mu.Lock()
	mcm.vmRegistry = newRegistry
	mcm.mu.Unlock()

	return nil
}

// StartAutoSync starts automatic VM registry synchronization
func (mcm *MultiCloudManager) StartAutoSync(ctx context.Context) {
	ticker := time.NewTicker(mcm.config.SyncInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := mcm.SyncVMRegistry(ctx); err != nil {
				fmt.Printf("Sync error: %v\n", err)
			}
		}
	}
}


