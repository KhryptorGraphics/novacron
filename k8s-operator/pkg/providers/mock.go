package providers

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	novacronv1 "github.com/khryptorgraphics/novacron/k8s-operator/pkg/apis/novacron/v1"
)

// MockProvider implements CloudProvider for testing
type MockProvider struct {
	name    string
	regions []string
	vms     map[string]*VMResult
}

// NewMockProvider creates a new mock provider
func NewMockProvider(config ProviderConfig) (CloudProvider, error) {
	return &MockProvider{
		name:    config.Name,
		regions: []string{"us-east-1", "us-west-2", "eu-west-1"},
		vms:     make(map[string]*VMResult),
	}, nil
}

// GetName returns the provider name
func (p *MockProvider) GetName() string {
	return p.name
}

// CreateVM creates a mock virtual machine
func (p *MockProvider) CreateVM(ctx context.Context, req *VMRequest) (*VMResult, error) {
	vmID := fmt.Sprintf("mock-%s-%d", req.Name, time.Now().Unix())
	
	vm := &VMResult{
		ID:           vmID,
		Name:         req.Name,
		Status:       "running",
		InstanceType: req.InstanceType,
		Region:       req.Region,
		IPAddress:    fmt.Sprintf("10.0.%d.%d", time.Now().Unix()%256, time.Now().Unix()%256),
		PrivateIP:    fmt.Sprintf("192.168.%d.%d", time.Now().Unix()%256, time.Now().Unix()%256),
		LaunchTime:   time.Now().Format(time.RFC3339),
		Tags:         req.Tags,
	}
	
	p.vms[vmID] = vm
	return vm, nil
}

// GetVM retrieves mock VM information
func (p *MockProvider) GetVM(ctx context.Context, vmID string) (*VMResult, error) {
	vm, exists := p.vms[vmID]
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}
	return vm, nil
}

// DeleteVM deletes a mock virtual machine
func (p *MockProvider) DeleteVM(ctx context.Context, vmID string) error {
	if _, exists := p.vms[vmID]; !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}
	delete(p.vms, vmID)
	return nil
}

// ListVMs lists all mock VMs
func (p *MockProvider) ListVMs(ctx context.Context, filters map[string]string) ([]*VMResult, error) {
	var vms []*VMResult
	for _, vm := range p.vms {
		// Apply filters
		match := true
		for key, value := range filters {
			if vm.Tags[key] != value {
				match = false
				break
			}
		}
		if match {
			vms = append(vms, vm)
		}
	}
	return vms, nil
}

// EstimateCost estimates mock cost
func (p *MockProvider) EstimateCost(region string, resources ResourceRequirements) (*novacronv1.ResourceCost, error) {
	// Simple mock cost calculation
	cpuCost := p.calculateCPUCost(resources.CPU)
	memoryCost := p.calculateMemoryCost(resources.Memory)
	storageCost := p.calculateStorageCost(resources.Storage)
	
	hourlyCost := cpuCost + memoryCost + storageCost
	
	// Regional multipliers
	switch region {
	case "us-east-1":
		hourlyCost *= 1.0 // Base cost
	case "us-west-2":
		hourlyCost *= 1.1 // 10% more expensive
	case "eu-west-1":
		hourlyCost *= 1.2 // 20% more expensive
	default:
		hourlyCost *= 1.0
	}
	
	return &novacronv1.ResourceCost{
		Currency:   "USD",
		HourlyCost: hourlyCost,
		TotalCost:  hourlyCost, // Will be calculated over time
		Breakdown: map[string]float64{
			"cpu":     cpuCost,
			"memory":  memoryCost,
			"storage": storageCost,
		},
	}, nil
}

func (p *MockProvider) calculateCPUCost(cpu string) float64 {
	// Parse CPU requirement (e.g., "2", "1000m")
	if strings.HasSuffix(cpu, "m") {
		// Millicores
		cpuStr := strings.TrimSuffix(cpu, "m")
		millicores, _ := strconv.ParseFloat(cpuStr, 64)
		return (millicores / 1000) * 0.05 // $0.05 per core per hour
	}
	
	cores, _ := strconv.ParseFloat(cpu, 64)
	return cores * 0.05 // $0.05 per core per hour
}

func (p *MockProvider) calculateMemoryCost(memory string) float64 {
	// Parse memory requirement (e.g., "4Gi", "4096Mi")
	var gb float64
	
	if strings.HasSuffix(memory, "Gi") {
		memStr := strings.TrimSuffix(memory, "Gi")
		gb, _ = strconv.ParseFloat(memStr, 64)
	} else if strings.HasSuffix(memory, "Mi") {
		memStr := strings.TrimSuffix(memory, "Mi")
		mi, _ := strconv.ParseFloat(memStr, 64)
		gb = mi / 1024
	} else if strings.HasSuffix(memory, "G") {
		memStr := strings.TrimSuffix(memory, "G")
		gb, _ = strconv.ParseFloat(memStr, 64)
	}
	
	return gb * 0.01 // $0.01 per GB per hour
}

func (p *MockProvider) calculateStorageCost(storage string) float64 {
	// Parse storage requirement
	var gb float64
	
	if strings.HasSuffix(storage, "Gi") {
		storStr := strings.TrimSuffix(storage, "Gi")
		gb, _ = strconv.ParseFloat(storStr, 64)
	} else if strings.HasSuffix(storage, "G") {
		storStr := strings.TrimSuffix(storage, "G")
		gb, _ = strconv.ParseFloat(storStr, 64)
	}
	
	return gb * 0.001 // $0.001 per GB per hour
}

// GetAvailableRegions returns mock available regions
func (p *MockProvider) GetAvailableRegions(ctx context.Context) ([]string, error) {
	return p.regions, nil
}

// GetAvailableInstanceTypes returns mock available instance types
func (p *MockProvider) GetAvailableInstanceTypes(ctx context.Context, region string) ([]InstanceType, error) {
	return []InstanceType{
		{
			Name:        "t3.micro",
			CPU:         1,
			Memory:      1024,
			Storage:     20,
			GPU:         0,
			Network:     "low",
			HourlyCost:  0.0104,
			Description: "Burstable performance instance",
		},
		{
			Name:        "t3.small",
			CPU:         2,
			Memory:      2048,
			Storage:     20,
			GPU:         0,
			Network:     "low",
			HourlyCost:  0.0208,
			Description: "Burstable performance instance",
		},
		{
			Name:        "m5.large",
			CPU:         2,
			Memory:      8192,
			Storage:     20,
			GPU:         0,
			Network:     "moderate",
			HourlyCost:  0.096,
			Description: "General purpose instance",
		},
		{
			Name:        "c5.xlarge",
			CPU:         4,
			Memory:      8192,
			Storage:     20,
			GPU:         0,
			Network:     "high",
			HourlyCost:  0.17,
			Description: "Compute optimized instance",
		},
	}, nil
}

// MigrateVM performs a mock VM migration
func (p *MockProvider) MigrateVM(ctx context.Context, vmID string, target MigrationTarget) error {
	vm, exists := p.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}
	
	// Update VM with new target information
	if target.Region != "" {
		vm.Region = target.Region
	}
	if target.InstanceType != "" {
		vm.InstanceType = target.InstanceType
	}
	
	// Simulate migration time
	time.Sleep(100 * time.Millisecond)
	
	return nil
}

// GetVMMetrics retrieves mock performance metrics
func (p *MockProvider) GetVMMetrics(ctx context.Context, vmID string) (*VMMetrics, error) {
	if _, exists := p.vms[vmID]; !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}
	
	// Generate mock metrics
	return &VMMetrics{
		Timestamp:       time.Now().Format(time.RFC3339),
		CPUUsage:        45.0 + float64(time.Now().Unix()%20), // 45-65%
		MemoryUsage:     60.0 + float64(time.Now().Unix()%15), // 60-75%
		NetworkIn:       1024000.0,  // 1 MB/s
		NetworkOut:      512000.0,   // 512 KB/s
		DiskRead:        100.0,      // 100 IOPS
		DiskWrite:       50.0,       // 50 IOPS
		CPUCreditUsage:  80.0,       // 80% credit usage
	}, nil
}