package providers

import (
	"context"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/federation/multicloud"
)

// AWSProvider implements CloudProvider for Amazon Web Services
type AWSProvider struct {
	config      multicloud.CloudProviderConfig
	regions     []string
	credentials map[string]string
}

// NewAWSProvider creates a new AWS provider
func NewAWSProvider() *AWSProvider {
	return &AWSProvider{
		regions: []string{
			"us-east-1", "us-east-2", "us-west-1", "us-west-2",
			"eu-west-1", "eu-west-2", "eu-central-1",
			"ap-southeast-1", "ap-southeast-2", "ap-northeast-1",
		},
	}
}

// Provider Information
func (p *AWSProvider) GetProviderType() multicloud.CloudProviderType {
	return multicloud.ProviderAWS
}

func (p *AWSProvider) GetName() string {
	return "Amazon Web Services"
}

func (p *AWSProvider) GetRegions() []string {
	return p.regions
}

func (p *AWSProvider) GetAvailabilityZones(region string) []string {
	// Return mock availability zones based on region
	switch region {
	case "us-east-1":
		return []string{"us-east-1a", "us-east-1b", "us-east-1c", "us-east-1d", "us-east-1f"}
	case "us-west-2":
		return []string{"us-west-2a", "us-west-2b", "us-west-2c", "us-west-2d"}
	case "eu-west-1":
		return []string{"eu-west-1a", "eu-west-1b", "eu-west-1c"}
	default:
		return []string{region + "a", region + "b", region + "c"}
	}
}

// Authentication and Configuration
func (p *AWSProvider) Initialize(ctx context.Context, config multicloud.CloudProviderConfig) error {
	p.config = config
	p.credentials = config.Credentials

	// Validate required credentials
	if _, ok := p.credentials["access_key_id"]; !ok {
		return fmt.Errorf("access_key_id is required for AWS provider")
	}
	if _, ok := p.credentials["secret_access_key"]; !ok {
		return fmt.Errorf("secret_access_key is required for AWS provider")
	}

	return nil
}

func (p *AWSProvider) Validate(ctx context.Context) error {
	// In a real implementation, this would test AWS API connectivity
	// For now, just validate configuration
	if p.config.DefaultRegion == "" {
		return fmt.Errorf("default region not set")
	}

	validRegion := false
	for _, region := range p.regions {
		if region == p.config.DefaultRegion {
			validRegion = true
			break
		}
	}
	if !validRegion {
		return fmt.Errorf("invalid default region: %s", p.config.DefaultRegion)
	}

	return nil
}

func (p *AWSProvider) GetCapabilities() []multicloud.CloudCapability {
	return []multicloud.CloudCapability{
		multicloud.CapabilityVMLiveMigration,
		multicloud.CapabilityAutoScaling,
		multicloud.CapabilityLoadBalancing,
		multicloud.CapabilityBlockStorage,
		multicloud.CapabilityObjectStorage,
		multicloud.CapabilityContainerRegistry,
		multicloud.CapabilityKubernetes,
		multicloud.CapabilitySpotInstances,
		multicloud.CapabilityReservedInstances,
		multicloud.CapabilityGPUCompute,
		multicloud.CapabilityDatabaseServices,
		multicloud.CapabilityMLServices,
		multicloud.CapabilityNetworkACLs,
		multicloud.CapabilityPrivateNetworking,
		multicloud.CapabilityVPNGateway,
		multicloud.CapabilityDirectConnect,
		multicloud.CapabilityIdentityManagement,
		multicloud.CapabilityKeyManagement,
		multicloud.CapabilityAuditLogging,
		multicloud.CapabilityBackupServices,
		multicloud.CapabilityDisasterRecovery,
	}
}

// VM Operations
func (p *AWSProvider) CreateVM(ctx context.Context, request *multicloud.VMCreateRequest) (*multicloud.VMInstance, error) {
	// In a real implementation, this would use AWS SDK to create EC2 instance
	// For now, return a mock VM instance
	
	vmID := fmt.Sprintf("i-%d", time.Now().Unix())
	
	vm := &multicloud.VMInstance{
		ID:               vmID,
		Name:             request.Name,
		Provider:         multicloud.ProviderAWS,
		Region:           request.Region,
		AvailabilityZone: request.AvailabilityZone,
		InstanceType:     request.InstanceType,
		State:            multicloud.VMStatePending,
		ImageID:          request.ImageID,
		KeyPair:          request.KeyPair,
		SecurityGroups:   request.SecurityGroups,
		Tags:             request.Tags,
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
		CPU:              p.getInstanceTypeCPU(request.InstanceType),
		Memory:           p.getInstanceTypeMemory(request.InstanceType),
		Storage:          request.Storage,
		NetworkBandwidth: p.getInstanceTypeNetworkBandwidth(request.InstanceType),
		HourlyRate:       p.getInstanceTypeHourlyRate(request.InstanceType, request.Region),
	}

	vm.MonthlyEstimate = vm.HourlyRate * 24 * 30

	// Set public/private IPs (mock)
	vm.PrivateIP = "10.0.1.100"
	if !request.Tags["private_only"] == "true" {
		vm.PublicIP = "54.123.45.67"
	}

	// Simulate instance startup delay
	go func() {
		time.Sleep(2 * time.Second)
		vm.State = multicloud.VMStateRunning
		vm.UpdatedAt = time.Now()
	}()

	return vm, nil
}

func (p *AWSProvider) GetVM(ctx context.Context, vmID string) (*multicloud.VMInstance, error) {
	// In a real implementation, this would call AWS DescribeInstances
	// For now, return mock data
	
	return &multicloud.VMInstance{
		ID:           vmID,
		Name:         "test-vm",
		Provider:     multicloud.ProviderAWS,
		Region:       p.config.DefaultRegion,
		InstanceType: "t3.micro",
		State:        multicloud.VMStateRunning,
		PublicIP:     "54.123.45.67",
		PrivateIP:    "10.0.1.100",
		CreatedAt:    time.Now().Add(-time.Hour),
		UpdatedAt:    time.Now(),
	}, nil
}

func (p *AWSProvider) ListVMs(ctx context.Context, filters map[string]string) ([]*multicloud.VMInstance, error) {
	// Mock implementation - return sample VMs
	vms := []*multicloud.VMInstance{
		{
			ID:           "i-1234567890abcdef0",
			Name:         "web-server",
			Provider:     multicloud.ProviderAWS,
			Region:       "us-east-1",
			InstanceType: "t3.medium",
			State:        multicloud.VMStateRunning,
			PublicIP:     "54.123.45.67",
			PrivateIP:    "10.0.1.100",
			CreatedAt:    time.Now().Add(-2 * time.Hour),
			UpdatedAt:    time.Now(),
		},
		{
			ID:           "i-abcdef1234567890",
			Name:         "database-server",
			Provider:     multicloud.ProviderAWS,
			Region:       "us-east-1",
			InstanceType: "r5.large",
			State:        multicloud.VMStateRunning,
			PublicIP:     "",
			PrivateIP:    "10.0.1.101",
			CreatedAt:    time.Now().Add(-4 * time.Hour),
			UpdatedAt:    time.Now(),
		},
	}

	// Apply filters
	return p.applyVMFilters(vms, filters), nil
}

func (p *AWSProvider) UpdateVM(ctx context.Context, vmID string, updates *multicloud.VMUpdateRequest) error {
	// In a real implementation, this would call AWS ModifyInstanceAttribute
	return nil
}

func (p *AWSProvider) DeleteVM(ctx context.Context, vmID string) error {
	// In a real implementation, this would call AWS TerminateInstances
	return nil
}

// VM Lifecycle
func (p *AWSProvider) StartVM(ctx context.Context, vmID string) error {
	// In a real implementation, this would call AWS StartInstances
	return nil
}

func (p *AWSProvider) StopVM(ctx context.Context, vmID string) error {
	// In a real implementation, this would call AWS StopInstances
	return nil
}

func (p *AWSProvider) RestartVM(ctx context.Context, vmID string) error {
	// In a real implementation, this would call AWS RebootInstances
	return nil
}

func (p *AWSProvider) SuspendVM(ctx context.Context, vmID string) error {
	// AWS doesn't support suspend, return error
	return fmt.Errorf("suspend operation not supported on AWS")
}

func (p *AWSProvider) ResumeVM(ctx context.Context, vmID string) error {
	// AWS doesn't support resume, return error
	return fmt.Errorf("resume operation not supported on AWS")
}

// Migration Support
func (p *AWSProvider) ExportVM(ctx context.Context, vmID string, format multicloud.VMExportFormat) (*multicloud.VMExportData, error) {
	// In a real implementation, this would create AMI and export it
	return &multicloud.VMExportData{
		Format:      format,
		Size:        10 * 1024 * 1024 * 1024, // 10GB
		Checksum:    "sha256:abc123...",
		DownloadURL: fmt.Sprintf("https://s3.amazonaws.com/exports/%s.%s", vmID, format),
		ExpiresAt:   time.Now().Add(24 * time.Hour),
		Metadata: map[string]interface{}{
			"ami_id":        "ami-12345678",
			"instance_type": "t3.micro",
		},
	}, nil
}

func (p *AWSProvider) ImportVM(ctx context.Context, data *multicloud.VMExportData) (*multicloud.VMInstance, error) {
	// In a real implementation, this would import AMI and create instance
	vmID := fmt.Sprintf("i-import-%d", time.Now().Unix())
	
	return &multicloud.VMInstance{
		ID:        vmID,
		Name:      "imported-vm",
		Provider:  multicloud.ProviderAWS,
		Region:    p.config.DefaultRegion,
		State:     multicloud.VMStatePending,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}, nil
}

func (p *AWSProvider) SnapshotVM(ctx context.Context, vmID string, name string) (*multicloud.VMSnapshot, error) {
	// In a real implementation, this would create EBS snapshots
	return &multicloud.VMSnapshot{
		ID:          fmt.Sprintf("snap-%d", time.Now().Unix()),
		Name:        name,
		VMID:        vmID,
		Description: fmt.Sprintf("Snapshot of %s", vmID),
		Size:        8 * 1024 * 1024 * 1024, // 8GB
		State:       "completed",
		Progress:    100,
		CreatedAt:   time.Now(),
	}, nil
}

func (p *AWSProvider) RestoreSnapshot(ctx context.Context, snapshotID string) error {
	// In a real implementation, this would restore from EBS snapshot
	return nil
}

// Resource Management
func (p *AWSProvider) GetResourceQuota(ctx context.Context) (*multicloud.ResourceQuota, error) {
	// In a real implementation, this would call AWS Service Quotas API
	return &multicloud.ResourceQuota{
		MaxVMs:            1000,
		MaxCPU:            5000,
		MaxMemory:         1024 * 1024, // 1TB in MB
		MaxStorage:        10 * 1024,   // 10TB in GB
		MaxNetworks:       5,
		MaxSecurityGroups: 500,
		MaxSnapshots:      10000,
		MaxFloatingIPs:    5,
		MaxLoadBalancers:  20,
	}, nil
}

func (p *AWSProvider) GetResourceUsage(ctx context.Context) (*multicloud.ResourceUsage, error) {
	// In a real implementation, this would aggregate actual usage from AWS APIs
	return &multicloud.ResourceUsage{
		UsedVMs:           15,
		UsedCPU:           45,
		UsedMemory:        128 * 1024, // 128GB in MB
		UsedStorage:       500,        // 500GB
		UsedNetworks:      2,
		UsedSecurityGroups: 8,
		UsedSnapshots:     25,
		UsedFloatingIPs:   2,
		UsedLoadBalancers: 1,
		TotalCost:         1250.75,
	}, nil
}

func (p *AWSProvider) GetPricing(ctx context.Context, resourceType string, region string) (*multicloud.PricingInfo, error) {
	// Mock pricing data - in real implementation, would use AWS Pricing API
	switch resourceType {
	case "vm":
		return &multicloud.PricingInfo{
			ResourceType: resourceType,
			Region:       region,
			Currency:     "USD",
			PricePerHour: 0.0116, // t3.micro price
			Unit:         "hour",
			TierPricing: []multicloud.PricingTier{
				{From: 0, To: 744, Price: 0.0116},   // First 744 hours (month)
				{From: 744, To: -1, Price: 0.0104}, // Additional hours
			},
			SpotPricing: &multicloud.SpotPricingInfo{
				CurrentPrice: 0.0035,
				AveragePrice: 0.0040,
				MinPrice:     0.0025,
				MaxPrice:     0.0116,
				LastUpdated:  time.Now(),
			},
		}, nil
	case "storage":
		return &multicloud.PricingInfo{
			ResourceType: resourceType,
			Region:       region,
			Currency:     "USD",
			PricePerMonth: 0.10, // EBS gp2 price per GB/month
			Unit:          "GB/month",
		}, nil
	default:
		return nil, fmt.Errorf("pricing not available for resource type: %s", resourceType)
	}
}

// Networking (simplified implementation)
func (p *AWSProvider) CreateNetwork(ctx context.Context, request *multicloud.NetworkCreateRequest) (*multicloud.Network, error) {
	return &multicloud.Network{
		ID:        fmt.Sprintf("vpc-%d", time.Now().Unix()),
		Name:      request.Name,
		CIDR:      request.CIDR,
		Region:    request.Region,
		State:     "available",
		Tags:      request.Tags,
		CreatedAt: time.Now(),
	}, nil
}

func (p *AWSProvider) GetNetwork(ctx context.Context, networkID string) (*multicloud.Network, error) {
	return &multicloud.Network{
		ID:        networkID,
		Name:      "default-vpc",
		CIDR:      "10.0.0.0/16",
		Region:    p.config.DefaultRegion,
		State:     "available",
		CreatedAt: time.Now().Add(-24 * time.Hour),
	}, nil
}

func (p *AWSProvider) ListNetworks(ctx context.Context) ([]*multicloud.Network, error) {
	return []*multicloud.Network{
		{
			ID:        "vpc-12345678",
			Name:      "default",
			CIDR:      "10.0.0.0/16",
			Region:    p.config.DefaultRegion,
			State:     "available",
			CreatedAt: time.Now().Add(-24 * time.Hour),
		},
	}, nil
}

func (p *AWSProvider) DeleteNetwork(ctx context.Context, networkID string) error {
	return nil
}

// Storage (simplified implementation)
func (p *AWSProvider) CreateStorage(ctx context.Context, request *multicloud.StorageCreateRequest) (*multicloud.Storage, error) {
	return &multicloud.Storage{
		ID:        fmt.Sprintf("vol-%d", time.Now().Unix()),
		Name:      request.Name,
		Type:      request.Type,
		Size:      request.Size,
		IOPS:      request.IOPS,
		Encrypted: request.Encrypted,
		Region:    request.Region,
		State:     "available",
		Tags:      request.Tags,
		CreatedAt: time.Now(),
	}, nil
}

func (p *AWSProvider) GetStorage(ctx context.Context, storageID string) (*multicloud.Storage, error) {
	return &multicloud.Storage{
		ID:        storageID,
		Name:      "root-volume",
		Type:      multicloud.StorageTypeBlockSSD,
		Size:      20,
		Encrypted: true,
		Region:    p.config.DefaultRegion,
		State:     "in-use",
		CreatedAt: time.Now().Add(-2 * time.Hour),
	}, nil
}

func (p *AWSProvider) ListStorage(ctx context.Context) ([]*multicloud.Storage, error) {
	return []*multicloud.Storage{
		{
			ID:        "vol-12345678",
			Name:      "root-volume",
			Type:      multicloud.StorageTypeBlockSSD,
			Size:      20,
			Encrypted: true,
			Region:    p.config.DefaultRegion,
			State:     "in-use",
			CreatedAt: time.Now().Add(-2 * time.Hour),
		},
	}, nil
}

func (p *AWSProvider) DeleteStorage(ctx context.Context, storageID string) error {
	return nil
}

// Monitoring and Health
func (p *AWSProvider) GetVMMetrics(ctx context.Context, vmID string, start, end time.Time) (*multicloud.VMMetrics, error) {
	// Mock metrics data
	return &multicloud.VMMetrics{
		VMID:      vmID,
		StartTime: start,
		EndTime:   end,
		Metrics: map[string][]multicloud.MetricPoint{
			"cpu_utilization": {
				{Timestamp: start, Value: 15.5},
				{Timestamp: start.Add(time.Minute), Value: 18.2},
				{Timestamp: start.Add(2 * time.Minute), Value: 12.8},
			},
			"memory_utilization": {
				{Timestamp: start, Value: 65.2},
				{Timestamp: start.Add(time.Minute), Value: 67.1},
				{Timestamp: start.Add(2 * time.Minute), Value: 64.8},
			},
		},
	}, nil
}

func (p *AWSProvider) GetProviderHealth(ctx context.Context) (*multicloud.ProviderHealthStatus, error) {
	return &multicloud.ProviderHealthStatus{
		Provider: multicloud.ProviderAWS,
		Overall:  multicloud.HealthStatusHealthy,
		Services: map[string]multicloud.HealthStatus{
			"ec2":     multicloud.HealthStatusHealthy,
			"ebs":     multicloud.HealthStatusHealthy,
			"vpc":     multicloud.HealthStatusHealthy,
			"s3":      multicloud.HealthStatusHealthy,
		},
		Regions: map[string]multicloud.HealthStatus{
			"us-east-1": multicloud.HealthStatusHealthy,
			"us-west-2": multicloud.HealthStatusHealthy,
			"eu-west-1": multicloud.HealthStatusHealthy,
		},
		LastChecked: time.Now(),
	}, nil
}

// Cost Management
func (p *AWSProvider) GetCostEstimate(ctx context.Context, request *multicloud.CostEstimateRequest) (*multicloud.CostEstimate, error) {
	// Mock cost estimation
	totalCost := 0.0
	var breakdown []multicloud.CostBreakdown

	for _, resource := range request.Resources {
		switch resource.Type {
		case "vm":
			unitCost := 0.0116 // t3.micro hourly rate
			cost := unitCost * float64(resource.Quantity)
			totalCost += cost
			breakdown = append(breakdown, multicloud.CostBreakdown{
				ResourceType: resource.Type,
				Cost:         cost,
				Quantity:     resource.Quantity,
				UnitCost:     unitCost,
			})
		}
	}

	return &multicloud.CostEstimate{
		TotalCost: totalCost,
		Currency:  "USD",
		Duration:  request.Duration,
		Breakdown: breakdown,
		Confidence: 0.95,
		CreatedAt:  time.Now(),
	}, nil
}

func (p *AWSProvider) GetBillingData(ctx context.Context, start, end time.Time) (*multicloud.BillingData, error) {
	// Mock billing data
	return &multicloud.BillingData{
		Provider:  multicloud.ProviderAWS,
		StartTime: start,
		EndTime:   end,
		TotalCost: 1250.75,
		Currency:  "USD",
		Resources: []multicloud.ResourceBilling{
			{
				ResourceID:   "i-1234567890abcdef0",
				ResourceType: "vm",
				Usage:        744, // hours in a month
				UnitCost:     0.0116,
				TotalCost:    86.30,
			},
			{
				ResourceID:   "vol-12345678",
				ResourceType: "storage",
				Usage:        20, // GB
				UnitCost:     0.10,
				TotalCost:    2.00,
			},
		},
	}, nil
}

// Compliance and Security
func (p *AWSProvider) GetComplianceStatus(ctx context.Context) (*multicloud.ComplianceStatus, error) {
	return &multicloud.ComplianceStatus{
		Provider:     multicloud.ProviderAWS,
		OverallScore: 95.5,
		Compliances: []multicloud.ComplianceFramework{
			{
				Name:          "SOC2",
				Version:       "2017",
				Score:         98.0,
				Status:        "compliant",
				Controls:      150,
				Passed:        147,
				Failed:        3,
				NotApplicable: 0,
			},
			{
				Name:          "HIPAA",
				Version:       "2013",
				Score:         94.0,
				Status:        "compliant",
				Controls:      78,
				Passed:        73,
				Failed:        5,
				NotApplicable: 0,
			},
			{
				Name:          "GDPR",
				Version:       "2018",
				Score:         92.0,
				Status:        "compliant",
				Controls:      99,
				Passed:        91,
				Failed:        8,
				NotApplicable: 0,
			},
		},
		DataResidency: multicloud.DataResidencyInfo{
			PrimaryRegion:       "us-east-1",
			AllowedRegions:      []string{"us-east-1", "us-west-2", "eu-west-1"},
			RestrictedRegions:   []string{"ap-southeast-1", "ap-northeast-1"},
			DataLocation:        "United States",
			CrossBorderTransfer: true,
		},
		Certifications:  []string{"SOC 1", "SOC 2", "SOC 3", "ISO 27001", "PCI DSS", "HIPAA", "FedRAMP"},
		LastAssessment:  time.Now().Add(-24 * time.Hour),
	}, nil
}

func (p *AWSProvider) GetSecurityGroups(ctx context.Context) ([]*multicloud.SecurityGroup, error) {
	return []*multicloud.SecurityGroup{
		{
			ID:          "sg-12345678",
			Name:        "default",
			Description: "Default security group",
			Rules: []multicloud.SecurityGroupRule{
				{Direction: "ingress", Protocol: "tcp", FromPort: 80, ToPort: 80, Source: "0.0.0.0/0", Action: "allow"},
				{Direction: "ingress", Protocol: "tcp", FromPort: 443, ToPort: 443, Source: "0.0.0.0/0", Action: "allow"},
				{Direction: "ingress", Protocol: "tcp", FromPort: 22, ToPort: 22, Source: "10.0.0.0/16", Action: "allow"},
			},
			CreatedAt: time.Now().Add(-24 * time.Hour),
		},
	}, nil
}

func (p *AWSProvider) CreateSecurityGroup(ctx context.Context, request *multicloud.SecurityGroupRequest) (*multicloud.SecurityGroup, error) {
	return &multicloud.SecurityGroup{
		ID:          fmt.Sprintf("sg-%d", time.Now().Unix()),
		Name:        request.Name,
		Description: request.Description,
		Rules:       request.Rules,
		Tags:        request.Tags,
		CreatedAt:   time.Now(),
	}, nil
}

// Helper methods
func (p *AWSProvider) getInstanceTypeCPU(instanceType string) int {
	// Mock CPU allocation based on instance type
	cpuMap := map[string]int{
		"t3.nano":     1,
		"t3.micro":    1,
		"t3.small":    1,
		"t3.medium":   2,
		"t3.large":    2,
		"t3.xlarge":   4,
		"m5.large":    2,
		"m5.xlarge":   4,
		"m5.2xlarge":  8,
		"c5.large":    2,
		"c5.xlarge":   4,
		"r5.large":    2,
		"r5.xlarge":   4,
	}
	
	if cpu, ok := cpuMap[instanceType]; ok {
		return cpu
	}
	return 1 // Default
}

func (p *AWSProvider) getInstanceTypeMemory(instanceType string) int64 {
	// Mock memory allocation based on instance type (in MB)
	memoryMap := map[string]int64{
		"t3.nano":     512,
		"t3.micro":    1024,
		"t3.small":    2048,
		"t3.medium":   4096,
		"t3.large":    8192,
		"t3.xlarge":   16384,
		"m5.large":    8192,
		"m5.xlarge":   16384,
		"m5.2xlarge":  32768,
		"c5.large":    4096,
		"c5.xlarge":   8192,
		"r5.large":    16384,
		"r5.xlarge":   32768,
	}
	
	if memory, ok := memoryMap[instanceType]; ok {
		return memory
	}
	return 1024 // Default
}

func (p *AWSProvider) getInstanceTypeNetworkBandwidth(instanceType string) int64 {
	// Mock network bandwidth (Mbps)
	bandwidthMap := map[string]int64{
		"t3.nano":     50,
		"t3.micro":    100,
		"t3.small":    200,
		"t3.medium":   300,
		"t3.large":    500,
		"t3.xlarge":   1000,
		"m5.large":    750,
		"m5.xlarge":   1250,
		"m5.2xlarge":  2500,
		"c5.large":    750,
		"c5.xlarge":   1250,
		"r5.large":    750,
		"r5.xlarge":   1250,
	}
	
	if bandwidth, ok := bandwidthMap[instanceType]; ok {
		return bandwidth
	}
	return 100 // Default
}

func (p *AWSProvider) getInstanceTypeHourlyRate(instanceType, region string) float64 {
	// Mock pricing based on instance type and region
	basePrices := map[string]float64{
		"t3.nano":     0.0052,
		"t3.micro":    0.0104,
		"t3.small":    0.0208,
		"t3.medium":   0.0416,
		"t3.large":    0.0832,
		"t3.xlarge":   0.1664,
		"m5.large":    0.096,
		"m5.xlarge":   0.192,
		"m5.2xlarge":  0.384,
		"c5.large":    0.085,
		"c5.xlarge":   0.17,
		"r5.large":    0.126,
		"r5.xlarge":   0.252,
	}

	basePrice := basePrices[instanceType]
	if basePrice == 0 {
		basePrice = 0.0104 // Default to t3.micro price
	}

	// Apply region multiplier
	regionMultiplier := 1.0
	switch region {
	case "us-east-1":
		regionMultiplier = 1.0
	case "us-west-2":
		regionMultiplier = 1.05
	case "eu-west-1":
		regionMultiplier = 1.1
	case "ap-southeast-1":
		regionMultiplier = 1.15
	}

	return basePrice * regionMultiplier
}

func (p *AWSProvider) applyVMFilters(vms []*multicloud.VMInstance, filters map[string]string) []*multicloud.VMInstance {
	if len(filters) == 0 {
		return vms
	}

	var filtered []*multicloud.VMInstance
	for _, vm := range vms {
		if region := filters["region"]; region != "" && vm.Region != region {
			continue
		}
		if state := filters["state"]; state != "" && string(vm.State) != state {
			continue
		}
		if name := filters["name"]; name != "" && vm.Name != name {
			continue
		}
		filtered = append(filtered, vm)
	}

	return filtered
}