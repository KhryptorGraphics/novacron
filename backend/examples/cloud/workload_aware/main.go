package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/novacron/backend/core/cloud"
	"github.com/novacron/backend/core/cloud/workload"
)

func main() {
	// Parse command line arguments
	if len(os.Args) > 1 && os.Args[1] == "help" {
		fmt.Println("Usage: go run main.go [workload-type]")
		fmt.Println("  workload-type: web, batch, database, ml-training, ml-inference, analytics, dev-test")
		fmt.Println("  If no workload type is specified, all workload types will be tested.")
		return
	}

	// Create context
	ctx := context.Background()

	// Setup cloud providers
	orchestrator, err := setupOrchestrator(ctx)
	if err != nil {
		log.Fatalf("Failed to setup orchestrator: %v", err)
	}

	// Determine which workload types to test
	var workloadTypes []workload.WorkloadType
	if len(os.Args) > 1 {
		// Test specific workload type
		workloadType := parseWorkloadType(os.Args[1])
		if workloadType == "" {
			log.Fatalf("Unknown workload type: %s", os.Args[1])
		}
		workloadTypes = []workload.WorkloadType{workloadType}
	} else {
		// Test all workload types
		workloadTypes = []workload.WorkloadType{
			workload.WebServer,
			workload.BatchProcessing,
			workload.DatabaseWorkload,
			workload.MLTraining,
			workload.MLInference,
			workload.AnalyticsWorkload,
			workload.DevTest,
		}
	}

	// Test each workload type
	for _, workloadType := range workloadTypes {
		fmt.Printf("\n=== Testing %s workload ===\n\n", workloadType)
		testWorkload(ctx, orchestrator, workloadType)
	}

	// Get cost optimization recommendations
	fmt.Println("\n=== Cost Optimization Recommendations ===\n")
	recommendations, err := orchestrator.GetCostOptimizationRecommendations(ctx)
	if err != nil {
		log.Printf("Failed to get cost optimization recommendations: %v", err)
	} else {
		for i, rec := range recommendations {
			fmt.Printf("Recommendation %d:\n", i+1)
			fmt.Printf("  Resource ID: %s\n", rec.ResourceID)
			fmt.Printf("  Resource Type: %s\n", rec.ResourceType)
			fmt.Printf("  Current Cost: $%.2f\n", rec.CurrentCost)
			fmt.Printf("  Recommended Action: %s\n", rec.RecommendedAction)
			if rec.TargetProvider != "" {
				fmt.Printf("  Target Provider: %s\n", rec.TargetProvider)
			}
			fmt.Printf("  Expected Savings: $%.2f\n", rec.ExpectedSavings)
			fmt.Printf("  Confidence: %.2f\n", rec.Confidence)
			fmt.Printf("  Reason: %s\n\n", rec.Reason)
		}
	}
}

func setupOrchestrator(ctx context.Context) (*cloud.EnhancedHybridCloudOrchestrator, error) {
	// Create enhanced orchestrator
	orchestrator := cloud.NewEnhancedHybridCloudOrchestrator()

	// Create and register AWS provider
	awsProvider := createAWSProvider()
	if err := orchestrator.RegisterProvider(awsProvider); err != nil {
		return nil, fmt.Errorf("failed to register AWS provider: %v", err)
	}

	// Create and register Azure provider
	azureProvider := createAzureProvider()
	if err := orchestrator.RegisterProvider(azureProvider); err != nil {
		return nil, fmt.Errorf("failed to register Azure provider: %v", err)
	}

	// Create and register GCP provider
	gcpProvider := createGCPProvider()
	if err := orchestrator.RegisterProvider(gcpProvider); err != nil {
		return nil, fmt.Errorf("failed to register GCP provider: %v", err)
	}

	// Set workload-aware balance strategy
	orchestrator.SetWorkloadAwareStrategy(cloud.WorkloadAwareBalance)

	// Enable cost optimization
	orchestrator.EnableCostOptimization(true, 1*time.Hour)

	return orchestrator, nil
}

func createAWSProvider() cloud.Provider {
	// In a real implementation, this would create a proper AWS provider
	// For this example, we'll use a mock provider
	return &MockProvider{
		name: "aws",
		pricing: map[string]float64{
			"t3.micro":   0.0104,
			"t3.small":   0.0208,
			"t3.medium":  0.0416,
			"t3.large":   0.0832,
			"m5.large":   0.096,
			"c5.large":   0.085,
			"c5.xlarge":  0.17,
			"c5.2xlarge": 0.34,
			"r5.large":   0.126,
			"r5.xlarge":  0.252,
			"p3.2xlarge": 3.06,
		},
	}
}

func createAzureProvider() cloud.Provider {
	// In a real implementation, this would create a proper Azure provider
	// For this example, we'll use a mock provider
	return &MockProvider{
		name: "azure",
		pricing: map[string]float64{
			"Standard_B1ms":    0.0207,
			"Standard_B2s":     0.0416,
			"Standard_D2s_v3":  0.0957,
			"Standard_D4s_v3":  0.1915,
			"Standard_E2s_v3":  0.126,
			"Standard_E4s_v3":  0.252,
			"Standard_F4s_v2":  0.17,
			"Standard_F8s_v2":  0.34,
			"Standard_NC6s_v3": 3.06,
		},
	}
}

func createGCPProvider() cloud.Provider {
	// In a real implementation, this would create a proper GCP provider
	// For this example, we'll use a mock provider
	return &MockProvider{
		name: "gcp",
		pricing: map[string]float64{
			"e2-micro":      0.0076,
			"e2-small":      0.0152,
			"e2-medium":     0.0304,
			"e2-standard-2": 0.0608,
			"e2-standard-4": 0.1216,
			"n1-standard-2": 0.0950,
			"n2-standard-2": 0.0969,
			"n2-highmem-2":  0.1286,
			"n2-highmem-4":  0.2571,
			"c2-standard-4": 0.1942,
			"c2-standard-8": 0.3884,
		},
	}
}

func testWorkload(ctx context.Context, orchestrator *cloud.EnhancedHybridCloudOrchestrator, workloadType workload.WorkloadType) {
	// Create workload metrics based on workload type
	metrics := createWorkloadMetrics(workloadType)

	// Create instance specs
	specs := cloud.InstanceSpecs{
		InstanceType: "standard",
		Region:       "us-west-1",
		Zone:         "us-west-1a",
		ImageID:      "ami-12345",
		Tags: map[string]string{
			"workload": string(workloadType),
		},
	}

	// Create instance with workload-aware placement
	instance, err := orchestrator.CreateWorkloadAwareInstance(ctx, specs, metrics)
	if err != nil {
		log.Printf("Failed to create instance for %s workload: %v", workloadType, err)
		return
	}

	fmt.Printf("Created instance %s with provider %s\n", instance.ID, instance.ProviderName)
	fmt.Printf("Instance type: %s\n", instance.InstanceType)
	fmt.Printf("Created at: %s\n", instance.CreatedAt.Format(time.RFC3339))
}

func createWorkloadMetrics(workloadType workload.WorkloadType) workload.Metrics {
	// Create base metrics
	metrics := workload.Metrics{
		AvgCPUUtilization:  30.0,
		PeakCPUUtilization: 60.0,
		CPUUtilizationP95:  55.0,

		AvgMemoryUtilization:  40.0,
		PeakMemoryUtilization: 70.0,
		MemoryUtilizationP95:  65.0,

		AvgIOPS:            100.0,
		PeakIOPS:           500.0,
		ReadWriteRatio:     1.0,
		RandomIOPercentage: 50.0,

		AvgNetworkIn:          5.0,
		AvgNetworkOut:         5.0,
		PeakNetworkIn:         20.0,
		PeakNetworkOut:        20.0,
		AvgActiveConnections:  10.0,
		PeakActiveConnections: 50.0,

		TimeOfDayPatterns:    make(map[int]float64),
		DayOfWeekPatterns:    make(map[int]float64),
		WeeklyPatternQuality: 0.5,
	}

	// Customize metrics based on workload type
	switch workloadType {
	case workload.WebServer:
		metrics.AvgCPUUtilization = 40.0
		metrics.AvgNetworkIn = 20.0
		metrics.AvgNetworkOut = 40.0
		metrics.AvgActiveConnections = 200.0
		metrics.PeakActiveConnections = 1000.0
		metrics.ReadWriteRatio = 5.0 // Read-heavy

	case workload.BatchProcessing:
		metrics.AvgCPUUtilization = 60.0
		metrics.PeakCPUUtilization = 95.0
		metrics.CPUUtilizationStdDev = 30.0
		metrics.WeeklyPatternQuality = 0.9
		// Set time patterns for batch - mostly nights and weekends
		for hour := 0; hour < 8; hour++ {
			metrics.TimeOfDayPatterns[hour] = 80.0
		}
		for hour := 8; hour < 18; hour++ {
			metrics.TimeOfDayPatterns[hour] = 20.0
		}
		for hour := 18; hour < 24; hour++ {
			metrics.TimeOfDayPatterns[hour] = 70.0
		}
		metrics.DayOfWeekPatterns[0] = 90.0 // Sunday
		metrics.DayOfWeekPatterns[6] = 90.0 // Saturday

	case workload.DatabaseWorkload:
		metrics.AvgIOPS = 2000.0
		metrics.PeakIOPS = 5000.0
		metrics.AvgMemoryUtilization = 70.0
		metrics.MemoryStability = true
		metrics.RandomIOPercentage = 80.0

	case workload.MLTraining:
		metrics.AvgCPUUtilization = 90.0
		metrics.PeakCPUUtilization = 98.0
		metrics.AvgMemoryUtilization = 85.0
		metrics.MemoryUtilizationP95 = 95.0

	case workload.MLInference:
		metrics.AvgCPUUtilization = 40.0
		metrics.PeakCPUUtilization = 90.0
		metrics.CPUUtilizationStdDev = 25.0
		metrics.AvgNetworkIn = 30.0
		metrics.AvgNetworkOut = 10.0

	case workload.AnalyticsWorkload:
		metrics.AvgMemoryUtilization = 75.0
		metrics.AvgIOPS = 1500.0
		metrics.WeeklyPatternQuality = 0.8
		// Set time patterns for analytics - business hours
		for hour := 8; hour < 18; hour++ {
			metrics.TimeOfDayPatterns[hour] = 80.0
		}
		for day := 1; day <= 5; day++ {
			metrics.DayOfWeekPatterns[day] = 90.0 // Weekdays
		}

	case workload.DevTest:
		metrics.AvgCPUUtilization = 10.0
		metrics.PeakCPUUtilization = 60.0
		metrics.CPUUtilizationStdDev = 15.0
		metrics.AvgMemoryUtilization = 20.0
		metrics.WeeklyPatternQuality = 0.7
		// Set time patterns for dev/test - business hours
		for hour := 9; hour < 17; hour++ {
			metrics.TimeOfDayPatterns[hour] = 80.0
		}
		for day := 1; day <= 5; day++ {
			metrics.DayOfWeekPatterns[day] = 90.0 // Weekdays
		}
	}

	return metrics
}

// Parse workload type from string
func parseWorkloadType(typeStr string) workload.WorkloadType {
	switch typeStr {
	case "web":
		return workload.WebServer
	case "batch":
		return workload.BatchProcessing
	case "database":
		return workload.DatabaseWorkload
	case "ml-training":
		return workload.MLTraining
	case "ml-inference":
		return workload.MLInference
	case "analytics":
		return workload.AnalyticsWorkload
	case "dev-test":
		return workload.DevTest
	default:
		return ""
	}
}

// MockProvider implements the cloud.Provider interface for testing
type MockProvider struct {
	name    string
	pricing map[string]float64
}

func (p *MockProvider) Name() string {
	return p.name
}

func (p *MockProvider) CreateInstance(ctx context.Context, specs cloud.InstanceSpecs) (*cloud.Instance, error) {
	// For this example, just create a mock instance
	return &cloud.Instance{
		ID:           fmt.Sprintf("%s-instance-%d", p.name, time.Now().UnixNano()),
		ProviderID:   fmt.Sprintf("%s-%d", p.name, time.Now().UnixNano()),
		Name:         fmt.Sprintf("%s-instance", p.name),
		InstanceType: specs.InstanceType,
		Region:       specs.Region,
		Zone:         specs.Zone,
		Status:       "running",
		PublicIP:     "10.0.0.1",
		PrivateIP:    "192.168.1.1",
		LaunchTime:   time.Now(),
		Tags:         specs.Tags,
	}, nil
}

func (p *MockProvider) GetInstance(ctx context.Context, instanceID string) (*cloud.Instance, error) {
	return nil, fmt.Errorf("not implemented")
}

func (p *MockProvider) ListInstances(ctx context.Context) ([]*cloud.Instance, error) {
	return nil, fmt.Errorf("not implemented")
}

func (p *MockProvider) TerminateInstance(ctx context.Context, instanceID string) error {
	return fmt.Errorf("not implemented")
}

func (p *MockProvider) CreateStorageVolume(ctx context.Context, specs cloud.StorageVolumeSpecs) (*cloud.StorageVolume, error) {
	return nil, fmt.Errorf("not implemented")
}

func (p *MockProvider) GetStorageVolume(ctx context.Context, volumeID string) (*cloud.StorageVolume, error) {
	return nil, fmt.Errorf("not implemented")
}

func (p *MockProvider) AttachStorageVolume(ctx context.Context, volumeID, instanceID string) error {
	return fmt.Errorf("not implemented")
}

func (p *MockProvider) DetachStorageVolume(ctx context.Context, volumeID string) error {
	return fmt.Errorf("not implemented")
}

func (p *MockProvider) GetPricing(ctx context.Context, resourceType string) (map[string]float64, error) {
	return p.pricing, nil
}
