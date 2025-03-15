package testsuite

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/novacron/backend/core/cloud"
)

// TestResult represents the result of a single test
type TestResult struct {
	TestName    string
	ProviderID  string
	Success     bool
	ElapsedTime time.Duration
	Error       error
	Message     string
}

// ProviderTestSuite is a comprehensive test suite for cloud providers
type ProviderTestSuite struct {
	providers map[string]cloud.Provider
	results   []TestResult
	verbose   bool
}

// NewProviderTestSuite creates a new test suite for the given providers
func NewProviderTestSuite(providerManager *cloud.ProviderManager, verbose bool) *ProviderTestSuite {
	providerList := providerManager.ListProviders()
	providers := make(map[string]cloud.Provider)

	for _, name := range providerList {
		provider, err := providerManager.GetProvider(name)
		if err != nil {
			log.Printf("Warning: Could not get provider %s: %v", name, err)
			continue
		}
		providers[name] = provider
	}

	return &ProviderTestSuite{
		providers: providers,
		results:   make([]TestResult, 0),
		verbose:   verbose,
	}
}

// RunAllTests runs all tests for all providers
func (ts *ProviderTestSuite) RunAllTests(ctx context.Context) {
	fmt.Println("=== Starting Provider Test Suite ===")

	for name, provider := range ts.providers {
		fmt.Printf("\nTesting provider: %s\n", name)

		// Instance tests
		ts.TestInstanceOperations(ctx, name, provider)

		// Image tests
		ts.TestImageOperations(ctx, name, provider)

		// Region tests
		ts.TestRegionOperations(ctx, name, provider)

		// Storage tests
		ts.TestStorageOperations(ctx, name, provider)

		// Snapshot tests
		ts.TestSnapshotOperations(ctx, name, provider)

		// Network tests
		ts.TestNetworkOperations(ctx, name, provider)

		// Pricing tests
		ts.TestPricingOperations(ctx, name, provider)
	}

	ts.PrintSummary()
}

// TestInstanceOperations tests all instance-related operations
func (ts *ProviderTestSuite) TestInstanceOperations(ctx context.Context, providerName string, provider cloud.Provider) {
	fmt.Printf("  Testing instance operations...\n")

	// Test GetInstances
	ts.RunTest(ctx, providerName, "GetInstances", func() error {
		instances, err := provider.GetInstances(ctx, cloud.ListOptions{Limit: 10})
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Found %d instances\n", len(instances))
			for i, instance := range instances {
				if i < 3 { // Limit output to first 3
					fmt.Printf("    - %s (%s): %s, %s\n", instance.Name, instance.ID, instance.InstanceType, instance.State)
				}
			}
			if len(instances) > 3 {
				fmt.Printf("    - ... and %d more\n", len(instances)-3)
			}
		}

		return nil
	})

	// Test CreateInstance and then use the created instance for subsequent tests
	var createdInstanceID string
	ts.RunTest(ctx, providerName, "CreateInstance", func() error {
		specs := cloud.InstanceSpecs{
			Name:           fmt.Sprintf("test-instance-%d", time.Now().Unix()),
			InstanceType:   GetDefaultInstanceType(providerName),
			ImageID:        GetDefaultImageID(providerName),
			Region:         GetDefaultRegion(providerName),
			Zone:           GetDefaultZone(providerName),
			CPUCores:       2,
			MemoryGB:       4,
			DiskGB:         20,
			AssignPublicIP: true,
			Tags: map[string]string{
				"purpose": "testing",
				"project": "novacron",
			},
		}

		instance, err := provider.CreateInstance(ctx, specs)
		if err != nil {
			return err
		}

		createdInstanceID = instance.ID
		if ts.verbose {
			fmt.Printf("    Created instance %s (%s)\n", instance.Name, instance.ID)
		}

		return nil
	})

	// Skip subsequent tests if instance creation failed
	if createdInstanceID == "" {
		fmt.Printf("    Skipping remaining instance tests because instance creation failed\n")
		return
	}

	// Test GetInstance
	ts.RunTest(ctx, providerName, "GetInstance", func() error {
		instance, err := provider.GetInstance(ctx, createdInstanceID)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Got instance details: %s (%s), State: %s\n", instance.Name, instance.ID, instance.State)
		}

		return nil
	})

	// Allow instance to reach stable state
	time.Sleep(2 * time.Second)

	// Test StopInstance
	ts.RunTest(ctx, providerName, "StopInstance", func() error {
		err := provider.StopInstance(ctx, createdInstanceID)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Stopped instance %s\n", createdInstanceID)
		}

		// Wait for state change
		time.Sleep(2 * time.Second)

		return nil
	})

	// Test StartInstance
	ts.RunTest(ctx, providerName, "StartInstance", func() error {
		err := provider.StartInstance(ctx, createdInstanceID)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Started instance %s\n", createdInstanceID)
		}

		// Wait for state change
		time.Sleep(2 * time.Second)

		return nil
	})

	// Test RestartInstance
	ts.RunTest(ctx, providerName, "RestartInstance", func() error {
		err := provider.RestartInstance(ctx, createdInstanceID)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Restarted instance %s\n", createdInstanceID)
		}

		// Wait for state change
		time.Sleep(2 * time.Second)

		return nil
	})

	// Test ResizeInstance
	ts.RunTest(ctx, providerName, "ResizeInstance", func() error {
		newSpecs := cloud.InstanceSpecs{
			InstanceType: GetUpgradedInstanceType(providerName),
			CPUCores:     4,
			MemoryGB:     8,
		}

		err := provider.ResizeInstance(ctx, createdInstanceID, newSpecs)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Resized instance %s to %s\n", createdInstanceID, newSpecs.InstanceType)
		}

		return nil
	})

	// Cleanup: Delete the created instance
	ts.RunTest(ctx, providerName, "DeleteInstance", func() error {
		err := provider.DeleteInstance(ctx, createdInstanceID)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Deleted instance %s\n", createdInstanceID)
		}

		return nil
	})
}

// TestImageOperations tests all image-related operations
func (ts *ProviderTestSuite) TestImageOperations(ctx context.Context, providerName string, provider cloud.Provider) {
	fmt.Printf("  Testing image operations...\n")

	// Test GetImageList
	ts.RunTest(ctx, providerName, "GetImageList", func() error {
		images, err := provider.GetImageList(ctx, cloud.ListOptions{Limit: 10})
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Found %d images\n", len(images))
			for i, image := range images {
				if i < 3 { // Limit output to first 3
					fmt.Printf("    - %s (%s): %s %s\n", image.Name, image.ID, image.OS, image.Version)
				}
			}
			if len(images) > 3 {
				fmt.Printf("    - ... and %d more\n", len(images)-3)
			}
		}

		return nil
	})
}

// TestRegionOperations tests all region-related operations
func (ts *ProviderTestSuite) TestRegionOperations(ctx context.Context, providerName string, provider cloud.Provider) {
	fmt.Printf("  Testing region operations...\n")

	// Test GetRegions
	ts.RunTest(ctx, providerName, "GetRegions", func() error {
		regions, err := provider.GetRegions(ctx)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Found %d regions\n", len(regions))
			for i, region := range regions {
				if i < 3 { // Limit output to first 3
					fmt.Printf("    - %s (%s): %d zones\n", region.Name, region.ID, len(region.Zones))
				}
			}
			if len(regions) > 3 {
				fmt.Printf("    - ... and %d more\n", len(regions)-3)
			}
		}

		return nil
	})
}

// TestStorageOperations tests all storage-related operations
func (ts *ProviderTestSuite) TestStorageOperations(ctx context.Context, providerName string, provider cloud.Provider) {
	fmt.Printf("  Testing storage operations...\n")

	// Test GetStorageVolumes
	ts.RunTest(ctx, providerName, "GetStorageVolumes", func() error {
		volumes, err := provider.GetStorageVolumes(ctx, cloud.ListOptions{Limit: 10})
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Found %d storage volumes\n", len(volumes))
			for i, volume := range volumes {
				if i < 3 { // Limit output to first 3
					fmt.Printf("    - %s (%s): %d GB, %s\n", volume.Name, volume.ID, volume.SizeGB, volume.Type)
				}
			}
			if len(volumes) > 3 {
				fmt.Printf("    - ... and %d more\n", len(volumes)-3)
			}
		}

		return nil
	})

	// Test CreateStorageVolume
	var createdVolumeID string
	ts.RunTest(ctx, providerName, "CreateStorageVolume", func() error {
		specs := cloud.StorageVolumeSpecs{
			Name:   fmt.Sprintf("test-volume-%d", time.Now().Unix()),
			SizeGB: 50,
			Type:   GetDefaultStorageType(providerName),
			Region: GetDefaultRegion(providerName),
			Zone:   GetDefaultZone(providerName),
			Tags: map[string]string{
				"purpose": "testing",
				"project": "novacron",
			},
		}

		volume, err := provider.CreateStorageVolume(ctx, specs)
		if err != nil {
			return err
		}

		createdVolumeID = volume.ID
		if ts.verbose {
			fmt.Printf("    Created volume %s (%s): %d GB\n", volume.Name, volume.ID, volume.SizeGB)
		}

		return nil
	})

	// Skip subsequent tests if volume creation failed
	if createdVolumeID == "" {
		fmt.Printf("    Skipping remaining storage tests because volume creation failed\n")
		return
	}

	// Create test instance for attachment
	var testInstanceID string
	ts.RunTest(ctx, providerName, "CreateInstanceForVolumeTest", func() error {
		specs := cloud.InstanceSpecs{
			Name:         fmt.Sprintf("test-volume-instance-%d", time.Now().Unix()),
			InstanceType: GetDefaultInstanceType(providerName),
			ImageID:      GetDefaultImageID(providerName),
			Region:       GetDefaultRegion(providerName),
			Zone:         GetDefaultZone(providerName),
			CPUCores:     2,
			MemoryGB:     4,
			DiskGB:       20,
		}

		instance, err := provider.CreateInstance(ctx, specs)
		if err != nil {
			return err
		}

		testInstanceID = instance.ID
		if ts.verbose {
			fmt.Printf("    Created test instance %s for volume operations\n", instance.ID)
		}

		// Wait for instance to be ready
		time.Sleep(3 * time.Second)

		return nil
	})

	// Skip attachment tests if instance creation failed
	if testInstanceID == "" {
		fmt.Printf("    Skipping volume attachment tests because instance creation failed\n")
	} else {
		// Test AttachStorageVolume
		ts.RunTest(ctx, providerName, "AttachStorageVolume", func() error {
			opts := cloud.AttachOptions{
				DevicePath: GetDefaultDevicePath(providerName),
				ReadOnly:   false,
			}

			err := provider.AttachStorageVolume(ctx, createdVolumeID, testInstanceID, opts)
			if err != nil {
				return err
			}

			if ts.verbose {
				fmt.Printf("    Attached volume %s to instance %s at %s\n",
					createdVolumeID, testInstanceID, opts.DevicePath)
			}

			// Allow attachment to complete
			time.Sleep(2 * time.Second)

			return nil
		})

		// Test DetachStorageVolume
		ts.RunTest(ctx, providerName, "DetachStorageVolume", func() error {
			err := provider.DetachStorageVolume(ctx, createdVolumeID, testInstanceID)
			if err != nil {
				return err
			}

			if ts.verbose {
				fmt.Printf("    Detached volume %s from instance %s\n", createdVolumeID, testInstanceID)
			}

			// Allow detachment to complete
			time.Sleep(2 * time.Second)

			return nil
		})

		// Cleanup: Delete the test instance
		ts.RunTest(ctx, providerName, "DeleteTestInstance", func() error {
			err := provider.DeleteInstance(ctx, testInstanceID)
			if err != nil {
				return err
			}

			if ts.verbose {
				fmt.Printf("    Deleted test instance %s\n", testInstanceID)
			}

			return nil
		})
	}

	// Cleanup: Delete the created volume
	ts.RunTest(ctx, providerName, "DeleteStorageVolume", func() error {
		err := provider.DeleteStorageVolume(ctx, createdVolumeID)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Deleted volume %s\n", createdVolumeID)
		}

		return nil
	})
}

// TestSnapshotOperations tests all snapshot-related operations
func (ts *ProviderTestSuite) TestSnapshotOperations(ctx context.Context, providerName string, provider cloud.Provider) {
	fmt.Printf("  Testing snapshot operations...\n")

	// Test GetSnapshots
	ts.RunTest(ctx, providerName, "GetSnapshots", func() error {
		snapshots, err := provider.GetSnapshots(ctx, cloud.ListOptions{Limit: 10})
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Found %d snapshots\n", len(snapshots))
			for i, snapshot := range snapshots {
				if i < 3 { // Limit output to first 3
					fmt.Printf("    - %s (%s): %d GB\n", snapshot.Name, snapshot.ID, snapshot.SizeGB)
				}
			}
			if len(snapshots) > 3 {
				fmt.Printf("    - ... and %d more\n", len(snapshots)-3)
			}
		}

		return nil
	})

	// Create test volume for snapshot
	var testVolumeID string
	ts.RunTest(ctx, providerName, "CreateVolumeForSnapshot", func() error {
		specs := cloud.StorageVolumeSpecs{
			Name:   fmt.Sprintf("test-snapshot-volume-%d", time.Now().Unix()),
			SizeGB: 10,
			Type:   GetDefaultStorageType(providerName),
			Region: GetDefaultRegion(providerName),
			Zone:   GetDefaultZone(providerName),
		}

		volume, err := provider.CreateStorageVolume(ctx, specs)
		if err != nil {
			return err
		}

		testVolumeID = volume.ID
		if ts.verbose {
			fmt.Printf("    Created test volume %s for snapshot operations\n", volume.ID)
		}

		// Wait for volume to be ready
		time.Sleep(2 * time.Second)

		return nil
	})

	// Skip snapshot tests if volume creation failed
	if testVolumeID == "" {
		fmt.Printf("    Skipping snapshot creation tests because volume creation failed\n")
		return
	}

	// Test CreateSnapshot
	var createdSnapshotID string
	ts.RunTest(ctx, providerName, "CreateSnapshot", func() error {
		specs := cloud.SnapshotSpecs{
			Name:        fmt.Sprintf("test-snapshot-%d", time.Now().Unix()),
			Description: "Test snapshot for provider test suite",
			Tags: map[string]string{
				"purpose": "testing",
				"project": "novacron",
			},
		}

		snapshot, err := provider.CreateSnapshot(ctx, testVolumeID, specs)
		if err != nil {
			return err
		}

		createdSnapshotID = snapshot.ID
		if ts.verbose {
			fmt.Printf("    Created snapshot %s from volume %s\n", snapshot.ID, testVolumeID)
		}

		// Allow snapshot to complete
		time.Sleep(3 * time.Second)

		return nil
	})

	// Cleanup: Delete the created snapshot
	if createdSnapshotID != "" {
		ts.RunTest(ctx, providerName, "DeleteSnapshot", func() error {
			err := provider.DeleteSnapshot(ctx, createdSnapshotID)
			if err != nil {
				return err
			}

			if ts.verbose {
				fmt.Printf("    Deleted snapshot %s\n", createdSnapshotID)
			}

			return nil
		})
	}

	// Cleanup: Delete the test volume
	ts.RunTest(ctx, providerName, "DeleteTestVolume", func() error {
		err := provider.DeleteStorageVolume(ctx, testVolumeID)
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Deleted test volume %s\n", testVolumeID)
		}

		return nil
	})
}

// TestNetworkOperations tests all network-related operations
func (ts *ProviderTestSuite) TestNetworkOperations(ctx context.Context, providerName string, provider cloud.Provider) {
	fmt.Printf("  Testing network operations...\n")

	// Test GetNetworks
	ts.RunTest(ctx, providerName, "GetNetworks", func() error {
		networks, err := provider.GetNetworks(ctx, cloud.ListOptions{Limit: 10})
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Found %d networks\n", len(networks))
			for i, network := range networks {
				if i < 3 { // Limit output to first 3
					fmt.Printf("    - %s (%s): %s\n", network.Name, network.ID, network.CIDR)
				}
			}
			if len(networks) > 3 {
				fmt.Printf("    - ... and %d more\n", len(networks)-3)
			}
		}

		return nil
	})

	// Test CreateNetwork
	var createdNetworkID string
	ts.RunTest(ctx, providerName, "CreateNetwork", func() error {
		specs := cloud.NetworkSpecs{
			Name:   fmt.Sprintf("test-network-%d", time.Now().Unix()),
			CIDR:   "192.168.0.0/24",
			Region: GetDefaultRegion(providerName),
			Tags: map[string]string{
				"purpose": "testing",
				"project": "novacron",
			},
		}

		network, err := provider.CreateNetwork(ctx, specs)
		if err != nil {
			return err
		}

		createdNetworkID = network.ID
		if ts.verbose {
			fmt.Printf("    Created network %s (%s): %s\n", network.Name, network.ID, network.CIDR)
		}

		// Allow network creation to complete
		time.Sleep(2 * time.Second)

		return nil
	})

	// Cleanup: Delete the created network
	if createdNetworkID != "" {
		ts.RunTest(ctx, providerName, "DeleteNetwork", func() error {
			err := provider.DeleteNetwork(ctx, createdNetworkID)
			if err != nil {
				return err
			}

			if ts.verbose {
				fmt.Printf("    Deleted network %s\n", createdNetworkID)
			}

			return nil
		})
	}
}

// TestPricingOperations tests all pricing-related operations
func (ts *ProviderTestSuite) TestPricingOperations(ctx context.Context, providerName string, provider cloud.Provider) {
	fmt.Printf("  Testing pricing operations...\n")

	// Test GetPricing for instances
	ts.RunTest(ctx, providerName, "GetInstancePricing", func() error {
		pricing, err := provider.GetPricing(ctx, "instance")
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Found pricing for %d instance types\n", len(pricing))
			count := 0
			for instanceType, price := range pricing {
				if count < 3 { // Limit output
					fmt.Printf("    - %s: $%.4f/hour\n", instanceType, price)
					count++
				}
			}
			if len(pricing) > 3 {
				fmt.Printf("    - ... and %d more\n", len(pricing)-3)
			}
		}

		return nil
	})

	// Test GetPricing for storage
	ts.RunTest(ctx, providerName, "GetStoragePricing", func() error {
		pricing, err := provider.GetPricing(ctx, "storage")
		if err != nil {
			return err
		}

		if ts.verbose {
			fmt.Printf("    Found pricing for %d storage types\n", len(pricing))
			count := 0
			for storageType, price := range pricing {
				if count < 3 { // Limit output
					fmt.Printf("    - %s: $%.4f/GB/month\n", storageType, price)
					count++
				}
			}
			if len(pricing) > 3 {
				fmt.Printf("    - ... and %d more\n", len(pricing)-3)
			}
		}

		return nil
	})
}

// RunTest runs a single test and records the result
// This function is exported so it can be used by test scenarios
func (ts *ProviderTestSuite) RunTest(ctx context.Context, providerID, testName string, testFunc func() error) {
	startTime := time.Now()
	var err error

	if ts.verbose {
		fmt.Printf("    Running test: %s...\n", testName)
	}

	err = testFunc()

	elapsedTime := time.Since(startTime)

	// Record result
	result := TestResult{
		TestName:    testName,
		ProviderID:  providerID,
		Success:     err == nil,
		ElapsedTime: elapsedTime,
		Error:       err,
	}

	if err != nil {
		result.Message = err.Error()
		if !ts.verbose {
			fmt.Printf("    × %s: %v\n", testName, err)
		} else {
			fmt.Printf("    × Failed: %v\n", err)
		}
	} else if !ts.verbose {
		fmt.Printf("    ✓ %s (%.2fs)\n", testName, elapsedTime.Seconds())
	}

	ts.results = append(ts.results, result)
}

// PrintSummary prints a summary of all test results
func (ts *ProviderTestSuite) PrintSummary() {
	fmt.Println("\n=== Test Suite Summary ===")

	// Group results by provider
	resultsByProvider := make(map[string][]TestResult)
	for _, result := range ts.results {
		resultsByProvider[result.ProviderID] = append(resultsByProvider[result.ProviderID], result)
	}

	totalTests := len(ts.results)
	totalSuccessful := 0

	for _, result := range ts.results {
		if result.Success {
			totalSuccessful++
		}
	}

	// Print summary for each provider
	for providerID, results := range resultsByProvider {
		fmt.Printf("\nProvider: %s\n", providerID)

		successCount := 0
		failCount := 0
		totalDuration := time.Duration(0)

		for _, result := range results {
			if result.Success {
				successCount++
			} else {
				failCount++
			}
			totalDuration += result.ElapsedTime
		}

		fmt.Printf("  Tests:      %d\n", len(results))
		fmt.Printf("  Succeeded:  %d (%.1f%%)\n", successCount, float64(successCount)/float64(len(results))*100)
		fmt.Printf("  Failed:     %d\n", failCount)
		fmt.Printf("  Total time: %.2fs\n", totalDuration.Seconds())

		// Print failed tests
		if failCount > 0 {
			fmt.Printf("\n  Failed tests:\n")
			for _, result := range results {
				if !result.Success {
					fmt.Printf("    × %s: %s\n", result.TestName, result.Message)
				}
			}
		}
	}

	// Print overall summary
	fmt.Printf("\nOverall Summary:\n")
	fmt.Printf("  Total tests:       %d\n", totalTests)
	fmt.Printf("  Total succeeded:   %d (%.1f%%)\n", totalSuccessful, float64(totalSuccessful)/float64(totalTests)*100)
	fmt.Printf("  Total failed:      %d\n", totalTests-totalSuccessful)
}

// Helper functions for provider-specific default values

// GetDefaultInstanceType returns the default instance type for a given provider
func GetDefaultInstanceType(provider string) string {
	switch strings.ToLower(provider) {
	case "aws":
		return "t3.medium"
	case "azure":
		return "Standard_D2s_v3"
	case "gcp":
		return "e2-medium"
	default:
		return "t3.medium"
	}
}

// GetUpgradedInstanceType returns a larger instance type for a given provider
func GetUpgradedInstanceType(provider string) string {
	switch strings.ToLower(provider) {
	case "aws":
		return "t3.large"
	case "azure":
		return "Standard_D4s_v3"
	case "gcp":
		return "e2-standard-2"
	default:
		return "t3.large"
	}
}

// GetDefaultImageID returns the default image ID for a given provider
func GetDefaultImageID(provider string) string {
	switch strings.ToLower(provider) {
	case "aws":
		return "ami-12345678" // Would be a real AMI ID in production
	case "azure":
		return "Canonical:UbuntuServer:18.04-LTS:latest"
	case "gcp":
		return "projects/debian-cloud/global/images/debian-10-buster-v20210721"
	default:
		return "ami-12345678"
	}
}

// GetDefaultRegion returns the default region for a given provider
func GetDefaultRegion(provider string) string {
	switch strings.ToLower(provider) {
	case "aws":
		return "us-east-1"
	case "azure":
		return "eastus"
	case "gcp":
		return "us-central1"
	default:
		return "us-east-1"
	}
}

// GetDefaultZone returns the default zone for a given provider
func GetDefaultZone(provider string) string {
	switch strings.ToLower(provider) {
	case "aws":
		return "us-east-1a"
	case "azure":
		return "eastus-1"
	case "gcp":
		return "us-central1-a"
	default:
		return "us-east-1a"
	}
}

// GetDefaultStorageType returns the default storage type for a given provider
func GetDefaultStorageType(provider string) string {
	switch strings.ToLower(provider) {
	case "aws":
		return "gp2"
	case "azure":
		return "Premium_LRS"
	case "gcp":
		return "pd-ssd"
	default:
		return "gp2"
	}
}

// GetDefaultDevicePath returns the default device path for a given provider
func GetDefaultDevicePath(provider string) string {
	switch strings.ToLower(provider) {
	case "aws":
		return "/dev/sdf"
	case "azure":
		return "/dev/sdc"
	case "gcp":
		return "/dev/sdc"
	default:
		return "/dev/sdf"
	}
}
