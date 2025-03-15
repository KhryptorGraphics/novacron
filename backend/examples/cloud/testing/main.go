package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/novacron/backend/core/cloud"
	"github.com/novacron/backend/examples/cloud/testsuite"
)

func main() {
	// Create a new provider manager
	providerManager := cloud.NewProviderManager()

	// Register provider types
	providerManager.RegisterProviderType(cloud.NewAWSProvider())
	providerManager.RegisterProviderType(cloud.NewAzureProvider())
	providerManager.RegisterProviderType(cloud.NewGCPProvider())

	// Initialize providers with environment variables
	initializeProviders(providerManager)

	// Display available providers
	providers := providerManager.ListProviders()
	fmt.Println("Initialized providers:")
	for _, provider := range providers {
		fmt.Printf("- %s\n", provider)
	}

	// Create a new test suite with verbose output
	fmt.Println("\nSetting up cloud provider test suite...")
	testSuite := testsuite.NewProviderTestSuite(providerManager, true)
	fmt.Println("Provider test suite is ready")

	// Run tests based on command line arguments
	ctx := context.Background()

	if len(os.Args) > 1 && os.Args[1] == "all" {
		// Run tests for all providers
		fmt.Println("\nRunning tests for all providers...")
		testSuite.RunAllTests(ctx)
	} else if len(os.Args) > 1 {
		// Run tests for a specific provider
		providerName := os.Args[1]
		fmt.Printf("\nRunning tests for %s provider...\n", providerName)

		provider, err := providerManager.GetProvider(providerName)
		if err != nil {
			log.Fatalf("Error getting provider %s: %v", providerName, err)
		}

		testInstanceOperations(ctx, providerName, provider, testSuite)
		testStorageOperations(ctx, providerName, provider, testSuite)
		testNetworkOperations(ctx, providerName, provider, testSuite)

		// Print summary for this provider
		testSuite.PrintSummary()
	} else {
		// Run a targeted test scenario
		fmt.Println("\nRunning targeted test scenario...")
		runMultiProviderTestScenario(ctx, providerManager, testSuite)
	}

	// Clean up
	errors := providerManager.CloseAllProviders()
	if len(errors) > 0 {
		for provider, err := range errors {
			log.Printf("Error closing provider %s: %v", provider, err)
		}
	}
}

// runMultiProviderTestScenario demonstrates a practical use case for the test suite
func runMultiProviderTestScenario(ctx context.Context, manager *cloud.ProviderManager, testSuite *testsuite.ProviderTestSuite) {
	fmt.Println("=== Multi-Provider Test Scenario ===")
	fmt.Println("This scenario tests a common multi-cloud workflow:")
	fmt.Println("1. Test network creation on each provider")
	fmt.Println("2. Test virtual machine deployments")
	fmt.Println("3. Test storage operations")
	fmt.Println("4. Test cross-provider connectivity")
	fmt.Println()

	// Step 1: Test network creation
	fmt.Println("Step 1: Testing network creation on each provider")
	networkIDs := make(map[string]string)

	providers := manager.ListProviders()
	for _, providerName := range providers {
		provider, _ := manager.GetProvider(providerName)
		fmt.Printf("Creating network on %s...\n", providerName)

		specs := cloud.NetworkSpecs{
			Name:   fmt.Sprintf("test-network-%d", time.Now().Unix()),
			CIDR:   "192.168.0.0/24",
			Region: testsuite.GetDefaultRegion(providerName),
			Tags: map[string]string{
				"purpose": "testing",
				"project": "novacron",
			},
		}

		testSuite.RunTest(ctx, providerName, "CreateNetwork", func() error {
			network, err := provider.CreateNetwork(ctx, specs)
			if err != nil {
				return err
			}

			networkIDs[providerName] = network.ID
			fmt.Printf("  Created network %s (%s): %s\n", network.Name, network.ID, network.CIDR)
			return nil
		})
	}

	// Step 2: Test instance creation in each network
	fmt.Println("\nStep 2: Testing VM deployments in each network")
	instanceIDs := make(map[string]string)

	for _, providerName := range providers {
		provider, _ := manager.GetProvider(providerName)
		networkID := networkIDs[providerName]

		if networkID == "" {
			fmt.Printf("Skipping instance creation on %s: no network available\n", providerName)
			continue
		}

		fmt.Printf("Creating instance on %s...\n", providerName)
		specs := cloud.InstanceSpecs{
			Name:           fmt.Sprintf("test-instance-%d", time.Now().Unix()),
			InstanceType:   testsuite.GetDefaultInstanceType(providerName),
			ImageID:        testsuite.GetDefaultImageID(providerName),
			Region:         testsuite.GetDefaultRegion(providerName),
			Zone:           testsuite.GetDefaultZone(providerName),
			CPUCores:       2,
			MemoryGB:       4,
			DiskGB:         20,
			NetworkID:      networkID,
			AssignPublicIP: true,
			Tags: map[string]string{
				"purpose": "testing",
				"project": "novacron",
			},
		}

		testSuite.RunTest(ctx, providerName, "CreateInstance", func() error {
			instance, err := provider.CreateInstance(ctx, specs)
			if err != nil {
				return err
			}

			instanceIDs[providerName] = instance.ID
			fmt.Printf("  Created instance %s (%s)\n", instance.Name, instance.ID)
			return nil
		})
	}

	// Step 3: Test storage operations and attachments
	fmt.Println("\nStep 3: Testing storage operations")
	volumeIDs := make(map[string]string)

	for _, providerName := range providers {
		provider, _ := manager.GetProvider(providerName)
		instanceID := instanceIDs[providerName]

		if instanceID == "" {
			fmt.Printf("Skipping storage tests on %s: no instance available\n", providerName)
			continue
		}

		fmt.Printf("Creating and attaching storage on %s...\n", providerName)

		// Create volume
		specs := cloud.StorageVolumeSpecs{
			Name:   fmt.Sprintf("test-volume-%d", time.Now().Unix()),
			SizeGB: 50,
			Type:   testsuite.GetDefaultStorageType(providerName),
			Region: testsuite.GetDefaultRegion(providerName),
			Zone:   testsuite.GetDefaultZone(providerName),
			Tags: map[string]string{
				"purpose": "testing",
				"project": "novacron",
			},
		}

		testSuite.RunTest(ctx, providerName, "CreateStorageVolume", func() error {
			volume, err := provider.CreateStorageVolume(ctx, specs)
			if err != nil {
				return err
			}

			volumeIDs[providerName] = volume.ID
			fmt.Printf("  Created volume %s (%s): %d GB\n", volume.Name, volume.ID, volume.SizeGB)
			return nil
		})

		// Attach volume
		volumeID := volumeIDs[providerName]
		if volumeID != "" {
			testSuite.RunTest(ctx, providerName, "AttachStorageVolume", func() error {
				opts := cloud.AttachOptions{
					DevicePath: testsuite.GetDefaultDevicePath(providerName),
					ReadOnly:   false,
				}

				err := provider.AttachStorageVolume(ctx, volumeID, instanceID, opts)
				if err != nil {
					return err
				}

				fmt.Printf("  Attached volume %s to instance %s\n", volumeID, instanceID)
				return nil
			})
		}
	}

	// Step 4: Test instance discovery across providers
	fmt.Println("\nStep 4: Testing instance discovery across all providers")

	testSuite.RunTest(ctx, "multi-provider", "ListAllInstances", func() error {
		allInstances, err := manager.GetInstancesAcrossProviders(ctx, cloud.ListOptions{})
		if err != nil {
			return err
		}

		totalCount := 0
		for provider, instances := range allInstances {
			totalCount += len(instances)
			fmt.Printf("  %s: %d instances\n", provider, len(instances))
		}

		fmt.Printf("  Total: %d instances across all providers\n", totalCount)
		return nil
	})

	// Step 5: Clean up resources
	fmt.Println("\nStep 5: Cleaning up all resources")

	// Detach and delete volumes
	for providerName, volumeID := range volumeIDs {
		if volumeID == "" {
			continue
		}

		provider, _ := manager.GetProvider(providerName)
		instanceID := instanceIDs[providerName]

		// Detach volume if attached
		if instanceID != "" {
			testSuite.RunTest(ctx, providerName, "DetachStorageVolume", func() error {
				err := provider.DetachStorageVolume(ctx, volumeID, instanceID)
				if err != nil {
					return err
				}

				fmt.Printf("  Detached volume %s from instance %s\n", volumeID, instanceID)
				return nil
			})
		}

		// Delete volume
		testSuite.RunTest(ctx, providerName, "DeleteStorageVolume", func() error {
			err := provider.DeleteStorageVolume(ctx, volumeID)
			if err != nil {
				return err
			}

			fmt.Printf("  Deleted volume %s\n", volumeID)
			return nil
		})
	}

	// Delete instances
	for providerName, instanceID := range instanceIDs {
		if instanceID == "" {
			continue
		}

		provider, _ := manager.GetProvider(providerName)

		testSuite.RunTest(ctx, providerName, "DeleteInstance", func() error {
			err := provider.DeleteInstance(ctx, instanceID)
			if err != nil {
				return err
			}

			fmt.Printf("  Deleted instance %s\n", instanceID)
			return nil
		})
	}

	// Delete networks
	for providerName, networkID := range networkIDs {
		if networkID == "" {
			continue
		}

		provider, _ := manager.GetProvider(providerName)

		testSuite.RunTest(ctx, providerName, "DeleteNetwork", func() error {
			err := provider.DeleteNetwork(ctx, networkID)
			if err != nil {
				return err
			}

			fmt.Printf("  Deleted network %s\n", networkID)
			return nil
		})
	}

	// Print test results
	testSuite.PrintSummary()
}

// testInstanceOperations runs instance-specific tests for a provider
func testInstanceOperations(ctx context.Context, providerName string, provider cloud.Provider, testSuite *testsuite.ProviderTestSuite) {
	fmt.Printf("  Testing instance operations for %s...\n", providerName)
	testSuite.TestInstanceOperations(ctx, providerName, provider)
}

// testStorageOperations runs storage-specific tests for a provider
func testStorageOperations(ctx context.Context, providerName string, provider cloud.Provider, testSuite *testsuite.ProviderTestSuite) {
	fmt.Printf("  Testing storage operations for %s...\n", providerName)
	testSuite.TestStorageOperations(ctx, providerName, provider)
}

// testNetworkOperations runs network-specific tests for a provider
func testNetworkOperations(ctx context.Context, providerName string, provider cloud.Provider, testSuite *testsuite.ProviderTestSuite) {
	fmt.Printf("  Testing network operations for %s...\n", providerName)
	testSuite.TestNetworkOperations(ctx, providerName, provider)
}

// initializeProviders initializes all providers with configuration from environment variables
func initializeProviders(manager *cloud.ProviderManager) {
	// AWS provider
	awsConfig := cloud.ProviderConfig{
		AuthConfig: map[string]string{
			"access_key_id":     os.Getenv("AWS_ACCESS_KEY_ID"),
			"secret_access_key": os.Getenv("AWS_SECRET_ACCESS_KEY"),
			"session_token":     os.Getenv("AWS_SESSION_TOKEN"),
		},
		DefaultRegion: "us-east-1",
		DefaultZone:   "us-east-1a",
		MaxRetries:    3,
		Timeout:       30 * time.Second,
	}

	// If AWS credentials not provided, use dummy values for testing
	if awsConfig.AuthConfig["access_key_id"] == "" {
		awsConfig.AuthConfig["access_key_id"] = "dummy_key"
		awsConfig.AuthConfig["secret_access_key"] = "dummy_secret"
	}

	err := manager.InitializeProvider("aws", awsConfig)
	if err != nil {
		log.Printf("Warning: Failed to initialize AWS provider: %v", err)
	}

	// Azure provider
	azureConfig := cloud.ProviderConfig{
		AuthConfig: map[string]string{
			"tenant_id":       os.Getenv("AZURE_TENANT_ID"),
			"client_id":       os.Getenv("AZURE_CLIENT_ID"),
			"client_secret":   os.Getenv("AZURE_CLIENT_SECRET"),
			"subscription_id": os.Getenv("AZURE_SUBSCRIPTION_ID"),
		},
		DefaultRegion: "eastus",
		DefaultZone:   "eastus-1",
		MaxRetries:    3,
		Timeout:       30 * time.Second,
	}

	// If Azure credentials not provided, use dummy values for testing
	if azureConfig.AuthConfig["tenant_id"] == "" {
		azureConfig.AuthConfig["tenant_id"] = "dummy_tenant"
		azureConfig.AuthConfig["client_id"] = "dummy_client"
		azureConfig.AuthConfig["client_secret"] = "dummy_secret"
		azureConfig.AuthConfig["subscription_id"] = "dummy_subscription"
	}

	err = manager.InitializeProvider("azure", azureConfig)
	if err != nil {
		log.Printf("Warning: Failed to initialize Azure provider: %v", err)
	}

	// GCP provider
	gcpConfig := cloud.ProviderConfig{
		AuthConfig: map[string]string{
			"credentials_file": os.Getenv("GOOGLE_APPLICATION_CREDENTIALS"),
		},
		ProjectID:     os.Getenv("GCP_PROJECT_ID"),
		DefaultRegion: "us-central1",
		DefaultZone:   "us-central1-a",
		MaxRetries:    3,
		Timeout:       30 * time.Second,
	}

	// If GCP credentials not provided, use dummy values for testing
	if gcpConfig.AuthConfig["credentials_file"] == "" {
		gcpConfig.AuthConfig["credentials_file"] = "dummy_credentials"
	}
	if gcpConfig.ProjectID == "" {
		gcpConfig.ProjectID = "dummy_project"
	}

	err = manager.InitializeProvider("gcp", gcpConfig)
	if err != nil {
		log.Printf("Warning: Failed to initialize GCP provider: %v", err)
	}
}

// PrintUsage shows the command-line usage for the testing example
func PrintUsage() {
	fmt.Println("Usage:")
	fmt.Println("  cloud-provider-tester                  - Run a multi-provider test scenario")
	fmt.Println("  cloud-provider-tester all              - Run all tests for all providers")
	fmt.Println("  cloud-provider-tester [provider_name]  - Run tests for a specific provider")
	fmt.Println("  cloud-provider-tester aws              - Run tests for AWS provider")
	fmt.Println("  cloud-provider-tester azure            - Run tests for Azure provider")
	fmt.Println("  cloud-provider-tester gcp              - Run tests for GCP provider")
}
