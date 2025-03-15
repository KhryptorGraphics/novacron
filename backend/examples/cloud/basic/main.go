package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/novacron/backend/core/cloud"
)

func main() {
	// Create a new provider manager
	providerManager := cloud.NewProviderManager()

	// Register provider types
	providerManager.RegisterProviderType(cloud.NewAWSProvider())
	providerManager.RegisterProviderType(cloud.NewAzureProvider())
	providerManager.RegisterProviderType(cloud.NewGCPProvider())

	// Initialize AWS provider
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

	err := providerManager.InitializeProvider("aws", awsConfig)
	if err != nil {
		log.Fatalf("Failed to initialize AWS provider: %v", err)
	}

	// Initialize Azure provider
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

	err = providerManager.InitializeProvider("azure", azureConfig)
	if err != nil {
		log.Fatalf("Failed to initialize Azure provider: %v", err)
	}

	// Initialize GCP provider
	gcpConfig := cloud.ProviderConfig{
		AuthConfig: map[string]string{
			// GCP typically uses application default credentials
			// or a service account key file
			"credentials_file": os.Getenv("GOOGLE_APPLICATION_CREDENTIALS"),
		},
		ProjectID:     os.Getenv("GCP_PROJECT_ID"),
		DefaultRegion: "us-central1",
		DefaultZone:   "us-central1-a",
		MaxRetries:    3,
		Timeout:       30 * time.Second,
	}

	err = providerManager.InitializeProvider("gcp", gcpConfig)
	if err != nil {
		log.Fatalf("Failed to initialize GCP provider: %v", err)
	}

	// Set AWS as the default provider
	err = providerManager.SetDefaultProvider("aws")
	if err != nil {
		log.Fatalf("Failed to set default provider: %v", err)
	}

	// Examples of using the provider manager
	ctx := context.Background()

	// Example 1: List instances across all providers
	fmt.Println("=== Example 1: List instances across all providers ===")
	listInstancesAcrossProviders(ctx, providerManager)

	// Example 2: Create an instance with the default provider
	fmt.Println("\n=== Example 2: Create an instance with the default provider ===")
	createInstance(ctx, providerManager, "")

	// Example 3: Find the cheapest provider for an instance type
	fmt.Println("\n=== Example 3: Find the cheapest provider for an instance type ===")
	findCheapestProvider(ctx, providerManager, "t3.medium")

	// Example 4: Compare pricing across providers
	fmt.Println("\n=== Example 4: Compare pricing across providers ===")
	comparePricing(ctx, providerManager)

	// Example 5: Create and manage hybrid resources with the orchestrator
	fmt.Println("\n=== Example 5: Hybrid resource management ===")
	hybridResourceManagement(ctx)

	// Clean up: Close all providers
	errors := providerManager.CloseAllProviders()
	if len(errors) > 0 {
		for _, err := range errors {
			log.Printf("Error during provider cleanup: %v", err)
		}
	}
}

// listInstancesAcrossProviders shows how to list instances from all providers
func listInstancesAcrossProviders(ctx context.Context, manager *cloud.ProviderManager) {
	instances, err := manager.GetInstancesAcrossProviders(ctx, cloud.ListOptions{
		Limit: 10,
	})

	if err != nil {
		log.Printf("Error getting instances: %v", err)
		return
	}

	for provider, providerInstances := range instances {
		fmt.Printf("Provider: %s\n", provider)
		fmt.Printf("Instances: %d\n", len(providerInstances))
		for i, instance := range providerInstances {
			if i >= 2 {
				fmt.Printf("  ... and %d more\n", len(providerInstances)-2)
				break
			}
			fmt.Printf("  - %s (%s): %s, Type: %s, State: %s\n",
				instance.Name, instance.ID, instance.Region, instance.InstanceType, instance.State)
		}
		fmt.Println()
	}
}

// createInstance demonstrates creating a new instance
func createInstance(ctx context.Context, manager *cloud.ProviderManager, providerName string) {
	provider, err := manager.GetProvider(providerName)
	if err != nil {
		log.Printf("Error getting provider: %v", err)
		return
	}

	specs := cloud.InstanceSpecs{
		Name:           "novacron-example-vm",
		InstanceType:   "t3.medium", // AWS instance type - would be different for other providers
		ImageID:        "ami-12345678",
		Region:         "us-east-1",
		Zone:           "us-east-1a",
		CPUCores:       2,
		MemoryGB:       4,
		DiskGB:         30,
		AssignPublicIP: true,
		Tags: map[string]string{
			"environment": "test",
			"project":     "novacron",
			"purpose":     "example",
		},
	}

	instance, err := provider.CreateInstance(ctx, specs)
	if err != nil {
		log.Printf("Error creating instance: %v", err)
		return
	}

	fmt.Printf("Created new instance:\n")
	fmt.Printf("  ID: %s\n", instance.ID)
	fmt.Printf("  Name: %s\n", instance.Name)
	fmt.Printf("  Type: %s\n", instance.InstanceType)
	fmt.Printf("  State: %s\n", instance.State)
	fmt.Printf("  Region: %s, Zone: %s\n", instance.Region, instance.Zone)
}

// findCheapestProvider demonstrates how to find the cheapest provider for an instance type
func findCheapestProvider(ctx context.Context, manager *cloud.ProviderManager, instanceType string) {
	provider, price, err := manager.FindCheapestProvider(ctx, instanceType)
	if err != nil {
		log.Printf("Error finding cheapest provider: %v", err)
		return
	}

	fmt.Printf("Cheapest provider for %s: %s at $%.4f/hour\n", instanceType, provider, price)
}

// comparePricing compares pricing for resources across providers
func comparePricing(ctx context.Context, manager *cloud.ProviderManager) {
	// Get instance pricing across providers
	instancePricing, err := manager.GetPricingAcrossProviders(ctx, "instance")
	if err != nil {
		log.Printf("Error getting instance pricing: %v", err)
		return
	}

	fmt.Println("Instance pricing comparison:")
	fmt.Println("----------------------------")

	// Define instance types to compare (with equivalents across providers)
	instanceTypeMap := map[string]map[string]string{
		"aws": {
			"small":  "t3.small",
			"medium": "t3.medium",
			"large":  "t3.large",
		},
		"azure": {
			"small":  "Standard_B2s",
			"medium": "Standard_D2s_v3",
			"large":  "Standard_D4s_v3",
		},
		"gcp": {
			"small":  "e2-small",
			"medium": "e2-medium",
			"large":  "e2-standard-2",
		},
	}

	// Compare pricing for each size category
	for size, _ := range instanceTypeMap["aws"] {
		fmt.Printf("\nSize category: %s\n", size)
		fmt.Printf("----------------\n")

		for provider, types := range instanceTypeMap {
			instanceType := types[size]
			if pricing, ok := instancePricing[provider]; ok {
				if price, ok := pricing[instanceType]; ok {
					fmt.Printf("  %s: %s = $%.4f/hour\n", provider, instanceType, price)
				} else {
					fmt.Printf("  %s: %s = price not available\n", provider, instanceType)
				}
			}
		}
	}

	// Compare storage pricing
	storagePricing, err := manager.GetPricingAcrossProviders(ctx, "storage")
	if err != nil {
		log.Printf("Error getting storage pricing: %v", err)
		return
	}

	fmt.Println("\nStorage pricing comparison:")
	fmt.Println("---------------------------")

	// Define storage types to compare
	storageTypeMap := map[string]map[string]string{
		"aws": {
			"standard": "standard",
			"ssd":      "gp2",
			"premium":  "io1",
		},
		"azure": {
			"standard": "Standard_LRS",
			"ssd":      "Premium_LRS",
			"premium":  "Premium_LRS",
		},
		"gcp": {
			"standard": "pd-standard",
			"ssd":      "pd-balanced",
			"premium":  "pd-ssd",
		},
	}

	// Compare pricing for each storage type
	for storageClass, _ := range storageTypeMap["aws"] {
		fmt.Printf("\nStorage class: %s\n", storageClass)
		fmt.Printf("----------------\n")

		for provider, types := range storageTypeMap {
			storageType := types[storageClass]
			if pricing, ok := storagePricing[provider]; ok {
				if price, ok := pricing[storageType]; ok {
					fmt.Printf("  %s: %s = $%.4f/GB/month\n", provider, storageType, price)
				} else {
					fmt.Printf("  %s: %s = price not available\n", provider, storageType)
				}
			}
		}
	}
}

// hybridResourceManagement demonstrates the hybrid cloud orchestrator
func hybridResourceManagement(ctx context.Context) {
	// Create a hybrid cloud orchestrator
	orchestrator := cloud.NewHybridCloudOrchestrator()

	// Register providers
	awsProvider := cloud.NewAWSProvider()
	azureProvider := cloud.NewAzureProvider()
	gcpProvider := cloud.NewGCPProvider()

	// Initialize providers (simplified - in a real app, use proper credentials)
	awsProvider.Initialize(cloud.ProviderConfig{
		AuthConfig: map[string]string{
			"access_key_id":     "dummy_key",
			"secret_access_key": "dummy_secret",
		},
		DefaultRegion: "us-east-1",
	})

	azureProvider.Initialize(cloud.ProviderConfig{
		AuthConfig: map[string]string{
			"tenant_id":     "dummy_tenant",
			"client_id":     "dummy_client",
			"client_secret": "dummy_secret",
		},
		DefaultRegion: "eastus",
	})

	gcpProvider.Initialize(cloud.ProviderConfig{
		AuthConfig: map[string]string{
			"credentials_file": "dummy_path",
		},
		ProjectID:     "dummy_project",
		DefaultRegion: "us-central1",
	})

	// Register providers with the orchestrator
	orchestrator.RegisterProvider(awsProvider)
	orchestrator.RegisterProvider(azureProvider)
	orchestrator.RegisterProvider(gcpProvider)

	// Set a provider selection strategy
	orchestrator.SetSelectionStrategy(cloud.SelectByPrice)

	// Create hybrid resources
	fmt.Println("Creating hybrid resources based on price optimization:")

	// Define instance specs for different workloads
	webServerSpecs := cloud.InstanceSpecs{
		Name:           "web-server",
		InstanceType:   "t3.medium", // Will be mapped to appropriate type per provider
		ImageID:        "ubuntu-20.04",
		CPUCores:       2,
		MemoryGB:       4,
		DiskGB:         30,
		AssignPublicIP: true,
	}

	databaseSpecs := cloud.InstanceSpecs{
		Name:         "database-server",
		InstanceType: "m5.large", // Will be mapped to appropriate type per provider
		ImageID:      "ubuntu-20.04",
		CPUCores:     2,
		MemoryGB:     8,
		DiskGB:       100,
	}

	// Define storage volumes
	logStorageSpecs := cloud.StorageVolumeSpecs{
		Name:   "log-storage",
		SizeGB: 500,
		Type:   "gp2", // Will be mapped to appropriate type per provider
	}

	dbStorageSpecs := cloud.StorageVolumeSpecs{
		Name:   "db-storage",
		SizeGB: 1000,
		Type:   "io1", // Will be mapped to appropriate type per provider
	}

	// Allocate resources with cost optimization
	allocation, err := orchestrator.AllocateResources(
		ctx,
		[]cloud.InstanceSpecs{webServerSpecs, databaseSpecs},
		[]cloud.StorageVolumeSpecs{logStorageSpecs, dbStorageSpecs},
	)

	if err != nil {
		fmt.Printf("Error allocating resources: %v\n", err)
		return
	}

	// Print allocation results
	fmt.Println("Resource allocation results:")
	for provider, resources := range allocation.AllocationByProvider {
		fmt.Printf("  Provider: %s\n", provider)
		fmt.Printf("    Resources: %v\n", resources)
		fmt.Printf("    Estimated cost: $%.2f/hour\n", allocation.CostByProvider[provider])
	}
	fmt.Printf("  Total estimated cost: $%.2f/hour\n", allocation.TotalCostEstimate)

	// Get cost optimization recommendations
	recommendations, err := orchestrator.GetCostOptimizationRecommendations(ctx)
	if err != nil {
		fmt.Printf("Error getting recommendations: %v\n", err)
		return
	}

	fmt.Println("\nCost optimization recommendations:")
	for _, rec := range recommendations {
		fmt.Printf("  Resource: %s (%s)\n", rec.ResourceID, rec.ResourceType)
		fmt.Printf("    Current cost: $%.2f/month\n", rec.CurrentCost*730) // 730 hours in a month
		fmt.Printf("    Recommendation: %s\n", rec.RecommendedAction)
		if rec.TargetProvider != "" {
			fmt.Printf("    Target provider: %s\n", rec.TargetProvider)
		}
		fmt.Printf("    Expected savings: $%.2f/month\n", rec.ExpectedSavings*730)
		fmt.Printf("    Reason: %s\n", rec.Reason)
		fmt.Printf("    Confidence: %.1f%%\n", rec.Confidence*100)
		fmt.Println()
	}
}

// printJSON pretty-prints an object as JSON
func printJSON(obj interface{}) {
	data, err := json.MarshalIndent(obj, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling to JSON: %v\n", err)
		return
	}
	fmt.Println(string(data))
}
