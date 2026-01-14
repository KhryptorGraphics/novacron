// Package main demonstrates usage of the NovaCron multi-cloud adapters
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/adapters/pkg/interfaces"
	"github.com/khryptorgraphics/novacron/adapters/pkg/factory"
	"github.com/khryptorgraphics/novacron/adapters/pkg/aws"
	"github.com/khryptorgraphics/novacron/adapters/pkg/azure"
	"github.com/khryptorgraphics/novacron/adapters/pkg/gcp"
)

func main() {
	ctx := context.Background()

	// Example 1: AWS adapter usage
	fmt.Println("=== AWS Adapter Example ===")
	if err := demonstrateAWS(ctx); err != nil {
		log.Printf("AWS demo failed: %v", err)
	}

	// Example 2: Azure adapter usage
	fmt.Println("\n=== Azure Adapter Example ===")
	if err := demonstrateAzure(ctx); err != nil {
		log.Printf("Azure demo failed: %v", err)
	}

	// Example 3: GCP adapter usage
	fmt.Println("\n=== GCP Adapter Example ===")
	if err := demonstrateGCP(ctx); err != nil {
		log.Printf("GCP demo failed: %v", err)
	}

	// Example 4: Factory usage
	fmt.Println("\n=== Factory Example ===")
	demonstrateFactory(ctx)
}

func demonstrateAWS(ctx context.Context) error {
	// Create AWS configuration
	config := &aws.Config{
		Region:          "us-west-2",
		AccessKeyID:     "your-access-key-id",
		SecretAccessKey: "your-secret-access-key",
	}

	// Create adapter using factory
	factory := factory.NewAdapterFactory()
	adapter, err := factory.CreateAdapter("aws", config)
	if err != nil {
		return fmt.Errorf("failed to create AWS adapter: %w", err)
	}

	// Validate credentials (skip if no real credentials)
	fmt.Printf("AWS Adapter: %s v%s\n", adapter.Name(), adapter.Version())
	fmt.Printf("Supported regions: %v\n", adapter.SupportedRegions()[:5]) // Show first 5
	fmt.Printf("Supported instance types: %v\n", adapter.SupportedInstanceTypes()[:5])

	// Create instance request
	createReq := &interfaces.CreateInstanceRequest{
		Name:           "demo-instance",
		ImageID:        "ami-0c02fb55956c7d316", // Amazon Linux 2 AMI
		InstanceType:   "t3.micro",
		KeyPairName:    "my-key-pair",
		RootVolumeSize: 20,
		Tags: map[string]string{
			"Environment": "demo",
			"Purpose":     "testing",
		},
	}

	fmt.Printf("Would create instance: %+v\n", createReq)

	// In a real scenario:
	// instance, err := adapter.CreateInstance(ctx, createReq)
	// if err != nil {
	//     return fmt.Errorf("failed to create instance: %w", err)
	// }
	// fmt.Printf("Created instance: %s\n", instance.ID)

	return nil
}

func demonstrateAzure(ctx context.Context) error {
	// Create Azure configuration
	config := &azure.Config{
		SubscriptionID: "your-subscription-id",
		TenantID:       "your-tenant-id",
		ClientID:       "your-client-id",
		ClientSecret:   "your-client-secret",
		ResourceGroup:  "novacron-demo",
		Location:       "eastus",
	}

	// Create adapter
	adapter := azure.NewAdapter()
	if err := adapter.Configure(config); err != nil {
		return fmt.Errorf("failed to configure Azure adapter: %w", err)
	}

	fmt.Printf("Azure Adapter: %s v%s\n", adapter.Name(), adapter.Version())
	fmt.Printf("Supported regions: %v\n", adapter.SupportedRegions()[:5])
	fmt.Printf("Supported VM sizes: %v\n", adapter.SupportedInstanceTypes()[:5])

	// Create instance request
	createReq := &interfaces.CreateInstanceRequest{
		Name:           "demo-vm",
		ImageID:        "Canonical:UbuntuServer:18.04-LTS:latest",
		InstanceType:   "Standard_B1s",
		KeyPairName:    "ssh-rsa AAAAB3NzaC1yc2E...",
		RootVolumeSize: 30,
		Tags: map[string]string{
			"Environment": "demo",
			"Purpose":     "testing",
		},
	}

	fmt.Printf("Would create VM: %+v\n", createReq)

	return nil
}

func demonstrateGCP(ctx context.Context) error {
	// Create GCP configuration
	config := &gcp.Config{
		ProjectID:       "your-project-id",
		Zone:           "us-central1-a",
		Region:         "us-central1",
		CredentialsFile: "/path/to/service-account-key.json",
	}

	// Create adapter
	adapter := gcp.NewAdapter()
	if err := adapter.Configure(config); err != nil {
		return fmt.Errorf("failed to configure GCP adapter: %w", err)
	}

	fmt.Printf("GCP Adapter: %s v%s\n", adapter.Name(), adapter.Version())
	fmt.Printf("Supported regions: %v\n", adapter.SupportedRegions()[:5])
	fmt.Printf("Supported machine types: %v\n", adapter.SupportedInstanceTypes()[:5])

	// Create instance request
	createReq := &interfaces.CreateInstanceRequest{
		Name:           "demo-instance",
		ImageID:        "projects/ubuntu-os-cloud/global/images/family/ubuntu-2004-lts",
		InstanceType:   "e2-micro",
		RootVolumeSize: 20,
		Tags: map[string]string{
			"environment": "demo",
			"purpose":     "testing",
		},
	}

	fmt.Printf("Would create instance: %+v\n", createReq)

	return nil
}

func demonstrateFactory(ctx context.Context) {
	factory := factory.NewAdapterFactory()

	fmt.Printf("Supported providers: %v\n", factory.SupportedProviders())

	// Demonstrate multi-cloud instance management
	providers := []string{"aws", "azure", "gcp"}
	
	for _, provider := range providers {
		fmt.Printf("\n--- %s Multi-cloud Operations ---\n", provider)
		
		var config interfaces.CloudConfig
		switch provider {
		case "aws":
			config = &aws.Config{
				Region:          "us-west-2",
				AccessKeyID:     "demo-key",
				SecretAccessKey: "demo-secret",
			}
		case "azure":
			config = &azure.Config{
				SubscriptionID: "demo-subscription",
				TenantID:       "demo-tenant",
				ClientID:       "demo-client",
				ClientSecret:   "demo-secret",
				ResourceGroup:  "demo-rg",
				Location:       "eastus",
			}
		case "gcp":
			config = &gcp.Config{
				ProjectID: "demo-project",
				Zone:     "us-central1-a",
				Region:   "us-central1",
			}
		}

		adapter, err := factory.CreateAdapter(provider, config)
		if err != nil {
			log.Printf("Failed to create %s adapter: %v", provider, err)
			continue
		}

		// Get adapter status
		status, err := adapter.GetStatus(ctx)
		if err != nil {
			log.Printf("Failed to get %s status: %v", provider, err)
			continue
		}

		fmt.Printf("Provider: %s\n", status.Provider)
		fmt.Printf("Status: %s\n", status.Status)
		fmt.Printf("Capabilities: %v\n", status.Capabilities)
		
		// Demonstrate unified API across providers
		demonstrateUnifiedAPI(ctx, adapter)
	}
}

func demonstrateUnifiedAPI(ctx context.Context, adapter interfaces.CloudAdapter) {
	fmt.Println("Unified API demonstration:")

	// List instances (would work across all providers)
	filters := &interfaces.ListInstanceFilters{
		States: []interfaces.InstanceState{
			interfaces.InstanceStateRunning,
			interfaces.InstanceStateStopped,
		},
	}

	fmt.Printf("Listing instances with filters: %+v\n", filters)
	
	// In real usage:
	// instances, err := adapter.ListInstances(ctx, filters)
	// if err != nil {
	//     log.Printf("Failed to list instances: %v", err)
	//     return
	// }
	// 
	// for _, instance := range instances {
	//     fmt.Printf("Instance: %s (%s) - %s\n", 
	//         instance.Name, instance.ID, instance.State)
	// }

	// Demonstrate cost analysis
	costOpts := &interfaces.CostOptions{
		StartTime:   time.Now().AddDate(0, -1, 0), // Last month
		EndTime:     time.Now(),
		Granularity: "DAILY",
	}

	fmt.Printf("Cost analysis options: %+v\n", costOpts)
	
	// In real usage:
	// costData, err := adapter.GetCosts(ctx, costOpts)
	// if err != nil {
	//     log.Printf("Failed to get cost data: %v", err)
	//     return
	// }
	// 
	// fmt.Printf("Total cost: %.2f %s\n", 
	//     costData.TotalCost, costData.Currency)

	fmt.Println("Unified API works across all cloud providers!")
}

// Example of creating a multi-cloud deployment manager
type MultiCloudManager struct {
	adapters map[string]interfaces.CloudAdapter
}

func NewMultiCloudManager() *MultiCloudManager {
	return &MultiCloudManager{
		adapters: make(map[string]interfaces.CloudAdapter),
	}
}

func (m *MultiCloudManager) AddProvider(name string, adapter interfaces.CloudAdapter) {
	m.adapters[name] = adapter
}

func (m *MultiCloudManager) DeployToAll(ctx context.Context, req *interfaces.CreateInstanceRequest) map[string]*interfaces.Instance {
	results := make(map[string]*interfaces.Instance)
	
	for provider, adapter := range m.adapters {
		instance, err := adapter.CreateInstance(ctx, req)
		if err != nil {
			log.Printf("Failed to deploy to %s: %v", provider, err)
			continue
		}
		results[provider] = instance
		log.Printf("Successfully deployed to %s: %s", provider, instance.ID)
	}
	
	return results
}

func (m *MultiCloudManager) GetCostSummary(ctx context.Context) map[string]*interfaces.CostData {
	results := make(map[string]*interfaces.CostData)
	
	opts := &interfaces.CostOptions{
		StartTime:   time.Now().AddDate(0, -1, 0),
		EndTime:     time.Now(),
		Granularity: "MONTHLY",
	}
	
	for provider, adapter := range m.adapters {
		costData, err := adapter.GetCosts(ctx, opts)
		if err != nil {
			log.Printf("Failed to get costs for %s: %v", provider, err)
			continue
		}
		results[provider] = costData
	}
	
	return results
}