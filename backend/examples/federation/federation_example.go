package main

import (
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/federation"
)

func main() {
	fmt.Println("Starting NovaCron Federation Example")

	// Create a federation manager for the local cluster
	localManager := createLocalCluster()

	// Add remote clusters to the federation
	addRemoteClusters(localManager)

	// Create federation policies
	createFederationPolicies(localManager)

	// Create a federated resource pool
	createResourcePool(localManager)

	// Simulate a cross-cluster migration
	simulateCrossClusterMigration(localManager)

	// Print the federation status
	printFederationStatus(localManager)

	fmt.Println("Federation Example Complete")
}

// createLocalCluster creates a federation manager for the local cluster
func createLocalCluster() *federation.FederationManager {
	fmt.Println("\n--- Creating Local Cluster ---")

	// Create a federation manager
	manager := federation.NewFederationManager(
		"cluster-1",
		federation.PrimaryCluster,
		federation.MeshMode,
	)

	// Start the federation manager
	err := manager.Start()
	if err != nil {
		log.Fatalf("Failed to start federation manager: %v", err)
	}

	fmt.Println("Local cluster created and federation manager started")
	return manager
}

// addRemoteClusters adds remote clusters to the federation
func addRemoteClusters(manager *federation.FederationManager) {
	fmt.Println("\n--- Adding Remote Clusters ---")

	// Create and add the first remote cluster
	cluster1 := &federation.Cluster{
		ID:       "cluster-2",
		Name:     "Data Center East",
		Endpoint: "https://dc-east.example.com:8443",
		Role:     federation.PeerCluster,
		Resources: &federation.ClusterResources{
			TotalCPU:           1024,
			TotalMemoryGB:      4096,
			TotalStorageGB:     102400,
			AvailableCPU:       512,
			AvailableMemoryGB:  2048,
			AvailableStorageGB: 51200,
			NodeCount:          64,
			VMCount:            128,
		},
		LocationInfo: &federation.ClusterLocation{
			Region:      "us-east",
			Zone:        "us-east-1a",
			DataCenter:  "dc-east",
			City:        "New York",
			Country:     "USA",
			Coordinates: [2]float64{40.7128, -74.0060},
		},
	}

	// Add the first remote cluster
	err := manager.AddCluster(cluster1)
	if err != nil {
		log.Fatalf("Failed to add remote cluster 1: %v", err)
	}
	fmt.Println("Added remote cluster: Data Center East")

	// Create and add the second remote cluster
	cluster2 := &federation.Cluster{
		ID:       "cluster-3",
		Name:     "Data Center West",
		Endpoint: "https://dc-west.example.com:8443",
		Role:     federation.PeerCluster,
		Resources: &federation.ClusterResources{
			TotalCPU:           2048,
			TotalMemoryGB:      8192,
			TotalStorageGB:     204800,
			AvailableCPU:       1024,
			AvailableMemoryGB:  4096,
			AvailableStorageGB: 102400,
			NodeCount:          128,
			VMCount:            256,
		},
		LocationInfo: &federation.ClusterLocation{
			Region:      "us-west",
			Zone:        "us-west-1a",
			DataCenter:  "dc-west",
			City:        "San Francisco",
			Country:     "USA",
			Coordinates: [2]float64{37.7749, -122.4194},
		},
	}

	// Add the second remote cluster
	err = manager.AddCluster(cluster2)
	if err != nil {
		log.Fatalf("Failed to add remote cluster 2: %v", err)
	}
	fmt.Println("Added remote cluster: Data Center West")

	// Create and add the third remote cluster
	cluster3 := &federation.Cluster{
		ID:       "cluster-4",
		Name:     "Data Center Europe",
		Endpoint: "https://dc-europe.example.com:8443",
		Role:     federation.PeerCluster,
		Resources: &federation.ClusterResources{
			TotalCPU:           1536,
			TotalMemoryGB:      6144,
			TotalStorageGB:     153600,
			AvailableCPU:       768,
			AvailableMemoryGB:  3072,
			AvailableStorageGB: 76800,
			NodeCount:          96,
			VMCount:            192,
		},
		LocationInfo: &federation.ClusterLocation{
			Region:      "eu-central",
			Zone:        "eu-central-1a",
			DataCenter:  "dc-europe",
			City:        "Frankfurt",
			Country:     "Germany",
			Coordinates: [2]float64{50.1109, 8.6821},
		},
	}

	// Add the third remote cluster
	err = manager.AddCluster(cluster3)
	if err != nil {
		log.Fatalf("Failed to add remote cluster 3: %v", err)
	}
	fmt.Println("Added remote cluster: Data Center Europe")
}

// createFederationPolicies creates federation policies
func createFederationPolicies(manager *federation.FederationManager) {
	fmt.Println("\n--- Creating Federation Policies ---")

	// Create resource sharing policy
	resourcePolicy := &federation.FederationPolicy{
		ID:          "resource-sharing-policy-1",
		Name:        "Global Resource Sharing Policy",
		Description: "Policy for sharing resources between all clusters",
		ResourceSharingRules: map[string]interface{}{
			"max_cpu_share_percent":     50,
			"max_memory_share_percent":  40,
			"max_storage_share_percent": 30,
			"priority_clusters":         []string{"cluster-1", "cluster-3"},
		},
		Priority:  10,
		Enabled:   true,
		AppliesTo: []string{"cluster-1", "cluster-2", "cluster-3", "cluster-4"},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Add resource sharing policy
	err := manager.CreateFederationPolicy(resourcePolicy)
	if err != nil {
		log.Fatalf("Failed to create resource sharing policy: %v", err)
	}
	fmt.Println("Created resource sharing policy: Global Resource Sharing Policy")

	// Create migration policy
	migrationPolicy := &federation.FederationPolicy{
		ID:          "migration-policy-1",
		Name:        "Global Migration Policy",
		Description: "Policy for VM migrations between all clusters",
		MigrationRules: map[string]interface{}{
			"allow_live_migration":      true,
			"max_concurrent_migrations": 5,
			"bandwidth_limit_mbps":      1000,
			"compression_enabled":       true,
			"compression_level":         5,
		},
		Priority:  20,
		Enabled:   true,
		AppliesTo: []string{"cluster-1", "cluster-2", "cluster-3", "cluster-4"},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Add migration policy
	err = manager.CreateFederationPolicy(migrationPolicy)
	if err != nil {
		log.Fatalf("Failed to create migration policy: %v", err)
	}
	fmt.Println("Created migration policy: Global Migration Policy")

	// Create authorization policy
	authPolicy := &federation.FederationPolicy{
		ID:          "auth-policy-1",
		Name:        "Global Auth Policy",
		Description: "Policy for authorization between all clusters",
		AuthorizationRules: map[string]interface{}{
			"allow_admin_operations":    true,
			"allow_read_operations":     true,
			"allow_write_operations":    true,
			"allow_delete_operations":   true,
			"require_mutual_auth":       true,
			"token_expiry_seconds":      3600,
			"max_inactive_time_seconds": 86400,
		},
		Priority:  30,
		Enabled:   true,
		AppliesTo: []string{"cluster-1", "cluster-2", "cluster-3", "cluster-4"},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Add authorization policy
	err = manager.CreateFederationPolicy(authPolicy)
	if err != nil {
		log.Fatalf("Failed to create authorization policy: %v", err)
	}
	fmt.Println("Created authorization policy: Global Auth Policy")
}

// createResourcePool creates a federated resource pool
func createResourcePool(manager *federation.FederationManager) {
	fmt.Println("\n--- Creating Federated Resource Pool ---")

	// Create a federated resource pool
	pool := &federation.FederatedResourcePool{
		ID:          "high-performance-pool",
		Name:        "High Performance Computing Pool",
		Description: "Federated resource pool for high-performance computing workloads",
		ClusterAllocations: map[string]*federation.ResourceAllocation{
			"cluster-1": {
				CPU:       256,
				MemoryGB:  1024,
				StorageGB: 10240,
				Priority:  10,
				AllocationRules: map[string]interface{}{
					"overcommit_cpu":    1.5,
					"overcommit_memory": 1.2,
				},
			},
			"cluster-2": {
				CPU:       128,
				MemoryGB:  512,
				StorageGB: 5120,
				Priority:  5,
				AllocationRules: map[string]interface{}{
					"overcommit_cpu":    1.2,
					"overcommit_memory": 1.1,
				},
			},
			"cluster-3": {
				CPU:       512,
				MemoryGB:  2048,
				StorageGB: 20480,
				Priority:  20,
				AllocationRules: map[string]interface{}{
					"overcommit_cpu":    2.0,
					"overcommit_memory": 1.5,
				},
			},
		},
		PolicyID:  "resource-sharing-policy-1",
		TenantID:  "tenant-hpc-1",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Add the federated resource pool
	err := manager.CreateResourcePool(pool)
	if err != nil {
		log.Fatalf("Failed to create federated resource pool: %v", err)
	}
	fmt.Println("Created federated resource pool: High Performance Computing Pool")

	// Create another federated resource pool
	pool2 := &federation.FederatedResourcePool{
		ID:          "web-services-pool",
		Name:        "Web Services Pool",
		Description: "Federated resource pool for web services workloads",
		ClusterAllocations: map[string]*federation.ResourceAllocation{
			"cluster-1": {
				CPU:       128,
				MemoryGB:  512,
				StorageGB: 5120,
				Priority:  5,
				AllocationRules: map[string]interface{}{
					"overcommit_cpu":    2.0,
					"overcommit_memory": 1.5,
				},
			},
			"cluster-3": {
				CPU:       128,
				MemoryGB:  512,
				StorageGB: 5120,
				Priority:  5,
				AllocationRules: map[string]interface{}{
					"overcommit_cpu":    2.0,
					"overcommit_memory": 1.5,
				},
			},
			"cluster-4": {
				CPU:       256,
				MemoryGB:  1024,
				StorageGB: 10240,
				Priority:  10,
				AllocationRules: map[string]interface{}{
					"overcommit_cpu":    2.0,
					"overcommit_memory": 1.5,
				},
			},
		},
		PolicyID:  "resource-sharing-policy-1",
		TenantID:  "tenant-web-1",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Add the second federated resource pool
	err = manager.CreateResourcePool(pool2)
	if err != nil {
		log.Fatalf("Failed to create second federated resource pool: %v", err)
	}
	fmt.Println("Created federated resource pool: Web Services Pool")
}

// simulateCrossClusterMigration simulates a cross-cluster migration
func simulateCrossClusterMigration(manager *federation.FederationManager) {
	fmt.Println("\n--- Simulating Cross-Cluster Migration ---")

	// Create a migration job
	job := &federation.MigrationJob{
		ID:                   "migration-job-1",
		VMID:                 "vm-1234",
		SourceClusterID:      "cluster-1",
		DestinationClusterID: "cluster-3",
		State:                "starting",
		Progress:             0,
		StartTime:            time.Now(),
		Options: map[string]interface{}{
			"migration_type":       "live",
			"bandwidth_limit_mbps": 1000,
			"compression_enabled":  true,
			"compression_level":    7,
			"memory_dirty_rate":    0.05,
			"precopy_iterations":   5,
		},
	}

	// Start the migration
	err := manager.crossClusterMigration.StartMigration(job)
	if err != nil {
		log.Fatalf("Failed to start cross-cluster migration: %v", err)
	}
	fmt.Println("Started cross-cluster migration of VM vm-1234 from cluster-1 to cluster-3")

	// Simulate migration progress
	fmt.Println("Simulating migration progress...")
	for i := 0; i < 10; i++ {
		time.Sleep(500 * time.Millisecond) // Simulate time passing
		progress := (i + 1) * 10
		state := "in_progress"
		if progress >= 100 {
			state = "completed"
		}

		err = manager.crossClusterMigration.NotifyMigrationProgress("migration-job-1", progress, state, "")
		if err != nil {
			log.Fatalf("Failed to update migration progress: %v", err)
		}
		fmt.Printf("Migration progress: %d%%, state: %s\n", progress, state)
	}

	// Get final migration status
	updatedJob, err := manager.crossClusterMigration.GetMigrationJob("migration-job-1")
	if err != nil {
		log.Fatalf("Failed to get migration job: %v", err)
	}

	fmt.Printf("Migration completed - Total duration: %v\n", updatedJob.EndTime.Sub(updatedJob.StartTime))
}

// printFederationStatus prints the status of the federation
func printFederationStatus(manager *federation.FederationManager) {
	fmt.Println("\n--- Federation Status ---")

	// Get all clusters
	clusters := manager.ListClusters()
	fmt.Printf("Number of clusters in federation: %d\n", len(clusters))

	// Print cluster information
	for _, cluster := range clusters {
		fmt.Printf("Cluster: %s (%s)\n", cluster.Name, cluster.ID)
		fmt.Printf("  - Role: %s\n", cluster.Role)
		fmt.Printf("  - State: %s\n", cluster.State)
		if cluster.LocationInfo != nil {
			fmt.Printf("  - Location: %s, %s\n", cluster.LocationInfo.City, cluster.LocationInfo.Country)
		}
		if cluster.Resources != nil {
			fmt.Printf("  - CPU: %d cores (available: %d)\n", cluster.Resources.TotalCPU, cluster.Resources.AvailableCPU)
			fmt.Printf("  - Memory: %d GB (available: %d GB)\n", cluster.Resources.TotalMemoryGB, cluster.Resources.AvailableMemoryGB)
			fmt.Printf("  - Storage: %d GB (available: %d GB)\n", cluster.Resources.TotalStorageGB, cluster.Resources.AvailableStorageGB)
			fmt.Printf("  - VMs: %d\n", cluster.Resources.VMCount)
		}
		fmt.Println()
	}

	// Get all policies
	policies := manager.ListFederationPolicies()
	fmt.Printf("Number of federation policies: %d\n", len(policies))

	// Get all resource pools
	pools := manager.ListResourcePools("")
	fmt.Printf("Number of federated resource pools: %d\n", len(pools))
	for _, pool := range pools {
		fmt.Printf("  - Pool: %s (%s)\n", pool.Name, pool.ID)
		fmt.Printf("    Tenant: %s\n", pool.TenantID)
		fmt.Printf("    Clusters: %d\n", len(pool.ClusterAllocations))
		fmt.Println()
	}

	// Get all operations
	operations := manager.ListCrossClusterOperations("", "")
	fmt.Printf("Number of cross-cluster operations: %d\n", len(operations))
	for _, op := range operations {
		fmt.Printf("  - Operation: %s\n", op.ID)
		fmt.Printf("    Type: %s\n", op.Type)
		fmt.Printf("    Status: %s\n", op.Status)
		fmt.Printf("    Progress: %d%%\n", op.Progress)
		fmt.Println()
	}
}
