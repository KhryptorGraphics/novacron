package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func main() {
	log.Println("NovaCron VM Migration Example")
	
	// Create temporary directories for source and destination nodes
	sourceDir, err := os.MkdirTemp("", "novacron-example-source")
	if err != nil {
		log.Fatalf("Failed to create source directory: %v", err)
	}
	defer os.RemoveAll(sourceDir)
	log.Printf("Source directory: %s", sourceDir)

	destDir, err := os.MkdirTemp("", "novacron-example-dest")
	if err != nil {
		log.Fatalf("Failed to create destination directory: %v", err)
	}
	defer os.RemoveAll(destDir)
	log.Printf("Destination directory: %s", destDir)

	// Create source and destination migration managers
	sourceManager := vm.NewVMMigrationManager("source-node", sourceDir)
	destManager := vm.NewVMMigrationManager("dest-node", destDir)

	// Create a test VM
	vmID := "example-vm"
	vmSpec := vm.VMSpec{
		ID:       vmID,
		Name:     "Example VM",
		VCPU:     2,
		MemoryMB: 2048,
		DiskMB:   20480,
		Image:    "ubuntu-22.04",
	}

	// Create test files for VM
	createTestFiles(vmID, sourceDir)

	// Run examples of different migration types
	runColdMigration(sourceManager, destManager, vmID, vmSpec)
	runWarmMigration(sourceManager, destManager, vmID, vmSpec)
	runLiveMigration(sourceManager, destManager, vmID, vmSpec)
}

// Create test files for VM state and memory
func createTestFiles(vmID, sourceDir string) {
	// Create state file
	stateFile := filepath.Join(sourceDir, vmID+".state")
	stateData := []byte("This is mock VM state data for demonstration purposes.\n" +
		"In a real implementation, this would contain VM configuration.\n")
	if err := os.WriteFile(stateFile, stateData, 0644); err != nil {
		log.Fatalf("Failed to create state file: %v", err)
	}

	// Create memory file for warm migration
	memFile := filepath.Join(sourceDir, vmID+".memory")
	memData := []byte("This is mock VM memory data for demonstration purposes.\n" +
		"In a real implementation, this would contain the VM's memory state.\n")
	if err := os.WriteFile(memFile, memData, 0644); err != nil {
		log.Fatalf("Failed to create memory file: %v", err)
	}

	// Create memory iteration files for live migration
	for i := 1; i <= 3; i++ {
		iterFile := filepath.Join(sourceDir, fmt.Sprintf("%s.memory.%c", vmID, '0'+i))
		iterData := []byte(fmt.Sprintf("This is mock VM memory data for iteration %d.\n"+
			"In a real implementation, this would contain memory pages modified during iteration %d.\n", i, i))
		if err := os.WriteFile(iterFile, iterData, 0644); err != nil {
			log.Fatalf("Failed to create memory iteration file: %v", err)
		}
	}
}

// Run a cold migration example
func runColdMigration(sourceManager, destManager *vm.VMMigrationManager, vmID string, vmSpec vm.VMSpec) {
	log.Println("\n==== Cold Migration Example ====")
	
	// Create migration object
	migration := &vm.VMMigration{
		ID:                "migration-cold-" + vmID,
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              vm.MigrationTypeCold,
		Status:            vm.MigrationStatusPending,
		VMSpec:            vmSpec,
		CreatedAt:         time.Now(),
	}

	// Execute migration
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	log.Println("Starting cold migration...")
	start := time.Now()
	
	err := sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		log.Printf("Cold migration failed: %v", err)
		return
	}

	duration := time.Since(start)
	log.Printf("Cold migration completed successfully in %v", duration)
	log.Printf("Migration status: %s, Progress: %.1f%%", migration.Status, migration.Progress)
}

// Run a warm migration example
func runWarmMigration(sourceManager, destManager *vm.VMMigrationManager, vmID string, vmSpec vm.VMSpec) {
	log.Println("\n==== Warm Migration Example ====")
	
	// Create migration object
	migration := &vm.VMMigration{
		ID:                "migration-warm-" + vmID,
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              vm.MigrationTypeWarm,
		Status:            vm.MigrationStatusPending,
		VMSpec:            vmSpec,
		CreatedAt:         time.Now(),
	}

	// Execute migration
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	log.Println("Starting warm migration...")
	start := time.Now()
	
	err := sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		log.Printf("Warm migration failed: %v", err)
		return
	}

	duration := time.Since(start)
	log.Printf("Warm migration completed successfully in %v", duration)
	log.Printf("Migration status: %s, Progress: %.1f%%", migration.Status, migration.Progress)
}

// Run a live migration example
func runLiveMigration(sourceManager, destManager *vm.VMMigrationManager, vmID string, vmSpec vm.VMSpec) {
	log.Println("\n==== Live Migration Example ====")
	
	// Create migration object
	migration := &vm.VMMigration{
		ID:                "migration-live-" + vmID,
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              vm.MigrationTypeLive,
		Status:            vm.MigrationStatusPending,
		VMSpec:            vmSpec,
		CreatedAt:         time.Now(),
		Options: map[string]string{
			"iterations": "3",
		},
	}

	// Execute migration
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	log.Println("Starting live migration...")
	start := time.Now()
	
	err := sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		log.Printf("Live migration failed: %v", err)
		return
	}

	duration := time.Since(start)
	log.Printf("Live migration completed successfully in %v", duration)
	log.Printf("Migration status: %s, Progress: %.1f%%", migration.Status, migration.Progress)
}
