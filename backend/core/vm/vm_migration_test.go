package vm

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestColdMigration(t *testing.T) {
	// Test setup: create temporary test directories
	sourceDir, err := os.MkdirTemp("", "novacron-test-source")
	if err != nil {
		t.Fatalf("Failed to create temporary source directory: %v", err)
	}
	defer os.RemoveAll(sourceDir)

	destDir, err := os.MkdirTemp("", "novacron-test-dest")
	if err != nil {
		t.Fatalf("Failed to create temporary destination directory: %v", err)
	}
	defer os.RemoveAll(destDir)

	// Create mock VM info
	vmID := "test-vm-cold"
	vmSpec := VMSpec{
		ID:       vmID,
		Name:     "TestVM",
		VCPU:     2,
		MemoryMB: 1024,
		DiskMB:   5120,
		Image:    "test-image",
	}

	// Create mock VM data file
	dataFile := filepath.Join(sourceDir, vmID+".state")
	mockData := []byte("mock VM state data for testing")
	if err := os.WriteFile(dataFile, mockData, 0644); err != nil {
		t.Fatalf("Failed to create mock VM data: %v", err)
	}

	// Create source and destination VM managers
	sourceManager := NewVMMigrationManager("source-node", sourceDir)
	destManager := NewVMMigrationManager("dest-node", destDir)

	// Execute cold migration
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	migration := &VMMigration{
		ID:                "migration-" + vmID,
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              MigrationTypeCold,
		Status:            MigrationStatusPending,
		VMSpec:            vmSpec,
		CreatedAt:         time.Now(),
	}

	// Execute migration
	err = sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		t.Fatalf("Cold migration failed: %v", err)
	}

	// Verify migration completed successfully
	if migration.Status != MigrationStatusCompleted {
		t.Errorf("Migration status is %s, expected %s", migration.Status, MigrationStatusCompleted)
	}

	// Verify VM state file was transferred
	destFile := filepath.Join(destDir, vmID+".state")
	destData, err := os.ReadFile(destFile)
	if err != nil {
		t.Fatalf("Failed to read destination VM state file: %v", err)
	}

	if string(destData) != string(mockData) {
		t.Errorf("Destination data doesn't match source data")
	}
}

func TestWarmMigration(t *testing.T) {
	// Test setup: create temporary test directories
	sourceDir, err := os.MkdirTemp("", "novacron-test-source")
	if err != nil {
		t.Fatalf("Failed to create temporary source directory: %v", err)
	}
	defer os.RemoveAll(sourceDir)

	destDir, err := os.MkdirTemp("", "novacron-test-dest")
	if err != nil {
		t.Fatalf("Failed to create temporary destination directory: %v", err)
	}
	defer os.RemoveAll(destDir)

	// Create mock VM info
	vmID := "test-vm-warm"
	vmSpec := VMSpec{
		ID:       vmID,
		Name:     "TestVM",
		VCPU:     2,
		MemoryMB: 1024,
		DiskMB:   5120,
		Image:    "test-image",
	}

	// Create mock VM data files
	stateFile := filepath.Join(sourceDir, vmID+".state")
	mockState := []byte("mock VM state data for testing")
	if err := os.WriteFile(stateFile, mockState, 0644); err != nil {
		t.Fatalf("Failed to create mock VM state: %v", err)
	}

	memFile := filepath.Join(sourceDir, vmID+".memory")
	mockMem := []byte("mock VM memory data for testing warm migration")
	if err := os.WriteFile(memFile, mockMem, 0644); err != nil {
		t.Fatalf("Failed to create mock VM memory: %v", err)
	}

	// Create source and destination VM managers
	sourceManager := NewVMMigrationManager("source-node", sourceDir)
	destManager := NewVMMigrationManager("dest-node", destDir)

	// Execute warm migration
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	migration := &VMMigration{
		ID:                "migration-" + vmID,
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              MigrationTypeWarm,
		Status:            MigrationStatusPending,
		VMSpec:            vmSpec,
		CreatedAt:         time.Now(),
	}

	// Execute migration
	err = sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		t.Fatalf("Warm migration failed: %v", err)
	}

	// Verify migration completed successfully
	if migration.Status != MigrationStatusCompleted {
		t.Errorf("Migration status is %s, expected %s", migration.Status, MigrationStatusCompleted)
	}

	// Verify VM state files were transferred
	destStateFile := filepath.Join(destDir, vmID+".state")
	destStateData, err := os.ReadFile(destStateFile)
	if err != nil {
		t.Fatalf("Failed to read destination VM state file: %v", err)
	}

	if string(destStateData) != string(mockState) {
		t.Errorf("Destination state data doesn't match source data")
	}

	destMemFile := filepath.Join(destDir, vmID+".memory")
	destMemData, err := os.ReadFile(destMemFile)
	if err != nil {
		t.Fatalf("Failed to read destination VM memory file: %v", err)
	}

	if string(destMemData) != string(mockMem) {
		t.Errorf("Destination memory data doesn't match source data")
	}
}

func TestLiveMigration(t *testing.T) {
	// Test setup: create temporary test directories
	sourceDir, err := os.MkdirTemp("", "novacron-test-source")
	if err != nil {
		t.Fatalf("Failed to create temporary source directory: %v", err)
	}
	defer os.RemoveAll(sourceDir)

	destDir, err := os.MkdirTemp("", "novacron-test-dest")
	if err != nil {
		t.Fatalf("Failed to create temporary destination directory: %v", err)
	}
	defer os.RemoveAll(destDir)

	// Create mock VM info
	vmID := "test-vm-live"
	vmSpec := VMSpec{
		ID:       vmID,
		Name:     "TestVM",
		VCPU:     2,
		MemoryMB: 1024,
		DiskMB:   5120,
		Image:    "test-image",
	}

	// Create mock VM data files with iterations to simulate live migration
	for i := 1; i <= 3; i++ {
		memFile := filepath.Join(sourceDir, vmID+".memory."+string(rune('0'+i)))
		mockMem := []byte("mock VM memory data iteration " + string(rune('0'+i)))
		if err := os.WriteFile(memFile, mockMem, 0644); err != nil {
			t.Fatalf("Failed to create mock VM memory: %v", err)
		}
	}

	stateFile := filepath.Join(sourceDir, vmID+".state")
	mockState := []byte("mock VM state data for testing")
	if err := os.WriteFile(stateFile, mockState, 0644); err != nil {
		t.Fatalf("Failed to create mock VM state: %v", err)
	}

	// Create source and destination VM managers
	sourceManager := NewVMMigrationManager("source-node", sourceDir)
	destManager := NewVMMigrationManager("dest-node", destDir)

	// Execute live migration
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	migration := &VMMigration{
		ID:                "migration-" + vmID,
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              MigrationTypeLive,
		Status:            MigrationStatusPending,
		VMSpec:            vmSpec,
		CreatedAt:         time.Now(),
		Options: map[string]string{
			"iterations": "3",
		},
	}

	// Execute migration
	err = sourceManager.ExecuteMigration(ctx, migration, destManager)
	if err != nil {
		t.Fatalf("Live migration failed: %v", err)
	}

	// Verify migration completed successfully
	if migration.Status != MigrationStatusCompleted {
		t.Errorf("Migration status is %s, expected %s", migration.Status, MigrationStatusCompleted)
	}

	// Verify final VM state was transferred
	destStateFile := filepath.Join(destDir, vmID+".state")
	_, err = os.Stat(destStateFile)
	if err != nil {
		t.Fatalf("Destination state file not found: %v", err)
	}

	// Verify final memory state was transferred
	destMemFile := filepath.Join(destDir, vmID+".memory")
	_, err = os.Stat(destMemFile)
	if err != nil {
		t.Fatalf("Destination memory file not found: %v", err)
	}
}

// Test migration failures and rollback
func TestMigrationRollback(t *testing.T) {
	// Test setup: create temporary test directories
	sourceDir, err := os.MkdirTemp("", "novacron-test-source")
	if err != nil {
		t.Fatalf("Failed to create temporary source directory: %v", err)
	}
	defer os.RemoveAll(sourceDir)

	destDir, err := os.MkdirTemp("", "novacron-test-dest")
	if err != nil {
		t.Fatalf("Failed to create temporary destination directory: %v", err)
	}
	defer os.RemoveAll(destDir)

	// Create mock VM info
	vmID := "test-vm-rollback"
	vmSpec := VMSpec{
		ID:       vmID,
		Name:     "TestVM",
		VCPU:     2,
		MemoryMB: 1024,
		DiskMB:   5120,
		Image:    "test-image",
	}

	// Create mock VM data file
	dataFile := filepath.Join(sourceDir, vmID+".state")
	mockData := []byte("mock VM state data for testing")
	if err := os.WriteFile(dataFile, mockData, 0644); err != nil {
		t.Fatalf("Failed to create mock VM data: %v", err)
	}

	// Create source migration manager but provide an invalid destination
	// to force a migration failure
	sourceManager := NewVMMigrationManager("source-node", sourceDir)

	// Use an invalid destination manager that will cause the migration to fail
	invalidDestManager := NewVMMigrationManager("dest-node", "/invalid/path/that/doesnt/exist")

	// Execute migration that should fail
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	migration := &VMMigration{
		ID:                "migration-" + vmID,
		VMID:              vmID,
		SourceNodeID:      "source-node",
		DestinationNodeID: "dest-node",
		Type:              MigrationTypeCold,
		Status:            MigrationStatusPending,
		VMSpec:            vmSpec,
		CreatedAt:         time.Now(),
	}

	// Execute migration which should fail and roll back
	err = sourceManager.ExecuteMigration(ctx, migration, invalidDestManager)
	if err == nil {
		t.Errorf("Migration should have failed but didn't")
	}

	// Verify migration status is failed
	if migration.Status != MigrationStatusFailed {
		t.Errorf("Migration status is %s, expected %s", migration.Status, MigrationStatusFailed)
	}

	// Verify VM is still running on source after rollback
	sourceVM, err := sourceManager.GetVM(vmID)
	if err != nil {
		t.Fatalf("Failed to get source VM after rollback: %v", err)
	}

	if sourceVM.Status != VMStatusRunning {
		t.Errorf("Source VM status is %s, expected %s after rollback", sourceVM.Status, VMStatusRunning)
	}
}
