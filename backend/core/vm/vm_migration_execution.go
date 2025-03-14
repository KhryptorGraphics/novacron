package vm

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// MigrationExecutorImpl implements the MigrationExecutor interface
type MigrationExecutorImpl struct {
	logger     *logrus.Logger
	storageDir string
	mu         sync.Mutex
}

// NewMigrationExecutor creates a new MigrationExecutor
func NewMigrationExecutor(logger *logrus.Logger, storageDir string) (*MigrationExecutorImpl, error) {
	// Ensure the storage directory exists
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create migration storage directory: %w", err)
	}

	return &MigrationExecutorImpl{
		logger:     logger,
		storageDir: storageDir,
	}, nil
}

// ExecuteColdMigration executes a cold migration (stop VM, transfer data, start VM on target)
func (e *MigrationExecutorImpl) ExecuteColdMigration(migrationID string, vm *VM, targetNode Node) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	logger := e.logger.WithFields(logrus.Fields{
		"migration_id":   migrationID,
		"vm_id":          vm.ID,
		"vm_name":        vm.Name,
		"source_node":    vm.NodeID,
		"target_node":    targetNode.GetID(),
		"migration_type": MigrationTypeCold,
	})

	logger.Info("Starting cold migration")

	// Step 1: Stop the VM on the source node
	logger.Info("Stopping VM on source node")
	if vm.State == VMStateRunning {
		if err := vm.Stop(); err != nil {
			return fmt.Errorf("failed to stop VM on source node: %w", err)
		}
	}

	// Step 2: Transfer VM data to target node
	logger.Info("Transferring VM data to target node")
	if err := e.transferVMData(migrationID, vm, targetNode); err != nil {
		return fmt.Errorf("failed to transfer VM data: %w", err)
	}

	// Step 3: Create VM on target node
	logger.Info("Creating VM on target node")
	if err := targetNode.CreateVM(vm.ID, vm.GetConfig()); err != nil {
		return fmt.Errorf("failed to create VM on target node: %w", err)
	}

	// Step 4: Start VM on target node if it was running before
	if vm.State == VMStateRunning {
		logger.Info("Starting VM on target node")
		targetVM, err := targetNode.GetVM(vm.ID)
		if err != nil {
			return fmt.Errorf("failed to get VM on target node: %w", err)
		}

		if err := targetVM.Start(); err != nil {
			return fmt.Errorf("failed to start VM on target node: %w", err)
		}
	}

	// Step 5: Verify VM is running on target node if needed
	if vm.State == VMStateRunning {
		logger.Info("Verifying VM is running on target node")
		targetVM, err := targetNode.GetVM(vm.ID)
		if err != nil {
			return fmt.Errorf("failed to get VM on target node: %w", err)
		}

		if targetVM.State != VMStateRunning {
			return fmt.Errorf("VM is not running on target node after migration")
		}
	}

	// Step 6: Delete VM on source node
	logger.Info("Deleting VM on source node")
	if err := vm.Delete(); err != nil {
		logger.WithError(err).Warn("Failed to delete VM on source node after migration")
		// Don't return an error here, as the migration has succeeded
	}

	logger.Info("Cold migration completed successfully")
	return nil
}

// ExecuteWarmMigration executes a warm migration (suspend VM, transfer data+memory, resume VM on target)
func (e *MigrationExecutorImpl) ExecuteWarmMigration(migrationID string, vm *VM, targetNode Node) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	logger := e.logger.WithFields(logrus.Fields{
		"migration_id":   migrationID,
		"vm_id":          vm.ID,
		"vm_name":        vm.Name,
		"source_node":    vm.NodeID,
		"target_node":    targetNode.GetID(),
		"migration_type": MigrationTypeWarm,
	})

	logger.Info("Starting warm migration")

	// Step 1: Check if VM is running
	if vm.State != VMStateRunning {
		return ErrVMNotRunning
	}

	// Step 2: Suspend the VM on the source node
	logger.Info("Suspending VM on source node")
	if err := vm.Suspend(); err != nil {
		return fmt.Errorf("failed to suspend VM on source node: %w", err)
	}

	// Step 3: Transfer VM data to target node
	logger.Info("Transferring VM data to target node")
	if err := e.transferVMData(migrationID, vm, targetNode); err != nil {
		// Try to resume the VM on the source node if transfer fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after transfer failure")
		}
		return fmt.Errorf("failed to transfer VM data: %w", err)
	}

	// Step 4: Transfer VM memory state
	logger.Info("Transferring VM memory state to target node")
	if err := e.transferVMMemoryState(migrationID, vm, targetNode); err != nil {
		// Try to resume the VM on the source node if transfer fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after memory state transfer failure")
		}
		return fmt.Errorf("failed to transfer VM memory state: %w", err)
	}

	// Step 5: Create VM on target node
	logger.Info("Creating VM on target node")
	if err := targetNode.CreateVM(vm.ID, vm.GetConfig()); err != nil {
		// Try to resume the VM on the source node if creation fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after target creation failure")
		}
		return fmt.Errorf("failed to create VM on target node: %w", err)
	}

	// Step 6: Resume VM on target node
	logger.Info("Resuming VM on target node")
	targetVM, err := targetNode.GetVM(vm.ID)
	if err != nil {
		// Try to resume the VM on the source node if getting the target VM fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after getting target VM failed")
		}
		return fmt.Errorf("failed to get VM on target node: %w", err)
	}

	if err := targetVM.ResumeFromState(); err != nil {
		// Try to resume the VM on the source node if resuming on the target fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after target resume failure")
		}
		return fmt.Errorf("failed to resume VM on target node: %w", err)
	}

	// Step 7: Verify VM is running on target node
	logger.Info("Verifying VM is running on target node")
	targetVM, err = targetNode.GetVM(vm.ID)
	if err != nil {
		return fmt.Errorf("failed to get VM on target node: %w", err)
	}

	if targetVM.State != VMStateRunning {
		// Try to resume the VM on the source node if verification fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after verification failure")
		}
		return fmt.Errorf("VM is not running on target node after migration")
	}

	// Step 8: Delete VM on source node
	logger.Info("Deleting VM on source node")
	if err := vm.Delete(); err != nil {
		logger.WithError(err).Warn("Failed to delete VM on source node after migration")
		// Don't return an error here, as the migration has succeeded
	}

	logger.Info("Warm migration completed successfully")
	return nil
}

// ExecuteLiveMigration executes a live migration (transfer memory state iteratively while VM runs)
func (e *MigrationExecutorImpl) ExecuteLiveMigration(migrationID string, vm *VM, targetNode Node) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	logger := e.logger.WithFields(logrus.Fields{
		"migration_id":   migrationID,
		"vm_id":          vm.ID,
		"vm_name":        vm.Name,
		"source_node":    vm.NodeID,
		"target_node":    targetNode.GetID(),
		"migration_type": MigrationTypeLive,
	})

	logger.Info("Starting live migration")

	// Step 1: Check if VM is running
	if vm.State != VMStateRunning {
		return ErrVMNotRunning
	}

	// Step 2: Transfer VM data to target node
	logger.Info("Transferring VM data to target node")
	if err := e.transferVMData(migrationID, vm, targetNode); err != nil {
		return fmt.Errorf("failed to transfer VM data: %w", err)
	}

	// Step 3: Create VM on target node (but don't start it yet)
	logger.Info("Creating VM on target node")
	if err := targetNode.CreateVM(vm.ID, vm.GetConfig()); err != nil {
		return fmt.Errorf("failed to create VM on target node: %w", err)
	}

	// Step 4: Pre-copy memory state iteratively
	logger.Info("Starting iterative memory state pre-copy")
	maxIterations := 5
	for i := 0; i < maxIterations; i++ {
		logger.Infof("Memory pre-copy iteration %d/%d", i+1, maxIterations)
		if err := e.transferVMMemoryDelta(migrationID, vm, targetNode, i); err != nil {
			return fmt.Errorf("failed to transfer VM memory delta (iteration %d): %w", i, err)
		}

		// Check if the memory dirty rate is low enough to proceed to final copy
		if isDirtyRateAcceptable(vm) {
			logger.Info("Memory dirty rate is acceptable, proceeding to final copy")
			break
		}
	}

	// Step 5: Suspend VM on source node briefly
	logger.Info("Suspending VM on source node for final memory copy")
	if err := vm.Suspend(); err != nil {
		return fmt.Errorf("failed to suspend VM on source node: %w", err)
	}

	// Step 6: Transfer final memory state
	logger.Info("Transferring final memory state")
	if err := e.transferVMMemoryState(migrationID, vm, targetNode); err != nil {
		// Try to resume the VM on the source node if transfer fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after final memory transfer failure")
		}
		return fmt.Errorf("failed to transfer final VM memory state: %w", err)
	}

	// Step 7: Resume VM on target node
	logger.Info("Resuming VM on target node")
	targetVM, err := targetNode.GetVM(vm.ID)
	if err != nil {
		// Try to resume the VM on the source node if getting the target VM fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after getting target VM failed")
		}
		return fmt.Errorf("failed to get VM on target node: %w", err)
	}

	if err := targetVM.ResumeFromState(); err != nil {
		// Try to resume the VM on the source node if resuming on the target fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after target resume failure")
		}
		return fmt.Errorf("failed to resume VM on target node: %w", err)
	}

	// Step 8: Verify VM is running on target node
	logger.Info("Verifying VM is running on target node")
	targetVM, err = targetNode.GetVM(vm.ID)
	if err != nil {
		return fmt.Errorf("failed to get VM on target node: %w", err)
	}

	if targetVM.State != VMStateRunning {
		// Try to resume the VM on the source node if verification fails
		if resumeErr := vm.Resume(); resumeErr != nil {
			logger.WithError(resumeErr).Error("Failed to resume VM on source node after verification failure")
		}
		return fmt.Errorf("VM is not running on target node after migration")
	}

	// Step 9: Delete VM on source node
	logger.Info("Deleting VM on source node")
	if err := vm.Delete(); err != nil {
		logger.WithError(err).Warn("Failed to delete VM on source node after migration")
		// Don't return an error here, as the migration has succeeded
	}

	logger.Info("Live migration completed successfully")
	return nil
}

// transferVMData transfers VM disk data from source to target node
func (e *MigrationExecutorImpl) transferVMData(migrationID string, vm *VM, targetNode Node) error {
	logger := e.logger.WithFields(logrus.Fields{
		"migration_id": migrationID,
		"vm_id":        vm.ID,
		"vm_name":      vm.Name,
		"source_node":  vm.NodeID,
		"target_node":  targetNode.GetID(),
	})

	logger.Info("Starting VM data transfer")

	// Create temporary directory for migration data
	tmpDir := filepath.Join(e.storageDir, fmt.Sprintf("migration_%s", migrationID))
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		return fmt.Errorf("failed to create temporary directory for migration: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Get VM disk paths
	diskPaths, err := vm.GetDiskPaths()
	if err != nil {
		return fmt.Errorf("failed to get VM disk paths: %w", err)
	}

	startTime := time.Now()
	var totalBytes int64

	// Transfer each disk
	for i, diskPath := range diskPaths {
		diskFileName := fmt.Sprintf("disk_%d.img", i)
		tmpDiskPath := filepath.Join(tmpDir, diskFileName)

		// Copy disk to temporary location
		logger.Infof("Copying disk %d to temporary location", i)
		copiedBytes, err := copyFile(diskPath, tmpDiskPath)
		if err != nil {
			return fmt.Errorf("failed to copy disk to temporary location: %w", err)
		}
		totalBytes += copiedBytes

		// Transfer disk to target node
		logger.Infof("Transferring disk %d to target node", i)
		targetDiskPath := targetNode.GetDiskPath(vm.ID, i)
		if err := targetNode.EnsureDirectoryExists(filepath.Dir(targetDiskPath)); err != nil {
			return fmt.Errorf("failed to create directory on target node: %w", err)
		}

		if err := targetNode.ReceiveFile(tmpDiskPath, targetDiskPath); err != nil {
			return fmt.Errorf("failed to transfer disk to target node: %w", err)
		}
	}

	// Calculate transfer stats
	transferTime := time.Since(startTime)
	transferRateMBps := float64(totalBytes) / 1024 / 1024 / transferTime.Seconds()

	logger.WithFields(logrus.Fields{
		"bytes_transferred": totalBytes,
		"transfer_time_ms":  transferTime.Milliseconds(),
		"transfer_rate_mbps": transferRateMBps,
	}).Info("VM data transfer completed")

	return nil
}

// transferVMMemoryState transfers VM memory state from source to target node
func (e *MigrationExecutorImpl) transferVMMemoryState(migrationID string, vm *VM, targetNode Node) error {
	logger := e.logger.WithFields(logrus.Fields{
		"migration_id": migrationID,
		"vm_id":        vm.ID,
		"vm_name":      vm.Name,
		"source_node":  vm.NodeID,
		"target_node":  targetNode.GetID(),
	})

	logger.Info("Starting VM memory state transfer")

	// Create temporary directory for migration data
	tmpDir := filepath.Join(e.storageDir, fmt.Sprintf("migration_%s", migrationID))
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		return fmt.Errorf("failed to create temporary directory for migration: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Get VM memory state path
	memoryStatePath, err := vm.GetMemoryStatePath()
	if err != nil {
		return fmt.Errorf("failed to get VM memory state path: %w", err)
	}

	tmpMemoryPath := filepath.Join(tmpDir, "memory.state")

	// Copy memory state to temporary location
	logger.Info("Copying memory state to temporary location")
	copiedBytes, err := copyFile(memoryStatePath, tmpMemoryPath)
	if err != nil {
		return fmt.Errorf("failed to copy memory state to temporary location: %w", err)
	}

	// Transfer memory state to target node
	logger.Info("Transferring memory state to target node")
	targetMemoryPath := targetNode.GetMemoryStatePath(vm.ID)
	if err := targetNode.EnsureDirectoryExists(filepath.Dir(targetMemoryPath)); err != nil {
		return fmt.Errorf("failed to create directory on target node: %w", err)
	}

	if err := targetNode.ReceiveFile(tmpMemoryPath, targetMemoryPath); err != nil {
		return fmt.Errorf("failed to transfer memory state to target node: %w", err)
	}

	logger.WithFields(logrus.Fields{
		"bytes_transferred": copiedBytes,
	}).Info("VM memory state transfer completed")

	return nil
}

// transferVMMemoryDelta transfers VM memory delta (dirty pages) for live migration
func (e *MigrationExecutorImpl) transferVMMemoryDelta(migrationID string, vm *VM, targetNode Node, iteration int) error {
	logger := e.logger.WithFields(logrus.Fields{
		"migration_id": migrationID,
		"vm_id":        vm.ID,
		"vm_name":      vm.Name,
		"source_node":  vm.NodeID,
		"target_node":  targetNode.GetID(),
		"iteration":    iteration,
	})

	logger.Info("Starting VM memory delta transfer")

	// Create temporary directory for migration data
	tmpDir := filepath.Join(e.storageDir, fmt.Sprintf("migration_%s", migrationID))
	if err := os.MkdirAll(tmpDir, 0755); err != nil {
		return fmt.Errorf("failed to create temporary directory for migration: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	// Get VM memory delta path
	memoryDeltaPath, err := vm.GetMemoryDeltaPath(iteration)
	if err != nil {
		return fmt.Errorf("failed to get VM memory delta path: %w", err)
	}

	tmpMemoryDeltaPath := filepath.Join(tmpDir, fmt.Sprintf("memory_delta_%d.state", iteration))

	// Copy memory delta to temporary location
	logger.Info("Copying memory delta to temporary location")
	copiedBytes, err := copyFile(memoryDeltaPath, tmpMemoryDeltaPath)
	if err != nil {
		return fmt.Errorf("failed to copy memory delta to temporary location: %w", err)
	}

	// Transfer memory delta to target node
	logger.Info("Transferring memory delta to target node")
	targetMemoryDeltaPath := targetNode.GetMemoryDeltaPath(vm.ID, iteration)
	if err := targetNode.EnsureDirectoryExists(filepath.Dir(targetMemoryDeltaPath)); err != nil {
		return fmt.Errorf("failed to create directory on target node: %w", err)
	}

	if err := targetNode.ReceiveFile(tmpMemoryDeltaPath, targetMemoryDeltaPath); err != nil {
		return fmt.Errorf("failed to transfer memory delta to target node: %w", err)
	}

	logger.WithFields(logrus.Fields{
		"bytes_transferred": copiedBytes,
	}).Info("VM memory delta transfer completed")

	return nil
}

// RollbackMigration rolls back a failed migration
func (e *MigrationExecutorImpl) RollbackMigration(migrationID string, vm *VM, sourceNode Node) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	logger := e.logger.WithFields(logrus.Fields{
		"migration_id": migrationID,
		"vm_id":        vm.ID,
		"vm_name":      vm.Name,
		"source_node":  sourceNode.GetID(),
	})

	logger.Info("Rolling back migration")

	// Resume VM on source node if it was suspended or paused
	if vm.State == VMStateSuspended || vm.State == VMStatePaused {
		logger.Info("Resuming VM on source node")
		if err := vm.Resume(); err != nil {
			return fmt.Errorf("failed to resume VM on source node: %w", err)
		}
	} else if vm.State == VMStateStopped {
		// Start VM on source node if it was running before migration
		logger.Info("Starting VM on source node")
		if err := vm.Start(); err != nil {
			return fmt.Errorf("failed to start VM on source node: %w", err)
		}
	}

	logger.Info("Migration rollback completed successfully")
	return nil
}

// Helper functions

// copyFile copies a file from src to dst, returns number of bytes copied
func copyFile(src, dst string) (int64, error) {
	sourceFile, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return 0, err
	}
	defer destFile.Close()

	return io.Copy(destFile, sourceFile)
}

// isDirtyRateAcceptable checks if the memory dirty rate is low enough to proceed
func isDirtyRateAcceptable(vm *VM) bool {
	// This is a placeholder. In a real implementation, this would check
	// the memory dirty rate against a threshold.
	return true
}
