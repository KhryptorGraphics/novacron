package federation

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/shared"
)

// BackupCoordinator coordinates backup operations across the federation
type BackupCoordinator struct {
	manager         FederationManager
	backupManager   shared.BackupManagerInterface
	replicationMgr  shared.ReplicationSystemInterface
	sharedMemory    *SharedMemory
	coordinationMu  sync.RWMutex
	activeBackups   map[string]*BackupTask
	logger          shared.Logger
}

// BackupTask represents a coordinated backup task
type BackupTask struct {
	ID              string                 `json:"id"`
	Type            BackupType             `json:"type"`
	SourceNodeID    string                 `json:"source_node_id"`
	TargetNodeIDs   []string               `json:"target_node_ids"`
	ResourceIDs     []string               `json:"resource_ids"`
	Status          BackupStatus           `json:"status"`
	Progress        float64                `json:"progress"`
	StartTime       time.Time              `json:"start_time"`
	CompletionTime  *time.Time             `json:"completion_time,omitempty"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// BackupType represents the type of backup operation
type BackupType string

const (
	BackupTypeFull        BackupType = "full"
	BackupTypeIncremental BackupType = "incremental"
	BackupTypeDifferential BackupType = "differential"
	BackupTypeSnapshot    BackupType = "snapshot"
	BackupTypeReplica     BackupType = "replica"
)

// BackupStatus represents the status of a backup task
type BackupStatus string

const (
	BackupStatusPending    BackupStatus = "pending"
	BackupStatusRunning    BackupStatus = "running"
	BackupStatusCompleted  BackupStatus = "completed"
	BackupStatusFailed     BackupStatus = "failed"
	BackupStatusCancelled  BackupStatus = "cancelled"
)

// SharedMemory provides thread-safe shared memory for coordination
type SharedMemory struct {
	data      map[string]interface{}
	mu        sync.RWMutex
	callbacks map[string][]func(key string, value interface{})
	callbackMu sync.RWMutex
}

// NewBackupCoordinator creates a new backup coordinator
func NewBackupCoordinator(
	manager FederationManager,
	backupManager shared.BackupManagerInterface,
	replicationMgr shared.ReplicationSystemInterface,
	logger shared.Logger,
) *BackupCoordinator {
	return &BackupCoordinator{
		manager:        manager,
		backupManager:  backupManager,
		replicationMgr: replicationMgr,
		sharedMemory:   NewSharedMemory(),
		activeBackups:  make(map[string]*BackupTask),
		logger:         logger,
	}
}

// NewSharedMemory creates a new shared memory instance
func NewSharedMemory() *SharedMemory {
	return &SharedMemory{
		data:      make(map[string]interface{}),
		callbacks: make(map[string][]func(key string, value interface{})),
	}
}

// CoordinateBackup coordinates a backup operation across the federation
func (bc *BackupCoordinator) CoordinateBackup(ctx context.Context, request *BackupRequest) (*BackupTask, error) {
	bc.coordinationMu.Lock()
	defer bc.coordinationMu.Unlock()

	// Create backup task
	task := &BackupTask{
		ID:            generateRequestID(),
		Type:          request.Type,
		SourceNodeID:  request.SourceNodeID,
		TargetNodeIDs: request.TargetNodeIDs,
		ResourceIDs:   request.ResourceIDs,
		Status:        BackupStatusPending,
		Progress:      0,
		StartTime:     time.Now(),
		Metadata:      make(map[string]interface{}),
	}

	// Store in active backups
	bc.activeBackups[task.ID] = task

	// Store in shared memory for coordination
	bc.sharedMemory.Set(fmt.Sprintf("backup:task:%s", task.ID), task)

	// If this node is the leader, coordinate the backup
	if bc.manager.IsLeader() {
		go bc.executeBackup(ctx, task)
	} else {
		// Forward to leader
		leader, err := bc.manager.GetLeader(ctx)
		if err != nil {
			task.Status = BackupStatusFailed
			return task, fmt.Errorf("failed to get leader: %w", err)
		}

		// Store request in shared memory for leader to pick up
		bc.sharedMemory.Set(fmt.Sprintf("backup:request:%s", task.ID), request)
		bc.sharedMemory.Notify(fmt.Sprintf("backup:leader:%s", leader.ID), task)
	}

	bc.logger.Info("Backup task coordinated", "task_id", task.ID, "type", task.Type)

	return task, nil
}

// executeBackup executes a backup task
func (bc *BackupCoordinator) executeBackup(ctx context.Context, task *BackupTask) {
	bc.logger.Info("Executing backup task", "task_id", task.ID)

	// Update status
	task.Status = BackupStatusRunning
	bc.updateTaskStatus(task)

	// Allocate resources for backup
	resourceRequest := &ResourceRequest{
		ID:           fmt.Sprintf("backup-%s", task.ID),
		ResourceType: "backup",
		CPUCores:     2,
		MemoryGB:     4,
		StorageGB:    100, // Temporary storage for backup
		Duration:     2 * time.Hour,
		Priority:     8, // High priority for backups
	}

	allocation, err := bc.manager.RequestResources(ctx, resourceRequest)
	if err != nil {
		bc.logger.Error("Failed to allocate resources for backup", "error", err)
		task.Status = BackupStatusFailed
		bc.updateTaskStatus(task)
		return
	}
	defer bc.manager.ReleaseResources(ctx, allocation.ID)

	// Store allocation info in shared memory
	bc.sharedMemory.Set(fmt.Sprintf("backup:allocation:%s", task.ID), allocation)

	// Perform backup based on type
	switch task.Type {
	case BackupTypeFull:
		err = bc.performFullBackup(ctx, task)
	case BackupTypeIncremental:
		err = bc.performIncrementalBackup(ctx, task)
	case BackupTypeSnapshot:
		err = bc.performSnapshotBackup(ctx, task)
	case BackupTypeReplica:
		err = bc.performReplicaBackup(ctx, task)
	default:
		err = fmt.Errorf("unsupported backup type: %s", task.Type)
	}

	if err != nil {
		bc.logger.Error("Backup failed", "task_id", task.ID, "error", err)
		task.Status = BackupStatusFailed
	} else {
		bc.logger.Info("Backup completed successfully", "task_id", task.ID)
		task.Status = BackupStatusCompleted
		now := time.Now()
		task.CompletionTime = &now
		task.Progress = 100
	}

	bc.updateTaskStatus(task)

	// Clean up shared memory
	bc.sharedMemory.Delete(fmt.Sprintf("backup:allocation:%s", task.ID))
}

// performFullBackup performs a full backup
func (bc *BackupCoordinator) performFullBackup(ctx context.Context, task *BackupTask) error {
	bc.logger.Info("Performing full backup", "task_id", task.ID)

	// Update progress
	task.Progress = 10
	bc.updateTaskStatus(task)

	// Create backup request for backup manager
	backupReq := &shared.BackupRequest{
		VMID:            task.ResourceIDs[0], // Assuming single VM for now
		Type:            shared.BackupTypeFull,
		Compression:     true,
		Encryption:      true,
		RetentionDays:   30,
		Priority:        shared.PriorityHigh,
	}

	// Execute backup
	backupResult, err := bc.backupManager.CreateBackup(ctx, backupReq)
	if err != nil {
		return fmt.Errorf("failed to create backup: %w", err)
	}

	// Store backup info in shared memory
	bc.sharedMemory.Set(fmt.Sprintf("backup:result:%s", task.ID), backupResult)

	task.Progress = 50
	bc.updateTaskStatus(task)

	// Replicate to target nodes if specified
	if len(task.TargetNodeIDs) > 0 {
		err = bc.replicateBackup(ctx, task, backupResult)
		if err != nil {
			return fmt.Errorf("failed to replicate backup: %w", err)
		}
	}

	task.Progress = 90
	bc.updateTaskStatus(task)

	// Verify backup integrity
	verifyReq := &shared.VerificationRequest{
		BackupID: backupResult.BackupID,
		Type:     shared.VerificationTypeChecksum,
	}

	verifyResult, err := bc.backupManager.VerifyBackup(ctx, verifyReq)
	if err != nil {
		return fmt.Errorf("backup verification failed: %w", err)
	}

	if !verifyResult.Valid {
		return fmt.Errorf("backup integrity check failed")
	}

	// Store verification result
	bc.sharedMemory.Set(fmt.Sprintf("backup:verification:%s", task.ID), verifyResult)

	return nil
}

// performIncrementalBackup performs an incremental backup
func (bc *BackupCoordinator) performIncrementalBackup(ctx context.Context, task *BackupTask) error {
	bc.logger.Info("Performing incremental backup", "task_id", task.ID)

	// Get last full backup from shared memory
	lastFullKey := fmt.Sprintf("backup:last_full:%s", task.ResourceIDs[0])
	lastFull, exists := bc.sharedMemory.Get(lastFullKey)
	if !exists {
		// No full backup exists, perform full backup instead
		bc.logger.Info("No full backup found, performing full backup", "resource_id", task.ResourceIDs[0])
		task.Type = BackupTypeFull
		return bc.performFullBackup(ctx, task)
	}

	task.Progress = 10
	bc.updateTaskStatus(task)

	// Create incremental backup request
	backupReq := &shared.BackupRequest{
		VMID:            task.ResourceIDs[0],
		Type:            shared.BackupTypeIncremental,
		BaseBackupID:    lastFull.(string),
		Compression:     true,
		Encryption:      true,
		RetentionDays:   7,
		Priority:        shared.PriorityMedium,
	}

	// Execute incremental backup
	backupResult, err := bc.backupManager.CreateBackup(ctx, backupReq)
	if err != nil {
		return fmt.Errorf("failed to create incremental backup: %w", err)
	}

	// Store in shared memory
	bc.sharedMemory.Set(fmt.Sprintf("backup:incremental:%s", task.ID), backupResult)

	task.Progress = 70
	bc.updateTaskStatus(task)

	// Replicate if needed
	if len(task.TargetNodeIDs) > 0 {
		err = bc.replicateBackup(ctx, task, backupResult)
		if err != nil {
			return fmt.Errorf("failed to replicate incremental backup: %w", err)
		}
	}

	return nil
}

// performSnapshotBackup performs a snapshot backup
func (bc *BackupCoordinator) performSnapshotBackup(ctx context.Context, task *BackupTask) error {
	bc.logger.Info("Performing snapshot backup", "task_id", task.ID)

	task.Progress = 10
	bc.updateTaskStatus(task)

	// Create snapshot request
	snapshotReq := &shared.SnapshotRequest{
		VMID:        task.ResourceIDs[0],
		Name:        fmt.Sprintf("snapshot-%s", task.ID),
		Description: "Federation coordinated snapshot",
		Memory:      true,
	}

	// Execute snapshot
	snapshot, err := bc.backupManager.CreateSnapshot(ctx, snapshotReq)
	if err != nil {
		return fmt.Errorf("failed to create snapshot: %w", err)
	}

	// Store snapshot info
	bc.sharedMemory.Set(fmt.Sprintf("backup:snapshot:%s", task.ID), snapshot)

	task.Progress = 80
	bc.updateTaskStatus(task)

	return nil
}

// performReplicaBackup performs a replica backup
func (bc *BackupCoordinator) performReplicaBackup(ctx context.Context, task *BackupTask) error {
	bc.logger.Info("Performing replica backup", "task_id", task.ID)

	if len(task.TargetNodeIDs) == 0 {
		return fmt.Errorf("no target nodes specified for replica backup")
	}

	task.Progress = 10
	bc.updateTaskStatus(task)

	// Setup replication
	replicationConfig := &shared.ReplicationConfig{
		SourceNode:    task.SourceNodeID,
		TargetNodes:   task.TargetNodeIDs,
		Mode:          shared.ReplicationModeAsync,
		Interval:      5 * time.Minute,
		RetryAttempts: 3,
	}

	// Start replication
	replicationID, err := bc.replicationMgr.StartReplication(ctx, task.ResourceIDs[0], replicationConfig)
	if err != nil {
		return fmt.Errorf("failed to start replication: %w", err)
	}

	// Store replication info
	bc.sharedMemory.Set(fmt.Sprintf("backup:replication:%s", task.ID), replicationID)

	// Monitor replication progress
	for {
		status, err := bc.replicationMgr.GetReplicationStatus(ctx, replicationID)
		if err != nil {
			return fmt.Errorf("failed to get replication status: %w", err)
		}

		task.Progress = status.Progress
		bc.updateTaskStatus(task)

		if status.Status == shared.ReplicationStatusCompleted {
			break
		} else if status.Status == shared.ReplicationStatusFailed {
			return fmt.Errorf("replication failed: %s", status.Error)
		}

		time.Sleep(5 * time.Second)
	}

	return nil
}

// replicateBackup replicates a backup to target nodes
func (bc *BackupCoordinator) replicateBackup(ctx context.Context, task *BackupTask, backupResult *shared.BackupResult) error {
	bc.logger.Info("Replicating backup to target nodes", "task_id", task.ID, "targets", task.TargetNodeIDs)

	// Create replication requests for each target
	var wg sync.WaitGroup
	errCh := make(chan error, len(task.TargetNodeIDs))

	for _, targetNodeID := range task.TargetNodeIDs {
		wg.Add(1)
		go func(nodeID string) {
			defer wg.Done()

			// Get target node
			node, err := bc.manager.GetNode(ctx, nodeID)
			if err != nil {
				errCh <- fmt.Errorf("failed to get node %s: %w", nodeID, err)
				return
			}

			// Create replication task
			replicationTask := &shared.ReplicationTask{
				BackupID:     backupResult.BackupID,
				SourcePath:   backupResult.Path,
				TargetNode:   node.Address,
				TargetPath:   fmt.Sprintf("/backups/%s/%s", task.ID, backupResult.BackupID),
				Compression:  true,
				Encryption:   true,
			}

			// Execute replication
			err = bc.replicationMgr.ReplicateBackup(ctx, replicationTask)
			if err != nil {
				errCh <- fmt.Errorf("failed to replicate to %s: %w", nodeID, err)
				return
			}

			// Store replication status in shared memory
			bc.sharedMemory.Set(
				fmt.Sprintf("backup:replica:%s:%s", task.ID, nodeID),
				map[string]interface{}{
					"status":     "completed",
					"timestamp":  time.Now(),
					"target":     node.Address,
					"backup_id":  backupResult.BackupID,
				},
			)
		}(targetNodeID)
	}

	wg.Wait()
	close(errCh)

	// Check for errors
	for err := range errCh {
		if err != nil {
			return err
		}
	}

	return nil
}

// GetBackupTask returns the status of a backup task
func (bc *BackupCoordinator) GetBackupTask(taskID string) (*BackupTask, error) {
	bc.coordinationMu.RLock()
	defer bc.coordinationMu.RUnlock()

	task, exists := bc.activeBackups[taskID]
	if !exists {
		// Check shared memory
		memKey := fmt.Sprintf("backup:task:%s", taskID)
		if val, ok := bc.sharedMemory.Get(memKey); ok {
			if t, ok := val.(*BackupTask); ok {
				return t, nil
			}
		}
		return nil, fmt.Errorf("backup task not found: %s", taskID)
	}

	return task, nil
}

// ListBackupTasks returns all active backup tasks
func (bc *BackupCoordinator) ListBackupTasks() []*BackupTask {
	bc.coordinationMu.RLock()
	defer bc.coordinationMu.RUnlock()

	tasks := make([]*BackupTask, 0, len(bc.activeBackups))
	for _, task := range bc.activeBackups {
		tasks = append(tasks, task)
	}

	return tasks
}

// CancelBackupTask cancels a backup task
func (bc *BackupCoordinator) CancelBackupTask(ctx context.Context, taskID string) error {
	bc.coordinationMu.Lock()
	defer bc.coordinationMu.Unlock()

	task, exists := bc.activeBackups[taskID]
	if !exists {
		return fmt.Errorf("backup task not found: %s", taskID)
	}

	if task.Status != BackupStatusPending && task.Status != BackupStatusRunning {
		return fmt.Errorf("cannot cancel task in status: %s", task.Status)
	}

	task.Status = BackupStatusCancelled
	bc.updateTaskStatus(task)

	// Notify cancellation through shared memory
	bc.sharedMemory.Set(fmt.Sprintf("backup:cancel:%s", taskID), true)

	bc.logger.Info("Backup task cancelled", "task_id", taskID)

	return nil
}

// updateTaskStatus updates the status of a backup task
func (bc *BackupCoordinator) updateTaskStatus(task *BackupTask) {
	bc.coordinationMu.Lock()
	bc.activeBackups[task.ID] = task
	bc.coordinationMu.Unlock()

	// Update in shared memory
	bc.sharedMemory.Set(fmt.Sprintf("backup:task:%s", task.ID), task)

	// Notify status change
	bc.sharedMemory.Notify(fmt.Sprintf("backup:status:%s", task.ID), task.Status)
}

// SharedMemory methods

// Set stores a value in shared memory
func (sm *SharedMemory) Set(key string, value interface{}) {
	sm.mu.Lock()
	sm.data[key] = value
	sm.mu.Unlock()

	// Trigger callbacks
	sm.triggerCallbacks(key, value)
}

// Get retrieves a value from shared memory
func (sm *SharedMemory) Get(key string) (interface{}, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	value, exists := sm.data[key]
	return value, exists
}

// Delete removes a value from shared memory
func (sm *SharedMemory) Delete(key string) {
	sm.mu.Lock()
	delete(sm.data, key)
	sm.mu.Unlock()
}

// Subscribe registers a callback for changes to a key pattern
func (sm *SharedMemory) Subscribe(pattern string, callback func(key string, value interface{})) {
	sm.callbackMu.Lock()
	defer sm.callbackMu.Unlock()

	if sm.callbacks[pattern] == nil {
		sm.callbacks[pattern] = make([]func(key string, value interface{}), 0)
	}
	sm.callbacks[pattern] = append(sm.callbacks[pattern], callback)
}

// Notify sends a notification through shared memory
func (sm *SharedMemory) Notify(key string, value interface{}) {
	sm.Set(key, value)
}

// triggerCallbacks triggers callbacks for a key
func (sm *SharedMemory) triggerCallbacks(key string, value interface{}) {
	sm.callbackMu.RLock()
	defer sm.callbackMu.RUnlock()

	for pattern, callbacks := range sm.callbacks {
		// Simple pattern matching (in production, use proper pattern matching)
		if pattern == key || pattern == "*" {
			for _, callback := range callbacks {
				go callback(key, value)
			}
		}
	}
}

// BackupRequest represents a backup request
type BackupRequest struct {
	Type          BackupType `json:"type"`
	SourceNodeID  string     `json:"source_node_id"`
	TargetNodeIDs []string   `json:"target_node_ids"`
	ResourceIDs   []string   `json:"resource_ids"`
	Priority      int        `json:"priority"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// SyncBackupState synchronizes backup state across the federation
func (bc *BackupCoordinator) SyncBackupState(ctx context.Context) error {
	// Get all nodes
	nodes, err := bc.manager.GetNodes(ctx)
	if err != nil {
		return fmt.Errorf("failed to get nodes: %w", err)
	}

	// Collect backup state from all nodes
	backupStates := make(map[string]interface{})
	
	for _, node := range nodes {
		// Get backup state from node's shared memory
		stateKey := fmt.Sprintf("backup:state:%s", node.ID)
		if state, exists := bc.sharedMemory.Get(stateKey); exists {
			backupStates[node.ID] = state
		}
	}

	// Store aggregated state
	bc.sharedMemory.Set("backup:federation:state", backupStates)

	// Broadcast state update
	bc.sharedMemory.Notify("backup:state:sync", backupStates)

	bc.logger.Info("Backup state synchronized", "nodes", len(nodes))

	return nil
}

// MonitorBackupHealth monitors the health of backup operations
func (bc *BackupCoordinator) MonitorBackupHealth(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			bc.checkBackupHealth()
		}
	}
}

func (bc *BackupCoordinator) checkBackupHealth() {
	bc.coordinationMu.RLock()
	tasks := make([]*BackupTask, 0, len(bc.activeBackups))
	for _, task := range bc.activeBackups {
		tasks = append(tasks, task)
	}
	bc.coordinationMu.RUnlock()

	now := time.Now()
	for _, task := range tasks {
		// Check for stalled tasks
		if task.Status == BackupStatusRunning {
			if now.Sub(task.StartTime) > 2*time.Hour {
				bc.logger.Warn("Backup task appears stalled", "task_id", task.ID, "duration", now.Sub(task.StartTime))
				
				// Store warning in shared memory
				bc.sharedMemory.Set(
					fmt.Sprintf("backup:warning:%s", task.ID),
					map[string]interface{}{
						"type":      "stalled",
						"duration":  now.Sub(task.StartTime).String(),
						"timestamp": now,
					},
				)
			}
		}
	}
}