package vm

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// LifecycleManager manages VM lifecycle operations with advanced state management
type LifecycleManager struct {
	vms           map[string]*EnhancedVM
	stateMachine  *VMStateMachine
	eventBus      *LifecycleEventBus
	scheduler     *LifecycleScheduler
	checkpointer  *VMCheckpointer
	snapshotter   *VMSnapshotter
	migrationMgr  *LiveMigrationManager
	mu            sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
	metrics       *LifecycleMetrics
}

// EnhancedVM extends the basic VM with lifecycle management capabilities
type EnhancedVM struct {
	*VM
	lifecycleState  State
	stateHistory    []StateTransition
	checkpoints     []*Checkpoint
	snapshots       []*Snapshot
	migrationStatus *LiveMigrationStatus
	healthCheck     *HealthChecker
	resourceLimits  *ResourceLimits
	policies        *LifecyclePolicies
	mu              sync.RWMutex
}

// Enhanced lifecycle states that extend the existing State type
const (
	StateInitializing = State("initializing")
	StateStarting     = State("starting") 
	StateCheckpointing = State("checkpointing")
	StateRestoring    = State("restoring")
	StateStopping     = State("stopping")
	StateTerminating  = State("terminating")
	StateTerminated   = State("terminated")
	StateMaintenance  = State("maintenance")
)

// StateTransition records a state change
type StateTransition struct {
	From      State `json:"from"`
	To        State `json:"to"`
	Timestamp time.Time      `json:"timestamp"`
	Reason    string         `json:"reason"`
	Duration  time.Duration  `json:"duration"`
	Success   bool           `json:"success"`
	Error     string         `json:"error,omitempty"`
}

// Checkpoint represents a VM checkpoint
type Checkpoint struct {
	ID            string                 `json:"id"`
	Timestamp     time.Time              `json:"timestamp"`
	MemoryState   []byte                 `json:"memory_state"`
	DiskState     map[string][]byte      `json:"disk_state"`
	NetworkState  map[string]interface{} `json:"network_state"`
	CPUState      map[string]interface{} `json:"cpu_state"`
	Metadata      map[string]string      `json:"metadata"`
	Size          int64                  `json:"size"`
	Compressed    bool                   `json:"compressed"`
	Encrypted     bool                   `json:"encrypted"`
}

// Snapshot represents a VM snapshot
type Snapshot struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Timestamp   time.Time         `json:"timestamp"`
	ParentID    string            `json:"parent_id"`
	DiskImages  []DiskImage       `json:"disk_images"`
	VMConfig    VMConfig          `json:"vm_config"`
	Metadata    map[string]string `json:"metadata"`
	Size        int64             `json:"size"`
	State       SnapshotState     `json:"state"`
}

// DiskImage represents a disk image in a snapshot
type DiskImage struct {
	Path         string `json:"path"`
	Size         int64  `json:"size"`
	Format       string `json:"format"`
	Checksum     string `json:"checksum"`
	Compressed   bool   `json:"compressed"`
	Incremental  bool   `json:"incremental"`
}

// SnapshotState represents snapshot states
type SnapshotState int

const (
	SnapshotCreating SnapshotState = iota
	SnapshotReady
	SnapshotDeleting
	SnapshotCorrupted
)

// LiveMigrationStatus tracks live migration status
type LiveMigrationStatus struct {
	InProgress      bool              `json:"in_progress"`
	Type            MigrationType     `json:"type"`
	Source          string            `json:"source"`
	Destination     string            `json:"destination"`
	Progress        float64           `json:"progress"`
	Phase           MigrationPhase    `json:"phase"`
	StartTime       time.Time         `json:"start_time"`
	EstimatedFinish time.Time         `json:"estimated_finish"`
	BytesTransferred int64            `json:"bytes_transferred"`
	TotalBytes      int64             `json:"total_bytes"`
	Error           string            `json:"error,omitempty"`
}

// MigrationPhase defines migration phases
type MigrationPhase int

const (
	PhasePreMigration MigrationPhase = iota
	PhaseMemoryCopy
	PhaseFinalSync
	PhaseHandover
	PhasePostMigration
)

// ResourceLimits defines resource constraints
type ResourceLimits struct {
	MaxCPU     int   `json:"max_cpu"`
	MaxMemory  int64 `json:"max_memory"`
	MaxStorage int64 `json:"max_storage"`
	MaxNetwork int64 `json:"max_network"`
}

// LifecyclePolicies defines automated policies
type LifecyclePolicies struct {
	AutoStart        bool          `json:"auto_start"`
	AutoStop         bool          `json:"auto_stop"`
	HealthCheckEnabled bool        `json:"health_check_enabled"`
	BackupEnabled    bool          `json:"backup_enabled"`
	BackupInterval   time.Duration `json:"backup_interval"`
	RetentionPolicy  RetentionPolicy `json:"retention_policy"`
}

// RetentionPolicy defines data retention rules
type RetentionPolicy struct {
	MaxSnapshots   int           `json:"max_snapshots"`
	MaxCheckpoints int           `json:"max_checkpoints"`
	SnapshotTTL    time.Duration `json:"snapshot_ttl"`
	CheckpointTTL  time.Duration `json:"checkpoint_ttl"`
}

// LifecycleMetrics tracks lifecycle operations
type LifecycleMetrics struct {
	StateTransitions  map[string]int64      `json:"state_transitions"`
	OperationDurations map[string]time.Duration `json:"operation_durations"`
	SuccessRates      map[string]float64    `json:"success_rates"`
	FailureCount      int64                 `json:"failure_count"`
	TotalOperations   int64                 `json:"total_operations"`
	mu                sync.RWMutex
}

// NewLifecycleManager creates a new VM lifecycle manager
func NewLifecycleManager() *LifecycleManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	lm := &LifecycleManager{
		vms:          make(map[string]*EnhancedVM),
		stateMachine: NewVMStateMachine(),
		eventBus:     NewLifecycleEventBus(),
		scheduler:    NewLifecycleScheduler(),
		checkpointer: NewVMCheckpointer(),
		snapshotter:  NewVMSnapshotter(),
		migrationMgr: NewLiveMigrationManager(),
		ctx:          ctx,
		cancel:       cancel,
		metrics:      NewLifecycleMetrics(),
	}
	
	// Start background workers
	go lm.stateMonitor()
	go lm.healthChecker()
	go lm.cleanupWorker()
	
	return lm
}

// CreateVM creates a new enhanced VM
func (lm *LifecycleManager) CreateVM(config VMConfig) (*EnhancedVM, error) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	
	// Create base VM
	baseVM, err := NewVM(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create base VM: %w", err)
	}
	
	// Create enhanced VM
	evm := &EnhancedVM{
		VM:             baseVM,
		lifecycleState: StateProvisioning,
		stateHistory:   make([]StateTransition, 0),
		checkpoints:    make([]*Checkpoint, 0),
		snapshots:      make([]*Snapshot, 0),
		healthCheck:    NewHealthChecker(baseVM.ID()),
		resourceLimits: &ResourceLimits{
			MaxCPU:     config.CPUShares,
			MaxMemory:  int64(config.MemoryMB) * 1024 * 1024,
			MaxStorage: 100 * 1024 * 1024 * 1024, // 100GB default
			MaxNetwork: 1000 * 1024 * 1024,       // 1Gbps default
		},
		policies: &LifecyclePolicies{
			AutoStart:        false,
			AutoStop:         false,
			HealthCheckEnabled: true,
			BackupEnabled:    false,
			BackupInterval:   24 * time.Hour,
			RetentionPolicy: RetentionPolicy{
				MaxSnapshots:   10,
				MaxCheckpoints: 5,
				SnapshotTTL:    30 * 24 * time.Hour, // 30 days
				CheckpointTTL:  7 * 24 * time.Hour,  // 7 days
			},
		},
	}
	
	lm.vms[baseVM.ID()] = evm
	
	// Record initial state transition
	evm.recordStateTransition(StateProvisioning, StateInitializing, "VM created", nil)
	
	// Emit creation event
	lm.eventBus.Emit(&LifecycleEvent{
		Type:      EventVMCreated,
		VMID:      evm.ID(),
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"config": config},
	})
	
	log.Printf("Enhanced VM created: %s", evm.ID())
	return evm, nil
}

// StartVM starts a VM with lifecycle management
func (lm *LifecycleManager) StartVM(vmID string) error {
	lm.mu.RLock()
	evm, exists := lm.vms[vmID]
	lm.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("VM not found: %s", vmID)
	}
	
	evm.mu.Lock()
	defer evm.mu.Unlock()
	
	startTime := time.Now()
	
	// Check state transition validity
	if !lm.stateMachine.CanTransition(evm.lifecycleState, StateStarting) {
		return fmt.Errorf("cannot start VM in state: %s", evm.lifecycleState)
	}
	
	// Update state
	evm.recordStateTransition(evm.lifecycleState, StateStarting, "Starting VM", nil)
	
	// Start the base VM
	err := evm.VM.Start()
	if err != nil {
		evm.recordStateTransition(StateStarting, StateFailed, "Failed to start VM", err)
		return fmt.Errorf("failed to start VM: %w", err)
	}
	
	// Update state to running
	evm.recordStateTransition(StateStarting, StateRunning, "VM started successfully", nil)
	
	// Start health checking if enabled
	if evm.policies.HealthCheckEnabled {
		evm.healthCheck.Start()
	}
	
	// Record metrics
	duration := time.Since(startTime)
	lm.metrics.RecordOperation("start", duration, true)
	
	// Emit event
	lm.eventBus.Emit(&LifecycleEvent{
		Type:      EventVMStarted,
		VMID:      vmID,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"duration": duration},
	})
	
	log.Printf("VM started successfully: %s (took %v)", vmID, duration)
	return nil
}

// StopVM stops a VM gracefully
func (lm *LifecycleManager) StopVM(vmID string, graceful bool) error {
	lm.mu.RLock()
	evm, exists := lm.vms[vmID]
	lm.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("VM not found: %s", vmID)
	}
	
	evm.mu.Lock()
	defer evm.mu.Unlock()
	
	startTime := time.Now()
	
	// Check state transition validity
	if !lm.stateMachine.CanTransition(evm.lifecycleState, StateStopping) {
		return fmt.Errorf("cannot stop VM in state: %s", evm.lifecycleState)
	}
	
	// Update state
	evm.recordStateTransition(evm.lifecycleState, StateStopping, "Stopping VM", nil)
	
	// Stop health checking
	if evm.healthCheck != nil {
		evm.healthCheck.Stop()
	}
	
	// Stop the base VM
	var err error
	if graceful {
		err = evm.VM.Stop()
	} else {
		// Force kill by stopping (VM struct doesn't have ForceKill method)
		err = evm.VM.Stop()
	}
	
	if err != nil {
		evm.recordStateTransition(StateStopping, StateFailed, "Failed to stop VM", err)
		return fmt.Errorf("failed to stop VM: %w", err)
	}
	
	// Update state to stopped
	evm.recordStateTransition(StateStopping, StateStopped, "VM stopped successfully", nil)
	
	// Record metrics
	duration := time.Since(startTime)
	lm.metrics.RecordOperation("stop", duration, true)
	
	// Emit event
	lm.eventBus.Emit(&LifecycleEvent{
		Type:      EventVMStopped,
		VMID:      vmID,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"graceful": graceful, "duration": duration},
	})
	
	log.Printf("VM stopped successfully: %s (took %v)", vmID, duration)
	return nil
}

// CreateCheckpoint creates a checkpoint of a running VM
func (lm *LifecycleManager) CreateCheckpoint(vmID string, metadata map[string]string) (*Checkpoint, error) {
	lm.mu.RLock()
	evm, exists := lm.vms[vmID]
	lm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("VM not found: %s", vmID)
	}
	
	evm.mu.Lock()
	defer evm.mu.Unlock()
	
	if evm.lifecycleState != StateRunning && evm.lifecycleState != StatePaused {
		return nil, fmt.Errorf("cannot checkpoint VM in state: %s", evm.lifecycleState)
	}
	
	startTime := time.Now()
	previousState := evm.lifecycleState
	
	// Update state
	evm.recordStateTransition(previousState, StateCheckpointing, "Creating checkpoint", nil)
	
	// Create checkpoint
	checkpoint, err := lm.checkpointer.CreateCheckpoint(evm.VM, metadata)
	if err != nil {
		evm.recordStateTransition(StateCheckpointing, previousState, "Failed to create checkpoint", err)
		return nil, fmt.Errorf("failed to create checkpoint: %w", err)
	}
	
	// Add to VM's checkpoint list
	evm.checkpoints = append(evm.checkpoints, checkpoint)
	
	// Enforce retention policy
	lm.enforceCheckpointRetention(evm)
	
	// Restore previous state
	evm.recordStateTransition(StateCheckpointing, previousState, "Checkpoint created successfully", nil)
	
	// Record metrics
	duration := time.Since(startTime)
	lm.metrics.RecordOperation("checkpoint", duration, true)
	
	// Emit event
	lm.eventBus.Emit(&LifecycleEvent{
		Type:      EventCheckpointCreated,
		VMID:      vmID,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"checkpoint_id": checkpoint.ID, "duration": duration},
	})
	
	log.Printf("Checkpoint created for VM %s: %s (took %v)", vmID, checkpoint.ID, duration)
	return checkpoint, nil
}

// RestoreCheckpoint restores a VM from a checkpoint
func (lm *LifecycleManager) RestoreCheckpoint(vmID, checkpointID string) error {
	lm.mu.RLock()
	evm, exists := lm.vms[vmID]
	lm.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("VM not found: %s", vmID)
	}
	
	evm.mu.Lock()
	defer evm.mu.Unlock()
	
	// Find checkpoint
	var checkpoint *Checkpoint
	for _, cp := range evm.checkpoints {
		if cp.ID == checkpointID {
			checkpoint = cp
			break
		}
	}
	
	if checkpoint == nil {
		return fmt.Errorf("checkpoint not found: %s", checkpointID)
	}
	
	startTime := time.Now()
	previousState := evm.lifecycleState
	
	// Update state
	evm.recordStateTransition(previousState, StateRestoring, "Restoring from checkpoint", nil)
	
	// Restore checkpoint
	err := lm.checkpointer.RestoreCheckpoint(evm.VM, checkpoint)
	if err != nil {
		evm.recordStateTransition(StateRestoring, StateFailed, "Failed to restore checkpoint", err)
		return fmt.Errorf("failed to restore checkpoint: %w", err)
	}
	
	// Update state to running
	evm.recordStateTransition(StateRestoring, StateRunning, "Restored from checkpoint successfully", nil)
	
	// Record metrics
	duration := time.Since(startTime)
	lm.metrics.RecordOperation("restore", duration, true)
	
	// Emit event
	lm.eventBus.Emit(&LifecycleEvent{
		Type:      EventCheckpointRestored,
		VMID:      vmID,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"checkpoint_id": checkpointID, "duration": duration},
	})
	
	log.Printf("VM restored from checkpoint %s: %s (took %v)", vmID, checkpointID, duration)
	return nil
}

// CreateSnapshot creates a snapshot of a VM
func (lm *LifecycleManager) CreateSnapshot(vmID, name, description string) (*Snapshot, error) {
	lm.mu.RLock()
	evm, exists := lm.vms[vmID]
	lm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("VM not found: %s", vmID)
	}
	
	startTime := time.Now()
	
	// Create snapshot
	snapshot, err := lm.snapshotter.CreateSnapshot(evm.VM, name, description)
	if err != nil {
		return nil, fmt.Errorf("failed to create snapshot: %w", err)
	}
	
	evm.mu.Lock()
	evm.snapshots = append(evm.snapshots, snapshot)
	lm.enforceSnapshotRetention(evm)
	evm.mu.Unlock()
	
	// Record metrics
	duration := time.Since(startTime)
	lm.metrics.RecordOperation("snapshot", duration, true)
	
	// Emit event
	lm.eventBus.Emit(&LifecycleEvent{
		Type:      EventSnapshotCreated,
		VMID:      vmID,
		Timestamp: time.Now(),
		Data:      map[string]interface{}{"snapshot_id": snapshot.ID, "duration": duration},
	})
	
	log.Printf("Snapshot created for VM %s: %s (took %v)", vmID, snapshot.ID, duration)
	return snapshot, nil
}

// GetVMState returns the current lifecycle state of a VM
func (lm *LifecycleManager) GetVMState(vmID string) (State, error) {
	lm.mu.RLock()
	evm, exists := lm.vms[vmID]
	lm.mu.RUnlock()
	
	if !exists {
		return StateFailed, fmt.Errorf("VM not found: %s", vmID)
	}
	
	evm.mu.RLock()
	state := evm.lifecycleState
	evm.mu.RUnlock()
	
	return state, nil
}

// GetVMHistory returns the state transition history of a VM
func (lm *LifecycleManager) GetVMHistory(vmID string) ([]StateTransition, error) {
	lm.mu.RLock()
	evm, exists := lm.vms[vmID]
	lm.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("VM not found: %s", vmID)
	}
	
	evm.mu.RLock()
	history := make([]StateTransition, len(evm.stateHistory))
	copy(history, evm.stateHistory)
	evm.mu.RUnlock()
	
	return history, nil
}

// Helper methods

func (evm *EnhancedVM) recordStateTransition(from, to State, reason string, err error) {
	transition := StateTransition{
		From:      from,
		To:        to,
		Timestamp: time.Now(),
		Reason:    reason,
		Success:   err == nil,
	}
	
	if err != nil {
		transition.Error = err.Error()
	}
	
	// Calculate duration from last transition
	if len(evm.stateHistory) > 0 {
		lastTransition := evm.stateHistory[len(evm.stateHistory)-1]
		transition.Duration = transition.Timestamp.Sub(lastTransition.Timestamp)
	}
	
	evm.stateHistory = append(evm.stateHistory, transition)
	evm.lifecycleState = to
}

func (lm *LifecycleManager) enforceCheckpointRetention(evm *EnhancedVM) {
	maxCheckpoints := evm.policies.RetentionPolicy.MaxCheckpoints
	ttl := evm.policies.RetentionPolicy.CheckpointTTL
	
	// Remove old checkpoints by count
	if len(evm.checkpoints) > maxCheckpoints {
		excess := len(evm.checkpoints) - maxCheckpoints
		evm.checkpoints = evm.checkpoints[excess:]
	}
	
	// Remove old checkpoints by TTL
	cutoff := time.Now().Add(-ttl)
	filtered := make([]*Checkpoint, 0)
	for _, cp := range evm.checkpoints {
		if cp.Timestamp.After(cutoff) {
			filtered = append(filtered, cp)
		}
	}
	evm.checkpoints = filtered
}

func (lm *LifecycleManager) enforceSnapshotRetention(evm *EnhancedVM) {
	maxSnapshots := evm.policies.RetentionPolicy.MaxSnapshots
	ttl := evm.policies.RetentionPolicy.SnapshotTTL
	
	// Remove old snapshots by count
	if len(evm.snapshots) > maxSnapshots {
		excess := len(evm.snapshots) - maxSnapshots
		evm.snapshots = evm.snapshots[excess:]
	}
	
	// Remove old snapshots by TTL
	cutoff := time.Now().Add(-ttl)
	filtered := make([]*Snapshot, 0)
	for _, snap := range evm.snapshots {
		if snap.Timestamp.After(cutoff) {
			filtered = append(filtered, snap)
		}
	}
	evm.snapshots = filtered
}

// Background workers

func (lm *LifecycleManager) stateMonitor() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			lm.monitorVMStates()
		case <-lm.ctx.Done():
			return
		}
	}
}

func (lm *LifecycleManager) healthChecker() {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			lm.performHealthChecks()
		case <-lm.ctx.Done():
			return
		}
	}
}

func (lm *LifecycleManager) cleanupWorker() {
	ticker := time.NewTicker(24 * time.Hour)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			lm.performCleanup()
		case <-lm.ctx.Done():
			return
		}
	}
}

func (lm *LifecycleManager) monitorVMStates() {
	lm.mu.RLock()
	vms := make([]*EnhancedVM, 0, len(lm.vms))
	for _, evm := range lm.vms {
		vms = append(vms, evm)
	}
	lm.mu.RUnlock()
	
	for _, evm := range vms {
		// Check for state inconsistencies and auto-correct
		lm.validateVMState(evm)
	}
}

func (lm *LifecycleManager) performHealthChecks() {
	lm.mu.RLock()
	vms := make([]*EnhancedVM, 0, len(lm.vms))
	for _, evm := range lm.vms {
		vms = append(vms, evm)
	}
	lm.mu.RUnlock()
	
	for _, evm := range vms {
		if evm.policies.HealthCheckEnabled && evm.lifecycleState == StateRunning {
			lm.checkVMHealth(evm)
		}
	}
}

func (lm *LifecycleManager) performCleanup() {
	lm.mu.RLock()
	vms := make([]*EnhancedVM, 0, len(lm.vms))
	for _, evm := range lm.vms {
		vms = append(vms, evm)
	}
	lm.mu.RUnlock()
	
	for _, evm := range vms {
		evm.mu.Lock()
		lm.enforceCheckpointRetention(evm)
		lm.enforceSnapshotRetention(evm)
		evm.mu.Unlock()
	}
}

func (lm *LifecycleManager) validateVMState(evm *EnhancedVM) {
	// Implementation would check for state inconsistencies
	// and perform auto-correction where possible
}

func (lm *LifecycleManager) checkVMHealth(evm *EnhancedVM) {
	// Implementation would perform health checks
	// and take corrective actions if needed
}

// Stop stops the lifecycle manager
func (lm *LifecycleManager) Stop() {
	lm.cancel()
	
	// Stop all VMs
	lm.mu.RLock()
	for _, evm := range lm.vms {
		if evm.lifecycleState == StateRunning {
			lm.StopVM(evm.ID(), true)
		}
	}
	lm.mu.RUnlock()
}

// Metrics helpers

// NewLifecycleMetrics creates new lifecycle metrics
func NewLifecycleMetrics() *LifecycleMetrics {
	return &LifecycleMetrics{
		StateTransitions:   make(map[string]int64),
		OperationDurations: make(map[string]time.Duration),
		SuccessRates:       make(map[string]float64),
	}
}

// RecordOperation records an operation in metrics
func (lm *LifecycleMetrics) RecordOperation(operation string, duration time.Duration, success bool) {
	lm.mu.Lock()
	defer lm.mu.Unlock()
	
	lm.TotalOperations++
	
	if success {
		lm.OperationDurations[operation] = duration
	} else {
		lm.FailureCount++
	}
	
	// Calculate success rate
	total := lm.StateTransitions[operation] + 1
	successCount := total
	if !success {
		successCount = total - 1
	}
	lm.SuccessRates[operation] = float64(successCount) / float64(total)
	lm.StateTransitions[operation] = total
}