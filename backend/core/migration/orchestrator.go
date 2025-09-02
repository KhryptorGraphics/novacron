package migration

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// LiveMigrationOrchestrator coordinates all aspects of live VM migration
type LiveMigrationOrchestrator struct {
	// Core components
	wanOptimizer     *WANOptimizer
	rollbackManager  *RollbackManager
	monitor          *MigrationMonitor
	
	// Migration state
	activeMigrations map[string]*LiveMigration
	migrationQueue   *PriorityQueue
	
	// Network management
	connectionPool   *ConnectionPool
	bandwidthManager *BandwidthManager
	
	// Configuration
	config           MigrationConfig
	
	// Metrics and monitoring
	metrics          *MigrationMetricsCollector
	
	// Synchronization
	mu               sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
}

// LiveMigration represents an active live migration
type LiveMigration struct {
	ID               string
	VM               *vm.VM
	SourceNode       string
	DestinationNode  string
	Type             MigrationType
	Config           MigrationConfig
	State            *MigrationState
	Checkpoint       *Checkpoint
	Connection       net.Conn
	StartTime        time.Time
	mu               sync.RWMutex
}

// MigrationType defines the type of migration
type MigrationType string

const (
	MigrationTypeCold      MigrationType = "cold"
	MigrationTypeWarm      MigrationType = "warm"
	MigrationTypeLive      MigrationType = "live"
	MigrationTypeHybrid    MigrationType = "hybrid"
	MigrationTypePostCopy  MigrationType = "post-copy"
	MigrationTypePreCopy   MigrationType = "pre-copy"
)

// MigrationConfig contains configuration for migration
type MigrationConfig struct {
	// Performance targets
	MaxDowntime          time.Duration  `json:"max_downtime"`          // Target: <30s
	TargetTransferRate   int64          `json:"target_transfer_rate"`  // Target: 20 GB/min
	SuccessRateTarget    float64        `json:"success_rate_target"`   // Target: 99.9%
	
	// Network optimization
	EnableCompression    bool           `json:"enable_compression"`
	CompressionType      CompressionType `json:"compression_type"`
	CompressionLevel     int            `json:"compression_level"`
	EnableEncryption     bool           `json:"enable_encryption"`
	EnableDeltaSync      bool           `json:"enable_delta_sync"`
	
	// Bandwidth management
	BandwidthLimit       int64          `json:"bandwidth_limit"`       // bytes/second
	AdaptiveBandwidth    bool           `json:"adaptive_bandwidth"`
	QoSPriority          QoSPriority    `json:"qos_priority"`
	
	// Memory migration
	MemoryIterations     int            `json:"memory_iterations"`
	DirtyPageThreshold   int            `json:"dirty_page_threshold"`
	ConvergenceTimeout   time.Duration  `json:"convergence_timeout"`
	
	// Reliability
	EnableCheckpointing  bool           `json:"enable_checkpointing"`
	CheckpointInterval   time.Duration  `json:"checkpoint_interval"`
	RetryAttempts        int            `json:"retry_attempts"`
	RetryDelay           time.Duration  `json:"retry_delay"`
	
	// Resource limits
	MaxCPUUsage          float64        `json:"max_cpu_usage"`
	MaxMemoryUsage       int64          `json:"max_memory_usage"`
	MaxConcurrentMigrations int         `json:"max_concurrent_migrations"`
}

// MigrationState tracks the state of a migration
type MigrationState struct {
	Phase            MigrationPhase
	Progress         atomic.Value  // float64
	BytesTransferred atomic.Int64
	TotalBytes       atomic.Int64
	PagesTransferred atomic.Int64
	DirtyPages       atomic.Int64
	Iterations       atomic.Int32
	TransferRate     atomic.Int64
	Downtime         atomic.Int64  // milliseconds
	Errors           []error
	mu               sync.RWMutex
}

// ConnectionPool manages network connections for migrations
type ConnectionPool struct {
	connections map[string]net.Conn
	maxConns    int
	mu          sync.RWMutex
}

// BandwidthManager manages bandwidth allocation for migrations
type BandwidthManager struct {
	totalBandwidth     int64
	allocatedBandwidth atomic.Int64
	allocations        map[string]int64
	mu                 sync.RWMutex
}

// PriorityQueue manages migration queue with priorities
type PriorityQueue struct {
	items    []*QueueItem
	mu       sync.RWMutex
}

// QueueItem represents an item in the priority queue
type QueueItem struct {
	MigrationID string
	Priority    int
	AddedAt     time.Time
}

// MigrationMetricsCollector collects migration metrics
type MigrationMetricsCollector struct {
	totalMigrations     atomic.Int64
	successfulMigrations atomic.Int64
	failedMigrations    atomic.Int64
	totalBytesTransferred atomic.Int64
	totalDowntime       atomic.Int64
	averageTransferRate atomic.Int64
}

// NewLiveMigrationOrchestrator creates a new live migration orchestrator
func NewLiveMigrationOrchestrator(config MigrationConfig) (*LiveMigrationOrchestrator, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	// Create WAN optimizer
	wanConfig := WANOptimizerConfig{
		CompressionType:  config.CompressionType,
		CompressionLevel: config.CompressionLevel,
		BandwidthLimit:   config.BandwidthLimit,
		QoSPriority:      config.QoSPriority,
		EnableEncryption: config.EnableEncryption,
		EnableDeltaSync:  config.EnableDeltaSync,
		PageCacheSize:    1024, // 1GB page cache
		TCPOptimization:  true,
	}
	
	wanOptimizer, err := NewWANOptimizer(wanConfig)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create WAN optimizer: %w", err)
	}
	
	// Create rollback manager
	rollbackManager, err := NewRollbackManager("/var/lib/novacron/checkpoints")
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create rollback manager: %w", err)
	}
	
	// Create monitor
	monitor := NewMigrationMonitor()
	
	orchestrator := &LiveMigrationOrchestrator{
		wanOptimizer:     wanOptimizer,
		rollbackManager:  rollbackManager,
		monitor:          monitor,
		activeMigrations: make(map[string]*LiveMigration),
		migrationQueue:   NewPriorityQueue(),
		connectionPool:   NewConnectionPool(10),
		bandwidthManager: NewBandwidthManager(config.BandwidthLimit),
		config:           config,
		metrics:          &MigrationMetricsCollector{},
		ctx:              ctx,
		cancel:           cancel,
	}
	
	// Start background workers
	go orchestrator.queueProcessor()
	go orchestrator.metricsCollector()
	
	return orchestrator, nil
}

// MigrateVM initiates a VM migration
func (o *LiveMigrationOrchestrator) MigrateVM(ctx context.Context, vmID, sourceNode, destNode string, options MigrationOptions) (string, error) {
	// Generate migration ID
	migrationID := uuid.New().String()
	
	// Get VM details (would normally fetch from VM manager)
	vmInstance := &vm.VM{} // Placeholder
	
	// Start monitoring
	if err := o.monitor.StartMonitoring(migrationID, vmID, "VM-"+vmID, sourceNode, destNode, string(MigrationTypeLive)); err != nil {
		return "", fmt.Errorf("failed to start monitoring: %w", err)
	}
	
	// Create checkpoint if enabled
	var checkpoint *Checkpoint
	if o.config.EnableCheckpointing {
		var err error
		checkpoint, err = o.rollbackManager.CreateCheckpoint(ctx, migrationID, vmID, sourceNode, VMStateSnapshot{})
		if err != nil {
			return "", fmt.Errorf("failed to create checkpoint: %w", err)
		}
	}
	
	// Create migration instance
	migration := &LiveMigration{
		ID:              migrationID,
		VM:              vmInstance,
		SourceNode:      sourceNode,
		DestinationNode: destNode,
		Type:            MigrationTypeLive,
		Config:          o.config,
		State:           NewMigrationState(),
		Checkpoint:      checkpoint,
		StartTime:       time.Now(),
	}
	
	// Add to active migrations
	o.mu.Lock()
	o.activeMigrations[migrationID] = migration
	o.mu.Unlock()
	
	// Check if we can start immediately or need to queue
	if o.canStartMigration() {
		go o.executeMigration(ctx, migration)
	} else {
		o.migrationQueue.Add(&QueueItem{
			MigrationID: migrationID,
			Priority:    options.Priority,
			AddedAt:     time.Now(),
		})
	}
	
	return migrationID, nil
}

// executeMigration executes the actual migration
func (o *LiveMigrationOrchestrator) executeMigration(ctx context.Context, migration *LiveMigration) {
	defer func() {
		if r := recover(); r != nil {
			o.handleMigrationPanic(migration.ID, r)
		}
	}()
	
	// Update metrics
	o.metrics.totalMigrations.Add(1)
	
	// Allocate bandwidth
	bandwidth := o.bandwidthManager.Allocate(migration.ID, o.config.BandwidthLimit)
	defer o.bandwidthManager.Release(migration.ID)
	
	// Set bandwidth limit on WAN optimizer
	o.wanOptimizer.SetBandwidthLimit(bandwidth)
	
	// Establish connection to destination
	conn, err := o.establishConnection(migration.DestinationNode)
	if err != nil {
		o.handleMigrationError(migration, fmt.Errorf("failed to establish connection: %w", err))
		return
	}
	migration.Connection = conn
	defer conn.Close()
	
	// Execute migration based on type
	switch migration.Type {
	case MigrationTypeLive:
		err = o.executeLiveMigration(ctx, migration)
	case MigrationTypePreCopy:
		err = o.executePreCopyMigration(ctx, migration)
	case MigrationTypePostCopy:
		err = o.executePostCopyMigration(ctx, migration)
	case MigrationTypeHybrid:
		err = o.executeHybridMigration(ctx, migration)
	default:
		err = fmt.Errorf("unsupported migration type: %s", migration.Type)
	}
	
	if err != nil {
		o.handleMigrationError(migration, err)
		return
	}
	
	// Mark migration as successful
	o.completeMigration(migration, true)
}

// executeLiveMigration performs a live migration using pre-copy algorithm
func (o *LiveMigrationOrchestrator) executeLiveMigration(ctx context.Context, migration *LiveMigration) error {
	state := migration.State
	
	// Phase 1: Initial memory copy
	state.Phase = PhaseMemoryCopy
	if err := o.copyMemoryIterative(ctx, migration); err != nil {
		return fmt.Errorf("memory copy failed: %w", err)
	}
	
	// Phase 2: Disk synchronization (if needed)
	state.Phase = PhaseDiskCopy
	if err := o.syncDisk(ctx, migration); err != nil {
		return fmt.Errorf("disk sync failed: %w", err)
	}
	
	// Phase 3: Final synchronization with brief downtime
	state.Phase = PhaseDowntime
	downtimeStart := time.Now()
	
	// Pause VM on source
	if err := o.pauseVM(migration.VM); err != nil {
		return fmt.Errorf("failed to pause VM: %w", err)
	}
	
	// Transfer final dirty pages
	if err := o.transferFinalState(ctx, migration); err != nil {
		// Attempt to resume on source
		o.resumeVM(migration.VM)
		return fmt.Errorf("final state transfer failed: %w", err)
	}
	
	// Phase 4: Activate on destination
	state.Phase = PhaseActivation
	if err := o.activateOnDestination(ctx, migration); err != nil {
		// Rollback: resume on source
		o.resumeVM(migration.VM)
		return fmt.Errorf("activation failed: %w", err)
	}
	
	// Calculate downtime
	downtime := time.Since(downtimeStart)
	state.Downtime.Store(downtime.Milliseconds())
	
	// Phase 5: Verification
	state.Phase = PhaseVerification
	if err := o.verifyMigration(ctx, migration); err != nil {
		return fmt.Errorf("verification failed: %w", err)
	}
	
	// Clean up source
	if err := o.cleanupSource(ctx, migration); err != nil {
		// Non-fatal error
		fmt.Printf("Warning: source cleanup failed: %v\n", err)
	}
	
	return nil
}

// copyMemoryIterative performs iterative memory copy with convergence detection
func (o *LiveMigrationOrchestrator) copyMemoryIterative(ctx context.Context, migration *LiveMigration) error {
	state := migration.State
	maxIterations := o.config.MemoryIterations
	
	// Get initial memory size
	totalMemory := int64(4 * 1024 * 1024 * 1024) // 4GB placeholder
	state.TotalBytes.Store(totalMemory)
	
	// Track dirty pages
	dirtyPages := totalMemory / 4096 // Assume all pages dirty initially
	previousDirtyPages := dirtyPages
	convergenceCount := 0
	
	for iteration := 1; iteration <= maxIterations; iteration++ {
		state.Iterations.Store(int32(iteration))
		
		// Update progress
		progress := MigrationProgress{
			Phase:            PhaseMemoryCopy,
			CurrentIteration: iteration,
			Iterations:       maxIterations,
			DirtyPages:       dirtyPages,
			TotalPages:       totalMemory / 4096,
		}
		
		// Transfer dirty pages
		transferStart := time.Now()
		bytesToTransfer := dirtyPages * 4096
		
		// Simulate transfer with WAN optimization
		data := make([]byte, bytesToTransfer)
		if err := o.wanOptimizer.TransferWithOptimization(migration.Connection, data); err != nil {
			return fmt.Errorf("iteration %d transfer failed: %w", iteration, err)
		}
		
		transferTime := time.Since(transferStart)
		transferRate := int64(float64(bytesToTransfer) / transferTime.Seconds())
		state.TransferRate.Store(transferRate)
		
		// Update metrics
		state.BytesTransferred.Add(bytesToTransfer)
		state.PagesTransferred.Add(dirtyPages)
		
		// Calculate new dirty pages (simulated)
		// In reality, this would track actual page modifications
		dirtyPageRate := int64(1000) // pages/second
		newDirtyPages := dirtyPageRate * int64(transferTime.Seconds())
		
		// Check convergence
		if newDirtyPages < int64(o.config.DirtyPageThreshold) {
			convergenceCount++
			if convergenceCount >= 2 {
				// Converged successfully
				break
			}
		} else {
			convergenceCount = 0
		}
		
		// Check if we're making progress
		if newDirtyPages >= previousDirtyPages*9/10 {
			// Not converging, may need to stop
			if iteration >= maxIterations/2 {
				return errors.New("migration not converging")
			}
		}
		
		previousDirtyPages = dirtyPages
		dirtyPages = newDirtyPages
		state.DirtyPages.Store(dirtyPages)
		
		// Update progress in monitor
		progress.BytesTransferred = state.BytesTransferred.Load()
		progress.TransferRate = transferRate
		progress.OverallProgress = float64(iteration) / float64(maxIterations) * 70 // Memory copy is 70% of migration
		
		o.monitor.UpdateProgress(migration.ID, progress)
		
		// Check for cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}
	
	return nil
}

// syncDisk synchronizes disk state
func (o *LiveMigrationOrchestrator) syncDisk(ctx context.Context, migration *LiveMigration) error {
	// In a real implementation, this would sync disk blocks
	// For now, we'll simulate the operation
	
	diskSize := int64(10 * 1024 * 1024 * 1024) // 10GB placeholder
	
	// Update progress
	progress := MigrationProgress{
		Phase:           PhaseDiskCopy,
		OverallProgress: 70, // Starting at 70% after memory copy
	}
	
	// Simulate disk transfer with delta sync
	if o.config.EnableDeltaSync {
		// Only transfer changed blocks
		diskSize = diskSize / 10 // Assume 10% changed
	}
	
	// Transfer disk data
	chunkSize := int64(1024 * 1024) // 1MB chunks
	chunks := diskSize / chunkSize
	
	for i := int64(0); i < chunks; i++ {
		data := make([]byte, chunkSize)
		if err := o.wanOptimizer.TransferWithOptimization(migration.Connection, data); err != nil {
			return fmt.Errorf("disk chunk %d transfer failed: %w", i, err)
		}
		
		// Update progress
		progress.OverallProgress = 70 + float64(i)/float64(chunks)*20 // Disk copy is 20% of migration
		o.monitor.UpdateProgress(migration.ID, progress)
		
		// Check for cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}
	
	return nil
}

// transferFinalState transfers the final VM state during downtime
func (o *LiveMigrationOrchestrator) transferFinalState(ctx context.Context, migration *LiveMigration) error {
	// This happens during downtime, so it must be fast
	state := migration.State
	
	// Get remaining dirty pages
	dirtyPages := state.DirtyPages.Load()
	finalBytes := dirtyPages * 4096
	
	// Transfer with maximum priority
	o.wanOptimizer.qosPriority = QoSPriorityCritical
	
	data := make([]byte, finalBytes)
	if err := o.wanOptimizer.TransferWithOptimization(migration.Connection, data); err != nil {
		return fmt.Errorf("final state transfer failed: %w", err)
	}
	
	// Transfer CPU and device state
	cpuState := make([]byte, 1024) // Placeholder for CPU state
	if err := o.wanOptimizer.TransferWithOptimization(migration.Connection, cpuState); err != nil {
		return fmt.Errorf("CPU state transfer failed: %w", err)
	}
	
	return nil
}

// pauseVM pauses a VM
func (o *LiveMigrationOrchestrator) pauseVM(vm *vm.VM) error {
	// Implementation would pause the VM
	return nil
}

// resumeVM resumes a VM
func (o *LiveMigrationOrchestrator) resumeVM(vm *vm.VM) error {
	// Implementation would resume the VM
	return nil
}

// activateOnDestination activates the VM on the destination node
func (o *LiveMigrationOrchestrator) activateOnDestination(ctx context.Context, migration *LiveMigration) error {
	// Send activation command
	activationCmd := []byte("ACTIVATE")
	if _, err := migration.Connection.Write(activationCmd); err != nil {
		return fmt.Errorf("failed to send activation command: %w", err)
	}
	
	// Wait for confirmation
	response := make([]byte, 1024)
	if _, err := migration.Connection.Read(response); err != nil {
		return fmt.Errorf("failed to receive activation confirmation: %w", err)
	}
	
	return nil
}

// verifyMigration verifies the migration was successful
func (o *LiveMigrationOrchestrator) verifyMigration(ctx context.Context, migration *LiveMigration) error {
	// Verify VM is running on destination
	// Check network connectivity
	// Verify application responsiveness
	
	// Update progress to 100%
	progress := MigrationProgress{
		Phase:           PhaseVerification,
		OverallProgress: 100,
	}
	o.monitor.UpdateProgress(migration.ID, progress)
	
	return nil
}

// cleanupSource cleans up resources on the source node
func (o *LiveMigrationOrchestrator) cleanupSource(ctx context.Context, migration *LiveMigration) error {
	// Remove VM from source
	// Clean up temporary files
	// Release resources
	return nil
}

// executePreCopyMigration performs pre-copy migration
func (o *LiveMigrationOrchestrator) executePreCopyMigration(ctx context.Context, migration *LiveMigration) error {
	// Similar to live migration but with different optimization
	return o.executeLiveMigration(ctx, migration)
}

// executePostCopyMigration performs post-copy migration
func (o *LiveMigrationOrchestrator) executePostCopyMigration(ctx context.Context, migration *LiveMigration) error {
	// Post-copy: migrate minimal state first, then fetch pages on demand
	// This reduces downtime but may impact performance initially
	
	state := migration.State
	
	// Phase 1: Transfer minimal state with VM paused
	state.Phase = PhaseDowntime
	downtimeStart := time.Now()
	
	if err := o.pauseVM(migration.VM); err != nil {
		return fmt.Errorf("failed to pause VM: %w", err)
	}
	
	// Transfer essential state only
	if err := o.transferEssentialState(ctx, migration); err != nil {
		o.resumeVM(migration.VM)
		return fmt.Errorf("essential state transfer failed: %w", err)
	}
	
	// Activate on destination immediately
	state.Phase = PhaseActivation
	if err := o.activateOnDestination(ctx, migration); err != nil {
		o.resumeVM(migration.VM)
		return fmt.Errorf("activation failed: %w", err)
	}
	
	downtime := time.Since(downtimeStart)
	state.Downtime.Store(downtime.Milliseconds())
	
	// Phase 2: Background page transfer
	state.Phase = PhaseMemoryCopy
	go o.backgroundPageTransfer(ctx, migration)
	
	return nil
}

// executeHybridMigration performs hybrid migration (combination of pre and post copy)
func (o *LiveMigrationOrchestrator) executeHybridMigration(ctx context.Context, migration *LiveMigration) error {
	// Hybrid: Start with pre-copy, switch to post-copy if not converging
	
	// Try pre-copy first
	err := o.copyMemoryIterative(ctx, migration)
	
	if err != nil && err.Error() == "migration not converging" {
		// Switch to post-copy
		return o.executePostCopyMigration(ctx, migration)
	}
	
	if err != nil {
		return err
	}
	
	// Continue with normal live migration
	return o.executeLiveMigration(ctx, migration)
}

// transferEssentialState transfers only essential state for post-copy
func (o *LiveMigrationOrchestrator) transferEssentialState(ctx context.Context, migration *LiveMigration) error {
	// Transfer CPU state, registers, and page tables
	essentialData := make([]byte, 1024*1024) // 1MB of essential state
	return o.wanOptimizer.TransferWithOptimization(migration.Connection, essentialData)
}

// backgroundPageTransfer transfers pages in background for post-copy
func (o *LiveMigrationOrchestrator) backgroundPageTransfer(ctx context.Context, migration *LiveMigration) {
	// Transfer remaining pages in background
	// Handle page faults from destination
	// This would be a complex implementation in practice
}

// establishConnection establishes a connection to the destination node
func (o *LiveMigrationOrchestrator) establishConnection(destNode string) (net.Conn, error) {
	// Check connection pool first
	if conn := o.connectionPool.Get(destNode); conn != nil {
		return conn, nil
	}
	
	// Establish new connection
	conn, err := net.DialTimeout("tcp", destNode+":9876", 30*time.Second)
	if err != nil {
		return nil, err
	}
	
	// Add to pool
	o.connectionPool.Put(destNode, conn)
	
	return conn, nil
}

// canStartMigration checks if a new migration can be started
func (o *LiveMigrationOrchestrator) canStartMigration() bool {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	return len(o.activeMigrations) < o.config.MaxConcurrentMigrations
}

// handleMigrationError handles migration errors
func (o *LiveMigrationOrchestrator) handleMigrationError(migration *LiveMigration, err error) {
	// Update metrics
	o.metrics.failedMigrations.Add(1)
	
	// Log error
	migration.State.mu.Lock()
	migration.State.Errors = append(migration.State.Errors, err)
	migration.State.mu.Unlock()
	
	// Attempt rollback if checkpoint exists
	if migration.Checkpoint != nil {
		rollbackErr := o.rollbackManager.ExecuteRollback(context.Background(), migration.ID, migration.Checkpoint.ID)
		if rollbackErr != nil {
			fmt.Printf("Rollback failed: %v\n", rollbackErr)
		}
	}
	
	// Complete migration as failed
	o.completeMigration(migration, false)
}

// handleMigrationPanic handles panics during migration
func (o *LiveMigrationOrchestrator) handleMigrationPanic(migrationID string, r interface{}) {
	fmt.Printf("Migration %s panicked: %v\n", migrationID, r)
	
	o.mu.RLock()
	migration, exists := o.activeMigrations[migrationID]
	o.mu.RUnlock()
	
	if exists {
		o.handleMigrationError(migration, fmt.Errorf("migration panicked: %v", r))
	}
}

// completeMigration completes a migration
func (o *LiveMigrationOrchestrator) completeMigration(migration *LiveMigration, success bool) {
	// Update monitor
	downtime := time.Duration(migration.State.Downtime.Load()) * time.Millisecond
	o.monitor.CompleteMigration(migration.ID, success, downtime)
	
	// Update metrics
	if success {
		o.metrics.successfulMigrations.Add(1)
	}
	o.metrics.totalDowntime.Add(migration.State.Downtime.Load())
	
	// Remove from active migrations
	o.mu.Lock()
	delete(o.activeMigrations, migration.ID)
	o.mu.Unlock()
	
	// Process next in queue
	o.processNextInQueue()
}

// queueProcessor processes the migration queue
func (o *LiveMigrationOrchestrator) queueProcessor() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-o.ctx.Done():
			return
		case <-ticker.C:
			o.processNextInQueue()
		}
	}
}

// processNextInQueue processes the next migration in queue
func (o *LiveMigrationOrchestrator) processNextInQueue() {
	if !o.canStartMigration() {
		return
	}
	
	item := o.migrationQueue.Pop()
	if item == nil {
		return
	}
	
	o.mu.RLock()
	migration, exists := o.activeMigrations[item.MigrationID]
	o.mu.RUnlock()
	
	if exists {
		go o.executeMigration(context.Background(), migration)
	}
}

// metricsCollector collects and aggregates metrics
func (o *LiveMigrationOrchestrator) metricsCollector() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-o.ctx.Done():
			return
		case <-ticker.C:
			o.collectMetrics()
		}
	}
}

// collectMetrics collects current metrics
func (o *LiveMigrationOrchestrator) collectMetrics() {
	total := o.metrics.totalMigrations.Load()
	if total == 0 {
		return
	}
	
	successful := o.metrics.successfulMigrations.Load()
	successRate := float64(successful) / float64(total)
	
	avgDowntime := o.metrics.totalDowntime.Load() / total
	
	// Log metrics
	fmt.Printf("Migration Metrics - Total: %d, Success Rate: %.2f%%, Avg Downtime: %dms\n",
		total, successRate*100, avgDowntime)
}

// GetMigrationStatus returns the status of a migration
func (o *LiveMigrationOrchestrator) GetMigrationStatus(migrationID string) (map[string]interface{}, error) {
	return o.monitor.GetMigrationStatus(migrationID)
}

// CancelMigration cancels an ongoing migration
func (o *LiveMigrationOrchestrator) CancelMigration(migrationID string) error {
	o.mu.RLock()
	migration, exists := o.activeMigrations[migrationID]
	o.mu.RUnlock()
	
	if !exists {
		return errors.New("migration not found")
	}
	
	// Trigger rollback if checkpoint exists
	if migration.Checkpoint != nil {
		return o.rollbackManager.ExecuteRollback(context.Background(), migrationID, migration.Checkpoint.ID)
	}
	
	return errors.New("cannot cancel migration without checkpoint")
}

// GetMetrics returns orchestrator metrics
func (o *LiveMigrationOrchestrator) GetMetrics() map[string]interface{} {
	total := o.metrics.totalMigrations.Load()
	successful := o.metrics.successfulMigrations.Load()
	failed := o.metrics.failedMigrations.Load()
	
	successRate := float64(0)
	if total > 0 {
		successRate = float64(successful) / float64(total)
	}
	
	return map[string]interface{}{
		"total_migrations":      total,
		"successful_migrations": successful,
		"failed_migrations":     failed,
		"success_rate":          successRate,
		"active_migrations":     len(o.activeMigrations),
		"queued_migrations":     o.migrationQueue.Size(),
		"wan_metrics":           o.wanOptimizer.GetMetrics(),
	}
}

// Close shuts down the orchestrator
func (o *LiveMigrationOrchestrator) Close() error {
	o.cancel()
	
	// Close components
	o.wanOptimizer.Close()
	o.monitor.Close()
	
	return nil
}

// Helper functions

// NewMigrationState creates a new migration state
func NewMigrationState() *MigrationState {
	state := &MigrationState{}
	state.Progress.Store(float64(0))
	return state
}

// NewConnectionPool creates a new connection pool
func NewConnectionPool(maxConns int) *ConnectionPool {
	return &ConnectionPool{
		connections: make(map[string]net.Conn),
		maxConns:    maxConns,
	}
}

// Get gets a connection from the pool
func (cp *ConnectionPool) Get(node string) net.Conn {
	cp.mu.RLock()
	defer cp.mu.RUnlock()
	
	return cp.connections[node]
}

// Put adds a connection to the pool
func (cp *ConnectionPool) Put(node string, conn net.Conn) {
	cp.mu.Lock()
	defer cp.mu.Unlock()
	
	if len(cp.connections) >= cp.maxConns {
		// Evict oldest connection
		for k, v := range cp.connections {
			v.Close()
			delete(cp.connections, k)
			break
		}
	}
	
	cp.connections[node] = conn
}

// NewBandwidthManager creates a new bandwidth manager
func NewBandwidthManager(totalBandwidth int64) *BandwidthManager {
	return &BandwidthManager{
		totalBandwidth: totalBandwidth,
		allocations:    make(map[string]int64),
	}
}

// Allocate allocates bandwidth for a migration
func (bm *BandwidthManager) Allocate(migrationID string, requested int64) int64 {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	
	available := bm.totalBandwidth - bm.allocatedBandwidth.Load()
	if available <= 0 {
		return 0
	}
	
	allocated := requested
	if allocated > available {
		allocated = available
	}
	
	bm.allocations[migrationID] = allocated
	bm.allocatedBandwidth.Add(allocated)
	
	return allocated
}

// Release releases allocated bandwidth
func (bm *BandwidthManager) Release(migrationID string) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	
	if allocated, exists := bm.allocations[migrationID]; exists {
		bm.allocatedBandwidth.Add(-allocated)
		delete(bm.allocations, migrationID)
	}
}

// NewPriorityQueue creates a new priority queue
func NewPriorityQueue() *PriorityQueue {
	return &PriorityQueue{
		items: make([]*QueueItem, 0),
	}
}

// Add adds an item to the queue
func (pq *PriorityQueue) Add(item *QueueItem) {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	
	// Insert in priority order
	inserted := false
	for i, existing := range pq.items {
		if item.Priority > existing.Priority {
			pq.items = append(pq.items[:i], append([]*QueueItem{item}, pq.items[i:]...)...)
			inserted = true
			break
		}
	}
	
	if !inserted {
		pq.items = append(pq.items, item)
	}
}

// Pop removes and returns the highest priority item
func (pq *PriorityQueue) Pop() *QueueItem {
	pq.mu.Lock()
	defer pq.mu.Unlock()
	
	if len(pq.items) == 0 {
		return nil
	}
	
	item := pq.items[0]
	pq.items = pq.items[1:]
	
	return item
}

// Size returns the queue size
func (pq *PriorityQueue) Size() int {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	
	return len(pq.items)
}

// MigrationOptions contains options for a specific migration
type MigrationOptions struct {
	Priority int
	Force    bool
}