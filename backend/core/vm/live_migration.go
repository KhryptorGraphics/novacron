package vm

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"
)

// LiveMigrationManager handles live VM migration with minimal downtime
type LiveMigrationManager struct {
	mu                sync.RWMutex
	activeMigrations  map[string]*MigrationState
	config            *LiveMigrationConfig
	memoryTransfer    *MemoryTransferEngine
	wanOptimizer      *WANOptimizer
	metricsCollector  *MigrationMetrics
	// WAN migration components
	wanMigrationOptimizer *WANMigrationOptimizer
	deltaSyncManager      *DeltaSyncManager
}

// LiveMigrationConfig configuration for live migration
type LiveMigrationConfig struct {
	MaxIterations        int           // Maximum pre-copy iterations
	MemoryDirtyRate      float64       // Threshold for dirty memory rate
	MaxDowntime          time.Duration // Maximum acceptable downtime
	BandwidthLimit       int64         // Bandwidth limit in bytes/sec
	CompressionEnabled   bool          // Enable compression
	EncryptionEnabled    bool          // Enable encryption
	PreCopyEnabled       bool          // Enable pre-copy phase
	PostCopyEnabled      bool          // Enable post-copy phase
	DeltaCompressionRate float64       // Delta compression ratio
}

// MigrationState tracks the state of an ongoing migration
type MigrationState struct {
	ID                string
	VMID              string
	SourceHost        string
	DestHost          string
	Phase             MigrationPhase
	StartTime         time.Time
	TotalMemory       int64
	TransferredMemory int64
	DirtyMemory       int64
	Iteration         int
	DowntimeStart     time.Time
	DowntimeEnd       time.Time
	Status            string
	Error             error
}

// MigrationPhase represents the current phase of migration
type MigrationPhase string

const (
	PhaseInitialization MigrationPhase = "initialization"
	PhasePreCopy        MigrationPhase = "pre-copy"
	PhaseStopAndCopy    MigrationPhase = "stop-and-copy"
	PhasePostCopy       MigrationPhase = "post-copy"
	PhaseFinalization   MigrationPhase = "finalization"
	PhaseComplete       MigrationPhase = "complete"
	PhaseFailed         MigrationPhase = "failed"
)

// vmDriverRegistry holds the global VM driver for PID resolution
// This should be set during application initialization
var vmDriverRegistry VMDriver

// SetVMDriverForPIDResolution sets the VM driver used for PID resolution
func SetVMDriverForPIDResolution(driver VMDriver) {
	vmDriverRegistry = driver
}

// VMPIDResolver is an interface for getting VM process PIDs
type VMPIDResolver interface {
	GetProcessPID(vmID string) int
}

// getVMProcessPID retrieves the hypervisor process PID for a given VM.
// This function uses the configured VM driver to get the actual QEMU process PID.
//
// For KVM/QEMU VMs, it tries multiple methods in order:
// 1. Check the driver's cached PID (if VM is tracked)
// 2. Read from PID file at /var/lib/novacron/vms/<vmid>/qemu.pid
// 3. Read from libvirt PID file at /var/run/libvirt/qemu/<vmid>.pid
// 4. Scan /proc for QEMU process with matching VM ID
//
// Returns 0 if the PID cannot be determined.
func getVMProcessPID(vmID string) uint32 {
	// Try using the registered driver
	if vmDriverRegistry != nil {
		if resolver, ok := vmDriverRegistry.(VMPIDResolver); ok {
			pid := resolver.GetProcessPID(vmID)
			if pid > 0 {
				return uint32(pid)
			}
		}
	}

	// Fallback: try to read from common PID file locations
	// Try libvirt path first
	libvirtPidPath := fmt.Sprintf("/var/run/libvirt/qemu/%s.pid", vmID)
	if pid := readPIDFromFile(libvirtPidPath); pid > 0 {
		return pid
	}

	// Try NovaCron's default path
	novacronPidPath := fmt.Sprintf("/var/lib/novacron/vms/%s/qemu.pid", vmID)
	if pid := readPIDFromFile(novacronPidPath); pid > 0 {
		return pid
	}

	// Final fallback: scan /proc for QEMU process
	pid := scanProcForQEMU(vmID)
	if pid > 0 {
		return pid
	}

	// PID not found - log warning
	logger.WithField("vmID", vmID).Warn("Could not determine VM process PID")
	return 0
}

// readPIDFromFile reads a PID from a file and verifies the process exists
func readPIDFromFile(path string) uint32 {
	data, err := readFile(path)
	if err != nil {
		return 0
	}

	// Trim whitespace and newlines
	pidStr := trimWhitespace(data)
	if len(pidStr) == 0 {
		return 0
	}

	pid := 0
	for _, c := range pidStr {
		if c < '0' || c > '9' {
			break
		}
		pid = pid*10 + int(c-'0')
	}

	if pid <= 0 {
		return 0
	}

	// Verify process exists
	procPath := fmt.Sprintf("/proc/%d", pid)
	if _, err := statFile(procPath); err != nil {
		return 0
	}

	return uint32(pid)
}

// readFile is a helper to read file contents
func readFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// statFile is a helper to check if a file/directory exists
func statFile(path string) (os.FileInfo, error) {
	return os.Stat(path)
}

// trimWhitespace removes leading and trailing whitespace
func trimWhitespace(s string) string {
	start := 0
	end := len(s)

	for start < end && (s[start] == ' ' || s[start] == '\n' || s[start] == '\r' || s[start] == '\t') {
		start++
	}
	for end > start && (s[end-1] == ' ' || s[end-1] == '\n' || s[end-1] == '\r' || s[end-1] == '\t') {
		end--
	}

	return s[start:end]
}

// scanProcForQEMU scans /proc for a QEMU process associated with the given VM ID
func scanProcForQEMU(vmID string) uint32 {
	procDir, err := os.Open("/proc")
	if err != nil {
		return 0
	}
	defer procDir.Close()

	entries, err := procDir.Readdirnames(-1)
	if err != nil {
		return 0
	}

	for _, entry := range entries {
		// Check if entry is a PID (all digits)
		isDigit := true
		for _, c := range entry {
			if c < '0' || c > '9' {
				isDigit = false
				break
			}
		}
		if !isDigit {
			continue
		}

		// Parse PID
		pid := 0
		for _, c := range entry {
			pid = pid*10 + int(c-'0')
		}

		// Read cmdline
		cmdlinePath := fmt.Sprintf("/proc/%d/cmdline", pid)
		cmdlineData, err := os.ReadFile(cmdlinePath)
		if err != nil {
			continue
		}

		cmdline := string(cmdlineData)

		// Check if it's a QEMU process with our VM ID
		if hasQEMUAndVMID(cmdline, vmID) {
			return uint32(pid)
		}
	}

	return 0
}

// hasQEMUAndVMID checks if a cmdline contains both "qemu" and the VM ID
func hasQEMUAndVMID(cmdline, vmID string) bool {
	hasQEMU := false
	hasVMID := false

	// cmdline has null-separated arguments
	for i := 0; i < len(cmdline); {
		// Find next null or end of string
		j := i
		for j < len(cmdline) && cmdline[j] != 0 {
			j++
		}

		arg := cmdline[i:j]

		// Check for qemu
		if !hasQEMU && len(arg) >= 4 {
			for k := 0; k <= len(arg)-4; k++ {
				if arg[k:k+4] == "qemu" {
					hasQEMU = true
					break
				}
			}
		}

		// Check for VM ID
		if !hasVMID && len(arg) >= len(vmID) {
			for k := 0; k <= len(arg)-len(vmID); k++ {
				if arg[k:k+len(vmID)] == vmID {
					hasVMID = true
					break
				}
			}
		}

		if hasQEMU && hasVMID {
			return true
		}

		i = j + 1
	}

	return false
}

// logger is a package-level logger for live migration
var logger = &migrationLogger{}

type migrationLogger struct{}

func (l *migrationLogger) WithField(key string, value interface{}) *migrationLogEntry {
	return &migrationLogEntry{fields: map[string]interface{}{key: value}}
}

type migrationLogEntry struct {
	fields map[string]interface{}
}

func (e *migrationLogEntry) Warn(msg string) {
	// In production, this would use the actual logger
	fmt.Printf("WARN: %s %v\n", msg, e.fields)
}

// NewLiveMigrationManager creates a new live migration manager
func NewLiveMigrationManager(config *LiveMigrationConfig) *LiveMigrationManager {
	return &LiveMigrationManager{
		activeMigrations: make(map[string]*MigrationState),
		config:           config,
		memoryTransfer:   NewMemoryTransferEngine(config),
		wanOptimizer:     NewWANOptimizer(config),
		metricsCollector: NewMigrationMetrics(),
	}
}

// StartLiveMigration initiates a live migration
func (lmm *LiveMigrationManager) StartLiveMigration(ctx context.Context, vmID, sourceHost, destHost string) (*MigrationState, error) {
	lmm.mu.Lock()
	defer lmm.mu.Unlock()

	migrationID := fmt.Sprintf("migration-%s-%d", vmID, time.Now().Unix())
	
	state := &MigrationState{
		ID:         migrationID,
		VMID:       vmID,
		SourceHost: sourceHost,
		DestHost:   destHost,
		Phase:      PhaseInitialization,
		StartTime:  time.Now(),
		Status:     "starting",
	}
	
	lmm.activeMigrations[migrationID] = state
	
	// Start migration in background
	go lmm.executeMigration(ctx, state)
	
	return state, nil
}

// executeMigration performs the actual migration
func (lmm *LiveMigrationManager) executeMigration(ctx context.Context, state *MigrationState) {
	// Phase 1: Initialization
	if err := lmm.initializeMigration(ctx, state); err != nil {
		lmm.failMigration(state, err)
		return
	}
	
	// Phase 2: Pre-copy (iterative memory transfer)
	if lmm.config.PreCopyEnabled {
		if err := lmm.preCopyPhase(ctx, state); err != nil {
			lmm.failMigration(state, err)
			return
		}
	}
	
	// Phase 3: Stop-and-copy (final memory transfer with VM paused)
	if err := lmm.stopAndCopyPhase(ctx, state); err != nil {
		lmm.failMigration(state, err)
		return
	}
	
	// Phase 4: Post-copy (optional, for remaining pages)
	if lmm.config.PostCopyEnabled {
		if err := lmm.postCopyPhase(ctx, state); err != nil {
			lmm.failMigration(state, err)
			return
		}
	}
	
	// Phase 5: Finalization
	if err := lmm.finalizeMigration(ctx, state); err != nil {
		lmm.failMigration(state, err)
		return
	}
	
	state.Phase = PhaseComplete
	state.Status = "completed"
	lmm.metricsCollector.RecordSuccess(state)
}

// initializeMigration prepares for migration
func (lmm *LiveMigrationManager) initializeMigration(ctx context.Context, state *MigrationState) error {
	state.Phase = PhaseInitialization
	state.Status = "initializing"

	// Get VM memory size
	state.TotalMemory = 8 * 1024 * 1024 * 1024 // 8GB example

	// Prepare destination host
	if err := lmm.prepareDestination(ctx, state); err != nil {
		return fmt.Errorf("failed to prepare destination: %w", err)
	}

	// Initialize WAN optimization if needed (legacy support)
	if state.SourceHost != state.DestHost {
		if lmm.wanOptimizer != nil {
			if err := lmm.wanOptimizer.Initialize(ctx, state.SourceHost, state.DestHost); err != nil {
				return fmt.Errorf("failed to initialize WAN optimizer: %w", err)
			}
		}

		// Initialize new WAN migration optimizer
		if lmm.config.CompressionEnabled {
			wanConfig := DefaultWANMigrationConfig()
			wanConfig.CompressionLevel = 3
			wanConfig.EnableDeltaSync = true
			wanConfig.MaxBandwidthMbps = int(lmm.config.BandwidthLimit / (1024 * 1024 / 8))

			lmm.wanMigrationOptimizer = NewWANMigrationOptimizer(wanConfig)

			// Initialize delta sync manager
			deltaSyncConfig := DefaultDeltaSyncConfig()
			deltaSyncConfig.EnableCompression = lmm.config.CompressionEnabled
			deltaSyncConfig.CompressionLevel = 3
			deltaSyncConfig.BlockSizeKB = 64

			// Enable eBPF if supported
			if IsEBPFSupported() {
				deltaSyncConfig.EnableEBPFFiltering = true
				deltaSyncConfig.FallbackOnEBPFError = true
				deltaSyncConfig.EBPFAgingThreshold = 5 * time.Second
			}

			lmm.deltaSyncManager = NewDeltaSyncManager(deltaSyncConfig)

			// Try to enable eBPF filtering for the VM process (if we have a PID)
			if deltaSyncConfig.EnableEBPFFiltering {
				// Get the actual VM process ID from the VM state
				// This assumes the VM has a method to get its hypervisor process PID
				vmPID := getVMProcessPID(state.VMID)
				if vmPID > 0 {
					if err := lmm.deltaSyncManager.EnableEBPFFiltering(vmPID); err != nil {
						// Log but don't fail - fallback to standard delta sync
						fmt.Printf("eBPF filtering initialization failed (non-fatal): %v\n", err)
					}
				} else {
					logger.WithField("vmID", state.VMID).Warn("Could not determine VM process PID, eBPF filtering disabled")
				}
			}
		}
	}

	return nil
}

// preCopyPhase performs iterative memory pre-copy
func (lmm *LiveMigrationManager) preCopyPhase(ctx context.Context, state *MigrationState) error {
	state.Phase = PhasePreCopy
	state.Status = "pre-copying memory"
	
	for state.Iteration = 0; state.Iteration < lmm.config.MaxIterations; state.Iteration++ {
		// Transfer dirty pages
		transferred, dirty, err := lmm.memoryTransfer.TransferDirtyPages(ctx, state)
		if err != nil {
			return fmt.Errorf("pre-copy iteration %d failed: %w", state.Iteration, err)
		}
		
		state.TransferredMemory += transferred
		state.DirtyMemory = dirty
		
		// Check convergence
		dirtyRate := float64(dirty) / float64(state.TotalMemory)
		if dirtyRate < lmm.config.MemoryDirtyRate {
			// Dirty rate is low enough, proceed to stop-and-copy
			break
		}
		
		// Check if we should give up on pre-copy
		if state.Iteration >= lmm.config.MaxIterations-1 {
			// Too many iterations, proceed anyway
			break
		}
	}

	return nil
}

// stopAndCopyPhase performs final memory transfer with VM paused
func (lmm *LiveMigrationManager) stopAndCopyPhase(ctx context.Context, state *MigrationState) error {
	state.Phase = PhaseStopAndCopy
	state.Status = "stop-and-copy"
	state.DowntimeStart = time.Now()

	// Pause VM on source
	if err := lmm.pauseVM(ctx, state.VMID, state.SourceHost); err != nil {
		return fmt.Errorf("failed to pause VM: %w", err)
	}

	// Transfer remaining memory and CPU state
	if err := lmm.memoryTransfer.TransferFinalState(ctx, state); err != nil {
		// Resume VM on failure
		_ = lmm.resumeVM(ctx, state.VMID, state.SourceHost)
		return fmt.Errorf("failed to transfer final state: %w", err)
	}

	// Resume VM on destination
	if err := lmm.resumeVM(ctx, state.VMID, state.DestHost); err != nil {
		return fmt.Errorf("failed to resume VM on destination: %w", err)
	}

	state.DowntimeEnd = time.Now()
	downtime := state.DowntimeEnd.Sub(state.DowntimeStart)

	// Check if downtime is acceptable
	if downtime > lmm.config.MaxDowntime {
		return fmt.Errorf("downtime %v exceeded maximum %v", downtime, lmm.config.MaxDowntime)
	}

	return nil
}

// postCopyPhase handles post-copy migration if enabled
func (lmm *LiveMigrationManager) postCopyPhase(ctx context.Context, state *MigrationState) error {
	state.Phase = PhasePostCopy
	state.Status = "post-copy"

	// Transfer remaining pages on-demand
	if err := lmm.memoryTransfer.TransferOnDemand(ctx, state); err != nil {
		return fmt.Errorf("post-copy failed: %w", err)
	}

	return nil
}

// finalizeMigration completes the migration
func (lmm *LiveMigrationManager) finalizeMigration(ctx context.Context, state *MigrationState) error {
	state.Phase = PhaseFinalization
	state.Status = "finalizing"

	// Cleanup WAN migration components
	if lmm.deltaSyncManager != nil {
		lmm.deltaSyncManager.Close()
		lmm.deltaSyncManager = nil
	}

	if lmm.wanMigrationOptimizer != nil {
		lmm.wanMigrationOptimizer.Close()
		lmm.wanMigrationOptimizer = nil
	}

	// Cleanup source VM
	if err := lmm.cleanupSource(ctx, state); err != nil {
		return fmt.Errorf("failed to cleanup source: %w", err)
	}

	// Update VM metadata
	if err := lmm.updateVMMetadata(ctx, state); err != nil {
		return fmt.Errorf("failed to update metadata: %w", err)
	}

	return nil
}

// Helper methods
func (lmm *LiveMigrationManager) prepareDestination(ctx context.Context, state *MigrationState) error {
	return nil
}

func (lmm *LiveMigrationManager) pauseVM(ctx context.Context, vmID, host string) error {
	return nil
}

func (lmm *LiveMigrationManager) resumeVM(ctx context.Context, vmID, host string) error {
	return nil
}

func (lmm *LiveMigrationManager) cleanupSource(ctx context.Context, state *MigrationState) error {
	return nil
}

func (lmm *LiveMigrationManager) updateVMMetadata(ctx context.Context, state *MigrationState) error {
	return nil
}

func (lmm *LiveMigrationManager) failMigration(state *MigrationState, err error) {
	state.Phase = PhaseFailed
	state.Status = "failed"
	state.Error = err
	lmm.metricsCollector.RecordFailure(state, err)

	// Cleanup WAN migration components on failure
	if lmm.deltaSyncManager != nil {
		lmm.deltaSyncManager.Close()
		lmm.deltaSyncManager = nil
	}

	if lmm.wanMigrationOptimizer != nil {
		lmm.wanMigrationOptimizer.Close()
		lmm.wanMigrationOptimizer = nil
	}
}

// GetMigrationState returns the current state of a migration
func (lmm *LiveMigrationManager) GetMigrationState(migrationID string) (*MigrationState, error) {
	lmm.mu.RLock()
	defer lmm.mu.RUnlock()

	state, exists := lmm.activeMigrations[migrationID]
	if !exists {
		return nil, fmt.Errorf("migration not found: %s", migrationID)
	}

	return state, nil
}

// MemoryTransferEngine handles memory transfer operations
type MemoryTransferEngine struct {
	config *LiveMigrationConfig
}

func NewMemoryTransferEngine(config *LiveMigrationConfig) *MemoryTransferEngine {
	return &MemoryTransferEngine{config: config}
}

func (mte *MemoryTransferEngine) TransferDirtyPages(ctx context.Context, state *MigrationState) (transferred, dirty int64, err error) {
	transferred = state.TotalMemory / 10
	dirty = transferred / 2

	if mte.config.CompressionEnabled {
		transferred = int64(float64(transferred) * mte.config.DeltaCompressionRate)
	}

	time.Sleep(100 * time.Millisecond)
	return transferred, dirty, nil
}

func (mte *MemoryTransferEngine) TransferFinalState(ctx context.Context, state *MigrationState) error {
	time.Sleep(50 * time.Millisecond)
	return nil
}

func (mte *MemoryTransferEngine) TransferOnDemand(ctx context.Context, state *MigrationState) error {
	time.Sleep(100 * time.Millisecond)
	return nil
}

// MigrationMetrics collects migration metrics
type MigrationMetrics struct {
	mu              sync.RWMutex
	totalMigrations int
	successCount    int
	failureCount    int
	totalDowntime   time.Duration
}

func NewMigrationMetrics() *MigrationMetrics {
	return &MigrationMetrics{}
}

func (mm *MigrationMetrics) RecordSuccess(state *MigrationState) {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	mm.totalMigrations++
	mm.successCount++
	mm.totalDowntime += state.DowntimeEnd.Sub(state.DowntimeStart)
}

func (mm *MigrationMetrics) RecordFailure(state *MigrationState, err error) {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	mm.totalMigrations++
	mm.failureCount++
}

func (mm *MigrationMetrics) GetStats() map[string]interface{} {
	mm.mu.RLock()
	defer mm.mu.RUnlock()

	avgDowntime := time.Duration(0)
	if mm.successCount > 0 {
		avgDowntime = mm.totalDowntime / time.Duration(mm.successCount)
	}

	return map[string]interface{}{
		"total_migrations": mm.totalMigrations,
		"success_count":    mm.successCount,
		"failure_count":    mm.failureCount,
		"success_rate":     float64(mm.successCount) / float64(mm.totalMigrations),
		"avg_downtime_ms":  avgDowntime.Milliseconds(),
	}
}
