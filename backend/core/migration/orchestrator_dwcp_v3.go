// Package migration implements DWCP v3 migration orchestrator with mode-aware capabilities
package migration

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/encoding"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/partition"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/prediction"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/sync"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/transport"
)

// DWCPv3Config configuration for DWCP v3 migration
type DWCPv3Config struct {
	// Network mode configuration
	NetworkMode         upgrade.NetworkMode `json:"network_mode"`
	AutoSwitchMode      bool               `json:"auto_switch_mode"`
	ModeThresholds      *ModeThresholds    `json:"mode_thresholds"`

	// Component enablement
	EnableAMSTv3        bool `json:"enable_amst_v3"`
	EnableHDEv3         bool `json:"enable_hde_v3"`
	EnablePBAv3         bool `json:"enable_pba_v3"`
	EnableITPv3         bool `json:"enable_itp_v3"`
	EnableASSv3         bool `json:"enable_ass_v3"`

	// Performance targets per mode
	DatacenterTargets   *PerformanceTargets `json:"datacenter_targets"`
	InternetTargets     *PerformanceTargets `json:"internet_targets"`
	HybridTargets       *PerformanceTargets `json:"hybrid_targets"`

	// Resource limits
	MaxMemoryUsage      int64         `json:"max_memory_usage"`
	MaxCPUPercent       float64       `json:"max_cpu_percent"`
	BandwidthLimit      int64         `json:"bandwidth_limit"`

	// Migration behavior
	EnablePrefetching   bool          `json:"enable_prefetching"`
	EnableCompression   bool          `json:"enable_compression"`
	CompressionLevel    int           `json:"compression_level"`
	EnableEncryption    bool          `json:"enable_encryption"`

	// Timeouts
	MigrationTimeout    time.Duration `json:"migration_timeout"`
	ConnectionTimeout   time.Duration `json:"connection_timeout"`
}

// ModeThresholds for automatic mode switching
type ModeThresholds struct {
	// Network conditions
	LatencyThreshold    time.Duration `json:"latency_threshold"`    // Switch to internet mode above this
	BandwidthThreshold  int64        `json:"bandwidth_threshold"`   // Switch to datacenter mode above this
	PacketLossThreshold float64      `json:"packet_loss_threshold"` // Switch to internet mode above this

	// Migration metrics
	DowntimeThreshold   time.Duration `json:"downtime_threshold"`   // Maximum acceptable downtime
	CompressionRatio    float64       `json:"compression_ratio"`     // Minimum compression effectiveness
}

// PerformanceTargets for different network modes
type PerformanceTargets struct {
	MaxDowntime         time.Duration `json:"max_downtime"`
	TargetThroughput    int64        `json:"target_throughput"`
	CompressionRatio    float64      `json:"compression_ratio"`
	MaxIterations       int          `json:"max_iterations"`
}

// DWCPv3Orchestrator orchestrates migration with DWCP v3 components
type DWCPv3Orchestrator struct {
	// Base orchestrator
	*LiveMigrationOrchestrator

	// Configuration
	config      DWCPv3Config

	// DWCP v3 components
	amst        *transport.AMSTv3
	hde         *encoding.HDEv3
	pba         *prediction.PBAv3
	itp         *partition.ITPv3
	ass         *sync.ASSv3

	// Mode management
	currentMode upgrade.NetworkMode
	modeMu      sync.RWMutex

	// Active migrations
	migrations  map[string]*DWCPv3Migration
	migMu       sync.RWMutex

	// Metrics
	metrics     *DWCPv3Metrics

	// Context
	ctx         context.Context
	cancel      context.CancelFunc
}

// DWCPv3Migration tracks a v3 migration
type DWCPv3Migration struct {
	*LiveMigration

	// DWCP v3 specific
	Mode                upgrade.NetworkMode
	CompressionAlgo     encoding.CompressionAlgorithm
	TransportStreams    int
	PredictedBandwidth  int64
	ActualBandwidth     int64
	CompressionRatio    float64

	// Phase tracking
	CurrentPhase        MigrationPhaseV3
	PhaseStartTime      time.Time
	PhaseDurations      map[MigrationPhaseV3]time.Duration

	// Memory tracking
	DirtyPageRate       float64
	ConvergenceRate     float64
	PrefetchedPages     int64
	CompressedPages     int64

	// Network tracking
	RTT                 time.Duration
	PacketLoss          float64
	Jitter              time.Duration

	// Adaptive parameters
	AdaptiveStreams     bool
	AdaptiveCompression bool
	ModeSwitches        int
}

// MigrationPhaseV3 represents enhanced migration phases
type MigrationPhaseV3 string

const (
	PhaseV3Init         MigrationPhaseV3 = "initialization"
	PhaseV3Prefetch     MigrationPhaseV3 = "prefetch"
	PhaseV3PreCopy      MigrationPhaseV3 = "pre-copy"
	PhaseV3Converge     MigrationPhaseV3 = "convergence"
	PhaseV3StopCopy     MigrationPhaseV3 = "stop-and-copy"
	PhaseV3PostCopy     MigrationPhaseV3 = "post-copy"
	PhaseV3Verify       MigrationPhaseV3 = "verification"
	PhaseV3Cleanup      MigrationPhaseV3 = "cleanup"
	PhaseV3Complete     MigrationPhaseV3 = "complete"
)

// DWCPv3Metrics tracks v3-specific metrics
type DWCPv3Metrics struct {
	// Mode metrics
	DatacenterMigrations atomic.Int64
	InternetMigrations   atomic.Int64
	HybridMigrations     atomic.Int64
	ModeSwitches         atomic.Int64

	// Performance metrics
	AverageDowntime      atomic.Int64 // milliseconds
	AverageSpeedup       atomic.Int64 // percentage * 100
	CompressionSavings   atomic.Int64 // bytes saved

	// Component metrics
	AMSTStreamsUsed      atomic.Int64
	HDECompressions      atomic.Int64
	PBAPredictions       atomic.Int64
	ITPPartitions        atomic.Int64
	ASSSyncs             atomic.Int64

	// Success metrics
	TotalMigrations      atomic.Int64
	SuccessfulMigrations atomic.Int64
	FailedMigrations     atomic.Int64
}

// DefaultDWCPv3Config returns default v3 configuration
func DefaultDWCPv3Config() DWCPv3Config {
	return DWCPv3Config{
		NetworkMode:      upgrade.ModeHybrid,
		AutoSwitchMode:   true,
		ModeThresholds: &ModeThresholds{
			LatencyThreshold:    10 * time.Millisecond,
			BandwidthThreshold:  1 * 1024 * 1024 * 1024, // 1 Gbps
			PacketLossThreshold: 0.001,
			DowntimeThreshold:   1 * time.Second,
			CompressionRatio:    1.5,
		},
		EnableAMSTv3:     true,
		EnableHDEv3:      true,
		EnablePBAv3:      true,
		EnableITPv3:      true,
		EnableASSv3:      true,
		DatacenterTargets: &PerformanceTargets{
			MaxDowntime:      500 * time.Millisecond,
			TargetThroughput: 10 * 1024 * 1024 * 1024, // 10 Gbps
			CompressionRatio: 1.2,                      // Minimal compression
			MaxIterations:    5,
		},
		InternetTargets: &PerformanceTargets{
			MaxDowntime:      90 * time.Second,
			TargetThroughput: 100 * 1024 * 1024, // 100 Mbps
			CompressionRatio: 3.0,                // Aggressive compression
			MaxIterations:    10,
		},
		HybridTargets: &PerformanceTargets{
			MaxDowntime:      5 * time.Second,
			TargetThroughput: 1 * 1024 * 1024 * 1024, // 1 Gbps
			CompressionRatio: 2.0,
			MaxIterations:    7,
		},
		MaxMemoryUsage:    4 * 1024 * 1024 * 1024, // 4GB
		MaxCPUPercent:     80.0,
		BandwidthLimit:    0, // No limit by default
		EnablePrefetching: true,
		EnableCompression: true,
		CompressionLevel:  6,
		EnableEncryption:  true,
		MigrationTimeout:  30 * time.Minute,
		ConnectionTimeout: 30 * time.Second,
	}
}

// NewDWCPv3Orchestrator creates a new v3 orchestrator
func NewDWCPv3Orchestrator(baseConfig MigrationConfig, dwcpConfig DWCPv3Config) (*DWCPv3Orchestrator, error) {
	// Create base orchestrator
	baseOrchestrator, err := NewLiveMigrationOrchestrator(baseConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create base orchestrator: %w", err)
	}

	ctx, cancel := context.WithCancel(context.Background())

	orchestrator := &DWCPv3Orchestrator{
		LiveMigrationOrchestrator: baseOrchestrator,
		config:                    dwcpConfig,
		currentMode:               dwcpConfig.NetworkMode,
		migrations:                make(map[string]*DWCPv3Migration),
		metrics:                   &DWCPv3Metrics{},
		ctx:                       ctx,
		cancel:                    cancel,
	}

	// Initialize components based on configuration
	if err := orchestrator.initializeComponents(); err != nil {
		cancel()
		return nil, fmt.Errorf("failed to initialize components: %w", err)
	}

	// Start background monitor
	go orchestrator.monitorLoop()

	return orchestrator, nil
}

// initializeComponents initializes DWCP v3 components
func (o *DWCPv3Orchestrator) initializeComponents() error {
	nodeID := fmt.Sprintf("migration-node-%d", time.Now().UnixNano())

	// Initialize AMST v3 (hybrid transport)
	if o.config.EnableAMSTv3 {
		amstConfig := transport.DefaultAMSTv3Config(nodeID)
		amstConfig.NetworkMode = o.config.NetworkMode
		amstConfig.BandwidthLimit = o.config.BandwidthLimit

		amst, err := transport.NewAMSTv3(amstConfig)
		if err != nil {
			return fmt.Errorf("failed to create AMST v3: %w", err)
		}
		o.amst = amst
	}

	// Initialize HDE v3 (ML compression)
	if o.config.EnableHDEv3 {
		hdeConfig := encoding.DefaultHDEv3Config(nodeID)
		hdeConfig.NetworkMode = o.config.NetworkMode
		hdeConfig.MaxMemoryUsage = o.config.MaxMemoryUsage

		hde, err := encoding.NewHDEv3(hdeConfig)
		if err != nil {
			return fmt.Errorf("failed to create HDE v3: %w", err)
		}
		o.hde = hde
	}

	// Initialize PBA v3 (predictive bandwidth)
	if o.config.EnablePBAv3 {
		pbaConfig := prediction.DefaultPBAv3Config(nodeID)

		pba, err := prediction.NewPBAv3(pbaConfig)
		if err != nil {
			return fmt.Errorf("failed to create PBA v3: %w", err)
		}
		o.pba = pba
	}

	// Initialize ITP v3 (intelligent partitioning)
	if o.config.EnableITPv3 {
		itpConfig := partition.DefaultITPv3Config(nodeID)

		itp, err := partition.NewITPv3(itpConfig)
		if err != nil {
			return fmt.Errorf("failed to create ITP v3: %w", err)
		}
		o.itp = itp
	}

	// Initialize ASS v3 (adaptive synchronization)
	if o.config.EnableASSv3 {
		assConfig := sync.DefaultASSv3Config(nodeID)

		ass, err := sync.NewASSv3(assConfig)
		if err != nil {
			return fmt.Errorf("failed to create ASS v3: %w", err)
		}
		o.ass = ass
	}

	return nil
}

// StartMigration starts a new v3 migration
func (o *DWCPv3Orchestrator) StartMigration(ctx context.Context, vmID string, sourceNode, destNode string) (*DWCPv3Migration, error) {
	// Create base migration
	baseMigration, err := o.createBaseMigration(vmID, sourceNode, destNode)
	if err != nil {
		return nil, fmt.Errorf("failed to create base migration: %w", err)
	}

	// Determine network mode
	mode := o.determineNetworkMode(sourceNode, destNode)

	// Create v3 migration
	migration := &DWCPv3Migration{
		LiveMigration:       baseMigration,
		Mode:                mode,
		CurrentPhase:        PhaseV3Init,
		PhaseStartTime:      time.Now(),
		PhaseDurations:      make(map[MigrationPhaseV3]time.Duration),
		AdaptiveStreams:     o.config.EnableAMSTv3,
		AdaptiveCompression: o.config.EnableHDEv3,
	}

	// Store migration
	o.migMu.Lock()
	o.migrations[migration.ID] = migration
	o.migMu.Unlock()

	// Start migration in background
	go o.executeMigrationV3(ctx, migration)

	return migration, nil
}

// determineNetworkMode determines the appropriate network mode
func (o *DWCPv3Orchestrator) determineNetworkMode(sourceNode, destNode string) upgrade.NetworkMode {
	// If auto-switch is disabled, use configured mode
	if !o.config.AutoSwitchMode {
		return o.config.NetworkMode
	}

	// Measure network conditions
	rtt, bandwidth, loss := o.measureNetworkConditions(sourceNode, destNode)

	// Determine mode based on thresholds
	thresholds := o.config.ModeThresholds

	// Datacenter mode: low latency, high bandwidth, minimal loss
	if rtt < thresholds.LatencyThreshold &&
	   bandwidth > thresholds.BandwidthThreshold &&
	   loss < thresholds.PacketLossThreshold {
		return upgrade.ModeDatacenter
	}

	// Internet mode: high latency, low bandwidth, or high loss
	if rtt > thresholds.LatencyThreshold*10 ||
	   bandwidth < thresholds.BandwidthThreshold/10 ||
	   loss > thresholds.PacketLossThreshold*10 {
		return upgrade.ModeInternet
	}

	// Hybrid mode: moderate conditions
	return upgrade.ModeHybrid
}

// measureNetworkConditions measures network conditions between nodes
func (o *DWCPv3Orchestrator) measureNetworkConditions(sourceNode, destNode string) (time.Duration, int64, float64) {
	// In production, would use actual network measurements
	// For now, return simulated values based on node names

	if sourceNode == destNode {
		// Same node
		return 100 * time.Microsecond, 40 * 1024 * 1024 * 1024, 0.0
	}

	// Check if nodes are in same datacenter (simplified check)
	if len(sourceNode) > 3 && len(destNode) > 3 && sourceNode[:3] == destNode[:3] {
		// Same datacenter
		return 500 * time.Microsecond, 10 * 1024 * 1024 * 1024, 0.0001
	}

	// Different datacenters or internet
	return 50 * time.Millisecond, 100 * 1024 * 1024, 0.001
}

// executeMigrationV3 executes the v3 migration
func (o *DWCPv3Orchestrator) executeMigrationV3(ctx context.Context, migration *DWCPv3Migration) {
	defer o.cleanupMigration(migration)

	// Update metrics based on mode
	switch migration.Mode {
	case upgrade.ModeDatacenter:
		o.metrics.DatacenterMigrations.Add(1)
	case upgrade.ModeInternet:
		o.metrics.InternetMigrations.Add(1)
	case upgrade.ModeHybrid:
		o.metrics.HybridMigrations.Add(1)
	}
	o.metrics.TotalMigrations.Add(1)

	// Execute migration phases
	phases := []struct {
		phase MigrationPhaseV3
		fn    func(context.Context, *DWCPv3Migration) error
	}{
		{PhaseV3Init, o.phaseInitialize},
		{PhaseV3Prefetch, o.phasePrefetch},
		{PhaseV3PreCopy, o.phasePreCopy},
		{PhaseV3Converge, o.phaseConverge},
		{PhaseV3StopCopy, o.phaseStopCopy},
		{PhaseV3PostCopy, o.phasePostCopy},
		{PhaseV3Verify, o.phaseVerify},
		{PhaseV3Cleanup, o.phaseCleanup},
	}

	for _, phase := range phases {
		migration.CurrentPhase = phase.phase
		migration.PhaseStartTime = time.Now()

		if err := phase.fn(ctx, migration); err != nil {
			o.handleMigrationError(migration, err)
			o.metrics.FailedMigrations.Add(1)
			return
		}

		// Record phase duration
		migration.PhaseDurations[phase.phase] = time.Since(migration.PhaseStartTime)
	}

	migration.CurrentPhase = PhaseV3Complete
	migration.State.EndTime = time.Now()
	o.metrics.SuccessfulMigrations.Add(1)

	// Update average metrics
	downtime := migration.State.Downtime.Load()
	o.updateAverageDowntime(downtime)

	speedup := o.calculateSpeedup(migration)
	o.updateAverageSpeedup(speedup)
}

// phaseInitialize initializes the migration
func (o *DWCPv3Orchestrator) phaseInitialize(ctx context.Context, migration *DWCPv3Migration) error {
	// Initialize transport with AMST v3
	if o.amst != nil {
		conn, err := o.amst.EstablishConnection(ctx, migration.DestinationNode, migration.Mode)
		if err != nil {
			return fmt.Errorf("failed to establish AMST connection: %w", err)
		}

		// Store connection in migration context
		migration.Context = context.WithValue(migration.Context, "amst_conn", conn)
	}

	// Predict bandwidth with PBA v3
	if o.pba != nil {
		predicted := o.pba.PredictBandwidth(ctx, migration.SourceNode, migration.DestinationNode)
		migration.PredictedBandwidth = predicted
		o.metrics.PBAPredictions.Add(1)
	}

	// Determine compression algorithm with HDE v3
	if o.hde != nil {
		// Sample VM memory for compression analysis
		sample := o.sampleVMMemory(migration.VM)
		algo := o.hde.SelectCompression(sample, migration.Mode)
		migration.CompressionAlgo = algo
	}

	// Initialize adaptive sync with ASS v3
	if o.ass != nil {
		syncMode := o.ass.DetermineMode(migration.Mode)
		migration.Context = context.WithValue(migration.Context, "sync_mode", syncMode)
		o.metrics.ASSSyncs.Add(1)
	}

	return nil
}

// phasePrefetch performs predictive prefetching
func (o *DWCPv3Orchestrator) phasePrefetch(ctx context.Context, migration *DWCPv3Migration) error {
	if !o.config.EnablePrefetching || o.pba == nil {
		return nil // Skip prefetching if disabled
	}

	// Use PBA to identify pages likely to be accessed
	hotPages := o.pba.PredictHotPages(ctx, migration.VM.ID, 1000)

	// Prefetch hot pages to destination
	for _, pageID := range hotPages {
		pageData := o.getVMPage(migration.VM, pageID)

		// Compress if enabled
		if o.hde != nil && o.config.EnableCompression {
			compressed, err := o.hde.Compress(pageData, migration.CompressionAlgo)
			if err == nil {
				pageData = compressed
				migration.CompressedPages++
			}
		}

		// Transfer page
		if err := o.transferPage(ctx, migration, pageID, pageData); err != nil {
			// Non-fatal, continue with other pages
			continue
		}

		migration.PrefetchedPages++
	}

	return nil
}

// phasePreCopy performs iterative memory pre-copy
func (o *DWCPv3Orchestrator) phasePreCopy(ctx context.Context, migration *DWCPv3Migration) error {
	targets := o.getPerformanceTargets(migration.Mode)
	maxIterations := targets.MaxIterations

	for iteration := 0; iteration < maxIterations; iteration++ {
		// Get dirty pages
		dirtyPages := o.getDirtyPages(migration.VM)
		totalDirty := len(dirtyPages)

		// Check if mode switch is needed
		if o.config.AutoSwitchMode && iteration > 0 {
			newMode := o.evaluateModeSwitch(migration)
			if newMode != migration.Mode {
				o.switchMode(migration, newMode)
			}
		}

		// Use ITP v3 for intelligent partitioning
		if o.itp != nil && totalDirty > 1000 {
			partitions := o.itp.PartitionPages(dirtyPages, migration.Mode)
			migration.Context = context.WithValue(migration.Context, "partitions", partitions)
			o.metrics.ITPPartitions.Add(1)
		}

		// Transfer dirty pages
		transferred := 0
		startTime := time.Now()

		for _, pageID := range dirtyPages {
			pageData := o.getVMPage(migration.VM, pageID)

			// Compress based on mode
			if o.shouldCompress(migration.Mode) && o.hde != nil {
				compressed, err := o.hde.CompressWithMode(pageData, migration.Mode)
				if err == nil {
					originalSize := len(pageData)
					compressedSize := len(compressed)
					migration.CompressionRatio = float64(originalSize) / float64(compressedSize)
					pageData = compressed
					o.metrics.HDECompressions.Add(1)
				}
			}

			// Transfer using AMST v3
			if err := o.transferPageWithAMST(ctx, migration, pageID, pageData); err != nil {
				return fmt.Errorf("failed to transfer page %d: %w", pageID, err)
			}

			transferred++
		}

		// Calculate transfer rate
		duration := time.Since(startTime)
		transferRate := float64(transferred*4096) / duration.Seconds()
		migration.ActualBandwidth = int64(transferRate)

		// Update dirty page rate
		migration.DirtyPageRate = float64(totalDirty) / float64(migration.VM.MemoryMB*256) // pages per MB

		// Check convergence
		if o.checkConvergence(migration, iteration) {
			break
		}
	}

	return nil
}

// phaseConverge ensures memory convergence
func (o *DWCPv3Orchestrator) phaseConverge(ctx context.Context, migration *DWCPv3Migration) error {
	// Use ASS v3 for adaptive synchronization
	if o.ass != nil {
		return o.ass.Converge(ctx, migration.VM.ID, migration.DestinationNode)
	}

	// Fallback convergence logic
	remainingDirty := o.countDirtyPages(migration.VM)
	targets := o.getPerformanceTargets(migration.Mode)

	// Calculate expected downtime
	transferRate := migration.ActualBandwidth
	if transferRate == 0 {
		transferRate = migration.PredictedBandwidth
	}

	expectedDowntime := time.Duration(float64(remainingDirty*4096) / float64(transferRate) * float64(time.Second))

	// If expected downtime is acceptable, proceed
	if expectedDowntime <= targets.MaxDowntime {
		return nil
	}

	// Otherwise, try to reduce dirty pages further
	return o.reduceDirtyPages(ctx, migration)
}

// phaseStopCopy performs final synchronization with VM stopped
func (o *DWCPv3Orchestrator) phaseStopCopy(ctx context.Context, migration *DWCPv3Migration) error {
	downtimeStart := time.Now()

	// Pause VM
	if err := o.pauseVM(migration.VM); err != nil {
		return fmt.Errorf("failed to pause VM: %w", err)
	}

	// Get final dirty pages
	finalDirty := o.getDirtyPages(migration.VM)

	// Transfer with maximum priority and compression
	for _, pageID := range finalDirty {
		pageData := o.getVMPage(migration.VM, pageID)

		// Use aggressive compression for stop-copy
		if o.hde != nil {
			compressed, _ := o.hde.CompressUrgent(pageData)
			if compressed != nil {
				pageData = compressed
			}
		}

		// Transfer with high priority
		if err := o.transferUrgent(ctx, migration, pageID, pageData); err != nil {
			o.resumeVM(migration.VM) // Try to resume on failure
			return fmt.Errorf("urgent transfer failed: %w", err)
		}
	}

	// Transfer VM state
	if err := o.transferVMState(ctx, migration); err != nil {
		o.resumeVM(migration.VM)
		return fmt.Errorf("VM state transfer failed: %w", err)
	}

	// Resume VM on destination
	if err := o.resumeVMOnDestination(ctx, migration); err != nil {
		o.resumeVM(migration.VM)
		return fmt.Errorf("failed to resume on destination: %w", err)
	}

	// Record downtime
	downtime := time.Since(downtimeStart)
	migration.State.Downtime.Store(downtime.Milliseconds())

	return nil
}

// phasePostCopy handles post-copy if needed
func (o *DWCPv3Orchestrator) phasePostCopy(ctx context.Context, migration *DWCPv3Migration) error {
	// Post-copy is optional, mainly for internet mode
	if migration.Mode != upgrade.ModeInternet {
		return nil
	}

	// Transfer any remaining non-critical pages in background
	go o.transferRemainingPages(ctx, migration)

	return nil
}

// phaseVerify verifies migration success
func (o *DWCPv3Orchestrator) phaseVerify(ctx context.Context, migration *DWCPv3Migration) error {
	// Verify VM is running on destination
	if !o.isVMRunningOnDestination(migration) {
		return fmt.Errorf("VM not running on destination")
	}

	// Verify memory integrity if ASS v3 is enabled
	if o.ass != nil {
		if err := o.ass.VerifySync(ctx, migration.VM.ID); err != nil {
			return fmt.Errorf("synchronization verification failed: %w", err)
		}
	}

	// Verify network connectivity
	if err := o.verifyNetworkConnectivity(ctx, migration); err != nil {
		return fmt.Errorf("network verification failed: %w", err)
	}

	return nil
}

// phaseCleanup cleans up after migration
func (o *DWCPv3Orchestrator) phaseCleanup(ctx context.Context, migration *DWCPv3Migration) error {
	// Clean up source VM
	if err := o.cleanupSourceVM(migration.VM); err != nil {
		// Non-fatal
		fmt.Printf("Warning: source cleanup failed: %v\n", err)
	}

	// Close AMST connection
	if conn := migration.Context.Value("amst_conn"); conn != nil {
		if amstConn, ok := conn.(*transport.Connection); ok {
			amstConn.Close()
		}
	}

	// Clear component state
	if o.hde != nil {
		o.hde.ClearBaseline(migration.VM.ID)
	}

	if o.pba != nil {
		o.pba.ClearPredictions(migration.VM.ID)
	}

	return nil
}

// Helper methods

func (o *DWCPv3Orchestrator) shouldCompress(mode upgrade.NetworkMode) bool {
	switch mode {
	case upgrade.ModeDatacenter:
		return false // Minimal compression in datacenter
	case upgrade.ModeInternet:
		return true // Always compress for internet
	case upgrade.ModeHybrid:
		return o.config.EnableCompression
	default:
		return o.config.EnableCompression
	}
}

func (o *DWCPv3Orchestrator) getPerformanceTargets(mode upgrade.NetworkMode) *PerformanceTargets {
	switch mode {
	case upgrade.ModeDatacenter:
		return o.config.DatacenterTargets
	case upgrade.ModeInternet:
		return o.config.InternetTargets
	case upgrade.ModeHybrid:
		return o.config.HybridTargets
	default:
		return o.config.HybridTargets
	}
}

func (o *DWCPv3Orchestrator) transferPageWithAMST(ctx context.Context, migration *DWCPv3Migration, pageID int, data []byte) error {
	if o.amst == nil {
		return o.transferPage(ctx, migration, pageID, data)
	}

	// Use AMST for adaptive streaming
	conn := migration.Context.Value("amst_conn")
	if conn == nil {
		return fmt.Errorf("no AMST connection")
	}

	amstConn := conn.(*transport.Connection)

	// Create page header
	header := make([]byte, 8)
	binary.BigEndian.PutUint32(header[0:4], uint32(pageID))
	binary.BigEndian.PutUint32(header[4:8], uint32(len(data)))

	// Send header and data
	if _, err := amstConn.Write(header); err != nil {
		return err
	}

	if _, err := amstConn.Write(data); err != nil {
		return err
	}

	o.metrics.AMSTStreamsUsed.Add(1)
	return nil
}

func (o *DWCPv3Orchestrator) checkConvergence(migration *DWCPv3Migration, iteration int) bool {
	// Check if dirty page rate is converging
	if migration.DirtyPageRate < 0.01 { // Less than 1% dirty
		return true
	}

	// Check if we've reached iteration limit
	targets := o.getPerformanceTargets(migration.Mode)
	if iteration >= targets.MaxIterations-1 {
		return true
	}

	// Check predicted convergence time
	if o.pba != nil {
		convergenceTime := o.pba.PredictConvergence(migration.VM.ID, migration.DirtyPageRate)
		if convergenceTime < targets.MaxDowntime {
			return true
		}
	}

	return false
}

func (o *DWCPv3Orchestrator) evaluateModeSwitch(migration *DWCPv3Migration) upgrade.NetworkMode {
	// Re-evaluate network conditions
	newMode := o.determineNetworkMode(migration.SourceNode, migration.DestinationNode)

	// Only switch if significantly different
	if newMode != migration.Mode {
		// Check if switch would be beneficial
		currentTargets := o.getPerformanceTargets(migration.Mode)
		newTargets := o.getPerformanceTargets(newMode)

		// Switch if new mode has better characteristics for current state
		if migration.DirtyPageRate > 0.1 && newTargets.CompressionRatio > currentTargets.CompressionRatio {
			return newMode
		}

		if migration.ActualBandwidth < currentTargets.TargetThroughput/2 &&
		   newTargets.TargetThroughput < currentTargets.TargetThroughput {
			return newMode
		}
	}

	return migration.Mode
}

func (o *DWCPv3Orchestrator) switchMode(migration *DWCPv3Migration, newMode upgrade.NetworkMode) {
	fmt.Printf("Switching migration mode from %s to %s\n", migration.Mode, newMode)

	oldMode := migration.Mode
	migration.Mode = newMode
	migration.ModeSwitches++
	o.metrics.ModeSwitches.Add(1)

	// Reconfigure components
	if o.amst != nil {
		o.amst.SwitchMode(newMode)
	}

	if o.hde != nil {
		o.hde.SetMode(newMode)
	}

	// Log mode switch
	o.monitor.LogEvent(migration.ID, fmt.Sprintf("Mode switched: %s -> %s", oldMode, newMode))
}

func (o *DWCPv3Orchestrator) calculateSpeedup(migration *DWCPv3Migration) float64 {
	// Calculate speedup vs baseline
	baselineRate := int64(100 * 1024 * 1024) // 100 MB/s baseline

	if migration.ActualBandwidth > 0 {
		return float64(migration.ActualBandwidth) / float64(baselineRate)
	}

	// Estimate from duration
	totalBytes := migration.State.TotalBytes.Load()
	duration := migration.State.EndTime.Sub(migration.State.StartTime)
	actualRate := float64(totalBytes) / duration.Seconds()

	return actualRate / float64(baselineRate)
}

func (o *DWCPv3Orchestrator) updateAverageDowntime(downtime int64) {
	current := o.metrics.AverageDowntime.Load()
	count := o.metrics.SuccessfulMigrations.Load()

	if count == 0 {
		o.metrics.AverageDowntime.Store(downtime)
	} else {
		// Calculate running average
		newAvg := (current*(count-1) + downtime) / count
		o.metrics.AverageDowntime.Store(newAvg)
	}
}

func (o *DWCPv3Orchestrator) updateAverageSpeedup(speedup float64) {
	speedupInt := int64(speedup * 100) // Store as percentage * 100
	current := o.metrics.AverageSpeedup.Load()
	count := o.metrics.SuccessfulMigrations.Load()

	if count == 0 {
		o.metrics.AverageSpeedup.Store(speedupInt)
	} else {
		// Calculate running average
		newAvg := (current*(count-1) + speedupInt) / count
		o.metrics.AverageSpeedup.Store(newAvg)
	}
}

func (o *DWCPv3Orchestrator) monitorLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			o.updateMetrics()
			o.checkMigrationHealth()

		case <-o.ctx.Done():
			return
		}
	}
}

func (o *DWCPv3Orchestrator) updateMetrics() {
	// Update component metrics
	if o.hde != nil {
		savings := o.hde.GetCompressionSavings()
		o.metrics.CompressionSavings.Store(savings)
	}
}

func (o *DWCPv3Orchestrator) checkMigrationHealth() {
	o.migMu.RLock()
	defer o.migMu.RUnlock()

	for _, migration := range o.migrations {
		// Check for stuck migrations
		if time.Since(migration.PhaseStartTime) > o.config.MigrationTimeout {
			o.handleMigrationTimeout(migration)
		}

		// Check for network issues
		if migration.ActualBandwidth < migration.PredictedBandwidth/10 {
			o.handleBandwidthDegradation(migration)
		}
	}
}

func (o *DWCPv3Orchestrator) handleMigrationTimeout(migration *DWCPv3Migration) {
	fmt.Printf("Migration %s timeout in phase %s\n", migration.ID, migration.CurrentPhase)
	// In production, would trigger recovery or rollback
}

func (o *DWCPv3Orchestrator) handleBandwidthDegradation(migration *DWCPv3Migration) {
	fmt.Printf("Migration %s bandwidth degradation: %d < %d\n",
		migration.ID, migration.ActualBandwidth, migration.PredictedBandwidth)
	// Could trigger mode switch or parameter adjustment
}

func (o *DWCPv3Orchestrator) cleanupMigration(migration *DWCPv3Migration) {
	o.migMu.Lock()
	delete(o.migrations, migration.ID)
	o.migMu.Unlock()
}

func (o *DWCPv3Orchestrator) handleMigrationError(migration *DWCPv3Migration, err error) {
	migration.State.Error = err
	migration.State.Phase = PhaseFailed
	o.monitor.LogEvent(migration.ID, fmt.Sprintf("Migration failed: %v", err))
}

// Placeholder methods for VM operations (would be implemented with actual hypervisor API)

func (o *DWCPv3Orchestrator) sampleVMMemory(vm *VM) []byte {
	// Sample 1MB of VM memory for compression analysis
	return make([]byte, 1024*1024)
}

func (o *DWCPv3Orchestrator) getVMPage(vm *VM, pageID int) []byte {
	// Get specific memory page from VM
	return make([]byte, 4096)
}

func (o *DWCPv3Orchestrator) getDirtyPages(vm *VM) []int {
	// Get list of dirty page IDs
	// In production, would query hypervisor
	numPages := vm.MemoryMB * 256 // 256 pages per MB
	dirtyCount := numPages / 10    // Simulate 10% dirty

	pages := make([]int, dirtyCount)
	for i := 0; i < dirtyCount; i++ {
		pages[i] = i * 10 // Every 10th page is dirty
	}
	return pages
}

func (o *DWCPv3Orchestrator) countDirtyPages(vm *VM) int {
	return len(o.getDirtyPages(vm))
}

func (o *DWCPv3Orchestrator) transferPage(ctx context.Context, migration *DWCPv3Migration, pageID int, data []byte) error {
	// Basic page transfer
	migration.State.BytesTransferred.Add(int64(len(data)))
	return nil
}

func (o *DWCPv3Orchestrator) transferUrgent(ctx context.Context, migration *DWCPv3Migration, pageID int, data []byte) error {
	// High-priority transfer
	return o.transferPage(ctx, migration, pageID, data)
}

func (o *DWCPv3Orchestrator) reduceDirtyPages(ctx context.Context, migration *DWCPv3Migration) error {
	// Try to reduce dirty pages (e.g., by throttling VM)
	return nil
}

func (o *DWCPv3Orchestrator) transferVMState(ctx context.Context, migration *DWCPv3Migration) error {
	// Transfer CPU, device state, etc.
	return nil
}

func (o *DWCPv3Orchestrator) transferRemainingPages(ctx context.Context, migration *DWCPv3Migration) {
	// Background transfer of remaining pages
}

func (o *DWCPv3Orchestrator) isVMRunningOnDestination(migration *DWCPv3Migration) bool {
	// Check if VM is running on destination
	return true // Placeholder
}

func (o *DWCPv3Orchestrator) verifyNetworkConnectivity(ctx context.Context, migration *DWCPv3Migration) error {
	// Verify network connectivity
	return nil
}

func (o *DWCPv3Orchestrator) resumeVMOnDestination(ctx context.Context, migration *DWCPv3Migration) error {
	// Resume VM on destination node
	return nil
}

func (o *DWCPv3Orchestrator) cleanupSourceVM(vm *VM) error {
	// Clean up source VM
	return nil
}

// GetMetrics returns current metrics
func (o *DWCPv3Orchestrator) GetMetrics() *DWCPv3Metrics {
	return o.metrics
}

// GetMigration returns a specific migration
func (o *DWCPv3Orchestrator) GetMigration(id string) (*DWCPv3Migration, bool) {
	o.migMu.RLock()
	defer o.migMu.RUnlock()

	migration, ok := o.migrations[id]
	return migration, ok
}

// ListMigrations returns all active migrations
func (o *DWCPv3Orchestrator) ListMigrations() []*DWCPv3Migration {
	o.migMu.RLock()
	defer o.migMu.RUnlock()

	migrations := make([]*DWCPv3Migration, 0, len(o.migrations))
	for _, m := range o.migrations {
		migrations = append(migrations, m)
	}

	return migrations
}

// Close shuts down the orchestrator
func (o *DWCPv3Orchestrator) Close() error {
	o.cancel()

	// Close components
	if o.amst != nil {
		o.amst.Close()
	}
	if o.hde != nil {
		o.hde.Close()
	}
	if o.pba != nil {
		o.pba.Close()
	}
	if o.itp != nil {
		o.itp.Close()
	}
	if o.ass != nil {
		o.ass.Close()
	}

	// Close base orchestrator
	return o.LiveMigrationOrchestrator.Close()
}