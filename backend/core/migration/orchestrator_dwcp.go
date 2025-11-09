// Package migration handles VM migration with DWCP optimization
package migration

import (
	"context"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// DWCPConfig extends MigrationConfig with DWCP-specific settings
type DWCPConfig struct {
	// Enable DWCP optimization
	EnableDWCP bool `json:"enable_dwcp"`

	// Fallback to standard migration if DWCP fails
	EnableFallback bool `json:"enable_fallback"`

	// AMST settings
	MinStreams      int `json:"min_streams"`       // Min parallel streams (default: 4)
	MaxStreams      int `json:"max_streams"`       // Max parallel streams (default: 256)
	InitialStreams  int `json:"initial_streams"`   // Initial streams (default: 16)

	// HDE settings
	EnableDelta     bool    `json:"enable_delta"`      // Enable delta encoding (default: true)
	DeltaThreshold  float64 `json:"delta_threshold"`   // Delta efficiency threshold (default: 0.7)
	EnableDictionary bool   `json:"enable_dictionary"` // Enable dictionary compression (default: true)

	// Performance targets
	TargetSpeedup   float64 `json:"target_speedup"`    // Target speedup over baseline (default: 2.5x)
}

// EnhancedLiveMigrationOrchestrator extends the base orchestrator with DWCP capabilities
type EnhancedLiveMigrationOrchestrator struct {
	*LiveMigrationOrchestrator

	// DWCP components
	dwcpAdapter *dwcp.MigrationAdapter
	dwcpConfig  DWCPConfig

	// Performance tracking
	dwcpMigrations   int64
	dwcpSpeedup      float64
	dwcpSuccessRate  float64
}

// NewEnhancedLiveMigrationOrchestrator creates a new orchestrator with DWCP support
func NewEnhancedLiveMigrationOrchestrator(baseConfig MigrationConfig, dwcpConfig DWCPConfig) (*EnhancedLiveMigrationOrchestrator, error) {
	// Create base orchestrator
	baseOrchestrator, err := NewLiveMigrationOrchestrator(baseConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create base orchestrator: %w", err)
	}

	// Set DWCP defaults
	if dwcpConfig.MinStreams <= 0 {
		dwcpConfig.MinStreams = 4
	}
	if dwcpConfig.MaxStreams <= 0 {
		dwcpConfig.MaxStreams = 256
	}
	if dwcpConfig.InitialStreams <= 0 {
		dwcpConfig.InitialStreams = 16
	}
	if dwcpConfig.DeltaThreshold <= 0 {
		dwcpConfig.DeltaThreshold = 0.7
	}
	if dwcpConfig.TargetSpeedup <= 0 {
		dwcpConfig.TargetSpeedup = 2.5
	}

	enhanced := &EnhancedLiveMigrationOrchestrator{
		LiveMigrationOrchestrator: baseOrchestrator,
		dwcpConfig:                 dwcpConfig,
	}

	// Create DWCP adapter if enabled
	if dwcpConfig.EnableDWCP {
		adapterConfig := dwcp.MigrationAdapterConfig{
			EnableDWCP:     true,
			EnableFallback: dwcpConfig.EnableFallback,
			AMSTConfig: dwcp.AMSTConfig{
				MinStreams:      dwcpConfig.MinStreams,
				MaxStreams:      dwcpConfig.MaxStreams,
				InitialStreams:  dwcpConfig.InitialStreams,
				EnableAdaptive:  true,
				BandwidthLimit:  baseConfig.BandwidthLimit,
				EnablePacing:    true,
			},
			HDEConfig: dwcp.HDEConfig{
				EnableDelta:      dwcpConfig.EnableDelta,
				DeltaThreshold:   dwcpConfig.DeltaThreshold,
				EnableDictionary: dwcpConfig.EnableDictionary,
				LocalLevel:       0,  // Fast compression for local
				RegionalLevel:    3,  // Balanced for regional
				GlobalLevel:      9,  // Best compression for WAN
			},
			TargetSpeedup: dwcpConfig.TargetSpeedup,
		}

		adapter, err := dwcp.NewMigrationAdapter(adapterConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create DWCP adapter: %w", err)
		}
		enhanced.dwcpAdapter = adapter

		// Start DWCP listener for incoming migrations
		go adapter.ListenForMigrations(context.Background())
	}

	return enhanced, nil
}

// copyMemoryIterativeWithDWCP performs memory copy with DWCP optimization
func (o *EnhancedLiveMigrationOrchestrator) copyMemoryIterativeWithDWCP(ctx context.Context, migration *LiveMigration) error {
	if !o.dwcpConfig.EnableDWCP || o.dwcpAdapter == nil {
		// Fall back to standard memory copy
		return o.copyMemoryIterative(ctx, migration)
	}

	state := migration.State
	maxIterations := o.config.MemoryIterations

	// Get initial memory size (placeholder - would get from VM)
	totalMemory := int64(4 * 1024 * 1024 * 1024) // 4GB placeholder
	state.TotalBytes.Store(totalMemory)

	// Simulate getting memory data (in production, would get from hypervisor)
	memoryData := make([]byte, totalMemory)

	fmt.Printf("DWCP: Starting optimized memory migration for VM %s\n", migration.VM.ID)
	startTime := time.Now()

	// Use DWCP for memory transfer
	err := o.dwcpAdapter.MigrateVMMemory(ctx, migration.VM.ID, memoryData, migration.DestinationNode,
		func(transferred int64) {
			// Update progress
			state.BytesTransferred.Store(transferred)
			progress := float64(transferred) / float64(totalMemory)
			state.Progress.Store(progress * 0.7) // Memory is 70% of migration

			// Update monitor
			o.monitor.UpdateProgress(migration.ID, MigrationProgress{
				Phase:            PhaseMemoryCopy,
				BytesTransferred: transferred,
				TotalBytes:       totalMemory,
				OverallProgress:  progress * 70,
			})
		})

	if err != nil {
		// If DWCP fails and fallback is enabled, try standard migration
		if o.dwcpConfig.EnableFallback {
			fmt.Printf("DWCP failed, falling back to standard migration: %v\n", err)
			return o.copyMemoryIterative(ctx, migration)
		}
		return fmt.Errorf("DWCP memory migration failed: %w", err)
	}

	duration := time.Since(startTime)
	transferRate := float64(totalMemory) / duration.Seconds()
	state.TransferRate.Store(int64(transferRate))

	// Calculate speedup
	baselineRate := float64(20 * 1024 * 1024) // 20 MB/s baseline
	speedup := transferRate / baselineRate
	o.dwcpSpeedup = speedup
	o.dwcpMigrations++

	fmt.Printf("DWCP: Memory migration completed in %.2fs (%.2f MB/s, %.2fx speedup)\n",
		duration.Seconds(), transferRate/1024/1024, speedup)

	return nil
}

// syncDiskWithDWCP synchronizes disk state with DWCP optimization
func (o *EnhancedLiveMigrationOrchestrator) syncDiskWithDWCP(ctx context.Context, migration *LiveMigration) error {
	if !o.dwcpConfig.EnableDWCP || o.dwcpAdapter == nil {
		// Fall back to standard disk sync
		return o.syncDisk(ctx, migration)
	}

	// In production, would get actual disk blocks from storage system
	diskSize := int64(10 * 1024 * 1024 * 1024) // 10GB placeholder
	blockSize := 1024 * 1024                    // 1MB blocks
	numBlocks := int(diskSize / int64(blockSize))

	// Simulate disk blocks (in production, would read from storage)
	diskBlocks := make(map[int][]byte)
	for i := 0; i < numBlocks; i++ {
		diskBlocks[i] = make([]byte, blockSize)
	}

	fmt.Printf("DWCP: Starting optimized disk migration for VM %s\n", migration.VM.ID)
	startTime := time.Now()

	// Use DWCP for disk transfer
	err := o.dwcpAdapter.MigrateVMDisk(ctx, migration.VM.ID, diskBlocks, migration.DestinationNode,
		func(transferred int64) {
			// Update progress
			progress := 0.7 + (float64(transferred)/float64(diskSize))*0.2 // Disk is 20% of migration
			migration.State.Progress.Store(progress)

			// Update monitor
			o.monitor.UpdateProgress(migration.ID, MigrationProgress{
				Phase:           PhaseDiskCopy,
				OverallProgress: progress * 100,
			})
		})

	if err != nil {
		// If DWCP fails and fallback is enabled, try standard migration
		if o.dwcpConfig.EnableFallback {
			fmt.Printf("DWCP disk sync failed, falling back to standard: %v\n", err)
			return o.syncDisk(ctx, migration)
		}
		return fmt.Errorf("DWCP disk migration failed: %w", err)
	}

	duration := time.Since(startTime)
	transferRate := float64(diskSize) / duration.Seconds()

	fmt.Printf("DWCP: Disk migration completed in %.2fs (%.2f MB/s)\n",
		duration.Seconds(), transferRate/1024/1024)

	return nil
}

// executeLiveMigrationWithDWCP performs live migration with DWCP optimization
func (o *EnhancedLiveMigrationOrchestrator) executeLiveMigrationWithDWCP(ctx context.Context, migration *LiveMigration) error {
	state := migration.State

	// Phase 1: Initial memory copy with DWCP
	state.Phase = PhaseMemoryCopy
	if err := o.copyMemoryIterativeWithDWCP(ctx, migration); err != nil {
		return fmt.Errorf("DWCP memory copy failed: %w", err)
	}

	// Phase 2: Disk synchronization with DWCP
	state.Phase = PhaseDiskCopy
	if err := o.syncDiskWithDWCP(ctx, migration); err != nil {
		return fmt.Errorf("DWCP disk sync failed: %w", err)
	}

	// Phase 3: Final synchronization with brief downtime
	state.Phase = PhaseDowntime
	downtimeStart := time.Now()

	// Pause VM on source
	if err := o.pauseVM(migration.VM); err != nil {
		return fmt.Errorf("failed to pause VM: %w", err)
	}

	// Transfer final dirty pages with DWCP (high priority)
	if err := o.transferFinalStateWithDWCP(ctx, migration); err != nil {
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

	// Clean up DWCP connection
	if o.dwcpAdapter != nil {
		o.dwcpAdapter.CleanupConnection(migration.VM.ID, migration.DestinationNode)
	}

	return nil
}

// transferFinalStateWithDWCP transfers the final VM state with DWCP during downtime
func (o *EnhancedLiveMigrationOrchestrator) transferFinalStateWithDWCP(ctx context.Context, migration *LiveMigration) error {
	if !o.dwcpConfig.EnableDWCP || o.dwcpAdapter == nil {
		// Fall back to standard transfer
		return o.transferFinalState(ctx, migration)
	}

	// This happens during downtime, so it must be fast
	state := migration.State

	// Get remaining dirty pages
	dirtyPages := state.DirtyPages.Load()
	finalBytes := dirtyPages * 4096

	// Create final state data (placeholder - would get from hypervisor)
	finalData := make([]byte, finalBytes)

	// Use DWCP with highest priority for minimal downtime
	err := o.dwcpAdapter.MigrateVMMemory(ctx, migration.VM.ID+"_final", finalData,
		migration.DestinationNode, nil)

	if err != nil && o.dwcpConfig.EnableFallback {
		// Quick fallback to standard if DWCP fails
		return o.transferFinalState(ctx, migration)
	}

	return err
}

// TrainDWCPDictionary trains compression dictionaries for specific VM types
func (o *EnhancedLiveMigrationOrchestrator) TrainDWCPDictionary(vmType string, samples [][]byte) error {
	if o.dwcpAdapter == nil {
		return fmt.Errorf("DWCP not enabled")
	}

	return o.dwcpAdapter.TrainDictionary(vmType, samples)
}

// GetDWCPMetrics returns DWCP-specific metrics
func (o *EnhancedLiveMigrationOrchestrator) GetDWCPMetrics() map[string]interface{} {
	metrics := map[string]interface{}{
		"dwcp_enabled":     o.dwcpConfig.EnableDWCP,
		"dwcp_migrations":  o.dwcpMigrations,
		"dwcp_speedup":     o.dwcpSpeedup,
		"dwcp_success_rate": o.dwcpSuccessRate,
	}

	if o.dwcpAdapter != nil {
		metrics["adapter_metrics"] = o.dwcpAdapter.GetMetrics()
	}

	return metrics
}

// Close shuts down the enhanced orchestrator
func (o *EnhancedLiveMigrationOrchestrator) Close() error {
	// Close DWCP adapter first
	if o.dwcpAdapter != nil {
		if err := o.dwcpAdapter.Close(); err != nil {
			fmt.Printf("Warning: failed to close DWCP adapter: %v\n", err)
		}
	}

	// Close base orchestrator
	return o.LiveMigrationOrchestrator.Close()
}