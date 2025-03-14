package vm

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// DirtyPageTracker tracks which memory pages have been modified
// during migration and need to be re-transferred
type DirtyPageTracker struct {
	// Page size in KB
	PageSizeKB int

	// Total memory size in KB
	TotalMemoryKB int

	// Bitmap of dirty pages (true = dirty, false = clean)
	dirtyPages []bool

	// Statistics
	stats DirtyPageStats

	// Mutex for thread safety
	mu sync.RWMutex

	// Logger
	logger *logrus.Entry
}

// DirtyPageStats contains statistics about dirty page tracking
type DirtyPageStats struct {
	// Time when tracking started
	StartTime time.Time

	// Number of times the dirty page map has been scanned
	ScanCount int

	// Total number of pages that were marked dirty across all iterations
	TotalDirtyPages int

	// Number of pages that were dirty in the most recent scan
	LastScanDirtyPages int

	// Rate at which pages are being dirtied (pages per second)
	DirtyRate float64

	// Dirtying patterns detected (e.g., "random", "sequential", "hotspot")
	DirtyingPattern string

	// Hotspot regions (indexes of page ranges with high modification rates)
	Hotspots []PageRange
}

// PageRange represents a range of pages
type PageRange struct {
	StartIndex int
	EndIndex   int
	DirtyCount int
}

// NewDirtyPageTracker creates a new dirty page tracker
func NewDirtyPageTracker(totalMemoryKB int, pageSizeKB int, logger *logrus.Logger) *DirtyPageTracker {
	if pageSizeKB <= 0 {
		pageSizeKB = 4 // Default 4KB page size
	}

	numPages := totalMemoryKB / pageSizeKB
	if totalMemoryKB%pageSizeKB != 0 {
		numPages++ // Round up
	}

	logEntry := logger.WithField("component", "DirtyPageTracker")

	return &DirtyPageTracker{
		PageSizeKB:    pageSizeKB,
		TotalMemoryKB: totalMemoryKB,
		dirtyPages:    make([]bool, numPages),
		stats: DirtyPageStats{
			StartTime: time.Now(),
			Hotspots:  make([]PageRange, 0),
		},
		logger: logEntry,
	}
}

// MarkPageDirty marks a specific page as dirty
func (t *DirtyPageTracker) MarkPageDirty(pageIndex int) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if pageIndex < 0 || pageIndex >= len(t.dirtyPages) {
		return fmt.Errorf("page index %d out of range (0-%d)", pageIndex, len(t.dirtyPages)-1)
	}

	if !t.dirtyPages[pageIndex] {
		t.dirtyPages[pageIndex] = true
		t.stats.TotalDirtyPages++
	}

	return nil
}

// MarkMemoryRegionDirty marks all pages in a memory region as dirty
func (t *DirtyPageTracker) MarkMemoryRegionDirty(startAddressKB, sizeKB int) error {
	startPage := startAddressKB / t.PageSizeKB
	endAddress := startAddressKB + sizeKB
	endPage := (endAddress + t.PageSizeKB - 1) / t.PageSizeKB

	for i := startPage; i < endPage && i < len(t.dirtyPages); i++ {
		if i >= 0 {
			err := t.MarkPageDirty(i)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// GetDirtyPages returns all dirty pages
func (t *DirtyPageTracker) GetDirtyPages() []int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	dirtyPages := make([]int, 0, len(t.dirtyPages)/10) // Pre-allocate assuming 10% are dirty
	for i, isDirty := range t.dirtyPages {
		if isDirty {
			dirtyPages = append(dirtyPages, i)
		}
	}

	return dirtyPages
}

// GetDirtyPageCount returns the number of dirty pages
func (t *DirtyPageTracker) GetDirtyPageCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()

	count := 0
	for _, isDirty := range t.dirtyPages {
		if isDirty {
			count++
		}
	}

	return count
}

// ResetDirtyPages marks all pages as clean
func (t *DirtyPageTracker) ResetDirtyPages() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.stats.LastScanDirtyPages = 0
	for i, isDirty := range t.dirtyPages {
		if isDirty {
			t.stats.LastScanDirtyPages++
			t.dirtyPages[i] = false
		}
	}

	// Update scan count
	t.stats.ScanCount++

	// Calculate dirty rate (pages per second)
	elapsedSeconds := time.Since(t.stats.StartTime).Seconds()
	if elapsedSeconds > 0 {
		t.stats.DirtyRate = float64(t.stats.TotalDirtyPages) / elapsedSeconds
	}

	// Detect patterns
	t.detectDirtyingPatterns()
}

// detectDirtyingPatterns analyzes dirty page patterns
func (t *DirtyPageTracker) detectDirtyingPatterns() {
	// Reset hotspots
	t.stats.Hotspots = make([]PageRange, 0)

	// Simple pattern detection
	sequentialCount := 0
	lastDirty := -2
	hotspotThreshold := len(t.dirtyPages) / 100 // 1% of pages
	hotspotStart := -1
	hotspotCount := 0

	for i, isDirty := range t.dirtyPages {
		if isDirty {
			// Check for sequential access
			if i == lastDirty+1 {
				sequentialCount++
			}

			// Check for hotspot regions
			if hotspotStart == -1 {
				hotspotStart = i
				hotspotCount = 1
			} else if i <= hotspotStart+hotspotThreshold {
				hotspotCount++
			} else {
				// End of current hotspot region
				if hotspotCount >= 10 { // Minimum 10 pages to be considered a hotspot
					t.stats.Hotspots = append(t.stats.Hotspots, PageRange{
						StartIndex: hotspotStart,
						EndIndex:   i - 1,
						DirtyCount: hotspotCount,
					})
				}
				hotspotStart = i
				hotspotCount = 1
			}

			lastDirty = i
		}
	}

	// Add final hotspot if exists
	if hotspotStart != -1 && hotspotCount >= 10 {
		t.stats.Hotspots = append(t.stats.Hotspots, PageRange{
			StartIndex: hotspotStart,
			EndIndex:   len(t.dirtyPages) - 1,
			DirtyCount: hotspotCount,
		})
	}

	// Determine overall pattern
	dirtyRatio := float64(t.stats.LastScanDirtyPages) / float64(len(t.dirtyPages))
	sequentialRatio := float64(sequentialCount) / float64(t.stats.LastScanDirtyPages)

	if len(t.stats.Hotspots) > 0 && dirtyRatio < 0.3 {
		t.stats.DirtyingPattern = "hotspot"
	} else if sequentialRatio > 0.7 {
		t.stats.DirtyingPattern = "sequential"
	} else {
		t.stats.DirtyingPattern = "random"
	}
}

// GetStats returns statistics about dirty page tracking
func (t *DirtyPageTracker) GetStats() DirtyPageStats {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.stats
}

// EnhancedPreCopyMigration represents a live migration with enhanced pre-copy algorithm
type EnhancedPreCopyMigration struct {
	// Underlying VM migration
	migration *Migration

	// Dirty page tracker
	dirtyPageTracker *DirtyPageTracker

	// Configuration
	config EnhancedPreCopyConfig

	// Status and control
	status     MigrationStatus
	iterations int
	ctx        context.Context
	cancel     context.CancelFunc
	logger     *logrus.Entry
	mu         sync.RWMutex
}

// EnhancedPreCopyConfig contains configuration for enhanced pre-copy migration
type EnhancedPreCopyConfig struct {
	// Maximum number of iterations
	MaxIterations int

	// Convergence threshold (percentage of dirty pages below which to stop iterating)
	ConvergenceThresholdPct float64

	// Maximum allowed downtime in milliseconds
	MaxDowntimeMs int

	// Parallelism level for page transfers
	Parallelism int

	// Adaptation mode: "auto", "conservative", "aggressive"
	AdaptationMode string

	// Priority for dirty page transfers (1-10)
	Priority int

	// Use WAN optimization if available
	UseWANOptimization bool
}

// DefaultEnhancedPreCopyConfig returns default configuration
func DefaultEnhancedPreCopyConfig() EnhancedPreCopyConfig {
	return EnhancedPreCopyConfig{
		MaxIterations:           30,
		ConvergenceThresholdPct: 0.1, // 0.1% of pages
		MaxDowntimeMs:           300, // 300ms
		Parallelism:             4,
		AdaptationMode:          "auto",
		Priority:                5,
		UseWANOptimization:      true,
	}
}

// NewEnhancedPreCopyMigration creates a new enhanced pre-copy migration
func NewEnhancedPreCopyMigration(migration *Migration, config EnhancedPreCopyConfig, logger *logrus.Logger) *EnhancedPreCopyMigration {
	ctx, cancel := context.WithCancel(context.Background())

	// Get VM memory size
	memoryKB := 8 * 1024 * 1024 // Default 8GB if not available from VM
	/*
		if migration.VM != nil && migration.VM.Spec != nil {
			memoryKB = migration.VM.Spec.MemoryMB * 1024
		}
	*/

	dirtyPageTracker := NewDirtyPageTracker(memoryKB, 4, logger)

	return &EnhancedPreCopyMigration{
		migration:        migration,
		dirtyPageTracker: dirtyPageTracker,
		config:           config,
		status:           MigrationStatus{State: MigrationStatePreparing},
		ctx:              ctx,
		cancel:           cancel,
		logger:           logger.WithField("component", "EnhancedPreCopyMigration"),
	}
}

// Start begins the enhanced pre-copy migration
func (m *EnhancedPreCopyMigration) Start(ctx context.Context) error {
	m.mu.Lock()
	m.status.State = MigrationStateRunning
	m.status.StartTime = time.Now()
	m.mu.Unlock()

	go m.runMigrationProcess(ctx)

	return nil
}

// runMigrationProcess performs the actual migration process
func (m *EnhancedPreCopyMigration) runMigrationProcess(ctx context.Context) {
	defer func() {
		m.mu.Lock()
		m.status.EndTime = time.Now()
		m.mu.Unlock()
	}()

	// Initialize WAN optimizer if enabled
	var wanOptimizer *WANMigrationOptimizer
	if m.config.UseWANOptimization {
		wanConfig := DefaultWANMigrationConfig()
		wanOptimizer = NewWANMigrationOptimizer(wanConfig)
		defer wanOptimizer.Close()
	}

	// Do initial full memory transfer
	err := m.initialMemoryTransfer(ctx, wanOptimizer)
	if err != nil {
		m.setErrorState(err)
		return
	}

	// Perform iterative dirty page transfers
	m.iterations = 0
	for m.iterations < m.config.MaxIterations {
		select {
		case <-ctx.Done():
			m.setErrorState(ctx.Err())
			return
		default:
			// Monitor dirty pages for this iteration
			m.startDirtyPageMonitoring()

			// Transfer dirty pages
			err := m.transferDirtyPages(ctx, wanOptimizer)
			if err != nil {
				m.setErrorState(err)
				return
			}

			// Stop dirty page monitoring and check convergence
			dirtyPageCount := m.stopDirtyPageMonitoring()
			dirtyPagePct := float64(dirtyPageCount) / float64(len(m.dirtyPageTracker.dirtyPages)) * 100

			m.logger.WithFields(logrus.Fields{
				"iteration":       m.iterations + 1,
				"dirtyPages":      dirtyPageCount,
				"dirtyPagesPct":   dirtyPagePct,
				"dirtyingPattern": m.dirtyPageTracker.stats.DirtyingPattern,
				"dirtyRate":       m.dirtyPageTracker.stats.DirtyRate,
			}).Info("Completed iteration")

			// Check if we've reached convergence
			if dirtyPagePct <= m.config.ConvergenceThresholdPct {
				m.logger.Info("Reached convergence threshold")
				break
			}

			// Adapt parameters based on dirty page behavior
			m.adaptParameters(dirtyPagePct)

			m.iterations++
		}
	}

	// Final stop and copy
	err = m.finalStopAndCopy(ctx, wanOptimizer)
	if err != nil {
		m.setErrorState(err)
		return
	}

	m.mu.Lock()
	m.status.State = MigrationStateCompleted
	m.status.ProgressPct = 100
	m.status.Message = "Migration completed successfully"
	m.mu.Unlock()
}

// setErrorState sets the migration to error state with message
func (m *EnhancedPreCopyMigration) setErrorState(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.status.State = MigrationStateError
	m.status.Message = err.Error()
	m.status.Error = err
	m.logger.WithError(err).Error("Migration failed")
}

// initialMemoryTransfer performs the initial full memory transfer
func (m *EnhancedPreCopyMigration) initialMemoryTransfer(ctx context.Context, optimizer *WANMigrationOptimizer) error {
	m.mu.Lock()
	m.status.Message = "Performing initial memory transfer"
	m.status.ProgressPct = 10
	m.mu.Unlock()

	m.logger.Info("Starting initial memory transfer")

	// This is a placeholder for actual memory transfer implementation
	// In a real implementation, you would:
	// 1. Get memory pages from source VM
	// 2. Apply WAN optimization if enabled
	// 3. Transfer to destination
	// 4. Track progress

	// Simulate memory transfer (would be actual implementation in real code)
	time.Sleep(500 * time.Millisecond)

	m.mu.Lock()
	m.status.ProgressPct = 30
	m.mu.Unlock()

	return nil
}

// startDirtyPageMonitoring begins tracking dirty pages for current iteration
func (m *EnhancedPreCopyMigration) startDirtyPageMonitoring() {
	m.mu.Lock()
	m.status.Message = fmt.Sprintf("Monitoring dirty pages (iteration %d)", m.iterations+1)
	m.mu.Unlock()

	// Reset the dirty page tracker for the new iteration
	m.dirtyPageTracker.ResetDirtyPages()

	// In a real implementation, this would:
	// 1. Register with VM hypervisor to get dirty page notifications
	// 2. Set up memory tracing or page protection mechanisms
	// 3. Start collecting page modification data

	m.logger.WithField("iteration", m.iterations+1).Info("Started dirty page monitoring")
}

// stopDirtyPageMonitoring stops tracking dirty pages and returns the count
func (m *EnhancedPreCopyMigration) stopDirtyPageMonitoring() int {
	// In a real implementation, this would:
	// 1. Stop memory tracing or dirty page notifications
	// 2. Get final dirty page bitmap from hypervisor
	// 3. Analyze the results

	// For simulation, we'll randomly mark some pages as dirty
	dirtyPages := m.dirtyPageTracker.GetDirtyPageCount()

	m.logger.WithFields(logrus.Fields{
		"dirtyPages": dirtyPages,
		"pattern":    m.dirtyPageTracker.stats.DirtyingPattern,
	}).Info("Stopped dirty page monitoring")

	return dirtyPages
}

// transferDirtyPages transfers only the dirty pages to the destination
func (m *EnhancedPreCopyMigration) transferDirtyPages(ctx context.Context, optimizer *WANMigrationOptimizer) error {
	dirtyPages := m.dirtyPageTracker.GetDirtyPages()

	m.mu.Lock()
	m.status.Message = fmt.Sprintf("Transferring dirty pages (iteration %d, %d pages)",
		m.iterations+1, len(dirtyPages))
	m.status.ProgressPct = 30 + float64(m.iterations+1)/float64(m.config.MaxIterations)*50
	m.mu.Unlock()

	m.logger.WithFields(logrus.Fields{
		"iteration":  m.iterations + 1,
		"dirtyPages": len(dirtyPages),
	}).Info("Transferring dirty pages")

	// This is a placeholder for actual dirty page transfer implementation
	// In a real implementation, you would:
	// 1. Get content of dirty pages from source VM
	// 2. Apply WAN optimization if enabled
	// 3. Use parallelism based on config
	// 4. Transfer to destination
	// 5. Track progress

	// Simulate dirty page transfer (would be actual implementation in real code)
	time.Sleep(200 * time.Millisecond)

	return nil
}

// adaptParameters adapts migration parameters based on dirty page behavior
func (m *EnhancedPreCopyMigration) adaptParameters(dirtyPagePct float64) {
	// Only adapt if in auto mode
	if m.config.AdaptationMode != "auto" {
		return
	}

	stats := m.dirtyPageTracker.GetStats()
	pattern := stats.DirtyingPattern
	dirtyRate := stats.DirtyRate

	// Adapt based on pattern and rate
	switch pattern {
	case "hotspot":
		// For hotspots, increase priority of those regions
		m.logger.Info("Detected hotspot pattern, optimizing for hotspot regions")
		// In a real implementation, this would adjust transfer priorities
	case "sequential":
		// For sequential access, adjust prefetching
		m.logger.Info("Detected sequential pattern, enabling prefetching")
		// In a real implementation, this would enable prefetching
	case "random":
		// For random access, adjust parallel transfers
		m.logger.Info("Detected random pattern, adjusting parallelism")
		// In a real implementation, this would adjust parallelism
	}

	// If dirty rate is high, we might need to pause the VM briefly
	if dirtyRate > float64(len(m.dirtyPageTracker.dirtyPages))/10 {
		m.logger.Warn("High dirty page rate detected, consider throttling VM")
		// In a real implementation, this might throttle the VM
	}
}

// finalStopAndCopy performs the final stop and copy phase
func (m *EnhancedPreCopyMigration) finalStopAndCopy(ctx context.Context, optimizer *WANMigrationOptimizer) error {
	m.mu.Lock()
	m.status.Message = "Final stop and copy phase"
	m.status.ProgressPct = 80
	m.mu.Unlock()

	m.logger.Info("Beginning final stop and copy phase")

	// This is a placeholder for actual stop-and-copy implementation
	// In a real implementation, you would:
	// 1. Pause the source VM
	// 2. Get final dirty pages
	// 3. Transfer final memory state
	// 4. Switch execution to destination
	// 5. Resume VM at destination

	// Simulate final transfer (would be actual implementation in real code)
	time.Sleep(300 * time.Millisecond)

	m.mu.Lock()
	m.status.ProgressPct = 95
	m.mu.Unlock()

	return nil
}

// GetStatus returns the current migration status
func (m *EnhancedPreCopyMigration) GetStatus() MigrationStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.status
}

// Stop stops the migration
func (m *EnhancedPreCopyMigration) Stop() error {
	m.cancel()
	return nil
}

// GetDirtyPageStats returns statistics about dirty pages
func (m *EnhancedPreCopyMigration) GetDirtyPageStats() DirtyPageStats {
	return m.dirtyPageTracker.GetStats()
}

// GetIterationCount returns the number of iterations completed
func (m *EnhancedPreCopyMigration) GetIterationCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.iterations
}
