package vm

import (
	"bytes"
	_ "embed"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/link"
	"github.com/cilium/ebpf/perf"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
)

//go:embed page_tracker.bpf.o
var pageTrackerBPF []byte

const (
	// PageSize is the standard page size (4KB)
	PageSize = 4096

	// MaxTrackedPages is the maximum number of pages we can track
	MaxTrackedPages = 1048576 // 4GB worth of pages

	// DefaultAgingThresholdNs is the default time threshold for marking pages as unused (5 seconds)
	DefaultAgingThresholdNs = 5 * time.Second

	// DefaultMinAccessCount is the default minimum access count to consider a page active
	DefaultMinAccessCount = 1
)

var (
	// ErrEBPFNotSupported is returned when eBPF is not supported on the system
	ErrEBPFNotSupported = errors.New("eBPF is not supported on this system")

	// ErrEBPFLoadFailed is returned when eBPF program loading fails
	ErrEBPFLoadFailed = errors.New("failed to load eBPF program")

	// ErrEBPFAttachFailed is returned when eBPF program attachment fails
	ErrEBPFAttachFailed = errors.New("failed to attach eBPF program")
)

// PageAccessInfo represents page access information from eBPF
type PageAccessInfo struct {
	LastAccessTime uint64 // Nanoseconds since boot
	AccessCount    uint32 // Number of accesses
	IsDirty        uint8  // Whether page has been written to
	IsUnused       uint8  // Whether page is marked as unused
}

// PageTrackerConfig represents configuration for the eBPF page tracker
type PageTrackerConfig struct {
	AgingThresholdNs uint64 // Time threshold for considering a page unused
	MinAccessCount   uint32 // Minimum access count to consider a page active
	PID              uint32 // Target process ID (VM process)
}

// EBPFMigrationFilter implements eBPF-based page filtering for migration
type EBPFMigrationFilter struct {
	mu             sync.RWMutex
	collection     *ebpf.Collection
	links          []link.Link
	pageAccessMap  *ebpf.Map
	configMap      *ebpf.Map
	vaddrToPfnMap  *ebpf.Map
	config         PageTrackerConfig
	logger         *logrus.Entry
	enabled        bool
	perfReader     *perf.Reader
	bpfTimeOffset  int64 // Offset between bpf_ktime_get_ns() and wall-clock time
	timeCalibrated bool  // Whether time offset has been calibrated
}

// NewEBPFMigrationFilter creates a new eBPF migration filter
// The pid parameter should be the actual VM/guest process ID in the appropriate namespace
func NewEBPFMigrationFilter(logger *logrus.Logger, pid uint32) (*EBPFMigrationFilter, error) {
	// Check if eBPF is supported
	if !IsEBPFSupported() {
		return nil, ErrEBPFNotSupported
	}

	// Create logger entry
	logEntry := logger.WithField("component", "EBPFMigrationFilter")

	// Note: For guest-context injection (attaching BPF programs within the guest's namespace),
	// the caller should use NewEBPFMigrationFilterInGuestNamespace instead.
	// This function operates in the host's namespace by default.

	// Load eBPF collection from embedded bytecode
	spec, err := ebpf.LoadCollectionSpecFromReader(bytes.NewReader(pageTrackerBPF))
	if err != nil {
		logEntry.WithError(err).Error("Failed to load eBPF collection spec")
		return nil, fmt.Errorf("%w: %v", ErrEBPFLoadFailed, err)
	}

	// Load the collection into the kernel
	collection, err := ebpf.NewCollection(spec)
	if err != nil {
		logEntry.WithError(err).Error("Failed to create eBPF collection")
		return nil, fmt.Errorf("%w: %v", ErrEBPFLoadFailed, err)
	}

	// Get maps from the collection
	pageAccessMap := collection.Maps["page_access_map"]
	configMap := collection.Maps["config_map"]
	vaddrToPfnMap := collection.Maps["vaddr_to_pfn_map"]

	if pageAccessMap == nil || configMap == nil || vaddrToPfnMap == nil {
		collection.Close()
		return nil, fmt.Errorf("%w: required maps not found", ErrEBPFLoadFailed)
	}

	// Initialize default configuration
	config := PageTrackerConfig{
		AgingThresholdNs: uint64(DefaultAgingThresholdNs.Nanoseconds()),
		MinAccessCount:   DefaultMinAccessCount,
		PID:              pid,
	}

	filter := &EBPFMigrationFilter{
		collection:     collection,
		pageAccessMap:  pageAccessMap,
		configMap:      configMap,
		vaddrToPfnMap:  vaddrToPfnMap,
		config:         config,
		logger:         logEntry,
		enabled:        false,
		links:          make([]link.Link, 0),
		timeCalibrated: false,
		bpfTimeOffset:  0,
	}

	// Write initial configuration to the map
	if err := filter.updateConfig(); err != nil {
		collection.Close()
		return nil, fmt.Errorf("failed to initialize config: %w", err)
	}

	logEntry.WithFields(logrus.Fields{
		"pid":                pid,
		"aging_threshold_ns": config.AgingThresholdNs,
		"min_access_count":   config.MinAccessCount,
	}).Info("eBPF migration filter created successfully")

	return filter, nil
}

// ErrNamespaceSwitch is returned when namespace switching fails
var ErrNamespaceSwitch = errors.New("failed to switch namespace")

// NewEBPFMigrationFilterInGuestNamespace creates an eBPF migration filter that operates
// within the guest VM's namespace context. This enables tracking page accesses from the
// guest's perspective, providing true guest-aware unused page detection.
//
// The function performs the following steps:
// 1. Opens the current (host) PID namespace for later restoration
// 2. Opens the target guest PID namespace
// 3. Switches to the guest namespace using unix.Setns
// 4. Loads and prepares eBPF programs with PID=1 (guest init)
// 5. Switches back to the original host namespace
//
// Requirements:
// - CAP_SYS_ADMIN capability
// - Access to /proc/<pid>/ns/pid namespace file
// - The guest process must be accessible
//
// Parameters:
// - logger: Logger instance
// - hostPID: The guest VM process PID as seen from the host (used for fallback)
// - guestNamespacePath: Path to the guest's PID namespace (e.g., "/proc/<pid>/ns/pid")
//
// Returns an error if:
// - Namespace switching fails and fallback is not possible
// - eBPF loading/attachment fails
//
// On namespace switch failure, gracefully falls back to host namespace tracking.
func NewEBPFMigrationFilterInGuestNamespace(logger *logrus.Logger, hostPID uint32, guestNamespacePath string) (*EBPFMigrationFilter, error) {
	logEntry := logger.WithFields(logrus.Fields{
		"component":        "EBPFMigrationFilter",
		"host_pid":         hostPID,
		"namespace_path":   guestNamespacePath,
		"injection_method": "guest_namespace",
	})

	// Check if eBPF is supported
	if !IsEBPFSupported() {
		return nil, ErrEBPFNotSupported
	}

	// Step 1: Open current (host) PID namespace for restoration
	origNSFD, err := unix.Open("/proc/self/ns/pid", unix.O_RDONLY|unix.O_CLOEXEC, 0)
	if err != nil {
		logEntry.WithError(err).Warn("Failed to open original namespace, falling back to host namespace")
		return NewEBPFMigrationFilter(logger, hostPID)
	}
	defer unix.Close(origNSFD)

	// Step 2: Open guest PID namespace
	guestNSFD, err := unix.Open(guestNamespacePath, unix.O_RDONLY|unix.O_CLOEXEC, 0)
	if err != nil {
		logEntry.WithError(err).Warn("Failed to open guest namespace, falling back to host namespace")
		return NewEBPFMigrationFilter(logger, hostPID)
	}
	defer unix.Close(guestNSFD)

	// Step 3: Switch to guest namespace
	if err := unix.Setns(guestNSFD, unix.CLONE_NEWPID); err != nil {
		logEntry.WithError(err).Warn("Failed to switch to guest namespace (may require CAP_SYS_ADMIN), falling back to host namespace")
		return NewEBPFMigrationFilter(logger, hostPID)
	}

	logEntry.Info("Successfully switched to guest namespace for eBPF injection")

	// Step 4: Create filter with PID=1 (guest init process)
	// In the guest namespace context, PID 1 is the guest's init process
	// This allows eBPF to track page accesses from the guest's perspective
	guestPID := uint32(1)

	// Load eBPF collection from embedded bytecode
	spec, err := ebpf.LoadCollectionSpecFromReader(bytes.NewReader(pageTrackerBPF))
	if err != nil {
		// Restore namespace before returning
		unix.Setns(origNSFD, unix.CLONE_NEWPID)
		logEntry.WithError(err).Error("Failed to load eBPF collection spec in guest namespace")
		return nil, fmt.Errorf("%w: %v", ErrEBPFLoadFailed, err)
	}

	// Load the collection into the kernel (in guest namespace context)
	collection, err := ebpf.NewCollection(spec)
	if err != nil {
		// Restore namespace before returning
		unix.Setns(origNSFD, unix.CLONE_NEWPID)
		logEntry.WithError(err).Error("Failed to create eBPF collection in guest namespace")
		return nil, fmt.Errorf("%w: %v", ErrEBPFLoadFailed, err)
	}

	// Get maps from the collection
	pageAccessMap := collection.Maps["page_access_map"]
	configMap := collection.Maps["config_map"]
	vaddrToPfnMap := collection.Maps["vaddr_to_pfn_map"]

	if pageAccessMap == nil || configMap == nil || vaddrToPfnMap == nil {
		collection.Close()
		// Restore namespace before returning
		unix.Setns(origNSFD, unix.CLONE_NEWPID)
		return nil, fmt.Errorf("%w: required maps not found", ErrEBPFLoadFailed)
	}

	// Initialize configuration for guest namespace
	config := PageTrackerConfig{
		AgingThresholdNs: uint64(DefaultAgingThresholdNs.Nanoseconds()),
		MinAccessCount:   DefaultMinAccessCount,
		PID:              guestPID, // PID 1 in guest namespace
	}

	filter := &EBPFMigrationFilter{
		collection:     collection,
		pageAccessMap:  pageAccessMap,
		configMap:      configMap,
		vaddrToPfnMap:  vaddrToPfnMap,
		config:         config,
		logger:         logEntry,
		enabled:        false,
		links:          make([]link.Link, 0),
		timeCalibrated: false,
		bpfTimeOffset:  0,
	}

	// Write initial configuration to the map
	if err := filter.updateConfig(); err != nil {
		collection.Close()
		// Restore namespace before returning
		unix.Setns(origNSFD, unix.CLONE_NEWPID)
		return nil, fmt.Errorf("failed to initialize config: %w", err)
	}

	// Step 5: Switch back to original (host) namespace
	if err := unix.Setns(origNSFD, unix.CLONE_NEWPID); err != nil {
		logEntry.WithError(err).Error("Failed to restore original namespace - this may cause issues")
		// Continue anyway since eBPF is loaded
	}

	logEntry.WithFields(logrus.Fields{
		"guest_pid":          guestPID,
		"aging_threshold_ns": config.AgingThresholdNs,
		"min_access_count":   config.MinAccessCount,
	}).Info("eBPF migration filter created in guest namespace context")

	return filter, nil
}

// updateConfig updates the eBPF configuration map
func (f *EBPFMigrationFilter) updateConfig() error {
	key := uint32(0)
	configBytes := make([]byte, 16) // sizeof(struct page_tracker_config)

	binary.LittleEndian.PutUint64(configBytes[0:8], f.config.AgingThresholdNs)
	binary.LittleEndian.PutUint32(configBytes[8:12], f.config.MinAccessCount)
	binary.LittleEndian.PutUint32(configBytes[12:16], f.config.PID)

	return f.configMap.Put(&key, configBytes)
}

// Attach attaches the eBPF programs to the kernel
func (f *EBPFMigrationFilter) Attach() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if f.enabled {
		return nil // Already attached
	}

	// Attach page fault tracepoint
	pageFaultProg := f.collection.Programs["trace_page_fault"]
	if pageFaultProg != nil {
		l, err := link.AttachTracing(link.TracingOptions{
			Program: pageFaultProg,
		})
		if err != nil {
			f.logger.WithError(err).Warn("Failed to attach page fault tracer (non-fatal)")
		} else {
			f.links = append(f.links, l)
			f.logger.Debug("Attached page fault tracer")
		}
	}

	// Attach memory write kprobe
	memWriteProg := f.collection.Programs["trace_memory_write"]
	if memWriteProg != nil {
		l, err := link.AttachTracing(link.TracingOptions{
			Program: memWriteProg,
		})
		if err != nil {
			f.logger.WithError(err).Warn("Failed to attach memory write tracer (non-fatal)")
		} else {
			f.links = append(f.links, l)
			f.logger.Debug("Attached memory write tracer")
		}
	}

	if len(f.links) == 0 {
		return fmt.Errorf("%w: no programs attached", ErrEBPFAttachFailed)
	}

	f.enabled = true
	f.logger.WithField("attached_programs", len(f.links)).Info("eBPF programs attached successfully")

	// Calibrate time offset between BPF monotonic time and wall-clock time
	if err := f.calibrateTimeOffset(); err != nil {
		f.logger.WithError(err).Warn("Failed to calibrate time offset, aging may be inaccurate")
	}

	return nil
}

// calibrateTimeOffset calibrates the offset between bpf_ktime_get_ns() and wall-clock time
func (f *EBPFMigrationFilter) calibrateTimeOffset() error {
	// Approach: Create a temporary page access event and immediately read the timestamp
	// This gives us a rough approximation of the offset

	// Get current wall-clock time
	wallTime := time.Now().UnixNano()

	// Try to read any existing page timestamp from the map
	// In a real implementation, we might trigger a page fault to get a fresh timestamp
	// For now, we'll use a simple approximation
	iter := f.pageAccessMap.Iterate()
	var pfn uint64
	var info PageAccessInfo

	if iter.Next(&pfn, &info) {
		// We have at least one page tracked
		bpfTime := int64(info.LastAccessTime)
		f.bpfTimeOffset = wallTime - bpfTime
		f.timeCalibrated = true

		f.logger.WithFields(logrus.Fields{
			"wall_time":   wallTime,
			"bpf_time":    bpfTime,
			"time_offset": f.bpfTimeOffset,
		}).Debug("Time offset calibrated")

		return nil
	}

	// No pages tracked yet, use zero offset as fallback
	// This will be recalibrated when first page is accessed
	f.bpfTimeOffset = 0
	f.timeCalibrated = false

	f.logger.Debug("No pages tracked yet, time calibration deferred")
	return nil
}

// Detach detaches all eBPF programs
func (f *EBPFMigrationFilter) Detach() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if !f.enabled {
		return nil
	}

	// Close all links
	for i, l := range f.links {
		if err := l.Close(); err != nil {
			f.logger.WithError(err).WithField("link_index", i).Warn("Failed to close link")
		}
	}

	f.links = nil
	f.enabled = false
	f.logger.Info("eBPF programs detached")

	return nil
}

// Close releases all resources
func (f *EBPFMigrationFilter) Close() error {
	// Detach programs first
	if err := f.Detach(); err != nil {
		f.logger.WithError(err).Warn("Error detaching programs during close")
	}

	// Close perf reader if exists
	if f.perfReader != nil {
		f.perfReader.Close()
	}

	// Close collection
	if f.collection != nil {
		return f.collection.Close()
	}

	return nil
}

// IsPageUnused checks if a given page frame number is unused
func (f *EBPFMigrationFilter) IsPageUnused(pfn uint64) bool {
	// First, check if we need to calibrate time (requires write lock)
	// We do this check separately to avoid lock upgrade issues
	needsCalibration := false
	f.mu.RLock()
	if f.enabled && !f.timeCalibrated {
		needsCalibration = true
	}
	f.mu.RUnlock()

	// If calibration is needed, do it under write lock
	if needsCalibration {
		f.mu.Lock()
		// Double-check after acquiring write lock
		if f.enabled && !f.timeCalibrated {
			f.calibrateTimeOffset()
		}
		f.mu.Unlock()
	}

	// Now perform the actual page check under read lock
	f.mu.RLock()
	defer f.mu.RUnlock()

	if !f.enabled {
		return false // Conservative: consider all pages used if not enabled
	}

	var info PageAccessInfo
	if err := f.pageAccessMap.Lookup(&pfn, &info); err != nil {
		// Page not in map means it was never accessed
		return true
	}

	// Check if page has been explicitly marked as unused
	if info.IsUnused != 0 {
		return true
	}

	// Check aging threshold with calibrated time offset
	if f.timeCalibrated {
		// Convert BPF monotonic time to wall-clock time
		bpfWallTime := int64(info.LastAccessTime) + f.bpfTimeOffset
		currentWallTime := time.Now().UnixNano()
		timeSinceAccess := currentWallTime - bpfWallTime

		if timeSinceAccess > int64(f.config.AgingThresholdNs) {
			// Page hasn't been accessed recently
			if info.AccessCount < f.config.MinAccessCount {
				return true
			}
		}
	} else {
		// Time still not calibrated after attempt, be conservative
		return false // Don't mark as unused if we can't determine age
	}

	return false
}

// GetPageInfo retrieves access information for a page
func (f *EBPFMigrationFilter) GetPageInfo(pfn uint64) (*PageAccessInfo, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	var info PageAccessInfo
	if err := f.pageAccessMap.Lookup(&pfn, &info); err != nil {
		return nil, fmt.Errorf("page not found: %w", err)
	}

	return &info, nil
}

// MarkPagesAsUnused marks aged-out pages as unused
func (f *EBPFMigrationFilter) MarkPagesAsUnused() (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if !f.enabled {
		return 0, nil
	}

	// Ensure time is calibrated
	if !f.timeCalibrated {
		if err := f.calibrateTimeOffset(); err != nil {
			f.logger.WithError(err).Warn("Time calibration failed, skipping aging")
			return 0, nil
		}
	}

	currentWallTime := time.Now().UnixNano()
	markedCount := 0

	// Iterate over all pages in the map
	iter := f.pageAccessMap.Iterate()
	var pfn uint64
	var info PageAccessInfo

	for iter.Next(&pfn, &info) {
		// Convert BPF time to wall-clock time
		bpfWallTime := int64(info.LastAccessTime) + f.bpfTimeOffset
		timeSinceAccess := currentWallTime - bpfWallTime

		// Check if page should be marked as unused
		if timeSinceAccess > int64(f.config.AgingThresholdNs) {
			if info.AccessCount < f.config.MinAccessCount && info.IsUnused == 0 {
				info.IsUnused = 1
				if err := f.pageAccessMap.Put(&pfn, &info); err != nil {
					f.logger.WithError(err).WithField("pfn", pfn).Warn("Failed to mark page as unused")
					continue
				}
				markedCount++
			}
		}
	}

	if err := iter.Err(); err != nil {
		return markedCount, fmt.Errorf("error iterating pages: %w", err)
	}

	f.logger.WithField("marked_count", markedCount).Debug("Marked aged-out pages as unused")

	return markedCount, nil
}

// GetStats returns statistics about tracked pages
func (f *EBPFMigrationFilter) GetStats() map[string]interface{} {
	f.mu.RLock()
	defer f.mu.RUnlock()

	stats := make(map[string]interface{})
	stats["enabled"] = f.enabled
	stats["attached_programs"] = len(f.links)

	if !f.enabled {
		return stats
	}

	// Count pages by state
	var totalPages, unusedPages, dirtyPages int

	iter := f.pageAccessMap.Iterate()
	var pfn uint64
	var info PageAccessInfo

	for iter.Next(&pfn, &info) {
		totalPages++
		if info.IsUnused != 0 {
			unusedPages++
		}
		if info.IsDirty != 0 {
			dirtyPages++
		}
	}

	stats["total_pages"] = totalPages
	stats["unused_pages"] = unusedPages
	stats["dirty_pages"] = dirtyPages
	stats["used_pages"] = totalPages - unusedPages

	if totalPages > 0 {
		stats["unused_percentage"] = float64(unusedPages) / float64(totalPages) * 100
	}

	return stats
}

// SetAgingThreshold sets the aging threshold for marking pages as unused
func (f *EBPFMigrationFilter) SetAgingThreshold(threshold time.Duration) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.config.AgingThresholdNs = uint64(threshold.Nanoseconds())
	return f.updateConfig()
}

// SetMinAccessCount sets the minimum access count to consider a page active
func (f *EBPFMigrationFilter) SetMinAccessCount(count uint32) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.config.MinAccessCount = count
	return f.updateConfig()
}

// FileOffsetToPageMapping represents mapping from file offsets to guest page frame numbers
// This is needed because eBPF tracks guest virtual addresses/PFNs, but delta sync operates on file offsets
type FileOffsetToPageMapping struct {
	// FileType indicates what kind of file this mapping is for
	// "memory_snapshot" - VM memory state file
	// "disk_image" - VM disk image file
	// "process_memory" - Direct process memory mapping
	FileType string

	// BaseOffset is the starting offset in the file
	BaseOffset int64

	// OffsetToPFN maps file offsets to guest page frame numbers
	// Key: file offset (in pages), Value: guest PFN
	OffsetToPFN map[int64]uint64

	// Reverse mapping for quick lookups
	PFNToOffset map[uint64]int64
}

// EBPFBlockFilter filters blocks based on eBPF page tracking
// IMPORTANT: Block filtering only works correctly when provided with a proper
// FileOffsetToPageMapping that translates file offsets to guest page frame numbers.
// Without this mapping, filtering is disabled to avoid incorrectly skipping used data.
type EBPFBlockFilter struct {
	filter    *EBPFMigrationFilter
	blockSize int
	logger    *logrus.Entry
	mapping   *FileOffsetToPageMapping // Optional: enables accurate filtering
}

// NewEBPFBlockFilter creates a new block filter
func NewEBPFBlockFilter(filter *EBPFMigrationFilter, blockSize int) *EBPFBlockFilter {
	return &EBPFBlockFilter{
		filter:    filter,
		blockSize: blockSize,
		logger:    filter.logger.WithField("component", "EBPFBlockFilter"),
		mapping:   nil, // No mapping by default - filtering disabled for safety
	}
}

// SetFileMapping sets the file-to-page mapping to enable accurate block filtering
func (bf *EBPFBlockFilter) SetFileMapping(mapping *FileOffsetToPageMapping) {
	bf.mapping = mapping
	bf.logger.WithFields(logrus.Fields{
		"file_type":     mapping.FileType,
		"base_offset":   mapping.BaseOffset,
		"mapping_count": len(mapping.OffsetToPFN),
	}).Info("File-to-page mapping configured")
}

// IsBlockUnused checks if all pages in a block are unused
// Returns false if no file-to-page mapping is configured (conservative)
func (bf *EBPFBlockFilter) IsBlockUnused(blockOffset int64) bool {
	if bf.filter == nil || !bf.filter.enabled {
		return false
	}

	// CRITICAL: Without a proper mapping, we cannot safely determine if a block is unused
	// The naive approach of startPage := blockOffset / PageSize is incorrect because:
	// 1. File offsets don't directly correspond to guest PFNs
	// 2. The file might be a memory snapshot with non-contiguous pages
	// 3. The file might be a disk image that doesn't relate to memory at all
	if bf.mapping == nil {
		// No mapping configured - be conservative and don't skip any blocks
		bf.logger.Debug("No file-to-page mapping configured, disabling block filtering")
		return false
	}

	// Calculate which file pages this block spans
	startFilePageOffset := (blockOffset - bf.mapping.BaseOffset) / PageSize
	pagesInBlock := int64((bf.blockSize + PageSize - 1) / PageSize)

	// Check if all pages in the block are unused according to the mapping
	for i := int64(0); i < pagesInBlock; i++ {
		filePageOffset := startFilePageOffset + i

		// Look up the guest PFN for this file page
		guestPFN, exists := bf.mapping.OffsetToPFN[filePageOffset]
		if !exists {
			// No mapping for this page - assume it's used to be safe
			return false
		}

		// Check if the guest PFN is unused
		if !bf.filter.IsPageUnused(guestPFN) {
			return false
		}
	}

	return true
}

// ShouldSkipBlock determines if a block should be skipped during migration
func (bf *EBPFBlockFilter) ShouldSkipBlock(blockIndex int, blockSize int) bool {
	blockOffset := int64(blockIndex) * int64(blockSize)
	return bf.IsBlockUnused(blockOffset)
}

// ClearFileMapping removes any configured file-to-page mapping
func (bf *EBPFBlockFilter) ClearFileMapping() {
	bf.mapping = nil
	bf.logger.Debug("File-to-page mapping cleared")
}

// HasFileMapping returns whether a file-to-page mapping is configured
func (bf *EBPFBlockFilter) HasFileMapping() bool {
	return bf.mapping != nil
}

// CreateMemorySnapshotMapping creates a FileOffsetToPageMapping for a QEMU memory snapshot file.
// For QEMU RAM dump files, pages are stored sequentially, so file_offset / PageSize = guest PFN.
// This enables accurate eBPF-based unused page detection during memory migration.
//
// Parameters:
//   - fileSize: The size of the memory snapshot file in bytes
//   - baseOffset: The starting offset in the file (usually 0 for memory snapshots)
//
// Returns a FileOffsetToPageMapping suitable for use with SetFileMapping.
func CreateMemorySnapshotMapping(fileSize int64, baseOffset int64) *FileOffsetToPageMapping {
	mapping := &FileOffsetToPageMapping{
		FileType:    "memory_snapshot",
		BaseOffset:  baseOffset,
		OffsetToPFN: make(map[int64]uint64),
		PFNToOffset: make(map[uint64]int64),
	}

	// For QEMU RAM dumps, mapping is sequential: file_page_offset = guest PFN
	// This is because QEMU dumps memory sequentially from guest physical address 0
	pageCount := (fileSize + PageSize - 1) / PageSize
	for i := int64(0); i < pageCount; i++ {
		mapping.OffsetToPFN[i] = uint64(i)
		mapping.PFNToOffset[uint64(i)] = i
	}

	return mapping
}

// CreateDiskImageMapping creates a FileOffsetToPageMapping for disk images.
// For disk images, eBPF filtering is typically not applicable since guest kernel
// page tracking relates to memory, not disk blocks. However, this function
// provides a framework for potential future disk-aware filtering.
//
// Parameters:
//   - fileSize: The size of the disk image file in bytes
//   - baseOffset: The starting offset in the file
//
// Returns nil to indicate disk images should not use eBPF page filtering.
// The caller should skip eBPF filtering for disk images.
func CreateDiskImageMapping(fileSize int64, baseOffset int64) *FileOffsetToPageMapping {
	// For disk images, we return nil to disable eBPF filtering
	// Guest kernel page tracking doesn't directly relate to disk blocks
	// Disk blocks are accessed via block device layer, not page faults
	return nil
}

// DetectFileType attempts to detect whether a file is a memory snapshot or disk image
// based on file path patterns and metadata.
//
// Returns "memory_snapshot", "disk_image", or "unknown"
func DetectFileType(filePath string) string {
	// Common patterns for memory state files
	memoryPatterns := []string{
		"memory.state",
		"memory_delta",
		".mem",
		"ram.dump",
		"memstate",
	}

	// Common patterns for disk images
	diskPatterns := []string{
		".qcow2",
		".qcow",
		".raw",
		".vmdk",
		".vhd",
		".vhdx",
		".img",
		"disk",
	}

	lowerPath := filePath
	for i := range lowerPath {
		if lowerPath[i] >= 'A' && lowerPath[i] <= 'Z' {
			lowerPath = filePath[:i] + string(lowerPath[i]+32) + filePath[i+1:]
		}
	}

	for _, pattern := range memoryPatterns {
		if containsString(lowerPath, pattern) {
			return "memory_snapshot"
		}
	}

	for _, pattern := range diskPatterns {
		if containsString(lowerPath, pattern) {
			return "disk_image"
		}
	}

	return "unknown"
}

// containsString checks if s contains substr (case-insensitive)
func containsString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
