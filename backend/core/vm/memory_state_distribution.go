// Package vm provides memory state distribution with delta synchronization
package vm

import (
	"bytes"
	"compress/zlib"
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pkg/errors"
	"go.uber.org/zap"
)

// MemoryStateDistribution manages distributed VM memory state with delta synchronization
type MemoryStateDistribution struct {
	mu                 sync.RWMutex
	logger             *zap.Logger
	nodeID             string
	deltaSync          WANMigrationDeltaSync
	shardingManager    *VMStateShardingManager
	prefetcher         *PredictivePrefetchingEngine
	memoryShards       map[string]*MemoryShard
	coherenceProtocol  *MemoryCoherenceProtocol
	compressionEngine  *MemoryCompressionEngine
	accessTracker      *MemoryAccessTracker
	recoveryManager    *MemoryRecoveryManager
	migrationStrategies map[string]MigrationStrategy
	metrics            *MemoryDistributionMetrics
	bandwidthLimiter   *BandwidthLimiter     // For throttling network usage
	dirtyBitmap        *DirtyPageBitmap      // Global dirty page tracking
}

// MemoryShard represents a distributed memory shard
type MemoryShard struct {
	ShardID       string
	VMID          string
	StartAddress  uint64
	EndAddress    uint64
	Pages         map[uint64]*DistributedMemoryPage
	ReplicaNodes  []string
	Version       uint64
	DirtyBitmap   *DirtyPageBitmap
	LastSync      time.Time
	CoherenceState CoherenceState
	PostCopyMode  bool                     // Whether this shard is in post-copy mode
	FaultHandler  *PostCopyFaultHandler    // Handler for post-copy page faults
}

// DistributedMemoryPage represents a memory page in distributed state
type DistributedMemoryPage struct {
	PageNumber      uint64
	Data            []byte
	Compressed      bool
	Deduplicated    bool
	DuplicateRef    string
	AccessCount     uint64
	LastAccess      time.Time
	Temperature     PageTemperature // Hot, Warm, Cold
	Checksum        [32]byte
	Version         uint64
	Dirty           bool
	Locked          bool
	Owner           string
	VMID            string          // VM this page belongs to
	ShardID         string          // Shard this page belongs to
	LastTransfer    time.Time       // Last time page was transferred
	TransferVersion uint64          // Version at last transfer
}

// PageTemperature indicates access frequency
type PageTemperature int

const (
	PageTemperatureHot PageTemperature = iota
	PageTemperatureWarm
	PageTemperatureCold
)

// CoherenceState represents memory coherence state
type CoherenceState int

const (
	CoherenceStateModified CoherenceState = iota
	CoherenceStateExclusive
	CoherenceStateShared
	CoherenceStateInvalid
)

// MigrationStrategy defines memory migration approach
type MigrationStrategy interface {
	Migrate(ctx context.Context, shard *MemoryShard, targetNode string) error
	GetType() string
}

// PreCopyMigration implements pre-copy migration strategy
type PreCopyMigration struct {
	maxIterations int
	dirtyThreshold float64
}

// PostCopyMigration implements post-copy migration strategy
type PostCopyMigration struct {
	pageFaultHandler *PageFaultHandler
}

// HybridMigration combines pre-copy and post-copy strategies
type HybridMigration struct {
	preCopy  *PreCopyMigration
	postCopy *PostCopyMigration
	aiOptimizer *AIOptimizer
}

// MemoryCoherenceProtocol manages memory coherence across nodes
type MemoryCoherenceProtocol struct {
	mu            sync.RWMutex
	coherenceMap  map[string]CoherenceState
	invalidations chan InvalidationMessage
	updates       chan UpdateMessage
}

// MemoryCompressionEngine handles memory-specific compression
type MemoryCompressionEngine struct {
	deduplicator    *MemoryDeduplicator
	zeroEliminator  *ZeroPageEliminator
	patternDetector *PatternCompressionDetector
	compressors     map[string]Compressor
}

// MemoryAccessTracker tracks memory access patterns
type MemoryAccessTracker struct {
	mu           sync.RWMutex
	accessLog    *CircularBuffer
	hotPages     *HotPageTracker
	coldPages    *ColdPageTracker
	predictions  *AccessPredictionModel
}

// MemoryRecoveryManager handles memory state recovery
type MemoryRecoveryManager struct {
	checkpoints     []*MemoryCheckpoint
	replicaManager  *ReplicaManager
	recoveryLog     *RecoveryLog
}

// NewMemoryStateDistribution creates a new memory state distribution manager
func NewMemoryStateDistribution(logger *zap.Logger, nodeID string, deltaSync *WANMigrationDeltaSync) *MemoryStateDistribution {
	msd := &MemoryStateDistribution{
		logger:              logger,
		nodeID:              nodeID,
		deltaSync:           deltaSync,
		memoryShards:        make(map[string]*MemoryShard),
		coherenceProtocol:   NewMemoryCoherenceProtocol(),
		compressionEngine:   NewMemoryCompressionEngine(),
		accessTracker:       NewMemoryAccessTracker(),
		recoveryManager:     NewMemoryRecoveryManager(),
		migrationStrategies: make(map[string]MigrationStrategy),
		metrics:             NewMemoryDistributionMetrics(),
	}

	// Initialize migration strategies
	msd.migrationStrategies["pre-copy"] = &PreCopyMigration{
		maxIterations:  5,
		dirtyThreshold: 0.1,
	}
	msd.migrationStrategies["post-copy"] = &PostCopyMigration{
		pageFaultHandler: NewPageFaultHandler(),
	}
	msd.migrationStrategies["hybrid"] = &HybridMigration{
		preCopy:     msd.migrationStrategies["pre-copy"].(*PreCopyMigration),
		postCopy:    msd.migrationStrategies["post-copy"].(*PostCopyMigration),
		aiOptimizer: NewAIOptimizer(),
	}

	return msd
}

// NewMemoryStateDistributionWithSharding creates memory distribution with sharding integration
func NewMemoryStateDistributionWithSharding(logger *zap.Logger, nodeID string, deltaSync *WANMigrationDeltaSync, shardingManager *VMStateShardingManager) *MemoryStateDistribution {
	msd := NewMemoryStateDistribution(logger, nodeID, deltaSync)
	msd.shardingManager = shardingManager
	return msd
}

// DistributeMemoryState distributes VM memory across nodes
func (msd *MemoryStateDistribution) DistributeMemoryState(ctx context.Context, vmID string, memorySize uint64) error {
	msd.mu.Lock()
	defer msd.mu.Unlock()

	msd.logger.Info("Distributing memory state",
		zap.String("vmID", vmID),
		zap.Uint64("memorySize", memorySize))

	// Calculate shard size and count
	pageSize := uint64(4096) // 4KB pages
	totalPages := memorySize / pageSize
	pagesPerShard := totalPages / uint64(msd.getNodeCount())

	// Create memory shards
	var startPage uint64
	for i := 0; i < msd.getNodeCount(); i++ {
		endPage := startPage + pagesPerShard
		if i == msd.getNodeCount()-1 {
			endPage = totalPages
		}

		shard := &MemoryShard{
			ShardID:        fmt.Sprintf("%s-shard-%d", vmID, i),
			VMID:           vmID,
			StartAddress:   startPage * pageSize,
			EndAddress:     endPage * pageSize,
			Pages:          make(map[uint64]*DistributedMemoryPage),
			ReplicaNodes:   msd.selectReplicaNodes(3),
			Version:        1,
			DirtyBitmap:    NewDirtyPageBitmap(endPage - startPage),
			LastSync:       time.Now(),
			CoherenceState: CoherenceStateInvalid,
		}

		// Set the base page for proper bitmap indexing
		shard.DirtyBitmap.BasePage = startPage

		// Initialize pages
		for pageNum := startPage; pageNum < endPage; pageNum++ {
			shard.Pages[pageNum] = &DistributedMemoryPage{
				PageNumber:  pageNum,
				Data:        make([]byte, pageSize),
				Temperature: PageTemperatureCold,
				Version:     1,
			}
		}

		msd.memoryShards[shard.ShardID] = shard
		startPage = endPage
	}

	msd.metrics.ShardsCreated.Add(int64(msd.getNodeCount()))
	return nil
}

// SyncMemoryDelta synchronizes memory changes using delta synchronization
func (msd *MemoryStateDistribution) SyncMemoryDelta(ctx context.Context, vmID string) error {
	msd.mu.RLock()
	shards := msd.getVMShards(vmID)
	msd.mu.RUnlock()

	for _, shard := range shards {
		if err := msd.syncShardDelta(ctx, shard); err != nil {
			msd.logger.Error("Failed to sync shard delta",
				zap.String("shardID", shard.ShardID),
				zap.Error(err))
			continue
		}
	}

	return nil
}

// syncShardDelta synchronizes a single shard's delta
func (msd *MemoryStateDistribution) syncShardDelta(ctx context.Context, shard *MemoryShard) error {
	// Get dirty pages
	dirtyPages := shard.DirtyBitmap.GetDirtyPages()
	if len(dirtyPages) == 0 {
		return nil
	}

	msd.logger.Debug("Syncing dirty pages",
		zap.String("shardID", shard.ShardID),
		zap.Int("dirtyCount", len(dirtyPages)))

	// Create delta for dirty pages
	delta := &MemoryDelta{
		ShardID:   shard.ShardID,
		Version:   shard.Version,
		Timestamp: time.Now(),
		Pages:     make([]*PageDelta, 0, len(dirtyPages)),
	}

	for _, pageNum := range dirtyPages {
		page := shard.Pages[pageNum]
		if page == nil {
			continue
		}

		// Compress page data
		compressedData, err := msd.compressPage(page)
		if err != nil {
			msd.logger.Warn("Failed to compress page",
				zap.Uint64("pageNum", pageNum),
				zap.Error(err))
			compressedData = page.Data
		}

		delta.Pages = append(delta.Pages, &PageDelta{
			PageNumber: pageNum,
			Data:       compressedData,
			Checksum:   page.Checksum,
			Version:    page.Version,
		})
	}

	// Send delta to replicas
	for _, replica := range shard.ReplicaNodes {
		if err := msd.sendDeltaToReplica(ctx, delta, replica); err != nil {
			msd.logger.Error("Failed to send delta to replica",
				zap.String("replica", replica),
				zap.Error(err))
		}
	}

	// Clear dirty bitmap
	shard.DirtyBitmap.Clear()
	shard.LastSync = time.Now()
	shard.Version++

	msd.metrics.DeltasSynced.Inc()
	msd.metrics.BytesSynced.Add(int64(len(dirtyPages) * 4096))

	return nil
}

// MigrateLiveMemory performs live memory migration with WAN delta sync integration
func (msd *MemoryStateDistribution) MigrateLiveMemory(ctx context.Context, vmID, targetNode, strategy string) error {
	msd.mu.Lock()
	defer msd.mu.Unlock()

	msd.logger.Info("Starting live memory migration",
		zap.String("vmID", vmID),
		zap.String("targetNode", targetNode),
		zap.String("strategy", strategy))

	migrationStrategy, exists := msd.migrationStrategies[strategy]
	if !exists {
		migrationStrategy = msd.migrationStrategies["hybrid"]
	}

	shards := msd.getVMShards(vmID)

	// Pre-migration: Use delta sync to prepare target node
	if msd.deltaSync != nil {
		for _, shard := range shards {
			// Get dirty pages for delta calculation
			dirtyPages := shard.DirtyBitmap.GetDirtyPages()
			if len(dirtyPages) > 0 {
				// Trigger pre-migration delta sync
				if err := msd.deltaSync.PreComputeDeltas(vmID, msd.convertPageNumbersToBlocks(dirtyPages)); err != nil {
					msd.logger.Warn("Failed to pre-compute deltas for migration",
						zap.String("vmID", vmID),
						zap.Error(err))
				}
			}
		}

		// Start WAN delta sync to target node
		if err := msd.deltaSync.StartWANSync(ctx, vmID, targetNode); err != nil {
			msd.logger.Warn("Failed to start WAN delta sync",
				zap.String("vmID", vmID),
				zap.String("targetNode", targetNode),
				zap.Error(err))
		}
	}

	// Use AI to predict access patterns during migration
	if msd.prefetcher != nil {
		predictions := msd.prefetcher.PredictAccessPatterns(vmID)
		msd.optimizeMigrationOrder(shards, predictions)
	}

	// Migrate each shard with delta sync support
	var wg sync.WaitGroup
	errChan := make(chan error, len(shards))

	for _, shard := range shards {
		wg.Add(1)
		go func(s *MemoryShard) {
			defer wg.Done()

			// Perform delta-based migration
			if err := msd.performDeltaMigration(ctx, s, targetNode, migrationStrategy); err != nil {
				errChan <- errors.Wrapf(err, "failed to migrate shard %s", s.ShardID)
			}
		}(shard)
	}

	wg.Wait()
	close(errChan)

	// Check for errors
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	// Post-migration: Finalize delta sync
	if msd.deltaSync != nil {
		if err := msd.deltaSync.FinalizeSync(ctx, vmID, targetNode); err != nil {
			msd.logger.Warn("Failed to finalize WAN delta sync",
				zap.String("vmID", vmID),
				zap.String("targetNode", targetNode),
				zap.Error(err))
		}
	}

	msd.metrics.MigrationsCompleted.Inc()
	return nil
}

// TrackMemoryAccess tracks memory access patterns
func (msd *MemoryStateDistribution) TrackMemoryAccess(vmID string, pageNumber uint64) {
	msd.mu.Lock()
	defer msd.mu.Unlock()

	// Find the shard containing this page
	for _, shard := range msd.memoryShards {
		if shard.VMID != vmID {
			continue
		}

		if page, exists := shard.Pages[pageNumber]; exists {
			atomic.AddUint64(&page.AccessCount, 1)
			page.LastAccess = time.Now()

			// Update temperature based on access frequency
			msd.updatePageTemperature(page)

			// Track in access tracker
			msd.accessTracker.TrackAccess(vmID, pageNumber)

			// Mark page as dirty if modified
			if page.Dirty {
				shard.DirtyBitmap.SetDirty(pageNumber)
			}

			break
		}
	}
}

// updatePageTemperature updates page temperature based on access patterns
func (msd *MemoryStateDistribution) updatePageTemperature(page *DistributedMemoryPage) {
	now := time.Now()
	timeSinceLastAccess := now.Sub(page.LastAccess)

	if timeSinceLastAccess < 1*time.Second && page.AccessCount > 100 {
		page.Temperature = PageTemperatureHot
	} else if timeSinceLastAccess < 1*time.Minute && page.AccessCount > 10 {
		page.Temperature = PageTemperatureWarm
	} else {
		page.Temperature = PageTemperatureCold
	}
}

// compressPage compresses a memory page
func (msd *MemoryStateDistribution) compressPage(page *DistributedMemoryPage) ([]byte, error) {
	// Check for zero page
	if msd.compressionEngine.zeroEliminator.IsZeroPage(page.Data) {
		return []byte{0}, nil // Single byte to indicate zero page
	}

	// Check for deduplication
	if duplicate := msd.compressionEngine.deduplicator.FindDuplicate(page.Data); duplicate != "" {
		page.Deduplicated = true
		page.DuplicateRef = duplicate
		return []byte(duplicate), nil
	}

	// Pattern-based compression
	if pattern := msd.compressionEngine.patternDetector.DetectPattern(page.Data); pattern != nil {
		return pattern.Compress(page.Data)
	}

	// Standard compression
	var buf bytes.Buffer
	w := zlib.NewWriter(&buf)
	_, err := w.Write(page.Data)
	w.Close()

	if err != nil {
		return nil, err
	}

	page.Compressed = true
	return buf.Bytes(), nil
}

// RecoverMemoryState recovers memory state from replicas
func (msd *MemoryStateDistribution) RecoverMemoryState(ctx context.Context, vmID string) error {
	msd.mu.Lock()
	defer msd.mu.Unlock()

	msd.logger.Info("Recovering memory state", zap.String("vmID", vmID))

	// Try checkpoint recovery first
	if checkpoint := msd.recoveryManager.GetLatestCheckpoint(vmID); checkpoint != nil {
		if err := msd.restoreFromCheckpoint(checkpoint); err == nil {
			msd.logger.Info("Recovered from checkpoint",
				zap.String("vmID", vmID),
				zap.String("checkpointID", checkpoint.ID))
			return nil
		}
	}

	// Recover from replicas
	shards := msd.getVMShards(vmID)
	for _, shard := range shards {
		if err := msd.recoverShardFromReplicas(ctx, shard); err != nil {
			msd.logger.Error("Failed to recover shard",
				zap.String("shardID", shard.ShardID),
				zap.Error(err))
			continue
		}
	}

	msd.metrics.RecoveriesCompleted.Inc()
	return nil
}

// Helper methods

func (msd *MemoryStateDistribution) getNodeCount() int {
	// Get actual node count from sharding manager or federation
	if msd.shardingManager != nil {
		// Get node count from consistent hash ring
		allNodes := msd.shardingManager.nodeRing.GetAllNodes()
		if len(allNodes) > 0 {
			return len(allNodes)
		}
	}

	// Fallback to default if no membership available
	return 4
}

func (msd *MemoryStateDistribution) selectReplicaNodes(count int) []string {
	// Use sharding ring to select distinct replica nodes
	if msd.shardingManager != nil && msd.shardingManager.nodeRing != nil {
		// Get all available nodes from the ring
		allNodes := msd.shardingManager.nodeRing.GetAllNodes()
		if len(allNodes) > 0 {
			// Convert map to slice
			nodeList := make([]string, 0, len(allNodes))
			for node := range allNodes {
				// Don't include local node as replica
				if node != msd.nodeID {
					nodeList = append(nodeList, node)
				}
			}

			// Select up to 'count' nodes
			replicas := make([]string, 0, count)
			for i := 0; i < count && i < len(nodeList); i++ {
				replicas = append(replicas, nodeList[i])
			}

			// If not enough real nodes, use what we have
			return replicas
		}
	}

	// Integration with federation manager to get real nodes
	if msd.shardingManager != nil && msd.shardingManager.federation != nil {
		// Get available nodes from federation
		availableNodes := msd.getAvailableNodesFromFederation()

		replicas := []string{}
		for i := 0; i < count && i < len(availableNodes); i++ {
			replicas = append(replicas, availableNodes[i])
		}

		return replicas
	}

	// Fallback to simulated nodes only when membership is truly empty
	replicas := []string{}
	for i := 0; i < count; i++ {
		replicas = append(replicas, fmt.Sprintf("sim-node-%d", i+1))
	}
	return replicas
}

func (msd *MemoryStateDistribution) getVMShards(vmID string) []*MemoryShard {
	shards := []*MemoryShard{}
	for _, shard := range msd.memoryShards {
		if shard.VMID == vmID {
			shards = append(shards, shard)
		}
	}
	return shards
}

func (msd *MemoryStateDistribution) sendDeltaToReplica(ctx context.Context, delta *MemoryDelta, replica string) error {
	// Implementation would send delta to replica via RPC
	return nil
}

func (msd *MemoryStateDistribution) optimizeMigrationOrder(shards []*MemoryShard, predictions *AccessPredictions) {
	// Use AI predictions to optimize migration order
	if predictions == nil || len(predictions.PagePredictions) == 0 {
		// No predictions available, use default order
		return
	}

	// Create a map of page priorities based on AI predictions
	pagePriorities := make(map[uint64]float64)
	for _, pred := range predictions.PagePredictions {
		// Higher access probability = higher priority (migrate last)
		pagePriorities[pred.PageNumber] = pred.AccessProbability
	}

	// Sort shards based on aggregate priority of their pages
	sort.Slice(shards, func(i, j int) bool {
		// Calculate average priority for each shard
		priI := msd.calculateShardPriority(shards[i], pagePriorities)
		priJ := msd.calculateShardPriority(shards[j], pagePriorities)

		// Lower priority shards migrate first (cold pages)
		// Higher priority shards migrate last (hot pages)
		return priI < priJ
	})

	// Log migration order optimization
	msd.logger.Info("Optimized migration order based on AI predictions",
		zap.Int("shardCount", len(shards)),
		zap.Int("predictionCount", len(predictions.PagePredictions)))
}

// calculateShardPriority calculates priority score for a shard based on page predictions
func (msd *MemoryStateDistribution) calculateShardPriority(shard *MemoryShard, pagePriorities map[uint64]float64) float64 {
	if len(shard.Pages) == 0 {
		return 0.0
	}

	var totalPriority float64
	var pageCount int

	for pageNum, page := range shard.Pages {
		// Get AI-predicted priority
		aiPriority, hasAIPrediction := pagePriorities[pageNum]

		// Combine AI prediction with current temperature
		var priority float64
		if hasAIPrediction {
			priority = aiPriority * 0.7 // 70% weight to AI prediction
		}

		// Add temperature-based priority (30% weight)
		switch page.Temperature {
		case PageTemperatureHot:
			priority += 0.3 * 1.0
		case PageTemperatureWarm:
			priority += 0.3 * 0.5
		case PageTemperatureCold:
			priority += 0.3 * 0.1
		}

		// Factor in access count
		if page.AccessCount > 0 {
			// Normalize access count (log scale to prevent extreme values)
			accessFactor := math.Log10(float64(page.AccessCount)+1) / 10.0
			priority += accessFactor * 0.2
		}

		totalPriority += priority
		pageCount++
	}

	if pageCount == 0 {
		return 0.0
	}

	return totalPriority / float64(pageCount)
}

func (msd *MemoryStateDistribution) restoreFromCheckpoint(checkpoint *MemoryCheckpoint) error {
	// Implementation would restore memory state from checkpoint
	return nil
}

func (msd *MemoryStateDistribution) recoverShardFromReplicas(ctx context.Context, shard *MemoryShard) error {
	// Implementation would recover shard from replicas
	return nil
}

// performDeltaMigration performs delta-based migration with AI optimization
func (msd *MemoryStateDistribution) performDeltaMigration(ctx context.Context, shard *MemoryShard, targetNode string, strategy MigrationStrategy) error {
	// Use AI predictions to determine migration type
	migrationType := msd.selectMigrationTypeWithAI(shard)

	msd.logger.Info("Performing delta-based migration",
		zap.String("shardID", shard.ShardID),
		zap.String("targetNode", targetNode),
		zap.String("migrationType", string(migrationType)))

	// Phase 1: Initial bulk transfer (cold pages first)
	coldPages := msd.filterPagesByTemperature(shard, PageTemperatureCold)
	if err := msd.transferPages(ctx, coldPages, targetNode); err != nil {
		return errors.Wrap(err, "failed to transfer cold pages")
	}

	// Phase 2: Warm pages with delta tracking
	warmPages := msd.filterPagesByTemperature(shard, PageTemperatureWarm)
	if err := msd.transferPagesWithDelta(ctx, warmPages, targetNode); err != nil {
		return errors.Wrap(err, "failed to transfer warm pages")
	}

	// Phase 3: Hot pages with minimal downtime
	hotPages := msd.filterPagesByTemperature(shard, PageTemperatureHot)

	switch migrationType {
	case MigrationTypePreCopy:
		// Pre-copy: Transfer hot pages multiple times
		for i := 0; i < 3; i++ {
			if err := msd.transferPagesWithDelta(ctx, hotPages, targetNode); err != nil {
				return errors.Wrap(err, "failed pre-copy iteration")
			}
			// Brief pause to accumulate new changes
			time.Sleep(100 * time.Millisecond)
		}

	case MigrationTypePostCopy:
		// Post-copy: Transfer minimal set, fetch on demand
		if err := msd.setupPostCopyFetching(ctx, shard, targetNode); err != nil {
			return errors.Wrap(err, "failed to setup post-copy")
		}

	case MigrationTypeHybrid:
		// Hybrid: AI-driven combination
		if err := msd.performHybridMigration(ctx, shard, targetNode, hotPages); err != nil {
			return errors.Wrap(err, "failed hybrid migration")
		}
	}

	// Final delta sync
	if msd.deltaSync != nil {
		if err := msd.deltaSync.FinalSync(ctx, shard.ShardID, targetNode); err != nil {
			return errors.Wrap(err, "failed final delta sync")
		}
	}

	return nil
}

// selectMigrationTypeWithAI uses AI predictions to select optimal migration type
func (msd *MemoryStateDistribution) selectMigrationTypeWithAI(shard *MemoryShard) MigrationType {
	if msd.prefetcher == nil {
		return MigrationTypeHybrid // Default to hybrid
	}

	// Get AI predictions for the shard
	predictions := msd.prefetcher.PredictAccessPatterns(shard.VMID)
	if predictions == nil || len(predictions.PagePredictions) == 0 {
		return MigrationTypeHybrid
	}

	// Analyze predictions to determine best strategy
	var hotPageRatio float64
	var totalPages int
	for _, page := range shard.Pages {
		totalPages++
		if page.Temperature == PageTemperatureHot {
			hotPageRatio++
		}
	}

	if totalPages > 0 {
		hotPageRatio /= float64(totalPages)
	}

	// Decision logic based on AI analysis
	if hotPageRatio > 0.6 {
		// Many hot pages - use post-copy to minimize downtime
		return MigrationTypePostCopy
	} else if hotPageRatio < 0.2 {
		// Few hot pages - use pre-copy for completeness
		return MigrationTypePreCopy
	} else {
		// Mixed workload - use hybrid approach
		return MigrationTypeHybrid
	}
}

// filterPagesByTemperature filters pages by temperature
func (msd *MemoryStateDistribution) filterPagesByTemperature(shard *MemoryShard, temp PageTemperature) []*DistributedMemoryPage {
	var pages []*DistributedMemoryPage
	for _, page := range shard.Pages {
		if page.Temperature == temp {
			pages = append(pages, page)
		}
	}
	return pages
}

// transferPages transfers pages to target node with batching and compression
func (msd *MemoryStateDistribution) transferPages(ctx context.Context, pages []*DistributedMemoryPage, targetNode string) error {
	if len(pages) == 0 {
		return nil
	}

	// Sort pages cold→warm→hot based on temperature and AI predictions
	predictions := msd.getPredictionsForPages(pages)
	sortedPages := msd.sortPagesByPriority(pages, predictions)

	// Batch pages into chunks (1-4MB)
	const batchSizeBytes = 4 * 1024 * 1024 // 4MB batches
	const pageSize = 4096                   // Standard page size

	batches := msd.createPageBatches(sortedPages, batchSizeBytes/pageSize)

	msd.logger.Info("Starting page transfer",
		zap.String("targetNode", targetNode),
		zap.Int("totalPages", len(pages)),
		zap.Int("batches", len(batches)))

	for i, batch := range batches {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Maintain coherence - invalidate pages on other nodes before transfer
		if err := msd.maintainCoherenceOnTransfer(ctx, batch, targetNode); err != nil {
			msd.logger.Warn("Failed to maintain coherence",
				zap.Error(err))
		}

		// Prepare batch metadata
		batchMetadata := &PageBatchMetadata{
			VMID:        batch[0].VMID,
			ShardID:     batch[0].ShardID,
			BatchNumber: uint32(i),
			PageCount:   uint32(len(batch)),
			Pages:       make([]PageMetadata, len(batch)),
		}

		// Compress and prepare pages for transfer
		compressedPages := make([][]byte, 0, len(batch))
		totalBytes := int64(0)

		for j, page := range batch {
			// Compress page data
			compressedData, err := msd.compressPage(page)
			if err != nil {
				msd.logger.Error("Failed to compress page",
					zap.Uint64("pageNumber", page.PageNumber),
					zap.Error(err))
				continue
			}

			compressedPages = append(compressedPages, compressedData)
			totalBytes += int64(len(compressedData))

			// Add page metadata
			batchMetadata.Pages[j] = PageMetadata{
				PageNumber:     page.PageNumber,
				Version:        page.Version,
				Checksum:       page.Checksum,
				CompressedSize: uint32(len(compressedData)),
				Temperature:    page.Temperature,
				AccessCount:    page.AccessCount,
			}

			// Update page state for transfer
			page.LastTransfer = time.Now()
			page.TransferVersion = page.Version
		}

		// Send batch to target node via RPC
		if err := msd.sendPageBatch(ctx, targetNode, batchMetadata, compressedPages); err != nil {
			return errors.Wrapf(err, "failed to send batch %d to %s", i, targetNode)
		}

		// Track metrics
		msd.metrics.BytesSynced.Add(totalBytes)
		msd.metrics.MigrationsCompleted.Inc()

		// Apply bandwidth throttling if needed
		if msd.bandwidthLimiter != nil {
			msd.bandwidthLimiter.Wait(totalBytes)
		}

		// Log progress
		if (i+1)%10 == 0 || i == len(batches)-1 {
			msd.logger.Info("Transfer progress",
				zap.Int("batchesComplete", i+1),
				zap.Int("totalBatches", len(batches)),
				zap.Int64("bytesTransferred", totalBytes))
		}
	}

	msd.logger.Info("Page transfer completed",
		zap.String("targetNode", targetNode),
		zap.Int("pagesTransferred", len(pages)))

	return nil
}

// transferPagesWithDelta transfers pages with delta tracking using WAN delta sync
func (msd *MemoryStateDistribution) transferPagesWithDelta(ctx context.Context, pages []*DistributedMemoryPage, targetNode string) error {
	if len(pages) == 0 {
		return nil
	}

	msd.logger.Info("Starting delta-based page transfer",
		zap.String("targetNode", targetNode),
		zap.Int("pageCount", len(pages)))

	// Get AI predictions for optimal ordering
	predictions := msd.getPredictionsForPages(pages)

	// Convert page numbers to block IDs for delta sync
	pageNumbers := make([]uint64, len(pages))
	for i, page := range pages {
		pageNumbers[i] = page.PageNumber
	}
	blockIDs := msd.convertPageNumbersToBlocks(pageNumbers)

	// Use deltaSync if available
	if msd.deltaSync != nil {
		// Pre-compute deltas for all pages
		if err := msd.deltaSync.PreComputeDeltas(pages[0].VMID, pageNumbers); err != nil {
			msd.logger.Warn("Failed to pre-compute deltas, falling back to full transfer",
				zap.Error(err))
		} else {
			// Sort pages by AI predictions: prioritize lower access probability earlier
			sortedPages := msd.sortPagesByAccessProbability(pages, predictions, false) // ascending order

			// Process pages in priority order
			for i, page := range sortedPages {
				select {
				case <-ctx.Done():
					return ctx.Err()
				default:
				}

				// Create delta for this page
				delta := &MemoryDelta{
					ShardID:   page.ShardID,
					Version:   page.Version,
					Timestamp: time.Now(),
					Pages: []*PageDelta{{
						PageNumber: page.PageNumber,
						Data:       page.Data,
						Checksum:   page.Checksum,
						Version:    page.Version,
					}},
				}

				// Send delta to target node
				if err := msd.deltaSync.SendDelta(ctx, targetNode, delta); err != nil {
					msd.logger.Error("Failed to send delta",
						zap.Uint64("pageNumber", page.PageNumber),
						zap.Error(err))
					// Try full page transfer as fallback
					if err := msd.transferSinglePage(ctx, page, targetNode); err != nil {
						return errors.Wrapf(err, "failed to transfer page %d", page.PageNumber)
					}
				}

				// Update metrics
				msd.metrics.BytesSynced.Add(int64(len(page.Data)))

				// Progress logging
				if (i+1)%100 == 0 || i == len(sortedPages)-1 {
					msd.logger.Debug("Delta transfer progress",
						zap.Int("pagesComplete", i+1),
						zap.Int("totalPages", len(sortedPages)))
				}
			}

			// Start WAN sync for continuous delta updates
			if err := msd.deltaSync.StartWANSync(ctx, pages[0].VMID, targetNode); err != nil {
				msd.logger.Warn("Failed to start WAN sync", zap.Error(err))
			}
		}
	} else {
		// Fallback to standard transfer with compression if deltaSync not available
		msd.logger.Info("DeltaSync not available, using standard compressed transfer")

		// Order pages by temperature and predictions for optimal transfer
		sortedPages := msd.sortPagesByPriority(pages, predictions)

		// Batch and transfer
		const batchSize = 32
		for i := 0; i < len(sortedPages); i += batchSize {
			end := i + batchSize
			if end > len(sortedPages) {
				end = len(sortedPages)
			}

			batch := sortedPages[i:end]
			if err := msd.transferPages(ctx, batch, targetNode); err != nil {
				return errors.Wrapf(err, "failed to transfer batch starting at %d", i)
			}
		}
	}

	// Mark pages as transferred and clear dirty bits
	for _, page := range pages {
		page.LastTransfer = time.Now()
		page.TransferVersion = page.Version

		// Clear dirty bit for this page
		if msd.dirtyBitmap != nil {
			msd.dirtyBitmap.ClearDirty(page.PageNumber)
		}
	}

	msd.logger.Info("Delta-based page transfer completed",
		zap.String("targetNode", targetNode),
		zap.Int("pagesTransferred", len(pages)))

	return nil
}

// setupPostCopyFetching sets up post-copy on-demand fetching with fault handler
func (msd *MemoryStateDistribution) setupPostCopyFetching(ctx context.Context, shard *MemoryShard, targetNode string) error {
	msd.logger.Info("Setting up post-copy fetching",
		zap.String("shardID", shard.ShardID),
		zap.String("targetNode", targetNode))

	// Initialize fault handler for on-demand page fetching
	faultHandler := &PostCopyFaultHandler{
		shardID:      shard.ShardID,
		sourceNode:   msd.nodeID,
		targetNode:   targetNode,
		pageCache:    make(map[uint64]*DistributedMemoryPage),
		pendingFetch: make(map[uint64]chan error),
		msd:          msd,
	}

	// Register fault handler with target node
	if err := msd.registerFaultHandler(ctx, targetNode, shard.ShardID, faultHandler); err != nil {
		return errors.Wrap(err, "failed to register fault handler")
	}

	// Pre-warm predicted hot pages using AI predictions
	if msd.prefetcher != nil {
		predictions := msd.prefetcher.PredictAccessPatterns(shard.VMID)
		if predictions != nil && len(predictions.HotPages) > 0 {
			// Trigger predictive delta sync for hot pages
			if err := msd.prefetcher.TriggerPredictiveDeltaSync(shard.VMID, targetNode); err != nil {
				msd.logger.Warn("Failed to trigger predictive delta sync",
					zap.Error(err))
			}

			// Pre-transfer a small set of predicted hot pages
			hotPageCount := min(len(predictions.HotPages), 100) // Pre-warm up to 100 hot pages
			for i := 0; i < hotPageCount; i++ {
				pageNum := predictions.HotPages[i]
				if page, exists := shard.Pages[pageNum]; exists {
					// Transfer hot page proactively
					if err := msd.transferSinglePage(ctx, page, targetNode); err != nil {
						msd.logger.Warn("Failed to pre-warm hot page",
							zap.Uint64("pageNumber", pageNum),
							zap.Error(err))
					}
				}
			}

			msd.logger.Info("Pre-warmed hot pages",
				zap.Int("hotPageCount", hotPageCount))
		}
	}

	// Set up minimal state transfer (critical pages only)
	criticalPages := msd.identifyCriticalPages(shard)
	if len(criticalPages) > 0 {
		msd.logger.Info("Transferring critical pages for post-copy",
			zap.Int("criticalPageCount", len(criticalPages)))

		// Transfer critical pages immediately
		if err := msd.transferPages(ctx, criticalPages, targetNode); err != nil {
			return errors.Wrap(err, "failed to transfer critical pages")
		}
	}

	// Enable on-demand fetching for remaining pages
	shard.PostCopyMode = true
	shard.FaultHandler = faultHandler

	// Start background thread to handle page faults
	go msd.handlePageFaults(ctx, faultHandler)

	msd.logger.Info("Post-copy fetching setup complete",
		zap.String("shardID", shard.ShardID),
		zap.String("targetNode", targetNode),
		zap.Int("remainingPages", len(shard.Pages)-len(criticalPages)))

	return nil
}

// performHybridMigration performs AI-optimized hybrid migration
func (msd *MemoryStateDistribution) performHybridMigration(ctx context.Context, shard *MemoryShard, targetNode string, hotPages []*DistributedMemoryPage) error {
	msd.logger.Info("Starting hybrid migration",
		zap.String("shardID", shard.ShardID),
		zap.String("targetNode", targetNode),
		zap.Int("hotPageCount", len(hotPages)))

	// Use AI predictions to categorize pages
	var predictions *AccessPredictions
	if msd.prefetcher != nil {
		predictions = msd.prefetcher.PredictAccessPatterns(shard.VMID)
	}

	// Categorize pages based on predictions and temperature
	preCopyPages := make([]*DistributedMemoryPage, 0)
	postCopyPages := make([]*DistributedMemoryPage, 0)
	iterativePages := make([]*DistributedMemoryPage, 0)

	// Group all pages from shard for analysis
	for _, page := range shard.Pages {
		accessProb := msd.getPageAccessProbability(page.PageNumber, predictions)

		if accessProb > 0.8 || page.Temperature == PageTemperatureHot {
			// Very likely to be accessed - defer for post-copy
			postCopyPages = append(postCopyPages, page)
		} else if accessProb > 0.3 || page.Temperature == PageTemperatureWarm {
			// Moderate access chance - iterative pre-copy
			iterativePages = append(iterativePages, page)
		} else {
			// Low access probability - pre-copy once
			preCopyPages = append(preCopyPages, page)
		}
	}

	msd.logger.Info("Hybrid migration page distribution",
		zap.Int("preCopy", len(preCopyPages)),
		zap.Int("iterative", len(iterativePages)),
		zap.Int("postCopy", len(postCopyPages)))

	// Phase 1: Pre-copy cold/low-probability pages
	if len(preCopyPages) > 0 {
		msd.logger.Info("Phase 1: Pre-copying cold pages")
		if err := msd.transferPagesWithDelta(ctx, preCopyPages, targetNode); err != nil {
			return errors.Wrap(err, "failed pre-copy phase")
		}
	}

	// Phase 2: Iterative pre-copy for warm pages with delta sync
	if len(iterativePages) > 0 {
		msd.logger.Info("Phase 2: Iterative pre-copy for warm pages")

		const maxIterations = 3
		const dirtyThreshold = 0.1 // Stop when <10% pages are dirty

		for iteration := 0; iteration < maxIterations; iteration++ {
			// Track dirty pages in this iteration
			dirtyCount := 0

			// Transfer pages with delta tracking
			if err := msd.transferPagesWithDelta(ctx, iterativePages, targetNode); err != nil {
				return errors.Wrapf(err, "failed iteration %d", iteration)
			}

			// Check dirty page ratio
			if msd.dirtyBitmap != nil {
				dirtyPages := msd.dirtyBitmap.GetDirtyPages()
				dirtyCount = len(dirtyPages)

				dirtyRatio := float64(dirtyCount) / float64(len(iterativePages))
				msd.logger.Info("Iteration complete",
					zap.Int("iteration", iteration+1),
					zap.Int("dirtyPages", dirtyCount),
					zap.Float64("dirtyRatio", dirtyRatio))

				// Stop if dirty ratio is below threshold
				if dirtyRatio < dirtyThreshold {
					msd.logger.Info("Dirty page threshold reached, stopping iterations")
					break
				}

				// Re-run delta sync for dirty pages
				if msd.deltaSync != nil {
					if err := msd.deltaSync.StartWANSync(ctx, shard.VMID, targetNode); err != nil {
						msd.logger.Warn("Failed to sync dirty pages", zap.Error(err))
					}
				}
			}

			// Brief pause between iterations
			time.Sleep(200 * time.Millisecond)
		}
	}

	// Phase 3: Setup post-copy for hot pages
	if len(postCopyPages) > 0 {
		msd.logger.Info("Phase 3: Setting up post-copy for hot pages")

		// Create temporary shard with just hot pages for post-copy setup
		hotShard := &MemoryShard{
			ShardID: shard.ShardID,
			VMID:    shard.VMID,
			Pages:   make(map[uint64]*DistributedMemoryPage),
		}
		for _, page := range postCopyPages {
			hotShard.Pages[page.PageNumber] = page
		}

		// Setup post-copy with on-demand fetching
		if err := msd.setupPostCopyFetching(ctx, hotShard, targetNode); err != nil {
			return errors.Wrap(err, "failed to setup post-copy")
		}
	}

	// Phase 4: Final synchronization
	if msd.deltaSync != nil {
		msd.logger.Info("Phase 4: Final synchronization")
		if err := msd.deltaSync.FinalSync(ctx, shard.VMID, targetNode); err != nil {
			msd.logger.Warn("Failed final sync", zap.Error(err))
		}
	}

	// Update migration metrics
	msd.metrics.MigrationsCompleted.Inc()

	msd.logger.Info("Hybrid migration completed successfully",
		zap.String("shardID", shard.ShardID),
		zap.String("targetNode", targetNode))

	return nil
}

// findPage finds a page by number in a shard
func (msd *MemoryStateDistribution) findPage(shard *MemoryShard, pageNumber uint64) *DistributedMemoryPage {
	if page, exists := shard.Pages[pageNumber]; exists {
		return page
	}
	return nil
}

// convertPageNumbersToBlocks converts page numbers to block IDs for delta sync
func (msd *MemoryStateDistribution) convertPageNumbersToBlocks(pageNumbers []uint64) []string {
	blocks := make([]string, len(pageNumbers))
	for i, pageNum := range pageNumbers {
		blocks[i] = fmt.Sprintf("page_%d", pageNum)
	}
	return blocks
}

// Helper functions for page management and transfer

func (msd *MemoryStateDistribution) getPredictionsForPages(pages []*DistributedMemoryPage) *AccessPredictions {
	if msd.prefetcher == nil || len(pages) == 0 {
		return nil
	}

	// Get predictions for the VM
	vmID := pages[0].VMID
	return msd.prefetcher.PredictAccessPatterns(vmID)
}

func (msd *MemoryStateDistribution) sortPagesByPriority(pages []*DistributedMemoryPage, predictions *AccessPredictions) []*DistributedMemoryPage {
	sorted := make([]*DistributedMemoryPage, len(pages))
	copy(sorted, pages)

	// Build priority map from predictions
	priorityMap := make(map[uint64]float64)
	if predictions != nil {
		for _, pred := range predictions.PagePredictions {
			priorityMap[pred.PageNumber] = pred.AccessProbability
		}
	}

	// Sort pages: cold→warm→hot (lower priority first)
	sort.Slice(sorted, func(i, j int) bool {
		// Get priorities
		priI := msd.getPagePriority(sorted[i], priorityMap)
		priJ := msd.getPagePriority(sorted[j], priorityMap)
		return priI < priJ // Lower priority transfers first
	})

	return sorted
}

func (msd *MemoryStateDistribution) sortPagesByAccessProbability(pages []*DistributedMemoryPage, predictions *AccessPredictions, descending bool) []*DistributedMemoryPage {
	sorted := make([]*DistributedMemoryPage, len(pages))
	copy(sorted, pages)

	// Build probability map from predictions
	probMap := make(map[uint64]float64)
	if predictions != nil {
		for _, pred := range predictions.PagePredictions {
			probMap[pred.PageNumber] = pred.AccessProbability
		}
	}

	// Sort by access probability
	sort.Slice(sorted, func(i, j int) bool {
		probI := probMap[sorted[i].PageNumber]
		probJ := probMap[sorted[j].PageNumber]
		if descending {
			return probI > probJ
		}
		return probI < probJ
	})

	return sorted
}

func (msd *MemoryStateDistribution) getPagePriority(page *DistributedMemoryPage, priorityMap map[uint64]float64) float64 {
	priority := 0.0

	// AI prediction weight (70%)
	if aiPriority, exists := priorityMap[page.PageNumber]; exists {
		priority += aiPriority * 0.7
	}

	// Temperature weight (30%)
	switch page.Temperature {
	case PageTemperatureHot:
		priority += 0.3 * 1.0
	case PageTemperatureWarm:
		priority += 0.3 * 0.5
	case PageTemperatureCold:
		priority += 0.3 * 0.1
	}

	return priority
}

func (msd *MemoryStateDistribution) getPageAccessProbability(pageNumber uint64, predictions *AccessPredictions) float64 {
	if predictions == nil {
		return 0.5 // Default middle probability
	}

	for _, pred := range predictions.PagePredictions {
		if pred.PageNumber == pageNumber {
			return pred.AccessProbability
		}
	}

	return 0.5 // Default if no prediction found
}

func (msd *MemoryStateDistribution) createPageBatches(pages []*DistributedMemoryPage, pagesPerBatch int) [][]*DistributedMemoryPage {
	if pagesPerBatch <= 0 {
		pagesPerBatch = 1024 // Default batch size
	}

	batches := make([][]*DistributedMemoryPage, 0)
	for i := 0; i < len(pages); i += pagesPerBatch {
		end := i + pagesPerBatch
		if end > len(pages) {
			end = len(pages)
		}
		batches = append(batches, pages[i:end])
	}

	return batches
}

func (msd *MemoryStateDistribution) sendPageBatch(ctx context.Context, targetNode string, metadata *PageBatchMetadata, compressedPages [][]byte) error {
	// This would send the batch via RPC to the target node
	// For now, log the operation
	msd.logger.Debug("Sending page batch",
		zap.String("targetNode", targetNode),
		zap.String("vmID", metadata.VMID),
		zap.Uint32("batchNumber", metadata.BatchNumber),
		zap.Uint32("pageCount", metadata.PageCount))

	// Simulate network transfer
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(10 * time.Millisecond): // Simulate network latency
		return nil
	}
}

func (msd *MemoryStateDistribution) transferSinglePage(ctx context.Context, page *DistributedMemoryPage, targetNode string) error {
	// Transfer a single page with compression
	compressedData, err := msd.compressPage(page)
	if err != nil {
		return errors.Wrapf(err, "failed to compress page %d", page.PageNumber)
	}

	metadata := &PageBatchMetadata{
		VMID:        page.VMID,
		ShardID:     page.ShardID,
		BatchNumber: 0,
		PageCount:   1,
		Pages: []PageMetadata{{
			PageNumber:     page.PageNumber,
			Version:        page.Version,
			Checksum:       page.Checksum,
			CompressedSize: uint32(len(compressedData)),
			Temperature:    page.Temperature,
			AccessCount:    page.AccessCount,
		}},
	}

	return msd.sendPageBatch(ctx, targetNode, metadata, [][]byte{compressedData})
}

func (msd *MemoryStateDistribution) identifyCriticalPages(shard *MemoryShard) []*DistributedMemoryPage {
	critical := make([]*DistributedMemoryPage, 0)

	// Identify critical pages (e.g., kernel structures, page tables, etc.)
	for pageNum, page := range shard.Pages {
		// Simple heuristic: first 100 pages are often critical
		// In a real system, this would check page attributes
		if pageNum < 100 || page.AccessCount > 1000 {
			critical = append(critical, page)
		}
	}

	return critical
}

func (msd *MemoryStateDistribution) registerFaultHandler(ctx context.Context, targetNode string, shardID string, handler *PostCopyFaultHandler) error {
	// Register the fault handler with the target node
	// This would use RPC in a real implementation
	msd.logger.Info("Registered fault handler",
		zap.String("targetNode", targetNode),
		zap.String("shardID", shardID))
	return nil
}

func (msd *MemoryStateDistribution) handlePageFaults(ctx context.Context, handler *PostCopyFaultHandler) {
	// Background thread to handle page faults
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// In a real system, this would listen for page fault notifications
			// and fetch requested pages on demand
		}
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Coherence maintenance functions

func (msd *MemoryStateDistribution) maintainCoherenceOnTransfer(ctx context.Context, pages []*DistributedMemoryPage, targetNode string) error {
	if msd.coherenceProtocol == nil {
		return nil
	}

	// Invalidate pages on all other nodes to maintain consistency
	for _, page := range pages {
		// Create invalidation message
		invalidationMsg := InvalidationMessage{
			PageNumber: page.PageNumber,
			VMID:       page.VMID,
			ShardID:    page.ShardID,
			Version:    page.Version,
			SourceNode: msd.nodeID,
			TargetNode: targetNode,
			Timestamp:  time.Now(),
		}

		// Send invalidation to coherence protocol
		select {
		case msd.coherenceProtocol.invalidations <- invalidationMsg:
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Channel full, log warning but continue
			msd.logger.Warn("Coherence invalidation channel full",
				zap.Uint64("pageNumber", page.PageNumber))
		}

		// Update page coherence state
		pageKey := fmt.Sprintf("%s:%d", page.VMID, page.PageNumber)
		msd.coherenceProtocol.coherenceMap[pageKey] = CoherenceStateShared

		// Mark page as transferring
		page.Owner = targetNode
		page.Version++
	}

	return nil
}

func (msd *MemoryStateDistribution) updateCoherenceAfterTransfer(ctx context.Context, pages []*DistributedMemoryPage, targetNode string) error {
	if msd.coherenceProtocol == nil {
		return nil
	}

	// Update coherence state after successful transfer
	for _, page := range pages {
		updateMsg := UpdateMessage{
			PageNumber: page.PageNumber,
			VMID:       page.VMID,
			ShardID:    page.ShardID,
			Version:    page.Version,
			NewOwner:   targetNode,
			OldOwner:   msd.nodeID,
			Timestamp:  time.Now(),
		}

		// Send update to coherence protocol
		select {
		case msd.coherenceProtocol.updates <- updateMsg:
		case <-ctx.Done():
			return ctx.Err()
		default:
			msd.logger.Warn("Coherence update channel full",
				zap.Uint64("pageNumber", page.PageNumber))
		}

		// Update local coherence state
		pageKey := fmt.Sprintf("%s:%d", page.VMID, page.PageNumber)
		msd.coherenceProtocol.coherenceMap[pageKey] = CoherenceStateInvalid
	}

	return nil
}

// Enhanced metrics tracking

func (msd *MemoryStateDistribution) trackTransferMetrics(pages []*DistributedMemoryPage, bytesTransferred int64, duration time.Duration) {
	if msd.metrics == nil {
		return
	}

	// Update transfer metrics
	msd.metrics.BytesSynced.Add(bytesTransferred)
	msd.metrics.PagesTransferred.Add(int64(len(pages)))

	// Calculate throughput
	if duration > 0 {
		throughputMBps := float64(bytesTransferred) / (1024 * 1024) / duration.Seconds()
		msd.logger.Info("Transfer performance",
			zap.Int64("bytesTransferred", bytesTransferred),
			zap.Int("pageCount", len(pages)),
			zap.Duration("duration", duration),
			zap.Float64("throughputMBps", throughputMBps))
	}

	// Track page temperature distribution
	hotCount, warmCount, coldCount := 0, 0, 0
	for _, page := range pages {
		switch page.Temperature {
		case PageTemperatureHot:
			hotCount++
		case PageTemperatureWarm:
			warmCount++
		case PageTemperatureCold:
			coldCount++
		}
	}

	msd.logger.Debug("Page temperature distribution",
		zap.Int("hotPages", hotCount),
		zap.Int("warmPages", warmCount),
		zap.Int("coldPages", coldCount))
}

// Supporting types

type MemoryDelta struct {
	ShardID   string
	Version   uint64
	Timestamp time.Time
	Pages     []*PageDelta
}

type PageDelta struct {
	PageNumber uint64
	Data       []byte
	Checksum   [32]byte
	Version    uint64
}

type PageBatchMetadata struct {
	VMID        string
	ShardID     string
	BatchNumber uint32
	PageCount   uint32
	Pages       []PageMetadata
}

type PageMetadata struct {
	PageNumber     uint64
	Version        uint64
	Checksum       [32]byte
	CompressedSize uint32
	Temperature    PageTemperature
	AccessCount    uint32
}

type PostCopyFaultHandler struct {
	shardID      string
	sourceNode   string
	targetNode   string
	pageCache    map[uint64]*DistributedMemoryPage
	pendingFetch map[uint64]chan error
	msd          *MemoryStateDistribution
	mu           sync.RWMutex
}

type BandwidthLimiter struct {
	maxBytesPerSecond int64
	lastTransfer      time.Time
	mu                sync.Mutex
}

func (bl *BandwidthLimiter) Wait(bytes int64) {
	bl.mu.Lock()
	defer bl.mu.Unlock()

	if bl.maxBytesPerSecond <= 0 {
		return // No limit
	}

	// Calculate required delay
	duration := time.Duration(bytes*1e9/bl.maxBytesPerSecond) * time.Nanosecond
	elapsed := time.Since(bl.lastTransfer)

	if elapsed < duration {
		time.Sleep(duration - elapsed)
	}

	bl.lastTransfer = time.Now()
}

type InvalidationMessage struct {
	PageNumber uint64
	VMID       string
	ShardID    string
	Version    uint64
	SourceNode string
	TargetNode string
	Timestamp  time.Time
}

type UpdateMessage struct {
	PageNumber uint64
	VMID       string
	ShardID    string
	Version    uint64
	NewOwner   string
	OldOwner   string
	Timestamp  time.Time
}

type DirtyPageBitmap struct {
	bitmap   []uint64
	size     uint64
	BasePage uint64 // Base page number for shard-relative indexing
	mu       sync.RWMutex
}

func NewDirtyPageBitmap(pageCount uint64) *DirtyPageBitmap {
	bitmapSize := (pageCount + 63) / 64
	return &DirtyPageBitmap{
		bitmap: make([]uint64, bitmapSize),
		size:   pageCount,
	}
}

func (b *DirtyPageBitmap) SetDirty(pageNum uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Convert to shard-relative page number
	relativePageNum := pageNum - b.BasePage
	if relativePageNum >= b.size {
		return
	}

	idx := relativePageNum / 64
	bit := relativePageNum % 64
	b.bitmap[idx] |= (uint64(1) << bit)
}

func (b *DirtyPageBitmap) GetDirtyPages() []uint64 {
	b.mu.RLock()
	defer b.mu.RUnlock()

	dirtyPages := []uint64{}
	for i, word := range b.bitmap {
		if word == 0 {
			continue
		}
		for bit := 0; bit < 64; bit++ {
			if word&(uint64(1)<<uint64(bit)) != 0 {
				relativePageNum := uint64(i*64 + bit)
				if relativePageNum < b.size {
					// Convert back to absolute page number
					absolutePageNum := relativePageNum + b.BasePage
					dirtyPages = append(dirtyPages, absolutePageNum)
				}
			}
		}
	}
	return dirtyPages
}

func (b *DirtyPageBitmap) Clear() {
	b.mu.Lock()
	defer b.mu.Unlock()

	for i := range b.bitmap {
		b.bitmap[i] = 0
	}
}

func (b *DirtyPageBitmap) ClearDirty(pageNum uint64) {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Convert to shard-relative page number
	relativePageNum := pageNum - b.BasePage
	if relativePageNum >= b.size {
		return
	}

	idx := relativePageNum / 64
	bit := relativePageNum % 64
	b.bitmap[idx] &^= (uint64(1) << bit) // Clear the bit
}

// Migration strategy implementations

func (p *PreCopyMigration) Migrate(ctx context.Context, shard *MemoryShard, targetNode string) error {
	// Pre-copy migration implementation
	return nil
}

func (p *PreCopyMigration) GetType() string {
	return "pre-copy"
}

func (p *PostCopyMigration) Migrate(ctx context.Context, shard *MemoryShard, targetNode string) error {
	// Post-copy migration implementation
	return nil
}

func (p *PostCopyMigration) GetType() string {
	return "post-copy"
}

func (h *HybridMigration) Migrate(ctx context.Context, shard *MemoryShard, targetNode string) error {
	// Hybrid migration uses the performHybridMigration implementation
	if h.msd == nil {
		return errors.New("memory state distribution not initialized")
	}

	// Filter hot pages for hybrid approach
	hotPages := h.msd.filterPagesByTemperature(shard, PageTemperatureHot)

	// Execute hybrid migration with AI optimization
	return h.msd.performHybridMigration(ctx, shard, targetNode, hotPages)
}

func (h *HybridMigration) GetType() string {
	return "hybrid"
}

// Supporting component stubs

func NewMemoryCoherenceProtocol() *MemoryCoherenceProtocol {
	return &MemoryCoherenceProtocol{
		coherenceMap:  make(map[string]CoherenceState),
		invalidations: make(chan InvalidationMessage, 100),
		updates:       make(chan UpdateMessage, 100),
	}
}

func NewMemoryCompressionEngine() *MemoryCompressionEngine {
	return &MemoryCompressionEngine{
		deduplicator:    NewMemoryDeduplicator(),
		zeroEliminator:  NewZeroPageEliminator(),
		patternDetector: NewPatternCompressionDetector(),
		compressors:     make(map[string]Compressor),
	}
}

func NewMemoryAccessTracker() *MemoryAccessTracker {
	return &MemoryAccessTracker{
		accessLog:   NewCircularBuffer(10000),
		hotPages:    NewHotPageTracker(),
		coldPages:   NewColdPageTracker(),
		predictions: NewAccessPredictionModel(),
	}
}

func (m *MemoryAccessTracker) TrackAccess(vmID string, pageNumber uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	// Track access implementation
}

func NewMemoryRecoveryManager() *MemoryRecoveryManager {
	return &MemoryRecoveryManager{
		checkpoints:    []*MemoryCheckpoint{},
		replicaManager: NewReplicaManager(),
		recoveryLog:    NewRecoveryLog(),
	}
}

func (m *MemoryRecoveryManager) GetLatestCheckpoint(vmID string) *MemoryCheckpoint {
	// Implementation would return latest checkpoint
	return nil
}

func NewMemoryDistributionMetrics() *MemoryDistributionMetrics {
	return &MemoryDistributionMetrics{}
}

// Stub types
type InvalidationMessage struct {
	PageNumber uint64
	ShardID    string
}

type UpdateMessage struct {
	PageNumber uint64
	ShardID    string
	Data       []byte
}

type PageFaultHandler struct{}
func NewPageFaultHandler() *PageFaultHandler { return &PageFaultHandler{} }

type AIOptimizer struct{}
func NewAIOptimizer() *AIOptimizer { return &AIOptimizer{} }

type MemoryDeduplicator struct{}
func NewMemoryDeduplicator() *MemoryDeduplicator { return &MemoryDeduplicator{} }
func (m *MemoryDeduplicator) FindDuplicate(data []byte) string { return "" }

type ZeroPageEliminator struct{}
func NewZeroPageEliminator() *ZeroPageEliminator { return &ZeroPageEliminator{} }
func (z *ZeroPageEliminator) IsZeroPage(data []byte) bool {
	for _, b := range data {
		if b != 0 {
			return false
		}
	}
	return true
}

type PatternCompressionDetector struct{}
func NewPatternCompressionDetector() *PatternCompressionDetector { return &PatternCompressionDetector{} }
func (p *PatternCompressionDetector) DetectPattern(data []byte) PatternCompressor { return nil }

type PatternCompressor interface {
	Compress([]byte) ([]byte, error)
}

type Compressor interface {
	Compress([]byte) ([]byte, error)
	Decompress([]byte) ([]byte, error)
}

type CircularBuffer struct{}
func NewCircularBuffer(size int) *CircularBuffer { return &CircularBuffer{} }

type HotPageTracker struct{}
func NewHotPageTracker() *HotPageTracker { return &HotPageTracker{} }

type ColdPageTracker struct{}
func NewColdPageTracker() *ColdPageTracker { return &ColdPageTracker{} }

type AccessPredictionModel struct{}
func NewAccessPredictionModel() *AccessPredictionModel { return &AccessPredictionModel{} }

type AccessPredictions struct {
	HotPages []uint64
	ColdPages []uint64
}

type MemoryCheckpoint struct {
	ID        string
	VMID      string
	Timestamp time.Time
	Data      []byte
}

type ReplicaManager struct{}
func NewReplicaManager() *ReplicaManager { return &ReplicaManager{} }

type RecoveryLog struct{}
func NewRecoveryLog() *RecoveryLog { return &RecoveryLog{} }

type MemoryDistributionMetrics struct {
	ShardsCreated        atomic.Int64
	DeltasSynced         atomic.Int64
	BytesSynced          atomic.Int64
	PagesTransferred     atomic.Int64
	MigrationsCompleted  atomic.Int64
	RecoveriesCompleted  atomic.Int64
}

// Helper methods for WAN delta sync integration

// convertPageNumbersToBlocks converts page numbers to block numbers for delta sync
func (msd *MemoryStateDistribution) convertPageNumbersToBlocks(pages []uint64) []uint64 {
	blocks := make([]uint64, 0, len(pages))
	blockMap := make(map[uint64]bool)

	for _, page := range pages {
		// Convert page to block (assuming 8 pages per block)
		block := page / 8
		if !blockMap[block] {
			blockMap[block] = true
			blocks = append(blocks, block)
		}
	}

	return blocks
}

// performDeltaMigration performs delta-based shard migration
func (msd *MemoryStateDistribution) performDeltaMigration(ctx context.Context, shard *MemoryShard, targetNode string, strategy MigrationStrategy) error {
	// First, send delta to target node
	if msd.deltaSync != nil {
		dirtyPages := shard.DirtyBitmap.GetDirtyPages()
		if len(dirtyPages) > 0 {
			delta := &MemoryDelta{
				ShardID:   shard.ShardID,
				Version:   shard.Version,
				Timestamp: time.Now(),
				Pages:     make([]*PageDelta, 0, len(dirtyPages)),
			}

			// Create page deltas for dirty pages
			for _, pageNum := range dirtyPages {
				if page, exists := shard.Pages[pageNum]; exists {
					delta.Pages = append(delta.Pages, &PageDelta{
						PageNumber: pageNum,
						Data:       page.Data,
						Checksum:   page.Checksum,
						Version:    page.Version,
					})
				}
			}

			// Send delta via WAN sync
			if err := msd.deltaSync.SendDelta(ctx, targetNode, delta); err != nil {
				return errors.Wrap(err, "failed to send delta")
			}
		}
	}

	// Then perform regular migration
	return strategy.Migrate(ctx, shard, targetNode)
}

// getAvailableNodesFromFederation gets available nodes from the federation manager
func (msd *MemoryStateDistribution) getAvailableNodesFromFederation() []string {
	if msd.shardingManager == nil || msd.shardingManager.federation == nil {
		// No federation available, return empty
		return []string{}
	}

	// Get clusters from federation
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	clusters, err := msd.shardingManager.federation.ListClusters(ctx)
	if err != nil {
		msd.logger.Warn("Failed to list clusters from federation", zap.Error(err))
		return []string{}
	}

	// Extract node IDs from connected clusters
	nodeIDs := make([]string, 0, len(clusters))
	for _, cluster := range clusters {
		if cluster.State == federation.ConnectedState {
			nodeIDs = append(nodeIDs, cluster.ID)
		}
	}

	return nodeIDs
}

// WANMigrationDeltaSync interface for WAN delta synchronization
type WANMigrationDeltaSync interface {
	PreComputeDeltas(vmID string, priorityBlocks []uint64) error
	StartWANSync(ctx context.Context, vmID, targetNode string) error
	FinalizeSync(ctx context.Context, vmID, targetNode string) error
	FinalSync(ctx context.Context, vmID, targetNode string) error
	SendDelta(ctx context.Context, targetNode string, delta *MemoryDelta) error
}