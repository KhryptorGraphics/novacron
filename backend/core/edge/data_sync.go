// Package edge provides edge data synchronization capabilities
package edge

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// SyncMode represents the synchronization mode
type SyncMode string

const (
	SyncModeFull        SyncMode = "full"
	SyncModeDelta       SyncMode = "delta"
	SyncModeIncremental SyncMode = "incremental"
	SyncModeEventDriven SyncMode = "event_driven"
	SyncModeBidirectional SyncMode = "bidirectional"
)

// SyncDirection represents the sync direction
type SyncDirection string

const (
	SyncDirectionPush SyncDirection = "push"
	SyncDirectionPull SyncDirection = "pull"
	SyncDirectionBoth SyncDirection = "both"
)

// ConflictResolution represents conflict resolution strategies
type ConflictResolution string

const (
	ConflictResolutionLatestWins     ConflictResolution = "latest_wins"
	ConflictResolutionSourceWins     ConflictResolution = "source_wins"
	ConflictResolutionTargetWins     ConflictResolution = "target_wins"
	ConflictResolutionMerge          ConflictResolution = "merge"
	ConflictResolutionManual         ConflictResolution = "manual"
	ConflictResolutionVersionVector  ConflictResolution = "version_vector"
)

// DataObject represents a data object to be synchronized
type DataObject struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Version      int64                  `json:"version"`
	Checksum     string                 `json:"checksum"`
	Size         uint64                 `json:"size"`
	Content      []byte                 `json:"content,omitempty"`
	Metadata     map[string]interface{} `json:"metadata"`
	Timestamp    time.Time              `json:"timestamp"`
	Location     string                 `json:"location"`
	Replicas     []string               `json:"replicas"`
	VersionVector map[string]int64      `json:"version_vector,omitempty"`
}

// SyncOperation represents a sync operation
type SyncOperation struct {
	ID            string         `json:"id"`
	Type          OperationType  `json:"type"`
	Object        *DataObject    `json:"object"`
	SourceNode    string         `json:"source_node"`
	TargetNode    string         `json:"target_node"`
	Status        SyncStatus     `json:"status"`
	StartTime     time.Time      `json:"start_time"`
	EndTime       *time.Time     `json:"end_time,omitempty"`
	BytesTransferred uint64      `json:"bytes_transferred"`
	Attempts      int            `json:"attempts"`
	Error         string         `json:"error,omitempty"`
}

// OperationType represents the type of sync operation
type OperationType string

const (
	OperationTypeCreate OperationType = "create"
	OperationTypeUpdate OperationType = "update"
	OperationTypeDelete OperationType = "delete"
	OperationTypeMerge  OperationType = "merge"
)

// SyncStatus represents the status of a sync operation
type SyncStatus string

const (
	SyncStatusPending    SyncStatus = "pending"
	SyncStatusInProgress SyncStatus = "in_progress"
	SyncStatusCompleted  SyncStatus = "completed"
	SyncStatusFailed     SyncStatus = "failed"
	SyncStatusConflict   SyncStatus = "conflict"
)

// DeltaSync represents a delta synchronization
type DeltaSync struct {
	BaseVersion   int64      `json:"base_version"`
	TargetVersion int64      `json:"target_version"`
	Deltas        []Delta    `json:"deltas"`
	Checksum      string     `json:"checksum"`
}

// Delta represents a change delta
type Delta struct {
	Type      DeltaType `json:"type"`
	Offset    int64     `json:"offset"`
	Length    int64     `json:"length"`
	Data      []byte    `json:"data,omitempty"`
	OldData   []byte    `json:"old_data,omitempty"`
}

// DeltaType represents the type of delta
type DeltaType string

const (
	DeltaTypeInsert  DeltaType = "insert"
	DeltaTypeDelete  DeltaType = "delete"
	DeltaTypeReplace DeltaType = "replace"
)

// DataSyncManager manages edge-to-core data synchronization
type DataSyncManager struct {
	nodeManager      *NodeManager
	storage          DataStorage
	deltaEngine      *DeltaEngine
	conflictResolver *ConflictResolver
	scheduler        *SyncScheduler
	transport        DataTransport
	metrics          *SyncMetrics
	config           *SyncConfig
	operations       sync.Map // map[string]*SyncOperation
	objectVersions   sync.Map // map[string]map[string]int64 (objectID -> nodeID -> version)
	ctx              context.Context
	cancel           context.CancelFunc
	wg               sync.WaitGroup
	mu               sync.RWMutex
}

// SyncConfig contains synchronization configuration
type SyncConfig struct {
	Mode                SyncMode
	Direction           SyncDirection
	ConflictResolution  ConflictResolution
	BatchSize           int
	MaxConcurrent       int
	RetryAttempts       int
	RetryDelay          time.Duration
	SyncInterval        time.Duration
	DeltaThreshold      float64
	CompressionEnabled  bool
	EncryptionEnabled   bool
	BandwidthLimit      float64 // Mbps
	PriorityLevels      int
}

// DataStorage interface for data storage
type DataStorage interface {
	Get(id string) (*DataObject, error)
	Put(obj *DataObject) error
	Delete(id string) error
	List(filter map[string]interface{}) ([]*DataObject, error)
	GetVersion(id string, version int64) (*DataObject, error)
}

// DataTransport interface for data transport
type DataTransport interface {
	Send(data []byte, target string) error
	Receive(source string) ([]byte, error)
	StreamSend(reader io.Reader, target string) error
	StreamReceive(source string, writer io.Writer) error
}

// DeltaEngine calculates and applies deltas
type DeltaEngine struct {
	algorithm    DeltaAlgorithm
	compression  CompressionAlgorithm
	chunkSize    int
	mu           sync.RWMutex
}

// DeltaAlgorithm represents a delta calculation algorithm
type DeltaAlgorithm interface {
	Calculate(old, new []byte) ([]Delta, error)
	Apply(base []byte, deltas []Delta) ([]byte, error)
}

// CompressionAlgorithm represents a compression algorithm
type CompressionAlgorithm interface {
	Compress(data []byte) ([]byte, error)
	Decompress(data []byte) ([]byte, error)
}

// ConflictResolver resolves synchronization conflicts
type ConflictResolver struct {
	strategy     ConflictResolution
	mergeFunc    MergeFunction
	versionStore VersionStore
	mu           sync.RWMutex
}

// MergeFunction defines a function for merging conflicting data
type MergeFunction func(local, remote *DataObject) (*DataObject, error)

// VersionStore stores version information
type VersionStore interface {
	GetVector(objectID string) (map[string]int64, error)
	UpdateVector(objectID string, nodeID string, version int64) error
	CompareVectors(v1, v2 map[string]int64) VectorComparison
}

// VectorComparison represents the result of version vector comparison
type VectorComparison string

const (
	VectorEqual      VectorComparison = "equal"
	VectorGreater    VectorComparison = "greater"
	VectorLess       VectorComparison = "less"
	VectorConcurrent VectorComparison = "concurrent"
)

// SyncScheduler schedules sync operations
type SyncScheduler struct {
	queue          *PriorityQueue
	bandwidthMgr   *BandwidthManager
	activeOps      sync.Map
	maxConcurrent  int
	mu             sync.RWMutex
}

// BandwidthManager manages bandwidth allocation
type BandwidthManager struct {
	totalBandwidth float64
	allocated      float64
	allocations    sync.Map // map[string]float64
	mu             sync.RWMutex
}

// SyncMetrics tracks synchronization metrics
type SyncMetrics struct {
	syncOperations    *prometheus.CounterVec
	syncDuration      *prometheus.HistogramVec
	bytesTransferred  *prometheus.CounterVec
	conflictCount     prometheus.Counter
	deltaEfficiency   prometheus.Gauge
	queueLength       prometheus.Gauge
	bandwidthUsage    prometheus.Gauge
}

// NewDataSyncManager creates a new data sync manager
func NewDataSyncManager(nodeManager *NodeManager, config *SyncConfig) *DataSyncManager {
	ctx, cancel := context.WithCancel(context.Background())

	dsm := &DataSyncManager{
		nodeManager:      nodeManager,
		deltaEngine:      NewDeltaEngine(),
		conflictResolver: NewConflictResolver(config.ConflictResolution),
		scheduler:        NewSyncScheduler(config.MaxConcurrent),
		metrics:          NewSyncMetrics(),
		config:           config,
		ctx:              ctx,
		cancel:           cancel,
	}

	// Start sync workers
	dsm.wg.Add(3)
	go dsm.syncWorker()
	go dsm.schedulerWorker()
	go dsm.conflictWorker()

	return dsm
}

// NewDeltaEngine creates a new delta engine
func NewDeltaEngine() *DeltaEngine {
	return &DeltaEngine{
		algorithm: &RollingHashDelta{},
		chunkSize: 4096,
	}
}

// NewConflictResolver creates a new conflict resolver
func NewConflictResolver(strategy ConflictResolution) *ConflictResolver {
	return &ConflictResolver{
		strategy: strategy,
	}
}

// NewSyncScheduler creates a new sync scheduler
func NewSyncScheduler(maxConcurrent int) *SyncScheduler {
	return &SyncScheduler{
		queue:          NewPriorityQueue(),
		bandwidthMgr:   NewBandwidthManager(1000.0), // 1Gbps default
		maxConcurrent:  maxConcurrent,
	}
}

// NewBandwidthManager creates a new bandwidth manager
func NewBandwidthManager(totalBandwidth float64) *BandwidthManager {
	return &BandwidthManager{
		totalBandwidth: totalBandwidth,
	}
}

// NewSyncMetrics creates new sync metrics
func NewSyncMetrics() *SyncMetrics {
	return &SyncMetrics{
		syncOperations: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "edge_sync_operations_total",
				Help: "Total number of sync operations",
			},
			[]string{"type", "status"},
		),
		syncDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "edge_sync_duration_seconds",
				Help:    "Duration of sync operations",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"type", "mode"},
		),
		bytesTransferred: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "edge_sync_bytes_total",
				Help: "Total bytes transferred during sync",
			},
			[]string{"direction"},
		),
		conflictCount: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_sync_conflicts_total",
				Help: "Total number of sync conflicts",
			},
		),
		deltaEfficiency: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_sync_delta_efficiency",
				Help: "Delta sync efficiency ratio",
			},
		),
		queueLength: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_sync_queue_length",
				Help: "Number of pending sync operations",
			},
		),
		bandwidthUsage: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_sync_bandwidth_usage_mbps",
				Help: "Current bandwidth usage in Mbps",
			},
		),
	}
}

// SyncData synchronizes data between edge and core
func (dsm *DataSyncManager) SyncData(objectID string, targetNode string) error {
	// Get object from storage
	obj, err := dsm.storage.Get(objectID)
	if err != nil {
		return fmt.Errorf("failed to get object: %w", err)
	}

	// Create sync operation
	op := &SyncOperation{
		ID:         fmt.Sprintf("sync-%d", time.Now().UnixNano()),
		Type:       OperationTypeUpdate,
		Object:     obj,
		TargetNode: targetNode,
		Status:     SyncStatusPending,
		StartTime:  time.Now(),
	}

	// Store operation
	dsm.operations.Store(op.ID, op)

	// Schedule sync
	dsm.scheduler.Schedule(op)

	// Update metrics
	dsm.metrics.queueLength.Set(float64(dsm.scheduler.queue.Len()))

	return nil
}

// PerformDeltaSync performs delta synchronization
func (dsm *DataSyncManager) PerformDeltaSync(objectID string, sourceNode, targetNode string) error {
	start := time.Now()
	defer func() {
		dsm.metrics.syncDuration.WithLabelValues("update", "delta").Observe(time.Since(start).Seconds())
	}()

	// Get current version from source
	current, err := dsm.storage.Get(objectID)
	if err != nil {
		return err
	}

	// Get base version from target
	targetVersion := dsm.getTargetVersion(objectID, targetNode)
	base, err := dsm.storage.GetVersion(objectID, targetVersion)
	if err != nil {
		// Fall back to full sync
		return dsm.performFullSync(current, targetNode)
	}

	// Calculate deltas
	deltas, err := dsm.deltaEngine.Calculate(base.Content, current.Content)
	if err != nil {
		return err
	}

	// Check if delta is efficient
	deltaSize := dsm.calculateDeltaSize(deltas)
	efficiency := float64(deltaSize) / float64(len(current.Content))
	dsm.metrics.deltaEfficiency.Set(1.0 - efficiency)

	if efficiency > dsm.config.DeltaThreshold {
		// Delta not efficient, do full sync
		return dsm.performFullSync(current, targetNode)
	}

	// Create delta sync object
	deltaSync := &DeltaSync{
		BaseVersion:   targetVersion,
		TargetVersion: current.Version,
		Deltas:        deltas,
		Checksum:      dsm.calculateChecksum(current.Content),
	}

	// Send delta
	if err := dsm.sendDelta(deltaSync, targetNode); err != nil {
		return err
	}

	// Update version tracking
	dsm.updateVersionTracking(objectID, targetNode, current.Version)

	// Update metrics
	dsm.metrics.bytesTransferred.WithLabelValues("push").Add(float64(deltaSize))

	return nil
}

// ResolveConflict resolves a synchronization conflict
func (dsm *DataSyncManager) ResolveConflict(localObj, remoteObj *DataObject) (*DataObject, error) {
	dsm.metrics.conflictCount.Inc()

	resolved, err := dsm.conflictResolver.Resolve(localObj, remoteObj)
	if err != nil {
		return nil, err
	}

	// Store resolved version
	if err := dsm.storage.Put(resolved); err != nil {
		return nil, err
	}

	return resolved, nil
}

// Calculate methods for DeltaEngine

func (de *DeltaEngine) Calculate(old, new []byte) ([]Delta, error) {
	return de.algorithm.Calculate(old, new)
}

func (de *DeltaEngine) Apply(base []byte, deltas []Delta) ([]byte, error) {
	return de.algorithm.Apply(base, deltas)
}

// RollingHashDelta implements delta calculation using rolling hash
type RollingHashDelta struct {
	windowSize int
}

func (r *RollingHashDelta) Calculate(old, new []byte) ([]Delta, error) {
	deltas := []Delta{}

	// Simple diff algorithm (would use more sophisticated in production)
	if bytes.Equal(old, new) {
		return deltas, nil
	}

	// Find common prefix
	commonPrefix := 0
	for i := 0; i < len(old) && i < len(new); i++ {
		if old[i] != new[i] {
			break
		}
		commonPrefix++
	}

	// Find common suffix
	commonSuffix := 0
	for i := 0; i < len(old)-commonPrefix && i < len(new)-commonPrefix; i++ {
		if old[len(old)-1-i] != new[len(new)-1-i] {
			break
		}
		commonSuffix++
	}

	// Create delta for middle part
	if commonPrefix < len(new)-commonSuffix {
		delta := Delta{
			Type:   DeltaTypeReplace,
			Offset: int64(commonPrefix),
			Length: int64(len(old) - commonPrefix - commonSuffix),
			Data:   new[commonPrefix : len(new)-commonSuffix],
		}
		if commonPrefix < len(old)-commonSuffix {
			delta.OldData = old[commonPrefix : len(old)-commonSuffix]
		}
		deltas = append(deltas, delta)
	}

	return deltas, nil
}

func (r *RollingHashDelta) Apply(base []byte, deltas []Delta) ([]byte, error) {
	result := make([]byte, 0, len(base))
	offset := int64(0)

	for _, delta := range deltas {
		// Copy unchanged part
		if delta.Offset > offset {
			result = append(result, base[offset:delta.Offset]...)
		}

		// Apply delta
		switch delta.Type {
		case DeltaTypeInsert:
			result = append(result, delta.Data...)
		case DeltaTypeDelete:
			offset = delta.Offset + delta.Length
		case DeltaTypeReplace:
			result = append(result, delta.Data...)
			offset = delta.Offset + delta.Length
		}
	}

	// Copy remaining
	if offset < int64(len(base)) {
		result = append(result, base[offset:]...)
	}

	return result, nil
}

// Conflict resolution methods

func (cr *ConflictResolver) Resolve(local, remote *DataObject) (*DataObject, error) {
	switch cr.strategy {
	case ConflictResolutionLatestWins:
		if local.Timestamp.After(remote.Timestamp) {
			return local, nil
		}
		return remote, nil

	case ConflictResolutionSourceWins:
		return local, nil

	case ConflictResolutionTargetWins:
		return remote, nil

	case ConflictResolutionVersionVector:
		return cr.resolveWithVersionVector(local, remote)

	case ConflictResolutionMerge:
		if cr.mergeFunc != nil {
			return cr.mergeFunc(local, remote)
		}
		return nil, fmt.Errorf("merge function not defined")

	case ConflictResolutionManual:
		return nil, fmt.Errorf("manual resolution required")

	default:
		return nil, fmt.Errorf("unknown conflict resolution strategy")
	}
}

func (cr *ConflictResolver) resolveWithVersionVector(local, remote *DataObject) (*DataObject, error) {
	comparison := cr.versionStore.CompareVectors(local.VersionVector, remote.VersionVector)

	switch comparison {
	case VectorEqual:
		return local, nil
	case VectorGreater:
		return local, nil
	case VectorLess:
		return remote, nil
	case VectorConcurrent:
		// Concurrent changes, need merge
		if cr.mergeFunc != nil {
			return cr.mergeFunc(local, remote)
		}
		// Fall back to latest wins
		if local.Timestamp.After(remote.Timestamp) {
			return local, nil
		}
		return remote, nil
	}

	return nil, fmt.Errorf("unable to resolve conflict")
}

// Bandwidth management methods

func (bm *BandwidthManager) Allocate(opID string, required float64) bool {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if bm.allocated+required > bm.totalBandwidth {
		return false
	}

	bm.allocated += required
	bm.allocations.Store(opID, required)
	return true
}

func (bm *BandwidthManager) Release(opID string) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if allocation, exists := bm.allocations.Load(opID); exists {
		bm.allocated -= allocation.(float64)
		bm.allocations.Delete(opID)
	}
}

// Sync scheduler methods

func (ss *SyncScheduler) Schedule(op *SyncOperation) {
	ss.queue.Push(&EdgeWorkload{
		ID:       op.ID,
		Priority: ss.calculatePriority(op),
	})
}

func (ss *SyncScheduler) calculatePriority(op *SyncOperation) WorkloadPriority {
	// Calculate priority based on operation type and data size
	if op.Type == OperationTypeDelete {
		return PriorityHigh
	}
	if op.Object.Size > 100*1024*1024 { // Large objects
		return PriorityLow
	}
	return PriorityMedium
}

// Worker loops

func (dsm *DataSyncManager) syncWorker() {
	defer dsm.wg.Done()

	ticker := time.NewTicker(dsm.config.SyncInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dsm.processSyncQueue()
		case <-dsm.ctx.Done():
			return
		}
	}
}

func (dsm *DataSyncManager) schedulerWorker() {
	defer dsm.wg.Done()

	for {
		select {
		case <-dsm.ctx.Done():
			return
		default:
			if dsm.scheduler.queue.Len() > 0 {
				dsm.processNextSync()
			} else {
				time.Sleep(100 * time.Millisecond)
			}
		}
	}
}

func (dsm *DataSyncManager) conflictWorker() {
	defer dsm.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dsm.checkForConflicts()
		case <-dsm.ctx.Done():
			return
		}
	}
}

// Helper methods

func (dsm *DataSyncManager) performFullSync(obj *DataObject, targetNode string) error {
	// Compress if enabled
	data := obj.Content
	if dsm.config.CompressionEnabled {
		compressed, err := dsm.compress(data)
		if err == nil {
			data = compressed
		}
	}

	// Send data
	if err := dsm.transport.Send(data, targetNode); err != nil {
		return err
	}

	// Update version tracking
	dsm.updateVersionTracking(obj.ID, targetNode, obj.Version)

	// Update metrics
	dsm.metrics.bytesTransferred.WithLabelValues("push").Add(float64(len(data)))

	return nil
}

func (dsm *DataSyncManager) sendDelta(delta *DeltaSync, targetNode string) error {
	// Serialize delta
	data, err := dsm.serializeDelta(delta)
	if err != nil {
		return err
	}

	// Send to target
	return dsm.transport.Send(data, targetNode)
}

func (dsm *DataSyncManager) getTargetVersion(objectID, targetNode string) int64 {
	versionsInterface, exists := dsm.objectVersions.Load(objectID)
	if !exists {
		return 0
	}

	versions := versionsInterface.(map[string]int64)
	if version, exists := versions[targetNode]; exists {
		return version
	}

	return 0
}

func (dsm *DataSyncManager) updateVersionTracking(objectID, nodeID string, version int64) {
	versionsInterface, _ := dsm.objectVersions.LoadOrStore(objectID, make(map[string]int64))
	versions := versionsInterface.(map[string]int64)
	versions[nodeID] = version
	dsm.objectVersions.Store(objectID, versions)
}

func (dsm *DataSyncManager) calculateDeltaSize(deltas []Delta) int {
	size := 0
	for _, delta := range deltas {
		size += 16 // Metadata
		size += len(delta.Data)
	}
	return size
}

func (dsm *DataSyncManager) calculateChecksum(data []byte) string {
	hash := md5.Sum(data)
	return hex.EncodeToString(hash[:])
}

func (dsm *DataSyncManager) serializeDelta(delta *DeltaSync) ([]byte, error) {
	// Simple serialization (would use protobuf in production)
	buf := new(bytes.Buffer)

	binary.Write(buf, binary.LittleEndian, delta.BaseVersion)
	binary.Write(buf, binary.LittleEndian, delta.TargetVersion)
	binary.Write(buf, binary.LittleEndian, int32(len(delta.Deltas)))

	for _, d := range delta.Deltas {
		buf.WriteString(string(d.Type))
		binary.Write(buf, binary.LittleEndian, d.Offset)
		binary.Write(buf, binary.LittleEndian, d.Length)
		binary.Write(buf, binary.LittleEndian, int32(len(d.Data)))
		buf.Write(d.Data)
	}

	buf.WriteString(delta.Checksum)

	return buf.Bytes(), nil
}

func (dsm *DataSyncManager) compress(data []byte) ([]byte, error) {
	// Placeholder for compression
	return data, nil
}

func (dsm *DataSyncManager) processSyncQueue() {
	// Process pending sync operations
	count := 0
	dsm.operations.Range(func(key, value interface{}) bool {
		if count >= dsm.config.MaxConcurrent {
			return false
		}

		op := value.(*SyncOperation)
		if op.Status == SyncStatusPending {
			go dsm.executeSyncOperation(op)
			count++
		}

		return true
	})
}

func (dsm *DataSyncManager) processNextSync() {
	// Get next operation from queue
	workload := dsm.scheduler.queue.Pop()
	if workload == nil {
		return
	}

	opInterface, exists := dsm.operations.Load(workload.ID)
	if !exists {
		return
	}

	op := opInterface.(*SyncOperation)
	dsm.executeSyncOperation(op)
}

func (dsm *DataSyncManager) executeSyncOperation(op *SyncOperation) {
	op.Status = SyncStatusInProgress
	op.Attempts++

	// Allocate bandwidth
	requiredBandwidth := float64(op.Object.Size) * 8 / 1000000 // Convert to Mbps
	if !dsm.scheduler.bandwidthMgr.Allocate(op.ID, requiredBandwidth) {
		// Retry later
		op.Status = SyncStatusPending
		return
	}
	defer dsm.scheduler.bandwidthMgr.Release(op.ID)

	// Perform sync based on mode
	var err error
	switch dsm.config.Mode {
	case SyncModeDelta:
		err = dsm.PerformDeltaSync(op.Object.ID, op.SourceNode, op.TargetNode)
	case SyncModeFull:
		err = dsm.performFullSync(op.Object, op.TargetNode)
	default:
		err = fmt.Errorf("unsupported sync mode: %s", dsm.config.Mode)
	}

	// Update operation status
	if err != nil {
		op.Status = SyncStatusFailed
		op.Error = err.Error()

		// Retry if attempts remaining
		if op.Attempts < dsm.config.RetryAttempts {
			time.Sleep(dsm.config.RetryDelay)
			op.Status = SyncStatusPending
		}
	} else {
		op.Status = SyncStatusCompleted
		now := time.Now()
		op.EndTime = &now
	}

	// Update metrics
	dsm.metrics.syncOperations.WithLabelValues(string(op.Type), string(op.Status)).Inc()
}

func (dsm *DataSyncManager) checkForConflicts() {
	// Check for version conflicts across nodes
	dsm.objectVersions.Range(func(key, value interface{}) bool {
		objectID := key.(string)
		versions := value.(map[string]int64)

		// Check for divergent versions
		var maxVersion int64
		conflictNodes := []string{}

		for nodeID, version := range versions {
			if version > maxVersion {
				maxVersion = version
			}
			if version < maxVersion {
				conflictNodes = append(conflictNodes, nodeID)
			}
		}

		// Resolve conflicts if found
		if len(conflictNodes) > 0 {
			go dsm.resolveVersionConflicts(objectID, conflictNodes)
		}

		return true
	})
}

func (dsm *DataSyncManager) resolveVersionConflicts(objectID string, nodes []string) {
	// Get latest version
	latest, err := dsm.storage.Get(objectID)
	if err != nil {
		return
	}

	// Sync to outdated nodes
	for _, nodeID := range nodes {
		dsm.SyncData(objectID, nodeID)
	}
}

// Stop stops the data sync manager
func (dsm *DataSyncManager) Stop() {
	dsm.cancel()
	dsm.wg.Wait()
}