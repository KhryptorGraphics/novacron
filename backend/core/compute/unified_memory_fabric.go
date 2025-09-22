package compute

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// UnifiedMemoryFabric provides distributed memory aggregation across clusters
type UnifiedMemoryFabric struct {
	mu                     sync.RWMutex
	federationMgr          *federation.FederationManager
	memoryStateDistributor *vm.MemoryStateDistribution
	vmStateShardingMgr     *vm.VMStateShardingManager
	memoryPools            map[string]*MemoryPool
	memoryRegions          map[string]*MemoryRegion
	allocationEngine       *MemoryAllocationEngine
	coherenceProtocol      *MemoryCoherenceProtocol
	cacheManager           *DistributedCacheManager
	performanceOptimizer   *MemoryPerformanceOptimizer
	processingFabric       *DistributedProcessingFabric
	topologyManager        *MemoryTopologyManager
	metadataService        *MemoryMetadataService
	metrics                *MemoryFabricMetrics
}

// MemoryPool represents a pool of memory resources across clusters
type MemoryPool struct {
	ID                string                   `json:"id"`
	Name              string                   `json:"name"`
	Type              MemoryPoolType           `json:"type"`
	TotalCapacity     int64                    `json:"total_capacity_bytes"`
	AvailableCapacity int64                    `json:"available_capacity_bytes"`
	ClusterNodes      map[string]*MemoryNode   `json:"cluster_nodes"`
	Characteristics   *MemoryCharacteristics   `json:"characteristics"`
	AccessPatterns    []MemoryAccessPattern    `json:"access_patterns"`
	LoadBalancing     *MemoryLoadBalancing     `json:"load_balancing"`
	ReplicationPolicy *MemoryReplicationPolicy `json:"replication_policy"`
	CoherencePolicy   MemoryCoherencePolicy    `json:"coherence_policy"`
	Health            MemoryPoolHealth         `json:"health"`
	LastUpdated       time.Time                `json:"last_updated"`
	Metrics           *MemoryPoolMetrics       `json:"metrics"`
}

type MemoryPoolType string

const (
	PoolTypeHighPerformance MemoryPoolType = "high_performance"
	PoolTypeHighCapacity    MemoryPoolType = "high_capacity"
	PoolTypePersistent      MemoryPoolType = "persistent"
	PoolTypeCache           MemoryPoolType = "cache"
	PoolTypeShared          MemoryPoolType = "shared"
	PoolTypeLocal           MemoryPoolType = "local"
	PoolTypeNVRAM           MemoryPoolType = "nvram"
)

// MemoryNode represents a memory node in a cluster
type MemoryNode struct {
	NodeID          string                 `json:"node_id"`
	ClusterID       string                 `json:"cluster_id"`
	IPAddress       string                 `json:"ip_address"`
	Port            int                    `json:"port"`
	Capacity        int64                  `json:"capacity_bytes"`
	Available       int64                  `json:"available_bytes"`
	Allocated       int64                  `json:"allocated_bytes"`
	MemoryType      MemoryType             `json:"memory_type"`
	AccessLatency   time.Duration          `json:"access_latency"`
	Bandwidth       float64                `json:"bandwidth_gbps"`
	Health          NodeHealth             `json:"health"`
	Topology        *NodeTopology          `json:"topology"`
	Capabilities    []MemoryCapability     `json:"capabilities"`
	SecurityContext *MemorySecurityContext `json:"security_context"`
	Performance     *NodePerformance       `json:"performance"`
	LastSeen        time.Time              `json:"last_seen"`
}

type MemoryType string

const (
	MemoryTypeDRAM    MemoryType = "dram"
	MemoryTypeHBM     MemoryType = "hbm"
	MemoryTypeNVRAM   MemoryType = "nvram"
	MemoryTypeStorage MemoryType = "storage"
	MemoryTypeRemote  MemoryType = "remote"
)

type NodeHealth string

const (
	HealthHealthy  NodeHealth = "healthy"
	HealthWarning  NodeHealth = "warning"
	HealthCritical NodeHealth = "critical"
	HealthOffline  NodeHealth = "offline"
)

type MemoryCapability string

const (
	CapabilityCompression   MemoryCapability = "compression"
	CapabilityEncryption    MemoryCapability = "encryption"
	CapabilityDeduplication MemoryCapability = "deduplication"
	CapabilityPrefetching   MemoryCapability = "prefetching"
	CapabilityMigration     MemoryCapability = "migration"
	CapabilityReplication   MemoryCapability = "replication"
)

// MemoryCharacteristics defines memory pool characteristics
type MemoryCharacteristics struct {
	AccessLatency       time.Duration    `json:"access_latency"`
	Bandwidth           float64          `json:"bandwidth_gbps"`
	Durability          float64          `json:"durability"`
	Consistency         ConsistencyLevel `json:"consistency"`
	Persistence         bool             `json:"persistence"`
	CompressionRatio    float64          `json:"compression_ratio"`
	EncryptionSupported bool             `json:"encryption_supported"`
	ReplicationFactor   int              `json:"replication_factor"`
	ErrorCorrection     bool             `json:"error_correction"`
}

type ConsistencyLevel string

const (
	ConsistencyStrong    ConsistencyLevel = "strong"
	ConsistencyEventual  ConsistencyLevel = "eventual"
	ConsistencyWeak      ConsistencyLevel = "weak"
	ConsistencyCausal    ConsistencyLevel = "causal"
	ConsistencyMonotonic ConsistencyLevel = "monotonic"
)

// MemoryRegion represents a logical memory region across clusters
type MemoryRegion struct {
	ID               string                    `json:"id"`
	Name             string                    `json:"name"`
	Size             int64                     `json:"size_bytes"`
	BaseAddress      uint64                    `json:"base_address"`
	AccessMode       MemoryAccessMode          `json:"access_mode"`
	CoherencePolicy  MemoryCoherencePolicy     `json:"coherence_policy"`
	ConsistencyLevel ConsistencyLevel          `json:"consistency_level"`
	Allocations      []MemoryAllocation        `json:"allocations"`
	Mappings         map[string]*MemoryMapping `json:"mappings"`
	AccessControl    *MemoryAccessControl      `json:"access_control"`
	Performance      *RegionPerformance        `json:"performance"`
	Health           RegionHealth              `json:"health"`
	CreatedAt        time.Time                 `json:"created_at"`
	LastAccessed     time.Time                 `json:"last_accessed"`
}

type MemoryAccessMode string

const (
	AccessModeReadOnly  MemoryAccessMode = "read_only"
	AccessModeWriteOnly MemoryAccessMode = "write_only"
	AccessModeReadWrite MemoryAccessMode = "read_write"
	AccessModeShared    MemoryAccessMode = "shared"
	AccessModeExclusive MemoryAccessMode = "exclusive"
)

type MemoryCoherencePolicy string

const (
	CoherenceMSI       MemoryCoherencePolicy = "msi"
	CoherenceMESI      MemoryCoherencePolicy = "mesi"
	CoherenceMOESI     MemoryCoherencePolicy = "moesi"
	CoherenceDirectory MemoryCoherencePolicy = "directory"
	CoherenceBroadcast MemoryCoherencePolicy = "broadcast"
	CoherenceNone      MemoryCoherencePolicy = "none"
)

type RegionHealth string

const (
	RegionHealthy   RegionHealth = "healthy"
	RegionDegraded  RegionHealth = "degraded"
	RegionCorrupted RegionHealth = "corrupted"
	RegionOffline   RegionHealth = "offline"
)

// MemoryAllocation represents an allocated memory segment
type MemoryAllocation struct {
	ID            string                 `json:"id"`
	JobID         string                 `json:"job_id,omitempty"`
	VMID          string                 `json:"vm_id,omitempty"`
	OwnerID       string                 `json:"owner_id"`
	Size          int64                  `json:"size_bytes"`
	Offset        int64                  `json:"offset"`
	Nodes         []string               `json:"nodes"`
	AccessPattern MemoryAccessPattern    `json:"access_pattern"`
	Priority      AllocationPriority     `json:"priority"`
	Lifetime      *AllocationLifetime    `json:"lifetime"`
	Status        AllocationStatus       `json:"status"`
	AllocatedAt   time.Time              `json:"allocated_at"`
	LastAccessed  time.Time              `json:"last_accessed"`
	Metadata      map[string]interface{} `json:"metadata"`
}

type MemoryAccessPattern string

const (
	PatternSequential       MemoryAccessPattern = "sequential"
	PatternRandom           MemoryAccessPattern = "random"
	PatternSpatialLocality  MemoryAccessPattern = "spatial_locality"
	PatternTemporalLocality MemoryAccessPattern = "temporal_locality"
	PatternStreaming        MemoryAccessPattern = "streaming"
	PatternBursty           MemoryAccessPattern = "bursty"
)

type AllocationPriority string

const (
	PriorityLow      AllocationPriority = "low"
	PriorityNormal   AllocationPriority = "normal"
	PriorityHigh     AllocationPriority = "high"
	PriorityCritical AllocationPriority = "critical"
)

type AllocationStatus string

const (
	StatusAllocated AllocationStatus = "allocated"
	StatusMigrating AllocationStatus = "migrating"
	StatusReleasing AllocationStatus = "releasing"
	StatusReleased  AllocationStatus = "released"
	StatusCorrupted AllocationStatus = "corrupted"
)

type AllocationLifetime struct {
	CreatedAt   time.Time      `json:"created_at"`
	ExpiresAt   *time.Time     `json:"expires_at,omitempty"`
	TTL         *time.Duration `json:"ttl,omitempty"`
	AutoRelease bool           `json:"auto_release"`
}

// MemoryMapping represents virtual to physical memory mapping
type MemoryMapping struct {
	VirtualAddress  uint64    `json:"virtual_address"`
	PhysicalAddress uint64    `json:"physical_address"`
	Size            int64     `json:"size_bytes"`
	NodeID          string    `json:"node_id"`
	ClusterID       string    `json:"cluster_id"`
	Permissions     int       `json:"permissions"`
	Cached          bool      `json:"cached"`
	Valid           bool      `json:"valid"`
	LastAccessed    time.Time `json:"last_accessed"`
}

// MemoryAllocationEngine handles intelligent memory allocation
type MemoryAllocationEngine struct {
	mu                   sync.RWMutex
	allocationStrategies map[AllocationStrategy]AllocationFunc
	placementOptimizer   *MemoryPlacementOptimizer
	predictionEngine     *MemoryPredictionEngine
	constraintSolver     *MemoryConstraintSolver
	loadBalancer         *MemoryLoadBalancer
}

type AllocationStrategy string

const (
	StrategyFirstFit       AllocationStrategy = "first_fit"
	StrategyBestFit        AllocationStrategy = "best_fit"
	StrategyWorstFit       AllocationStrategy = "worst_fit"
	StrategyLocalityAware  AllocationStrategy = "locality_aware"
	StrategyPerformanceOpt AllocationStrategy = "performance_optimized"
	StrategyCostOpt        AllocationStrategy = "cost_optimized"
	StrategyMLBased        AllocationStrategy = "ml_based"
)

type AllocationFunc func(ctx context.Context, request *MemoryAllocationRequest, pools []*MemoryPool) (*MemoryAllocation, error)

type MemoryAllocationRequest struct {
	RequesterID          string                      `json:"requester_id"`
	Size                 int64                       `json:"size_bytes"`
	AccessPattern        MemoryAccessPattern         `json:"access_pattern"`
	AccessMode           MemoryAccessMode            `json:"access_mode"`
	Priority             AllocationPriority          `json:"priority"`
	Lifetime             *AllocationLifetime         `json:"lifetime"`
	Constraints          []MemoryConstraint          `json:"constraints"`
	Preferences          *AllocationPreferences      `json:"preferences"`
	SecurityRequirements *MemorySecurityRequirements `json:"security_requirements"`
}

type MemoryConstraint struct {
	Type      ConstraintType `json:"type"`
	Target    string         `json:"target"`
	Operator  string         `json:"operator"`
	Value     interface{}    `json:"value"`
	Mandatory bool           `json:"mandatory"`
}

type ConstraintType string

const (
	ConstraintLatency   ConstraintType = "latency"
	ConstraintBandwidth ConstraintType = "bandwidth"
	ConstraintLocation  ConstraintType = "location"
	ConstraintCost      ConstraintType = "cost"
	ConstraintSecurity  ConstraintType = "security"
	ConstraintAffinity  ConstraintType = "affinity"
)

type AllocationPreferences struct {
	PreferredClusters   []string      `json:"preferred_clusters"`
	PreferredMemoryType MemoryType    `json:"preferred_memory_type"`
	MaxLatency          time.Duration `json:"max_latency"`
	MinBandwidth        float64       `json:"min_bandwidth_gbps"`
	CostSensitive       bool          `json:"cost_sensitive"`
	LocalityPreference  float64       `json:"locality_preference"`
}

// MemoryCoherenceProtocol handles distributed memory coherence
type MemoryCoherenceProtocol struct {
	mu               sync.RWMutex
	protocol         MemoryCoherencePolicy
	coherenceEngine  *CoherenceEngine
	invalidationMgr  *InvalidationManager
	consistencyMgr   *ConsistencyManager
	versionVector    *VersionVector
	conflictResolver *ConflictResolver
}

type CoherenceEngine struct {
	// Coherence protocol implementation
	coherenceStates map[string]CoherenceState
	messageQueue    chan CoherenceMessage
	replicationMgr  *ReplicationManager
}

type CoherenceState string

const (
	StateInvalid   CoherenceState = "invalid"
	StateShared    CoherenceState = "shared"
	StateExclusive CoherenceState = "exclusive"
	StateModified  CoherenceState = "modified"
	StateOwned     CoherenceState = "owned"
)

type CoherenceMessage struct {
	Type      CoherenceMessageType `json:"type"`
	SourceID  string               `json:"source_id"`
	TargetID  string               `json:"target_id"`
	Address   uint64               `json:"address"`
	Size      int64                `json:"size"`
	Data      []byte               `json:"data,omitempty"`
	Timestamp time.Time            `json:"timestamp"`
}

type CoherenceMessageType string

const (
	MessageRead       CoherenceMessageType = "read"
	MessageWrite      CoherenceMessageType = "write"
	MessageInvalidate CoherenceMessageType = "invalidate"
	MessageUpdate     CoherenceMessageType = "update"
	MessageAck        CoherenceMessageType = "ack"
	MessageNack       CoherenceMessageType = "nack"
)

// DistributedCacheManager handles caching across clusters
type DistributedCacheManager struct {
	mu                sync.RWMutex
	cacheLevels       map[CacheLevel]*CacheLevel
	cacheHierarchy    *CacheHierarchy
	prefetchEngine    *PrefetchEngine
	replacementPolicy CacheReplacementPolicy
	coherenceManager  *CacheCoherenceManager
}

type CacheLevel struct {
	Level         int           `json:"level"`
	Name          string        `json:"name"`
	Size          int64         `json:"size_bytes"`
	LineSize      int           `json:"line_size_bytes"`
	Associativity int           `json:"associativity"`
	AccessTime    time.Duration `json:"access_time"`
	HitRate       float64       `json:"hit_rate"`
	Nodes         []string      `json:"nodes"`
}

type CacheHierarchy struct {
	L1Cache *CacheLevel `json:"l1_cache"`
	L2Cache *CacheLevel `json:"l2_cache"`
	L3Cache *CacheLevel `json:"l3_cache"`
	LLC     *CacheLevel `json:"llc"` // Last Level Cache
}

type CacheReplacementPolicy string

const (
	ReplacementLRU    CacheReplacementPolicy = "lru"
	ReplacementLFU    CacheReplacementPolicy = "lfu"
	ReplacementFIFO   CacheReplacementPolicy = "fifo"
	ReplacementRandom CacheReplacementPolicy = "random"
	ReplacementARC    CacheReplacementPolicy = "arc"
)

// DistributedProcessingFabric aggregates CPU and GPU resources
type DistributedProcessingFabric struct {
	mu                  sync.RWMutex
	processingPools     map[string]*ProcessingPool
	workStealingMgr     *WorkStealingManager
	loadBalancer        *ProcessingLoadBalancer
	schedulingEngine    *DistributedSchedulingEngine
	taskMigrationMgr    *TaskMigrationManager
	resourceVirtualizer *ResourceVirtualizer
}

type ProcessingPool struct {
	ID             string                     `json:"id"`
	Name           string                     `json:"name"`
	Type           ProcessingType             `json:"type"`
	TotalCores     int                        `json:"total_cores"`
	AvailableCores int                        `json:"available_cores"`
	TotalGPUs      int                        `json:"total_gpus"`
	AvailableGPUs  int                        `json:"available_gpus"`
	Nodes          map[string]*ProcessingNode `json:"nodes"`
	Capabilities   []ProcessingCapability     `json:"capabilities"`
	Performance    *ProcessingPerformance     `json:"performance"`
}

type ProcessingType string

const (
	ProcessingCPU    ProcessingType = "cpu"
	ProcessingGPU    ProcessingType = "gpu"
	ProcessingFPGA   ProcessingType = "fpga"
	ProcessingTPU    ProcessingType = "tpu"
	ProcessingHybrid ProcessingType = "hybrid"
)

type ProcessingNode struct {
	NodeID       string                 `json:"node_id"`
	ClusterID    string                 `json:"cluster_id"`
	CPUCores     int                    `json:"cpu_cores"`
	GPUCount     int                    `json:"gpu_count"`
	Architecture string                 `json:"architecture"`
	Capabilities []ProcessingCapability `json:"capabilities"`
	Utilization  *ProcessingUtilization `json:"utilization"`
	Performance  *NodePerformance       `json:"performance"`
}

type ProcessingCapability string

const (
	CapabilityVectorization   ProcessingCapability = "vectorization"
	CapabilityParallel        ProcessingCapability = "parallel"
	CapabilityMLAcceleration  ProcessingCapability = "ml_acceleration"
	CapabilityEncryptionAccel ProcessingCapability = "encryption_accel"
)

// Supporting structures
type NodeTopology struct {
	NUMANode  int           `json:"numa_node"`
	SocketID  int           `json:"socket_id"`
	CoreID    int           `json:"core_id"`
	ThreadID  int           `json:"thread_id"`
	Distance  []int         `json:"distance"`
	Bandwidth float64       `json:"bandwidth_gbps"`
	Latency   time.Duration `json:"latency"`
}

type MemorySecurityContext struct {
	EncryptionEnabled   bool                 `json:"encryption_enabled"`
	EncryptionAlgorithm string               `json:"encryption_algorithm"`
	AccessControl       *MemoryAccessControl `json:"access_control"`
	AuditingEnabled     bool                 `json:"auditing_enabled"`
	IntegrityChecking   bool                 `json:"integrity_checking"`
}

type MemoryAccessControl struct {
	OwnerID     string               `json:"owner_id"`
	Permissions map[string]int       `json:"permissions"`
	Groups      []string             `json:"groups"`
	ACL         []AccessControlEntry `json:"acl"`
}

type AccessControlEntry struct {
	PrincipalID   string     `json:"principal_id"`
	PrincipalType string     `json:"principal_type"`
	Permissions   int        `json:"permissions"`
	GrantedAt     time.Time  `json:"granted_at"`
	ExpiresAt     *time.Time `json:"expires_at,omitempty"`
}

type NodePerformance struct {
	IOPS         float64       `json:"iops"`
	Throughput   float64       `json:"throughput_gbps"`
	Latency      time.Duration `json:"latency"`
	Utilization  float64       `json:"utilization"`
	ErrorRate    float64       `json:"error_rate"`
	LastMeasured time.Time     `json:"last_measured"`
}

type RegionPerformance struct {
	ReadLatency    time.Duration `json:"read_latency"`
	WriteLatency   time.Duration `json:"write_latency"`
	ReadBandwidth  float64       `json:"read_bandwidth_gbps"`
	WriteBandwidth float64       `json:"write_bandwidth_gbps"`
	HitRate        float64       `json:"hit_rate"`
	ErrorRate      float64       `json:"error_rate"`
}

type ProcessingUtilization struct {
	CPUUtilization    float64 `json:"cpu_utilization"`
	GPUUtilization    float64 `json:"gpu_utilization"`
	MemoryUtilization float64 `json:"memory_utilization"`
	IOUtilization     float64 `json:"io_utilization"`
}

type ProcessingPerformance struct {
	FLOPS       float64 `json:"flops"`
	Throughput  float64 `json:"throughput"`
	Efficiency  float64 `json:"efficiency"`
	PowerUsage  float64 `json:"power_usage_watts"`
	Temperature float64 `json:"temperature_celsius"`
}

// NewUnifiedMemoryFabric creates a new unified memory fabric
func NewUnifiedMemoryFabric(
	federationMgr *federation.FederationManager,
	memoryStateDistributor *vm.MemoryStateDistribution,
	vmStateShardingMgr *vm.VMStateShardingManager) *UnifiedMemoryFabric {

	fabric := &UnifiedMemoryFabric{
		federationMgr:          federationMgr,
		memoryStateDistributor: memoryStateDistributor,
		vmStateShardingMgr:     vmStateShardingMgr,
		memoryPools:            make(map[string]*MemoryPool),
		memoryRegions:          make(map[string]*MemoryRegion),
		allocationEngine:       NewMemoryAllocationEngine(),
		coherenceProtocol:      NewMemoryCoherenceProtocol(),
		cacheManager:           NewDistributedCacheManager(),
		performanceOptimizer:   NewMemoryPerformanceOptimizer(),
		processingFabric:       NewDistributedProcessingFabric(),
		topologyManager:        NewMemoryTopologyManager(),
		metadataService:        NewMemoryMetadataService(),
		metrics:                NewMemoryFabricMetrics(),
	}

	return fabric
}

// AllocateDistributedMemory allocates memory across clusters
func (f *UnifiedMemoryFabric) AllocateDistributedMemory(ctx context.Context,
	request *MemoryAllocationRequest) (*MemoryAllocation, error) {

	f.mu.RLock()
	pools := make([]*MemoryPool, 0, len(f.memoryPools))
	for _, pool := range f.memoryPools {
		pools = append(pools, pool)
	}
	f.mu.RUnlock()

	// Filter pools based on constraints
	candidatePools, err := f.allocationEngine.FilterPools(ctx, request, pools)
	if err != nil {
		return nil, fmt.Errorf("failed to filter memory pools: %w", err)
	}

	// Allocate memory using the allocation engine
	allocation, err := f.allocationEngine.Allocate(ctx, request, candidatePools)
	if err != nil {
		return nil, fmt.Errorf("memory allocation failed: %w", err)
	}

	// Update fabric metrics
	f.metrics.RecordAllocation(allocation)

	return allocation, nil
}

// DeallocateDistributedMemory releases allocated memory
func (f *UnifiedMemoryFabric) DeallocateDistributedMemory(ctx context.Context,
	allocationID string) error {

	// Find allocation
	allocation, err := f.findAllocation(allocationID)
	if err != nil {
		return fmt.Errorf("allocation not found: %w", err)
	}

	// Invalidate cache entries
	if err := f.cacheManager.InvalidateAllocation(ctx, allocation); err != nil {
		return fmt.Errorf("cache invalidation failed: %w", err)
	}

	// Update coherence protocol
	if err := f.coherenceProtocol.InvalidateAllocation(ctx, allocation); err != nil {
		return fmt.Errorf("coherence invalidation failed: %w", err)
	}

	// Release resources
	for _, nodeID := range allocation.Nodes {
		if err := f.releaseNodeMemory(ctx, nodeID, allocation); err != nil {
			return fmt.Errorf("failed to release memory on node %s: %w", nodeID, err)
		}
	}

	// Update allocation status
	allocation.Status = StatusReleased

	// Update metrics
	f.metrics.RecordDeallocation(allocation)

	return nil
}

// CreateMemoryRegion creates a logical memory region
func (f *UnifiedMemoryFabric) CreateMemoryRegion(ctx context.Context,
	spec *MemoryRegionSpec) (*MemoryRegion, error) {

	region := &MemoryRegion{
		ID:               spec.ID,
		Name:             spec.Name,
		Size:             spec.Size,
		AccessMode:       spec.AccessMode,
		CoherencePolicy:  spec.CoherencePolicy,
		ConsistencyLevel: spec.ConsistencyLevel,
		Health:           RegionHealthy,
		CreatedAt:        time.Now(),
		Mappings:         make(map[string]*MemoryMapping),
	}

	// Create memory mappings across clusters
	if err := f.createMemoryMappings(ctx, region, spec); err != nil {
		return nil, fmt.Errorf("failed to create memory mappings: %w", err)
	}

	// Register with coherence protocol
	if err := f.coherenceProtocol.RegisterRegion(ctx, region); err != nil {
		return nil, fmt.Errorf("failed to register with coherence protocol: %w", err)
	}

	f.mu.Lock()
	f.memoryRegions[region.ID] = region
	f.mu.Unlock()

	return region, nil
}

// GetMemoryPools returns available memory pools
func (f *UnifiedMemoryFabric) GetMemoryPools(ctx context.Context,
	filters *MemoryPoolFilters) ([]*MemoryPool, error) {

	f.mu.RLock()
	defer f.mu.RUnlock()

	var result []*MemoryPool
	for _, pool := range f.memoryPools {
		if filters == nil || filters.Matches(pool) {
			result = append(result, pool)
		}
	}

	return result, nil
}

// OptimizeMemoryPlacement optimizes memory placement for better performance
func (f *UnifiedMemoryFabric) OptimizeMemoryPlacement(ctx context.Context) error {
	// Get access patterns
	patterns, err := f.performanceOptimizer.AnalyzeAccessPatterns(ctx)
	if err != nil {
		return fmt.Errorf("failed to analyze access patterns: %w", err)
	}

	// Generate optimization recommendations
	recommendations, err := f.performanceOptimizer.GenerateOptimizations(ctx, patterns)
	if err != nil {
		return fmt.Errorf("failed to generate optimizations: %w", err)
	}

	// Apply optimizations
	for _, rec := range recommendations {
		if err := f.applyOptimization(ctx, rec); err != nil {
			// Log error but continue with other optimizations
			continue
		}
	}

	return nil
}

// GetMemoryUsage returns current memory usage statistics
func (f *UnifiedMemoryFabric) GetMemoryUsage(ctx context.Context) (*MemoryUsageStats, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	stats := &MemoryUsageStats{
		TotalCapacity:     0,
		UsedCapacity:      0,
		AvailableCapacity: 0,
		PoolUsage:         make(map[string]*PoolUsageStats),
		RegionUsage:       make(map[string]*RegionUsageStats),
		ClusterUsage:      make(map[string]*ClusterUsageStats),
	}

	// Aggregate pool statistics
	for poolID, pool := range f.memoryPools {
		stats.TotalCapacity += pool.TotalCapacity
		stats.AvailableCapacity += pool.AvailableCapacity
		stats.UsedCapacity += (pool.TotalCapacity - pool.AvailableCapacity)

		poolStats := &PoolUsageStats{
			TotalCapacity:     pool.TotalCapacity,
			AvailableCapacity: pool.AvailableCapacity,
			UsedCapacity:      pool.TotalCapacity - pool.AvailableCapacity,
			UtilizationRate:   float64(pool.TotalCapacity-pool.AvailableCapacity) / float64(pool.TotalCapacity),
			ActiveAllocations: len(pool.LoadBalancing.Allocations),
		}
		stats.PoolUsage[poolID] = poolStats
	}

	// Calculate utilization rate
	if stats.TotalCapacity > 0 {
		stats.UtilizationRate = float64(stats.UsedCapacity) / float64(stats.TotalCapacity)
	}

	return stats, nil
}

// Helper methods and supporting implementations

func (f *UnifiedMemoryFabric) findAllocation(allocationID string) (*MemoryAllocation, error) {
	// Search through all pools for the allocation
	f.mu.RLock()
	defer f.mu.RUnlock()

	for _, pool := range f.memoryPools {
		if pool.LoadBalancing != nil {
			for _, allocation := range pool.LoadBalancing.Allocations {
				if allocation.ID == allocationID {
					return allocation, nil
				}
			}
		}
	}

	return nil, fmt.Errorf("allocation %s not found", allocationID)
}

func (f *UnifiedMemoryFabric) releaseNodeMemory(ctx context.Context,
	nodeID string, allocation *MemoryAllocation) error {
	// Release memory on specific node
	return nil
}

func (f *UnifiedMemoryFabric) createMemoryMappings(ctx context.Context,
	region *MemoryRegion, spec *MemoryRegionSpec) error {
	// Create memory mappings implementation
	return nil
}

func (f *UnifiedMemoryFabric) applyOptimization(ctx context.Context,
	optimization *MemoryOptimization) error {
	// Apply memory optimization
	return nil
}

// Supporting types and interfaces
type MemoryRegionSpec struct {
	ID               string                `json:"id"`
	Name             string                `json:"name"`
	Size             int64                 `json:"size_bytes"`
	AccessMode       MemoryAccessMode      `json:"access_mode"`
	CoherencePolicy  MemoryCoherencePolicy `json:"coherence_policy"`
	ConsistencyLevel ConsistencyLevel      `json:"consistency_level"`
	Placement        *PlacementSpec        `json:"placement"`
}

type PlacementSpec struct {
	PreferredClusters  []string `json:"preferred_clusters"`
	ReplicationFactor  int      `json:"replication_factor"`
	DistributionPolicy string   `json:"distribution_policy"`
}

type MemoryPoolFilters struct {
	Type        *MemoryPoolType   `json:"type,omitempty"`
	MinCapacity *int64            `json:"min_capacity,omitempty"`
	MaxLatency  *time.Duration    `json:"max_latency,omitempty"`
	ClusterIDs  []string          `json:"cluster_ids,omitempty"`
	Health      *MemoryPoolHealth `json:"health,omitempty"`
}

func (f *MemoryPoolFilters) Matches(pool *MemoryPool) bool {
	if f.Type != nil && pool.Type != *f.Type {
		return false
	}
	if f.MinCapacity != nil && pool.AvailableCapacity < *f.MinCapacity {
		return false
	}
	if f.Health != nil && pool.Health != *f.Health {
		return false
	}
	if len(f.ClusterIDs) > 0 {
		found := false
		for _, clusterID := range f.ClusterIDs {
			if _, exists := pool.ClusterNodes[clusterID]; exists {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

type MemoryUsageStats struct {
	TotalCapacity     int64                         `json:"total_capacity_bytes"`
	UsedCapacity      int64                         `json:"used_capacity_bytes"`
	AvailableCapacity int64                         `json:"available_capacity_bytes"`
	UtilizationRate   float64                       `json:"utilization_rate"`
	PoolUsage         map[string]*PoolUsageStats    `json:"pool_usage"`
	RegionUsage       map[string]*RegionUsageStats  `json:"region_usage"`
	ClusterUsage      map[string]*ClusterUsageStats `json:"cluster_usage"`
}

type PoolUsageStats struct {
	TotalCapacity     int64   `json:"total_capacity_bytes"`
	AvailableCapacity int64   `json:"available_capacity_bytes"`
	UsedCapacity      int64   `json:"used_capacity_bytes"`
	UtilizationRate   float64 `json:"utilization_rate"`
	ActiveAllocations int     `json:"active_allocations"`
}

type RegionUsageStats struct {
	Size            int64   `json:"size_bytes"`
	AllocatedSize   int64   `json:"allocated_size_bytes"`
	UtilizationRate float64 `json:"utilization_rate"`
	AccessCount     int64   `json:"access_count"`
	ErrorCount      int64   `json:"error_count"`
}

type ClusterUsageStats struct {
	TotalMemory     int64   `json:"total_memory_bytes"`
	UsedMemory      int64   `json:"used_memory_bytes"`
	AvailableMemory int64   `json:"available_memory_bytes"`
	UtilizationRate float64 `json:"utilization_rate"`
	NodeCount       int     `json:"node_count"`
	HealthyNodes    int     `json:"healthy_nodes"`
}

type MemoryOptimization struct {
	Type     OptimizationType `json:"type"`
	Target   string           `json:"target"`
	Action   string           `json:"action"`
	Expected interface{}      `json:"expected_benefit"`
	Risk     float64          `json:"risk_score"`
	Priority int              `json:"priority"`
}

type OptimizationType string

const (
	OptimizationMigration   OptimizationType = "migration"
	OptimizationRebalancing OptimizationType = "rebalancing"
	OptimizationCompression OptimizationType = "compression"
	OptimizationPrefetching OptimizationType = "prefetching"
	OptimizationCaching     OptimizationType = "caching"
)

// Placeholder supporting structures with minimal implementations
type MemoryLoadBalancing struct {
	Algorithm   string              `json:"algorithm"`
	Allocations []*MemoryAllocation `json:"allocations"`
}

type MemoryReplicationPolicy struct {
	Factor   int    `json:"factor"`
	Strategy string `json:"strategy"`
	Enabled  bool   `json:"enabled"`
}

type MemoryPoolHealth string

const (
	PoolHealthy  MemoryPoolHealth = "healthy"
	PoolWarning  MemoryPoolHealth = "warning"
	PoolCritical MemoryPoolHealth = "critical"
	PoolOffline  MemoryPoolHealth = "offline"
)

type MemoryPoolMetrics struct {
	AllocationRate   float64 `json:"allocation_rate"`
	DeallocationRate float64 `json:"deallocation_rate"`
	HitRate          float64 `json:"hit_rate"`
	ErrorRate        float64 `json:"error_rate"`
}

type MemorySecurityRequirements struct {
	EncryptionRequired  bool     `json:"encryption_required"`
	AccessControl       bool     `json:"access_control"`
	AuditingRequired    bool     `json:"auditing_required"`
	SecurityLevel       string   `json:"security_level"`
	ComplianceStandards []string `json:"compliance_standards"`
}

// Constructor functions for supporting components
func NewMemoryAllocationEngine() *MemoryAllocationEngine {
	return &MemoryAllocationEngine{
		allocationStrategies: make(map[AllocationStrategy]AllocationFunc),
	}
}

func (e *MemoryAllocationEngine) FilterPools(ctx context.Context,
	request *MemoryAllocationRequest, pools []*MemoryPool) ([]*MemoryPool, error) {
	// Filter pools based on constraints
	return pools, nil
}

func (e *MemoryAllocationEngine) Allocate(ctx context.Context,
	request *MemoryAllocationRequest, pools []*MemoryPool) (*MemoryAllocation, error) {
	// Allocation implementation
	return &MemoryAllocation{
		ID:          fmt.Sprintf("alloc-%d", time.Now().UnixNano()),
		OwnerID:     request.RequesterID,
		Size:        request.Size,
		Status:      StatusAllocated,
		AllocatedAt: time.Now(),
	}, nil
}

func NewMemoryCoherenceProtocol() *MemoryCoherenceProtocol {
	return &MemoryCoherenceProtocol{}
}

func (p *MemoryCoherenceProtocol) RegisterRegion(ctx context.Context, region *MemoryRegion) error {
	return nil
}

func (p *MemoryCoherenceProtocol) InvalidateAllocation(ctx context.Context, allocation *MemoryAllocation) error {
	return nil
}

func NewDistributedCacheManager() *DistributedCacheManager {
	return &DistributedCacheManager{}
}

func (m *DistributedCacheManager) InvalidateAllocation(ctx context.Context, allocation *MemoryAllocation) error {
	return nil
}

func NewMemoryPerformanceOptimizer() *MemoryPerformanceOptimizer {
	return &MemoryPerformanceOptimizer{}
}

func NewDistributedProcessingFabric() *DistributedProcessingFabric {
	return &DistributedProcessingFabric{
		processingPools: make(map[string]*ProcessingPool),
	}
}

func NewMemoryTopologyManager() *MemoryTopologyManager {
	return &MemoryTopologyManager{}
}

func NewMemoryMetadataService() *MemoryMetadataService {
	return &MemoryMetadataService{}
}

func NewMemoryFabricMetrics() *MemoryFabricMetrics {
	return &MemoryFabricMetrics{}
}

// Placeholder types for supporting components
type MemoryPlacementOptimizer struct{}
type MemoryPredictionEngine struct{}
type MemoryConstraintSolver struct{}
type MemoryLoadBalancer struct{}
type InvalidationManager struct{}
type ConsistencyManager struct{}
type VersionVector struct{}
type ConflictResolver struct{}
type ReplicationManager struct{}
type PrefetchEngine struct{}
type CacheCoherenceManager struct{}
type WorkStealingManager struct{}
type ProcessingLoadBalancer struct{}
type DistributedSchedulingEngine struct{}
type TaskMigrationManager struct{}
type ResourceVirtualizer struct{}
type MemoryTopologyManager struct{}
type MemoryMetadataService struct{}
type MemoryFabricMetrics struct{}
type MemoryPerformanceOptimizer struct{}

func (o *MemoryPerformanceOptimizer) AnalyzeAccessPatterns(ctx context.Context) ([]*AccessPattern, error) {
	return []*AccessPattern{}, nil
}

func (o *MemoryPerformanceOptimizer) GenerateOptimizations(ctx context.Context, patterns []*AccessPattern) ([]*MemoryOptimization, error) {
	return []*MemoryOptimization{}, nil
}

func (m *MemoryFabricMetrics) RecordAllocation(allocation *MemoryAllocation)   {}
func (m *MemoryFabricMetrics) RecordDeallocation(allocation *MemoryAllocation) {}

type AccessPattern struct{}

// Unified Memory Compatibility Methods - Comment 12
// These methods provide backward compatibility and integration with existing memory management systems

// GetCompatibleMemoryInterface returns a memory interface compatible with legacy systems
func (f *UnifiedMemoryFabric) GetCompatibleMemoryInterface() MemoryInterface {
	return &UnifiedMemoryCompatibilityAdapter{fabric: f}
}

// UnifiedMemoryCompatibilityAdapter adapts unified memory fabric to legacy interfaces
type UnifiedMemoryCompatibilityAdapter struct {
	fabric *UnifiedMemoryFabric
}

// MemoryInterface defines the compatibility interface for legacy memory systems
type MemoryInterface interface {
	Allocate(size int64, ownerID string) (string, error)
	Deallocate(allocationID string) error
	GetUsage() (int64, int64, error) // used, total
	GetStats() map[string]interface{}
}

// Allocate implements legacy memory allocation interface
func (a *UnifiedMemoryCompatibilityAdapter) Allocate(size int64, ownerID string) (string, error) {
	ctx := context.Background()
	request := &MemoryAllocationRequest{
		RequesterID:   ownerID,
		Size:          size,
		AccessPattern: PatternSequential,
		AccessMode:    AccessModeReadWrite,
		Priority:      PriorityNormal,
	}

	allocation, err := a.fabric.AllocateDistributedMemory(ctx, request)
	if err != nil {
		return "", err
	}

	return allocation.ID, nil
}

// Deallocate implements legacy memory deallocation interface
func (a *UnifiedMemoryCompatibilityAdapter) Deallocate(allocationID string) error {
	ctx := context.Background()
	return a.fabric.DeallocateDistributedMemory(ctx, allocationID)
}

// GetUsage implements legacy usage statistics interface
func (a *UnifiedMemoryCompatibilityAdapter) GetUsage() (int64, int64, error) {
	ctx := context.Background()
	stats, err := a.fabric.GetMemoryUsage(ctx)
	if err != nil {
		return 0, 0, err
	}

	return stats.UsedCapacity, stats.TotalCapacity, nil
}

// GetStats implements legacy statistics interface
func (a *UnifiedMemoryCompatibilityAdapter) GetStats() map[string]interface{} {
	ctx := context.Background()
	stats, err := a.fabric.GetMemoryUsage(ctx)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}

	return map[string]interface{}{
		"total_capacity":     stats.TotalCapacity,
		"used_capacity":      stats.UsedCapacity,
		"available_capacity": stats.AvailableCapacity,
		"utilization_rate":   stats.UtilizationRate,
		"pool_count":         len(stats.PoolUsage),
		"region_count":       len(stats.RegionUsage),
		"cluster_count":      len(stats.ClusterUsage),
	}
}

// RegisterLegacyMemoryPool registers a legacy memory pool with the unified fabric
func (f *UnifiedMemoryFabric) RegisterLegacyMemoryPool(pool *LegacyMemoryPool) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Convert legacy pool to unified memory pool
	unifiedPool := &MemoryPool{
		ID:                pool.ID,
		Name:              pool.Name,
		Type:              PoolTypeLocal, // Default for legacy pools
		TotalCapacity:     pool.TotalCapacity,
		AvailableCapacity: pool.AvailableCapacity,
		ClusterNodes:      make(map[string]*MemoryNode),
		Health:            PoolHealthy,
		LastUpdated:       time.Now(),
	}

	// Create memory node for legacy pool
	node := &MemoryNode{
		NodeID:     pool.NodeID,
		ClusterID:  pool.ClusterID,
		Capacity:   pool.TotalCapacity,
		Available:  pool.AvailableCapacity,
		MemoryType: MemoryTypeDRAM,
		Health:     HealthHealthy,
		LastSeen:   time.Now(),
	}

	unifiedPool.ClusterNodes[pool.ClusterID] = node
	f.memoryPools[pool.ID] = unifiedPool

	return nil
}

// LegacyMemoryPool represents a legacy memory pool structure
type LegacyMemoryPool struct {
	ID                string `json:"id"`
	Name              string `json:"name"`
	NodeID            string `json:"node_id"`
	ClusterID         string `json:"cluster_id"`
	TotalCapacity     int64  `json:"total_capacity"`
	AvailableCapacity int64  `json:"available_capacity"`
}

// MigrateLegacyAllocations migrates existing legacy allocations to unified fabric
func (f *UnifiedMemoryFabric) MigrateLegacyAllocations(legacyAllocations []LegacyAllocation) error {
	for _, legacy := range legacyAllocations {
		// Convert legacy allocation to unified format
		allocation := &MemoryAllocation{
			ID:            legacy.ID,
			JobID:         legacy.JobID,
			VMID:          legacy.VMID,
			OwnerID:       legacy.OwnerID,
			Size:          legacy.Size,
			AccessPattern: PatternSequential, // Default for legacy
			Priority:      PriorityNormal,    // Default for legacy
			Status:        StatusAllocated,
			AllocatedAt:   legacy.CreatedAt,
			LastAccessed:  legacy.LastAccessed,
			Nodes:         []string{legacy.NodeID},
		}

		// Find appropriate pool and add allocation
		f.mu.Lock()
		for _, pool := range f.memoryPools {
			for nodeID, node := range pool.ClusterNodes {
				if nodeID == legacy.ClusterID || node.NodeID == legacy.NodeID {
					if pool.LoadBalancing == nil {
						pool.LoadBalancing = &MemoryLoadBalancing{
							Algorithm:   "legacy_compat",
							Allocations: []*MemoryAllocation{},
						}
					}
					pool.LoadBalancing.Allocations = append(pool.LoadBalancing.Allocations, allocation)
					break
				}
			}
		}
		f.mu.Unlock()
	}

	return nil
}

// LegacyAllocation represents a legacy memory allocation
type LegacyAllocation struct {
	ID           string    `json:"id"`
	JobID        string    `json:"job_id,omitempty"`
	VMID         string    `json:"vm_id,omitempty"`
	OwnerID      string    `json:"owner_id"`
	NodeID       string    `json:"node_id"`
	ClusterID    string    `json:"cluster_id"`
	Size         int64     `json:"size"`
	CreatedAt    time.Time `json:"created_at"`
	LastAccessed time.Time `json:"last_accessed"`
}

// GetCompatibilityMode returns the current compatibility mode settings
func (f *UnifiedMemoryFabric) GetCompatibilityMode() *CompatibilityMode {
	return &CompatibilityMode{
		LegacySupported:      true,
		BackwardCompatible:   true,
		AutoMigrationEnabled: true,
		InterfaceVersion:     "1.0",
		SupportedLegacyAPIs:  []string{"v1", "v2", "legacy"},
		DeprecationWarnings:  false,
	}
}

// CompatibilityMode defines memory system compatibility settings
type CompatibilityMode struct {
	LegacySupported      bool     `json:"legacy_supported"`
	BackwardCompatible   bool     `json:"backward_compatible"`
	AutoMigrationEnabled bool     `json:"auto_migration_enabled"`
	InterfaceVersion     string   `json:"interface_version"`
	SupportedLegacyAPIs  []string `json:"supported_legacy_apis"`
	DeprecationWarnings  bool     `json:"deprecation_warnings"`
}

// SetCompatibilityMode updates compatibility mode settings
func (f *UnifiedMemoryFabric) SetCompatibilityMode(mode *CompatibilityMode) error {
	// Validate compatibility mode settings
	if mode == nil {
		return fmt.Errorf("compatibility mode cannot be nil")
	}

	// Apply compatibility settings (this would update internal state)
	// For now, this is a placeholder implementation
	return nil
}

// Legacy compatibility methods for tests

// Legacy DTOs for backward compatibility
type LegacyMemoryPoolConfig struct {
	CoherenceProtocol string `json:"coherence_protocol"`
	ReplicationFactor int    `json:"replication_factor"`
	EnableCompression bool   `json:"enable_compression"`
	EnableEncryption  bool   `json:"enable_encryption"`
}

type LegacyMemoryAllocation struct {
	AllocationID string               `json:"allocation_id"`
	PoolID       string               `json:"pool_id"`
	Size         int64                `json:"size"` // Size in bytes
	Access       LegacyMemoryAccess   `json:"access"`
	Locality     LegacyMemoryLocality `json:"locality"`
}

type LegacyMemoryAccess string

const (
	MemoryAccessReadWrite LegacyMemoryAccess = "read_write"
	MemoryAccessReadOnly  LegacyMemoryAccess = "read_only"
	MemoryAccessWriteOnly LegacyMemoryAccess = "write_only"
)

type LegacyMemoryLocality string

const (
	MemoryLocalityLocal  LegacyMemoryLocality = "local"
	MemoryLocalityRemote LegacyMemoryLocality = "remote"
	MemoryLocalityShared LegacyMemoryLocality = "shared"
)

type LegacyMemoryPoolStatus struct {
	PoolID          string  `json:"pool_id"`
	AllocatedSizeGB float64 `json:"allocated_size_gb"`
	AvailableSizeGB float64 `json:"available_size_gb"`
	TotalSizeGB     float64 `json:"total_size_gb"`
	Status          string  `json:"status"`
}

// CreateMemoryPool creates a memory pool (legacy compatibility)
func (f *UnifiedMemoryFabric) CreateMemoryPool(ctx context.Context, pool *LegacyMemoryPool) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Convert legacy pool to internal format
	internalPool := &MemoryPool{
		ID:                pool.ID,
		Name:              pool.Name,
		Type:              PoolTypeShared,
		TotalCapacity:     pool.TotalCapacity,
		AvailableCapacity: pool.AvailableCapacity,
		ClusterNodes:      make(map[string]*MemoryNode),
		Health:            PoolHealthy,
		LastUpdated:       time.Now(),
	}

	// Create a memory node for the pool
	node := &MemoryNode{
		NodeID:     pool.NodeID,
		ClusterID:  pool.ClusterID,
		Capacity:   internalPool.TotalCapacity,
		Available:  internalPool.AvailableCapacity,
		MemoryType: MemoryTypeDRAM,
		Health:     HealthHealthy,
		LastSeen:   time.Now(),
	}

	internalPool.ClusterNodes[pool.ClusterID] = node
	f.memoryPools[pool.ID] = internalPool

	return nil
}

// AllocateMemory allocates memory (legacy compatibility)
func (f *UnifiedMemoryFabric) AllocateMemory(ctx context.Context, allocation *LegacyMemoryAllocation) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	pool, exists := f.memoryPools[allocation.PoolID]
	if !exists {
		return fmt.Errorf("memory pool %s not found", allocation.PoolID)
	}

	if pool.AvailableCapacity < allocation.Size {
		return fmt.Errorf("insufficient memory in pool %s", allocation.PoolID)
	}

	// Create internal allocation
	internalAllocation := &MemoryAllocation{
		ID:            allocation.AllocationID,
		OwnerID:       "legacy",
		Size:          allocation.Size,
		AccessPattern: PatternSequential,
		Priority:      AllocationPriority("normal"),
		Status:        StatusAllocated,
		AllocatedAt:   time.Now(),
		LastAccessed:  time.Now(),
		Nodes:         []string{allocation.PoolID},
	}

	// Update pool capacity
	pool.AvailableCapacity -= allocation.Size

	// Add allocation to pool
	if pool.LoadBalancing == nil {
		pool.LoadBalancing = &MemoryLoadBalancing{
			Algorithm:   "legacy",
			Allocations: []*MemoryAllocation{},
		}
	}
	pool.LoadBalancing.Allocations = append(pool.LoadBalancing.Allocations, internalAllocation)

	return nil
}

// GetPoolStatus gets memory pool status (legacy compatibility)
func (f *UnifiedMemoryFabric) GetPoolStatus(ctx context.Context, poolID string) (*LegacyMemoryPoolStatus, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	pool, exists := f.memoryPools[poolID]
	if !exists {
		return nil, fmt.Errorf("memory pool %s not found", poolID)
	}

	allocatedBytes := pool.TotalCapacity - pool.AvailableCapacity

	status := &LegacyMemoryPoolStatus{
		PoolID:          poolID,
		AllocatedSizeGB: float64(allocatedBytes) / (1024 * 1024 * 1024), // Convert bytes to GB
		AvailableSizeGB: float64(pool.AvailableCapacity) / (1024 * 1024 * 1024),
		TotalSizeGB:     float64(pool.TotalCapacity) / (1024 * 1024 * 1024),
		Status:          string(pool.Health),
	}

	return status, nil
}

// ReleaseMemory releases allocated memory (legacy compatibility)
func (f *UnifiedMemoryFabric) ReleaseMemory(ctx context.Context, allocationID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Find and remove allocation from all pools
	for _, pool := range f.memoryPools {
		if pool.LoadBalancing != nil {
			for i, allocation := range pool.LoadBalancing.Allocations {
				if allocation.ID == allocationID {
					// Release memory back to pool
					pool.AvailableCapacity += allocation.Size

					// Remove allocation from slice
					pool.LoadBalancing.Allocations = append(
						pool.LoadBalancing.Allocations[:i],
						pool.LoadBalancing.Allocations[i+1:]...)

					return nil
				}
			}
		}
	}

	return fmt.Errorf("allocation %s not found", allocationID)
}
