package vm

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
)

const (
	// Predictive prefetching targets
	TARGET_PREDICTION_ACCURACY   = 0.85 // 85% accuracy
	TARGET_PREDICTION_LATENCY_MS = 10   // 10ms max prediction latency
)

// State represents the current state of a VM
type State string

const (
	// StateUnknown represents an unknown VM state
	StateUnknown State = "unknown"

	// StateCreated means the VM has been created but not started
	StateCreated State = "created"

	// StateCreating represents a VM that is being created
	StateCreating State = "creating"

	// StateProvisioning represents a VM that is being provisioned
	StateProvisioning State = "provisioning"

	// StateRunning means the VM is currently running
	StateRunning State = "running"

	// StateStopped means the VM has been stopped
	StateStopped State = "stopped"

	// StatePaused represents a paused VM
	StatePaused State = "paused"

	// StatePausing represents a VM that is being paused
	StatePausing State = "pausing"

	// StateResuming represents a VM that is being resumed
	StateResuming State = "resuming"

	// StateRestarting represents a VM that is being restarted
	StateRestarting State = "restarting"

	// StateDeleting represents a VM that is being deleted
	StateDeleting State = "deleting"

	// StateMigrating represents a VM that is being migrated
	StateMigrating State = "migrating"

	// StateFailed means the VM has failed to start or has crashed
	StateFailed State = "failed"

	// Legacy state constants for compatibility
	VMStateError     = StateFailed
	VMStateRunning   = StateRunning
	VMStateDeleting  = StateDeleting
	VMStateCreating  = StateCreating
	VMStateMigrating = StateMigrating
	VMStateSuspended = StatePaused
	VMStatePaused    = StatePaused
	VMStateStopped   = StateStopped
)

// VMConfig holds configuration for a VM
type VMConfig struct {
	ID                      string                           `yaml:"id" json:"id"`
	Name                    string                           `yaml:"name" json:"name"`
	Type                    VMType                           `yaml:"type" json:"type"`
	Command                 string                           `yaml:"command" json:"command"`
	Args                    []string                         `yaml:"args" json:"args"`
	CPUShares               int                              `yaml:"cpu_shares" json:"cpu_shares"`
	MemoryMB                int                              `yaml:"memory_mb" json:"memory_mb"`
	DiskSizeGB              int                              `yaml:"disk_size_gb" json:"disk_size_gb"`
	RootFS                  string                           `yaml:"rootfs" json:"rootfs"`
	Image                   string                           `yaml:"image" json:"image"`
	Mounts                  []Mount                          `yaml:"mounts" json:"mounts"`
	Env                     map[string]string                `yaml:"env" json:"env"`
	NetworkID               string                           `yaml:"network_id" json:"network_id"`
	WorkDir                 string                           `yaml:"work_dir" json:"work_dir"`
	Tags                    map[string]string                `yaml:"tags" json:"tags"`
	PredictivePrefetching   *PredictivePrefetchingConfig     `yaml:"predictive_prefetching,omitempty" json:"predictive_prefetching,omitempty"`
}

// PredictivePrefetchingConfig configures AI-driven predictive prefetching for VM migrations
type PredictivePrefetchingConfig struct {
	Enabled                bool              `yaml:"enabled" json:"enabled"`
	PredictionAccuracy     float64           `yaml:"prediction_accuracy" json:"prediction_accuracy"`         // Target accuracy (default: 0.85)
	MaxCacheSize           int64             `yaml:"max_cache_size" json:"max_cache_size"`                   // Max cache size in bytes
	PredictionLatencyMs    int64             `yaml:"prediction_latency_ms" json:"prediction_latency_ms"`     // Max prediction latency in ms
	ModelType              string            `yaml:"model_type" json:"model_type"`                           // "neural_network", "random_forest", etc.
	TrainingDataSize       int64             `yaml:"training_data_size" json:"training_data_size"`           // Max training samples to retain
	ContinuousLearning     bool              `yaml:"continuous_learning" json:"continuous_learning"`         // Enable continuous model training
	PrefetchAheadTime      string            `yaml:"prefetch_ahead_time" json:"prefetch_ahead_time"`         // How far ahead to prefetch (e.g., "5m")
	AIModelConfig          map[string]string `yaml:"ai_model_config,omitempty" json:"ai_model_config,omitempty"` // Model-specific configuration
}

// DefaultPredictivePrefetchingConfig returns default configuration for predictive prefetching
func DefaultPredictivePrefetchingConfig() *PredictivePrefetchingConfig {
	return &PredictivePrefetchingConfig{
		Enabled:                true,
		PredictionAccuracy:     TARGET_PREDICTION_ACCURACY,    // 0.85
		MaxCacheSize:           1024 * 1024 * 1024,            // 1GB
		PredictionLatencyMs:    TARGET_PREDICTION_LATENCY_MS,  // 10ms
		ModelType:              "neural_network",
		TrainingDataSize:       100000,                        // 100k samples
		ContinuousLearning:     true,
		PrefetchAheadTime:      "5m",
		AIModelConfig: map[string]string{
			"learning_rate":    "0.001",
			"batch_size":       "32",
			"epochs":          "100",
			"hidden_layers":   "128,64",
			"activation":      "relu",
		},
	}
}

// Mount represents a filesystem mount for a VM
type Mount struct {
	Source string `yaml:"source" json:"source"`
	Target string `yaml:"target" json:"target"`
	Type   string `yaml:"type" json:"type"`
	Flags  int    `yaml:"flags" json:"flags"`
	Data   string `yaml:"data" json:"data"`
}

// VMInfo contains runtime information about a VM
type VMInfo struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	State        State             `json:"state"`
	PID          int               `json:"pid"`
	CPUShares    int               `json:"cpu_shares"`
	MemoryMB     int               `json:"memory_mb"`
	CPUUsage     float64           `json:"cpu_usage"`
	MemoryUsage  int64             `json:"memory_usage"`
	NetworkSent  int64             `json:"network_sent"`
	NetworkRecv  int64             `json:"network_recv"`
	CreatedAt    time.Time         `json:"created_at"`
	StartedAt    *time.Time        `json:"started_at"`
	StoppedAt    *time.Time        `json:"stopped_at"`
	Tags         map[string]string `json:"tags"`
	NetworkID    string            `json:"network_id"`
	IPAddress    string            `json:"ip_address"`
	RootFS       string            `json:"rootfs"`
	Image        string            `json:"image"`
	ErrorMessage string            `json:"error_message,omitempty"`
}

// DistributedStateInfo holds information about VM's distributed state
type DistributedStateInfo struct {
	StateVersion        uint64            `json:"state_version"`        // Vector clock for state consistency
	LastStateUpdate     time.Time         `json:"last_state_update"`    // Timestamp of last state change
	StateDirty          bool              `json:"state_dirty"`          // Whether state needs synchronization
	ShardID             string            `json:"shard_id"`             // Which shard contains this VM's state
	ReplicationNodes    []string          `json:"replication_nodes"`    // Nodes that replicate this VM's state
	GlobalCoordinates   *GlobalPosition   `json:"global_coordinates"`   // Position in global cluster space
	MigrationState      *MigrationInfo    `json:"migration_state"`      // Current migration status
	PredictiveCache     *PredictiveCache  `json:"predictive_cache"`     // AI-driven predictive caching info
	CrossClusterRefs    []string          `json:"cross_cluster_refs"`   // References to other clusters
	StateConsistency    ConsistencyLevel  `json:"state_consistency"`    // Required consistency level
	LastConsistencySync time.Time         `json:"last_consistency_sync"` // Last consistency check
}

// GlobalPosition represents VM's position in the global distributed space
type GlobalPosition struct {
	ClusterID       string    `json:"cluster_id"`
	NodeID          string    `json:"node_id"`
	RegionID        string    `json:"region_id"`
	AvailabilityZone string   `json:"availability_zone"`
	GeographicCoord *GeoCoord `json:"geographic_coord,omitempty"`
	NetworkLatency  float64   `json:"network_latency"` // ms to cluster root
}

// GeoCoord represents geographic coordinates
type GeoCoord struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

// MigrationInfo tracks ongoing migration state
type MigrationInfo struct {
	InProgress      bool              `json:"in_progress"`
	SourceNode      string            `json:"source_node"`
	TargetNode      string            `json:"target_node"`
	MigrationType   string            `json:"migration_type"` // "live", "offline", "hybrid"
	StartedAt       time.Time         `json:"started_at"`
	EstimatedCompletion time.Time     `json:"estimated_completion"`
	Progress        float64           `json:"progress"` // 0.0 to 1.0
	Strategy        MigrationStrategy `json:"strategy"`
	BandwidthUsage  int64             `json:"bandwidth_usage"` // bytes/sec
}

// MigrationStrategy defines migration approach
type MigrationStrategy struct {
	MemoryStrategy    string  `json:"memory_strategy"`    // "pre-copy", "post-copy", "hybrid"
	NetworkOptimized  bool    `json:"network_optimized"`  // Use bandwidth optimization
	PredictiveEnabled bool    `json:"predictive_enabled"` // Use AI-driven prediction
	CompressionLevel  int     `json:"compression_level"`  // 0-9
	MaxDowntime       int64   `json:"max_downtime"`       // milliseconds
}

// PredictiveCache holds AI-driven predictive caching information
type PredictiveCache struct {
	Enabled           bool                    `json:"enabled"`
	CacheHitRate      float64                 `json:"cache_hit_rate"`
	PredictionAccuracy float64               `json:"prediction_accuracy"`
	PrefetchedPages   map[string]time.Time    `json:"prefetched_pages"` // page_id -> prefetch_time
	ModelVersion      string                  `json:"model_version"`
	LastModelUpdate   time.Time               `json:"last_model_update"`
	PredictionsActive []PredictionInfo        `json:"predictions_active"`
	CacheSize         int64                   `json:"cache_size"` // bytes
}

// PredictionInfo represents an active prediction
type PredictionInfo struct {
	PageID       string    `json:"page_id"`
	Confidence   float64   `json:"confidence"`   // 0.0 to 1.0
	PredictedAt  time.Time `json:"predicted_at"`
	AccessTime   time.Time `json:"access_time"`  // predicted access time
	Priority     int       `json:"priority"`     // 1-10
}

// ConsistencyLevel defines required state consistency
type ConsistencyLevel string

const (
	ConsistencyEventual ConsistencyLevel = "eventual"
	ConsistencyStrong   ConsistencyLevel = "strong"
	ConsistencySession  ConsistencyLevel = "session"
	ConsistencyLinear   ConsistencyLevel = "linearizable"
)

// MemoryDistributionInfo tracks distributed memory state
type MemoryDistributionInfo struct {
	Enabled               bool                         `json:"enabled"`
	TotalPages            int64                        `json:"total_pages"`
	DistributedPages      int64                        `json:"distributed_pages"`
	LocalPages            int64                        `json:"local_pages"`
	RemotePages           int64                        `json:"remote_pages"`
	PageDistribution      map[string]int64             `json:"page_distribution"` // node_id -> page_count
	DirtyPages            []string                     `json:"dirty_pages"`       // page IDs that need sync
	LastSyncTime          time.Time                    `json:"last_sync_time"`
	SyncInProgress        bool                         `json:"sync_in_progress"`
	CompressionEnabled    bool                         `json:"compression_enabled"`
	CompressionRatio      float64                      `json:"compression_ratio"`
	CoherenceProtocol     string                       `json:"coherence_protocol"` // "MSI", "MESI", "MOESI"
	AccessPatterns        map[string]AccessPattern     `json:"access_patterns"`    // page_id -> pattern
}

// AccessPattern tracks memory access patterns for optimization
type AccessPattern struct {
	AccessCount     int64     `json:"access_count"`
	LastAccess      time.Time `json:"last_access"`
	AccessFrequency float64   `json:"access_frequency"` // accesses per second
	ReadWriteRatio  float64   `json:"read_write_ratio"` // reads / writes
	Locality        float64   `json:"locality"`         // spatial locality score
}

// VM represents a virtual machine with distributed state management
type VM struct {
	config     VMConfig
	state      State
	pid        int
	cmd        *exec.Cmd
	mutex      sync.RWMutex
	createdAt  time.Time
	startedAt  *time.Time
	stoppedAt  *time.Time
	cgroupPath string
	netns      string
	ipAddress  string
	statsLock  sync.RWMutex
	stats      VMStats
	// New fields for enhanced VM management
	nodeID      string
	resourceID  string
	updatedAt   time.Time
	processInfo VMProcessInfo

	// Distributed State Management Fields
	distributedState    *DistributedStateInfo     `json:"distributed_state"`
	memoryDistribution  *MemoryDistributionInfo   `json:"memory_distribution"`
	stateCoordinator    string                    `json:"state_coordinator"`    // Node ID of state coordinator
	federationContext   *FederationContext        `json:"federation_context"`   // Cross-cluster federation info
	stateHistory        []StateSnapshot           `json:"state_history"`        // Recent state snapshots
	conflictResolution  *ConflictResolutionInfo   `json:"conflict_resolution"`  // Conflict handling state
	performanceMetrics  *DistributedMetrics       `json:"performance_metrics"`  // Performance tracking
	stateLock           sync.RWMutex              // Protects distributed state fields
}

// FederationContext holds cross-cluster federation information
type FederationContext struct {
	HomeClusterID       string            `json:"home_cluster_id"`
	CurrentClusterID    string            `json:"current_cluster_id"`
	AuthorizedClusters  []string          `json:"authorized_clusters"`
	FederationTokens    map[string]string `json:"federation_tokens"`
	CrossClusterRefs    []ClusterRef      `json:"cross_cluster_refs"`
	SecurityContext     SecurityContext   `json:"security_context"`
	NetworkPolicy       NetworkPolicy     `json:"network_policy"`
}

// ClusterRef represents a reference to another cluster
type ClusterRef struct {
	ClusterID       string    `json:"cluster_id"`
	ClusterEndpoint string    `json:"cluster_endpoint"`
	LastContact     time.Time `json:"last_contact"`
	Available       bool      `json:"available"`
	RTT             float64   `json:"rtt"` // Round trip time in ms
}

// SecurityContext defines security settings for distributed operations
type SecurityContext struct {
	EncryptionEnabled bool              `json:"encryption_enabled"`
	KeyVersion        string            `json:"key_version"`
	AccessPolicies    []string          `json:"access_policies"`
	AuditEnabled      bool              `json:"audit_enabled"`
	Certificates      map[string]string `json:"certificates"`
}

// NetworkPolicy defines networking policies for distributed VM operations
type NetworkPolicy struct {
	AllowedNetworks     []string          `json:"allowed_networks"`
	BandwidthLimits     map[string]int64  `json:"bandwidth_limits"` // operation -> bytes/sec
	CompressionEnabled  bool              `json:"compression_enabled"`
	PriorityClass       string            `json:"priority_class"`
	QoSSettings         QoSSettings       `json:"qos_settings"`
}

// QoSSettings defines Quality of Service settings
type QoSSettings struct {
	MaxLatency      int64   `json:"max_latency"`      // milliseconds
	MinBandwidth    int64   `json:"min_bandwidth"`    // bytes/sec
	MaxJitter       int64   `json:"max_jitter"`       // milliseconds
	PacketLossLimit float64 `json:"packet_loss_limit"` // percentage
}

// StateSnapshot represents a point-in-time state snapshot
type StateSnapshot struct {
	Timestamp       time.Time              `json:"timestamp"`
	StateVersion    uint64                 `json:"state_version"`
	VMState         State                  `json:"vm_state"`
	MemoryChecksum  string                 `json:"memory_checksum"`
	ConfigChecksum  string                 `json:"config_checksum"`
	Metadata        map[string]interface{} `json:"metadata"`
	Size            int64                  `json:"size"`
	Compressed      bool                   `json:"compressed"`
}

// ConflictResolutionInfo tracks conflict resolution state
type ConflictResolutionInfo struct {
	ActiveConflicts   []StateConflict      `json:"active_conflicts"`
	ResolutionPolicy  ConflictPolicy       `json:"resolution_policy"`
	LastResolution    time.Time            `json:"last_resolution"`
	ConflictHistory   []ResolvedConflict   `json:"conflict_history"`
	AutoResolveRules  []AutoResolveRule    `json:"auto_resolve_rules"`
}

// StateConflict represents a state conflict between nodes
type StateConflict struct {
	ConflictID      string                 `json:"conflict_id"`
	ConflictType    string                 `json:"conflict_type"`
	SourceNode      string                 `json:"source_node"`
	TargetNode      string                 `json:"target_node"`
	ConflictField   string                 `json:"conflict_field"`
	SourceValue     interface{}            `json:"source_value"`
	TargetValue     interface{}            `json:"target_value"`
	DetectedAt      time.Time              `json:"detected_at"`
	Severity        ConflictSeverity       `json:"severity"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// ConflictPolicy defines how conflicts should be resolved
type ConflictPolicy struct {
	DefaultStrategy   ConflictStrategy `json:"default_strategy"`
	FieldPolicies     map[string]ConflictStrategy `json:"field_policies"`
	TimeoutMs         int64            `json:"timeout_ms"`
	RequireConsensus  bool             `json:"require_consensus"`
	VotingEnabled     bool             `json:"voting_enabled"`
}

// ConflictStrategy defines conflict resolution strategies
type ConflictStrategy string

const (
	ConflictLastWriteWins  ConflictStrategy = "last_write_wins"
	ConflictHighestVersion ConflictStrategy = "highest_version"
	ConflictMerge          ConflictStrategy = "merge"
	ConflictManual         ConflictStrategy = "manual"
	ConflictVoting         ConflictStrategy = "voting"
)

// ConflictSeverity defines the severity of a conflict
type ConflictSeverity string

const (
	ConflictSeverityLow      ConflictSeverity = "low"
	ConflictSeverityMedium   ConflictSeverity = "medium"
	ConflictSeverityHigh     ConflictSeverity = "high"
	ConflictSeverityCritical ConflictSeverity = "critical"
)

// ResolvedConflict represents a resolved conflict
type ResolvedConflict struct {
	ConflictID      string           `json:"conflict_id"`
	Resolution      ConflictStrategy `json:"resolution"`
	ResolvedAt      time.Time        `json:"resolved_at"`
	ResolvedBy      string           `json:"resolved_by"`
	FinalValue      interface{}      `json:"final_value"`
	ConflictDuration time.Duration   `json:"conflict_duration"`
}

// AutoResolveRule defines automatic conflict resolution rules
type AutoResolveRule struct {
	RuleID          string                 `json:"rule_id"`
	FieldPattern    string                 `json:"field_pattern"`
	ConflictTypes   []string               `json:"conflict_types"`
	Strategy        ConflictStrategy       `json:"strategy"`
	Conditions      map[string]interface{} `json:"conditions"`
	Priority        int                    `json:"priority"`
	Enabled         bool                   `json:"enabled"`
}

// DistributedMetrics tracks performance metrics for distributed operations
type DistributedMetrics struct {
	StateAccess         *AccessMetrics       `json:"state_access"`
	Migration           *MigrationMetrics    `json:"migration"`
	MemoryDistribution  *MemoryMetrics       `json:"memory_distribution"`
	NetworkPerformance  *NetworkMetrics      `json:"network_performance"`
	ConsistencyMetrics  *ConsistencyMetrics  `json:"consistency_metrics"`
	PredictiveMetrics   *PredictiveMetrics   `json:"predictive_metrics"`
	LastUpdate          time.Time            `json:"last_update"`
}

// AccessMetrics tracks state access performance
type AccessMetrics struct {
	ReadLatencyMs       float64   `json:"read_latency_ms"`
	WriteLatencyMs      float64   `json:"write_latency_ms"`
	CacheHitRate        float64   `json:"cache_hit_rate"`
	LocalHitRate        float64   `json:"local_hit_rate"`
	RemoteHitRate       float64   `json:"remote_hit_rate"`
	ErrorRate           float64   `json:"error_rate"`
	ThroughputOpsPerSec float64   `json:"throughput_ops_per_sec"`
	LastMeasurement     time.Time `json:"last_measurement"`
}

// MigrationMetrics tracks migration performance
type MigrationMetrics struct {
	AverageMigrationTime    time.Duration `json:"average_migration_time"`
	SuccessRate             float64       `json:"success_rate"`
	DowntimeMs              float64       `json:"downtime_ms"`
	DataTransferRateMBps    float64       `json:"data_transfer_rate_mbps"`
	CompressionRatio        float64       `json:"compression_ratio"`
	PredictionAccuracy      float64       `json:"prediction_accuracy"`
	TotalMigrations         int64         `json:"total_migrations"`
	FailedMigrations        int64         `json:"failed_migrations"`
	LastMigration           time.Time     `json:"last_migration"`
}

// MemoryMetrics tracks memory distribution performance
type MemoryMetrics struct {
	SyncLatencyMs           float64   `json:"sync_latency_ms"`
	CompressionRatio        float64   `json:"compression_ratio"`
	DeduplicationRatio      float64   `json:"deduplication_ratio"`
	PageFaultRate           float64   `json:"page_fault_rate"`
	RemotePageAccess        float64   `json:"remote_page_access"`
	LocalityScore           float64   `json:"locality_score"`
	CoherenceOverheadMs     float64   `json:"coherence_overhead_ms"`
	LastMemorySync          time.Time `json:"last_memory_sync"`
}

// NetworkMetrics tracks network performance for distributed operations
type NetworkMetrics struct {
	BandwidthUtilization    float64   `json:"bandwidth_utilization"`
	LatencyMs               float64   `json:"latency_ms"`
	PacketLossRate          float64   `json:"packet_loss_rate"`
	ThroughputMbps          float64   `json:"throughput_mbps"`
	ConnectionPoolSize      int       `json:"connection_pool_size"`
	ActiveConnections       int       `json:"active_connections"`
	CompressionSavings      float64   `json:"compression_savings"`
	RetransmissionRate      float64   `json:"retransmission_rate"`
	LastNetworkMeasurement  time.Time `json:"last_network_measurement"`
}

// ConsistencyMetrics tracks consistency performance
type ConsistencyMetrics struct {
	ConsistencyLatencyMs    float64   `json:"consistency_latency_ms"`
	ConflictRate            float64   `json:"conflict_rate"`
	ResolutionTimeMs        float64   `json:"resolution_time_ms"`
	SyncSuccessRate         float64   `json:"sync_success_rate"`
	VectorClockDrift        float64   `json:"vector_clock_drift"`
	StateVersions           int64     `json:"state_versions"`
	LastConsistencyCheck    time.Time `json:"last_consistency_check"`
}

// PredictiveMetrics tracks AI-driven predictive performance
type PredictiveMetrics struct {
	PredictionAccuracy      float64   `json:"prediction_accuracy"`
	PredictionLatencyMs     float64   `json:"prediction_latency_ms"`
	CacheHitImprovement     float64   `json:"cache_hit_improvement"`
	PrefetchSuccessRate     float64   `json:"prefetch_success_rate"`
	ModelTrainingTime       time.Duration `json:"model_training_time"`
	FalsePositiveRate       float64   `json:"false_positive_rate"`
	FalseNegativeRate       float64   `json:"false_negative_rate"`
	LastModelUpdate         time.Time `json:"last_model_update"`
	PredictionsPerSecond    float64   `json:"predictions_per_second"`
}

// VMStats holds runtime statistics for a VM
type VMStats struct {
	CPUUsage    float64
	MemoryUsage int64
	NetworkSent int64
	NetworkRecv int64
	LastUpdated time.Time
}

// VMProcessInfo holds process information for a VM
type VMProcessInfo struct {
	PID             int
	PPID            int
	Command         string
	Args            []string
	StartTime       time.Time
	CPUTime         time.Duration
	MemoryRSS       int64
	MemoryVSZ       int64
	CPUUsagePercent float64
	MemoryUsageMB   int64
}

// NewVM creates a new VM instance
func NewVM(config VMConfig) (*VM, error) {
	if config.Name == "" {
		return nil, fmt.Errorf("VM name must be specified")
	}

	if config.Command == "" {
		return nil, fmt.Errorf("VM command must be specified")
	}

	if config.ID == "" {
		config.ID = uuid.New().String()
	}

	// Default resource limits if not specified
	if config.CPUShares == 0 {
		config.CPUShares = 1024 // Default CPU shares
	}
	if config.MemoryMB == 0 {
		config.MemoryMB = 512 // Default 512MB of memory
	}

	vm := &VM{
		config:    config,
		state:     StateCreated,
		createdAt: time.Now(),
		stats: VMStats{
			LastUpdated: time.Now(),
		},
	}

	return vm, nil
}

// ID returns the VM's ID
func (vm *VM) ID() string {
	return vm.config.ID
}

// Name returns the VM's name
func (vm *VM) Name() string {
	return vm.config.Name
}

// State returns the VM's current state
func (vm *VM) State() State {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.state
}

// SetState sets the VM's state
func (vm *VM) SetState(state State) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.state = state
	vm.updatedAt = time.Now()
}

// IsRunning returns true if the VM is running
func (vm *VM) IsRunning() bool {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.state == StateRunning
}

// Start starts the VM
func (vm *VM) Start() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state == StateRunning {
		return fmt.Errorf("VM is already running")
	}

	// Set up cgroups
	if err := vm.setupCgroups(); err != nil {
		return fmt.Errorf("failed to set up cgroups: %w", err)
	}

	// Set up namespaces
	if err := vm.setupNamespaces(); err != nil {
		return fmt.Errorf("failed to set up namespaces: %w", err)
	}

	// Set up command
	vm.cmd = exec.Command(vm.config.Command, vm.config.Args...)

	// Set up process namespaces
	vm.cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWUTS | syscall.CLONE_NEWPID | syscall.CLONE_NEWNS | syscall.CLONE_NEWNET | syscall.CLONE_NEWIPC,
	}

	// Set up environment variables
	if vm.config.Env != nil {
		env := os.Environ()
		for k, v := range vm.config.Env {
			env = append(env, fmt.Sprintf("%s=%s", k, v))
		}
		vm.cmd.Env = env
	}

	// Set up working directory
	if vm.config.WorkDir != "" {
		vm.cmd.Dir = vm.config.WorkDir
	}

	// Start the process
	if err := vm.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start VM process: %w", err)
	}

	vm.pid = vm.cmd.Process.Pid

	// Move process to appropriate cgroups
	if err := vm.assignToCgroups(); err != nil {
		return fmt.Errorf("failed to assign process to cgroups: %w", err)
	}

	// Set up network if needed
	if vm.config.NetworkID != "" {
		if err := vm.setupNetwork(); err != nil {
			return fmt.Errorf("failed to set up network: %w", err)
		}
	}

	now := time.Now()
	vm.startedAt = &now
	vm.state = StateRunning

	// Monitor the process
	go vm.monitor()

	log.Printf("Started VM %s (%s) with PID %d", vm.Name(), vm.ID(), vm.pid)
	return nil
}

// Stop stops the VM
func (vm *VM) Stop() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state != StateRunning {
		return fmt.Errorf("VM is not running")
	}

	// Send SIGTERM signal
	if err := vm.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		log.Printf("Failed to send SIGTERM to VM %s: %v, will try SIGKILL", vm.ID(), err)

		// If SIGTERM fails, try SIGKILL
		if err := vm.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill VM: %w", err)
		}
	}

	// Wait for the process to exit
	done := make(chan error)
	go func() {
		done <- vm.cmd.Wait()
	}()

	// Wait for process to exit or timeout
	select {
	case err := <-done:
		if err != nil {
			log.Printf("VM %s process exited with error: %v", vm.ID(), err)
		}
	case <-time.After(10 * time.Second):
		// If it doesn't exit within timeout, force kill
		if err := vm.cmd.Process.Kill(); err != nil {
			log.Printf("Failed to force kill VM %s: %v", vm.ID(), err)
		}
		<-done // Wait for the process to be killed
	}

	now := time.Now()
	vm.stoppedAt = &now
	vm.state = StateStopped

	// Clean up cgroups and namespaces
	if err := vm.cleanupCgroups(); err != nil {
		log.Printf("Failed to clean up cgroups for VM %s: %v", vm.ID(), err)
	}

	if err := vm.cleanupNamespaces(); err != nil {
		log.Printf("Failed to clean up namespaces for VM %s: %v", vm.ID(), err)
	}

	log.Printf("Stopped VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// Cleanup cleans up VM resources
func (vm *VM) Cleanup() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state == StateRunning {
		return fmt.Errorf("cannot clean up a running VM")
	}

	// Clean up any remaining resources
	if err := vm.cleanupCgroups(); err != nil {
		log.Printf("Failed to clean up cgroups for VM %s: %v", vm.ID(), err)
	}

	if err := vm.cleanupNamespaces(); err != nil {
		log.Printf("Failed to clean up namespaces for VM %s: %v", vm.ID(), err)
	}

	// Clean up any storage used by the VM
	// This will be implemented based on specific storage requirements

	log.Printf("Cleaned up VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// GetInfo returns information about the VM
func (vm *VM) GetInfo() VMInfo {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	vm.statsLock.RLock()
	defer vm.statsLock.RUnlock()

	info := VMInfo{
		ID:          vm.config.ID,
		Name:        vm.config.Name,
		State:       vm.state,
		PID:         vm.pid,
		CPUShares:   vm.config.CPUShares,
		MemoryMB:    vm.config.MemoryMB,
		CPUUsage:    vm.stats.CPUUsage,
		MemoryUsage: vm.stats.MemoryUsage,
		NetworkSent: vm.stats.NetworkSent,
		NetworkRecv: vm.stats.NetworkRecv,
		CreatedAt:   vm.createdAt,
		StartedAt:   vm.startedAt,
		StoppedAt:   vm.stoppedAt,
		Tags:        vm.config.Tags,
		NetworkID:   vm.config.NetworkID,
		IPAddress:   vm.ipAddress,
		RootFS:      vm.config.RootFS,
	}

	return info
}

// GetNodeID returns the node ID
func (vm *VM) GetNodeID() string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.nodeID
}

// NodeID returns the node ID (compatibility method)
func (vm *VM) NodeID() string {
	return vm.GetNodeID()
}

// GetResourceID returns the resource ID
func (vm *VM) GetResourceID() string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.resourceID
}

// ResourceID returns the resource ID (compatibility method)
func (vm *VM) ResourceID() string {
	return vm.GetResourceID()
}

// GetUpdatedAt returns the last update time
func (vm *VM) GetUpdatedAt() time.Time {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.updatedAt
}

// UpdatedAt returns the last update time (compatibility method)
func (vm *VM) UpdatedAt() time.Time {
	return vm.GetUpdatedAt()
}

// GetProcessInfo returns the process information
func (vm *VM) GetProcessInfo() VMProcessInfo {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.processInfo
}

// ProcessInfo returns the process information (compatibility method)
func (vm *VM) ProcessInfo() VMProcessInfo {
	return vm.GetProcessInfo()
}

// SetNodeID sets the node ID
func (vm *VM) SetNodeID(nodeID string) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.nodeID = nodeID
	vm.updatedAt = time.Now()
}

// SetResourceID sets the resource ID
func (vm *VM) SetResourceID(resourceID string) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.resourceID = resourceID
	vm.updatedAt = time.Now()
}

// ApplyUpdateSpec applies an update specification to the VM
func (vm *VM) ApplyUpdateSpec(spec VMUpdateSpec) error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	
	// Apply updates to VM configuration
	if spec.Name != nil {
		vm.config.Name = *spec.Name
	}
	if spec.CPU != nil {
		vm.config.CPUShares = *spec.CPU
	}
	if spec.Memory != nil {
		vm.config.MemoryMB = int(*spec.Memory)
	}
	if spec.Disk != nil {
		vm.config.DiskSizeGB = int(*spec.Disk)
	}
	if spec.Tags != nil {
		if vm.config.Tags == nil {
			vm.config.Tags = make(map[string]string)
		}
		for k, v := range spec.Tags {
			vm.config.Tags[k] = v
		}
	}
	
	// Update timestamp
	vm.updatedAt = time.Now()
	
	return nil
}

// SetProcessInfo sets the process information
func (vm *VM) SetProcessInfo(processInfo VMProcessInfo) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.processInfo = processInfo
	vm.updatedAt = time.Now()
}

// SetUpdatedAt sets the updated time
func (vm *VM) SetUpdatedAt(t time.Time) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.updatedAt = t
}

// Private methods

func (vm *VM) setupCgroups() error {
	// Create cgroup path
	vm.cgroupPath = filepath.Join("/sys/fs/cgroup", "novacron", vm.config.ID)

	// In a full implementation, we would create cgroups for CPU, memory, etc.
	// For now, we'll just log and return success
	log.Printf("Set up cgroups for VM %s at %s", vm.ID(), vm.cgroupPath)
	return nil
}

func (vm *VM) setupNamespaces() error {
	// In a full implementation, we would set up all required namespaces
	// For now, we'll just log and return success
	log.Printf("Set up namespaces for VM %s", vm.ID())
	return nil
}

func (vm *VM) assignToCgroups() error {
	// In a full implementation, we would assign the process to created cgroups
	// For now, we'll just log and return success
	log.Printf("Assigned VM %s (PID %d) to cgroups", vm.ID(), vm.pid)
	return nil
}

func (vm *VM) setupNetwork() error {
	// In a full implementation, we would set up the network namespace
	// For now, we'll just log and return success
	vm.ipAddress = "10.0.0.1" // Placeholder
	log.Printf("Set up network for VM %s with IP %s", vm.ID(), vm.ipAddress)
	return nil
}

func (vm *VM) cleanupCgroups() error {
	// In a full implementation, we would remove cgroups
	// For now, we'll just log and return success
	log.Printf("Cleaned up cgroups for VM %s", vm.ID())
	return nil
}

func (vm *VM) cleanupNamespaces() error {
	// In a full implementation, we would clean up namespaces
	// For now, we'll just log and return success
	log.Printf("Cleaned up namespaces for VM %s", vm.ID())
	return nil
}

func (vm *VM) monitor() {
	// In a full implementation, this would monitor the VM and collect stats
	for {
		// Wait for process to exit
		if err := vm.cmd.Wait(); err != nil {
			vm.mutex.Lock()
			if vm.state == StateRunning {
				log.Printf("VM %s process exited unexpectedly: %v", vm.ID(), err)
				vm.state = StateFailed
			}
			vm.mutex.Unlock()
			return
		}

		// If we get here, the process exited normally
		vm.mutex.Lock()
		if vm.state == StateRunning {
			now := time.Now()
			vm.stoppedAt = &now
			vm.state = StateStopped
			log.Printf("VM %s process exited normally", vm.ID())
		}
		vm.mutex.Unlock()
		return
	}
}

func (vm *VM) collectStats() {
	// In a full implementation, this would collect resource usage stats
	// For now, we'll just update the last updated time
	vm.statsLock.Lock()
	defer vm.statsLock.Unlock()
	vm.stats.LastUpdated = time.Now()
}

// SetStartedAt sets the started time
func (vm *VM) SetStartedAt(t time.Time) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.startedAt = &t
	vm.updatedAt = time.Now()
}

// Config returns the VM configuration
func (vm *VM) Config() VMConfig {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config
}

// ErrorMessage returns the VM's error message
func (vm *VM) ErrorMessage() string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	// Return empty string for now, can be extended with actual error field
	return ""
}

// CreatedAt returns when the VM was created
func (vm *VM) CreatedAt() time.Time {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.createdAt
}

// GetCreatedAt returns when the VM was created
func (vm *VM) GetCreatedAt() time.Time {
	return vm.CreatedAt()
}

// GetCommand returns the VM's command
func (vm *VM) GetCommand() string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.Command
}

// GetArgs returns the VM's arguments
func (vm *VM) GetArgs() []string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.Args
}

// GetCPUShares returns the VM's CPU shares
func (vm *VM) GetCPUShares() int {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.CPUShares
}

// GetMemoryMB returns the VM's memory in MB
func (vm *VM) GetMemoryMB() int {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.MemoryMB
}

// GetDiskSizeGB returns the VM's disk size in GB
func (vm *VM) GetDiskSizeGB() int {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.DiskSizeGB
}

// GetTags returns the VM's tags
func (vm *VM) GetTags() map[string]string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.Tags
}

// GetStats returns the VM's statistics
func (vm *VM) GetStats() VMStats {
	vm.statsLock.RLock()
	defer vm.statsLock.RUnlock()
	return vm.stats
}

// SetName sets the VM's name
func (vm *VM) SetName(name string) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.Name = name
	vm.updatedAt = time.Now()
}

// SetCPUShares sets the VM's CPU shares
func (vm *VM) SetCPUShares(cpuShares int) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.CPUShares = cpuShares
	vm.updatedAt = time.Now()
}

// SetMemoryMB sets the VM's memory in MB
func (vm *VM) SetMemoryMB(memoryMB int) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.MemoryMB = memoryMB
	vm.updatedAt = time.Now()
}

// SetDiskSizeGB sets the VM's disk size in GB
func (vm *VM) SetDiskSizeGB(diskSizeGB int) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.DiskSizeGB = diskSizeGB
	vm.updatedAt = time.Now()
}

// SetTags sets the VM's tags
func (vm *VM) SetTags(tags map[string]string) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.Tags = tags
	vm.updatedAt = time.Now()
}

// Delete deletes the VM (stub implementation)
func (vm *VM) Delete() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.state = StateDeleting
	// Stub implementation - would actually delete VM resources
	return nil
}

// ResumeFromState resumes VM from a saved state (stub implementation)
func (vm *VM) ResumeFromState(statePath string) error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.state = StateRunning
	// Stub implementation - would actually resume VM from state
	return nil
}

// GetDiskPaths returns the disk paths for the VM (stub implementation)
func (vm *VM) GetDiskPaths() ([]string, error) {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	// Stub implementation - would return actual disk paths
	return []string{"/var/lib/novacron/vms/" + vm.config.ID + "/disk.qcow2"}, nil
}

// GetMemoryStatePath returns the memory state path for the VM (stub implementation)
func (vm *VM) GetMemoryStatePath() (string, error) {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	// Stub implementation - would return actual memory state path
	return "/var/lib/novacron/vms/" + vm.config.ID + "/memory.state", nil
}

// GetMemoryDeltaPath returns the memory delta path for the VM (stub implementation)
func (vm *VM) GetMemoryDeltaPath(iteration int) (string, error) {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	// Stub implementation - would return actual memory delta path
	return "/var/lib/novacron/vms/" + vm.config.ID + "/memory.delta." + fmt.Sprintf("%d", iteration), nil
}

// Suspend suspends the VM (stub implementation)
func (vm *VM) Suspend() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	if vm.state != StateRunning {
		return fmt.Errorf("VM is not running")
	}
	vm.state = StatePaused
	// Stub implementation - would actually suspend VM
	return nil
}

// Resume resumes the VM (stub implementation)
func (vm *VM) Resume() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	if vm.state != StatePaused && vm.state != VMStateSuspended {
		return fmt.Errorf("VM is not suspended or paused")
	}
	vm.state = StateRunning
	// Stub implementation - would actually resume VM
	return nil
}

// ===== DISTRIBUTED STATE MANAGEMENT METHODS =====

// InitializeDistributedState initializes distributed state management for the VM
func (vm *VM) InitializeDistributedState(ctx context.Context, shardID string, coordinatorNode string) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	if vm.distributedState != nil {
		return fmt.Errorf("distributed state already initialized for VM %s", vm.ID())
	}

	now := time.Now()
	vm.distributedState = &DistributedStateInfo{
		StateVersion:        1,
		LastStateUpdate:     now,
		StateDirty:          false,
		ShardID:             shardID,
		ReplicationNodes:    []string{},
		StateConsistency:    ConsistencyEventual,
		LastConsistencySync: now,
		GlobalCoordinates: &GlobalPosition{
			NodeID:         vm.nodeID,
			NetworkLatency: 0.0,
		},
		PredictiveCache: &PredictiveCache{
			Enabled:           true,
			CacheHitRate:      0.0,
			PredictionAccuracy: TARGET_PREDICTION_ACCURACY,
			PrefetchedPages:   make(map[string]time.Time),
			PredictionsActive: []PredictionInfo{},
			CacheSize:         0,
			ModelVersion:      "v1.0",
			LastModelUpdate:   now,
		},
		CrossClusterRefs: []string{},
	}

	vm.memoryDistribution = &MemoryDistributionInfo{
		Enabled:            true,
		TotalPages:         0,
		DistributedPages:   0,
		LocalPages:         0,
		RemotePages:        0,
		PageDistribution:   make(map[string]int64),
		DirtyPages:         []string{},
		LastSyncTime:       now,
		SyncInProgress:     false,
		CompressionEnabled: true,
		CompressionRatio:   1.0,
		CoherenceProtocol:  "MESI",
		AccessPatterns:     make(map[string]AccessPattern),
	}

	vm.stateCoordinator = coordinatorNode

	vm.federationContext = &FederationContext{
		HomeClusterID:      "local",
		CurrentClusterID:   "local",
		AuthorizedClusters: []string{},
		FederationTokens:   make(map[string]string),
		CrossClusterRefs:   []ClusterRef{},
		SecurityContext: SecurityContext{
			EncryptionEnabled: true,
			KeyVersion:        "v1",
			AccessPolicies:    []string{"default"},
			AuditEnabled:      true,
			Certificates:      make(map[string]string),
		},
		NetworkPolicy: NetworkPolicy{
			AllowedNetworks:    []string{"0.0.0.0/0"},
			BandwidthLimits:    make(map[string]int64),
			CompressionEnabled: true,
			PriorityClass:      "normal",
			QoSSettings: QoSSettings{
				MaxLatency:      100,  // 100ms
				MinBandwidth:    1024, // 1KB/s
				MaxJitter:       10,   // 10ms
				PacketLossLimit: 0.01, // 1%
			},
		},
	}

	vm.stateHistory = []StateSnapshot{}
	vm.conflictResolution = &ConflictResolutionInfo{
		ActiveConflicts:  []StateConflict{},
		ResolutionPolicy: ConflictPolicy{
			DefaultStrategy:  ConflictLastWriteWins,
			FieldPolicies:    make(map[string]ConflictStrategy),
			TimeoutMs:        30000, // 30 seconds
			RequireConsensus: false,
			VotingEnabled:    false,
		},
		LastResolution:   now,
		ConflictHistory:  []ResolvedConflict{},
		AutoResolveRules: []AutoResolveRule{},
	}

	vm.performanceMetrics = &DistributedMetrics{
		StateAccess: &AccessMetrics{
			LastMeasurement: now,
		},
		Migration: &MigrationMetrics{
			LastMigration: now,
		},
		MemoryDistribution: &MemoryMetrics{
			LastMemorySync: now,
		},
		NetworkPerformance: &NetworkMetrics{
			LastNetworkMeasurement: now,
		},
		ConsistencyMetrics: &ConsistencyMetrics{
			LastConsistencyCheck: now,
		},
		PredictiveMetrics: &PredictiveMetrics{
			LastModelUpdate: now,
		},
		LastUpdate: now,
	}

	log.Printf("Initialized distributed state for VM %s in shard %s with coordinator %s", vm.ID(), shardID, coordinatorNode)
	return nil
}

// GetDistributedState returns the distributed state information
func (vm *VM) GetDistributedState() *DistributedStateInfo {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	return vm.distributedState
}

// GetMemoryDistribution returns the memory distribution information
func (vm *VM) GetMemoryDistribution() *MemoryDistributionInfo {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	return vm.memoryDistribution
}

// GetStateCoordinator returns the state coordinator node ID
func (vm *VM) GetStateCoordinator() string {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	return vm.stateCoordinator
}

// GetFederationContext returns the federation context
func (vm *VM) GetFederationContext() *FederationContext {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	return vm.federationContext
}

// GetPerformanceMetrics returns the performance metrics
func (vm *VM) GetPerformanceMetrics() *DistributedMetrics {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	return vm.performanceMetrics
}

// UpdateDistributedState updates the distributed state with thread safety
func (vm *VM) UpdateDistributedState(updateFunc func(*DistributedStateInfo) error) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	if vm.distributedState == nil {
		return fmt.Errorf("distributed state not initialized for VM %s", vm.ID())
	}

	if err := updateFunc(vm.distributedState); err != nil {
		return err
	}

	vm.distributedState.StateVersion++
	vm.distributedState.LastStateUpdate = time.Now()
	vm.distributedState.StateDirty = true

	return nil
}

// UpdateMemoryDistribution updates the memory distribution information
func (vm *VM) UpdateMemoryDistribution(updateFunc func(*MemoryDistributionInfo) error) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	if vm.memoryDistribution == nil {
		return fmt.Errorf("memory distribution not initialized for VM %s", vm.ID())
	}

	return updateFunc(vm.memoryDistribution)
}

// SetStateCoordinator sets the state coordinator node
func (vm *VM) SetStateCoordinator(coordinatorNode string) {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()
	vm.stateCoordinator = coordinatorNode
}

// UpdateFederationContext updates the federation context
func (vm *VM) UpdateFederationContext(updateFunc func(*FederationContext) error) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	if vm.federationContext == nil {
		return fmt.Errorf("federation context not initialized for VM %s", vm.ID())
	}

	return updateFunc(vm.federationContext)
}

// AddStateSnapshot adds a state snapshot to the history
func (vm *VM) AddStateSnapshot(snapshot StateSnapshot) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	// Limit history to last 10 snapshots
	const maxSnapshots = 10
	vm.stateHistory = append(vm.stateHistory, snapshot)
	if len(vm.stateHistory) > maxSnapshots {
		vm.stateHistory = vm.stateHistory[len(vm.stateHistory)-maxSnapshots:]
	}

	return nil
}

// GetStateHistory returns the state history
func (vm *VM) GetStateHistory() []StateSnapshot {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	return vm.stateHistory
}

// AddConflict adds a state conflict for resolution
func (vm *VM) AddConflict(conflict StateConflict) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	if vm.conflictResolution == nil {
		return fmt.Errorf("conflict resolution not initialized for VM %s", vm.ID())
	}

	vm.conflictResolution.ActiveConflicts = append(vm.conflictResolution.ActiveConflicts, conflict)
	return nil
}

// ResolveConflict resolves a state conflict
func (vm *VM) ResolveConflict(conflictID string, resolution ConflictStrategy, resolvedBy string, finalValue interface{}) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	if vm.conflictResolution == nil {
		return fmt.Errorf("conflict resolution not initialized for VM %s", vm.ID())
	}

	// Find and remove the conflict
	var resolvedConflict *StateConflict
	for i, conflict := range vm.conflictResolution.ActiveConflicts {
		if conflict.ConflictID == conflictID {
			resolvedConflict = &conflict
			vm.conflictResolution.ActiveConflicts = append(
				vm.conflictResolution.ActiveConflicts[:i],
				vm.conflictResolution.ActiveConflicts[i+1:]...,
			)
			break
		}
	}

	if resolvedConflict == nil {
		return fmt.Errorf("conflict %s not found", conflictID)
	}

	// Add to resolution history
	now := time.Now()
	resolved := ResolvedConflict{
		ConflictID:       conflictID,
		Resolution:       resolution,
		ResolvedAt:       now,
		ResolvedBy:       resolvedBy,
		FinalValue:       finalValue,
		ConflictDuration: now.Sub(resolvedConflict.DetectedAt),
	}

	vm.conflictResolution.ConflictHistory = append(vm.conflictResolution.ConflictHistory, resolved)
	vm.conflictResolution.LastResolution = now

	// Limit history
	const maxHistory = 100
	if len(vm.conflictResolution.ConflictHistory) > maxHistory {
		vm.conflictResolution.ConflictHistory = vm.conflictResolution.ConflictHistory[len(vm.conflictResolution.ConflictHistory)-maxHistory:]
	}

	return nil
}

// UpdatePerformanceMetrics updates performance metrics
func (vm *VM) UpdatePerformanceMetrics(updateFunc func(*DistributedMetrics) error) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	if vm.performanceMetrics == nil {
		return fmt.Errorf("performance metrics not initialized for VM %s", vm.ID())
	}

	if err := updateFunc(vm.performanceMetrics); err != nil {
		return err
	}

	vm.performanceMetrics.LastUpdate = time.Now()
	return nil
}

// IsDistributedStateEnabled returns whether distributed state is enabled
func (vm *VM) IsDistributedStateEnabled() bool {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	return vm.distributedState != nil
}

// GetShardID returns the shard ID for this VM
func (vm *VM) GetShardID() string {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	if vm.distributedState == nil {
		return ""
	}
	return vm.distributedState.ShardID
}

// GetStateVersion returns the current state version
func (vm *VM) GetStateVersion() uint64 {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	if vm.distributedState == nil {
		return 0
	}
	return vm.distributedState.StateVersion
}

// IsStateDirty returns whether the state needs synchronization
func (vm *VM) IsStateDirty() bool {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	if vm.distributedState == nil {
		return false
	}
	return vm.distributedState.StateDirty
}

// MarkStateClean marks the state as synchronized
func (vm *VM) MarkStateClean() {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()
	if vm.distributedState != nil {
		vm.distributedState.StateDirty = false
		vm.distributedState.LastConsistencySync = time.Now()
	}
}

// GetActiveConflicts returns the list of active conflicts
func (vm *VM) GetActiveConflicts() []StateConflict {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	if vm.conflictResolution == nil {
		return []StateConflict{}
	}
	return vm.conflictResolution.ActiveConflicts
}

// HasActiveConflicts returns whether there are any active conflicts
func (vm *VM) HasActiveConflicts() bool {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	if vm.conflictResolution == nil {
		return false
	}
	return len(vm.conflictResolution.ActiveConflicts) > 0
}

// GetPredictiveCache returns the predictive cache information
func (vm *VM) GetPredictiveCache() *PredictiveCache {
	vm.stateLock.RLock()
	defer vm.stateLock.RUnlock()
	if vm.distributedState == nil {
		return nil
	}
	return vm.distributedState.PredictiveCache
}

// UpdatePredictiveCache updates the predictive cache information
func (vm *VM) UpdatePredictiveCache(updateFunc func(*PredictiveCache) error) error {
	vm.stateLock.Lock()
	defer vm.stateLock.Unlock()

	if vm.distributedState == nil || vm.distributedState.PredictiveCache == nil {
		return fmt.Errorf("predictive cache not initialized for VM %s", vm.ID())
	}

	return updateFunc(vm.distributedState.PredictiveCache)
}
