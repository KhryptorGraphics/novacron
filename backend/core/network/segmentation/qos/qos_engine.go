package qos

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
)

// QoSAlgorithm represents different QoS scheduling algorithms
type QoSAlgorithm string

const (
	AlgorithmHTB         QoSAlgorithm = "htb"     // Hierarchical Token Bucket
	AlgorithmCBQ         QoSAlgorithm = "cbq"     // Class Based Queuing
	AlgorithmHFSC        QoSAlgorithm = "hfsc"    // Hierarchical Fair Service Curve
	AlgorithmFQ_Codel    QoSAlgorithm = "fq_codel" // Fair Queuing CoDel
	AlgorithmCake        QoSAlgorithm = "cake"    // Common Applications Kept Enhanced
	AlgorithmTBF         QoSAlgorithm = "tbf"     // Token Bucket Filter
	AlgorithmSFQ         QoSAlgorithm = "sfq"     // Stochastic Fair Queuing
	AlgorithmPFIFO       QoSAlgorithm = "pfifo"   // Packet FIFO
)

// TrafficClass represents different classes of network traffic
type TrafficClass string

const (
	ClassRealtime     TrafficClass = "realtime"     // VoIP, video conferencing
	ClassInteractive  TrafficClass = "interactive"  // SSH, web browsing
	ClassStreaming    TrafficClass = "streaming"    // Video/audio streaming
	ClassBulk         TrafficClass = "bulk"         // File transfers, backups
	ClassBestEffort   TrafficClass = "best_effort"  // Default traffic
	ClassScavenger    TrafficClass = "scavenger"    // P2P, updates
)

// QoSMarkingType represents DSCP marking types
type QoSMarkingType string

const (
	MarkingDSCP QoSMarkingType = "dscp"
	MarkingTOS  QoSMarkingType = "tos"
	MarkingExp  QoSMarkingType = "exp"  // MPLS EXP bits
)

// BandwidthUnit represents bandwidth measurement units
type BandwidthUnit string

const (
	UnitBps  BandwidthUnit = "bps"   // bits per second
	UnitKbps BandwidthUnit = "kbps"  // kilobits per second
	UnitMbps BandwidthUnit = "mbps"  // megabits per second
	UnitGbps BandwidthUnit = "gbps"  // gigabits per second
)

// QoSPolicy represents a QoS policy for traffic management
type QoSPolicy struct {
	ID                string                  `json:"id"`
	Name              string                  `json:"name"`
	TenantID          string                  `json:"tenant_id"`
	Description       string                  `json:"description"`
	Priority          int                     `json:"priority"`
	Enabled           bool                    `json:"enabled"`
	Algorithm         QoSAlgorithm            `json:"algorithm"`
	Classes           map[string]*QoSClass    `json:"classes"`
	Rules             map[string]*QoSRule     `json:"rules"`
	GlobalLimits      *BandwidthLimits        `json:"global_limits"`
	MarkingConfig     *QoSMarking             `json:"marking_config,omitempty"`
	ShapingConfig     *TrafficShaping         `json:"shaping_config,omitempty"`
	SchedulingConfig  *SchedulingConfig       `json:"scheduling_config,omitempty"`
	MonitoringConfig  *MonitoringConfig       `json:"monitoring_config,omitempty"`
	Metadata          map[string]string       `json:"metadata,omitempty"`
	CreatedAt         time.Time               `json:"created_at"`
	UpdatedAt         time.Time               `json:"updated_at"`
	CreatedBy         string                  `json:"created_by"`
}

// QoSClass represents a traffic class with specific QoS parameters
type QoSClass struct {
	ID                string           `json:"id"`
	Name              string           `json:"name"`
	Class             TrafficClass     `json:"class"`
	Priority          int              `json:"priority"`
	Weight            int              `json:"weight"`
	BandwidthLimits   *BandwidthLimits `json:"bandwidth_limits"`
	LatencyTarget     time.Duration    `json:"latency_target"`
	JitterTarget      time.Duration    `json:"jitter_target"`
	PacketLossTarget  float64          `json:"packet_loss_target"` // percentage
	BufferSize        uint64           `json:"buffer_size"`        // bytes
	QueueDepth        uint32           `json:"queue_depth"`        // packets
	DSCPMarking       uint8            `json:"dscp_marking"`
	DropPolicy        string           `json:"drop_policy"`        // tail_drop, red, wred
	ECNEnabled        bool             `json:"ecn_enabled"`        // Explicit Congestion Notification
	Guaranteed        bool             `json:"guaranteed"`         // Guaranteed bandwidth
	Borrowing         bool             `json:"borrowing"`          // Can borrow unused bandwidth
	ParentClass       string           `json:"parent_class,omitempty"`
	ChildClasses      []string         `json:"child_classes,omitempty"`
}

// BandwidthLimits defines bandwidth limits for a class or policy
type BandwidthLimits struct {
	MinRate       uint64        `json:"min_rate"`        // Guaranteed bandwidth
	MaxRate       uint64        `json:"max_rate"`        // Maximum bandwidth
	BurstRate     uint64        `json:"burst_rate"`      // Burst allowance
	Unit          BandwidthUnit `json:"unit"`
	Ceil          uint64        `json:"ceil,omitempty"`  // Ceiling rate for HTB
	Floor         uint64        `json:"floor,omitempty"` // Floor rate for guaranteed service
	SharedRate    uint64        `json:"shared_rate,omitempty"` // Shared pool rate
}

// QoSRule defines traffic classification rules
type QoSRule struct {
	ID              string            `json:"id"`
	Name            string            `json:"name"`
	Priority        int               `json:"priority"`
	Enabled         bool              `json:"enabled"`
	ClassID         string            `json:"class_id"`
	MatchCriteria   *MatchCriteria    `json:"match_criteria"`
	Actions         *QoSActions       `json:"actions"`
	Statistics      *RuleStatistics   `json:"statistics"`
	TimeRestrictions *TimeRestrictions `json:"time_restrictions,omitempty"`
}

// MatchCriteria defines traffic matching criteria
type MatchCriteria struct {
	SrcIP           string            `json:"src_ip,omitempty"`
	DstIP           string            `json:"dst_ip,omitempty"`
	SrcPort         uint16            `json:"src_port,omitempty"`
	DstPort         uint16            `json:"dst_port,omitempty"`
	Protocol        string            `json:"protocol,omitempty"`
	DSCP            uint8             `json:"dscp,omitempty"`
	ToS             uint8             `json:"tos,omitempty"`
	VLANTag         uint16            `json:"vlan_tag,omitempty"`
	InterfaceName   string            `json:"interface_name,omitempty"`
	ApplicationID   string            `json:"application_id,omitempty"`
	UserID          string            `json:"user_id,omitempty"`
	DeviceType      string            `json:"device_type,omitempty"`
	TrafficType     string            `json:"traffic_type,omitempty"`
	PayloadPattern  string            `json:"payload_pattern,omitempty"` // regex
	PacketSize      *SizeRange        `json:"packet_size,omitempty"`
	FlowRate        *RateRange        `json:"flow_rate,omitempty"`
	CustomFields    map[string]string `json:"custom_fields,omitempty"`
}

// SizeRange represents a range of packet sizes
type SizeRange struct {
	Min uint32 `json:"min"`
	Max uint32 `json:"max"`
}

// RateRange represents a range of flow rates
type RateRange struct {
	Min uint64        `json:"min"`
	Max uint64        `json:"max"`
	Unit BandwidthUnit `json:"unit"`
}

// QoSActions defines actions to take on matching traffic
type QoSActions struct {
	SetDSCP         *uint8  `json:"set_dscp,omitempty"`
	SetToS          *uint8  `json:"set_tos,omitempty"`
	SetPriority     *int    `json:"set_priority,omitempty"`
	RateLimit       *uint64 `json:"rate_limit,omitempty"`       // bps
	DropProbability *float64 `json:"drop_probability,omitempty"` // 0.0-1.0
	RedirectQueue   *string `json:"redirect_queue,omitempty"`
	Mirror          *string `json:"mirror,omitempty"`           // mirror destination
	Log             bool    `json:"log"`
	Count           bool    `json:"count"`
	Police          *PolicingAction `json:"police,omitempty"`
}

// PolicingAction defines traffic policing parameters
type PolicingAction struct {
	Rate       uint64 `json:"rate"`        // bits per second
	BurstSize  uint64 `json:"burst_size"`  // bytes
	ExceedAction string `json:"exceed_action"` // drop, remark
	ViolateAction string `json:"violate_action"` // drop, remark
	ConformAction string `json:"conform_action"` // transmit, remark
}

// RuleStatistics tracks rule matching statistics
type RuleStatistics struct {
	PacketsMatched  uint64    `json:"packets_matched"`
	BytesMatched    uint64    `json:"bytes_matched"`
	LastMatch       time.Time `json:"last_match"`
	MatchRate       float64   `json:"match_rate"`       // matches per second
	DropCount       uint64    `json:"drop_count"`
	RemarkCount     uint64    `json:"remark_count"`
}

// TimeRestrictions defines time-based rule application
type TimeRestrictions struct {
	StartTime   string   `json:"start_time"`   // HH:MM format
	EndTime     string   `json:"end_time"`     // HH:MM format
	DaysOfWeek  []string `json:"days_of_week"` // mon, tue, wed, thu, fri, sat, sun
	DateRange   *DateRange `json:"date_range,omitempty"`
	Timezone    string   `json:"timezone"`
}

// DateRange defines a date range restriction
type DateRange struct {
	StartDate string `json:"start_date"` // YYYY-MM-DD format
	EndDate   string `json:"end_date"`   // YYYY-MM-DD format
}

// QoSMarking defines DSCP and ToS marking configuration
type QoSMarking struct {
	Enabled      bool            `json:"enabled"`
	DefaultDSCP  uint8           `json:"default_dscp"`
	DefaultToS   uint8           `json:"default_tos"`
	ClassMapping map[string]uint8 `json:"class_mapping"` // ClassID -> DSCP/ToS
	PreserveMarkings bool        `json:"preserve_markings"`
}

// TrafficShaping defines traffic shaping configuration
type TrafficShaping struct {
	Enabled        bool              `json:"enabled"`
	Algorithm      QoSAlgorithm      `json:"algorithm"`
	DefaultQueue   string            `json:"default_queue"`
	QueueConfig    map[string]*QueueConfig `json:"queue_config"`
	HierarchyDepth int               `json:"hierarchy_depth"`
	HTBConfig      *HTBConfig        `json:"htb_config,omitempty"`
	CBQConfig      *CBQConfig        `json:"cbq_config,omitempty"`
	HFSCConfig     *HFSCConfig       `json:"hfsc_config,omitempty"`
}

// QueueConfig defines queue-specific configuration
type QueueConfig struct {
	QueueID       string           `json:"queue_id"`
	Algorithm     QoSAlgorithm     `json:"algorithm"`
	Priority      int              `json:"priority"`
	Weight        int              `json:"weight"`
	Limits        *BandwidthLimits `json:"limits"`
	BufferSize    uint64           `json:"buffer_size"`
	ECNThreshold  uint32           `json:"ecn_threshold"`
	REDConfig     *REDConfig       `json:"red_config,omitempty"`
	WREDConfig    *WREDConfig      `json:"wred_config,omitempty"`
}

// HTBConfig defines Hierarchical Token Bucket configuration
type HTBConfig struct {
	DefaultClass  string  `json:"default_class"`
	R2Q          uint32  `json:"r2q"`          // Rate to quantum ratio
	DCacheSize   uint32  `json:"dcache_size"`  // Direct cache size
	Version      uint32  `json:"version"`
	RateEst      bool    `json:"rate_est"`     // Rate estimation
}

// CBQConfig defines Class Based Queuing configuration  
type CBQConfig struct {
	BandwidthEstimator string `json:"bandwidth_estimator"`
	CellSize          uint16 `json:"cell_size"`
	MPU              uint16 `json:"mpu"`        // Minimum packet unit
	Overhead         uint16 `json:"overhead"`
	LinkLayer        string `json:"link_layer"`
}

// HFSCConfig defines Hierarchical Fair Service Curve configuration
type HFSCConfig struct {
	DefaultClass string `json:"default_class"`
	RScale      uint32 `json:"rscale"`      // Rate scale
	GScale      uint32 `json:"gscale"`      // Guaranteed scale
	UScale      uint32 `json:"uscale"`      // Upper limit scale
}

// REDConfig defines Random Early Detection configuration
type REDConfig struct {
	MinThreshold  uint32  `json:"min_threshold"`  // packets
	MaxThreshold  uint32  `json:"max_threshold"`  // packets
	MaxProbability float64 `json:"max_probability"` // 0.0-1.0
	WQ            float64 `json:"wq"`             // Weight queue
	Scell         uint8   `json:"scell"`
	Flags         uint8   `json:"flags"`
}

// WREDConfig defines Weighted Random Early Detection configuration
type WREDConfig struct {
	Profiles map[uint8]*REDConfig `json:"profiles"` // DSCP -> RED config
	ECN      bool                 `json:"ecn"`
}

// SchedulingConfig defines packet scheduling configuration
type SchedulingConfig struct {
	Algorithm      QoSAlgorithm           `json:"algorithm"`
	Quantum        uint32                 `json:"quantum"`      // For DRR, WRR
	Weights        map[string]int         `json:"weights"`      // ClassID -> Weight
	Deficits       map[string]uint32      `json:"deficits"`     // For DRR
	ServiceCurves  map[string]*ServiceCurve `json:"service_curves,omitempty"`
	SchedulingMode string                 `json:"scheduling_mode"` // strict, wrr, drr, wfq
}

// ServiceCurve defines HFSC service curve parameters
type ServiceCurve struct {
	RealTimeRate    uint64 `json:"realtime_rate"`    // Real-time service rate
	LinkShareRate   uint64 `json:"linkshare_rate"`   // Link-sharing rate
	UpperLimitRate  uint64 `json:"upperlimit_rate"`  // Upper limit rate
	RealTimeDelay   time.Duration `json:"realtime_delay"`   // Real-time delay
	LinkShareDelay  time.Duration `json:"linkshare_delay"`  // Link-sharing delay
}

// MonitoringConfig defines QoS monitoring configuration
type MonitoringConfig struct {
	Enabled              bool          `json:"enabled"`
	CollectionInterval   time.Duration `json:"collection_interval"`
	RetentionPeriod      time.Duration `json:"retention_period"`
	EnableDetailedStats  bool          `json:"enable_detailed_stats"`
	EnableLatencyTracking bool         `json:"enable_latency_tracking"`
	EnableThroughputTracking bool      `json:"enable_throughput_tracking"`
	EnableQueueMonitoring bool         `json:"enable_queue_monitoring"`
	AlertThresholds      *AlertThresholds `json:"alert_thresholds,omitempty"`
	ExportConfig         *MetricsExport `json:"export_config,omitempty"`
}

// AlertThresholds defines thresholds for QoS alerts
type AlertThresholds struct {
	BandwidthUtilization float64       `json:"bandwidth_utilization"` // percentage
	LatencyThreshold     time.Duration `json:"latency_threshold"`
	PacketLossThreshold  float64       `json:"packet_loss_threshold"` // percentage
	QueueDepthThreshold  uint32        `json:"queue_depth_threshold"`
	DropRateThreshold    float64       `json:"drop_rate_threshold"`   // percentage
}

// MetricsExport defines metrics export configuration
type MetricsExport struct {
	Enabled     bool     `json:"enabled"`
	Formats     []string `json:"formats"`     // prometheus, json, csv
	Endpoints   []string `json:"endpoints"`   // URLs for export
	Interval    time.Duration `json:"interval"`
	Compression bool     `json:"compression"`
}

// QoSStatistics tracks QoS performance metrics
type QoSStatistics struct {
	PolicyID            string                    `json:"policy_id"`
	ClassStatistics     map[string]*ClassStats   `json:"class_statistics"`
	RuleStatistics      map[string]*RuleStatistics `json:"rule_statistics"`
	InterfaceStatistics map[string]*InterfaceStats `json:"interface_statistics"`
	OverallStats        *OverallStats             `json:"overall_stats"`
	LastUpdated         time.Time                 `json:"last_updated"`
}

// ClassStats tracks statistics for a QoS class
type ClassStats struct {
	ClassID         string    `json:"class_id"`
	PacketsSent     uint64    `json:"packets_sent"`
	BytesSent       uint64    `json:"bytes_sent"`
	PacketsDropped  uint64    `json:"packets_dropped"`
	BytesDropped    uint64    `json:"bytes_dropped"`
	PacketsQueued   uint32    `json:"packets_queued"`
	QueueDepth      uint32    `json:"queue_depth"`
	AverageLatency  time.Duration `json:"average_latency"`
	MaxLatency      time.Duration `json:"max_latency"`
	Jitter          time.Duration `json:"jitter"`
	BandwidthUsed   uint64    `json:"bandwidth_used"`   // bps
	UtilizationRate float64   `json:"utilization_rate"` // percentage
	ConformCount    uint64    `json:"conform_count"`
	ExceedCount     uint64    `json:"exceed_count"`
	ViolateCount    uint64    `json:"violate_count"`
	LastActivity    time.Time `json:"last_activity"`
}

// InterfaceStats tracks interface-level QoS statistics
type InterfaceStats struct {
	InterfaceName   string    `json:"interface_name"`
	TotalBandwidth  uint64    `json:"total_bandwidth"`  // bps
	UsedBandwidth   uint64    `json:"used_bandwidth"`   // bps
	AvailableBandwidth uint64 `json:"available_bandwidth"` // bps
	UtilizationRate float64   `json:"utilization_rate"` // percentage
	PacketsProcessed uint64   `json:"packets_processed"`
	BytesProcessed  uint64    `json:"bytes_processed"`
	ErrorCount      uint64    `json:"error_count"`
	DropCount       uint64    `json:"drop_count"`
	QueueOverflows  uint64    `json:"queue_overflows"`
	LastUpdated     time.Time `json:"last_updated"`
}

// OverallStats tracks overall QoS system statistics
type OverallStats struct {
	TotalPolicies       int       `json:"total_policies"`
	ActivePolicies      int       `json:"active_policies"`
	TotalClasses        int       `json:"total_classes"`
	TotalRules          int       `json:"total_rules"`
	TotalInterfaces     int       `json:"total_interfaces"`
	PacketsClassified   uint64    `json:"packets_classified"`
	PacketsUnclassified uint64    `json:"packets_unclassified"`
	ClassificationRate  float64   `json:"classification_rate"` // percentage
	SystemLatency       time.Duration `json:"system_latency"`
	ProcessingLoad      float64   `json:"processing_load"`     // percentage
}

// QoSEvent represents QoS-related events
type QoSEvent struct {
	Type        string      `json:"type"`
	TenantID    string      `json:"tenant_id"`
	PolicyID    string      `json:"policy_id,omitempty"`
	ClassID     string      `json:"class_id,omitempty"`
	RuleID      string      `json:"rule_id,omitempty"`
	Severity    string      `json:"severity"`
	Message     string      `json:"message"`
	Data        interface{} `json:"data,omitempty"`
	Timestamp   time.Time   `json:"timestamp"`
}

// QoSEventListener is a callback for QoS events
type QoSEventListener func(event QoSEvent)

// QoSEngine manages Quality of Service policies and traffic shaping
type QoSEngine struct {
	ID              string                    `json:"id"`
	Name            string                    `json:"name"`
	Policies        map[string]*QoSPolicy     `json:"policies"`
	Statistics      map[string]*QoSStatistics `json:"statistics"`
	EventListeners  []QoSEventListener        `json:"-"`
	Config          *QoSEngineConfig          `json:"config"`
	ClassifierEngine *TrafficClassifier       `json:"-"`
	ShaperEngine    *TrafficShaper            `json:"-"`
	MonitoringEngine *QoSMonitor              `json:"-"`
	mutex           sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
	wg              sync.WaitGroup
}

// QoSEngineConfig holds QoS engine configuration
type QoSEngineConfig struct {
	DefaultAlgorithm      QoSAlgorithm  `json:"default_algorithm"`
	MaxPolicies          int           `json:"max_policies"`
	MaxClassesPerPolicy  int           `json:"max_classes_per_policy"`
	MaxRulesPerPolicy    int           `json:"max_rules_per_policy"`
	StatisticsInterval   time.Duration `json:"statistics_interval"`
	EnableRealTimeStats  bool          `json:"enable_realtime_stats"`
	EnablePerformanceOptimization bool `json:"enable_performance_optimization"`
	CacheSize            int           `json:"cache_size"`
	WorkerCount          int           `json:"worker_count"`
	BufferSize           int           `json:"buffer_size"`
}

// TrafficClassifier classifies network traffic into QoS classes
type TrafficClassifier struct {
	engine    *QoSEngine
	rules     []*QoSRule
	cache     map[string]string // Flow -> ClassID cache
	mutex     sync.RWMutex
	hitCount  uint64
	missCount uint64
}

// TrafficShaper shapes network traffic according to QoS policies  
type TrafficShaper struct {
	engine     *QoSEngine
	queues     map[string]*TrafficQueue
	schedulers map[string]*PacketScheduler
	shapers    map[string]*BandwidthShaper
	mutex      sync.RWMutex
	ctx        context.Context
	cancel     context.CancelFunc
	wg         sync.WaitGroup
}

// TrafficQueue represents a traffic queue for QoS
type TrafficQueue struct {
	ID          string
	ClassID     string
	Algorithm   QoSAlgorithm
	Config      *QueueConfig
	Packets     chan *QoSPacket
	Stats       *QueueStats
	mutex       sync.RWMutex
	ctx         context.Context
	cancel      context.CancelFunc
}

// QoSPacket represents a packet with QoS metadata
type QoSPacket struct {
	ID          string        `json:"id"`
	Data        []byte        `json:"data"`
	Size        uint32        `json:"size"`
	Priority    int           `json:"priority"`
	ClassID     string        `json:"class_id"`
	DSCP        uint8         `json:"dscp"`
	Timestamp   time.Time     `json:"timestamp"`
	EnqueueTime time.Time     `json:"enqueue_time"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// QueueStats tracks queue-level statistics
type QueueStats struct {
	EnqueuedPackets uint64        `json:"enqueued_packets"`
	DequeuedPackets uint64        `json:"dequeued_packets"`
	DroppedPackets  uint64        `json:"dropped_packets"`
	CurrentDepth    uint32        `json:"current_depth"`
	MaxDepth        uint32        `json:"max_depth"`
	AverageDelay    time.Duration `json:"average_delay"`
	TotalBytes      uint64        `json:"total_bytes"`
	LastActivity    time.Time     `json:"last_activity"`
}

// PacketScheduler schedules packets from multiple queues
type PacketScheduler struct {
	ID         string
	Algorithm  QoSAlgorithm
	Config     *SchedulingConfig
	Queues     []*TrafficQueue
	CurrentPos int
	Deficits   map[string]uint32
	mutex      sync.RWMutex
}

// BandwidthShaper enforces bandwidth limits
type BandwidthShaper struct {
	ID          string
	Config      *BandwidthLimits
	TokenBucket *TokenBucket
	Stats       *ShaperStats
	mutex       sync.RWMutex
}

// TokenBucket implements token bucket algorithm for rate limiting
type TokenBucket struct {
	Capacity     uint64
	Tokens       uint64
	RefillRate   uint64
	LastRefill   time.Time
	BurstAllowed bool
	mutex        sync.Mutex
}

// ShaperStats tracks bandwidth shaper statistics
type ShaperStats struct {
	PacketsConform uint64    `json:"packets_conform"`
	PacketsExceed  uint64    `json:"packets_exceed"`
	PacketsViolate uint64    `json:"packets_violate"`
	BytesConform   uint64    `json:"bytes_conform"`
	BytesExceed    uint64    `json:"bytes_exceed"`
	BytesViolate   uint64    `json:"bytes_violate"`
	LastActivity   time.Time `json:"last_activity"`
}

// QoSMonitor monitors QoS performance and generates alerts
type QoSMonitor struct {
	engine      *QoSEngine
	collectors  map[string]*MetricsCollector
	alertEngine *AlertEngine
	exporters   []MetricsExporter
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
}

// MetricsCollector collects QoS metrics
type MetricsCollector struct {
	ID              string
	Type            string // class, rule, interface, overall
	CollectionInterval time.Duration
	LastCollection  time.Time
	Metrics         map[string]interface{}
	mutex           sync.RWMutex
}

// AlertEngine generates QoS alerts
type AlertEngine struct {
	thresholds  *AlertThresholds
	alerts      chan *QoSAlert
	handlers    []AlertHandler
	ctx         context.Context
	cancel      context.CancelFunc
	wg          sync.WaitGroup
}

// QoSAlert represents a QoS alert
type QoSAlert struct {
	ID          string      `json:"id"`
	Type        string      `json:"type"`
	Severity    string      `json:"severity"`
	Message     string      `json:"message"`
	PolicyID    string      `json:"policy_id,omitempty"`
	ClassID     string      `json:"class_id,omitempty"`
	Threshold   interface{} `json:"threshold"`
	CurrentValue interface{} `json:"current_value"`
	Timestamp   time.Time   `json:"timestamp"`
}

// AlertHandler handles QoS alerts
type AlertHandler func(alert *QoSAlert)

// MetricsExporter exports QoS metrics
type MetricsExporter interface {
	Export(metrics map[string]interface{}) error
	GetFormat() string
}

// NewQoSEngine creates a new QoS engine
func NewQoSEngine(name string, config *QoSEngineConfig) *QoSEngine {
	ctx, cancel := context.WithCancel(context.Background())
	
	engine := &QoSEngine{
		ID:           uuid.New().String(),
		Name:         name,
		Policies:     make(map[string]*QoSPolicy),
		Statistics:   make(map[string]*QoSStatistics),
		EventListeners: make([]QoSEventListener, 0),
		Config:       config,
		ctx:          ctx,
		cancel:       cancel,
	}
	
	// Initialize components
	engine.ClassifierEngine = NewTrafficClassifier(engine)
	engine.ShaperEngine = NewTrafficShaper(engine)
	engine.MonitoringEngine = NewQoSMonitor(engine)
	
	return engine
}

// Start starts the QoS engine
func (qe *QoSEngine) Start() error {
	qe.mutex.Lock()
	defer qe.mutex.Unlock()
	
	log.Printf("Starting QoS engine %s", qe.Name)
	
	// Start components
	if err := qe.ClassifierEngine.Start(); err != nil {
		return fmt.Errorf("failed to start traffic classifier: %w", err)
	}
	
	if err := qe.ShaperEngine.Start(); err != nil {
		return fmt.Errorf("failed to start traffic shaper: %w", err)
	}
	
	if err := qe.MonitoringEngine.Start(); err != nil {
		return fmt.Errorf("failed to start monitoring engine: %w", err)
	}
	
	// Start statistics collection
	if qe.Config.StatisticsInterval > 0 {
		qe.wg.Add(1)
		go qe.collectStatistics()
	}
	
	qe.emitEvent(QoSEvent{
		Type:      "engine_started",
		Message:   fmt.Sprintf("QoS engine %s started", qe.Name),
		Timestamp: time.Now(),
	})
	
	log.Printf("QoS engine %s started successfully", qe.Name)
	return nil
}

// Stop stops the QoS engine
func (qe *QoSEngine) Stop() error {
	qe.mutex.Lock()
	defer qe.mutex.Unlock()
	
	log.Printf("Stopping QoS engine %s", qe.Name)
	
	qe.cancel()
	qe.wg.Wait()
	
	// Stop components
	if qe.MonitoringEngine != nil {
		qe.MonitoringEngine.Stop()
	}
	
	if qe.ShaperEngine != nil {
		qe.ShaperEngine.Stop()
	}
	
	if qe.ClassifierEngine != nil {
		qe.ClassifierEngine.Stop()
	}
	
	qe.emitEvent(QoSEvent{
		Type:      "engine_stopped",
		Message:   fmt.Sprintf("QoS engine %s stopped", qe.Name),
		Timestamp: time.Now(),
	})
	
	log.Printf("QoS engine %s stopped", qe.Name)
	return nil
}

// CreatePolicy creates a new QoS policy
func (qe *QoSEngine) CreatePolicy(policy *QoSPolicy) error {
	qe.mutex.Lock()
	defer qe.mutex.Unlock()
	
	if policy.ID == "" {
		policy.ID = uuid.New().String()
	}
	
	// Validate policy
	if err := qe.validatePolicy(policy); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}
	
	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()
	
	// Initialize classes and rules
	if policy.Classes == nil {
		policy.Classes = make(map[string]*QoSClass)
	}
	if policy.Rules == nil {
		policy.Rules = make(map[string]*QoSRule)
	}
	
	qe.Policies[policy.ID] = policy
	
	// Initialize statistics
	qe.Statistics[policy.ID] = &QoSStatistics{
		PolicyID:            policy.ID,
		ClassStatistics:     make(map[string]*ClassStats),
		RuleStatistics:      make(map[string]*RuleStatistics),
		InterfaceStatistics: make(map[string]*InterfaceStats),
		OverallStats: &OverallStats{
			TotalPolicies: len(qe.Policies),
		},
		LastUpdated: time.Now(),
	}
	
	// Apply policy to traffic shaper
	if err := qe.ShaperEngine.ApplyPolicy(policy); err != nil {
		delete(qe.Policies, policy.ID)
		delete(qe.Statistics, policy.ID)
		return fmt.Errorf("failed to apply policy to shaper: %w", err)
	}
	
	qe.emitEvent(QoSEvent{
		Type:      "policy_created",
		PolicyID:  policy.ID,
		Message:   fmt.Sprintf("QoS policy %s created", policy.Name),
		Data:      policy,
		Timestamp: time.Now(),
	})
	
	log.Printf("Created QoS policy %s (%s)", policy.Name, policy.ID)
	return nil
}

// UpdatePolicy updates an existing QoS policy
func (qe *QoSEngine) UpdatePolicy(policy *QoSPolicy) error {
	qe.mutex.Lock()
	defer qe.mutex.Unlock()
	
	existingPolicy, exists := qe.Policies[policy.ID]
	if !exists {
		return fmt.Errorf("policy %s not found", policy.ID)
	}
	
	// Validate updated policy
	if err := qe.validatePolicy(policy); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}
	
	policy.CreatedAt = existingPolicy.CreatedAt
	policy.UpdatedAt = time.Now()
	
	qe.Policies[policy.ID] = policy
	
	// Update traffic shaper
	if err := qe.ShaperEngine.UpdatePolicy(policy); err != nil {
		return fmt.Errorf("failed to update policy in shaper: %w", err)
	}
	
	qe.emitEvent(QoSEvent{
		Type:      "policy_updated",
		PolicyID:  policy.ID,
		Message:   fmt.Sprintf("QoS policy %s updated", policy.Name),
		Data:      policy,
		Timestamp: time.Now(),
	})
	
	return nil
}

// DeletePolicy deletes a QoS policy
func (qe *QoSEngine) DeletePolicy(policyID string) error {
	qe.mutex.Lock()
	defer qe.mutex.Unlock()
	
	policy, exists := qe.Policies[policyID]
	if !exists {
		return fmt.Errorf("policy %s not found", policyID)
	}
	
	// Remove from traffic shaper
	if err := qe.ShaperEngine.RemovePolicy(policyID); err != nil {
		return fmt.Errorf("failed to remove policy from shaper: %w", err)
	}
	
	delete(qe.Policies, policyID)
	delete(qe.Statistics, policyID)
	
	qe.emitEvent(QoSEvent{
		Type:      "policy_deleted",
		PolicyID:  policyID,
		Message:   fmt.Sprintf("QoS policy %s deleted", policy.Name),
		Timestamp: time.Now(),
	})
	
	return nil
}

// GetPolicy retrieves a QoS policy
func (qe *QoSEngine) GetPolicy(policyID string) (*QoSPolicy, error) {
	qe.mutex.RLock()
	defer qe.mutex.RUnlock()
	
	policy, exists := qe.Policies[policyID]
	if !exists {
		return nil, fmt.Errorf("policy %s not found", policyID)
	}
	
	return policy, nil
}

// ListPolicies returns all QoS policies
func (qe *QoSEngine) ListPolicies() []*QoSPolicy {
	qe.mutex.RLock()
	defer qe.mutex.RUnlock()
	
	policies := make([]*QoSPolicy, 0, len(qe.Policies))
	for _, policy := range qe.Policies {
		policies = append(policies, policy)
	}
	
	return policies
}

// GetStatistics returns QoS statistics for a policy
func (qe *QoSEngine) GetStatistics(policyID string) (*QoSStatistics, error) {
	qe.mutex.RLock()
	defer qe.mutex.RUnlock()
	
	stats, exists := qe.Statistics[policyID]
	if !exists {
		return nil, fmt.Errorf("statistics for policy %s not found", policyID)
	}
	
	return stats, nil
}

// ProcessPacket processes a packet through the QoS system
func (qe *QoSEngine) ProcessPacket(data []byte, metadata map[string]interface{}) (*QoSPacket, error) {
	packet := &QoSPacket{
		ID:        uuid.New().String(),
		Data:      data,
		Size:      uint32(len(data)),
		Timestamp: time.Now(),
		Metadata:  metadata,
	}
	
	// Classify packet
	classID, err := qe.ClassifierEngine.ClassifyPacket(packet)
	if err != nil {
		return nil, fmt.Errorf("packet classification failed: %w", err)
	}
	
	packet.ClassID = classID
	
	// Shape packet
	result, err := qe.ShaperEngine.ShapePacket(packet)
	if err != nil {
		return nil, fmt.Errorf("packet shaping failed: %w", err)
	}
	
	return result, nil
}

// validatePolicy validates a QoS policy
func (qe *QoSEngine) validatePolicy(policy *QoSPolicy) error {
	if policy.Name == "" {
		return fmt.Errorf("policy name cannot be empty")
	}
	
	if policy.TenantID == "" {
		return fmt.Errorf("tenant ID cannot be empty")
	}
	
	if len(qe.Policies) >= qe.Config.MaxPolicies {
		return fmt.Errorf("maximum number of policies (%d) reached", qe.Config.MaxPolicies)
	}
	
	if len(policy.Classes) > qe.Config.MaxClassesPerPolicy {
		return fmt.Errorf("maximum number of classes per policy (%d) exceeded", qe.Config.MaxClassesPerPolicy)
	}
	
	if len(policy.Rules) > qe.Config.MaxRulesPerPolicy {
		return fmt.Errorf("maximum number of rules per policy (%d) exceeded", qe.Config.MaxRulesPerPolicy)
	}
	
	// Validate classes
	for _, class := range policy.Classes {
		if err := qe.validateClass(class); err != nil {
			return fmt.Errorf("class %s validation failed: %w", class.ID, err)
		}
	}
	
	// Validate rules
	for _, rule := range policy.Rules {
		if err := qe.validateRule(rule); err != nil {
			return fmt.Errorf("rule %s validation failed: %w", rule.ID, err)
		}
	}
	
	return nil
}

// validateClass validates a QoS class
func (qe *QoSEngine) validateClass(class *QoSClass) error {
	if class.Name == "" {
		return fmt.Errorf("class name cannot be empty")
	}
	
	if class.BandwidthLimits != nil {
		if class.BandwidthLimits.MinRate > class.BandwidthLimits.MaxRate {
			return fmt.Errorf("minimum rate cannot exceed maximum rate")
		}
	}
	
	if class.PacketLossTarget < 0 || class.PacketLossTarget > 100 {
		return fmt.Errorf("packet loss target must be between 0 and 100")
	}
	
	return nil
}

// validateRule validates a QoS rule
func (qe *QoSEngine) validateRule(rule *QoSRule) error {
	if rule.Name == "" {
		return fmt.Errorf("rule name cannot be empty")
	}
	
	if rule.ClassID == "" {
		return fmt.Errorf("class ID cannot be empty")
	}
	
	if rule.MatchCriteria == nil {
		return fmt.Errorf("match criteria cannot be nil")
	}
	
	return nil
}

// collectStatistics collects QoS statistics
func (qe *QoSEngine) collectStatistics() {
	defer qe.wg.Done()
	
	ticker := time.NewTicker(qe.Config.StatisticsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-qe.ctx.Done():
			return
		case <-ticker.C:
			qe.updateStatistics()
		}
	}
}

// updateStatistics updates QoS statistics
func (qe *QoSEngine) updateStatistics() {
	qe.mutex.Lock()
	defer qe.mutex.Unlock()
	
	for policyID, stats := range qe.Statistics {
		stats.LastUpdated = time.Now()
		
		// Update overall stats
		stats.OverallStats.TotalPolicies = len(qe.Policies)
		activeCount := 0
		for _, policy := range qe.Policies {
			if policy.Enabled {
				activeCount++
			}
		}
		stats.OverallStats.ActivePolicies = activeCount
		
		// Update class statistics (would collect from actual traffic shaper)
		for classID := range qe.Policies[policyID].Classes {
			if _, exists := stats.ClassStatistics[classID]; !exists {
				stats.ClassStatistics[classID] = &ClassStats{
					ClassID:      classID,
					LastActivity: time.Now(),
				}
			}
		}
	}
}

// emitEvent emits a QoS event
func (qe *QoSEngine) emitEvent(event QoSEvent) {
	for _, listener := range qe.EventListeners {
		go func(l QoSEventListener, e QoSEvent) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("QoS event listener panic: %v", r)
				}
			}()
			l(e)
		}(listener, event)
	}
}

// AddEventListener adds a QoS event listener
func (qe *QoSEngine) AddEventListener(listener QoSEventListener) {
	qe.mutex.Lock()
	defer qe.mutex.Unlock()
	
	qe.EventListeners = append(qe.EventListeners, listener)
}

// Component factory functions (simplified implementations)

// NewTrafficClassifier creates a new traffic classifier
func NewTrafficClassifier(engine *QoSEngine) *TrafficClassifier {
	return &TrafficClassifier{
		engine: engine,
		rules:  make([]*QoSRule, 0),
		cache:  make(map[string]string),
	}
}

// Start starts the traffic classifier
func (tc *TrafficClassifier) Start() error {
	// Load rules from policies
	tc.updateRules()
	return nil
}

// Stop stops the traffic classifier (no-op for now)
func (tc *TrafficClassifier) Stop() error {
	return nil
}

// ClassifyPacket classifies a packet into a QoS class
func (tc *TrafficClassifier) ClassifyPacket(packet *QoSPacket) (string, error) {
	// Generate flow key for caching
	flowKey := fmt.Sprintf("%s_%d", packet.ID, packet.Size)
	
	// Check cache
	tc.mutex.RLock()
	if classID, exists := tc.cache[flowKey]; exists {
		tc.mutex.RUnlock()
		tc.hitCount++
		return classID, nil
	}
	tc.mutex.RUnlock()
	
	tc.missCount++
	
	// Classify based on rules
	for _, rule := range tc.rules {
		if tc.ruleMatches(rule, packet) {
			// Cache result
			tc.mutex.Lock()
			tc.cache[flowKey] = rule.ClassID
			tc.mutex.Unlock()
			
			return rule.ClassID, nil
		}
	}
	
	// Default class
	return "default", nil
}

// ruleMatches checks if a rule matches a packet
func (tc *TrafficClassifier) ruleMatches(rule *QoSRule, packet *QoSPacket) bool {
	if !rule.Enabled {
		return false
	}
	
	// Simplified matching logic
	// In a real implementation, this would check all match criteria
	
	return true // For now, always match first rule
}

// updateRules updates the classifier rules from policies
func (tc *TrafficClassifier) updateRules() {
	tc.mutex.Lock()
	defer tc.mutex.Unlock()
	
	tc.rules = make([]*QoSRule, 0)
	
	for _, policy := range tc.engine.Policies {
		if !policy.Enabled {
			continue
		}
		
		for _, rule := range policy.Rules {
			tc.rules = append(tc.rules, rule)
		}
	}
	
	// Sort rules by priority (higher first)
	for i := 0; i < len(tc.rules)-1; i++ {
		for j := i + 1; j < len(tc.rules); j++ {
			if tc.rules[i].Priority < tc.rules[j].Priority {
				tc.rules[i], tc.rules[j] = tc.rules[j], tc.rules[i]
			}
		}
	}
}

// NewTrafficShaper creates a new traffic shaper
func NewTrafficShaper(engine *QoSEngine) *TrafficShaper {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &TrafficShaper{
		engine:     engine,
		queues:     make(map[string]*TrafficQueue),
		schedulers: make(map[string]*PacketScheduler),
		shapers:    make(map[string]*BandwidthShaper),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start starts the traffic shaper
func (ts *TrafficShaper) Start() error {
	// Start background processing
	ts.wg.Add(1)
	go ts.processQueues()
	
	return nil
}

// Stop stops the traffic shaper
func (ts *TrafficShaper) Stop() error {
	ts.cancel()
	ts.wg.Wait()
	return nil
}

// ApplyPolicy applies a QoS policy to the traffic shaper
func (ts *TrafficShaper) ApplyPolicy(policy *QoSPolicy) error {
	ts.mutex.Lock()
	defer ts.mutex.Unlock()
	
	// Create queues for each class
	for classID := range policy.Classes {
		queue := &TrafficQueue{
			ID:      classID,
			ClassID: classID,
			Algorithm: policy.Algorithm,
			Packets: make(chan *QoSPacket, 1000),
			Stats:   &QueueStats{},
		}
		
		ts.queues[classID] = queue
	}
	
	// Create bandwidth shapers
	if policy.GlobalLimits != nil {
		shaper := &BandwidthShaper{
			ID:     policy.ID,
			Config: policy.GlobalLimits,
			TokenBucket: &TokenBucket{
				Capacity:   policy.GlobalLimits.BurstRate,
				Tokens:     policy.GlobalLimits.BurstRate,
				RefillRate: policy.GlobalLimits.MaxRate,
				LastRefill: time.Now(),
			},
			Stats: &ShaperStats{},
		}
		
		ts.shapers[policy.ID] = shaper
	}
	
	return nil
}

// UpdatePolicy updates a QoS policy in the traffic shaper
func (ts *TrafficShaper) UpdatePolicy(policy *QoSPolicy) error {
	// Remove existing policy
	ts.RemovePolicy(policy.ID)
	
	// Apply updated policy
	return ts.ApplyPolicy(policy)
}

// RemovePolicy removes a QoS policy from the traffic shaper
func (ts *TrafficShaper) RemovePolicy(policyID string) error {
	ts.mutex.Lock()
	defer ts.mutex.Unlock()
	
	// Remove queues and shapers associated with this policy
	for queueID := range ts.queues {
		// In a real implementation, would check if queue belongs to policy
		delete(ts.queues, queueID)
	}
	
	delete(ts.shapers, policyID)
	
	return nil
}

// ShapePacket shapes a packet according to QoS policies
func (ts *TrafficShaper) ShapePacket(packet *QoSPacket) (*QoSPacket, error) {
	ts.mutex.RLock()
	defer ts.mutex.RUnlock()
	
	// Find appropriate queue
	queue, exists := ts.queues[packet.ClassID]
	if !exists {
		queue = ts.queues["default"] // fallback to default queue
		if queue == nil {
			return packet, nil // no shaping
		}
	}
	
	// Check bandwidth limits
	// Simplified - in real implementation would use proper token bucket
	
	// Enqueue packet
	packet.EnqueueTime = time.Now()
	
	select {
	case queue.Packets <- packet:
		queue.Stats.EnqueuedPackets++
		queue.Stats.CurrentDepth++
		queue.Stats.TotalBytes += uint64(packet.Size)
		queue.Stats.LastActivity = time.Now()
	default:
		// Queue full - drop packet
		queue.Stats.DroppedPackets++
		return nil, fmt.Errorf("queue full, packet dropped")
	}
	
	return packet, nil
}

// processQueues processes packets from queues
func (ts *TrafficShaper) processQueues() {
	defer ts.wg.Done()
	
	ticker := time.NewTicker(1 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ts.ctx.Done():
			return
		case <-ticker.C:
			ts.schedulePackets()
		}
	}
}

// schedulePackets schedules packets from queues for transmission
func (ts *TrafficShaper) schedulePackets() {
	ts.mutex.RLock()
	defer ts.mutex.RUnlock()
	
	// Simple round-robin scheduling
	for _, queue := range ts.queues {
		select {
		case packet := <-queue.Packets:
			// Process packet (simulate transmission)
			queue.Stats.DequeuedPackets++
			queue.Stats.CurrentDepth--
			
			// Calculate delay
			delay := time.Since(packet.EnqueueTime)
			queue.Stats.AverageDelay = (queue.Stats.AverageDelay + delay) / 2
			
		default:
			// Queue empty
		}
	}
}

// NewQoSMonitor creates a new QoS monitor
func NewQoSMonitor(engine *QoSEngine) *QoSMonitor {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &QoSMonitor{
		engine:     engine,
		collectors: make(map[string]*MetricsCollector),
		exporters:  make([]MetricsExporter, 0),
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start starts the QoS monitor
func (qm *QoSMonitor) Start() error {
	// Start collectors
	qm.wg.Add(1)
	go qm.collectMetrics()
	
	return nil
}

// Stop stops the QoS monitor
func (qm *QoSMonitor) Stop() error {
	qm.cancel()
	qm.wg.Wait()
	return nil
}

// collectMetrics collects QoS metrics
func (qm *QoSMonitor) collectMetrics() {
	defer qm.wg.Done()
	
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-qm.ctx.Done():
			return
		case <-ticker.C:
			// Collect metrics from engine components
			// This would gather real metrics in production
		}
	}
}

// DefaultQoSEngineConfig returns default QoS engine configuration
func DefaultQoSEngineConfig() *QoSEngineConfig {
	return &QoSEngineConfig{
		DefaultAlgorithm:              AlgorithmHTB,
		MaxPolicies:                   100,
		MaxClassesPerPolicy:           50,
		MaxRulesPerPolicy:             200,
		StatisticsInterval:            30 * time.Second,
		EnableRealTimeStats:           true,
		EnablePerformanceOptimization: true,
		CacheSize:                     10000,
		WorkerCount:                   4,
		BufferSize:                    10000,
	}
}