package network

import (
	"context"
	"fmt"
	"net"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// QoSTestSuite provides comprehensive testing for Quality of Service and traffic shaping
type QoSTestSuite struct {
	suite.Suite
	qosManager       *QoSManager
	trafficShaper    *TrafficShaper
	bandwidthMonitor *BandwidthMonitor
	testInterfaces   []string
	testContext      context.Context
	testCancel       context.CancelFunc
	qosSimulator     *QoSSimulator
}

// QoSManager manages Quality of Service policies and enforcement
type QoSManager struct {
	policies        map[string]*QoSPolicy
	classifiers     map[string]*TrafficClassifier
	queues          map[string]*QueueManager
	shapers         map[string]*TrafficShaper
	meters          map[string]*TrafficMeter
	policers        map[string]*TrafficPolicer
	mutex           sync.RWMutex
	eventListeners  []QoSEventListener
	metricsProvider MetricsProvider
}

// QoSPolicy defines quality of service requirements
type QoSPolicy struct {
	ID              string
	Name            string
	Description     string
	Priority        int
	ClassifierRules []ClassificationRule
	Actions         []QoSAction
	SLA             *ServiceLevelAgreement
	Statistics      *PolicyStatistics
	Created         time.Time
	LastModified    time.Time
	Active          bool
}

// TrafficClassifier classifies traffic into different service classes
type TrafficClassifier struct {
	ID              string
	Name            string
	Rules           []ClassificationRule
	DefaultClass    string
	MatchStats      map[string]*MatchStatistics
	LastUpdated     time.Time
}

// ClassificationRule defines traffic classification criteria
type ClassificationRule struct {
	ID          string
	Priority    int
	Match       TrafficMatch
	Action      ClassificationAction
	Statistics  *RuleStatistics
}

// TrafficMatch defines packet matching criteria
type TrafficMatch struct {
	SrcIP       *net.IPNet
	DstIP       *net.IPNet
	SrcPort     PortRange
	DstPort     PortRange
	Protocol    uint8
	DSCP        uint8
	ToS         uint8
	VLANTag     uint16
	PacketSize  SizeRange
	Application string
}

// QoSAction defines actions to take on classified traffic
type QoSAction struct {
	Type       QoSActionType
	Parameters map[string]interface{}
}

// QoSActionType represents different QoS actions
type QoSActionType string

const (
	ActionSetDSCP       QoSActionType = "set_dscp"
	ActionSetClass      QoSActionType = "set_class"
	ActionSetPriority   QoSActionType = "set_priority"
	ActionSetRate       QoSActionType = "set_rate"
	ActionSetBurst      QoSActionType = "set_burst"
	ActionDrop          QoSActionType = "drop"
	ActionPolice        QoSActionType = "police"
	ActionShape         QoSActionType = "shape"
	ActionMark          QoSActionType = "mark"
)

// ServiceLevelAgreement defines SLA requirements
type ServiceLevelAgreement struct {
	MinBandwidth    int64         // bits per second
	MaxLatency      time.Duration
	MaxJitter       time.Duration
	MaxPacketLoss   float64       // percentage
	Availability    float64       // percentage
	BurstTolerance  int64         // bytes
	PriorityClass   string
	SchedulerWeight int
}

// TrafficShaper implements traffic shaping and rate limiting
type TrafficShaper struct {
	ID               string
	Interface        string
	RootQueue        *QueueHierarchy
	Schedulers       map[string]*QueueScheduler
	RateLimiters     map[string]*RateLimiter
	TokenBuckets     map[string]*TokenBucket
	Statistics       *ShaperStatistics
	LastUpdate       time.Time
}

// QueueManager manages traffic queues and scheduling
type QueueManager struct {
	Queues     map[string]*TrafficQueue
	Scheduler  QueueScheduler
	Statistics *QueueStatistics
}

// TrafficQueue represents a traffic queue with specific characteristics
type TrafficQueue struct {
	ID           string
	Class        string
	Priority     int
	Weight       int
	MinRate      int64
	MaxRate      int64
	BurstSize    int64
	QueueSize    int
	DropPolicy   DropPolicy
	Packets      []QueuedPacket
	Statistics   *QueueStatistics
	LastUpdate   time.Time
}

// QueueHierarchy represents hierarchical queue structure
type QueueHierarchy struct {
	RootClass    *QueueClass
	Classes      map[string]*QueueClass
	LeafQueues   map[string]*TrafficQueue
	MaxDepth     int
}

// QueueClass represents a class in queue hierarchy
type QueueClass struct {
	ID           string
	ParentID     string
	Children     []string
	Rate         int64
	Ceiling      int64
	Burst        int64
	Priority     int
	Scheduler    SchedulerType
	Statistics   *ClassStatistics
}

// QueueScheduler defines queue scheduling algorithms
type QueueScheduler struct {
	Type       SchedulerType
	Parameters map[string]interface{}
	Queues     []string
	Weights    map[string]int
	Priorities map[string]int
}

// TrafficMeter measures traffic rates and volumes
type TrafficMeter struct {
	ID         string
	Interface  string
	RateLimit  int64
	BurstSize  int64
	Conforming *MeterBucket
	Exceeding  *MeterBucket
	Violating  *MeterBucket
	Statistics *MeterStatistics
}

// TrafficPolicer enforces traffic policies
type TrafficPolicer struct {
	ID              string
	Interface       string
	Rules           []PolicingRule
	Actions         map[string]PolicingAction
	Statistics      *PolicerStatistics
	ViolationCount  int64
	LastViolation   time.Time
}

// BandwidthMonitor monitors network bandwidth usage
type BandwidthMonitor struct {
	interfaces     map[string]*InterfaceMonitor
	measurements   map[string][]*BandwidthMeasurement
	thresholds     map[string]*BandwidthThreshold
	alertHandlers  []BandwidthAlertHandler
	mutex          sync.RWMutex
	monitoringActive bool
}

// QoSSimulator simulates QoS behavior for testing
type QoSSimulator struct {
	networkConditions map[string]*NetworkCondition
	trafficGenerators map[string]*TrafficGenerator
	qualityMeasurements map[string]*QualityMeasurement
	mutex             sync.RWMutex
}

// Supporting types and enums

type SchedulerType string
const (
	SchedulerTypeHTB    SchedulerType = "htb"     // Hierarchical Token Bucket
	SchedulerTypeCBQ    SchedulerType = "cbq"     // Class-Based Queuing
	SchedulerTypeHFSC   SchedulerType = "hfsc"    // Hierarchical Fair Service Curve
	SchedulerTypeFIFO   SchedulerType = "fifo"    // First In, First Out
	SchedulerTypeSFQ    SchedulerType = "sfq"     // Stochastic Fair Queuing
	SchedulerTypeWFQ    SchedulerType = "wfq"     // Weighted Fair Queuing
	SchedulerTypePRIO   SchedulerType = "prio"    // Priority Queuing
)

type DropPolicy string
const (
	DropPolicyTailDrop DropPolicy = "tail_drop"
	DropPolicyRED      DropPolicy = "red"        // Random Early Detection
	DropPolicyWRED     DropPolicy = "wred"       // Weighted Random Early Detection
	DropPolicyCODel    DropPolicy = "codel"      // Controlled Delay
	DropPolicyFQCODel  DropPolicy = "fq_codel"   // Fair Queuing with CODel
)

type PortRange struct {
	Min uint16
	Max uint16
}

type SizeRange struct {
	Min int
	Max int
}

type ClassificationAction struct {
	SetClass    string
	SetPriority int
	SetDSCP     uint8
	SetRate     int64
}

// Statistics structures
type PolicyStatistics struct {
	PacketsMatched   uint64
	BytesMatched     uint64
	PacketsDropped   uint64
	BytesDropped     uint64
	AverageLatency   time.Duration
	PacketLoss       float64
	BandwidthUsed    int64
	ViolationCount   uint64
	LastViolation    time.Time
	LastUpdate       time.Time
}

type MatchStatistics struct {
	RuleID         string
	Matches        uint64
	Bytes          uint64
	FirstMatch     time.Time
	LastMatch      time.Time
}

type RuleStatistics struct {
	Matches        uint64
	Bytes          uint64
	Drops          uint64
	LastMatch      time.Time
}

type QueueStatistics struct {
	PacketsEnqueued  uint64
	PacketsDequeued  uint64
	PacketsDropped   uint64
	BytesEnqueued    uint64
	BytesDequeued    uint64
	BytesDropped     uint64
	CurrentDepth     int
	MaxDepth         int
	AverageDelay     time.Duration
	Utilization      float64
	LastUpdate       time.Time
}

type ClassStatistics struct {
	PacketsTransmitted uint64
	BytesTransmitted   uint64
	PacketsDropped     uint64
	BytesDropped       uint64
	BandwidthUsed      int64
	Utilization        float64
	QueueDelay         time.Duration
}

type ShaperStatistics struct {
	PacketsShaped    uint64
	BytesShaped      uint64
	PacketsDropped   uint64
	BytesDropped     uint64
	CurrentRate      int64
	PeakRate         int64
	BurstsSent       uint64
	TokensUsed       uint64
}

type MeterStatistics struct {
	ConformingPackets uint64
	ConformingBytes   uint64
	ExceedingPackets  uint64
	ExceedingBytes    uint64
	ViolatingPackets  uint64
	ViolatingBytes    uint64
	LastUpdate        time.Time
}

type PolicerStatistics struct {
	PacketsPoliced   uint64
	BytesPoliced     uint64
	ViolationsCount  uint64
	ActionsApplied   map[string]uint64
}

type BandwidthMeasurement struct {
	Timestamp    time.Time
	Interface    string
	RxBytes      uint64
	TxBytes      uint64
	RxPackets    uint64
	TxPackets    uint64
	RxBandwidth  int64
	TxBandwidth  int64
	Utilization  float64
}

type InterfaceMonitor struct {
	Interface     string
	LastMeasurement *BandwidthMeasurement
	History       []*BandwidthMeasurement
	MaxHistory    int
	AlertThreshold *BandwidthThreshold
}

type BandwidthThreshold struct {
	Interface       string
	MaxUtilization  float64
	MaxBandwidth    int64
	AlertOnExceed   bool
	AlertCooldown   time.Duration
	LastAlert       time.Time
}

// Additional structures
type QueuedPacket struct {
	ID        string
	Size      int
	Priority  int
	Class     string
	EnqueueTime time.Time
	Deadline    time.Time
}

type RateLimiter struct {
	ID           string
	Rate         int64
	BurstSize    int64
	TokenBucket  *TokenBucket
	Statistics   *RateLimiterStats
}

type TokenBucket struct {
	Capacity    int64
	Tokens      int64
	RefillRate  int64
	LastRefill  time.Time
	mutex       sync.Mutex
}

type RateLimiterStats struct {
	PacketsAllowed uint64
	PacketsDropped uint64
	BytesAllowed   uint64
	BytesDropped   uint64
}

type MeterBucket struct {
	Rate      int64
	BurstSize int64
	Tokens    int64
}

type PolicingRule struct {
	Match  TrafficMatch
	Action PolicingAction
	Rate   int64
	Burst  int64
}

type PolicingAction struct {
	Type   PolicingActionType
	Params map[string]interface{}
}

type PolicingActionType string
const (
	PolicingActionDrop   PolicingActionType = "drop"
	PolicingActionAllow  PolicingActionType = "allow"
	PolicingActionMark   PolicingActionType = "mark"
	PolicingActionRemark PolicingActionType = "remark"
)

type NetworkCondition struct {
	Bandwidth    int64
	Latency      time.Duration
	Jitter       time.Duration
	PacketLoss   float64
	Congestion   float64
}

type QualityMeasurement struct {
	Throughput   int64
	Latency      time.Duration
	Jitter       time.Duration
	PacketLoss   float64
	MOS          float64 // Mean Opinion Score
	Timestamp    time.Time
}

// Event handling
type QoSEventListener func(event QoSEvent)

type QoSEvent struct {
	Type      QoSEventType
	Timestamp time.Time
	Source    string
	Data      map[string]interface{}
}

type QoSEventType string
const (
	EventPolicyViolation    QoSEventType = "policy_violation"
	EventThresholdExceeded  QoSEventType = "threshold_exceeded"
	EventQueueOverflow      QoSEventType = "queue_overflow"
	EventSLABreach          QoSEventType = "sla_breach"
	EventRateLimitHit       QoSEventType = "rate_limit_hit"
)

type BandwidthAlertHandler func(alert BandwidthAlert)

type BandwidthAlert struct {
	Interface   string
	Threshold   *BandwidthThreshold
	Current     *BandwidthMeasurement
	Severity    AlertSeverity
	Timestamp   time.Time
	Message     string
}

type AlertSeverity string
const (
	AlertSeverityLow      AlertSeverity = "low"
	AlertSeverityMedium   AlertSeverity = "medium"
	AlertSeverityHigh     AlertSeverity = "high"
	AlertSeverityCritical AlertSeverity = "critical"
)

type MetricsProvider interface {
	RecordCounter(name string, value int64, tags map[string]string)
	RecordHistogram(name string, value float64, tags map[string]string)
	RecordGauge(name string, value float64, tags map[string]string)
}

// SetupSuite initializes the test suite
func (suite *QoSTestSuite) SetupSuite() {
	suite.testContext, suite.testCancel = context.WithCancel(context.Background())
	
	// Initialize QoS manager
	suite.qosManager = NewQoSManager()
	
	// Initialize traffic shaper
	suite.trafficShaper = NewTrafficShaper("test-shaper", "eth0")
	
	// Initialize bandwidth monitor
	suite.bandwidthMonitor = NewBandwidthMonitor()
	
	// Initialize QoS simulator
	suite.qosSimulator = NewQoSSimulator()
	
	// Create test interfaces
	suite.createTestInterfaces()
	
	// Setup test policies
	suite.setupTestPolicies()
}

// TearDownSuite cleans up after all tests
func (suite *QoSTestSuite) TearDownSuite() {
	// Cleanup test interfaces
	suite.cleanupTestInterfaces()
	
	// Stop monitoring
	if suite.bandwidthMonitor != nil {
		suite.bandwidthMonitor.Stop()
	}
	
	// Cleanup simulator
	if suite.qosSimulator != nil {
		suite.qosSimulator.Cleanup()
	}
	
	if suite.testCancel != nil {
		suite.testCancel()
	}
}

// NewQoSManager creates a new QoS manager
func NewQoSManager() *QoSManager {
	return &QoSManager{
		policies:       make(map[string]*QoSPolicy),
		classifiers:    make(map[string]*TrafficClassifier),
		queues:         make(map[string]*QueueManager),
		shapers:        make(map[string]*TrafficShaper),
		meters:         make(map[string]*TrafficMeter),
		policers:       make(map[string]*TrafficPolicer),
		eventListeners: []QoSEventListener{},
	}
}

// NewTrafficShaper creates a new traffic shaper
func NewTrafficShaper(id, iface string) *TrafficShaper {
	return &TrafficShaper{
		ID:           id,
		Interface:    iface,
		RootQueue:    &QueueHierarchy{Classes: make(map[string]*QueueClass), LeafQueues: make(map[string]*TrafficQueue)},
		Schedulers:   make(map[string]*QueueScheduler),
		RateLimiters: make(map[string]*RateLimiter),
		TokenBuckets: make(map[string]*TokenBucket),
		Statistics:   &ShaperStatistics{},
	}
}

// NewBandwidthMonitor creates a new bandwidth monitor
func NewBandwidthMonitor() *BandwidthMonitor {
	return &BandwidthMonitor{
		interfaces:    make(map[string]*InterfaceMonitor),
		measurements:  make(map[string][]*BandwidthMeasurement),
		thresholds:    make(map[string]*BandwidthThreshold),
		alertHandlers: []BandwidthAlertHandler{},
	}
}

// NewQoSSimulator creates a new QoS simulator
func NewQoSSimulator() *QoSSimulator {
	return &QoSSimulator{
		networkConditions:   make(map[string]*NetworkCondition),
		trafficGenerators:   make(map[string]*TrafficGenerator),
		qualityMeasurements: make(map[string]*QualityMeasurement),
	}
}

// createTestInterfaces creates virtual test interfaces
func (suite *QoSTestSuite) createTestInterfaces() {
	// Create test interfaces for QoS testing
	interfaces := []string{"test-eth0", "test-eth1", "test-wlan0"}
	
	for _, iface := range interfaces {
		suite.testInterfaces = append(suite.testInterfaces, iface)
		
		// Setup interface in bandwidth monitor
		suite.bandwidthMonitor.AddInterface(iface, &BandwidthThreshold{
			Interface:      iface,
			MaxUtilization: 0.8,
			MaxBandwidth:   1000000000, // 1 Gbps
			AlertOnExceed:  true,
			AlertCooldown:  30 * time.Second,
		})
	}
}

// setupTestPolicies creates standard test QoS policies
func (suite *QoSTestSuite) setupTestPolicies() {
	// High priority policy for real-time traffic
	highPriorityPolicy := &QoSPolicy{
		ID:          "high-priority-" + uuid.New().String(),
		Name:        "High Priority Traffic",
		Description: "Policy for real-time voice and video traffic",
		Priority:    100,
		ClassifierRules: []ClassificationRule{
			{
				ID:       "voice-traffic",
				Priority: 100,
				Match: TrafficMatch{
					DstPort:  PortRange{Min: 5060, Max: 5060}, // SIP
					Protocol: 17,                              // UDP
					DSCP:     46,                              // EF (Expedited Forwarding)
				},
				Action: ClassificationAction{
					SetClass:    "voice",
					SetPriority: 7,
					SetDSCP:     46,
				},
			},
		},
		Actions: []QoSAction{
			{
				Type: ActionSetPriority,
				Parameters: map[string]interface{}{
					"priority": 7,
				},
			},
			{
				Type: ActionSetRate,
				Parameters: map[string]interface{}{
					"min_rate": int64(1000000),  // 1 Mbps guaranteed
					"max_rate": int64(10000000), // 10 Mbps ceiling
				},
			},
		},
		SLA: &ServiceLevelAgreement{
			MinBandwidth:    1000000,           // 1 Mbps
			MaxLatency:      10 * time.Millisecond,
			MaxJitter:       2 * time.Millisecond,
			MaxPacketLoss:   0.1, // 0.1%
			Availability:    99.9,
			BurstTolerance:  10000,
			PriorityClass:   "voice",
			SchedulerWeight: 10,
		},
		Statistics: &PolicyStatistics{},
		Created:    time.Now(),
		Active:     true,
	}
	
	suite.qosManager.AddPolicy(highPriorityPolicy)
	
	// Best effort policy for bulk data
	bulkDataPolicy := &QoSPolicy{
		ID:          "bulk-data-" + uuid.New().String(),
		Name:        "Bulk Data Traffic",
		Description: "Policy for best-effort bulk data transfers",
		Priority:    10,
		ClassifierRules: []ClassificationRule{
			{
				ID:       "bulk-data",
				Priority: 10,
				Match: TrafficMatch{
					DstPort:  PortRange{Min: 80, Max: 80}, // HTTP
					Protocol: 6,                           // TCP
				},
				Action: ClassificationAction{
					SetClass:    "bulk",
					SetPriority: 1,
				},
			},
		},
		Actions: []QoSAction{
			{
				Type: ActionSetClass,
				Parameters: map[string]interface{}{
					"class": "bulk",
				},
			},
		},
		SLA: &ServiceLevelAgreement{
			MinBandwidth:    100000,            // 100 Kbps minimum
			MaxLatency:      1 * time.Second,
			MaxJitter:       100 * time.Millisecond,
			MaxPacketLoss:   1.0, // 1%
			Availability:    95.0,
			PriorityClass:   "bulk",
			SchedulerWeight: 1,
		},
		Statistics: &PolicyStatistics{},
		Created:    time.Now(),
		Active:     true,
	}
	
	suite.qosManager.AddPolicy(bulkDataPolicy)
}

// cleanupTestInterfaces removes test interfaces
func (suite *QoSTestSuite) cleanupTestInterfaces() {
	for _, iface := range suite.testInterfaces {
		suite.bandwidthMonitor.RemoveInterface(iface)
	}
	suite.testInterfaces = nil
}

// TestQoSPolicyEnforcement tests QoS policy creation and enforcement
func (suite *QoSTestSuite) TestQoSPolicyEnforcement() {
	testCases := []struct {
		name       string
		policy     *QoSPolicy
		testTraffic *TrafficFlow
		validate   func(t *testing.T, result *QoSResult)
	}{
		{
			name:   "HighPriorityVoiceTraffic",
			policy: suite.getTestPolicy("High Priority Traffic"),
			testTraffic: &TrafficFlow{
				SrcIP:       net.IPv4(192, 168, 1, 10),
				DstIP:       net.IPv4(192, 168, 1, 20),
				SrcPort:     12345,
				DstPort:     5060,
				Protocol:    17,
				DSCP:        46,
				PacketSize:  64,
				Rate:        1000, // packets per second
				Duration:    10 * time.Second,
			},
			validate: func(t *testing.T, result *QoSResult) {
				assert.Equal(t, "voice", result.AssignedClass)
				assert.Equal(t, 7, result.Priority)
				assert.Equal(t, uint8(46), result.DSCP)
				assert.LessOrEqual(t, result.AverageLatency, 10*time.Millisecond)
				assert.LessOrEqual(t, result.PacketLoss, 0.001) // 0.1%
			},
		},
		{
			name:   "BulkDataTraffic",
			policy: suite.getTestPolicy("Bulk Data Traffic"),
			testTraffic: &TrafficFlow{
				SrcIP:       net.IPv4(192, 168, 1, 30),
				DstIP:       net.IPv4(192, 168, 1, 40),
				SrcPort:     54321,
				DstPort:     80,
				Protocol:    6,
				PacketSize:  1500,
				Rate:        100, // packets per second
				Duration:    30 * time.Second,
			},
			validate: func(t *testing.T, result *QoSResult) {
				assert.Equal(t, "bulk", result.AssignedClass)
				assert.Equal(t, 1, result.Priority)
				assert.GreaterOrEqual(t, result.ThroughputMbps, 0.1) // At least 100 Kbps
			},
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.name, func(t *testing.T) {
			// Simulate traffic flow
			result, err := suite.simulateTrafficWithQoS(tc.testTraffic, tc.policy)
			require.NoError(t, err, "Traffic simulation should succeed")
			
			if tc.validate != nil {
				tc.validate(t, result)
			}
			
			// Verify policy statistics were updated
			stats := tc.policy.Statistics
			assert.Greater(t, stats.PacketsMatched, uint64(0), "Policy should match packets")
			assert.Greater(t, stats.BytesMatched, uint64(0), "Policy should match bytes")
		})
	}
}

// TestTrafficShaping tests traffic shaping functionality
func (suite *QoSTestSuite) TestTrafficShaping() {
	shapingTests := []struct {
		name        string
		config      *ShapingConfig
		trafficLoad *TrafficLoad
		validate    func(t *testing.T, result *ShapingResult)
	}{
		{
			name: "BasicRateLimit",
			config: &ShapingConfig{
				Interface:   "test-eth0",
				MaxRate:     10000000, // 10 Mbps
				BurstSize:   100000,   // 100KB
				Algorithm:   "token_bucket",
			},
			trafficLoad: &TrafficLoad{
				Rate:       20000000, // 20 Mbps (above limit)
				PacketSize: 1500,
				Duration:   10 * time.Second,
			},
			validate: func(t *testing.T, result *ShapingResult) {
				// Rate should be limited to ~10 Mbps
				assert.LessOrEqual(t, result.ActualRate, int64(11000000), "Rate should be limited")
				assert.Greater(t, result.PacketsDropped, uint64(0), "Some packets should be dropped")
				assert.Greater(t, result.ShapingEfficiency, 0.8, "Shaping should be efficient")
			},
		},
		{
			name: "HierarchicalShaping",
			config: &ShapingConfig{
				Interface: "test-eth1",
				Hierarchy: &QueueHierarchy{
					Classes: map[string]*QueueClass{
						"root": {
							ID:       "root",
							Rate:     50000000, // 50 Mbps total
							Ceiling:  50000000,
							Priority: 1,
						},
						"priority": {
							ID:       "priority",
							ParentID: "root",
							Rate:     20000000, // 20 Mbps guaranteed
							Ceiling:  40000000, // 40 Mbps max
							Priority: 7,
						},
						"best_effort": {
							ID:       "best_effort",
							ParentID: "root",
							Rate:     10000000, // 10 Mbps guaranteed
							Ceiling:  50000000, // Can use remaining
							Priority: 1,
						},
					},
				},
			},
			trafficLoad: &TrafficLoad{
				Flows: []TrafficFlow{
					{
						Class:      "priority",
						Rate:       15000000,
						PacketSize: 1000,
						Duration:   15 * time.Second,
					},
					{
						Class:      "best_effort",
						Rate:       30000000,
						PacketSize: 1500,
						Duration:   15 * time.Second,
					},
				},
			},
			validate: func(t *testing.T, result *ShapingResult) {
				// Priority class should get its guaranteed rate
				priorityStats := result.ClassStats["priority"]
				assert.GreaterOrEqual(t, priorityStats.AchievedRate, int64(14000000))
				
				// Best effort should get remaining bandwidth
				bestEffortStats := result.ClassStats["best_effort"]
				assert.LessOrEqual(t, bestEffortStats.AchievedRate, int64(35000000))
				
				// Total should not exceed link capacity
				totalRate := priorityStats.AchievedRate + bestEffortStats.AchievedRate
				assert.LessOrEqual(t, totalRate, int64(52000000))
			},
		},
	}
	
	for _, test := range shapingTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Setup traffic shaper with configuration
			shaper := suite.setupTrafficShaper(test.config)
			
			// Run traffic shaping test
			result, err := suite.runShapingTest(shaper, test.trafficLoad)
			require.NoError(t, err, "Shaping test should complete successfully")
			
			if test.validate != nil {
				test.validate(t, result)
			}
			
			// Cleanup
			suite.cleanupTrafficShaper(shaper)
		})
	}
}

// TestQueueManagement tests queue management algorithms
func (suite *QoSTestSuite) TestQueueManagement() {
	queueTests := []struct {
		name      string
		scheduler SchedulerType
		queues    []*TrafficQueue
		traffic   *TrafficPattern
		validate  func(t *testing.T, result *QueueResult)
	}{
		{
			name:      "PriorityQueuing",
			scheduler: SchedulerTypePRIO,
			queues: []*TrafficQueue{
				{
					ID:        "high",
					Priority:  7,
					MaxRate:   10000000,
					QueueSize: 100,
				},
				{
					ID:        "medium",
					Priority:  4,
					MaxRate:   5000000,
					QueueSize: 50,
				},
				{
					ID:        "low",
					Priority:  1,
					MaxRate:   1000000,
					QueueSize: 10,
				},
			},
			traffic: &TrafficPattern{
				Flows: []QueueFlow{
					{Queue: "high", Rate: 5000000, Duration: 10 * time.Second},
					{Queue: "medium", Rate: 8000000, Duration: 10 * time.Second},
					{Queue: "low", Rate: 2000000, Duration: 10 * time.Second},
				},
			},
			validate: func(t *testing.T, result *QueueResult) {
				// High priority should get full bandwidth
				assert.GreaterOrEqual(t, result.QueueStats["high"].ThroughputMbps, 4.0)
				
				// Medium priority should get remaining bandwidth
				assert.GreaterOrEqual(t, result.QueueStats["medium"].ThroughputMbps, 2.0)
				
				// Low priority should get minimal bandwidth
				assert.LessOrEqual(t, result.QueueStats["low"].ThroughputMbps, 1.0)
			},
		},
		{
			name:      "WeightedFairQueuing",
			scheduler: SchedulerTypeWFQ,
			queues: []*TrafficQueue{
				{ID: "queue1", Weight: 3, QueueSize: 50},
				{ID: "queue2", Weight: 2, QueueSize: 50},
				{ID: "queue3", Weight: 1, QueueSize: 50},
			},
			traffic: &TrafficPattern{
				Flows: []QueueFlow{
					{Queue: "queue1", Rate: 10000000, Duration: 15 * time.Second},
					{Queue: "queue2", Rate: 10000000, Duration: 15 * time.Second},
					{Queue: "queue3", Rate: 10000000, Duration: 15 * time.Second},
				},
			},
			validate: func(t *testing.T, result *QueueResult) {
				// Bandwidth should be distributed according to weights (3:2:1)
				stats1 := result.QueueStats["queue1"]
				stats2 := result.QueueStats["queue2"]
				stats3 := result.QueueStats["queue3"]
				
				ratio1to2 := stats1.ThroughputMbps / stats2.ThroughputMbps
				ratio2to3 := stats2.ThroughputMbps / stats3.ThroughputMbps
				
				assert.InDelta(t, 1.5, ratio1to2, 0.2, "Queue1:Queue2 ratio should be ~3:2")
				assert.InDelta(t, 2.0, ratio2to3, 0.3, "Queue2:Queue3 ratio should be ~2:1")
			},
		},
	}
	
	for _, test := range queueTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Setup queue manager
			queueMgr := suite.setupQueueManager(test.scheduler, test.queues)
			
			// Run queue management test
			result, err := suite.runQueueTest(queueMgr, test.traffic)
			require.NoError(t, err, "Queue test should complete successfully")
			
			if test.validate != nil {
				test.validate(t, result)
			}
		})
	}
}

// TestBandwidthMonitoring tests bandwidth monitoring and alerting
func (suite *QoSTestSuite) TestBandwidthMonitoring() {
	// Setup monitoring for test interface
	iface := "test-eth0"
	threshold := &BandwidthThreshold{
		Interface:      iface,
		MaxUtilization: 0.8, // 80%
		MaxBandwidth:   100000000, // 100 Mbps
		AlertOnExceed:  true,
		AlertCooldown:  1 * time.Second,
	}
	
	suite.bandwidthMonitor.AddInterface(iface, threshold)
	
	// Setup alert handler
	alertReceived := make(chan BandwidthAlert, 1)
	suite.bandwidthMonitor.AddAlertHandler(func(alert BandwidthAlert) {
		alertReceived <- alert
	})
	
	// Start monitoring
	err := suite.bandwidthMonitor.Start()
	require.NoError(suite.T(), err)
	
	// Simulate high bandwidth usage
	usage := &BandwidthMeasurement{
		Interface:   iface,
		RxBandwidth: 90000000, // 90 Mbps (exceeds 80% threshold)
		TxBandwidth: 10000000, // 10 Mbps
		Utilization: 0.9,      // 90%
		Timestamp:   time.Now(),
	}
	
	suite.bandwidthMonitor.UpdateMeasurement(usage)
	
	// Wait for alert
	select {
	case alert := <-alertReceived:
		assert.Equal(suite.T(), iface, alert.Interface)
		assert.Equal(suite.T(), AlertSeverityHigh, alert.Severity)
		assert.Contains(suite.T(), alert.Message, "threshold exceeded")
	case <-time.After(2 * time.Second):
		suite.T().Fatal("Expected alert was not received")
	}
	
	// Test threshold not exceeded
	normalUsage := &BandwidthMeasurement{
		Interface:   iface,
		RxBandwidth: 50000000, // 50 Mbps (below threshold)
		TxBandwidth: 10000000, // 10 Mbps
		Utilization: 0.6,      // 60%
		Timestamp:   time.Now(),
	}
	
	suite.bandwidthMonitor.UpdateMeasurement(normalUsage)
	
	// Should not receive alert
	select {
	case <-alertReceived:
		suite.T().Fatal("Unexpected alert received")
	case <-time.After(500 * time.Millisecond):
		// Expected - no alert should be received
	}
}

// TestCongestionControl tests congestion control mechanisms
func (suite *QoSTestSuite) TestCongestionControl() {
	congestionTests := []struct {
		name          string
		dropPolicy    DropPolicy
		queueSize     int
		trafficRate   int64
		expectedDrop  bool
		validate      func(t *testing.T, result *CongestionResult)
	}{
		{
			name:         "TailDropCongestion",
			dropPolicy:   DropPolicyTailDrop,
			queueSize:    10,
			trafficRate:  20000000, // 20 Mbps into 10 Mbps queue
			expectedDrop: true,
			validate: func(t *testing.T, result *CongestionResult) {
				assert.Greater(t, result.DroppedPackets, uint64(0))
				assert.Equal(t, "tail_drop", result.DropReason)
			},
		},
		{
			name:         "REDEarlyDrop",
			dropPolicy:   DropPolicyRED,
			queueSize:    100,
			trafficRate:  15000000, // Moderate load
			expectedDrop: false,    // Should not drop at moderate load
			validate: func(t *testing.T, result *CongestionResult) {
				assert.LessOrEqual(t, result.DroppedPackets, uint64(5)) // Minimal drops
				if result.DroppedPackets > 0 {
					assert.Equal(t, "red", result.DropReason)
				}
			},
		},
	}
	
	for _, test := range congestionTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Setup queue with congestion control
			queue := &TrafficQueue{
				ID:         "test-queue",
				QueueSize:  test.queueSize,
				DropPolicy: test.dropPolicy,
				Statistics: &QueueStatistics{},
			}
			
			// Simulate congestion
			result, err := suite.simulateCongestion(queue, test.trafficRate, 5*time.Second)
			require.NoError(t, err)
			
			if test.expectedDrop {
				assert.Greater(t, result.DroppedPackets, uint64(0), "Should drop packets under congestion")
			}
			
			if test.validate != nil {
				test.validate(t, result)
			}
		})
	}
}

// TestQoSMeasurements tests quality measurements (MOS, jitter, etc.)
func (suite *QoSTestSuite) TestQoSMeasurements() {
	measurementTests := []struct {
		name         string
		trafficType  string
		conditions   *NetworkCondition
		expectMOS    float64
		tolerance    float64
	}{
		{
			name:        "VoiceQualityGood",
			trafficType: "voice",
			conditions: &NetworkCondition{
				Latency:    20 * time.Millisecond,
				Jitter:     2 * time.Millisecond,
				PacketLoss: 0.1, // 0.1%
			},
			expectMOS: 4.0, // Good quality
			tolerance: 0.3,
		},
		{
			name:        "VoiceQualityPoor",
			trafficType: "voice",
			conditions: &NetworkCondition{
				Latency:    200 * time.Millisecond,
				Jitter:     50 * time.Millisecond,
				PacketLoss: 3.0, // 3%
			},
			expectMOS: 2.0, // Poor quality
			tolerance: 0.5,
		},
		{
			name:        "VideoQualityFair",
			trafficType: "video",
			conditions: &NetworkCondition{
				Latency:    100 * time.Millisecond,
				Jitter:     20 * time.Millisecond,
				PacketLoss: 1.0, // 1%
			},
			expectMOS: 3.0, // Fair quality
			tolerance: 0.4,
		},
	}
	
	for _, test := range measurementTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Simulate quality measurements
			measurement, err := suite.measureQuality(test.trafficType, test.conditions)
			require.NoError(t, err)
			
			// Validate MOS score
			assert.InDelta(t, test.expectMOS, measurement.MOS, test.tolerance,
				"MOS score should match expected quality")
			
			// Validate measurements match conditions
			assert.InDelta(t, test.conditions.Latency.Seconds()*1000, 
				measurement.Latency.Seconds()*1000, 10, "Latency should match")
			assert.InDelta(t, test.conditions.PacketLoss, 
				measurement.PacketLoss*100, 0.5, "Packet loss should match")
		})
	}
}

// Helper methods and supporting structures

type TrafficFlow struct {
	SrcIP      net.IP
	DstIP      net.IP
	SrcPort    uint16
	DstPort    uint16
	Protocol   uint8
	DSCP       uint8
	PacketSize int
	Rate       int
	Duration   time.Duration
	Class      string
}

type QoSResult struct {
	AssignedClass    string
	Priority         int
	DSCP             uint8
	AverageLatency   time.Duration
	PacketLoss       float64
	ThroughputMbps   float64
}

type ShapingConfig struct {
	Interface   string
	MaxRate     int64
	BurstSize   int64
	Algorithm   string
	Hierarchy   *QueueHierarchy
}

type TrafficLoad struct {
	Rate       int64
	PacketSize int
	Duration   time.Duration
	Flows      []TrafficFlow
}

type ShapingResult struct {
	ActualRate        int64
	PacketsDropped    uint64
	ShapingEfficiency float64
	ClassStats        map[string]*ClassPerformance
}

type ClassPerformance struct {
	AchievedRate int64
	Utilization  float64
	Drops        uint64
}

type TrafficPattern struct {
	Flows []QueueFlow
}

type QueueFlow struct {
	Queue    string
	Rate     int64
	Duration time.Duration
}

type QueueResult struct {
	QueueStats map[string]*QueuePerformance
}

type QueuePerformance struct {
	ThroughputMbps float64
	AverageDelay   time.Duration
	DropRate       float64
	Utilization    float64
}

type CongestionResult struct {
	DroppedPackets uint64
	DropReason     string
	QueueDepth     int
	Utilization    float64
}

// Implementation methods (stubs for compilation)
func (qm *QoSManager) AddPolicy(policy *QoSPolicy) error {
	qm.mutex.Lock()
	defer qm.mutex.Unlock()
	qm.policies[policy.ID] = policy
	return nil
}

func (suite *QoSTestSuite) getTestPolicy(name string) *QoSPolicy {
	for _, policy := range suite.qosManager.policies {
		if policy.Name == name {
			return policy
		}
	}
	return nil
}

func (suite *QoSTestSuite) simulateTrafficWithQoS(flow *TrafficFlow, policy *QoSPolicy) (*QoSResult, error) {
	// Simulation logic would be implemented here
	return &QoSResult{
		AssignedClass:  "voice",
		Priority:       7,
		DSCP:           46,
		AverageLatency: 5 * time.Millisecond,
		PacketLoss:     0.0001,
		ThroughputMbps: 1.0,
	}, nil
}

// Additional helper method implementations would go here...

func (bm *BandwidthMonitor) AddInterface(iface string, threshold *BandwidthThreshold) error {
	bm.mutex.Lock()
	defer bm.mutex.Unlock()
	
	monitor := &InterfaceMonitor{
		Interface:      iface,
		History:        make([]*BandwidthMeasurement, 0),
		MaxHistory:     100,
		AlertThreshold: threshold,
	}
	
	bm.interfaces[iface] = monitor
	bm.thresholds[iface] = threshold
	return nil
}

func (bm *BandwidthMonitor) RemoveInterface(iface string) error {
	bm.mutex.Lock()
	defer bm.mutex.Unlock()
	
	delete(bm.interfaces, iface)
	delete(bm.thresholds, iface)
	delete(bm.measurements, iface)
	return nil
}

func (bm *BandwidthMonitor) AddAlertHandler(handler BandwidthAlertHandler) {
	bm.mutex.Lock()
	defer bm.mutex.Unlock()
	bm.alertHandlers = append(bm.alertHandlers, handler)
}

func (bm *BandwidthMonitor) Start() error {
	bm.mutex.Lock()
	defer bm.mutex.Unlock()
	bm.monitoringActive = true
	return nil
}

func (bm *BandwidthMonitor) Stop() error {
	bm.mutex.Lock()
	defer bm.mutex.Unlock()
	bm.monitoringActive = false
	return nil
}

func (bm *BandwidthMonitor) UpdateMeasurement(measurement *BandwidthMeasurement) error {
	bm.mutex.Lock()
	defer bm.mutex.Unlock()
	
	// Check threshold
	threshold := bm.thresholds[measurement.Interface]
	if threshold != nil && threshold.AlertOnExceed {
		if measurement.Utilization > threshold.MaxUtilization {
			// Check cooldown
			if time.Since(threshold.LastAlert) > threshold.AlertCooldown {
				alert := BandwidthAlert{
					Interface: measurement.Interface,
					Threshold: threshold,
					Current:   measurement,
					Severity:  AlertSeverityHigh,
					Timestamp: time.Now(),
					Message:   fmt.Sprintf("Bandwidth utilization %.1f%% exceeds threshold %.1f%%", 
						measurement.Utilization*100, threshold.MaxUtilization*100),
				}
				
				// Send alert to handlers
				for _, handler := range bm.alertHandlers {
					go handler(alert)
				}
				
				threshold.LastAlert = time.Now()
			}
		}
	}
	
	return nil
}

// Cleanup method for simulator
func (sim *QoSSimulator) Cleanup() {
	sim.mutex.Lock()
	defer sim.mutex.Unlock()
	
	sim.networkConditions = make(map[string]*NetworkCondition)
	sim.trafficGenerators = make(map[string]*TrafficGenerator)
	sim.qualityMeasurements = make(map[string]*QualityMeasurement)
}

// Stub implementations for test helper methods
func (suite *QoSTestSuite) setupTrafficShaper(config *ShapingConfig) *TrafficShaper {
	return NewTrafficShaper("test-shaper", config.Interface)
}

func (suite *QoSTestSuite) runShapingTest(shaper *TrafficShaper, load *TrafficLoad) (*ShapingResult, error) {
	return &ShapingResult{
		ActualRate:        10000000,
		PacketsDropped:    100,
		ShapingEfficiency: 0.9,
		ClassStats:        make(map[string]*ClassPerformance),
	}, nil
}

func (suite *QoSTestSuite) cleanupTrafficShaper(shaper *TrafficShaper) {
	// Cleanup logic
}

func (suite *QoSTestSuite) setupQueueManager(scheduler SchedulerType, queues []*TrafficQueue) *QueueManager {
	queueMap := make(map[string]*TrafficQueue)
	for _, queue := range queues {
		queueMap[queue.ID] = queue
	}
	
	return &QueueManager{
		Queues: queueMap,
		Scheduler: QueueScheduler{
			Type: scheduler,
		},
		Statistics: &QueueStatistics{},
	}
}

func (suite *QoSTestSuite) runQueueTest(queueMgr *QueueManager, pattern *TrafficPattern) (*QueueResult, error) {
	stats := make(map[string]*QueuePerformance)
	
	for _, flow := range pattern.Flows {
		stats[flow.Queue] = &QueuePerformance{
			ThroughputMbps: float64(flow.Rate) / 1000000.0,
			AverageDelay:   10 * time.Millisecond,
			DropRate:       0.01,
			Utilization:    0.8,
		}
	}
	
	return &QueueResult{QueueStats: stats}, nil
}

func (suite *QoSTestSuite) simulateCongestion(queue *TrafficQueue, rate int64, duration time.Duration) (*CongestionResult, error) {
	return &CongestionResult{
		DroppedPackets: 10,
		DropReason:     string(queue.DropPolicy),
		QueueDepth:     queue.QueueSize,
		Utilization:    0.9,
	}, nil
}

func (suite *QoSTestSuite) measureQuality(trafficType string, conditions *NetworkCondition) (*QualityMeasurement, error) {
	// Calculate MOS based on conditions (simplified R-factor model)
	rFactor := 94.0
	
	// Latency impact
	if conditions.Latency > 150*time.Millisecond {
		rFactor -= float64(conditions.Latency/time.Millisecond-150) * 0.1
	}
	
	// Packet loss impact
	if conditions.PacketLoss > 0 {
		rFactor -= conditions.PacketLoss * 20
	}
	
	// Convert R-factor to MOS
	var mos float64
	if rFactor > 80 {
		mos = 4.5
	} else if rFactor > 70 {
		mos = 4.0
	} else if rFactor > 60 {
		mos = 3.5
	} else if rFactor > 50 {
		mos = 3.0
	} else {
		mos = 2.0
	}
	
	return &QualityMeasurement{
		Throughput: 1000000, // 1 Mbps
		Latency:    conditions.Latency,
		Jitter:     conditions.Jitter,
		PacketLoss: conditions.PacketLoss / 100.0, // Convert to decimal
		MOS:        mos,
		Timestamp:  time.Now(),
	}, nil
}

// TestQoSTestSuite runs the complete QoS test suite
func TestQoSTestSuite(t *testing.T) {
	suite.Run(t, new(QoSTestSuite))
}

// Standalone test functions
func TestQoSManagerCreation(t *testing.T) {
	manager := NewQoSManager()
	assert.NotNil(t, manager)
	assert.Empty(t, manager.policies)
	assert.Empty(t, manager.classifiers)
}

func TestTrafficClassification(t *testing.T) {
	classifier := &TrafficClassifier{
		ID:   "test-classifier",
		Name: "Test Classifier",
		Rules: []ClassificationRule{
			{
				ID:       "rule1",
				Priority: 100,
				Match: TrafficMatch{
					Protocol: 6,    // TCP
					DstPort:  PortRange{Min: 80, Max: 80},
				},
				Action: ClassificationAction{
					SetClass: "web",
				},
			},
		},
		MatchStats: make(map[string]*MatchStatistics),
	}
	
	assert.NotNil(t, classifier)
	assert.Equal(t, "test-classifier", classifier.ID)
	assert.Len(t, classifier.Rules, 1)
	assert.Equal(t, uint8(6), classifier.Rules[0].Match.Protocol)
}

func TestTokenBucket(t *testing.T) {
	bucket := &TokenBucket{
		Capacity:   100,
		Tokens:     100,
		RefillRate: 10,
		LastRefill: time.Now(),
	}
	
	// Test token consumption
	available := bucket.ConsumeTokens(50)
	assert.True(t, available, "Should be able to consume 50 tokens")
	assert.Equal(t, int64(50), bucket.Tokens, "Should have 50 tokens remaining")
	
	// Test over-consumption
	available = bucket.ConsumeTokens(60)
	assert.False(t, available, "Should not be able to consume 60 tokens")
	assert.Equal(t, int64(50), bucket.Tokens, "Token count should remain unchanged")
}

// ConsumeTokens method for TokenBucket
func (tb *TokenBucket) ConsumeTokens(tokens int64) bool {
	tb.mutex.Lock()
	defer tb.mutex.Unlock()
	
	if tb.Tokens >= tokens {
		tb.Tokens -= tokens
		return true
	}
	return false
}

func TestQoSPolicyPriority(t *testing.T) {
	policies := []*QoSPolicy{
		{ID: "p1", Priority: 50},
		{ID: "p2", Priority: 100},
		{ID: "p3", Priority: 25},
	}
	
	// Sort by priority (highest first)
	sort.Slice(policies, func(i, j int) bool {
		return policies[i].Priority > policies[j].Priority
	})
	
	assert.Equal(t, "p2", policies[0].ID, "Highest priority policy should be first")
	assert.Equal(t, "p1", policies[1].ID, "Medium priority policy should be second")
	assert.Equal(t, "p3", policies[2].ID, "Lowest priority policy should be last")
}