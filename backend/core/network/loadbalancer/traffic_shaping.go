package loadbalancer

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// TrafficShaper provides advanced traffic shaping and QoS management
type TrafficShaper struct {
	// Configuration
	config            TrafficShapingConfig
	
	// Traffic control components
	tokenBuckets      map[string]*TokenBucket
	bucketMutex       sync.RWMutex
	
	// QoS management
	qosManager        *QoSManager
	trafficClassifier *TrafficClassifier
	bandwidthAllocator *BandwidthAllocator
	
	// Priority queues
	priorityQueues    map[TrafficPriority]*PriorityQueue
	queueMutex        sync.RWMutex
	
	// Rate limiting
	shapingRules      map[string]*ShapingRule
	rulesMutex        sync.RWMutex
	
	// Connection tracking
	connectionTracker *ConnectionTracker
	
	// Metrics and monitoring
	metrics           *TrafficShapingMetrics
	metricsMutex      sync.RWMutex
	
	// Runtime state
	ctx               context.Context
	cancel            context.CancelFunc
	initialized       bool
}

// TrafficShapingConfig holds traffic shaping configuration
type TrafficShapingConfig struct {
	// Global bandwidth limits
	MaxBandwidthMbps      float64           `json:"max_bandwidth_mbps"`
	MaxIngressMbps        float64           `json:"max_ingress_mbps"`
	MaxEgressMbps         float64           `json:"max_egress_mbps"`
	
	// QoS settings
	EnableQoS             bool              `json:"enable_qos"`
	QoSMode              QoSMode           `json:"qos_mode"`
	DefaultTrafficClass   TrafficClass      `json:"default_traffic_class"`
	
	// Priority queue configuration
	PriorityLevels        int               `json:"priority_levels"`
	HighPriorityWeight    int               `json:"high_priority_weight"`
	MediumPriorityWeight  int               `json:"medium_priority_weight"`
	LowPriorityWeight     int               `json:"low_priority_weight"`
	
	// Rate limiting
	EnablePerIPLimiting   bool              `json:"enable_per_ip_limiting"`
	DefaultIPBandwidthMbps float64          `json:"default_ip_bandwidth_mbps"`
	EnablePerServiceLimiting bool          `json:"enable_per_service_limiting"`
	
	// Burst handling
	BurstAllowance        time.Duration     `json:"burst_allowance"`
	BurstMultiplier       float64           `json:"burst_multiplier"`
	
	// Connection limits
	MaxConnectionsPerIP   int               `json:"max_connections_per_ip"`
	MaxTotalConnections   int               `json:"max_total_connections"`
	ConnectionTimeoutSec  int               `json:"connection_timeout_sec"`
	
	// Buffer settings
	BufferSizeKB          int               `json:"buffer_size_kb"`
	MaxBufferSizeKB       int               `json:"max_buffer_size_kb"`
	BufferTimeoutMs       int               `json:"buffer_timeout_ms"`
	
	// Congestion control
	EnableCongestionControl bool            `json:"enable_congestion_control"`
	CongestionThreshold   float64           `json:"congestion_threshold"`
	CongestionBackoffMs   int               `json:"congestion_backoff_ms"`
	
	// Fair queuing
	EnableFairQueuing     bool              `json:"enable_fair_queuing"`
	FairQueueWeight       map[string]int    `json:"fair_queue_weight"`
	
	// Monitoring
	EnableMetrics         bool              `json:"enable_metrics"`
	MetricsInterval       time.Duration     `json:"metrics_interval"`
}

// Types and enums
type QoSMode string
type TrafficClass string
type ShapingAction string

const (
	QoSModeDisabled    QoSMode = "disabled"
	QoSModeBasic       QoSMode = "basic"
	QoSModeAdvanced    QoSMode = "advanced"
	QoSModeCustom      QoSMode = "custom"
	
	TrafficClassRealTime    TrafficClass = "realtime"
	TrafficClassInteractive TrafficClass = "interactive"
	TrafficClassBulk        TrafficClass = "bulk"
	TrafficClassBackground  TrafficClass = "background"
	
	// Traffic priorities for traffic shaping (aliases for TrafficPriority from types.go)
	TSPriorityHigh        TrafficPriority = TrafficPriorityHigh
	TSPriorityMedium      TrafficPriority = TrafficPriorityNormal  
	TSPriorityLow         TrafficPriority = TrafficPriorityLow
	TSPriorityBackground  TrafficPriority = TrafficPriorityLow
	
	// Traffic shaping actions (different from ActionType)
	ActionLimit         ShapingAction = "limit"
	ActionDelay         ShapingAction = "delay"
	ActionDrop          ShapingAction = "drop"
	ActionQueue         ShapingAction = "queue"
)

// TokenBucket implements token bucket rate limiting
type TokenBucket struct {
	// Configuration
	rate             float64       // tokens per second
	burst            int64         // max tokens
	
	// State
	tokens           int64         // current tokens
	lastRefill       time.Time     // last refill time
	mutex            sync.Mutex    // synchronization
}

// QoSManager manages Quality of Service policies
type QoSManager struct {
	policies         map[string]*QoSPolicy
	defaultPolicy    *QoSPolicy
	mutex            sync.RWMutex
}

// QoSPolicy defines QoS parameters for traffic
type QoSPolicy struct {
	ID               string          `json:"id"`
	Name             string          `json:"name"`
	Class            TrafficClass    `json:"class"`
	Priority         TrafficPriority `json:"priority"`
	MinBandwidthMbps float64         `json:"min_bandwidth_mbps"`
	MaxBandwidthMbps float64         `json:"max_bandwidth_mbps"`
	MaxLatencyMs     int             `json:"max_latency_ms"`
	MaxJitterMs      int             `json:"max_jitter_ms"`
	MinReliability   float64         `json:"min_reliability"`
	DSCP             int             `json:"dscp"`
	Weight           int             `json:"weight"`
	CreatedAt        time.Time       `json:"created_at"`
	UpdatedAt        time.Time       `json:"updated_at"`
}

// TrafficClassifier classifies traffic into QoS classes
type TrafficClassifier struct {
	rules            []*ClassificationRule
	mutex            sync.RWMutex
}

// ClassificationRule defines how to classify traffic
type ClassificationRule struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	Priority         int                    `json:"priority"`
	Conditions       []ClassificationCondition `json:"conditions"`
	Action           ClassificationAction   `json:"action"`
	Enabled          bool                   `json:"enabled"`
	CreatedAt        time.Time              `json:"created_at"`
	Statistics       RuleStatistics         `json:"statistics"`
}

// ClassificationCondition defines traffic matching conditions
type ClassificationCondition struct {
	Field            string                 `json:"field"`
	Operator         string                 `json:"operator"`
	Value            string                 `json:"value"`
	CaseSensitive    bool                   `json:"case_sensitive"`
}

// ClassificationAction defines the action for classified traffic
type ClassificationAction struct {
	TrafficClass     TrafficClass           `json:"traffic_class"`
	Priority         TrafficPriority        `json:"priority"`
	BandwidthLimitMbps float64             `json:"bandwidth_limit_mbps"`
	QoSPolicyID      string                 `json:"qos_policy_id"`
}

// BandwidthAllocator manages bandwidth allocation across services
type BandwidthAllocator struct {
	totalBandwidth   float64                // total available bandwidth (Mbps)
	allocations      map[string]*BandwidthAllocation
	mutex            sync.RWMutex
}

// BandwidthAllocation represents allocated bandwidth for an entity
type BandwidthAllocation struct {
	EntityID         string                 `json:"entity_id"`
	EntityType       string                 `json:"entity_type"`
	AllocatedMbps    float64                `json:"allocated_mbps"`
	UsedMbps         float64                `json:"used_mbps"`
	MinGuaranteedMbps float64               `json:"min_guaranteed_mbps"`
	MaxAllowedMbps   float64                `json:"max_allowed_mbps"`
	Priority         TrafficPriority        `json:"priority"`
	LastUpdated      time.Time              `json:"last_updated"`
}

// PriorityQueue implements priority-based packet queuing
type PriorityQueue struct {
	priority         TrafficPriority        `json:"priority"`
	weight           int                    `json:"weight"`
	maxSize          int                    `json:"max_size"`
	currentSize      int64                  `json:"current_size"`
	packets          chan *QueuedPacket     `json:"-"`
	droppedPackets   int64                  `json:"dropped_packets"`
	processedPackets int64                  `json:"processed_packets"`
	mutex            sync.RWMutex
}

// QueuedPacket represents a packet in a priority queue
type QueuedPacket struct {
	ID               string                 `json:"id"`
	Priority         TrafficPriority        `json:"priority"`
	Size             int64                  `json:"size"`
	QueuedAt         time.Time              `json:"queued_at"`
	ExpiresAt        time.Time              `json:"expires_at"`
	Metadata         map[string]interface{} `json:"metadata"`
	ProcessCallback  func() error           `json:"-"`
}

// ShapingRule defines traffic shaping behavior
type ShapingRule struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	Enabled          bool                   `json:"enabled"`
	Priority         int                    `json:"priority"`
	
	// Matching criteria
	SourceIP         string                 `json:"source_ip,omitempty"`
	SourceSubnet     string                 `json:"source_subnet,omitempty"`
	DestinationPort  int                    `json:"destination_port,omitempty"`
	Protocol         string                 `json:"protocol,omitempty"`
	ServiceID        string                 `json:"service_id,omitempty"`
	
	// Shaping parameters
	Action           ShapingAction          `json:"action"`
	BandwidthLimitMbps float64             `json:"bandwidth_limit_mbps,omitempty"`
	BurstSizeKB      int64                  `json:"burst_size_kb,omitempty"`
	DelayMs          int                    `json:"delay_ms,omitempty"`
	TrafficPriority  TrafficPriority        `json:"traffic_priority,omitempty"`
	
	// Statistics
	Statistics       ShapingRuleStatistics  `json:"statistics"`
	CreatedAt        time.Time              `json:"created_at"`
	UpdatedAt        time.Time              `json:"updated_at"`
}

// ShapingRuleStatistics holds statistics for a shaping rule
type ShapingRuleStatistics struct {
	MatchedPackets   int64                  `json:"matched_packets"`
	ShapedPackets    int64                  `json:"shaped_packets"`
	DroppedPackets   int64                  `json:"dropped_packets"`
	DelayedPackets   int64                  `json:"delayed_packets"`
	BytesProcessed   int64                  `json:"bytes_processed"`
	LastUpdated      time.Time              `json:"last_updated"`
}

// ConnectionTracker tracks active connections for limiting
type ConnectionTracker struct {
	connections      map[string]*ConnectionInfo
	ipConnections    map[string]int
	totalConnections int64
	mutex            sync.RWMutex
}

// ConnectionInfo holds information about a connection
type ConnectionInfo struct {
	ID               string                 `json:"id"`
	SourceIP         string                 `json:"source_ip"`
	DestinationIP    string                 `json:"destination_ip"`
	SourcePort       int                    `json:"source_port"`
	DestinationPort  int                    `json:"destination_port"`
	Protocol         string                 `json:"protocol"`
	CreatedAt        time.Time              `json:"created_at"`
	LastActivity     time.Time              `json:"last_activity"`
	BytesSent        int64                  `json:"bytes_sent"`
	BytesReceived    int64                  `json:"bytes_received"`
	QoSClass         TrafficClass           `json:"qos_class"`
	Priority         TrafficPriority        `json:"priority"`
}

// TrafficShapingMetrics holds traffic shaping metrics
type TrafficShapingMetrics struct {
	TotalPackets         int64                  `json:"total_packets"`
	ShapedPackets        int64                  `json:"shaped_packets"`
	DroppedPackets       int64                  `json:"dropped_packets"`
	DelayedPackets       int64                  `json:"delayed_packets"`
	QueuedPackets        int64                  `json:"queued_packets"`
	
	TotalBandwidthMbps   float64                `json:"total_bandwidth_mbps"`
	UsedBandwidthMbps    float64                `json:"used_bandwidth_mbps"`
	AvailableBandwidthMbps float64              `json:"available_bandwidth_mbps"`
	
	PacketsByPriority    map[TrafficPriority]int64 `json:"packets_by_priority"`
	PacketsByClass       map[TrafficClass]int64     `json:"packets_by_class"`
	
	ActiveConnections    int64                  `json:"active_connections"`
	ConnectionsByIP      map[string]int64       `json:"connections_by_ip"`
	
	QueueUtilization     map[TrafficPriority]float64 `json:"queue_utilization"`
	AverageLatencyMs     float64                `json:"average_latency_ms"`
	PacketLossRatio      float64                `json:"packet_loss_ratio"`
	
	LastUpdated          time.Time              `json:"last_updated"`
}

// NewTrafficShaper creates a new traffic shaper
func NewTrafficShaper(config TrafficShapingConfig) *TrafficShaper {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &TrafficShaper{
		config:        config,
		tokenBuckets:  make(map[string]*TokenBucket),
		shapingRules:  make(map[string]*ShapingRule),
		priorityQueues: make(map[TrafficPriority]*PriorityQueue),
		metrics: &TrafficShapingMetrics{
			PacketsByPriority: make(map[TrafficPriority]int64),
			PacketsByClass:    make(map[TrafficClass]int64),
			ConnectionsByIP:   make(map[string]int64),
			QueueUtilization:  make(map[TrafficPriority]float64),
			LastUpdated:       time.Now(),
		},
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start initializes and starts the traffic shaper
func (ts *TrafficShaper) Start() error {
	if ts.initialized {
		return fmt.Errorf("traffic shaper already started")
	}
	
	// Initialize QoS manager
	ts.qosManager = &QoSManager{
		policies: make(map[string]*QoSPolicy),
		defaultPolicy: &QoSPolicy{
			ID:               "default",
			Name:             "Default Policy",
			Class:            ts.config.DefaultTrafficClass,
			Priority:         TSPriorityMedium,
			MinBandwidthMbps: 0,
			MaxBandwidthMbps: ts.config.MaxBandwidthMbps,
			MaxLatencyMs:     1000,
			MaxJitterMs:      100,
			MinReliability:   0.99,
			DSCP:             0,
			Weight:           1,
			CreatedAt:        time.Now(),
			UpdatedAt:        time.Now(),
		},
	}
	
	// Initialize traffic classifier
	ts.trafficClassifier = &TrafficClassifier{
		rules: make([]*ClassificationRule, 0),
	}
	
	// Initialize bandwidth allocator
	ts.bandwidthAllocator = &BandwidthAllocator{
		totalBandwidth: ts.config.MaxBandwidthMbps,
		allocations:    make(map[string]*BandwidthAllocation),
	}
	
	// Initialize priority queues
	ts.initializePriorityQueues()
	
	// Initialize connection tracker
	ts.connectionTracker = &ConnectionTracker{
		connections:   make(map[string]*ConnectionInfo),
		ipConnections: make(map[string]int),
	}
	
	// Create default shaping rules
	ts.createDefaultShapingRules()
	
	// Start background processes
	go ts.queueProcessingLoop()
	go ts.bandwidthMonitoringLoop()
	go ts.connectionCleanupLoop()
	
	if ts.config.EnableMetrics {
		go ts.metricsCollectionLoop()
	}
	
	ts.initialized = true
	return nil
}

// Stop stops the traffic shaper
func (ts *TrafficShaper) Stop() error {
	ts.cancel()
	
	// Close all priority queues
	ts.queueMutex.Lock()
	for _, queue := range ts.priorityQueues {
		close(queue.packets)
	}
	ts.queueMutex.Unlock()
	
	ts.initialized = false
	return nil
}

// initializePriorityQueues creates and configures priority queues
func (ts *TrafficShaper) initializePriorityQueues() {
	priorities := []TrafficPriority{TSPriorityHigh, TSPriorityMedium, TSPriorityLow, TSPriorityBackground}
	weights := map[TrafficPriority]int{
		TSPriorityHigh:       ts.config.HighPriorityWeight,
		TSPriorityMedium:     ts.config.MediumPriorityWeight,
		TSPriorityLow:        ts.config.LowPriorityWeight,
		TSPriorityBackground: 1,
	}
	
	for _, priority := range priorities {
		weight := weights[priority]
		if weight == 0 {
			weight = 1
		}
		
		ts.priorityQueues[priority] = &PriorityQueue{
			priority:    priority,
			weight:      weight,
			maxSize:     1000, // Configurable
			packets:     make(chan *QueuedPacket, 1000),
		}
	}
}

// createDefaultShapingRules creates default traffic shaping rules
func (ts *TrafficShaper) createDefaultShapingRules() {
	// Default rule for background traffic
	backgroundRule := &ShapingRule{
		ID:                 "default-background",
		Name:               "Default Background Traffic",
		Enabled:            true,
		Priority:           100,
		Action:             ActionLimit,
		BandwidthLimitMbps: ts.config.MaxBandwidthMbps * 0.1, // 10% for background
		Priority:           TSPriorityBackground,
		Statistics:         ShapingRuleStatistics{LastUpdated: time.Now()},
		CreatedAt:          time.Now(),
		UpdatedAt:          time.Now(),
	}
	
	ts.rulesMutex.Lock()
	ts.shapingRules[backgroundRule.ID] = backgroundRule
	ts.rulesMutex.Unlock()
}

// ShapeTraffic applies traffic shaping to a request
func (ts *TrafficShaper) ShapeTraffic(sourceIP, serviceID string, packetSize int64) (*TrafficShapingDecision, error) {
	// Classify traffic
	classification := ts.classifyTraffic(sourceIP, serviceID, packetSize)
	
	// Check connection limits
	if ts.config.MaxConnectionsPerIP > 0 {
		if ts.connectionTracker.getConnectionCountByIP(sourceIP) >= ts.config.MaxConnectionsPerIP {
			return &TrafficShapingDecision{
				Action:    ActionDrop,
				Reason:    "Connection limit per IP exceeded",
				Priority:  TSPriorityBackground,
			}, nil
		}
	}
	
	// Apply shaping rules
	decision := ts.applyShapingRules(sourceIP, serviceID, packetSize, classification)
	if decision.Action != ActionAllow {
		ts.updateMetrics(decision, packetSize)
		return decision, nil
	}
	
	// Check bandwidth availability
	if ts.config.MaxBandwidthMbps > 0 {
		currentUsage := ts.getCurrentBandwidthUsage()
		if currentUsage+ts.bytesToMbps(packetSize) > ts.config.MaxBandwidthMbps {
			// Try to queue the packet if congestion control is enabled
			if ts.config.EnableCongestionControl {
				decision = ts.handleCongestion(sourceIP, serviceID, packetSize, classification)
			} else {
				decision = &TrafficShapingDecision{
					Action:   ActionDrop,
					Reason:   "Bandwidth limit exceeded",
					Priority: classification.Priority,
				}
			}
		}
	}
	
	// Apply token bucket rate limiting
	if decision.Action == ActionAllow {
		decision = ts.applyTokenBucketLimiting(sourceIP, packetSize, classification)
	}
	
	ts.updateMetrics(decision, packetSize)
	return decision, nil
}

// TrafficShapingDecision represents the decision for traffic shaping
type TrafficShapingDecision struct {
	Action           ShapingAction          `json:"action"`
	Reason           string                 `json:"reason"`
	Priority         TrafficPriority        `json:"priority"`
	DelayMs          int                    `json:"delay_ms,omitempty"`
	BandwidthLimitMbps float64             `json:"bandwidth_limit_mbps,omitempty"`
	QueueID          string                 `json:"queue_id,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

// TrafficClassification holds traffic classification result
type TrafficClassification struct {
	Class            TrafficClass           `json:"class"`
	Priority         TrafficPriority        `json:"priority"`
	QoSPolicyID      string                 `json:"qos_policy_id"`
	BandwidthLimitMbps float64             `json:"bandwidth_limit_mbps"`
}

// classifyTraffic classifies traffic based on configured rules
func (ts *TrafficShaper) classifyTraffic(sourceIP, serviceID string, packetSize int64) *TrafficClassification {
	ts.trafficClassifier.mutex.RLock()
	defer ts.trafficClassifier.mutex.RUnlock()
	
	// Apply classification rules in priority order
	for _, rule := range ts.trafficClassifier.rules {
		if !rule.Enabled {
			continue
		}
		
		if ts.matchesClassificationRule(rule, sourceIP, serviceID, packetSize) {
			atomic.AddInt64(&rule.Statistics.TotalMatches, 1)
			return &TrafficClassification{
				Class:              rule.Action.TrafficClass,
				Priority:           rule.Action.Priority,
				QoSPolicyID:        rule.Action.QoSPolicyID,
				BandwidthLimitMbps: rule.Action.BandwidthLimitMbps,
			}
		}
	}
	
	// Default classification
	return &TrafficClassification{
		Class:              ts.config.DefaultTrafficClass,
		Priority:           TSPriorityMedium,
		QoSPolicyID:        "default",
		BandwidthLimitMbps: ts.config.MaxBandwidthMbps,
	}
}

// matchesClassificationRule checks if traffic matches a classification rule
func (ts *TrafficShaper) matchesClassificationRule(rule *ClassificationRule, sourceIP, serviceID string, packetSize int64) bool {
	// Simplified rule matching - in practice this would be more comprehensive
	for _, condition := range rule.Conditions {
		switch condition.Field {
		case "source_ip":
			if sourceIP != condition.Value {
				return false
			}
		case "service_id":
			if serviceID != condition.Value {
				return false
			}
		case "packet_size":
			// Would need to parse numeric conditions
		}
	}
	
	return true
}

// applyShapingRules applies traffic shaping rules
func (ts *TrafficShaper) applyShapingRules(sourceIP, serviceID string, packetSize int64, classification *TrafficClassification) *TrafficShapingDecision {
	ts.rulesMutex.RLock()
	defer ts.rulesMutex.RUnlock()
	
	// Check rules in priority order (higher priority first)
	var matchedRules []*ShapingRule
	for _, rule := range ts.shapingRules {
		if !rule.Enabled {
			continue
		}
		
		if ts.matchesShapingRule(rule, sourceIP, serviceID) {
			matchedRules = append(matchedRules, rule)
		}
	}
	
	// Sort by priority
	for i := 0; i < len(matchedRules)-1; i++ {
		for j := i + 1; j < len(matchedRules); j++ {
			if matchedRules[i].Priority < matchedRules[j].Priority {
				matchedRules[i], matchedRules[j] = matchedRules[j], matchedRules[i]
			}
		}
	}
	
	// Apply first matching rule
	for _, rule := range matchedRules {
		atomic.AddInt64(&rule.Statistics.MatchedPackets, 1)
		
		switch rule.Action {
		case ActionDrop:
			atomic.AddInt64(&rule.Statistics.DroppedPackets, 1)
			return &TrafficShapingDecision{
				Action: ActionDrop,
				Reason: fmt.Sprintf("Matched rule: %s", rule.Name),
			}
		case ActionLimit:
			if rule.BandwidthLimitMbps > 0 && ts.bytesToMbps(packetSize) > rule.BandwidthLimitMbps {
				atomic.AddInt64(&rule.Statistics.ShapedPackets, 1)
				return &TrafficShapingDecision{
					Action:             ActionLimit,
					Reason:             fmt.Sprintf("Bandwidth limited by rule: %s", rule.Name),
					BandwidthLimitMbps: rule.BandwidthLimitMbps,
				}
			}
		case ActionDelay:
			atomic.AddInt64(&rule.Statistics.DelayedPackets, 1)
			return &TrafficShapingDecision{
				Action:  ActionDelay,
				Reason:  fmt.Sprintf("Delayed by rule: %s", rule.Name),
				DelayMs: rule.DelayMs,
			}
		case ActionQueue:
			return &TrafficShapingDecision{
				Action:   ActionQueue,
				Reason:   fmt.Sprintf("Queued by rule: %s", rule.Name),
				Priority: rule.Priority,
			}
		}
	}
	
	return &TrafficShapingDecision{Action: ActionAllow}
}

// matchesShapingRule checks if traffic matches a shaping rule
func (ts *TrafficShaper) matchesShapingRule(rule *ShapingRule, sourceIP, serviceID string) bool {
	if rule.SourceIP != "" && rule.SourceIP != sourceIP {
		return false
	}
	
	if rule.ServiceID != "" && rule.ServiceID != serviceID {
		return false
	}
	
	// Add more matching criteria as needed
	
	return true
}

// applyTokenBucketLimiting applies token bucket rate limiting
func (ts *TrafficShaper) applyTokenBucketLimiting(sourceIP string, packetSize int64, classification *TrafficClassification) *TrafficShapingDecision {
	if !ts.config.EnablePerIPLimiting {
		return &TrafficShapingDecision{Action: ActionAllow}
	}
	
	bucket := ts.getOrCreateTokenBucket(sourceIP, classification)
	
	if !bucket.consume(packetSize) {
		return &TrafficShapingDecision{
			Action: ActionLimit,
			Reason: "Token bucket rate limit exceeded",
		}
	}
	
	return &TrafficShapingDecision{Action: ActionAllow}
}

// getOrCreateTokenBucket gets or creates a token bucket for an entity
func (ts *TrafficShaper) getOrCreateTokenBucket(key string, classification *TrafficClassification) *TokenBucket {
	ts.bucketMutex.Lock()
	defer ts.bucketMutex.Unlock()
	
	bucket, exists := ts.tokenBuckets[key]
	if !exists {
		rate := ts.config.DefaultIPBandwidthMbps * 1024 * 1024 / 8 // Convert Mbps to bytes/sec
		burst := int64(rate * ts.config.BurstAllowance.Seconds())
		
		if classification.BandwidthLimitMbps > 0 {
			rate = classification.BandwidthLimitMbps * 1024 * 1024 / 8
			burst = int64(rate * ts.config.BurstAllowance.Seconds())
		}
		
		bucket = NewTokenBucket(rate, burst)
		ts.tokenBuckets[key] = bucket
	}
	
	return bucket
}

// handleCongestion handles network congestion
func (ts *TrafficShaper) handleCongestion(sourceIP, serviceID string, packetSize int64, classification *TrafficClassification) *TrafficShapingDecision {
	// Try to queue the packet based on priority
	queue := ts.priorityQueues[classification.Priority]
	if queue == nil {
		queue = ts.priorityQueues[PriorityMedium] // Fallback
	}
	
	packet := &QueuedPacket{
		ID:        uuid.New().String(),
		Priority:  classification.Priority,
		Size:      packetSize,
		QueuedAt:  time.Now(),
		ExpiresAt: time.Now().Add(time.Duration(ts.config.BufferTimeoutMs) * time.Millisecond),
		Metadata: map[string]interface{}{
			"source_ip":  sourceIP,
			"service_id": serviceID,
		},
	}
	
	select {
	case queue.packets <- packet:
		atomic.AddInt64(&queue.currentSize, 1)
		return &TrafficShapingDecision{
			Action:   ActionQueue,
			Reason:   "Queued due to congestion",
			Priority: classification.Priority,
			QueueID:  packet.ID,
		}
	default:
		// Queue is full, drop packet
		atomic.AddInt64(&queue.droppedPackets, 1)
		return &TrafficShapingDecision{
			Action: ActionDrop,
			Reason: "Queue full, packet dropped",
		}
	}
}

// getCurrentBandwidthUsage calculates current bandwidth usage
func (ts *TrafficShaper) getCurrentBandwidthUsage() float64 {
	ts.bandwidthAllocator.mutex.RLock()
	defer ts.bandwidthAllocator.mutex.RUnlock()
	
	var totalUsed float64
	for _, allocation := range ts.bandwidthAllocator.allocations {
		totalUsed += allocation.UsedMbps
	}
	
	return totalUsed
}

// bytesToMbps converts bytes per second to megabits per second
func (ts *TrafficShaper) bytesToMbps(bytes int64) float64 {
	return float64(bytes) * 8 / (1024 * 1024)
}

// Token bucket implementation

// NewTokenBucket creates a new token bucket
func NewTokenBucket(rate float64, burst int64) *TokenBucket {
	return &TokenBucket{
		rate:       rate,
		burst:      burst,
		tokens:     burst,
		lastRefill: time.Now(),
	}
}

// consume tries to consume tokens from the bucket
func (tb *TokenBucket) consume(tokens int64) bool {
	tb.mutex.Lock()
	defer tb.mutex.Unlock()
	
	now := time.Now()
	elapsed := now.Sub(tb.lastRefill).Seconds()
	
	// Refill tokens based on elapsed time
	tokensToAdd := int64(elapsed * tb.rate)
	tb.tokens += tokensToAdd
	
	// Cap at burst size
	if tb.tokens > tb.burst {
		tb.tokens = tb.burst
	}
	
	tb.lastRefill = now
	
	// Try to consume tokens
	if tb.tokens >= tokens {
		tb.tokens -= tokens
		return true
	}
	
	return false
}

// getTokens returns current token count
func (tb *TokenBucket) getTokens() int64 {
	tb.mutex.Lock()
	defer tb.mutex.Unlock()
	return tb.tokens
}

// Connection tracking implementation

// trackConnection starts tracking a connection
func (ct *ConnectionTracker) trackConnection(connInfo *ConnectionInfo) {
	ct.mutex.Lock()
	defer ct.mutex.Unlock()
	
	ct.connections[connInfo.ID] = connInfo
	ct.ipConnections[connInfo.SourceIP]++
	atomic.AddInt64(&ct.totalConnections, 1)
}

// untrackConnection stops tracking a connection
func (ct *ConnectionTracker) untrackConnection(connID string) {
	ct.mutex.Lock()
	defer ct.mutex.Unlock()
	
	if conn, exists := ct.connections[connID]; exists {
		delete(ct.connections, connID)
		ct.ipConnections[conn.SourceIP]--
		
		if ct.ipConnections[conn.SourceIP] <= 0 {
			delete(ct.ipConnections, conn.SourceIP)
		}
		
		atomic.AddInt64(&ct.totalConnections, -1)
	}
}

// getConnectionCountByIP returns connection count for an IP
func (ct *ConnectionTracker) getConnectionCountByIP(ip string) int {
	ct.mutex.RLock()
	defer ct.mutex.RUnlock()
	return ct.ipConnections[ip]
}

// getTotalConnectionCount returns total connection count
func (ct *ConnectionTracker) getTotalConnectionCount() int64 {
	return atomic.LoadInt64(&ct.totalConnections)
}

// Background processing loops

// queueProcessingLoop processes queued packets
func (ts *TrafficShaper) queueProcessingLoop() {
	ticker := time.NewTicker(10 * time.Millisecond) // Process every 10ms
	defer ticker.Stop()
	
	for {
		select {
		case <-ts.ctx.Done():
			return
		case <-ticker.C:
			ts.processQueues()
		}
	}
}

// processQueues processes packets from priority queues
func (ts *TrafficShaper) processQueues() {
	priorities := []TrafficPriority{TSPriorityHigh, TSPriorityMedium, TSPriorityLow, TSPriorityBackground}
	
	for _, priority := range priorities {
		queue := ts.priorityQueues[priority]
		if queue == nil {
			continue
		}
		
		// Process packets based on weight
		packetsToProcess := queue.weight
		
		for i := 0; i < packetsToProcess; i++ {
			select {
			case packet := <-queue.packets:
				// Check if packet has expired
				if time.Now().After(packet.ExpiresAt) {
					atomic.AddInt64(&queue.droppedPackets, 1)
					continue
				}
				
				// Process packet
				if packet.ProcessCallback != nil {
					packet.ProcessCallback()
				}
				
				atomic.AddInt64(&queue.processedPackets, 1)
				atomic.AddInt64(&queue.currentSize, -1)
			default:
				break // No more packets in queue
			}
		}
	}
}

// bandwidthMonitoringLoop monitors bandwidth usage
func (ts *TrafficShaper) bandwidthMonitoringLoop() {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ts.ctx.Done():
			return
		case <-ticker.C:
			ts.updateBandwidthAllocations()
		}
	}
}

// updateBandwidthAllocations updates bandwidth allocations
func (ts *TrafficShaper) updateBandwidthAllocations() {
	ts.bandwidthAllocator.mutex.Lock()
	defer ts.bandwidthAllocator.mutex.Unlock()
	
	// Reset usage counters (simplified)
	for _, allocation := range ts.bandwidthAllocator.allocations {
		allocation.UsedMbps = 0 // Would be updated based on actual usage
		allocation.LastUpdated = time.Now()
	}
}

// connectionCleanupLoop cleans up expired connections
func (ts *TrafficShaper) connectionCleanupLoop() {
	ticker := time.NewTicker(30 * time.Second) // Cleanup every 30 seconds
	defer ticker.Stop()
	
	for {
		select {
		case <-ts.ctx.Done():
			return
		case <-ticker.C:
			ts.cleanupExpiredConnections()
		}
	}
}

// cleanupExpiredConnections removes expired connections
func (ts *TrafficShaper) cleanupExpiredConnections() {
	ts.connectionTracker.mutex.Lock()
	defer ts.connectionTracker.mutex.Unlock()
	
	now := time.Now()
	timeout := time.Duration(ts.config.ConnectionTimeoutSec) * time.Second
	
	for connID, conn := range ts.connectionTracker.connections {
		if now.Sub(conn.LastActivity) > timeout {
			delete(ts.connectionTracker.connections, connID)
			ts.connectionTracker.ipConnections[conn.SourceIP]--
			
			if ts.connectionTracker.ipConnections[conn.SourceIP] <= 0 {
				delete(ts.connectionTracker.ipConnections, conn.SourceIP)
			}
			
			atomic.AddInt64(&ts.connectionTracker.totalConnections, -1)
		}
	}
}

// metricsCollectionLoop collects traffic shaping metrics
func (ts *TrafficShaper) metricsCollectionLoop() {
	ticker := time.NewTicker(ts.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ts.ctx.Done():
			return
		case <-ticker.C:
			ts.updateMetrics(&TrafficShapingDecision{}, 0)
		}
	}
}

// updateMetrics updates traffic shaping metrics
func (ts *TrafficShaper) updateMetrics(decision *TrafficShapingDecision, packetSize int64) {
	ts.metricsMutex.Lock()
	defer ts.metricsMutex.Unlock()
	
	if decision != nil {
		atomic.AddInt64(&ts.metrics.TotalPackets, 1)
		
		switch decision.Action {
		case ActionLimit:
			atomic.AddInt64(&ts.metrics.ShapedPackets, 1)
		case ActionDrop:
			atomic.AddInt64(&ts.metrics.DroppedPackets, 1)
		case ActionDelay:
			atomic.AddInt64(&ts.metrics.DelayedPackets, 1)
		case ActionQueue:
			atomic.AddInt64(&ts.metrics.QueuedPackets, 1)
		}
		
		if decision.Priority != "" {
			ts.metrics.PacketsByPriority[decision.Priority]++
		}
	}
	
	// Update bandwidth metrics
	ts.metrics.UsedBandwidthMbps = ts.getCurrentBandwidthUsage()
	ts.metrics.AvailableBandwidthMbps = ts.config.MaxBandwidthMbps - ts.metrics.UsedBandwidthMbps
	
	// Update connection metrics
	ts.metrics.ActiveConnections = ts.connectionTracker.getTotalConnectionCount()
	
	// Update queue utilization
	for priority, queue := range ts.priorityQueues {
		if queue.maxSize > 0 {
			utilization := float64(atomic.LoadInt64(&queue.currentSize)) / float64(queue.maxSize)
			ts.metrics.QueueUtilization[priority] = utilization
		}
	}
	
	// Calculate packet loss ratio
	if ts.metrics.TotalPackets > 0 {
		ts.metrics.PacketLossRatio = float64(ts.metrics.DroppedPackets) / float64(ts.metrics.TotalPackets)
	}
	
	ts.metrics.LastUpdated = time.Now()
}

// Public API methods

// CreateQoSPolicy creates a new QoS policy
func (ts *TrafficShaper) CreateQoSPolicy(policy *QoSPolicy) error {
	if policy.ID == "" {
		policy.ID = uuid.New().String()
	}
	
	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()
	
	ts.qosManager.mutex.Lock()
	ts.qosManager.policies[policy.ID] = policy
	ts.qosManager.mutex.Unlock()
	
	return nil
}

// CreateShapingRule creates a new traffic shaping rule
func (ts *TrafficShaper) CreateShapingRule(rule *ShapingRule) error {
	if rule.ID == "" {
		rule.ID = uuid.New().String()
	}
	
	rule.Statistics = ShapingRuleStatistics{LastUpdated: time.Now()}
	rule.CreatedAt = time.Now()
	rule.UpdatedAt = time.Now()
	
	ts.rulesMutex.Lock()
	ts.shapingRules[rule.ID] = rule
	ts.rulesMutex.Unlock()
	
	return nil
}

// GetMetrics returns traffic shaping metrics
func (ts *TrafficShaper) GetMetrics() *TrafficShapingMetrics {
	ts.metricsMutex.RLock()
	defer ts.metricsMutex.RUnlock()
	
	// Return copy of metrics
	metricsCopy := *ts.metrics
	
	// Copy maps
	metricsCopy.PacketsByPriority = make(map[TrafficPriority]int64)
	for k, v := range ts.metrics.PacketsByPriority {
		metricsCopy.PacketsByPriority[k] = v
	}
	
	metricsCopy.PacketsByClass = make(map[TrafficClass]int64)
	for k, v := range ts.metrics.PacketsByClass {
		metricsCopy.PacketsByClass[k] = v
	}
	
	metricsCopy.ConnectionsByIP = make(map[string]int64)
	for k, v := range ts.metrics.ConnectionsByIP {
		metricsCopy.ConnectionsByIP[k] = v
	}
	
	metricsCopy.QueueUtilization = make(map[TrafficPriority]float64)
	for k, v := range ts.metrics.QueueUtilization {
		metricsCopy.QueueUtilization[k] = v
	}
	
	return &metricsCopy
}

// GetShapingRules returns all traffic shaping rules
func (ts *TrafficShaper) GetShapingRules() []*ShapingRule {
	ts.rulesMutex.RLock()
	defer ts.rulesMutex.RUnlock()
	
	rules := make([]*ShapingRule, 0, len(ts.shapingRules))
	for _, rule := range ts.shapingRules {
		ruleCopy := *rule
		rules = append(rules, &ruleCopy)
	}
	
	return rules
}

// GetQoSPolicies returns all QoS policies
func (ts *TrafficShaper) GetQoSPolicies() []*QoSPolicy {
	ts.qosManager.mutex.RLock()
	defer ts.qosManager.mutex.RUnlock()
	
	policies := make([]*QoSPolicy, 0, len(ts.qosManager.policies))
	for _, policy := range ts.qosManager.policies {
		policyCopy := *policy
		policies = append(policies, &policyCopy)
	}
	
	return policies
}

// DefaultTrafficShapingConfig returns default traffic shaping configuration
func DefaultTrafficShapingConfig() TrafficShapingConfig {
	return TrafficShapingConfig{
		MaxBandwidthMbps:         1000.0, // 1 Gbps
		MaxIngressMbps:           500.0,
		MaxEgressMbps:            500.0,
		EnableQoS:                true,
		QoSMode:                  QoSModeAdvanced,
		DefaultTrafficClass:      TrafficClassInteractive,
		PriorityLevels:           4,
		HighPriorityWeight:       8,
		MediumPriorityWeight:     4,
		LowPriorityWeight:        2,
		EnablePerIPLimiting:      true,
		DefaultIPBandwidthMbps:   10.0,
		EnablePerServiceLimiting: true,
		BurstAllowance:           time.Second,
		BurstMultiplier:          2.0,
		MaxConnectionsPerIP:      100,
		MaxTotalConnections:      10000,
		ConnectionTimeoutSec:     300,
		BufferSizeKB:             64,
		MaxBufferSizeKB:          1024,
		BufferTimeoutMs:          1000,
		EnableCongestionControl:  true,
		CongestionThreshold:      0.8,
		CongestionBackoffMs:      100,
		EnableFairQueuing:        true,
		FairQueueWeight:          make(map[string]int),
		EnableMetrics:            true,
		MetricsInterval:          30 * time.Second,
	}
}