package ovs

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// QoSManager manages Quality of Service policies for OVS bridges
type QoSManager struct {
	bridgeManager *BridgeManager
	qosPolicies   map[string]*QoSPolicyConfig
	qosRules      map[string]*QoSRuleConfig
	queueConfigs  map[string]*QueueConfiguration
	policiesMutex sync.RWMutex
	rulesMutex    sync.RWMutex
	queuesMutex   sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
	config        QoSManagerConfig
}

// QoSManagerConfig holds configuration for QoS management
type QoSManagerConfig struct {
	DefaultQoSType        QoSType       `json:"default_qos_type"`
	MaxQueues            int           `json:"max_queues"`
	DefaultMinRate       int64         `json:"default_min_rate_bps"`
	DefaultMaxRate       int64         `json:"default_max_rate_bps"`
	MonitoringInterval   time.Duration `json:"monitoring_interval"`
	EnableBurstControl   bool          `json:"enable_burst_control"`
	EnablePriorityQueues bool          `json:"enable_priority_queues"`
	EnableDSCPMarking    bool          `json:"enable_dscp_marking"`
}

// DefaultQoSManagerConfig returns default QoS configuration
func DefaultQoSManagerConfig() QoSManagerConfig {
	return QoSManagerConfig{
		DefaultQoSType:        QoSTypeHTB,
		MaxQueues:            8,
		DefaultMinRate:       1000000,  // 1 Mbps
		DefaultMaxRate:       100000000, // 100 Mbps
		MonitoringInterval:   30 * time.Second,
		EnableBurstControl:   true,
		EnablePriorityQueues: true,
		EnableDSCPMarking:    true,
	}
}

// QoSPolicyConfig represents a comprehensive QoS policy configuration
type QoSPolicyConfig struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Description       string                 `json:"description"`
	BridgeName        string                 `json:"bridge_name"`
	Type              QoSType                `json:"type"`
	Queues            map[int]*QueueConfig   `json:"queues"`
	Rules             []*TrafficRule         `json:"rules"`
	DefaultQueueID    int                    `json:"default_queue_id"`
	Status            PolicyStatus           `json:"status"`
	Statistics        PolicyStatistics       `json:"statistics"`
	Options           map[string]string      `json:"options,omitempty"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
	LastAppliedAt     time.Time              `json:"last_applied_at"`
}

// QueueConfig represents a traffic queue configuration
type QueueConfig struct {
	ID             int                    `json:"id"`
	Name           string                 `json:"name"`
	MinRate        int64                  `json:"min_rate_bps"`
	MaxRate        int64                  `json:"max_rate_bps"`
	BurstSize      int64                  `json:"burst_size_bytes"`
	Priority       int                    `json:"priority"`
	DSCP           int                    `json:"dscp,omitempty"`
	CeilRate       int64                  `json:"ceil_rate_bps,omitempty"`
	Weight         int                    `json:"weight,omitempty"`
	Quantum        int                    `json:"quantum,omitempty"`
	Options        map[string]string      `json:"options,omitempty"`
	Statistics     QueueStatistics        `json:"statistics"`
	LastUpdated    time.Time              `json:"last_updated"`
}

// TrafficRule represents a rule for traffic classification
type TrafficRule struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Priority        int                    `json:"priority"`
	Match           TrafficMatch           `json:"match"`
	Action          TrafficAction          `json:"action"`
	QueueID         int                    `json:"queue_id"`
	Enabled         bool                   `json:"enabled"`
	HitCount        int64                  `json:"hit_count"`
	ByteCount       int64                  `json:"byte_count"`
	LastHit         time.Time              `json:"last_hit"`
	CreatedAt       time.Time              `json:"created_at"`
}

// TrafficMatch represents traffic matching criteria
type TrafficMatch struct {
	InPort         string   `json:"in_port,omitempty"`
	EtherType      string   `json:"ether_type,omitempty"`
	SrcMAC         string   `json:"src_mac,omitempty"`
	DstMAC         string   `json:"dst_mac,omitempty"`
	SrcIP          string   `json:"src_ip,omitempty"`
	DstIP          string   `json:"dst_ip,omitempty"`
	Protocol       string   `json:"protocol,omitempty"`
	SrcPort        int      `json:"src_port,omitempty"`
	DstPort        int      `json:"dst_port,omitempty"`
	DSCP           int      `json:"dscp,omitempty"`
	VlanID         int      `json:"vlan_id,omitempty"`
	TCPFlags       []string `json:"tcp_flags,omitempty"`
	PacketSize     string   `json:"packet_size,omitempty"`
	CustomFields   map[string]string `json:"custom_fields,omitempty"`
}

// TrafficAction represents actions to take on matched traffic
type TrafficAction struct {
	Type           ActionType             `json:"type"`
	QueueID        int                    `json:"queue_id,omitempty"`
	DSCP           int                    `json:"dscp,omitempty"`
	Priority       int                    `json:"priority,omitempty"`
	RateLimit      int64                  `json:"rate_limit_bps,omitempty"`
	BurstLimit     int64                  `json:"burst_limit_bytes,omitempty"`
	Drop           bool                   `json:"drop,omitempty"`
	Mirror         string                 `json:"mirror_port,omitempty"`
	Redirect       string                 `json:"redirect_port,omitempty"`
	Parameters     map[string]string      `json:"parameters,omitempty"`
}

// ActionType represents the type of QoS action
type ActionType string

const (
	ActionTypeQueue       ActionType = "queue"
	ActionTypeDSCP        ActionType = "dscp"
	ActionTypePriority    ActionType = "priority"
	ActionTypeRateLimit   ActionType = "rate_limit"
	ActionTypeDrop        ActionType = "drop"
	ActionTypeMirror      ActionType = "mirror"
	ActionTypeRedirect    ActionType = "redirect"
	ActionTypePolicing    ActionType = "policing"
)

// PolicyStatus represents the status of a QoS policy
type PolicyStatus string

const (
	PolicyStatusActive    PolicyStatus = "active"
	PolicyStatusInactive  PolicyStatus = "inactive"
	PolicyStatusError     PolicyStatus = "error"
	PolicyStatusPending   PolicyStatus = "pending"
)

// PolicyStatistics represents statistics for a QoS policy
type PolicyStatistics struct {
	TotalPackets      int64     `json:"total_packets"`
	TotalBytes        int64     `json:"total_bytes"`
	DroppedPackets    int64     `json:"dropped_packets"`
	DroppedBytes      int64     `json:"dropped_bytes"`
	QueuedPackets     int64     `json:"queued_packets"`
	QueuedBytes       int64     `json:"queued_bytes"`
	AvgLatency        float64   `json:"avg_latency_ms"`
	MaxLatency        float64   `json:"max_latency_ms"`
	LastUpdated       time.Time `json:"last_updated"`
}

// QueueStatistics represents statistics for a traffic queue
type QueueStatistics struct {
	TxPackets       int64     `json:"tx_packets"`
	TxBytes         int64     `json:"tx_bytes"`
	TxErrors        int64     `json:"tx_errors"`
	TxDropped       int64     `json:"tx_dropped"`
	BacklogBytes    int64     `json:"backlog_bytes"`
	BacklogPackets  int64     `json:"backlog_packets"`
	Requeues        int64     `json:"requeues"`
	Overlimits      int64     `json:"overlimits"`
	QueueLength     int       `json:"queue_length"`
	AvgQueueLength  float64   `json:"avg_queue_length"`
	LastUpdated     time.Time `json:"last_updated"`
}

// QueueConfiguration represents the system-level queue configuration
type QueueConfiguration struct {
	ID           string            `json:"id"`
	BridgeName   string            `json:"bridge_name"`
	QoSType      QoSType           `json:"qos_type"`
	Queues       map[int]*QueueConfig `json:"queues"`
	DefaultQueue int               `json:"default_queue"`
	Options      map[string]string `json:"options,omitempty"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
}

// QoSRuleConfig represents a complete QoS rule configuration
type QoSRuleConfig struct {
	ID           string                 `json:"id"`
	PolicyID     string                 `json:"policy_id"`
	BridgeName   string                 `json:"bridge_name"`
	Rule         *TrafficRule           `json:"rule"`
	FlowRuleID   string                 `json:"flow_rule_id"`
	Status       string                 `json:"status"`
	Statistics   RuleStatistics         `json:"statistics"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
}

// RuleStatistics represents statistics for a QoS rule
type RuleStatistics struct {
	MatchCount      int64     `json:"match_count"`
	ByteCount       int64     `json:"byte_count"`
	PacketCount     int64     `json:"packet_count"`
	LastMatch       time.Time `json:"last_match"`
	MatchRate       float64   `json:"match_rate_per_sec"`
	ByteRate        float64   `json:"byte_rate_bps"`
}

// NewQoSManager creates a new QoS manager
func NewQoSManager(bridgeManager *BridgeManager, config QoSManagerConfig) *QoSManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &QoSManager{
		bridgeManager: bridgeManager,
		qosPolicies:   make(map[string]*QoSPolicyConfig),
		qosRules:      make(map[string]*QoSRuleConfig),
		queueConfigs:  make(map[string]*QueueConfiguration),
		config:        config,
		ctx:           ctx,
		cancel:        cancel,
	}
}

// Start starts the QoS manager
func (qm *QoSManager) Start() error {
	log.Println("Starting OVS QoS Manager")
	
	// Load existing QoS configurations
	if err := qm.loadExistingQoSConfigurations(); err != nil {
		log.Printf("Warning: Failed to load existing QoS configurations: %v", err)
	}
	
	// Start monitoring loop
	go qm.monitoringLoop()
	
	log.Println("OVS QoS Manager started successfully")
	return nil
}

// Stop stops the QoS manager
func (qm *QoSManager) Stop() error {
	log.Println("Stopping OVS QoS Manager")
	qm.cancel()
	return nil
}

// CreateQoSPolicy creates a new QoS policy
func (qm *QoSManager) CreateQoSPolicy(ctx context.Context, policy *QoSPolicyConfig) error {
	if policy.ID == "" {
		policy.ID = uuid.New().String()
	}
	
	qm.policiesMutex.Lock()
	defer qm.policiesMutex.Unlock()
	
	// Check if policy already exists
	if _, exists := qm.qosPolicies[policy.ID]; exists {
		return fmt.Errorf("QoS policy %s already exists", policy.ID)
	}
	
	// Validate policy
	if err := qm.validatePolicy(policy); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}
	
	// Set timestamps
	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()
	policy.Status = PolicyStatusPending
	
	// Create queue configuration for the bridge
	if err := qm.createQueueConfiguration(policy); err != nil {
		return fmt.Errorf("failed to create queue configuration: %w", err)
	}
	
	// Apply policy to OVS
	if err := qm.applyPolicyToOVS(policy); err != nil {
		return fmt.Errorf("failed to apply policy to OVS: %w", err)
	}
	
	// Store policy
	qm.qosPolicies[policy.ID] = policy
	policy.Status = PolicyStatusActive
	policy.LastAppliedAt = time.Now()
	
	log.Printf("Created QoS policy: %s for bridge %s", policy.Name, policy.BridgeName)
	return nil
}

// validatePolicy validates a QoS policy configuration
func (qm *QoSManager) validatePolicy(policy *QoSPolicyConfig) error {
	if policy.Name == "" {
		return fmt.Errorf("policy name cannot be empty")
	}
	
	if policy.BridgeName == "" {
		return fmt.Errorf("bridge name cannot be empty")
	}
	
	// Check if bridge exists
	_, err := qm.bridgeManager.GetBridge(policy.BridgeName)
	if err != nil {
		return fmt.Errorf("bridge %s not found: %w", policy.BridgeName, err)
	}
	
	// Validate queue configurations
	if len(policy.Queues) == 0 {
		return fmt.Errorf("policy must have at least one queue")
	}
	
	if len(policy.Queues) > qm.config.MaxQueues {
		return fmt.Errorf("policy has too many queues (max: %d)", qm.config.MaxQueues)
	}
	
	// Validate default queue
	if _, exists := policy.Queues[policy.DefaultQueueID]; !exists {
		return fmt.Errorf("default queue %d not found in policy", policy.DefaultQueueID)
	}
	
	// Validate individual queues
	for id, queue := range policy.Queues {
		if id != queue.ID {
			return fmt.Errorf("queue ID mismatch: %d != %d", id, queue.ID)
		}
		
		if queue.MinRate <= 0 {
			return fmt.Errorf("queue %d min rate must be positive", id)
		}
		
		if queue.MaxRate <= queue.MinRate {
			return fmt.Errorf("queue %d max rate must be greater than min rate", id)
		}
		
		if queue.BurstSize <= 0 {
			queue.BurstSize = queue.MaxRate / 8 // Default to 1 second of traffic
		}
	}
	
	// Validate rules
	for i, rule := range policy.Rules {
		if rule.ID == "" {
			rule.ID = uuid.New().String()
		}
		
		if _, exists := policy.Queues[rule.QueueID]; !exists {
			return fmt.Errorf("rule %d references non-existent queue %d", i, rule.QueueID)
		}
	}
	
	return nil
}

// createQueueConfiguration creates queue configuration for a bridge
func (qm *QoSManager) createQueueConfiguration(policy *QoSPolicyConfig) error {
	qm.queuesMutex.Lock()
	defer qm.queuesMutex.Unlock()
	
	configID := fmt.Sprintf("%s-%s", policy.BridgeName, policy.ID)
	
	queueConfig := &QueueConfiguration{
		ID:           configID,
		BridgeName:   policy.BridgeName,
		QoSType:      policy.Type,
		Queues:       policy.Queues,
		DefaultQueue: policy.DefaultQueueID,
		Options:      policy.Options,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}
	
	qm.queueConfigs[configID] = queueConfig
	return nil
}

// applyPolicyToOVS applies a QoS policy to Open vSwitch
func (qm *QoSManager) applyPolicyToOVS(policy *QoSPolicyConfig) error {
	bridgeName := policy.BridgeName
	
	// Create QoS record
	qosRecord := fmt.Sprintf("type=%s", policy.Type)
	
	// Add queue configurations to QoS record
	var queueSpecs []string
	for _, queue := range policy.Queues {
		queueSpec := fmt.Sprintf("%d=@q%d", queue.ID, queue.ID)
		queueSpecs = append(queueSpecs, queueSpec)
		
		// Create queue arguments
		queueArgs := []string{
			"--", "--id=@q" + strconv.Itoa(queue.ID),
			"create", "queue",
		}
		
		// Add queue-specific options based on QoS type
		switch policy.Type {
		case QoSTypeHTB:
			if queue.MinRate > 0 {
				queueArgs = append(queueArgs, fmt.Sprintf("other-config:min-rate=%d", queue.MinRate))
			}
			if queue.MaxRate > 0 {
				queueArgs = append(queueArgs, fmt.Sprintf("other-config:max-rate=%d", queue.MaxRate))
			}
			if queue.BurstSize > 0 {
				queueArgs = append(queueArgs, fmt.Sprintf("other-config:burst=%d", queue.BurstSize))
			}
			if queue.Priority > 0 {
				queueArgs = append(queueArgs, fmt.Sprintf("other-config:priority=%d", queue.Priority))
			}
		case QoSTypeFQ:
			if queue.Quantum > 0 {
				queueArgs = append(queueArgs, fmt.Sprintf("other-config:quantum=%d", queue.Quantum))
			}
		}
		
		// Add custom options
		for key, value := range queue.Options {
			queueArgs = append(queueArgs, fmt.Sprintf("other-config:%s=%s", key, value))
		}
	}
	
	if len(queueSpecs) > 0 {
		qosRecord += " queues=" + strings.Join(queueSpecs, ",")
	}
	
	// Apply QoS to all ports on the bridge
	bridge, err := qm.bridgeManager.GetBridge(bridgeName)
	if err != nil {
		return err
	}
	
	for _, port := range bridge.Ports {
		// Set QoS on port
		cmd := exec.Command("ovs-vsctl", "set", "port", port.Name, fmt.Sprintf("qos=@newqos"))
		args := append(cmd.Args, "--", "--id=@newqos", "create", "qos", qosRecord)
		
		// Add queue creation commands
		for _, queue := range policy.Queues {
			queueArgs := []string{
				"--", "--id=@q" + strconv.Itoa(queue.ID),
				"create", "queue",
			}
			
			// Add queue configuration
			switch policy.Type {
			case QoSTypeHTB:
				if queue.MinRate > 0 {
					queueArgs = append(queueArgs, fmt.Sprintf("other-config:min-rate=%d", queue.MinRate))
				}
				if queue.MaxRate > 0 {
					queueArgs = append(queueArgs, fmt.Sprintf("other-config:max-rate=%d", queue.MaxRate))
				}
				if queue.BurstSize > 0 {
					queueArgs = append(queueArgs, fmt.Sprintf("other-config:burst=%d", queue.BurstSize))
				}
			}
			
			args = append(args, queueArgs...)
		}
		
		cmd.Args = args
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to set QoS on port %s: %w", port.Name, err)
		}
	}
	
	// Create flow rules for traffic classification
	for _, rule := range policy.Rules {
		if err := qm.createFlowRuleForTrafficRule(bridgeName, rule); err != nil {
			return fmt.Errorf("failed to create flow rule for traffic rule %s: %w", rule.ID, err)
		}
	}
	
	return nil
}

// createFlowRuleForTrafficRule creates an OpenFlow rule for a traffic classification rule
func (qm *QoSManager) createFlowRuleForTrafficRule(bridgeName string, rule *TrafficRule) error {
	// Convert traffic rule to flow rule
	flowRule := FlowRule{
		ID:       rule.ID,
		Priority: rule.Priority,
		Table:    0,
		Cookie:   uint64(rule.Priority), // Use priority as cookie for identification
		Match:    qm.trafficMatchToFlowMatch(rule.Match),
		Actions:  qm.trafficActionToFlowActions(rule.Action, rule.QueueID),
		Metadata: map[string]string{
			"rule_id":    rule.ID,
			"rule_name":  rule.Name,
			"policy_id":  rule.ID, // This would be set by the caller
		},
		CreatedAt: time.Now(),
	}
	
	// Add the flow rule through the bridge manager
	if err := qm.bridgeManager.AddFlowRule(context.Background(), bridgeName, flowRule); err != nil {
		return err
	}
	
	// Store rule configuration
	qm.rulesMutex.Lock()
	qm.qosRules[rule.ID] = &QoSRuleConfig{
		ID:         rule.ID,
		BridgeName: bridgeName,
		Rule:       rule,
		FlowRuleID: flowRule.ID,
		Status:     "active",
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}
	qm.rulesMutex.Unlock()
	
	return nil
}

// trafficMatchToFlowMatch converts TrafficMatch to FlowMatch
func (qm *QoSManager) trafficMatchToFlowMatch(match TrafficMatch) FlowMatch {
	flowMatch := FlowMatch{}
	
	if match.InPort != "" {
		flowMatch.InPort = match.InPort
	}
	if match.EtherType != "" {
		flowMatch.EthType = match.EtherType
	}
	if match.SrcMAC != "" {
		flowMatch.EthSrc = match.SrcMAC
	}
	if match.DstMAC != "" {
		flowMatch.EthDst = match.DstMAC
	}
	if match.SrcIP != "" {
		flowMatch.IPSrc = match.SrcIP
	}
	if match.DstIP != "" {
		flowMatch.IPDst = match.DstIP
	}
	if match.Protocol == "tcp" {
		flowMatch.IPProto = 6
	} else if match.Protocol == "udp" {
		flowMatch.IPProto = 17
	} else if match.Protocol == "icmp" {
		flowMatch.IPProto = 1
	}
	if match.SrcPort > 0 {
		flowMatch.TCPSrc = match.SrcPort
	}
	if match.DstPort > 0 {
		flowMatch.TCPDst = match.DstPort
	}
	if match.VlanID > 0 {
		flowMatch.VlanID = match.VlanID
	}
	
	return flowMatch
}

// trafficActionToFlowActions converts TrafficAction to FlowActions
func (qm *QoSManager) trafficActionToFlowActions(action TrafficAction, queueID int) []FlowAction {
	var actions []FlowAction
	
	// Set queue action
	if action.Type == ActionTypeQueue || queueID > 0 {
		actions = append(actions, FlowAction{
			Type: "set_queue",
			Params: map[string]string{
				"queue_id": strconv.Itoa(queueID),
			},
		})
	}
	
	// DSCP marking
	if action.DSCP > 0 && qm.config.EnableDSCPMarking {
		actions = append(actions, FlowAction{
			Type: "set_field",
			Params: map[string]string{
				"field": "ip_dscp",
				"value": strconv.Itoa(action.DSCP),
			},
		})
	}
	
	// Drop action
	if action.Drop {
		actions = append(actions, FlowAction{
			Type: "drop",
		})
		return actions // Don't add output action if dropping
	}
	
	// Mirror action
	if action.Mirror != "" {
		actions = append(actions, FlowAction{
			Type: "output",
			Params: map[string]string{
				"port": action.Mirror,
			},
		})
	}
	
	// Redirect action
	if action.Redirect != "" {
		actions = append(actions, FlowAction{
			Type: "output",
			Params: map[string]string{
				"port": action.Redirect,
			},
		})
	} else {
		// Normal forwarding
		actions = append(actions, FlowAction{
			Type: "output",
			Params: map[string]string{
				"port": "normal",
			},
		})
	}
	
	return actions
}

// UpdateQoSPolicy updates an existing QoS policy
func (qm *QoSManager) UpdateQoSPolicy(ctx context.Context, policyID string, policy *QoSPolicyConfig) error {
	qm.policiesMutex.Lock()
	defer qm.policiesMutex.Unlock()
	
	// Check if policy exists
	existingPolicy, exists := qm.qosPolicies[policyID]
	if !exists {
		return fmt.Errorf("QoS policy %s not found", policyID)
	}
	
	// Validate updated policy
	policy.ID = policyID
	if err := qm.validatePolicy(policy); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}
	
	// Remove existing policy from OVS
	if err := qm.removePolicyFromOVS(existingPolicy); err != nil {
		log.Printf("Warning: Failed to remove existing policy from OVS: %v", err)
	}
	
	// Apply updated policy to OVS
	if err := qm.applyPolicyToOVS(policy); err != nil {
		// Try to restore original policy
		qm.applyPolicyToOVS(existingPolicy)
		return fmt.Errorf("failed to apply updated policy: %w", err)
	}
	
	// Update policy
	policy.CreatedAt = existingPolicy.CreatedAt
	policy.UpdatedAt = time.Now()
	policy.LastAppliedAt = time.Now()
	policy.Status = PolicyStatusActive
	
	qm.qosPolicies[policyID] = policy
	
	log.Printf("Updated QoS policy: %s", policy.Name)
	return nil
}

// DeleteQoSPolicy deletes a QoS policy
func (qm *QoSManager) DeleteQoSPolicy(ctx context.Context, policyID string) error {
	qm.policiesMutex.Lock()
	defer qm.policiesMutex.Unlock()
	
	// Check if policy exists
	policy, exists := qm.qosPolicies[policyID]
	if !exists {
		return fmt.Errorf("QoS policy %s not found", policyID)
	}
	
	// Remove policy from OVS
	if err := qm.removePolicyFromOVS(policy); err != nil {
		return fmt.Errorf("failed to remove policy from OVS: %w", err)
	}
	
	// Remove from our maps
	delete(qm.qosPolicies, policyID)
	
	// Remove associated queue configuration
	qm.queuesMutex.Lock()
	configID := fmt.Sprintf("%s-%s", policy.BridgeName, policy.ID)
	delete(qm.queueConfigs, configID)
	qm.queuesMutex.Unlock()
	
	// Remove associated rules
	qm.rulesMutex.Lock()
	for ruleID, rule := range qm.qosRules {
		if rule.PolicyID == policyID {
			delete(qm.qosRules, ruleID)
		}
	}
	qm.rulesMutex.Unlock()
	
	log.Printf("Deleted QoS policy: %s", policy.Name)
	return nil
}

// removePolicyFromOVS removes a QoS policy from Open vSwitch
func (qm *QoSManager) removePolicyFromOVS(policy *QoSPolicyConfig) error {
	bridgeName := policy.BridgeName
	
	// Remove flow rules
	for _, rule := range policy.Rules {
		if err := qm.bridgeManager.DeleteFlowRule(context.Background(), bridgeName, rule.ID); err != nil {
			log.Printf("Warning: Failed to delete flow rule %s: %v", rule.ID, err)
		}
	}
	
	// Remove QoS from ports
	bridge, err := qm.bridgeManager.GetBridge(bridgeName)
	if err != nil {
		return err
	}
	
	for _, port := range bridge.Ports {
		cmd := exec.Command("ovs-vsctl", "clear", "port", port.Name, "qos")
		if err := cmd.Run(); err != nil {
			log.Printf("Warning: Failed to clear QoS from port %s: %v", port.Name, err)
		}
	}
	
	return nil
}

// GetQoSPolicy returns a QoS policy by ID
func (qm *QoSManager) GetQoSPolicy(policyID string) (*QoSPolicyConfig, error) {
	qm.policiesMutex.RLock()
	defer qm.policiesMutex.RUnlock()
	
	policy, exists := qm.qosPolicies[policyID]
	if !exists {
		return nil, fmt.Errorf("QoS policy %s not found", policyID)
	}
	
	return policy, nil
}

// ListQoSPolicies returns all QoS policies
func (qm *QoSManager) ListQoSPolicies() []*QoSPolicyConfig {
	qm.policiesMutex.RLock()
	defer qm.policiesMutex.RUnlock()
	
	policies := make([]*QoSPolicyConfig, 0, len(qm.qosPolicies))
	for _, policy := range qm.qosPolicies {
		policies = append(policies, policy)
	}
	
	return policies
}

// GetQoSStatistics returns statistics for a QoS policy
func (qm *QoSManager) GetQoSStatistics(policyID string) (*PolicyStatistics, error) {
	qm.policiesMutex.RLock()
	policy, exists := qm.qosPolicies[policyID]
	qm.policiesMutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("QoS policy %s not found", policyID)
	}
	
	// Collect statistics from OVS
	stats, err := qm.collectPolicyStatistics(policy)
	if err != nil {
		return nil, fmt.Errorf("failed to collect statistics: %w", err)
	}
	
	return stats, nil
}

// collectPolicyStatistics collects statistics for a QoS policy from OVS
func (qm *QoSManager) collectPolicyStatistics(policy *QoSPolicyConfig) (*PolicyStatistics, error) {
	stats := &PolicyStatistics{
		LastUpdated: time.Now(),
	}
	
	// Collect flow rule statistics
	for _, rule := range policy.Rules {
		ruleStats, err := qm.collectRuleStatistics(policy.BridgeName, rule.ID)
		if err != nil {
			log.Printf("Warning: Failed to collect statistics for rule %s: %v", rule.ID, err)
			continue
		}
		
		stats.TotalPackets += ruleStats.PacketCount
		stats.TotalBytes += ruleStats.ByteCount
	}
	
	// Collect queue statistics
	for _, queue := range policy.Queues {
		queueStats, err := qm.collectQueueStatistics(policy.BridgeName, queue.ID)
		if err != nil {
			log.Printf("Warning: Failed to collect statistics for queue %d: %v", queue.ID, err)
			continue
		}
		
		stats.QueuedPackets += queueStats.BacklogPackets
		stats.QueuedBytes += queueStats.BacklogBytes
		stats.DroppedPackets += queueStats.TxDropped
		stats.DroppedBytes += queueStats.TxDropped * 1500 // Estimate
	}
	
	return stats, nil
}

// collectRuleStatistics collects statistics for a specific rule
func (qm *QoSManager) collectRuleStatistics(bridgeName, ruleID string) (*RuleStatistics, error) {
	// This would parse ovs-ofctl dump-flows output to get rule statistics
	// For now, return mock data
	return &RuleStatistics{
		MatchCount:  0,
		ByteCount:   0,
		PacketCount: 0,
		LastMatch:   time.Now(),
	}, nil
}

// collectQueueStatistics collects statistics for a specific queue
func (qm *QoSManager) collectQueueStatistics(bridgeName string, queueID int) (*QueueStatistics, error) {
	// This would parse ovs-vsctl list queue output to get queue statistics
	// For now, return mock data
	return &QueueStatistics{
		TxPackets:      0,
		TxBytes:        0,
		TxErrors:       0,
		TxDropped:      0,
		BacklogBytes:   0,
		BacklogPackets: 0,
		LastUpdated:    time.Now(),
	}, nil
}

// loadExistingQoSConfigurations loads existing QoS configurations from OVS
func (qm *QoSManager) loadExistingQoSConfigurations() error {
	// This would parse existing OVS QoS configurations
	// For now, just log that this would be implemented
	log.Println("Loading existing QoS configurations (not implemented)")
	return nil
}

// monitoringLoop periodically updates QoS statistics
func (qm *QoSManager) monitoringLoop() {
	ticker := time.NewTicker(qm.config.MonitoringInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-qm.ctx.Done():
			return
		case <-ticker.C:
			qm.updateAllStatistics()
		}
	}
}

// updateAllStatistics updates statistics for all QoS policies
func (qm *QoSManager) updateAllStatistics() {
	qm.policiesMutex.RLock()
	policies := make([]*QoSPolicyConfig, 0, len(qm.qosPolicies))
	for _, policy := range qm.qosPolicies {
		policies = append(policies, policy)
	}
	qm.policiesMutex.RUnlock()
	
	for _, policy := range policies {
		stats, err := qm.collectPolicyStatistics(policy)
		if err != nil {
			log.Printf("Warning: Failed to update statistics for policy %s: %v", policy.ID, err)
			continue
		}
		
		qm.policiesMutex.Lock()
		if currentPolicy, exists := qm.qosPolicies[policy.ID]; exists {
			currentPolicy.Statistics = *stats
		}
		qm.policiesMutex.Unlock()
	}
}