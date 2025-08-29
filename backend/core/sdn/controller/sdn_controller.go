package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/network"
	"github.com/google/uuid"
)

// Intent represents a high-level network intention
type Intent struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Priority    int                    `json:"priority"`
	Constraints []Constraint           `json:"constraints"`
	Goals       []Goal                 `json:"goals"`
	Scope       IntentScope            `json:"scope"`
	Status      IntentStatus           `json:"status"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// Constraint defines network constraints for intent
type Constraint struct {
	Type      ConstraintType         `json:"type"`
	Params    map[string]interface{} `json:"params"`
	Mandatory bool                   `json:"mandatory"`
}

// Goal defines network goals for intent
type Goal struct {
	Type     GoalType               `json:"type"`
	Target   float64                `json:"target"`
	Operator GoalOperator           `json:"operator"`
	Params   map[string]interface{} `json:"params,omitempty"`
}

// IntentScope defines the scope of network intent
type IntentScope struct {
	Nodes     []string `json:"nodes,omitempty"`
	Networks  []string `json:"networks,omitempty"`
	Services  []string `json:"services,omitempty"`
	Global    bool     `json:"global"`
}

// Enums for intent system
type IntentStatus string
type ConstraintType string
type GoalType string
type GoalOperator string

const (
	IntentStatusPending    IntentStatus = "pending"
	IntentStatusActive     IntentStatus = "active"
	IntentStatusFailed     IntentStatus = "failed"
	IntentStatusCompleted  IntentStatus = "completed"

	ConstraintTypeLatency     ConstraintType = "latency"
	ConstraintTypeBandwidth   ConstraintType = "bandwidth"
	ConstraintTypeAvailability ConstraintType = "availability"
	ConstraintTypeSecurity    ConstraintType = "security"
	ConstraintTypeLocation    ConstraintType = "location"

	GoalTypeMinimize       GoalType = "minimize"
	GoalTypeMaximize       GoalType = "maximize"
	GoalTypeMaintain       GoalType = "maintain"
	GoalTypeBalance        GoalType = "balance"

	GoalOperatorLessThan    GoalOperator = "lt"
	GoalOperatorGreaterThan GoalOperator = "gt"
	GoalOperatorEqual       GoalOperator = "eq"
	GoalOperatorBetween     GoalOperator = "between"
)

// FlowRule represents an OpenFlow-style flow rule
type FlowRule struct {
	ID          string            `json:"id"`
	Priority    int               `json:"priority"`
	Match       FlowMatch         `json:"match"`
	Actions     []FlowAction      `json:"actions"`
	TableID     int               `json:"table_id"`
	IdleTimeout int               `json:"idle_timeout"`
	HardTimeout int               `json:"hard_timeout"`
	Cookie      uint64            `json:"cookie"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// FlowMatch represents flow matching criteria
type FlowMatch struct {
	InPort      string `json:"in_port,omitempty"`
	EthSrc      string `json:"eth_src,omitempty"`
	EthDst      string `json:"eth_dst,omitempty"`
	EthType     string `json:"eth_type,omitempty"`
	VlanID      int    `json:"vlan_id,omitempty"`
	IPSrc       string `json:"ip_src,omitempty"`
	IPDst       string `json:"ip_dst,omitempty"`
	IPProto     int    `json:"ip_proto,omitempty"`
	TCPSrc      int    `json:"tcp_src,omitempty"`
	TCPDst      int    `json:"tcp_dst,omitempty"`
	UDPSrc      int    `json:"udp_src,omitempty"`
	UDPDst      int    `json:"udp_dst,omitempty"`
}

// FlowAction represents actions to be taken on matched flows
type FlowAction struct {
	Type   ActionType            `json:"type"`
	Params map[string]interface{} `json:"params,omitempty"`
}

type ActionType string

const (
	ActionOutput          ActionType = "output"
	ActionDrop            ActionType = "drop"
	ActionSetField        ActionType = "set_field"
	ActionPushVlan        ActionType = "push_vlan"
	ActionPopVlan         ActionType = "pop_vlan"
	ActionSetQueue        ActionType = "set_queue"
	ActionGroup           ActionType = "group"
	ActionController      ActionType = "controller"
)

// NetworkSlice represents a network slice with QoS guarantees
type NetworkSlice struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description"`
	Type          SliceType              `json:"type"`
	QoSProfile    QoSProfile             `json:"qos_profile"`
	Resources     SliceResources         `json:"resources"`
	Endpoints     []SliceEndpoint        `json:"endpoints"`
	Policies      []SlicePolicy          `json:"policies"`
	Status        SliceStatus            `json:"status"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

type SliceType string
type SliceStatus string

const (
	SliceTypeUltraReliable SliceType = "ultra_reliable"
	SliceTypeHighThroughput SliceType = "high_throughput"
	SliceTypeLowLatency    SliceType = "low_latency"
	SliceTypeBestEffort    SliceType = "best_effort"

	SliceStatusActive     SliceStatus = "active"
	SliceStatusInactive   SliceStatus = "inactive"
	SliceStatusDeploying  SliceStatus = "deploying"
	SliceStatusError      SliceStatus = "error"
)

// QoSProfile defines quality of service requirements
type QoSProfile struct {
	MaxLatency      time.Duration `json:"max_latency"`
	MinBandwidth    int64         `json:"min_bandwidth"`
	MaxJitter       time.Duration `json:"max_jitter"`
	MaxPacketLoss   float64       `json:"max_packet_loss"`
	Availability    float64       `json:"availability"`
	Priority        int           `json:"priority"`
	DSCP            int           `json:"dscp,omitempty"`
}

// SliceResources defines allocated resources for a network slice
type SliceResources struct {
	BandwidthMbps     int64    `json:"bandwidth_mbps"`
	ComputeNodes      []string `json:"compute_nodes,omitempty"`
	StorageNodes      []string `json:"storage_nodes,omitempty"`
	NetworkFunctions  []string `json:"network_functions,omitempty"`
}

// SliceEndpoint represents an endpoint in a network slice
type SliceEndpoint struct {
	ID       string `json:"id"`
	NodeID   string `json:"node_id"`
	Type     string `json:"type"`
	Address  string `json:"address"`
	Port     int    `json:"port,omitempty"`
}

// SlicePolicy represents policies applied to a network slice
type SlicePolicy struct {
	ID      string                 `json:"id"`
	Type    PolicyType             `json:"type"`
	Rules   []PolicyRule           `json:"rules"`
	Params  map[string]interface{} `json:"params,omitempty"`
}

type PolicyType string

const (
	PolicyTypeFirewall      PolicyType = "firewall"
	PolicyTypeLoadBalancing PolicyType = "load_balancing"
	PolicyTypeTrafficShaping PolicyType = "traffic_shaping"
	PolicyTypeQoS           PolicyType = "qos"
	PolicyTypeSecurity      PolicyType = "security"
)

// PolicyRule represents a single policy rule
type PolicyRule struct {
	ID        string                 `json:"id"`
	Match     map[string]interface{} `json:"match"`
	Action    string                 `json:"action"`
	Params    map[string]interface{} `json:"params,omitempty"`
	Priority  int                    `json:"priority"`
}

// SDNController manages the software-defined network
type SDNController struct {
	// Core components
	intents       map[string]*Intent
	intentsMutex  sync.RWMutex
	
	flowRules     map[string]*FlowRule
	flowRulesMutex sync.RWMutex
	
	networkSlices map[string]*NetworkSlice
	slicesMutex   sync.RWMutex
	
	// Dependencies
	networkTopology *networkscheduler.NetworkTopology
	aiEngine        AIOptimizer
	
	// State management
	ctx    context.Context
	cancel context.CancelFunc
	
	// Configuration
	config SDNControllerConfig
	
	// Monitoring
	metrics *SDNMetrics
}

// SDNControllerConfig holds configuration for the SDN controller
type SDNControllerConfig struct {
	IntentEvaluationInterval time.Duration `json:"intent_evaluation_interval"`
	FlowRuleTimeout         time.Duration `json:"flow_rule_timeout"`
	MaxConcurrentOperations int           `json:"max_concurrent_operations"`
	EnableAIOptimization    bool          `json:"enable_ai_optimization"`
	EnableP4Support         bool          `json:"enable_p4_support"`
	ControllerPort          int           `json:"controller_port"`
}

// DefaultSDNControllerConfig returns default configuration
func DefaultSDNControllerConfig() SDNControllerConfig {
	return SDNControllerConfig{
		IntentEvaluationInterval: 30 * time.Second,
		FlowRuleTimeout:         300 * time.Second,
		MaxConcurrentOperations: 100,
		EnableAIOptimization:    true,
		EnableP4Support:         false,
		ControllerPort:          6653,
	}
}

// SDNMetrics tracks SDN controller performance
type SDNMetrics struct {
	ActiveIntents     int64     `json:"active_intents"`
	ActiveFlowRules   int64     `json:"active_flow_rules"`
	ActiveSlices      int64     `json:"active_slices"`
	IntentSuccessRate float64   `json:"intent_success_rate"`
	AvgLatency        float64   `json:"avg_latency_ms"`
	TotalThroughput   int64     `json:"total_throughput_mbps"`
	LastUpdated       time.Time `json:"last_updated"`
}

// AIOptimizer interface for AI-powered network optimization
type AIOptimizer interface {
	OptimizeIntent(intent *Intent, topology *networkscheduler.NetworkTopology) ([]FlowRule, error)
	PredictTrafficPatterns(nodes []string, timeWindow time.Duration) (map[string]float64, error)
	OptimizeSliceAllocation(slices []*NetworkSlice) (map[string]SliceResources, error)
}

// NewSDNController creates a new SDN controller
func NewSDNController(config SDNControllerConfig, topology *networkscheduler.NetworkTopology, aiEngine AIOptimizer) *SDNController {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &SDNController{
		intents:         make(map[string]*Intent),
		flowRules:       make(map[string]*FlowRule),
		networkSlices:   make(map[string]*NetworkSlice),
		networkTopology: topology,
		aiEngine:        aiEngine,
		config:          config,
		ctx:             ctx,
		cancel:          cancel,
		metrics:         &SDNMetrics{},
	}
}

// Start starts the SDN controller
func (c *SDNController) Start() error {
	log.Println("Starting SDN Controller")
	
	// Start intent evaluation loop
	go c.intentEvaluationLoop()
	
	// Start metrics collection
	go c.metricsCollectionLoop()
	
	// Start flow rule cleanup
	go c.flowRuleCleanupLoop()
	
	log.Printf("SDN Controller started on port %d", c.config.ControllerPort)
	return nil
}

// Stop stops the SDN controller
func (c *SDNController) Stop() error {
	log.Println("Stopping SDN Controller")
	c.cancel()
	return nil
}

// CreateIntent creates a new network intent
func (c *SDNController) CreateIntent(intent *Intent) error {
	if intent.ID == "" {
		intent.ID = uuid.New().String()
	}
	
	intent.Status = IntentStatusPending
	intent.CreatedAt = time.Now()
	intent.UpdatedAt = time.Now()
	
	c.intentsMutex.Lock()
	c.intents[intent.ID] = intent
	c.intentsMutex.Unlock()
	
	log.Printf("Created network intent: %s", intent.Name)
	return nil
}

// GetIntent retrieves a network intent
func (c *SDNController) GetIntent(intentID string) (*Intent, error) {
	c.intentsMutex.RLock()
	defer c.intentsMutex.RUnlock()
	
	intent, exists := c.intents[intentID]
	if !exists {
		return nil, fmt.Errorf("intent not found: %s", intentID)
	}
	
	return intent, nil
}

// CreateNetworkSlice creates a new network slice
func (c *SDNController) CreateNetworkSlice(slice *NetworkSlice) error {
	if slice.ID == "" {
		slice.ID = uuid.New().String()
	}
	
	slice.Status = SliceStatusDeploying
	slice.CreatedAt = time.Now()
	slice.UpdatedAt = time.Now()
	
	// Validate slice requirements
	if err := c.validateSliceRequirements(slice); err != nil {
		return fmt.Errorf("slice validation failed: %w", err)
	}
	
	// Allocate resources using AI optimization if enabled
	if c.config.EnableAIOptimization && c.aiEngine != nil {
		slices := []*NetworkSlice{slice}
		optimizedResources, err := c.aiEngine.OptimizeSliceAllocation(slices)
		if err != nil {
			log.Printf("AI optimization failed, using default allocation: %v", err)
		} else {
			if resources, exists := optimizedResources[slice.ID]; exists {
				slice.Resources = resources
			}
		}
	}
	
	c.slicesMutex.Lock()
	c.networkSlices[slice.ID] = slice
	c.slicesMutex.Unlock()
	
	// Deploy slice configuration
	go c.deploySlice(slice)
	
	log.Printf("Created network slice: %s", slice.Name)
	return nil
}

// deploySlice deploys a network slice
func (c *SDNController) deploySlice(slice *NetworkSlice) {
	// Generate flow rules for the slice
	flowRules, err := c.generateSliceFlowRules(slice)
	if err != nil {
		log.Printf("Failed to generate flow rules for slice %s: %v", slice.ID, err)
		c.updateSliceStatus(slice.ID, SliceStatusError)
		return
	}
	
	// Install flow rules
	for _, rule := range flowRules {
		if err := c.installFlowRule(rule); err != nil {
			log.Printf("Failed to install flow rule for slice %s: %v", slice.ID, err)
			c.updateSliceStatus(slice.ID, SliceStatusError)
			return
		}
	}
	
	// Update slice status
	c.updateSliceStatus(slice.ID, SliceStatusActive)
	log.Printf("Successfully deployed network slice: %s", slice.Name)
}

// generateSliceFlowRules generates flow rules for a network slice
func (c *SDNController) generateSliceFlowRules(slice *NetworkSlice) ([]*FlowRule, error) {
	var rules []*FlowRule
	
	// Generate rules based on slice type and QoS requirements
	switch slice.Type {
	case SliceTypeLowLatency:
		rules = append(rules, c.generateLowLatencyRules(slice)...)
	case SliceTypeHighThroughput:
		rules = append(rules, c.generateHighThroughputRules(slice)...)
	case SliceTypeUltraReliable:
		rules = append(rules, c.generateUltraReliableRules(slice)...)
	case SliceTypeBestEffort:
		rules = append(rules, c.generateBestEffortRules(slice)...)
	}
	
	// Add QoS rules
	qosRules := c.generateQoSRules(slice)
	rules = append(rules, qosRules...)
	
	return rules, nil
}

// generateLowLatencyRules generates rules optimized for low latency
func (c *SDNController) generateLowLatencyRules(slice *NetworkSlice) []*FlowRule {
	var rules []*FlowRule
	
	// High priority rules for slice traffic
	rule := &FlowRule{
		ID:          uuid.New().String(),
		Priority:    9000, // High priority
		TableID:     0,
		IdleTimeout: 30,
		HardTimeout: 300,
		Actions: []FlowAction{
			{
				Type: ActionSetQueue,
				Params: map[string]interface{}{
					"queue_id": 0, // Highest priority queue
				},
			},
			{
				Type: ActionOutput,
				Params: map[string]interface{}{
					"port": "normal",
				},
			},
		},
		Metadata: map[string]string{
			"slice_id":   slice.ID,
			"slice_type": string(slice.Type),
		},
	}
	
	rules = append(rules, rule)
	return rules
}

// generateHighThroughputRules generates rules optimized for high throughput
func (c *SDNController) generateHighThroughputRules(slice *NetworkSlice) []*FlowRule {
	var rules []*FlowRule
	
	// Configure for maximum throughput
	rule := &FlowRule{
		ID:          uuid.New().String(),
		Priority:    8000,
		TableID:     0,
		IdleTimeout: 60,
		HardTimeout: 600,
		Actions: []FlowAction{
			{
				Type: ActionSetQueue,
				Params: map[string]interface{}{
					"queue_id": 1, // High throughput queue
				},
			},
			{
				Type: ActionOutput,
				Params: map[string]interface{}{
					"port": "normal",
				},
			},
		},
		Metadata: map[string]string{
			"slice_id":   slice.ID,
			"slice_type": string(slice.Type),
		},
	}
	
	rules = append(rules, rule)
	return rules
}

// generateUltraReliableRules generates rules for ultra-reliable communication
func (c *SDNController) generateUltraReliableRules(slice *NetworkSlice) []*FlowRule {
	var rules []*FlowRule
	
	// Multi-path rules for reliability
	for i, endpoint := range slice.Endpoints {
		rule := &FlowRule{
			ID:          uuid.New().String(),
			Priority:    8500,
			TableID:     0,
			IdleTimeout: 0, // No idle timeout for reliability
			HardTimeout: 0, // No hard timeout
			Actions: []FlowAction{
				{
					Type: ActionGroup,
					Params: map[string]interface{}{
						"group_id": fmt.Sprintf("reliable_group_%d", i),
					},
				},
			},
			Metadata: map[string]string{
				"slice_id":     slice.ID,
				"slice_type":   string(slice.Type),
				"endpoint_id":  endpoint.ID,
			},
		}
		rules = append(rules, rule)
	}
	
	return rules
}

// generateBestEffortRules generates rules for best effort service
func (c *SDNController) generateBestEffortRules(slice *NetworkSlice) []*FlowRule {
	var rules []*FlowRule
	
	// Lower priority rules
	rule := &FlowRule{
		ID:          uuid.New().String(),
		Priority:    1000, // Low priority
		TableID:     0,
		IdleTimeout: 300,
		HardTimeout: 3600,
		Actions: []FlowAction{
			{
				Type: ActionSetQueue,
				Params: map[string]interface{}{
					"queue_id": 3, // Best effort queue
				},
			},
			{
				Type: ActionOutput,
				Params: map[string]interface{}{
					"port": "normal",
				},
			},
		},
		Metadata: map[string]string{
			"slice_id":   slice.ID,
			"slice_type": string(slice.Type),
		},
	}
	
	rules = append(rules, rule)
	return rules
}

// generateQoSRules generates QoS enforcement rules
func (c *SDNController) generateQoSRules(slice *NetworkSlice) []*FlowRule {
	var rules []*FlowRule
	
	// DSCP marking rule if specified
	if slice.QoSProfile.DSCP > 0 {
		rule := &FlowRule{
			ID:       uuid.New().String(),
			Priority: 7000,
			TableID:  0,
			Actions: []FlowAction{
				{
					Type: ActionSetField,
					Params: map[string]interface{}{
						"field": "ip_dscp",
						"value": slice.QoSProfile.DSCP,
					},
				},
				{
					Type: ActionOutput,
					Params: map[string]interface{}{
						"port": "normal",
					},
				},
			},
			Metadata: map[string]string{
				"slice_id": slice.ID,
				"type":     "qos_dscp",
			},
		}
		rules = append(rules, rule)
	}
	
	return rules
}

// installFlowRule installs a flow rule on the network
func (c *SDNController) installFlowRule(rule *FlowRule) error {
	c.flowRulesMutex.Lock()
	defer c.flowRulesMutex.Unlock()
	
	c.flowRules[rule.ID] = rule
	
	// In a real implementation, this would send the rule to OpenFlow switches
	log.Printf("Installed flow rule: %s (priority: %d)", rule.ID, rule.Priority)
	
	return nil
}

// intentEvaluationLoop periodically evaluates and processes intents
func (c *SDNController) intentEvaluationLoop() {
	ticker := time.NewTicker(c.config.IntentEvaluationInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.evaluateIntents()
		}
	}
}

// evaluateIntents evaluates all pending intents
func (c *SDNController) evaluateIntents() {
	c.intentsMutex.RLock()
	pendingIntents := make([]*Intent, 0)
	for _, intent := range c.intents {
		if intent.Status == IntentStatusPending {
			pendingIntents = append(pendingIntents, intent)
		}
	}
	c.intentsMutex.RUnlock()
	
	for _, intent := range pendingIntents {
		go c.processIntent(intent)
	}
}

// processIntent processes a single intent
func (c *SDNController) processIntent(intent *Intent) {
	// Generate flow rules from intent using AI if available
	var flowRules []FlowRule
	var err error
	
	if c.config.EnableAIOptimization && c.aiEngine != nil {
		flowRules, err = c.aiEngine.OptimizeIntent(intent, c.networkTopology)
	} else {
		flowRules, err = c.generateIntentFlowRules(intent)
	}
	
	if err != nil {
		log.Printf("Failed to generate flow rules for intent %s: %v", intent.ID, err)
		c.updateIntentStatus(intent.ID, IntentStatusFailed)
		return
	}
	
	// Install flow rules
	for _, rule := range flowRules {
		if err := c.installFlowRule(&rule); err != nil {
			log.Printf("Failed to install flow rule for intent %s: %v", intent.ID, err)
			c.updateIntentStatus(intent.ID, IntentStatusFailed)
			return
		}
	}
	
	// Update intent status
	c.updateIntentStatus(intent.ID, IntentStatusActive)
	log.Printf("Successfully processed intent: %s", intent.Name)
}

// generateIntentFlowRules generates flow rules from an intent (non-AI version)
func (c *SDNController) generateIntentFlowRules(intent *Intent) ([]FlowRule, error) {
	var rules []FlowRule
	
	// Basic rule generation based on intent constraints and goals
	for _, goal := range intent.Goals {
		switch goal.Type {
		case GoalTypeMinimize:
			if goal.Operator == GoalOperatorLessThan {
				// Generate rules to minimize the specified metric
				rule := FlowRule{
					ID:       uuid.New().String(),
					Priority: 5000,
					TableID:  0,
					Actions: []FlowAction{
						{
							Type: ActionSetQueue,
							Params: map[string]interface{}{
								"queue_id": 0,
							},
						},
						{
							Type: ActionOutput,
							Params: map[string]interface{}{
								"port": "normal",
							},
						},
					},
				}
				rules = append(rules, rule)
			}
		case GoalTypeMaximize:
			// Generate rules to maximize the specified metric
			rule := FlowRule{
				ID:       uuid.New().String(),
				Priority: 4000,
				TableID:  0,
				Actions: []FlowAction{
					{
						Type: ActionOutput,
						Params: map[string]interface{}{
							"port": "flood",
						},
					},
				},
			}
			rules = append(rules, rule)
		}
	}
	
	return rules, nil
}

// updateIntentStatus updates the status of an intent
func (c *SDNController) updateIntentStatus(intentID string, status IntentStatus) {
	c.intentsMutex.Lock()
	defer c.intentsMutex.Unlock()
	
	if intent, exists := c.intents[intentID]; exists {
		intent.Status = status
		intent.UpdatedAt = time.Now()
	}
}

// updateSliceStatus updates the status of a network slice
func (c *SDNController) updateSliceStatus(sliceID string, status SliceStatus) {
	c.slicesMutex.Lock()
	defer c.slicesMutex.Unlock()
	
	if slice, exists := c.networkSlices[sliceID]; exists {
		slice.Status = status
		slice.UpdatedAt = time.Now()
	}
}

// validateSliceRequirements validates network slice requirements
func (c *SDNController) validateSliceRequirements(slice *NetworkSlice) error {
	if slice.Name == "" {
		return fmt.Errorf("slice name cannot be empty")
	}
	
	if len(slice.Endpoints) == 0 {
		return fmt.Errorf("slice must have at least one endpoint")
	}
	
	// Validate QoS profile
	if slice.QoSProfile.MaxLatency <= 0 {
		return fmt.Errorf("max latency must be positive")
	}
	
	if slice.QoSProfile.MinBandwidth <= 0 {
		return fmt.Errorf("min bandwidth must be positive")
	}
	
	if slice.QoSProfile.Availability < 0 || slice.QoSProfile.Availability > 1 {
		return fmt.Errorf("availability must be between 0 and 1")
	}
	
	return nil
}

// metricsCollectionLoop collects and updates SDN metrics
func (c *SDNController) metricsCollectionLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.updateMetrics()
		}
	}
}

// updateMetrics updates the SDN controller metrics
func (c *SDNController) updateMetrics() {
	c.intentsMutex.RLock()
	c.flowRulesMutex.RLock()
	c.slicesMutex.RLock()
	
	defer c.intentsMutex.RUnlock()
	defer c.flowRulesMutex.RUnlock()
	defer c.slicesMutex.RUnlock()
	
	activeIntents := int64(0)
	for _, intent := range c.intents {
		if intent.Status == IntentStatusActive {
			activeIntents++
		}
	}
	
	activeSlices := int64(0)
	for _, slice := range c.networkSlices {
		if slice.Status == SliceStatusActive {
			activeSlices++
		}
	}
	
	c.metrics.ActiveIntents = activeIntents
	c.metrics.ActiveFlowRules = int64(len(c.flowRules))
	c.metrics.ActiveSlices = activeSlices
	c.metrics.LastUpdated = time.Now()
}

// flowRuleCleanupLoop cleans up expired flow rules
func (c *SDNController) flowRuleCleanupLoop() {
	ticker := time.NewTicker(60 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.cleanupExpiredFlowRules()
		}
	}
}

// cleanupExpiredFlowRules removes expired flow rules
func (c *SDNController) cleanupExpiredFlowRules() {
	c.flowRulesMutex.Lock()
	defer c.flowRulesMutex.Unlock()
	
	now := time.Now()
	for id, rule := range c.flowRules {
		// Check if rule has expired (simplified - in reality would track installation time)
		if rule.HardTimeout > 0 {
			// For this example, assume rules are installed when created
			// In practice, would track actual installation timestamps
			log.Printf("Checking flow rule %s for expiration", id)
		}
	}
}

// GetMetrics returns current SDN controller metrics
func (c *SDNController) GetMetrics() *SDNMetrics {
	return c.metrics
}

// ListIntents returns all intents
func (c *SDNController) ListIntents() []*Intent {
	c.intentsMutex.RLock()
	defer c.intentsMutex.RUnlock()
	
	intents := make([]*Intent, 0, len(c.intents))
	for _, intent := range c.intents {
		intents = append(intents, intent)
	}
	
	return intents
}

// ListNetworkSlices returns all network slices
func (c *SDNController) ListNetworkSlices() []*NetworkSlice {
	c.slicesMutex.RLock()
	defer c.slicesMutex.RUnlock()
	
	slices := make([]*NetworkSlice, 0, len(c.networkSlices))
	for _, slice := range c.networkSlices {
		slices = append(slices, slice)
	}
	
	return slices
}