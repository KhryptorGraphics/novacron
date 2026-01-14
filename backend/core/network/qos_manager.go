package network

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"net"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

type QoSPolicy struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	Description      string                 `json:"description"`
	NetworkID        string                 `json:"network_id"`
	InterfaceName    string                 `json:"interface_name"`
	Priority         int                    `json:"priority"`
	Rules            []ClassificationRule   `json:"rules"`
	Actions          []QoSAction           `json:"actions"`
	Statistics       QoSStatistics         `json:"statistics"`
	Enabled          bool                  `json:"enabled"`
	CreatedAt        time.Time             `json:"created_at"`
	UpdatedAt        time.Time             `json:"updated_at"`
	Metadata         map[string]string     `json:"metadata"`
}

type ClassificationRule struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	SourceIP    string            `json:"source_ip,omitempty"`
	DestIP      string            `json:"dest_ip,omitempty"`
	SourcePort  int               `json:"source_port,omitempty"`
	DestPort    int               `json:"dest_port,omitempty"`
	Protocol    string            `json:"protocol,omitempty"`
	DSCPMark    int               `json:"dscp_mark,omitempty"`
	Application string            `json:"application,omitempty"`
	Match       string            `json:"match"`
	Metadata    map[string]string `json:"metadata"`
}

type QoSAction struct {
	Type         string            `json:"type"`
	RateLimit    uint64            `json:"rate_limit,omitempty"`
	BurstLimit   uint64            `json:"burst_limit,omitempty"`
	Priority     int               `json:"priority,omitempty"`
	DSCPMark     int               `json:"dscp_mark,omitempty"`
	QueueName    string            `json:"queue_name,omitempty"`
	Parameters   map[string]string `json:"parameters,omitempty"`
}

type QoSStatistics struct {
	PacketsMatched    uint64    `json:"packets_matched"`
	BytesMatched      uint64    `json:"bytes_matched"`
	PacketsDropped    uint64    `json:"packets_dropped"`
	BytesDropped      uint64    `json:"bytes_dropped"`
	QueueLength       int       `json:"queue_length"`
	AverageLatency    float64   `json:"average_latency_ms"`
	LastUpdated       time.Time `json:"last_updated"`
	ThroughputBps     uint64    `json:"throughput_bps"`
	UtilizationPercent float64  `json:"utilization_percent"`
}

type TrafficClass struct {
	ID            string    `json:"id"`
	Name          string    `json:"name"`
	MinBandwidth  uint64    `json:"min_bandwidth"`
	MaxBandwidth  uint64    `json:"max_bandwidth"`
	Priority      int       `json:"priority"`
	QueueType     string    `json:"queue_type"`
	TcClassID     string    `json:"tc_class_id"` // Store traffic control class ID for consistent retrieval
	Statistics    QoSStatistics `json:"statistics"`
}

type QueueConfig struct {
	Name           string  `json:"name"`
	Type           string  `json:"type"`
	MinRate        uint64  `json:"min_rate"`
	MaxRate        uint64  `json:"max_rate"`
	BurstSize      uint64  `json:"burst_size"`
	Priority       int     `json:"priority"`
	Weight         int     `json:"weight"`
	QueueLimit     int     `json:"queue_limit"`
	REDMinThresh   int     `json:"red_min_thresh,omitempty"`
	REDMaxThresh   int     `json:"red_max_thresh,omitempty"`
	REDProbability float64 `json:"red_probability,omitempty"`
}

type TrafficClassifier struct {
	rules     map[string][]ClassificationRule
	policies  map[string]*QoSPolicy
	logger    *zap.Logger
	mu        sync.RWMutex
}

func NewTrafficClassifier(logger *zap.Logger) *TrafficClassifier {
	return &TrafficClassifier{
		rules:    make(map[string][]ClassificationRule),
		policies: make(map[string]*QoSPolicy),
		logger:   logger,
	}
}

func (tc *TrafficClassifier) AddRule(interfaceName string, rule ClassificationRule) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	if rule.ID == "" {
		rule.ID = generateID()
	}

	tc.rules[interfaceName] = append(tc.rules[interfaceName], rule)
	
	tc.logger.Debug("Classification rule added", 
		zap.String("interface", interfaceName),
		zap.String("rule_id", rule.ID),
		zap.String("rule_name", rule.Name))
	
	return nil
}

func (tc *TrafficClassifier) ClassifyPacket(interfaceName, srcIP, dstIP string, srcPort, dstPort int, protocol string) (*ClassificationRule, error) {
	tc.mu.RLock()
	defer tc.mu.RUnlock()

	rules, exists := tc.rules[interfaceName]
	if !exists {
		return nil, fmt.Errorf("no rules defined for interface %s", interfaceName)
	}

	for _, rule := range rules {
		if tc.matchRule(&rule, srcIP, dstIP, srcPort, dstPort, protocol) {
			return &rule, nil
		}
	}

	return nil, nil
}

func (tc *TrafficClassifier) matchRule(rule *ClassificationRule, srcIP, dstIP string, srcPort, dstPort int, protocol string) bool {
	if rule.SourceIP != "" && !tc.matchCIDR(rule.SourceIP, srcIP) {
		return false
	}
	
	if rule.DestIP != "" && !tc.matchCIDR(rule.DestIP, dstIP) {
		return false
	}
	
	if rule.SourcePort != 0 && rule.SourcePort != srcPort {
		return false
	}
	
	if rule.DestPort != 0 && rule.DestPort != dstPort {
		return false
	}
	
	if rule.Protocol != "" && strings.ToLower(rule.Protocol) != strings.ToLower(protocol) {
		return false
	}

	return true
}

func (tc *TrafficClassifier) matchCIDR(cidr, ip string) bool {
	if cidr == ip {
		return true
	}

	_, ipNet, err := net.ParseCIDR(cidr)
	if err != nil {
		return false
	}

	targetIP := net.ParseIP(ip)
	if targetIP == nil {
		return false
	}

	return ipNet.Contains(targetIP)
}

type TrafficShaper struct {
	interfaces map[string]*InterfaceShaper
	logger     *zap.Logger
	mu         sync.RWMutex
}

type InterfaceShaper struct {
	name        string
	classes     map[string]*TrafficClass
	queues      map[string]*QueueConfig
	htbHandle   string
	rootQdiscSetup bool // Track if root qdisc has been set up to prevent repeated setup
	mu          sync.RWMutex
}

func NewTrafficShaper(logger *zap.Logger) *TrafficShaper {
	return &TrafficShaper{
		interfaces: make(map[string]*InterfaceShaper),
		logger:     logger,
	}
}

func (ts *TrafficShaper) SetupInterface(interfaceName string) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if _, exists := ts.interfaces[interfaceName]; exists {
		return fmt.Errorf("interface %s already configured", interfaceName)
	}

	shaper := &InterfaceShaper{
		name:      interfaceName,
		classes:   make(map[string]*TrafficClass),
		queues:    make(map[string]*QueueConfig),
		htbHandle: "1:",
	}

	if err := ts.setupRootQdisc(interfaceName); err != nil {
		return fmt.Errorf("failed to setup root qdisc: %w", err)
	}

	ts.interfaces[interfaceName] = shaper
	
	ts.logger.Info("Traffic shaping configured for interface", 
		zap.String("interface", interfaceName))
	
	return nil
}

func (ts *TrafficShaper) setupRootQdisc(interfaceName string) error {
	return ts.setupRootQdiscWithRate(interfaceName, 0)
}

func (ts *TrafficShaper) setupRootQdiscWithRate(interfaceName string, rateBps uint64) error {
	ts.mu.RLock()
	shaper, exists := ts.interfaces[interfaceName]
	ts.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("interface %s not found", interfaceName)
	}
	
	shaper.mu.Lock()
	defer shaper.mu.Unlock()
	
	// Prevent repeated root qdisc setup
	if shaper.rootQdiscSetup {
		ts.logger.Debug("Root qdisc already set up for interface", zap.String("interface", interfaceName))
		return nil
	}
	
	// Determine the default rate
	defaultRate := "1000mbit"
	if rateBps > 0 {
		defaultRate = bpsToTcRate(rateBps)
	}
	
	cmd := exec.Command("tc", "qdisc", "add", "dev", interfaceName, "root", "handle", "1:", "htb", "default", "999")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to add root htb qdisc: %w", err)
	}

	cmd = exec.Command("tc", "class", "add", "dev", interfaceName, "parent", "1:", "classid", "1:999", "htb", "rate", defaultRate)
	if err := cmd.Run(); err != nil {
		ts.logger.Warn("Failed to add default class", zap.Error(err))
	}

	// Mark root qdisc as set up
	shaper.rootQdiscSetup = true

	ts.logger.Info("Root qdisc configured",
		zap.String("interface", interfaceName),
		zap.String("default_rate", defaultRate))

	return nil
}

func (ts *TrafficShaper) AddTrafficClass(interfaceName string, class *TrafficClass) error {
	ts.mu.RLock()
	shaper, exists := ts.interfaces[interfaceName]
	ts.mu.RUnlock()

	if !exists {
		return fmt.Errorf("interface %s not configured for shaping", interfaceName)
	}

	shaper.mu.Lock()
	defer shaper.mu.Unlock()

	if class.ID == "" {
		class.ID = generateID()
	}

	classID := fmt.Sprintf("1:%d", 10+len(shaper.classes))
	
	// Store the TC class ID in the TrafficClass for consistent retrieval
	class.TcClassID = classID
	
	minRate := bpsToTcRate(class.MinBandwidth)
	maxRate := bpsToTcRate(class.MaxBandwidth)

	cmd := exec.Command("tc", "class", "add", "dev", interfaceName, "parent", "1:", 
		"classid", classID, "htb", "rate", minRate, "ceil", maxRate, "prio", strconv.Itoa(class.Priority))
	
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to add traffic class: %w", err)
	}

	qdisc := "pfifo"
	if class.QueueType != "" {
		qdisc = class.QueueType
	}

	cmd = exec.Command("tc", "qdisc", "add", "dev", interfaceName, "parent", classID, qdisc)
	if err := cmd.Run(); err != nil {
		ts.logger.Warn("Failed to add queue discipline", 
			zap.String("interface", interfaceName),
			zap.String("class_id", classID),
			zap.Error(err))
	}

	shaper.classes[class.ID] = class
	
	ts.logger.Info("Traffic class added", 
		zap.String("interface", interfaceName),
		zap.String("class_id", class.ID),
		zap.String("tc_class_id", classID))
	
	return nil
}

func (ts *TrafficShaper) ApplyRateLimit(interfaceName, classID string, rate, burst uint64) error {
	ts.mu.RLock()
	shaper, exists := ts.interfaces[interfaceName]
	ts.mu.RUnlock()

	if !exists {
		return fmt.Errorf("interface %s not configured", interfaceName)
	}

	shaper.mu.Lock()
	class, exists := shaper.classes[classID]
	if !exists {
		shaper.mu.Unlock()
		return fmt.Errorf("traffic class %s not found", classID)
	}
	
	// Store the old rate for logging
	oldRate := class.MaxBandwidth
	class.MaxBandwidth = rate
	
	// Get the tc class ID that was assigned when the class was created
	tcClassID := ""
	for i, c := range shaper.classes {
		if c.ID == classID {
			// The tc class ID was assigned as "1:X" where X = 10 + index
			classIndex := 0
			for id := range shaper.classes {
				if id == i {
					break
				}
				classIndex++
			}
			tcClassID = fmt.Sprintf("1:%d", 10+classIndex)
			break
		}
	}
	shaper.mu.Unlock()
	
	if tcClassID == "" {
		return fmt.Errorf("could not determine tc class ID for class %s", classID)
	}
	
	// Apply the rate limit to the kernel using tc command
	// This actually enforces the rate limit at the kernel level
	rateStr := bpsToTcRate(rate)
	
	// Use 'tc class change' to update the existing class rate
	cmd := exec.Command("tc", "class", "change", "dev", interfaceName,
		"parent", "1:", "classid", tcClassID, "htb", "rate", rateStr)
	
	if burst > 0 {
		burstStr := fmt.Sprintf("%d", burst)
		cmd.Args = append(cmd.Args, "burst", burstStr)
	}
	
	if err := cmd.Run(); err != nil {
		// Log the error but don't fail completely
		ts.logger.Error("Failed to apply rate limit in kernel",
			zap.String("interface", interfaceName),
			zap.String("class", classID),
			zap.String("tc_class_id", tcClassID),
			zap.Uint64("rate", rate),
			zap.Error(err))
	}
	
	ts.logger.Info("Rate limit applied", 
		zap.String("interface", interfaceName),
		zap.String("class", classID),
		zap.String("tc_class_id", tcClassID),
		zap.Uint64("old_rate", oldRate),
		zap.Uint64("new_rate", rate),
		zap.Uint64("burst", burst))
	
	return nil
}

type QueueManager struct {
	queues     map[string]*QueueConfig
	statistics map[string]*QoSStatistics
	logger     *zap.Logger
	mu         sync.RWMutex
}

func NewQueueManager(logger *zap.Logger) *QueueManager {
	return &QueueManager{
		queues:     make(map[string]*QueueConfig),
		statistics: make(map[string]*QoSStatistics),
		logger:     logger,
	}
}

func (qm *QueueManager) CreateQueue(config *QueueConfig) error {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	if config.Name == "" {
		return fmt.Errorf("queue name is required")
	}

	if _, exists := qm.queues[config.Name]; exists {
		return fmt.Errorf("queue %s already exists", config.Name)
	}

	qm.queues[config.Name] = config
	qm.statistics[config.Name] = &QoSStatistics{
		LastUpdated: time.Now(),
	}

	qm.logger.Info("Queue created", 
		zap.String("queue", config.Name),
		zap.String("type", config.Type),
		zap.Uint64("max_rate", config.MaxRate))

	return nil
}

func (qm *QueueManager) GetQueueStatistics(queueName string) (*QoSStatistics, error) {
	qm.mu.RLock()
	defer qm.mu.RUnlock()

	stats, exists := qm.statistics[queueName]
	if !exists {
		return nil, fmt.Errorf("queue %s not found", queueName)
	}

	return stats, nil
}

func (qm *QueueManager) UpdateQueueStatistics(queueName string, stats *QoSStatistics) error {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	qm.statistics[queueName] = stats
	return nil
}

type QoSManager struct {
	config           *QoSManagerConfig
	policies         map[string]*QoSPolicy
	classifier       *TrafficClassifier
	shaper          *TrafficShaper
	queueManager    *QueueManager
	bandwidthMonitor *BandwidthMonitor
	logger          *zap.Logger
	ctx             context.Context
	cancel          context.CancelFunc
	mu              sync.RWMutex
	running         bool
	appliedClasses   map[string]string   // policy ID -> tc class ID mapping
}

type QoSManagerConfig struct {
	EnableTrafficShaping  bool                   `json:"enable_traffic_shaping"`
	EnableDSCPMarking    bool                   `json:"enable_dscp_marking"`
	UpdateInterval       time.Duration          `json:"update_interval"`
	DefaultPolicies      []*QoSPolicy          `json:"default_policies"`
	StatisticsRetention  time.Duration          `json:"statistics_retention"`
	MaxPoliciesPerInterface int                `json:"max_policies_per_interface"`
	DefaultRateBps       uint64                 `json:"default_rate_bps"`  // Default rate for root qdisc
}

func NewQoSManager(config *QoSManagerConfig, bandwidthMonitor *BandwidthMonitor, logger *zap.Logger) *QoSManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	if config.UpdateInterval == 0 {
		config.UpdateInterval = 30 * time.Second
	}
	if config.StatisticsRetention == 0 {
		config.StatisticsRetention = 24 * time.Hour
	}
	if config.MaxPoliciesPerInterface == 0 {
		config.MaxPoliciesPerInterface = 100
	}

	qm := &QoSManager{
		config:           config,
		policies:         make(map[string]*QoSPolicy),
		classifier:       NewTrafficClassifier(logger),
		shaper:          NewTrafficShaper(logger),
		queueManager:    NewQueueManager(logger),
		bandwidthMonitor: bandwidthMonitor,
		logger:          logger,
		ctx:             ctx,
		cancel:          cancel,
		appliedClasses:   make(map[string]string),
	}

	if bandwidthMonitor != nil {
		bandwidthMonitor.AddQoSHook(qm.handleBandwidthAlert)
	}

	return qm
}

func (qm *QoSManager) Start() error {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	if qm.running {
		return fmt.Errorf("QoS manager is already running")
	}

	for _, policy := range qm.config.DefaultPolicies {
		if err := qm.addPolicyUnsafe(policy); err != nil {
			qm.logger.Warn("Failed to apply default policy", 
				zap.String("policy", policy.Name),
				zap.Error(err))
		}
	}

	qm.running = true
	go qm.statisticsUpdateLoop()
	go qm.reconciliationLoop()

	qm.logger.Info("QoS Manager started", 
		zap.Bool("traffic_shaping", qm.config.EnableTrafficShaping),
		zap.Bool("dscp_marking", qm.config.EnableDSCPMarking))

	return nil
}

func (qm *QoSManager) Stop() error {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	if !qm.running {
		return nil
	}

	qm.cancel()
	qm.running = false

	qm.logger.Info("QoS Manager stopped")
	return nil
}

func (qm *QoSManager) AddPolicy(policy *QoSPolicy) error {
	qm.mu.Lock()
	defer qm.mu.Unlock()
	return qm.addPolicyUnsafe(policy)
}

func (qm *QoSManager) addPolicyUnsafe(policy *QoSPolicy) error {
	if policy.ID == "" {
		policy.ID = generateID()
	}
	
	if policy.CreatedAt.IsZero() {
		policy.CreatedAt = time.Now()
	}
	policy.UpdatedAt = time.Now()

	if qm.config.EnableTrafficShaping && policy.InterfaceName != "" {
		// Setup interface with configured default rate if available
		if qm.config.DefaultRateBps > 0 {
			// Use setupRootQdiscWithRate to configure with specific rate
			if err := qm.shaper.setupRootQdiscWithRate(policy.InterfaceName, qm.config.DefaultRateBps); err != nil {
				qm.logger.Warn("Failed to setup traffic shaping with configured rate", 
					zap.String("interface", policy.InterfaceName),
					zap.Uint64("rate_bps", qm.config.DefaultRateBps),
					zap.Error(err))
				// Fall back to regular setup
				if err := qm.shaper.SetupInterface(policy.InterfaceName); err != nil {
					qm.logger.Warn("Failed to setup traffic shaping for interface", 
						zap.String("interface", policy.InterfaceName),
						zap.Error(err))
				}
			}
		} else {
			if err := qm.shaper.SetupInterface(policy.InterfaceName); err != nil {
				qm.logger.Warn("Failed to setup traffic shaping for interface", 
					zap.String("interface", policy.InterfaceName),
					zap.Error(err))
			}
		}
	}

	for _, rule := range policy.Rules {
		if err := qm.classifier.AddRule(policy.InterfaceName, rule); err != nil {
			return fmt.Errorf("failed to add classification rule: %w", err)
		}
	}

	// Apply QoS actions via traffic control
	for _, action := range policy.Actions {
		if err := qm.applyQoSAction(policy, action); err != nil {
			qm.logger.Warn("Failed to apply QoS action",
				zap.String("policy", policy.Name),
				zap.String("action_type", action.Type),
				zap.Error(err))
		}
	}

	qm.policies[policy.ID] = policy

	qm.logger.Info("QoS policy added", 
		zap.String("policy_id", policy.ID),
		zap.String("policy_name", policy.Name),
		zap.String("interface", policy.InterfaceName))

	return nil
}

func (qm *QoSManager) RemovePolicy(policyID string) error {
	qm.mu.Lock()
	defer qm.mu.Unlock()

	policy, exists := qm.policies[policyID]
	if !exists {
		return fmt.Errorf("policy %s not found", policyID)
	}

	delete(qm.policies, policyID)

	qm.logger.Info("QoS policy removed", 
		zap.String("policy_id", policyID),
		zap.String("policy_name", policy.Name))

	return nil
}

func (qm *QoSManager) GetPolicy(policyID string) (*QoSPolicy, error) {
	qm.mu.RLock()
	defer qm.mu.RUnlock()

	policy, exists := qm.policies[policyID]
	if !exists {
		return nil, fmt.Errorf("policy %s not found", policyID)
	}

	return policy, nil
}

func (qm *QoSManager) ListPolicies() []*QoSPolicy {
	qm.mu.RLock()
	defer qm.mu.RUnlock()

	policies := make([]*QoSPolicy, 0, len(qm.policies))
	for _, policy := range qm.policies {
		policies = append(policies, policy)
	}

	return policies
}

func (qm *QoSManager) GetInterfacePolicies(interfaceName string) []*QoSPolicy {
	qm.mu.RLock()
	defer qm.mu.RUnlock()

	var policies []*QoSPolicy
	for _, policy := range qm.policies {
		if policy.InterfaceName == interfaceName && policy.Enabled {
			policies = append(policies, policy)
		}
	}

	return policies
}

func (qm *QoSManager) statisticsUpdateLoop() {
	ticker := time.NewTicker(qm.config.UpdateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-qm.ctx.Done():
			return
		case <-ticker.C:
			qm.updateStatistics()
		}
	}
}

func (qm *QoSManager) updateStatistics() {
	qm.mu.RLock()
	policies := make([]*QoSPolicy, 0, len(qm.policies))
	for _, policy := range qm.policies {
		policies = append(policies, policy)
	}
	qm.mu.RUnlock()

	for _, policy := range policies {
		stats, err := qm.collectPolicyStatistics(policy)
		if err != nil {
			qm.logger.Warn("Failed to collect policy statistics", 
				zap.String("policy_id", policy.ID),
				zap.Error(err))
			continue
		}

		qm.mu.Lock()
		if existingPolicy, exists := qm.policies[policy.ID]; exists {
			existingPolicy.Statistics = *stats
			existingPolicy.UpdatedAt = time.Now()
		}
		qm.mu.Unlock()
	}
}

func (qm *QoSManager) collectPolicyStatistics(policy *QoSPolicy) (*QoSStatistics, error) {
	stats := &QoSStatistics{
		LastUpdated: time.Now(),
	}

	if qm.bandwidthMonitor != nil {
		measurement, err := qm.bandwidthMonitor.GetCurrentMeasurement(policy.InterfaceName)
		if err == nil {
			stats.ThroughputBps = uint64(measurement.RXRate + measurement.TXRate)
			stats.UtilizationPercent = measurement.Utilization
		}
	}

	return stats, nil
}

func (qm *QoSManager) handleBandwidthAlert(interfaceName string, utilization float64) {
	policies := qm.GetInterfacePolicies(interfaceName)
	
	qm.logger.Info("Handling bandwidth alert for QoS adjustment", 
		zap.String("interface", interfaceName),
		zap.Float64("utilization", utilization),
		zap.Int("policies", len(policies)))

	for _, policy := range policies {
		for _, action := range policy.Actions {
			if action.Type == "rate_limit" && utilization > 80 {
				if action.RateLimit > 0 {
					newRate := uint64(float64(action.RateLimit) * 0.8)
					qm.logger.Info("Reducing rate limit due to congestion", 
						zap.String("interface", interfaceName),
						zap.String("policy", policy.Name),
						zap.Uint64("old_rate", action.RateLimit),
						zap.Uint64("new_rate", newRate))
					
					action.RateLimit = newRate
					
					// Apply the new rate limit via traffic control
					if classID, exists := qm.appliedClasses[policy.ID]; exists {
						if err := qm.shaper.ApplyRateLimit(interfaceName, classID, newRate, action.BurstLimit); err != nil {
							qm.logger.Error("Failed to apply rate limit",
								zap.String("interface", interfaceName),
								zap.String("policy", policy.Name),
								zap.Error(err))
						}
					}
				}
			}
		}
	}
}

func (qm *QoSManager) GetNetworkQoSStatus(networkID string) map[string]interface{} {
	qm.mu.RLock()
	defer qm.mu.RUnlock()

	status := map[string]interface{}{
		"network_id": networkID,
		"policies":   make([]*QoSPolicy, 0),
		"total_policies": 0,
		"active_policies": 0,
		"last_updated": time.Now(),
	}

	var policies []*QoSPolicy
	for _, policy := range qm.policies {
		if policy.NetworkID == networkID {
			policies = append(policies, policy)
			if policy.Enabled {
				status["active_policies"] = status["active_policies"].(int) + 1
			}
		}
	}

	status["policies"] = policies
	status["total_policies"] = len(policies)

	return status
}

// applyQoSAction applies a QoS action via traffic control
func (qm *QoSManager) applyQoSAction(policy *QoSPolicy, action QoSAction) error {
	if !qm.config.EnableTrafficShaping {
		qm.logger.Debug("Traffic shaping disabled, skipping QoS action",
			zap.String("policy", policy.Name),
			zap.String("action_type", action.Type))
		return nil
	}
	switch action.Type {
	case "rate_limit":
		// Create a traffic class for this policy
		class := &TrafficClass{
			ID:           policy.ID,
			Name:         policy.Name,
			MinBandwidth: action.RateLimit / 2, // Min is half of max
			MaxBandwidth: action.RateLimit,
			Priority:     action.Priority,
			QueueType:    "pfifo",
		}
		
		if err := qm.shaper.AddTrafficClass(policy.InterfaceName, class); err != nil {
			return fmt.Errorf("failed to add traffic class: %w", err)
		}
		
		// Store the mapping
		qm.appliedClasses[policy.ID] = class.ID
		
		// Apply the rate limit
		if err := qm.shaper.ApplyRateLimit(policy.InterfaceName, class.ID, action.RateLimit, action.BurstLimit); err != nil {
			return fmt.Errorf("failed to apply rate limit: %w", err)
		}
		
	case "dscp_mark":
		if qm.config.EnableDSCPMarking {
			// Would implement DSCP marking via iptables or tc filter
			qm.logger.Info("DSCP marking requested",
				zap.String("policy", policy.Name),
				zap.Int("dscp", action.DSCPMark))
		}
		
	case "queue":
		// Create a queue for this policy
		queueConfig := &QueueConfig{
			Name:      action.QueueName,
			Type:      "htb",
			MaxRate:   action.RateLimit,
			BurstSize: action.BurstLimit,
			Priority:  action.Priority,
		}
		
		if err := qm.queueManager.CreateQueue(queueConfig); err != nil {
			return fmt.Errorf("failed to create queue: %w", err)
		}
	}
	
	return nil
}

// reconciliationLoop periodically reconciles desired state with actual state
func (qm *QoSManager) reconciliationLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-qm.ctx.Done():
			return
		case <-ticker.C:
			qm.reconcileState()
		}
	}
}

// reconcileState ensures the desired QoS state matches the actual state
func (qm *QoSManager) reconcileState() {
	qm.mu.RLock()
	policies := make([]*QoSPolicy, 0, len(qm.policies))
	for _, policy := range qm.policies {
		if policy.Enabled {
			policies = append(policies, policy)
		}
	}
	qm.mu.RUnlock()
	
	for _, policy := range policies {
		// Check if policy is still applied
		if _, exists := qm.appliedClasses[policy.ID]; !exists {
			// Re-apply the policy
			qm.logger.Info("Re-applying QoS policy during reconciliation",
				zap.String("policy", policy.Name))
			
			for _, action := range policy.Actions {
				if err := qm.applyQoSAction(policy, action); err != nil {
					qm.logger.Error("Failed to re-apply QoS action",
						zap.String("policy", policy.Name),
						zap.Error(err))
				}
			}
		}
	}
}

// bpsToTcRate converts bits per second to tc rate format string
func bpsToTcRate(bps uint64) string {
	if bps >= 1_000_000_000 {
		return fmt.Sprintf("%.1fgbit", float64(bps)/1_000_000_000)
	} else if bps >= 1_000_000 {
		return fmt.Sprintf("%.1fmbit", float64(bps)/1_000_000)
	} else if bps >= 1_000 {
		return fmt.Sprintf("%.1fkbit", float64(bps)/1_000)
	}
	return fmt.Sprintf("%dbit", bps)
}

func generateID() string {
	bytes := make([]byte, 8)
	rand.Read(bytes)
	return hex.EncodeToString(bytes)
}

// applyRateLimitWithTC applies rate limiting using tc (traffic control) command
func (qm *QoSManager) applyRateLimitWithTC(interfaceName, classID string, rateKbps uint64) error {
	// Convert rate to tc format
	rateStr := fmt.Sprintf("%dkbit", rateKbps)

	// Apply rate limit using tc command
	cmd := exec.Command("tc", "class", "change", "dev", interfaceName,
		"parent", "1:", "classid", classID, "htb", "rate", rateStr)

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to apply rate limit with tc: %w", err)
	}

	qm.logger.Info("Rate limit applied with tc",
		zap.String("interface", interfaceName),
		zap.String("class", classID),
		zap.String("rate", rateStr))

	return nil
}