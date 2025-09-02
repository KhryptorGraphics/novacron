package orchestration

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/placement"
)

// Local types and interfaces to support healing/evacuation without new deps
// NodeStatus tracks node health seen by the orchestration engine
type NodeStatus struct {
	ID         string
	Healthy    bool
	LastChange time.Time
	Reason     string
}

// EvacuationHandler defines a minimal interface the engine can call to evacuate a node
// Implementations can live in higher layers and be injected at startup
type EvacuationHandler interface {
	EvacuateNode(ctx context.Context, nodeID string) error
}

// Lightweight helpers for numeric extraction from event payloads
func toFloat64(v interface{}) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case float32:
		return float64(t)
	case int:
		return float64(t)
	case int32:
		return float64(t)
	case int64:
		return float64(t)
	case uint32:
		return float64(t)
	case uint64:
		return float64(t)
	default:
		return 0
	}
}

func toInt(v interface{}) int {
	switch t := v.(type) {
	case int:
		return t
	case int32:
		return int(t)

	case int64:
		return int(t)
	case uint32:
		return int(t)
	case uint64:
		return int(t)
	case float64:
		return int(t)
	case float32:
		return int(t)
	default:
		return 0
	}
}


// DefaultOrchestrationEngine implements the main orchestration engine
type DefaultOrchestrationEngine struct {
	mu              sync.RWMutex
	state           EngineState
	startTime       time.Time
	policies        map[string]*OrchestrationPolicy
	eventBus        events.EventBus
	placementEngine placement.PlacementEngine
	logger          *logrus.Logger

	// Metrics
	eventsProcessed uint64
	decisionsCount  uint64
	metrics         map[string]interface{}

	// Node tracking and healing hooks
	nodeStatuses    map[string]NodeStatus
	lastNodeEvents  map[string]time.Time
	evacuationHandler EvacuationHandler

	// Context for lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
}

// NewDefaultOrchestrationEngine creates a new orchestration engine
func NewDefaultOrchestrationEngine(logger *logrus.Logger) *DefaultOrchestrationEngine {
	ctx, cancel := context.WithCancel(context.Background())

	return &DefaultOrchestrationEngine{
		state:             EngineStateStopped,
		policies:          make(map[string]*OrchestrationPolicy),
		eventBus:          events.NewNoopEventBus(),
		placementEngine:   placement.NewDefaultPlacementEngine(logger),
		logger:            logger,
		metrics:           make(map[string]interface{}),
		nodeStatuses:      make(map[string]NodeStatus),
		lastNodeEvents:    make(map[string]time.Time),
		evacuationHandler: nil,
		ctx:               ctx,
		cancel:            cancel,
	}
}

// Placement exposes the placement engine for adapter wiring
func (e *DefaultOrchestrationEngine) Placement() placement.PlacementEngine {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.placementEngine
}


// EventBus exposes the event bus for consumers like WebSocket manager
func (e *DefaultOrchestrationEngine) EventBus() events.EventBus {
	e.mu.RLock(); defer e.mu.RUnlock()
	return e.eventBus
}



// EvacuateNode invokes the configured evacuation handler for the node
func (e *DefaultOrchestrationEngine) EvacuateNode(ctx context.Context, nodeID string) error {
	e.mu.RLock()
	h := e.evacuationHandler
	e.mu.RUnlock()
	if h == nil {
		return fmt.Errorf("no evacuation handler configured")
	}
	return h.EvacuateNode(ctx, nodeID)
}

// SetEvacuationHandler configures the handler used to evacuate VMs from failed nodes
func (e *DefaultOrchestrationEngine) SetEvacuationHandler(h EvacuationHandler) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.evacuationHandler = h
}

// GetNodeStatuses returns a snapshot of tracked node statuses
func (e *DefaultOrchestrationEngine) GetNodeStatuses() map[string]NodeStatus {
	e.mu.RLock(); defer e.mu.RUnlock()
	out := make(map[string]NodeStatus, len(e.nodeStatuses))
	for k, v := range e.nodeStatuses { out[k] = v }
	return out
}



// Start begins orchestration operations
func (e *DefaultOrchestrationEngine) Start(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.state == EngineStateRunning {
		return fmt.Errorf("orchestration engine is already running")
	}

	e.logger.Info("Starting orchestration engine")
	e.state = EngineStateStarting

	// Connect to event bus
	eventConfig := events.EventBusConfig{
		URL:           "nats://localhost:4222", // Default NATS URL
		ClusterID:     "novacron-cluster",
		ClientID:      "orchestration-engine",
		MaxReconnects: 10,
		ReconnectWait: 5 * time.Second,
		PingInterval:  2 * time.Minute,
		BufferSize:    1000,
		QueueGroup:    "orchestration",
	}

	if err := e.eventBus.Connect(ctx, eventConfig); err != nil {
		e.state = EngineStateError
		return fmt.Errorf("failed to connect to event bus: %w", err)
	}

	// Subscribe to orchestration events
	eventHandler := events.NewCompositeEventHandler("orchestration-main", e.logger)

	// Add handlers for different event types
	eventHandler.AddHandler(
		[]events.EventType{events.EventTypeVMCreated, events.EventTypeVMStarted, events.EventTypeVMStopped},
		events.NewEventHandlerFunc("vm-events", "VM Event Handler", e.handleVMEvent),
	)

	eventHandler.AddHandler(
		[]events.EventType{events.EventTypeNodeMetrics, events.EventTypeNodeFailure},
		events.NewEventHandlerFunc("node-events", "Node Event Handler", e.handleNodeEvent),
	)

	eventHandler.AddHandler(
		[]events.EventType{events.EventTypeScalingTriggered},
		events.NewEventHandlerFunc("scaling-events", "Scaling Event Handler", e.handleScalingEvent),
	)

	_, err := e.eventBus.SubscribeToAll(ctx, eventHandler)
	if err != nil {
		e.state = EngineStateError
		return fmt.Errorf("failed to subscribe to events: %w", err)
	}

	e.state = EngineStateRunning
	e.startTime = time.Now()

	e.logger.WithFields(logrus.Fields{
		"policies_count": len(e.policies),
		"placement_algorithm": e.placementEngine.GetAlgorithm(),
	}).Info("Orchestration engine started successfully")

	// Start background processing
	go e.processLoop()

	return nil
}

// Stop gracefully shuts down the orchestration engine
func (e *DefaultOrchestrationEngine) Stop(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.state != EngineStateRunning {
		return fmt.Errorf("orchestration engine is not running")
	}

	e.logger.Info("Stopping orchestration engine")
	e.state = EngineStateStopping

	// Cancel background processes
	e.cancel()

	// Disconnect from event bus
	if err := e.eventBus.Disconnect(); err != nil {
		e.logger.WithError(err).Error("Error disconnecting from event bus")
	}

	e.state = EngineStateStopped
	e.logger.Info("Orchestration engine stopped")

	return nil
}

// GetStatus returns the current status of the orchestration engine
func (e *DefaultOrchestrationEngine) GetStatus() EngineStatus {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return EngineStatus{
		State:           e.state,
		StartTime:       e.startTime,
		ActivePolicies:  len(e.policies),
		EventsProcessed: e.eventsProcessed,
		Metrics:         e.getMetricsCopy(),
	}
}

// RegisterPolicy registers a new orchestration policy
func (e *DefaultOrchestrationEngine) RegisterPolicy(policy *OrchestrationPolicy) error {
	if policy == nil {
		return fmt.Errorf("policy cannot be nil")
	}

	if policy.ID == "" {
		return fmt.Errorf("policy ID cannot be empty")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.policies[policy.ID] = policy

	e.logger.WithFields(logrus.Fields{
		"policy_id":   policy.ID,
		"policy_name": policy.Name,
		"enabled":     policy.Enabled,
		"priority":    policy.Priority,
	}).Info("Policy registered")

	// Publish policy update event
	event := &events.OrchestrationEvent{
		Type:      events.EventTypePolicyUpdated,
		Source:    "orchestration-engine",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"action":    "registered",
			"policy_id": policy.ID,
			"policy":    policy,
		},
		Priority: events.PriorityNormal,
	}

	if err := e.eventBus.Publish(e.ctx, event); err != nil {
		e.logger.WithError(err).Error("Failed to publish policy update event")
	}

	return nil
}

// UnregisterPolicy removes an orchestration policy
func (e *DefaultOrchestrationEngine) UnregisterPolicy(policyID string) error {
	if policyID == "" {
		return fmt.Errorf("policy ID cannot be empty")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	policy, exists := e.policies[policyID]
	if !exists {
		return fmt.Errorf("policy %s not found", policyID)
	}

	delete(e.policies, policyID)

	e.logger.WithField("policy_id", policyID).Info("Policy unregistered")

	// Publish policy update event
	event := &events.OrchestrationEvent{
		Type:      events.EventTypePolicyUpdated,
		Source:    "orchestration-engine",
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"action":    "unregistered",
			"policy_id": policyID,
			"policy":    policy,
		},
		Priority: events.PriorityNormal,
	}

	if err := e.eventBus.Publish(e.ctx, event); err != nil {
		e.logger.WithError(err).Error("Failed to publish policy update event")
	}

	return nil
}

// MakeVMPlacementDecision makes a placement decision for a VM
func (e *DefaultOrchestrationEngine) MakeVMPlacementDecision(ctx context.Context, vmSpec placement.VMSpec, strategy placement.PlacementStrategy) (*OrchestrationDecision, error) {
	e.mu.RLock()
	placementEngine := e.placementEngine
	e.mu.RUnlock()

	// Create placement request
	placementRequest := &placement.PlacementRequest{
		VMID:        vmSpec.Labels["vm_id"],
		VMSpec:      vmSpec,
		Strategy:    strategy,
		Constraints: e.getRelevantConstraints(vmSpec.Labels),
		Preferences: e.getRelevantPreferences(vmSpec.Labels),
	}

	// Make placement decision
	decision, err := placementEngine.PlaceVM(ctx, placementRequest)
	if err != nil {
		return nil, fmt.Errorf("placement decision failed: %w", err)
	}

	// Convert to orchestration decision
	orchDecision := &OrchestrationDecision{
		ID:           decision.RequestID,
		DecisionType: DecisionTypePlacement,
		Context: DecisionContext{
			RequestID: decision.RequestID,
			Timestamp: time.Now(),
			Trigger:   "vm_placement_request",
			VMID:      decision.VMID,
		},
		Recommendation: fmt.Sprintf("Place VM %s on node %s", decision.VMID, decision.SelectedNode),
		Score:         decision.Score,
		Confidence:    decision.Confidence,
		Explanation:   decision.Explanation,
		Actions: []DecisionAction{
			{
				Type:   ActionTypeSchedule,
				Target: decision.SelectedNode,
				Parameters: map[string]interface{}{
					"vm_id":   decision.VMID,
					"node_id": decision.SelectedNode,
				},
			},
		},
		Timestamp: decision.Timestamp,
		Status:    DecisionStatusPending,
		Metadata: map[string]interface{}{
			"placement_decision": decision,
		},
	}

	// Update metrics
	e.mu.Lock()
	e.decisionsCount++
	e.mu.Unlock()

	e.logger.WithFields(logrus.Fields{
		"vm_id":       decision.VMID,
		"selected_node": decision.SelectedNode,
		"score":       decision.Score,
		"confidence":  decision.Confidence,
	}).Info("VM placement decision made")

	return orchDecision, nil
}

// processLoop runs the main processing loop
func (e *DefaultOrchestrationEngine) processLoop() {
	ticker := time.NewTicker(30 * time.Second) // Process every 30 seconds
	defer ticker.Stop()

	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			e.performPeriodicProcessing()
		}
	}
}

// performPeriodicProcessing performs periodic orchestration tasks
func (e *DefaultOrchestrationEngine) performPeriodicProcessing() {
	e.logger.Debug("Performing periodic orchestration processing")

	// Update metrics
	e.mu.Lock()
	e.metrics["last_processed_at"] = time.Now()
	e.metrics["uptime_seconds"] = time.Since(e.startTime).Seconds()

	// Add event bus metrics
	busMetrics := e.eventBus.GetMetrics()
	e.metrics["events_published"] = busMetrics.EventsPublished
	e.metrics["events_received"] = busMetrics.EventsReceived
	e.metrics["events_processed"] = busMetrics.EventsProcessed
	e.metrics["subscription_count"] = busMetrics.SubscriptionCount
	e.mu.Unlock()

	// TODO: Add more periodic tasks:
	// - Check for policy violations
	// - Trigger optimization recommendations
	// - Update node health scores
	// - Perform predictive scaling analysis
}

// handleVMEvent handles VM-related events
func (e *DefaultOrchestrationEngine) handleVMEvent(ctx context.Context, event *events.OrchestrationEvent) error {
	e.mu.Lock()
	e.eventsProcessed++
	e.mu.Unlock()

	e.logger.WithFields(logrus.Fields{
		"event_id":   event.ID,
		"event_type": event.Type,
		"source":     event.Source,
	}).Debug("Handling VM event")

	switch event.Type {
	case events.EventTypeVMCreated:
		return e.handleVMCreated(ctx, event)
	case events.EventTypeVMStarted:
		return e.handleVMStarted(ctx, event)
	case events.EventTypeVMStopped:
		return e.handleVMStopped(ctx, event)
	}

	return nil
}

// handleNodeEvent handles node-related events
func (e *DefaultOrchestrationEngine) handleNodeEvent(ctx context.Context, event *events.OrchestrationEvent) error {
	e.mu.Lock()
	e.eventsProcessed++
	e.mu.Unlock()

	e.logger.WithFields(logrus.Fields{
		"event_id":   event.ID,
		"event_type": event.Type,
		"source":     event.Source,
	}).Debug("Handling node event")

	switch event.Type {
	case events.EventTypeNodeFailure:
		return e.handleNodeFailure(ctx, event)
	case events.EventTypeNodeMetrics:
		return e.handleNodeMetrics(ctx, event)
	}

	return nil
}

// handleScalingEvent handles scaling-related events
func (e *DefaultOrchestrationEngine) handleScalingEvent(ctx context.Context, event *events.OrchestrationEvent) error {
	e.mu.Lock()
	e.eventsProcessed++
	e.mu.Unlock()

	e.logger.WithFields(logrus.Fields{
		"event_id":   event.ID,
		"event_type": event.Type,
		"source":     event.Source,
	}).Debug("Handling scaling event")

	// TODO: Implement scaling logic
	return nil
}

// Event handlers (simplified implementations)

func (e *DefaultOrchestrationEngine) handleVMCreated(ctx context.Context, event *events.OrchestrationEvent) error {
	e.logger.WithField("vm_id", event.Data["vm_id"]).Info("VM created")
	return nil
}

func (e *DefaultOrchestrationEngine) handleVMStarted(ctx context.Context, event *events.OrchestrationEvent) error {
	e.logger.WithField("vm_id", event.Data["vm_id"]).Info("VM started")
	return nil
}

func (e *DefaultOrchestrationEngine) handleVMStopped(ctx context.Context, event *events.OrchestrationEvent) error {
	e.logger.WithField("vm_id", event.Data["vm_id"]).Info("VM stopped")
	return nil
}

func (e *DefaultOrchestrationEngine) handleNodeFailure(ctx context.Context, event *events.OrchestrationEvent) error {
	nodeID := event.Data["node_id"]

	// Update node status and publish healing trigger
	if nid, ok := event.Data["node_id"].(string); ok && nid != "" {
		// mark unhealthy
		e.mu.Lock()
		e.nodeStatuses[nid] = NodeStatus{ID: nid, Healthy: false, LastChange: time.Now(), Reason: "failure_event"}
		e.lastNodeEvents[nid] = time.Now()
		e.mu.Unlock()
		// publish healing event
		healEvent := &events.OrchestrationEvent{
			Type:      events.EventTypeHealingTriggered,
			Source:    "orchestration-engine",
			Timestamp: time.Now(),
			Data: map[string]interface{}{"node_id": nid, "cause": "node.failure"},
			Priority: events.PriorityHigh,
		}
		if err := e.eventBus.Publish(ctx, healEvent); err != nil {
			e.logger.WithError(err).Error("Failed to publish healing event")
		}
		// best-effort evacuation using optional handler
		if e.evacuationHandler != nil {
			go func(target string) {
				if err := e.evacuationHandler.EvacuateNode(context.Background(), target); err != nil {
					e.logger.WithError(err).WithField("node_id", target).Error("Node evacuation failed")
				} else {
					e.logger.WithField("node_id", target).Info("Node evacuation initiated")
				}
			}(nid)
		}
	}

	e.logger.WithField("node_id", nodeID).Warn("Node failure detected")

	// TODO: Implement node failure handling:
	// - Migrate VMs from failed node
	// - Update node status
	// - Trigger healing policies

	return nil
}

func (e *DefaultOrchestrationEngine) handleNodeMetrics(ctx context.Context, event *events.OrchestrationEvent) error {
	// Ingest node metrics into engine metrics map for later policy evaluation
	nodeID, _ := event.Data["node_id"].(string)
	if nodeID == "" {
		return nil
	}
	cpu := toFloat64(event.Data["cpu_utilization"]) // percent
	mem := toFloat64(event.Data["memory_utilization"]) // percent
	disk := toFloat64(event.Data["disk_utilization"]) // percent
	net := toFloat64(event.Data["network_utilization"]) // percent
	activeVMs := toInt(event.Data["active_vms"])
	healthy := true
	if h, ok := event.Data["healthy"].(bool); ok {
		healthy = h
	}

	e.mu.Lock()
	e.metrics[fmt.Sprintf("nodes.%s.cpu_utilization", nodeID)] = cpu
	e.metrics[fmt.Sprintf("nodes.%s.memory_utilization", nodeID)] = mem
	e.metrics[fmt.Sprintf("nodes.%s.disk_utilization", nodeID)] = disk
	e.metrics[fmt.Sprintf("nodes.%s.network_utilization", nodeID)] = net
	e.metrics[fmt.Sprintf("nodes.%s.active_vms", nodeID)] = activeVMs
	e.nodeStatuses[nodeID] = NodeStatus{ID: nodeID, Healthy: healthy, LastChange: time.Now()}
	e.lastNodeEvents[nodeID] = time.Now()
	e.mu.Unlock()

	return nil
}

// Helper methods

func (e *DefaultOrchestrationEngine) getRelevantConstraints(labels map[string]string) []placement.Constraint {
	var constraints []placement.Constraint

	// Apply constraints from active policies based on label selectors
	for _, policy := range e.policies {
		if !policy.Enabled {
			continue
		}

		if e.policyApplies(policy, labels) {
			for _, rule := range policy.Rules {
				if rule.Type == RuleTypePlacement && rule.Enabled {
					// Convert rule to constraints
					// This is simplified - real implementation would be more complex
					constraints = append(constraints, placement.Constraint{
						Type:        placement.ConstraintTypeResourceLimit,
						Enforcement: placement.EnforcementSoft,
						Parameters:  rule.Parameters,
						Weight:      float64(rule.Priority),
					})
				}
			}
		}
	}

	return constraints
}

func (e *DefaultOrchestrationEngine) getRelevantPreferences(labels map[string]string) []placement.Preference {
	var preferences []placement.Preference

	// Apply preferences from active policies
	for _, policy := range e.policies {
		if !policy.Enabled {
			continue
		}

		if e.policyApplies(policy, labels) {
			// Convert policy rules to preferences
			// This is simplified
			preferences = append(preferences, placement.Preference{
				Type:   placement.PreferenceTypeLowCost,
				Weight: 1.0,
			})
		}
	}

	return preferences
}

func (e *DefaultOrchestrationEngine) policyApplies(policy *OrchestrationPolicy, labels map[string]string) bool {
	// Check if policy selector matches the given labels
	for key, value := range policy.Selector.Labels {
		if labels[key] != value {
			return false
		}
	}

	for key, value := range policy.Selector.Tags {
		if labels[key] != value {
			return false
		}
	}

	return true
}

func (e *DefaultOrchestrationEngine) getMetricsCopy() map[string]interface{} {
	metrics := make(map[string]interface{})
	for k, v := range e.metrics {
		metrics[k] = v
	}
	return metrics
}