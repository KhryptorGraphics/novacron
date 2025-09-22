package security

import (
	"context"
	"crypto/tls"
	"encoding/base64"
	"fmt"
	"log"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	securityEventsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_security_events_total",
			Help: "Total number of security events processed",
		},
		[]string{"event_type", "severity", "cluster"},
	)

	securityEventsDropped = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "novacron_security_events_dropped_total",
			Help: "Total number of security events dropped due to backpressure",
		},
	)

	crossClusterLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "novacron_cross_cluster_security_latency_seconds",
			Help: "Latency of cross-cluster security operations",
		},
		[]string{"operation", "target_cluster"},
	)
)

// SecurityEvent represents a distributed security event
type SecurityEvent struct {
	ID          string                 `json:"id"`
	Type        SecurityEventType      `json:"type"`
	Severity    ThreatSeverity         `json:"severity"`
	Source      string                 `json:"source"`
	Target      string                 `json:"target,omitempty"`
	Data        map[string]interface{} `json:"data"`
	Timestamp   time.Time              `json:"timestamp"`
	ClusterID   string                 `json:"cluster_id"`
	NodeID      string                 `json:"node_id"`
	Signature   string                 `json:"signature"`
	Propagated  bool                   `json:"propagated"`
}

// SecurityEventType defines types of security events
type SecurityEventType string

const (
	EventTypeAuthFailure        SecurityEventType = "auth_failure"
	EventTypeUnauthorizedAccess SecurityEventType = "unauthorized_access"
	EventTypeSuspiciousActivity SecurityEventType = "suspicious_activity"
	EventTypeVulnerabilityFound SecurityEventType = "vulnerability_found"
	EventTypeComplianceViolation SecurityEventType = "compliance_violation"
	EventTypeSecurityBreach     SecurityEventType = "security_breach"
	EventTypeIntrusionAttempt   SecurityEventType = "intrusion_attempt"
	EventTypeDataExfiltration   SecurityEventType = "data_exfiltration"
	EventTypeCrossCusterThreat  SecurityEventType = "cross_cluster_threat"
)

// SecuritySeverity is now using ThreatSeverity from api_security.go
// to avoid duplicate definitions

// ClusterSecurityState represents security state for a cluster
type ClusterSecurityState struct {
	ClusterID       string                 `json:"cluster_id"`
	ThreatLevel     ThreatSeverity         `json:"threat_level"`
	ActiveThreats   []SecurityEvent        `json:"active_threats"`
	LastUpdate      time.Time              `json:"last_update"`
	SecurityMetrics map[string]float64     `json:"security_metrics"`
	ComplianceScore float64                `json:"compliance_score"`
	Quarantined     bool                   `json:"quarantined"`
	Policies        map[string]interface{} `json:"policies"`
}

// DistributedSecurityCoordinator manages security across clusters
type DistributedSecurityCoordinator struct {
	mu                       sync.RWMutex
	clusterStates            map[string]*ClusterSecurityState
	eventQueue               chan SecurityEvent // Legacy queue for compatibility
	backpressureManager      *EventQueueBackpressureManager
	encryptionManager        *EncryptionManager
	auditLogger              audit.AuditLogger
	securityPolicies         map[string]SecurityPolicy
	threatIntelligence       ThreatIntelligenceProvider
	responseHandlers         map[SecurityEventType]ResponseHandler
	crossClusterChannels     map[string]SecureCommunicationChannel
	running                  bool
	ctx                      context.Context
	cancel                   context.CancelFunc
	logger                   *slog.Logger
}

// SecurityPolicy defines security policy structure
type SecurityPolicy struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	Rules            []PolicyRule           `json:"rules"`
	EnforcementLevel string                 `json:"enforcement_level"`
	Scope            []string               `json:"scope"`
	Actions          []PolicyAction         `json:"actions"`
	Metadata         map[string]interface{} `json:"metadata"`
}

// PolicyRule defines individual policy rules
type PolicyRule struct {
	ID          string                 `json:"id"`
	Condition   string                 `json:"condition"`
	Action      string                 `json:"action"`
	Severity    ThreatSeverity         `json:"severity"`
	Parameters  map[string]interface{} `json:"parameters"`
	Enabled     bool                   `json:"enabled"`
}

// PolicyAction defines actions for policy violations
type PolicyAction struct {
	Type       string                 `json:"type"`
	Parameters map[string]interface{} `json:"parameters"`
	Timeout    time.Duration          `json:"timeout"`
}

// ThreatIntelligenceProvider interface for threat intelligence
type ThreatIntelligenceProvider interface {
	GetThreatLevel(clusterID string) (ThreatSeverity, error)
	GetActiveThreats(clusterID string) ([]SecurityEvent, error)
	ReportThreat(event SecurityEvent) error
	SubscribeToFeeds(ctx context.Context) (<-chan SecurityEvent, error)
}

// ResponseHandler interface for handling security events
type ResponseHandler interface {
	Handle(ctx context.Context, event SecurityEvent) error
	CanHandle(event SecurityEvent) bool
	Priority() int
}

// SecureCommunicationChannel interface for cross-cluster communication
type SecureCommunicationChannel interface {
	Send(ctx context.Context, event SecurityEvent) error
	Receive(ctx context.Context) (<-chan SecurityEvent, error)
	Close() error
	IsHealthy() bool
	GetTLSConfig() *tls.Config
}

// NewDistributedSecurityCoordinator creates a new coordinator
func NewDistributedSecurityCoordinator(encMgr *EncryptionManager, auditLogger audit.AuditLogger) *DistributedSecurityCoordinator {
	ctx, cancel := context.WithCancel(context.Background())

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	// Initialize backpressure manager with robust configuration
	backpressureConfig := DefaultBackpressureConfig()
	backpressureManager := NewEventQueueBackpressureManager(backpressureConfig, logger)

	return &DistributedSecurityCoordinator{
		clusterStates:         make(map[string]*ClusterSecurityState),
		eventQueue:           make(chan SecurityEvent, 10000), // Legacy queue for compatibility
		backpressureManager:  backpressureManager,
		encryptionManager:    encMgr,
		auditLogger:         auditLogger,
		securityPolicies:    make(map[string]SecurityPolicy),
		responseHandlers:    make(map[SecurityEventType]ResponseHandler),
		crossClusterChannels: make(map[string]SecureCommunicationChannel),
		ctx:                 ctx,
		cancel:              cancel,
		logger:              logger,
	}
}

// Start begins the distributed security coordination
func (dsc *DistributedSecurityCoordinator) Start() error {
	dsc.mu.Lock()
	if dsc.running {
		dsc.mu.Unlock()
		return fmt.Errorf("coordinator already running")
	}
	dsc.running = true
	dsc.mu.Unlock()

	// Start the backpressure manager first
	if err := dsc.backpressureManager.Start(); err != nil {
		return fmt.Errorf("failed to start backpressure manager: %w", err)
	}

	// Start event processing (legacy queue for compatibility)
	go dsc.processEvents()

	// Start cross-cluster synchronization
	go dsc.syncClusters()

	// Start threat intelligence updates
	go dsc.updateThreatIntelligence()

	dsc.logger.Info("Distributed security coordinator started with robust backpressure handling")
	return nil
}

// Stop stops the coordinator
func (dsc *DistributedSecurityCoordinator) Stop() error {
	dsc.mu.Lock()
	if !dsc.running {
		dsc.mu.Unlock()
		return nil
	}
	dsc.running = false
	dsc.mu.Unlock()

	dsc.cancel()

	// Stop backpressure manager first
	if err := dsc.backpressureManager.Stop(); err != nil {
		dsc.logger.Error("Error stopping backpressure manager", "error", err)
	}

	// Close legacy queue
	close(dsc.eventQueue)

	for _, channel := range dsc.crossClusterChannels {
		channel.Close()
	}

	dsc.logger.Info("Distributed security coordinator stopped")
	return nil
}

// ProcessSecurityEvent processes a security event with robust backpressure handling
func (dsc *DistributedSecurityCoordinator) ProcessSecurityEvent(event SecurityEvent) error {
	// Add timestamp if not set
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}

	// Sign the event
	signature, err := dsc.encryptionManager.SignMessage([]byte(fmt.Sprintf("%+v", event)))
	if err != nil {
		return fmt.Errorf("failed to sign event: %w", err)
	}
	event.Signature = base64.StdEncoding.EncodeToString(signature)

	// Use the robust backpressure manager
	if err := dsc.backpressureManager.EnqueueEvent(event); err != nil {
		// If backpressure manager fails, fall back to legacy queue for critical events
		if event.Severity == SeverityCritical {
			select {
			case dsc.eventQueue <- event:
				securityEventsTotal.WithLabelValues(string(event.Type), string(event.Severity), event.ClusterID).Inc()
				dsc.logger.Warn("Used legacy queue for critical event due to backpressure", "event_id", event.ID)
			default:
				securityEventsDropped.Inc()
				dsc.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
					Action:   "delete",
					Resource: "security_event_queue",
					Result:   "failure",
					Details:   map[string]interface{}{"description": fmt.Sprintf("Critical security event dropped due to backpressure: %s", event.ID), "event_id": event.ID, "event_type": string(event.Type)},
				})
				return fmt.Errorf("critical event dropped - both queues full")
			}
		} else {
			// Non-critical event handling by backpressure manager (may be spilled, throttled, etc.)
			dsc.logger.Debug("Event handled by backpressure manager", "event_id", event.ID, "error", err)
		}
	} else {
		// Successfully enqueued in robust queue
		securityEventsTotal.WithLabelValues(string(event.Type), string(event.Severity), event.ClusterID).Inc()
	}

	// Audit log the event
	dsc.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
		UserID:   event.NodeID,
		Action:   "read",
		Resource: "security_event",
		Result:   "success",
		Details:   map[string]interface{}{"description": fmt.Sprintf("Security event processed: %s", string(event.Type)), "event": event},
	})

	return nil
}

// processEvents handles events from the queue
func (dsc *DistributedSecurityCoordinator) processEvents() {
	for event := range dsc.eventQueue {
		if err := dsc.handleSecurityEvent(event); err != nil {
			log.Printf("Error handling security event: %v", err)
		}
	}
}

// handleSecurityEvent processes individual security events
func (dsc *DistributedSecurityCoordinator) handleSecurityEvent(event SecurityEvent) error {
	// Update cluster state
	dsc.updateClusterState(event)

	// Check policies
	if err := dsc.enforcePolicies(event); err != nil {
		log.Printf("Policy enforcement error: %v", err)
	}

	// Execute response handlers
	if handler, exists := dsc.responseHandlers[event.Type]; exists {
		if err := handler.Handle(dsc.ctx, event); err != nil {
			log.Printf("Response handler error: %v", err)
		}
	}

	// Propagate to other clusters if needed
	if !event.Propagated && dsc.shouldPropagateEvent(event) {
		if err := dsc.propagateEvent(event); err != nil {
			log.Printf("Event propagation error: %v", err)
		}
	}

	return nil
}

// updateClusterState updates the security state for a cluster
func (dsc *DistributedSecurityCoordinator) updateClusterState(event SecurityEvent) {
	dsc.mu.Lock()
	defer dsc.mu.Unlock()

	state, exists := dsc.clusterStates[event.ClusterID]
	if !exists {
		state = &ClusterSecurityState{
			ClusterID:       event.ClusterID,
			ActiveThreats:   []SecurityEvent{},
			SecurityMetrics: make(map[string]float64),
			Policies:        make(map[string]interface{}),
		}
		dsc.clusterStates[event.ClusterID] = state
	}

	state.LastUpdate = time.Now()

	// Update threat level based on event severity
	if event.Severity == SeverityCritical {
		state.ThreatLevel = SeverityCritical
	} else if state.ThreatLevel != SeverityCritical && event.Severity == SeverityHigh {
		state.ThreatLevel = SeverityHigh
	}

	// Add to active threats if significant
	if event.Severity == SeverityCritical || event.Severity == SeverityHigh {
		state.ActiveThreats = append(state.ActiveThreats, event)

		// Keep only recent threats (last 24 hours)
		cutoff := time.Now().Add(-24 * time.Hour)
		activeThreats := []SecurityEvent{}
		for _, threat := range state.ActiveThreats {
			if threat.Timestamp.After(cutoff) {
				activeThreats = append(activeThreats, threat)
			}
		}
		state.ActiveThreats = activeThreats
	}
}

// enforcePolicies checks and enforces security policies
func (dsc *DistributedSecurityCoordinator) enforcePolicies(event SecurityEvent) error {
	for _, policy := range dsc.securityPolicies {
		for _, rule := range policy.Rules {
			if rule.Enabled && dsc.matchesRule(event, rule) {
				if err := dsc.executeAction(policy.Actions, event); err != nil {
					return fmt.Errorf("failed to execute policy action: %w", err)
				}
			}
		}
	}
	return nil
}

// matchesRule checks if an event matches a policy rule
func (dsc *DistributedSecurityCoordinator) matchesRule(event SecurityEvent, rule PolicyRule) bool {
	// Simple condition matching - in real implementation, this would be more sophisticated
	return string(event.Type) == rule.Condition || string(event.Severity) == rule.Condition
}

// executeAction executes policy actions
func (dsc *DistributedSecurityCoordinator) executeAction(actions []PolicyAction, event SecurityEvent) error {
	for _, action := range actions {
		switch action.Type {
		case "quarantine":
			if err := dsc.quarantineCluster(event.ClusterID); err != nil {
				return err
			}
		case "alert":
			// Send alert
			log.Printf("SECURITY ALERT: %s on cluster %s", string(event.Type), event.ClusterID)
		case "block":
			// Block the source
			log.Printf("BLOCKING source: %s", event.Source)
		}
	}
	return nil
}

// quarantineCluster quarantines a cluster
func (dsc *DistributedSecurityCoordinator) quarantineCluster(clusterID string) error {
	dsc.mu.Lock()
	defer dsc.mu.Unlock()

	if state, exists := dsc.clusterStates[clusterID]; exists {
		state.Quarantined = true

		dsc.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
			Action:   "update",
			Resource: "cluster",
			Result:   "success",
			Details:   map[string]interface{}{"description": fmt.Sprintf("Cluster %s has been quarantined", clusterID), "cluster_id": clusterID},
		})
	}

	return nil
}

// shouldPropagateEvent determines if an event should be propagated to other clusters
func (dsc *DistributedSecurityCoordinator) shouldPropagateEvent(event SecurityEvent) bool {
	// Propagate critical events and cross-cluster threats
	return event.Severity == SeverityCritical ||
		   event.Type == EventTypeCrossCusterThreat ||
		   event.Type == EventTypeSecurityBreach
}

// propagateEvent propagates an event to other clusters
func (dsc *DistributedSecurityCoordinator) propagateEvent(event SecurityEvent) error {
	event.Propagated = true

	for clusterID, channel := range dsc.crossClusterChannels {
		if clusterID != event.ClusterID {
			timer := prometheus.NewTimer(crossClusterLatency.WithLabelValues("propagate_event", clusterID))
			err := channel.Send(dsc.ctx, event)
			timer.ObserveDuration()

			if err != nil {
				log.Printf("Failed to propagate event to cluster %s: %v", clusterID, err)
			}
		}
	}

	return nil
}

// syncClusters synchronizes security state across clusters
func (dsc *DistributedSecurityCoordinator) syncClusters() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dsc.performCrossClusterSync()
		case <-dsc.ctx.Done():
			return
		}
	}
}

// performCrossClusterSync performs cross-cluster synchronization
func (dsc *DistributedSecurityCoordinator) performCrossClusterSync() {
	dsc.mu.RLock()
	states := make(map[string]*ClusterSecurityState)
	for k, v := range dsc.clusterStates {
		// Deep copy for thread safety
		stateCopy := *v
		stateCopy.ActiveThreats = make([]SecurityEvent, len(v.ActiveThreats))
		copy(stateCopy.ActiveThreats, v.ActiveThreats)
		states[k] = &stateCopy
	}
	dsc.mu.RUnlock()

	// Sync with other clusters
	for clusterID, channel := range dsc.crossClusterChannels {
		if state, exists := states[clusterID]; exists && channel.IsHealthy() {
			// Send state update
			event := SecurityEvent{
				ID:        fmt.Sprintf("sync_%s_%d", clusterID, time.Now().Unix()),
				Type:      "cluster_state_sync",
				Data:      map[string]interface{}{"state": state},
				Timestamp: time.Now(),
				ClusterID: clusterID,
			}

			timer := prometheus.NewTimer(crossClusterLatency.WithLabelValues("sync_state", clusterID))
			channel.Send(dsc.ctx, event)
			timer.ObserveDuration()
		}
	}
}

// updateThreatIntelligence updates threat intelligence data
func (dsc *DistributedSecurityCoordinator) updateThreatIntelligence() {
	if dsc.threatIntelligence == nil {
		return
	}

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			dsc.refreshThreatIntelligence()
		case <-dsc.ctx.Done():
			return
		}
	}
}

// refreshThreatIntelligence refreshes threat intelligence data
func (dsc *DistributedSecurityCoordinator) refreshThreatIntelligence() {
	dsc.mu.RLock()
	clusterIDs := make([]string, 0, len(dsc.clusterStates))
	for clusterID := range dsc.clusterStates {
		clusterIDs = append(clusterIDs, clusterID)
	}
	dsc.mu.RUnlock()

	for _, clusterID := range clusterIDs {
		if level, err := dsc.threatIntelligence.GetThreatLevel(clusterID); err == nil {
			dsc.mu.Lock()
			if state, exists := dsc.clusterStates[clusterID]; exists {
				state.ThreatLevel = level
			}
			dsc.mu.Unlock()
		}

		if threats, err := dsc.threatIntelligence.GetActiveThreats(clusterID); err == nil {
			for _, threat := range threats {
				dsc.ProcessSecurityEvent(threat)
			}
		}
	}
}

// GetClusterSecurityState returns the security state for a cluster
func (dsc *DistributedSecurityCoordinator) GetClusterSecurityState(clusterID string) (*ClusterSecurityState, error) {
	dsc.mu.RLock()
	defer dsc.mu.RUnlock()

	state, exists := dsc.clusterStates[clusterID]
	if !exists {
		return nil, fmt.Errorf("cluster %s not found", clusterID)
	}

	// Return a copy to avoid race conditions
	stateCopy := *state
	stateCopy.ActiveThreats = make([]SecurityEvent, len(state.ActiveThreats))
	copy(stateCopy.ActiveThreats, state.ActiveThreats)

	return &stateCopy, nil
}

// GetAllClusterStates returns all cluster security states
func (dsc *DistributedSecurityCoordinator) GetAllClusterStates() map[string]*ClusterSecurityState {
	dsc.mu.RLock()
	defer dsc.mu.RUnlock()

	states := make(map[string]*ClusterSecurityState)
	for k, v := range dsc.clusterStates {
		stateCopy := *v
		stateCopy.ActiveThreats = make([]SecurityEvent, len(v.ActiveThreats))
		copy(stateCopy.ActiveThreats, v.ActiveThreats)
		states[k] = &stateCopy
	}

	return states
}

// AddSecurityPolicy adds a new security policy
func (dsc *DistributedSecurityCoordinator) AddSecurityPolicy(policy SecurityPolicy) error {
	dsc.mu.Lock()
	defer dsc.mu.Unlock()

	dsc.securityPolicies[policy.ID] = policy

	dsc.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
		Action:   "write",
		Resource: "security_policy",
		Result:   "success",
		Details:   map[string]interface{}{"description": fmt.Sprintf("Security policy added: %s", policy.Name), "policy": policy},
	})

	return nil
}

// RegisterResponseHandler registers a response handler for security events
func (dsc *DistributedSecurityCoordinator) RegisterResponseHandler(eventType SecurityEventType, handler ResponseHandler) {
	dsc.mu.Lock()
	defer dsc.mu.Unlock()

	dsc.responseHandlers[eventType] = handler
}

// AddClusterChannel adds a secure communication channel for a cluster
func (dsc *DistributedSecurityCoordinator) AddClusterChannel(clusterID string, channel SecureCommunicationChannel) error {
	dsc.mu.Lock()
	defer dsc.mu.Unlock()

	dsc.crossClusterChannels[clusterID] = channel
	return nil
}

// GetBackpressureStatus returns the current backpressure manager status
func (dsc *DistributedSecurityCoordinator) GetBackpressureStatus() map[string]interface{} {
	if dsc.backpressureManager == nil {
		return map[string]interface{}{
			"enabled": false,
			"status":  "disabled",
		}
	}

	status := dsc.backpressureManager.GetStatus()
	status["enabled"] = true

	// Add spill manager stats if available
	if dsc.backpressureManager.priorityQueue != nil && dsc.backpressureManager.priorityQueue.spillMgr != nil {
		status["spill_stats"] = dsc.backpressureManager.priorityQueue.spillMgr.GetSpillStats()
	}

	return status
}

// GetSystemHealth returns overall system health including backpressure status
func (dsc *DistributedSecurityCoordinator) GetSystemHealth() map[string]interface{} {
	health := map[string]interface{}{
		"coordinator_running": dsc.running,
		"cluster_count":       len(dsc.clusterStates),
		"policy_count":        len(dsc.securityPolicies),
		"handler_count":       len(dsc.responseHandlers),
		"channel_count":       len(dsc.crossClusterChannels),
	}

	// Add backpressure health
	backpressureStatus := dsc.GetBackpressureStatus()
	health["backpressure"] = backpressureStatus

	// Calculate overall health status
	healthy := true
	issues := []string{}

	if !dsc.running {
		healthy = false
		issues = append(issues, "coordinator not running")
	}

	if status, ok := backpressureStatus["running"].(bool); ok && !status {
		healthy = false
		issues = append(issues, "backpressure manager not running")
	}

	health["healthy"] = healthy
	health["issues"] = issues

	return health
}