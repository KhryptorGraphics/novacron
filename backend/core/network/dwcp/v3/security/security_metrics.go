package security

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// SecurityMetrics tracks security-related metrics and events
type SecurityMetrics struct {
	mu sync.RWMutex

	nodeID string
	logger *zap.Logger

	// Event counters (atomic)
	byzantineDetections     atomic.Uint64
	signatureValidations    atomic.Uint64
	signatureFailures       atomic.Uint64
	equivocations          atomic.Uint64
	quarantineEvents       atomic.Uint64
	modeChanges            atomic.Uint64
	tlsHandshakes          atomic.Uint64
	tlsHandshakeFailures   atomic.Uint64

	// Time-series data
	detectionHistory   []DetectionEvent
	reputationHistory  []ReputationSnapshot
	modeHistory        []ModeChangeEvent

	// Performance metrics
	validationLatency    []time.Duration
	detectionLatency     []time.Duration
	tlsHandshakeLatency  []time.Duration

	// Current state
	currentMode         SecurityMode
	activeQuarantines   int
	averageReputation   float64
	networkTrust        float64

	// Alerts
	alerts         []SecurityAlert
	alertThreshold map[AlertType]float64

	// Components
	byzantineDetector *ByzantineDetector
	reputationSystem  *ReputationSystem
	modeSecurity      *ModeAwareSecurity

	ctx    context.Context
	cancel context.CancelFunc
}

// DetectionEvent represents a Byzantine detection event
type DetectionEvent struct {
	Timestamp  time.Time
	NodeID     string
	AttackType AttackType
	Confidence float64
	Evidence   interface{}
}

// ReputationSnapshot represents reputation state at a point in time
type ReputationSnapshot struct {
	Timestamp         time.Time
	TotalNodes        int
	TrustedNodes      int
	SuspiciousNodes   int
	QuarantinedNodes  int
	AverageScore      float64
}

// ModeChangeEvent represents a security mode change
type ModeChangeEvent struct {
	Timestamp time.Time
	OldMode   SecurityMode
	NewMode   SecurityMode
	Reason    string
	Triggered string // auto or manual
}

// SecurityAlert represents a security alert
type SecurityAlert struct {
	Timestamp  time.Time
	Type       AlertType
	Severity   AlertSeverity
	Message    string
	NodeID     string
	Evidence   interface{}
	Resolved   bool
	ResolvedAt time.Time
}

// AlertType defines types of security alerts
type AlertType int

const (
	AlertByzantineDetected AlertType = iota
	AlertHighQuarantineRate
	AlertReputationDropping
	AlertModeChangeFrequent
	AlertTLSFailureSpike
	AlertSignatureFailureSpike
	AlertNetworkTrustLow
	AlertSuspiciousPattern
)

// AlertSeverity defines alert severity levels
type AlertSeverity int

const (
	SeverityLow AlertSeverity = iota
	SeverityMedium
	SeverityHigh
	SeverityCritical
)

// NewSecurityMetrics creates a new security metrics tracker
func NewSecurityMetrics(nodeID string, detector *ByzantineDetector, reputation *ReputationSystem, modeSec *ModeAwareSecurity, logger *zap.Logger) *SecurityMetrics {
	ctx, cancel := context.WithCancel(context.Background())

	sm := &SecurityMetrics{
		nodeID:            nodeID,
		logger:            logger,
		detectionHistory:  make([]DetectionEvent, 0),
		reputationHistory: make([]ReputationSnapshot, 0),
		modeHistory:       make([]ModeChangeEvent, 0),
		validationLatency: make([]time.Duration, 0),
		detectionLatency:  make([]time.Duration, 0),
		tlsHandshakeLatency: make([]time.Duration, 0),
		alerts:            make([]SecurityAlert, 0),
		alertThreshold:    DefaultAlertThresholds(),
		byzantineDetector: detector,
		reputationSystem:  reputation,
		modeSecurity:      modeSec,
		ctx:               ctx,
		cancel:            cancel,
	}

	go sm.metricsCollectionLoop()
	go sm.alertMonitoringLoop()

	return sm
}

// DefaultAlertThresholds returns default alert thresholds
func DefaultAlertThresholds() map[AlertType]float64 {
	return map[AlertType]float64{
		AlertByzantineDetected:      1.0,  // Any detection
		AlertHighQuarantineRate:     0.15, // 15% quarantine rate
		AlertReputationDropping:     5.0,  // 5 point drop in avg reputation
		AlertModeChangeFrequent:     5.0,  // 5 changes in 10 minutes
		AlertTLSFailureSpike:        0.10, // 10% failure rate
		AlertSignatureFailureSpike:  0.08, // 8% failure rate
		AlertNetworkTrustLow:        0.3,  // 30% trust
		AlertSuspiciousPattern:      3.0,  // 3 suspicious nodes
	}
}

// RecordByzantineDetection records a Byzantine detection event
func (sm *SecurityMetrics) RecordByzantineDetection(nodeID string, attackType AttackType, confidence float64, evidence interface{}) {
	sm.byzantineDetections.Add(1)

	event := DetectionEvent{
		Timestamp:  time.Now(),
		NodeID:     nodeID,
		AttackType: attackType,
		Confidence: confidence,
		Evidence:   evidence,
	}

	sm.mu.Lock()
	sm.detectionHistory = append(sm.detectionHistory, event)
	// Keep last 1000 events
	if len(sm.detectionHistory) > 1000 {
		sm.detectionHistory = sm.detectionHistory[len(sm.detectionHistory)-1000:]
	}
	sm.mu.Unlock()

	// Generate alert
	sm.generateAlert(AlertByzantineDetected, SeverityCritical,
		"Byzantine node detected", nodeID, evidence)

	sm.logger.Error("Byzantine detection recorded",
		zap.String("node_id", nodeID),
		zap.String("attack_type", attackTypeString(attackType)),
		zap.Float64("confidence", confidence),
	)
}

// RecordSignatureValidation records signature validation
func (sm *SecurityMetrics) RecordSignatureValidation(success bool, latency time.Duration) {
	sm.signatureValidations.Add(1)
	if !success {
		sm.signatureFailures.Add(1)
	}

	sm.mu.Lock()
	sm.validationLatency = append(sm.validationLatency, latency)
	if len(sm.validationLatency) > 1000 {
		sm.validationLatency = sm.validationLatency[len(sm.validationLatency)-1000:]
	}
	sm.mu.Unlock()
}

// RecordEquivocation records an equivocation event
func (sm *SecurityMetrics) RecordEquivocation(nodeID string) {
	sm.equivocations.Add(1)
	sm.logger.Warn("Equivocation recorded", zap.String("node_id", nodeID))
}

// RecordQuarantine records a quarantine event
func (sm *SecurityMetrics) RecordQuarantine(nodeID string, reason string) {
	sm.quarantineEvents.Add(1)

	sm.mu.Lock()
	sm.activeQuarantines++
	sm.mu.Unlock()

	sm.generateAlert(AlertByzantineDetected, SeverityHigh,
		"Node quarantined: "+reason, nodeID, nil)

	sm.logger.Warn("Quarantine recorded",
		zap.String("node_id", nodeID),
		zap.String("reason", reason),
	)
}

// RecordQuarantineRelease records a quarantine release
func (sm *SecurityMetrics) RecordQuarantineRelease(nodeID string) {
	sm.mu.Lock()
	if sm.activeQuarantines > 0 {
		sm.activeQuarantines--
	}
	sm.mu.Unlock()

	sm.logger.Info("Quarantine release recorded", zap.String("node_id", nodeID))
}

// RecordModeChange records a security mode change
func (sm *SecurityMetrics) RecordModeChange(oldMode, newMode SecurityMode, reason string, triggered string) {
	sm.modeChanges.Add(1)

	event := ModeChangeEvent{
		Timestamp: time.Now(),
		OldMode:   oldMode,
		NewMode:   newMode,
		Reason:    reason,
		Triggered: triggered,
	}

	sm.mu.Lock()
	sm.modeHistory = append(sm.modeHistory, event)
	sm.currentMode = newMode
	// Keep last 100 changes
	if len(sm.modeHistory) > 100 {
		sm.modeHistory = sm.modeHistory[len(sm.modeHistory)-100:]
	}
	sm.mu.Unlock()

	sm.logger.Info("Mode change recorded",
		zap.String("old_mode", modeString(oldMode)),
		zap.String("new_mode", modeString(newMode)),
		zap.String("reason", reason),
	)
}

// RecordTLSHandshake records a TLS handshake event
func (sm *SecurityMetrics) RecordTLSHandshake(success bool, latency time.Duration) {
	sm.tlsHandshakes.Add(1)
	if !success {
		sm.tlsHandshakeFailures.Add(1)
	}

	sm.mu.Lock()
	if success {
		sm.tlsHandshakeLatency = append(sm.tlsHandshakeLatency, latency)
		if len(sm.tlsHandshakeLatency) > 1000 {
			sm.tlsHandshakeLatency = sm.tlsHandshakeLatency[len(sm.tlsHandshakeLatency)-1000:]
		}
	}
	sm.mu.Unlock()
}

// GetMetrics returns current security metrics
func (sm *SecurityMetrics) GetMetrics() map[string]interface{} {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	totalValidations := sm.signatureValidations.Load()
	totalFailures := sm.signatureFailures.Load()
	failureRate := 0.0
	if totalValidations > 0 {
		failureRate = float64(totalFailures) / float64(totalValidations)
	}

	tlsTotal := sm.tlsHandshakes.Load()
	tlsFailures := sm.tlsHandshakeFailures.Load()
	tlsFailureRate := 0.0
	if tlsTotal > 0 {
		tlsFailureRate = float64(tlsFailures) / float64(tlsTotal)
	}

	return map[string]interface{}{
		"byzantine_detections":      sm.byzantineDetections.Load(),
		"signature_validations":     totalValidations,
		"signature_failures":        totalFailures,
		"signature_failure_rate":    failureRate,
		"equivocations":             sm.equivocations.Load(),
		"quarantine_events":         sm.quarantineEvents.Load(),
		"active_quarantines":        sm.activeQuarantines,
		"mode_changes":              sm.modeChanges.Load(),
		"current_mode":              modeString(sm.currentMode),
		"tls_handshakes":            tlsTotal,
		"tls_handshake_failures":    tlsFailures,
		"tls_failure_rate":          tlsFailureRate,
		"average_reputation":        sm.averageReputation,
		"network_trust":             sm.networkTrust,
		"active_alerts":             len(sm.getActiveAlerts()),
		"avg_validation_latency_ms": sm.getAvgLatency(sm.validationLatency),
		"avg_detection_latency_ms":  sm.getAvgLatency(sm.detectionLatency),
		"avg_tls_handshake_ms":      sm.getAvgLatency(sm.tlsHandshakeLatency),
	}
}

// GetDetectionHistory returns recent detection events
func (sm *SecurityMetrics) GetDetectionHistory(limit int) []DetectionEvent {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	if limit <= 0 || limit > len(sm.detectionHistory) {
		limit = len(sm.detectionHistory)
	}

	start := len(sm.detectionHistory) - limit
	result := make([]DetectionEvent, limit)
	copy(result, sm.detectionHistory[start:])
	return result
}

// GetReputationHistory returns reputation snapshots
func (sm *SecurityMetrics) GetReputationHistory(limit int) []ReputationSnapshot {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	if limit <= 0 || limit > len(sm.reputationHistory) {
		limit = len(sm.reputationHistory)
	}

	start := len(sm.reputationHistory) - limit
	result := make([]ReputationSnapshot, limit)
	copy(result, sm.reputationHistory[start:])
	return result
}

// GetModeHistory returns mode change history
func (sm *SecurityMetrics) GetModeHistory(limit int) []ModeChangeEvent {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	if limit <= 0 || limit > len(sm.modeHistory) {
		limit = len(sm.modeHistory)
	}

	start := len(sm.modeHistory) - limit
	result := make([]ModeChangeEvent, limit)
	copy(result, sm.modeHistory[start:])
	return result
}

// GetActiveAlerts returns all active alerts
func (sm *SecurityMetrics) GetActiveAlerts() []SecurityAlert {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.getActiveAlerts()
}

// getActiveAlerts returns active alerts (without lock)
func (sm *SecurityMetrics) getActiveAlerts() []SecurityAlert {
	active := make([]SecurityAlert, 0)
	for _, alert := range sm.alerts {
		if !alert.Resolved {
			active = append(active, alert)
		}
	}
	return active
}

// generateAlert generates a security alert
func (sm *SecurityMetrics) generateAlert(alertType AlertType, severity AlertSeverity, message, nodeID string, evidence interface{}) {
	alert := SecurityAlert{
		Timestamp: time.Now(),
		Type:      alertType,
		Severity:  severity,
		Message:   message,
		NodeID:    nodeID,
		Evidence:  evidence,
		Resolved:  false,
	}

	sm.mu.Lock()
	sm.alerts = append(sm.alerts, alert)
	// Keep last 500 alerts
	if len(sm.alerts) > 500 {
		sm.alerts = sm.alerts[len(sm.alerts)-500:]
	}
	sm.mu.Unlock()

	sm.logger.Warn("Security alert generated",
		zap.String("type", alertTypeString(alertType)),
		zap.String("severity", severityString(severity)),
		zap.String("message", message),
		zap.String("node_id", nodeID),
	)
}

// metricsCollectionLoop periodically collects metrics
func (sm *SecurityMetrics) metricsCollectionLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-sm.ctx.Done():
			return
		case <-ticker.C:
			sm.collectMetrics()
		}
	}
}

// collectMetrics collects current metrics snapshot
func (sm *SecurityMetrics) collectMetrics() {
	// Collect reputation snapshot
	if sm.reputationSystem != nil {
		stats := sm.reputationSystem.GetStats()

		snapshot := ReputationSnapshot{
			Timestamp:        time.Now(),
			TotalNodes:       stats["total_nodes"].(int),
			TrustedNodes:     stats["trusted_nodes"].(int),
			SuspiciousNodes:  stats["suspicious_nodes"].(int),
			QuarantinedNodes: stats["quarantined"].(int),
			AverageScore:     stats["average_score"].(float64),
		}

		sm.mu.Lock()
		sm.reputationHistory = append(sm.reputationHistory, snapshot)
		sm.averageReputation = snapshot.AverageScore
		// Keep last 1000 snapshots
		if len(sm.reputationHistory) > 1000 {
			sm.reputationHistory = sm.reputationHistory[len(sm.reputationHistory)-1000:]
		}
		sm.mu.Unlock()
	}

	// Update network trust
	if sm.modeSecurity != nil {
		stats := sm.modeSecurity.GetStats()
		if trust, ok := stats["network_trust"].(float64); ok {
			sm.mu.Lock()
			sm.networkTrust = trust
			sm.mu.Unlock()
		}
	}
}

// alertMonitoringLoop monitors for alert conditions
func (sm *SecurityMetrics) alertMonitoringLoop() {
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-sm.ctx.Done():
			return
		case <-ticker.C:
			sm.checkAlertConditions()
		}
	}
}

// checkAlertConditions checks for alert conditions
func (sm *SecurityMetrics) checkAlertConditions() {
	// Check quarantine rate
	if sm.reputationSystem != nil {
		stats := sm.reputationSystem.GetStats()
		totalNodes := stats["total_nodes"].(int)
		quarantined := stats["quarantined"].(int)

		if totalNodes > 0 {
			quarantineRate := float64(quarantined) / float64(totalNodes)
			if quarantineRate > sm.alertThreshold[AlertHighQuarantineRate] {
				sm.generateAlert(AlertHighQuarantineRate, SeverityHigh,
					"High quarantine rate detected", "", map[string]interface{}{
						"rate":        quarantineRate,
						"quarantined": quarantined,
						"total":       totalNodes,
					})
			}
		}
	}

	// Check signature failure rate
	totalValidations := sm.signatureValidations.Load()
	totalFailures := sm.signatureFailures.Load()
	if totalValidations > 100 {
		failureRate := float64(totalFailures) / float64(totalValidations)
		if failureRate > sm.alertThreshold[AlertSignatureFailureSpike] {
			sm.generateAlert(AlertSignatureFailureSpike, SeverityMedium,
				"Signature failure spike detected", "", map[string]interface{}{
					"rate":     failureRate,
					"failures": totalFailures,
					"total":    totalValidations,
				})
		}
	}

	// Check network trust
	sm.mu.RLock()
	networkTrust := sm.networkTrust
	sm.mu.RUnlock()

	if networkTrust < sm.alertThreshold[AlertNetworkTrustLow] {
		sm.generateAlert(AlertNetworkTrustLow, SeverityMedium,
			"Low network trust detected", "", map[string]interface{}{
				"trust": networkTrust,
			})
	}
}

// getAvgLatency calculates average latency
func (sm *SecurityMetrics) getAvgLatency(latencies []time.Duration) float64 {
	if len(latencies) == 0 {
		return 0
	}

	var sum time.Duration
	for _, l := range latencies {
		sum += l
	}

	return float64(sum.Milliseconds()) / float64(len(latencies))
}

// Stop stops the security metrics tracker
func (sm *SecurityMetrics) Stop() {
	sm.cancel()
}

// Helper functions
func attackTypeString(t AttackType) string {
	types := []string{
		"Equivocation",
		"MessageTampering",
		"TimingManipulation",
		"DenialOfService",
		"ViewChangeAttack",
		"SelfishMining",
		"Collusion",
	}
	if int(t) < len(types) {
		return types[t]
	}
	return "Unknown"
}

func alertTypeString(t AlertType) string {
	types := []string{
		"ByzantineDetected",
		"HighQuarantineRate",
		"ReputationDropping",
		"ModeChangeFrequent",
		"TLSFailureSpike",
		"SignatureFailureSpike",
		"NetworkTrustLow",
		"SuspiciousPattern",
	}
	if int(t) < len(types) {
		return types[t]
	}
	return "Unknown"
}

func severityString(s AlertSeverity) string {
	severities := []string{"Low", "Medium", "High", "Critical"}
	if int(s) < len(severities) {
		return severities[s]
	}
	return "Unknown"
}

func modeString(m SecurityMode) string {
	modes := []string{"datacenter", "internet", "hybrid"}
	if int(m) < len(modes) {
		return modes[m]
	}
	return "unknown"
}
