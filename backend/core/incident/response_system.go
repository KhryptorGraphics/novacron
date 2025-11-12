package incident

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Severity levels for incident classification
type Severity string

const (
	SeverityP0 Severity = "P0" // Critical - Total service failure
	SeverityP1 Severity = "P1" // High - Significant degradation
	SeverityP2 Severity = "P2" // Medium - Partial degradation
	SeverityP3 Severity = "P3" // Low - Minor issues
	SeverityP4 Severity = "P4" // Info - Monitoring only
)

// FailureType categorizes the type of failure
type FailureType string

const (
	FailureServiceDown          FailureType = "service_failure"
	FailureByzantine            FailureType = "byzantine_behavior"
	FailureNetworkPartition     FailureType = "network_partition"
	FailurePerformanceDegradation FailureType = "performance_degradation"
	FailureResourceExhaustion    FailureType = "resource_exhaustion"
	FailureDatabaseFailure       FailureType = "database_failure"
	FailureConsensusLost         FailureType = "consensus_lost"
)

// Incident represents a production incident
type Incident struct {
	ID          string       `json:"id"`
	Severity    Severity     `json:"severity"`
	Component   string       `json:"component"`
	FailureType FailureType  `json:"failure_type"`
	Description string       `json:"description"`
	ImpactScore int          `json:"impact_score"`
	StartTime   time.Time    `json:"start_time"`
	EndTime     *time.Time   `json:"end_time,omitempty"`
	Status      string       `json:"status"` // active, remediating, resolved
	Remediation *Remediation `json:"remediation,omitempty"`
	Metrics     Metrics      `json:"metrics"`
}

// Remediation tracks automated remediation attempts
type Remediation struct {
	Strategy     string    `json:"strategy"`
	Attempts     int       `json:"attempts"`
	MaxAttempts  int       `json:"max_attempts"`
	Success      bool      `json:"success"`
	StartTime    time.Time `json:"start_time"`
	EndTime      *time.Time `json:"end_time,omitempty"`
	Actions      []string  `json:"actions"`
	FailureReason string   `json:"failure_reason,omitempty"`
}

// Metrics captures incident-related metrics
type Metrics struct {
	ErrorRate     float64 `json:"error_rate"`
	LatencyP99    int64   `json:"latency_p99_ms"`
	LatencyP50    int64   `json:"latency_p50_ms"`
	CPUPercent    float64 `json:"cpu_percent"`
	MemoryPercent float64 `json:"memory_percent"`
	RequestRate   float64 `json:"request_rate"`
	SuccessRate   float64 `json:"success_rate"`
}

// ResponseSystem manages incident detection and response
type ResponseSystem struct {
	mu                sync.RWMutex
	incidents         map[string]*Incident
	activeIncidents   map[string]*Incident
	responseSLAs      map[Severity]time.Duration
	notificationQueue chan *Incident
	ctx               context.Context
	cancel            context.CancelFunc
	detectors         []Detector
	remediators       map[FailureType]Remediator
	metricsCollector  MetricsCollector
}

// Detector interface for failure detection
type Detector interface {
	Detect(ctx context.Context) (*DetectionResult, error)
	Name() string
}

// DetectionResult contains detection information
type DetectionResult struct {
	Component   string
	FailureType FailureType
	Description string
	Metrics     Metrics
	Confidence  float64
}

// Remediator interface for automated remediation
type Remediator interface {
	Remediate(ctx context.Context, incident *Incident) error
	CanRemediate(incident *Incident) bool
	MaxAttempts() int
}

// MetricsCollector interface for collecting system metrics
type MetricsCollector interface {
	CollectCurrent(ctx context.Context) (*Metrics, error)
}

// NewResponseSystem creates a new incident response system
func NewResponseSystem(ctx context.Context) *ResponseSystem {
	ctx, cancel := context.WithCancel(ctx)

	rs := &ResponseSystem{
		incidents:         make(map[string]*Incident),
		activeIncidents:   make(map[string]*Incident),
		notificationQueue: make(chan *Incident, 100),
		ctx:               ctx,
		cancel:            cancel,
		detectors:         make([]Detector, 0),
		remediators:       make(map[FailureType]Remediator),
		responseSLAs:      makeDefaultSLAs(),
	}

	return rs
}

// makeDefaultSLAs creates default response SLAs
func makeDefaultSLAs() map[Severity]time.Duration {
	return map[Severity]time.Duration{
		SeverityP0: 30 * time.Second,
		SeverityP1: 60 * time.Second,
		SeverityP2: 5 * time.Minute,
		SeverityP3: 15 * time.Minute,
		SeverityP4: 60 * time.Minute,
	}
}

// RegisterDetector adds a detector to the system
func (rs *ResponseSystem) RegisterDetector(detector Detector) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.detectors = append(rs.detectors, detector)
}

// RegisterRemediator adds a remediator for a failure type
func (rs *ResponseSystem) RegisterRemediator(failureType FailureType, remediator Remediator) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.remediators[failureType] = remediator
}

// SetMetricsCollector sets the metrics collector
func (rs *ResponseSystem) SetMetricsCollector(collector MetricsCollector) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	rs.metricsCollector = collector
}

// Start begins incident monitoring
func (rs *ResponseSystem) Start() error {
	go rs.detectionLoop()
	go rs.notificationLoop()
	return nil
}

// Stop halts incident monitoring
func (rs *ResponseSystem) Stop() error {
	rs.cancel()
	return nil
}

// detectionLoop continuously monitors for incidents
func (rs *ResponseSystem) detectionLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-rs.ctx.Done():
			return
		case <-ticker.C:
			rs.runDetection()
		}
	}
}

// runDetection executes all registered detectors
func (rs *ResponseSystem) runDetection() {
	rs.mu.RLock()
	detectors := rs.detectors
	rs.mu.RUnlock()

	for _, detector := range detectors {
		go func(d Detector) {
			result, err := d.Detect(rs.ctx)
			if err != nil {
				// Log detection error
				return
			}

			if result != nil {
				rs.handleDetection(result)
			}
		}(detector)
	}
}

// handleDetection processes a detection result
func (rs *ResponseSystem) handleDetection(result *DetectionResult) {
	// Calculate impact score
	impactScore := rs.calculateImpactScore(result.Component, result.FailureType)

	// Classify incident
	severity := rs.classifyIncident(result.Component, result.FailureType, impactScore)

	// Create incident
	incident := &Incident{
		ID:          generateIncidentID(),
		Severity:    severity,
		Component:   result.Component,
		FailureType: result.FailureType,
		Description: result.Description,
		ImpactScore: impactScore,
		StartTime:   time.Now(),
		Status:      "active",
		Metrics:     result.Metrics,
	}

	// Store incident
	rs.mu.Lock()
	rs.incidents[incident.ID] = incident
	rs.activeIncidents[incident.ID] = incident
	rs.mu.Unlock()

	// Queue notification
	select {
	case rs.notificationQueue <- incident:
	default:
		// Queue full, log warning
	}

	// Attempt remediation for critical incidents
	if severity == SeverityP0 || severity == SeverityP1 {
		go rs.attemptRemediation(incident)
	}
}

// calculateImpactScore determines the impact of a failure
func (rs *ResponseSystem) calculateImpactScore(component string, failureType FailureType) int {
	baseScore := 0

	// Component criticality scores
	switch component {
	case "consensus", "core":
		baseScore = 100
	case "database", "network":
		baseScore = 80
	case "api", "scheduler":
		baseScore = 60
	case "monitoring", "logging":
		baseScore = 40
	default:
		baseScore = 20
	}

	// Failure type modifiers
	modifier := 1.0
	switch failureType {
	case FailureServiceDown:
		modifier = 1.0
	case FailureByzantine:
		modifier = 0.9
	case FailureNetworkPartition:
		modifier = 0.85
	case FailurePerformanceDegradation:
		modifier = 0.6
	case FailureResourceExhaustion:
		modifier = 0.5
	default:
		modifier = 0.3
	}

	return int(float64(baseScore) * modifier)
}

// classifyIncident determines the severity level
func (rs *ResponseSystem) classifyIncident(component string, failureType FailureType, impactScore int) Severity {
	// P0: Total service failure
	if failureType == FailureServiceDown && (component == "core" || component == "consensus") {
		return SeverityP0
	}

	// P1: Critical component degradation
	if failureType == FailureByzantine || failureType == FailureNetworkPartition {
		return SeverityP1
	}

	// P2: Performance degradation
	if failureType == FailurePerformanceDegradation && impactScore > 50 {
		return SeverityP2
	}

	// P3: Resource issues
	if failureType == FailureResourceExhaustion {
		return SeverityP3
	}

	// P4: Minor issues
	return SeverityP4
}

// attemptRemediation tries to automatically remediate an incident
func (rs *ResponseSystem) attemptRemediation(incident *Incident) {
	rs.mu.RLock()
	remediator, exists := rs.remediators[incident.FailureType]
	rs.mu.RUnlock()

	if !exists || !remediator.CanRemediate(incident) {
		return
	}

	// Initialize remediation tracking
	incident.Remediation = &Remediation{
		Strategy:    string(incident.FailureType),
		Attempts:    0,
		MaxAttempts: remediator.MaxAttempts(),
		StartTime:   time.Now(),
		Actions:     make([]string, 0),
	}

	// Update incident status
	rs.mu.Lock()
	incident.Status = "remediating"
	rs.mu.Unlock()

	// Attempt remediation with retries
	for attempt := 1; attempt <= remediator.MaxAttempts(); attempt++ {
		incident.Remediation.Attempts = attempt

		err := remediator.Remediate(rs.ctx, incident)
		if err == nil {
			// Remediation successful
			now := time.Now()
			incident.Remediation.Success = true
			incident.Remediation.EndTime = &now
			incident.Status = "resolved"
			incident.EndTime = &now

			rs.mu.Lock()
			delete(rs.activeIncidents, incident.ID)
			rs.mu.Unlock()

			// Send success notification
			select {
			case rs.notificationQueue <- incident:
			default:
			}

			return
		}

		// Remediation failed, wait before retry
		if attempt < remediator.MaxAttempts() {
			time.Sleep(time.Duration(attempt*5) * time.Second)
		}
	}

	// All remediation attempts failed
	now := time.Now()
	incident.Remediation.EndTime = &now
	incident.Remediation.Success = false
	incident.Remediation.FailureReason = "Max attempts exceeded"
	incident.Status = "active"

	// Send failure notification
	select {
	case rs.notificationQueue <- incident:
	default:
	}
}

// notificationLoop processes incident notifications
func (rs *ResponseSystem) notificationLoop() {
	for {
		select {
		case <-rs.ctx.Done():
			return
		case incident := <-rs.notificationQueue:
			rs.sendNotification(incident)
		}
	}
}

// sendNotification sends incident notification
func (rs *ResponseSystem) sendNotification(incident *Incident) {
	// Implementation would send to notification service
	// (PagerDuty, Slack, email, etc.)

	payload, _ := json.Marshal(incident)
	_ = payload // Use payload to send notification
}

// GetActiveIncidents returns all currently active incidents
func (rs *ResponseSystem) GetActiveIncidents() []*Incident {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	incidents := make([]*Incident, 0, len(rs.activeIncidents))
	for _, incident := range rs.activeIncidents {
		incidents = append(incidents, incident)
	}

	return incidents
}

// GetIncident retrieves an incident by ID
func (rs *ResponseSystem) GetIncident(id string) (*Incident, error) {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	incident, exists := rs.incidents[id]
	if !exists {
		return nil, fmt.Errorf("incident not found: %s", id)
	}

	return incident, nil
}

// GetIncidentStats returns incident statistics
func (rs *ResponseSystem) GetIncidentStats() map[string]interface{} {
	rs.mu.RLock()
	defer rs.mu.RUnlock()

	stats := make(map[string]interface{})

	// Count by severity
	severityCounts := make(map[Severity]int)
	for _, incident := range rs.incidents {
		severityCounts[incident.Severity]++
	}

	stats["total_incidents"] = len(rs.incidents)
	stats["active_incidents"] = len(rs.activeIncidents)
	stats["severity_counts"] = severityCounts

	// Calculate MTTR (Mean Time To Remediate)
	var totalRemediationTime time.Duration
	resolvedCount := 0

	for _, incident := range rs.incidents {
		if incident.EndTime != nil {
			totalRemediationTime += incident.EndTime.Sub(incident.StartTime)
			resolvedCount++
		}
	}

	if resolvedCount > 0 {
		stats["mttr_seconds"] = totalRemediationTime.Seconds() / float64(resolvedCount)
	}

	// Calculate success rate
	successCount := 0
	for _, incident := range rs.incidents {
		if incident.Remediation != nil && incident.Remediation.Success {
			successCount++
		}
	}

	if len(rs.incidents) > 0 {
		stats["remediation_success_rate"] = float64(successCount) / float64(len(rs.incidents))
	}

	return stats
}

// generateIncidentID creates a unique incident ID
func generateIncidentID() string {
	return fmt.Sprintf("INC-%d", time.Now().UnixNano())
}

// =============================================================================
// Built-in Detectors
// =============================================================================

// ServiceFailureDetector detects service failures
type ServiceFailureDetector struct {
	component string
	healthURL string
}

func NewServiceFailureDetector(component, healthURL string) *ServiceFailureDetector {
	return &ServiceFailureDetector{
		component: component,
		healthURL: healthURL,
	}
}

func (d *ServiceFailureDetector) Name() string {
	return fmt.Sprintf("service_failure_%s", d.component)
}

func (d *ServiceFailureDetector) Detect(ctx context.Context) (*DetectionResult, error) {
	// Implementation would check service health
	// This is a simplified example

	// If service is down, return detection result
	// return &DetectionResult{
	//     Component:   d.component,
	//     FailureType: FailureServiceDown,
	//     Description: "Service is not responding",
	//     Confidence:  1.0,
	// }, nil

	return nil, nil // No failure detected
}

// PerformanceDegradationDetector detects performance issues
type PerformanceDegradationDetector struct {
	component        string
	latencyThreshold time.Duration
}

func NewPerformanceDegradationDetector(component string, latencyThreshold time.Duration) *PerformanceDegradationDetector {
	return &PerformanceDegradationDetector{
		component:        component,
		latencyThreshold: latencyThreshold,
	}
}

func (d *PerformanceDegradationDetector) Name() string {
	return fmt.Sprintf("performance_degradation_%s", d.component)
}

func (d *PerformanceDegradationDetector) Detect(ctx context.Context) (*DetectionResult, error) {
	// Implementation would measure latency
	return nil, nil
}

// =============================================================================
// Built-in Remediators
// =============================================================================

// ServiceRestartRemediator restarts failed services
type ServiceRestartRemediator struct {
	maxAttempts int
}

func NewServiceRestartRemediator() *ServiceRestartRemediator {
	return &ServiceRestartRemediator{
		maxAttempts: 3,
	}
}

func (r *ServiceRestartRemediator) CanRemediate(incident *Incident) bool {
	return incident.FailureType == FailureServiceDown
}

func (r *ServiceRestartRemediator) MaxAttempts() int {
	return r.maxAttempts
}

func (r *ServiceRestartRemediator) Remediate(ctx context.Context, incident *Incident) error {
	// Implementation would restart the service
	incident.Remediation.Actions = append(incident.Remediation.Actions, "restart_service")
	return nil
}

// CacheClearRemediator clears caches to resolve performance issues
type CacheClearRemediator struct{}

func NewCacheClearRemediator() *CacheClearRemediator {
	return &CacheClearRemediator{}
}

func (r *CacheClearRemediator) CanRemediate(incident *Incident) bool {
	return incident.FailureType == FailurePerformanceDegradation
}

func (r *CacheClearRemediator) MaxAttempts() int {
	return 1
}

func (r *CacheClearRemediator) Remediate(ctx context.Context, incident *Incident) error {
	// Implementation would clear caches
	incident.Remediation.Actions = append(incident.Remediation.Actions, "clear_cache")
	return nil
}

// ByzantineIsolationRemediator isolates Byzantine nodes
type ByzantineIsolationRemediator struct{}

func NewByzantineIsolationRemediator() *ByzantineIsolationRemediator {
	return &ByzantineIsolationRemediator{}
}

func (r *ByzantineIsolationRemediator) CanRemediate(incident *Incident) bool {
	return incident.FailureType == FailureByzantine
}

func (r *ByzantineIsolationRemediator) MaxAttempts() int {
	return 1
}

func (r *ByzantineIsolationRemediator) Remediate(ctx context.Context, incident *Incident) error {
	// Implementation would isolate Byzantine node
	incident.Remediation.Actions = append(incident.Remediation.Actions, "isolate_byzantine_node")
	return nil
}
