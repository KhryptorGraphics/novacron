package security

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

// IncidentResponder handles security incident response and management
type IncidentResponder struct {
	config                *AuditConfig
	detector             *ThreatDetector
	escalationManager    *EscalationManager
	forensicsCollector   *ForensicsCollector
	recoveryManager      *RecoveryManager
	playbooks           map[string]*ResponsePlaybook
	activeIncidents     map[string]*SecurityIncident
	incidentHistory     []*SecurityIncident
	alertRules          []*AlertRule
	mutex              sync.RWMutex
}

// SecurityIncident represents a security incident
type SecurityIncident struct {
	ID                string                 `json:"id"`
	Type              IncidentType           `json:"type"`
	Severity          IncidentSeverity       `json:"severity"`
	Status            IncidentStatus         `json:"status"`
	Title             string                 `json:"title"`
	Description       string                 `json:"description"`
	Source            string                 `json:"source"`
	DetectedAt        time.Time             `json:"detected_at"`
	ReportedAt        time.Time             `json:"reported_at"`
	AcknowledgedAt    *time.Time            `json:"acknowledged_at,omitempty"`
	ResolvedAt        *time.Time            `json:"resolved_at,omitempty"`
	AssignedTo        string                 `json:"assigned_to,omitempty"`
	Reporter          string                 `json:"reporter,omitempty"`
	AffectedSystems   []string              `json:"affected_systems"`
	AffectedUsers     []string              `json:"affected_users"`
	Indicators        []ThreatIndicator     `json:"indicators"`
	Timeline          []IncidentEvent       `json:"timeline"`
	Evidence          []ForensicsEvidence   `json:"evidence"`
	Actions           []ResponseAction      `json:"actions"`
	Resolution        string                 `json:"resolution,omitempty"`
	LessonsLearned    string                 `json:"lessons_learned,omitempty"`
	TotalDowntime     time.Duration         `json:"total_downtime"`
	EstimatedCost     float64               `json:"estimated_cost"`
	Tags              []string              `json:"tags"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// IncidentType categorizes types of security incidents
type IncidentType string

const (
	IncidentTypeMalware          IncidentType = "malware"
	IncidentTypePhishing         IncidentType = "phishing"
	IncidentTypeDataBreach       IncidentType = "data_breach"
	IncidentTypeUnauthorizedAccess IncidentType = "unauthorized_access"
	IncidentTypeDenialOfService  IncidentType = "denial_of_service"
	IncidentTypeInsiderThreat    IncidentType = "insider_threat"
	IncidentTypeSystemCompromise IncidentType = "system_compromise"
	IncidentTypeDataLoss         IncidentType = "data_loss"
	IncidentTypePolicyViolation  IncidentType = "policy_violation"
	IncidentTypeVulnerabilityExploit IncidentType = "vulnerability_exploit"
)

// IncidentSeverity defines incident severity levels
type IncidentSeverity string

const (
	SeverityLow      IncidentSeverity = "low"
	SeverityMedium   IncidentSeverity = "medium"
	SeverityHigh     IncidentSeverity = "high"
	SeverityCritical IncidentSeverity = "critical"
)

// IncidentStatus tracks the status of an incident
type IncidentStatus string

const (
	StatusNew         IncidentStatus = "new"
	StatusAcknowledged IncidentStatus = "acknowledged"
	StatusInvestigating IncidentStatus = "investigating"
	StatusContained   IncidentStatus = "contained"
	StatusEradicating IncidentStatus = "eradicating"
	StatusRecovering  IncidentStatus = "recovering"
	StatusResolved    IncidentStatus = "resolved"
	StatusClosed      IncidentStatus = "closed"
)

// ThreatIndicator represents an indicator of compromise
type ThreatIndicator struct {
	ID          string              `json:"id"`
	Type        IndicatorType       `json:"type"`
	Value       string              `json:"value"`
	Confidence  float64             `json:"confidence"`
	Source      string              `json:"source"`
	FirstSeen   time.Time          `json:"first_seen"`
	LastSeen    time.Time          `json:"last_seen"`
	Context     string              `json:"context"`
	ThreatLevel ThreatLevel         `json:"threat_level"`
	Tags        []string            `json:"tags"`
	Metadata    map[string]string   `json:"metadata"`
}

// IndicatorType defines types of threat indicators
type IndicatorType string

const (
	IndicatorTypeIP       IndicatorType = "ip"
	IndicatorTypeDomain   IndicatorType = "domain"
	IndicatorTypeURL      IndicatorType = "url"
	IndicatorTypeHash     IndicatorType = "hash"
	IndicatorTypeEmail    IndicatorType = "email"
	IndicatorTypeUser     IndicatorType = "user"
	IndicatorTypeProcess  IncatorType = "process"
	IndicatorTypeRegistry IndicatorType = "registry"
	IndicatorTypeFile     IndicatorType = "file"
)

// ThreatLevel defines threat levels for indicators
type ThreatLevel string

const (
	ThreatLevelInfo     ThreatLevel = "info"
	ThreatLevelLow      ThreatLevel = "low"
	ThreatLevelMedium   ThreatLevel = "medium"
	ThreatLevelHigh     ThreatLevel = "high"
	ThreatLevelCritical ThreatLevel = "critical"
)

// IncidentEvent represents an event in the incident timeline
type IncidentEvent struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time             `json:"timestamp"`
	Type        EventType             `json:"type"`
	Actor       string                `json:"actor"`
	Description string                `json:"description"`
	Details     map[string]interface{} `json:"details"`
}

// EventType categorizes incident events
type EventType string

const (
	EventTypeDetection   EventType = "detection"
	EventTypeEscalation  EventType = "escalation"
	EventTypeInvestigation EventType = "investigation"
	EventTypeContainment EventType = "containment"
	EventTypeEradication EventType = "eradication"
	EventTypeRecovery    EventType = "recovery"
	EventTypeCommunication EventType = "communication"
)

// ResponseAction represents an action taken during incident response
type ResponseAction struct {
	ID          string                 `json:"id"`
	Type        ActionType             `json:"type"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Assignee    string                 `json:"assignee"`
	Status      ActionStatus           `json:"status"`
	Priority    ActionPriority         `json:"priority"`
	CreatedAt   time.Time             `json:"created_at"`
	DueAt       *time.Time            `json:"due_at,omitempty"`
	CompletedAt *time.Time            `json:"completed_at,omitempty"`
	Results     string                 `json:"results,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ActionType categorizes response actions
type ActionType string

const (
	ActionTypeInvestigate ActionType = "investigate"
	ActionTypeContain     ActionType = "contain"
	ActionTypeEradicate   ActionType = "eradicate"
	ActionTypeRecover     ActionType = "recover"
	ActionTypeCommunicate ActionType = "communicate"
	ActionTypeDocument    ActionType = "document"
	ActionTypePrevent     ActionType = "prevent"
)

// ActionStatus tracks the status of response actions
type ActionStatus string

const (
	ActionStatusPending    ActionStatus = "pending"
	ActionStatusInProgress ActionStatus = "in_progress"
	ActionStatusCompleted  ActionStatus = "completed"
	ActionStatusCancelled  ActionStatus = "cancelled"
	ActionStatusBlocked    ActionStatus = "blocked"
)

// ActionPriority defines action priorities
type ActionPriority string

const (
	PriorityLow    ActionPriority = "low"
	PriorityMedium ActionPriority = "medium"
	PriorityHigh   ActionPriority = "high"
	PriorityCritical ActionPriority = "critical"
)

// ForensicsEvidence represents digital forensics evidence
type ForensicsEvidence struct {
	ID           string                 `json:"id"`
	Type         EvidenceType           `json:"type"`
	Source       string                 `json:"source"`
	Description  string                 `json:"description"`
	Hash         string                 `json:"hash"`
	Size         int64                  `json:"size"`
	CollectedAt  time.Time             `json:"collected_at"`
	CollectedBy  string                 `json:"collected_by"`
	ChainOfCustody []CustodyRecord      `json:"chain_of_custody"`
	Analysis     []AnalysisResult       `json:"analysis"`
	Tags         []string               `json:"tags"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// EvidenceType categorizes types of digital evidence
type EvidenceType string

const (
	EvidenceTypeLog        EvidenceType = "log"
	EvidenceTypeNetworkCapture EvidenceType = "network_capture"
	EvidenceTypeMemoryDump EvidenceType = "memory_dump"
	EvidenceTypeDiskImage  EvidenceType = "disk_image"
	EvidenceTypeFile       EvidenceType = "file"
	EvidenceTypeDatabase   EvidenceType = "database"
	EvidenceTypeConfiguration EvidenceType = "configuration"
	EvidenceTypeScreenshot EvidenceType = "screenshot"
)

// CustodyRecord tracks evidence chain of custody
type CustodyRecord struct {
	Timestamp time.Time `json:"timestamp"`
	Action    string    `json:"action"`
	Actor     string    `json:"actor"`
	Location  string    `json:"location"`
	Notes     string    `json:"notes"`
}

// AnalysisResult represents the result of evidence analysis
type AnalysisResult struct {
	ID          string                 `json:"id"`
	Tool        string                 `json:"tool"`
	Analyst     string                 `json:"analyst"`
	Timestamp   time.Time             `json:"timestamp"`
	Findings    string                 `json:"findings"`
	Indicators  []ThreatIndicator     `json:"indicators"`
	Confidence  float64               `json:"confidence"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ResponsePlaybook defines automated response procedures
type ResponsePlaybook struct {
	ID             string                 `json:"id"`
	Name           string                 `json:"name"`
	Description    string                 `json:"description"`
	Version        string                 `json:"version"`
	IncidentTypes  []IncidentType         `json:"incident_types"`
	Triggers       []PlaybookTrigger      `json:"triggers"`
	Steps          []PlaybookStep         `json:"steps"`
	Variables      map[string]interface{} `json:"variables"`
	Prerequisites  []string               `json:"prerequisites"`
	CreatedAt      time.Time             `json:"created_at"`
	UpdatedAt      time.Time             `json:"updated_at"`
	CreatedBy      string                 `json:"created_by"`
	Approved       bool                   `json:"approved"`
	ApprovedBy     string                 `json:"approved_by,omitempty"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// PlaybookTrigger defines when a playbook should be activated
type PlaybookTrigger struct {
	Type       TriggerType            `json:"type"`
	Condition  string                 `json:"condition"`
	Threshold  float64                `json:"threshold"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// TriggerType defines types of playbook triggers
type TriggerType string

const (
	TriggerTypeIncidentType   TriggerType = "incident_type"
	TriggerTypeSeverity       TriggerType = "severity"
	TriggerTypeIndicatorCount TriggerType = "indicator_count"
	TriggerTypeAffectedSystems TriggerType = "affected_systems"
	TriggerTypeCustom         TriggerType = "custom"
)

// PlaybookStep defines a step in a response playbook
type PlaybookStep struct {
	ID             string                 `json:"id"`
	Name           string                 `json:"name"`
	Type           StepType               `json:"type"`
	Description    string                 `json:"description"`
	Action         string                 `json:"action"`
	Parameters     map[string]interface{} `json:"parameters"`
	Dependencies   []string               `json:"dependencies"`
	Timeout        time.Duration          `json:"timeout"`
	RetryCount     int                    `json:"retry_count"`
	ContinueOnFail bool                   `json:"continue_on_fail"`
	Automated      bool                   `json:"automated"`
	RequiredRole   string                 `json:"required_role"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// StepType categorizes playbook steps
type StepType string

const (
	StepTypeInvestigation  StepType = "investigation"
	StepTypeContainment    StepType = "containment"
	StepTypeEradication    StepType = "eradication"
	StepTypeRecovery       StepType = "recovery"
	StepTypeNotification   StepType = "notification"
	StepTypeDocumentation  StepType = "documentation"
	StepTypeCustom         StepType = "custom"
)

// AlertRule defines automated alert rules
type AlertRule struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Enabled     bool                   `json:"enabled"`
	Severity    IncidentSeverity       `json:"severity"`
	Conditions  []AlertCondition       `json:"conditions"`
	Actions     []AlertAction          `json:"actions"`
	Throttle    time.Duration          `json:"throttle"`
	LastTriggered *time.Time           `json:"last_triggered,omitempty"`
	TriggerCount  int                  `json:"trigger_count"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// AlertCondition defines conditions for alert triggers
type AlertCondition struct {
	Field    string      `json:"field"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
	Window   time.Duration `json:"window"`
}

// AlertAction defines actions to take when alerts trigger
type AlertAction struct {
	Type       string                 `json:"type"`
	Target     string                 `json:"target"`
	Parameters map[string]interface{} `json:"parameters"`
}

// IncidentResults aggregates incident response metrics
type IncidentResults struct {
	TotalIncidents      int                 `json:"total_incidents"`
	OpenIncidents       int                 `json:"open_incidents"`
	ResolvedIncidents   int                 `json:"resolved_incidents"`
	UnresolvedCritical  int                 `json:"unresolved_critical"`
	AverageResponseTime time.Duration       `json:"average_response_time"`
	AverageResolutionTime time.Duration     `json:"average_resolution_time"`
	IncidentsByType     map[IncidentType]int `json:"incidents_by_type"`
	IncidentsBySeverity map[IncidentSeverity]int `json:"incidents_by_severity"`
	TrendData          IncidentTrends       `json:"trend_data"`
	TopThreats         []ThreatIndicator    `json:"top_threats"`
	PlaybookEffectiveness map[string]float64 `json:"playbook_effectiveness"`
}

// IncidentTrends tracks incident trends over time
type IncidentTrends struct {
	Daily   []IncidentDataPoint `json:"daily"`
	Weekly  []IncidentDataPoint `json:"weekly"`
	Monthly []IncidentDataPoint `json:"monthly"`
}

// IncidentDataPoint represents incident data for trending
type IncidentDataPoint struct {
	Date     time.Time `json:"date"`
	Count    int       `json:"count"`
	Severity map[IncidentSeverity]int `json:"severity"`
}

// NewIncidentResponder creates a new incident responder
func NewIncidentResponder(config *AuditConfig) *IncidentResponder {
	responder := &IncidentResponder{
		config:           config,
		detector:         NewThreatDetector(config),
		escalationManager: NewEscalationManager(config),
		forensicsCollector: NewForensicsCollector(config),
		recoveryManager:   NewRecoveryManager(config),
		playbooks:        make(map[string]*ResponsePlaybook),
		activeIncidents:  make(map[string]*SecurityIncident),
		incidentHistory:  []*SecurityIncident{},
		alertRules:       []*AlertRule{},
	}
	
	responder.initializePlaybooks()
	responder.initializeAlertRules()
	
	return responder
}

// CreateIncident creates a new security incident
func (ir *IncidentResponder) CreateIncident(ctx context.Context, incidentType IncidentType, severity IncidentSeverity, title, description, source string) (*SecurityIncident, error) {
	ir.mutex.Lock()
	defer ir.mutex.Unlock()
	
	incident := &SecurityIncident{
		ID:              uuid.New().String(),
		Type:            incidentType,
		Severity:        severity,
		Status:          StatusNew,
		Title:           title,
		Description:     description,
		Source:          source,
		DetectedAt:      time.Now(),
		ReportedAt:      time.Now(),
		AffectedSystems: []string{},
		AffectedUsers:   []string{},
		Indicators:      []ThreatIndicator{},
		Timeline:        []IncidentEvent{},
		Evidence:        []ForensicsEvidence{},
		Actions:         []ResponseAction{},
		Tags:            []string{},
		Metadata:        make(map[string]interface{}),
	}
	
	// Add initial timeline event
	incident.Timeline = append(incident.Timeline, IncidentEvent{
		ID:          uuid.New().String(),
		Timestamp:   time.Now(),
		Type:        EventTypeDetection,
		Actor:       "system",
		Description: fmt.Sprintf("Incident detected: %s", title),
		Details:     map[string]interface{}{"source": source},
	})
	
	ir.activeIncidents[incident.ID] = incident
	
	// Trigger automated response
	go ir.triggerAutomatedResponse(ctx, incident)
	
	return incident, nil
}

// triggerAutomatedResponse starts automated incident response
func (ir *IncidentResponder) triggerAutomatedResponse(ctx context.Context, incident *SecurityIncident) {
	// Find applicable playbooks
	applicablePlaybooks := ir.findApplicablePlaybooks(incident)
	
	for _, playbook := range applicablePlaybooks {
		go ir.executePlaybook(ctx, incident, playbook)
	}
	
	// Handle escalation
	ir.escalationManager.EvaluateEscalation(ctx, incident)
	
	// Start forensics collection
	go ir.forensicsCollector.CollectEvidence(ctx, incident)
}

// findApplicablePlaybooks finds playbooks that match the incident
func (ir *IncidentResponder) findApplicablePlaybooks(incident *SecurityIncident) []*ResponsePlaybook {
	var applicable []*ResponsePlaybook
	
	for _, playbook := range ir.playbooks {
		if ir.playbookMatches(playbook, incident) {
			applicable = append(applicable, playbook)
		}
	}
	
	return applicable
}

// playbookMatches checks if a playbook matches an incident
func (ir *IncidentResponder) playbookMatches(playbook *ResponsePlaybook, incident *SecurityIncident) bool {
	// Check incident type match
	for _, incidentType := range playbook.IncidentTypes {
		if incidentType == incident.Type {
			return true
		}
	}
	
	// Check trigger conditions
	for _, trigger := range playbook.Triggers {
		if ir.evaluateTrigger(trigger, incident) {
			return true
		}
	}
	
	return false
}

// evaluateTrigger evaluates if a trigger condition is met
func (ir *IncidentResponder) evaluateTrigger(trigger PlaybookTrigger, incident *SecurityIncident) bool {
	switch trigger.Type {
	case TriggerTypeIncidentType:
		return string(incident.Type) == trigger.Condition
	case TriggerTypeSeverity:
		severityValue := map[IncidentSeverity]float64{
			SeverityLow:      1.0,
			SeverityMedium:   2.0,
			SeverityHigh:     3.0,
			SeverityCritical: 4.0,
		}
		return severityValue[incident.Severity] >= trigger.Threshold
	case TriggerTypeIndicatorCount:
		return float64(len(incident.Indicators)) >= trigger.Threshold
	case TriggerTypeAffectedSystems:
		return float64(len(incident.AffectedSystems)) >= trigger.Threshold
	default:
		return false
	}
}

// executePlaybook executes a response playbook
func (ir *IncidentResponder) executePlaybook(ctx context.Context, incident *SecurityIncident, playbook *ResponsePlaybook) {
	log.Printf("Executing playbook %s for incident %s", playbook.Name, incident.ID)
	
	for _, step := range playbook.Steps {
		if step.Automated {
			err := ir.executeStep(ctx, incident, &step)
			if err != nil {
				log.Printf("Playbook step %s failed: %v", step.Name, err)
				if !step.ContinueOnFail {
					break
				}
			}
		}
	}
}

// executeStep executes a single playbook step
func (ir *IncidentResponder) executeStep(ctx context.Context, incident *SecurityIncident, step *PlaybookStep) error {
	// Create response action
	action := ResponseAction{
		ID:          uuid.New().String(),
		Type:        ActionType(step.Type),
		Title:       step.Name,
		Description: step.Description,
		Status:      ActionStatusInProgress,
		Priority:    PriorityMedium,
		CreatedAt:   time.Now(),
		Metadata:    step.Parameters,
	}
	
	// Add action to incident
	ir.mutex.Lock()
	incident.Actions = append(incident.Actions, action)
	ir.mutex.Unlock()
	
	// Execute step based on type
	switch step.Type {
	case StepTypeContainment:
		return ir.executeContainmentStep(ctx, incident, step)
	case StepTypeInvestigation:
		return ir.executeInvestigationStep(ctx, incident, step)
	case StepTypeNotification:
		return ir.executeNotificationStep(ctx, incident, step)
	default:
		return fmt.Errorf("unsupported step type: %s", step.Type)
	}
}

// executeContainmentStep executes a containment step
func (ir *IncidentResponder) executeContainmentStep(ctx context.Context, incident *SecurityIncident, step *PlaybookStep) error {
	log.Printf("Executing containment step: %s", step.Name)
	
	// Add timeline event
	ir.addTimelineEvent(incident, EventTypeContainment, "system", 
		fmt.Sprintf("Executed containment step: %s", step.Name))
	
	return nil
}

// executeInvestigationStep executes an investigation step
func (ir *IncidentResponder) executeInvestigationStep(ctx context.Context, incident *SecurityIncident, step *PlaybookStep) error {
	log.Printf("Executing investigation step: %s", step.Name)
	
	// Add timeline event
	ir.addTimelineEvent(incident, EventTypeInvestigation, "system", 
		fmt.Sprintf("Executed investigation step: %s", step.Name))
	
	return nil
}

// executeNotificationStep executes a notification step
func (ir *IncidentResponder) executeNotificationStep(ctx context.Context, incident *SecurityIncident, step *PlaybookStep) error {
	log.Printf("Executing notification step: %s", step.Name)
	
	// Add timeline event
	ir.addTimelineEvent(incident, EventTypeCommunication, "system", 
		fmt.Sprintf("Executed notification step: %s", step.Name))
	
	return nil
}

// addTimelineEvent adds an event to the incident timeline
func (ir *IncidentResponder) addTimelineEvent(incident *SecurityIncident, eventType EventType, actor, description string) {
	ir.mutex.Lock()
	defer ir.mutex.Unlock()
	
	event := IncidentEvent{
		ID:          uuid.New().String(),
		Timestamp:   time.Now(),
		Type:        eventType,
		Actor:       actor,
		Description: description,
		Details:     make(map[string]interface{}),
	}
	
	incident.Timeline = append(incident.Timeline, event)
}

// Analyze performs incident analysis and generates metrics
func (ir *IncidentResponder) Analyze(ctx context.Context, level AuditLevel) (*IncidentResults, error) {
	ir.mutex.RLock()
	defer ir.mutex.RUnlock()
	
	results := &IncidentResults{
		IncidentsByType:     make(map[IncidentType]int),
		IncidentsBySeverity: make(map[IncidentSeverity]int),
		PlaybookEffectiveness: make(map[string]float64),
	}
	
	// Count active incidents
	results.OpenIncidents = len(ir.activeIncidents)
	
	// Analyze historical incidents
	for _, incident := range ir.incidentHistory {
		results.TotalIncidents++
		
		if incident.Status == StatusResolved || incident.Status == StatusClosed {
			results.ResolvedIncidents++
		}
		
		if incident.Severity == SeverityCritical && 
		   (incident.Status != StatusResolved && incident.Status != StatusClosed) {
			results.UnresolvedCritical++
		}
		
		results.IncidentsByType[incident.Type]++
		results.IncidentsBySeverity[incident.Severity]++
	}
	
	// Calculate average response and resolution times
	ir.calculateAverageTimes(results)
	
	// Generate trend data
	results.TrendData = ir.generateTrendData()
	
	// Identify top threats
	results.TopThreats = ir.getTopThreats(10)
	
	return results, nil
}

// calculateAverageTimes calculates average response and resolution times
func (ir *IncidentResponder) calculateAverageTimes(results *IncidentResults) {
	var totalResponseTime, totalResolutionTime time.Duration
	var responseCount, resolutionCount int
	
	for _, incident := range ir.incidentHistory {
		if incident.AcknowledgedAt != nil {
			responseTime := incident.AcknowledgedAt.Sub(incident.DetectedAt)
			totalResponseTime += responseTime
			responseCount++
		}
		
		if incident.ResolvedAt != nil {
			resolutionTime := incident.ResolvedAt.Sub(incident.DetectedAt)
			totalResolutionTime += resolutionTime
			resolutionCount++
		}
	}
	
	if responseCount > 0 {
		results.AverageResponseTime = totalResponseTime / time.Duration(responseCount)
	}
	
	if resolutionCount > 0 {
		results.AverageResolutionTime = totalResolutionTime / time.Duration(resolutionCount)
	}
}

// generateTrendData generates incident trend analysis
func (ir *IncidentResponder) generateTrendData() IncidentTrends {
	trends := IncidentTrends{
		Daily:   []IncidentDataPoint{},
		Weekly:  []IncidentDataPoint{},
		Monthly: []IncidentDataPoint{},
	}
	
	// Group incidents by time periods
	dailyMap := make(map[string]IncidentDataPoint)
	weeklyMap := make(map[string]IncidentDataPoint)
	monthlyMap := make(map[string]IncidentDataPoint)
	
	for _, incident := range ir.incidentHistory {
		date := incident.DetectedAt
		
		// Daily
		dayKey := date.Format("2006-01-02")
		if point, exists := dailyMap[dayKey]; exists {
			point.Count++
			point.Severity[incident.Severity]++
			dailyMap[dayKey] = point
		} else {
			dailyMap[dayKey] = IncidentDataPoint{
				Date:     date,
				Count:    1,
				Severity: map[IncidentSeverity]int{incident.Severity: 1},
			}
		}
		
		// Similar for weekly and monthly...
	}
	
	// Convert maps to slices and sort by date
	for _, point := range dailyMap {
		trends.Daily = append(trends.Daily, point)
	}
	sort.Slice(trends.Daily, func(i, j int) bool {
		return trends.Daily[i].Date.Before(trends.Daily[j].Date)
	})
	
	return trends
}

// getTopThreats returns the most common threat indicators
func (ir *IncidentResponder) getTopThreats(limit int) []ThreatIndicator {
	indicatorMap := make(map[string]*ThreatIndicator)
	
	for _, incident := range ir.incidentHistory {
		for _, indicator := range incident.Indicators {
			if existing, exists := indicatorMap[indicator.Value]; exists {
				existing.LastSeen = indicator.LastSeen
			} else {
				indicatorMap[indicator.Value] = &indicator
			}
		}
	}
	
	indicators := make([]ThreatIndicator, 0, len(indicatorMap))
	for _, indicator := range indicatorMap {
		indicators = append(indicators, *indicator)
	}
	
	// Sort by threat level and confidence
	sort.Slice(indicators, func(i, j int) bool {
		if indicators[i].ThreatLevel != indicators[j].ThreatLevel {
			return indicators[i].ThreatLevel > indicators[j].ThreatLevel
		}
		return indicators[i].Confidence > indicators[j].Confidence
	})
	
	if len(indicators) > limit {
		return indicators[:limit]
	}
	return indicators
}

// initializePlaybooks initializes default response playbooks
func (ir *IncidentResponder) initializePlaybooks() {
	malwarePlaybook := &ResponsePlaybook{
		ID:          "malware-response",
		Name:        "Malware Response",
		Description: "Automated response for malware incidents",
		Version:     "1.0",
		IncidentTypes: []IncidentType{IncidentTypeMalware},
		Triggers: []PlaybookTrigger{
			{Type: TriggerTypeIncidentType, Condition: "malware", Threshold: 1.0},
		},
		Steps: []PlaybookStep{
			{
				ID:          "isolate-system",
				Name:        "Isolate Affected System",
				Type:        StepTypeContainment,
				Description: "Isolate the affected system from the network",
				Action:      "network_isolation",
				Automated:   true,
				Parameters:  map[string]interface{}{"method": "vlan_isolation"},
			},
			{
				ID:          "collect-evidence",
				Name:        "Collect Forensics Evidence",
				Type:        StepTypeInvestigation,
				Description: "Collect memory dump and system artifacts",
				Action:      "forensics_collection",
				Automated:   true,
				Parameters:  map[string]interface{}{"types": []string{"memory", "logs", "files"}},
			},
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		CreatedBy: "system",
		Approved:  true,
		Metadata:  make(map[string]interface{}),
	}
	
	ir.playbooks[malwarePlaybook.ID] = malwarePlaybook
}

// initializeAlertRules initializes default alert rules
func (ir *IncidentResponder) initializeAlertRules() {
	criticalIncidentRule := &AlertRule{
		ID:          "critical-incident-alert",
		Name:        "Critical Incident Alert",
		Description: "Alert for critical severity incidents",
		Enabled:     true,
		Severity:    SeverityCritical,
		Conditions: []AlertCondition{
			{Field: "severity", Operator: "equals", Value: "critical", Window: time.Minute},
		},
		Actions: []AlertAction{
			{Type: "email", Target: "security-team@company.com"},
			{Type: "sms", Target: "+1234567890"},
			{Type: "webhook", Target: "https://alerting.company.com/webhook"},
		},
		Throttle:    time.Minute * 5,
		TriggerCount: 0,
		Metadata:    make(map[string]interface{}),
	}
	
	ir.alertRules = append(ir.alertRules, criticalIncidentRule)
}

// Additional component implementations would go here:
// - ThreatDetector
// - EscalationManager  
// - ForensicsCollector
// - RecoveryManager

// ThreatDetector stub implementation
type ThreatDetector struct {
	config *AuditConfig
}

func NewThreatDetector(config *AuditConfig) *ThreatDetector {
	return &ThreatDetector{config: config}
}

// EscalationManager stub implementation
type EscalationManager struct {
	config *AuditConfig
}

func NewEscalationManager(config *AuditConfig) *EscalationManager {
	return &EscalationManager{config: config}
}

func (em *EscalationManager) EvaluateEscalation(ctx context.Context, incident *SecurityIncident) {
	// Implementation would handle escalation logic
}

// ForensicsCollector stub implementation
type ForensicsCollector struct {
	config *AuditConfig
}

func NewForensicsCollector(config *AuditConfig) *ForensicsCollector {
	return &ForensicsCollector{config: config}
}

func (fc *ForensicsCollector) CollectEvidence(ctx context.Context, incident *SecurityIncident) {
	// Implementation would handle evidence collection
}

// RecoveryManager stub implementation
type RecoveryManager struct {
	config *AuditConfig
}

func NewRecoveryManager(config *AuditConfig) *RecoveryManager {
	return &RecoveryManager{config: config}
}