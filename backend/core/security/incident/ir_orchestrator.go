// Package incident implements security incident response orchestration
package incident

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// IncidentSeverity represents incident severity
type IncidentSeverity string

const (
	SeverityLow      IncidentSeverity = "low"
	SeverityMedium   IncidentSeverity = "medium"
	SeverityHigh     IncidentSeverity = "high"
	SeverityCritical IncidentSeverity = "critical"
)

// IncidentStatus represents incident status
type IncidentStatus string

const (
	StatusNew          IncidentStatus = "new"
	StatusTriaged      IncidentStatus = "triaged"
	StatusInvestigating IncidentStatus = "investigating"
	StatusContained    IncidentStatus = "contained"
	StatusMitigated    IncidentStatus = "mitigated"
	StatusResolved     IncidentStatus = "resolved"
	StatusClosed       IncidentStatus = "closed"
)

// IncidentType represents the type of security incident
type IncidentType string

const (
	TypeMalware         IncidentType = "malware"
	TypeIntrusion       IncidentType = "intrusion"
	TypeDataBreach      IncidentType = "data_breach"
	TypeDDoS            IncidentType = "ddos"
	TypeInsiderThreat   IncidentType = "insider_threat"
	TypeRansomware      IncidentType = "ransomware"
	TypePhishing        IncidentType = "phishing"
	TypeUnauthorizedAccess IncidentType = "unauthorized_access"
)

// Incident represents a security incident
type Incident struct {
	ID             string
	Type           IncidentType
	Severity       IncidentSeverity
	Status         IncidentStatus
	Title          string
	Description    string
	Source         string
	Target         string
	DetectedAt     time.Time
	TriagedAt      time.Time
	ContainedAt    time.Time
	ResolvedAt     time.Time
	MTTD           time.Duration // Mean Time To Detect
	MTTR           time.Duration // Mean Time To Respond
	Indicators     []string
	Evidence       []string
	Actions        []string
	Assignee       string
	Priority       int
	Metadata       map[string]interface{}
}

// Alert represents a security alert
type Alert struct {
	ID          string
	IncidentID  string
	Severity    IncidentSeverity
	Title       string
	Description string
	Source      string
	Timestamp   time.Time
	Acknowledged bool
	Metadata    map[string]interface{}
}

// Playbook represents an incident response playbook
type Playbook struct {
	ID          string
	Name        string
	Type        IncidentType
	Enabled     bool
	Steps       []PlaybookStep
	AutoExecute bool
	Metadata    map[string]interface{}
}

// PlaybookStep represents a step in incident response
type PlaybookStep struct {
	ID          string
	Order       int
	Name        string
	Action      string
	Description string
	Automated   bool
	Timeout     time.Duration
	OnSuccess   string
	OnFailure   string
}

// ContainmentAction represents an automated containment action
type ContainmentAction struct {
	Type       string // "isolate_vm", "block_ip", "disable_user", "kill_process"
	Target     string
	ExecutedAt time.Time
	Success    bool
	Metadata   map[string]interface{}
}

// Orchestrator manages security incident response
type Orchestrator struct {
	incidents           map[string]*Incident
	alerts              map[string]*Alert
	playbooks           map[string]*Playbook
	autoDetection       bool
	autoContainment     bool
	playbookExecution   bool
	forensicsCollection bool
	alertPrioritization bool
	mttdTarget          time.Duration
	mttrTarget          time.Duration
	mu                  sync.RWMutex
	totalIncidents      int64
	resolvedIncidents   int64
	avgMTTD             time.Duration
	avgMTTR             time.Duration
}

// NewOrchestrator creates a new incident response orchestrator
func NewOrchestrator(mttdTarget, mttrTarget time.Duration) *Orchestrator {
	return &Orchestrator{
		incidents:           make(map[string]*Incident),
		alerts:              make(map[string]*Alert),
		playbooks:           make(map[string]*Playbook),
		autoDetection:       true,
		autoContainment:     true,
		playbookExecution:   true,
		forensicsCollection: true,
		alertPrioritization: true,
		mttdTarget:          mttdTarget,
		mttrTarget:          mttrTarget,
	}
}

// CreateIncident creates a new security incident
func (o *Orchestrator) CreateIncident(incidentType IncidentType, severity IncidentSeverity, title, description, source string) (*Incident, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	incident := &Incident{
		ID:          generateIncidentID(),
		Type:        incidentType,
		Severity:    severity,
		Status:      StatusNew,
		Title:       title,
		Description: description,
		Source:      source,
		DetectedAt:  time.Now(),
		Indicators:  make([]string, 0),
		Evidence:    make([]string, 0),
		Actions:     make([]string, 0),
		Priority:    o.calculatePriority(severity),
		Metadata:    make(map[string]interface{}),
	}

	o.incidents[incident.ID] = incident
	o.totalIncidents++

	// Auto-triage if enabled
	if o.alertPrioritization {
		o.triageIncident(incident)
	}

	// Auto-execute playbook if enabled
	if o.playbookExecution {
		if err := o.executePlaybook(incident); err != nil {
			return incident, fmt.Errorf("playbook execution failed: %w", err)
		}
	}

	return incident, nil
}

// triageIncident triages an incident
func (o *Orchestrator) triageIncident(incident *Incident) {
	incident.Status = StatusTriaged
	incident.TriagedAt = time.Now()
	incident.MTTD = incident.TriagedAt.Sub(incident.DetectedAt)

	// Update average MTTD
	if o.totalIncidents > 0 {
		o.avgMTTD = (o.avgMTTD*time.Duration(o.totalIncidents-1) + incident.MTTD) / time.Duration(o.totalIncidents)
	}
}

// calculatePriority calculates incident priority
func (o *Orchestrator) calculatePriority(severity IncidentSeverity) int {
	switch severity {
	case SeverityCritical:
		return 1
	case SeverityHigh:
		return 2
	case SeverityMedium:
		return 3
	case SeverityLow:
		return 4
	default:
		return 5
	}
}

// AddPlaybook adds an incident response playbook
func (o *Orchestrator) AddPlaybook(playbook *Playbook) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	if playbook.ID == "" {
		playbook.ID = generatePlaybookID()
	}

	o.playbooks[playbook.ID] = playbook
	return nil
}

// executePlaybook executes a playbook for an incident
func (o *Orchestrator) executePlaybook(incident *Incident) error {
	// Find applicable playbook
	var playbook *Playbook
	for _, pb := range o.playbooks {
		if pb.Type == incident.Type && pb.Enabled {
			playbook = pb
			break
		}
	}

	if playbook == nil {
		return fmt.Errorf("no playbook found for incident type: %s", incident.Type)
	}

	if !playbook.AutoExecute {
		return nil
	}

	// Execute playbook steps
	for _, step := range playbook.Steps {
		if step.Automated {
			if err := o.executePlaybookStep(incident, &step); err != nil {
				return fmt.Errorf("step %s failed: %w", step.Name, err)
			}
		}
	}

	return nil
}

// executePlaybookStep executes a single playbook step
func (o *Orchestrator) executePlaybookStep(incident *Incident, step *PlaybookStep) error {
	incident.Actions = append(incident.Actions, fmt.Sprintf("Executing: %s", step.Name))

	switch step.Action {
	case "contain":
		return o.containIncident(incident)
	case "collect_forensics":
		return o.collectForensics(incident)
	case "notify":
		return o.notifyTeam(incident)
	case "isolate":
		return o.isolateTarget(incident)
	default:
		return fmt.Errorf("unknown action: %s", step.Action)
	}
}

// containIncident performs automated containment
func (o *Orchestrator) containIncident(incident *Incident) error {
	if !o.autoContainment {
		return nil
	}

	var actions []ContainmentAction

	// Determine containment actions based on incident type
	switch incident.Type {
	case TypeMalware, TypeRansomware:
		actions = append(actions, ContainmentAction{
			Type:       "isolate_vm",
			Target:     incident.Target,
			ExecutedAt: time.Now(),
			Success:    true,
			Metadata:   make(map[string]interface{}),
		})

	case TypeIntrusion, TypeUnauthorizedAccess:
		actions = append(actions, ContainmentAction{
			Type:       "block_ip",
			Target:     incident.Source,
			ExecutedAt: time.Now(),
			Success:    true,
			Metadata:   make(map[string]interface{}),
		})

	case TypeDDoS:
		actions = append(actions, ContainmentAction{
			Type:       "block_ip",
			Target:     incident.Source,
			ExecutedAt: time.Now(),
			Success:    true,
			Metadata:   make(map[string]interface{}),
		})
	}

	// Execute containment actions
	for _, action := range actions {
		incident.Actions = append(incident.Actions, fmt.Sprintf("Containment: %s on %s", action.Type, action.Target))
	}

	incident.Status = StatusContained
	incident.ContainedAt = time.Now()

	return nil
}

// collectForensics collects forensics data
func (o *Orchestrator) collectForensics(incident *Incident) error {
	if !o.forensicsCollection {
		return nil
	}

	// Simulate forensics collection
	evidence := []string{
		fmt.Sprintf("memory_dump_%s", incident.Target),
		fmt.Sprintf("disk_image_%s", incident.Target),
		fmt.Sprintf("network_logs_%s", incident.Source),
		fmt.Sprintf("system_logs_%s", incident.Target),
	}

	incident.Evidence = append(incident.Evidence, evidence...)
	incident.Actions = append(incident.Actions, fmt.Sprintf("Collected %d forensics artifacts", len(evidence)))

	return nil
}

// notifyTeam notifies security team
func (o *Orchestrator) notifyTeam(incident *Incident) error {
	// Simulate team notification
	incident.Actions = append(incident.Actions, "Notified security team")
	return nil
}

// isolateTarget isolates the affected target
func (o *Orchestrator) isolateTarget(incident *Incident) error {
	// Simulate target isolation
	incident.Actions = append(incident.Actions, fmt.Sprintf("Isolated target: %s", incident.Target))
	return nil
}

// ResolveIncident marks an incident as resolved
func (o *Orchestrator) ResolveIncident(incidentID string, resolution string) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	incident, exists := o.incidents[incidentID]
	if !exists {
		return fmt.Errorf("incident not found: %s", incidentID)
	}

	incident.Status = StatusResolved
	incident.ResolvedAt = time.Now()
	incident.MTTR = incident.ResolvedAt.Sub(incident.DetectedAt)
	incident.Metadata["resolution"] = resolution

	o.resolvedIncidents++

	// Update average MTTR
	if o.resolvedIncidents > 0 {
		o.avgMTTR = (o.avgMTTR*time.Duration(o.resolvedIncidents-1) + incident.MTTR) / time.Duration(o.resolvedIncidents)
	}

	return nil
}

// CreateAlert creates a security alert
func (o *Orchestrator) CreateAlert(incidentID string, severity IncidentSeverity, title, description, source string) (*Alert, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	alert := &Alert{
		ID:           generateAlertID(),
		IncidentID:   incidentID,
		Severity:     severity,
		Title:        title,
		Description:  description,
		Source:       source,
		Timestamp:    time.Now(),
		Acknowledged: false,
		Metadata:     make(map[string]interface{}),
	}

	o.alerts[alert.ID] = alert
	return alert, nil
}

// AcknowledgeAlert acknowledges an alert
func (o *Orchestrator) AcknowledgeAlert(alertID string) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	alert, exists := o.alerts[alertID]
	if !exists {
		return fmt.Errorf("alert not found: %s", alertID)
	}

	alert.Acknowledged = true
	return nil
}

// GetIncident retrieves an incident
func (o *Orchestrator) GetIncident(incidentID string) (*Incident, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()

	incident, exists := o.incidents[incidentID]
	if !exists {
		return nil, fmt.Errorf("incident not found: %s", incidentID)
	}

	return incident, nil
}

// ListIncidents lists all incidents
func (o *Orchestrator) ListIncidents() []*Incident {
	o.mu.RLock()
	defer o.mu.RUnlock()

	incidents := make([]*Incident, 0, len(o.incidents))
	for _, incident := range o.incidents {
		incidents = append(incidents, incident)
	}

	// Sort by priority
	for i := 0; i < len(incidents)-1; i++ {
		for j := i + 1; j < len(incidents); j++ {
			if incidents[i].Priority > incidents[j].Priority {
				incidents[i], incidents[j] = incidents[j], incidents[i]
			}
		}
	}

	return incidents
}

// GetMetrics returns incident response metrics
func (o *Orchestrator) GetMetrics() map[string]interface{} {
	o.mu.RLock()
	defer o.mu.RUnlock()

	incidentsBySeverity := make(map[IncidentSeverity]int)
	incidentsByStatus := make(map[IncidentStatus]int)
	incidentsByType := make(map[IncidentType]int)

	for _, incident := range o.incidents {
		incidentsBySeverity[incident.Severity]++
		incidentsByStatus[incident.Status]++
		incidentsByType[incident.Type]++
	}

	unacknowledgedAlerts := 0
	for _, alert := range o.alerts {
		if !alert.Acknowledged {
			unacknowledgedAlerts++
		}
	}

	mttdCompliance := 0.0
	if o.mttdTarget > 0 && o.avgMTTD > 0 {
		mttdCompliance = float64(o.mttdTarget) / float64(o.avgMTTD)
		if mttdCompliance > 1.0 {
			mttdCompliance = 1.0
		}
	}

	mttrCompliance := 0.0
	if o.mttrTarget > 0 && o.avgMTTR > 0 {
		mttrCompliance = float64(o.mttrTarget) / float64(o.avgMTTR)
		if mttrCompliance > 1.0 {
			mttrCompliance = 1.0
		}
	}

	resolutionRate := 0.0
	if o.totalIncidents > 0 {
		resolutionRate = float64(o.resolvedIncidents) / float64(o.totalIncidents)
	}

	return map[string]interface{}{
		"total_incidents":        o.totalIncidents,
		"resolved_incidents":     o.resolvedIncidents,
		"resolution_rate":        resolutionRate,
		"incidents_by_severity":  incidentsBySeverity,
		"incidents_by_status":    incidentsByStatus,
		"incidents_by_type":      incidentsByType,
		"total_alerts":           len(o.alerts),
		"unacknowledged_alerts":  unacknowledgedAlerts,
		"total_playbooks":        len(o.playbooks),
		"avg_mttd_ms":            o.avgMTTD.Milliseconds(),
		"avg_mttr_ms":            o.avgMTTR.Milliseconds(),
		"mttd_target_ms":         o.mttdTarget.Milliseconds(),
		"mttr_target_ms":         o.mttrTarget.Milliseconds(),
		"mttd_compliance":        mttdCompliance,
		"mttr_compliance":        mttrCompliance,
		"auto_detection":         o.autoDetection,
		"auto_containment":       o.autoContainment,
		"playbook_execution":     o.playbookExecution,
		"forensics_collection":   o.forensicsCollection,
		"alert_prioritization":   o.alertPrioritization,
	}
}

// Helper functions

func generateIncidentID() string {
	b := make([]byte, 16)
	hash := sha256.Sum256([]byte(time.Now().String()))
	copy(b, hash[:16])
	return fmt.Sprintf("inc-%s", hex.EncodeToString(b))
}

func generateAlertID() string {
	b := make([]byte, 16)
	hash := sha256.Sum256([]byte(time.Now().String()))
	copy(b, hash[:16])
	return fmt.Sprintf("alert-%s", hex.EncodeToString(b))
}

func generatePlaybookID() string {
	b := make([]byte, 16)
	hash := sha256.Sum256([]byte(time.Now().String()))
	copy(b, hash[:16])
	return fmt.Sprintf("playbook-%s", hex.EncodeToString(b))
}
