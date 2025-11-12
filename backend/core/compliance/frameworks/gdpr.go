// Package frameworks implements GDPR compliance automation
package frameworks

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"novacron/backend/core/compliance"
)

// GDPREngine implements GDPR compliance automation
type GDPREngine struct {
	controls          map[string]*compliance.ComplianceControl
	privacyRequests   map[string]*compliance.DataPrivacyRequest
	consentRecords    map[string]*compliance.ConsentRecord
	dataInventory     map[string]*DataInventoryItem
	processingRecords map[string]*ProcessingRecord
	mu                sync.RWMutex
}

// DataInventoryItem represents a data inventory entry
type DataInventoryItem struct {
	ID              string            `json:"id"`
	DataCategory    string            `json:"data_category"` // personal, sensitive, special
	DataType        string            `json:"data_type"`     // name, email, health, etc.
	Location        string            `json:"location"`
	Purpose         string            `json:"purpose"`
	LegalBasis      string            `json:"legal_basis"`   // Art. 6(1)
	RetentionPeriod time.Duration     `json:"retention_period"`
	DataSubjects    int               `json:"data_subjects_count"`
	Encrypted       bool              `json:"encrypted"`
	BackedUp        bool              `json:"backed_up"`
	Shared          []string          `json:"shared_with"` // Third parties
	CreatedAt       time.Time         `json:"created_at"`
	UpdatedAt       time.Time         `json:"updated_at"`
	Metadata        map[string]string `json:"metadata"`
}

// ProcessingRecord represents GDPR Article 30 processing record
type ProcessingRecord struct {
	ID                  string            `json:"id"`
	Controller          string            `json:"controller"`
	DPO                 string            `json:"dpo"` // Data Protection Officer
	Purpose             string            `json:"purpose"`
	LegalBasis          string            `json:"legal_basis"`
	DataCategories      []string          `json:"data_categories"`
	DataSubjectTypes    []string          `json:"data_subject_types"`
	Recipients          []string          `json:"recipients"`
	ThirdCountries      []string          `json:"third_countries"`
	SafeguardsMeasures  []string          `json:"safeguards_measures"`
	RetentionPeriod     string            `json:"retention_period"`
	SecurityMeasures    []string          `json:"security_measures"`
	CreatedAt           time.Time         `json:"created_at"`
	UpdatedAt           time.Time         `json:"updated_at"`
	Metadata            map[string]string `json:"metadata"`
}

// NewGDPREngine creates a new GDPR compliance engine
func NewGDPREngine() *GDPREngine {
	engine := &GDPREngine{
		controls:          make(map[string]*compliance.ComplianceControl),
		privacyRequests:   make(map[string]*compliance.DataPrivacyRequest),
		consentRecords:    make(map[string]*compliance.ConsentRecord),
		dataInventory:     make(map[string]*DataInventoryItem),
		processingRecords: make(map[string]*ProcessingRecord),
	}

	engine.registerGDPRControls()
	return engine
}

// registerGDPRControls registers GDPR compliance controls
func (e *GDPREngine) registerGDPRControls() {
	controls := []compliance.ComplianceControl{
		// Art. 5 - Principles
		{
			ID:          "GDPR-ART5-LAWFULNESS",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Lawfulness, Fairness, and Transparency",
			Description: "Process personal data lawfully, fairly, and transparently",
			Requirement: "Clear legal basis, transparent processing, fair data practices",
			Automated:   true,
			Owner:       "data-protection",
		},
		{
			ID:          "GDPR-ART5-PURPOSE",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Purpose Limitation",
			Description: "Collect data for specified, explicit, and legitimate purposes only",
			Requirement: "Document purposes, prevent purpose creep, obtain new consent for new purposes",
			Automated:   true,
			Owner:       "data-protection",
		},
		{
			ID:          "GDPR-ART5-MINIMIZATION",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Data Minimization",
			Description: "Collect only adequate, relevant, and necessary data",
			Requirement: "Minimize data collection, justify each data field, regular reviews",
			Automated:   true,
			Owner:       "data-protection",
		},
		{
			ID:          "GDPR-ART5-ACCURACY",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Data Accuracy",
			Description: "Ensure personal data is accurate and up to date",
			Requirement: "Data validation, correction mechanisms, regular updates",
			Automated:   true,
			Owner:       "data-protection",
		},
		{
			ID:          "GDPR-ART5-RETENTION",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Storage Limitation",
			Description: "Keep personal data no longer than necessary",
			Requirement: "Retention policies, automated deletion, periodic reviews",
			Automated:   true,
			Owner:       "data-protection",
		},
		{
			ID:          "GDPR-ART5-SECURITY",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Integrity and Confidentiality",
			Description: "Ensure appropriate security of personal data",
			Requirement: "Encryption, access controls, security testing, breach procedures",
			Automated:   true,
			Owner:       "security-team",
		},
		// Art. 7 - Consent
		{
			ID:          "GDPR-ART7-CONSENT",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Consent Management",
			Description: "Demonstrate consent, allow withdrawal, keep records",
			Requirement: "Opt-in consent, easy withdrawal, consent records, granular consent",
			Automated:   true,
			Owner:       "data-protection",
		},
		// Art. 12-14 - Transparency
		{
			ID:          "GDPR-ART12-TRANSPARENCY",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Transparency and Communication",
			Description: "Provide clear, accessible privacy information",
			Requirement: "Privacy notices, clear language, multiple formats",
			Automated:   false,
			Owner:       "legal",
		},
		// Art. 15 - Right to Access
		{
			ID:          "GDPR-ART15-ACCESS",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Right to Access",
			Description: "Enable data subjects to access their personal data",
			Requirement: "Automated data export, response within 30 days, identity verification",
			Automated:   true,
			Owner:       "data-protection",
		},
		// Art. 16 - Right to Rectification
		{
			ID:          "GDPR-ART16-RECTIFICATION",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Right to Rectification",
			Description: "Enable data subjects to correct inaccurate data",
			Requirement: "Self-service correction, admin review, audit trail",
			Automated:   true,
			Owner:       "data-protection",
		},
		// Art. 17 - Right to Erasure
		{
			ID:          "GDPR-ART17-ERASURE",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Right to Erasure (Right to be Forgotten)",
			Description: "Enable data subjects to request data deletion",
			Requirement: "Automated deletion, cascade to backups, verification, exceptions handling",
			Automated:   true,
			Owner:       "data-protection",
		},
		// Art. 18 - Right to Restriction
		{
			ID:          "GDPR-ART18-RESTRICTION",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Right to Restriction of Processing",
			Description: "Enable data subjects to restrict processing",
			Requirement: "Processing flags, restriction enforcement, notifications",
			Automated:   true,
			Owner:       "data-protection",
		},
		// Art. 20 - Data Portability
		{
			ID:          "GDPR-ART20-PORTABILITY",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Right to Data Portability",
			Description: "Provide data in structured, machine-readable format",
			Requirement: "JSON/CSV export, API access, direct transfer to other controllers",
			Automated:   true,
			Owner:       "data-protection",
		},
		// Art. 21 - Right to Object
		{
			ID:          "GDPR-ART21-OBJECTION",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Right to Object",
			Description: "Enable objection to processing (especially profiling)",
			Requirement: "Object mechanism, stop processing, assessment of legitimate interests",
			Automated:   true,
			Owner:       "data-protection",
		},
		// Art. 25 - Data Protection by Design
		{
			ID:          "GDPR-ART25-DESIGN",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Data Protection by Design and Default",
			Description: "Implement privacy by design and default",
			Requirement: "Privacy impact assessments, default privacy settings, pseudonymization",
			Automated:   true,
			Owner:       "engineering",
		},
		// Art. 30 - Records of Processing
		{
			ID:          "GDPR-ART30-RECORDS",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Records of Processing Activities",
			Description: "Maintain comprehensive processing records",
			Requirement: "Documented processing activities, controller/processor records, regular updates",
			Automated:   true,
			Owner:       "data-protection",
		},
		// Art. 32 - Security of Processing
		{
			ID:          "GDPR-ART32-SECURITY",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Security of Processing",
			Description: "Implement appropriate technical and organizational measures",
			Requirement: "Encryption, pseudonymization, resilience, regular testing, security certifications",
			Automated:   true,
			Owner:       "security-team",
		},
		// Art. 33-34 - Breach Notification
		{
			ID:          "GDPR-ART33-BREACH",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryIncidentResponse,
			Name:        "Personal Data Breach Notification",
			Description: "Notify authorities and data subjects of breaches",
			Requirement: "72-hour notification to DPA, data subject notification, breach register",
			Automated:   true,
			Owner:       "security-operations",
		},
		// Art. 35 - DPIA
		{
			ID:          "GDPR-ART35-DPIA",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Data Protection Impact Assessment",
			Description: "Conduct DPIA for high-risk processing",
			Requirement: "DPIA template, risk assessment, mitigation measures, DPO consultation",
			Automated:   false,
			Owner:       "data-protection",
		},
		// Art. 37-39 - DPO
		{
			ID:          "GDPR-ART37-DPO",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "Data Protection Officer",
			Description: "Designate and empower DPO",
			Requirement: "DPO contact details published, independence, resources, access to top management",
			Automated:   false,
			Owner:       "executive",
		},
		// Art. 44-49 - International Transfers
		{
			ID:          "GDPR-ART44-TRANSFERS",
			Framework:   compliance.FrameworkGDPR,
			Category:    compliance.CategoryDataProtection,
			Name:        "International Data Transfers",
			Description: "Ensure lawful international transfers",
			Requirement: "Adequacy decisions, SCCs, BCRs, transfer impact assessments",
			Automated:   true,
			Owner:       "legal",
		},
	}

	for _, control := range controls {
		control.Status = compliance.StatusUnknown
		control.Metadata = make(map[string]string)
		e.controls[control.ID] = &control
	}
}

// ProcessPrivacyRequest handles GDPR data subject requests
func (e *GDPREngine) ProcessPrivacyRequest(ctx context.Context, req *compliance.DataPrivacyRequest) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	req.ID = generateID("privacy-req")
	req.RequestedAt = time.Now()
	req.Status = "pending"

	// Store request
	e.privacyRequests[req.ID] = req

	// Route to appropriate handler
	switch req.Type {
	case compliance.RequestAccessData:
		return e.handleAccessRequest(ctx, req)
	case compliance.RequestRectification:
		return e.handleRectificationRequest(ctx, req)
	case compliance.RequestErasure:
		return e.handleErasureRequest(ctx, req)
	case compliance.RequestRestriction:
		return e.handleRestrictionRequest(ctx, req)
	case compliance.RequestPortability:
		return e.handlePortabilityRequest(ctx, req)
	case compliance.RequestObjection:
		return e.handleObjectionRequest(ctx, req)
	case compliance.RequestWithdrawConsent:
		return e.handleConsentWithdrawal(ctx, req)
	default:
		return fmt.Errorf("unknown privacy request type: %s", req.Type)
	}
}

// handleAccessRequest implements Art. 15 - Right to Access
func (e *GDPREngine) handleAccessRequest(ctx context.Context, req *compliance.DataPrivacyRequest) error {
	// 1. Verify identity
	// 2. Collect all personal data about the subject
	// 3. Generate comprehensive report
	// 4. Provide in accessible format

	req.Status = "processing"

	// Simulate data collection (would query actual databases)
	data := map[string]interface{}{
		"subject_id": req.SubjectID,
		"collected_at": time.Now(),
		"data_categories": []string{"account", "usage", "preferences"},
		"processing_purposes": []string{"service provision", "analytics"},
		"recipients": []string{"cloud provider", "email service"},
		"retention_period": "2 years from last activity",
		"rights": "access, rectification, erasure, restriction, portability, objection",
	}

	req.Data = data
	req.Status = "completed"
	now := time.Now()
	req.CompletedAt = &now

	return nil
}

// handleRectificationRequest implements Art. 16 - Right to Rectification
func (e *GDPREngine) handleRectificationRequest(ctx context.Context, req *compliance.DataPrivacyRequest) error {
	// 1. Verify identity
	// 2. Validate correction data
	// 3. Update across all systems
	// 4. Notify recipients if required

	req.Status = "processing"

	// Simulate data correction
	// In production, would update databases and notify affected parties

	req.Status = "completed"
	now := time.Now()
	req.CompletedAt = &now

	return nil
}

// handleErasureRequest implements Art. 17 - Right to be Forgotten
func (e *GDPREngine) handleErasureRequest(ctx context.Context, req *compliance.DataPrivacyRequest) error {
	// 1. Verify identity
	// 2. Check for legitimate reasons to refuse (legal obligations, public interest)
	// 3. Delete from all systems including backups
	// 4. Notify recipients who may have received the data
	// 5. Maintain minimal audit trail for compliance

	req.Status = "processing"

	// Simulate deletion process
	// In production, would:
	// - Delete from primary databases
	// - Schedule backup purging
	// - Remove from caches
	// - Notify third parties
	// - Keep minimal record of erasure request for compliance

	req.Status = "completed"
	now := time.Now()
	req.CompletedAt = &now

	return nil
}

// handleRestrictionRequest implements Art. 18 - Right to Restriction
func (e *GDPREngine) handleRestrictionRequest(ctx context.Context, req *compliance.DataPrivacyRequest) error {
	// 1. Verify identity
	// 2. Set processing restriction flag
	// 3. Notify systems to stop processing (except storage)
	// 4. Inform data subject when restriction will be lifted

	req.Status = "processing"

	// Simulate restriction
	// In production, would set flags in database to prevent processing

	req.Status = "completed"
	now := time.Now()
	req.CompletedAt = &now

	return nil
}

// handlePortabilityRequest implements Art. 20 - Right to Data Portability
func (e *GDPREngine) handlePortabilityRequest(ctx context.Context, req *compliance.DataPrivacyRequest) error {
	// 1. Verify identity
	// 2. Export data in structured, machine-readable format (JSON/CSV)
	// 3. Provide secure download link
	// 4. Optional: direct transfer to another controller

	req.Status = "processing"

	// Simulate data export
	exportData := map[string]interface{}{
		"format":     "JSON",
		"export_date": time.Now(),
		"data":       "Base64EncodedDataPackage",
	}

	req.Data = exportData
	req.Status = "completed"
	now := time.Now()
	req.CompletedAt = &now

	return nil
}

// handleObjectionRequest implements Art. 21 - Right to Object
func (e *GDPREngine) handleObjectionRequest(ctx context.Context, req *compliance.DataPrivacyRequest) error {
	// 1. Verify identity
	// 2. Stop processing (especially profiling/direct marketing)
	// 3. Assess compelling legitimate grounds
	// 4. Delete data if no legitimate grounds

	req.Status = "processing"

	// Simulate objection processing
	// In production, would stop specific processing activities

	req.Status = "completed"
	now := time.Now()
	req.CompletedAt = &now

	return nil
}

// handleConsentWithdrawal implements Art. 7(3) - Withdraw Consent
func (e *GDPREngine) handleConsentWithdrawal(ctx context.Context, req *compliance.DataPrivacyRequest) error {
	// 1. Verify identity
	// 2. Withdraw consent
	// 3. Stop processing based on that consent
	// 4. Keep minimal record for compliance

	req.Status = "processing"

	// Update consent record
	for _, consent := range e.consentRecords {
		if consent.SubjectID == req.SubjectID {
			now := time.Now()
			consent.WithdrawnDate = &now
			consent.ConsentGiven = false
		}
	}

	req.Status = "completed"
	now := time.Now()
	req.CompletedAt = &now

	return nil
}

// RecordConsent records user consent for GDPR compliance
func (e *GDPREngine) RecordConsent(ctx context.Context, consent *compliance.ConsentRecord) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	consent.ID = generateID("consent")
	consent.ConsentDate = time.Now()

	e.consentRecords[consent.ID] = consent

	return nil
}

// GetConsent retrieves consent record
func (e *GDPREngine) GetConsent(subjectID, purpose string) (*compliance.ConsentRecord, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	for _, consent := range e.consentRecords {
		if consent.SubjectID == subjectID && consent.Purpose == purpose {
			return consent, nil
		}
	}

	return nil, fmt.Errorf("consent record not found")
}

// ValidateConsent checks if valid consent exists
func (e *GDPREngine) ValidateConsent(subjectID, purpose string) (bool, error) {
	consent, err := e.GetConsent(subjectID, purpose)
	if err != nil {
		return false, err
	}

	// Check if consent is withdrawn
	if !consent.ConsentGiven || consent.WithdrawnDate != nil {
		return false, nil
	}

	// Check if consent expired
	if consent.ExpiryDate != nil && time.Now().After(*consent.ExpiryDate) {
		return false, nil
	}

	return true, nil
}

// RegisterDataInventoryItem adds item to data inventory
func (e *GDPREngine) RegisterDataInventoryItem(item *DataInventoryItem) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	item.ID = generateID("data-inv")
	item.CreatedAt = time.Now()
	item.UpdatedAt = time.Now()

	e.dataInventory[item.ID] = item

	return nil
}

// CreateProcessingRecord creates Art. 30 processing record
func (e *GDPREngine) CreateProcessingRecord(record *ProcessingRecord) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	record.ID = generateID("processing-rec")
	record.CreatedAt = time.Now()
	record.UpdatedAt = time.Now()

	e.processingRecords[record.ID] = record

	return nil
}

// AssessCompliance performs GDPR compliance assessment
func (e *GDPREngine) AssessCompliance(ctx context.Context) (*compliance.ComplianceReport, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	report := &compliance.ComplianceReport{
		ID:          generateID("gdpr-report"),
		Framework:   compliance.FrameworkGDPR,
		GeneratedAt: time.Now(),
		GeneratedBy: "automated-system",
		Period: compliance.Period{
			Start: time.Now().AddDate(0, -1, 0),
			End:   time.Now(),
		},
		Controls: make([]compliance.ControlResult, 0),
		Findings: make([]compliance.Finding, 0),
		Evidence: make([]compliance.Evidence, 0),
	}

	// Assess each control
	totalScore := 0.0
	for _, control := range e.controls {
		result := e.assessControl(ctx, control)
		report.Controls = append(report.Controls, result)
		totalScore += result.Score

		if result.Status != compliance.StatusCompliant {
			finding := compliance.Finding{
				ID:          generateID("finding"),
				Severity:    e.determineSeverity(control),
				Title:       fmt.Sprintf("GDPR %s: %s", control.ID, control.Name),
				Description: result.Details,
				ControlID:   control.ID,
				Impact:      "GDPR compliance risk, potential regulatory action",
				Risk:        "High - GDPR fines up to 4% of annual global turnover or €20M",
				CreatedAt:   time.Now(),
			}
			report.Findings = append(report.Findings, finding)
		}
	}

	// Calculate score
	if len(report.Controls) > 0 {
		report.Score = totalScore / float64(len(report.Controls))
	}

	// Determine status
	if report.Score >= 95.0 {
		report.Status = compliance.StatusCompliant
	} else if report.Score >= 70.0 {
		report.Status = compliance.StatusPartial
	} else {
		report.Status = compliance.StatusNonCompliant
	}

	report.Summary = e.generateSummary(report)

	return report, nil
}

// assessControl assesses a single GDPR control
func (e *GDPREngine) assessControl(ctx context.Context, control *compliance.ComplianceControl) compliance.ControlResult {
	result := compliance.ControlResult{
		ControlID: control.ID,
		CheckedAt: time.Now(),
		Score:     100.0,
		Status:    compliance.StatusCompliant,
	}

	// Simplified assessment logic
	// In production, would perform detailed checks

	issues := []string{}

	switch control.ID {
	case "GDPR-ART15-ACCESS":
		if len(e.privacyRequests) == 0 {
			issues = append(issues, "No access request handling implemented")
			result.Score -= 50
		}
	case "GDPR-ART17-ERASURE":
		if len(e.privacyRequests) == 0 {
			issues = append(issues, "No erasure mechanism implemented")
			result.Score -= 50
		}
	case "GDPR-ART30-RECORDS":
		if len(e.processingRecords) == 0 {
			issues = append(issues, "No processing records maintained")
			result.Score -= 50
		}
	case "GDPR-ART7-CONSENT":
		if len(e.consentRecords) == 0 {
			issues = append(issues, "No consent management system")
			result.Score -= 50
		}
	}

	if len(issues) > 0 {
		result.Status = compliance.StatusPartial
		result.Details = fmt.Sprintf("Issues found: %v", issues)
	} else {
		result.Details = "Control properly implemented"
	}

	return result
}

func (e *GDPREngine) generateSummary(report *compliance.ComplianceReport) compliance.ReportSummary {
	summary := compliance.ReportSummary{
		TotalControls:   len(report.Controls),
		ComplianceScore: report.Score,
	}

	for _, control := range report.Controls {
		switch control.Status {
		case compliance.StatusCompliant:
			summary.CompliantControls++
		case compliance.StatusNonCompliant:
			summary.FailedControls++
		case compliance.StatusPartial:
			summary.PartialControls++
		}
	}

	for _, finding := range report.Findings {
		switch finding.Severity {
		case "critical":
			summary.CriticalFindings++
		case "high":
			summary.HighFindings++
		case "medium":
			summary.MediumFindings++
		case "low":
			summary.LowFindings++
		}
	}

	return summary
}

func (e *GDPREngine) determineSeverity(control *compliance.ComplianceControl) string {
	// GDPR controls are generally high severity due to significant fines
	criticalControls := map[string]bool{
		"GDPR-ART17-ERASURE":  true,
		"GDPR-ART32-SECURITY": true,
		"GDPR-ART33-BREACH":   true,
		"GDPR-ART44-TRANSFERS": true,
	}

	if criticalControls[control.ID] {
		return "critical"
	}

	return "high"
}

// GetPrivacyRequests returns all privacy requests
func (e *GDPREngine) GetPrivacyRequests(status string) ([]*compliance.DataPrivacyRequest, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var requests []*compliance.DataPrivacyRequest
	for _, req := range e.privacyRequests {
		if status == "" || req.Status == status {
			requests = append(requests, req)
		}
	}

	return requests, nil
}

// GetDataInventory returns data inventory
func (e *GDPREngine) GetDataInventory() ([]*DataInventoryItem, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var items []*DataInventoryItem
	for _, item := range e.dataInventory {
		items = append(items, item)
	}

	return items, nil
}

// GetProcessingRecords returns processing records
func (e *GDPREngine) GetProcessingRecords() ([]*ProcessingRecord, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var records []*ProcessingRecord
	for _, record := range e.processingRecords {
		records = append(records, record)
	}

	return records, nil
}

// CheckBreachNotificationRequired determines if 72-hour breach notification is needed
func (e *GDPREngine) CheckBreachNotificationRequired(ctx context.Context, breach BreachDetails) (bool, string) {
	// Art. 33: Notify within 72 hours if breach poses risk to rights and freedoms

	riskLevel := e.assessBreachRisk(breach)

	switch riskLevel {
	case "high", "critical":
		return true, "Notification to supervisory authority required within 72 hours (Art. 33). Data subject notification also required (Art. 34)."
	case "medium":
		return true, "Notification to supervisory authority required within 72 hours (Art. 33)."
	default:
		return false, "Breach does not pose significant risk. Document internally but notification not required."
	}
}

// BreachDetails contains information about a data breach
type BreachDetails struct {
	AffectedRecords     int
	DataCategories      []string
	ContainsSpecialData bool // Art. 9 special categories
	SecurityMeasures    []string
	ContainmentStatus   string
	DetectedAt          time.Time
}

func (e *GDPREngine) assessBreachRisk(breach BreachDetails) string {
	score := 0

	// Number of affected records
	if breach.AffectedRecords > 10000 {
		score += 3
	} else if breach.AffectedRecords > 1000 {
		score += 2
	} else if breach.AffectedRecords > 100 {
		score += 1
	}

	// Special categories of data (Art. 9)
	if breach.ContainsSpecialData {
		score += 3
	}

	// Security measures in place
	if len(breach.SecurityMeasures) < 2 {
		score += 2
	}

	// Containment
	if breach.ContainmentStatus != "contained" {
		score += 2
	}

	// Map score to risk level
	if score >= 7 {
		return "critical"
	} else if score >= 5 {
		return "high"
	} else if score >= 3 {
		return "medium"
	}
	return "low"
}
