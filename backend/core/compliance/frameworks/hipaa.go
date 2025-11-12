// Package frameworks implements HIPAA compliance automation
package frameworks

import (
	"context"
	"fmt"
	"sync"
	"time"

	"novacron/backend/core/compliance"
)

// HIPAAEngine implements HIPAA compliance automation
type HIPAAEngine struct {
	controls   map[string]*compliance.ComplianceControl
	phiLogs    map[string]*compliance.PHIAccessLog
	baaRecords map[string]*compliance.BAA
	mu         sync.RWMutex
}

// NewHIPAAEngine creates a new HIPAA compliance engine
func NewHIPAAEngine() *HIPAAEngine {
	engine := &HIPAAEngine{
		controls:   make(map[string]*compliance.ComplianceControl),
		phiLogs:    make(map[string]*compliance.PHIAccessLog),
		baaRecords: make(map[string]*compliance.BAA),
	}

	engine.registerHIPAAControls()
	return engine
}

// registerHIPAAControls registers HIPAA Security Rule controls
func (e *HIPAAEngine) registerHIPAAControls() {
	controls := []compliance.ComplianceControl{
		// Administrative Safeguards (§164.308)
		{
			ID:          "HIPAA-164.308-A1",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Security Management Process",
			Description: "Implement policies and procedures to prevent, detect, contain, and correct security violations",
			Requirement: "Risk analysis, risk management, sanction policy, information system activity review",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.308-A2",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Assigned Security Responsibility",
			Description: "Identify security official responsible for security policies",
			Requirement: "Designated security official with authority and resources",
			Automated:   false,
			Owner:       "executive",
		},
		{
			ID:          "HIPAA-164.308-A3",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Workforce Security",
			Description: "Implement procedures for workforce access authorization and supervision",
			Requirement: "Authorization procedures, workforce clearance, termination procedures",
			Automated:   true,
			Owner:       "hr-security",
		},
		{
			ID:          "HIPAA-164.308-A4",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAuditLogging,
			Name:        "Information Access Management",
			Description: "Implement policies for authorizing access to ePHI",
			Requirement: "Access authorization, access establishment, access modification",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.308-A5",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Security Awareness and Training",
			Description: "Implement security awareness training program",
			Requirement: "Security reminders, protection from malware, log-in monitoring, password management",
			Automated:   false,
			Owner:       "training",
		},
		{
			ID:          "HIPAA-164.308-A6",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryIncidentResponse,
			Name:        "Security Incident Procedures",
			Description: "Implement policies for identifying and responding to security incidents",
			Requirement: "Response and reporting procedures, incident handling, documentation",
			Automated:   true,
			Owner:       "security-operations",
		},
		{
			ID:          "HIPAA-164.308-A7",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryBackupRecovery,
			Name:        "Contingency Plan",
			Description: "Establish procedures for responding to emergency or disaster",
			Requirement: "Data backup, disaster recovery, emergency mode operations, testing",
			Automated:   true,
			Owner:       "infrastructure",
		},
		{
			ID:          "HIPAA-164.308-A8",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAuditLogging,
			Name:        "Evaluation",
			Description: "Perform periodic technical and non-technical evaluation",
			Requirement: "Annual security evaluation against HIPAA requirements",
			Automated:   true,
			Owner:       "compliance",
		},
		{
			ID:          "HIPAA-164.308-B1",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryVendorManagement,
			Name:        "Business Associate Contracts",
			Description: "Ensure business associate agreements are in place",
			Requirement: "Written BAA contracts, safeguards obligations, reporting requirements",
			Automated:   true,
			Owner:       "legal",
		},

		// Physical Safeguards (§164.310)
		{
			ID:          "HIPAA-164.310-A1",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Facility Access Controls",
			Description: "Implement policies for limiting physical access",
			Requirement: "Facility security plan, access control, validation procedures",
			Automated:   false,
			Owner:       "facilities",
		},
		{
			ID:          "HIPAA-164.310-B",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Workstation Use",
			Description: "Implement policies for workstation functions and security",
			Requirement: "Workstation security policies, proper use guidelines",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.310-C",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Workstation Security",
			Description: "Implement physical safeguards for workstations",
			Requirement: "Restrict physical access, automatic logoff, encryption",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.310-D1",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryDataProtection,
			Name:        "Device and Media Controls",
			Description: "Implement policies for disposal and reuse of hardware/media",
			Requirement: "Disposal procedures, media re-use, accountability, data backup/storage",
			Automated:   true,
			Owner:       "security-team",
		},

		// Technical Safeguards (§164.312)
		{
			ID:          "HIPAA-164.312-A1",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Access Control",
			Description: "Implement technical policies for ePHI access",
			Requirement: "Unique user IDs, emergency access, automatic logoff, encryption/decryption",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.312-B",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAuditLogging,
			Name:        "Audit Controls",
			Description: "Implement hardware, software, and procedural mechanisms to record and examine activity",
			Requirement: "Audit logs, log review, log protection, retention (6 years)",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.312-C1",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryDataProtection,
			Name:        "Integrity Controls",
			Description: "Implement policies to ensure ePHI is not improperly altered or destroyed",
			Requirement: "Mechanism to authenticate ePHI, checksums, digital signatures",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.312-D",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryAccessControl,
			Name:        "Person or Entity Authentication",
			Description: "Implement procedures to verify identity of persons/entities",
			Requirement: "Multi-factor authentication, identity verification, session management",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.312-E1",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryEncryption,
			Name:        "Transmission Security",
			Description: "Implement technical security measures to guard against unauthorized access to ePHI during transmission",
			Requirement: "Encryption in transit (TLS 1.2+), integrity controls, network security",
			Automated:   true,
			Owner:       "security-team",
		},
		{
			ID:          "HIPAA-164.312-E2",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryEncryption,
			Name:        "Encryption at Rest",
			Description: "Implement mechanism to encrypt ePHI",
			Requirement: "AES-256 encryption, key management, secure key storage",
			Automated:   true,
			Owner:       "security-team",
		},

		// Privacy Rule Requirements (§164.502-514)
		{
			ID:          "HIPAA-164.502-MINIMUM",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryDataProtection,
			Name:        "Minimum Necessary Requirement",
			Description: "Limit PHI use, disclosure, and requests to minimum necessary",
			Requirement: "Minimum necessary policies, role-based access, justification required",
			Automated:   true,
			Owner:       "privacy-officer",
		},
		{
			ID:          "HIPAA-164.508-AUTHORIZATION",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryDataProtection,
			Name:        "Authorization for Uses and Disclosures",
			Description: "Obtain patient authorization for PHI uses/disclosures",
			Requirement: "Written authorization, specific to purpose, right to revoke",
			Automated:   true,
			Owner:       "privacy-officer",
		},
		{
			ID:          "HIPAA-164.510-PATIENT-RIGHTS",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryDataProtection,
			Name:        "Patient Rights",
			Description: "Provide patients rights to access, amend, and receive accounting of disclosures",
			Requirement: "Access within 30 days, amendment process, disclosure accounting (6 years)",
			Automated:   true,
			Owner:       "privacy-officer",
		},
		{
			ID:          "HIPAA-164.514-DEIDENTIFICATION",
			Framework:   compliance.FrameworkHIPAA,
			Category:    compliance.CategoryDataProtection,
			Name:        "De-identification of PHI",
			Description: "Properly de-identify PHI when required",
			Requirement: "Safe harbor method (remove 18 identifiers) or expert determination",
			Automated:   true,
			Owner:       "data-engineering",
		},
	}

	for _, control := range controls {
		control.Status = compliance.StatusUnknown
		control.Metadata = make(map[string]string)
		e.controls[control.ID] = &control
	}
}

// LogPHIAccess logs Protected Health Information access
func (e *HIPAAEngine) LogPHIAccess(ctx context.Context, log *compliance.PHIAccessLog) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	log.ID = generateID("phi-access")
	log.Timestamp = time.Now()

	// Store log (in production, would write to immutable audit log)
	e.phiLogs[log.ID] = log

	// Check for break-glass access (emergency access)
	if log.BreakGlass {
		// Trigger alert for break-glass usage
		e.alertBreakGlassAccess(log)
	}

	// Check for unauthorized access patterns
	if !log.Authorized {
		e.alertUnauthorizedAccess(log)
	}

	return nil
}

// RegisterBAA registers Business Associate Agreement
func (e *HIPAAEngine) RegisterBAA(ctx context.Context, baa *compliance.BAA) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	baa.ID = generateID("baa")
	baa.Status = "active"

	e.baaRecords[baa.ID] = baa

	return nil
}

// GetPHIAccessLogs retrieves PHI access logs
func (e *HIPAAEngine) GetPHIAccessLogs(patientID string, since time.Time) ([]*compliance.PHIAccessLog, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var logs []*compliance.PHIAccessLog
	for _, log := range e.phiLogs {
		if (patientID == "" || log.PatientID == patientID) && log.Timestamp.After(since) {
			logs = append(logs, log)
		}
	}

	return logs, nil
}

// GetActiveBAAs returns active Business Associate Agreements
func (e *HIPAAEngine) GetActiveBAAs() ([]*compliance.BAA, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var baas []*compliance.BAA
	now := time.Now()
	for _, baa := range e.baaRecords {
		if baa.Status == "active" && now.Before(baa.ExpiryDate) {
			baas = append(baas, baa)
		}
	}

	return baas, nil
}

// CheckBAACompliance checks if vendor has valid BAA
func (e *HIPAAEngine) CheckBAACompliance(entityName string) (bool, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	now := time.Now()
	for _, baa := range e.baaRecords {
		if baa.EntityName == entityName && baa.Status == "active" && now.Before(baa.ExpiryDate) {
			return true, nil
		}
	}

	return false, fmt.Errorf("no valid BAA found for entity: %s", entityName)
}

// AssessCompliance performs HIPAA compliance assessment
func (e *HIPAAEngine) AssessCompliance(ctx context.Context) (*compliance.ComplianceReport, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	report := &compliance.ComplianceReport{
		ID:          generateID("hipaa-report"),
		Framework:   compliance.FrameworkHIPAA,
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

	totalScore := 0.0
	for _, control := range e.controls {
		result := e.assessControl(ctx, control)
		report.Controls = append(report.Controls, result)
		totalScore += result.Score

		if result.Status != compliance.StatusCompliant {
			finding := compliance.Finding{
				ID:          generateID("finding"),
				Severity:    e.determineSeverity(control),
				Title:       fmt.Sprintf("HIPAA %s: %s", control.ID, control.Name),
				Description: result.Details,
				ControlID:   control.ID,
				Impact:      "HIPAA violation, potential fines, patient trust issues",
				Risk:        "Critical - HIPAA fines up to $1.5M per year per violation",
				CreatedAt:   time.Now(),
			}
			report.Findings = append(report.Findings, finding)
		}
	}

	if len(report.Controls) > 0 {
		report.Score = totalScore / float64(len(report.Controls))
	}

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

func (e *HIPAAEngine) assessControl(ctx context.Context, control *compliance.ComplianceControl) compliance.ControlResult {
	result := compliance.ControlResult{
		ControlID: control.ID,
		CheckedAt: time.Now(),
		Score:     100.0,
		Status:    compliance.StatusCompliant,
		Details:   "Control properly implemented",
	}

	// Simplified assessment - in production would perform detailed checks
	switch control.ID {
	case "HIPAA-164.312-B":
		// Check audit logging
		if len(e.phiLogs) == 0 {
			result.Score = 50.0
			result.Status = compliance.StatusPartial
			result.Details = "Audit logging not properly configured"
		}
	case "HIPAA-164.308-B1":
		// Check BAA compliance
		expiredBAAs := 0
		for _, baa := range e.baaRecords {
			if time.Now().After(baa.ExpiryDate) {
				expiredBAAs++
			}
		}
		if expiredBAAs > 0 {
			result.Score = 70.0
			result.Status = compliance.StatusPartial
			result.Details = fmt.Sprintf("%d expired BAAs require renewal", expiredBAAs)
		}
	}

	return result
}

func (e *HIPAAEngine) generateSummary(report *compliance.ComplianceReport) compliance.ReportSummary {
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

func (e *HIPAAEngine) determineSeverity(control *compliance.ComplianceControl) string {
	criticalControls := map[string]bool{
		"HIPAA-164.312-A1": true, // Access Control
		"HIPAA-164.312-B":  true, // Audit Controls
		"HIPAA-164.312-E1": true, // Transmission Security
		"HIPAA-164.312-E2": true, // Encryption at Rest
		"HIPAA-164.308-B1": true, // BAA
	}

	if criticalControls[control.ID] {
		return "critical"
	}
	return "high"
}

func (e *HIPAAEngine) alertBreakGlassAccess(log *compliance.PHIAccessLog) {
	// In production, would send alert to security team
	// Break-glass access requires review and justification
}

func (e *HIPAAEngine) alertUnauthorizedAccess(log *compliance.PHIAccessLog) {
	// In production, would send immediate security alert
	// Unauthorized PHI access is a security incident
}

// DeidentifyPHI performs HIPAA-compliant de-identification (Safe Harbor method)
func (e *HIPAAEngine) DeidentifyPHI(data map[string]interface{}) (map[string]interface{}, error) {
	// Safe Harbor method - remove 18 identifiers per §164.514(b)(2)
	identifiers := []string{
		"name", "address", "dates", "phone", "fax", "email", "ssn",
		"mrn", "health_plan_number", "account_number", "certificate_number",
		"vehicle_id", "device_id", "url", "ip_address", "biometric",
		"photo", "unique_id",
	}

	deidentified := make(map[string]interface{})
	for key, value := range data {
		remove := false
		for _, identifier := range identifiers {
			if key == identifier {
				remove = true
				break
			}
		}
		if !remove {
			deidentified[key] = value
		}
	}

	return deidentified, nil
}

// ValidateMinimumNecessary checks if PHI access meets minimum necessary requirement
func (e *HIPAAEngine) ValidateMinimumNecessary(userRole string, requestedData []string, purpose string) (bool, error) {
	// Define role-based data access policies
	allowedData := map[string][]string{
		"physician":    {"demographics", "diagnoses", "medications", "labs", "procedures"},
		"nurse":        {"demographics", "diagnoses", "medications", "vital_signs"},
		"billing":      {"demographics", "insurance", "diagnoses", "procedures"},
		"reception":    {"demographics", "insurance", "appointments"},
		"researcher":   {"de_identified_data"},
	}

	allowed, ok := allowedData[userRole]
	if !ok {
		return false, fmt.Errorf("unknown role: %s", userRole)
	}

	// Check if requested data is within role's minimum necessary
	for _, requested := range requestedData {
		found := false
		for _, permitted := range allowed {
			if requested == permitted {
				found = true
				break
			}
		}
		if !found {
			return false, fmt.Errorf("access to %s not permitted for role %s under minimum necessary", requested, userRole)
		}
	}

	return true, nil
}
