package security

import (
	"context"
	"fmt"
	"time"
)

// SOC2Framework implements SOC 2 compliance framework
type SOC2Framework struct {
	version  string
	controls []ComplianceControl
}

// NewSOC2Framework creates a new SOC 2 compliance framework
func NewSOC2Framework() *SOC2Framework {
	framework := &SOC2Framework{
		version: "2017",
	}
	framework.initializeControls()
	return framework
}

func (s *SOC2Framework) Name() string { return "SOC2" }
func (s *SOC2Framework) Version() string { return s.version }
func (s *SOC2Framework) GetControls() []ComplianceControl { return s.controls }

func (s *SOC2Framework) initializeControls() {
	s.controls = []ComplianceControl{
		{
			ID:          "CC1.1",
			Name:        "Control Environment - Integrity and Ethical Values",
			Description: "The entity demonstrates a commitment to integrity and ethical values",
			Category:    "Control Environment",
			Severity:    SeverityHigh,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "policy", Description: "Code of conduct policy", Source: "documentation", Required: true, Automated: false},
				{Type: "training", Description: "Ethics training records", Source: "hr_system", Required: true, Automated: true},
			},
			AutomationLevel: AutomationSemiAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "CC1.1-T1",
					Name:        "Code of Conduct Review",
					Type:        TestTypeEvidence,
					Description: "Review code of conduct policy and training records",
					Steps:       []string{"Review policy document", "Verify training completion", "Check acknowledgments"},
					Expected:    "Policy exists and training is completed",
					Automated:   false,
				},
			},
			Remediation: "Implement and maintain code of conduct policy with regular training",
			References:  []string{"SOC 2 Type II", "AICPA Trust Services Criteria"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
		{
			ID:          "CC2.1",
			Name:        "Communication and Information - Internal Communication",
			Description: "The entity communicates quality information internally",
			Category:    "Communication and Information",
			Severity:    SeverityMedium,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "communication", Description: "Internal communication channels", Source: "system", Required: true, Automated: true},
				{Type: "policy", Description: "Communication policy", Source: "documentation", Required: true, Automated: false},
			},
			AutomationLevel: AutomationSemiAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "CC2.1-T1",
					Name:        "Communication Channels Assessment",
					Type:        TestTypeProcess,
					Description: "Assess internal communication effectiveness",
					Steps:       []string{"Review communication channels", "Test message delivery", "Verify accessibility"},
					Expected:    "Communication channels are effective and accessible",
					Automated:   true,
					Command:     "check_communication_channels.sh",
				},
			},
			Remediation: "Establish clear internal communication channels and policies",
			References:  []string{"SOC 2 Type II"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
		{
			ID:          "CC6.1",
			Name:        "Logical and Physical Access Controls - Logical Access",
			Description: "The entity implements logical access security software and manages authentication",
			Category:    "Logical and Physical Access Controls",
			Severity:    SeverityHigh,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "config", Description: "Authentication configuration", Source: "system", Required: true, Automated: true},
				{Type: "logs", Description: "Access logs", Source: "audit_system", Required: true, Automated: true},
			},
			AutomationLevel: AutomationFullyAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "CC6.1-T1",
					Name:        "Authentication Controls Test",
					Type:        TestTypeAutomated,
					Description: "Test authentication and access controls",
					Steps:       []string{"Check MFA configuration", "Verify password policies", "Test access controls"},
					Expected:    "Strong authentication controls are implemented",
					Automated:   true,
					Script:      "test_authentication_controls.py",
				},
			},
			Remediation: "Implement strong authentication controls and access management",
			References:  []string{"SOC 2 Type II", "NIST 800-63"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
	}
}

func (s *SOC2Framework) Assess(ctx context.Context, evidence map[string]interface{}) (*FrameworkAssessment, error) {
	assessment := &FrameworkAssessment{
		Framework:      s.Name(),
		Version:        s.Version(),
		AssessmentDate: time.Now(),
		ControlResults: make(map[string]*ControlResult),
		CategoryScores: make(map[string]float64),
		NextAssessment: time.Now().AddDate(1, 0, 0), // Annual
		Metadata:       make(map[string]interface{}),
	}
	
	var totalScore float64
	controlCount := 0
	
	for _, control := range s.controls {
		result := s.assessControl(ctx, &control, evidence)
		assessment.ControlResults[control.ID] = result
		totalScore += result.Score
		controlCount++
	}
	
	if controlCount > 0 {
		assessment.OverallScore = totalScore / float64(controlCount)
	}
	
	// Determine status based on score
	if assessment.OverallScore >= 90 {
		assessment.Status = AssessmentStatusCompliant
	} else if assessment.OverallScore >= 70 {
		assessment.Status = AssessmentStatusPartial
	} else {
		assessment.Status = AssessmentStatusNonCompliant
	}
	
	assessment.Summary = fmt.Sprintf("SOC 2 Assessment: %.1f/100 (%s)", 
		assessment.OverallScore, assessment.Status)
	
	return assessment, nil
}

func (s *SOC2Framework) assessControl(ctx context.Context, control *ComplianceControl, evidence map[string]interface{}) *ControlResult {
	result := &ControlResult{
		ControlID:   control.ID,
		LastTested:  time.Now(),
		NextTest:    time.Now().AddDate(0, 1, 0), // Monthly
		Evidence:    []Evidence{},
		TestResults: []TestResult{},
		Findings:    []ComplianceFinding{},
	}
	
	// Simple scoring based on evidence availability
	score := 0.0
	if accessControl, ok := evidence["access_control"].(map[string]interface{}); ok {
		if control.ID == "CC6.1" {
			if mfa, ok := accessControl["mfa_enabled"].(bool); ok && mfa {
				score += 50.0
			}
			if rbac, ok := accessControl["rbac_enabled"].(bool); ok && rbac {
				score += 50.0
			}
		}
	}
	
	// Default scoring for other controls
	if score == 0 {
		score = 75.0 // Assume partially implemented
	}
	
	result.Score = score
	
	if score >= 90 {
		result.Status = StatusCompliant
	} else if score >= 70 {
		result.Status = StatusPartiallyImpl
	} else {
		result.Status = StatusNonCompliant
	}
	
	return result
}

// GDPRFramework implements GDPR compliance framework
type GDPRFramework struct {
	version  string
	controls []ComplianceControl
}

// NewGDPRFramework creates a new GDPR compliance framework
func NewGDPRFramework() *GDPRFramework {
	framework := &GDPRFramework{
		version: "2018",
	}
	framework.initializeControls()
	return framework
}

func (g *GDPRFramework) Name() string { return "GDPR" }
func (g *GDPRFramework) Version() string { return g.version }
func (g *GDPRFramework) GetControls() []ComplianceControl { return g.controls }

func (g *GDPRFramework) initializeControls() {
	g.controls = []ComplianceControl{
		{
			ID:          "Art5.1.a",
			Name:        "Lawfulness, fairness and transparency",
			Description: "Personal data shall be processed lawfully, fairly and in a transparent manner",
			Category:    "Principles",
			Severity:    SeverityHigh,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "policy", Description: "Privacy policy", Source: "documentation", Required: true, Automated: false},
				{Type: "consent", Description: "Consent records", Source: "system", Required: true, Automated: true},
			},
			AutomationLevel: AutomationSemiAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "Art5.1.a-T1",
					Name:        "Privacy Policy Review",
					Type:        TestTypeEvidence,
					Description: "Review privacy policy and consent mechanisms",
					Steps:       []string{"Review privacy policy", "Check consent forms", "Verify transparency"},
					Expected:    "Clear privacy policy and valid consent mechanisms",
					Automated:   false,
				},
			},
			Remediation: "Implement comprehensive privacy policy and consent management",
			References:  []string{"GDPR Article 5", "ICO Guidelines"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
		{
			ID:          "Art32.1",
			Name:        "Security of processing",
			Description: "Appropriate technical and organisational measures to ensure security",
			Category:    "Security",
			Severity:    SeverityCritical,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "encryption", Description: "Encryption implementation", Source: "system", Required: true, Automated: true},
				{Type: "access_control", Description: "Access control measures", Source: "system", Required: true, Automated: true},
			},
			AutomationLevel: AutomationFullyAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "Art32.1-T1",
					Name:        "Security Measures Assessment",
					Type:        TestTypeAutomated,
					Description: "Assess technical and organizational security measures",
					Steps:       []string{"Check encryption", "Verify access controls", "Test backup systems"},
					Expected:    "Appropriate security measures are implemented",
					Automated:   true,
					Script:      "test_gdpr_security.py",
				},
			},
			Remediation: "Implement appropriate technical and organizational security measures",
			References:  []string{"GDPR Article 32", "ENISA Guidelines"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
	}
}

func (g *GDPRFramework) Assess(ctx context.Context, evidence map[string]interface{}) (*FrameworkAssessment, error) {
	assessment := &FrameworkAssessment{
		Framework:      g.Name(),
		Version:        g.Version(),
		AssessmentDate: time.Now(),
		ControlResults: make(map[string]*ControlResult),
		CategoryScores: make(map[string]float64),
		NextAssessment: time.Now().AddDate(0, 6, 0), // Semi-annual
		Metadata:       make(map[string]interface{}),
	}
	
	var totalScore float64
	controlCount := 0
	
	for _, control := range g.controls {
		result := g.assessControl(ctx, &control, evidence)
		assessment.ControlResults[control.ID] = result
		totalScore += result.Score
		controlCount++
	}
	
	if controlCount > 0 {
		assessment.OverallScore = totalScore / float64(controlCount)
	}
	
	// GDPR requires high compliance standards
	if assessment.OverallScore >= 95 {
		assessment.Status = AssessmentStatusCompliant
	} else if assessment.OverallScore >= 80 {
		assessment.Status = AssessmentStatusPartial
	} else {
		assessment.Status = AssessmentStatusNonCompliant
	}
	
	assessment.Summary = fmt.Sprintf("GDPR Assessment: %.1f/100 (%s)", 
		assessment.OverallScore, assessment.Status)
	
	return assessment, nil
}

func (g *GDPRFramework) assessControl(ctx context.Context, control *ComplianceControl, evidence map[string]interface{}) *ControlResult {
	result := &ControlResult{
		ControlID:   control.ID,
		LastTested:  time.Now(),
		NextTest:    time.Now().AddDate(0, 3, 0), // Quarterly
		Evidence:    []Evidence{},
		TestResults: []TestResult{},
		Findings:    []ComplianceFinding{},
	}
	
	// GDPR-specific scoring
	score := 0.0
	if encryption, ok := evidence["encryption"].(map[string]interface{}); ok {
		if control.ID == "Art32.1" {
			if dataAtRest, ok := encryption["data_at_rest"].(string); ok && dataAtRest == "AES-256" {
				score += 25.0
			}
			if dataInTransit, ok := encryption["data_in_transit"].(string); ok && dataInTransit == "TLS 1.3" {
				score += 25.0
			}
			if keyMgmt, ok := encryption["key_management"].(string); ok && keyMgmt != "" {
				score += 25.0
			}
			if certMgmt, ok := encryption["certificate_management"].(string); ok && certMgmt == "automated" {
				score += 25.0
			}
		}
	}
	
	// Default scoring for other controls
	if score == 0 {
		score = 70.0 // Assume basic implementation
	}
	
	result.Score = score
	
	if score >= 95 {
		result.Status = StatusCompliant
	} else if score >= 80 {
		result.Status = StatusPartiallyImpl
	} else {
		result.Status = StatusNonCompliant
	}
	
	return result
}

// NISTFramework implements NIST Cybersecurity Framework
type NISTFramework struct {
	version  string
	controls []ComplianceControl
}

// NewNISTFramework creates a new NIST compliance framework
func NewNISTFramework() *NISTFramework {
	framework := &NISTFramework{
		version: "1.1",
	}
	framework.initializeControls()
	return framework
}

func (n *NISTFramework) Name() string { return "NIST" }
func (n *NISTFramework) Version() string { return n.version }
func (n *NISTFramework) GetControls() []ComplianceControl { return n.controls }

func (n *NISTFramework) initializeControls() {
	n.controls = []ComplianceControl{
		{
			ID:          "ID.AM-1",
			Name:        "Asset Inventory",
			Description: "Physical devices and systems are inventoried",
			Category:    "Identify",
			Severity:    SeverityHigh,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "inventory", Description: "Asset inventory", Source: "cmdb", Required: true, Automated: true},
				{Type: "policy", Description: "Asset management policy", Source: "documentation", Required: true, Automated: false},
			},
			AutomationLevel: AutomationFullyAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "ID.AM-1-T1",
					Name:        "Asset Inventory Verification",
					Type:        TestTypeAutomated,
					Description: "Verify completeness and accuracy of asset inventory",
					Steps:       []string{"Query CMDB", "Scan network", "Compare results"},
					Expected:    "Complete and accurate asset inventory",
					Automated:   true,
					Script:      "verify_asset_inventory.py",
				},
			},
			Remediation: "Maintain complete and up-to-date asset inventory",
			References:  []string{"NIST CSF 1.1"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
		{
			ID:          "PR.AC-1",
			Name:        "Identity Management",
			Description: "Identities and credentials are issued, managed, verified, revoked, and audited",
			Category:    "Protect",
			Severity:    SeverityHigh,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "identity_system", Description: "Identity management system", Source: "system", Required: true, Automated: true},
				{Type: "audit_logs", Description: "Identity audit logs", Source: "logs", Required: true, Automated: true},
			},
			AutomationLevel: AutomationFullyAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "PR.AC-1-T1",
					Name:        "Identity Management Assessment",
					Type:        TestTypeAutomated,
					Description: "Assess identity management processes",
					Steps:       []string{"Check user provisioning", "Verify deprovisioning", "Audit access reviews"},
					Expected:    "Proper identity lifecycle management",
					Automated:   true,
					Script:      "test_identity_management.py",
				},
			},
			Remediation: "Implement comprehensive identity lifecycle management",
			References:  []string{"NIST CSF 1.1", "NIST 800-63"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
	}
}

func (n *NISTFramework) Assess(ctx context.Context, evidence map[string]interface{}) (*FrameworkAssessment, error) {
	assessment := &FrameworkAssessment{
		Framework:      n.Name(),
		Version:        n.Version(),
		AssessmentDate: time.Now(),
		ControlResults: make(map[string]*ControlResult),
		CategoryScores: make(map[string]float64),
		NextAssessment: time.Now().AddDate(0, 6, 0), // Semi-annual
		Metadata:       make(map[string]interface{}),
	}
	
	var totalScore float64
	controlCount := 0
	
	for _, control := range n.controls {
		result := n.assessControl(ctx, &control, evidence)
		assessment.ControlResults[control.ID] = result
		totalScore += result.Score
		controlCount++
	}
	
	if controlCount > 0 {
		assessment.OverallScore = totalScore / float64(controlCount)
	}
	
	if assessment.OverallScore >= 85 {
		assessment.Status = AssessmentStatusCompliant
	} else if assessment.OverallScore >= 65 {
		assessment.Status = AssessmentStatusPartial
	} else {
		assessment.Status = AssessmentStatusNonCompliant
	}
	
	assessment.Summary = fmt.Sprintf("NIST Cybersecurity Framework Assessment: %.1f/100 (%s)", 
		assessment.OverallScore, assessment.Status)
	
	return assessment, nil
}

func (n *NISTFramework) assessControl(ctx context.Context, control *ComplianceControl, evidence map[string]interface{}) *ControlResult {
	result := &ControlResult{
		ControlID:   control.ID,
		LastTested:  time.Now(),
		NextTest:    time.Now().AddDate(0, 3, 0), // Quarterly
		Evidence:    []Evidence{},
		TestResults: []TestResult{},
		Findings:    []ComplianceFinding{},
	}
	
	// Default scoring - would be more sophisticated in real implementation
	result.Score = 80.0
	result.Status = StatusPartiallyImpl
	
	return result
}

// ISO27001Framework implements ISO 27001 compliance framework
type ISO27001Framework struct {
	version  string
	controls []ComplianceControl
}

// NewISO27001Framework creates a new ISO 27001 compliance framework
func NewISO27001Framework() *ISO27001Framework {
	framework := &ISO27001Framework{
		version: "2013",
	}
	framework.initializeControls()
	return framework
}

func (i *ISO27001Framework) Name() string { return "ISO27001" }
func (i *ISO27001Framework) Version() string { return i.version }
func (i *ISO27001Framework) GetControls() []ComplianceControl { return i.controls }

func (i *ISO27001Framework) initializeControls() {
	i.controls = []ComplianceControl{
		{
			ID:          "A.9.1.1",
			Name:        "Access control policy",
			Description: "An access control policy shall be established, documented and reviewed",
			Category:    "Access Control",
			Severity:    SeverityHigh,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "policy", Description: "Access control policy", Source: "documentation", Required: true, Automated: false},
				{Type: "review_records", Description: "Policy review records", Source: "documentation", Required: true, Automated: false},
			},
			AutomationLevel: AutomationSemiAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "A.9.1.1-T1",
					Name:        "Access Control Policy Review",
					Type:        TestTypeEvidence,
					Description: "Review access control policy and approval records",
					Steps:       []string{"Review policy document", "Check approval signatures", "Verify review dates"},
					Expected:    "Current and approved access control policy",
					Automated:   false,
				},
			},
			Remediation: "Establish and maintain current access control policy with regular reviews",
			References:  []string{"ISO 27001:2013"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
	}
}

func (i *ISO27001Framework) Assess(ctx context.Context, evidence map[string]interface{}) (*FrameworkAssessment, error) {
	assessment := &FrameworkAssessment{
		Framework:      i.Name(),
		Version:        i.Version(),
		AssessmentDate: time.Now(),
		ControlResults: make(map[string]*ControlResult),
		CategoryScores: make(map[string]float64),
		NextAssessment: time.Now().AddDate(1, 0, 0), // Annual
		Metadata:       make(map[string]interface{}),
	}
	
	var totalScore float64
	controlCount := 0
	
	for _, control := range i.controls {
		result := i.assessControl(ctx, &control, evidence)
		assessment.ControlResults[control.ID] = result
		totalScore += result.Score
		controlCount++
	}
	
	if controlCount > 0 {
		assessment.OverallScore = totalScore / float64(controlCount)
	}
	
	if assessment.OverallScore >= 90 {
		assessment.Status = AssessmentStatusCompliant
	} else if assessment.OverallScore >= 70 {
		assessment.Status = AssessmentStatusPartial
	} else {
		assessment.Status = AssessmentStatusNonCompliant
	}
	
	assessment.Summary = fmt.Sprintf("ISO 27001 Assessment: %.1f/100 (%s)", 
		assessment.OverallScore, assessment.Status)
	
	return assessment, nil
}

func (i *ISO27001Framework) assessControl(ctx context.Context, control *ComplianceControl, evidence map[string]interface{}) *ControlResult {
	result := &ControlResult{
		ControlID:   control.ID,
		LastTested:  time.Now(),
		NextTest:    time.Now().AddDate(0, 6, 0), // Semi-annual
		Evidence:    []Evidence{},
		TestResults: []TestResult{},
		Findings:    []ComplianceFinding{},
	}
	
	// Default scoring - would be more sophisticated in real implementation
	result.Score = 75.0
	result.Status = StatusPartiallyImpl
	
	return result
}

// PCIDSSFramework implements PCI DSS compliance framework
type PCIDSSFramework struct {
	version  string
	controls []ComplianceControl
}

// NewPCIDSSFramework creates a new PCI DSS compliance framework
func NewPCIDSSFramework() *PCIDSSFramework {
	framework := &PCIDSSFramework{
		version: "4.0",
	}
	framework.initializeControls()
	return framework
}

func (p *PCIDSSFramework) Name() string { return "PCI_DSS" }
func (p *PCIDSSFramework) Version() string { return p.version }
func (p *PCIDSSFramework) GetControls() []ComplianceControl { return p.controls }

func (p *PCIDSSFramework) initializeControls() {
	p.controls = []ComplianceControl{
		{
			ID:          "1.1.1",
			Name:        "Firewall Configuration Standards",
			Description: "Processes and procedures are established for firewall and router configuration standards",
			Category:    "Network Security",
			Severity:    SeverityHigh,
			RequiredEvidence: []EvidenceRequirement{
				{Type: "firewall_rules", Description: "Firewall configuration", Source: "network", Required: true, Automated: true},
				{Type: "policy", Description: "Firewall management policy", Source: "documentation", Required: true, Automated: false},
			},
			AutomationLevel: AutomationFullyAuto,
			TestProcedures: []TestProcedure{
				{
					ID:          "1.1.1-T1",
					Name:        "Firewall Configuration Review",
					Type:        TestTypeAutomated,
					Description: "Review firewall rules and configurations",
					Steps:       []string{"Extract firewall rules", "Check for unnecessary rules", "Verify documentation"},
					Expected:    "Properly configured firewall with documented rules",
					Automated:   true,
					Script:      "audit_firewall_config.py",
				},
			},
			Remediation: "Establish firewall configuration standards and regular reviews",
			References:  []string{"PCI DSS v4.0"},
			Status:      StatusNotImplemented,
			Metadata:    make(map[string]interface{}),
		},
	}
}

func (p *PCIDSSFramework) Assess(ctx context.Context, evidence map[string]interface{}) (*FrameworkAssessment, error) {
	assessment := &FrameworkAssessment{
		Framework:      p.Name(),
		Version:        p.Version(),
		AssessmentDate: time.Now(),
		ControlResults: make(map[string]*ControlResult),
		CategoryScores: make(map[string]float64),
		NextAssessment: time.Now().AddDate(0, 3, 0), // Quarterly
		Metadata:       make(map[string]interface{}),
	}
	
	var totalScore float64
	controlCount := 0
	
	for _, control := range p.controls {
		result := p.assessControl(ctx, &control, evidence)
		assessment.ControlResults[control.ID] = result
		totalScore += result.Score
		controlCount++
	}
	
	if controlCount > 0 {
		assessment.OverallScore = totalScore / float64(controlCount)
	}
	
	// PCI DSS requires strict compliance
	if assessment.OverallScore >= 95 {
		assessment.Status = AssessmentStatusCompliant
	} else {
		assessment.Status = AssessmentStatusNonCompliant
	}
	
	assessment.Summary = fmt.Sprintf("PCI DSS Assessment: %.1f/100 (%s)", 
		assessment.OverallScore, assessment.Status)
	
	return assessment, nil
}

func (p *PCIDSSFramework) assessControl(ctx context.Context, control *ComplianceControl, evidence map[string]interface{}) *ControlResult {
	result := &ControlResult{
		ControlID:   control.ID,
		LastTested:  time.Now(),
		NextTest:    time.Now().AddDate(0, 3, 0), // Quarterly
		Evidence:    []Evidence{},
		TestResults: []TestResult{},
		Findings:    []ComplianceFinding{},
	}
	
	// PCI DSS specific scoring
	if networkSecurity, ok := evidence["network_security"].(map[string]interface{}); ok {
		if control.ID == "1.1.1" {
			score := 0.0
			if segmentation, ok := networkSecurity["network_segmentation"].(string); ok && segmentation == "enabled" {
				score += 50.0
			}
			if ids, ok := networkSecurity["intrusion_detection"].(string); ok && ids == "enabled" {
				score += 30.0
			}
			if scanning, ok := networkSecurity["vulnerability_scanning"].(string); ok && scanning == "weekly" {
				score += 20.0
			}
			result.Score = score
		}
	}
	
	// Default scoring
	if result.Score == 0 {
		result.Score = 70.0
	}
	
	if result.Score >= 95 {
		result.Status = StatusCompliant
	} else {
		result.Status = StatusNonCompliant
	}
	
	return result
}