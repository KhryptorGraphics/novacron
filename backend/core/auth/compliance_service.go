package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ComplianceFramework represents different compliance frameworks
type ComplianceFramework string

const (
	SOC2     ComplianceFramework = "SOC2"
	GDPR     ComplianceFramework = "GDPR"
	HIPAA    ComplianceFramework = "HIPAA"
	PCIDSS   ComplianceFramework = "PCI-DSS"
	ISO27001 ComplianceFramework = "ISO27001"
	NIST     ComplianceFramework = "NIST"
)

// ComplianceStatus represents compliance check status
type ComplianceStatus string

const (
	Compliant    ComplianceStatus = "compliant"
	NonCompliant ComplianceStatus = "non_compliant"
	Partial      ComplianceStatus = "partial"
	NotTested    ComplianceStatus = "not_tested"
	Exempt       ComplianceStatus = "exempt"
)

// ComplianceControl represents a compliance control requirement
type ComplianceControl struct {
	ID          string              `json:"id"`
	Framework   ComplianceFramework `json:"framework"`
	Title       string              `json:"title"`
	Description string              `json:"description"`
	Category    string              `json:"category"`
	Severity    string              `json:"severity"` // low, medium, high, critical
	Required    bool                `json:"required"`
	Automated   bool                `json:"automated"`
	Testable    bool                `json:"testable"`
	Evidence    []string            `json:"evidence,omitempty"`
	References  []string            `json:"references,omitempty"`
	Tags        []string            `json:"tags,omitempty"`
	CreatedAt   time.Time           `json:"created_at"`
	UpdatedAt   time.Time           `json:"updated_at"`
}

// ComplianceAssessment represents a compliance assessment result
type ComplianceAssessment struct {
	ID              string                     `json:"id"`
	Framework       ComplianceFramework        `json:"framework"`
	TenantID        string                     `json:"tenant_id,omitempty"`
	AssessorID      string                     `json:"assessor_id"`
	StartDate       time.Time                  `json:"start_date"`
	EndDate         time.Time                  `json:"end_date"`
	Status          ComplianceStatus           `json:"status"`
	OverallScore    float64                    `json:"overall_score"`
	ControlResults  map[string]*ControlResult  `json:"control_results"`
	Findings        []ComplianceFinding        `json:"findings"`
	Recommendations []ComplianceRecommendation `json:"recommendations"`
	Evidence        []ComplianceEvidence       `json:"evidence"`
	Metadata        map[string]interface{}     `json:"metadata,omitempty"`
	CreatedAt       time.Time                  `json:"created_at"`
	UpdatedAt       time.Time                  `json:"updated_at"`
}

// ControlResult represents the result of testing a specific control
type ControlResult struct {
	ControlID   string           `json:"control_id"`
	Status      ComplianceStatus `json:"status"`
	Score       float64          `json:"score"`
	Tested      bool             `json:"tested"`
	TestedAt    time.Time        `json:"tested_at"`
	TesterID    string           `json:"tester_id,omitempty"`
	Evidence    []string         `json:"evidence,omitempty"`
	Findings    []string         `json:"findings,omitempty"`
	Remediation string           `json:"remediation,omitempty"`
	Notes       string           `json:"notes,omitempty"`
	NextTest    time.Time        `json:"next_test,omitempty"`
}

// ComplianceFinding represents a compliance issue or gap
type ComplianceFinding struct {
	ID          string              `json:"id"`
	ControlID   string              `json:"control_id"`
	Framework   ComplianceFramework `json:"framework"`
	Severity    string              `json:"severity"`
	Title       string              `json:"title"`
	Description string              `json:"description"`
	Risk        string              `json:"risk"`
	Impact      string              `json:"impact"`
	Status      string              `json:"status"` // open, in_progress, resolved, accepted
	AssignedTo  string              `json:"assigned_to,omitempty"`
	DueDate     time.Time           `json:"due_date,omitempty"`
	CreatedAt   time.Time           `json:"created_at"`
	ResolvedAt  time.Time           `json:"resolved_at,omitempty"`
}

// ComplianceRecommendation represents recommended actions
type ComplianceRecommendation struct {
	ID          string              `json:"id"`
	Framework   ComplianceFramework `json:"framework"`
	Priority    string              `json:"priority"` // low, medium, high, critical
	Title       string              `json:"title"`
	Description string              `json:"description"`
	Actions     []string            `json:"actions"`
	Timeframe   string              `json:"timeframe"`
	Effort      string              `json:"effort"`
	Cost        string              `json:"cost,omitempty"`
	Benefit     string              `json:"benefit"`
	Status      string              `json:"status"` // pending, in_progress, completed
	CreatedAt   time.Time           `json:"created_at"`
}

// ComplianceEvidence represents evidence for compliance
type ComplianceEvidence struct {
	ID          string              `json:"id"`
	ControlID   string              `json:"control_id"`
	Framework   ComplianceFramework `json:"framework"`
	Type        string              `json:"type"` // document, screenshot, log, config, etc.
	Title       string              `json:"title"`
	Description string              `json:"description,omitempty"`
	URL         string              `json:"url,omitempty"`
	Hash        string              `json:"hash,omitempty"`
	Size        int64               `json:"size,omitempty"`
	CreatedBy   string              `json:"created_by"`
	CreatedAt   time.Time           `json:"created_at"`
}

// CompliancePolicy represents a compliance policy
type CompliancePolicy struct {
	ID            string              `json:"id"`
	Framework     ComplianceFramework `json:"framework"`
	Name          string              `json:"name"`
	Description   string              `json:"description"`
	Owner         string              `json:"owner"`
	Approver      string              `json:"approver,omitempty"`
	Version       string              `json:"version"`
	EffectiveDate time.Time           `json:"effective_date"`
	ReviewDate    time.Time           `json:"review_date"`
	Content       string              `json:"content"`
	Controls      []string            `json:"controls"`
	Status        string              `json:"status"` // draft, active, archived
	CreatedAt     time.Time           `json:"created_at"`
	UpdatedAt     time.Time           `json:"updated_at"`
}

// ComplianceService provides compliance validation and management
type ComplianceService struct {
	controls     map[string]*ComplianceControl
	assessments  map[string]*ComplianceAssessment
	policies     map[string]*CompliancePolicy
	mu           sync.RWMutex
	auditService AuditService
	encryption   *EncryptionService

	// Optional real-signal sources for the automated checks below. Each is
	// nil until wired via the matching SetXxx method; a nil source means
	// the corresponding checks honestly report NotTested instead of
	// fabricating a result. See novacron-yvo.
	passwordService  *PasswordSecurityService
	twoFactorService *TwoFactorService
	ztNetworkService *ZeroTrustNetworkService
	roleService      RoleService
}

// NewComplianceService creates a new compliance service
func NewComplianceService(auditService AuditService, encryptionService *EncryptionService) *ComplianceService {
	service := &ComplianceService{
		controls:     make(map[string]*ComplianceControl),
		assessments:  make(map[string]*ComplianceAssessment),
		policies:     make(map[string]*CompliancePolicy),
		auditService: auditService,
		encryption:   encryptionService,
	}

	// Load default controls
	service.loadDefaultControls()

	return service
}

// SetPasswordService wires a password-security service so password-policy
// and default-password checks reflect real configuration instead of
// assuming compliance.
func (c *ComplianceService) SetPasswordService(ps *PasswordSecurityService) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.passwordService = ps
}

// SetTwoFactorService wires an existing 2FA service so the MFA check
// reflects real adoption instead of assuming it is enabled.
func (c *ComplianceService) SetTwoFactorService(tfs *TwoFactorService) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.twoFactorService = tfs
}

// SetZeroTrustNetworkService wires a zero-trust network service so
// firewall, segmentation, and transmission-encryption checks reflect real
// policy state instead of assuming compliance.
func (c *ComplianceService) SetZeroTrustNetworkService(zt *ZeroTrustNetworkService) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ztNetworkService = zt
}

// SetRoleService wires a role service so RBAC-related checks reflect real
// role/permission definitions instead of assuming compliance.
func (c *ComplianceService) SetRoleService(rs RoleService) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.roleService = rs
}

// CreateAssessment creates a new compliance assessment
func (c *ComplianceService) CreateAssessment(framework ComplianceFramework, tenantID, assessorID string) (*ComplianceAssessment, error) {
	assessment := &ComplianceAssessment{
		ID:              c.generateID(),
		Framework:       framework,
		TenantID:        tenantID,
		AssessorID:      assessorID,
		StartDate:       time.Now(),
		Status:          NotTested,
		ControlResults:  make(map[string]*ControlResult),
		Findings:        make([]ComplianceFinding, 0),
		Recommendations: make([]ComplianceRecommendation, 0),
		Evidence:        make([]ComplianceEvidence, 0),
		Metadata:        make(map[string]interface{}),
		CreatedAt:       time.Now(),
		UpdatedAt:       time.Now(),
	}

	// Initialize control results for framework
	for _, control := range c.getControlsForFramework(framework) {
		assessment.ControlResults[control.ID] = &ControlResult{
			ControlID: control.ID,
			Status:    NotTested,
			Tested:    false,
			Evidence:  make([]string, 0),
			Findings:  make([]string, 0),
		}
	}

	c.mu.Lock()
	c.assessments[assessment.ID] = assessment
	c.mu.Unlock()

	// Log assessment creation
	if c.auditService != nil {
		c.auditService.LogAccess(&AuditEntry{
			UserID:       assessorID,
			TenantID:     tenantID,
			ResourceType: "compliance_assessment",
			ResourceID:   assessment.ID,
			Action:       "create",
			Success:      true,
			Timestamp:    time.Now(),
			AdditionalData: map[string]interface{}{
				"framework": framework,
			},
		})
	}

	return assessment, nil
}

// RunAutomatedTests runs automated compliance tests
func (c *ComplianceService) RunAutomatedTests(ctx context.Context, assessmentID string) error {
	c.mu.Lock()
	assessment, exists := c.assessments[assessmentID]
	c.mu.Unlock()

	if !exists {
		return fmt.Errorf("assessment not found: %s", assessmentID)
	}

	// Run tests based on framework
	switch assessment.Framework {
	case SOC2:
		return c.runSOC2AutomatedTests(ctx, assessment)
	case GDPR:
		return c.runGDPRAutomatedTests(ctx, assessment)
	case HIPAA:
		return c.runHIPAAAutomatedTests(ctx, assessment)
	case PCIDSS:
		return c.runPCIDSSAutomatedTests(ctx, assessment)
	default:
		return fmt.Errorf("unsupported framework for automated testing: %s", assessment.Framework)
	}
}

// runSOC2AutomatedTests runs SOC2 specific automated tests
func (c *ComplianceService) runSOC2AutomatedTests(ctx context.Context, assessment *ComplianceAssessment) error {
	// Test CC1.1 - Management demonstrates commitment to integrity and ethical values
	c.testControlCC1_1(assessment)

	// Test CC2.1 - Communication of information security policies
	c.testControlCC2_1(assessment)

	// Test CC6.1 - Logical and physical access controls
	c.testControlCC6_1(assessment)

	// Test CC6.7 - Data transmission controls
	c.testControlCC6_7(assessment)

	// Test CC7.1 - System monitoring
	c.testControlCC7_1(assessment)

	// Update overall assessment status
	c.updateAssessmentStatus(assessment)

	return nil
}

// runGDPRAutomatedTests runs GDPR specific automated tests
func (c *ComplianceService) runGDPRAutomatedTests(ctx context.Context, assessment *ComplianceAssessment) error {
	// Test Article 25 - Data protection by design and by default
	c.testGDPRArticle25(assessment)

	// Test Article 32 - Security of processing
	c.testGDPRArticle32(assessment)

	// Test Article 33 - Notification of personal data breach
	c.testGDPRArticle33(assessment)

	// Test Article 35 - Data protection impact assessment
	c.testGDPRArticle35(assessment)

	c.updateAssessmentStatus(assessment)

	return nil
}

// runHIPAAAutomatedTests runs HIPAA specific automated tests
func (c *ComplianceService) runHIPAAAutomatedTests(ctx context.Context, assessment *ComplianceAssessment) error {
	// Test 164.308 - Administrative safeguards
	c.testHIPAA164_308(assessment)

	// Test 164.310 - Physical safeguards
	c.testHIPAA164_310(assessment)

	// Test 164.312 - Technical safeguards
	c.testHIPAA164_312(assessment)

	// Test 164.314 - Organizational requirements
	c.testHIPAA164_314(assessment)

	c.updateAssessmentStatus(assessment)

	return nil
}

// runPCIDSSAutomatedTests runs PCI-DSS specific automated tests
func (c *ComplianceService) runPCIDSSAutomatedTests(ctx context.Context, assessment *ComplianceAssessment) error {
	// Test Requirement 1 - Firewall configuration
	c.testPCIDSS1(assessment)

	// Test Requirement 2 - Default passwords and security parameters
	c.testPCIDSS2(assessment)

	// Test Requirement 3 - Protect stored cardholder data
	c.testPCIDSS3(assessment)

	// Test Requirement 4 - Encrypt transmission of cardholder data
	c.testPCIDSS4(assessment)

	c.updateAssessmentStatus(assessment)

	return nil
}

// SOC2 Control Tests
func (c *ComplianceService) testControlCC1_1(assessment *ComplianceAssessment) {
	controlID := "CC1.1"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check if integrity and ethics policies exist
	score := 0
	verified := false

	s, v := applyCheck(result, "Code of conduct policy", c.checkPolicyExists("code_of_conduct"), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Ethics training records", c.checkTrainingRecords("ethics"), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testControlCC2_1(assessment *ComplianceAssessment) {
	controlID := "CC2.1"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check if security policies are documented and communicated
	score := 0
	verified := false

	s, v := applyCheck(result, "Information security policy", c.checkPolicyExists("information_security"), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Policy communication", c.checkPolicyCommunication("information_security"), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testControlCC6_1(assessment *ComplianceAssessment) {
	controlID := "CC6.1"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check logical access controls
	score := 0
	verified := false

	s, v := applyCheck(result, "Multi-factor authentication", c.checkMFAEnabled(), 40)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Password policy", c.checkPasswordPolicy(), 30)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Access reviews", c.checkAccessReviews(), 30)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 80:
		result.Status = Compliant
	case score >= 60:
		result.Status = Partial
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testControlCC6_7(assessment *ComplianceAssessment) {
	controlID := "CC6.7"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check data transmission controls
	score := 0
	verified := false

	s, v := applyCheck(result, "TLS enabled", c.checkTLSEnabled(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Encryption in transit", c.checkEncryptionInTransit(), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testControlCC7_1(assessment *ComplianceAssessment) {
	controlID := "CC7.1"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check system monitoring
	score := 0
	verified := false

	s, v := applyCheck(result, "Logging enabled", c.checkLoggingEnabled(), 40)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "System monitoring", c.checkMonitoring(), 40)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Alerting enabled", c.checkAlertingEnabled(), 20)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 80:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

// GDPR Tests
func (c *ComplianceService) testGDPRArticle25(assessment *ComplianceAssessment) {
	controlID := "GDPR-25"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check data protection by design
	score := 0
	verified := false

	s, v := applyCheck(result, "Privacy by design", c.checkPrivacyByDesign(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Data minimization", c.checkDataMinimization(), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testGDPRArticle32(assessment *ComplianceAssessment) {
	controlID := "GDPR-32"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check security of processing
	score := 0
	verified := false

	s, v := applyCheck(result, "Encryption at rest", c.checkEncryptionAtRest(), 40)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Access controls", c.checkAccessControls(), 40)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Backup security", c.checkBackupSecurity(), 20)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 80:
		result.Status = Compliant
	default:
		result.Status = Partial
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testGDPRArticle33(assessment *ComplianceAssessment) {
	controlID := "GDPR-33"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check breach notification procedures
	score := 0
	verified := false

	s, v := applyCheck(result, "Breach procedures", c.checkBreachProcedures(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Notification timing", c.checkNotificationTiming(), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testGDPRArticle35(assessment *ComplianceAssessment) {
	controlID := "GDPR-35"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check DPIA process
	score := 0
	verified := false

	s, v := applyCheck(result, "DPIA process", c.checkDPIAProcess(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Risk assessment", c.checkRiskAssessment(), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

// HIPAA Tests
func (c *ComplianceService) testHIPAA164_308(assessment *ComplianceAssessment) {
	controlID := "HIPAA-164.308"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check administrative safeguards
	score := 0
	verified := false

	s, v := applyCheck(result, "Security officer designated", c.checkSecurityOfficer(), 40)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Workforce training", c.checkWorkforceTraining(), 30)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Access management", c.checkAccessManagement(), 30)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 80:
		result.Status = Compliant
	default:
		result.Status = Partial
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testHIPAA164_310(assessment *ComplianceAssessment) {
	controlID := "HIPAA-164.310"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check physical safeguards
	score := 0
	verified := false

	s, v := applyCheck(result, "Facility access controls", c.checkFacilityAccess(), 40)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Workstation security", c.checkWorkstationSecurity(), 30)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Device and media controls", c.checkDeviceControls(), 30)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 80:
		result.Status = Compliant
	default:
		result.Status = Partial
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testHIPAA164_312(assessment *ComplianceAssessment) {
	controlID := "HIPAA-164.312"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check technical safeguards
	score := 0
	verified := false

	s, v := applyCheck(result, "Technical access control", c.checkTechnicalAccessControl(), 30)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Audit controls", c.checkAuditControls(), 25)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Data integrity", c.checkDataIntegrity(), 25)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Transmission security", c.checkTransmissionSecurity(), 20)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 80:
		result.Status = Compliant
	default:
		result.Status = Partial
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testHIPAA164_314(assessment *ComplianceAssessment) {
	controlID := "HIPAA-164.314"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check organizational requirements
	score := 0
	verified := false

	s, v := applyCheck(result, "Business associate agreements", c.checkBusinessAssociates(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Contractual safeguards", c.checkContractualSafeguards(), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

// PCI-DSS Tests
func (c *ComplianceService) testPCIDSS1(assessment *ComplianceAssessment) {
	controlID := "PCI-1"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check firewall configuration
	score := 0
	verified := false

	s, v := applyCheck(result, "Firewall configuration", c.checkFirewallConfiguration(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Network segmentation", c.checkNetworkSegmentation(), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testPCIDSS2(assessment *ComplianceAssessment) {
	controlID := "PCI-2"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check default passwords and security parameters
	score := 0
	verified := false

	s, v := applyCheck(result, "No default passwords", c.checkDefaultPasswords(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Security parameters", c.checkSecurityParameters(), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testPCIDSS3(assessment *ComplianceAssessment) {
	controlID := "PCI-3"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check stored data protection
	score := 0
	verified := false

	s, v := applyCheck(result, "Stored data encryption", c.checkStoredDataEncryption(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Data retention", c.checkDataRetention(), 25)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Key management", c.checkKeyManagement(), 25)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 80:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

func (c *ComplianceService) testPCIDSS4(assessment *ComplianceAssessment) {
	controlID := "PCI-4"
	result := assessment.ControlResults[controlID]
	if result == nil {
		return
	}

	// Check transmission encryption
	score := 0
	verified := false

	s, v := applyCheck(result, "Transmission encryption", c.checkTransmissionEncryption(), 50)
	score += s
	verified = verified || v
	s, v = applyCheck(result, "Strong cryptography", c.checkStrongCryptography(), 50)
	score += s
	verified = verified || v

	switch {
	case !verified:
		result.Status = NotTested
	case score >= 100:
		result.Status = Compliant
	default:
		result.Status = NonCompliant
	}

	result.Score = float64(score)
	result.Tested = true
	result.TestedAt = time.Now()
}

// CheckResult is the outcome of a single automated compliance sub-check. It
// deliberately supports more than pass/fail: a check with no real signal to
// evaluate must say so (NotTested) rather than fabricate a result, and a
// check that structurally cannot apply to a software system (e.g. physical
// facility access) reports Exempt. See novacron-yvo — every checkXxx below
// used to unconditionally return true (or, for checkDefaultPasswords,
// false) regardless of actual system state.
type CheckResult struct {
	Status ComplianceStatus
	Reason string
}

func passCheck(reason string) CheckResult {
	return CheckResult{Status: Compliant, Reason: reason}
}

func failCheck(reason string) CheckResult {
	return CheckResult{Status: NonCompliant, Reason: reason}
}

// unknownCheck reports that no automated signal exists yet to evaluate this
// check. It must never be treated as compliant.
func unknownCheck(reason string) CheckResult {
	return CheckResult{Status: NotTested, Reason: reason}
}

// exemptCheck reports that the check does not apply to this system (e.g. a
// physical-safeguard control being evaluated for software that manages no
// physical facility).
func exemptCheck(reason string) CheckResult {
	return CheckResult{Status: Exempt, Reason: reason}
}

func (r CheckResult) compliant() bool {
	return r.Status == Compliant
}

// applyCheck scores a single sub-check's contribution to a control test. It
// awards points only for a genuinely Compliant result, records a
// human-readable finding for anything else, and reports whether the check
// produced a real signal at all (false for NotTested/Exempt) so the caller
// can tell "verified non-compliant" apart from "nothing could be verified".
// Unknown/Exempt results can only ever withhold points, never award them —
// that is what keeps this honest.
func applyCheck(result *ControlResult, label string, cr CheckResult, points int) (scored int, verified bool) {
	if cr.Status != Compliant {
		result.Findings = append(result.Findings, fmt.Sprintf("%s: %s", label, cr.Reason))
	}
	switch cr.Status {
	case Compliant:
		return points, true
	case NotTested, Exempt:
		return 0, false
	default:
		return 0, true
	}
}

// Helper functions for compliance checks.
//
// Each check below either reflects real, observable state from a sibling
// service in this package (wired via the SetXxx methods on
// ComplianceService), or — where this codebase has no automatable signal —
// honestly reports NotTested/Exempt with a reason instead of fabricating a
// result. See novacron-yvo.

func (c *ComplianceService) checkPolicyExists(policyType string) CheckResult {
	return unknownCheck(fmt.Sprintf("no policy document management system is wired into ComplianceService to verify the %q policy exists", policyType))
}

func (c *ComplianceService) checkTrainingRecords(trainingType string) CheckResult {
	return unknownCheck(fmt.Sprintf("no training-record system is wired into ComplianceService to verify %q training completion", trainingType))
}

func (c *ComplianceService) checkPolicyCommunication(policyType string) CheckResult {
	return unknownCheck(fmt.Sprintf("no policy-communication tracking system is wired into ComplianceService for the %q policy", policyType))
}

func (c *ComplianceService) checkMFAEnabled() CheckResult {
	c.mu.RLock()
	tfs := c.twoFactorService
	c.mu.RUnlock()
	if tfs == nil {
		return unknownCheck("no 2FA service is wired into ComplianceService")
	}
	stats := tfs.GetStats()
	total, _ := stats["total_users"].(int)
	enabled, _ := stats["enabled_users"].(int)
	if total == 0 {
		return unknownCheck("2FA service is wired but no users are enrolled yet")
	}
	if enabled == 0 {
		return failCheck(fmt.Sprintf("0 of %d enrolled users have MFA enabled", total))
	}
	return passCheck(fmt.Sprintf("%d of %d enrolled users have MFA enabled", enabled, total))
}

func (c *ComplianceService) checkPasswordPolicy() CheckResult {
	c.mu.RLock()
	ps := c.passwordService
	c.mu.RUnlock()
	if ps == nil {
		return unknownCheck("no password security service is wired into ComplianceService")
	}
	cfg := ps.config
	var missing []string
	if cfg.MinLength < 12 {
		missing = append(missing, fmt.Sprintf("MinLength=%d (<12)", cfg.MinLength))
	}
	if !cfg.RequireUppercase {
		missing = append(missing, "RequireUppercase=false")
	}
	if !cfg.RequireLowercase {
		missing = append(missing, "RequireLowercase=false")
	}
	if !cfg.RequireNumbers {
		missing = append(missing, "RequireNumbers=false")
	}
	if !cfg.RequireSpecialChars {
		missing = append(missing, "RequireSpecialChars=false")
	}
	if len(missing) > 0 {
		return failCheck("password policy below minimum bar: " + strings.Join(missing, ", "))
	}
	return passCheck("password policy enforces length >=12 and all character classes")
}

func (c *ComplianceService) checkAccessReviews() CheckResult {
	return unknownCheck("no periodic access-review record system is wired into ComplianceService")
}

// mtlsPolicyConfigured backs every check that ultimately asks the same real
// question from a different framework's angle: does an enabled zero-trust
// network policy require mutual TLS for sensitive traffic?
func (c *ComplianceService) mtlsPolicyConfigured() CheckResult {
	c.mu.RLock()
	zt := c.ztNetworkService
	c.mu.RUnlock()
	if zt == nil {
		return unknownCheck("no zero-trust network service is wired into ComplianceService")
	}
	for _, policy := range zt.GetPolicies() {
		if policy.Enabled && policy.Action == NetworkPolicyRequireMTLS {
			return passCheck(fmt.Sprintf("network policy %q enforces mTLS", policy.ID))
		}
	}
	return failCheck("no enabled network policy requires mTLS")
}

func (c *ComplianceService) checkTLSEnabled() CheckResult {
	return c.mtlsPolicyConfigured()
}

func (c *ComplianceService) checkEncryptionInTransit() CheckResult {
	return c.mtlsPolicyConfigured()
}

// auditLoggingConfigured backs every check that ultimately asks whether an
// audit trail is actually being recorded.
func (c *ComplianceService) auditLoggingConfigured() CheckResult {
	if c.auditService == nil {
		return failCheck("no audit service is configured")
	}
	return passCheck("an audit service is configured and receiving events")
}

func (c *ComplianceService) checkLoggingEnabled() CheckResult {
	return c.auditLoggingConfigured()
}

func (c *ComplianceService) checkMonitoring() CheckResult {
	return unknownCheck("no monitoring/metrics subsystem is wired into ComplianceService (see backend/core/monitoring)")
}

func (c *ComplianceService) checkAlertingEnabled() CheckResult {
	return unknownCheck("no alerting subsystem is wired into ComplianceService")
}

func (c *ComplianceService) checkPrivacyByDesign() CheckResult {
	return unknownCheck("privacy-by-design review is a process/design-review artifact with no automatable signal in this codebase")
}

func (c *ComplianceService) checkDataMinimization() CheckResult {
	return unknownCheck("no data classification/minimization tracking exists in this codebase")
}

// encryptionAtRestConfigured backs every check that ultimately asks whether
// data is actually being encrypted at rest by the wired EncryptionService.
func (c *ComplianceService) encryptionAtRestConfigured() CheckResult {
	if c.encryption == nil {
		return unknownCheck("no encryption service is wired into ComplianceService")
	}
	keys := c.encryption.GetActiveKeys()
	if len(keys) == 0 {
		return failCheck("no active encryption keys exist")
	}
	return passCheck(fmt.Sprintf("%d active encryption key(s)", len(keys)))
}

func (c *ComplianceService) checkEncryptionAtRest() CheckResult {
	return c.encryptionAtRestConfigured()
}

// rolesConfigured backs every check that ultimately asks whether real RBAC
// roles with actual permissions have been defined.
func (c *ComplianceService) rolesConfigured() CheckResult {
	c.mu.RLock()
	rs := c.roleService
	c.mu.RUnlock()
	if rs == nil {
		return unknownCheck("no role service is wired into ComplianceService")
	}
	roles, err := rs.List(nil)
	if err != nil {
		return unknownCheck(fmt.Sprintf("role service query failed: %v", err))
	}
	if len(roles) == 0 {
		return failCheck("no roles are defined")
	}
	for _, role := range roles {
		if len(role.Permissions) > 0 {
			return passCheck(fmt.Sprintf("%d role(s) defined, at least one with explicit permissions", len(roles)))
		}
	}
	return failCheck("roles are defined but none grant explicit permissions")
}

func (c *ComplianceService) checkAccessControls() CheckResult {
	return c.rolesConfigured()
}

func (c *ComplianceService) checkBackupSecurity() CheckResult {
	return unknownCheck("backup encryption/security lives in backend/core/backup and is not wired into ComplianceService")
}

func (c *ComplianceService) checkBreachProcedures() CheckResult {
	return unknownCheck("breach-response procedures are a runbook/process artifact with no automatable signal in this codebase")
}

func (c *ComplianceService) checkNotificationTiming() CheckResult {
	return unknownCheck("breach notification timing is a process record with no automatable signal in this codebase")
}

func (c *ComplianceService) checkDPIAProcess() CheckResult {
	return unknownCheck("DPIA process documentation has no automatable signal in this codebase")
}

func (c *ComplianceService) checkRiskAssessment() CheckResult {
	return unknownCheck("risk assessment procedures are a process/document artifact with no automatable signal in this codebase")
}

func (c *ComplianceService) checkSecurityOfficer() CheckResult {
	return unknownCheck("security officer designation is a personnel/org record with no automatable signal in this codebase")
}

func (c *ComplianceService) checkWorkforceTraining() CheckResult {
	return unknownCheck("no training-record system is wired into ComplianceService")
}

func (c *ComplianceService) checkAccessManagement() CheckResult {
	return c.rolesConfigured()
}

func (c *ComplianceService) checkFacilityAccess() CheckResult {
	return exemptCheck("physical facility access control is not observable by this software system's compliance automation")
}

func (c *ComplianceService) checkWorkstationSecurity() CheckResult {
	return exemptCheck("workstation/endpoint physical security is not observable by this software system's compliance automation")
}

func (c *ComplianceService) checkDeviceControls() CheckResult {
	return exemptCheck("physical device and media controls are not observable by this software system's compliance automation")
}

func (c *ComplianceService) checkTechnicalAccessControl() CheckResult {
	return c.rolesConfigured()
}

func (c *ComplianceService) checkAuditControls() CheckResult {
	return c.auditLoggingConfigured()
}

// checkDataIntegrity reuses the encryption-at-rest signal: the only two
// algorithms EncryptionService supports (AES-256-GCM, ChaCha20Poly1305) are
// both AEAD ciphers, so authenticity/tamper-detection is inherent to any
// data they encrypt — it is not a separate mechanism to check.
func (c *ComplianceService) checkDataIntegrity() CheckResult {
	result := c.encryptionAtRestConfigured()
	if result.Status == Compliant {
		return passCheck(result.Reason + "; both supported algorithms are AEAD ciphers with built-in integrity verification")
	}
	return result
}

func (c *ComplianceService) checkTransmissionSecurity() CheckResult {
	return c.mtlsPolicyConfigured()
}

func (c *ComplianceService) checkBusinessAssociates() CheckResult {
	return unknownCheck("business associate agreements are a legal/contract record with no automatable signal in this codebase")
}

func (c *ComplianceService) checkContractualSafeguards() CheckResult {
	return unknownCheck("contractual safeguards are a legal/contract record with no automatable signal in this codebase")
}

func (c *ComplianceService) checkFirewallConfiguration() CheckResult {
	c.mu.RLock()
	zt := c.ztNetworkService
	c.mu.RUnlock()
	if zt == nil {
		return unknownCheck("no zero-trust network service is wired into ComplianceService")
	}
	policies := zt.GetPolicies()
	if len(policies) == 0 {
		return failCheck("no network policies are configured")
	}
	for _, policy := range policies {
		if policy.Enabled && policy.Action == NetworkPolicyDeny {
			return passCheck(fmt.Sprintf("%d network policy(ies) configured, including a default-deny rule", len(policies)))
		}
	}
	return failCheck("network policies exist but none deny traffic by default")
}

func (c *ComplianceService) checkNetworkSegmentation() CheckResult {
	c.mu.RLock()
	zt := c.ztNetworkService
	c.mu.RUnlock()
	if zt == nil {
		return unknownCheck("no zero-trust network service is wired into ComplianceService")
	}
	segments := zt.GetMicrosegments()
	if len(segments) == 0 {
		return failCheck("no network microsegments are defined")
	}
	return passCheck(fmt.Sprintf("%d network microsegment(s) defined", len(segments)))
}

// checkDefaultPasswords reports Compliant when default/common passwords are
// actively guarded against. This intentionally flips the polarity of the
// original stub (which returned bare `false` to mean "compliant") so that,
// like every other check here, Compliant always means "passes".
func (c *ComplianceService) checkDefaultPasswords() CheckResult {
	c.mu.RLock()
	ps := c.passwordService
	c.mu.RUnlock()
	if ps == nil {
		return unknownCheck("no password security service is wired into ComplianceService")
	}
	if !ps.config.ForbidCommonPasswords {
		return failCheck("password policy does not forbid common/default passwords (ForbidCommonPasswords=false)")
	}
	if len(ps.commonPasswords) == 0 {
		return failCheck("ForbidCommonPasswords is enabled but the common-password block-list failed to load")
	}
	return passCheck(fmt.Sprintf("common/default password block-list is active (%d entries)", len(ps.commonPasswords)))
}

func (c *ComplianceService) checkSecurityParameters() CheckResult {
	strong := c.checkStrongCryptography()
	c.mu.RLock()
	ps := c.passwordService
	c.mu.RUnlock()
	if ps == nil {
		if strong.compliant() {
			return unknownCheck("encryption parameters are strong, but no password service is wired to verify hashing parameters")
		}
		return strong
	}
	algo := ps.config.HashAlgorithm
	okHash := algo == "argon2" || (algo == "bcrypt" && ps.config.BcryptCost >= 10)
	if !okHash {
		return failCheck(fmt.Sprintf("password hash algorithm %q (cost=%d) is below the secure baseline", algo, ps.config.BcryptCost))
	}
	if !strong.compliant() {
		return strong
	}
	return passCheck("password hashing and encryption parameters both meet the secure baseline")
}

func (c *ComplianceService) checkStoredDataEncryption() CheckResult {
	return c.encryptionAtRestConfigured()
}

func (c *ComplianceService) checkDataRetention() CheckResult {
	return unknownCheck("retention policy enforcement lives in backend/core/backup and is not wired into ComplianceService")
}

func (c *ComplianceService) checkKeyManagement() CheckResult {
	if c.encryption == nil {
		return unknownCheck("no encryption service is wired into ComplianceService")
	}
	if c.encryption.config.KeyRotationInterval <= 0 {
		return failCheck("key rotation interval is not configured")
	}
	keys := c.encryption.GetActiveKeys()
	if len(keys) == 0 {
		return failCheck("no active encryption keys exist to rotate")
	}
	return passCheck(fmt.Sprintf("%d active key(s), rotation interval %s", len(keys), c.encryption.config.KeyRotationInterval))
}

func (c *ComplianceService) checkTransmissionEncryption() CheckResult {
	return c.mtlsPolicyConfigured()
}

func (c *ComplianceService) checkStrongCryptography() CheckResult {
	if c.encryption == nil {
		return unknownCheck("no encryption service is wired into ComplianceService")
	}
	switch c.encryption.config.DefaultAlgorithm {
	case "AES-256-GCM", "ChaCha20Poly1305":
		return passCheck(fmt.Sprintf("default algorithm %s is a strong AEAD cipher", c.encryption.config.DefaultAlgorithm))
	default:
		return failCheck(fmt.Sprintf("default algorithm %q is not on the approved strong-cipher list", c.encryption.config.DefaultAlgorithm))
	}
}

// updateAssessmentStatus calculates overall assessment status
func (c *ComplianceService) updateAssessmentStatus(assessment *ComplianceAssessment) {
	totalScore := 0.0
	testedControls := 0
	compliantControls := 0

	for _, result := range assessment.ControlResults {
		if result.Tested {
			testedControls++
			totalScore += result.Score
			if result.Status == Compliant {
				compliantControls++
			}
		}
	}

	if testedControls > 0 {
		assessment.OverallScore = totalScore / float64(testedControls)

		if assessment.OverallScore >= 90 {
			assessment.Status = Compliant
		} else if assessment.OverallScore >= 70 {
			assessment.Status = Partial
		} else {
			assessment.Status = NonCompliant
		}
	} else {
		assessment.Status = NotTested
	}

	assessment.UpdatedAt = time.Now()
}

// getControlsForFramework returns controls for a specific framework
func (c *ComplianceService) getControlsForFramework(framework ComplianceFramework) []*ComplianceControl {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var controls []*ComplianceControl
	for _, control := range c.controls {
		if control.Framework == framework {
			controls = append(controls, control)
		}
	}

	return controls
}

// loadDefaultControls loads default compliance controls
func (c *ComplianceService) loadDefaultControls() {
	// SOC2 Controls
	c.addControl(&ComplianceControl{
		ID:          "CC1.1",
		Framework:   SOC2,
		Title:       "Integrity and Ethical Values",
		Description: "Management demonstrates commitment to integrity and ethical values",
		Category:    "Control Environment",
		Severity:    "high",
		Required:    true,
		Automated:   true,
		Testable:    true,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	})

	c.addControl(&ComplianceControl{
		ID:          "CC2.1",
		Framework:   SOC2,
		Title:       "Communication and Information",
		Description: "Information security policies are documented and communicated",
		Category:    "Communication and Information",
		Severity:    "high",
		Required:    true,
		Automated:   true,
		Testable:    true,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	})

	// GDPR Controls
	c.addControl(&ComplianceControl{
		ID:          "GDPR-25",
		Framework:   GDPR,
		Title:       "Data Protection by Design and by Default",
		Description: "Privacy by design and default implemented in processing",
		Category:    "Data Protection",
		Severity:    "critical",
		Required:    true,
		Automated:   true,
		Testable:    true,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	})

	// HIPAA Controls
	c.addControl(&ComplianceControl{
		ID:          "HIPAA-164.308",
		Framework:   HIPAA,
		Title:       "Administrative Safeguards",
		Description: "Administrative safeguards for PHI protection",
		Category:    "Administrative Safeguards",
		Severity:    "critical",
		Required:    true,
		Automated:   true,
		Testable:    true,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	})

	// PCI-DSS Controls
	c.addControl(&ComplianceControl{
		ID:          "PCI-1",
		Framework:   PCIDSS,
		Title:       "Install and Maintain Firewall Configuration",
		Description: "Install and maintain a firewall configuration to protect cardholder data",
		Category:    "Network Security",
		Severity:    "critical",
		Required:    true,
		Automated:   true,
		Testable:    true,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	})
}

// addControl adds a control to the service
func (c *ComplianceService) addControl(control *ComplianceControl) {
	c.controls[control.ID] = control
}

// generateID generates a unique ID
func (c *ComplianceService) generateID() string {
	return fmt.Sprintf("comp_%d", time.Now().UnixNano())
}

// GetAssessment returns an assessment by ID
func (c *ComplianceService) GetAssessment(id string) (*ComplianceAssessment, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	assessment, exists := c.assessments[id]
	if !exists {
		return nil, fmt.Errorf("assessment not found: %s", id)
	}

	return assessment, nil
}

// GenerateComplianceReport generates a compliance report
func (c *ComplianceService) GenerateComplianceReport(assessmentID string) ([]byte, error) {
	assessment, err := c.GetAssessment(assessmentID)
	if err != nil {
		return nil, err
	}

	report := map[string]interface{}{
		"assessment":      assessment,
		"summary":         c.generateSummary(assessment),
		"recommendations": assessment.Recommendations,
		"generated_at":    time.Now(),
	}

	return json.MarshalIndent(report, "", "  ")
}

// generateSummary generates an assessment summary
func (c *ComplianceService) generateSummary(assessment *ComplianceAssessment) map[string]interface{} {
	totalControls := len(assessment.ControlResults)
	testedControls := 0
	compliantControls := 0
	nonCompliantControls := 0
	partialControls := 0

	for _, result := range assessment.ControlResults {
		if result.Tested {
			testedControls++
			switch result.Status {
			case Compliant:
				compliantControls++
			case NonCompliant:
				nonCompliantControls++
			case Partial:
				partialControls++
			}
		}
	}

	return map[string]interface{}{
		"total_controls":         totalControls,
		"tested_controls":        testedControls,
		"compliant_controls":     compliantControls,
		"non_compliant_controls": nonCompliantControls,
		"partial_controls":       partialControls,
		"overall_score":          assessment.OverallScore,
		"status":                 assessment.Status,
		"total_findings":         len(assessment.Findings),
		"recommendations":        len(assessment.Recommendations),
	}
}
