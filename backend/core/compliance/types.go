// Package compliance provides enterprise compliance frameworks
package compliance

import (
	"time"
)

// ComplianceFramework defines supported compliance frameworks
type ComplianceFramework string

const (
	FrameworkSOC2    ComplianceFramework = "SOC2"
	FrameworkGDPR    ComplianceFramework = "GDPR"
	FrameworkHIPAA   ComplianceFramework = "HIPAA"
	FrameworkPCIDSS  ComplianceFramework = "PCI-DSS"
	FrameworkISO27001 ComplianceFramework = "ISO27001"
	FrameworkNIST    ComplianceFramework = "NIST"
)

// ComplianceStatus represents the compliance state
type ComplianceStatus string

const (
	StatusCompliant    ComplianceStatus = "compliant"
	StatusNonCompliant ComplianceStatus = "non_compliant"
	StatusPartial      ComplianceStatus = "partial"
	StatusUnknown      ComplianceStatus = "unknown"
	StatusInProgress   ComplianceStatus = "in_progress"
)

// ControlCategory defines control categories
type ControlCategory string

const (
	CategoryAccessControl    ControlCategory = "access_control"
	CategoryDataProtection   ControlCategory = "data_protection"
	CategoryAuditLogging     ControlCategory = "audit_logging"
	CategoryIncidentResponse ControlCategory = "incident_response"
	CategoryChangeManagement ControlCategory = "change_management"
	CategoryNetworkSecurity  ControlCategory = "network_security"
	CategoryEncryption       ControlCategory = "encryption"
	CategoryMonitoring       ControlCategory = "monitoring"
	CategoryBackupRecovery   ControlCategory = "backup_recovery"
	CategoryVendorManagement ControlCategory = "vendor_management"
)

// ComplianceControl represents a specific control requirement
type ComplianceControl struct {
	ID          string              `json:"id"`
	Framework   ComplianceFramework `json:"framework"`
	Category    ControlCategory     `json:"category"`
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Requirement string              `json:"requirement"`
	Status      ComplianceStatus    `json:"status"`
	Evidence    []Evidence          `json:"evidence"`
	Automated   bool                `json:"automated"`
	LastChecked time.Time           `json:"last_checked"`
	NextCheck   time.Time           `json:"next_check"`
	Owner       string              `json:"owner"`
	Remediation *Remediation        `json:"remediation,omitempty"`
	Metadata    map[string]string   `json:"metadata"`
}

// Evidence represents proof of compliance
type Evidence struct {
	ID          string            `json:"id"`
	Type        EvidenceType      `json:"type"`
	Description string            `json:"description"`
	CollectedAt time.Time         `json:"collected_at"`
	CollectedBy string            `json:"collected_by"`
	Location    string            `json:"location"` // File path, URL, etc.
	Hash        string            `json:"hash"`     // For tamper detection
	Metadata    map[string]string `json:"metadata"`
}

// EvidenceType defines types of compliance evidence
type EvidenceType string

const (
	EvidenceAuditLog      EvidenceType = "audit_log"
	EvidenceConfiguration EvidenceType = "configuration"
	EvidenceScreenshot    EvidenceType = "screenshot"
	EvidenceReport        EvidenceType = "report"
	EvidenceAttestation   EvidenceType = "attestation"
	EvidenceTest          EvidenceType = "test_result"
	EvidenceCertificate   EvidenceType = "certificate"
)

// Remediation represents steps to fix non-compliance
type Remediation struct {
	ID          string        `json:"id"`
	Status      string        `json:"status"`
	Priority    string        `json:"priority"`
	Description string        `json:"description"`
	Steps       []string      `json:"steps"`
	Owner       string        `json:"owner"`
	DueDate     time.Time     `json:"due_date"`
	CreatedAt   time.Time     `json:"created_at"`
	CompletedAt *time.Time    `json:"completed_at,omitempty"`
	Evidence    []Evidence    `json:"evidence"`
}

// ComplianceReport represents a compliance assessment report
type ComplianceReport struct {
	ID          string              `json:"id"`
	Framework   ComplianceFramework `json:"framework"`
	Status      ComplianceStatus    `json:"status"`
	Score       float64             `json:"score"` // 0-100
	GeneratedAt time.Time           `json:"generated_at"`
	GeneratedBy string              `json:"generated_by"`
	Period      Period              `json:"period"`
	Controls    []ControlResult     `json:"controls"`
	Summary     ReportSummary       `json:"summary"`
	Findings    []Finding           `json:"findings"`
	Evidence    []Evidence          `json:"evidence"`
}

// Period represents a reporting period
type Period struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// ControlResult represents the result of a control check
type ControlResult struct {
	ControlID   string           `json:"control_id"`
	Status      ComplianceStatus `json:"status"`
	Score       float64          `json:"score"`
	CheckedAt   time.Time        `json:"checked_at"`
	Details     string           `json:"details"`
	Evidence    []Evidence       `json:"evidence"`
	Remediation *Remediation     `json:"remediation,omitempty"`
}

// ReportSummary provides high-level compliance metrics
type ReportSummary struct {
	TotalControls     int              `json:"total_controls"`
	CompliantControls int              `json:"compliant_controls"`
	FailedControls    int              `json:"failed_controls"`
	PartialControls   int              `json:"partial_controls"`
	AutomatedControls int              `json:"automated_controls"`
	CriticalFindings  int              `json:"critical_findings"`
	HighFindings      int              `json:"high_findings"`
	MediumFindings    int              `json:"medium_findings"`
	LowFindings       int              `json:"low_findings"`
	ComplianceScore   float64          `json:"compliance_score"`
	TrendDirection    string           `json:"trend_direction"` // improving, declining, stable
}

// Finding represents a compliance issue or observation
type Finding struct {
	ID          string    `json:"id"`
	Severity    string    `json:"severity"` // critical, high, medium, low
	Title       string    `json:"title"`
	Description string    `json:"description"`
	ControlID   string    `json:"control_id"`
	Impact      string    `json:"impact"`
	Risk        string    `json:"risk"`
	Evidence    []Evidence `json:"evidence"`
	Remediation *Remediation `json:"remediation,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
}

// DataPrivacyRequest represents GDPR/privacy requests
type DataPrivacyRequest struct {
	ID          string            `json:"id"`
	Type        PrivacyRequestType `json:"type"`
	SubjectID   string            `json:"subject_id"`
	Status      string            `json:"status"`
	RequestedAt time.Time         `json:"requested_at"`
	CompletedAt *time.Time        `json:"completed_at,omitempty"`
	RequestedBy string            `json:"requested_by"`
	ProcessedBy string            `json:"processed_by,omitempty"`
	Data        map[string]interface{} `json:"data,omitempty"`
	Metadata    map[string]string `json:"metadata"`
}

// PrivacyRequestType defines types of privacy requests
type PrivacyRequestType string

const (
	RequestAccessData     PrivacyRequestType = "access_data"      // GDPR Art. 15
	RequestRectification  PrivacyRequestType = "rectification"    // GDPR Art. 16
	RequestErasure        PrivacyRequestType = "erasure"          // GDPR Art. 17 (Right to be forgotten)
	RequestRestriction    PrivacyRequestType = "restriction"      // GDPR Art. 18
	RequestPortability    PrivacyRequestType = "data_portability" // GDPR Art. 20
	RequestObjection      PrivacyRequestType = "objection"        // GDPR Art. 21
	RequestWithdrawConsent PrivacyRequestType = "withdraw_consent" // GDPR Art. 7
)

// ConsentRecord represents user consent tracking
type ConsentRecord struct {
	ID             string            `json:"id"`
	SubjectID      string            `json:"subject_id"`
	Purpose        string            `json:"purpose"`
	ConsentGiven   bool              `json:"consent_given"`
	ConsentDate    time.Time         `json:"consent_date"`
	WithdrawnDate  *time.Time        `json:"withdrawn_date,omitempty"`
	ExpiryDate     *time.Time        `json:"expiry_date,omitempty"`
	LegalBasis     string            `json:"legal_basis"` // GDPR Art. 6
	DataCategories []string          `json:"data_categories"`
	Metadata       map[string]string `json:"metadata"`
}

// PHIAccessLog represents HIPAA PHI access tracking
type PHIAccessLog struct {
	ID          string            `json:"id"`
	UserID      string            `json:"user_id"`
	UserRole    string            `json:"user_role"`
	PatientID   string            `json:"patient_id"`
	Action      string            `json:"action"`
	Resource    string            `json:"resource"`
	Timestamp   time.Time         `json:"timestamp"`
	IPAddress   string            `json:"ip_address"`
	Purpose     string            `json:"purpose"`      // Treatment, payment, operations
	Authorized  bool              `json:"authorized"`
	BreakGlass  bool              `json:"break_glass"`  // Emergency access
	Metadata    map[string]string `json:"metadata"`
}

// BAA represents Business Associate Agreement
type BAA struct {
	ID              string     `json:"id"`
	EntityName      string     `json:"entity_name"`
	EntityType      string     `json:"entity_type"` // business_associate, covered_entity
	Status          string     `json:"status"`
	SignedDate      time.Time  `json:"signed_date"`
	ExpiryDate      time.Time  `json:"expiry_date"`
	ReviewDate      time.Time  `json:"review_date"`
	ContactName     string     `json:"contact_name"`
	ContactEmail    string     `json:"contact_email"`
	Services        []string   `json:"services"`
	PHICategories   []string   `json:"phi_categories"`
	SecurityControls []string  `json:"security_controls"`
	Metadata        map[string]string `json:"metadata"`
}

// SecurityPosture represents overall security posture
type SecurityPosture struct {
	ID              string            `json:"id"`
	Timestamp       time.Time         `json:"timestamp"`
	OverallScore    float64           `json:"overall_score"` // 0-100
	RiskLevel       string            `json:"risk_level"`    // low, medium, high, critical
	Vulnerabilities VulnerabilitySummary `json:"vulnerabilities"`
	Compliance      map[ComplianceFramework]float64 `json:"compliance"`
	Controls        map[ControlCategory]float64     `json:"controls"`
	Trends          PostureTrends     `json:"trends"`
	Recommendations []Recommendation  `json:"recommendations"`
}

// VulnerabilitySummary summarizes vulnerability scan results
type VulnerabilitySummary struct {
	Critical int       `json:"critical"`
	High     int       `json:"high"`
	Medium   int       `json:"medium"`
	Low      int       `json:"low"`
	Total    int       `json:"total"`
	LastScan time.Time `json:"last_scan"`
}

// PostureTrends shows security posture trends
type PostureTrends struct {
	ScoreChange7d   float64 `json:"score_change_7d"`
	ScoreChange30d  float64 `json:"score_change_30d"`
	VulnChange7d    int     `json:"vuln_change_7d"`
	VulnChange30d   int     `json:"vuln_change_30d"`
	Direction       string  `json:"direction"` // improving, declining, stable
}

// Recommendation represents a security recommendation
type Recommendation struct {
	ID          string    `json:"id"`
	Priority    string    `json:"priority"`
	Category    string    `json:"category"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Impact      string    `json:"impact"`
	Effort      string    `json:"effort"`
	References  []string  `json:"references"`
	CreatedAt   time.Time `json:"created_at"`
}

// AuditEvent represents a tamper-proof audit event
type AuditEvent struct {
	ID            string            `json:"id"`
	Timestamp     time.Time         `json:"timestamp"`
	EventType     string            `json:"event_type"`
	Actor         Actor             `json:"actor"`
	Action        string            `json:"action"`
	Resource      Resource          `json:"resource"`
	Result        string            `json:"result"`
	Severity      string            `json:"severity"`
	IPAddress     string            `json:"ip_address"`
	UserAgent     string            `json:"user_agent"`
	RequestID     string            `json:"request_id"`
	SessionID     string            `json:"session_id"`
	Details       map[string]interface{} `json:"details"`
	Hash          string            `json:"hash"`           // Event hash
	PreviousHash  string            `json:"previous_hash"`  // Previous event hash (blockchain)
	BlockNumber   int64             `json:"block_number"`
	Metadata      map[string]string `json:"metadata"`
}

// Actor represents the entity performing an action
type Actor struct {
	ID       string `json:"id"`
	Type     string `json:"type"`     // user, service, system
	Name     string `json:"name"`
	Email    string `json:"email,omitempty"`
	Roles    []string `json:"roles"`
	TenantID string `json:"tenant_id,omitempty"`
}

// Resource represents the resource being acted upon
type Resource struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Name     string `json:"name"`
	Tags     map[string]string `json:"tags,omitempty"`
	TenantID string `json:"tenant_id,omitempty"`
}

// ComplianceEngine defines the compliance automation interface
type ComplianceEngine interface {
	// Control Management
	RegisterControl(control ComplianceControl) error
	GetControl(id string) (*ComplianceControl, error)
	ListControls(framework ComplianceFramework) ([]ComplianceControl, error)
	UpdateControlStatus(id string, status ComplianceStatus, evidence []Evidence) error

	// Compliance Assessment
	AssessCompliance(framework ComplianceFramework) (*ComplianceReport, error)
	CheckControl(controlID string) (*ControlResult, error)
	RunAutomatedChecks(framework ComplianceFramework) error

	// Evidence Management
	CollectEvidence(controlID string, evidence Evidence) error
	GetEvidence(controlID string) ([]Evidence, error)
	VerifyEvidence(evidenceID string) (bool, error)

	// Reporting
	GenerateReport(framework ComplianceFramework, period Period) (*ComplianceReport, error)
	ExportReport(reportID string, format string) ([]byte, error)
	GetComplianceScore(framework ComplianceFramework) (float64, error)

	// Remediation
	CreateRemediation(finding Finding) (*Remediation, error)
	UpdateRemediation(id string, remediation Remediation) error
	GetRemediations(status string) ([]Remediation, error)

	// Continuous Monitoring
	EnableContinuousMonitoring(framework ComplianceFramework) error
	GetComplianceStatus() (map[ComplianceFramework]ComplianceStatus, error)
}

// PolicyEngine defines the policy enforcement interface
type PolicyEngine interface {
	// Policy Management
	CreatePolicy(policy Policy) error
	UpdatePolicy(id string, policy Policy) error
	DeletePolicy(id string) error
	GetPolicy(id string) (*Policy, error)
	ListPolicies() ([]Policy, error)

	// Policy Evaluation
	Evaluate(request PolicyRequest) (*PolicyDecision, error)
	EvaluateBatch(requests []PolicyRequest) ([]PolicyDecision, error)

	// Policy Testing
	TestPolicy(policy Policy, testCases []PolicyTestCase) ([]PolicyTestResult, error)
	ValidatePolicy(policy Policy) error

	// Policy Enforcement
	EnablePolicy(id string) error
	DisablePolicy(id string) error
	GetPolicyViolations(policyID string, since time.Time) ([]PolicyViolation, error)
}

// Policy represents a policy definition
type Policy struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Type        PolicyType        `json:"type"`
	Scope       PolicyScope       `json:"scope"`
	Enabled     bool              `json:"enabled"`
	Priority    int               `json:"priority"`
	Rules       []PolicyRule      `json:"rules"`
	Actions     []PolicyAction    `json:"actions"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	Version     int               `json:"version"`
	Metadata    map[string]string `json:"metadata"`
}

// PolicyType defines policy types
type PolicyType string

const (
	PolicyTypeAccess      PolicyType = "access"
	PolicyTypeData        PolicyType = "data"
	PolicyTypeNetwork     PolicyType = "network"
	PolicyTypeCompliance  PolicyType = "compliance"
	PolicyTypeGovernance  PolicyType = "governance"
)

// PolicyScope defines where policy applies
type PolicyScope struct {
	Resources []string `json:"resources"`
	Actions   []string `json:"actions"`
	Principals []string `json:"principals"`
	Conditions map[string]interface{} `json:"conditions,omitempty"`
}

// PolicyRule defines a policy rule
type PolicyRule struct {
	ID          string                 `json:"id"`
	Effect      PolicyEffect           `json:"effect"` // allow, deny
	Conditions  map[string]interface{} `json:"conditions"`
	Description string                 `json:"description"`
}

// PolicyEffect defines the effect of a rule
type PolicyEffect string

const (
	PolicyEffectAllow PolicyEffect = "allow"
	PolicyEffectDeny  PolicyEffect = "deny"
)

// PolicyAction defines an action to take
type PolicyAction struct {
	Type   string                 `json:"type"`
	Config map[string]interface{} `json:"config"`
}

// PolicyRequest represents a policy evaluation request
type PolicyRequest struct {
	Principal string                 `json:"principal"`
	Action    string                 `json:"action"`
	Resource  string                 `json:"resource"`
	Context   map[string]interface{} `json:"context"`
}

// PolicyDecision represents a policy evaluation result
type PolicyDecision struct {
	Allowed     bool              `json:"allowed"`
	Decision    PolicyEffect      `json:"decision"`
	Reasons     []string          `json:"reasons"`
	MatchedPolicies []string      `json:"matched_policies"`
	EvaluatedAt time.Time         `json:"evaluated_at"`
	Metadata    map[string]string `json:"metadata"`
}

// PolicyTestCase represents a policy test case
type PolicyTestCase struct {
	Name        string        `json:"name"`
	Request     PolicyRequest `json:"request"`
	Expected    PolicyEffect  `json:"expected"`
	Description string        `json:"description"`
}

// PolicyTestResult represents a policy test result
type PolicyTestResult struct {
	TestCase PolicyTestCase `json:"test_case"`
	Passed   bool           `json:"passed"`
	Actual   PolicyEffect   `json:"actual"`
	Message  string         `json:"message"`
}

// PolicyViolation represents a policy violation
type PolicyViolation struct {
	ID        string        `json:"id"`
	PolicyID  string        `json:"policy_id"`
	Request   PolicyRequest `json:"request"`
	Timestamp time.Time     `json:"timestamp"`
	Severity  string        `json:"severity"`
	Details   string        `json:"details"`
}

// GovernanceEngine defines governance automation interface
type GovernanceEngine interface {
	// Resource Tagging
	EnforceTagging(resourceType string, requiredTags []string) error
	ValidateTags(resourceID string) error
	GetUntaggedResources() ([]string, error)

	// Cost Management
	SetBudget(scope string, amount float64, period Period) error
	GetCostAllocation(scope string) (map[string]float64, error)
	GetBudgetAlerts() ([]BudgetAlert, error)

	// Access Reviews
	ScheduleAccessReview(scope string, frequency time.Duration) error
	GetPendingAccessReviews() ([]AccessReview, error)
	CompleteAccessReview(id string, decisions []AccessDecision) error

	// Compliance Automation
	AutoRemediateViolations(framework ComplianceFramework) error
	GetRemediationStatus() ([]RemediationStatus, error)
}

// BudgetAlert represents a budget alert
type BudgetAlert struct {
	ID         string    `json:"id"`
	BudgetID   string    `json:"budget_id"`
	Scope      string    `json:"scope"`
	Current    float64   `json:"current"`
	Budget     float64   `json:"budget"`
	Percentage float64   `json:"percentage"`
	Severity   string    `json:"severity"`
	CreatedAt  time.Time `json:"created_at"`
}

// AccessReview represents an access review
type AccessReview struct {
	ID          string          `json:"id"`
	Scope       string          `json:"scope"`
	Reviewer    string          `json:"reviewer"`
	Status      string          `json:"status"`
	DueDate     time.Time       `json:"due_date"`
	Accesses    []AccessItem    `json:"accesses"`
	CreatedAt   time.Time       `json:"created_at"`
	CompletedAt *time.Time      `json:"completed_at,omitempty"`
}

// AccessItem represents an access to review
type AccessItem struct {
	ID        string            `json:"id"`
	Principal string            `json:"principal"`
	Resource  string            `json:"resource"`
	Role      string            `json:"role"`
	GrantedAt time.Time         `json:"granted_at"`
	LastUsed  *time.Time        `json:"last_used,omitempty"`
	Metadata  map[string]string `json:"metadata"`
}

// AccessDecision represents an access review decision
type AccessDecision struct {
	AccessID string `json:"access_id"`
	Decision string `json:"decision"` // approve, revoke, modify
	Reason   string `json:"reason"`
}

// RemediationStatus represents remediation status
type RemediationStatus struct {
	ID          string    `json:"id"`
	Finding     string    `json:"finding"`
	Status      string    `json:"status"`
	StartedAt   time.Time `json:"started_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
	Success     bool      `json:"success"`
	Message     string    `json:"message"`
}
