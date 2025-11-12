package security

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
)

// ZeroTrustManager implements zero-trust security architecture
type ZeroTrustManager struct {
	mu                    sync.RWMutex
	identityStore         *IdentityStore
	policyEngine          *PolicyEngine
	authEngine            *ContinuousAuthEngine
	microsegmentation     *MicrosegmentationEngine
	accessProvisioner     *JITAccessProvisioner
	trustScorer           *TrustScorer
	sessionManager        *SessionManager
	config                *ZeroTrustConfig
	auditLogger           AuditLogger
	metrics               *ZeroTrustMetrics
}

// ZeroTrustConfig configuration for zero-trust architecture
type ZeroTrustConfig struct {
	// Identity-based access control
	EnableIBAC                bool
	RequireDeviceAttestation  bool
	RequireGeoVerification    bool
	RequireBehaviorAnalysis   bool

	// Continuous authentication
	ReauthInterval            time.Duration
	SessionTimeout            time.Duration
	MaxConcurrentSessions     int
	EnableAdaptiveAuth        bool

	// Micro-segmentation
	EnableMicrosegmentation   bool
	DefaultDenyAll            bool
	NetworkPolicyMode         NetworkPolicyMode

	// Just-in-time access
	EnableJITAccess           bool
	MaxAccessDuration         time.Duration
	RequireApproval           bool

	// Trust scoring
	MinTrustScore             float64
	TrustDecayRate            float64
	RiskThresholds            map[RiskLevel]float64

	// Advanced features
	EnableZeroTrustAnalytics  bool
	EnableThreatIntel         bool
	EnableMLVerification      bool
}

// NetworkPolicyMode defines network policy enforcement mode
type NetworkPolicyMode string

const (
	NetworkPolicyEnforce NetworkPolicyMode = "enforce"
	NetworkPolicyAudit   NetworkPolicyMode = "audit"
	NetworkPolicyDisable NetworkPolicyMode = "disable"
)

// RiskLevel defines risk levels for trust scoring
type RiskLevel string

const (
	RiskLevelCritical RiskLevel = "critical"
	RiskLevelHigh     RiskLevel = "high"
	RiskLevelMedium   RiskLevel = "medium"
	RiskLevelLow      RiskLevel = "low"
	RiskLevelNone     RiskLevel = "none"
)

// IdentityStore manages identities and their attributes
type IdentityStore struct {
	mu         sync.RWMutex
	identities map[string]*Identity
	devices    map[string]*Device
	locations  map[string]*Location
}

// Identity represents a verified identity
type Identity struct {
	ID                string
	Type              IdentityType
	Principal         string
	Attributes        map[string]interface{}
	Verified          bool
	TrustScore        float64
	LastVerified      time.Time
	VerificationCount int
	DeviceIDs         []string
	LocationHistory   []string
	BehaviorProfile   *BehaviorProfile
	RiskFactors       []RiskFactor
	CreatedAt         time.Time
	UpdatedAt         time.Time
}

// IdentityType defines types of identities
type IdentityType string

const (
	IdentityTypeUser      IdentityType = "user"
	IdentityTypeService   IdentityType = "service"
	IdentityTypeDevice    IdentityType = "device"
	IdentityTypeWorkload  IdentityType = "workload"
)

// Device represents a device identity
type Device struct {
	ID               string
	Type             DeviceType
	Fingerprint      string
	Attested         bool
	AttestationData  *AttestationData
	TrustScore       float64
	Compliant        bool
	ComplianceChecks []ComplianceCheck
	LastSeen         time.Time
	CreatedAt        time.Time
}

// DeviceType defines device types
type DeviceType string

const (
	DeviceTypeMobile     DeviceType = "mobile"
	DeviceTypeDesktop    DeviceType = "desktop"
	DeviceTypeServer     DeviceType = "server"
	DeviceTypeIoT        DeviceType = "iot"
	DeviceTypeContainer  DeviceType = "container"
)

// AttestationData contains device attestation information
type AttestationData struct {
	Platform        string
	HardwareID      string
	SecureBootState bool
	TPMEnabled      bool
	EncryptionState bool
	PatchLevel      string
	Timestamp       time.Time
	Signature       []byte
}

// ComplianceCheck represents a compliance verification
type ComplianceCheck struct {
	CheckType string
	Passed    bool
	Details   string
	Timestamp time.Time
}

// Location represents a geographic location
type Location struct {
	ID        string
	Country   string
	Region    string
	City      string
	IPRange   *net.IPNet
	Trusted   bool
	RiskScore float64
	Timestamp time.Time
}

// BehaviorProfile represents user behavior patterns
type BehaviorProfile struct {
	TypicalLocations    []string
	TypicalDevices      []string
	TypicalAccessTimes  []TimeWindow
	TypicalResources    []string
	AnomalyScore        float64
	LastUpdated         time.Time
}

// TimeWindow represents a time window
type TimeWindow struct {
	StartHour int
	EndHour   int
	DaysOfWeek []time.Weekday
}

// RiskFactor represents a risk factor
type RiskFactor struct {
	Type        string
	Severity    RiskLevel
	Description string
	DetectedAt  time.Time
}

// PolicyEngine evaluates access policies
type PolicyEngine struct {
	mu       sync.RWMutex
	policies map[string]*Policy
	rules    map[string]*PolicyRule
}

// Policy represents an access policy
type Policy struct {
	ID          string
	Name        string
	Description string
	Priority    int
	Enabled     bool
	Rules       []string
	Conditions  []Condition
	Actions     []Action
	CreatedAt   time.Time
	UpdatedAt   time.Time
}

// PolicyRule represents a policy rule
type PolicyRule struct {
	ID          string
	Type        RuleType
	Conditions  []Condition
	Effect      Effect
	Priority    int
	Description string
}

// RuleType defines policy rule types
type RuleType string

const (
	RuleTypeIdentity   RuleType = "identity"
	RuleTypeDevice     RuleType = "device"
	RuleTypeLocation   RuleType = "location"
	RuleTypeBehavior   RuleType = "behavior"
	RuleTypeRisk       RuleType = "risk"
	RuleTypeTime       RuleType = "time"
	RuleTypeContext    RuleType = "context"
)

// Condition represents a policy condition
type Condition struct {
	Type     ConditionType
	Operator Operator
	Value    interface{}
	Metadata map[string]interface{}
}

// ConditionType defines condition types
type ConditionType string

const (
	ConditionIdentityVerified   ConditionType = "identity_verified"
	ConditionDeviceAttested     ConditionType = "device_attested"
	ConditionLocationTrusted    ConditionType = "location_trusted"
	ConditionTrustScoreMinimum  ConditionType = "trust_score_minimum"
	ConditionRiskLevelMaximum   ConditionType = "risk_level_maximum"
	ConditionTimeWindow         ConditionType = "time_window"
	ConditionBehaviorNormal     ConditionType = "behavior_normal"
	ConditionMFAVerified        ConditionType = "mfa_verified"
)

// Operator defines condition operators
type Operator string

const (
	OperatorEquals             Operator = "equals"
	OperatorNotEquals          Operator = "not_equals"
	OperatorGreaterThan        Operator = "greater_than"
	OperatorLessThan           Operator = "less_than"
	OperatorContains           Operator = "contains"
	OperatorIn                 Operator = "in"
	OperatorMatches            Operator = "matches"
)

// Effect defines policy effects
type Effect string

const (
	EffectAllow Effect = "allow"
	EffectDeny  Effect = "deny"
	EffectAudit Effect = "audit"
)

// Action represents a policy action
type Action struct {
	Type       ActionType
	Parameters map[string]interface{}
}

// ActionType defines action types
type ActionType string

const (
	ActionGrant          ActionType = "grant"
	ActionDeny           ActionType = "deny"
	ActionRequireMFA     ActionType = "require_mfa"
	ActionRequireReauth  ActionType = "require_reauth"
	ActionStepUp         ActionType = "step_up"
	ActionAlert          ActionType = "alert"
	ActionLog            ActionType = "log"
)

// ContinuousAuthEngine handles continuous authentication
type ContinuousAuthEngine struct {
	mu              sync.RWMutex
	sessions        map[string]*AuthSession
	verifiers       []AuthVerifier
	reauthScheduler *ReauthScheduler
}

// AuthSession represents an authentication session
type AuthSession struct {
	ID              string
	IdentityID      string
	DeviceID        string
	StartTime       time.Time
	LastActivity    time.Time
	LastReauth      time.Time
	TrustScore      float64
	RiskLevel       RiskLevel
	VerificationLog []VerificationEvent
	Active          bool
	Metadata        map[string]interface{}
}

// VerificationEvent represents a verification event
type VerificationEvent struct {
	Type      VerificationType
	Timestamp time.Time
	Success   bool
	Details   map[string]interface{}
}

// VerificationType defines verification types
type VerificationType string

const (
	VerificationPassword  VerificationType = "password"
	VerificationMFA       VerificationType = "mfa"
	VerificationBiometric VerificationType = "biometric"
	VerificationDevice    VerificationType = "device"
	VerificationBehavior  VerificationType = "behavior"
	VerificationLocation  VerificationType = "location"
)

// AuthVerifier interface for authentication verifiers
type AuthVerifier interface {
	Verify(ctx context.Context, session *AuthSession, evidence interface{}) (bool, error)
	GetType() VerificationType
}

// ReauthScheduler schedules reauthentication
type ReauthScheduler struct {
	mu        sync.RWMutex
	schedules map[string]*ReauthSchedule
}

// ReauthSchedule represents a reauthentication schedule
type ReauthSchedule struct {
	SessionID  string
	NextReauth time.Time
	Interval   time.Duration
	Required   bool
}

// MicrosegmentationEngine implements network micro-segmentation
type MicrosegmentationEngine struct {
	mu             sync.RWMutex
	segments       map[string]*NetworkSegment
	policies       map[string]*SegmentPolicy
	topology       *NetworkTopology
	enforcer       *PolicyEnforcer
}

// NetworkSegment represents a network segment
type NetworkSegment struct {
	ID          string
	Name        string
	CIDR        *net.IPNet
	Type        SegmentType
	SecurityZone string
	Workloads   []string
	Policies    []string
	Isolated    bool
	CreatedAt   time.Time
}

// SegmentType defines segment types
type SegmentType string

const (
	SegmentTypeProduction  SegmentType = "production"
	SegmentTypeStaging     SegmentType = "staging"
	SegmentTypeDevelopment SegmentType = "development"
	SegmentTypeManagement  SegmentType = "management"
	SegmentTypeDMZ         SegmentType = "dmz"
)

// SegmentPolicy represents a segmentation policy
type SegmentPolicy struct {
	ID             string
	Name           string
	SourceSegment  string
	DestSegment    string
	Protocol       string
	Ports          []int
	Action         PolicyAction
	Priority       int
	Conditions     []Condition
	Bidirectional  bool
	Logging        bool
	CreatedAt      time.Time
}

// PolicyAction defines policy actions
type PolicyAction string

const (
	PolicyActionAllow  PolicyAction = "allow"
	PolicyActionDeny   PolicyAction = "deny"
	PolicyActionAlert  PolicyAction = "alert"
	PolicyActionLog    PolicyAction = "log"
)

// NetworkTopology represents network topology
type NetworkTopology struct {
	Segments     map[string]*NetworkSegment
	Connections  []SegmentConnection
	SecurityZones map[string]*SecurityZone
}

// SegmentConnection represents connection between segments
type SegmentConnection struct {
	FromSegment string
	ToSegment   string
	Allowed     bool
	PolicyID    string
}

// SecurityZone represents a security zone
type SecurityZone struct {
	ID          string
	Name        string
	TrustLevel  int
	Segments    []string
	Policies    []string
}

// PolicyEnforcer enforces network policies
type PolicyEnforcer struct {
	mu       sync.RWMutex
	rules    map[string]*EnforcementRule
	mode     NetworkPolicyMode
	violations map[string]*PolicyViolation
}

// EnforcementRule represents an enforcement rule
type EnforcementRule struct {
	ID         string
	PolicyID   string
	Active     bool
	Enforced   bool
	LastUpdate time.Time
}

// PolicyViolation represents a policy violation
type PolicyViolation struct {
	ID           string
	RuleID       string
	Source       string
	Destination  string
	Protocol     string
	Port         int
	Action       string
	Timestamp    time.Time
	Severity     RiskLevel
}

// JITAccessProvisioner provides just-in-time access
type JITAccessProvisioner struct {
	mu           sync.RWMutex
	grants       map[string]*AccessGrant
	approvals    map[string]*AccessApproval
	provisioner  *ResourceProvisioner
}

// AccessGrant represents an access grant
type AccessGrant struct {
	ID          string
	IdentityID  string
	Resource    string
	Permissions []string
	StartTime   time.Time
	EndTime     time.Time
	Approved    bool
	ApprovalID  string
	Provisioned bool
	Revoked     bool
	Metadata    map[string]interface{}
}

// AccessApproval represents an approval workflow
type AccessApproval struct {
	ID          string
	GrantID     string
	Requester   string
	Approver    string
	Status      ApprovalStatus
	Reason      string
	CreatedAt   time.Time
	ProcessedAt time.Time
}

// ApprovalStatus defines approval status
type ApprovalStatus string

const (
	ApprovalPending  ApprovalStatus = "pending"
	ApprovalApproved ApprovalStatus = "approved"
	ApprovalDenied   ApprovalStatus = "denied"
	ApprovalExpired  ApprovalStatus = "expired"
)

// ResourceProvisioner provisions resources
type ResourceProvisioner struct {
	mu        sync.RWMutex
	resources map[string]*ProvisionedResource
}

// ProvisionedResource represents a provisioned resource
type ProvisionedResource struct {
	ID          string
	GrantID     string
	Resource    string
	Credentials interface{}
	CreatedAt   time.Time
	ExpiresAt   time.Time
}

// TrustScorer calculates trust scores
type TrustScorer struct {
	mu       sync.RWMutex
	scores   map[string]*TrustScore
	factors  []TrustFactor
	weights  map[string]float64
	decay    float64
}

// TrustScore represents a trust score
type TrustScore struct {
	IdentityID    string
	Score         float64
	Components    map[string]float64
	LastUpdated   time.Time
	Trend         TrendDirection
	History       []ScoreSnapshot
}

// TrendDirection defines score trend
type TrendDirection string

const (
	TrendIncreasing TrendDirection = "increasing"
	TrendStable     TrendDirection = "stable"
	TrendDecreasing TrendDirection = "decreasing"
)

// ScoreSnapshot represents a score snapshot
type ScoreSnapshot struct {
	Score     float64
	Timestamp time.Time
	Event     string
}

// TrustFactor represents a trust factor
type TrustFactor struct {
	Name        string
	Weight      float64
	Calculator  func(identity *Identity) float64
	Description string
}

// SessionManager manages sessions
type SessionManager struct {
	mu       sync.RWMutex
	sessions map[string]*Session
	limits   map[string]int
}

// Session represents a session
type Session struct {
	ID         string
	IdentityID string
	DeviceID   string
	StartTime  time.Time
	LastAccess time.Time
	ExpiresAt  time.Time
	Active     bool
	Context    *SessionContext
}

// SessionContext contains session context
type SessionContext struct {
	IPAddress   string
	Location    string
	UserAgent   string
	TrustScore  float64
	RiskLevel   RiskLevel
	Metadata    map[string]interface{}
}

// ZeroTrustMetrics contains metrics
type ZeroTrustMetrics struct {
	mu                    sync.RWMutex
	TotalIdentities       int64
	ActiveSessions        int64
	TrustScoreAverage     float64
	PolicyViolations      int64
	BlockedAccess         int64
	JITAccessGrants       int64
	ReauthenticationCount int64
	AnomaliesDetected     int64
	LastUpdated           time.Time
}

// NewZeroTrustManager creates a new zero-trust manager
func NewZeroTrustManager(config *ZeroTrustConfig, auditLogger AuditLogger) *ZeroTrustManager {
	ztm := &ZeroTrustManager{
		identityStore:     NewIdentityStore(),
		policyEngine:      NewPolicyEngine(),
		authEngine:        NewContinuousAuthEngine(),
		microsegmentation: NewMicrosegmentationEngine(),
		accessProvisioner: NewJITAccessProvisioner(),
		trustScorer:       NewTrustScorer(config.TrustDecayRate),
		sessionManager:    NewSessionManager(),
		config:            config,
		auditLogger:       auditLogger,
		metrics:           &ZeroTrustMetrics{},
	}

	ztm.initializeDefaultPolicies()
	ztm.startBackgroundTasks()

	return ztm
}

// NewIdentityStore creates identity store
func NewIdentityStore() *IdentityStore {
	return &IdentityStore{
		identities: make(map[string]*Identity),
		devices:    make(map[string]*Device),
		locations:  make(map[string]*Location),
	}
}

// NewPolicyEngine creates policy engine
func NewPolicyEngine() *PolicyEngine {
	return &PolicyEngine{
		policies: make(map[string]*Policy),
		rules:    make(map[string]*PolicyRule),
	}
}

// NewContinuousAuthEngine creates continuous auth engine
func NewContinuousAuthEngine() *ContinuousAuthEngine {
	return &ContinuousAuthEngine{
		sessions:        make(map[string]*AuthSession),
		verifiers:       make([]AuthVerifier, 0),
		reauthScheduler: &ReauthScheduler{schedules: make(map[string]*ReauthSchedule)},
	}
}

// NewMicrosegmentationEngine creates microsegmentation engine
func NewMicrosegmentationEngine() *MicrosegmentationEngine {
	return &MicrosegmentationEngine{
		segments: make(map[string]*NetworkSegment),
		policies: make(map[string]*SegmentPolicy),
		topology: &NetworkTopology{
			Segments:      make(map[string]*NetworkSegment),
			Connections:   make([]SegmentConnection, 0),
			SecurityZones: make(map[string]*SecurityZone),
		},
		enforcer: &PolicyEnforcer{
			rules:      make(map[string]*EnforcementRule),
			violations: make(map[string]*PolicyViolation),
		},
	}
}

// NewJITAccessProvisioner creates JIT access provisioner
func NewJITAccessProvisioner() *JITAccessProvisioner {
	return &JITAccessProvisioner{
		grants:      make(map[string]*AccessGrant),
		approvals:   make(map[string]*AccessApproval),
		provisioner: &ResourceProvisioner{resources: make(map[string]*ProvisionedResource)},
	}
}

// NewTrustScorer creates trust scorer
func NewTrustScorer(decay float64) *TrustScorer {
	return &TrustScorer{
		scores:  make(map[string]*TrustScore),
		factors: make([]TrustFactor, 0),
		weights: make(map[string]float64),
		decay:   decay,
	}
}

// NewSessionManager creates session manager
func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions: make(map[string]*Session),
		limits:   make(map[string]int),
	}
}

// VerifyAccess verifies access request with zero-trust principles
func (ztm *ZeroTrustManager) VerifyAccess(ctx context.Context, req *AccessRequest) (*AccessDecision, error) {
	ztm.mu.RLock()
	defer ztm.mu.RUnlock()

	startTime := time.Now()
	decision := &AccessDecision{
		RequestID:  req.ID,
		Timestamp:  startTime,
		Granted:    false,
		Reasons:    make([]string, 0),
		Conditions: make([]string, 0),
	}

	// Step 1: Verify identity
	identity, err := ztm.identityStore.GetIdentity(req.IdentityID)
	if err != nil {
		decision.Reasons = append(decision.Reasons, fmt.Sprintf("identity not found: %v", err))
		ztm.auditLogger.Log(ctx, &AuditEvent{
			Action:    "access.denied",
			UserID:    req.IdentityID,
			Resource:  req.Resource,
			Success:   false,
			Timestamp: time.Now(),
		})
		return decision, nil
	}

	// Step 2: Calculate trust score
	trustScore, err := ztm.trustScorer.CalculateTrustScore(identity)
	if err != nil {
		decision.Reasons = append(decision.Reasons, fmt.Sprintf("trust score calculation failed: %v", err))
		return decision, nil
	}

	if trustScore < ztm.config.MinTrustScore {
		decision.Reasons = append(decision.Reasons, fmt.Sprintf("trust score %.2f below minimum %.2f", trustScore, ztm.config.MinTrustScore))
		return decision, nil
	}

	// Step 3: Verify device if required
	if ztm.config.RequireDeviceAttestation && req.DeviceID != "" {
		device, err := ztm.identityStore.GetDevice(req.DeviceID)
		if err != nil || !device.Attested {
			decision.Reasons = append(decision.Reasons, "device not attested")
			return decision, nil
		}
	}

	// Step 4: Evaluate policies
	policyDecision, err := ztm.policyEngine.Evaluate(ctx, req, identity)
	if err != nil {
		decision.Reasons = append(decision.Reasons, fmt.Sprintf("policy evaluation failed: %v", err))
		return decision, nil
	}

	if !policyDecision.Allowed {
		decision.Reasons = append(decision.Reasons, policyDecision.Reason)
		return decision, nil
	}

	// Step 5: Check micro-segmentation
	if ztm.config.EnableMicrosegmentation {
		allowed, err := ztm.microsegmentation.CheckAccess(req.SourceIP, req.DestinationIP, req.Protocol, req.Port)
		if err != nil || !allowed {
			decision.Reasons = append(decision.Reasons, "network policy violation")
			return decision, nil
		}
	}

	// Step 6: Grant access
	decision.Granted = true
	decision.TrustScore = trustScore
	decision.Duration = ztm.config.MaxAccessDuration

	// Audit successful access
	ztm.auditLogger.Log(ctx, &AuditEvent{
		Action:    "access.granted",
		UserID:    req.IdentityID,
		Resource:  req.Resource,
		Success:   true,
		Timestamp: time.Now(),
		Details:   map[string]interface{}{"trust_score": trustScore},
	})

	ztm.metrics.mu.Lock()
	ztm.metrics.LastUpdated = time.Now()
	ztm.metrics.mu.Unlock()

	return decision, nil
}

// AccessRequest represents an access request
type AccessRequest struct {
	ID            string
	IdentityID    string
	DeviceID      string
	Resource      string
	Action        string
	SourceIP      string
	DestinationIP string
	Protocol      string
	Port          int
	Context       map[string]interface{}
	Timestamp     time.Time
}

// AccessDecision represents an access decision
type AccessDecision struct {
	RequestID  string
	Granted    bool
	TrustScore float64
	Reasons    []string
	Conditions []string
	Duration   time.Duration
	Timestamp  time.Time
}

// GetIdentity retrieves identity
func (is *IdentityStore) GetIdentity(id string) (*Identity, error) {
	is.mu.RLock()
	defer is.mu.RUnlock()

	identity, ok := is.identities[id]
	if !ok {
		return nil, fmt.Errorf("identity %s not found", id)
	}

	return identity, nil
}

// GetDevice retrieves device
func (is *IdentityStore) GetDevice(id string) (*Device, error) {
	is.mu.RLock()
	defer is.mu.RUnlock()

	device, ok := is.devices[id]
	if !ok {
		return nil, fmt.Errorf("device %s not found", id)
	}

	return device, nil
}

// CalculateTrustScore calculates trust score
func (ts *TrustScorer) CalculateTrustScore(identity *Identity) (float64, error) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	score := 100.0

	// Apply trust factors
	for _, factor := range ts.factors {
		factorScore := factor.Calculator(identity)
		weight := ts.weights[factor.Name]
		score -= (100.0 - factorScore) * weight
	}

	// Apply decay
	timeSinceVerification := time.Since(identity.LastVerified)
	decay := ts.decay * timeSinceVerification.Hours()
	score -= decay

	if score < 0 {
		score = 0
	}
	if score > 100 {
		score = 100
	}

	// Update score
	ts.scores[identity.ID] = &TrustScore{
		IdentityID:  identity.ID,
		Score:       score,
		LastUpdated: time.Now(),
	}

	return score, nil
}

// Evaluate evaluates policy
func (pe *PolicyEngine) Evaluate(ctx context.Context, req *AccessRequest, identity *Identity) (*PolicyDecision, error) {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	decision := &PolicyDecision{
		Allowed: false,
		Reason:  "no matching policy",
	}

	// Evaluate all policies
	for _, policy := range pe.policies {
		if !policy.Enabled {
			continue
		}

		// Check if policy applies
		if pe.policyApplies(policy, req, identity) {
			// Evaluate rules
			for _, ruleID := range policy.Rules {
				rule, ok := pe.rules[ruleID]
				if !ok {
					continue
				}

				if pe.evaluateRule(rule, req, identity) {
					if rule.Effect == EffectAllow {
						decision.Allowed = true
						decision.Reason = fmt.Sprintf("allowed by policy %s", policy.Name)
						return decision, nil
					} else if rule.Effect == EffectDeny {
						decision.Allowed = false
						decision.Reason = fmt.Sprintf("denied by policy %s", policy.Name)
						return decision, nil
					}
				}
			}
		}
	}

	return decision, nil
}

// PolicyDecision represents policy decision
type PolicyDecision struct {
	Allowed bool
	Reason  string
}

// policyApplies checks if policy applies
func (pe *PolicyEngine) policyApplies(policy *Policy, req *AccessRequest, identity *Identity) bool {
	for _, condition := range policy.Conditions {
		if !pe.evaluateCondition(condition, req, identity) {
			return false
		}
	}
	return true
}

// evaluateRule evaluates rule
func (pe *PolicyEngine) evaluateRule(rule *PolicyRule, req *AccessRequest, identity *Identity) bool {
	for _, condition := range rule.Conditions {
		if !pe.evaluateCondition(condition, req, identity) {
			return false
		}
	}
	return true
}

// evaluateCondition evaluates condition
func (pe *PolicyEngine) evaluateCondition(condition Condition, req *AccessRequest, identity *Identity) bool {
	switch condition.Type {
	case ConditionIdentityVerified:
		return identity.Verified
	case ConditionTrustScoreMinimum:
		minScore, ok := condition.Value.(float64)
		return ok && identity.TrustScore >= minScore
	default:
		return true
	}
}

// CheckAccess checks network access
func (me *MicrosegmentationEngine) CheckAccess(srcIP, dstIP, protocol string, port int) (bool, error) {
	me.mu.RLock()
	defer me.mu.RUnlock()

	// Find source and destination segments
	srcSegment := me.findSegmentForIP(srcIP)
	dstSegment := me.findSegmentForIP(dstIP)

	if srcSegment == nil || dstSegment == nil {
		return false, fmt.Errorf("segment not found")
	}

	// Check policies
	for _, policy := range me.policies {
		if policy.SourceSegment == srcSegment.ID && policy.DestSegment == dstSegment.ID {
			if policy.Protocol == protocol && me.portMatches(policy.Ports, port) {
				return policy.Action == PolicyActionAllow, nil
			}
		}
	}

	// Default deny
	return false, nil
}

// findSegmentForIP finds segment for IP
func (me *MicrosegmentationEngine) findSegmentForIP(ipStr string) *NetworkSegment {
	ip := net.ParseIP(ipStr)
	if ip == nil {
		return nil
	}

	for _, segment := range me.segments {
		if segment.CIDR.Contains(ip) {
			return segment
		}
	}

	return nil
}

// portMatches checks if port matches
func (me *MicrosegmentationEngine) portMatches(ports []int, port int) bool {
	if len(ports) == 0 {
		return true
	}

	for _, p := range ports {
		if p == port {
			return true
		}
	}

	return false
}

// initializeDefaultPolicies initializes default policies
func (ztm *ZeroTrustManager) initializeDefaultPolicies() {
	// Add default trust factors
	ztm.trustScorer.factors = []TrustFactor{
		{
			Name:   "verification_recency",
			Weight: 0.3,
			Calculator: func(id *Identity) float64 {
				age := time.Since(id.LastVerified)
				if age < time.Hour {
					return 100.0
				}
				return 100.0 - (age.Hours() / 24.0 * 10.0)
			},
		},
		{
			Name:   "device_trust",
			Weight: 0.4,
			Calculator: func(id *Identity) float64 {
				return id.TrustScore
			},
		},
		{
			Name:   "behavior_normal",
			Weight: 0.3,
			Calculator: func(id *Identity) float64 {
				if id.BehaviorProfile == nil {
					return 50.0
				}
				return 100.0 - id.BehaviorProfile.AnomalyScore
			},
		},
	}
}

// startBackgroundTasks starts background tasks
func (ztm *ZeroTrustManager) startBackgroundTasks() {
	// Start trust score decay
	go ztm.runTrustScoreDecay()

	// Start session monitoring
	go ztm.runSessionMonitoring()

	// Start metrics collection
	go ztm.runMetricsCollection()
}

// runTrustScoreDecay runs trust score decay
func (ztm *ZeroTrustManager) runTrustScoreDecay() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		ztm.identityStore.mu.Lock()
		for _, identity := range ztm.identityStore.identities {
			timeSinceVerification := time.Since(identity.LastVerified)
			decay := ztm.config.TrustDecayRate * timeSinceVerification.Hours()
			identity.TrustScore -= decay
			if identity.TrustScore < 0 {
				identity.TrustScore = 0
			}
		}
		ztm.identityStore.mu.Unlock()
	}
}

// runSessionMonitoring runs session monitoring
func (ztm *ZeroTrustManager) runSessionMonitoring() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		ztm.sessionManager.mu.Lock()
		for id, session := range ztm.sessionManager.sessions {
			if time.Since(session.LastAccess) > ztm.config.SessionTimeout {
				session.Active = false
				delete(ztm.sessionManager.sessions, id)
			}
		}
		ztm.sessionManager.mu.Unlock()
	}
}

// runMetricsCollection runs metrics collection
func (ztm *ZeroTrustManager) runMetricsCollection() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		ztm.metrics.mu.Lock()
		ztm.metrics.TotalIdentities = int64(len(ztm.identityStore.identities))
		ztm.metrics.ActiveSessions = int64(len(ztm.authEngine.sessions))

		// Calculate average trust score
		var totalScore float64
		for _, identity := range ztm.identityStore.identities {
			totalScore += identity.TrustScore
		}
		if len(ztm.identityStore.identities) > 0 {
			ztm.metrics.TrustScoreAverage = totalScore / float64(len(ztm.identityStore.identities))
		}

		ztm.metrics.LastUpdated = time.Now()
		ztm.metrics.mu.Unlock()
	}
}

// GetMetrics returns metrics
func (ztm *ZeroTrustManager) GetMetrics() *ZeroTrustMetrics {
	ztm.metrics.mu.RLock()
	defer ztm.metrics.mu.RUnlock()

	metricsCopy := *ztm.metrics
	return &metricsCopy
}

// RegisterIdentity registers a new identity
func (ztm *ZeroTrustManager) RegisterIdentity(ctx context.Context, identity *Identity) error {
	ztm.identityStore.mu.Lock()
	defer ztm.identityStore.mu.Unlock()

	if identity.ID == "" {
		identity.ID = uuid.New().String()
	}

	identity.CreatedAt = time.Now()
	identity.UpdatedAt = time.Now()
	identity.TrustScore = 50.0 // Start with medium trust

	ztm.identityStore.identities[identity.ID] = identity

	return nil
}

// CreateNetworkSegment creates a network segment
func (ztm *ZeroTrustManager) CreateNetworkSegment(ctx context.Context, segment *NetworkSegment) error {
	ztm.microsegmentation.mu.Lock()
	defer ztm.microsegmentation.mu.Unlock()

	if segment.ID == "" {
		segment.ID = uuid.New().String()
	}

	segment.CreatedAt = time.Now()
	ztm.microsegmentation.segments[segment.ID] = segment

	return nil
}

// RequestJITAccess requests just-in-time access
func (ztm *ZeroTrustManager) RequestJITAccess(ctx context.Context, req *JITAccessRequest) (*AccessGrant, error) {
	grant := &AccessGrant{
		ID:          uuid.New().String(),
		IdentityID:  req.IdentityID,
		Resource:    req.Resource,
		Permissions: req.Permissions,
		StartTime:   time.Now(),
		EndTime:     time.Now().Add(req.Duration),
		Approved:    false,
	}

	if ztm.config.RequireApproval {
		approval := &AccessApproval{
			ID:        uuid.New().String(),
			GrantID:   grant.ID,
			Requester: req.IdentityID,
			Status:    ApprovalPending,
			Reason:    req.Reason,
			CreatedAt: time.Now(),
		}

		ztm.accessProvisioner.mu.Lock()
		ztm.accessProvisioner.approvals[approval.ID] = approval
		ztm.accessProvisioner.mu.Unlock()

		grant.ApprovalID = approval.ID
	} else {
		grant.Approved = true
	}

	ztm.accessProvisioner.mu.Lock()
	ztm.accessProvisioner.grants[grant.ID] = grant
	ztm.accessProvisioner.mu.Unlock()

	return grant, nil
}

// JITAccessRequest represents JIT access request
type JITAccessRequest struct {
	IdentityID  string
	Resource    string
	Permissions []string
	Duration    time.Duration
	Reason      string
}

// HashIdentity creates identity hash
func HashIdentity(data string) string {
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}
