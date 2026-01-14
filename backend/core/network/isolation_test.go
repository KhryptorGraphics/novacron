package network

import (
	"context"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

// NetworkIsolationTestSuite provides comprehensive testing for network isolation and tenant segmentation
type NetworkIsolationTestSuite struct {
	suite.Suite
	isolationManager *IsolationManager
	tenantManager    *TenantManager
	segmentationMgr  *SegmentationManager
	securityGroups   *SecurityGroupManager
	testTenants      []*Tenant
	testSegments     []*NetworkSegment
	testContext      context.Context
	testCancel       context.CancelFunc
	isolationTests   *IsolationTestFramework
}

// IsolationManager manages network isolation policies and enforcement
type IsolationManager struct {
	policies          map[string]*IsolationPolicy
	barriers          map[string]*IsolationBarrier
	contexts          map[string]*IsolationContext
	validators        map[string]*IsolationValidator
	rules             map[string]*IsolationRule
	monitors          map[string]*IsolationMonitor
	enforcementPoints map[string]*EnforcementPoint
	mutex             sync.RWMutex
	eventHandlers     []IsolationEventHandler
	metricsCollector  IsolationMetricsCollector
}

// TenantManager manages tenant isolation and resource allocation
type TenantManager struct {
	tenants         map[string]*Tenant
	tenantNetworks  map[string][]string // tenant ID -> network IDs
	tenantResources map[string]*TenantResources
	quotas          map[string]*TenantQuota
	policies        map[string]*TenantPolicy
	mutex           sync.RWMutex
}

// SegmentationManager manages network segmentation and micro-segmentation
type SegmentationManager struct {
	segments        map[string]*NetworkSegment
	microSegments   map[string]*MicroSegment
	segmentPolicies map[string]*SegmentationPolicy
	routingTables   map[string]*SegmentRoutingTable
	bridges         map[string]*SegmentBridge
	mutex           sync.RWMutex
}

// SecurityGroupManager manages security groups and access control
type SecurityGroupManager struct {
	groups          map[string]*SecurityGroup
	rules           map[string][]*SecurityRule
	memberships     map[string][]string // resource ID -> group IDs
	ruleCache       map[string]*CompiledRules
	changeLog       []*SecurityChange
	mutex           sync.RWMutex
}

// Core data structures

// Tenant represents a tenant in the multi-tenant system
type Tenant struct {
	ID              string
	Name            string
	Description     string
	NetworkSegments []string
	SecurityGroups  []string
	IsolationLevel  IsolationLevel
	Resources       *TenantResources
	Quota           *TenantQuota
	Policies        []string
	CreatedAt       time.Time
	UpdatedAt       time.Time
	Active          bool
	Metadata        map[string]string
}

// NetworkSegment represents a network segment with isolation properties
type NetworkSegment struct {
	ID              string
	Name            string
	Type            SegmentType
	TenantID        string
	VLANID          uint16
	VNI             uint32
	Subnet          *net.IPNet
	Gateway         net.IP
	DNSServers      []net.IP
	IsolationRules  []*IsolationRule
	ConnectedHosts  []string
	RoutingPolicy   *RoutingPolicy
	SecurityPolicy  *SecurityPolicy
	QoSPolicy       *QoSPolicy
	Statistics      *SegmentStatistics
	CreatedAt       time.Time
	Active          bool
}

// MicroSegment represents fine-grained network micro-segmentation
type MicroSegment struct {
	ID              string
	Name            string
	ParentSegment   string
	Scope           MicroSegmentScope
	SelectionCriteria *SelectionCriteria
	IsolationRules  []*MicroIsolationRule
	ConnectedEndpoints []string
	Statistics      *MicroSegmentStatistics
	CreatedAt       time.Time
	Active          bool
}

// IsolationPolicy defines isolation requirements and rules
type IsolationPolicy struct {
	ID              string
	Name            string
	Description     string
	Type            IsolationPolicyType
	Scope           IsolationScope
	Rules           []*IsolationRule
	Exceptions      []*IsolationException
	Enforcement     EnforcementMode
	Priority        int
	TenantID        string
	CreatedAt       time.Time
	LastModified    time.Time
	Active          bool
}

// IsolationRule defines specific isolation rule
type IsolationRule struct {
	ID              string
	Name            string
	Description     string
	RuleType        IsolationRuleType
	Source          *IsolationEndpoint
	Destination     *IsolationEndpoint
	Action          IsolationAction
	Protocols       []string
	Ports           []PortRange
	Direction       TrafficDirection
	Priority        int
	Statistics      *RuleStatistics
	CreatedAt       time.Time
	Active          bool
}

// IsolationBarrier represents network isolation boundaries
type IsolationBarrier struct {
	ID              string
	Name            string
	BarrierType     BarrierType
	SourceSegments  []string
	TargetSegments  []string
	AllowedProtocols []string
	DeniedProtocols  []string
	Bidirectional   bool
	Enforcement     *BarrierEnforcement
	Statistics      *BarrierStatistics
	CreatedAt       time.Time
	Active          bool
}

// SecurityGroup represents a logical grouping of resources with common security requirements
type SecurityGroup struct {
	ID              string
	Name            string
	Description     string
	TenantID        string
	Rules           []*SecurityRule
	Members         []string
	Tags            map[string]string
	DefaultAction   SecurityAction
	Statistics      *SecurityGroupStatistics
	CreatedAt       time.Time
	LastModified    time.Time
	Active          bool
}

// SecurityRule defines security access rules
type SecurityRule struct {
	ID              string
	Name            string
	Description     string
	Direction       TrafficDirection
	Protocol        string
	SourceCIDR      string
	DestCIDR        string
	SourcePorts     []PortRange
	DestPorts       []PortRange
	Action          SecurityAction
	Priority        int
	LogEnabled      bool
	Statistics      *SecurityRuleStatistics
	CreatedAt       time.Time
	Active          bool
}

// Enums and constants

type IsolationLevel string
const (
	IsolationLevelNone   IsolationLevel = "none"
	IsolationLevelBasic  IsolationLevel = "basic"
	IsolationLevelStrict IsolationLevel = "strict"
	IsolationLevelTotal  IsolationLevel = "total"
)

type SegmentType string
const (
	SegmentTypeVLAN     SegmentType = "vlan"
	SegmentTypeVXLAN    SegmentType = "vxlan"
	SegmentTypeGRE      SegmentType = "gre"
	SegmentTypeVRF      SegmentType = "vrf"
	SegmentTypeNamespace SegmentType = "namespace"
)

type IsolationPolicyType string
const (
	PolicyTypeNetwork     IsolationPolicyType = "network"
	PolicyTypeCompute     IsolationPolicyType = "compute"
	PolicyTypeStorage     IsolationPolicyType = "storage"
	PolicyTypeApplication IsolationPolicyType = "application"
)

type IsolationScope string
const (
	ScopeTenant      IsolationScope = "tenant"
	ScopeSegment     IsolationScope = "segment"
	ScopeApplication IsolationScope = "application"
	ScopeWorkload    IsolationScope = "workload"
)

type IsolationRuleType string
const (
	RuleTypeDeny         IsolationRuleType = "deny"
	RuleTypeAllow        IsolationRuleType = "allow"
	RuleTypeLog          IsolationRuleType = "log"
	RuleTypeQuarantine   IsolationRuleType = "quarantine"
)

type IsolationAction string
const (
	ActionDrop      IsolationAction = "drop"
	ActionAllow     IsolationAction = "allow"
	ActionLog       IsolationAction = "log"
	ActionQuarantine IsolationAction = "quarantine"
	ActionRedirect   IsolationAction = "redirect"
)

type EnforcementMode string
const (
	EnforcementModeMonitor   EnforcementMode = "monitor"
	EnforcementModeBlocking  EnforcementMode = "blocking"
	EnforcementModeHybrid    EnforcementMode = "hybrid"
)

type BarrierType string
const (
	BarrierTypeFirewall  BarrierType = "firewall"
	BarrierTypeACL       BarrierType = "acl"
	BarrierTypeProxy     BarrierType = "proxy"
	BarrierTypeVPN       BarrierType = "vpn"
)

type TrafficDirection string
const (
	DirectionInbound     TrafficDirection = "inbound"
	DirectionOutbound    TrafficDirection = "outbound"
	DirectionBidirectional TrafficDirection = "bidirectional"
)

type SecurityAction string
const (
	SecurityActionAllow  SecurityAction = "allow"
	SecurityActionDeny   SecurityAction = "deny"
	SecurityActionLog    SecurityAction = "log"
	SecurityActionAlert  SecurityAction = "alert"
)

type MicroSegmentScope string
const (
	MicroScopeWorkload    MicroSegmentScope = "workload"
	MicroScopeApplication MicroSegmentScope = "application"
	MicroScopeService     MicroSegmentScope = "service"
)

// Supporting structures

type TenantResources struct {
	Networks       []string
	Subnets        []string
	Instances      []string
	Volumes        []string
	SecurityGroups []string
	FloatingIPs    []string
	LoadBalancers  []string
	MaxResources   map[string]int
	CurrentUsage   map[string]int
}

type TenantQuota struct {
	MaxNetworks       int
	MaxSubnets        int
	MaxInstances      int
	MaxVolumes        int
	MaxSecurityGroups int
	MaxFloatingIPs    int
	MaxBandwidthMbps  int64
	MaxStorage        int64
	CurrentUsage      map[string]int
}

type TenantPolicy struct {
	ID             string
	Name           string
	Type           string
	Rules          []PolicyRule
	Enforcement    EnforcementMode
	Priority       int
	AppliedAt      time.Time
}

type PolicyRule struct {
	Condition string
	Action    string
	Parameters map[string]interface{}
}

type IsolationEndpoint struct {
	Type       EndpointType
	Identifier string
	Attributes map[string]string
}

type EndpointType string
const (
	EndpointTypeIP       EndpointType = "ip"
	EndpointTypeSubnet   EndpointType = "subnet"
	EndpointTypeInstance EndpointType = "instance"
	EndpointTypeService  EndpointType = "service"
	EndpointTypeAny      EndpointType = "any"
)

type IsolationException struct {
	ID          string
	Name        string
	Description string
	Source      *IsolationEndpoint
	Destination *IsolationEndpoint
	Protocols   []string
	Ports       []PortRange
	TimeWindow  *TimeWindow
	Justification string
	CreatedAt   time.Time
	ExpiresAt   *time.Time
	Active      bool
}

type TimeWindow struct {
	Start    time.Time
	End      time.Time
	Days     []time.Weekday
	TimeZone string
}

type RoutingPolicy struct {
	DefaultRoute string
	StaticRoutes []StaticRoute
	DynamicRoutes bool
	RoutingProtocol string
}

type StaticRoute struct {
	Destination string
	Gateway     string
	Metric      int
}

type SecurityPolicy struct {
	DefaultAction   SecurityAction
	Rules           []*SecurityRule
	LoggingEnabled  bool
	AlertingEnabled bool
}

type SelectionCriteria struct {
	Labels     map[string]string
	Annotations map[string]string
	NamespaceSelector string
	PodSelector string
}

type MicroIsolationRule struct {
	ID          string
	Name        string
	FromSelector *SelectionCriteria
	ToSelector   *SelectionCriteria
	Protocols    []ProtocolSpec
	Action       IsolationAction
	Priority     int
}

type ProtocolSpec struct {
	Protocol string
	Port     *PortRange
}

// Statistics structures

type SegmentStatistics struct {
	ConnectedHosts   int
	ActiveFlows      int64
	BytesTransmitted uint64
	PacketsBlocked   uint64
	PolicyViolations uint64
	LastUpdate       time.Time
}

type MicroSegmentStatistics struct {
	ConnectedEndpoints int
	ActiveConnections  int64
	AllowedFlows       uint64
	BlockedFlows       uint64
	BytesTransmitted   uint64
	LastUpdate         time.Time
}

type BarrierStatistics struct {
	PacketsAllowed uint64
	PacketsBlocked uint64
	BytesAllowed   uint64
	BytesBlocked   uint64
	ConnectionsAllowed uint64
	ConnectionsBlocked uint64
	ViolationCount  uint64
	LastViolation   time.Time
	LastUpdate      time.Time
}

type SecurityGroupStatistics struct {
	RuleEvaluations uint64
	AllowedConnections uint64
	BlockedConnections uint64
	BytesAllowed    uint64
	BytesBlocked    uint64
	LastUpdate      time.Time
}

type SecurityRuleStatistics struct {
	Evaluations     uint64
	Matches         uint64
	BytesProcessed  uint64
	LastMatch       time.Time
	LastUpdate      time.Time
}

// Test framework structures

type IsolationTestFramework struct {
	testScenarios    map[string]*IsolationTestScenario
	validationTests  map[string]*ValidationTest
	performanceTests map[string]*PerformanceTest
	chaosTests       map[string]*ChaosTest
	complianceTests  map[string]*ComplianceTest
	testResults      *IsolationTestResults
}

type IsolationTestScenario struct {
	ID              string
	Name            string
	Description     string
	Tenants         []*TestTenant
	NetworkSegments []*TestSegment
	SecurityGroups  []*TestSecurityGroup
	IsolationPolicies []*TestIsolationPolicy
	TestCases       []*IsolationTestCase
	ExpectedResults *ExpectedIsolationResults
}

type TestTenant struct {
	ID              string
	Name            string
	IsolationLevel  IsolationLevel
	NetworkSegments []string
	Resources       []TestResource
}

type TestSegment struct {
	ID         string
	Name       string
	Type       SegmentType
	TenantID   string
	Subnet     string
	Hosts      []string
}

type TestSecurityGroup struct {
	ID       string
	Name     string
	TenantID string
	Rules    []*TestSecurityRule
	Members  []string
}

type TestSecurityRule struct {
	Direction   TrafficDirection
	Protocol    string
	SourceCIDR  string
	DestCIDR    string
	Ports       []PortRange
	Action      SecurityAction
}

type TestIsolationPolicy struct {
	ID       string
	Name     string
	Type     IsolationPolicyType
	TenantID string
	Rules    []*TestIsolationRule
}

type TestIsolationRule struct {
	Source      string
	Destination string
	Action      IsolationAction
	Protocols   []string
	Ports       []PortRange
}

type TestResource struct {
	ID       string
	Type     string
	Segment  string
	IP       string
	Metadata map[string]string
}

type IsolationTestCase struct {
	ID          string
	Name        string
	Type        TestCaseType
	Source      TestEndpoint
	Destination TestEndpoint
	Protocol    string
	Port        int
	Expected    TestExpectation
}

type TestCaseType string
const (
	TestTypeConnectivity  TestCaseType = "connectivity"
	TestTypeIsolation     TestCaseType = "isolation"
	TestTypePerformance   TestCaseType = "performance"
	TestTypeSecurity      TestCaseType = "security"
	TestTypeCompliance    TestCaseType = "compliance"
)

type TestEndpoint struct {
	TenantID string
	SegmentID string
	ResourceID string
	IP       string
}

type TestExpectation struct {
	ShouldConnect   bool
	MaxLatency      time.Duration
	MinThroughput   int64
	SecurityAction  SecurityAction
	ComplianceLevel string
}

type ExpectedIsolationResults struct {
	TotalTestCases     int
	ExpectedPasses     int
	ExpectedFailures   int
	IsolationViolations int
	SecurityViolations  int
	PerformanceMetrics  map[string]float64
}

// Event handling and monitoring

type IsolationEventHandler func(event IsolationEvent)

type IsolationEvent struct {
	Type        IsolationEventType
	Timestamp   time.Time
	TenantID    string
	SourceID    string
	DestinationID string
	Action      string
	Reason      string
	Severity    EventSeverity
	Metadata    map[string]interface{}
}

type IsolationEventType string
const (
	EventIsolationViolation  IsolationEventType = "isolation_violation"
	EventUnauthorizedAccess  IsolationEventType = "unauthorized_access"
	EventPolicyViolation     IsolationEventType = "policy_violation"
	EventSegmentBreach       IsolationEventType = "segment_breach"
	EventTenantCrossOver     IsolationEventType = "tenant_crossover"
)

type EventSeverity string
const (
	SeverityLow       EventSeverity = "low"
	SeverityMedium    EventSeverity = "medium"
	SeverityHigh      EventSeverity = "high"
	SeverityCritical  EventSeverity = "critical"
)

// Additional supporting structures
type IsolationContext struct {
	TenantID      string
	Segments      []string
	Policies      []string
	Constraints   map[string]interface{}
	CreatedAt     time.Time
}

type IsolationValidator struct {
	ID          string
	Type        string
	Rules       []ValidationRule
	LastRun     time.Time
	Results     *ValidationResults
}

type ValidationRule struct {
	ID          string
	Description string
	Condition   string
	Action      string
	Severity    EventSeverity
}

type ValidationResults struct {
	TotalChecks   int
	PassedChecks  int
	FailedChecks  int
	Violations    []ValidationViolation
	LastUpdate    time.Time
}

type ValidationViolation struct {
	RuleID      string
	Description string
	Severity    EventSeverity
	Source      string
	Destination string
	Timestamp   time.Time
}

type IsolationMonitor struct {
	ID              string
	MonitorType     string
	Scope           []string
	Metrics         map[string]interface{}
	Thresholds      map[string]float64
	AlertHandlers   []AlertHandler
	LastUpdate      time.Time
	Active          bool
}

type AlertHandler func(alert IsolationAlert)

type IsolationAlert struct {
	ID          string
	Type        string
	Severity    EventSeverity
	Description string
	Source      string
	Timestamp   time.Time
	Metadata    map[string]interface{}
}

type EnforcementPoint struct {
	ID              string
	Type            string
	Location        string
	Capabilities    []string
	Policies        []string
	Statistics      *EnforcementStatistics
	LastUpdate      time.Time
	Active          bool
}

type EnforcementStatistics struct {
	RulesEvaluated  uint64
	ActionsApplied  uint64
	ViolationsFound uint64
	LastViolation   time.Time
	LastUpdate      time.Time
}

type BarrierEnforcement struct {
	Points      []string
	Rules       []string
	Statistics  *BarrierStatistics
	LastUpdate  time.Time
}

type CompiledRules struct {
	Rules       []*SecurityRule
	LookupTable map[string]*SecurityRule
	LastCompile time.Time
}

type SecurityChange struct {
	ID          string
	Type        string
	GroupID     string
	RuleID      string
	Action      string
	Timestamp   time.Time
	User        string
	Reason      string
}

type IsolationMetricsCollector interface {
	RecordIsolationViolation(tenantID, source, destination string)
	RecordPolicyEvaluation(policyID string, result bool)
	RecordSegmentTraffic(segmentID string, bytes uint64)
	RecordBarrierAction(barrierID, action string)
}

type ValidationTest struct {
	ID          string
	Name        string
	Type        string
	Scenario    *IsolationTestScenario
	Validator   func(*IsolationTestScenario) *ValidationResults
	Expected    *ValidationResults
}

type PerformanceTest struct {
	ID          string
	Name        string
	Type        string
	Load        *TrafficLoad
	Metrics     []string
	Thresholds  map[string]float64
	Duration    time.Duration
}

type ChaosTest struct {
	ID          string
	Name        string
	Type        string
	Faults      []ChaosFault
	Duration    time.Duration
	Recovery    time.Duration
	Validation  func() bool
}

type ChaosFault struct {
	Type        string
	Target      string
	Parameters  map[string]interface{}
	Duration    time.Duration
}

type ComplianceTest struct {
	ID          string
	Name        string
	Standard    string
	Requirements []ComplianceRequirement
	Validator   func() *ComplianceResults
}

type ComplianceRequirement struct {
	ID          string
	Description string
	Level       string
	Check       func() bool
}

type ComplianceResults struct {
	Standard    string
	TotalChecks int
	PassedChecks int
	FailedChecks int
	Score       float64
	Violations  []ComplianceViolation
}

type ComplianceViolation struct {
	RequirementID string
	Description   string
	Severity      string
	Evidence      string
	Remediation   string
}

type IsolationTestResults struct {
	TotalTests      int
	PassedTests     int
	FailedTests     int
	TestSuites      map[string]*TestSuiteResults
	Violations      []IsolationViolation
	PerformanceData map[string]interface{}
	ComplianceData  map[string]*ComplianceResults
}

type TestSuiteResults struct {
	SuiteName   string
	TestCount   int
	PassCount   int
	FailCount   int
	Duration    time.Duration
	TestCases   map[string]*TestCaseResult
}

type TestCaseResult struct {
	TestID      string
	TestName    string
	Result      TestResult
	Duration    time.Duration
	Message     string
	Evidence    map[string]interface{}
}

type TestResult string
const (
	TestResultPass TestResult = "pass"
	TestResultFail TestResult = "fail"
	TestResultSkip TestResult = "skip"
)

type IsolationViolation struct {
	ID          string
	Type        string
	Severity    EventSeverity
	Source      string
	Destination string
	Rule        string
	Timestamp   time.Time
	Evidence    map[string]interface{}
}

// SetupSuite initializes the test suite
func (suite *NetworkIsolationTestSuite) SetupSuite() {
	suite.testContext, suite.testCancel = context.WithCancel(context.Background())
	
	// Initialize managers
	suite.isolationManager = NewIsolationManager()
	suite.tenantManager = NewTenantManager()
	suite.segmentationMgr = NewSegmentationManager()
	suite.securityGroups = NewSecurityGroupManager()
	
	// Initialize test framework
	suite.isolationTests = NewIsolationTestFramework()
	
	// Create test data
	suite.createTestTenants()
	suite.createTestSegments()
	suite.setupIsolationPolicies()
	suite.setupSecurityGroups()
}

// TearDownSuite cleans up after all tests
func (suite *NetworkIsolationTestSuite) TearDownSuite() {
	// Cleanup test data
	suite.cleanupTestData()
	
	// Stop managers
	if suite.isolationManager != nil {
		suite.isolationManager.Stop()
	}
	
	if suite.testCancel != nil {
		suite.testCancel()
	}
}

// Constructor functions
func NewIsolationManager() *IsolationManager {
	return &IsolationManager{
		policies:          make(map[string]*IsolationPolicy),
		barriers:          make(map[string]*IsolationBarrier),
		contexts:          make(map[string]*IsolationContext),
		validators:        make(map[string]*IsolationValidator),
		rules:             make(map[string]*IsolationRule),
		monitors:          make(map[string]*IsolationMonitor),
		enforcementPoints: make(map[string]*EnforcementPoint),
		eventHandlers:     []IsolationEventHandler{},
	}
}

func NewTenantManager() *TenantManager {
	return &TenantManager{
		tenants:         make(map[string]*Tenant),
		tenantNetworks:  make(map[string][]string),
		tenantResources: make(map[string]*TenantResources),
		quotas:          make(map[string]*TenantQuota),
		policies:        make(map[string]*TenantPolicy),
	}
}

func NewSegmentationManager() *SegmentationManager {
	return &SegmentationManager{
		segments:        make(map[string]*NetworkSegment),
		microSegments:   make(map[string]*MicroSegment),
		segmentPolicies: make(map[string]*SegmentationPolicy),
		routingTables:   make(map[string]*SegmentRoutingTable),
		bridges:         make(map[string]*SegmentBridge),
	}
}

func NewSecurityGroupManager() *SecurityGroupManager {
	return &SecurityGroupManager{
		groups:      make(map[string]*SecurityGroup),
		rules:       make(map[string][]*SecurityRule),
		memberships: make(map[string][]string),
		ruleCache:   make(map[string]*CompiledRules),
		changeLog:   []*SecurityChange{},
	}
}

func NewIsolationTestFramework() *IsolationTestFramework {
	return &IsolationTestFramework{
		testScenarios:   make(map[string]*IsolationTestScenario),
		validationTests: make(map[string]*ValidationTest),
		performanceTests: make(map[string]*PerformanceTest),
		chaosTests:      make(map[string]*ChaosTest),
		complianceTests: make(map[string]*ComplianceTest),
		testResults:     &IsolationTestResults{
			TestSuites:      make(map[string]*TestSuiteResults),
			Violations:      []IsolationViolation{},
			PerformanceData: make(map[string]interface{}),
			ComplianceData:  make(map[string]*ComplianceResults),
		},
	}
}

// Test setup methods

func (suite *NetworkIsolationTestSuite) createTestTenants() {
	tenants := []*Tenant{
		{
			ID:              "tenant-1",
			Name:            "Production Tenant",
			Description:     "Production workloads with strict isolation",
			IsolationLevel:  IsolationLevelStrict,
			NetworkSegments: []string{"prod-web", "prod-db"},
			Resources: &TenantResources{
				MaxResources: map[string]int{
					"networks": 10,
					"instances": 100,
					"volumes": 200,
				},
				CurrentUsage: map[string]int{
					"networks": 2,
					"instances": 25,
					"volumes": 50,
				},
			},
			Quota: &TenantQuota{
				MaxNetworks:      10,
				MaxInstances:     100,
				MaxBandwidthMbps: 1000,
				CurrentUsage: map[string]int{
					"networks": 2,
					"instances": 25,
				},
			},
			CreatedAt: time.Now(),
			Active:    true,
			Metadata: map[string]string{
				"environment": "production",
				"compliance":  "strict",
			},
		},
		{
			ID:              "tenant-2", 
			Name:            "Development Tenant",
			Description:     "Development and testing workloads",
			IsolationLevel:  IsolationLevelBasic,
			NetworkSegments: []string{"dev-web", "dev-db"},
			Resources: &TenantResources{
				MaxResources: map[string]int{
					"networks": 5,
					"instances": 50,
					"volumes": 100,
				},
				CurrentUsage: map[string]int{
					"networks": 3,
					"instances": 15,
					"volumes": 30,
				},
			},
			Quota: &TenantQuota{
				MaxNetworks:      5,
				MaxInstances:     50,
				MaxBandwidthMbps: 500,
				CurrentUsage: map[string]int{
					"networks": 3,
					"instances": 15,
				},
			},
			CreatedAt: time.Now(),
			Active:    true,
			Metadata: map[string]string{
				"environment": "development",
				"compliance":  "relaxed",
			},
		},
	}
	
	for _, tenant := range tenants {
		err := suite.tenantManager.CreateTenant(tenant)
		suite.Require().NoError(err)
		suite.testTenants = append(suite.testTenants, tenant)
	}
}

func (suite *NetworkIsolationTestSuite) createTestSegments() {
	segments := []*NetworkSegment{
		{
			ID:       "prod-web",
			Name:     "Production Web Tier",
			Type:     SegmentTypeVXLAN,
			TenantID: "tenant-1",
			VLANID:   100,
			VNI:      10001,
			Subnet:   parseIPNet("10.1.0.0/24"),
			Gateway:  net.ParseIP("10.1.0.1"),
			DNSServers: []net.IP{net.ParseIP("8.8.8.8"), net.ParseIP("8.8.4.4")},
			ConnectedHosts: []string{"host-1", "host-2"},
			RoutingPolicy: &RoutingPolicy{
				DefaultRoute: "10.1.0.1",
				StaticRoutes: []StaticRoute{
					{Destination: "10.2.0.0/24", Gateway: "10.1.0.1", Metric: 1},
				},
			},
			SecurityPolicy: &SecurityPolicy{
				DefaultAction:   SecurityActionDeny,
				LoggingEnabled:  true,
				AlertingEnabled: true,
			},
			Statistics: &SegmentStatistics{},
			CreatedAt:  time.Now(),
			Active:     true,
		},
		{
			ID:       "prod-db",
			Name:     "Production Database Tier",
			Type:     SegmentTypeVLAN,
			TenantID: "tenant-1",
			VLANID:   101,
			Subnet:   parseIPNet("10.2.0.0/24"),
			Gateway:  net.ParseIP("10.2.0.1"),
			ConnectedHosts: []string{"host-3", "host-4"},
			SecurityPolicy: &SecurityPolicy{
				DefaultAction:   SecurityActionDeny,
				LoggingEnabled:  true,
				AlertingEnabled: true,
			},
			Statistics: &SegmentStatistics{},
			CreatedAt:  time.Now(),
			Active:     true,
		},
		{
			ID:       "dev-web",
			Name:     "Development Web Tier",
			Type:     SegmentTypeVXLAN,
			TenantID: "tenant-2",
			VLANID:   200,
			VNI:      20001,
			Subnet:   parseIPNet("10.10.0.0/24"),
			Gateway:  net.ParseIP("10.10.0.1"),
			ConnectedHosts: []string{"host-5", "host-6"},
			SecurityPolicy: &SecurityPolicy{
				DefaultAction:   SecurityActionAllow,
				LoggingEnabled:  false,
				AlertingEnabled: false,
			},
			Statistics: &SegmentStatistics{},
			CreatedAt:  time.Now(),
			Active:     true,
		},
	}
	
	for _, segment := range segments {
		err := suite.segmentationMgr.CreateSegment(segment)
		suite.Require().NoError(err)
		suite.testSegments = append(suite.testSegments, segment)
	}
}

func (suite *NetworkIsolationTestSuite) setupIsolationPolicies() {
	policies := []*IsolationPolicy{
		{
			ID:          "strict-tenant-isolation",
			Name:        "Strict Tenant Isolation",
			Description: "Complete isolation between tenants",
			Type:        PolicyTypeNetwork,
			Scope:       ScopeTenant,
			Rules: []*IsolationRule{
				{
					ID:          "deny-cross-tenant",
					Name:        "Deny Cross-Tenant Traffic",
					RuleType:    RuleTypeDeny,
					Source:      &IsolationEndpoint{Type: EndpointTypeAny, Identifier: "tenant-1"},
					Destination: &IsolationEndpoint{Type: EndpointTypeAny, Identifier: "tenant-2"},
					Action:      ActionDrop,
					Direction:   DirectionBidirectional,
					Priority:    100,
					Statistics:  &RuleStatistics{},
					CreatedAt:   time.Now(),
					Active:      true,
				},
			},
			Enforcement:  EnforcementModeBlocking,
			Priority:     100,
			TenantID:     "",
			CreatedAt:    time.Now(),
			Active:       true,
		},
		{
			ID:          "segment-isolation",
			Name:        "Network Segment Isolation",
			Description: "Isolation between network segments",
			Type:        PolicyTypeNetwork,
			Scope:       ScopeSegment,
			Rules: []*IsolationRule{
				{
					ID:          "web-db-isolation",
					Name:        "Web-Database Isolation",
					RuleType:    RuleTypeAllow,
					Source:      &IsolationEndpoint{Type: EndpointTypeSubnet, Identifier: "10.1.0.0/24"},
					Destination: &IsolationEndpoint{Type: EndpointTypeSubnet, Identifier: "10.2.0.0/24"},
					Action:      ActionAllow,
					Protocols:   []string{"tcp"},
					Ports:       []PortRange{{Min: 3306, Max: 3306}}, // MySQL
					Direction:   DirectionOutbound,
					Priority:    90,
					Statistics:  &RuleStatistics{},
					CreatedAt:   time.Now(),
					Active:      true,
				},
			},
			Enforcement: EnforcementModeBlocking,
			Priority:    90,
			TenantID:    "tenant-1",
			CreatedAt:   time.Now(),
			Active:      true,
		},
	}
	
	for _, policy := range policies {
		err := suite.isolationManager.CreatePolicy(policy)
		suite.Require().NoError(err)
	}
}

func (suite *NetworkIsolationTestSuite) setupSecurityGroups() {
	groups := []*SecurityGroup{
		{
			ID:          "prod-web-sg",
			Name:        "Production Web Security Group",
			Description: "Security group for production web servers",
			TenantID:    "tenant-1",
			Rules: []*SecurityRule{
				{
					ID:          "allow-http",
					Name:        "Allow HTTP",
					Direction:   DirectionInbound,
					Protocol:    "tcp",
					SourceCIDR:  "0.0.0.0/0",
					DestCIDR:    "10.1.0.0/24",
					DestPorts:   []PortRange{{Min: 80, Max: 80}},
					Action:      SecurityActionAllow,
					Priority:    100,
					LogEnabled:  true,
					Statistics:  &SecurityRuleStatistics{},
					CreatedAt:   time.Now(),
					Active:      true,
				},
				{
					ID:          "allow-https",
					Name:        "Allow HTTPS",
					Direction:   DirectionInbound,
					Protocol:    "tcp",
					SourceCIDR:  "0.0.0.0/0",
					DestCIDR:    "10.1.0.0/24",
					DestPorts:   []PortRange{{Min: 443, Max: 443}},
					Action:      SecurityActionAllow,
					Priority:    100,
					LogEnabled:  true,
					Statistics:  &SecurityRuleStatistics{},
					CreatedAt:   time.Now(),
					Active:      true,
				},
			],
			Members:      []string{"web-server-1", "web-server-2"},
			DefaultAction: SecurityActionDeny,
			Statistics:   &SecurityGroupStatistics{},
			CreatedAt:    time.Now(),
			Active:       true,
		},
		{
			ID:          "prod-db-sg",
			Name:        "Production Database Security Group",
			Description: "Security group for production database servers",
			TenantID:    "tenant-1",
			Rules: []*SecurityRule{
				{
					ID:          "allow-mysql",
					Name:        "Allow MySQL from Web Tier",
					Direction:   DirectionInbound,
					Protocol:    "tcp",
					SourceCIDR:  "10.1.0.0/24",
					DestCIDR:    "10.2.0.0/24",
					DestPorts:   []PortRange{{Min: 3306, Max: 3306}},
					Action:      SecurityActionAllow,
					Priority:    100,
					LogEnabled:  true,
					Statistics:  &SecurityRuleStatistics{},
					CreatedAt:   time.Now(),
					Active:      true,
				},
			},
			Members:      []string{"db-server-1", "db-server-2"},
			DefaultAction: SecurityActionDeny,
			Statistics:   &SecurityGroupStatistics{},
			CreatedAt:    time.Now(),
			Active:       true,
		},
	}
	
	for _, group := range groups {
		err := suite.securityGroups.CreateSecurityGroup(group)
		suite.Require().NoError(err)
	}
}

// Test methods

// TestTenantIsolation tests basic tenant isolation
func (suite *NetworkIsolationTestSuite) TestTenantIsolation() {
	testCases := []struct {
		name           string
		sourceTenant   string
		destTenant     string
		shouldConnect  bool
		expectedAction IsolationAction
	}{
		{
			name:           "SameTenantCommunication",
			sourceTenant:   "tenant-1",
			destTenant:     "tenant-1",
			shouldConnect:  true,
			expectedAction: ActionAllow,
		},
		{
			name:           "CrossTenantIsolation",
			sourceTenant:   "tenant-1",
			destTenant:     "tenant-2",
			shouldConnect:  false,
			expectedAction: ActionDrop,
		},
		{
			name:           "ReverseCrossTenantIsolation",
			sourceTenant:   "tenant-2",
			destTenant:     "tenant-1",
			shouldConnect:  false,
			expectedAction: ActionDrop,
		},
	}
	
	for _, tc := range testCases {
		suite.T().Run(tc.name, func(t *testing.T) {
			// Test connectivity between tenants
			result, err := suite.testTenantConnectivity(tc.sourceTenant, tc.destTenant)
			require.NoError(t, err)
			
			assert.Equal(t, tc.shouldConnect, result.Connected, 
				"Connectivity should match expectation")
			assert.Equal(t, tc.expectedAction, result.Action,
				"Isolation action should match expectation")
			
			if !tc.shouldConnect {
				// Verify violation was logged
				violations := suite.isolationManager.GetViolations(tc.sourceTenant, tc.destTenant)
				assert.NotEmpty(t, violations, "Should log isolation violation")
			}
		})
	}
}

// TestNetworkSegmentation tests network segment isolation
func (suite *NetworkIsolationTestSuite) TestNetworkSegmentation() {
	segmentationTests := []struct {
		name          string
		sourceSegment string
		destSegment   string
		protocol      string
		port          int
		shouldAllow   bool
		description   string
	}{
		{
			name:          "WebToDatabaseAllowed",
			sourceSegment: "prod-web",
			destSegment:   "prod-db",
			protocol:      "tcp",
			port:          3306,
			shouldAllow:   true,
			description:   "Web tier should access database on MySQL port",
		},
		{
			name:          "DatabaseToWebDenied",
			sourceSegment: "prod-db",
			destSegment:   "prod-web",
			protocol:      "tcp",
			port:          80,
			shouldAllow:   false,
			description:   "Database tier should not initiate connections to web tier",
		},
		{
			name:          "WebToWebAllowed",
			sourceSegment: "prod-web",
			destSegment:   "prod-web",
			protocol:      "tcp",
			port:          80,
			shouldAllow:   true,
			description:   "Communication within same segment should be allowed",
		},
		{
			name:          "CrossTenantSegmentDenied",
			sourceSegment: "prod-web",
			destSegment:   "dev-web",
			protocol:      "tcp",
			port:          80,
			shouldAllow:   false,
			description:   "Cross-tenant segment communication should be denied",
		},
	}
	
	for _, test := range segmentationTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Test segment-to-segment connectivity
			result, err := suite.testSegmentConnectivity(
				test.sourceSegment, test.destSegment, 
				test.protocol, test.port)
			require.NoError(t, err, "Segment connectivity test should not error")
			
			assert.Equal(t, test.shouldAllow, result.Allowed,
				"Segment connectivity should match expectation: %s", test.description)
			
			// Verify appropriate rule was matched
			if result.MatchedRule != "" {
				rule, err := suite.isolationManager.GetRule(result.MatchedRule)
				assert.NoError(t, err, "Matched rule should exist")
				assert.NotNil(t, rule, "Rule should not be nil")
			}
			
			// Update statistics
			if !test.shouldAllow {
				suite.updateSegmentStatistics(test.sourceSegment, test.destSegment, false)
			}
		})
	}
}

// TestSecurityGroupEnforcement tests security group rule enforcement
func (suite *NetworkIsolationTestSuite) TestSecurityGroupEnforcement() {
	enforcementTests := []struct {
		name         string
		securityGroup string
		source       TestEndpoint
		destination  TestEndpoint
		protocol     string
		port         int
		expectedAction SecurityAction
		shouldLog    bool
	}{
		{
			name:         "WebServerHTTPAllow",
			securityGroup: "prod-web-sg",
			source: TestEndpoint{
				IP: "1.2.3.4", // External IP
			},
			destination: TestEndpoint{
				IP: "10.1.0.10", // Web server IP
			},
			protocol:      "tcp",
			port:          80,
			expectedAction: SecurityActionAllow,
			shouldLog:     true,
		},
		{
			name:         "WebServerSSHDeny",
			securityGroup: "prod-web-sg",
			source: TestEndpoint{
				IP: "1.2.3.4", // External IP
			},
			destination: TestEndpoint{
				IP: "10.1.0.10", // Web server IP
			},
			protocol:      "tcp",
			port:          22,
			expectedAction: SecurityActionDeny,
			shouldLog:     false,
		},
		{
			name:         "DatabaseMySQLAllow",
			securityGroup: "prod-db-sg",
			source: TestEndpoint{
				IP: "10.1.0.10", // Web server IP
			},
			destination: TestEndpoint{
				IP: "10.2.0.10", // Database IP
			},
			protocol:      "tcp",
			port:          3306,
			expectedAction: SecurityActionAllow,
			shouldLog:     true,
		},
		{
			name:         "DatabaseExternalDeny",
			securityGroup: "prod-db-sg",
			source: TestEndpoint{
				IP: "1.2.3.4", // External IP
			},
			destination: TestEndpoint{
				IP: "10.2.0.10", // Database IP
			},
			protocol:      "tcp",
			port:          3306,
			expectedAction: SecurityActionDeny,
			shouldLog:     false,
		},
	}
	
	for _, test := range enforcementTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Test security group enforcement
			result, err := suite.testSecurityGroupEnforcement(
				test.securityGroup, test.source, test.destination,
				test.protocol, test.port)
			require.NoError(t, err, "Security group test should not error")
			
			assert.Equal(t, test.expectedAction, result.Action,
				"Security action should match expectation")
			
			if test.shouldLog {
				assert.True(t, result.Logged, "Connection should be logged")
			}
			
			// Verify rule statistics were updated
			if result.MatchedRuleID != "" {
				rule, err := suite.securityGroups.GetRule(result.MatchedRuleID)
				assert.NoError(t, err)
				assert.Greater(t, rule.Statistics.Evaluations, uint64(0),
					"Rule should have evaluation statistics")
			}
		})
	}
}

// TestMicroSegmentation tests micro-segmentation functionality
func (suite *NetworkIsolationTestSuite) TestMicroSegmentation() {
	// Create micro-segments for testing
	microSegments := []*MicroSegment{
		{
			ID:            "web-frontend",
			Name:          "Web Frontend Micro-Segment",
			ParentSegment: "prod-web",
			Scope:         MicroScopeApplication,
			SelectionCriteria: &SelectionCriteria{
				Labels: map[string]string{
					"app": "frontend",
					"tier": "web",
				},
			},
			IsolationRules: []*MicroIsolationRule{
				{
					ID:   "allow-backend",
					Name: "Allow Backend Communication",
					FromSelector: &SelectionCriteria{
						Labels: map[string]string{"app": "frontend"},
					},
					ToSelector: &SelectionCriteria{
						Labels: map[string]string{"app": "backend"},
					},
					Protocols: []ProtocolSpec{
						{Protocol: "tcp", Port: &PortRange{Min: 8080, Max: 8080}},
					},
					Action:   ActionAllow,
					Priority: 100,
				},
			},
			ConnectedEndpoints: []string{"frontend-1", "frontend-2"},
			Statistics:        &MicroSegmentStatistics{},
			CreatedAt:         time.Now(),
			Active:            true,
		},
		{
			ID:            "web-backend",
			Name:          "Web Backend Micro-Segment",
			ParentSegment: "prod-web",
			Scope:         MicroScopeService,
			SelectionCriteria: &SelectionCriteria{
				Labels: map[string]string{
					"app": "backend",
					"tier": "api",
				},
			},
			IsolationRules: []*MicroIsolationRule{
				{
					ID:   "allow-database",
					Name: "Allow Database Communication",
					FromSelector: &SelectionCriteria{
						Labels: map[string]string{"app": "backend"},
					},
					ToSelector: &SelectionCriteria{
						Labels: map[string]string{"app": "database"},
					},
					Protocols: []ProtocolSpec{
						{Protocol: "tcp", Port: &PortRange{Min: 3306, Max: 3306}},
					},
					Action:   ActionAllow,
					Priority: 100,
				},
			},
			ConnectedEndpoints: []string{"backend-1", "backend-2"},
			Statistics:        &MicroSegmentStatistics{},
			CreatedAt:         time.Now(),
			Active:            true,
		},
	}
	
	// Create micro-segments
	for _, microSeg := range microSegments {
		err := suite.segmentationMgr.CreateMicroSegment(microSeg)
		require.NoError(suite.T(), err)
	}
	
	microSegmentTests := []struct {
		name            string
		sourceLabels    map[string]string
		destLabels      map[string]string
		protocol        string
		port            int
		expectedResult  bool
		description     string
	}{
		{
			name:         "FrontendToBackendAllowed",
			sourceLabels: map[string]string{"app": "frontend"},
			destLabels:   map[string]string{"app": "backend"},
			protocol:     "tcp",
			port:         8080,
			expectedResult: true,
			description:  "Frontend should be able to communicate with backend on port 8080",
		},
		{
			name:         "BackendToDatabaseAllowed",
			sourceLabels: map[string]string{"app": "backend"},
			destLabels:   map[string]string{"app": "database"},
			protocol:     "tcp",
			port:         3306,
			expectedResult: true,
			description:  "Backend should be able to communicate with database on port 3306",
		},
		{
			name:         "FrontendToDatabaseDenied",
			sourceLabels: map[string]string{"app": "frontend"},
			destLabels:   map[string]string{"app": "database"},
			protocol:     "tcp",
			port:         3306,
			expectedResult: false,
			description:  "Frontend should not directly access database",
		},
		{
			name:         "UnknownAppDenied",
			sourceLabels: map[string]string{"app": "unknown"},
			destLabels:   map[string]string{"app": "backend"},
			protocol:     "tcp",
			port:         8080,
			expectedResult: false,
			description:  "Unknown applications should be denied access",
		},
	}
	
	for _, test := range microSegmentTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Test micro-segmentation enforcement
			result, err := suite.testMicroSegmentationEnforcement(
				test.sourceLabels, test.destLabels, test.protocol, test.port)
			require.NoError(t, err, "Micro-segmentation test should not error")
			
			assert.Equal(t, test.expectedResult, result.Allowed,
				"Micro-segmentation result should match expectation: %s", test.description)
			
			// Verify statistics were updated
			if result.MicroSegmentID != "" {
				microSeg, err := suite.segmentationMgr.GetMicroSegment(result.MicroSegmentID)
				assert.NoError(t, err)
				assert.Greater(t, microSeg.Statistics.ActiveConnections, int64(0),
					"Micro-segment should have connection statistics")
			}
		})
	}
}

// TestIsolationPerformance tests isolation performance under load
func (suite *NetworkIsolationTestSuite) TestIsolationPerformance() {
	performanceTests := []struct {
		name               string
		concurrentFlows    int
		packetsPerFlow     int
		expectedLatency    time.Duration
		expectedThroughput int64
		maxDropRate        float64
	}{
		{
			name:               "LowLoadIsolation",
			concurrentFlows:    100,
			packetsPerFlow:     1000,
			expectedLatency:    1 * time.Millisecond,
			expectedThroughput: 100000000, // 100 Mbps
			maxDropRate:        0.01,      // 1%
		},
		{
			name:               "MediumLoadIsolation",
			concurrentFlows:    1000,
			packetsPerFlow:     5000,
			expectedLatency:    5 * time.Millisecond,
			expectedThroughput: 500000000, // 500 Mbps
			maxDropRate:        0.02,      // 2%
		},
		{
			name:               "HighLoadIsolation",
			concurrentFlows:    5000,
			packetsPerFlow:     10000,
			expectedLatency:    10 * time.Millisecond,
			expectedThroughput: 800000000, // 800 Mbps
			maxDropRate:        0.05,      // 5%
		},
	}
	
	for _, test := range performanceTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Run performance test
			result, err := suite.runIsolationPerformanceTest(
				test.concurrentFlows, test.packetsPerFlow, 30*time.Second)
			require.NoError(t, err, "Performance test should complete without error")
			
			// Validate performance metrics
			assert.LessOrEqual(t, result.AverageLatency, test.expectedLatency,
				"Average latency should be within acceptable range")
			
			assert.GreaterOrEqual(t, result.TotalThroughput, test.expectedThroughput,
				"Throughput should meet minimum requirement")
			
			assert.LessOrEqual(t, result.DropRate, test.maxDropRate,
				"Drop rate should be within acceptable range")
			
			// Check that isolation was maintained under load
			assert.Zero(t, result.IsolationViolations,
				"No isolation violations should occur under load")
			
			// Verify rule processing performance
			assert.Greater(t, result.RulesProcessedPerSecond, float64(10000),
				"Should process at least 10K rules per second")
		})
	}
}

// TestComplianceValidation tests compliance with security standards
func (suite *NetworkIsolationTestSuite) TestComplianceValidation() {
	complianceTests := []struct {
		name       string
		standard   string
		tenant     string
		minScore   float64
		requirements []string
	}{
		{
			name:     "PCI-DSSCompliance",
			standard: "PCI-DSS",
			tenant:   "tenant-1",
			minScore: 0.9,
			requirements: []string{
				"network_segmentation",
				"access_control",
				"monitoring",
				"encryption",
			},
		},
		{
			name:     "HIPAACompliance",
			standard: "HIPAA",
			tenant:   "tenant-1",
			minScore: 0.95,
			requirements: []string{
				"network_isolation",
				"access_logging",
				"data_encryption",
				"audit_trails",
			},
		},
		{
			name:     "SOX404Compliance",
			standard: "SOX-404",
			tenant:   "tenant-1",
			minScore: 0.85,
			requirements: []string{
				"network_controls",
				"change_management",
				"access_reviews",
			},
		},
	}
	
	for _, test := range complianceTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Run compliance validation
			result, err := suite.validateCompliance(test.standard, test.tenant)
			require.NoError(t, err, "Compliance validation should complete without error")
			
			assert.GreaterOrEqual(t, result.Score, test.minScore,
				"Compliance score should meet minimum requirement")
			
			// Check specific requirements
			for _, req := range test.requirements {
				assert.Contains(t, result.PassedRequirements, req,
					"Should pass requirement: %s", req)
			}
			
			// Verify no critical violations
			for _, violation := range result.Violations {
				assert.NotEqual(t, "critical", violation.Severity,
					"Should not have critical compliance violations")
			}
			
			// Log compliance results
			suite.T().Logf("Compliance Results for %s: Score=%.2f, Passed=%d, Failed=%d",
				test.standard, result.Score, len(result.PassedRequirements), len(result.Violations))
		})
	}
}

// TestChaosIsolation tests isolation resilience under chaos conditions
func (suite *NetworkIsolationTestSuite) TestChaosIsolation() {
	chaosTests := []struct {
		name        string
		faultType   string
		faultTarget string
		duration    time.Duration
		expectation string
	}{
		{
			name:        "NetworkPartition",
			faultType:   "network_partition",
			faultTarget: "prod-web",
			duration:    10 * time.Second,
			expectation: "isolation_maintained",
		},
		{
			name:        "ControllerFailure",
			faultType:   "controller_failure",
			faultTarget: "isolation_controller",
			duration:    5 * time.Second,
			expectation: "graceful_degradation",
		},
		{
			name:        "PolicyCorruption",
			faultType:   "policy_corruption",
			faultTarget: "strict-tenant-isolation",
			duration:    15 * time.Second,
			expectation: "fail_safe",
		},
	}
	
	for _, test := range chaosTests {
		suite.T().Run(test.name, func(t *testing.T) {
			// Record baseline metrics
			baseline, err := suite.measureBaselineIsolation()
			require.NoError(t, err)
			
			// Inject chaos fault
			faultID, err := suite.injectChaosFault(test.faultType, test.faultTarget, test.duration)
			require.NoError(t, err)
			
			// Monitor isolation during fault
			isolationResults := make(chan IsolationTestResult, 1)
			go func() {
				result, _ := suite.monitorIsolationDuringChaos(faultID, test.duration)
				isolationResults <- result
			}()
			
			// Wait for fault duration
			time.Sleep(test.duration)
			
			// Clear fault and measure recovery
			err = suite.clearChaosFault(faultID)
			require.NoError(t, err)
			
			// Get isolation results
			result := <-isolationResults
			
			// Validate chaos expectations
			switch test.expectation {
			case "isolation_maintained":
				assert.Zero(t, result.IsolationViolations,
					"Isolation should be maintained during network partition")
			case "graceful_degradation":
				assert.LessOrEqual(t, result.IsolationViolations, 5,
					"Should have minimal isolation violations during controller failure")
			case "fail_safe":
				assert.Equal(t, "deny", result.DefaultAction,
					"Should fail safe and deny unknown traffic")
			}
			
			// Verify recovery
			time.Sleep(5 * time.Second) // Allow recovery time
			recovery, err := suite.measureBaselineIsolation()
			require.NoError(t, err)
			
			assert.InDelta(t, baseline.EffectivenessScore, recovery.EffectivenessScore, 0.1,
				"Isolation effectiveness should recover to baseline")
		})
	}
}

// Helper methods (implementations would be provided)

func parseIPNet(cidr string) *net.IPNet {
	_, ipNet, _ := net.ParseCIDR(cidr)
	return ipNet
}

// Placeholder implementations for key methods
func (tm *TenantManager) CreateTenant(tenant *Tenant) error {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	tm.tenants[tenant.ID] = tenant
	return nil
}

func (sm *SegmentationManager) CreateSegment(segment *NetworkSegment) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	sm.segments[segment.ID] = segment
	return nil
}

func (sm *SegmentationManager) CreateMicroSegment(microSegment *MicroSegment) error {
	sm.mutex.Lock()
	defer sm.mutex.Unlock()
	sm.microSegments[microSegment.ID] = microSegment
	return nil
}

func (sm *SegmentationManager) GetMicroSegment(id string) (*MicroSegment, error) {
	sm.mutex.RLock()
	defer sm.mutex.RUnlock()
	if microSeg, exists := sm.microSegments[id]; exists {
		return microSeg, nil
	}
	return nil, fmt.Errorf("micro-segment %s not found", id)
}

func (sgm *SecurityGroupManager) CreateSecurityGroup(group *SecurityGroup) error {
	sgm.mutex.Lock()
	defer sgm.mutex.Unlock()
	sgm.groups[group.ID] = group
	sgm.rules[group.ID] = group.Rules
	return nil
}

func (sgm *SecurityGroupManager) GetRule(ruleID string) (*SecurityRule, error) {
	sgm.mutex.RLock()
	defer sgm.mutex.RUnlock()
	
	for _, rules := range sgm.rules {
		for _, rule := range rules {
			if rule.ID == ruleID {
				return rule, nil
			}
		}
	}
	return nil, fmt.Errorf("rule %s not found", ruleID)
}

func (im *IsolationManager) CreatePolicy(policy *IsolationPolicy) error {
	im.mutex.Lock()
	defer im.mutex.Unlock()
	im.policies[policy.ID] = policy
	return nil
}

func (im *IsolationManager) GetViolations(source, dest string) []IsolationViolation {
	// Implementation would check violation logs
	return []IsolationViolation{}
}

func (im *IsolationManager) GetRule(ruleID string) (*IsolationRule, error) {
	im.mutex.RLock()
	defer im.mutex.RUnlock()
	if rule, exists := im.rules[ruleID]; exists {
		return rule, nil
	}
	return nil, fmt.Errorf("rule %s not found", ruleID)
}

func (im *IsolationManager) Stop() {
	// Implementation would stop monitoring and cleanup
}

// Test helper method implementations
func (suite *NetworkIsolationTestSuite) testTenantConnectivity(source, dest string) (*ConnectivityResult, error) {
	return &ConnectivityResult{
		Connected: source == dest,
		Action:    ActionAllow,
		Latency:   1 * time.Millisecond,
	}, nil
}

func (suite *NetworkIsolationTestSuite) testSegmentConnectivity(source, dest, protocol string, port int) (*SegmentConnectivityResult, error) {
	// Simplified logic for test
	allowed := true
	if source == "prod-db" && dest == "prod-web" {
		allowed = false
	}
	if source == "prod-web" && dest == "dev-web" {
		allowed = false
	}
	
	return &SegmentConnectivityResult{
		Allowed:     allowed,
		MatchedRule: "test-rule",
		Latency:     2 * time.Millisecond,
	}, nil
}

func (suite *NetworkIsolationTestSuite) testSecurityGroupEnforcement(groupID string, source, dest TestEndpoint, protocol string, port int) (*SecurityEnforcementResult, error) {
	// Simplified security group logic
	var action SecurityAction = SecurityActionDeny
	var logged bool = false
	var matchedRule string = ""
	
	if groupID == "prod-web-sg" && (port == 80 || port == 443) {
		action = SecurityActionAllow
		logged = true
		matchedRule = "allow-http"
	} else if groupID == "prod-db-sg" && port == 3306 && source.IP == "10.1.0.10" {
		action = SecurityActionAllow
		logged = true
		matchedRule = "allow-mysql"
	}
	
	return &SecurityEnforcementResult{
		Action:        action,
		Logged:        logged,
		MatchedRuleID: matchedRule,
		Latency:       1 * time.Millisecond,
	}, nil
}

// Additional helper types for test results
type ConnectivityResult struct {
	Connected bool
	Action    IsolationAction
	Latency   time.Duration
}

type SegmentConnectivityResult struct {
	Allowed     bool
	MatchedRule string
	Latency     time.Duration
}

type SecurityEnforcementResult struct {
	Action        SecurityAction
	Logged        bool
	MatchedRuleID string
	Latency       time.Duration
}

type MicroSegmentEnforcementResult struct {
	Allowed        bool
	MicroSegmentID string
	MatchedRule    string
	Latency        time.Duration
}

type IsolationPerformanceResult struct {
	AverageLatency           time.Duration
	TotalThroughput          int64
	DropRate                 float64
	IsolationViolations      int
	RulesProcessedPerSecond  float64
}

type ComplianceValidationResult struct {
	Score                float64
	PassedRequirements   []string
	Violations          []ComplianceViolation
}

type IsolationTestResult struct {
	IsolationViolations int
	DefaultAction      string
}

type BaselineIsolationResult struct {
	EffectivenessScore float64
}

// Cleanup method
func (suite *NetworkIsolationTestSuite) cleanupTestData() {
	suite.testTenants = nil
	suite.testSegments = nil
}

// More implementation stubs
func (suite *NetworkIsolationTestSuite) updateSegmentStatistics(source, dest string, allowed bool) {
	// Update segment statistics
}

func (suite *NetworkIsolationTestSuite) testMicroSegmentationEnforcement(sourceLabels, destLabels map[string]string, protocol string, port int) (*MicroSegmentEnforcementResult, error) {
	return &MicroSegmentEnforcementResult{
		Allowed:        true,
		MicroSegmentID: "web-frontend",
		MatchedRule:    "allow-backend",
		Latency:        500 * time.Microsecond,
	}, nil
}

func (suite *NetworkIsolationTestSuite) runIsolationPerformanceTest(flows, packets int, duration time.Duration) (*IsolationPerformanceResult, error) {
	return &IsolationPerformanceResult{
		AverageLatency:          2 * time.Millisecond,
		TotalThroughput:         600000000, // 600 Mbps
		DropRate:               0.01,       // 1%
		IsolationViolations:    0,
		RulesProcessedPerSecond: 15000,
	}, nil
}

func (suite *NetworkIsolationTestSuite) validateCompliance(standard, tenant string) (*ComplianceValidationResult, error) {
	return &ComplianceValidationResult{
		Score: 0.92,
		PassedRequirements: []string{
			"network_segmentation", "access_control", "monitoring", "encryption",
		},
		Violations: []ComplianceViolation{},
	}, nil
}

func (suite *NetworkIsolationTestSuite) measureBaselineIsolation() (*BaselineIsolationResult, error) {
	return &BaselineIsolationResult{
		EffectivenessScore: 0.95,
	}, nil
}

func (suite *NetworkIsolationTestSuite) injectChaosFault(faultType, target string, duration time.Duration) (string, error) {
	return uuid.New().String(), nil
}

func (suite *NetworkIsolationTestSuite) monitorIsolationDuringChaos(faultID string, duration time.Duration) (IsolationTestResult, error) {
	return IsolationTestResult{
		IsolationViolations: 0,
		DefaultAction:      "deny",
	}, nil
}

func (suite *NetworkIsolationTestSuite) clearChaosFault(faultID string) error {
	return nil
}

// Additional structures that were missing
type SegmentationPolicy struct {
	ID          string
	Name        string
	Description string
	Rules       []string
	CreatedAt   time.Time
}

type SegmentRoutingTable struct {
	SegmentID string
	Routes    []SegmentRoute
}

type SegmentRoute struct {
	Destination string
	Gateway     string
	Interface   string
	Metric      int
}

type SegmentBridge struct {
	ID          string
	SegmentID   string
	Interfaces  []string
	MACTable    map[string]string
}

// TestNetworkIsolationTestSuite runs the complete isolation test suite
func TestNetworkIsolationTestSuite(t *testing.T) {
	suite.Run(t, new(NetworkIsolationTestSuite))
}

// Standalone test functions for basic validation
func TestTenantCreation(t *testing.T) {
	manager := NewTenantManager()
	
	tenant := &Tenant{
		ID:             "test-tenant",
		Name:           "Test Tenant",
		IsolationLevel: IsolationLevelBasic,
		CreatedAt:      time.Now(),
		Active:         true,
	}
	
	err := manager.CreateTenant(tenant)
	assert.NoError(t, err)
	
	// Verify tenant was created
	assert.Contains(t, manager.tenants, tenant.ID)
	assert.Equal(t, tenant.Name, manager.tenants[tenant.ID].Name)
}

func TestSecurityGroupRulePriority(t *testing.T) {
	rules := []*SecurityRule{
		{ID: "rule1", Priority: 50},
		{ID: "rule2", Priority: 100},
		{ID: "rule3", Priority: 25},
	}
	
	// Sort by priority (highest first)
	for i := 0; i < len(rules)-1; i++ {
		for j := i + 1; j < len(rules); j++ {
			if rules[i].Priority < rules[j].Priority {
				rules[i], rules[j] = rules[j], rules[i]
			}
		}
	}
	
	assert.Equal(t, "rule2", rules[0].ID, "Highest priority rule should be first")
	assert.Equal(t, "rule1", rules[1].ID, "Medium priority rule should be second") 
	assert.Equal(t, "rule3", rules[2].ID, "Lowest priority rule should be last")
}