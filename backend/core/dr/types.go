package dr

import (
	"time"
)

// DRState represents the current disaster recovery state
type DRState int

const (
	// StateNormal indicates normal operation
	StateNormal DRState = iota
	// StateDegraded indicates degraded but operational state
	StateDegraded
	// StateFailingOver indicates active failover in progress
	StateFailingOver
	// StateRecovery indicates recovery operations in progress
	StateRecovery
	// StateFailed indicates DR operations failed
	StateFailed
)

func (s DRState) String() string {
	switch s {
	case StateNormal:
		return "Normal"
	case StateDegraded:
		return "Degraded"
	case StateFailingOver:
		return "FailingOver"
	case StateRecovery:
		return "Recovery"
	case StateFailed:
		return "Failed"
	default:
		return "Unknown"
	}
}

// FailureType classifies the type of failure detected
type FailureType int

const (
	FailureTypeNode FailureType = iota
	FailureTypeRegion
	FailureTypeNetwork
	FailureTypeDataCenter
	FailureTypeCascading
	FailureTypeData
	FailureTypeSecurity
)

func (f FailureType) String() string {
	switch f {
	case FailureTypeNode:
		return "Node"
	case FailureTypeRegion:
		return "Region"
	case FailureTypeNetwork:
		return "Network"
	case FailureTypeDataCenter:
		return "DataCenter"
	case FailureTypeCascading:
		return "Cascading"
	case FailureTypeData:
		return "Data"
	case FailureTypeSecurity:
		return "Security"
	default:
		return "Unknown"
	}
}

// BackupType defines the type of backup
type BackupType int

const (
	BackupTypeFull BackupType = iota
	BackupTypeIncremental
	BackupTypeDifferential
	BackupTypeTransaction
	BackupTypeSnapshot
)

func (b BackupType) String() string {
	switch b {
	case BackupTypeFull:
		return "Full"
	case BackupTypeIncremental:
		return "Incremental"
	case BackupTypeDifferential:
		return "Differential"
	case BackupTypeTransaction:
		return "Transaction"
	case BackupTypeSnapshot:
		return "Snapshot"
	default:
		return "Unknown"
	}
}

// FailureEvent represents a detected failure
type FailureEvent struct {
	ID           string
	Type         FailureType
	Severity     int // 1-10, 10 being critical
	DetectedAt   time.Time
	AffectedZone string
	Description  string
	Metrics      map[string]interface{}
	AutoFailover bool
}

// DRStatus represents the current DR system status
type DRStatus struct {
	State              DRState
	PrimaryRegion      string
	SecondaryRegions   []string
	ActiveFailovers    int
	LastFailover       time.Time
	LastBackup         time.Time
	LastSuccessfulTest time.Time
	HealthScore        float64
	RTO                time.Duration
	RPO                time.Duration
	BackupCount        int64
	RestoreCount       int64
}

// TriggerCondition defines when automatic failover should trigger
type TriggerCondition struct {
	MetricName    string
	Threshold     float64
	Duration      time.Duration
	Operator      string // "gt", "lt", "eq", "ne"
	RequireQuorum bool
}

// BackupLocation defines where backups are stored
type BackupLocation struct {
	ID       string
	Type     string // "s3", "azure", "gcs", "tape", "local"
	Region   string
	Endpoint string
	Bucket   string
	Priority int // Lower is higher priority
	Metadata map[string]string
}

// BackupSchedule defines when backups occur
type BackupSchedule struct {
	FullBackup        string // Cron expression
	IncrementalBackup string
	TransactionLog    bool
	SnapshotInterval  time.Duration
}

// RetentionPolicy defines how long backups are kept
type RetentionPolicy struct {
	HourlyRetentionDays   int
	DailyRetentionDays    int
	WeeklyRetentionDays   int
	MonthlyRetentionDays  int
	YearlyRetentionYears  int
	LegalHoldEnabled      bool
	ComplianceMode        string // "SOC2", "ISO27001", "HIPAA", "PCI-DSS"
}

// FailoverPolicy defines automatic failover behavior
type FailoverPolicy struct {
	AutomaticTriggers   []TriggerCondition
	MinHealthyRegions   int
	QuorumRequirement   int
	MaxFailoverAttempts int
	RollbackOnFailure   bool
	ApprovalRequired    bool
	NotifyBeforeFailover bool
}

// RestoreTarget defines what to restore
type RestoreTarget struct {
	Type         string // "vm", "cluster", "region", "all"
	TargetID     string
	TargetRegion string
	PointInTime  time.Time
	Selective    []string // Specific resources to restore
}

// ValidationReport contains results from DR validation
type ValidationReport struct {
	TestID           string
	Timestamp        time.Time
	Success          bool
	RTO              time.Duration
	RPO              time.Duration
	DataLoss         int64 // bytes
	FailoverTime     time.Duration
	RestoreTime      time.Duration
	Issues           []ValidationIssue
	RecommendedActions []string
}

// ValidationIssue represents a problem found during validation
type ValidationIssue struct {
	Severity    string // "critical", "high", "medium", "low"
	Component   string
	Description string
	Impact      string
	Remediation string
}

// RecoveryMetrics tracks DR performance
type RecoveryMetrics struct {
	RTO                 time.Duration
	RPO                 time.Duration
	MTTR                time.Duration // Mean Time To Recovery
	MTBF                time.Duration // Mean Time Between Failures
	BackupSuccessRate   float64
	RestoreSuccessRate  float64
	FailoverSuccessRate float64
	DataLossIncidents   int64
	LastIncident        time.Time
}

// FencingMechanism defines how to fence failed nodes
type FencingMechanism struct {
	Type     string // "STONITH", "network", "disk"
	Enabled  bool
	Target   string
	Metadata map[string]string
}

// RunbookExecution tracks runbook execution
type RunbookExecution struct {
	ID              string
	RunbookID       string
	StartedAt       time.Time
	CompletedAt     time.Time
	Status          string // "running", "completed", "failed", "aborted"
	Steps           []RunbookStep
	ApprovalsPending int
	ExecutedBy      string
	AuditLog        []AuditEntry
}

// RunbookStep represents a single step in a runbook
type RunbookStep struct {
	ID               string
	Name             string
	Description      string
	Status           string // "pending", "running", "completed", "failed", "skipped"
	RequiresApproval bool
	AutoRollback     bool
	StartedAt        time.Time
	CompletedAt      time.Time
	Error            string
}

// AuditEntry logs DR actions for compliance
type AuditEntry struct {
	Timestamp  time.Time
	Action     string
	User       string
	Resource   string
	Success    bool
	Details    map[string]interface{}
	IPAddress  string
	SessionID  string
}

// ChaosExperiment defines a chaos engineering test
type ChaosExperiment struct {
	ID               string
	Name             string
	Description      string
	TargetType       string // "pod", "node", "region", "network"
	TargetSelector   map[string]string
	FailureType      string // "kill", "latency", "packet_loss", "resource_exhaustion"
	Severity         int    // 1-10
	Duration         time.Duration
	BlastRadius      int    // Max affected resources
	SafetyChecks     []string
	AutoAbort        bool
	BusinessHoursOnly bool
	ScheduledTime    time.Time
}

// HealthCheck represents a health check configuration
type HealthCheck struct {
	Level       int    // L1-L4
	Name        string
	Endpoint    string
	Interval    time.Duration
	Timeout     time.Duration
	HealthyThreshold   int
	UnhealthyThreshold int
	ExpectedStatus     int
}

// RegionHealth tracks the health of a region
type RegionHealth struct {
	RegionID      string
	State         string // "healthy", "degraded", "failing", "failed"
	HealthScore   float64
	Capacity      float64
	Latency       time.Duration
	ErrorRate     float64
	LastCheck     time.Time
	FailureReason string
}
