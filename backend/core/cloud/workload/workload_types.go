package workload

// WorkloadType defines the classification of a workload
type WorkloadType string

const (
	// WebServer represents a typical web server workload
	WebServer WorkloadType = "web-server"

	// BatchProcessing represents batch jobs that run periodically
	BatchProcessing WorkloadType = "batch-processing"

	// DatabaseWorkload represents database server workloads
	DatabaseWorkload WorkloadType = "database"

	// MLTraining represents machine learning/AI training workloads
	MLTraining WorkloadType = "ml-training"

	// MLInference represents machine learning/AI inference workloads
	MLInference WorkloadType = "ml-inference"

	// AnalyticsWorkload represents data analytics workloads
	AnalyticsWorkload WorkloadType = "analytics"

	// DevTest represents development and testing environments
	DevTest WorkloadType = "dev-test"

	// GeneralPurpose represents general-purpose workloads
	GeneralPurpose WorkloadType = "general-purpose"
)

// WorkloadCharacteristics defines the resource usage characteristics of a workload
type WorkloadCharacteristics struct {
	// Type of the workload
	Type WorkloadType

	// CPU characteristics
	CPUIntensive bool
	CPUStability bool // true = stable, false = bursty

	// Memory characteristics
	MemoryIntensive bool
	MemoryStability bool // true = stable, false = variable

	// Storage characteristics
	IOIntensive bool
	IOPattern   IOPattern // read-heavy, write-heavy, balanced

	// Network characteristics
	NetworkIntensive bool
	NetworkPattern   NetworkPattern // inbound-heavy, outbound-heavy, balanced

	// Scheduling characteristics
	Interruptible bool // Can tolerate interruptions (good for spot/preemptible)
	TimeOfDay     []TimeWindow
	DayOfWeek     []int // 0 = Sunday, 6 = Saturday

	// Compliance requirements
	DataSovereignty       []string // List of required regions/countries
	ComplianceStandards   []string // Required compliance standards (e.g., "PCI-DSS", "HIPAA")
	NetworkIsolation      bool     // Requires network isolation
	DedicatedHardware     bool     // Requires dedicated hardware (not shared)
	EncryptionAtRest      bool     // Requires encryption at rest
	EncryptionInTransit   bool     // Requires encryption in transit
	BackupRequirements    bool     // Has specific backup requirements
	DisasterRecovery      bool     // Has disaster recovery requirements
	MultiRegionRedundancy bool     // Requires multi-region redundancy
}

// IOPattern represents IO access patterns
type IOPattern string

const (
	// ReadHeavy indicates predominantly read operations
	ReadHeavy IOPattern = "read-heavy"

	// WriteHeavy indicates predominantly write operations
	WriteHeavy IOPattern = "write-heavy"

	// BalancedIO indicates balanced read and write operations
	BalancedIO IOPattern = "balanced"

	// RandomAccess indicates random access patterns
	RandomAccess IOPattern = "random"

	// SequentialAccess indicates sequential access patterns
	SequentialAccess IOPattern = "sequential"
)

// NetworkPattern represents network traffic patterns
type NetworkPattern string

const (
	// InboundHeavy indicates predominantly inbound traffic
	InboundHeavy NetworkPattern = "inbound-heavy"

	// OutboundHeavy indicates predominantly outbound traffic
	OutboundHeavy NetworkPattern = "outbound-heavy"

	// BalancedNetwork indicates balanced inbound and outbound traffic
	BalancedNetwork NetworkPattern = "balanced"

	// HighlyInterconnected indicates high traffic between services
	HighlyInterconnected NetworkPattern = "highly-interconnected"

	// IsolatedNetwork indicates minimal traffic between services
	IsolatedNetwork NetworkPattern = "isolated"
)

// TimeWindow represents a time window for scheduling
type TimeWindow struct {
	StartHour int // 0-23
	EndHour   int // 0-23
}

// Metrics represents historical metrics for a workload
type Metrics struct {
	// CPU metrics
	AvgCPUUtilization    float64
	PeakCPUUtilization   float64
	CPUUtilizationP95    float64
	CPUUtilizationStdDev float64

	// Memory metrics
	AvgMemoryUtilization    float64
	PeakMemoryUtilization   float64
	MemoryUtilizationP95    float64
	MemoryUtilizationStdDev float64

	// IO metrics
	AvgIOPS              float64
	PeakIOPS             float64
	AvgThroughput        float64
	PeakThroughput       float64
	ReadWriteRatio       float64
	AvgLatency           float64
	P95Latency           float64
	IOOperationSize      float64
	RandomIOPercentage   float64
	SequentialIOBurstPct float64

	// Network metrics
	AvgNetworkIn                float64
	AvgNetworkOut               float64
	PeakNetworkIn               float64
	PeakNetworkOut              float64
	NetworkPacketsPerSecond     float64
	NetworkConnectionsPerSecond float64
	AvgActiveConnections        float64
	PeakActiveConnections       float64

	// Time-based patterns
	TimeOfDayPatterns    map[int]float64 // Hour (0-23) -> utilization
	DayOfWeekPatterns    map[int]float64 // Day (0-6) -> utilization
	WeeklyPatternQuality float64         // 0-1 indicator of how consistent the pattern is
}

// WorkloadProfile represents a complete workload profile including its characteristics and metrics
type WorkloadProfile struct {
	// Workload metadata
	ID          string
	Name        string
	Description string
	Tags        map[string]string

	// Resource requests/allocations
	RequestedCPU      int
	RequestedMemoryGB int
	RequestedDiskGB   int

	// Workload characteristics and metrics
	Characteristics WorkloadCharacteristics
	Metrics         Metrics

	// Cost data
	CurrentMonthlyCost float64
	TargetMonthlyCost  float64

	// Provider-specific optimization data
	ProviderFit map[string]ProviderFitScore
}

// ProviderFitScore represents how well a provider fits a workload
type ProviderFitScore struct {
	ProviderName         string
	OverallScore         float64 // 0-1, higher is better
	CostScore            float64 // 0-1, higher is better
	PerformanceScore     float64 // 0-1, higher is better
	ReliabilityScore     float64 // 0-1, higher is better
	ComplianceScore      float64 // 0-1, higher is better (0 = non-compliant)
	OptimalInstanceType  string
	EstimatedMonthlyCost float64
	RecommendedAction    string
	ReasonForScore       string
}
