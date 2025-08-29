package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MultiCloudVMSpec defines the desired state of MultiCloudVM
type MultiCloudVMSpec struct {
	// Multi-cloud deployment strategy
	DeploymentStrategy MultiCloudDeploymentStrategy `json:"deploymentStrategy"`

	// Provider-specific configurations
	Providers []CloudProvider `json:"providers"`

	// VM configuration
	VMTemplate VMTemplateRef `json:"vmTemplate"`

	// Cost optimization preferences
	CostOptimization *CostOptimizationPolicy `json:"costOptimization,omitempty"`

	// Disaster recovery configuration
	DisasterRecovery *DisasterRecoveryConfig `json:"disasterRecovery,omitempty"`

	// Migration policies across clouds
	MigrationPolicy *CrossCloudMigrationPolicy `json:"migrationPolicy,omitempty"`
}

// MultiCloudVMStatus defines the observed state of MultiCloudVM
type MultiCloudVMStatus struct {
	// Current deployment state across clouds
	CloudDeployments []CloudDeploymentStatus `json:"cloudDeployments,omitempty"`

	// Primary active cloud provider
	PrimaryProvider string `json:"primaryProvider,omitempty"`

	// Total cost across all providers
	TotalCost *ResourceCost `json:"totalCost,omitempty"`

	// Migration status
	Migration *CrossCloudMigrationStatus `json:"migration,omitempty"`

	// Overall health status
	Health MultiCloudHealthStatus `json:"health,omitempty"`

	// Conditions
	Conditions []MultiCloudVMCondition `json:"conditions,omitempty"`

	// Last observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// FederatedClusterSpec defines the desired state of FederatedCluster
type FederatedClusterSpec struct {
	// Cluster members across different clouds
	Members []ClusterMember `json:"members"`

	// Federation configuration
	Federation FederationConfig `json:"federation"`

	// Load balancing strategy across clusters
	LoadBalancing LoadBalancingStrategy `json:"loadBalancing,omitempty"`

	// Data replication policies
	DataReplication DataReplicationPolicy `json:"dataReplication,omitempty"`

	// Network policies for cross-cluster communication
	NetworkPolicy CrossClusterNetworkPolicy `json:"networkPolicy,omitempty"`
}

// FederatedClusterStatus defines the observed state of FederatedCluster
type FederatedClusterStatus struct {
	// Status of each cluster member
	MemberStatus []ClusterMemberStatus `json:"memberStatus,omitempty"`

	// Active members count
	ActiveMembers int32 `json:"activeMembers"`

	// Total capacity across all members
	TotalCapacity ResourceCapacity `json:"totalCapacity,omitempty"`

	// Federation health
	FederationHealth FederationHealthStatus `json:"federationHealth,omitempty"`

	// Conditions
	Conditions []FederatedClusterCondition `json:"conditions,omitempty"`

	// Last observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// AISchedulingPolicySpec defines AI-powered scheduling policies
type AISchedulingPolicySpec struct {
	// ML model configuration
	ModelConfig AIModelConfig `json:"modelConfig"`

	// Scheduling objectives
	Objectives []SchedulingObjective `json:"objectives"`

	// Historical data sources
	DataSources []DataSource `json:"dataSources,omitempty"`

	// Prediction windows
	PredictionWindow PredictionWindowConfig `json:"predictionWindow,omitempty"`

	// Feedback learning configuration
	LearningConfig LearningConfig `json:"learningConfig,omitempty"`
}

// AISchedulingPolicyStatus defines the observed state of AI scheduling
type AISchedulingPolicyStatus struct {
	// Model status
	ModelStatus AIModelStatus `json:"modelStatus,omitempty"`

	// Prediction accuracy metrics
	AccuracyMetrics AccuracyMetrics `json:"accuracyMetrics,omitempty"`

	// Recent decisions made by AI
	RecentDecisions []AISchedulingDecision `json:"recentDecisions,omitempty"`

	// Learning progress
	LearningProgress LearningProgress `json:"learningProgress,omitempty"`

	// Conditions
	Conditions []AISchedulingPolicyCondition `json:"conditions,omitempty"`

	// Last observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// CacheIntegrationSpec defines Redis cache integration
type CacheIntegrationSpec struct {
	// Redis cluster configuration
	RedisConfig RedisClusterConfig `json:"redisConfig"`

	// Caching strategy
	Strategy CacheStrategy `json:"strategy"`

	// TTL policies
	TTLPolicies []TTLPolicy `json:"ttlPolicies,omitempty"`

	// Cache warming configuration
	WarmingConfig CacheWarmingConfig `json:"warmingConfig,omitempty"`

	// Eviction policies
	EvictionPolicy CacheEvictionPolicy `json:"evictionPolicy,omitempty"`
}

// CacheIntegrationStatus defines the observed state of cache integration
type CacheIntegrationStatus struct {
	// Redis cluster health
	ClusterHealth RedisClusterHealth `json:"clusterHealth,omitempty"`

	// Cache performance metrics
	PerformanceMetrics CachePerformanceMetrics `json:"performanceMetrics,omitempty"`

	// Memory usage across cache nodes
	MemoryUsage CacheMemoryUsage `json:"memoryUsage,omitempty"`

	// Conditions
	Conditions []CacheIntegrationCondition `json:"conditions,omitempty"`

	// Last observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// Supporting types for Multi-Cloud functionality

type MultiCloudDeploymentStrategy struct {
	// Strategy type: active-passive, active-active, burst, cost-optimized
	Type string `json:"type"`

	// Primary cloud provider
	Primary string `json:"primary"`

	// Secondary providers for failover/burst
	Secondary []string `json:"secondary,omitempty"`

	// Failover triggers
	FailoverTriggers []FailoverTrigger `json:"failoverTriggers,omitempty"`
}

type CloudProvider struct {
	// Provider name (aws, azure, gcp, etc.)
	Name string `json:"name"`

	// Region for deployment
	Region string `json:"region"`

	// Credentials reference
	CredentialsSecret string `json:"credentialsSecret"`

	// Provider-specific configuration
	Config map[string]string `json:"config,omitempty"`

	// Cost preferences
	CostTier string `json:"costTier,omitempty"`

	// Availability requirements
	AvailabilityZones []string `json:"availabilityZones,omitempty"`
}

type CostOptimizationPolicy struct {
	// Enable spot/preemptible instances
	UseSpotInstances bool `json:"useSpotInstances,omitempty"`

	// Maximum cost per hour
	MaxCostPerHour string `json:"maxCostPerHour,omitempty"`

	// Cost alerts configuration
	CostAlerts []CostAlert `json:"costAlerts,omitempty"`

	// Auto-scaling based on cost
	CostBasedScaling bool `json:"costBasedScaling,omitempty"`
}

type DisasterRecoveryConfig struct {
	// Enable disaster recovery
	Enabled bool `json:"enabled"`

	// Recovery time objective (RTO)
	RTO string `json:"rto,omitempty"`

	// Recovery point objective (RPO)  
	RPO string `json:"rpo,omitempty"`

	// Backup strategy
	BackupStrategy BackupStrategy `json:"backupStrategy,omitempty"`
}

type CrossCloudMigrationPolicy struct {
	// Migration triggers
	Triggers []MigrationTrigger `json:"triggers,omitempty"`

	// Migration type (live, warm, cold)
	Type string `json:"type,omitempty"`

	// Cost-based migration thresholds
	CostThresholds CostThresholds `json:"costThresholds,omitempty"`

	// Performance-based migration
	PerformanceThresholds PerformanceThresholds `json:"performanceThresholds,omitempty"`
}

type CloudDeploymentStatus struct {
	// Cloud provider name
	Provider string `json:"provider"`

	// Deployment status
	Status string `json:"status"`

	// VM instances in this cloud
	Instances []VMInstance `json:"instances,omitempty"`

	// Current cost
	Cost *ResourceCost `json:"cost,omitempty"`

	// Performance metrics
	Performance *PerformanceMetrics `json:"performance,omitempty"`

	// Last update time
	LastUpdated *metav1.Time `json:"lastUpdated,omitempty"`
}

type ResourceCost struct {
	// Currency (USD, EUR, etc.)
	Currency string `json:"currency"`

	// Cost per hour
	HourlyCost float64 `json:"hourlyCost"`

	// Total cost to date
	TotalCost float64 `json:"totalCost"`

	// Cost breakdown by resource type
	Breakdown map[string]float64 `json:"breakdown,omitempty"`
}

type CrossCloudMigrationStatus struct {
	// Source cloud
	SourceCloud string `json:"sourceCloud"`

	// Target cloud
	TargetCloud string `json:"targetCloud"`

	// Migration progress percentage
	Progress float64 `json:"progress"`

	// Migration type being performed
	Type string `json:"type"`

	// Start time
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// Estimated completion time
	EstimatedEndTime *metav1.Time `json:"estimatedEndTime,omitempty"`

	// Migration phases
	Phases []MigrationPhase `json:"phases,omitempty"`
}

type MultiCloudHealthStatus struct {
	// Overall health status
	Status string `json:"status"`

	// Health checks results
	Checks []HealthCheck `json:"checks,omitempty"`

	// Availability percentage
	Availability float64 `json:"availability,omitempty"`
}

// Federation types

type ClusterMember struct {
	// Cluster name
	Name string `json:"name"`

	// Cloud provider
	Provider string `json:"provider"`

	// Cluster endpoint
	Endpoint string `json:"endpoint"`

	// Credentials for accessing cluster
	CredentialsSecret string `json:"credentialsSecret"`

	// Weight for load balancing
	Weight int32 `json:"weight,omitempty"`

	// Capacity information
	Capacity ResourceCapacity `json:"capacity,omitempty"`
}

type FederationConfig struct {
	// Federation strategy
	Strategy string `json:"strategy"`

	// Consensus algorithm
	Consensus string `json:"consensus,omitempty"`

	// Heartbeat interval
	HeartbeatInterval string `json:"heartbeatInterval,omitempty"`

	// Failure detection timeout
	FailureTimeout string `json:"failureTimeout,omitempty"`
}

type ClusterMemberStatus struct {
	// Member name
	Name string `json:"name"`

	// Connection status
	Status string `json:"status"`

	// Available capacity
	AvailableCapacity ResourceCapacity `json:"availableCapacity,omitempty"`

	// Current workload
	CurrentWorkload int32 `json:"currentWorkload,omitempty"`

	// Last heartbeat
	LastHeartbeat *metav1.Time `json:"lastHeartbeat,omitempty"`
}

type ResourceCapacity struct {
	// CPU capacity (cores)
	CPU string `json:"cpu"`

	// Memory capacity 
	Memory string `json:"memory"`

	// Storage capacity
	Storage string `json:"storage"`

	// GPU capacity (if applicable)
	GPU string `json:"gpu,omitempty"`
}

type FederationHealthStatus struct {
	// Overall federation health
	Status string `json:"status"`

	// Consensus health
	Consensus string `json:"consensus"`

	// Network connectivity health
	NetworkHealth string `json:"networkHealth"`

	// Data synchronization status
	SyncStatus string `json:"syncStatus"`
}

// AI Scheduling types

type AIModelConfig struct {
	// Model type (neural-network, decision-tree, reinforcement-learning)
	ModelType string `json:"modelType"`

	// Model version
	Version string `json:"version,omitempty"`

	// Training configuration
	TrainingConfig TrainingConfig `json:"trainingConfig,omitempty"`

	// Model parameters
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

type SchedulingObjective struct {
	// Objective type (cost, performance, availability, energy)
	Type string `json:"type"`

	// Objective weight (0.0 to 1.0)
	Weight float64 `json:"weight"`

	// Target value
	Target interface{} `json:"target,omitempty"`

	// Constraints
	Constraints []Constraint `json:"constraints,omitempty"`
}

type DataSource struct {
	// Source type (prometheus, influxdb, cloudwatch)
	Type string `json:"type"`

	// Connection configuration
	Connection map[string]string `json:"connection"`

	// Metrics to collect
	Metrics []string `json:"metrics"`

	// Collection interval
	Interval string `json:"interval,omitempty"`
}

type PredictionWindowConfig struct {
	// Short-term prediction window
	ShortTerm string `json:"shortTerm,omitempty"`

	// Long-term prediction window  
	LongTerm string `json:"longTerm,omitempty"`

	// Prediction granularity
	Granularity string `json:"granularity,omitempty"`
}

type LearningConfig struct {
	// Enable online learning
	OnlineLearning bool `json:"onlineLearning,omitempty"`

	// Learning rate
	LearningRate float64 `json:"learningRate,omitempty"`

	// Batch size for training
	BatchSize int32 `json:"batchSize,omitempty"`

	// Model retraining interval
	RetrainingInterval string `json:"retrainingInterval,omitempty"`
}

type AIModelStatus struct {
	// Model state (training, ready, error)
	State string `json:"state"`

	// Model accuracy
	Accuracy float64 `json:"accuracy,omitempty"`

	// Last training time
	LastTraining *metav1.Time `json:"lastTraining,omitempty"`

	// Training progress
	TrainingProgress float64 `json:"trainingProgress,omitempty"`
}

type AccuracyMetrics struct {
	// Prediction accuracy over different time windows
	ShortTermAccuracy float64 `json:"shortTermAccuracy,omitempty"`
	LongTermAccuracy  float64 `json:"longTermAccuracy,omitempty"`

	// Accuracy by objective type
	AccuracyByObjective map[string]float64 `json:"accuracyByObjective,omitempty"`
}

type AISchedulingDecision struct {
	// Decision timestamp
	Timestamp *metav1.Time `json:"timestamp"`

	// VM or workload identifier
	WorkloadID string `json:"workloadId"`

	// Chosen placement
	Placement PlacementDecision `json:"placement"`

	// Confidence score
	Confidence float64 `json:"confidence"`

	// Reasoning
	Reasoning string `json:"reasoning,omitempty"`
}

type PlacementDecision struct {
	// Target node or cluster
	Target string `json:"target"`

	// Target cloud provider
	Provider string `json:"provider,omitempty"`

	// Resource allocation
	Resources ResourceCapacity `json:"resources"`

	// Expected performance
	ExpectedPerformance map[string]interface{} `json:"expectedPerformance,omitempty"`
}

type LearningProgress struct {
	// Training iterations completed
	Iterations int64 `json:"iterations"`

	// Loss function value
	Loss float64 `json:"loss,omitempty"`

	// Validation accuracy
	ValidationAccuracy float64 `json:"validationAccuracy,omitempty"`
}

// Cache Integration types

type RedisClusterConfig struct {
	// Redis endpoints
	Endpoints []string `json:"endpoints"`

	// Credentials secret
	CredentialsSecret string `json:"credentialsSecret,omitempty"`

	// High availability configuration
	HA RedisHAConfig `json:"ha,omitempty"`

	// Security configuration
	Security RedisSecurityConfig `json:"security,omitempty"`
}

type CacheStrategy struct {
	// Strategy type (write-through, write-behind, read-through)
	Type string `json:"type"`

	// Cache levels
	Levels []CacheLevel `json:"levels,omitempty"`

	// Consistency level
	Consistency string `json:"consistency,omitempty"`
}

type TTLPolicy struct {
	// Data type pattern
	Pattern string `json:"pattern"`

	// Time to live
	TTL string `json:"ttl"`

	// Refresh strategy
	RefreshStrategy string `json:"refreshStrategy,omitempty"`
}

type CacheWarmingConfig struct {
	// Enable cache warming
	Enabled bool `json:"enabled"`

	// Warming strategies
	Strategies []WarmingStrategy `json:"strategies,omitempty"`

	// Schedule for warming
	Schedule string `json:"schedule,omitempty"`
}

type CacheEvictionPolicy struct {
	// Eviction algorithm (LRU, LFU, FIFO)
	Algorithm string `json:"algorithm"`

	// Memory thresholds
	MemoryThresholds MemoryThresholds `json:"memoryThresholds,omitempty"`
}

type RedisClusterHealth struct {
	// Cluster status
	Status string `json:"status"`

	// Node status
	Nodes []RedisNodeStatus `json:"nodes,omitempty"`

	// Replication health
	Replication string `json:"replication,omitempty"`
}

type CachePerformanceMetrics struct {
	// Hit rate
	HitRate float64 `json:"hitRate"`

	// Miss rate
	MissRate float64 `json:"missRate"`

	// Average response time
	ResponseTime string `json:"responseTime"`

	// Throughput (operations per second)
	Throughput float64 `json:"throughput"`
}

type CacheMemoryUsage struct {
	// Total memory
	TotalMemory string `json:"totalMemory"`

	// Used memory
	UsedMemory string `json:"usedMemory"`

	// Memory usage percentage
	UsagePercentage float64 `json:"usagePercentage"`

	// Memory usage by cache level
	UsageByLevel map[string]string `json:"usageByLevel,omitempty"`
}

// Additional supporting types

type FailoverTrigger struct {
	Type      string `json:"type"`
	Threshold string `json:"threshold,omitempty"`
	Duration  string `json:"duration,omitempty"`
}

type CostAlert struct {
	Threshold   string `json:"threshold"`
	Type        string `json:"type"`
	Frequency   string `json:"frequency,omitempty"`
	Destination string `json:"destination"`
}

type BackupStrategy struct {
	Type        string   `json:"type"`
	Schedule    string   `json:"schedule,omitempty"`
	Retention   string   `json:"retention,omitempty"`
	Destinations []string `json:"destinations,omitempty"`
}

type CostThresholds struct {
	MigrationCostThreshold string `json:"migrationCostThreshold,omitempty"`
	MaxAcceptableCost      string `json:"maxAcceptableCost,omitempty"`
}

type PerformanceThresholds struct {
	CPUUtilization    float64 `json:"cpuUtilization,omitempty"`
	MemoryUtilization float64 `json:"memoryUtilization,omitempty"`
	NetworkLatency    string  `json:"networkLatency,omitempty"`
}

type VMInstance struct {
	ID       string `json:"id"`
	Status   string `json:"status"`
	NodeID   string `json:"nodeId,omitempty"`
	IP       string `json:"ip,omitempty"`
}

type PerformanceMetrics struct {
	CPU     float64 `json:"cpu"`
	Memory  float64 `json:"memory"`
	Network NetworkMetrics `json:"network,omitempty"`
	Storage StorageMetrics `json:"storage,omitempty"`
}

type NetworkMetrics struct {
	Latency   string  `json:"latency,omitempty"`
	Bandwidth string  `json:"bandwidth,omitempty"`
	PacketLoss float64 `json:"packetLoss,omitempty"`
}

type StorageMetrics struct {
	IOPS      float64 `json:"iops,omitempty"`
	Latency   string  `json:"latency,omitempty"`
	Throughput string `json:"throughput,omitempty"`
}

type MigrationPhase struct {
	Name        string       `json:"name"`
	Status      string       `json:"status"`
	Progress    float64      `json:"progress"`
	StartTime   *metav1.Time `json:"startTime,omitempty"`
	EndTime     *metav1.Time `json:"endTime,omitempty"`
	Description string       `json:"description,omitempty"`
}

type HealthCheck struct {
	Name      string       `json:"name"`
	Status    string       `json:"status"`
	Message   string       `json:"message,omitempty"`
	Timestamp *metav1.Time `json:"timestamp,omitempty"`
}

type LoadBalancingStrategy struct {
	Algorithm string            `json:"algorithm"`
	Weights   map[string]int32  `json:"weights,omitempty"`
	Config    map[string]string `json:"config,omitempty"`
}

type DataReplicationPolicy struct {
	Strategy    string   `json:"strategy"`
	Replicas    int32    `json:"replicas"`
	Consistency string   `json:"consistency,omitempty"`
	Regions     []string `json:"regions,omitempty"`
}

type CrossClusterNetworkPolicy struct {
	Encryption  bool                      `json:"encryption,omitempty"`
	QoSPolicy   QoSPolicy                `json:"qosPolicy,omitempty"`
	Firewall    FirewallRules            `json:"firewall,omitempty"`
	VPN         VPNConfiguration         `json:"vpn,omitempty"`
}

type QoSPolicy struct {
	BandwidthLimits map[string]string `json:"bandwidthLimits,omitempty"`
	PriorityClass   string            `json:"priorityClass,omitempty"`
}

type FirewallRules struct {
	IngressRules []FirewallRule `json:"ingressRules,omitempty"`
	EgressRules  []FirewallRule `json:"egressRules,omitempty"`
}

type FirewallRule struct {
	Port     int32  `json:"port"`
	Protocol string `json:"protocol"`
	Source   string `json:"source,omitempty"`
}

type VPNConfiguration struct {
	Type       string            `json:"type"`
	Config     map[string]string `json:"config,omitempty"`
	Endpoints  []string          `json:"endpoints,omitempty"`
}

type TrainingConfig struct {
	DatasetSize     int64             `json:"datasetSize,omitempty"`
	ValidationSplit float64           `json:"validationSplit,omitempty"`
	Epochs          int32             `json:"epochs,omitempty"`
	Features        []string          `json:"features,omitempty"`
	Hyperparameters map[string]interface{} `json:"hyperparameters,omitempty"`
}

type Constraint struct {
	Name     string      `json:"name"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
	Weight   float64     `json:"weight,omitempty"`
}

type CacheLevel struct {
	Name     string `json:"name"`
	Size     string `json:"size"`
	TTL      string `json:"ttl,omitempty"`
	Strategy string `json:"strategy,omitempty"`
}

type WarmingStrategy struct {
	Type       string   `json:"type"`
	Patterns   []string `json:"patterns,omitempty"`
	Priority   int32    `json:"priority,omitempty"`
}

type MemoryThresholds struct {
	Warning   float64 `json:"warning,omitempty"`
	Critical  float64 `json:"critical,omitempty"`
	Eviction  float64 `json:"eviction,omitempty"`
}

type RedisHAConfig struct {
	SentinelEnabled bool     `json:"sentinelEnabled,omitempty"`
	Replicas        int32    `json:"replicas,omitempty"`
	SentinelHosts   []string `json:"sentinelHosts,omitempty"`
}

type RedisSecurityConfig struct {
	TLSEnabled    bool   `json:"tlsEnabled,omitempty"`
	AuthEnabled   bool   `json:"authEnabled,omitempty"`
	CertSecret    string `json:"certSecret,omitempty"`
}

type RedisNodeStatus struct {
	NodeID   string `json:"nodeId"`
	Status   string `json:"status"`
	Role     string `json:"role"`
	Memory   string `json:"memory,omitempty"`
	LastSeen *metav1.Time `json:"lastSeen,omitempty"`
}

// Condition types

type MultiCloudVMCondition struct {
	Type               MultiCloudVMConditionType `json:"type"`
	Status             metav1.ConditionStatus    `json:"status"`
	LastTransitionTime metav1.Time               `json:"lastTransitionTime"`
	Reason             string                    `json:"reason,omitempty"`
	Message            string                    `json:"message,omitempty"`
}

type FederatedClusterCondition struct {
	Type               FederatedClusterConditionType `json:"type"`
	Status             metav1.ConditionStatus        `json:"status"`
	LastTransitionTime metav1.Time                   `json:"lastTransitionTime"`
	Reason             string                        `json:"reason,omitempty"`
	Message            string                        `json:"message,omitempty"`
}

type AISchedulingPolicyCondition struct {
	Type               AISchedulingPolicyConditionType `json:"type"`
	Status             metav1.ConditionStatus          `json:"status"`
	LastTransitionTime metav1.Time                     `json:"lastTransitionTime"`
	Reason             string                          `json:"reason,omitempty"`
	Message            string                          `json:"message,omitempty"`
}

type CacheIntegrationCondition struct {
	Type               CacheIntegrationConditionType `json:"type"`
	Status             metav1.ConditionStatus        `json:"status"`
	LastTransitionTime metav1.Time                   `json:"lastTransitionTime"`
	Reason             string                        `json:"reason,omitempty"`
	Message            string                        `json:"message,omitempty"`
}

// Condition type enums

type MultiCloudVMConditionType string

const (
	MultiCloudVMReady             MultiCloudVMConditionType = "Ready"
	MultiCloudVMDeployed          MultiCloudVMConditionType = "Deployed"
	MultiCloudVMMigrating         MultiCloudVMConditionType = "Migrating"
	MultiCloudVMCostOptimized     MultiCloudVMConditionType = "CostOptimized"
)

type FederatedClusterConditionType string

const (
	FederatedClusterReady      FederatedClusterConditionType = "Ready"
	FederatedClusterSynced     FederatedClusterConditionType = "Synced"
	FederatedClusterHealthy    FederatedClusterConditionType = "Healthy"
)

type AISchedulingPolicyConditionType string

const (
	AISchedulingPolicyReady     AISchedulingPolicyConditionType = "Ready"
	AISchedulingPolicyTraining  AISchedulingPolicyConditionType = "Training"
	AISchedulingPolicyActive    AISchedulingPolicyConditionType = "Active"
)

type CacheIntegrationConditionType string

const (
	CacheIntegrationReady       CacheIntegrationConditionType = "Ready"
	CacheIntegrationHealthy     CacheIntegrationConditionType = "Healthy"
	CacheIntegrationPerforming  CacheIntegrationConditionType = "Performing"
)

// Root resource definitions

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Strategy",type="string",JSONPath=".spec.deploymentStrategy.type"
// +kubebuilder:printcolumn:name="Primary",type="string",JSONPath=".status.primaryProvider"
// +kubebuilder:printcolumn:name="Cost",type="string",JSONPath=".status.totalCost.hourlyCost"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// MultiCloudVM represents a VM deployed across multiple cloud providers
type MultiCloudVM struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   MultiCloudVMSpec   `json:"spec,omitempty"`
	Status MultiCloudVMStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// MultiCloudVMList contains a list of MultiCloudVM
type MultiCloudVMList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []MultiCloudVM `json:"items"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Strategy",type="string",JSONPath=".spec.federation.strategy"
// +kubebuilder:printcolumn:name="Members",type="integer",JSONPath=".status.activeMembers"
// +kubebuilder:printcolumn:name="Health",type="string",JSONPath=".status.federationHealth.status"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// FederatedCluster represents a cluster spanning multiple cloud providers
type FederatedCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   FederatedClusterSpec   `json:"spec,omitempty"`
	Status FederatedClusterStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// FederatedClusterList contains a list of FederatedCluster
type FederatedClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []FederatedCluster `json:"items"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Model",type="string",JSONPath=".spec.modelConfig.modelType"
// +kubebuilder:printcolumn:name="Status",type="string",JSONPath=".status.modelStatus.state"
// +kubebuilder:printcolumn:name="Accuracy",type="string",JSONPath=".status.accuracyMetrics.shortTermAccuracy"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// AISchedulingPolicy represents AI-powered scheduling policies
type AISchedulingPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AISchedulingPolicySpec   `json:"spec,omitempty"`
	Status AISchedulingPolicyStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// AISchedulingPolicyList contains a list of AISchedulingPolicy
type AISchedulingPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AISchedulingPolicy `json:"items"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Strategy",type="string",JSONPath=".spec.strategy.type"
// +kubebuilder:printcolumn:name="Hit Rate",type="string",JSONPath=".status.performanceMetrics.hitRate"
// +kubebuilder:printcolumn:name="Health",type="string",JSONPath=".status.clusterHealth.status"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// CacheIntegration represents Redis cache integration configuration
type CacheIntegration struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CacheIntegrationSpec   `json:"spec,omitempty"`
	Status CacheIntegrationStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// CacheIntegrationList contains a list of CacheIntegration
type CacheIntegrationList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CacheIntegration `json:"items"`
}

