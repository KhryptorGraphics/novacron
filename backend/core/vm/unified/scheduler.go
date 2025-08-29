package unified

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"novacron/backend/core/vm"
	"novacron/backend/core/vm/kata"
)

// UnifiedScheduler manages scheduling for both VMs and containers
type UnifiedScheduler struct {
	// Node management
	nodes         map[string]*SchedulingNode
	nodesMutex    sync.RWMutex
	
	// Workload management
	workloads     map[string]*ScheduledWorkload
	workloadsMutex sync.RWMutex
	
	// Scheduling policies
	policies      map[string]SchedulingPolicy
	defaultPolicy string
	
	// AI integration
	aiClient      AISchedulingClient
	
	// Metrics and monitoring
	metrics       *SchedulingMetrics
	
	// Configuration
	config        *SchedulerConfig
}

// SchedulingNode represents a node that can host workloads
type SchedulingNode struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Type          NodeType              `json:"type"`
	Location      NodeLocation          `json:"location"`
	Capabilities  NodeCapabilities      `json:"capabilities"`
	Resources     NodeResources         `json:"resources"`
	Workloads     map[string]*ScheduledWorkload `json:"workloads"`
	Status        NodeStatus            `json:"status"`
	Labels        map[string]string     `json:"labels"`
	Taints        []NodeTaint           `json:"taints"`
	LastHeartbeat time.Time             `json:"last_heartbeat"`
	CreatedAt     time.Time             `json:"created_at"`
	UpdatedAt     time.Time             `json:"updated_at"`
}

type NodeType string

const (
	NodeTypeCloud    NodeType = "cloud"
	NodeTypeEdge     NodeType = "edge"
	NodeTypeHybrid   NodeType = "hybrid"
	NodeTypeGPU      NodeType = "gpu"
	NodeTypeStorage  NodeType = "storage"
)

type NodeLocation struct {
	Region           string  `json:"region"`
	Zone             string  `json:"zone"`
	CloudProvider    string  `json:"cloud_provider"`
	EdgeCluster      string  `json:"edge_cluster,omitempty"`
	Latitude         float64 `json:"latitude,omitempty"`
	Longitude        float64 `json:"longitude,omitempty"`
	NetworkLatencyMS int     `json:"network_latency_ms"`
}

type NodeCapabilities struct {
	SupportedWorkloadTypes []WorkloadType    `json:"supported_workload_types"`
	VMHypervisors         []string          `json:"vm_hypervisors"`
	ContainerRuntimes     []string          `json:"container_runtimes"`
	GPUSupport            bool              `json:"gpu_support"`
	GPUTypes              []string          `json:"gpu_types"`
	NetworkingFeatures    []string          `json:"networking_features"`
	StorageTypes          []string          `json:"storage_types"`
	SecurityFeatures      []string          `json:"security_features"`
}

type NodeResources struct {
	// Compute resources
	CPUCores           int     `json:"cpu_cores"`
	CPUUtilization     float64 `json:"cpu_utilization"`
	MemoryMB           int64   `json:"memory_mb"`
	MemoryUtilization  float64 `json:"memory_utilization"`
	
	// GPU resources
	GPUCount           int     `json:"gpu_count"`
	GPUMemoryMB        int64   `json:"gpu_memory_mb"`
	GPUUtilization     float64 `json:"gpu_utilization"`
	
	// Storage resources
	StorageGB          int64   `json:"storage_gb"`
	StorageUtilization float64 `json:"storage_utilization"`
	StorageIOPS        int     `json:"storage_iops"`
	
	// Network resources
	NetworkBandwidthMbps int     `json:"network_bandwidth_mbps"`
	NetworkUtilization   float64 `json:"network_utilization"`
	
	// Capacity tracking
	AllocatedCPU       float64 `json:"allocated_cpu"`
	AllocatedMemory    int64   `json:"allocated_memory"`
	AllocatedGPU       float64 `json:"allocated_gpu"`
	AllocatedStorage   int64   `json:"allocated_storage"`
	AllocatedBandwidth int     `json:"allocated_bandwidth"`
}

type NodeStatus struct {
	State              NodeState `json:"state"`
	Ready              bool      `json:"ready"`
	SchedulingEnabled  bool      `json:"scheduling_enabled"`
	HealthScore        float64   `json:"health_score"`
	LastHealthCheck    time.Time `json:"last_health_check"`
	MaintenanceWindow  *MaintenanceWindow `json:"maintenance_window,omitempty"`
}

type NodeState string

const (
	NodeStateReady       NodeState = "ready"
	NodeStateNotReady    NodeState = "not_ready"
	NodeStateMaintenance NodeState = "maintenance"
	NodeStateUnknown     NodeState = "unknown"
)

type NodeTaint struct {
	Key    string      `json:"key"`
	Value  string      `json:"value"`
	Effect TaintEffect `json:"effect"`
}

type TaintEffect string

const (
	TaintEffectNoSchedule       TaintEffect = "NoSchedule"
	TaintEffectPreferNoSchedule TaintEffect = "PreferNoSchedule"
	TaintEffectNoExecute        TaintEffect = "NoExecute"
)

type MaintenanceWindow struct {
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	Reason    string    `json:"reason"`
}

// ScheduledWorkload represents a workload scheduled on the system
type ScheduledWorkload struct {
	ID             string                 `json:"id"`
	Name           string                 `json:"name"`
	Type           WorkloadType          `json:"type"`
	Spec           WorkloadSpec          `json:"spec"`
	Placement      WorkloadPlacement     `json:"placement"`
	Status         WorkloadStatus        `json:"status"`
	Requirements   WorkloadRequirements  `json:"requirements"`
	Preferences    WorkloadPreferences   `json:"preferences"`
	Policies       []string              `json:"policies"`
	Labels         map[string]string     `json:"labels"`
	Annotations    map[string]string     `json:"annotations"`
	CreatedAt      time.Time             `json:"created_at"`
	ScheduledAt    *time.Time            `json:"scheduled_at,omitempty"`
	StartedAt      *time.Time            `json:"started_at,omitempty"`
	CompletedAt    *time.Time            `json:"completed_at,omitempty"`
}

type WorkloadType string

const (
	WorkloadTypeVM        WorkloadType = "vm"
	WorkloadTypeContainer WorkloadType = "container"
	WorkloadTypeFunction  WorkloadType = "function"
	WorkloadTypeBatch     WorkloadType = "batch"
)

type WorkloadSpec struct {
	// Common properties
	Image       string            `json:"image,omitempty"`
	Command     []string          `json:"command,omitempty"`
	Args        []string          `json:"args,omitempty"`
	Environment map[string]string `json:"environment,omitempty"`
	
	// VM-specific properties
	VMSpec      *vm.VMSpec        `json:"vm_spec,omitempty"`
	
	// Container-specific properties
	KataSpec    *kata.KataConfig  `json:"kata_spec,omitempty"`
	
	// Resource requests
	Resources   ResourceRequests  `json:"resources"`
	
	// Networking
	NetworkMode string            `json:"network_mode,omitempty"`
	Ports       []PortMapping     `json:"ports,omitempty"`
	
	// Storage
	Volumes     []VolumeMount     `json:"volumes,omitempty"`
}

type ResourceRequests struct {
	CPUCores      float64 `json:"cpu_cores"`
	MemoryMB      int64   `json:"memory_mb"`
	GPUCount      int     `json:"gpu_count,omitempty"`
	GPUMemoryMB   int64   `json:"gpu_memory_mb,omitempty"`
	StorageGB     int64   `json:"storage_gb,omitempty"`
	BandwidthMbps int     `json:"bandwidth_mbps,omitempty"`
}

type PortMapping struct {
	ContainerPort int    `json:"container_port"`
	HostPort      int    `json:"host_port,omitempty"`
	Protocol      string `json:"protocol"`
}

type VolumeMount struct {
	Source      string `json:"source"`
	Destination string `json:"destination"`
	ReadOnly    bool   `json:"read_only"`
}

type WorkloadPlacement struct {
	NodeID            string                 `json:"node_id"`
	NodeName          string                 `json:"node_name"`
	PlacementScore    float64               `json:"placement_score"`
	PlacementReason   string                `json:"placement_reason"`
	SchedulingLatency time.Duration         `json:"scheduling_latency"`
	Constraints       []PlacementConstraint `json:"constraints"`
}

type PlacementConstraint struct {
	Type        ConstraintType `json:"type"`
	Key         string         `json:"key"`
	Operator    string         `json:"operator"`
	Values      []string       `json:"values"`
	Weight      int            `json:"weight"`
}

type ConstraintType string

const (
	ConstraintTypeNodeAffinity    ConstraintType = "node_affinity"
	ConstraintTypeAntiAffinity    ConstraintType = "anti_affinity"
	ConstraintTypeTopologySpread  ConstraintType = "topology_spread"
	ConstraintTypeResource        ConstraintType = "resource"
)

type WorkloadStatus struct {
	Phase         WorkloadPhase `json:"phase"`
	Ready         bool          `json:"ready"`
	Message       string        `json:"message,omitempty"`
	LastUpdate    time.Time     `json:"last_update"`
	RestartCount  int           `json:"restart_count"`
}

type WorkloadPhase string

const (
	WorkloadPhasePending   WorkloadPhase = "pending"
	WorkloadPhaseScheduled WorkloadPhase = "scheduled"
	WorkloadPhaseRunning   WorkloadPhase = "running"
	WorkloadPhaseSucceeded WorkloadPhase = "succeeded"
	WorkloadPhaseFailed    WorkloadPhase = "failed"
	WorkloadPhaseUnknown   WorkloadPhase = "unknown"
)

type WorkloadRequirements struct {
	// Hard requirements (must be met)
	MinCPU          float64                 `json:"min_cpu"`
	MinMemory       int64                   `json:"min_memory"`
	RequiredLabels  map[string]string       `json:"required_labels,omitempty"`
	Tolerations     []Toleration            `json:"tolerations,omitempty"`
	
	// Compliance requirements
	DataResidency   []string                `json:"data_residency,omitempty"`
	SecurityLevel   string                  `json:"security_level,omitempty"`
}

type WorkloadPreferences struct {
	// Soft preferences (scoring factors)
	PreferredZones     []string            `json:"preferred_zones,omitempty"`
	PreferredNodes     []string            `json:"preferred_nodes,omitempty"`
	AvoidNodes         []string            `json:"avoid_nodes,omitempty"`
	CostOptimization   bool                `json:"cost_optimization"`
	PerformanceFirst   bool                `json:"performance_first"`
	NetworkLocality    bool                `json:"network_locality"`
}

type Toleration struct {
	Key               string        `json:"key"`
	Operator          string        `json:"operator"`
	Value             string        `json:"value,omitempty"`
	Effect            TaintEffect   `json:"effect"`
	TolerationSeconds *int64        `json:"toleration_seconds,omitempty"`
}

// SchedulingPolicy defines how scheduling decisions are made
type SchedulingPolicy interface {
	Name() string
	Score(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (float64, error)
	Filter(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (bool, error)
	Priority() int
}

// AI client interface for intelligent scheduling
type AISchedulingClient interface {
	PredictOptimalPlacement(ctx context.Context, workload *ScheduledWorkload, nodes []*SchedulingNode) (*PlacementPrediction, error)
	OptimizeResourceAllocation(ctx context.Context, nodeID string) (*ResourceOptimization, error)
	PredictWorkloadDemand(ctx context.Context, timeHorizon time.Duration) (*DemandPrediction, error)
}

type PlacementPrediction struct {
	RecommendedNodeID string             `json:"recommended_node_id"`
	Confidence        float64            `json:"confidence"`
	Reasoning         []string           `json:"reasoning"`
	AlternativeNodes  []NodeRecommendation `json:"alternative_nodes"`
	PredictedMetrics  PredictedMetrics   `json:"predicted_metrics"`
}

type NodeRecommendation struct {
	NodeID     string  `json:"node_id"`
	Score      float64 `json:"score"`
	Confidence float64 `json:"confidence"`
	Reason     string  `json:"reason"`
}

type PredictedMetrics struct {
	ExpectedCPUUtilization    float64 `json:"expected_cpu_utilization"`
	ExpectedMemoryUtilization float64 `json:"expected_memory_utilization"`
	PredictedLatencyMS        float64 `json:"predicted_latency_ms"`
	CostEstimate              float64 `json:"cost_estimate"`
}

type ResourceOptimization struct {
	NodeID              string                    `json:"node_id"`
	Recommendations     []OptimizationRecommendation `json:"recommendations"`
	ExpectedImprovement map[string]float64        `json:"expected_improvement"`
}

type OptimizationRecommendation struct {
	Type        string      `json:"type"`
	Action      string      `json:"action"`
	Target      string      `json:"target"`
	Parameters  interface{} `json:"parameters"`
	Confidence  float64     `json:"confidence"`
	Impact      string      `json:"impact"`
}

type DemandPrediction struct {
	TimeHorizon        time.Duration              `json:"time_horizon"`
	PredictedWorkloads []PredictedWorkload       `json:"predicted_workloads"`
	ResourceDemand     map[string]ResourceDemand `json:"resource_demand"`
	Confidence         float64                   `json:"confidence"`
}

type PredictedWorkload struct {
	Type           WorkloadType     `json:"type"`
	Count          int              `json:"count"`
	Resources      ResourceRequests `json:"resources"`
	ArrivalTime    time.Time        `json:"arrival_time"`
	Duration       time.Duration    `json:"duration"`
	Priority       int              `json:"priority"`
}

type ResourceDemand struct {
	CPU       float64 `json:"cpu"`
	Memory    int64   `json:"memory"`
	GPU       int     `json:"gpu"`
	Storage   int64   `json:"storage"`
	Bandwidth int     `json:"bandwidth"`
}

type SchedulingMetrics struct {
	// Scheduling performance
	TotalSchedulingRequests      int64         `json:"total_scheduling_requests"`
	SuccessfulSchedulingRequests int64         `json:"successful_scheduling_requests"`
	FailedSchedulingRequests     int64         `json:"failed_scheduling_requests"`
	AverageSchedulingLatency     time.Duration `json:"average_scheduling_latency"`
	P95SchedulingLatency         time.Duration `json:"p95_scheduling_latency"`
	
	// Resource utilization
	OverallCPUUtilization        float64       `json:"overall_cpu_utilization"`
	OverallMemoryUtilization     float64       `json:"overall_memory_utilization"`
	OverallGPUUtilization        float64       `json:"overall_gpu_utilization"`
	
	// Node health
	TotalNodes                   int           `json:"total_nodes"`
	HealthyNodes                 int           `json:"healthy_nodes"`
	UnhealthyNodes               int           `json:"unhealthy_nodes"`
	
	// Workload distribution
	ActiveWorkloads              int           `json:"active_workloads"`
	WorkloadsByType              map[WorkloadType]int `json:"workloads_by_type"`
	WorkloadsByNode              map[string]int `json:"workloads_by_node"`
	
	// Performance metrics
	AISchedulingAccuracy         float64       `json:"ai_scheduling_accuracy"`
	PlacementOptimalityScore     float64       `json:"placement_optimality_score"`
	
	LastUpdate                   time.Time     `json:"last_update"`
}

type SchedulerConfig struct {
	// General settings
	SchedulingInterval          time.Duration `json:"scheduling_interval"`
	MaxPendingWorkloads         int           `json:"max_pending_workloads"`
	MaxConcurrentScheduling     int           `json:"max_concurrent_scheduling"`
	
	// AI integration
	EnableAIScheduling          bool          `json:"enable_ai_scheduling"`
	AIConfidenceThreshold       float64       `json:"ai_confidence_threshold"`
	AISchedulingEndpoint        string        `json:"ai_scheduling_endpoint"`
	
	// Node management
	NodeHealthCheckInterval     time.Duration `json:"node_health_check_interval"`
	NodeUnhealthyThreshold      time.Duration `json:"node_unhealthy_threshold"`
	
	// Scheduling policies
	EnableResourceFragmentation bool          `json:"enable_resource_fragmentation"`
	EnableTopologyAwareness     bool          `json:"enable_topology_awareness"`
	EnableCostOptimization      bool          `json:"enable_cost_optimization"`
	
	// Performance tuning
	EnablePreemption            bool          `json:"enable_preemption"`
	MaxPreemptionCandidates     int           `json:"max_preemption_candidates"`
	
	// Monitoring
	MetricsRetentionDays        int           `json:"metrics_retention_days"`
	DetailedMetrics             bool          `json:"detailed_metrics"`
}

// NewUnifiedScheduler creates a new unified scheduler
func NewUnifiedScheduler(config *SchedulerConfig, aiClient AISchedulingClient) *UnifiedScheduler {
	if config == nil {
		config = getDefaultSchedulerConfig()
	}
	
	scheduler := &UnifiedScheduler{
		nodes:         make(map[string]*SchedulingNode),
		workloads:     make(map[string]*ScheduledWorkload),
		policies:      make(map[string]SchedulingPolicy),
		defaultPolicy: "balanced",
		aiClient:      aiClient,
		config:        config,
		metrics:       &SchedulingMetrics{
			WorkloadsByType: make(map[WorkloadType]int),
			WorkloadsByNode: make(map[string]int),
		},
	}
	
	// Register default scheduling policies
	scheduler.registerDefaultPolicies()
	
	log.Printf("Unified scheduler initialized successfully")
	return scheduler
}

func getDefaultSchedulerConfig() *SchedulerConfig {
	return &SchedulerConfig{
		SchedulingInterval:          5 * time.Second,
		MaxPendingWorkloads:         1000,
		MaxConcurrentScheduling:     50,
		EnableAIScheduling:          true,
		AIConfidenceThreshold:       0.7,
		AISchedulingEndpoint:        "http://localhost:8093",
		NodeHealthCheckInterval:     30 * time.Second,
		NodeUnhealthyThreshold:      5 * time.Minute,
		EnableResourceFragmentation: true,
		EnableTopologyAwareness:     true,
		EnableCostOptimization:      true,
		EnablePreemption:            false,
		MaxPreemptionCandidates:     10,
		MetricsRetentionDays:        7,
		DetailedMetrics:             true,
	}
}

// Core scheduling methods
func (s *UnifiedScheduler) ScheduleWorkload(ctx context.Context, workload *ScheduledWorkload) error {
	startTime := time.Now()
	
	log.Printf("Scheduling workload %s (type: %s)", workload.ID, workload.Type)
	
	// Filter suitable nodes
	candidateNodes, err := s.filterNodes(ctx, workload)
	if err != nil {
		return fmt.Errorf("failed to filter nodes: %w", err)
	}
	
	if len(candidateNodes) == 0 {
		return fmt.Errorf("no suitable nodes found for workload %s", workload.ID)
	}
	
	// Score nodes
	scoredNodes, err := s.scoreNodes(ctx, workload, candidateNodes)
	if err != nil {
		return fmt.Errorf("failed to score nodes: %w", err)
	}
	
	// Use AI for optimal placement if enabled
	var selectedNode *SchedulingNode
	if s.config.EnableAIScheduling && s.aiClient != nil {
		selectedNode, err = s.aiAssistedPlacement(ctx, workload, scoredNodes)
		if err != nil {
			log.Printf("AI placement failed, falling back to scoring: %v", err)
			selectedNode = scoredNodes[0].Node
		}
	} else {
		selectedNode = scoredNodes[0].Node
	}
	
	// Place workload on selected node
	if err := s.placeWorkload(ctx, workload, selectedNode); err != nil {
		return fmt.Errorf("failed to place workload: %w", err)
	}
	
	// Update placement information
	workload.Placement = WorkloadPlacement{
		NodeID:            selectedNode.ID,
		NodeName:          selectedNode.Name,
		PlacementScore:    scoredNodes[0].Score,
		PlacementReason:   "unified scheduling",
		SchedulingLatency: time.Since(startTime),
	}
	
	workload.Status.Phase = WorkloadPhaseScheduled
	workload.Status.LastUpdate = time.Now()
	now := time.Now()
	workload.ScheduledAt = &now
	
	// Update metrics
	s.updateSchedulingMetrics(workload, time.Since(startTime), true)
	
	s.workloadsMutex.Lock()
	s.workloads[workload.ID] = workload
	s.workloadsMutex.Unlock()
	
	log.Printf("Workload %s scheduled on node %s in %v", 
		workload.ID, selectedNode.ID, time.Since(startTime))
	
	return nil
}

func (s *UnifiedScheduler) filterNodes(ctx context.Context, workload *ScheduledWorkload) ([]*SchedulingNode, error) {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()
	
	var candidates []*SchedulingNode
	
	for _, node := range s.nodes {
		// Check basic node health and readiness
		if !node.Status.Ready || !node.Status.SchedulingEnabled {
			continue
		}
		
		// Check resource availability
		if !s.hasAvailableResources(node, workload.Spec.Resources) {
			continue
		}
		
		// Check workload type support
		if !s.supportsWorkloadType(node, workload.Type) {
			continue
		}
		
		// Check taints and tolerations
		if !s.toleratesTaints(node, workload.Requirements.Tolerations) {
			continue
		}
		
		// Apply scheduling policies
		policyPassed := true
		for _, policy := range s.policies {
			if passed, err := policy.Filter(ctx, workload, node); err != nil {
				log.Printf("Policy %s filter error: %v", policy.Name(), err)
				continue
			} else if !passed {
				policyPassed = false
				break
			}
		}
		
		if policyPassed {
			candidates = append(candidates, node)
		}
	}
	
	return candidates, nil
}

type ScoredNode struct {
	Node  *SchedulingNode
	Score float64
	Details map[string]float64
}

func (s *UnifiedScheduler) scoreNodes(ctx context.Context, workload *ScheduledWorkload, nodes []*SchedulingNode) ([]*ScoredNode, error) {
	var scoredNodes []*ScoredNode
	
	for _, node := range nodes {
		totalScore := 0.0
		scoreDetails := make(map[string]float64)
		
		// Apply scoring policies
		for _, policy := range s.policies {
			score, err := policy.Score(ctx, workload, node)
			if err != nil {
				log.Printf("Policy %s scoring error: %v", policy.Name(), err)
				continue
			}
			
			// Weight by policy priority
			weightedScore := score * float64(policy.Priority())
			totalScore += weightedScore
			scoreDetails[policy.Name()] = weightedScore
		}
		
		scoredNodes = append(scoredNodes, &ScoredNode{
			Node:    node,
			Score:   totalScore,
			Details: scoreDetails,
		})
	}
	
	// Sort by score (highest first)
	sort.Slice(scoredNodes, func(i, j int) bool {
		return scoredNodes[i].Score > scoredNodes[j].Score
	})
	
	return scoredNodes, nil
}

func (s *UnifiedScheduler) aiAssistedPlacement(ctx context.Context, workload *ScheduledWorkload, scoredNodes []*ScoredNode) (*SchedulingNode, error) {
	nodes := make([]*SchedulingNode, len(scoredNodes))
	for i, sn := range scoredNodes {
		nodes[i] = sn.Node
	}
	
	prediction, err := s.aiClient.PredictOptimalPlacement(ctx, workload, nodes)
	if err != nil {
		return nil, err
	}
	
	if prediction.Confidence < s.config.AIConfidenceThreshold {
		return nil, fmt.Errorf("AI confidence %f below threshold %f", 
			prediction.Confidence, s.config.AIConfidenceThreshold)
	}
	
	// Find the recommended node
	for _, node := range nodes {
		if node.ID == prediction.RecommendedNodeID {
			return node, nil
		}
	}
	
	return nil, fmt.Errorf("AI recommended node %s not found in candidates", 
		prediction.RecommendedNodeID)
}

func (s *UnifiedScheduler) placeWorkload(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) error {
	// Reserve resources on the node
	s.nodesMutex.Lock()
	node.Resources.AllocatedCPU += workload.Spec.Resources.CPUCores
	node.Resources.AllocatedMemory += workload.Spec.Resources.MemoryMB * 1024 * 1024 // MB to bytes
	node.Resources.AllocatedGPU += float64(workload.Spec.Resources.GPUCount)
	node.Resources.AllocatedStorage += workload.Spec.Resources.StorageGB * 1024 * 1024 * 1024 // GB to bytes
	node.Resources.AllocatedBandwidth += workload.Spec.Resources.BandwidthMbps
	
	// Add workload to node
	if node.Workloads == nil {
		node.Workloads = make(map[string]*ScheduledWorkload)
	}
	node.Workloads[workload.ID] = workload
	
	node.UpdatedAt = time.Now()
	s.nodesMutex.Unlock()
	
	return nil
}

// Helper methods
func (s *UnifiedScheduler) hasAvailableResources(node *SchedulingNode, required ResourceRequests) bool {
	// Check CPU
	availableCPU := float64(node.Resources.CPUCores) - node.Resources.AllocatedCPU
	if availableCPU < required.CPUCores {
		return false
	}
	
	// Check memory (convert MB to bytes for comparison)
	availableMemory := node.Resources.MemoryMB*1024*1024 - node.Resources.AllocatedMemory
	if availableMemory < required.MemoryMB*1024*1024 {
		return false
	}
	
	// Check GPU if required
	if required.GPUCount > 0 {
		availableGPU := float64(node.Resources.GPUCount) - node.Resources.AllocatedGPU
		if availableGPU < float64(required.GPUCount) {
			return false
		}
	}
	
	// Check storage if required
	if required.StorageGB > 0 {
		availableStorage := node.Resources.StorageGB*1024*1024*1024 - node.Resources.AllocatedStorage
		if availableStorage < required.StorageGB*1024*1024*1024 {
			return false
		}
	}
	
	return true
}

func (s *UnifiedScheduler) supportsWorkloadType(node *SchedulingNode, workloadType WorkloadType) bool {
	for _, supportedType := range node.Capabilities.SupportedWorkloadTypes {
		if supportedType == workloadType {
			return true
		}
	}
	return false
}

func (s *UnifiedScheduler) toleratesTaints(node *SchedulingNode, tolerations []Toleration) bool {
	for _, taint := range node.Taints {
		tolerated := false
		for _, toleration := range tolerations {
			if s.matchesToleration(taint, toleration) {
				tolerated = true
				break
			}
		}
		if !tolerated && taint.Effect == TaintEffectNoSchedule {
			return false
		}
	}
	return true
}

func (s *UnifiedScheduler) matchesToleration(taint NodeTaint, toleration Toleration) bool {
	if toleration.Key != taint.Key {
		return false
	}
	
	if toleration.Effect != taint.Effect {
		return false
	}
	
	switch toleration.Operator {
	case "Equal":
		return toleration.Value == taint.Value
	case "Exists":
		return true
	default:
		return false
	}
}

func (s *UnifiedScheduler) updateSchedulingMetrics(workload *ScheduledWorkload, latency time.Duration, success bool) {
	s.metrics.TotalSchedulingRequests++
	if success {
		s.metrics.SuccessfulSchedulingRequests++
	} else {
		s.metrics.FailedSchedulingRequests++
	}
	
	// Update latency (simplified averaging)
	s.metrics.AverageSchedulingLatency = (s.metrics.AverageSchedulingLatency + latency) / 2
	
	// Update workload counts
	s.metrics.WorkloadsByType[workload.Type]++
	s.metrics.ActiveWorkloads++
	
	s.metrics.LastUpdate = time.Now()
}

// Default scheduling policies
func (s *UnifiedScheduler) registerDefaultPolicies() {
	s.policies["resource_balance"] = &ResourceBalancePolicy{priority: 10}
	s.policies["node_affinity"] = &NodeAffinityPolicy{priority: 8}
	s.policies["cost_optimization"] = &CostOptimizationPolicy{priority: 6}
	s.policies["topology_spread"] = &TopologySpreadPolicy{priority: 5}
}

// Example policy implementations
type ResourceBalancePolicy struct {
	priority int
}

func (p *ResourceBalancePolicy) Name() string { return "resource_balance" }
func (p *ResourceBalancePolicy) Priority() int { return p.priority }

func (p *ResourceBalancePolicy) Filter(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (bool, error) {
	// Basic resource availability already checked in main filter
	return true, nil
}

func (p *ResourceBalancePolicy) Score(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (float64, error) {
	// Score based on resource utilization balance
	cpuUtil := node.Resources.AllocatedCPU / float64(node.Resources.CPUCores)
	memUtil := float64(node.Resources.AllocatedMemory) / float64(node.Resources.MemoryMB*1024*1024)
	
	// Prefer nodes with lower utilization
	avgUtilization := (cpuUtil + memUtil) / 2
	score := 100.0 * (1.0 - avgUtilization)
	
	return math.Max(0, score), nil
}

type NodeAffinityPolicy struct {
	priority int
}

func (p *NodeAffinityPolicy) Name() string { return "node_affinity" }
func (p *NodeAffinityPolicy) Priority() int { return p.priority }

func (p *NodeAffinityPolicy) Filter(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (bool, error) {
	// Check required labels
	for key, value := range workload.Requirements.RequiredLabels {
		if nodeValue, exists := node.Labels[key]; !exists || nodeValue != value {
			return false, nil
		}
	}
	return true, nil
}

func (p *NodeAffinityPolicy) Score(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (float64, error) {
	score := 0.0
	
	// Boost score for preferred nodes
	for _, preferredNode := range workload.Preferences.PreferredNodes {
		if node.ID == preferredNode || node.Name == preferredNode {
			score += 50.0
		}
	}
	
	// Reduce score for avoided nodes
	for _, avoidNode := range workload.Preferences.AvoidNodes {
		if node.ID == avoidNode || node.Name == avoidNode {
			score -= 30.0
		}
	}
	
	return score, nil
}

type CostOptimizationPolicy struct {
	priority int
}

func (p *CostOptimizationPolicy) Name() string { return "cost_optimization" }
func (p *CostOptimizationPolicy) Priority() int { return p.priority }

func (p *CostOptimizationPolicy) Filter(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (bool, error) {
	return true, nil
}

func (p *CostOptimizationPolicy) Score(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (float64, error) {
	if !workload.Preferences.CostOptimization {
		return 0, nil
	}
	
	// Simplified cost scoring based on node type
	switch node.Type {
	case NodeTypeCloud:
		return 30.0, nil // Medium cost
	case NodeTypeEdge:
		return 60.0, nil // Lower cost
	case NodeTypeGPU:
		return 10.0, nil // Higher cost
	default:
		return 40.0, nil
	}
}

type TopologySpreadPolicy struct {
	priority int
}

func (p *TopologySpreadPolicy) Name() string { return "topology_spread" }
func (p *TopologySpreadPolicy) Priority() int { return p.priority }

func (p *TopologySpreadPolicy) Filter(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (bool, error) {
	return true, nil
}

func (p *TopologySpreadPolicy) Score(ctx context.Context, workload *ScheduledWorkload, node *SchedulingNode) (float64, error) {
	// Simple topology spread: prefer nodes with fewer workloads of the same type
	workloadCount := 0
	for _, w := range node.Workloads {
		if w.Type == workload.Type {
			workloadCount++
		}
	}
	
	// Higher score for nodes with fewer similar workloads
	return math.Max(0, 50.0-float64(workloadCount)*10.0), nil
}

// Public API methods
func (s *UnifiedScheduler) RegisterNode(node *SchedulingNode) error {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()
	
	node.UpdatedAt = time.Now()
	if node.CreatedAt.IsZero() {
		node.CreatedAt = time.Now()
	}
	
	s.nodes[node.ID] = node
	
	log.Printf("Registered node %s (type: %s, location: %s)", 
		node.ID, node.Type, node.Location.Region)
	
	return nil
}

func (s *UnifiedScheduler) UnregisterNode(nodeID string) error {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()
	
	node, exists := s.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node %s not found", nodeID)
	}
	
	// Check if node has active workloads
	if len(node.Workloads) > 0 {
		return fmt.Errorf("cannot unregister node %s with active workloads", nodeID)
	}
	
	delete(s.nodes, nodeID)
	
	log.Printf("Unregistered node %s", nodeID)
	return nil
}

func (s *UnifiedScheduler) GetSchedulingMetrics() *SchedulingMetrics {
	// Return a copy to prevent race conditions
	metricsCopy := *s.metrics
	return &metricsCopy
}

func (s *UnifiedScheduler) ListNodes() []*SchedulingNode {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()
	
	nodes := make([]*SchedulingNode, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}
	
	return nodes
}

func (s *UnifiedScheduler) ListWorkloads() []*ScheduledWorkload {
	s.workloadsMutex.RLock()
	defer s.workloadsMutex.RUnlock()
	
	workloads := make([]*ScheduledWorkload, 0, len(s.workloads))
	for _, workload := range s.workloads {
		workloads = append(workloads, workload)
	}
	
	return workloads
}