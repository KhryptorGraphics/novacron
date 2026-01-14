package compute

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/federation"
)

// ClusterCapacityProvider defines the interface for retrieving cluster resource information
type ClusterCapacityProvider interface {
	GetClusterResources(ctx context.Context, clusterID string) (*federation.ClusterResources, error)
}

// LoadBalancer interface for compute job load balancing
type LoadBalancer interface {
	SelectOptimalClusters(ctx context.Context, job *ComputeJob) ([]ClusterPlacement, error)
	UpdateClusterMetrics(clusterID string, metrics *ClusterMetrics) error
	GetLoadDistribution() map[string]*ClusterLoad
	SetPolicy(policy LoadBalancingPolicy) error
}

// ComputeJobLoadBalancer implements sophisticated load balancing algorithms
type ComputeJobLoadBalancer struct {
	mu                 sync.RWMutex
	federationMgr      federation.Provider
	clusterMetrics     map[string]*ClusterMetrics
	clusterLoad        map[string]*ClusterLoad
	performanceHistory map[string]*PerformanceHistory
	policy             LoadBalancingPolicy
	algorithms         map[LoadBalancingAlgorithm]LoadBalancingFunc
	affinityResolver   *AffinityResolver
	constraintSolver   *ConstraintSolver
	learningEngine     *MLLearningEngine
}

type LoadBalancingPolicy struct {
	Algorithm            LoadBalancingAlgorithm `json:"algorithm"`
	MaxClustersPerJob    int                    `json:"max_clusters_per_job"`
	MinResourceThreshold float64                `json:"min_resource_threshold"`
	LoadBalanceInterval  time.Duration          `json:"load_balance_interval"`
	PerformanceWeight    float64                `json:"performance_weight"`
	CostWeight           float64                `json:"cost_weight"`
	LatencyWeight        float64                `json:"latency_weight"`
	SecurityWeight       float64                `json:"security_weight"`
	DataLocalityWeight   float64                `json:"data_locality_weight"`
}

type LoadBalancingAlgorithm string

const (
	AlgorithmLeastLoaded      LoadBalancingAlgorithm = "least_loaded"
	AlgorithmWeightedRR       LoadBalancingAlgorithm = "weighted_round_robin"
	AlgorithmNetworkAware     LoadBalancingAlgorithm = "network_aware"
	AlgorithmCostOptimized    LoadBalancingAlgorithm = "cost_optimized"
	AlgorithmPerformanceBased LoadBalancingAlgorithm = "performance_based"
	AlgorithmMLPredictive     LoadBalancingAlgorithm = "ml_predictive"
	AlgorithmHybrid           LoadBalancingAlgorithm = "hybrid"
)

type LoadBalancingFunc func(ctx context.Context, job *ComputeJob, clusters []*ClusterInfo) ([]ClusterPlacement, error)

type ClusterMetrics struct {
	ClusterID           string                 `json:"cluster_id"`
	CPUUtilization      float64                `json:"cpu_utilization"`
	MemoryUtilization   float64                `json:"memory_utilization"`
	GPUUtilization      float64                `json:"gpu_utilization"`
	NetworkUtilization  float64                `json:"network_utilization"`
	DiskUtilization     float64                `json:"disk_utilization"`
	JobQueueLength      int                    `json:"job_queue_length"`
	AvgJobWaitTime      time.Duration          `json:"avg_job_wait_time"`
	AvgJobExecutionTime time.Duration          `json:"avg_job_execution_time"`
	SuccessRate         float64                `json:"success_rate"`
	Cost                *CostMetrics           `json:"cost,omitempty"`
	Network             *NetworkMetrics        `json:"network,omitempty"`
	Health              ClusterHealthStatus    `json:"health"`
	LastUpdated         time.Time              `json:"last_updated"`
	CustomMetrics       map[string]interface{} `json:"custom_metrics,omitempty"`
}

type ClusterLoad struct {
	ClusterID         string    `json:"cluster_id"`
	CurrentLoad       float64   `json:"current_load"`       // 0-1
	PredictedLoad     float64   `json:"predicted_load"`     // 0-1
	Capacity          float64   `json:"capacity"`           // Total capacity units
	AvailableCapacity float64   `json:"available_capacity"` // Available capacity units
	LoadTrend         LoadTrend `json:"load_trend"`
	Weight            float64   `json:"weight"` // Load balancing weight
	LastUpdated       time.Time `json:"last_updated"`
}

type LoadTrend string

const (
	TrendIncreasing LoadTrend = "increasing"
	TrendDecreasing LoadTrend = "decreasing"
	TrendStable     LoadTrend = "stable"
	TrendVolatile   LoadTrend = "volatile"
)

type CostMetrics struct {
	CPUCostPerCore   float64 `json:"cpu_cost_per_core"`
	MemoryCostPerGB  float64 `json:"memory_cost_per_gb"`
	GPUCostPerHour   float64 `json:"gpu_cost_per_hour"`
	NetworkCostPerGB float64 `json:"network_cost_per_gb"`
	StorageCostPerGB float64 `json:"storage_cost_per_gb"`
	TotalCostPerHour float64 `json:"total_cost_per_hour"`
}

type NetworkMetrics struct {
	Latency         time.Duration            `json:"latency"`
	Bandwidth       float64                  `json:"bandwidth_mbps"`
	PacketLoss      float64                  `json:"packet_loss"`
	Jitter          time.Duration            `json:"jitter"`
	InterClusterRTT map[string]time.Duration `json:"inter_cluster_rtt"`
}

type ClusterHealthStatus string

const (
	HealthHealthy     ClusterHealthStatus = "healthy"
	HealthDegraded    ClusterHealthStatus = "degraded"
	HealthUnhealthy   ClusterHealthStatus = "unhealthy"
	HealthMaintenance ClusterHealthStatus = "maintenance"
)

type PerformanceHistory struct {
	ClusterID           string                          `json:"cluster_id"`
	JobTypePerformance  map[JobType]*JobTypePerformance `json:"job_type_performance"`
	ResourceUtilization []ResourceUtilizationSample     `json:"resource_utilization"`
	PerformanceTrends   *PerformanceTrends              `json:"performance_trends"`
	MLPredictions       *MLPredictions                  `json:"ml_predictions,omitempty"`
}

type JobTypePerformance struct {
	JobType            JobType       `json:"job_type"`
	AvgExecutionTime   time.Duration `json:"avg_execution_time"`
	SuccessRate        float64       `json:"success_rate"`
	ThroughputJobs     float64       `json:"throughput_jobs_per_hour"`
	ResourceEfficiency float64       `json:"resource_efficiency"`
	SLACompliance      float64       `json:"sla_compliance"`
}

type ResourceUtilizationSample struct {
	Timestamp          time.Time `json:"timestamp"`
	CPUUtilization     float64   `json:"cpu_utilization"`
	MemoryUtilization  float64   `json:"memory_utilization"`
	GPUUtilization     float64   `json:"gpu_utilization"`
	NetworkUtilization float64   `json:"network_utilization"`
}

type PerformanceTrends struct {
	CPUTrend     TrendDirection `json:"cpu_trend"`
	MemoryTrend  TrendDirection `json:"memory_trend"`
	GPUTrend     TrendDirection `json:"gpu_trend"`
	NetworkTrend TrendDirection `json:"network_trend"`
	OverallTrend TrendDirection `json:"overall_trend"`
}

type TrendDirection string

const (
	TrendUp    TrendDirection = "up"
	TrendDown  TrendDirection = "down"
	TrendFlat  TrendDirection = "flat"
	TrendSpiky TrendDirection = "spiky"
)

type MLPredictions struct {
	PredictedJobTime     time.Duration `json:"predicted_job_time"`
	PredictedSuccess     float64       `json:"predicted_success"`
	PredictedUtilization float64       `json:"predicted_utilization"`
	ConfidenceScore      float64       `json:"confidence_score"`
}

// ClusterInfo represents cluster information for load balancing
type ClusterInfo struct {
	ID                 string                 `json:"id"`
	Name               string                 `json:"name"`
	Region             string                 `json:"region"`
	Zone               string                 `json:"zone"`
	TotalResources     ResourceAllocation     `json:"total_resources"`
	AvailableResources ResourceAllocation     `json:"available_resources"`
	Metrics            *ClusterMetrics        `json:"metrics"`
	Load               *ClusterLoad           `json:"load"`
	History            *PerformanceHistory    `json:"history"`
	SecurityTags       []string               `json:"security_tags"`
	ComplianceLevel    string                 `json:"compliance_level"`
	CostProfile        *CostMetrics           `json:"cost_profile"`
	NetworkProfile     *NetworkMetrics        `json:"network_profile"`
	Capabilities       []string               `json:"capabilities"`
	Constraints        map[string]interface{} `json:"constraints"`
}

// AffinityResolver handles job placement affinity rules
type AffinityResolver struct {
	mu            sync.RWMutex
	affinityRules map[string][]AffinityRule
	topology      *ClusterTopology
}

type ClusterTopology struct {
	Clusters    map[string]*ClusterInfo           `json:"clusters"`
	Connections map[string]map[string]*Connection `json:"connections"`
}

type Connection struct {
	Latency   time.Duration `json:"latency"`
	Bandwidth float64       `json:"bandwidth_mbps"`
	Cost      float64       `json:"cost"`
	Quality   float64       `json:"quality"`
}

// ConstraintSolver handles complex job constraints
type ConstraintSolver struct {
	mu          sync.RWMutex
	constraints map[string]ConstraintFunc
}

type ConstraintFunc func(job *ComputeJob, cluster *ClusterInfo) (bool, float64)

// MLLearningEngine provides machine learning-based optimization
type MLLearningEngine struct {
	mu           sync.RWMutex
	models       map[string]MLModel
	trainingData []TrainingDataPoint
	predictions  map[string]*MLPredictions
	enabled      bool
}

type MLModel interface {
	Predict(features []float64) ([]float64, error)
	Train(data []TrainingDataPoint) error
	GetFeatureImportance() map[string]float64
}

type TrainingDataPoint struct {
	JobFeatures     []float64 `json:"job_features"`
	ClusterFeatures []float64 `json:"cluster_features"`
	Outcome         []float64 `json:"outcome"`
	Timestamp       time.Time `json:"timestamp"`
}

// NewComputeJobLoadBalancer creates a new load balancer
func NewComputeJobLoadBalancer(federationMgr federation.Provider) *ComputeJobLoadBalancer {
	lb := &ComputeJobLoadBalancer{
		federationMgr:      federationMgr,
		clusterMetrics:     make(map[string]*ClusterMetrics),
		clusterLoad:        make(map[string]*ClusterLoad),
		performanceHistory: make(map[string]*PerformanceHistory),
		algorithms:         make(map[LoadBalancingAlgorithm]LoadBalancingFunc),
		affinityResolver:   NewAffinityResolver(),
		constraintSolver:   NewConstraintSolver(),
		learningEngine:     NewMLLearningEngine(),
		policy: LoadBalancingPolicy{
			Algorithm:            AlgorithmHybrid,
			MaxClustersPerJob:    5,
			MinResourceThreshold: 0.1,
			LoadBalanceInterval:  30 * time.Second,
			PerformanceWeight:    0.3,
			CostWeight:           0.2,
			LatencyWeight:        0.2,
			SecurityWeight:       0.15,
			DataLocalityWeight:   0.15,
		},
	}

	// Register load balancing algorithms
	lb.registerAlgorithms()

	// Initialize cluster resource capacities from federation manager
	lb.initializeClusterCapacities()

	return lb
}

// SelectOptimalClusters selects the best clusters for a job
func (lb *ComputeJobLoadBalancer) SelectOptimalClusters(ctx context.Context, job *ComputeJob) ([]ClusterPlacement, error) {
	// Nil checks
	if lb == nil {
		return nil, fmt.Errorf("load balancer is nil")
	}
	if job == nil {
		return nil, fmt.Errorf("job is nil")
	}
	if ctx == nil {
		return nil, fmt.Errorf("context is nil")
	}

	lb.mu.RLock()
	defer lb.mu.RUnlock()

	// Get available clusters
	clusters, err := lb.getAvailableClusters(ctx, job)
	if err != nil {
		return nil, fmt.Errorf("failed to get available clusters: %w", err)
	}

	if len(clusters) == 0 {
		return nil, fmt.Errorf("no suitable clusters available for job %s", job.ID)
	}

	// Apply constraints
	var filteredClusters []*ClusterInfo
	if lb.constraintSolver != nil {
		var err error
		filteredClusters, err = lb.constraintSolver.Filter(job, clusters)
		if err != nil {
			return nil, fmt.Errorf("constraint filtering failed: %w", err)
		}
	} else {
		// No constraint solver available, use all clusters
		filteredClusters = clusters
	}

	// Apply affinity rules
	affinityClusters := filteredClusters
	if lb.affinityResolver != nil {
		var err error
		affinityClusters, err = lb.affinityResolver.Resolve(job, filteredClusters)
		if err != nil {
			return nil, fmt.Errorf("affinity resolution failed: %w", err)
		}
	}

	// Select algorithm and execute
	algorithm := lb.algorithms[lb.policy.Algorithm]
	if algorithm == nil {
		return nil, fmt.Errorf("unknown load balancing algorithm: %s", lb.policy.Algorithm)
	}

	placements, err := algorithm(ctx, job, affinityClusters)
	if err != nil {
		return nil, fmt.Errorf("load balancing algorithm failed: %w", err)
	}

	// Update learning engine
	if lb.learningEngine != nil && lb.learningEngine.enabled {
		lb.learningEngine.RecordPlacement(job, placements)
	}

	return placements, nil
}

// registerAlgorithms registers all load balancing algorithms
func (lb *ComputeJobLoadBalancer) registerAlgorithms() {
	lb.algorithms[AlgorithmLeastLoaded] = lb.leastLoadedAlgorithm
	lb.algorithms[AlgorithmWeightedRR] = lb.weightedRoundRobinAlgorithm
	lb.algorithms[AlgorithmNetworkAware] = lb.networkAwareAlgorithm
	lb.algorithms[AlgorithmCostOptimized] = lb.costOptimizedAlgorithm
	lb.algorithms[AlgorithmPerformanceBased] = lb.performanceBasedAlgorithm
	lb.algorithms[AlgorithmMLPredictive] = lb.mlPredictiveAlgorithm
	lb.algorithms[AlgorithmHybrid] = lb.hybridAlgorithm
}

// leastLoadedAlgorithm implements least loaded cluster selection
func (lb *ComputeJobLoadBalancer) leastLoadedAlgorithm(ctx context.Context, job *ComputeJob, clusters []*ClusterInfo) ([]ClusterPlacement, error) {
	// Nil checks
	if clusters == nil || len(clusters) == 0 {
		return nil, fmt.Errorf("no clusters available")
	}
	if job == nil {
		return nil, fmt.Errorf("job is nil")
	}

	// Sort clusters by load
	sort.Slice(clusters, func(i, j int) bool {
		if clusters[i] == nil || clusters[j] == nil {
			return false
		}
		if clusters[i].Load == nil || clusters[j].Load == nil {
			return false
		}
		return clusters[i].Load.CurrentLoad < clusters[j].Load.CurrentLoad
	})

	var placements []ClusterPlacement
	remainingResources := job.Resources

	for _, cluster := range clusters {
		if cluster == nil {
			continue
		}

		if len(placements) >= lb.policy.MaxClustersPerJob {
			break
		}

		// Check if cluster can accommodate some resources
		allocation := lb.calculateAllocation(remainingResources, cluster.AvailableResources)
		if lb.isSignificantAllocation(allocation) {
			placement := ClusterPlacement{
				ClusterID: cluster.ID,
				Resources: allocation,
				Status:    PlacementPending,
			}
			placements = append(placements, placement)

			// Update remaining resources
			remainingResources = lb.subtractResources(remainingResources, allocation)
			if lb.isResourcesSatisfied(remainingResources) {
				break
			}
		}
	}

	return placements, nil
}

// weightedRoundRobinAlgorithm implements weighted round-robin selection
func (lb *ComputeJobLoadBalancer) weightedRoundRobinAlgorithm(ctx context.Context, job *ComputeJob, clusters []*ClusterInfo) ([]ClusterPlacement, error) {
	// Nil checks
	if clusters == nil || len(clusters) == 0 {
		return nil, fmt.Errorf("no clusters available")
	}
	if job == nil {
		return nil, fmt.Errorf("job is nil")
	}

	// Calculate total weight
	totalWeight := 0.0
	for _, cluster := range clusters {
		if cluster != nil && cluster.Load != nil {
			totalWeight += cluster.Load.Weight
		}
	}

	var placements []ClusterPlacement
	remainingResources := job.Resources

	for _, cluster := range clusters {
		if cluster == nil || cluster.Load == nil {
			continue
		}

		if len(placements) >= lb.policy.MaxClustersPerJob {
			break
		}

		// Allocate based on weight
		weightRatio := cluster.Load.Weight / totalWeight
		targetAllocation := lb.scaleResources(job.Resources, weightRatio)

		// Constrain by available resources
		allocation := lb.calculateAllocation(targetAllocation, cluster.AvailableResources)
		if lb.isSignificantAllocation(allocation) {
			placement := ClusterPlacement{
				ClusterID: cluster.ID,
				Resources: allocation,
				Status:    PlacementPending,
			}
			placements = append(placements, placement)

			remainingResources = lb.subtractResources(remainingResources, allocation)
			if lb.isResourcesSatisfied(remainingResources) {
				break
			}
		}
	}

	return placements, nil
}

// networkAwareAlgorithm considers network topology
func (lb *ComputeJobLoadBalancer) networkAwareAlgorithm(ctx context.Context, job *ComputeJob, clusters []*ClusterInfo) ([]ClusterPlacement, error) {
	// Score clusters based on network metrics
	type clusterScore struct {
		cluster *ClusterInfo
		score   float64
	}

	var scored []clusterScore
	for _, cluster := range clusters {
		score := lb.calculateNetworkScore(cluster, job)
		scored = append(scored, clusterScore{cluster, score})
	}

	// Sort by network score (higher is better)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var placements []ClusterPlacement
	remainingResources := job.Resources

	for _, item := range scored {
		if len(placements) >= lb.policy.MaxClustersPerJob {
			break
		}

		cluster := item.cluster
		allocation := lb.calculateAllocation(remainingResources, cluster.AvailableResources)
		if lb.isSignificantAllocation(allocation) {
			placement := ClusterPlacement{
				ClusterID: cluster.ID,
				Resources: allocation,
				Status:    PlacementPending,
			}
			placements = append(placements, placement)

			remainingResources = lb.subtractResources(remainingResources, allocation)
			if lb.isResourcesSatisfied(remainingResources) {
				break
			}
		}
	}

	return placements, nil
}

// costOptimizedAlgorithm minimizes cost
func (lb *ComputeJobLoadBalancer) costOptimizedAlgorithm(ctx context.Context, job *ComputeJob, clusters []*ClusterInfo) ([]ClusterPlacement, error) {
	// Sort clusters by cost (lower is better)
	sort.Slice(clusters, func(i, j int) bool {
		cost1 := lb.calculateJobCost(job, clusters[i])
		cost2 := lb.calculateJobCost(job, clusters[j])
		return cost1 < cost2
	})

	var placements []ClusterPlacement
	remainingResources := job.Resources

	for _, cluster := range clusters {
		if len(placements) >= lb.policy.MaxClustersPerJob {
			break
		}

		allocation := lb.calculateAllocation(remainingResources, cluster.AvailableResources)
		if lb.isSignificantAllocation(allocation) {
			placement := ClusterPlacement{
				ClusterID: cluster.ID,
				Resources: allocation,
				Status:    PlacementPending,
			}
			placements = append(placements, placement)

			remainingResources = lb.subtractResources(remainingResources, allocation)
			if lb.isResourcesSatisfied(remainingResources) {
				break
			}
		}
	}

	return placements, nil
}

// performanceBasedAlgorithm considers historical performance
func (lb *ComputeJobLoadBalancer) performanceBasedAlgorithm(ctx context.Context, job *ComputeJob, clusters []*ClusterInfo) ([]ClusterPlacement, error) {
	// Score clusters based on historical performance for this job type
	type clusterScore struct {
		cluster *ClusterInfo
		score   float64
	}

	var scored []clusterScore
	for _, cluster := range clusters {
		score := lb.calculatePerformanceScore(cluster, job)
		scored = append(scored, clusterScore{cluster, score})
	}

	// Sort by performance score (higher is better)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var placements []ClusterPlacement
	remainingResources := job.Resources

	for _, item := range scored {
		if len(placements) >= lb.policy.MaxClustersPerJob {
			break
		}

		cluster := item.cluster
		allocation := lb.calculateAllocation(remainingResources, cluster.AvailableResources)
		if lb.isSignificantAllocation(allocation) {
			placement := ClusterPlacement{
				ClusterID: cluster.ID,
				Resources: allocation,
				Status:    PlacementPending,
			}
			placements = append(placements, placement)

			remainingResources = lb.subtractResources(remainingResources, allocation)
			if lb.isResourcesSatisfied(remainingResources) {
				break
			}
		}
	}

	return placements, nil
}

// mlPredictiveAlgorithm uses machine learning predictions
func (lb *ComputeJobLoadBalancer) mlPredictiveAlgorithm(ctx context.Context, job *ComputeJob, clusters []*ClusterInfo) ([]ClusterPlacement, error) {
	if !lb.learningEngine.enabled {
		// Fallback to performance-based algorithm
		return lb.performanceBasedAlgorithm(ctx, job, clusters)
	}

	// Get ML predictions for each cluster
	type clusterPrediction struct {
		cluster    *ClusterInfo
		prediction *MLPredictions
		score      float64
	}

	var predictions []clusterPrediction
	for _, cluster := range clusters {
		pred, err := lb.learningEngine.PredictJobOutcome(job, cluster)
		if err != nil {
			continue
		}

		// Calculate composite score
		score := pred.PredictedSuccess * pred.ConfidenceScore * (1.0 - pred.PredictedUtilization)
		predictions = append(predictions, clusterPrediction{cluster, pred, score})
	}

	// Sort by ML score (higher is better)
	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].score > predictions[j].score
	})

	var placements []ClusterPlacement
	remainingResources := job.Resources

	for _, item := range predictions {
		if len(placements) >= lb.policy.MaxClustersPerJob {
			break
		}

		cluster := item.cluster
		allocation := lb.calculateAllocation(remainingResources, cluster.AvailableResources)
		if lb.isSignificantAllocation(allocation) {
			placement := ClusterPlacement{
				ClusterID: cluster.ID,
				Resources: allocation,
				Status:    PlacementPending,
			}
			placements = append(placements, placement)

			remainingResources = lb.subtractResources(remainingResources, allocation)
			if lb.isResourcesSatisfied(remainingResources) {
				break
			}
		}
	}

	return placements, nil
}

// hybridAlgorithm combines multiple algorithms
func (lb *ComputeJobLoadBalancer) hybridAlgorithm(ctx context.Context, job *ComputeJob, clusters []*ClusterInfo) ([]ClusterPlacement, error) {
	// Calculate composite score for each cluster
	type clusterScore struct {
		cluster *ClusterInfo
		score   float64
	}

	var scored []clusterScore
	for _, cluster := range clusters {
		score := lb.calculateHybridScore(cluster, job)
		scored = append(scored, clusterScore{cluster, score})
	}

	// Sort by hybrid score (higher is better)
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var placements []ClusterPlacement
	remainingResources := job.Resources

	for _, item := range scored {
		if len(placements) >= lb.policy.MaxClustersPerJob {
			break
		}

		cluster := item.cluster
		allocation := lb.calculateAllocation(remainingResources, cluster.AvailableResources)
		if lb.isSignificantAllocation(allocation) {
			placement := ClusterPlacement{
				ClusterID: cluster.ID,
				Resources: allocation,
				Status:    PlacementPending,
			}
			placements = append(placements, placement)

			remainingResources = lb.subtractResources(remainingResources, allocation)
			if lb.isResourcesSatisfied(remainingResources) {
				break
			}
		}
	}

	return placements, nil
}

// calculateHybridScore calculates a weighted composite score
func (lb *ComputeJobLoadBalancer) calculateHybridScore(cluster *ClusterInfo, job *ComputeJob) float64 {
	// Nil checks
	if cluster == nil || job == nil {
		return 0.0
	}

	// Performance score (0-1, higher is better)
	perfScore := lb.calculatePerformanceScore(cluster, job)

	// Cost score (0-1, higher is better for lower cost)
	cost := lb.calculateJobCost(job, cluster)
	costScore := 1.0 / (1.0 + cost) // Inverse cost

	// Network score (0-1, higher is better)
	networkScore := lb.calculateNetworkScore(cluster, job)

	// Load score (0-1, higher is better for lower load)
	var loadScore float64
	if cluster.Load != nil {
		loadScore = 1.0 - cluster.Load.CurrentLoad
	} else {
		loadScore = 0.5 // Default score if load data is unavailable
	}

	// Security score (0-1, higher is better)
	securityScore := lb.calculateSecurityScore(cluster, job)

	// Weighted composite score
	score := (perfScore * lb.policy.PerformanceWeight) +
		(costScore * lb.policy.CostWeight) +
		(networkScore * lb.policy.LatencyWeight) +
		(loadScore * (1.0 - lb.policy.PerformanceWeight - lb.policy.CostWeight - lb.policy.LatencyWeight - lb.policy.SecurityWeight - lb.policy.DataLocalityWeight)) +
		(securityScore * lb.policy.SecurityWeight)

	return score
}

// Helper methods for resource calculations
func (lb *ComputeJobLoadBalancer) calculateAllocation(requested, available ResourceAllocation) ResourceAllocation {
	return ResourceAllocation{
		CPUCores:    math.Min(requested.CPUCores, available.CPUCores),
		MemoryGB:    math.Min(requested.MemoryGB, available.MemoryGB),
		GPUCount:    int(math.Min(float64(requested.GPUCount), float64(available.GPUCount))),
		StorageGB:   math.Min(requested.StorageGB, available.StorageGB),
		NetworkMbps: math.Min(requested.NetworkMbps, available.NetworkMbps),
	}
}

func (lb *ComputeJobLoadBalancer) isSignificantAllocation(allocation ResourceAllocation) bool {
	return allocation.CPUCores >= lb.policy.MinResourceThreshold ||
		allocation.MemoryGB >= lb.policy.MinResourceThreshold ||
		allocation.GPUCount > 0
}

func (lb *ComputeJobLoadBalancer) subtractResources(a, b ResourceAllocation) ResourceAllocation {
	return ResourceAllocation{
		CPUCores:    math.Max(0, a.CPUCores-b.CPUCores),
		MemoryGB:    math.Max(0, a.MemoryGB-b.MemoryGB),
		GPUCount:    int(math.Max(0, float64(a.GPUCount-b.GPUCount))),
		StorageGB:   math.Max(0, a.StorageGB-b.StorageGB),
		NetworkMbps: math.Max(0, a.NetworkMbps-b.NetworkMbps),
	}
}

func (lb *ComputeJobLoadBalancer) scaleResources(resources ResourceAllocation, scale float64) ResourceAllocation {
	return ResourceAllocation{
		CPUCores:    resources.CPUCores * scale,
		MemoryGB:    resources.MemoryGB * scale,
		GPUCount:    int(float64(resources.GPUCount) * scale),
		StorageGB:   resources.StorageGB * scale,
		NetworkMbps: resources.NetworkMbps * scale,
	}
}

func (lb *ComputeJobLoadBalancer) isResourcesSatisfied(resources ResourceAllocation) bool {
	threshold := lb.policy.MinResourceThreshold
	return resources.CPUCores <= threshold &&
		resources.MemoryGB <= threshold &&
		resources.GPUCount == 0 &&
		resources.StorageGB <= threshold
}

// Scoring methods
func (lb *ComputeJobLoadBalancer) calculateNetworkScore(cluster *ClusterInfo, job *ComputeJob) float64 {
	if cluster.NetworkProfile == nil {
		return 0.5 // Default score
	}

	// Score based on latency (lower is better) and bandwidth (higher is better)
	latencyScore := 1.0 / (1.0 + cluster.NetworkProfile.Latency.Seconds())
	bandwidthScore := cluster.NetworkProfile.Bandwidth / 10000.0 // Normalize to 10Gbps
	if bandwidthScore > 1.0 {
		bandwidthScore = 1.0
	}

	// Packet loss penalty
	packetLossScore := 1.0 - cluster.NetworkProfile.PacketLoss

	return (latencyScore + bandwidthScore + packetLossScore) / 3.0
}

func (lb *ComputeJobLoadBalancer) calculateJobCost(job *ComputeJob, cluster *ClusterInfo) float64 {
	// Nil checks
	if job == nil || cluster == nil {
		return 1000.0 // Default high cost
	}

	if cluster.CostProfile == nil {
		return 1000.0 // Default high cost
	}

	cost := cluster.CostProfile.CPUCostPerCore*job.Resources.CPUCores +
		cluster.CostProfile.MemoryCostPerGB*job.Resources.MemoryGB +
		cluster.CostProfile.GPUCostPerHour*float64(job.Resources.GPUCount) +
		cluster.CostProfile.StorageCostPerGB*job.Resources.StorageGB

	return cost
}

func (lb *ComputeJobLoadBalancer) calculatePerformanceScore(cluster *ClusterInfo, job *ComputeJob) float64 {
	if cluster.History == nil {
		return 0.5 // Default score
	}

	jobTypePerf, exists := cluster.History.JobTypePerformance[job.Type]
	if !exists {
		return 0.5 // Default score
	}

	// Score based on success rate, efficiency, and SLA compliance
	successScore := jobTypePerf.SuccessRate
	efficiencyScore := jobTypePerf.ResourceEfficiency
	slaScore := jobTypePerf.SLACompliance

	return (successScore + efficiencyScore + slaScore) / 3.0
}

func (lb *ComputeJobLoadBalancer) calculateSecurityScore(cluster *ClusterInfo, job *ComputeJob) float64 {
	// Nil checks
	if cluster == nil || job == nil {
		return 0.0
	}

	// Check if job has security constraints
	if job.Constraints.SecurityTags == nil || len(job.Constraints.SecurityTags) == 0 {
		return 1.0 // No security requirements
	}

	// Check if cluster has security tags
	if cluster.SecurityTags == nil || len(cluster.SecurityTags) == 0 {
		return 0.0 // Cluster has no security tags but job requires them
	}

	matchingTags := 0
	for _, jobTag := range job.Constraints.SecurityTags {
		for _, clusterTag := range cluster.SecurityTags {
			if jobTag == clusterTag {
				matchingTags++
				break
			}
		}
	}

	return float64(matchingTags) / float64(len(job.Constraints.SecurityTags))
}

// getAvailableClusters retrieves available clusters from federation with populated capacity information
func (lb *ComputeJobLoadBalancer) getAvailableClusters(ctx context.Context, job *ComputeJob) ([]*ClusterInfo, error) {
	// Get cluster list from federation manager
	federatedClusters, err := lb.federationMgr.GetFederatedClusters(ctx)
	if err != nil {
		return nil, err
	}

	var clusters []*ClusterInfo
	for _, fedCluster := range federatedClusters {
		// Convert federation cluster to ClusterInfo
		cluster := &ClusterInfo{
			ID:     fedCluster.ID,
			Name:   fedCluster.Name,
			Region: fedCluster.Region,
			Zone:   fedCluster.Zone,
		}

		// Query cluster resources from federation manager
		clusterResources, err := lb.federationMgr.GetClusterResources(ctx, fedCluster.ID)
		if err != nil {
			// Log error but don't fail entirely - cluster might still be usable with default values
			clusterResources = &federation.ClusterResources{}
		}

		// Map federation.ClusterResources to ResourceAllocation for total resources
		cluster.TotalResources = ResourceAllocation{
			CPUCores:    float64(clusterResources.TotalCPU),
			MemoryGB:    float64(clusterResources.TotalMemoryGB),
			StorageGB:   float64(clusterResources.TotalStorageGB),
			GPUCount:    0,      // TODO: Add GPU support to federation.ClusterResources
			NetworkMbps: 1000.0, // TODO: Add network capacity to federation.ClusterResources
		}

		// Map federation.ClusterResources to ResourceAllocation for available resources
		cluster.AvailableResources = ResourceAllocation{
			CPUCores:    float64(clusterResources.AvailableCPU),
			MemoryGB:    float64(clusterResources.AvailableMemoryGB),
			StorageGB:   float64(clusterResources.AvailableStorageGB),
			GPUCount:    0,      // TODO: Add available GPU support to federation.ClusterResources
			NetworkMbps: 1000.0, // TODO: Add available network capacity to federation.ClusterResources
		}

		// Add metrics and load information
		if metrics, exists := lb.clusterMetrics[fedCluster.ID]; exists {
			cluster.Metrics = metrics
		}
		if load, exists := lb.clusterLoad[fedCluster.ID]; exists {
			cluster.Load = load
		}
		if history, exists := lb.performanceHistory[fedCluster.ID]; exists {
			cluster.History = history
		}

		clusters = append(clusters, cluster)
	}

	return clusters, nil
}

// UpdateClusterMetrics updates metrics for a cluster
func (lb *ComputeJobLoadBalancer) UpdateClusterMetrics(clusterID string, metrics *ClusterMetrics) error {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	metrics.LastUpdated = time.Now()
	lb.clusterMetrics[clusterID] = metrics

	// Update load information
	load := &ClusterLoad{
		ClusterID:   clusterID,
		CurrentLoad: lb.calculateCurrentLoad(metrics),
		LastUpdated: time.Now(),
	}
	lb.clusterLoad[clusterID] = load

	return nil
}

func (lb *ComputeJobLoadBalancer) calculateCurrentLoad(metrics *ClusterMetrics) float64 {
	// Weighted average of resource utilizations
	weights := map[string]float64{
		"cpu":     0.3,
		"memory":  0.3,
		"gpu":     0.2,
		"network": 0.1,
		"disk":    0.1,
	}

	load := weights["cpu"]*metrics.CPUUtilization +
		weights["memory"]*metrics.MemoryUtilization +
		weights["gpu"]*metrics.GPUUtilization +
		weights["network"]*metrics.NetworkUtilization +
		weights["disk"]*metrics.DiskUtilization

	return load / 100.0 // Convert percentage to 0-1
}

// GetLoadDistribution returns current load distribution
func (lb *ComputeJobLoadBalancer) GetLoadDistribution() map[string]*ClusterLoad {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	result := make(map[string]*ClusterLoad)
	for id, load := range lb.clusterLoad {
		result[id] = load
	}
	return result
}

// SetPolicy updates the load balancing policy
func (lb *ComputeJobLoadBalancer) SetPolicy(policy LoadBalancingPolicy) error {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	lb.policy = policy
	return nil
}

// Placeholder implementations for supporting components
func NewAffinityResolver() *AffinityResolver {
	return &AffinityResolver{
		affinityRules: make(map[string][]AffinityRule),
		topology: &ClusterTopology{
			Clusters:    make(map[string]*ClusterInfo),
			Connections: make(map[string]map[string]*Connection),
		},
	}
}

func (ar *AffinityResolver) Resolve(job *ComputeJob, clusters []*ClusterInfo) ([]*ClusterInfo, error) {
	// Apply affinity rules - placeholder implementation
	return clusters, nil
}

func NewConstraintSolver() *ConstraintSolver {
	return &ConstraintSolver{
		constraints: make(map[string]ConstraintFunc),
	}
}

func (cs *ConstraintSolver) Filter(job *ComputeJob, clusters []*ClusterInfo) ([]*ClusterInfo, error) {
	// Apply constraints - placeholder implementation
	return clusters, nil
}

func NewMLLearningEngine() *MLLearningEngine {
	return &MLLearningEngine{
		models:       make(map[string]MLModel),
		trainingData: make([]TrainingDataPoint, 0),
		predictions:  make(map[string]*MLPredictions),
		enabled:      false, // Disabled by default
	}
}

func (ml *MLLearningEngine) RecordPlacement(job *ComputeJob, placements []ClusterPlacement) {
	// Record placement for learning - placeholder implementation
}

func (ml *MLLearningEngine) PredictJobOutcome(job *ComputeJob, cluster *ClusterInfo) (*MLPredictions, error) {
	// ML prediction - placeholder implementation
	return &MLPredictions{
		PredictedJobTime:     30 * time.Minute,
		PredictedSuccess:     0.95,
		PredictedUtilization: 0.7,
		ConfidenceScore:      0.8,
	}, nil
}

// initializeClusterCapacities populates cluster resource capacities from federation manager
func (lb *ComputeJobLoadBalancer) initializeClusterCapacities() {
	if lb.federationMgr == nil {
		return
	}

	// Get cluster list from federation manager
	// This is a placeholder - actual implementation would depend on federation manager API
	clusters := []string{"cluster-1", "cluster-2", "cluster-3"} // Would come from federationMgr.ListClusters()

	for _, clusterID := range clusters {
		// Initialize cluster metrics with default capacities
		lb.clusterMetrics[clusterID] = &ClusterMetrics{
			ClusterID:           clusterID,
			CPUUtilization:      0.0,
			MemoryUtilization:   0.0,
			GPUUtilization:      0.0,
			NetworkUtilization:  0.0,
			DiskUtilization:     0.0,
			JobQueueLength:      0,
			AvgJobWaitTime:      0,
			AvgJobExecutionTime: 0,
			SuccessRate:         1.0,
			Health:              ClusterHealthHealthy,
			LastUpdated:         time.Now(),
		}

		// Initialize cluster load with default capacity values
		lb.clusterLoad[clusterID] = &ClusterLoad{
			ClusterID:         clusterID,
			CurrentLoad:       0.0,
			PredictedLoad:     0.0,
			Capacity:          1000.0, // Default capacity units
			AvailableCapacity: 1000.0, // Full capacity available initially
			LoadTrend:         TrendStable,
			Weight:            1.0,
			LastUpdated:       time.Now(),
		}

		// Initialize performance history
		lb.performanceHistory[clusterID] = &PerformanceHistory{
			ClusterID: clusterID,
			Records:   make([]*PerformanceRecord, 0),
		}
	}
}

// PopulateClusterCapacities updates cluster resource capacities from federation manager
func (lb *ComputeJobLoadBalancer) PopulateClusterCapacities() error {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	if lb.federationMgr == nil {
		return fmt.Errorf("federation manager not initialized")
	}

	// Get updated cluster information from federation manager
	// This would call federationMgr.GetClusterCapacities() or similar
	// For now, we'll update with current values

	for clusterID := range lb.clusterLoad {
		// Update cluster load information with actual capacity data
		// This would typically involve calling federation manager APIs
		if clusterLoad, exists := lb.clusterLoad[clusterID]; exists {
			// Placeholder: In real implementation, get actual capacity from federation manager
			clusterLoad.LastUpdated = time.Now()
			// clusterLoad.Capacity = federationMgr.GetClusterCapacity(clusterID)
			// clusterLoad.AvailableCapacity = federationMgr.GetAvailableCapacity(clusterID)
		}
	}

	return nil
}

// Load Balancer API Methods - Comment 15
// These methods expose load balancer functionality for external API access

// GetClusterMetrics returns current cluster metrics for API consumers
func (lb *ComputeJobLoadBalancer) GetClusterMetrics(clusterID string) (*ClusterMetrics, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	if clusterID == "" {
		return nil, fmt.Errorf("cluster ID cannot be empty")
	}

	metrics, exists := lb.clusterMetrics[clusterID]
	if !exists {
		return nil, fmt.Errorf("metrics not found for cluster %s", clusterID)
	}

	// Return a copy to prevent external modification
	return &ClusterMetrics{
		ClusterID:           metrics.ClusterID,
		CPUUtilization:      metrics.CPUUtilization,
		MemoryUtilization:   metrics.MemoryUtilization,
		StorageUtilization:  metrics.StorageUtilization,
		NetworkUtilization:  metrics.NetworkUtilization,
		ActiveJobs:          metrics.ActiveJobs,
		QueuedJobs:          metrics.QueuedJobs,
		AverageResponseTime: metrics.AverageResponseTime,
		ErrorRate:           metrics.ErrorRate,
		Availability:        metrics.Availability,
		LastUpdated:         metrics.LastUpdated,
	}, nil
}

// GetAllClusterMetrics returns metrics for all clusters
func (lb *ComputeJobLoadBalancer) GetAllClusterMetrics() (map[string]*ClusterMetrics, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	result := make(map[string]*ClusterMetrics)
	for clusterID, metrics := range lb.clusterMetrics {
		result[clusterID] = &ClusterMetrics{
			ClusterID:           metrics.ClusterID,
			CPUUtilization:      metrics.CPUUtilization,
			MemoryUtilization:   metrics.MemoryUtilization,
			StorageUtilization:  metrics.StorageUtilization,
			NetworkUtilization:  metrics.NetworkUtilization,
			ActiveJobs:          metrics.ActiveJobs,
			QueuedJobs:          metrics.QueuedJobs,
			AverageResponseTime: metrics.AverageResponseTime,
			ErrorRate:           metrics.ErrorRate,
			Availability:        metrics.Availability,
			LastUpdated:         metrics.LastUpdated,
		}
	}

	return result, nil
}

// GetClusterLoad returns current load information for a specific cluster
func (lb *ComputeJobLoadBalancer) GetClusterLoad(clusterID string) (*ClusterLoad, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	if clusterID == "" {
		return nil, fmt.Errorf("cluster ID cannot be empty")
	}

	load, exists := lb.clusterLoad[clusterID]
	if !exists {
		return nil, fmt.Errorf("load information not found for cluster %s", clusterID)
	}

	// Return a copy to prevent external modification
	return &ClusterLoad{
		ClusterID:         load.ClusterID,
		LoadScore:         load.LoadScore,
		Capacity:          load.Capacity,
		AvailableCapacity: load.AvailableCapacity,
		LastUpdated:       load.LastUpdated,
	}, nil
}

// GetLoadBalancingPolicy returns the current load balancing policy
func (lb *ComputeJobLoadBalancer) GetLoadBalancingPolicy() *LoadBalancingPolicy {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	// Return a copy to prevent external modification
	return &LoadBalancingPolicy{
		Algorithm:            lb.policy.Algorithm,
		MaxClustersPerJob:    lb.policy.MaxClustersPerJob,
		MinResourceThreshold: lb.policy.MinResourceThreshold,
		LoadBalanceInterval:  lb.policy.LoadBalanceInterval,
		PerformanceWeight:    lb.policy.PerformanceWeight,
		CostWeight:           lb.policy.CostWeight,
		LatencyWeight:        lb.policy.LatencyWeight,
		SecurityWeight:       lb.policy.SecurityWeight,
		DataLocalityWeight:   lb.policy.DataLocalityWeight,
	}
}

// UpdateLoadBalancingPolicy updates the load balancing policy with validation
func (lb *ComputeJobLoadBalancer) UpdateLoadBalancingPolicy(policy *LoadBalancingPolicy) error {
	if policy == nil {
		return fmt.Errorf("policy cannot be nil")
	}

	// Validate policy parameters
	if policy.MaxClustersPerJob < 1 {
		return fmt.Errorf("max clusters per job must be at least 1")
	}

	if policy.MinResourceThreshold < 0 || policy.MinResourceThreshold > 1 {
		return fmt.Errorf("min resource threshold must be between 0 and 1")
	}

	if policy.LoadBalanceInterval <= 0 {
		return fmt.Errorf("load balance interval must be positive")
	}

	// Validate weights sum to reasonable total (allow some flexibility)
	totalWeight := policy.PerformanceWeight + policy.CostWeight + policy.LatencyWeight + policy.SecurityWeight + policy.DataLocalityWeight
	if totalWeight <= 0 {
		return fmt.Errorf("total weight must be positive")
	}

	lb.mu.Lock()
	defer lb.mu.Unlock()

	lb.policy = *policy
	return nil
}

// GetClusterRanking returns clusters ranked by their suitability for a job type
func (lb *ComputeJobLoadBalancer) GetClusterRanking(ctx context.Context, jobType JobType) ([]ClusterRanking, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	rankings := make([]ClusterRanking, 0, len(lb.clusterMetrics))

	for clusterID, metrics := range lb.clusterMetrics {
		load, exists := lb.clusterLoad[clusterID]
		if !exists {
			continue
		}

		// Calculate ranking score based on multiple factors
		score := lb.calculateClusterScore(metrics, load, jobType)

		rankings = append(rankings, ClusterRanking{
			ClusterID:    clusterID,
			Score:        score,
			Availability: metrics.Availability,
			LoadScore:    load.LoadScore,
			Capacity:     load.AvailableCapacity,
			LastUpdated:  metrics.LastUpdated,
		})
	}

	// Sort by score (higher is better)
	sort.Slice(rankings, func(i, j int) bool {
		return rankings[i].Score > rankings[j].Score
	})

	return rankings, nil
}

// GetLoadBalancingStats returns comprehensive load balancing statistics
func (lb *ComputeJobLoadBalancer) GetLoadBalancingStats() (*LoadBalancingStats, error) {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	stats := &LoadBalancingStats{
		TotalClusters:       len(lb.clusterMetrics),
		HealthyClusters:     0,
		TotalCapacity:       0,
		AvailableCapacity:   0,
		AverageLoadScore:    0,
		ClusterDistribution: make(map[string]float64),
		LastUpdated:         time.Now(),
	}

	var totalLoadScore float64
	for clusterID, metrics := range lb.clusterMetrics {
		if metrics.Availability > 0.9 { // Consider healthy if > 90% availability
			stats.HealthyClusters++
		}

		if load, exists := lb.clusterLoad[clusterID]; exists {
			stats.TotalCapacity += load.Capacity
			stats.AvailableCapacity += load.AvailableCapacity
			totalLoadScore += load.LoadScore
			stats.ClusterDistribution[clusterID] = load.LoadScore
		}
	}

	if len(lb.clusterMetrics) > 0 {
		stats.AverageLoadScore = totalLoadScore / float64(len(lb.clusterMetrics))
	}

	if stats.TotalCapacity > 0 {
		stats.UtilizationRate = (stats.TotalCapacity - stats.AvailableCapacity) / stats.TotalCapacity
	}

	return stats, nil
}

// RebalanceClusters performs manual cluster rebalancing
func (lb *ComputeJobLoadBalancer) RebalanceClusters(ctx context.Context) (*RebalanceResult, error) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	result := &RebalanceResult{
		StartTime:        time.Now(),
		ClustersAnalyzed: len(lb.clusterMetrics),
		ActionsPerformed: []RebalanceAction{},
	}

	// Identify overloaded and underloaded clusters
	var overloaded, underloaded []string

	for clusterID, load := range lb.clusterLoad {
		if load.LoadScore > 0.8 { // Consider overloaded if > 80%
			overloaded = append(overloaded, clusterID)
		} else if load.LoadScore < 0.3 { // Consider underloaded if < 30%
			underloaded = append(underloaded, clusterID)
		}
	}

	// For each overloaded cluster, suggest actions
	for _, clusterID := range overloaded {
		action := RebalanceAction{
			ClusterID:   clusterID,
			ActionType:  "redistribute_jobs",
			Description: fmt.Sprintf("Redistribute jobs from overloaded cluster %s", clusterID),
			Priority:    "high",
			Timestamp:   time.Now(),
		}
		result.ActionsPerformed = append(result.ActionsPerformed, action)
	}

	// For underloaded clusters, suggest scaling down or job migration
	for _, clusterID := range underloaded {
		action := RebalanceAction{
			ClusterID:   clusterID,
			ActionType:  "scale_up_utilization",
			Description: fmt.Sprintf("Increase job allocation to underutilized cluster %s", clusterID),
			Priority:    "medium",
			Timestamp:   time.Now(),
		}
		result.ActionsPerformed = append(result.ActionsPerformed, action)
	}

	result.EndTime = time.Now()
	result.Duration = result.EndTime.Sub(result.StartTime)
	result.Success = true

	return result, nil
}

// Helper method to calculate cluster score for ranking
func (lb *ComputeJobLoadBalancer) calculateClusterScore(metrics *ClusterMetrics, load *ClusterLoad, jobType JobType) float64 {
	// Base score from availability and inverse load
	score := metrics.Availability * (1.0 - load.LoadScore)

	// Adjust for response time (lower is better)
	if metrics.AverageResponseTime > 0 {
		responseTimeFactor := 1.0 / (1.0 + metrics.AverageResponseTime.Seconds())
		score *= responseTimeFactor
	}

	// Adjust for error rate (lower is better)
	errorFactor := 1.0 - metrics.ErrorRate
	score *= errorFactor

	// Job type specific adjustments
	switch jobType {
	case JobTypeMPI:
		// MPI jobs prefer clusters with low network latency
		score *= (1.0 - metrics.NetworkUtilization)
	case JobTypeContainer:
		// Container jobs prefer clusters with available CPU
		score *= (1.0 - metrics.CPUUtilization)
	case JobTypeStream:
		// Streaming jobs prefer consistent performance
		if metrics.ErrorRate < 0.01 { // Very low error rate bonus
			score *= 1.2
		}
	}

	return math.Max(0, math.Min(1, score)) // Normalize to 0-1 range
}

// Supporting types for API methods
type ClusterRanking struct {
	ClusterID    string    `json:"cluster_id"`
	Score        float64   `json:"score"`
	Availability float64   `json:"availability"`
	LoadScore    float64   `json:"load_score"`
	Capacity     float64   `json:"capacity"`
	LastUpdated  time.Time `json:"last_updated"`
}

type LoadBalancingStats struct {
	TotalClusters       int                `json:"total_clusters"`
	HealthyClusters     int                `json:"healthy_clusters"`
	TotalCapacity       float64            `json:"total_capacity"`
	AvailableCapacity   float64            `json:"available_capacity"`
	UtilizationRate     float64            `json:"utilization_rate"`
	AverageLoadScore    float64            `json:"average_load_score"`
	ClusterDistribution map[string]float64 `json:"cluster_distribution"`
	LastUpdated         time.Time          `json:"last_updated"`
}

type RebalanceResult struct {
	StartTime        time.Time         `json:"start_time"`
	EndTime          time.Time         `json:"end_time"`
	Duration         time.Duration     `json:"duration"`
	Success          bool              `json:"success"`
	ClustersAnalyzed int               `json:"clusters_analyzed"`
	ActionsPerformed []RebalanceAction `json:"actions_performed"`
	ErrorMessage     string            `json:"error_message,omitempty"`
}

type RebalanceAction struct {
	ClusterID   string    `json:"cluster_id"`
	ActionType  string    `json:"action_type"`
	Description string    `json:"description"`
	Priority    string    `json:"priority"`
	Timestamp   time.Time `json:"timestamp"`
}

// Additional fields needed for lifecycle management
type LoadBalancerMetrics struct {
	TotalJobs        int64         `json:"total_jobs"`
	SuccessfulJobs   int64         `json:"successful_jobs"`
	FailedJobs       int64         `json:"failed_jobs"`
	AverageLatency   time.Duration `json:"average_latency"`
	ThroughputPerSec float64       `json:"throughput_per_sec"`
	LastUpdated      time.Time     `json:"last_updated"`
}

// Add lifecycle fields to ComputeJobLoadBalancer
func (lb *ComputeJobLoadBalancer) addLifecycleFields() {
	// These fields would be added to the struct definition:
	// isRunning bool
	// startTime time.Time
	// stopTime time.Time
	// cancelFunc context.CancelFunc
	// metrics *LoadBalancerMetrics
	// clusters map[string]*ClusterInfo
	// currentAlgorithm LoadBalancingAlgorithm
}

// Lifecycle management methods

// Start initializes and starts the load balancer
func (lb *ComputeJobLoadBalancer) Start(ctx context.Context) error {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	// For now, just mark as started - in real implementation would have isRunning field
	return nil
}

// Stop gracefully shuts down the load balancer
func (lb *ComputeJobLoadBalancer) Stop(ctx context.Context) error {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	// For now, just mark as stopped - in real implementation would have isRunning field
	return nil
}

// IsHealthy returns the health status of the load balancer
func (lb *ComputeJobLoadBalancer) IsHealthy() bool {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	// Check if we have healthy clusters
	healthyClusters := 0
	for _, metrics := range lb.clusterMetrics {
		if metrics.Health == HealthHealthy {
			healthyClusters++
		}
	}

	// Consider healthy if at least one cluster is available
	return healthyClusters > 0
}

// GetCurrentAlgorithm returns the currently active load balancing algorithm
func (lb *ComputeJobLoadBalancer) GetCurrentAlgorithm() LoadBalancingAlgorithm {
	lb.mu.RLock()
	defer lb.mu.RUnlock()
	return lb.policy.Algorithm
}

// SetAlgorithm changes the load balancing algorithm
func (lb *ComputeJobLoadBalancer) SetAlgorithm(algorithm LoadBalancingAlgorithm) error {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	// Validate algorithm
	if _, exists := lb.algorithms[algorithm]; !exists {
		return fmt.Errorf("unsupported algorithm: %s", algorithm)
	}

	lb.policy.Algorithm = algorithm
	return nil
}

// API Surface methods for external access

// GetStatus returns the current status of the load balancer
func (lb *ComputeJobLoadBalancer) GetStatus() *LoadBalancerStatus {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	status := &LoadBalancerStatus{
		IsRunning:        lb.IsHealthy(), // Use health as proxy for running status
		CurrentAlgorithm: lb.policy.Algorithm,
		TotalClusters:    len(lb.clusterMetrics),
		HealthyClusters:  0,
		LastUpdated:      time.Now(),
	}

	// Count healthy clusters
	for _, metrics := range lb.clusterMetrics {
		if metrics.Health == HealthHealthy {
			status.HealthyClusters++
		}
	}

	return status
}

// GetAvailableAlgorithms returns all supported load balancing algorithms
func (lb *ComputeJobLoadBalancer) GetAvailableAlgorithms() []AlgorithmInfo {
	algorithms := []AlgorithmInfo{
		{
			Name:        AlgorithmLeastLoaded,
			DisplayName: "Least Loaded",
			Description: "Routes jobs to clusters with the lowest current load",
			Category:    "Resource-based",
		},
		{
			Name:        AlgorithmWeightedRR,
			DisplayName: "Weighted Round Robin",
			Description: "Distributes jobs based on cluster weights in round-robin fashion",
			Category:    "Distribution-based",
		},
		{
			Name:        AlgorithmNetworkAware,
			DisplayName: "Network Aware",
			Description: "Considers network topology and latency for job placement",
			Category:    "Network-optimized",
		},
		{
			Name:        AlgorithmCostOptimized,
			DisplayName: "Cost Optimized",
			Description: "Minimizes cost by selecting most cost-effective clusters",
			Category:    "Cost-optimized",
		},
		{
			Name:        AlgorithmPerformanceBased,
			DisplayName: "Performance Based",
			Description: "Uses historical performance data for optimal placement",
			Category:    "Performance-optimized",
		},
		{
			Name:        AlgorithmMLPredictive,
			DisplayName: "ML Predictive",
			Description: "Uses machine learning to predict optimal job placement",
			Category:    "AI-powered",
		},
		{
			Name:        AlgorithmHybrid,
			DisplayName: "Hybrid",
			Description: "Combines multiple algorithms with weighted scoring",
			Category:    "Multi-factor",
		},
	}

	return algorithms
}

// GetMetrics returns comprehensive load balancer metrics
func (lb *ComputeJobLoadBalancer) GetMetrics() *LoadBalancerMetrics {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	// Calculate metrics from cluster data
	var totalJobs, successfulJobs, failedJobs int64
	var totalLatency time.Duration
	var throughput float64

	for _, metrics := range lb.clusterMetrics {
		// Aggregate job counts (these would be tracked in real implementation)
		totalJobs += int64(metrics.JobQueueLength)

		// Calculate success/failure based on success rate
		if metrics.SuccessRate > 0 {
			successfulJobs += int64(float64(metrics.JobQueueLength) * metrics.SuccessRate)
			failedJobs += int64(float64(metrics.JobQueueLength) * (1.0 - metrics.SuccessRate))
		}

		// Aggregate latency
		totalLatency += metrics.AvgJobExecutionTime
	}

	// Calculate average latency
	var avgLatency time.Duration
	if len(lb.clusterMetrics) > 0 {
		avgLatency = totalLatency / time.Duration(len(lb.clusterMetrics))
	}

	// Calculate throughput (jobs per second)
	if totalJobs > 0 && avgLatency > 0 {
		throughput = float64(totalJobs) / avgLatency.Seconds()
	}

	return &LoadBalancerMetrics{
		TotalJobs:        totalJobs,
		SuccessfulJobs:   successfulJobs,
		FailedJobs:       failedJobs,
		AverageLatency:   avgLatency,
		ThroughputPerSec: throughput,
		LastUpdated:      time.Now(),
	}
}

// Supporting types for API surface
type LoadBalancerStatus struct {
	IsRunning        bool                   `json:"is_running"`
	CurrentAlgorithm LoadBalancingAlgorithm `json:"current_algorithm"`
	TotalClusters    int                    `json:"total_clusters"`
	HealthyClusters  int                    `json:"healthy_clusters"`
	LastUpdated      time.Time              `json:"last_updated"`
}

type AlgorithmInfo struct {
	Name        LoadBalancingAlgorithm `json:"name"`
	DisplayName string                 `json:"display_name"`
	Description string                 `json:"description"`
	Category    string                 `json:"category"`
}
