// Package controller implements the Global Federation Controller for DWCP v3
// Provides cross-region VM placement, global load balancing, and region failover automation
package controller

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Global metrics for federation controller
var (
	placementDecisions = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_placement_decisions_total",
			Help: "Total number of VM placement decisions across regions",
		},
		[]string{"region", "decision_type"},
	)

	placementLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dwcp_federation_placement_latency_ms",
			Help:    "Latency of placement decisions in milliseconds",
			Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000},
		},
		[]string{"region"},
	)

	failoverDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dwcp_federation_failover_duration_seconds",
			Help:    "Duration of region failover operations",
			Buckets: []float64{5, 10, 15, 20, 25, 30, 45, 60, 90, 120},
		},
		[]string{"from_region", "to_region"},
	)

	activeRegions = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "dwcp_federation_active_regions",
			Help: "Number of active regions in the federation",
		},
	)

	migrationOperations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_federation_migrations_total",
			Help: "Total number of cross-region VM migrations",
		},
		[]string{"from_region", "to_region", "status"},
	)

	loadBalancingDecisions = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dwcp_federation_load_balancing_latency_ms",
			Help:    "Latency of global load balancing decisions",
			Buckets: []float64{1, 2, 5, 10, 20, 30, 40, 50},
		},
		[]string{"decision_type"},
	)
)

// RegionConfig defines configuration for a single region in the federation
type RegionConfig struct {
	ID              string             // Unique region identifier (e.g., "us-east-1")
	Name            string             // Human-readable name
	Location        GeoLocation        // Geographic coordinates
	CloudProvider   string             // "aws", "azure", "gcp", "edge"
	AvailabilityZones []string         // List of AZs in this region
	Capacity        RegionCapacity     // Available resources
	HealthStatus    HealthStatus       // Current health state
	NetworkLatency  map[string]float64 // Latency to other regions (ms)
	CostProfile     CostProfile        // Pricing information
	ComplianceZones []string           // Data sovereignty zones
	Priority        int                // Region priority for placement (1-100)
	Enabled         bool               // Whether region is active
	LastHeartbeat   time.Time          // Last health check timestamp
}

// GeoLocation represents geographic coordinates
type GeoLocation struct {
	Latitude  float64
	Longitude float64
	Country   string
	Continent string
}

// RegionCapacity tracks available resources in a region
type RegionCapacity struct {
	TotalCPU      int64   // Total CPU cores
	AvailableCPU  int64   // Available CPU cores
	TotalMemory   int64   // Total memory in MB
	AvailableMemory int64 // Available memory in MB
	TotalStorage  int64   // Total storage in GB
	AvailableStorage int64 // Available storage in GB
	TotalBandwidth int64  // Total bandwidth in Mbps
	AvailableBandwidth int64
	CPUUtilization float64 // CPU usage percentage
	MemoryUtilization float64
	StorageUtilization float64
}

// HealthStatus tracks region health
type HealthStatus struct {
	State           string    // "healthy", "degraded", "unhealthy", "maintenance"
	Score           float64   // Health score 0-100
	LastCheck       time.Time
	ConsecutiveFailures int
	Issues          []string  // List of current issues
}

// CostProfile defines pricing for a region
type CostProfile struct {
	CPUCostPerHour    float64 // Cost per CPU core hour
	MemoryCostPerHour float64 // Cost per GB memory hour
	StorageCostPerGB  float64 // Cost per GB storage per month
	NetworkCostPerGB  float64 // Cost per GB egress
	CurrencyCode      string  // ISO currency code
}

// PlacementRequest represents a VM placement request
type PlacementRequest struct {
	ID               string
	TenantID         string
	VMSpec           VMSpecification
	Constraints      PlacementConstraints
	Priority         int // 1-10 (10 = highest)
	Deadline         time.Time
	RequestTime      time.Time
	Metadata         map[string]string
}

// VMSpecification defines resource requirements
type VMSpecification struct {
	CPU         int64 // CPU cores
	Memory      int64 // Memory in MB
	Storage     int64 // Storage in GB
	Bandwidth   int64 // Required bandwidth in Mbps
	GPUs        int   // Number of GPUs
	GPUType     string
	OS          string
	VMType      string // "compute", "memory", "storage", "gpu"
}

// PlacementConstraints define placement rules
type PlacementConstraints struct {
	AllowedRegions    []string // Whitelist of regions
	ForbiddenRegions  []string // Blacklist of regions
	RequiredCompliance []string // Required compliance zones
	MaxLatencyMS      float64  // Maximum acceptable latency
	MinHealthScore    float64  // Minimum region health score
	AffinityGroups    []string // VM groups that should co-locate
	AntiAffinityGroups []string // VM groups that should not co-locate
	CostLimit         float64  // Maximum cost per hour
	PreferredProvider string   // Preferred cloud provider
	RequireEdge       bool     // Must be placed in edge location
	DataLocality      string   // Data residency requirement
}

// PlacementDecision represents the result of placement algorithm
type PlacementDecision struct {
	RequestID        string
	SelectedRegion   string
	AvailabilityZone string
	Score            float64 // Placement score 0-100
	DecisionTime     time.Duration
	DecisionFactors  map[string]float64 // Factor weights
	AlternativeRegions []string
	EstimatedCost    float64
	ExpectedLatency  float64
	Timestamp        time.Time
}

// MigrationRequest represents a cross-region migration
type MigrationRequest struct {
	ID            string
	VMID          string
	SourceRegion  string
	TargetRegion  string
	Priority      int
	Reason        string // "failover", "rebalance", "cost-optimization", "manual"
	PreCopyEnabled bool  // Enable pre-copy live migration
	MaxDowntime   time.Duration
	RequestTime   time.Time
	Deadline      time.Time
}

// FailoverPolicy defines region failover configuration
type FailoverPolicy struct {
	Enabled                bool
	MaxFailoverTime        time.Duration // Target RTO
	AutoFailover           bool
	FailoverThreshold      float64 // Health score threshold
	PrimaryRegions         []string
	SecondaryRegions       []string
	NotificationWebhooks   []string
	RequireManualApproval  bool
	MinHealthyRegions      int
}

// GlobalFederationController manages multi-region orchestration
type GlobalFederationController struct {
	mu                sync.RWMutex
	regions           map[string]*RegionConfig
	placementCache    map[string]*PlacementDecision
	migrationQueue    chan *MigrationRequest
	failoverPolicy    *FailoverPolicy
	metrics           *ControllerMetrics
	stopCh            chan struct{}
	wg                sync.WaitGroup
	healthCheckInterval time.Duration
	placementAlgorithm string // "latency", "cost", "balanced", "ai"
	enablePreemption   bool
	maxConcurrentMigrations int
}

// ControllerMetrics tracks controller performance
type ControllerMetrics struct {
	PlacementDecisions      int64
	SuccessfulPlacements    int64
	FailedPlacements        int64
	AveragePlacementLatency time.Duration
	TotalMigrations         int64
	SuccessfulMigrations    int64
	FailedMigrations        int64
	AverageFailoverTime     time.Duration
	RegionFailovers         int64
	ActiveMigrations        int64
	mu                      sync.RWMutex
}

// NewGlobalFederationController creates a new federation controller
func NewGlobalFederationController(cfg *FederationConfig) (*GlobalFederationController, error) {
	if cfg == nil {
		return nil, fmt.Errorf("federation config cannot be nil")
	}

	gfc := &GlobalFederationController{
		regions:                 make(map[string]*RegionConfig),
		placementCache:          make(map[string]*PlacementDecision),
		migrationQueue:          make(chan *MigrationRequest, 1000),
		metrics:                 &ControllerMetrics{},
		stopCh:                  make(chan struct{}),
		healthCheckInterval:     cfg.HealthCheckInterval,
		placementAlgorithm:      cfg.PlacementAlgorithm,
		enablePreemption:        cfg.EnablePreemption,
		maxConcurrentMigrations: cfg.MaxConcurrentMigrations,
		failoverPolicy: &FailoverPolicy{
			Enabled:            cfg.AutoFailover,
			MaxFailoverTime:    30 * time.Second,
			AutoFailover:       cfg.AutoFailover,
			FailoverThreshold:  cfg.FailoverThreshold,
			MinHealthyRegions:  2,
		},
	}

	// Initialize regions
	for _, regionCfg := range cfg.Regions {
		if err := gfc.AddRegion(regionCfg); err != nil {
			return nil, fmt.Errorf("failed to add region %s: %w", regionCfg.ID, err)
		}
	}

	return gfc, nil
}

// FederationConfig defines controller configuration
type FederationConfig struct {
	Regions                 []*RegionConfig
	HealthCheckInterval     time.Duration
	PlacementAlgorithm      string
	EnablePreemption        bool
	MaxConcurrentMigrations int
	AutoFailover            bool
	FailoverThreshold       float64
}

// Start starts the federation controller
func (gfc *GlobalFederationController) Start(ctx context.Context) error {
	// Start health check loop
	gfc.wg.Add(1)
	go gfc.healthCheckLoop(ctx)

	// Start migration processor
	gfc.wg.Add(1)
	go gfc.processMigrations(ctx)

	// Start metrics collector
	gfc.wg.Add(1)
	go gfc.collectMetrics(ctx)

	return nil
}

// Stop stops the federation controller
func (gfc *GlobalFederationController) Stop() error {
	close(gfc.stopCh)
	gfc.wg.Wait()
	return nil
}

// AddRegion adds a new region to the federation
func (gfc *GlobalFederationController) AddRegion(region *RegionConfig) error {
	gfc.mu.Lock()
	defer gfc.mu.Unlock()

	if region == nil {
		return fmt.Errorf("region config cannot be nil")
	}

	if _, exists := gfc.regions[region.ID]; exists {
		return fmt.Errorf("region %s already exists", region.ID)
	}

	region.LastHeartbeat = time.Now()
	gfc.regions[region.ID] = region
	activeRegions.Set(float64(len(gfc.regions)))

	return nil
}

// PlaceVM performs intelligent VM placement across regions
// Target: <50ms placement decision latency
func (gfc *GlobalFederationController) PlaceVM(ctx context.Context, req *PlacementRequest) (*PlacementDecision, error) {
	startTime := time.Now()
	defer func() {
		latency := time.Since(startTime)
		placementLatency.WithLabelValues("all").Observe(float64(latency.Milliseconds()))
	}()

	gfc.mu.RLock()
	defer gfc.mu.RUnlock()

	// Validate request
	if req == nil {
		return nil, fmt.Errorf("placement request cannot be nil")
	}

	// Filter eligible regions
	eligibleRegions := gfc.filterEligibleRegions(req)
	if len(eligibleRegions) == 0 {
		placementDecisions.WithLabelValues("none", "no_eligible_regions").Inc()
		return nil, fmt.Errorf("no eligible regions found for placement")
	}

	// Score each region
	regionScores := gfc.scoreRegions(req, eligibleRegions)

	// Sort by score (descending)
	sort.Slice(regionScores, func(i, j int) bool {
		return regionScores[i].Score > regionScores[j].Score
	})

	// Select best region
	best := regionScores[0]

	// Select availability zone within region
	az := gfc.selectAvailabilityZone(best.RegionID, req)

	decision := &PlacementDecision{
		RequestID:        req.ID,
		SelectedRegion:   best.RegionID,
		AvailabilityZone: az,
		Score:            best.Score,
		DecisionTime:     time.Since(startTime),
		DecisionFactors:  best.Factors,
		AlternativeRegions: extractAlternatives(regionScores, 3),
		EstimatedCost:    best.EstimatedCost,
		ExpectedLatency:  best.ExpectedLatency,
		Timestamp:        time.Now(),
	}

	// Cache decision
	gfc.placementCache[req.ID] = decision

	// Update metrics
	gfc.metrics.mu.Lock()
	gfc.metrics.PlacementDecisions++
	gfc.metrics.SuccessfulPlacements++
	gfc.metrics.AveragePlacementLatency =
		(gfc.metrics.AveragePlacementLatency*time.Duration(gfc.metrics.PlacementDecisions-1) + decision.DecisionTime) /
		time.Duration(gfc.metrics.PlacementDecisions)
	gfc.metrics.mu.Unlock()

	placementDecisions.WithLabelValues(decision.SelectedRegion, "success").Inc()

	return decision, nil
}

// RegionScore represents a scored region candidate
type RegionScore struct {
	RegionID        string
	Score           float64
	Factors         map[string]float64
	EstimatedCost   float64
	ExpectedLatency float64
}

// filterEligibleRegions filters regions based on constraints
func (gfc *GlobalFederationController) filterEligibleRegions(req *PlacementRequest) []*RegionConfig {
	var eligible []*RegionConfig

	for _, region := range gfc.regions {
		// Check enabled
		if !region.Enabled {
			continue
		}

		// Check health
		if region.HealthStatus.Score < req.Constraints.MinHealthScore {
			continue
		}

		// Check allowed/forbidden lists
		if len(req.Constraints.AllowedRegions) > 0 {
			if !contains(req.Constraints.AllowedRegions, region.ID) {
				continue
			}
		}
		if contains(req.Constraints.ForbiddenRegions, region.ID) {
			continue
		}

		// Check compliance
		if !hasRequiredCompliance(region.ComplianceZones, req.Constraints.RequiredCompliance) {
			continue
		}

		// Check capacity
		if !hasCapacity(region.Capacity, req.VMSpec) {
			continue
		}

		// Check edge requirement
		if req.Constraints.RequireEdge && region.CloudProvider != "edge" {
			continue
		}

		eligible = append(eligible, region)
	}

	return eligible
}

// scoreRegions scores each eligible region using multi-factor algorithm
func (gfc *GlobalFederationController) scoreRegions(req *PlacementRequest, regions []*RegionConfig) []RegionScore {
	scores := make([]RegionScore, 0, len(regions))

	for _, region := range regions {
		factors := make(map[string]float64)

		// Factor 1: Latency (30% weight)
		latencyScore := gfc.calculateLatencyScore(region, req)
		factors["latency"] = latencyScore

		// Factor 2: Cost (25% weight)
		costScore := gfc.calculateCostScore(region, req)
		factors["cost"] = costScore

		// Factor 3: Capacity (20% weight)
		capacityScore := gfc.calculateCapacityScore(region, req)
		factors["capacity"] = capacityScore

		// Factor 4: Health (15% weight)
		healthScore := region.HealthStatus.Score / 100.0
		factors["health"] = healthScore

		// Factor 5: Priority (10% weight)
		priorityScore := float64(region.Priority) / 100.0
		factors["priority"] = priorityScore

		// Calculate weighted score
		totalScore := (latencyScore * 0.30) +
			(costScore * 0.25) +
			(capacityScore * 0.20) +
			(healthScore * 0.15) +
			(priorityScore * 0.10)

		// Adjust for algorithm preference
		totalScore = gfc.adjustScoreByAlgorithm(totalScore, factors)

		// Calculate estimated cost
		estimatedCost := gfc.calculateEstimatedCost(region, req)

		// Calculate expected latency
		expectedLatency := gfc.calculateExpectedLatency(region, req)

		scores = append(scores, RegionScore{
			RegionID:        region.ID,
			Score:           totalScore * 100, // Scale to 0-100
			Factors:         factors,
			EstimatedCost:   estimatedCost,
			ExpectedLatency: expectedLatency,
		})
	}

	return scores
}

// calculateLatencyScore calculates latency-based score
func (gfc *GlobalFederationController) calculateLatencyScore(region *RegionConfig, req *PlacementRequest) float64 {
	// For edge deployments, latency is critical
	if req.Constraints.RequireEdge {
		// Assume edge has <10ms latency to users
		return 1.0
	}

	// Check max latency constraint
	if req.Constraints.MaxLatencyMS > 0 {
		avgLatency := gfc.getAverageLatencyToRegion(region.ID)
		if avgLatency > req.Constraints.MaxLatencyMS {
			return 0.0 // Disqualified
		}
		// Score inversely proportional to latency
		return 1.0 - (avgLatency / req.Constraints.MaxLatencyMS)
	}

	// Default: assume lower latency for regions with more connectivity
	return 0.8 // Baseline score
}

// calculateCostScore calculates cost-based score
func (gfc *GlobalFederationController) calculateCostScore(region *RegionConfig, req *PlacementRequest) float64 {
	estimatedCost := gfc.calculateEstimatedCost(region, req)

	if req.Constraints.CostLimit > 0 {
		if estimatedCost > req.Constraints.CostLimit {
			return 0.0 // Exceeds budget
		}
		// Higher score for lower cost
		return 1.0 - (estimatedCost / req.Constraints.CostLimit)
	}

	// Default: normalize against average cost
	avgCost := 0.5 // Baseline
	return math.Max(0, 1.0-(estimatedCost/avgCost))
}

// calculateCapacityScore calculates capacity-based score
func (gfc *GlobalFederationController) calculateCapacityScore(region *RegionConfig, req *PlacementRequest) float64 {
	cpuScore := float64(region.Capacity.AvailableCPU) / float64(region.Capacity.TotalCPU)
	memScore := float64(region.Capacity.AvailableMemory) / float64(region.Capacity.TotalMemory)
	storageScore := float64(region.Capacity.AvailableStorage) / float64(region.Capacity.TotalStorage)

	// Weighted average
	return (cpuScore*0.4 + memScore*0.4 + storageScore*0.2)
}

// calculateEstimatedCost estimates hourly cost for VM in region
func (gfc *GlobalFederationController) calculateEstimatedCost(region *RegionConfig, req *PlacementRequest) float64 {
	cpuCost := float64(req.VMSpec.CPU) * region.CostProfile.CPUCostPerHour
	memCost := float64(req.VMSpec.Memory) / 1024.0 * region.CostProfile.MemoryCostPerHour
	return cpuCost + memCost
}

// calculateExpectedLatency estimates latency to region
func (gfc *GlobalFederationController) calculateExpectedLatency(region *RegionConfig, req *PlacementRequest) float64 {
	return gfc.getAverageLatencyToRegion(region.ID)
}

// getAverageLatencyToRegion gets average latency to a region
func (gfc *GlobalFederationController) getAverageLatencyToRegion(regionID string) float64 {
	region := gfc.regions[regionID]
	if region == nil {
		return 100.0 // Default high latency
	}

	if len(region.NetworkLatency) == 0 {
		return 50.0 // Default moderate latency
	}

	var total float64
	for _, latency := range region.NetworkLatency {
		total += latency
	}
	return total / float64(len(region.NetworkLatency))
}

// adjustScoreByAlgorithm adjusts score based on placement algorithm
func (gfc *GlobalFederationController) adjustScoreByAlgorithm(score float64, factors map[string]float64) float64 {
	switch gfc.placementAlgorithm {
	case "latency":
		// Heavily weight latency
		return factors["latency"]*0.7 + score*0.3
	case "cost":
		// Heavily weight cost
		return factors["cost"]*0.7 + score*0.3
	case "balanced":
		// Keep default weighting
		return score
	case "ai":
		// TODO: Use ML model for scoring
		return score
	default:
		return score
	}
}

// selectAvailabilityZone selects best AZ within region
func (gfc *GlobalFederationController) selectAvailabilityZone(regionID string, req *PlacementRequest) string {
	region := gfc.regions[regionID]
	if region == nil || len(region.AvailabilityZones) == 0 {
		return ""
	}

	// Simple round-robin for now
	// TODO: Implement AZ-level capacity and health checking
	return region.AvailabilityZones[0]
}

// MigrateVM initiates cross-region VM migration
func (gfc *GlobalFederationController) MigrateVM(ctx context.Context, req *MigrationRequest) error {
	if req == nil {
		return fmt.Errorf("migration request cannot be nil")
	}

	// Validate source and target regions exist
	gfc.mu.RLock()
	_, sourceExists := gfc.regions[req.SourceRegion]
	_, targetExists := gfc.regions[req.TargetRegion]
	gfc.mu.RUnlock()

	if !sourceExists {
		return fmt.Errorf("source region %s not found", req.SourceRegion)
	}
	if !targetExists {
		return fmt.Errorf("target region %s not found", req.TargetRegion)
	}

	// Queue migration
	select {
	case gfc.migrationQueue <- req:
		gfc.metrics.mu.Lock()
		gfc.metrics.ActiveMigrations++
		gfc.metrics.mu.Unlock()
		return nil
	case <-ctx.Done():
		return ctx.Err()
	default:
		return fmt.Errorf("migration queue full")
	}
}

// processMigrations processes migration requests
func (gfc *GlobalFederationController) processMigrations(ctx context.Context) {
	defer gfc.wg.Done()

	semaphore := make(chan struct{}, gfc.maxConcurrentMigrations)

	for {
		select {
		case <-ctx.Done():
			return
		case <-gfc.stopCh:
			return
		case req := <-gfc.migrationQueue:
			// Acquire semaphore
			semaphore <- struct{}{}

			go func(r *MigrationRequest) {
				defer func() { <-semaphore }()

				if err := gfc.executeMigration(ctx, r); err != nil {
					gfc.metrics.mu.Lock()
					gfc.metrics.FailedMigrations++
					gfc.metrics.mu.Unlock()
					migrationOperations.WithLabelValues(r.SourceRegion, r.TargetRegion, "failed").Inc()
				} else {
					gfc.metrics.mu.Lock()
					gfc.metrics.SuccessfulMigrations++
					gfc.metrics.mu.Unlock()
					migrationOperations.WithLabelValues(r.SourceRegion, r.TargetRegion, "success").Inc()
				}

				gfc.metrics.mu.Lock()
				gfc.metrics.TotalMigrations++
				gfc.metrics.ActiveMigrations--
				gfc.metrics.mu.Unlock()
			}(req)
		}
	}
}

// executeMigration executes a single migration
func (gfc *GlobalFederationController) executeMigration(ctx context.Context, req *MigrationRequest) error {
	startTime := time.Now()

	// Phase 1: Pre-copy (if enabled)
	if req.PreCopyEnabled {
		// Implement iterative pre-copy
		// TODO: Integrate with DWCP v3 live migration
	}

	// Phase 2: Stop-and-copy
	// TODO: Coordinate with source and target regions

	// Phase 3: Activate on target
	// TODO: Update routing and DNS

	duration := time.Since(startTime)

	// Check if within max downtime
	if duration > req.MaxDowntime {
		return fmt.Errorf("migration exceeded max downtime: %v > %v", duration, req.MaxDowntime)
	}

	return nil
}

// PerformFailover performs region failover
// Target: <30s RTO (Recovery Time Objective)
func (gfc *GlobalFederationController) PerformFailover(ctx context.Context, failedRegion string) error {
	startTime := time.Now()
	defer func() {
		duration := time.Since(startTime)
		failoverDuration.WithLabelValues(failedRegion, "multiple").Observe(duration.Seconds())
	}()

	gfc.mu.Lock()
	defer gfc.mu.Unlock()

	region, exists := gfc.regions[failedRegion]
	if !exists {
		return fmt.Errorf("region %s not found", failedRegion)
	}

	// Mark region as failed
	region.Enabled = false
	region.HealthStatus.State = "unhealthy"

	// Find healthy target regions
	targetRegions := gfc.findHealthyRegions(failedRegion)
	if len(targetRegions) == 0 {
		return fmt.Errorf("no healthy target regions for failover")
	}

	// Distribute VMs across target regions
	// TODO: Query VMs in failed region and redistribute

	gfc.metrics.mu.Lock()
	gfc.metrics.RegionFailovers++
	gfc.metrics.AverageFailoverTime =
		(gfc.metrics.AverageFailoverTime*time.Duration(gfc.metrics.RegionFailovers-1) + time.Since(startTime)) /
		time.Duration(gfc.metrics.RegionFailovers)
	gfc.metrics.mu.Unlock()

	return nil
}

// findHealthyRegions finds healthy regions for failover
func (gfc *GlobalFederationController) findHealthyRegions(excludeRegion string) []string {
	var healthy []string

	for id, region := range gfc.regions {
		if id == excludeRegion {
			continue
		}
		if region.Enabled && region.HealthStatus.State == "healthy" {
			healthy = append(healthy, id)
		}
	}

	return healthy
}

// healthCheckLoop performs periodic health checks
func (gfc *GlobalFederationController) healthCheckLoop(ctx context.Context) {
	defer gfc.wg.Done()

	ticker := time.NewTicker(gfc.healthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gfc.stopCh:
			return
		case <-ticker.C:
			gfc.performHealthChecks(ctx)
		}
	}
}

// performHealthChecks checks health of all regions
func (gfc *GlobalFederationController) performHealthChecks(ctx context.Context) {
	gfc.mu.Lock()
	defer gfc.mu.Unlock()

	for _, region := range gfc.regions {
		score := gfc.calculateRegionHealth(region)
		region.HealthStatus.Score = score
		region.HealthStatus.LastCheck = time.Now()

		if score >= 80 {
			region.HealthStatus.State = "healthy"
			region.HealthStatus.ConsecutiveFailures = 0
		} else if score >= 50 {
			region.HealthStatus.State = "degraded"
		} else {
			region.HealthStatus.State = "unhealthy"
			region.HealthStatus.ConsecutiveFailures++

			// Auto-failover if enabled
			if gfc.failoverPolicy.AutoFailover &&
			   score < gfc.failoverPolicy.FailoverThreshold &&
			   region.HealthStatus.ConsecutiveFailures >= 3 {
				go gfc.PerformFailover(context.Background(), region.ID)
			}
		}
	}
}

// calculateRegionHealth calculates overall region health score
func (gfc *GlobalFederationController) calculateRegionHealth(region *RegionConfig) float64 {
	var score float64 = 100.0

	// Deduct for high utilization
	if region.Capacity.CPUUtilization > 90 {
		score -= 20
	} else if region.Capacity.CPUUtilization > 80 {
		score -= 10
	}

	// Deduct for memory pressure
	if region.Capacity.MemoryUtilization > 90 {
		score -= 20
	} else if region.Capacity.MemoryUtilization > 80 {
		score -= 10
	}

	// Deduct for stale heartbeat
	if time.Since(region.LastHeartbeat) > 2*gfc.healthCheckInterval {
		score -= 30
	}

	// Deduct for issues
	score -= float64(len(region.HealthStatus.Issues)) * 5

	return math.Max(0, score)
}

// collectMetrics periodically collects and exports metrics
func (gfc *GlobalFederationController) collectMetrics(ctx context.Context) {
	defer gfc.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-gfc.stopCh:
			return
		case <-ticker.C:
			gfc.exportMetrics()
		}
	}
}

// exportMetrics exports current metrics
func (gfc *GlobalFederationController) exportMetrics() {
	gfc.mu.RLock()
	activeCount := 0
	for _, region := range gfc.regions {
		if region.Enabled && region.HealthStatus.State == "healthy" {
			activeCount++
		}
	}
	gfc.mu.RUnlock()

	activeRegions.Set(float64(activeCount))
}

// GetMetrics returns current controller metrics
func (gfc *GlobalFederationController) GetMetrics() *ControllerMetrics {
	gfc.metrics.mu.RLock()
	defer gfc.metrics.mu.RUnlock()

	// Return a copy
	return &ControllerMetrics{
		PlacementDecisions:      gfc.metrics.PlacementDecisions,
		SuccessfulPlacements:    gfc.metrics.SuccessfulPlacements,
		FailedPlacements:        gfc.metrics.FailedPlacements,
		AveragePlacementLatency: gfc.metrics.AveragePlacementLatency,
		TotalMigrations:         gfc.metrics.TotalMigrations,
		SuccessfulMigrations:    gfc.metrics.SuccessfulMigrations,
		FailedMigrations:        gfc.metrics.FailedMigrations,
		AverageFailoverTime:     gfc.metrics.AverageFailoverTime,
		RegionFailovers:         gfc.metrics.RegionFailovers,
		ActiveMigrations:        gfc.metrics.ActiveMigrations,
	}
}

// Helper functions

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func hasRequiredCompliance(available, required []string) bool {
	if len(required) == 0 {
		return true
	}
	for _, req := range required {
		if !contains(available, req) {
			return false
		}
	}
	return true
}

func hasCapacity(capacity RegionCapacity, spec VMSpecification) bool {
	return capacity.AvailableCPU >= spec.CPU &&
		capacity.AvailableMemory >= spec.Memory &&
		capacity.AvailableStorage >= spec.Storage
}

func extractAlternatives(scores []RegionScore, count int) []string {
	var alternatives []string
	for i := 1; i < len(scores) && i <= count; i++ {
		alternatives = append(alternatives, scores[i].RegionID)
	}
	return alternatives
}
