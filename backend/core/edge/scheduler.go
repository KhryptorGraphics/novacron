// Package edge provides edge workload scheduling with latency awareness
package edge

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// WorkloadType represents the type of workload
type WorkloadType string

const (
	WorkloadTypeStateless  WorkloadType = "stateless"
	WorkloadTypeStateful   WorkloadType = "stateful"
	WorkloadTypeInference  WorkloadType = "inference"
	WorkloadTypeStreaming  WorkloadType = "streaming"
	WorkloadTypeBatch      WorkloadType = "batch"
	WorkloadTypeRealtime   WorkloadType = "realtime"
)

// WorkloadPriority represents workload priority
type WorkloadPriority int

const (
	PriorityLow WorkloadPriority = iota
	PriorityMedium
	PriorityHigh
	PriorityCritical
)

// EdgeWorkload represents a workload to be scheduled on edge
type EdgeWorkload struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Type              WorkloadType           `json:"type"`
	Priority          WorkloadPriority       `json:"priority"`
	Requirements      ResourceRequirements   `json:"requirements"`
	Constraints       PlacementConstraints   `json:"constraints"`
	UserLocation      GeographicLocation     `json:"user_location"`
	LatencyTarget     float64                `json:"latency_target_ms"`
	DataSize          uint64                 `json:"data_size_bytes"`
	State             WorkloadState          `json:"state"`
	AssignedNode      string                 `json:"assigned_node,omitempty"`
	MigrationHistory  []MigrationEvent       `json:"migration_history,omitempty"`
	CreatedAt         time.Time              `json:"created_at"`
	ScheduledAt       *time.Time             `json:"scheduled_at,omitempty"`
	StartedAt         *time.Time             `json:"started_at,omitempty"`
	CompletedAt       *time.Time             `json:"completed_at,omitempty"`
	Metadata          map[string]interface{} `json:"metadata"`
	mu                sync.RWMutex
}

// WorkloadState represents the state of a workload
type WorkloadState string

const (
	WorkloadStatePending    WorkloadState = "pending"
	WorkloadStateScheduling WorkloadState = "scheduling"
	WorkloadStateScheduled  WorkloadState = "scheduled"
	WorkloadStateRunning    WorkloadState = "running"
	WorkloadStateMigrating  WorkloadState = "migrating"
	WorkloadStateCompleted  WorkloadState = "completed"
	WorkloadStateFailed     WorkloadState = "failed"
)

// ResourceRequirements defines resource requirements for a workload
type ResourceRequirements struct {
	CPU          float64                `json:"cpu_cores"`
	Memory       uint64                 `json:"memory_bytes"`
	Storage      uint64                 `json:"storage_bytes"`
	NetworkBW    float64                `json:"network_mbps"`
	GPU          int                    `json:"gpu_count,omitempty"`
	Accelerators []AcceleratorRequirement `json:"accelerators,omitempty"`
}

// AcceleratorRequirement represents accelerator requirements
type AcceleratorRequirement struct {
	Type  string  `json:"type"`
	Model string  `json:"model,omitempty"`
	Count int     `json:"count"`
}

// PlacementConstraints defines placement constraints
type PlacementConstraints struct {
	NodeTypes        []NodeType             `json:"node_types,omitempty"`
	RequiredLabels   map[string]string      `json:"required_labels,omitempty"`
	PreferredLabels  map[string]string      `json:"preferred_labels,omitempty"`
	AntiAffinity     []string               `json:"anti_affinity,omitempty"`
	Affinity         []string               `json:"affinity,omitempty"`
	MaxLatency       float64                `json:"max_latency_ms,omitempty"`
	GeoProximity     *GeographicLocation    `json:"geo_proximity,omitempty"`
	MaxDistance      float64                `json:"max_distance_km,omitempty"`
	DataLocality     []string               `json:"data_locality,omitempty"`
	SecurityZone     string                 `json:"security_zone,omitempty"`
}

// MigrationEvent represents a workload migration event
type MigrationEvent struct {
	Timestamp    time.Time `json:"timestamp"`
	FromNode     string    `json:"from_node"`
	ToNode       string    `json:"to_node"`
	Reason       string    `json:"reason"`
	Duration     time.Duration `json:"duration"`
	DataTransfer uint64    `json:"data_transfer_bytes"`
}

// SchedulingDecision represents a scheduling decision
type SchedulingDecision struct {
	WorkloadID   string                 `json:"workload_id"`
	NodeID       string                 `json:"node_id"`
	Score        float64                `json:"score"`
	Latency      float64                `json:"latency_ms"`
	Reasons      []string               `json:"reasons"`
	Alternatives []AlternativePlacement `json:"alternatives,omitempty"`
}

// AlternativePlacement represents an alternative placement option
type AlternativePlacement struct {
	NodeID  string  `json:"node_id"`
	Score   float64 `json:"score"`
	Latency float64 `json:"latency_ms"`
}

// EdgeScheduler manages edge workload scheduling
type EdgeScheduler struct {
	nodeManager      *NodeManager
	workloads        sync.Map // map[string]*EdgeWorkload
	pendingQueue     *PriorityQueue
	migrationManager *MigrationManager
	predictor        *LatencyPredictor
	optimizer        *PlacementOptimizer
	metrics          *SchedulerMetrics
	config           *SchedulerConfig
	ctx              context.Context
	cancel           context.CancelFunc
	wg               sync.WaitGroup
	mu               sync.RWMutex
}

// SchedulerConfig contains scheduler configuration
type SchedulerConfig struct {
	SchedulingInterval    time.Duration
	MigrationThreshold    float64
	LatencyWeight        float64
	ResourceWeight       float64
	LocalityWeight       float64
	EnableMigration      bool
	EnablePrediction     bool
	MaxMigrationsPerHour int
	PreferEdgeNodes      bool
	FallbackToDatacenter bool
}

// PriorityQueue manages workloads by priority
type PriorityQueue struct {
	items    []*QueueItem
	mu       sync.RWMutex
	notEmpty *sync.Cond
}

// QueueItem represents an item in the priority queue
type QueueItem struct {
	Workload  *EdgeWorkload
	Priority  float64
	Timestamp time.Time
}

// MigrationManager handles workload migration
type MigrationManager struct {
	migrations      sync.Map // Active migrations
	history         []MigrationEvent
	migrationRate   float64
	maxConcurrent   int
	mu              sync.RWMutex
}

// LatencyPredictor predicts network latency
type LatencyPredictor struct {
	measurements sync.Map // Historical measurements
	model        *LatencyModel
	mu           sync.RWMutex
}

// LatencyModel represents a latency prediction model
type LatencyModel struct {
	BaseLatency   float64
	DistanceFactor float64
	HopPenalty    float64
	Congestion    map[string]float64
}

// PlacementOptimizer optimizes workload placement
type PlacementOptimizer struct {
	strategy     PlacementStrategy
	constraints  []GlobalConstraint
	objectives   []OptimizationObjective
}

// PlacementStrategy represents a placement strategy
type PlacementStrategy string

const (
	StrategyLatencyFirst   PlacementStrategy = "latency_first"
	StrategyResourceFirst  PlacementStrategy = "resource_first"
	StrategyBalanced       PlacementStrategy = "balanced"
	StrategyGeoProximity   PlacementStrategy = "geo_proximity"
	StrategyDataLocality   PlacementStrategy = "data_locality"
)

// GlobalConstraint represents a global constraint
type GlobalConstraint interface {
	Validate(workload *EdgeWorkload, node *EdgeNode) bool
}

// OptimizationObjective represents an optimization objective
type OptimizationObjective interface {
	Score(workload *EdgeWorkload, node *EdgeNode) float64
}

// SchedulerMetrics tracks scheduler metrics
type SchedulerMetrics struct {
	schedulingLatency   prometheus.Histogram
	placementScore      prometheus.Histogram
	migrationCount      prometheus.Counter
	workloadLatency     *prometheus.HistogramVec
	schedulingFailures  prometheus.Counter
	queueLength         prometheus.Gauge
}

// NewEdgeScheduler creates a new edge scheduler
func NewEdgeScheduler(nodeManager *NodeManager, config *SchedulerConfig) *EdgeScheduler {
	ctx, cancel := context.WithCancel(context.Background())

	scheduler := &EdgeScheduler{
		nodeManager:      nodeManager,
		pendingQueue:     NewPriorityQueue(),
		migrationManager: NewMigrationManager(),
		predictor:        NewLatencyPredictor(),
		optimizer:        NewPlacementOptimizer(StrategyBalanced),
		metrics:          NewSchedulerMetrics(),
		config:           config,
		ctx:              ctx,
		cancel:           cancel,
	}

	// Start scheduling loop
	scheduler.wg.Add(2)
	go scheduler.schedulingLoop()
	go scheduler.migrationLoop()

	return scheduler
}

// NewPriorityQueue creates a new priority queue
func NewPriorityQueue() *PriorityQueue {
	pq := &PriorityQueue{
		items: make([]*QueueItem, 0),
	}
	pq.notEmpty = sync.NewCond(&pq.mu)
	return pq
}

// NewMigrationManager creates a new migration manager
func NewMigrationManager() *MigrationManager {
	return &MigrationManager{
		history:       make([]MigrationEvent, 0),
		maxConcurrent: 5,
	}
}

// NewLatencyPredictor creates a new latency predictor
func NewLatencyPredictor() *LatencyPredictor {
	return &LatencyPredictor{
		model: &LatencyModel{
			BaseLatency:    1.0,
			DistanceFactor: 0.05,
			HopPenalty:     2.0,
			Congestion:     make(map[string]float64),
		},
	}
}

// NewPlacementOptimizer creates a new placement optimizer
func NewPlacementOptimizer(strategy PlacementStrategy) *PlacementOptimizer {
	return &PlacementOptimizer{
		strategy:    strategy,
		constraints: []GlobalConstraint{},
		objectives:  []OptimizationObjective{},
	}
}

// NewSchedulerMetrics creates new scheduler metrics
func NewSchedulerMetrics() *SchedulerMetrics {
	return &SchedulerMetrics{
		schedulingLatency: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "edge_scheduling_duration_seconds",
				Help:    "Time taken to schedule workloads",
				Buckets: prometheus.DefBuckets,
			},
		),
		placementScore: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "edge_placement_score",
				Help:    "Placement optimization score",
				Buckets: []float64{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
			},
		),
		migrationCount: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_migration_total",
				Help: "Total number of workload migrations",
			},
		),
		workloadLatency: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "edge_workload_latency_milliseconds",
				Help:    "Workload to node latency",
				Buckets: []float64{1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000},
			},
			[]string{"workload_type", "node_type"},
		),
		schedulingFailures: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_scheduling_failures_total",
				Help: "Total number of scheduling failures",
			},
		),
		queueLength: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_queue_length",
				Help: "Number of workloads in scheduling queue",
			},
		),
	}
}

// ScheduleWorkload schedules a workload to an edge node
func (s *EdgeScheduler) ScheduleWorkload(workload *EdgeWorkload) (*SchedulingDecision, error) {
	start := time.Now()
	defer func() {
		s.metrics.schedulingLatency.Observe(time.Since(start).Seconds())
	}()

	// Validate workload
	if err := s.validateWorkload(workload); err != nil {
		s.metrics.schedulingFailures.Inc()
		return nil, fmt.Errorf("workload validation failed: %w", err)
	}

	// Add to pending queue
	s.pendingQueue.Push(workload)
	s.metrics.queueLength.Set(float64(s.pendingQueue.Len()))

	// Find optimal placement
	decision, err := s.findOptimalPlacement(workload)
	if err != nil {
		s.metrics.schedulingFailures.Inc()
		return nil, fmt.Errorf("placement failed: %w", err)
	}

	// Assign workload to node
	if err := s.assignWorkload(workload, decision.NodeID); err != nil {
		s.metrics.schedulingFailures.Inc()
		return nil, fmt.Errorf("assignment failed: %w", err)
	}

	// Update metrics
	s.metrics.placementScore.Observe(decision.Score)
	s.metrics.workloadLatency.WithLabelValues(
		string(workload.Type),
		"edge",
	).Observe(decision.Latency)

	return decision, nil
}

// RescheduleWorkload reschedules an existing workload
func (s *EdgeScheduler) RescheduleWorkload(workloadID string) error {
	workloadInterface, exists := s.workloads.Load(workloadID)
	if !exists {
		return fmt.Errorf("workload %s not found", workloadID)
	}

	workload := workloadInterface.(*EdgeWorkload)

	// Reset state
	workload.mu.Lock()
	workload.State = WorkloadStatePending
	workload.AssignedNode = ""
	workload.mu.Unlock()

	// Schedule again
	_, err := s.ScheduleWorkload(workload)
	return err
}

// MigrateWorkload migrates a workload to a different node
func (s *EdgeScheduler) MigrateWorkload(workloadID string, targetNodeID string) error {
	if !s.config.EnableMigration {
		return fmt.Errorf("migration is disabled")
	}

	workloadInterface, exists := s.workloads.Load(workloadID)
	if !exists {
		return fmt.Errorf("workload %s not found", workloadID)
	}

	workload := workloadInterface.(*EdgeWorkload)

	// Validate target node
	targetNode, err := s.nodeManager.GetNode(targetNodeID)
	if err != nil {
		return fmt.Errorf("target node not found: %w", err)
	}

	// Check if migration is beneficial
	if !s.shouldMigrate(workload, targetNode) {
		return fmt.Errorf("migration not beneficial")
	}

	// Perform migration
	migration := &MigrationEvent{
		Timestamp: time.Now(),
		FromNode:  workload.AssignedNode,
		ToNode:    targetNodeID,
		Reason:    "manual",
	}

	if err := s.migrationManager.Migrate(workload, targetNode, migration); err != nil {
		return fmt.Errorf("migration failed: %w", err)
	}

	// Update metrics
	s.metrics.migrationCount.Inc()

	return nil
}

// findOptimalPlacement finds the optimal node for a workload
func (s *EdgeScheduler) findOptimalPlacement(workload *EdgeWorkload) (*SchedulingDecision, error) {
	nodes := s.nodeManager.GetActiveNodes()
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no active nodes available")
	}

	// Score each node
	type nodeScore struct {
		node    *EdgeNode
		score   float64
		latency float64
	}

	scores := make([]nodeScore, 0, len(nodes))

	for _, node := range nodes {
		// Check if node meets requirements
		if !s.meetsRequirements(workload, node) {
			continue
		}

		// Check constraints
		if !s.satisfiesConstraints(workload, node) {
			continue
		}

		// Calculate score
		score := s.calculatePlacementScore(workload, node)
		latency := s.predictor.PredictLatency(workload, node)

		scores = append(scores, nodeScore{
			node:    node,
			score:   score,
			latency: latency,
		})
	}

	if len(scores) == 0 {
		return nil, fmt.Errorf("no suitable nodes found")
	}

	// Sort by score (descending)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Create decision
	best := scores[0]
	decision := &SchedulingDecision{
		WorkloadID: workload.ID,
		NodeID:     best.node.ID,
		Score:      best.score,
		Latency:    best.latency,
		Reasons:    s.explainDecision(workload, best.node),
	}

	// Add alternatives
	for i := 1; i < len(scores) && i < 4; i++ {
		decision.Alternatives = append(decision.Alternatives, AlternativePlacement{
			NodeID:  scores[i].node.ID,
			Score:   scores[i].score,
			Latency: scores[i].latency,
		})
	}

	return decision, nil
}

// calculatePlacementScore calculates the placement score for a node
func (s *EdgeScheduler) calculatePlacementScore(workload *EdgeWorkload, node *EdgeNode) float64 {
	var score float64

	// Latency score (lower is better)
	latency := s.predictor.PredictLatency(workload, node)
	latencyScore := 100.0 * math.Exp(-latency/workload.LatencyTarget)
	score += latencyScore * s.config.LatencyWeight

	// Resource availability score
	resourceScore := s.calculateResourceScore(workload, node)
	score += resourceScore * s.config.ResourceWeight

	// Data locality score
	localityScore := s.calculateLocalityScore(workload, node)
	score += localityScore * s.config.LocalityWeight

	// Geographic proximity score
	if workload.Constraints.GeoProximity != nil {
		distance := GetDistance(
			workload.Constraints.GeoProximity.Latitude,
			workload.Constraints.GeoProximity.Longitude,
			node.Location.Latitude,
			node.Location.Longitude,
		)
		proximityScore := 100.0 * math.Exp(-distance/100.0)
		score += proximityScore * 0.2
	}

	// Node type preference
	if s.config.PreferEdgeNodes && node.Type == NodeTypeCompute {
		score += 10.0
	}

	return score
}

// calculateResourceScore calculates resource availability score
func (s *EdgeScheduler) calculateResourceScore(workload *EdgeWorkload, node *EdgeNode) float64 {
	var score float64

	// CPU score
	cpuAvailable := float64(node.Resources.CPU.Cores) * (100 - node.Resources.CPU.Usage) / 100
	cpuScore := math.Min(100, (cpuAvailable/workload.Requirements.CPU)*100)
	score += cpuScore * 0.3

	// Memory score
	memAvailable := node.Resources.Memory.Available
	memScore := math.Min(100, (float64(memAvailable)/float64(workload.Requirements.Memory))*100)
	score += memScore * 0.3

	// Storage score
	storageAvailable := node.Resources.Storage.Available
	storageScore := math.Min(100, (float64(storageAvailable)/float64(workload.Requirements.Storage))*100)
	score += storageScore * 0.2

	// Network score
	networkAvailable := node.Resources.Network.BandwidthMbps
	networkScore := math.Min(100, (networkAvailable/workload.Requirements.NetworkBW)*100)
	score += networkScore * 0.2

	return score
}

// calculateLocalityScore calculates data locality score
func (s *EdgeScheduler) calculateLocalityScore(workload *EdgeWorkload, node *EdgeNode) float64 {
	if len(workload.Constraints.DataLocality) == 0 {
		return 50.0 // Neutral score if no locality requirements
	}

	var localData int
	for _, dataID := range workload.Constraints.DataLocality {
		// Check if data is available on or near this node
		if s.isDataLocal(dataID, node.ID) {
			localData++
		}
	}

	return 100.0 * float64(localData) / float64(len(workload.Constraints.DataLocality))
}

// meetsRequirements checks if a node meets workload requirements
func (s *EdgeScheduler) meetsRequirements(workload *EdgeWorkload, node *EdgeNode) bool {
	// Check CPU
	cpuAvailable := float64(node.Resources.CPU.Cores) * (100 - node.Resources.CPU.Usage) / 100
	if cpuAvailable < workload.Requirements.CPU {
		return false
	}

	// Check memory
	if node.Resources.Memory.Available < workload.Requirements.Memory {
		return false
	}

	// Check storage
	if node.Resources.Storage.Available < workload.Requirements.Storage {
		return false
	}

	// Check network bandwidth
	if node.Resources.Network.BandwidthMbps < workload.Requirements.NetworkBW {
		return false
	}

	// Check GPU requirements
	if workload.Requirements.GPU > 0 && len(node.Resources.GPU) < workload.Requirements.GPU {
		return false
	}

	return true
}

// satisfiesConstraints checks if a node satisfies placement constraints
func (s *EdgeScheduler) satisfiesConstraints(workload *EdgeWorkload, node *EdgeNode) bool {
	constraints := workload.Constraints

	// Check node type constraints
	if len(constraints.NodeTypes) > 0 {
		found := false
		for _, nt := range constraints.NodeTypes {
			if node.Type == nt {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check required labels
	for key, value := range constraints.RequiredLabels {
		if nodeValue, exists := node.Metadata[key]; !exists || nodeValue != value {
			return false
		}
	}

	// Check anti-affinity
	for _, antiWorkloadID := range constraints.AntiAffinity {
		if s.isColocated(antiWorkloadID, node.ID) {
			return false
		}
	}

	// Check max latency
	if constraints.MaxLatency > 0 {
		latency := s.predictor.PredictLatency(workload, node)
		if latency > constraints.MaxLatency {
			return false
		}
	}

	// Check max distance
	if constraints.MaxDistance > 0 && constraints.GeoProximity != nil {
		distance := GetDistance(
			constraints.GeoProximity.Latitude,
			constraints.GeoProximity.Longitude,
			node.Location.Latitude,
			node.Location.Longitude,
		)
		if distance > constraints.MaxDistance {
			return false
		}
	}

	return true
}

// PredictLatency predicts latency between workload and node
func (p *LatencyPredictor) PredictLatency(workload *EdgeWorkload, node *EdgeNode) float64 {
	// Base network latency
	latency := p.model.BaseLatency

	// Distance-based latency
	if workload.UserLocation.Latitude != 0 && workload.UserLocation.Longitude != 0 {
		distance := GetDistance(
			workload.UserLocation.Latitude,
			workload.UserLocation.Longitude,
			node.Location.Latitude,
			node.Location.Longitude,
		)
		latency += distance * p.model.DistanceFactor
	}

	// Network hop penalty
	hops := p.estimateNetworkHops(workload.UserLocation, node.Location)
	latency += float64(hops) * p.model.HopPenalty

	// Congestion factor
	if congestion, exists := p.model.Congestion[node.ID]; exists {
		latency *= (1 + congestion)
	}

	// Processing latency based on workload type
	switch workload.Type {
	case WorkloadTypeRealtime:
		latency += 0.5 // Minimal processing
	case WorkloadTypeInference:
		latency += 2.0 // ML inference overhead
	case WorkloadTypeStreaming:
		latency += 1.0 // Streaming buffer
	default:
		latency += 1.5 // Default processing
	}

	return latency
}

// estimateNetworkHops estimates the number of network hops
func (p *LatencyPredictor) estimateNetworkHops(src, dst GeographicLocation) int {
	// Simplified estimation based on geographic distance
	distance := GetDistance(src.Latitude, src.Longitude, dst.Latitude, dst.Longitude)

	if distance < 50 { // Same city
		return 1
	} else if distance < 500 { // Same region
		return 3
	} else if distance < 2000 { // Same country
		return 5
	} else if distance < 5000 { // Same continent
		return 8
	}
	return 12 // Inter-continental
}

// Migrate performs workload migration
func (m *MigrationManager) Migrate(workload *EdgeWorkload, targetNode *EdgeNode, event *MigrationEvent) error {
	// Check concurrent migrations
	count := 0
	m.migrations.Range(func(_, _ interface{}) bool {
		count++
		return count < m.maxConcurrent
	})

	if count >= m.maxConcurrent {
		return fmt.Errorf("too many concurrent migrations")
	}

	// Start migration
	workload.mu.Lock()
	workload.State = WorkloadStateMigrating
	sourceNode := workload.AssignedNode
	workload.mu.Unlock()

	// Store active migration
	m.migrations.Store(workload.ID, event)
	defer m.migrations.Delete(workload.ID)

	// Simulate data transfer
	startTime := time.Now()
	transferDuration := m.calculateTransferTime(workload.DataSize, targetNode)
	time.Sleep(transferDuration) // Simulated transfer

	// Update workload assignment
	workload.mu.Lock()
	workload.AssignedNode = targetNode.ID
	workload.State = WorkloadStateRunning
	workload.MigrationHistory = append(workload.MigrationHistory, *event)
	workload.mu.Unlock()

	// Record migration
	event.Duration = time.Since(startTime)
	event.DataTransfer = workload.DataSize

	m.mu.Lock()
	m.history = append(m.history, *event)
	m.migrationRate = float64(len(m.history)) / time.Since(m.history[0].Timestamp).Hours()
	m.mu.Unlock()

	return nil
}

// calculateTransferTime calculates data transfer time
func (m *MigrationManager) calculateTransferTime(dataSize uint64, node *EdgeNode) time.Duration {
	// Calculate based on network bandwidth
	bandwidth := node.Resources.Network.BandwidthMbps * 1024 * 1024 / 8 // Convert to bytes/sec
	seconds := float64(dataSize) / bandwidth
	return time.Duration(seconds * float64(time.Second))
}

// Helper methods

func (s *EdgeScheduler) validateWorkload(workload *EdgeWorkload) error {
	if workload.ID == "" {
		return fmt.Errorf("workload ID is required")
	}
	if workload.Type == "" {
		return fmt.Errorf("workload type is required")
	}
	if workload.Requirements.CPU <= 0 {
		return fmt.Errorf("CPU requirement must be positive")
	}
	if workload.Requirements.Memory == 0 {
		return fmt.Errorf("memory requirement must be positive")
	}
	return nil
}

func (s *EdgeScheduler) assignWorkload(workload *EdgeWorkload, nodeID string) error {
	node, err := s.nodeManager.GetNode(nodeID)
	if err != nil {
		return err
	}

	workload.mu.Lock()
	defer workload.mu.Unlock()

	workload.AssignedNode = nodeID
	workload.State = WorkloadStateScheduled
	now := time.Now()
	workload.ScheduledAt = &now

	// Update node state
	node.mu.Lock()
	node.State = NodeStateBusy
	node.mu.Unlock()

	// Store workload
	s.workloads.Store(workload.ID, workload)

	return nil
}

func (s *EdgeScheduler) shouldMigrate(workload *EdgeWorkload, targetNode *EdgeNode) bool {
	// Check migration threshold
	currentLatency := s.predictor.PredictLatency(workload, nil)
	newLatency := s.predictor.PredictLatency(workload, targetNode)

	improvement := (currentLatency - newLatency) / currentLatency
	return improvement > s.config.MigrationThreshold
}

func (s *EdgeScheduler) isDataLocal(dataID string, nodeID string) bool {
	// Check if data is available locally
	// This would integrate with the caching layer
	return false // Placeholder
}

func (s *EdgeScheduler) isColocated(workloadID string, nodeID string) bool {
	if w, exists := s.workloads.Load(workloadID); exists {
		workload := w.(*EdgeWorkload)
		return workload.AssignedNode == nodeID
	}
	return false
}

func (s *EdgeScheduler) explainDecision(workload *EdgeWorkload, node *EdgeNode) []string {
	reasons := []string{}

	latency := s.predictor.PredictLatency(workload, node)
	reasons = append(reasons, fmt.Sprintf("Predicted latency: %.2fms", latency))

	if node.Type == NodeTypeCompute {
		reasons = append(reasons, "Compute-optimized node")
	}

	if node.Location.City == workload.UserLocation.City {
		reasons = append(reasons, "Same city as user")
	}

	return reasons
}

// Priority queue methods

func (pq *PriorityQueue) Push(workload *EdgeWorkload) {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	item := &QueueItem{
		Workload:  workload,
		Priority:  float64(workload.Priority),
		Timestamp: time.Now(),
	}

	pq.items = append(pq.items, item)
	pq.sortItems()
	pq.notEmpty.Signal()
}

func (pq *PriorityQueue) Pop() *EdgeWorkload {
	pq.mu.Lock()
	defer pq.mu.Unlock()

	for len(pq.items) == 0 {
		pq.notEmpty.Wait()
	}

	item := pq.items[0]
	pq.items = pq.items[1:]

	return item.Workload
}

func (pq *PriorityQueue) Len() int {
	pq.mu.RLock()
	defer pq.mu.RUnlock()
	return len(pq.items)
}

func (pq *PriorityQueue) sortItems() {
	sort.Slice(pq.items, func(i, j int) bool {
		if pq.items[i].Priority != pq.items[j].Priority {
			return pq.items[i].Priority > pq.items[j].Priority
		}
		return pq.items[i].Timestamp.Before(pq.items[j].Timestamp)
	})
}

// Scheduling loops

func (s *EdgeScheduler) schedulingLoop() {
	defer s.wg.Done()

	ticker := time.NewTicker(s.config.SchedulingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.processQueue()
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *EdgeScheduler) migrationLoop() {
	defer s.wg.Done()

	if !s.config.EnableMigration {
		return
	}

	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.checkMigrationOpportunities()
		case <-s.ctx.Done():
			return
		}
	}
}

func (s *EdgeScheduler) processQueue() {
	for s.pendingQueue.Len() > 0 {
		workload := s.pendingQueue.Pop()
		_, err := s.findOptimalPlacement(workload)
		if err != nil {
			// Re-queue if failed
			s.pendingQueue.Push(workload)
			break
		}
	}
}

func (s *EdgeScheduler) checkMigrationOpportunities() {
	s.workloads.Range(func(key, value interface{}) bool {
		workload := value.(*EdgeWorkload)

		if workload.State != WorkloadStateRunning {
			return true
		}

		// Check if better placement exists
		decision, err := s.findOptimalPlacement(workload)
		if err != nil {
			return true
		}

		if decision.NodeID != workload.AssignedNode && s.shouldMigrate(workload, nil) {
			s.MigrateWorkload(workload.ID, decision.NodeID)
		}

		return true
	})
}

// Stop stops the scheduler
func (s *EdgeScheduler) Stop() {
	s.cancel()
	s.wg.Wait()
}