package compute

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
)

// ComputeJob represents a distributed compute job
type ComputeJob struct {
	ID                    string                 `json:"id"`
	Name                  string                 `json:"name"`
	Description           string                 `json:"description"`
	Type                  JobType                `json:"type"`
	Status                JobStatus              `json:"status"`
	Priority              JobPriority            `json:"priority"`
	QueueName             string                 `json:"queue_name"`
	Command               []string               `json:"command"`
	Environment           map[string]string      `json:"environment"`
	Resources             ResourceRequirements   `json:"resources"`
	ExecutionConstraints  ExecutionConstraints   `json:"execution_constraints"`
	Constraints           []JobConstraint        `json:"constraints"`
	Tags                  map[string]string      `json:"tags"`
	Timeout               time.Duration          `json:"timeout"`
	Metadata              map[string]interface{} `json:"metadata"`
	SubmittedAt           time.Time              `json:"submitted_at"`
	QueuedAt              *time.Time             `json:"queued_at,omitempty"`
	SchedulingStartedAt   *time.Time             `json:"scheduling_started_at,omitempty"`
	SchedulingCompletedAt *time.Time             `json:"scheduling_completed_at,omitempty"`
	StartedAt             *time.Time             `json:"started_at,omitempty"`
	CompletedAt           *time.Time             `json:"completed_at,omitempty"`
	ClusterPlacements     []ClusterPlacement     `json:"cluster_placements"`
	Dependencies          []string               `json:"dependencies"`
	Progress              JobProgress            `json:"progress"`
	ErrorMessage          string                 `json:"error_message,omitempty"`
}

type JobType string

const (
	JobTypeBatch       JobType = "batch"
	JobTypeInteractive JobType = "interactive"
	JobTypeMPI         JobType = "mpi"
	JobTypeContainer   JobType = "container"
	JobTypeStream      JobType = "stream"
)

type JobStatus string

const (
	JobStatusPending    JobStatus = "pending"
	JobStatusQueued     JobStatus = "queued"
	JobStatusScheduling JobStatus = "scheduling"
	JobStatusRunning    JobStatus = "running"
	JobStatusCompleted  JobStatus = "completed"
	JobStatusFailed     JobStatus = "failed"
	JobStatusCancelled  JobStatus = "cancelled"
	JobStatusSuspended  JobStatus = "suspended"
)

type JobPriority int

const (
	PriorityLow JobPriority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
	PriorityUrgent
)

type ResourceRequirements struct {
	CPUCores    float64 `json:"cpu_cores"`
	MemoryGB    float64 `json:"memory_gb"`
	GPUCount    int     `json:"gpu_count"`
	GPUType     string  `json:"gpu_type,omitempty"`
	StorageGB   float64 `json:"storage_gb"`
	NetworkMbps float64 `json:"network_mbps"`
	DiskIOPS    int     `json:"disk_iops"`
	Preemptible bool    `json:"preemptible"`
	Dedicated   bool    `json:"dedicated"`
}

type ExecutionConstraints struct {
	Deadline          *time.Time     `json:"deadline,omitempty"`
	MaxRuntime        *time.Duration `json:"max_runtime,omitempty"`
	PreferredClusters []string       `json:"preferred_clusters,omitempty"`
	ExcludedClusters  []string       `json:"excluded_clusters,omitempty"`
	AffinityRules     []AffinityRule `json:"affinity_rules,omitempty"`
	SecurityTags      []string       `json:"security_tags,omitempty"`
	ComplianceReqs    []string       `json:"compliance_requirements,omitempty"`
}

// JobConstraint represents a job scheduling constraint for API compatibility
type JobConstraint struct {
	Type     string      `json:"type"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
}

type AffinityRule struct {
	Type     AffinityType `json:"type"`
	Target   string       `json:"target"`
	Strength int          `json:"strength"` // 1-10, higher is stronger
}

type AffinityType string

const (
	AffinityCluster     AffinityType = "cluster"
	AffinityNode        AffinityType = "node"
	AffinityDataCenter  AffinityType = "datacenter"
	AffinityZone        AffinityType = "zone"
	AntiAffinityCluster AffinityType = "anti-cluster"
	AntiAffinityNode    AffinityType = "anti-node"
)

type ClusterPlacement struct {
	ClusterID   string             `json:"cluster_id"`
	NodeIDs     []string           `json:"node_ids"`
	Resources   ResourceAllocation `json:"resources"`
	Status      PlacementStatus    `json:"status"`
	StartedAt   *time.Time         `json:"started_at,omitempty"`
	CompletedAt *time.Time         `json:"completed_at,omitempty"`
}

type ResourceAllocation struct {
	CPUCores    float64 `json:"cpu_cores"`
	MemoryGB    float64 `json:"memory_gb"`
	GPUCount    int     `json:"gpu_count"`
	StorageGB   float64 `json:"storage_gb"`
	NetworkMbps float64 `json:"network_mbps"`
}

type PlacementStatus string

const (
	PlacementPending   PlacementStatus = "pending"
	PlacementAllocated PlacementStatus = "allocated"
	PlacementRunning   PlacementStatus = "running"
	PlacementCompleted PlacementStatus = "completed"
	PlacementFailed    PlacementStatus = "failed"
)

type JobProgress struct {
	Percentage    float64            `json:"percentage"`
	TasksTotal    int                `json:"tasks_total"`
	TasksComplete int                `json:"tasks_complete"`
	TasksFailed   int                `json:"tasks_failed"`
	Metrics       map[string]float64 `json:"metrics,omitempty"`
}

// ComputeJobManager manages the lifecycle of compute jobs
type ComputeJobManager struct {
	mu                       sync.RWMutex
	jobs                     map[string]*ComputeJob
	jobQueue                 *JobQueue
	federationMgr            federation.Provider
	scheduler                scheduler.SchedulerInterface
	loadBalancer             LoadBalancer
	performanceOpt           *PerformanceOptimizer
	executionEngines         map[JobType]ExecutionEngine
	metrics                  *JobMetrics
	transactionalResourceMgr *TransactionalResourceManager

	// Dispatcher control
	dispatcherRunning bool
	dispatcherCtx     context.Context
	dispatcherCancel  context.CancelFunc
	dispatcherWg      sync.WaitGroup
}

// JobQueue manages priority-based job queuing
type JobQueue struct {
	mu       sync.RWMutex
	queues   map[JobPriority][]*ComputeJob
	policy   SchedulingPolicy
	backfill bool
}

type SchedulingPolicy string

const (
	PolicyFIFO      SchedulingPolicy = "fifo"
	PolicyPriority  SchedulingPolicy = "priority"
	PolicyFairShare SchedulingPolicy = "fair_share"
	PolicyBackfill  SchedulingPolicy = "backfill"
)

// ExecutionEngine interface for different job types
type ExecutionEngine interface {
	Execute(ctx context.Context, job *ComputeJob) error
	Monitor(ctx context.Context, job *ComputeJob) (*JobProgress, error)
	Cancel(ctx context.Context, job *ComputeJob) error
	Suspend(ctx context.Context, job *ComputeJob) error
	Resume(ctx context.Context, job *ComputeJob) error
}

// JobMetrics tracks job execution metrics
// JobMetrics tracks comprehensive job execution metrics - Comment 17
type JobMetrics struct {
	mu sync.RWMutex

	// Basic counters
	totalJobs     int64
	completedJobs int64
	failedJobs    int64
	cancelledJobs int64
	runningJobs   int64
	queuedJobs    int64
	suspendedJobs int64

	// Job type metrics
	jobTypeMetrics map[JobType]*JobTypeMetrics

	// Performance metrics
	avgExecutionTime  time.Duration
	avgQueueTime      time.Duration
	avgSchedulingTime time.Duration
	minExecutionTime  time.Duration
	maxExecutionTime  time.Duration
	executionTimeP95  time.Duration
	executionTimeP99  time.Duration

	// Resource metrics
	resourceUtilization map[string]float64
	resourceEfficiency  map[string]float64
	peakResourceUsage   map[string]float64
	resourceWaste       map[string]float64

	// Throughput and capacity
	throughputPerMinute float64
	throughputPerHour   float64
	dailyThroughput     float64
	peakThroughput      float64
	capacityUtilization float64

	// SLA and quality metrics
	slaCompliance   float64
	slaViolations   int64
	deadlinesMet    int64
	deadlinesMissed int64

	// Cluster and federation metrics
	clusterMetrics    map[string]*ClusterJobMetrics
	federationMetrics *FederationJobMetrics

	// Error and retry metrics
	errorsByType map[string]int64
	retriesByJob map[string]int64
	totalRetries int64

	// Time-based metrics (sliding windows)
	hourlyMetrics *TimeSeriesMetrics
	dailyMetrics  *TimeSeriesMetrics
	weeklyMetrics *TimeSeriesMetrics

	// Cost and efficiency
	costPerJob          map[JobType]float64
	costEfficiencyScore float64
	energyConsumption   map[string]float64

	// User and workload patterns
	userMetrics      map[string]*UserJobMetrics
	workloadPatterns *WorkloadPatterns

	// Predictive metrics
	queueTimePredictor     *QueueTimePredictor
	resourceDemandForecast *ResourceForecast

	// Historical tracking
	metricsHistory []MetricsSnapshot
	lastResetTime  time.Time
	startTime      time.Time
}

// Enhanced Metrics Support Types - Comment 17

// JobTypeMetrics tracks metrics per job type
type JobTypeMetrics struct {
	JobCount      int64              `json:"job_count"`
	SuccessRate   float64            `json:"success_rate"`
	AvgExecTime   time.Duration      `json:"avg_execution_time"`
	AvgQueueTime  time.Duration      `json:"avg_queue_time"`
	ResourceUsage map[string]float64 `json:"resource_usage"`
	TotalCost     float64            `json:"total_cost"`
}

// ClusterJobMetrics tracks job metrics per cluster
type ClusterJobMetrics struct {
	ClusterID     string             `json:"cluster_id"`
	JobsScheduled int64              `json:"jobs_scheduled"`
	JobsCompleted int64              `json:"jobs_completed"`
	JobsFailed    int64              `json:"jobs_failed"`
	AvgExecTime   time.Duration      `json:"avg_execution_time"`
	ResourceUtil  map[string]float64 `json:"resource_utilization"`
	LoadScore     float64            `json:"load_score"`
	LastUpdate    time.Time          `json:"last_update"`
}

// FederationJobMetrics tracks cross-cluster job metrics
type FederationJobMetrics struct {
	CrossClusterJobs     int64         `json:"cross_cluster_jobs"`
	MigrationCount       int64         `json:"migration_count"`
	MigrationSuccessRate float64       `json:"migration_success_rate"`
	NetworkLatency       time.Duration `json:"network_latency"`
	BandwidthUsage       float64       `json:"bandwidth_usage"`
	SyncOverhead         time.Duration `json:"sync_overhead"`
}

// UserJobMetrics tracks metrics per user
type UserJobMetrics struct {
	UserID          string             `json:"user_id"`
	JobsSubmitted   int64              `json:"jobs_submitted"`
	JobsCompleted   int64              `json:"jobs_completed"`
	ResourceQuota   map[string]float64 `json:"resource_quota"`
	ResourceUsed    map[string]float64 `json:"resource_used"`
	CostAccumulated float64            `json:"cost_accumulated"`
	LastActivity    time.Time          `json:"last_activity"`
}

// WorkloadPatterns analyzes workload trends
type WorkloadPatterns struct {
	PeakHours       []int               `json:"peak_hours"`
	JobTypeTrends   map[JobType]float64 `json:"job_type_trends"`
	SeasonalPattern map[string]float64  `json:"seasonal_pattern"`
	GrowthRate      float64             `json:"growth_rate"`
	Cyclical        bool                `json:"cyclical"`
}

// TimeSeriesMetrics for sliding window analysis
type TimeSeriesMetrics struct {
	Timestamps    []time.Time   `json:"timestamps"`
	JobCounts     []int64       `json:"job_counts"`
	Throughputs   []float64     `json:"throughputs"`
	ResourceUtils []float64     `json:"resource_utilizations"`
	WindowSize    time.Duration `json:"window_size"`
}

// QueueTimePredictor for predictive analytics
type QueueTimePredictor struct {
	Model           interface{}              `json:"model"`
	Accuracy        float64                  `json:"accuracy"`
	LastTrained     time.Time                `json:"last_trained"`
	PredictionCache map[string]time.Duration `json:"prediction_cache"`
}

// ResourceForecast for capacity planning
type ResourceForecast struct {
	CPUForecast     []float64     `json:"cpu_forecast"`
	MemoryForecast  []float64     `json:"memory_forecast"`
	StorageForecast []float64     `json:"storage_forecast"`
	NetworkForecast []float64     `json:"network_forecast"`
	TimeHorizon     time.Duration `json:"time_horizon"`
	Confidence      float64       `json:"confidence"`
}

// MetricsSnapshot for historical tracking
type MetricsSnapshot struct {
	Timestamp      time.Time          `json:"timestamp"`
	TotalJobs      int64              `json:"total_jobs"`
	Throughput     float64            `json:"throughput"`
	ResourceUtil   map[string]float64 `json:"resource_utilization"`
	SLACompliance  float64            `json:"sla_compliance"`
	CostEfficiency float64            `json:"cost_efficiency"`
}

// JobLifecycleEvent tracks job state transitions for metrics
type JobLifecycleEvent struct {
	JobID     string        `json:"job_id"`
	Event     string        `json:"event"`
	Timestamp time.Time     `json:"timestamp"`
	PrevState JobStatus     `json:"previous_state"`
	NewState  JobStatus     `json:"new_state"`
	Duration  time.Duration `json:"duration"`
	ClusterID string        `json:"cluster_id,omitempty"`
	ErrorMsg  string        `json:"error_message,omitempty"`
}

// NewComputeJobManager creates a new compute job manager
func NewComputeJobManager(federationMgr federation.Provider,
	scheduler scheduler.SchedulerInterface) *ComputeJobManager {

	mgr := &ComputeJobManager{
		jobs:             make(map[string]*ComputeJob),
		jobQueue:         NewJobQueue(PolicyPriority, true),
		federationMgr:    federationMgr,
		scheduler:        scheduler,
		loadBalancer:     NewComputeJobLoadBalancer(federationMgr),
		executionEngines: make(map[JobType]ExecutionEngine),
		metrics:          NewJobMetrics(),
	}

	// Register execution engines
	mgr.executionEngines[JobTypeBatch] = NewBatchExecutionEngine()
	mgr.executionEngines[JobTypeContainer] = NewContainerExecutionEngine()
	mgr.executionEngines[JobTypeMPI] = NewMPIExecutionEngine()
	mgr.executionEngines[JobTypeInteractive] = NewInteractiveExecutionEngine()
	mgr.executionEngines[JobTypeStream] = NewStreamExecutionEngine()

	// Initialize dispatcher context
	mgr.dispatcherCtx, mgr.dispatcherCancel = context.WithCancel(context.Background())

	return mgr
}

// NewJobQueue creates a new job queue
func NewJobQueue(policy SchedulingPolicy, backfill bool) *JobQueue {
	return &JobQueue{
		queues:   make(map[JobPriority][]*ComputeJob),
		policy:   policy,
		backfill: backfill,
	}
}

// NewJobMetrics creates comprehensive job metrics tracker - Comment 17
func NewJobMetrics() *JobMetrics {
	now := time.Now()
	return &JobMetrics{
		// Initialize maps
		jobTypeMetrics:      make(map[JobType]*JobTypeMetrics),
		resourceUtilization: make(map[string]float64),
		resourceEfficiency:  make(map[string]float64),
		peakResourceUsage:   make(map[string]float64),
		resourceWaste:       make(map[string]float64),
		clusterMetrics:      make(map[string]*ClusterJobMetrics),
		errorsByType:        make(map[string]int64),
		retriesByJob:        make(map[string]int64),
		costPerJob:          make(map[JobType]float64),
		energyConsumption:   make(map[string]float64),
		userMetrics:         make(map[string]*UserJobMetrics),

		// Initialize time-based metrics
		hourlyMetrics: &TimeSeriesMetrics{WindowSize: time.Hour},
		dailyMetrics:  &TimeSeriesMetrics{WindowSize: 24 * time.Hour},
		weeklyMetrics: &TimeSeriesMetrics{WindowSize: 7 * 24 * time.Hour},

		// Initialize federation metrics
		federationMetrics: &FederationJobMetrics{},

		// Initialize workload patterns
		workloadPatterns: &WorkloadPatterns{
			PeakHours:       make([]int, 24),
			JobTypeTrends:   make(map[JobType]float64),
			SeasonalPattern: make(map[string]float64),
		},

		// Initialize predictive components
		queueTimePredictor: &QueueTimePredictor{
			PredictionCache: make(map[string]time.Duration),
		},
		resourceDemandForecast: &ResourceForecast{
			TimeHorizon: 24 * time.Hour,
		},

		// Initialize time tracking
		metricsHistory: make([]MetricsSnapshot, 0),
		lastResetTime:  now,
		startTime:      now,

		// Initialize performance metrics to reasonable defaults
		minExecutionTime: time.Hour * 24, // Will be updated with first job
		maxExecutionTime: 0,
	}
}

// StartDispatcher starts the job dispatcher loop
func (m *ComputeJobManager) StartDispatcher() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.dispatcherRunning {
		return fmt.Errorf("dispatcher is already running")
	}

	m.dispatcherRunning = true
	m.dispatcherWg.Add(1)

	go m.dispatcherLoop()

	return nil
}

// StopDispatcher stops the job dispatcher loop
func (m *ComputeJobManager) StopDispatcher() error {
	m.mu.Lock()
	if !m.dispatcherRunning {
		m.mu.Unlock()
		return fmt.Errorf("dispatcher is not running")
	}

	m.dispatcherRunning = false
	m.dispatcherCancel()
	m.mu.Unlock()

	// Wait for dispatcher to finish
	m.dispatcherWg.Wait()

	return nil
}

// dispatcherLoop is the main dispatcher loop that processes jobs from the queue
func (m *ComputeJobManager) dispatcherLoop() {
	defer m.dispatcherWg.Done()

	ticker := time.NewTicker(1 * time.Second) // Check queue every second
	defer ticker.Stop()

	for {
		select {
		case <-m.dispatcherCtx.Done():
			return
		case <-ticker.C:
			m.processNextJob()
		}
	}
}

// processNextJob processes the next job from the queue
func (m *ComputeJobManager) processNextJob() {
	job := m.jobQueue.Dequeue()
	if job == nil {
		return // No jobs in queue
	}

	// Process the job in a separate goroutine
	go m.processJobFromDispatcher(job)
}

// processJobFromDispatcher processes a job dequeued by the dispatcher
func (m *ComputeJobManager) processJobFromDispatcher(job *ComputeJob) {
	if job == nil {
		return
	}

	ctx := m.dispatcherCtx

	// Check if job was cancelled while in queue
	m.mu.RLock()
	currentJob, exists := m.jobs[job.ID]
	if !exists || currentJob.Status == JobStatusCancelled {
		m.mu.RUnlock()
		return
	}
	m.mu.RUnlock()

	// Try to schedule and execute the job
	if err := m.scheduleJob(ctx, job); err != nil {
		m.mu.Lock()
		job.Status = JobStatusFailed
		job.ErrorMessage = err.Error()
		now := time.Now()
		job.CompletedAt = &now
		m.mu.Unlock()
		m.metrics.IncrementFailedJobs()
		return
	}

	// Job is now scheduled and running, monitor it
	m.monitorJob(ctx, job)
}

// SubmitJob submits a new compute job and returns job ID
func (m *ComputeJobManager) SubmitJob(ctx context.Context, job *ComputeJob) (string, error) {
	if job.ID == "" {
		job.ID = uuid.New().String()
	}

	// Validate job
	if err := m.validateJob(job); err != nil {
		return "", fmt.Errorf("job validation failed: %w", err)
	}

	// Set initial status and timestamp
	now := time.Now()
	job.Status = JobStatusPending
	job.SubmittedAt = now
	job.QueuedAt = &now

	m.mu.Lock()
	m.jobs[job.ID] = job
	m.mu.Unlock()

	// Add to queue - the dispatcher will pick it up
	m.jobQueue.Enqueue(job)

	m.metrics.IncrementTotalJobs()
	m.metrics.IncrementQueuedJobs()

	return job.ID, nil
}

// GetJob retrieves a job by ID
func (m *ComputeJobManager) GetJob(ctx context.Context, jobID string) (*ComputeJob, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	job, exists := m.jobs[jobID]
	if !exists {
		return nil, fmt.Errorf("job %s not found", jobID)
	}

	return job, nil
}

// ListJobs lists jobs with optional filtering
func (m *ComputeJobManager) ListJobs(ctx context.Context, filters JobFilters) ([]*ComputeJob, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var result []*ComputeJob
	for _, job := range m.jobs {
		if filters.Matches(job) {
			result = append(result, job)
		}
	}

	return result, nil
}

// CancelJob cancels a running or queued job
func (m *ComputeJobManager) CancelJob(ctx context.Context, jobID string) error {
	m.mu.Lock()
	job, exists := m.jobs[jobID]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("job %s not found", jobID)
	}

	if job.Status == JobStatusCompleted || job.Status == JobStatusFailed || job.Status == JobStatusCancelled {
		m.mu.Unlock()
		return fmt.Errorf("job %s is already in terminal state: %s", jobID, job.Status)
	}

	// Store original status before changing it
	originalStatus := job.Status
	m.mu.Unlock()

	var cancelErr error

	// Cancel execution if running - do this BEFORE setting status
	if originalStatus == JobStatusRunning {
		if engine, exists := m.executionEngines[job.Type]; exists {
			cancelErr = engine.Cancel(ctx, job)
		}
	}

	// Remove from queue if queued - do this BEFORE setting status
	if originalStatus == JobStatusQueued || originalStatus == JobStatusPending {
		m.jobQueue.Remove(jobID)
	}

	// Only set status to cancelled after actual cancellation operations
	m.mu.Lock()
	// Re-check job still exists and hasn't changed to terminal state
	if currentJob, exists := m.jobs[jobID]; exists {
		if currentJob.Status != JobStatusCompleted && currentJob.Status != JobStatusFailed {
			currentJob.Status = JobStatusCancelled
			now := time.Now()
			currentJob.CompletedAt = &now
		}
	}
	m.mu.Unlock()

	return cancelErr
}

// validateJob validates job specification
func (m *ComputeJobManager) validateJob(job *ComputeJob) error {
	if job.Name == "" {
		return fmt.Errorf("job name is required")
	}

	if job.Resources.CPUCores <= 0 {
		return fmt.Errorf("CPU cores must be positive")
	}

	if job.Resources.MemoryGB <= 0 {
		return fmt.Errorf("memory must be positive")
	}

	if job.Type == "" {
		return fmt.Errorf("job type is required")
	}

	// Validate execution engine exists
	if _, exists := m.executionEngines[job.Type]; !exists {
		return fmt.Errorf("unsupported job type: %s", job.Type)
	}

	// Validate constraints
	if job.Constraints.Deadline != nil && job.Constraints.Deadline.Before(time.Now()) {
		return fmt.Errorf("deadline cannot be in the past")
	}

	return nil
}

// scheduleJob schedules a job across clusters
func (m *ComputeJobManager) scheduleJob(ctx context.Context, job *ComputeJob) error {
	// Record scheduling start time for metrics
	schedulingStartTime := time.Now()

	// Update status and record scheduling start
	m.mu.Lock()
	job.Status = JobStatusScheduling
	job.SchedulingStartedAt = &schedulingStartTime
	m.mu.Unlock()

	// Get optimal cluster placement
	placement, err := m.loadBalancer.SelectOptimalClusters(ctx, job)
	if err != nil {
		return fmt.Errorf("cluster selection failed: %w", err)
	}

	// Use transactional resource allocation for rollback safety
	transaction, err := m.transactionalResourceMgr.BeginTransaction(ctx, job.ID, 5*time.Minute)
	if err != nil {
		return fmt.Errorf("failed to begin resource transaction: %w", err)
	}

	// Allocate resources transactionally
	for _, clusterPlacement := range placement {
		// Allocate CPU resources through transactional manager
		if err := m.transactionalResourceMgr.AllocateInTransaction(ctx, transaction.ID, clusterPlacement.ClusterID, "cpu", clusterPlacement.Resources.CPUCores, "cores", nil); err != nil {
			// Rollback all previous allocations
			m.transactionalResourceMgr.RollbackTransaction(ctx, transaction.ID)
			return fmt.Errorf("failed to allocate CPU resources on cluster %s: %w", clusterPlacement.ClusterID, err)
		}

		// Allocate memory resources through transactional manager
		if err := m.transactionalResourceMgr.AllocateInTransaction(ctx, transaction.ID, clusterPlacement.ClusterID, "memory", clusterPlacement.Resources.MemoryGB, "GB", nil); err != nil {
			// Rollback all previous allocations
			m.transactionalResourceMgr.RollbackTransaction(ctx, transaction.ID)
			return fmt.Errorf("failed to allocate memory resources on cluster %s: %w", clusterPlacement.ClusterID, err)
		}

		// Also allocate through federation manager for compatibility
		_, err := m.federationMgr.AllocateResources(ctx, &federation.ResourceAllocationRequest{
			ID:          fmt.Sprintf("job-%s-cluster-%s", job.ID, clusterPlacement.ClusterID),
			RequesterID: job.ID,
			CPUCores:    clusterPlacement.Resources.CPUCores,
			MemoryGB:    clusterPlacement.Resources.MemoryGB,
			StorageGB:   clusterPlacement.Resources.StorageGB,
			Priority:    int(job.Priority),
		})
		if err != nil {
			// Rollback transaction on federation allocation failure
			m.transactionalResourceMgr.RollbackTransaction(ctx, transaction.ID)
			return fmt.Errorf("federation resource allocation failed for cluster %s: %w",
				clusterPlacement.ClusterID, err)
		}
	}

	// Commit transaction if all allocations succeeded
	if err := m.transactionalResourceMgr.CommitTransaction(ctx, transaction.ID); err != nil {
		// Attempt rollback on commit failure
		m.transactionalResourceMgr.RollbackTransaction(ctx, transaction.ID)
		return fmt.Errorf("failed to commit resource transaction: %w", err)
	}

	// Record scheduling completion and update job with placement
	schedulingEndTime := time.Now()
	m.mu.Lock()
	job.ClusterPlacements = placement
	job.Status = JobStatusRunning
	job.SchedulingCompletedAt = &schedulingEndTime
	job.StartedAt = &schedulingEndTime
	m.mu.Unlock()

	// Update scheduling metrics
	if job.SchedulingStartedAt != nil {
		schedulingLatency := schedulingEndTime.Sub(*job.SchedulingStartedAt)
		m.metrics.updateSchedulingTimeMetrics(schedulingLatency)
		m.metrics.updateJobTypeMetrics(job.ID, true) // Scheduling succeeded
	}

	// Start execution
	engine := m.executionEngines[job.Type]
	go func() {
		if err := engine.Execute(ctx, job); err != nil {
			m.mu.Lock()
			job.Status = JobStatusFailed
			job.ErrorMessage = err.Error()
			now := time.Now()
			job.CompletedAt = &now
			m.mu.Unlock()

			m.metrics.IncrementFailedJobs()

			// Record execution time even for failed jobs
			if job.StartedAt != nil {
				executionTime := now.Sub(*job.StartedAt)
				m.metrics.updateExecutionTimeMetrics(executionTime)
			}
		} else {
			m.mu.Lock()
			job.Status = JobStatusCompleted
			now := time.Now()
			job.CompletedAt = &now
			m.mu.Unlock()

			m.metrics.IncrementCompletedJobs()

			// Record execution time for successful jobs
			if job.StartedAt != nil {
				executionTime := now.Sub(*job.StartedAt)
				m.metrics.updateExecutionTimeMetrics(executionTime)
			}

			// Record queue time if available
			if job.QueuedAt != nil && job.StartedAt != nil {
				queueTime := job.StartedAt.Sub(*job.QueuedAt)
				m.metrics.updateQueueTimeMetrics(queueTime)
			}
		}
	}()

	return nil
}

// GetPerformanceMetrics returns comprehensive performance metrics
func (m *ComputeJobManager) GetPerformanceMetrics() *PerformanceMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return &PerformanceMetrics{
		AvgQueueTime:      m.metrics.avgQueueTime,
		AvgSchedulingTime: m.metrics.avgSchedulingTime,
		AvgExecutionTime:  m.metrics.avgExecutionTime,
		MinExecutionTime:  m.metrics.minExecutionTime,
		MaxExecutionTime:  m.metrics.maxExecutionTime,
		TotalJobs:         m.metrics.totalJobs,
		CompletedJobs:     m.metrics.completedJobs,
		FailedJobs:        m.metrics.failedJobs,
		QueuedJobs:        m.metrics.queuedJobs,
		RunningJobs:       m.metrics.runningJobs,
		Throughput:        m.metrics.throughputPerMinute,
		SLACompliance:     m.metrics.slaCompliance,
		LastUpdated:       time.Now(),
	}
}

// PerformanceMetrics represents comprehensive job performance metrics
type PerformanceMetrics struct {
	AvgQueueTime      time.Duration `json:"avg_queue_time"`
	AvgSchedulingTime time.Duration `json:"avg_scheduling_time"`
	AvgExecutionTime  time.Duration `json:"avg_execution_time"`
	MinExecutionTime  time.Duration `json:"min_execution_time"`
	MaxExecutionTime  time.Duration `json:"max_execution_time"`
	TotalJobs         int64         `json:"total_jobs"`
	CompletedJobs     int64         `json:"completed_jobs"`
	FailedJobs        int64         `json:"failed_jobs"`
	QueuedJobs        int64         `json:"queued_jobs"`
	RunningJobs       int64         `json:"running_jobs"`
	Throughput        float64       `json:"throughput"`
	SLACompliance     float64       `json:"sla_compliance"`
	LastUpdated       time.Time     `json:"last_updated"`
}

// monitorJob monitors job execution progress
func (m *ComputeJobManager) monitorJob(ctx context.Context, job *ComputeJob) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	engine := m.executionEngines[job.Type]

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.mu.RLock()
			currentJob := m.jobs[job.ID]
			status := currentJob.Status
			m.mu.RUnlock()

			if status != JobStatusRunning {
				return
			}

			// Get progress update
			progress, err := engine.Monitor(ctx, currentJob)
			if err != nil {
				continue
			}

			m.mu.Lock()
			currentJob.Progress = *progress
			m.mu.Unlock()
		}
	}
}

// Enqueue adds a job to the queue
func (q *JobQueue) Enqueue(job *ComputeJob) {
	q.mu.Lock()
	defer q.mu.Unlock()

	job.Status = JobStatusQueued
	q.queues[job.Priority] = append(q.queues[job.Priority], job)
}

// Dequeue removes and returns the next job to process
func (q *JobQueue) Dequeue() *ComputeJob {
	q.mu.Lock()
	defer q.mu.Unlock()

	// Process by priority order
	priorities := []JobPriority{PriorityUrgent, PriorityCritical, PriorityHigh, PriorityNormal, PriorityLow}

	for _, priority := range priorities {
		if len(q.queues[priority]) > 0 {
			job := q.queues[priority][0]
			q.queues[priority] = q.queues[priority][1:]
			return job
		}
	}

	return nil
}

// Remove removes a job from the queue
func (q *JobQueue) Remove(jobID string) {
	q.mu.Lock()
	defer q.mu.Unlock()

	for priority, jobs := range q.queues {
		for i, job := range jobs {
			if job.ID == jobID {
				q.queues[priority] = append(jobs[:i], jobs[i+1:]...)
				return
			}
		}
	}
}

// JobFilters represents job filtering criteria
type JobFilters struct {
	Status   *JobStatus
	Type     *JobType
	Priority *JobPriority
	UserID   string
	Since    *time.Time
	Until    *time.Time
}

// Matches checks if a job matches the filters
func (f *JobFilters) Matches(job *ComputeJob) bool {
	if f.Status != nil && job.Status != *f.Status {
		return false
	}
	if f.Type != nil && job.Type != *f.Type {
		return false
	}
	if f.Priority != nil && job.Priority != *f.Priority {
		return false
	}
	if f.Since != nil && job.SubmittedAt.Before(*f.Since) {
		return false
	}
	if f.Until != nil && job.SubmittedAt.After(*f.Until) {
		return false
	}
	return true
}

// Enhanced Metrics Methods - Comment 17

// Job lifecycle tracking methods
func (m *JobMetrics) IncrementTotalJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.totalJobs++
}

func (m *JobMetrics) IncrementCompletedJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.completedJobs++
}

func (m *JobMetrics) IncrementFailedJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.failedJobs++
}

func (m *JobMetrics) IncrementCancelledJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cancelledJobs++
}

func (m *JobMetrics) IncrementRunningJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.runningJobs++
}

func (m *JobMetrics) DecrementRunningJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.runningJobs > 0 {
		m.runningJobs--
	}
}

func (m *JobMetrics) IncrementQueuedJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queuedJobs++
}

func (m *JobMetrics) DecrementQueuedJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.queuedJobs > 0 {
		m.queuedJobs--
	}
}

func (m *JobMetrics) IncrementSuspendedJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.suspendedJobs++
}

func (m *JobMetrics) DecrementSuspendedJobs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.suspendedJobs > 0 {
		m.suspendedJobs--
	}
}

// Job event tracking
func (m *JobMetrics) RecordJobLifecycleEvent(event *JobLifecycleEvent) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Update timing metrics based on event
	switch event.Event {
	case "job_completed":
		m.updateExecutionTimeMetrics(event.Duration)
		m.updateJobTypeMetrics(event.JobID, true)
	case "job_failed":
		m.errorsByType[event.ErrorMsg]++
		m.updateJobTypeMetrics(event.JobID, false)
	case "job_queued":
		m.updateQueueTimeMetrics(event.Duration)
	case "sla_violation":
		m.slaViolations++
		m.deadlinesMissed++
	case "deadline_met":
		m.deadlinesMet++
	}

	// Update time series metrics
	m.updateTimeSeriesMetrics(event.Timestamp)
}

// Performance metrics updates
func (m *JobMetrics) updateExecutionTimeMetrics(duration time.Duration) {
	if duration < m.minExecutionTime {
		m.minExecutionTime = duration
	}
	if duration > m.maxExecutionTime {
		m.maxExecutionTime = duration
	}
	// Update average execution time (simple moving average)
	if m.completedJobs > 0 {
		m.avgExecutionTime = time.Duration((int64(m.avgExecutionTime)*m.completedJobs + int64(duration)) / (m.completedJobs + 1))
	} else {
		m.avgExecutionTime = duration
	}
}

func (m *JobMetrics) updateQueueTimeMetrics(duration time.Duration) {
	// Update average queue time (simple moving average)
	if m.totalJobs > 0 {
		m.avgQueueTime = time.Duration((int64(m.avgQueueTime)*m.totalJobs + int64(duration)) / (m.totalJobs + 1))
	} else {
		m.avgQueueTime = duration
	}
}

func (m *JobMetrics) updateSchedulingTimeMetrics(duration time.Duration) {
	// Update average scheduling time (simple moving average)
	if m.totalJobs > 0 {
		m.avgSchedulingTime = time.Duration((int64(m.avgSchedulingTime)*m.totalJobs + int64(duration)) / (m.totalJobs + 1))
	} else {
		m.avgSchedulingTime = duration
	}
}

func (m *JobMetrics) updateJobTypeMetrics(jobID string, success bool) {
	// This would need job type lookup - simplified for now
	// In real implementation, would track per job type
}

func (m *JobMetrics) updateTimeSeriesMetrics(timestamp time.Time) {
	// Update hourly metrics
	if len(m.hourlyMetrics.Timestamps) == 0 ||
		timestamp.Sub(m.hourlyMetrics.Timestamps[len(m.hourlyMetrics.Timestamps)-1]) >= time.Hour {
		m.hourlyMetrics.Timestamps = append(m.hourlyMetrics.Timestamps, timestamp)
		m.hourlyMetrics.JobCounts = append(m.hourlyMetrics.JobCounts, m.totalJobs)
		m.hourlyMetrics.Throughputs = append(m.hourlyMetrics.Throughputs, m.throughputPerHour)
	}

	// Update daily metrics
	if len(m.dailyMetrics.Timestamps) == 0 ||
		timestamp.Sub(m.dailyMetrics.Timestamps[len(m.dailyMetrics.Timestamps)-1]) >= 24*time.Hour {
		m.dailyMetrics.Timestamps = append(m.dailyMetrics.Timestamps, timestamp)
		m.dailyMetrics.JobCounts = append(m.dailyMetrics.JobCounts, m.totalJobs)
		m.dailyMetrics.Throughputs = append(m.dailyMetrics.Throughputs, m.dailyThroughput)
	}
}

// Resource metrics
func (m *JobMetrics) UpdateResourceUtilization(resource string, utilization float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.resourceUtilization[resource] = utilization

	// Track peak usage
	if utilization > m.peakResourceUsage[resource] {
		m.peakResourceUsage[resource] = utilization
	}

	// Calculate efficiency (utilization vs allocated)
	if allocated, exists := m.resourceUtilization[resource+"_allocated"]; exists && allocated > 0 {
		m.resourceEfficiency[resource] = utilization / allocated
		m.resourceWaste[resource] = allocated - utilization
	}
}

// Cluster metrics
func (m *JobMetrics) UpdateClusterMetrics(clusterID string, metrics *ClusterJobMetrics) {
	m.mu.Lock()
	defer m.mu.Unlock()

	metrics.LastUpdate = time.Now()
	m.clusterMetrics[clusterID] = metrics
}

// User metrics
func (m *JobMetrics) UpdateUserMetrics(userID string, metrics *UserJobMetrics) {
	m.mu.Lock()
	defer m.mu.Unlock()

	metrics.LastActivity = time.Now()
	m.userMetrics[userID] = metrics
}

// Throughput calculation
func (m *JobMetrics) UpdateThroughputMetrics() {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	elapsed := now.Sub(m.startTime)

	if elapsed > 0 {
		// Jobs per minute
		m.throughputPerMinute = float64(m.completedJobs) / elapsed.Minutes()

		// Jobs per hour
		m.throughputPerHour = float64(m.completedJobs) / elapsed.Hours()

		// Daily throughput (extrapolated)
		if elapsed >= 24*time.Hour {
			m.dailyThroughput = float64(m.completedJobs) / (elapsed.Hours() / 24)
		}

		// Update peak throughput
		if m.throughputPerHour > m.peakThroughput {
			m.peakThroughput = m.throughputPerHour
		}
	}
}

// SLA metrics
func (m *JobMetrics) UpdateSLACompliance() {
	m.mu.Lock()
	defer m.mu.Unlock()

	totalDeadlines := m.deadlinesMet + m.deadlinesMissed
	if totalDeadlines > 0 {
		m.slaCompliance = float64(m.deadlinesMet) / float64(totalDeadlines)
	}
}

// Cost metrics
func (m *JobMetrics) UpdateCostMetrics(jobType JobType, cost float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if existing, exists := m.costPerJob[jobType]; exists {
		// Running average
		if typeMetrics, exists := m.jobTypeMetrics[jobType]; exists && typeMetrics.JobCount > 0 {
			m.costPerJob[jobType] = (existing*float64(typeMetrics.JobCount-1) + cost) / float64(typeMetrics.JobCount)
		}
	} else {
		m.costPerJob[jobType] = cost
	}
}

// Comprehensive metrics retrieval
func (m *JobMetrics) GetMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Calculate derived metrics
	successRate := float64(0)
	if m.totalJobs > 0 {
		successRate = float64(m.completedJobs) / float64(m.totalJobs)
	}

	failureRate := float64(0)
	if m.totalJobs > 0 {
		failureRate = float64(m.failedJobs) / float64(m.totalJobs)
	}

	return map[string]interface{}{
		// Basic counters
		"total_jobs":     m.totalJobs,
		"completed_jobs": m.completedJobs,
		"failed_jobs":    m.failedJobs,
		"cancelled_jobs": m.cancelledJobs,
		"running_jobs":   m.runningJobs,
		"queued_jobs":    m.queuedJobs,
		"suspended_jobs": m.suspendedJobs,

		// Rates and percentages
		"success_rate":   successRate,
		"failure_rate":   failureRate,
		"sla_compliance": m.slaCompliance,

		// Timing metrics
		"avg_execution_time":  m.avgExecutionTime.Seconds(),
		"avg_queue_time":      m.avgQueueTime.Seconds(),
		"avg_scheduling_time": m.avgSchedulingTime.Seconds(),
		"min_execution_time":  m.minExecutionTime.Seconds(),
		"max_execution_time":  m.maxExecutionTime.Seconds(),
		"execution_time_p95":  m.executionTimeP95.Seconds(),
		"execution_time_p99":  m.executionTimeP99.Seconds(),

		// Resource metrics
		"resource_utilization": m.resourceUtilization,
		"resource_efficiency":  m.resourceEfficiency,
		"peak_resource_usage":  m.peakResourceUsage,
		"resource_waste":       m.resourceWaste,

		// Throughput metrics
		"throughput_per_minute": m.throughputPerMinute,
		"throughput_per_hour":   m.throughputPerHour,
		"daily_throughput":      m.dailyThroughput,
		"peak_throughput":       m.peakThroughput,
		"capacity_utilization":  m.capacityUtilization,

		// SLA metrics
		"sla_violations":   m.slaViolations,
		"deadlines_met":    m.deadlinesMet,
		"deadlines_missed": m.deadlinesMissed,

		// Job type metrics
		"job_type_metrics": m.jobTypeMetrics,

		// Cluster metrics
		"cluster_metrics": m.clusterMetrics,

		// Federation metrics
		"federation_metrics": m.federationMetrics,

		// Error metrics
		"errors_by_type": m.errorsByType,
		"total_retries":  m.totalRetries,

		// Cost metrics
		"cost_per_job":          m.costPerJob,
		"cost_efficiency_score": m.costEfficiencyScore,
		"energy_consumption":    m.energyConsumption,

		// User metrics
		"user_metrics": m.userMetrics,

		// Workload patterns
		"workload_patterns": m.workloadPatterns,

		// Time series data
		"hourly_metrics": m.hourlyMetrics,
		"daily_metrics":  m.dailyMetrics,
		"weekly_metrics": m.weeklyMetrics,

		// Metadata
		"metrics_start_time": m.startTime,
		"last_reset_time":    m.lastResetTime,
		"uptime_seconds":     time.Since(m.startTime).Seconds(),
	}
}

// Advanced analytics methods
func (m *JobMetrics) GetJobTypeAnalytics() map[JobType]*JobTypeMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[JobType]*JobTypeMetrics)
	for jobType, metrics := range m.jobTypeMetrics {
		result[jobType] = metrics
	}
	return result
}

func (m *JobMetrics) GetClusterAnalytics() map[string]*ClusterJobMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*ClusterJobMetrics)
	for clusterID, metrics := range m.clusterMetrics {
		result[clusterID] = metrics
	}
	return result
}

func (m *JobMetrics) GetUserAnalytics() map[string]*UserJobMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*UserJobMetrics)
	for userID, metrics := range m.userMetrics {
		result[userID] = metrics
	}
	return result
}

func (m *JobMetrics) GetWorkloadPatterns() *WorkloadPatterns {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.workloadPatterns
}

func (m *JobMetrics) GetPredictiveMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]interface{}{
		"queue_time_predictor":     m.queueTimePredictor,
		"resource_demand_forecast": m.resourceDemandForecast,
	}
}

// Reset and maintenance methods
func (m *JobMetrics) ResetMetrics() {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()

	// Add current state to history before reset
	snapshot := MetricsSnapshot{
		Timestamp:      now,
		TotalJobs:      m.totalJobs,
		Throughput:     m.throughputPerHour,
		ResourceUtil:   make(map[string]float64),
		SLACompliance:  m.slaCompliance,
		CostEfficiency: m.costEfficiencyScore,
	}

	// Copy resource utilization
	for k, v := range m.resourceUtilization {
		snapshot.ResourceUtil[k] = v
	}

	m.metricsHistory = append(m.metricsHistory, snapshot)

	// Reset counters but preserve structure
	m.totalJobs = 0
	m.completedJobs = 0
	m.failedJobs = 0
	m.cancelledJobs = 0
	m.runningJobs = 0
	m.queuedJobs = 0
	m.suspendedJobs = 0
	m.slaViolations = 0
	m.deadlinesMet = 0
	m.deadlinesMissed = 0
	m.totalRetries = 0

	// Reset timing metrics
	m.avgExecutionTime = 0
	m.avgQueueTime = 0
	m.avgSchedulingTime = 0
	m.minExecutionTime = time.Hour * 24
	m.maxExecutionTime = 0

	// Clear maps but keep structure
	for k := range m.resourceUtilization {
		m.resourceUtilization[k] = 0
	}
	for k := range m.resourceEfficiency {
		m.resourceEfficiency[k] = 0
	}
	for k := range m.errorsByType {
		m.errorsByType[k] = 0
	}

	m.lastResetTime = now
}

// Integrated Metrics Tracking in Job Manager - Comment 17

// trackJobLifecycleEvent creates and records a job lifecycle event
func (m *ComputeJobManager) trackJobLifecycleEvent(jobID, event string, prevState, newState JobStatus, duration time.Duration, clusterID, errorMsg string) {
	lifecycleEvent := &JobLifecycleEvent{
		JobID:     jobID,
		Event:     event,
		Timestamp: time.Now(),
		PrevState: prevState,
		NewState:  newState,
		Duration:  duration,
		ClusterID: clusterID,
		ErrorMsg:  errorMsg,
	}

	m.metrics.RecordJobLifecycleEvent(lifecycleEvent)

	// Update relevant counters based on event
	switch event {
	case "job_submitted":
		m.metrics.IncrementTotalJobs()
		m.metrics.IncrementQueuedJobs()
	case "job_started":
		m.metrics.DecrementQueuedJobs()
		m.metrics.IncrementRunningJobs()
	case "job_completed":
		m.metrics.DecrementRunningJobs()
		m.metrics.IncrementCompletedJobs()
	case "job_failed":
		m.metrics.DecrementRunningJobs()
		m.metrics.IncrementFailedJobs()
	case "job_cancelled":
		if prevState == JobStatusQueued {
			m.metrics.DecrementQueuedJobs()
		} else if prevState == JobStatusRunning {
			m.metrics.DecrementRunningJobs()
		}
		m.metrics.IncrementCancelledJobs()
	case "job_suspended":
		m.metrics.DecrementRunningJobs()
		m.metrics.IncrementSuspendedJobs()
	case "job_resumed":
		m.metrics.DecrementSuspendedJobs()
		m.metrics.IncrementRunningJobs()
	}

	// Update throughput and SLA compliance
	m.metrics.UpdateThroughputMetrics()
	m.metrics.UpdateSLACompliance()
}

// updateJobMetricsOnStatusChange updates metrics when job status changes
func (m *ComputeJobManager) updateJobMetricsOnStatusChange(job *ComputeJob, prevStatus JobStatus) {
	now := time.Now()
	var duration time.Duration
	var event string

	switch job.Status {
	case JobStatusQueued:
		event = "job_queued"
		duration = now.Sub(job.SubmittedAt)
	case JobStatusRunning:
		event = "job_started"
		if job.StartedAt != nil {
			duration = job.StartedAt.Sub(job.SubmittedAt) // Queue time
		}
	case JobStatusCompleted:
		event = "job_completed"
		if job.StartedAt != nil && job.CompletedAt != nil {
			duration = job.CompletedAt.Sub(*job.StartedAt) // Execution time
		}
	case JobStatusFailed:
		event = "job_failed"
		if job.StartedAt != nil && job.CompletedAt != nil {
			duration = job.CompletedAt.Sub(*job.StartedAt)
		}
	case JobStatusCancelled:
		event = "job_cancelled"
		if job.StartedAt != nil {
			duration = now.Sub(*job.StartedAt)
		} else {
			duration = now.Sub(job.SubmittedAt)
		}
	case JobStatusSuspended:
		event = "job_suspended"
		if job.StartedAt != nil {
			duration = now.Sub(*job.StartedAt)
		}
	}

	// Extract cluster ID from job placements
	clusterID := ""
	if len(job.ClusterPlacements) > 0 {
		clusterID = job.ClusterPlacements[0].ClusterID
	}

	m.trackJobLifecycleEvent(job.ID, event, prevStatus, job.Status, duration, clusterID, job.ErrorMessage)

	// Update job type metrics
	if typeMetrics, exists := m.metrics.jobTypeMetrics[job.Type]; exists {
		typeMetrics.JobCount++
		if job.Status == JobStatusCompleted {
			// Update success rate
			totalJobs := typeMetrics.JobCount
			successJobs := int64(float64(totalJobs)*typeMetrics.SuccessRate) + 1
			typeMetrics.SuccessRate = float64(successJobs) / float64(totalJobs)
		}
		if job.Status == JobStatusFailed || job.Status == JobStatusCompleted {
			// Update average execution time
			if job.StartedAt != nil && job.CompletedAt != nil {
				execTime := job.CompletedAt.Sub(*job.StartedAt)
				if typeMetrics.AvgExecTime == 0 {
					typeMetrics.AvgExecTime = execTime
				} else {
					typeMetrics.AvgExecTime = time.Duration((int64(typeMetrics.AvgExecTime) + int64(execTime)) / 2)
				}
			}
		}
	} else {
		// Create new job type metrics
		successRate := float64(0)
		if job.Status == JobStatusCompleted {
			successRate = 1.0
		}

		newTypeMetrics := &JobTypeMetrics{
			JobCount:      1,
			SuccessRate:   successRate,
			ResourceUsage: make(map[string]float64),
		}

		if job.StartedAt != nil && job.CompletedAt != nil {
			newTypeMetrics.AvgExecTime = job.CompletedAt.Sub(*job.StartedAt)
		}

		m.metrics.jobTypeMetrics[job.Type] = newTypeMetrics
	}

	// Update cluster metrics if cluster placement is available
	if clusterID != "" {
		if clusterMetrics, exists := m.metrics.clusterMetrics[clusterID]; exists {
			clusterMetrics.JobsScheduled++
			if job.Status == JobStatusCompleted {
				clusterMetrics.JobsCompleted++
			} else if job.Status == JobStatusFailed {
				clusterMetrics.JobsFailed++
			}
		} else {
			newClusterMetrics := &ClusterJobMetrics{
				ClusterID:     clusterID,
				JobsScheduled: 1,
				ResourceUtil:  make(map[string]float64),
			}

			if job.Status == JobStatusCompleted {
				newClusterMetrics.JobsCompleted = 1
			} else if job.Status == JobStatusFailed {
				newClusterMetrics.JobsFailed = 1
			}

			m.metrics.clusterMetrics[clusterID] = newClusterMetrics
		}
	}
}

// updateResourceMetrics updates resource utilization metrics
func (m *ComputeJobManager) updateResourceMetrics(job *ComputeJob, utilization map[string]float64) {
	for resource, util := range utilization {
		m.metrics.UpdateResourceUtilization(resource, util)
	}

	// Update cost metrics based on resource usage
	resourceCost := m.calculateJobResourceCost(job, utilization)
	m.metrics.UpdateCostMetrics(job.Type, resourceCost)
}

// calculateJobResourceCost estimates the cost of running a job
func (m *ComputeJobManager) calculateJobResourceCost(job *ComputeJob, utilization map[string]float64) float64 {
	// Simplified cost calculation - in real implementation would use actual pricing
	baseCost := 0.0

	if cpuUtil, exists := utilization["cpu"]; exists {
		baseCost += cpuUtil * 0.10 // $0.10 per CPU hour
	}

	if memUtil, exists := utilization["memory"]; exists {
		baseCost += memUtil * 0.05 // $0.05 per GB hour
	}

	if gpuUtil, exists := utilization["gpu"]; exists {
		baseCost += gpuUtil * 2.00 // $2.00 per GPU hour
	}

	// Apply duration multiplier
	if job.StartedAt != nil && job.CompletedAt != nil {
		duration := job.CompletedAt.Sub(*job.StartedAt)
		baseCost *= duration.Hours()
	}

	return baseCost
}

// Enhanced integration points for job lifecycle events

// SubmitJob with enhanced metrics tracking
func (m *ComputeJobManager) submitJobWithMetrics(ctx context.Context, job *ComputeJob) error {
	prevStatus := job.Status

	// Submit the job
	if err := m.SubmitJob(ctx, job); err != nil {
		return err
	}

	// Track metrics
	m.updateJobMetricsOnStatusChange(job, prevStatus)

	return nil
}

// scheduleJobWithMetrics wraps job scheduling with metrics
func (m *ComputeJobManager) scheduleJobWithMetrics(ctx context.Context, job *ComputeJob) error {
	startTime := time.Now()
	prevStatus := job.Status

	err := m.scheduleJob(ctx, job)

	schedulingDuration := time.Since(startTime)

	// Update scheduling time metrics
	m.metrics.mu.Lock()
	if m.metrics.avgSchedulingTime == 0 {
		m.metrics.avgSchedulingTime = schedulingDuration
	} else {
		// Simple moving average
		m.metrics.avgSchedulingTime = time.Duration((int64(m.metrics.avgSchedulingTime) + int64(schedulingDuration)) / 2)
	}
	m.metrics.mu.Unlock()

	// Track lifecycle event
	if err != nil {
		m.trackJobLifecycleEvent(job.ID, "scheduling_failed", prevStatus, JobStatusFailed, schedulingDuration, "", err.Error())
	} else {
		m.trackJobLifecycleEvent(job.ID, "scheduling_succeeded", prevStatus, job.Status, schedulingDuration, "", "")
	}

	return err
}

// Periodic metrics maintenance
func (m *ComputeJobManager) startMetricsMaintenanceLoop() {
	go func() {
		ticker := time.NewTicker(5 * time.Minute) // Update metrics every 5 minutes
		defer ticker.Stop()

		for {
			select {
			case <-m.dispatcherCtx.Done():
				return
			case <-ticker.C:
				m.performMetricsMaintenance()
			}
		}
	}()
}

// performMetricsMaintenance runs periodic metrics calculations
func (m *ComputeJobManager) performMetricsMaintenance() {
	// Update throughput metrics
	m.metrics.UpdateThroughputMetrics()

	// Update SLA compliance
	m.metrics.UpdateSLACompliance()

	// Update workload patterns
	m.updateWorkloadPatterns()

	// Clean up old time series data (keep last 30 days)
	m.cleanupOldMetricsData()

	// Update predictive models
	m.updatePredictiveModels()
}

// updateWorkloadPatterns analyzes and updates workload patterns
func (m *ComputeJobManager) updateWorkloadPatterns() {
	m.metrics.mu.Lock()
	defer m.metrics.mu.Unlock()

	now := time.Now()
	hour := now.Hour()

	// Update peak hours (simplified - just count jobs per hour)
	if hour < len(m.metrics.workloadPatterns.PeakHours) {
		m.metrics.workloadPatterns.PeakHours[hour]++
	}

	// Update job type trends
	for jobType, typeMetrics := range m.metrics.jobTypeMetrics {
		if typeMetrics.JobCount > 0 {
			// Calculate trend based on recent activity
			trend := float64(typeMetrics.JobCount) / time.Since(m.metrics.startTime).Hours()
			m.metrics.workloadPatterns.JobTypeTrends[jobType] = trend
		}
	}

	// Calculate overall growth rate
	if len(m.metrics.metricsHistory) > 1 {
		recent := m.metrics.metricsHistory[len(m.metrics.metricsHistory)-1]
		previous := m.metrics.metricsHistory[len(m.metrics.metricsHistory)-2]

		if previous.TotalJobs > 0 {
			growth := float64(recent.TotalJobs-previous.TotalJobs) / float64(previous.TotalJobs)
			m.metrics.workloadPatterns.GrowthRate = growth
		}
	}
}

// cleanupOldMetricsData removes old time series data to prevent memory leaks
func (m *ComputeJobManager) cleanupOldMetricsData() {
	m.metrics.mu.Lock()
	defer m.metrics.mu.Unlock()

	cutoff := time.Now().Add(-30 * 24 * time.Hour) // 30 days

	// Clean hourly metrics
	m.cleanupTimeSeriesData(m.metrics.hourlyMetrics, cutoff)

	// Clean daily metrics
	m.cleanupTimeSeriesData(m.metrics.dailyMetrics, cutoff)

	// Clean metrics history
	m.cleanupMetricsHistory(cutoff)
}

func (m *ComputeJobManager) cleanupTimeSeriesData(ts *TimeSeriesMetrics, cutoff time.Time) {
	var keepIndices []int

	for i, timestamp := range ts.Timestamps {
		if timestamp.After(cutoff) {
			keepIndices = append(keepIndices, i)
		}
	}

	if len(keepIndices) < len(ts.Timestamps) {
		// Compact arrays
		newTimestamps := make([]time.Time, len(keepIndices))
		newJobCounts := make([]int64, len(keepIndices))
		newThroughputs := make([]float64, len(keepIndices))

		for newIdx, oldIdx := range keepIndices {
			newTimestamps[newIdx] = ts.Timestamps[oldIdx]
			newJobCounts[newIdx] = ts.JobCounts[oldIdx]
			newThroughputs[newIdx] = ts.Throughputs[oldIdx]
		}

		ts.Timestamps = newTimestamps
		ts.JobCounts = newJobCounts
		ts.Throughputs = newThroughputs
	}
}

func (m *ComputeJobManager) cleanupMetricsHistory(cutoff time.Time) {
	var keepIndices []int

	for i, snapshot := range m.metrics.metricsHistory {
		if snapshot.Timestamp.After(cutoff) {
			keepIndices = append(keepIndices, i)
		}
	}

	if len(keepIndices) < len(m.metrics.metricsHistory) {
		newHistory := make([]MetricsSnapshot, len(keepIndices))
		for newIdx, oldIdx := range keepIndices {
			newHistory[newIdx] = m.metrics.metricsHistory[oldIdx]
		}
		m.metrics.metricsHistory = newHistory
	}
}

// updatePredictiveModels updates machine learning models for predictions
func (m *ComputeJobManager) updatePredictiveModels() {
	m.metrics.mu.Lock()
	defer m.metrics.mu.Unlock()

	// Update queue time predictor (simplified)
	if len(m.metrics.hourlyMetrics.Timestamps) > 10 {
		// Simple average-based prediction
		totalThroughput := float64(0)
		for _, throughput := range m.metrics.hourlyMetrics.Throughputs {
			totalThroughput += throughput
		}
		avgThroughput := totalThroughput / float64(len(m.metrics.hourlyMetrics.Throughputs))

		// Estimate queue time based on current queue size and throughput
		predictedQueueTime := time.Duration(float64(m.metrics.queuedJobs)/avgThroughput) * time.Hour

		// Cache prediction
		m.metrics.queueTimePredictor.PredictionCache["average"] = predictedQueueTime
		m.metrics.queueTimePredictor.LastTrained = time.Now()
	}

	// Update resource demand forecast (simplified)
	if len(m.metrics.dailyMetrics.Timestamps) > 7 {
		// Simple linear extrapolation for next 24 hours
		recentDays := m.metrics.dailyMetrics.JobCounts[len(m.metrics.dailyMetrics.JobCounts)-7:]

		// Calculate trend
		sum := float64(0)
		for _, jobs := range recentDays {
			sum += float64(jobs)
		}
		avgJobs := sum / 7.0

		// Project resource needs
		m.metrics.resourceDemandForecast.CPUForecast = []float64{avgJobs * 2.0}      // Assume 2 CPU per job
		m.metrics.resourceDemandForecast.MemoryForecast = []float64{avgJobs * 4.0}   // Assume 4GB per job
		m.metrics.resourceDemandForecast.StorageForecast = []float64{avgJobs * 10.0} // Assume 10GB per job
		m.metrics.resourceDemandForecast.Confidence = 0.7                            // Medium confidence
	}
}

// Queue management methods

// ListQueues lists all job queues
func (m *ComputeJobManager) ListQueues(ctx context.Context) ([]string, error) {
	queues := []string{"low", "normal", "high", "critical", "urgent"}
	return queues, nil
}

// GetQueue gets information about a specific queue
func (m *ComputeJobManager) GetQueue(ctx context.Context, queueName string) (map[string]interface{}, error) {
	m.jobQueue.mu.RLock()
	defer m.jobQueue.mu.RUnlock()

	// Map queue names to priorities
	var priority JobPriority
	switch queueName {
	case "low":
		priority = PriorityLow
	case "normal":
		priority = PriorityNormal
	case "high":
		priority = PriorityHigh
	case "critical":
		priority = PriorityCritical
	case "urgent":
		priority = PriorityUrgent
	default:
		return nil, fmt.Errorf("queue %s not found", queueName)
	}

	queueJobs := m.jobQueue.queues[priority]
	queueInfo := map[string]interface{}{
		"name":             queueName,
		"priority":         int(priority),
		"pending_jobs":     len(queueJobs),
		"policy":           string(m.jobQueue.policy),
		"backfill_enabled": m.jobQueue.backfill,
	}

	return queueInfo, nil
}

// GetQueueJobs gets jobs in a specific queue
func (m *ComputeJobManager) GetQueueJobs(ctx context.Context, queueName string) ([]*ComputeJob, error) {
	m.jobQueue.mu.RLock()
	defer m.jobQueue.mu.RUnlock()

	// Map queue names to priorities
	var priority JobPriority
	switch queueName {
	case "low":
		priority = PriorityLow
	case "normal":
		priority = PriorityNormal
	case "high":
		priority = PriorityHigh
	case "critical":
		priority = PriorityCritical
	case "urgent":
		priority = PriorityUrgent
	default:
		return nil, fmt.Errorf("queue %s not found", queueName)
	}

	return m.jobQueue.queues[priority], nil
}

// PauseQueue pauses job processing for a queue
func (m *ComputeJobManager) PauseQueue(ctx context.Context, queueName string) error {
	// For now, this is a no-op - would require queue-specific state management
	return fmt.Errorf("queue pause not implemented yet")
}

// ResumeQueue resumes job processing for a queue
func (m *ComputeJobManager) ResumeQueue(ctx context.Context, queueName string) error {
	// For now, this is a no-op - would require queue-specific state management
	return fmt.Errorf("queue resume not implemented yet")
}

// DrainQueue drains all jobs from a queue
func (m *ComputeJobManager) DrainQueue(ctx context.Context, queueName string) error {
	m.jobQueue.mu.Lock()
	defer m.jobQueue.mu.Unlock()

	// Map queue names to priorities
	var priority JobPriority
	switch queueName {
	case "low":
		priority = PriorityLow
	case "normal":
		priority = PriorityNormal
	case "high":
		priority = PriorityHigh
	case "critical":
		priority = PriorityCritical
	case "urgent":
		priority = PriorityUrgent
	default:
		return fmt.Errorf("queue %s not found", queueName)
	}

	m.jobQueue.queues[priority] = nil
	return nil
}

// Performance and health methods

// GetPerformanceMetricsMap gets overall performance metrics as a map
func (m *ComputeJobManager) GetPerformanceMetricsMap(ctx context.Context) (map[string]interface{}, error) {
	return m.metrics.GetMetrics(), nil
}

// OptimizePerformance runs performance optimization
func (m *ComputeJobManager) OptimizePerformance(ctx context.Context) error {
	// Placeholder - would implement actual optimization logic
	return nil
}

// RunBenchmark runs a performance benchmark
func (m *ComputeJobManager) RunBenchmark(ctx context.Context) (map[string]interface{}, error) {
	// Placeholder - would run actual benchmarks
	benchmarkResults := map[string]interface{}{
		"jobs_per_second":     100.0,
		"avg_response_time":   0.05,
		"resource_efficiency": 0.85,
		"benchmark_time":      time.Now(),
	}
	return benchmarkResults, nil
}

// IsHealthy checks if the job manager is healthy
func (m *ComputeJobManager) IsHealthy(ctx context.Context) bool {
	// Basic health checks
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Check if core components are initialized
	if m.jobQueue == nil || m.federationMgr == nil || m.scheduler == nil {
		return false
	}

	// Check if execution engines are available
	if len(m.executionEngines) == 0 {
		return false
	}

	return true
}

// GetStatistics gets overall job manager statistics
func (m *ComputeJobManager) GetStatistics(ctx context.Context) (map[string]interface{}, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	totalJobsCount := len(m.jobs)
	statsByStatus := make(map[JobStatus]int)
	statsByType := make(map[JobType]int)
	statsByPriority := make(map[JobPriority]int)

	for _, job := range m.jobs {
		statsByStatus[job.Status]++
		statsByType[job.Type]++
		statsByPriority[job.Priority]++
	}

	queueLengths := make(map[JobPriority]int)
	m.jobQueue.mu.RLock()
	for priority, jobs := range m.jobQueue.queues {
		queueLengths[priority] = len(jobs)
	}
	m.jobQueue.mu.RUnlock()

	stats := map[string]interface{}{
		"total_jobs":        totalJobsCount,
		"jobs_by_status":    statsByStatus,
		"jobs_by_type":      statsByType,
		"jobs_by_priority":  statsByPriority,
		"queue_lengths":     queueLengths,
		"execution_engines": len(m.executionEngines),
		"health":            m.IsHealthy(ctx),
		"timestamp":         time.Now(),
	}

	return stats, nil
}

// Placeholder execution engines - to be implemented based on specific requirements
type BatchExecutionEngine struct{}

func NewBatchExecutionEngine() *BatchExecutionEngine                               { return &BatchExecutionEngine{} }
func (e *BatchExecutionEngine) Execute(ctx context.Context, job *ComputeJob) error { return nil }
func (e *BatchExecutionEngine) Monitor(ctx context.Context, job *ComputeJob) (*JobProgress, error) {
	return &JobProgress{}, nil
}
func (e *BatchExecutionEngine) Cancel(ctx context.Context, job *ComputeJob) error  { return nil }
func (e *BatchExecutionEngine) Suspend(ctx context.Context, job *ComputeJob) error { return nil }
func (e *BatchExecutionEngine) Resume(ctx context.Context, job *ComputeJob) error  { return nil }

type ContainerExecutionEngine struct{}

func NewContainerExecutionEngine() *ContainerExecutionEngine                           { return &ContainerExecutionEngine{} }
func (e *ContainerExecutionEngine) Execute(ctx context.Context, job *ComputeJob) error { return nil }
func (e *ContainerExecutionEngine) Monitor(ctx context.Context, job *ComputeJob) (*JobProgress, error) {
	return &JobProgress{}, nil
}
func (e *ContainerExecutionEngine) Cancel(ctx context.Context, job *ComputeJob) error  { return nil }
func (e *ContainerExecutionEngine) Suspend(ctx context.Context, job *ComputeJob) error { return nil }
func (e *ContainerExecutionEngine) Resume(ctx context.Context, job *ComputeJob) error  { return nil }

type MPIExecutionEngine struct{}

func NewMPIExecutionEngine() *MPIExecutionEngine                                 { return &MPIExecutionEngine{} }
func (e *MPIExecutionEngine) Execute(ctx context.Context, job *ComputeJob) error { return nil }
func (e *MPIExecutionEngine) Monitor(ctx context.Context, job *ComputeJob) (*JobProgress, error) {
	return &JobProgress{}, nil
}
func (e *MPIExecutionEngine) Cancel(ctx context.Context, job *ComputeJob) error  { return nil }
func (e *MPIExecutionEngine) Suspend(ctx context.Context, job *ComputeJob) error { return nil }
func (e *MPIExecutionEngine) Resume(ctx context.Context, job *ComputeJob) error  { return nil }

type InteractiveExecutionEngine struct{}

func NewInteractiveExecutionEngine() *InteractiveExecutionEngine {
	return &InteractiveExecutionEngine{}
}
func (e *InteractiveExecutionEngine) Execute(ctx context.Context, job *ComputeJob) error { return nil }
func (e *InteractiveExecutionEngine) Monitor(ctx context.Context, job *ComputeJob) (*JobProgress, error) {
	return &JobProgress{}, nil
}
func (e *InteractiveExecutionEngine) Cancel(ctx context.Context, job *ComputeJob) error  { return nil }
func (e *InteractiveExecutionEngine) Suspend(ctx context.Context, job *ComputeJob) error { return nil }
func (e *InteractiveExecutionEngine) Resume(ctx context.Context, job *ComputeJob) error  { return nil }

type StreamExecutionEngine struct{}

func NewStreamExecutionEngine() *StreamExecutionEngine                              { return &StreamExecutionEngine{} }
func (e *StreamExecutionEngine) Execute(ctx context.Context, job *ComputeJob) error { return nil }
func (e *StreamExecutionEngine) Monitor(ctx context.Context, job *ComputeJob) (*JobProgress, error) {
	return &JobProgress{}, nil
}
func (e *StreamExecutionEngine) Cancel(ctx context.Context, job *ComputeJob) error  { return nil }
func (e *StreamExecutionEngine) Suspend(ctx context.Context, job *ComputeJob) error { return nil }
func (e *StreamExecutionEngine) Resume(ctx context.Context, job *ComputeJob) error  { return nil }

// UpdateJob updates an existing job
func (m *ComputeJobManager) UpdateJob(ctx context.Context, jobID string, updates *ComputeJob) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	job, exists := m.jobs[jobID]
	if !exists {
		return fmt.Errorf("job %s not found", jobID)
	}

	// Only allow updates for pending or queued jobs
	if job.Status != JobStatusPending && job.Status != JobStatusQueued {
		return fmt.Errorf("cannot update job %s in status %s", jobID, job.Status)
	}

	// Update allowed fields
	if updates.Name != "" {
		job.Name = updates.Name
	}
	if updates.Description != "" {
		job.Description = updates.Description
	}
	if updates.Priority != 0 {
		job.Priority = updates.Priority
	}
	if updates.Tags != nil {
		job.Tags = updates.Tags
	}
	if updates.Environment != nil {
		job.Environment = updates.Environment
	}

	return nil
}

// DeleteJob deletes a job
func (m *ComputeJobManager) DeleteJob(ctx context.Context, jobID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	job, exists := m.jobs[jobID]
	if !exists {
		return fmt.Errorf("job %s not found", jobID)
	}

	// Can only delete completed, failed, or cancelled jobs
	if job.Status != JobStatusCompleted && job.Status != JobStatusFailed && job.Status != JobStatusCancelled {
		return fmt.Errorf("cannot delete job %s in status %s", jobID, job.Status)
	}

	delete(m.jobs, jobID)
	return nil
}

// StartJob starts a paused or stopped job
func (m *ComputeJobManager) StartJob(ctx context.Context, jobID string) error {
	m.mu.Lock()
	job, exists := m.jobs[jobID]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("job %s not found", jobID)
	}

	if job.Status != JobStatusSuspended {
		m.mu.Unlock()
		return fmt.Errorf("job %s is not in suspended state", jobID)
	}

	job.Status = JobStatusRunning
	m.mu.Unlock()

	// Resume execution
	if engine, exists := m.executionEngines[job.Type]; exists {
		return engine.Resume(ctx, job)
	}

	return nil
}

// StopJob stops a running job
func (m *ComputeJobManager) StopJob(ctx context.Context, jobID string) error {
	return m.CancelJob(ctx, jobID)
}

// PauseJob pauses a running job
func (m *ComputeJobManager) PauseJob(ctx context.Context, jobID string) error {
	m.mu.Lock()
	job, exists := m.jobs[jobID]
	if !exists {
		m.mu.Unlock()
		return fmt.Errorf("job %s not found", jobID)
	}

	if job.Status != JobStatusRunning {
		m.mu.Unlock()
		return fmt.Errorf("job %s is not running", jobID)
	}

	job.Status = JobStatusSuspended
	m.mu.Unlock()

	// Suspend execution
	if engine, exists := m.executionEngines[job.Type]; exists {
		return engine.Suspend(ctx, job)
	}

	return nil
}

// ResumeJob resumes a paused job
func (m *ComputeJobManager) ResumeJob(ctx context.Context, jobID string) error {
	return m.StartJob(ctx, jobID)
}

// GetJobLogs retrieves logs for a job
func (m *ComputeJobManager) GetJobLogs(ctx context.Context, jobID string, lines int) (string, error) {
	m.mu.RLock()
	job, exists := m.jobs[jobID]
	m.mu.RUnlock()

	if !exists {
		return "", fmt.Errorf("job %s not found", jobID)
	}

	// For now, return a placeholder - actual implementation would depend on logging infrastructure
	return fmt.Sprintf("Logs for job %s (last %d lines):\nJob execution logs would be retrieved from the execution engine", job.Name, lines), nil
}

// GetJobStatus retrieves the status of a job
func (m *ComputeJobManager) GetJobStatus(ctx context.Context, jobID string) (JobStatus, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	job, exists := m.jobs[jobID]
	if !exists {
		return "", fmt.Errorf("job %s not found", jobID)
	}

	return job.Status, nil
}

// GetJobMetrics retrieves metrics for a specific job
func (m *ComputeJobManager) GetJobMetrics(ctx context.Context, jobID string) (map[string]interface{}, error) {
	m.mu.RLock()
	job, exists := m.jobs[jobID]
	m.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("job %s not found", jobID)
	}

	metrics := map[string]interface{}{
		"job_id":         job.ID,
		"status":         job.Status,
		"progress":       job.Progress,
		"submitted_at":   job.SubmittedAt,
		"started_at":     job.StartedAt,
		"completed_at":   job.CompletedAt,
		"resource_usage": job.ClusterPlacements,
	}

	if job.StartedAt != nil {
		if job.CompletedAt != nil {
			metrics["execution_time"] = job.CompletedAt.Sub(*job.StartedAt).Seconds()
		} else {
			metrics["execution_time"] = time.Since(*job.StartedAt).Seconds()
		}
	}

	return metrics, nil
}

// Lifecycle Methods - Comment 14
// Start initializes and starts all job manager components
func (m *ComputeJobManager) Start(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Check if already started
	if m.dispatcherRunning {
		return fmt.Errorf("job manager already started")
	}

	// Initialize federation manager if it supports lifecycle
	if m.federationMgr != nil {
		if fm, ok := m.federationMgr.(interface{ Start(context.Context) error }); ok {
			if err := fm.Start(ctx); err != nil {
				return fmt.Errorf("failed to start federation manager: %w", err)
			}
		}
	}

	// Initialize scheduler
	if startable, ok := m.scheduler.(interface{ Start(context.Context) error }); ok {
		if err := startable.Start(ctx); err != nil {
			return fmt.Errorf("failed to start scheduler: %w", err)
		}
	}

	// Initialize load balancer
	if startable, ok := m.loadBalancer.(interface{ Start(context.Context) error }); ok {
		if err := startable.Start(ctx); err != nil {
			return fmt.Errorf("failed to start load balancer: %w", err)
		}
	}

	// Initialize performance optimizer
	if m.performanceOpt != nil {
		if startable, ok := m.performanceOpt.(interface{ Start(context.Context) error }); ok {
			if err := startable.Start(ctx); err != nil {
				return fmt.Errorf("failed to start performance optimizer: %w", err)
			}
		}
	}

	// Set up dispatcher context
	m.dispatcherCtx, m.dispatcherCancel = context.WithCancel(ctx)

	// Start dispatcher using the existing StartDispatcher method
	if err := m.StartDispatcher(); err != nil {
		return fmt.Errorf("failed to start dispatcher: %w", err)
	}

	return nil
}

// Stop gracefully shuts down all job manager components
func (m *ComputeJobManager) Stop(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var errors []error

	// Stop dispatcher using the existing StopDispatcher method
	if m.dispatcherRunning {
		if err := m.StopDispatcher(); err != nil {
			errors = append(errors, fmt.Errorf("failed to stop dispatcher: %w", err))
		}
	}

	// Stop performance optimizer
	if m.performanceOpt != nil {
		if stoppable, ok := m.performanceOpt.(interface{ Stop(context.Context) error }); ok {
			if err := stoppable.Stop(ctx); err != nil {
				errors = append(errors, fmt.Errorf("failed to stop performance optimizer: %w", err))
			}
		}
	}

	// Stop load balancer
	if stoppable, ok := m.loadBalancer.(interface{ Stop(context.Context) error }); ok {
		if err := stoppable.Stop(ctx); err != nil {
			errors = append(errors, fmt.Errorf("failed to stop load balancer: %w", err))
		}
	}

	// Stop scheduler
	if stoppable, ok := m.scheduler.(interface{ Stop(context.Context) error }); ok {
		if err := stoppable.Stop(ctx); err != nil {
			errors = append(errors, fmt.Errorf("failed to stop scheduler: %w", err))
		}
	}

	// Stop federation manager if it supports lifecycle
	if m.federationMgr != nil {
		if fm, ok := m.federationMgr.(interface{ Stop(context.Context) error }); ok {
			if err := fm.Stop(ctx); err != nil {
				errors = append(errors, fmt.Errorf("failed to stop federation manager: %w", err))
			}
		}
	}

	// Return combined errors if any
	if len(errors) > 0 {
		return fmt.Errorf("shutdown errors: %v", errors)
	}

	return nil
}

// IsRunning returns true if the job manager is currently running
func (m *ComputeJobManager) IsRunning() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.dispatcherRunning
}

// Health returns the health status of the job manager and its components
func (m *ComputeJobManager) Health(ctx context.Context) (*ComponentHealth, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	health := &ComponentHealth{
		ComponentName: "ComputeJobManager",
		Status:        "healthy",
		Timestamp:     time.Now(),
		Subcomponents: make(map[string]*ComponentHealth),
	}

	// Check federation manager health
	if m.federationMgr != nil {
		if healthChecker, ok := m.federationMgr.(interface {
			GetHealth(context.Context) (*federation.HealthCheck, error)
		}); ok {
			if fedHealth, err := healthChecker.GetHealth(ctx); err == nil && fedHealth.Healthy {
				health.Subcomponents["federation"] = &ComponentHealth{
					ComponentName: "FederationManager",
					Status:        "healthy",
					Timestamp:     time.Now(),
				}
			} else {
				health.Status = "degraded"
				health.Subcomponents["federation"] = &ComponentHealth{
					ComponentName: "FederationManager",
					Status:        "unhealthy",
					Timestamp:     time.Now(),
					ErrorMessage:  fmt.Sprintf("federation health check failed: %v", err),
				}
			}
		}
	}

	// Check scheduler health
	if healthChecker, ok := m.scheduler.(interface {
		Health(context.Context) (*ComponentHealth, error)
	}); ok {
		if schedHealth, err := healthChecker.Health(ctx); err == nil {
			health.Subcomponents["scheduler"] = schedHealth
			if schedHealth.Status != "healthy" {
				health.Status = "degraded"
			}
		} else {
			health.Status = "degraded"
			health.Subcomponents["scheduler"] = &ComponentHealth{
				ComponentName: "Scheduler",
				Status:        "unhealthy",
				Timestamp:     time.Now(),
				ErrorMessage:  fmt.Sprintf("scheduler health check failed: %v", err),
			}
		}
	}

	// Check load balancer health
	if healthChecker, ok := m.loadBalancer.(interface {
		Health(context.Context) (*ComponentHealth, error)
	}); ok {
		if lbHealth, err := healthChecker.Health(ctx); err == nil {
			health.Subcomponents["loadbalancer"] = lbHealth
			if lbHealth.Status != "healthy" {
				health.Status = "degraded"
			}
		}
	}

	// Check if dispatcher is running
	if !m.dispatcherRunning {
		health.Status = "unhealthy"
		health.ErrorMessage = "dispatcher not running"
	}

	return health, nil
}

// ComponentHealth represents the health status of a component
type ComponentHealth struct {
	ComponentName string                      `json:"component_name"`
	Status        string                      `json:"status"` // healthy, degraded, unhealthy
	Timestamp     time.Time                   `json:"timestamp"`
	ErrorMessage  string                      `json:"error_message,omitempty"`
	Subcomponents map[string]*ComponentHealth `json:"subcomponents,omitempty"`
}

// Transactional Resource Allocation - Comment 16
// Provides ACID-compliant resource allocation with commit/rollback capabilities

// ResourceTransaction represents a transactional resource allocation
type ResourceTransaction struct {
	ID                  string                        `json:"id"`
	JobID               string                        `json:"job_id"`
	Status              TransactionStatus             `json:"status"`
	ResourceAllocations []*TransactionalAllocation    `json:"resource_allocations"`
	ClusterAllocations  map[string]*ClusterAllocation `json:"cluster_allocations"`
	CreatedAt           time.Time                     `json:"created_at"`
	CommittedAt         *time.Time                    `json:"committed_at,omitempty"`
	RolledBackAt        *time.Time                    `json:"rolled_back_at,omitempty"`
	Timeout             time.Duration                 `json:"timeout"`
	ErrorMessage        string                        `json:"error_message,omitempty"`
	mu                  sync.RWMutex                  `json:"-"`
}

type TransactionStatus string

const (
	TransactionStatusPending     TransactionStatus = "pending"
	TransactionStatusPreparing   TransactionStatus = "preparing"
	TransactionStatusPrepared    TransactionStatus = "prepared"
	TransactionStatusCommitting  TransactionStatus = "committing"
	TransactionStatusCommitted   TransactionStatus = "committed"
	TransactionStatusRollingBack TransactionStatus = "rolling_back"
	TransactionStatusRolledBack  TransactionStatus = "rolled_back"
	TransactionStatusFailed      TransactionStatus = "failed"
	TransactionStatusExpired     TransactionStatus = "expired"
)

// TransactionalAllocation represents a resource allocation within a transaction
type TransactionalAllocation struct {
	ID             string                 `json:"id"`
	ClusterID      string                 `json:"cluster_id"`
	ResourceType   string                 `json:"resource_type"`
	Amount         float64                `json:"amount"`
	Unit           string                 `json:"unit"`
	Reserved       bool                   `json:"reserved"`
	Constraints    map[string]interface{} `json:"constraints"`
	ReservationID  string                 `json:"reservation_id,omitempty"`
	PreCommitState *ResourceState         `json:"pre_commit_state,omitempty"`
}

// ClusterAllocation represents resource allocation at cluster level
type ClusterAllocation struct {
	ClusterID       string        `json:"cluster_id"`
	ReservedCPU     float64       `json:"reserved_cpu"`
	ReservedMemory  float64       `json:"reserved_memory"`
	ReservedStorage float64       `json:"reserved_storage"`
	ReservationIDs  []string      `json:"reservation_ids"`
	PreCommitState  *ClusterState `json:"pre_commit_state"`
}

// ResourceState captures the state of resources before allocation
type ResourceState struct {
	AvailableCPU     float64   `json:"available_cpu"`
	AvailableMemory  float64   `json:"available_memory"`
	AvailableStorage float64   `json:"available_storage"`
	Timestamp        time.Time `json:"timestamp"`
}

// ClusterState captures cluster state before allocation
type ClusterState struct {
	TotalCPU         float64   `json:"total_cpu"`
	AvailableCPU     float64   `json:"available_cpu"`
	TotalMemory      float64   `json:"total_memory"`
	AvailableMemory  float64   `json:"available_memory"`
	TotalStorage     float64   `json:"total_storage"`
	AvailableStorage float64   `json:"available_storage"`
	ActiveJobs       int       `json:"active_jobs"`
	Timestamp        time.Time `json:"timestamp"`
}

// TransactionalResourceManager handles transactional resource allocation
type TransactionalResourceManager struct {
	mu                 sync.RWMutex
	transactions       map[string]*ResourceTransaction
	jobManager         *ComputeJobManager
	federationMgr      *federation.FederationManager
	reservationTimeout time.Duration
	maxConcurrentTxns  int
	cleanupInterval    time.Duration
	cleanupTicker      *time.Ticker
	stopCleanup        chan struct{}
}

// NewTransactionalResourceManager creates a new transactional resource manager
func NewTransactionalResourceManager(jobManager *ComputeJobManager, federationMgr *federation.FederationManager) *TransactionalResourceManager {
	trm := &TransactionalResourceManager{
		transactions:       make(map[string]*ResourceTransaction),
		jobManager:         jobManager,
		federationMgr:      federationMgr,
		reservationTimeout: 5 * time.Minute,
		maxConcurrentTxns:  100,
		cleanupInterval:    1 * time.Minute,
		stopCleanup:        make(chan struct{}),
	}

	// Start cleanup goroutine
	trm.cleanupTicker = time.NewTicker(trm.cleanupInterval)
	go trm.runCleanup()

	return trm
}

// BeginTransaction starts a new resource allocation transaction
func (trm *TransactionalResourceManager) BeginTransaction(ctx context.Context, jobID string, timeout time.Duration) (*ResourceTransaction, error) {
	trm.mu.Lock()
	defer trm.mu.Unlock()

	// Check transaction limits
	if len(trm.transactions) >= trm.maxConcurrentTxns {
		return nil, fmt.Errorf("maximum concurrent transactions exceeded (%d)", trm.maxConcurrentTxns)
	}

	if timeout <= 0 {
		timeout = trm.reservationTimeout
	}

	txn := &ResourceTransaction{
		ID:                  generateTransactionID(),
		JobID:               jobID,
		Status:              TransactionStatusPending,
		ResourceAllocations: []*TransactionalAllocation{},
		ClusterAllocations:  make(map[string]*ClusterAllocation),
		CreatedAt:           time.Now(),
		Timeout:             timeout,
	}

	trm.transactions[txn.ID] = txn
	return txn, nil
}

// AllocateInTransaction reserves resources within a transaction
func (trm *TransactionalResourceManager) AllocateInTransaction(ctx context.Context, txnID string,
	clusterID string, resourceType string, amount float64, unit string, constraints map[string]interface{}) error {

	trm.mu.Lock()
	defer trm.mu.Unlock()

	txn, exists := trm.transactions[txnID]
	if !exists {
		return fmt.Errorf("transaction %s not found", txnID)
	}

	txn.mu.Lock()
	defer txn.mu.Unlock()

	if txn.Status != TransactionStatusPending && txn.Status != TransactionStatusPreparing {
		return fmt.Errorf("transaction %s is not in pending/preparing state: %s", txnID, txn.Status)
	}

	// Check if transaction has expired
	if time.Since(txn.CreatedAt) > txn.Timeout {
		txn.Status = TransactionStatusExpired
		return fmt.Errorf("transaction %s has expired", txnID)
	}

	// Capture current cluster state before allocation
	clusterState, err := trm.captureClusterState(ctx, clusterID)
	if err != nil {
		return fmt.Errorf("failed to capture cluster state: %w", err)
	}

	// Check if resources are available
	if !trm.checkResourceAvailability(clusterState, resourceType, amount) {
		return fmt.Errorf("insufficient %s resources in cluster %s: requested %.2f %s",
			resourceType, clusterID, amount, unit)
	}

	// Create reservation
	reservationID, err := trm.createReservation(ctx, clusterID, resourceType, amount)
	if err != nil {
		return fmt.Errorf("failed to create resource reservation: %w", err)
	}

	// Create transactional allocation
	allocation := &TransactionalAllocation{
		ID:            generateAllocationID(),
		ClusterID:     clusterID,
		ResourceType:  resourceType,
		Amount:        amount,
		Unit:          unit,
		Reserved:      true,
		Constraints:   constraints,
		ReservationID: reservationID,
		PreCommitState: &ResourceState{
			AvailableCPU:     clusterState.AvailableCPU,
			AvailableMemory:  clusterState.AvailableMemory,
			AvailableStorage: clusterState.AvailableStorage,
			Timestamp:        time.Now(),
		},
	}

	txn.ResourceAllocations = append(txn.ResourceAllocations, allocation)

	// Update cluster allocation tracking
	if clusterAlloc, exists := txn.ClusterAllocations[clusterID]; exists {
		trm.updateClusterAllocation(clusterAlloc, resourceType, amount, reservationID)
	} else {
		txn.ClusterAllocations[clusterID] = &ClusterAllocation{
			ClusterID:      clusterID,
			ReservationIDs: []string{reservationID},
			PreCommitState: clusterState,
		}
		trm.updateClusterAllocation(txn.ClusterAllocations[clusterID], resourceType, amount, reservationID)
	}

	txn.Status = TransactionStatusPreparing
	return nil
}

// PrepareTransaction validates all allocations are ready for commit
func (trm *TransactionalResourceManager) PrepareTransaction(ctx context.Context, txnID string) error {
	trm.mu.Lock()
	defer trm.mu.Unlock()

	txn, exists := trm.transactions[txnID]
	if !exists {
		return fmt.Errorf("transaction %s not found", txnID)
	}

	txn.mu.Lock()
	defer txn.mu.Unlock()

	if txn.Status != TransactionStatusPreparing {
		return fmt.Errorf("transaction %s is not in preparing state: %s", txnID, txn.Status)
	}

	// Check if transaction has expired
	if time.Since(txn.CreatedAt) > txn.Timeout {
		txn.Status = TransactionStatusExpired
		return fmt.Errorf("transaction %s has expired", txnID)
	}

	// Validate all reservations are still valid
	for _, allocation := range txn.ResourceAllocations {
		if !trm.validateReservation(ctx, allocation.ReservationID) {
			txn.Status = TransactionStatusFailed
			txn.ErrorMessage = fmt.Sprintf("reservation %s is no longer valid", allocation.ReservationID)
			return fmt.Errorf("reservation validation failed: %s", txn.ErrorMessage)
		}
	}

	// All validations passed
	txn.Status = TransactionStatusPrepared
	return nil
}

// CommitTransaction commits all resource allocations
func (trm *TransactionalResourceManager) CommitTransaction(ctx context.Context, txnID string) error {
	trm.mu.Lock()
	defer trm.mu.Unlock()

	txn, exists := trm.transactions[txnID]
	if !exists {
		return fmt.Errorf("transaction %s not found", txnID)
	}

	txn.mu.Lock()
	defer txn.mu.Unlock()

	if txn.Status != TransactionStatusPrepared {
		return fmt.Errorf("transaction %s is not prepared for commit: %s", txnID, txn.Status)
	}

	txn.Status = TransactionStatusCommitting

	// Commit all allocations
	var commitErrors []error
	for _, allocation := range txn.ResourceAllocations {
		if err := trm.commitAllocation(ctx, allocation); err != nil {
			commitErrors = append(commitErrors, err)
		}
	}

	if len(commitErrors) > 0 {
		// Attempt rollback on commit failure
		txn.Status = TransactionStatusFailed
		txn.ErrorMessage = fmt.Sprintf("commit failed: %v", commitErrors)

		// Try to rollback already committed allocations
		trm.rollbackTransactionInternal(ctx, txn)

		return fmt.Errorf("transaction commit failed: %v", commitErrors)
	}

	// Success
	now := time.Now()
	txn.Status = TransactionStatusCommitted
	txn.CommittedAt = &now

	return nil
}

// RollbackTransaction rolls back all resource allocations
func (trm *TransactionalResourceManager) RollbackTransaction(ctx context.Context, txnID string) error {
	trm.mu.Lock()
	defer trm.mu.Unlock()

	txn, exists := trm.transactions[txnID]
	if !exists {
		return fmt.Errorf("transaction %s not found", txnID)
	}

	txn.mu.Lock()
	defer txn.mu.Unlock()

	if txn.Status == TransactionStatusCommitted {
		return fmt.Errorf("cannot rollback committed transaction %s", txnID)
	}

	txn.Status = TransactionStatusRollingBack

	err := trm.rollbackTransactionInternal(ctx, txn)

	now := time.Now()
	txn.Status = TransactionStatusRolledBack
	txn.RolledBackAt = &now

	return err
}

// GetTransaction returns transaction details
func (trm *TransactionalResourceManager) GetTransaction(txnID string) (*ResourceTransaction, error) {
	trm.mu.RLock()
	defer trm.mu.RUnlock()

	txn, exists := trm.transactions[txnID]
	if !exists {
		return nil, fmt.Errorf("transaction %s not found", txnID)
	}

	// Return a copy to prevent external modification
	txn.mu.RLock()
	defer txn.mu.RUnlock()

	return &ResourceTransaction{
		ID:                  txn.ID,
		JobID:               txn.JobID,
		Status:              txn.Status,
		ResourceAllocations: txn.ResourceAllocations, // TODO: Deep copy
		ClusterAllocations:  txn.ClusterAllocations,  // TODO: Deep copy
		CreatedAt:           txn.CreatedAt,
		CommittedAt:         txn.CommittedAt,
		RolledBackAt:        txn.RolledBackAt,
		Timeout:             txn.Timeout,
		ErrorMessage:        txn.ErrorMessage,
	}, nil
}

// Helper methods for transactional resource management

func (trm *TransactionalResourceManager) captureClusterState(ctx context.Context, clusterID string) (*ClusterState, error) {
	// This would typically call federation manager to get actual cluster state
	// For now, return a placeholder state
	return &ClusterState{
		TotalCPU:         1000.0,
		AvailableCPU:     800.0,
		TotalMemory:      2000.0,
		AvailableMemory:  1600.0,
		TotalStorage:     5000.0,
		AvailableStorage: 4000.0,
		ActiveJobs:       10,
		Timestamp:        time.Now(),
	}, nil
}

func (trm *TransactionalResourceManager) checkResourceAvailability(state *ClusterState, resourceType string, amount float64) bool {
	switch resourceType {
	case "cpu":
		return state.AvailableCPU >= amount
	case "memory":
		return state.AvailableMemory >= amount
	case "storage":
		return state.AvailableStorage >= amount
	default:
		return false
	}
}

func (trm *TransactionalResourceManager) createReservation(ctx context.Context, clusterID, resourceType string, amount float64) (string, error) {
	// Create a reservation ID and mark resources as reserved
	reservationID := generateReservationID()
	// In a real implementation, this would call federation manager to create actual reservation
	return reservationID, nil
}

func (trm *TransactionalResourceManager) validateReservation(ctx context.Context, reservationID string) bool {
	// In a real implementation, this would verify the reservation is still valid
	return true
}

func (trm *TransactionalResourceManager) commitAllocation(ctx context.Context, allocation *TransactionalAllocation) error {
	// In a real implementation, this would convert reservation to actual allocation
	// via federation manager
	return nil
}

func (trm *TransactionalResourceManager) rollbackTransactionInternal(ctx context.Context, txn *ResourceTransaction) error {
	var rollbackErrors []error

	// Release all reservations
	for _, allocation := range txn.ResourceAllocations {
		if allocation.Reserved && allocation.ReservationID != "" {
			if err := trm.releaseReservation(ctx, allocation.ReservationID); err != nil {
				rollbackErrors = append(rollbackErrors, err)
			}
		}
	}

	if len(rollbackErrors) > 0 {
		return fmt.Errorf("rollback errors: %v", rollbackErrors)
	}

	return nil
}

func (trm *TransactionalResourceManager) releaseReservation(ctx context.Context, reservationID string) error {
	// In a real implementation, this would release the reservation via federation manager
	return nil
}

func (trm *TransactionalResourceManager) updateClusterAllocation(clusterAlloc *ClusterAllocation, resourceType string, amount float64, reservationID string) {
	clusterAlloc.ReservationIDs = append(clusterAlloc.ReservationIDs, reservationID)

	switch resourceType {
	case "cpu":
		clusterAlloc.ReservedCPU += amount
	case "memory":
		clusterAlloc.ReservedMemory += amount
	case "storage":
		clusterAlloc.ReservedStorage += amount
	}
}

func (trm *TransactionalResourceManager) runCleanup() {
	for {
		select {
		case <-trm.cleanupTicker.C:
			trm.cleanupExpiredTransactions()
		case <-trm.stopCleanup:
			trm.cleanupTicker.Stop()
			return
		}
	}
}

func (trm *TransactionalResourceManager) cleanupExpiredTransactions() {
	trm.mu.Lock()
	defer trm.mu.Unlock()

	now := time.Now()
	var toDelete []string

	for txnID, txn := range trm.transactions {
		txn.mu.RLock()
		expired := now.Sub(txn.CreatedAt) > txn.Timeout
		status := txn.Status
		txn.mu.RUnlock()

		if expired && (status == TransactionStatusPending || status == TransactionStatusPreparing) {
			// Mark as expired and rollback
			txn.mu.Lock()
			txn.Status = TransactionStatusExpired
			trm.rollbackTransactionInternal(context.Background(), txn)
			txn.mu.Unlock()
		}

		// Clean up completed transactions after 1 hour
		if (status == TransactionStatusCommitted || status == TransactionStatusRolledBack ||
			status == TransactionStatusFailed || status == TransactionStatusExpired) &&
			now.Sub(txn.CreatedAt) > time.Hour {
			toDelete = append(toDelete, txnID)
		}
	}

	for _, txnID := range toDelete {
		delete(trm.transactions, txnID)
	}
}

// Stop cleanup goroutine
func (trm *TransactionalResourceManager) Stop() {
	close(trm.stopCleanup)
}

// Utility functions
func generateTransactionID() string {
	return fmt.Sprintf("txn-%d", time.Now().UnixNano())
}

func generateAllocationID() string {
	return fmt.Sprintf("alloc-%d", time.Now().UnixNano())
}

func generateReservationID() string {
	return fmt.Sprintf("rsv-%d", time.Now().UnixNano())
}

// Wrapper methods to handle API compatibility for handlers

// UpdateJob with map[string]interface{} for API compatibility
func (m *ComputeJobManager) UpdateJob(ctx context.Context, jobID string, updates map[string]interface{}) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	job, exists := m.jobs[jobID]
	if !exists {
		return fmt.Errorf("job %s not found", jobID)
	}

	// Only allow updates for pending or queued jobs
	if job.Status != JobStatusPending && job.Status != JobStatusQueued {
		return fmt.Errorf("cannot update job %s in status %s", jobID, job.Status)
	}

	// Apply partial updates from map
	if name, ok := updates["name"].(string); ok && name != "" {
		job.Name = name
	}
	if description, ok := updates["description"].(string); ok && description != "" {
		job.Description = description
	}
	if priority, ok := updates["priority"].(int); ok && priority != 0 {
		job.Priority = JobPriority(priority)
	}
	if tags, ok := updates["tags"].(map[string]string); ok && tags != nil {
		job.Tags = tags
	}
	if env, ok := updates["environment"].(map[string]string); ok && env != nil {
		job.Environment = env
	}
	if constraints, ok := updates["constraints"].([]interface{}); ok && constraints != nil {
		job.Constraints = convertConstraints(constraints)
	}
	if timeout, ok := updates["timeout"].(int); ok && timeout > 0 {
		job.Timeout = time.Duration(timeout) * time.Second
	}

	return nil
}

// ListJobsWithParams wrapper for API compatibility with individual parameters
func (m *ComputeJobManager) ListJobsWithParams(ctx context.Context, queueName, status string, limit, offset int) ([]*ComputeJob, error) {
	filters := JobFilters{}

	if status != "" {
		jobStatus := JobStatus(status)
		filters.Status = &jobStatus
	}

	jobs, err := m.ListJobs(ctx, filters)
	if err != nil {
		return nil, err
	}

	// Apply queue name filtering
	if queueName != "" {
		filtered := make([]*ComputeJob, 0)
		for _, job := range jobs {
			if job.QueueName == queueName {
				filtered = append(filtered, job)
			}
		}
		jobs = filtered
	}

	// Apply pagination
	if offset > 0 && offset < len(jobs) {
		jobs = jobs[offset:]
	}
	if limit > 0 && limit < len(jobs) {
		jobs = jobs[:limit]
	}

	return jobs, nil
}

// Helper function to convert constraint interfaces to JobConstraints
func convertConstraints(constraints []interface{}) []JobConstraint {
	result := make([]JobConstraint, 0, len(constraints))
	for _, c := range constraints {
		if constraint, ok := c.(map[string]interface{}); ok {
			jc := JobConstraint{}
			if t, ok := constraint["type"].(string); ok {
				jc.Type = t
			}
			if op, ok := constraint["operator"].(string); ok {
				jc.Operator = op
			}
			if val, ok := constraint["value"]; ok {
				jc.Value = val
			}
			result = append(result, jc)
		}
	}
	return result
}
