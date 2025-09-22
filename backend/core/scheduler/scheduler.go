package scheduler

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"
)

// generateID creates a unique ID by combining the current timestamp with random bytes
func generateID() string {
	timestamp := fmt.Sprintf("%d", time.Now().UnixNano())

	// Add some randomness
	randomBytes := make([]byte, 8)
	rand.Read(randomBytes)
	randomHex := hex.EncodeToString(randomBytes)

	return fmt.Sprintf("%s-%s", timestamp, randomHex)
}

// ResourceType represents a type of resource
type ResourceType string

// ResourceTypes
const (
	ResourceCPU     ResourceType = "cpu"
	ResourceMemory  ResourceType = "memory"
	ResourceDisk    ResourceType = "disk"
	ResourceNetwork ResourceType = "network"
)

// Resource represents an available resource
type Resource struct {
	Type     ResourceType
	Capacity float64
	Used     float64
}

// AvailablePercentage returns the percentage of available resource
func (r *Resource) AvailablePercentage() float64 {
	if r.Capacity <= 0 {
		return 0
	}
	return 100 * (1 - r.Used/r.Capacity)
}

// Available returns the amount of available resource
func (r *Resource) Available() float64 {
	return r.Capacity - r.Used
}

// NodeResources represents resources available on a node
type NodeResources struct {
	NodeID     string
	Resources  map[ResourceType]*Resource
	LastUpdate time.Time
	Metrics    map[string]float64
	Available  bool
}

// ResourceConstraint represents a constraint on resources
type ResourceConstraint struct {
	Type      ResourceType
	MinAmount float64
	MaxAmount float64
}

// NetworkConstraint represents network-specific constraints
type NetworkConstraint struct {
	MinBandwidthBps   uint64        `json:"min_bandwidth_bps"`   // Minimum bandwidth required
	MaxLatencyMs      float64       `json:"max_latency_ms"`      // Maximum acceptable latency
	RequiredTopology  string        `json:"required_topology"`   // Required network topology (e.g., "low-latency")
	MinConnections    int           `json:"min_connections"`     // Minimum number of network connections
	BandwidthGuarantee bool         `json:"bandwidth_guarantee"` // Whether bandwidth must be guaranteed
}

// ResourceRequest represents a request for resources
type ResourceRequest struct {
	ID                 string
	Constraints        []ResourceConstraint
	NetworkConstraints *NetworkConstraint // Optional network constraints
	Priority           int
	Timeout            time.Duration
	CreatedAt          time.Time
	ExpiresAt          time.Time
}

// ResourceAllocation represents allocated resources
type ResourceAllocation struct {
	RequestID   string
	NodeID      string
	Resources   map[ResourceType]float64
	AllocatedAt time.Time
	ExpiresAt   time.Time
	Released    bool
}

// TaskDistribution represents a task to be distributed across nodes
type TaskDistribution struct {
	TaskID          string
	ResourceRequest ResourceRequest
	TargetNodeCount int
	Allocations     []ResourceAllocation
	DistributedAt   time.Time
	CompletedAt     time.Time
	Status          TaskStatus
}

// TaskStatus represents the status of a task
type TaskStatus string

// TaskStatuses
const (
	TaskPending   TaskStatus = "pending"
	TaskAllocated TaskStatus = "allocated"
	TaskRunning   TaskStatus = "running"
	TaskCompleted TaskStatus = "completed"
	TaskFailed    TaskStatus = "failed"
)

// SchedulerConfig contains configuration for the scheduler
type SchedulerConfig struct {
	// AllocationInterval is the interval between allocation runs
	AllocationInterval time.Duration

	// NodeTimeout is the timeout for nodes considered offline
	NodeTimeout time.Duration

	// EnablePreemption enables preemption of lower priority tasks
	EnablePreemption bool

	// MaxRequestTimeout is the maximum timeout for resource requests
	MaxRequestTimeout time.Duration

	// MinimumNodeCount is the minimum number of nodes required for scheduling
	MinimumNodeCount int

	// OvercommitRatio allows for resource overcommitment
	OvercommitRatio map[ResourceType]float64

	// BalancingWeight determines weight given to load balancing vs. resource efficiency
	BalancingWeight float64
	
	// NetworkAwarenessEnabled enables network-aware scheduling
	NetworkAwarenessEnabled bool
	
	// NetworkScoreWeight weight of network score in overall scoring
	NetworkScoreWeight float64
	
	// MaxNetworkUtilization is the maximum allowed network utilization percentage
	MaxNetworkUtilization float64
	
	// BandwidthPredictionEnabled enables bandwidth prediction for scheduling
	BandwidthPredictionEnabled bool
}

// ValidateAndNormalizeWeights validates and normalizes scheduler weights
func ValidateAndNormalizeWeights(config *SchedulerConfig) error {
	// Validate individual weight ranges
	if config.BalancingWeight < 0.0 || config.BalancingWeight > 1.0 {
		return fmt.Errorf("BalancingWeight must be between 0.0 and 1.0, got %f", config.BalancingWeight)
	}
	
	if config.NetworkScoreWeight < 0.0 || config.NetworkScoreWeight > 1.0 {
		return fmt.Errorf("NetworkScoreWeight must be between 0.0 and 1.0, got %f", config.NetworkScoreWeight)
	}
	
	// Ensure weights don't exceed 1.0 when combined
	if config.NetworkAwarenessEnabled {
		totalWeight := config.BalancingWeight + config.NetworkScoreWeight
		if totalWeight > 1.0 {
			// Normalize weights to sum to 1.0 while preserving their relative proportions
			resourceWeight := 1.0 - totalWeight
			if resourceWeight < 0.0 {
				// Proportionally scale down weights to fit
				normalizationFactor := 1.0 / totalWeight
				config.BalancingWeight *= normalizationFactor * 0.8 // Leave 20% for resource weight
				config.NetworkScoreWeight *= normalizationFactor * 0.8
				log.Printf("Normalized weights: BalancingWeight=%.3f, NetworkScoreWeight=%.3f", 
					config.BalancingWeight, config.NetworkScoreWeight)
			}
		}
	}
	
	return nil
}

// DefaultSchedulerConfig returns a default scheduler configuration
func DefaultSchedulerConfig() SchedulerConfig {
	config := SchedulerConfig{
		AllocationInterval: 5 * time.Second,
		NodeTimeout:        2 * time.Minute,
		EnablePreemption:   true,
		MaxRequestTimeout:  1 * time.Hour,
		MinimumNodeCount:   1,
		OvercommitRatio: map[ResourceType]float64{
			ResourceCPU:     1.5, // CPU can be overcommitted by 50%
			ResourceMemory:  1.0, // Memory cannot be overcommitted
			ResourceDisk:    1.2, // Disk can be overcommitted by 20%
			ResourceNetwork: 2.0, // Network can be overcommitted by 100%
		},
		BalancingWeight: 0.5, // Equal weight to load balancing and resource efficiency
		NetworkAwarenessEnabled: false,
		NetworkScoreWeight: 0.3,
		MaxNetworkUtilization: 90.0,
		BandwidthPredictionEnabled: false,
	}
	
	// Validate and normalize weights
	if err := ValidateAndNormalizeWeights(&config); err != nil {
		log.Printf("Warning: Weight validation failed for default config: %v", err)
	}
	
	return config
}

// NetworkTopologyProvider interface for network-aware scheduling
type NetworkTopologyProvider interface {
	GetNetworkCost(nodeA, nodeB string) (float64, error)
	GetBandwidthAvailability(nodeID string) (float64, error)
	GetNetworkUtilization() map[string]float64
	GetBandwidthCapability(nodeID string) (uint64, error)        // Get maximum bandwidth capability
	GetLatencyBetweenNodes(nodeA, nodeB string) (float64, error) // Get latency in milliseconds
	ValidateNetworkConstraints(nodeID string, constraints *NetworkConstraint) error // Validate network constraints
}

// AIProvider interface for AI-powered scheduling optimization
type AIProvider interface {
	// Resource prediction
	PredictResourceDemand(nodeID string, resourceType ResourceType, horizonMinutes int) ([]float64, float64, error)

	// Performance optimization
	OptimizePerformance(clusterData map[string]interface{}, goals []string) (map[string]interface{}, error)

	// Workload analysis
	AnalyzeWorkload(vmID string, workloadData []map[string]interface{}) (map[string]interface{}, error)

	// Anomaly detection
	DetectAnomalies(metrics map[string]float64) (bool, float64, []string, error)

	// Scaling recommendations
	GetScalingRecommendations(vmID string, currentResources map[string]float64, historicalData []map[string]interface{}) ([]map[string]interface{}, error)

	// Migration optimization
	OptimizeMigration(vmID string, sourceHost string, targetHosts []string, vmMetrics map[string]float64) (map[string]interface{}, error)

	// Bandwidth optimization
	OptimizeBandwidth(networkID string, trafficData []map[string]interface{}, qosRequirements map[string]float64) (map[string]interface{}, error)
}

// AISchedulingRecommendation represents AI-based scheduling recommendations
type AISchedulingRecommendation struct {
	NodeID          string                 `json:"node_id"`
	Score           float64                `json:"score"`
	Confidence      float64                `json:"confidence"`
	Reasoning       []string               `json:"reasoning"`
	PredictedLoad   map[ResourceType]float64 `json:"predicted_load"`
	RiskAssessment  map[string]float64     `json:"risk_assessment"`
	OptimizationTips []string              `json:"optimization_tips"`
}

// AIMetricsData represents data sent to AI engine for analysis
type AIMetricsData struct {
	Timestamp    time.Time              `json:"timestamp"`
	NodeID       string                 `json:"node_id"`
	ResourceType ResourceType           `json:"resource_type"`
	CurrentUsage float64                `json:"current_usage"`
	Capacity     float64                `json:"capacity"`
	Metrics      map[string]float64     `json:"metrics"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// AISchedulerConfig contains AI-specific configuration
type AISchedulerConfig struct {
	Enabled                 bool          `json:"enabled"`
	AIEngineURL            string        `json:"ai_engine_url"`
	PredictionHorizon      int           `json:"prediction_horizon_minutes"`
	ConfidenceThreshold    float64       `json:"confidence_threshold"`
	AnomalySensitivity     float64       `json:"anomaly_sensitivity"`
	MaxRetries             int           `json:"max_retries"`
	RequestTimeout         time.Duration `json:"request_timeout"`
	MetricsCollectionInterval time.Duration `json:"metrics_collection_interval"`
	EnableProactiveScaling bool          `json:"enable_proactive_scaling"`
	EnableAnomalyDetection bool          `json:"enable_anomaly_detection"`
	AIWeightInScheduling   float64       `json:"ai_weight_in_scheduling"`
}

// Scheduler handles resource scheduling across nodes
type Scheduler struct {
	config             SchedulerConfig
	aiConfig           AISchedulerConfig
	nodes              map[string]*NodeResources
	requests           map[string]*ResourceRequest
	allocations        map[string]*ResourceAllocation
	tasks              map[string]*TaskDistribution
	nodeMutex          sync.RWMutex
	requestMutex       sync.RWMutex
	allocationMutex    sync.RWMutex
	taskMutex          sync.RWMutex
	ctx                context.Context
	cancel             context.CancelFunc
	networkProvider    NetworkTopologyProvider
	aiProvider         AIProvider
	httpClient         *http.Client
	metricsHistory     []AIMetricsData
	metricsMutex       sync.RWMutex
	globalPool         *GlobalResourcePool   // Global resource pooling for federated scheduling
	federationProvider FederationProvider   // Provider for cross-cluster communication
}

// SetNetworkProvider sets the network topology provider for network-aware scheduling
func (s *Scheduler) SetNetworkProvider(provider NetworkTopologyProvider) {
	s.networkProvider = provider
}

// SetAIProvider sets the AI provider for intelligent scheduling
func (s *Scheduler) SetAIProvider(provider AIProvider) {
	s.aiProvider = provider
}

// DefaultAISchedulerConfig returns a default AI scheduler configuration
func DefaultAISchedulerConfig() AISchedulerConfig {
	return AISchedulerConfig{
		Enabled:                   false,
		AIEngineURL:              "http://localhost:8095",
		PredictionHorizon:        60,
		ConfidenceThreshold:      0.7,
		AnomalySensitivity:       0.1,
		MaxRetries:               3,
		RequestTimeout:           30 * time.Second,
		MetricsCollectionInterval: 5 * time.Minute,
		EnableProactiveScaling:    true,
		EnableAnomalyDetection:    true,
		AIWeightInScheduling:      0.3,
	}
}

// HTTPAIProvider implements AIProvider using HTTP calls to AI engine
type HTTPAIProvider struct {
	baseURL    string
	httpClient *http.Client
	maxRetries int
}

// NewHTTPAIProvider creates a new HTTP-based AI provider
func NewHTTPAIProvider(baseURL string, timeout time.Duration, maxRetries int) *HTTPAIProvider {
	return &HTTPAIProvider{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
		maxRetries: maxRetries,
	}
}

// PredictResourceDemand predicts resource demand using AI engine
func (p *HTTPAIProvider) PredictResourceDemand(nodeID string, resourceType ResourceType, horizonMinutes int) ([]float64, float64, error) {
	requestData := map[string]interface{}{
		"resource_id":        nodeID,
		"metrics":           map[string]float64{string(resourceType) + "_usage": 0.5},
		"prediction_horizon": horizonMinutes,
	}

	response, err := p.makeRequest("POST", "/predict/resources", requestData)
	if err != nil {
		return nil, 0.0, err
	}

	var result struct {
		Prediction interface{} `json:"prediction"`
		Confidence float64     `json:"confidence"`
	}

	if err := json.Unmarshal(response, &result); err != nil {
		return nil, 0.0, err
	}

	// Handle both single value and array predictions
	var predictions []float64
	switch v := result.Prediction.(type) {
	case float64:
		predictions = []float64{v}
	case []interface{}:
		predictions = make([]float64, len(v))
		for i, val := range v {
			if f, ok := val.(float64); ok {
				predictions[i] = f
			}
		}
	default:
		predictions = []float64{0.5} // fallback
	}

	return predictions, result.Confidence, nil
}

// OptimizePerformance optimizes performance using AI engine
func (p *HTTPAIProvider) OptimizePerformance(clusterData map[string]interface{}, goals []string) (map[string]interface{}, error) {
	requestData := map[string]interface{}{
		"cluster_id":        "default",
		"performance_data":  []map[string]interface{}{clusterData},
		"optimization_goals": goals,
	}

	response, err := p.makeRequest("POST", "/optimize/performance", requestData)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(response, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// AnalyzeWorkload analyzes workload patterns using AI engine
func (p *HTTPAIProvider) AnalyzeWorkload(vmID string, workloadData []map[string]interface{}) (map[string]interface{}, error) {
	requestData := map[string]interface{}{
		"vm_id":        vmID,
		"workload_data": workloadData,
	}

	response, err := p.makeRequest("POST", "/analyze/workload", requestData)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(response, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// DetectAnomalies detects anomalies using AI engine
func (p *HTTPAIProvider) DetectAnomalies(metrics map[string]float64) (bool, float64, []string, error) {
	requestData := map[string]interface{}{
		"resource_id": "node_metrics",
		"metrics":     metrics,
	}

	response, err := p.makeRequest("POST", "/detect/anomaly", requestData)
	if err != nil {
		return false, 0.0, nil, err
	}

	var result struct {
		IsAnomaly       bool     `json:"is_anomaly"`
		AnomalyScore    float64  `json:"anomaly_score"`
		Recommendations []string `json:"recommendations"`
	}

	if err := json.Unmarshal(response, &result); err != nil {
		return false, 0.0, nil, err
	}

	return result.IsAnomaly, result.AnomalyScore, result.Recommendations, nil
}

// GetScalingRecommendations gets scaling recommendations using AI engine
func (p *HTTPAIProvider) GetScalingRecommendations(vmID string, currentResources map[string]float64, historicalData []map[string]interface{}) ([]map[string]interface{}, error) {
	requestData := map[string]interface{}{
		"vm_id":             vmID,
		"current_resources": currentResources,
		"historical_data":   historicalData,
	}

	response, err := p.makeRequest("POST", "/optimize/scaling", requestData)
	if err != nil {
		return nil, err
	}

	var result struct {
		Decisions []map[string]interface{} `json:"decisions"`
	}

	if err := json.Unmarshal(response, &result); err != nil {
		return nil, err
	}

	return result.Decisions, nil
}

// OptimizeMigration optimizes migration using AI engine
func (p *HTTPAIProvider) OptimizeMigration(vmID string, sourceHost string, targetHosts []string, vmMetrics map[string]float64) (map[string]interface{}, error) {
	requestData := map[string]interface{}{
		"vm_id":        vmID,
		"source_host":  sourceHost,
		"target_hosts": targetHosts,
		"vm_metrics":   vmMetrics,
	}

	response, err := p.makeRequest("POST", "/predict/migration", requestData)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(response, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// OptimizeBandwidth optimizes bandwidth using AI engine
func (p *HTTPAIProvider) OptimizeBandwidth(networkID string, trafficData []map[string]interface{}, qosRequirements map[string]float64) (map[string]interface{}, error) {
	requestData := map[string]interface{}{
		"network_id":       networkID,
		"traffic_data":     trafficData,
		"qos_requirements": qosRequirements,
	}

	response, err := p.makeRequest("POST", "/optimize/bandwidth", requestData)
	if err != nil {
		return nil, err
	}

	var result map[string]interface{}
	if err := json.Unmarshal(response, &result); err != nil {
		return nil, err
	}

	return result, nil
}

// makeRequest makes HTTP request to AI engine with retries
func (p *HTTPAIProvider) makeRequest(method, endpoint string, data interface{}) ([]byte, error) {
	var lastErr error

	for attempt := 0; attempt <= p.maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff
			time.Sleep(time.Duration(math.Pow(2, float64(attempt))) * time.Second)
		}

		requestBytes, err := json.Marshal(data)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request data: %v", err)
		}

		req, err := http.NewRequest(method, p.baseURL+endpoint, bytes.NewBuffer(requestBytes))
		if err != nil {
			lastErr = err
			continue
		}

		req.Header.Set("Content-Type", "application/json")

		resp, err := p.httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("AI engine returned status %d", resp.StatusCode)
			continue
		}

		var responseBytes bytes.Buffer
		_, err = responseBytes.ReadFrom(resp.Body)
		if err != nil {
			lastErr = err
			continue
		}

		return responseBytes.Bytes(), nil
	}

	return nil, fmt.Errorf("AI request failed after %d retries: %v", p.maxRetries, lastErr)
}

// NewScheduler creates a new scheduler
func NewScheduler(config SchedulerConfig) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())

	// Validate and normalize weights in the provided config
	if err := ValidateAndNormalizeWeights(&config); err != nil {
		log.Printf("Warning: Weight validation failed, using normalized values: %v", err)
	}

	return &Scheduler{
		config:         config,
		aiConfig:       DefaultAISchedulerConfig(),
		nodes:          make(map[string]*NodeResources),
		requests:       make(map[string]*ResourceRequest),
		allocations:    make(map[string]*ResourceAllocation),
		tasks:          make(map[string]*TaskDistribution),
		ctx:            ctx,
		cancel:         cancel,
		httpClient:     &http.Client{Timeout: 30 * time.Second},
		metricsHistory: make([]AIMetricsData, 0),
	}
}

// NewSchedulerWithAI creates a new scheduler with AI configuration
func NewSchedulerWithAI(config SchedulerConfig, aiConfig AISchedulerConfig) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())

	// Validate and normalize weights in the provided config
	if err := ValidateAndNormalizeWeights(&config); err != nil {
		log.Printf("Warning: Weight validation failed, using normalized values: %v", err)
	}

	scheduler := &Scheduler{
		config:         config,
		aiConfig:       aiConfig,
		nodes:          make(map[string]*NodeResources),
		requests:       make(map[string]*ResourceRequest),
		allocations:    make(map[string]*ResourceAllocation),
		tasks:          make(map[string]*TaskDistribution),
		ctx:            ctx,
		cancel:         cancel,
		httpClient:     &http.Client{Timeout: aiConfig.RequestTimeout},
		metricsHistory: make([]AIMetricsData, 0),
	}

	// Initialize AI provider if enabled (with fallback)
	if aiConfig.Enabled {
		httpAIProvider := NewHTTPAIProvider(aiConfig.AIEngineURL, aiConfig.RequestTimeout, aiConfig.MaxRetries)
		safeAIProvider := NewSafeAIProvider(httpAIProvider, config)
		scheduler.SetAIProvider(safeAIProvider)
		log.Printf("AI-powered scheduling enabled with endpoint: %s (with fallback)", aiConfig.AIEngineURL)
	} else {
		// Even without AI config, use fallback for defensive programming
		fallbackProvider := NewSafeAIProvider(nil, config)
		scheduler.SetAIProvider(fallbackProvider)
		log.Printf("AI scheduling disabled, using heuristic fallback only")
	}

	return scheduler
}

// Start starts the scheduler
func (s *Scheduler) Start() error {
	log.Println("Starting scheduler")

	// Start the allocation loop
	go s.allocationLoop()

	// Start the cleanup loop
	go s.cleanupLoop()

	// Start AI metrics collection if enabled
	if s.aiConfig.Enabled {
		go s.aiMetricsCollectionLoop()
		log.Println("AI metrics collection started")
	}

	return nil
}

// Stop stops the scheduler
func (s *Scheduler) Stop() error {
	log.Println("Stopping scheduler")

	s.cancel()

	return nil
}

// RegisterNode registers a node with the scheduler
func (s *Scheduler) RegisterNode(nodeID string, resources map[ResourceType]*Resource) error {
	s.nodeMutex.Lock()
	defer s.nodeMutex.Unlock()

	s.nodes[nodeID] = &NodeResources{
		NodeID:     nodeID,
		Resources:  resources,
		LastUpdate: time.Now(),
		Metrics:    make(map[string]float64),
		Available:  true,
	}

	log.Printf("Registered node %s with resources: %v", nodeID, resources)

	return nil
}

// UpdateNodeResources updates resources for a node
func (s *Scheduler) UpdateNodeResources(nodeID string, resources map[ResourceType]*Resource) error {
	s.nodeMutex.Lock()
	defer s.nodeMutex.Unlock()

	node, exists := s.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node %s not found", nodeID)
	}

	node.Resources = resources
	node.LastUpdate = time.Now()

	return nil
}

// RequestResources requests resources
func (s *Scheduler) RequestResources(constraints []ResourceConstraint, priority int, timeout time.Duration) (string, error) {
	return s.RequestResourcesWithNetworkConstraints(constraints, nil, priority, timeout)
}

// RequestResourcesWithNetworkConstraints requests resources with optional network constraints
func (s *Scheduler) RequestResourcesWithNetworkConstraints(constraints []ResourceConstraint, networkConstraints *NetworkConstraint, priority int, timeout time.Duration) (string, error) {
	// Generate a unique ID for the request
	requestID := generateID()

	// Create the request
	request := &ResourceRequest{
		ID:                 requestID,
		Constraints:        constraints,
		NetworkConstraints: networkConstraints,
		Priority:           priority,
		Timeout:            timeout,
		CreatedAt:          time.Now(),
		ExpiresAt:          time.Now().Add(timeout),
	}

	// Store the request
	s.requestMutex.Lock()
	s.requests[requestID] = request
	s.requestMutex.Unlock()

	log.Printf("Created resource request %s with constraints: %v", requestID, constraints)

	return requestID, nil
}

// CancelRequest cancels a resource request
func (s *Scheduler) CancelRequest(requestID string) error {
	s.requestMutex.Lock()
	defer s.requestMutex.Unlock()

	_, exists := s.requests[requestID]
	if !exists {
		return fmt.Errorf("request %s not found", requestID)
	}

	delete(s.requests, requestID)

	return nil
}

// ReleaseAllocation releases a resource allocation
func (s *Scheduler) ReleaseAllocation(allocationID string) error {
	s.allocationMutex.Lock()
	defer s.allocationMutex.Unlock()

	allocation, exists := s.allocations[allocationID]
	if !exists {
		return fmt.Errorf("allocation %s not found", allocationID)
	}

	allocation.Released = true

	return nil
}

// DistributeTask distributes a task across nodes
func (s *Scheduler) DistributeTask(requestID string, targetNodeCount int) (string, error) {
	s.requestMutex.RLock()
	request, exists := s.requests[requestID]
	s.requestMutex.RUnlock()

	if !exists {
		return "", fmt.Errorf("request %s not found", requestID)
	}

	// Generate a unique ID for the task
	taskID := generateID()

	// Create the task
	task := &TaskDistribution{
		TaskID:          taskID,
		ResourceRequest: *request,
		TargetNodeCount: targetNodeCount,
		Allocations:     []ResourceAllocation{},
		DistributedAt:   time.Now(),
		Status:          TaskPending,
	}

	// Store the task
	s.taskMutex.Lock()
	s.tasks[taskID] = task
	s.taskMutex.Unlock()

	log.Printf("Created task %s for request %s", taskID, requestID)

	return taskID, nil
}

// GetTaskStatus returns the status of a task
func (s *Scheduler) GetTaskStatus(taskID string) (TaskStatus, error) {
	s.taskMutex.RLock()
	defer s.taskMutex.RUnlock()

	task, exists := s.tasks[taskID]
	if !exists {
		return "", fmt.Errorf("task %s not found", taskID)
	}

	return task.Status, nil
}

// allocationLoop periodically allocates resources
func (s *Scheduler) allocationLoop() {
	ticker := time.NewTicker(s.config.AllocationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.allocateResources()
		}
	}
}

// cleanupLoop periodically cleans up expired requests and allocations
func (s *Scheduler) cleanupLoop() {
	ticker := time.NewTicker(s.config.AllocationInterval * 2)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.cleanupExpired()
		}
	}
}

// allocateResources allocates resources to pending requests
func (s *Scheduler) allocateResources() {
	// Read the current state
	s.requestMutex.RLock()
	pendingRequests := make([]*ResourceRequest, 0)
	for _, request := range s.requests {
		pendingRequests = append(pendingRequests, request)
	}
	s.requestMutex.RUnlock()

	// Sort requests by priority (higher priority first)
	sort.Slice(pendingRequests, func(i, j int) bool {
		return pendingRequests[i].Priority > pendingRequests[j].Priority
	})

	// Get available nodes
	s.nodeMutex.RLock()
	availableNodes := make([]*NodeResources, 0)
	for _, node := range s.nodes {
		if node.Available {
			availableNodes = append(availableNodes, node)
		}
	}
	s.nodeMutex.RUnlock()

	// Check if we have enough nodes
	if len(availableNodes) < s.config.MinimumNodeCount {
		log.Printf("Not enough available nodes (%d < %d), skipping allocation", len(availableNodes), s.config.MinimumNodeCount)
		return
	}

	// Allocate resources for each request
	for _, request := range pendingRequests {
		// Skip expired requests
		if time.Now().After(request.ExpiresAt) {
			continue
		}

		// Find best node for the request
		bestNode, allocationPossible := s.findBestNode(request, availableNodes)
		if !allocationPossible {
			log.Printf("Could not allocate resources for request %s", request.ID)
			continue
		}

		// Allocate resources on the best node
		allocation, err := s.allocateResourcesOnNode(request, bestNode)
		if err != nil {
			log.Printf("Error allocating resources on node %s: %v", bestNode.NodeID, err)
			continue
		}

		// Store the allocation
		s.allocationMutex.Lock()
		s.allocations[request.ID] = allocation
		s.allocationMutex.Unlock()

		// Update node resources
		s.nodeMutex.Lock()
		for resourceType, amount := range allocation.Resources {
			bestNode.Resources[resourceType].Used += amount
		}
		s.nodeMutex.Unlock()

		// Update tasks
		s.updateTasksWithAllocation(allocation)

		log.Printf("Allocated resources for request %s on node %s", request.ID, bestNode.NodeID)
	}
}

// findBestNode finds the best node for a request
func (s *Scheduler) findBestNode(request *ResourceRequest, nodes []*NodeResources) (*NodeResources, bool) {
	// Check if we should limit allocation based on network utilization
	if s.config.NetworkAwarenessEnabled && s.networkProvider != nil {
		utilization := s.networkProvider.GetNetworkUtilization()
		for nodeID, u := range utilization {
			if u > s.config.MaxNetworkUtilization {
				// Network is overloaded, be more conservative with allocation
				log.Printf("High network utilization detected on %s (%.1f%%), applying conservative scheduling", nodeID, u)
				// Filter out nodes with high network utilization
				newNodes := make([]*NodeResources, 0)
				for _, node := range nodes {
					if node.NodeID != nodeID {
						newNodes = append(newNodes, node)
					}
				}
				nodes = newNodes
			}
		}
		// If bandwidth prediction is enabled, use it to filter nodes
		if s.config.BandwidthPredictionEnabled {
			// This would integrate with a predictor interface
			log.Printf("Bandwidth prediction enabled for scheduling decision")
		}
	}
	// Filter nodes that can fulfill the request
	candidates := make([]*NodeResources, 0)
	for _, node := range nodes {
		if s.canNodeFulfillRequest(node, request) {
			candidates = append(candidates, node)
		}
	}

	if len(candidates) == 0 {
		return nil, false
	}

	// Score candidates based on resource availability and load balancing
	type nodeScore struct {
		node  *NodeResources
		score float64
	}

	scores := make([]nodeScore, len(candidates))

	for i, node := range candidates {
		// Calculate resource efficiency score (higher is better)
		resourceScore := s.calculateResourceScore(node, request)

		// Calculate load balancing score (higher is better)
		loadScore := s.calculateLoadScore(node)

		// Calculate network score if enabled
		networkScore := 0.5 // Default neutral score
		if s.config.NetworkAwarenessEnabled && s.networkProvider != nil {
			if bw, err := s.networkProvider.GetBandwidthAvailability(node.NodeID); err == nil {
				networkScore = bw / 100.0 // Normalize to 0-1
			}
		}

		// Combine scores based on weights
		var combinedScore float64
		if s.config.NetworkAwarenessEnabled {
			combinedScore = s.config.BalancingWeight*loadScore + 
				(1-s.config.BalancingWeight-s.config.NetworkScoreWeight)*resourceScore +
				s.config.NetworkScoreWeight*networkScore
		} else {
			combinedScore = s.config.BalancingWeight*loadScore + (1-s.config.BalancingWeight)*resourceScore
		}

		scores[i] = nodeScore{node: node, score: combinedScore}
	}

	// Sort by score (higher is better)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	return scores[0].node, true
}

// canNodeFulfillRequest checks if a node can fulfill a request
func (s *Scheduler) canNodeFulfillRequest(node *NodeResources, request *ResourceRequest) bool {
	// Check basic resource constraints
	for _, constraint := range request.Constraints {
		resource, exists := node.Resources[constraint.Type]
		if !exists {
			return false
		}

		// Apply overcommit ratio
		overcommitRatio, exists := s.config.OvercommitRatio[constraint.Type]
		if !exists {
			overcommitRatio = 1.0
		}

		availableWithOvercommit := resource.Available() * overcommitRatio

		if constraint.MinAmount > availableWithOvercommit {
			return false
		}
	}

	// Check network constraints if enabled and provider is available
	if s.config.NetworkAwarenessEnabled && s.networkProvider != nil {
		// Check general network utilization threshold
		utilization := s.networkProvider.GetNetworkUtilization()
		if nodeUtil, exists := utilization[node.NodeID]; exists {
			if nodeUtil > s.config.MaxNetworkUtilization {
				// Node's network is too utilized
				return false
			}
		}
		
		// Validate specific network constraints if present
		if request.NetworkConstraints != nil {
			if err := s.validateNetworkConstraintsForNode(node.NodeID, request.NetworkConstraints); err != nil {
				log.Printf("Node %s failed network constraint validation: %v", node.NodeID, err)
				return false
			}
		}
	}

	return true
}

// validateNetworkConstraintsForNode validates network constraints for a specific node
func (s *Scheduler) validateNetworkConstraintsForNode(nodeID string, constraints *NetworkConstraint) error {
	if s.networkProvider == nil {
		return fmt.Errorf("network provider not available")
	}
	
	// Check minimum bandwidth requirement
	if constraints.MinBandwidthBps > 0 {
		bandwidthCap, err := s.networkProvider.GetBandwidthCapability(nodeID)
		if err != nil {
			return fmt.Errorf("failed to get bandwidth capability: %w", err)
		}
		
		// Get current bandwidth availability
		bandwidthAvail, err := s.networkProvider.GetBandwidthAvailability(nodeID)
		if err != nil {
			return fmt.Errorf("failed to get bandwidth availability: %w", err)
		}
		
		// Calculate available bandwidth in bps (availability is in percentage)
		availableBps := uint64(float64(bandwidthCap) * (bandwidthAvail / 100.0))
		
		if constraints.MinBandwidthBps > availableBps {
			return fmt.Errorf("insufficient bandwidth: required %d bps, available %d bps", 
				constraints.MinBandwidthBps, availableBps)
		}
	}
	
	// Check maximum latency requirement
	if constraints.MaxLatencyMs > 0 {
		// Check latency to all other nodes to ensure the constraint can be met
		s.nodeMutex.RLock()
		otherNodes := make([]string, 0)
		for nID := range s.nodes {
			if nID != nodeID {
				otherNodes = append(otherNodes, nID)
			}
		}
		s.nodeMutex.RUnlock()
		
		// Validate latency requirements against other nodes
		for _, otherNodeID := range otherNodes {
			latency, err := s.networkProvider.GetLatencyBetweenNodes(nodeID, otherNodeID)
			if err != nil {
				// Log warning but don't fail validation if latency data is unavailable
				log.Printf("Warning: Failed to get latency between %s and %s: %v", nodeID, otherNodeID, err)
				continue
			}
			
			if latency > constraints.MaxLatencyMs {
				return fmt.Errorf("latency constraint violated: latency to node %s is %.2f ms, maximum allowed is %.2f ms", 
					otherNodeID, latency, constraints.MaxLatencyMs)
			}
		}
	}
	
	// Check minimum connections requirement
	if constraints.MinConnections > 0 {
		connectionCount := len(otherNodes) // Number of other nodes this node can connect to
		if connectionCount < constraints.MinConnections {
			return fmt.Errorf("insufficient network connections: node has %d connections, minimum required is %d", 
				connectionCount, constraints.MinConnections)
		}
	}
	
	// Use the NetworkTopologyProvider's validation if it supports constraints validation
	if err := s.networkProvider.ValidateNetworkConstraints(nodeID, constraints); err != nil {
		return fmt.Errorf("network constraint validation failed: %w", err)
	}
	
	return nil
}

// calculateResourceScore calculates a score for resource efficiency
func (s *Scheduler) calculateResourceScore(node *NodeResources, request *ResourceRequest) float64 {
	// Higher score means better resource efficiency

	// Calculate average resource utilization after allocation
	total := 0.0
	count := 0

	for _, constraint := range request.Constraints {
		resource, exists := node.Resources[constraint.Type]
		if !exists {
			continue
		}

		// Calculate utilization after allocation
		requestAmount := constraint.MinAmount
		newUtilization := (resource.Used + requestAmount) / resource.Capacity

		// Add to average
		total += newUtilization
		count++
	}

	if count == 0 {
		return 0
	}

	// Average utilization (0-1)
	avgUtilization := total / float64(count)

	// Optimal utilization is around 70-80%
	// Score is highest when utilization is around 75%
	distance := avgUtilization - 0.75

	return 1.0 - (distance * distance * 4.0) // Quadratic function with peak at 0.75
}

// calculateLoadScore calculates a score for load balancing
func (s *Scheduler) calculateLoadScore(node *NodeResources) float64 {
	// Higher score means better for load balancing

	// Calculate average resource utilization
	total := 0.0
	count := 0

	for _, resource := range node.Resources {
		utilization := resource.Used / resource.Capacity
		total += utilization
		count++
	}

	if count == 0 {
		return 0
	}

	// Average utilization (0-1)
	avgUtilization := total / float64(count)

	// Lower utilization is better for load balancing
	return 1.0 - avgUtilization
}

// allocateResourcesOnNode allocates resources on a node
func (s *Scheduler) allocateResourcesOnNode(request *ResourceRequest, node *NodeResources) (*ResourceAllocation, error) {
	resources := make(map[ResourceType]float64)

	for _, constraint := range request.Constraints {
		_, exists := node.Resources[constraint.Type]
		if !exists {
			return nil, fmt.Errorf("resource %s not available on node", constraint.Type)
		}

		// Allocate the minimum amount
		resources[constraint.Type] = constraint.MinAmount
	}

	allocation := &ResourceAllocation{
		RequestID:   request.ID,
		NodeID:      node.NodeID,
		Resources:   resources,
		AllocatedAt: time.Now(),
		ExpiresAt:   request.ExpiresAt,
		Released:    false,
	}

	return allocation, nil
}

// updateTasksWithAllocation updates tasks with a new allocation
func (s *Scheduler) updateTasksWithAllocation(allocation *ResourceAllocation) {
	s.taskMutex.Lock()
	defer s.taskMutex.Unlock()

	for _, task := range s.tasks {
		if task.ResourceRequest.ID == allocation.RequestID {
			// Add the allocation to the task
			task.Allocations = append(task.Allocations, *allocation)

			// Update task status if target node count reached
			if len(task.Allocations) >= task.TargetNodeCount {
				task.Status = TaskAllocated
			}
		}
	}
}

// cleanupExpired cleans up expired requests and allocations
func (s *Scheduler) cleanupExpired() {
	now := time.Now()

	// Clean up expired requests
	s.requestMutex.Lock()
	for id, request := range s.requests {
		if now.After(request.ExpiresAt) {
			delete(s.requests, id)
			log.Printf("Cleaned up expired request %s", id)
		}
	}
	s.requestMutex.Unlock()

	// Clean up expired allocations
	s.allocationMutex.Lock()
	for id, allocation := range s.allocations {
		if now.After(allocation.ExpiresAt) || allocation.Released {
			delete(s.allocations, id)
			log.Printf("Cleaned up expired allocation %s", id)

			// Update node resources
			s.nodeMutex.Lock()
			node, exists := s.nodes[allocation.NodeID]
			if exists {
				for resourceType, amount := range allocation.Resources {
					if resource, ok := node.Resources[resourceType]; ok {
						resource.Used -= amount
						if resource.Used < 0 {
							resource.Used = 0
						}
					}
				}
			}
			s.nodeMutex.Unlock()
		}
	}
	s.allocationMutex.Unlock()

	// Clean up nodes that haven't been updated
	s.nodeMutex.Lock()
	for id, node := range s.nodes {
		if now.Sub(node.LastUpdate) > s.config.NodeTimeout {
			node.Available = false
			log.Printf("Marked node %s as unavailable due to timeout", id)
		}
	}
	s.nodeMutex.Unlock()
}

// GetNodesStatus returns the status of all nodes
func (s *Scheduler) GetNodesStatus() map[string]NodeResources {
	s.nodeMutex.RLock()
	defer s.nodeMutex.RUnlock()

	result := make(map[string]NodeResources)
	for id, node := range s.nodes {
		result[id] = *node
	}

	return result
}

// GetPendingRequests returns all pending requests
func (s *Scheduler) GetPendingRequests() map[string]ResourceRequest {
	s.requestMutex.RLock()
	defer s.requestMutex.RUnlock()

	result := make(map[string]ResourceRequest)
	for id, request := range s.requests {
		result[id] = *request
	}

	return result
}

// GetActiveAllocations returns all active allocations
func (s *Scheduler) GetActiveAllocations() map[string]ResourceAllocation {
	s.allocationMutex.RLock()
	defer s.allocationMutex.RUnlock()

	result := make(map[string]ResourceAllocation)
	for id, allocation := range s.allocations {
		if !allocation.Released && time.Now().Before(allocation.ExpiresAt) {
			result[id] = *allocation
		}
	}

	return result
}

// GetTasks returns all tasks
func (s *Scheduler) GetTasks() map[string]TaskDistribution {
	s.taskMutex.RLock()
	defer s.taskMutex.RUnlock()

	result := make(map[string]TaskDistribution)
	for id, task := range s.tasks {
		result[id] = *task
	}

	return result
}

// AI-Enhanced Methods

// aiMetricsCollectionLoop collects metrics for AI analysis
func (s *Scheduler) aiMetricsCollectionLoop() {
	ticker := time.NewTicker(s.aiConfig.MetricsCollectionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.collectMetricsForAI()
		}
	}
}

// collectMetricsForAI collects current metrics and sends to AI engine
func (s *Scheduler) collectMetricsForAI() {
	s.nodeMutex.RLock()
	defer s.nodeMutex.RUnlock()

	timestamp := time.Now()

	for nodeID, node := range s.nodes {
		if !node.Available {
			continue
		}

		for resourceType, resource := range node.Resources {
			metricsData := AIMetricsData{
				Timestamp:    timestamp,
				NodeID:       nodeID,
				ResourceType: resourceType,
				CurrentUsage: resource.Used,
				Capacity:     resource.Capacity,
				Metrics:      node.Metrics,
				Metadata: map[string]interface{}{
					"available_percentage": resource.AvailablePercentage(),
					"last_update":         node.LastUpdate,
				},
			}

			// Store metrics history
			s.metricsMutex.Lock()
			s.metricsHistory = append(s.metricsHistory, metricsData)

			// Keep only recent metrics (last hour)
			cutoff := timestamp.Add(-time.Hour)
			filteredMetrics := s.metricsHistory[:0]
			for _, m := range s.metricsHistory {
				if m.Timestamp.After(cutoff) {
					filteredMetrics = append(filteredMetrics, m)
				}
			}
			s.metricsHistory = filteredMetrics
			s.metricsMutex.Unlock()

			// Check for anomalies if enabled
			if s.aiConfig.EnableAnomalyDetection && s.aiProvider != nil {
				go s.checkForAnomalies(nodeID, node.Metrics)
			}
		}
	}
}

// checkForAnomalies checks for anomalies in node metrics
func (s *Scheduler) checkForAnomalies(nodeID string, metrics map[string]float64) {
	if s.aiProvider == nil {
		return
	}

	isAnomaly, score, recommendations, err := s.aiProvider.DetectAnomalies(metrics)
	if err != nil {
		log.Printf("AI anomaly detection failed for node %s: %v", nodeID, err)
		return
	}

	if isAnomaly && score > s.aiConfig.AnomalySensitivity {
		log.Printf("ANOMALY DETECTED on node %s: score=%.3f, recommendations=%v",
			nodeID, score, recommendations)

		// Mark node for investigation
		s.nodeMutex.Lock()
		if node, exists := s.nodes[nodeID]; exists {
			if node.Metrics == nil {
				node.Metrics = make(map[string]float64)
			}
			node.Metrics["anomaly_score"] = score
			node.Metrics["anomaly_detected_at"] = float64(time.Now().Unix())
		}
		s.nodeMutex.Unlock()
	}
}

// GetAISchedulingRecommendations gets AI-based scheduling recommendations
func (s *Scheduler) GetAISchedulingRecommendations(constraints []ResourceConstraint) ([]AISchedulingRecommendation, error) {
	if !s.aiConfig.Enabled || s.aiProvider == nil {
		return nil, fmt.Errorf("AI scheduling not enabled")
	}

	var recommendations []AISchedulingRecommendation

	s.nodeMutex.RLock()
	defer s.nodeMutex.RUnlock()

	for nodeID, node := range s.nodes {
		if !node.Available {
			continue
		}

		// Get AI predictions for this node
		var nodeScore float64 = 0.5
		var confidence float64 = 0.5
		var reasoning []string

		// Predict resource demand for each resource type
		predictedLoad := make(map[ResourceType]float64)
		riskAssessment := make(map[string]float64)

		for resourceType := range node.Resources {
			predictions, pred_confidence, err := s.aiProvider.PredictResourceDemand(
				nodeID, resourceType, s.aiConfig.PredictionHorizon,
			)

			if err == nil && len(predictions) > 0 {
				// Use average of predictions
				var sum float64
				for _, pred := range predictions {
					sum += pred
				}
				avgPrediction := sum / float64(len(predictions))
				predictedLoad[resourceType] = avgPrediction

				// Update confidence and scoring
				confidence = (confidence + pred_confidence) / 2

				// Calculate risk based on predicted utilization
				if avgPrediction > 0.8 {
					riskAssessment[string(resourceType)+"_overload"] = avgPrediction
					reasoning = append(reasoning, fmt.Sprintf("High %s utilization predicted: %.2f", resourceType, avgPrediction))
				} else if avgPrediction < 0.3 {
					riskAssessment[string(resourceType)+"_underutilized"] = 1.0 - avgPrediction
					reasoning = append(reasoning, fmt.Sprintf("Low %s utilization predicted: %.2f", resourceType, avgPrediction))
				}
			} else {
				// Fallback to current utilization
				if resource, ok := node.Resources[resourceType]; ok {
					predictedLoad[resourceType] = resource.Used / resource.Capacity
				}
			}
		}

		// Calculate overall node score
		nodeScore = s.calculateAINodeScore(node, predictedLoad, constraints)

		// Generate optimization tips
		var optimizationTips []string
		if nodeScore > 0.8 {
			optimizationTips = append(optimizationTips, "Consider migrating workloads from this node")
		} else if nodeScore < 0.3 {
			optimizationTips = append(optimizationTips, "This node has capacity for additional workloads")
		}

		if confidence < s.aiConfig.ConfidenceThreshold {
			optimizationTips = append(optimizationTips, "Low prediction confidence - monitor closely")
		}

		recommendations = append(recommendations, AISchedulingRecommendation{
			NodeID:          nodeID,
			Score:           nodeScore,
			Confidence:      confidence,
			Reasoning:       reasoning,
			PredictedLoad:   predictedLoad,
			RiskAssessment:  riskAssessment,
			OptimizationTips: optimizationTips,
		})
	}

	// Sort by score (best nodes first)
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].Score > recommendations[j].Score
	})

	return recommendations, nil
}

// calculateAINodeScore calculates AI-enhanced node suitability score
func (s *Scheduler) calculateAINodeScore(node *NodeResources, predictedLoad map[ResourceType]float64, constraints []ResourceConstraint) float64 {
	var score float64 = 1.0

	// Factor in predicted resource utilization
	for resourceType, predictedUtil := range predictedLoad {
		if predictedUtil > 0.9 { // Very high utilization predicted
			score *= 0.2
		} else if predictedUtil > 0.8 { // High utilization predicted
			score *= 0.5
		} else if predictedUtil < 0.2 { // Very low utilization - good for new workloads
			score *= 1.2
		}
	}

	// Factor in constraints satisfaction with AI predictions
	for _, constraint := range constraints {
		if resource, ok := node.Resources[constraint.Type]; ok {
			predictedUsage := predictedLoad[constraint.Type] * resource.Capacity
			available := resource.Capacity - predictedUsage

			if available < constraint.MinAmount {
				score *= 0.1 // Severely penalize if constraints won't be met
			} else if available < constraint.MinAmount * 1.2 {
				score *= 0.7 // Penalize if tight fit
			}
		}
	}

	// Factor in anomaly detection
	if anomalyScore, exists := node.Metrics["anomaly_score"]; exists && anomalyScore > 0.5 {
		score *= (1.0 - anomalyScore) // Reduce score for anomalous nodes
	}

	// Ensure score is within bounds
	if score > 1.0 {
		score = 1.0
	} else if score < 0.0 {
		score = 0.0
	}

	return score
}

// OptimizeSchedulingWithAI uses AI to optimize resource allocation
func (s *Scheduler) OptimizeSchedulingWithAI(request *ResourceRequest) (string, error) {
	if !s.aiConfig.Enabled || s.aiProvider == nil {
		// Fall back to standard scheduling
		return s.allocateResources(request)
	}

	// Get AI recommendations
	recommendations, err := s.GetAISchedulingRecommendations(request.Constraints)
	if err != nil {
		log.Printf("AI recommendations failed, falling back to standard scheduling: %v", err)
		return s.allocateResources(request)
	}

	// Try to allocate based on AI recommendations
	for _, rec := range recommendations {
		if rec.Confidence < s.aiConfig.ConfidenceThreshold {
			continue
		}

		// Check if this node can satisfy the request
		s.nodeMutex.RLock()
		node, exists := s.nodes[rec.NodeID]
		if !exists || !node.Available {
			s.nodeMutex.RUnlock()
			continue
		}

		canAllocate := true
		for _, constraint := range request.Constraints {
			resource := node.Resources[constraint.Type]
			if resource == nil || resource.Available() < constraint.MinAmount {
				canAllocate = false
				break
			}
		}
		s.nodeMutex.RUnlock()

		if canAllocate {
			// Attempt allocation on this AI-recommended node
			allocation := s.tryAllocateOnNode(request, rec.NodeID)
			if allocation != nil {
				log.Printf("AI-optimized allocation: request %s allocated to node %s (AI score: %.3f, confidence: %.3f)",
					request.ID, rec.NodeID, rec.Score, rec.Confidence)
				return allocation.RequestID, nil
			}
		}
	}

	// If AI-based allocation fails, fall back to standard allocation
	log.Printf("AI-optimized allocation failed, falling back to standard scheduling for request %s", request.ID)
	return s.allocateResources(request)
}

// tryAllocateOnNode attempts to allocate resources on a specific node
func (s *Scheduler) tryAllocateOnNode(request *ResourceRequest, nodeID string) *ResourceAllocation {
	s.nodeMutex.Lock()
	defer s.nodeMutex.Unlock()
	s.allocationMutex.Lock()
	defer s.allocationMutex.Unlock()

	node := s.nodes[nodeID]
	if node == nil || !node.Available {
		return nil
	}

	// Check if all constraints can be satisfied
	allocatedResources := make(map[ResourceType]float64)
	for _, constraint := range request.Constraints {
		resource := node.Resources[constraint.Type]
		if resource == nil || resource.Available() < constraint.MinAmount {
			return nil
		}
		allocatedResources[constraint.Type] = constraint.MinAmount
	}

	// Create the allocation
	allocation := &ResourceAllocation{
		RequestID:   request.ID,
		NodeID:      nodeID,
		Resources:   allocatedResources,
		AllocatedAt: time.Now(),
		ExpiresAt:   time.Now().Add(request.Timeout),
		Released:    false,
	}

	// Update resource usage
	for resourceType, amount := range allocatedResources {
		resource := node.Resources[resourceType]
		resource.Used += amount
	}

	// Store the allocation
	s.allocations[request.ID] = allocation

	// Remove the request from pending requests
	s.requestMutex.Lock()
	delete(s.requests, request.ID)
	s.requestMutex.Unlock()

	return allocation
}

// GetAIMetrics returns AI-related metrics and status
func (s *Scheduler) GetAIMetrics() map[string]interface{} {
	s.metricsMutex.RLock()
	defer s.metricsMutex.RUnlock()

	totalNodes := 0
	anomalousNodes := 0

	s.nodeMutex.RLock()
	for _, node := range s.nodes {
		totalNodes++
		if score, exists := node.Metrics["anomaly_score"]; exists && score > s.aiConfig.AnomalySensitivity {
			anomalousNodes++
		}
	}
	s.nodeMutex.RUnlock()

	return map[string]interface{}{
		"ai_enabled":              s.aiConfig.Enabled,
		"ai_engine_url":          s.aiConfig.AIEngineURL,
		"metrics_history_count":  len(s.metricsHistory),
		"prediction_horizon_min": s.aiConfig.PredictionHorizon,
		"confidence_threshold":   s.aiConfig.ConfidenceThreshold,
		"anomaly_sensitivity":    s.aiConfig.AnomalySensitivity,
		"total_nodes":           totalNodes,
		"anomalous_nodes":       anomalousNodes,
		"ai_provider_available": s.aiProvider != nil,
		"proactive_scaling":     s.aiConfig.EnableProactiveScaling,
		"anomaly_detection":     s.aiConfig.EnableAnomalyDetection,
		"ai_scheduling_weight":  s.aiConfig.AIWeightInScheduling,
	}
}

// Global Resource Pooling Extensions for Distributed Supercompute Fabric

// ClusterResourceInfo represents resource information for a federated cluster
type ClusterResourceInfo struct {
	ClusterID     string                     `json:"cluster_id"`
	Location      string                     `json:"location"`
	Resources     map[ResourceType]*Resource `json:"resources"`
	Nodes         map[string]*NodeResources  `json:"nodes"`
	NetworkCost   float64                    `json:"network_cost"`
	LastHeartbeat time.Time                  `json:"last_heartbeat"`
	IsHealthy     bool                       `json:"is_healthy"`
	Priority      int                        `json:"priority"`
	Capabilities  []string                   `json:"capabilities"`
}

// GlobalResourcePool manages resources across multiple federated clusters
type GlobalResourcePool struct {
	clusters       map[string]*ClusterResourceInfo
	clusterMutex   sync.RWMutex
	parentCluster  string // ID of the parent cluster
	federationMode bool   // Whether this scheduler is in federation mode
}

// FederationProvider interface for cross-cluster communication
type FederationProvider interface {
	// Cluster management
	GetFederatedClusters() ([]ClusterResourceInfo, error)
	GetClusterResources(clusterID string) (*ClusterResourceInfo, error)
	UpdateClusterHeartbeat(clusterID string) error

	// Resource coordination
	AllocateResourcesOnCluster(clusterID string, request *ResourceRequest) (*ResourceAllocation, error)
	ReleaseResourcesOnCluster(clusterID string, allocationID string) error

	// Network optimization
	GetClusterNetworkCost(fromCluster, toCluster string) (float64, error)
	GetOptimalClusterPath(fromCluster, toCluster string) ([]string, error)

	// Cross-cluster migration support
	InitiateCrossClusterMigration(vmID string, sourceCluster, targetCluster string, migrationOptions map[string]interface{}) error
	GetMigrationStatus(migrationID string) (map[string]interface{}, error)
}

// InitializeGlobalResourcePool initializes the global resource pool for federated scheduling
func (s *Scheduler) InitializeGlobalResourcePool(federationProvider FederationProvider, parentClusterID string) error {
	s.config.NetworkAwarenessEnabled = true // Enable network awareness for federated scheduling

	globalPool := &GlobalResourcePool{
		clusters:       make(map[string]*ClusterResourceInfo),
		parentCluster:  parentClusterID,
		federationMode: true,
	}

	// Set the global pool in scheduler (we'll add this field)
	s.globalPool = globalPool
	s.federationProvider = federationProvider

	log.Printf("Initialized global resource pool for cluster %s", parentClusterID)
	return nil
}

// DiscoverFederatedClusters discovers and registers federated clusters
func (s *Scheduler) DiscoverFederatedClusters() error {
	if s.federationProvider == nil {
		return fmt.Errorf("federation provider not initialized")
	}

	clusters, err := s.federationProvider.GetFederatedClusters()
	if err != nil {
		return fmt.Errorf("failed to discover federated clusters: %w", err)
	}

	s.globalPool.clusterMutex.Lock()
	defer s.globalPool.clusterMutex.Unlock()

	for _, cluster := range clusters {
		clusterCopy := cluster
		s.globalPool.clusters[cluster.ClusterID] = &clusterCopy
		log.Printf("Registered federated cluster %s at %s with %d nodes",
			cluster.ClusterID, cluster.Location, len(cluster.Nodes))
	}

	return nil
}

// GetGlobalResourceInventory returns resource inventory across all federated clusters
func (s *Scheduler) GetGlobalResourceInventory() (map[string]ClusterResourceInfo, error) {
	if s.globalPool == nil {
		return nil, fmt.Errorf("global resource pool not initialized")
	}

	s.globalPool.clusterMutex.RLock()
	defer s.globalPool.clusterMutex.RUnlock()

	inventory := make(map[string]ClusterResourceInfo)
	for clusterID, cluster := range s.globalPool.clusters {
		if cluster.IsHealthy && time.Since(cluster.LastHeartbeat) < 5*time.Minute {
			inventory[clusterID] = *cluster
		}
	}

	// Add local cluster resources
	localResources := make(map[ResourceType]*Resource)
	s.nodeMutex.RLock()
	for _, node := range s.nodes {
		if !node.Available {
			continue
		}
		for resourceType, resource := range node.Resources {
			if localResources[resourceType] == nil {
				localResources[resourceType] = &Resource{
					Type:     resourceType,
					Capacity: 0,
					Used:     0,
				}
			}
			localResources[resourceType].Capacity += resource.Capacity
			localResources[resourceType].Used += resource.Used
		}
	}
	s.nodeMutex.RUnlock()

	inventory[s.globalPool.parentCluster] = ClusterResourceInfo{
		ClusterID:     s.globalPool.parentCluster,
		Location:      "local",
		Resources:     localResources,
		Nodes:         s.GetNodesStatus(),
		NetworkCost:   0.0, // Local cluster has no network cost
		LastHeartbeat: time.Now(),
		IsHealthy:     true,
		Priority:      1,
		Capabilities:  []string{"compute", "storage", "networking"},
	}

	return inventory, nil
}

// ScheduleGlobalResourceAllocation schedules resources across federated clusters
func (s *Scheduler) ScheduleGlobalResourceAllocation(request *ResourceRequest) (*ResourceAllocation, error) {
	if s.globalPool == nil || !s.globalPool.federationMode {
		// Fall back to local scheduling
		return s.scheduleLocalResourceAllocation(request)
	}

	// Get global resource inventory
	inventory, err := s.GetGlobalResourceInventory()
	if err != nil {
		log.Printf("Failed to get global inventory, falling back to local: %v", err)
		return s.scheduleLocalResourceAllocation(request)
	}

	// Score clusters for resource allocation
	type clusterScore struct {
		clusterID string
		score     float64
		info      ClusterResourceInfo
	}

	var scores []clusterScore

	for clusterID, cluster := range inventory {
		if !s.canClusterFulfillRequest(&cluster, request) {
			continue
		}

		score := s.calculateGlobalClusterScore(&cluster, request)
		scores = append(scores, clusterScore{
			clusterID: clusterID,
			score:     score,
			info:      cluster,
		})
	}

	if len(scores) == 0 {
		return nil, fmt.Errorf("no clusters can fulfill resource request %s", request.ID)
	}

	// Sort by score (highest first)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Try to allocate on best cluster
	bestCluster := scores[0]

	if bestCluster.clusterID == s.globalPool.parentCluster {
		// Allocate locally
		return s.scheduleLocalResourceAllocation(request)
	} else {
		// Allocate on remote cluster
		return s.allocateOnRemoteCluster(bestCluster.clusterID, request)
	}
}

// canClusterFulfillRequest checks if a cluster can fulfill a resource request
func (s *Scheduler) canClusterFulfillRequest(cluster *ClusterResourceInfo, request *ResourceRequest) bool {
	for _, constraint := range request.Constraints {
		resource, exists := cluster.Resources[constraint.Type]
		if !exists {
			return false
		}

		if resource.Available() < constraint.MinAmount {
			return false
		}
	}

	// Check network constraints for cross-cluster requests
	if request.NetworkConstraints != nil {
		if cluster.NetworkCost > 0 { // Remote cluster
			// For remote clusters, we need to be more conservative with network constraints
			if request.NetworkConstraints.MaxLatencyMs > 0 &&
			   cluster.NetworkCost > request.NetworkConstraints.MaxLatencyMs/100.0 {
				return false
			}
		}
	}

	return true
}

// calculateGlobalClusterScore calculates a score for global cluster selection
func (s *Scheduler) calculateGlobalClusterScore(cluster *ClusterResourceInfo, request *ResourceRequest) float64 {
	score := 1.0

	// Resource availability score
	resourceScore := 0.0
	constraintCount := 0

	for _, constraint := range request.Constraints {
		if resource, exists := cluster.Resources[constraint.Type]; exists {
			utilization := resource.Used / resource.Capacity
			availabilityScore := 1.0 - utilization

			// Prefer clusters with moderate utilization (not completely empty, not overloaded)
			if utilization >= 0.2 && utilization <= 0.7 {
				availabilityScore *= 1.2
			}

			resourceScore += availabilityScore
			constraintCount++
		}
	}

	if constraintCount > 0 {
		resourceScore /= float64(constraintCount)
	}

	score *= resourceScore

	// Network cost score (lower cost is better)
	networkScore := 1.0
	if cluster.NetworkCost > 0 {
		networkScore = 1.0 / (1.0 + cluster.NetworkCost/100.0) // Normalize network cost
	}
	score *= networkScore

	// Priority score
	priorityScore := float64(cluster.Priority) / 10.0
	if priorityScore > 1.0 {
		priorityScore = 1.0
	}
	score *= priorityScore

	// Health score
	if !cluster.IsHealthy {
		score *= 0.1
	}

	// Heartbeat freshness score
	heartbeatAge := time.Since(cluster.LastHeartbeat).Minutes()
	if heartbeatAge > 10.0 {
		score *= 0.5 // Penalize stale heartbeats
	}

	return score
}

// scheduleLocalResourceAllocation schedules resources on the local cluster
func (s *Scheduler) scheduleLocalResourceAllocation(request *ResourceRequest) (*ResourceAllocation, error) {
	// Use existing local scheduling logic
	s.requestMutex.RLock()
	pendingRequests := []*ResourceRequest{request}
	s.requestMutex.RUnlock()

	s.nodeMutex.RLock()
	availableNodes := make([]*NodeResources, 0)
	for _, node := range s.nodes {
		if node.Available {
			availableNodes = append(availableNodes, node)
		}
	}
	s.nodeMutex.RUnlock()

	if len(availableNodes) == 0 {
		return nil, fmt.Errorf("no available nodes in local cluster")
	}

	bestNode, possible := s.findBestNode(request, availableNodes)
	if !possible {
		return nil, fmt.Errorf("cannot allocate resources locally")
	}

	return s.allocateResourcesOnNode(request, bestNode)
}

// allocateOnRemoteCluster allocates resources on a remote federated cluster
func (s *Scheduler) allocateOnRemoteCluster(clusterID string, request *ResourceRequest) (*ResourceAllocation, error) {
	if s.federationProvider == nil {
		return nil, fmt.Errorf("federation provider not available")
	}

	log.Printf("Attempting cross-cluster allocation: request %s on cluster %s", request.ID, clusterID)

	allocation, err := s.federationProvider.AllocateResourcesOnCluster(clusterID, request)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate on remote cluster %s: %w", clusterID, err)
	}

	// Store the allocation in local tracking
	s.allocationMutex.Lock()
	s.allocations[request.ID] = allocation
	s.allocationMutex.Unlock()

	log.Printf("Successfully allocated request %s on remote cluster %s", request.ID, clusterID)

	return allocation, nil
}

// UpdateGlobalResourceUsage updates resource usage across federated clusters
func (s *Scheduler) UpdateGlobalResourceUsage() error {
	if s.globalPool == nil {
		return fmt.Errorf("global resource pool not initialized")
	}

	s.globalPool.clusterMutex.Lock()
	defer s.globalPool.clusterMutex.Unlock()

	for clusterID := range s.globalPool.clusters {
		if clusterID == s.globalPool.parentCluster {
			continue // Skip local cluster, we have direct access
		}

		// Fetch updated resource information from remote cluster
		clusterInfo, err := s.federationProvider.GetClusterResources(clusterID)
		if err != nil {
			log.Printf("Failed to update resources for cluster %s: %v", clusterID, err)
			// Mark cluster as potentially unhealthy
			s.globalPool.clusters[clusterID].IsHealthy = false
			continue
		}

		// Update cluster information
		s.globalPool.clusters[clusterID] = clusterInfo
		s.globalPool.clusters[clusterID].LastHeartbeat = time.Now()
		s.globalPool.clusters[clusterID].IsHealthy = true
	}

	return nil
}

// GetGlobalResourceUtilization returns resource utilization across all clusters
func (s *Scheduler) GetGlobalResourceUtilization() (map[string]map[ResourceType]float64, error) {
	inventory, err := s.GetGlobalResourceInventory()
	if err != nil {
		return nil, err
	}

	utilization := make(map[string]map[ResourceType]float64)

	for clusterID, cluster := range inventory {
		clusterUtil := make(map[ResourceType]float64)
		for resourceType, resource := range cluster.Resources {
			if resource.Capacity > 0 {
				clusterUtil[resourceType] = resource.Used / resource.Capacity
			} else {
				clusterUtil[resourceType] = 0.0
			}
		}
		utilization[clusterID] = clusterUtil
	}

	return utilization, nil
}

// ScheduleGlobalWorkloadDistribution distributes workloads across clusters for optimal performance
func (s *Scheduler) ScheduleGlobalWorkloadDistribution(workloads []ResourceRequest, strategy string) ([]ResourceAllocation, error) {
	var allocations []ResourceAllocation

	switch strategy {
	case "balanced":
		return s.scheduleBalancedDistribution(workloads)
	case "cost-optimized":
		return s.scheduleCostOptimizedDistribution(workloads)
	case "performance-first":
		return s.schedulePerformanceFirstDistribution(workloads)
	case "locality-aware":
		return s.scheduleLocalityAwareDistribution(workloads)
	default:
		return s.scheduleBalancedDistribution(workloads)
	}
}

// scheduleBalancedDistribution distributes workloads evenly across clusters
func (s *Scheduler) scheduleBalancedDistribution(workloads []ResourceRequest) ([]ResourceAllocation, error) {
	inventory, err := s.GetGlobalResourceInventory()
	if err != nil {
		return nil, err
	}

	var allocations []ResourceAllocation
	clusterIDs := make([]string, 0, len(inventory))

	for clusterID := range inventory {
		clusterIDs = append(clusterIDs, clusterID)
	}

	if len(clusterIDs) == 0 {
		return nil, fmt.Errorf("no clusters available for workload distribution")
	}

	// Round-robin distribution
	for i, workload := range workloads {
		targetCluster := clusterIDs[i%len(clusterIDs)]

		var allocation *ResourceAllocation
		if targetCluster == s.globalPool.parentCluster {
			allocation, err = s.scheduleLocalResourceAllocation(&workload)
		} else {
			allocation, err = s.allocateOnRemoteCluster(targetCluster, &workload)
		}

		if err != nil {
			log.Printf("Failed to allocate workload %s on cluster %s: %v", workload.ID, targetCluster, err)
			// Try next cluster
			continue
		}

		allocations = append(allocations, *allocation)
	}

	return allocations, nil
}

// scheduleCostOptimizedDistribution prioritizes cost-effective clusters
func (s *Scheduler) scheduleCostOptimizedDistribution(workloads []ResourceRequest) ([]ResourceAllocation, error) {
	inventory, err := s.GetGlobalResourceInventory()
	if err != nil {
		return nil, err
	}

	// Sort clusters by network cost (lower cost first)
	type clusterCost struct {
		clusterID string
		cost      float64
		info      ClusterResourceInfo
	}

	var sortedClusters []clusterCost
	for clusterID, cluster := range inventory {
		sortedClusters = append(sortedClusters, clusterCost{
			clusterID: clusterID,
			cost:      cluster.NetworkCost,
			info:      cluster,
		})
	}

	sort.Slice(sortedClusters, func(i, j int) bool {
		return sortedClusters[i].cost < sortedClusters[j].cost
	})

	var allocations []ResourceAllocation

	// Allocate workloads to most cost-effective clusters first
	for _, workload := range workloads {
		allocated := false

		for _, cluster := range sortedClusters {
			if !s.canClusterFulfillRequest(&cluster.info, &workload) {
				continue
			}

			var allocation *ResourceAllocation
			if cluster.clusterID == s.globalPool.parentCluster {
				allocation, err = s.scheduleLocalResourceAllocation(&workload)
			} else {
				allocation, err = s.allocateOnRemoteCluster(cluster.clusterID, &workload)
			}

			if err == nil {
				allocations = append(allocations, *allocation)
				allocated = true
				break
			}
		}

		if !allocated {
			log.Printf("Failed to allocate workload %s on any cluster", workload.ID)
		}
	}

	return allocations, nil
}

// schedulePerformanceFirstDistribution prioritizes high-performance clusters
func (s *Scheduler) schedulePerformanceFirstDistribution(workloads []ResourceRequest) ([]ResourceAllocation, error) {
	inventory, err := s.GetGlobalResourceInventory()
	if err != nil {
		return nil, err
	}

	// Score clusters by performance (low network cost, high priority, good resources)
	type clusterPerf struct {
		clusterID string
		perfScore float64
		info      ClusterResourceInfo
	}

	var sortedClusters []clusterPerf
	for clusterID, cluster := range inventory {
		perfScore := float64(cluster.Priority) * 10.0
		if cluster.NetworkCost > 0 {
			perfScore /= (1.0 + cluster.NetworkCost/10.0)
		}

		sortedClusters = append(sortedClusters, clusterPerf{
			clusterID: clusterID,
			perfScore: perfScore,
			info:      cluster,
		})
	}

	sort.Slice(sortedClusters, func(i, j int) bool {
		return sortedClusters[i].perfScore > sortedClusters[j].perfScore
	})

	var allocations []ResourceAllocation

	// Allocate workloads to highest performance clusters first
	for _, workload := range workloads {
		allocated := false

		for _, cluster := range sortedClusters {
			if !s.canClusterFulfillRequest(&cluster.info, &workload) {
				continue
			}

			var allocation *ResourceAllocation
			if cluster.clusterID == s.globalPool.parentCluster {
				allocation, err = s.scheduleLocalResourceAllocation(&workload)
			} else {
				allocation, err = s.allocateOnRemoteCluster(cluster.clusterID, &workload)
			}

			if err == nil {
				allocations = append(allocations, *allocation)
				allocated = true
				break
			}
		}

		if !allocated {
			log.Printf("Failed to allocate workload %s on any cluster", workload.ID)
		}
	}

	return allocations, nil
}

// scheduleLocalityAwareDistribution considers data locality and network topology
func (s *Scheduler) scheduleLocalityAwareDistribution(workloads []ResourceRequest) ([]ResourceAllocation, error) {
	// For locality-aware scheduling, we prefer local cluster first, then nearby clusters
	var allocations []ResourceAllocation

	for _, workload := range workloads {
		// Try local cluster first
		allocation, err := s.scheduleLocalResourceAllocation(&workload)
		if err == nil {
			allocations = append(allocations, *allocation)
			continue
		}

		// Try remote clusters in order of network cost
		inventory, err := s.GetGlobalResourceInventory()
		if err != nil {
			continue
		}

		type clusterDistance struct {
			clusterID string
			distance  float64
			info      ClusterResourceInfo
		}

		var nearestClusters []clusterDistance
		for clusterID, cluster := range inventory {
			if clusterID == s.globalPool.parentCluster {
				continue
			}

			nearestClusters = append(nearestClusters, clusterDistance{
				clusterID: clusterID,
				distance:  cluster.NetworkCost,
				info:      cluster,
			})
		}

		sort.Slice(nearestClusters, func(i, j int) bool {
			return nearestClusters[i].distance < nearestClusters[j].distance
		})

		allocated := false
		for _, cluster := range nearestClusters {
			if !s.canClusterFulfillRequest(&cluster.info, &workload) {
				continue
			}

			allocation, err = s.allocateOnRemoteCluster(cluster.clusterID, &workload)
			if err == nil {
				allocations = append(allocations, *allocation)
				allocated = true
				break
			}
		}

		if !allocated {
			log.Printf("Failed to allocate workload %s on any cluster", workload.ID)
		}
	}

	return allocations, nil
}
