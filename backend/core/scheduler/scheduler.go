package scheduler

import (
	"context"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

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
	NodeID    string
	Resources map[ResourceType]*Resource
	LastUpdate time.Time
	Metrics   map[string]float64
	Available bool
}

// ResourceConstraint represents a constraint on resources
type ResourceConstraint struct {
	Type      ResourceType
	MinAmount float64
	MaxAmount float64
}

// ResourceRequest represents a request for resources
type ResourceRequest struct {
	ID          string
	Constraints []ResourceConstraint
	Priority    int
	Timeout     time.Duration
	CreatedAt   time.Time
	ExpiresAt   time.Time
}

// ResourceAllocation represents allocated resources
type ResourceAllocation struct {
	RequestID  string
	NodeID     string
	Resources  map[ResourceType]float64
	AllocatedAt time.Time
	ExpiresAt   time.Time
	Released    bool
}

// TaskDistribution represents a task to be distributed across nodes
type TaskDistribution struct {
	TaskID       string
	ResourceRequest ResourceRequest
	TargetNodeCount int
	Allocations  []ResourceAllocation
	DistributedAt time.Time
	CompletedAt   time.Time
	Status        TaskStatus
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
}

// DefaultSchedulerConfig returns a default scheduler configuration
func DefaultSchedulerConfig() SchedulerConfig {
	return SchedulerConfig{
		AllocationInterval: 5 * time.Second,
		NodeTimeout:        2 * time.Minute,
		EnablePreemption:   true,
		MaxRequestTimeout:  1 * time.Hour,
		MinimumNodeCount:   1,
		OvercommitRatio: map[ResourceType]float64{
			ResourceCPU:     1.5,  // CPU can be overcommitted by 50%
			ResourceMemory:  1.0,  // Memory cannot be overcommitted
			ResourceDisk:    1.2,  // Disk can be overcommitted by 20%
			ResourceNetwork: 2.0,  // Network can be overcommitted by 100%
		},
		BalancingWeight: 0.5, // Equal weight to load balancing and resource efficiency
	}
}

// Scheduler handles resource scheduling across nodes
type Scheduler struct {
	config          SchedulerConfig
	nodes           map[string]*NodeResources
	requests        map[string]*ResourceRequest
	allocations     map[string]*ResourceAllocation
	tasks           map[string]*TaskDistribution
	nodeMutex       sync.RWMutex
	requestMutex    sync.RWMutex
	allocationMutex sync.RWMutex
	taskMutex       sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc
}

// NewScheduler creates a new scheduler
func NewScheduler(config SchedulerConfig) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &Scheduler{
		config:      config,
		nodes:       make(map[string]*NodeResources),
		requests:    make(map[string]*ResourceRequest),
		allocations: make(map[string]*ResourceAllocation),
		tasks:       make(map[string]*TaskDistribution),
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Start starts the scheduler
func (s *Scheduler) Start() error {
	log.Println("Starting scheduler")
	
	// Start the allocation loop
	go s.allocationLoop()
	
	// Start the cleanup loop
	go s.cleanupLoop()
	
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
	// Generate a unique ID for the request
	requestID := uuid.New().String()
	
	// Create the request
	request := &ResourceRequest{
		ID:          requestID,
		Constraints: constraints,
		Priority:    priority,
		Timeout:     timeout,
		CreatedAt:   time.Now(),
		ExpiresAt:   time.Now().Add(timeout),
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
	taskID := uuid.New().String()
	
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
		
		// Combine scores based on balancing weight
		combinedScore := s.config.BalancingWeight*loadScore + (1-s.config.BalancingWeight)*resourceScore
		
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
	
	return true
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
		resource, exists := node.Resources[constraint.Type]
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
