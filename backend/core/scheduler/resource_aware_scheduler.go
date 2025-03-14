package scheduler

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/migration"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/workload"
)

// PlacementPolicy defines how VMs should be placed on nodes
type PlacementPolicy string

// Placement policies
const (
	// PolicyBalanced balances load across all nodes
	PolicyBalanced PlacementPolicy = "balanced"

	// PolicyConsolidated consolidates VMs on fewer nodes
	PolicyConsolidated PlacementPolicy = "consolidated"

	// PolicyPerformance prioritizes performance over efficiency
	PolicyPerformance PlacementPolicy = "performance"

	// PolicyEfficiency prioritizes resource efficiency over performance
	PolicyEfficiency PlacementPolicy = "efficiency"

	// PolicyNetworkAware considers network topology in placement decisions
	PolicyNetworkAware PlacementPolicy = "network-aware"

	// PolicyCustom uses custom rules for placement
	PolicyCustom PlacementPolicy = "custom"
)

// ConstraintType defines the type of placement constraint
type ConstraintType string

// Constraint types
const (
	// ConstraintAffinityVMToVM defines affinity between VMs (place together)
	ConstraintAffinityVMToVM ConstraintType = "affinity-vm-to-vm"

	// ConstraintAntiAffinityVMToVM defines anti-affinity between VMs (place apart)
	ConstraintAntiAffinityVMToVM ConstraintType = "anti-affinity-vm-to-vm"

	// ConstraintAffinityVMToNode defines affinity between a VM and a node
	ConstraintAffinityVMToNode ConstraintType = "affinity-vm-to-node"

	// ConstraintAntiAffinityVMToNode defines anti-affinity between a VM and a node
	ConstraintAntiAffinityVMToNode ConstraintType = "anti-affinity-vm-to-node"

	// ConstraintResourceRequirement defines a minimum resource requirement
	ConstraintResourceRequirement ConstraintType = "resource-requirement"

	// ConstraintZoneRequirement defines a zone placement requirement
	ConstraintZoneRequirement ConstraintType = "zone-requirement"
)

// PlacementConstraint defines a constraint for VM placement
type PlacementConstraint struct {
	// Type is the type of constraint
	Type ConstraintType

	// ID is the unique identifier for this constraint
	ID string

	// EntityIDs are the IDs of entities involved in the constraint
	EntityIDs []string

	// ResourceType is the type of resource for resource constraints
	ResourceType string

	// MinimumAmount is the minimum amount of resource required
	MinimumAmount float64

	// MaximumAmount is the maximum amount of resource allowed
	MaximumAmount float64

	// Weight is the importance of this constraint (0.0-1.0)
	// Higher weight means more important
	Weight float64

	// Mandatory indicates if this constraint must be satisfied
	// If true, placement will fail if constraint cannot be satisfied
	Mandatory bool

	// Created is when this constraint was created
	Created time.Time

	// Expires is when this constraint expires
	// Zero time means never expires
	Expires time.Time
}

// PlacementRequest represents a request to place a VM
type PlacementRequest struct {
	// ID is the unique identifier for this request
	ID string

	// VMID is the ID of the VM to place
	VMID string

	// Policy is the placement policy to use
	Policy PlacementPolicy

	// Constraints are constraints on placement
	Constraints []PlacementConstraint

	// ResourceRequirements defines the resources required by the VM
	ResourceRequirements map[string]float64

	// PreferredNodes are nodes that are preferred for placement
	PreferredNodes []string

	// ExcludedNodes are nodes that should not be considered for placement
	ExcludedNodes []string

	// Priority is the priority of this request (higher is more important)
	Priority int

	// Created is when this request was created
	Created time.Time

	// ExpiresAt is when this request expires
	// Zero time means never expires
	ExpiresAt time.Time
}

// PlacementResult represents the result of a placement decision
type PlacementResult struct {
	// RequestID is the ID of the placement request
	RequestID string

	// VMID is the ID of the VM that was placed
	VMID string

	// SelectedNode is the node selected for placement
	SelectedNode string

	// AlternativeNodes are alternative nodes that could have been used
	AlternativeNodes []string

	// Score is the score of the selected node
	Score float64

	// Reasoning is a human-readable explanation of the decision
	Reasoning string

	// SatisfiedConstraints are constraints that were satisfied
	SatisfiedConstraints []PlacementConstraint

	// ViolatedConstraints are constraints that were violated
	ViolatedConstraints []PlacementConstraint

	// Created is when this result was created
	Created time.Time

	// Success indicates if placement was successful
	Success bool

	// Error is the error message if placement failed
	Error string
}

// ResourceAwareSchedulerConfig contains configuration for the scheduler
type ResourceAwareSchedulerConfig struct {
	// PlacementTimeout is the timeout for placement decisions
	PlacementTimeout time.Duration

	// DefaultPolicy is the default placement policy
	DefaultPolicy PlacementPolicy

	// MaxConcurrentPlacements is the maximum number of concurrent placements
	MaxConcurrentPlacements int

	// ConsiderWorkloadTypes indicates if workload types should be considered in placement
	ConsiderWorkloadTypes bool

	// ConsiderMigrationCosts indicates if migration costs should be considered
	ConsiderMigrationCosts bool

	// MinWorkloadProfileConfidence is the minimum confidence required for workload profiles
	MinWorkloadProfileConfidence float64

	// ReoptimizationInterval is how often to re-optimize placements
	ReoptimizationInterval time.Duration

	// AutomaticMigration enables automatic VM migration based on optimization
	AutomaticMigration bool

	// PreemptionEnabled enables preemption of low-priority VMs
	PreemptionEnabled bool

	// SchedulingWeights maps aspects to their weights in scoring
	SchedulingWeights map[string]float64

	// AntiAffinityDefaultWeight is the default weight for anti-affinity constraints
	AntiAffinityDefaultWeight float64

	// ResourceScarcityThreshold is the threshold for resource scarcity (0.0-1.0)
	// Resources with availability below this threshold are considered scarce
	ResourceScarcityThreshold float64
}

// DefaultResourceAwareSchedulerConfig returns a default configuration
func DefaultResourceAwareSchedulerConfig() ResourceAwareSchedulerConfig {
	return ResourceAwareSchedulerConfig{
		PlacementTimeout:             30 * time.Second,
		DefaultPolicy:                PolicyBalanced,
		MaxConcurrentPlacements:      10,
		ConsiderWorkloadTypes:        true,
		ConsiderMigrationCosts:       true,
		MinWorkloadProfileConfidence: 0.5,
		ReoptimizationInterval:       1 * time.Hour,
		AutomaticMigration:           false,
		PreemptionEnabled:            false,
		SchedulingWeights: map[string]float64{
			"resource_efficiency": 0.3,
			"load_balancing":      0.2,
			"migration_cost":      0.2,
			"constraints":         0.3,
		},
		AntiAffinityDefaultWeight: 0.7,
		ResourceScarcityThreshold: 0.2,
	}
}

// ResourceAwareScheduler implements a scheduler that considers workload types
// and migration costs when making placement decisions
type ResourceAwareScheduler struct {
	config ResourceAwareSchedulerConfig

	// workloadAnalyzer provides VM workload analysis
	workloadAnalyzer *workload.WorkloadAnalyzer

	// migrationCostEstimator provides migration cost estimation
	migrationCostEstimator *migration.MigrationCostEstimator

	// baseScheduler is the underlying scheduler for basic scheduling operations
	baseScheduler *Scheduler

	// nodes maps node IDs to node resources
	nodes     map[string]*NodeResources
	nodeMutex sync.RWMutex

	// vmPlacements maps VM IDs to their current placements
	vmPlacements     map[string]string
	vmPlacementMutex sync.RWMutex

	// constraints maps constraint IDs to constraints
	constraints     map[string]PlacementConstraint
	constraintMutex sync.RWMutex

	// pendingRequests is the queue of pending placement requests
	pendingRequests     []*PlacementRequest
	pendingRequestMutex sync.RWMutex

	// placementResults maps request IDs to placement results
	placementResults     map[string]*PlacementResult
	placementResultMutex sync.RWMutex

	// vmGroups maps group IDs to VM IDs
	vmGroups     map[string][]string
	vmGroupMutex sync.RWMutex

	// placementSemaphore limits concurrent placements
	placementSemaphore chan struct{}

	ctx    context.Context
	cancel context.CancelFunc
}

// NewResourceAwareScheduler creates a new resource-aware scheduler
func NewResourceAwareScheduler(
	config ResourceAwareSchedulerConfig,
	baseScheduler *Scheduler,
	workloadAnalyzer *workload.WorkloadAnalyzer,
	migrationCostEstimator *migration.MigrationCostEstimator,
) *ResourceAwareScheduler {
	ctx, cancel := context.WithCancel(context.Background())

	return &ResourceAwareScheduler{
		config:                 config,
		baseScheduler:          baseScheduler,
		workloadAnalyzer:       workloadAnalyzer,
		migrationCostEstimator: migrationCostEstimator,
		nodes:                  make(map[string]*NodeResources),
		vmPlacements:           make(map[string]string),
		constraints:            make(map[string]PlacementConstraint),
		pendingRequests:        make([]*PlacementRequest, 0),
		placementResults:       make(map[string]*PlacementResult),
		vmGroups:               make(map[string][]string),
		placementSemaphore:     make(chan struct{}, config.MaxConcurrentPlacements),
		ctx:                    ctx,
		cancel:                 cancel,
	}
}

// Start starts the resource-aware scheduler
func (s *ResourceAwareScheduler) Start() error {
	log.Println("Starting resource-aware scheduler")

	// Start the base scheduler
	err := s.baseScheduler.Start()
	if err != nil {
		return fmt.Errorf("failed to start base scheduler: %w", err)
	}

	// Start the placement loop
	go s.placementLoop()

	// Start the optimization loop if enabled
	if s.config.ReoptimizationInterval > 0 {
		go s.optimizationLoop()
	}

	return nil
}

// Stop stops the resource-aware scheduler
func (s *ResourceAwareScheduler) Stop() error {
	log.Println("Stopping resource-aware scheduler")

	s.cancel()

	// Stop the base scheduler
	err := s.baseScheduler.Stop()
	if err != nil {
		return fmt.Errorf("failed to stop base scheduler: %w", err)
	}

	return nil
}

// placementLoop processes pending placement requests
func (s *ResourceAwareScheduler) placementLoop() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.processPendingRequests()
		}
	}
}

// optimizationLoop periodically re-optimizes placements
func (s *ResourceAwareScheduler) optimizationLoop() {
	ticker := time.NewTicker(s.config.ReoptimizationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.optimizePlacements()
		}
	}
}

// UpdateNodeResources updates resources for a node
func (s *ResourceAwareScheduler) UpdateNodeResources(nodeID string, resources map[ResourceType]*Resource) error {
	// Update the base scheduler
	err := s.baseScheduler.UpdateNodeResources(nodeID, resources)
	if err != nil {
		return err
	}

	// Update our own node resources
	s.nodeMutex.Lock()
	defer s.nodeMutex.Unlock()

	nodeResources := &NodeResources{
		NodeID:     nodeID,
		Resources:  resources,
		LastUpdate: time.Now(),
		Available:  true,
	}
	s.nodes[nodeID] = nodeResources

	// Convert to the format used by the migration cost estimator
	migrationNodeInfo := &migration.NodeInfo{
		NodeID:      nodeID,
		Available:   true,
		Resources:   make(map[string]*migration.NodeResource),
		LastUpdated: time.Now(),
	}

	for resourceType, resource := range resources {
		migrationNodeInfo.Resources[string(resourceType)] = &migration.NodeResource{
			Type:     string(resourceType),
			Capacity: resource.Capacity,
			Used:     resource.Used,
		}
	}

	// Update the migration cost estimator
	s.migrationCostEstimator.UpdateNodeInfo(migrationNodeInfo)

	return nil
}

// RequestPlacement requests placement for a VM
func (s *ResourceAwareScheduler) RequestPlacement(vmID string, policy PlacementPolicy, constraints []PlacementConstraint, resources map[string]float64, priority int) (string, error) {
	// Generate a unique ID for the request
	requestID := fmt.Sprintf("req-%d", time.Now().UnixNano())

	// Create the request
	request := &PlacementRequest{
		ID:                   requestID,
		VMID:                 vmID,
		Policy:               policy,
		Constraints:          constraints,
		ResourceRequirements: resources,
		PreferredNodes:       []string{},
		ExcludedNodes:        []string{},
		Priority:             priority,
		Created:              time.Now(),
		ExpiresAt:            time.Now().Add(s.config.PlacementTimeout),
	}

	// Add to pending requests
	s.pendingRequestMutex.Lock()
	s.pendingRequests = append(s.pendingRequests, request)
	s.pendingRequestMutex.Unlock()

	log.Printf("Created placement request %s for VM %s", requestID, vmID)

	return requestID, nil
}

// GetPlacementResult gets the result of a placement request
func (s *ResourceAwareScheduler) GetPlacementResult(requestID string) (*PlacementResult, error) {
	s.placementResultMutex.RLock()
	defer s.placementResultMutex.RUnlock()

	result, exists := s.placementResults[requestID]
	if !exists {
		return nil, fmt.Errorf("no placement result for request %s", requestID)
	}

	return result, nil
}

// AddConstraint adds a placement constraint
func (s *ResourceAwareScheduler) AddConstraint(constraint PlacementConstraint) error {
	s.constraintMutex.Lock()
	defer s.constraintMutex.Unlock()

	// If no ID is provided, generate one
	if constraint.ID == "" {
		constraint.ID = fmt.Sprintf("const-%d", time.Now().UnixNano())
	}

	if constraint.Created.IsZero() {
		constraint.Created = time.Now()
	}

	s.constraints[constraint.ID] = constraint

	log.Printf("Added constraint %s of type %s", constraint.ID, constraint.Type)

	return nil
}

// RemoveConstraint removes a placement constraint
func (s *ResourceAwareScheduler) RemoveConstraint(constraintID string) error {
	s.constraintMutex.Lock()
	defer s.constraintMutex.Unlock()

	if _, exists := s.constraints[constraintID]; !exists {
		return fmt.Errorf("constraint %s not found", constraintID)
	}

	delete(s.constraints, constraintID)

	log.Printf("Removed constraint %s", constraintID)

	return nil
}

// UpdateVMPlacement updates the placement of a VM
func (s *ResourceAwareScheduler) UpdateVMPlacement(vmID string, nodeID string) error {
	s.vmPlacementMutex.Lock()
	defer s.vmPlacementMutex.Unlock()

	s.vmPlacements[vmID] = nodeID

	// Create a VM info object for the migration cost estimator
	vmInfo := &migration.VMInfo{
		VMID:          vmID,
		CurrentNodeID: nodeID,
		LastUpdated:   time.Now(),
	}

	// Update the migration cost estimator
	s.migrationCostEstimator.UpdateVMInfo(vmInfo)

	log.Printf("Updated placement of VM %s to node %s", vmID, nodeID)

	return nil
}

// processPendingRequests processes pending placement requests
func (s *ResourceAwareScheduler) processPendingRequests() {
	// Get pending requests
	s.pendingRequestMutex.Lock()
	if len(s.pendingRequests) == 0 {
		s.pendingRequestMutex.Unlock()
		return
	}

	// Sort requests by priority (higher priority first)
	sort.Slice(s.pendingRequests, func(i, j int) bool {
		return s.pendingRequests[i].Priority > s.pendingRequests[j].Priority
	})

	// Take the highest priority request
	request := s.pendingRequests[0]
	s.pendingRequests = s.pendingRequests[1:]
	s.pendingRequestMutex.Unlock()

	// Check if the request has expired
	if !request.ExpiresAt.IsZero() && time.Now().After(request.ExpiresAt) {
		s.createFailedPlacementResult(request, "request expired")
		return
	}

	// Process the request
	go s.processPlacementRequest(request)
}

// processPlacementRequest processes a single placement request
func (s *ResourceAwareScheduler) processPlacementRequest(request *PlacementRequest) {
	// Acquire a semaphore slot
	select {
	case s.placementSemaphore <- struct{}{}:
		// Got a slot
		defer func() {
			<-s.placementSemaphore
		}()
	case <-s.ctx.Done():
		return
	}

	// Create context with timeout
	ctx, cancel := context.WithTimeout(s.ctx, s.config.PlacementTimeout)
	defer cancel()

	// Determine the applicable constraints
	constraints := s.getConstraintsForVM(request.VMID)

	// Add constraints from the request
	constraints = append(constraints, request.Constraints...)

	// Get candidate nodes
	candidateNodes := s.getCandidateNodes(ctx, request)

	if len(candidateNodes) == 0 {
		s.createFailedPlacementResult(request, "no candidate nodes available")
		return
	}

	// Score candidate nodes
	scoredNodes, err := s.scoreNodes(ctx, request, candidateNodes, constraints)
	if err != nil {
		s.createFailedPlacementResult(request, fmt.Sprintf("failed to score nodes: %v", err))
		return
	}

	if len(scoredNodes) == 0 {
		s.createFailedPlacementResult(request, "no nodes satisfy mandatory constraints")
		return
	}

	// Select the best node
	bestNode := scoredNodes[0].nodeID
	bestScore := scoredNodes[0].score
	satisfiedConstraints := scoredNodes[0].satisfiedConstraints
	violatedConstraints := scoredNodes[0].violatedConstraints

	// Create alternative nodes list (nodes with score close to the best)
	alternativeNodes := make([]string, 0)
	for i := 1; i < len(scoredNodes) && i < 5; i++ {
		// Only include nodes with score within 10% of the best
		if scoredNodes[i].score >= bestScore*0.9 {
			alternativeNodes = append(alternativeNodes, scoredNodes[i].nodeID)
		}
	}

	// Create reasoning text
	reasoning := generateReasoningText(request, bestNode, bestScore, satisfiedConstraints)

	// Create successful placement result
	result := &PlacementResult{
		RequestID:            request.ID,
		VMID:                 request.VMID,
		SelectedNode:         bestNode,
		AlternativeNodes:     alternativeNodes,
		Score:                bestScore,
		Reasoning:            reasoning,
		SatisfiedConstraints: satisfiedConstraints,
		ViolatedConstraints:  violatedConstraints,
		Created:              time.Now(),
		Success:              true,
	}

	// Store the result
	s.placementResultMutex.Lock()
	s.placementResults[request.ID] = result
	s.placementResultMutex.Unlock()

	// Update VM placement
	s.UpdateVMPlacement(request.VMID, bestNode)

	log.Printf("Placed VM %s on node %s with score %.2f", request.VMID, bestNode, bestScore)
}

// getCandidateNodes gets candidate nodes for placement
func (s *ResourceAwareScheduler) getCandidateNodes(ctx context.Context, request *PlacementRequest) []string {
	s.nodeMutex.RLock()
	defer s.nodeMutex.RUnlock()

	// Start with all available nodes
	candidates := make([]string, 0, len(s.nodes))
	for nodeID, node := range s.nodes {
		if node.Available {
			candidates = append(candidates, nodeID)
		}
	}

	// Filter out excluded nodes
	if len(request.ExcludedNodes) > 0 {
		excludedMap := make(map[string]bool)
		for _, nodeID := range request.ExcludedNodes {
			excludedMap[nodeID] = true
		}

		filteredCandidates := make([]string, 0, len(candidates))
		for _, nodeID := range candidates {
			if !excludedMap[nodeID] {
				filteredCandidates = append(filteredCandidates, nodeID)
			}
		}
		candidates = filteredCandidates
	}

	// Filter nodes based on resource requirements
	if len(request.ResourceRequirements) > 0 {
		filteredCandidates := make([]string, 0, len(candidates))
		for _, nodeID := range candidates {
			node := s.nodes[nodeID]
			if s.nodeHasSufficientResources(node, request.ResourceRequirements) {
				filteredCandidates = append(filteredCandidates, nodeID)
			}
		}
		candidates = filteredCandidates
	}

	return candidates
}

// nodeHasSufficientResources checks if a node has sufficient resources
func (s *ResourceAwareScheduler) nodeHasSufficientResources(node *NodeResources, requirements map[string]float64) bool {
	for resourceName, requiredAmount := range requirements {
		resourceType := ResourceType(resourceName)
		resource, exists := node.Resources[resourceType]
		if !exists {
			return false
		}

		available := resource.Capacity - resource.Used
		if available < requiredAmount {
			return false
		}
	}

	return true
}

// getConstraintsForVM gets constraints applicable to a VM
func (s *ResourceAwareScheduler) getConstraintsForVM(vmID string) []PlacementConstraint {
	s.constraintMutex.RLock()
	defer s.constraintMutex.RUnlock()

	constraints := make([]PlacementConstraint, 0)
	now := time.Now()

	for _, constraint := range s.constraints {
		// Skip expired constraints
		if !constraint.Expires.IsZero() && now.After(constraint.Expires) {
			continue
		}

		// Check if this constraint applies to this VM
		applies := false
		for _, entityID := range constraint.EntityIDs {
			if entityID == vmID {
				applies = true
				break
			}
		}

		if applies {
			constraints = append(constraints, constraint)
		}
	}

	return constraints
}

// scoredNode represents a node with a placement score
type scoredNode struct {
	nodeID               string
	score                float64
	satisfiedConstraints []PlacementConstraint
	violatedConstraints  []PlacementConstraint
	componentScores      map[string]float64
}

// scoreNodes scores candidate nodes for placement
func (s *ResourceAwareScheduler) scoreNodes(ctx context.Context, request *PlacementRequest, candidateNodes []string, constraints []PlacementConstraint) ([]scoredNode, error) {
	// Get information about the VM
	vmProfile, err := s.getVMProfile(request.VMID)
	if err != nil {
		log.Printf("Warning: No workload profile for VM %s: %v", request.VMID, err)
		// Continue without workload profile
	}

	// Get current placement if available
	s.vmPlacementMutex.RLock()
	currentNodeID := s.vmPlacements[request.VMID]
	s.vmPlacementMutex.RUnlock()

	// Score each node
	scoredNodes := make([]scoredNode, 0, len(candidateNodes))

	for _, nodeID := range candidateNodes {
		// Score the node
		score, componentScores, satisfied, violated, err := s.scoreNode(ctx, request, nodeID, currentNodeID, vmProfile, constraints)
		if err != nil {
			log.Printf("Error scoring node %s: %v", nodeID, err)
			continue
		}

		// Check if any mandatory constraints were violated
		mandatoryViolated := false
		for _, constraint := range violated {
			if constraint.Mandatory {
				mandatoryViolated = true
				break
			}
		}

		// Skip nodes that violate mandatory constraints
		if mandatoryViolated {
			continue
		}

		scoredNodes = append(scoredNodes, scoredNode{
			nodeID:               nodeID,
			score:                score,
			satisfiedConstraints: satisfied,
			violatedConstraints:  violated,
			componentScores:      componentScores,
		})
	}

	// Sort by score (higher is better)
	sort.Slice(scoredNodes, func(i, j int) bool {
		return scoredNodes[i].score > scoredNodes[j].score
	})

	return scoredNodes, nil
}

// scoreNode scores a node for VM placement
func (s *ResourceAwareScheduler) scoreNode(
	ctx context.Context,
	request *PlacementRequest,
	nodeID string,
	currentNodeID string,
	vmProfile *workload.WorkloadProfile,
	constraints []PlacementConstraint,
) (float64, map[string]float64, []PlacementConstraint, []PlacementConstraint, error) {
	componentScores := make(map[string]float64)
	satisfiedConstraints := make([]PlacementConstraint, 0)
	violatedConstraints := make([]PlacementConstraint, 0)

	// Get node information
	s.nodeMutex.RLock()
	node, exists := s.nodes[nodeID]
	s.nodeMutex.RUnlock()

	if !exists {
		return 0, nil, nil, nil, fmt.Errorf("node %s not found", nodeID)
	}

	// Score based on constraint satisfaction
	constraintScore, satisfiedLocalConstraints, violatedLocalConstraints := s.scoreConstraints(nodeID, request.VMID, constraints)
	componentScores["constraints"] = constraintScore
	satisfiedConstraints = append(satisfiedConstraints, satisfiedLocalConstraints...)
	violatedConstraints = append(violatedConstraints, violatedLocalConstraints...)

	// Score based on resource efficiency
	resourceScore := s.scoreResourceEfficiency(node, request.ResourceRequirements)
	componentScores["resource_efficiency"] = resourceScore

	// Score based on load balancing
	loadBalancingScore := s.scoreLoadBalancing(node)
	componentScores["load_balancing"] = loadBalancingScore

	// Score based on migration cost if this is a migration
	migrationScore := 1.0 // Default to perfect score if not a migration
	if currentNodeID != "" && currentNodeID != nodeID && s.config.ConsiderMigrationCosts {
		// This is a migration, estimate the cost
		migrationCost, err := s.migrationCostEstimator.EstimateMigrationCost(ctx, request.VMID, nodeID)
		if err != nil {
			log.Printf("Warning: Failed to estimate migration cost for VM %s to node %s: %v", request.VMID, nodeID, err)
		} else {
			// Convert cost to score (lower cost = higher score)
			migrationScore = 1.0 - math.Min(migrationCost.TotalCostScore, 1.0)
			componentScores["migration_cost"] = migrationScore
		}
	}

	// Score based on workload compatibility if profile available
	workloadScore := 1.0 // Default to perfect score if no profile
	if vmProfile != nil && s.config.ConsiderWorkloadTypes {
		workloadScore = s.scoreWorkloadCompatibility(node, vmProfile)
		componentScores["workload_compatibility"] = workloadScore
	}

	// Apply policy-specific scoring adjustments
	s.applyPolicyScoring(request.Policy, componentScores)

	// Compute weighted score
	totalScore := 0.0
	for component, score := range componentScores {
		weight := s.config.SchedulingWeights[component]
		if weight > 0 {
			totalScore += score * weight
		}
	}

	// Special case for preferred nodes
	for _, preferredNodeID := range request.PreferredNodes {
		if preferredNodeID == nodeID {
			// Boost score for preferred nodes
			totalScore *= 1.1
			break
		}
	}

	return totalScore, componentScores, satisfiedConstraints, violatedConstraints, nil
}

// No need to redefine NodeResources as it's already defined in scheduler.go

// scoreConstraints scores constraint satisfaction for a node
func (s *ResourceAwareScheduler) scoreConstraints(nodeID string, vmID string, constraints []PlacementConstraint) (float64, []PlacementConstraint, []PlacementConstraint) {
	satisfiedConstraints := make([]PlacementConstraint, 0)
	violatedConstraints := make([]PlacementConstraint, 0)

	if len(constraints) == 0 {
		return 1.0, satisfiedConstraints, violatedConstraints
	}

	totalWeight := 0.0
	weightedSum := 0.0

	for _, constraint := range constraints {
		satisfied := false
		weight := constraint.Weight
		if weight <= 0 {
			// Use default weights if weight is not specified
			if constraint.Type == ConstraintAntiAffinityVMToVM || constraint.Type == ConstraintAntiAffinityVMToNode {
				weight = s.config.AntiAffinityDefaultWeight
			} else {
				weight = 0.5 // Default weight for other constraints
			}
		}

		// Check different constraint types
		switch constraint.Type {
		case ConstraintAffinityVMToNode:
			// VM should be placed on a specific node
			for _, entityID := range constraint.EntityIDs {
				if entityID == nodeID {
					satisfied = true
					break
				}
			}

		case ConstraintAntiAffinityVMToNode:
			// VM should NOT be placed on a specific node
			satisfied = true // Assume satisfied until proven otherwise
			for _, entityID := range constraint.EntityIDs {
				if entityID == nodeID {
					satisfied = false
					break
				}
			}

		case ConstraintAffinityVMToVM:
			// VM should be placed with other VMs
			s.vmPlacementMutex.RLock()
			satisfied = false // Assume not satisfied until proven otherwise
			for _, entityID := range constraint.EntityIDs {
				if entityPlacement, exists := s.vmPlacements[entityID]; exists && entityPlacement == nodeID {
					satisfied = true
					break
				}
			}
			s.vmPlacementMutex.RUnlock()

		case ConstraintAntiAffinityVMToVM:
			// VM should NOT be placed with other VMs
			s.vmPlacementMutex.RLock()
			satisfied = true // Assume satisfied until proven otherwise
			for _, entityID := range constraint.EntityIDs {
				if entityPlacement, exists := s.vmPlacements[entityID]; exists && entityPlacement == nodeID {
					satisfied = false
					break
				}
			}
			s.vmPlacementMutex.RUnlock()

		case ConstraintResourceRequirement:
			// VM requires a specific amount of a resource
			s.nodeMutex.RLock()
			node, exists := s.nodes[nodeID]
			if !exists {
				satisfied = false
			} else {
				resourceType := ResourceType(constraint.ResourceType)
				resource, resourceExists := node.Resources[resourceType]
				if !resourceExists {
					satisfied = false
				} else {
					available := resource.Capacity - resource.Used
					if constraint.MinimumAmount > 0 && available < constraint.MinimumAmount {
						satisfied = false
					} else if constraint.MaximumAmount > 0 && available > constraint.MaximumAmount {
						satisfied = false
					} else {
						satisfied = true
					}
				}
			}
			s.nodeMutex.RUnlock()

		default:
			// Unknown constraint type, ignore
			continue
		}

		if satisfied {
			satisfiedConstraints = append(satisfiedConstraints, constraint)
			weightedSum += weight
		} else {
			violatedConstraints = append(violatedConstraints, constraint)
		}

		totalWeight += weight
	}

	// Calculate score based on weighted satisfaction
	if totalWeight <= 0 {
		return 1.0, satisfiedConstraints, violatedConstraints
	}

	return weightedSum / totalWeight, satisfiedConstraints, violatedConstraints
}

// scoreResourceEfficiency scores resource efficiency for a node
func (s *ResourceAwareScheduler) scoreResourceEfficiency(node *NodeResources, requirements map[string]float64) float64 {
	if len(requirements) == 0 {
		return 1.0 // Perfect score if no resources required
	}

	// Calculate how efficiently the node's resources are used by this VM
	efficiencyScores := make([]float64, 0, len(requirements))

	for resourceName, requiredAmount := range requirements {
		resourceType := ResourceType(resourceName)
		resource, exists := node.Resources[resourceType]
		if !exists {
			return 0.0 // Node doesn't have this resource
		}

		// Calculate available amount
		available := resource.Capacity - resource.Used
		if available <= 0 {
			return 0.0 // No capacity available
		}

		// Calculate efficiency score for this resource
		// Higher score when the required amount fits well with available
		// 1.0 = perfect fit, 0.0 = very inefficient
		efficiency := 0.0

		// If the required amount is more than available, not a good fit
		if requiredAmount > available {
			efficiency = 0.0
		} else {
			// Calculate how well the required amount fits the available space
			// A perfect fit would use all the available space
			usage := requiredAmount / available

			// Ideally we want usage to be in the sweet spot around 70-90%
			// Too low = wasteful, too high = no room for growth
			if usage >= 0.7 && usage <= 0.9 {
				efficiency = 1.0
			} else if usage > 0.9 {
				// Nearly full - not ideal but still good
				efficiency = 0.9
			} else {
				// Less than optimal usage - score based on how close to optimal
				efficiency = usage / 0.7
			}
		}

		efficiencyScores = append(efficiencyScores, efficiency)
	}

	// Calculate average efficiency score
	var totalScore float64
	for _, score := range efficiencyScores {
		totalScore += score
	}
	return totalScore / float64(len(efficiencyScores))
}

// scoreLoadBalancing scores load balancing for a node
func (s *ResourceAwareScheduler) scoreLoadBalancing(node *NodeResources) float64 {
	// For balanced policy, we want to distribute load evenly
	// Calculate average utilization across all resources

	var totalUtilization float64
	resourceCount := 0

	for _, resource := range node.Resources {
		if resource.Capacity > 0 {
			utilization := resource.Used / resource.Capacity
			totalUtilization += utilization
			resourceCount++
		}
	}

	if resourceCount == 0 {
		return 1.0 // No resources to consider
	}

	averageUtilization := totalUtilization / float64(resourceCount)

	// For load balancing, lower utilization is better
	// Score is higher when utilization is lower
	// 1.0 = 0% utilized, 0.0 = 100% utilized
	return 1.0 - averageUtilization
}

// scoreWorkloadCompatibility scores workload compatibility for a node
func (s *ResourceAwareScheduler) scoreWorkloadCompatibility(node *NodeResources, profile *workload.WorkloadProfile) float64 {
	// No profile, return neutral score
	if profile == nil {
		return 0.5
	}

	// If the profile is not confident enough, return neutral score
	if profile.Confidence < s.config.MinWorkloadProfileConfidence {
		return 0.5
	}

	// Score based on the dominant workload type
	var score float64 = 0.5 // Default neutral score

	switch profile.DominantWorkloadType {
	case workload.WorkloadTypeCPUIntensive:
		// For CPU-intensive workloads, check CPU availability
		cpuResource, exists := node.Resources[ResourceCPU]
		if exists && cpuResource.Capacity > 0 {
			cpuUtilization := cpuResource.Used / cpuResource.Capacity
			// Lower utilization is better for CPU-intensive workloads
			score = 1.0 - cpuUtilization
		}

	case workload.WorkloadTypeMemoryIntensive:
		// For memory-intensive workloads, check memory availability
		memResource, exists := node.Resources[ResourceMemory]
		if exists && memResource.Capacity > 0 {
			memUtilization := memResource.Used / memResource.Capacity
			// Lower utilization is better for memory-intensive workloads
			score = 1.0 - memUtilization
		}

	case workload.WorkloadTypeIOIntensive:
		// For I/O-intensive workloads, check disk I/O availability
		diskResource, exists := node.Resources[ResourceDisk]
		if exists && diskResource.Capacity > 0 {
			diskUtilization := diskResource.Used / diskResource.Capacity
			// Lower utilization is better for I/O-intensive workloads
			score = 1.0 - diskUtilization
		}

	case workload.WorkloadTypeNetworkIntensive:
		// For network-intensive workloads, check network bandwidth availability
		networkResource, exists := node.Resources[ResourceNetwork]
		if exists && networkResource.Capacity > 0 {
			networkUtilization := networkResource.Used / networkResource.Capacity
			// Lower utilization is better for network-intensive workloads
			score = 1.0 - networkUtilization
		}
	}

	return score
}

// applyPolicyScoring applies policy-specific scoring adjustments
func (s *ResourceAwareScheduler) applyPolicyScoring(policy PlacementPolicy, scores map[string]float64) {
	switch policy {
	case PolicyBalanced:
		// Balanced policy prioritizes load balancing
		if score, exists := scores["load_balancing"]; exists {
			scores["load_balancing"] = score * 1.5
		}

	case PolicyConsolidated:
		// Consolidated policy prioritizes resource efficiency
		if score, exists := scores["resource_efficiency"]; exists {
			scores["resource_efficiency"] = score * 1.5
		}
		// And de-emphasizes load balancing
		if score, exists := scores["load_balancing"]; exists {
			scores["load_balancing"] = score * 0.5
		}

	case PolicyPerformance:
		// Performance policy prioritizes workload compatibility
		if score, exists := scores["workload_compatibility"]; exists {
			scores["workload_compatibility"] = score * 2.0
		}
		// And de-emphasizes migration cost
		if score, exists := scores["migration_cost"]; exists {
			scores["migration_cost"] = score * 0.5
		}

	case PolicyEfficiency:
		// Efficiency policy prioritizes resource efficiency
		if score, exists := scores["resource_efficiency"]; exists {
			scores["resource_efficiency"] = score * 2.0
		}

	case PolicyNetworkAware:
		// Network-aware policy prioritizes network proximity
		// Would need additional network topology information
		// For now, just prioritize workload compatibility for network workloads
		if score, exists := scores["workload_compatibility"]; exists {
			scores["workload_compatibility"] = score * 1.5
		}
	}
}

// getVMProfile gets the workload profile for a VM
func (s *ResourceAwareScheduler) getVMProfile(vmID string) (*workload.WorkloadProfile, error) {
	if s.workloadAnalyzer == nil {
		return nil, fmt.Errorf("workload analyzer not available")
	}

	// Get the workload profile from the analyzer
	return s.workloadAnalyzer.GetWorkloadProfile(vmID)
}

// createFailedPlacementResult creates a result for a failed placement
func (s *ResourceAwareScheduler) createFailedPlacementResult(request *PlacementRequest, reason string) {
	result := &PlacementResult{
		RequestID:        request.ID,
		VMID:             request.VMID,
		SelectedNode:     "",
		AlternativeNodes: []string{},
		Score:            0.0,
		Reasoning:        reason,
		Created:          time.Now(),
		Success:          false,
		Error:            reason,
	}

	s.placementResultMutex.Lock()
	s.placementResults[request.ID] = result
	s.placementResultMutex.Unlock()

	log.Printf("Failed to place VM %s: %s", request.VMID, reason)
}

// generateReasoningText generates reasoning text for a placement decision
func generateReasoningText(request *PlacementRequest, nodeID string, score float64, constraints []PlacementConstraint) string {
	var reasoning string

	// Base reasoning on the placement policy
	switch request.Policy {
	case PolicyBalanced:
		reasoning = fmt.Sprintf("Node %s was selected for balanced load distribution", nodeID)
	case PolicyConsolidated:
		reasoning = fmt.Sprintf("Node %s was selected to consolidate resources", nodeID)
	case PolicyPerformance:
		reasoning = fmt.Sprintf("Node %s was selected for optimal performance", nodeID)
	case PolicyEfficiency:
		reasoning = fmt.Sprintf("Node %s was selected for resource efficiency", nodeID)
	case PolicyNetworkAware:
		reasoning = fmt.Sprintf("Node %s was selected for network proximity", nodeID)
	default:
		reasoning = fmt.Sprintf("Node %s was selected with score %.2f", nodeID, score)
	}

	// Add constraint information if constraints were satisfied
	if len(constraints) > 0 {
		reasoning += " while satisfying placement constraints"
	}

	return reasoning
}

// optimizePlacements periodically re-optimizes VM placements
func (s *ResourceAwareScheduler) optimizePlacements() {
	log.Println("Starting placement optimization")

	// Skip if automatic migration is not enabled
	if !s.config.AutomaticMigration {
		log.Println("Automatic migration is disabled, skipping optimization")
		return
	}

	// Get current VM placements
	s.vmPlacementMutex.RLock()
	placements := make(map[string]string)
	for vmID, nodeID := range s.vmPlacements {
		placements[vmID] = nodeID
	}
	s.vmPlacementMutex.RUnlock()

	if len(placements) == 0 {
		log.Println("No VMs to optimize")
		return
	}

	// Score current placements and find candidates for migration
	type migrationCandidate struct {
		vmID           string
		currentNodeID  string
		targetNodeID   string
		currentScore   float64
		targetScore    float64
		improvementPct float64
	}

	// Find migration candidates
	candidates := make([]migrationCandidate, 0)

	ctx, cancel := context.WithTimeout(s.ctx, s.config.PlacementTimeout)
	defer cancel()

	// Get all available nodes
	s.nodeMutex.RLock()
	nodes := make([]string, 0, len(s.nodes))
	for nodeID, node := range s.nodes {
		if node.Available {
			nodes = append(nodes, nodeID)
		}
	}
	s.nodeMutex.RUnlock()

	// For each VM, check if a better placement exists
	for vmID, currentNodeID := range placements {
		// Create a placement request
		request := &PlacementRequest{
			ID:                   fmt.Sprintf("opt-%d", time.Now().UnixNano()),
			VMID:                 vmID,
			Policy:               s.config.DefaultPolicy,
			ResourceRequirements: make(map[string]float64),
			PreferredNodes:       []string{},
			ExcludedNodes:        []string{currentNodeID}, // Exclude current node to force finding a new one
			Priority:             0,
			Created:              time.Now(),
		}

		// Get constraints for this VM
		constraints := s.getConstraintsForVM(vmID)

		// Get VM profile
		vmProfile, err := s.getVMProfile(vmID)
		if err != nil {
			log.Printf("Warning: No workload profile for VM %s: %v", vmID, err)
			// Continue without workload profile
		}

		// Score current placement
		currentScore, _, _, _, err := s.scoreNode(ctx, request, currentNodeID, "", vmProfile, constraints)
		if err != nil {
			log.Printf("Error scoring current placement for VM %s: %v", vmID, err)
			continue
		}

		// Find best alternative placement
		bestNodeID := ""
		bestScore := 0.0

		for _, nodeID := range nodes {
			if nodeID == currentNodeID {
				continue // Skip current node
			}

			// Score this node
			score, _, _, _, err := s.scoreNode(ctx, request, nodeID, currentNodeID, vmProfile, constraints)
			if err != nil {
				log.Printf("Error scoring node %s for VM %s: %v", nodeID, vmID, err)
				continue
			}

			// Check if this is better than our current best
			if score > bestScore {
				bestScore = score
				bestNodeID = nodeID
			}
		}

		// If we found a better placement
		if bestNodeID != "" && bestScore > currentScore {
			improvementPct := (bestScore - currentScore) / currentScore * 100.0

			// Only consider significant improvements (>10%)
			if improvementPct >= 10.0 {
				candidates = append(candidates, migrationCandidate{
					vmID:           vmID,
					currentNodeID:  currentNodeID,
					targetNodeID:   bestNodeID,
					currentScore:   currentScore,
					targetScore:    bestScore,
					improvementPct: improvementPct,
				})
			}
		}
	}

	// Sort candidates by improvement percentage (highest first)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].improvementPct > candidates[j].improvementPct
	})

	// Take the top 3 candidates for migration
	maxMigrations := 3
	if len(candidates) > maxMigrations {
		candidates = candidates[:maxMigrations]
	}

	// Log the candidates
	for _, candidate := range candidates {
		log.Printf("VM %s can be migrated from %s to %s for %.1f%% improvement",
			candidate.vmID, candidate.currentNodeID, candidate.targetNodeID, candidate.improvementPct)

		// In a real implementation, trigger migration here
		// For now, just update the placement
		s.UpdateVMPlacement(candidate.vmID, candidate.targetNodeID)
	}

	log.Printf("Placement optimization complete, found %d migration candidates", len(candidates))
}
