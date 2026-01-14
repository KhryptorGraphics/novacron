package scheduler

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/migration"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/workload"
)

// EnhancedResourceSchedulerConfig contains configuration for the enhanced scheduler
type EnhancedResourceSchedulerConfig struct {
	// Base resource-aware scheduler config
	BaseConfig ResourceAwareSchedulerConfig

	// WorkloadPatternDetectionEnabled enables pattern detection in workloads
	WorkloadPatternDetectionEnabled bool

	// ProactiveMigrationPlanningEnabled enables proactive migration planning
	ProactiveMigrationPlanningEnabled bool

	// NetworkTopologyAwarenessEnabled enables network topology awareness
	NetworkTopologyAwarenessEnabled bool

	// PredictiveResourceAllocationEnabled enables predictive resource allocation
	PredictiveResourceAllocationEnabled bool

	// WorkloadAnalysisInterval is how often to analyze workload patterns
	WorkloadAnalysisInterval time.Duration

	// MigrationPlanningInterval is how often to plan migrations
	MigrationPlanningInterval time.Duration

	// ResourcePredictionHorizon is how far ahead to predict resource needs
	ResourcePredictionHorizon time.Duration

	// MinWorkloadHistoryDuration is minimum history needed before pattern detection
	MinWorkloadHistoryDuration time.Duration

	// PatternConfidenceThreshold is the minimum confidence for pattern detection
	PatternConfidenceThreshold float64

	// OptimalMigrationWindowMargin is how much margin to add around optimal windows
	OptimalMigrationWindowMargin time.Duration

	// MaxPlannedMigrationsPerCycle is maximum migrations to plan per cycle
	MaxPlannedMigrationsPerCycle int
}

// DefaultEnhancedResourceSchedulerConfig returns a default configuration
func DefaultEnhancedResourceSchedulerConfig() EnhancedResourceSchedulerConfig {
	return EnhancedResourceSchedulerConfig{
		BaseConfig:                          DefaultResourceAwareSchedulerConfig(),
		WorkloadPatternDetectionEnabled:     true,
		ProactiveMigrationPlanningEnabled:   true,
		NetworkTopologyAwarenessEnabled:     false, // Not implemented yet
		PredictiveResourceAllocationEnabled: true,
		WorkloadAnalysisInterval:            1 * time.Hour,
		MigrationPlanningInterval:           4 * time.Hour,
		ResourcePredictionHorizon:           24 * time.Hour,
		MinWorkloadHistoryDuration:          24 * time.Hour,
		PatternConfidenceThreshold:          0.6,
		OptimalMigrationWindowMargin:        30 * time.Minute,
		MaxPlannedMigrationsPerCycle:        5,
	}
}

// EnhancedResourceScheduler extends the ResourceAwareScheduler with advanced
// workload pattern recognition and proactive migration planning
type EnhancedResourceScheduler struct {
	config EnhancedResourceSchedulerConfig

	// baseScheduler is the underlying resource-aware scheduler
	baseScheduler *ResourceAwareScheduler

	// workloadAnalyzer is the workload analyzer component
	workloadAnalyzer *workload.WorkloadAnalyzer

	// migrationPlanner is the migration planner component
	migrationPlanner *migration.MigrationPlanner

	// enhancedProfiles stores enhanced workload profiles for VMs
	enhancedProfiles map[string]*workload.EnhancedWorkloadProfile

	// migrationCostEstimator is the migration cost estimator component
	migrationCostEstimator *migration.MigrationCostEstimator

	ctx    context.Context
	cancel context.CancelFunc
}

// NewEnhancedResourceScheduler creates a new enhanced resource scheduler
func NewEnhancedResourceScheduler(
	config EnhancedResourceSchedulerConfig,
	baseScheduler *ResourceAwareScheduler,
	workloadAnalyzer *workload.WorkloadAnalyzer,
	migrationCostEstimator *migration.MigrationCostEstimator,
) (*EnhancedResourceScheduler, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Create migration planner
	migrationPlannerConfig := migration.DefaultMigrationPlannerConfig()
	migrationPlanner := migration.NewMigrationPlanner(
		migrationPlannerConfig,
		workloadAnalyzer,
		migrationCostEstimator,
	)

	return &EnhancedResourceScheduler{
		config:                 config,
		baseScheduler:          baseScheduler,
		workloadAnalyzer:       workloadAnalyzer,
		migrationPlanner:       migrationPlanner,
		enhancedProfiles:       make(map[string]*workload.EnhancedWorkloadProfile),
		migrationCostEstimator: migrationCostEstimator,
		ctx:                    ctx,
		cancel:                 cancel,
	}, nil
}

// Start starts the enhanced resource scheduler
func (s *EnhancedResourceScheduler) Start() error {
	log.Println("Starting enhanced resource scheduler")

	// Start the base scheduler
	err := s.baseScheduler.Start()
	if err != nil {
		return fmt.Errorf("failed to start base scheduler: %w", err)
	}

	// Start the migration planner
	err = s.migrationPlanner.Start()
	if err != nil {
		return fmt.Errorf("failed to start migration planner: %w", err)
	}

	// Start the workload analysis loop
	if s.config.WorkloadPatternDetectionEnabled {
		go s.workloadAnalysisLoop()
	}

	// Start the migration planning loop
	if s.config.ProactiveMigrationPlanningEnabled {
		go s.migrationPlanningLoop()
	}

	// Start the resource prediction loop
	if s.config.PredictiveResourceAllocationEnabled {
		go s.resourcePredictionLoop()
	}

	return nil
}

// Stop stops the enhanced resource scheduler
func (s *EnhancedResourceScheduler) Stop() error {
	log.Println("Stopping enhanced resource scheduler")

	s.cancel()

	// Stop the migration planner
	err := s.migrationPlanner.Stop()
	if err != nil {
		log.Printf("Error stopping migration planner: %v", err)
	}

	// Stop the base scheduler
	err = s.baseScheduler.Stop()
	if err != nil {
		return fmt.Errorf("failed to stop base scheduler: %w", err)
	}

	return nil
}

// workloadAnalysisLoop periodically analyzes workload patterns
func (s *EnhancedResourceScheduler) workloadAnalysisLoop() {
	ticker := time.NewTicker(s.config.WorkloadAnalysisInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.analyzeWorkloadPatterns()
		}
	}
}

// migrationPlanningLoop periodically plans migrations
func (s *EnhancedResourceScheduler) migrationPlanningLoop() {
	ticker := time.NewTicker(s.config.MigrationPlanningInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.planProactiveMigrations()
		}
	}
}

// resourcePredictionLoop periodically predicts resource needs
func (s *EnhancedResourceScheduler) resourcePredictionLoop() {
	ticker := time.NewTicker(s.config.WorkloadAnalysisInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.predictResourceNeeds()
		}
	}
}

// analyzeWorkloadPatterns analyzes workload patterns for all VMs
func (s *EnhancedResourceScheduler) analyzeWorkloadPatterns() {
	log.Println("Analyzing workload patterns")

	// Get all VMs from the base scheduler
	s.baseScheduler.vmPlacementMutex.RLock()
	vmIDs := make([]string, 0, len(s.baseScheduler.vmPlacements))
	for vmID := range s.baseScheduler.vmPlacements {
		vmIDs = append(vmIDs, vmID)
	}
	s.baseScheduler.vmPlacementMutex.RUnlock()

	for _, vmID := range vmIDs {
		// Get base workload profile
		baseProfile, err := s.workloadAnalyzer.GetWorkloadProfile(vmID)
		if err != nil {
			log.Printf("Warning: Failed to get workload profile for VM %s: %v", vmID, err)
			continue
		}

		// Check if we have enough history
		if baseProfile.HistoryDuration < s.config.MinWorkloadHistoryDuration {
			log.Printf("Not enough history for VM %s (%.1f hours < %.1f hours required)",
				vmID, baseProfile.HistoryDuration.Hours(), s.config.MinWorkloadHistoryDuration.Hours())
			continue
		}

		// Create or update enhanced profile
		enhancedProfile, exists := s.enhancedProfiles[vmID]
		if !exists {
			enhancedProfile = workload.NewEnhancedProfile(baseProfile)
		} else {
			// Update with latest base profile
			enhancedProfile.WorkloadProfile = baseProfile
		}

		// Detect patterns
		enhancedProfile.DetectPatterns()

		// Store the enhanced profile
		s.enhancedProfiles[vmID] = enhancedProfile

		// Update migration planner with enhanced profile
		s.migrationPlanner.UpdateEnhancedProfile(vmID, enhancedProfile)

		// Log pattern detection results
		if enhancedProfile.WorkloadStability > s.config.PatternConfidenceThreshold {
			log.Printf("Detected stable workload pattern for VM %s (stability: %.2f)",
				vmID, enhancedProfile.WorkloadStability)
		}
	}
}

// getOptimalMigrationWindow gets the optimal migration window for a VM
func (s *EnhancedResourceScheduler) getOptimalMigrationWindow(vmID string) (*migration.MigrationWindow, error) {
	// Get enhanced profile
	enhancedProfile, exists := s.enhancedProfiles[vmID]
	if !exists {
		return nil, fmt.Errorf("no enhanced profile for VM %s", vmID)
	}

	// Get optimal migration windows for the next 24 hours
	now := time.Now()
	end := now.Add(24 * time.Hour)
	windows := enhancedProfile.GetOptimalMigrationWindows(now, end)

	if len(windows) == 0 {
		return nil, fmt.Errorf("no optimal migration windows for VM %s", vmID)
	}

	// Convert to migration.MigrationWindow
	window := migration.MigrationWindow{
		StartTime: windows[0].StartTime.Add(-s.config.OptimalMigrationWindowMargin),
		EndTime:   windows[0].EndTime.Add(s.config.OptimalMigrationWindowMargin),
		Quality:   windows[0].Quality,
		Reason:    windows[0].Reason,
	}

	return &window, nil
}

// planProactiveMigrations plans migrations based on workload patterns and system condition
func (s *EnhancedResourceScheduler) planProactiveMigrations() {
	log.Println("Planning proactive migrations")

	// Get all VMs with enhanced profiles
	vmIDs := make([]string, 0, len(s.enhancedProfiles))
	for vmID, profile := range s.enhancedProfiles {
		// Only consider VMs with stable workload patterns
		if profile.WorkloadStability >= s.config.PatternConfidenceThreshold {
			vmIDs = append(vmIDs, vmID)
		}
	}

	if len(vmIDs) == 0 {
		log.Println("No VMs with stable workload patterns, skipping migration planning")
		return
	}

	// Get current VM placements
	s.baseScheduler.vmPlacementMutex.RLock()
	placements := make(map[string]string)
	for _, vmID := range vmIDs {
		if nodeID, exists := s.baseScheduler.vmPlacements[vmID]; exists {
			placements[vmID] = nodeID
		}
	}
	s.baseScheduler.vmPlacementMutex.RUnlock()

	// Get all candidate nodes
	s.baseScheduler.nodeMutex.RLock()
	nodeIDs := make([]string, 0, len(s.baseScheduler.nodes))
	for nodeID, node := range s.baseScheduler.nodes {
		if node.Available {
			nodeIDs = append(nodeIDs, nodeID)
		}
	}
	s.baseScheduler.nodeMutex.RUnlock()

	// For each VM, check if it should be migrated
	plannedMigrations := 0
	ctx := s.ctx

	for _, vmID := range vmIDs {
		if plannedMigrations >= s.config.MaxPlannedMigrationsPerCycle {
			break // Limit number of migrations planned per cycle
		}

		currentNodeID, exists := placements[vmID]
		if !exists {
			continue // Skip VMs with unknown placement
		}

		// Find the best destination node
		bestNodeID, err := s.migrationCostEstimator.FindBestMigrationTarget(ctx, vmID, nodeIDs)
		if err != nil {
			log.Printf("Warning: Failed to find best migration target for VM %s: %v", vmID, err)
			continue
		}

		// Skip if best node is the current node
		if bestNodeID == currentNodeID {
			continue
		}

		// Get optimal migration window
		window, err := s.getOptimalMigrationWindow(vmID)
		if err != nil {
			log.Printf("Warning: Failed to get optimal migration window for VM %s: %v", vmID, err)
			// Use default window (start now + 2 hours)
			window = &migration.MigrationWindow{
				StartTime: time.Now().Add(2 * time.Hour),
				EndTime:   time.Now().Add(4 * time.Hour),
				Quality:   0.5,
				Reason:    "Default migration window",
			}
		}

		// Plan the migration
		planID, err := s.migrationPlanner.PlanMigration(
			vmID,
			bestNodeID,
			window.StartTime,
			1, // Default priority
		)

		if err != nil {
			log.Printf("Warning: Failed to plan migration for VM %s: %v", vmID, err)
			continue
		}

		log.Printf("Planned migration %s: VM %s from %s to %s at %v",
			planID, vmID, currentNodeID, bestNodeID, window.StartTime)

		plannedMigrations++
	}

	log.Printf("Planned %d proactive migrations", plannedMigrations)
}

// predictResourceNeeds predicts future resource needs based on workload patterns
func (s *EnhancedResourceScheduler) predictResourceNeeds() {
	log.Println("Predicting future resource needs")

	// Time points to predict resource usage for
	now := time.Now()
	predictionPoints := []time.Time{
		now.Add(1 * time.Hour),
		now.Add(4 * time.Hour),
		now.Add(12 * time.Hour),
		now.Add(24 * time.Hour),
	}

	// Calculate predicted resource usage per node
	nodeResourcePredictions := make(map[string]map[string]map[time.Time]float64)

	// Get all VMs with enhanced profiles
	for vmID, profile := range s.enhancedProfiles {
		// Only consider VMs with stable workload patterns
		if profile.WorkloadStability < s.config.PatternConfidenceThreshold {
			continue
		}

		// Get current placement
		s.baseScheduler.vmPlacementMutex.RLock()
		nodeID, exists := s.baseScheduler.vmPlacements[vmID]
		s.baseScheduler.vmPlacementMutex.RUnlock()

		if !exists {
			continue
		}

		// Initialize node predictions if needed
		if _, exists := nodeResourcePredictions[nodeID]; !exists {
			nodeResourcePredictions[nodeID] = make(map[string]map[time.Time]float64)
		}

		// Predict usage for each resource type
		for resourceType := range profile.ResourceUsagePatterns {
			// Initialize resource predictions if needed
			if _, exists := nodeResourcePredictions[nodeID][resourceType]; !exists {
				nodeResourcePredictions[nodeID][resourceType] = make(map[time.Time]float64)
			}

			// Predict usage at each time point
			for _, t := range predictionPoints {
				usage := profile.PredictResourceUsage(resourceType, t)
				if usage >= 0 {
					// Add to node's predicted usage
					nodeResourcePredictions[nodeID][resourceType][t] += usage
				}
			}
		}
	}

	// Log predictions
	for nodeID, resources := range nodeResourcePredictions {
		for resourceType, predictions := range resources {
			log.Printf("Resource predictions for node %s, %s:", nodeID, resourceType)
			for t, usage := range predictions {
				log.Printf("  %v: %.1f%%", t.Format("15:04:05"), usage)
			}
		}
	}

	// Use predictions for proactive scheduling decisions
	// In a real implementation, this would trigger actions like:
	// - Reserving capacity for predicted peaks
	// - Planning migrations away from nodes predicted to become overloaded
	// - Adjusting placement scores based on future resource availability
}

// RequestPlacement requests VM placement with enhanced awareness
func (s *EnhancedResourceScheduler) RequestPlacement(
	vmID string,
	policy PlacementPolicy,
	constraints []PlacementConstraint,
	resources map[string]float64,
	priority int,
) (string, error) {
	// Check if we have an enhanced profile for this VM
	enhancedProfile, exists := s.enhancedProfiles[vmID]

	if exists && s.config.PredictiveResourceAllocationEnabled {
		// Use predicted resource needs instead of current
		predictedResources := make(map[string]float64)

		// Look 1 hour ahead for resource predictions
		predictionTime := time.Now().Add(1 * time.Hour)

		// For each resource type, add predicted usage
		for resourceType := range resources {
			predicted := enhancedProfile.PredictResourceUsage(resourceType, predictionTime)
			if predicted >= 0 {
				// Use predicted value, with a small safety margin
				predictedResources[resourceType] = predicted * 1.1 // 10% safety margin
			} else {
				// Use provided value as fallback
				predictedResources[resourceType] = resources[resourceType]
			}
		}

		// Add workload-specific constraints based on pattern
		if s.config.WorkloadPatternDetectionEnabled &&
			enhancedProfile.WorkloadStability >= s.config.PatternConfidenceThreshold {

			// Add appropriate constraints based on workload patterns
			for _, pattern := range enhancedProfile.RecognizedPatterns {
				switch pattern.PatternType {
				case workload.PatternTypeDiurnal:
					// For diurnal patterns, add resource requirements that vary by time of day
					resourceType := pattern.ResourceType
					if resourceType != "" {
						constraints = append(constraints, PlacementConstraint{
							Type:          ConstraintResourceRequirement,
							ResourceType:  resourceType,
							MinimumAmount: predictedResources[resourceType],
							Weight:        0.8,
							Mandatory:     false,
							Created:       time.Now(),
						})
					}
				case workload.PatternTypeBursty:
					// For bursty workloads, ensure ample resources
					resourceType := pattern.ResourceType
					if resourceType != "" && pattern.Parameters["baseline"] > 0 {
						// Use baseline + peak amplitude as requirement
						baseline := pattern.Parameters["baseline"]
						amplitude := pattern.Parameters["amplitude"]
						constraints = append(constraints, PlacementConstraint{
							Type:          ConstraintResourceRequirement,
							ResourceType:  resourceType,
							MinimumAmount: baseline + amplitude,
							Weight:        0.7,
							Mandatory:     false,
							Created:       time.Now(),
						})
					}
				}
			}
		}

		// Use predicted resources for placement
		return s.baseScheduler.RequestPlacement(vmID, policy, constraints, predictedResources, priority)
	}

	// Fall back to base scheduler logic
	return s.baseScheduler.RequestPlacement(vmID, policy, constraints, resources, priority)
}
