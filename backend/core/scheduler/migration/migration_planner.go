package migration

import (
	"context"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/workload"
)

// MigrationStatus represents the status of a migration plan
type MigrationStatus string

// Migration statuses
const (
	MigrationStatusPlanned     MigrationStatus = "planned"
	MigrationStatusInProgress  MigrationStatus = "in_progress"
	MigrationStatusCompleted   MigrationStatus = "completed"
	MigrationStatusFailed      MigrationStatus = "failed"
	MigrationStatusCancelled   MigrationStatus = "cancelled"
	MigrationStatusRescheduled MigrationStatus = "rescheduled"
)

// MigrationWindow represents an optimal time window for migration
type MigrationWindow struct {
	// StartTime is when the window starts
	StartTime time.Time

	// EndTime is when the window ends
	EndTime time.Time

	// Quality indicates the suitability of this window (0-1)
	Quality float64

	// Reason explains why this window was selected
	Reason string
}

// MigrationPlan represents a planned migration
type MigrationPlan struct {
	// ID is the unique identifier for this plan
	ID string

	// VMID is the VM to migrate
	VMID string

	// SourceNodeID is the current node
	SourceNodeID string

	// DestNodeID is the target node
	DestNodeID string

	// ScheduledWindow is when the migration should occur
	ScheduledWindow MigrationWindow

	// EstimatedCost is the projected migration cost
	EstimatedCost *MigrationCost

	// Status tracks the execution status
	Status MigrationStatus

	// CreatedAt is when this plan was created
	CreatedAt time.Time

	// UpdatedAt is when this plan was last updated
	UpdatedAt time.Time

	// CompletedAt is when the migration was completed
	CompletedAt time.Time

	// ErrorMessage contains any error message if the migration failed
	ErrorMessage string

	// Priority indicates the priority of this migration (higher is more important)
	Priority int

	// ResourceReservations tracks any reserved resources for this migration
	ResourceReservations map[string]float64
}

// MigrationGroup represents a group of related migrations
type MigrationGroup struct {
	// ID is the unique identifier for this group
	ID string

	// MigrationIDs are the IDs of migrations in this group
	MigrationIDs []string

	// ScheduledWindow is when the migrations should occur
	ScheduledWindow MigrationWindow

	// Status tracks the execution status
	Status MigrationStatus

	// CreatedAt is when this group was created
	CreatedAt time.Time

	// Priority indicates the priority of this group (higher is more important)
	Priority int

	// Description explains why these migrations are grouped
	Description string
}

// ResourceReservation represents a reservation of resources for a future use
type ResourceReservation struct {
	// ID is the unique identifier for this reservation
	ID string

	// NodeID is the node where resources are reserved
	NodeID string

	// Resources maps resource types to amounts
	Resources map[string]float64

	// ReservedFor indicates what the reservation is for
	ReservedFor string

	// StartTime is when the reservation starts
	StartTime time.Time

	// EndTime is when the reservation ends
	EndTime time.Time

	// Priority indicates the priority of this reservation
	Priority int

	// IsActive indicates if this reservation is currently active
	IsActive bool
}

// MigrationPlannerConfig contains configuration for the migration planner
type MigrationPlannerConfig struct {
	// DefaultWindowDuration is the default duration for migration windows
	DefaultWindowDuration time.Duration

	// LookAheadPeriod is how far ahead to plan migrations
	LookAheadPeriod time.Duration

	// MinMigrationInterval is the minimum time between migrations for a node
	MinMigrationInterval time.Duration

	// MaxConcurrentMigrations is the maximum number of concurrent migrations
	MaxConcurrentMigrations int

	// ResourceReminderThreshold is the threshold for resource reminder alerts
	ResourceReminderThreshold float64

	// PlanningInterval is how often to run the planning process
	PlanningInterval time.Duration

	// MigrationGroupingEnabled enables grouping of related migrations
	MigrationGroupingEnabled bool

	// MaxGroupSize is the maximum number of migrations in a group
	MaxGroupSize int

	// PreferredMigrationTimes are preferred times of day for migrations
	PreferredMigrationTimes []time.Duration
}

// DefaultMigrationPlannerConfig returns a default configuration
func DefaultMigrationPlannerConfig() MigrationPlannerConfig {
	return MigrationPlannerConfig{
		DefaultWindowDuration:     2 * time.Hour,
		LookAheadPeriod:           7 * 24 * time.Hour, // 1 week
		MinMigrationInterval:      12 * time.Hour,
		MaxConcurrentMigrations:   3,
		ResourceReminderThreshold: 0.8, // 80%
		PlanningInterval:          1 * time.Hour,
		MigrationGroupingEnabled:  true,
		MaxGroupSize:              5,
		PreferredMigrationTimes: []time.Duration{
			// Prefer migrations during off-hours
			2 * time.Hour,  // 2 AM
			3 * time.Hour,  // 3 AM
			4 * time.Hour,  // 4 AM
			22 * time.Hour, // 10 PM
			23 * time.Hour, // 11 PM
		},
	}
}

// MigrationPlanner plans and schedules VM migrations
type MigrationPlanner struct {
	config MigrationPlannerConfig

	// plans stores migration plans
	plans     map[string]*MigrationPlan
	planMutex sync.RWMutex

	// groups stores migration groups
	groups     map[string]*MigrationGroup
	groupMutex sync.RWMutex

	// reservations stores resource reservations
	reservations     map[string]*ResourceReservation
	reservationMutex sync.RWMutex

	// workloadAnalyzer is used to get VM workload profiles
	workloadAnalyzer *workload.WorkloadAnalyzer

	// enhancedProfiles stores enhanced workload profiles
	enhancedProfiles map[string]*workload.EnhancedWorkloadProfile
	profileMutex     sync.RWMutex

	// migrationCostEstimator is used to estimate migration costs
	migrationCostEstimator *MigrationCostEstimator

	// nodeMigrationHistory tracks recent migrations per node
	nodeMigrationHistory      map[string][]time.Time
	nodeMigrationHistoryMutex sync.RWMutex

	ctx    context.Context
	cancel context.CancelFunc
}

// NewMigrationPlanner creates a new migration planner
func NewMigrationPlanner(
	config MigrationPlannerConfig,
	analyzer *workload.WorkloadAnalyzer,
	costEstimator *MigrationCostEstimator,
) *MigrationPlanner {
	ctx, cancel := context.WithCancel(context.Background())

	return &MigrationPlanner{
		config:                 config,
		plans:                  make(map[string]*MigrationPlan),
		groups:                 make(map[string]*MigrationGroup),
		reservations:           make(map[string]*ResourceReservation),
		workloadAnalyzer:       analyzer,
		enhancedProfiles:       make(map[string]*workload.EnhancedWorkloadProfile),
		migrationCostEstimator: costEstimator,
		nodeMigrationHistory:   make(map[string][]time.Time),
		ctx:                    ctx,
		cancel:                 cancel,
	}
}

// Start starts the migration planner
func (p *MigrationPlanner) Start() error {
	log.Println("Starting migration planner")

	// Start the planning loop
	go p.planningLoop()

	// Start the reservation cleanup loop
	go p.reservationCleanupLoop()

	return nil
}

// Stop stops the migration planner
func (p *MigrationPlanner) Stop() error {
	log.Println("Stopping migration planner")

	p.cancel()

	return nil
}

// planningLoop periodically plans migrations
func (p *MigrationPlanner) planningLoop() {
	ticker := time.NewTicker(p.config.PlanningInterval)
	defer ticker.Stop()

	for {
		select {
		case <-p.ctx.Done():
			return
		case <-ticker.C:
			p.planMigrations()
		}
	}
}

// reservationCleanupLoop periodically cleans up expired reservations
func (p *MigrationPlanner) reservationCleanupLoop() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-p.ctx.Done():
			return
		case <-ticker.C:
			p.cleanupExpiredReservations()
		}
	}
}

// UpdateEnhancedProfile updates an enhanced workload profile
func (p *MigrationPlanner) UpdateEnhancedProfile(vmID string, profile *workload.EnhancedWorkloadProfile) {
	p.profileMutex.Lock()
	defer p.profileMutex.Unlock()

	p.enhancedProfiles[vmID] = profile
}

// GetEnhancedProfile gets an enhanced workload profile
func (p *MigrationPlanner) GetEnhancedProfile(vmID string) (*workload.EnhancedWorkloadProfile, error) {
	p.profileMutex.RLock()
	profile, exists := p.enhancedProfiles[vmID]
	p.profileMutex.RUnlock()

	if exists {
		return profile, nil
	}

	// Try to get the base profile and enhance it
	baseProfile, err := p.workloadAnalyzer.GetWorkloadProfile(vmID)
	if err != nil {
		return nil, err
	}

	// Create an enhanced profile from the base profile
	enhancedProfile := workload.NewEnhancedProfile(vmID)

	// Initialize the enhanced profile with base profile data if available
	if baseProfile != nil {
		// Create a profile adapter from the base profile
		adapter := &workload.WorkloadProfileAdapter{
			VMID: vmID,
			HistoryDuration: time.Hour * 24, // Default 24 hours
			LastUpdated: time.Now(),
			ResourceUsage: make(map[string]workload.ResourceUsageStats),
		}
		enhancedProfile.SetWorkloadProfile(adapter)
	}

	// Store the enhanced profile
	p.profileMutex.Lock()
	p.enhancedProfiles[vmID] = enhancedProfile
	p.profileMutex.Unlock()

	return enhancedProfile, nil
}

// PlanMigration plans a VM migration
func (p *MigrationPlanner) PlanMigration(
	vmID string,
	destNodeID string,
	scheduledStartTime time.Time,
	priority int,
) (string, error) {
	// Generate a unique ID for the plan
	planID := fmt.Sprintf("mig-%d", time.Now().UnixNano())

	// Get the VM's current node
	// In a real implementation, this would come from the VM manager
	sourceNodeID := "unknown" // Placeholder

	// Estimate migration cost
	cost, err := p.migrationCostEstimator.EstimateMigrationCost(p.ctx, vmID, destNodeID)
	if err != nil {
		log.Printf("Warning: Failed to estimate migration cost: %v", err)
		// Continue without cost information
	}

	// Find the optimal window if not specified
	window := MigrationWindow{
		StartTime: scheduledStartTime,
		EndTime:   scheduledStartTime.Add(p.config.DefaultWindowDuration),
		Quality:   0.5, // Default quality
		Reason:    "User-scheduled migration",
	}

	if scheduledStartTime.IsZero() {
		// Find an optimal window
		enhancedProfile, err := p.GetEnhancedProfile(vmID)
		if err == nil && enhancedProfile.IsStableWorkload() {
			// For stable workloads, find a time during preferred hours
			now := time.Now()
			window = p.findPreferredMigrationWindow(now, now.Add(p.config.LookAheadPeriod))
		} else {
			// If no profile or unstable workload, use preferred hours
			now := time.Now()
			window = p.findPreferredMigrationWindow(now, now.Add(p.config.LookAheadPeriod))
		}
	}

	// Create the migration plan
	plan := &MigrationPlan{
		ID:                   planID,
		VMID:                 vmID,
		SourceNodeID:         sourceNodeID,
		DestNodeID:           destNodeID,
		ScheduledWindow:      window,
		EstimatedCost:        cost,
		Status:               MigrationStatusPlanned,
		CreatedAt:            time.Now(),
		UpdatedAt:            time.Now(),
		Priority:             priority,
		ResourceReservations: make(map[string]float64),
	}

	// Store the plan
	p.planMutex.Lock()
	p.plans[planID] = plan
	p.planMutex.Unlock()

	// Reserve resources for the migration
	if cost != nil {
		p.reserveResourcesForMigration(plan)
	}

	// Add to node migration history
	p.nodeMigrationHistoryMutex.Lock()
	if _, exists := p.nodeMigrationHistory[sourceNodeID]; !exists {
		p.nodeMigrationHistory[sourceNodeID] = make([]time.Time, 0)
	}
	if _, exists := p.nodeMigrationHistory[destNodeID]; !exists {
		p.nodeMigrationHistory[destNodeID] = make([]time.Time, 0)
	}
	p.nodeMigrationHistoryMutex.Unlock()

	log.Printf("Created migration plan %s for VM %s to node %s", planID, vmID, destNodeID)

	return planID, nil
}

// CancelMigrationPlan cancels a migration plan
func (p *MigrationPlanner) CancelMigrationPlan(planID string) error {
	p.planMutex.Lock()
	defer p.planMutex.Unlock()

	plan, exists := p.plans[planID]
	if !exists {
		return fmt.Errorf("migration plan %s not found", planID)
	}

	if plan.Status != MigrationStatusPlanned {
		return fmt.Errorf("cannot cancel migration with status %s", plan.Status)
	}

	plan.Status = MigrationStatusCancelled
	plan.UpdatedAt = time.Now()

	// Release any resource reservations
	p.releaseResourcesForMigration(plan)

	log.Printf("Cancelled migration plan %s", planID)

	return nil
}

// GetMigrationPlan gets a migration plan
func (p *MigrationPlanner) GetMigrationPlan(planID string) (*MigrationPlan, error) {
	p.planMutex.RLock()
	defer p.planMutex.RUnlock()

	plan, exists := p.plans[planID]
	if !exists {
		return nil, fmt.Errorf("migration plan %s not found", planID)
	}

	return plan, nil
}

// GetMigrationPlansForVM gets all migration plans for a VM
func (p *MigrationPlanner) GetMigrationPlansForVM(vmID string) ([]*MigrationPlan, error) {
	p.planMutex.RLock()
	defer p.planMutex.RUnlock()

	plans := make([]*MigrationPlan, 0)
	for _, plan := range p.plans {
		if plan.VMID == vmID {
			plans = append(plans, plan)
		}
	}

	return plans, nil
}

// GetPendingMigrationPlans gets all pending migration plans
func (p *MigrationPlanner) GetPendingMigrationPlans() []*MigrationPlan {
	p.planMutex.RLock()
	defer p.planMutex.RUnlock()

	plans := make([]*MigrationPlan, 0)
	for _, plan := range p.plans {
		if plan.Status == MigrationStatusPlanned {
			plans = append(plans, plan)
		}
	}

	return plans
}

// UpdateMigrationStatus updates the status of a migration plan
func (p *MigrationPlanner) UpdateMigrationStatus(planID string, status MigrationStatus, errorMessage string) error {
	p.planMutex.Lock()
	defer p.planMutex.Unlock()

	plan, exists := p.plans[planID]
	if !exists {
		return fmt.Errorf("migration plan %s not found", planID)
	}

	plan.Status = status
	plan.UpdatedAt = time.Now()
	plan.ErrorMessage = errorMessage

	if status == MigrationStatusCompleted {
		plan.CompletedAt = time.Now()

		// Update node migration history
		p.nodeMigrationHistoryMutex.Lock()
		p.nodeMigrationHistory[plan.SourceNodeID] = append(p.nodeMigrationHistory[plan.SourceNodeID], plan.CompletedAt)
		p.nodeMigrationHistory[plan.DestNodeID] = append(p.nodeMigrationHistory[plan.DestNodeID], plan.CompletedAt)
		p.nodeMigrationHistoryMutex.Unlock()
	}

	log.Printf("Updated migration plan %s status to %s", planID, status)

	return nil
}

// planMigrations plans migrations based on workload patterns and system conditions
func (p *MigrationPlanner) planMigrations() {
	log.Println("Planning migrations...")

	// In a real implementation, this would check for VMs that should be migrated
	// based on resource usage, node load, etc.

	// Group related migrations if enabled
	if p.config.MigrationGroupingEnabled {
		p.groupRelatedMigrations()
	}

	// Check if any planned migrations should be executed now
	p.checkReadyMigrations()
}

// findPreferredMigrationWindow finds a preferred time for migration
func (p *MigrationPlanner) findPreferredMigrationWindow(start, end time.Time) MigrationWindow {
	if len(p.config.PreferredMigrationTimes) == 0 {
		// No preferred times, use start time
		return MigrationWindow{
			StartTime: start,
			EndTime:   start.Add(p.config.DefaultWindowDuration),
			Quality:   0.5,
			Reason:    "No preferred migration times configured",
		}
	}

	// Find the next preferred time after start
	now := time.Now()
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())

	// Sort the preferred times
	sortedTimes := make([]time.Duration, len(p.config.PreferredMigrationTimes))
	copy(sortedTimes, p.config.PreferredMigrationTimes)
	sort.Slice(sortedTimes, func(i, j int) bool {
		return sortedTimes[i] < sortedTimes[j]
	})

	// Find the next preferred time
	var windowStart time.Time
	for day := 0; day < 7; day++ {
		candidate := today.AddDate(0, 0, day)
		for _, preferredTime := range sortedTimes {
			hours := int(preferredTime.Hours())
			mins := int(preferredTime.Minutes()) % 60
			candidate = time.Date(
				candidate.Year(), candidate.Month(), candidate.Day(),
				hours, mins, 0, 0, candidate.Location(),
			)

			if candidate.After(start) && candidate.Before(end) {
				// Check if this time is available for the nodes
				// (not too close to other migrations)
				if p.isWindowAvailable(candidate) {
					windowStart = candidate
					break
				}
			}
		}

		if !windowStart.IsZero() {
			break
		}
	}

	if windowStart.IsZero() {
		// No preferred time found, use start time
		return MigrationWindow{
			StartTime: start,
			EndTime:   start.Add(p.config.DefaultWindowDuration),
			Quality:   0.5,
			Reason:    "No available preferred migration times",
		}
	}

	return MigrationWindow{
		StartTime: windowStart,
		EndTime:   windowStart.Add(p.config.DefaultWindowDuration),
		Quality:   0.8,
		Reason:    "Preferred migration time",
	}
}

// isWindowAvailable checks if a time window is available for migration
func (p *MigrationPlanner) isWindowAvailable(t time.Time) bool {
	// Check if there are any migrations too close to this time
	p.nodeMigrationHistoryMutex.RLock()
	defer p.nodeMigrationHistoryMutex.RUnlock()

	// Check all nodes
	for _, migrations := range p.nodeMigrationHistory {
		for _, migTime := range migrations {
			timeDiff := t.Sub(migTime).Hours()
			if timeDiff < 0 {
				timeDiff = -timeDiff
			}
			if timeDiff < p.config.MinMigrationInterval.Hours() {
				return false
			}
		}
	}

	// Also check planned migrations
	p.planMutex.RLock()
	defer p.planMutex.RUnlock()

	for _, plan := range p.plans {
		if plan.Status == MigrationStatusPlanned {
			timeDiff := t.Sub(plan.ScheduledWindow.StartTime).Hours()
			if timeDiff < 0 {
				timeDiff = -timeDiff
			}
			if timeDiff < p.config.MinMigrationInterval.Hours() {
				return false
			}
		}
	}

	return true
}

// reserveResourcesForMigration reserves resources for a migration
func (p *MigrationPlanner) reserveResourcesForMigration(plan *MigrationPlan) {
	// Generate a unique ID for the reservation
	reservationID := fmt.Sprintf("res-%d", time.Now().UnixNano())

	// Create resource requirements based on the migration cost
	resources := make(map[string]float64)

	// In a real implementation, this would calculate actual resource needs
	// For now, we'll use placeholder values
	resources["memory"] = 1024 // 1 GB of memory
	resources["cpu"] = 2       // 2 CPU cores
	resources["network"] = 100 // 100 Mbps

	// Create the reservation
	reservation := &ResourceReservation{
		ID:          reservationID,
		NodeID:      plan.DestNodeID,
		Resources:   resources,
		ReservedFor: fmt.Sprintf("migration:%s", plan.ID),
		StartTime:   plan.ScheduledWindow.StartTime,
		EndTime:     plan.ScheduledWindow.EndTime,
		Priority:    plan.Priority,
		IsActive:    false,
	}

	// Store the reservation
	p.reservationMutex.Lock()
	p.reservations[reservationID] = reservation
	p.reservationMutex.Unlock()

	// Update the plan with the reservation
	for resourceType, amount := range resources {
		plan.ResourceReservations[resourceType] = amount
	}

	log.Printf("Reserved resources for migration %s", plan.ID)
}

// releaseResourcesForMigration releases resources reserved for a migration
func (p *MigrationPlanner) releaseResourcesForMigration(plan *MigrationPlan) {
	p.reservationMutex.Lock()
	defer p.reservationMutex.Unlock()

	// Find and remove reservations for this migration
	for id, reservation := range p.reservations {
		if reservation.ReservedFor == fmt.Sprintf("migration:%s", plan.ID) {
			delete(p.reservations, id)
			log.Printf("Released resources for migration %s", plan.ID)
		}
	}

	// Clear the plan's reservations
	plan.ResourceReservations = make(map[string]float64)
}

// cleanupExpiredReservations removes expired resource reservations
func (p *MigrationPlanner) cleanupExpiredReservations() {
	p.reservationMutex.Lock()
	defer p.reservationMutex.Unlock()

	now := time.Now()
	for id, reservation := range p.reservations {
		if now.After(reservation.EndTime) {
			delete(p.reservations, id)
			log.Printf("Cleaned up expired reservation %s", id)
		}
	}
}

// groupRelatedMigrations groups related migrations together
func (p *MigrationPlanner) groupRelatedMigrations() {
	// Get pending migrations
	pendingMigrations := p.GetPendingMigrationPlans()
	if len(pendingMigrations) < 2 {
		return // Not enough migrations to group
	}

	// Find migrations to the same destination node
	destNodeGroups := make(map[string][]*MigrationPlan)
	for _, plan := range pendingMigrations {
		destNodeGroups[plan.DestNodeID] = append(destNodeGroups[plan.DestNodeID], plan)
	}

	// Create groups where there are multiple migrations to the same node
	for nodeID, plans := range destNodeGroups {
		if len(plans) < 2 || len(plans) > p.config.MaxGroupSize {
			continue
		}

		// Check if these migrations can be scheduled together
		// In a real implementation, this would be more sophisticated
		canGroup := true
		for i := 1; i < len(plans); i++ {
			timeDiff := plans[0].ScheduledWindow.StartTime.Sub(plans[i].ScheduledWindow.StartTime).Hours()
			if timeDiff < 0 {
				timeDiff = -timeDiff
			}
			if timeDiff > 6 {
				canGroup = false
				break
			}
		}

		if !canGroup {
			continue
		}

		// Create a group ID
		groupID := fmt.Sprintf("group-%d", time.Now().UnixNano())

		// Find the best window that works for all migrations
		window := p.findCommonWindow(plans)

		// Create the migration group
		group := &MigrationGroup{
			ID:              groupID,
			MigrationIDs:    make([]string, len(plans)),
			ScheduledWindow: window,
			Status:          MigrationStatusPlanned,
			CreatedAt:       time.Now(),
			Priority:        0, // Will calculate average priority
			Description:     fmt.Sprintf("Grouped migrations to node %s", nodeID),
		}

		// Add migration IDs and calculate average priority
		totalPriority := 0
		for i, plan := range plans {
			group.MigrationIDs[i] = plan.ID
			totalPriority += plan.Priority
		}
		group.Priority = totalPriority / len(plans)

		// Store the group
		p.groupMutex.Lock()
		p.groups[groupID] = group
		p.groupMutex.Unlock()

		// Update each migration plan with the new window
		p.planMutex.Lock()
		for _, plan := range plans {
			plan.ScheduledWindow = window
			plan.UpdatedAt = time.Now()
		}
		p.planMutex.Unlock()

		log.Printf("Created migration group %s with %d migrations", groupID, len(plans))
	}
}

// findCommonWindow finds a time window that works for all migrations
func (p *MigrationPlanner) findCommonWindow(plans []*MigrationPlan) MigrationWindow {
	if len(plans) == 0 {
		return MigrationWindow{}
	}

	// Start with the first plan's window
	window := plans[0].ScheduledWindow

	// Try to find a window that contains as many of the original windows as possible
	// This is a simplified approach; in a real implementation, this would be more sophisticated
	latest := window.StartTime
	earliest := window.EndTime

	for _, plan := range plans {
		if plan.ScheduledWindow.StartTime.After(latest) {
			latest = plan.ScheduledWindow.StartTime
		}
		if plan.ScheduledWindow.EndTime.Before(earliest) {
			earliest = plan.ScheduledWindow.EndTime
		}
	}

	// If we can't find a common window, use the first plan's window
	if latest.After(earliest) {
		// Use the average of all start times
		var totalStart time.Time
		for _, plan := range plans {
			totalStart = totalStart.Add(plan.ScheduledWindow.StartTime.Sub(time.Time{}))
		}
		avgStartSeconds := totalStart.Sub(time.Time{}).Seconds() / float64(len(plans))
		startTime := time.Unix(int64(avgStartSeconds), 0)

		return MigrationWindow{
			StartTime: startTime,
			EndTime:   startTime.Add(p.config.DefaultWindowDuration),
			Quality:   0.6,
			Reason:    "Averaged time for migration group",
		}
	}

	return MigrationWindow{
		StartTime: latest,
		EndTime:   earliest,
		Quality:   0.8,
		Reason:    "Common window for migration group",
	}
}

// checkReadyMigrations checks if any planned migrations should be executed now
func (p *MigrationPlanner) checkReadyMigrations() {
	now := time.Now()

	// Get pending migrations
	pendingMigrations := p.GetPendingMigrationPlans()

	// Check each migration
	for _, plan := range pendingMigrations {
		// If the start time is in the past or very soon, mark as ready
		if !plan.ScheduledWindow.StartTime.After(now.Add(5 * time.Minute)) {
			log.Printf("Migration %s is ready for execution", plan.ID)
			// In a real implementation, this would trigger the migration process
		}
	}
}

// convertToMigrationWindow converts a workload.MigrationWindow to a migration.MigrationWindow
func convertToMigrationWindow(workloadWindow workload.MigrationWindow) MigrationWindow {
	return MigrationWindow{
		StartTime: workloadWindow.StartTime,
		EndTime:   workloadWindow.EndTime,
		Quality:   workloadWindow.Quality,
		Reason:    workloadWindow.Reason,
	}
}

// absDuration returns the absolute value of a time.Duration
func absDuration(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}
