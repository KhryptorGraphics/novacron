package vm

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// MigrationPriority represents the priority level of a migration
type MigrationPriority int

const (
	// MigrationPriorityLow indicates low priority, migration can be delayed
	MigrationPriorityLow MigrationPriority = 1

	// MigrationPriorityNormal indicates normal priority
	MigrationPriorityNormal MigrationPriority = 5

	// MigrationPriorityHigh indicates high priority, should be scheduled ASAP
	MigrationPriorityHigh MigrationPriority = 8

	// MigrationPriorityCritical indicates critical priority (e.g., host failure)
	MigrationPriorityCritical MigrationPriority = 10
)

// ResourceImpact represents the impact on different resources during migration
type ResourceImpact struct {
	// CPU usage as percentage of one CPU core (100 = 1 core)
	CPUPct float64

	// Memory usage in MB
	MemoryMB int

	// Network bandwidth requirement in Mbps
	BandwidthMbps float64

	// Estimated duration in seconds
	DurationSec int

	// IO operations per second
	IOPs int
}

// DefaultResourceImpact returns default resource impact estimation
func DefaultResourceImpact() ResourceImpact {
	return ResourceImpact{
		CPUPct:        50,
		MemoryMB:      256,
		BandwidthMbps: 100,
		DurationSec:   60,
		IOPs:          1000,
	}
}

// ResourceAvailability represents available resources on a system
type ResourceAvailability struct {
	// Available CPU percentage
	CPUPct float64

	// Available memory in MB
	MemoryMB int

	// Available network bandwidth in Mbps
	BandwidthMbps float64

	// Available IO operations per second
	IOPs int
}

// MigrationSchedule represents a scheduled migration
type MigrationSchedule struct {
	// Migration ID
	MigrationID string

	// VM ID to migrate
	VMID string

	// Source and destination nodes
	SourceNodeID      string
	DestinationNodeID string

	// Priority of the migration
	Priority MigrationPriority

	// Scheduled start time
	ScheduledStartTime time.Time

	// Estimated completion time
	EstimatedEndTime time.Time

	// Migration type (cold, warm, live)
	MigrationType string

	// Estimated resource impact
	ResourceImpact ResourceImpact

	// Migration options
	Options map[string]interface{}

	// Actual start and end time (filled after execution)
	ActualStartTime time.Time
	ActualEndTime   time.Time

	// State of the scheduled migration
	State string

	// Maximum allowed downtime in milliseconds
	MaxDowntimeMs int
}

const (
	// ScheduleStatePending indicates the migration is scheduled but not started
	ScheduleStatePending = "pending"

	// ScheduleStateRunning indicates the migration is in progress
	ScheduleStateRunning = "running"

	// ScheduleStateCompleted indicates the migration completed successfully
	ScheduleStateCompleted = "completed"

	// ScheduleStateFailed indicates the migration failed
	ScheduleStateFailed = "failed"

	// ScheduleStateCancelled indicates the migration was cancelled
	ScheduleStateCancelled = "cancelled"
)

// MigrationWindow defines a time window when migrations are allowed
type MigrationWindow struct {
	// Days of week (0 = Sunday, 1 = Monday, ..., 6 = Saturday)
	// Empty means all days
	DaysOfWeek []int

	// Start time in 24-hour format (e.g., "22:00")
	StartTime string

	// End time in 24-hour format (e.g., "06:00")
	EndTime string

	// Priority threshold (only migrations with this priority or higher
	// can run outside the window)
	PriorityThreshold MigrationPriority
}

// MigrationSchedulerConfig contains configuration for the migration scheduler
type MigrationSchedulerConfig struct {
	// Maximum concurrent migrations per node
	MaxConcurrentMigrationsPerNode int

	// Maximum concurrent migrations globally
	MaxConcurrentMigrationsGlobal int

	// Migration windows
	MigrationWindows []MigrationWindow

	// Default priority
	DefaultPriority MigrationPriority

	// Resource headroom percentages
	CPUHeadroomPct     float64
	MemoryHeadroomPct  float64
	NetworkHeadroomPct float64
	IOHeadroomPct      float64

	// Retry settings
	MaxRetries         int
	RetryDelayMs       int
	RetryBackoffFactor float64

	// Default migration type
	DefaultMigrationType string
}

// DefaultMigrationSchedulerConfig returns the default configuration
func DefaultMigrationSchedulerConfig() MigrationSchedulerConfig {
	return MigrationSchedulerConfig{
		MaxConcurrentMigrationsPerNode: 2,
		MaxConcurrentMigrationsGlobal:  10,
		MigrationWindows: []MigrationWindow{
			{
				// Default maintenance window: Every day from 1 AM to 5 AM
				StartTime:         "01:00",
				EndTime:           "05:00",
				PriorityThreshold: MigrationPriorityHigh,
			},
		},
		DefaultPriority:      MigrationPriorityNormal,
		CPUHeadroomPct:       20,
		MemoryHeadroomPct:    10,
		NetworkHeadroomPct:   30,
		IOHeadroomPct:        20,
		MaxRetries:           3,
		RetryDelayMs:         60000, // 1 minute
		RetryBackoffFactor:   2.0,
		DefaultMigrationType: "live",
	}
}

// MigrationScheduler manages scheduling and prioritization of VM migrations
type MigrationScheduler struct {
	// Configuration
	config MigrationSchedulerConfig

	// Pending migrations queue
	pendingMigrations []*MigrationSchedule

	// Running migrations
	runningMigrations map[string]*MigrationSchedule

	// Completed migrations history (limited size)
	completedMigrations []*MigrationSchedule

	// Node resource information
	nodeResources map[string]ResourceAvailability

	// Mutex for thread safety
	mu sync.RWMutex

	// Context for cancellation
	ctx    context.Context
	cancel context.CancelFunc

	// Migration manager to execute migrations
	migrationManager *MigrationManager

	// Logger
	logger *logrus.Entry
}

// NewMigrationScheduler creates a new migration scheduler
func NewMigrationScheduler(config MigrationSchedulerConfig, migrationManager *MigrationManager, logger *logrus.Logger) *MigrationScheduler {
	ctx, cancel := context.WithCancel(context.Background())

	return &MigrationScheduler{
		config:              config,
		pendingMigrations:   make([]*MigrationSchedule, 0),
		runningMigrations:   make(map[string]*MigrationSchedule),
		completedMigrations: make([]*MigrationSchedule, 0, 100), // Keep last 100
		nodeResources:       make(map[string]ResourceAvailability),
		ctx:                 ctx,
		cancel:              cancel,
		migrationManager:    migrationManager,
		logger:              logger.WithField("component", "MigrationScheduler"),
	}
}

// Start starts the migration scheduler
func (s *MigrationScheduler) Start() error {
	s.logger.Info("Starting migration scheduler")

	// Start periodic scheduling
	go s.schedulingLoop()

	return nil
}

// Stop stops the migration scheduler
func (s *MigrationScheduler) Stop() error {
	s.logger.Info("Stopping migration scheduler")
	s.cancel()
	return nil
}

// ScheduleMigration schedules a migration with default priority
func (s *MigrationScheduler) ScheduleMigration(
	vmID string,
	destNodeID string,
	migrationType string,
	maxDowntimeMs int,
	options map[string]interface{},
) (string, error) {
	return s.ScheduleMigrationWithPriority(vmID, destNodeID, migrationType,
		maxDowntimeMs, options, s.config.DefaultPriority)
}

// ScheduleMigrationWithPriority schedules a migration with a specific priority
func (s *MigrationScheduler) ScheduleMigrationWithPriority(
	vmID string,
	destNodeID string,
	migrationType string,
	maxDowntimeMs int,
	options map[string]interface{},
	priority MigrationPriority,
) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Generate migration ID
	migrationID := fmt.Sprintf("migration-%s-%d", vmID, time.Now().UnixNano())

	// Get source node ID
	sourceNodeID := "unknown" // This would be fetched from VM info in a real implementation

	// Estimate resource impact
	impact := s.estimateResourceImpact(vmID, migrationType)

	// Calculate estimated start and end times
	startTime := s.estimateStartTime(priority, impact)
	endTime := startTime.Add(time.Duration(impact.DurationSec) * time.Second)

	// Create migration schedule
	schedule := &MigrationSchedule{
		MigrationID:        migrationID,
		VMID:               vmID,
		SourceNodeID:       sourceNodeID,
		DestinationNodeID:  destNodeID,
		Priority:           priority,
		ScheduledStartTime: startTime,
		EstimatedEndTime:   endTime,
		MigrationType:      migrationType,
		ResourceImpact:     impact,
		Options:            options,
		State:              ScheduleStatePending,
		MaxDowntimeMs:      maxDowntimeMs,
	}

	// Add to pending migrations
	s.pendingMigrations = append(s.pendingMigrations, schedule)

	// Sort pending migrations by priority and scheduled time
	s.sortPendingMigrations()

	s.logger.WithFields(logrus.Fields{
		"migrationID":   migrationID,
		"vmID":          vmID,
		"priority":      priority,
		"scheduledTime": startTime,
		"type":          migrationType,
	}).Info("Migration scheduled")

	// If this is a high priority migration, trigger immediate scheduling
	if priority >= MigrationPriorityHigh {
		go s.triggerScheduling()
	}

	return migrationID, nil
}

// CancelMigration cancels a scheduled migration
func (s *MigrationScheduler) CancelMigration(migrationID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check pending migrations
	for i, m := range s.pendingMigrations {
		if m.MigrationID == migrationID {
			// Remove from pending
			s.pendingMigrations = append(s.pendingMigrations[:i], s.pendingMigrations[i+1:]...)

			// Add to completed with cancelled state
			m.State = ScheduleStateCancelled
			s.addToCompleted(m)

			s.logger.WithField("migrationID", migrationID).Info("Cancelled pending migration")
			return nil
		}
	}

	// Check if it's running
	if m, exists := s.runningMigrations[migrationID]; exists {
		// Mark as cancelled
		m.State = ScheduleStateCancelled

		// In a real implementation, we would call to the migration manager to stop it
		// s.migrationManager.StopMigration(migrationID)

		s.logger.WithField("migrationID", migrationID).Info("Cancelled running migration")
		return nil
	}

	return fmt.Errorf("migration %s not found or already completed", migrationID)
}

// GetMigrationStatus returns the status of a migration
func (s *MigrationScheduler) GetMigrationStatus(migrationID string) (*MigrationSchedule, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Check pending migrations
	for _, m := range s.pendingMigrations {
		if m.MigrationID == migrationID {
			return m, nil
		}
	}

	// Check running migrations
	if m, exists := s.runningMigrations[migrationID]; exists {
		return m, nil
	}

	// Check completed migrations
	for _, m := range s.completedMigrations {
		if m.MigrationID == migrationID {
			return m, nil
		}
	}

	return nil, fmt.Errorf("migration %s not found", migrationID)
}

// GetAllMigrations returns all migrations in all states
func (s *MigrationScheduler) GetAllMigrations() []*MigrationSchedule {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Calculate total size
	totalSize := len(s.pendingMigrations) + len(s.runningMigrations) + len(s.completedMigrations)

	// Create result slice
	result := make([]*MigrationSchedule, 0, totalSize)

	// Add all migrations
	result = append(result, s.pendingMigrations...)

	for _, m := range s.runningMigrations {
		result = append(result, m)
	}

	result = append(result, s.completedMigrations...)

	return result
}

// UpdateNodeResources updates the available resources for a node
func (s *MigrationScheduler) UpdateNodeResources(nodeID string, resources ResourceAvailability) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.nodeResources[nodeID] = resources

	// This might affect scheduling, so trigger a scheduling pass
	go s.triggerScheduling()
}

// estimateResourceImpact estimates the resource impact of a migration
func (s *MigrationScheduler) estimateResourceImpact(vmID string, migrationType string) ResourceImpact {
	// In a real implementation, this would analyze the VM's characteristics
	// and estimate resource usage based on VM size, workload, etc.

	// For now, return default impact
	impact := DefaultResourceImpact()

	// Adjust based on migration type
	switch migrationType {
	case "cold":
		// Cold migrations typically use less CPU but more IO
		impact.CPUPct = 30
		impact.IOPs = 2000
		impact.DurationSec = 120
	case "warm":
		// Warm migrations use moderate resources
		impact.CPUPct = 50
		impact.IOPs = 1500
		impact.DurationSec = 90
	case "live":
		// Live migrations use more CPU and memory
		impact.CPUPct = 70
		impact.MemoryMB = 512
		impact.DurationSec = 60
	}

	return impact
}

// estimateStartTime estimates when a migration should start
func (s *MigrationScheduler) estimateStartTime(priority MigrationPriority, impact ResourceImpact) time.Time {
	now := time.Now()

	// High priority starts immediately
	if priority >= MigrationPriorityHigh {
		return now
	}

	// Check if we're in a migration window
	if s.isInMigrationWindow(now, priority) {
		return now
	}

	// Find the next migration window
	nextWindow := s.getNextMigrationWindow(now, priority)
	if !nextWindow.IsZero() {
		return nextWindow
	}

	// If no window found and not high priority, delay by a reasonable amount
	// e.g., 1 hour for normal, 6 hours for low
	switch priority {
	case MigrationPriorityNormal:
		return now.Add(1 * time.Hour)
	case MigrationPriorityLow:
		return now.Add(6 * time.Hour)
	default:
		return now
	}
}

// isInMigrationWindow checks if the given time is within a migration window
func (s *MigrationScheduler) isInMigrationWindow(t time.Time, priority MigrationPriority) bool {
	// If no windows defined, all times are valid
	if len(s.config.MigrationWindows) == 0 {
		return true
	}

	// Check each window
	for _, window := range s.config.MigrationWindows {
		// If the priority is above the threshold, it's always in a window
		if priority >= window.PriorityThreshold {
			return true
		}

		// Check day of week
		if len(window.DaysOfWeek) > 0 {
			dayMatch := false
			for _, day := range window.DaysOfWeek {
				if int(t.Weekday()) == day {
					dayMatch = true
					break
				}
			}
			if !dayMatch {
				continue
			}
		}

		// Parse start and end times
		startParts := parseTimeString(window.StartTime)
		endParts := parseTimeString(window.EndTime)

		start := time.Date(t.Year(), t.Month(), t.Day(), startParts[0], startParts[1], 0, 0, t.Location())
		end := time.Date(t.Year(), t.Month(), t.Day(), endParts[0], endParts[1], 0, 0, t.Location())

		// Handle windows that span midnight
		if end.Before(start) {
			end = end.AddDate(0, 0, 1)
		}

		// Check if current time is in window
		if (t.Equal(start) || t.After(start)) && t.Before(end) {
			return true
		}
	}

	return false
}

// getNextMigrationWindow finds the next migration window
func (s *MigrationScheduler) getNextMigrationWindow(t time.Time, priority MigrationPriority) time.Time {
	// If no windows defined or priority above threshold, return zero time
	if len(s.config.MigrationWindows) == 0 || priority >= MigrationPriorityHigh {
		return time.Time{}
	}

	// Find the earliest window that starts after t
	var earliest time.Time

	for _, window := range s.config.MigrationWindows {
		// Skip if priority is below threshold
		if priority < window.PriorityThreshold {
			continue
		}

		// Parse start time
		startParts := parseTimeString(window.StartTime)

		// Check each of the next 7 days
		for i := 0; i < 7; i++ {
			day := t.AddDate(0, 0, i)

			// Check day of week if specified
			if len(window.DaysOfWeek) > 0 {
				dayMatch := false
				for _, d := range window.DaysOfWeek {
					if int(day.Weekday()) == d {
						dayMatch = true
						break
					}
				}
				if !dayMatch {
					continue
				}
			}

			// Calculate window start on this day
			start := time.Date(day.Year(), day.Month(), day.Day(),
				startParts[0], startParts[1], 0, 0, t.Location())

			// If this window is in the future and either the first one found or earlier than previous
			if start.After(t) && (earliest.IsZero() || start.Before(earliest)) {
				earliest = start
			}
		}
	}

	return earliest
}

// parseTimeString parses a time string in format "HH:MM" to [hour, minute]
func parseTimeString(timeStr string) [2]int {
	var hour, minute int
	fmt.Sscanf(timeStr, "%d:%d", &hour, &minute)

	// Clamp values to valid ranges
	if hour < 0 {
		hour = 0
	} else if hour > 23 {
		hour = 23
	}

	if minute < 0 {
		minute = 0
	} else if minute > 59 {
		minute = 59
	}

	return [2]int{hour, minute}
}

// addToCompleted adds a migration to the completed list, maintaining the max size
func (s *MigrationScheduler) addToCompleted(migration *MigrationSchedule) {
	// Keep only the last 100 completed migrations
	const maxCompleted = 100

	s.completedMigrations = append(s.completedMigrations, migration)
	if len(s.completedMigrations) > maxCompleted {
		s.completedMigrations = s.completedMigrations[len(s.completedMigrations)-maxCompleted:]
	}
}

// sortPendingMigrations sorts pending migrations by priority and scheduled time
func (s *MigrationScheduler) sortPendingMigrations() {
	sort.Slice(s.pendingMigrations, func(i, j int) bool {
		a, b := s.pendingMigrations[i], s.pendingMigrations[j]

		// First sort by priority (higher first)
		if a.Priority != b.Priority {
			return a.Priority > b.Priority
		}

		// Then by scheduled time (earlier first)
		return a.ScheduledStartTime.Before(b.ScheduledStartTime)
	})
}

// schedulingLoop is the main scheduling loop
func (s *MigrationScheduler) schedulingLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.processSchedule()
		}
	}
}

// triggerScheduling triggers a scheduling pass immediately
func (s *MigrationScheduler) triggerScheduling() {
	s.processSchedule()
}

// processSchedule processes the migration schedule
func (s *MigrationScheduler) processSchedule() {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now()

	// First, update status of running migrations
	s.updateRunningMigrations()

	// Check if we can start more migrations
	runningCount := len(s.runningMigrations)
	if runningCount >= s.config.MaxConcurrentMigrationsGlobal {
		return
	}

	// Check per-node running counts
	nodeRunningCounts := make(map[string]int)
	for _, m := range s.runningMigrations {
		nodeRunningCounts[m.SourceNodeID]++
		nodeRunningCounts[m.DestinationNodeID]++
	}

	// Process pending migrations
	for i := 0; i < len(s.pendingMigrations); i++ {
		m := s.pendingMigrations[i]

		// Check if it's time to start this migration
		if m.ScheduledStartTime.After(now) && m.Priority < MigrationPriorityHigh {
			continue
		}

		// Check if source or destination node already has max migrations
		if nodeRunningCounts[m.SourceNodeID] >= s.config.MaxConcurrentMigrationsPerNode ||
			nodeRunningCounts[m.DestinationNodeID] >= s.config.MaxConcurrentMigrationsPerNode {
			continue
		}

		// Check if there are enough resources available
		if !s.checkResourceAvailability(m) {
			if m.Priority >= MigrationPriorityHigh {
				// For high priority, warn but continue anyway
				s.logger.WithField("migrationID", m.MigrationID).Warn(
					"Starting high priority migration despite resource constraints")
			} else {
				// For normal priority, skip
				continue
			}
		}

		// Start the migration
		err := s.startMigration(m)
		if err != nil {
			s.logger.WithFields(logrus.Fields{
				"migrationID": m.MigrationID,
				"error":       err,
			}).Error("Failed to start migration")
			continue
		}

		// Move from pending to running
		s.pendingMigrations = append(s.pendingMigrations[:i], s.pendingMigrations[i+1:]...)
		s.runningMigrations[m.MigrationID] = m
		i-- // Adjust index after removal

		// Update node running counts
		nodeRunningCounts[m.SourceNodeID]++
		nodeRunningCounts[m.DestinationNodeID]++

		// Check if we've reached the global limit
		runningCount++
		if runningCount >= s.config.MaxConcurrentMigrationsGlobal {
			break
		}
	}
}

// updateRunningMigrations updates the status of running migrations
func (s *MigrationScheduler) updateRunningMigrations() {
	for id, m := range s.runningMigrations {
		// In a real implementation, we would query the migration manager
		// status := s.migrationManager.GetMigrationStatus(id)

		// For now, just simulate migrations completing after their estimated duration
		if m.ActualStartTime.Add(time.Duration(m.ResourceImpact.DurationSec) * time.Second).Before(time.Now()) {
			// Migration completed
			m.State = ScheduleStateCompleted
			m.ActualEndTime = time.Now()

			// Move to completed list
			s.addToCompleted(m)

			// Remove from running
			delete(s.runningMigrations, id)

			s.logger.WithField("migrationID", id).Info("Migration completed")
		}
	}
}

// checkResourceAvailability checks if there are enough resources to run a migration
func (s *MigrationScheduler) checkResourceAvailability(migration *MigrationSchedule) bool {
	// Check source node resources
	sourceResources, sourceExists := s.nodeResources[migration.SourceNodeID]
	destResources, destExists := s.nodeResources[migration.DestinationNodeID]

	// If we don't have resource info, assume enough resources
	if !sourceExists || !destExists {
		return true
	}

	impact := migration.ResourceImpact

	// Check source node
	if sourceResources.CPUPct < impact.CPUPct*(1+s.config.CPUHeadroomPct/100) {
		return false
	}
	if sourceResources.MemoryMB < impact.MemoryMB*(1+int(s.config.MemoryHeadroomPct/100)) {
		return false
	}
	if sourceResources.BandwidthMbps < impact.BandwidthMbps*(1+s.config.NetworkHeadroomPct/100) {
		return false
	}
	if sourceResources.IOPs < impact.IOPs*(1+int(s.config.IOHeadroomPct/100)) {
		return false
	}

	// Check destination node
	if destResources.CPUPct < impact.CPUPct*(1+s.config.CPUHeadroomPct/100) {
		return false
	}
	if destResources.MemoryMB < impact.MemoryMB*(1+int(s.config.MemoryHeadroomPct/100)) {
		return false
	}
	if destResources.BandwidthMbps < impact.BandwidthMbps*(1+s.config.NetworkHeadroomPct/100) {
		return false
	}
	if destResources.IOPs < impact.IOPs*(1+int(s.config.IOHeadroomPct/100)) {
		return false
	}

	return true
}

// startMigration starts a migration
func (s *MigrationScheduler) startMigration(migration *MigrationSchedule) error {
	// In a real implementation, this would call the migration manager
	// err := s.migrationManager.StartMigration(migration.VMID, migration.DestinationNodeID, options)

	// For now, just update state
	migration.State = ScheduleStateRunning
	migration.ActualStartTime = time.Now()

	s.logger.WithFields(logrus.Fields{
		"migrationID": migration.MigrationID,
		"vmID":        migration.VMID,
		"type":        migration.MigrationType,
	}).Info("Started migration")

	return nil
}
