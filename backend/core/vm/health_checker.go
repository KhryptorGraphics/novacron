package vm

import (
	"context"
	"crypto/rand"
	"fmt"
	"sync"
	"time"
)

// HealthChecker monitors VM health
type HealthChecker struct {
	vmID           string
	checks         []HealthCheckSpec
	interval       time.Duration
	ctx            context.Context
	cancel         context.CancelFunc
	lastStatus     HealthState
	statusHistory  []HealthCheckResult
	mu             sync.RWMutex
	eventCallback  func(result HealthCheckResult)
}

// HealthCheckSpec defines a health check specification
type HealthCheckSpec struct {
	Name        string
	Description string
	CheckFunc   func() HealthCheckResult
	Timeout     time.Duration
	Critical    bool
}

// HealthState represents overall health state
type HealthState int

const (
	HealthUnknown HealthState = iota
	HealthHealthy
	HealthWarning
	HealthCritical
)

func (hs HealthState) String() string {
	states := []string{"Unknown", "Healthy", "Warning", "Critical"}
	if int(hs) < len(states) {
		return states[hs]
	}
	return "Unknown"
}

// HealthCheckResult represents the result of a health check
type HealthCheckResult struct {
	CheckName   string        `json:"check_name"`
	Status      HealthState  `json:"status"`
	Message     string        `json:"message"`
	Timestamp   time.Time     `json:"timestamp"`
	Duration    time.Duration `json:"duration"`
	Error       string        `json:"error,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// NewHealthChecker creates a new health checker for a VM
func NewHealthChecker(vmID string) *HealthChecker {
	ctx, cancel := context.WithCancel(context.Background())
	
	hc := &HealthChecker{
		vmID:          vmID,
		checks:        make([]HealthCheckSpec, 0),
		interval:      30 * time.Second,
		ctx:           ctx,
		cancel:        cancel,
		lastStatus:    HealthUnknown,
		statusHistory: make([]HealthCheckResult, 0),
	}
	
	// Add default health checks
	hc.addDefaultChecks()
	
	return hc
}

// AddCheck adds a custom health check
func (hc *HealthChecker) AddCheck(check HealthCheckSpec) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	
	hc.checks = append(hc.checks, check)
}

// SetInterval sets the health check interval
func (hc *HealthChecker) SetInterval(interval time.Duration) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	
	hc.interval = interval
}

// SetEventCallback sets a callback for health check events
func (hc *HealthChecker) SetEventCallback(callback func(result HealthCheckResult)) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	
	hc.eventCallback = callback
}

// Start starts the health checker
func (hc *HealthChecker) Start() {
	go hc.run()
}

// Stop stops the health checker
func (hc *HealthChecker) Stop() {
	hc.cancel()
}

// GetStatus returns the current health status
func (hc *HealthChecker) GetStatus() HealthState {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	
	return hc.lastStatus
}

// GetHistory returns the health check history
func (hc *HealthChecker) GetHistory(limit int) []HealthCheckResult {
	hc.mu.RLock()
	defer hc.mu.RUnlock()
	
	if limit <= 0 || limit > len(hc.statusHistory) {
		limit = len(hc.statusHistory)
	}
	
	result := make([]HealthCheckResult, limit)
	copy(result, hc.statusHistory[len(hc.statusHistory)-limit:])
	
	return result
}

// RunHealthCheck performs an immediate health check
func (hc *HealthChecker) RunHealthCheck() []HealthCheckResult {
	hc.mu.RLock()
	checks := make([]HealthCheckSpec, len(hc.checks))
	copy(checks, hc.checks)
	hc.mu.RUnlock()
	
	results := make([]HealthCheckResult, 0, len(checks))
	overallStatus := HealthHealthy
	
	for _, check := range checks {
		result := hc.runSingleCheck(check)
		results = append(results, result)
		
		// Update overall status
		if result.Status == HealthCritical {
			overallStatus = HealthCritical
		} else if result.Status == HealthWarning && overallStatus != HealthCritical {
			overallStatus = HealthWarning
		}
	}
	
	// Update status and history
	hc.mu.Lock()
	hc.lastStatus = overallStatus
	hc.statusHistory = append(hc.statusHistory, results...)
	
	// Keep only recent history
	maxHistory := 1000
	if len(hc.statusHistory) > maxHistory {
		hc.statusHistory = hc.statusHistory[len(hc.statusHistory)-maxHistory:]
	}
	hc.mu.Unlock()
	
	// Trigger callbacks
	if hc.eventCallback != nil {
		for _, result := range results {
			go hc.eventCallback(result)
		}
	}
	
	return results
}

// Internal methods

func (hc *HealthChecker) run() {
	ticker := time.NewTicker(hc.interval)
	defer ticker.Stop()
	
	// Perform initial health check
	hc.RunHealthCheck()
	
	for {
		select {
		case <-ticker.C:
			hc.RunHealthCheck()
		case <-hc.ctx.Done():
			return
		}
	}
}

func (hc *HealthChecker) runSingleCheck(check HealthCheckSpec) HealthCheckResult {
	start := time.Now()
	
	// Set up timeout context
	ctx, cancel := context.WithTimeout(hc.ctx, check.Timeout)
	defer cancel()
	
	// Run the check in a goroutine to handle timeouts
	resultChan := make(chan HealthCheckResult, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				resultChan <- HealthCheckResult{
					CheckName: check.Name,
					Status:    HealthCritical,
					Message:   "Health check panicked",
					Timestamp: time.Now(),
					Duration:  time.Since(start),
					Error:     "panic recovered",
				}
			}
		}()
		
		result := check.CheckFunc()
		result.CheckName = check.Name
		result.Timestamp = time.Now()
		result.Duration = time.Since(start)
		resultChan <- result
	}()
	
	// Wait for result or timeout
	select {
	case result := <-resultChan:
		return result
	case <-ctx.Done():
		return HealthCheckResult{
			CheckName: check.Name,
			Status:    HealthCritical,
			Message:   "Health check timed out",
			Timestamp: time.Now(),
			Duration:  check.Timeout,
			Error:     "timeout",
		}
	}
}

func (hc *HealthChecker) addDefaultChecks() {
	// Process existence check
	hc.AddCheck(HealthCheckSpec{
		Name:        "process_check",
		Description: "Check if VM process is running",
		CheckFunc: func() HealthCheckResult {
			// In a real implementation, this would check if the VM process is running
			return HealthCheckResult{
				Status:  HealthHealthy,
				Message: "VM process is running",
				Metadata: map[string]interface{}{
					"pid": 12345,
				},
			}
		},
		Timeout:  5 * time.Second,
		Critical: true,
	})
	
	// Memory usage check
	hc.AddCheck(HealthCheckSpec{
		Name:        "memory_check",
		Description: "Check VM memory usage",
		CheckFunc: func() HealthCheckResult {
			// Simulate memory check
			memoryUsage := 75.0 // Percentage
			
			status := HealthHealthy
			message := "Memory usage normal"
			
			if memoryUsage > 90 {
				status = HealthCritical
				message = "Memory usage critical"
			} else if memoryUsage > 80 {
				status = HealthWarning
				message = "Memory usage high"
			}
			
			return HealthCheckResult{
				Status:  status,
				Message: message,
				Metadata: map[string]interface{}{
					"memory_usage_percent": memoryUsage,
				},
			}
		},
		Timeout:  3 * time.Second,
		Critical: false,
	})
	
	// CPU usage check
	hc.AddCheck(HealthCheckSpec{
		Name:        "cpu_check",
		Description: "Check VM CPU usage",
		CheckFunc: func() HealthCheckResult {
			// Simulate CPU check
			cpuUsage := 45.0 // Percentage
			
			status := HealthHealthy
			message := "CPU usage normal"
			
			if cpuUsage > 95 {
				status = HealthCritical
				message = "CPU usage critical"
			} else if cpuUsage > 85 {
				status = HealthWarning
				message = "CPU usage high"
			}
			
			return HealthCheckResult{
				Status:  status,
				Message: message,
				Metadata: map[string]interface{}{
					"cpu_usage_percent": cpuUsage,
				},
			}
		},
		Timeout:  3 * time.Second,
		Critical: false,
	})
	
	// Disk space check
	hc.AddCheck(HealthCheckSpec{
		Name:        "disk_check",
		Description: "Check VM disk space",
		CheckFunc: func() HealthCheckResult {
			// Simulate disk space check
			diskUsage := 70.0 // Percentage
			
			status := HealthHealthy
			message := "Disk space normal"
			
			if diskUsage > 95 {
				status = HealthCritical
				message = "Disk space critical"
			} else if diskUsage > 85 {
				status = HealthWarning
				message = "Disk space low"
			}
			
			return HealthCheckResult{
				Status:  status,
				Message: message,
				Metadata: map[string]interface{}{
					"disk_usage_percent": diskUsage,
				},
			}
		},
		Timeout:  3 * time.Second,
		Critical: false,
	})
	
	// Network connectivity check
	hc.AddCheck(HealthCheckSpec{
		Name:        "network_check",
		Description: "Check VM network connectivity",
		CheckFunc: func() HealthCheckResult {
			// Simulate network connectivity check
			networkOk := true
			
			if networkOk {
				return HealthCheckResult{
					Status:  HealthHealthy,
					Message: "Network connectivity OK",
					Metadata: map[string]interface{}{
						"interfaces_up": 1,
						"ping_latency":  "5ms",
					},
				}
			}
			
			return HealthCheckResult{
				Status:  HealthCritical,
				Message: "Network connectivity failed",
				Error:   "network interface down",
			}
		},
		Timeout:  10 * time.Second,
		Critical: false,
	})
}

// LifecycleScheduler handles scheduled lifecycle operations
type LifecycleScheduler struct {
	tasks []ScheduledTask
	mu    sync.RWMutex
	ctx   context.Context
	cancel context.CancelFunc
}

// ScheduledTask represents a scheduled lifecycle task
type ScheduledTask struct {
	ID          string
	VMID        string
	Operation   string
	Schedule    string // Cron expression
	NextRun     time.Time
	LastRun     time.Time
	Enabled     bool
	Parameters  map[string]interface{}
}

// NewLifecycleScheduler creates a new lifecycle scheduler
func NewLifecycleScheduler() *LifecycleScheduler {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &LifecycleScheduler{
		tasks:  make([]ScheduledTask, 0),
		ctx:    ctx,
		cancel: cancel,
	}
}

// AddTask adds a scheduled task
func (ls *LifecycleScheduler) AddTask(task ScheduledTask) {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	
	ls.tasks = append(ls.tasks, task)
}

// RemoveTask removes a scheduled task
func (ls *LifecycleScheduler) RemoveTask(taskID string) {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	
	for i, task := range ls.tasks {
		if task.ID == taskID {
			ls.tasks = append(ls.tasks[:i], ls.tasks[i+1:]...)
			break
		}
	}
}

// Start starts the scheduler
func (ls *LifecycleScheduler) Start() {
	go ls.run()
}

// Stop stops the scheduler
func (ls *LifecycleScheduler) Stop() {
	ls.cancel()
}

func (ls *LifecycleScheduler) run() {
	ticker := time.NewTicker(60 * time.Second) // Check every minute
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			ls.processTasks()
		case <-ls.ctx.Done():
			return
		}
	}
}

func (ls *LifecycleScheduler) processTasks() {
	ls.mu.RLock()
	tasks := make([]ScheduledTask, len(ls.tasks))
	copy(tasks, ls.tasks)
	ls.mu.RUnlock()
	
	now := time.Now()
	
	for _, task := range tasks {
		if task.Enabled && now.After(task.NextRun) {
			go ls.executeTask(task)
		}
	}
}

func (ls *LifecycleScheduler) executeTask(task ScheduledTask) {
	// In a real implementation, this would execute the scheduled operation
	// For now, just log the execution
}

// LiveMigrationManager handles live VM migration
type LiveMigrationManager struct {
	activeMigrations map[string]*MigrationSession
	mu              sync.RWMutex
}

// MigrationSession represents an active migration session
type MigrationSession struct {
	ID          string
	VMID        string
	Source      string
	Destination string
	Type        MigrationType
	Status      *LiveMigrationStatus
	StartTime   time.Time
	ctx         context.Context
	cancel      context.CancelFunc
}

// NewLiveMigrationManager creates a new live migration manager
func NewLiveMigrationManager() *LiveMigrationManager {
	return &LiveMigrationManager{
		activeMigrations: make(map[string]*MigrationSession),
	}
}

// StartMigration starts a live migration
func (lmm *LiveMigrationManager) StartMigration(vmID, source, destination string, migrationType MigrationType) (*MigrationSession, error) {
	lmm.mu.Lock()
	defer lmm.mu.Unlock()
	
	// Check if migration is already in progress
	for _, session := range lmm.activeMigrations {
		if session.VMID == vmID && session.Status.InProgress {
			return nil, fmt.Errorf("migration already in progress for VM %s", vmID)
		}
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	
	session := &MigrationSession{
		ID:          generateMigrationID(),
		VMID:        vmID,
		Source:      source,
		Destination: destination,
		Type:        migrationType,
		StartTime:   time.Now(),
		ctx:         ctx,
		cancel:      cancel,
		Status: &LiveMigrationStatus{
			InProgress:      true,
			Type:            migrationType,
			Source:          source,
			Destination:     destination,
			Progress:        0.0,
			Phase:           PhasePreMigration,
			StartTime:       time.Now(),
			BytesTransferred: 0,
			TotalBytes:      estimateMigrationSize(vmID),
		},
	}
	
	lmm.activeMigrations[session.ID] = session
	
	// Start migration process
	go lmm.performMigration(session)
	
	return session, nil
}

// GetMigrationStatus returns the status of a migration
func (lmm *LiveMigrationManager) GetMigrationStatus(migrationID string) (*LiveMigrationStatus, error) {
	lmm.mu.RLock()
	defer lmm.mu.RUnlock()
	
	session, exists := lmm.activeMigrations[migrationID]
	if !exists {
		return nil, fmt.Errorf("migration not found: %s", migrationID)
	}
	
	return session.Status, nil
}

func (lmm *LiveMigrationManager) performMigration(session *MigrationSession) {
	// Simulate migration phases
	phases := []MigrationPhase{
		PhasePreMigration,
		PhaseMemoryCopy,
		PhaseFinalSync,
		PhaseHandover,
		PhasePostMigration,
	}
	
	for i, phase := range phases {
		session.Status.Phase = phase
		session.Status.Progress = float64(i) / float64(len(phases)) * 100
		
		// Simulate phase duration
		time.Sleep(time.Duration(i+1) * 100 * time.Millisecond)
		
		select {
		case <-session.ctx.Done():
			session.Status.InProgress = false
			session.Status.Error = "Migration cancelled"
			return
		default:
		}
	}
	
	// Migration completed
	session.Status.InProgress = false
	session.Status.Progress = 100.0
	session.Status.EstimatedFinish = time.Now()
	
	// Clean up
	lmm.mu.Lock()
	delete(lmm.activeMigrations, session.ID)
	lmm.mu.Unlock()
}

func generateMigrationID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return fmt.Sprintf("migration_%x", bytes)
}

func estimateMigrationSize(vmID string) int64 {
	// Simulate size estimation
	return 4 * 1024 * 1024 * 1024 // 4GB
}