package backup

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

const (
	// GFS retention schedule constants
	DefaultDailyRetention   = 7   // Keep daily backups for 7 days
	DefaultWeeklyRetention  = 4   // Keep weekly backups for 4 weeks
	DefaultMonthlyRetention = 12  // Keep monthly backups for 12 months
	DefaultYearlyRetention  = 3   // Keep yearly backups for 3 years
	
	// Retention policy file
	RetentionPolicyFile = "retention_policy.json"
	
	// Retention job status
	RetentionJobRunning   = "running"
	RetentionJobCompleted = "completed"
	RetentionJobFailed    = "failed"
)

// RetentionManager manages backup retention policies
type RetentionManager struct {
	config       *RetentionConfig
	policies     map[string]*RetentionPolicy
	mutex        sync.RWMutex
	scheduler    *RetentionScheduler
	backupManager *IncrementalBackupManager
}

// RetentionConfig configures retention management
type RetentionConfig struct {
	BasePath              string        `json:"base_path"`
	DefaultPolicy         string        `json:"default_policy"`
	CleanupInterval       time.Duration `json:"cleanup_interval"`
	EnableGFS             bool          `json:"enable_gfs"`
	MaxStorageQuota       int64         `json:"max_storage_quota"` // bytes
	StorageWarningPercent int           `json:"storage_warning_percent"`
	DryRun                bool          `json:"dry_run"`
}

// RetentionPolicy defines backup retention rules
type RetentionPolicy struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Rules       *RetentionRules   `json:"rules"`
	GFSConfig   *GFSConfig        `json:"gfs_config,omitempty"`
	Metadata    map[string]string `json:"metadata"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
	Enabled     bool              `json:"enabled"`
}

// RetentionRules defines retention rules
type RetentionRules struct {
	// Time-based retention
	MaxAge          time.Duration `json:"max_age"`
	MaxCount        int           `json:"max_count"`
	MinCount        int           `json:"min_count"` // Always keep at least this many
	
	// Size-based retention
	MaxTotalSize    int64         `json:"max_total_size"` // bytes
	
	// Pattern-based retention
	KeepDaily       int           `json:"keep_daily"`
	KeepWeekly      int           `json:"keep_weekly"`
	KeepMonthly     int           `json:"keep_monthly"`
	KeepYearly      int           `json:"keep_yearly"`
	
	// Custom rules
	CustomRules     []CustomRule  `json:"custom_rules,omitempty"`
}

// GFSConfig defines Grandfather-Father-Son retention scheme
type GFSConfig struct {
	Enabled         bool `json:"enabled"`
	DailyRetention  int  `json:"daily_retention"`
	WeeklyRetention int  `json:"weekly_retention"`
	MonthlyRetention int `json:"monthly_retention"`
	YearlyRetention int  `json:"yearly_retention"`
	
	// Week starts on (0 = Sunday, 1 = Monday, etc.)
	WeekStartDay    int `json:"week_start_day"`
	
	// Month starts on day (1-31)
	MonthStartDay   int `json:"month_start_day"`
}

// CustomRule defines a custom retention rule
type CustomRule struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Condition   string            `json:"condition"`   // Expression to evaluate
	Action      string            `json:"action"`      // keep, delete, archive
	Priority    int               `json:"priority"`    // Higher priority rules evaluated first
	Metadata    map[string]string `json:"metadata"`
}

// RetentionScheduler schedules and executes retention jobs
type RetentionScheduler struct {
	config    *RetentionConfig
	jobs      map[string]*RetentionJob
	mutex     sync.RWMutex
	ticker    *time.Ticker
	stopCh    chan struct{}
	doneCh    chan struct{}
}

// RetentionJob represents a retention cleanup job
type RetentionJob struct {
	ID            string            `json:"id"`
	PolicyID      string            `json:"policy_id"`
	VMID          string            `json:"vm_id"`
	Status        string            `json:"status"`
	StartedAt     time.Time         `json:"started_at"`
	CompletedAt   time.Time         `json:"completed_at,omitempty"`
	ProcessedBackups int            `json:"processed_backups"`
	DeletedBackups   int            `json:"deleted_backups"`
	FreedBytes       int64          `json:"freed_bytes"`
	Error            string         `json:"error,omitempty"`
	Metadata         map[string]string `json:"metadata"`
}

// BackupItem represents a backup for retention processing
type BackupItem struct {
	ID          string            `json:"id"`
	VMID        string            `json:"vm_id"`
	Type        BackupType        `json:"type"`
	Size        int64             `json:"size"`
	CreatedAt   time.Time         `json:"created_at"`
	Path        string            `json:"path"`
	Metadata    map[string]string `json:"metadata"`
	PolicyID    string            `json:"policy_id"`
	Protected   bool              `json:"protected"` // Cannot be deleted by retention
}

// NewRetentionManager creates a new retention manager
func NewRetentionManager(config *RetentionConfig, backupManager *IncrementalBackupManager) (*RetentionManager, error) {
	if config == nil {
		config = &RetentionConfig{
			BasePath:              "/var/lib/novacron/retention",
			DefaultPolicy:         "default-gfs",
			CleanupInterval:       time.Hour,
			EnableGFS:             true,
			MaxStorageQuota:       100 * 1024 * 1024 * 1024, // 100GB
			StorageWarningPercent: 85,
			DryRun:                false,
		}
	}
	
	// Ensure base path exists
	if err := os.MkdirAll(config.BasePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create retention base path: %w", err)
	}
	
	manager := &RetentionManager{
		config:        config,
		policies:      make(map[string]*RetentionPolicy),
		backupManager: backupManager,
	}
	
	// Initialize scheduler
	scheduler := &RetentionScheduler{
		config: config,
		jobs:   make(map[string]*RetentionJob),
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}
	manager.scheduler = scheduler
	
	// Load existing policies
	if err := manager.loadPolicies(); err != nil {
		return nil, fmt.Errorf("failed to load retention policies: %w", err)
	}
	
	// Create default policy if none exists
	if len(manager.policies) == 0 {
		if err := manager.createDefaultPolicy(); err != nil {
			return nil, fmt.Errorf("failed to create default policy: %w", err)
		}
	}
	
	return manager, nil
}

// Start starts the retention manager
func (rm *RetentionManager) Start(ctx context.Context) error {
	return rm.scheduler.Start(ctx, rm)
}

// Stop stops the retention manager
func (rm *RetentionManager) Stop() error {
	return rm.scheduler.Stop()
}

// CreatePolicy creates a new retention policy
func (rm *RetentionManager) CreatePolicy(policy *RetentionPolicy) error {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()
	
	if _, exists := rm.policies[policy.ID]; exists {
		return fmt.Errorf("policy %s already exists", policy.ID)
	}
	
	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()
	
	rm.policies[policy.ID] = policy
	
	return rm.savePolicy(policy)
}

// UpdatePolicy updates an existing retention policy
func (rm *RetentionManager) UpdatePolicy(policy *RetentionPolicy) error {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()
	
	existing, exists := rm.policies[policy.ID]
	if !exists {
		return fmt.Errorf("policy %s not found", policy.ID)
	}
	
	policy.CreatedAt = existing.CreatedAt
	policy.UpdatedAt = time.Now()
	
	rm.policies[policy.ID] = policy
	
	return rm.savePolicy(policy)
}

// DeletePolicy deletes a retention policy
func (rm *RetentionManager) DeletePolicy(policyID string) error {
	rm.mutex.Lock()
	defer rm.mutex.Unlock()
	
	if policyID == rm.config.DefaultPolicy {
		return fmt.Errorf("cannot delete default policy")
	}
	
	if _, exists := rm.policies[policyID]; !exists {
		return fmt.Errorf("policy %s not found", policyID)
	}
	
	delete(rm.policies, policyID)
	
	// Remove policy file
	policyPath := filepath.Join(rm.config.BasePath, "policies", policyID+".json")
	return os.Remove(policyPath)
}

// GetPolicy retrieves a retention policy
func (rm *RetentionManager) GetPolicy(policyID string) (*RetentionPolicy, error) {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	
	policy, exists := rm.policies[policyID]
	if !exists {
		return nil, fmt.Errorf("policy %s not found", policyID)
	}
	
	// Return a copy
	policyCopy := *policy
	return &policyCopy, nil
}

// ListPolicies returns all retention policies
func (rm *RetentionManager) ListPolicies() []*RetentionPolicy {
	rm.mutex.RLock()
	defer rm.mutex.RUnlock()
	
	policies := make([]*RetentionPolicy, 0, len(rm.policies))
	for _, policy := range rm.policies {
		policyCopy := *policy
		policies = append(policies, &policyCopy)
	}
	
	return policies
}

// ApplyRetention applies retention policy to VM backups
func (rm *RetentionManager) ApplyRetention(vmID string, policyID string) (*RetentionJob, error) {
	// Get policy
	policy, err := rm.GetPolicy(policyID)
	if err != nil {
		// Fall back to default policy
		policy, err = rm.GetPolicy(rm.config.DefaultPolicy)
		if err != nil {
			return nil, fmt.Errorf("failed to get retention policy: %w", err)
		}
	}
	
	if !policy.Enabled {
		return nil, fmt.Errorf("policy %s is disabled", policyID)
	}
	
	// Create retention job
	job := &RetentionJob{
		ID:        fmt.Sprintf("%s-%s-%d", vmID, policyID, time.Now().Unix()),
		PolicyID:  policyID,
		VMID:      vmID,
		Status:    RetentionJobRunning,
		StartedAt: time.Now(),
		Metadata:  make(map[string]string),
	}
	
	// Add to scheduler
	rm.scheduler.mutex.Lock()
	rm.scheduler.jobs[job.ID] = job
	rm.scheduler.mutex.Unlock()
	
	// Execute retention in background
	go rm.executeRetention(job, policy)
	
	return job, nil
}

// executeRetention executes the retention policy
func (rm *RetentionManager) executeRetention(job *RetentionJob, policy *RetentionPolicy) {
	defer func() {
		job.CompletedAt = time.Now()
		if job.Status == RetentionJobRunning {
			job.Status = RetentionJobCompleted
		}
	}()
	
	// Get backups for VM
	backups, err := rm.getVMBackups(job.VMID)
	if err != nil {
		job.Status = RetentionJobFailed
		job.Error = fmt.Sprintf("failed to get VM backups: %v", err)
		return
	}
	
	job.ProcessedBackups = len(backups)
	
	// Apply retention logic
	toDelete, err := rm.calculateDeletions(backups, policy)
	if err != nil {
		job.Status = RetentionJobFailed
		job.Error = fmt.Sprintf("failed to calculate deletions: %v", err)
		return
	}
	
	// Delete backups
	var freedBytes int64
	deletedCount := 0
	
	for _, backup := range toDelete {
		if backup.Protected {
			continue // Skip protected backups
		}
		
		if rm.config.DryRun {
			freedBytes += backup.Size
			deletedCount++
			continue
		}
		
		// Delete backup
		if err := rm.deleteBackup(backup); err != nil {
			job.Error = fmt.Sprintf("failed to delete backup %s: %v", backup.ID, err)
			continue
		}
		
		freedBytes += backup.Size
		deletedCount++
	}
	
	job.DeletedBackups = deletedCount
	job.FreedBytes = freedBytes
}

// calculateDeletions determines which backups to delete based on policy
func (rm *RetentionManager) calculateDeletions(backups []*BackupItem, policy *RetentionPolicy) ([]*BackupItem, error) {
	if len(backups) == 0 {
		return nil, nil
	}
	
	// Sort backups by creation time (newest first)
	sort.Slice(backups, func(i, j int) bool {
		return backups[i].CreatedAt.After(backups[j].CreatedAt)
	})
	
	var toDelete []*BackupItem
	
	// Apply GFS retention if enabled
	if policy.GFSConfig != nil && policy.GFSConfig.Enabled {
		toDelete = rm.applyGFSRetention(backups, policy.GFSConfig)
	} else {
		// Apply standard retention rules
		toDelete = rm.applyStandardRetention(backups, policy.Rules)
	}
	
	return toDelete, nil
}

// applyGFSRetention applies Grandfather-Father-Son retention scheme
func (rm *RetentionManager) applyGFSRetention(backups []*BackupItem, gfsConfig *GFSConfig) []*BackupItem {
	if len(backups) == 0 {
		return nil
	}
	
	now := time.Now()
	toKeep := make(map[string]bool)
	
	// Daily retention
	dailyBackups := rm.filterBackupsByPeriod(backups, "daily", gfsConfig.DailyRetention, now)
	for _, backup := range dailyBackups {
		toKeep[backup.ID] = true
	}
	
	// Weekly retention
	weeklyBackups := rm.filterBackupsByPeriod(backups, "weekly", gfsConfig.WeeklyRetention, now)
	for _, backup := range weeklyBackups {
		toKeep[backup.ID] = true
	}
	
	// Monthly retention
	monthlyBackups := rm.filterBackupsByPeriod(backups, "monthly", gfsConfig.MonthlyRetention, now)
	for _, backup := range monthlyBackups {
		toKeep[backup.ID] = true
	}
	
	// Yearly retention
	yearlyBackups := rm.filterBackupsByPeriod(backups, "yearly", gfsConfig.YearlyRetention, now)
	for _, backup := range yearlyBackups {
		toKeep[backup.ID] = true
	}
	
	// Determine what to delete
	var toDelete []*BackupItem
	for _, backup := range backups {
		if !toKeep[backup.ID] && !backup.Protected {
			toDelete = append(toDelete, backup)
		}
	}
	
	return toDelete
}

// applyStandardRetention applies standard retention rules
func (rm *RetentionManager) applyStandardRetention(backups []*BackupItem, rules *RetentionRules) []*BackupItem {
	if len(backups) == 0 {
		return nil
	}
	
	var toDelete []*BackupItem
	now := time.Now()
	
	// Filter by age
	var validBackups []*BackupItem
	for _, backup := range backups {
		age := now.Sub(backup.CreatedAt)
		if rules.MaxAge > 0 && age > rules.MaxAge && !backup.Protected {
			toDelete = append(toDelete, backup)
		} else {
			validBackups = append(validBackups, backup)
		}
	}
	
	// Filter by count
	if rules.MaxCount > 0 && len(validBackups) > rules.MaxCount {
		// Keep the newest ones, delete the oldest
		excess := len(validBackups) - rules.MaxCount
		for i := len(validBackups) - excess; i < len(validBackups); i++ {
			if !validBackups[i].Protected {
				toDelete = append(toDelete, validBackups[i])
			}
		}
	}
	
	// Apply minimum count constraint
	if rules.MinCount > 0 {
		keptCount := len(validBackups) - len(toDelete)
		if keptCount < rules.MinCount {
			// Remove some backups from deletion list to satisfy min count
			needed := rules.MinCount - keptCount
			if needed > len(toDelete) {
				needed = len(toDelete)
			}
			toDelete = toDelete[needed:]
		}
	}
	
	// Apply size-based retention
	if rules.MaxTotalSize > 0 {
		totalSize := rm.calculateTotalSize(validBackups)
		if totalSize > rules.MaxTotalSize {
			// Sort by creation time (oldest first) and delete until under limit
			sort.Slice(validBackups, func(i, j int) bool {
				return validBackups[i].CreatedAt.Before(validBackups[j].CreatedAt)
			})
			
			for _, backup := range validBackups {
				if totalSize <= rules.MaxTotalSize {
					break
				}
				if !backup.Protected {
					found := false
					for _, existing := range toDelete {
						if existing.ID == backup.ID {
							found = true
							break
						}
					}
					if !found {
						toDelete = append(toDelete, backup)
						totalSize -= backup.Size
					}
				}
			}
		}
	}
	
	return toDelete
}

// filterBackupsByPeriod filters backups by time period for GFS retention
func (rm *RetentionManager) filterBackupsByPeriod(backups []*BackupItem, period string, count int, now time.Time) []*BackupItem {
	if count <= 0 {
		return nil
	}
	
	groups := make(map[string]*BackupItem)
	
	for _, backup := range backups {
		var key string
		switch period {
		case "daily":
			key = backup.CreatedAt.Format("2006-01-02")
		case "weekly":
			year, week := backup.CreatedAt.ISOWeek()
			key = fmt.Sprintf("%d-W%d", year, week)
		case "monthly":
			key = backup.CreatedAt.Format("2006-01")
		case "yearly":
			key = backup.CreatedAt.Format("2006")
		}
		
		// Keep the newest backup in each period
		if existing, exists := groups[key]; !exists || backup.CreatedAt.After(existing.CreatedAt) {
			groups[key] = backup
		}
	}
	
	// Sort periods and take the most recent ones
	var keys []string
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	
	// Reverse to get newest first
	for i := len(keys)/2 - 1; i >= 0; i-- {
		opp := len(keys) - 1 - i
		keys[i], keys[opp] = keys[opp], keys[i]
	}
	
	// Take up to 'count' periods
	var result []*BackupItem
	taken := 0
	for _, key := range keys {
		if taken >= count {
			break
		}
		if backup := groups[key]; backup != nil {
			result = append(result, backup)
			taken++
		}
	}
	
	return result
}

// calculateTotalSize calculates total size of backups
func (rm *RetentionManager) calculateTotalSize(backups []*BackupItem) int64 {
	var total int64
	for _, backup := range backups {
		total += backup.Size
	}
	return total
}

// getVMBackups retrieves all backups for a VM
func (rm *RetentionManager) getVMBackups(vmID string) ([]*BackupItem, error) {
	backupIDs, err := rm.backupManager.ListBackups(vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to list backups: %w", err)
	}
	
	var backups []*BackupItem
	for _, backupID := range backupIDs {
		manifest, err := rm.backupManager.GetBackupManifest(backupID)
		if err != nil {
			continue // Skip invalid backups
		}
		
		backup := &BackupItem{
			ID:        manifest.BackupID,
			VMID:      manifest.VMID,
			Type:      manifest.Type,
			Size:      manifest.Size,
			CreatedAt: manifest.CreatedAt,
			Path:      filepath.Join(rm.backupManager.config.BasePath, "data", manifest.BackupID),
			Metadata:  manifest.Metadata,
			Protected: false, // TODO: Implement protection logic
		}
		
		backups = append(backups, backup)
	}
	
	return backups, nil
}

// deleteBackup deletes a backup
func (rm *RetentionManager) deleteBackup(backup *BackupItem) error {
	// Remove backup directory
	if err := os.RemoveAll(backup.Path); err != nil {
		return fmt.Errorf("failed to remove backup directory: %w", err)
	}
	
	// Update backup chain if necessary
	// TODO: Implement chain update logic
	
	return nil
}

// createDefaultPolicy creates the default GFS retention policy
func (rm *RetentionManager) createDefaultPolicy() error {
	defaultPolicy := &RetentionPolicy{
		ID:          rm.config.DefaultPolicy,
		Name:        "Default GFS Policy",
		Description: "Standard Grandfather-Father-Son retention policy",
		Rules: &RetentionRules{
			MaxAge:   30 * 24 * time.Hour, // 30 days
			MinCount: 1,                    // Always keep at least one backup
		},
		GFSConfig: &GFSConfig{
			Enabled:          rm.config.EnableGFS,
			DailyRetention:   DefaultDailyRetention,
			WeeklyRetention:  DefaultWeeklyRetention,
			MonthlyRetention: DefaultMonthlyRetention,
			YearlyRetention:  DefaultYearlyRetention,
			WeekStartDay:     1, // Monday
			MonthStartDay:    1, // First of month
		},
		Metadata:  make(map[string]string),
		Enabled:   true,
	}
	
	return rm.CreatePolicy(defaultPolicy)
}

// loadPolicies loads retention policies from disk
func (rm *RetentionManager) loadPolicies() error {
	policiesDir := filepath.Join(rm.config.BasePath, "policies")
	if _, err := os.Stat(policiesDir); os.IsNotExist(err) {
		return nil // No policies to load
	}
	
	files, err := filepath.Glob(filepath.Join(policiesDir, "*.json"))
	if err != nil {
		return fmt.Errorf("failed to glob policy files: %w", err)
	}
	
	for _, file := range files {
		var policy RetentionPolicy
		if err := loadJSON(file, &policy); err != nil {
			continue // Skip invalid policy files
		}
		rm.policies[policy.ID] = &policy
	}
	
	return nil
}

// savePolicy saves a retention policy to disk
func (rm *RetentionManager) savePolicy(policy *RetentionPolicy) error {
	policiesDir := filepath.Join(rm.config.BasePath, "policies")
	if err := os.MkdirAll(policiesDir, 0755); err != nil {
		return fmt.Errorf("failed to create policies directory: %w", err)
	}
	
	policyPath := filepath.Join(policiesDir, policy.ID+".json")
	return saveJSON(policyPath, policy)
}

// GetRetentionJob returns a retention job by ID
func (rm *RetentionManager) GetRetentionJob(jobID string) (*RetentionJob, error) {
	rm.scheduler.mutex.RLock()
	defer rm.scheduler.mutex.RUnlock()
	
	job, exists := rm.scheduler.jobs[jobID]
	if !exists {
		return nil, fmt.Errorf("retention job %s not found", jobID)
	}
	
	// Return a copy
	jobCopy := *job
	return &jobCopy, nil
}

// ListRetentionJobs returns all retention jobs
func (rm *RetentionManager) ListRetentionJobs() []*RetentionJob {
	rm.scheduler.mutex.RLock()
	defer rm.scheduler.mutex.RUnlock()
	
	jobs := make([]*RetentionJob, 0, len(rm.scheduler.jobs))
	for _, job := range rm.scheduler.jobs {
		jobCopy := *job
		jobs = append(jobs, &jobCopy)
	}
	
	return jobs
}

// RetentionScheduler methods

// Start starts the retention scheduler
func (rs *RetentionScheduler) Start(ctx context.Context, rm *RetentionManager) error {
	rs.ticker = time.NewTicker(rs.config.CleanupInterval)
	
	go func() {
		defer close(rs.doneCh)
		
		for {
			select {
			case <-ctx.Done():
				return
			case <-rs.stopCh:
				return
			case <-rs.ticker.C:
				// TODO: Implement scheduled retention cleanup
				// This would iterate through all VMs and apply their retention policies
			}
		}
	}()
	
	return nil
}

// Stop stops the retention scheduler
func (rs *RetentionScheduler) Stop() error {
	close(rs.stopCh)
	if rs.ticker != nil {
		rs.ticker.Stop()
	}
	<-rs.doneCh
	return nil
}