package snapshot

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// SnapshotType represents the type of snapshot
type SnapshotType string

const (
	// MemorySnapshot is a snapshot that includes memory state
	MemorySnapshot SnapshotType = "memory"

	// DiskSnapshot is a snapshot of disks only
	DiskSnapshot SnapshotType = "disk"

	// ApplicationConsistentSnapshot is a snapshot with application consistency
	ApplicationConsistentSnapshot SnapshotType = "application_consistent"
)

// SnapshotState represents the state of a snapshot
type SnapshotState string

const (
	// CreatingState means the snapshot is being created
	CreatingState SnapshotState = "creating"

	// AvailableState means the snapshot is available for use
	AvailableState SnapshotState = "available"

	// ErrorState means there was an error creating the snapshot
	ErrorState SnapshotState = "error"

	// DeletingState means the snapshot is being deleted
	DeletingState SnapshotState = "deleting"

	// RestoringState means the snapshot is being restored
	RestoringState SnapshotState = "restoring"
)

// Snapshot represents a VM snapshot
type Snapshot struct {
	// ID is the unique identifier of the snapshot
	ID string `json:"id"`

	// Name is the name of the snapshot
	Name string `json:"name"`

	// Description describes the snapshot
	Description string `json:"description"`

	// VMID is the ID of the VM this snapshot belongs to
	VMID string `json:"vm_id"`

	// Type is the type of snapshot
	Type SnapshotType `json:"type"`

	// State is the state of the snapshot
	State SnapshotState `json:"state"`

	// CreatedAt is when the snapshot was created
	CreatedAt time.Time `json:"created_at"`

	// CreatedBy is the ID of the user who created this snapshot
	CreatedBy string `json:"created_by"`

	// ExpiresAt is when the snapshot expires (optional)
	ExpiresAt time.Time `json:"expires_at,omitempty"`

	// Size is the size of the snapshot in bytes
	Size int64 `json:"size"`

	// StorageLocation is where the snapshot is stored
	StorageLocation string `json:"storage_location"`

	// ParentSnapshotID is the ID of the parent snapshot (if incremental)
	ParentSnapshotID string `json:"parent_snapshot_id,omitempty"`

	// Metadata is additional metadata for the snapshot
	Metadata map[string]string `json:"metadata,omitempty"`

	// TenantID is the ID of the tenant this snapshot belongs to
	TenantID string `json:"tenant_id"`

	// Tags are tags for the snapshot
	Tags []string `json:"tags,omitempty"`

	// LockStatus indicates if this snapshot is locked (can't be deleted)
	LockStatus bool `json:"lock_status"`

	// Error is the error message if the snapshot is in error state
	Error string `json:"error,omitempty"`
}

// SnapshotRestoreRequest represents a request to restore from a snapshot
type SnapshotRestoreRequest struct {
	// ID is the unique identifier of the request
	ID string `json:"id"`

	// SnapshotID is the ID of the snapshot to restore from
	SnapshotID string `json:"snapshot_id"`

	// VMID is the ID of the VM to restore
	VMID string `json:"vm_id"`

	// TargetVMID is the ID of the VM to restore to (if different from VMID)
	TargetVMID string `json:"target_vm_id,omitempty"`

	// CreatedAt is when the request was created
	CreatedAt time.Time `json:"created_at"`

	// CreatedBy is the ID of the user who created this request
	CreatedBy string `json:"created_by"`

	// Status is the status of the restore request
	Status string `json:"status"`

	// StartedAt is when the restore started
	StartedAt time.Time `json:"started_at,omitempty"`

	// CompletedAt is when the restore completed
	CompletedAt time.Time `json:"completed_at,omitempty"`

	// TenantID is the ID of the tenant this request belongs to
	TenantID string `json:"tenant_id"`

	// Error is the error message if the restore failed
	Error string `json:"error,omitempty"`

	// Options are options for the restore
	Options *SnapshotRestoreOptions `json:"options,omitempty"`
}

// SnapshotRestoreOptions represents options for a snapshot restore
type SnapshotRestoreOptions struct {
	// RestoreMemory indicates if memory should be restored
	RestoreMemory bool `json:"restore_memory"`

	// PowerOnAfterRestore indicates if the VM should be powered on after restore
	PowerOnAfterRestore bool `json:"power_on_after_restore"`

	// TargetResourcePool is the resource pool to restore to
	TargetResourcePool string `json:"target_resource_pool,omitempty"`

	// TargetDatastore is the datastore to restore to
	TargetDatastore string `json:"target_datastore,omitempty"`

	// NetworkMapping maps source networks to target networks
	NetworkMapping map[string]string `json:"network_mapping,omitempty"`

	// AdditionalOptions are additional options for the restore
	AdditionalOptions map[string]interface{} `json:"additional_options,omitempty"`
}

// ScheduledSnapshot represents a scheduled snapshot configuration
type ScheduledSnapshot struct {
	// ID is the unique identifier of the scheduled snapshot
	ID string `json:"id"`

	// Name is the name of the scheduled snapshot
	Name string `json:"name"`

	// Description describes the scheduled snapshot
	Description string `json:"description"`

	// VMID is the ID of the VM to snapshot
	VMID string `json:"vm_id"`

	// Type is the type of snapshot
	Type SnapshotType `json:"type"`

	// Schedule is the schedule for taking snapshots
	Schedule *SnapshotSchedule `json:"schedule"`

	// Retention is the retention policy for snapshots
	Retention *SnapshotRetention `json:"retention"`

	// Enabled indicates if the scheduled snapshot is enabled
	Enabled bool `json:"enabled"`

	// CreatedAt is when the scheduled snapshot was created
	CreatedAt time.Time `json:"created_at"`

	// CreatedBy is the ID of the user who created this scheduled snapshot
	CreatedBy string `json:"created_by"`

	// LastRunAt is when the scheduled snapshot last ran
	LastRunAt time.Time `json:"last_run_at,omitempty"`

	// LastRunStatus is the status of the last run
	LastRunStatus string `json:"last_run_status,omitempty"`

	// NextRunAt is when the scheduled snapshot will next run
	NextRunAt time.Time `json:"next_run_at,omitempty"`

	// TenantID is the ID of the tenant this scheduled snapshot belongs to
	TenantID string `json:"tenant_id"`

	// Tags are tags for the scheduled snapshot
	Tags []string `json:"tags,omitempty"`

	// QuiesceVM indicates if the VM should be quiesced before snapshot
	QuiesceVM bool `json:"quiesce_vm"`

	// MemorySnapshot indicates if memory should be included in the snapshot
	MemorySnapshot bool `json:"memory_snapshot"`

	// Options are additional options for the snapshot
	Options map[string]interface{} `json:"options,omitempty"`
}

// SnapshotSchedule represents a schedule for taking snapshots
type SnapshotSchedule struct {
	// Type is the type of schedule (cron, interval, etc.)
	Type string `json:"type"`

	// Expression is the schedule expression
	Expression string `json:"expression"`

	// TimeZone is the time zone for the schedule
	TimeZone string `json:"time_zone,omitempty"`

	// StartTime is when the schedule should start
	StartTime time.Time `json:"start_time,omitempty"`

	// EndTime is when the schedule should end
	EndTime time.Time `json:"end_time,omitempty"`
}

// SnapshotRetention represents a retention policy for snapshots
type SnapshotRetention struct {
	// MaxSnapshots is the maximum number of snapshots to keep
	MaxSnapshots int `json:"max_snapshots"`

	// MaxAge is the maximum age of snapshots to keep
	MaxAge time.Duration `json:"max_age"`

	// KeepDaily is the number of daily snapshots to keep
	KeepDaily int `json:"keep_daily"`

	// KeepWeekly is the number of weekly snapshots to keep
	KeepWeekly int `json:"keep_weekly"`

	// KeepMonthly is the number of monthly snapshots to keep
	KeepMonthly int `json:"keep_monthly"`

	// RetentionType is the type of retention policy (simple, gfs, etc.)
	RetentionType string `json:"retention_type"`
}

// SnapshotConsistencyGroup represents a group of snapshots that are consistent with each other
type SnapshotConsistencyGroup struct {
	// ID is the unique identifier of the consistency group
	ID string `json:"id"`

	// Name is the name of the consistency group
	Name string `json:"name"`

	// Description describes the consistency group
	Description string `json:"description"`

	// SnapshotIDs are the IDs of the snapshots in this group
	SnapshotIDs []string `json:"snapshot_ids"`

	// VMIDs are the IDs of the VMs in this group
	VMIDs []string `json:"vm_ids"`

	// CreatedAt is when the consistency group was created
	CreatedAt time.Time `json:"created_at"`

	// CreatedBy is the ID of the user who created this consistency group
	CreatedBy string `json:"created_by"`

	// TenantID is the ID of the tenant this consistency group belongs to
	TenantID string `json:"tenant_id"`

	// Tags are tags for the consistency group
	Tags []string `json:"tags,omitempty"`
}

// SnapshotProvider defines the interface for snapshot providers
type SnapshotProvider interface {
	// ID returns the provider ID
	ID() string

	// Name returns the provider name
	Name() string

	// CreateSnapshot creates a new snapshot
	CreateSnapshot(vm string, snapshotType SnapshotType, options map[string]interface{}) (*Snapshot, error)

	// DeleteSnapshot deletes a snapshot
	DeleteSnapshot(snapshotID string) error

	// RestoreSnapshot restores a snapshot
	RestoreSnapshot(request *SnapshotRestoreRequest) error

	// GetSnapshot gets a snapshot by ID
	GetSnapshot(snapshotID string) (*Snapshot, error)

	// ListSnapshots lists snapshots
	ListSnapshots(vmID string, filter map[string]interface{}) ([]*Snapshot, error)
}

// SnapshotManager manages VM snapshots
type SnapshotManager struct {
	// providers is a map of provider ID to provider
	providers map[string]SnapshotProvider

	// snapshots is a map of snapshot ID to snapshot
	snapshots map[string]*Snapshot

	// vmSnapshots is a map of VM ID to snapshot IDs
	vmSnapshots map[string][]string

	// tenantSnapshots is a map of tenant ID to snapshot IDs
	tenantSnapshots map[string][]string

	// scheduledSnapshots is a map of scheduled snapshot ID to scheduled snapshot
	scheduledSnapshots map[string]*ScheduledSnapshot

	// vmScheduledSnapshots is a map of VM ID to scheduled snapshot IDs
	vmScheduledSnapshots map[string][]string

	// consistencyGroups is a map of consistency group ID to consistency group
	consistencyGroups map[string]*SnapshotConsistencyGroup

	// restoreRequests is a map of restore request ID to restore request
	restoreRequests map[string]*SnapshotRestoreRequest

	// scheduler is the scheduler for scheduled snapshots
	scheduler *SnapshotScheduler

	// mutex protects the maps
	mutex sync.RWMutex
}

// NewSnapshotManager creates a new snapshot manager
func NewSnapshotManager() *SnapshotManager {
	manager := &SnapshotManager{
		providers:            make(map[string]SnapshotProvider),
		snapshots:            make(map[string]*Snapshot),
		vmSnapshots:          make(map[string][]string),
		tenantSnapshots:      make(map[string][]string),
		scheduledSnapshots:   make(map[string]*ScheduledSnapshot),
		vmScheduledSnapshots: make(map[string][]string),
		consistencyGroups:    make(map[string]*SnapshotConsistencyGroup),
		restoreRequests:      make(map[string]*SnapshotRestoreRequest),
	}

	// Create scheduler
	manager.scheduler = NewSnapshotScheduler(manager)

	return manager
}

// Start starts the snapshot manager
func (m *SnapshotManager) Start() error {
	// Start scheduler
	return m.scheduler.Start()
}

// Stop stops the snapshot manager
func (m *SnapshotManager) Stop() error {
	// Stop scheduler
	return m.scheduler.Stop()
}

// RegisterProvider registers a snapshot provider
func (m *SnapshotManager) RegisterProvider(provider SnapshotProvider) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if provider already exists
	if _, exists := m.providers[provider.ID()]; exists {
		return fmt.Errorf("provider with ID %s already exists", provider.ID())
	}

	// Add provider
	m.providers[provider.ID()] = provider

	return nil
}

// GetProvider gets a provider by ID
func (m *SnapshotManager) GetProvider(providerID string) (SnapshotProvider, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if provider exists
	provider, exists := m.providers[providerID]
	if !exists {
		return nil, fmt.Errorf("provider with ID %s does not exist", providerID)
	}

	return provider, nil
}

// CreateSnapshot creates a new snapshot
func (m *SnapshotManager) CreateSnapshot(vmID string, name string, description string, snapshotType SnapshotType, options map[string]interface{}, tenantID string, createdBy string) (*Snapshot, error) {
	// Select a provider
	var provider SnapshotProvider
	var err error

	// For now, select the first provider
	m.mutex.RLock()
	for _, p := range m.providers {
		provider = p
		break
	}
	m.mutex.RUnlock()

	if provider == nil {
		return nil, errors.New("no snapshot provider available")
	}

	// Create snapshot
	snapshot, err := provider.CreateSnapshot(vmID, snapshotType, options)
	if err != nil {
		return nil, err
	}

	// Update snapshot with additional info
	snapshot.Name = name
	snapshot.Description = description
	snapshot.CreatedBy = createdBy
	snapshot.TenantID = tenantID
	snapshot.CreatedAt = time.Now()

	// Add snapshot to maps
	m.mutex.Lock()
	m.snapshots[snapshot.ID] = snapshot
	m.vmSnapshots[vmID] = append(m.vmSnapshots[vmID], snapshot.ID)
	m.tenantSnapshots[tenantID] = append(m.tenantSnapshots[tenantID], snapshot.ID)
	m.mutex.Unlock()

	return snapshot, nil
}

// DeleteSnapshot deletes a snapshot
func (m *SnapshotManager) DeleteSnapshot(snapshotID string) error {
	// Get snapshot
	m.mutex.RLock()
	snapshot, exists := m.snapshots[snapshotID]
	if !exists {
		m.mutex.RUnlock()
		return fmt.Errorf("snapshot with ID %s does not exist", snapshotID)
	}

	// Check if snapshot is locked
	if snapshot.LockStatus {
		m.mutex.RUnlock()
		return errors.New("snapshot is locked and cannot be deleted")
	}

	// Get provider for VM type
	var provider SnapshotProvider
	for _, p := range m.providers {
		provider = p
		break
	}
	m.mutex.RUnlock()

	if provider == nil {
		return errors.New("no snapshot provider available")
	}

	// Delete snapshot
	err := provider.DeleteSnapshot(snapshotID)
	if err != nil {
		return err
	}

	// Remove snapshot from maps
	m.mutex.Lock()
	delete(m.snapshots, snapshotID)

	// Remove from VM snapshots
	vmSnaps := m.vmSnapshots[snapshot.VMID]
	for i, id := range vmSnaps {
		if id == snapshotID {
			m.vmSnapshots[snapshot.VMID] = append(vmSnaps[:i], vmSnaps[i+1:]...)
			break
		}
	}

	// Remove from tenant snapshots
	tenantSnaps := m.tenantSnapshots[snapshot.TenantID]
	for i, id := range tenantSnaps {
		if id == snapshotID {
			m.tenantSnapshots[snapshot.TenantID] = append(tenantSnaps[:i], tenantSnaps[i+1:]...)
			break
		}
	}
	m.mutex.Unlock()

	return nil
}

// GetSnapshot gets a snapshot by ID
func (m *SnapshotManager) GetSnapshot(snapshotID string) (*Snapshot, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if snapshot exists in local cache
	snapshot, exists := m.snapshots[snapshotID]
	if !exists {
		return nil, fmt.Errorf("snapshot with ID %s does not exist", snapshotID)
	}

	return snapshot, nil
}

// ListSnapshots lists snapshots
func (m *SnapshotManager) ListSnapshots(vmID string, tenantID string) ([]*Snapshot, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var snapshots []*Snapshot

	if vmID != "" {
		// List snapshots for VM
		snapshotIDs, exists := m.vmSnapshots[vmID]
		if !exists {
			return []*Snapshot{}, nil
		}

		snapshots = make([]*Snapshot, 0, len(snapshotIDs))
		for _, id := range snapshotIDs {
			if s, exists := m.snapshots[id]; exists {
				// Filter by tenant if specified
				if tenantID == "" || s.TenantID == tenantID {
					snapshots = append(snapshots, s)
				}
			}
		}
	} else if tenantID != "" {
		// List snapshots for tenant
		snapshotIDs, exists := m.tenantSnapshots[tenantID]
		if !exists {
			return []*Snapshot{}, nil
		}

		snapshots = make([]*Snapshot, 0, len(snapshotIDs))
		for _, id := range snapshotIDs {
			if s, exists := m.snapshots[id]; exists {
				snapshots = append(snapshots, s)
			}
		}
	} else {
		// List all snapshots
		snapshots = make([]*Snapshot, 0, len(m.snapshots))
		for _, s := range m.snapshots {
			snapshots = append(snapshots, s)
		}
	}

	return snapshots, nil
}

// CreateRestoreRequest creates a request to restore from a snapshot
func (m *SnapshotManager) CreateRestoreRequest(request *SnapshotRestoreRequest) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if request ID already exists
	if _, exists := m.restoreRequests[request.ID]; exists {
		return fmt.Errorf("restore request with ID %s already exists", request.ID)
	}

	// Check if snapshot exists
	if _, exists := m.snapshots[request.SnapshotID]; !exists {
		return fmt.Errorf("snapshot with ID %s does not exist", request.SnapshotID)
	}

	// Set created time and status
	request.CreatedAt = time.Now()
	request.Status = "pending"

	// Add request
	m.restoreRequests[request.ID] = request

	return nil
}

// StartRestore starts the restore process
func (m *SnapshotManager) StartRestore(requestID string) error {
	m.mutex.Lock()
	request, exists := m.restoreRequests[requestID]
	if !exists {
		m.mutex.Unlock()
		return fmt.Errorf("restore request with ID %s does not exist", requestID)
	}

	if request.Status != "pending" {
		m.mutex.Unlock()
		return fmt.Errorf("restore request %s is not in pending state", requestID)
	}

	// Update request
	request.Status = "restoring"
	request.StartedAt = time.Now()
	m.mutex.Unlock()

	// Get provider
	var provider SnapshotProvider
	m.mutex.RLock()
	for _, p := range m.providers {
		provider = p
		break
	}
	m.mutex.RUnlock()

	if provider == nil {
		return errors.New("no snapshot provider available")
	}

	// Start restore in a goroutine
	go func() {
		err := provider.RestoreSnapshot(request)
		m.mutex.Lock()
		defer m.mutex.Unlock()

		// Update request
		request.CompletedAt = time.Now()
		if err != nil {
			request.Status = "failed"
			request.Error = err.Error()
		} else {
			request.Status = "completed"
		}
	}()

	return nil
}

// GetRestoreRequest gets a restore request by ID
func (m *SnapshotManager) GetRestoreRequest(requestID string) (*SnapshotRestoreRequest, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if request exists
	request, exists := m.restoreRequests[requestID]
	if !exists {
		return nil, fmt.Errorf("restore request with ID %s does not exist", requestID)
	}

	return request, nil
}

// ListRestoreRequests lists restore requests
func (m *SnapshotManager) ListRestoreRequests(tenantID string) ([]*SnapshotRestoreRequest, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var requests []*SnapshotRestoreRequest

	if tenantID != "" {
		// List requests for tenant
		requests = make([]*SnapshotRestoreRequest, 0)
		for _, r := range m.restoreRequests {
			if r.TenantID == tenantID {
				requests = append(requests, r)
			}
		}
	} else {
		// List all requests
		requests = make([]*SnapshotRestoreRequest, 0, len(m.restoreRequests))
		for _, r := range m.restoreRequests {
			requests = append(requests, r)
		}
	}

	return requests, nil
}

// CreateScheduledSnapshot creates a scheduled snapshot
func (m *SnapshotManager) CreateScheduledSnapshot(scheduled *ScheduledSnapshot) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if scheduled snapshot ID already exists
	if _, exists := m.scheduledSnapshots[scheduled.ID]; exists {
		return fmt.Errorf("scheduled snapshot with ID %s already exists", scheduled.ID)
	}

	// Set created time
	scheduled.CreatedAt = time.Now()

	// Add scheduled snapshot
	m.scheduledSnapshots[scheduled.ID] = scheduled
	m.vmScheduledSnapshots[scheduled.VMID] = append(m.vmScheduledSnapshots[scheduled.VMID], scheduled.ID)

	// Schedule it if enabled
	if scheduled.Enabled {
		m.scheduler.ScheduleSnapshot(scheduled)
	}

	return nil
}

// GetScheduledSnapshot gets a scheduled snapshot by ID
func (m *SnapshotManager) GetScheduledSnapshot(scheduledID string) (*ScheduledSnapshot, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if scheduled snapshot exists
	scheduled, exists := m.scheduledSnapshots[scheduledID]
	if !exists {
		return nil, fmt.Errorf("scheduled snapshot with ID %s does not exist", scheduledID)
	}

	return scheduled, nil
}

// UpdateScheduledSnapshot updates a scheduled snapshot
func (m *SnapshotManager) UpdateScheduledSnapshot(scheduled *ScheduledSnapshot) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if scheduled snapshot exists
	existing, exists := m.scheduledSnapshots[scheduled.ID]
	if !exists {
		return fmt.Errorf("scheduled snapshot with ID %s does not exist", scheduled.ID)
	}

	// Preserve creation time and last run info
	scheduled.CreatedAt = existing.CreatedAt
	scheduled.LastRunAt = existing.LastRunAt
	scheduled.LastRunStatus = existing.LastRunStatus

	// Update scheduled snapshot
	m.scheduledSnapshots[scheduled.ID] = scheduled

	// Update VM mapping if VM changed
	if existing.VMID != scheduled.VMID {
		// Remove from old VM
		vmScheduled := m.vmScheduledSnapshots[existing.VMID]
		for i, id := range vmScheduled {
			if id == scheduled.ID {
				m.vmScheduledSnapshots[existing.VMID] = append(vmScheduled[:i], vmScheduled[i+1:]...)
				break
			}
		}

		// Add to new VM
		m.vmScheduledSnapshots[scheduled.VMID] = append(m.vmScheduledSnapshots[scheduled.VMID], scheduled.ID)
	}

	// Reschedule if enabled changed
	if scheduled.Enabled && !existing.Enabled {
		m.scheduler.ScheduleSnapshot(scheduled)
	} else if !scheduled.Enabled && existing.Enabled {
		m.scheduler.UnscheduleSnapshot(scheduled.ID)
	} else if scheduled.Enabled && existing.Enabled {
		// Schedule info might have changed, reschedule
		m.scheduler.ScheduleSnapshot(scheduled)
	}

	return nil
}

// DeleteScheduledSnapshot deletes a scheduled snapshot
func (m *SnapshotManager) DeleteScheduledSnapshot(scheduledID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if scheduled snapshot exists
	scheduled, exists := m.scheduledSnapshots[scheduledID]
	if !exists {
		return fmt.Errorf("scheduled snapshot with ID %s does not exist", scheduledID)
	}

	// Remove scheduled snapshot
	delete(m.scheduledSnapshots, scheduledID)

	// Remove from VM
	vmScheduled := m.vmScheduledSnapshots[scheduled.VMID]
	for i, id := range vmScheduled {
		if id == scheduledID {
			m.vmScheduledSnapshots[scheduled.VMID] = append(vmScheduled[:i], vmScheduled[i+1:]...)
			break
		}
	}

	// Unschedule it
	m.scheduler.UnscheduleSnapshot(scheduledID)

	return nil
}

// ListScheduledSnapshots lists scheduled snapshots
func (m *SnapshotManager) ListScheduledSnapshots(vmID string, tenantID string) ([]*ScheduledSnapshot, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var scheduled []*ScheduledSnapshot

	if vmID != "" {
		// List scheduled snapshots for VM
		scheduledIDs, exists := m.vmScheduledSnapshots[vmID]
		if !exists {
			return []*ScheduledSnapshot{}, nil
		}

		scheduled = make([]*ScheduledSnapshot, 0, len(scheduledIDs))
		for _, id := range scheduledIDs {
			if s, exists := m.scheduledSnapshots[id]; exists {
				// Filter by tenant if specified
				if tenantID == "" || s.TenantID == tenantID {
					scheduled = append(scheduled, s)
				}
			}
		}
	} else if tenantID != "" {
		// List scheduled snapshots for tenant
		scheduled = make([]*ScheduledSnapshot, 0)
		for _, s := range m.scheduledSnapshots {
			if s.TenantID == tenantID {
				scheduled = append(scheduled, s)
			}
		}
	} else {
		// List all scheduled snapshots
		scheduled = make([]*ScheduledSnapshot, 0, len(m.scheduledSnapshots))
		for _, s := range m.scheduledSnapshots {
			scheduled = append(scheduled, s)
		}
	}

	return scheduled, nil
}

// ExecuteScheduledSnapshot executes a scheduled snapshot
func (m *SnapshotManager) ExecuteScheduledSnapshot(scheduledID string) (*Snapshot, error) {
	// Get scheduled snapshot
	m.mutex.RLock()
	scheduled, exists := m.scheduledSnapshots[scheduledID]
	if !exists {
		m.mutex.RUnlock()
		return nil, fmt.Errorf("scheduled snapshot with ID %s does not exist", scheduledID)
	}
	m.mutex.RUnlock()

	// Create snapshot
	options := make(map[string]interface{})
	if scheduled.Options != nil {
		options = scheduled.Options
	}
	if scheduled.QuiesceVM {
		options["quiesce"] = true
	}
	if scheduled.MemorySnapshot {
		options["memory"] = true
	}

	// Generate name if not specified
	name := fmt.Sprintf("%s-%s", scheduled.Name, time.Now().Format("20060102-150405"))

	// Create snapshot
	snapshot, err := m.CreateSnapshot(
		scheduled.VMID,
		name,
		fmt.Sprintf("Scheduled snapshot: %s", scheduled.Description),
		scheduled.Type,
		options,
		scheduled.TenantID,
		"system",
	)

	// Update scheduled snapshot with last run info
	m.mutex.Lock()
	defer m.mutex.Unlock()

	scheduled.LastRunAt = time.Now()
	if err != nil {
		scheduled.LastRunStatus = "failed"
	} else {
		scheduled.LastRunStatus = "completed"
		// Update next run time
		if scheduled.Schedule != nil {
			scheduled.NextRunAt = calculateNextSnapshotTime(scheduled.Schedule)
		}
	}

	// Apply retention policy if needed
	if scheduled.Retention != nil && err == nil {
		m.applyRetentionPolicy(scheduled)
	}

	return snapshot, err
}

// CreateConsistencyGroup creates a consistency group
func (m *SnapshotManager) CreateConsistencyGroup(group *SnapshotConsistencyGroup) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if group ID already exists
	if _, exists := m.consistencyGroups[group.ID]; exists {
		return fmt.Errorf("consistency group with ID %s already exists", group.ID)
	}

	// Set created time
	group.CreatedAt = time.Now()

	// Add group
	m.consistencyGroups[group.ID] = group

	return nil
}

// GetConsistencyGroup gets a consistency group by ID
func (m *SnapshotManager) GetConsistencyGroup(groupID string) (*SnapshotConsistencyGroup, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Check if group exists
	group, exists := m.consistencyGroups[groupID]
	if !exists {
		return nil, fmt.Errorf("consistency group with ID %s does not exist", groupID)
	}

	return group, nil
}

// ListConsistencyGroups lists consistency groups
func (m *SnapshotManager) ListConsistencyGroups(tenantID string) ([]*SnapshotConsistencyGroup, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var groups []*SnapshotConsistencyGroup

	if tenantID != "" {
		// List groups for tenant
		groups = make([]*SnapshotConsistencyGroup, 0)
		for _, g := range m.consistencyGroups {
			if g.TenantID == tenantID {
				groups = append(groups, g)
			}
		}
	} else {
		// List all groups
		groups = make([]*SnapshotConsistencyGroup, 0, len(m.consistencyGroups))
		for _, g := range m.consistencyGroups {
			groups = append(groups, g)
		}
	}

	return groups, nil
}

// DeleteConsistencyGroup deletes a consistency group
func (m *SnapshotManager) DeleteConsistencyGroup(groupID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if group exists
	if _, exists := m.consistencyGroups[groupID]; !exists {
		return fmt.Errorf("consistency group with ID %s does not exist", groupID)
	}

	// Remove group
	delete(m.consistencyGroups, groupID)

	return nil
}

// CreateConsistencyGroupSnapshot creates a snapshot for all VMs in a consistency group
func (m *SnapshotManager) CreateConsistencyGroupSnapshot(groupID string, createdBy string) ([]*Snapshot, error) {
	// Get group
	m.mutex.RLock()
	group, exists := m.consistencyGroups[groupID]
	if !exists {
		m.mutex.RUnlock()
		return nil, fmt.Errorf("consistency group with ID %s does not exist", groupID)
	}
	vmIDs := group.VMIDs
	tenantID := group.TenantID
	m.mutex.RUnlock()

	// Create snapshots for all VMs
	snapshots := make([]*Snapshot, 0, len(vmIDs))
	for _, vmID := range vmIDs {
		// Generate name
		name := fmt.Sprintf("cg-%s-%s", groupID, time.Now().Format("20060102-150405"))

		// Create snapshot
		snapshot, err := m.CreateSnapshot(
			vmID,
			name,
			fmt.Sprintf("Consistency group snapshot: %s", group.Name),
			ApplicationConsistentSnapshot,
			map[string]interface{}{"consistency_group": groupID},
			tenantID,
			createdBy,
		)
		if err != nil {
			// TODO: Handle error, potentially roll back already created snapshots
			return snapshots, err
		}

		snapshots = append(snapshots, snapshot)
	}

	// Update group with snapshot IDs
	m.mutex.Lock()
	snapshotIDs := make([]string, len(snapshots))
	for i, snapshot := range snapshots {
		snapshotIDs[i] = snapshot.ID
	}
	group.SnapshotIDs = append(group.SnapshotIDs, snapshotIDs...)
	m.mutex.Unlock()

	return snapshots, nil
}

// applyRetentionPolicy applies a retention policy to snapshots
func (m *SnapshotManager) applyRetentionPolicy(scheduled *ScheduledSnapshot) {
	// Get snapshots for VM
	snapshots, err := m.ListSnapshots(scheduled.VMID, scheduled.TenantID)
	if err != nil {
		// Log error
		return
	}

	// Filter to snapshots from this scheduled job
	jobSnapshots := make([]*Snapshot, 0)
	for _, snapshot := range snapshots {
		if strings.HasPrefix(snapshot.Name, scheduled.Name+"-") {
			jobSnapshots = append(jobSnapshots, snapshot)
		}
	}

	// Sort snapshots by creation time (newest first)
	sort.Slice(jobSnapshots, func(i, j int) bool {
		return jobSnapshots[i].CreatedAt.After(jobSnapshots[j].CreatedAt)
	})

	// Apply max snapshots retention
	if scheduled.Retention.MaxSnapshots > 0 && len(jobSnapshots) > scheduled.Retention.MaxSnapshots {
		// Delete oldest snapshots
		for i := scheduled.Retention.MaxSnapshots; i < len(jobSnapshots); i++ {
			m.DeleteSnapshot(jobSnapshots[i].ID)
		}
	}

	// Apply max age retention
	if scheduled.Retention.MaxAge > 0 {
		cutoff := time.Now().Add(-scheduled.Retention.MaxAge)
		for _, snapshot := range jobSnapshots {
			if snapshot.CreatedAt.Before(cutoff) {
				m.DeleteSnapshot(snapshot.ID)
			}
		}
	}

	// More complex retention policies like GFS (Grandfather-Father-Son)
	// would be implemented here
}

// SnapshotScheduler schedules and executes snapshot jobs
type SnapshotScheduler struct {
	// manager is the snapshot manager
	manager *SnapshotManager

	// scheduledJobs maps scheduled snapshot ID to their next run time
	scheduledJobs map[string]time.Time

	// mutex protects scheduledJobs
	mutex sync.RWMutex

	// stopChan is used to signal the scheduler to stop
	stopChan chan struct{}

	// wg is used to wait for the scheduler to stop
	wg sync.WaitGroup
}

// NewSnapshotScheduler creates a new snapshot scheduler
func NewSnapshotScheduler(manager *SnapshotManager) *SnapshotScheduler {
	return &SnapshotScheduler{
		manager:       manager,
		scheduledJobs: make(map[string]time.Time),
		stopChan:      make(chan struct{}),
	}
}

// Start starts the scheduler
func (s *SnapshotScheduler) Start() error {
	s.wg.Add(1)
	go s.run()
	return nil
}

// Stop stops the scheduler
func (s *SnapshotScheduler) Stop() error {
	close(s.stopChan)
	s.wg.Wait()
	return nil
}

// ScheduleSnapshot schedules a snapshot
func (s *SnapshotScheduler) ScheduleSnapshot(scheduled *ScheduledSnapshot) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	// Calculate next run time
	nextRun := calculateNextSnapshotTime(scheduled.Schedule)
	s.scheduledJobs[scheduled.ID] = nextRun

	// Update next run time in scheduled snapshot
	scheduled.NextRunAt = nextRun
}

// UnscheduleSnapshot unschedules a snapshot
func (s *SnapshotScheduler) UnscheduleSnapshot(scheduledID string) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	delete(s.scheduledJobs, scheduledID)
}

// run is the main scheduler loop
func (s *SnapshotScheduler) run() {
	defer s.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-s.stopChan:
			return
		case <-ticker.C:
			s.checkAndRunDueJobs()
		}
	}
}

// checkAndRunDueJobs checks for and runs due jobs
func (s *SnapshotScheduler) checkAndRunDueJobs() {
	now := time.Now()
	jobsToRun := make([]string, 0)

	// Find jobs due to run
	s.mutex.Lock()
	for id, nextRun := range s.scheduledJobs {
		if nextRun.Before(now) || nextRun.Equal(now) {
			jobsToRun = append(jobsToRun, id)
		}
	}
	s.mutex.Unlock()

	// Run due jobs
	for _, id := range jobsToRun {
		go func(scheduledID string) {
			_, err := s.manager.ExecuteScheduledSnapshot(scheduledID)
			if err != nil {
				// Log error but continue
				fmt.Printf("Error executing scheduled snapshot %s: %v\n", scheduledID, err)
			}
		}(id)
	}
}

// calculateNextSnapshotTime calculates the next run time for a schedule
func calculateNextSnapshotTime(schedule *SnapshotSchedule) time.Time {
	// This is a simplified implementation
	// A real implementation would parse the schedule expression
	// and calculate the exact next run time

	// For now, just return a time 24 hours in the future
	return time.Now().Add(24 * time.Hour)
}
