package vm

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	// Assuming NodeManager interface is defined elsewhere (e.g., node package or hypervisor package)
	// Assuming MigrationStorage and MigrationExecutor interfaces are defined elsewhere
)

// MigrationManagerImpl implements the MigrationManager interface (defined in vm_types.go)
type MigrationManagerImpl struct {
	vmManager        *VMManager            // Use concrete type pointer
	nodeManager      NodeManager           // Use interface type defined elsewhere
	storage          MigrationStorage      // Use interface type defined elsewhere
	executor         MigrationExecutor     // Use interface type defined elsewhere
	activeMigrations map[string]*Migration // Assuming Migration type is defined elsewhere
	mu               sync.RWMutex
	logger           *logrus.Logger
}

// NewMigrationManager creates a new MigrationManager
// Note: Signature needs to match the one potentially defined in vm_types.go
// Using concrete *VMManager and interfaces for others
func NewMigrationManagerImpl(vmManager *VMManager, nodeManager NodeManager, storage MigrationStorage, executor MigrationExecutor, logger *logrus.Logger) *MigrationManagerImpl {
	return &MigrationManagerImpl{
		vmManager:        vmManager,   // Assign concrete type
		nodeManager:      nodeManager, // Assign interface
		storage:          storage,
		executor:         executor,
		activeMigrations: make(map[string]*Migration),
		logger:           logger,
	}
}

// Migrate initiates a VM migration
func (m *MigrationManagerImpl) Migrate(vmID, targetNodeID string, options MigrationOptions) (*MigrationRecord, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Get the VM
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		return nil, err
	}

	// Check if the VM can be migrated
	if err := m.vmManager.CheckVMCanMigrate(vmID); err != nil {
		return nil, err
	}

	// Check if there's already a migration in progress for this VM
	for _, migration := range m.activeMigrations {
		if migration.record.VMID == vmID && isActiveMigration(migration.record.State) {
			return nil, fmt.Errorf("VM %s already has a migration in progress", vmID)
		}
	}

	// Get source node
	sourceNode, err := m.nodeManager.GetNode(vm.NodeID)
	if err != nil {
		return nil, err
	}

	// Get target node
	targetNode, err := m.nodeManager.GetNode(targetNodeID)
	if err != nil {
		return nil, err
	}

	// Check if target node has enough resources
	if err := m.nodeManager.CheckNodeResources(targetNodeID, vm.Resources); err != nil {
		return nil, err
	}

	// Create migration record
	record := NewMigrationRecord(vmID, vm.Name, sourceNode.GetID(), targetNodeID, options.Type)

	// Set configuration options
	record.BandwidthLimit = options.BandwidthLimit
	record.CompressionLevel = options.CompressionLevel
	record.MemoryIterations = options.MemoryIterations
	record.Priority = options.Priority
	record.Force = options.Force
	record.SkipVerification = options.SkipVerification

	// Save the record
	if err := m.storage.SaveMigrationRecord(record); err != nil {
		return nil, err
	}

	// Create migration instance
	migration := NewMigration(record, vm, sourceNode, targetNode, m, m.executor)

	// Add to active migrations
	m.activeMigrations[record.ID] = migration

	// Start the migration
	if err := migration.Start(); err != nil {
		delete(m.activeMigrations, record.ID)
		return nil, err
	}

	return record, nil
}

// CancelMigration cancels an ongoing migration
func (m *MigrationManagerImpl) CancelMigration(migrationID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Find the migration
	migration, ok := m.activeMigrations[migrationID]
	if !ok {
		// Check if it exists in storage
		record, err := m.storage.LoadMigrationRecord(migrationID)
		if err != nil {
			return err
		}

		// Check if it's in a state that can't be cancelled
		if !isActiveMigration(record.State) {
			return fmt.Errorf("migration %s is in state %s and cannot be cancelled", migrationID, record.State)
		}

		return fmt.Errorf("migration %s not found in active migrations", migrationID)
	}

	// Cancel the migration
	return migration.Cancel()
}

// GetMigrationStatus returns the status of a migration
func (m *MigrationManagerImpl) GetMigrationStatus(migrationID string) (*MigrationStatus, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Check active migrations
	if migration, ok := m.activeMigrations[migrationID]; ok {
		return migration.GetStatus(), nil
	}

	// Check storage
	record, err := m.storage.LoadMigrationRecord(migrationID)
	if err != nil {
		return nil, err
	}

	// Convert record to status
	return &MigrationStatus{
		ID:               record.ID,
		VMID:             record.VMID,
		VMName:           record.VMName,
		SourceNodeID:     record.SourceNodeID,
		TargetNodeID:     record.TargetNodeID,
		MigrationType:    record.MigrationType,
		State:            record.State,
		Progress:         record.Progress,
		StartTime:        record.StartTime,
		CompletionTime:   record.CompletionTime,
		ErrorMessage:     record.ErrorMessage,
		BytesTransferred: record.BytesTransferred,
		TotalBytes:       record.TotalBytes,
		TransferRate:     record.TransferRate,
	}, nil
}

// ListMigrations returns a list of all migrations
func (m *MigrationManagerImpl) ListMigrations() ([]*MigrationStatus, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Get all records from storage
	records, err := m.storage.ListMigrationRecords()
	if err != nil {
		return nil, err
	}

	// Convert records to statuses
	statuses := make([]*MigrationStatus, 0, len(records))
	for _, record := range records {
		// Check if it's an active migration
		if migration, ok := m.activeMigrations[record.ID]; ok {
			statuses = append(statuses, migration.GetStatus())
		} else {
			// Use record from storage
			statuses = append(statuses, &MigrationStatus{
				ID:               record.ID,
				VMID:             record.VMID,
				VMName:           record.VMName,
				SourceNodeID:     record.SourceNodeID,
				TargetNodeID:     record.TargetNodeID,
				MigrationType:    record.MigrationType,
				State:            record.State,
				Progress:         record.Progress,
				StartTime:        record.StartTime,
				CompletionTime:   record.CompletionTime,
				ErrorMessage:     record.ErrorMessage,
				BytesTransferred: record.BytesTransferred,
				TotalBytes:       record.TotalBytes,
				TransferRate:     record.TransferRate,
			})
		}
	}

	return statuses, nil
}

// ListMigrationsForVM returns a list of migrations for a specific VM
func (m *MigrationManagerImpl) ListMigrationsForVM(vmID string) ([]*MigrationStatus, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Get all records for the VM from storage
	records, err := m.storage.ListMigrationRecordsForVM(vmID)
	if err != nil {
		return nil, err
	}

	// Convert records to statuses
	statuses := make([]*MigrationStatus, 0, len(records))
	for _, record := range records {
		// Check if it's an active migration
		if migration, ok := m.activeMigrations[record.ID]; ok {
			statuses = append(statuses, migration.GetStatus())
		} else {
			// Use record from storage
			statuses = append(statuses, &MigrationStatus{
				ID:               record.ID,
				VMID:             record.VMID,
				VMName:           record.VMName,
				SourceNodeID:     record.SourceNodeID,
				TargetNodeID:     record.TargetNodeID,
				MigrationType:    record.MigrationType,
				State:            record.State,
				Progress:         record.Progress,
				StartTime:        record.StartTime,
				CompletionTime:   record.CompletionTime,
				ErrorMessage:     record.ErrorMessage,
				BytesTransferred: record.BytesTransferred,
				TotalBytes:       record.TotalBytes,
				TransferRate:     record.TransferRate,
			})
		}
	}

	return statuses, nil
}

// SubscribeToMigrationEvents subscribes to events for a specific migration
func (m *MigrationManagerImpl) SubscribeToMigrationEvents(migrationID string) (<-chan MigrationEvent, func(), error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Check if the migration exists
	migration, ok := m.activeMigrations[migrationID]
	if !ok {
		// Check if it exists in storage
		record, err := m.storage.LoadMigrationRecord(migrationID)
		if err != nil {
			return nil, nil, err
		}

		// If the migration is not active, return an empty channel
		if !isActiveMigration(record.State) {
			ch := make(chan MigrationEvent)
			close(ch)
			return ch, func() {}, nil
		}

		return nil, nil, errors.New("migration not found in active migrations")
	}

	// Get the event channel
	eventCh := migration.Events()

	// Create a new channel for this subscriber
	subscriberCh := make(chan MigrationEvent, 100)

	// Create a new goroutine to forward events
	var once sync.Once
	stopCh := make(chan struct{})

	go func() {
		defer close(subscriberCh)

		for {
			select {
			case event, ok := <-eventCh:
				if !ok {
					return
				}

				// Forward the event
				select {
				case subscriberCh <- event:
					// Event forwarded
				default:
					// Subscriber channel full, drop the event
				}
			case <-stopCh:
				return
			}
		}
	}()

	// Return the channel and a function to unsubscribe
	unsubscribe := func() {
		once.Do(func() {
			close(stopCh)
		})
	}

	return subscriberCh, unsubscribe, nil
}

// SaveMigrationRecord saves a migration record
func (m *MigrationManagerImpl) SaveMigrationRecord(record *MigrationRecord) error {
	return m.storage.SaveMigrationRecord(record)
}

// LoadMigrationRecord loads a migration record
func (m *MigrationManagerImpl) LoadMigrationRecord(migrationID string) (*MigrationRecord, error) {
	return m.storage.LoadMigrationRecord(migrationID)
}

// ListMigrationRecords lists all migration records
func (m *MigrationManagerImpl) ListMigrationRecords() ([]*MigrationRecord, error) {
	return m.storage.ListMigrationRecords()
}

// Cleanup removes completed migrations from the active migrations map
func (m *MigrationManagerImpl) Cleanup() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, migration := range m.activeMigrations {
		status := migration.GetStatus()
		if !isActiveMigration(status.State) {
			delete(m.activeMigrations, id)
		}
	}
}

// Start background tasks
func (m *MigrationManagerImpl) Start() {
	// Start a goroutine to periodically clean up completed migrations
	go func() {
		ticker := time.NewTicker(10 * time.Minute)
		defer ticker.Stop()

		for range ticker.C {
			m.Cleanup()
		}
	}()
}

// Stop the manager and clean up resources
func (m *MigrationManagerImpl) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Cancel all active migrations
	for _, migration := range m.activeMigrations {
		_ = migration.Cancel()
	}
}

// isActiveMigration checks if a migration state is active
func isActiveMigration(state string) bool {
	return state == MigrationStatePending ||
		state == MigrationStateInitiating ||
		state == MigrationStateTransferring ||
		state == MigrationStateActivating ||
		state == MigrationStateRollingBack
}
