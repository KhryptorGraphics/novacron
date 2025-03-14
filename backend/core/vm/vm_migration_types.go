package vm

import (
	"time"

	"github.com/google/uuid"
)

// Migration types
const (
	MigrationTypeCold string = "cold"
	MigrationTypeWarm string = "warm"
	MigrationTypeLive string = "live"
)

// Migration states
const (
	MigrationStatePending     string = "pending"
	MigrationStateInitiating  string = "initiating"
	MigrationStateTransferring string = "transferring"
	MigrationStateActivating  string = "activating"
	MigrationStateCompleted   string = "completed"
	MigrationStateFailed      string = "failed"
	MigrationStateRollingBack string = "rollingback"
	MigrationStateRolledBack  string = "rolledback"
)

// MigrationRecord represents the persistent record of a VM migration
type MigrationRecord struct {
	ID             string    `json:"id"`
	VMID           string    `json:"vmId"`
	VMName         string    `json:"vmName"`
	SourceNodeID   string    `json:"sourceNodeId"`
	TargetNodeID   string    `json:"targetNodeId"`
	MigrationType  string    `json:"migrationType"`
	State          string    `json:"state"`
	Progress       float64   `json:"progress"`
	StartTime      time.Time `json:"startTime"`
	CompletionTime time.Time `json:"completionTime,omitempty"`
	ErrorMessage   string    `json:"errorMessage,omitempty"`
	
	// Performance metrics and details
	TransferRate      int64 `json:"transferRate,omitempty"`      // Bytes per second
	BytesTransferred  int64 `json:"bytesTransferred,omitempty"`  // Total bytes transferred so far
	TotalBytes        int64 `json:"totalBytes,omitempty"`        // Total bytes to transfer
	DowntimeMs        int64 `json:"downtimeMs,omitempty"`        // Downtime in milliseconds
	TransferTimeMs    int64 `json:"transferTimeMs,omitempty"`    // Transfer time in milliseconds
	ActivationTimeMs  int64 `json:"activationTimeMs,omitempty"`  // Activation time in milliseconds
	
	// Configuration options
	BandwidthLimit   int64  `json:"bandwidthLimit,omitempty"`   // Bandwidth limit in bytes per second
	CompressionLevel int    `json:"compressionLevel,omitempty"` // Compression level (0-9)
	MemoryIterations int    `json:"memoryIterations,omitempty"` // Number of memory iterations for live migration
	Priority         int    `json:"priority,omitempty"`         // Migration priority (higher numbers = higher priority)
	Force            bool   `json:"force,omitempty"`            // Force migration despite warnings
	SkipVerification bool   `json:"skipVerification,omitempty"` // Skip VM verification after migration
}

// NewMigrationRecord creates a new migration record with default values
func NewMigrationRecord(vmID, vmName, sourceNodeID, targetNodeID, migrationType string) *MigrationRecord {
	return &MigrationRecord{
		ID:            uuid.New().String(),
		VMID:          vmID,
		VMName:        vmName,
		SourceNodeID:  sourceNodeID,
		TargetNodeID:  targetNodeID,
		MigrationType: migrationType,
		State:         MigrationStatePending,
		Progress:      0.0,
		StartTime:     time.Now(),
	}
}

// MigrationOptions defines configuration options for a VM migration
type MigrationOptions struct {
	Type             string // Migration type: cold, warm, live
	BandwidthLimit   int64  // Bandwidth limit in bytes per second (0 = unlimited)
	CompressionLevel int    // Compression level (0-9)
	MemoryIterations int    // Number of memory iterations for live migration
	Priority         int    // Migration priority (higher numbers = higher priority)
	Force            bool   // Force migration despite warnings
	SkipVerification bool   // Skip VM verification after migration
}

// DefaultMigrationOptions returns the default migration options
func DefaultMigrationOptions() MigrationOptions {
	return MigrationOptions{
		Type:             MigrationTypeCold,
		BandwidthLimit:   0, // Unlimited
		CompressionLevel: 6,
		MemoryIterations: 5,
		Priority:         5,
		Force:            false,
		SkipVerification: false,
	}
}

// MigrationEvent represents an event related to VM migration
type MigrationEvent struct {
	ID               string    `json:"id"`
	MigrationID      string    `json:"migrationId"`
	Type             string    `json:"type"`
	Timestamp        time.Time `json:"timestamp"`
	Message          string    `json:"message"`
	Progress         float64   `json:"progress"`
	BytesTransferred int64     `json:"bytesTransferred,omitempty"`
	TransferRate     int64     `json:"transferRate,omitempty"`
}

// MigrationEventTypes define the different types of migration events
const (
	MigrationEventInitiated       string = "migration_initiated"
	MigrationEventStarted         string = "migration_started"
	MigrationEventProgress        string = "migration_progress"
	MigrationEventTransferStarted string = "transfer_started"
	MigrationEventTransferDone    string = "transfer_done"
	MigrationEventActivationStarted string = "activation_started"
	MigrationEventActivationDone  string = "activation_done"
	MigrationEventCompleted       string = "migration_completed"
	MigrationEventFailed          string = "migration_failed"
	MigrationEventRollbackStarted string = "rollback_started"
	MigrationEventRollbackDone    string = "rollback_done"
)

// MigrationStatus provides a snapshot of the current migration status
type MigrationStatus struct {
	ID               string    `json:"id"`
	VMID             string    `json:"vmId"`
	VMName           string    `json:"vmName"`
	SourceNodeID     string    `json:"sourceNodeId"`
	TargetNodeID     string    `json:"targetNodeId"`
	MigrationType    string    `json:"migrationType"`
	State            string    `json:"state"`
	Progress         float64   `json:"progress"`
	StartTime        time.Time `json:"startTime"`
	CompletionTime   time.Time `json:"completionTime,omitempty"`
	ErrorMessage     string    `json:"errorMessage,omitempty"`
	BytesTransferred int64     `json:"bytesTransferred,omitempty"`
	TotalBytes       int64     `json:"totalBytes,omitempty"`
	TransferRate     int64     `json:"transferRate,omitempty"`
}

// MigrationManager defines the interface for migration management
type MigrationManager interface {
	// Migration operations
	Migrate(vmID, targetNodeID string, options MigrationOptions) (*MigrationRecord, error)
	CancelMigration(migrationID string) error
	
	// Migration status
	GetMigrationStatus(migrationID string) (*MigrationStatus, error)
	ListMigrations() ([]*MigrationStatus, error)
	ListMigrationsForVM(vmID string) ([]*MigrationStatus, error)
	
	// Event subscription
	SubscribeToMigrationEvents(migrationID string) (<-chan MigrationEvent, func(), error)
	
	// Storage
	SaveMigrationRecord(record *MigrationRecord) error
	LoadMigrationRecord(migrationID string) (*MigrationRecord, error)
	ListMigrationRecords() ([]*MigrationRecord, error)
}

// MigrationExecutor defines the interface for migration execution
type MigrationExecutor interface {
	// Execute different types of migrations
	ExecuteColdMigration(migrationID string, vm *VM, targetNode Node) error
	ExecuteWarmMigration(migrationID string, vm *VM, targetNode Node) error
	ExecuteLiveMigration(migrationID string, vm *VM, targetNode Node) error
	
	// Rollback migrations
	RollbackMigration(migrationID string, vm *VM, sourceNode Node) error
}
