package vm

// MigrationType represents the type of migration
type MigrationType string

// Node represents a node in the cluster
type Node struct {
	ID       string            `json:"id"`
	Name     string            `json:"name"`
	Address  string            `json:"address"`
	Status   string            `json:"status"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// MigrationStatus represents the status of a migration operation (enum type)
type MigrationStatus string

const (
	MigrationStatusPending    MigrationStatus = "pending"
	MigrationStatusInProgress MigrationStatus = "in_progress"
	MigrationStatusCompleted  MigrationStatus = "completed"
	MigrationStatusFailed     MigrationStatus = "failed"
	MigrationStatusRolledBack MigrationStatus = "rolled_back"
)

// Migration types
const (
	MigrationTypeCold MigrationType = "cold"
	MigrationTypeWarm MigrationType = "warm"
	MigrationTypeLive MigrationType = "live"
)
