package healing

import "context"

// VMController is the minimal VM-lifecycle surface healing depends on.
// Implementations wrap the real backend (api-server's VMManager/RestartSupervisor);
// errors are returned verbatim to the caller — never swallowed into a fake
// success.
type VMController interface {
	RestartVM(ctx context.Context, vmID string) error
	MigrateVM(ctx context.Context, vmID string, targetNode string, options map[string]string) error
}

// MigrationTargetSelector picks a destination node for a VM healing
// migration.
type MigrationTargetSelector interface {
	SelectTarget(ctx context.Context, vmID string) (string, error)
}
