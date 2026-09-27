package main

import (
	"context"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/orchestration/healing"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// vmHealingController adapts the real VM backend to healing.VMController.
// Restarts are routed through the package-level restartSupervisor so an
// ad-hoc healing restart never bypasses supervisor policy (RecordStop,
// policy=no, exhausted attempts — see RestartSupervisor.RequestRestart);
// migrations go through the same admission-controlled transfer machinery
// node drain uses (queueMigrationTransfer).
type vmHealingController struct {
	vmManager  *core_vm.VMManager
	supervisor *core_vm.RestartSupervisor
}

func (a *vmHealingController) RestartVM(ctx context.Context, vmID string) error {
	if a.supervisor == nil {
		return fmt.Errorf("vm healing restart: no restart supervisor configured")
	}
	return a.supervisor.RequestRestart(ctx, vmID)
}

func (a *vmHealingController) MigrateVM(ctx context.Context, vmID, targetNode string, options map[string]string) error {
	vm, err := a.vmManager.GetVM(vmID)
	if err != nil {
		return fmt.Errorf("healing migrate: %w", err)
	}
	reason := fmt.Sprintf("healing-triggered migration of %s to %s", vmID, targetNode)
	_, err = queueMigrationTransfer(transfers, a.vmManager, vmID, int64(vm.GetMemoryMB()), targetNode, reason)
	return err
}

// vmMigrationTargetSelector reuses the drain coordinator's own reachable-peer
// candidate list (never placeVM/allNodeCapacities, whose list starts with the
// local node) and the same median-bandwidth-first placement drain uses, so a
// healing migration and a drain migration pick targets under one policy.
type vmMigrationTargetSelector struct {
	vmManager  *core_vm.VMManager
	candidates func(excludeNode string) []drainCandidate
}

func (s *vmMigrationTargetSelector) SelectTarget(ctx context.Context, vmID string) (string, error) {
	vm, err := s.vmManager.GetVM(vmID)
	if err != nil {
		return "", fmt.Errorf("healing migration target: %w", err)
	}
	memMB := int64(vm.GetMemoryMB())
	for _, c := range medianBWOrder(s.candidates(selfNodeID())) {
		if c.MemFreeMB >= memMB {
			return c.NodeID, nil
		}
	}
	return "", fmt.Errorf("no cluster peer has capacity to receive migrated VM %s", vmID)
}

// vmHealthSource implements healing.HealthSource against the real VM
// backend, deferring to the RestartSupervisor to decide when a failed VM is
// actually healing's problem:
//
//   - running                                          -> healthy
//   - failed, and the supervisor is not going to act
//     on it (untracked, or its restart attempts are
//     already exhausted)                                -> unhealthy
//   - failed, but the supervisor will act on it
//     (still watching/backing off)                       -> nil (no sample:
//     already handled by the supervisor's own scheduled restart)
//   - RecordStop'd or policy=no                          -> nil (no sample:
//     the operator/policy opted this VM out of recovery)
//   - every other VM state (stopped, paused, restarting,
//     migrating, creating, deleting, unknown)             -> nil (no sample)
//   - missing VM                                          -> error (never a
//     fabricated sample; healing.HealthSource's documented contract keeps
//     errors out of the failure detector)
func vmHealthSource(vmManager *core_vm.VMManager, supervisor *core_vm.RestartSupervisor) healing.HealthSource {
	return func(targetID string) (*healing.HealthSample, error) {
		vm, err := vmManager.GetVM(targetID)
		if err != nil {
			return nil, fmt.Errorf("vm health source: %w", err)
		}
		switch vm.State() {
		case core_vm.StateRunning:
			return &healing.HealthSample{TargetID: targetID, Timestamp: time.Now(), Healthy: true}, nil
		case core_vm.StateFailed:
			if supervisor == nil {
				return &healing.HealthSample{TargetID: targetID, Timestamp: time.Now(), Healthy: false}, nil
			}
			status, tracked := supervisor.Inspect(context.Background(), targetID)
			if !tracked {
				// The supervisor has no record of this VM, so it will never
				// act on the crash: healing must.
				return &healing.HealthSample{TargetID: targetID, Timestamp: time.Now(), Healthy: false}, nil
			}
			if status.IsRecordStopped() || status.Policy == core_vm.RestartPolicyNo {
				return nil, nil // opted out of recovery entirely
			}
			if status.IsPermanentFailure() {
				return &healing.HealthSample{TargetID: targetID, Timestamp: time.Now(), Healthy: false}, nil
			}
			// Still watching or backing off: the supervisor's own scheduled
			// restart is already the plan of record.
			return nil, nil
		default:
			return nil, nil
		}
	}
}
