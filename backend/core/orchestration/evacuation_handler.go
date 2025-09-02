package orchestration

import (
	"context"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
)

// VMListFunc returns VM IDs on a given node
type VMListFunc func(nodeID string) ([]string, error)

// SelectTargetFunc chooses a destination node for a VM evacuation
// It should return a viable target node ID different from the source
type SelectTargetFunc func(vmID string, sourceNodeID string) (string, error)

// MigrateFunc triggers a migration of a VM to the target node
// Implementation may choose cold/warm/live based on context
type MigrateFunc func(ctx context.Context, vmID, targetNodeID string) error

// DefaultEvacuationHandler is a minimal, injectable implementation to evacuate VMs from a node
// Dependencies are function-based to avoid tight coupling with specific packages
// Provide lightweight wrappers at composition time (e.g., using VMManager and MigrationManager)
type DefaultEvacuationHandler struct {
	listVMsByNode VMListFunc
	selectTarget  SelectTargetFunc
	migrate       MigrateFunc
	logger        *logrus.Logger
}

// NewDefaultEvacuationHandler constructs the handler
func NewDefaultEvacuationHandler(listFn VMListFunc, pickFn SelectTargetFunc, migrateFn MigrateFunc, logger *logrus.Logger) *DefaultEvacuationHandler {
	return &DefaultEvacuationHandler{
		listVMsByNode: listFn,
		selectTarget:  pickFn,
		migrate:       migrateFn,
		logger:        logger,
	}
}

// EvacuateNode evacuates VMs off a failed/unhealthy node.
// Best-effort: continues on errors and logs them, returning the first error encountered.
func (h *DefaultEvacuationHandler) EvacuateNode(ctx context.Context, nodeID string) error {
	if h == nil {
		return fmt.Errorf("evacuation handler not configured")
	}
	if h.listVMsByNode == nil || h.selectTarget == nil || h.migrate == nil {
		return fmt.Errorf("evacuation handler dependencies not configured")
	}

	start := time.Now()
	h.logger.WithField("node_id", nodeID).Info("Starting node evacuation")

	vmIDs, err := h.listVMsByNode(nodeID)
	if err != nil {
		return fmt.Errorf("list VMs on node %s: %w", nodeID, err)
	}
	if len(vmIDs) == 0 {
		h.logger.WithField("node_id", nodeID).Info("No VMs to evacuate")
		return nil
	}

	var firstErr error
	for _, id := range vmIDs {
		// Choose target node
		target, pickErr := h.selectTarget(id, nodeID)
		if pickErr != nil || target == "" || target == nodeID {
			if firstErr == nil && pickErr != nil {
				firstErr = pickErr
			}
			h.logger.WithFields(logrus.Fields{"vm_id": id, "node_id": nodeID}).Warn("No valid target for evacuation; skipping")
			continue
		}

		// Migrate VM
		if migErr := h.migrate(ctx, id, target); migErr != nil {
			if firstErr == nil {
				firstErr = migErr
			}
			h.logger.WithError(migErr).WithFields(logrus.Fields{"vm_id": id, "target": target}).Error("Evacuation migration failed")
			continue
		}

		h.logger.WithFields(logrus.Fields{"vm_id": id, "target": target}).Info("Evacuation migration initiated")
	}

	h.logger.WithFields(logrus.Fields{"node_id": nodeID, "duration": time.Since(start)}).Info("Node evacuation completed")
	return firstErr
}

