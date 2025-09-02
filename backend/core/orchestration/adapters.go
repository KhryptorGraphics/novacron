package orchestration

import (
	"context"
	"fmt"

	"github.com/khryptorgraphics/novacron/backend/core/orchestration/placement"
	vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// OrchestrationAdapters bundles the real adapters the engine will use for evacuation
// This avoids tight compile-time coupling in DefaultOrchestrationEngine itself.
type OrchestrationAdapters struct {
	VMManager       *vm.VMManager
	PlacementEngine placement.PlacementEngine
}

// ListVMsByNodeAdapter uses VMManager.ListVMsByNode and returns VM IDs
func (a *OrchestrationAdapters) ListVMsByNodeAdapter(nodeID string) ([]string, error) {
	if a == nil || a.VMManager == nil {
		return nil, fmt.Errorf("vm manager not configured")
	}
	vms := a.VMManager.ListVMsByNode(nodeID)
	ids := make([]string, 0, len(vms))
	for _, v := range vms {
		if v == nil { continue }
		ids = append(ids, v.ID())
	}
	return ids, nil
}

// SelectTargetAdapter uses placementEngine to pick a viable node avoiding the source
func (a *OrchestrationAdapters) SelectTargetAdapter(vmID string, sourceNodeID string) (string, error) {
	if a == nil || a.PlacementEngine == nil {
		return "", fmt.Errorf("placement engine not configured")
	}
	// Build a minimal placement request; in a real system include VMSpec and constraints
	req := &placement.PlacementRequest{
		VMID:         vmID,
		Strategy:     placement.StrategyBalanced,
		ExcludeNodes: []string{sourceNodeID},
	}
	decision, err := a.PlacementEngine.PlaceVM(context.Background(), req)
	if err != nil {
		return "", err
	}
	if decision == nil || decision.SelectedNode == "" || decision.SelectedNode == sourceNodeID {
		return "", fmt.Errorf("no suitable target node found")
	}
	return decision.SelectedNode, nil
}

// MigrateAdapter triggers a migration via VM drivers
// Falls back to container/KVM driver Migrate if MigrationManager impl is not available
func (a *OrchestrationAdapters) MigrateAdapter(ctx context.Context, vmID, targetNodeID string) error {
	if a == nil || a.VMManager == nil {
		return fmt.Errorf("vm manager not configured")
	}
	// Retrieve VM to infer its driver/config and call driver Migrate via vm_operations
	vmObj, err := a.VMManager.GetVM(vmID)
	if err != nil {
		return fmt.Errorf("get vm %s: %w", vmID, err)
	}
	// Use the VM's config via method to avoid accessing unexported fields
	config := vmObj.Config()
	driver, err := a.VMManager.GetDriverForConfig(config)
	if err != nil {
		return fmt.Errorf("get driver: %w", err)
	}
	params := map[string]string{"reason": "evacuation"}
	return driver.Migrate(ctx, vmID, targetNodeID, params)
}

