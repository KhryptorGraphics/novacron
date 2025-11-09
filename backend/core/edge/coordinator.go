package edge

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// EdgeCloudCoordinator manages hierarchical edge-cloud orchestration
type EdgeCloudCoordinator struct {
	config         *EdgeConfig
	discovery      *EdgeDiscovery
	placement      *PlacementEngine
	migrations     map[string]*MigrationStatus
	migrationsMu   sync.RWMutex
	vmPlacements   map[string]string // vmID -> edgeNodeID
	placementsMu   sync.RWMutex
}

// NewEdgeCloudCoordinator creates a new edge-cloud coordinator
func NewEdgeCloudCoordinator(config *EdgeConfig, discovery *EdgeDiscovery, placement *PlacementEngine) *EdgeCloudCoordinator {
	return &EdgeCloudCoordinator{
		config:       config,
		discovery:    discovery,
		placement:    placement,
		migrations:   make(map[string]*MigrationStatus),
		vmPlacements: make(map[string]string),
	}
}

// DeployToEdge deploys a VM to the optimal edge location
func (ecc *EdgeCloudCoordinator) DeployToEdge(ctx context.Context, req *PlacementRequest) (*PlacementDecision, error) {
	// Find optimal placement
	decision, err := ecc.placement.PlaceVM(ctx, req)
	if err != nil {
		return nil, err
	}

	// Record placement
	ecc.placementsMu.Lock()
	ecc.vmPlacements[req.VMID] = decision.EdgeNodeID
	ecc.placementsMu.Unlock()

	return decision, nil
}

// MigrateVM migrates a VM between edge and cloud or between edge nodes
func (ecc *EdgeCloudCoordinator) MigrateVM(ctx context.Context, req *MigrationRequest) (*MigrationStatus, error) {
	// Check if migration already in progress
	ecc.migrationsMu.RLock()
	existing, inProgress := ecc.migrations[req.VMID]
	ecc.migrationsMu.RUnlock()

	if inProgress && existing.State == MigrationStateRunning {
		return nil, ErrMigrationInProgress
	}

	// Validate source and target nodes
	sourceNode, err := ecc.discovery.GetNode(req.SourceNodeID)
	if err != nil {
		return nil, fmt.Errorf("source node not found: %w", err)
	}

	targetNode, err := ecc.discovery.GetNode(req.TargetNodeID)
	if err != nil {
		return nil, fmt.Errorf("target node not found: %w", err)
	}

	// Check target node has sufficient resources
	if targetNode.Status.State != EdgeNodeStateOnline {
		return nil, ErrInvalidMigrationTarget
	}

	// Create migration status
	status := &MigrationStatus{
		VMID:        req.VMID,
		State:       MigrationStatePending,
		Progress:    0.0,
		StartedAt:   time.Now(),
	}

	ecc.migrationsMu.Lock()
	ecc.migrations[req.VMID] = status
	ecc.migrationsMu.Unlock()

	// Start migration asynchronously
	go ecc.executeMigration(ctx, req, status, sourceNode, targetNode)

	return status, nil
}

// executeMigration executes the migration process
func (ecc *EdgeCloudCoordinator) executeMigration(ctx context.Context, req *MigrationRequest, status *MigrationStatus, source, target *EdgeNode) {
	defer func() {
		if r := recover(); r != nil {
			ecc.updateMigrationStatus(status, MigrationStateFailed, 0, fmt.Sprintf("panic: %v", r))
		}
	}()

	migrationCtx, cancel := context.WithTimeout(ctx, ecc.config.MigrationTimeout)
	defer cancel()

	// Update state to running
	ecc.updateMigrationStatus(status, MigrationStateRunning, 0, "")

	// Phase 1: Pre-migration checks (10%)
	if err := ecc.preMigrationChecks(migrationCtx, req, source, target); err != nil {
		ecc.updateMigrationStatus(status, MigrationStateFailed, 10, err.Error())
		return
	}
	ecc.updateMigrationStatus(status, MigrationStateRunning, 10, "")

	// Phase 2: Snapshot VM state (30%)
	snapshotSize, err := ecc.snapshotVM(migrationCtx, req.VMID, source)
	if err != nil {
		ecc.updateMigrationStatus(status, MigrationStateFailed, 30, err.Error())
		return
	}
	status.TotalBytes = snapshotSize
	ecc.updateMigrationStatus(status, MigrationStateRunning, 30, "")

	// Phase 3: Transfer VM data (70%)
	if err := ecc.transferVMData(migrationCtx, req, status, source, target); err != nil {
		ecc.updateMigrationStatus(status, MigrationStateFailed, 70, err.Error())
		return
	}
	ecc.updateMigrationStatus(status, MigrationStateRunning, 70, "")

	// Phase 4: Switch over (90%)
	downtimeStart := time.Now()
	if err := ecc.switchoverVM(migrationCtx, req.VMID, source, target); err != nil {
		ecc.updateMigrationStatus(status, MigrationStateFailed, 90, err.Error())
		return
	}
	status.DowntimeMs = time.Since(downtimeStart).Milliseconds()
	ecc.updateMigrationStatus(status, MigrationStateRunning, 90, "")

	// Phase 5: Cleanup (100%)
	ecc.cleanupMigration(migrationCtx, req.VMID, source)
	ecc.updateMigrationStatus(status, MigrationStateCompleted, 100, "")

	// Update placement record
	ecc.placementsMu.Lock()
	ecc.vmPlacements[req.VMID] = target.ID
	ecc.placementsMu.Unlock()

	// Check if downtime met target
	if status.DowntimeMs > req.MaxDowntime.Milliseconds() {
		// Log warning
	}
}

// preMigrationChecks performs pre-migration validation
func (ecc *EdgeCloudCoordinator) preMigrationChecks(ctx context.Context, req *MigrationRequest, source, target *EdgeNode) error {
	// Check target has sufficient resources
	// This is a simplified check - in production would need actual VM resource data
	availCPU := target.Resources.TotalCPUCores - target.Resources.UsedCPUCores
	availMem := target.Resources.TotalMemoryMB - target.Resources.UsedMemoryMB

	if availCPU < 1 || availMem < 1024 {
		return ErrInsufficientEdgeResources
	}

	// Check network connectivity
	if target.Network.VPNEndpoint == "" && ecc.config.VPNEnabled {
		return ErrVPNTunnelFailed
	}

	return nil
}

// snapshotVM creates a snapshot of the VM
func (ecc *EdgeCloudCoordinator) snapshotVM(ctx context.Context, vmID string, node *EdgeNode) (int64, error) {
	// In production, this would call the actual VM snapshot API
	// For now, return simulated snapshot size
	return 10 * 1024 * 1024 * 1024, nil // 10 GB
}

// transferVMData transfers VM data to target node
func (ecc *EdgeCloudCoordinator) transferVMData(ctx context.Context, req *MigrationRequest, status *MigrationStatus, source, target *EdgeNode) error {
	// Simulate data transfer with progress updates
	totalBytes := status.TotalBytes
	transferred := int64(0)
	chunkSize := int64(100 * 1024 * 1024) // 100 MB chunks

	transferRate := ecc.config.MigrationBandwidth // bytes/sec
	chunks := totalBytes / chunkSize

	for i := int64(0); i < chunks; i++ {
		select {
		case <-ctx.Done():
			return ErrMigrationTimeout
		default:
		}

		// Simulate transfer time
		transferTime := time.Duration(float64(chunkSize)/float64(transferRate)) * time.Second
		time.Sleep(transferTime / 10) // Speed up for testing

		transferred += chunkSize
		status.BytesCopied = transferred
		progress := 30.0 + (float64(transferred)/float64(totalBytes))*40.0
		status.Progress = progress
	}

	status.BytesCopied = totalBytes
	return nil
}

// switchoverVM performs the VM switchover
func (ecc *EdgeCloudCoordinator) switchoverVM(ctx context.Context, vmID string, source, target *EdgeNode) error {
	// In production, this would:
	// 1. Pause VM on source
	// 2. Transfer final state
	// 3. Start VM on target
	// 4. Update network routing
	time.Sleep(500 * time.Millisecond) // Simulate switchover time
	return nil
}

// cleanupMigration cleans up source node after migration
func (ecc *EdgeCloudCoordinator) cleanupMigration(ctx context.Context, vmID string, source *EdgeNode) error {
	// In production, this would delete the VM from source node
	return nil
}

// updateMigrationStatus updates migration status
func (ecc *EdgeCloudCoordinator) updateMigrationStatus(status *MigrationStatus, state MigrationState, progress float64, errMsg string) {
	status.State = state
	status.Progress = progress
	status.ElapsedTime = time.Since(status.StartedAt)

	if state == MigrationStateCompleted || state == MigrationStateFailed {
		now := time.Now()
		status.CompletedAt = &now
	}

	if errMsg != "" {
		status.Error = errMsg
	}

	// Estimate remaining time
	if progress > 0 && state == MigrationStateRunning {
		totalTime := float64(status.ElapsedTime) / (progress / 100.0)
		status.EstimatedTime = time.Duration(totalTime) - status.ElapsedTime
	}
}

// GetMigrationStatus retrieves migration status
func (ecc *EdgeCloudCoordinator) GetMigrationStatus(vmID string) (*MigrationStatus, error) {
	ecc.migrationsMu.RLock()
	defer ecc.migrationsMu.RUnlock()

	status, exists := ecc.migrations[vmID]
	if !exists {
		return nil, fmt.Errorf("no migration found for VM %s", vmID)
	}

	return status, nil
}

// GetVMPlacement retrieves the current edge node for a VM
func (ecc *EdgeCloudCoordinator) GetVMPlacement(vmID string) (string, error) {
	ecc.placementsMu.RLock()
	defer ecc.placementsMu.RUnlock()

	nodeID, exists := ecc.vmPlacements[vmID]
	if !exists {
		return "", fmt.Errorf("no placement found for VM %s", vmID)
	}

	return nodeID, nil
}

// OptimizePlacements rebalances VMs across edge nodes
func (ecc *EdgeCloudCoordinator) OptimizePlacements(ctx context.Context) error {
	// Get all edge nodes
	nodes := ecc.discovery.GetHealthyNodes()
	if len(nodes) == 0 {
		return ErrInsufficientEdgeNodes
	}

	// Find overloaded nodes (>90% utilization)
	var overloaded []*EdgeNode
	for _, node := range nodes {
		if node.Resources.UtilizationPercent > 90.0 {
			overloaded = append(overloaded, node)
		}
	}

	// For each overloaded node, migrate some VMs
	for _, node := range overloaded {
		// In production, would identify specific VMs to migrate
		// and find optimal target nodes
	}

	return nil
}

// SyncState synchronizes state across edge and cloud tiers
func (ecc *EdgeCloudCoordinator) SyncState(ctx context.Context, vmID string) error {
	// Get current placement
	nodeID, err := ecc.GetVMPlacement(vmID)
	if err != nil {
		return err
	}

	// Get node
	node, err := ecc.discovery.GetNode(nodeID)
	if err != nil {
		return err
	}

	// In production, this would:
	// 1. Fetch VM state from edge node
	// 2. Sync to cloud backend
	// 3. Update central database
	// 4. Handle conflicts

	_ = node // Use node
	return nil
}

// HandleEdgeFailure handles edge node failure
func (ecc *EdgeCloudCoordinator) HandleEdgeFailure(ctx context.Context, nodeID string) error {
	// Find all VMs on failed node
	ecc.placementsMu.RLock()
	affectedVMs := make([]string, 0)
	for vmID, placedNodeID := range ecc.vmPlacements {
		if placedNodeID == nodeID {
			affectedVMs = append(affectedVMs, vmID)
		}
	}
	ecc.placementsMu.RUnlock()

	// Migrate each affected VM to healthy node
	for _, vmID := range affectedVMs {
		// Find alternative healthy node
		nodes := ecc.discovery.GetHealthyNodes()
		if len(nodes) == 0 {
			return ErrInsufficientEdgeNodes
		}

		// Select node with lowest utilization
		var targetNode *EdgeNode
		minUtil := 100.0
		for _, node := range nodes {
			if node.ID != nodeID && node.Resources.UtilizationPercent < minUtil {
				targetNode = node
				minUtil = node.Resources.UtilizationPercent
			}
		}

		if targetNode == nil {
			continue
		}

		// Initiate migration
		migReq := &MigrationRequest{
			VMID:          vmID,
			SourceNodeID:  nodeID,
			TargetNodeID:  targetNode.ID,
			MigrationType: MigrationTypeColdMigrate,
			MaxDowntime:   30 * time.Second,
			Priority:      10,
			Reason:        "edge_node_failure",
		}

		_, err := ecc.MigrateVM(ctx, migReq)
		if err != nil {
			// Log error but continue with other VMs
			continue
		}
	}

	return nil
}
