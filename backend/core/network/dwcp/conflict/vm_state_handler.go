package conflict

import (
	"context"
	"fmt"
	"time"
)

// VMState represents virtual machine state
type VMState struct {
	ID              string
	PowerState      string
	CPUAllocation   int
	MemoryMB        int
	NetworkConfig   map[string]interface{}
	DiskConfig      []DiskInfo
	SnapshotID      string
	MigrationStatus string
	LastUpdated     time.Time
	NodeID          string
}

// DiskInfo represents disk configuration
type DiskInfo struct {
	ID       string
	SizeGB   int
	Type     string
	Path     string
}

// VMStateConflictHandler handles VM-specific conflicts
type VMStateConflictHandler struct {
	policyManager *PolicyManager
	detector      *ConflictDetector
}

// NewVMStateConflictHandler creates a new VM state conflict handler
func NewVMStateConflictHandler(pm *PolicyManager, detector *ConflictDetector) *VMStateConflictHandler {
	handler := &VMStateConflictHandler{
		policyManager: pm,
		detector:      detector,
	}

	// Register VM-specific policy
	handler.registerVMPolicy()
	return handler
}

// registerVMPolicy registers VM-specific conflict resolution policy
func (h *VMStateConflictHandler) registerVMPolicy() {
	policy := NewPolicyBuilder("vm_state").
		WithDefaultStrategy(StrategySemanticMerge).
		WithFieldStrategy("power_state", StrategyHighestPriority).
		WithFieldStrategy("cpu_allocation", StrategyLastWriteWins).
		WithFieldStrategy("memory_mb", StrategyLastWriteWins).
		WithFieldStrategy("network_config", StrategySemanticMerge).
		WithRule("power_state_conflict", h.isPowerStateConflict, StrategyCustomFunction, 100).
		WithRule("migration_conflict", h.isMigrationConflict, StrategyManualIntervention, 90).
		WithRule("resource_conflict", h.isResourceConflict, StrategySemanticMerge, 80).
		WithEscalation("running_vs_stopped", h.isCriticalPowerConflict, EscalationActionManual, 100).
		WithMaxRetries(5).
		WithManualThreshold(0.8).
		WithTimeout(60 * time.Second).
		Build()

	h.policyManager.RegisterPolicy(policy)
}

// ResolveVMStateConflict resolves VM state conflicts
func (h *VMStateConflictHandler) ResolveVMStateConflict(ctx context.Context, vmID string, local, remote *VMState) (*VMState, error) {
	// Create versions from VM states
	localVersion := &Version{
		VectorClock: NewVectorClock(),
		Timestamp:   local.LastUpdated,
		NodeID:      local.NodeID,
		Data:        local,
	}

	remoteVersion := &Version{
		VectorClock: NewVectorClock(),
		Timestamp:   remote.LastUpdated,
		NodeID:      remote.NodeID,
		Data:        remote,
	}

	// Detect conflict
	conflict, err := h.detector.DetectConflict(ctx, vmID, localVersion, remoteVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to detect conflict: %w", err)
	}

	if conflict == nil {
		// No conflict, return remote
		return remote, nil
	}

	// Add VM-specific context
	h.enrichConflictContext(conflict, local, remote)

	// Resolve using VM policy
	result, err := h.policyManager.ResolveConflict(ctx, conflict, "vm_state")
	if err != nil {
		return nil, fmt.Errorf("failed to resolve VM state conflict: %w", err)
	}

	if !result.Success {
		return nil, fmt.Errorf("conflict resolution failed: %s", result.Message)
	}

	// Convert result back to VMState
	resolvedState, ok := result.ResolvedData.(*VMState)
	if !ok {
		return nil, fmt.Errorf("invalid resolved data type")
	}

	return resolvedState, nil
}

// enrichConflictContext adds VM-specific context to conflict
func (h *VMStateConflictHandler) enrichConflictContext(conflict *Conflict, local, remote *VMState) {
	conflict.Context["vm_id"] = local.ID
	conflict.Context["local_power_state"] = local.PowerState
	conflict.Context["remote_power_state"] = remote.PowerState
	conflict.Context["local_node"] = local.NodeID
	conflict.Context["remote_node"] = remote.NodeID

	// Identify affected components
	if local.PowerState != remote.PowerState {
		conflict.AffectedFields = append(conflict.AffectedFields, "power_state")
	}
	if local.CPUAllocation != remote.CPUAllocation {
		conflict.AffectedFields = append(conflict.AffectedFields, "cpu_allocation")
	}
	if local.MemoryMB != remote.MemoryMB {
		conflict.AffectedFields = append(conflict.AffectedFields, "memory_mb")
	}
	if local.MigrationStatus != remote.MigrationStatus {
		conflict.AffectedFields = append(conflict.AffectedFields, "migration_status")
	}
}

// Condition functions for VM conflicts
func (h *VMStateConflictHandler) isPowerStateConflict(c *Conflict) bool {
	localPower, localOk := c.Context["local_power_state"].(string)
	remotePower, remoteOk := c.Context["remote_power_state"].(string)
	return localOk && remoteOk && localPower != remotePower
}

func (h *VMStateConflictHandler) isCriticalPowerConflict(c *Conflict) bool {
	localPower, localOk := c.Context["local_power_state"].(string)
	remotePower, remoteOk := c.Context["remote_power_state"].(string)

	if !localOk || !remoteOk {
		return false
	}

	// Critical if one is running and other is stopped
	critical := (localPower == "running" && remotePower == "stopped") ||
		(localPower == "stopped" && remotePower == "running")

	return critical
}

func (h *VMStateConflictHandler) isMigrationConflict(c *Conflict) bool {
	for _, field := range c.AffectedFields {
		if field == "migration_status" {
			return true
		}
	}
	return false
}

func (h *VMStateConflictHandler) isResourceConflict(c *Conflict) bool {
	for _, field := range c.AffectedFields {
		if field == "cpu_allocation" || field == "memory_mb" {
			return true
		}
	}
	return false
}

// ResolvePowerStateConflict resolves power state conflicts
func (h *VMStateConflictHandler) ResolvePowerStateConflict(local, remote *VMState) (string, error) {
	// Priority: running > paused > stopped > unknown
	priority := map[string]int{
		"running": 4,
		"paused":  3,
		"stopped": 2,
		"unknown": 1,
	}

	localPriority := priority[local.PowerState]
	remotePriority := priority[remote.PowerState]

	if remotePriority > localPriority {
		return remote.PowerState, nil
	}
	return local.PowerState, nil
}

// ResolveResourceAllocation resolves CPU/memory conflicts
func (h *VMStateConflictHandler) ResolveResourceAllocation(local, remote *VMState) (int, int, error) {
	// Use higher allocation to avoid resource starvation
	cpu := local.CPUAllocation
	if remote.CPUAllocation > cpu {
		cpu = remote.CPUAllocation
	}

	memory := local.MemoryMB
	if remote.MemoryMB > memory {
		memory = remote.MemoryMB
	}

	return cpu, memory, nil
}

// ResolveNetworkConfig resolves network configuration conflicts
func (h *VMStateConflictHandler) ResolveNetworkConfig(local, remote *VMState) (map[string]interface{}, error) {
	// Merge network configurations
	merged := make(map[string]interface{})

	// Copy local config
	for k, v := range local.NetworkConfig {
		merged[k] = v
	}

	// Overlay remote config
	for k, v := range remote.NetworkConfig {
		if _, exists := merged[k]; !exists {
			merged[k] = v
		} else {
			// For existing keys, prefer remote
			merged[k] = v
		}
	}

	return merged, nil
}

// ResolveDiskConfig resolves disk configuration conflicts
func (h *VMStateConflictHandler) ResolveDiskConfig(local, remote *VMState) ([]DiskInfo, error) {
	diskMap := make(map[string]DiskInfo)

	// Add local disks
	for _, disk := range local.DiskConfig {
		diskMap[disk.ID] = disk
	}

	// Merge remote disks
	for _, disk := range remote.DiskConfig {
		if existing, exists := diskMap[disk.ID]; exists {
			// Use larger size
			if disk.SizeGB > existing.SizeGB {
				diskMap[disk.ID] = disk
			}
		} else {
			diskMap[disk.ID] = disk
		}
	}

	// Convert back to slice
	result := make([]DiskInfo, 0, len(diskMap))
	for _, disk := range diskMap {
		result = append(result, disk)
	}

	return result, nil
}

// ResolveSnapshotConflict resolves snapshot conflicts
func (h *VMStateConflictHandler) ResolveSnapshotConflict(local, remote *VMState) (string, error) {
	// Prefer non-empty snapshot ID
	if remote.SnapshotID != "" {
		return remote.SnapshotID, nil
	}
	return local.SnapshotID, nil
}

// ResolveMigrationStatus resolves migration status conflicts
func (h *VMStateConflictHandler) ResolveMigrationStatus(local, remote *VMState) (string, error) {
	// Priority: migrating > preparing > completed > failed
	priority := map[string]int{
		"migrating": 4,
		"preparing": 3,
		"completed": 2,
		"failed":    1,
		"":          0,
	}

	localPriority := priority[local.MigrationStatus]
	remotePriority := priority[remote.MigrationStatus]

	if remotePriority > localPriority {
		return remote.MigrationStatus, nil
	}
	return local.MigrationStatus, nil
}

// MergeVMStates performs comprehensive VM state merge
func (h *VMStateConflictHandler) MergeVMStates(local, remote *VMState) (*VMState, error) {
	merged := &VMState{
		ID:          local.ID,
		LastUpdated: time.Now(),
	}

	// Resolve each component
	var err error

	merged.PowerState, err = h.ResolvePowerStateConflict(local, remote)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve power state: %w", err)
	}

	merged.CPUAllocation, merged.MemoryMB, err = h.ResolveResourceAllocation(local, remote)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve resource allocation: %w", err)
	}

	merged.NetworkConfig, err = h.ResolveNetworkConfig(local, remote)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve network config: %w", err)
	}

	merged.DiskConfig, err = h.ResolveDiskConfig(local, remote)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve disk config: %w", err)
	}

	merged.SnapshotID, err = h.ResolveSnapshotConflict(local, remote)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve snapshot: %w", err)
	}

	merged.MigrationStatus, err = h.ResolveMigrationStatus(local, remote)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve migration status: %w", err)
	}

	// Use remote node if different
	if local.NodeID != remote.NodeID {
		merged.NodeID = remote.NodeID
	} else {
		merged.NodeID = local.NodeID
	}

	return merged, nil
}

// ValidateVMState validates VM state after conflict resolution
func (h *VMStateConflictHandler) ValidateVMState(state *VMState) error {
	if state.ID == "" {
		return fmt.Errorf("VM ID cannot be empty")
	}

	validPowerStates := map[string]bool{
		"running": true,
		"stopped": true,
		"paused":  true,
		"unknown": true,
	}
	if !validPowerStates[state.PowerState] {
		return fmt.Errorf("invalid power state: %s", state.PowerState)
	}

	if state.CPUAllocation <= 0 {
		return fmt.Errorf("CPU allocation must be positive")
	}

	if state.MemoryMB <= 0 {
		return fmt.Errorf("memory allocation must be positive")
	}

	return nil
}
