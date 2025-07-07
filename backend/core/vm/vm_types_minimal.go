package vm

import (
	"context"
	"time"
)

// VMType represents the type of VM
type VMType string

const (
	VMTypeKVM       VMType = "kvm"
	VMTypeContainer VMType = "container"
	VMTypeContainerd VMType = "containerd"
	VMTypeProcess   VMType = "process"
)

// VMDriverFactory is a function that creates a VM driver
type VMDriverFactory func(config VMConfig) (VMDriver, error)

// VMDriver interface for different VM implementations
type VMDriver interface {
	Create(ctx context.Context, config VMConfig) (string, error)
	Start(ctx context.Context, vmID string) error
	Stop(ctx context.Context, vmID string) error
	Delete(ctx context.Context, vmID string) error
	GetStatus(ctx context.Context, vmID string) (VMState, error)
	GetInfo(ctx context.Context, vmID string) (*VMInfo, error)
	GetMetrics(ctx context.Context, vmID string) (*VMInfo, error)
	ListVMs(ctx context.Context) ([]VMInfo, error)

	// Optional operations
	SupportsPause() bool
	SupportsResume() bool
	SupportsSnapshot() bool
	SupportsMigrate() bool

	Pause(ctx context.Context, vmID string) error
	Resume(ctx context.Context, vmID string) error
	Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error)
	Migrate(ctx context.Context, vmID, target string, params map[string]string) error
}

// VMManagerInterface for migration
type VMManagerInterface interface {
	GetVM(vmID string) (*VM, error)
	// Add other methods as needed
}

// NodeManagerInterface for migration
type NodeManagerInterface interface {
	GetNode(nodeID string) (*Node, error)
	// Add other methods as needed
}

// StorageManagerInterface for migration
type StorageManagerInterface interface {
	// Add storage methods as needed
}

// NetworkManagerInterface for migration
type NetworkManagerInterface interface {
	// Add network methods as needed
}

// Alias VMState to State for compatibility
type VMState = State

// Node represents a compute node in the cluster (defined here to avoid import cycles)
// Note: This may conflict with vm_migration_types.go - use conditional compilation or merge definitions

// These types are defined in other files to avoid duplication:
// - MigrationType (migration_manager.go) 
// - VMConfig (vm.go)
// - VMManagerConfig (vm_manager.go)
// - VMSchedulerConfig (vm_manager.go)

// CreateVMRequest represents a request to create a VM
type CreateVMRequest struct {
	Name string            `json:"name"`
	Spec VMConfig          `json:"spec"`
	Tags map[string]string `json:"tags"`
}

// VMOperationRequest represents a request for VM operations
type VMOperationRequest struct {
	VMID      string                 `json:"vm_id"`
	Operation string                 `json:"operation"`
	Params    map[string]interface{} `json:"params"`
}

// VMOperationResponse represents a response to a VM operation
type VMOperationResponse struct {
	Success      bool              `json:"success"`
	ErrorMessage string            `json:"error_message,omitempty"`
	VM           *VM               `json:"vm,omitempty"`
	Data         map[string]string `json:"data,omitempty"`
}