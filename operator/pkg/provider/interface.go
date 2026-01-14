package provider

import (
	"context"
	
	novacronv1alpha1 "github.com/novacron/operator/api/v1alpha1"
)

// VMManager interface for managing VMs
type VMManager interface {
	// CreateVM creates a new VM on the specified node
	CreateVM(ctx context.Context, node string, spec novacronv1alpha1.VMSpec) (string, error)
	
	// DeleteVM deletes a VM
	DeleteVM(ctx context.Context, vmID string) error
	
	// StartVM starts a stopped VM
	StartVM(ctx context.Context, vmID string) error
	
	// StopVM stops a running VM
	StopVM(ctx context.Context, vmID string) error
	
	// GetVMStatus gets the current status of a VM
	GetVMStatus(ctx context.Context, vmID string) (*VMStatus, error)
	
	// GetVMResourceUsage gets resource usage of a VM
	GetVMResourceUsage(ctx context.Context, vmID string) (*ResourceUsage, error)
	
	// CheckVMHealth checks if a VM is healthy
	CheckVMHealth(ctx context.Context, vmID string) (bool, error)
	
	// GetMigrationProgress gets migration progress
	GetMigrationProgress(ctx context.Context, vmID string) (*MigrationProgress, error)
	
	// MigrateVM migrates a VM to another node
	MigrateVM(ctx context.Context, vmID string, targetNode string) error
}

// VMStatus represents VM status
type VMStatus struct {
	State       string
	IPAddresses []string
	Message     string
}

// ResourceUsage represents resource usage
type ResourceUsage struct {
	CPUUsed           string
	CPUAvailable      string
	CPUPercentage     float64
	MemoryUsed        string
	MemoryAvailable   string
	MemoryPercentage  float64
	DiskUsed          string
	DiskAvailable     string
	DiskPercentage    float64
}

// MigrationProgress represents migration progress
type MigrationProgress struct {
	State      string
	Percentage int32
	Error      string
}