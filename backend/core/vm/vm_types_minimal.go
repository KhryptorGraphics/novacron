package vm

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// VMType represents the type of VM
type VMType string

const (
	// Legacy/existing types
	VMTypeKVM            VMType = "kvm"
	VMTypeContainer      VMType = "container"
	VMTypeContainerd     VMType = "containerd"
	VMTypeKataContainers VMType = "kata-containers"
	VMTypeProcess        VMType = "process"
	
	// Phase 2: Comprehensive Hypervisor Integration
	VMTypeVMware         VMType = "vmware"
	VMTypeVSphere        VMType = "vsphere"
	VMTypeHyperV         VMType = "hyperv"
	VMTypeXen            VMType = "xen"
	VMTypeXenServer      VMType = "xenserver"
	VMTypeProxmox        VMType = "proxmox"
	VMTypeProxmoxVE      VMType = "proxmox-ve"
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
	SupportsLiveMigration() bool
	SupportsHotPlug() bool
	SupportsGPUPassthrough() bool
	SupportsSRIOV() bool
	SupportsNUMA() bool
	
	// Capability detection
	GetCapabilities(ctx context.Context) (*HypervisorCapabilities, error)
	GetHypervisorInfo(ctx context.Context) (*HypervisorInfo, error)

	Pause(ctx context.Context, vmID string) error
	Resume(ctx context.Context, vmID string) error
	Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error)
	Migrate(ctx context.Context, vmID, target string, params map[string]string) error
	
	// Advanced operations for Phase 2
	HotPlugDevice(ctx context.Context, vmID string, device *DeviceConfig) error
	HotUnplugDevice(ctx context.Context, vmID string, deviceID string) error
	ConfigureCPUPinning(ctx context.Context, vmID string, pinning *CPUPinningConfig) error
	ConfigureNUMA(ctx context.Context, vmID string, topology *NUMATopology) error
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

// VMUpdateSpec represents specification for updating VM configuration
type VMUpdateSpec struct {
	Name   *string             `json:"name,omitempty"`    // VM name update
	CPU    *int                `json:"cpu,omitempty"`     // CPU shares update
	Memory *int64              `json:"memory,omitempty"`  // Memory update in MB
	Disk   *int64              `json:"disk,omitempty"`    // Disk size update in GB
	Tags   map[string]string   `json:"tags,omitempty"`    // Tag updates
}

// BackupVerificationResult moved to backup package as VerificationResult

// MigrationStatus defined in vm_migration_types.go to avoid duplicates

// MigrationProgress represents migration progress information
type MigrationProgress struct {
	ID          string          `json:"id"`
	VMID        string          `json:"vm_id"`
	SourceNode  string          `json:"source_node"`
	TargetNode  string          `json:"target_node"`
	Status      MigrationStatus `json:"status"`
	Percentage  int             `json:"percentage"`
	StartedAt   time.Time       `json:"started_at"`
	CompletedAt *time.Time      `json:"completed_at,omitempty"`
	Details     map[string]interface{} `json:"details,omitempty"`
}

// Enhanced error types for better error handling
var (
	ErrVMNotFound              = errors.New("VM not found")
	ErrOperationNotSupported   = errors.New("operation not supported by driver")
	ErrInvalidVMState         = errors.New("invalid VM state for operation")
	ErrBackupNotFound         = errors.New("backup not found")
	ErrBackupCorrupted        = errors.New("backup is corrupted")
)

type VMError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Cause   error  `json:"-"`
}

func (e *VMError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%s: %s (caused by: %v)", e.Code, e.Message, e.Cause)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

func (e *VMError) Unwrap() error {
	return e.Cause
}

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

// HypervisorCapabilities represents the capabilities of a hypervisor
type HypervisorCapabilities struct {
	Type                   VMType   `json:"type"`
	Version               string   `json:"version"`
	SupportsPause         bool     `json:"supports_pause"`
	SupportsResume        bool     `json:"supports_resume"`
	SupportsSnapshot      bool     `json:"supports_snapshot"`
	SupportsMigrate       bool     `json:"supports_migrate"`
	SupportsLiveMigration bool     `json:"supports_live_migration"`
	SupportsHotPlug       bool     `json:"supports_hot_plug"`
	SupportsGPUPassthrough bool    `json:"supports_gpu_passthrough"`
	SupportsSRIOV         bool     `json:"supports_sriov"`
	SupportsNUMA          bool     `json:"supports_numa"`
	MaxVCPUs              int      `json:"max_vcpus"`
	MaxMemoryMB           int64    `json:"max_memory_mb"`
	SupportedFeatures     []string `json:"supported_features"`
	HardwareExtensions    []string `json:"hardware_extensions"`
}

// HypervisorInfo represents information about the hypervisor
type HypervisorInfo struct {
	Type            VMType                 `json:"type"`
	Version         string                 `json:"version"`
	ConnectionURI   string                 `json:"connection_uri"`
	Hostname        string                 `json:"hostname"`
	CPUModel        string                 `json:"cpu_model"`
	CPUCores        int                    `json:"cpu_cores"`
	MemoryMB        int64                  `json:"memory_mb"`
	Virtualization  string                 `json:"virtualization"` // VT-x, AMD-V, etc.
	IOMMUEnabled    bool                   `json:"iommu_enabled"`
	NUMANodes       int                    `json:"numa_nodes"`
	GPUDevices      []GPUDevice            `json:"gpu_devices"`
	NetworkDevices  []NetworkDevice        `json:"network_devices"`
	StorageDevices  []StorageDevice        `json:"storage_devices"`
	ActiveVMs       int                    `json:"active_vms"`
	Capabilities    *HypervisorCapabilities `json:"capabilities"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// DeviceConfig represents a device configuration for hot-plug operations
type DeviceConfig struct {
	Type       string                 `json:"type"` // cpu, memory, disk, network, gpu, usb
	Name       string                 `json:"name"`
	Address    string                 `json:"address,omitempty"`
	Bus        string                 `json:"bus,omitempty"`
	Slot       string                 `json:"slot,omitempty"`
	Parameters map[string]interface{} `json:"parameters"`
}

// CPUPinningConfig represents CPU pinning configuration
type CPUPinningConfig struct {
	VCPUs       []VCPUPinning          `json:"vcpus"`
	IOThreads   []IOThreadPinning      `json:"io_threads,omitempty"`
	EmulatorPin string                 `json:"emulator_pin,omitempty"`
	Options     map[string]interface{} `json:"options,omitempty"`
}

// VCPUPinning represents virtual CPU to physical CPU mapping
type VCPUPinning struct {
	VCPU        int    `json:"vcpu"`
	CPUSet      string `json:"cpuset"` // e.g., "0-3,8,9"
	Policy      string `json:"policy,omitempty"`
}

// IOThreadPinning represents I/O thread to CPU mapping
type IOThreadPinning struct {
	IOThread int    `json:"iothread"`
	CPUSet   string `json:"cpuset"`
}

// NUMATopology represents NUMA topology configuration
type NUMATopology struct {
	Nodes   []NUMANode             `json:"nodes"`
	Mode    string                 `json:"mode"`    // strict, preferred, interleave
	Options map[string]interface{} `json:"options,omitempty"`
}

// NUMANode represents a NUMA node configuration
type NUMANode struct {
	ID       int      `json:"id"`
	CPUs     string   `json:"cpus"`     // CPU set, e.g., "0-3"
	MemoryMB int64    `json:"memory_mb"`
	Distance []int    `json:"distance,omitempty"`
}

// GPUDevice represents a GPU device
type GPUDevice struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Vendor       string `json:"vendor"`
	Model        string `json:"model"`
	MemoryMB     int64  `json:"memory_mb"`
	PCIAddress   string `json:"pci_address"`
	IOMMUGroup   int    `json:"iommu_group"`
	Virtualizable bool   `json:"virtualizable"`
	InUse        bool   `json:"in_use"`
}

// NetworkDevice represents a network device
type NetworkDevice struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Type       string `json:"type"` // ethernet, infiniband, etc.
	MACAddress string `json:"mac_address"`
	PCIAddress string `json:"pci_address"`
	IOMMUGroup int    `json:"iommu_group"`
	SRIOV      bool   `json:"sriov"`
	VFCount    int    `json:"vf_count,omitempty"`
	InUse      bool   `json:"in_use"`
}

// StorageDevice represents a storage device
type StorageDevice struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Type         string `json:"type"` // ssd, nvme, hdd, etc.
	SizeGB       int64  `json:"size_gb"`
	Path         string `json:"path"`
	ReadIOPS     int64  `json:"read_iops,omitempty"`
	WriteIOPS    int64  `json:"write_iops,omitempty"`
	Passthrough  bool   `json:"passthrough"`
	InUse        bool   `json:"in_use"`
}
