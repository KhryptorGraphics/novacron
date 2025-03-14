package vm

import (
	"context"
	"time"
)

// VMState represents the state of a VM
type VMState string

// VM states
const (
	VMStateUnknown    VMState = "unknown"
	VMStateCreating   VMState = "creating"
	VMStateRunning    VMState = "running"
	VMStateStopped    VMState = "stopped"
	VMStatePaused     VMState = "paused"
	VMStateError      VMState = "error"
	VMStateRestarting VMState = "restarting"
	VMStateMigrating  VMState = "migrating"
	VMStateDeleting   VMState = "deleting"
)

// VMType represents the type of VM
type VMType string

// VM types
const (
	VMTypeContainer   VMType = "container"    // Docker-based containers
	VMTypeContainerd  VMType = "containerd"   // Containerd-based containers
	VMTypeKVM         VMType = "kvm"          // QEMU/KVM virtual machines
	VMTypeProcess     VMType = "process"      // Process-based lightweight VMs
)

// VMSpec represents VM specifications
type VMSpec struct {
	VCPU     int               `json:"vcpu"`
	MemoryMB int               `json:"memory_mb"`
	DiskMB   int               `json:"disk_mb"`
	Networks []VMNetworkSpec   `json:"networks"`
	Volumes  []VMVolumeSpec    `json:"volumes"`
	Env      map[string]string `json:"env"`
	Labels   map[string]string `json:"labels"`
	Type     VMType            `json:"type"`
	Image    string            `json:"image"`
}

// VMNetworkSpec represents VM network specifications
type VMNetworkSpec struct {
	NetworkID  string   `json:"network_id"`
	IPAddress  string   `json:"ip_address,omitempty"`
	MACAddress string   `json:"mac_address,omitempty"`
	DNS        []string `json:"dns,omitempty"`
}

// VMVolumeSpec represents VM volume specifications
type VMVolumeSpec struct {
	VolumeID string `json:"volume_id"`
	Path     string `json:"path"`
	ReadOnly bool   `json:"read_only"`
	SizeMB   int    `json:"size_mb"`
}

// VM represents a virtual machine
type VM struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	Spec         VMSpec            `json:"spec"`
	State        VMState           `json:"state"`
	NodeID       string            `json:"node_id"`
	CreatedAt    time.Time         `json:"created_at"`
	UpdatedAt    time.Time         `json:"updated_at"`
	StartedAt    time.Time         `json:"started_at"`
	StoppedAt    time.Time         `json:"stopped_at"`
	Tags         map[string]string `json:"tags"`
	Owner        string            `json:"owner"`
	ResourceID   string            `json:"resource_id"`
	NetworkInfo  []VMNetworkInfo   `json:"network_info"`
	StorageInfo  []VMStorageInfo   `json:"storage_info"`
	ProcessInfo  VMProcessInfo     `json:"process_info"`
	ErrorMessage string            `json:"error_message"`
}

// VMNetworkInfo represents VM network information
type VMNetworkInfo struct {
	NetworkID     string    `json:"network_id"`
	IPAddress     string    `json:"ip_address"`
	MACAddress    string    `json:"mac_address"`
	DNS           []string  `json:"dns"`
	RXBytes       int64     `json:"rx_bytes"`
	TXBytes       int64     `json:"tx_bytes"`
	RXPackets     int64     `json:"rx_packets"`
	TXPackets     int64     `json:"tx_packets"`
	RXErrors      int64     `json:"rx_errors"`
	TXErrors      int64     `json:"tx_errors"`
	LastUpdatedAt time.Time `json:"last_updated_at"`
}

// VMStorageInfo represents VM storage information
type VMStorageInfo struct {
	VolumeID      string    `json:"volume_id"`
	Path          string    `json:"path"`
	ReadOnly      bool      `json:"read_only"`
	SizeMB        int       `json:"size_mb"`
	UsedMB        int       `json:"used_mb"`
	ReadOps       int64     `json:"read_ops"`
	WriteOps      int64     `json:"write_ops"`
	ReadBytes     int64     `json:"read_bytes"`
	WriteBytes    int64     `json:"write_bytes"`
	LastUpdatedAt time.Time `json:"last_updated_at"`
}

// VMProcessInfo represents VM process information
type VMProcessInfo struct {
	PID             int       `json:"pid"`
	MemoryUsageMB   int       `json:"memory_usage_mb"`
	CPUUsagePercent float64   `json:"cpu_usage_percent"`
	ThreadCount     int       `json:"thread_count"`
	StartTime       time.Time `json:"start_time"`
	LastUpdatedAt   time.Time `json:"last_updated_at"`
}

// CreateVMRequest represents a request to create a VM
type CreateVMRequest struct {
	Name          string            `json:"name"`
	Spec          VMSpec            `json:"spec"`
	Tags          map[string]string `json:"tags"`
	Owner         string            `json:"owner"`
	PreferredNode string            `json:"preferred_node,omitempty"`
}

// VMOperation represents VM operation types
type VMOperation string

// VM operations
const (
	VMOperationStart     VMOperation = "start"
	VMOperationStop      VMOperation = "stop"
	VMOperationRestart   VMOperation = "restart"
	VMOperationPause     VMOperation = "pause"
	VMOperationResume    VMOperation = "resume"
	VMOperationDelete    VMOperation = "delete"
	VMOperationMigrate   VMOperation = "migrate"
	VMOperationSnapshot  VMOperation = "snapshot"
)

// VMOperationRequest represents a request to perform an operation on a VM
type VMOperationRequest struct {
	Operation VMOperation       `json:"operation"`
	VMID      string            `json:"vm_id"`
	Params    map[string]string `json:"params,omitempty"`
}

// VMOperationResponse represents a response to a VM operation
type VMOperationResponse struct {
	Success      bool   `json:"success"`
	ErrorMessage string `json:"error_message,omitempty"`
	VM           *VM    `json:"vm,omitempty"`
}

// VMEventType represents VM event types
type VMEventType string

// VM event types
const (
	VMEventCreated   VMEventType = "created"
	VMEventDeleted   VMEventType = "deleted"
	VMEventStarted   VMEventType = "started"
	VMEventStopped   VMEventType = "stopped"
	VMEventPaused    VMEventType = "paused"
	VMEventResumed   VMEventType = "resumed"
	VMEventMigrated  VMEventType = "migrated"
	VMEventError     VMEventType = "error"
	VMEventUpdated   VMEventType = "updated"
	VMEventSnapshot  VMEventType = "snapshot"
)

// VMEvent represents events related to VMs
type VMEvent struct {
	Type      VMEventType `json:"type"`
	VM        VM          `json:"vm"`
	Timestamp time.Time   `json:"timestamp"`
	NodeID    string      `json:"node_id"`
	Message   string      `json:"message,omitempty"`
}

// VMManagerEventListener is a callback for VM events
type VMManagerEventListener func(event VMEvent)

// VMDriver handles VM operations for a specific type of VM
type VMDriver interface {
	// Create creates a new VM
	Create(ctx context.Context, spec VMSpec) (string, error)
	
	// Start starts a VM
	Start(ctx context.Context, vmID string) error
	
	// Stop stops a VM
	Stop(ctx context.Context, vmID string) error
	
	// Delete deletes a VM
	Delete(ctx context.Context, vmID string) error
	
	// GetStatus gets the status of a VM
	GetStatus(ctx context.Context, vmID string) (VMState, error)
	
	// GetInfo gets information about a VM
	GetInfo(ctx context.Context, vmID string) (*VM, error)
}

// VMDriverFactory creates VM drivers for a specific VM type
type VMDriverFactory func(vmType VMType) (VMDriver, error)

// VMManagerConfig contains configuration for the VM manager
type VMManagerConfig struct {
	// UpdateInterval is the interval at which VMs are updated
	UpdateInterval time.Duration
	
	// CleanupInterval is the interval at which expired VMs are cleaned up
	CleanupInterval time.Duration
	
	// DefaultVMType is the default VM type
	DefaultVMType VMType
	
	// RetentionPeriod is the period after which deleted VMs are cleaned up
	RetentionPeriod time.Duration
}

// DefaultVMManagerConfig returns a default VM manager configuration
func DefaultVMManagerConfig() VMManagerConfig {
	return VMManagerConfig{
		UpdateInterval:  30 * time.Second,
		CleanupInterval: 5 * time.Minute,
		DefaultVMType:   VMTypeContainer,
		RetentionPeriod: 24 * time.Hour,
	}
}
