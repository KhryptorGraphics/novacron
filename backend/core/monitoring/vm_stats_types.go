// This file contains type definitions for the VM telemetry system
package monitoring

// VMState represents the current state of a VM
type VMState int

const (
	// VMStateUnknown indicates the VM state is unknown
	VMStateUnknown VMState = iota

	// VMStateRunning indicates the VM is running
	VMStateRunning

	// VMStateStopped indicates the VM is stopped
	VMStateStopped

	// VMStateStopping indicates the VM is in the process of stopping
	VMStateStopping

	// VMStateStarting indicates the VM is in the process of starting
	VMStateStarting

	// VMStateTerminated indicates the VM is terminated or deleted
	VMStateTerminated

	// VMStatePaused indicates the VM is paused
	VMStatePaused

	// VMStateSuspended indicates the VM is suspended
	VMStateSuspended
)

// String returns a string representation of the VM state
func (s VMState) String() string {
	switch s {
	case VMStateRunning:
		return "running"
	case VMStateStopped:
		return "stopped"
	case VMStateStopping:
		return "stopping"
	case VMStateStarting:
		return "starting"
	case VMStateTerminated:
		return "terminated"
	case VMStatePaused:
		return "paused"
	case VMStateSuspended:
		return "suspended"
	default:
		return "unknown"
	}
}

// CloudVMStats contains all telemetry data for a virtual machine from cloud providers
// This is separate from the internal VMStats used by the vm_telemetry_collector
type CloudVMStats struct {
	// VMID is the unique identifier for the VM
	VMID string

	// VMName is the display name of the VM
	VMName string

	// HostSystem is the type of hypervisor or platform running the VM
	HostSystem string

	// State represents the current state of the VM
	State VMState

	// CPU contains CPU-related metrics
	CPU *CloudCPUStats

	// Memory contains memory-related metrics
	Memory *CloudMemoryStats

	// Disks contains disk-related metrics keyed by device name
	Disks map[string]*CloudDiskStats

	// Networks contains network-related metrics keyed by interface name
	Networks map[string]*CloudNetworkStats

	// Processes contains process-related metrics if process monitoring is enabled
	Processes map[string]*CloudProcessStats

	// Metadata contains additional provider-specific information about the VM
	Metadata map[string]string

	// Tags contains user-defined tags associated with the VM
	Tags map[string]string

	// Timestamp is the time when these stats were collected
	Timestamp string
}

// CloudCPUStats contains CPU metrics for a VM
type CloudCPUStats struct {
	// Usage is the overall CPU usage percentage (0-100)
	Usage float64

	// SystemTime is the percentage of CPU time spent in system mode
	SystemTime float64

	// UserTime is the percentage of CPU time spent in user mode
	UserTime float64

	// IOWaitTime is the percentage of CPU time spent waiting for I/O
	IOWaitTime float64

	// StealTime is the percentage of CPU time stolen by the hypervisor
	StealTime float64

	// ReadyTime is the percentage of time the VM was ready but waiting for physical CPU
	ReadyTime float64

	// CoreUsage contains per-core CPU usage if available
	CoreUsage map[string]float64
}

// CloudMemoryStats contains memory metrics for a VM
type CloudMemoryStats struct {
	// Usage is the memory usage percentage (0-100)
	Usage float64

	// UsagePercent is an alias for Usage for compatibility
	UsagePercent float64

	// Used is the amount of memory used in bytes
	Used float64

	// Total is the total amount of memory allocated to the VM in bytes
	Total float64

	// Free is the amount of free memory in bytes
	Free float64

	// SwapUsed is the amount of swap memory used in bytes
	SwapUsed float64

	// SwapTotal is the total amount of swap memory available in bytes
	SwapTotal float64

	// PageFaults is the number of page faults per second
	PageFaults float64

	// MajorPageFaults is the number of major page faults per second
	MajorPageFaults float64

	// BalloonTarget is the target memory allocation from the hypervisor
	BalloonTarget float64

	// BalloonCurrent is the current ballooned memory
	BalloonCurrent float64

	// BalloonedMemory is the amount of memory ballooned by the hypervisor
	BalloonedMemory float64
}

// CloudDiskStats contains disk metrics for a VM disk
type CloudDiskStats struct {
	// DiskID is a unique identifier for this disk
	DiskID string

	// Path is the device path or mount point
	Path string

	// Type is the disk type (system, data, etc.)
	Type string

	// Usage is the disk usage percentage (0-100)
	Usage float64

	// UsagePercent is an alias for Usage for compatibility
	UsagePercent float64

	// Used is the amount of disk space used in bytes
	Used float64

	// Total is the total disk space in bytes
	Total float64

	// Size is an alias for Total for compatibility
	Size float64

	// ReadIOPS is the number of read operations per second
	ReadIOPS float64

	// WriteIOPS is the number of write operations per second
	WriteIOPS float64

	// ReadThroughput is the disk read throughput in bytes per second
	ReadThroughput float64

	// WriteThroughput is the disk write throughput in bytes per second
	WriteThroughput float64

	// ReadLatency is the average disk read latency in milliseconds
	ReadLatency float64

	// WriteLatency is the average disk write latency in milliseconds
	WriteLatency float64

	// QueueDepth is the average disk queue depth
	QueueDepth float64
}

// CloudNetworkStats contains network metrics for a VM network interface
type CloudNetworkStats struct {
	// InterfaceID is a unique identifier for this interface
	InterfaceID string

	// Name is the interface name
	Name string

	// RxBytes is the number of bytes received per second
	RxBytes float64

	// TxBytes is the number of bytes transmitted per second
	TxBytes float64

	// RxPackets is the number of packets received per second
	RxPackets float64

	// TxPackets is the number of packets transmitted per second
	TxPackets float64

	// RxErrors is the number of receive errors per second
	RxErrors float64

	// TxErrors is the number of transmit errors per second
	TxErrors float64

	// RxDropped is the number of packets dropped on receive per second
	RxDropped float64

	// TxDropped is the number of packets dropped on transmit per second
	TxDropped float64
}

// CloudProcessStats contains metrics for a process running in a VM
type CloudProcessStats struct {
	// PID is the process ID
	PID int

	// Name is the process name
	Name string

	// Command is the full command line
	Command string

	// CPU is the CPU usage percentage for this process
	CPU float64

	// CPUUsage is an alias for CPU for compatibility
	CPUUsage float64

	// Memory is the memory usage percentage for this process
	Memory float64

	// MemoryUsage is an alias for MemoryUsed for compatibility
	MemoryUsage float64

	// MemoryPercent is an alias for Memory for compatibility
	MemoryPercent float64

	// MemoryUsed is the amount of memory used by this process in bytes
	MemoryUsed float64

	// ReadIOPS is the disk read operations per second
	ReadIOPS float64

	// WriteIOPS is the disk write operations per second
	WriteIOPS float64

	// ReadThroughput is the disk read bytes per second
	ReadThroughput float64

	// WriteThroughput is the disk write bytes per second
	WriteThroughput float64

	// DiskRead is the disk read rate in bytes per second
	DiskRead float64

	// DiskWrite is the disk write rate in bytes per second
	DiskWrite float64

	// OpenFiles is the number of open file descriptors
	OpenFiles int

	// StartTime is the process start time as a Unix timestamp
	StartTime int64

	// RunTime is the running time in seconds
	RunTime float64

	// Priority is the process priority
	Priority int

	// State is the process state (running, sleeping, etc.)
	State string
}
