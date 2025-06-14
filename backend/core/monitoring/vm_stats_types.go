// This file contains type definitions for the VM telemetry system
package monitoring

import "time"

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

// VMStats contains all stats for a VM
type VMStats struct {
	// VMID is the unique identifier for the VM
	VMID string

	// CPU statistics
	CPU VMCPUStats

	// Memory statistics
	Memory VMMemoryStats

	// Disk statistics
	Disks []VMDiskStats

	// Network statistics
	Networks []VMNetworkStats

	// Process statistics if available
	Processes []VMProcessStats

	// Timestamp when the stats were collected
	Timestamp time.Time
}

// VMCPUStats contains CPU metrics for a VM
type VMCPUStats struct {
	// Usage percentage (0-100)
	Usage float64

	// Usage per core if available
	CoreUsage []float64

	// Number of vCPUs
	NumCPUs int

	// Time spent in steal (hypervisor overhead)
	StealTime float64

	// Ready time (time VM was ready but couldn't get CPU time)
	ReadyTime float64

	// System time percentage
	SystemTime float64

	// User time percentage
	UserTime float64

	// IO wait time percentage
	IOWaitTime float64
}

// VMMemoryStats contains memory metrics for a VM
type VMMemoryStats struct {
	// Total memory allocated to the VM in bytes
	Total int64

	// Used memory in bytes
	Used int64

	// Used percentage (0-100)
	UsagePercent float64

	// Free memory in bytes
	Free int64

	// Swap usage in bytes
	SwapUsed int64

	// Swap total in bytes
	SwapTotal int64

	// Page faults per second
	PageFaults float64

	// Major page faults per second
	MajorPageFaults float64

	// Ballooning target if using dynamic memory
	BalloonTarget int64

	// Current balloon size if using dynamic memory
	BalloonCurrent int64
}

// VMDiskStats contains disk metrics for a VM disk
type VMDiskStats struct {
	// Disk identifier
	DiskID string

	// Path or name
	Path string

	// Total size in bytes
	Size int64

	// Used space in bytes
	Used int64

	// Used percentage (0-100)
	UsagePercent float64

	// Read operations per second
	ReadIOPS float64

	// Write operations per second
	WriteIOPS float64

	// Read throughput in bytes per second
	ReadThroughput float64

	// Write throughput in bytes per second
	WriteThroughput float64

	// Average read latency in milliseconds
	ReadLatency float64

	// Average write latency in milliseconds
	WriteLatency float64

	// Disk type (e.g., system, data)
	Type string
}

// VMNetworkStats contains network metrics for a VM interface
type VMNetworkStats struct {
	// Interface identifier
	InterfaceID string

	// Interface name
	Name string

	// Bytes received per second
	RxBytes float64

	// Bytes transmitted per second
	TxBytes float64

	// Packets received per second
	RxPackets float64

	// Packets transmitted per second
	TxPackets float64

	// Dropped packets received
	RxDropped float64

	// Dropped packets transmitted
	TxDropped float64

	// Error packets received
	RxErrors float64

	// Error packets transmitted
	TxErrors float64
}

// VMProcessStats contains metrics for a process running in a VM
type VMProcessStats struct {
	// Process ID
	PID int64

	// Process name
	Name string

	// Process command line
	Command string

	// CPU usage percentage
	CPUUsage float64

	// Memory usage percentage
	MemoryPercent float64

	// Memory usage in bytes
	MemoryUsage int64

	// Read operations per second
	ReadIOPS float64

	// Write operations per second
	WriteIOPS float64

	// Read throughput in bytes per second
	ReadThroughput float64

	// Write throughput in bytes per second
	WriteThroughput float64

	// Open file descriptors
	OpenFiles int64

	// Running time in seconds
	RunTime float64
}