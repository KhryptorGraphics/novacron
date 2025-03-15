package cloud

import (
	"context"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// AWSVMManager implements the VMManagerInterface for AWS EC2 instances
type AWSVMManager struct {
	provider *AWSProvider
}

// NewAWSVMManager creates a new VM manager for AWS
func NewAWSVMManager(provider *AWSProvider) *AWSVMManager {
	return &AWSVMManager{
		provider: provider,
	}
}

// GetVMs returns a list of all VM IDs from AWS
func (m *AWSVMManager) GetVMs(ctx context.Context) ([]string, error) {
	instances, err := m.provider.ListInstances(ctx)
	if err != nil {
		return nil, fmt.Errorf("error listing AWS instances: %w", err)
	}

	var vmIDs []string
	for _, instance := range instances {
		vmIDs = append(vmIDs, instance.ID)
	}

	return vmIDs, nil
}

// GetVMStats retrieves stats for a specific VM from AWS
func (m *AWSVMManager) GetVMStats(
	ctx context.Context,
	vmID string,
	detailLevel monitoring.VMMetricDetailLevel,
) (*monitoring.VMStats, error) {
	// Get instance details
	instance, err := m.provider.GetInstance(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("error getting AWS instance %s: %w", vmID, err)
	}

	// Get instance metrics from CloudWatch
	metrics, err := m.provider.GetInstanceMetrics(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("error getting AWS instance metrics for %s: %w", vmID, err)
	}

	// Convert metrics to VMStats
	stats := &monitoring.VMStats{
		VMID:      vmID,
		Timestamp: time.Now(),
	}

	// Initialize CPU stats
	cpuValue, ok := metrics["CPUUtilization"]
	if ok {
		stats.CPU = monitoring.VMCPUStats{
			Usage: cpuValue,
		}
	}

	// Initialize memory stats
	memoryUsedValue, memoryOk := metrics["MemoryUtilization"]
	if memoryOk {
		stats.Memory = monitoring.VMMemoryStats{
			UsagePercent: memoryUsedValue,
		}
	}

	// Initialize disk stats
	diskReadOps, diskReadOpsOk := metrics["DiskReadOps"]
	diskWriteOps, diskWriteOpsOk := metrics["DiskWriteOps"]
	diskReadBytes, diskReadBytesOk := metrics["DiskReadBytes"]
	diskWriteBytes, diskWriteBytesOk := metrics["DiskWriteBytes"]

	if diskReadOpsOk || diskWriteOpsOk || diskReadBytesOk || diskWriteBytesOk {
		stats.Disks = []monitoring.VMDiskStats{
			{
				DiskID:          "root",
				Path:            "/dev/sda1",
				Type:            "ebs",
				ReadIOPS:        diskReadOps,
				WriteIOPS:       diskWriteOps,
				ReadThroughput:  diskReadBytes,
				WriteThroughput: diskWriteBytes,
			},
		}
	}

	// Initialize network stats
	netInBytes, netInOk := metrics["NetworkIn"]
	netOutBytes, netOutOk := metrics["NetworkOut"]
	netPacketsIn, netPacketsInOk := metrics["NetworkPacketsIn"]
	netPacketsOut, netPacketsOutOk := metrics["NetworkPacketsOut"]

	if netInOk || netOutOk || netPacketsInOk || netPacketsOutOk {
		stats.Networks = []monitoring.VMNetworkStats{
			{
				InterfaceID: "eth0",
				Name:        "primary",
				RxBytes:     netInBytes,
				TxBytes:     netOutBytes,
				RxPackets:   netPacketsIn,
				TxPackets:   netPacketsOut,
			},
		}
	}

	return stats, nil
}

// AzureVMManager implements the VMManagerInterface for Azure VMs
type AzureVMManager struct {
	provider *AzureProvider
}

// NewAzureVMManager creates a new VM manager for Azure
func NewAzureVMManager(provider *AzureProvider) *AzureVMManager {
	return &AzureVMManager{
		provider: provider,
	}
}

// GetVMs returns a list of all VM IDs from Azure
func (m *AzureVMManager) GetVMs(ctx context.Context) ([]string, error) {
	vms, err := m.provider.ListVirtualMachines(ctx)
	if err != nil {
		return nil, fmt.Errorf("error listing Azure VMs: %w", err)
	}

	var vmIDs []string
	for _, vm := range vms {
		vmIDs = append(vmIDs, vm.ID)
	}

	return vmIDs, nil
}

// GetVMStats retrieves stats for a specific VM from Azure
func (m *AzureVMManager) GetVMStats(
	ctx context.Context,
	vmID string,
	detailLevel monitoring.VMMetricDetailLevel,
) (*monitoring.VMStats, error) {
	// Get VM details
	vm, err := m.provider.GetVirtualMachine(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("error getting Azure VM %s: %w", vmID, err)
	}

	// Get VM metrics
	metrics, err := m.provider.GetVMMetrics(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("error getting Azure VM metrics for %s: %w", vmID, err)
	}

	// Convert metrics to VMStats
	stats := &monitoring.VMStats{
		VMID:      vmID,
		Timestamp: time.Now(),
	}

	// Initialize CPU stats
	cpuValue, ok := metrics["Percentage CPU"]
	if ok {
		stats.CPU = monitoring.VMCPUStats{
			Usage: cpuValue,
		}
	}

	// Initialize memory stats
	memoryUsedValue, memoryOk := metrics["Available Memory Bytes"]
	memoryTotalValue, memoryTotalOk := metrics["VM Memory"]

	if memoryOk && memoryTotalOk && memoryTotalValue > 0 {
		memoryUsagePercent := 100 * (1 - (memoryUsedValue / memoryTotalValue))
		stats.Memory = monitoring.VMMemoryStats{
			UsagePercent: memoryUsagePercent,
			Used:         int64(memoryTotalValue - memoryUsedValue),
			Total:        int64(memoryTotalValue),
			Free:         int64(memoryUsedValue),
		}
	}

	// Initialize disk stats
	diskReadOps, diskReadOpsOk := metrics["Disk Read Operations/Sec"]
	diskWriteOps, diskWriteOpsOk := metrics["Disk Write Operations/Sec"]
	diskReadBytes, diskReadBytesOk := metrics["Disk Read Bytes/Sec"]
	diskWriteBytes, diskWriteBytesOk := metrics["Disk Write Bytes/Sec"]

	if diskReadOpsOk || diskWriteOpsOk || diskReadBytesOk || diskWriteBytesOk {
		stats.Disks = []monitoring.VMDiskStats{
			{
				DiskID:          "os",
				Path:            "C:",
				Type:            "managed",
				ReadIOPS:        diskReadOps,
				WriteIOPS:       diskWriteOps,
				ReadThroughput:  diskReadBytes,
				WriteThroughput: diskWriteBytes,
			},
		}
	}

	// Initialize network stats
	netInBytes, netInOk := metrics["Network In Total"]
	netOutBytes, netOutOk := metrics["Network Out Total"]

	if netInOk || netOutOk {
		stats.Networks = []monitoring.VMNetworkStats{
			{
				InterfaceID: "eth0",
				Name:        "primary",
				RxBytes:     netInBytes,
				TxBytes:     netOutBytes,
			},
		}
	}

	return stats, nil
}

// GCPVMManager implements the VMManagerInterface for GCP instances
type GCPVMManager struct {
	provider *GCPProvider
}

// NewGCPVMManager creates a new VM manager for GCP
func NewGCPVMManager(provider *GCPProvider) *GCPVMManager {
	return &GCPVMManager{
		provider: provider,
	}
}

// GetVMs returns a list of all VM IDs from GCP
func (m *GCPVMManager) GetVMs(ctx context.Context) ([]string, error) {
	instances, err := m.provider.ListInstances(ctx)
	if err != nil {
		return nil, fmt.Errorf("error listing GCP instances: %w", err)
	}

	var vmIDs []string
	for _, instance := range instances {
		vmIDs = append(vmIDs, instance.ID)
	}

	return vmIDs, nil
}

// GetVMStats retrieves stats for a specific VM from GCP
func (m *GCPVMManager) GetVMStats(
	ctx context.Context,
	vmID string,
	detailLevel monitoring.VMMetricDetailLevel,
) (*monitoring.VMStats, error) {
	// Get instance details
	instance, err := m.provider.GetInstance(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("error getting GCP instance %s: %w", vmID, err)
	}

	// Get instance metrics
	metrics, err := m.provider.GetInstanceMetrics(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("error getting GCP instance metrics for %s: %w", vmID, err)
	}

	// Convert metrics to VMStats
	stats := &monitoring.VMStats{
		VMID:      vmID,
		Timestamp: time.Now(),
	}

	// Initialize CPU stats
	cpuValue, ok := metrics["cpu/utilization"]
	if ok {
		stats.CPU = monitoring.VMCPUStats{
			Usage: cpuValue * 100, // GCP returns as 0-1 value, convert to percentage
		}
	}

	// Initialize memory stats
	memoryUsedValue, memoryOk := metrics["memory/used_bytes"]
	memoryTotalValue, memoryTotalOk := metrics["memory/total_bytes"]

	if memoryOk && memoryTotalOk && memoryTotalValue > 0 {
		memoryUsagePercent := 100 * (memoryUsedValue / memoryTotalValue)
		stats.Memory = monitoring.VMMemoryStats{
			UsagePercent: memoryUsagePercent,
			Used:         int64(memoryUsedValue),
			Total:        int64(memoryTotalValue),
			Free:         int64(memoryTotalValue - memoryUsedValue),
		}
	}

	// Initialize disk stats
	diskReadOps, diskReadOpsOk := metrics["disk/read_ops_count"]
	diskWriteOps, diskWriteOpsOk := metrics["disk/write_ops_count"]
	diskReadBytes, diskReadBytesOk := metrics["disk/read_bytes_count"]
	diskWriteBytes, diskWriteBytesOk := metrics["disk/write_bytes_count"]

	if diskReadOpsOk || diskWriteOpsOk || diskReadBytesOk || diskWriteBytesOk {
		stats.Disks = []monitoring.VMDiskStats{
			{
				DiskID:          "pd-standard",
				Path:            "/dev/sda1",
				Type:            "persistent-disk",
				ReadIOPS:        diskReadOps,
				WriteIOPS:       diskWriteOps,
				ReadThroughput:  diskReadBytes,
				WriteThroughput: diskWriteBytes,
			},
		}
	}

	// Initialize network stats
	netInBytes, netInOk := metrics["network/received_bytes_count"]
	netOutBytes, netOutOk := metrics["network/sent_bytes_count"]
	netInPackets, netInPacketsOk := metrics["network/received_packets_count"]
	netOutPackets, netOutPacketsOk := metrics["network/sent_packets_count"]

	if netInOk || netOutOk || netInPacketsOk || netOutPacketsOk {
		stats.Networks = []monitoring.VMNetworkStats{
			{
				InterfaceID: "nic0",
				Name:        "default",
				RxBytes:     netInBytes,
				TxBytes:     netOutBytes,
				RxPackets:   netInPackets,
				TxPackets:   netOutPackets,
			},
		}
	}

	return stats, nil
}

// Helper functions to convert cloud provider specific states to monitoring VM states

func convertAWSStateToVMState(state string) monitoring.VMState {
	switch state {
	case "running":
		return monitoring.VMStateRunning
	case "stopped":
		return monitoring.VMStateStopped
	case "stopping":
		return monitoring.VMStateStopping
	case "pending":
		return monitoring.VMStateStarting
	case "shutting-down", "terminated":
		return monitoring.VMStateTerminated
	default:
		return monitoring.VMStateUnknown
	}
}

func convertAzureStateToVMState(state string) monitoring.VMState {
	switch state {
	case "PowerState/running":
		return monitoring.VMStateRunning
	case "PowerState/stopped":
		return monitoring.VMStateStopped
	case "PowerState/stopping":
		return monitoring.VMStateStopping
	case "PowerState/starting":
		return monitoring.VMStateStarting
	case "PowerState/deallocated", "PowerState/deallocating":
		return monitoring.VMStateTerminated
	default:
		return monitoring.VMStateUnknown
	}
}

func convertGCPStateToVMState(state string) monitoring.VMState {
	switch state {
	case "RUNNING":
		return monitoring.VMStateRunning
	case "STOPPED":
		return monitoring.VMStateStopped
	case "STOPPING":
		return monitoring.VMStateStopping
	case "PROVISIONING", "STAGING":
		return monitoring.VMStateStarting
	case "SUSPENDING", "SUSPENDED", "TERMINATED":
		return monitoring.VMStateTerminated
	default:
		return monitoring.VMStateUnknown
	}
}
