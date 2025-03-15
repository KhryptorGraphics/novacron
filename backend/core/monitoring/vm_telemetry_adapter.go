package monitoring

import (
	"fmt"
	"time"
)

// ConvertInternalToCloudStats converts the internal VMStats structure to the cloud-compatible CloudVMStats
func ConvertInternalToCloudStats(vmStats *VMStats) *CloudVMStats {
	if vmStats == nil {
		return nil
	}

	// Create the base cloud stats structure
	cloudStats := &CloudVMStats{
		VMID:      vmStats.VMID,
		State:     VMStateRunning, // Default to running state since internal structure doesn't have state
		Timestamp: vmStats.Timestamp.Format(time.RFC3339),
		Metadata:  make(map[string]string),
		Tags:      make(map[string]string),
	}

	// Convert CPU stats
	cloudStats.CPU = &CloudCPUStats{
		Usage:      vmStats.CPU.Usage,
		SystemTime: vmStats.CPU.SystemTime,
		UserTime:   vmStats.CPU.UserTime,
		IOWaitTime: vmStats.CPU.IOWaitTime,
		StealTime:  vmStats.CPU.StealTime,
		ReadyTime:  vmStats.CPU.ReadyTime,
		CoreUsage:  make(map[string]float64),
	}

	// Convert core usage from array to map
	for i, usage := range vmStats.CPU.CoreUsage {
		coreID := fmt.Sprintf("core-%d", i)
		cloudStats.CPU.CoreUsage[coreID] = usage
	}

	// Convert memory stats
	cloudStats.Memory = &CloudMemoryStats{
		Usage:           vmStats.Memory.UsagePercent,
		UsagePercent:    vmStats.Memory.UsagePercent,
		Used:            float64(vmStats.Memory.Used),
		Total:           float64(vmStats.Memory.Total),
		Free:            float64(vmStats.Memory.Free),
		SwapUsed:        float64(vmStats.Memory.SwapUsed),
		SwapTotal:       float64(vmStats.Memory.SwapTotal),
		PageFaults:      vmStats.Memory.PageFaults,
		MajorPageFaults: vmStats.Memory.MajorPageFaults,
		BalloonTarget:   float64(vmStats.Memory.BalloonTarget),
		BalloonCurrent:  float64(vmStats.Memory.BalloonCurrent),
	}

	// Convert disk stats from array to map
	cloudStats.Disks = make(map[string]*CloudDiskStats)
	for _, disk := range vmStats.Disks {
		diskID := disk.DiskID
		if diskID == "" {
			diskID = disk.Path
		}
		cloudStats.Disks[diskID] = &CloudDiskStats{
			DiskID:          diskID,
			Path:            disk.Path,
			Type:            disk.Type,
			Usage:           disk.UsagePercent,
			UsagePercent:    disk.UsagePercent,
			Used:            float64(disk.Used),
			Total:           float64(disk.Size),
			Size:            float64(disk.Size),
			ReadIOPS:        disk.ReadIOPS,
			WriteIOPS:       disk.WriteIOPS,
			ReadThroughput:  disk.ReadThroughput,
			WriteThroughput: disk.WriteThroughput,
			ReadLatency:     disk.ReadLatency,
			WriteLatency:    disk.WriteLatency,
			QueueDepth:      0, // Not available in internal format
		}
	}

	// Convert network stats from array to map
	cloudStats.Networks = make(map[string]*CloudNetworkStats)
	for _, network := range vmStats.Networks {
		netID := network.InterfaceID
		if netID == "" {
			netID = network.Name
		}
		cloudStats.Networks[netID] = &CloudNetworkStats{
			InterfaceID: netID,
			Name:        network.Name,
			RxBytes:     network.RxBytes,
			TxBytes:     network.TxBytes,
			RxPackets:   network.RxPackets,
			TxPackets:   network.TxPackets,
			RxErrors:    network.RxErrors,
			TxErrors:    network.TxErrors,
			RxDropped:   network.RxDropped,
			TxDropped:   network.TxDropped,
		}
	}

	// Convert process stats from array to map
	cloudStats.Processes = make(map[string]*CloudProcessStats)
	for _, process := range vmStats.Processes {
		procID := fmt.Sprintf("%d", process.PID)
		cloudStats.Processes[procID] = &CloudProcessStats{
			PID:             int(process.PID),
			Name:            process.Name,
			Command:         process.Command,
			CPU:             process.CPUUsage,
			CPUUsage:        process.CPUUsage,
			Memory:          process.MemoryPercent,
			MemoryPercent:   process.MemoryPercent,
			MemoryUsed:      float64(process.MemoryUsage),
			MemoryUsage:     float64(process.MemoryUsage),
			ReadIOPS:        process.ReadIOPS,
			WriteIOPS:       process.WriteIOPS,
			ReadThroughput:  process.ReadThroughput,
			WriteThroughput: process.WriteThroughput,
			DiskRead:        process.ReadThroughput,  // Approximate mapping
			DiskWrite:       process.WriteThroughput, // Approximate mapping
			OpenFiles:       int(process.OpenFiles),
			StartTime:       0, // Not available in internal format
			RunTime:         process.RunTime,
			Priority:        0,  // Not available in internal format
			State:           "", // Not available in internal format
		}
	}

	return cloudStats
}
