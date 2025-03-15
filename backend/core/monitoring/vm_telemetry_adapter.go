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

// ConvertCloudToInternalStats converts the cloud-compatible CloudVMStats to the internal VMStats structure
func ConvertCloudToInternalStats(cloudStats *CloudVMStats) *VMStats {
	if cloudStats == nil {
		return nil
	}

	// Parse timestamp
	timestamp, err := time.Parse(time.RFC3339, cloudStats.Timestamp)
	if err != nil {
		timestamp = time.Now() // Fallback to current time if parsing fails
	}

	// Create the base internal stats structure
	vmStats := &VMStats{
		VMID:      cloudStats.VMID,
		Timestamp: timestamp,
	}

	// Convert CPU stats
	if cloudStats.CPU != nil {
		vmStats.CPU = VMCPUStats{
			Usage:      cloudStats.CPU.Usage,
			SystemTime: cloudStats.CPU.SystemTime,
			UserTime:   cloudStats.CPU.UserTime,
			IOWaitTime: cloudStats.CPU.IOWaitTime,
			StealTime:  cloudStats.CPU.StealTime,
			ReadyTime:  cloudStats.CPU.ReadyTime,
		}

		// Convert core usage from map to array
		// We need to determine the size of the array first by finding the highest core number
		maxCoreNum := -1
		for coreID := range cloudStats.CPU.CoreUsage {
			var coreNum int
			_, err := fmt.Sscanf(coreID, "core-%d", &coreNum)
			if err == nil && coreNum > maxCoreNum {
				maxCoreNum = coreNum
			}
		}

		if maxCoreNum >= 0 {
			vmStats.CPU.CoreUsage = make([]float64, maxCoreNum+1)
			for coreID, usage := range cloudStats.CPU.CoreUsage {
				var coreNum int
				_, err := fmt.Sscanf(coreID, "core-%d", &coreNum)
				if err == nil && coreNum >= 0 && coreNum <= maxCoreNum {
					vmStats.CPU.CoreUsage[coreNum] = usage
				}
			}
		}
	}

	// Convert memory stats
	if cloudStats.Memory != nil {
		vmStats.Memory = VMMemoryStats{
			UsagePercent:    cloudStats.Memory.UsagePercent,
			Used:            int64(cloudStats.Memory.Used),
			Total:           int64(cloudStats.Memory.Total),
			Free:            int64(cloudStats.Memory.Free),
			SwapUsed:        int64(cloudStats.Memory.SwapUsed),
			SwapTotal:       int64(cloudStats.Memory.SwapTotal),
			PageFaults:      cloudStats.Memory.PageFaults,
			MajorPageFaults: cloudStats.Memory.MajorPageFaults,
			BalloonTarget:   int64(cloudStats.Memory.BalloonTarget),
			BalloonCurrent:  int64(cloudStats.Memory.BalloonCurrent),
		}
	}

	// Convert disk stats from map to array
	if len(cloudStats.Disks) > 0 {
		vmStats.Disks = make([]VMDiskStats, 0, len(cloudStats.Disks))
		for _, disk := range cloudStats.Disks {
			vmStats.Disks = append(vmStats.Disks, VMDiskStats{
				DiskID:          disk.DiskID,
				Path:            disk.Path,
				Type:            disk.Type,
				UsagePercent:    disk.UsagePercent,
				Used:            int64(disk.Used),
				Size:            int64(disk.Total),
				ReadIOPS:        disk.ReadIOPS,
				WriteIOPS:       disk.WriteIOPS,
				ReadThroughput:  disk.ReadThroughput,
				WriteThroughput: disk.WriteThroughput,
				ReadLatency:     disk.ReadLatency,
				WriteLatency:    disk.WriteLatency,
			})
		}
	}

	// Convert network stats from map to array
	if len(cloudStats.Networks) > 0 {
		vmStats.Networks = make([]VMNetworkStats, 0, len(cloudStats.Networks))
		for _, network := range cloudStats.Networks {
			vmStats.Networks = append(vmStats.Networks, VMNetworkStats{
				InterfaceID: network.InterfaceID,
				Name:        network.Name,
				RxBytes:     network.RxBytes,
				TxBytes:     network.TxBytes,
				RxPackets:   network.RxPackets,
				TxPackets:   network.TxPackets,
				RxErrors:    network.RxErrors,
				TxErrors:    network.TxErrors,
				RxDropped:   network.RxDropped,
				TxDropped:   network.TxDropped,
			})
		}
	}

	// Convert process stats from map to array
	if len(cloudStats.Processes) > 0 {
		vmStats.Processes = make([]VMProcessStats, 0, len(cloudStats.Processes))
		for _, process := range cloudStats.Processes {
			vmStats.Processes = append(vmStats.Processes, VMProcessStats{
				PID:             int64(process.PID),
				Name:            process.Name,
				Command:         process.Command,
				CPUUsage:        process.CPUUsage,
				MemoryPercent:   process.MemoryPercent,
				MemoryUsage:     int64(process.MemoryUsed),
				ReadIOPS:        process.ReadIOPS,
				WriteIOPS:       process.WriteIOPS,
				ReadThroughput:  process.ReadThroughput,
				WriteThroughput: process.WriteThroughput,
				OpenFiles:       int64(process.OpenFiles),
				RunTime:         process.RunTime,
			})
		}
	}

	// Copy over any metadata fields to ensure they are preserved
	if len(cloudStats.Metadata) > 0 || len(cloudStats.Tags) > 0 {
		// Note: since internal VMStats doesn't have a direct metadata/tags field,
		// we might need to create a custom extension to store this data
		// For now, just log that we have metadata/tags
		// In a real implementation, we would want to preserve this data
	}

	return vmStats
}
