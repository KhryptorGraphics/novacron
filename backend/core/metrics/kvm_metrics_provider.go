package metrics

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// KVMMetricsProvider collects metrics from KVM VMs
type KVMMetricsProvider struct {
	vmID           string
	nodeID         string
	pid            int
	socketPath     string
	lastCPUStats   *CPUStats
	lastNetStats   map[string]*NetStats
	lastDiskStats  map[string]*DiskStats
	statsMutex     sync.Mutex
	lastCollection time.Time
	detailLevel    string
}

// CPUStats holds CPU statistics for calculating usage
type CPUStats struct {
	Timestamp time.Time
	UserTime  int64
	SystemTime int64
	TotalTime int64
}

// NetStats holds network statistics for calculating rates
type NetStats struct {
	Timestamp time.Time
	RxBytes   int64
	TxBytes   int64
	RxPackets int64
	TxPackets int64
}

// DiskStats holds disk statistics for calculating rates
type DiskStats struct {
	Timestamp time.Time
	ReadBytes int64
	WriteBytes int64
	ReadOps   int64
	WriteOps  int64
	IOTimeMs  int64
}

// NewKVMMetricsProvider creates a new KVM metrics provider
func NewKVMMetricsProvider(vmID, nodeID string, pid int, socketPath, detailLevel string) *KVMMetricsProvider {
	return &KVMMetricsProvider{
		vmID:          vmID,
		nodeID:        nodeID,
		pid:           pid,
		socketPath:    socketPath,
		lastNetStats:  make(map[string]*NetStats),
		lastDiskStats: make(map[string]*DiskStats),
		detailLevel:   detailLevel,
	}
}

// GetVMID returns the VM ID
func (p *KVMMetricsProvider) GetVMID() string {
	return p.vmID
}

// Close closes the metrics provider
func (p *KVMMetricsProvider) Close() error {
	// Nothing to close for KVM metrics provider
	return nil
}

// GetMetrics returns metrics for a KVM VM
func (p *KVMMetricsProvider) GetMetrics(ctx context.Context) (*VMMetrics, error) {
	p.statsMutex.Lock()
	defer p.statsMutex.Unlock()
	
	now := time.Now()
	
	// Create metrics object
	metrics := &VMMetrics{
		VMID:      p.vmID,
		NodeID:    p.nodeID,
		Timestamp: now,
		Disk:      make(map[string]DiskMetrics),
		Network:   make(map[string]NetMetrics),
		Labels:    make(map[string]string),
	}
	
	// Add basic labels
	metrics.Labels["vm_type"] = "kvm"
	metrics.Labels["pid"] = strconv.Itoa(p.pid)
	
	// Collect CPU metrics
	if err := p.collectCPUMetrics(ctx, metrics); err != nil {
		log.Printf("Warning: Failed to collect CPU metrics for VM %s: %v", p.vmID, err)
	}
	
	// Collect memory metrics
	if err := p.collectMemoryMetrics(ctx, metrics); err != nil {
		log.Printf("Warning: Failed to collect memory metrics for VM %s: %v", p.vmID, err)
	}
	
	// Collect disk metrics
	if err := p.collectDiskMetrics(ctx, metrics); err != nil {
		log.Printf("Warning: Failed to collect disk metrics for VM %s: %v", p.vmID, err)
	}
	
	// Collect network metrics
	if err := p.collectNetworkMetrics(ctx, metrics); err != nil {
		log.Printf("Warning: Failed to collect network metrics for VM %s: %v", p.vmID, err)
	}
	
	// Update last collection time
	p.lastCollection = now
	
	return metrics, nil
}

// collectCPUMetrics collects CPU metrics for a KVM VM
func (p *KVMMetricsProvider) collectCPUMetrics(ctx context.Context, metrics *VMMetrics) error {
	// Get CPU usage from /proc/{pid}/stat
	cmd := exec.CommandContext(ctx, "cat", fmt.Sprintf("/proc/%d/stat", p.pid))
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get CPU stats: %w", err)
	}
	
	// Parse the output
	fields := strings.Fields(string(output))
	if len(fields) < 17 {
		return fmt.Errorf("unexpected format in /proc/%d/stat", p.pid)
	}
	
	// Extract CPU times
	userTime, _ := strconv.ParseInt(fields[13], 10, 64)
	systemTime, _ := strconv.ParseInt(fields[14], 10, 64)
	totalTime := userTime + systemTime
	
	// Get number of cores
	cmd = exec.CommandContext(ctx, "grep", "-c", "processor", "/proc/cpuinfo")
	output, err = cmd.Output()
	if err != nil {
		log.Printf("Warning: Failed to get CPU core count: %v", err)
	}
	cores, _ := strconv.Atoi(strings.TrimSpace(string(output)))
	if cores == 0 {
		cores = 1 // Default to 1 if we can't determine
	}
	
	// Calculate CPU usage if we have previous stats
	now := time.Now()
	if p.lastCPUStats != nil {
		// Calculate time difference
		timeDiff := now.Sub(p.lastCPUStats.Timestamp).Seconds()
		if timeDiff > 0 {
			// Calculate CPU usage
			userDiff := float64(userTime - p.lastCPUStats.UserTime)
			systemDiff := float64(systemTime - p.lastCPUStats.SystemTime)
			totalDiff := float64(totalTime - p.lastCPUStats.TotalTime)
			
			// Calculate percentages
			metrics.CPU.UserPercent = (userDiff / timeDiff) / float64(cores) * 100
			metrics.CPU.SystemPercent = (systemDiff / timeDiff) / float64(cores) * 100
			metrics.CPU.UsagePercent = (totalDiff / timeDiff) / float64(cores) * 100
		}
	}
	
	// Update last CPU stats
	p.lastCPUStats = &CPUStats{
		Timestamp:  now,
		UserTime:   userTime,
		SystemTime: systemTime,
		TotalTime:  totalTime,
	}
	
	// Set cores
	metrics.CPU.Cores = cores
	
	// Get CPU throttling information if available and detail level is high
	if p.detailLevel == "detailed" {
		// This would typically come from cgroups
		// For simplicity, we'll leave these as 0
		metrics.CPU.ThrottledPeriods = 0
		metrics.CPU.ThrottledTime = 0
	}
	
	return nil
}

// collectMemoryMetrics collects memory metrics for a KVM VM
func (p *KVMMetricsProvider) collectMemoryMetrics(ctx context.Context, metrics *VMMetrics) error {
	// Get memory usage from /proc/{pid}/status
	cmd := exec.CommandContext(ctx, "cat", fmt.Sprintf("/proc/%d/status", p.pid))
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get memory stats: %w", err)
	}
	
	// Parse the output
	lines := strings.Split(string(output), "\n")
	var vmRSS, vmSwap int64
	
	for _, line := range lines {
		if strings.HasPrefix(line, "VmRSS:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				value, _ := strconv.ParseInt(fields[1], 10, 64)
				vmRSS = value * 1024 // Convert from KB to bytes
			}
		} else if strings.HasPrefix(line, "VmSwap:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				value, _ := strconv.ParseInt(fields[1], 10, 64)
				vmSwap = value * 1024 // Convert from KB to bytes
			}
		}
	}
	
	// Get total memory from QMP if available
	var totalMemory int64
	if p.socketPath != "" {
		// Use QMP to get total memory
		// This is a simplified example - in a real implementation, you'd use a QMP client
		totalMemory = p.getQMPMemoryInfo(ctx)
	}
	
	// If we couldn't get total memory from QMP, try to get it from /proc/{pid}/cmdline
	if totalMemory == 0 {
		cmd = exec.CommandContext(ctx, "cat", fmt.Sprintf("/proc/%d/cmdline", p.pid))
		output, err = cmd.Output()
		if err == nil {
			cmdline := strings.Replace(string(output), "\x00", " ", -1)
			// Look for -m parameter
			if idx := strings.Index(cmdline, " -m "); idx >= 0 {
				fields := strings.Fields(cmdline[idx+4:])
				if len(fields) > 0 {
					value, _ := strconv.ParseInt(fields[0], 10, 64)
					totalMemory = value * 1024 * 1024 // Convert from MB to bytes
				}
			}
		}
	}
	
	// Set memory metrics
	metrics.Memory.RSSBytes = vmRSS
	metrics.Memory.SwapBytes = vmSwap
	metrics.Memory.UsedBytes = vmRSS
	metrics.Memory.TotalBytes = totalMemory
	
	// Calculate percentages
	if totalMemory > 0 {
		metrics.Memory.UsagePercent = float64(vmRSS) / float64(totalMemory) * 100
	}
	
	// Get page faults if detail level is high
	if p.detailLevel == "detailed" {
		// This would typically come from /proc/{pid}/stat
		// For simplicity, we'll leave these as 0
		metrics.Memory.MajorPageFaults = 0
		metrics.Memory.MinorPageFaults = 0
	}
	
	return nil
}

// collectDiskMetrics collects disk metrics for a KVM VM
func (p *KVMMetricsProvider) collectDiskMetrics(ctx context.Context, metrics *VMMetrics) error {
	// Get disk usage from QMP if available
	now := time.Now()
	if p.socketPath != "" {
		// Use QMP to get disk stats
		// This is a simplified example - in a real implementation, you'd use a QMP client
		diskStats := p.getQMPDiskStats(ctx)
		
		for device, stats := range diskStats {
			// Create disk metrics
			diskMetrics := DiskMetrics{
				Device:     device,
				ReadBytes:  stats.ReadBytes,
				WriteBytes: stats.WriteBytes,
				ReadOps:    stats.ReadOps,
				WriteOps:   stats.WriteOps,
				IOTimeMs:   stats.IOTimeMs,
			}
			
			// Calculate rates if we have previous stats
			lastStats, exists := p.lastDiskStats[device]
			if exists {
				// Calculate time difference
				timeDiff := now.Sub(lastStats.Timestamp).Seconds()
				if timeDiff > 0 {
					// Calculate read/write rates
					readBytesRate := float64(stats.ReadBytes-lastStats.ReadBytes) / timeDiff
					writeBytesRate := float64(stats.WriteBytes-lastStats.WriteBytes) / timeDiff
					
					// Add rates to annotations
					if metrics.Annotations == nil {
						metrics.Annotations = make(map[string]string)
					}
					metrics.Annotations[fmt.Sprintf("disk.%s.read_bytes_per_sec", device)] = fmt.Sprintf("%.2f", readBytesRate)
					metrics.Annotations[fmt.Sprintf("disk.%s.write_bytes_per_sec", device)] = fmt.Sprintf("%.2f", writeBytesRate)
				}
			}
			
			// Update last disk stats
			p.lastDiskStats[device] = &DiskStats{
				Timestamp:  now,
				ReadBytes:  stats.ReadBytes,
				WriteBytes: stats.WriteBytes,
				ReadOps:    stats.ReadOps,
				WriteOps:   stats.WriteOps,
				IOTimeMs:   stats.IOTimeMs,
			}
			
			// Add disk metrics
			metrics.Disk[device] = diskMetrics
		}
	}
	
	return nil
}

// collectNetworkMetrics collects network metrics for a KVM VM
func (p *KVMMetricsProvider) collectNetworkMetrics(ctx context.Context, metrics *VMMetrics) error {
	// Get network usage from QMP if available
	now := time.Now()
	if p.socketPath != "" {
		// Use QMP to get network stats
		// This is a simplified example - in a real implementation, you'd use a QMP client
		netStats := p.getQMPNetStats(ctx)
		
		for iface, stats := range netStats {
			// Create network metrics
			netMetrics := NetMetrics{
				Interface: iface,
				RxBytes:   stats.RxBytes,
				TxBytes:   stats.TxBytes,
				RxPackets: stats.RxPackets,
				TxPackets: stats.TxPackets,
			}
			
			// Calculate rates if we have previous stats
			lastStats, exists := p.lastNetStats[iface]
			if exists {
				// Calculate time difference
				timeDiff := now.Sub(lastStats.Timestamp).Seconds()
				if timeDiff > 0 {
					// Calculate rx/tx rates
					rxBytesRate := float64(stats.RxBytes-lastStats.RxBytes) / timeDiff
					txBytesRate := float64(stats.TxBytes-lastStats.TxBytes) / timeDiff
					
					// Add rates to metrics
					netMetrics.RxBytesPerSec = rxBytesRate
					netMetrics.TxBytesPerSec = txBytesRate
				}
			}
			
			// Update last network stats
			p.lastNetStats[iface] = &NetStats{
				Timestamp: now,
				RxBytes:   stats.RxBytes,
				TxBytes:   stats.TxBytes,
				RxPackets: stats.RxPackets,
				TxPackets: stats.TxPackets,
			}
			
			// Add network metrics
			metrics.Network[iface] = netMetrics
		}
	}
	
	return nil
}

// getQMPMemoryInfo gets memory information from QMP
func (p *KVMMetricsProvider) getQMPMemoryInfo(ctx context.Context) int64 {
	// This is a simplified example - in a real implementation, you'd use a QMP client
	// For now, we'll use socat to send a QMP command
	
	// First, send capabilities negotiation
	capCmd := fmt.Sprintf("echo '{\"execute\": \"qmp_capabilities\"}' | socat - UNIX-CONNECT:%s", p.socketPath)
	cmd := exec.CommandContext(ctx, "bash", "-c", capCmd)
	if _, err := cmd.CombinedOutput(); err != nil {
		log.Printf("Warning: Failed to negotiate QMP capabilities: %v", err)
		return 0
	}
	
	// Then send query-memory-size command
	queryCmd := fmt.Sprintf("echo '{\"execute\": \"query-memory-size\"}' | socat - UNIX-CONNECT:%s", p.socketPath)
	cmd = exec.CommandContext(ctx, "bash", "-c", queryCmd)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Warning: Failed to query memory size: %v", err)
		return 0
	}
	
	// Parse the output
	var response struct {
		Return struct {
			Base struct {
				Size int64 `json:"size"`
			} `json:"base"`
		} `json:"return"`
	}
	
	if err := json.Unmarshal(output, &response); err != nil {
		log.Printf("Warning: Failed to parse memory size response: %v", err)
		return 0
	}
	
	return response.Return.Base.Size
}

// getQMPDiskStats gets disk statistics from QMP
func (p *KVMMetricsProvider) getQMPDiskStats(ctx context.Context) map[string]*DiskStats {
	// This is a simplified example - in a real implementation, you'd use a QMP client
	// For now, we'll use socat to send a QMP command
	now := time.Now()
	
	// First, send capabilities negotiation
	capCmd := fmt.Sprintf("echo '{\"execute\": \"qmp_capabilities\"}' | socat - UNIX-CONNECT:%s", p.socketPath)
	cmd := exec.CommandContext(ctx, "bash", "-c", capCmd)
	if _, err := cmd.CombinedOutput(); err != nil {
		log.Printf("Warning: Failed to negotiate QMP capabilities: %v", err)
		return nil
	}
	
	// Then send query-block command
	queryCmd := fmt.Sprintf("echo '{\"execute\": \"query-block\"}' | socat - UNIX-CONNECT:%s", p.socketPath)
	cmd = exec.CommandContext(ctx, "bash", "-c", queryCmd)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Warning: Failed to query block devices: %v", err)
		return nil
	}
	
	// Parse the output
	var response struct {
		Return []struct {
			Device string `json:"device"`
			Stats  struct {
				ReadBytes  int64 `json:"rd_bytes"`
				WriteBytes int64 `json:"wr_bytes"`
				ReadOps    int64 `json:"rd_operations"`
				WriteOps   int64 `json:"wr_operations"`
			} `json:"stats"`
		} `json:"return"`
	}
	
	if err := json.Unmarshal(output, &response); err != nil {
		log.Printf("Warning: Failed to parse block device response: %v", err)
		return nil
	}
	
	// Create disk stats map
	stats := make(map[string]*DiskStats)
	
	for _, block := range response.Return {
		stats[block.Device] = &DiskStats{
			Timestamp:  now,
			ReadBytes:  block.Stats.ReadBytes,
			WriteBytes: block.Stats.WriteBytes,
			ReadOps:    block.Stats.ReadOps,
			WriteOps:   block.Stats.WriteOps,
			IOTimeMs:   0, // Not available from QMP
		}
	}
	
	return stats
}

// getQMPNetStats gets network statistics from QMP
func (p *KVMMetricsProvider) getQMPNetStats(ctx context.Context) map[string]*NetStats {
	// This is a simplified example - in a real implementation, you'd use a QMP client
	// For now, we'll use socat to send a QMP command
	now := time.Now()
	
	// First, send capabilities negotiation
	capCmd := fmt.Sprintf("echo '{\"execute\": \"qmp_capabilities\"}' | socat - UNIX-CONNECT:%s", p.socketPath)
	cmd := exec.CommandContext(ctx, "bash", "-c", capCmd)
	if _, err := cmd.CombinedOutput(); err != nil {
		log.Printf("Warning: Failed to negotiate QMP capabilities: %v", err)
		return nil
	}
	
	// Then send query-netdev command
	queryCmd := fmt.Sprintf("echo '{\"execute\": \"query-netdev\"}' | socat - UNIX-CONNECT:%s", p.socketPath)
	cmd = exec.CommandContext(ctx, "bash", "-c", queryCmd)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Warning: Failed to query network devices: %v", err)
		return nil
	}
	
	// Parse the output
	var response struct {
		Return []struct {
			ID    string `json:"id"`
			Stats struct {
				RxBytes   int64 `json:"rx_bytes"`
				TxBytes   int64 `json:"tx_bytes"`
				RxPackets int64 `json:"rx_packets"`
				TxPackets int64 `json:"tx_packets"`
			} `json:"stats"`
		} `json:"return"`
	}
	
	if err := json.Unmarshal(output, &response); err != nil {
		log.Printf("Warning: Failed to parse network device response: %v", err)
		return nil
	}
	
	// Create network stats map
	stats := make(map[string]*NetStats)
	
	for _, netdev := range response.Return {
		stats[netdev.ID] = &NetStats{
			Timestamp: now,
			RxBytes:   netdev.Stats.RxBytes,
			TxBytes:   netdev.Stats.TxBytes,
			RxPackets: netdev.Stats.RxPackets,
			TxPackets: netdev.Stats.TxPackets,
		}
	}
	
	return stats
}
