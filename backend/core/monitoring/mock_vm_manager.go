package monitoring

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// MockVMManager implements VMManagerInterface for testing and examples
type MockVMManager struct {
	// List of VM IDs to simulate
	vmIDs []string

	// VM statistics store
	vmStats map[string]*VMStats

	// VM parameters for generating realistic data
	vmParams map[string]*mockVMParams

	// Last update time for each VM
	lastUpdates map[string]time.Time

	// Lock for concurrent access
	mutex sync.RWMutex

	// Default parameters
	defaultCPUCores    int
	defaultMemoryBytes int64
	defaultDiskBytes   int64
	defaultNetworkIFs  int
	defaultProcs       int

	// Optional failure rate for testing error handling
	failureRate float64
}

// mockVMParams contains parameters for generating realistic VM data
type mockVMParams struct {
	// CPU cores and base usage
	CPUCores       int
	CPUBaseUsage   float64
	CPUVariability float64

	// Memory parameters
	MemoryTotal     int64
	MemoryBaseUsage float64
	MemoryVariance  float64

	// Disk parameters
	Disks         []mockDiskParams
	DiskReadIOPS  float64
	DiskWriteIOPS float64

	// Network parameters
	Networks    []mockNetworkParams
	NetworkLoad float64

	// Process parameters
	ProcessCount  int
	PredefinedIDs bool // Use predefined process IDs for consistency

	// Variance factor - higher means more variability in metrics
	VarianceFactor float64

	// Spike probability - chance of generating metric spikes
	SpikeProbability float64
}

// mockDiskParams contains parameters for disk simulation
type mockDiskParams struct {
	ID           string
	Path         string
	SizeBytes    int64
	UsagePercent float64
	Type         string
}

// mockNetworkParams contains parameters for network interface simulation
type mockNetworkParams struct {
	ID   string
	Name string
	Load float64
}

// NewMockVMManager creates a new mock VM manager with the specified VM IDs
func NewMockVMManager(vmIDs []string) *MockVMManager {
	if len(vmIDs) == 0 {
		// Default VM IDs
		vmIDs = []string{"vm-001", "vm-002", "vm-003", "vm-004", "vm-005"}
	}

	mockManager := &MockVMManager{
		vmIDs:              vmIDs,
		vmStats:            make(map[string]*VMStats),
		vmParams:           make(map[string]*mockVMParams),
		lastUpdates:        make(map[string]time.Time),
		defaultCPUCores:    4,
		defaultMemoryBytes: 8 * 1024 * 1024 * 1024,   // 8 GB
		defaultDiskBytes:   100 * 1024 * 1024 * 1024, // 100 GB
		defaultNetworkIFs:  2,
		defaultProcs:       20,
		failureRate:        0.0,
	}

	// Initialize VM parameters with randomization for each VM
	for _, vmID := range vmIDs {
		// Randomize VM resources within reasonable bounds
		cpuCores := mockManager.defaultCPUCores
		if rand.Intn(10) > 7 {
			// Some VMs have more or fewer cores
			cpuCores = rand.Intn(12) + 1
		}

		memoryMult := 1.0 + (rand.Float64()*2-1)*0.5 // Vary by +/- 50%
		memoryTotal := int64(float64(mockManager.defaultMemoryBytes) * memoryMult)

		// Create disk params
		diskCount := rand.Intn(3) + 1 // 1 to 3 disks
		disks := make([]mockDiskParams, diskCount)

		// First disk is always system
		disks[0] = mockDiskParams{
			ID:           fmt.Sprintf("%s-disk-0", vmID),
			Path:         "/dev/sda",
			SizeBytes:    mockManager.defaultDiskBytes,
			UsagePercent: 40 + rand.Float64()*30, // 40-70% usage
			Type:         "system",
		}

		// Add data disks if any
		for i := 1; i < diskCount; i++ {
			diskSizeMult := 1.0 + rand.Float64()*3 // 1x to 4x the default size
			disks[i] = mockDiskParams{
				ID:           fmt.Sprintf("%s-disk-%d", vmID, i),
				Path:         fmt.Sprintf("/dev/sd%c", 'b'+i-1),
				SizeBytes:    int64(float64(mockManager.defaultDiskBytes) * diskSizeMult),
				UsagePercent: 20 + rand.Float64()*60, // 20-80% usage
				Type:         "data",
			}
		}

		// Create network interface params
		networkCount := mockManager.defaultNetworkIFs
		networks := make([]mockNetworkParams, networkCount)
		for i := 0; i < networkCount; i++ {
			networks[i] = mockNetworkParams{
				ID:   fmt.Sprintf("%s-net-%d", vmID, i),
				Name: fmt.Sprintf("eth%d", i),
				Load: 0.1 + rand.Float64()*0.4, // 10-50% baseline load
			}
		}

		// Set up VM params
		mockManager.vmParams[vmID] = &mockVMParams{
			CPUCores:         cpuCores,
			CPUBaseUsage:     20 + rand.Float64()*40,   // 20-60% base CPU usage
			CPUVariability:   0.2 + rand.Float64()*0.3, // 20-50% variability
			MemoryTotal:      memoryTotal,
			MemoryBaseUsage:  50 + rand.Float64()*30,     // 50-80% base memory usage
			MemoryVariance:   0.15 + rand.Float64()*0.15, // 15-30% variance
			Disks:            disks,
			DiskReadIOPS:     100 + rand.Float64()*400, // 100-500 IOPS base
			DiskWriteIOPS:    50 + rand.Float64()*200,  // 50-250 IOPS base
			Networks:         networks,
			NetworkLoad:      0.3 + rand.Float64()*0.4, // 30-70% network load
			ProcessCount:     5 + rand.Intn(mockManager.defaultProcs),
			PredefinedIDs:    true,
			VarianceFactor:   0.1 + rand.Float64()*0.4,  // 10-50% variance
			SpikeProbability: 0.05 + rand.Float64()*0.1, // 5-15% spike probability
		}

		// Initialize stats
		mockManager.vmStats[vmID] = &VMStats{
			VMID:      vmID,
			Timestamp: time.Now(),
		}
		mockManager.lastUpdates[vmID] = time.Now().Add(-time.Minute)
	}

	return mockManager
}

// GetVMs returns the list of VM IDs
func (m *MockVMManager) GetVMs(ctx context.Context) ([]string, error) {
	// Simulate random failures if enabled
	if m.failureRate > 0 && rand.Float64() < m.failureRate {
		return nil, fmt.Errorf("simulated failure in GetVMs")
	}

	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Return a copy to avoid external modification
	result := make([]string, len(m.vmIDs))
	copy(result, m.vmIDs)
	return result, nil
}

// GetVMStats retrieves stats for a specific VM
func (m *MockVMManager) GetVMStats(ctx context.Context, vmID string, detailLevel VMMetricDetailLevel) (*VMStats, error) {
	// Simulate random failures if enabled
	if m.failureRate > 0 && rand.Float64() < m.failureRate {
		return nil, fmt.Errorf("simulated failure in GetVMStats for VM %s", vmID)
	}

	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Check if VM exists
	params, exists := m.vmParams[vmID]
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}

	// Get last stats and time since last update
	lastStats, exists := m.vmStats[vmID]
	if !exists {
		lastStats = &VMStats{
			VMID:      vmID,
			Timestamp: time.Now(),
		}
		m.vmStats[vmID] = lastStats
	}

	lastUpdate := m.lastUpdates[vmID]
	timeSinceUpdate := time.Since(lastUpdate)
	m.lastUpdates[vmID] = time.Now()

	// Generate new stats based on parameters and time evolution
	stats := m.generateVMStats(vmID, params, lastStats, timeSinceUpdate, detailLevel)

	// Store the new stats
	m.vmStats[vmID] = stats

	// Return a copy to prevent modification
	return m.copyVMStats(stats), nil
}

// generateVMStats creates realistic VM stats with time evolution
func (m *MockVMManager) generateVMStats(vmID string, params *mockVMParams, lastStats *VMStats,
	timeSinceUpdate time.Duration, detailLevel VMMetricDetailLevel) *VMStats {

	// Create new stats object
	stats := &VMStats{
		VMID:      vmID,
		Timestamp: time.Now(),
	}

	// Time factor affects how much change to apply (normalized to seconds)
	timeFactor := timeSinceUpdate.Seconds() / 60.0 // Base unit is minutes
	if timeFactor > 5.0 {
		timeFactor = 5.0 // Cap at 5 minutes for sanity
	} else if timeFactor < 0.01 {
		timeFactor = 0.01 // Minimum to ensure some change
	}

	// ------------------------------------------------------------
	// CPU stats generation
	// ------------------------------------------------------------
	// Base CPU usage with natural oscillation
	oscillation := (rand.Float64()*2 - 1) * params.CPUVariability * timeFactor
	spikeChance := rand.Float64() < params.SpikeProbability*timeFactor

	cpuUsage := params.CPUBaseUsage + oscillation*100
	if spikeChance {
		cpuUsage += (100 - params.CPUBaseUsage) * 0.7 // Spike to 70% of remaining capacity
	}

	// Ensure CPU usage is within bounds
	if cpuUsage < 0.1 {
		cpuUsage = 0.1
	} else if cpuUsage > 99.9 {
		cpuUsage = 99.9
	}

	// Generate per-core CPU usage
	coreUsage := make([]float64, params.CPUCores)
	avgCoreUsage := cpuUsage / float64(params.CPUCores)

	for i := range coreUsage {
		// Core usage varies around the average
		variance := (rand.Float64()*2 - 1) * 15 // +/- 15%
		coreUsage[i] = avgCoreUsage + variance

		// Ensure core usage is within bounds
		if coreUsage[i] < 0.1 {
			coreUsage[i] = 0.1
		} else if coreUsage[i] > 100.0 {
			coreUsage[i] = 100.0
		}
	}

	// Steal time is higher under load
	stealTime := 0.5 + (cpuUsage/100.0)*4.0 // 0.5-4.5%

	// Ready time increases with CPU usage
	readyTime := 0.1 + (cpuUsage/100.0)*2.0 // 0.1-2.1%

	// Set CPU stats
	stats.CPU = VMCPUStats{
		Usage:      cpuUsage,
		CoreUsage:  coreUsage,
		NumCPUs:    params.CPUCores,
		StealTime:  stealTime,
		ReadyTime:  readyTime,
		SystemTime: cpuUsage * 0.3,           // ~30% of CPU is system time
		UserTime:   cpuUsage * 0.7,           // ~70% of CPU is user time
		IOWaitTime: 0.5 + rand.Float64()*1.5, // 0.5-2.0% IO wait
	}

	// ------------------------------------------------------------
	// Memory stats generation
	// ------------------------------------------------------------
	// Memory usage with natural fluctuation
	memOscillation := (rand.Float64()*2 - 1) * params.MemoryVariance * timeFactor
	memSpikeChance := rand.Float64() < params.SpikeProbability*timeFactor*0.7 // Less frequent than CPU

	memUsagePercent := params.MemoryBaseUsage + memOscillation*100
	if memSpikeChance {
		memUsagePercent += (100 - params.MemoryBaseUsage) * 0.6 // Spike to 60% of remaining capacity
	}

	// Ensure memory usage is within bounds
	if memUsagePercent < 1.0 {
		memUsagePercent = 1.0
	} else if memUsagePercent > 99.9 {
		memUsagePercent = 99.9
	}

	// Calculate memory values
	memUsed := int64(float64(params.MemoryTotal) * memUsagePercent / 100.0)
	memFree := params.MemoryTotal - memUsed

	// Swap usage correlates with memory pressure
	swapTotal := params.MemoryTotal / 2 // Swap is half of RAM
	swapUsedPercent := 0.0

	if memUsagePercent > 85.0 {
		// Start using swap when memory is high
		swapUsedPercent = (memUsagePercent - 85.0) * 6.0 // 0-90% of swap
	}

	swapUsed := int64(float64(swapTotal) * swapUsedPercent / 100.0)

	// Page faults increase with memory pressure
	pageFaults := 10.0 + (memUsagePercent/100.0)*500.0    // 10-510 per second
	majorPageFaults := 0.1 + (memUsagePercent/100.0)*10.0 // 0.1-10.1 per second

	// Ballooning metrics if memory is tight
	balloonTarget := int64(0)
	balloonCurrent := int64(0)

	if memUsagePercent > 90.0 {
		// Try to reclaim 10-20% of memory
		reclaimPercent := 10.0 + rand.Float64()*10.0
		balloonTarget = int64(float64(params.MemoryTotal) * reclaimPercent / 100.0)
		balloonCurrent = int64(float64(balloonTarget) * (0.7 + rand.Float64()*0.3)) // 70-100% of target
	}

	// Set memory stats
	stats.Memory = VMMemoryStats{
		Total:           params.MemoryTotal,
		Used:            memUsed,
		UsagePercent:    memUsagePercent,
		Free:            memFree,
		SwapUsed:        swapUsed,
		SwapTotal:       swapTotal,
		PageFaults:      pageFaults,
		MajorPageFaults: majorPageFaults,
		BalloonTarget:   balloonTarget,
		BalloonCurrent:  balloonCurrent,
	}

	// ------------------------------------------------------------
	// Disk stats generation
	// ------------------------------------------------------------
	diskCount := len(params.Disks)
	stats.Disks = make([]VMDiskStats, diskCount)

	for i, diskParams := range params.Disks {
		// Disk usage slowly increases over time for system disks
		usagePercent := diskParams.UsagePercent
		if diskParams.Type == "system" {
			// Increase by 0.01-0.05% per minute
			usagePercent += (0.01 + rand.Float64()*0.04) * timeFactor

			// Cap at 98%
			if usagePercent > 98.0 {
				usagePercent = 98.0
			}
		}

		// Calculate disk space
		usedBytes := int64(float64(diskParams.SizeBytes) * usagePercent / 100.0)

		// IOPS varies based on CPU and memory activity
		iopsMultiplier := 1.0 + (stats.CPU.Usage/100.0)*0.5 + (stats.Memory.UsagePercent/100.0)*0.5

		// Base IOPS modified by activity and random variance
		readIOPS := params.DiskReadIOPS * iopsMultiplier * (0.8 + rand.Float64()*0.4)
		writeIOPS := params.DiskWriteIOPS * iopsMultiplier * (0.8 + rand.Float64()*0.4)

		// IO spikes
		if rand.Float64() < params.SpikeProbability*timeFactor {
			readIOPS *= 2.0 + rand.Float64()*3.0 // 2-5x spike
		}

		if rand.Float64() < params.SpikeProbability*timeFactor {
			writeIOPS *= 2.0 + rand.Float64()*3.0 // 2-5x spike
		}

		// Throughput correlates with IOPS
		// Assuming average IO size of 32K
		readThroughput := readIOPS * 32 * 1024
		writeThroughput := writeIOPS * 32 * 1024

		// Latency increases with IOPS
		// Base latency 0.5-2ms + load factor
		baseReadLatency := 0.5 + rand.Float64()*1.5
		baseWriteLatency := 0.8 + rand.Float64()*2.0

		// Latency increases non-linearly with IOPS
		iopsLoadFactor := 1.0 + (readIOPS/params.DiskReadIOPS)*2.0
		readLatency := baseReadLatency * iopsLoadFactor
		writeLatency := baseWriteLatency * iopsLoadFactor

		// Set disk stats
		stats.Disks[i] = VMDiskStats{
			DiskID:          diskParams.ID,
			Path:            diskParams.Path,
			Size:            diskParams.SizeBytes,
			Used:            usedBytes,
			UsagePercent:    usagePercent,
			ReadIOPS:        readIOPS,
			WriteIOPS:       writeIOPS,
			ReadThroughput:  readThroughput,
			WriteThroughput: writeThroughput,
			ReadLatency:     readLatency,
			WriteLatency:    writeLatency,
			Type:            diskParams.Type,
		}
	}

	// ------------------------------------------------------------
	// Network stats generation
	// ------------------------------------------------------------
	networkCount := len(params.Networks)
	stats.Networks = make([]VMNetworkStats, networkCount)

	for i, netParams := range params.Networks {
		// Base network load with oscillation
		loadFactor := netParams.Load * (0.7 + rand.Float64()*0.6) *
			(1.0 + (stats.CPU.Usage/100.0)*0.3) // CPU affects network

		// Network spikes
		if rand.Float64() < params.SpikeProbability*timeFactor*1.5 { // Network spikes more common
			loadFactor *= 3.0 + rand.Float64()*7.0 // 3-10x spike
		}

		// Calculate rates - assuming 1Gbps link (125MB/s)
		maxThroughputBytes := 125.0 * 1024 * 1024

		// RX is usually higher than TX
		rxBytes := maxThroughputBytes * loadFactor * (0.7 + rand.Float64()*0.6)
		txBytes := maxThroughputBytes * loadFactor * (0.3 + rand.Float64()*0.4)

		// Packet size average: ~1500 bytes per packet
		rxPackets := rxBytes / 1500.0
		txPackets := txBytes / 1500.0

		// Errors and drops increase with load
		dropRate := 0.0001 + (loadFactor * loadFactor * 0.001)    // 0.01-0.11% base + load
		errorRate := 0.00001 + (loadFactor * loadFactor * 0.0001) // 0.001-0.011% base + load

		rxDropped := rxPackets * dropRate
		txDropped := txPackets * dropRate
		rxErrors := rxPackets * errorRate
		txErrors := txPackets * errorRate

		// Set network stats
		stats.Networks[i] = VMNetworkStats{
			InterfaceID: netParams.ID,
			Name:        netParams.Name,
			RxBytes:     rxBytes,
			TxBytes:     txBytes,
			RxPackets:   rxPackets,
			TxPackets:   txPackets,
			RxDropped:   rxDropped,
			TxDropped:   txDropped,
			RxErrors:    rxErrors,
			TxErrors:    txErrors,
		}
	}

	// ------------------------------------------------------------
	// Process stats generation (for detailed metrics only)
	// ------------------------------------------------------------
	if detailLevel >= DetailedMetrics {
		processCount := params.ProcessCount
		stats.Processes = make([]VMProcessStats, processCount)

		// CPU and memory available to distribute
		totalCPU := stats.CPU.Usage * float64(params.CPUCores)
		totalMemory := float64(stats.Memory.Used)

		// Top processes get most of the resources based on power law
		for i := 0; i < processCount; i++ {
			// Use rank to distribute resources (power law)
			rank := i + 1

			// Resource distribution
			// First process gets ~20% of resources, then 10%, 7%, etc.
			fraction := 1.0 / (float64(rank) * 1.5)

			// Randomize the allocation slightly
			fractionVariance := fraction * (0.7 + rand.Float64()*0.6)

			// Calculate amount of CPU and memory
			procCPU := totalCPU * fractionVariance
			if procCPU > 100.0 {
				procCPU = 100.0 // Cap at 100%
			}

			// Memory is a bit more evenly distributed
			memoryFraction := 1.0 / (float64(rank) * 1.2)
			memoryVariance := memoryFraction * (0.8 + rand.Float64()*0.4)
			procMemory := totalMemory * memoryVariance
			procMemoryPercent := (procMemory / float64(params.MemoryTotal)) * 100.0

			// IO operations correlate somewhat with CPU/memory
			ioFactor := (procCPU/100.0 + procMemoryPercent/100.0) / 2.0
			readIOPS := params.DiskReadIOPS * 0.1 * ioFactor * (0.5 + rand.Float64()*1.0)
			writeIOPS := params.DiskWriteIOPS * 0.1 * ioFactor * (0.5 + rand.Float64()*1.0)

			// Throughput correlates with IOPS (32KB blocks)
			readThroughput := readIOPS * 32 * 1024
			writeThroughput := writeIOPS * 32 * 1024

			// Open files is roughly based on memory usage
			openFiles := int64(20 + (procMemoryPercent * 2.0) + rand.Float64()*50)

			// Runtime is 0-24 hours
			runTime := rand.Float64() * 24 * 60 * 60

			// Generate a realistic process name and command
			procName, procCommand := m.generateProcessInfo(i)

			// Set process stats
			stats.Processes[i] = VMProcessStats{
				PID:             int64(i + 1), // Start PIDs at 1
				Name:            procName,
				Command:         procCommand,
				CPUUsage:        procCPU,
				MemoryUsage:     int64(procMemory),
				MemoryPercent:   procMemoryPercent,
				ReadIOPS:        readIOPS,
				WriteIOPS:       writeIOPS,
				ReadThroughput:  readThroughput,
				WriteThroughput: writeThroughput,
				OpenFiles:       openFiles,
				RunTime:         runTime,
			}
		}
	}

	return stats
}

// generateProcessInfo creates realistic process names and commands
func (m *MockVMManager) generateProcessInfo(index int) (string, string) {
	// Common process names and commands
	commonProcesses := []struct {
		Name    string
		Command string
	}{
		{"systemd", "/usr/lib/systemd/systemd --system --deserialize 31"},
		{"sshd", "/usr/sbin/sshd -D"},
		{"nginx", "/usr/sbin/nginx -g 'daemon on; master_process on;'"},
		{"postgres", "/usr/pgsql-14/bin/postgres -D /var/lib/pgsql/14/data"},
		{"java", "java -Xmx2048m -jar /opt/app/service.jar"},
		{"python", "python /opt/app/worker.py --workers 4"},
		{"node", "node /var/www/app/server.js"},
		{"redis-server", "redis-server 127.0.0.1:6379"},
		{"mongod", "/usr/bin/mongod --config /etc/mongod.conf"},
		{"httpd", "/usr/sbin/httpd -DFOREGROUND"},
		{"mysqld", "/usr/sbin/mysqld --daemonize --pid-file=/var/run/mysqld/mysqld.pid"},
		{"docker", "docker-containerd --config /var/run/docker/containerd/containerd.toml"},
		{"dockerd", "/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock"},
		{"kube-apiserver", "/usr/local/bin/kube-apiserver --etcd-servers=localhost:2379"},
		{"kubelet", "/usr/bin/kubelet --config=/var/lib/kubelet/config.yaml"},
		{"prometheus", "/bin/prometheus --config.file=/etc/prometheus/prometheus.yml"},
		{"grafana-server", "/usr/sbin/grafana-server --config=/etc/grafana/grafana.ini"},
		{"cron", "/usr/sbin/crond -n"},
		{"rsyslogd", "/usr/sbin/rsyslogd -n"},
		{"bash", "bash"},
		{"sh", "sh"},
		{"sshd", "sshd: user@pts/0"},
		{"gunicorn", "/usr/bin/gunicorn app:app --workers 4"},
		{"uwsgi", "uwsgi --ini /etc/uwsgi/apps-enabled/app.ini"},
		{"haproxy", "/usr/sbin/haproxy -f /etc/haproxy/haproxy.cfg"},
		{"memcached", "/usr/bin/memcached -m 64 -p 11211 -u memcached -l 127.0.0.1"},
		{"telegraf", "/usr/bin/telegraf --config /etc/telegraf/telegraf.conf"},
		{"chronyd", "/usr/sbin/chronyd"},
		{"collectd", "/usr/sbin/collectd -C /etc/collectd.conf -f"},
		{"ntpd", "/usr/sbin/ntpd -u ntp:ntp -g"},
	}

	// For the first few indices, use common processes
	if index < len(commonProcesses) {
		proc := commonProcesses[index]
		return proc.Name, proc.Command
	}

	// For others, generate random worker processes
	workerTypes := []string{"worker", "agent", "daemon", "service", "handler", "processor"}
	workerType := workerTypes[rand.Intn(len(workerTypes))]

	languages := []string{"java", "python", "node", "ruby", "go"}
	language := languages[rand.Intn(len(languages))]

	workerId := rand.Intn(10) + 1

	var name, command string

	switch language {
	case "java":
		name = fmt.Sprintf("%s%d", workerType, workerId)
		command = fmt.Sprintf("java -Xmx512m -jar /opt/app/%s.jar --id %d", workerType, workerId)
	case "python":
		name = "python"
		command = fmt.Sprintf("python /opt/app/%s.py --id %d --threads 2", workerType, workerId)
	case "node":
		name = "node"
		command = fmt.Sprintf("node /opt/app/%s.js --id %d", workerType, workerId)
	case "ruby":
		name = "ruby"
		command = fmt.Sprintf("ruby /opt/app/%s.rb --id %d", workerType, workerId)
	case "go":
		name = workerType
		command = fmt.Sprintf("/opt/app/%s --id %d", workerType, workerId)
	}

	return name, command
}

// copyVMStats creates a deep copy of VMStats
func (m *MockVMManager) copyVMStats(src *VMStats) *VMStats {
	if src == nil {
		return nil
	}

	dst := &VMStats{
		VMID:      src.VMID,
		CPU:       src.CPU,
		Memory:    src.Memory,
		Timestamp: src.Timestamp,
	}

	// Copy disk stats
	if len(src.Disks) > 0 {
		dst.Disks = make([]VMDiskStats, len(src.Disks))
		copy(dst.Disks, src.Disks)
	}

	// Copy network stats
	if len(src.Networks) > 0 {
		dst.Networks = make([]VMNetworkStats, len(src.Networks))
		copy(dst.Networks, src.Networks)
	}

	// Copy process stats if available
	if len(src.Processes) > 0 {
		dst.Processes = make([]VMProcessStats, len(src.Processes))
		copy(dst.Processes, src.Processes)
	}

	return dst
}
