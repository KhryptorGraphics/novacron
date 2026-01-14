package edge

import (
	"context"
	"runtime"
	"sync"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/disk"
	"github.com/shirou/gopsutil/v3/host"
	"github.com/shirou/gopsutil/v3/load"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/net"
)

// SystemInfo contains static system information
type SystemInfo struct {
	Architecture string            `json:"architecture"`
	OS           string            `json:"os"`
	Platform     string            `json:"platform"`
	Hostname     string            `json:"hostname"`
	CPUModel     string            `json:"cpu_model"`
	CPUCores     int               `json:"cpu_cores"`
	TotalMemory  uint64            `json:"total_memory_gb"`
	TotalStorage uint64            `json:"total_storage_gb"`
	NetworkIfaces []*NetworkInterface `json:"network_interfaces"`
	Capabilities []string          `json:"capabilities"`
}

// NetworkInterface represents a network interface
type NetworkInterface struct {
	Name      string   `json:"name"`
	Addresses []string `json:"addresses"`
	MTU       int      `json:"mtu"`
	Flags     []string `json:"flags"`
}

// SystemMetrics contains current system metrics
type SystemMetrics struct {
	Timestamp    time.Time          `json:"timestamp"`
	CPU          *CPUMetrics        `json:"cpu"`
	Memory       *MemoryMetrics     `json:"memory"`
	Disk         map[string]*DiskMetrics `json:"disk"`
	Network      *NetworkMetrics    `json:"network"`
	Load         *LoadMetrics       `json:"load"`
	Processes    *ProcessMetrics    `json:"processes"`
	Temperature  *TemperatureMetrics `json:"temperature,omitempty"`
}

// CPUMetrics contains CPU usage information
type CPUMetrics struct {
	Usage      []float64 `json:"usage_per_core"`
	Average    float64   `json:"average_usage"`
	UserTime   float64   `json:"user_time"`
	SystemTime float64   `json:"system_time"`
	IdleTime   float64   `json:"idle_time"`
	IOWaitTime float64   `json:"iowait_time"`
	Frequency  []float64 `json:"frequency_mhz"`
}

// MemoryMetrics contains memory usage information
type MemoryMetrics struct {
	Total     uint64  `json:"total_bytes"`
	Used      uint64  `json:"used_bytes"`
	Free      uint64  `json:"free_bytes"`
	Available uint64  `json:"available_bytes"`
	Cached    uint64  `json:"cached_bytes"`
	Buffers   uint64  `json:"buffers_bytes"`
	UsedPct   float64 `json:"used_percent"`
	// Swap information
	SwapTotal uint64  `json:"swap_total_bytes"`
	SwapUsed  uint64  `json:"swap_used_bytes"`
	SwapFree  uint64  `json:"swap_free_bytes"`
	SwapPct   float64 `json:"swap_percent"`
}

// DiskMetrics contains disk usage information
type DiskMetrics struct {
	Device     string  `json:"device"`
	Mountpoint string  `json:"mountpoint"`
	Fstype     string  `json:"filesystem_type"`
	Total      uint64  `json:"total_bytes"`
	Used       uint64  `json:"used_bytes"`
	Free       uint64  `json:"free_bytes"`
	UsedPct    float64 `json:"used_percent"`
	// IO statistics
	ReadBytes  uint64 `json:"read_bytes"`
	WriteBytes uint64 `json:"write_bytes"`
	ReadOps    uint64 `json:"read_operations"`
	WriteOps   uint64 `json:"write_operations"`
	ReadTime   uint64 `json:"read_time_ms"`
	WriteTime  uint64 `json:"write_time_ms"`
	IOTime     uint64 `json:"io_time_ms"`
}

// NetworkMetrics contains network usage information
type NetworkMetrics struct {
	Interfaces  map[string]*NetInterfaceMetrics `json:"interfaces"`
	TotalRxBytes uint64 `json:"total_rx_bytes"`
	TotalTxBytes uint64 `json:"total_tx_bytes"`
	TotalRxPackets uint64 `json:"total_rx_packets"`
	TotalTxPackets uint64 `json:"total_tx_packets"`
	TotalErrors    uint64 `json:"total_errors"`
	TotalDropped   uint64 `json:"total_dropped"`
}

// NetInterfaceMetrics contains per-interface network metrics
type NetInterfaceMetrics struct {
	Name       string `json:"name"`
	RxBytes    uint64 `json:"rx_bytes"`
	TxBytes    uint64 `json:"tx_bytes"`
	RxPackets  uint64 `json:"rx_packets"`
	TxPackets  uint64 `json:"tx_packets"`
	RxErrors   uint64 `json:"rx_errors"`
	TxErrors   uint64 `json:"tx_errors"`
	RxDropped  uint64 `json:"rx_dropped"`
	TxDropped  uint64 `json:"tx_dropped"`
	Speed      uint64 `json:"speed_mbps"`
}

// LoadMetrics contains system load information
type LoadMetrics struct {
	Load1  float64 `json:"load_1min"`
	Load5  float64 `json:"load_5min"`
	Load15 float64 `json:"load_15min"`
}

// ProcessMetrics contains process information
type ProcessMetrics struct {
	Total      int `json:"total_processes"`
	Running    int `json:"running_processes"`
	Sleeping   int `json:"sleeping_processes"`
	Zombie     int `json:"zombie_processes"`
	Stopped    int `json:"stopped_processes"`
}

// TemperatureMetrics contains temperature information
type TemperatureMetrics struct {
	CPU    float64 `json:"cpu_temperature_celsius"`
	GPU    float64 `json:"gpu_temperature_celsius,omitempty"`
	System float64 `json:"system_temperature_celsius,omitempty"`
}

// MetricSnapshot represents a point-in-time metric collection
type MetricSnapshot struct {
	Timestamp time.Time      `json:"timestamp"`
	AgentID   string         `json:"agent_id"`
	Metrics   *SystemMetrics `json:"metrics"`
}

// SystemMonitor monitors system resources and metrics
type SystemMonitor struct {
	interval  time.Duration
	startTime time.Time

	// Current metrics
	currentMetrics *SystemMetrics
	systemInfo     *SystemInfo
	metricsMutex   sync.RWMutex

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// Previous values for calculating deltas
	prevNetStats  map[string]net.IOCountersStat
	prevDiskStats map[string]disk.IOCountersStat
	prevCPUStats  []cpu.TimesStat
}

// NewSystemMonitor creates a new system monitor
func NewSystemMonitor(interval time.Duration) *SystemMonitor {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &SystemMonitor{
		interval:      interval,
		startTime:     time.Now(),
		ctx:           ctx,
		cancel:        cancel,
		prevNetStats:  make(map[string]net.IOCountersStat),
		prevDiskStats: make(map[string]disk.IOCountersStat),
	}
}

// Start begins system monitoring
func (s *SystemMonitor) Start() error {
	// Collect initial system information
	if err := s.collectSystemInfo(); err != nil {
		return err
	}

	// Collect initial metrics
	if err := s.collectMetrics(); err != nil {
		return err
	}

	// Start monitoring loop
	s.wg.Add(1)
	go s.monitorLoop()

	return nil
}

// Stop stops system monitoring
func (s *SystemMonitor) Stop() error {
	s.cancel()
	s.wg.Wait()
	return nil
}

// GetSystemInfo returns static system information
func (s *SystemMonitor) GetSystemInfo() *SystemInfo {
	s.metricsMutex.RLock()
	defer s.metricsMutex.RUnlock()
	return s.systemInfo
}

// GetCurrentMetrics returns current system metrics
func (s *SystemMonitor) GetCurrentMetrics() *SystemMetrics {
	s.metricsMutex.RLock()
	defer s.metricsMutex.RUnlock()
	return s.currentMetrics
}

// GetDetailedMetrics returns detailed system metrics with additional context
func (s *SystemMonitor) GetDetailedMetrics() *SystemMetrics {
	s.metricsMutex.RLock()
	defer s.metricsMutex.RUnlock()
	
	// Return a copy with additional calculated metrics
	metrics := *s.currentMetrics
	return &metrics
}

// monitorLoop periodically collects system metrics
func (s *SystemMonitor) monitorLoop() {
	defer s.wg.Done()

	ticker := time.NewTicker(s.interval)
	defer ticker.Stop()

	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			if err := s.collectMetrics(); err != nil {
				// Log error but continue monitoring
				continue
			}
		}
	}
}

// collectSystemInfo gathers static system information
func (s *SystemMonitor) collectSystemInfo() error {
	hostInfo, err := host.Info()
	if err != nil {
		return err
	}

	cpuInfo, err := cpu.Info()
	if err != nil {
		return err
	}

	memInfo, err := mem.VirtualMemory()
	if err != nil {
		return err
	}

	// Network interfaces
	netInterfaces, err := net.Interfaces()
	if err != nil {
		return err
	}

	// Disk information
	diskPartitions, err := disk.Partitions(true)
	if err != nil {
		return err
	}

	var totalStorage uint64
	for _, partition := range diskPartitions {
		if usage, err := disk.Usage(partition.Mountpoint); err == nil {
			totalStorage += usage.Total
		}
	}

	// Build network interface info
	networkIfaces := make([]*NetworkInterface, len(netInterfaces))
	for i, iface := range netInterfaces {
		networkIfaces[i] = &NetworkInterface{
			Name:      iface.Name,
			Addresses: iface.Addrs,
			MTU:       iface.MTU,
			Flags:     iface.Flags,
		}
	}

	// Determine capabilities
	capabilities := s.detectCapabilities()

	s.systemInfo = &SystemInfo{
		Architecture: runtime.GOARCH,
		OS:           runtime.GOOS,
		Platform:     hostInfo.Platform,
		Hostname:     hostInfo.Hostname,
		CPUModel:     cpuInfo[0].ModelName,
		CPUCores:     int(cpuInfo[0].Cores),
		TotalMemory:  memInfo.Total / (1024 * 1024 * 1024), // Convert to GB
		TotalStorage: totalStorage / (1024 * 1024 * 1024),  // Convert to GB
		NetworkIfaces: networkIfaces,
		Capabilities: capabilities,
	}

	return nil
}

// detectCapabilities detects system capabilities
func (s *SystemMonitor) detectCapabilities() []string {
	capabilities := []string{
		"monitoring",
		"caching",
		"task_execution",
	}

	// Architecture-specific capabilities
	switch runtime.GOARCH {
	case "arm64", "arm":
		capabilities = append(capabilities, "low_power_computing")
	case "riscv64":
		capabilities = append(capabilities, "riscv_architecture")
	case "amd64":
		capabilities = append(capabilities, "high_performance_computing")
	}

	// OS-specific capabilities
	if runtime.GOOS == "linux" {
		capabilities = append(capabilities, "containers", "cgroups")
	}

	return capabilities
}

// collectMetrics gathers current system metrics
func (s *SystemMonitor) collectMetrics() error {
	timestamp := time.Now()

	// CPU metrics
	cpuMetrics, err := s.collectCPUMetrics()
	if err != nil {
		return err
	}

	// Memory metrics
	memMetrics, err := s.collectMemoryMetrics()
	if err != nil {
		return err
	}

	// Disk metrics
	diskMetrics, err := s.collectDiskMetrics()
	if err != nil {
		return err
	}

	// Network metrics
	netMetrics, err := s.collectNetworkMetrics()
	if err != nil {
		return err
	}

	// Load metrics
	loadMetrics, err := s.collectLoadMetrics()
	if err != nil {
		return err
	}

	// Process metrics
	procMetrics := s.collectProcessMetrics()

	// Temperature metrics (best effort)
	tempMetrics := s.collectTemperatureMetrics()

	s.metricsMutex.Lock()
	s.currentMetrics = &SystemMetrics{
		Timestamp:   timestamp,
		CPU:         cpuMetrics,
		Memory:      memMetrics,
		Disk:        diskMetrics,
		Network:     netMetrics,
		Load:        loadMetrics,
		Processes:   procMetrics,
		Temperature: tempMetrics,
	}
	s.metricsMutex.Unlock()

	return nil
}

// collectCPUMetrics collects CPU usage metrics
func (s *SystemMonitor) collectCPUMetrics() (*CPUMetrics, error) {
	// Get per-CPU usage
	perCPU, err := cpu.Percent(0, true)
	if err != nil {
		return nil, err
	}

	// Get overall CPU usage
	overall, err := cpu.Percent(0, false)
	if err != nil {
		return nil, err
	}

	// Get CPU times
	times, err := cpu.Times(false)
	if err != nil {
		return nil, err
	}

	// Get CPU frequencies (best effort)
	frequencies, _ := cpu.Info()
	freqs := make([]float64, len(frequencies))
	for i, freq := range frequencies {
		freqs[i] = freq.Mhz
	}

	var avgUsage float64
	if len(overall) > 0 {
		avgUsage = overall[0]
	}

	cpuTimes := times[0]
	
	return &CPUMetrics{
		Usage:      perCPU,
		Average:    avgUsage,
		UserTime:   cpuTimes.User,
		SystemTime: cpuTimes.System,
		IdleTime:   cpuTimes.Idle,
		IOWaitTime: cpuTimes.Iowait,
		Frequency:  freqs,
	}, nil
}

// collectMemoryMetrics collects memory usage metrics
func (s *SystemMonitor) collectMemoryMetrics() (*MemoryMetrics, error) {
	vmem, err := mem.VirtualMemory()
	if err != nil {
		return nil, err
	}

	swap, err := mem.SwapMemory()
	if err != nil {
		// Swap might not be available
		swap = &mem.SwapMemoryStat{}
	}

	return &MemoryMetrics{
		Total:     vmem.Total,
		Used:      vmem.Used,
		Free:      vmem.Free,
		Available: vmem.Available,
		Cached:    vmem.Cached,
		Buffers:   vmem.Buffers,
		UsedPct:   vmem.UsedPercent,
		SwapTotal: swap.Total,
		SwapUsed:  swap.Used,
		SwapFree:  swap.Free,
		SwapPct:   swap.UsedPercent,
	}, nil
}

// collectDiskMetrics collects disk usage and I/O metrics
func (s *SystemMonitor) collectDiskMetrics() (map[string]*DiskMetrics, error) {
	partitions, err := disk.Partitions(true)
	if err != nil {
		return nil, err
	}

	diskMetrics := make(map[string]*DiskMetrics)

	for _, partition := range partitions {
		usage, err := disk.Usage(partition.Mountpoint)
		if err != nil {
			continue
		}

		// Get I/O statistics
		ioCounters, err := disk.IOCounters(partition.Device)
		if err != nil {
			// Continue without I/O stats
			diskMetrics[partition.Device] = &DiskMetrics{
				Device:     partition.Device,
				Mountpoint: partition.Mountpoint,
				Fstype:     partition.Fstype,
				Total:      usage.Total,
				Used:       usage.Used,
				Free:       usage.Free,
				UsedPct:    usage.UsedPercent,
			}
			continue
		}

		// Get current I/O stats
		var ioStat disk.IOCountersStat
		if stats, exists := ioCounters[partition.Device]; exists {
			ioStat = stats
		}

		diskMetrics[partition.Device] = &DiskMetrics{
			Device:     partition.Device,
			Mountpoint: partition.Mountpoint,
			Fstype:     partition.Fstype,
			Total:      usage.Total,
			Used:       usage.Used,
			Free:       usage.Free,
			UsedPct:    usage.UsedPercent,
			ReadBytes:  ioStat.ReadBytes,
			WriteBytes: ioStat.WriteBytes,
			ReadOps:    ioStat.ReadCount,
			WriteOps:   ioStat.WriteCount,
			ReadTime:   ioStat.ReadTime,
			WriteTime:  ioStat.WriteTime,
			IOTime:     ioStat.IoTime,
		}
	}

	return diskMetrics, nil
}

// collectNetworkMetrics collects network usage metrics
func (s *SystemMonitor) collectNetworkMetrics() (*NetworkMetrics, error) {
	netIO, err := net.IOCounters(true)
	if err != nil {
		return nil, err
	}

	interfaces := make(map[string]*NetInterfaceMetrics)
	var totalRx, totalTx, totalRxPackets, totalTxPackets, totalErrors, totalDropped uint64

	for _, io := range netIO {
		// Skip loopback interfaces
		if io.Name == "lo" {
			continue
		}

		interfaces[io.Name] = &NetInterfaceMetrics{
			Name:      io.Name,
			RxBytes:   io.BytesRecv,
			TxBytes:   io.BytesSent,
			RxPackets: io.PacketsRecv,
			TxPackets: io.PacketsSent,
			RxErrors:  io.Errin,
			TxErrors:  io.Errout,
			RxDropped: io.Dropin,
			TxDropped: io.Dropout,
		}

		totalRx += io.BytesRecv
		totalTx += io.BytesSent
		totalRxPackets += io.PacketsRecv
		totalTxPackets += io.PacketsSent
		totalErrors += io.Errin + io.Errout
		totalDropped += io.Dropin + io.Dropout
	}

	return &NetworkMetrics{
		Interfaces:     interfaces,
		TotalRxBytes:   totalRx,
		TotalTxBytes:   totalTx,
		TotalRxPackets: totalRxPackets,
		TotalTxPackets: totalTxPackets,
		TotalErrors:    totalErrors,
		TotalDropped:   totalDropped,
	}, nil
}

// collectLoadMetrics collects system load metrics
func (s *SystemMonitor) collectLoadMetrics() (*LoadMetrics, error) {
	loadAvg, err := load.Avg()
	if err != nil {
		return nil, err
	}

	return &LoadMetrics{
		Load1:  loadAvg.Load1,
		Load5:  loadAvg.Load5,
		Load15: loadAvg.Load15,
	}, nil
}

// collectProcessMetrics collects process information
func (s *SystemMonitor) collectProcessMetrics() *ProcessMetrics {
	// This is a simplified implementation
	// In practice, you'd enumerate processes and count by state
	return &ProcessMetrics{
		Total:    runtime.NumGoroutine(), // Placeholder
		Running:  1,                      // Placeholder
		Sleeping: 0,                      // Placeholder
		Zombie:   0,                      // Placeholder
		Stopped:  0,                      // Placeholder
	}
}

// collectTemperatureMetrics collects temperature metrics (best effort)
func (s *SystemMonitor) collectTemperatureMetrics() *TemperatureMetrics {
	// Temperature monitoring is platform-specific and may not be available
	// This is a placeholder - real implementation would read from /sys/class/thermal
	// or use platform-specific APIs
	return nil
}