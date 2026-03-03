package vm

import (
	"context"
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Ubuntu2404PerformanceOptimizer provides performance optimization for Ubuntu 24.04 VMs
type Ubuntu2404PerformanceOptimizer struct {
	// KVM driver reference
	Driver *KVMDriver
	
	// Performance profiles
	Profiles map[string]PerformanceProfile
}

// PerformanceProfile defines a set of performance optimizations
type PerformanceProfile struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	CPUSettings CPUOptimizationSettings `json:"cpu_settings"`
	IOSettings  IOOptimizationSettings  `json:"io_settings"`
	NetSettings NetOptimizationSettings `json:"net_settings"`
	MemSettings MemOptimizationSettings `json:"mem_settings"`
}

// CPUOptimizationSettings contains CPU optimization settings
type CPUOptimizationSettings struct {
	CPUModel       string `json:"cpu_model"`
	CPUFeatures    string `json:"cpu_features"`
	CPUPinning     bool   `json:"cpu_pinning"`
	CPUShares      int    `json:"cpu_shares"`
	CPUQuota       int    `json:"cpu_quota"`
	CPUPeriod      int    `json:"cpu_period"`
	CPUThreads     int    `json:"cpu_threads"`
	CPUSockets     int    `json:"cpu_sockets"`
	CPUCores       int    `json:"cpu_cores"`
	CPUCache       string `json:"cpu_cache"`
	CPUGovernor    string `json:"cpu_governor"`
	CPUBoostMode   string `json:"cpu_boost_mode"`
	CPUPriority    int    `json:"cpu_priority"`
}

// IOOptimizationSettings contains I/O optimization settings
type IOOptimizationSettings struct {
	DiskCacheMode     string `json:"disk_cache_mode"`
	DiskIOMode        string `json:"disk_io_mode"`
	DiskIOThread      bool   `json:"disk_io_thread"`
	DiskAIOBackend    string `json:"disk_aio_backend"`
	DiskWriteThrough  bool   `json:"disk_write_through"`
	DiskWriteBack     bool   `json:"disk_write_back"`
	DiskBarrier       bool   `json:"disk_barrier"`
	DiskIOScheduler   string `json:"disk_io_scheduler"`
	DiskReadAhead     int    `json:"disk_read_ahead"`
	DiskIOPriority    int    `json:"disk_io_priority"`
	DiskIONice        int    `json:"disk_io_nice"`
	DiskIOWeight      int    `json:"disk_io_weight"`
	DiskIOLatency     int    `json:"disk_io_latency"`
	DiskIOThroughput  int    `json:"disk_io_throughput"`
}

// NetOptimizationSettings contains network optimization settings
type NetOptimizationSettings struct {
	NetModel         string `json:"net_model"`
	NetQueues        int    `json:"net_queues"`
	NetMTU           int    `json:"net_mtu"`
	NetTXQueueLen    int    `json:"net_tx_queue_len"`
	NetRXQueueLen    int    `json:"net_rx_queue_len"`
	NetTCPNoDelay    bool   `json:"net_tcp_no_delay"`
	NetTCPFastOpen   bool   `json:"net_tcp_fast_open"`
	NetTCPKeepAlive  bool   `json:"net_tcp_keep_alive"`
	NetTCPCongestion string `json:"net_tcp_congestion"`
	NetVHostNet      bool   `json:"net_vhost_net"`
	NetVHostUser     bool   `json:"net_vhost_user"`
	NetMultiQueue    bool   `json:"net_multi_queue"`
	NetIOThread      bool   `json:"net_io_thread"`
}

// MemOptimizationSettings contains memory optimization settings
type MemOptimizationSettings struct {
	MemBalloon       bool   `json:"mem_balloon"`
	MemShared        bool   `json:"mem_shared"`
	MemLocked        bool   `json:"mem_locked"`
	MemHugePages     bool   `json:"mem_huge_pages"`
	MemHugePageSize  string `json:"mem_huge_page_size"`
	MemSwapHardLimit int    `json:"mem_swap_hard_limit"`
	MemSwapSoftLimit int    `json:"mem_swap_soft_limit"`
	MemKSM           bool   `json:"mem_ksm"`
	MemMergeAcross   bool   `json:"mem_merge_across"`
	MemOvercommit    float64 `json:"mem_overcommit"`
	MemCompression   bool   `json:"mem_compression"`
	MemNuma          bool   `json:"mem_numa"`
	MemNumaNodes     int    `json:"mem_numa_nodes"`
}

// PerformanceMetrics contains VM performance metrics
type PerformanceMetrics struct {
	VMID            string    `json:"vm_id"`
	Timestamp       time.Time `json:"timestamp"`
	CPUUsage        float64   `json:"cpu_usage"`
	MemoryUsage     float64   `json:"memory_usage"`
	DiskIORead      int64     `json:"disk_io_read"`
	DiskIOWrite     int64     `json:"disk_io_write"`
	NetworkReceived int64     `json:"network_received"`
	NetworkSent     int64     `json:"network_sent"`
	IOWait          float64   `json:"io_wait"`
	LoadAverage     [3]float64 `json:"load_average"`
}

// NewUbuntu2404PerformanceOptimizer creates a new performance optimizer for Ubuntu 24.04 VMs
func NewUbuntu2404PerformanceOptimizer(driver *KVMDriver) *Ubuntu2404PerformanceOptimizer {
	return &Ubuntu2404PerformanceOptimizer{
		Driver:   driver,
		Profiles: createDefaultPerformanceProfiles(),
	}
}

// createDefaultPerformanceProfiles creates default performance profiles
func createDefaultPerformanceProfiles() map[string]PerformanceProfile {
	profiles := make(map[string]PerformanceProfile)
	
	// Balanced profile (default)
	profiles["balanced"] = PerformanceProfile{
		Name:        "balanced",
		Description: "Balanced performance profile for general workloads",
		CPUSettings: CPUOptimizationSettings{
			CPUModel:     "host",
			CPUFeatures:  "+pcid,+ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,+rdrand",
			CPUPinning:   false,
			CPUShares:    1024,
			CPUThreads:   2,
			CPUSockets:   1,
			CPUCores:     2,
			CPUGovernor:  "ondemand",
			CPUBoostMode: "conservative",
			CPUPriority:  0,
		},
		IOSettings: IOOptimizationSettings{
			DiskCacheMode:    "writeback",
			DiskIOMode:       "native",
			DiskIOThread:     true,
			DiskAIOBackend:   "io_uring",
			DiskWriteBack:    true,
			DiskIOScheduler:  "mq-deadline",
			DiskReadAhead:    256,
			DiskIOPriority:   4,
			DiskIOWeight:     500,
		},
		NetSettings: NetOptimizationSettings{
			NetModel:         "virtio-net",
			NetQueues:        2,
			NetMTU:           1500,
			NetTXQueueLen:    1000,
			NetTCPNoDelay:    true,
			NetTCPFastOpen:   true,
			NetTCPKeepAlive:  true,
			NetTCPCongestion: "cubic",
			NetVHostNet:      true,
			NetMultiQueue:    true,
		},
		MemSettings: MemOptimizationSettings{
			MemBalloon:      true,
			MemShared:       false,
			MemLocked:       false,
			MemHugePages:    true,
			MemHugePageSize: "2MB",
			MemKSM:          true,
			MemOvercommit:   1.0,
			MemCompression:  false,
			MemNuma:         false,
		},
	}
	
	// High performance profile
	profiles["high-performance"] = PerformanceProfile{
		Name:        "high-performance",
		Description: "High performance profile for compute-intensive workloads",
		CPUSettings: CPUOptimizationSettings{
			CPUModel:     "host-passthrough",
			CPUFeatures:  "+pcid,+ssse3,+sse4.2,+popcnt,+avx,+avx2,+aes,+xsave,+rdrand",
			CPUPinning:   true,
			CPUShares:    2048,
			CPUThreads:   2,
			CPUSockets:   1,
			CPUCores:     4,
			CPUGovernor:  "performance",
			CPUBoostMode: "aggressive",
			CPUPriority:  -10,
		},
		IOSettings: IOOptimizationSettings{
			DiskCacheMode:    "unsafe",
			DiskIOMode:       "native",
			DiskIOThread:     true,
			DiskAIOBackend:   "io_uring",
			DiskWriteBack:    true,
			DiskIOScheduler:  "none",
			DiskReadAhead:    512,
			DiskIOPriority:   0,
			DiskIOWeight:     800,
		},
		NetSettings: NetOptimizationSettings{
			NetModel:         "virtio-net",
			NetQueues:        4,
			NetMTU:           9000,
			NetTXQueueLen:    10000,
			NetTCPNoDelay:    true,
			NetTCPFastOpen:   true,
			NetTCPKeepAlive:  true,
			NetTCPCongestion: "bbr",
			NetVHostNet:      true,
			NetMultiQueue:    true,
			NetIOThread:      true,
		},
		MemSettings: MemOptimizationSettings{
			MemBalloon:      false,
			MemShared:       false,
			MemLocked:       true,
			MemHugePages:    true,
			MemHugePageSize: "1GB",
			MemKSM:          false,
			MemOvercommit:   0.8,
			MemCompression:  false,
			MemNuma:         true,
			MemNumaNodes:    2,
		},
	}
	
	// Low latency profile
	profiles["low-latency"] = PerformanceProfile{
		Name:        "low-latency",
		Description: "Low latency profile for real-time workloads",
		CPUSettings: CPUOptimizationSettings{
			CPUModel:     "host-passthrough",
			CPUFeatures:  "+tsc-deadline,+invtsc,+pcid,+ssse3,+sse4.2,+popcnt,+avx,+aes,+xsave,+rdrand",
			CPUPinning:   true,
			CPUShares:    2048,
			CPUThreads:   1,
			CPUSockets:   1,
			CPUCores:     4,
			CPUGovernor:  "performance",
			CPUBoostMode: "aggressive",
			CPUPriority:  -20,
		},
		IOSettings: IOOptimizationSettings{
			DiskCacheMode:    "none",
			DiskIOMode:       "native",
			DiskIOThread:     true,
			DiskAIOBackend:   "io_uring",
			DiskWriteThrough: true,
			DiskBarrier:      false,
			DiskIOScheduler:  "none",
			DiskIOPriority:   0,
			DiskIOLatency:    10,
		},
		NetSettings: NetOptimizationSettings{
			NetModel:         "virtio-net",
			NetQueues:        8,
			NetMTU:           1500,
			NetTXQueueLen:    50000,
			NetTCPNoDelay:    true,
			NetTCPFastOpen:   true,
			NetTCPKeepAlive:  true,
			NetTCPCongestion: "bbr",
			NetVHostNet:      true,
			NetMultiQueue:    true,
			NetIOThread:      true,
		},
		MemSettings: MemOptimizationSettings{
			MemBalloon:      false,
			MemShared:       false,
			MemLocked:       true,
			MemHugePages:    true,
			MemHugePageSize: "1GB",
			MemKSM:          false,
			MemOvercommit:   0.5,
			MemCompression:  false,
			MemNuma:         true,
			MemNumaNodes:    2,
		},
	}
	
	// Energy efficient profile
	profiles["energy-efficient"] = PerformanceProfile{
		Name:        "energy-efficient",
		Description: "Energy efficient profile for low power consumption",
		CPUSettings: CPUOptimizationSettings{
			CPUModel:     "qemu64",
			CPUFeatures:  "+pcid,+ssse3,+sse4.2,+aes",
			CPUPinning:   false,
			CPUShares:    512,
			CPUThreads:   1,
			CPUSockets:   1,
			CPUCores:     1,
			CPUGovernor:  "powersave",
			CPUBoostMode: "conservative",
			CPUPriority:  10,
		},
		IOSettings: IOOptimizationSettings{
			DiskCacheMode:    "writeback",
			DiskIOMode:       "threads",
			DiskIOThread:     false,
			DiskAIOBackend:   "threads",
			DiskWriteBack:    true,
			DiskIOScheduler:  "mq-deadline",
			DiskReadAhead:    128,
			DiskIOPriority:   7,
			DiskIOWeight:     100,
		},
		NetSettings: NetOptimizationSettings{
			NetModel:         "virtio-net",
			NetQueues:        1,
			NetMTU:           1500,
			NetTXQueueLen:    500,
			NetTCPNoDelay:    false,
			NetTCPFastOpen:   false,
			NetTCPKeepAlive:  true,
			NetTCPCongestion: "cubic",
			NetVHostNet:      false,
			NetMultiQueue:    false,
		},
		MemSettings: MemOptimizationSettings{
			MemBalloon:      true,
			MemShared:       true,
			MemLocked:       false,
			MemHugePages:    false,
			MemKSM:          true,
			MemOvercommit:   1.5,
			MemCompression:  true,
			MemNuma:         false,
		},
	}
	
	return profiles
}

// ApplyPerformanceProfile applies a performance profile to a VM
func (o *Ubuntu2404PerformanceOptimizer) ApplyPerformanceProfile(ctx context.Context, vmID, profileName string) error {
	log.Printf("Applying performance profile %s to VM %s", profileName, vmID)
	
	// Get VM info
	o.Driver.vmLock.RLock()
	vmInfo, exists := o.Driver.vms[vmID]
	o.Driver.vmLock.RUnlock()
	
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}
	
	// Get profile
	profile, exists := o.Profiles[profileName]
	if !exists {
		return fmt.Errorf("performance profile %s not found", profileName)
	}
	
	// Check if VM is running
	vmState, err := o.Driver.GetStatus(ctx, vmID)
	if err != nil {
		return fmt.Errorf("failed to get VM status: %w", err)
	}
	
	// If VM is running, we need to stop it to apply some settings
	needsRestart := false
	if vmState == VMStateRunning {
		// Some settings require VM restart
		if profile.CPUSettings.CPUModel != "" || 
		   profile.CPUSettings.CPUFeatures != "" ||
		   profile.CPUSettings.CPUPinning ||
		   profile.CPUSettings.CPUThreads > 0 ||
		   profile.CPUSettings.CPUSockets > 0 ||
		   profile.CPUSettings.CPUCores > 0 ||
		   profile.IOSettings.DiskCacheMode != "" ||
		   profile.IOSettings.DiskIOMode != "" ||
		   profile.NetSettings.NetModel != "" ||
		   profile.NetSettings.NetQueues > 0 ||
		   profile.NetSettings.NetVHostNet ||
		   profile.NetSettings.NetMultiQueue ||
		   profile.MemSettings.MemHugePages ||
		   profile.MemSettings.MemLocked ||
		   profile.MemSettings.MemNuma {
			
			log.Printf("VM needs to be restarted to apply performance profile")
			needsRestart = true
			
			// Stop the VM
			if err := o.Driver.Stop(ctx, vmID); err != nil {
				return fmt.Errorf("failed to stop VM: %w", err)
			}
		}
	}
	
	// Apply CPU settings
	if err := o.applyCPUSettings(ctx, vmID, vmInfo, profile.CPUSettings); err != nil {
		log.Printf("Warning: Failed to apply CPU settings: %v", err)
	}
	
	// Apply I/O settings
	if err := o.applyIOSettings(ctx, vmID, vmInfo, profile.IOSettings); err != nil {
		log.Printf("Warning: Failed to apply I/O settings: %v", err)
	}
	
	// Apply network settings
	if err := o.applyNetworkSettings(ctx, vmID, vmInfo, profile.NetSettings); err != nil {
		log.Printf("Warning: Failed to apply network settings: %v", err)
	}
	
	// Apply memory settings
	if err := o.applyMemorySettings(ctx, vmID, vmInfo, profile.MemSettings); err != nil {
		log.Printf("Warning: Failed to apply memory settings: %v", err)
	}
	
	// Restart VM if needed
	if needsRestart && vmState == VMStateRunning {
		if err := o.Driver.Start(ctx, vmID); err != nil {
			return fmt.Errorf("failed to restart VM: %w", err)
		}
	}
	
	log.Printf("Applied performance profile %s to VM %s", profileName, vmID)
	return nil
}

// GetPerformanceMetrics gets performance metrics for a VM
func (o *Ubuntu2404PerformanceOptimizer) GetPerformanceMetrics(ctx context.Context, vmID string) (*PerformanceMetrics, error) {
	log.Printf("Getting performance metrics for VM %s", vmID)
	
	// Get VM info
	o.Driver.vmLock.RLock()
	vmInfo, exists := o.Driver.vms[vmID]
	o.Driver.vmLock.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}
	
	// Check if VM is running
	vmState, err := o.Driver.GetStatus(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM status: %w", err)
	}
	
	if vmState != VMStateRunning {
		return nil, fmt.Errorf("VM is not running")
	}
	
	// Get metrics using QMP
	metrics := &PerformanceMetrics{
		VMID:      vmID,
		Timestamp: time.Now(),
	}
	
	// Get CPU usage
	cpuUsage, err := o.getVMCPUUsage(ctx, vmInfo)
	if err != nil {
		log.Printf("Warning: Failed to get CPU usage: %v", err)
	} else {
		metrics.CPUUsage = cpuUsage
	}
	
	// Get memory usage
	memoryUsage, err := o.getVMMemoryUsage(ctx, vmInfo)
	if err != nil {
		log.Printf("Warning: Failed to get memory usage: %v", err)
	} else {
		metrics.MemoryUsage = memoryUsage
	}
	
	// Get disk I/O
	diskRead, diskWrite, err := o.getVMDiskIO(ctx, vmInfo)
	if err != nil {
		log.Printf("Warning: Failed to get disk I/O: %v", err)
	} else {
		metrics.DiskIORead = diskRead
		metrics.DiskIOWrite = diskWrite
	}
	
	// Get network I/O
	netReceived, netSent, err := o.getVMNetworkIO(ctx, vmInfo)
	if err != nil {
		log.Printf("Warning: Failed to get network I/O: %v", err)
	} else {
		metrics.NetworkReceived = netReceived
		metrics.NetworkSent = netSent
	}
	
	// Get I/O wait
	ioWait, err := o.getVMIOWait(ctx, vmInfo)
	if err != nil {
		log.Printf("Warning: Failed to get I/O wait: %v", err)
	} else {
		metrics.IOWait = ioWait
	}
	
	// Get load average
	loadAvg, err := o.getVMLoadAverage(ctx, vmInfo)
	if err != nil {
		log.Printf("Warning: Failed to get load average: %v", err)
	} else {
		metrics.LoadAverage = loadAvg
	}
	
	return metrics, nil
}

// ListPerformanceProfiles lists all available performance profiles
func (o *Ubuntu2404PerformanceOptimizer) ListPerformanceProfiles() []PerformanceProfile {
	profiles := make([]PerformanceProfile, 0, len(o.Profiles))
	for _, profile := range o.Profiles {
		profiles = append(profiles, profile)
	}
	return profiles
}

// Helper function to apply CPU settings
func (o *Ubuntu2404PerformanceOptimizer) applyCPUSettings(ctx context.Context, vmID string, vmInfo *KVMInfo, settings CPUOptimizationSettings) error {
	// In a real implementation, this would modify the VM's QEMU command line
	// or use QMP to apply settings to a running VM
	
	// For now, we'll just update the VM's configuration file
	// In a real implementation, this would be more sophisticated
	
	return nil
}

// Helper function to apply I/O settings
func (o *Ubuntu2404PerformanceOptimizer) applyIOSettings(ctx context.Context, vmID string, vmInfo *KVMInfo, settings IOOptimizationSettings) error {
	// In a real implementation, this would modify the VM's QEMU command line
	// or use QMP to apply settings to a running VM
	
	return nil
}

// Helper function to apply network settings
func (o *Ubuntu2404PerformanceOptimizer) applyNetworkSettings(ctx context.Context, vmID string, vmInfo *KVMInfo, settings NetOptimizationSettings) error {
	// In a real implementation, this would modify the VM's QEMU command line
	// or use QMP to apply settings to a running VM
	
	return nil
}

// Helper function to apply memory settings
func (o *Ubuntu2404PerformanceOptimizer) applyMemorySettings(ctx context.Context, vmID string, vmInfo *KVMInfo, settings MemOptimizationSettings) error {
	// In a real implementation, this would modify the VM's QEMU command line
	// or use QMP to apply settings to a running VM
	
	return nil
}

// Helper function to get VM CPU usage
func (o *Ubuntu2404PerformanceOptimizer) getVMCPUUsage(ctx context.Context, vmInfo *KVMInfo) (float64, error) {
	// In a real implementation, this would use QMP or guest agent to get CPU usage
	// For simplicity, we'll use a placeholder implementation
	
	// Use ps to get CPU usage of the QEMU process
	if vmInfo.PID <= 0 {
		return 0, fmt.Errorf("VM PID not available")
	}
	
	cmd := exec.CommandContext(ctx, "ps", "-p", strconv.Itoa(vmInfo.PID), "-o", "%cpu", "--no-headers")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("failed to get CPU usage: %w", err)
	}
	
	// Parse CPU usage
	cpuStr := strings.TrimSpace(string(output))
	cpu, err := strconv.ParseFloat(cpuStr, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse CPU usage: %w", err)
	}
	
	return cpu, nil
}

// Helper function to get VM memory usage
func (o *Ubuntu2404PerformanceOptimizer) getVMMemoryUsage(ctx context.Context, vmInfo *KVMInfo) (float64, error) {
	// In a real implementation, this would use QMP or guest agent to get memory usage
	// For simplicity, we'll use a placeholder implementation
	
	// Use ps to get memory usage of the QEMU process
	if vmInfo.PID <= 0 {
		return 0, fmt.Errorf("VM PID not available")
	}
	
	cmd := exec.CommandContext(ctx, "ps", "-p", strconv.Itoa(vmInfo.PID), "-o", "%mem", "--no-headers")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("failed to get memory usage: %w", err)
	}
	
	// Parse memory usage
	memStr := strings.TrimSpace(string(output))
	mem, err := strconv.ParseFloat(memStr, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse memory usage: %w", err)
	}
	
	return mem, nil
}

// Helper function to get VM disk I/O
func (o *Ubuntu2404PerformanceOptimizer) getVMDiskIO(ctx context.Context, vmInfo *KVMInfo) (int64, int64, error) {
	// In a real implementation, this would use QMP or guest agent to get disk I/O
	// For simplicity, we'll return placeholder values
	return 0, 0, nil
}

// Helper function to get VM network I/O
func (o *Ubuntu2404PerformanceOptimizer) getVMNetworkIO(ctx context.Context, vmInfo *KVMInfo) (int64, int64, error) {
	// In a real implementation, this would use QMP or guest agent to get network I/O
	// For simplicity, we'll return placeholder values
	return 0, 0, nil
}

// Helper function to get VM I/O wait
func (o *Ubuntu2404PerformanceOptimizer) getVMIOWait(ctx context.Context, vmInfo *KVMInfo) (float64, error) {
	// In a real implementation, this would use QMP or guest agent to get I/O wait
	// For simplicity, we'll return a placeholder value
	return 0, nil
}

// Helper function to get VM load average
func (o *Ubuntu2404PerformanceOptimizer) getVMLoadAverage(ctx context.Context, vmInfo *KVMInfo) ([3]float64, error) {
	// In a real implementation, this would use QMP or guest agent to get load average
	// For simplicity, we'll return placeholder values
	return [3]float64{0, 0, 0}, nil
}
