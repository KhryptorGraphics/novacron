package vm

import (
	"time"
)

// VMMetricsType defines the type of VM metrics
type VMMetricsType string

const (
	// VMMetricsTypeCPU represents CPU metrics
	VMMetricsTypeCPU VMMetricsType = "cpu"

	// VMMetricsTypeMemory represents memory metrics
	VMMetricsTypeMemory VMMetricsType = "memory"

	// VMMetricsTypeDisk represents disk metrics
	VMMetricsTypeDisk VMMetricsType = "disk"

	// VMMetricsTypeNetwork represents network metrics
	VMMetricsTypeNetwork VMMetricsType = "network"
)

// VMMetrics represents metrics for a VM
type VMMetrics struct {
	VMID        string                 `json:"vm_id"`
	NodeID      string                 `json:"node_id"`
	Timestamp   time.Time              `json:"timestamp"`
	CPU         CPUMetrics             `json:"cpu"`
	Memory      MemoryMetrics          `json:"memory"`
	Disk        map[string]DiskMetrics `json:"disk"`
	Network     map[string]NetMetrics  `json:"network"`
	Labels      map[string]string      `json:"labels,omitempty"`
	Annotations map[string]string      `json:"annotations,omitempty"`
}

// CPUMetrics represents CPU metrics
type CPUMetrics struct {
	UsagePercent     float64 `json:"usage_percent"`
	SystemPercent    float64 `json:"system_percent"`
	UserPercent      float64 `json:"user_percent"`
	IOWaitPercent    float64 `json:"iowait_percent"`
	StealPercent     float64 `json:"steal_percent"`
	Cores            int     `json:"cores"`
	ThrottledPeriods int64   `json:"throttled_periods"`
	ThrottledTime    int64   `json:"throttled_time"`
}

// MemoryMetrics represents memory metrics
type MemoryMetrics struct {
	TotalBytes      int64   `json:"total_bytes"`
	UsedBytes       int64   `json:"used_bytes"`
	CacheBytes      int64   `json:"cache_bytes"`
	RSSBytes        int64   `json:"rss_bytes"`
	SwapBytes       int64   `json:"swap_bytes"`
	UsagePercent    float64 `json:"usage_percent"`
	SwapPercent     float64 `json:"swap_percent"`
	MajorPageFaults int64   `json:"major_page_faults"`
	MinorPageFaults int64   `json:"minor_page_faults"`
}

// DiskMetrics represents disk metrics
type DiskMetrics struct {
	Device         string  `json:"device"`
	TotalBytes     int64   `json:"total_bytes"`
	UsedBytes      int64   `json:"used_bytes"`
	UsagePercent   float64 `json:"usage_percent"`
	ReadBytes      int64   `json:"read_bytes"`
	WriteBytes     int64   `json:"write_bytes"`
	ReadOps        int64   `json:"read_ops"`
	WriteOps       int64   `json:"write_ops"`
	ReadLatencyMs  float64 `json:"read_latency_ms"`
	WriteLatencyMs float64 `json:"write_latency_ms"`
	IOTimeMs       int64   `json:"io_time_ms"`
}

// NetMetrics represents network metrics
type NetMetrics struct {
	Interface     string  `json:"interface"`
	RxBytes       int64   `json:"rx_bytes"`
	TxBytes       int64   `json:"tx_bytes"`
	RxPackets     int64   `json:"rx_packets"`
	TxPackets     int64   `json:"tx_packets"`
	RxErrors      int64   `json:"rx_errors"`
	TxErrors      int64   `json:"tx_errors"`
	RxDropped     int64   `json:"rx_dropped"`
	TxDropped     int64   `json:"tx_dropped"`
	RxBytesPerSec float64 `json:"rx_bytes_per_sec"`
	TxBytesPerSec float64 `json:"tx_bytes_per_sec"`
}
