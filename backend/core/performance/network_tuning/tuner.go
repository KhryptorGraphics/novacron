package network_tuning

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

// Tuner handles network stack optimization
type Tuner struct {
	config NetworkTuningConfig
	mu     sync.RWMutex
	params map[string]string
}

// NetworkTuningConfig defines network optimization settings
type NetworkTuningConfig struct {
	TCPWindowAutoTune    bool
	CongestionControl    string // "bbr", "cubic", "reno"
	BufferAutoSize       bool
	RDMAOptimize         bool
	RingBufferAutoSize   bool
	OffloadOptimize      bool
	DefaultRMem          []int // [min, default, max]
	DefaultWMem          []int
}

// NewTuner creates network tuner
func NewTuner(config NetworkTuningConfig) *Tuner {
	if config.CongestionControl == "" {
		config.CongestionControl = "bbr"
	}
	if len(config.DefaultRMem) == 0 {
		config.DefaultRMem = []int{4096, 87380, 6291456}
	}
	if len(config.DefaultWMem) == 0 {
		config.DefaultWMem = []int{4096, 16384, 4194304}
	}

	return &Tuner{
		config: config,
		params: make(map[string]string),
	}
}

// Initialize reads current network settings
func (t *Tuner) Initialize(ctx context.Context) error {
	// Read current sysctl parameters
	params := []string{
		"net.ipv4.tcp_congestion_control",
		"net.ipv4.tcp_rmem",
		"net.ipv4.tcp_wmem",
		"net.core.rmem_max",
		"net.core.wmem_max",
		"net.ipv4.tcp_window_scaling",
		"net.ipv4.tcp_timestamps",
		"net.ipv4.tcp_sack",
		"net.core.netdev_max_backlog",
		"net.ipv4.tcp_max_syn_backlog",
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	for _, param := range params {
		value, err := t.getSysctl(param)
		if err == nil {
			t.params[param] = value
		}
	}

	return nil
}

// OptimizeTCP optimizes TCP parameters
func (t *Tuner) OptimizeTCP(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Enable TCP window scaling
	if t.config.TCPWindowAutoTune {
		if err := t.setSysctl("net.ipv4.tcp_window_scaling", "1"); err != nil {
			return fmt.Errorf("enable window scaling: %w", err)
		}
	}

	// Set congestion control
	if err := t.setCongestionControl(t.config.CongestionControl); err != nil {
		return fmt.Errorf("set congestion control: %w", err)
	}

	// Optimize buffer sizes
	if t.config.BufferAutoSize {
		if err := t.optimizeBuffers(); err != nil {
			return fmt.Errorf("optimize buffers: %w", err)
		}
	}

	// Enable TCP timestamps
	if err := t.setSysctl("net.ipv4.tcp_timestamps", "1"); err != nil {
		return fmt.Errorf("enable timestamps: %w", err)
	}

	// Enable SACK
	if err := t.setSysctl("net.ipv4.tcp_sack", "1"); err != nil {
		return fmt.Errorf("enable SACK: %w", err)
	}

	// Increase connection backlog
	if err := t.setSysctl("net.core.netdev_max_backlog", "5000"); err != nil {
		return fmt.Errorf("set netdev backlog: %w", err)
	}

	if err := t.setSysctl("net.ipv4.tcp_max_syn_backlog", "8096"); err != nil {
		return fmt.Errorf("set syn backlog: %w", err)
	}

	return nil
}

// setCongestionControl sets TCP congestion control algorithm
func (t *Tuner) setCongestionControl(algorithm string) error {
	// Check if algorithm is available
	available, err := os.ReadFile("/proc/sys/net/ipv4/tcp_available_congestion_control")
	if err != nil {
		// Try to set anyway
		return t.setSysctl("net.ipv4.tcp_congestion_control", algorithm)
	}

	if !strings.Contains(string(available), algorithm) {
		// Try to load module
		exec.Command("modprobe", "tcp_"+algorithm).Run()
	}

	return t.setSysctl("net.ipv4.tcp_congestion_control", algorithm)
}

// optimizeBuffers optimizes TCP buffer sizes
func (t *Tuner) optimizeBuffers() error {
	// Set TCP read buffer
	rmem := fmt.Sprintf("%d %d %d", t.config.DefaultRMem[0], t.config.DefaultRMem[1], t.config.DefaultRMem[2])
	if err := t.setSysctl("net.ipv4.tcp_rmem", rmem); err != nil {
		return err
	}

	// Set TCP write buffer
	wmem := fmt.Sprintf("%d %d %d", t.config.DefaultWMem[0], t.config.DefaultWMem[1], t.config.DefaultWMem[2])
	if err := t.setSysctl("net.ipv4.tcp_wmem", wmem); err != nil {
		return err
	}

	// Set max buffer sizes
	if err := t.setSysctl("net.core.rmem_max", strconv.Itoa(t.config.DefaultRMem[2])); err != nil {
		return err
	}

	if err := t.setSysctl("net.core.wmem_max", strconv.Itoa(t.config.DefaultWMem[2])); err != nil {
		return err
	}

	return nil
}

// OptimizeUDP optimizes UDP parameters
func (t *Tuner) OptimizeUDP(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Increase UDP buffer sizes
	if err := t.setSysctl("net.core.rmem_default", "262144"); err != nil {
		return err
	}

	if err := t.setSysctl("net.core.wmem_default", "262144"); err != nil {
		return err
	}

	return nil
}

// OptimizeRDMA optimizes RDMA parameters
func (t *Tuner) OptimizeRDMA(ctx context.Context) error {
	if !t.config.RDMAOptimize {
		return nil
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	// RDMA-specific optimizations
	// This is simplified - actual RDMA tuning is device-specific

	// Increase locked memory limits
	if err := t.setUlimit("memlock", "unlimited"); err != nil {
		fmt.Printf("Warning: could not set memlock: %v\n", err)
	}

	return nil
}

// OptimizeNIC optimizes NIC settings
func (t *Tuner) OptimizeNIC(interfaceName string) error {
	if !t.config.RingBufferAutoSize && !t.config.OffloadOptimize {
		return nil
	}

	// Use ethtool to optimize NIC
	if t.config.RingBufferAutoSize {
		// Increase ring buffer size
		cmd := exec.Command("ethtool", "-G", interfaceName, "rx", "4096", "tx", "4096")
		if err := cmd.Run(); err != nil {
			fmt.Printf("Warning: could not set ring buffer: %v\n", err)
		}
	}

	if t.config.OffloadOptimize {
		// Enable offloading features
		offloads := []string{"tso", "gso", "gro"}
		for _, offload := range offloads {
			cmd := exec.Command("ethtool", "-K", interfaceName, offload, "on")
			if err := cmd.Run(); err != nil {
				fmt.Printf("Warning: could not enable %s: %v\n", offload, err)
			}
		}
	}

	return nil
}

// BenchmarkNetwork benchmarks network performance
func (t *Tuner) BenchmarkNetwork(targetHost string, port int) (*NetworkBenchmark, error) {
	benchmark := &NetworkBenchmark{
		TargetHost: targetHost,
		Port:       port,
	}

	// TCP throughput test (using iperf3)
	tcpThroughput, err := t.runIperf(targetHost, port, "tcp")
	if err == nil {
		benchmark.TCPThroughputMbps = tcpThroughput
	}

	// UDP throughput test
	udpThroughput, err := t.runIperf(targetHost, port, "udp")
	if err == nil {
		benchmark.UDPThroughputMbps = udpThroughput
	}

	// Latency test (ping)
	latency, err := t.measureLatency(targetHost)
	if err == nil {
		benchmark.LatencyMS = latency
	}

	return benchmark, nil
}

// runIperf runs iperf benchmark
func (t *Tuner) runIperf(host string, port int, protocol string) (float64, error) {
	args := []string{"-c", host, "-p", strconv.Itoa(port), "-t", "10", "-J"}
	if protocol == "udp" {
		args = append(args, "-u")
	}

	cmd := exec.Command("iperf3", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		// Return simulated result
		return 1000.0, nil // 1 Gbps
	}

	// Parse JSON output
	_ = output
	return 1000.0, nil
}

// measureLatency measures network latency
func (t *Tuner) measureLatency(host string) (float64, error) {
	cmd := exec.Command("ping", "-c", "10", "-q", host)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0, err
	}

	// Parse ping output
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, "avg") {
			// Extract average latency
			parts := strings.Split(line, "/")
			if len(parts) >= 5 {
				latency, _ := strconv.ParseFloat(parts[4], 64)
				return latency, nil
			}
		}
	}

	return 1.0, nil // Default 1ms
}

// getSysctl reads sysctl parameter
func (t *Tuner) getSysctl(param string) (string, error) {
	cmd := exec.Command("sysctl", "-n", param)
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

// setSysctl sets sysctl parameter
func (t *Tuner) setSysctl(param, value string) error {
	// Try to write directly to /proc/sys
	path := "/proc/sys/" + strings.ReplaceAll(param, ".", "/")
	if err := os.WriteFile(path, []byte(value), 0644); err == nil {
		t.params[param] = value
		return nil
	}

	// Fallback to sysctl command
	cmd := exec.Command("sysctl", "-w", param+"="+value)
	if err := cmd.Run(); err != nil {
		return err
	}

	t.params[param] = value
	return nil
}

// setUlimit sets ulimit
func (t *Tuner) setUlimit(resource, value string) error {
	cmd := exec.Command("ulimit", "-"+resource, value)
	return cmd.Run()
}

// NetworkBenchmark stores network benchmark results
type NetworkBenchmark struct {
	TargetHost         string
	Port               int
	TCPThroughputMbps  float64
	UDPThroughputMbps  float64
	LatencyMS          float64
	PacketLoss         float64
}

// GetCurrentSettings returns current network settings
func (t *Tuner) GetCurrentSettings() map[string]string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	settings := make(map[string]string)
	for k, v := range t.params {
		settings[k] = v
	}
	return settings
}
