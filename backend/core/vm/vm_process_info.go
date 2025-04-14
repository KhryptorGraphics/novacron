package vm

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// VMProcessInfo contains information about a VM process
type VMProcessInfo struct {
	PID             int       `json:"pid"`
	StartTime       time.Time `json:"start_time"`
	CPUUsagePercent float64   `json:"cpu_usage_percent"`
	MemoryUsageMB   int       `json:"memory_usage_mb"`
	ThreadCount     int       `json:"thread_count"`
	LastUpdatedAt   time.Time `json:"last_updated_at"`
}

// ResourceUsage represents resource usage information
type ResourceUsage struct {
	CPU       float64
	MemoryMB  int
	Threads   int
	IOReadMB  float64
	IOWriteMB float64
}

// getProcessResourceUsage gets resource usage for a process
func getProcessResourceUsage(pid int) (*ResourceUsage, error) {
	// Get CPU and memory usage using ps
	cmd := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "%cpu,%mem,rss,nlwp", "--no-headers")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to get process stats: %w", err)
	}
	
	// Parse the output
	fields := strings.Fields(string(output))
	if len(fields) < 4 {
		return nil, fmt.Errorf("unexpected ps output format")
	}
	
	// Parse CPU usage
	cpu, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CPU usage: %w", err)
	}
	
	// Parse memory percentage
	memPercent, err := strconv.ParseFloat(fields[1], 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse memory percentage: %w", err)
	}
	
	// Parse RSS (resident set size) in KB
	rss, err := strconv.ParseInt(fields[2], 10, 64)
	if err != nil {
		return nil, fmt.Errorf("failed to parse RSS: %w", err)
	}
	
	// Convert RSS from KB to MB
	memoryMB := int(rss / 1024)
	
	// Parse thread count
	threads, err := strconv.Atoi(fields[3])
	if err != nil {
		return nil, fmt.Errorf("failed to parse thread count: %w", err)
	}
	
	// Get I/O statistics using /proc/{pid}/io
	var ioRead, ioWrite float64
	ioCmd := exec.Command("cat", fmt.Sprintf("/proc/%d/io", pid))
	ioOutput, err := ioCmd.Output()
	if err == nil {
		ioLines := strings.Split(string(ioOutput), "\n")
		for _, line := range ioLines {
			if strings.HasPrefix(line, "read_bytes:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					bytes, _ := strconv.ParseInt(fields[1], 10, 64)
					ioRead = float64(bytes) / (1024 * 1024) // Convert to MB
				}
			} else if strings.HasPrefix(line, "write_bytes:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					bytes, _ := strconv.ParseInt(fields[1], 10, 64)
					ioWrite = float64(bytes) / (1024 * 1024) // Convert to MB
				}
			}
		}
	}
	
	return &ResourceUsage{
		CPU:       cpu,
		MemoryMB:  memoryMB,
		Threads:   threads,
		IOReadMB:  ioRead,
		IOWriteMB: ioWrite,
	}, nil
}
