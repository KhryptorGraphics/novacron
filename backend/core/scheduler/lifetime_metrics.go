package scheduler

import (
	"math"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// LifetimeMetrics holds lifetime-aware scheduling metrics
var (
	EmptyHostPercentage = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "scheduler_lifetime_empty_host_percentage",
		Help: "Percentage of empty hosts in the cluster",
	})

	CPUStrandingPercentage = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "scheduler_lifetime_cpu_stranding_percentage",
		Help: "Percentage of stranded CPU (unusable on non-empty hosts)",
	})

	MemoryStrandingPercentage = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "scheduler_lifetime_memory_stranding_percentage",
		Help: "Percentage of stranded memory (unusable on non-empty hosts)",
	})
)

// UpdateLifetimeMetrics updates metrics by scanning node inventories
func UpdateLifetimeMetrics(nodes map[string]*NodeResources) {
	numHosts := len(nodes)
	numEmpty := 0
	strandedCPU := 0.0
	totalUsableCPU := 0.0
	strandedMemory := 0.0
	totalUsableMemory := 0.0

	for _, node := range nodes {
		if node == nil {
			continue
		}

		isEmpty := true
		cpuUsed := 0.0
		memoryUsed := 0.0

		for _, res := range node.Resources {
			if res.Used > 0 {
				isEmpty = false
			}
			if res.Type == ResourceCPU {
				cpuUsed = res.Used
			}
			if res.Type == ResourceMemory {
				memoryUsed = res.Used
			}
		}

		if isEmpty {
			numEmpty++
		} else {
			// Stranding: unusable resources on non-empty host
			// Simple heuristic: if utilization < 10%, consider stranded
			if cpuUsed > 0 && cpuUsed < node.Resources[ResourceCPU].Capacity*0.1 {
				strandedCPU += node.Resources[ResourceCPU].Capacity - cpuUsed
			}
			totalUsableCPU += node.Resources[ResourceCPU].Capacity

			if memoryUsed > 0 && memoryUsed < node.Resources[ResourceMemory].Capacity*0.1 {
				strandedMemory += node.Resources[ResourceMemory].Capacity - memoryUsed
			}
			totalUsableMemory += node.Resources[ResourceMemory].Capacity
		}
	}

	// Update metrics
	if numHosts > 0 {
		EmptyHostPercentage.Set(float64(numEmpty) / float64(numHosts) * 100)
	}
	if totalUsableCPU > 0 {
		CPUStrandingPercentage.Set(math.Max(0, strandedCPU/totalUsableCPU*100))
	}
	if totalUsableMemory > 0 {
		MemoryStrandingPercentage.Set(math.Max(0, strandedMemory/totalUsableMemory*100))
	}
}
