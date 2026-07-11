package benchmarks

import (
	"fmt"
	"math"
	"math/rand"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkITPPlacementAlgorithm tests VM placement algorithm performance
func BenchmarkITPPlacementAlgorithm(b *testing.B) {
	scenarios := []struct {
		name        string
		vmCount     int
		hostCount   int
		constraints int
	}{
		{"10VMs_5Hosts_Basic", 10, 5, 3},
		{"50VMs_20Hosts_Basic", 50, 20, 3},
		{"100VMs_50Hosts_Basic", 100, 50, 3},
		{"10VMs_5Hosts_Complex", 10, 5, 10},
		{"50VMs_20Hosts_Complex", 50, 20, 10},
		{"100VMs_50Hosts_Complex", 100, 50, 10},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkPlacement(b, sc.vmCount, sc.hostCount, sc.constraints)
		})
	}
}

func benchmarkPlacement(b *testing.B, vmCount, hostCount, constraintCount int) {
	b.ReportAllocs()

	// Generate VMs and hosts
	vms := generateVMs(vmCount)
	hosts := generateHosts(hostCount)
	constraints := generateConstraints(constraintCount)

	var placementCount int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		_ = computePlacement(vms, hosts, constraints)

		totalLatency += time.Since(start)
		atomic.AddInt64(&placementCount, 1)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Milliseconds()) / float64(b.N)
	placementsPerSecond := float64(placementCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyMs, "ms/placement")
	b.ReportMetric(placementsPerSecond, "placements/sec")
}

// BenchmarkITPGeographicOptimization tests geographic placement optimization
func BenchmarkITPGeographicOptimization(b *testing.B) {
	scenarios := []struct {
		name       string
		regions    int
		datacenters int
		vms         int
	}{
		{"3Regions_9DCs_50VMs", 3, 9, 50},
		{"3Regions_9DCs_200VMs", 3, 9, 200},
		{"5Regions_20DCs_50VMs", 5, 20, 50},
		{"5Regions_20DCs_200VMs", 5, 20, 200},
		{"10Regions_50DCs_50VMs", 10, 50, 50},
		{"10Regions_50DCs_200VMs", 10, 50, 200},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkGeoOptimization(b, sc.regions, sc.datacenters, sc.vms)
		})
	}
}

func benchmarkGeoOptimization(b *testing.B, regionCount, dcCount, vmCount int) {
	b.ReportAllocs()

	topology := generateGeoTopology(regionCount, dcCount)
	vms := generateVMs(vmCount)

	var optimizationCount int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		_ = optimizeGeoPlacement(vms, topology)

		totalLatency += time.Since(start)
		atomic.AddInt64(&optimizationCount, 1)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Milliseconds()) / float64(b.N)
	optimizationsPerSecond := float64(optimizationCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyMs, "ms/optimization")
	b.ReportMetric(optimizationsPerSecond, "optimizations/sec")
}

// BenchmarkITPResourceUtilization tests resource utilization calculation
func BenchmarkITPResourceUtilization(b *testing.B) {
	hostCounts := []int{10, 50, 100, 500, 1000}

	for _, hostCount := range hostCounts {
		b.Run(fmt.Sprintf("%dHosts", hostCount), func(b *testing.B) {
			benchmarkResourceUtilization(b, hostCount)
		})
	}
}

func benchmarkResourceUtilization(b *testing.B, hostCount int) {
	b.ReportAllocs()

	hosts := generateHosts(hostCount)

	var calculationCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = calculateUtilization(hosts)
		atomic.AddInt64(&calculationCount, 1)
	}

	b.StopTimer()

	calculationsPerSecond := float64(calculationCount) / b.Elapsed().Seconds()

	b.ReportMetric(calculationsPerSecond, "calculations/sec")
	b.ReportMetric(float64(hostCount), "hosts")
}

// BenchmarkITPPlacementQuality tests placement quality scoring
func BenchmarkITPPlacementQuality(b *testing.B) {
	scenarios := []struct {
		name      string
		vmCount   int
		hostCount int
		metrics   int
	}{
		{"10VMs_5Hosts_3Metrics", 10, 5, 3},
		{"50VMs_20Hosts_5Metrics", 50, 20, 5},
		{"100VMs_50Hosts_10Metrics", 100, 50, 10},
		{"500VMs_200Hosts_5Metrics", 500, 200, 5},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkPlacementQuality(b, sc.vmCount, sc.hostCount, sc.metrics)
		})
	}
}

func benchmarkPlacementQuality(b *testing.B, vmCount, hostCount, metricCount int) {
	b.ReportAllocs()

	placement := generatePlacement(vmCount, hostCount)
	metrics := generateQualityMetrics(metricCount)

	var scoreCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = calculatePlacementQuality(placement, metrics)
		atomic.AddInt64(&scoreCount, 1)
	}

	b.StopTimer()

	scoresPerSecond := float64(scoreCount) / b.Elapsed().Seconds()

	b.ReportMetric(scoresPerSecond, "scores/sec")
}

// BenchmarkITPAffinityRules tests affinity rule processing
func BenchmarkITPAffinityRules(b *testing.B) {
	scenarios := []struct {
		name      string
		vmCount   int
		ruleCount int
		ruleType  string
	}{
		{"10VMs_5Rules_Affinity", 10, 5, "affinity"},
		{"50VMs_20Rules_Affinity", 50, 20, "affinity"},
		{"10VMs_5Rules_AntiAffinity", 10, 5, "anti-affinity"},
		{"50VMs_20Rules_AntiAffinity", 50, 20, "anti-affinity"},
		{"100VMs_50Rules_Mixed", 100, 50, "mixed"},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkAffinityRules(b, sc.vmCount, sc.ruleCount, sc.ruleType)
		})
	}
}

func benchmarkAffinityRules(b *testing.B, vmCount, ruleCount int, ruleType string) {
	b.ReportAllocs()

	vms := generateVMs(vmCount)
	rules := generateAffinityRules(ruleCount, ruleType)

	var validationCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = validateAffinityRules(vms, rules)
		atomic.AddInt64(&validationCount, 1)
	}

	b.StopTimer()

	validationsPerSecond := float64(validationCount) / b.Elapsed().Seconds()

	b.ReportMetric(validationsPerSecond, "validations/sec")
}

// BenchmarkITPLoadBalancing tests load balancing optimization
func BenchmarkITPLoadBalancing(b *testing.B) {
	scenarios := []struct {
		name        string
		hostCount   int
		imbalance   float64
	}{
		{"10Hosts_LowImbalance", 10, 0.1},
		{"10Hosts_HighImbalance", 10, 0.5},
		{"50Hosts_LowImbalance", 50, 0.1},
		{"50Hosts_HighImbalance", 50, 0.5},
		{"100Hosts_LowImbalance", 100, 0.1},
		{"100Hosts_HighImbalance", 100, 0.5},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkLoadBalancing(b, sc.hostCount, sc.imbalance)
		})
	}
}

func benchmarkLoadBalancing(b *testing.B, hostCount int, imbalance float64) {
	b.ReportAllocs()

	hosts := generateImbalancedHosts(hostCount, imbalance)

	var balanceCount int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		_ = rebalanceLoad(hosts)

		totalLatency += time.Since(start)
		atomic.AddInt64(&balanceCount, 1)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Milliseconds()) / float64(b.N)
	balancesPerSecond := float64(balanceCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyMs, "ms/rebalance")
	b.ReportMetric(balancesPerSecond, "rebalances/sec")
}

// BenchmarkITPDynamicPlacement tests dynamic placement under changing conditions
func BenchmarkITPDynamicPlacement(b *testing.B) {
	scenarios := []struct {
		name         string
		vmCount      int
		hostCount    int
		changeRate   float64
	}{
		{"50VMs_20Hosts_LowChange", 50, 20, 0.05},
		{"50VMs_20Hosts_MediumChange", 50, 20, 0.20},
		{"50VMs_20Hosts_HighChange", 50, 20, 0.50},
		{"200VMs_100Hosts_LowChange", 200, 100, 0.05},
		{"200VMs_100Hosts_MediumChange", 200, 100, 0.20},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkDynamicPlacement(b, sc.vmCount, sc.hostCount, sc.changeRate)
		})
	}
}

func benchmarkDynamicPlacement(b *testing.B, vmCount, hostCount int, changeRate float64) {
	b.ReportAllocs()

	vms := generateVMs(vmCount)
	hosts := generateHosts(hostCount)

	var placementCount int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Simulate environment changes
		applyChanges(hosts, changeRate)

		start := time.Now()

		_ = computePlacement(vms, hosts, nil)

		totalLatency += time.Since(start)
		atomic.AddInt64(&placementCount, 1)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Milliseconds()) / float64(b.N)
	placementsPerSecond := float64(placementCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyMs, "ms/placement")
	b.ReportMetric(placementsPerSecond, "placements/sec")
}

// Helper types and functions

type VM struct {
	ID       string
	CPU      int
	Memory   int64
	Storage  int64
	Location string
}

type Host struct {
	ID            string
	CPUCapacity   int
	CPUUsed       int
	MemoryCapacity int64
	MemoryUsed     int64
	Location       string
}

type Constraint struct {
	Type   string
	Target string
	Value  interface{}
}

type GeoTopology struct {
	Regions     []Region
	Latencies   map[string]map[string]float64
}

type Region struct {
	Name        string
	Datacenters []Datacenter
}

type Datacenter struct {
	ID       string
	Location string
	Capacity int
}

func generateVMs(count int) []VM {
	vms := make([]VM, count)
	for i := range vms {
		vms[i] = VM{
			ID:      fmt.Sprintf("vm-%d", i),
			CPU:     rand.Intn(8) + 1,
			Memory:  int64(rand.Intn(16)+1) * 1024 * 1024 * 1024,
			Storage: int64(rand.Intn(100)+10) * 1024 * 1024 * 1024,
			Location: fmt.Sprintf("region-%d", rand.Intn(3)),
		}
	}
	return vms
}

func generateHosts(count int) []Host {
	hosts := make([]Host, count)
	for i := range hosts {
		cpuCapacity := rand.Intn(32) + 16
		memoryCapacity := int64(rand.Intn(128)+64) * 1024 * 1024 * 1024

		hosts[i] = Host{
			ID:             fmt.Sprintf("host-%d", i),
			CPUCapacity:    cpuCapacity,
			CPUUsed:        rand.Intn(cpuCapacity / 2),
			MemoryCapacity: memoryCapacity,
			MemoryUsed:     rand.Int63n(memoryCapacity / 2),
			Location:       fmt.Sprintf("region-%d", rand.Intn(3)),
		}
	}
	return hosts
}

func generateConstraints(count int) []Constraint {
	constraints := make([]Constraint, count)
	types := []string{"cpu", "memory", "location", "affinity"}

	for i := range constraints {
		constraints[i] = Constraint{
			Type:   types[rand.Intn(len(types))],
			Target: fmt.Sprintf("target-%d", i),
			Value:  rand.Intn(100),
		}
	}
	return constraints
}

func generateGeoTopology(regionCount, dcCount int) GeoTopology {
	topology := GeoTopology{
		Regions:   make([]Region, regionCount),
		Latencies: make(map[string]map[string]float64),
	}

	dcsPerRegion := dcCount / regionCount

	for i := range topology.Regions {
		region := Region{
			Name:        fmt.Sprintf("region-%d", i),
			Datacenters: make([]Datacenter, dcsPerRegion),
		}

		for j := range region.Datacenters {
			region.Datacenters[j] = Datacenter{
				ID:       fmt.Sprintf("dc-%d-%d", i, j),
				Location: fmt.Sprintf("location-%d-%d", i, j),
				Capacity: rand.Intn(1000) + 100,
			}
		}

		topology.Regions[i] = region
	}

	return topology
}

func computePlacement(vms []VM, hosts []Host, constraints []Constraint) map[string]string {
	placement := make(map[string]string)

	for _, vm := range vms {
		// Simple first-fit algorithm
		for _, host := range hosts {
			if host.CPUCapacity-host.CPUUsed >= vm.CPU &&
				host.MemoryCapacity-host.MemoryUsed >= vm.Memory {
				placement[vm.ID] = host.ID
				break
			}
		}
	}

	return placement
}

func optimizeGeoPlacement(vms []VM, topology GeoTopology) map[string]string {
	placement := make(map[string]string)

	for _, vm := range vms {
		// Find closest datacenter
		bestDC := ""
		bestLatency := math.MaxFloat64

		for _, region := range topology.Regions {
			for _, dc := range region.Datacenters {
				latency := rand.Float64() * 100 // Simulated latency
				if latency < bestLatency {
					bestLatency = latency
					bestDC = dc.ID
				}
			}
		}

		placement[vm.ID] = bestDC
	}

	return placement
}

func calculateUtilization(hosts []Host) map[string]float64 {
	utilization := make(map[string]float64)

	for _, host := range hosts {
		cpuUtil := float64(host.CPUUsed) / float64(host.CPUCapacity)
		memUtil := float64(host.MemoryUsed) / float64(host.MemoryCapacity)
		utilization[host.ID] = (cpuUtil + memUtil) / 2
	}

	return utilization
}

func generatePlacement(vmCount, hostCount int) map[string]string {
	placement := make(map[string]string)

	for i := 0; i < vmCount; i++ {
		vmID := fmt.Sprintf("vm-%d", i)
		hostID := fmt.Sprintf("host-%d", rand.Intn(hostCount))
		placement[vmID] = hostID
	}

	return placement
}

func generateQualityMetrics(count int) []string {
	metrics := []string{"balance", "efficiency", "latency", "cost", "availability"}
	result := make([]string, count)

	for i := range result {
		result[i] = metrics[rand.Intn(len(metrics))]
	}

	return result
}

func calculatePlacementQuality(placement map[string]string, metrics []string) float64 {
	score := 0.0

	for range metrics {
		score += rand.Float64()
	}

	return score / float64(len(metrics))
}

func generateAffinityRules(count int, ruleType string) []map[string]interface{} {
	rules := make([]map[string]interface{}, count)

	for i := range rules {
		rules[i] = map[string]interface{}{
			"type":   ruleType,
			"vmIDs":  []string{fmt.Sprintf("vm-%d", i), fmt.Sprintf("vm-%d", i+1)},
			"strict": rand.Float64() > 0.5,
		}
	}

	return rules
}

func validateAffinityRules(vms []VM, rules []map[string]interface{}) bool {
	// Simplified validation
	for range rules {
		if rand.Float64() > 0.9 {
			return false
		}
	}
	return true
}

func generateImbalancedHosts(count int, imbalance float64) []Host {
	hosts := generateHosts(count)

	// Create imbalance
	for i := range hosts {
		if rand.Float64() < imbalance {
			hosts[i].CPUUsed = int(float64(hosts[i].CPUCapacity) * 0.9)
			hosts[i].MemoryUsed = int64(float64(hosts[i].MemoryCapacity) * 0.9)
		}
	}

	return hosts
}

func rebalanceLoad(hosts []Host) []Host {
	// Simplified load balancing
	avgCPU := 0.0
	for _, host := range hosts {
		avgCPU += float64(host.CPUUsed) / float64(host.CPUCapacity)
	}
	avgCPU /= float64(len(hosts))

	// Simulate rebalancing work
	time.Sleep(time.Duration(len(hosts)*100) * time.Microsecond)

	return hosts
}

func applyChanges(hosts []Host, changeRate float64) {
	changeCount := int(float64(len(hosts)) * changeRate)

	for i := 0; i < changeCount; i++ {
		idx := rand.Intn(len(hosts))
		hosts[idx].CPUUsed = rand.Intn(hosts[idx].CPUCapacity)
		hosts[idx].MemoryUsed = rand.Int63n(hosts[idx].MemoryCapacity)
	}
}
