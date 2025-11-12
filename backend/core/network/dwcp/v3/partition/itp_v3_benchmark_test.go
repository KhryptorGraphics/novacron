package partition

import (
	"context"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// BenchmarkVMPlacement benchmarks single VM placement
func BenchmarkVMPlacement(b *testing.B) {
	modes := []upgrade.NetworkMode{
		upgrade.ModeDatacenter,
		upgrade.ModeInternet,
		upgrade.ModeHybrid,
	}

	nodeCounts := []int{10, 100, 1000}

	for _, mode := range modes {
		for _, nodeCount := range nodeCounts {
			b.Run(fmt.Sprintf("%s_%d_nodes", mode.String(), nodeCount), func(b *testing.B) {
				// Setup
				itp, _ := NewITPv3(mode)
				nodes := generateBenchmarkNodes(nodeCount)
				for _, node := range nodes {
					itp.AddNode(node)
				}

				ctx := context.Background()
				vm := generateRandomVM()

				// Reset timer to exclude setup
				b.ResetTimer()

				// Benchmark
				for i := 0; i < b.N; i++ {
					_, err := itp.PlaceVM(ctx, vm, nil)
					if err != nil {
						b.Logf("Placement failed: %v", err)
					}

					// Clean up for next iteration
					if vm.PlacedNode != nil {
						itp.deallocateResources(vm.PlacedNode, vm)
						vm.PlacedNode = nil
					}
				}

				// Report metrics
				metrics := itp.GetMetrics()
				b.ReportMetric(float64(metrics["placement_latency"].(int64)), "ms/op")
				b.ReportMetric(metrics["resource_utilization"].(float64)*100, "%util")
			})
		}
	}
}

// BenchmarkBatchPlacement benchmarks batch VM placement
func BenchmarkBatchPlacement(b *testing.B) {
	batchSizes := []int{10, 50, 100}
	nodeCount := 100

	for _, batchSize := range batchSizes {
		b.Run(fmt.Sprintf("batch_%d", batchSize), func(b *testing.B) {
			// Setup
			itp, _ := NewITPv3(upgrade.ModeHybrid)
			nodes := generateBenchmarkNodes(nodeCount)
			for _, node := range nodes {
				itp.AddNode(node)
			}

			ctx := context.Background()

			// Reset timer
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Generate batch of VMs
				vms := generateVMBatch(batchSize)

				// Place batch
				placements, err := itp.PlaceVMBatch(ctx, vms, nil)
				if err != nil {
					b.Logf("Batch placement failed: %v", err)
				}

				// Clean up
				for vmID, node := range placements {
					for _, vm := range vms {
						if vm.ID == vmID {
							itp.deallocateResources(node, vm)
							break
						}
					}
				}
			}

			// Report batch metrics
			metrics := itp.GetMetrics()
			successRate := metrics["success_rate"].(float64)
			b.ReportMetric(successRate*100, "%success")
		})
	}
}

// BenchmarkGeographicOptimization benchmarks geographic placement optimization
func BenchmarkGeographicOptimization(b *testing.B) {
	regionCounts := []int{5, 10, 20}

	for _, regionCount := range regionCounts {
		b.Run(fmt.Sprintf("%d_regions", regionCount), func(b *testing.B) {
			// Setup
			itp, _ := NewITPv3(upgrade.ModeInternet)

			// Create regions
			regions := generateBenchmarkRegions(regionCount)
			for _, region := range regions {
				itp.AddRegion(region)
			}

			// Create nodes across regions
			nodesPerRegion := 10
			for _, region := range regions {
				nodes := generateRegionNodes(region, nodesPerRegion)
				for _, node := range nodes {
					itp.AddNode(node)
				}
			}

			ctx := context.Background()

			// Reset timer
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				// Create VM with random region requirements
				vm := &VM{
					ID:              fmt.Sprintf("vm-%d", i),
					RequestedCPU:    4,
					RequestedMemory: 8 * 1e9,
					RequiredRegions: []string{regions[rand.Intn(regionCount)].ID},
				}

				// Place VM
				node, err := itp.PlaceVM(ctx, vm, nil)
				if err == nil && node != nil {
					// Clean up
					itp.deallocateResources(node, vm)
				}
			}
		})
	}
}

// BenchmarkConstraintEvaluation benchmarks constraint evaluation performance
func BenchmarkConstraintEvaluation(b *testing.B) {
	constraintTypes := []string{"simple", "complex", "extreme"}

	for _, constraintType := range constraintTypes {
		b.Run(constraintType, func(b *testing.B) {
			// Setup
			itp, _ := NewITPv3(upgrade.ModeHybrid)
			nodes := generateBenchmarkNodes(100)
			for _, node := range nodes {
				itp.AddNode(node)
			}

			constraints := generateConstraints(constraintType)
			ctx := context.Background()

			// Reset timer
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				vm := generateRandomVM()

				// Place with constraints
				node, err := itp.PlaceVM(ctx, vm, constraints)
				if err == nil && node != nil {
					// Clean up
					itp.deallocateResources(node, vm)
				}
			}
		})
	}
}

// BenchmarkHeterogeneousPlacement benchmarks heterogeneous node placement
func BenchmarkHeterogeneousPlacement(b *testing.B) {
	nodeTypeMixes := []string{"homogeneous", "mixed", "heterogeneous"}

	for _, mix := range nodeTypeMixes {
		b.Run(mix, func(b *testing.B) {
			// Setup
			itp, _ := NewITPv3(upgrade.ModeHybrid)
			nodes := generateNodesByMix(mix, 100)
			for _, node := range nodes {
				itp.AddNode(node)
			}

			ctx := context.Background()

			// Reset timer
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				vm := generateVMWithNodeTypePreference()

				// Place VM
				node, err := itp.PlaceVM(ctx, vm, nil)
				if err == nil && node != nil {
					// Clean up
					itp.deallocateResources(node, vm)
				}
			}
		})
	}
}

// BenchmarkResourceUtilization benchmarks resource utilization optimization
func BenchmarkResourceUtilization(b *testing.B) {
	utilizationTargets := []float64{0.5, 0.7, 0.9}

	for _, target := range utilizationTargets {
		b.Run(fmt.Sprintf("target_%.0f%%", target*100), func(b *testing.B) {
			// Setup
			itp, _ := NewITPv3(upgrade.ModeDatacenter)
			nodes := generateBenchmarkNodes(50)
			for _, node := range nodes {
				itp.AddNode(node)
			}

			ctx := context.Background()

			// Pre-fill to target utilization
			fillToUtilization(itp, target)

			// Reset timer
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				vm := generateSmallVM() // Small VMs for bin packing

				// Try to place
				node, err := itp.PlaceVM(ctx, vm, nil)
				if err == nil && node != nil {
					// Clean up immediately to maintain target
					itp.deallocateResources(node, vm)
				}
			}

			// Report final utilization
			metrics := itp.GetMetrics()
			b.ReportMetric(metrics["resource_utilization"].(float64)*100, "%util")
		})
	}
}

// Helper functions for benchmarks

func generateBenchmarkNodes(count int) []*Node {
	nodes := make([]*Node, count)
	rand.Seed(time.Now().UnixNano())

	for i := 0; i < count; i++ {
		nodes[i] = &Node{
			ID:              fmt.Sprintf("node-%d", i),
			Name:            fmt.Sprintf("bench-node-%d", i),
			Type:            NodeType(rand.Intn(4)),
			Region:          fmt.Sprintf("region-%d", i%10),
			TotalCPU:        8 + rand.Intn(56),  // 8-64 CPUs
			TotalMemory:     int64((16 + rand.Intn(240)) * 1e9), // 16-256 GB
			TotalDisk:       int64((100 + rand.Intn(900)) * 1e9), // 100-1000 GB
			TotalGPU:        rand.Intn(8), // 0-8 GPUs
			NetworkBandwidth: float64(1 + rand.Intn(100)), // 1-100 Gbps
			CostPerHour:     rand.Float64() * 10, // $0-10/hour
			Uptime:          time.Hour * time.Duration(24*rand.Intn(365)),
			FailureRate:     rand.Float64() * 0.1, // 0-10% failure rate
			Labels:          make(map[string]string),
		}

		// Set available resources to total (empty node)
		nodes[i].AvailableCPU = nodes[i].TotalCPU
		nodes[i].AvailableMemory = nodes[i].TotalMemory
		nodes[i].AvailableDisk = nodes[i].TotalDisk
		nodes[i].AvailableGPU = nodes[i].TotalGPU
	}

	return nodes
}

func generateRandomVM() *VM {
	return &VM{
		ID:              fmt.Sprintf("vm-%d", rand.Int()),
		Name:            "benchmark-vm",
		RequestedCPU:    1 + rand.Intn(8),
		RequestedMemory: int64((1 + rand.Intn(32)) * 1e9),
		RequestedDisk:   int64((10 + rand.Intn(100)) * 1e9),
		RequestedGPU:    rand.Intn(2),
		Priority:        rand.Float64(),
	}
}

func generateVMBatch(size int) []*VM {
	vms := make([]*VM, size)
	for i := 0; i < size; i++ {
		vms[i] = &VM{
			ID:              fmt.Sprintf("batch-vm-%d", i),
			Name:            fmt.Sprintf("batch-vm-%d", i),
			RequestedCPU:    1 + rand.Intn(4),
			RequestedMemory: int64((1 + rand.Intn(16)) * 1e9),
			RequestedDisk:   int64((10 + rand.Intn(50)) * 1e9),
			Priority:        float64(size-i) / float64(size), // Decreasing priority
		}
	}
	return vms
}

func generateBenchmarkRegions(count int) []*Region {
	regions := make([]*Region, count)

	for i := 0; i < count; i++ {
		regions[i] = &Region{
			ID:        fmt.Sprintf("region-%d", i),
			Name:      fmt.Sprintf("Region %d", i),
			Continent: []string{"NA", "EU", "AS", "SA", "AF", "OC"}[i%6],
			Latitude:  -90 + rand.Float64()*180,
			Longitude: -180 + rand.Float64()*360,
			InternetLatency: make(map[string]time.Duration),
		}

		// Generate inter-region latencies
		for j := 0; j < count; j++ {
			if i != j {
				// Distance-based latency simulation
				latency := time.Duration(10+rand.Intn(200)) * time.Millisecond
				regions[i].InternetLatency[fmt.Sprintf("region-%d", j)] = latency
			}
		}
	}

	return regions
}

func generateRegionNodes(region *Region, count int) []*Node {
	nodes := make([]*Node, count)

	for i := 0; i < count; i++ {
		nodes[i] = &Node{
			ID:              fmt.Sprintf("%s-node-%d", region.ID, i),
			Name:            fmt.Sprintf("%s-node-%d", region.Name, i),
			Type:            NodeType(rand.Intn(4)),
			Region:          region.ID,
			TotalCPU:        8 + rand.Intn(24),
			TotalMemory:     int64((16 + rand.Intn(48)) * 1e9),
			TotalDisk:       int64((100 + rand.Intn(400)) * 1e9),
			NetworkBandwidth: float64(1 + rand.Intn(40)),
			CostPerHour:     rand.Float64() * 5,
			Uptime:          time.Hour * time.Duration(24*rand.Intn(180)),
			Labels:          make(map[string]string),
		}

		nodes[i].AvailableCPU = nodes[i].TotalCPU
		nodes[i].AvailableMemory = nodes[i].TotalMemory
		nodes[i].AvailableDisk = nodes[i].TotalDisk
	}

	return nodes
}

func generateConstraints(constraintType string) *Constraints {
	switch constraintType {
	case "simple":
		return &Constraints{
			MaxCostPerHour: 5.0,
		}

	case "complex":
		return &Constraints{
			MaxLatency:       100 * time.Millisecond,
			MinBandwidth:     10.0,
			RequiredUptime:   0.99,
			MaxCostPerHour:   3.0,
			RequiredNodeType: NodeTypeCloud,
		}

	case "extreme":
		return &Constraints{
			MaxLatency:       10 * time.Millisecond,
			MinBandwidth:     50.0,
			RequiredUptime:   0.9999,
			MaxCostPerHour:   1.0,
			DataLocality:     true,
			RequiredNodeType: NodeTypeDatacenter,
		}

	default:
		return nil
	}
}

func generateNodesByMix(mix string, count int) []*Node {
	nodes := generateBenchmarkNodes(count)

	switch mix {
	case "homogeneous":
		// All nodes same type
		for i := range nodes {
			nodes[i].Type = NodeTypeCloud
		}

	case "mixed":
		// Even distribution
		for i := range nodes {
			nodes[i].Type = NodeType(i % 4)
		}

	case "heterogeneous":
		// Weighted distribution
		weights := []int{40, 30, 20, 10} // Cloud, DC, Edge, Volunteer percentages
		typeIdx := 0
		typeCount := 0
		targetCount := count * weights[0] / 100

		for i := range nodes {
			nodes[i].Type = NodeType(typeIdx)
			typeCount++

			if typeCount >= targetCount && typeIdx < 3 {
				typeIdx++
				if typeIdx < 4 {
					targetCount = count * weights[typeIdx] / 100
					typeCount = 0
				}
			}
		}
	}

	return nodes
}

func generateVMWithNodeTypePreference() *VM {
	vm := generateRandomVM()

	// Add node type preference randomly
	nodeTypes := []string{"cloud", "datacenter", "edge", "volunteer"}
	if rand.Float64() < 0.5 {
		vm.RequiredLabels = map[string]string{
			"node-type": nodeTypes[rand.Intn(4)],
		}
	}

	return vm
}

func generateSmallVM() *VM {
	return &VM{
		ID:              fmt.Sprintf("small-vm-%d", rand.Int()),
		Name:            "small-vm",
		RequestedCPU:    1 + rand.Intn(2),
		RequestedMemory: int64((1 + rand.Intn(4)) * 1e9),
		RequestedDisk:   int64((10 + rand.Intn(20)) * 1e9),
		Priority:        rand.Float64(),
	}
}

func fillToUtilization(itp *ITPv3, target float64) {
	ctx := context.Background()

	// Keep placing VMs until target utilization is reached
	for {
		metrics := itp.GetMetrics()
		current := metrics["resource_utilization"].(float64)

		if current >= target {
			break
		}

		vm := generateSmallVM()
		_, err := itp.PlaceVM(ctx, vm, nil)
		if err != nil {
			// Can't place more VMs
			break
		}
	}
}