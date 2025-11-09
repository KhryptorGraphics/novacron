package cpu_pinning

import (
	"context"
	"fmt"
	"sync"
)

// Engine handles automatic CPU affinity tuning
type Engine struct {
	config        CPUPinningConfig
	mu            sync.RWMutex
	allocations   map[string]*CPUAllocation
	physicalCPUs  []PhysicalCPU
	hyperThreading bool
}

// CPUPinningConfig defines CPU pinning settings
type CPUPinningConfig struct {
	Strategy         string  // "dedicated", "shared", "mixed"
	OvercommitRatio  float64 // 1.0 (no overcommit), 2.0 (2:1)
	HyperthreadingOpt bool
	CacheAffinity    bool
	IsolateNoisy     bool
	RebalanceInterval int // seconds
}

// CPUAllocation represents CPU allocation for a VM
type CPUAllocation struct {
	VMID            string
	VCPUs           int
	PhysicalCPUs    []int
	Strategy        string
	CacheAffinity   int // L1=1, L2=2, L3=3
	Isolated        bool
	OvercommitRatio float64
}

// PhysicalCPU represents a physical CPU core
type PhysicalCPU struct {
	ID             int
	CoreID         int
	SocketID       int
	SiblingID      int  // Hyperthread sibling
	L1Cache        int  // Cache size in KB
	L2Cache        int
	L3Cache        int
	Allocated      bool
	Workload       string // "high", "medium", "low", "idle"
}

// NewEngine creates CPU pinning engine
func NewEngine(config CPUPinningConfig) *Engine {
	return &Engine{
		config:      config,
		allocations: make(map[string]*CPUAllocation),
	}
}

// Initialize detects CPU topology
func (e *Engine) Initialize(ctx context.Context) error {
	cpus, hyperThreading, err := e.detectCPUTopology()
	if err != nil {
		return fmt.Errorf("detect CPU topology: %w", err)
	}

	e.mu.Lock()
	e.physicalCPUs = cpus
	e.hyperThreading = hyperThreading
	e.mu.Unlock()

	return nil
}

// detectCPUTopology detects physical CPU topology
func (e *Engine) detectCPUTopology() ([]PhysicalCPU, bool, error) {
	// Simplified topology - in production, read from /sys/devices/system/cpu

	numCPUs := 8
	numCores := 4
	hyperThreading := numCPUs > numCores

	cpus := make([]PhysicalCPU, numCPUs)
	for i := 0; i < numCPUs; i++ {
		coreID := i % numCores
		socketID := 0
		siblingID := -1

		if hyperThreading {
			if i < numCores {
				siblingID = i + numCores
			} else {
				siblingID = i - numCores
			}
		}

		cpus[i] = PhysicalCPU{
			ID:        i,
			CoreID:    coreID,
			SocketID:  socketID,
			SiblingID: siblingID,
			L1Cache:   32,   // 32 KB
			L2Cache:   256,  // 256 KB
			L3Cache:   8192, // 8 MB
			Allocated: false,
			Workload:  "idle",
		}
	}

	return cpus, hyperThreading, nil
}

// AllocateCPUs allocates CPUs for VM
func (e *Engine) AllocateCPUs(vmID string, vCPUs int, workloadType string) (*CPUAllocation, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	var strategy string
	var physicalCPUs []int

	switch e.config.Strategy {
	case "dedicated":
		// 1:1 vCPU to pCPU mapping
		cpus, err := e.allocateDedicated(vCPUs, workloadType)
		if err != nil {
			return nil, err
		}
		physicalCPUs = cpus
		strategy = "dedicated"

	case "shared":
		// Shared CPUs with overcommit
		cpus, err := e.allocateShared(vCPUs, workloadType)
		if err != nil {
			return nil, err
		}
		physicalCPUs = cpus
		strategy = "shared"

	case "mixed":
		// Mix of dedicated and shared based on workload
		if workloadType == "high" || e.config.IsolateNoisy {
			cpus, err := e.allocateDedicated(vCPUs, workloadType)
			if err != nil {
				// Fallback to shared
				cpus, err = e.allocateShared(vCPUs, workloadType)
				if err != nil {
					return nil, err
				}
				strategy = "shared"
			} else {
				strategy = "dedicated"
			}
			physicalCPUs = cpus
		} else {
			cpus, err := e.allocateShared(vCPUs, workloadType)
			if err != nil {
				return nil, err
			}
			physicalCPUs = cpus
			strategy = "shared"
		}

	default:
		return nil, fmt.Errorf("unknown strategy: %s", e.config.Strategy)
	}

	// Optimize for cache affinity
	cacheLevel := 3 // L3 by default
	if e.config.CacheAffinity {
		physicalCPUs = e.optimizeCacheAffinity(physicalCPUs)
		cacheLevel = e.determineCacheLevel(physicalCPUs)
	}

	allocation := &CPUAllocation{
		VMID:            vmID,
		VCPUs:           vCPUs,
		PhysicalCPUs:    physicalCPUs,
		Strategy:        strategy,
		CacheAffinity:   cacheLevel,
		Isolated:        strategy == "dedicated" && e.config.IsolateNoisy,
		OvercommitRatio: e.config.OvercommitRatio,
	}

	e.allocations[vmID] = allocation

	// Mark CPUs as allocated
	for _, cpuID := range physicalCPUs {
		e.physicalCPUs[cpuID].Allocated = true
		e.physicalCPUs[cpuID].Workload = workloadType
	}

	return allocation, nil
}

// allocateDedicated allocates dedicated CPUs (1:1)
func (e *Engine) allocateDedicated(vCPUs int, workloadType string) ([]int, error) {
	var allocated []int

	// Find unallocated CPUs
	for i := range e.physicalCPUs {
		if !e.physicalCPUs[i].Allocated {
			allocated = append(allocated, e.physicalCPUs[i].ID)
			if len(allocated) >= vCPUs {
				break
			}
		}
	}

	if len(allocated) < vCPUs {
		return nil, fmt.Errorf("insufficient free CPUs: need %d, available %d", vCPUs, len(allocated))
	}

	return allocated, nil
}

// allocateShared allocates shared CPUs with overcommit
func (e *Engine) allocateShared(vCPUs int, workloadType string) ([]int, error) {
	// Can overcommit - select least loaded CPUs
	type cpuLoad struct {
		id   int
		load float64
	}

	var loads []cpuLoad
	for i := range e.physicalCPUs {
		load := e.calculateCPULoad(i)
		loads = append(loads, cpuLoad{id: i, load: load})
	}

	// Sort by load (ascending)
	for i := 0; i < len(loads); i++ {
		for j := i + 1; j < len(loads); j++ {
			if loads[i].load > loads[j].load {
				loads[i], loads[j] = loads[j], loads[i]
			}
		}
	}

	// Allocate to least loaded CPUs
	allocated := make([]int, vCPUs)
	for i := 0; i < vCPUs; i++ {
		allocated[i] = loads[i%len(loads)].id
	}

	return allocated, nil
}

// calculateCPULoad calculates current CPU load
func (e *Engine) calculateCPULoad(cpuID int) float64 {
	// Count allocations on this CPU
	count := 0
	for _, alloc := range e.allocations {
		for _, id := range alloc.PhysicalCPUs {
			if id == cpuID {
				count++
			}
		}
	}

	return float64(count)
}

// optimizeCacheAffinity optimizes CPU selection for cache locality
func (e *Engine) optimizeCacheAffinity(cpus []int) []int {
	if len(cpus) <= 1 {
		return cpus
	}

	// Try to allocate CPUs from same socket/core for cache sharing
	optimized := make([]int, len(cpus))
	copy(optimized, cpus)

	// Sort by socket and core
	for i := 0; i < len(optimized); i++ {
		for j := i + 1; j < len(optimized); j++ {
			cpu1 := e.physicalCPUs[optimized[i]]
			cpu2 := e.physicalCPUs[optimized[j]]

			// Prefer same socket, then same core
			if cpu2.SocketID < cpu1.SocketID ||
				(cpu2.SocketID == cpu1.SocketID && cpu2.CoreID < cpu1.CoreID) {
				optimized[i], optimized[j] = optimized[j], optimized[i]
			}
		}
	}

	return optimized
}

// determineCacheLevel determines cache sharing level
func (e *Engine) determineCacheLevel(cpus []int) int {
	if len(cpus) <= 1 {
		return 1 // L1 cache
	}

	// Check if all CPUs share L1 (same core)
	firstCore := e.physicalCPUs[cpus[0]].CoreID
	sameCore := true
	for _, cpuID := range cpus[1:] {
		if e.physicalCPUs[cpuID].CoreID != firstCore {
			sameCore = false
			break
		}
	}
	if sameCore {
		return 1 // L1
	}

	// Check if all CPUs share L2 (usually same core, but depends on arch)
	// For simplicity, assume L2 = same core
	if sameCore {
		return 2
	}

	// Check if all CPUs share L3 (same socket)
	firstSocket := e.physicalCPUs[cpus[0]].SocketID
	sameSocket := true
	for _, cpuID := range cpus[1:] {
		if e.physicalCPUs[cpuID].SocketID != firstSocket {
			sameSocket = false
			break
		}
	}
	if sameSocket {
		return 3 // L3
	}

	return 0 // No shared cache
}

// OptimizeHyperthreading optimizes hyperthread usage
func (e *Engine) OptimizeHyperthreading(vmID string) error {
	if !e.config.HyperthreadingOpt || !e.hyperThreading {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	alloc, exists := e.allocations[vmID]
	if !exists {
		return fmt.Errorf("no allocation for VM %s", vmID)
	}

	// For high-performance workloads, avoid hyperthreads
	// For throughput workloads, use hyperthreads

	optimized := make([]int, 0, len(alloc.PhysicalCPUs))
	used := make(map[int]bool)

	for _, cpuID := range alloc.PhysicalCPUs {
		cpu := e.physicalCPUs[cpuID]

		// If high-performance and hyperthread, try to use only one thread per core
		if alloc.Strategy == "dedicated" && cpu.SiblingID >= 0 {
			if !used[cpu.CoreID] {
				optimized = append(optimized, cpuID)
				used[cpu.CoreID] = true
			}
		} else {
			optimized = append(optimized, cpuID)
		}
	}

	if len(optimized) < alloc.VCPUs {
		// Need more CPUs, add siblings
		for _, cpuID := range alloc.PhysicalCPUs {
			if len(optimized) >= alloc.VCPUs {
				break
			}
			if !contains(optimized, cpuID) {
				optimized = append(optimized, cpuID)
			}
		}
	}

	alloc.PhysicalCPUs = optimized
	return nil
}

// Rebalance rebalances CPU allocations
func (e *Engine) Rebalance(ctx context.Context) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Analyze current load distribution
	loads := make(map[int]float64)
	for i := range e.physicalCPUs {
		loads[i] = e.calculateCPULoad(i)
	}

	// Find imbalanced allocations
	avgLoad := 0.0
	for _, load := range loads {
		avgLoad += load
	}
	avgLoad /= float64(len(loads))

	// Rebalance if variance is high
	variance := 0.0
	for _, load := range loads {
		diff := load - avgLoad
		variance += diff * diff
	}
	variance /= float64(len(loads))

	if variance < 1.0 {
		// Well balanced
		return nil
	}

	// Reallocate VMs from overloaded to underloaded CPUs
	fmt.Printf("Rebalancing CPU allocations (variance=%.2f)\n", variance)

	// Simplified rebalancing - in production, use more sophisticated algorithm
	return nil
}

// Helper functions
func contains(slice []int, val int) bool {
	for _, v := range slice {
		if v == val {
			return true
		}
	}
	return false
}
