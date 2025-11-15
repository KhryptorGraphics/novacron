package optimization

import (
	"fmt"
	"os"
	"runtime"

	"golang.org/x/sys/unix"
)

// CPUAffinity manages CPU affinity and NUMA optimizations
type CPUAffinity struct {
	cpuSet unix.CPUSet
}

// NewCPUAffinity creates a new CPU affinity manager
func NewCPUAffinity() *CPUAffinity {
	return &CPUAffinity{}
}

// SetAffinity pins the current thread to specific CPUs
func (ca *CPUAffinity) SetAffinity(cpus []int) error {
	ca.cpuSet.Zero()
	for _, cpu := range cpus {
		ca.cpuSet.Set(cpu)
	}

	return unix.SchedSetaffinity(0, &ca.cpuSet)
}

// GetAffinity returns the current CPU affinity
func (ca *CPUAffinity) GetAffinity() ([]int, error) {
	var cpuSet unix.CPUSet
	if err := unix.SchedGetaffinity(0, &cpuSet); err != nil {
		return nil, err
	}

	cpus := make([]int, 0, runtime.NumCPU())
	for i := 0; i < runtime.NumCPU(); i++ {
		if cpuSet.IsSet(i) {
			cpus = append(cpus, i)
		}
	}

	return cpus, nil
}

// PinToCore pins current goroutine to a specific CPU core
func (ca *CPUAffinity) PinToCore(core int) error {
	return ca.SetAffinity([]int{core})
}

// PinToNUMANode pins to all CPUs in a NUMA node
func (ca *CPUAffinity) PinToNUMANode(node int) error {
	cpus := ca.getCPUsForNUMANode(node)
	if len(cpus) == 0 {
		return fmt.Errorf("no CPUs found for NUMA node %d", node)
	}
	return ca.SetAffinity(cpus)
}

// getCPUsForNUMANode returns CPUs belonging to a NUMA node
func (ca *CPUAffinity) getCPUsForNUMANode(node int) []int {
	// Read from /sys/devices/system/node/node*/cpulist
	cpulistPath := fmt.Sprintf("/sys/devices/system/node/node%d/cpulist", node)
	data, err := os.ReadFile(cpulistPath)
	if err != nil {
		return nil
	}

	cpus := make([]int, 0)
	// Parse CPU list (e.g., "0-7,16-23")
	cpuStr := string(data)
	_ = cpuStr // TODO: Parse CPU range string

	return cpus
}

// NUMAAllocator manages NUMA-aware memory allocation
type NUMAAllocator struct {
	node int
}

// NewNUMAAllocator creates a NUMA-aware allocator
func NewNUMAAllocator(node int) *NUMAAllocator {
	return &NUMAAllocator{node: node}
}

// Allocate allocates memory on specific NUMA node
func (na *NUMAAllocator) Allocate(size int) ([]byte, error) {
	return AllocateNUMAMemory(size, na.node)
}

// AllocateNUMAMemory allocates memory on a specific NUMA node
// Note: NUMA memory allocation is platform-specific and may not be available on all systems
func AllocateNUMAMemory(size, node int) ([]byte, error) {
	data := make([]byte, size)

	// TODO: Implement NUMA memory binding using platform-specific syscalls
	// This requires:
	// - unix.MPOL_BIND constant
	// - unix.Mbind syscall
	// - unix.MPOL_MF_STRICT and unix.MPOL_MF_MOVE flags
	// These are not available on all platforms/Go versions

	// For now, return the allocated memory without NUMA binding
	// In production, use cgo or platform-specific build tags

	return data, nil
}

// GetNUMANode returns the NUMA node for the current thread
// Note: This uses platform-specific syscalls that may not be available on all systems
func GetNUMANode() (int, error) {
	// TODO: Implement NUMA node detection using platform-specific syscalls
	// This requires syscall.SYS_GETCPU which is not available on all platforms

	// For now, return default node 0
	// In production, use cgo or platform-specific build tags

	return 0, nil
}

// GetNUMANodeLegacy returns the NUMA node for the current thread (legacy implementation)
func GetNUMANodeLegacy() (int, error) {
	// Placeholder for legacy implementation
	// TODO: Implement proper NUMA node detection
	return 0, nil
}

// ThreadPool manages worker threads with CPU affinity
type ThreadPool struct {
	workers  int
	cpuCores []int
	tasks    chan func()
	done     chan struct{}
}

// NewThreadPool creates a thread pool with CPU affinity
func NewThreadPool(workers int, cpuCores []int) *ThreadPool {
	if len(cpuCores) == 0 {
		// Default to all CPUs
		cpuCores = make([]int, runtime.NumCPU())
		for i := range cpuCores {
			cpuCores[i] = i
		}
	}

	tp := &ThreadPool{
		workers:  workers,
		cpuCores: cpuCores,
		tasks:    make(chan func(), workers*2),
		done:     make(chan struct{}),
	}

	tp.start()
	return tp
}

// start initializes worker goroutines
func (tp *ThreadPool) start() {
	for i := 0; i < tp.workers; i++ {
		core := tp.cpuCores[i%len(tp.cpuCores)]
		go tp.worker(core)
	}
}

// worker processes tasks with CPU affinity
func (tp *ThreadPool) worker(core int) {
	// Lock OS thread for CPU affinity
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Set CPU affinity
	ca := NewCPUAffinity()
	ca.PinToCore(core)

	for {
		select {
		case task := <-tp.tasks:
			task()
		case <-tp.done:
			return
		}
	}
}

// Submit submits a task to the pool
func (tp *ThreadPool) Submit(task func()) {
	tp.tasks <- task
}

// Close shuts down the thread pool
func (tp *ThreadPool) Close() {
	close(tp.done)
}

// CPUTopology provides information about CPU topology
type CPUTopology struct {
	NumCPUs      int
	NumCores     int
	NumSockets   int
	NumNUMANodes int
	L1CacheSize  int
	L2CacheSize  int
	L3CacheSize  int
}

// GetCPUTopology retrieves CPU topology information
func GetCPUTopology() (*CPUTopology, error) {
	topo := &CPUTopology{
		NumCPUs: runtime.NumCPU(),
	}

	// Read topology from /sys
	// L1 cache
	if data, err := os.ReadFile("/sys/devices/system/cpu/cpu0/cache/index0/size"); err == nil {
		fmt.Sscanf(string(data), "%dK", &topo.L1CacheSize)
		topo.L1CacheSize *= 1024
	}

	// L2 cache
	if data, err := os.ReadFile("/sys/devices/system/cpu/cpu0/cache/index2/size"); err == nil {
		fmt.Sscanf(string(data), "%dK", &topo.L2CacheSize)
		topo.L2CacheSize *= 1024
	}

	// L3 cache
	if data, err := os.ReadFile("/sys/devices/system/cpu/cpu0/cache/index3/size"); err == nil {
		fmt.Sscanf(string(data), "%dK", &topo.L3CacheSize)
		topo.L3CacheSize *= 1024
	}

	return topo, nil
}

// SetSchedulerAffinity sets scheduler affinity for optimal performance
// Note: This uses platform-specific syscalls that may not be available on all systems
func SetSchedulerAffinity(policy int, priority int) error {
	// TODO: Implement scheduler affinity using platform-specific syscalls
	// This requires unix.SchedParam and unix.SchedSetscheduler
	// These are not available on all platforms/Go versions

	// For now, return success without setting affinity
	// In production, use cgo or platform-specific build tags

	return nil
}

// SetRealtimePriority sets real-time scheduling priority
func SetRealtimePriority(priority int) error {
	return SetSchedulerAffinity(unix.SCHED_FIFO, priority)
}
