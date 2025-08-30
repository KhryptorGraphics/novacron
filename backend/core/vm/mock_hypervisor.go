package vm

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"
)

// MockHypervisor provides a mock implementation of VMDriver for testing
type MockHypervisor struct {
	vms           map[string]*MockVM
	vmLock        sync.RWMutex
	nodeID        string
	hypervisorID  string
	capabilities  MockCapabilities
	failures      MockFailureConfig
	latencyConfig MockLatencyConfig
}

// MockVM represents a mock virtual machine
type MockVM struct {
	ID              string
	Config          VMConfig
	State           State
	CreatedAt       time.Time
	StartedAt       *time.Time
	StoppedAt       *time.Time
	PID             int
	CPUUsage        float64
	MemoryUsage     int64
	NetworkRx       int64
	NetworkTx       int64
	DiskRead        int64
	DiskWrite       int64
	snapshots       map[string]*MockSnapshot
	migrationTarget string
}

// MockSnapshot represents a mock VM snapshot
type MockSnapshot struct {
	ID        string
	Name      string
	CreatedAt time.Time
	Size      int64
}

// MockCapabilities defines what the mock hypervisor supports
type MockCapabilities struct {
	SupportsPause     bool
	SupportsResume    bool
	SupportsSnapshot  bool
	SupportsMigrate   bool
	MaxVMs            int
	MaxCPUPerVM       int
	MaxMemoryPerVM    int64
	SupportedVMTypes  []VMType
}

// MockFailureConfig allows injecting failures for testing
type MockFailureConfig struct {
	CreateFailureRate   float64 // 0.0 to 1.0
	StartFailureRate    float64
	StopFailureRate     float64
	PauseFailureRate    float64
	ResumeFailureRate   float64
	SnapshotFailureRate float64
	MigrateFailureRate  float64
	StatusFailureRate   float64
	RandomFailures      bool
	FailAfterTime       time.Duration // Fail operations after this time
}

// MockLatencyConfig simulates network and processing latency
type MockLatencyConfig struct {
	CreateLatency   time.Duration
	StartLatency    time.Duration
	StopLatency     time.Duration
	StatusLatency   time.Duration
	NetworkLatency  time.Duration
	DiskLatency     time.Duration
	VariabilityPct  float64 // 0.0 to 1.0 for latency variation
}

// NewMockHypervisor creates a new mock hypervisor
func NewMockHypervisor(nodeID, hypervisorID string) *MockHypervisor {
	return &MockHypervisor{
		vms:          make(map[string]*MockVM),
		nodeID:       nodeID,
		hypervisorID: hypervisorID,
		capabilities: MockCapabilities{
			SupportsPause:     true,
			SupportsResume:    true,
			SupportsSnapshot:  true,
			SupportsMigrate:   true,
			MaxVMs:           100,
			MaxCPUPerVM:      16,
			MaxMemoryPerVM:   32 * 1024, // 32GB in MB
			SupportedVMTypes: []VMType{VMTypeKVM, VMTypeContainer, VMTypeContainerd},
		},
		failures: MockFailureConfig{
			CreateFailureRate:   0.0,
			StartFailureRate:    0.0,
			StopFailureRate:     0.0,
			PauseFailureRate:    0.0,
			ResumeFailureRate:   0.0,
			SnapshotFailureRate: 0.0,
			MigrateFailureRate:  0.0,
			StatusFailureRate:   0.0,
			RandomFailures:      false,
		},
		latencyConfig: MockLatencyConfig{
			CreateLatency:  100 * time.Millisecond,
			StartLatency:   2 * time.Second,
			StopLatency:    1 * time.Second,
			StatusLatency:  10 * time.Millisecond,
			NetworkLatency: 5 * time.Millisecond,
			DiskLatency:    50 * time.Millisecond,
			VariabilityPct: 0.2, // 20% variation
		},
	}
}

// Configure allows changing mock behavior
func (m *MockHypervisor) Configure(failures MockFailureConfig, latency MockLatencyConfig, capabilities MockCapabilities) {
	m.vmLock.Lock()
	defer m.vmLock.Unlock()
	
	m.failures = failures
	m.latencyConfig = latency
	m.capabilities = capabilities
}

// simulateLatency introduces realistic latency with variability
func (m *MockHypervisor) simulateLatency(baseLatency time.Duration) {
	if baseLatency == 0 {
		return
	}

	variation := 1.0 + (rand.Float64()-0.5)*2*m.latencyConfig.VariabilityPct
	actualLatency := time.Duration(float64(baseLatency) * variation)
	time.Sleep(actualLatency)
}

// simulateFailure returns true if a failure should be injected
func (m *MockHypervisor) simulateFailure(failureRate float64) bool {
	if m.failures.RandomFailures && rand.Float64() < 0.01 {
		return true
	}
	return rand.Float64() < failureRate
}

// Create creates a mock VM
func (m *MockHypervisor) Create(ctx context.Context, config VMConfig) (string, error) {
	m.simulateLatency(m.latencyConfig.CreateLatency)
	
	if m.simulateFailure(m.failures.CreateFailureRate) {
		return "", fmt.Errorf("mock failure: create operation failed")
	}

	m.vmLock.Lock()
	defer m.vmLock.Unlock()

	// Check VM limits
	if len(m.vms) >= m.capabilities.MaxVMs {
		return "", fmt.Errorf("maximum VM limit (%d) reached", m.capabilities.MaxVMs)
	}

	// Validate resource limits
	if config.CPUShares > m.capabilities.MaxCPUPerVM {
		return "", fmt.Errorf("CPU shares %d exceeds maximum %d", config.CPUShares, m.capabilities.MaxCPUPerVM)
	}

	if config.MemoryMB > int(m.capabilities.MaxMemoryPerVM) {
		return "", fmt.Errorf("memory %d MB exceeds maximum %d MB", config.MemoryMB, m.capabilities.MaxMemoryPerVM)
	}

	vmID := config.ID
	if vmID == "" {
		vmID = fmt.Sprintf("mock-vm-%d", time.Now().UnixNano())
	}

	if _, exists := m.vms[vmID]; exists {
		return "", fmt.Errorf("VM with ID %s already exists", vmID)
	}

	mockVM := &MockVM{
		ID:        vmID,
		Config:    config,
		State:     StateCreated,
		CreatedAt: time.Now(),
		PID:       rand.Intn(65535) + 1000, // Mock PID
		snapshots: make(map[string]*MockSnapshot),
	}

	m.vms[vmID] = mockVM

	log.Printf("Mock hypervisor %s: Created VM %s (%s)", m.hypervisorID, config.Name, vmID)
	return vmID, nil
}

// Start starts a mock VM
func (m *MockHypervisor) Start(ctx context.Context, vmID string) error {
	m.simulateLatency(m.latencyConfig.StartLatency)
	
	if m.simulateFailure(m.failures.StartFailureRate) {
		return fmt.Errorf("mock failure: start operation failed")
	}

	m.vmLock.Lock()
	defer m.vmLock.Unlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vm.State == StateRunning {
		return fmt.Errorf("VM %s is already running", vmID)
	}

	vm.State = StateRunning
	now := time.Now()
	vm.StartedAt = &now

	// Start mock resource usage
	go m.simulateResourceUsage(vmID)

	log.Printf("Mock hypervisor %s: Started VM %s", m.hypervisorID, vmID)
	return nil
}

// Stop stops a mock VM
func (m *MockHypervisor) Stop(ctx context.Context, vmID string) error {
	m.simulateLatency(m.latencyConfig.StopLatency)
	
	if m.simulateFailure(m.failures.StopFailureRate) {
		return fmt.Errorf("mock failure: stop operation failed")
	}

	m.vmLock.Lock()
	defer m.vmLock.Unlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vm.State != StateRunning {
		return fmt.Errorf("VM %s is not running", vmID)
	}

	vm.State = StateStopped
	now := time.Now()
	vm.StoppedAt = &now

	log.Printf("Mock hypervisor %s: Stopped VM %s", m.hypervisorID, vmID)
	return nil
}

// Delete deletes a mock VM
func (m *MockHypervisor) Delete(ctx context.Context, vmID string) error {
	m.simulateLatency(m.latencyConfig.CreateLatency / 2) // Delete is typically faster
	
	if m.simulateFailure(m.failures.CreateFailureRate / 2) { // Lower failure rate for delete
		return fmt.Errorf("mock failure: delete operation failed")
	}

	m.vmLock.Lock()
	defer m.vmLock.Unlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	// Stop VM if running
	if vm.State == StateRunning {
		vm.State = StateStopped
	}

	delete(m.vms, vmID)

	log.Printf("Mock hypervisor %s: Deleted VM %s", m.hypervisorID, vmID)
	return nil
}

// GetStatus returns the status of a mock VM
func (m *MockHypervisor) GetStatus(ctx context.Context, vmID string) (VMState, error) {
	m.simulateLatency(m.latencyConfig.StatusLatency)
	
	if m.simulateFailure(m.failures.StatusFailureRate) {
		return StateUnknown, fmt.Errorf("mock failure: status operation failed")
	}

	m.vmLock.RLock()
	defer m.vmLock.RUnlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return StateUnknown, fmt.Errorf("VM %s not found", vmID)
	}

	return VMState(vm.State), nil
}

// GetInfo returns information about a mock VM
func (m *MockHypervisor) GetInfo(ctx context.Context, vmID string) (*VMInfo, error) {
	m.simulateLatency(m.latencyConfig.StatusLatency)

	m.vmLock.RLock()
	defer m.vmLock.RUnlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}

	return &VMInfo{
		ID:           vm.ID,
		Name:         vm.Config.Name,
		State:        vm.State,
		PID:          vm.PID,
		CPUShares:    vm.Config.CPUShares,
		MemoryMB:     vm.Config.MemoryMB,
		CreatedAt:    vm.CreatedAt,
		StartedAt:    vm.StartedAt,
		StoppedAt:    vm.StoppedAt,
		CPUUsage:     vm.CPUUsage,
		MemoryUsage:  vm.MemoryUsage,
		NetworkSent:  vm.NetworkTx,
		NetworkRecv:  vm.NetworkRx,
		Tags:         vm.Config.Tags,
		NetworkID:    vm.Config.NetworkID,
		RootFS:       vm.Config.RootFS,
	}, nil
}

// GetMetrics returns metrics for a mock VM
func (m *MockHypervisor) GetMetrics(ctx context.Context, vmID string) (*VMInfo, error) {
	return m.GetInfo(ctx, vmID)
}

// ListVMs returns a list of all mock VMs
func (m *MockHypervisor) ListVMs(ctx context.Context) ([]VMInfo, error) {
	m.simulateLatency(m.latencyConfig.StatusLatency * 2)

	m.vmLock.RLock()
	defer m.vmLock.RUnlock()

	vms := make([]VMInfo, 0, len(m.vms))
	for _, vm := range m.vms {
		vms = append(vms, VMInfo{
			ID:           vm.ID,
			Name:         vm.Config.Name,
			State:        vm.State,
			PID:          vm.PID,
			CPUShares:    vm.Config.CPUShares,
			MemoryMB:     vm.Config.MemoryMB,
			CreatedAt:    vm.CreatedAt,
			StartedAt:    vm.StartedAt,
			StoppedAt:    vm.StoppedAt,
			CPUUsage:     vm.CPUUsage,
			MemoryUsage:  vm.MemoryUsage,
			Tags:         vm.Config.Tags,
			NetworkID:    vm.Config.NetworkID,
			RootFS:       vm.Config.RootFS,
		})
	}

	return vms, nil
}

// Capability checks
func (m *MockHypervisor) SupportsPause() bool     { return m.capabilities.SupportsPause }
func (m *MockHypervisor) SupportsResume() bool    { return m.capabilities.SupportsResume }
func (m *MockHypervisor) SupportsSnapshot() bool  { return m.capabilities.SupportsSnapshot }
func (m *MockHypervisor) SupportsMigrate() bool   { return m.capabilities.SupportsMigrate }

// Pause pauses a mock VM
func (m *MockHypervisor) Pause(ctx context.Context, vmID string) error {
	if !m.capabilities.SupportsPause {
		return fmt.Errorf("pause not supported")
	}

	if m.simulateFailure(m.failures.PauseFailureRate) {
		return fmt.Errorf("mock failure: pause operation failed")
	}

	m.vmLock.Lock()
	defer m.vmLock.Unlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vm.State != StateRunning {
		return fmt.Errorf("VM %s is not running", vmID)
	}

	vm.State = StatePaused
	log.Printf("Mock hypervisor %s: Paused VM %s", m.hypervisorID, vmID)
	return nil
}

// Resume resumes a mock VM
func (m *MockHypervisor) Resume(ctx context.Context, vmID string) error {
	if !m.capabilities.SupportsResume {
		return fmt.Errorf("resume not supported")
	}

	if m.simulateFailure(m.failures.ResumeFailureRate) {
		return fmt.Errorf("mock failure: resume operation failed")
	}

	m.vmLock.Lock()
	defer m.vmLock.Unlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vm.State != StatePaused {
		return fmt.Errorf("VM %s is not paused", vmID)
	}

	vm.State = StateRunning
	log.Printf("Mock hypervisor %s: Resumed VM %s", m.hypervisorID, vmID)
	return nil
}

// Snapshot creates a snapshot of a mock VM
func (m *MockHypervisor) Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error) {
	if !m.capabilities.SupportsSnapshot {
		return "", fmt.Errorf("snapshots not supported")
	}

	if m.simulateFailure(m.failures.SnapshotFailureRate) {
		return "", fmt.Errorf("mock failure: snapshot operation failed")
	}

	m.vmLock.Lock()
	defer m.vmLock.Unlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return "", fmt.Errorf("VM %s not found", vmID)
	}

	snapshotID := fmt.Sprintf("snap-%s-%s-%d", vmID, name, time.Now().Unix())
	
	snapshot := &MockSnapshot{
		ID:        snapshotID,
		Name:      name,
		CreatedAt: time.Now(),
		Size:      int64(vm.Config.MemoryMB * 1024 * 1024), // Mock size based on memory
	}

	vm.snapshots[snapshotID] = snapshot

	log.Printf("Mock hypervisor %s: Created snapshot %s for VM %s", m.hypervisorID, snapshotID, vmID)
	return snapshotID, nil
}

// Migrate migrates a mock VM
func (m *MockHypervisor) Migrate(ctx context.Context, vmID, target string, params map[string]string) error {
	if !m.capabilities.SupportsMigrate {
		return fmt.Errorf("migration not supported")
	}

	if m.simulateFailure(m.failures.MigrateFailureRate) {
		return fmt.Errorf("mock failure: migrate operation failed")
	}

	m.vmLock.Lock()
	defer m.vmLock.Unlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	// Simulate migration time
	migrationTime := time.Duration(vm.Config.MemoryMB) * time.Millisecond * 10 // 10ms per MB
	m.simulateLatency(migrationTime)

	vm.migrationTarget = target
	vm.State = StateMigrating

	// Simulate migration completion
	go func() {
		time.Sleep(100 * time.Millisecond)
		m.vmLock.Lock()
		if vm, exists := m.vms[vmID]; exists {
			vm.State = StateRunning
		}
		m.vmLock.Unlock()
	}()

	log.Printf("Mock hypervisor %s: Started migration of VM %s to %s", m.hypervisorID, vmID, target)
	return nil
}

// simulateResourceUsage generates realistic resource usage patterns
func (m *MockHypervisor) simulateResourceUsage(vmID string) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			m.vmLock.Lock()
			vm, exists := m.vms[vmID]
			if !exists || vm.State != StateRunning {
				m.vmLock.Unlock()
				return
			}

			// Simulate CPU usage (0-100%)
			vm.CPUUsage = rand.Float64() * 100

			// Simulate memory usage (based on allocated memory)
			vm.MemoryUsage = int64(float64(vm.Config.MemoryMB*1024*1024) * (0.3 + rand.Float64()*0.4))

			// Simulate network I/O
			vm.NetworkRx += int64(rand.Intn(1000000))  // Up to 1MB/s
			vm.NetworkTx += int64(rand.Intn(500000))   // Up to 500KB/s

			// Simulate disk I/O
			vm.DiskRead += int64(rand.Intn(10000000))  // Up to 10MB/s
			vm.DiskWrite += int64(rand.Intn(5000000))  // Up to 5MB/s

			m.vmLock.Unlock()
		}
	}
}

// GetSnapshots returns all snapshots for a VM (additional method for testing)
func (m *MockHypervisor) GetSnapshots(vmID string) ([]*MockSnapshot, error) {
	m.vmLock.RLock()
	defer m.vmLock.RUnlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}

	snapshots := make([]*MockSnapshot, 0, len(vm.snapshots))
	for _, snapshot := range vm.snapshots {
		snapshots = append(snapshots, snapshot)
	}

	return snapshots, nil
}

// Reset clears all VMs (for testing)
func (m *MockHypervisor) Reset() {
	m.vmLock.Lock()
	defer m.vmLock.Unlock()
	
	m.vms = make(map[string]*MockVM)
	log.Printf("Mock hypervisor %s: Reset completed", m.hypervisorID)
}

// GetHypervisorInfo returns information about the mock hypervisor
func (m *MockHypervisor) GetHypervisorInfo() map[string]interface{} {
	m.vmLock.RLock()
	defer m.vmLock.RUnlock()

	return map[string]interface{}{
		"node_id":        m.nodeID,
		"hypervisor_id":  m.hypervisorID,
		"type":          "mock",
		"vm_count":      len(m.vms),
		"max_vms":       m.capabilities.MaxVMs,
		"capabilities":  m.capabilities,
		"failure_rates": m.failures,
		"latency":       m.latencyConfig,
	}
}