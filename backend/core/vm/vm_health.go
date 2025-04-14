package vm

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// HealthStatus represents the health status of a VM
type HealthStatus string

const (
	// HealthStatusHealthy indicates the VM is healthy
	HealthStatusHealthy HealthStatus = "healthy"

	// HealthStatusDegraded indicates the VM is degraded
	HealthStatusDegraded HealthStatus = "degraded"

	// HealthStatusUnhealthy indicates the VM is unhealthy
	HealthStatusUnhealthy HealthStatus = "unhealthy"

	// HealthStatusUnknown indicates the VM's health is unknown
	HealthStatusUnknown HealthStatus = "unknown"
)

// HealthCheck represents a health check for a VM
type HealthCheck struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Status      HealthStatus      `json:"status"`
	LastChecked time.Time         `json:"last_checked"`
	LastSuccess *time.Time        `json:"last_success,omitempty"`
	LastFailure *time.Time        `json:"last_failure,omitempty"`
	FailCount   int               `json:"fail_count"`
	Message     string            `json:"message,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// VMHealth represents the health of a VM
type VMHealth struct {
	VMID        string                  `json:"vm_id"`
	Status      HealthStatus            `json:"status"`
	LastChecked time.Time               `json:"last_checked"`
	Checks      map[string]*HealthCheck `json:"checks"`
}

// HealthCheckFunc is a function that performs a health check
type HealthCheckFunc func(ctx context.Context, vm *VM) (*HealthCheck, error)

// VMHealthManager manages VM health checks
type VMHealthManager struct {
	vmManager     *VMManager
	checks        map[string]HealthCheckFunc
	checksMutex   sync.RWMutex
	health        map[string]*VMHealth
	healthMutex   sync.RWMutex
	checkInterval time.Duration
	ctx           context.Context
	cancel        context.CancelFunc
	eventManager  *VMEventManager
}

// NewVMHealthManager creates a new VM health manager
func NewVMHealthManager(vmManager *VMManager, eventManager *VMEventManager, checkInterval time.Duration) *VMHealthManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &VMHealthManager{
		vmManager:     vmManager,
		checks:        make(map[string]HealthCheckFunc),
		health:        make(map[string]*VMHealth),
		checkInterval: checkInterval,
		ctx:           ctx,
		cancel:        cancel,
		eventManager:  eventManager,
	}
}

// RegisterCheck registers a health check
func (m *VMHealthManager) RegisterCheck(id, name, description string, check HealthCheckFunc) {
	m.checksMutex.Lock()
	defer m.checksMutex.Unlock()

	m.checks[id] = check

	log.Printf("Registered health check %s: %s", id, name)
}

// UnregisterCheck unregisters a health check
func (m *VMHealthManager) UnregisterCheck(id string) {
	m.checksMutex.Lock()
	defer m.checksMutex.Unlock()

	delete(m.checks, id)

	log.Printf("Unregistered health check %s", id)
}

// Start starts the health check manager
func (m *VMHealthManager) Start() {
	log.Println("Starting VM health check manager")

	// Start the health check loop
	go m.checkLoop()
}

// Stop stops the health check manager
func (m *VMHealthManager) Stop() {
	log.Println("Stopping VM health check manager")
	m.cancel()
}

// GetVMHealth returns the health of a VM
func (m *VMHealthManager) GetVMHealth(vmID string) (*VMHealth, error) {
	m.healthMutex.RLock()
	defer m.healthMutex.RUnlock()

	health, exists := m.health[vmID]
	if !exists {
		return nil, fmt.Errorf("no health information for VM %s", vmID)
	}

	return health, nil
}

// GetAllVMHealth returns the health of all VMs
func (m *VMHealthManager) GetAllVMHealth() map[string]*VMHealth {
	m.healthMutex.RLock()
	defer m.healthMutex.RUnlock()

	// Create a copy of the health map
	health := make(map[string]*VMHealth, len(m.health))
	for id, h := range m.health {
		health[id] = h
	}

	return health
}

// CheckVMHealth checks the health of a VM
func (m *VMHealthManager) CheckVMHealth(ctx context.Context, vmID string) (*VMHealth, error) {
	// Get the VM
	vm, err := m.vmManager.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM: %w", err)
	}

	// Get registered checks
	m.checksMutex.RLock()
	checks := make(map[string]HealthCheckFunc, len(m.checks))
	for id, check := range m.checks {
		checks[id] = check
	}
	m.checksMutex.RUnlock()

	// Create health object
	health := &VMHealth{
		VMID:        vmID,
		Status:      HealthStatusUnknown,
		LastChecked: time.Now(),
		Checks:      make(map[string]*HealthCheck),
	}

	// Run checks
	var unhealthyCount, degradedCount, healthyCount int
	for id, check := range checks {
		checkResult, err := check(ctx, vm)
		if err != nil {
			log.Printf("Error running health check %s for VM %s: %v", id, vmID, err)
			continue
		}

		health.Checks[id] = checkResult

		switch checkResult.Status {
		case HealthStatusUnhealthy:
			unhealthyCount++
		case HealthStatusDegraded:
			degradedCount++
		case HealthStatusHealthy:
			healthyCount++
		}
	}

	// Determine overall health status
	if unhealthyCount > 0 {
		health.Status = HealthStatusUnhealthy
	} else if degradedCount > 0 {
		health.Status = HealthStatusDegraded
	} else if healthyCount > 0 {
		health.Status = HealthStatusHealthy
	} else {
		health.Status = HealthStatusUnknown
	}

	// Store health
	m.healthMutex.Lock()
	m.health[vmID] = health
	m.healthMutex.Unlock()

	// Emit event if status changed
	m.emitHealthEvent(vm, health)

	return health, nil
}

// checkLoop runs health checks periodically
func (m *VMHealthManager) checkLoop() {
	ticker := time.NewTicker(m.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.checkAllVMs()
		}
	}
}

// checkAllVMs checks the health of all VMs
func (m *VMHealthManager) checkAllVMs() {
	// Get all VMs
	vms := m.vmManager.ListVMs()

	// Check each VM
	for _, vm := range vms {
		// Skip VMs that are not running
		if vm.State() != StateRunning {
			continue
		}

		// Create a context with timeout
		ctx, cancel := context.WithTimeout(m.ctx, m.checkInterval/2)

		// Check VM health
		_, err := m.CheckVMHealth(ctx, vm.ID())
		if err != nil {
			log.Printf("Error checking health for VM %s: %v", vm.ID(), err)
		}

		cancel()
	}
}

// emitHealthEvent emits a health event if the status changed
func (m *VMHealthManager) emitHealthEvent(vm *VM, health *VMHealth) {
	// Get previous health
	m.healthMutex.RLock()
	prevHealth, exists := m.health[vm.ID()]
	m.healthMutex.RUnlock()

	// If no previous health or status changed, emit event
	if !exists || prevHealth.Status != health.Status {
		// Create event data
		data := map[string]interface{}{
			"health_status": health.Status,
			"checks":        health.Checks,
		}

		// Determine event message
		var message string
		switch health.Status {
		case HealthStatusHealthy:
			message = "VM is healthy"
		case HealthStatusDegraded:
			message = "VM is degraded"
		case HealthStatusUnhealthy:
			message = "VM is unhealthy"
		default:
			message = "VM health status is unknown"
		}

		// Emit event
		m.eventManager.EmitEvent(VMEvent{
			Type:      VMEventUpdated,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.GetNodeID(),
			Message:   message,
			Data:      data,
		})
	}
}

// Standard health checks

// ProcessRunningCheck checks if the VM process is running
func ProcessRunningCheck(ctx context.Context, vm *VM) (*HealthCheck, error) {
	check := &HealthCheck{
		ID:          "process_running",
		Name:        "Process Running",
		Description: "Checks if the VM process is running",
		LastChecked: time.Now(),
		Status:      HealthStatusUnknown,
	}

	// Check if VM is running
	if vm.State() != StateRunning {
		check.Status = HealthStatusUnhealthy
		check.Message = fmt.Sprintf("VM is not running (state: %s)", vm.State())
		now := time.Now()
		check.LastFailure = &now
		check.FailCount++
		return check, nil
	}

	// Check if process info is available
	processInfo := vm.GetProcessInfo()
	if processInfo.PID == 0 {
		check.Status = HealthStatusUnhealthy
		check.Message = "VM process information not available"
		now := time.Now()
		check.LastFailure = &now
		check.FailCount++
		return check, nil
	}

	// Check if process is running
	// In a real implementation, this would check if the process is still running
	// For simplicity, we'll assume it's running if we have a PID
	check.Status = HealthStatusHealthy
	check.Message = fmt.Sprintf("VM process is running (PID: %d)", processInfo.PID)
	now := time.Now()
	check.LastSuccess = &now

	return check, nil
}

// ResourceUsageCheck checks if the VM's resource usage is within acceptable limits
func ResourceUsageCheck(ctx context.Context, vm *VM) (*HealthCheck, error) {
	check := &HealthCheck{
		ID:          "resource_usage",
		Name:        "Resource Usage",
		Description: "Checks if the VM's resource usage is within acceptable limits",
		LastChecked: time.Now(),
		Status:      HealthStatusUnknown,
	}

	// Check if VM is running
	if vm.State() != StateRunning {
		check.Status = HealthStatusUnhealthy
		check.Message = fmt.Sprintf("VM is not running (state: %s)", vm.State())
		now := time.Now()
		check.LastFailure = &now
		check.FailCount++
		return check, nil
	}

	// Check if process info is available
	processInfo := vm.GetProcessInfo()
	if processInfo.PID == 0 {
		check.Status = HealthStatusUnhealthy
		check.Message = "VM process information not available"
		now := time.Now()
		check.LastFailure = &now
		check.FailCount++
		return check, nil
	}

	// Check CPU usage
	if processInfo.CPUUsagePercent > 90 {
		check.Status = HealthStatusDegraded
		check.Message = fmt.Sprintf("VM CPU usage is high: %.2f%%", processInfo.CPUUsagePercent)
		now := time.Now()
		check.LastFailure = &now
		check.FailCount++
		return check, nil
	}

	// Check memory usage
	// In a real implementation, this would compare against the VM's memory limit
	// For simplicity, we'll use a fixed threshold
	if int64(processInfo.MemoryUsageMB) > 1024 {
		check.Status = HealthStatusDegraded
		check.Message = fmt.Sprintf("VM memory usage is high: %d MB", processInfo.MemoryUsageMB)
		now := time.Now()
		check.LastFailure = &now
		check.FailCount++
		return check, nil
	}

	// All checks passed
	check.Status = HealthStatusHealthy
	check.Message = "VM resource usage is within acceptable limits"
	now := time.Now()
	check.LastSuccess = &now

	return check, nil
}
