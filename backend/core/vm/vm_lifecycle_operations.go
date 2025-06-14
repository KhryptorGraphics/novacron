package vm

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// RebootVM reboots the VM (stop and start)
func (vm *VM) Reboot() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state != StateRunning {
		return fmt.Errorf("VM is not running")
	}

	log.Printf("Rebooting VM %s (%s)", vm.Name(), vm.ID())
	
	// Set state to restarting
	vm.state = StateRestarting

	// Stop the VM
	if err := vm.stopInternal(); err != nil {
		return fmt.Errorf("failed to stop VM for reboot: %w", err)
	}

	// Wait a moment before restarting
	time.Sleep(2 * time.Second)

	// Start the VM again
	if err := vm.startInternal(); err != nil {
		vm.state = StateFailed
		return fmt.Errorf("failed to start VM after reboot: %w", err)
	}

	log.Printf("Successfully rebooted VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// Pause pauses the VM
func (vm *VM) Pause() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state != StateRunning {
		return fmt.Errorf("VM is not running")
	}

	log.Printf("Pausing VM %s (%s)", vm.Name(), vm.ID())

	// Send SIGSTOP to pause the process
	if vm.cmd != nil && vm.cmd.Process != nil {
		if err := vm.cmd.Process.Signal(syscall.SIGSTOP); err != nil {
			return fmt.Errorf("failed to send SIGSTOP: %w", err)
		}
	}

	vm.state = StatePaused
	vm.updatedAt = time.Now()

	log.Printf("Successfully paused VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// Resume resumes a paused VM
func (vm *VM) Resume() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state != StatePaused {
		return fmt.Errorf("VM is not paused")
	}

	log.Printf("Resuming VM %s (%s)", vm.Name(), vm.ID())

	// Send SIGCONT to resume the process
	if vm.cmd != nil && vm.cmd.Process != nil {
		if err := vm.cmd.Process.Signal(syscall.SIGCONT); err != nil {
			return fmt.Errorf("failed to send SIGCONT: %w", err)
		}
	}

	vm.state = StateRunning
	vm.updatedAt = time.Now()

	log.Printf("Successfully resumed VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// Suspend suspends the VM to disk (hibernate)
func (vm *VM) Suspend() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state != StateRunning {
		return fmt.Errorf("VM is not running")
	}

	log.Printf("Suspending VM %s (%s)", vm.Name(), vm.ID())

	// For process-based VMs, we'll use pause instead of true hibernation
	// In a real KVM implementation, this would save memory state to disk
	if err := vm.cmd.Process.Signal(syscall.SIGSTOP); err != nil {
		return fmt.Errorf("failed to suspend VM: %w", err)
	}

	vm.state = StatePaused // Using paused state for suspended VMs
	vm.updatedAt = time.Now()

	log.Printf("Successfully suspended VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// Clone creates a copy of the VM
func (vm *VM) Clone(newName string) (*VM, error) {
	vm.mutex.RLock()
	config := vm.config
	vm.mutex.RUnlock()

	// Create a new config based on the current VM
	newConfig := VMConfig{
		Name:      newName,
		Command:   config.Command,
		Args:      make([]string, len(config.Args)),
		CPUShares: config.CPUShares,
		MemoryMB:  config.MemoryMB,
		RootFS:    config.RootFS,
		Mounts:    make([]Mount, len(config.Mounts)),
		Env:       make(map[string]string),
		NetworkID: config.NetworkID,
		WorkDir:   config.WorkDir,
		Tags:      make(map[string]string),
	}

	// Copy arrays and maps
	copy(newConfig.Args, config.Args)
	copy(newConfig.Mounts, config.Mounts)
	
	for k, v := range config.Env {
		newConfig.Env[k] = v
	}
	
	for k, v := range config.Tags {
		newConfig.Tags[k] = v
	}

	// Add clone-specific tags
	newConfig.Tags["cloned_from"] = vm.ID()
	newConfig.Tags["clone_time"] = time.Now().Format(time.RFC3339)

	// Create the new VM
	clonedVM, err := NewVM(newConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create cloned VM: %w", err)
	}

	log.Printf("Successfully cloned VM %s (%s) to %s (%s)", 
		vm.Name(), vm.ID(), clonedVM.Name(), clonedVM.ID())

	return clonedVM, nil
}

// GetResourceUsage returns current resource usage
func (vm *VM) GetResourceUsage() VMResourceUsage {
	vm.statsLock.RLock()
	defer vm.statsLock.RUnlock()

	return VMResourceUsage{
		CPUPercent:    vm.stats.CPUUsage,
		MemoryBytes:   vm.stats.MemoryUsage,
		NetworkSent:   vm.stats.NetworkSent,
		NetworkRecv:   vm.stats.NetworkRecv,
		LastCollected: vm.stats.LastUpdated,
	}
}

// UpdateResourceLimits updates the resource limits for the VM
func (vm *VM) UpdateResourceLimits(cpuShares int, memoryMB int) error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if cpuShares > 0 {
		vm.config.CPUShares = cpuShares
	}
	
	if memoryMB > 0 {
		vm.config.MemoryMB = memoryMB
	}

	vm.updatedAt = time.Now()

	// In a real implementation, we would update cgroup limits here
	log.Printf("Updated resource limits for VM %s: CPU=%d, Memory=%dMB", 
		vm.ID(), vm.config.CPUShares, vm.config.MemoryMB)

	return nil
}

// ForceKill forcefully kills the VM process
func (vm *VM) ForceKill() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.cmd == nil || vm.cmd.Process == nil {
		return fmt.Errorf("VM process not found")
	}

	log.Printf("Force killing VM %s (%s)", vm.Name(), vm.ID())

	if err := vm.cmd.Process.Kill(); err != nil {
		return fmt.Errorf("failed to force kill VM: %w", err)
	}

	// Wait for process to exit
	go func() {
		vm.cmd.Wait()
		vm.mutex.Lock()
		now := time.Now()
		vm.stoppedAt = &now
		vm.state = StateStopped
		vm.updatedAt = time.Now()
		vm.mutex.Unlock()
	}()

	log.Printf("Force killed VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// GetProcessStats returns detailed process statistics
func (vm *VM) GetProcessStats() (VMProcessStats, error) {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()

	if vm.pid == 0 {
		return VMProcessStats{}, fmt.Errorf("VM process not running")
	}

	// In a real implementation, we would read from /proc/[pid]/stat
	// For now, return basic information
	stats := VMProcessStats{
		PID:         vm.pid,
		State:       string(vm.state),
		CPUTime:     0, // Would be read from /proc/[pid]/stat
		MemoryRSS:   0, // Would be read from /proc/[pid]/status
		MemoryVSZ:   0, // Would be read from /proc/[pid]/status
		OpenFiles:   0, // Would be read from /proc/[pid]/fd/
		Threads:     1, // Would be read from /proc/[pid]/status
		StartTime:   vm.createdAt,
	}

	return stats, nil
}

// SendSignal sends a custom signal to the VM process
func (vm *VM) SendSignal(signal os.Signal) error {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()

	if vm.cmd == nil || vm.cmd.Process == nil {
		return fmt.Errorf("VM process not found")
	}

	if err := vm.cmd.Process.Signal(signal); err != nil {
		return fmt.Errorf("failed to send signal %v: %w", signal, err)
	}

	log.Printf("Sent signal %v to VM %s (%s)", signal, vm.Name(), vm.ID())
	return nil
}

// Internal helper methods

// startInternal starts the VM without locking (for internal use)
func (vm *VM) startInternal() error {
	// Set up cgroups
	if err := vm.setupCgroups(); err != nil {
		return fmt.Errorf("failed to set up cgroups: %w", err)
	}

	// Set up namespaces
	if err := vm.setupNamespaces(); err != nil {
		return fmt.Errorf("failed to set up namespaces: %w", err)
	}

	// Set up command
	vm.cmd = exec.Command(vm.config.Command, vm.config.Args...)

	// Set up process namespaces
	vm.cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWUTS | syscall.CLONE_NEWPID | syscall.CLONE_NEWNS | syscall.CLONE_NEWNET | syscall.CLONE_NEWIPC,
	}

	// Set up environment variables
	if vm.config.Env != nil {
		env := os.Environ()
		for k, v := range vm.config.Env {
			env = append(env, fmt.Sprintf("%s=%s", k, v))
		}
		vm.cmd.Env = env
	}

	// Set up working directory
	if vm.config.WorkDir != "" {
		vm.cmd.Dir = vm.config.WorkDir
	}

	// Start the process
	if err := vm.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start VM process: %w", err)
	}

	vm.pid = vm.cmd.Process.Pid

	// Move process to appropriate cgroups
	if err := vm.assignToCgroups(); err != nil {
		return fmt.Errorf("failed to assign process to cgroups: %w", err)
	}

	// Set up network if needed
	if vm.config.NetworkID != "" {
		if err := vm.setupNetwork(); err != nil {
			return fmt.Errorf("failed to set up network: %w", err)
		}
	}

	now := time.Now()
	vm.startedAt = &now
	vm.state = StateRunning
	vm.updatedAt = time.Now()

	// Monitor the process
	go vm.monitor()

	return nil
}

// stopInternal stops the VM without locking (for internal use)
func (vm *VM) stopInternal() error {
	if vm.state != StateRunning && vm.state != StatePaused {
		return fmt.Errorf("VM is not running or paused")
	}

	// Send SIGTERM signal
	if err := vm.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		log.Printf("Failed to send SIGTERM to VM %s: %v, will try SIGKILL", vm.ID(), err)

		// If SIGTERM fails, try SIGKILL
		if err := vm.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill VM: %w", err)
		}
	}

	// Wait for the process to exit
	done := make(chan error)
	go func() {
		done <- vm.cmd.Wait()
	}()

	// Wait for process to exit or timeout
	select {
	case err := <-done:
		if err != nil {
			log.Printf("VM %s process exited with error: %v", vm.ID(), err)
		}
	case <-time.After(10 * time.Second):
		// If it doesn't exit within timeout, force kill
		if err := vm.cmd.Process.Kill(); err != nil {
			log.Printf("Failed to force kill VM %s: %v", vm.ID(), err)
		}
		<-done // Wait for the process to be killed
	}

	now := time.Now()
	vm.stoppedAt = &now
	vm.state = StateStopped
	vm.updatedAt = time.Now()

	// Clean up cgroups and namespaces
	if err := vm.cleanupCgroups(); err != nil {
		log.Printf("Failed to clean up cgroups for VM %s: %v", vm.ID(), err)
	}

	if err := vm.cleanupNamespaces(); err != nil {
		log.Printf("Failed to clean up namespaces for VM %s: %v", vm.ID(), err)
	}

	return nil
}

// VMResourceUsage contains resource usage information
type VMResourceUsage struct {
	CPUPercent    float64   `json:"cpu_percent"`
	MemoryBytes   int64     `json:"memory_bytes"`
	NetworkSent   int64     `json:"network_sent"`
	NetworkRecv   int64     `json:"network_recv"`
	LastCollected time.Time `json:"last_collected"`
}

// VMProcessStats contains detailed process statistics
type VMProcessStats struct {
	PID         int       `json:"pid"`
	State       string    `json:"state"`
	CPUTime     int64     `json:"cpu_time_ms"`
	MemoryRSS   int64     `json:"memory_rss_bytes"`
	MemoryVSZ   int64     `json:"memory_vsz_bytes"`
	OpenFiles   int       `json:"open_files"`
	Threads     int       `json:"threads"`
	StartTime   time.Time `json:"start_time"`
}