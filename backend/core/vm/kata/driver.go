package kata

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"sync"
	"time"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cio"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/oci"
	"github.com/opencontainers/runtime-spec/specs-go"
	
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// KataDriver implements the VM driver interface for Kata Containers
// Provides VM-level isolation with container efficiency
type KataDriver struct {
	client     *containerd.Client
	runtime    string
	namespace  string
	containers map[string]*KataContainer
	mutex      sync.RWMutex
	metrics    *KataMetrics
	config     *KataConfig
}

// KataConfig holds Kata Containers driver configuration
type KataConfig struct {
	// Containerd settings
	ContainerdSocket   string `json:"containerd_socket"`
	ContainerdNamespace string `json:"containerd_namespace"`
	KataRuntime        string `json:"kata_runtime"`
	
	// VM isolation settings
	DefaultVCPUs       int    `json:"default_vcpus"`
	DefaultMemoryMB    int64  `json:"default_memory_mb"`
	DefaultKernelPath  string `json:"default_kernel_path"`
	DefaultInitrdPath  string `json:"default_initrd_path"`
	DefaultRootfsPath  string `json:"default_rootfs_path"`
	
	// Security settings
	EnableNetworkNamespace bool `json:"enable_network_namespace"`
	EnablePIDNamespace     bool `json:"enable_pid_namespace"`
	EnableUserNamespace    bool `json:"enable_user_namespace"`
	EnableSeccomp          bool `json:"enable_seccomp"`
	EnableAppArmor         bool `json:"enable_apparmor"`
	
	// Performance settings
	EnableHugepages        bool   `json:"enable_hugepages"`
	EnableVhostUser        bool   `json:"enable_vhost_user"`
	EnableVsock            bool   `json:"enable_vsock"`
	IOEngine               string `json:"io_engine"` // io_uring, epoll
	
	// Resource limits
	MaxContainers          int   `json:"max_containers"`
	MaxMemoryPerContainer  int64 `json:"max_memory_per_container_mb"`
	MaxCPUPerContainer     int   `json:"max_cpu_per_container"`
	
	// Monitoring
	EnableMetrics          bool   `json:"enable_metrics"`
	MetricsInterval        string `json:"metrics_interval"`
}

// KataContainer represents a Kata Container instance
type KataContainer struct {
	// Container identification
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Image       string            `json:"image"`
	Labels      map[string]string `json:"labels"`
	
	// VM-level properties
	VMConfig    VMConfiguration   `json:"vm_config"`
	VMStatus    VMStatus         `json:"vm_status"`
	
	// Container properties
	ContainerID string           `json:"container_id"`
	Spec        *specs.Spec      `json:"spec"`
	Status      ContainerStatus  `json:"status"`
	
	// Resources
	Resources   ResourceAllocation `json:"resources"`
	Metrics     ContainerMetrics  `json:"metrics"`
	
	// Lifecycle
	CreatedAt   time.Time        `json:"created_at"`
	StartedAt   *time.Time       `json:"started_at,omitempty"`
	FinishedAt  *time.Time       `json:"finished_at,omitempty"`
	
	// Runtime state
	Task        containerd.Task  `json:"-"`
	Container   containerd.Container `json:"-"`
}

type VMConfiguration struct {
	VCPUs      int    `json:"vcpus"`
	MemoryMB   int64  `json:"memory_mb"`
	KernelPath string `json:"kernel_path"`
	InitrdPath string `json:"initrd_path"`
	RootfsPath string `json:"rootfs_path"`
	MachineType string `json:"machine_type"`
	Hypervisor  string `json:"hypervisor"`
}

type VMStatus struct {
	State       string    `json:"state"` // starting, running, paused, stopped
	VMPID       int       `json:"vm_pid"`
	QMPSocket   string    `json:"qmp_socket"`
	VSockCID    uint32    `json:"vsock_cid"`
	IPAddress   string    `json:"ip_address"`
	LastUpdated time.Time `json:"last_updated"`
}

type ContainerStatus struct {
	State       string    `json:"state"` // created, running, paused, stopped
	PID         uint32    `json:"pid"`
	ExitCode    uint32    `json:"exit_code"`
	Error       string    `json:"error,omitempty"`
	LastUpdated time.Time `json:"last_updated"`
}

type ResourceAllocation struct {
	CPUShares     int64   `json:"cpu_shares"`
	CPUQuota      int64   `json:"cpu_quota"`
	CPUPeriod     int64   `json:"cpu_period"`
	MemoryLimit   int64   `json:"memory_limit"`
	MemorySwap    int64   `json:"memory_swap"`
	PidsLimit     int64   `json:"pids_limit"`
	BlkioWeight   uint16  `json:"blkio_weight"`
	NetworkBandwidth int64 `json:"network_bandwidth"`
}

type ContainerMetrics struct {
	CPUUsage      float64 `json:"cpu_usage"`
	MemoryUsage   int64   `json:"memory_usage"`
	MemoryLimit   int64   `json:"memory_limit"`
	NetworkRxBytes int64  `json:"network_rx_bytes"`
	NetworkTxBytes int64  `json:"network_tx_bytes"`
	BlockRead     int64   `json:"block_read"`
	BlockWrite    int64   `json:"block_write"`
	PIDs          int64   `json:"pids"`
	Timestamp     time.Time `json:"timestamp"`
}

type KataMetrics struct {
	TotalContainers      int64   `json:"total_containers"`
	RunningContainers    int64   `json:"running_containers"`
	TotalMemoryUsage     int64   `json:"total_memory_usage"`
	TotalCPUUsage        float64 `json:"total_cpu_usage"`
	AverageStartupTime   float64 `json:"average_startup_time"`
	FailureRate          float64 `json:"failure_rate"`
	LastMetricUpdate     time.Time `json:"last_metric_update"`
}

// NewKataDriver creates a new Kata Containers driver
func NewKataDriver(config *KataConfig) (*KataDriver, error) {
	if config == nil {
		config = getDefaultKataConfig()
	}
	
	// Connect to containerd
	client, err := containerd.New(config.ContainerdSocket)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to containerd: %w", err)
	}
	
	driver := &KataDriver{
		client:     client,
		runtime:    config.KataRuntime,
		namespace:  config.ContainerdNamespace,
		containers: make(map[string]*KataContainer),
		config:     config,
		metrics:    &KataMetrics{},
	}
	
	// Verify Kata runtime availability
	if err := driver.verifyKataRuntime(); err != nil {
		return nil, fmt.Errorf("kata runtime verification failed: %w", err)
	}
	
	log.Printf("Kata Containers driver initialized successfully")
	return driver, nil
}

func getDefaultKataConfig() *KataConfig {
	return &KataConfig{
		ContainerdSocket:       "/run/containerd/containerd.sock",
		ContainerdNamespace:    "novacron-kata",
		KataRuntime:            "io.containerd.kata.v2",
		DefaultVCPUs:           1,
		DefaultMemoryMB:        512,
		DefaultKernelPath:      "/usr/share/kata-containers/vmlinux.container",
		DefaultInitrdPath:      "/usr/share/kata-containers/kata-containers-initrd.img",
		EnableNetworkNamespace: true,
		EnablePIDNamespace:     true,
		EnableUserNamespace:    false,
		EnableSeccomp:          true,
		EnableAppArmor:         true,
		EnableHugepages:        false,
		EnableVhostUser:        true,
		EnableVsock:            true,
		IOEngine:               "io_uring",
		MaxContainers:          100,
		MaxMemoryPerContainer:  4096, // 4GB
		MaxCPUPerContainer:     4,
		EnableMetrics:          true,
		MetricsInterval:        "30s",
	}
}

// Implement VMDriver interface
func (k *KataDriver) CreateVM(ctx context.Context, vmSpec vm.VMSpec) (*vm.VM, error) {
	// Convert VM spec to Kata container
	kataContainer, err := k.createKataContainer(ctx, vmSpec)
	if err != nil {
		return nil, fmt.Errorf("failed to create kata container: %w", err)
	}
	
	// Convert back to VM representation
	vmInstance := k.containerToVM(kataContainer)
	
	k.mutex.Lock()
	k.containers[kataContainer.ID] = kataContainer
	k.mutex.Unlock()
	
	log.Printf("Created Kata container %s (VM %s)", kataContainer.ID, vmInstance.ID)
	return vmInstance, nil
}

func (k *KataDriver) StartVM(ctx context.Context, vmID string) error {
	k.mutex.RLock()
	kataContainer, exists := k.containers[vmID]
	k.mutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("container %s not found", vmID)
	}
	
	// Start the container task
	task, err := kataContainer.Container.NewTask(ctx, cio.NewCreator(cio.WithStdio))
	if err != nil {
		return fmt.Errorf("failed to create task: %w", err)
	}
	
	kataContainer.Task = task
	
	// Start the task
	if err := task.Start(ctx); err != nil {
		return fmt.Errorf("failed to start task: %w", err)
	}
	
	// Update status
	now := time.Now()
	kataContainer.StartedAt = &now
	kataContainer.Status.State = "running"
	kataContainer.Status.PID = task.Pid()
	kataContainer.Status.LastUpdated = now
	
	log.Printf("Started Kata container %s", vmID)
	return nil
}

func (k *KataDriver) StopVM(ctx context.Context, vmID string) error {
	k.mutex.RLock()
	kataContainer, exists := k.containers[vmID]
	k.mutex.RUnlock()
	
	if !exists {
		return fmt.Errorf("container %s not found", vmID)
	}
	
	if kataContainer.Task == nil {
		return fmt.Errorf("container %s not running", vmID)
	}
	
	// Stop the task
	if err := kataContainer.Task.Kill(ctx, syscall.SIGTERM); err != nil {
		log.Printf("Failed to send SIGTERM to container %s: %v", vmID, err)
		// Force kill
		if err := kataContainer.Task.Kill(ctx, syscall.SIGKILL); err != nil {
			return fmt.Errorf("failed to force kill container: %w", err)
		}
	}
	
	// Wait for exit
	exitStatus, err := kataContainer.Task.Wait(ctx)
	if err != nil {
		log.Printf("Error waiting for container %s to exit: %v", vmID, err)
	}
	
	// Update status
	now := time.Now()
	kataContainer.FinishedAt = &now
	kataContainer.Status.State = "stopped"
	kataContainer.Status.ExitCode = exitStatus.ExitCode()
	kataContainer.Status.LastUpdated = now
	
	log.Printf("Stopped Kata container %s", vmID)
	return nil
}

func (k *KataDriver) DeleteVM(ctx context.Context, vmID string) error {
	k.mutex.Lock()
	defer k.mutex.Unlock()
	
	kataContainer, exists := k.containers[vmID]
	if !exists {
		return fmt.Errorf("container %s not found", vmID)
	}
	
	// Stop if running
	if kataContainer.Task != nil && kataContainer.Status.State == "running" {
		if err := k.StopVM(ctx, vmID); err != nil {
			log.Printf("Warning: failed to stop container before delete: %v", err)
		}
	}
	
	// Delete task
	if kataContainer.Task != nil {
		if _, err := kataContainer.Task.Delete(ctx); err != nil {
			log.Printf("Warning: failed to delete task: %v", err)
		}
	}
	
	// Delete container
	if err := kataContainer.Container.Delete(ctx, containerd.WithSnapshotCleanup); err != nil {
		return fmt.Errorf("failed to delete container: %w", err)
	}
	
	delete(k.containers, vmID)
	
	log.Printf("Deleted Kata container %s", vmID)
	return nil
}

func (k *KataDriver) GetVM(ctx context.Context, vmID string) (*vm.VM, error) {
	k.mutex.RLock()
	kataContainer, exists := k.containers[vmID]
	k.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("container %s not found", vmID)
	}
	
	// Refresh container status
	if err := k.refreshContainerStatus(ctx, kataContainer); err != nil {
		log.Printf("Warning: failed to refresh container status: %v", err)
	}
	
	return k.containerToVM(kataContainer), nil
}

func (k *KataDriver) ListVMs(ctx context.Context) ([]*vm.VM, error) {
	k.mutex.RLock()
	defer k.mutex.RUnlock()
	
	vms := make([]*vm.VM, 0, len(k.containers))
	for _, kataContainer := range k.containers {
		vms = append(vms, k.containerToVM(kataContainer))
	}
	
	return vms, nil
}

func (k *KataDriver) MigrateVM(ctx context.Context, vmID string, targetHost string) error {
	// Kata containers don't support traditional live migration
	// Instead, we implement checkpoint/restore migration
	return k.checkpointRestoreMigration(ctx, vmID, targetHost)
}

func (k *KataDriver) GetVMMetrics(ctx context.Context, vmID string) (*vm.VMMetrics, error) {
	k.mutex.RLock()
	kataContainer, exists := k.containers[vmID]
	k.mutex.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("container %s not found", vmID)
	}
	
	// Collect metrics from container
	if err := k.collectContainerMetrics(ctx, kataContainer); err != nil {
		return nil, fmt.Errorf("failed to collect metrics: %w", err)
	}
	
	// Convert to VM metrics
	vmMetrics := &vm.VMMetrics{
		VMID:           vmID,
		CPUUsage:       kataContainer.Metrics.CPUUsage,
		MemoryUsage:    kataContainer.Metrics.MemoryUsage,
		MemoryLimit:    kataContainer.Metrics.MemoryLimit,
		NetworkRxBytes: kataContainer.Metrics.NetworkRxBytes,
		NetworkTxBytes: kataContainer.Metrics.NetworkTxBytes,
		DiskReadBytes:  kataContainer.Metrics.BlockRead,
		DiskWriteBytes: kataContainer.Metrics.BlockWrite,
		Timestamp:      kataContainer.Metrics.Timestamp,
	}
	
	return vmMetrics, nil
}

// Kata-specific implementation methods
func (k *KataDriver) createKataContainer(ctx context.Context, vmSpec vm.VMSpec) (*KataContainer, error) {
	// Use containerd namespace
	ctx = namespaces.WithNamespace(ctx, k.namespace)
	
	// Pull image if specified
	var image containerd.Image
	var err error
	
	if vmSpec.Image != "" {
		image, err = k.client.Pull(ctx, vmSpec.Image, containerd.WithPullUnpack)
		if err != nil {
			return nil, fmt.Errorf("failed to pull image %s: %w", vmSpec.Image, err)
		}
	}
	
	// Create container spec
	spec, err := k.createContainerSpec(vmSpec)
	if err != nil {
		return nil, fmt.Errorf("failed to create container spec: %w", err)
	}
	
	// Create container
	containerOpts := []containerd.NewContainerOpts{
		containerd.WithSpec(spec),
		containerd.WithRuntime(k.runtime, nil),
	}
	
	if image != nil {
		containerOpts = append(containerOpts, containerd.WithImage(image))
		containerOpts = append(containerOpts, containerd.WithSnapshot(vmSpec.ID))
	}
	
	container, err := k.client.NewContainer(ctx, vmSpec.ID, containerOpts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}
	
	// Create Kata container wrapper
	kataContainer := &KataContainer{
		ID:          vmSpec.ID,
		Name:        vmSpec.Name,
		Image:       vmSpec.Image,
		Labels:      vmSpec.Labels,
		ContainerID: vmSpec.ID,
		Spec:        spec,
		Container:   container,
		CreatedAt:   time.Now(),
		VMConfig: VMConfiguration{
			VCPUs:       vmSpec.VCPUs,
			MemoryMB:    vmSpec.MemoryMB,
			KernelPath:  k.config.DefaultKernelPath,
			InitrdPath:  k.config.DefaultInitrdPath,
			MachineType: "q35",
			Hypervisor:  "qemu",
		},
		Resources: ResourceAllocation{
			CPUShares:   1024,
			MemoryLimit: vmSpec.MemoryMB * 1024 * 1024, // Convert MB to bytes
			PidsLimit:   1024,
			BlkioWeight: 500,
		},
		Status: ContainerStatus{
			State:       "created",
			LastUpdated: time.Now(),
		},
		VMStatus: VMStatus{
			State:       "created",
			LastUpdated: time.Now(),
		},
	}
	
	return kataContainer, nil
}

func (k *KataDriver) createContainerSpec(vmSpec vm.VMSpec) (*specs.Spec, error) {
	// Create OCI spec
	spec := oci.Spec{
		Version: "1.0.0",
		Process: &specs.Process{
			Terminal: false,
			Args:     vmSpec.Command,
			Env:      vmSpec.Environment,
			Cwd:      "/",
		},
		Root: &specs.Root{
			Path:     "rootfs",
			Readonly: false,
		},
		Hostname: vmSpec.Name,
	}
	
	// Set resource limits
	spec.Linux = &specs.Linux{
		Resources: &specs.LinuxResources{
			CPU: &specs.LinuxCPU{
				Shares: uint64Ptr(1024),
				Quota:  int64Ptr(100000 * int64(vmSpec.VCPUs)), // 100% per vCPU
				Period: uint64Ptr(100000),
			},
			Memory: &specs.LinuxMemory{
				Limit: int64Ptr(vmSpec.MemoryMB * 1024 * 1024),
			},
			Pids: &specs.LinuxPids{
				Limit: 1024,
			},
		},
		Namespaces: []specs.LinuxNamespace{
			{Type: specs.PIDNamespace},
			{Type: specs.NetworkNamespace},
			{Type: specs.IPCNamespace},
			{Type: specs.UTSNamespace},
			{Type: specs.MountNamespace},
		},
	}
	
	// Add security settings
	if k.config.EnableSeccomp {
		spec.Linux.Seccomp = &specs.LinuxSeccomp{
			DefaultAction: specs.ActErrno,
			Architectures: []specs.Arch{specs.ArchX86_64},
		}
	}
	
	return &spec, nil
}

func (k *KataDriver) verifyKataRuntime() error {
	ctx := context.Background()
	ctx = namespaces.WithNamespace(ctx, k.namespace)
	
	// Try to create a test container with Kata runtime
	testImage := "hello-world:latest"
	
	image, err := k.client.Pull(ctx, testImage, containerd.WithPullUnpack)
	if err != nil {
		return fmt.Errorf("failed to pull test image: %w", err)
	}
	
	spec, err := oci.GenerateSpec(ctx, k.client, &containerd.Container{})
	if err != nil {
		return fmt.Errorf("failed to generate test spec: %w", err)
	}
	
	testContainer, err := k.client.NewContainer(ctx, "kata-test", 
		containerd.WithImage(image),
		containerd.WithSpec(spec),
		containerd.WithRuntime(k.runtime, nil),
	)
	if err != nil {
		return fmt.Errorf("failed to create test container with kata runtime: %w", err)
	}
	
	// Clean up test container
	testContainer.Delete(ctx, containerd.WithSnapshotCleanup)
	
	log.Printf("Kata runtime %s verified successfully", k.runtime)
	return nil
}

func (k *KataDriver) containerToVM(kataContainer *KataContainer) *vm.VM {
	vmStatus := vm.VMStatusStopped
	if kataContainer.Status.State == "running" {
		vmStatus = vm.VMStatusRunning
	} else if kataContainer.Status.State == "paused" {
		vmStatus = vm.VMStatusPaused
	}
	
	return &vm.VM{
		ID:       kataContainer.ID,
		Name:     kataContainer.Name,
		Status:   vmStatus,
		VCPUs:    kataContainer.VMConfig.VCPUs,
		MemoryMB: kataContainer.VMConfig.MemoryMB,
		Labels:   kataContainer.Labels,
		CreatedAt: kataContainer.CreatedAt,
		UpdatedAt: kataContainer.Status.LastUpdated,
		Driver:    "kata",
	}
}

func (k *KataDriver) refreshContainerStatus(ctx context.Context, kataContainer *KataContainer) error {
	if kataContainer.Task == nil {
		return nil
	}
	
	ctx = namespaces.WithNamespace(ctx, k.namespace)
	
	// Get task status
	status, err := kataContainer.Task.Status(ctx)
	if err != nil {
		return fmt.Errorf("failed to get task status: %w", err)
	}
	
	// Update status
	kataContainer.Status.State = string(status.Status)
	kataContainer.Status.LastUpdated = time.Now()
	
	return nil
}

func (k *KataDriver) collectContainerMetrics(ctx context.Context, kataContainer *KataContainer) error {
	if kataContainer.Task == nil {
		return nil
	}
	
	ctx = namespaces.WithNamespace(ctx, k.namespace)
	
	// Get metrics from containerd
	metric, err := kataContainer.Task.Metrics(ctx)
	if err != nil {
		return fmt.Errorf("failed to get task metrics: %w", err)
	}
	
	// Parse metrics (simplified for demo)
	kataContainer.Metrics = ContainerMetrics{
		CPUUsage:    45.0, // Would parse from metric.Data
		MemoryUsage: kataContainer.Resources.MemoryLimit / 2, // Placeholder
		MemoryLimit: kataContainer.Resources.MemoryLimit,
		Timestamp:   time.Now(),
	}
	
	return nil
}

func (k *KataDriver) checkpointRestoreMigration(ctx context.Context, vmID string, targetHost string) error {
	// Implementation for checkpoint/restore migration
	// This would involve:
	// 1. Checkpoint container state
	// 2. Transfer checkpoint data to target
	// 3. Restore container on target
	// 4. Update routing/networking
	
	return fmt.Errorf("checkpoint/restore migration not yet implemented")
}

// Utility functions
func uint64Ptr(v uint64) *uint64 {
	return &v
}

func int64Ptr(v int64) *int64 {
	return &v
}

// Close cleans up the Kata driver
func (k *KataDriver) Close() error {
	if k.client != nil {
		return k.client.Close()
	}
	return nil
}