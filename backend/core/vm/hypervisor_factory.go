package vm

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// HypervisorFactory provides a unified interface for creating hypervisor drivers
type HypervisorFactory struct {
	config              *HypervisorConfig
	drivers             map[VMType]VMDriver
	driverLock          sync.RWMutex
	capabilityDetector  *CapabilityDetector
}

// HypervisorConfig contains configuration for all hypervisor types
type HypervisorConfig struct {
	// Node configuration
	NodeID   string
	NodeName string

	// KVM/QEMU configuration
	KVM KVMConfig

	// VMware vSphere configuration
	VMware VMwareConfig

	// Hyper-V configuration
	HyperV HyperVConfig

	// XenServer configuration
	Xen XenConfig

	// Proxmox VE configuration
	Proxmox ProxmoxConfig

	// Container configurations
	Container   ContainerConfig
	Containerd  ContainerdConfig
	
	// Auto-detection settings
	AutoDetect bool
	Priorities []VMType // Preferred order for auto-detection
}

// KVMConfig contains KVM/QEMU specific configuration
type KVMConfig struct {
	Enabled        bool
	LibvirtURI     string
	QEMUBinaryPath string
	VMBasePath     string
	UseLibvirt     bool
}

// VMwareConfig contains VMware vSphere specific configuration
type VMwareConfig struct {
	Enabled    bool
	URL        string
	Username   string
	Password   string
	Insecure   bool
	Datacenter string
}

// HyperVConfig contains Hyper-V specific configuration
type HyperVConfig struct {
	Enabled  bool
	Hostname string
	Username string
	Password string
	Domain   string
	UseSSL   bool
	Port     int
}

// XenConfig contains XenServer specific configuration
type XenConfig struct {
	Enabled  bool
	URL      string
	Username string
	Password string
	Pool     string
}

// ProxmoxConfig contains Proxmox VE specific configuration
type ProxmoxConfig struct {
	Enabled  bool
	URL      string
	Username string
	Password string
	Realm    string
	Node     string
	Insecure bool
}

// ContainerConfig contains container runtime configuration
type ContainerConfig struct {
	Enabled    bool
	DockerPath string
}

// ContainerdConfig contains containerd specific configuration
type ContainerdConfig struct {
	Enabled   bool
	Address   string
	Namespace string
}

// CapabilityDetector detects available hypervisors and their capabilities
type CapabilityDetector struct {
	cache      map[VMType]*HypervisorCapabilities
	cacheLock  sync.RWMutex
	lastUpdate time.Time
	cacheValid time.Duration
}

// HypervisorSupport represents support status for a hypervisor
type HypervisorSupport struct {
	Type        VMType                   `json:"type"`
	Available   bool                     `json:"available"`
	Enabled     bool                     `json:"enabled"`
	Version     string                   `json:"version,omitempty"`
	Error       string                   `json:"error,omitempty"`
	Capabilities *HypervisorCapabilities `json:"capabilities,omitempty"`
	Priority    int                      `json:"priority"`
}

// NewHypervisorFactory creates a new hypervisor factory
func NewHypervisorFactory(config *HypervisorConfig) (*HypervisorFactory, error) {
	if config == nil {
		config = DefaultHypervisorConfig()
	}

	factory := &HypervisorFactory{
		config:  config,
		drivers: make(map[VMType]VMDriver),
		capabilityDetector: &CapabilityDetector{
			cache:      make(map[VMType]*HypervisorCapabilities),
			cacheValid: 5 * time.Minute,
		},
	}

	// Perform auto-detection if enabled
	if config.AutoDetect {
		if err := factory.detectAvailableHypervisors(); err != nil {
			log.Printf("Warning: Auto-detection failed: %v", err)
		}
	}

	return factory, nil
}

// DefaultHypervisorConfig returns a default hypervisor configuration
func DefaultHypervisorConfig() *HypervisorConfig {
	return &HypervisorConfig{
		NodeID:   "default-node",
		NodeName: "novacron-node",
		AutoDetect: true,
		Priorities: []VMType{
			VMTypeKVM,        // Prefer KVM on Linux
			VMTypeProxmox,    // Proxmox VE
			VMTypeVMware,     // VMware vSphere
			VMTypeXen,        // XenServer/XCP-ng
			VMTypeHyperV,     // Hyper-V
			VMTypeContainerd, // Containerd
			VMTypeContainer,  // Docker
		},
		KVM: KVMConfig{
			Enabled:        true,
			LibvirtURI:     "qemu:///system",
			QEMUBinaryPath: "/usr/bin/qemu-system-x86_64",
			VMBasePath:     "/var/lib/novacron/vms",
			UseLibvirt:     true,
		},
		VMware: VMwareConfig{
			Enabled:  false,
			Insecure: true,
		},
		HyperV: HyperVConfig{
			Enabled:  false,
			Hostname: "localhost",
			Port:     5985,
			UseSSL:   false,
		},
		Xen: XenConfig{
			Enabled: false,
		},
		Proxmox: ProxmoxConfig{
			Enabled:  false,
			Realm:    "pam",
			Insecure: true,
		},
		Container: ContainerConfig{
			Enabled:    true,
			DockerPath: "docker",
		},
		Containerd: ContainerdConfig{
			Enabled:   true,
			Address:   "/run/containerd/containerd.sock",
			Namespace: "novacron",
		},
	}
}

// detectAvailableHypervisors detects which hypervisors are available on the system
func (f *HypervisorFactory) detectAvailableHypervisors() error {
	log.Printf("Detecting available hypervisors...")

	detectionResults := make(map[VMType]*HypervisorSupport)
	
	// Detect in priority order
	for i, vmType := range f.config.Priorities {
		support := &HypervisorSupport{
			Type:     vmType,
			Priority: i + 1,
		}

		switch vmType {
		case VMTypeKVM:
			support.Available, support.Error = f.detectKVM()
			support.Enabled = f.config.KVM.Enabled && support.Available
		case VMTypeVMware, VMTypeVSphere:
			support.Available, support.Error = f.detectVMware()
			support.Enabled = f.config.VMware.Enabled && support.Available
		case VMTypeHyperV:
			support.Available, support.Error = f.detectHyperV()
			support.Enabled = f.config.HyperV.Enabled && support.Available
		case VMTypeXen, VMTypeXenServer:
			support.Available, support.Error = f.detectXen()
			support.Enabled = f.config.Xen.Enabled && support.Available
		case VMTypeProxmox, VMTypeProxmoxVE:
			support.Available, support.Error = f.detectProxmox()
			support.Enabled = f.config.Proxmox.Enabled && support.Available
		case VMTypeContainer:
			support.Available, support.Error = f.detectDocker()
			support.Enabled = f.config.Container.Enabled && support.Available
		case VMTypeContainerd:
			support.Available, support.Error = f.detectContainerd()
			support.Enabled = f.config.Containerd.Enabled && support.Available
		}

		detectionResults[vmType] = support
		
		if support.Available {
			log.Printf("Detected hypervisor: %s (enabled: %v)", vmType, support.Enabled)
		} else if support.Error != "" {
			log.Printf("Hypervisor %s not available: %s", vmType, support.Error)
		}
	}

	return nil
}

// detectKVM detects KVM availability
func (f *HypervisorFactory) detectKVM() (bool, string) {
	// Check for KVM kernel module
	if _, err := os.Stat("/dev/kvm"); err != nil {
		return false, "KVM device not available"
	}

	// Check for QEMU binary
	qemuPath := f.config.KVM.QEMUBinaryPath
	if qemuPath == "" {
		qemuPath = "/usr/bin/qemu-system-x86_64"
	}
	
	if _, err := os.Stat(qemuPath); err != nil {
		return false, "QEMU binary not found"
	}

	// Check for libvirt if enabled
	if f.config.KVM.UseLibvirt {
		if _, err := os.Stat("/var/run/libvirt/libvirt-sock"); err != nil {
			log.Printf("Libvirt not available, will use direct QEMU")
			f.config.KVM.UseLibvirt = false
		}
	}

	return true, ""
}

// detectVMware detects VMware vSphere availability
func (f *HypervisorFactory) detectVMware() (bool, string) {
	if f.config.VMware.URL == "" {
		return false, "vSphere URL not configured"
	}
	if f.config.VMware.Username == "" || f.config.VMware.Password == "" {
		return false, "vSphere credentials not configured"
	}
	return true, ""
}

// detectHyperV detects Hyper-V availability
func (f *HypervisorFactory) detectHyperV() (bool, string) {
	// On Windows, check for Hyper-V PowerShell module
	// On Linux, require explicit configuration
	if f.config.HyperV.Hostname == "" {
		return false, "Hyper-V hostname not configured"
	}
	return true, ""
}

// detectXen detects XenServer availability
func (f *HypervisorFactory) detectXen() (bool, string) {
	if f.config.Xen.URL == "" {
		return false, "XenServer URL not configured"
	}
	if f.config.Xen.Username == "" || f.config.Xen.Password == "" {
		return false, "XenServer credentials not configured"
	}
	return true, ""
}

// detectProxmox detects Proxmox VE availability
func (f *HypervisorFactory) detectProxmox() (bool, string) {
	if f.config.Proxmox.URL == "" {
		return false, "Proxmox VE URL not configured"
	}
	if f.config.Proxmox.Username == "" || f.config.Proxmox.Password == "" {
		return false, "Proxmox VE credentials not configured"
	}
	return true, ""
}

// detectDocker detects Docker availability
func (f *HypervisorFactory) detectDocker() (bool, string) {
	dockerPath := f.config.Container.DockerPath
	if dockerPath == "" {
		dockerPath = "docker"
	}
	
	if _, err := exec.LookPath(dockerPath); err != nil {
		return false, "Docker binary not found in PATH"
	}
	
	// Check if Docker daemon is running
	cmd := exec.Command(dockerPath, "version")
	if err := cmd.Run(); err != nil {
		return false, "Docker daemon not accessible"
	}
	
	return true, ""
}

// detectContainerd detects containerd availability
func (f *HypervisorFactory) detectContainerd() (bool, string) {
	sockPath := f.config.Containerd.Address
	if sockPath == "" {
		sockPath = "/run/containerd/containerd.sock"
	}
	
	if _, err := os.Stat(sockPath); err != nil {
		return false, "containerd socket not available"
	}
	
	return true, ""
}

// GetDriver returns a driver for the specified VM type
func (f *HypervisorFactory) GetDriver(ctx context.Context, vmType VMType) (VMDriver, error) {
	f.driverLock.Lock()
	defer f.driverLock.Unlock()

	// Return cached driver if available
	if driver, exists := f.drivers[vmType]; exists {
		return driver, nil
	}

	// Create new driver
	driver, err := f.createDriver(ctx, vmType)
	if err != nil {
		return nil, fmt.Errorf("failed to create driver for %s: %w", vmType, err)
	}

	// Cache the driver
	f.drivers[vmType] = driver

	return driver, nil
}

// createDriver creates a new driver instance
func (f *HypervisorFactory) createDriver(ctx context.Context, vmType VMType) (VMDriver, error) {
	switch vmType {
	case VMTypeKVM:
		return f.createKVMDriver(ctx)
	case VMTypeVMware, VMTypeVSphere:
		return f.createVMwareDriver(ctx)
	case VMTypeHyperV:
		return f.createHyperVDriver(ctx)
	case VMTypeXen, VMTypeXenServer:
		return f.createXenDriver(ctx)
	case VMTypeProxmox, VMTypeProxmoxVE:
		return f.createProxmoxDriver(ctx)
	case VMTypeContainer:
		return f.createContainerDriver(ctx)
	case VMTypeContainerd:
		return f.createContainerdDriver(ctx)
	default:
		return nil, fmt.Errorf("unsupported VM type: %s", vmType)
	}
}

// createKVMDriver creates a KVM driver
func (f *HypervisorFactory) createKVMDriver(ctx context.Context) (VMDriver, error) {
	config := f.config.KVM
	
	if config.UseLibvirt {
		// Try libvirt driver first
		// Note: This would use the libvirt driver we created
		// For now, return error as we'd need to import it properly
		return nil, fmt.Errorf("libvirt driver integration pending")
	}
	
	// Fallback to enhanced KVM driver
	driverConfig := map[string]interface{}{
		"node_id":   f.config.NodeID,
		"qemu_path": config.QEMUBinaryPath,
		"vm_path":   config.VMBasePath,
	}
	
	return NewKVMDriver(driverConfig)
}

// createVMwareDriver creates a VMware vSphere driver
func (f *HypervisorFactory) createVMwareDriver(ctx context.Context) (VMDriver, error) {
	// Note: This would use the vSphere driver we created
	return nil, fmt.Errorf("vSphere driver integration pending")
}

// createHyperVDriver creates a Hyper-V driver
func (f *HypervisorFactory) createHyperVDriver(ctx context.Context) (VMDriver, error) {
	// Note: This would use the Hyper-V driver we created
	return nil, fmt.Errorf("Hyper-V driver integration pending")
}

// createXenDriver creates a XenServer driver
func (f *HypervisorFactory) createXenDriver(ctx context.Context) (VMDriver, error) {
	// Note: This would use the XenServer driver we created
	return nil, fmt.Errorf("XenServer driver integration pending")
}

// createProxmoxDriver creates a Proxmox VE driver
func (f *HypervisorFactory) createProxmoxDriver(ctx context.Context) (VMDriver, error) {
	// Note: This would use the Proxmox driver we created
	return nil, fmt.Errorf("Proxmox VE driver integration pending")
}

// createContainerDriver creates a container (Docker) driver
func (f *HypervisorFactory) createContainerDriver(ctx context.Context) (VMDriver, error) {
	config := f.config.Container
	
	driverConfig := map[string]interface{}{
		"node_id":     f.config.NodeID,
		"docker_path": config.DockerPath,
	}
	
	return NewContainerDriver(driverConfig)
}

// createContainerdDriver creates a containerd driver
func (f *HypervisorFactory) createContainerdDriver(ctx context.Context) (VMDriver, error) {
	config := f.config.Containerd
	
	driverConfig := map[string]interface{}{
		"node_id":   f.config.NodeID,
		"address":   config.Address,
		"namespace": config.Namespace,
	}
	
	return NewContainerdDriver(driverConfig)
}

// GetBestDriver returns the best available driver based on priorities
func (f *HypervisorFactory) GetBestDriver(ctx context.Context) (VMDriver, VMType, error) {
	for _, vmType := range f.config.Priorities {
		if driver, err := f.GetDriver(ctx, vmType); err == nil {
			return driver, vmType, nil
		}
	}
	
	return nil, "", fmt.Errorf("no suitable hypervisor found")
}

// GetCapabilities returns capabilities for a specific hypervisor
func (f *HypervisorFactory) GetCapabilities(ctx context.Context, vmType VMType) (*HypervisorCapabilities, error) {
	f.capabilityDetector.cacheLock.RLock()
	
	// Check cache validity
	if time.Since(f.capabilityDetector.lastUpdate) < f.capabilityDetector.cacheValid {
		if caps, exists := f.capabilityDetector.cache[vmType]; exists {
			f.capabilityDetector.cacheLock.RUnlock()
			return caps, nil
		}
	}
	f.capabilityDetector.cacheLock.RUnlock()

	// Get driver and query capabilities
	driver, err := f.GetDriver(ctx, vmType)
	if err != nil {
		return nil, fmt.Errorf("failed to get driver: %w", err)
	}

	// Check if driver supports capability reporting
	if capDriver, ok := driver.(interface {
		GetCapabilities(context.Context) (*HypervisorCapabilities, error)
	}); ok {
		caps, err := capDriver.GetCapabilities(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to get capabilities: %w", err)
		}

		// Cache the result
		f.capabilityDetector.cacheLock.Lock()
		f.capabilityDetector.cache[vmType] = caps
		f.capabilityDetector.lastUpdate = time.Now()
		f.capabilityDetector.cacheLock.Unlock()

		return caps, nil
	}

	return nil, fmt.Errorf("driver does not support capability reporting")
}

// GetAllCapabilities returns capabilities for all available hypervisors
func (f *HypervisorFactory) GetAllCapabilities(ctx context.Context) (map[VMType]*HypervisorCapabilities, error) {
	capabilities := make(map[VMType]*HypervisorCapabilities)

	for _, vmType := range f.config.Priorities {
		if caps, err := f.GetCapabilities(ctx, vmType); err == nil {
			capabilities[vmType] = caps
		}
	}

	return capabilities, nil
}

// GetSupportedTypes returns all supported VM types
func (f *HypervisorFactory) GetSupportedTypes() []VMType {
	return []VMType{
		VMTypeKVM,
		VMTypeVMware,
		VMTypeVSphere,
		VMTypeHyperV,
		VMTypeXen,
		VMTypeXenServer,
		VMTypeProxmox,
		VMTypeProxmoxVE,
		VMTypeContainer,
		VMTypeContainerd,
		VMTypeKataContainers,
		VMTypeProcess,
	}
}

// GetDetectionResults returns the results of hypervisor detection
func (f *HypervisorFactory) GetDetectionResults(ctx context.Context) (map[VMType]*HypervisorSupport, error) {
	results := make(map[VMType]*HypervisorSupport)

	for i, vmType := range f.config.Priorities {
		support := &HypervisorSupport{
			Type:     vmType,
			Priority: i + 1,
		}

		// Try to get capabilities to determine availability
		if caps, err := f.GetCapabilities(ctx, vmType); err == nil {
			support.Available = true
			support.Capabilities = caps
			support.Version = caps.Version
			
			// Check if enabled in configuration
			switch vmType {
			case VMTypeKVM:
				support.Enabled = f.config.KVM.Enabled
			case VMTypeVMware, VMTypeVSphere:
				support.Enabled = f.config.VMware.Enabled
			case VMTypeHyperV:
				support.Enabled = f.config.HyperV.Enabled
			case VMTypeXen, VMTypeXenServer:
				support.Enabled = f.config.Xen.Enabled
			case VMTypeProxmox, VMTypeProxmoxVE:
				support.Enabled = f.config.Proxmox.Enabled
			case VMTypeContainer:
				support.Enabled = f.config.Container.Enabled
			case VMTypeContainerd:
				support.Enabled = f.config.Containerd.Enabled
			}
		} else {
			support.Available = false
			support.Error = err.Error()
		}

		results[vmType] = support
	}

	return results, nil
}

// RefreshCapabilities refreshes the capability cache
func (f *HypervisorFactory) RefreshCapabilities(ctx context.Context) error {
	f.capabilityDetector.cacheLock.Lock()
	defer f.capabilityDetector.cacheLock.Unlock()

	// Clear cache
	f.capabilityDetector.cache = make(map[VMType]*HypervisorCapabilities)
	f.capabilityDetector.lastUpdate = time.Time{}

	// Re-detect available hypervisors
	return f.detectAvailableHypervisors()
}

// Close closes all initialized drivers
func (f *HypervisorFactory) Close() error {
	f.driverLock.Lock()
	defer f.driverLock.Unlock()

	var errors []string

	for vmType, driver := range f.drivers {
		if closer, ok := driver.(interface{ Close() error }); ok {
			if err := closer.Close(); err != nil {
				errors = append(errors, fmt.Sprintf("%s: %v", vmType, err))
			}
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("errors closing drivers: %s", strings.Join(errors, ", "))
	}

	return nil
}

// ValidateConfiguration validates the hypervisor configuration
func (f *HypervisorFactory) ValidateConfiguration() error {
	var errors []string

	// Validate KVM configuration
	if f.config.KVM.Enabled {
		if f.config.KVM.QEMUBinaryPath == "" {
			errors = append(errors, "KVM: QEMU binary path not specified")
		}
		if f.config.KVM.VMBasePath == "" {
			errors = append(errors, "KVM: VM base path not specified")
		}
	}

	// Validate VMware configuration
	if f.config.VMware.Enabled {
		if f.config.VMware.URL == "" {
			errors = append(errors, "VMware: URL not specified")
		}
		if f.config.VMware.Username == "" {
			errors = append(errors, "VMware: username not specified")
		}
		if f.config.VMware.Password == "" {
			errors = append(errors, "VMware: password not specified")
		}
	}

	// Validate Hyper-V configuration
	if f.config.HyperV.Enabled {
		if f.config.HyperV.Hostname == "" {
			errors = append(errors, "Hyper-V: hostname not specified")
		}
	}

	// Validate XenServer configuration
	if f.config.Xen.Enabled {
		if f.config.Xen.URL == "" {
			errors = append(errors, "XenServer: URL not specified")
		}
		if f.config.Xen.Username == "" {
			errors = append(errors, "XenServer: username not specified")
		}
		if f.config.Xen.Password == "" {
			errors = append(errors, "XenServer: password not specified")
		}
	}

	// Validate Proxmox configuration
	if f.config.Proxmox.Enabled {
		if f.config.Proxmox.URL == "" {
			errors = append(errors, "Proxmox: URL not specified")
		}
		if f.config.Proxmox.Username == "" {
			errors = append(errors, "Proxmox: username not specified")
		}
		if f.config.Proxmox.Password == "" {
			errors = append(errors, "Proxmox: password not specified")
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("configuration validation failed: %s", strings.Join(errors, "; "))
	}

	return nil
}