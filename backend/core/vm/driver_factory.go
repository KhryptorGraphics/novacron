package vm

import (
	"fmt"
	"log"
	"os"
	// "path/filepath" // Currently unused
)

// VMDriverConfig contains configuration for VM drivers
type VMDriverConfig struct {
	// Node ID
	NodeID string
	
	// Container driver config
	DockerPath string
	
	// Containerd driver config
	ContainerdAddress string
	ContainerdNamespace string
	
	// KVM driver config
	QEMUBinaryPath string 
	VMBasePath     string
	
	// Process driver config
	ProcessBasePath string
}

// DefaultVMDriverConfig returns a default VM driver configuration
func DefaultVMDriverConfig(nodeID string) VMDriverConfig {
	return VMDriverConfig{
		NodeID:               nodeID,
		DockerPath:           "docker",
		ContainerdAddress:    "/run/containerd/containerd.sock",
		ContainerdNamespace:  "novacron",
		QEMUBinaryPath:       "qemu-system-x86_64",
		VMBasePath:           "/var/lib/novacron/vms",
		ProcessBasePath:      "/var/lib/novacron/processes",
	}
}

// NewVMDriverFactory creates a new VM driver factory
func NewVMDriverFactory(config VMDriverConfig) VMDriverFactory {
	log.Printf("Creating VM driver factory with node ID %s", config.NodeID)
	
	// Create a cache of initialized drivers
	drivers := make(map[VMType]VMDriver)
	
	// Return the factory function
	return func(vmConfig VMConfig) (VMDriver, error) {
		// Determine VM type from config - for now use a default
		vmType := VMTypeKVM // Default type
		if vmConfig.Tags != nil {
			if t, ok := vmConfig.Tags["vm_type"]; ok {
				vmType = VMType(t)
			}
		}
		// Check if we have already initialized this driver type
		if driver, exists := drivers[vmType]; exists {
			return driver, nil
		}
		
		// Initialize the appropriate driver based on type
		var driver VMDriver
		var err error
		
		switch vmType {
		case VMTypeContainer:
			log.Printf("Initializing container driver")
			driverConfig := map[string]interface{}{
				"node_id": config.NodeID,
			}
			driver, err = NewContainerDriver(driverConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to initialize container driver: %w", err)
			}
			
		case VMTypeContainerd:
			log.Printf("Initializing containerd driver with namespace %s", config.ContainerdNamespace)
			driverConfig := map[string]interface{}{
				"node_id":   config.NodeID,
				"address":   config.ContainerdAddress,
				"namespace": config.ContainerdNamespace,
			}
			driver, err = NewContainerdDriver(driverConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to initialize containerd driver: %w", err)
			}
			
		case VMTypeKVM:
			log.Printf("Initializing KVM driver with base path %s", config.VMBasePath)
			// Create the base path if it doesn't exist
			if err := makeDirectoryIfNotExists(config.VMBasePath); err != nil {
				log.Printf("Warning: Failed to create VM base path %s: %v", config.VMBasePath, err)
			}
			driverConfig := map[string]interface{}{
				"node_id":    config.NodeID,
				"qemu_path":  config.QEMUBinaryPath,
				"vm_path":    config.VMBasePath,
			}
			driver, err = NewKVMDriver(driverConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to initialize KVM driver: %w", err)
			}
			
		case VMTypeProcess:
			log.Printf("Initializing process driver with base path %s", config.ProcessBasePath)
			// Create the base path if it doesn't exist
			if err := makeDirectoryIfNotExists(config.ProcessBasePath); err != nil {
				log.Printf("Warning: Failed to create process base path %s: %v", config.ProcessBasePath, err)
			}
			// Process driver doesn't exist yet, return error for now
			return nil, fmt.Errorf("process driver not implemented yet")
			
		default:
			return nil, fmt.Errorf("unsupported VM type: %s", vmType)
		}
		
		// Cache the driver for future use
		drivers[vmType] = driver
		
		return driver, nil
	}
}

// Helper to create a directory if it doesn't exist
func makeDirectoryIfNotExists(path string) error {
	// Check if directory exists
	if _, err := os.Stat(path); err == nil {
		return nil // Directory exists
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("failed to check directory: %w", err)
	}
	
	// Create all directories in the path
	if err := os.MkdirAll(path, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}
	
	return nil
}

// VMDriverManager manages VM drivers
type VMDriverManager struct {
	config  VMDriverConfig
	factory VMDriverFactory
	drivers map[VMType]VMDriver
}

// NewVMDriverManager creates a new VM driver manager
func NewVMDriverManager(config VMDriverConfig) *VMDriverManager {
	return &VMDriverManager{
		config:  config,
		factory: NewVMDriverFactory(config),
		drivers: make(map[VMType]VMDriver),
	}
}

// GetDriver gets a driver for a VM type
func (m *VMDriverManager) GetDriver(vmConfig VMConfig) (VMDriver, error) {
	// Determine VM type from config - for now use a default
	vmType := VMTypeKVM // Default type
	if vmConfig.Tags != nil {
		if t, ok := vmConfig.Tags["vm_type"]; ok {
			vmType = VMType(t)
		}
	}
	
	// Check if we have already initialized this driver
	if driver, exists := m.drivers[vmType]; exists {
		return driver, nil
	}
	
	// Initialize the driver
	driver, err := m.factory(vmConfig)
	if err != nil {
		return nil, err
	}
	
	// Cache the driver
	m.drivers[vmType] = driver
	
	return driver, nil
}

// Close closes all drivers
func (m *VMDriverManager) Close() {
	for vmType, driver := range m.drivers {
		if closer, ok := driver.(interface{ Close() error }); ok {
			if err := closer.Close(); err != nil {
				log.Printf("Error closing driver %s: %v", vmType, err)
			}
		}
	}
}

// ListSupportedTypes returns a list of supported VM types
func (m *VMDriverManager) ListSupportedTypes() []VMType {
	return []VMType{
		VMTypeContainer,
		VMTypeContainerd,
		VMTypeKVM,
		VMTypeProcess,
	}
}
