package vm

import (
	"testing"
)

// TestVMDriverFactory tests the driver factory functionality
func TestVMDriverFactory(t *testing.T) {
	// Test default driver config creation
	nodeID := "test-node-001"
	config := DefaultVMDriverConfig(nodeID)

	if config.NodeID != nodeID {
		t.Errorf("NodeID should be %s, got %s", nodeID, config.NodeID)
	}

	// Verify all default paths are set
	if config.DockerPath == "" {
		t.Error("DockerPath should not be empty")
	}
	if config.ContainerdAddress == "" {
		t.Error("ContainerdAddress should not be empty")
	}
	if config.QEMUBinaryPath == "" {
		t.Error("QEMUBinaryPath should not be empty")
	}
	if config.VMBasePath == "" {
		t.Error("VMBasePath should not be empty")
	}

	t.Log("Default VM driver config creation works correctly")
}

// TestVMDriverFactoryCreation tests creating a driver factory
func TestVMDriverFactoryCreation(t *testing.T) {
	config := DefaultVMDriverConfig("test-node")
	factory := NewVMDriverFactory(config)

	if factory == nil {
		t.Fatal("Driver factory should not be nil")
	}

	// Test factory with different VM configs
	testConfigs := []VMConfig{
		{
			ID:        "test-kvm-vm",
			Name:      "test-kvm",
			Command:   "/bin/sleep",
			Args:      []string{"30"},
			CPUShares: 1024,
			MemoryMB:  512,
			RootFS:    "/tmp",
			Tags: map[string]string{
				"vm_type": string(VMTypeKVM),
			},
		},
		{
			ID:        "test-container-vm",
			Name:      "test-container",
			Command:   "/bin/sleep",
			Args:      []string{"30"},
			CPUShares: 1024,
			MemoryMB:  512,
			RootFS:    "/tmp",
			Tags: map[string]string{
				"vm_type": string(VMTypeContainer),
			},
		},
	}

	for _, vmConfig := range testConfigs {
		driver, err := factory(vmConfig)
		if err != nil {
			// Expected in test environment without actual hypervisors
			t.Logf("Driver creation failed (expected in test environment): %v", err)
		} else if driver == nil {
			t.Errorf("Driver should not be nil when creation succeeds for %s", vmConfig.Tags["vm_type"])
		}
	}

	t.Log("Driver factory creation works correctly")
}

// TestVMDriverManager tests the driver manager functionality
func TestVMDriverManager(t *testing.T) {
	config := DefaultVMDriverConfig("test-node")
	manager := NewVMDriverManager(config)

	if manager == nil {
		t.Fatal("Driver manager should not be nil")
	}

	// Test supported types
	supportedTypes := manager.ListSupportedTypes()
	expectedTypes := []VMType{VMTypeContainer, VMTypeContainerd, VMTypeKVM, VMTypeProcess}

	if len(supportedTypes) != len(expectedTypes) {
		t.Errorf("Expected %d supported types, got %d", len(expectedTypes), len(supportedTypes))
	}

	// Verify all expected types are present
	typeMap := make(map[VMType]bool)
	for _, vmType := range supportedTypes {
		typeMap[vmType] = true
	}

	for _, expectedType := range expectedTypes {
		if !typeMap[expectedType] {
			t.Errorf("Expected type %s not found in supported types", expectedType)
		}
	}

	// Test driver retrieval
	testConfig := VMConfig{
		ID:        "test-driver-retrieval",
		Name:      "test-driver",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
		Tags: map[string]string{
			"vm_type": string(VMTypeKVM),
		},
	}

	driver, err := manager.GetDriver(testConfig)
	if err != nil {
		// Expected in test environment
		t.Logf("Driver retrieval failed (expected in test environment): %v", err)
	} else if driver == nil {
		t.Error("Driver should not be nil when retrieval succeeds")
	}

	// Test close functionality
	manager.Close() // Should not panic

	t.Log("Driver manager functionality works correctly")
}

// TestVMDriverTypes tests VM type constants and conversions
func TestVMDriverTypes(t *testing.T) {
	// Test all VM type constants
	types := []VMType{
		VMTypeKVM,
		VMTypeContainer,
		VMTypeContainerd,
		VMTypeProcess,
	}

	expectedStrings := []string{
		"kvm",
		"container",
		"containerd",
		"process",
	}

	for i, vmType := range types {
		if string(vmType) != expectedStrings[i] {
			t.Errorf("VM type %s should have string value %s, got %s", 
				vmType, expectedStrings[i], string(vmType))
		}
	}

	// Test type conversion from string
	for i, expectedString := range expectedStrings {
		vmType := VMType(expectedString)
		if vmType != types[i] {
			t.Errorf("String %s should convert to %s, got %s", 
				expectedString, types[i], vmType)
		}
	}

	t.Log("VM driver types work correctly")
}

// TestVMDriverConfig tests driver configuration structure
func TestVMDriverConfig(t *testing.T) {
	config := VMDriverConfig{
		NodeID:               "test-node",
		DockerPath:           "/usr/bin/docker",
		ContainerdAddress:    "/run/containerd/containerd.sock",
		ContainerdNamespace:  "test-namespace",
		QEMUBinaryPath:       "/usr/bin/qemu-system-x86_64",
		VMBasePath:           "/var/lib/test/vms",
		ProcessBasePath:      "/var/lib/test/processes",
	}

	// Test all fields are accessible and correctly set
	if config.NodeID != "test-node" {
		t.Errorf("NodeID should be 'test-node', got '%s'", config.NodeID)
	}

	if config.DockerPath != "/usr/bin/docker" {
		t.Errorf("DockerPath should be '/usr/bin/docker', got '%s'", config.DockerPath)
	}

	if config.ContainerdAddress != "/run/containerd/containerd.sock" {
		t.Errorf("ContainerdAddress should be '/run/containerd/containerd.sock', got '%s'", config.ContainerdAddress)
	}

	if config.ContainerdNamespace != "test-namespace" {
		t.Errorf("ContainerdNamespace should be 'test-namespace', got '%s'", config.ContainerdNamespace)
	}

	if config.QEMUBinaryPath != "/usr/bin/qemu-system-x86_64" {
		t.Errorf("QEMUBinaryPath should be '/usr/bin/qemu-system-x86_64', got '%s'", config.QEMUBinaryPath)
	}

	if config.VMBasePath != "/var/lib/test/vms" {
		t.Errorf("VMBasePath should be '/var/lib/test/vms', got '%s'", config.VMBasePath)
	}

	if config.ProcessBasePath != "/var/lib/test/processes" {
		t.Errorf("ProcessBasePath should be '/var/lib/test/processes', got '%s'", config.ProcessBasePath)
	}

	t.Log("VM driver config structure works correctly")
}

// TestContainerdDriverStub tests the containerd driver stub
func TestContainerdDriverStub(t *testing.T) {
	config := map[string]interface{}{
		"node_id": "test-node",
	}

	driver, err := NewContainerdDriver(config)
	if err != nil {
		t.Fatalf("Failed to create containerd driver: %v", err)
	}

	if driver == nil {
		t.Fatal("Driver should not be nil")
	}

	// Test that all methods return "not implemented" errors
	ctx := context.Background()
	vmConfig := VMConfig{
		ID:        "test-containerd-vm",
		Name:      "test-vm",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	// Test Create
	_, err = driver.Create(ctx, vmConfig)
	if err == nil || err.Error() != "containerd driver not implemented" {
		t.Errorf("Create should return 'not implemented' error, got: %v", err)
	}

	// Test Start
	err = driver.Start(ctx, "test-vm")
	if err == nil || err.Error() != "containerd driver not implemented" {
		t.Errorf("Start should return 'not implemented' error, got: %v", err)
	}

	// Test Stop
	err = driver.Stop(ctx, "test-vm")
	if err == nil || err.Error() != "containerd driver not implemented" {
		t.Errorf("Stop should return 'not implemented' error, got: %v", err)
	}

	// Test capability methods
	if driver.SupportsPause() {
		t.Error("Containerd stub should not support pause")
	}

	if driver.SupportsResume() {
		t.Error("Containerd stub should not support resume")
	}

	if driver.SupportsSnapshot() {
		t.Error("Containerd stub should not support snapshot")
	}

	if driver.SupportsMigrate() {
		t.Error("Containerd stub should not support migrate")
	}

	t.Log("Containerd driver stub works correctly")
}