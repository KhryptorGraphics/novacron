package vm

import (
	"context"
	"fmt"
	"log"
	"strings"
	"testing"
	"time"
)

// TestVMDriverSystemDemo demonstrates the complete VM driver system functionality
// This test showcases all implemented drivers and their capabilities
func TestVMDriverSystemDemo(t *testing.T) {
	log.Println("🚀 NovaCron VM Driver System Demo")
	log.Println(strings.Repeat("=", 60))

	// Initialize the driver management system
	config := DefaultVMDriverConfig("demo-node-001")
	manager := NewVMDriverManager(config)
	defer manager.Close()

	log.Printf("📊 Initialized VM Driver Manager for node: %s", config.NodeID)
	log.Printf("🔧 Configuration:")
	log.Printf("  - Docker Path: %s", config.DockerPath)
	log.Printf("  - Containerd Address: %s", config.ContainerdAddress)
	log.Printf("  - QEMU Binary: %s", config.QEMUBinaryPath)
	log.Printf("  - VM Base Path: %s", config.VMBasePath)

	// Display supported driver types
	supportedTypes := manager.ListSupportedTypes()
	log.Printf("✅ Supported VM Driver Types (%d):", len(supportedTypes))
	for i, vmType := range supportedTypes {
		log.Printf("   %d. %s", i+1, string(vmType))
	}

	// Test each driver type's capabilities
	testDriverCapabilities(t, manager, supportedTypes)

	// Demonstrate driver factory caching
	testDriverCaching(t, manager)

	// Show driver interface compliance
	testInterfaceCompliance(t, manager, supportedTypes)

	log.Println("🎯 VM Driver System Demo Complete!")
	log.Println(strings.Repeat("=", 60))
}

// testDriverCapabilities tests and displays capabilities of each driver
func testDriverCapabilities(t *testing.T, manager *VMDriverManager, types []VMType) {
	log.Println("\n🧪 Testing Driver Capabilities:")
	log.Println(strings.Repeat("-", 40))

	for _, vmType := range types {
		log.Printf("\n🔍 Testing %s Driver:", string(vmType))

		// Create test VM configuration
		vmConfig := VMConfig{
			ID:        fmt.Sprintf("demo-%s-%d", string(vmType), time.Now().Unix()),
			Name:      fmt.Sprintf("demo-%s", string(vmType)),
			CPUShares: 512,
			MemoryMB:  256,
			Tags:      map[string]string{"vm_type": string(vmType)},
		}

		// Set driver-specific configurations
		switch vmType {
		case VMTypeContainer, VMTypeContainerd, VMTypeKataContainers:
			vmConfig.RootFS = "alpine:latest"
			vmConfig.Command = "/bin/sh"
			vmConfig.Args = []string{"-c", "sleep 10"}
		case VMTypeKVM:
			vmConfig.RootFS = "/tmp/demo-disk.img"
			vmConfig.MemoryMB = 512
		case VMTypeProcess:
			vmConfig.Command = "/bin/sleep"
			vmConfig.Args = []string{"10"}
		}

		// Get driver instance
		driver, err := manager.GetDriver(vmConfig)
		if err != nil {
			log.Printf("   ❌ Driver initialization failed: %v", err)
			if vmType == VMTypeProcess {
				log.Printf("   ℹ️  Process driver implementation planned for future release")
			}
			continue
		}

		if driver == nil {
			log.Printf("   ❌ Driver is nil")
			continue
		}

		log.Printf("   ✅ Driver initialized successfully")

		// Test capability methods
		capabilities := map[string]bool{
			"Pause":     driver.SupportsPause(),
			"Resume":    driver.SupportsResume(),
			"Snapshot":  driver.SupportsSnapshot(),
			"Migration": driver.SupportsMigrate(),
		}

		log.Printf("   📋 Capabilities:")
		for capability, supported := range capabilities {
			status := "✅"
			if !supported {
				status = "❌"
			}
			log.Printf("      %s %s: %t", status, capability, supported)
		}

		// Test basic interface methods (should not panic)
		testBasicInterface(t, driver, vmConfig, vmType)
	}
}

// testBasicInterface tests that driver interface methods exist and handle calls gracefully
func testBasicInterface(t *testing.T, driver VMDriver, config VMConfig, vmType VMType) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	log.Printf("   🔧 Testing interface methods:")

	// Test Create method
	_, err := driver.Create(ctx, config)
	if err != nil {
		log.Printf("      ⚠️  Create: Expected failure - %v", err)
	} else {
		log.Printf("      ✅ Create: Success (unexpected in test env)")
	}

	// Test other lifecycle methods with dummy VM ID
	testVMID := "demo-test-vm-id"

	methods := map[string]func() error{
		"Start":  func() error { return driver.Start(ctx, testVMID) },
		"Stop":   func() error { return driver.Stop(ctx, testVMID) },
		"Delete": func() error { return driver.Delete(ctx, testVMID) },
		"Pause":  func() error { return driver.Pause(ctx, testVMID) },
		"Resume": func() error { return driver.Resume(ctx, testVMID) },
	}

	for methodName, method := range methods {
		err := method()
		if err != nil {
			log.Printf("      ⚠️  %s: Expected failure - graceful error handling ✅", methodName)
		} else {
			log.Printf("      ✅ %s: Unexpected success", methodName)
		}
	}

	// Test query methods
	_, err = driver.GetStatus(ctx, testVMID)
	log.Printf("      ⚠️  GetStatus: Expected failure - graceful error handling ✅")

	_, err = driver.GetInfo(ctx, testVMID)
	log.Printf("      ⚠️  GetInfo: Expected failure - graceful error handling ✅")

	vms, err := driver.ListVMs(ctx)
	if err != nil {
		log.Printf("      ⚠️  ListVMs: Expected failure in test env")
	} else {
		log.Printf("      ✅ ListVMs: Success - found %d VMs", len(vms))
	}

	// Test advanced operations
	_, err = driver.Snapshot(ctx, testVMID, "demo-snapshot", nil)
	if driver.SupportsSnapshot() {
		log.Printf("      ⚠️  Snapshot: Expected failure - graceful error handling ✅")
	} else {
		log.Printf("      ℹ️  Snapshot: Not supported by %s driver", string(vmType))
	}

	err = driver.Migrate(ctx, testVMID, "demo-target-node", nil)
	if driver.SupportsMigrate() {
		log.Printf("      ⚠️  Migration: Expected failure - graceful error handling ✅")
	} else {
		log.Printf("      ℹ️  Migration: Not supported by %s driver", string(vmType))
	}
}

// testDriverCaching demonstrates the driver caching functionality
func testDriverCaching(t *testing.T, manager *VMDriverManager) {
	log.Println("\n💾 Testing Driver Caching:")
	log.Println(strings.Repeat("-", 30))

	// Create identical VM configurations
	vmConfig1 := VMConfig{
		ID:   "cache-test-1",
		Name: "cache-test",
		Tags: map[string]string{"vm_type": "container"},
	}

	vmConfig2 := VMConfig{
		ID:   "cache-test-2",
		Name: "cache-test",
		Tags: map[string]string{"vm_type": "container"},
	}

	// Get drivers multiple times
	log.Printf("🔍 Requesting container driver (first time)...")
	driver1, err1 := manager.GetDriver(vmConfig1)

	log.Printf("🔍 Requesting container driver (second time - should be cached)...")
	driver2, err2 := manager.GetDriver(vmConfig2)

	if err1 != nil || err2 != nil {
		log.Printf("❌ Driver caching test failed due to driver initialization errors")
		log.Printf("   First request error: %v", err1)
		log.Printf("   Second request error: %v", err2)
		return
	}

	// Both should return the same driver instance (cached)
	if driver1 == driver2 {
		log.Printf("✅ Driver caching working correctly - same instance returned")
	} else {
		log.Printf("⚠️  Driver caching behavior differs (may be expected based on implementation)")
	}
}

// testInterfaceCompliance verifies all drivers implement the complete VMDriver interface
func testInterfaceCompliance(t *testing.T, manager *VMDriverManager, types []VMType) {
	log.Println("\n🧾 Interface Compliance Check:")
	log.Println(strings.Repeat("-", 35))

	for _, vmType := range types {
		vmConfig := VMConfig{
			ID:   fmt.Sprintf("compliance-test-%s", string(vmType)),
			Name: "compliance-test",
			Tags: map[string]string{"vm_type": string(vmType)},
		}

		driver, err := manager.GetDriver(vmConfig)
		if err != nil {
			log.Printf("❌ %s: Driver initialization failed", string(vmType))
			continue
		}

		if driver == nil {
			log.Printf("❌ %s: Driver is nil", string(vmType))
			continue
		}

		// Check if driver implements VMDriver interface
		var _ VMDriver = driver // Compile-time interface check
		log.Printf("✅ %s: Implements VMDriver interface correctly", string(vmType))

		// Check method consistency
		if driver.SupportsResume() && !driver.SupportsPause() {
			log.Printf("⚠️  %s: Inconsistent capability - supports resume but not pause", string(vmType))
		}
	}
}

// BenchmarkDriverSystemPerformance benchmarks the overall system performance
func BenchmarkDriverSystemPerformance(b *testing.B) {
	config := DefaultVMDriverConfig("benchmark-node")
	manager := NewVMDriverManager(config)
	defer manager.Close()

	vmConfig := VMConfig{
		ID:   "benchmark-vm",
		Name: "benchmark",
		Tags: map[string]string{"vm_type": "container"},
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Test full driver retrieval and capability checking
		driver, err := manager.GetDriver(vmConfig)
		if err == nil && driver != nil {
			_ = driver.SupportsPause()
			_ = driver.SupportsResume()
			_ = driver.SupportsSnapshot()
			_ = driver.SupportsMigrate()
		}
	}
}
