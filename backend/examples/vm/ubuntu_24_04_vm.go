package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func main() {
	fmt.Println("NovaCron Ubuntu 24.04 VM Test")

	// Create a VM manager with default configuration
	nodeID := "test-node-1"
	config := vm.DefaultVMManagerConfig()
	
	// Create a VM driver factory
	driverConfig := vm.DefaultVMDriverConfig(nodeID)
	
	// Override the VM base path for testing
	testVMPath := "/tmp/novacron-test-vms"
	driverConfig.VMBasePath = testVMPath
	
	// Create the test directory if it doesn't exist
	if err := os.MkdirAll(testVMPath, 0755); err != nil {
		log.Fatalf("Failed to create test VM directory: %v", err)
	}
	
	// Create a driver factory
	driverFactory := vm.NewVMDriverFactory(driverConfig)
	
	// Create a VM manager
	manager := vm.NewVMManager(nodeID, driverFactory, config)
	
	// Start the VM manager
	if err := manager.Start(); err != nil {
		log.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()
	
	// Create a context
	ctx := context.Background()
	
	// Create a VM spec for Ubuntu 24.04
	spec := vm.VMSpec{
		VCPU:     2,
		MemoryMB: 2048,
		DiskMB:   20480, // 20GB
		Type:     vm.VMTypeKVM,
		Image:    "/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2",
		Networks: []vm.VMNetworkSpec{
			{
				NetworkID: "default",
			},
		},
		Env: map[string]string{
			"OS_VERSION": "24.04",
			"OS_NAME":    "Ubuntu",
			"OS_CODENAME": "Noble Numbat",
		},
		Labels: map[string]string{
			"os":      "ubuntu",
			"version": "24.04",
			"lts":     "true",
		},
	}
	
	// Create a VM request
	request := vm.CreateVMRequest{
		Name:  "ubuntu-24-04-test",
		Spec:  spec,
		Owner: "admin",
		Tags: map[string]string{
			"purpose": "testing",
			"os":      "ubuntu-24.04",
		},
	}
	
	// Create the VM
	fmt.Println("Creating Ubuntu 24.04 VM...")
	vmID, err := manager.CreateVM(ctx, request)
	if err != nil {
		log.Fatalf("Failed to create VM: %v", err)
	}
	
	fmt.Printf("Created VM with ID: %s\n", vmID)
	
	// Get VM info
	vmInfo, err := manager.GetVM(ctx, vmID)
	if err != nil {
		log.Fatalf("Failed to get VM info: %v", err)
	}
	
	fmt.Printf("VM Info: %+v\n", vmInfo)
	
	// Start the VM
	fmt.Println("Starting VM...")
	if err := manager.StartVM(ctx, vmID); err != nil {
		log.Fatalf("Failed to start VM: %v", err)
	}
	
	// Wait for VM to start
	fmt.Println("Waiting for VM to start...")
	time.Sleep(5 * time.Second)
	
	// Get VM status
	status, err := manager.GetVMStatus(ctx, vmID)
	if err != nil {
		log.Fatalf("Failed to get VM status: %v", err)
	}
	
	fmt.Printf("VM Status: %s\n", status)
	
	// Keep the VM running for testing
	fmt.Println("VM is now running. Press Ctrl+C to stop and clean up.")
	fmt.Println("VM ID: ", vmID)
	fmt.Println("VM Path: ", testVMPath+"/"+vmID)
	
	// Wait for user to press Ctrl+C
	select {}
}
