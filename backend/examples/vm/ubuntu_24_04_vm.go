package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func main() {
	fmt.Println("NovaCron Ubuntu 24.04 VM Test")

	// Create a VM manager with the KVM driver enabled
	config := vm.DefaultVMManagerConfig()
	config.Drivers = map[vm.VMType]vm.VMDriverConfigManager{
		vm.VMTypeKVM: {Enabled: true},
	}

	manager, err := vm.NewVMManager(config)
	if err != nil {
		log.Fatalf("Failed to create VM manager: %v", err)
	}

	// Start the VM manager
	if err := manager.Start(); err != nil {
		log.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()

	// Create a context
	ctx := context.Background()

	// Create a VM config for Ubuntu 24.04
	vmConfig := vm.VMConfig{
		Name:       "ubuntu-24-04-test",
		Type:       vm.VMTypeKVM,
		VCPUs:      2,
		MemoryMB:   2048,
		DiskSizeGB: 20,
		Image:      "/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2",
		Env: map[string]string{
			"OS_VERSION":  "24.04",
			"OS_NAME":     "Ubuntu",
			"OS_CODENAME": "Noble Numbat",
		},
		Tags: map[string]string{
			"os":      "ubuntu",
			"version": "24.04",
			"lts":     "true",
			"purpose": "testing",
		},
	}

	// Create a VM request
	request := vm.CreateVMRequest{
		Name: vmConfig.Name,
		Spec: vmConfig,
		Tags: vmConfig.Tags,
	}

	// Create the VM
	fmt.Println("Creating Ubuntu 24.04 VM...")
	createdVM, err := manager.CreateVM(ctx, request)
	if err != nil {
		log.Fatalf("Failed to create VM: %v", err)
	}
	vmID := createdVM.ID()

	fmt.Printf("Created VM with ID: %s\n", vmID)

	// Get VM info
	vmInfo, err := manager.GetVM(vmID)
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

	fmt.Printf("VM Status: %s\n", status.Status)

	// Keep the VM running for testing
	fmt.Println("VM is now running. Press Ctrl+C to stop and clean up.")
	fmt.Println("VM ID: ", vmID)

	// Wait for user to press Ctrl+C
	select {}
}
