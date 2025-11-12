package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/novacron/dwcp-sdk-go"
)

func main() {
	// Create client configuration
	config := dwcp.DefaultConfig()
	config.Address = "localhost"
	config.Port = 9000
	config.APIKey = "your-api-key"
	config.TLSEnabled = true

	// Create client
	client, err := dwcp.NewClient(config)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Disconnect()

	// Connect to server
	ctx := context.Background()
	if err := client.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}

	fmt.Println("Connected to DWCP server")

	// Create VM
	vmConfig := dwcp.VMConfig{
		Name:   "example-vm",
		Memory: 2 * 1024 * 1024 * 1024, // 2GB
		CPUs:   2,
		Disk:   20 * 1024 * 1024 * 1024, // 20GB
		Image:  "ubuntu-22.04",
		Network: dwcp.NetworkConfig{
			Mode: "bridge",
			Interfaces: []dwcp.NetIf{
				{
					Name: "eth0",
					Type: "virtio",
				},
			},
		},
		Labels: map[string]string{
			"env":  "production",
			"team": "platform",
		},
	}

	vmClient := client.VM()
	vm, err := vmClient.Create(ctx, vmConfig)
	if err != nil {
		log.Fatalf("Failed to create VM: %v", err)
	}

	fmt.Printf("Created VM: %s (ID: %s)\n", vm.Name, vm.ID)

	// Start VM
	if err := vmClient.Start(ctx, vm.ID); err != nil {
		log.Fatalf("Failed to start VM: %v", err)
	}

	fmt.Println("VM started successfully")

	// Watch VM events
	events, err := vmClient.Watch(ctx, vm.ID)
	if err != nil {
		log.Fatalf("Failed to watch VM: %v", err)
	}

	// Monitor events for 30 seconds
	timeout := time.After(30 * time.Second)
	for {
		select {
		case event := <-events:
			fmt.Printf("Event: %s - %s (State: %s)\n",
				event.Type, event.Message, event.VM.State)
		case <-timeout:
			fmt.Println("Monitoring complete")
			goto cleanup
		}
	}

cleanup:
	// Stop VM
	if err := vmClient.Stop(ctx, vm.ID, false); err != nil {
		log.Printf("Failed to stop VM: %v", err)
	}

	// Destroy VM
	if err := vmClient.Destroy(ctx, vm.ID); err != nil {
		log.Printf("Failed to destroy VM: %v", err)
	}

	fmt.Println("Cleanup complete")
}
