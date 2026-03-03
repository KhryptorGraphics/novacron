package benchmarks

import (
	"context"
	"fmt"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// BenchmarkVMCreation benchmarks VM creation performance
func BenchmarkVMCreation(b *testing.B) {
	config := vm.VMConfig{
		ID:        "benchmark-vm",
		Name:      "benchmark",
		Command:   "/bin/true",
		Args:      []string{},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vm, err := vm.NewVM(config)
		if err != nil {
			b.Fatalf("Failed to create VM: %v", err)
		}
		vm.Cleanup()
	}
}

// BenchmarkVMManagerOperations benchmarks VM manager operations
func BenchmarkVMManagerOperations(b *testing.B) {
	// Create VM manager
	config := vm.DefaultVMManagerConfig()
	manager, err := vm.NewVMManagerFixed(config, "benchmark-node")
	if err != nil {
		b.Fatalf("Failed to create VM manager: %v", err)
	}

	err = manager.Start()
	if err != nil {
		b.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()

	ctx := context.Background()

	// Benchmark VM creation through manager
	b.Run("CreateVM", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vmConfig := vm.VMConfig{
				ID:        fmt.Sprintf("benchmark-vm-%d", i),
				Name:      fmt.Sprintf("benchmark-%d", i),
				Command:   "/bin/true",
				Args:      []string{},
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "/tmp",
			}

			vm, err := manager.CreateVM(ctx, vmConfig)
			if err != nil {
				b.Fatalf("Failed to create VM: %v", err)
			}

			// Clean up
			manager.DeleteVM(ctx, vm.ID())
		}
	})

	// Benchmark VM listing
	b.Run("ListVMs", func(b *testing.B) {
		// Create some VMs first
		for i := 0; i < 10; i++ {
			vmConfig := vm.VMConfig{
				ID:        fmt.Sprintf("list-test-vm-%d", i),
				Name:      fmt.Sprintf("list-test-%d", i),
				Command:   "/bin/true",
				Args:      []string{},
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "/tmp",
			}
			manager.CreateVM(ctx, vmConfig)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			vms := manager.ListVMs()
			_ = vms // Use the result to avoid optimization
		}

		// Clean up
		for i := 0; i < 10; i++ {
			vmID := fmt.Sprintf("list-test-vm-%d", i)
			manager.DeleteVM(ctx, vmID)
		}
	})
}

// BenchmarkConcurrentVMOperations benchmarks concurrent VM operations
func BenchmarkConcurrentVMOperations(b *testing.B) {
	config := vm.DefaultVMManagerConfig()
	manager, err := vm.NewVMManagerFixed(config, "concurrent-benchmark-node")
	if err != nil {
		b.Fatalf("Failed to create VM manager: %v", err)
	}

	err = manager.Start()
	if err != nil {
		b.Fatalf("Failed to start VM manager: %v", err)
	}
	defer manager.Stop()

	ctx := context.Background()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			vmConfig := vm.VMConfig{
				ID:        fmt.Sprintf("concurrent-vm-%d-%d", b.N, i),
				Name:      fmt.Sprintf("concurrent-%d-%d", b.N, i),
				Command:   "/bin/true",
				Args:      []string{},
				CPUShares: 1024,
				MemoryMB:  512,
				RootFS:    "/tmp",
			}

			vm, err := manager.CreateVM(ctx, vmConfig)
			if err != nil {
				b.Errorf("Failed to create VM: %v", err)
				continue
			}

			// Clean up immediately
			manager.DeleteVM(ctx, vm.ID())
			i++
		}
	})
}
