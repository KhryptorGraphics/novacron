package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"
	"github.com/novacron/dwcp-sdk-go"
)

var vmCmd = &cobra.Command{
	Use:   "vm",
	Short: "Manage virtual machines",
	Long:  `Create, manage, and monitor virtual machines through DWCP.`,
}

var vmListCmd = &cobra.Command{
	Use:   "list",
	Short: "List all virtual machines",
	Long:  `Display a list of all virtual machines with their current status.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		client, err := createClient()
		if err != nil {
			return err
		}
		defer client.Disconnect()

		vmClient := client.VM()
		vms, err := vmClient.List(context.Background(), nil)
		if err != nil {
			return fmt.Errorf("failed to list VMs: %w", err)
		}

		// Print as table
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
		fmt.Fprintln(w, "ID\tNAME\tSTATE\tCPUs\tMEMORY\tNODE\tCREATED")

		for _, vm := range vms {
			memGB := float64(vm.Config.Memory) / (1024 * 1024 * 1024)
			created := vm.CreatedAt.Format("2006-01-02 15:04")
			fmt.Fprintf(w, "%s\t%s\t%s\t%d\t%.1fG\t%s\t%s\n",
				vm.ID, vm.Name, vm.State, vm.Config.CPUs, memGB, vm.Node, created)
		}

		w.Flush()
		return nil
	},
}

var vmCreateCmd = &cobra.Command{
	Use:   "create [name]",
	Short: "Create a new virtual machine",
	Long:  `Create a new virtual machine with specified configuration.`,
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		name := args[0]

		// Get flags
		memory, _ := cmd.Flags().GetString("memory")
		cpus, _ := cmd.Flags().GetInt("cpus")
		disk, _ := cmd.Flags().GetString("disk")
		image, _ := cmd.Flags().GetString("image")
		template, _ := cmd.Flags().GetString("template")

		// Parse sizes
		memBytes, err := parseSize(memory)
		if err != nil {
			return fmt.Errorf("invalid memory size: %w", err)
		}

		diskBytes, err := parseSize(disk)
		if err != nil {
			return fmt.Errorf("invalid disk size: %w", err)
		}

		// Create client
		client, err := createClient()
		if err != nil {
			return err
		}
		defer client.Disconnect()

		// Build config
		config := dwcp.VMConfig{
			Name:   name,
			Memory: memBytes,
			CPUs:   uint32(cpus),
			Disk:   diskBytes,
			Image:  image,
			Network: dwcp.NetworkConfig{
				Mode: "bridge",
				Interfaces: []dwcp.NetIf{
					{
						Name: "eth0",
						Type: "virtio",
					},
				},
			},
		}

		// Load from template if specified
		if template != "" {
			if err := loadTemplate(&config, template); err != nil {
				return fmt.Errorf("failed to load template: %w", err)
			}
		}

		// Create VM
		vmClient := client.VM()
		vm, err := vmClient.Create(context.Background(), config)
		if err != nil {
			return fmt.Errorf("failed to create VM: %w", err)
		}

		fmt.Printf("✓ Created VM: %s (ID: %s)\n", vm.Name, vm.ID)
		fmt.Printf("  State: %s\n", vm.State)
		fmt.Printf("  Node: %s\n", vm.Node)
		fmt.Printf("  Memory: %.1f GB\n", float64(vm.Config.Memory)/(1024*1024*1024))
		fmt.Printf("  CPUs: %d\n", vm.Config.CPUs)

		return nil
	},
}

var vmStartCmd = &cobra.Command{
	Use:   "start [vm-id]",
	Short: "Start a virtual machine",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		vmID := args[0]

		client, err := createClient()
		if err != nil {
			return err
		}
		defer client.Disconnect()

		vmClient := client.VM()
		if err := vmClient.Start(context.Background(), vmID); err != nil {
			return fmt.Errorf("failed to start VM: %w", err)
		}

		fmt.Printf("✓ Started VM: %s\n", vmID)
		return nil
	},
}

var vmStopCmd = &cobra.Command{
	Use:   "stop [vm-id]",
	Short: "Stop a virtual machine",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		vmID := args[0]
		force, _ := cmd.Flags().GetBool("force")

		client, err := createClient()
		if err != nil {
			return err
		}
		defer client.Disconnect()

		vmClient := client.VM()
		if err := vmClient.Stop(context.Background(), vmID, force); err != nil {
			return fmt.Errorf("failed to stop VM: %w", err)
		}

		if force {
			fmt.Printf("✓ Forcefully stopped VM: %s\n", vmID)
		} else {
			fmt.Printf("✓ Stopped VM: %s\n", vmID)
		}
		return nil
	},
}

var vmDestroyCmd = &cobra.Command{
	Use:   "destroy [vm-id]",
	Short: "Destroy a virtual machine",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		vmID := args[0]

		client, err := createClient()
		if err != nil {
			return err
		}
		defer client.Disconnect()

		vmClient := client.VM()
		if err := vmClient.Destroy(context.Background(), vmID); err != nil {
			return fmt.Errorf("failed to destroy VM: %w", err)
		}

		fmt.Printf("✓ Destroyed VM: %s\n", vmID)
		return nil
	},
}

var vmShowCmd = &cobra.Command{
	Use:   "show [vm-id]",
	Short: "Show detailed VM information",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		vmID := args[0]

		client, err := createClient()
		if err != nil {
			return err
		}
		defer client.Disconnect()

		vmClient := client.VM()
		vm, err := vmClient.Get(context.Background(), vmID)
		if err != nil {
			return fmt.Errorf("failed to get VM: %w", err)
		}

		// Print detailed info
		fmt.Printf("VM Details:\n")
		fmt.Printf("  ID: %s\n", vm.ID)
		fmt.Printf("  Name: %s\n", vm.Name)
		fmt.Printf("  State: %s\n", vm.State)
		fmt.Printf("  Node: %s\n", vm.Node)
		fmt.Printf("\nResources:\n")
		fmt.Printf("  CPUs: %d\n", vm.Config.CPUs)
		fmt.Printf("  Memory: %.1f GB\n", float64(vm.Config.Memory)/(1024*1024*1024))
		fmt.Printf("  Disk: %.1f GB\n", float64(vm.Config.Disk)/(1024*1024*1024))
		fmt.Printf("\nTimestamps:\n")
		fmt.Printf("  Created: %s\n", vm.CreatedAt.Format(time.RFC3339))
		fmt.Printf("  Updated: %s\n", vm.UpdatedAt.Format(time.RFC3339))
		if vm.StartedAt != nil {
			fmt.Printf("  Started: %s\n", vm.StartedAt.Format(time.RFC3339))
		}

		if vm.Metrics != nil {
			fmt.Printf("\nMetrics:\n")
			fmt.Printf("  CPU Usage: %.2f%%\n", vm.Metrics.CPUUsage)
			fmt.Printf("  Memory Used: %.1f GB\n", float64(vm.Metrics.MemoryUsed)/(1024*1024*1024))
			fmt.Printf("  Network RX: %.1f MB\n", float64(vm.Metrics.NetworkRx)/(1024*1024))
			fmt.Printf("  Network TX: %.1f MB\n", float64(vm.Metrics.NetworkTx)/(1024*1024))
		}

		if len(vm.Labels) > 0 {
			fmt.Printf("\nLabels:\n")
			for k, v := range vm.Labels {
				fmt.Printf("  %s: %s\n", k, v)
			}
		}

		return nil
	},
}

var vmMigrateCmd = &cobra.Command{
	Use:   "migrate [vm-id] --target [node]",
	Short: "Migrate a virtual machine to another node",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		vmID := args[0]
		targetNode, _ := cmd.Flags().GetString("target")
		live, _ := cmd.Flags().GetBool("live")

		if targetNode == "" {
			return fmt.Errorf("target node is required")
		}

		client, err := createClient()
		if err != nil {
			return err
		}
		defer client.Disconnect()

		options := dwcp.MigrationOptions{
			Live:         live,
			Compression:  true,
			AutoConverge: true,
			MaxDowntime:  500,
		}

		vmClient := client.VM()
		status, err := vmClient.Migrate(context.Background(), vmID, targetNode, options)
		if err != nil {
			return fmt.Errorf("failed to start migration: %w", err)
		}

		fmt.Printf("✓ Migration started: %s\n", status.ID)
		fmt.Printf("  VM: %s\n", status.VMID)
		fmt.Printf("  Source: %s → Target: %s\n", status.SourceNode, status.TargetNode)
		fmt.Printf("  Type: ")
		if live {
			fmt.Println("Live")
		} else {
			fmt.Println("Offline")
		}

		// Monitor progress
		fmt.Println("\nMonitoring migration progress...")
		for {
			time.Sleep(2 * time.Second)

			status, err := vmClient.GetMigrationStatus(context.Background(), status.ID)
			if err != nil {
				return fmt.Errorf("failed to get migration status: %w", err)
			}

			throughputMB := float64(status.Throughput) / (1024 * 1024)
			fmt.Printf("\r  Progress: %.1f%% | Throughput: %.1f MB/s | State: %s",
				status.Progress, throughputMB, status.State)

			if status.State == dwcp.MigrationStateCompleted {
				fmt.Printf("\n\n✓ Migration completed in %.2fs\n", time.Since(status.StartedAt).Seconds())
				fmt.Printf("  Downtime: %d ms\n", status.Downtime)
				break
			}

			if status.State == dwcp.MigrationStateFailed {
				return fmt.Errorf("\n✗ Migration failed: %s", status.Error)
			}
		}

		return nil
	},
}

var vmSnapshotCmd = &cobra.Command{
	Use:   "snapshot [vm-id] [snapshot-name]",
	Short: "Create a VM snapshot",
	Args:  cobra.ExactArgs(2),
	RunE: func(cmd *cobra.Command, args []string) error {
		vmID := args[0]
		snapshotName := args[1]
		description, _ := cmd.Flags().GetString("description")
		includeMemory, _ := cmd.Flags().GetBool("memory")

		client, err := createClient()
		if err != nil {
			return err
		}
		defer client.Disconnect()

		options := dwcp.SnapshotOptions{
			IncludeMemory: includeMemory,
			Description:   description,
			Quiesce:       true,
		}

		vmClient := client.VM()
		snapshot, err := vmClient.Snapshot(context.Background(), vmID, snapshotName, options)
		if err != nil {
			return fmt.Errorf("failed to create snapshot: %w", err)
		}

		fmt.Printf("✓ Snapshot created: %s (ID: %s)\n", snapshot.Name, snapshot.ID)
		fmt.Printf("  Size: %.1f GB\n", float64(snapshot.Size)/(1024*1024*1024))
		fmt.Printf("  Created: %s\n", snapshot.CreatedAt.Format(time.RFC3339))

		return nil
	},
}

func init() {
	rootCmd.AddCommand(vmCmd)

	// Add subcommands
	vmCmd.AddCommand(vmListCmd)
	vmCmd.AddCommand(vmCreateCmd)
	vmCmd.AddCommand(vmStartCmd)
	vmCmd.AddCommand(vmStopCmd)
	vmCmd.AddCommand(vmDestroyCmd)
	vmCmd.AddCommand(vmShowCmd)
	vmCmd.AddCommand(vmMigrateCmd)
	vmCmd.AddCommand(vmSnapshotCmd)

	// vm create flags
	vmCreateCmd.Flags().String("memory", "2G", "Memory size (e.g., 2G, 4096M)")
	vmCreateCmd.Flags().Int("cpus", 2, "Number of CPUs")
	vmCreateCmd.Flags().String("disk", "20G", "Disk size (e.g., 20G, 40960M)")
	vmCreateCmd.Flags().String("image", "ubuntu-22.04", "Base image")
	vmCreateCmd.Flags().String("template", "", "Use template")

	// vm stop flags
	vmStopCmd.Flags().Bool("force", false, "Force stop")

	// vm migrate flags
	vmMigrateCmd.Flags().String("target", "", "Target node")
	vmMigrateCmd.Flags().Bool("live", true, "Live migration")
	vmMigrateCmd.MarkFlagRequired("target")

	// vm snapshot flags
	vmSnapshotCmd.Flags().String("description", "", "Snapshot description")
	vmSnapshotCmd.Flags().Bool("memory", true, "Include memory state")
}

// Helper functions

func createClient() (*dwcp.Client, error) {
	config := dwcp.DefaultConfig()
	config.Address = address
	config.Port = port
	config.APIKey = apiKey

	client, err := dwcp.NewClient(config)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := client.Connect(ctx); err != nil {
		return nil, err
	}

	return client, nil
}

func parseSize(size string) (uint64, error) {
	// Simple size parser (supports G, M, K suffixes)
	var value uint64
	var unit rune

	_, err := fmt.Sscanf(size, "%d%c", &value, &unit)
	if err != nil {
		return 0, err
	}

	switch unit {
	case 'G', 'g':
		return value * 1024 * 1024 * 1024, nil
	case 'M', 'm':
		return value * 1024 * 1024, nil
	case 'K', 'k':
		return value * 1024, nil
	default:
		return value, nil
	}
}

func loadTemplate(config *dwcp.VMConfig, template string) error {
	// Load template from marketplace or local file
	// This is a placeholder - actual implementation would fetch from marketplace
	return nil
}
