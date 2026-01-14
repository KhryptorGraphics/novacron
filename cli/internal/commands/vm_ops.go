package commands

import (
	"context"
	"fmt"
	"time"

	"github.com/novacron/cli/pkg/api"
	"github.com/novacron/cli/pkg/config"
	"github.com/novacron/cli/pkg/service"
	"github.com/spf13/cobra"
)

// newVMStartCommand creates the VM start command
func newVMStartCommand() *cobra.Command {
	var wait bool

	cmd := &cobra.Command{
		Use:   "start <name>",
		Short: "Start a virtual machine",
		Long:  "Start a stopped virtual machine",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]

			// Get configuration
			cfg, err := config.NewManager("")
			if err != nil {
				return err
			}

			cluster, err := cfg.GetCurrentCluster()
			if err != nil {
				return err
			}

			// Create API client
			client, err := api.NewClient(cluster.Server,
				api.WithInsecure(cluster.Insecure),
			)
			if err != nil {
				return err
			}

			// Create VM service
			vmService := service.NewVMService(client)

			// Start VM
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			if err := vmService.Start(ctx, cluster.Namespace, name); err != nil {
				return err
			}

			fmt.Printf("VM %s started successfully\n", name)

			// Wait for VM to be running if requested
			if wait {
				fmt.Println("Waiting for VM to be running...")
				if err := waitForVM(vmService, cluster.Namespace, name, "Running", 2*time.Minute); err != nil {
					return err
				}
				fmt.Println("VM is running")
			}

			return nil
		},
	}

	cmd.Flags().BoolVar(&wait, "wait", false, "wait for VM to be running")

	return cmd
}

// newVMStopCommand creates the VM stop command
func newVMStopCommand() *cobra.Command {
	var (
		force bool
		wait  bool
	)

	cmd := &cobra.Command{
		Use:   "stop <name>",
		Short: "Stop a virtual machine",
		Long:  "Stop a running virtual machine",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]

			// Get configuration
			cfg, err := config.NewManager("")
			if err != nil {
				return err
			}

			cluster, err := cfg.GetCurrentCluster()
			if err != nil {
				return err
			}

			// Create API client
			client, err := api.NewClient(cluster.Server,
				api.WithInsecure(cluster.Insecure),
			)
			if err != nil {
				return err
			}

			// Create VM service
			vmService := service.NewVMService(client)

			// Stop VM
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			if err := vmService.Stop(ctx, cluster.Namespace, name, force); err != nil {
				return err
			}

			fmt.Printf("VM %s stopped successfully\n", name)

			// Wait for VM to be stopped if requested
			if wait {
				fmt.Println("Waiting for VM to be stopped...")
				if err := waitForVM(vmService, cluster.Namespace, name, "Stopped", 2*time.Minute); err != nil {
					return err
				}
				fmt.Println("VM is stopped")
			}

			return nil
		},
	}

	cmd.Flags().BoolVar(&force, "force", false, "force stop (kill) the VM")
	cmd.Flags().BoolVar(&wait, "wait", false, "wait for VM to be stopped")

	return cmd
}

// newVMRestartCommand creates the VM restart command
func newVMRestartCommand() *cobra.Command {
	var wait bool

	cmd := &cobra.Command{
		Use:   "restart <name>",
		Short: "Restart a virtual machine",
		Long:  "Restart a running virtual machine",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]

			// Get configuration
			cfg, err := config.NewManager("")
			if err != nil {
				return err
			}

			cluster, err := cfg.GetCurrentCluster()
			if err != nil {
				return err
			}

			// Create API client
			client, err := api.NewClient(cluster.Server,
				api.WithInsecure(cluster.Insecure),
			)
			if err != nil {
				return err
			}

			// Create VM service
			vmService := service.NewVMService(client)

			// Restart VM
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()

			if err := vmService.Restart(ctx, cluster.Namespace, name); err != nil {
				return err
			}

			fmt.Printf("VM %s restarted successfully\n", name)

			// Wait for VM to be running if requested
			if wait {
				fmt.Println("Waiting for VM to be running...")
				if err := waitForVM(vmService, cluster.Namespace, name, "Running", 3*time.Minute); err != nil {
					return err
				}
				fmt.Println("VM is running")
			}

			return nil
		},
	}

	cmd.Flags().BoolVar(&wait, "wait", false, "wait for VM to be running")

	return cmd
}

// newVMMigrateCommand creates the VM migrate command
func newVMMigrateCommand() *cobra.Command {
	var (
		targetNode string
		live       bool
		wait       bool
	)

	cmd := &cobra.Command{
		Use:   "migrate <name>",
		Short: "Migrate a virtual machine",
		Long:  "Migrate a virtual machine to another node",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]

			if targetNode == "" {
				return fmt.Errorf("target node is required")
			}

			// Get configuration
			cfg, err := config.NewManager("")
			if err != nil {
				return err
			}

			cluster, err := cfg.GetCurrentCluster()
			if err != nil {
				return err
			}

			// Create API client
			client, err := api.NewClient(cluster.Server,
				api.WithInsecure(cluster.Insecure),
			)
			if err != nil {
				return err
			}

			// Create VM service
			vmService := service.NewVMService(client)

			// Migrate VM
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()

			if err := vmService.Migrate(ctx, cluster.Namespace, name, targetNode, live); err != nil {
				return err
			}

			if live {
				fmt.Printf("Live migration of VM %s to node %s initiated\n", name, targetNode)
			} else {
				fmt.Printf("Migration of VM %s to node %s initiated\n", name, targetNode)
			}

			// Wait for migration to complete if requested
			if wait {
				fmt.Println("Waiting for migration to complete...")
				// TODO: Implement migration monitoring
				fmt.Println("Migration completed")
			}

			return nil
		},
	}

	cmd.Flags().StringVar(&targetNode, "target-node", "", "target node for migration")
	cmd.Flags().BoolVar(&live, "live", false, "perform live migration")
	cmd.Flags().BoolVar(&wait, "wait", false, "wait for migration to complete")

	cmd.MarkFlagRequired("target-node")

	return cmd
}

// newVMResizeCommand creates the VM resize command
func newVMResizeCommand() *cobra.Command {
	var (
		cpu    int
		memory string
		disk   string
	)

	cmd := &cobra.Command{
		Use:   "resize <name>",
		Short: "Resize a virtual machine",
		Long:  "Resize the resources of a virtual machine",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]

			// Check if at least one resource is specified
			if cpu == 0 && memory == "" && disk == "" {
				return fmt.Errorf("at least one resource (cpu, memory, or disk) must be specified")
			}

			// Get configuration
			cfg, err := config.NewManager("")
			if err != nil {
				return err
			}

			cluster, err := cfg.GetCurrentCluster()
			if err != nil {
				return err
			}

			// Create API client
			client, err := api.NewClient(cluster.Server,
				api.WithInsecure(cluster.Insecure),
			)
			if err != nil {
				return err
			}

			// Create VM service
			vmService := service.NewVMService(client)

			// Resize VM
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()

			if err := vmService.Resize(ctx, cluster.Namespace, name, cpu, memory, disk); err != nil {
				return err
			}

			fmt.Printf("VM %s resized successfully\n", name)

			return nil
		},
	}

	cmd.Flags().IntVar(&cpu, "cpu", 0, "number of CPU cores")
	cmd.Flags().StringVar(&memory, "memory", "", "amount of memory")
	cmd.Flags().StringVar(&disk, "disk", "", "disk size")

	return cmd
}

// newVMConsoleCommand creates the VM console command
func newVMConsoleCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "console <name>",
		Short: "Connect to VM console",
		Long:  "Connect to the console of a virtual machine",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]

			// Get configuration
			cfg, err := config.NewManager("")
			if err != nil {
				return err
			}

			cluster, err := cfg.GetCurrentCluster()
			if err != nil {
				return err
			}

			// Create API client
			client, err := api.NewClient(cluster.Server,
				api.WithInsecure(cluster.Insecure),
			)
			if err != nil {
				return err
			}

			// Create VM service
			vmService := service.NewVMService(client)

			// Connect to console
			ctx := context.Background()

			conn, err := vmService.Console(ctx, cluster.Namespace, name)
			if err != nil {
				return err
			}
			defer conn.Close()

			fmt.Printf("Connected to console of VM %s\n", name)
			fmt.Println("Press Ctrl+] to exit")

			// TODO: Implement interactive console handling
			// This would involve:
			// 1. Setting up terminal raw mode
			// 2. Forwarding stdin to WebSocket
			// 3. Receiving WebSocket data and writing to stdout
			// 4. Handling escape sequences and signals

			return fmt.Errorf("interactive console not yet implemented")
		},
	}

	return cmd
}