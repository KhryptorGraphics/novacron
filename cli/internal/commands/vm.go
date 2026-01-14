package commands

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/novacron/cli/pkg/api"
	"github.com/novacron/cli/pkg/config"
	"github.com/novacron/cli/pkg/output"
	"github.com/novacron/cli/pkg/service"
	"github.com/spf13/cobra"
)

// NewVMCommand creates the VM command
func NewVMCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "vm",
		Aliases: []string{"vms", "virtualmachine", "virtualmachines"},
		Short:   "Manage virtual machines",
		Long:    "Commands for managing virtual machine lifecycle, configuration, and operations",
	}

	// Add subcommands
	cmd.AddCommand(
		newVMListCommand(),
		newVMGetCommand(),
		newVMCreateCommand(),
		newVMDeleteCommand(),
		newVMStartCommand(),
		newVMStopCommand(),
		newVMRestartCommand(),
		newVMMigrateCommand(),
		newVMResizeCommand(),
		newVMConsoleCommand(),
	)

	return cmd
}

// newVMListCommand creates the VM list command
func newVMListCommand() *cobra.Command {
	var (
		allNamespaces bool
		selector      string
		showNodes     bool
	)

	cmd := &cobra.Command{
		Use:     "list",
		Aliases: []string{"ls"},
		Short:   "List virtual machines",
		Long:    "List all virtual machines in the cluster or specific namespace",
		RunE: func(cmd *cobra.Command, args []string) error {
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

			// Determine namespace
			namespace := cluster.Namespace
			if allNamespaces {
				namespace = ""
			}

			// List VMs
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			vms, err := vmService.List(ctx, namespace)
			if err != nil {
				return err
			}

			// Filter by selector if provided
			if selector != "" {
				vms = filterVMsBySelector(vms, selector)
			}

			// Print results
			format := output.GetFormat()
			printer := output.NewPrinter(format)

			if format == "table" || format == "wide" {
				return printVMTable(vms, showNodes, format == "wide")
			}

			return printer.Print(vms)
		},
	}

	cmd.Flags().BoolVarP(&allNamespaces, "all-namespaces", "A", false, "list VMs across all namespaces")
	cmd.Flags().StringVarP(&selector, "selector", "l", "", "filter by label selector")
	cmd.Flags().BoolVar(&showNodes, "show-nodes", false, "show node information")

	return cmd
}

// newVMGetCommand creates the VM get command
func newVMGetCommand() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get <name>",
		Short: "Get details of a virtual machine",
		Long:  "Display detailed information about a specific virtual machine",
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

			// Get VM
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			vm, err := vmService.Get(ctx, cluster.Namespace, name)
			if err != nil {
				return err
			}

			// Print result
			printer := output.NewPrinter(output.GetFormat())
			return printer.Print(vm)
		},
	}

	return cmd
}

// newVMCreateCommand creates the VM create command
func newVMCreateCommand() *cobra.Command {
	var (
		file     string
		cpu      int
		memory   string
		disk     string
		image    string
		network  string
		userData string
		labels   []string
		wait     bool
	)

	cmd := &cobra.Command{
		Use:   "create <name>",
		Short: "Create a new virtual machine",
		Long:  "Create a new virtual machine with specified configuration",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			var vm *api.VirtualMachine

			if file != "" {
				// Load from file
				data, err := os.ReadFile(file)
				if err != nil {
					return fmt.Errorf("failed to read file: %w", err)
				}

				// Parse YAML/JSON
				vm, err = parseVMManifest(data)
				if err != nil {
					return err
				}
			} else {
				// Create from flags
				if len(args) == 0 {
					return fmt.Errorf("VM name is required")
				}

				name := args[0]
				vm = &api.VirtualMachine{
					Name: name,
					Spec: api.VMSpec{
						Running: true,
						Template: api.VMTemplate{
							Spec: api.VMTemplateSpec{
								Resources: api.Resources{
									CPU:    cpu,
									Memory: memory,
									Disk:   disk,
								},
								Image: api.VMImage{
									Source: image,
								},
								UserData: userData,
							},
						},
					},
				}

				// Add network if specified
				if network != "" {
					vm.Spec.Template.Spec.Networks = []api.NetworkInterface{
						{
							Name: "default",
							Type: network,
							IPv4: &api.IPConfig{
								Method: "dhcp",
							},
						},
					}
				}

				// Parse labels
				if len(labels) > 0 {
					vm.Metadata.Labels = parseLabels(labels)
				}
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

			// Create VM
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
			defer cancel()

			created, err := vmService.Create(ctx, cluster.Namespace, vm)
			if err != nil {
				return err
			}

			fmt.Printf("VM %s created successfully\n", created.Name)

			// Wait for VM to be ready if requested
			if wait {
				fmt.Println("Waiting for VM to be ready...")
				if err := waitForVM(vmService, cluster.Namespace, created.Name, "Running", 5*time.Minute); err != nil {
					return err
				}
				fmt.Println("VM is ready")
			}

			return nil
		},
	}

	cmd.Flags().StringVarP(&file, "file", "f", "", "create VM from YAML/JSON file")
	cmd.Flags().IntVar(&cpu, "cpu", 2, "number of CPU cores")
	cmd.Flags().StringVar(&memory, "memory", "4Gi", "amount of memory")
	cmd.Flags().StringVar(&disk, "disk", "20Gi", "disk size")
	cmd.Flags().StringVar(&image, "image", "", "VM image source")
	cmd.Flags().StringVar(&network, "network", "bridge", "network type")
	cmd.Flags().StringVar(&userData, "user-data", "", "cloud-init user data")
	cmd.Flags().StringSliceVarP(&labels, "labels", "l", nil, "labels to apply (key=value)")
	cmd.Flags().BoolVar(&wait, "wait", false, "wait for VM to be ready")

	return cmd
}

// newVMDeleteCommand creates the VM delete command
func newVMDeleteCommand() *cobra.Command {
	var (
		force bool
		wait  bool
	)

	cmd := &cobra.Command{
		Use:     "delete <name>",
		Aliases: []string{"rm", "remove"},
		Short:   "Delete a virtual machine",
		Long:    "Delete a virtual machine and its associated resources",
		Args:    cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			name := args[0]

			// Confirm deletion if not forced
			if !force {
				fmt.Printf("Are you sure you want to delete VM %s? (y/N): ", name)
				var response string
				fmt.Scanln(&response)
				if strings.ToLower(response) != "y" {
					fmt.Println("Deletion cancelled")
					return nil
				}
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

			// Delete VM
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			if err := vmService.Delete(ctx, cluster.Namespace, name); err != nil {
				return err
			}

			fmt.Printf("VM %s deleted successfully\n", name)

			// Wait for deletion if requested
			if wait {
				fmt.Println("Waiting for VM to be deleted...")
				if err := waitForVMDeleted(vmService, cluster.Namespace, name, 2*time.Minute); err != nil {
					return err
				}
				fmt.Println("VM deleted")
			}

			return nil
		},
	}

	cmd.Flags().BoolVar(&force, "force", false, "force deletion without confirmation")
	cmd.Flags().BoolVar(&wait, "wait", false, "wait for VM to be deleted")

	return cmd
}

// Helper functions

func filterVMsBySelector(vms []api.VirtualMachine, selector string) []api.VirtualMachine {
	// Parse selector
	selectors := parseLabels([]string{selector})
	
	var filtered []api.VirtualMachine
	for _, vm := range vms {
		if matchLabels(vm.Metadata.Labels, selectors) {
			filtered = append(filtered, vm)
		}
	}
	
	return filtered
}

func parseLabels(labels []string) map[string]string {
	result := make(map[string]string)
	for _, label := range labels {
		parts := strings.SplitN(label, "=", 2)
		if len(parts) == 2 {
			result[parts[0]] = parts[1]
		}
	}
	return result
}

func matchLabels(vmLabels, selectors map[string]string) bool {
	for k, v := range selectors {
		if vmLabels[k] != v {
			return false
		}
	}
	return true
}

func parseVMManifest(data []byte) (*api.VirtualMachine, error) {
	// TODO: Implement YAML/JSON parsing
	return nil, fmt.Errorf("manifest parsing not implemented")
}

func waitForVM(service *service.VMService, namespace, name, targetPhase string, timeout time.Duration) error {
	// TODO: Implement wait logic
	return nil
}

func waitForVMDeleted(service *service.VMService, namespace, name string, timeout time.Duration) error {
	// TODO: Implement wait logic
	return nil
}

func printVMTable(vms []api.VirtualMachine, showNodes, wide bool) error {
	// TODO: Implement table printing
	fmt.Println("NAME\tSTATUS\tAGE")
	for _, vm := range vms {
		age := time.Since(vm.CreatedAt).Round(time.Second)
		fmt.Printf("%s\t%s\t%s\n", vm.Name, vm.Status.Phase, age)
	}
	return nil
}