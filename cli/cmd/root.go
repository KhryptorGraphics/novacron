package cmd

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	cfgFile string
	apiKey  string
	address string
	port    int
	verbose bool
)

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "novacron",
	Short: "NovaCron DWCP CLI - Manage distributed virtual machines",
	Long: `novacron is a powerful CLI tool for managing virtual machines through the
Distributed Worker Control Protocol (DWCP) v3.

Features:
  - VM lifecycle management (create, start, stop, destroy)
  - Live migration with progress tracking
  - Snapshot and restore operations
  - Real-time monitoring and metrics
  - Cluster management
  - Template marketplace
  - Rich TUI interfaces

Examples:
  # List all VMs
  novacron vm list

  # Create a new VM from template
  novacron vm create --template ubuntu-22.04 --name my-vm

  # Start interactive dashboard
  novacron monitor

  # Migrate VM to another node
  novacron vm migrate vm-123 --target node-02 --live

For more information, visit: https://docs.novacron.io`,
	Version: "3.0.0",
}

// Execute adds all child commands to the root command and sets flags appropriately.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func init() {
	cobra.OnInitialize(initConfig)

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.novacron.yaml)")
	rootCmd.PersistentFlags().StringVar(&apiKey, "api-key", "", "API key for authentication")
	rootCmd.PersistentFlags().StringVar(&address, "address", "localhost", "DWCP server address")
	rootCmd.PersistentFlags().IntVar(&port, "port", 9000, "DWCP server port")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")

	// Bind flags to viper
	viper.BindPFlag("api_key", rootCmd.PersistentFlags().Lookup("api-key"))
	viper.BindPFlag("address", rootCmd.PersistentFlags().Lookup("address"))
	viper.BindPFlag("port", rootCmd.PersistentFlags().Lookup("port"))
}

func initConfig() {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		cobra.CheckErr(err)

		viper.AddConfigPath(home)
		viper.SetConfigType("yaml")
		viper.SetConfigName(".novacron")
	}

	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err == nil {
		if verbose {
			fmt.Fprintln(os.Stderr, "Using config file:", viper.ConfigFileUsed())
		}
	}
}
