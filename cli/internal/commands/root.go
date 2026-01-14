package commands

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/mitchellh/go-homedir"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var (
	cfgFile      string
	verbose      bool
	outputFormat string
	noColor      bool
	clusterName  string
	insecure     bool
)

// NewRootCommand creates the root command
func NewRootCommand(version, gitCommit, buildDate string) *cobra.Command {
	rootCmd := &cobra.Command{
		Use:   "novacron",
		Short: "NovaCron CLI - Distributed VM Management",
		Long: `NovaCron CLI is a command-line interface for managing virtual machines
in a distributed NovaCron cluster. It provides commands for VM lifecycle
management, migration, monitoring, and cluster administration.`,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			// Initialize configuration
			if err := initConfig(); err != nil {
				return fmt.Errorf("failed to initialize config: %w", err)
			}

			// Set log level
			if verbose {
				logrus.SetLevel(logrus.DebugLevel)
			}

			return nil
		},
	}

	// Global flags
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default is $HOME/.novacron/config.yaml)")
	rootCmd.PersistentFlags().BoolVarP(&verbose, "verbose", "v", false, "verbose output")
	rootCmd.PersistentFlags().StringVarP(&outputFormat, "output", "o", "table", "output format (table|json|yaml|wide)")
	rootCmd.PersistentFlags().BoolVar(&noColor, "no-color", false, "disable colored output")
	rootCmd.PersistentFlags().StringVarP(&clusterName, "cluster", "c", "", "target cluster name")
	rootCmd.PersistentFlags().BoolVar(&insecure, "insecure", false, "skip TLS certificate verification")

	// Bind flags to viper
	viper.BindPFlag("output", rootCmd.PersistentFlags().Lookup("output"))
	viper.BindPFlag("no-color", rootCmd.PersistentFlags().Lookup("no-color"))
	viper.BindPFlag("cluster", rootCmd.PersistentFlags().Lookup("cluster"))
	viper.BindPFlag("insecure", rootCmd.PersistentFlags().Lookup("insecure"))

	// Add subcommands
	rootCmd.AddCommand(
		NewVersionCommand(version, gitCommit, buildDate),
		NewLoginCommand(),
		NewConfigCommand(),
		NewVMCommand(),
		NewNodeCommand(),
		NewClusterCommand(),
		NewMigrateCommand(),
		NewSnapshotCommand(),
		NewMonitorCommand(),
		NewLogsCommand(),
		NewExecCommand(),
		NewCopyCommand(),
		NewPortForwardCommand(),
		NewTopCommand(),
		NewApplyCommand(),
		NewDeleteCommand(),
		NewDescribeCommand(),
		NewGetCommand(),
		NewCreateCommand(),
		NewUpdateCommand(),
		NewScaleCommand(),
		NewRolloutCommand(),
		NewCompletionCommand(),
		NewPluginCommand(),
	)

	return rootCmd
}

func initConfig() error {
	if cfgFile != "" {
		// Use config file from flag
		viper.SetConfigFile(cfgFile)
	} else {
		// Find home directory
		home, err := homedir.Dir()
		if err != nil {
			return err
		}

		// Search config in home directory
		configPath := filepath.Join(home, ".novacron")
		viper.AddConfigPath(configPath)
		viper.SetConfigName("config")
		viper.SetConfigType("yaml")

		// Create config directory if it doesn't exist
		if err := os.MkdirAll(configPath, 0755); err != nil {
			return err
		}
	}

	// Set environment variable prefix
	viper.SetEnvPrefix("NOVACRON")
	viper.AutomaticEnv()

	// Read config file
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return err
		}
		// Config file not found is okay, we'll use defaults
	}

	return nil
}