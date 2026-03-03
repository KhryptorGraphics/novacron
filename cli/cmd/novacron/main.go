package main

import (
	"fmt"
	"os"

	"github.com/novacron/cli/internal/commands"
	"github.com/sirupsen/logrus"
)

var (
	// Version information (set by build flags)
	Version   = "dev"
	GitCommit = "unknown"
	BuildDate = "unknown"
)

func main() {
	// Initialize logger
	logrus.SetFormatter(&logrus.TextFormatter{
		DisableColors: false,
		FullTimestamp: true,
	})

	// Create root command
	rootCmd := commands.NewRootCommand(Version, GitCommit, BuildDate)

	// Execute command
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}