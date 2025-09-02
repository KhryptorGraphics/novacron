package commands

import (
	"fmt"
	"runtime"

	"github.com/novacron/cli/pkg/output"
	"github.com/spf13/cobra"
)

// VersionInfo contains version information
type VersionInfo struct {
	Version   string `json:"version" yaml:"version"`
	GitCommit string `json:"gitCommit" yaml:"gitCommit"`
	BuildDate string `json:"buildDate" yaml:"buildDate"`
	GoVersion string `json:"goVersion" yaml:"goVersion"`
	OS        string `json:"os" yaml:"os"`
	Arch      string `json:"arch" yaml:"arch"`
	APIVersion string `json:"apiVersion" yaml:"apiVersion"`
}

// NewVersionCommand creates the version command
func NewVersionCommand(version, gitCommit, buildDate string) *cobra.Command {
	var short bool

	cmd := &cobra.Command{
		Use:   "version",
		Short: "Print the version information",
		Long:  "Print detailed version information about the NovaCron CLI",
		RunE: func(cmd *cobra.Command, args []string) error {
			if short {
				fmt.Println(version)
				return nil
			}

			info := VersionInfo{
				Version:    version,
				GitCommit:  gitCommit,
				BuildDate:  buildDate,
				GoVersion:  runtime.Version(),
				OS:         runtime.GOOS,
				Arch:       runtime.GOARCH,
				APIVersion: "v1alpha1",
			}

			printer := output.NewPrinter(output.GetFormat())
			return printer.Print(info)
		},
	}

	cmd.Flags().BoolVar(&short, "short", false, "print version number only")

	return cmd
}