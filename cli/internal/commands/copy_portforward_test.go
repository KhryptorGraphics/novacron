package commands

import (
	"strings"
	"testing"
)

func TestCopyCommandReportsMissingBackendContract(t *testing.T) {
	cmd := NewCopyCommand()
	cmd.SetArgs([]string{"local.txt", "vm-1:/tmp/local.txt"})

	err := cmd.Execute()
	if err == nil {
		t.Fatalf("expected copy contract error")
	}
	for _, expected := range []string{
		"backend contract is not implemented",
		"docs/api/vm-io-contracts.md",
		"novacron-lmh",
	} {
		if !strings.Contains(err.Error(), expected) {
			t.Fatalf("expected error to contain %q, got %v", expected, err)
		}
	}
}

func TestPortForwardCommandReportsMissingBackendContract(t *testing.T) {
	cmd := NewPortForwardCommand()
	cmd.SetArgs([]string{"vm-1", "8080:80"})

	err := cmd.Execute()
	if err == nil {
		t.Fatalf("expected port-forward contract error")
	}
	for _, expected := range []string{
		"backend contract is not implemented",
		"docs/api/vm-io-contracts.md",
		"novacron-lmh",
	} {
		if !strings.Contains(err.Error(), expected) {
			t.Fatalf("expected error to contain %q, got %v", expected, err)
		}
	}
}
