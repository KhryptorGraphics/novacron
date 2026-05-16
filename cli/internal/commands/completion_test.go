package commands

import (
	"bytes"
	"strings"
	"testing"
)

func TestCompletionCommandGeneratesBash(t *testing.T) {
	var output bytes.Buffer
	root := NewRootCommand("test", "commit", "date")
	root.SetOut(&output)
	root.SetArgs([]string{"completion", "bash"})

	if err := root.Execute(); err != nil {
		t.Fatalf("completion bash failed: %v", err)
	}

	for _, expected := range []string{"# bash completion for", "novacron"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected bash completion to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestCompletionCommandRejectsUnsupportedShell(t *testing.T) {
	root := NewRootCommand("test", "commit", "date")
	root.SetOut(&bytes.Buffer{})
	root.SetErr(&bytes.Buffer{})
	root.SetArgs([]string{"completion", "cmd"})

	err := root.Execute()
	if err == nil || !strings.Contains(err.Error(), "unsupported shell") {
		t.Fatalf("expected unsupported shell error, got %v", err)
	}
}
