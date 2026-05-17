package commands

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPluginListShowsSharedObjects(t *testing.T) {
	pluginDir := t.TempDir()
	writePluginFile(t, filepath.Join(pluginDir, "metrics.so"))
	writePluginFile(t, filepath.Join(pluginDir, "network.so"))
	if err := os.WriteFile(filepath.Join(pluginDir, "README.md"), []byte("ignore"), 0o600); err != nil {
		t.Fatalf("write ignored file: %v", err)
	}

	var output bytes.Buffer
	cmd := NewPluginCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"list", "--dir", pluginDir})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("plugin list failed: %v", err)
	}

	for _, expected := range []string{"NAME", "PATH", "metrics", "network", "metrics.so", "network.so"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected plugin list output to contain %q, got:\n%s", expected, output.String())
		}
	}
	if strings.Contains(output.String(), "README") {
		t.Fatalf("expected non-plugin file to be ignored, got:\n%s", output.String())
	}
}

func TestPluginListHandlesEmptyDirectory(t *testing.T) {
	var output bytes.Buffer
	cmd := NewPluginCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"list", "--dir", t.TempDir()})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("plugin list failed: %v", err)
	}
	if !strings.Contains(output.String(), "No plugins found") {
		t.Fatalf("expected empty plugin directory message, got:\n%s", output.String())
	}
}

func TestPluginListReportsMissingDirectory(t *testing.T) {
	cmd := NewPluginCommand()
	cmd.SetArgs([]string{"list", "--dir", filepath.Join(t.TempDir(), "missing")})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "plugin directory") {
		t.Fatalf("expected missing directory error, got %v", err)
	}
}

func writePluginFile(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, []byte("plugin"), 0o600); err != nil {
		t.Fatalf("write plugin file: %v", err)
	}
}
