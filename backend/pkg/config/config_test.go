package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadIncludesRuntimeManifestSummary(t *testing.T) {
	t.Setenv("AUTH_SECRET", "test-secret")

	manifestPath := filepath.Join(t.TempDir(), "runtime.yaml")
	manifestYAML := `
system:
  node_id: "node-a"
  data_dir: "/tmp/novacron"
runtime:
  version: "v1alpha1"
  deployment_profile: "multi-node"
  discovery_mode: "seeded"
  federation_mode: "trusted"
  migration_mode: "cold"
  auth_mode: "runtime"
  storage_classes:
    - "default"
    - "fast-ssd"
  enabled_services:
    - "api"
    - "auth"
    - "vm"
    - "scheduler"
`
	if err := os.WriteFile(manifestPath, []byte(manifestYAML), 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	t.Setenv("NOVACRON_RUNTIME_MANIFEST_PATH", manifestPath)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}

	if !cfg.RuntimeManifest.Loaded {
		t.Fatal("expected runtime manifest to be loaded")
	}
	if got, want := cfg.RuntimeManifest.Version, "v1alpha1"; got != want {
		t.Fatalf("runtime manifest version = %q, want %q", got, want)
	}
	if got, want := cfg.RuntimeManifest.DeploymentProfile, "multi-node"; got != want {
		t.Fatalf("runtime deployment profile = %q, want %q", got, want)
	}
	if got, want := cfg.RuntimeManifest.DiscoveryMode, "seeded"; got != want {
		t.Fatalf("runtime discovery mode = %q, want %q", got, want)
	}
	if !containsValue(cfg.RuntimeManifest.EnabledServices, "scheduler") {
		t.Fatalf("expected enabled services to include scheduler, got %#v", cfg.RuntimeManifest.EnabledServices)
	}
}

func TestLoadRequiresManifestPathWhenRequested(t *testing.T) {
	t.Setenv("AUTH_SECRET", "test-secret")
	t.Setenv("NOVACRON_REQUIRE_RUNTIME_MANIFEST", "true")

	if _, err := Load(); err == nil {
		t.Fatal("expected error when runtime manifest is required without a path")
	}
}

func containsValue(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
