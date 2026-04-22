// Package config provides configuration loading tests
package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLoader_Load(t *testing.T) {
	tests := []struct {
		name        string
		configYAML  string
		wantErr     bool
		errContains string
	}{
		{
			name: "valid configuration",
			configYAML: `
system:
  node_id: "test-node-1"
  data_dir: "/tmp/novacron"
  log_level: "info"
`,
			wantErr: false,
		},
		{
			name: "missing required fields",
			configYAML: `
system:
  log_level: "info"
`,
			wantErr:     true,
			errContains: "node_id is required",
		},
		{
			name: "invalid log level",
			configYAML: `
system:
  node_id: "test"
  data_dir: "/tmp"
  log_level: "invalid"
`,
			wantErr:     true,
			errContains: "invalid log_level",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temp config file
			tmpDir := t.TempDir()
			configPath := filepath.Join(tmpDir, "config.yaml")

			if err := os.WriteFile(configPath, []byte(tt.configYAML), 0644); err != nil {
				t.Fatalf("Failed to write config: %v", err)
			}

			loader := NewLoader(configPath)
			cfg, err := loader.Load()

			if tt.wantErr {
				if err == nil {
					t.Fatal("Expected error but got none")
				}
				if tt.errContains != "" && !contains(err.Error(), tt.errContains) {
					t.Errorf("Error %q does not contain %q", err.Error(), tt.errContains)
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			if cfg == nil {
				t.Fatal("Config is nil")
			}
		})
	}
}

func TestLoader_ApplyDefaults(t *testing.T) {
	loader := NewLoader("")
	cfg := &Config{}

	if err := loader.applyDefaults(cfg); err != nil {
		t.Fatalf("Failed to apply defaults: %v", err)
	}

	// Check system defaults
	if cfg.System.LogLevel != "info" {
		t.Errorf("Expected log_level=info, got %s", cfg.System.LogLevel)
	}

	if cfg.System.MaxConcurrency != 1000 {
		t.Errorf("Expected max_concurrency=1000, got %d", cfg.System.MaxConcurrency)
	}

	if cfg.System.ShutdownTimeout != 30*time.Second {
		t.Errorf("Expected shutdown_timeout=30s, got %v", cfg.System.ShutdownTimeout)
	}

	if cfg.Runtime.Version != DefaultRuntimeManifestVersion {
		t.Errorf("Expected runtime.version=%s, got %s", DefaultRuntimeManifestVersion, cfg.Runtime.Version)
	}

	if cfg.Runtime.DeploymentProfile != "single-node" {
		t.Errorf("Expected runtime.deployment_profile=single-node, got %s", cfg.Runtime.DeploymentProfile)
	}

	if !containsString(cfg.Runtime.EnabledServices, "api") {
		t.Fatalf("Expected runtime.enabled_services to contain api, got %#v", cfg.Runtime.EnabledServices)
	}

	// Check DWCP defaults
	if cfg.DWCP.Transport.MinStreams != 4 {
		t.Errorf("Expected min_streams=4, got %d", cfg.DWCP.Transport.MinStreams)
	}

	if cfg.DWCP.Compression.Algorithm != "zstd" {
		t.Errorf("Expected compression=zstd, got %s", cfg.DWCP.Compression.Algorithm)
	}
}

func TestLoader_Validate(t *testing.T) {
	tests := []struct {
		name        string
		config      Config
		wantErr     bool
		errContains string
	}{
		{
			name: "valid config",
			config: Config{
				System: SystemConfig{
					NodeID:   "test",
					DataDir:  "/tmp",
					LogLevel: "info",
				},
				Network: NetworkConfig{
					BindPort: 9090,
				},
				Storage: StorageConfig{
					Backend: "sqlite",
				},
			},
			wantErr: false,
		},
		{
			name: "missing node_id",
			config: Config{
				System: SystemConfig{
					DataDir:  "/tmp",
					LogLevel: "info",
				},
			},
			wantErr:     true,
			errContains: "node_id is required",
		},
		{
			name: "invalid bind_port",
			config: Config{
				System: SystemConfig{
					NodeID:   "test",
					DataDir:  "/tmp",
					LogLevel: "info",
				},
				Network: NetworkConfig{
					BindPort: 100,
				},
			},
			wantErr:     true,
			errContains: "invalid bind_port",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loader := NewLoader("")
			err := loader.validate(&tt.config)

			if tt.wantErr {
				if err == nil {
					t.Fatal("Expected error but got none")
				}
				if tt.errContains != "" && !contains(err.Error(), tt.errContains) {
					t.Errorf("Error %q does not contain %q", err.Error(), tt.errContains)
				}
				return
			}

			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
		})
	}
}

func TestGenerateDefault(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	if err := GenerateDefault(configPath); err != nil {
		t.Fatalf("Failed to generate default config: %v", err)
	}

	// Verify file was created
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Fatal("Config file was not created")
	}

	// Load and validate the generated config
	loader := NewLoader(configPath)
	cfg, err := loader.Load()
	if err != nil {
		t.Fatalf("Failed to load generated config: %v", err)
	}

	if cfg.System.NodeID == "" {
		t.Error("Generated config missing node_id")
	}

	if cfg.Runtime.Version != DefaultRuntimeManifestVersion {
		t.Fatalf("Generated config runtime.version=%q, want %q", cfg.Runtime.Version, DefaultRuntimeManifestVersion)
	}
}

func TestLoader_LoadRuntimeManifestFields(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	configYAML := `
system:
  node_id: "node-manifest"
  data_dir: "/tmp/novacron"
runtime:
  version: "v1alpha1"
  deployment_profile: "multi-node"
  discovery_mode: "seeded"
  federation_mode: "trusted"
  migration_mode: "cold"
  auth_mode: "external"
  storage_classes:
    - "default"
    - "fast-ssd"
  enabled_services:
    - "api"
    - "auth"
    - "vm"
`

	if err := os.WriteFile(configPath, []byte(configYAML), 0o644); err != nil {
		t.Fatalf("Failed to write config: %v", err)
	}

	loader := NewLoader(configPath)
	cfg, err := loader.Load()
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}

	if got, want := cfg.Runtime.DeploymentProfile, "multi-node"; got != want {
		t.Fatalf("runtime.deployment_profile=%q, want %q", got, want)
	}
	if got, want := cfg.Runtime.DiscoveryMode, "seeded"; got != want {
		t.Fatalf("runtime.discovery_mode=%q, want %q", got, want)
	}
	if got, want := cfg.Runtime.AuthMode, "external"; got != want {
		t.Fatalf("runtime.auth_mode=%q, want %q", got, want)
	}
	if !containsString(cfg.Runtime.StorageClasses, "fast-ssd") {
		t.Fatalf("runtime.storage_classes missing fast-ssd: %#v", cfg.Runtime.StorageClasses)
	}
	if !containsString(cfg.Runtime.EnabledServices, "vm") {
		t.Fatalf("runtime.enabled_services missing vm: %#v", cfg.Runtime.EnabledServices)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr || len(s) > len(substr) && contains(s[1:], substr)
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
