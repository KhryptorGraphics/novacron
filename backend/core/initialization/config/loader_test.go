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
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr || len(s) > len(substr) && contains(s[1:], substr)
}
