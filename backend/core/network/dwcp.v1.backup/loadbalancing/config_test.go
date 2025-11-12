package loadbalancing

import (
	"testing"
	"time"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.Algorithm != "geo-proximity" {
		t.Errorf("Expected algorithm 'geo-proximity', got '%s'", config.Algorithm)
	}

	if config.HealthCheckInterval != 5*time.Second {
		t.Errorf("Expected health check interval 5s, got %v", config.HealthCheckInterval)
	}

	if config.FailoverTimeout > 100*time.Millisecond {
		t.Errorf("Failover timeout exceeds 100ms: %v", config.FailoverTimeout)
	}
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    *LoadBalancerConfig
		expectErr bool
	}{
		{
			name:      "Valid config",
			config:    DefaultConfig(),
			expectErr: false,
		},
		{
			name: "Invalid health check interval",
			config: &LoadBalancerConfig{
				HealthCheckInterval: 500 * time.Millisecond,
				UnhealthyThreshold:  3,
				HealthyThreshold:    2,
				ConnectionTimeout:   2 * time.Second,
				MaxConnections:      1000,
				FailoverTimeout:     100 * time.Millisecond,
			},
			expectErr: true,
		},
		{
			name: "Invalid unhealthy threshold",
			config: &LoadBalancerConfig{
				HealthCheckInterval: 5 * time.Second,
				UnhealthyThreshold:  0,
				HealthyThreshold:    2,
				ConnectionTimeout:   2 * time.Second,
				MaxConnections:      1000,
				FailoverTimeout:     100 * time.Millisecond,
			},
			expectErr: true,
		},
		{
			name: "Failover timeout too high",
			config: &LoadBalancerConfig{
				HealthCheckInterval: 5 * time.Second,
				UnhealthyThreshold:  3,
				HealthyThreshold:    2,
				ConnectionTimeout:   2 * time.Second,
				MaxConnections:      1000,
				FailoverTimeout:     2 * time.Second,
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.expectErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}
