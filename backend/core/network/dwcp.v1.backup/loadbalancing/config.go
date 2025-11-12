// Package loadbalancing provides global load balancing with geographic awareness
// for NovaCron's DWCP Phase 3 multi-region deployment.
package loadbalancing

import (
	"time"
)

// LoadBalancerConfig defines the configuration for the global load balancer
type LoadBalancerConfig struct {
	// Algorithm specifies the load balancing algorithm to use
	// Supported: "geo-proximity", "least-latency", "round-robin", "weighted-round-robin",
	// "least-connections", "ip-hash"
	Algorithm string `json:"algorithm" yaml:"algorithm"`

	// HealthCheckInterval is the interval between active health checks
	HealthCheckInterval time.Duration `json:"health_check_interval" yaml:"health_check_interval"`

	// PassiveHealthCheckInterval for passive health monitoring
	PassiveHealthCheckInterval time.Duration `json:"passive_health_check_interval" yaml:"passive_health_check_interval"`

	// UnhealthyThreshold is the number of consecutive failed health checks
	// before marking a server as unhealthy
	UnhealthyThreshold int `json:"unhealthy_threshold" yaml:"unhealthy_threshold"`

	// HealthyThreshold is the number of consecutive successful health checks
	// before marking a server as healthy again
	HealthyThreshold int `json:"healthy_threshold" yaml:"healthy_threshold"`

	// ConnectionTimeout is the timeout for establishing connections to backend servers
	ConnectionTimeout time.Duration `json:"connection_timeout" yaml:"connection_timeout"`

	// MaxConnections is the maximum number of concurrent connections per server
	MaxConnections int `json:"max_connections" yaml:"max_connections"`

	// SessionAffinityTTL is the time-to-live for session affinity entries
	SessionAffinityTTL time.Duration `json:"session_affinity_ttl" yaml:"session_affinity_ttl"`

	// EnableSessionAffinity enables sticky sessions
	EnableSessionAffinity bool `json:"enable_session_affinity" yaml:"enable_session_affinity"`

	// VirtualNodesPerServer for consistent hashing
	VirtualNodesPerServer int `json:"virtual_nodes_per_server" yaml:"virtual_nodes_per_server"`

	// FailoverTimeout is the maximum time to wait for failover
	FailoverTimeout time.Duration `json:"failover_timeout" yaml:"failover_timeout"`

	// MaxRetries for failed requests
	MaxRetries int `json:"max_retries" yaml:"max_retries"`

	// CircuitBreakerThreshold is the error rate threshold to trip circuit breaker
	CircuitBreakerThreshold float64 `json:"circuit_breaker_threshold" yaml:"circuit_breaker_threshold"`

	// CircuitBreakerTimeout is the timeout before attempting recovery
	CircuitBreakerTimeout time.Duration `json:"circuit_breaker_timeout" yaml:"circuit_breaker_timeout"`

	// EnableGeoRouting enables geographic proximity routing
	EnableGeoRouting bool `json:"enable_geo_routing" yaml:"enable_geo_routing"`

	// GeoIPDatabasePath is the path to the GeoIP database
	GeoIPDatabasePath string `json:"geoip_database_path" yaml:"geoip_database_path"`

	// DrainTimeout is the timeout for connection draining during server removal
	DrainTimeout time.Duration `json:"drain_timeout" yaml:"drain_timeout"`

	// MetricsInterval is the interval for collecting metrics
	MetricsInterval time.Duration `json:"metrics_interval" yaml:"metrics_interval"`
}

// DefaultConfig returns the default load balancer configuration
func DefaultConfig() *LoadBalancerConfig {
	return &LoadBalancerConfig{
		Algorithm:                  "geo-proximity",
		HealthCheckInterval:        5 * time.Second,
		PassiveHealthCheckInterval: 30 * time.Second,
		UnhealthyThreshold:         3,
		HealthyThreshold:           2,
		ConnectionTimeout:          2 * time.Second,
		MaxConnections:             100000,
		SessionAffinityTTL:         30 * time.Minute,
		EnableSessionAffinity:      true,
		VirtualNodesPerServer:      150,
		FailoverTimeout:            100 * time.Millisecond,
		MaxRetries:                 3,
		CircuitBreakerThreshold:    0.5,
		CircuitBreakerTimeout:      30 * time.Second,
		EnableGeoRouting:           true,
		GeoIPDatabasePath:          "/var/lib/geoip/GeoLite2-City.mmdb",
		DrainTimeout:               30 * time.Second,
		MetricsInterval:            10 * time.Second,
	}
}

// Validate validates the configuration
func (c *LoadBalancerConfig) Validate() error {
	if c.HealthCheckInterval < time.Second {
		return ErrInvalidHealthCheckInterval
	}
	if c.UnhealthyThreshold < 1 {
		return ErrInvalidUnhealthyThreshold
	}
	if c.HealthyThreshold < 1 {
		return ErrInvalidHealthyThreshold
	}
	if c.ConnectionTimeout < 100*time.Millisecond {
		return ErrInvalidConnectionTimeout
	}
	if c.MaxConnections < 1 {
		return ErrInvalidMaxConnections
	}
	if c.FailoverTimeout > 1*time.Second {
		return ErrFailoverTimeoutTooHigh
	}
	return nil
}

// HealthCheckType defines the type of health check
type HealthCheckType string

const (
	// HealthCheckTCP performs TCP connection check
	HealthCheckTCP HealthCheckType = "tcp"
	// HealthCheckHTTP performs HTTP endpoint check
	HealthCheckHTTP HealthCheckType = "http"
	// HealthCheckHTTPS performs HTTPS endpoint check
	HealthCheckHTTPS HealthCheckType = "https"
	// HealthCheckCustom performs custom application health check
	HealthCheckCustom HealthCheckType = "custom"
)

// HealthCheckConfig defines health check configuration
type HealthCheckConfig struct {
	Type         HealthCheckType   `json:"type" yaml:"type"`
	Endpoint     string            `json:"endpoint" yaml:"endpoint"`
	Interval     time.Duration     `json:"interval" yaml:"interval"`
	Timeout      time.Duration     `json:"timeout" yaml:"timeout"`
	ExpectedCode int               `json:"expected_code" yaml:"expected_code"`
	Headers      map[string]string `json:"headers" yaml:"headers"`
}

// DefaultHealthCheckConfig returns default health check configuration
func DefaultHealthCheckConfig() *HealthCheckConfig {
	return &HealthCheckConfig{
		Type:         HealthCheckHTTP,
		Endpoint:     "/health",
		Interval:     5 * time.Second,
		Timeout:      2 * time.Second,
		ExpectedCode: 200,
		Headers:      make(map[string]string),
	}
}
