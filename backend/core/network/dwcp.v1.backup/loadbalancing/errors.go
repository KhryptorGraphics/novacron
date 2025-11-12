package loadbalancing

import "errors"

var (
	// Configuration errors
	ErrInvalidHealthCheckInterval = errors.New("invalid health check interval")
	ErrInvalidUnhealthyThreshold  = errors.New("invalid unhealthy threshold")
	ErrInvalidHealthyThreshold    = errors.New("invalid healthy threshold")
	ErrInvalidConnectionTimeout   = errors.New("invalid connection timeout")
	ErrInvalidMaxConnections      = errors.New("invalid max connections")
	ErrFailoverTimeoutTooHigh     = errors.New("failover timeout exceeds 1 second")

	// Server pool errors
	ErrNoHealthyServers   = errors.New("no healthy servers available")
	ErrServerNotFound     = errors.New("server not found in pool")
	ErrServerAlreadyExists = errors.New("server already exists in pool")
	ErrServerDraining     = errors.New("server is draining connections")

	// Health check errors
	ErrHealthCheckFailed    = errors.New("health check failed")
	ErrHealthCheckTimeout   = errors.New("health check timeout")
	ErrCircuitBreakerOpen   = errors.New("circuit breaker is open")

	// Geographic routing errors
	ErrGeoIPDatabaseNotFound = errors.New("geoip database not found")
	ErrInvalidGeoLocation    = errors.New("invalid geographic location")
	ErrNoRegionMatch         = errors.New("no matching region found")

	// Session affinity errors
	ErrSessionNotFound     = errors.New("session not found")
	ErrAffinityServerDown  = errors.New("affinity server is down")

	// Load balancing errors
	ErrUnsupportedAlgorithm = errors.New("unsupported load balancing algorithm")
	ErrMaxRetriesExceeded   = errors.New("maximum retries exceeded")
)
