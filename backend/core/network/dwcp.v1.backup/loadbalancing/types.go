package loadbalancing

import (
	"sync"
	"time"
)

// ServerHealth represents the health state of a backend server
type ServerHealth string

const (
	// ServerHealthy indicates the server is healthy and accepting requests
	ServerHealthy ServerHealth = "healthy"
	// ServerDegraded indicates the server is partially healthy
	ServerDegraded ServerHealth = "degraded"
	// ServerUnhealthy indicates the server is unhealthy
	ServerUnhealthy ServerHealth = "unhealthy"
	// ServerDraining indicates the server is draining connections
	ServerDraining ServerHealth = "draining"
)

// Server represents a backend server in the pool
type Server struct {
	ID       string  `json:"id"`
	Address  string  `json:"address"`
	Port     int     `json:"port"`
	Region   string  `json:"region"`
	Latitude float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Weight   int     `json:"weight"`

	// Health tracking
	Health            ServerHealth  `json:"health"`
	HealthScore       float64       `json:"health_score"`
	ConsecutiveFailures int         `json:"consecutive_failures"`
	ConsecutiveSuccesses int        `json:"consecutive_successes"`
	LastHealthCheck   time.Time     `json:"last_health_check"`

	// Performance metrics
	ActiveConnections int32         `json:"active_connections"`
	TotalRequests     uint64        `json:"total_requests"`
	TotalFailures     uint64        `json:"total_failures"`
	AvgResponseTime   time.Duration `json:"avg_response_time"`
	CPUUtilization    float64       `json:"cpu_utilization"`
	MemoryUtilization float64       `json:"memory_utilization"`

	// Circuit breaker state
	CircuitBreakerState CircuitBreakerState `json:"circuit_breaker_state"`
	CircuitOpenedAt     time.Time           `json:"circuit_opened_at"`

	mu sync.RWMutex
}

// CircuitBreakerState represents the circuit breaker state
type CircuitBreakerState string

const (
	// CircuitClosed allows requests to pass through
	CircuitClosed CircuitBreakerState = "closed"
	// CircuitOpen blocks requests
	CircuitOpen CircuitBreakerState = "open"
	// CircuitHalfOpen allows limited requests for testing
	CircuitHalfOpen CircuitBreakerState = "half-open"
)

// GeoLocation represents a geographic location
type GeoLocation struct {
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
	Region    string  `json:"region"`
	Country   string  `json:"country"`
	City      string  `json:"city"`
}

// SessionAffinity represents session affinity information
type SessionAffinity struct {
	SessionID string    `json:"session_id"`
	ServerID  string    `json:"server_id"`
	CreatedAt time.Time `json:"created_at"`
	ExpiresAt time.Time `json:"expires_at"`
	RequestCount uint64 `json:"request_count"`
}

// RoutingDecision represents the result of a routing decision
type RoutingDecision struct {
	Server      *Server       `json:"server"`
	Algorithm   string        `json:"algorithm"`
	Latency     time.Duration `json:"latency"`
	Timestamp   time.Time     `json:"timestamp"`
	ReasonCode  string        `json:"reason_code"`
	IsFailover  bool          `json:"is_failover"`
}

// LoadBalancerStats represents statistics for the load balancer
type LoadBalancerStats struct {
	TotalRequests       uint64        `json:"total_requests"`
	TotalFailures       uint64        `json:"total_failures"`
	TotalFailovers      uint64        `json:"total_failovers"`
	AvgRoutingLatency   time.Duration `json:"avg_routing_latency"`
	AvgFailoverTime     time.Duration `json:"avg_failover_time"`
	HealthyServers      int           `json:"healthy_servers"`
	DegradedServers     int           `json:"degraded_servers"`
	UnhealthyServers    int           `json:"unhealthy_servers"`
	DrainingServers     int           `json:"draining_servers"`
	ActiveSessions      int           `json:"active_sessions"`
	TotalConnections    int32         `json:"total_connections"`
	RequestsPerSecond   float64       `json:"requests_per_second"`
	P50ResponseTime     time.Duration `json:"p50_response_time"`
	P95ResponseTime     time.Duration `json:"p95_response_time"`
	P99ResponseTime     time.Duration `json:"p99_response_time"`

	mu sync.RWMutex
}

// Algorithm types
const (
	AlgorithmRoundRobin         = "round-robin"
	AlgorithmWeightedRoundRobin = "weighted-round-robin"
	AlgorithmLeastConnections   = "least-connections"
	AlgorithmLeastLatency       = "least-latency"
	AlgorithmGeoProximity       = "geo-proximity"
	AlgorithmIPHash             = "ip-hash"
)
