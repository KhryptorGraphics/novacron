package resilience

import (
	"context"
	"fmt"
	"time"

	"go.uber.org/zap"
)

// Example demonstrates how to use the resilience package with DWCP

// DWCPResilienceConfig holds configuration for DWCP resilience
type DWCPResilienceConfig struct {
	// Circuit Breaker Config
	MaxFailures       int
	CircuitTimeout    time.Duration
	CircuitResetTime  time.Duration

	// Retry Config
	MaxRetries        int
	InitialRetryDelay time.Duration
	MaxRetryDelay     time.Duration
	RetryMultiplier   float64

	// Health Check Config
	HealthCheckInterval time.Duration
	HealthCheckTimeout  time.Duration

	// Rate Limiter Config
	RequestsPerSecond float64
	BurstSize         int

	// Bulkhead Config
	MaxConcurrent int
	MaxQueueSize  int
	MaxWaitTime   time.Duration
}

// DefaultDWCPResilienceConfig returns default configuration
func DefaultDWCPResilienceConfig() DWCPResilienceConfig {
	return DWCPResilienceConfig{
		// Circuit Breaker: Open after 5 failures, reset after 60s
		MaxFailures:      5,
		CircuitTimeout:   30 * time.Second,
		CircuitResetTime: 60 * time.Second,

		// Retry: 3 attempts with exponential backoff
		MaxRetries:        3,
		InitialRetryDelay: 100 * time.Millisecond,
		MaxRetryDelay:     10 * time.Second,
		RetryMultiplier:   2.0,

		// Health Check: Every 10s with 5s timeout
		HealthCheckInterval: 10 * time.Second,
		HealthCheckTimeout:  5 * time.Second,

		// Rate Limiter: 1000 RPS with burst of 100
		RequestsPerSecond: 1000,
		BurstSize:         100,

		// Bulkhead: 100 concurrent with 50 queue and 5s max wait
		MaxConcurrent: 100,
		MaxQueueSize:  50,
		MaxWaitTime:   5 * time.Second,
	}
}

// DWCPResilientClient demonstrates a resilient DWCP client
type DWCPResilientClient struct {
	resilienceManager *ResilienceManager
	logger            *zap.Logger
}

// NewDWCPResilientClient creates a new resilient DWCP client
func NewDWCPResilientClient(config DWCPResilienceConfig, logger *zap.Logger) *DWCPResilientClient {
	if logger == nil {
		logger = zap.NewNop()
	}

	// Initialize Prometheus metrics
	InitMetrics("dwcp")

	// Create resilience manager
	rm := NewResilienceManager("dwcp", logger)

	// Register circuit breakers for different operations
	rm.RegisterCircuitBreaker("network-send",
		config.MaxFailures,
		config.CircuitTimeout,
		config.CircuitResetTime)

	rm.RegisterCircuitBreaker("network-receive",
		config.MaxFailures,
		config.CircuitTimeout,
		config.CircuitResetTime)

	rm.RegisterCircuitBreaker("peer-discovery",
		config.MaxFailures/2, // More sensitive for discovery
		config.CircuitTimeout,
		config.CircuitResetTime)

	// Register retry policies
	rm.RegisterRetryPolicy("network-retry",
		config.MaxRetries,
		config.InitialRetryDelay,
		config.MaxRetryDelay,
		config.RetryMultiplier,
		true) // Enable jitter

	// Register rate limiters
	rm.RegisterRateLimiter("outbound-messages",
		config.RequestsPerSecond,
		config.BurstSize)

	// Register bulkheads
	rm.RegisterBulkhead("concurrent-connections",
		config.MaxConcurrent,
		config.MaxQueueSize,
		config.MaxWaitTime)

	// Register error budgets
	rm.RegisterErrorBudget("slo-availability",
		0.999,              // 99.9% availability SLO
		24*time.Hour)       // 24h window

	rm.RegisterErrorBudget("slo-latency",
		0.99,               // 99% latency SLO
		time.Hour)          // 1h window

	// Register latency budgets
	rm.RegisterLatencyBudget("p95-latency",
		100*time.Millisecond, // 100ms target
		0.95,                  // 95th percentile
		1000)                  // Sample size

	// Register health checks
	rm.RegisterHealthCheck(NewPingHealthCheck("network-connectivity",
		func(ctx context.Context) error {
			// Check network connectivity
			return nil
		}))

	rm.RegisterHealthCheck(NewThresholdHealthCheck("connection-pool",
		func() float64 {
			// Return current connection pool utilization
			return 0.7
		},
		0, 0.9)) // Alert if > 90% utilization

	// Start health monitoring
	rm.StartHealthMonitoring()

	return &DWCPResilientClient{
		resilienceManager: rm,
		logger:            logger,
	}
}

// SendMessage demonstrates sending a message with full resilience
func (c *DWCPResilientClient) SendMessage(ctx context.Context, message []byte) error {
	return c.resilienceManager.ExecuteWithAllProtections(ctx, "network-send",
		func(ctx context.Context) error {
			// Actual send logic
			c.logger.Debug("Sending message", zap.Int("size", len(message)))

			// Simulate network operation
			time.Sleep(10 * time.Millisecond)

			return nil
		})
}

// ReceiveMessage demonstrates receiving a message with circuit breaker
func (c *DWCPResilientClient) ReceiveMessage(ctx context.Context) ([]byte, error) {
	var result []byte
	var err error

	err = c.resilienceManager.ExecuteWithCircuitBreaker("network-receive",
		func() error {
			// Actual receive logic
			result = []byte("received message")
			return nil
		})

	return result, err
}

// DiscoverPeers demonstrates peer discovery with retry logic
func (c *DWCPResilientClient) DiscoverPeers() ([]string, error) {
	var peers []string
	var err error

	err = c.resilienceManager.ExecuteWithRetry("network-retry",
		func() error {
			// Actual discovery logic
			peers = []string{"peer1", "peer2", "peer3"}
			return nil
		})

	return peers, err
}

// EstablishConnection demonstrates connection with bulkhead
func (c *DWCPResilientClient) EstablishConnection(peerID string) error {
	return c.resilienceManager.ExecuteWithBulkhead("concurrent-connections",
		func() error {
			c.logger.Info("Establishing connection", zap.String("peer", peerID))

			// Simulate connection establishment
			time.Sleep(50 * time.Millisecond)

			return nil
		})
}

// GetHealthStatus returns the current health status
func (c *DWCPResilientClient) GetHealthStatus() bool {
	return c.resilienceManager.IsHealthy()
}

// GetMetrics returns comprehensive resilience metrics
func (c *DWCPResilientClient) GetMetrics() ResilienceMetrics {
	return c.resilienceManager.GetAllMetrics()
}

// Shutdown gracefully shuts down the resilient client
func (c *DWCPResilientClient) Shutdown() {
	c.resilienceManager.StopHealthMonitoring()
	c.logger.Info("Resilient client shutdown complete")
}

// ExampleUsage demonstrates how to use the resilient client
func ExampleUsage() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Create resilient client with default config
	config := DefaultDWCPResilienceConfig()
	client := NewDWCPResilientClient(config, logger)
	defer client.Shutdown()

	ctx := context.Background()

	// Send message with full protection
	err := client.SendMessage(ctx, []byte("Hello, DWCP!"))
	if err != nil {
		logger.Error("Failed to send message", zap.Error(err))
	}

	// Receive message with circuit breaker
	msg, err := client.ReceiveMessage(ctx)
	if err != nil {
		logger.Error("Failed to receive message", zap.Error(err))
	} else {
		logger.Info("Received message", zap.String("msg", string(msg)))
	}

	// Discover peers with retry
	peers, err := client.DiscoverPeers()
	if err != nil {
		logger.Error("Failed to discover peers", zap.Error(err))
	} else {
		logger.Info("Discovered peers", zap.Int("count", len(peers)))
	}

	// Establish connection with bulkhead
	err = client.EstablishConnection("peer1")
	if err != nil {
		logger.Error("Failed to establish connection", zap.Error(err))
	}

	// Check health
	if client.GetHealthStatus() {
		logger.Info("System is healthy")
	} else {
		logger.Warn("System is unhealthy")
	}

	// Get metrics
	metrics := client.GetMetrics()
	logger.Info("Resilience metrics",
		zap.String("component", metrics.Name),
		zap.Bool("healthy", metrics.HealthChecker.IsHealthy))
}

// ExampleCustomCircuitBreaker shows advanced circuit breaker usage
func ExampleCustomCircuitBreaker() {
	logger, _ := zap.NewProduction()

	// Create custom circuit breaker with specific settings
	cb := NewCircuitBreaker(
		"custom-operation",
		5,                  // Max 5 failures
		10*time.Second,     // 10s operation timeout
		30*time.Second,     // 30s reset timeout
		logger,
	)

	// Use circuit breaker
	for i := 0; i < 10; i++ {
		err := cb.Execute(func() error {
			// Your operation here
			if i%3 == 0 {
				return fmt.Errorf("simulated failure %d", i)
			}
			return nil
		})

		if err != nil {
			logger.Error("Operation failed",
				zap.Error(err),
				zap.String("state", cb.GetState().String()))
		}

		time.Sleep(time.Second)
	}

	// Check metrics
	metrics := cb.GetMetrics()
	logger.Info("Circuit breaker metrics",
		zap.String("name", metrics.Name),
		zap.String("state", metrics.State),
		zap.Int64("total_requests", metrics.TotalRequests),
		zap.Float64("success_rate", metrics.SuccessRate))
}
