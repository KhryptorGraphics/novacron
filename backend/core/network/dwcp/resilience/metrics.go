package resilience

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// PrometheusMetrics holds all Prometheus metrics for resilience components
type PrometheusMetrics struct {
	// Circuit Breaker Metrics
	CircuitBreakerState *prometheus.GaugeVec
	CircuitBreakerRequests *prometheus.CounterVec
	CircuitBreakerFailures *prometheus.CounterVec
	CircuitBreakerStateChanges *prometheus.CounterVec
	CircuitBreakerLatency *prometheus.HistogramVec

	// Retry Metrics
	RetryAttempts *prometheus.CounterVec
	RetrySuccess *prometheus.CounterVec
	RetryFailures *prometheus.CounterVec
	RetryBackoffDelay *prometheus.HistogramVec

	// Health Check Metrics
	HealthCheckStatus *prometheus.GaugeVec
	HealthCheckDuration *prometheus.HistogramVec
	HealthCheckTotal *prometheus.CounterVec
	HealthCheckFailures *prometheus.CounterVec

	// Rate Limiter Metrics
	RateLimiterAllowed *prometheus.CounterVec
	RateLimiterRejected *prometheus.CounterVec
	RateLimiterCurrentRate *prometheus.GaugeVec

	// Bulkhead Metrics
	BulkheadConcurrent *prometheus.GaugeVec
	BulkheadQueued *prometheus.GaugeVec
	BulkheadRejected *prometheus.CounterVec
	BulkheadLatency *prometheus.HistogramVec

	// Error Budget Metrics
	ErrorBudgetRemaining *prometheus.GaugeVec
	ErrorBudgetConsumed *prometheus.GaugeVec
	ErrorBudgetExhausted *prometheus.CounterVec

	// Latency Budget Metrics
	LatencyBudgetP50 *prometheus.GaugeVec
	LatencyBudgetP95 *prometheus.GaugeVec
	LatencyBudgetP99 *prometheus.GaugeVec
	LatencyBudgetViolations *prometheus.CounterVec
}

// NewPrometheusMetrics creates and registers all Prometheus metrics
func NewPrometheusMetrics(namespace string) *PrometheusMetrics {
	return &PrometheusMetrics{
		// Circuit Breaker
		CircuitBreakerState: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "circuit_breaker_state",
			Help:      "Circuit breaker state (0=closed, 1=half-open, 2=open)",
		}, []string{"name"}),

		CircuitBreakerRequests: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "circuit_breaker_requests_total",
			Help:      "Total number of requests through circuit breaker",
		}, []string{"name", "result"}),

		CircuitBreakerFailures: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "circuit_breaker_failures_total",
			Help:      "Total number of circuit breaker failures",
		}, []string{"name"}),

		CircuitBreakerStateChanges: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "circuit_breaker_state_changes_total",
			Help:      "Total number of circuit breaker state changes",
		}, []string{"name", "from_state", "to_state"}),

		CircuitBreakerLatency: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "circuit_breaker_request_duration_seconds",
			Help:      "Circuit breaker request duration in seconds",
			Buckets:   prometheus.DefBuckets,
		}, []string{"name"}),

		// Retry
		RetryAttempts: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "retry_attempts_total",
			Help:      "Total number of retry attempts",
		}, []string{"policy", "attempt"}),

		RetrySuccess: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "retry_success_total",
			Help:      "Total number of successful retries",
		}, []string{"policy"}),

		RetryFailures: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "retry_failures_total",
			Help:      "Total number of failed retries",
		}, []string{"policy", "reason"}),

		RetryBackoffDelay: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "retry_backoff_delay_seconds",
			Help:      "Retry backoff delay in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 10), // 1ms to ~1s
		}, []string{"policy"}),

		// Health Check
		HealthCheckStatus: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "health_check_status",
			Help:      "Health check status (1=healthy, 0=unhealthy)",
		}, []string{"checker", "check"}),

		HealthCheckDuration: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "health_check_duration_seconds",
			Help:      "Health check duration in seconds",
			Buckets:   prometheus.DefBuckets,
		}, []string{"checker", "check"}),

		HealthCheckTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "health_check_total",
			Help:      "Total number of health checks",
		}, []string{"checker", "check", "result"}),

		HealthCheckFailures: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "health_check_failures_total",
			Help:      "Total number of health check failures",
		}, []string{"checker", "check"}),

		// Rate Limiter
		RateLimiterAllowed: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "rate_limiter_allowed_total",
			Help:      "Total number of allowed requests",
		}, []string{"limiter"}),

		RateLimiterRejected: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "rate_limiter_rejected_total",
			Help:      "Total number of rejected requests",
		}, []string{"limiter"}),

		RateLimiterCurrentRate: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "rate_limiter_current_rate",
			Help:      "Current rate limit",
		}, []string{"limiter"}),

		// Bulkhead
		BulkheadConcurrent: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "bulkhead_concurrent",
			Help:      "Current number of concurrent executions",
		}, []string{"bulkhead"}),

		BulkheadQueued: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "bulkhead_queued",
			Help:      "Current number of queued requests",
		}, []string{"bulkhead"}),

		BulkheadRejected: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "bulkhead_rejected_total",
			Help:      "Total number of rejected requests",
		}, []string{"bulkhead"}),

		BulkheadLatency: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Name:      "bulkhead_wait_duration_seconds",
			Help:      "Bulkhead wait duration in seconds",
			Buckets:   prometheus.DefBuckets,
		}, []string{"bulkhead"}),

		// Error Budget
		ErrorBudgetRemaining: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "error_budget_remaining",
			Help:      "Remaining error budget (0-1)",
		}, []string{"budget"}),

		ErrorBudgetConsumed: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "error_budget_consumed",
			Help:      "Consumed error budget (0-1)",
		}, []string{"budget"}),

		ErrorBudgetExhausted: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "error_budget_exhausted_total",
			Help:      "Total number of times error budget was exhausted",
		}, []string{"budget"}),

		// Latency Budget
		LatencyBudgetP50: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "latency_budget_p50_seconds",
			Help:      "Latency budget 50th percentile in seconds",
		}, []string{"budget"}),

		LatencyBudgetP95: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "latency_budget_p95_seconds",
			Help:      "Latency budget 95th percentile in seconds",
		}, []string{"budget"}),

		LatencyBudgetP99: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Name:      "latency_budget_p99_seconds",
			Help:      "Latency budget 99th percentile in seconds",
		}, []string{"budget"}),

		LatencyBudgetViolations: promauto.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Name:      "latency_budget_violations_total",
			Help:      "Total number of latency budget violations",
		}, []string{"budget"}),
	}
}

// Global metrics instance
var globalMetrics *PrometheusMetrics

// InitMetrics initializes global Prometheus metrics
func InitMetrics(namespace string) {
	if globalMetrics == nil {
		globalMetrics = NewPrometheusMetrics(namespace)
	}
}

// GetMetrics returns the global metrics instance
func GetMetrics() *PrometheusMetrics {
	if globalMetrics == nil {
		InitMetrics("dwcp")
	}
	return globalMetrics
}
