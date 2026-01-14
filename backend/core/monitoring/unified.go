package monitoring

import (
	"context"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	promexporter "go.opentelemetry.io/otel/exporters/prometheus"
	"go.opentelemetry.io/otel/metric"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"net/http"
)

// UnifiedMonitoringSystem provides comprehensive observability
type UnifiedMonitoringSystem struct {
	meterProvider metric.MeterProvider
	registry      *prometheus.Registry
	server        *http.Server
}

// NewUnifiedMonitoringSystem creates a new monitoring system
func NewUnifiedMonitoringSystem() (*UnifiedMonitoringSystem, error) {
	registry := prometheus.NewRegistry()
	
	exporter, err := promexporter.New(
		promexporter.WithRegisterer(registry),
		promexporter.WithoutCounterSuffixes(),
		promexporter.WithoutScopeInfo(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create prometheus exporter: %w", err)
	}

	provider := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(exporter),
	)
	otel.SetMeterProvider(provider)

	return &UnifiedMonitoringSystem{
		meterProvider: provider,
		registry:      registry,
	}, nil
}

// Start begins the monitoring system
func (u *UnifiedMonitoringSystem) Start(addr string) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(u.registry, promhttp.HandlerOpts{}))
	
	u.server = &http.Server{
		Addr:    addr,
		Handler: mux,
	}
	
	go func() {
		if err := u.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			// Log error but don't crash
			fmt.Printf("Metrics server error: %v\n", err)
		}
	}()
	
	return nil
}

// Stop shuts down the monitoring system
func (u *UnifiedMonitoringSystem) Stop() error {
	if u.server != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return u.server.Shutdown(ctx)
	}
	return nil
}

// GetMeter returns a meter for creating instruments
func (u *UnifiedMonitoringSystem) GetMeter(name string) metric.Meter {
	return u.meterProvider.Meter(name)
}

// MetricsCollector provides application-specific metrics collection
type MetricsCollector struct {
	meter         metric.Meter
	requestCount  metric.Int64Counter
	responseTime  metric.Float64Histogram
	activeConns   metric.Int64UpDownCounter
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(monitoring *UnifiedMonitoringSystem) (*MetricsCollector, error) {
	meter := monitoring.GetMeter("novacron")
	
	requestCount, err := meter.Int64Counter(
		"http_requests_total",
		metric.WithDescription("Total number of HTTP requests"),
	)
	if err != nil {
		return nil, err
	}
	
	responseTime, err := meter.Float64Histogram(
		"http_request_duration_seconds",
		metric.WithDescription("HTTP request duration in seconds"),
	)
	if err != nil {
		return nil, err
	}
	
	activeConns, err := meter.Int64UpDownCounter(
		"active_connections",
		metric.WithDescription("Number of active connections"),
	)
	if err != nil {
		return nil, err
	}
	
	return &MetricsCollector{
		meter:         meter,
		requestCount:  requestCount,
		responseTime:  responseTime,
		activeConns:   activeConns,
	}, nil
}

// RecordRequest records an HTTP request
func (m *MetricsCollector) RecordRequest(ctx context.Context, method, endpoint string, status int, duration time.Duration) {
	m.requestCount.Add(ctx, 1,
		metric.WithAttributes(
			attribute.String("method", method),
			attribute.String("endpoint", endpoint),
			attribute.Int("status", status),
		),
	)
	
	m.responseTime.Record(ctx, duration.Seconds(),
		metric.WithAttributes(
			attribute.String("method", method),
			attribute.String("endpoint", endpoint),
		),
	)
}

// SetActiveConnections updates the active connections gauge
func (m *MetricsCollector) SetActiveConnections(ctx context.Context, count int64) {
	m.activeConns.Add(ctx, count)
}