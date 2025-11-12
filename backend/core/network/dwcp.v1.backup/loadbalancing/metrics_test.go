package loadbalancing

import (
	"context"
	"testing"
	"time"
)

func TestMetricsCollectorCreation(t *testing.T) {
	mc := NewMetricsCollector(DefaultConfig())
	if mc == nil {
		t.Fatal("Expected non-nil metrics collector")
	}
}

func TestRecordRoutingDecision(t *testing.T) {
	mc := NewMetricsCollector(DefaultConfig())

	server := createTestServer("1", "us-east-1")
	decision := &RoutingDecision{
		Server:     server,
		Algorithm:  AlgorithmRoundRobin,
		Latency:    500 * time.Microsecond,
		Timestamp:  time.Now(),
		ReasonCode: "test",
		IsFailover: false,
	}

	mc.RecordRoutingDecision(decision)

	metrics := mc.GetMetrics()
	if metrics.TotalRequests != 1 {
		t.Errorf("Expected 1 request, got %d", metrics.TotalRequests)
	}
}

func TestRecordResponseMetrics(t *testing.T) {
	mc := NewMetricsCollector(DefaultConfig())

	// Record successful responses
	for i := 0; i < 10; i++ {
		mc.RecordResponse(10*time.Millisecond, true)
	}

	// Record failures
	for i := 0; i < 3; i++ {
		mc.RecordResponse(50*time.Millisecond, false)
	}

	metrics := mc.GetMetrics()
	if metrics.TotalFailures != 3 {
		t.Errorf("Expected 3 failures, got %d", metrics.TotalFailures)
	}
}

func TestRecordFailover(t *testing.T) {
	mc := NewMetricsCollector(DefaultConfig())

	failoverTimes := []time.Duration{
		50 * time.Millisecond,
		75 * time.Millisecond,
		60 * time.Millisecond,
	}

	for _, ft := range failoverTimes {
		mc.RecordFailover(ft)
	}

	metrics := mc.GetMetrics()
	if metrics.TotalFailovers != 3 {
		t.Errorf("Expected 3 failovers, got %d", metrics.TotalFailovers)
	}

	// Average should be around 61-62ms
	if metrics.AvgFailoverTime < 55*time.Millisecond ||
		metrics.AvgFailoverTime > 70*time.Millisecond {
		t.Errorf("Average failover time out of expected range: %v", metrics.AvgFailoverTime)
	}
}

func TestPercentileCalculation(t *testing.T) {
	mc := NewMetricsCollector(DefaultConfig())

	// Record response times: 10ms, 20ms, 30ms, ..., 100ms
	for i := 1; i <= 10; i++ {
		mc.RecordResponse(time.Duration(i*10)*time.Millisecond, true)
	}

	metrics := mc.GetMetrics()

	// P50 should be around 50ms
	if metrics.P50ResponseTime < 40*time.Millisecond ||
		metrics.P50ResponseTime > 60*time.Millisecond {
		t.Errorf("P50 out of range: %v", metrics.P50ResponseTime)
	}

	// P95 should be around 95ms
	if metrics.P95ResponseTime < 85*time.Millisecond ||
		metrics.P95ResponseTime > 105*time.Millisecond {
		t.Errorf("P95 out of range: %v", metrics.P95ResponseTime)
	}

	// P99 should be around 99ms
	if metrics.P99ResponseTime < 90*time.Millisecond ||
		metrics.P99ResponseTime > 110*time.Millisecond {
		t.Errorf("P99 out of range: %v", metrics.P99ResponseTime)
	}
}

func TestRegionDistribution(t *testing.T) {
	mc := NewMetricsCollector(DefaultConfig())

	// Create decisions for different regions
	regions := []string{"us-east-1", "us-west-1", "eu-west-1"}
	for _, region := range regions {
		for i := 0; i < 5; i++ {
			server := createTestServer("1", region)
			decision := &RoutingDecision{
				Server:    server,
				Algorithm: AlgorithmGeoProximity,
			}
			mc.RecordRoutingDecision(decision)
		}
	}

	dist := mc.GetRegionDistribution()
	if len(dist) != 3 {
		t.Errorf("Expected 3 regions, got %d", len(dist))
	}

	for region, count := range dist {
		if count != 5 {
			t.Errorf("Region %s: expected 5 requests, got %d", region, count)
		}
	}
}

func TestMetricsReset(t *testing.T) {
	mc := NewMetricsCollector(DefaultConfig())

	// Record some metrics
	mc.RecordFailure()
	mc.RecordResponse(10*time.Millisecond, true)
	mc.IncrementConnections()

	// Reset
	mc.Reset()

	metrics := mc.GetMetrics()
	if metrics.TotalRequests != 0 {
		t.Errorf("Expected 0 requests after reset, got %d", metrics.TotalRequests)
	}
	if metrics.TotalFailures != 0 {
		t.Errorf("Expected 0 failures after reset, got %d", metrics.TotalFailures)
	}
}

func TestConnectionTracking(t *testing.T) {
	mc := NewMetricsCollector(DefaultConfig())

	// Increment connections
	for i := 0; i < 10; i++ {
		mc.IncrementConnections()
	}

	metrics := mc.GetMetrics()
	if metrics.TotalConnections != 10 {
		t.Errorf("Expected 10 connections, got %d", metrics.TotalConnections)
	}

	// Decrement connections
	for i := 0; i < 3; i++ {
		mc.DecrementConnections()
	}

	metrics = mc.GetMetrics()
	if metrics.TotalConnections != 7 {
		t.Errorf("Expected 7 connections, got %d", metrics.TotalConnections)
	}
}

func TestMetricsAggregation(t *testing.T) {
	config := DefaultConfig()
	config.MetricsInterval = 100 * time.Millisecond

	mc := NewMetricsCollector(config)
	mc.ctx, mc.cancel = context.WithCancel(context.Background())

	mc.Start()
	defer mc.Stop()

	// Record requests over time
	for i := 0; i < 50; i++ {
		decision := &RoutingDecision{
			Server:    createTestServer("1", "us-east-1"),
			Algorithm: AlgorithmRoundRobin,
		}
		mc.RecordRoutingDecision(decision)
		time.Sleep(10 * time.Millisecond)
	}

	// Wait for aggregation
	time.Sleep(150 * time.Millisecond)

	metrics := mc.GetMetrics()

	// RPS should be calculated
	if metrics.RequestsPerSecond == 0 {
		t.Error("Expected non-zero requests per second")
	}
}
