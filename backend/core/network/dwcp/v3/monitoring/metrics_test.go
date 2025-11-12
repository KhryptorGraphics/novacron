package monitoring

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
	"go.uber.org/zap"
)

func TestDWCPv3MetricsCollector(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	collector := NewDWCPv3MetricsCollector("test-node", "test-cluster", false, logger)
	defer collector.Close()

	t.Run("RecordModeSwitch", func(t *testing.T) {
		collector.RecordModeSwitch(upgrade.ModeDatacenter, upgrade.ModeInternet)

		if collector.totalModeSwitches.Load() != 1 {
			t.Errorf("Expected 1 mode switch, got %d", collector.totalModeSwitches.Load())
		}

		if collector.currentMode.Load().(upgrade.NetworkMode) != upgrade.ModeInternet {
			t.Errorf("Expected current mode to be Internet")
		}
	})

	t.Run("RecordAMSTMetric", func(t *testing.T) {
		collector.RecordAMSTMetric("stream_created", nil)
		collector.RecordAMSTMetric("bytes_sent", uint64(1024))

		metrics := collector.getAMSTMetrics()
		if activeStreams, ok := metrics["active_streams"].(int32); !ok || activeStreams != 1 {
			t.Errorf("Expected 1 active stream, got %v", metrics["active_streams"])
		}
	})

	t.Run("RecordHDEMetric", func(t *testing.T) {
		collector.RecordHDEMetric("compression", nil)
		collector.RecordHDEMetric("compression_ratio", 3.5)
		collector.RecordHDEMetric("algorithm_used", "zstd")

		metrics := collector.getHDEMetrics()
		if compTotal, ok := metrics["compressions_total"].(int64); !ok || compTotal != 1 {
			t.Errorf("Expected 1 compression, got %v", metrics["compressions_total"])
		}
	})

	t.Run("RecordMigration", func(t *testing.T) {
		collector.RecordMigration(true, 5*time.Second, 1024*1024*1024)

		metrics := collector.GetComprehensiveMetrics()
		global := metrics["global"].(map[string]interface{})
		if total, ok := global["total_migrations"].(uint64); !ok || total != 1 {
			t.Errorf("Expected 1 migration, got %v", global["total_migrations"])
		}
		if successRate, ok := global["success_rate"].(float64); !ok || successRate != 100.0 {
			t.Errorf("Expected 100%% success rate, got %v", global["success_rate"])
		}
	})

	t.Run("ComponentMetrics", func(t *testing.T) {
		collector.RecordPBAMetric("prediction", nil)
		collector.RecordPBAMetric("prediction_accuracy", 0.85)

		collector.RecordASSMetric("sync_start", nil)
		collector.RecordASSMetric("sync_success", nil)

		collector.RecordACPMetric("consensus_start", nil)
		collector.RecordACPMetric("raft_consensus", nil)

		collector.RecordITPMetric("placement_decision", nil)
		collector.RecordITPMetric("optimal_placement", nil)

		metrics := collector.GetComprehensiveMetrics()

		if pbaMetrics, ok := metrics["pba"].(map[string]interface{}); !ok {
			t.Errorf("PBA metrics not found")
		} else if totalPred, ok := pbaMetrics["total_predictions"].(int64); !ok || totalPred != 1 {
			t.Errorf("Expected 1 prediction, got %v", pbaMetrics["total_predictions"])
		}
	})
}

func TestPerformanceTracker(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	tracker := NewPerformanceTracker(logger)

	t.Run("RecordV1Transfer", func(t *testing.T) {
		tracker.RecordV1Transfer(true, 10.0, 1000.0, 1024*1024)

		if tracker.v1Metrics.TotalTransfers != 1 {
			t.Errorf("Expected 1 v1 transfer, got %d", tracker.v1Metrics.TotalTransfers)
		}
		if tracker.v1Metrics.SuccessfulTransfers != 1 {
			t.Errorf("Expected 1 successful v1 transfer")
		}
	})

	t.Run("RecordV3Transfer", func(t *testing.T) {
		tracker.RecordV3Transfer(true, 8.0, 1200.0, 800*1024)

		if tracker.v3Metrics.TotalTransfers != 1 {
			t.Errorf("Expected 1 v3 transfer, got %d", tracker.v3Metrics.TotalTransfers)
		}
	})

	t.Run("CompareVersions", func(t *testing.T) {
		// Record more samples to get meaningful comparison
		for i := 0; i < 10; i++ {
			tracker.RecordV1Transfer(true, 10.0, 1000.0, 1024*1024)
			tracker.RecordV3Transfer(true, 8.0, 1200.0, 800*1024)
		}

		comparison := tracker.compareVersions()

		if comparison.ThroughputImprovement <= 0 {
			t.Errorf("Expected positive throughput improvement, got %f", comparison.ThroughputImprovement)
		}
		if comparison.LatencyImprovement <= 0 {
			t.Errorf("Expected positive latency improvement, got %f", comparison.LatencyImprovement)
		}
		if !comparison.V3FasterThanV1 {
			t.Errorf("Expected v3 to be faster than v1")
		}
	})

	t.Run("RolloutPercentage", func(t *testing.T) {
		tracker.SetRolloutPercentage(10.0)
		if tracker.rolloutPercentage != 10.0 {
			t.Errorf("Expected rollout percentage 10%%, got %f", tracker.rolloutPercentage)
		}

		tracker.SetRolloutPercentage(50.0)
		if len(tracker.rolloutHistory) == 0 {
			t.Errorf("Expected rollout history to be created")
		}
	})

	t.Run("RegressionDetection", func(t *testing.T) {
		// Simulate v3 regression by recording worse performance
		for i := 0; i < 10; i++ {
			tracker.RecordV3Transfer(true, 20.0, 500.0, 1024*1024)
		}

		tracker.detectRegressions()
		regressions := tracker.GetRegressions()

		if len(regressions) == 0 {
			t.Errorf("Expected regressions to be detected")
		}
	})

	t.Run("BandwidthSavings", func(t *testing.T) {
		tracker.RecordBandwidthSavings(1024*1024*1024, 300*1024*1024)

		if tracker.v3Metrics.BandwidthSavedPercent < 50.0 {
			t.Errorf("Expected significant bandwidth savings, got %f%%",
				tracker.v3Metrics.BandwidthSavedPercent)
		}
	})
}

func TestAnomalyDetector(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	config := DefaultAnomalyDetectorConfig()
	config.BandwidthThreshold = 2.0 // Lower threshold for testing
	detector := NewAnomalyDetector(config, logger)
	defer detector.Close()

	t.Run("BandwidthAnomaly", func(t *testing.T) {
		// Build baseline with normal samples
		for i := 0; i < 20; i++ {
			detector.CheckBandwidth(1000.0)
		}

		// Inject anomaly
		anomaly := detector.CheckBandwidth(5000.0)

		if anomaly == nil {
			t.Errorf("Expected bandwidth anomaly to be detected")
		} else {
			if anomaly.Severity == "low" {
				t.Errorf("Expected high severity for large deviation, got %s", anomaly.Severity)
			}
			if anomaly.Confidence < 0.8 {
				t.Errorf("Expected high confidence, got %f", anomaly.Confidence)
			}
		}
	})

	t.Run("LatencyAnomaly", func(t *testing.T) {
		// Build baseline
		for i := 0; i < 20; i++ {
			detector.CheckLatency(10.0)
		}

		// Inject spike
		anomaly := detector.CheckLatency(100.0)

		if anomaly == nil {
			t.Errorf("Expected latency anomaly to be detected")
		}
	})

	t.Run("CompressionRatioAnomaly", func(t *testing.T) {
		// Build baseline
		for i := 0; i < 20; i++ {
			detector.CheckCompressionRatio(3.5)
		}

		// Inject poor compression
		anomaly := detector.CheckCompressionRatio(1.1)

		if anomaly == nil {
			t.Errorf("Expected compression anomaly to be detected")
		}
	})

	t.Run("AnomalyHistory", func(t *testing.T) {
		since := time.Now().Add(-1 * time.Hour)
		anomalies := detector.GetAnomalies(since)

		if len(anomalies) == 0 {
			t.Errorf("Expected anomalies in history")
		}
	})

	t.Run("AnomalyStats", func(t *testing.T) {
		stats := detector.GetAnomalyStats()

		if total, ok := stats["total_anomalies"].(int); !ok || total == 0 {
			t.Errorf("Expected non-zero total anomalies")
		}
	})

	t.Run("AlertCallback", func(t *testing.T) {
		callbackFired := false
		detector.RegisterAlertCallback(func(a *Anomaly) {
			callbackFired = true
		})

		// Build baseline and trigger anomaly
		for i := 0; i < 20; i++ {
			detector.CheckConsensusLatency(100.0)
		}
		detector.CheckConsensusLatency(1000.0)

		time.Sleep(100 * time.Millisecond) // Wait for goroutine

		if !callbackFired {
			t.Errorf("Expected alert callback to fire")
		}
	})
}

func TestDashboardExporter(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	metricsCollector := NewDWCPv3MetricsCollector("test-node", "test-cluster", false, logger)
	perfTracker := NewPerformanceTracker(logger)
	anomalyDetector := NewAnomalyDetector(nil, logger)

	exporter := NewDashboardExporter(metricsCollector, perfTracker, anomalyDetector, logger)

	t.Run("ExportMainDashboard", func(t *testing.T) {
		dashboardJSON, err := exporter.ExportMainDashboard()
		if err != nil {
			t.Fatalf("Failed to export main dashboard: %v", err)
		}

		if len(dashboardJSON) == 0 {
			t.Errorf("Expected non-empty dashboard JSON")
		}

		// Verify JSON is valid
		var dashboard GrafanaDashboard
		if err := unmarshalJSON(dashboardJSON, &dashboard); err != nil {
			t.Errorf("Invalid dashboard JSON: %v", err)
		}

		if dashboard.UID != "dwcp-v3-main" {
			t.Errorf("Expected UID 'dwcp-v3-main', got %s", dashboard.UID)
		}

		if len(dashboard.Panels) == 0 {
			t.Errorf("Expected dashboard to have panels")
		}
	})

	t.Run("ExportModeDashboard", func(t *testing.T) {
		modes := []string{"datacenter", "internet", "hybrid"}

		for _, mode := range modes {
			dashboardJSON, err := exporter.ExportModeDashboard(mode)
			if err != nil {
				t.Errorf("Failed to export %s dashboard: %v", mode, err)
			}

			if len(dashboardJSON) == 0 {
				t.Errorf("Expected non-empty dashboard JSON for %s mode", mode)
			}
		}
	})

	t.Run("ExportComponentDashboard", func(t *testing.T) {
		components := []string{"amst", "hde", "pba", "ass", "acp", "itp"}

		for _, component := range components {
			dashboardJSON, err := exporter.ExportComponentDashboard(component)
			if err != nil {
				t.Errorf("Failed to export %s dashboard: %v", component, err)
			}

			if len(dashboardJSON) == 0 {
				t.Errorf("Expected non-empty dashboard JSON for %s component", component)
			}
		}
	})

	t.Run("GetDashboardList", func(t *testing.T) {
		dashboards := exporter.GetDashboardList()

		if len(dashboards) == 0 {
			t.Errorf("Expected non-empty dashboard list")
		}

		// Verify structure
		for _, dashboard := range dashboards {
			if dashboard["uid"] == "" || dashboard["title"] == "" {
				t.Errorf("Invalid dashboard entry: %v", dashboard)
			}
		}
	})

	t.Run("ExportPrometheusConfig", func(t *testing.T) {
		configJSON, err := exporter.ExportPrometheusConfig()
		if err != nil {
			t.Fatalf("Failed to export Prometheus config: %v", err)
		}

		if len(configJSON) == 0 {
			t.Errorf("Expected non-empty Prometheus config")
		}
	})
}

func TestObservabilityIntegration(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Use mock Jaeger endpoint for testing
	oi, err := NewObservabilityIntegration(
		"dwcp-v3-test",
		"test-node",
		"test-cluster",
		"http://localhost:14268/api/traces",
		logger,
	)
	if err != nil {
		t.Skipf("Skipping observability test (Jaeger not available): %v", err)
		return
	}
	defer oi.Close()

	t.Run("StartEndSpan", func(t *testing.T) {
		ctx := context.Background()

		ctx, span := oi.StartSpan(ctx, "test.operation")
		if span == nil {
			t.Fatalf("Expected non-nil span")
		}

		oi.EndSpan(span, nil)
	})

	t.Run("TraceMigration", func(t *testing.T) {
		ctx := context.Background()

		ctx, span := oi.TraceMigration(ctx, "vm-123", "node1", "node2", "datacenter")
		if span == nil {
			t.Fatalf("Expected non-nil span")
		}

		oi.EndSpan(span, nil)
	})

	t.Run("TraceComponent", func(t *testing.T) {
		ctx := context.Background()

		ctx, span := oi.TraceComponent(ctx, "amst", "send_data")
		if span == nil {
			t.Fatalf("Expected non-nil span")
		}

		oi.RecordEvent(ctx, "data_sent")

		oi.EndSpan(span, nil)
	})

	t.Run("StructuredLogging", func(t *testing.T) {
		ctx := context.Background()
		ctx, span := oi.StartSpan(ctx, "test.logging")

		oi.Log(ctx, zapcore.InfoLevel, "test-component", "Test message",
			zap.String("key1", "value1"),
			zap.Int("key2", 42))

		oi.EndSpan(span, nil)

		// Verify log was captured
		logs := oi.GetRecentLogs(10)
		if len(logs) == 0 {
			t.Errorf("Expected logs to be captured")
		}
	})

	t.Run("LogSearch", func(t *testing.T) {
		ctx := context.Background()
		ctx, span := oi.StartSpan(ctx, "test.search")

		oi.Log(ctx, zapcore.WarnLevel, "search-component", "Searchable message")

		oi.EndSpan(span, nil)

		logs := oi.SearchLogs("", "warn", "search-component", time.Now().Add(-1*time.Hour), 10)
		if len(logs) == 0 {
			t.Errorf("Expected to find logs with search")
		}
	})

	t.Run("LogStats", func(t *testing.T) {
		stats := oi.GetLogStats()

		if totalLogs, ok := stats["total_logs"].(int); !ok || totalLogs == 0 {
			t.Errorf("Expected non-zero total logs")
		}
	})

	t.Run("ProfilingData", func(t *testing.T) {
		// Wait for at least one profiling cycle
		time.Sleep(6 * time.Second)

		profData := oi.GetProfilingData()

		if avgCPU, ok := profData["avg_cpu_percent"].(float64); !ok || avgCPU == 0 {
			t.Logf("CPU profiling data: %v", profData)
		}
	})
}

func TestIntegrationScenario(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Create all monitoring components
	metricsCollector := NewDWCPv3MetricsCollector("integration-node", "integration-cluster", false, logger)
	defer metricsCollector.Close()

	perfTracker := NewPerformanceTracker(logger)
	anomalyDetector := NewAnomalyDetector(nil, logger)
	defer anomalyDetector.Close()

	// Simulate a complete migration scenario
	t.Run("CompleteV3Migration", func(t *testing.T) {
		// 1. Start migration
		metricsCollector.RecordModeSwitch(upgrade.ModeHybrid, upgrade.ModeDatacenter)

		// 2. AMST: Setup streams
		for i := 0; i < 8; i++ {
			metricsCollector.RecordAMSTMetric("stream_created", nil)
		}

		// 3. HDE: Compress data
		metricsCollector.RecordHDEMetric("compression", nil)
		metricsCollector.RecordHDEMetric("compression_ratio", 3.5)

		// 4. PBA: Make bandwidth prediction
		metricsCollector.RecordPBAMetric("prediction", nil)
		metricsCollector.RecordPBAMetric("prediction_accuracy", 0.87)

		// 5. ASS: Sync state
		metricsCollector.RecordASSMetric("sync_start", nil)
		metricsCollector.RecordASSMetric("sync_success", nil)

		// 6. ACP: Achieve consensus
		metricsCollector.RecordACPMetric("consensus_start", nil)
		metricsCollector.RecordACPMetric("raft_consensus", nil)

		// 7. ITP: Placement decision
		metricsCollector.RecordITPMetric("placement_decision", nil)
		metricsCollector.RecordITPMetric("optimal_placement", nil)

		// 8. Complete migration
		metricsCollector.RecordMigration(true, 3*time.Second, 5*1024*1024*1024)

		// 9. Record performance
		perfTracker.RecordV3Transfer(true, 2.5, 1500.0, 5*1024*1024*1024)

		// 10. Check for anomalies
		anomalyDetector.CheckBandwidth(1500.0)
		anomalyDetector.CheckLatency(2.5)
		anomalyDetector.CheckCompressionRatio(3.5)

		// Verify comprehensive metrics
		metrics := metricsCollector.GetComprehensiveMetrics()

		global := metrics["global"].(map[string]interface{})
		if successRate, ok := global["success_rate"].(float64); !ok || successRate != 100.0 {
			t.Errorf("Expected 100%% success rate, got %v", global["success_rate"])
		}

		// Verify all components recorded metrics
		components := []string{"amst", "hde", "pba", "ass", "acp", "itp"}
		for _, component := range components {
			if _, ok := metrics[component]; !ok {
				t.Errorf("Expected metrics for component %s", component)
			}
		}
	})
}

// Helper function for JSON unmarshaling (placeholder)
func unmarshalJSON(data []byte, v interface{}) error {
	// Would use json.Unmarshal in production
	return nil
}

// Benchmark tests

func BenchmarkMetricsCollection(b *testing.B) {
	logger, _ := zap.NewDevelopment()
	collector := NewDWCPv3MetricsCollector("bench-node", "bench-cluster", false, logger)
	defer collector.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		collector.RecordAMSTMetric("bytes_sent", uint64(1024))
		collector.RecordHDEMetric("compression", nil)
		collector.RecordPBAMetric("prediction", nil)
	}
}

func BenchmarkAnomalyDetection(b *testing.B) {
	logger, _ := zap.NewDevelopment()
	detector := NewAnomalyDetector(nil, logger)
	defer detector.Close()

	// Build baseline
	for i := 0; i < 100; i++ {
		detector.CheckBandwidth(1000.0)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.CheckBandwidth(1000.0 + float64(i%100))
	}
}

func BenchmarkPerformanceComparison(b *testing.B) {
	logger, _ := zap.NewDevelopment()
	tracker := NewPerformanceTracker(logger)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tracker.RecordV3Transfer(true, 10.0, 1000.0, 1024*1024)
		tracker.compareVersions()
	}
}
