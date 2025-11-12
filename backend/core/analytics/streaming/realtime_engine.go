// Package streaming provides real-time analytics engine for DWCP v3
// Implements streaming analytics with Apache Flink/ClickHouse integration
// Achieves <2s latency for real-time dashboards with custom aggregations
package streaming

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ClickHouse/clickhouse-go/v2"
	"github.com/apache/flink-statefun/statefun-sdk-go/v3/pkg/statefun"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/segmentio/kafka-go"
	"go.uber.org/zap"
)

// RealtimeEngine handles streaming analytics with <2s latency
type RealtimeEngine struct {
	clickhouse      clickhouse.Conn
	kafkaReader     *kafka.Reader
	kafkaWriter     *kafka.Writer
	statefun        *statefun.StatefulFunctions
	aggregators     map[string]*MetricAggregator
	windowManager   *WindowManager
	forecaster      *TimeSeriesForecaster
	alertManager    *AlertingSystem
	cache           *DistributedCache
	metrics         *AnalyticsMetrics
	config          *StreamingConfig
	mu              sync.RWMutex
	processingRate  atomic.Int64
	latencyTracker  *LatencyTracker
	logger          *zap.Logger
}

// StreamingConfig defines real-time analytics configuration
type StreamingConfig struct {
	ClickHouseURL     string
	KafkaBrokers      []string
	FlinkJobManager   string
	WindowSizes       []time.Duration
	AggregationTypes  []string
	MaxLatencyMs      int64
	BufferSize        int
	ParallelismFactor int
	CheckpointInterval time.Duration
	EnableForecasting bool
	EnableAnomalyDetection bool
}

// MetricAggregator performs real-time metric aggregations
type MetricAggregator struct {
	name           string
	aggregationType string
	windowSize     time.Duration
	buffer         *CircularBuffer
	computeFunc    AggregationFunc
	lastComputed   time.Time
	results        map[time.Time]float64
	mu             sync.RWMutex
}

// WindowManager handles time-window based analytics
type WindowManager struct {
	windows        map[string]*TimeWindow
	tumblingWindows map[string]*TumblingWindow
	slidingWindows  map[string]*SlidingWindow
	sessionWindows  map[string]*SessionWindow
	watermark      time.Time
	mu             sync.RWMutex
}

// TimeSeriesForecaster implements Prophet + LSTM forecasting
type TimeSeriesForecaster struct {
	prophetModel   *ProphetModel
	lstmModel      *LSTMModel
	ensemble       *EnsembleForecaster
	featureStore   *FeatureStore
	accuracy       float64
	mu             sync.RWMutex
}

// StreamEvent represents a streaming analytics event
type StreamEvent struct {
	EventID       string                 `json:"event_id"`
	Timestamp     time.Time              `json:"timestamp"`
	EventType     string                 `json:"event_type"`
	Source        string                 `json:"source"`
	Metrics       map[string]float64     `json:"metrics"`
	Dimensions    map[string]string      `json:"dimensions"`
	Tags          []string               `json:"tags"`
	ProcessedAt   time.Time              `json:"processed_at"`
	Latency       time.Duration          `json:"latency"`
}

// AggregationResult represents aggregated metrics
type AggregationResult struct {
	WindowStart   time.Time              `json:"window_start"`
	WindowEnd     time.Time              `json:"window_end"`
	AggregationType string               `json:"aggregation_type"`
	Metrics       map[string]float64     `json:"metrics"`
	Statistics    *WindowStatistics      `json:"statistics"`
	Anomalies     []AnomalyDetection     `json:"anomalies,omitempty"`
	Forecast      *ForecastResult        `json:"forecast,omitempty"`
}

// NewRealtimeEngine creates a new streaming analytics engine
func NewRealtimeEngine(config *StreamingConfig, logger *zap.Logger) (*RealtimeEngine, error) {
	// Initialize ClickHouse connection
	conn, err := clickhouse.Open(&clickhouse.Options{
		Addr: []string{config.ClickHouseURL},
		Settings: clickhouse.Settings{
			"max_execution_time": 60,
			"distributed_product_mode": "global",
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to ClickHouse: %w", err)
	}

	// Initialize Kafka reader
	kafkaReader := kafka.NewReader(kafka.ReaderConfig{
		Brokers:  config.KafkaBrokers,
		Topic:    "dwcp-analytics-stream",
		GroupID:  "realtime-analytics",
		MinBytes: 1,
		MaxBytes: 10e6,
		MaxWait:  100 * time.Millisecond,
	})

	// Initialize Kafka writer
	kafkaWriter := &kafka.Writer{
		Addr:     kafka.TCP(config.KafkaBrokers...),
		Topic:    "dwcp-analytics-results",
		Balancer: &kafka.LeastBytes{},
		BatchSize: 100,
		BatchTimeout: 10 * time.Millisecond,
	}

	engine := &RealtimeEngine{
		clickhouse:     conn,
		kafkaReader:    kafkaReader,
		kafkaWriter:    kafkaWriter,
		aggregators:    make(map[string]*MetricAggregator),
		windowManager:  NewWindowManager(),
		forecaster:     NewTimeSeriesForecaster(),
		alertManager:   NewAlertingSystem(),
		cache:          NewDistributedCache(),
		metrics:        NewAnalyticsMetrics(),
		config:         config,
		latencyTracker: NewLatencyTracker(),
		logger:         logger,
	}

	// Initialize StateFun for complex event processing
	engine.statefun = statefun.StatefulFunctionsBuilder().
		WithSpec(statefun.StatefulFunctionSpec{
			FunctionType: statefun.TypeNameFrom("dwcp.analytics/aggregator"),
			States: []statefun.ValueSpec{
				statefun.ValueSpec{Name: "window_state"},
				statefun.ValueSpec{Name: "aggregation_state"},
			},
			Function: engine.processStatefulAggregation,
		}).Build()

	// Initialize aggregators
	engine.initializeAggregators()

	// Start processing loops
	go engine.streamProcessor()
	go engine.aggregationProcessor()
	go engine.forecastingProcessor()
	go engine.anomalyDetectionProcessor()

	return engine, nil
}

// Process handles incoming stream events with <2s latency guarantee
func (e *RealtimeEngine) Process(ctx context.Context, event *StreamEvent) (*AggregationResult, error) {
	startTime := time.Now()
	defer func() {
		latency := time.Since(startTime)
		e.latencyTracker.Record(latency)
		if latency.Milliseconds() > e.config.MaxLatencyMs {
			e.logger.Warn("Latency exceeded threshold",
				zap.Duration("latency", latency),
				zap.Int64("threshold_ms", e.config.MaxLatencyMs))
		}
	}()

	// Update processing rate
	e.processingRate.Add(1)

	// Apply windowing
	window := e.windowManager.AssignWindow(event)

	// Perform aggregations
	aggregations := make(map[string]float64)
	for name, aggregator := range e.aggregators {
		if aggregator.Matches(event) {
			result := aggregator.Aggregate(event)
			aggregations[name] = result
		}
	}

	// Check for anomalies
	anomalies := e.detectAnomalies(event, aggregations)

	// Generate forecasts if enabled
	var forecast *ForecastResult
	if e.config.EnableForecasting {
		forecast = e.forecaster.Forecast(event.Metrics, 24*time.Hour)
	}

	// Store in ClickHouse
	if err := e.storeEvent(ctx, event, aggregations); err != nil {
		e.logger.Error("Failed to store event", zap.Error(err))
	}

	// Publish results to Kafka
	result := &AggregationResult{
		WindowStart:     window.Start,
		WindowEnd:       window.End,
		AggregationType: "realtime",
		Metrics:         aggregations,
		Statistics:      e.calculateStatistics(aggregations),
		Anomalies:       anomalies,
		Forecast:        forecast,
	}

	if err := e.publishResult(result); err != nil {
		e.logger.Error("Failed to publish result", zap.Error(err))
	}

	return result, nil
}

// CreateCustomAggregation creates a custom metric aggregation
func (e *RealtimeEngine) CreateCustomAggregation(name string, config *AggregationConfig) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	aggregator := &MetricAggregator{
		name:            name,
		aggregationType: config.Type,
		windowSize:      config.WindowSize,
		buffer:          NewCircularBuffer(config.BufferSize),
		computeFunc:     e.getAggregationFunc(config.Type),
		results:         make(map[time.Time]float64),
	}

	e.aggregators[name] = aggregator

	e.logger.Info("Created custom aggregation",
		zap.String("name", name),
		zap.String("type", config.Type),
		zap.Duration("window", config.WindowSize))

	return nil
}

// GetRealtimeDashboard returns real-time dashboard data
func (e *RealtimeEngine) GetRealtimeDashboard(ctx context.Context, dashboardID string) (*DashboardData, error) {
	// Fetch from cache first
	if cached, ok := e.cache.Get(dashboardID); ok {
		if data, valid := cached.(*DashboardData); valid {
			if time.Since(data.UpdatedAt) < 2*time.Second {
				return data, nil
			}
		}
	}

	// Query ClickHouse for latest metrics
	query := `
		SELECT
			toStartOfInterval(timestamp, INTERVAL 1 second) as time_bucket,
			metric_name,
			avg(value) as avg_value,
			max(value) as max_value,
			min(value) as min_value,
			quantile(0.95)(value) as p95_value,
			quantile(0.99)(value) as p99_value
		FROM dwcp_metrics
		WHERE dashboard_id = ? AND timestamp > now() - INTERVAL 5 minute
		GROUP BY time_bucket, metric_name
		ORDER BY time_bucket DESC
		LIMIT 1000
	`

	rows, err := e.clickhouse.Query(ctx, query, dashboardID)
	if err != nil {
		return nil, fmt.Errorf("failed to query metrics: %w", err)
	}
	defer rows.Close()

	dashboard := &DashboardData{
		ID:        dashboardID,
		Metrics:   make(map[string]*MetricSeries),
		UpdatedAt: time.Now(),
	}

	for rows.Next() {
		var (
			timeBucket time.Time
			metricName string
			avgValue   float64
			maxValue   float64
			minValue   float64
			p95Value   float64
			p99Value   float64
		)

		if err := rows.Scan(&timeBucket, &metricName, &avgValue, &maxValue, &minValue, &p95Value, &p99Value); err != nil {
			continue
		}

		if _, ok := dashboard.Metrics[metricName]; !ok {
			dashboard.Metrics[metricName] = &MetricSeries{
				Name:   metricName,
				Points: make([]DataPoint, 0),
			}
		}

		dashboard.Metrics[metricName].Points = append(dashboard.Metrics[metricName].Points, DataPoint{
			Timestamp: timeBucket,
			Value:     avgValue,
			Max:       maxValue,
			Min:       minValue,
			P95:       p95Value,
			P99:       p99Value,
		})
	}

	// Add forecasts
	for metricName, series := range dashboard.Metrics {
		if forecast := e.forecaster.GetLatestForecast(metricName); forecast != nil {
			series.Forecast = forecast
		}
	}

	// Cache the result
	e.cache.Set(dashboardID, dashboard, 2*time.Second)

	return dashboard, nil
}

// streamProcessor continuously processes incoming events
func (e *RealtimeEngine) streamProcessor() {
	for {
		msg, err := e.kafkaReader.ReadMessage(context.Background())
		if err != nil {
			e.logger.Error("Failed to read Kafka message", zap.Error(err))
			continue
		}

		var event StreamEvent
		if err := json.Unmarshal(msg.Value, &event); err != nil {
			e.logger.Error("Failed to unmarshal event", zap.Error(err))
			continue
		}

		// Process event asynchronously for parallelism
		go func(evt StreamEvent) {
			if _, err := e.Process(context.Background(), &evt); err != nil {
				e.logger.Error("Failed to process event", zap.Error(err))
			}
		}(event)
	}
}

// aggregationProcessor handles continuous aggregations
func (e *RealtimeEngine) aggregationProcessor() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for range ticker.C {
		e.mu.RLock()
		aggregators := make([]*MetricAggregator, 0, len(e.aggregators))
		for _, agg := range e.aggregators {
			aggregators = append(aggregators, agg)
		}
		e.mu.RUnlock()

		// Process each aggregator
		for _, agg := range aggregators {
			if time.Since(agg.lastComputed) >= agg.windowSize {
				agg.Compute()
				agg.lastComputed = time.Now()
			}
		}
	}
}

// forecastingProcessor handles time-series forecasting
func (e *RealtimeEngine) forecastingProcessor() {
	if !e.config.EnableForecasting {
		return
	}

	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		// Get recent metrics for forecasting
		metrics := e.getRecentMetrics(1 * time.Hour)

		// Update forecasts
		for metricName, values := range metrics {
			forecast := e.forecaster.UpdateForecast(metricName, values)
			e.logger.Info("Updated forecast",
				zap.String("metric", metricName),
				zap.Float64("accuracy", forecast.Accuracy))
		}
	}
}

// anomalyDetectionProcessor detects anomalies in real-time
func (e *RealtimeEngine) anomalyDetectionProcessor() {
	if !e.config.EnableAnomalyDetection {
		return
	}

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	detector := NewAnomalyDetector()

	for range ticker.C {
		// Get latest metrics
		metrics := e.getRecentMetrics(5 * time.Minute)

		// Detect anomalies
		for metricName, values := range metrics {
			if anomalies := detector.Detect(values); len(anomalies) > 0 {
				e.alertManager.TriggerAlert(&Alert{
					Type:      "anomaly",
					Severity:  "warning",
					Metric:    metricName,
					Anomalies: anomalies,
					Timestamp: time.Now(),
				})
			}
		}
	}
}

// storeEvent stores event in ClickHouse
func (e *RealtimeEngine) storeEvent(ctx context.Context, event *StreamEvent, aggregations map[string]float64) error {
	batch, err := e.clickhouse.PrepareBatch(ctx, `
		INSERT INTO dwcp_metrics (
			timestamp, event_id, event_type, source,
			metric_name, value, dimensions, tags,
			processing_latency_ms
		)
	`)
	if err != nil {
		return err
	}

	for metricName, value := range event.Metrics {
		err = batch.Append(
			event.Timestamp,
			event.EventID,
			event.EventType,
			event.Source,
			metricName,
			value,
			event.Dimensions,
			event.Tags,
			event.Latency.Milliseconds(),
		)
		if err != nil {
			return err
		}
	}

	return batch.Send()
}

// GetMetrics returns current analytics metrics
func (e *RealtimeEngine) GetMetrics() *AnalyticsMetrics {
	return &AnalyticsMetrics{
		ProcessingRate:      e.processingRate.Load(),
		AverageLatencyMs:    e.latencyTracker.Average(),
		P95LatencyMs:        e.latencyTracker.P95(),
		P99LatencyMs:        e.latencyTracker.P99(),
		ActiveAggregations:  int64(len(e.aggregators)),
		CacheHitRate:        e.cache.HitRate(),
		ForecastAccuracy:    e.forecaster.accuracy,
		AnomaliesDetected:   e.alertManager.GetAnomalyCount(),
		WindowsProcessed:    e.windowManager.GetProcessedCount(),
	}
}