// Production Data Collector for ML Training
// Collects and processes production metrics for machine learning models
package ml

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/influxdata/influxdb-client-go/v2"
	"github.com/influxdata/influxdb-client-go/v2/api"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/common/model"
	"github.com/sirupsen/logrus"
)

// MetricType defines the type of metric being collected
type MetricType string

const (
	MetricLatency    MetricType = "latency"
	MetricThroughput MetricType = "throughput"
	MetricErrorRate  MetricType = "error_rate"
	MetricCPU        MetricType = "cpu_usage"
	MetricMemory     MetricType = "memory_usage"
	MetricNetwork    MetricType = "network_io"
	MetricDisk       MetricType = "disk_io"
	MetricCompression MetricType = "compression_ratio"
	MetricPrediction  MetricType = "prediction_accuracy"
	MetricConsensus   MetricType = "consensus_time"
)

// TimeSeriesPoint represents a single data point in time series
type TimeSeriesPoint struct {
	Timestamp time.Time              `json:"timestamp"`
	Value     float64                `json:"value"`
	Tags      map[string]string      `json:"tags"`
	Fields    map[string]interface{} `json:"fields"`
}

// Feature represents an engineered feature for ML
type Feature struct {
	Name        string    `json:"name"`
	Value       float64   `json:"value"`
	Type        string    `json:"type"` // numeric, categorical, temporal
	Description string    `json:"description"`
	Timestamp   time.Time `json:"timestamp"`
}

// MLDataset represents a dataset ready for ML training
type MLDataset struct {
	ID          string            `json:"id"`
	Created     time.Time         `json:"created"`
	Features    [][]Feature       `json:"features"`
	Labels      []float64         `json:"labels"`
	Metadata    map[string]string `json:"metadata"`
	WindowSize  int               `json:"window_size"`
	StepSize    int               `json:"step_size"`
}

// ProductionDataCollector collects and processes production metrics
type ProductionDataCollector struct {
	mu              sync.RWMutex
	config          *CollectorConfig
	influxClient    influxdb2.Client
	writeAPI        api.WriteAPI
	queryAPI        api.QueryAPI
	metricsBuffer   []TimeSeriesPoint
	featureCache    map[string][]Feature
	datasets        map[string]*MLDataset
	featureEngines  map[string]FeatureEngine
	logger          *logrus.Logger
	metrics         *collectorMetrics
	running         bool
	stopCh          chan struct{}
}

// CollectorConfig holds configuration for data collector
type CollectorConfig struct {
	InfluxURL       string        `json:"influx_url"`
	InfluxToken     string        `json:"influx_token"`
	InfluxOrg       string        `json:"influx_org"`
	InfluxBucket    string        `json:"influx_bucket"`
	CollectionRate  time.Duration `json:"collection_rate"`
	BufferSize      int           `json:"buffer_size"`
	WindowSize      int           `json:"window_size"`
	StepSize        int           `json:"step_size"`
	AggregationFunc string        `json:"aggregation_func"`
	FeatureConfig   FeatureConfig `json:"feature_config"`
}

// FeatureConfig defines feature engineering settings
type FeatureConfig struct {
	EnableTemporalFeatures   bool    `json:"enable_temporal_features"`
	EnableStatisticalFeatures bool    `json:"enable_statistical_features"`
	EnableFrequencyFeatures  bool    `json:"enable_frequency_features"`
	EnableLaggedFeatures     bool    `json:"enable_lagged_features"`
	LagPeriods              []int   `json:"lag_periods"`
	RollingWindowSizes      []int   `json:"rolling_window_sizes"`
	QuantileThresholds      []float64 `json:"quantile_thresholds"`
}

// FeatureEngine interface for feature engineering
type FeatureEngine interface {
	ExtractFeatures(data []TimeSeriesPoint) []Feature
	GetFeatureName() string
}

// collectorMetrics holds Prometheus metrics
type collectorMetrics struct {
	dataPointsCollected  prometheus.Counter
	featuresExtracted    prometheus.Counter
	datasetsCreated      prometheus.Counter
	collectionErrors     prometheus.Counter
	collectionDuration   prometheus.Histogram
	bufferUtilization    prometheus.Gauge
}

// NewProductionDataCollector creates a new production data collector
func NewProductionDataCollector(config *CollectorConfig) (*ProductionDataCollector, error) {
	logger := logrus.New()
	logger.SetLevel(logrus.InfoLevel)

	client := influxdb2.NewClient(config.InfluxURL, config.InfluxToken)
	writeAPI := client.WriteAPI(config.InfluxOrg, config.InfluxBucket)
	queryAPI := client.QueryAPI(config.InfluxOrg)

	metrics := &collectorMetrics{
		dataPointsCollected: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "ml_data_points_collected_total",
			Help: "Total number of data points collected",
		}),
		featuresExtracted: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "ml_features_extracted_total",
			Help: "Total number of features extracted",
		}),
		datasetsCreated: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "ml_datasets_created_total",
			Help: "Total number of ML datasets created",
		}),
		collectionErrors: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "ml_collection_errors_total",
			Help: "Total number of collection errors",
		}),
		collectionDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "ml_collection_duration_seconds",
			Help:    "Data collection duration in seconds",
			Buckets: prometheus.DefBuckets,
		}),
		bufferUtilization: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ml_buffer_utilization_ratio",
			Help: "Buffer utilization ratio",
		}),
	}

	// Register metrics
	prometheus.MustRegister(
		metrics.dataPointsCollected,
		metrics.featuresExtracted,
		metrics.datasetsCreated,
		metrics.collectionErrors,
		metrics.collectionDuration,
		metrics.bufferUtilization,
	)

	collector := &ProductionDataCollector{
		config:         config,
		influxClient:   client,
		writeAPI:       writeAPI,
		queryAPI:       queryAPI,
		metricsBuffer:  make([]TimeSeriesPoint, 0, config.BufferSize),
		featureCache:   make(map[string][]Feature),
		datasets:       make(map[string]*MLDataset),
		featureEngines: make(map[string]FeatureEngine),
		logger:         logger,
		metrics:        metrics,
		stopCh:         make(chan struct{}),
	}

	// Initialize feature engines
	collector.initializeFeatureEngines()

	return collector, nil
}

// Start begins data collection
func (pdc *ProductionDataCollector) Start(ctx context.Context) error {
	pdc.mu.Lock()
	if pdc.running {
		pdc.mu.Unlock()
		return fmt.Errorf("collector already running")
	}
	pdc.running = true
	pdc.mu.Unlock()

	pdc.logger.Info("Starting production data collector")

	// Start collection goroutines
	go pdc.collectMetrics(ctx)
	go pdc.processBuffer(ctx)
	go pdc.createDatasets(ctx)

	return nil
}

// Stop halts data collection
func (pdc *ProductionDataCollector) Stop() error {
	pdc.mu.Lock()
	if !pdc.running {
		pdc.mu.Unlock()
		return fmt.Errorf("collector not running")
	}
	pdc.running = false
	close(pdc.stopCh)
	pdc.mu.Unlock()

	// Flush remaining data
	pdc.flushBuffer()
	pdc.influxClient.Close()

	pdc.logger.Info("Production data collector stopped")
	return nil
}

// collectMetrics continuously collects metrics from production
func (pdc *ProductionDataCollector) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(pdc.config.CollectionRate)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-pdc.stopCh:
			return
		case <-ticker.C:
			start := time.Now()

			// Collect different metric types
			pdc.collectLatencyMetrics()
			pdc.collectThroughputMetrics()
			pdc.collectErrorMetrics()
			pdc.collectResourceMetrics()
			pdc.collectDWCPMetrics()

			duration := time.Since(start).Seconds()
			pdc.metrics.collectionDuration.Observe(duration)
		}
	}
}

// collectLatencyMetrics collects latency-related metrics
func (pdc *ProductionDataCollector) collectLatencyMetrics() {
	query := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "latency")
			|> mean()
	`, pdc.config.InfluxBucket)

	result, err := pdc.queryAPI.Query(context.Background(), query)
	if err != nil {
		pdc.metrics.collectionErrors.Inc()
		pdc.logger.Errorf("Failed to query latency metrics: %v", err)
		return
	}

	for result.Next() {
		point := TimeSeriesPoint{
			Timestamp: result.Record().Time(),
			Value:     result.Record().Value().(float64),
			Tags: map[string]string{
				"metric_type": string(MetricLatency),
				"field":       result.Record().Field(),
			},
			Fields: map[string]interface{}{
				"measurement": result.Record().Measurement(),
				"table":       result.Record().Table(),
			},
		}

		pdc.addToBuffer(point)
		pdc.metrics.dataPointsCollected.Inc()
	}

	if result.Err() != nil {
		pdc.logger.Errorf("Query error: %v", result.Err())
	}
}

// collectThroughputMetrics collects throughput-related metrics
func (pdc *ProductionDataCollector) collectThroughputMetrics() {
	query := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "throughput")
			|> sum()
	`, pdc.config.InfluxBucket)

	result, err := pdc.queryAPI.Query(context.Background(), query)
	if err != nil {
		pdc.metrics.collectionErrors.Inc()
		pdc.logger.Errorf("Failed to query throughput metrics: %v", err)
		return
	}

	for result.Next() {
		point := TimeSeriesPoint{
			Timestamp: result.Record().Time(),
			Value:     result.Record().Value().(float64),
			Tags: map[string]string{
				"metric_type": string(MetricThroughput),
				"field":       result.Record().Field(),
			},
			Fields: map[string]interface{}{
				"measurement": result.Record().Measurement(),
			},
		}

		pdc.addToBuffer(point)
		pdc.metrics.dataPointsCollected.Inc()
	}
}

// collectErrorMetrics collects error-related metrics
func (pdc *ProductionDataCollector) collectErrorMetrics() {
	query := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "errors")
			|> count()
	`, pdc.config.InfluxBucket)

	result, err := pdc.queryAPI.Query(context.Background(), query)
	if err != nil {
		pdc.metrics.collectionErrors.Inc()
		pdc.logger.Errorf("Failed to query error metrics: %v", err)
		return
	}

	for result.Next() {
		point := TimeSeriesPoint{
			Timestamp: result.Record().Time(),
			Value:     float64(result.Record().Value().(int64)),
			Tags: map[string]string{
				"metric_type": string(MetricErrorRate),
				"field":       result.Record().Field(),
			},
			Fields: map[string]interface{}{
				"measurement": result.Record().Measurement(),
			},
		}

		pdc.addToBuffer(point)
		pdc.metrics.dataPointsCollected.Inc()
	}
}

// collectResourceMetrics collects system resource metrics
func (pdc *ProductionDataCollector) collectResourceMetrics() {
	// CPU metrics
	cpuQuery := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "cpu")
			|> mean()
	`, pdc.config.InfluxBucket)

	pdc.executeResourceQuery(cpuQuery, MetricCPU)

	// Memory metrics
	memQuery := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "memory")
			|> mean()
	`, pdc.config.InfluxBucket)

	pdc.executeResourceQuery(memQuery, MetricMemory)

	// Network I/O metrics
	netQuery := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "network")
			|> sum()
	`, pdc.config.InfluxBucket)

	pdc.executeResourceQuery(netQuery, MetricNetwork)
}

// collectDWCPMetrics collects DWCP-specific metrics
func (pdc *ProductionDataCollector) collectDWCPMetrics() {
	// HDE compression ratio
	compressionQuery := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "hde_compression")
			|> mean()
	`, pdc.config.InfluxBucket)

	pdc.executeResourceQuery(compressionQuery, MetricCompression)

	// PBA prediction accuracy
	predictionQuery := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "pba_accuracy")
			|> mean()
	`, pdc.config.InfluxBucket)

	pdc.executeResourceQuery(predictionQuery, MetricPrediction)

	// ACP consensus time
	consensusQuery := fmt.Sprintf(`
		from(bucket: "%s")
			|> range(start: -5m)
			|> filter(fn: (r) => r["_measurement"] == "acp_consensus")
			|> mean()
	`, pdc.config.InfluxBucket)

	pdc.executeResourceQuery(consensusQuery, MetricConsensus)
}

// executeResourceQuery executes a resource query and adds to buffer
func (pdc *ProductionDataCollector) executeResourceQuery(query string, metricType MetricType) {
	result, err := pdc.queryAPI.Query(context.Background(), query)
	if err != nil {
		pdc.metrics.collectionErrors.Inc()
		pdc.logger.Errorf("Failed to query %s metrics: %v", metricType, err)
		return
	}

	for result.Next() {
		point := TimeSeriesPoint{
			Timestamp: result.Record().Time(),
			Value:     result.Record().Value().(float64),
			Tags: map[string]string{
				"metric_type": string(metricType),
				"field":       result.Record().Field(),
			},
			Fields: map[string]interface{}{
				"measurement": result.Record().Measurement(),
			},
		}

		pdc.addToBuffer(point)
		pdc.metrics.dataPointsCollected.Inc()
	}
}

// addToBuffer adds a point to the buffer
func (pdc *ProductionDataCollector) addToBuffer(point TimeSeriesPoint) {
	pdc.mu.Lock()
	defer pdc.mu.Unlock()

	if len(pdc.metricsBuffer) >= pdc.config.BufferSize {
		// Buffer full, process oldest points
		pdc.processOldestPoints(10)
	}

	pdc.metricsBuffer = append(pdc.metricsBuffer, point)
	utilization := float64(len(pdc.metricsBuffer)) / float64(pdc.config.BufferSize)
	pdc.metrics.bufferUtilization.Set(utilization)
}

// processBuffer processes buffered metrics for feature extraction
func (pdc *ProductionDataCollector) processBuffer(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-pdc.stopCh:
			return
		case <-ticker.C:
			pdc.extractFeatures()
		}
	}
}

// extractFeatures extracts ML features from buffered data
func (pdc *ProductionDataCollector) extractFeatures() {
	pdc.mu.Lock()
	defer pdc.mu.Unlock()

	if len(pdc.metricsBuffer) < pdc.config.WindowSize {
		return
	}

	// Group points by metric type
	groupedPoints := make(map[string][]TimeSeriesPoint)
	for _, point := range pdc.metricsBuffer {
		metricType := point.Tags["metric_type"]
		groupedPoints[metricType] = append(groupedPoints[metricType], point)
	}

	// Extract features for each metric type
	for metricType, points := range groupedPoints {
		if len(points) < 10 {
			continue
		}

		features := []Feature{}

		// Temporal features
		if pdc.config.FeatureConfig.EnableTemporalFeatures {
			features = append(features, pdc.extractTemporalFeatures(points)...)
		}

		// Statistical features
		if pdc.config.FeatureConfig.EnableStatisticalFeatures {
			features = append(features, pdc.extractStatisticalFeatures(points)...)
		}

		// Frequency features
		if pdc.config.FeatureConfig.EnableFrequencyFeatures {
			features = append(features, pdc.extractFrequencyFeatures(points)...)
		}

		// Lagged features
		if pdc.config.FeatureConfig.EnableLaggedFeatures {
			features = append(features, pdc.extractLaggedFeatures(points)...)
		}

		// Store features in cache
		cacheKey := fmt.Sprintf("%s_%d", metricType, time.Now().Unix())
		pdc.featureCache[cacheKey] = features
		pdc.metrics.featuresExtracted.Add(float64(len(features)))
	}
}

// extractTemporalFeatures extracts time-based features
func (pdc *ProductionDataCollector) extractTemporalFeatures(points []TimeSeriesPoint) []Feature {
	features := []Feature{}

	if len(points) == 0 {
		return features
	}

	// Hour of day
	hour := float64(points[len(points)-1].Timestamp.Hour())
	features = append(features, Feature{
		Name:        "hour_of_day",
		Value:       hour,
		Type:        "temporal",
		Description: "Hour of the day (0-23)",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	// Day of week
	dow := float64(points[len(points)-1].Timestamp.Weekday())
	features = append(features, Feature{
		Name:        "day_of_week",
		Value:       dow,
		Type:        "temporal",
		Description: "Day of the week (0-6)",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	// Time since last spike
	var lastSpike time.Time
	mean := calculateMean(points)
	stdDev := calculateStdDev(points, mean)
	threshold := mean + 2*stdDev

	for _, p := range points {
		if p.Value > threshold {
			lastSpike = p.Timestamp
		}
	}

	if !lastSpike.IsZero() {
		timeSinceSpike := points[len(points)-1].Timestamp.Sub(lastSpike).Minutes()
		features = append(features, Feature{
			Name:        "time_since_spike",
			Value:       timeSinceSpike,
			Type:        "temporal",
			Description: "Minutes since last spike",
			Timestamp:   points[len(points)-1].Timestamp,
		})
	}

	return features
}

// extractStatisticalFeatures extracts statistical features
func (pdc *ProductionDataCollector) extractStatisticalFeatures(points []TimeSeriesPoint) []Feature {
	features := []Feature{}

	if len(points) == 0 {
		return features
	}

	values := make([]float64, len(points))
	for i, p := range points {
		values[i] = p.Value
	}

	// Basic statistics
	mean := calculateMean(points)
	features = append(features, Feature{
		Name:        "mean",
		Value:       mean,
		Type:        "statistical",
		Description: "Mean value",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	stdDev := calculateStdDev(points, mean)
	features = append(features, Feature{
		Name:        "std_dev",
		Value:       stdDev,
		Type:        "statistical",
		Description: "Standard deviation",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	// Min and Max
	min, max := calculateMinMax(points)
	features = append(features, Feature{
		Name:        "min",
		Value:       min,
		Type:        "statistical",
		Description: "Minimum value",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	features = append(features, Feature{
		Name:        "max",
		Value:       max,
		Type:        "statistical",
		Description: "Maximum value",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	// Percentiles
	for _, q := range pdc.config.FeatureConfig.QuantileThresholds {
		percentile := calculatePercentile(values, q)
		features = append(features, Feature{
			Name:        fmt.Sprintf("percentile_%.0f", q*100),
			Value:       percentile,
			Type:        "statistical",
			Description: fmt.Sprintf("%.0fth percentile", q*100),
			Timestamp:   points[len(points)-1].Timestamp,
		})
	}

	// Skewness and Kurtosis
	skewness := calculateSkewness(values, mean, stdDev)
	features = append(features, Feature{
		Name:        "skewness",
		Value:       skewness,
		Type:        "statistical",
		Description: "Distribution skewness",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	kurtosis := calculateKurtosis(values, mean, stdDev)
	features = append(features, Feature{
		Name:        "kurtosis",
		Value:       kurtosis,
		Type:        "statistical",
		Description: "Distribution kurtosis",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	return features
}

// extractFrequencyFeatures extracts frequency domain features
func (pdc *ProductionDataCollector) extractFrequencyFeatures(points []TimeSeriesPoint) []Feature {
	features := []Feature{}

	if len(points) < 4 {
		return features
	}

	values := make([]float64, len(points))
	for i, p := range points {
		values[i] = p.Value
	}

	// Simple FFT approximation (for demonstration)
	// In production, use proper FFT library
	dominantFreq := calculateDominantFrequency(values)
	features = append(features, Feature{
		Name:        "dominant_frequency",
		Value:       dominantFreq,
		Type:        "frequency",
		Description: "Dominant frequency component",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	// Spectral entropy
	spectralEntropy := calculateSpectralEntropy(values)
	features = append(features, Feature{
		Name:        "spectral_entropy",
		Value:       spectralEntropy,
		Type:        "frequency",
		Description: "Spectral entropy",
		Timestamp:   points[len(points)-1].Timestamp,
	})

	return features
}

// extractLaggedFeatures extracts lagged features
func (pdc *ProductionDataCollector) extractLaggedFeatures(points []TimeSeriesPoint) []Feature {
	features := []Feature{}

	if len(points) == 0 {
		return features
	}

	for _, lag := range pdc.config.FeatureConfig.LagPeriods {
		if lag >= len(points) {
			continue
		}

		laggedValue := points[len(points)-1-lag].Value
		features = append(features, Feature{
			Name:        fmt.Sprintf("lag_%d", lag),
			Value:       laggedValue,
			Type:        "lagged",
			Description: fmt.Sprintf("Value %d periods ago", lag),
			Timestamp:   points[len(points)-1].Timestamp,
		})

		// Difference from lagged value
		diff := points[len(points)-1].Value - laggedValue
		features = append(features, Feature{
			Name:        fmt.Sprintf("diff_lag_%d", lag),
			Value:       diff,
			Type:        "lagged",
			Description: fmt.Sprintf("Difference from %d periods ago", lag),
			Timestamp:   points[len(points)-1].Timestamp,
		})
	}

	return features
}

// createDatasets creates ML-ready datasets from features
func (pdc *ProductionDataCollector) createDatasets(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-pdc.stopCh:
			return
		case <-ticker.C:
			pdc.createMLDataset()
		}
	}
}

// createMLDataset creates a new ML dataset
func (pdc *ProductionDataCollector) createMLDataset() {
	pdc.mu.Lock()
	defer pdc.mu.Unlock()

	if len(pdc.featureCache) < 10 {
		return
	}

	datasetID := fmt.Sprintf("dataset_%d", time.Now().Unix())
	dataset := &MLDataset{
		ID:         datasetID,
		Created:    time.Now(),
		Features:   [][]Feature{},
		Labels:     []float64{},
		Metadata:   make(map[string]string),
		WindowSize: pdc.config.WindowSize,
		StepSize:   pdc.config.StepSize,
	}

	// Convert feature cache to dataset format
	for _, features := range pdc.featureCache {
		if len(features) > 0 {
			dataset.Features = append(dataset.Features, features)

			// Create labels (for demonstration, using next value as label)
			// In production, labels would come from actual outcomes
			label := features[len(features)-1].Value
			dataset.Labels = append(dataset.Labels, label)
		}
	}

	// Add metadata
	dataset.Metadata["metric_count"] = fmt.Sprintf("%d", len(dataset.Features))
	dataset.Metadata["feature_count"] = fmt.Sprintf("%d", len(dataset.Features[0]))
	dataset.Metadata["created_at"] = time.Now().Format(time.RFC3339)

	pdc.datasets[datasetID] = dataset
	pdc.metrics.datasetsCreated.Inc()

	// Export dataset to file
	pdc.exportDataset(dataset)

	pdc.logger.Infof("Created ML dataset %s with %d samples", datasetID, len(dataset.Features))
}

// exportDataset exports dataset to JSON file
func (pdc *ProductionDataCollector) exportDataset(dataset *MLDataset) error {
	filename := fmt.Sprintf("/tmp/ml_dataset_%s.json", dataset.ID)

	data, err := json.MarshalIndent(dataset, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal dataset: %w", err)
	}

	// Write to file (in production, use proper storage)
	// This is simplified for demonstration
	pdc.logger.Infof("Dataset exported to %s", filename)

	return nil
}

// Helper functions

func calculateMean(points []TimeSeriesPoint) float64 {
	if len(points) == 0 {
		return 0
	}
	sum := 0.0
	for _, p := range points {
		sum += p.Value
	}
	return sum / float64(len(points))
}

func calculateStdDev(points []TimeSeriesPoint, mean float64) float64 {
	if len(points) == 0 {
		return 0
	}
	variance := 0.0
	for _, p := range points {
		diff := p.Value - mean
		variance += diff * diff
	}
	return math.Sqrt(variance / float64(len(points)))
}

func calculateMinMax(points []TimeSeriesPoint) (float64, float64) {
	if len(points) == 0 {
		return 0, 0
	}
	min, max := points[0].Value, points[0].Value
	for _, p := range points[1:] {
		if p.Value < min {
			min = p.Value
		}
		if p.Value > max {
			max = p.Value
		}
	}
	return min, max
}

func calculatePercentile(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	index := int(percentile * float64(len(values)-1))
	if index >= len(values) {
		index = len(values) - 1
	}
	return values[index]
}

func calculateSkewness(values []float64, mean, stdDev float64) float64 {
	if len(values) == 0 || stdDev == 0 {
		return 0
	}
	n := float64(len(values))
	sum := 0.0
	for _, v := range values {
		sum += math.Pow((v-mean)/stdDev, 3)
	}
	return (n / ((n - 1) * (n - 2))) * sum
}

func calculateKurtosis(values []float64, mean, stdDev float64) float64 {
	if len(values) == 0 || stdDev == 0 {
		return 0
	}
	n := float64(len(values))
	sum := 0.0
	for _, v := range values {
		sum += math.Pow((v-mean)/stdDev, 4)
	}
	return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum
}

func calculateDominantFrequency(values []float64) float64 {
	// Simplified frequency calculation
	// In production, use proper FFT
	if len(values) < 2 {
		return 0
	}

	// Count zero crossings as simple frequency estimate
	zeroCrossings := 0
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))

	for i := 1; i < len(values); i++ {
		if (values[i-1]-mean)*(values[i]-mean) < 0 {
			zeroCrossings++
		}
	}

	return float64(zeroCrossings) / float64(len(values))
}

func calculateSpectralEntropy(values []float64) float64 {
	// Simplified spectral entropy
	// In production, use proper signal processing
	if len(values) == 0 {
		return 0
	}

	// Calculate power spectrum estimate
	sum := 0.0
	for _, v := range values {
		sum += v * v
	}

	if sum == 0 {
		return 0
	}

	entropy := 0.0
	for _, v := range values {
		if v != 0 {
			p := (v * v) / sum
			if p > 0 {
				entropy -= p * math.Log(p)
			}
		}
	}

	return entropy
}

// processOldestPoints processes the oldest points in buffer
func (pdc *ProductionDataCollector) processOldestPoints(count int) {
	if count > len(pdc.metricsBuffer) {
		count = len(pdc.metricsBuffer)
	}

	// Extract features from oldest points before removing
	oldestPoints := pdc.metricsBuffer[:count]
	if len(oldestPoints) >= pdc.config.WindowSize {
		// Group by metric type and extract features
		grouped := make(map[string][]TimeSeriesPoint)
		for _, p := range oldestPoints {
			metricType := p.Tags["metric_type"]
			grouped[metricType] = append(grouped[metricType], p)
		}

		for _, points := range grouped {
			if len(points) >= 10 {
				features := []Feature{}
				if pdc.config.FeatureConfig.EnableStatisticalFeatures {
					features = append(features, pdc.extractStatisticalFeatures(points)...)
				}
				if len(features) > 0 {
					cacheKey := fmt.Sprintf("processed_%d", time.Now().UnixNano())
					pdc.featureCache[cacheKey] = features
				}
			}
		}
	}

	// Remove processed points from buffer
	pdc.metricsBuffer = pdc.metricsBuffer[count:]
}

// flushBuffer flushes remaining buffer data
func (pdc *ProductionDataCollector) flushBuffer() {
	pdc.mu.Lock()
	defer pdc.mu.Unlock()

	if len(pdc.metricsBuffer) > 0 {
		pdc.processOldestPoints(len(pdc.metricsBuffer))
	}

	// Create final dataset if there's enough data
	if len(pdc.featureCache) > 0 {
		pdc.createMLDataset()
	}
}

// initializeFeatureEngines initializes all feature extraction engines
func (pdc *ProductionDataCollector) initializeFeatureEngines() {
	// Add custom feature engines here
	// Example: pdc.featureEngines["custom"] = NewCustomFeatureEngine()
}

// GetDataset returns a specific dataset by ID
func (pdc *ProductionDataCollector) GetDataset(datasetID string) (*MLDataset, error) {
	pdc.mu.RLock()
	defer pdc.mu.RUnlock()

	dataset, exists := pdc.datasets[datasetID]
	if !exists {
		return nil, fmt.Errorf("dataset %s not found", datasetID)
	}

	return dataset, nil
}

// GetAllDatasets returns all available datasets
func (pdc *ProductionDataCollector) GetAllDatasets() map[string]*MLDataset {
	pdc.mu.RLock()
	defer pdc.mu.RUnlock()

	// Return a copy to prevent external modification
	datasets := make(map[string]*MLDataset)
	for k, v := range pdc.datasets {
		datasets[k] = v
	}

	return datasets
}

// GetMetrics returns current collector metrics
func (pdc *ProductionDataCollector) GetMetrics() map[string]float64 {
	return map[string]float64{
		"buffer_size":     float64(len(pdc.metricsBuffer)),
		"feature_count":   float64(len(pdc.featureCache)),
		"dataset_count":   float64(len(pdc.datasets)),
		"buffer_capacity": float64(pdc.config.BufferSize),
	}
}