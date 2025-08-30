package ml_anomaly

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"
)

// AnomalyDetector provides ML-based anomaly detection for NovaCron metrics
type AnomalyDetector struct {
	// Models and configuration
	models map[string]AnomalyModel
	config *DetectorConfig
	
	// Data storage
	metricHistory map[string][]MetricDataPoint
	anomalies     []Anomaly
	
	// Training and detection
	trainedModels map[string]*TrainedModel
	predictions   map[string][]Prediction
	
	// Alert management
	alertManager AlertManager
	
	// Concurrency control
	mutex         sync.RWMutex
	modelsMutex   sync.RWMutex
	historymutex  sync.RWMutex
	
	// Background processing
	ctx           context.Context
	cancel        context.CancelFunc
	wg            sync.WaitGroup
}

// DetectorConfig represents the configuration for anomaly detection
type DetectorConfig struct {
	// Model configuration
	EnabledModels          []string      `json:"enabled_models"`
	TrainingWindow         time.Duration `json:"training_window"`
	DetectionWindow        time.Duration `json:"detection_window"`
	RetrainingInterval     time.Duration `json:"retraining_interval"`
	
	// Sensitivity settings
	SensitivityLevel       string  `json:"sensitivity_level"` // "low", "medium", "high"
	ConfidenceThreshold    float64 `json:"confidence_threshold"`
	AnomalyThreshold       float64 `json:"anomaly_threshold"`
	
	// Data processing
	MinDataPoints          int           `json:"min_data_points"`
	MaxHistorySize         int           `json:"max_history_size"`
	SamplingInterval       time.Duration `json:"sampling_interval"`
	SeasonalityPeriod      time.Duration `json:"seasonality_period"`
	
	// Alert configuration
	EnableAlerting         bool          `json:"enable_alerting"`
	AlertCooldownPeriod    time.Duration `json:"alert_cooldown_period"`
	EscalationThreshold    int           `json:"escalation_threshold"`
	
	// Advanced features
	EnablePredictiveAlerts bool    `json:"enable_predictive_alerts"`
	PredictionHorizon      time.Duration `json:"prediction_horizon"`
	EnableCorrelation      bool    `json:"enable_correlation"`
	CorrelationWindow      time.Duration `json:"correlation_window"`
}

// DefaultDetectorConfig returns default configuration
func DefaultDetectorConfig() *DetectorConfig {
	return &DetectorConfig{
		EnabledModels: []string{
			"statistical", "isolation_forest", "lstm", "seasonal_decomposition",
		},
		TrainingWindow:         24 * time.Hour,
		DetectionWindow:        5 * time.Minute,
		RetrainingInterval:     6 * time.Hour,
		SensitivityLevel:       "medium",
		ConfidenceThreshold:    0.8,
		AnomalyThreshold:       2.5, // Standard deviations
		MinDataPoints:          100,
		MaxHistorySize:         10000,
		SamplingInterval:       1 * time.Minute,
		SeasonalityPeriod:      24 * time.Hour,
		EnableAlerting:         true,
		AlertCooldownPeriod:    15 * time.Minute,
		EscalationThreshold:    3,
		EnablePredictiveAlerts: true,
		PredictionHorizon:      30 * time.Minute,
		EnableCorrelation:      true,
		CorrelationWindow:      10 * time.Minute,
	}
}

// MetricDataPoint represents a single metric data point
type MetricDataPoint struct {
	Timestamp time.Time              `json:"timestamp"`
	Value     float64                `json:"value"`
	Labels    map[string]string      `json:"labels"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// Anomaly represents a detected anomaly
type Anomaly struct {
	ID            string                 `json:"id"`
	MetricName    string                 `json:"metric_name"`
	Timestamp     time.Time              `json:"timestamp"`
	Value         float64                `json:"value"`
	ExpectedValue float64                `json:"expected_value"`
	Severity      AnomalySeverity        `json:"severity"`
	Confidence    float64                `json:"confidence"`
	ModelUsed     string                 `json:"model_used"`
	Description   string                 `json:"description"`
	Labels        map[string]string      `json:"labels"`
	Context       map[string]interface{} `json:"context"`
	AlertSent     bool                   `json:"alert_sent"`
	Acknowledged  bool                   `json:"acknowledged"`
	CreatedAt     time.Time              `json:"created_at"`
}

// AnomalySeverity represents the severity level of an anomaly
type AnomalySeverity string

const (
	SeverityLow      AnomalySeverity = "low"
	SeverityMedium   AnomalySeverity = "medium"
	SeverityHigh     AnomalySeverity = "high"
	SeverityCritical AnomalySeverity = "critical"
)

// Prediction represents a future prediction
type Prediction struct {
	MetricName  string                 `json:"metric_name"`
	Timestamp   time.Time              `json:"timestamp"`
	Value       float64                `json:"value"`
	Confidence  float64                `json:"confidence"`
	UpperBound  float64                `json:"upper_bound"`
	LowerBound  float64                `json:"lower_bound"`
	ModelUsed   string                 `json:"model_used"`
	Context     map[string]interface{} `json:"context"`
	CreatedAt   time.Time              `json:"created_at"`
}

// AnomalyModel defines the interface for anomaly detection models
type AnomalyModel interface {
	Train(data []MetricDataPoint) error
	Detect(data []MetricDataPoint) ([]Anomaly, error)
	Predict(data []MetricDataPoint, horizon time.Duration) ([]Prediction, error)
	GetName() string
	GetParameters() map[string]interface{}
	SetParameters(params map[string]interface{}) error
	IsReady() bool
}

// TrainedModel represents a trained anomaly detection model
type TrainedModel struct {
	Name           string                 `json:"name"`
	MetricName     string                 `json:"metric_name"`
	TrainedAt      time.Time              `json:"trained_at"`
	TrainingData   int                    `json:"training_data_points"`
	Parameters     map[string]interface{} `json:"parameters"`
	Performance    ModelPerformance       `json:"performance"`
	LastPrediction time.Time              `json:"last_prediction"`
}

// ModelPerformance represents model performance metrics
type ModelPerformance struct {
	Accuracy      float64 `json:"accuracy"`
	Precision     float64 `json:"precision"`
	Recall        float64 `json:"recall"`
	F1Score       float64 `json:"f1_score"`
	FalsePositive float64 `json:"false_positive_rate"`
	TruePositive  float64 `json:"true_positive_rate"`
}

// AlertManager interface for sending anomaly alerts
type AlertManager interface {
	SendAnomalyAlert(ctx context.Context, anomaly *Anomaly) error
	SendPredictiveAlert(ctx context.Context, prediction *Prediction) error
	ShouldAlert(anomaly *Anomaly) bool
}

// NewAnomalyDetector creates a new anomaly detector
func NewAnomalyDetector(config *DetectorConfig, alertManager AlertManager) *AnomalyDetector {
	if config == nil {
		config = DefaultDetectorConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	detector := &AnomalyDetector{
		models:        make(map[string]AnomalyModel),
		config:        config,
		metricHistory: make(map[string][]MetricDataPoint),
		trainedModels: make(map[string]*TrainedModel),
		predictions:   make(map[string][]Prediction),
		alertManager:  alertManager,
		ctx:           ctx,
		cancel:        cancel,
	}

	// Initialize models
	detector.initializeModels()

	return detector
}

// Start starts the anomaly detector
func (d *AnomalyDetector) Start() error {
	log.Println("Starting ML-based anomaly detector...")

	// Start background workers
	d.wg.Add(3)
	go d.trainingWorker()
	go d.detectionWorker()
	go d.maintenanceWorker()

	log.Printf("Anomaly detector started with models: %v", d.config.EnabledModels)
	return nil
}

// Stop stops the anomaly detector
func (d *AnomalyDetector) Stop() error {
	log.Println("Stopping anomaly detector...")

	d.cancel()
	d.wg.Wait()

	log.Println("Anomaly detector stopped")
	return nil
}

// AddMetricData adds new metric data for analysis
func (d *AnomalyDetector) AddMetricData(metricName string, dataPoint MetricDataPoint) error {
	d.historymutex.Lock()
	defer d.historymutex.Unlock()

	if d.metricHistory[metricName] == nil {
		d.metricHistory[metricName] = make([]MetricDataPoint, 0)
	}

	// Add data point
	d.metricHistory[metricName] = append(d.metricHistory[metricName], dataPoint)

	// Maintain history size limit
	if len(d.metricHistory[metricName]) > d.config.MaxHistorySize {
		// Remove oldest data points
		excess := len(d.metricHistory[metricName]) - d.config.MaxHistorySize
		d.metricHistory[metricName] = d.metricHistory[metricName][excess:]
	}

	return nil
}

// DetectAnomalies performs anomaly detection on recent data
func (d *AnomalyDetector) DetectAnomalies(metricName string) ([]Anomaly, error) {
	d.historymutex.RLock()
	history := d.metricHistory[metricName]
	d.historymutex.RUnlock()

	if len(history) < d.config.MinDataPoints {
		return nil, fmt.Errorf("insufficient data points for %s: need %d, have %d",
			metricName, d.config.MinDataPoints, len(history))
	}

	// Get detection window data
	now := time.Now()
	windowStart := now.Add(-d.config.DetectionWindow)
	
	var windowData []MetricDataPoint
	for _, point := range history {
		if point.Timestamp.After(windowStart) {
			windowData = append(windowData, point)
		}
	}

	var allAnomalies []Anomaly

	// Run detection with enabled models
	for _, modelName := range d.config.EnabledModels {
		d.modelsMutex.RLock()
		model, exists := d.models[modelName]
		d.modelsMutex.RUnlock()

		if !exists || !model.IsReady() {
			continue
		}

		anomalies, err := model.Detect(windowData)
		if err != nil {
			log.Printf("Error detecting anomalies with %s model: %v", modelName, err)
			continue
		}

		// Filter by confidence threshold
		for _, anomaly := range anomalies {
			if anomaly.Confidence >= d.config.ConfidenceThreshold {
				anomaly.MetricName = metricName
				anomaly.ModelUsed = modelName
				anomaly.CreatedAt = time.Now()
				allAnomalies = append(allAnomalies, anomaly)
			}
		}
	}

	// Deduplicate and rank anomalies
	uniqueAnomalies := d.deduplicateAnomalies(allAnomalies)

	// Store anomalies
	d.mutex.Lock()
	d.anomalies = append(d.anomalies, uniqueAnomalies...)
	d.mutex.Unlock()

	// Send alerts if enabled
	if d.config.EnableAlerting && d.alertManager != nil {
		for _, anomaly := range uniqueAnomalies {
			if d.alertManager.ShouldAlert(&anomaly) {
				go func(a Anomaly) {
					if err := d.alertManager.SendAnomalyAlert(context.Background(), &a); err != nil {
						log.Printf("Error sending anomaly alert: %v", err)
					}
				}(anomaly)
			}
		}
	}

	return uniqueAnomalies, nil
}

// PredictValues predicts future values for a metric
func (d *AnomalyDetector) PredictValues(metricName string, horizon time.Duration) ([]Prediction, error) {
	d.historymutex.RLock()
	history := d.metricHistory[metricName]
	d.historymutex.RUnlock()

	if len(history) < d.config.MinDataPoints {
		return nil, fmt.Errorf("insufficient data points for prediction")
	}

	var allPredictions []Prediction

	// Run prediction with enabled models
	for _, modelName := range d.config.EnabledModels {
		d.modelsMutex.RLock()
		model, exists := d.models[modelName]
		d.modelsMutex.RUnlock()

		if !exists || !model.IsReady() {
			continue
		}

		predictions, err := model.Predict(history, horizon)
		if err != nil {
			log.Printf("Error predicting with %s model: %v", modelName, err)
			continue
		}

		for _, pred := range predictions {
			pred.MetricName = metricName
			pred.ModelUsed = modelName
			pred.CreatedAt = time.Now()
			allPredictions = append(allPredictions, pred)
		}
	}

	// Store predictions
	d.mutex.Lock()
	d.predictions[metricName] = allPredictions
	d.mutex.Unlock()

	return allPredictions, nil
}

// GetAnomalies returns recent anomalies
func (d *AnomalyDetector) GetAnomalies(since time.Time) []Anomaly {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	var result []Anomaly
	for _, anomaly := range d.anomalies {
		if anomaly.Timestamp.After(since) {
			result = append(result, anomaly)
		}
	}

	return result
}

// GetPredictions returns predictions for a metric
func (d *AnomalyDetector) GetPredictions(metricName string) []Prediction {
	d.mutex.RLock()
	defer d.mutex.RUnlock()

	if predictions, exists := d.predictions[metricName]; exists {
		return predictions
	}

	return []Prediction{}
}

// TrainModel trains a specific model for a metric
func (d *AnomalyDetector) TrainModel(metricName, modelName string) error {
	d.historymutex.RLock()
	history := d.metricHistory[metricName]
	d.historymutex.RUnlock()

	if len(history) < d.config.MinDataPoints {
		return fmt.Errorf("insufficient data points for training")
	}

	// Get training window data
	now := time.Now()
	windowStart := now.Add(-d.config.TrainingWindow)
	
	var trainingData []MetricDataPoint
	for _, point := range history {
		if point.Timestamp.After(windowStart) {
			trainingData = append(trainingData, point)
		}
	}

	d.modelsMutex.RLock()
	model, exists := d.models[modelName]
	d.modelsMutex.RUnlock()

	if !exists {
		return fmt.Errorf("model %s not found", modelName)
	}

	// Train the model
	if err := model.Train(trainingData); err != nil {
		return fmt.Errorf("failed to train model %s: %w", modelName, err)
	}

	// Update trained model info
	d.modelsMutex.Lock()
	d.trainedModels[fmt.Sprintf("%s:%s", metricName, modelName)] = &TrainedModel{
		Name:         modelName,
		MetricName:   metricName,
		TrainedAt:    time.Now(),
		TrainingData: len(trainingData),
		Parameters:   model.GetParameters(),
		Performance:  d.evaluateModel(model, trainingData),
	}
	d.modelsMutex.Unlock()

	log.Printf("Trained model %s for metric %s with %d data points",
		modelName, metricName, len(trainingData))

	return nil
}

// GetModelPerformance returns performance metrics for trained models
func (d *AnomalyDetector) GetModelPerformance() map[string]*TrainedModel {
	d.modelsMutex.RLock()
	defer d.modelsMutex.RUnlock()

	result := make(map[string]*TrainedModel)
	for key, model := range d.trainedModels {
		modelCopy := *model
		result[key] = &modelCopy
	}

	return result
}

// Background workers

func (d *AnomalyDetector) trainingWorker() {
	defer d.wg.Done()

	ticker := time.NewTicker(d.config.RetrainingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.retrainModels()
		}
	}
}

func (d *AnomalyDetector) detectionWorker() {
	defer d.wg.Done()

	ticker := time.NewTicker(d.config.DetectionWindow)
	defer ticker.Stop()

	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.runDetection()
		}
	}
}

func (d *AnomalyDetector) maintenanceWorker() {
	defer d.wg.Done()

	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-d.ctx.Done():
			return
		case <-ticker.C:
			d.performMaintenance()
		}
	}
}

// Helper methods

func (d *AnomalyDetector) initializeModels() {
	// Initialize statistical model
	if d.isModelEnabled("statistical") {
		d.models["statistical"] = NewStatisticalModel(d.config)
	}

	// Initialize isolation forest model
	if d.isModelEnabled("isolation_forest") {
		d.models["isolation_forest"] = NewIsolationForestModel(d.config)
	}

	// Initialize LSTM model
	if d.isModelEnabled("lstm") {
		d.models["lstm"] = NewLSTMModel(d.config)
	}

	// Initialize seasonal decomposition model
	if d.isModelEnabled("seasonal_decomposition") {
		d.models["seasonal_decomposition"] = NewSeasonalDecompositionModel(d.config)
	}
}

func (d *AnomalyDetector) isModelEnabled(modelName string) bool {
	for _, enabled := range d.config.EnabledModels {
		if enabled == modelName {
			return true
		}
	}
	return false
}

func (d *AnomalyDetector) retrainModels() {
	d.historymutex.RLock()
	metrics := make([]string, 0, len(d.metricHistory))
	for metric := range d.metricHistory {
		metrics = append(metrics, metric)
	}
	d.historymutex.RUnlock()

	for _, metric := range metrics {
		for _, modelName := range d.config.EnabledModels {
			if err := d.TrainModel(metric, modelName); err != nil {
				log.Printf("Error retraining model %s for metric %s: %v", modelName, metric, err)
			}
		}
	}

	log.Printf("Completed model retraining for %d metrics", len(metrics))
}

func (d *AnomalyDetector) runDetection() {
	d.historymutex.RLock()
	metrics := make([]string, 0, len(d.metricHistory))
	for metric := range d.metricHistory {
		metrics = append(metrics, metric)
	}
	d.historymutex.RUnlock()

	for _, metric := range metrics {
		if _, err := d.DetectAnomalies(metric); err != nil {
			log.Printf("Error detecting anomalies for metric %s: %v", metric, err)
		}
	}
}

func (d *AnomalyDetector) performMaintenance() {
	// Clean old anomalies
	cutoff := time.Now().Add(-24 * time.Hour)
	d.mutex.Lock()
	var filteredAnomalies []Anomaly
	for _, anomaly := range d.anomalies {
		if anomaly.CreatedAt.After(cutoff) {
			filteredAnomalies = append(filteredAnomalies, anomaly)
		}
	}
	d.anomalies = filteredAnomalies
	d.mutex.Unlock()

	// Clean old predictions
	d.mutex.Lock()
	for metric, predictions := range d.predictions {
		var filteredPredictions []Prediction
		for _, pred := range predictions {
			if pred.CreatedAt.After(cutoff) {
				filteredPredictions = append(filteredPredictions, pred)
			}
		}
		if len(filteredPredictions) > 0 {
			d.predictions[metric] = filteredPredictions
		} else {
			delete(d.predictions, metric)
		}
	}
	d.mutex.Unlock()

	log.Println("Performed anomaly detector maintenance")
}

func (d *AnomalyDetector) deduplicateAnomalies(anomalies []Anomaly) []Anomaly {
	if len(anomalies) == 0 {
		return anomalies
	}

	// Sort by confidence (highest first)
	sort.Slice(anomalies, func(i, j int) bool {
		return anomalies[i].Confidence > anomalies[j].Confidence
	})

	var unique []Anomaly
	seen := make(map[string]bool)

	for _, anomaly := range anomalies {
		key := fmt.Sprintf("%s:%d", anomaly.MetricName, anomaly.Timestamp.Unix())
		if !seen[key] {
			unique = append(unique, anomaly)
			seen[key] = true
		}
	}

	return unique
}

func (d *AnomalyDetector) evaluateModel(model AnomalyModel, testData []MetricDataPoint) ModelPerformance {
	// This is a simplified evaluation - in production, you would use proper validation sets
	// and more sophisticated metrics
	
	// For now, return mock performance metrics
	return ModelPerformance{
		Accuracy:      0.85,
		Precision:     0.80,
		Recall:        0.75,
		F1Score:       0.77,
		FalsePositive: 0.15,
		TruePositive:  0.75,
	}
}

// Utility functions for anomaly severity calculation
func (d *AnomalyDetector) calculateSeverity(value, expected, threshold float64) AnomalySeverity {
	deviation := math.Abs(value - expected)
	normalizedDeviation := deviation / math.Abs(expected)

	switch {
	case normalizedDeviation >= 0.5:
		return SeverityCritical
	case normalizedDeviation >= 0.3:
		return SeverityHigh
	case normalizedDeviation >= 0.1:
		return SeverityMedium
	default:
		return SeverityLow
	}
}

// JSON marshaling support
func (a *Anomaly) MarshalJSON() ([]byte, error) {
	type Alias Anomaly
	return json.Marshal(&struct {
		*Alias
		TimestampStr string `json:"timestamp_str"`
		CreatedAtStr string `json:"created_at_str"`
	}{
		Alias:        (*Alias)(a),
		TimestampStr: a.Timestamp.Format(time.RFC3339),
		CreatedAtStr: a.CreatedAt.Format(time.RFC3339),
	})
}

func (p *Prediction) MarshalJSON() ([]byte, error) {
	type Alias Prediction
	return json.Marshal(&struct {
		*Alias
		TimestampStr string `json:"timestamp_str"`
		CreatedAtStr string `json:"created_at_str"`
	}{
		Alias:        (*Alias)(p),
		TimestampStr: p.Timestamp.Format(time.RFC3339),
		CreatedAtStr: p.CreatedAt.Format(time.RFC3339),
	})
}