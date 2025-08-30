package predictive

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// PredictiveEngine manages multiple predictive models for different metrics
type PredictiveEngine struct {
	mu               sync.RWMutex
	models           map[string]PredictiveModel // metric name -> model
	metricsProvider  monitoring.MetricsProvider
	modelFactory     *ModelFactory
	config           PredictiveEngineConfig
	
	// Background update management
	ctx              context.Context
	cancel           context.CancelFunc
	updateTicker     *time.Ticker
	
	// Model performance tracking
	modelPerformance map[string]*ModelPerformance
	
	// Forecast cache
	forecastCache    map[string]*CachedForecast
	cacheMutex       sync.RWMutex
}

// PredictiveEngineConfig configures the predictive engine
type PredictiveEngineConfig struct {
	// Default model configurations
	DefaultModels    map[string]ModelConfig `json:"default_models"`
	
	// Update intervals
	ModelUpdateInterval    time.Duration `json:"model_update_interval"`
	ForecastUpdateInterval time.Duration `json:"forecast_update_interval"`
	
	// Cache settings
	ForecastCacheTTL       time.Duration `json:"forecast_cache_ttl"`
	MaxCachedForecasts     int           `json:"max_cached_forecasts"`
	
	// Model selection
	AutoModelSelection     bool          `json:"auto_model_selection"`
	MinAccuracyThreshold   float64       `json:"min_accuracy_threshold"`
	
	// Training data requirements
	MinTrainingPeriod      time.Duration `json:"min_training_period"`
	MaxTrainingPeriod      time.Duration `json:"max_training_period"`
	
	// Performance monitoring
	EnablePerformanceTracking bool       `json:"enable_performance_tracking"`
	PerformanceWindow        time.Duration `json:"performance_window"`
}

// ModelPerformance tracks the performance of a predictive model
type ModelPerformance struct {
	ModelName         string              `json:"model_name"`
	ModelType         ModelType           `json:"model_type"`
	Accuracy          float64             `json:"accuracy"`
	LastTraining      time.Time           `json:"last_training"`
	LastUpdate        time.Time           `json:"last_update"`
	TotalPredictions  int                 `json:"total_predictions"`
	
	// Prediction accuracy over time
	RecentAccuracy    []AccuracyPoint     `json:"recent_accuracy"`
	
	// Error metrics
	MAE               float64             `json:"mae"` // Mean Absolute Error
	RMSE              float64             `json:"rmse"` // Root Mean Square Error
	MAPE              float64             `json:"mape"` // Mean Absolute Percentage Error
	
	// Training performance
	TrainingTime      time.Duration       `json:"training_time"`
	DataPoints        int                 `json:"data_points"`
	
	// Model-specific metrics
	Metadata          map[string]interface{} `json:"metadata"`
}

// AccuracyPoint represents accuracy at a specific point in time
type AccuracyPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Accuracy  float64   `json:"accuracy"`
	Error     float64   `json:"error"`
}

// CachedForecast represents a cached forecast result
type CachedForecast struct {
	MetricName  string          `json:"metric_name"`
	Forecast    *ForecastResult `json:"forecast"`
	CreatedAt   time.Time       `json:"created_at"`
	ExpiresAt   time.Time       `json:"expires_at"`
	Horizon     time.Duration   `json:"horizon"`
	Interval    time.Duration   `json:"interval"`
}

// ForecastRequest represents a request for predictions
type ForecastRequest struct {
	MetricName string        `json:"metric_name"`
	Horizon    time.Duration `json:"horizon"`
	Interval   time.Duration `json:"interval"`
	Force      bool          `json:"force"` // Force refresh, ignore cache
	
	// Optional model override
	ModelType  ModelType     `json:"model_type,omitempty"`
	ModelConfig *ModelConfig `json:"model_config,omitempty"`
}

// NewPredictiveEngine creates a new predictive engine
func NewPredictiveEngine(metricsProvider monitoring.MetricsProvider, config PredictiveEngineConfig) *PredictiveEngine {
	ctx, cancel := context.WithCancel(context.Background())
	
	if config.ModelUpdateInterval == 0 {
		config.ModelUpdateInterval = 1 * time.Hour
	}
	if config.ForecastUpdateInterval == 0 {
		config.ForecastUpdateInterval = 15 * time.Minute
	}
	if config.ForecastCacheTTL == 0 {
		config.ForecastCacheTTL = 30 * time.Minute
	}
	if config.MinAccuracyThreshold == 0 {
		config.MinAccuracyThreshold = 0.6
	}
	if config.MinTrainingPeriod == 0 {
		config.MinTrainingPeriod = 24 * time.Hour
	}
	if config.MaxTrainingPeriod == 0 {
		config.MaxTrainingPeriod = 7 * 24 * time.Hour
	}
	if config.MaxCachedForecasts == 0 {
		config.MaxCachedForecasts = 100
	}
	
	return &PredictiveEngine{
		models:           make(map[string]PredictiveModel),
		metricsProvider:  metricsProvider,
		modelFactory:     &ModelFactory{},
		config:           config,
		ctx:              ctx,
		cancel:           cancel,
		modelPerformance: make(map[string]*ModelPerformance),
		forecastCache:    make(map[string]*CachedForecast),
	}
}

// Start starts the predictive engine background processes
func (e *PredictiveEngine) Start() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	// Start update ticker
	e.updateTicker = time.NewTicker(e.config.ModelUpdateInterval)
	
	// Start background goroutines
	go e.modelUpdateLoop()
	go e.forecastUpdateLoop()
	if e.config.EnablePerformanceTracking {
		go e.performanceTrackingLoop()
	}
	
	return nil
}

// Stop stops the predictive engine
func (e *PredictiveEngine) Stop() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if e.cancel != nil {
		e.cancel()
	}
	
	if e.updateTicker != nil {
		e.updateTicker.Stop()
	}
	
	return nil
}

// RegisterMetric registers a metric for predictive modeling
func (e *PredictiveEngine) RegisterMetric(metricName string, modelConfig ModelConfig) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	// Create model
	model, err := e.modelFactory.CreateModel(modelConfig)
	if err != nil {
		return fmt.Errorf("failed to create model for metric %s: %w", metricName, err)
	}
	
	// Store model
	e.models[metricName] = model
	
	// Initialize performance tracking
	if e.config.EnablePerformanceTracking {
		e.modelPerformance[metricName] = &ModelPerformance{
			ModelName: metricName,
			ModelType: modelConfig.Type,
			LastUpdate: time.Now(),
			RecentAccuracy: make([]AccuracyPoint, 0),
			Metadata: make(map[string]interface{}),
		}
	}
	
	// Train model with historical data
	go func() {
		if err := e.trainModel(metricName); err != nil {
			fmt.Printf("Warning: failed to train model for %s: %v\n", metricName, err)
		}
	}()
	
	return nil
}

// UnregisterMetric unregisters a metric
func (e *PredictiveEngine) UnregisterMetric(metricName string) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	delete(e.models, metricName)
	delete(e.modelPerformance, metricName)
	
	// Clear cached forecasts
	e.clearCachedForecastsForMetric(metricName)
	
	return nil
}

// GetForecast gets a forecast for a metric
func (e *PredictiveEngine) GetForecast(ctx context.Context, request ForecastRequest) (*ForecastResult, error) {
	// Check cache first (unless force refresh requested)
	if !request.Force {
		if cached := e.getCachedForecast(request); cached != nil {
			return cached.Forecast, nil
		}
	}
	
	e.mu.RLock()
	model, exists := e.models[request.MetricName]
	e.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("no model registered for metric: %s", request.MetricName)
	}
	
	if !model.IsReady() {
		return nil, fmt.Errorf("model for metric %s is not ready", request.MetricName)
	}
	
	// Generate forecast
	forecast, err := model.Predict(ctx, request.Horizon, request.Interval)
	if err != nil {
		return nil, fmt.Errorf("failed to generate forecast: %w", err)
	}
	
	forecast.MetricName = request.MetricName
	
	// Cache the forecast
	e.cacheForecasts(request, forecast)
	
	// Update performance metrics
	if e.config.EnablePerformanceTracking {
		e.updatePerformanceMetrics(request.MetricName, forecast)
	}
	
	return forecast, nil
}

// GetMultipleForecasts gets forecasts for multiple metrics
func (e *PredictiveEngine) GetMultipleForecasts(ctx context.Context, requests []ForecastRequest) (map[string]*ForecastResult, error) {
	results := make(map[string]*ForecastResult)
	errors := make(map[string]error)
	
	// Use goroutines for parallel processing
	var wg sync.WaitGroup
	resultChan := make(chan struct {
		metric   string
		forecast *ForecastResult
		err      error
	}, len(requests))
	
	for _, request := range requests {
		wg.Add(1)
		go func(req ForecastRequest) {
			defer wg.Done()
			forecast, err := e.GetForecast(ctx, req)
			resultChan <- struct {
				metric   string
				forecast *ForecastResult
				err      error
			}{req.MetricName, forecast, err}
		}(request)
	}
	
	// Wait for all to complete
	go func() {
		wg.Wait()
		close(resultChan)
	}()
	
	// Collect results
	for result := range resultChan {
		if result.err != nil {
			errors[result.metric] = result.err
		} else {
			results[result.metric] = result.forecast
		}
	}
	
	// Return error if any forecast failed
	if len(errors) > 0 {
		return results, fmt.Errorf("failed to generate forecasts for some metrics: %v", errors)
	}
	
	return results, nil
}

// TrainModel manually triggers training for a specific metric
func (e *PredictiveEngine) TrainModel(metricName string) error {
	return e.trainModel(metricName)
}

// GetModelPerformance gets performance metrics for a model
func (e *PredictiveEngine) GetModelPerformance(metricName string) (*ModelPerformance, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	
	performance, exists := e.modelPerformance[metricName]
	if !exists {
		return nil, fmt.Errorf("no performance data for metric: %s", metricName)
	}
	
	return performance, nil
}

// ListRegisteredMetrics returns a list of all registered metrics
func (e *PredictiveEngine) ListRegisteredMetrics() []string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	
	metrics := make([]string, 0, len(e.models))
	for metric := range e.models {
		metrics = append(metrics, metric)
	}
	
	return metrics
}

// GetModelAccuracy gets the accuracy of a specific model
func (e *PredictiveEngine) GetModelAccuracy(metricName string) (float64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	
	model, exists := e.models[metricName]
	if !exists {
		return 0, fmt.Errorf("no model registered for metric: %s", metricName)
	}
	
	return model.GetAccuracy(), nil
}

// trainModel trains a model with historical data
func (e *PredictiveEngine) trainModel(metricName string) error {
	e.mu.RLock()
	model, exists := e.models[metricName]
	e.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("no model registered for metric: %s", metricName)
	}
	
	// Get historical data
	endTime := time.Now()
	startTime := endTime.Add(-e.config.MaxTrainingPeriod)
	
	// Get metric history from provider
	metricHistory, err := e.metricsProvider.GetMetricHistory(
		context.Background(),
		monitoring.MetricType(metricName), // Convert to appropriate type
		"",                                 // resource ID - empty for aggregate
		startTime,
		endTime,
	)
	if err != nil {
		return fmt.Errorf("failed to get metric history: %w", err)
	}
	
	if len(metricHistory) < model.GetConfig().MinDataPoints {
		return fmt.Errorf("insufficient historical data: got %d points, need at least %d", 
			len(metricHistory), model.GetConfig().MinDataPoints)
	}
	
	// Convert to time series data
	points := make([]TimeSeriesPoint, 0, len(metricHistory))
	for timestamp, value := range metricHistory {
		points = append(points, TimeSeriesPoint{
			Timestamp: timestamp,
			Value:     value,
		})
	}
	
	timeSeriesData := &TimeSeriesData{
		MetricName: metricName,
		Points:     points,
		Tags:       make(map[string]string),
	}
	
	// Train the model
	startTraining := time.Now()
	if err := model.Train(context.Background(), timeSeriesData); err != nil {
		return fmt.Errorf("failed to train model: %w", err)
	}
	
	// Update performance tracking
	if e.config.EnablePerformanceTracking {
		e.mu.Lock()
		if perf, exists := e.modelPerformance[metricName]; exists {
			perf.LastTraining = time.Now()
			perf.TrainingTime = time.Since(startTraining)
			perf.DataPoints = len(points)
			perf.Accuracy = model.GetAccuracy()
		}
		e.mu.Unlock()
	}
	
	return nil
}

// getCachedForecast retrieves a cached forecast if available and not expired
func (e *PredictiveEngine) getCachedForecast(request ForecastRequest) *CachedForecast {
	e.cacheMutex.RLock()
	defer e.cacheMutex.RUnlock()
	
	cacheKey := fmt.Sprintf("%s:%v:%v", request.MetricName, request.Horizon, request.Interval)
	cached, exists := e.forecastCache[cacheKey]
	
	if !exists {
		return nil
	}
	
	// Check if expired
	if time.Now().After(cached.ExpiresAt) {
		return nil
	}
	
	return cached
}

// cacheForecasts stores a forecast in the cache
func (e *PredictiveEngine) cacheForecasts(request ForecastRequest, forecast *ForecastResult) {
	e.cacheMutex.Lock()
	defer e.cacheMutex.Unlock()
	
	cacheKey := fmt.Sprintf("%s:%v:%v", request.MetricName, request.Horizon, request.Interval)
	
	cached := &CachedForecast{
		MetricName: request.MetricName,
		Forecast:   forecast,
		CreatedAt:  time.Now(),
		ExpiresAt:  time.Now().Add(e.config.ForecastCacheTTL),
		Horizon:    request.Horizon,
		Interval:   request.Interval,
	}
	
	e.forecastCache[cacheKey] = cached
	
	// Clean up old cache entries if we exceed the limit
	if len(e.forecastCache) > e.config.MaxCachedForecasts {
		e.cleanupCache()
	}
}

// clearCachedForecastsForMetric clears cached forecasts for a specific metric
func (e *PredictiveEngine) clearCachedForecastsForMetric(metricName string) {
	e.cacheMutex.Lock()
	defer e.cacheMutex.Unlock()
	
	keysToDelete := make([]string, 0)
	for key, cached := range e.forecastCache {
		if cached.MetricName == metricName {
			keysToDelete = append(keysToDelete, key)
		}
	}
	
	for _, key := range keysToDelete {
		delete(e.forecastCache, key)
	}
}

// cleanupCache removes expired and oldest cache entries
func (e *PredictiveEngine) cleanupCache() {
	now := time.Now()
	
	// First, remove expired entries
	for key, cached := range e.forecastCache {
		if now.After(cached.ExpiresAt) {
			delete(e.forecastCache, key)
		}
	}
	
	// If still over limit, remove oldest entries
	if len(e.forecastCache) > e.config.MaxCachedForecasts {
		type cacheEntry struct {
			key     string
			created time.Time
		}
		
		entries := make([]cacheEntry, 0, len(e.forecastCache))
		for key, cached := range e.forecastCache {
			entries = append(entries, cacheEntry{key, cached.CreatedAt})
		}
		
		// Sort by creation time
		for i := 0; i < len(entries)-1; i++ {
			for j := i + 1; j < len(entries); j++ {
				if entries[i].created.After(entries[j].created) {
					entries[i], entries[j] = entries[j], entries[i]
				}
			}
		}
		
		// Remove oldest entries
		removeCount := len(e.forecastCache) - e.config.MaxCachedForecasts
		for i := 0; i < removeCount && i < len(entries); i++ {
			delete(e.forecastCache, entries[i].key)
		}
	}
}

// updatePerformanceMetrics updates performance tracking for a model
func (e *PredictiveEngine) updatePerformanceMetrics(metricName string, forecast *ForecastResult) {
	e.mu.Lock()
	defer e.mu.Unlock()
	
	perf, exists := e.modelPerformance[metricName]
	if !exists {
		return
	}
	
	perf.TotalPredictions++
	perf.LastUpdate = time.Now()
	
	// Add accuracy point
	accuracyPoint := AccuracyPoint{
		Timestamp: time.Now(),
		Accuracy:  forecast.ModelAccuracy,
	}
	
	perf.RecentAccuracy = append(perf.RecentAccuracy, accuracyPoint)
	
	// Keep only recent accuracy points
	maxPoints := 100
	if len(perf.RecentAccuracy) > maxPoints {
		perf.RecentAccuracy = perf.RecentAccuracy[len(perf.RecentAccuracy)-maxPoints:]
	}
	
	// Update current accuracy
	perf.Accuracy = forecast.ModelAccuracy
}

// modelUpdateLoop runs the background model update process
func (e *PredictiveEngine) modelUpdateLoop() {
	for {
		select {
		case <-e.updateTicker.C:
			e.updateAllModels()
		case <-e.ctx.Done():
			return
		}
	}
}

// forecastUpdateLoop runs the background forecast update process  
func (e *PredictiveEngine) forecastUpdateLoop() {
	ticker := time.NewTicker(e.config.ForecastUpdateInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			e.refreshForecasts()
		case <-e.ctx.Done():
			return
		}
	}
}

// performanceTrackingLoop runs the performance monitoring process
func (e *PredictiveEngine) performanceTrackingLoop() {
	ticker := time.NewTicker(5 * time.Minute) // Run every 5 minutes
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			e.updateAllPerformanceMetrics()
		case <-e.ctx.Done():
			return
		}
	}
}

// updateAllModels updates all registered models with new data
func (e *PredictiveEngine) updateAllModels() {
	e.mu.RLock()
	metrics := make([]string, 0, len(e.models))
	for metricName := range e.models {
		metrics = append(metrics, metricName)
	}
	e.mu.RUnlock()
	
	for _, metricName := range metrics {
		go func(metric string) {
			if err := e.updateModel(metric); err != nil {
				fmt.Printf("Warning: failed to update model %s: %v\n", metric, err)
			}
		}(metricName)
	}
}

// updateModel updates a specific model with new data
func (e *PredictiveEngine) updateModel(metricName string) error {
	e.mu.RLock()
	model, exists := e.models[metricName]
	e.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("model not found: %s", metricName)
	}
	
	// Get recent data for model update
	endTime := time.Now()
	startTime := endTime.Add(-1 * time.Hour) // Last hour of data
	
	metricHistory, err := e.metricsProvider.GetMetricHistory(
		context.Background(),
		monitoring.MetricType(metricName),
		"",
		startTime,
		endTime,
	)
	if err != nil {
		return fmt.Errorf("failed to get recent metric history: %w", err)
	}
	
	if len(metricHistory) == 0 {
		return nil // No new data
	}
	
	// Convert to time series points
	points := make([]TimeSeriesPoint, 0, len(metricHistory))
	for timestamp, value := range metricHistory {
		points = append(points, TimeSeriesPoint{
			Timestamp: timestamp,
			Value:     value,
		})
	}
	
	// Update the model
	return model.Update(context.Background(), points)
}

// refreshForecasts refreshes cached forecasts
func (e *PredictiveEngine) refreshForecasts() {
	e.cacheMutex.Lock()
	defer e.cacheMutex.Unlock()
	
	// Clean up expired forecasts
	e.cleanupCache()
	
	// Could implement proactive forecast refresh here
}

// updateAllPerformanceMetrics updates performance metrics for all models
func (e *PredictiveEngine) updateAllPerformanceMetrics() {
	e.mu.RLock()
	metrics := make([]string, 0, len(e.modelPerformance))
	for metricName := range e.modelPerformance {
		metrics = append(metrics, metricName)
	}
	e.mu.RUnlock()
	
	for _, metricName := range metrics {
		go func(metric string) {
			e.calculateAdvancedPerformanceMetrics(metric)
		}(metricName)
	}
}

// calculateAdvancedPerformanceMetrics calculates MAE, RMSE, MAPE
func (e *PredictiveEngine) calculateAdvancedPerformanceMetrics(metricName string) {
	// This would involve comparing recent predictions with actual values
	// For now, just update the timestamp
	e.mu.Lock()
	defer e.mu.Unlock()
	
	if perf, exists := e.modelPerformance[metricName]; exists {
		perf.LastUpdate = time.Now()
		
		// TODO: Implement actual error calculation by comparing predictions with reality
		// This would require storing predictions and comparing with actual metrics
	}
}

// AutoSelectModel automatically selects the best model for a metric based on performance
func (e *PredictiveEngine) AutoSelectModel(metricName string, candidateConfigs []ModelConfig) error {
	if !e.config.AutoModelSelection {
		return fmt.Errorf("auto model selection is disabled")
	}
	
	bestModel := PredictiveModel(nil)
	bestAccuracy := 0.0
	var bestConfig ModelConfig
	
	// Test each candidate model
	for _, config := range candidateConfigs {
		model, err := e.modelFactory.CreateModel(config)
		if err != nil {
			continue
		}
		
		// Train with recent data for evaluation
		if err := e.trainModel(metricName); err != nil {
			continue
		}
		
		accuracy := model.GetAccuracy()
		if accuracy > bestAccuracy && accuracy >= e.config.MinAccuracyThreshold {
			bestModel = model
			bestAccuracy = accuracy
			bestConfig = config
		}
	}
	
	if bestModel == nil {
		return fmt.Errorf("no suitable model found with minimum accuracy %.2f", e.config.MinAccuracyThreshold)
	}
	
	// Replace the existing model
	e.mu.Lock()
	e.models[metricName] = bestModel
	e.mu.Unlock()
	
	// Update performance tracking
	if e.config.EnablePerformanceTracking {
		e.mu.Lock()
		e.modelPerformance[metricName] = &ModelPerformance{
			ModelName:        metricName,
			ModelType:        bestConfig.Type,
			Accuracy:         bestAccuracy,
			LastTraining:     time.Now(),
			LastUpdate:       time.Now(),
			RecentAccuracy:   make([]AccuracyPoint, 0),
			Metadata:         make(map[string]interface{}),
		}
		e.mu.Unlock()
	}
	
	return nil
}