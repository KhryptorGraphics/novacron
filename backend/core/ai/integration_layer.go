package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// AIIntegrationLayer provides a bridge between Go services and Python AI engine
type AIIntegrationLayer struct {
	client         *http.Client
	endpoint       string
	timeout        time.Duration
	retries        int

	// Connection pooling
	maxConnections int
	activeRequests atomic.Int32

	// Metrics
	metrics        *AIMetrics

	// Circuit breaker
	circuitBreaker *CircuitBreaker

	// Cache for frequent requests
	cache          *ResponseCache

	// Authentication
	apiKey         string

	// Synchronization
	mu             sync.RWMutex
}

// AIMetrics tracks AI integration metrics
type AIMetrics struct {
	TotalRequests      atomic.Int64
	SuccessfulRequests atomic.Int64
	FailedRequests     atomic.Int64
	AverageResponseTime atomic.Int64 // milliseconds
	CircuitBreakerTrips atomic.Int64
	CacheHits          atomic.Int64
	CacheMisses        atomic.Int64
}

// CircuitBreaker implements circuit breaker pattern for AI service calls
type CircuitBreaker struct {
	state           CircuitState
	failureCount    atomic.Int32
	successCount    atomic.Int32
	threshold       int32
	timeout         time.Duration
	lastFailureTime time.Time
	mu              sync.RWMutex
}

// CircuitState represents the state of the circuit breaker
type CircuitState int

const (
	CircuitClosed CircuitState = iota
	CircuitOpen
	CircuitHalfOpen
)

// ResponseCache provides caching for AI responses
type ResponseCache struct {
	cache       map[string]CacheEntry
	maxSize     int
	defaultTTL  time.Duration
	mu          sync.RWMutex
}

// CacheEntry represents a cached response
type CacheEntry struct {
	Data      interface{}
	ExpiresAt time.Time
	Hits      int
}

// AIRequest represents a generic AI service request
type AIRequest struct {
	ID       string                 `json:"id"`
	Service  string                 `json:"service"`
	Method   string                 `json:"method"`
	Data     map[string]interface{} `json:"data"`
	Metadata map[string]string      `json:"metadata,omitempty"`
}

// AIResponse represents a generic AI service response
type AIResponse struct {
	ID           string                 `json:"id"`
	Success      bool                   `json:"success"`
	Data         map[string]interface{} `json:"data,omitempty"`
	Error        string                 `json:"error,omitempty"`
	Confidence   float64                `json:"confidence,omitempty"`
	ProcessTime  float64                `json:"process_time,omitempty"`
	ModelVersion string                 `json:"model_version,omitempty"`
}

// ResourcePredictionRequest represents resource prediction request
type ResourcePredictionRequest struct {
	NodeID        string                 `json:"node_id"`
	ResourceType  string                 `json:"resource_type"`
	HorizonMinutes int                   `json:"horizon_minutes"`
	HistoricalData []ResourceDataPoint   `json:"historical_data"`
	Context       map[string]interface{} `json:"context,omitempty"`
}

// ResourceDataPoint represents a single resource measurement
type ResourceDataPoint struct {
	Timestamp time.Time `json:"timestamp"`
	Value     float64   `json:"value"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// ResourcePredictionResponse represents resource prediction response
type ResourcePredictionResponse struct {
	Predictions []float64 `json:"predictions"`
	Confidence  float64   `json:"confidence"`
	ModelInfo   ModelInfo `json:"model_info"`
}

// ModelInfo provides information about the model used
type ModelInfo struct {
	Name         string    `json:"name"`
	Version      string    `json:"version"`
	TrainingData string    `json:"training_data"`
	Accuracy     float64   `json:"accuracy"`
	LastTrained  time.Time `json:"last_trained"`
}

// PerformanceOptimizationRequest represents performance optimization request
type PerformanceOptimizationRequest struct {
	ClusterID    string                 `json:"cluster_id"`
	ClusterData  map[string]interface{} `json:"cluster_data"`
	Goals        []string               `json:"goals"`
	Constraints  map[string]interface{} `json:"constraints,omitempty"`
}

// PerformanceOptimizationResponse represents performance optimization response
type PerformanceOptimizationResponse struct {
	Recommendations []OptimizationRecommendation `json:"recommendations"`
	ExpectedGains   map[string]float64           `json:"expected_gains"`
	RiskAssessment  RiskAssessment               `json:"risk_assessment"`
	Confidence      float64                      `json:"confidence"`
}

// OptimizationRecommendation represents a single optimization recommendation
type OptimizationRecommendation struct {
	Type        string                 `json:"type"`
	Target      string                 `json:"target"`
	Action      string                 `json:"action"`
	Parameters  map[string]interface{} `json:"parameters"`
	Priority    int                    `json:"priority"`
	Impact      string                 `json:"impact"`
	Confidence  float64                `json:"confidence"`
}

// RiskAssessment represents risk assessment for optimization
type RiskAssessment struct {
	OverallRisk   float64            `json:"overall_risk"`
	RiskFactors   []string           `json:"risk_factors"`
	Mitigations   []string           `json:"mitigations"`
	RiskBreakdown map[string]float64 `json:"risk_breakdown"`
}

// AnomalyDetectionRequest represents anomaly detection request
type AnomalyDetectionRequest struct {
	ResourceID     string                 `json:"resource_id"`
	MetricType     string                 `json:"metric_type"`
	DataPoints     []ResourceDataPoint    `json:"data_points"`
	Sensitivity    float64                `json:"sensitivity"`
	Context        map[string]interface{} `json:"context,omitempty"`
}

// AnomalyDetectionResponse represents anomaly detection response
type AnomalyDetectionResponse struct {
	Anomalies       []AnomalyAlert    `json:"anomalies"`
	OverallScore    float64           `json:"overall_score"`
	Baseline        map[string]float64 `json:"baseline"`
	ModelInfo       ModelInfo         `json:"model_info"`
}

// AnomalyAlert represents a detected anomaly
type AnomalyAlert struct {
	Timestamp     time.Time              `json:"timestamp"`
	AnomalyType   string                 `json:"anomaly_type"`
	Severity      string                 `json:"severity"`
	Score         float64                `json:"score"`
	Description   string                 `json:"description"`
	AffectedMetrics []string             `json:"affected_metrics"`
	Recommendations []string             `json:"recommendations"`
	Context       map[string]interface{} `json:"context,omitempty"`
}

// WorkloadPatternRequest represents workload pattern analysis request
type WorkloadPatternRequest struct {
	WorkloadID     string              `json:"workload_id"`
	TimeRange      TimeRange           `json:"time_range"`
	MetricTypes    []string            `json:"metric_types"`
	DataPoints     []ResourceDataPoint `json:"data_points"`
}

// TimeRange represents a time range for analysis
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// WorkloadPatternResponse represents workload pattern analysis response
type WorkloadPatternResponse struct {
	Patterns        []WorkloadPattern `json:"patterns"`
	Classification  string            `json:"classification"`
	Seasonality     SeasonalityInfo   `json:"seasonality"`
	Recommendations []string          `json:"recommendations"`
	Confidence      float64           `json:"confidence"`
}

// WorkloadPattern represents a detected workload pattern
type WorkloadPattern struct {
	Type        string    `json:"type"`
	StartTime   time.Time `json:"start_time"`
	EndTime     time.Time `json:"end_time"`
	Intensity   float64   `json:"intensity"`
	Frequency   string    `json:"frequency"`
	Confidence  float64   `json:"confidence"`
	Description string    `json:"description"`
}

// SeasonalityInfo represents seasonality information
type SeasonalityInfo struct {
	HasSeasonality  bool              `json:"has_seasonality"`
	Period          time.Duration     `json:"period"`
	Strength        float64           `json:"strength"`
	Components      []string          `json:"components"`
	PeakTimes       []time.Time       `json:"peak_times"`
	LowTimes        []time.Time       `json:"low_times"`
}

// NewAIIntegrationLayer creates a new AI integration layer
func NewAIIntegrationLayer(endpoint, apiKey string, config AIConfig) *AIIntegrationLayer {
	client := &http.Client{
		Timeout: config.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:        config.MaxConnections,
			MaxIdleConnsPerHost: config.MaxConnections / 4,
			IdleConnTimeout:     30 * time.Second,
		},
	}

	circuitBreaker := &CircuitBreaker{
		state:     CircuitClosed,
		threshold: int32(config.CircuitBreakerThreshold),
		timeout:   config.CircuitBreakerTimeout,
	}

	cache := &ResponseCache{
		cache:      make(map[string]CacheEntry),
		maxSize:    config.CacheSize,
		defaultTTL: config.CacheTTL,
	}

	return &AIIntegrationLayer{
		client:         client,
		endpoint:       endpoint,
		timeout:        config.Timeout,
		retries:        config.Retries,
		maxConnections: config.MaxConnections,
		metrics:        &AIMetrics{},
		circuitBreaker: circuitBreaker,
		cache:          cache,
		apiKey:         apiKey,
	}
}

// AIConfig contains configuration for AI integration
type AIConfig struct {
	Timeout                  time.Duration
	Retries                  int
	MaxConnections          int
	CircuitBreakerThreshold int
	CircuitBreakerTimeout   time.Duration
	CacheSize               int
	CacheTTL                time.Duration
}

// DefaultAIConfig returns default AI configuration
func DefaultAIConfig() AIConfig {
	return AIConfig{
		Timeout:                  30 * time.Second,
		Retries:                  3,
		MaxConnections:          50,
		CircuitBreakerThreshold: 5,
		CircuitBreakerTimeout:   60 * time.Second,
		CacheSize:               1000,
		CacheTTL:                5 * time.Minute,
	}
}

// PredictResourceDemand predicts resource demand for a node
func (ai *AIIntegrationLayer) PredictResourceDemand(ctx context.Context, req ResourcePredictionRequest) (*ResourcePredictionResponse, error) {
	cacheKey := fmt.Sprintf("predict_resource_%s_%s_%d", req.NodeID, req.ResourceType, req.HorizonMinutes)

	// Check cache first
	if cached := ai.cache.Get(cacheKey); cached != nil {
		ai.metrics.CacheHits.Add(1)
		return cached.(*ResourcePredictionResponse), nil
	}
	ai.metrics.CacheMisses.Add(1)

	aiReq := AIRequest{
		ID:      uuid.New().String(),
		Service: "resource_prediction",
		Method:  "predict_demand",
		Data:    structToMap(req),
	}

	resp, err := ai.makeRequest(ctx, aiReq)
	if err != nil {
		return nil, err
	}

	var predResp ResourcePredictionResponse
	if err := mapToStruct(resp.Data, &predResp); err != nil {
		return nil, fmt.Errorf("failed to parse prediction response: %w", err)
	}

	// Cache the response
	ai.cache.Set(cacheKey, &predResp, ai.cache.defaultTTL)

	return &predResp, nil
}

// OptimizePerformance optimizes cluster performance
func (ai *AIIntegrationLayer) OptimizePerformance(ctx context.Context, req PerformanceOptimizationRequest) (*PerformanceOptimizationResponse, error) {
	aiReq := AIRequest{
		ID:      uuid.New().String(),
		Service: "performance_optimization",
		Method:  "optimize_cluster",
		Data:    structToMap(req),
	}

	resp, err := ai.makeRequest(ctx, aiReq)
	if err != nil {
		return nil, err
	}

	var optResp PerformanceOptimizationResponse
	if err := mapToStruct(resp.Data, &optResp); err != nil {
		return nil, fmt.Errorf("failed to parse optimization response: %w", err)
	}

	return &optResp, nil
}

// DetectAnomalies detects anomalies in resource metrics
func (ai *AIIntegrationLayer) DetectAnomalies(ctx context.Context, req AnomalyDetectionRequest) (*AnomalyDetectionResponse, error) {
	aiReq := AIRequest{
		ID:      uuid.New().String(),
		Service: "anomaly_detection",
		Method:  "detect",
		Data:    structToMap(req),
	}

	resp, err := ai.makeRequest(ctx, aiReq)
	if err != nil {
		return nil, err
	}

	var anomResp AnomalyDetectionResponse
	if err := mapToStruct(resp.Data, &anomResp); err != nil {
		return nil, fmt.Errorf("failed to parse anomaly response: %w", err)
	}

	return &anomResp, nil
}

// AnalyzeWorkloadPattern analyzes workload patterns
func (ai *AIIntegrationLayer) AnalyzeWorkloadPattern(ctx context.Context, req WorkloadPatternRequest) (*WorkloadPatternResponse, error) {
	aiReq := AIRequest{
		ID:      uuid.New().String(),
		Service: "workload_pattern_recognition",
		Method:  "analyze_patterns",
		Data:    structToMap(req),
	}

	resp, err := ai.makeRequest(ctx, aiReq)
	if err != nil {
		return nil, err
	}

	var patternResp WorkloadPatternResponse
	if err := mapToStruct(resp.Data, &patternResp); err != nil {
		return nil, fmt.Errorf("failed to parse pattern response: %w", err)
	}

	return &patternResp, nil
}

// PredictScalingNeeds predicts scaling requirements
func (ai *AIIntegrationLayer) PredictScalingNeeds(ctx context.Context, clusterData map[string]interface{}) (map[string]interface{}, error) {
	aiReq := AIRequest{
		ID:      uuid.New().String(),
		Service: "predictive_scaling",
		Method:  "predict_scaling",
		Data:    clusterData,
	}

	resp, err := ai.makeRequest(ctx, aiReq)
	if err != nil {
		return nil, err
	}

	return resp.Data, nil
}

// TrainModel triggers model training with new data
func (ai *AIIntegrationLayer) TrainModel(ctx context.Context, modelType string, trainingData interface{}) error {
	aiReq := AIRequest{
		ID:      uuid.New().String(),
		Service: "model_training",
		Method:  "train",
		Data: map[string]interface{}{
			"model_type":     modelType,
			"training_data":  trainingData,
		},
	}

	_, err := ai.makeRequest(ctx, aiReq)
	return err
}

// GetModelInfo retrieves information about AI models
func (ai *AIIntegrationLayer) GetModelInfo(ctx context.Context, modelType string) (*ModelInfo, error) {
	aiReq := AIRequest{
		ID:      uuid.New().String(),
		Service: "model_management",
		Method:  "get_info",
		Data: map[string]interface{}{
			"model_type": modelType,
		},
	}

	resp, err := ai.makeRequest(ctx, aiReq)
	if err != nil {
		return nil, err
	}

	var modelInfo ModelInfo
	if err := mapToStruct(resp.Data, &modelInfo); err != nil {
		return nil, fmt.Errorf("failed to parse model info: %w", err)
	}

	return &modelInfo, nil
}

// HealthCheck performs health check on AI service
func (ai *AIIntegrationLayer) HealthCheck(ctx context.Context) error {
	aiReq := AIRequest{
		ID:      uuid.New().String(),
		Service: "health",
		Method:  "check",
		Data:    map[string]interface{}{},
	}

	_, err := ai.makeRequest(ctx, aiReq)
	return err
}

// makeRequest makes a request to the AI service with circuit breaker and retry logic
func (ai *AIIntegrationLayer) makeRequest(ctx context.Context, req AIRequest) (*AIResponse, error) {
	start := time.Now()
	ai.metrics.TotalRequests.Add(1)

	// Check circuit breaker
	if !ai.circuitBreaker.CanExecute() {
		ai.metrics.CircuitBreakerTrips.Add(1)
		return nil, fmt.Errorf("circuit breaker is open")
	}

	// Rate limiting
	if ai.activeRequests.Load() >= int32(ai.maxConnections) {
		return nil, fmt.Errorf("too many active requests")
	}

	ai.activeRequests.Add(1)
	defer ai.activeRequests.Add(-1)

	var lastErr error
	for attempt := 0; attempt < ai.retries; attempt++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Marshal request
		jsonData, err := json.Marshal(req)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}

		// Create HTTP request
		httpReq, err := http.NewRequestWithContext(ctx, "POST", ai.endpoint+"/api/v1/process", bytes.NewBuffer(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create HTTP request: %w", err)
		}

		// Set headers
		httpReq.Header.Set("Content-Type", "application/json")
		if ai.apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+ai.apiKey)
		}

		// Make request
		resp, err := ai.client.Do(httpReq)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed (attempt %d): %w", attempt+1, err)
			ai.circuitBreaker.RecordFailure()
			if attempt < ai.retries-1 {
				time.Sleep(time.Duration(attempt+1) * time.Second)
				continue
			}
			break
		}

		// Read response
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()

		if err != nil {
			lastErr = fmt.Errorf("failed to read response body: %w", err)
			ai.circuitBreaker.RecordFailure()
			if attempt < ai.retries-1 {
				time.Sleep(time.Duration(attempt+1) * time.Second)
				continue
			}
			break
		}

		// Check status code
		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("AI service returned status %d: %s", resp.StatusCode, string(body))
			ai.circuitBreaker.RecordFailure()
			if attempt < ai.retries-1 {
				time.Sleep(time.Duration(attempt+1) * time.Second)
				continue
			}
			break
		}

		// Parse response
		var aiResp AIResponse
		if err := json.Unmarshal(body, &aiResp); err != nil {
			lastErr = fmt.Errorf("failed to parse AI response: %w", err)
			ai.circuitBreaker.RecordFailure()
			if attempt < ai.retries-1 {
				time.Sleep(time.Duration(attempt+1) * time.Second)
				continue
			}
			break
		}

		// Check if AI processing was successful
		if !aiResp.Success {
			lastErr = fmt.Errorf("AI processing failed: %s", aiResp.Error)
			ai.circuitBreaker.RecordFailure()
			if attempt < ai.retries-1 {
				time.Sleep(time.Duration(attempt+1) * time.Second)
				continue
			}
			break
		}

		// Success
		ai.circuitBreaker.RecordSuccess()
		ai.metrics.SuccessfulRequests.Add(1)

		duration := time.Since(start).Milliseconds()
		ai.metrics.AverageResponseTime.Store(duration)

		return &aiResp, nil
	}

	ai.metrics.FailedRequests.Add(1)
	return nil, lastErr
}

// Circuit breaker methods
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	switch cb.state {
	case CircuitClosed:
		return true
	case CircuitOpen:
		if time.Since(cb.lastFailureTime) > cb.timeout {
			cb.state = CircuitHalfOpen
			return true
		}
		return false
	case CircuitHalfOpen:
		return true
	default:
		return false
	}
}

func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.successCount.Add(1)
	cb.failureCount.Store(0)

	if cb.state == CircuitHalfOpen {
		cb.state = CircuitClosed
	}
}

func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.failureCount.Add(1)
	cb.lastFailureTime = time.Now()

	if cb.failureCount.Load() >= cb.threshold {
		cb.state = CircuitOpen
	}
}

// Cache methods
func (c *ResponseCache) Get(key string) interface{} {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.cache[key]
	if !exists || time.Now().After(entry.ExpiresAt) {
		return nil
	}

	// Update hit count
	entry.Hits++
	c.cache[key] = entry

	return entry.Data
}

func (c *ResponseCache) Set(key string, data interface{}, ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Evict expired entries if cache is full
	if len(c.cache) >= c.maxSize {
		c.evictExpired()
		if len(c.cache) >= c.maxSize {
			c.evictLRU()
		}
	}

	c.cache[key] = CacheEntry{
		Data:      data,
		ExpiresAt: time.Now().Add(ttl),
		Hits:      0,
	}
}

func (c *ResponseCache) evictExpired() {
	now := time.Now()
	for key, entry := range c.cache {
		if now.After(entry.ExpiresAt) {
			delete(c.cache, key)
		}
	}
}

func (c *ResponseCache) evictLRU() {
	minHits := int(^uint(0) >> 1) // Max int
	var lruKey string

	for key, entry := range c.cache {
		if entry.Hits < minHits {
			minHits = entry.Hits
			lruKey = key
		}
	}

	if lruKey != "" {
		delete(c.cache, lruKey)
	}
}

// GetMetrics returns AI integration metrics
func (ai *AIIntegrationLayer) GetMetrics() map[string]interface{} {
	total := ai.metrics.TotalRequests.Load()
	successful := ai.metrics.SuccessfulRequests.Load()
	failed := ai.metrics.FailedRequests.Load()

	successRate := float64(0)
	if total > 0 {
		successRate = float64(successful) / float64(total)
	}

	return map[string]interface{}{
		"total_requests":          total,
		"successful_requests":     successful,
		"failed_requests":         failed,
		"success_rate":           successRate,
		"avg_response_time_ms":   ai.metrics.AverageResponseTime.Load(),
		"circuit_breaker_trips":  ai.metrics.CircuitBreakerTrips.Load(),
		"cache_hits":             ai.metrics.CacheHits.Load(),
		"cache_misses":           ai.metrics.CacheMisses.Load(),
		"active_requests":        ai.activeRequests.Load(),
		"circuit_breaker_state":  ai.getCircuitBreakerState(),
	}
}

func (ai *AIIntegrationLayer) getCircuitBreakerState() string {
	ai.circuitBreaker.mu.RLock()
	defer ai.circuitBreaker.mu.RUnlock()

	switch ai.circuitBreaker.state {
	case CircuitClosed:
		return "closed"
	case CircuitOpen:
		return "open"
	case CircuitHalfOpen:
		return "half_open"
	default:
		return "unknown"
	}
}

// Helper functions for struct/map conversion
func structToMap(data interface{}) map[string]interface{} {
	jsonBytes, _ := json.Marshal(data)
	var result map[string]interface{}
	json.Unmarshal(jsonBytes, &result)
	return result
}

func mapToStruct(data map[string]interface{}, dest interface{}) error {
	jsonBytes, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return json.Unmarshal(jsonBytes, dest)
}

// Close gracefully shuts down the AI integration layer
func (ai *AIIntegrationLayer) Close() error {
	// Wait for active requests to complete or timeout
	timeout := time.After(30 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			return fmt.Errorf("timeout waiting for active requests to complete")
		case <-ticker.C:
			if ai.activeRequests.Load() == 0 {
				ai.client.CloseIdleConnections()
				return nil
			}
		}
	}
}