package network

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// PerformancePredictor interface for network performance prediction
type PerformancePredictor interface {
	PredictBandwidth(ctx context.Context, request PredictionRequest) (*BandwidthPrediction, error)
	StoreNetworkMetrics(ctx context.Context, metrics NetworkMetrics) error
	StoreWorkloadCharacteristics(ctx context.Context, workload WorkloadCharacteristics) error
	GetModelPerformance(ctx context.Context) (*ModelPerformance, error)
	IsHealthy(ctx context.Context) bool
}

// PredictionRequest represents a bandwidth prediction request
type PredictionRequest struct {
	SourceNode          string                  `json:"source_node"`
	TargetNode          string                  `json:"target_node"`
	WorkloadChars       WorkloadCharacteristics `json:"workload"`
	TimeHorizonHours    int                     `json:"time_horizon_hours"`
	ConfidenceLevel     float64                 `json:"confidence_level"`
	IncludeUncertainty  bool                    `json:"include_uncertainty"`
}

// WorkloadCharacteristics represents VM workload characteristics
type WorkloadCharacteristics struct {
	VMID                    string  `json:"vm_id"`
	WorkloadType           string  `json:"workload_type"` // interactive, batch, streaming, compute
	CPUCores               int     `json:"cpu_cores"`
	MemoryGB               float64 `json:"memory_gb"`
	StorageGB              float64 `json:"storage_gb"`
	NetworkIntensive       bool    `json:"network_intensive"`
	ExpectedConnections    int     `json:"expected_connections"`
	DataTransferPattern    string  `json:"data_transfer_pattern"` // burst, steady, periodic
	PeakHours             []int   `json:"peak_hours"`
	HistoricalBandwidth   float64 `json:"historical_bandwidth"`
}

// NetworkMetrics represents network performance measurements
type NetworkMetrics struct {
	Timestamp         time.Time `json:"timestamp"`
	SourceNode        string    `json:"source_node"`
	TargetNode        string    `json:"target_node"`
	BandwidthMbps     float64   `json:"bandwidth_mbps"`
	LatencyMs         float64   `json:"latency_ms"`
	PacketLoss        float64   `json:"packet_loss"`
	JitterMs          float64   `json:"jitter_ms"`
	ThroughputMbps    float64   `json:"throughput_mbps"`
	ConnectionQuality float64   `json:"connection_quality"`
	RouteHops         int       `json:"route_hops"`
	CongestionLevel   float64   `json:"congestion_level"`
}

// BandwidthPrediction represents the result of bandwidth prediction
type BandwidthPrediction struct {
	PredictedBandwidth   float64                   `json:"predicted_bandwidth"`
	ConfidenceInterval   []float64                 `json:"confidence_interval"`
	PredictionConfidence float64                   `json:"prediction_confidence"`
	OptimalTimeWindow    []time.Time               `json:"optimal_time_window"`
	AlternativeRoutes    []AlternativeRoute        `json:"alternative_routes"`
	CongestionForecast   map[string]float64        `json:"congestion_forecast"`
	Recommendation       string                    `json:"recommendation"`
}

// AlternativeRoute represents an alternative network route
type AlternativeRoute struct {
	RouteID           string    `json:"route_id"`
	IntermediateNodes []string  `json:"intermediate_nodes"`
	PredictedBandwidth float64  `json:"predicted_bandwidth"`
	EstimatedLatency   float64  `json:"estimated_latency"`
	ReliabilityScore   float64  `json:"reliability_score"`
}

// ModelPerformance represents ML model performance metrics
type ModelPerformance struct {
	Models map[string]ModelMetrics `json:"models"`
}

// ModelMetrics represents performance metrics for a specific model
type ModelMetrics struct {
	MAE             float64 `json:"mae"`
	MSE             float64 `json:"mse"`
	R2Score         float64 `json:"r2_score"`
	TrainingSamples int     `json:"training_samples"`
}

// AIPerformancePredictor implements PerformancePredictor using AI/ML backend
type AIPerformancePredictor struct {
	baseURL      string
	httpClient   *http.Client
	logger       *log.Logger
	cache        map[string]*cachedPrediction
	cacheMutex   sync.RWMutex
	cacheTTL     time.Duration
	retryCount   int
	retryDelay   time.Duration
}

// cachedPrediction represents a cached prediction with expiration
type cachedPrediction struct {
	prediction *BandwidthPrediction
	expiresAt  time.Time
}

// NewAIPerformancePredictor creates a new AI-powered performance predictor
func NewAIPerformancePredictor(baseURL string, logger *log.Logger) *AIPerformancePredictor {
	return &AIPerformancePredictor{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		logger:     logger,
		cache:      make(map[string]*cachedPrediction),
		cacheTTL:   5 * time.Minute,
		retryCount: 3,
		retryDelay: time.Second,
	}
}

// PredictBandwidth predicts bandwidth for a network route and workload
func (a *AIPerformancePredictor) PredictBandwidth(ctx context.Context, request PredictionRequest) (*BandwidthPrediction, error) {
	// Check cache first
	cacheKey := a.generateCacheKey(request)
	if cached := a.getCachedPrediction(cacheKey); cached != nil {
		a.logger.Printf("Returning cached bandwidth prediction for %s -> %s", request.SourceNode, request.TargetNode)
		return cached, nil
	}

	// Prepare HTTP request
	requestBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal prediction request: %w", err)
	}

	var prediction *BandwidthPrediction
	var lastErr error

	// Retry mechanism
	for attempt := 0; attempt < a.retryCount; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(a.retryDelay * time.Duration(attempt)):
			}
		}

		req, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/predict", bytes.NewReader(requestBody))
		if err != nil {
			lastErr = fmt.Errorf("failed to create HTTP request: %w", err)
			continue
		}

		req.Header.Set("Content-Type", "application/json")

		resp, err := a.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("HTTP request failed (attempt %d/%d): %w", attempt+1, a.retryCount, err)
			a.logger.Printf("Prediction request failed: %v", lastErr)
			continue
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()

		if err != nil {
			lastErr = fmt.Errorf("failed to read response body: %w", err)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("prediction API returned status %d: %s", resp.StatusCode, string(body))
			continue
		}

		var response struct {
			Success    bool                `json:"success"`
			Prediction *BandwidthPrediction `json:"prediction"`
			Error      string              `json:"error"`
		}

		if err := json.Unmarshal(body, &response); err != nil {
			lastErr = fmt.Errorf("failed to unmarshal response: %w", err)
			continue
		}

		if !response.Success {
			lastErr = fmt.Errorf("prediction failed: %s", response.Error)
			continue
		}

		prediction = response.Prediction
		break
	}

	if prediction == nil {
		return nil, fmt.Errorf("all prediction attempts failed, last error: %w", lastErr)
	}

	// Cache the result
	a.cachePrediction(cacheKey, prediction)

	a.logger.Printf("Successfully predicted bandwidth %.2f Mbps for %s -> %s (confidence: %.3f)",
		prediction.PredictedBandwidth, request.SourceNode, request.TargetNode, prediction.PredictionConfidence)

	return prediction, nil
}

// StoreNetworkMetrics stores network performance metrics
func (a *AIPerformancePredictor) StoreNetworkMetrics(ctx context.Context, metrics NetworkMetrics) error {
	requestBody, err := json.Marshal(metrics)
	if err != nil {
		return fmt.Errorf("failed to marshal network metrics: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/metrics", bytes.NewReader(requestBody))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to store network metrics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("metrics storage failed with status %d: %s", resp.StatusCode, string(body))
	}

	a.logger.Printf("Successfully stored network metrics for %s -> %s", metrics.SourceNode, metrics.TargetNode)
	return nil
}

// StoreWorkloadCharacteristics stores VM workload characteristics
func (a *AIPerformancePredictor) StoreWorkloadCharacteristics(ctx context.Context, workload WorkloadCharacteristics) error {
	requestBody, err := json.Marshal(workload)
	if err != nil {
		return fmt.Errorf("failed to marshal workload characteristics: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", a.baseURL+"/workload", bytes.NewReader(requestBody))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to store workload characteristics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("workload storage failed with status %d: %s", resp.StatusCode, string(body))
	}

	a.logger.Printf("Successfully stored workload characteristics for VM %s", workload.VMID)
	return nil
}

// GetModelPerformance retrieves ML model performance metrics
func (a *AIPerformancePredictor) GetModelPerformance(ctx context.Context) (*ModelPerformance, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", a.baseURL+"/performance", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get model performance: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("performance API returned status %d: %s", resp.StatusCode, string(body))
	}

	var performance ModelPerformance
	if err := json.Unmarshal(body, &performance); err != nil {
		return nil, fmt.Errorf("failed to unmarshal performance metrics: %w", err)
	}

	return &performance, nil
}

// IsHealthy checks if the AI prediction service is healthy
func (a *AIPerformancePredictor) IsHealthy(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, "GET", a.baseURL+"/health", nil)
	if err != nil {
		a.logger.Printf("Health check failed to create request: %v", err)
		return false
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		a.logger.Printf("Health check failed: %v", err)
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

// generateCacheKey generates a cache key for a prediction request
func (a *AIPerformancePredictor) generateCacheKey(request PredictionRequest) string {
	return fmt.Sprintf("%s_%s_%s_%d", request.SourceNode, request.TargetNode, 
		request.WorkloadChars.VMID, request.TimeHorizonHours)
}

// getCachedPrediction retrieves a cached prediction if available and not expired
func (a *AIPerformancePredictor) getCachedPrediction(key string) *BandwidthPrediction {
	a.cacheMutex.RLock()
	defer a.cacheMutex.RUnlock()

	cached, exists := a.cache[key]
	if !exists || time.Now().After(cached.expiresAt) {
		return nil
	}

	return cached.prediction
}

// cachePrediction stores a prediction in the cache
func (a *AIPerformancePredictor) cachePrediction(key string, prediction *BandwidthPrediction) {
	a.cacheMutex.Lock()
	defer a.cacheMutex.Unlock()

	a.cache[key] = &cachedPrediction{
		prediction: prediction,
		expiresAt:  time.Now().Add(a.cacheTTL),
	}

	// Clean up expired entries
	now := time.Now()
	for k, v := range a.cache {
		if now.After(v.expiresAt) {
			delete(a.cache, k)
		}
	}
}

// InvalidateCache invalidates all cached predictions
func (a *AIPerformancePredictor) InvalidateCache() {
	a.cacheMutex.Lock()
	defer a.cacheMutex.Unlock()
	
	a.cache = make(map[string]*cachedPrediction)
	a.logger.Println("All prediction cache entries invalidated")
}

// InvalidateCacheEntry invalidates a specific cache entry
func (a *AIPerformancePredictor) InvalidateCacheEntry(key string) {
	a.cacheMutex.Lock()
	defer a.cacheMutex.Unlock()
	
	if _, exists := a.cache[key]; exists {
		delete(a.cache, key)
		a.logger.Printf("Cache entry invalidated: %s", key)
	}
}

// InvalidateCacheByRoute invalidates cache entries for a specific route
func (a *AIPerformancePredictor) InvalidateCacheByRoute(sourceNode, targetNode string) {
	a.cacheMutex.Lock()
	defer a.cacheMutex.Unlock()
	
	deletedCount := 0
	for key := range a.cache {
		// Check if the cache key contains the source and target nodes
		if containsRoute(key, sourceNode, targetNode) {
			delete(a.cache, key)
			deletedCount++
		}
	}
	
	if deletedCount > 0 {
		a.logger.Printf("Invalidated %d cache entries for route %s -> %s", deletedCount, sourceNode, targetNode)
	}
}

// InvalidateCacheByVM invalidates cache entries for a specific VM
func (a *AIPerformancePredictor) InvalidateCacheByVM(vmID string) {
	a.cacheMutex.Lock()
	defer a.cacheMutex.Unlock()
	
	deletedCount := 0
	for key := range a.cache {
		// Check if the cache key contains the VM ID
		if containsVM(key, vmID) {
			delete(a.cache, key)
			deletedCount++
		}
	}
	
	if deletedCount > 0 {
		a.logger.Printf("Invalidated %d cache entries for VM %s", deletedCount, vmID)
	}
}

// CleanExpiredCache manually cleans up expired cache entries
func (a *AIPerformancePredictor) CleanExpiredCache() {
	a.cacheMutex.Lock()
	defer a.cacheMutex.Unlock()
	
	now := time.Now()
	deletedCount := 0
	
	for key, cached := range a.cache {
		if now.After(cached.expiresAt) {
			delete(a.cache, key)
			deletedCount++
		}
	}
	
	if deletedCount > 0 {
		a.logger.Printf("Cleaned up %d expired cache entries", deletedCount)
	}
}

// GetCacheStats returns cache statistics
func (a *AIPerformancePredictor) GetCacheStats() map[string]interface{} {
	a.cacheMutex.RLock()
	defer a.cacheMutex.RUnlock()
	
	now := time.Now()
	totalEntries := len(a.cache)
	expiredEntries := 0
	
	for _, cached := range a.cache {
		if now.After(cached.expiresAt) {
			expiredEntries++
		}
	}
	
	return map[string]interface{}{
		"total_entries":   totalEntries,
		"expired_entries": expiredEntries,
		"active_entries":  totalEntries - expiredEntries,
		"cache_ttl":       a.cacheTTL.String(),
	}
}

// containsRoute checks if a cache key contains the specified route
func containsRoute(key, sourceNode, targetNode string) bool {
	return (fmt.Sprintf("%s_%s", sourceNode, targetNode) == key[:len(sourceNode)+len(targetNode)+1]) ||
		   (fmt.Sprintf("%s_%s", targetNode, sourceNode) == key[:len(targetNode)+len(sourceNode)+1])
}

// containsVM checks if a cache key contains the specified VM ID
func containsVM(key, vmID string) bool {
	// Cache key format: sourceNode_targetNode_vmID_timeHorizon
	// Check if vmID appears as a component in the underscore-separated key
	parts := strings.Split(key, "_")
	for _, part := range parts {
		if part == vmID {
			return true
		}
	}
	return false
}

// HeuristicPerformancePredictor implements PerformancePredictor using heuristic rules
type HeuristicPerformancePredictor struct {
	logger           *log.Logger
	baselineLatency  float64
	baselineBandwidth float64
	congestionFactor float64
}

// NewHeuristicPerformancePredictor creates a heuristic-based performance predictor
func NewHeuristicPerformancePredictor(logger *log.Logger) *HeuristicPerformancePredictor {
	return &HeuristicPerformancePredictor{
		logger:           logger,
		baselineLatency:  10.0,  // 10ms baseline latency
		baselineBandwidth: 100.0, // 100 Mbps baseline bandwidth
		congestionFactor: 0.3,   // 30% congestion factor
	}
}

// PredictBandwidth provides heuristic-based bandwidth prediction
func (h *HeuristicPerformancePredictor) PredictBandwidth(ctx context.Context, request PredictionRequest) (*BandwidthPrediction, error) {
	// Simple heuristic calculation
	baseBandwidth := h.baselineBandwidth

	// Adjust based on workload characteristics
	if request.WorkloadChars.NetworkIntensive {
		baseBandwidth *= 1.5
	}

	switch request.WorkloadChars.WorkloadType {
	case "streaming":
		baseBandwidth *= 1.2
	case "batch":
		baseBandwidth *= 0.8
	case "interactive":
		baseBandwidth *= 1.1
	case "compute":
		baseBandwidth *= 0.9
	}

	// Apply time-based congestion
	now := time.Now()
	hour := now.Hour()
	
	// Peak hours: 8-10 AM, 4-6 PM
	if (hour >= 8 && hour <= 10) || (hour >= 16 && hour <= 18) {
		baseBandwidth *= (1.0 - h.congestionFactor)
	}

	// Weekend factor
	if now.Weekday() == time.Saturday || now.Weekday() == time.Sunday {
		baseBandwidth *= 1.1 // Less congestion on weekends
	}

	// Calculate confidence interval
	variance := baseBandwidth * 0.2 // 20% variance
	confidenceInterval := []float64{
		baseBandwidth - variance,
		baseBandwidth + variance,
	}

	// Generate congestion forecast
	congestionForecast := make(map[string]float64)
	for i := 1; i <= request.TimeHorizonHours; i++ {
		futureHour := (hour + i) % 24
		congestion := 0.3 // Base congestion

		if (futureHour >= 8 && futureHour <= 10) || (futureHour >= 16 && futureHour <= 18) {
			congestion = 0.7 // Peak congestion
		} else if futureHour >= 0 && futureHour <= 6 {
			congestion = 0.1 // Low congestion
		}

		congestionForecast[fmt.Sprintf("hour_%d", i)] = congestion
	}

	// Generate recommendation
	var recommendation string
	if baseBandwidth < 50 {
		recommendation = "LOW_BANDWIDTH: Consider alternative scheduling"
	} else if baseBandwidth > 200 {
		recommendation = "HIGH_BANDWIDTH: Optimal conditions"
	} else {
		recommendation = "MODERATE_BANDWIDTH: Acceptable conditions"
	}

	return &BandwidthPrediction{
		PredictedBandwidth:   baseBandwidth,
		ConfidenceInterval:   confidenceInterval,
		PredictionConfidence: 0.7, // Moderate confidence for heuristic approach
		OptimalTimeWindow:    []time.Time{now.Add(2 * time.Hour), now.Add(3 * time.Hour)},
		AlternativeRoutes:    []AlternativeRoute{},
		CongestionForecast:   congestionForecast,
		Recommendation:       recommendation,
	}, nil
}

// StoreNetworkMetrics is a no-op for heuristic predictor
func (h *HeuristicPerformancePredictor) StoreNetworkMetrics(ctx context.Context, metrics NetworkMetrics) error {
	// Heuristic predictor doesn't store metrics
	return nil
}

// StoreWorkloadCharacteristics is a no-op for heuristic predictor
func (h *HeuristicPerformancePredictor) StoreWorkloadCharacteristics(ctx context.Context, workload WorkloadCharacteristics) error {
	// Heuristic predictor doesn't store workload characteristics
	return nil
}

// GetModelPerformance returns dummy performance metrics for heuristic predictor
func (h *HeuristicPerformancePredictor) GetModelPerformance(ctx context.Context) (*ModelPerformance, error) {
	return &ModelPerformance{
		Models: map[string]ModelMetrics{
			"heuristic": {
				MAE:             15.0,
				MSE:             350.0,
				R2Score:         0.6,
				TrainingSamples: 0,
			},
		},
	}, nil
}

// IsHealthy always returns true for heuristic predictor
func (h *HeuristicPerformancePredictor) IsHealthy(ctx context.Context) bool {
	return true
}

// PredictorFactory creates performance predictors based on configuration
type PredictorFactory struct {
	logger *log.Logger
}

// NewPredictorFactory creates a new predictor factory
func NewPredictorFactory(logger *log.Logger) *PredictorFactory {
	return &PredictorFactory{logger: logger}
}

// CreatePredictor creates a performance predictor based on type
func (f *PredictorFactory) CreatePredictor(predictorType string, config map[string]interface{}) (PerformancePredictor, error) {
	switch predictorType {
	case "ai":
		baseURL, ok := config["base_url"].(string)
		if !ok {
			return nil, fmt.Errorf("ai predictor requires base_url configuration")
		}
		return NewAIPerformancePredictor(baseURL, f.logger), nil

	case "heuristic":
		return NewHeuristicPerformancePredictor(f.logger), nil

	default:
		return nil, fmt.Errorf("unknown predictor type: %s", predictorType)
	}
}

// PerformancePredictorManager manages multiple predictors with fallback
type PerformancePredictorManager struct {
	primary   PerformancePredictor
	fallback  PerformancePredictor
	logger    *log.Logger
	mutex     sync.RWMutex
	healthTTL time.Duration
	lastCheck time.Time
	isHealthy bool
}

// NewPerformancePredictorManager creates a new predictor manager
func NewPerformancePredictorManager(primary, fallback PerformancePredictor, logger *log.Logger) *PerformancePredictorManager {
	return &PerformancePredictorManager{
		primary:   primary,
		fallback:  fallback,
		logger:    logger,
		healthTTL: 60 * time.Second,
		isHealthy: true,
	}
}

// PredictBandwidth predicts bandwidth with fallback support
func (m *PerformancePredictorManager) PredictBandwidth(ctx context.Context, request PredictionRequest) (*BandwidthPrediction, error) {
	// Check primary predictor health
	if !m.isPrimaryHealthy(ctx) {
		m.logger.Printf("Primary predictor unhealthy, using fallback")
		return m.fallback.PredictBandwidth(ctx, request)
	}

	// Try primary predictor
	prediction, err := m.primary.PredictBandwidth(ctx, request)
	if err != nil {
		m.logger.Printf("Primary predictor failed, using fallback: %v", err)
		return m.fallback.PredictBandwidth(ctx, request)
	}

	return prediction, nil
}

// isPrimaryHealthy checks if primary predictor is healthy with caching
func (m *PerformancePredictorManager) isPrimaryHealthy(ctx context.Context) bool {
	m.mutex.RLock()
	if time.Since(m.lastCheck) < m.healthTTL {
		healthy := m.isHealthy
		m.mutex.RUnlock()
		return healthy
	}
	m.mutex.RUnlock()

	// Perform health check
	m.mutex.Lock()
	defer m.mutex.Unlock()

	m.isHealthy = m.primary.IsHealthy(ctx)
	m.lastCheck = time.Now()

	return m.isHealthy
}

// Delegate other methods to primary predictor
func (m *PerformancePredictorManager) StoreNetworkMetrics(ctx context.Context, metrics NetworkMetrics) error {
	return m.primary.StoreNetworkMetrics(ctx, metrics)
}

func (m *PerformancePredictorManager) StoreWorkloadCharacteristics(ctx context.Context, workload WorkloadCharacteristics) error {
	return m.primary.StoreWorkloadCharacteristics(ctx, workload)
}

func (m *PerformancePredictorManager) GetModelPerformance(ctx context.Context) (*ModelPerformance, error) {
	return m.primary.GetModelPerformance(ctx)
}

func (m *PerformancePredictorManager) IsHealthy(ctx context.Context) bool {
	return m.isPrimaryHealthy(ctx) || m.fallback.IsHealthy(ctx)
}