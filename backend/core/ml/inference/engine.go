package inference

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// InferenceEngine provides high-performance model inference
type InferenceEngine struct {
	config     *InferenceConfig
	models     map[string]*LoadedModel
	cache      *PredictionCache
	balancer   *LoadBalancer
	mu         sync.RWMutex
	metrics    *InferenceMetrics
}

// InferenceConfig defines inference configuration
type InferenceConfig struct {
	MaxBatchSize    int
	BatchTimeout    time.Duration
	CacheEnabled    bool
	CacheTTL        time.Duration
	GPUEnabled      bool
	NumWorkers      int
	LatencyTarget   time.Duration // Target: <10ms
}

// LoadedModel represents a model loaded for inference
type LoadedModel struct {
	Name      string
	Version   string
	Weights   interface{}
	Framework string
	LoadedAt  time.Time
	WarmupDone bool
}

// PredictionCache caches predictions
type PredictionCache struct {
	cache map[string]*CachedPrediction
	ttl   time.Duration
	mu    sync.RWMutex
}

// CachedPrediction represents a cached prediction
type CachedPrediction struct {
	Input      string
	Output     []float64
	CachedAt   time.Time
}

// LoadBalancer balances inference requests
type LoadBalancer struct {
	workers []*InferenceWorker
	current int
	mu      sync.Mutex
}

// InferenceWorker processes inference requests
type InferenceWorker struct {
	ID       int
	Model    *LoadedModel
	Queue    chan *InferenceRequest
	Active   bool
}

// InferenceRequest represents an inference request
type InferenceRequest struct {
	Input    []float64
	Metadata map[string]string
	Response chan *InferenceResponse
}

// InferenceResponse represents an inference response
type InferenceResponse struct {
	Prediction []float64
	Latency    time.Duration
	FromCache  bool
	Error      error
}

// InferenceMetrics tracks inference performance
type InferenceMetrics struct {
	TotalRequests   int64
	CacheHits       int64
	CacheMisses     int64
	AvgLatency      float64
	P50Latency      float64
	P95Latency      float64
	P99Latency      float64
	Throughput      float64
	ErrorCount      int64
	LastUpdated     time.Time
	LatencyHist     []float64
	mu              sync.RWMutex
}

// NewInferenceEngine creates a new inference engine
func NewInferenceEngine(config *InferenceConfig) *InferenceEngine {
	if config == nil {
		config = DefaultInferenceConfig()
	}

	engine := &InferenceEngine{
		config:  config,
		models:  make(map[string]*LoadedModel),
		metrics: &InferenceMetrics{
			LatencyHist: make([]float64, 0, 10000),
		},
	}

	if config.CacheEnabled {
		engine.cache = NewPredictionCache(config.CacheTTL)
	}

	engine.balancer = NewLoadBalancer(config.NumWorkers)

	return engine
}

// DefaultInferenceConfig returns default inference configuration
func DefaultInferenceConfig() *InferenceConfig {
	return &InferenceConfig{
		MaxBatchSize:  32,
		BatchTimeout:  10 * time.Millisecond,
		CacheEnabled:  true,
		CacheTTL:      5 * time.Minute,
		GPUEnabled:    false,
		NumWorkers:    4,
		LatencyTarget: 10 * time.Millisecond,
	}
}

// LoadModel loads a model for inference
func (e *InferenceEngine) LoadModel(name, version string, weights interface{}, framework string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	modelKey := fmt.Sprintf("%s:%s", name, version)

	loadedModel := &LoadedModel{
		Name:      name,
		Version:   version,
		Weights:   weights,
		Framework: framework,
		LoadedAt:  time.Now(),
	}

	// Warmup model
	if err := e.warmupModel(loadedModel); err != nil {
		return fmt.Errorf("model warmup failed: %w", err)
	}

	loadedModel.WarmupDone = true
	e.models[modelKey] = loadedModel

	// Initialize workers with model
	e.balancer.InitializeWorkers(loadedModel)

	return nil
}

// Predict performs inference
func (e *InferenceEngine) Predict(ctx context.Context, modelName, version string, input []float64) (*InferenceResponse, error) {
	startTime := time.Now()

	modelKey := fmt.Sprintf("%s:%s", modelName, version)

	// Check cache
	if e.config.CacheEnabled {
		if cached := e.cache.Get(modelKey, input); cached != nil {
			e.recordMetrics(time.Since(startTime), true, nil)
			return &InferenceResponse{
				Prediction: cached.Output,
				Latency:    time.Since(startTime),
				FromCache:  true,
			}, nil
		}
	}

	// Get model
	e.mu.RLock()
	model, exists := e.models[modelKey]
	e.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("model %s:%s not loaded", modelName, version)
	}

	// Submit to worker
	request := &InferenceRequest{
		Input:    input,
		Response: make(chan *InferenceResponse, 1),
	}

	worker := e.balancer.GetNextWorker()

	select {
	case worker.Queue <- request:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Wait for response
	select {
	case response := <-request.Response:
		if response.Error == nil && e.config.CacheEnabled {
			e.cache.Put(modelKey, input, response.Prediction)
		}
		e.recordMetrics(time.Since(startTime), false, response.Error)
		return response, response.Error
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// BatchPredict performs batch inference
func (e *InferenceEngine) BatchPredict(ctx context.Context, modelName, version string, inputs [][]float64) ([][]float64, error) {
	results := make([][]float64, len(inputs))
	errors := make([]error, len(inputs))

	var wg sync.WaitGroup
	for i, input := range inputs {
		wg.Add(1)
		go func(idx int, inp []float64) {
			defer wg.Done()
			response, err := e.Predict(ctx, modelName, version, inp)
			if err != nil {
				errors[idx] = err
			} else {
				results[idx] = response.Prediction
			}
		}(i, input)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// warmupModel warms up model with dummy inference
func (e *InferenceEngine) warmupModel(model *LoadedModel) error {
	// Perform dummy inferences to warm up model
	dummyInput := make([]float64, 10)
	for i := range dummyInput {
		dummyInput[i] = 0.5
	}

	for i := 0; i < 10; i++ {
		_, err := e.performInference(model, dummyInput)
		if err != nil {
			return err
		}
	}

	return nil
}

// performInference performs actual inference
func (e *InferenceEngine) performInference(model *LoadedModel, input []float64) ([]float64, error) {
	// Simplified inference - in practice, use framework-specific inference
	weights, ok := model.Weights.([][]float64)
	if !ok {
		return nil, fmt.Errorf("invalid weights format")
	}

	output := make([]float64, len(weights))
	for i := range weights {
		sum := 0.0
		for j := range weights[i] {
			if j < len(input) {
				sum += weights[i][j] * input[j]
			}
		}
		output[i] = 1.0 / (1.0 + Math.Exp(-sum)) // Sigmoid
	}

	return output, nil
}

// recordMetrics records inference metrics
func (e *InferenceEngine) recordMetrics(latency time.Duration, fromCache bool, err error) {
	e.metrics.mu.Lock()
	defer e.metrics.mu.Unlock()

	e.metrics.TotalRequests++

	if fromCache {
		e.metrics.CacheHits++
	} else {
		e.metrics.CacheMisses++
	}

	if err != nil {
		e.metrics.ErrorCount++
		return
	}

	latencyMs := float64(latency.Microseconds()) / 1000.0
	e.metrics.LatencyHist = append(e.metrics.LatencyHist, latencyMs)

	// Calculate percentiles
	if len(e.metrics.LatencyHist) > 100 {
		e.updatePercentiles()
	}

	e.metrics.LastUpdated = time.Now()
}

// updatePercentiles updates latency percentiles
func (e *InferenceEngine) updatePercentiles() {
	hist := make([]float64, len(e.metrics.LatencyHist))
	copy(hist, e.metrics.LatencyHist)

	// Sort
	for i := 0; i < len(hist)-1; i++ {
		for j := i + 1; j < len(hist); j++ {
			if hist[j] < hist[i] {
				hist[i], hist[j] = hist[j], hist[i]
			}
		}
	}

	n := len(hist)
	e.metrics.P50Latency = hist[n*50/100]
	e.metrics.P95Latency = hist[n*95/100]
	e.metrics.P99Latency = hist[n*99/100]

	sum := 0.0
	for _, l := range hist {
		sum += l
	}
	e.metrics.AvgLatency = sum / float64(n)
}

// GetMetrics returns current metrics
func (e *InferenceEngine) GetMetrics() *InferenceMetrics {
	e.metrics.mu.RLock()
	defer e.metrics.mu.RUnlock()

	metrics := &InferenceMetrics{}
	*metrics = *e.metrics
	return metrics
}

// NewPredictionCache creates a new prediction cache
func NewPredictionCache(ttl time.Duration) *PredictionCache {
	return &PredictionCache{
		cache: make(map[string]*CachedPrediction),
		ttl:   ttl,
	}
}

// Get retrieves cached prediction
func (c *PredictionCache) Get(modelKey string, input []float64) *CachedPrediction {
	c.mu.RLock()
	defer c.mu.RUnlock()

	key := c.generateKey(modelKey, input)
	cached, exists := c.cache[key]
	if !exists {
		return nil
	}

	// Check TTL
	if time.Since(cached.CachedAt) > c.ttl {
		return nil
	}

	return cached
}

// Put stores prediction in cache
func (c *PredictionCache) Put(modelKey string, input []float64, output []float64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := c.generateKey(modelKey, input)
	c.cache[key] = &CachedPrediction{
		Input:    key,
		Output:   output,
		CachedAt: time.Now(),
	}
}

// generateKey generates cache key
func (c *PredictionCache) generateKey(modelKey string, input []float64) string {
	return fmt.Sprintf("%s:%v", modelKey, input)
}

// NewLoadBalancer creates a new load balancer
func NewLoadBalancer(numWorkers int) *LoadBalancer {
	return &LoadBalancer{
		workers: make([]*InferenceWorker, numWorkers),
	}
}

// InitializeWorkers initializes inference workers
func (lb *LoadBalancer) InitializeWorkers(model *LoadedModel) {
	for i := range lb.workers {
		worker := &InferenceWorker{
			ID:     i,
			Model:  model,
			Queue:  make(chan *InferenceRequest, 100),
			Active: true,
		}
		lb.workers[i] = worker
		go worker.Start()
	}
}

// GetNextWorker returns next available worker (round-robin)
func (lb *LoadBalancer) GetNextWorker() *InferenceWorker {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	worker := lb.workers[lb.current]
	lb.current = (lb.current + 1) % len(lb.workers)
	return worker
}

// Start starts inference worker
func (w *InferenceWorker) Start() {
	for request := range w.Queue {
		startTime := time.Now()

		// Perform inference
		weights, ok := w.Model.Weights.([][]float64)
		if !ok {
			request.Response <- &InferenceResponse{
				Error: fmt.Errorf("invalid weights"),
			}
			continue
		}

		output := make([]float64, len(weights))
		for i := range weights {
			sum := 0.0
			for j := range weights[i] {
				if j < len(request.Input) {
					sum += weights[i][j] * request.Input[j]
				}
			}
			output[i] = 1.0 / (1.0 + Math.Exp(-sum))
		}

		request.Response <- &InferenceResponse{
			Prediction: output,
			Latency:    time.Since(startTime),
			FromCache:  false,
		}
	}
}

// Math helpers
type Math struct{}

func (Math) Exp(x float64) float64 {
	// Simplified exp function
	return 1.0 + x + x*x/2.0 + x*x*x/6.0
}
