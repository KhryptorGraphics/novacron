// Package novacron provides an enhanced Go SDK for the NovaCron VM management platform
// with multi-cloud federation, AI integration, and advanced reliability features
package novacron

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"sort"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/sony/gobreaker"
)

// CloudProvider represents supported cloud providers
type CloudProvider string

const (
	CloudProviderLocal     CloudProvider = "local"
	CloudProviderAWS       CloudProvider = "aws"
	CloudProviderAzure     CloudProvider = "azure"
	CloudProviderGCP       CloudProvider = "gcp"
	CloudProviderOpenStack CloudProvider = "openstack"
	CloudProviderVMware    CloudProvider = "vmware"
)

// AIFeature represents AI-powered features
type AIFeature string

const (
	AIFeatureIntelligentPlacement AIFeature = "intelligent_placement"
	AIFeaturePredictiveScaling    AIFeature = "predictive_scaling"
	AIFeatureAnomalyDetection     AIFeature = "anomaly_detection"
	AIFeatureCostOptimization     AIFeature = "cost_optimization"
)

// EnhancedClient represents the enhanced NovaCron API client
type EnhancedClient struct {
	baseURL                 string
	httpClient              *http.Client
	apiToken                string
	tokenExpiresAt          time.Time
	userAgent               string
	redis                   *redis.Client
	cacheTTL                time.Duration
	enableAIFeatures        bool
	cloudProvider           CloudProvider
	region                  string
	circuitBreakers         map[string]*gobreaker.CircuitBreaker
	requestMetrics          map[string][]float64
	metricsLock             sync.RWMutex
	enableMetrics           bool
	tokenRefreshTicker      *time.Ticker
	tokenRefreshStop        chan bool
	logger                  Logger
}

// Logger interface for logging
type Logger interface {
	Info(msg string, fields ...interface{})
	Warn(msg string, fields ...interface{})
	Error(msg string, fields ...interface{})
	Debug(msg string, fields ...interface{})
}

// DefaultLogger implements a basic logger
type DefaultLogger struct{}

func (l *DefaultLogger) Info(msg string, fields ...interface{})  { fmt.Printf("INFO: "+msg+"\n", fields...) }
func (l *DefaultLogger) Warn(msg string, fields ...interface{})  { fmt.Printf("WARN: "+msg+"\n", fields...) }
func (l *DefaultLogger) Error(msg string, fields ...interface{}) { fmt.Printf("ERROR: "+msg+"\n", fields...) }
func (l *DefaultLogger) Debug(msg string, fields ...interface{}) { fmt.Printf("DEBUG: "+msg+"\n", fields...) }

// EnhancedClientConfig holds configuration for the enhanced NovaCron client
type EnhancedClientConfig struct {
	BaseURL                   string
	APIToken                  string
	Username                  string
	Password                  string
	Timeout                   time.Duration
	UserAgent                 string
	RedisURL                  string
	CacheTTL                  time.Duration
	EnableAIFeatures          bool
	CloudProvider             CloudProvider
	Region                    string
	CircuitBreakerThreshold   uint32
	CircuitBreakerTimeout     time.Duration
	EnableMetrics             bool
	Logger                    Logger
}

// PlacementRecommendation represents AI-powered placement recommendation
type PlacementRecommendation struct {
	RecommendedNode   string                     `json:"recommended_node"`
	ConfidenceScore   float64                    `json:"confidence_score"`
	Reasoning         string                     `json:"reasoning"`
	AlternativeNodes  []AlternativeNodeOption    `json:"alternative_nodes"`
}

// AlternativeNodeOption represents an alternative node option
type AlternativeNodeOption struct {
	NodeID string   `json:"node_id"`
	Score  float64  `json:"score"`
	Pros   []string `json:"pros"`
	Cons   []string `json:"cons"`
}

// MigrationSpec represents a migration specification for batch operations
type MigrationSpec struct {
	VMID            string `json:"vm_id"`
	TargetNodeID    string `json:"target_node_id"`
	Type            string `json:"type,omitempty"`
	Force           bool   `json:"force,omitempty"`
	BandwidthLimit  *int   `json:"bandwidth_limit,omitempty"`
	Compression     bool   `json:"compression,omitempty"`
}

// RequestMetrics represents performance metrics for API requests
type RequestMetrics struct {
	Count       int     `json:"count"`
	AvgDuration float64 `json:"avg_duration"`
	MinDuration float64 `json:"min_duration"`
	MaxDuration float64 `json:"max_duration"`
	P95Duration float64 `json:"p95_duration"`
}

// CircuitBreakerStatus represents the status of a circuit breaker
type CircuitBreakerStatus struct {
	State       string     `json:"state"`
	Failures    uint32     `json:"failures"`
	LastFailure *time.Time `json:"last_failure,omitempty"`
}

// NewEnhancedClient creates a new enhanced NovaCron client
func NewEnhancedClient(config EnhancedClientConfig) (*EnhancedClient, error) {
	if config.BaseURL == "" {
		return nil, fmt.Errorf("base URL is required")
	}

	// Set defaults
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.UserAgent == "" {
		config.UserAgent = fmt.Sprintf("NovaCron-Go-SDK/2.0.0 (%s)", config.CloudProvider)
	}
	if config.CacheTTL == 0 {
		config.CacheTTL = 5 * time.Minute
	}
	if config.CloudProvider == "" {
		config.CloudProvider = CloudProviderLocal
	}
	if config.CircuitBreakerThreshold == 0 {
		config.CircuitBreakerThreshold = 5
	}
	if config.CircuitBreakerTimeout == 0 {
		config.CircuitBreakerTimeout = 60 * time.Second
	}
	if config.Logger == nil {
		config.Logger = &DefaultLogger{}
	}

	client := &EnhancedClient{
		baseURL:          config.BaseURL,
		apiToken:         config.APIToken,
		userAgent:        config.UserAgent,
		cacheTTL:         config.CacheTTL,
		enableAIFeatures: config.EnableAIFeatures,
		cloudProvider:    config.CloudProvider,
		region:           config.Region,
		circuitBreakers:  make(map[string]*gobreaker.CircuitBreaker),
		requestMetrics:   make(map[string][]float64),
		enableMetrics:    config.EnableMetrics,
		tokenRefreshStop: make(chan bool),
		logger:           config.Logger,
	}

	// Create HTTP client with custom transport
	client.httpClient = &http.Client{
		Timeout: config.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     30 * time.Second,
		},
	}

	// Initialize Redis if URL provided
	if config.RedisURL != "" {
		opt, err := redis.ParseURL(config.RedisURL)
		if err != nil {
			client.logger.Warn("Failed to parse Redis URL: %v", err)
		} else {
			client.redis = redis.NewClient(opt)
			
			// Test connection
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			
			if err := client.redis.Ping(ctx).Err(); err != nil {
				client.logger.Warn("Redis connection failed: %v", err)
				client.redis = nil
			} else {
				client.logger.Info("Redis connected successfully")
			}
		}
	}

	// Start token refresh if API token is provided
	if client.apiToken != "" {
		client.startTokenRefresh()
	}

	return client, nil
}

// startTokenRefresh starts the background token refresh process
func (c *EnhancedClient) startTokenRefresh() {
	c.tokenRefreshTicker = time.NewTicker(1 * time.Minute)
	
	go func() {
		for {
			select {
			case <-c.tokenRefreshTicker.C:
				if !c.tokenExpiresAt.IsZero() {
					refreshAt := c.tokenExpiresAt.Add(-5 * time.Minute) // 5 minutes before expiry
					if time.Now().After(refreshAt) {
						ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
						if newToken, err := c.RefreshToken(ctx); err == nil {
							c.apiToken = newToken
							c.logger.Info("Token refreshed successfully")
						} else {
							c.logger.Error("Token refresh failed: %v", err)
						}
						cancel()
					}
				}
			case <-c.tokenRefreshStop:
				c.tokenRefreshTicker.Stop()
				return
			}
		}
	}()
}

// Close closes the client and cleans up resources
func (c *EnhancedClient) Close() error {
	// Stop token refresh
	if c.tokenRefreshTicker != nil {
		close(c.tokenRefreshStop)
		c.tokenRefreshTicker.Stop()
	}

	// Close Redis connection
	if c.redis != nil {
		return c.redis.Close()
	}

	return nil
}

// getOrCreateCircuitBreaker gets or creates a circuit breaker for an endpoint
func (c *EnhancedClient) getOrCreateCircuitBreaker(endpoint string) *gobreaker.CircuitBreaker {
	if cb, exists := c.circuitBreakers[endpoint]; exists {
		return cb
	}

	settings := gobreaker.Settings{
		Name:        endpoint,
		MaxRequests: 3,
		Interval:    10 * time.Second,
		Timeout:     60 * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures >= 5
		},
		OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
			c.logger.Info("Circuit breaker %s changed from %v to %v", name, from, to)
		},
	}

	cb := gobreaker.NewCircuitBreaker(settings)
	c.circuitBreakers[endpoint] = cb
	return cb
}

// recordMetrics records request metrics
func (c *EnhancedClient) recordMetrics(endpoint string, duration float64) {
	if !c.enableMetrics {
		return
	}

	c.metricsLock.Lock()
	defer c.metricsLock.Unlock()

	if _, exists := c.requestMetrics[endpoint]; !exists {
		c.requestMetrics[endpoint] = make([]float64, 0, 100)
	}

	metrics := c.requestMetrics[endpoint]
	metrics = append(metrics, duration)

	// Keep only last 100 measurements
	if len(metrics) > 100 {
		metrics = metrics[1:]
	}

	c.requestMetrics[endpoint] = metrics
}

// getFromCache retrieves data from Redis cache
func (c *EnhancedClient) getFromCache(ctx context.Context, key string) ([]byte, error) {
	if c.redis == nil {
		return nil, fmt.Errorf("redis not available")
	}

	return c.redis.Get(ctx, key).Bytes()
}

// setCache stores data in Redis cache
func (c *EnhancedClient) setCache(ctx context.Context, key string, data []byte, ttl time.Duration) error {
	if c.redis == nil {
		return fmt.Errorf("redis not available")
	}

	return c.redis.SetEX(ctx, key, data, ttl).Err()
}

// getCacheKey generates a cache key
func (c *EnhancedClient) getCacheKey(method, path string, params interface{}) string {
	key := fmt.Sprintf("novacron:%s:%s:%s", method, path, c.cloudProvider)
	if c.region != "" {
		key += ":" + c.region
	}
	if params != nil {
		if paramBytes, err := json.Marshal(params); err == nil {
			key += ":" + string(paramBytes)
		}
	}
	return key
}

// request performs an HTTP request with circuit breaker, caching, and metrics
func (c *EnhancedClient) request(ctx context.Context, method, path string, body interface{}, result interface{}, useCache bool) error {
	endpoint := fmt.Sprintf("%s:%s", method, path)
	
	// Get circuit breaker for this endpoint
	cb := c.getOrCreateCircuitBreaker(endpoint)

	// Execute request through circuit breaker
	_, err := cb.Execute(func() (interface{}, error) {
		return c.doRequest(ctx, method, path, body, result, useCache)
	})

	return err
}

// doRequest performs the actual HTTP request
func (c *EnhancedClient) doRequest(ctx context.Context, method, path string, body interface{}, result interface{}, useCache bool) (interface{}, error) {
	startTime := time.Now()
	endpoint := fmt.Sprintf("%s:%s", method, path)

	// Check cache for GET requests
	if method == "GET" && useCache {
		cacheKey := c.getCacheKey(method, path, nil)
		if cached, err := c.getFromCache(ctx, cacheKey); err == nil {
			if result != nil {
				if err := json.Unmarshal(cached, result); err == nil {
					c.logger.Debug("Cache hit for %s", cacheKey)
					return result, nil
				}
			}
		}
	}

	// Prepare request body
	var reqBody io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonBody)
	}

	// Create request
	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("X-Cloud-Provider", string(c.cloudProvider))

	if c.region != "" {
		req.Header.Set("X-Cloud-Region", c.region)
	}

	if c.apiToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiToken)
	}

	// Perform request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		duration := time.Since(startTime).Seconds()
		c.recordMetrics(endpoint, duration)
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	duration := time.Since(startTime).Seconds()
	c.recordMetrics(endpoint, duration)

	// Check status code
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	// Parse response
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if result != nil && len(bodyBytes) > 0 {
		if err := json.Unmarshal(bodyBytes, result); err != nil {
			return nil, fmt.Errorf("failed to parse response: %w", err)
		}
	}

	// Cache successful GET responses
	if method == "GET" && useCache && resp.StatusCode == 200 {
		cacheKey := c.getCacheKey(method, path, nil)
		if err := c.setCache(ctx, cacheKey, bodyBytes, c.cacheTTL); err != nil {
			c.logger.Debug("Failed to cache response: %v", err)
		}
	}

	return result, nil
}

// AI-Powered Methods

// GetIntelligentPlacementRecommendation gets AI-powered VM placement recommendations
func (c *EnhancedClient) GetIntelligentPlacementRecommendation(ctx context.Context, vmSpecs map[string]interface{}, constraints map[string]interface{}) (*PlacementRecommendation, error) {
	if !c.enableAIFeatures {
		return nil, fmt.Errorf("AI features not enabled")
	}

	requestData := map[string]interface{}{
		"vm_specs":       vmSpecs,
		"constraints":    constraints,
		"cloud_provider": c.cloudProvider,
		"region":         c.region,
	}

	var recommendation PlacementRecommendation
	if err := c.request(ctx, "POST", "/api/ai/placement", requestData, &recommendation, false); err != nil {
		return nil, fmt.Errorf("failed to get placement recommendation: %w", err)
	}

	return &recommendation, nil
}

// GetPredictiveScalingForecast gets predictive scaling forecast for a VM
func (c *EnhancedClient) GetPredictiveScalingForecast(ctx context.Context, vmID string, forecastHours int) (map[string]interface{}, error) {
	if !c.enableAIFeatures {
		return nil, fmt.Errorf("AI features not enabled")
	}

	path := fmt.Sprintf("/api/ai/scaling/%s?forecast_hours=%d", vmID, forecastHours)
	
	var forecast map[string]interface{}
	if err := c.request(ctx, "GET", path, nil, &forecast, true); err != nil {
		return nil, fmt.Errorf("failed to get scaling forecast: %w", err)
	}

	return forecast, nil
}

// DetectAnomalies detects anomalies in VM or system metrics
func (c *EnhancedClient) DetectAnomalies(ctx context.Context, vmID string, timeWindow int) ([]map[string]interface{}, error) {
	if !c.enableAIFeatures {
		return nil, fmt.Errorf("AI features not enabled")
	}

	path := fmt.Sprintf("/api/ai/anomalies?time_window=%d", timeWindow)
	if vmID != "" {
		path += "&vm_id=" + vmID
	}

	var anomalies []map[string]interface{}
	if err := c.request(ctx, "GET", path, nil, &anomalies, true); err != nil {
		return nil, fmt.Errorf("failed to detect anomalies: %w", err)
	}

	return anomalies, nil
}

// GetCostOptimizationRecommendations gets cost optimization recommendations
func (c *EnhancedClient) GetCostOptimizationRecommendations(ctx context.Context, tenantID string) ([]map[string]interface{}, error) {
	if !c.enableAIFeatures {
		return nil, fmt.Errorf("AI features not enabled")
	}

	path := "/api/ai/cost-optimization"
	if tenantID != "" {
		path += "?tenant_id=" + tenantID
	}

	var recommendations []map[string]interface{}
	if err := c.request(ctx, "GET", path, nil, &recommendations, true); err != nil {
		return nil, fmt.Errorf("failed to get cost recommendations: %w", err)
	}

	return recommendations, nil
}

// Multi-Cloud Federation Methods

// ListFederatedClusters lists all federated clusters across cloud providers
func (c *EnhancedClient) ListFederatedClusters(ctx context.Context) ([]map[string]interface{}, error) {
	var clusters []map[string]interface{}
	if err := c.request(ctx, "GET", "/api/federation/clusters", nil, &clusters, true); err != nil {
		return nil, fmt.Errorf("failed to list federated clusters: %w", err)
	}
	return clusters, nil
}

// CreateCrossCloudMigration creates a cross-cloud migration
func (c *EnhancedClient) CreateCrossCloudMigration(ctx context.Context, vmID, targetCluster string, targetProvider CloudProvider, targetRegion string, options map[string]interface{}) (*Migration, error) {
	requestData := map[string]interface{}{
		"vm_id":           vmID,
		"target_cluster":  targetCluster,
		"target_provider": targetProvider,
		"target_region":   targetRegion,
		"options":         options,
	}

	var migration Migration
	if err := c.request(ctx, "POST", "/api/federation/migrations", requestData, &migration, false); err != nil {
		return nil, fmt.Errorf("failed to create cross-cloud migration: %w", err)
	}

	return &migration, nil
}

// GetCrossCloudCosts gets cost comparison for cross-cloud deployment
func (c *EnhancedClient) GetCrossCloudCosts(ctx context.Context, sourceProvider, targetProvider CloudProvider, vmSpecs map[string]interface{}) (map[string]interface{}, error) {
	requestData := map[string]interface{}{
		"source_provider": sourceProvider,
		"target_provider": targetProvider,
		"vm_specs":        vmSpecs,
	}

	var costs map[string]interface{}
	if err := c.request(ctx, "POST", "/api/federation/cost-comparison", requestData, &costs, true); err != nil {
		return nil, fmt.Errorf("failed to get cross-cloud costs: %w", err)
	}

	return costs, nil
}

// Enhanced VM Management

// CreateVMWithAIPlacement creates a VM with AI-powered intelligent placement
func (c *EnhancedClient) CreateVMWithAIPlacement(ctx context.Context, req *CreateVMRequest, useAIPlacement bool, placementConstraints map[string]interface{}) (*VM, error) {
	requestData := map[string]interface{}{
		"name":         req.Name,
		"command":      req.Command,
		"args":         req.Args,
		"cpu_shares":   req.CPUShares,
		"memory_mb":    req.MemoryMB,
		"disk_size_gb": req.DiskSizeGB,
		"tags":         req.Tags,
		"tenant_id":    req.TenantID,
	}

	if useAIPlacement && c.enableAIFeatures {
		// Get placement recommendation first
		vmSpecs := map[string]interface{}{
			"cpu_shares":   req.CPUShares,
			"memory_mb":    req.MemoryMB,
			"disk_size_gb": req.DiskSizeGB,
		}

		recommendation, err := c.GetIntelligentPlacementRecommendation(ctx, vmSpecs, placementConstraints)
		if err != nil {
			c.logger.Warn("Failed to get AI placement recommendation: %v", err)
		} else {
			requestData["preferred_node"] = recommendation.RecommendedNode
			requestData["placement_reasoning"] = recommendation.Reasoning
		}
	}

	var vm VM
	if err := c.request(ctx, "POST", "/api/vms", requestData, &vm, false); err != nil {
		return nil, fmt.Errorf("failed to create VM: %w", err)
	}

	return &vm, nil
}

// Batch Operations

// BatchCreateVMs creates multiple VMs in batch with controlled concurrency
func (c *EnhancedClient) BatchCreateVMs(ctx context.Context, requests []*CreateVMRequest, concurrency int, useAIPlacement bool) ([]interface{}, error) {
	semaphore := make(chan struct{}, concurrency)
	results := make([]interface{}, len(requests))
	var wg sync.WaitGroup

	for i, req := range requests {
		wg.Add(1)
		go func(index int, request *CreateVMRequest) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release

			vm, err := c.CreateVMWithAIPlacement(ctx, request, useAIPlacement, nil)
			if err != nil {
				results[index] = err
			} else {
				results[index] = vm
			}
		}(i, req)
	}

	wg.Wait()
	return results, nil
}

// BatchMigrateVMs migrates multiple VMs in batch
func (c *EnhancedClient) BatchMigrateVMs(ctx context.Context, migrations []MigrationSpec, concurrency int) ([]interface{}, error) {
	semaphore := make(chan struct{}, concurrency)
	results := make([]interface{}, len(migrations))
	var wg sync.WaitGroup

	for i, spec := range migrations {
		wg.Add(1)
		go func(index int, migrationSpec MigrationSpec) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire
			defer func() { <-semaphore }() // Release

			req := &MigrationRequest{
				TargetNodeID:   migrationSpec.TargetNodeID,
				Type:           migrationSpec.Type,
				Force:          migrationSpec.Force,
				BandwidthLimit: migrationSpec.BandwidthLimit,
				Compression:    migrationSpec.Compression,
			}

			migration, err := c.MigrateVM(ctx, migrationSpec.VMID, req)
			if err != nil {
				results[index] = err
			} else {
				results[index] = migration
			}
		}(i, spec)
	}

	wg.Wait()
	return results, nil
}

// Performance Monitoring

// GetRequestMetrics returns performance metrics for API requests
func (c *EnhancedClient) GetRequestMetrics() map[string]*RequestMetrics {
	if !c.enableMetrics {
		return nil
	}

	c.metricsLock.RLock()
	defer c.metricsLock.RUnlock()

	result := make(map[string]*RequestMetrics)

	for endpoint, timings := range c.requestMetrics {
		if len(timings) > 0 {
			sort.Float64s(timings)
			
			var sum float64
			for _, t := range timings {
				sum += t
			}

			metrics := &RequestMetrics{
				Count:       len(timings),
				AvgDuration: sum / float64(len(timings)),
				MinDuration: timings[0],
				MaxDuration: timings[len(timings)-1],
				P95Duration: timings[int(math.Ceil(float64(len(timings))*0.95))-1],
			}

			result[endpoint] = metrics
		}
	}

	return result
}

// GetCircuitBreakerStatus returns the status of all circuit breakers
func (c *EnhancedClient) GetCircuitBreakerStatus() map[string]*CircuitBreakerStatus {
	result := make(map[string]*CircuitBreakerStatus)

	for endpoint, cb := range c.circuitBreakers {
		counts := cb.Counts()
		status := &CircuitBreakerStatus{
			State:    cb.State().String(),
			Failures: counts.ConsecutiveFailures,
		}

		result[endpoint] = status
	}

	return result
}

// Enhanced Authentication

// Authenticate performs authentication and returns JWT token with expiration tracking
func (c *EnhancedClient) Authenticate(ctx context.Context, username, password string) (string, error) {
	authReq := map[string]string{
		"username": username,
		"password": password,
	}

	var authResp map[string]interface{}
	if err := c.request(ctx, "POST", "/api/auth/login", authReq, &authResp, false); err != nil {
		return "", fmt.Errorf("authentication failed: %w", err)
	}

	token, ok := authResp["token"].(string)
	if !ok {
		return "", fmt.Errorf("invalid token in response")
	}

	// Set token expiration if provided
	if expiresIn, ok := authResp["expires_in"].(float64); ok {
		c.tokenExpiresAt = time.Now().Add(time.Duration(expiresIn) * time.Second)
	}

	c.apiToken = token
	return token, nil
}

// RefreshToken refreshes the JWT token with expiration tracking
func (c *EnhancedClient) RefreshToken(ctx context.Context) (string, error) {
	var authResp map[string]interface{}
	if err := c.request(ctx, "POST", "/api/auth/refresh", nil, &authResp, false); err != nil {
		return "", fmt.Errorf("token refresh failed: %w", err)
	}

	token, ok := authResp["token"].(string)
	if !ok {
		return "", fmt.Errorf("invalid token in response")
	}

	// Set token expiration if provided
	if expiresIn, ok := authResp["expires_in"].(float64); ok {
		c.tokenExpiresAt = time.Now().Add(time.Duration(expiresIn) * time.Second)
	}

	c.apiToken = token
	return token, nil
}

// MigrateVM migrates a VM to another node (from original client)
func (c *EnhancedClient) MigrateVM(ctx context.Context, vmID string, req *MigrationRequest) (*Migration, error) {
	var migration Migration
	if err := c.request(ctx, "POST", fmt.Sprintf("/api/vms/%s/migrate", vmID), req, &migration, false); err != nil {
		return nil, fmt.Errorf("failed to migrate VM: %w", err)
	}
	return &migration, nil
}