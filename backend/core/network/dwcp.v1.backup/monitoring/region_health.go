package monitoring

import (
	"sync"
	"time"
)

// RegionHealth represents regional health status
type RegionHealth struct {
	mu sync.RWMutex

	region         string
	healthScore    float64 // 0-100
	components     map[string]*ComponentHealth
	dependencies   []string
	lastUpdate     time.Time

	// Health factors
	availability   float64
	resourceUtil   float64
	networkLatency float64
	errorRate      float64
	consensusHealth float64

	// Status
	isDegraded     bool
	isHealthy      bool
	trend          HealthTrend
}

// ComponentHealth represents health of a component
type ComponentHealth struct {
	Name       string
	Status     HealthStatus
	Score      float64
	LastCheck  time.Time
	Message    string
}

// HealthStatus represents component status
type HealthStatus int

const (
	HealthStatusHealthy HealthStatus = iota
	HealthStatusDegraded
	HealthStatusUnhealthy
	HealthStatusUnknown
)

// HealthTrend represents health trend
type HealthTrend int

const (
	TrendImproving HealthTrend = iota
	TrendStable
	TrendDegrading
)

// HealthMonitor monitors regional health
type HealthMonitor struct {
	mu      sync.RWMutex
	regions map[string]*RegionHealth

	// Thresholds
	healthyThreshold   float64
	degradedThreshold  float64

	// Weights for health factors
	weights HealthWeights
}

// HealthWeights defines weights for health calculation
type HealthWeights struct {
	Availability   float64
	ResourceUtil   float64
	NetworkLatency float64
	ErrorRate      float64
	Consensus      float64
}

// NewHealthMonitor creates a new health monitor
func NewHealthMonitor() *HealthMonitor {
	return &HealthMonitor{
		regions:            make(map[string]*RegionHealth),
		healthyThreshold:   80.0,
		degradedThreshold:  60.0,
		weights: HealthWeights{
			Availability:   0.3,
			ResourceUtil:   0.2,
			NetworkLatency: 0.2,
			ErrorRate:      0.2,
			Consensus:      0.1,
		},
	}
}

// UpdateRegionHealth updates health metrics for a region
func (hm *HealthMonitor) UpdateRegionHealth(region string, metrics *HealthMetrics) {
	hm.mu.Lock()
	defer hm.mu.Unlock()

	health, ok := hm.regions[region]
	if !ok {
		health = &RegionHealth{
			region:     region,
			components: make(map[string]*ComponentHealth),
		}
		hm.regions[region] = health
	}

	health.mu.Lock()
	defer health.mu.Unlock()

	// Update factors
	health.availability = metrics.Availability
	health.resourceUtil = metrics.ResourceUtilization
	health.networkLatency = metrics.NetworkLatency
	health.errorRate = metrics.ErrorRate
	health.consensusHealth = metrics.ConsensusHealth

	// Calculate overall health score
	oldScore := health.healthScore
	health.healthScore = hm.calculateHealthScore(health)

	// Determine trend
	if health.healthScore > oldScore+5 {
		health.trend = TrendImproving
	} else if health.healthScore < oldScore-5 {
		health.trend = TrendDegrading
	} else {
		health.trend = TrendStable
	}

	// Update status
	health.isHealthy = health.healthScore >= hm.healthyThreshold
	health.isDegraded = health.healthScore < hm.healthyThreshold && health.healthScore >= hm.degradedThreshold

	health.lastUpdate = time.Now()
}

// HealthMetrics contains health metrics
type HealthMetrics struct {
	Availability        float64 // 0-100
	ResourceUtilization float64 // 0-100
	NetworkLatency      float64 // ms (converted to score)
	ErrorRate           float64 // 0-1 (converted to score)
	ConsensusHealth     float64 // 0-100
}

// calculateHealthScore calculates weighted health score
func (hm *HealthMonitor) calculateHealthScore(health *RegionHealth) float64 {
	w := hm.weights

	// Convert latency to score (lower is better)
	latencyScore := 100.0
	if health.networkLatency > 0 {
		latencyScore = max(0, 100-health.networkLatency)
	}

	// Convert error rate to score (lower is better)
	errorScore := max(0, 100-(health.errorRate*100))

	score := (w.Availability * health.availability) +
		(w.ResourceUtil * (100 - health.resourceUtil)) + // Lower utilization is better
		(w.NetworkLatency * latencyScore) +
		(w.ErrorRate * errorScore) +
		(w.Consensus * health.consensusHealth)

	return min(100, max(0, score))
}

// UpdateComponentHealth updates health of a specific component
func (hm *HealthMonitor) UpdateComponentHealth(region, component string, status HealthStatus, score float64, message string) {
	hm.mu.Lock()
	health, ok := hm.regions[region]
	if !ok {
		health = &RegionHealth{
			region:     region,
			components: make(map[string]*ComponentHealth),
		}
		hm.regions[region] = health
	}
	hm.mu.Unlock()

	health.mu.Lock()
	defer health.mu.Unlock()

	health.components[component] = &ComponentHealth{
		Name:      component,
		Status:    status,
		Score:     score,
		LastCheck: time.Now(),
		Message:   message,
	}
}

// GetRegionHealth retrieves health for a region
func (hm *HealthMonitor) GetRegionHealth(region string) (*HealthScore, bool) {
	hm.mu.RLock()
	health, ok := hm.regions[region]
	hm.mu.RUnlock()

	if !ok {
		return nil, false
	}

	health.mu.RLock()
	defer health.mu.RUnlock()

	components := make(map[string]ComponentStatus)
	for name, comp := range health.components {
		components[name] = ComponentStatus{
			Status:  comp.Status,
			Score:   comp.Score,
			Message: comp.Message,
		}
	}

	return &HealthScore{
		Region:      region,
		Score:       health.healthScore,
		IsHealthy:   health.isHealthy,
		IsDegraded:  health.isDegraded,
		Trend:       health.trend,
		Components:  components,
		LastUpdate:  health.lastUpdate,
	}, true
}

// HealthScore represents regional health score
type HealthScore struct {
	Region     string
	Score      float64
	IsHealthy  bool
	IsDegraded bool
	Trend      HealthTrend
	Components map[string]ComponentStatus
	LastUpdate time.Time
}

// ComponentStatus represents component status
type ComponentStatus struct {
	Status  HealthStatus
	Score   float64
	Message string
}

// GetAllRegionsHealth returns health for all regions
func (hm *HealthMonitor) GetAllRegionsHealth() map[string]*HealthScore {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	result := make(map[string]*HealthScore)
	for region := range hm.regions {
		if score, ok := hm.GetRegionHealth(region); ok {
			result[region] = score
		}
	}

	return result
}

// GetDegradedRegions returns list of degraded regions
func (hm *HealthMonitor) GetDegradedRegions() []string {
	hm.mu.RLock()
	defer hm.mu.RUnlock()

	var degraded []string
	for region, health := range hm.regions {
		health.mu.RLock()
		if health.isDegraded || !health.isHealthy {
			degraded = append(degraded, region)
		}
		health.mu.RUnlock()
	}

	return degraded
}

// AddDependency adds a dependency between regions
func (hm *HealthMonitor) AddDependency(region, dependsOn string) {
	hm.mu.Lock()
	health, ok := hm.regions[region]
	if !ok {
		health = &RegionHealth{
			region:     region,
			components: make(map[string]*ComponentHealth),
		}
		hm.regions[region] = health
	}
	hm.mu.Unlock()

	health.mu.Lock()
	defer health.mu.Unlock()

	health.dependencies = append(health.dependencies, dependsOn)
}

// GetDependencies returns dependencies for a region
func (hm *HealthMonitor) GetDependencies(region string) []string {
	hm.mu.RLock()
	health, ok := hm.regions[region]
	hm.mu.RUnlock()

	if !ok {
		return nil
	}

	health.mu.RLock()
	defer health.mu.RUnlock()

	deps := make([]string, len(health.dependencies))
	copy(deps, health.dependencies)
	return deps
}

// AnalyzeTrend analyzes health trend over time
func (hm *HealthMonitor) AnalyzeTrend(region string, window time.Duration) HealthTrend {
	// Simplified trend analysis
	// Production would maintain time series data
	health, ok := hm.GetRegionHealth(region)
	if !ok {
		return TrendStable
	}

	return health.Trend
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
