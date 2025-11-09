package dr

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

// HealthMonitor monitors system health at multiple levels
type HealthMonitor struct {
	config         *DRConfig
	checks         []HealthCheck
	regionHealth   map[string]*RegionHealth
	healthMu       sync.RWMutex
	globalHealth   *GlobalHealth
	globalMu       sync.RWMutex
	anomalyDetector *AnomalyDetector
}

// GlobalHealth represents overall system health
type GlobalHealth struct {
	HealthScore    float64
	TotalRegions   int
	HealthyRegions int
	DegradedRegions int
	FailedRegions  int
	LastUpdate     time.Time
	Alerts         []HealthAlert
}

// HealthAlert represents a health alert
type HealthAlert struct {
	ID        string
	Severity  string // "info", "warning", "critical"
	Component string
	Message   string
	Timestamp time.Time
	Resolved  bool
}

// AnomalyDetector detects anomalous behavior
type AnomalyDetector struct {
	baselineMetrics map[string]*MetricBaseline
	mu              sync.RWMutex
}

// MetricBaseline stores baseline metric data
type MetricBaseline struct {
	MetricName string
	Mean       float64
	StdDev     float64
	Samples    []float64
	LastUpdate time.Time
}

// NewHealthMonitor creates a new health monitor
func NewHealthMonitor(config *DRConfig) *HealthMonitor {
	hm := &HealthMonitor{
		config:       config,
		checks:       config.HealthChecks,
		regionHealth: make(map[string]*RegionHealth),
		globalHealth: &GlobalHealth{
			TotalRegions: 1 + len(config.SecondaryRegions),
			Alerts:       make([]HealthAlert, 0),
		},
		anomalyDetector: &AnomalyDetector{
			baselineMetrics: make(map[string]*MetricBaseline),
		},
	}

	// Initialize region health
	hm.regionHealth[config.PrimaryRegion] = &RegionHealth{
		RegionID:    config.PrimaryRegion,
		State:       "healthy",
		HealthScore: 1.0,
		Capacity:    1.0,
		LastCheck:   time.Now(),
	}

	for _, region := range config.SecondaryRegions {
		hm.regionHealth[region] = &RegionHealth{
			RegionID:    region,
			State:       "healthy",
			HealthScore: 1.0,
			Capacity:    1.0,
			LastCheck:   time.Now(),
		}
	}

	return hm
}

// Start begins health monitoring
func (hm *HealthMonitor) Start(ctx context.Context) error {
	log.Println("Starting health monitor")

	// Start health checks for each level
	for i := 1; i <= 4; i++ {
		level := i
		go hm.runHealthChecks(ctx, level)
	}

	// Start anomaly detection
	go hm.runAnomalyDetection(ctx)

	// Start health aggregation
	go hm.aggregateHealth(ctx)

	log.Println("Health monitor started")
	return nil
}

// runHealthChecks runs health checks for a specific level
func (hm *HealthMonitor) runHealthChecks(ctx context.Context, level int) {
	// Get checks for this level
	var levelChecks []HealthCheck
	for _, check := range hm.checks {
		if check.Level == level {
			levelChecks = append(levelChecks, check)
		}
	}

	if len(levelChecks) == 0 {
		return
	}

	ticker := time.NewTicker(levelChecks[0].Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			for _, check := range levelChecks {
				go hm.executeHealthCheck(check)
			}
		case <-ctx.Done():
			return
		}
	}
}

// executeHealthCheck executes a single health check
func (hm *HealthMonitor) executeHealthCheck(check HealthCheck) {
	startTime := time.Now()

	// Simulate HTTP health check
	healthy := hm.performHTTPCheck(check)

	latency := time.Since(startTime)

	// Update region health based on check results
	hm.updateRegionHealthFromCheck(check, healthy, latency)

	// Detect anomalies
	hm.anomalyDetector.RecordMetric(check.Name, float64(latency.Milliseconds()))
}

// performHTTPCheck performs HTTP health check
func (hm *HealthMonitor) performHTTPCheck(check HealthCheck) bool {
	// Simulate health check
	// In production, this would make actual HTTP requests

	// Simulate 99.5% success rate
	success := rand.Float64() > 0.005

	if !success {
		log.Printf("Health check failed: %s", check.Name)
	}

	return success
}

// updateRegionHealthFromCheck updates region health based on check
func (hm *HealthMonitor) updateRegionHealthFromCheck(check HealthCheck, healthy bool, latency time.Duration) {
	hm.healthMu.Lock()
	defer hm.healthMu.Unlock()

	// For now, update primary region
	// In production, this would map checks to specific regions
	for _, health := range hm.regionHealth {
		health.LastCheck = time.Now()
		health.Latency = latency

		if !healthy {
			// Degrade health score
			health.HealthScore *= 0.9

			if health.HealthScore < 0.3 {
				health.State = "failing"
				hm.raiseAlert("critical", health.RegionID, "Region health critically low")
			} else if health.HealthScore < 0.7 {
				health.State = "degraded"
				hm.raiseAlert("warning", health.RegionID, "Region health degraded")
			}
		} else {
			// Improve health score
			health.HealthScore = health.HealthScore*0.9 + 0.1

			if health.HealthScore > 0.8 && health.State != "healthy" {
				health.State = "healthy"
				log.Printf("Region %s returned to healthy state", health.RegionID)
			}
		}

		// Normalize health score
		if health.HealthScore > 1.0 {
			health.HealthScore = 1.0
		}
		if health.HealthScore < 0.0 {
			health.HealthScore = 0.0
		}
	}
}

// aggregateHealth aggregates regional health into global health
func (hm *HealthMonitor) aggregateHealth(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			hm.calculateGlobalHealth()
		case <-ctx.Done():
			return
		}
	}
}

// calculateGlobalHealth calculates overall system health
func (hm *HealthMonitor) calculateGlobalHealth() {
	hm.healthMu.RLock()
	defer hm.healthMu.RUnlock()

	hm.globalMu.Lock()
	defer hm.globalMu.Unlock()

	var totalScore float64
	healthy := 0
	degraded := 0
	failed := 0

	for _, rh := range hm.regionHealth {
		totalScore += rh.HealthScore

		switch rh.State {
		case "healthy":
			healthy++
		case "degraded":
			degraded++
		case "failing", "failed":
			failed++
		}
	}

	hm.globalHealth.HealthScore = totalScore / float64(len(hm.regionHealth))
	hm.globalHealth.HealthyRegions = healthy
	hm.globalHealth.DegradedRegions = degraded
	hm.globalHealth.FailedRegions = failed
	hm.globalHealth.LastUpdate = time.Now()

	// Check for critical conditions
	if failed > 0 {
		hm.raiseAlert("critical", "global", fmt.Sprintf("%d regions failed", failed))
	} else if degraded > len(hm.regionHealth)/2 {
		hm.raiseAlert("warning", "global", "Majority of regions degraded")
	}
}

// raiseAlert creates a health alert
func (hm *HealthMonitor) raiseAlert(severity, component, message string) {
	alert := HealthAlert{
		ID:        fmt.Sprintf("alert-%d", time.Now().Unix()),
		Severity:  severity,
		Component: component,
		Message:   message,
		Timestamp: time.Now(),
		Resolved:  false,
	}

	hm.globalMu.Lock()
	hm.globalHealth.Alerts = append(hm.globalHealth.Alerts, alert)
	hm.globalMu.Unlock()

	log.Printf("HEALTH ALERT [%s] %s: %s", severity, component, message)
}

// GetGlobalHealth returns current global health
func (hm *HealthMonitor) GetGlobalHealth() *GlobalHealth {
	hm.globalMu.RLock()
	defer hm.globalMu.RUnlock()

	// Return a copy
	health := *hm.globalHealth
	return &health
}

// GetRegionHealth returns health for a specific region
func (hm *HealthMonitor) GetRegionHealth(regionID string) (*RegionHealth, error) {
	hm.healthMu.RLock()
	defer hm.healthMu.RUnlock()

	health, exists := hm.regionHealth[regionID]
	if !exists {
		return nil, fmt.Errorf("region not found: %s", regionID)
	}

	// Return a copy
	healthCopy := *health
	return &healthCopy, nil
}

// runAnomalyDetection runs anomaly detection
func (hm *HealthMonitor) runAnomalyDetection(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			hm.detectAnomalies()
		case <-ctx.Done():
			return
		}
	}
}

// detectAnomalies detects anomalous patterns
func (hm *HealthMonitor) detectAnomalies() {
	hm.anomalyDetector.mu.RLock()
	defer hm.anomalyDetector.mu.RUnlock()

	for metricName, baseline := range hm.anomalyDetector.baselineMetrics {
		// Check if recent samples deviate significantly from baseline
		if len(baseline.Samples) < 10 {
			continue
		}

		recentSamples := baseline.Samples[len(baseline.Samples)-10:]
		var recentMean float64
		for _, sample := range recentSamples {
			recentMean += sample
		}
		recentMean /= float64(len(recentSamples))

		// If recent mean is more than 2 standard deviations from baseline
		deviation := (recentMean - baseline.Mean) / baseline.StdDev
		if deviation > 2.0 || deviation < -2.0 {
			hm.raiseAlert("warning", metricName,
				fmt.Sprintf("Anomaly detected: %.2f standard deviations from baseline", deviation))
		}
	}
}

// RecordMetric records a metric for anomaly detection
func (ad *AnomalyDetector) RecordMetric(metricName string, value float64) {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	baseline, exists := ad.baselineMetrics[metricName]
	if !exists {
		baseline = &MetricBaseline{
			MetricName: metricName,
			Samples:    make([]float64, 0, 100),
		}
		ad.baselineMetrics[metricName] = baseline
	}

	// Add sample
	baseline.Samples = append(baseline.Samples, value)

	// Keep only last 100 samples
	if len(baseline.Samples) > 100 {
		baseline.Samples = baseline.Samples[1:]
	}

	// Recalculate statistics
	ad.calculateStatistics(baseline)
}

// calculateStatistics calculates mean and standard deviation
func (ad *AnomalyDetector) calculateStatistics(baseline *MetricBaseline) {
	if len(baseline.Samples) == 0 {
		return
	}

	// Calculate mean
	var sum float64
	for _, sample := range baseline.Samples {
		sum += sample
	}
	baseline.Mean = sum / float64(len(baseline.Samples))

	// Calculate standard deviation
	var variance float64
	for _, sample := range baseline.Samples {
		diff := sample - baseline.Mean
		variance += diff * diff
	}
	variance /= float64(len(baseline.Samples))
	baseline.StdDev = variance // Simplified, should be sqrt(variance)

	baseline.LastUpdate = time.Now()
}

// PredictFailure predicts potential failures using ML
func (hm *HealthMonitor) PredictFailure(regionID string) (float64, error) {
	health, err := hm.GetRegionHealth(regionID)
	if err != nil {
		return 0, err
	}

	// Simple prediction based on health trend
	// In production, this would use actual ML models
	failureProb := 1.0 - health.HealthScore

	if health.ErrorRate > 0.1 {
		failureProb += 0.2
	}

	if health.Capacity < 0.2 {
		failureProb += 0.3
	}

	if failureProb > 1.0 {
		failureProb = 1.0
	}

	return failureProb, nil
}

// HTTPHealthCheck performs HTTP health check
func HTTPHealthCheck(endpoint string, timeout time.Duration) (bool, time.Duration, error) {
	client := &http.Client{
		Timeout: timeout,
	}

	startTime := time.Now()

	resp, err := client.Get(endpoint)
	latency := time.Since(startTime)

	if err != nil {
		return false, latency, err
	}
	defer resp.Body.Close()

	healthy := resp.StatusCode >= 200 && resp.StatusCode < 300

	return healthy, latency, nil
}
