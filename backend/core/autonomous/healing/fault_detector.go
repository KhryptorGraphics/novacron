package healing

import (
	"context"
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

// Fault represents a detected system fault
type Fault struct {
	ID          string
	Type        string
	Component   string
	Description string
	Severity    float64
	Timestamp   time.Time
	Metrics     map[string]float64
	Anomaly     bool
}

// AnomalyDetector detects anomalies using statistical methods
type AnomalyDetector struct {
	logger           *zap.Logger
	windowSize       int
	threshold        float64
	history          map[string][]float64
	mu               sync.RWMutex
	zScoreThreshold  float64
	madMultiplier    float64
}

// HealthChecker interface for component health checks
type HealthChecker interface {
	Check(ctx context.Context) (*HealthStatus, error)
	GetComponent() string
}

// HealthStatus represents component health
type HealthStatus struct {
	Healthy   bool
	Component string
	Message   string
	Metrics   map[string]float64
}

// AlertManager manages fault alerts
type AlertManager struct {
	logger    *zap.Logger
	alerts    map[string]*Alert
	mu        sync.RWMutex
	threshold int
}

// Alert represents a system alert
type Alert struct {
	ID        string
	Severity  string
	Component string
	Message   string
	Count     int
	FirstSeen time.Time
	LastSeen  time.Time
}

// NewFaultDetector creates a new fault detector
func NewFaultDetector(logger *zap.Logger) *FaultDetector {
	return &FaultDetector{
		logger:          logger,
		anomalyDetector: NewAnomalyDetector(logger),
		healthCheckers:  make(map[string]HealthChecker),
		alertManager:    NewAlertManager(logger),
		detectionTime: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "fault_detection_duration_seconds",
			Help:    "Time taken to detect faults",
			Buckets: []float64{0.001, 0.01, 0.1, 0.5, 1.0},
		}),
	}
}

// NewAnomalyDetector creates a new anomaly detector
func NewAnomalyDetector(logger *zap.Logger) *AnomalyDetector {
	return &AnomalyDetector{
		logger:          logger,
		windowSize:      100,
		threshold:       0.95,
		history:         make(map[string][]float64),
		zScoreThreshold: 3.0,
		madMultiplier:   3.0,
	}
}

// NewAlertManager creates a new alert manager
func NewAlertManager(logger *zap.Logger) *AlertManager {
	return &AlertManager{
		logger:    logger,
		alerts:    make(map[string]*Alert),
		threshold: 3,
	}
}

// Detect performs sub-second fault detection
func (fd *FaultDetector) Detect(ctx context.Context) []*Fault {
	start := time.Now()
	defer func() {
		fd.detectionTime.Observe(time.Since(start).Seconds())
	}()

	var faults []*Fault
	var wg sync.WaitGroup

	// Parallel health checks for sub-second detection
	results := make(chan *Fault, len(fd.healthCheckers))

	for name, checker := range fd.healthCheckers {
		wg.Add(1)
		go func(n string, c HealthChecker) {
			defer wg.Done()

			// Set timeout for sub-second detection
			checkCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
			defer cancel()

			status, err := c.Check(checkCtx)
			if err != nil || !status.Healthy {
				fault := &Fault{
					ID:          generateFaultID(),
					Type:        "health_check_failure",
					Component:   n,
					Description: status.Message,
					Severity:    fd.calculateSeverity(status),
					Timestamp:   time.Now(),
					Metrics:     status.Metrics,
				}
				results <- fault
			}

			// Check for anomalies in metrics
			for metric, value := range status.Metrics {
				if fd.anomalyDetector.IsAnomaly(metric, value) {
					fault := &Fault{
						ID:          generateFaultID(),
						Type:        "anomaly_detected",
						Component:   n,
						Description: "Anomaly detected in " + metric,
						Severity:    fd.calculateAnomalySeverity(metric, value),
						Timestamp:   time.Now(),
						Metrics:     map[string]float64{metric: value},
						Anomaly:     true,
					}
					results <- fault
				}
			}
		}(name, checker)
	}

	// Close results channel when all checks complete
	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect faults
	for fault := range results {
		faults = append(faults, fault)
		fd.alertManager.ProcessFault(fault)
	}

	// Check for pattern-based faults
	patternFaults := fd.detectPatterns(ctx)
	faults = append(faults, patternFaults...)

	return faults
}

// IsAnomaly detects if a value is anomalous
func (ad *AnomalyDetector) IsAnomaly(metric string, value float64) bool {
	ad.mu.Lock()
	defer ad.mu.Unlock()

	// Initialize history if needed
	if _, exists := ad.history[metric]; !exists {
		ad.history[metric] = make([]float64, 0, ad.windowSize)
	}

	history := ad.history[metric]

	// Not enough data yet
	if len(history) < 10 {
		ad.history[metric] = append(history, value)
		return false
	}

	// Calculate Z-score
	mean := ad.calculateMean(history)
	stdDev := ad.calculateStdDev(history, mean)

	if stdDev == 0 {
		return false
	}

	zScore := math.Abs((value - mean) / stdDev)

	// Calculate MAD (Median Absolute Deviation)
	mad := ad.calculateMAD(history)
	madScore := math.Abs((value - ad.calculateMedian(history)) / mad)

	// Update history
	if len(history) >= ad.windowSize {
		ad.history[metric] = append(history[1:], value)
	} else {
		ad.history[metric] = append(history, value)
	}

	// Anomaly if either Z-score or MAD score exceeds threshold
	return zScore > ad.zScoreThreshold || madScore > ad.madMultiplier
}

// calculateMean calculates the mean of values
func (ad *AnomalyDetector) calculateMean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// calculateStdDev calculates standard deviation
func (ad *AnomalyDetector) calculateStdDev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}

	return math.Sqrt(sumSquares / float64(len(values)))
}

// calculateMedian calculates the median of values
func (ad *AnomalyDetector) calculateMedian(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Simple median calculation (could be optimized)
	sorted := make([]float64, len(values))
	copy(sorted, values)

	// Bubble sort for simplicity (replace with better sorting for production)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}

// calculateMAD calculates Median Absolute Deviation
func (ad *AnomalyDetector) calculateMAD(values []float64) float64 {
	median := ad.calculateMedian(values)

	deviations := make([]float64, len(values))
	for i, v := range values {
		deviations[i] = math.Abs(v - median)
	}

	mad := ad.calculateMedian(deviations)
	if mad == 0 {
		return 1.0 // Avoid division by zero
	}
	return mad
}

// calculateSeverity calculates fault severity
func (fd *FaultDetector) calculateSeverity(status *HealthStatus) float64 {
	severity := 0.5 // Base severity

	// Increase severity based on metrics
	for metric, value := range status.Metrics {
		switch metric {
		case "cpu_usage":
			if value > 0.9 {
				severity = math.Max(severity, 0.9)
			} else if value > 0.8 {
				severity = math.Max(severity, 0.7)
			}
		case "memory_usage":
			if value > 0.95 {
				severity = math.Max(severity, 0.95)
			} else if value > 0.85 {
				severity = math.Max(severity, 0.75)
			}
		case "error_rate":
			if value > 0.1 {
				severity = math.Max(severity, 0.85)
			}
		case "response_time":
			if value > 5.0 {
				severity = math.Max(severity, 0.8)
			}
		}
	}

	return severity
}

// calculateAnomalySeverity calculates anomaly severity
func (fd *FaultDetector) calculateAnomalySeverity(metric string, value float64) float64 {
	// Base severity on metric type and deviation
	baseSeverity := 0.6

	switch metric {
	case "cpu_usage", "memory_usage":
		if value > 0.95 {
			return 0.95
		}
		return baseSeverity + (value * 0.3)
	case "error_rate":
		if value > 0.2 {
			return 0.9
		}
		return baseSeverity + (value * 2)
	case "latency", "response_time":
		if value > 10.0 {
			return 0.85
		}
		return baseSeverity + (value / 20)
	default:
		return baseSeverity
	}
}

// detectPatterns detects pattern-based faults
func (fd *FaultDetector) detectPatterns(ctx context.Context) []*Fault {
	var faults []*Fault

	// Check for cascading failures
	if fd.alertManager.HasCascadingFailures() {
		faults = append(faults, &Fault{
			ID:          generateFaultID(),
			Type:        "cascading_failure",
			Component:   "system",
			Description: "Cascading failures detected across multiple components",
			Severity:    0.9,
			Timestamp:   time.Now(),
		})
	}

	// Check for thundering herd
	if fd.alertManager.HasThunderingHerd() {
		faults = append(faults, &Fault{
			ID:          generateFaultID(),
			Type:        "thundering_herd",
			Component:   "system",
			Description: "Thundering herd pattern detected",
			Severity:    0.8,
			Timestamp:   time.Now(),
		})
	}

	return faults
}

// ProcessFault processes a detected fault
func (am *AlertManager) ProcessFault(fault *Fault) {
	am.mu.Lock()
	defer am.mu.Unlock()

	key := fault.Component + ":" + fault.Type

	if alert, exists := am.alerts[key]; exists {
		alert.Count++
		alert.LastSeen = time.Now()
	} else {
		am.alerts[key] = &Alert{
			ID:        fault.ID,
			Severity:  am.getSeverityLevel(fault.Severity),
			Component: fault.Component,
			Message:   fault.Description,
			Count:     1,
			FirstSeen: time.Now(),
			LastSeen:  time.Now(),
		}
	}
}

// HasCascadingFailures checks for cascading failures
func (am *AlertManager) HasCascadingFailures() bool {
	am.mu.RLock()
	defer am.mu.RUnlock()

	componentFailures := make(map[string]int)
	for _, alert := range am.alerts {
		if time.Since(alert.LastSeen) < 5*time.Minute {
			componentFailures[alert.Component]++
		}
	}

	// Cascading if multiple components failing
	failedComponents := 0
	for _, count := range componentFailures {
		if count >= am.threshold {
			failedComponents++
		}
	}

	return failedComponents >= 3
}

// HasThunderingHerd checks for thundering herd pattern
func (am *AlertManager) HasThunderingHerd() bool {
	am.mu.RLock()
	defer am.mu.RUnlock()

	recentAlerts := 0
	for _, alert := range am.alerts {
		if time.Since(alert.FirstSeen) < 30*time.Second {
			recentAlerts += alert.Count
		}
	}

	// Thundering herd if many alerts in short time
	return recentAlerts > 20
}

// getSeverityLevel converts numeric severity to level
func (am *AlertManager) getSeverityLevel(severity float64) string {
	switch {
	case severity >= 0.9:
		return "critical"
	case severity >= 0.7:
		return "high"
	case severity >= 0.5:
		return "medium"
	case severity >= 0.3:
		return "low"
	default:
		return "info"
	}
}

// generateFaultID generates unique fault ID
func generateFaultID() string {
	return "fault-" + generateID()
}

// RegisterHealthChecker registers a health checker
func (fd *FaultDetector) RegisterHealthChecker(name string, checker HealthChecker) {
	fd.healthCheckers[name] = checker
	fd.logger.Info("Registered health checker", zap.String("name", name))
}