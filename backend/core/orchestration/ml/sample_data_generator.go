// Package ml provides sample data generation for ML model training and testing
package ml

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/sirupsen/logrus"
)

// SampleDataGenerator generates realistic training data for ML models
type SampleDataGenerator struct {
	logger *logrus.Logger
	rand   *rand.Rand
}

// DataProfile defines characteristics of generated data
type DataProfile struct {
	ModelType         ModelType              `json:"model_type"`
	SampleCount       int                    `json:"sample_count"`
	NoiseLevel        float64                `json:"noise_level"`
	SeasonalityFactor float64                `json:"seasonality_factor"`
	TrendFactor       float64                `json:"trend_factor"`
	OutlierPercent    float64                `json:"outlier_percent"`
	FeatureScaling    map[string]ScaleParams `json:"feature_scaling"`
	TimeRange         TimeRange              `json:"time_range"`
}

// ScaleParams defines scaling parameters for features
type ScaleParams struct {
	Min         float64 `json:"min"`
	Max         float64 `json:"max"`
	Mean        float64 `json:"mean"`
	StdDev      float64 `json:"std_dev"`
	Distribution string `json:"distribution"` // uniform, normal, exponential
}

// TimeRange defines time range for temporal data
type TimeRange struct {
	Start    time.Time `json:"start"`
	End      time.Time `json:"end"`
	Interval time.Duration `json:"interval"`
}

// GeneratedDataset contains generated training data
type GeneratedDataset struct {
	Profile      DataProfile          `json:"profile"`
	Features     [][]float64          `json:"features"`
	Labels       []float64            `json:"labels"`
	FeatureNames []string             `json:"feature_names"`
	Timestamps   []time.Time          `json:"timestamps,omitempty"`
	Metadata     map[string]interface{} `json:"metadata"`
	GeneratedAt  time.Time            `json:"generated_at"`
}

// NewSampleDataGenerator creates a new sample data generator
func NewSampleDataGenerator(logger *logrus.Logger) *SampleDataGenerator {
	return &SampleDataGenerator{
		logger: logger,
		rand:   rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// GenerateDataset generates a complete dataset based on profile
func (sdg *SampleDataGenerator) GenerateDataset(profile DataProfile) (*GeneratedDataset, error) {
	sdg.logger.WithFields(logrus.Fields{
		"model_type":    profile.ModelType,
		"sample_count":  profile.SampleCount,
		"noise_level":   profile.NoiseLevel,
	}).Info("Generating sample dataset")

	switch profile.ModelType {
	case ModelTypePlacementPredictor:
		return sdg.generatePlacementData(profile)
	case ModelTypeScalingPredictor:
		return sdg.generateScalingData(profile)
	case ModelTypeResourcePredictor:
		return sdg.generateResourceData(profile)
	case ModelTypeFailurePredictor:
		return sdg.generateFailureData(profile)
	default:
		return sdg.generateGenericData(profile)
	}
}

// generatePlacementData generates VM placement prediction data
func (sdg *SampleDataGenerator) generatePlacementData(profile DataProfile) (*GeneratedDataset, error) {
	features := make([][]float64, profile.SampleCount)
	labels := make([]float64, profile.SampleCount)
	featureNames := []string{
		"vm_cpu_cores", "vm_memory_mb", "vm_disk_gb", "vm_network_mbps",
		"node_cpu_cores", "node_memory_mb", "node_disk_gb", "node_load_pct",
		"node_network_util", "placement_score", "affinity_score", "cost_factor",
	}
	
	timestamps := sdg.generateTimestamps(profile)
	
	for i := 0; i < profile.SampleCount; i++ {
		// Generate VM requirements
		vmCores := float64(1 + sdg.rand.Intn(16))
		vmMemory := float64(1024 + sdg.rand.Intn(32)*1024)
		vmDisk := float64(20 + sdg.rand.Intn(500))
		vmNetwork := float64(100 + sdg.rand.Intn(900))
		
		// Generate node characteristics
		nodeCores := float64(4 + sdg.rand.Intn(32))
		nodeMemory := float64(8192 + sdg.rand.Intn(64)*1024)
		nodeDisk := float64(500 + sdg.rand.Intn(2000))
		nodeLoad := sdg.rand.Float64()
		nodeNetworkUtil := sdg.rand.Float64()
		
		// Calculate placement factors
		resourceFit := sdg.calculateResourceFit(vmCores, vmMemory, vmDisk, nodeCores, nodeMemory, nodeDisk)
		loadPenalty := nodeLoad * 0.5
		networkScore := 1.0 - nodeNetworkUtil*0.3
		
		// Add seasonality and trend
		timeScore := sdg.addTimeBasedVariation(i, profile, timestamps[i])
		
		// Affinity and cost factors
		affinityScore := sdg.rand.Float64()
		costFactor := sdg.rand.Float64()
		
		features[i] = []float64{
			vmCores, vmMemory, vmDisk, vmNetwork,
			nodeCores, nodeMemory, nodeDisk, nodeLoad,
			nodeNetworkUtil, resourceFit, affinityScore, costFactor,
		}
		
		// Calculate placement success probability
		placementScore := resourceFit * (1.0 - loadPenalty) * networkScore * timeScore
		placementScore = math.Max(0, math.Min(1, placementScore))
		
		// Add noise
		placementScore = sdg.addNoise(placementScore, profile.NoiseLevel)
		
		// Add outliers
		if sdg.rand.Float64() < profile.OutlierPercent {
			placementScore = sdg.generateOutlier(placementScore)
		}
		
		labels[i] = placementScore
	}
	
	dataset := &GeneratedDataset{
		Profile:      profile,
		Features:     features,
		Labels:       labels,
		FeatureNames: featureNames,
		Timestamps:   timestamps,
		GeneratedAt:  time.Now(),
		Metadata: map[string]interface{}{
			"description": "VM placement prediction data with resource utilization and affinity factors",
			"label_type":  "placement_success_probability",
			"features":    len(featureNames),
		},
	}
	
	return dataset, nil
}

// generateScalingData generates auto-scaling prediction data
func (sdg *SampleDataGenerator) generateScalingData(profile DataProfile) (*GeneratedDataset, error) {
	features := make([][]float64, profile.SampleCount)
	labels := make([]float64, profile.SampleCount)
	featureNames := []string{
		"cpu_utilization", "memory_utilization", "disk_io_util", "network_io_util",
		"request_rate", "response_time_ms", "error_rate", "queue_length",
		"active_connections", "throughput_rps", "cache_hit_rate", "db_connection_pool",
	}
	
	timestamps := sdg.generateTimestamps(profile)
	
	for i := 0; i < profile.SampleCount; i++ {
		// Base utilization metrics
		cpuUtil := sdg.rand.Float64()
		memUtil := sdg.rand.Float64()
		diskUtil := sdg.rand.Float64()
		networkUtil := sdg.rand.Float64()
		
		// Application metrics
		requestRate := 50.0 + sdg.rand.Float64()*950.0
		responseTime := 50.0 + sdg.rand.Float64()*450.0
		errorRate := sdg.rand.Float64() * 0.1
		queueLength := sdg.rand.Float64() * 100
		
		// Connection metrics
		activeConns := sdg.rand.Float64() * 1000
		throughput := requestRate * (1.0 - errorRate)
		cacheHitRate := 0.7 + sdg.rand.Float64()*0.29
		dbConnPool := sdg.rand.Float64() * 0.8
		
		// Add temporal patterns
		timeMultiplier := sdg.addTimeBasedVariation(i, profile, timestamps[i])
		cpuUtil *= timeMultiplier
		requestRate *= timeMultiplier
		
		features[i] = []float64{
			cpuUtil, memUtil, diskUtil, networkUtil,
			requestRate, responseTime, errorRate, queueLength,
			activeConns, throughput, cacheHitRate, dbConnPool,
		}
		
		// Determine scaling decision
		scalingDecision := sdg.calculateScalingDecision(cpuUtil, memUtil, requestRate, responseTime, errorRate)
		
		// Add noise and outliers
		scalingDecision = sdg.addNoise(scalingDecision, profile.NoiseLevel)
		if sdg.rand.Float64() < profile.OutlierPercent {
			scalingDecision = sdg.generateOutlier(scalingDecision)
		}
		
		labels[i] = scalingDecision
	}
	
	dataset := &GeneratedDataset{
		Profile:      profile,
		Features:     features,
		Labels:       labels,
		FeatureNames: featureNames,
		Timestamps:   timestamps,
		GeneratedAt:  time.Now(),
		Metadata: map[string]interface{}{
			"description": "Auto-scaling decision data with resource and application metrics",
			"label_type":  "scaling_decision", // -1 = scale down, 0 = no change, 1 = scale up
			"features":    len(featureNames),
		},
	}
	
	return dataset, nil
}

// generateResourceData generates resource prediction data
func (sdg *SampleDataGenerator) generateResourceData(profile DataProfile) (*GeneratedDataset, error) {
	features := make([][]float64, profile.SampleCount)
	labels := make([]float64, profile.SampleCount)
	featureNames := []string{
		"historical_cpu_1h", "historical_cpu_24h", "historical_mem_1h", "historical_mem_24h",
		"workload_type", "user_count", "scheduled_jobs", "day_of_week",
		"hour_of_day", "seasonal_factor", "growth_trend", "external_load",
	}
	
	timestamps := sdg.generateTimestamps(profile)
	
	for i := 0; i < profile.SampleCount; i++ {
		// Historical metrics
		histCpu1h := sdg.rand.Float64()
		histCpu24h := sdg.rand.Float64()
		histMem1h := sdg.rand.Float64()
		histMem24h := sdg.rand.Float64()
		
		// Workload characteristics
		workloadType := float64(sdg.rand.Intn(5)) // 5 different workload types
		userCount := 100.0 + sdg.rand.Float64()*9900.0
		scheduledJobs := float64(sdg.rand.Intn(50))
		
		// Time features
		dayOfWeek := float64(timestamps[i].Weekday())
		hourOfDay := float64(timestamps[i].Hour())
		seasonalFactor := math.Sin(float64(timestamps[i].YearDay()) * 2 * math.Pi / 365)
		
		// Growth and external factors
		growthTrend := profile.TrendFactor * float64(i) / float64(profile.SampleCount)
		externalLoad := sdg.rand.Float64()
		
		features[i] = []float64{
			histCpu1h, histCpu24h, histMem1h, histMem24h,
			workloadType, userCount, scheduledJobs, dayOfWeek,
			hourOfDay, seasonalFactor, growthTrend, externalLoad,
		}
		
		// Predict future resource usage
		futureUsage := sdg.calculateFutureResourceUsage(
			histCpu1h, histMem1h, userCount, hourOfDay, seasonalFactor, growthTrend,
		)
		
		// Add noise and outliers
		futureUsage = sdg.addNoise(futureUsage, profile.NoiseLevel)
		if sdg.rand.Float64() < profile.OutlierPercent {
			futureUsage = sdg.generateOutlier(futureUsage)
		}
		
		labels[i] = futureUsage
	}
	
	dataset := &GeneratedDataset{
		Profile:      profile,
		Features:     features,
		Labels:       labels,
		FeatureNames: featureNames,
		Timestamps:   timestamps,
		GeneratedAt:  time.Now(),
		Metadata: map[string]interface{}{
			"description": "Resource usage prediction data with historical and temporal features",
			"label_type":  "future_resource_utilization",
			"features":    len(featureNames),
			"prediction_horizon": "1 hour",
		},
	}
	
	return dataset, nil
}

// generateFailureData generates failure prediction data
func (sdg *SampleDataGenerator) generateFailureData(profile DataProfile) (*GeneratedDataset, error) {
	features := make([][]float64, profile.SampleCount)
	labels := make([]float64, profile.SampleCount)
	featureNames := []string{
		"cpu_anomaly_score", "memory_leak_indicator", "disk_health_score", "network_errors",
		"service_response_time", "error_rate_trend", "log_anomalies", "dependency_health",
		"uptime_hours", "restart_count", "patch_level", "config_changes",
	}
	
	timestamps := sdg.generateTimestamps(profile)
	
	for i := 0; i < profile.SampleCount; i++ {
		// Health indicators
		cpuAnomaly := sdg.rand.Float64()
		memoryLeak := sdg.rand.Float64()
		diskHealth := 1.0 - sdg.rand.Float64()*0.3 // Generally healthy
		networkErrors := sdg.rand.Float64() * 0.1
		
		// Service metrics
		responseTime := 50.0 + sdg.rand.Float64()*200.0
		errorRateTrend := sdg.rand.Float64() * 0.05
		logAnomalies := sdg.rand.Float64()
		dependencyHealth := 0.8 + sdg.rand.Float64()*0.2
		
		// System info
		uptimeHours := sdg.rand.Float64() * 720 // Up to 30 days
		restartCount := float64(sdg.rand.Intn(10))
		patchLevel := sdg.rand.Float64()
		configChanges := float64(sdg.rand.Intn(5))
		
		features[i] = []float64{
			cpuAnomaly, memoryLeak, diskHealth, networkErrors,
			responseTime, errorRateTrend, logAnomalies, dependencyHealth,
			uptimeHours, restartCount, patchLevel, configChanges,
		}
		
		// Calculate failure probability
		failureProbability := sdg.calculateFailureProbability(
			cpuAnomaly, memoryLeak, diskHealth, networkErrors,
			errorRateTrend, dependencyHealth, restartCount,
		)
		
		// Add noise and outliers
		failureProbability = sdg.addNoise(failureProbability, profile.NoiseLevel)
		if sdg.rand.Float64() < profile.OutlierPercent {
			failureProbability = sdg.generateOutlier(failureProbability)
		}
		
		labels[i] = failureProbability
	}
	
	dataset := &GeneratedDataset{
		Profile:      profile,
		Features:     features,
		Labels:       labels,
		FeatureNames: featureNames,
		Timestamps:   timestamps,
		GeneratedAt:  time.Now(),
		Metadata: map[string]interface{}{
			"description": "Failure prediction data with health indicators and system metrics",
			"label_type":  "failure_probability",
			"features":    len(featureNames),
			"prediction_window": "24 hours",
		},
	}
	
	return dataset, nil
}

// generateGenericData generates generic ML training data
func (sdg *SampleDataGenerator) generateGenericData(profile DataProfile) (*GeneratedDataset, error) {
	featureCount := 10
	features := make([][]float64, profile.SampleCount)
	labels := make([]float64, profile.SampleCount)
	featureNames := make([]string, featureCount)
	
	for i := 0; i < featureCount; i++ {
		featureNames[i] = fmt.Sprintf("feature_%d", i)
	}
	
	timestamps := sdg.generateTimestamps(profile)
	
	for i := 0; i < profile.SampleCount; i++ {
		features[i] = make([]float64, featureCount)
		
		for j := 0; j < featureCount; j++ {
			features[i][j] = sdg.rand.NormFloat64()
		}
		
		// Generate label as linear combination with noise
		label := 0.0
		for j := 0; j < featureCount; j++ {
			weight := sdg.rand.Float64()
			label += features[i][j] * weight
		}
		
		// Add noise and outliers
		label = sdg.addNoise(label, profile.NoiseLevel)
		if sdg.rand.Float64() < profile.OutlierPercent {
			label = sdg.generateOutlier(label)
		}
		
		labels[i] = label
	}
	
	dataset := &GeneratedDataset{
		Profile:      profile,
		Features:     features,
		Labels:       labels,
		FeatureNames: featureNames,
		Timestamps:   timestamps,
		GeneratedAt:  time.Now(),
		Metadata: map[string]interface{}{
			"description": "Generic synthetic dataset with normally distributed features",
			"label_type":  "continuous_target",
			"features":    featureCount,
		},
	}
	
	return dataset, nil
}

// SaveDatasetToFiles saves dataset to various file formats
func (sdg *SampleDataGenerator) SaveDatasetToFiles(dataset *GeneratedDataset, outputDir string) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Save as JSON
	jsonPath := filepath.Join(outputDir, "dataset.json")
	if err := sdg.saveAsJSON(dataset, jsonPath); err != nil {
		return fmt.Errorf("failed to save JSON: %w", err)
	}
	
	// Save as CSV
	csvPath := filepath.Join(outputDir, "dataset.csv")
	if err := sdg.saveAsCSV(dataset, csvPath); err != nil {
		return fmt.Errorf("failed to save CSV: %w", err)
	}
	
	// Save metadata
	metadataPath := filepath.Join(outputDir, "metadata.json")
	if err := sdg.saveMetadata(dataset, metadataPath); err != nil {
		return fmt.Errorf("failed to save metadata: %w", err)
	}
	
	sdg.logger.WithFields(logrus.Fields{
		"output_dir": outputDir,
		"samples":    len(dataset.Features),
		"features":   len(dataset.FeatureNames),
	}).Info("Dataset saved to files")
	
	return nil
}

// Helper methods

func (sdg *SampleDataGenerator) generateTimestamps(profile DataProfile) []time.Time {
	timestamps := make([]time.Time, profile.SampleCount)
	
	if profile.TimeRange.Start.IsZero() {
		profile.TimeRange.Start = time.Now().AddDate(0, -1, 0) // 1 month ago
	}
	if profile.TimeRange.End.IsZero() {
		profile.TimeRange.End = time.Now()
	}
	if profile.TimeRange.Interval == 0 {
		profile.TimeRange.Interval = time.Hour
	}
	
	for i := 0; i < profile.SampleCount; i++ {
		offset := time.Duration(i) * profile.TimeRange.Interval
		timestamps[i] = profile.TimeRange.Start.Add(offset)
	}
	
	return timestamps
}

func (sdg *SampleDataGenerator) calculateResourceFit(vmCpu, vmMem, vmDisk, nodeCpu, nodeMem, nodeDisk float64) float64 {
	cpuFit := math.Min(nodeCpu/vmCpu, 2.0) / 2.0
	memFit := math.Min(nodeMem/vmMem, 2.0) / 2.0
	diskFit := math.Min(nodeDisk/vmDisk, 2.0) / 2.0
	
	return (cpuFit + memFit + diskFit) / 3.0
}

func (sdg *SampleDataGenerator) addTimeBasedVariation(index int, profile DataProfile, timestamp time.Time) float64 {
	// Daily pattern
	hourOfDay := float64(timestamp.Hour())
	dailyPattern := 0.7 + 0.3*math.Sin((hourOfDay-6)*math.Pi/12)
	
	// Weekly pattern
	dayOfWeek := float64(timestamp.Weekday())
	weeklyPattern := 1.0
	if dayOfWeek == 0 || dayOfWeek == 6 { // Weekend
		weeklyPattern = 0.6
	}
	
	// Seasonal pattern
	seasonalPattern := 1.0 + profile.SeasonalityFactor*math.Sin(float64(timestamp.YearDay())*2*math.Pi/365)
	
	// Trend
	trendPattern := 1.0 + profile.TrendFactor*float64(index)/float64(1000)
	
	return dailyPattern * weeklyPattern * seasonalPattern * trendPattern
}

func (sdg *SampleDataGenerator) calculateScalingDecision(cpuUtil, memUtil, requestRate, responseTime, errorRate float64) float64 {
	// Scale up conditions
	if cpuUtil > 0.8 || memUtil > 0.85 || responseTime > 300 || errorRate > 0.05 {
		return 1.0 // Scale up
	}
	
	// Scale down conditions
	if cpuUtil < 0.3 && memUtil < 0.4 && responseTime < 100 && errorRate < 0.01 {
		return -1.0 // Scale down
	}
	
	return 0.0 // No change
}

func (sdg *SampleDataGenerator) calculateFutureResourceUsage(histCpu, histMem, userCount, hourOfDay, seasonal, trend float64) float64 {
	// Base usage from historical data
	baseUsage := (histCpu + histMem) / 2.0
	
	// User load factor
	userFactor := math.Min(userCount/10000.0, 2.0)
	
	// Time of day factor
	timeFactor := 0.6 + 0.4*math.Sin((hourOfDay-6)*math.Pi/12)
	
	// Combine factors
	futureUsage := baseUsage * userFactor * timeFactor * (1.0 + seasonal*0.1) * (1.0 + trend)
	
	return math.Max(0, math.Min(1, futureUsage))
}

func (sdg *SampleDataGenerator) calculateFailureProbability(cpuAnomaly, memoryLeak, diskHealth, networkErrors, errorTrend, depHealth, restarts float64) float64 {
	// Weight different factors
	riskScore := cpuAnomaly*0.2 + memoryLeak*0.25 + (1.0-diskHealth)*0.2 + 
		networkErrors*0.1 + errorTrend*0.15 + (1.0-depHealth)*0.1
	
	// Restart penalty
	if restarts > 5 {
		riskScore += 0.1
	}
	
	return math.Max(0, math.Min(1, riskScore))
}

func (sdg *SampleDataGenerator) addNoise(value, noiseLevel float64) float64 {
	if noiseLevel <= 0 {
		return value
	}
	
	noise := sdg.rand.NormFloat64() * noiseLevel
	return value + noise
}

func (sdg *SampleDataGenerator) generateOutlier(value float64) float64 {
	// Generate extreme values
	if sdg.rand.Float64() < 0.5 {
		return value + math.Abs(sdg.rand.NormFloat64()*2) // Positive outlier
	} else {
		return value - math.Abs(sdg.rand.NormFloat64()*2) // Negative outlier
	}
}

func (sdg *SampleDataGenerator) saveAsJSON(dataset *GeneratedDataset, path string) error {
	data, err := json.MarshalIndent(dataset, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(path, data, 0644)
}

func (sdg *SampleDataGenerator) saveAsCSV(dataset *GeneratedDataset, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	
	writer := csv.NewWriter(file)
	defer writer.Flush()
	
	// Write header
	header := append(dataset.FeatureNames, "label")
	if len(dataset.Timestamps) > 0 {
		header = append([]string{"timestamp"}, header...)
	}
	
	if err := writer.Write(header); err != nil {
		return err
	}
	
	// Write data
	for i := 0; i < len(dataset.Features); i++ {
		record := make([]string, 0)
		
		// Add timestamp if available
		if len(dataset.Timestamps) > 0 {
			record = append(record, dataset.Timestamps[i].Format(time.RFC3339))
		}
		
		// Add features
		for _, feature := range dataset.Features[i] {
			record = append(record, fmt.Sprintf("%.6f", feature))
		}
		
		// Add label
		record = append(record, fmt.Sprintf("%.6f", dataset.Labels[i]))
		
		if err := writer.Write(record); err != nil {
			return err
		}
	}
	
	return nil
}

func (sdg *SampleDataGenerator) saveMetadata(dataset *GeneratedDataset, path string) error {
	metadata := map[string]interface{}{
		"profile":       dataset.Profile,
		"feature_names": dataset.FeatureNames,
		"sample_count":  len(dataset.Features),
		"feature_count": len(dataset.FeatureNames),
		"generated_at":  dataset.GeneratedAt,
		"metadata":      dataset.Metadata,
	}
	
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(path, data, 0644)
}

// GetDefaultDataProfile returns default data generation profile for a model type
func GetDefaultDataProfile(modelType ModelType, sampleCount int) DataProfile {
	return DataProfile{
		ModelType:         modelType,
		SampleCount:       sampleCount,
		NoiseLevel:        0.1,
		SeasonalityFactor: 0.2,
		TrendFactor:       0.05,
		OutlierPercent:    0.02,
		TimeRange: TimeRange{
			Start:    time.Now().AddDate(0, -3, 0), // 3 months ago
			End:      time.Now(),
			Interval: time.Hour,
		},
	}
}