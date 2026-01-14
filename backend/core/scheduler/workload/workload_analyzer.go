package workload

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// WorkloadType represents the type of workload
type WorkloadType string

// Workload types
const (
	WorkloadTypeCPUIntensive     WorkloadType = "cpu-intensive"
	WorkloadTypeMemoryIntensive  WorkloadType = "memory-intensive"
	WorkloadTypeIOIntensive      WorkloadType = "io-intensive"
	WorkloadTypeNetworkIntensive WorkloadType = "network-intensive"
	WorkloadTypeBalanced         WorkloadType = "balanced"
	WorkloadTypeUnknown          WorkloadType = "unknown"
)

// ResourceUsagePattern represents a pattern of resource usage over time
type ResourceUsagePattern struct {
	// ResourceType is the type of resource being tracked
	ResourceType string

	// UsagePattern stores historical usage data points
	// The key is a timestamp, the value is the usage percentage (0-100)
	UsagePattern map[time.Time]float64

	// VariabilityScore indicates how variable the usage is (0-1)
	// 0 = stable, 1 = highly variable
	VariabilityScore float64

	// PeakUsage is the highest observed usage
	PeakUsage float64

	// AverageUsage is the average observed usage
	AverageUsage float64

	// PredictedUsage is the predicted usage for the near future
	PredictedUsage float64
}

// WorkloadProfile represents a profile of a VM workload
type WorkloadProfile struct {
	// VMID is the unique identifier for the VM
	VMID string

	// DominantWorkloadType is the primary workload type
	DominantWorkloadType WorkloadType

	// SecondaryWorkloadType is the secondary workload type, if applicable
	SecondaryWorkloadType WorkloadType

	// ResourceUsagePatterns maps resource types to their usage patterns
	ResourceUsagePatterns map[string]*ResourceUsagePattern

	// UpdatedAt is the last time this profile was updated
	UpdatedAt time.Time

	// Confidence is a measure of how confident we are in this profile (0-1)
	Confidence float64

	// HistoryDuration is how long we've been tracking this VM
	HistoryDuration time.Duration
}

// WorkloadAnalyzerConfig contains configuration for the workload analyzer
type WorkloadAnalyzerConfig struct {
	// SamplingInterval is how often to sample resource usage
	SamplingInterval time.Duration

	// AnalysisInterval is how often to analyze workload patterns
	AnalysisInterval time.Duration

	// HistoryRetention is how long to keep historical data
	HistoryRetention time.Duration

	// MinSamplesForProfiling is minimum number of samples needed before profiling
	MinSamplesForProfiling int

	// VariabilityThreshold determines when a resource is considered variable
	VariabilityThreshold float64

	// CPUIntensiveThreshold is the threshold for considering a workload CPU-intensive
	CPUIntensiveThreshold float64

	// MemoryIntensiveThreshold is the threshold for considering a workload memory-intensive
	MemoryIntensiveThreshold float64

	// IOIntensiveThreshold is the threshold for considering a workload IO-intensive
	IOIntensiveThreshold float64

	// NetworkIntensiveThreshold is the threshold for considering a workload network-intensive
	NetworkIntensiveThreshold float64
}

// DefaultWorkloadAnalyzerConfig returns a default configuration
func DefaultWorkloadAnalyzerConfig() WorkloadAnalyzerConfig {
	return WorkloadAnalyzerConfig{
		SamplingInterval:          1 * time.Minute,
		AnalysisInterval:          10 * time.Minute,
		HistoryRetention:          7 * 24 * time.Hour, // 1 week
		MinSamplesForProfiling:    10,
		VariabilityThreshold:      0.3,
		CPUIntensiveThreshold:     70.0,
		MemoryIntensiveThreshold:  70.0,
		IOIntensiveThreshold:      70.0,
		NetworkIntensiveThreshold: 70.0,
	}
}

// ResourceSnapshot represents a snapshot of resource usage at a point in time
type ResourceSnapshot struct {
	// VMID is the unique identifier for the VM
	VMID string

	// Timestamp is when the snapshot was taken
	Timestamp time.Time

	// Metrics contains resource usage metrics
	Metrics map[string]float64
}

// WorkloadAnalyzer analyzes VM workload patterns
type WorkloadAnalyzer struct {
	config WorkloadAnalyzerConfig

	// profiles stores workload profiles for VMs
	profiles     map[string]*WorkloadProfile
	profileMutex sync.RWMutex

	// snapshots stores resource usage snapshots
	snapshots     map[string][]ResourceSnapshot
	snapshotMutex sync.RWMutex

	// collectors is a list of registered data collectors
	collectors []DataCollector

	ctx    context.Context
	cancel context.CancelFunc
}

// DataCollector is an interface for collecting resource usage data
type DataCollector interface {
	// GetResourceUsage returns current resource usage for a VM
	GetResourceUsage(ctx context.Context, vmID string) (map[string]float64, error)

	// GetSupportedMetrics returns the list of metrics this collector can provide
	GetSupportedMetrics() []string
}

// NewWorkloadAnalyzer creates a new workload analyzer
func NewWorkloadAnalyzer(config WorkloadAnalyzerConfig) *WorkloadAnalyzer {
	ctx, cancel := context.WithCancel(context.Background())

	return &WorkloadAnalyzer{
		config:     config,
		profiles:   make(map[string]*WorkloadProfile),
		snapshots:  make(map[string][]ResourceSnapshot),
		collectors: []DataCollector{},
		ctx:        ctx,
		cancel:     cancel,
	}
}

// Start starts the workload analyzer
func (wa *WorkloadAnalyzer) Start() error {
	log.Println("Starting workload analyzer")

	// Start the data collection loop
	go wa.collectionLoop()

	// Start the analysis loop
	go wa.analysisLoop()

	// Start the cleanup loop
	go wa.cleanupLoop()

	return nil
}

// Stop stops the workload analyzer
func (wa *WorkloadAnalyzer) Stop() error {
	log.Println("Stopping workload analyzer")

	wa.cancel()

	return nil
}

// RegisterDataCollector registers a data collector
func (wa *WorkloadAnalyzer) RegisterDataCollector(collector DataCollector) {
	wa.collectors = append(wa.collectors, collector)
	log.Printf("Registered data collector supporting metrics: %v", collector.GetSupportedMetrics())
}

// RegisterVM registers a VM for workload analysis
func (wa *WorkloadAnalyzer) RegisterVM(vmID string) error {
	wa.profileMutex.Lock()
	defer wa.profileMutex.Unlock()

	// Create initial profile
	wa.profiles[vmID] = &WorkloadProfile{
		VMID:                  vmID,
		DominantWorkloadType:  WorkloadTypeUnknown,
		ResourceUsagePatterns: make(map[string]*ResourceUsagePattern),
		UpdatedAt:             time.Now(),
		Confidence:            0.0,
		HistoryDuration:       0,
	}

	log.Printf("Registered VM %s for workload analysis", vmID)

	return nil
}

// UnregisterVM unregisters a VM
func (wa *WorkloadAnalyzer) UnregisterVM(vmID string) error {
	wa.profileMutex.Lock()
	defer wa.profileMutex.Unlock()

	delete(wa.profiles, vmID)

	wa.snapshotMutex.Lock()
	defer wa.snapshotMutex.Unlock()

	delete(wa.snapshots, vmID)

	log.Printf("Unregistered VM %s from workload analysis", vmID)

	return nil
}

// GetWorkloadProfile gets the workload profile for a VM
func (wa *WorkloadAnalyzer) GetWorkloadProfile(vmID string) (*WorkloadProfile, error) {
	wa.profileMutex.RLock()
	defer wa.profileMutex.RUnlock()

	profile, exists := wa.profiles[vmID]
	if !exists {
		return nil, fmt.Errorf("no workload profile for VM %s", vmID)
	}

	return profile, nil
}

// collectionLoop periodically collects resource usage data
func (wa *WorkloadAnalyzer) collectionLoop() {
	ticker := time.NewTicker(wa.config.SamplingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-wa.ctx.Done():
			return
		case <-ticker.C:
			wa.collectResourceUsage()
		}
	}
}

// analysisLoop periodically analyzes workload patterns
func (wa *WorkloadAnalyzer) analysisLoop() {
	ticker := time.NewTicker(wa.config.AnalysisInterval)
	defer ticker.Stop()

	for {
		select {
		case <-wa.ctx.Done():
			return
		case <-ticker.C:
			wa.analyzeWorkloads()
		}
	}
}

// cleanupLoop periodically cleans up old data
func (wa *WorkloadAnalyzer) cleanupLoop() {
	ticker := time.NewTicker(wa.config.AnalysisInterval * 2)
	defer ticker.Stop()

	for {
		select {
		case <-wa.ctx.Done():
			return
		case <-ticker.C:
			wa.cleanupOldData()
		}
	}
}

// collectResourceUsage collects current resource usage for all registered VMs
func (wa *WorkloadAnalyzer) collectResourceUsage() {
	wa.profileMutex.RLock()
	vmIDs := make([]string, 0, len(wa.profiles))
	for vmID := range wa.profiles {
		vmIDs = append(vmIDs, vmID)
	}
	wa.profileMutex.RUnlock()

	for _, vmID := range vmIDs {
		// Collect metrics from all registered collectors
		metrics := make(map[string]float64)

		for _, collector := range wa.collectors {
			// Create a context with timeout for each collection
			ctx, cancel := context.WithTimeout(wa.ctx, 10*time.Second)

			// Get resource usage from this collector
			collectorMetrics, err := collector.GetResourceUsage(ctx, vmID)
			cancel()

			if err != nil {
				log.Printf("Error collecting metrics for VM %s: %v", vmID, err)
				continue
			}

			// Merge metrics from this collector
			for k, v := range collectorMetrics {
				metrics[k] = v
			}
		}

		// If we got some metrics, create a snapshot
		if len(metrics) > 0 {
			snapshot := ResourceSnapshot{
				VMID:      vmID,
				Timestamp: time.Now(),
				Metrics:   metrics,
			}

			// Store the snapshot
			wa.snapshotMutex.Lock()
			if _, exists := wa.snapshots[vmID]; !exists {
				wa.snapshots[vmID] = make([]ResourceSnapshot, 0)
			}
			wa.snapshots[vmID] = append(wa.snapshots[vmID], snapshot)
			wa.snapshotMutex.Unlock()
		}
	}
}

// analyzeWorkloads analyzes workload patterns for all registered VMs
func (wa *WorkloadAnalyzer) analyzeWorkloads() {
	wa.snapshotMutex.RLock()
	vmIDs := make([]string, 0, len(wa.snapshots))
	for vmID := range wa.snapshots {
		vmIDs = append(vmIDs, vmID)
	}
	wa.snapshotMutex.RUnlock()

	for _, vmID := range vmIDs {
		wa.analyzeVM(vmID)
	}
}

// analyzeVM analyzes workload patterns for a specific VM
func (wa *WorkloadAnalyzer) analyzeVM(vmID string) {
	// Get snapshots for this VM
	wa.snapshotMutex.RLock()
	snapshots, exists := wa.snapshots[vmID]
	wa.snapshotMutex.RUnlock()

	if !exists || len(snapshots) < wa.config.MinSamplesForProfiling {
		return
	}

	// Create or get existing profile
	wa.profileMutex.Lock()
	profile, exists := wa.profiles[vmID]
	if !exists {
		profile = &WorkloadProfile{
			VMID:                  vmID,
			DominantWorkloadType:  WorkloadTypeUnknown,
			ResourceUsagePatterns: make(map[string]*ResourceUsagePattern),
			UpdatedAt:             time.Now(),
			Confidence:            0.0,
			HistoryDuration:       0,
		}
		wa.profiles[vmID] = profile
	}
	wa.profileMutex.Unlock()

	// Build resource usage patterns
	resourcePatterns := make(map[string]*ResourceUsagePattern)

	// Collect all metric types
	metricTypes := make(map[string]bool)
	for _, snapshot := range snapshots {
		for metric := range snapshot.Metrics {
			metricTypes[metric] = true
		}
	}

	// Analyze each metric
	for metric := range metricTypes {
		// Create a pattern for this metric
		pattern := &ResourceUsagePattern{
			ResourceType: metric,
			UsagePattern: make(map[time.Time]float64),
		}

		// Collect usage data
		var total float64
		var peak float64
		var count int

		for _, snapshot := range snapshots {
			if value, exists := snapshot.Metrics[metric]; exists {
				pattern.UsagePattern[snapshot.Timestamp] = value
				total += value
				count++
				if value > peak {
					peak = value
				}
			}
		}

		// Calculate averages
		if count > 0 {
			pattern.AverageUsage = total / float64(count)
			pattern.PeakUsage = peak

			// Calculate variability
			var varianceSum float64
			for _, value := range pattern.UsagePattern {
				diff := value - pattern.AverageUsage
				varianceSum += diff * diff
			}
			variance := varianceSum / float64(count)
			pattern.VariabilityScore = variance / (pattern.AverageUsage*pattern.AverageUsage + 1.0)

			// Simple prediction (just use the average for now)
			pattern.PredictedUsage = pattern.AverageUsage
		}

		resourcePatterns[metric] = pattern
	}

	// Update the profile
	wa.profileMutex.Lock()
	defer wa.profileMutex.Unlock()

	profile.ResourceUsagePatterns = resourcePatterns
	profile.UpdatedAt = time.Now()
	profile.HistoryDuration = time.Since(snapshots[0].Timestamp)
	profile.Confidence = calculateConfidence(profile.HistoryDuration, wa.config.HistoryRetention, len(snapshots))

	// Determine workload type
	profile.DominantWorkloadType = wa.determineWorkloadType(resourcePatterns)

	log.Printf("Updated workload profile for VM %s: %s (confidence: %.2f)",
		vmID, profile.DominantWorkloadType, profile.Confidence)
}

// determineWorkloadType determines the dominant workload type based on resource patterns
func (wa *WorkloadAnalyzer) determineWorkloadType(patterns map[string]*ResourceUsagePattern) WorkloadType {
	var cpuScore, memoryScore, ioScore, networkScore float64

	// Extract scores for each resource type
	for metric, pattern := range patterns {
		switch metric {
		case "cpu_usage":
			cpuScore = pattern.AverageUsage
		case "memory_usage":
			memoryScore = pattern.AverageUsage
		case "disk_io":
			ioScore = pattern.AverageUsage
		case "network_io":
			networkScore = pattern.AverageUsage
		}
	}

	// Simple scoring to determine workload type
	// In a real system, this would be much more sophisticated

	if cpuScore >= wa.config.CPUIntensiveThreshold && cpuScore > memoryScore && cpuScore > ioScore && cpuScore > networkScore {
		return WorkloadTypeCPUIntensive
	} else if memoryScore >= wa.config.MemoryIntensiveThreshold && memoryScore > cpuScore && memoryScore > ioScore && memoryScore > networkScore {
		return WorkloadTypeMemoryIntensive
	} else if ioScore >= wa.config.IOIntensiveThreshold && ioScore > cpuScore && ioScore > memoryScore && ioScore > networkScore {
		return WorkloadTypeIOIntensive
	} else if networkScore >= wa.config.NetworkIntensiveThreshold && networkScore > cpuScore && networkScore > memoryScore && networkScore > ioScore {
		return WorkloadTypeNetworkIntensive
	}

	return WorkloadTypeBalanced
}

// cleanupOldData removes snapshots older than the retention period
func (wa *WorkloadAnalyzer) cleanupOldData() {
	cutoff := time.Now().Add(-wa.config.HistoryRetention)

	wa.snapshotMutex.Lock()
	defer wa.snapshotMutex.Unlock()

	for vmID, snapshots := range wa.snapshots {
		newSnapshots := make([]ResourceSnapshot, 0)
		for _, snapshot := range snapshots {
			if snapshot.Timestamp.After(cutoff) {
				newSnapshots = append(newSnapshots, snapshot)
			}
		}
		wa.snapshots[vmID] = newSnapshots
	}
}

// calculateConfidence calculates a confidence score for a profile
func calculateConfidence(historyDuration, targetDuration time.Duration, sampleCount int) float64 {
	// More history and more samples = higher confidence
	durationFactor := float64(historyDuration) / float64(targetDuration)
	if durationFactor > 1.0 {
		durationFactor = 1.0
	}

	sampleFactor := float64(sampleCount) / 100.0
	if sampleFactor > 1.0 {
		sampleFactor = 1.0
	}

	// Combined confidence score
	return durationFactor*0.6 + sampleFactor*0.4
}
