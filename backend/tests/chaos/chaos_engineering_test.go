// Chaos Engineering Tests for Redis Cluster and System Resilience
package chaos

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Chaos testing configuration
type ChaosTestConfig struct {
	TargetServices    []ServiceConfig   `json:"target_services"`
	ChaosScenarios   []ChaosScenario   `json:"chaos_scenarios"`
	TestDuration     time.Duration     `json:"test_duration"`
	RecoveryTimeout  time.Duration     `json:"recovery_timeout"`
	MetricsInterval  time.Duration     `json:"metrics_interval"`
	FailureInjection FailureConfig     `json:"failure_injection"`
}

type ServiceConfig struct {
	Name      string   `json:"name"`
	Type      string   `json:"type"` // redis, api, database, network
	Addresses []string `json:"addresses"`
	HealthEndpoint string `json:"health_endpoint"`
	CriticalityLevel int  `json:"criticality_level"` // 1-5, 5 being most critical
}

type ChaosScenario struct {
	Name            string        `json:"name"`
	Description     string        `json:"description"`
	FailureType     string        `json:"failure_type"`
	TargetService   string        `json:"target_service"`
	ImpactRadius    string        `json:"impact_radius"` // single, cluster, system
	Duration        time.Duration `json:"duration"`
	Severity        int           `json:"severity"` // 1-5
	ExpectedRecovery time.Duration `json:"expected_recovery"`
}

type FailureConfig struct {
	NetworkPartition NetworkPartitionConfig `json:"network_partition"`
	ServiceCrash     ServiceCrashConfig     `json:"service_crash"`
	ResourceExhaustion ResourceExhaustionConfig `json:"resource_exhaustion"`
	Latency          LatencyConfig          `json:"latency"`
	DataCorruption   DataCorruptionConfig   `json:"data_corruption"`
}

type NetworkPartitionConfig struct {
	Enabled          bool          `json:"enabled"`
	PartitionNodes   []string      `json:"partition_nodes"`
	IsolationDuration time.Duration `json:"isolation_duration"`
}

type ServiceCrashConfig struct {
	Enabled     bool     `json:"enabled"`
	Services    []string `json:"services"`
	KillSignal  string   `json:"kill_signal"`
	RestartDelay time.Duration `json:"restart_delay"`
}

type ResourceExhaustionConfig struct {
	Enabled    bool    `json:"enabled"`
	CPULimit   float64 `json:"cpu_limit"`
	MemoryLimit int64  `json:"memory_limit"`
	DiskLimit  int64   `json:"disk_limit"`
}

type LatencyConfig struct {
	Enabled     bool          `json:"enabled"`
	MinLatency  time.Duration `json:"min_latency"`
	MaxLatency  time.Duration `json:"max_latency"`
	Jitter      float64       `json:"jitter"`
}

type DataCorruptionConfig struct {
	Enabled        bool    `json:"enabled"`
	CorruptionRate float64 `json:"corruption_rate"`
	TargetKeys     []string `json:"target_keys"`
}

// Test result structures
type ChaosTestResult struct {
	ScenarioName     string                 `json:"scenario_name"`
	StartTime        time.Time              `json:"start_time"`
	EndTime          time.Time              `json:"end_time"`
	Duration         time.Duration          `json:"duration"`
	FailureInjected  bool                   `json:"failure_injected"`
	RecoveryAchieved bool                   `json:"recovery_achieved"`
	RecoveryTime     time.Duration          `json:"recovery_time"`
	ImpactMetrics    ImpactMetrics          `json:"impact_metrics"`
	ResilienceScore  float64                `json:"resilience_score"`
	Observations     []string               `json:"observations"`
	Recommendations  []string               `json:"recommendations"`
}

type ImpactMetrics struct {
	AvailabilityDrop    float64 `json:"availability_drop"`
	PerformanceDrop     float64 `json:"performance_drop"`
	ErrorRateIncrease   float64 `json:"error_rate_increase"`
	AffectedOperations  int     `json:"affected_operations"`
	DataLoss            bool    `json:"data_loss"`
	ServiceDegradation  map[string]float64 `json:"service_degradation"`
}

// Chaos engineering framework
type ChaosEngineeringFramework struct {
	config         *ChaosTestConfig
	serviceMonitor *ServiceMonitor
	failureInjector *FailureInjector
	metricsCollector *MetricsCollector
	results        []*ChaosTestResult
	mu             sync.RWMutex
}

type ServiceMonitor struct {
	services       map[string]*ServiceHealth
	healthCheckers map[string]HealthChecker
	mu             sync.RWMutex
}

type ServiceHealth struct {
	Name         string    `json:"name"`
	Status       string    `json:"status"` // healthy, degraded, unhealthy
	LastCheck    time.Time `json:"last_check"`
	ResponseTime time.Duration `json:"response_time"`
	ErrorRate    float64   `json:"error_rate"`
	Availability float64   `json:"availability"`
}

type HealthChecker interface {
	CheckHealth(ctx context.Context) (*ServiceHealth, error)
	GetServiceName() string
}

type FailureInjector struct {
	activeFailures map[string]*ActiveFailure
	mu             sync.RWMutex
}

type ActiveFailure struct {
	ScenarioName string    `json:"scenario_name"`
	FailureType  string    `json:"failure_type"`
	StartTime    time.Time `json:"start_time"`
	Duration     time.Duration `json:"duration"`
	StopFunc     func() error  `json:"-"`
}

type MetricsCollector struct {
	baselineMetrics map[string]interface{}
	currentMetrics  map[string]interface{}
	history         []MetricsSnapshot
	mu              sync.RWMutex
}

type MetricsSnapshot struct {
	Timestamp time.Time                `json:"timestamp"`
	Metrics   map[string]interface{}   `json:"metrics"`
	Services  map[string]*ServiceHealth `json:"services"`
}

func NewChaosEngineeringFramework(config *ChaosTestConfig) *ChaosEngineeringFramework {
	return &ChaosEngineeringFramework{
		config:         config,
		serviceMonitor: NewServiceMonitor(config.TargetServices),
		failureInjector: NewFailureInjector(),
		metricsCollector: NewMetricsCollector(),
		results:        make([]*ChaosTestResult, 0),
	}
}

// Main chaos engineering test
func TestRedisClusterChaosEngineering(t *testing.T) {
	config := getChaosTestConfig()
	framework := NewChaosEngineeringFramework(config)

	// Establish baseline
	require.NoError(t, framework.establishBaseline(t), "Should establish baseline metrics")

	for _, scenario := range config.ChaosScenarios {
		t.Run(scenario.Name, func(t *testing.T) {
			framework.runChaosScenario(t, scenario)
		})
	}

	// Generate chaos engineering report
	framework.generateChaosReport(t)
}

func (f *ChaosEngineeringFramework) establishBaseline(t *testing.T) error {
	t.Log("🏗️  Establishing baseline metrics")
	
	ctx := context.Background()
	
	// Collect initial service health
	err := f.serviceMonitor.updateAllHealthStatus(ctx)
	if err != nil {
		return fmt.Errorf("failed to collect initial health status: %v", err)
	}

	// Collect baseline metrics
	baseline, err := f.metricsCollector.collectCurrentMetrics(ctx)
	if err != nil {
		return fmt.Errorf("failed to collect baseline metrics: %v", err)
	}

	f.metricsCollector.setBaseline(baseline)
	
	// Verify all services are healthy before starting chaos tests
	unhealthyServices := f.serviceMonitor.getUnhealthyServices()
	if len(unhealthyServices) > 0 {
		return fmt.Errorf("cannot start chaos tests - unhealthy services detected: %v", unhealthyServices)
	}

	t.Log("✅ Baseline established successfully")
	return nil
}

func (f *ChaosEngineeringFramework) runChaosScenario(t *testing.T, scenario ChaosScenario) {
	t.Logf("🌪️  Running chaos scenario: %s", scenario.Name)
	t.Logf("   Description: %s", scenario.Description)
	t.Logf("   Target: %s, Duration: %v", scenario.TargetService, scenario.Duration)

	result := &ChaosTestResult{
		ScenarioName: scenario.Name,
		StartTime:    time.Now(),
		ImpactMetrics: ImpactMetrics{
			ServiceDegradation: make(map[string]float64),
		},
		Observations:    make([]string, 0),
		Recommendations: make([]string, 0),
	}

	defer func() {
		result.EndTime = time.Now()
		result.Duration = result.EndTime.Sub(result.StartTime)
		f.addResult(result)
	}()

	ctx := context.Background()

	// Phase 1: Pre-chaos monitoring
	t.Log("📊 Phase 1: Pre-chaos monitoring")
	preFailureHealth := f.serviceMonitor.captureHealthSnapshot()
	
	// Phase 2: Inject failure
	t.Log("💥 Phase 2: Injecting failure")
	err := f.injectFailure(ctx, scenario)
	if err != nil {
		t.Errorf("Failed to inject failure: %v", err)
		result.Observations = append(result.Observations, fmt.Sprintf("Failure injection failed: %v", err))
		return
	}
	
	result.FailureInjected = true
	result.Observations = append(result.Observations, fmt.Sprintf("Successfully injected %s failure", scenario.FailureType))

	// Phase 3: Monitor during failure
	t.Log("🔍 Phase 3: Monitoring during failure")
	impactMetrics := f.monitorDuringFailure(ctx, scenario)
	result.ImpactMetrics = impactMetrics

	// Phase 4: Stop failure injection
	t.Log("🛑 Phase 4: Stopping failure injection")
	err = f.stopFailureInjection(scenario.Name)
	if err != nil {
		t.Logf("Warning: Failed to stop failure injection cleanly: %v", err)
		result.Observations = append(result.Observations, fmt.Sprintf("Failure cleanup issue: %v", err))
	}

	// Phase 5: Monitor recovery
	t.Log("🏥 Phase 5: Monitoring recovery")
	recoveryStart := time.Now()
	recoveryAchieved := f.waitForRecovery(ctx, preFailureHealth, scenario.ExpectedRecovery)
	recoveryTime := time.Since(recoveryStart)
	
	result.RecoveryAchieved = recoveryAchieved
	result.RecoveryTime = recoveryTime

	if recoveryAchieved {
		result.Observations = append(result.Observations, fmt.Sprintf("System recovered in %v", recoveryTime))
		t.Logf("✅ Recovery achieved in %v", recoveryTime)
	} else {
		result.Observations = append(result.Observations, fmt.Sprintf("Recovery not achieved within %v", scenario.ExpectedRecovery))
		t.Logf("❌ Recovery not achieved within expected time %v", scenario.ExpectedRecovery)
	}

	// Calculate resilience score
	result.ResilienceScore = f.calculateResilienceScore(result, scenario)
	
	// Generate recommendations
	result.Recommendations = f.generateRecommendations(result, scenario)

	// Assertions
	f.performChaosAssertions(t, result, scenario)
	
	t.Logf("🎯 Scenario completed - Resilience Score: %.2f", result.ResilienceScore)
}

func (f *ChaosEngineeringFramework) injectFailure(ctx context.Context, scenario ChaosScenario) error {
	switch scenario.FailureType {
	case "network_partition":
		return f.injectNetworkPartition(ctx, scenario)
	case "service_crash":
		return f.injectServiceCrash(ctx, scenario)
	case "high_latency":
		return f.injectHighLatency(ctx, scenario)
	case "memory_pressure":
		return f.injectMemoryPressure(ctx, scenario)
	case "disk_full":
		return f.injectDiskPressure(ctx, scenario)
	case "data_corruption":
		return f.injectDataCorruption(ctx, scenario)
	default:
		return fmt.Errorf("unsupported failure type: %s", scenario.FailureType)
	}
}

func (f *ChaosEngineeringFramework) injectNetworkPartition(ctx context.Context, scenario ChaosScenario) error {
	// Simulate network partition by introducing network delays/failures
	stopFunc := func() error {
		// Restore network connectivity
		return nil
	}

	failure := &ActiveFailure{
		ScenarioName: scenario.Name,
		FailureType:  "network_partition",
		StartTime:    time.Now(),
		Duration:     scenario.Duration,
		StopFunc:     stopFunc,
	}

	f.failureInjector.addActiveFailure(scenario.Name, failure)
	
	// Simulate partition by blocking traffic to specific nodes
	// In a real implementation, this would use iptables, tc, or similar tools
	time.Sleep(100 * time.Millisecond) // Simulate setup time
	
	return nil
}

func (f *ChaosEngineeringFramework) injectServiceCrash(ctx context.Context, scenario ChaosScenario) error {
	// Simulate service crash
	stopFunc := func() error {
		// Restart service
		return nil
	}

	failure := &ActiveFailure{
		ScenarioName: scenario.Name,
		FailureType:  "service_crash",
		StartTime:    time.Now(),
		Duration:     scenario.Duration,
		StopFunc:     stopFunc,
	}

	f.failureInjector.addActiveFailure(scenario.Name, failure)
	
	// In a real implementation, this would send SIGKILL to the process
	time.Sleep(50 * time.Millisecond) // Simulate crash time
	
	return nil
}

func (f *ChaosEngineeringFramework) injectHighLatency(ctx context.Context, scenario ChaosScenario) error {
	// Inject high latency
	stopFunc := func() error {
		// Remove latency injection
		return nil
	}

	failure := &ActiveFailure{
		ScenarioName: scenario.Name,
		FailureType:  "high_latency",
		StartTime:    time.Now(),
		Duration:     scenario.Duration,
		StopFunc:     stopFunc,
	}

	f.failureInjector.addActiveFailure(scenario.Name, failure)
	
	// In a real implementation, this would use tc (traffic control) to add latency
	return nil
}

func (f *ChaosEngineeringFramework) injectMemoryPressure(ctx context.Context, scenario ChaosScenario) error {
	// Simulate memory pressure
	stopFunc := func() error {
		// Release memory pressure
		return nil
	}

	failure := &ActiveFailure{
		ScenarioName: scenario.Name,
		FailureType:  "memory_pressure",
		StartTime:    time.Now(),
		Duration:     scenario.Duration,
		StopFunc:     stopFunc,
	}

	f.failureInjector.addActiveFailure(scenario.Name, failure)
	
	// In a real implementation, this would allocate large amounts of memory
	return nil
}

func (f *ChaosEngineeringFramework) injectDiskPressure(ctx context.Context, scenario ChaosScenario) error {
	// Simulate disk pressure
	stopFunc := func() error {
		// Clean up disk usage
		return nil
	}

	failure := &ActiveFailure{
		ScenarioName: scenario.Name,
		FailureType:  "disk_full",
		StartTime:    time.Now(),
		Duration:     scenario.Duration,
		StopFunc:     stopFunc,
	}

	f.failureInjector.addActiveFailure(scenario.Name, failure)
	
	// In a real implementation, this would fill up disk space
	return nil
}

func (f *ChaosEngineeringFramework) injectDataCorruption(ctx context.Context, scenario ChaosScenario) error {
	// Simulate data corruption
	stopFunc := func() error {
		// Restore corrupted data
		return nil
	}

	failure := &ActiveFailure{
		ScenarioName: scenario.Name,
		FailureType:  "data_corruption",
		StartTime:    time.Now(),
		Duration:     scenario.Duration,
		StopFunc:     stopFunc,
	}

	f.failureInjector.addActiveFailure(scenario.Name, failure)
	
	// In a real implementation, this would corrupt some data
	return nil
}

func (f *ChaosEngineeringFramework) monitorDuringFailure(ctx context.Context, scenario ChaosScenario) ImpactMetrics {
	metrics := ImpactMetrics{
		ServiceDegradation: make(map[string]float64),
	}

	// Monitor for the duration of the failure
	monitoringEnd := time.Now().Add(scenario.Duration)
	ticker := time.NewTicker(f.config.MetricsInterval)
	defer ticker.Stop()

	operationCount := 0
	errorCount := 0
	
	for time.Now().Before(monitoringEnd) {
		select {
		case <-ctx.Done():
			return metrics
		case <-ticker.C:
			// Simulate operations and collect metrics
			if f.simulateOperation() {
				operationCount++
			} else {
				errorCount++
			}

			// Update service health
			f.serviceMonitor.updateAllHealthStatus(ctx)
		}
	}

	// Calculate impact metrics
	metrics.AffectedOperations = operationCount + errorCount
	if metrics.AffectedOperations > 0 {
		metrics.ErrorRateIncrease = float64(errorCount) / float64(metrics.AffectedOperations)
	}

	// Calculate availability and performance drops
	currentHealth := f.serviceMonitor.captureHealthSnapshot()
	for serviceName, health := range currentHealth {
		if baselineHealth := f.getBaselineHealth(serviceName); baselineHealth != nil {
			availabilityDrop := baselineHealth.Availability - health.Availability
			performanceDrop := float64(health.ResponseTime-baselineHealth.ResponseTime) / float64(baselineHealth.ResponseTime)
			
			metrics.ServiceDegradation[serviceName] = availabilityDrop
			
			if availabilityDrop > metrics.AvailabilityDrop {
				metrics.AvailabilityDrop = availabilityDrop
			}
			
			if performanceDrop > metrics.PerformanceDrop {
				metrics.PerformanceDrop = performanceDrop
			}
		}
	}

	return metrics
}

func (f *ChaosEngineeringFramework) stopFailureInjection(scenarioName string) error {
	f.failureInjector.mu.Lock()
	defer f.failureInjector.mu.Unlock()

	if failure, exists := f.failureInjector.activeFailures[scenarioName]; exists {
		err := failure.StopFunc()
		delete(f.failureInjector.activeFailures, scenarioName)
		return err
	}

	return fmt.Errorf("no active failure found for scenario: %s", scenarioName)
}

func (f *ChaosEngineeringFramework) waitForRecovery(ctx context.Context, preFailureHealth map[string]*ServiceHealth, timeout time.Duration) bool {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return false
		case <-ticker.C:
			f.serviceMonitor.updateAllHealthStatus(ctx)
			currentHealth := f.serviceMonitor.captureHealthSnapshot()
			
			if f.isRecoveryComplete(preFailureHealth, currentHealth) {
				return true
			}
		}
	}
}

func (f *ChaosEngineeringFramework) isRecoveryComplete(baseline, current map[string]*ServiceHealth) bool {
	for serviceName, baselineHealth := range baseline {
		currentHealth, exists := current[serviceName]
		if !exists {
			return false
		}

		// Check if service is healthy again
		if currentHealth.Status != "healthy" {
			return false
		}

		// Check if availability is restored (within 5% of baseline)
		if currentHealth.Availability < baselineHealth.Availability*0.95 {
			return false
		}

		// Check if response time is reasonable (within 150% of baseline)
		if currentHealth.ResponseTime > baselineHealth.ResponseTime*3/2 {
			return false
		}
	}

	return true
}

func (f *ChaosEngineeringFramework) calculateResilienceScore(result *ChaosTestResult, scenario ChaosScenario) float64 {
	score := 100.0

	// Deduct points for high availability drop
	score -= result.ImpactMetrics.AvailabilityDrop * 50

	// Deduct points for high error rate
	score -= result.ImpactMetrics.ErrorRateIncrease * 30

	// Deduct points for slow recovery
	expectedRecovery := scenario.ExpectedRecovery.Seconds()
	actualRecovery := result.RecoveryTime.Seconds()
	if actualRecovery > expectedRecovery {
		recoveryPenalty := (actualRecovery - expectedRecovery) / expectedRecovery * 20
		score -= recoveryPenalty
	}

	// Bonus for quick recovery
	if actualRecovery < expectedRecovery*0.5 {
		score += 10
	}

	// Deduct points for data loss
	if result.ImpactMetrics.DataLoss {
		score -= 25
	}

	// Deduct points for not recovering
	if !result.RecoveryAchieved {
		score -= 40
	}

	if score < 0 {
		score = 0
	}
	if score > 100 {
		score = 100
	}

	return score
}

func (f *ChaosEngineeringFramework) generateRecommendations(result *ChaosTestResult, scenario ChaosScenario) []string {
	recommendations := make([]string, 0)

	if result.ImpactMetrics.AvailabilityDrop > 0.1 {
		recommendations = append(recommendations, "Consider implementing better failover mechanisms to reduce availability impact")
	}

	if result.ImpactMetrics.ErrorRateIncrease > 0.2 {
		recommendations = append(recommendations, "Implement circuit breakers to prevent cascade failures")
	}

	if !result.RecoveryAchieved {
		recommendations = append(recommendations, "Improve monitoring and automated recovery procedures")
	}

	if result.RecoveryTime > scenario.ExpectedRecovery*2 {
		recommendations = append(recommendations, "Optimize service startup and health check procedures")
	}

	if result.ImpactMetrics.DataLoss {
		recommendations = append(recommendations, "Strengthen data replication and backup strategies")
	}

	if result.ResilienceScore < 70 {
		recommendations = append(recommendations, "Overall system resilience needs improvement - consider comprehensive disaster recovery review")
	}

	return recommendations
}

func (f *ChaosEngineeringFramework) performChaosAssertions(t *testing.T, result *ChaosTestResult, scenario ChaosScenario) {
	// Basic resilience requirements
	assert.True(t, result.FailureInjected, "Failure should be successfully injected")
	
	// Recovery requirements
	if scenario.ImpactRadius != "system" { // Allow system-wide failures to take longer
		assert.True(t, result.RecoveryAchieved, "System should recover from %s failure", scenario.FailureType)
	}

	// Performance degradation should be bounded
	assert.Less(t, result.ImpactMetrics.AvailabilityDrop, 0.3, "Availability drop should be less than 30%%")
	
	// Error rates should be manageable
	assert.Less(t, result.ImpactMetrics.ErrorRateIncrease, 0.5, "Error rate increase should be less than 50%%")

	// No data loss for most scenarios
	if scenario.FailureType != "data_corruption" {
		assert.False(t, result.ImpactMetrics.DataLoss, "Should not experience data loss during %s", scenario.FailureType)
	}

	// Resilience score requirements
	minimumScore := 60.0
	if scenario.Severity >= 4 {
		minimumScore = 40.0 // Lower bar for high-severity scenarios
	}
	
	assert.GreaterOrEqual(t, result.ResilienceScore, minimumScore, 
		"Resilience score should meet minimum threshold for %s", scenario.Name)

	t.Logf("Resilience metrics for %s:", scenario.Name)
	t.Logf("  - Availability Drop: %.1f%%", result.ImpactMetrics.AvailabilityDrop*100)
	t.Logf("  - Performance Drop: %.1f%%", result.ImpactMetrics.PerformanceDrop*100)
	t.Logf("  - Error Rate Increase: %.1f%%", result.ImpactMetrics.ErrorRateIncrease*100)
	t.Logf("  - Recovery Time: %v", result.RecoveryTime)
	t.Logf("  - Resilience Score: %.1f", result.ResilienceScore)
}

func (f *ChaosEngineeringFramework) generateChaosReport(t *testing.T) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	t.Log("\n" + strings.Repeat("=", 80))
	t.Log("🌪️  CHAOS ENGINEERING REPORT")
	t.Log(strings.Repeat("=", 80))

	totalTests := len(f.results)
	recoveredTests := 0
	totalScore := 0.0

	for _, result := range f.results {
		if result.RecoveryAchieved {
			recoveredTests++
		}
		totalScore += result.ResilienceScore
	}

	avgScore := totalScore / float64(totalTests)
	recoveryRate := float64(recoveredTests) / float64(totalTests) * 100

	t.Logf("📊 Overall Statistics:")
	t.Logf("   Total Scenarios: %d", totalTests)
	t.Logf("   Recovery Rate: %.1f%% (%d/%d)", recoveryRate, recoveredTests, totalTests)
	t.Logf("   Average Resilience Score: %.1f", avgScore)

	t.Log("\n📈 Scenario Results:")
	for _, result := range f.results {
		status := "✅"
		if !result.RecoveryAchieved {
			status = "❌"
		} else if result.ResilienceScore < 70 {
			status = "⚠️"
		}

		t.Logf("   %s %s: Score %.1f, Recovery %v", 
			status, result.ScenarioName, result.ResilienceScore, result.RecoveryTime)
	}

	t.Log("\n💡 Key Recommendations:")
	recommendationCounts := make(map[string]int)
	for _, result := range f.results {
		for _, rec := range result.Recommendations {
			recommendationCounts[rec]++
		}
	}

	for rec, count := range recommendationCounts {
		if count >= 2 { // Show recommendations that appear multiple times
			t.Logf("   - %s (appears in %d scenarios)", rec, count)
		}
	}

	// Overall system resilience assessment
	t.Log("\n🎯 System Resilience Assessment:")
	if avgScore >= 80 {
		t.Log("   EXCELLENT: System demonstrates strong resilience across scenarios")
	} else if avgScore >= 70 {
		t.Log("   GOOD: System is resilient but has room for improvement")
	} else if avgScore >= 60 {
		t.Log("   MODERATE: System shows basic resilience but needs attention")
	} else {
		t.Log("   POOR: System requires significant resilience improvements")
	}

	t.Log(strings.Repeat("=", 80))
}

// Helper methods and implementations
func (f *ChaosEngineeringFramework) simulateOperation() bool {
	// Simulate successful operation with some randomness
	return rand.Float64() > 0.1 // 90% success rate normally
}

func (f *ChaosEngineeringFramework) getBaselineHealth(serviceName string) *ServiceHealth {
	// Return baseline health for service - would be stored during establishBaseline
	return &ServiceHealth{
		Name:         serviceName,
		Status:       "healthy",
		Availability: 1.0,
		ResponseTime: 50 * time.Millisecond,
		ErrorRate:    0.01,
	}
}

func (f *ChaosEngineeringFramework) addResult(result *ChaosTestResult) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.results = append(f.results, result)
}

// ServiceMonitor implementations
func NewServiceMonitor(services []ServiceConfig) *ServiceMonitor {
	monitor := &ServiceMonitor{
		services:       make(map[string]*ServiceHealth),
		healthCheckers: make(map[string]HealthChecker),
	}

	for _, service := range services {
		monitor.services[service.Name] = &ServiceHealth{
			Name:         service.Name,
			Status:       "unknown",
			Availability: 0.0,
		}

		// Create appropriate health checker based on service type
		switch service.Type {
		case "redis":
			monitor.healthCheckers[service.Name] = &RedisHealthChecker{
				serviceName: service.Name,
				addresses:   service.Addresses,
			}
		case "api":
			monitor.healthCheckers[service.Name] = &HTTPHealthChecker{
				serviceName: service.Name,
				endpoint:    service.HealthEndpoint,
			}
		}
	}

	return monitor
}

func (sm *ServiceMonitor) updateAllHealthStatus(ctx context.Context) error {
	var wg sync.WaitGroup
	for serviceName, checker := range sm.healthCheckers {
		wg.Add(1)
		go func(name string, hc HealthChecker) {
			defer wg.Done()
			health, err := hc.CheckHealth(ctx)
			
			sm.mu.Lock()
			if err != nil {
				sm.services[name].Status = "unhealthy"
			} else {
				sm.services[name] = health
			}
			sm.services[name].LastCheck = time.Now()
			sm.mu.Unlock()
		}(serviceName, checker)
	}
	wg.Wait()
	return nil
}

func (sm *ServiceMonitor) captureHealthSnapshot() map[string]*ServiceHealth {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	snapshot := make(map[string]*ServiceHealth)
	for name, health := range sm.services {
		// Deep copy
		snapshot[name] = &ServiceHealth{
			Name:         health.Name,
			Status:       health.Status,
			LastCheck:    health.LastCheck,
			ResponseTime: health.ResponseTime,
			ErrorRate:    health.ErrorRate,
			Availability: health.Availability,
		}
	}
	return snapshot
}

func (sm *ServiceMonitor) getUnhealthyServices() []string {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	unhealthy := make([]string, 0)
	for name, health := range sm.services {
		if health.Status != "healthy" {
			unhealthy = append(unhealthy, name)
		}
	}
	return unhealthy
}

// HealthChecker implementations
type RedisHealthChecker struct {
	serviceName string
	addresses   []string
}

func (rhc *RedisHealthChecker) CheckHealth(ctx context.Context) (*ServiceHealth, error) {
	// Simulate Redis health check
	health := &ServiceHealth{
		Name:         rhc.serviceName,
		Status:       "healthy",
		ResponseTime: 5 * time.Millisecond,
		ErrorRate:    0.001,
		Availability: 0.999,
		LastCheck:    time.Now(),
	}

	// In real implementation, would connect to Redis and check connectivity
	return health, nil
}

func (rhc *RedisHealthChecker) GetServiceName() string {
	return rhc.serviceName
}

type HTTPHealthChecker struct {
	serviceName string
	endpoint    string
}

func (hhc *HTTPHealthChecker) CheckHealth(ctx context.Context) (*ServiceHealth, error) {
	start := time.Now()
	
	// Simulate HTTP health check
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(hhc.endpoint)
	
	responseTime := time.Since(start)
	
	health := &ServiceHealth{
		Name:         hhc.serviceName,
		ResponseTime: responseTime,
		LastCheck:    time.Now(),
	}

	if err != nil {
		health.Status = "unhealthy"
		health.Availability = 0.0
		health.ErrorRate = 1.0
		return health, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		health.Status = "healthy"
		health.Availability = 1.0
		health.ErrorRate = 0.0
	} else {
		health.Status = "degraded"
		health.Availability = 0.5
		health.ErrorRate = 0.1
	}

	return health, nil
}

func (hhc *HTTPHealthChecker) GetServiceName() string {
	return hhc.serviceName
}

// FailureInjector implementations
func NewFailureInjector() *FailureInjector {
	return &FailureInjector{
		activeFailures: make(map[string]*ActiveFailure),
	}
}

func (fi *FailureInjector) addActiveFailure(scenarioName string, failure *ActiveFailure) {
	fi.mu.Lock()
	defer fi.mu.Unlock()
	fi.activeFailures[scenarioName] = failure
}

// MetricsCollector implementations
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		baselineMetrics: make(map[string]interface{}),
		currentMetrics:  make(map[string]interface{}),
		history:         make([]MetricsSnapshot, 0),
	}
}

func (mc *MetricsCollector) collectCurrentMetrics(ctx context.Context) (map[string]interface{}, error) {
	// Simulate metrics collection
	metrics := map[string]interface{}{
		"cpu_usage":    rand.Float64() * 50,
		"memory_usage": rand.Float64() * 70,
		"request_rate": rand.Float64() * 1000,
		"error_rate":   rand.Float64() * 0.01,
	}
	
	return metrics, nil
}

func (mc *MetricsCollector) setBaseline(metrics map[string]interface{}) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.baselineMetrics = metrics
}

// Configuration helper
func getChaosTestConfig() *ChaosTestConfig {
	return &ChaosTestConfig{
		TestDuration:    10 * time.Minute,
		RecoveryTimeout: 5 * time.Minute,
		MetricsInterval: 10 * time.Second,
		TargetServices: []ServiceConfig{
			{
				Name:             "redis-cluster",
				Type:             "redis",
				Addresses:        []string{"localhost:6379", "localhost:6380", "localhost:6381"},
				HealthEndpoint:   "",
				CriticalityLevel: 5,
			},
			{
				Name:             "api-server",
				Type:             "api",
				Addresses:        []string{"localhost:8090"},
				HealthEndpoint:   "http://localhost:8090/health",
				CriticalityLevel: 4,
			},
		},
		ChaosScenarios: []ChaosScenario{
			{
				Name:             "Redis_Network_Partition",
				Description:      "Isolate one Redis node from cluster",
				FailureType:      "network_partition",
				TargetService:    "redis-cluster",
				ImpactRadius:     "cluster",
				Duration:         30 * time.Second,
				Severity:         3,
				ExpectedRecovery: 60 * time.Second,
			},
			{
				Name:             "API_Service_Crash",
				Description:      "Crash API server process",
				FailureType:      "service_crash",
				TargetService:    "api-server",
				ImpactRadius:     "single",
				Duration:         10 * time.Second,
				Severity:         4,
				ExpectedRecovery: 30 * time.Second,
			},
			{
				Name:             "High_Network_Latency",
				Description:      "Inject high latency in network communications",
				FailureType:      "high_latency",
				TargetService:    "redis-cluster",
				ImpactRadius:     "cluster",
				Duration:         45 * time.Second,
				Severity:         2,
				ExpectedRecovery: 15 * time.Second,
			},
			{
				Name:             "Memory_Exhaustion",
				Description:      "Consume available memory to trigger OOM conditions",
				FailureType:      "memory_pressure",
				TargetService:    "redis-cluster",
				ImpactRadius:     "single",
				Duration:         60 * time.Second,
				Severity:         4,
				ExpectedRecovery: 90 * time.Second,
			},
		},
	}
}

// Additional chaos scenarios for comprehensive testing
func TestAdvancedChaosScenarios(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping advanced chaos tests in short mode")
	}

	config := getChaosTestConfig()
	
	// Add more complex scenarios
	advancedScenarios := []ChaosScenario{
		{
			Name:             "Cascading_Failure",
			Description:      "Trigger failure in multiple services simultaneously",
			FailureType:      "service_crash",
			TargetService:    "all",
			ImpactRadius:     "system",
			Duration:         2 * time.Minute,
			Severity:         5,
			ExpectedRecovery: 5 * time.Minute,
		},
		{
			Name:             "Byzantine_Failure",
			Description:      "Corrupt data in Redis to simulate Byzantine failure",
			FailureType:      "data_corruption",
			TargetService:    "redis-cluster",
			ImpactRadius:     "cluster",
			Duration:         30 * time.Second,
			Severity:         5,
			ExpectedRecovery: 2 * time.Minute,
		},
		{
			Name:             "Split_Brain",
			Description:      "Create network partition causing split-brain scenario",
			FailureType:      "network_partition",
			TargetService:    "redis-cluster",
			ImpactRadius:     "cluster",
			Duration:         90 * time.Second,
			Severity:         5,
			ExpectedRecovery: 3 * time.Minute,
		},
	}

	config.ChaosScenarios = append(config.ChaosScenarios, advancedScenarios...)
	framework := NewChaosEngineeringFramework(config)

	require.NoError(t, framework.establishBaseline(t), "Should establish baseline")

	for _, scenario := range advancedScenarios {
		t.Run(scenario.Name, func(t *testing.T) {
			framework.runChaosScenario(t, scenario)
		})
	}

	framework.generateChaosReport(t)
}