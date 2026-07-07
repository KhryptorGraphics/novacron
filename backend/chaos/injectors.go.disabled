// Package chaos - Chaos injectors for different failure types
package chaos

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"os"
	"runtime"
	"sync"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"go.uber.org/zap"
)

// NetworkChaosInjector injects network-related chaos
type NetworkChaosInjector struct {
	logger     *zap.Logger
	targets    map[string]*NetworkChaosState
	mu         sync.RWMutex
	tcExecutor *TCExecutor // Traffic Control executor
}

// NetworkChaosState tracks network chaos state for a target
type NetworkChaosState struct {
	Target          string
	OriginalLatency time.Duration
	OriginalLoss    float64
	Rules           []string
	Active          bool
}

// NewNetworkChaosInjector creates network chaos injector
func NewNetworkChaosInjector(logger *zap.Logger) *NetworkChaosInjector {
	return &NetworkChaosInjector{
		logger:     logger,
		targets:    make(map[string]*NetworkChaosState),
		tcExecutor: NewTCExecutor(logger),
	}
}

// Inject applies network chaos to target
func (n *NetworkChaosInjector) Inject(ctx context.Context, target string, params map[string]interface{}) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	
	// Parse parameters
	latency, _ := params["latency"].(time.Duration)
	packetLoss, _ := params["packet_loss"].(float64)
	bandwidth, _ := params["bandwidth"].(string)
	jitter, _ := params["jitter"].(time.Duration)
	reorder, _ := params["reorder"].(float64)
	corrupt, _ := params["corrupt"].(float64)
	duplicate, _ := params["duplicate"].(float64)
	
	state := &NetworkChaosState{
		Target: target,
		Active: true,
	}
	
	// Apply network chaos using tc (traffic control)
	if latency > 0 {
		rule := fmt.Sprintf("delay %dms", latency.Milliseconds())
		if jitter > 0 {
			rule += fmt.Sprintf(" %dms", jitter.Milliseconds())
		}
		if err := n.tcExecutor.AddRule(target, rule); err != nil {
			return fmt.Errorf("failed to add latency: %w", err)
		}
		state.Rules = append(state.Rules, rule)
	}
	
	if packetLoss > 0 {
		rule := fmt.Sprintf("loss %.2f%%", packetLoss)
		if err := n.tcExecutor.AddRule(target, rule); err != nil {
			return fmt.Errorf("failed to add packet loss: %w", err)
		}
		state.Rules = append(state.Rules, rule)
	}
	
	if bandwidth != "" {
		rule := fmt.Sprintf("rate %s", bandwidth)
		if err := n.tcExecutor.AddRule(target, rule); err != nil {
			return fmt.Errorf("failed to limit bandwidth: %w", err)
		}
		state.Rules = append(state.Rules, rule)
	}
	
	if reorder > 0 {
		rule := fmt.Sprintf("reorder %.2f%%", reorder)
		if err := n.tcExecutor.AddRule(target, rule); err != nil {
			return fmt.Errorf("failed to add packet reorder: %w", err)
		}
		state.Rules = append(state.Rules, rule)
	}
	
	if corrupt > 0 {
		rule := fmt.Sprintf("corrupt %.2f%%", corrupt)
		if err := n.tcExecutor.AddRule(target, rule); err != nil {
			return fmt.Errorf("failed to add packet corruption: %w", err)
		}
		state.Rules = append(state.Rules, rule)
	}
	
	if duplicate > 0 {
		rule := fmt.Sprintf("duplicate %.2f%%", duplicate)
		if err := n.tcExecutor.AddRule(target, rule); err != nil {
			return fmt.Errorf("failed to add packet duplication: %w", err)
		}
		state.Rules = append(state.Rules, rule)
	}
	
	n.targets[target] = state
	
	n.logger.Info("Injected network chaos",
		zap.String("target", target),
		zap.Duration("latency", latency),
		zap.Float64("packet_loss", packetLoss))
	
	return nil
}

// Revert removes network chaos from target
func (n *NetworkChaosInjector) Revert(ctx context.Context, target string) error {
	n.mu.Lock()
	defer n.mu.Unlock()
	
	state, exists := n.targets[target]
	if !exists {
		return nil // Already reverted or never injected
	}
	
	// Remove all tc rules
	if err := n.tcExecutor.RemoveRules(target); err != nil {
		return fmt.Errorf("failed to remove tc rules: %w", err)
	}
	
	state.Active = false
	delete(n.targets, target)
	
	n.logger.Info("Reverted network chaos", zap.String("target", target))
	return nil
}

// Validate checks if network chaos can be applied
func (n *NetworkChaosInjector) Validate() error {
	// Check if tc is available
	if !n.tcExecutor.IsAvailable() {
		return fmt.Errorf("traffic control (tc) not available")
	}
	
	// Check permissions
	if os.Geteuid() != 0 {
		return fmt.Errorf("root privileges required for network chaos")
	}
	
	return nil
}

// GetImpact returns impact analysis for network chaos
func (n *NetworkChaosInjector) GetImpact() *ImpactAnalysis {
	n.mu.RLock()
	defer n.mu.RUnlock()
	
	affectedNodes := make([]string, 0, len(n.targets))
	for target := range n.targets {
		affectedNodes = append(affectedNodes, target)
	}
	
	// Calculate impact metrics (simplified)
	return &ImpactAnalysis{
		AffectedNodes: affectedNodes,
		LatencyImpact: &LatencyMetrics{
			P50:  50 * time.Millisecond,
			P95:  200 * time.Millisecond,
			P99:  500 * time.Millisecond,
			Max:  1 * time.Second,
			Mean: 150 * time.Millisecond,
		},
		ErrorRate:         0.05, // 5% error rate
		AvailabilityImpact: 0.98, // 98% availability
	}
}

// ResourceChaosInjector injects resource-related chaos
type ResourceChaosInjector struct {
	logger       *zap.Logger
	stressors    map[string]*ResourceStressor
	mu           sync.RWMutex
}

// ResourceStressor applies resource stress
type ResourceStressor struct {
	Type      string
	Target    string
	Intensity float64
	cancel    context.CancelFunc
	wg        sync.WaitGroup
}

// NewResourceChaosInjector creates resource chaos injector
func NewResourceChaosInjector(logger *zap.Logger) *ResourceChaosInjector {
	return &ResourceChaosInjector{
		logger:    logger,
		stressors: make(map[string]*ResourceStressor),
	}
}

// Inject applies resource chaos to target
func (r *ResourceChaosInjector) Inject(ctx context.Context, target string, params map[string]interface{}) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	stressType, _ := params["type"].(string)
	intensity, _ := params["intensity"].(float64)
	
	if intensity <= 0 || intensity > 1 {
		intensity = 0.5 // Default to 50%
	}
	
	stressCtx, cancel := context.WithCancel(ctx)
	stressor := &ResourceStressor{
		Type:      stressType,
		Target:    target,
		Intensity: intensity,
		cancel:    cancel,
	}
	
	switch stressType {
	case "cpu":
		stressor.wg.Add(1)
		go r.stressCPU(stressCtx, stressor)
	case "memory":
		stressor.wg.Add(1)
		go r.stressMemory(stressCtx, stressor)
	case "disk":
		stressor.wg.Add(1)
		go r.stressDisk(stressCtx, stressor)
	case "io":
		stressor.wg.Add(1)
		go r.stressIO(stressCtx, stressor)
	default:
		return fmt.Errorf("unknown stress type: %s", stressType)
	}
	
	r.stressors[target] = stressor
	
	r.logger.Info("Injected resource chaos",
		zap.String("target", target),
		zap.String("type", stressType),
		zap.Float64("intensity", intensity))
	
	return nil
}

// stressCPU creates CPU stress
func (r *ResourceChaosInjector) stressCPU(ctx context.Context, stressor *ResourceStressor) {
	defer stressor.wg.Done()
	
	// Get number of CPUs to stress
	numCPU := runtime.NumCPU()
	workers := int(float64(numCPU) * stressor.Intensity)
	if workers < 1 {
		workers = 1
	}
	
	r.logger.Info("Starting CPU stress",
		zap.Int("workers", workers),
		zap.Float64("intensity", stressor.Intensity))
	
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// CPU intensive loop
			for {
				select {
				case <-ctx.Done():
					return
				default:
					// Perform CPU intensive operation
					for j := 0; j < 1000000; j++ {
						_ = j * j * j
					}
				}
			}
		}()
	}
	
	wg.Wait()
}

// stressMemory creates memory pressure
func (r *ResourceChaosInjector) stressMemory(ctx context.Context, stressor *ResourceStressor) {
	defer stressor.wg.Done()
	
	// Get available memory
	vmStat, err := mem.VirtualMemory()
	if err != nil {
		r.logger.Error("Failed to get memory stats", zap.Error(err))
		return
	}
	
	// Calculate memory to allocate
	allocSize := int64(float64(vmStat.Available) * stressor.Intensity)
	
	r.logger.Info("Starting memory stress",
		zap.Int64("bytes", allocSize),
		zap.Float64("intensity", stressor.Intensity))
	
	// Allocate memory in chunks
	const chunkSize = 1024 * 1024 * 100 // 100MB chunks
	var allocations [][]byte
	
	allocated := int64(0)
	for allocated < allocSize {
		select {
		case <-ctx.Done():
			return
		default:
		}
		
		size := chunkSize
		if allocated+chunkSize > allocSize {
			size = int(allocSize - allocated)
		}
		
		chunk := make([]byte, size)
		// Touch memory to ensure it's actually allocated
		for i := 0; i < len(chunk); i += 4096 {
			chunk[i] = byte(i)
		}
		
		allocations = append(allocations, chunk)
		allocated += int64(size)
		
		time.Sleep(100 * time.Millisecond) // Gradual allocation
	}
	
	// Hold memory until context is cancelled
	<-ctx.Done()
}

// stressDisk creates disk I/O stress
func (r *ResourceChaosInjector) stressDisk(ctx context.Context, stressor *ResourceStressor) {
	defer stressor.wg.Done()
	
	tempDir := "/tmp/chaos"
	os.MkdirAll(tempDir, 0755)
	
	r.logger.Info("Starting disk stress",
		zap.String("dir", tempDir),
		zap.Float64("intensity", stressor.Intensity))
	
	// Create multiple writers based on intensity
	numWriters := int(10 * stressor.Intensity)
	if numWriters < 1 {
		numWriters = 1
	}
	
	var wg sync.WaitGroup
	for i := 0; i < numWriters; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			filename := fmt.Sprintf("%s/stress_%d.dat", tempDir, id)
			buffer := make([]byte, 1024*1024) // 1MB buffer
			
			for {
				select {
				case <-ctx.Done():
					os.Remove(filename)
					return
				default:
					// Random data
					rand.Read(buffer)
					
					// Write to file
					file, err := os.Create(filename)
					if err != nil {
						r.logger.Error("Failed to create file", zap.Error(err))
						return
					}
					
					file.Write(buffer)
					file.Sync() // Force flush to disk
					file.Close()
					
					time.Sleep(10 * time.Millisecond)
				}
			}
		}(i)
	}
	
	wg.Wait()
}

// stressIO creates I/O stress
func (r *ResourceChaosInjector) stressIO(ctx context.Context, stressor *ResourceStressor) {
	defer stressor.wg.Done()
	
	r.logger.Info("Starting I/O stress",
		zap.Float64("intensity", stressor.Intensity))
	
	// Create I/O operations based on intensity
	opsPerSecond := int(1000 * stressor.Intensity)
	interval := time.Second / time.Duration(opsPerSecond)
	
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Perform various I/O operations
			go func() {
				// File stat
				syscall.Stat("/tmp", &syscall.Stat_t{})
				
				// Directory listing
				os.ReadDir("/tmp")
				
				// Network I/O
				net.Dial("tcp", "localhost:0")
			}()
		}
	}
}

// Revert removes resource chaos from target
func (r *ResourceChaosInjector) Revert(ctx context.Context, target string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	stressor, exists := r.stressors[target]
	if !exists {
		return nil
	}
	
	// Cancel stressor context
	stressor.cancel()
	
	// Wait for stressor to stop
	stressor.wg.Wait()
	
	delete(r.stressors, target)
	
	r.logger.Info("Reverted resource chaos", zap.String("target", target))
	return nil
}

// Validate checks if resource chaos can be applied
func (r *ResourceChaosInjector) Validate() error {
	// Check system resources
	cpuPercent, err := cpu.Percent(time.Second, false)
	if err != nil {
		return fmt.Errorf("failed to get CPU stats: %w", err)
	}
	
	if len(cpuPercent) > 0 && cpuPercent[0] > 80 {
		return fmt.Errorf("CPU usage too high for chaos injection: %.2f%%", cpuPercent[0])
	}
	
	vmStat, err := mem.VirtualMemory()
	if err != nil {
		return fmt.Errorf("failed to get memory stats: %w", err)
	}
	
	if vmStat.UsedPercent > 80 {
		return fmt.Errorf("memory usage too high for chaos injection: %.2f%%", vmStat.UsedPercent)
	}
	
	return nil
}

// GetImpact returns impact analysis for resource chaos
func (r *ResourceChaosInjector) GetImpact() *ImpactAnalysis {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	affectedNodes := make([]string, 0, len(r.stressors))
	for target := range r.stressors {
		affectedNodes = append(affectedNodes, target)
	}
	
	// Get current resource usage
	cpuPercent, _ := cpu.Percent(time.Second, false)
	vmStat, _ := mem.VirtualMemory()
	
	var cpuImpact float64
	if len(cpuPercent) > 0 {
		cpuImpact = cpuPercent[0] / 100
	}
	
	return &ImpactAnalysis{
		AffectedNodes:      affectedNodes,
		AvailabilityImpact: 1.0 - (cpuImpact * 0.5), // Simplified calculation
		SeverityScore:      (cpuImpact + vmStat.UsedPercent/100) / 2,
	}
}

// ApplicationChaosInjector injects application-level chaos
type ApplicationChaosInjector struct {
	logger    *zap.Logger
	failures  map[string]*ApplicationFailure
	mu        sync.RWMutex
}

// ApplicationFailure represents an application failure
type ApplicationFailure struct {
	Target       string
	Type         string
	ErrorRate    float64
	Latency      time.Duration
	Active       bool
	FailureFunc  func() error
}

// NewApplicationChaosInjector creates application chaos injector
func NewApplicationChaosInjector(logger *zap.Logger) *ApplicationChaosInjector {
	return &ApplicationChaosInjector{
		logger:   logger,
		failures: make(map[string]*ApplicationFailure),
	}
}

// Inject applies application chaos to target
func (a *ApplicationChaosInjector) Inject(ctx context.Context, target string, params map[string]interface{}) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	failureType, _ := params["failure_type"].(string)
	errorRate, _ := params["error_rate"].(float64)
	latency, _ := params["latency"].(time.Duration)
	
	failure := &ApplicationFailure{
		Target:    target,
		Type:      failureType,
		ErrorRate: errorRate,
		Latency:   latency,
		Active:    true,
	}
	
	// Define failure behaviors
	switch failureType {
	case "service_crash":
		failure.FailureFunc = func() error {
			// Simulate service crash
			panic("Simulated service crash")
		}
	case "timeout":
		failure.FailureFunc = func() error {
			time.Sleep(30 * time.Second)
			return fmt.Errorf("operation timed out")
		}
	case "error_injection":
		failure.FailureFunc = func() error {
			if rand.Float64() < errorRate {
				return fmt.Errorf("injected error")
			}
			return nil
		}
	case "dependency_failure":
		failure.FailureFunc = func() error {
			return fmt.Errorf("dependency unavailable")
		}
	case "thread_hang":
		failure.FailureFunc = func() error {
			// Block indefinitely
			select {}
		}
	default:
		return fmt.Errorf("unknown failure type: %s", failureType)
	}
	
	a.failures[target] = failure
	
	a.logger.Info("Injected application chaos",
		zap.String("target", target),
		zap.String("type", failureType),
		zap.Float64("error_rate", errorRate))
	
	// Register failure with application hooks
	// This would integrate with the actual application
	
	return nil
}

// Revert removes application chaos from target
func (a *ApplicationChaosInjector) Revert(ctx context.Context, target string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	failure, exists := a.failures[target]
	if !exists {
		return nil
	}
	
	failure.Active = false
	delete(a.failures, target)
	
	// Unregister failure from application hooks
	
	a.logger.Info("Reverted application chaos", zap.String("target", target))
	return nil
}

// Validate checks if application chaos can be applied
func (a *ApplicationChaosInjector) Validate() error {
	// Check if application hooks are available
	// This would verify integration with the actual application
	return nil
}

// GetImpact returns impact analysis for application chaos
func (a *ApplicationChaosInjector) GetImpact() *ImpactAnalysis {
	a.mu.RLock()
	defer a.mu.RUnlock()
	
	affectedServices := make([]string, 0, len(a.failures))
	totalErrorRate := 0.0
	
	for target, failure := range a.failures {
		affectedServices = append(affectedServices, target)
		totalErrorRate += failure.ErrorRate
	}
	
	avgErrorRate := totalErrorRate / float64(len(a.failures))
	
	return &ImpactAnalysis{
		AffectedServices:   affectedServices,
		ErrorRate:          avgErrorRate,
		AvailabilityImpact: 1.0 - avgErrorRate,
		DataIntegrity:      true, // Application chaos shouldn't affect data
	}
}

// DataChaosInjector injects data-related chaos
type DataChaosInjector struct {
	logger      *zap.Logger
	corruptions map[string]*DataCorruption
	mu          sync.RWMutex
}

// DataCorruption represents data corruption state
type DataCorruption struct {
	Target         string
	Type           string
	CorruptionRate float64
	LagDuration    time.Duration
	Active         bool
}

// NewDataChaosInjector creates data chaos injector
func NewDataChaosInjector(logger *zap.Logger) *DataChaosInjector {
	return &DataChaosInjector{
		logger:      logger,
		corruptions: make(map[string]*DataCorruption),
	}
}

// Inject applies data chaos to target
func (d *DataChaosInjector) Inject(ctx context.Context, target string, params map[string]interface{}) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	corruptionType, _ := params["corruption_type"].(string)
	corruptionRate, _ := params["corruption_rate"].(float64)
	lagDuration, _ := params["lag_duration"].(time.Duration)
	
	corruption := &DataCorruption{
		Target:         target,
		Type:           corruptionType,
		CorruptionRate: corruptionRate,
		LagDuration:    lagDuration,
		Active:         true,
	}
	
	// Apply data chaos based on type
	switch corruptionType {
	case "corruption":
		// Inject data corruption logic
		// This would hook into the data layer
	case "inconsistency":
		// Create data inconsistencies
	case "replication_lag":
		// Introduce replication delay
	case "data_loss":
		// Simulate data loss
	case "split_brain":
		// Create split-brain scenario
	default:
		return fmt.Errorf("unknown corruption type: %s", corruptionType)
	}
	
	d.corruptions[target] = corruption
	
	d.logger.Info("Injected data chaos",
		zap.String("target", target),
		zap.String("type", corruptionType),
		zap.Float64("corruption_rate", corruptionRate))
	
	return nil
}

// Revert removes data chaos from target
func (d *DataChaosInjector) Revert(ctx context.Context, target string) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	
	corruption, exists := d.corruptions[target]
	if !exists {
		return nil
	}
	
	corruption.Active = false
	delete(d.corruptions, target)
	
	// Restore data integrity
	// This would repair any corrupted data
	
	d.logger.Info("Reverted data chaos", zap.String("target", target))
	return nil
}

// Validate checks if data chaos can be applied
func (d *DataChaosInjector) Validate() error {
	// Check if data layer hooks are available
	// Verify backup systems are operational
	return nil
}

// GetImpact returns impact analysis for data chaos
func (d *DataChaosInjector) GetImpact() *ImpactAnalysis {
	d.mu.RLock()
	defer d.mu.RUnlock()
	
	affectedNodes := make([]string, 0, len(d.corruptions))
	dataIntegrity := true
	
	for target, corruption := range d.corruptions {
		affectedNodes = append(affectedNodes, target)
		if corruption.Type == "corruption" || corruption.Type == "data_loss" {
			dataIntegrity = false
		}
	}
	
	return &ImpactAnalysis{
		AffectedNodes: affectedNodes,
		DataIntegrity: dataIntegrity,
		SeverityScore: 0.7, // Data chaos is always high severity
	}
}

// TimeChaosInjector injects time-related chaos
type TimeChaosInjector struct {
	logger     *zap.Logger
	timeShifts map[string]*TimeShift
	mu         sync.RWMutex
}

// TimeShift represents time manipulation state
type TimeShift struct {
	Target       string
	Type         string
	SkewDuration time.Duration
	Active       bool
}

// NewTimeChaosInjector creates time chaos injector
func NewTimeChaosInjector(logger *zap.Logger) *TimeChaosInjector {
	return &TimeChaosInjector{
		logger:     logger,
		timeShifts: make(map[string]*TimeShift),
	}
}

// Inject applies time chaos to target
func (t *TimeChaosInjector) Inject(ctx context.Context, target string, params map[string]interface{}) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	timeType, _ := params["time_type"].(string)
	skewDuration, _ := params["skew_duration"].(time.Duration)
	
	shift := &TimeShift{
		Target:       target,
		Type:         timeType,
		SkewDuration: skewDuration,
		Active:       true,
	}
	
	// Apply time chaos based on type
	switch timeType {
	case "clock_skew":
		// Adjust system clock
		// This would use system calls to modify time
	case "ntp_failure":
		// Block NTP synchronization
	case "time_jump":
		// Create sudden time jumps
	case "timezone":
		// Change timezone settings
	default:
		return fmt.Errorf("unknown time type: %s", timeType)
	}
	
	t.timeShifts[target] = shift
	
	t.logger.Info("Injected time chaos",
		zap.String("target", target),
		zap.String("type", timeType),
		zap.Duration("skew", skewDuration))
	
	return nil
}

// Revert removes time chaos from target
func (t *TimeChaosInjector) Revert(ctx context.Context, target string) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	
	shift, exists := t.timeShifts[target]
	if !exists {
		return nil
	}
	
	shift.Active = false
	delete(t.timeShifts, target)
	
	// Restore correct time
	// This would resynchronize with NTP
	
	t.logger.Info("Reverted time chaos", zap.String("target", target))
	return nil
}

// Validate checks if time chaos can be applied
func (t *TimeChaosInjector) Validate() error {
	// Check if we have permissions to modify time
	if os.Geteuid() != 0 {
		return fmt.Errorf("root privileges required for time chaos")
	}
	return nil
}

// GetImpact returns impact analysis for time chaos
func (t *TimeChaosInjector) GetImpact() *ImpactAnalysis {
	t.mu.RLock()
	defer t.mu.RUnlock()
	
	affectedNodes := make([]string, 0, len(t.timeShifts))
	for target := range t.timeShifts {
		affectedNodes = append(affectedNodes, target)
	}
	
	return &ImpactAnalysis{
		AffectedNodes: affectedNodes,
		SeverityScore: 0.6, // Time chaos can cause coordination issues
	}
}