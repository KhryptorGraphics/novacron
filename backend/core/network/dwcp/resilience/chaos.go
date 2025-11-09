package resilience

import (
	"errors"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// ChaosMonkey implements chaos engineering fault injection
type ChaosMonkey struct {
	name        string
	enabled     bool
	probability float64
	faults      []FaultInjector
	logger      *zap.Logger
	mu          sync.RWMutex

	// Metrics
	totalChecks      int64
	totalInjections  int64
	injectionsByType map[string]int64
	injectionsMu     sync.RWMutex
}

// NewChaosMonkey creates a new chaos monkey
func NewChaosMonkey(name string, probability float64, logger *zap.Logger) *ChaosMonkey {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &ChaosMonkey{
		name:             name,
		enabled:          false, // Disabled by default for safety
		probability:      probability,
		faults:           make([]FaultInjector, 0),
		logger:           logger,
		injectionsByType: make(map[string]int64),
	}
}

// Enable enables chaos monkey
func (cm *ChaosMonkey) Enable() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.enabled = true
	cm.logger.Warn("Chaos monkey ENABLED",
		zap.String("name", cm.name),
		zap.Float64("probability", cm.probability))
}

// Disable disables chaos monkey
func (cm *ChaosMonkey) Disable() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.enabled = false
	cm.logger.Info("Chaos monkey disabled",
		zap.String("name", cm.name))
}

// IsEnabled returns whether chaos monkey is enabled
func (cm *ChaosMonkey) IsEnabled() bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	return cm.enabled
}

// SetProbability sets the fault injection probability
func (cm *ChaosMonkey) SetProbability(probability float64) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.probability = probability
	cm.logger.Info("Chaos monkey probability updated",
		zap.String("name", cm.name),
		zap.Float64("probability", probability))
}

// RegisterFault registers a fault injector
func (cm *ChaosMonkey) RegisterFault(fault FaultInjector) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.faults = append(cm.faults, fault)
	cm.logger.Info("Fault injector registered",
		zap.String("chaosMonkey", cm.name),
		zap.String("fault", fault.Name()))
}

// MaybeInject potentially injects a fault
func (cm *ChaosMonkey) MaybeInject() error {
	atomic.AddInt64(&cm.totalChecks, 1)

	cm.mu.RLock()
	enabled := cm.enabled
	probability := cm.probability
	faultsCount := len(cm.faults)
	cm.mu.RUnlock()

	if !enabled || faultsCount == 0 {
		return nil
	}

	// Check if we should inject a fault
	if rand.Float64() > probability {
		return nil
	}

	// Select a random fault
	cm.mu.RLock()
	fault := cm.faults[rand.Intn(len(cm.faults))]
	cm.mu.RUnlock()

	// Inject the fault
	atomic.AddInt64(&cm.totalInjections, 1)

	cm.injectionsMu.Lock()
	cm.injectionsByType[fault.Name()]++
	cm.injectionsMu.Unlock()

	cm.logger.Debug("Chaos monkey injecting fault",
		zap.String("chaosMonkey", cm.name),
		zap.String("fault", fault.Name()))

	return fault.Inject()
}

// GetMetrics returns chaos monkey metrics
func (cm *ChaosMonkey) GetMetrics() ChaosMonkeyMetrics {
	cm.injectionsMu.RLock()
	injectionsByType := make(map[string]int64)
	for k, v := range cm.injectionsByType {
		injectionsByType[k] = v
	}
	cm.injectionsMu.RUnlock()

	totalChecks := atomic.LoadInt64(&cm.totalChecks)
	totalInjections := atomic.LoadInt64(&cm.totalInjections)

	injectionRate := float64(0)
	if totalChecks > 0 {
		injectionRate = float64(totalInjections) / float64(totalChecks)
	}

	return ChaosMonkeyMetrics{
		Name:             cm.name,
		Enabled:          cm.IsEnabled(),
		Probability:      cm.probability,
		TotalChecks:      totalChecks,
		TotalInjections:  totalInjections,
		InjectionRate:    injectionRate,
		InjectionsByType: injectionsByType,
	}
}

// FaultInjector interface for fault injection
type FaultInjector interface {
	Name() string
	Inject() error
}

// LatencyFault injects latency
type LatencyFault struct {
	name     string
	minDelay time.Duration
	maxDelay time.Duration
}

// NewLatencyFault creates a latency fault injector
func NewLatencyFault(name string, minDelay, maxDelay time.Duration) *LatencyFault {
	return &LatencyFault{
		name:     name,
		minDelay: minDelay,
		maxDelay: maxDelay,
	}
}

// Name returns the fault name
func (lf *LatencyFault) Name() string {
	return lf.name
}

// Inject injects latency
func (lf *LatencyFault) Inject() error {
	delay := lf.minDelay
	if lf.maxDelay > lf.minDelay {
		delta := lf.maxDelay - lf.minDelay
		delay = lf.minDelay + time.Duration(rand.Float64()*float64(delta))
	}

	time.Sleep(delay)
	return nil
}

// ErrorFault injects errors
type ErrorFault struct {
	name      string
	errorRate float64
	errors    []error
}

// NewErrorFault creates an error fault injector
func NewErrorFault(name string, errorRate float64, errs ...error) *ErrorFault {
	if len(errs) == 0 {
		errs = []error{errors.New("chaos monkey injected error")}
	}

	return &ErrorFault{
		name:      name,
		errorRate: errorRate,
		errors:    errs,
	}
}

// Name returns the fault name
func (ef *ErrorFault) Name() string {
	return ef.name
}

// Inject injects an error
func (ef *ErrorFault) Inject() error {
	if rand.Float64() < ef.errorRate {
		// Select a random error
		err := ef.errors[rand.Intn(len(ef.errors))]
		return err
	}
	return nil
}

// PanicFault injects panics (use with extreme caution!)
type PanicFault struct {
	name      string
	panicRate float64
	message   string
}

// NewPanicFault creates a panic fault injector
func NewPanicFault(name string, panicRate float64, message string) *PanicFault {
	return &PanicFault{
		name:      name,
		panicRate: panicRate,
		message:   message,
	}
}

// Name returns the fault name
func (pf *PanicFault) Name() string {
	return pf.name
}

// Inject injects a panic
func (pf *PanicFault) Inject() error {
	if rand.Float64() < pf.panicRate {
		panic(pf.message)
	}
	return nil
}

// MemoryLeakFault simulates memory leaks
type MemoryLeakFault struct {
	name     string
	leakSize int // bytes to leak
	leaks    [][]byte
	mu       sync.Mutex
}

// NewMemoryLeakFault creates a memory leak fault injector
func NewMemoryLeakFault(name string, leakSize int) *MemoryLeakFault {
	return &MemoryLeakFault{
		name:     name,
		leakSize: leakSize,
		leaks:    make([][]byte, 0),
	}
}

// Name returns the fault name
func (mlf *MemoryLeakFault) Name() string {
	return mlf.name
}

// Inject injects a memory leak
func (mlf *MemoryLeakFault) Inject() error {
	mlf.mu.Lock()
	defer mlf.mu.Unlock()

	// Allocate memory that won't be freed
	leak := make([]byte, mlf.leakSize)
	for i := range leak {
		leak[i] = byte(rand.Intn(256))
	}
	mlf.leaks = append(mlf.leaks, leak)

	return nil
}

// NetworkPartitionFault simulates network partitions
type NetworkPartitionFault struct {
	name         string
	partitionDur time.Duration
	onPartition  func() error
	onRecover    func() error
}

// NewNetworkPartitionFault creates a network partition fault injector
func NewNetworkPartitionFault(name string, duration time.Duration, onPartition, onRecover func() error) *NetworkPartitionFault {
	return &NetworkPartitionFault{
		name:         name,
		partitionDur: duration,
		onPartition:  onPartition,
		onRecover:    onRecover,
	}
}

// Name returns the fault name
func (npf *NetworkPartitionFault) Name() string {
	return npf.name
}

// Inject injects a network partition
func (npf *NetworkPartitionFault) Inject() error {
	// Trigger partition
	if npf.onPartition != nil {
		if err := npf.onPartition(); err != nil {
			return err
		}
	}

	// Wait for partition duration
	time.Sleep(npf.partitionDur)

	// Recover from partition
	if npf.onRecover != nil {
		return npf.onRecover()
	}

	return nil
}

// CPUSpikeFault injects CPU spikes
type CPUSpikeFault struct {
	name     string
	duration time.Duration
	threads  int
}

// NewCPUSpikeFault creates a CPU spike fault injector
func NewCPUSpikeFault(name string, duration time.Duration, threads int) *CPUSpikeFault {
	return &CPUSpikeFault{
		name:     name,
		duration: duration,
		threads:  threads,
	}
}

// Name returns the fault name
func (csf *CPUSpikeFault) Name() string {
	return csf.name
}

// Inject injects a CPU spike
func (csf *CPUSpikeFault) Inject() error {
	stopCh := make(chan struct{})

	// Start CPU-intensive goroutines
	for i := 0; i < csf.threads; i++ {
		go func() {
			x := 0
			for {
				select {
				case <-stopCh:
					return
				default:
					// Busy loop to consume CPU
					x++
					if x > 1000000 {
						x = 0
					}
				}
			}
		}()
	}

	// Let it run for duration
	time.Sleep(csf.duration)
	close(stopCh)

	return nil
}

// DiskFillFault simulates disk filling
type DiskFillFault struct {
	name     string
	fillSize int64
	tempData []byte
	mu       sync.Mutex
}

// NewDiskFillFault creates a disk fill fault injector
func NewDiskFillFault(name string, fillSize int64) *DiskFillFault {
	return &DiskFillFault{
		name:     name,
		fillSize: fillSize,
	}
}

// Name returns the fault name
func (dff *DiskFillFault) Name() string {
	return dff.name
}

// Inject injects a disk fill
func (dff *DiskFillFault) Inject() error {
	dff.mu.Lock()
	defer dff.mu.Unlock()

	// Allocate memory to simulate disk usage
	dff.tempData = make([]byte, dff.fillSize)
	for i := range dff.tempData {
		dff.tempData[i] = byte(rand.Intn(256))
	}

	return nil
}

// TimeoutFault forces operations to timeout
type TimeoutFault struct {
	name    string
	timeout time.Duration
}

// NewTimeoutFault creates a timeout fault injector
func NewTimeoutFault(name string, timeout time.Duration) *TimeoutFault {
	return &TimeoutFault{
		name:    name,
		timeout: timeout,
	}
}

// Name returns the fault name
func (tf *TimeoutFault) Name() string {
	return tf.name
}

// Inject injects a timeout
func (tf *TimeoutFault) Inject() error {
	time.Sleep(tf.timeout)
	return errors.New("operation timed out due to chaos injection")
}

// CompositeFault combines multiple faults
type CompositeFault struct {
	name   string
	faults []FaultInjector
}

// NewCompositeFault creates a composite fault injector
func NewCompositeFault(name string, faults ...FaultInjector) *CompositeFault {
	return &CompositeFault{
		name:   name,
		faults: faults,
	}
}

// Name returns the fault name
func (cf *CompositeFault) Name() string {
	return cf.name
}

// Inject injects all faults
func (cf *CompositeFault) Inject() error {
	for _, fault := range cf.faults {
		if err := fault.Inject(); err != nil {
			return err
		}
	}
	return nil
}

// Metrics types

// ChaosMonkeyMetrics contains chaos monkey metrics
type ChaosMonkeyMetrics struct {
	Name             string
	Enabled          bool
	Probability      float64
	TotalChecks      int64
	TotalInjections  int64
	InjectionRate    float64
	InjectionsByType map[string]int64
}