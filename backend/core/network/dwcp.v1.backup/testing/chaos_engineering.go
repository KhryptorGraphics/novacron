package testing

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// ChaosExperiment represents a chaos engineering experiment
type ChaosExperiment struct {
	Name        string
	Description string
	FaultType   FaultType
	Probability float64
	Duration    time.Duration
	ImpactLevel ImpactLevel
	Recovery    RecoveryStrategy
	Metrics     *ChaosMetrics
}

// FaultType defines types of faults to inject
type FaultType int

const (
	FaultNetworkPartition FaultType = iota
	FaultHighLatency
	FaultPacketLoss
	FaultBandwidthDegradation
	FaultNodeFailure
	FaultDiskFailure
	FaultMemoryPressure
	FaultCPUStarvation
	FaultClockSkew
	FaultDNSFailure
)

// ImpactLevel defines the severity of the fault
type ImpactLevel int

const (
	ImpactLow ImpactLevel = iota
	ImpactMedium
	ImpactHigh
	ImpactCritical
)

// RecoveryStrategy defines how to recover from a fault
type RecoveryStrategy int

const (
	RecoveryAutomatic RecoveryStrategy = iota
	RecoveryManual
	RecoveryGradual
	RecoveryImmediate
)

// ChaosMetrics tracks chaos experiment metrics
type ChaosMetrics struct {
	FaultsInjected    int
	RecoveryAttempts  int
	RecoverySuccesses int
	RecoveryFailures  int
	MTTR              time.Duration // Mean Time To Recovery
	ImpactDuration    time.Duration
	mu                sync.RWMutex
}

// ChaosEngine manages chaos experiments
type ChaosEngine struct {
	experiments   []*ChaosExperiment
	simulator     *NetworkSimulator
	tcController  *TrafficController
	running       map[string]bool
	ctx           context.Context
	cancel        context.CancelFunc
	mu            sync.RWMutex
}

// NewChaosEngine creates a new chaos engine
func NewChaosEngine(simulator *NetworkSimulator, tcController *TrafficController) *ChaosEngine {
	ctx, cancel := context.WithCancel(context.Background())

	return &ChaosEngine{
		experiments:  make([]*ChaosExperiment, 0),
		simulator:    simulator,
		tcController: tcController,
		running:      make(map[string]bool),
		ctx:          ctx,
		cancel:       cancel,
	}
}

// AddExperiment adds a chaos experiment
func (ce *ChaosEngine) AddExperiment(exp *ChaosExperiment) {
	ce.mu.Lock()
	defer ce.mu.Unlock()

	exp.Metrics = &ChaosMetrics{}
	ce.experiments = append(ce.experiments, exp)
}

// RunExperiment runs a specific chaos experiment
func (ce *ChaosEngine) RunExperiment(experimentName string) error {
	ce.mu.RLock()
	var experiment *ChaosExperiment
	for _, exp := range ce.experiments {
		if exp.Name == experimentName {
			experiment = exp
			break
		}
	}
	ce.mu.RUnlock()

	if experiment == nil {
		return fmt.Errorf("experiment not found: %s", experimentName)
	}

	ce.mu.Lock()
	if ce.running[experimentName] {
		ce.mu.Unlock()
		return fmt.Errorf("experiment already running: %s", experimentName)
	}
	ce.running[experimentName] = true
	ce.mu.Unlock()

	defer func() {
		ce.mu.Lock()
		delete(ce.running, experimentName)
		ce.mu.Unlock()
	}()

	fmt.Printf("Starting chaos experiment: %s\n", experimentName)

	// Run experiment
	ctx, cancel := context.WithTimeout(ce.ctx, experiment.Duration)
	defer cancel()

	return ce.injectFaults(ctx, experiment)
}

// injectFaults injects faults according to the experiment
func (ce *ChaosEngine) injectFaults(ctx context.Context, exp *ChaosExperiment) error {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			// Experiment duration completed
			fmt.Printf("Chaos experiment %s completed\n", exp.Name)
			return ce.recoverFromFault(exp)

		case <-ticker.C:
			// Probabilistic fault injection
			if rand.Float64() < exp.Probability {
				if err := ce.injectFault(exp); err != nil {
					fmt.Printf("Error injecting fault: %v\n", err)
				} else {
					exp.Metrics.mu.Lock()
					exp.Metrics.FaultsInjected++
					exp.Metrics.mu.Unlock()
				}
			}
		}
	}
}

// injectFault injects a specific fault
func (ce *ChaosEngine) injectFault(exp *ChaosExperiment) error {
	faultStart := time.Now()

	switch exp.FaultType {
	case FaultNetworkPartition:
		return ce.injectNetworkPartition(exp)

	case FaultHighLatency:
		return ce.injectHighLatency(exp)

	case FaultPacketLoss:
		return ce.injectPacketLoss(exp)

	case FaultBandwidthDegradation:
		return ce.injectBandwidthDegradation(exp)

	case FaultNodeFailure:
		return ce.injectNodeFailure(exp)

	case FaultClockSkew:
		return ce.injectClockSkew(exp)

	default:
		return fmt.Errorf("unsupported fault type: %v", exp.FaultType)
	}

	exp.Metrics.mu.Lock()
	exp.Metrics.ImpactDuration += time.Since(faultStart)
	exp.Metrics.mu.Unlock()

	return nil
}

// injectNetworkPartition simulates network partition
func (ce *ChaosEngine) injectNetworkPartition(exp *ChaosExperiment) error {
	fmt.Printf("[CHAOS] Injecting network partition\n")

	// Block all traffic
	config := &ComplexNetworkConfig{
		PacketLoss: 100.0, // 100% packet loss
	}

	return ce.tcController.ApplyComplex(config)
}

// injectHighLatency injects high latency
func (ce *ChaosEngine) injectHighLatency(exp *ChaosExperiment) error {
	latency := ce.getLatencyForImpact(exp.ImpactLevel)
	fmt.Printf("[CHAOS] Injecting high latency: %v\n", latency)

	return ce.tcController.ApplyLatency(latency, latency/10)
}

// injectPacketLoss injects packet loss
func (ce *ChaosEngine) injectPacketLoss(exp *ChaosExperiment) error {
	lossRate := ce.getPacketLossForImpact(exp.ImpactLevel)
	fmt.Printf("[CHAOS] Injecting packet loss: %.2f%%\n", lossRate)

	return ce.tcController.ApplyPacketLoss(lossRate)
}

// injectBandwidthDegradation injects bandwidth degradation
func (ce *ChaosEngine) injectBandwidthDegradation(exp *ChaosExperiment) error {
	bandwidth := ce.getBandwidthForImpact(exp.ImpactLevel)
	fmt.Printf("[CHAOS] Injecting bandwidth degradation: %d Mbps\n", bandwidth)

	return ce.tcController.ApplyBandwidth(bandwidth)
}

// injectNodeFailure simulates node failure
func (ce *ChaosEngine) injectNodeFailure(exp *ChaosExperiment) error {
	fmt.Printf("[CHAOS] Injecting node failure\n")
	// In a real implementation, this would stop/kill a node process
	return nil
}

// injectClockSkew injects clock skew
func (ce *ChaosEngine) injectClockSkew(exp *ChaosExperiment) error {
	skew := ce.getClockSkewForImpact(exp.ImpactLevel)
	fmt.Printf("[CHAOS] Injecting clock skew: %v\n", skew)
	// In a real implementation, this would adjust system time
	return nil
}

// recoverFromFault recovers from injected fault
func (ce *ChaosEngine) recoverFromFault(exp *ChaosExperiment) error {
	recoveryStart := time.Now()

	fmt.Printf("[CHAOS] Recovering from %s\n", exp.Name)

	exp.Metrics.mu.Lock()
	exp.Metrics.RecoveryAttempts++
	exp.Metrics.mu.Unlock()

	var err error

	switch exp.Recovery {
	case RecoveryImmediate:
		err = ce.tcController.Reset()

	case RecoveryGradual:
		// Gradually restore network conditions
		err = ce.gradualRecovery(exp)

	case RecoveryAutomatic:
		err = ce.tcController.Reset()

	case RecoveryManual:
		fmt.Println("[CHAOS] Manual recovery required")
		return nil
	}

	recoveryTime := time.Since(recoveryStart)

	exp.Metrics.mu.Lock()
	if err == nil {
		exp.Metrics.RecoverySuccesses++
	} else {
		exp.Metrics.RecoveryFailures++
	}
	exp.Metrics.MTTR = (exp.Metrics.MTTR + recoveryTime) / 2
	exp.Metrics.mu.Unlock()

	return err
}

// gradualRecovery gradually restores network conditions
func (ce *ChaosEngine) gradualRecovery(exp *ChaosExperiment) error {
	steps := 5
	stepDuration := 2 * time.Second

	for i := steps; i > 0; i-- {
		impact := float64(i) / float64(steps)

		switch exp.FaultType {
		case FaultHighLatency:
			latency := time.Duration(float64(ce.getLatencyForImpact(exp.ImpactLevel)) * impact)
			if err := ce.tcController.ApplyLatency(latency, latency/10); err != nil {
				return err
			}

		case FaultPacketLoss:
			lossRate := ce.getPacketLossForImpact(exp.ImpactLevel) * impact
			if err := ce.tcController.ApplyPacketLoss(lossRate); err != nil {
				return err
			}

		case FaultBandwidthDegradation:
			baseBandwidth := ce.getBandwidthForImpact(exp.ImpactLevel)
			fullBandwidth := 10000 // 10 Gbps
			bandwidth := int(float64(fullBandwidth-baseBandwidth)*impact) + baseBandwidth
			if err := ce.tcController.ApplyBandwidth(bandwidth); err != nil {
				return err
			}
		}

		time.Sleep(stepDuration)
	}

	return ce.tcController.Reset()
}

// Helper functions to determine fault severity
func (ce *ChaosEngine) getLatencyForImpact(impact ImpactLevel) time.Duration {
	switch impact {
	case ImpactLow:
		return 50 * time.Millisecond
	case ImpactMedium:
		return 200 * time.Millisecond
	case ImpactHigh:
		return 500 * time.Millisecond
	case ImpactCritical:
		return 2 * time.Second
	default:
		return 100 * time.Millisecond
	}
}

func (ce *ChaosEngine) getPacketLossForImpact(impact ImpactLevel) float64 {
	switch impact {
	case ImpactLow:
		return 1.0 // 1%
	case ImpactMedium:
		return 5.0 // 5%
	case ImpactHigh:
		return 15.0 // 15%
	case ImpactCritical:
		return 50.0 // 50%
	default:
		return 5.0
	}
}

func (ce *ChaosEngine) getBandwidthForImpact(impact ImpactLevel) int {
	switch impact {
	case ImpactLow:
		return 1000 // 1 Gbps
	case ImpactMedium:
		return 100 // 100 Mbps
	case ImpactHigh:
		return 10 // 10 Mbps
	case ImpactCritical:
		return 1 // 1 Mbps
	default:
		return 100
	}
}

func (ce *ChaosEngine) getClockSkewForImpact(impact ImpactLevel) time.Duration {
	switch impact {
	case ImpactLow:
		return 100 * time.Millisecond
	case ImpactMedium:
		return 1 * time.Second
	case ImpactHigh:
		return 10 * time.Second
	case ImpactCritical:
		return 1 * time.Minute
	default:
		return 1 * time.Second
	}
}

// Stop stops all chaos experiments
func (ce *ChaosEngine) Stop() {
	ce.cancel()
	ce.tcController.Reset()
}

// GetExperimentStatus returns experiment status
func (ce *ChaosEngine) GetExperimentStatus(name string) (*ChaosMetrics, bool) {
	ce.mu.RLock()
	defer ce.mu.RUnlock()

	for _, exp := range ce.experiments {
		if exp.Name == name {
			return exp.Metrics, ce.running[name]
		}
	}

	return nil, false
}

// PredefinedExperiments returns common chaos experiments
func PredefinedExperiments() []*ChaosExperiment {
	return []*ChaosExperiment{
		{
			Name:        "Network Partition",
			Description: "Simulate complete network partition",
			FaultType:   FaultNetworkPartition,
			Probability: 0.1,
			Duration:    1 * time.Minute,
			ImpactLevel: ImpactCritical,
			Recovery:    RecoveryImmediate,
		},
		{
			Name:        "High Latency",
			Description: "Inject high network latency",
			FaultType:   FaultHighLatency,
			Probability: 0.2,
			Duration:    5 * time.Minute,
			ImpactLevel: ImpactHigh,
			Recovery:    RecoveryGradual,
		},
		{
			Name:        "Packet Loss",
			Description: "Inject random packet loss",
			FaultType:   FaultPacketLoss,
			Probability: 0.3,
			Duration:    3 * time.Minute,
			ImpactLevel: ImpactMedium,
			Recovery:    RecoveryGradual,
		},
		{
			Name:        "Bandwidth Degradation",
			Description: "Reduce available bandwidth",
			FaultType:   FaultBandwidthDegradation,
			Probability: 0.25,
			Duration:    10 * time.Minute,
			ImpactLevel: ImpactMedium,
			Recovery:    RecoveryGradual,
		},
		{
			Name:        "Clock Skew",
			Description: "Inject clock skew between nodes",
			FaultType:   FaultClockSkew,
			Probability: 0.15,
			Duration:    5 * time.Minute,
			ImpactLevel: ImpactLow,
			Recovery:    RecoveryAutomatic,
		},
	}
}
