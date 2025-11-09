package chaos

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"
)

// ChaosEngine manages chaos engineering experiments
type ChaosEngine struct {
	config      *ChaosConfig
	experiments map[string]*ChaosExperiment
	expMu       sync.RWMutex
	active      bool
	activeMu    sync.RWMutex
	safetyMgr   *SafetyManager
}

// ChaosConfig configures chaos engineering
type ChaosConfig struct {
	Enabled           bool
	Schedule          string
	MaxBlastRadius    int
	BusinessHoursOnly bool
	SafetyChecks      []string
	AutoAbort         bool
}

// ChaosExperiment defines a chaos test
type ChaosExperiment struct {
	ID               string
	Name             string
	Description      string
	TargetType       string // "pod", "node", "region", "network"
	TargetSelector   map[string]string
	FailureType      string
	Severity         int
	Duration         time.Duration
	BlastRadius      int
	Status           string // "scheduled", "running", "completed", "aborted"
	StartedAt        time.Time
	CompletedAt      time.Time
	AffectedTargets  []string
	Observations     []Observation
	AutoAbort        bool
}

// Observation tracks experiment observations
type Observation struct {
	Timestamp   time.Time
	Metric      string
	Value       float64
	Expected    float64
	Deviation   float64
	Description string
}

// SafetyManager enforces safety controls
type SafetyManager struct {
	checks      []SafetyCheck
	violations  []SafetyViolation
	mu          sync.RWMutex
}

// SafetyCheck defines a safety constraint
type SafetyCheck struct {
	Name        string
	Check       func(ctx context.Context) error
	Critical    bool
}

// SafetyViolation tracks safety violations
type SafetyViolation struct {
	CheckName   string
	Timestamp   time.Time
	Description string
	Critical    bool
}

// NewChaosEngine creates a new chaos engine
func NewChaosEngine(config *ChaosConfig) *ChaosEngine {
	ce := &ChaosEngine{
		config:      config,
		experiments: make(map[string]*ChaosExperiment),
		active:      config.Enabled,
		safetyMgr:   NewSafetyManager(),
	}

	return ce
}

// NewSafetyManager creates a safety manager
func NewSafetyManager() *SafetyManager {
	sm := &SafetyManager{
		checks:     make([]SafetyCheck, 0),
		violations: make([]SafetyViolation, 0),
	}

	// Register default safety checks
	sm.RegisterCheck(SafetyCheck{
		Name:     "quorum_available",
		Critical: true,
		Check:    checkQuorumAvailable,
	})

	sm.RegisterCheck(SafetyCheck{
		Name:     "no_ongoing_incidents",
		Critical: true,
		Check:    checkNoIncidents,
	})

	sm.RegisterCheck(SafetyCheck{
		Name:     "business_hours",
		Critical: false,
		Check:    checkBusinessHours,
	})

	sm.RegisterCheck(SafetyCheck{
		Name:     "backup_recent",
		Critical: true,
		Check:    checkRecentBackup,
	})

	return sm
}

// RegisterCheck registers a safety check
func (sm *SafetyManager) RegisterCheck(check SafetyCheck) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.checks = append(sm.checks, check)
}

// RunSafetyChecks executes all safety checks
func (sm *SafetyManager) RunSafetyChecks(ctx context.Context) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	for _, check := range sm.checks {
		if err := check.Check(ctx); err != nil {
			violation := SafetyViolation{
				CheckName:   check.Name,
				Timestamp:   time.Now(),
				Description: err.Error(),
				Critical:    check.Critical,
			}

			sm.violations = append(sm.violations, violation)

			if check.Critical {
				return fmt.Errorf("critical safety check failed: %s - %v", check.Name, err)
			}

			log.Printf("Safety check failed (non-critical): %s - %v", check.Name, err)
		}
	}

	return nil
}

// ScheduleExperiment schedules a chaos experiment
func (ce *ChaosEngine) ScheduleExperiment(experiment *ChaosExperiment) (string, error) {
	ce.activeMu.RLock()
	if !ce.active {
		ce.activeMu.RUnlock()
		return "", fmt.Errorf("chaos engineering is disabled")
	}
	ce.activeMu.RUnlock()

	// Validate experiment
	if experiment.BlastRadius > ce.config.MaxBlastRadius {
		return "", fmt.Errorf("blast radius %d exceeds maximum %d",
			experiment.BlastRadius, ce.config.MaxBlastRadius)
	}

	experiment.ID = fmt.Sprintf("chaos-%d", time.Now().Unix())
	experiment.Status = "scheduled"

	ce.expMu.Lock()
	ce.experiments[experiment.ID] = experiment
	ce.expMu.Unlock()

	log.Printf("Scheduled chaos experiment: %s - %s", experiment.ID, experiment.Name)

	return experiment.ID, nil
}

// RunExperiment executes a chaos experiment
func (ce *ChaosEngine) RunExperiment(ctx context.Context, experimentID string) error {
	ce.expMu.RLock()
	experiment, exists := ce.experiments[experimentID]
	ce.expMu.RUnlock()

	if !exists {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	// Run safety checks
	if err := ce.safetyMgr.RunSafetyChecks(ctx); err != nil {
		log.Printf("Safety checks failed, aborting experiment: %v", err)
		experiment.Status = "aborted"
		return err
	}

	experiment.Status = "running"
	experiment.StartedAt = time.Now()

	log.Printf("Starting chaos experiment: %s", experiment.Name)

	// Execute experiment
	go ce.executeExperiment(ctx, experiment)

	return nil
}

// executeExperiment executes the experiment
func (ce *ChaosEngine) executeExperiment(ctx context.Context, experiment *ChaosExperiment) {
	// Create experiment context with timeout
	expCtx, cancel := context.WithTimeout(ctx, experiment.Duration)
	defer cancel()

	// Inject failure based on type
	switch experiment.FailureType {
	case "kill":
		ce.injectPodKill(expCtx, experiment)
	case "latency":
		ce.injectNetworkLatency(expCtx, experiment)
	case "packet_loss":
		ce.injectPacketLoss(expCtx, experiment)
	case "resource_exhaustion":
		ce.injectResourceExhaustion(expCtx, experiment)
	default:
		log.Printf("Unknown failure type: %s", experiment.FailureType)
		experiment.Status = "failed"
		return
	}

	// Monitor and observe
	ce.observeExperiment(expCtx, experiment)

	experiment.CompletedAt = time.Now()
	experiment.Status = "completed"

	log.Printf("Chaos experiment completed: %s (duration: %v)",
		experiment.ID, experiment.CompletedAt.Sub(experiment.StartedAt))
}

// injectPodKill kills random pods
func (ce *ChaosEngine) injectPodKill(ctx context.Context, experiment *ChaosExperiment) {
	log.Printf("[Chaos] Injecting pod kills (blast radius: %d)", experiment.BlastRadius)

	for i := 0; i < experiment.BlastRadius; i++ {
		select {
		case <-ctx.Done():
			return
		default:
			podID := fmt.Sprintf("pod-%d", rand.Intn(1000))
			log.Printf("[Chaos] Killing pod: %s", podID)
			experiment.AffectedTargets = append(experiment.AffectedTargets, podID)

			// Simulate pod kill
			time.Sleep(500 * time.Millisecond)

			// Check if system recovers
			time.Sleep(2 * time.Second)

			// Record observation
			experiment.Observations = append(experiment.Observations, Observation{
				Timestamp:   time.Now(),
				Metric:      "pod_recovery_time",
				Value:       2.0,
				Expected:    5.0,
				Deviation:   -0.6,
				Description: fmt.Sprintf("Pod %s killed and recovered", podID),
			})
		}
	}
}

// injectNetworkLatency adds network latency
func (ce *ChaosEngine) injectNetworkLatency(ctx context.Context, experiment *ChaosExperiment) {
	latency := time.Duration(100+rand.Intn(400)) * time.Millisecond

	log.Printf("[Chaos] Injecting network latency: %v", latency)

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("[Chaos] Network latency injection stopped")
			return
		case <-ticker.C:
			// Measure impact
			experiment.Observations = append(experiment.Observations, Observation{
				Timestamp:   time.Now(),
				Metric:      "request_latency",
				Value:       float64(latency.Milliseconds()),
				Expected:    50.0,
				Deviation:   (float64(latency.Milliseconds()) - 50.0) / 50.0,
				Description: "Network latency injected",
			})
		}
	}
}

// injectPacketLoss injects packet loss
func (ce *ChaosEngine) injectPacketLoss(ctx context.Context, experiment *ChaosExperiment) {
	lossRate := 0.05 + rand.Float64()*0.15 // 5-20% loss

	log.Printf("[Chaos] Injecting packet loss: %.1f%%", lossRate*100)

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("[Chaos] Packet loss injection stopped")
			return
		case <-ticker.C:
			experiment.Observations = append(experiment.Observations, Observation{
				Timestamp:   time.Now(),
				Metric:      "packet_loss_rate",
				Value:       lossRate * 100,
				Expected:    0.0,
				Deviation:   lossRate * 100,
				Description: "Packet loss injected",
			})
		}
	}
}

// injectResourceExhaustion exhausts resources
func (ce *ChaosEngine) injectResourceExhaustion(ctx context.Context, experiment *ChaosExperiment) {
	log.Println("[Chaos] Injecting resource exhaustion")

	// Simulate CPU/memory exhaustion
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	cpuUsage := 50.0

	for {
		select {
		case <-ctx.Done():
			log.Println("[Chaos] Resource exhaustion stopped")
			return
		case <-ticker.C:
			cpuUsage += 10.0
			if cpuUsage > 95.0 {
				cpuUsage = 95.0
			}

			experiment.Observations = append(experiment.Observations, Observation{
				Timestamp:   time.Now(),
				Metric:      "cpu_usage",
				Value:       cpuUsage,
				Expected:    50.0,
				Deviation:   (cpuUsage - 50.0) / 50.0,
				Description: "Resource exhaustion injected",
			})

			// Check if auto-abort needed
			if experiment.AutoAbort && cpuUsage > 90.0 {
				log.Println("[Chaos] Auto-aborting due to high CPU usage")
				return
			}
		}
	}
}

// observeExperiment monitors experiment progress
func (ce *ChaosEngine) observeExperiment(ctx context.Context, experiment *ChaosExperiment) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// Run safety checks
			if err := ce.safetyMgr.RunSafetyChecks(ctx); err != nil {
				if experiment.AutoAbort {
					log.Printf("[Chaos] Auto-aborting experiment due to safety violation: %v", err)
					experiment.Status = "aborted"
					return
				}
			}

			// Check for unexpected behavior
			ce.detectUnexpectedBehavior(experiment)
		}
	}
}

// detectUnexpectedBehavior detects anomalies during experiment
func (ce *ChaosEngine) detectUnexpectedBehavior(experiment *ChaosExperiment) {
	// Analyze observations for anomalies
	if len(experiment.Observations) < 2 {
		return
	}

	lastObs := experiment.Observations[len(experiment.Observations)-1]

	// If deviation is too extreme, consider aborting
	if lastObs.Deviation > 5.0 && experiment.AutoAbort {
		log.Printf("[Chaos] Extreme deviation detected: %.2f", lastObs.Deviation)
		experiment.Status = "aborted"
	}
}

// GetExperiment returns experiment details
func (ce *ChaosEngine) GetExperiment(experimentID string) (*ChaosExperiment, error) {
	ce.expMu.RLock()
	defer ce.expMu.RUnlock()

	exp, exists := ce.experiments[experimentID]
	if !exists {
		return nil, fmt.Errorf("experiment not found: %s", experimentID)
	}

	return exp, nil
}

// Safety check implementations
func checkQuorumAvailable(ctx context.Context) error {
	// Check if quorum is available
	// Simulate check
	return nil
}

func checkNoIncidents(ctx context.Context) error {
	// Check for ongoing incidents
	// Simulate check
	return nil
}

func checkBusinessHours(ctx context.Context) error {
	// Check if within business hours
	hour := time.Now().Hour()
	if hour >= 9 && hour <= 17 {
		return nil
	}
	return fmt.Errorf("outside business hours")
}

func checkRecentBackup(ctx context.Context) error {
	// Check for recent backup
	// Simulate check
	return nil
}
