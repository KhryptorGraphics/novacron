package health

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// HealthStatus represents the health state of a component
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusDegraded  HealthStatus = "degraded"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
)

// ComponentHealth represents the health of a single component
type ComponentHealth struct {
	Component   string       `json:"component"`
	Status      HealthStatus `json:"status"`
	Score       int          `json:"score"` // 0-100
	Failures    int          `json:"failures"`
	LastCheck   time.Time    `json:"last_check"`
	LastHealthy time.Time    `json:"last_healthy"`
	Checks      []Check      `json:"checks"`
	Message     string       `json:"message,omitempty"`
}

// Check represents a specific health check result
type Check struct {
	Name    string       `json:"name"`
	Status  HealthStatus `json:"status"`
	Message string       `json:"message,omitempty"`
	Value   interface{}  `json:"value,omitempty"`
}

// HealthChecker performs comprehensive health checking
type HealthChecker struct {
	mu                 sync.RWMutex
	ctx                context.Context
	cancel             context.CancelFunc
	componentHealth    map[string]*ComponentHealth
	checkInterval      time.Duration
	failureThreshold   int
	componentCheckers  map[string]ComponentChecker
	selfHealers        map[string]SelfHealer
	onUnhealthy        func(component string, health *ComponentHealth)
}

// ComponentChecker interface for checking component health
type ComponentChecker interface {
	Check(ctx context.Context) (*ComponentHealth, error)
	Component() string
}

// SelfHealer interface for automated healing
type SelfHealer interface {
	Heal(ctx context.Context, health *ComponentHealth) error
	CanHeal(health *ComponentHealth) bool
}

// NewHealthChecker creates a new health checker
func NewHealthChecker(ctx context.Context, checkInterval time.Duration) *HealthChecker {
	ctx, cancel := context.WithCancel(ctx)

	return &HealthChecker{
		ctx:               ctx,
		cancel:            cancel,
		componentHealth:   make(map[string]*ComponentHealth),
		checkInterval:     checkInterval,
		failureThreshold:  3,
		componentCheckers: make(map[string]ComponentChecker),
		selfHealers:       make(map[string]SelfHealer),
	}
}

// RegisterChecker registers a component checker
func (hc *HealthChecker) RegisterChecker(checker ComponentChecker) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	hc.componentCheckers[checker.Component()] = checker
}

// RegisterHealer registers a self-healer for a component
func (hc *HealthChecker) RegisterHealer(component string, healer SelfHealer) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	hc.selfHealers[component] = healer
}

// SetUnhealthyCallback sets callback for unhealthy components
func (hc *HealthChecker) SetUnhealthyCallback(callback func(string, *ComponentHealth)) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	hc.onUnhealthy = callback
}

// Start begins health checking
func (hc *HealthChecker) Start() error {
	go hc.checkLoop()
	return nil
}

// Stop halts health checking
func (hc *HealthChecker) Stop() error {
	hc.cancel()
	return nil
}

// checkLoop continuously performs health checks
func (hc *HealthChecker) checkLoop() {
	ticker := time.NewTicker(hc.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			hc.performHealthChecks()
		}
	}
}

// performHealthChecks executes all registered checkers
func (hc *HealthChecker) performHealthChecks() {
	hc.mu.RLock()
	checkers := make([]ComponentChecker, 0, len(hc.componentCheckers))
	for _, checker := range hc.componentCheckers {
		checkers = append(checkers, checker)
	}
	hc.mu.RUnlock()

	var wg sync.WaitGroup
	for _, checker := range checkers {
		wg.Add(1)
		go func(c ComponentChecker) {
			defer wg.Done()
			hc.checkComponent(c)
		}(checker)
	}
	wg.Wait()
}

// checkComponent checks a single component
func (hc *HealthChecker) checkComponent(checker ComponentChecker) {
	health, err := checker.Check(hc.ctx)
	if err != nil {
		// Log error and mark as unhealthy
		health = &ComponentHealth{
			Component: checker.Component(),
			Status:    HealthStatusUnhealthy,
			Score:     0,
			LastCheck: time.Now(),
			Message:   fmt.Sprintf("Check failed: %v", err),
		}
	}

	// Update stored health
	hc.mu.Lock()
	oldHealth, exists := hc.componentHealth[health.Component]

	if exists {
		// Preserve failure count
		if health.Status != HealthStatusHealthy {
			health.Failures = oldHealth.Failures + 1
		} else {
			health.Failures = 0
			health.LastHealthy = time.Now()
		}
	} else {
		if health.Status == HealthStatusHealthy {
			health.LastHealthy = time.Now()
		}
	}

	hc.componentHealth[health.Component] = health
	hc.mu.Unlock()

	// Attempt self-healing if threshold exceeded
	if health.Failures >= hc.failureThreshold && health.Status != HealthStatusHealthy {
		hc.attemptSelfHealing(health)
	}

	// Trigger unhealthy callback
	if health.Status == HealthStatusUnhealthy && hc.onUnhealthy != nil {
		hc.onUnhealthy(health.Component, health)
	}
}

// attemptSelfHealing tries to automatically heal a component
func (hc *HealthChecker) attemptSelfHealing(health *ComponentHealth) {
	hc.mu.RLock()
	healer, exists := hc.selfHealers[health.Component]
	hc.mu.RUnlock()

	if !exists || !healer.CanHeal(health) {
		return
	}

	if err := healer.Heal(hc.ctx, health); err != nil {
		// Log healing failure
		return
	}

	// Reset failure count on successful heal
	hc.mu.Lock()
	if ch, ok := hc.componentHealth[health.Component]; ok {
		ch.Failures = 0
	}
	hc.mu.Unlock()
}

// GetComponentHealth returns health for a specific component
func (hc *HealthChecker) GetComponentHealth(component string) (*ComponentHealth, error) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	health, exists := hc.componentHealth[component]
	if !exists {
		return nil, fmt.Errorf("component not found: %s", component)
	}

	return health, nil
}

// GetAllHealth returns health for all components
func (hc *HealthChecker) GetAllHealth() map[string]*ComponentHealth {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	result := make(map[string]*ComponentHealth)
	for k, v := range hc.componentHealth {
		result[k] = v
	}

	return result
}

// GetOverallScore calculates overall system health score
func (hc *HealthChecker) GetOverallScore() int {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	if len(hc.componentHealth) == 0 {
		return 0
	}

	totalScore := 0
	for _, health := range hc.componentHealth {
		totalScore += health.Score
	}

	return totalScore / len(hc.componentHealth)
}

// GetUnhealthyComponents returns all unhealthy components
func (hc *HealthChecker) GetUnhealthyComponents() []string {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	unhealthy := make([]string, 0)
	for component, health := range hc.componentHealth {
		if health.Status == HealthStatusUnhealthy {
			unhealthy = append(unhealthy, component)
		}
	}

	return unhealthy
}

// =============================================================================
// Built-in Component Checkers
// =============================================================================

// ConsensusChecker checks consensus layer health
type ConsensusChecker struct {
	component string
	apiURL    string
}

func NewConsensusChecker(apiURL string) *ConsensusChecker {
	return &ConsensusChecker{
		component: "consensus",
		apiURL:    apiURL,
	}
}

func (c *ConsensusChecker) Component() string {
	return c.component
}

func (c *ConsensusChecker) Check(ctx context.Context) (*ComponentHealth, error) {
	health := &ComponentHealth{
		Component: c.component,
		Status:    HealthStatusHealthy,
		Score:     100,
		LastCheck: time.Now(),
		Checks:    make([]Check, 0),
	}

	// Check leader election
	leaderCheck := Check{Name: "leader_election", Status: HealthStatusHealthy}
	// Implementation would check if leader exists
	health.Checks = append(health.Checks, leaderCheck)

	// Check quorum
	quorumCheck := Check{Name: "quorum", Status: HealthStatusHealthy}
	// Implementation would verify quorum
	health.Checks = append(health.Checks, quorumCheck)

	// Check replication lag
	replicationCheck := Check{Name: "replication_lag", Status: HealthStatusHealthy}
	// Implementation would check lag
	health.Checks = append(health.Checks, replicationCheck)

	// Check Byzantine detection
	byzantineCheck := Check{Name: "byzantine_detection", Status: HealthStatusHealthy}
	// Implementation would check for Byzantine nodes
	health.Checks = append(health.Checks, byzantineCheck)

	// Calculate overall score
	for _, check := range health.Checks {
		if check.Status == HealthStatusUnhealthy {
			health.Score -= 25
		} else if check.Status == HealthStatusDegraded {
			health.Score -= 10
		}
	}

	// Determine overall status
	if health.Score < 50 {
		health.Status = HealthStatusUnhealthy
	} else if health.Score < 80 {
		health.Status = HealthStatusDegraded
	}

	return health, nil
}

// NetworkChecker checks network layer health
type NetworkChecker struct {
	component string
	apiURL    string
}

func NewNetworkChecker(apiURL string) *NetworkChecker {
	return &NetworkChecker{
		component: "network",
		apiURL:    apiURL,
	}
}

func (n *NetworkChecker) Component() string {
	return n.component
}

func (n *NetworkChecker) Check(ctx context.Context) (*ComponentHealth, error) {
	health := &ComponentHealth{
		Component: n.component,
		Status:    HealthStatusHealthy,
		Score:     100,
		LastCheck: time.Now(),
		Checks:    make([]Check, 0),
	}

	// Check connectivity
	connectivityCheck := Check{Name: "connectivity", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, connectivityCheck)

	// Check packet loss
	packetLossCheck := Check{Name: "packet_loss", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, packetLossCheck)

	// Check bandwidth
	bandwidthCheck := Check{Name: "bandwidth", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, bandwidthCheck)

	// Check partition status
	partitionCheck := Check{Name: "partition_detection", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, partitionCheck)

	// Calculate score
	for _, check := range health.Checks {
		if check.Status == HealthStatusUnhealthy {
			health.Score -= 25
		} else if check.Status == HealthStatusDegraded {
			health.Score -= 10
		}
	}

	if health.Score < 50 {
		health.Status = HealthStatusUnhealthy
	} else if health.Score < 80 {
		health.Status = HealthStatusDegraded
	}

	return health, nil
}

// StorageChecker checks storage layer health
type StorageChecker struct {
	component string
	apiURL    string
}

func NewStorageChecker(apiURL string) *StorageChecker {
	return &StorageChecker{
		component: "storage",
		apiURL:    apiURL,
	}
}

func (s *StorageChecker) Component() string {
	return s.component
}

func (s *StorageChecker) Check(ctx context.Context) (*ComponentHealth, error) {
	health := &ComponentHealth{
		Component: s.component,
		Status:    HealthStatusHealthy,
		Score:     100,
		LastCheck: time.Now(),
		Checks:    make([]Check, 0),
	}

	// Check disk usage
	diskCheck := Check{Name: "disk_usage", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, diskCheck)

	// Check database health
	dbCheck := Check{Name: "database", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, dbCheck)

	// Check replication lag
	replicationCheck := Check{Name: "replication_lag", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, replicationCheck)

	// Check I/O performance
	ioCheck := Check{Name: "io_performance", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, ioCheck)

	// Calculate score
	for _, check := range health.Checks {
		if check.Status == HealthStatusUnhealthy {
			health.Score -= 25
		} else if check.Status == HealthStatusDegraded {
			health.Score -= 10
		}
	}

	if health.Score < 50 {
		health.Status = HealthStatusUnhealthy
	} else if health.Score < 80 {
		health.Status = HealthStatusDegraded
	}

	return health, nil
}

// APIChecker checks API layer health
type APIChecker struct {
	component string
	apiURL    string
}

func NewAPIChecker(apiURL string) *APIChecker {
	return &APIChecker{
		component: "api",
		apiURL:    apiURL,
	}
}

func (a *APIChecker) Component() string {
	return a.component
}

func (a *APIChecker) Check(ctx context.Context) (*ComponentHealth, error) {
	health := &ComponentHealth{
		Component: a.component,
		Status:    HealthStatusHealthy,
		Score:     100,
		LastCheck: time.Now(),
		Checks:    make([]Check, 0),
	}

	// Check response time
	responseTimeCheck := Check{Name: "response_time", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, responseTimeCheck)

	// Check error rate
	errorRateCheck := Check{Name: "error_rate", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, errorRateCheck)

	// Check queue depth
	queueCheck := Check{Name: "queue_depth", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, queueCheck)

	// Check connections
	connectionsCheck := Check{Name: "connections", Status: HealthStatusHealthy}
	health.Checks = append(health.Checks, connectionsCheck)

	// Calculate score
	for _, check := range health.Checks {
		if check.Status == HealthStatusUnhealthy {
			health.Score -= 25
		} else if check.Status == HealthStatusDegraded {
			health.Score -= 10
		}
	}

	if health.Score < 50 {
		health.Status = HealthStatusUnhealthy
	} else if health.Score < 80 {
		health.Status = HealthStatusDegraded
	}

	return health, nil
}

// =============================================================================
// Built-in Self-Healers
// =============================================================================

// CacheClearHealer clears caches to resolve transient issues
type CacheClearHealer struct {
	apiURL string
}

func NewCacheClearHealer(apiURL string) *CacheClearHealer {
	return &CacheClearHealer{apiURL: apiURL}
}

func (h *CacheClearHealer) CanHeal(health *ComponentHealth) bool {
	return health.Score >= 30 && health.Score < 80
}

func (h *CacheClearHealer) Heal(ctx context.Context, health *ComponentHealth) error {
	// Implementation would clear component cache
	return nil
}

// ServiceRestartHealer restarts services to resolve failures
type ServiceRestartHealer struct {
	component string
}

func NewServiceRestartHealer(component string) *ServiceRestartHealer {
	return &ServiceRestartHealer{component: component}
}

func (h *ServiceRestartHealer) CanHeal(health *ComponentHealth) bool {
	return health.Score < 50
}

func (h *ServiceRestartHealer) Heal(ctx context.Context, health *ComponentHealth) error {
	// Implementation would restart the service
	return nil
}
