package multiregion

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// PathRedundancy manages primary and backup paths with failover
type PathRedundancy struct {
	primary   *Route
	secondary []*Route
	monitor   *PathMonitor
	config    *RedundancyConfig
	mu        sync.RWMutex
}

// RedundancyConfig configures redundancy behavior
type RedundancyConfig struct {
	MaxSecondaryPaths int
	FailoverTimeout   time.Duration
	RetryAttempts     int
	RetryDelay        time.Duration
}

// PathMonitor monitors path health
type PathMonitor struct {
	probes         map[string]*ProbeResult
	probeInterval  time.Duration
	topology       *GlobalTopology
	mu             sync.RWMutex
	stopChan       chan struct{}
	wg             sync.WaitGroup
}

// ProbeResult contains path probe results
type ProbeResult struct {
	Success    bool
	Latency    time.Duration
	FailedHop  string
	Timestamp  time.Time
	PacketLoss float64
}

// PathHealth represents overall path health
type PathHealth struct {
	IsHealthy       bool
	Latency         time.Duration
	PacketLoss      float64
	LastProbe       time.Time
	ConsecutiveFails int
}

// NewPathRedundancy creates a new path redundancy manager
func NewPathRedundancy(primary *Route, topology *GlobalTopology) *PathRedundancy {
	monitor := &PathMonitor{
		probes:        make(map[string]*ProbeResult),
		probeInterval: 10 * time.Second,
		topology:      topology,
		stopChan:      make(chan struct{}),
	}

	pr := &PathRedundancy{
		primary:   primary,
		secondary: make([]*Route, 0),
		monitor:   monitor,
		config: &RedundancyConfig{
			MaxSecondaryPaths: 3,
			FailoverTimeout:   5 * time.Second,
			RetryAttempts:     3,
			RetryDelay:        100 * time.Millisecond,
		},
	}

	// Start monitoring
	go monitor.start()

	return pr
}

// AddSecondaryPath adds a backup path
func (pr *PathRedundancy) AddSecondaryPath(path *Route) error {
	pr.mu.Lock()
	defer pr.mu.Unlock()

	if len(pr.secondary) >= pr.config.MaxSecondaryPaths {
		return fmt.Errorf("maximum secondary paths (%d) reached", pr.config.MaxSecondaryPaths)
	}

	pr.secondary = append(pr.secondary, path)
	return nil
}

// SendWithFailover sends data with automatic failover to backup paths
func (pr *PathRedundancy) SendWithFailover(data []byte) error {
	// Try primary path first
	err := pr.send(pr.primary, data)
	if err == nil {
		return nil
	}

	// Primary failed, try secondary paths
	pr.mu.RLock()
	secondaryPaths := make([]*Route, len(pr.secondary))
	copy(secondaryPaths, pr.secondary)
	pr.mu.RUnlock()

	for _, backup := range secondaryPaths {
		if pr.monitor.IsHealthy(backup) {
			err = pr.sendWithRetry(backup, data)
			if err == nil {
				// Notify path failure
				pr.monitor.ReportFailure(pr.primary)
				return nil
			}
		}
	}

	return errors.New("all paths failed")
}

// send sends data along a specific path
func (pr *PathRedundancy) send(route *Route, data []byte) error {
	// In production, this would actually send data
	// For now, simulate based on path health

	if !pr.monitor.IsHealthy(route) {
		return errors.New("path unhealthy")
	}

	// Simulate transmission
	time.Sleep(route.Metric.Latency)

	return nil
}

// sendWithRetry sends data with retry logic
func (pr *PathRedundancy) sendWithRetry(route *Route, data []byte) error {
	var lastErr error

	for attempt := 0; attempt < pr.config.RetryAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(pr.config.RetryDelay)
		}

		err := pr.send(route, data)
		if err == nil {
			return nil
		}

		lastErr = err
	}

	return fmt.Errorf("send failed after %d attempts: %w", pr.config.RetryAttempts, lastErr)
}

// GetPrimaryPath returns the primary path
func (pr *PathRedundancy) GetPrimaryPath() *Route {
	pr.mu.RLock()
	defer pr.mu.RUnlock()
	return pr.primary
}

// GetSecondaryPaths returns all secondary paths
func (pr *PathRedundancy) GetSecondaryPaths() []*Route {
	pr.mu.RLock()
	defer pr.mu.RUnlock()

	paths := make([]*Route, len(pr.secondary))
	copy(paths, pr.secondary)
	return paths
}

// PromoteSecondaryPath promotes a secondary path to primary
func (pr *PathRedundancy) PromoteSecondaryPath(path *Route) error {
	pr.mu.Lock()
	defer pr.mu.Unlock()

	// Find the path in secondary
	found := false
	for i, secondary := range pr.secondary {
		if secondary == path {
			// Remove from secondary
			pr.secondary = append(pr.secondary[:i], pr.secondary[i+1:]...)
			found = true
			break
		}
	}

	if !found {
		return errors.New("path not found in secondary paths")
	}

	// Demote current primary to secondary
	pr.secondary = append(pr.secondary, pr.primary)

	// Promote new primary
	pr.primary = path

	return nil
}

// PathMonitor methods

// start begins the monitoring loop
func (pm *PathMonitor) start() {
	pm.wg.Add(1)
	defer pm.wg.Done()

	ticker := time.NewTicker(pm.probeInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pm.probeAllPaths()
		case <-pm.stopChan:
			return
		}
	}
}

// Stop stops the path monitor
func (pm *PathMonitor) Stop() {
	close(pm.stopChan)
	pm.wg.Wait()
}

// probeAllPaths probes all monitored paths
func (pm *PathMonitor) probeAllPaths() {
	pm.mu.RLock()
	paths := make(map[string]*ProbeResult)
	for path := range pm.probes {
		paths[path] = pm.probes[path]
	}
	pm.mu.RUnlock()

	// Probe each path
	for pathID := range paths {
		// In production, pathID would be used to identify the actual route
		// For now, we'll simulate probe results
		result := &ProbeResult{
			Success:   true,
			Latency:   10 * time.Millisecond,
			Timestamp: time.Now(),
		}

		pm.mu.Lock()
		pm.probes[pathID] = result
		pm.mu.Unlock()
	}
}

// ProbePath sends probes along a path to check connectivity
func (pm *PathMonitor) ProbePath(route *Route) *ProbeResult {
	startTime := time.Now()

	// Probe each hop in the path
	for _, hop := range route.Path {
		if !pm.probeHop(hop) {
			return &ProbeResult{
				Success:   false,
				FailedHop: hop,
				Timestamp: time.Now(),
			}
		}
	}

	latency := time.Since(startTime)

	result := &ProbeResult{
		Success:   true,
		Latency:   latency,
		Timestamp: time.Now(),
	}

	// Store result
	pathID := fmt.Sprintf("%s-%s", route.Path[0], route.Destination)
	pm.mu.Lock()
	pm.probes[pathID] = result
	pm.mu.Unlock()

	return result
}

// probeHop sends a probe to a specific hop
func (pm *PathMonitor) probeHop(hop string) bool {
	// In production, this would send ICMP echo or UDP probe
	// For now, check if region exists and is healthy

	region, err := pm.topology.GetRegion(hop)
	if err != nil {
		return false
	}

	// Check if region has healthy links
	links := pm.topology.GetOutgoingLinks(region.ID)
	for _, link := range links {
		if link.Health == HealthUp {
			return true
		}
	}

	return len(links) == 0 // If no links, consider it reachable (leaf node)
}

// IsHealthy checks if a path is healthy
func (pm *PathMonitor) IsHealthy(route *Route) bool {
	if route == nil {
		return false
	}

	pathID := fmt.Sprintf("%s-%s", route.Path[0], route.Destination)

	pm.mu.RLock()
	result, exists := pm.probes[pathID]
	pm.mu.RUnlock()

	if !exists {
		// No probe data, probe now
		result = pm.ProbePath(route)
	}

	// Check if probe is recent and successful
	if time.Since(result.Timestamp) > 2*pm.probeInterval {
		// Stale data, re-probe
		result = pm.ProbePath(route)
	}

	return result.Success && result.PacketLoss < 5.0 // Less than 5% packet loss
}

// ReportFailure reports a path failure
func (pm *PathMonitor) ReportFailure(route *Route) {
	if route == nil {
		return
	}

	pathID := fmt.Sprintf("%s-%s", route.Path[0], route.Destination)

	pm.mu.Lock()
	pm.probes[pathID] = &ProbeResult{
		Success:   false,
		Timestamp: time.Now(),
	}
	pm.mu.Unlock()

	// Mark links in path as potentially degraded
	for _, linkID := range route.Links {
		if link, err := pm.topology.GetLink(linkID); err == nil {
			link.mu.Lock()
			if link.Health == HealthUp {
				link.Health = HealthDegraded
			}
			link.mu.Unlock()
		}
	}
}

// GetPathHealth returns detailed health information for a path
func (pm *PathMonitor) GetPathHealth(route *Route) *PathHealth {
	if route == nil {
		return &PathHealth{IsHealthy: false}
	}

	pathID := fmt.Sprintf("%s-%s", route.Path[0], route.Destination)

	pm.mu.RLock()
	result, exists := pm.probes[pathID]
	pm.mu.RUnlock()

	if !exists {
		return &PathHealth{
			IsHealthy: false,
			LastProbe: time.Time{},
		}
	}

	// Calculate consecutive failures
	consecutiveFails := 0
	if !result.Success {
		consecutiveFails = 1 // In production, track this over time
	}

	return &PathHealth{
		IsHealthy:        result.Success,
		Latency:          result.Latency,
		PacketLoss:       result.PacketLoss,
		LastProbe:        result.Timestamp,
		ConsecutiveFails: consecutiveFails,
	}
}

// SetProbeInterval changes the probe interval
func (pm *PathMonitor) SetProbeInterval(interval time.Duration) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.probeInterval = interval
}

// ClearProbeData clears all probe results
func (pm *PathMonitor) ClearProbeData() {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.probes = make(map[string]*ProbeResult)
}
