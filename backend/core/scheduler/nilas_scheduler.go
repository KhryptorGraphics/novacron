package scheduler

import (
	"sync"
	"time"
)

// NILASConfig holds NILAS-specific configuration
type NILASConfig struct {
	Enabled             bool
	TemporalCostBuckets []time.Duration
}

// NILASScheduler implements non-invasive lifetime-aware scheduling as tie-breaker
type NILASScheduler struct {
	resourceAware *ResourceAwareScheduler
	predictor     LifetimePredictor
	config        NILASConfig
	hostLifetimeCache map[string]time.Time
	hostLifetimeMu    sync.RWMutex
	repredictInterval time.Duration
}

// NewNILASScheduler creates a new NILAS scheduler wrapping resource-aware scheduler
func NewNILASScheduler(resourceAware *ResourceAwareScheduler, predictor LifetimePredictor, config NILASConfig) *NILASScheduler {
	s := &NILASScheduler{
		resourceAware:     resourceAware,
		predictor:         predictor,
		config:            config,
		hostLifetimeCache: make(map[string]time.Time),
		repredictInterval: 5 * time.Minute,
	}
	return s
}

// ComputeTemporalCost computes temporal cost for NILAS tie-breaking
func (s *NILASScheduler) ComputeTemporalCost(vmID string, nodeID string, vmLifetimeClass LifetimeClass) float64 {
	if !s.config.Enabled {
		return 0.0 // Neutral if disabled
	}

	// Get host's average VM lifetime from cache
	s.hostLifetimeMu.RLock()
	hostAvgLifetime, exists := s.hostLifetimeCache[nodeID]
	s.hostLifetimeMu.RUnlock()

	if !exists {
		// Default to 1 hour if no data
		hostAvgLifetime = time.Hour
	}

	// Temporal cost bucketization based on NILAS paper
	// Map lifetime class to expected duration
	var expectedLifetime time.Duration
	switch vmLifetimeClass {
	case LifetimeClassShort:
		expectedLifetime = 30 * time.Minute
	case LifetimeClassMedium:
		expectedLifetime = 4 * time.Hour
	case LifetimeClassLong:
		expectedLifetime = 24 * time.Hour
	default: // LifetimeClassUnknown
		expectedLifetime = time.Hour
	}

	// Compute temporal cost as ratio of expected to host average
	// Lower cost means better fit
	temporalCost := float64(expectedLifetime) / float64(hostAvgLifetime)

	// Normalize to 0-1 range
	if temporalCost > 2.0 {
		temporalCost = 2.0
	}

	return temporalCost / 2.0 // Scale to 0-1
}

// UpdateHostLifetimeCache updates the cached average lifetime for a host
func (s *NILASScheduler) UpdateHostLifetimeCache(nodeID string, avgLifetime time.Duration) {
	s.hostLifetimeMu.Lock()
	defer s.hostLifetimeMu.Unlock()
	s.hostLifetimeCache[nodeID] = avgLifetime
}

// IsEnabled returns if NILAS is enabled
func (s *NILASScheduler) IsEnabled() bool {
	return s.config.Enabled
}