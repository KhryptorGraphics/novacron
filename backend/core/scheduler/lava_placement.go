package scheduler

import (
	"sync"
	"time"
)

// LifetimeClass represents VM lifetime class for LAVA
type LifetimeClass string

const (
	LifetimeClassShort   LifetimeClass = "short"
	LifetimeClassMedium  LifetimeClass = "medium"
	LifetimeClassLong    LifetimeClass = "long"
	LifetimeClassUnknown LifetimeClass = "unknown"
)

// LAVAHostState tracks host state for LAVA scoring
type LAVAHostState struct {
	NodeID         string
	LifetimeClass  LifetimeClass
	VMCountByClass map[LifetimeClass]int
	LastUpdated    time.Time
}

// LAVAScheduler implements Lifetime-Aware VM Allocation (LAVA)
type LAVAScheduler struct {
	enabled    bool
	mu         sync.RWMutex
	hostStates map[string]*LAVAHostState
}

// NewLAVAScheduler creates a new LAVA scheduler
func NewLAVAScheduler(enabled bool) *LAVAScheduler {
	return &LAVAScheduler{
		enabled:    enabled,
		hostStates: make(map[string]*LAVAHostState),
	}
}

// UpdateHostState updates host LAVA state
func (s *LAVAScheduler) UpdateHostState(nodeID string, lifetimeClass LifetimeClass, vmClasses map[LifetimeClass]int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.hostStates[nodeID] = &LAVAHostState{
		NodeID:         nodeID,
		LifetimeClass:  lifetimeClass,
		VMCountByClass: vmClasses,
		LastUpdated:    time.Now(),
	}
}

// LAVAScore returns LAVA score for placing vm on host (higher better, 0-1.0)
func (s *LAVAScheduler) LAVAScore(hostID string, vmLifetimeClass LifetimeClass) float64 {
	if !s.enabled {
		return 1.0 // Neutral if disabled
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	hostState, exists := s.hostStates[hostID]
	if !exists {
		return 0.5 // Neutral for unknown host
	}

	// LAVA scoring logic: prefer diversity in lifetime classes
	// Penalty for host dominated by same class as VM
	hostVMCount := 0
	for c, count := range hostState.VMCountByClass {
		hostVMCount += count
	}

	if hostVMCount == 0 {
		return 1.0 // Empty host perfect
	}

	var hostVMClass LifetimeClass
	count := 0
	for c, cnt := range hostState.VMCountByClass {
		if cnt > count {
			hostVMClass = c
			count = cnt
		}
	}

	if hostVMClass == vmLifetimeClass {
		// Same class dominance penalty
		dominance := float64(count) / float64(hostVMCount)
		return 1.0 - dominance*0.8 // Max 80% penalty
	}

	// Diversity bonus
	return 1.0
}

// IsEnabled returns if LAVA is enabled
func (s *LAVAScheduler) IsEnabled() bool {
	return s.enabled
}

// SetEnabled enables/disables LAVA
func (s *LAVAScheduler) SetEnabled(enabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enabled = enabled
}
