package scheduler

import (
	"log"
	"sort"
	"sync"
	"time"
)

// LARSMigratableVM is an interface for VMs that can be sorted for migration
// This avoids import cycles with the vm package
type LARSMigratableVM interface {
	ID() string
	GetCreatedAt() time.Time
	// GetStartedAt returns the time the VM was started, or nil if not started
	GetStartedAtTime() *time.Time
}

// LARSRescheduler implements Lifetime-Aware Rescheduling (LARS) for migration ordering
type LARSRescheduler struct {
	enabled   bool
	predictor LifetimePredictor
	mu        sync.RWMutex
}

// NewLARSRescheduler creates a new LARS rescheduler
func NewLARSRescheduler(enabled bool, predictor LifetimePredictor) *LARSRescheduler {
	return &LARSRescheduler{
		enabled:   enabled,
		predictor: predictor,
	}
}

// SortVMsForMigration sorts VMs for migration by predicted remaining lifetime (shortest first)
func (s *LARSRescheduler) SortVMsForMigration(vms []LARSMigratableVM) []LARSMigratableVM {
	if !s.enabled || s.predictor == nil {
		// Return as-is if disabled
		return vms
	}

	// Create sortable list with predicted lifetimes
	type vmLifetime struct {
		vm       LARSMigratableVM
		lifetime time.Duration
	}

	sortable := make([]vmLifetime, len(vms))
	for i, vm := range vms {
		// TODO: Use vm.PredictedLifetime once VM struct is updated with lifetime fields
		// For now, use uptime as a proxy - VMs with longer uptime are prioritized for migration
		var predLifetime time.Duration
		startedAt := vm.GetStartedAtTime()
		if startedAt != nil {
			uptime := time.Since(*startedAt)
			predLifetime = uptime // Use uptime as proxy for remaining lifetime
		} else {
			predLifetime = time.Hour // Default fallback
		}

		sortable[i] = vmLifetime{
			vm:       vm,
			lifetime: predLifetime,
		}
	}

	// Sort by predicted lifetime ascending (shortest first)
	sort.Slice(sortable, func(i, j int) bool {
		return sortable[i].lifetime < sortable[j].lifetime
	})

	// Extract sorted VMs
	sortedVMs := make([]LARSMigratableVM, len(vms))
	for i, vl := range sortable {
		sortedVMs[i] = vl.vm
	}

	log.Printf("LARS reordered %d VMs for migration by predicted lifetime", len(vms))

	return sortedVMs
}

// IsEnabled returns if LARS is enabled
func (s *LARSRescheduler) IsEnabled() bool {
	return s.enabled
}

// SetEnabled enables/disables LARS
func (s *LARSRescheduler) SetEnabled(enabled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enabled = enabled
}
