package sync

import (
	"encoding/json"
	"sync"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync/crdt"
)

// VectorClockManager manages vector clocks for the ASS engine
type VectorClockManager struct {
	nodeID string
	clock  crdt.VectorClock
	mu     sync.RWMutex
}

// NewVectorClockManager creates a new vector clock manager
func NewVectorClockManager(nodeID string) *VectorClockManager {
	return &VectorClockManager{
		nodeID: nodeID,
		clock:  make(crdt.VectorClock),
	}
}

// Increment increments the local clock
func (vcm *VectorClockManager) Increment() {
	vcm.mu.Lock()
	defer vcm.mu.Unlock()
	vcm.clock.Increment(vcm.nodeID)
}

// Update updates the clock with another clock
func (vcm *VectorClockManager) Update(other crdt.VectorClock) {
	vcm.mu.Lock()
	defer vcm.mu.Unlock()
	vcm.clock.Update(other)
	vcm.clock.Increment(vcm.nodeID)
}

// Get returns a clone of the current clock
func (vcm *VectorClockManager) Get() crdt.VectorClock {
	vcm.mu.RLock()
	defer vcm.mu.RUnlock()
	return vcm.clock.Clone()
}

// Compare compares with another clock
func (vcm *VectorClockManager) Compare(other crdt.VectorClock) crdt.PartialOrder {
	vcm.mu.RLock()
	defer vcm.mu.RUnlock()
	return vcm.clock.Compare(other)
}

// Marshal serializes the vector clock
func (vcm *VectorClockManager) Marshal() ([]byte, error) {
	vcm.mu.RLock()
	defer vcm.mu.RUnlock()
	return vcm.clock.Marshal()
}

// Unmarshal deserializes the vector clock
func (vcm *VectorClockManager) Unmarshal(data []byte) error {
	vcm.mu.Lock()
	defer vcm.mu.Unlock()
	return vcm.clock.Unmarshal(data)
}

// CausalTracker tracks causal ordering of events
type CausalTracker struct {
	nodeID         string
	vectorClock    *VectorClockManager
	deliveredEvents map[string]struct{}
	pendingEvents  map[string]*CausalEvent
	mu             sync.RWMutex
}

// CausalEvent represents an event with causal dependencies
type CausalEvent struct {
	ID          string              `json:"id"`
	Type        string              `json:"type"`
	Timestamp   crdt.Timestamp      `json:"timestamp"`
	VectorClock crdt.VectorClock    `json:"vector_clock"`
	Payload     interface{}         `json:"payload"`
	Dependencies []string           `json:"dependencies"`
}

// NewCausalTracker creates a new causal tracker
func NewCausalTracker(nodeID string) *CausalTracker {
	return &CausalTracker{
		nodeID:          nodeID,
		vectorClock:     NewVectorClockManager(nodeID),
		deliveredEvents: make(map[string]struct{}),
		pendingEvents:   make(map[string]*CausalEvent),
	}
}

// TrackEvent processes an event and ensures causal ordering
func (ct *CausalTracker) TrackEvent(event *CausalEvent) (bool, error) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	// Check if already delivered
	if _, delivered := ct.deliveredEvents[event.ID]; delivered {
		return false, nil
	}

	// Check if can deliver now
	if ct.canDeliverInternal(event) {
		ct.deliverEventInternal(event)
		ct.checkPendingEvents()
		return true, nil
	}

	// Buffer event for later
	ct.pendingEvents[event.ID] = event
	return false, nil
}

func (ct *CausalTracker) canDeliverInternal(event *CausalEvent) bool {
	// Check all dependencies are satisfied
	for _, depID := range event.Dependencies {
		if _, delivered := ct.deliveredEvents[depID]; !delivered {
			return false
		}
	}

	// Check vector clock causality
	localClock := ct.vectorClock.Get()
	eventClock := event.VectorClock

	for node, clock := range eventClock {
		if node == event.Timestamp.Node {
			// For the originating node, clock should be exactly next
			if clock != localClock[node]+1 {
				return false
			}
		} else {
			// For other nodes, should not exceed local clock
			if clock > localClock[node] {
				return false
			}
		}
	}

	return true
}

func (ct *CausalTracker) deliverEventInternal(event *CausalEvent) {
	ct.deliveredEvents[event.ID] = struct{}{}
	ct.vectorClock.Update(event.VectorClock)
	delete(ct.pendingEvents, event.ID)
}

func (ct *CausalTracker) checkPendingEvents() {
	delivered := true
	for delivered {
		delivered = false
		for id, event := range ct.pendingEvents {
			if ct.canDeliverInternal(event) {
				ct.deliverEventInternal(event)
				delivered = true
				break
			}
		}
	}
}

// CreateEvent creates a new causal event
func (ct *CausalTracker) CreateEvent(eventType string, payload interface{}, dependencies []string) *CausalEvent {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	ct.vectorClock.Increment()

	event := &CausalEvent{
		ID:           generateEventID(ct.nodeID),
		Type:         eventType,
		VectorClock:  ct.vectorClock.Get(),
		Timestamp:    crdt.Timestamp{Node: ct.nodeID, Clock: ct.vectorClock.clock[ct.nodeID]},
		Payload:      payload,
		Dependencies: dependencies,
	}

	return event
}

// GetPendingCount returns the number of pending events
func (ct *CausalTracker) GetPendingCount() int {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	return len(ct.pendingEvents)
}

// GetVectorClock returns the current vector clock
func (ct *CausalTracker) GetVectorClock() crdt.VectorClock {
	return ct.vectorClock.Get()
}

func generateEventID(nodeID string) string {
	// In production, use UUID or similar
	return nodeID + "-" + string(rune(len(nodeID)))
}

// Marshal serializes the causal event
func (ce *CausalEvent) Marshal() ([]byte, error) {
	return json.Marshal(ce)
}

// Unmarshal deserializes the causal event
func (ce *CausalEvent) Unmarshal(data []byte) error {
	return json.Unmarshal(data, ce)
}
