package consensus

import (
	"sync"
	"time"
)

// EventualConsistency implements CRDT-based eventual consistency
type EventualConsistency struct {
	mu sync.RWMutex

	nodeID       string
	stateMachine StateMachine

	// CRDT state
	lwwMap     *LWWMap         // Last-Write-Wins map
	orSet      *ORSet          // Observed-Remove set
	pnCounter  *PNCounter      // Positive-Negative counter
	vectorClock *VectorClock

	// Anti-entropy
	gossipInterval time.Duration
	peers          []string
}

// LWWMap is a Last-Write-Wins map CRDT
type LWWMap struct {
	mu     sync.RWMutex
	values map[string]*LWWValue
}

// LWWValue represents a value with timestamp
type LWWValue struct {
	Value     []byte
	Timestamp Timestamp
	NodeID    string
}

// ORSet is an Observed-Remove set CRDT
type ORSet struct {
	mu       sync.RWMutex
	elements map[string]map[string]bool // element -> {unique_tag -> bool}
	tombstones map[string]map[string]bool
}

// PNCounter is a Positive-Negative counter CRDT
type PNCounter struct {
	mu       sync.RWMutex
	positive map[string]int64
	negative map[string]int64
}

// VectorClock tracks causality
type VectorClock struct {
	mu    sync.RWMutex
	clock map[string]uint64
}

// Timestamp represents a logical timestamp
type Timestamp struct {
	Wall    int64  // Wall clock time
	Logical uint64 // Logical clock
	NodeID  string
}

// NewTimestamp creates a new timestamp
func NewTimestamp() Timestamp {
	return Timestamp{
		Wall:    time.Now().UnixNano(),
		Logical: 0,
	}
}

// After checks if this timestamp is after another
func (t Timestamp) After(other Timestamp) bool {
	if t.Wall != other.Wall {
		return t.Wall > other.Wall
	}
	if t.Logical != other.Logical {
		return t.Logical > other.Logical
	}
	return t.NodeID > other.NodeID
}

// NewEventualConsistency creates a new eventual consistency instance
func NewEventualConsistency(nodeID string, sm StateMachine) *EventualConsistency {
	return &EventualConsistency{
		nodeID:         nodeID,
		stateMachine:   sm,
		lwwMap:         NewLWWMap(),
		orSet:          NewORSet(),
		pnCounter:      NewPNCounter(),
		vectorClock:    NewVectorClock(),
		gossipInterval: 1 * time.Second,
	}
}

// NewLWWMap creates a new LWW map
func NewLWWMap() *LWWMap {
	return &LWWMap{
		values: make(map[string]*LWWValue),
	}
}

// NewORSet creates a new OR set
func NewORSet() *ORSet {
	return &ORSet{
		elements:   make(map[string]map[string]bool),
		tombstones: make(map[string]map[string]bool),
	}
}

// NewPNCounter creates a new PN counter
func NewPNCounter() *PNCounter {
	return &PNCounter{
		positive: make(map[string]int64),
		negative: make(map[string]int64),
	}
}

// NewVectorClock creates a new vector clock
func NewVectorClock() *VectorClock {
	return &VectorClock{
		clock: make(map[string]uint64),
	}
}

// Update performs an eventual consistency update
func (ec *EventualConsistency) Update(key string, value []byte) error {
	timestamp := NewTimestamp()
	timestamp.NodeID = ec.nodeID

	ec.lwwMap.Set(key, value, timestamp, ec.nodeID)
	ec.vectorClock.Increment(ec.nodeID)

	// Apply to state machine
	cmd := Command{
		Type:      "write",
		Key:       key,
		Value:     value,
		Timestamp: timestamp,
	}

	_, err := ec.stateMachine.Apply(cmd)
	return err
}

// Get retrieves a value
func (ec *EventualConsistency) Get(key string) ([]byte, bool) {
	return ec.lwwMap.Get(key)
}

// Set sets a value in LWW map
func (lww *LWWMap) Set(key string, value []byte, timestamp Timestamp, nodeID string) {
	lww.mu.Lock()
	defer lww.mu.Unlock()

	existing, exists := lww.values[key]
	if !exists || timestamp.After(existing.Timestamp) {
		lww.values[key] = &LWWValue{
			Value:     value,
			Timestamp: timestamp,
			NodeID:    nodeID,
		}
	}
}

// Get retrieves a value from LWW map
func (lww *LWWMap) Get(key string) ([]byte, bool) {
	lww.mu.RLock()
	defer lww.mu.RUnlock()

	val, exists := lww.values[key]
	if !exists {
		return nil, false
	}
	return val.Value, true
}

// Merge merges another LWW map into this one
func (lww *LWWMap) Merge(other *LWWMap) {
	lww.mu.Lock()
	defer lww.mu.Unlock()

	other.mu.RLock()
	defer other.mu.RUnlock()

	for key, otherVal := range other.values {
		existing, exists := lww.values[key]
		if !exists || otherVal.Timestamp.After(existing.Timestamp) {
			lww.values[key] = otherVal
		}
	}
}

// Add adds an element to OR set
func (ors *ORSet) Add(element string, tag string) {
	ors.mu.Lock()
	defer ors.mu.Unlock()

	if ors.elements[element] == nil {
		ors.elements[element] = make(map[string]bool)
	}
	ors.elements[element][tag] = true
}

// Remove removes an element from OR set
func (ors *ORSet) Remove(element string) {
	ors.mu.Lock()
	defer ors.mu.Unlock()

	if tags, exists := ors.elements[element]; exists {
		if ors.tombstones[element] == nil {
			ors.tombstones[element] = make(map[string]bool)
		}
		for tag := range tags {
			ors.tombstones[element][tag] = true
		}
	}
}

// Contains checks if element is in OR set
func (ors *ORSet) Contains(element string) bool {
	ors.mu.RLock()
	defer ors.mu.RUnlock()

	tags, exists := ors.elements[element]
	if !exists {
		return false
	}

	tombstones := ors.tombstones[element]
	for tag := range tags {
		if !tombstones[tag] {
			return true
		}
	}

	return false
}

// Increment increments PN counter
func (pn *PNCounter) Increment(nodeID string, delta int64) {
	pn.mu.Lock()
	defer pn.mu.Unlock()

	if delta >= 0 {
		pn.positive[nodeID] += delta
	} else {
		pn.negative[nodeID] += -delta
	}
}

// Value returns the counter value
func (pn *PNCounter) Value() int64 {
	pn.mu.RLock()
	defer pn.mu.RUnlock()

	var pos, neg int64
	for _, v := range pn.positive {
		pos += v
	}
	for _, v := range pn.negative {
		neg += v
	}

	return pos - neg
}

// Increment increments vector clock
func (vc *VectorClock) Increment(nodeID string) {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	vc.clock[nodeID]++
}

// Update updates vector clock with another
func (vc *VectorClock) Update(other *VectorClock) {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	other.mu.RLock()
	defer other.mu.RUnlock()

	for nodeID, otherTime := range other.clock {
		if vc.clock[nodeID] < otherTime {
			vc.clock[nodeID] = otherTime
		}
	}
}

// HappensBefore checks if this clock happens before another
func (vc *VectorClock) HappensBefore(other *VectorClock) bool {
	vc.mu.RLock()
	defer vc.mu.RUnlock()

	other.mu.RLock()
	defer other.mu.RUnlock()

	lessThanOrEqual := true
	strictlyLess := false

	// Check all nodes in this clock
	for nodeID, myTime := range vc.clock {
		otherTime := other.clock[nodeID]
		if myTime > otherTime {
			lessThanOrEqual = false
			break
		}
		if myTime < otherTime {
			strictlyLess = true
		}
	}

	// Also check nodes only in other clock
	for nodeID, otherTime := range other.clock {
		if _, exists := vc.clock[nodeID]; !exists {
			if otherTime > 0 {
				strictlyLess = true
			}
		}
	}

	return lessThanOrEqual && strictlyLess
}


// LoadSnapshot loads a snapshot
func (ec *EventualConsistency) LoadSnapshot(snapshot *Snapshot) error {
	ec.mu.Lock()
	defer ec.mu.Unlock()

	return ec.stateMachine.Restore(snapshot)
}

// GossipState performs anti-entropy gossip
func (ec *EventualConsistency) GossipState() {
	ticker := time.NewTicker(ec.gossipInterval)
	defer ticker.Stop()

	for range ticker.C {
		ec.performGossip()
	}
}

// performGossip performs one round of gossip
func (ec *EventualConsistency) performGossip() {
	// Select random peer and exchange state
	// In real implementation, would send state to peer and merge responses

	ec.mu.RLock()
	defer ec.mu.RUnlock()

	// Simulate merging with peer state
	// In reality, would receive peer's LWWMap, ORSet, etc. and merge
}
