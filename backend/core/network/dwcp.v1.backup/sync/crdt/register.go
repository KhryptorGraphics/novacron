package crdt

import (
	"encoding/json"
	"sync"
	"time"
)

// LWWRegister implements a Last-Write-Wins Register CRDT
type LWWRegister struct {
	nodeID    string
	value     interface{}
	timestamp Timestamp
	mu        sync.RWMutex
}

// NewLWWRegister creates a new LWW-Register
func NewLWWRegister(nodeID string) *LWWRegister {
	return &LWWRegister{
		nodeID: nodeID,
		timestamp: Timestamp{
			Node:  nodeID,
			Clock: 0,
			Wall:  time.Now(),
		},
	}
}

// Set sets a new value with current timestamp
func (lww *LWWRegister) Set(value interface{}) {
	lww.mu.Lock()
	defer lww.mu.Unlock()

	lww.value = value
	lww.timestamp = Timestamp{
		Node:  lww.nodeID,
		Clock: lww.timestamp.Clock + 1,
		Wall:  time.Now(),
	}
}

// Get returns the current value
func (lww *LWWRegister) Get() interface{} {
	lww.mu.RLock()
	defer lww.mu.RUnlock()
	return lww.value
}

// Value returns the current value (CvRDT interface)
func (lww *LWWRegister) Value() interface{} {
	return lww.Get()
}

// Merge combines two LWW-Register states
func (lww *LWWRegister) Merge(other CvRDT) error {
	otherReg, ok := other.(*LWWRegister)
	if !ok {
		return ErrIncompatibleType
	}

	lww.mu.Lock()
	defer lww.mu.Unlock()

	// Compare timestamps
	otherTS := otherReg.timestamp
	if otherTS.Wall.After(lww.timestamp.Wall) ||
		(otherTS.Wall.Equal(lww.timestamp.Wall) && otherTS.Clock > lww.timestamp.Clock) ||
		(otherTS.Wall.Equal(lww.timestamp.Wall) && otherTS.Clock == lww.timestamp.Clock && otherTS.Node > lww.timestamp.Node) {
		lww.value = otherReg.value
		lww.timestamp = otherTS
	}

	return nil
}

// Compare determines partial order
func (lww *LWWRegister) Compare(other CvRDT) PartialOrder {
	otherReg, ok := other.(*LWWRegister)
	if !ok {
		return OrderingConcurrent
	}

	lww.mu.RLock()
	defer lww.mu.RUnlock()

	cmp := lww.timestamp.Compare(otherReg.timestamp)
	switch {
	case cmp < 0:
		return OrderingBefore
	case cmp > 0:
		return OrderingAfter
	default:
		return OrderingEqual
	}
}

// Clone creates a deep copy
func (lww *LWWRegister) Clone() CvRDT {
	lww.mu.RLock()
	defer lww.mu.RUnlock()

	clone := NewLWWRegister(lww.nodeID)
	clone.value = lww.value
	clone.timestamp = lww.timestamp
	return clone
}

// Marshal serializes the LWW-Register
func (lww *LWWRegister) Marshal() ([]byte, error) {
	lww.mu.RLock()
	defer lww.mu.RUnlock()

	data := struct {
		NodeID    string      `json:"node_id"`
		Value     interface{} `json:"value"`
		Timestamp Timestamp   `json:"timestamp"`
	}{
		NodeID:    lww.nodeID,
		Value:     lww.value,
		Timestamp: lww.timestamp,
	}
	return json.Marshal(data)
}

// Unmarshal deserializes the LWW-Register
func (lww *LWWRegister) Unmarshal(data []byte) error {
	lww.mu.Lock()
	defer lww.mu.Unlock()

	var parsed struct {
		NodeID    string      `json:"node_id"`
		Value     interface{} `json:"value"`
		Timestamp Timestamp   `json:"timestamp"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	lww.nodeID = parsed.NodeID
	lww.value = parsed.Value
	lww.timestamp = parsed.Timestamp
	return nil
}

// MVRegister implements a Multi-Value Register CRDT
type MVRegister struct {
	nodeID       string
	values       map[Timestamp]interface{}
	vectorClock  VectorClock
	mu           sync.RWMutex
}

// NewMVRegister creates a new MV-Register
func NewMVRegister(nodeID string) *MVRegister {
	return &MVRegister{
		nodeID:      nodeID,
		values:      make(map[Timestamp]interface{}),
		vectorClock: make(VectorClock),
	}
}

// Set sets a new value
func (mv *MVRegister) Set(value interface{}) {
	mv.mu.Lock()
	defer mv.mu.Unlock()

	// Increment vector clock
	mv.vectorClock.Increment(mv.nodeID)

	// Clear old values and set new one
	mv.values = make(map[Timestamp]interface{})
	ts := Timestamp{
		Node:  mv.nodeID,
		Clock: mv.vectorClock[mv.nodeID],
		Wall:  time.Now(),
	}
	mv.values[ts] = value
}

// Get returns all concurrent values
func (mv *MVRegister) Get() []interface{} {
	mv.mu.RLock()
	defer mv.mu.RUnlock()

	result := make([]interface{}, 0, len(mv.values))
	for _, value := range mv.values {
		result = append(result, value)
	}
	return result
}

// Value returns all concurrent values (CvRDT interface)
func (mv *MVRegister) Value() interface{} {
	return mv.Get()
}

// Merge combines two MV-Register states
func (mv *MVRegister) Merge(other CvRDT) error {
	otherReg, ok := other.(*MVRegister)
	if !ok {
		return ErrIncompatibleType
	}

	mv.mu.Lock()
	defer mv.mu.Unlock()

	// Merge vector clocks
	oldClock := mv.vectorClock.Clone()
	mv.vectorClock.Update(otherReg.vectorClock)

	// Determine which values to keep
	newValues := make(map[Timestamp]interface{})

	// Keep values that are not dominated by the other's vector clock
	for ts, value := range mv.values {
		tsClock := VectorClock{ts.Node: ts.Clock}
		if otherReg.vectorClock.Compare(tsClock) != OrderingAfter {
			newValues[ts] = value
		}
	}

	// Add values from other that are not dominated by our old clock
	for ts, value := range otherReg.values {
		tsClock := VectorClock{ts.Node: ts.Clock}
		if oldClock.Compare(tsClock) != OrderingAfter {
			newValues[ts] = value
		}
	}

	mv.values = newValues
	return nil
}

// Compare determines partial order
func (mv *MVRegister) Compare(other CvRDT) PartialOrder {
	otherReg, ok := other.(*MVRegister)
	if !ok {
		return OrderingConcurrent
	}

	mv.mu.RLock()
	defer mv.mu.RUnlock()

	return mv.vectorClock.Compare(otherReg.vectorClock)
}

// Clone creates a deep copy
func (mv *MVRegister) Clone() CvRDT {
	mv.mu.RLock()
	defer mv.mu.RUnlock()

	clone := NewMVRegister(mv.nodeID)
	clone.vectorClock = mv.vectorClock.Clone()
	for ts, value := range mv.values {
		clone.values[ts] = value
	}
	return clone
}

// Marshal serializes the MV-Register
func (mv *MVRegister) Marshal() ([]byte, error) {
	mv.mu.RLock()
	defer mv.mu.RUnlock()

	type valueEntry struct {
		Timestamp Timestamp   `json:"timestamp"`
		Value     interface{} `json:"value"`
	}

	values := make([]valueEntry, 0, len(mv.values))
	for ts, value := range mv.values {
		values = append(values, valueEntry{
			Timestamp: ts,
			Value:     value,
		})
	}

	data := struct {
		NodeID      string        `json:"node_id"`
		Values      []valueEntry  `json:"values"`
		VectorClock VectorClock   `json:"vector_clock"`
	}{
		NodeID:      mv.nodeID,
		Values:      values,
		VectorClock: mv.vectorClock,
	}
	return json.Marshal(data)
}

// Unmarshal deserializes the MV-Register
func (mv *MVRegister) Unmarshal(data []byte) error {
	mv.mu.Lock()
	defer mv.mu.Unlock()

	type valueEntry struct {
		Timestamp Timestamp   `json:"timestamp"`
		Value     interface{} `json:"value"`
	}

	var parsed struct {
		NodeID      string        `json:"node_id"`
		Values      []valueEntry  `json:"values"`
		VectorClock VectorClock   `json:"vector_clock"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	mv.nodeID = parsed.NodeID
	mv.vectorClock = parsed.VectorClock
	mv.values = make(map[Timestamp]interface{})

	for _, entry := range parsed.Values {
		mv.values[entry.Timestamp] = entry.Value
	}

	return nil
}
