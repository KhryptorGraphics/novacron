package crdt

import (
	"encoding/json"
	"sync"
)

// GCounter implements a Grow-only Counter CRDT
type GCounter struct {
	nodeID  string
	payload map[string]uint64
	mu      sync.RWMutex
}

// NewGCounter creates a new G-Counter
func NewGCounter(nodeID string) *GCounter {
	return &GCounter{
		nodeID:  nodeID,
		payload: make(map[string]uint64),
	}
}

// Increment increments the counter for this node
func (gc *GCounter) Increment(delta uint64) {
	gc.mu.Lock()
	defer gc.mu.Unlock()
	gc.payload[gc.nodeID] += delta
}

// Value returns the current counter value (sum of all node counters)
func (gc *GCounter) Value() interface{} {
	gc.mu.RLock()
	defer gc.mu.RUnlock()

	var sum uint64
	for _, value := range gc.payload {
		sum += value
	}
	return sum
}

// Merge combines two G-Counter states
func (gc *GCounter) Merge(other CvRDT) error {
	otherCounter, ok := other.(*GCounter)
	if !ok {
		return ErrIncompatibleType
	}

	gc.mu.Lock()
	defer gc.mu.Unlock()

	for node, otherValue := range otherCounter.payload {
		currentValue := gc.payload[node]
		if otherValue > currentValue {
			gc.payload[node] = otherValue
		}
	}
	return nil
}

// Compare determines partial order between two G-Counters
func (gc *GCounter) Compare(other CvRDT) PartialOrder {
	otherCounter, ok := other.(*GCounter)
	if !ok {
		return OrderingConcurrent
	}

	gc.mu.RLock()
	defer gc.mu.RUnlock()

	less, greater := false, false

	allNodes := make(map[string]struct{})
	for node := range gc.payload {
		allNodes[node] = struct{}{}
	}
	for node := range otherCounter.payload {
		allNodes[node] = struct{}{}
	}

	for node := range allNodes {
		gcValue := gc.payload[node]
		otherValue := otherCounter.payload[node]

		if gcValue < otherValue {
			less = true
		}
		if gcValue > otherValue {
			greater = true
		}
	}

	if less && !greater {
		return OrderingBefore
	}
	if greater && !less {
		return OrderingAfter
	}
	if !less && !greater {
		return OrderingEqual
	}
	return OrderingConcurrent
}

// Clone creates a deep copy
func (gc *GCounter) Clone() CvRDT {
	gc.mu.RLock()
	defer gc.mu.RUnlock()

	clone := NewGCounter(gc.nodeID)
	for node, value := range gc.payload {
		clone.payload[node] = value
	}
	return clone
}

// Marshal serializes the G-Counter
func (gc *GCounter) Marshal() ([]byte, error) {
	gc.mu.RLock()
	defer gc.mu.RUnlock()

	data := struct {
		NodeID  string            `json:"node_id"`
		Payload map[string]uint64 `json:"payload"`
	}{
		NodeID:  gc.nodeID,
		Payload: gc.payload,
	}
	return json.Marshal(data)
}

// Unmarshal deserializes the G-Counter
func (gc *GCounter) Unmarshal(data []byte) error {
	gc.mu.Lock()
	defer gc.mu.Unlock()

	var parsed struct {
		NodeID  string            `json:"node_id"`
		Payload map[string]uint64 `json:"payload"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	gc.nodeID = parsed.NodeID
	gc.payload = parsed.Payload
	return nil
}

// PNCounter implements a Positive-Negative Counter CRDT
type PNCounter struct {
	nodeID string
	pos    *GCounter
	neg    *GCounter
	mu     sync.RWMutex
}

// NewPNCounter creates a new PN-Counter
func NewPNCounter(nodeID string) *PNCounter {
	return &PNCounter{
		nodeID: nodeID,
		pos:    NewGCounter(nodeID),
		neg:    NewGCounter(nodeID),
	}
}

// Increment increments the counter
func (pn *PNCounter) Increment(delta int64) {
	pn.mu.Lock()
	defer pn.mu.Unlock()

	if delta > 0 {
		pn.pos.Increment(uint64(delta))
	} else if delta < 0 {
		pn.neg.Increment(uint64(-delta))
	}
}

// Value returns the current counter value (positive - negative)
func (pn *PNCounter) Value() interface{} {
	pn.mu.RLock()
	defer pn.mu.RUnlock()

	posValue := pn.pos.Value().(uint64)
	negValue := pn.neg.Value().(uint64)
	return int64(posValue) - int64(negValue)
}

// Merge combines two PN-Counter states
func (pn *PNCounter) Merge(other CvRDT) error {
	otherCounter, ok := other.(*PNCounter)
	if !ok {
		return ErrIncompatibleType
	}

	pn.mu.Lock()
	defer pn.mu.Unlock()

	if err := pn.pos.Merge(otherCounter.pos); err != nil {
		return err
	}
	return pn.neg.Merge(otherCounter.neg)
}

// Compare determines partial order between two PN-Counters
func (pn *PNCounter) Compare(other CvRDT) PartialOrder {
	otherCounter, ok := other.(*PNCounter)
	if !ok {
		return OrderingConcurrent
	}

	pn.mu.RLock()
	defer pn.mu.RUnlock()

	posOrder := pn.pos.Compare(otherCounter.pos)
	negOrder := pn.neg.Compare(otherCounter.neg)

	if posOrder == OrderingEqual && negOrder == OrderingEqual {
		return OrderingEqual
	}
	if (posOrder == OrderingBefore || posOrder == OrderingEqual) &&
		(negOrder == OrderingBefore || negOrder == OrderingEqual) {
		return OrderingBefore
	}
	if (posOrder == OrderingAfter || posOrder == OrderingEqual) &&
		(negOrder == OrderingAfter || negOrder == OrderingEqual) {
		return OrderingAfter
	}
	return OrderingConcurrent
}

// Clone creates a deep copy
func (pn *PNCounter) Clone() CvRDT {
	pn.mu.RLock()
	defer pn.mu.RUnlock()

	clone := NewPNCounter(pn.nodeID)
	clone.pos = pn.pos.Clone().(*GCounter)
	clone.neg = pn.neg.Clone().(*GCounter)
	return clone
}

// Marshal serializes the PN-Counter
func (pn *PNCounter) Marshal() ([]byte, error) {
	pn.mu.RLock()
	defer pn.mu.RUnlock()

	posData, err := pn.pos.Marshal()
	if err != nil {
		return nil, err
	}

	negData, err := pn.neg.Marshal()
	if err != nil {
		return nil, err
	}

	data := struct {
		NodeID string          `json:"node_id"`
		Pos    json.RawMessage `json:"pos"`
		Neg    json.RawMessage `json:"neg"`
	}{
		NodeID: pn.nodeID,
		Pos:    posData,
		Neg:    negData,
	}
	return json.Marshal(data)
}

// Unmarshal deserializes the PN-Counter
func (pn *PNCounter) Unmarshal(data []byte) error {
	pn.mu.Lock()
	defer pn.mu.Unlock()

	var parsed struct {
		NodeID string          `json:"node_id"`
		Pos    json.RawMessage `json:"pos"`
		Neg    json.RawMessage `json:"neg"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	pn.nodeID = parsed.NodeID
	pn.pos = NewGCounter(pn.nodeID)
	pn.neg = NewGCounter(pn.nodeID)

	if err := pn.pos.Unmarshal(parsed.Pos); err != nil {
		return err
	}
	return pn.neg.Unmarshal(parsed.Neg)
}

var ErrIncompatibleType = &CRDTError{Message: "incompatible CRDT types"}

type CRDTError struct {
	Message string
}

func (e *CRDTError) Error() string {
	return e.Message
}
