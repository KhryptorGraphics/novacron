package crdt

import (
	"encoding/json"
	"time"
)

// PartialOrder represents the ordering relationship between two CRDTs
type PartialOrder int

const (
	OrderingBefore PartialOrder = iota
	OrderingAfter
	OrderingEqual
	OrderingConcurrent
)

// Timestamp represents a unique timestamp for CRDT operations
type Timestamp struct {
	Node  string    `json:"node"`
	Clock uint64    `json:"clock"`
	Wall  time.Time `json:"wall"`
}

// Compare compares two timestamps
func (t Timestamp) Compare(other Timestamp) int {
	if t.Clock < other.Clock {
		return -1
	}
	if t.Clock > other.Clock {
		return 1
	}
	if t.Node < other.Node {
		return -1
	}
	if t.Node > other.Node {
		return 1
	}
	return 0
}

// CvRDT represents a state-based CRDT (Convergent CRDT)
type CvRDT interface {
	// Merge combines two CRDT states
	Merge(other CvRDT) error

	// Compare determines partial order
	Compare(other CvRDT) PartialOrder

	// Value returns current interpreted value
	Value() interface{}

	// Clone creates a deep copy
	Clone() CvRDT

	// Marshal serializes the CRDT state
	Marshal() ([]byte, error)

	// Unmarshal deserializes the CRDT state
	Unmarshal(data []byte) error
}

// CmRDT represents an operation-based CRDT (Commutative CRDT)
type CmRDT interface {
	// Prepare creates an operation
	Prepare(op Operation) (Operation, error)

	// Effect applies an operation locally
	Effect(op Operation) error

	// Downstream propagates operation to replicas
	Downstream(op Operation) []Operation

	// Value returns current interpreted value
	Value() interface{}
}

// Operation represents a CRDT operation
type Operation struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	Timestamp Timestamp              `json:"timestamp"`
	Data      map[string]interface{} `json:"data"`
	NodeID    string                 `json:"node_id"`
}

// Marshal serializes the operation
func (o *Operation) Marshal() ([]byte, error) {
	return json.Marshal(o)
}

// Unmarshal deserializes the operation
func (o *Operation) Unmarshal(data []byte) error {
	return json.Unmarshal(data, o)
}

// VectorClock represents a vector clock for causal ordering
type VectorClock map[string]uint64

// Increment increments the clock for a given node
func (vc VectorClock) Increment(nodeID string) {
	vc[nodeID]++
}

// Update updates the vector clock with another clock
func (vc VectorClock) Update(other VectorClock) {
	for node, clock := range other {
		if clock > vc[node] {
			vc[node] = clock
		}
	}
}

// Compare compares two vector clocks
func (vc VectorClock) Compare(other VectorClock) PartialOrder {
	less, greater := false, false

	allNodes := make(map[string]struct{})
	for node := range vc {
		allNodes[node] = struct{}{}
	}
	for node := range other {
		allNodes[node] = struct{}{}
	}

	for node := range allNodes {
		vcClock := vc[node]
		otherClock := other[node]

		if vcClock < otherClock {
			less = true
		}
		if vcClock > otherClock {
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

// Clone creates a deep copy of the vector clock
func (vc VectorClock) Clone() VectorClock {
	clone := make(VectorClock)
	for node, clock := range vc {
		clone[node] = clock
	}
	return clone
}

// Marshal serializes the vector clock
func (vc VectorClock) Marshal() ([]byte, error) {
	return json.Marshal(vc)
}

// Unmarshal deserializes the vector clock
func (vc *VectorClock) Unmarshal(data []byte) error {
	return json.Unmarshal(data, vc)
}

// Digest represents a compact summary of CRDT state for comparison
type Digest struct {
	NodeID      string            `json:"node_id"`
	VectorClock VectorClock       `json:"vector_clock"`
	Checksums   map[string]string `json:"checksums"` // key -> hash
	Timestamp   time.Time         `json:"timestamp"`
}

// Delta represents the difference between two CRDT states
type Delta struct {
	Missing []string `json:"missing"` // Keys we don't have
	Theirs  []string `json:"theirs"`  // Keys they don't have
	Stale   []string `json:"stale"`   // Keys where we're behind
}
