package crdt

import (
	"encoding/json"
	"sync"
	"time"
)

// RGA implements a Replicated Growable Array CRDT (operation-based)
type RGA struct {
	nodeID     string
	sequence   []vertex
	tombstones map[string]struct{}
	clock      uint64
	mu         sync.RWMutex
}

type vertex struct {
	ID         string      `json:"id"`
	Element    interface{} `json:"element"`
	LeftOrigin *string     `json:"left_origin,omitempty"`
	Timestamp  Timestamp   `json:"timestamp"`
}

// NewRGA creates a new RGA
func NewRGA(nodeID string) *RGA {
	return &RGA{
		nodeID:     nodeID,
		sequence:   make([]vertex, 0),
		tombstones: make(map[string]struct{}),
		clock:      0,
	}
}

// Insert inserts an element at the specified position
func (rga *RGA) Insert(position int, element interface{}) error {
	rga.mu.Lock()
	defer rga.mu.Unlock()

	if position < 0 || position > rga.visibleLengthInternal() {
		return &CRDTError{Message: "position out of bounds"}
	}

	// Create new vertex
	rga.clock++
	v := vertex{
		ID:      rga.generateID(),
		Element: element,
		Timestamp: Timestamp{
			Node:  rga.nodeID,
			Clock: rga.clock,
			Wall:  time.Now(),
		},
	}

	// Find left origin
	if position > 0 {
		leftVertex := rga.getVisibleVertexInternal(position - 1)
		if leftVertex != nil {
			v.LeftOrigin = &leftVertex.ID
		}
	}

	// Find insertion index
	insertIndex := rga.findInsertionIndex(&v, position)

	// Insert vertex
	rga.sequence = append(rga.sequence[:insertIndex], append([]vertex{v}, rga.sequence[insertIndex:]...)...)

	return nil
}

// Remove removes an element at the specified position
func (rga *RGA) Remove(position int) error {
	rga.mu.Lock()
	defer rga.mu.Unlock()

	if position < 0 || position >= rga.visibleLengthInternal() {
		return &CRDTError{Message: "position out of bounds"}
	}

	visibleVertex := rga.getVisibleVertexInternal(position)
	if visibleVertex != nil {
		rga.tombstones[visibleVertex.ID] = struct{}{}
		return nil
	}

	return &CRDTError{Message: "vertex not found"}
}

// Get returns the element at the specified position
func (rga *RGA) Get(position int) (interface{}, error) {
	rga.mu.RLock()
	defer rga.mu.RUnlock()

	if position < 0 || position >= rga.visibleLengthInternal() {
		return nil, &CRDTError{Message: "position out of bounds"}
	}

	visibleVertex := rga.getVisibleVertexInternal(position)
	if visibleVertex != nil {
		return visibleVertex.Element, nil
	}

	return nil, &CRDTError{Message: "vertex not found"}
}

// ToArray returns all visible elements as a slice
func (rga *RGA) ToArray() []interface{} {
	rga.mu.RLock()
	defer rga.mu.RUnlock()

	result := make([]interface{}, 0)
	for _, v := range rga.sequence {
		if _, tombstoned := rga.tombstones[v.ID]; !tombstoned {
			result = append(result, v.Element)
		}
	}
	return result
}

// Value returns all visible elements (CvRDT interface)
func (rga *RGA) Value() interface{} {
	return rga.ToArray()
}

// Length returns the number of visible elements
func (rga *RGA) Length() int {
	rga.mu.RLock()
	defer rga.mu.RUnlock()
	return rga.visibleLengthInternal()
}

func (rga *RGA) visibleLengthInternal() int {
	count := 0
	for _, v := range rga.sequence {
		if _, tombstoned := rga.tombstones[v.ID]; !tombstoned {
			count++
		}
	}
	return count
}

func (rga *RGA) getVisibleVertexInternal(position int) *vertex {
	visibleCount := 0
	for i := range rga.sequence {
		if _, tombstoned := rga.tombstones[rga.sequence[i].ID]; !tombstoned {
			if visibleCount == position {
				return &rga.sequence[i]
			}
			visibleCount++
		}
	}
	return nil
}

func (rga *RGA) findInsertionIndex(v *vertex, targetPosition int) int {
	// If no left origin, insert at beginning
	if v.LeftOrigin == nil {
		// Find first position or by timestamp order
		for i, existing := range rga.sequence {
			if existing.LeftOrigin == nil {
				if v.Timestamp.Compare(existing.Timestamp) < 0 {
					return i
				}
			} else {
				return i
			}
		}
		return 0
	}

	// Find the left origin in sequence
	leftIndex := -1
	for i, existing := range rga.sequence {
		if existing.ID == *v.LeftOrigin {
			leftIndex = i
			break
		}
	}

	if leftIndex == -1 {
		// Left origin not found, append at end
		return len(rga.sequence)
	}

	// Insert after left origin, but before any vertices with same left origin
	// that have later timestamps
	insertIndex := leftIndex + 1
	for i := leftIndex + 1; i < len(rga.sequence); i++ {
		existing := rga.sequence[i]
		if existing.LeftOrigin != nil && *existing.LeftOrigin == *v.LeftOrigin {
			if v.Timestamp.Compare(existing.Timestamp) < 0 {
				return i
			}
			insertIndex = i + 1
		} else {
			break
		}
	}

	return insertIndex
}

func (rga *RGA) generateID() string {
	return rga.nodeID + "-" + string(rune(rga.clock))
}

// Merge combines two RGA states
func (rga *RGA) Merge(other CvRDT) error {
	otherRGA, ok := other.(*RGA)
	if !ok {
		return ErrIncompatibleType
	}

	rga.mu.Lock()
	defer rga.mu.Unlock()

	// Merge tombstones
	for id := range otherRGA.tombstones {
		rga.tombstones[id] = struct{}{}
	}

	// Merge sequences
	merged := rga.mergeSequences(rga.sequence, otherRGA.sequence)
	rga.sequence = merged

	// Update clock
	if otherRGA.clock > rga.clock {
		rga.clock = otherRGA.clock
	}

	return nil
}

func (rga *RGA) mergeSequences(seq1, seq2 []vertex) []vertex {
	// Create map of existing vertices
	existing := make(map[string]struct{})
	for _, v := range seq1 {
		existing[v.ID] = struct{}{}
	}

	// Add vertices from seq2 that don't exist in seq1
	merged := make([]vertex, len(seq1))
	copy(merged, seq1)

	for _, v := range seq2 {
		if _, exists := existing[v.ID]; !exists {
			// Find insertion position based on left origin and timestamp
			insertIndex := rga.findInsertionIndexInSequence(&v, merged)
			merged = append(merged[:insertIndex], append([]vertex{v}, merged[insertIndex:]...)...)
		}
	}

	return merged
}

func (rga *RGA) findInsertionIndexInSequence(v *vertex, sequence []vertex) int {
	if v.LeftOrigin == nil {
		for i, existing := range sequence {
			if existing.LeftOrigin == nil {
				if v.Timestamp.Compare(existing.Timestamp) < 0 {
					return i
				}
			} else {
				return i
			}
		}
		return 0
	}

	leftIndex := -1
	for i, existing := range sequence {
		if existing.ID == *v.LeftOrigin {
			leftIndex = i
			break
		}
	}

	if leftIndex == -1 {
		return len(sequence)
	}

	insertIndex := leftIndex + 1
	for i := leftIndex + 1; i < len(sequence); i++ {
		existing := sequence[i]
		if existing.LeftOrigin != nil && *existing.LeftOrigin == *v.LeftOrigin {
			if v.Timestamp.Compare(existing.Timestamp) < 0 {
				return i
			}
			insertIndex = i + 1
		} else {
			break
		}
	}

	return insertIndex
}

// Compare determines partial order
func (rga *RGA) Compare(other CvRDT) PartialOrder {
	// Simplified comparison
	return OrderingConcurrent
}

// Clone creates a deep copy
func (rga *RGA) Clone() CvRDT {
	rga.mu.RLock()
	defer rga.mu.RUnlock()

	clone := NewRGA(rga.nodeID)
	clone.clock = rga.clock
	clone.sequence = make([]vertex, len(rga.sequence))
	copy(clone.sequence, rga.sequence)

	for id := range rga.tombstones {
		clone.tombstones[id] = struct{}{}
	}

	return clone
}

// Marshal serializes the RGA
func (rga *RGA) Marshal() ([]byte, error) {
	rga.mu.RLock()
	defer rga.mu.RUnlock()

	tombstoneList := make([]string, 0, len(rga.tombstones))
	for id := range rga.tombstones {
		tombstoneList = append(tombstoneList, id)
	}

	data := struct {
		NodeID     string   `json:"node_id"`
		Sequence   []vertex `json:"sequence"`
		Tombstones []string `json:"tombstones"`
		Clock      uint64   `json:"clock"`
	}{
		NodeID:     rga.nodeID,
		Sequence:   rga.sequence,
		Tombstones: tombstoneList,
		Clock:      rga.clock,
	}

	return json.Marshal(data)
}

// Unmarshal deserializes the RGA
func (rga *RGA) Unmarshal(data []byte) error {
	rga.mu.Lock()
	defer rga.mu.Unlock()

	var parsed struct {
		NodeID     string   `json:"node_id"`
		Sequence   []vertex `json:"sequence"`
		Tombstones []string `json:"tombstones"`
		Clock      uint64   `json:"clock"`
	}

	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	rga.nodeID = parsed.NodeID
	rga.sequence = parsed.Sequence
	rga.clock = parsed.Clock
	rga.tombstones = make(map[string]struct{})

	for _, id := range parsed.Tombstones {
		rga.tombstones[id] = struct{}{}
	}

	return nil
}
