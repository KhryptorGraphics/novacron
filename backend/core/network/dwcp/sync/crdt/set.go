package crdt

import (
	"encoding/json"
	"sync"
	"time"
)

// GSet implements a Grow-only Set CRDT
type GSet struct {
	elements map[string]struct{}
	mu       sync.RWMutex
}

// NewGSet creates a new G-Set
func NewGSet() *GSet {
	return &GSet{
		elements: make(map[string]struct{}),
	}
}

// Add adds an element to the set
func (gs *GSet) Add(element string) {
	gs.mu.Lock()
	defer gs.mu.Unlock()
	gs.elements[element] = struct{}{}
}

// Contains checks if an element is in the set
func (gs *GSet) Contains(element string) bool {
	gs.mu.RLock()
	defer gs.mu.RUnlock()
	_, exists := gs.elements[element]
	return exists
}

// Value returns all elements in the set
func (gs *GSet) Value() interface{} {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	result := make([]string, 0, len(gs.elements))
	for element := range gs.elements {
		result = append(result, element)
	}
	return result
}

// Merge combines two G-Set states
func (gs *GSet) Merge(other CvRDT) error {
	otherSet, ok := other.(*GSet)
	if !ok {
		return ErrIncompatibleType
	}

	gs.mu.Lock()
	defer gs.mu.Unlock()

	for element := range otherSet.elements {
		gs.elements[element] = struct{}{}
	}
	return nil
}

// Compare determines partial order between two G-Sets
func (gs *GSet) Compare(other CvRDT) PartialOrder {
	otherSet, ok := other.(*GSet)
	if !ok {
		return OrderingConcurrent
	}

	gs.mu.RLock()
	defer gs.mu.RUnlock()

	isSubset, isSuperset := true, true

	for element := range gs.elements {
		if _, exists := otherSet.elements[element]; !exists {
			isSubset = false
			break
		}
	}

	for element := range otherSet.elements {
		if _, exists := gs.elements[element]; !exists {
			isSuperset = false
			break
		}
	}

	if isSubset && isSuperset {
		return OrderingEqual
	}
	if isSubset {
		return OrderingBefore
	}
	if isSuperset {
		return OrderingAfter
	}
	return OrderingConcurrent
}

// Clone creates a deep copy
func (gs *GSet) Clone() CvRDT {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	clone := NewGSet()
	for element := range gs.elements {
		clone.elements[element] = struct{}{}
	}
	return clone
}

// Marshal serializes the G-Set
func (gs *GSet) Marshal() ([]byte, error) {
	gs.mu.RLock()
	defer gs.mu.RUnlock()

	elements := make([]string, 0, len(gs.elements))
	for element := range gs.elements {
		elements = append(elements, element)
	}
	return json.Marshal(elements)
}

// Unmarshal deserializes the G-Set
func (gs *GSet) Unmarshal(data []byte) error {
	gs.mu.Lock()
	defer gs.mu.Unlock()

	var elements []string
	if err := json.Unmarshal(data, &elements); err != nil {
		return err
	}

	gs.elements = make(map[string]struct{})
	for _, element := range elements {
		gs.elements[element] = struct{}{}
	}
	return nil
}

// TwoPhaseSet implements a Two-Phase Set CRDT (2P-Set)
type TwoPhaseSet struct {
	added   *GSet
	removed *GSet
	mu      sync.RWMutex
}

// NewTwoPhaseSet creates a new 2P-Set
func NewTwoPhaseSet() *TwoPhaseSet {
	return &TwoPhaseSet{
		added:   NewGSet(),
		removed: NewGSet(),
	}
}

// Add adds an element to the set
func (tps *TwoPhaseSet) Add(element string) {
	tps.mu.Lock()
	defer tps.mu.Unlock()
	tps.added.Add(element)
}

// Remove removes an element from the set
func (tps *TwoPhaseSet) Remove(element string) {
	tps.mu.Lock()
	defer tps.mu.Unlock()
	if tps.added.Contains(element) {
		tps.removed.Add(element)
	}
}

// Contains checks if an element is in the set
func (tps *TwoPhaseSet) Contains(element string) bool {
	tps.mu.RLock()
	defer tps.mu.RUnlock()
	return tps.added.Contains(element) && !tps.removed.Contains(element)
}

// Value returns all elements in the set
func (tps *TwoPhaseSet) Value() interface{} {
	tps.mu.RLock()
	defer tps.mu.RUnlock()

	result := make([]string, 0)
	addedElements := tps.added.Value().([]string)
	for _, element := range addedElements {
		if !tps.removed.Contains(element) {
			result = append(result, element)
		}
	}
	return result
}

// Merge combines two 2P-Set states
func (tps *TwoPhaseSet) Merge(other CvRDT) error {
	otherSet, ok := other.(*TwoPhaseSet)
	if !ok {
		return ErrIncompatibleType
	}

	tps.mu.Lock()
	defer tps.mu.Unlock()

	if err := tps.added.Merge(otherSet.added); err != nil {
		return err
	}
	return tps.removed.Merge(otherSet.removed)
}

// Compare determines partial order
func (tps *TwoPhaseSet) Compare(other CvRDT) PartialOrder {
	otherSet, ok := other.(*TwoPhaseSet)
	if !ok {
		return OrderingConcurrent
	}

	tps.mu.RLock()
	defer tps.mu.RUnlock()

	addedOrder := tps.added.Compare(otherSet.added)
	removedOrder := tps.removed.Compare(otherSet.removed)

	if addedOrder == OrderingEqual && removedOrder == OrderingEqual {
		return OrderingEqual
	}
	if (addedOrder == OrderingBefore || addedOrder == OrderingEqual) &&
		(removedOrder == OrderingBefore || removedOrder == OrderingEqual) {
		return OrderingBefore
	}
	if (addedOrder == OrderingAfter || addedOrder == OrderingEqual) &&
		(removedOrder == OrderingAfter || removedOrder == OrderingEqual) {
		return OrderingAfter
	}
	return OrderingConcurrent
}

// Clone creates a deep copy
func (tps *TwoPhaseSet) Clone() CvRDT {
	tps.mu.RLock()
	defer tps.mu.RUnlock()

	clone := NewTwoPhaseSet()
	clone.added = tps.added.Clone().(*GSet)
	clone.removed = tps.removed.Clone().(*GSet)
	return clone
}

// Marshal serializes the 2P-Set
func (tps *TwoPhaseSet) Marshal() ([]byte, error) {
	tps.mu.RLock()
	defer tps.mu.RUnlock()

	addedData, err := tps.added.Marshal()
	if err != nil {
		return nil, err
	}

	removedData, err := tps.removed.Marshal()
	if err != nil {
		return nil, err
	}

	data := struct {
		Added   json.RawMessage `json:"added"`
		Removed json.RawMessage `json:"removed"`
	}{
		Added:   addedData,
		Removed: removedData,
	}
	return json.Marshal(data)
}

// Unmarshal deserializes the 2P-Set
func (tps *TwoPhaseSet) Unmarshal(data []byte) error {
	tps.mu.Lock()
	defer tps.mu.Unlock()

	var parsed struct {
		Added   json.RawMessage `json:"added"`
		Removed json.RawMessage `json:"removed"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	tps.added = NewGSet()
	tps.removed = NewGSet()

	if err := tps.added.Unmarshal(parsed.Added); err != nil {
		return err
	}
	return tps.removed.Unmarshal(parsed.Removed)
}

// ORSet implements an Observed-Remove Set CRDT
type ORSet struct {
	nodeID   string
	adds     map[string]map[Timestamp]struct{} // element -> {timestamps}
	removes  map[string]map[Timestamp]struct{} // element -> {timestamps}
	clock    uint64
	mu       sync.RWMutex
}

// NewORSet creates a new OR-Set
func NewORSet(nodeID string) *ORSet {
	return &ORSet{
		nodeID:  nodeID,
		adds:    make(map[string]map[Timestamp]struct{}),
		removes: make(map[string]map[Timestamp]struct{}),
		clock:   0,
	}
}

// Add adds an element to the set
func (ors *ORSet) Add(element string) Timestamp {
	ors.mu.Lock()
	defer ors.mu.Unlock()

	ors.clock++
	ts := Timestamp{
		Node:  ors.nodeID,
		Clock: ors.clock,
		Wall:  time.Now(),
	}

	if ors.adds[element] == nil {
		ors.adds[element] = make(map[Timestamp]struct{})
	}
	ors.adds[element][ts] = struct{}{}

	return ts
}

// Remove removes an element from the set
func (ors *ORSet) Remove(element string) {
	ors.mu.Lock()
	defer ors.mu.Unlock()

	// Remove all observed timestamps
	if timestamps, exists := ors.adds[element]; exists {
		if ors.removes[element] == nil {
			ors.removes[element] = make(map[Timestamp]struct{})
		}
		for ts := range timestamps {
			ors.removes[element][ts] = struct{}{}
		}
	}
}

// Contains checks if an element is in the set
func (ors *ORSet) Contains(element string) bool {
	ors.mu.RLock()
	defer ors.mu.RUnlock()

	adds := ors.adds[element]
	removes := ors.removes[element]

	// Element exists if any add timestamp not in removes
	for ts := range adds {
		if _, removed := removes[ts]; !removed {
			return true
		}
	}
	return false
}

// Value returns all elements in the set
func (ors *ORSet) Value() interface{} {
	ors.mu.RLock()
	defer ors.mu.RUnlock()

	result := make([]string, 0)
	for element := range ors.adds {
		if ors.containsInternal(element) {
			result = append(result, element)
		}
	}
	return result
}

func (ors *ORSet) containsInternal(element string) bool {
	adds := ors.adds[element]
	removes := ors.removes[element]

	for ts := range adds {
		if _, removed := removes[ts]; !removed {
			return true
		}
	}
	return false
}

// Merge combines two OR-Set states
func (ors *ORSet) Merge(other CvRDT) error {
	otherSet, ok := other.(*ORSet)
	if !ok {
		return ErrIncompatibleType
	}

	ors.mu.Lock()
	defer ors.mu.Unlock()

	// Merge adds
	for element, timestamps := range otherSet.adds {
		if ors.adds[element] == nil {
			ors.adds[element] = make(map[Timestamp]struct{})
		}
		for ts := range timestamps {
			ors.adds[element][ts] = struct{}{}
		}
	}

	// Merge removes
	for element, timestamps := range otherSet.removes {
		if ors.removes[element] == nil {
			ors.removes[element] = make(map[Timestamp]struct{})
		}
		for ts := range timestamps {
			ors.removes[element][ts] = struct{}{}
		}
	}

	// Update clock
	if otherSet.clock > ors.clock {
		ors.clock = otherSet.clock
	}

	return nil
}

// Compare determines partial order
func (ors *ORSet) Compare(other CvRDT) PartialOrder {
	// Simplified comparison - full implementation would check all timestamps
	return OrderingConcurrent
}

// Clone creates a deep copy
func (ors *ORSet) Clone() CvRDT {
	ors.mu.RLock()
	defer ors.mu.RUnlock()

	clone := NewORSet(ors.nodeID)
	clone.clock = ors.clock

	for element, timestamps := range ors.adds {
		clone.adds[element] = make(map[Timestamp]struct{})
		for ts := range timestamps {
			clone.adds[element][ts] = struct{}{}
		}
	}

	for element, timestamps := range ors.removes {
		clone.removes[element] = make(map[Timestamp]struct{})
		for ts := range timestamps {
			clone.removes[element][ts] = struct{}{}
		}
	}

	return clone
}

// Marshal serializes the OR-Set
func (ors *ORSet) Marshal() ([]byte, error) {
	ors.mu.RLock()
	defer ors.mu.RUnlock()

	type timestampSet []Timestamp

	adds := make(map[string]timestampSet)
	for element, timestamps := range ors.adds {
		tsSlice := make(timestampSet, 0, len(timestamps))
		for ts := range timestamps {
			tsSlice = append(tsSlice, ts)
		}
		adds[element] = tsSlice
	}

	removes := make(map[string]timestampSet)
	for element, timestamps := range ors.removes {
		tsSlice := make(timestampSet, 0, len(timestamps))
		for ts := range timestamps {
			tsSlice = append(tsSlice, ts)
		}
		removes[element] = tsSlice
	}

	data := struct {
		NodeID  string                   `json:"node_id"`
		Adds    map[string]timestampSet  `json:"adds"`
		Removes map[string]timestampSet  `json:"removes"`
		Clock   uint64                   `json:"clock"`
	}{
		NodeID:  ors.nodeID,
		Adds:    adds,
		Removes: removes,
		Clock:   ors.clock,
	}
	return json.Marshal(data)
}

// Unmarshal deserializes the OR-Set
func (ors *ORSet) Unmarshal(data []byte) error {
	ors.mu.Lock()
	defer ors.mu.Unlock()

	type timestampSet []Timestamp

	var parsed struct {
		NodeID  string                   `json:"node_id"`
		Adds    map[string]timestampSet  `json:"adds"`
		Removes map[string]timestampSet  `json:"removes"`
		Clock   uint64                   `json:"clock"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	ors.nodeID = parsed.NodeID
	ors.clock = parsed.Clock
	ors.adds = make(map[string]map[Timestamp]struct{})
	ors.removes = make(map[string]map[Timestamp]struct{})

	for element, timestamps := range parsed.Adds {
		ors.adds[element] = make(map[Timestamp]struct{})
		for _, ts := range timestamps {
			ors.adds[element][ts] = struct{}{}
		}
	}

	for element, timestamps := range parsed.Removes {
		ors.removes[element] = make(map[Timestamp]struct{})
		for _, ts := range timestamps {
			ors.removes[element][ts] = struct{}{}
		}
	}

	return nil
}
