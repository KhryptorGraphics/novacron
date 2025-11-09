package crdt

import (
	"encoding/json"
	"sync"
)

// ORMap implements an Observed-Remove Map CRDT
type ORMap struct {
	nodeID  string
	keys    *ORSet
	values  map[string]CvRDT
	mu      sync.RWMutex
}

// NewORMap creates a new OR-Map
func NewORMap(nodeID string) *ORMap {
	return &ORMap{
		nodeID: nodeID,
		keys:   NewORSet(nodeID),
		values: make(map[string]CvRDT),
	}
}

// Set sets a key-value pair
func (om *ORMap) Set(key string, value CvRDT) {
	om.mu.Lock()
	defer om.mu.Unlock()

	om.keys.Add(key)
	om.values[key] = value
}

// SetLWW sets a key with a LWW-Register value
func (om *ORMap) SetLWW(key string, value interface{}) {
	om.mu.Lock()
	defer om.mu.Unlock()

	om.keys.Add(key)

	reg, exists := om.values[key].(*LWWRegister)
	if !exists {
		reg = NewLWWRegister(om.nodeID)
		om.values[key] = reg
	}
	reg.Set(value)
}

// Get retrieves a value by key
func (om *ORMap) Get(key string) (CvRDT, bool) {
	om.mu.RLock()
	defer om.mu.RUnlock()

	if !om.keys.Contains(key) {
		return nil, false
	}

	value, exists := om.values[key]
	return value, exists
}

// GetLWW retrieves a LWW-Register value
func (om *ORMap) GetLWW(key string) (interface{}, bool) {
	om.mu.RLock()
	defer om.mu.RUnlock()

	if !om.keys.Contains(key) {
		return nil, false
	}

	reg, exists := om.values[key].(*LWWRegister)
	if !exists {
		return nil, false
	}

	return reg.Get(), true
}

// Remove removes a key
func (om *ORMap) Remove(key string) {
	om.mu.Lock()
	defer om.mu.Unlock()

	om.keys.Remove(key)
	delete(om.values, key)
}

// Contains checks if a key exists
func (om *ORMap) Contains(key string) bool {
	om.mu.RLock()
	defer om.mu.RUnlock()
	return om.keys.Contains(key)
}

// Keys returns all keys in the map
func (om *ORMap) Keys() []string {
	om.mu.RLock()
	defer om.mu.RUnlock()

	keyList := om.keys.Value().([]string)
	return keyList
}

// Value returns all key-value pairs
func (om *ORMap) Value() interface{} {
	om.mu.RLock()
	defer om.mu.RUnlock()

	result := make(map[string]interface{})
	keyList := om.keys.Value().([]string)

	for _, key := range keyList {
		if value, exists := om.values[key]; exists {
			result[key] = value.Value()
		}
	}
	return result
}

// Merge combines two OR-Map states
func (om *ORMap) Merge(other CvRDT) error {
	otherMap, ok := other.(*ORMap)
	if !ok {
		return ErrIncompatibleType
	}

	om.mu.Lock()
	defer om.mu.Unlock()

	// Merge key sets
	if err := om.keys.Merge(otherMap.keys); err != nil {
		return err
	}

	// Merge values for keys that exist in both
	for key, otherValue := range otherMap.values {
		if om.keys.Contains(key) {
			if ourValue, exists := om.values[key]; exists {
				// Merge if both have the value
				if err := ourValue.Merge(otherValue); err != nil {
					return err
				}
			} else {
				// We don't have it, take theirs
				om.values[key] = otherValue.Clone()
			}
		}
	}

	// Remove values for keys that were removed
	for key := range om.values {
		if !om.keys.Contains(key) {
			delete(om.values, key)
		}
	}

	return nil
}

// Compare determines partial order
func (om *ORMap) Compare(other CvRDT) PartialOrder {
	otherMap, ok := other.(*ORMap)
	if !ok {
		return OrderingConcurrent
	}

	om.mu.RLock()
	defer om.mu.RUnlock()

	// Compare key sets
	keysOrder := om.keys.Compare(otherMap.keys)
	if keysOrder == OrderingConcurrent {
		return OrderingConcurrent
	}

	// If key sets are equal, compare values
	if keysOrder == OrderingEqual {
		less, greater := false, false

		for key, ourValue := range om.values {
			otherValue, exists := otherMap.values[key]
			if !exists {
				continue
			}

			valueOrder := ourValue.Compare(otherValue)
			if valueOrder == OrderingBefore {
				less = true
			} else if valueOrder == OrderingAfter {
				greater = true
			} else if valueOrder == OrderingConcurrent {
				return OrderingConcurrent
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

	return keysOrder
}

// Clone creates a deep copy
func (om *ORMap) Clone() CvRDT {
	om.mu.RLock()
	defer om.mu.RUnlock()

	clone := NewORMap(om.nodeID)
	clone.keys = om.keys.Clone().(*ORSet)

	for key, value := range om.values {
		clone.values[key] = value.Clone()
	}

	return clone
}

// Marshal serializes the OR-Map
func (om *ORMap) Marshal() ([]byte, error) {
	om.mu.RLock()
	defer om.mu.RUnlock()

	keysData, err := om.keys.Marshal()
	if err != nil {
		return nil, err
	}

	// Serialize values
	type valueEntry struct {
		Key  string          `json:"key"`
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}

	values := make([]valueEntry, 0, len(om.values))
	for key, value := range om.values {
		valueData, err := value.Marshal()
		if err != nil {
			return nil, err
		}

		valueType := "unknown"
		switch value.(type) {
		case *LWWRegister:
			valueType = "lww_register"
		case *MVRegister:
			valueType = "mv_register"
		case *GCounter:
			valueType = "g_counter"
		case *PNCounter:
			valueType = "pn_counter"
		case *ORSet:
			valueType = "or_set"
		case *ORMap:
			valueType = "or_map"
		}

		values = append(values, valueEntry{
			Key:  key,
			Type: valueType,
			Data: valueData,
		})
	}

	data := struct {
		NodeID string          `json:"node_id"`
		Keys   json.RawMessage `json:"keys"`
		Values []valueEntry    `json:"values"`
	}{
		NodeID: om.nodeID,
		Keys:   keysData,
		Values: values,
	}

	return json.Marshal(data)
}

// Unmarshal deserializes the OR-Map
func (om *ORMap) Unmarshal(data []byte) error {
	om.mu.Lock()
	defer om.mu.Unlock()

	type valueEntry struct {
		Key  string          `json:"key"`
		Type string          `json:"type"`
		Data json.RawMessage `json:"data"`
	}

	var parsed struct {
		NodeID string          `json:"node_id"`
		Keys   json.RawMessage `json:"keys"`
		Values []valueEntry    `json:"values"`
	}

	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	om.nodeID = parsed.NodeID
	om.keys = NewORSet(om.nodeID)

	if err := om.keys.Unmarshal(parsed.Keys); err != nil {
		return err
	}

	om.values = make(map[string]CvRDT)
	for _, entry := range parsed.Values {
		var value CvRDT

		switch entry.Type {
		case "lww_register":
			value = NewLWWRegister(om.nodeID)
		case "mv_register":
			value = NewMVRegister(om.nodeID)
		case "g_counter":
			value = NewGCounter(om.nodeID)
		case "pn_counter":
			value = NewPNCounter(om.nodeID)
		case "or_set":
			value = NewORSet(om.nodeID)
		case "or_map":
			value = NewORMap(om.nodeID)
		default:
			continue
		}

		if err := value.Unmarshal(entry.Data); err != nil {
			return err
		}

		om.values[entry.Key] = value
	}

	return nil
}

// Size returns the number of keys in the map
func (om *ORMap) Size() int {
	om.mu.RLock()
	defer om.mu.RUnlock()

	keyList := om.keys.Value().([]string)
	return len(keyList)
}

// Clear removes all keys from the map
func (om *ORMap) Clear() {
	om.mu.Lock()
	defer om.mu.Unlock()

	keyList := om.keys.Value().([]string)
	for _, key := range keyList {
		om.keys.Remove(key)
	}
	om.values = make(map[string]CvRDT)
}
