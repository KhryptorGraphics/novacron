package consensus

import (
	"fmt"
	"sync"
	"time"
)

// SimpleStateMachine is a simple key-value state machine for testing
type SimpleStateMachine struct {
	mu   sync.RWMutex
	data map[string][]byte
}

// NewSimpleStateMachine creates a new simple state machine
func NewSimpleStateMachine() *SimpleStateMachine {
	return &SimpleStateMachine{
		data: make(map[string][]byte),
	}
}

// Apply applies a command to the state machine
func (sm *SimpleStateMachine) Apply(cmd Command) ([]byte, error) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	switch cmd.Type {
	case "write", "put", "set":
		sm.data[cmd.Key] = cmd.Value
		return cmd.Value, nil

	case "read", "get":
		value, exists := sm.data[cmd.Key]
		if !exists {
			return nil, fmt.Errorf("key not found: %s", cmd.Key)
		}
		return value, nil

	case "delete", "remove":
		delete(sm.data, cmd.Key)
		return nil, nil

	default:
		return nil, fmt.Errorf("unknown command type: %s", cmd.Type)
	}
}

// Snapshot creates a snapshot of the state machine
func (sm *SimpleStateMachine) Snapshot() (*Snapshot, error) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	data := make(map[string][]byte)
	for k, v := range sm.data {
		data[k] = v
	}

	return &Snapshot{
		Data:      data,
		Index:     0,
		Term:      0,
		Timestamp: time.Now(),
	}, nil
}

// Restore restores the state machine from a snapshot
func (sm *SimpleStateMachine) Restore(snapshot *Snapshot) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.data = make(map[string][]byte)
	for k, v := range snapshot.Data {
		sm.data[k] = v
	}

	return nil
}

// Get retrieves a value from the state machine
func (sm *SimpleStateMachine) Get(key string) ([]byte, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	value, exists := sm.data[key]
	return value, exists
}

// Size returns the number of keys in the state machine
func (sm *SimpleStateMachine) Size() int {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	return len(sm.data)
}

// Clear clears all data from the state machine
func (sm *SimpleStateMachine) Clear() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.data = make(map[string][]byte)
}
