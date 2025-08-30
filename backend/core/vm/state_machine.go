package vm

import (
	"fmt"
	"sync"
)

// VMStateMachine manages valid state transitions for VM lifecycle
type VMStateMachine struct {
	transitions map[State][]State
	mu          sync.RWMutex
}

// NewVMStateMachine creates a new VM state machine
func NewVMStateMachine() *VMStateMachine {
	sm := &VMStateMachine{
		transitions: make(map[State][]State),
	}
	
	sm.initializeTransitions()
	return sm
}

// initializeTransitions sets up valid state transitions
func (sm *VMStateMachine) initializeTransitions() {
	sm.transitions[StateProvisioning] = []State{
		StateInitializing, StateFailed, StateTerminated,
	}
	
	sm.transitions[StateInitializing] = []State{
		StateStarting, StateFailed, StateTerminated,
	}
	
	sm.transitions[StateStarting] = []State{
		StateRunning, StateFailed, StateTerminated,
	}
	
	sm.transitions[StateRunning] = []State{
		StatePaused, StateStopping, StateMigrating, 
		StateCheckpointing, StateMaintenance, StateFailed, StateTerminated,
	}
	
	sm.transitions[StatePaused] = []State{
		StateRunning, StateStopping, StateCheckpointing,
		StateRestoring, StateFailed, StateTerminated,
	}
	
	sm.transitions[StateMigrating] = []State{
		StateRunning, StateFailed, StateTerminated,
	}
	
	sm.transitions[StateCheckpointing] = []State{
		StateRunning, StatePaused, StateFailed, StateTerminated,
	}
	
	sm.transitions[StateRestoring] = []State{
		StateRunning, StateFailed, StateTerminated,
	}
	
	sm.transitions[StateStopping] = []State{
		StateStopped, StateFailed, StateTerminated,
	}
	
	sm.transitions[StateStopped] = []State{
		StateStarting, StateTerminating, StateTerminated,
	}
	
	sm.transitions[StateTerminating] = []State{
		StateTerminated, StateFailed,
	}
	
	sm.transitions[StateFailed] = []State{
		StateStarting, StateStopping, StateTerminating, StateTerminated,
	}
	
	sm.transitions[StateMaintenance] = []State{
		StateRunning, StateStopping, StateFailed, StateTerminated,
	}
	
	// Terminal state
	sm.transitions[StateTerminated] = []State{}
}

// CanTransition checks if a state transition is valid
func (sm *VMStateMachine) CanTransition(from, to State) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	validTransitions, exists := sm.transitions[from]
	if !exists {
		return false
	}
	
	for _, validTo := range validTransitions {
		if validTo == to {
			return true
		}
	}
	
	return false
}

// GetValidTransitions returns all valid transitions from a given state
func (sm *VMStateMachine) GetValidTransitions(from State) []State {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	validTransitions, exists := sm.transitions[from]
	if !exists {
		return []State{}
	}
	
	// Return a copy to avoid data races
	result := make([]State, len(validTransitions))
	copy(result, validTransitions)
	return result
}

// ValidateTransition validates a state transition and returns an error if invalid
func (sm *VMStateMachine) ValidateTransition(from, to State) error {
	if !sm.CanTransition(from, to) {
		return fmt.Errorf("invalid state transition from %s to %s", from, to)
	}
	return nil
}

// IsTerminalState checks if a state is terminal (no further transitions possible)
func (sm *VMStateMachine) IsTerminalState(state State) bool {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	validTransitions, exists := sm.transitions[state]
	return exists && len(validTransitions) == 0
}

// GetStateGraph returns the complete state transition graph
func (sm *VMStateMachine) GetStateGraph() map[State][]State {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	
	result := make(map[State][]State)
	for state, transitions := range sm.transitions {
		result[state] = make([]State, len(transitions))
		copy(result[state], transitions)
	}
	
	return result
}