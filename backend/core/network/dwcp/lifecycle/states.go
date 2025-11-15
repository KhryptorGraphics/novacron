package lifecycle

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// State represents the lifecycle state of a component
type State int32

const (
	// StateUninitialized indicates component is created but not initialized
	StateUninitialized State = iota

	// StateInitialized indicates component is initialized but not started
	StateInitialized

	// StateStarting indicates component is in the process of starting
	StateStarting

	// StateRunning indicates component is operational
	StateRunning

	// StateStopping indicates component is in the process of stopping
	StateStopping

	// StateStopped indicates component has been cleanly stopped
	StateStopped

	// StateFailed indicates component has encountered an error
	StateFailed
)

// String returns string representation of state
func (s State) String() string {
	switch s {
	case StateUninitialized:
		return "uninitialized"
	case StateInitialized:
		return "initialized"
	case StateStarting:
		return "starting"
	case StateRunning:
		return "running"
	case StateStopping:
		return "stopping"
	case StateStopped:
		return "stopped"
	case StateFailed:
		return "failed"
	default:
		return "unknown"
	}
}

// IsTerminal returns true if state is terminal (no further transitions)
func (s State) IsTerminal() bool {
	return s == StateFailed
}

// IsOperational returns true if component can process requests
func (s State) IsOperational() bool {
	return s == StateRunning
}

// StateTransition represents a valid state transition
type StateTransition struct {
	From State
	To   State
}

// validTransitions defines the valid state machine transitions
var validTransitions = map[StateTransition]bool{
	// Normal startup flow
	{StateUninitialized, StateInitialized}: true,
	{StateInitialized, StateStarting}:      true,
	{StateStarting, StateRunning}:          true,

	// Normal shutdown flow
	{StateRunning, StateStopping}: true,
	{StateStopping, StateStopped}: true,

	// Restart flow
	{StateStopped, StateStarting}: true,

	// Error transitions (can fail from any state except terminal)
	{StateUninitialized, StateFailed}: true,
	{StateInitialized, StateFailed}:   true,
	{StateStarting, StateFailed}:      true,
	{StateRunning, StateFailed}:       true,
	{StateStopping, StateFailed}:      true,
	{StateStopped, StateFailed}:       true,

	// Recovery transitions (can recover from failed state)
	{StateFailed, StateInitialized}: true,
	{StateFailed, StateStarting}:    true,
}

// IsValidTransition checks if a state transition is valid
func IsValidTransition(from, to State) bool {
	return validTransitions[StateTransition{from, to}]
}

// StateMachine manages component state transitions
type StateMachine struct {
	currentState      atomic.Value // stores State
	transitionHistory []StateTransitionRecord
	historyMutex      sync.RWMutex
	observers         []Observer
	observerMutex     sync.RWMutex
	componentName     string
}

// StateTransitionRecord records a state transition with metadata
type StateTransitionRecord struct {
	From      State
	To        State
	Timestamp time.Time
	Duration  time.Duration
	Error     error
}

// NewStateMachine creates a new state machine
func NewStateMachine(componentName string) *StateMachine {
	sm := &StateMachine{
		componentName:     componentName,
		transitionHistory: make([]StateTransitionRecord, 0, 100),
		observers:         make([]Observer, 0),
	}
	sm.currentState.Store(StateUninitialized)
	return sm
}

// GetState returns the current state (thread-safe)
func (sm *StateMachine) GetState() State {
	return sm.currentState.Load().(State)
}

// TransitionTo attempts to transition to a new state
func (sm *StateMachine) TransitionTo(newState State) error {
	currentState := sm.GetState()

	// Check if transition is valid
	if !IsValidTransition(currentState, newState) {
		return fmt.Errorf("invalid state transition: %s -> %s",
			currentState.String(), newState.String())
	}

	// Record transition start time
	startTime := time.Now()

	// Perform the transition
	sm.currentState.Store(newState)

	// Record transition
	record := StateTransitionRecord{
		From:      currentState,
		To:        newState,
		Timestamp: startTime,
		Duration:  time.Since(startTime),
	}

	sm.historyMutex.Lock()
	sm.transitionHistory = append(sm.transitionHistory, record)
	// Keep only last 100 transitions to prevent unbounded growth
	if len(sm.transitionHistory) > 100 {
		sm.transitionHistory = sm.transitionHistory[1:]
	}
	sm.historyMutex.Unlock()

	// Notify observers
	sm.notifyStateChange(currentState, newState)

	return nil
}

// MustTransitionTo transitions to new state or panics
func (sm *StateMachine) MustTransitionTo(newState State) {
	if err := sm.TransitionTo(newState); err != nil {
		panic(fmt.Sprintf("state transition failed: %v", err))
	}
}

// GetTransitionHistory returns the transition history
func (sm *StateMachine) GetTransitionHistory() []StateTransitionRecord {
	sm.historyMutex.RLock()
	defer sm.historyMutex.RUnlock()

	// Return a copy to prevent external modification
	history := make([]StateTransitionRecord, len(sm.transitionHistory))
	copy(history, sm.transitionHistory)
	return history
}

// GetTransitionCount returns count of state transitions
func (sm *StateMachine) GetTransitionCount() int {
	sm.historyMutex.RLock()
	defer sm.historyMutex.RUnlock()
	return len(sm.transitionHistory)
}

// RegisterObserver adds an observer to receive state change notifications
func (sm *StateMachine) RegisterObserver(observer Observer) {
	sm.observerMutex.Lock()
	defer sm.observerMutex.Unlock()
	sm.observers = append(sm.observers, observer)
}

// UnregisterObserver removes an observer
func (sm *StateMachine) UnregisterObserver(observer Observer) {
	sm.observerMutex.Lock()
	defer sm.observerMutex.Unlock()

	for i, obs := range sm.observers {
		if obs == observer {
			sm.observers = append(sm.observers[:i], sm.observers[i+1:]...)
			break
		}
	}
}

// notifyStateChange notifies all observers of state change
func (sm *StateMachine) notifyStateChange(oldState, newState State) {
	sm.observerMutex.RLock()
	observers := make([]Observer, len(sm.observers))
	copy(observers, sm.observers)
	sm.observerMutex.RUnlock()

	for _, observer := range observers {
		// Call observer asynchronously to prevent blocking
		go observer.OnStateChange(sm.componentName, oldState, newState)
	}
}

// CanTransitionTo checks if transition is possible without performing it
func (sm *StateMachine) CanTransitionTo(newState State) bool {
	currentState := sm.GetState()
	return IsValidTransition(currentState, newState)
}

// WaitForState waits until component reaches desired state or timeout
func (sm *StateMachine) WaitForState(ctx context.Context, desiredState State, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if sm.GetState() == desiredState {
				return nil
			}
			if time.Now().After(deadline) {
				return fmt.Errorf("timeout waiting for state %s, current state: %s",
					desiredState.String(), sm.GetState().String())
			}
		}
	}
}

// Reset resets the state machine to uninitialized
func (sm *StateMachine) Reset() {
	sm.currentState.Store(StateUninitialized)
	sm.historyMutex.Lock()
	sm.transitionHistory = sm.transitionHistory[:0]
	sm.historyMutex.Unlock()
}

// GetStateMetrics returns metrics about state machine
func (sm *StateMachine) GetStateMetrics() StateMetrics {
	sm.historyMutex.RLock()
	defer sm.historyMutex.RUnlock()

	metrics := StateMetrics{
		CurrentState:     sm.GetState(),
		TransitionCount:  len(sm.transitionHistory),
		StateHistory:     make(map[State]int),
		TotalTransitions: len(sm.transitionHistory),
	}

	// Calculate state durations and counts
	for _, record := range sm.transitionHistory {
		metrics.StateHistory[record.To]++
		if record.Error != nil {
			metrics.FailedTransitions++
		}
	}

	return metrics
}

// StateMetrics contains metrics about state machine
type StateMetrics struct {
	CurrentState       State
	TransitionCount    int
	FailedTransitions  int
	StateHistory       map[State]int
	TotalTransitions   int
	AverageTransitTime time.Duration
}

// StateValidator validates state before transitions
type StateValidator func(from, to State) error

// SetValidator sets a custom validator for state transitions
func (sm *StateMachine) SetValidator(validator StateValidator) {
	// This could be extended to support custom validation logic
	// For now, we use the built-in validTransitions map
}
