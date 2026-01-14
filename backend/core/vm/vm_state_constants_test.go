package vm

import (
	"testing"
)

// TestVMStateConstants tests that all required VM state constants are defined
func TestVMStateConstants(t *testing.T) {
	// Test that all state constants are properly defined and accessible
	states := []State{
		StateUnknown,
		StateCreated,
		StateCreating,
		StateProvisioning,
		StateRunning,
		StateStopped,
		StatePaused,
		StatePausing,
		StateResuming,
		StateRestarting,
		StateDeleting,
		StateMigrating,
		StateFailed,
	}

	// Verify each state has a non-empty string value
	for _, state := range states {
		if string(state) == "" {
			t.Errorf("State constant should have a non-empty string value")
		}
	}

	// Test specific states used in the fixes
	if StateRunning != "running" {
		t.Errorf("StateRunning should be 'running', got '%s'", StateRunning)
	}

	if StateFailed != "failed" {
		t.Errorf("StateFailed should be 'failed', got '%s'", StateFailed)
	}

	if StateDeleting != "deleting" {
		t.Errorf("StateDeleting should be 'deleting', got '%s'", StateDeleting)
	}

	t.Log("All VM state constants are properly defined")
}

// TestVMStateTransitions tests valid state transitions
func TestVMStateTransitions(t *testing.T) {
	config := VMConfig{
		ID:        "test-vm-states-001",
		Name:      "test-vm-states",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(config)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test initial state
	if vm.GetState() != StateCreated {
		t.Errorf("Initial state should be %s, got %s", StateCreated, vm.GetState())
	}

	// Test state transitions
	testTransitions := []struct {
		from State
		to   State
		desc string
	}{
		{StateCreated, StateRunning, "created to running"},
		{StateRunning, StatePaused, "running to paused"},
		{StatePaused, StateRunning, "paused to running"},
		{StateRunning, StateStopped, "running to stopped"},
		{StateStopped, StateRunning, "stopped to running"},
		{StateRunning, StateFailed, "running to failed"},
		{StateFailed, StateRunning, "failed to running (recovery)"},
		{StateRunning, StateDeleting, "running to deleting"},
	}

	for _, transition := range testTransitions {
		vm.SetState(transition.from)
		if vm.GetState() != transition.from {
			t.Errorf("Failed to set state to %s", transition.from)
			continue
		}

		vm.SetState(transition.to)
		if vm.GetState() != transition.to {
			t.Errorf("Failed transition %s: expected %s, got %s",
				transition.desc, transition.to, vm.GetState())
		}
	}

	t.Log("VM state transitions work correctly")
}

// TestVMStateComparison tests state comparison operations
func TestVMStateComparison(t *testing.T) {
	// Test state equality
	if StateRunning != StateRunning {
		t.Error("StateRunning should equal itself")
	}

	if StateRunning == StateFailed {
		t.Error("StateRunning should not equal StateFailed")
	}

	// Test state string conversion
	if string(StateRunning) != "running" {
		t.Errorf("StateRunning string should be 'running', got '%s'", string(StateRunning))
	}

	// Test state in switch statements (compilation test)
	testState := StateRunning
	var result string
	switch testState {
	case StateRunning:
		result = "running"
	case StateFailed:
		result = "failed"
	case StateDeleting:
		result = "deleting"
	default:
		result = "unknown"
	}

	if result != "running" {
		t.Errorf("Switch statement should return 'running', got '%s'", result)
	}

	t.Log("VM state comparison operations work correctly")
}
