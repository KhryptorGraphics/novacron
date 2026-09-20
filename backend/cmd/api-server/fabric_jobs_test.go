package main

import (
	"testing"
)

// nodeProfile fixtures: a fast local link and a slow remote one.
func fastLink() *linkProfile { return &linkProfile{RTTMS: 0.4} }
func slowLink() *linkProfile { return &linkProfile{RTTMS: 40} }

// TestEstimatedMoveCostPrefersFastLink: same bytes cost less over the lower-RTT
// link, and zero bytes means only the run-time proxy remains.
func TestEstimatedMoveCostPrefersFastLink(t *testing.T) {
	fast := nodeProfile{NodeCapacity: NodeCapacity{NodeID: "fast"}, Link: fastLink()}
	slow := nodeProfile{NodeCapacity: NodeCapacity{NodeID: "slow"}, Link: slowLink()}
	spec := fabricJobSpec{BytesToMove: 1 << 30, MemoryMB: 512}

	cf, cs := estimatedMoveCostS(fast, spec), estimatedMoveCostS(slow, spec)
	if !(cf < cs) {
		t.Fatalf("fast-link cost %v should be below slow-link cost %v", cf, cs)
	}

	// With no bytes to move, cost is the run-time proxy only: equal across links.
	spec0 := fabricJobSpec{MemoryMB: 512}
	if estimatedMoveCostS(fast, spec0) != estimatedMoveCostS(slow, spec0) {
		t.Fatal("zero-byte jobs should have link-independent cost")
	}
}

// TestRTTToBpsFloorsAndDefaults: an unmeasured link is treated as fast (not
// zero, which would divide to +Inf), and a huge RTT never goes below the floor.
func TestRTTToBpsFloorsAndDefaults(t *testing.T) {
	if got := rttToBps(nodeProfile{}); got <= 0 {
		t.Fatalf("unmeasured link must yield a positive rate, got %v", got)
	}
	if got := rttToBps(nodeProfile{Link: &linkProfile{RTTMS: 100000}}); got < 1e7 {
		t.Fatalf("rate floor violated: %v", got)
	}
}

// TestRemoteStatusEffectiveStatus maps peer-reported driver truth onto job
// outcomes, including the killed-vs-finished distinction.
func TestRemoteStatusEffectiveStatus(t *testing.T) {
	zero := 0
	three := 3
	cases := []struct {
		name   string
		remote remoteVMStatus
		stored string
		want   string
	}{
		{"running", remoteVMStatus{State: "running"}, jobStatusRunning, jobStatusRunning},
		{"exited zero", remoteVMStatus{State: "stopped", ExitCode: &zero}, jobStatusRunning, jobStatusCompleted},
		{"exited nonzero", remoteVMStatus{State: "stopped", ExitCode: &three}, jobStatusRunning, jobStatusFailed},
		{"failed state", remoteVMStatus{State: "failed"}, jobStatusRunning, jobStatusFailed},
		{"unknown keeps stored", remoteVMStatus{State: "unknown"}, jobStatusRunning, jobStatusRunning},
		{"stopped without code keeps stored running", remoteVMStatus{State: "stopped"}, jobStatusRunning, jobStatusCompleted},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.remote.effectiveStatus(tc.stored); got != tc.want {
				t.Fatalf("effectiveStatus = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestClampJobMemory bounds the placement inputs.
func TestClampJobMemory(t *testing.T) {
	if clampJobMemory(0) != 128 {
		t.Fatal("zero memory must default to 128 MB")
	}
	if clampJobMemory(1<<30) != 32768 {
		t.Fatal("oversized memory must clamp to 32768 MB")
	}
	if clampJobMemory(512) != 512 {
		t.Fatal("valid memory must pass through")
	}
}

// TestFabricJobNameStable: an unnamed job gets a deterministic prefix from the
// command and the job id, so it is identifiable in the VM list.
func TestFabricJobNameStable(t *testing.T) {
	spec := fabricJobSpec{Command: "/usr/bin/env"}
	got := fabricJobName(spec, "01234567-89ab-cdef-0123-456789abcdef")
	if got == "" || got[:11] != "fabric-job-" {
		t.Fatalf("unexpected job name %q", got)
	}
	if named := fabricJobName(fabricJobSpec{Command: "x", Name: "explicit"}, "01234567-89ab-cdef-0123-456789abcdef"); named != "explicit" {
		t.Fatalf("explicit name not honored: %q", named)
	}
}