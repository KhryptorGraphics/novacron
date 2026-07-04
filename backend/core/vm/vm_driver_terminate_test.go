package vm

import (
	"os"
	"os/exec"
	"testing"
	"time"
)

// TestStopVMInternal_KillsSigtermIgnoringProcess proves the wave-2 driver fix:
// termination confirms death via /proc and escalates to SIGKILL, so a process
// that ignores SIGTERM (as a re-adopted qemu might) is actually killed rather
// than falsely reported "exited" by Process.Wait() -- which returns ECHILD
// immediately for a non-child process (an adopted VM's Process comes from
// os.FindProcess, so it is not our child).
func TestStopVMInternal_KillsSigtermIgnoringProcess(t *testing.T) {
	// `exec` carries the ignored-TERM disposition (SIG_IGN survives exec) into
	// sleep at the SAME pid; only SIGKILL can end it.
	cmd := exec.Command("sh", "-c", "trap '' TERM; exec sleep 60")
	if err := cmd.Start(); err != nil {
		t.Skipf("cannot start helper process: %v", err)
	}
	pid := cmd.Process.Pid
	go func() { _, _ = cmd.Process.Wait() }() // reap when it dies

	// Shorten the SIGTERM grace so the test escalates to SIGKILL quickly.
	orig := stopGracePeriod
	stopGracePeriod = 200 * time.Millisecond
	defer func() { stopGracePeriod = orig }()

	// Mimic an adopted VM: PID set, Process from os.FindProcess (not our child).
	proc, _ := os.FindProcess(pid)
	d := &KVMDriverEnhanced{vms: map[string]*KVMVMInfo{}}
	vmInfo := &KVMVMInfo{ID: "terminate-test", State: StateRunning, PID: pid, Process: proc}

	if err := d.stopVMInternal(vmInfo); err != nil {
		t.Fatalf("stopVMInternal returned error: %v", err)
	}
	if processAlive(pid) {
		t.Fatalf("SIGTERM-ignoring process %d survived stopVMInternal (no SIGKILL escalation)", pid)
	}
	if vmInfo.State != StateStopped {
		t.Fatalf("vmInfo.State = %v, want StateStopped", vmInfo.State)
	}
}
