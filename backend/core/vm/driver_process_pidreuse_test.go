package vm

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
)

// TestProcessDriverDetectsPIDReuse reproduces novacron-k8p: a process VM's
// pidfile can outlive the process it named if that pid gets reused by an
// unrelated process (classic after a host/manager restart). The OLD
// liveness check (processAlive(pid) only) would report the VM as still
// running forever in that case. The fix cross-checks /proc/<pid>/stat's
// starttime against the value recorded at launch.
func TestProcessDriverDetectsPIDReuse(t *testing.T) {
	ctx := context.Background()
	d, err := NewProcessDriver(map[string]interface{}{"node_id": "test-node", "base_path": t.TempDir()})
	if err != nil {
		t.Fatalf("NewProcessDriver: %v", err)
	}
	drv := d.(*ProcessDriver)

	id, err := drv.Create(ctx, VMConfig{Name: "pidreuse", Command: "true"})
	if err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Our own test process is real and definitely alive -- stand-in for "an
	// unrelated process that now happens to hold a reused pid".
	realPID := os.Getpid()
	realTicks, ok := processStartTimeTicks(realPID)
	if !ok {
		t.Skip("skip: cannot read /proc/<pid>/stat on this system")
	}

	t.Run("wrong starttime is detected as a different process", func(t *testing.T) {
		wrongTicks := realTicks + 999999 // definitely not this process's real starttime
		if err := os.WriteFile(drv.pidFile(id), []byte(fmt.Sprintf("%d %d", realPID, wrongTicks)), 0644); err != nil {
			t.Fatalf("write crafted pidfile: %v", err)
		}
		st, err := drv.GetStatus(ctx, id)
		if err != nil {
			t.Fatalf("GetStatus: %v", err)
		}
		if st != StateStopped {
			t.Fatalf("GetStatus = %v, want %v (pid is alive but belongs to a different process instance -- the novacron-k8p false positive)", st, StateStopped)
		}
		if _, statErr := os.Stat(drv.pidFile(id)); statErr == nil {
			t.Fatal("stale pidfile (wrong starttime) was not cleaned up by GetStatus")
		}
	})

	t.Run("matching starttime is recognised as the same process", func(t *testing.T) {
		if err := os.WriteFile(drv.pidFile(id), []byte(fmt.Sprintf("%d %d", realPID, realTicks)), 0644); err != nil {
			t.Fatalf("write correct pidfile: %v", err)
		}
		st, err := drv.GetStatus(ctx, id)
		if err != nil {
			t.Fatalf("GetStatus: %v", err)
		}
		if st != StateRunning {
			t.Fatalf("GetStatus = %v, want %v (pid alive with the recorded starttime)", st, StateRunning)
		}
	})

	t.Run("legacy pid-only pidfile (no starttime) still works", func(t *testing.T) {
		if err := os.WriteFile(drv.pidFile(id), []byte(strconv.Itoa(realPID)), 0644); err != nil {
			t.Fatalf("write legacy pidfile: %v", err)
		}
		st, err := drv.GetStatus(ctx, id)
		if err != nil {
			t.Fatalf("GetStatus: %v", err)
		}
		if st != StateRunning {
			t.Fatalf("GetStatus = %v, want %v (pre-upgrade pidfile format must still work -- startTicks=0 falls back to a plain liveness check)", st, StateRunning)
		}
	})

	_ = os.Remove(drv.pidFile(id))
	if err := drv.Delete(ctx, id); err != nil {
		t.Fatalf("Delete: %v", err)
	}
}

// TestProcessStartTimeTicksMatchesRealProcess sanity-checks the /proc/<pid>/stat
// parser directly: called twice for the same live pid, it must return the
// same value (a process's own starttime never changes), and it must fail
// cleanly for a pid that plainly does not exist.
func TestProcessStartTimeTicksMatchesRealProcess(t *testing.T) {
	t1, ok1 := processStartTimeTicks(os.Getpid())
	if !ok1 {
		t.Skip("skip: cannot read /proc/<pid>/stat on this system")
	}
	t2, ok2 := processStartTimeTicks(os.Getpid())
	if !ok2 || t1 != t2 {
		t.Fatalf("processStartTimeTicks not stable for the same pid: %d (ok=%v) vs %d (ok=%v)", t1, ok1, t2, ok2)
	}
	if t1 == 0 {
		t.Fatal("processStartTimeTicks returned 0 for a genuinely running process")
	}

	// PID 1 always exists (init/systemd); a very large, almost certainly
	// unused pid should not.
	if _, ok := processStartTimeTicks(1 << 30); ok {
		t.Fatal("processStartTimeTicks succeeded for an implausible pid")
	}
}
