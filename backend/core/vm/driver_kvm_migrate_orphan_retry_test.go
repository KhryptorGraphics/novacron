package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// TestStaleIncomingDestEvictedOnRetry reproduces novacron-nxy: an interrupted
// migration attempt leaves an orphaned, still-running dest qemu holding the
// disk file open. Before the fix, retrying StartIncomingBlock for the SAME
// destID would collide with that orphan and eventually surface QEMU's opaque
// "Failed to get write lock". The fix (evictStaleIncomingDestLocked) detects
// the still-tracked stale dest and stops it before standing up the retry.
func TestStaleIncomingDestEvictedOnRetry(t *testing.T) {
	qemuBin, _ := findQemuAndCirros()
	if qemuBin == "" {
		t.Skip("skip: no qemu-system for this arch")
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	rawImg := filepath.Join(base, "tiny.raw")
	if out, err := exec.Command("qemu-img", "create", "-f", "raw", rawImg, "8M").CombinedOutput(); err != nil {
		t.Fatalf("create raw image: %v: %s", err, out)
	}
	virtBytes := int64(8 * 1024 * 1024)

	ctx := context.Background()
	const destID = "nxy-dest"
	destDir := filepath.Join(vmBase, destID)

	// Attempt 1: stand up an incoming dest and DELIBERATELY never finish the
	// migration (simulates the source dying mid-flight, before the no-resume
	// watchdog would fire). Its qemu is left running, holding destDir's disk.
	firstPID, _, err := d.StartIncomingBlock(ctx, destID, destDir, "tcp:127.0.0.1:0", "127.0.0.1", virtBytes, VMConfig{
		ID: destID, Name: destID, Type: VMTypeKVM, MemoryMB: 128,
	})
	if err != nil {
		t.Fatalf("attempt 1 StartIncomingBlock: %v", err)
	}
	if firstPID != destID {
		t.Fatalf("unexpected dest id %q", firstPID)
	}
	firstInfo, err := d.GetInfo(ctx, destID)
	if err != nil || firstInfo.PID <= 0 {
		t.Fatalf("attempt 1 dest not running: info=%+v err=%v", firstInfo, err)
	}
	t.Logf("attempt 1: orphaned dest running as pid %d, holding %s", firstInfo.PID, destDir)

	// Attempt 2 (the retry): without the fix, this either hangs on the disk
	// lock or fails once qemu reports it. With the fix it must succeed, and
	// the FIRST attempt's qemu process must actually be gone afterward.
	deadline := time.Now().Add(30 * time.Second)
	var secondErr error
	for time.Now().Before(deadline) {
		_, _, secondErr = d.StartIncomingBlock(ctx, destID, destDir, "tcp:127.0.0.1:0", "127.0.0.1", virtBytes, VMConfig{
			ID: destID, Name: destID, Type: VMTypeKVM, MemoryMB: 128,
		})
		if secondErr == nil {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}
	if secondErr != nil {
		t.Fatalf("retry StartIncomingBlock FAILED (this is the nxy repro if it mentions a write lock): %v", secondErr)
	}

	if processAlive(firstInfo.PID) {
		t.Fatalf("attempt 1's orphaned qemu (pid %d) is still alive after the retry evicted it", firstInfo.PID)
	}

	secondInfo, err := d.GetInfo(ctx, destID)
	if err != nil || secondInfo.PID <= 0 || secondInfo.PID == firstInfo.PID {
		t.Fatalf("retry did not produce a fresh running dest: info=%+v err=%v (first pid was %d)", secondInfo, err, firstInfo.PID)
	}
	t.Logf("PASS: retry evicted the orphan (pid %d gone) and stood up a fresh dest (pid %d)", firstInfo.PID, secondInfo.PID)

	if stopErr := d.Stop(context.Background(), destID); stopErr != nil {
		t.Logf("cleanup Stop error: %v", stopErr)
	}
}

// TestAbortIncomingDestRetriesTransientFailures proves abortIncomingDest
// (novacron-nxy) does not give up after a single lost/failed attempt: a
// server that fails the first two requests and succeeds on the third must
// still be reached.
func TestAbortIncomingDestRetriesTransientFailures(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		var body struct {
			VMID string `json:"vm_id"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"aborted": true, "vm_id": body.VMID})
	}))
	defer srv.Close()

	addr := srv.Listener.Addr().String()
	start := time.Now()
	abortIncomingDest(addr, "vm-nxy-retry")
	elapsed := time.Since(start)

	if calls != 3 {
		t.Fatalf("abortIncomingDest made %d requests, want 3 (2 failures + 1 success)", calls)
	}
	// Backoff is 1s then 2s between failed attempts (see abortIncomingDest);
	// a bare-single-attempt implementation would return in well under 1s.
	if elapsed < time.Second {
		t.Fatalf("abortIncomingDest returned in %s, too fast to have actually retried with backoff", elapsed)
	}
	t.Logf("PASS: abortIncomingDest reached the destination on attempt 3/3 after %s", elapsed)
}

// TestAbortIncomingDestGivesUpAfterExhaustingRetries proves it does not retry
// forever: a destination that always fails must still return in bounded time.
func TestAbortIncomingDestGivesUpAfterExhaustingRetries(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	addr := srv.Listener.Addr().String()
	done := make(chan struct{})
	go func() {
		abortIncomingDest(addr, "vm-nxy-give-up")
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("abortIncomingDest did not return within 20s -- it must give up, not retry forever")
	}
	if calls != 3 {
		t.Fatalf("abortIncomingDest made %d requests, want exactly 3 (bounded retries)", calls)
	}
}
