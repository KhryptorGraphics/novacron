package vm

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestLiveMigrationLocalhostCutover reproduces the proven two-qemu localhost live
// migration end to end: a cirros guest emits a monotonic "MIGTICK <unixtime>"
// counter to its serial console; a destination qemu is launched with -incoming
// over shared storage; the source is migrated to it. It then asserts the four
// facts that distinguish a real live cutover from a cold restart:
//
//  1. query-migrate reached "completed" (migrateWithStats returns nil),
//  2. the QEMU downtime was recorded,
//  3. the counter continues on the destination console with NO fresh-boot marker,
//  4. the source qemu process exits.
//
// It SKIPS (never fails) when qemu / an ISO tool / a cirros image are absent, so
// it is safe on CI. It runs under TCG when /dev/kvm is not group-accessible.
//
// NOTE: this file is self-contained but the sibling vm/*_test.go files currently
// fail to compile (a duplicate TestVMDriverManager), so the package test binary
// will not build until that is fixed by the test-owning agent. The same flow was
// reproduced and observed to PASS via a standalone harness.
func TestLiveMigrationLocalhostCutover(t *testing.T) {
	qemuBin := defaultQEMUBinary("") // arch-aware: host qemu (x86_64 on amd64, aarch64 on arm64)

	if _, err := exec.LookPath(qemuBin); err != nil {
		t.Skipf("skip: %s not installed", qemuBin)
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}
	isoTool := firstInPath("genisoimage", "mkisofs", "xorriso")
	if isoTool == "" {
		t.Skip("skip: no ISO tool (genisoimage/mkisofs/xorriso) to build the cloud-init seed")
	}
	cirros := findCirrosImage()
	if cirros == "" {
		t.Skip("skip: cirros aarch64 image not found in known locations")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	seed, err := buildCounterSeedISO(filepath.Join(base, "seed"), isoTool)
	if err != nil {
		t.Skipf("skip: could not build cloud-init seed ISO: %v", err)
	}

	ctx := context.Background()
	const srcID, destID = "mig-src", "mig-dst"

	if _, err := d.Create(ctx, VMConfig{
		ID: srcID, Name: srcID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 2,
		Image: cirros, CloudInitISO: seed,
	}); err != nil {
		t.Fatalf("create source VM: %v", err)
	}
	if err := d.StartMigrationSource(ctx, srcID); err != nil {
		t.Fatalf("start migration source: %v", err)
	}
	srcInfo, err := d.GetInfo(ctx, srcID)
	if err != nil {
		t.Fatalf("get source VM info: %v", err)
	}
	srcPID := srcInfo.PID

	// Belt-and-suspenders cleanup: kill anything still alive so t.TempDir removal
	// (and the host) is not left with orphaned qemu processes. Locked reads: raw
	// d.vms[destID] here would race monitorVM's locked PID/State writes.
	defer func() {
		_ = d.Stop(context.Background(), destID)
		if info, err := d.GetInfo(context.Background(), destID); err == nil && info.PID > 0 {
			_ = syscall.Kill(info.PID, syscall.SIGKILL)
		}
		if syscall.Kill(srcPID, 0) == nil {
			_ = syscall.Kill(srcPID, syscall.SIGKILL)
		}
	}()

	// Wait for the guest to boot and start emitting the counter on the source
	// console (TCG boot of cirros takes ~30s here).
	srcConsole := filepath.Join(vmBase, srcID, "console.log")
	srcPre, ok := waitForMigTick(srcConsole, 0, 150*time.Second)
	if !ok {
		t.Fatalf("source guest never emitted MIGTICK (boot failed); console: %s\nstderr: %s",
			srcConsole, tailFile(filepath.Join(vmBase, srcID, "qemu-stderr.log")))
	}

	// Launch the destination waiting for the incoming stream, then migrate.
	port := freeTCPPort()
	uri := fmt.Sprintf("tcp:127.0.0.1:%d", port)
	destDir := filepath.Join(base, "dst")
	if _, err := d.StartIncoming(ctx, srcID, destID, destDir, uri); err != nil {
		t.Fatalf("start incoming destination: %v", err)
	}

	downtimeMs, totalMs, err := d.migrateWithStats(ctx, srcID, uri, nil)
	if err != nil {
		t.Fatalf("migrate did not complete: %v", err) // non-nil == query-migrate never reached "completed"
	}
	t.Logf("query-migrate=completed downtime=%dms total=%dms", downtimeMs, totalMs)
	if downtimeMs < 0 {
		t.Fatalf("downtime not recorded (got %d)", downtimeMs)
	}

	// (4) source qemu must exit (Migrate quits it after completion).
	if !waitProcessGone(srcPID, 15*time.Second) {
		t.Fatalf("source qemu PID %d still alive after migration", srcPID)
	}

	// (3) destination console: the counter must resume, monotonically, with no
	// fresh cirros boot banner (which would mean a cold boot, not a live cutover).
	destConsole := filepath.Join(destDir, "console.log")
	destTick, ok := waitForMigTick(destConsole, srcPre, 30*time.Second)
	if !ok {
		t.Fatalf("destination never emitted MIGTICK after cutover; console: %s", destConsole)
	}
	if destTick < srcPre {
		t.Fatalf("counter went backwards after cutover: dest=%d < src_pre=%d", destTick, srcPre)
	}
	// A live cutover must NOT re-run the kernel boot on the destination. The
	// reliable discriminator is the kernel boot banner ("Linux version"), printed
	// exactly once at t=0 -- ~30s before the cutover here -- so it was long since
	// flushed to the SOURCE console and can never still sit in the guest's
	// migrated serial buffer at cutover. A cold boot on the dest reprints it; a
	// live cutover never shows it.
	//
	// We deliberately do NOT key on late-boot text like "login:" or
	// "cirros-cloud.net": the getty prompt / cloud-init tail printed in the last
	// second before cutover can still be sitting in the guest tty buffer + UART
	// FIFO (migrated as guest RAM + device state) and legitimately flush onto the
	// DEST console after resume -- observed on ~1/5 real cutovers, source console
	// never shows it -- which is NOT a cold boot. Keying on those made this test
	// intermittently fail (bd novacron-vue).
	if data, _ := os.ReadFile(destConsole); data != nil {
		if strings.Contains(string(data), "Linux version") {
			t.Fatalf("destination console shows kernel boot banner \"Linux version\": cold boot, not a live cutover")
		}
	}
	t.Logf("PASS: counter continued src=%d -> dest=%d monotonically, no reboot; source qemu exited", srcPre, destTick)
}

// --- test helpers (local; keep this file self-contained) ---------------------

func firstInPath(cands ...string) string {
	for _, c := range cands {
		if _, err := exec.LookPath(c); err == nil {
			return c
		}
	}
	return ""
}

// findCirrosImage probes the locations where an arm64 cirros image is cached in
// this environment. Returns "" when none exists (caller skips).
func findCirrosImage() string {
	home, _ := os.UserHomeDir()
	// arch-aware: cirros images are per-arch and the KVM driver launches
	// qemu-system-<arch> to match the host, so pick the guest image for this
	// host arch. Lets the real-qemu tests run on both arm64 and x86_64.
	arch := "aarch64"
	if runtime.GOARCH == "amd64" {
		arch = "x86_64"
	}
	img := "cirros-0.6.2-" + arch + "-disk.img"
	cands := []string{
		filepath.Join(home, "novacron-run/images", img),
		filepath.Join(home, "novacron-e2e/images", img),
	}
	for _, c := range cands {
		if fi, err := os.Stat(c); err == nil && fi.Size() > 0 {
			return c
		}
	}
	return ""
}

// buildCounterSeedISO writes a NoCloud (cidata) seed whose user-data emits a
// once-per-second "MIGTICK <unixtime>" line to the guest serial console, then
// packs it into an ISO with the given tool. cirros reads this via cloud-init.
func buildCounterSeedISO(dir, tool string) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	userData := "#!/bin/sh\n(while true; do echo \"MIGTICK $(date +%s)\"; sleep 1; done) > /dev/console 2>&1 &\n"
	if err := os.WriteFile(filepath.Join(dir, "user-data"), []byte(userData), 0o644); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(dir, "meta-data"),
		[]byte(`{"instance-id": "iid-mig01", "local-hostname": "cirros-mig"}`), 0o644); err != nil {
		return "", err
	}
	iso := filepath.Join(dir, "seed.iso")
	var cmd *exec.Cmd
	switch tool {
	case "xorriso":
		cmd = exec.Command("xorriso", "-as", "genisoimage", "-output", iso,
			"-volid", "cidata", "-joliet", "-rock",
			filepath.Join(dir, "user-data"), filepath.Join(dir, "meta-data"))
	default: // genisoimage / mkisofs share the same flags
		cmd = exec.Command(tool, "-output", iso, "-volid", "cidata", "-joliet", "-rock",
			filepath.Join(dir, "user-data"), filepath.Join(dir, "meta-data"))
	}
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("%s: %w: %s", tool, err, string(out))
	}
	return iso, nil
}

// waitForMigTick polls a serial console.log for the last "MIGTICK <n>" line whose
// value is >= floor, returning that value. floor lets the dest wait for a tick
// at or past the source's pre-migration value.
func waitForMigTick(consolePath string, floor int64, timeout time.Duration) (int64, bool) {
	deadline := time.Now().Add(timeout)
	for {
		if data, err := os.ReadFile(consolePath); err == nil {
			if v, ok := lastMigTick(string(data)); ok && v >= floor {
				return v, true
			}
		}
		if time.Now().After(deadline) {
			return 0, false
		}
		time.Sleep(500 * time.Millisecond)
	}
}

// lastMigTick returns the numeric value of the last "MIGTICK <n>" token in s.
func lastMigTick(s string) (int64, bool) {
	var last int64
	var found bool
	for _, line := range strings.Split(s, "\n") {
		idx := strings.LastIndex(line, "MIGTICK ")
		if idx < 0 {
			continue
		}
		rest := line[idx+len("MIGTICK "):]
		end := 0
		for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
			end++
		}
		if end == 0 {
			continue
		}
		var v int64
		for i := 0; i < end; i++ {
			v = v*10 + int64(rest[i]-'0')
		}
		last, found = v, true
	}
	return last, found
}

func waitProcessGone(pid int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if syscall.Kill(pid, 0) != nil {
			return true
		}
		time.Sleep(200 * time.Millisecond)
	}
	return syscall.Kill(pid, 0) != nil
}

// freeTCPPort asks the kernel for an unused localhost TCP port. Small TOCTOU
// window between close and qemu's bind; acceptable for a test harness.
func freeTCPPort() int {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 40000
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

func tailFile(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return "(none)"
	}
	if len(data) > 800 {
		data = data[len(data)-800:]
	}
	return string(data)
}
