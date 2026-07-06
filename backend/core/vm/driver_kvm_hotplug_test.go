package vm

import (
	"context"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestHotPlugDiskRealQMP boots a real qemu guest, hot-plugs a scratch qcow2 disk
// via HotPlugDevice, and asserts over an INDEPENDENT QMP connection that the new
// virtio-blk-pci device is actually attached (query-block lists a BlockBackend
// whose qdev path names our device id). It then hot-unplugs and asserts the
// device disappears (device_del is asynchronous, so it polls).
//
// Discriminating by construction: a bare blockdev-add node is NOT a BlockBackend
// and never appears in query-block -- only device_add makes it show up. So if the
// device_add in hotPlugDisk were removed, the presence assertion below fails
// (empirically confirmed: without device_add, query-block does not list the node).
//
// Works under TCG (no /dev/kvm needed): device_add attaches the backend at once,
// regardless of guest boot state. SKIPS (never fails) when qemu / qemu-img / a
// cirros image are unavailable, matching the sibling migration/cpu-pinning tests.
func TestHotPlugDiskRealQMP(t *testing.T) {
	qemuBin := defaultQEMUBinary("") // arch-aware: qemu-system-x86_64 on amd64, -aarch64 on arm64

	if _, err := exec.LookPath(qemuBin); err != nil {
		t.Skipf("skip: %s not installed", qemuBin)
	}
	if _, err := exec.LookPath("qemu-img"); err != nil {
		t.Skip("skip: qemu-img not installed")
	}
	cirros := findCirrosImage() // defined in driver_kvm_migrate_test.go
	if cirros == "" {
		t.Skip("skip: cirros aarch64 image not found in known locations")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	ctx := context.Background()
	const vmID = "hotplug-vm"
	if _, err := d.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 1, Image: cirros,
	}); err != nil {
		t.Fatalf("create VM: %v", err)
	}
	if err := d.Start(ctx, vmID); err != nil {
		t.Fatalf("start VM: %v", err)
	}
	defer func() {
		_ = d.Stop(context.Background(), vmID)
		if p := d.vms[vmID]; p != nil && p.PID > 0 && syscall.Kill(p.PID, 0) == nil {
			_ = syscall.Kill(p.PID, syscall.SIGKILL)
		}
	}()

	// A small scratch qcow2 to hot-plug.
	scratch := filepath.Join(base, "scratch.qcow2")
	if out, err := exec.Command("qemu-img", "create", "-f", "qcow2", scratch, "64M").CombinedOutput(); err != nil {
		t.Fatalf("qemu-img create scratch: %v (%s)", err, out)
	}

	const devName = "hotdisk0"
	sock := filepath.Join(vmBase, vmID, "qmp.sock")

	// Precondition: the device is NOT present before hot-plug (guards against a
	// stale/false-positive match and makes the post-plug assertion meaningful).
	if deviceInQueryBlock(t, sock, devName) {
		t.Fatalf("device %q unexpectedly present before HotPlugDevice", devName)
	}

	if err := d.HotPlugDevice(ctx, vmID, &DeviceConfig{
		Type: "disk", Name: devName,
		Parameters: map[string]interface{}{"path": scratch},
	}); err != nil {
		t.Fatalf("HotPlugDevice: %v", err)
	}

	// Assert (independent QMP conn) the drive is now attached to a guest device.
	if !deviceInQueryBlock(t, sock, devName) {
		t.Fatalf("hot-plugged device %q not present in query-block after HotPlugDevice", devName)
	}
	t.Logf("PASS plug: device %q attached and visible in query-block", devName)

	// Hot-unplug and assert the device disappears. device_del is async, so poll.
	if err := d.HotUnplugDevice(ctx, vmID, devName); err != nil {
		t.Fatalf("HotUnplugDevice: %v", err)
	}
	gone := false
	for i := 0; i < 40; i++ { // up to ~10s
		if !deviceInQueryBlock(t, sock, devName) {
			gone = true
			break
		}
		time.Sleep(250 * time.Millisecond)
	}
	if !gone {
		t.Fatalf("device %q still present in query-block after HotUnplugDevice", devName)
	}
	t.Logf("PASS unplug: device %q gone from query-block", devName)
}

// deviceInQueryBlock opens a fresh QMP connection (independent of the driver's)
// and reports whether query-block lists a BlockBackend whose qdev path names the
// given device id -- i.e. a device_add'd block device with that qdev id exists.
func deviceInQueryBlock(t *testing.T, sock, deviceID string) bool {
	t.Helper()
	q, err := qmpDial(sock, 10*time.Second)
	if err != nil {
		t.Fatalf("dial QMP %s: %v", sock, err)
	}
	defer q.Close()
	raw, err := q.execute("query-block", nil)
	if err != nil {
		t.Fatalf("query-block: %v", err)
	}
	var blocks []struct {
		Qdev string `json:"qdev"` // e.g. /machine/peripheral/hotdisk0/virtio-backend
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		t.Fatalf("parse query-block %q: %v", string(raw), err)
	}
	for _, b := range blocks {
		if strings.Contains(b.Qdev, "/"+deviceID+"/") {
			return true
		}
	}
	return false
}
