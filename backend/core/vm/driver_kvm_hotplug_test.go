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

// TestHotPlugDiskRealQMP boots a real cirros guest, waits for it to FULLY boot
// (first MIGTICK from the cloud-init seed -- userspace is up, so the guest's PCIe
// hotplug driver is initialised), then hot-plugs a scratch qcow2 disk via
// HotPlugDevice and asserts over an INDEPENDENT QMP connection that the new
// virtio-blk-pci device is attached (query-block lists a BlockBackend whose qdev
// path names our device id). It then hot-unplugs and asserts the device is
// REMOVED (device_del is a guest-driven graceful eject, so it polls).
//
// Discriminating by construction: a bare blockdev-add node is NOT a BlockBackend
// and never appears in query-block -- only device_add makes it show up. So if the
// device_add in hotPlugDisk were removed, the presence assertion below fails
// (empirically confirmed: without device_add, query-block does not list the node).
//
// Why wait for boot before unplug: device_del asks the guest kernel to eject the
// device; if it is issued before the guest has enumerated the slot (plug+unplug
// milliseconds after Start on fast KVM), the eject event is lost and the device
// never leaves. Booting first (and a brief post-plug settle so the guest
// enumerates the new device) makes the eject complete on BOTH x86 "pc" and
// aarch64 "virt" -- empirically confirmed on real x86 KVM and arm64 TCG.
//
// SKIPS (never fails) when qemu / qemu-img / an ISO tool / a cirros image are
// unavailable, matching the sibling migration test.
func TestHotPlugDiskRealQMP(t *testing.T) {
	qemuBin := defaultQEMUBinary("") // arch-aware: qemu-system-x86_64 on amd64, -aarch64 on arm64

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
	cirros := findCirrosImage() // defined in driver_kvm_migrate_test.go (arch-aware)
	if cirros == "" {
		t.Skip("skip: cirros image not found in known locations")
	}

	base := t.TempDir()
	vmBase := filepath.Join(base, "vms")
	drv, err := newKVMDriverEnhanced(qemuBin, vmBase, 3*time.Second)
	if err != nil {
		t.Skipf("skip: KVM driver init failed: %v", err)
	}
	d := drv.(*KVMDriverEnhanced)

	// A cloud-init seed emitting MIGTICK once userspace is up -- our "guest fully
	// booted" signal, so the guest's PCIe hotplug driver is live before plug/unplug.
	seed, err := buildCounterSeedISO(filepath.Join(base, "seed"), isoTool)
	if err != nil {
		t.Skipf("skip: could not build cloud-init seed ISO: %v", err)
	}

	ctx := context.Background()
	const vmID = "hotplug-vm"
	if _, err := d.Create(ctx, VMConfig{
		ID: vmID, Name: vmID, Type: VMTypeKVM,
		MemoryMB: 512, CPUShares: 1, Image: cirros, CloudInitISO: seed,
	}); err != nil {
		t.Fatalf("create VM: %v", err)
	}
	if err := d.Start(ctx, vmID); err != nil {
		t.Fatalf("start VM: %v", err)
	}
	defer func() {
		_ = d.Stop(context.Background(), vmID)
		// Locked read: raw d.vms[vmID] access here would race monitorVM's
		// locked PID/State writes on process exit (novacron -race gate).
		if info, err := d.GetInfo(context.Background(), vmID); err == nil && info.PID > 0 && syscall.Kill(info.PID, 0) == nil {
			_ = syscall.Kill(info.PID, syscall.SIGKILL)
		}
	}()
	// Wait for the guest to FULLY boot before touching hotplug: device_del is a
	// guest-driven eject, silently lost if issued before the guest enumerates the
	// slot. First MIGTICK == userspace up == PCIe hotplug driver initialised.
	consolePath := filepath.Join(vmBase, vmID, "console.log")
	if _, ok := waitForMigTick(consolePath, 0, 150*time.Second); !ok {
		t.Skipf("skip: guest never booted (no MIGTICK within 150s); console: %s", consolePath)
	}

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

	// ponytail: fixed settle for the guest to enumerate the freshly-plugged device
	// before we ask it to eject. A deterministic signal would be guest dmesg via a
	// guest agent; a few seconds is enough here and avoids a QGA dependency.
	time.Sleep(4 * time.Second)

	// Hot-unplug and assert the device disappears. device_del is async, so poll.
	if err := d.HotUnplugDevice(ctx, vmID, devName); err != nil {
		t.Fatalf("HotUnplugDevice: %v", err)
	}
	gone := false
	for i := 0; i < 60; i++ { // up to ~30s (TCG headroom; a booted guest ejects in <1s on KVM)
		if !deviceInQueryBlock(t, sock, devName) {
			gone = true
			break
		}
		time.Sleep(500 * time.Millisecond)
	}
	// With the guest fully booted (waited above) and having enumerated the device
	// (settle above), device_del's graceful eject completes on both x86 "pc" and
	// aarch64 "virt" -- so this is a HARD assertion, not best-effort.
	if !gone {
		t.Fatalf("hot-unplugged device %q still present in query-block ~30s after device_del", devName)
	}
	t.Logf("PASS unplug: device %q gone from query-block after device_del", devName)
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
