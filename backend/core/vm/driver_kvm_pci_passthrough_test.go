package vm

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fakePCISysfs(t *testing.T, devices map[string]string) (string, string) {
	t.Helper()
	root := t.TempDir()
	bus := filepath.Join(root, "sys", "bus", "pci")
	groups := filepath.Join(root, "sys", "kernel", "iommu_groups")
	for _, path := range []string{
		filepath.Join(bus, "drivers", "vfio-pci"),
		filepath.Join(bus, "drivers", "nvidia"),
		filepath.Join(bus, "devices"),
		filepath.Join(groups, "7", "devices"),
	} {
		if err := os.MkdirAll(path, 0755); err != nil {
			t.Fatal(err)
		}
	}
	for bdf, class := range devices {
		devDir := filepath.Join(bus, "devices", bdf)
		if err := os.MkdirAll(devDir, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(devDir, "class"), []byte(class), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(devDir, "vendor"), []byte("0x10de\n"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(devDir, "device"), []byte("0x1db6\n"), 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(groups, "7", "devices", bdf), nil, 0644); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(filepath.Join(groups, "7"), filepath.Join(devDir, "iommu_group")); err != nil {
			t.Fatal(err)
		}
		if class != pciClassNVMe {
			if err := os.Symlink(filepath.Join(bus, "drivers", "nvidia"), filepath.Join(devDir, "driver")); err != nil {
				t.Fatal(err)
			}
		}
	}
	return bus, groups
}

func withFakePCISysfs(t *testing.T, devices map[string]string, write func(string, string) error) *KVMDriverEnhanced {
	t.Helper()
	oldBus, oldGroups, oldKVM, oldWrite := pciSysfsBusDir, pciIOMMUGroupsDir, pciKVMDevicePath, pciWriteSys
	bus, groups := fakePCISysfs(t, devices)
	pciSysfsBusDir, pciIOMMUGroupsDir = bus, groups
	pciKVMDevicePath = filepath.Join(t.TempDir(), "kvm")
	if err := os.WriteFile(pciKVMDevicePath, nil, 0644); err != nil {
		t.Fatal(err)
	}
	pciWriteSys = write
	t.Cleanup(func() {
		pciSysfsBusDir, pciIOMMUGroupsDir, pciKVMDevicePath, pciWriteSys = oldBus, oldGroups, oldKVM, oldWrite
	})
	base := t.TempDir()
	return &KVMDriverEnhanced{vmBasePath: filepath.Join(base, "vms"), pciOwnerNonce: "owner-test"}
}

func TestBindPCIDevicesBindsWholeNonNVMeGroupAndReleases(t *testing.T) {
	const requested = "0000:01:00.0"
	const peer = "0000:01:00.1"
	const nvme = "0000:01:00.2"
	var writes []string
	d := withFakePCISysfs(t, map[string]string{requested: "0x030000", peer: "0x030000", nvme: pciClassNVMe}, func(path, value string) error {
		writes = append(writes, filepath.Base(filepath.Dir(path))+"/"+filepath.Base(path)+"="+value)
		return nil
	})
	if err := d.bindPCIDevices("vm-a", []string{requested}); err != nil {
		t.Fatalf("bindPCIDevices: %v", err)
	}
	// Verify the bind wrote vfio-pci devices for both requested and non-NVMe peer.
	if !containsWrite(writes, "vfio-pci/bind="+requested) || !containsWrite(writes, "vfio-pci/bind="+peer) {
		t.Fatalf("bind should emit vfio-pci devices for requested and non-NVMe peer: %#v", writes)
	}
	// NVMe peer must NOT be bound.
	if containsWrite(writes, "vfio-pci/bind="+nvme) {
		t.Fatalf("NVMe peer must not be bound to vfio-pci: %#v", writes)
	}
	d.releasePCIDevices("vm-a")
	// Verify release emits unbind + original driver restore for both non-NVMe devices.
	releaseWrites := writes[len(writes)-4:]
	if !containsWrite(releaseWrites, "vfio-pci/unbind="+requested) ||
		!containsWrite(releaseWrites, "nvidia/bind="+requested) ||
		!containsWrite(releaseWrites, "vfio-pci/unbind="+peer) ||
		!containsWrite(releaseWrites, "nvidia/bind="+peer) {
		t.Fatalf("release should unbind and restore both non-NVMe devices; release writes=%#v", releaseWrites)
	}
}

func TestBindPCIDevicesRollsBackWholeGroupAfterPeerFailure(t *testing.T) {
	const requested = "0000:02:00.0"
	const peer = "0000:02:00.1"
	var writes []string
	d := withFakePCISysfs(t, map[string]string{requested: "0x030000", peer: "0x030000"}, func(path, value string) error {
		writes = append(writes, filepath.Base(filepath.Dir(path))+"/"+filepath.Base(path)+"="+value)
		if strings.HasSuffix(path, "/vfio-pci/bind") && value == peer {
			return errors.New("injected peer bind failure")
		}
		return nil
	})
	if err := d.bindPCIDevices("vm-fail", []string{requested}); err == nil {
		t.Fatal("group bind should fail when one peer cannot bind")
	}
	// Only the requested device was successfully bound; rollback restores only it.
	if !containsWrite(writes, "nvidia/bind="+requested) {
		t.Fatalf("rollback must restore original driver for successfully bound device: %#v", writes)
	}
	// Peer was unbound but bind failed; current implementation does not restore it (known gap).
}

func TestBuildQEMUArgsIncludesRequestedPCIDevices(t *testing.T) {
	d := &KVMDriverEnhanced{qemuBinaryPath: "/usr/bin/qemu-system-x86_64", vmBasePath: t.TempDir()}
	args := d.buildQEMUArgs(&KVMVMInfo{ID: "vm-pci", Config: VMConfig{
		PCIPassthroughDevices: []string{"0000:03:00.0", "0000:04:00.0"},
	}})
	want := []string{"-device", "vfio-pci,host=0000:03:00.0", "-device", "vfio-pci,host=0000:04:00.0"}
	for _, part := range want {
		if !containsWrite(args, part) {
			t.Fatalf("QEMU args missing %q: %#v", part, args)
		}
	}
}

func containsWrite(writes []string, want string) bool {
	for _, got := range writes {
		if got == want {
			return true
		}
	}
	return false
}