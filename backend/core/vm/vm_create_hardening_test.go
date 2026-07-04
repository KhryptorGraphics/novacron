package vm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestKVMCreateRollback_NoOrphanOnDiskFailure proves the Phase-4 partial-failure
// rollback: when createDiskImage fails AFTER the VM dir has been created, Create
// must leave no orphaned dir and no d.vms entry. Discriminating — without the
// rollback defer in Create, the dir survives and this test fails.
func TestKVMCreateRollback_NoOrphanOnDiskFailure(t *testing.T) {
	base := t.TempDir()
	drv, err := newKVMDriverEnhanced("", base)
	if err != nil {
		t.Skipf("qemu driver unavailable (%v); rollback test needs a qemu binary", err)
	}
	d, ok := drv.(*KVMDriverEnhanced)
	if !ok {
		t.Fatalf("unexpected driver type %T", drv)
	}

	// An http image ref that can't be fetched (nothing listens on 127.0.0.1:1)
	// makes resolveBootImage/createDiskImage fail deterministically AFTER the VM
	// dir is created — without relying on qemu-img's lenient raw auto-detection
	// of local files.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	const id = "rollback-test-vm"
	if _, cerr := d.Create(ctx, VMConfig{
		ID:    id,
		Name:  "rollback-test",
		Type:  VMTypeKVM,
		Image: "http://127.0.0.1:1/nonexistent-boot-image.img",
	}); cerr == nil {
		t.Fatal("expected Create to fail on an unfetchable base image, got nil")
	}

	vmDir := filepath.Join(base, id)
	if _, statErr := os.Stat(vmDir); !os.IsNotExist(statErr) {
		t.Fatalf("orphaned VM dir survived a failed create: %s (stat=%v)", vmDir, statErr)
	}
	d.vmLock.RLock()
	_, tracked := d.vms[id]
	d.vmLock.RUnlock()
	if tracked {
		t.Fatal("failed create left a d.vms entry")
	}
}

// TestCreateVMRequestValidate_Bounds covers the Phase-4 input ceilings without
// regressing the "0 == use default" contract the driver relies on.
func TestCreateVMRequestValidate_Bounds(t *testing.T) {
	valid := func() CreateVMRequest {
		return CreateVMRequest{
			Name:                  "ok",
			AllowMissingOwnership: true,
			Spec:                  VMConfig{Name: "ok", Type: VMTypeKVM},
		}
	}

	if err := valid().Normalized().Validate(); err != nil {
		t.Fatalf("baseline request rejected: %v", err)
	}

	over := valid()
	over.Spec.DiskSizeGB = maxDiskSizeGB() + 1
	if err := over.Normalized().Validate(); err == nil {
		t.Fatal("over-ceiling disk_size_gb should be rejected")
	}

	bigMem := valid()
	bigMem.Spec.MemoryMB = maxMemoryMB() + 1
	if err := bigMem.Normalized().Validate(); err == nil {
		t.Fatal("over-ceiling memory_mb should be rejected")
	}

	longName := valid()
	longName.Name = strings.Repeat("x", maxVMNameLen+1)
	longName.Spec.Name = longName.Name
	if err := longName.Normalized().Validate(); err == nil {
		t.Fatal("oversized name should be rejected")
	}

	// Regression guard: 0 must stay valid (the driver defaults it), not rejected.
	zero := valid()
	zero.Spec.DiskSizeGB = 0
	zero.Spec.MemoryMB = 0
	zero.Spec.CPUShares = 0
	if err := zero.Normalized().Validate(); err != nil {
		t.Fatalf("zero cpu/mem/disk (means default) must stay valid: %v", err)
	}
}
