package vm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestKVMDriverDeleteRetainsVMWhenDirectoryCleanupFails(t *testing.T) {
	const vmID = "cleanup-failure"
	vmDir := "/proc/self/status"
	if _, err := os.Stat(vmDir); err != nil {
		t.Skipf("/proc cleanup failure fixture unavailable: %v", err)
	}

	d := &KVMDriverEnhanced{
		vmBasePath: t.TempDir(),
		vms: map[string]*KVMVMInfo{
			vmID: {ID: vmID, State: StateStopped, DiskPath: filepath.Join(vmDir, "disk.img")},
		},
	}

	err := d.Delete(context.Background(), vmID)
	if err == nil {
		t.Fatal("Delete succeeded despite an unremovable VM directory")
	}
	if !strings.Contains(err.Error(), vmDir) {
		t.Errorf("Delete error %q does not include VM directory path %q", err, vmDir)
	}
	if _, tracked := d.vms[vmID]; !tracked {
		t.Fatal("VM was removed from driver map after directory cleanup failed")
	}
}

func TestKVMDriverDeleteRemovesVMAfterDirectoryCleanupSucceeds(t *testing.T) {
	const vmID = "cleanup-success"
	vmBase := t.TempDir()
	vmDir := filepath.Join(vmBase, vmID)
	if err := os.Mkdir(vmDir, 0o700); err != nil {
		t.Fatal(err)
	}
	diskPath := filepath.Join(vmDir, "disk.img")
	if err := os.WriteFile(diskPath, []byte("disk"), 0o600); err != nil {
		t.Fatal(err)
	}

	d := &KVMDriverEnhanced{
		vmBasePath: vmBase,
		vms: map[string]*KVMVMInfo{
			vmID: {ID: vmID, State: StateStopped, DiskPath: diskPath},
		},
	}

	if err := d.Delete(context.Background(), vmID); err != nil {
		t.Fatalf("Delete returned an error: %v", err)
	}
	if _, tracked := d.vms[vmID]; tracked {
		t.Fatal("VM remains in driver map after successful directory cleanup")
	}
	if _, err := os.Stat(vmDir); !os.IsNotExist(err) {
		t.Fatalf("VM directory still exists after Delete: stat error = %v", err)
	}
}
