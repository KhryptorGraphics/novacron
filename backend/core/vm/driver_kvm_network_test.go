package vm

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestKVMNICArgs(t *testing.T) {
	sysfs := t.TempDir()
	old := sysClassNet
	sysClassNet = sysfs
	t.Cleanup(func() { sysClassNet = old })

	userMode := []string{"-netdev", "user,id=net0", "-device", "virtio-net-pci,netdev=net0"}
	for _, networkID := range []string{"", "default", "test-network"} {
		if got := kvmNICArgs("vm-1", networkID); !slices.Equal(got, userMode) {
			t.Fatalf("network %q: %v, want the unchanged user-mode NIC", networkID, got)
		}
	}

	const netID = "0f3c2a9e-51d4-4b7a-9c2e-7d1f00aa1234"
	got := kvmNICArgs("vm-1", netID)
	if len(got) != 4 || got[1] != "bridge,id=net0,br=ncbr-0f3c2a9e51" {
		t.Fatalf("catalog network args = %v", got)
	}
	mac := strings.TrimPrefix(got[3], "virtio-net-pci,netdev=net0,mac=")
	if mac == got[3] || !strings.HasPrefix(mac, "52:54:") || len(mac) != 17 {
		t.Fatalf("device = %q, want a 52:54 MAC and no host_mtu at MTU 1500/unknown", got[3])
	}
	if again := kvmNICArgs("vm-1", netID); again[3] != got[3] {
		t.Fatalf("MAC not stable across launches: %q vs %q", again[3], got[3])
	}
	if other := kvmNICArgs("vm-2", netID); other[3] == got[3] {
		t.Fatalf("two VMs on one bridge share %q", got[3])
	}

	// A non-default bridge MTU is advertised to the guest.
	if err := os.MkdirAll(filepath.Join(sysfs, "ncbr-0f3c2a9e51"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sysfs, "ncbr-0f3c2a9e51", "mtu"), []byte("9000\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := kvmNICArgs("vm-1", netID); !strings.HasSuffix(got[3], ",host_mtu=9000") {
		t.Fatalf("device = %q, want host_mtu=9000", got[3])
	}
}
