package vm

import (
	"context"
	"path/filepath"
	"testing"
)

func TestKVMDriverEnhancedGuestAgentSocketPath(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "qga.sock")
	driver := &KVMDriverEnhanced{
		vms: map[string]*KVMVMInfo{
			"vm-1": {ID: "vm-1", AgentSockPath: socketPath},
		},
	}

	got, err := driver.GuestAgentSocketPath(context.Background(), "vm-1")
	if err != nil {
		t.Fatalf("guest agent socket path: %v", err)
	}
	if got != socketPath {
		t.Fatalf("expected socket path %s, got %s", socketPath, got)
	}
}

func TestVMManagerGuestAgentSocketPathUsesCachedDriver(t *testing.T) {
	driver := &guestAgentSocketDriver{socketPath: "/tmp/vm-1/qga.sock"}
	manager := &VMManager{
		drivers: map[VMType]VMDriver{VMTypeKVM: driver},
		vms: map[string]*VM{
			"vm-1": mustNewGuestAgentTestVM(t),
		},
	}

	got, err := manager.GuestAgentSocketPath(context.Background(), "vm-1")
	if err != nil {
		t.Fatalf("guest agent socket path: %v", err)
	}
	if got != driver.socketPath {
		t.Fatalf("expected socket path %s, got %s", driver.socketPath, got)
	}
}

func mustNewGuestAgentTestVM(t *testing.T) *VM {
	t.Helper()
	instance, err := NewVM(VMConfig{
		ID:       "vm-1",
		Name:     "vm-1",
		Type:     VMTypeKVM,
		OwnerID:  "owner-a",
		TenantID: "tenant-a",
		Tags:     map[string]string{"vm_type": string(VMTypeKVM)},
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}
	return instance
}

type guestAgentSocketDriver struct {
	CoreStubDriver
	socketPath string
}

func (d *guestAgentSocketDriver) GuestAgentSocketPath(context.Context, string) (string, error) {
	return d.socketPath, nil
}
