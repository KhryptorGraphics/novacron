package main

import (
	"context"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestCreateVMRequestNormalizedSetsCanonicalContractDefaults(t *testing.T) {
	req := vm.CreateVMRequest{
		Spec: vm.VMConfig{
			Name:     "contract-vm",
			OwnerID:  "user-123",
			TenantID: "tenant-a",
			NetworkAttachments: []vm.VMNetworkAttachment{
				{NetworkID: "net-secondary"},
				{NetworkID: "net-primary", Primary: true},
			},
		},
		Tags: map[string]string{"purpose": "contract-test"},
	}

	normalized := req.Normalized()

	if got, want := normalized.Name, "contract-vm"; got != want {
		t.Fatalf("normalized request name = %q, want %q", got, want)
	}
	if got, want := normalized.Spec.Type, vm.VMTypeKVM; got != want {
		t.Fatalf("normalized vm type = %q, want %q", got, want)
	}
	if got, want := normalized.Spec.Tags["vm_type"], string(vm.VMTypeKVM); got != want {
		t.Fatalf("normalized vm_type tag = %q, want %q", got, want)
	}
	if got, want := normalized.Spec.Tags["owner_id"], "user-123"; got != want {
		t.Fatalf("owner_id tag = %q, want %q", got, want)
	}
	if got, want := normalized.Spec.Tags["tenant_id"], "tenant-a"; got != want {
		t.Fatalf("tenant_id tag = %q, want %q", got, want)
	}
	if got, want := normalized.Spec.NetworkID, "net-primary"; got != want {
		t.Fatalf("network id = %q, want %q", got, want)
	}
}

func TestCreateVMRequestValidateRejectsInvalidPolicies(t *testing.T) {
	req := vm.CreateVMRequest{
		Name: "invalid-policy",
		Spec: vm.VMConfig{
			Type: vm.VMTypeKVM,
			Placement: &vm.VMPlacementSpec{
				Policy: "unsupported-policy",
			},
		},
	}

	if err := req.Normalized().Validate(); err == nil {
		t.Fatal("expected invalid placement policy to fail validation")
	}
}

func TestVMManagerCreateVMDefaultsToKVMContract(t *testing.T) {
	t.Setenv("PATH", installUbuntu2404TestTools(t))

	managerConfig := vm.DefaultVMManagerConfig()
	managerConfig.Drivers[vm.VMTypeKVM] = vm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"node_id":   "contract-test-node",
			"qemu_path": "qemu-system-x86_64",
			"vm_path":   t.TempDir(),
		},
	}

	manager, err := vm.NewVMManager(managerConfig)
	if err != nil {
		t.Fatalf("NewVMManager returned error: %v", err)
	}
	defer manager.Stop()

	vmInstance, err := manager.CreateVM(context.Background(), vm.CreateVMRequest{
		Name: "contract-default-kvm",
		Spec: vm.VMConfig{
			CPUShares:  1,
			MemoryMB:   512,
			DiskSizeGB: 8,
		},
	})
	if err != nil {
		t.Fatalf("CreateVM returned error: %v", err)
	}

	config := vmInstance.Config()
	if got, want := config.Type, vm.VMTypeKVM; got != want {
		t.Fatalf("vm type = %q, want %q", got, want)
	}
	if got, want := config.Tags["vm_type"], string(vm.VMTypeKVM); got != want {
		t.Fatalf("vm_type tag = %q, want %q", got, want)
	}
}
