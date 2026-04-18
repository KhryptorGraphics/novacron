package rest

import (
	"testing"

	corevm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestCreateVMRequestToVMCreateRequestCarriesCanonicalFields(t *testing.T) {
	req := CreateVMRequest{
		Name:      "rest-contract-vm",
		Type:      corevm.VMTypeKVM,
		CPU:       2,
		Memory:    2048,
		Disk:      20,
		RootFS:    "/images/ubuntu-24.04.qcow2",
		NetworkID: "net-a",
		OwnerID:   "user-7",
		TenantID:  "tenant-blue",
		Tags:      map[string]string{"purpose": "api-test"},
		VolumeAttachments: []corevm.VMVolumeAttachment{
			{VolumeID: "vol-1", Device: "vdb"},
		},
		NetworkAttachments: []corevm.VMNetworkAttachment{
			{NetworkID: "net-a", Primary: true},
		},
	}

	createReq := req.toVMCreateRequest()

	if got, want := createReq.Name, req.Name; got != want {
		t.Fatalf("request name = %q, want %q", got, want)
	}
	if got, want := createReq.Spec.Type, corevm.VMTypeKVM; got != want {
		t.Fatalf("vm type = %q, want %q", got, want)
	}
	if got, want := createReq.Spec.Image, req.RootFS; got != want {
		t.Fatalf("image = %q, want %q", got, want)
	}
	if got, want := createReq.Spec.RootFS, req.RootFS; got != want {
		t.Fatalf("rootfs = %q, want %q", got, want)
	}
	if got, want := createReq.Spec.OwnerID, req.OwnerID; got != want {
		t.Fatalf("owner id = %q, want %q", got, want)
	}
	if got, want := createReq.Spec.TenantID, req.TenantID; got != want {
		t.Fatalf("tenant id = %q, want %q", got, want)
	}
	if got, want := len(createReq.Spec.VolumeAttachments), 1; got != want {
		t.Fatalf("volume attachment count = %d, want %d", got, want)
	}
	if got, want := len(createReq.Spec.NetworkAttachments), 1; got != want {
		t.Fatalf("network attachment count = %d, want %d", got, want)
	}

	req.Tags["purpose"] = "mutated"
	if got, want := createReq.Spec.Tags["purpose"], "api-test"; got != want {
		t.Fatalf("spec tags should be cloned, got %q want %q", got, want)
	}
	if got, want := createReq.Tags["purpose"], "api-test"; got != want {
		t.Fatalf("request tags should be cloned, got %q want %q", got, want)
	}
}
