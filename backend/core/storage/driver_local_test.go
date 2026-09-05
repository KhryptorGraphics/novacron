package storage

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"
)

func newTestLocalDriver(t *testing.T) StorageDriver {
	t.Helper()
	d, err := newLocalDriver(map[string]interface{}{
		"base_path": filepath.Join(t.TempDir(), "local"),
	})
	if err != nil {
		t.Fatalf("newLocalDriver: %v", err)
	}
	if err := d.Initialize(); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	return d
}

func TestLocalDriver_CreateWriteReadRoundTrip(t *testing.T) {
	ctx := context.Background()
	d := newTestLocalDriver(t)

	if err := d.CreateVolume(ctx, "vol-a", 1<<20); err != nil {
		t.Fatalf("CreateVolume: %v", err)
	}
	if err := d.CreateVolume(ctx, "vol-a", 1<<20); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("duplicate CreateVolume: got %v, want already-exists error", err)
	}

	payload := bytes.Repeat([]byte("novacron"), 100)
	if err := d.WriteVolume(ctx, "vol-a", 4096, payload); err != nil {
		t.Fatalf("WriteVolume: %v", err)
	}
	got, err := d.ReadVolume(ctx, "vol-a", 4096, len(payload))
	if err != nil {
		t.Fatalf("ReadVolume: %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("round-trip mismatch: got %d bytes, want %d identical bytes", len(got), len(payload))
	}

	// Read past EOF returns the short read without error.
	tail, err := d.ReadVolume(ctx, "vol-a", (1<<20)-10, 100)
	if err != nil {
		t.Fatalf("short ReadVolume: %v", err)
	}
	if len(tail) != 10 {
		t.Fatalf("short read: got %d bytes, want 10", len(tail))
	}
}

func TestLocalDriver_SnapshotLifecycle(t *testing.T) {
	ctx := context.Background()
	d := newTestLocalDriver(t)

	if err := d.CreateVolume(ctx, "vol-snap", 64<<10); err != nil {
		t.Fatalf("CreateVolume: %v", err)
	}
	original := []byte("original-data")
	if err := d.WriteVolume(ctx, "vol-snap", 0, original); err != nil {
		t.Fatalf("WriteVolume: %v", err)
	}

	if err := d.CreateSnapshot(ctx, "vol-snap", "snap-1"); err != nil {
		t.Fatalf("CreateSnapshot: %v", err)
	}

	overwritten := bytes.Repeat([]byte{0xFF}, len(original))
	if err := d.WriteVolume(ctx, "vol-snap", 0, overwritten); err != nil {
		t.Fatalf("overwrite WriteVolume: %v", err)
	}
	if err := d.RestoreSnapshot(ctx, "vol-snap", "snap-1"); err != nil {
		t.Fatalf("RestoreSnapshot: %v", err)
	}
	got, err := d.ReadVolume(ctx, "vol-snap", 0, len(original))
	if err != nil {
		t.Fatalf("ReadVolume after restore: %v", err)
	}
	if !bytes.Equal(got, original) {
		t.Fatalf("restore did not bring back original data: got %q", got)
	}

	if err := d.DeleteSnapshot(ctx, "vol-snap", "snap-1"); err != nil {
		t.Fatalf("DeleteSnapshot: %v", err)
	}
	if err := d.DeleteSnapshot(ctx, "vol-snap", "snap-1"); err == nil {
		t.Fatal("deleting a missing snapshot should fail")
	}

	vols, err := d.ListVolumes(ctx)
	if err != nil {
		t.Fatalf("ListVolumes: %v", err)
	}
	if len(vols) != 1 || vols[0] != "vol-snap" {
		t.Fatalf("ListVolumes = %v, want [vol-snap]", vols)
	}

	if err := d.DeleteVolume(ctx, "vol-snap"); err != nil {
		t.Fatalf("DeleteVolume: %v", err)
	}
	vols, err = d.ListVolumes(ctx)
	if err != nil {
		t.Fatalf("ListVolumes after delete: %v", err)
	}
	if len(vols) != 0 {
		t.Fatalf("ListVolumes after delete = %v, want empty", vols)
	}
}

func TestLocalDriver_AttachDetach(t *testing.T) {
	ctx := context.Background()
	d := newTestLocalDriver(t)

	if err := d.CreateVolume(ctx, "vol-att", 1<<20); err != nil {
		t.Fatalf("CreateVolume: %v", err)
	}
	if err := d.AttachVolume(ctx, "vol-att", "node-a"); err != nil {
		t.Fatalf("AttachVolume: %v", err)
	}
	// Re-attaching to the same node is idempotent.
	if err := d.AttachVolume(ctx, "vol-att", "node-a"); err != nil {
		t.Fatalf("idempotent AttachVolume: %v", err)
	}
	if err := d.AttachVolume(ctx, "vol-att", "node-b"); err == nil || !strings.Contains(err.Error(), "already attached to node-a") {
		t.Fatalf("conflicting AttachVolume: got %v, want already-attached error", err)
	}
	// RestoreSnapshot is refused while attached.
	if err := d.RestoreSnapshot(ctx, "vol-att", "snap-x"); err == nil {
		t.Fatal("RestoreSnapshot while attached should fail")
	}
	if err := d.DetachVolume(ctx, "vol-att", "node-b"); err == nil {
		t.Fatal("DetachVolume to the wrong node should fail")
	}
	if err := d.DetachVolume(ctx, "vol-att", "node-a"); err != nil {
		t.Fatalf("DetachVolume: %v", err)
	}
	if err := d.DetachVolume(ctx, "vol-att", "node-a"); err == nil {
		t.Fatal("DetachVolume of an unattached volume should fail")
	}

	info, err := d.GetVolumeInfo(ctx, "vol-att")
	if err != nil {
		t.Fatalf("GetVolumeInfo: %v", err)
	}
	if info.ID != "vol-att" || info.Size != 1<<20 {
		t.Fatalf("GetVolumeInfo = %+v", info)
	}
}

func TestLocalDriver_RejectsPathTraversal(t *testing.T) {
	ctx := context.Background()
	d := newTestLocalDriver(t)

	for _, id := range []string{"../escape", "a/b", "", ".hidden/../x"} {
		if err := d.CreateVolume(ctx, id, 1<<20); err == nil || !strings.Contains(err.Error(), "invalid volume id") {
			t.Fatalf("CreateVolume(%q): got %v, want invalid-volume-id error", id, err)
		}
	}
}

func TestLocalDriver_DeleteMissingVolume(t *testing.T) {
	ctx := context.Background()
	d := newTestLocalDriver(t)

	if err := d.DeleteVolume(ctx, "nope"); err == nil {
		t.Fatal("DeleteVolume of a missing volume should fail")
	}
}
