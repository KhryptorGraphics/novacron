package storage

import (
	"context"
	"os"
	"testing"
)

func TestStorageManagerUpdateVolumeTier(t *testing.T) {
	t.Parallel()

	manager, err := NewStorageManager(StorageManagerConfig{
		BasePath: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("create storage manager: %v", err)
	}

	volume, err := manager.CreateVolume(context.Background(), VolumeCreateOptions{
		Name:   "tiered-volume",
		Type:   VolumeTypeFile,
		Size:   2 * 1024 * 1024 * 1024,
		Format: VolumeFormatRAW,
		Metadata: map[string]string{
			"tier": "hot",
		},
	})
	if err != nil {
		t.Fatalf("create volume: %v", err)
	}

	updated, err := manager.UpdateVolumeTier(context.Background(), volume.ID, "cold")
	if err != nil {
		t.Fatalf("update volume tier: %v", err)
	}

	if got := updated.Metadata["tier"]; got != "cold" {
		t.Fatalf("expected metadata tier cold, got %q", got)
	}
	if got := updated.Labels["tier"]; got != "cold" {
		t.Fatalf("expected label tier cold, got %q", got)
	}

	reloaded, err := manager.GetVolume(context.Background(), volume.ID)
	if err != nil {
		t.Fatalf("reload volume: %v", err)
	}
	if got := reloaded.Metadata["tier"]; got != "cold" {
		t.Fatalf("expected persisted metadata tier cold, got %q", got)
	}
	if got := reloaded.Labels["tier"]; got != "cold" {
		t.Fatalf("expected persisted label tier cold, got %q", got)
	}
}

func TestStorageManagerCreateVolumeCreatesFilesSynchronously(t *testing.T) {
	t.Parallel()

	manager, err := NewStorageManager(StorageManagerConfig{
		BasePath: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("create storage manager: %v", err)
	}

	volume, err := manager.CreateVolume(context.Background(), VolumeCreateOptions{
		Name:   "ready-volume",
		Type:   VolumeTypeFile,
		Size:   1024 * 1024 * 1024,
		Format: VolumeFormatRAW,
		Metadata: map[string]string{
			"tier":  "hot",
			"vm_id": "vm-42",
		},
	})
	if err != nil {
		t.Fatalf("create volume: %v", err)
	}

	if volume.Status != VolumeStatusAvailable {
		t.Fatalf("expected available status, got %s", volume.Status)
	}
	if volume.State != VolumeStateAvailable {
		t.Fatalf("expected available state, got %s", volume.State)
	}
	if !volume.Available {
		t.Fatal("expected volume to be marked available")
	}
	if got := volume.Metadata["vm_id"]; got != "vm-42" {
		t.Fatalf("expected metadata vm_id vm-42, got %q", got)
	}
	if got := volume.Labels["vm_id"]; got != "vm-42" {
		t.Fatalf("expected label vm_id vm-42, got %q", got)
	}

	if _, err := os.Stat(volume.Path); err != nil {
		t.Fatalf("expected volume file to exist: %v", err)
	}
	if _, err := os.Stat(volume.Path + ".meta"); err != nil {
		t.Fatalf("expected metadata file to exist: %v", err)
	}
}

func TestStorageManagerListVolumesReturnsStableNameOrder(t *testing.T) {
	t.Parallel()

	manager, err := NewStorageManager(StorageManagerConfig{
		BasePath: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("create storage manager: %v", err)
	}

	for _, name := range []string{"vol-c", "vol-a", "vol-b"} {
		if _, err := manager.CreateVolume(context.Background(), VolumeCreateOptions{
			Name:   name,
			Type:   VolumeTypeFile,
			Size:   512 * 1024 * 1024,
			Format: VolumeFormatRAW,
		}); err != nil {
			t.Fatalf("create volume %s: %v", name, err)
		}
	}

	volumes, err := manager.ListVolumes(context.Background())
	if err != nil {
		t.Fatalf("list volumes: %v", err)
	}

	expected := []string{"vol-a", "vol-b", "vol-c"}
	if len(volumes) != len(expected) {
		t.Fatalf("expected %d volumes, got %d", len(expected), len(volumes))
	}

	for i, name := range expected {
		if volumes[i].Name != name {
			t.Fatalf("expected position %d to be %s, got %s", i, name, volumes[i].Name)
		}
	}
}
