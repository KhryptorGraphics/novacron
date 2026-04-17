package storage

import (
	"context"
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
