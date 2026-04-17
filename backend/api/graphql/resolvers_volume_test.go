package graphql

import (
	"context"
	"testing"

	corestorage "github.com/khryptorgraphics/novacron/backend/core/storage"
)

func TestResolverVolumeLifecycleUsesStorageManager(t *testing.T) {
	t.Parallel()

	store, err := corestorage.NewStorageManager(corestorage.StorageManagerConfig{
		BasePath: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("create storage manager: %v", err)
	}

	resolver := NewResolverWithVolumeStore(nil, nil, store)
	vmID := "vm-123"

	created, err := resolver.CreateVolume(context.Background(), struct{ Input CreateVolumeInput }{
		Input: CreateVolumeInput{
			Name: "graphql-volume",
			Size: 10,
			Tier: "HOT",
			VMID: &vmID,
		},
	})
	if err != nil {
		t.Fatalf("create volume: %v", err)
	}

	if created.Name != "graphql-volume" {
		t.Fatalf("expected volume name graphql-volume, got %s", created.Name)
	}
	if created.Size != 10 {
		t.Fatalf("expected graphql size 10, got %d", created.Size)
	}
	if created.Tier != "HOT" {
		t.Fatalf("expected tier HOT, got %s", created.Tier)
	}
	if created.VMID != vmID {
		t.Fatalf("expected vm id %s, got %s", vmID, created.VMID)
	}

	volumes, err := resolver.Volumes(context.Background(), struct{ Pagination *PaginationInput }{})
	if err != nil {
		t.Fatalf("list volumes: %v", err)
	}
	if len(volumes) != 1 {
		t.Fatalf("expected 1 volume, got %d", len(volumes))
	}
	if volumes[0].ID != created.ID {
		t.Fatalf("expected listed volume %s, got %s", created.ID, volumes[0].ID)
	}

	updated, err := resolver.ChangeVolumeTier(context.Background(), struct {
		ID   string
		Tier string
	}{
		ID:   created.ID,
		Tier: "COLD",
	})
	if err != nil {
		t.Fatalf("change volume tier: %v", err)
	}
	if updated.Tier != "COLD" {
		t.Fatalf("expected updated tier COLD, got %s", updated.Tier)
	}

	stored, err := store.GetVolume(context.Background(), created.ID)
	if err != nil {
		t.Fatalf("get stored volume: %v", err)
	}
	if got := stored.Labels["tier"]; got != "cold" {
		t.Fatalf("expected persisted label tier cold, got %q", got)
	}
	if got := stored.Metadata["tier"]; got != "cold" {
		t.Fatalf("expected persisted metadata tier cold, got %q", got)
	}
}

func TestResolverVolumesPagination(t *testing.T) {
	t.Parallel()

	store, err := corestorage.NewStorageManager(corestorage.StorageManagerConfig{
		BasePath: t.TempDir(),
	})
	if err != nil {
		t.Fatalf("create storage manager: %v", err)
	}

	resolver := NewResolverWithVolumeStore(nil, nil, store)
	ctx := context.Background()

	for _, name := range []string{"vol-a", "vol-b", "vol-c"} {
		_, err := resolver.CreateVolume(ctx, struct{ Input CreateVolumeInput }{
			Input: CreateVolumeInput{
				Name: name,
				Size: 1,
				Tier: "WARM",
			},
		})
		if err != nil {
			t.Fatalf("create volume %s: %v", name, err)
		}
	}

	firstPage, err := resolver.Volumes(ctx, struct{ Pagination *PaginationInput }{
		Pagination: &PaginationInput{Page: 0, PageSize: 2},
	})
	if err != nil {
		t.Fatalf("list first page: %v", err)
	}
	if len(firstPage) != 2 {
		t.Fatalf("expected first page length 2, got %d", len(firstPage))
	}

	emptyPage, err := resolver.Volumes(ctx, struct{ Pagination *PaginationInput }{
		Pagination: &PaginationInput{Page: 2, PageSize: 2},
	})
	if err != nil {
		t.Fatalf("list empty page: %v", err)
	}
	if len(emptyPage) != 0 {
		t.Fatalf("expected empty page, got %d results", len(emptyPage))
	}
}

func TestResolverVolumeOperationsRequireStore(t *testing.T) {
	t.Parallel()

	resolver := NewResolver(nil, nil)

	_, err := resolver.Volumes(context.Background(), struct{ Pagination *PaginationInput }{})
	if err == nil {
		t.Fatal("expected volumes to fail without configured volume store")
	}
}
