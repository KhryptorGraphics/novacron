package graphql

import (
	"context"
	"fmt"
	"strings"

	corestorage "github.com/khryptorgraphics/novacron/backend/core/storage"
)

// Resolver is the root GraphQL resolver
type Resolver struct {
	volumeStore volumeStore
}

type volumeStore interface {
	CreateVolume(ctx context.Context, opts corestorage.VolumeCreateOptions) (*corestorage.VolumeInfo, error)
	ListVolumes(ctx context.Context) ([]*corestorage.VolumeInfo, error)
	UpdateVolumeTier(ctx context.Context, volumeID, tier string) (*corestorage.VolumeInfo, error)
}

// NewResolver creates a new GraphQL resolver
func NewResolver() *Resolver {
	return &Resolver{}
}

// NewResolverWithVolumeStore creates a resolver with storage-backed volume operations.
func NewResolverWithVolumeStore(volumeStore volumeStore) *Resolver {
	resolver := NewResolver()
	resolver.volumeStore = volumeStore
	return resolver
}

// WithVolumeStore configures storage-backed volume operations on the resolver.
func (r *Resolver) WithVolumeStore(volumeStore volumeStore) *Resolver {
	r.volumeStore = volumeStore
	return r
}

// Storage Resolvers

// Volumes returns all storage volumes
func (r *Resolver) Volumes(ctx context.Context, args struct{ Pagination *PaginationInput }) ([]*StorageVolume, error) {
	store, err := r.requireVolumeStore()
	if err != nil {
		return nil, err
	}

	volumes, err := store.ListVolumes(ctx)
	if err != nil {
		return nil, err
	}

	// Apply pagination if provided.
	if args.Pagination != nil {
		start := args.Pagination.Page * args.Pagination.PageSize
		end := start + args.Pagination.PageSize

		if start > len(volumes) {
			return []*StorageVolume{}, nil
		}
		if end > len(volumes) {
			end = len(volumes)
		}

		volumes = volumes[start:end]
	}

	result := make([]*StorageVolume, len(volumes))
	for i, volume := range volumes {
		result[i] = convertStorageVolume(volume)
	}

	return result, nil
}

// CreateVolume creates a new storage volume
func (r *Resolver) CreateVolume(ctx context.Context, args struct{ Input CreateVolumeInput }) (*StorageVolume, error) {
	store, err := r.requireVolumeStore()
	if err != nil {
		return nil, err
	}

	tier, err := normalizeStorageTier(args.Input.Tier)
	if err != nil {
		return nil, err
	}

	metadata := map[string]string{
		"tier": tier,
	}
	if args.Input.VMID != nil && *args.Input.VMID != "" {
		metadata["vm_id"] = *args.Input.VMID
	}

	volume, err := store.CreateVolume(ctx, corestorage.VolumeCreateOptions{
		Name:     args.Input.Name,
		Type:     corestorage.VolumeTypeFile,
		Size:     graphQLVolumeSizeToBytes(args.Input.Size),
		Format:   corestorage.VolumeFormatRAW,
		Metadata: metadata,
	})
	if err != nil {
		return nil, err
	}

	return convertStorageVolume(volume), nil
}

// ChangeVolumeTier changes the storage tier of a volume
func (r *Resolver) ChangeVolumeTier(ctx context.Context, args struct {
	ID   string
	Tier string
}) (*StorageVolume, error) {
	store, err := r.requireVolumeStore()
	if err != nil {
		return nil, err
	}

	tier, err := normalizeStorageTier(args.Tier)
	if err != nil {
		return nil, err
	}

	volume, err := store.UpdateVolumeTier(ctx, args.ID, tier)
	if err != nil {
		return nil, err
	}

	return convertStorageVolume(volume), nil
}

// Helper functions to convert between internal and GraphQL types

func convertStorageVolume(volume *corestorage.VolumeInfo) *StorageVolume {
	if volume == nil {
		return nil
	}

	return &StorageVolume{
		ID:        volume.ID,
		Name:      volume.Name,
		Size:      bytesToGraphQLVolumeSize(volume.Size),
		Tier:      graphqlStorageTier(volume),
		VMID:      graphqlVolumeVMID(volume),
		CreatedAt: volume.CreatedAt,
		UpdatedAt: volume.UpdatedAt,
	}
}

func (r *Resolver) requireVolumeStore() (volumeStore, error) {
	if r.volumeStore == nil {
		return nil, fmt.Errorf("volume store not configured")
	}
	return r.volumeStore, nil
}

func normalizeStorageTier(tier string) (string, error) {
	switch strings.ToUpper(strings.TrimSpace(tier)) {
	case "HOT":
		return "hot", nil
	case "WARM":
		return "warm", nil
	case "COLD":
		return "cold", nil
	case "ARCHIVE":
		return "archive", nil
	default:
		return "", fmt.Errorf("unsupported storage tier %q", tier)
	}
}

func graphqlStorageTier(volume *corestorage.VolumeInfo) string {
	for _, candidate := range []string{
		readVolumeTier(volume.Metadata),
		readVolumeTier(volume.Labels),
	} {
		switch candidate {
		case "hot":
			return "HOT"
		case "warm":
			return "WARM"
		case "cold":
			return "COLD"
		case "archive":
			return "ARCHIVE"
		}
	}

	return "HOT"
}

func graphqlVolumeVMID(volume *corestorage.VolumeInfo) string {
	if volume.AttachedToVM != "" {
		return volume.AttachedToVM
	}
	if vmID := readVolumeField(volume.Metadata, "vm_id"); vmID != "" {
		return vmID
	}
	return readVolumeField(volume.Labels, "vm_id")
}

func readVolumeTier(values map[string]string) string {
	return readVolumeField(values, "tier")
}

func readVolumeField(values map[string]string, key string) string {
	if values == nil {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(values[key]))
}

func graphQLVolumeSizeToBytes(size int) int64 {
	return int64(size) * 1024 * 1024 * 1024
}

func bytesToGraphQLVolumeSize(sizeBytes int64) int {
	if sizeBytes <= 0 {
		return 0
	}
	return int(sizeBytes / (1024 * 1024 * 1024))
}
