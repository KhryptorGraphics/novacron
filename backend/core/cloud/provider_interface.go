package cloud

import (
	"context"
	"time"
)

// Provider defines the interface that all cloud providers must implement
type Provider interface {
	// Name returns the name of the cloud provider
	Name() string

	// Initialize initializes the provider with the given configuration
	Initialize(config ProviderConfig) error

	// GetInstances returns a list of instances
	GetInstances(ctx context.Context, options ListOptions) ([]Instance, error)

	// GetInstance returns details about a specific instance
	GetInstance(ctx context.Context, id string) (*Instance, error)

	// CreateInstance creates a new instance
	CreateInstance(ctx context.Context, specs InstanceSpecs) (*Instance, error)

	// DeleteInstance deletes an instance
	DeleteInstance(ctx context.Context, id string) error

	// StartInstance starts a stopped instance
	StartInstance(ctx context.Context, id string) error

	// StopInstance stops a running instance
	StopInstance(ctx context.Context, id string) error

	// RestartInstance restarts an instance
	RestartInstance(ctx context.Context, id string) error

	// ResizeInstance changes the size/specs of an instance
	ResizeInstance(ctx context.Context, id string, newSpecs InstanceSpecs) error

	// GetImageList returns a list of available images
	GetImageList(ctx context.Context, options ListOptions) ([]Image, error)

	// GetRegions returns a list of available regions
	GetRegions(ctx context.Context) ([]Region, error)

	// GetStorageVolumes returns a list of storage volumes
	GetStorageVolumes(ctx context.Context, options ListOptions) ([]StorageVolume, error)

	// CreateStorageVolume creates a new storage volume
	CreateStorageVolume(ctx context.Context, specs StorageVolumeSpecs) (*StorageVolume, error)

	// DeleteStorageVolume deletes a storage volume
	DeleteStorageVolume(ctx context.Context, id string) error

	// AttachStorageVolume attaches a storage volume to an instance
	AttachStorageVolume(ctx context.Context, volumeID, instanceID string, opts AttachOptions) error

	// DetachStorageVolume detaches a storage volume from an instance
	DetachStorageVolume(ctx context.Context, volumeID, instanceID string) error

	// CreateSnapshot creates a snapshot of an instance or volume
	CreateSnapshot(ctx context.Context, sourceID string, specs SnapshotSpecs) (*Snapshot, error)

	// GetSnapshots returns a list of snapshots
	GetSnapshots(ctx context.Context, options ListOptions) ([]Snapshot, error)

	// DeleteSnapshot deletes a snapshot
	DeleteSnapshot(ctx context.Context, id string) error

	// GetNetworks returns a list of networks
	GetNetworks(ctx context.Context, options ListOptions) ([]Network, error)

	// CreateNetwork creates a new network
	CreateNetwork(ctx context.Context, specs NetworkSpecs) (*Network, error)

	// DeleteNetwork deletes a network
	DeleteNetwork(ctx context.Context, id string) error

	// GetPricing returns pricing information for resources
	GetPricing(ctx context.Context, resourceType string) (map[string]float64, error)

	// Close closes the provider connection and releases resources
	Close() error
}

// ProviderConfig contains configuration for a cloud provider
type ProviderConfig struct {
	// Provider-specific authentication credentials
	AuthConfig map[string]string

	// Region to use by default
	DefaultRegion string

	// Zone to use by default
	DefaultZone string

	// Default project or organization ID
	ProjectID string

	// HTTP connection timeout
	Timeout time.Duration

	// Maximum number of retries for API requests
	MaxRetries int

	// Additional provider-specific configuration
	Options map[string]interface{}
}

// Instance represents a virtual machine or container instance
type Instance struct {
	// Unique instance ID
	ID string

	// Human-readable name
	Name string

	// Current state (running, stopped, etc.)
	State string

	// Creation time
	CreatedAt time.Time

	// Public IP addresses
	PublicIPs []string

	// Private IP addresses
	PrivateIPs []string

	// Instance type/size
	InstanceType string

	// Region where the instance is located
	Region string

	// Zone where the instance is located
	Zone string

	// OS/system image ID
	ImageID string

	// CPU cores
	CPUCores int

	// Memory in GB
	MemoryGB int

	// Ephemeral disk size in GB
	DiskGB int

	// Network bandwidth in Mbps
	BandwidthMbps int

	// Tags assigned to this instance
	Tags []string

	// Custom metadata
	Metadata map[string]string

	// Provider-specific information
	ProviderData map[string]interface{}
}

// InstanceSpecs defines the specifications for a new instance
type InstanceSpecs struct {
	// Human-readable name
	Name string

	// Instance type/size (e.g., t2.micro)
	InstanceType string

	// Image ID to use
	ImageID string

	// Region to deploy in
	Region string

	// Zone to deploy in
	Zone string

	// Number of CPU cores (for custom sizing)
	CPUCores int

	// Memory in GB (for custom sizing)
	MemoryGB int

	// Disk size in GB for the boot volume
	DiskGB int

	// Initial user data (cloud-init or similar)
	UserData string

	// SSH public key for access
	SSHPublicKey string

	// Network ID to connect to
	NetworkID string

	// Assign public IP
	AssignPublicIP bool

	// Security group IDs
	SecurityGroupIDs []string

	// Tags to assign to the instance
	Tags map[string]string

	// Additional options specific to the provider
	ProviderOptions map[string]interface{}
}

// StorageVolume represents a block storage volume
type StorageVolume struct {
	// Unique volume ID
	ID string

	// Human-readable name
	Name string

	// Size in GB
	SizeGB int

	// Type of storage (e.g., SSD, HDD)
	Type string

	// Current state
	State string

	// Region where the volume is located
	Region string

	// Zone where the volume is located
	Zone string

	// ID of the instance this volume is attached to (if any)
	AttachedTo string

	// Device path when attached (e.g., /dev/sda)
	DevicePath string

	// Creation time
	CreatedAt time.Time

	// Tags assigned to this volume
	Tags []string

	// Provider-specific information
	ProviderData map[string]interface{}
}

// StorageVolumeSpecs defines the specifications for a new storage volume
type StorageVolumeSpecs struct {
	// Human-readable name
	Name string

	// Size in GB
	SizeGB int

	// Type of storage (e.g., SSD, HDD)
	Type string

	// Region to create in
	Region string

	// Zone to create in
	Zone string

	// Create from snapshot ID (optional)
	SnapshotID string

	// Tags to assign to the volume
	Tags map[string]string

	// Additional options specific to the provider
	ProviderOptions map[string]interface{}
}

// AttachOptions defines options for attaching a storage volume
type AttachOptions struct {
	// Device path to use (e.g., /dev/sda)
	DevicePath string

	// Whether to mark the volume as read-only
	ReadOnly bool
}

// Snapshot represents a point-in-time snapshot of an instance or volume
type Snapshot struct {
	// Unique snapshot ID
	ID string

	// Human-readable name
	Name string

	// Type (instance, volume)
	Type string

	// ID of the source instance or volume
	SourceID string

	// Size in GB
	SizeGB int

	// Current state
	State string

	// Region where the snapshot is stored
	Region string

	// Creation time
	CreatedAt time.Time

	// Description
	Description string

	// Tags assigned to this snapshot
	Tags []string

	// Provider-specific information
	ProviderData map[string]interface{}
}

// SnapshotSpecs defines the specifications for a new snapshot
type SnapshotSpecs struct {
	// Human-readable name
	Name string

	// Description
	Description string

	// Tags to assign to the snapshot
	Tags map[string]string

	// Additional options specific to the provider
	ProviderOptions map[string]interface{}
}

// Image represents an OS or system image
type Image struct {
	// Unique image ID
	ID string

	// Human-readable name
	Name string

	// Operating system
	OS string

	// Version
	Version string

	// Architecture (x86_64, arm64, etc.)
	Architecture string

	// Whether this is a public image
	Public bool

	// Minimum disk size required in GB
	MinDiskGB int

	// Size of the image in GB
	SizeGB float64

	// Status of the image
	Status string

	// Creation time
	CreatedAt time.Time

	// Description
	Description string

	// Provider-specific information
	ProviderData map[string]interface{}
}

// Network represents a virtual network
type Network struct {
	// Unique network ID
	ID string

	// Human-readable name
	Name string

	// CIDR range
	CIDR string

	// Whether this is the default network
	Default bool

	// Region where the network is located
	Region string

	// Current state
	State string

	// Creation time
	CreatedAt time.Time

	// Tags assigned to this network
	Tags []string

	// Provider-specific information
	ProviderData map[string]interface{}
}

// NetworkSpecs defines the specifications for a new network
type NetworkSpecs struct {
	// Human-readable name
	Name string

	// CIDR range
	CIDR string

	// Region to create in
	Region string

	// Make this the default network
	Default bool

	// Tags to assign to the network
	Tags map[string]string

	// Additional options specific to the provider
	ProviderOptions map[string]interface{}
}

// Region represents a geographic region
type Region struct {
	// Unique region identifier
	ID string

	// Human-readable name
	Name string

	// Available zones in this region
	Zones []string

	// Whether the region is available
	Available bool

	// Geographic location
	Location string

	// Provider-specific information
	ProviderData map[string]interface{}
}

// ListOptions defines options for listing resources
type ListOptions struct {
	// Filter results by a specific field and value
	Filters map[string]string

	// Maximum number of results to return
	Limit int

	// Offset for pagination
	Offset int

	// Whether to include deleted resources
	IncludeDeleted bool

	// Region to filter by
	Region string

	// Zone to filter by
	Zone string

	// Additional options specific to the provider
	ProviderOptions map[string]interface{}
}

// ProviderRegistry is a registry of available cloud providers
type ProviderRegistry struct {
	providers map[string]Provider
}

// NewProviderRegistry creates a new provider registry
func NewProviderRegistry() *ProviderRegistry {
	return &ProviderRegistry{
		providers: make(map[string]Provider),
	}
}

// RegisterProvider registers a provider with the registry
func (r *ProviderRegistry) RegisterProvider(provider Provider) {
	r.providers[provider.Name()] = provider
}

// GetProvider returns a provider by name
func (r *ProviderRegistry) GetProvider(name string) (Provider, bool) {
	provider, ok := r.providers[name]
	return provider, ok
}

// ListProviders returns a list of registered provider names
func (r *ProviderRegistry) ListProviders() []string {
	names := make([]string, 0, len(r.providers))
	for name := range r.providers {
		names = append(names, name)
	}
	return names
}
