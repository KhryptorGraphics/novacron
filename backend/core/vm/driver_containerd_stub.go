package vm

import (
	"context"
	"errors"
)

// ContainerdDriverStub is a stub implementation of the containerd driver
// TODO: Implement full containerd integration
type ContainerdDriverStub struct {
	nodeID string
}

// NewContainerdDriver creates a new containerd driver (stub implementation)
func NewContainerdDriver(config map[string]interface{}) (VMDriver, error) {
	nodeID := ""
	if id, ok := config["node_id"].(string); ok {
		nodeID = id
	}

	return &ContainerdDriverStub{
		nodeID: nodeID,
	}, nil
}

// Create creates a new containerd container VM (stub)
func (d *ContainerdDriverStub) Create(ctx context.Context, config VMConfig) (string, error) {
	return "", errors.New("containerd driver not implemented")
}

// Start starts a containerd container VM (stub)
func (d *ContainerdDriverStub) Start(ctx context.Context, vmID string) error {
	return errors.New("containerd driver not implemented")
}

// Stop stops a containerd container VM (stub)
func (d *ContainerdDriverStub) Stop(ctx context.Context, vmID string) error {
	return errors.New("containerd driver not implemented")
}

// Delete deletes a containerd container VM (stub)
func (d *ContainerdDriverStub) Delete(ctx context.Context, vmID string) error {
	return errors.New("containerd driver not implemented")
}

// GetStatus gets the status of a containerd container VM (stub)
func (d *ContainerdDriverStub) GetStatus(ctx context.Context, vmID string) (State, error) {
	return StateUnknown, errors.New("containerd driver not implemented")
}

// GetInfo gets information about a containerd container VM (stub)
func (d *ContainerdDriverStub) GetInfo(ctx context.Context, vmID string) (*VMInfo, error) {
	return nil, errors.New("containerd driver not implemented")
}

// GetMetrics gets metrics for a containerd container VM (stub)
func (d *ContainerdDriverStub) GetMetrics(ctx context.Context, vmID string) (*VMInfo, error) {
	return nil, errors.New("containerd driver not implemented")
}

// ListVMs lists all containerd container VMs (stub)
func (d *ContainerdDriverStub) ListVMs(ctx context.Context) ([]VMInfo, error) {
	return nil, errors.New("containerd driver not implemented")
}

// SupportsPause returns whether the driver supports pausing VMs
func (d *ContainerdDriverStub) SupportsPause() bool {
	return false
}

// SupportsResume returns whether the driver supports resuming VMs
func (d *ContainerdDriverStub) SupportsResume() bool {
	return false
}

// SupportsSnapshot returns whether the driver supports snapshots
func (d *ContainerdDriverStub) SupportsSnapshot() bool {
	return false
}

// SupportsMigrate returns whether the driver supports migration
func (d *ContainerdDriverStub) SupportsMigrate() bool {
	return false
}

// Pause pauses a containerd container VM (stub)
func (d *ContainerdDriverStub) Pause(ctx context.Context, vmID string) error {
	return errors.New("containerd driver not implemented")
}

// Resume resumes a containerd container VM (stub)
func (d *ContainerdDriverStub) Resume(ctx context.Context, vmID string) error {
	return errors.New("containerd driver not implemented")
}

// Snapshot creates a snapshot of a containerd container VM (stub)
func (d *ContainerdDriverStub) Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error) {
	return "", errors.New("containerd driver not implemented")
}

// Migrate migrates a containerd container VM (stub)
func (d *ContainerdDriverStub) Migrate(ctx context.Context, vmID, target string, params map[string]string) error {
	return errors.New("containerd driver not implemented")
}

// SupportsLiveMigration returns whether the driver supports live migration
func (d *ContainerdDriverStub) SupportsLiveMigration() bool {
	return false
}

// SupportsHotPlug returns whether the driver supports hot-plugging devices
func (d *ContainerdDriverStub) SupportsHotPlug() bool {
	return false
}

// SupportsGPUPassthrough returns whether the driver supports GPU passthrough
func (d *ContainerdDriverStub) SupportsGPUPassthrough() bool {
	return false
}

// SupportsSRIOV returns whether the driver supports SR-IOV
func (d *ContainerdDriverStub) SupportsSRIOV() bool {
	return false
}

// SupportsNUMA returns whether the driver supports NUMA configuration
func (d *ContainerdDriverStub) SupportsNUMA() bool {
	return false
}

// GetCapabilities returns the capabilities of the containerd driver (stub)
func (d *ContainerdDriverStub) GetCapabilities(ctx context.Context) (*HypervisorCapabilities, error) {
	return nil, errors.New("containerd driver not implemented")
}

// GetHypervisorInfo returns information about the containerd runtime (stub)
func (d *ContainerdDriverStub) GetHypervisorInfo(ctx context.Context) (*HypervisorInfo, error) {
	return nil, errors.New("containerd driver not implemented")
}

// HotPlugDevice hot-plugs a device (stub)
func (d *ContainerdDriverStub) HotPlugDevice(ctx context.Context, vmID string, device *DeviceConfig) error {
	return errors.New("containerd driver not implemented")
}

// HotUnplugDevice hot-unplugs a device (stub)
func (d *ContainerdDriverStub) HotUnplugDevice(ctx context.Context, vmID string, deviceID string) error {
	return errors.New("containerd driver not implemented")
}

// ConfigureCPUPinning configures CPU pinning (stub)
func (d *ContainerdDriverStub) ConfigureCPUPinning(ctx context.Context, vmID string, pinning *CPUPinningConfig) error {
	return errors.New("containerd driver not implemented")
}

// ConfigureNUMA configures NUMA topology (stub)
func (d *ContainerdDriverStub) ConfigureNUMA(ctx context.Context, vmID string, topology *NUMATopology) error {
	return errors.New("containerd driver not implemented")
}
