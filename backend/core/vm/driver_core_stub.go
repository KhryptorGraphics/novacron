//go:build !experimental

package vm

import (
	"context"
)

// CoreStubDriver is a no-op VM driver for core mode and tests
// It implements VMDriver with minimal behavior so CreateVM and actions succeed

type CoreStubDriver struct{}

func NewCoreStubDriver(config map[string]interface{}) (VMDriver, error) { return &CoreStubDriver{}, nil }

func (d *CoreStubDriver) Create(ctx context.Context, cfg VMConfig) (string, error) { return cfg.ID, nil }
func (d *CoreStubDriver) Start(ctx context.Context, vmID string) error { return nil }
func (d *CoreStubDriver) Stop(ctx context.Context, vmID string) error { return nil }
func (d *CoreStubDriver) Delete(ctx context.Context, vmID string) error { return nil }
func (d *CoreStubDriver) GetStatus(ctx context.Context, vmID string) (State, error) { return StateUnknown, nil }
func (d *CoreStubDriver) GetInfo(ctx context.Context, vmID string) (*VMInfo, error) { return &VMInfo{ID: vmID}, nil }
func (d *CoreStubDriver) GetMetrics(ctx context.Context, vmID string) (*VMInfo, error) { return &VMInfo{ID: vmID}, nil }
func (d *CoreStubDriver) ListVMs(ctx context.Context) ([]VMInfo, error) { return nil, nil }
func (d *CoreStubDriver) SupportsPause() bool { return true }
func (d *CoreStubDriver) SupportsResume() bool { return true }
func (d *CoreStubDriver) SupportsSnapshot() bool { return false }
func (d *CoreStubDriver) SupportsMigrate() bool { return false }
func (d *CoreStubDriver) Pause(ctx context.Context, vmID string) error { return nil }
func (d *CoreStubDriver) Resume(ctx context.Context, vmID string) error { return nil }
func (d *CoreStubDriver) Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error) { return "", nil }
func (d *CoreStubDriver) Migrate(ctx context.Context, vmID, target string, params map[string]string) error { return nil }
func (d *CoreStubDriver) SupportsLiveMigration() bool { return false }
func (d *CoreStubDriver) SupportsHotPlug() bool { return false }
func (d *CoreStubDriver) SupportsGPUPassthrough() bool { return false }
func (d *CoreStubDriver) SupportsSRIOV() bool { return false }
func (d *CoreStubDriver) SupportsNUMA() bool { return false }
func (d *CoreStubDriver) GetCapabilities(ctx context.Context) (*HypervisorCapabilities, error) { return &HypervisorCapabilities{}, nil }
func (d *CoreStubDriver) GetHypervisorInfo(ctx context.Context) (*HypervisorInfo, error) { return &HypervisorInfo{}, nil }
func (d *CoreStubDriver) HotPlugDevice(ctx context.Context, vmID string, device *DeviceConfig) error { return nil }
func (d *CoreStubDriver) HotUnplugDevice(ctx context.Context, vmID string, deviceID string) error { return nil }
func (d *CoreStubDriver) ConfigureCPUPinning(ctx context.Context, vmID string, pinning *CPUPinningConfig) error { return nil }
func (d *CoreStubDriver) ConfigureNUMA(ctx context.Context, vmID string, topology *NUMATopology) error { return nil }

